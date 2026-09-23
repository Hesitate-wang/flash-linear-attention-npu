# ChunkFwdHOFused 设计

本文定义当前实现的 Host tiling、kernel、workspace 和同步约束。公开接口与 TilingData 字段顺序以 `api.md` 和 `chunk_fwd_h_o_fused_struct.h` 为准。

## 1. 架构

定义任务数 `T = shapeBatch * vNumHead`、物理 MIX core 数 `P`。当前路径要求 `T <= floor(P / 2)`，活动 MIX core 数恒为 `A = 2T`。A2 (`DAV_2201`) 与 A5 (`DAV_3510`) 使用相同的 core 分区：前 `T` 个 core 是 H producer，从 `consumerCoreBase=T` 开始的 `T` 个 core 是一一对应的 O consumer。每个逻辑 MIX core 只处理一个 `(batch, value-head)` 任务；同一任务的不同 chunk 在两个 ping-pong slot 间流水。不满足核数前提时当前直接返回失败。

| SoC | Tiling key | blockDim | H/O 调度 | 同步 |
| --- | --- | --- | --- | --- |
| Atlas A2 | V128/V256 | `2 * T` | 一任务一 producer H + 一 consumer O | 每 chunk `IBSet/IBWait` |
| Ascend 950 A5 | V128/V256 | `2 * T` | 一任务一 producer H + 一 consumer O | 每 chunk `IBSet/IBWait` |

入口先由 `chunk_fwd_h_o_fused_arch.h` 根据编译目标定义且仅定义一个架构宏：A2 使用 `CHUNK_FWD_HO_ARCH_A2`，A5 使用 `CHUNK_FWD_HO_ARCH35`。统一源文件随后只包含对应的架构入口；架构入口和 H/O kernel 头对错误宏组合执行预处理报错。进入所选实现后，再通过 key 1/2 选择 V128/V256 的已注册 kernel object；key 不表示 exp/exp2。`useExp2` 只在已选定的架构实现内部选择指数模式，A5 在 `useExp2=false` 时仍使用 `arch35` FwdO；dtype、layout 和 head 维度同样在 object 内继续选择模板路径。

A5 专用 kernel、scheduler、epilogue、exp2 O tiling 投影、tile 常量和 UB 布局统一位于 `op_kernel/arch35/`。根目录仅保留公共 ABI/投影、统一 dispatcher 和 A2 实现，避免公共头通过内部 `__CCE_AICORE__` 分支混合两套架构定义。

## 2. 执行流程与同步

H 阶段先计算 `h` 与 `v_new`，O 阶段消费它们：

```text
v_new = u - w @ h_before
h_after = h_before * gate_end + k^T @ (v_new * gate_delta)
o = scale * (q @ h_before + masked(q @ k^T) @ v_new)
```

`GetMixedCoreIdx() < producerCoreNum` 时只运行 H，否则只运行 O。A2 与 A5 都不得采用“全核 H、全局同步、全核 O”的顺序执行模型。下文的核内流水细节描述当前 A5 自然指数实现，即 `arch35/gemm/kernel/gdn_fwd_h_kernel.hpp` 与 `gdn_fwd_o_kernel.hpp`；A2 使用相同的 H/O 配对和 IB handoff 协议，但其核内实现以 A2 对应 kernel 为准。

### 2.1 同步域

当前实现存在三个彼此独立的同步域：

| 同步域 | 原语 | 作用范围 | 保护对象 |
| --- | --- | --- | --- |
| AIV 流水线内部 | `SetFlag/WaitFlag`、`PipeBarrier` | 单个 AIV 内的 MTE2/V/MTE3 队列 | UB ping-pong 槽和同一 AIV 内的数据可见性 |
| 同一 MIX core 的 AIC/AIV 协作 | `CrossCoreSetFlag/CrossCoreWaitFlag` | 一个 MIX core 内的 AIC 与两个 AIV | Cube/Vector 中间结果和 ping-pong workspace 复用 |
| H producer 到 O consumer | `IBSet/IBWait` | 两个不同 MIX core 的对应 AIV | GM 中的 `H`、`V_new` handoff 就绪状态 |

前两类 flag 是每个 MIX core 自己的局部协议，不会在 H core 与 O core 之间传递。IB event 才是 H/O 物理 core 之间的同步；不能用核内 `cube*Done/vec*Done` 推断另一个 MIX core 的状态。

### 2.2 H 核内同步

H 的两个 stream 分别使用 flag `0/1`、`2/3`、`4/5`、`6/7`：

| flag | 发布方 | 等待方 | 含义 |
| --- | --- | --- | --- |
| `cube1Done[s]` | AIC Cube1 | 两个 AIV Vec1 | `W @ H` 的 workspace 已可读；短尾块由 AIC 直接从 MTE2 发布 |
| `vec1Done[s]` | 两个 AIV Vec1 | AIC Cube2 | `VUpdate`（以及 gated K workspace）已可读 |
| `cube2Done[s]` | AIC Cube2 | 两个 AIV Vec2 | `K^T @ VUpdate` 的 H workspace 已可读；无需 Cube2 的分支也必须发布匹配代次 |
| `vec2Done[s]` | 两个 AIV Vec2 | AIC 下一次 Cube1 | 当前 stream 的 H/V/H-workspace 已消费完，可以复用 |

每个 stream 的稳态闭环为：

```text
初始 H0 写回
    -> IBSet HReady[lane]

AIV 预置 vec2Done[s]
    -> AIC wait vec2Done[s]
    -> Cube1: W @ H
    -> AIC set cube1Done[s]
    -> AIV wait cube1Done[s], 生成并写回 V_new/VUpdate
    -> AIV set vec1Done[s]
    -> IBSet VReady[lane]
    -> AIC wait vec1Done[s]
    -> Cube2: K^T @ VUpdate
    -> AIC set cube2Done[s]
    -> AIV wait cube2Done[s], 更新并写回 H_after
    -> 非 final chunk: IBSet HReady[lane]
    -> AIV set vec2Done[s]
```

首轮若没有 AIV 对 `vec2Done[0..1]` 的预置，AIC 会在第一次 Cube1 前永久等待。结束时 AIC 再等待两个 `vec2Done`，保证最后一代 Vec2 已消费 workspace。短尾块虽然可能绕过 MMAD，仍会走同代次的 `cube1Done/cube2Done` 发布，避免另一侧等待一个不存在的 Cube 任务。

`useDirectFp32Ub` 路径另有 `0x4` 域的 free/ready flag，负责 AIC 与 AIV 直接共享 UB 槽；它不替代上述 `0x2` 完成链。AIV 内部 epilogue 还使用 MTE2/V/MTE3 hard event 管理 UB ping-pong 槽，入口预置 free event，退出逐一 `WaitFlag` 回收，保证没有悬空的本地事件。

### 2.3 O 核内同步

O 的两个 stream 使用相同的编号空间，但语义不同：

| flag | 方向 | 含义 |
| --- | --- | --- |
| `cube1Done[s]` | AIC→AIV，随后 AIV→AIC | 第一代表示 `Q @ K^T` workspace 可读；同一 flag 的下一代是两个 AIV 在收到 `HReady` 后给 AIC 的 ACK |
| `vec1Done[s]` | AIV→AIC | masked QK 已写入 workspace，且 `VReady` 已消费，Cube3 的两个输入均可用 |
| `cube3Done[s]` | AIC→AIV | `masked(QK) @ V_new` workspace 可读 |
| `vec2Done[s]` | AIV→AIC | 输出 epilogue 已完成对 H/V workspace 的最终读取，AIC 可以复用该 ping-pong 槽 |

当前代码的单个 stream 顺序为：

```text
AIV 预置 vec2Done[s]

AIC: Cube1(Q @ K^T) -> set cube1Done[s]
AIV: wait cube1Done[s]
     -> IBWait HReady[lane]
     -> set cube1Done[s]                 # 反向 ACK
     -> 计算 masked(QK)
     -> IBWait VReady[lane]
     -> set vec1Done[s]

AIC: wait cube1Done[s]                   # 等待 HReady ACK
     -> Cube2(Q @ H_before)
     -> wait vec2Done[s]                 # 复用 H/V workspace 前
     -> wait vec1Done[s]
     -> Cube3(masked(QK) @ V_new)
     -> set cube3Done[s]

AIV: wait cube3Done[s]
     -> 融合 Cube2/Cube3 结果并写 O
     -> set vec2Done[s]
```

`cube1Done[s]` 是双向、分代复用的握手：AIC 发出第 `2n` 代，两个 AIV 各自等待后发出第 `2n+1` 代 ACK，AIC 必须消费 ACK 后才允许同一 stream 再发下一代 Cube1 完成事件。两个 ping-pong stream 使当前 Cube1 与上一任务的 Cube2/Cube3 重叠，但不能改变同一 stream 的代次顺序。

`QK` 和 `masked(QK)` 在数学上都不依赖 H。当前代码仍按“wait Cube1 → wait HReady → ACK → masked(QK)”执行，因此 `HReady` 会阻塞 mask 计算；这里的 `HReady` 实际用于门控 AIC 的 `Q @ H_before`，不是 mask 的数据依赖。若后续把 `IBWait(HReady)` 下移以扩大并行度，必须同时保留 AIV→AIC 的 ACK，并证明 `cube1Done` 双向代次没有被重排。

### 2.4 H 到 O 的 IB handoff

H producer `p in [0,T)` 负责 task `p`，O consumer `T+p` 按相同的 task/chunk 顺序消费。每个 producer AIV 使用 2 个 event：

| event id | 事件 | H 发布时机 | O 等待后允许的操作 |
| --- | --- | --- | --- |
| `0` | `HReady` | chunk 0 的 H0 写回 GM 后，或 chunk `i-1` 的 Vec2 写回 `H_i` 后 | AIC 启动当前 chunk 的 `Q @ H_i` |
| `1` | `VReady` | 当前 chunk 的 Vec1 写回 `V_new_i` 后 | AIC 启动当前 chunk 的 `masked(QK) @ V_new_i` |

每个任务固定使用 `taskLane=0`。O 直接使用 `producerCoreIdx=taskIdx`，再用相同的 `subBlockIdx` 得到 `producerAivIdx=producerCoreIdx*subBlockNum+subBlockIdx`，因此两个 O AIV 分别等待对应 H AIV 发布的 H/V 切片，不会互相代替。

IB 槽按 `[eventId][logicalAivIdx][8 x int32]` 编址，其中 `logicalAivNum=activeCoreNum*subBlockNum`。kernel 分流前，所有活动 AIV 分工把全部 event 槽显式写零。本地零值由 `Duplicate` 在 `PIPE_V` 生成，通过配对的 `SetFlag/WaitFlag<HardEvent::V_MTE3>` 交给后续 UB→GM `DataCopy`；全部异步 `DataCopy` 提交后，再用 `SetFlag/WaitFlag<HardEvent::MTE3_MTE2>` 等待 GM 写回完成，确保后续 MTE2 上的 `IBSet/IBWait` 不会读取未落盘的共享槽，随后 AIC/AIV 执行一次 `SyncAll<false>()`，确认初始化完成后才进入 H 或 O。该初始化清零的是 GM event table；传给 `IBSet/IBWait` 的 32-byte UB tensor 是 API 的本地工作区，不是跨 core 共享状态。

`IBSet/IBWait` 内部在数据搬入和搬出前后执行 `PipeBarrier<Pipe_all>`，因此 H 直接在对应 GM 写回操作后调用 `IBSet`，不再额外插入 `PipeBarrier<PIPE_MTE3>()`。O 的 `IBWait` 消费 ready 并归还同一个二值槽；同一 task 的后续 chunk 复用该槽，所以 H 若过早追上 O，会阻塞在下一次 `IBSet`，而不是覆盖一个未消费的 ready。handoff 数据本身按任务/chunk 独立编址，event 槽只表达就绪和背压。

### 2.5 调度匹配与无死锁条件

H 的每个 producer 只处理一个 task，并在两个内部 stream/slot 间处理相邻 chunk。O consumer 使用相同的 task 和 chunk 顺序，因此每个 producer 的 IB 发布顺序与对应 consumer 的等待顺序严格一致。

在下列不变量成立时，当前静态同步图不存在必然环路：

1. `activeCoreNum=2T`，producer/consumer 一一配对，H/O scheduler 使用相同的 `T` 和 chunk 数。
2. 所有 GM IB event 在第一次 `IBSet/IBWait` 前已清零，且 `pipelineSyncWorkspaceOffset`、event 数和实际分配大小一致。
3. H 的每个有效 `HReady/VReady` 恰好有一个对应 O `IBWait`；无效 lane 和 final chunk 不额外等待不存在的事件。
4. 两个 AIV 都执行每个 `0x2` 聚合握手；任何一个 AIV 提前退出都会使同 MIX core 的 AIC 永久等待。
5. 每条 bypass/tail 分支也发布与常规路径相同代次的完成 flag，首轮 free flag 与末轮 drain wait 成对存在。
6. `IBSet/IBWait` 的 32-byte 本地 UB 工作区与同一时刻的 Vector/Fixpipe UB 区域不重叠。

FwdO 的每个 consumer 固定负责一个 head task，并在 stage0/stage1 之间交替处理
相邻 chunk，使 `Vec1(chunk i)` 与 `Vec2(chunk i-1)` 重叠。两个 stage 的初始
`vec2Done` token 都必须预置，否则首轮落入任一 stage 时都可能等待未发布的 free token。
因此，若注释 IB 后超时消失，优先检查的不是 GM 数值内容，而是上述任一不变量是否在运行时被破坏，尤其是 task/chunk 映射、某个 AIV 分支跳过发布、同步 workspace 越界或初始化 barrier 参与者不一致。`PRINTF` 会改变发射和流水时序，只能暴露或掩盖时序问题，不能作为同步正确性的组成部分。

### 2.6 IB 通信专用 UB 区域

A5 H/O kernel 统一使用 `chunk_fwd_h_o_fused_ub_layout.h` 声明 UB 布局。O 的数据区和 IB 通信区划分为：

```text
[0 KiB,   71 KiB)  base region
[71 KiB, 199 KiB)  Cube Fixpipe work slots
[199 KiB,248 KiB)  Vec scratch
[248 KiB,256 KiB)  IB communication reserved region
```

`IBSet/IBWait` 的 32-byte local tensor 固定放在通信区起始地址 `248 KiB`。该区域不参与 Cube、Fixpipe 或 Vector 数据复用，因此 L0C 到 UB 的写入不会覆盖 IB 指令的本地操作数。H 与 O 使用同一常量，避免两侧各自维护裸偏移。

共享布局对通信区的 32-byte 对齐、32-byte IB operand 容量和 256 KiB UB 总容量执行编译期检查；两个 O epilogue 也以通信区起点作为数据区硬上界。后续扩展任何 O UB tensor 时，若侵入 `[248 KiB, 256 KiB)` 将直接编译失败。A2 使用独立的 192 KiB UB 布局，继续保留其现有同步偏移，不复用该 A5 常量。

当前 arch35 exp2 专用 O 的底层 cube/vector 类尚未暴露 handoff 的 IBWait 接口，因此实现阶段暂时在 exp2 分支保留 H/O 边界同步作为兼容保护；该分支在补齐专用 O 的 IBWait 后必须删除该同步，恢复与自然指数路径相同的 chunk pipeline。

## 3. Host tiling 约束

Host 设置 `producerCoreNum=consumerCoreNum=T`、`activeCoreNum=2*T`、`consumerCoreBase=T` 和 `scheduleMode=1`。进入该路径前必须满足 `T <= floor(physicalCoreNum/2)`；不满足时当前返回失败。当前实现仅接受定长输入，且至少需要两个物理 AIC core。提供 `cuSeqlens` 或 `chunkIndices` 时返回失败。

## 4. TilingData ABI

以下字段顺序不可改变：

```cpp
BEGIN_TILING_DATA_DEF(ChunkFwdHOFusedTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, seqlen);
TILING_DATA_FIELD_DEF(int64_t, kNumHead);
TILING_DATA_FIELD_DEF(int64_t, vNumHead);
TILING_DATA_FIELD_DEF(int64_t, kHeadDim);
TILING_DATA_FIELD_DEF(int64_t, vHeadDim);
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
TILING_DATA_FIELD_DEF(bool, useInitialState);
TILING_DATA_FIELD_DEF(bool, storeFinalState);
TILING_DATA_FIELD_DEF(int64_t, dataType);
TILING_DATA_FIELD_DEF(int64_t, gDataType);
TILING_DATA_FIELD_DEF(int64_t, stateDataType);
TILING_DATA_FIELD_DEF(int64_t, isVariedLen);
TILING_DATA_FIELD_DEF(int64_t, shapeBatch);
TILING_DATA_FIELD_DEF(int64_t, tokenBatch);
TILING_DATA_FIELD_DEF(bool, useG);
TILING_DATA_FIELD_DEF(bool, useGk);
TILING_DATA_FIELD_DEF(int64_t, vWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, vUpdateWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, kDecayWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, hWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, numSeqWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, numChunksWorkspaceOffset);
TILING_DATA_FIELD_DEF(bool, useExp2);
TILING_DATA_FIELD_DEF(int64_t, outputLayout);
TILING_DATA_FIELD_DEF(float, scale);
TILING_DATA_FIELD_DEF(int64_t, chunkNum);
TILING_DATA_FIELD_DEF(int64_t, numChunksPerBatch);
TILING_DATA_FIELD_DEF(int64_t, hvPerHk);
TILING_DATA_FIELD_DEF(int64_t, taskGroupSize);
TILING_DATA_FIELD_DEF(int64_t, producerCoreNum);
TILING_DATA_FIELD_DEF(int64_t, consumerCoreBase);
TILING_DATA_FIELD_DEF(int64_t, activeCoreNum);
TILING_DATA_FIELD_DEF(int64_t, handoffHWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, handoffVWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, pipelineSyncWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, pipelineEventCount);
TILING_DATA_FIELD_DEF(int64_t, oVWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, oHWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, oAttnWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, oAfterMaskWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, oMaskWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, oAPrimeWorkspaceOffset);
END_TILING_DATA_DEF;
```

H 只解释从 `batch` 到 `numChunksWorkspaceOffset` 的前缀；O 使用显式 stage projection，不得把完整 TilingData 强转为另一种布局。`offsetof(useExp2)` 必须继续与 H stage struct 大小一致。

## 5. Workspace

所有 offset 相对 `AscendC::GetUserWorkspace(workspace)`，每段按 512 字节对齐。`E` 为输入元素字节数，`F=4`，`S=2`：

| 区域 | 大小 |
| --- | --- |
| handoff H | `T * NC * K * V * E` |
| handoff V | `T * tokens * V * E` |
| pipeline sync | `A * 2 * 2 * 8 * 4`（每个 task 一个 HReady 和 VReady） |
| H v/v-update | 各 `R * C * V * F * S` |
| H k-decay（有 GK） | `R * C * K * F * S` |
| H state | `R * K * V * F * S` |
| O v/h | 各 `(A-R) * C * V * F * S` |
| O attention/after-mask | 各 `(A-R) * C * C * F * S` |
| O mask | `C * C` |
| A5 exp2 A-prime | `(A-R) * CHUNK_FWD_O_APRIME_WORKSPACE_BYTES` |

O 和 A-prime 临时区按 consumer-local index 编址；H 临时区按 producer-local index 编址。`pipelineSyncWorkspaceOffset` 必须指向实际分配的同步区，不能置零。

## 6. 接口与验证

公开 ACLNN/L0 输入、输出、属性顺序和 kernel ABI 顺序保持现有实现不变。修改后至少静态验证：TilingData 字段顺序/类型/大小一致；A2/A5 key 1/2 均有对应 entry；`activeCoreNum=2*producerCoreNum` 且 producer/consumer 数量相等；handoff workspace 使用任务数 `T`，core-local workspace 使用对应 producer/consumer 数；A5 自然指数路径无 H/O 边界 `SyncAll` 并逐 chunk 执行 IBWait。
