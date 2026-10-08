# ChunkFwdHOFused 设计

本文定义当前实现的 Host tiling、kernel、workspace 和同步约束。公开接口与 TilingData 字段顺序以 `api.md` 和 `chunk_fwd_h_o_fused_struct.h` 为准。

## 1. 架构

定义任务数 `T = shapeBatch * vNumHead`、物理 MIX core 数 `P`。当前路径要求 `T <= floor(P / 2)`，活动 MIX core 数恒为 `A = 2T`。A2 (`DAV_2201`) 与 A5 (`DAV_3510`) 使用相同的 core 分区：前 `T` 个 core 是 H producer，从 `consumerCoreBase=T` 开始的 `T` 个 core 是一一对应的 O consumer。每个逻辑 MIX core 只处理一个 `(batch, value-head)` 任务；同一任务的不同 chunk 在两个 ping-pong slot 间流水。不满足核数前提时当前直接返回失败。

| SoC | Tiling key | blockDim | H/O 调度 | 同步 |
| --- | --- | --- | --- | --- |
| Atlas A2 | V128/V256 | `2 * T` | 一任务一 producer H + 一 consumer O | mode `0x0` `H_READY/V_READY` Set→Wait |
| Ascend 950 A5 | V128/V256 | `2 * T` | 一任务一 producer H + 一 consumer O | mode `0x0` `H_READY/V_READY` Set→Wait |

入口先由 `chunk_fwd_h_o_fused_arch.h` 根据编译目标定义且仅定义一个架构宏：A2 使用 `CHUNK_FWD_HO_ARCH_A2`，A5 使用 `CHUNK_FWD_HO_ARCH35`。统一源文件随后只包含对应的架构入口；架构入口和 H/O kernel 头对错误宏组合执行预处理报错。进入所选实现后，再通过 key 1/2 选择 V128/V256 的已注册 kernel object；key 不表示 exp/exp2。`useExp2` 只在已选定的架构实现内部选择指数模式，A5 在 `useExp2=false` 时仍使用 `arch35` FwdO；dtype、layout 和 head 维度同样在 object 内继续选择模板路径。

A5 专用 kernel、scheduler、epilogue、exp2 O tiling 投影、tile 常量和 UB 布局统一位于 `op_kernel/arch35/`。根目录仅保留公共 ABI/投影、统一 dispatcher 和 A2 实现，避免公共头通过内部 `__CCE_AICORE__` 分支混合两套架构定义。

## 2. 执行流程与同步

H 阶段先计算 `h` 与 `v_new`，O 阶段消费它们：

```text
v_new = u - w @ h_before
h_after = h_before * gate_end + k^T @ (v_new * gate_delta)
o = scale * (q @ h_before + masked(q @ k^T) @ v_new)
```

`GetMixedCoreIdx() < producerCoreNum` 时只运行 H，否则只运行 O。A2 与 A5 都不得采用“全核 H、全局同步、全核 O”的顺序执行模型。下文的核内流水细节描述当前 A5 自然指数实现，即 `arch35/gemm/kernel/gdn_fwd_h_kernel.hpp` 与 `gdn_fwd_o_kernel.hpp`；A2 使用相同的 H/O 配对和 mode-0 handoff 协议，但其核内实现以 A2 对应 kernel 为准。

### 2.1 同步域

当前实现存在三个彼此独立的同步域：

| 同步域 | 原语 | 作用范围 | 保护对象 |
| --- | --- | --- | --- |
| AIV 流水线内部 | `SetFlag/WaitFlag`、`PipeBarrier` | 单个 AIV 内的 MTE2/V/MTE3 队列 | UB ping-pong 槽和同一 AIV 内的数据可见性 |
| 同一 MIX core 的 AIC/AIV 协作 | `CrossCoreSetFlag/CrossCoreWaitFlag` | 一个 MIX core 内的 AIC 与两个 AIV | Cube/Vector 中间结果和 ping-pong workspace 复用 |
| H producer 到 O consumer | mode `0x0` `CrossCoreSetFlag/CrossCoreWaitFlag` | 所有活动 H/O AIV | 当前 chunk 的 `H`、`V_new` handoff 就绪状态 |

前两类 flag 是每个 MIX core 自己的局部协议，不会在 H core 与 O core 之间传递。mode-0 H/O flag 才是 H/O 物理 core 之间的同步；不能用核内 `cube*Done/vec*Done` 推断另一个 MIX core 的状态。

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
    -> mode-0 Set/Wait(H_READY)

AIV 预置 vec2Done[s]
    -> AIC wait vec2Done[s]
    -> Cube1: W @ H
    -> AIC set cube1Done[s]
    -> AIV wait cube1Done[s], 生成并写回 V_new/VUpdate
    -> AIV set vec1Done[s]
    -> mode-0 Set/Wait(V_READY)
    -> AIC wait vec1Done[s]
    -> Cube2: K^T @ VUpdate
    -> AIC set cube2Done[s]
    -> AIV wait cube2Done[s], 更新并写回 H_after
    -> 非 final chunk: mode-0 Set/Wait(H_READY)
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
| `cube2Done[s]` | AIC→AIV，仅 V128 | `Q @ H_before` 的 FP32 workspace 已可读；允许先计算 `Cube2 * exp(g)` |
| `cube3Done[s]` | AIC→AIV | `masked(QK) @ V_new` workspace 可读 |
| `vec2Done[s]` | AIV→AIC | 输出 epilogue 已完成对 H/V workspace 的最终读取，AIC 可以复用该 ping-pong 槽 |

当前代码的单个 stream 顺序为：

```text
AIV 预置 vec2Done[s]

AIC: Cube1(Q @ K^T) -> set cube1Done[s]
AIV: wait cube1Done[s]
     -> 计算 gate 差、指数和 causal mask
     -> mode-0 Set/Wait(H_READY)
     -> set cube1Done[s]                 # 反向 ACK
     -> 读取 QK，计算并写回 masked(QK)
     -> mode-0 Set/Wait(V_READY)
     -> set vec1Done[s]

AIC: wait cube1Done[s]                   # 等待 HReady ACK
     -> Cube2(Q @ H_before)
     -> wait vec2Done[s]                 # 复用 H/V workspace 前
     -> wait vec1Done[s]
     -> set cube2Done[s]                 # V128, Cube2 FixPipe 完成
     -> Cube3(masked(QK) @ V_new)
     -> set cube3Done[s]

AIV (V128): wait cube2Done[s]
     -> 整片 FP32 Cube2 * exp(g)，驻留 H UB 槽
     -> wait cube3Done[s]
     -> 逐 tile 计算 (gated Cube2 + Cube3) * scale 并写 O
     -> set vec2Done[s]

AIV (V256): wait cube3Done[s]
     -> 沿用原融合计算与写回
     -> set vec2Done[s]
```

`cube1Done[s]` 是双向、分代复用的握手：AIC 发出第 `2n` 代，两个 AIV 各自等待后发出第 `2n+1` 代 ACK，AIC 必须消费 ACK 后才允许同一 stream 再发下一代 Cube1 完成事件。两个 ping-pong stream 使当前 Cube1 与上一任务的 Cube2/Cube3 重叠，但不能改变同一 stream 的代次顺序。

`QK` 和 `masked(QK)` 在数学上都不依赖 H。AIV 在等待 Cube1 结果后先发射
gate 搬运、差值、指数及 causal mask 的 Vector 工作，再在 qkmask epilogue 内执行
`H_READY` Set/Wait，紧接着向 AIC ACK，之后才读取 QK、乘 mask、转换并写回。
HReady 仍门控 Cube2 的 `Q @ H_before`；ACK 在 Vec1 的输出 MTE3 之前发出，
使 Cube2 可与其余 Vec1 操作重叠。`chunk=128` 的双段 epilogue 仅在第一段
执行一次 HReady/ACK。非 pipeline 路径跳过 H/O mode-0 rendezvous，但保持 ACK 代次。
mode-0 rendezvous 要求所有活动 AIV 按同一 chunk 代次参与；H 提前 ready 时仍会由
全体参与者的 Wait 共同放行 Cube2，需按目标 shape 对比精度和 profiling 总耗时。

V128 的 Cube2 H workspace 每个 AIV 最多持有 `64 x 128 x 4 = 32 KiB`，
完整载入已有的 ping/pong H UB 槽后原位乘 `exp(g)`，无需新 GM workspace。
Cube3 仍使用独立的 attn workspace；`cube2Done` 在 Cube2 的 FixPipe 完成后发布，
`cube3Done` 在 Cube3 完成后发布。第二段对同一个 FP32 H tile 先 Add 再 Muls，
保持原融合公式和 cast 顺序。两个 stage 的 H UB 槽不重叠；`vec2Done` 在最后一块
attn MTE2 结束后通知 AIC 复用 GM workspace。V256 的每 AIV 半片可达 64 KiB，
超过单个 H UB 槽，保留原单阶段输出流程。

### 2.4 H 到 O 的 CrossCore handoff

H producer `p in [0,T)` 负责 task `p`，O consumer `T+p` 按相同的 task/chunk 顺序消费。自然指数路径使用 mode `0x0` 的 H_READY/V_READY flag，与每个 MIX core 内部使用的 mode `0x2` flag 分离。A5 V128 preload 路径的 V_READY 由 H/O 两侧对应 AIV 执行 mode-0 `Set/Wait`；H_READY 保留 AIC rendezvous。H_READY 按七个 MIX task pair 一组映射到 flag `0/2/4`。V_READY 使用本侧相对 AIV block index，按六个 AIV 一组连续映射到 flag `0..6`；O AIV 先减去 `producerCoreNum * subBlockNum`，再与 H AIV 使用相同的相对编号。

| flag | 事件 | 生产方 | 消费方 |
| --- | --- | --- | --- |
| `0/2/4` | `HReady` | H 初始状态或 Vec2 写回 `H_i`；O Vec1 完成独立 gate/mask 计算后 | 每七个 MIX task pair 一组；双方等待完成后，O AIV 通过 `cube1Done` 通知 O AIC 读取 H |
| `0..6` | `VReady` | H Vec1 完成 `V_new_i` 的 GM 写回后；O Vec1 完成 masked QK 后 | 每六个相对 AIV block index 一组；随后 O AIV 通过 `vec1Done` 通知 O AIC 执行 Cube3 |

每个任务固定使用 `taskLane=0`。V_READY 的 H 侧相对 AIV 编号为原始 AIV `blockIdx`；O 侧相对编号为 `blockIdx - producerCoreNum * subBlockNum`。因此同一 task 的两个 H/O AIV 分别得到相同编号，并映射到相同的 V_READY flag group。

`pipelineSyncWorkspaceOffset` 和原有 IB workspace allocation 保持 ABI 兼容，但自然指数 H/O 不再初始化或访问其中的 event 槽；入口不再执行同步区初始化或额外启动屏障。mode `0x0` flag 不携带 task id，正确性依赖所有活动 H/O AIV 按相同 chunk 顺序参与集合握手。

H 的 GM 写回完成后再发布 ready。A5 V128 中，O AIV 等到 V_READY 后才发布本核 mode `0x2` `vec1Done`，保证 Cube3 同时看到 masked QK 和 V_new；handoff 数据仍按任务/chunk 独立编址。

### 2.5 调度匹配与无死锁条件

H 的每个 producer 只处理一个 task，并在两个内部 stream/slot 间处理相邻 chunk。O consumer 使用相同的 task 和 chunk 顺序，因此每个 producer 的 IB 发布顺序与对应 consumer 的等待顺序严格一致。

在下列不变量成立时，当前静态同步图不存在必然环路：

1. `activeCoreNum=2T`，producer/consumer 一一配对，H/O scheduler 使用相同的 `T` 和 chunk 数。
2. 每个有效 H/O AIV 对每个 chunk 恰好执行一次对应 `Set` 和一次 `Wait`；所有活动 task 的参与顺序一致。
3. H 的 final chunk 不发布下一代 `H_READY`；单 chunk 仍执行初始 `H_READY` 与当前 `V_READY`，不额外等待不存在的事件。
4. 两个 AIV 都执行每个 `0x2` 聚合握手；任何一个 AIV 提前退出都会使同 MIX core 的 AIC 永久等待。
5. 每条 bypass/tail 分支也发布与常规路径相同代次的完成 flag，首轮 free flag 与末轮 drain wait 成对存在。
6. mode `0x0` 的 flag 12/13 不与任何 AIC/AIV 本地 flag 复用；本核 Cube/Vector 代次仍保持原有 `0x2` 资源闭环。

FwdO 的每个 consumer 固定负责一个 head task，并在 stage0/stage1 之间交替处理
相邻 chunk，使 `Vec1(chunk i)` 与 `Vec2(chunk i-1)` 重叠。两个 stage 的初始
`vec2Done` token 都必须预置，否则首轮落入任一 stage 时都可能等待未发布的 free token。
因此，若移除旧 IB 访问后超时消失，优先检查的不是 GM 数值内容，而是上述任一不变量是否在运行时被破坏，尤其是 task/chunk 映射、某个 AIV 分支跳过发布、同步 workspace 越界或初始化 barrier 参与者不一致。`PRINTF` 会改变发射和流水时序，只能暴露或掩盖时序问题，不能作为同步正确性的组成部分。

### 2.6 IB 通信专用 UB 区域

A5 H/O kernel 统一使用 `chunk_fwd_h_o_fused_ub_layout.h` 声明 UB 布局。O 的数据区和 IB 通信区划分为：

```text
[0 KiB,   71 KiB)  base region
[71 KiB, 199 KiB)  Cube Fixpipe work slots
[199 KiB,248 KiB)  Vec scratch
[248 KiB,256 KiB)  IB communication reserved region
```

旧 IB 协议的 32-byte local tensor 保留在通信区起始地址 `248 KiB`。该区域不参与 Cube、Fixpipe 或 Vector 数据复用，因此 L0C 到 UB 的写入不会覆盖 IB 指令的本地操作数。H 与 O 使用同一常量，避免两侧各自维护裸偏移。

共享布局对通信区的 32-byte 对齐、32-byte IB operand 容量和 256 KiB UB 总容量执行编译期检查；两个 O epilogue 也以通信区起点作为数据区硬上界。后续扩展任何 O UB tensor 时，若侵入 `[248 KiB, 256 KiB)` 将直接编译失败。A2 使用独立的 192 KiB UB 布局，继续保留其现有同步偏移，不复用该 A5 常量。

当前 arch35 exp2 专用 O 仍保留原有阶段边界同步；mode-0 H/O 握手只用于自然指数路径。

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

公开 ACLNN/L0 输入、输出、属性顺序和 kernel ABI 顺序保持现有实现不变。修改后至少静态验证：TilingData 字段顺序/类型/大小一致；A2/A5 key 1/2 均有对应 entry；`activeCoreNum=2*producerCoreNum` 且 producer/consumer 数量相等；handoff workspace 使用任务数 `T`，core-local workspace 使用对应 producer/consumer 数；A5 自然指数路径无 H/O 边界 `SyncAll`，并逐 chunk 执行 mode-0 H_READY/V_READY Set→Wait。

## 7. A5 H 主路径编译期分离

A5 H producer 按数据通路实例化三种 `GDNFwdHPath`：

| specialization | 选择条件 | Cube/Vector 中间结果 |
| --- | --- | --- |
| `DirectUb` | 定长、无尾 chunk、`16 <= chunkSize <= 64`、`K=V=128` 且任务数覆盖 producer core | Cube Fixpipe 直接写 AIV UB，Vector 通过 direct ready/free flag 消费 |
| `BoundedGm` | 变长或 `seqlen % chunkSize != 0` | bounded MMAD 写 GM workspace，Vector 由 MTE2 搬入 UB |
| `StandardGm` | 其余场景 | 普通 MMAD 写 GM workspace；小于 16 token 的任务使用 Vector fallback |

入口只执行一次运行时路径选择；选定后 Cube1/Cube2 循环以及 Vec1/Vec2 epilogue
均通过 `if constexpr` 删除其他路径的 GM 搬运、direct flag 和收尾代码。首尾 chunk、
初始/最终状态等随任务变化的条件仍在 specialization 内运行时判断。该调整不改变
TilingData、workspace 布局、ready/free 代次和数学计算顺序。

`DirectUb` 的 Vec2 在与独立 FwdH preload 路径相同的 `V=128`、无 GK 条件下，每个
AIV 一次处理其负责的连续 64 行 state：一次 MTE2 搬入 `64 x 128` 旧 state，一次
完成 cast、衰减、与 Cube2 FP32 update 相加及回转，再按既有输出语义统一写回。
未请求 final state 时，每个 AIV 只以一次 MTE3 写回 `64 x 128` 新 state。该路径的
`calcUbTensor`、Cube2 update UB 和低精度输出 UB 分别占 32 KiB、32 KiB 和 16 KiB，
均在既有区域内；两个 AIV 的本地 UB 相互独立。带 GK 或非 DirectUb 的场景继续使用
16 行通用 tile；final-state 分支沿用 fused 已有的 FP32 驻留和写回语义。
