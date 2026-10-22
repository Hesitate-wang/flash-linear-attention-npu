# ChunkFwdHOFused 设计

本文定义当前实现的 Host tiling、kernel、workspace 和同步约束。公开接口与 TilingData 字段顺序以 `api.md` 和 `chunk_fwd_h_o_fused_struct.h` 为准。

## 1. 架构

定义任务数 `T = shapeBatch * vNumHead`、物理 MIX core 数 `P`。MODE1 要求 `T <= floor(P / 2)`，活动 MIX core 数为 `A = 2T`：前 `T` 个 core 是 H producer，后 `T` 个 core 是一一对应的 O consumer；同一任务的 chunk 在两个 slot 间流水。A5 自然指数 MODE2 覆盖 `floor(P / 2) < T <= P`，活动 MIX core 数为 `A = T`，每个 core 拥有一个 `(batch, value-head)` task，并在一个单槽上按 chunk 串行执行 H/O。其余核数范围继续返回失败。

| SoC | Tiling key | blockDim | H/O 调度 | 同步 |
| --- | --- | --- | --- | --- |
| Atlas A2 | V128/V256 | `2 * T` | 一任务一 producer H + 一 consumer O | mode `0x0` `H_READY/V_READY` Set→Wait |
| Ascend 950 A5 | V128/V256 | `2 * T` | 一任务一 producer H + 一 consumer O | mode `0x0` `H_READY/V_READY` Set→Wait |
| Ascend 950 A5 MODE2 | V128 | `T` | 一任务一 MIX core，chunk-serial H/O | 仅 mode `0x2` 核内 AIC/AIV 同步 |

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
| `hc1WhReady[s]` | AIC Cube1 | 两个 AIV Vec1 | `W @ H` 的 workspace 已可读；短尾块由 AIC 直接从 MTE2 发布 |
| `hv1VUpdateReadyForHc2[s]` | 两个 AIV Vec1 | AIC Cube2 | `VUpdate`（以及 gated K workspace）已可读 |
| `hc2KvUpdateReady[s]` | AIC Cube2 | 两个 AIV Vec2 | `K^T @ VUpdate` 的 H workspace 已可读；无需 Cube2 的分支也必须发布匹配代次 |
| `vec2Done[s]` | 两个 AIV Vec2 | AIC 下一次 Cube1 | 当前 stream 的 H/V/H-workspace 已消费完，可以复用 |

每个 stream 的稳态闭环为：

```text
初始 H0 写回
    -> mode-0 Set/Wait(H_READY)

AIV 预置 vec2Done[s]
    -> AIC wait vec2Done[s]
    -> Cube1: W @ H
    -> AIC set hc1WhReady[s]
    -> AIV wait hc1WhReady[s], 生成并写回 V_new/VUpdate
    -> AIV set hv1VUpdateReadyForHc2[s]
    -> mode-0 Set/Wait(V_READY)
    -> AIC wait hv1VUpdateReadyForHc2[s]
    -> Cube2: K^T @ VUpdate
    -> AIC set hc2KvUpdateReady[s]
    -> AIV wait hc2KvUpdateReady[s], 更新并写回 H_after
    -> 非 final chunk: mode-0 Set/Wait(H_READY)
    -> AIV set vec2Done[s]
```

首轮若没有 AIV 对 `vec2Done[0..1]` 的预置，AIC 会在第一次 Cube1 前永久等待。结束时 AIC 再等待两个 `vec2Done`，保证最后一代 Vec2 已消费 workspace。短尾块虽然可能绕过 MMAD，仍会走同代次的 `hc1WhReady/hc2KvUpdateReady` 发布，避免另一侧等待一个不存在的 Cube 任务。

`useDirectFp32Ub` 路径另有 `0x4` 域的 free/ready flag，负责 AIC 与 AIV 直接共享 UB 槽；它不替代上述 `0x2` 完成链。AIV 内部 epilogue 还使用 MTE2/V/MTE3 hard event 管理 UB ping-pong 槽，入口预置 free event，退出逐一 `WaitFlag` 回收，保证没有悬空的本地事件。

### 2.3 O 核内同步

O 的两个 stream 使用相同的编号空间，但语义不同：

| flag | 方向 | 含义 |
| --- | --- | --- |
| `hc1WhReady[s]` | AIC→AIV，随后 AIV→AIC | 第一代表示 `Q @ K^T` workspace 可读；同一 flag 的下一代是两个 AIV 在收到 `HReady` 后给 AIC 的 ACK |
| `hv1VUpdateReadyForHc2[s]` | AIV→AIC | masked QK 已写入 workspace，且 `VReady` 已消费，Cube3 的两个输入均可用 |
| `hc2KvUpdateReady[s]` | AIC→AIV，仅 V128 | `Q @ H_before` 的 FP32 workspace 已可读；允许先计算 `Cube2 * exp(g)` |
| `cube3Done[s]` | AIC→AIV | `masked(QK) @ V_new` workspace 可读 |
| `vec2Done[s]` | AIV→AIC | 输出 epilogue 已完成对 H/V workspace 的最终读取，AIC 可以复用该 ping-pong 槽 |

当前代码的单个 stream 顺序为：

```text
AIV 预置 vec2Done[s]

AIC: Cube1(Q @ K^T) -> set hc1WhReady[s]
AIV: wait hc1WhReady[s]
     -> 计算 gate 差、指数和 causal mask
     -> mode-0 Set/Wait(H_READY)
     -> set hc1WhReady[s]                 # 反向 ACK
     -> 读取 QK，计算并写回 masked(QK)
     -> mode-0 Set/Wait(V_READY)
     -> set hv1VUpdateReadyForHc2[s]

AIC: wait hc1WhReady[s]                   # 等待 HReady ACK
     -> Cube2(Q @ H_before)
     -> wait vec2Done[s]                 # 复用 H/V workspace 前
     -> wait hv1VUpdateReadyForHc2[s]
     -> set hc2KvUpdateReady[s]                 # V128, Cube2 FixPipe 完成
     -> Cube3(masked(QK) @ V_new)
     -> set cube3Done[s]

AIV (V128): wait hc2KvUpdateReady[s]
     -> 整片 FP32 Cube2 * exp(g)，驻留 H UB 槽
     -> wait cube3Done[s]
     -> 逐 tile 计算 (gated Cube2 + Cube3) * scale 并写 O
     -> set vec2Done[s]

AIV (V256): wait cube3Done[s]
     -> 沿用原融合计算与写回
     -> set vec2Done[s]
```

`hc1WhReady[s]` 是双向、分代复用的握手：AIC 发出第 `2n` 代，两个 AIV 各自等待后发出第 `2n+1` 代 ACK，AIC 必须消费 ACK 后才允许同一 stream 再发下一代 Cube1 完成事件。两个 ping-pong stream 使当前 Cube1 与上一任务的 Cube2/Cube3 重叠，但不能改变同一 stream 的代次顺序。

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
Cube3 仍使用独立的 attn workspace；`hc2KvUpdateReady` 在 Cube2 的 FixPipe 完成后发布，
`cube3Done` 在 Cube3 完成后发布。第二段对同一个 FP32 H tile 先 Add 再 Muls，
保持原融合公式和 cast 顺序。两个 stage 的 H UB 槽不重叠；`vec2Done` 在最后一块
attn MTE2 结束后通知 AIC 复用 GM workspace。V256 的每 AIV 半片可达 64 KiB，
超过单个 H UB 槽，保留原单阶段输出流程。

### 2.4 H 到 O 的 CrossCore handoff

H producer `p in [0,T)` 负责 task `p`，O consumer `T+p` 按相同的 task/chunk 顺序消费。自然指数路径使用 mode `0x0` 的 H_READY/V_READY flag，与每个 MIX core 内部使用的 mode `0x2` flag 分离。A5 V128 preload 路径的 V_READY 由 H/O 两侧对应 AIV 执行 mode-0 `Set/Wait`；H_READY 保留 AIC rendezvous。H_READY 按七个 MIX task pair 一组映射到 flag `0/2/4`。V_READY 使用本侧相对 AIV block index，按六个 AIV 一组连续映射到 flag `0..6`；O AIV 先减去 `producerCoreNum * subBlockNum`，再与 H AIV 使用相同的相对编号。

| flag | 事件 | 生产方 | 消费方 |
| --- | --- | --- | --- |
| `0/2/4` | `HReady` | H 初始状态或 Vec2 写回 `H_i`；O Vec1 完成独立 gate/mask 计算后 | 每七个 MIX task pair 一组；双方等待完成后，O AIV 通过 `hc1WhReady` 通知 O AIC 读取 H |
| `0..6` | `VReady` | H Vec1 完成 `V_new_i` 的 GM 写回后；O Vec1 完成 masked QK 后 | 每六个相对 AIV block index 一组；随后 O AIV 通过 `hv1VUpdateReadyForHc2` 通知 O AIC 执行 Cube3 |

每个任务固定使用 `taskLane=0`。V_READY 的 H 侧相对 AIV 编号为原始 AIV `blockIdx`；O 侧相对编号为 `blockIdx - producerCoreNum * subBlockNum`。因此同一 task 的两个 H/O AIV 分别得到相同编号，并映射到相同的 V_READY flag group。

`pipelineSyncWorkspaceOffset` 和原有 IB workspace allocation 保持 ABI 兼容，但自然指数 H/O 不再初始化或访问其中的 event 槽；入口不再执行同步区初始化或额外启动屏障。mode `0x0` flag 不携带 task id，正确性依赖所有活动 H/O AIV 按相同 chunk 顺序参与集合握手。

H 的 GM 写回完成后再发布 ready。A5 V128 中，O AIV 等到 V_READY 后才发布本核 mode `0x2` `hv1VUpdateReadyForHc2`，保证 Cube3 同时看到 masked QK 和 V_new；handoff 数据仍按任务/chunk 独立编址。

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

A5 普通 H/O kernel 的默认布局由 `chunk_fwd_h_o_fused_ub_layout.h` 声明；下列分区描述普通 A5 O/MODE1 epilogue 的地址合同。MODE2 fused epilogue 显式选择 `chunk_fwd_ho_fused_a5_single_slot_ub_layout.h` 中按临时变量存活期分配的独立数据区，通信保留区仍沿用相同边界：

```text
[0 KiB,   71 KiB)  base region
[71 KiB, 199 KiB)  Cube Fixpipe work slots
[199 KiB,248 KiB)  Vec scratch
[248 KiB,256 KiB)  IB communication reserved region
```

旧 IB 协议的 32-byte local tensor 保留在通信区起始地址 `248 KiB`。该区域不参与 Cube、Fixpipe 或 Vector 数据复用，因此 L0C 到 UB 的写入不会覆盖 IB 指令的本地操作数。H 与 O 使用同一常量，避免两侧各自维护裸偏移。

共享布局对通信区的 32-byte 对齐、32-byte IB operand 容量和 256 KiB UB 总容量执行编译期检查；两个 O epilogue 也以通信区起点作为数据区硬上界。后续扩展任何 O UB tensor 时，若侵入 `[248 KiB, 256 KiB)` 将直接编译失败。A2 使用独立的 192 KiB UB 布局，继续保留其现有同步偏移，不复用该 A5 常量。

当前 arch35 exp2 专用 O 仍保留原有阶段边界同步；mode-0 H/O 握手只用于自然指数路径。

MODE2 fused layout 的完整变量区间、跨阶段 alias 和交接条件见下方
“MODE2 Local Event 与 UB 生命周期”。普通 O epilogue 使用默认构造参数，不读取该 fused layout。

## 3. Host tiling 约束

MODE1 Host 设置 `producerCoreNum=consumerCoreNum=T`、`activeCoreNum=2*T`、`consumerCoreBase=T` 和 `scheduleMode=1`，要求 `T <= floor(P/2)`。A5 MODE2 在 `floor(P/2) < T <= P` 时设置 `activeCoreNum=T`、`executionMode=2` 和 `singleCoreTaskNum=T`，producer/consumer 分区字段置零以避免误入 MODE1 handoff。当前 MODE2 限定自然指数、定长、V=128、无 gk；不支持的 saturated-range 输入显式失败。两种路径均拒绝变长 `cuSeqlens`/`chunkIndices`。

MODE2 每个 MIX core 单槽处理 chunk。首轮 AIV 预置 `hv2HReadyForHc1`（供 HC1 消费）；非尾块 HV2 完成后才发布下一代信号。每 chunk 的依赖为：AIC `OC1 -> HC1(wait hv2HReadyForHc1) -> OC2 -> HC2(wait HV1) -> OC3(wait OV1/HV1 token)`；AIV `OV1(wait OC1) -> HV1(wait HC1) -> HV2(wait HC2) -> OV2(wait OC2, OC3)`。HC1 已在加载 H 前消费 H-ready 信号，顺序执行的 OC2 使用同一份 H，无需再次等待。尾块不发布仅供下一 chunk 消费的 `hv2HReadyForHc1` 和 `ov2ChunkFinished`，避免 kernel 结束时遗留跨核 token。所有这些通知均使用核内 mode `0x2`；MODE1 的 kernel 和 mode-0 producer/consumer 协议不变。

MODE2 所有 chunk（包括尾块）统一按实际 `blockTokens` 组装 MMAD shape 和 Vector 长度，不设尾块专用执行分支。H 的四个 stage、成员和初始化直接归入 `gdn_fwd_ho_fused_kernel.hpp`，HC1/HC2 使用私有 single-slot preload epilogue 和共享 H_DIRECT UB 输出。五个矩阵均使用 `MmadPingpongTlaPreload`：H 保留带 `vecDone` 的重载；OC1/OC2/OC3 调用不带 `vecDone` 的四参数重载，跨核等待由 driver 的 chunk gate 和 OC3 ready token 保证。HC1/HC2 共用 L1 [0,128 KiB)，OC1、OC2、OC3 分别使用 [128,256)、[256,384)、[384,512) KiB，三个 O 窗口彼此独立；OC2 执行完整 preload 调用，不再单独提前预取。L1 footprint 由编译期断言检查。当前工作区的 Cube event 生命周期为循环前一次 preset、循环后一次 drain；本次 Vector 事件隔离不修改该生命周期。最终状态无需保存时的 HC2/HV2 跳过属于状态输出条件，不是尾块 shape 分支。

### MODE2 Local Event 与 UB 生命周期

MODE1 和共享 O epilogue 保持不变。MODE2 的 local HardEvent 以 `(方向, ID)` 为键；
O 使用 ID0-3，两个 H single-slot epilogue 将原 ID0-3 完整映射到 ID4-7：

| 所有者 | ID | 协议 |
| --- | --- | --- |
| O epilogue | 0-3 | 保留调用内 Set/Wait；MODE2 OV2 使用单输出槽并在每段复用前等待 MTE3 |
| H output | 4 | `MTE3_V` 在 driver preset，HV1/HV2 wait 后 republish，退出 drain |
| H transfer/init | 5 | epilogue 调用内配对；初始化返回前 drain MTE3，不遗留 token |
| H input | 6 | `MTE3_MTE2` 在 driver preset，HV2 wait 后 republish，退出 drain |
| H scalar/fence | 7 | scalar 事件调用内配对；H→O fence 使用不同方向的事件 |
| O→H fence | 3 | OV1→HV2、OV2→下一 chunk OV1 前配对 Set/Wait，不保留 token |

H 入口/出口只保留 ID4、6 的上述两个持续 token；最终 HV2 跳过时不改变它们。
空 token 行也必须走原有事件协议，不能引入未配对的早退。
初始化使用 ID5 闭环，不依赖 O 域 preset。跨核 mode-`0x2` flag 编号及其消费方不变。
HV1 在 `Vec1PostVF` 最后读取 H_DIRECT 并完成 V→MTE3 依赖后，才发布
`hv1VUpdateReadyForHc2`，防止 HC2 Fixpipe 提前覆盖同一槽。

AIV 顺序为 `初始化 → mask/OV1 → HV1 → O→H fence → HV2 → H→O fence → OV2`。
OV1 与 HV2 复用 QK output/update scratch，HV1 与 OV2 复用 H_DIRECT/work 区域。
因此 HV1 不等待 OV1 的整段 MTE3；只有 HV2 覆盖 OV1 scratch 前执行 O→H fence，
只有 OV2 覆盖 HV1 区域前执行 H→O fence。OV2 完成后仍执行 O fence，保护下一 chunk
重新进入 OV1 时对 `[71,103)` 的复用。每个 fence 配对 `MTE2_V`，执行 `PipeBarrier<PIPE_V>`，再配对 `V_MTE3`、
`MTE3_V`、`MTE3_MTE2`，将前阶段的读、计算、写队列交接给后阶段；不用全核屏障。
HV2 跳过仍执行 H→O fence，以覆盖 HV1 输出。OV2 内部 token drain 后再执行 O fence；
仅非尾块随后发布 `ov2ChunkFinished`，供下一轮 AIC chunk gate 消费。尾块不发布该 token，
下一轮 mask 重建之前的 O fence 只在确有下一轮时才会执行。

AIC 在每轮开始先等待上一轮的 `ov2ChunkFinished`，再发射 OC1/HC1/OC2/HC2/OC3。
不能把该等待移到 HC1 之后：AIV 的 OV2 等待当前轮 OC2/OC3 ready，而 OC2/OC3
由 AIC 在 HC1 后产生；若 AIC 先等 OV2 完成再发 OC2/OC3，会形成环形等待。同时，
入口 chunk gate 保证下一轮 HC1 不会与上一轮 OV2 对 `H_DIRECT` 的复用并发。

MODE2 的 O epilogue 显式启用统一 UB layout；普通 A5 O kernel 使用默认构造分支，
保留原地址和行为。以下区间为每个 AIV 的实际最大访问跨度，单位 KiB，右端不含：

| 临时变量 | 区间 | 存活期/复用约束 |
| --- | --- | --- |
| 初始状态/H handoff | `[183,215)` / `[215,247)` | 初始化专用 staging；与运行期临时区分离 |
| causal mask / H calc | `[231,247)` / `[0,32)` | mask 为每个 AIV 初始化一次的只读常量；全 kernel 期间独占该区间，OV1 各 chunk 复用 |
| QK left/up/gcomp/BRC scratch | `[16,36)` / `[36,52)` / `[52,53)` / `[53,69)` | OV1 同时存活；OV2 后续可复用 left/up/BRC |
| U/V_NEW/VUpdate / H tile | `[32,48)` / `[48,64)` | HV1、HV2 依序使用；不与初始化 staging 同时存活 |
| QK a/out/out_low | `[71,87)` / `[87,103)` / `[103,111)` | OV1 活跃 scratch；HV2 update scratch 复用 `[71,103)` |
| H_DIRECT | `[117,181)` | AIC HC1/HC2 Fixpipe 目标，AIV 后续读取；HV1 最后读取完成前不可覆盖 |
| gate / G_LAST | `[181,182)` / `[247,248)` | 独立 float 槽；避免 HV1/HV2 地址复用 |
| gate input | `[182,183)` | 低精度输入暂存；与 gate float 分槽，覆盖 O/H 所需 dtype |
| HV2 update scratch | `[71,103)` | 32 KiB；OV1 完成后复用 QK A/out |
| OV2 A/H work | `[117,133)` / `[133,165)` | 单份 A 16 KiB、H 32 KiB；H→O fence 后复用 H_DIRECT |
| OV2 output FP32/low | `[72,88)` / `[88,96)` | 单份；下一个片段开始前等待前一段 MTE3 写回完成 |

`[117,181)` 由 H_DIRECT 与 OV2 A/H work 按阶段复用；OV1 QK scratch、HV 临时区和
OV2 output 也按阶段有序复用。阶段切换由 OV1→HV2、HV1→OV2 和 OV2→下一 OV1 的 fence 保证。OV2 每次复用输出槽前等待
异步 GM DataCopy 完成，不保留 ping/pong。最高活动终点为 `248 KiB`，紧邻但不进入
`[248,256) KiB` 通信保留区。L1 的 OC1/HC1、OC2、OC3 窗口调整为互不重叠；静态断言
检查各窗口容量和 UB 数据上界。
离线回归检查事件状态与源码交接顺序，不代替硬件流水线/跨核代次验证。

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
TILING_DATA_FIELD_DEF(int64_t, executionMode);
TILING_DATA_FIELD_DEF(int64_t, singleCoreTaskNum);
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

公开 ACLNN/L0 输入、输出、属性顺序和 kernel ABI 顺序保持现有实现不变。新增 MODE2 字段追加在 H 前缀之后，`offsetof(useExp2)` 与 H stage struct 的静态断言保持成立。验证需覆盖：MODE1 原路径配置不变；MODE2 满足 `activeCoreNum=T` 且只有同核 mode-0x2 同步；MODE2 的首轮预置、每 chunk token 代次和 AIC/AIV 等待闭环正确。

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


## 固定 shape MODE2 QK-mask 独立实现（规则 V2，尚未接入调度）

本节定义 `EpilogueAtlasGDNFusedMode2Qkmask` / `block_epilogue_gdn_fused_qkmask.hpp`
的限定实现。仅覆盖自然指数、完整 chunk=64、K=V=128、每 MIX core 两个 AIV，
不改变现有公开接口、host 支持范围或 MODE1/现有 MODE2 分派。
本轮仅实现 OV1 构件；不能将它直接替换当前 epilogue 后声称整表已经接入。

### OV1 数据、地址与公式

AIV lane s=0/1 处理绝对行 `i=s*32+r`、`r∈[0,32)`，列 `j∈[0,64)`。
输入 QK 为 FP32（GM搬入或OC1 Fixpipe直接写Ub::attn_ping），gate由调用方从GM
FP16/BF16/FP32准备至UB；输出maskedQK为FP16/BF16，支持GM或64x64 nZ L1。
语义保持现有 QK-mask：有效位置计算 `QK[i,j]*exp(min(g[i]-g[j],0))`，
未来位置输出精确零。指数使用 FP32、非正指数避免正向溢出；无效位置在 Exp 前选择安全零，
乘法后再次选择零。该约束继承有限合法输入，未定义新的 NaN/Inf 公共接口语义。

| UB 区域（每 AIV，KiB） | 大小 | 内容 / 生命周期 |
| --- | --- | --- |
| [0,8) | 8 KiB | QK 32×64 FP32，MTE2→VF |
| [8,16) | 8 KiB | 行 gate 广播，VF 原地改为 maskedQK FP32 |
| [16,24) | 8 KiB | 列 gate 广播，VF 后复用起始 4 KiB 作为 cast 输出 |
| [24,28) | 4 KiB | 常驻 T32 下三角，由调用方在循环前初始化一次；两个 lane 内容相同 |
| [28,28.125) | 128 B | 16 位 gate 输入；FP32 输入直接搬至 gate FP32 区 |
| [28.5,28.75) | 256 B | 64 个 FP32 gate，由调用方每 chunk 准备一次，整 chunk 只读 |
| [29,37) | 8 KiB | 两次 Broadcast 依次复用，结束后复用为临时 32×64 mask |

`gdnFusedMode2UbMallocTable` 是本实现唯一地址来源。持久 mask 只有一份 4 KiB；
临时全宽 mask 不增加分配。lane0 在工作区中组装 `[T32,0]`，lane1 组装 `[1,T32]`。
`Copy(mask32, repeat=32, mask=32, strides={1,1,8,4})` 以 256 B 目的行距和
128 B 源行距把 T32 放入对应半行，读取严格限制在 T32 的 4 KiB 区间。

两次 Broadcast 间、Broadcast→临时 mask、mask→VF、VF→cast 均有 PIPE_V 屏障。
一个不接收 attn 的 VF 遍历本 lane 的 32 行，完成 Sub→Mins→Exp→因果置零。
广播、mask 组装和该 VF 均在 OC1 等待之前；随后等待 OC1→搬入 QK→MTE2_V→Mul→cast/MTE3。
GM路径每lane输入8KiB QK、输出4KiB maskedQK；enableL0C2UB跳过QK的GM搬入，
enableUB2L1将输出改为写入调用方提供的64x64 nZ L1，开关独立且默认false。OV1为OC3的左操作数A，
按A5约束使用nZ；通过16x16 UB分形转置后写入L1。调用方每 chunk 另搬入一次
gate 128/256 B 并按需转 FP32，供 OV1/HV1/HV2/OV2 共用。
最大活动结束地址 37 KiB，HC1 新布局从 37 KiB 开始，与 OV1 并行区域分离。
Broadcast 分配的 8 KiB 是否满足目标 CANN 对这两个 shape 的最小 workspace 要求，
必须经该 SDK 的临时空间查询和设备验证确认；尚未据硬件证据确认。

### 集成与双向同步约束

- 创建本 specialization 前，确认 chunk=64、K=V=128、两个 AIV 和自然指数；尾 chunk
  或其它 shape 留在原分派路径，不能调用本固定 shape 入口。
- 调用方在 chunk 循环前初始化 T32 一次，将 mask 的 LocalTensor 传入构造函数；
  epilogue 不获取或初始化 mask 地址。整表其它阶段不得覆盖 [24,28) KiB。
- 调用方每 chunk 执行一次 `GateBuffer<GateT>::Prepare(gmG)`，并向所有 epilogue
  构造函数传入其 `Tensor()`。当前 chunk 所有读者完成前不得准备下一 chunk。
  GateBuffer 独占闭合的本地事件 7，不得被调用方额外预置 token。
- OV1 调用参数只有 maskedQK、QK 当前 chunk 的 GM 起点，内部按 lane 加 `32*64` 元素行偏移；
  调用方传入 `oc1Ready`，OV1 内部在 QK 的 GM→UB 搬运前等待该信号；调用方不得重复等待。
  Cube 在 QK 写回完成后发布 ready。返回前 MTE3 完成，调用方才能发布 OV1 ready。
- 上一 chunk 的消费者释放后才能再次写 GM workspace。构件只闭合自身
  MTE2/V/MTE3 事件，不替代 AIC/AIV ready/free 协议。
- 整体切换内存表还需同时迁移 HC1 目的 UB 到 37 KiB、HC2 到 141 KiB，
  配套改写 HV1/HV2/OV2 以及初始化 staging，并重新验证所有阶段并发地址与释放条件。
  现有布局的 HV calc [0,32) 会覆盖本表的常驻 T32，故本构件保持独立未被 driver 实例化。
- 本轮方案评审：固定 shape 数量/地址/象限分解及事件配对经静态检查；设备资源与精度
  验收保留为未完成，不宣称本独立构件已经可在生产调度直接启用。


### 固定64/128其它构件的等待位置与RegBase边界

独立HV1/HV2/OV2按最晚的数据依赖点等待Cube：HV1在HC1前完成u的搬运/cast和
exp(g[63]-g[:])；HV2在HC2前完成旧状态h*exp(g[63])；OV2在OC2前准备exp(g[:])，
在OC3前完成inter的搬运和逐行衰减。等待后的函数不再接收用于提前准备的raw gate。
HV1/OV2分别复用29KiB开头256B存储衰减，不增加内存表分配，必须在OV1/HV1读者完成后交接。
详细函数、顺序和本地检查见mode2_fixed64_hv_ov_epilogues.md的“其余构件的依赖边界检查”。

### 固定64/128 OV1直连配置

构造参数新增enableL0C2UB/enableUB2L1和可选L1输出tensor；前者要求OC1 Fixpipe以SPLIT_M
将每AIV32x64 FP32、stride64写入UB偏移0，并在完成后发OC1 ready。后者要求完整64x64 nZ L1
目的，按16x16 tile转置并分两AIV行区间写入，后续Cube按A=nZ读取，不能套用128x128矩阵布局。
Gate/mask提前计算顺序保持；GM fallback的MTE2_V只在实际搬运时执行；所有路径搬出完成才返回。
该构件未接入driver；Cube输入UB复用和L1输出消费/释放的双向协议仍须由调用方配套。


### 固定64/128 HV1/HV2/OV2的直连/GM配置

三个新构件构造参数统一为resource、gate、enableL0C2UB=false、enableUB2L1=false及L1目标。
HV1增加GM whInput（64x128），HV2增加GM hWorkInput（128x128），OV2沿用两份GM输入；
仅false输入在对应ready后搬入。true输入使用37KiB WH、141KiB h_work、37/53KiB inter/intra。
L1输出使用共享助手CopyRowsToL1Zn：HV1两块64x128 zN，HV2非末chunk一块128x128 zN，
OV2一块64x128 zN；最终需保存的HV2 final_state始终按原dtype写GM。
HV2在GM/L1两种输出路径均保留h_ping供下一chunk使用，Vec直接递推FP32结果，DataT副本仅供Cube读取；
FP32 final_state递推保持原流程。h_ping独占跨chunk，同一task沿用同一UB，task首块重置初态。
Cube直写37/53KiB前须等待HV1读者释放；ready只证明结果就绪，不能替代生产者覆盖前的free。
所有新路径均未接入旧driver，L1槽由新调用方预留，布局和双AIV完成/释放协议见独立构件文档。


### 固定64/128 HV1分段生产与ready

按用户要求HV1先只计算vnewDecay，完成cast和GM/L1搬出后发decayReadyForHc2，
再独立Sub计算vnew，完成cast与搬出后发vnewReadyForOc3。两个flag由调用方传入，
内部用mode0x2/PIPE_MTE3发布；返回后的外部重复发布必须移除。
两段共用53KiB的8KiB cast槽，61KiB的u_fp32和37KiB的WH保留到第二次Sub完成。
早发HC2 ready不能作为输入UB free；HC2目的141KiB与这些存活输入不重叠。
原FP32减法/衰减精度保持，decay计算不得提前舍入，vnew不能从舍入后的decay反推。


### 固定64/128 HV2初态所有权

初始state由driver在每task的chunk循环前准备到每AIV h_ping（64x128 FP32），
完成必要清零/搬运/转换和同步后保持到首HV2。新HV2移除initialState/useInitialState参数；
所有chunk直接原地消费并更新h_ping，不恢复state。hInput/isInitialState参数已删除，
DataT staging仅承接Cube输入或最终输出转换；Vec递推始终保留FP32，不受storeFinalState影响。
调用方另行准备首HC1/OC2所需H。示例见独立构件文档“初始state由driver在循环前准备”。


### 固定64/128 HV2生产者通知

HV2新增传入hReadyForHc1。非末chunk在GM/L1 H搬出及MTE3本地等待完成后，
内部以mode0x2/PIPE_MTE3发布ready；末chunk不发布。调用方移除外部重复set，
Cube侧等待协议及目的槽free仍由driver配套，当前新构件尚未接入旧driver。


### 固定64/128 每AIV H空间压缩

H总shape为128×128；每AIV64×128 FP32仅需32KiB。h_ping保持109KiB，
h_work_ping改为141KiB，结束于173KiB。OV2复用37/53/69KiB地址不变，
HV2 DataT暂存仍位于93KiB，避免与37KiB OC2目的重叠。


### 固定64/128 独立OV2区域的编译期边界

按当前内存表，OV2的inter/intra/out分别从173/189/207KiB开始，输出结束于215KiB；
替代前述37/53/69KiB复用方案。内存表统一检查32B对齐及各存活区域相邻边界，
包括h_work到o_inter、o_inter到o_intra、o_intra到o_out及o_out到248KiB通信保留区。
允许205到207KiB之间保留空隙。OV2局部断言改为通信边界，HV2检查其与OC2目的的边界。


### 固定64/128 OV2生产者完成通知

OV2新增传入chunkFinished，GM/L1输出指令之后内部以mode0x2/PIPE_MTE3发布完成通知。
调用方不重复set，每次调用发布一次；末chunk的token消费需在driver接入时配套。
输入仍依次等待OC2/OC3 ready，保留当前本地事件设置。


### Fixed64/128 Mode2 driver rewrite (2026-10-19)

Add independent schedule/driver files; retain the legacy path for other shapes.
Scope: fixed length, natural exp, K=V=128, chunk=64, complete chunks, one task per MIX core.
Use uint64 offsets, value-head W/U/g and grouped Q/K. Cube: OC1/HC1/OC2/HC2/OC3.
Vector: OV1/HV1/HV2/OV2. Cube outputs support GM or SPLIT_M UB; Vector intermediates
support GM or independent L1. O/final_state remain GM. Default: Cube-to-UB plus Vector-to-GM (CubeToUB=true, VecToL1=false).
All five Cube stages call the shared BlockMmadTla implementation in
common/kernel_utils/block/block_mmad_pingpong_tla_preload.hpp. Configure one L1A,
L1B, L0A, L0B and L0C stage for the single-slot design. Use each stage's exact
compile-time M/N/K: OC1=64/64/128, HC1 and OC2=64/128/128,
HC2=128/128/64, OC3=64/128/64. This preserves the external L1 zN strides.
A private adapter only binds the protected L1 operand slots and substitutes a
no-op GM-to-L1 copy for operands already produced in L1. The shared block owns
L1-to-L0, MMAD, Fixpipe and local events. All shapes bind scratch A at 0 KiB
(capacity 16 KiB) and scratch B at 16 KiB (capacity 48 KiB, ending at 64 KiB).
The driver presets shared block events once before the entire Cube chunk loop
and drains them once after the loop. Each block consumes and releases the same
single-stage A/B/C tokens, allowing operand loading to overlap the preceding
Fixpipe without overwriting live L1/L0 data. One L0C slot still serializes its
reuse between Fixpipe and the next MMAD. Resident GM cache is disabled.
Cross-core producer ready signals remain in the driver.
L1 KiB: scratch=[0,64), H=[64,96), mask=[96,104), Vnew=[104,120), decay=[120,136).
Actual M/N/K layouts must match CopyRowsToL1Zn FullRows.
Initialize FP32 UB H, Cube H and the 32x32 mask before the loop. Publish initial H-ready.
HC1 consumes H-ready; OC2 uses the same H before HV2 overwrites it.
HV1 publishes decay-ready and Vnew-ready; HV2 publishes next H-ready only for nonfinal chunks.
OV2 publishes finished for every chunk; Cube consumes it before the next chunk and at final exit.
Skipped final HC2 still consumes decay-ready and publishes HC2-ready to match HV2.
HV1's two casts reuse 53KiB within one chunk: set MTE3_V after decay copy and wait just
before the Vnew cast, allowing Sub to overlap. No per-chunk HV2 drain. Driver fences
final MTE3 completion before returning. Retain host ABI/workspace and legacy fallback.
CPU checks are development evidence; CANN/NPU build, precision and pipeline validation remain required.


### Mode2 final review and default transport (2026-10-19)

Default CubeToUB=true, VecToL1=false as requested: the five Cube outputs go to UB;
Vector intermediate outputs go to GM and the shared preload block loads them to L1.
Vec-to-L1 remains an explicit optional template choice, not the default.
OV2 sets V_MTE2 EVENT_ID7 immediately after ScaleInter completes the last use of
gate/decay scratch. Each noninitial chunk waits for that event before the next
gate DMA; kernel exit consumes the final chunk's event. There are N sets and N
waits for N chunks, including the one-chunk case. This event releases gate/scratch;
the existing final MTE3 fences remain responsible for output completion.
Full OV1/HV1 overlap still requires splitting OV2 and adjusting Cube slot waits.
FP32 final_state is copied only on the final chunk, matching the DataT state path.
