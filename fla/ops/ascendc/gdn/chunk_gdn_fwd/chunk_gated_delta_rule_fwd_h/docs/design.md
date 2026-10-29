# ChunkGatedDeltaRuleFwdH Design

方案设计规则版本：V2。公开接口见 [api.md](api.md)。

## 1. 目标

### 1.1 融合范围与调用链

在既有 `chunk_gated_delta_rule_fwd_h` 内新增 `TilingKey 3` 的 chunk 并行（CP）
路径，作为单 kernel 内的三个阶段实现；`TilingKey 1/2`（链并行，V=128/256）与
arch22（A2/A3）实现完全不变。公开 aclnn/Python 接口、输入输出 dtype/layout、
`state_v_first` 语义均不变；workspace 仍由 host tiling 返回并按 stream 分配。

调用链仍为 `aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize` →
`aclnnChunkGatedDeltaRuleFwdH(workspace, size, executor, stream)`；CP 路径只改变
device 内实现与 workspace 内部布局。

### 1.2 数学边界

CP 路径只接受：

- Ascend 950；
- dense（无 `cu_seqlens`/`chunk_indices`）；
- 仅 scalar gate `g`（`useG && !useGk`；gk 组合回退 `TilingKey 1/2`）；
- `K = V = 128`，`chunk_size ∈ {32, 64, 128}`；
- `GDN_FWD_H_VSPLITS` 未开启（`vSplits == 1`）；
- `ceil(T/chunk) >= 2` 且 `B*HV <= AIC 核数`，保证 task 串行结构不劣于链并行；
- 尾块 `T - (C-1)*chunk >= 16`：CP 把 `tokens` 直接当作 Cube GEMM 的 M，而串行实现
  把不足 16 个 token 的短块交给向量侧，因此更短的尾块一律回退 `TilingKey 1/2`；
- 环境变量 `GDN_FWD_H_CP=0` 可强制回退链并行（紧急回退与 A/B）。

其余输入按原有规则走 `TilingKey 1/2`。R 取 `min(AIC 核数, chunk 数)`，因此每个
rank 至少拥有一个 chunk，不存在空 rank；`blockDim = R`。

每 chunk 的精确递推（与 CPU FP64 标杆一致）：

```
v_new_c = U_c - W_c @ H_c
q_t     = exp(g_last - g_t)              g_last = g[chunk 最后一个有效 token]
D       = exp(g_last)
H_{c+1} = D * H_c + K_c^T @ (v_new_c ⊙_row q)
h[c]    = H_c                            (chunk 起始状态)
```

仿射形式 `H' = B_c H + A_c`，`B_c = D*I - (q⊙K_c)^T W_c`，`A_c = (q⊙K_c)^T U_c`。

### 1.3 分核方案与性能可达性

按用户指定方案：`task = (batch, value head)` 顺序执行，全部 AIC 作为该 task 的
rank 沿 chunk 轴切分（覆盖通用规则 R20 的 HV 轴分核默认值）；每个 AIC 配对两个
AIV。`taskCount = B*HV`，`rank r` 负责连续区间
`[r*base + min(r, rem), ...)`，`base = C/R`，`rem = C%R`。

每 chunk 归一化到 `128³` MAC 的计算量：阶段 1 = 2（`[P|v_work]` 与
`[T|S_work]` 两次 GEMM），阶段 2 = `2*rank`（每前序 rank 一次
`128×256×128`），阶段 3 = 1（与串行同构）。CP 相对链并行的 break-even 由
`B*HV × (每 task 时间) / R` 决定；`B*HV` 接近核数时可能劣于链并行，因此
`validation.md` 要求逐 case 记录实测并允许用 `GDN_FWD_H_CP` 回退。

## 2. Stage 完整详设

### 2.1 符号与任务域

| 符号 | 含义 |
| --- | --- |
| `BT` | `chunkSize`（32/64/128） |
| `K=V=128` | head 维度（CP 仅支持 128） |
| `C = ceil(T/BT)` | 单序列 chunk 数 |
| `R = min(AIC, C)` | rank 数（= blockDim） |
| `r` | AIC `GetBlockIdx()`；AIV `GetBlockIdx()/GetSubBlockNum()` |
| `X = [M \| S]` | 128×256 FP32 rolling 状态：M 为转移矩阵，S 为段偏置 |
| `[B̄_r \| Ā_r]` | rank r 的 summary（128×256 FP32） |

每 task 的 `b = task/HV`、`hv = task%HV`、`hk = hv/(HV/HK)`。

### 2.2 存储与生命周期

GM workspace（相对 `GetUserWorkspace()`，全部 512 B 对齐）：

| 区域 | 每 rank 大小 | 内容 | 生命周期 |
| --- | --- | --- | --- |
| summary | 128 KiB（32768 float） | `[B̄_r \| Ā_r]` | 阶段 1 末写，阶段 2 读（全局屏障后），阶段 3 及下一 task 的初始化之前不再使用 |
| scratch | 704 KiB（180224 float，含 512 B 对齐） | `X(128×256)`、`H0/S_in(128×128)`、`YT(128×256)`、`Wf(128×128)`、`Kf/v_update(128×128)`、`Pv(128×256)`、`RpRv(128×256)` | 每 rank 私有，task 内复用；`X` 在阶段 1 结束后可由阶段 2 的 Y 覆盖 |

host 侧公式（`chunk_gated_delta_rule_fwd_h_tiling_processor.h`）：

```
summaryBytes = AlignUp(2*128*128*4)
scratchBytes = AlignUp((4*128*256 + 3*128*128)*4)
cpSummaryOffset = ws; ws += summaryBytes * R
cpScratchOffset = ws; ws += scratchBytes * R
```

CP 路径不再分配 serial 的 `vWorkspace/vUpdate/kDecay/hWorkspace/stateWorkspace`
（tiling 中置 0）。

UB（每 AIV，Catlass `Resource::ubBuf` 分配 248 KiB）。所有 GEMM 结果经
`CopyL0CToUBMode::NO_SPLIT` 直接落在 `UB_DIRECT`（`[0,64K)`，行距 256；阶段 3 行距 128），
每个子核只拿到自己那一半 M 行且从 UB 偏移 0 开始，因此 AIV 原址消费。其余区域按阶段复用，
跨阶段或跨一次 GEMM 往返的别名是有意设计，同一阶段内不重叠：

| 函数 | 使用（字节偏移） |
| --- | --- |
| 阶段 1 GEMM-A/GEMM-B | `[0,64K)` 结果 tile（AIV 在各子核把自己那段 M 行写到偏移 0） |
| PrepareOperands | `[128K,144K)` in W、`[144K,176K)` Wf、`[176K,192K)` in K、`[192K,224K)` Kf |
| ComputeRpRv | `[0,64K)` Pv（原址改写）、`[64K,80K)` U 输入、`[240K,241K)` g、`[241K,242K)` q/-q |
| UpdateRollingState | `[0,64K)` T/S_work、`[64K,128K)` X/X' |
| 阶段 2 初始化 | `[64K,128K)` 单位阵/H0、`[128K,192K)` H0 输入 |
| 阶段 2 前缀 | `[0,64K)` GEMM 输出、`[128K,160K)` Ā、`[192K,208K)` h 输出 |
| 阶段 3 V1 | `[0,32K)` v_work（原址改写）、`[64K,80K)` U 输入、`[144K,160K)` v_update 输出、`[160K,176K)` v_new 输出、`[240K,242K)` g/q |
| 阶段 3 V2 | `[0,32K)` h_work、`[64K,96K)` S/S_next、`[128K,144K)` final_state 输出、`[160K,176K)` h 输出 |

峰值 224 KiB（阶段 1）≤ 248 KiB。

L1/L0（Catlass `BlockMmadTla`）：

- FP32 前缀/摘要 GEMM：L1 tile `[128,256,128]`，L1A 64 KiB×2、L1B 128 KiB×2
  （384 KiB ≤ 512 KiB）；L0 tile `[128,256,32]`，L0A 16 KiB×2、L0B 32 KiB×2
  （L0B 上限 64 KiB）、L0C 128 KiB。
- 重放 GEMM：L1 tile `[128,128,128]`，L1A/L1B 32 KiB×2；L0 tile
  `[128,128,64]`，L0A/L0B 16 KiB×2、L0C 64 KiB。两个 `Common::BlockMmadTla`
  实例按阶段串行使用同一 L1 起点，生命周期由构造函数 `SetFlag` / 析构函数
  `WaitFlag` 闭合（`Common::MmadPingpong` 不提供 `preSetFlags/finalWaitFlags`）。
- 阶段 1/2 的 FP32 GEMM 与阶段 3 的重放 GEMM 都用
  `Common::MmadPingpong<Ascend950, ENABLE_UNIT_FLAG=false, USE_HF32_MODE=false, 1>`
  配 `PackedTileCopyTlaToUB`（`CopyL0CToUBMode::NO_SPLIT`）：L0C 经 Fixpipe 按 M 行
  分给两个 AIV 子核直写 UB。FP32 路径显式 `SetHF32Mode(false)`，即原生 IEEE FP32。

### 2.3 Stage 0：AIV 初始化（每 task）

- `X = [I | 0]`：sub-block 0 写 M 列（128×128 单位阵），sub-block 1 写 S 列
  （全 0）；各用一次 `DataCopyPad`（blockLen 512 B，row stride 1 KiB）。
- 阶段 2 的 `Y = [I | H0]`：M 列同为 I；S 列由 `initial_state` 转换 FP32
  （或 0），同样一次 `DataCopyPad`。
- 全部初始化在置位对应 flag 前用 `PipeBarrier<PIPE_ALL>` / `PipeBarrier<PIPE_MTE3>`
  排空。

### 2.4 Stage 1：AIV/ AIC 交替，rank 局部摘要

每 chunk（`tokens = min(BT, T - c*BT)`）：

```
AIV: W/K 的有效行 cast FP32 → Wf/Kf（各半区，连续块）
AIV: set flag Operand (0x4, 每子核一槽)；set flag UbFree
AIC: wait UbFree；[P|v_work] = Wf @ X  (tokens, 256, 128)，L0C→UB 直连
AIC: set flag GemmA（mmad 内部 PIPE_FIX，两个槽）
AIV: Rp = -q*P；Rv = q*(U - v_work)   （寄存器基逐行缩放，原地）
AIV: set flag RpRv；set flag UbFree
AIC: wait UbFree；[T|S_work] = Kf^T @ [Rp|Rv] (128, 256, tokens)，L0C→UB 直连
AIC: set flag GemmB
AIV: X' = D*X + [T|S_work]；最后一个本地 chunk 同时写 summary 区
AIV: set flag XUpdated
```

`UbFree`（ID 11）是 L0C→UB 槽的回收握手：AIV 用 `CrossCoreSetFlag<0x4, PIPE_V>`
置位（保证向量读已退役），mmad 在自己的输出循环里用 `PIPE_FIX` 等待。每 chunk 置两次、
等两次，与两次 GEMM 一一对应。槽是单缓冲的：阶段 1 是 `X → GEMM-A → RpRv → GEMM-B → X'`
的严格链，第二个槽没有可重叠的对象。

`M' = D*M - (q⊙K)^T (W@M) = D*M + K^T(-q⊙P)` 与伪代码的
`M = B_c @ M` 数学等价，但不物化 `B_c`：每 chunk 的 M 链从 1.25 个 `128³`
降到 0.5 个，且与 S 链共享同一次 GEMM（N 方向拼接为 256）。

无 `gk`、无 causal mask；tail 行只按 `tokens` 参与 GEMM（tail 变体），gate 只读
有效 token。`g` 缺失时 API 不允许（至少一个 gate），`useG` 恒为真。

### 2.5 Stage 2：前缀（rank 升序、左乘）

所有 AIC 在阶段 1 末执行 `PipeBarrier<PIPE_ALL>` 后
`CrossCoreBarrier<0x0, PIPE_FIX>()`（Catlass `AIC_INTER_BLOCK_BARRIER = 9`），
发布全部 summary。

```
AIV: Y = [I | H0]；set YInit
AIC: wait YInit
for prior = 0 .. rank-1:
    AIC: [B_trans|S_local] = B̄_prior @ Y           (128, 256, 128) → YT
    AIC: set PrefixGemm
    AIV: S_local += Ā_prior（半区行分工）；Y = 结果
    AIV: set PrefixAdd
AIV: S_in = Y 的 S 列 → SCR_H0；cast 写 h[chunkBegin]
AIV: 子块屏障（CrossCoreBarrier<0x1>，ID 10）
AIV: set ReplayReady
```

把 `H0` 放进 Y 的初值即实现伪代码的 `S_in = B_trans @ H0 + S_local`，无需额外的
`B_trans @ H0` GEMM；`H0` 每 task 只初始化一次。rank 0 没有 prior 循环，`S_in = H0`。

### 2.6 Stage 3：本地重放

```
for c in [chunkBegin, chunkEnd):
    AIV: set UbFree
    AIC: wait UbFree；v_work = W_c @ h[c] (tokens,128,128)，L0C→UB 直连
    AIC: set C1Done
    AIV: v_new = U - v_work（原地）；v_update = q*v_new（原地）；写 v_new 与 v_update（低精度）
    AIV: set VUpdate
    AIV: if needStage2: set UbFree
    AIC: if needStage2: wait UbFree；h_work = K_c^T @ v_update (128,128,tokens)，L0C→UB 直连；set C2Done
    AIV: if needStage2:
            S_next = D*S + h_work（FP32，SCR_H0 rolling，原地）
            if c+1 < chunkEnd: h[c+1] = cast(S_next)
            if 全局最后 chunk 且 storeFinalState: final_state = cast(S_next) / FP32
            set HDone
AIV: set TaskDone
AIC: wait TaskDone
```

`needStage2 = storeFinalState || c+1 != C`。h 的每个槽只由一个 rank 写：rank 的
第一个 chunk 起始状态在阶段 2 末尾写入，其余由本 rank 的 V2 写 `c+1`；跨 rank
边界不写（下一 rank 从前缀 `S_in` 写）。`storeFinalState && state dtype==FP32`
时 final_state 直接写 FP32；否则 cast 到 state dtype。

数值约定：

- summary、前缀、`S_in`、阶段 1 的 M/S 链全部 FP32；重放的状态滚动也在 FP32
  （比串行的低精度 h 载体更接近 FP64 标杆）。
- `v_new` 输出按输入 dtype 舍入；`v_update` 以输入 dtype 进入 C2；`h` 按输入
  dtype 输出。与串行的舍入点一致。
- gate 的指数范围继承 CPU 标杆值域；本路径不新增指数因子，`q ∈ (0,1]`、
  `D ∈ (0,1]`。

### 2.7 同步资源表

| ID | 方向 | 阶段 | 语义 |
| --- | --- | --- | --- |
| 0 | AIV→AIC | 1 | W/K 已就绪 |
| 1 | AIC→AIV | 1 | `[P\|v_work]` 结果就绪 |
| 2 | AIV→AIC | 1 | `[Rp\|Rv]` 已就绪 |
| 3 | AIC→AIV | 1 | `[T\|S_work]` 结果就绪 |
| 4 | AIV→AIC | 1/2/3 | X 已写回 / ReplayReady / TaskDone |
| 5 | AIV→AIC | 2 | Y 初始化完成 |
| 6 | AIC→AIV | 2 | 一轮前缀 GEMM 完成 |
| 7 | AIV→AIC | 2 | 偏置已加、Y 已写回 |
| 0/1/2/3 | AIC↔AIV | 3 | C1Done / VUpdate / C2Done / HDone |
| 8 | AIV 全体 | 每 task 末 | `CrossCoreBarrier<0x0, PIPE_MTE3>`：跨 rank 的 task 边界，防止快 rank 在下一 task 覆盖 summary/scratch |
| 9 | AIC 全体 | 1 末 | `CrossCoreBarrier<0x0, PIPE_FIX>` |
| 10 | AIV 子块 | 2 末 | `CrossCoreBarrier<0x1, PIPE_MTE3>` |
| 11 | AIV→AIC | 1/2/3 | L0C→UB 直连槽已回收（mmad 在 `PIPE_FIX` 上等待） |

mode 0x4（Ascend950）：AIV 子核写自己的槽（id / id+16），AIC 显式 wait/set 两个
槽；AIC 到 AIV 的 set 必须两个槽各置一次。ID 11 是首个空闲号（0..7 为交接协议、
8/9/10 为 Catlass 屏障、12/13/14 为 `SyncAll`）；它与 0x4 模式下 `+16` 的子核槽
（27）一起使用，未在其他算子里验证过，设备上需先跑哨兵确认。
阶段之间由一个 AIC 全局屏障或配对 flag 链完全排空；每个 task 末尾另有 AIV 全体
屏障（ID 8），保证所有 rank 的 summary/scratch 读写都结束后才进入下一 task。
因此跨阶段复用 ID 不改变计数平衡；每个 ID 的 set/wait 在每阶段内逐 task 配对，
未等待深度 ≤ 1。

### 2.8 数据搬运汇总

每 chunk（阶段 1）：W/K 各读一次（`tokens×128`），U 读一次（`tokens×128`），
FP32 操作数写 `2*tokens×128`；`[P|v_work]` 与 `[T|S_work]` 由 Fixpipe 直写 UB，
不再经过 GM；RpRv 写读各 `tokens×256`；X 读 `128×256`、写 `128×256`。阶段 2 每
prior 读 summary `128×256`、Y 读 `128×256`、写 `128×256`（GEMM 结果不再经 GM）。
阶段 3 每 chunk 与串行实现相同的 h/v_new/w/v/u 流量，C1/C2 结果直接落 UB。
按 chunk 统计，阶段 1 的 GM 字节从约 1.1 MiB 降到约 0.6 MiB；阶段 3 从约 0.9 MiB
降到约 0.4 MiB。所有搬运使用 `DataCopy`/`DataCopyPad`，UB 复用由
`MTE2_V`/`V_MTE3` 与 `PipeBarrier<PIPE_MTE3>` 闭合；`AivSetToAic` 前显式
`PipeBarrier<PIPE_MTE3>`，直连槽的回收用 `CrossCoreSetFlag<0x4, PIPE_V>` 排在
向量读之后。

### 2.9 评审自检（R01–R21 摘要）

- R04/R05/R06：所有计算节点归属唯一 Stage；每 task 的输入输出与 workspace 地址
  由 `ScratchBase()/SummaryBase()` 唯一确定；无原位 alias（X 与 YT 分离）。
- R10/R12：tail chunk 以 `tokens` 作为 GEMM 的 M/K；h 的 K 行固定 128；h 的
  跨 rank 边界不重复写。
- R20：采用用户指定的 chunk 轴 rank 并行，偏离默认 HV 轴；任务域为
  `B*HV <= AIC`。
- R21：FP32 GEMM（HF32 关闭）、`q/D ∈ (0,1]`、复合次数 `C/R + R`；设备侧
  `validation.md` 记录精度与性能证据。
- 已知限制：CP 不支持 gk、varlen、V-split；`B*HV` 接近核数时性能可能低于链并行；
  同步协议在 Ascend950 上的无死锁性以 `validation.md` 的哨兵实验为准。
