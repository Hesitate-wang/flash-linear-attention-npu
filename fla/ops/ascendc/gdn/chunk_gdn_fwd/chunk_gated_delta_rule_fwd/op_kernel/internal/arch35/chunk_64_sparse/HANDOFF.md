# chunk64 sparse（key 64）producer/consumer 实现交接

> 面向下一次会话。先读本文，再看 `gdn_fwd.md`、`../../../../../docs/design.md`、
> 计划文件 `~/.claude/plans/crystalline-tickling-bentley.md`。
> 最近更新：2026-11-02（含首轮真机编译反馈修正）。

## 1. 当前实现摘要

key 64（`B*Hv <= AIC/2`、chunk=64、K=V=128、BF16 输入 FP32 计算）的
producer/consumer 流水已经**在源码层完整落地**：

- `sparse_pipeline.h`：`RunSparsePipeline` 实现 NT+1 轮循环。链 `c` 用 producer 组 `c`、
  consumer 组 `C+c`；第 i 轮 producer 算 chunk i、consumer 算 chunk i-1；每轮末尾全体
  AIV 进一次 mode-0 栅栏（id 7）发布 GM 并节拍；空转组只进栅栏；组内 AIC 由自己的
  AIV 用 mode-2（id 8）转发迭代边界。`schedule=0` 仍走原有串行 Prepare→H→O 回退。
- `sparse_producer.h`：Stage0–8 完整（L2Norm/cumsum/gate/beta → KKT → solve/VCS →
  Y → A → vb/kbg → W/U → W/U 落 GM），AIV0 做全部向量、AIV1 只镜像握手；AIC 只读
  L1、Fixpipe 到 AIV0 UB，W/U 由 AIV0 落 GM。
- `sparse_consumer.h`：H→O 全流程。状态 fp32+bf16 镜像常驻 UB，按 K（`stateVFirst`
  时按 V）64/64 切分；行/向量按 32/32 切分；AIC 依次做 `P=W@H`、`O_s=Q@H`
  （复用 L0B 的 H）、`A_raw=Q@Kᵀ`、`D=Kᵀ@right`（`stateVFirst` 时 `rightᵀ@K`）、
  `O_l=A'@V_new`，每个结果半块 Fixpipe 到两个 AIV；AIV 完成 `v_new/right`、`A'`、
  `o`、`R=αR+D`，并写 `o`（BSND）、`v_new`、`h`、`final_state`。
- Host：`BuildChunkGatedDeltaRuleFwdPipelinePlan` 接入选定条件，`PipelineTilingData`
  随 POD 下发，`schedule=1` 时 device 走流水。
- 首轮 ccec 编译反馈的 4 类错误已全部修复，并额外做了作用域/参数/类型的系统核对
  （见 §6）。

**当前状态**：源码完成，本机 host 契约测试通过；CANN 编译仍在迭代（已修完已知
诊断，尚无“编译通过”证据），设备精度/同步/性能全部未验证。

## 2. 目标与范围

`gdn_fwd.md` 的规格：

```
producer: L2Norm -> cumsum -> KK.T -> solve_tri -> recompute_wu -> W,U, g_sum
consumer: W, U -> fwdh -> v_new, h -> fwdo
```

旧的 `sparse_phase_adapter.h` 只是把独立 Prepare → H → O 串起来（两次
`PhaseRendezvous()`，无重叠），保留为 `schedule=0` 回退。

## 3. 已锁定的设计决策

1. **每链 2 个 MIX 组**：组 `c` = producer，组 `C+c` = consumer，
   `C = B*Hv`（varlen 取上界 `(cu_seqlens-1)*Hv`）。`blockDim = GetCoreNumAic()`，
   前 `2C` 组担角色，其余组空转。
2. **跨 role 同步只用 AIV + mode 0x0**（`AivRoleBarrier`）。AIC 不进 mode-0 栅栏；
   迭代边界由同组 AIV 用 mode 0x2（PIPE_MTE3）转发，AIC 用
   `AicWaitRelayMte1`（PIPE_MTE1）接收。
3. **AIC 不碰 GM**：GM 读写归 AIV；AIC 只读本组 AIV 写好的 L1、把结果 Fixpipe 到 UB。
   已知例外：producer 的公开 `A` 输出仍由 AIC Fixpipe 直写 GM（Stage5A）；它不在
   producer→consumer 可见性边上，后续要严格"GM 只走 AIV"时再改成先落 AIV0 UB。
4. **在 `chunk_64_sparse/` 重写编排/内存/同步**，纯数学助手复用已验证实现（§7）。
5. **consumer 切分**：chunk 行 32/32（AIV0 行 0-31，AIV1 32-63），state 按 K 64/64；
   O 段（A_raw/gateA/A'/O_s/O_l/o）同样按行半落在两个 AIV（旧版"整块放 AIV0"与
   已完成的 UB 表/VF 接口冲突，已改为对称半块）。
6. **cube 约定照 chunk32**（`chunk_gated_delta_rule_fwd_32_matmul.h`）：
   - `transposeA=true` = 源按 `[k,m]` 存；`transposeB=true` = 源按 `[k,n]` 存；
     `false` 分别表示源按 `[m,k]` / `[n,k]`。
   - 半块下发**不用 `dualDstCtl`**：每个半块单独发一次 Fixpipe，`subBlockId` 选 AIV，
     L0C 源从 `(fullM/2)*16` 元素偏移开始（NZ m-frac 步长 16×16）。

### 五条 matmul（实际实现参数）

| 结果 | A | B | m,n,k | transposeA, transposeB |
|---|---|---|---|---|
| `P = W@H` | W `[64,128]` | H `[K,V]` 或 `[V,K]` | 64,128,128 | false, `stateVFirst==0` |
| `O_s = Q@H` | Q `[64,128]` | 复用 P 的 L0B H | 64,128,128 | false, 同上 |
| `A_raw = Q@Kᵀ` | Q `[64,128]` | K `[64,128]` | 64,64,128 | false, false |
| `D = Kᵀ@right` | K `[64,128]`（`stateVFirst` 时 right） | right（`stateVFirst` 时 K） | 128,128,64 | true, true |
| `O_l = A'@V_new` | A' `[64,64]` | V_new `[64,128]` | 64,128,64 | false, true |

## 4. 角色实现细节

### 4.1 producer（每 chunk）

| Stage | 核 | 内容 |
|---|---|---|
| 0 | AIV0 | 常驻 `I_vcs`、leaf 零、64×64 identity 上 L1（`auxReady_` 只做一次） |
| 1 | AIV0 | 载入 q/k/g/beta；tail 零填充；L2Norm；GateSoftplus+CumsumScale（exp/exp2）；BetaSigmoid；写 `g_cumsum_bht`；按需写 q_hat/k_hat/rstd/beta_eff；k̂ 上 L1 |
| 2 | AIC | `KKT = k̂ k̂ᵀ`，Fixpipe fp32 到 AIV0 `ubKkt_` |
| 3 | AIV0 | `-L`（GateLowerL⊙KKT）、diag leaves、VCS 32×32 逆、`-L`/leaf 上 L1 |
| 4 | AIC | `Y = I + LeafLeft @ (-L)`，直接 Fixpipe fp32 NZ 到 L1 `l1Y_` |
| 5 | AIC | `A = I@LeafLeft + Y@LeafRight`；公开 A 用 `FixpipeL0cToGmNd<bf16>`（valid rows）；L1 A 由 `FixpipeAToL1` 写 |
| 6 | AIV0 | `vb=v*beta`、`kbg=k̂*beta*exp(g)` 上 L1 |
| 7 | AIC | `W=A@kbg`、`U=A@vb`，`FixpipeWuToUb`（ROW_MAJOR+F322BF16，m=64/n=128）到 AIV0 bf16 UB |
| 8 | AIV0 | W/U 从 UB 落 GM（valid rows） |

AIV1 只镜像 Stage1/3/6/8 的握手（Set inputs/solve/WuIn，Wait KKT/Wu）。

### 4.2 consumer（每 chunk）

1. **AIV**：`StoreH`（把当前 bf16 状态写到 `h[chunk]`）→ 载入 U/W/q̂/k̂/g（各 32 行带，
   尾部清零）→ `LastGateVF`（gLast/alpha）+ `GateRowsVF`（gateScale、gateO）→
   上传 W/Q/K 半带 + H 状态镜像（chunk 奇偶 ping-pong）→ mode-2 输入转发。
2. **AIC**：`P=W@H` → 半块 Fixpipe；`O_s=Q@H`（复用 L0B H）；`A_raw=Q@Kᵀ`。
3. **AIV**：等 P → `VNewRightVF`（v_new=U-P，right=gateScale*v_new）→ 写 v_new GM、
   上传 V_new/right 半带 → mode-2 Vn 转发。
4. **AIC**：等 Vn → `D=Kᵀ@right`（或 `rightᵀ@K`）→ 半块 Fixpipe（D 每个 AIV 64 行）。
5. **AIV**：等 A_raw → `GateMatrixVF`+`APrimeVF`（**含对角线**，见 §6）→ A' 半带上 L1 →
   mode-2 APrime 转发。
6. **AIC**：等 APrime → `O_l=A'@V_new` → 半块 Fixpipe。
7. **AIV**：等 O_s → `GateOutputVF`；等 O_l → `OFuseVF`（`(O_s' + O_l)*scale`）→ o GM
   （BSND，按 head stride 写出）；等 D → `StateUpdateVF`（`R=αR+D`）→ 下一个 chunk 的
   H 上传/`h` 输出从这里取；链结束后 `Finish()` 写 `final_state`（仅 `store_output_final_state`）。

### 4.3 流水时序

- 迭代 `i = 0..NT`（共 NT+1 轮，`NT = td.pipeline.maxChunks`）。
  producer：`i < chunks` 时 `RunChunk(i)`；consumer：`i >= 1 && i-1 < chunks` 时
  `RunChunk(i-1)`；其余组/轮只进栅栏。
- 每轮末尾所有 AIV 调 `AivRoleBarrier(7)`，随后 `AivRelayToAic(8)`，
  AIC 侧 `AicWaitRelayMte1(8)`。
- producer 的 W/U/q̂/k̂/g 在第 i 轮栅栏发布，consumer 第 i+1 轮读取（领先 1 个 chunk）。

### 4.4 同步协议与 flag 分配

- **mode 0**：id 7，仅 AIV（所有组、两个 subblock）。`PipeBarrier<PIPE_ALL>` 后
  Set（PIPE_MTE3）/Wait（PIPE_MTE2）。
- **mode 2**：id 8（迭代边界）、consumer 的 0/4/6（inputs/Vn/APrime）。
- **mode 4**：producer 0–4（inputs/kkt/solve/wuIn/wu）；consumer 0–6、9
  （inputs/P/Qk/OS/Vn/D/APrime/O_l）。AIV 本地 id 0–15，AIC 0–31。
- mode 0/1/2 与 mode 4 的 flagId 空间相互独立；仍按上表分开使用，便于审查。
- 单个 flagId 未消费计数 ≤15；每条 Set 都有对应 Wait。

### 4.5 内存映射（`sparse_memory.h`）

- producer：UB 177.5 KiB / L1 136 KiB；L1 常驻 `I_64`，其余 `k̂/-L/leaf/Y/A/kbg/vb`。
- consumer：UB 173 KiB（含新增 256 B 标量槽 `kConsUbScalar` 存 gLast/alpha）/
  L1 152 KiB；H 双缓冲，`V_new/right/A'` 各一份。
- L0：producer 两个 16 KiB 槽 + 4 个 64 KiB L0C tile；consumer L0A `{0,16K}`、
  L0B `{0,32K}`、L0C 4×64 KiB。
- UB 预算上限 248 KiB/核，regbase 目标 ≤208 KiB，最终以 `GetCoreMemSize(UB)` 为准。
- 所有偏移为半开字节表 + `static_assert`。

### 4.6 tail 处理

两个 AIV 各用 32 行 UB 带：先把本带无效行清零，再上传整条 32 行带，因此 L1 的
Q/K/W/V_new/right/A' 镜像始终完整定义；cube 维数保持 64/128，不依赖 L1 残留，
也不需要额外清零。`state` 永远整块上传；`h/v_new/o` 只写 valid rows；
`v_new/right` 的无效行在 UB 内清零，`A'` 无效行清零。

## 5. 文件清单与状态

| 文件 | 状态 |
|---|---|
| `sparse_fused_tiling.h` | ✅ `PipelineTilingData`（schedule/barrierFlagId/chainCount/maxChunks/producer•consumerGroups/workspace）、`kSparseScheduleSerial/Pipeline`、`kSparseBarrierFlagId=7` |
| `op_host/.../chunk_64_sparse/chunk_gated_delta_rule_fwd_sparse_tiling.{h,cpp}` | ✅ 选定条件、POD 填充、workspace 追加；host 契约测试通过 |
| `chunk_gated_delta_rule_fwd_arch35.cpp` | ✅ 按 `pipeline.schedule` 分派 |
| `sparse_pipeline.h` | ✅ NT+1 轮 producer/consumer 循环；串行回退保留 |
| `sparse_common.h` | ✅ `ChainLoc`/`ResolveChain`/`ResolveWindow`/GM 偏移 |
| `sparse_memory.h` | ✅ 两套 UB/L1/L0 半开区间表 + `static_assert` |
| `sparse_barrier.h` | ✅ `AivRoleBarrier`/`AivRelayToAic`/`AicWaitRelayMte1`/mode-4 握手；AIV 传 raw id |
| `sparse_producer.h` | ✅ 约 570 行，Stage0–8；`ChunkCount()`；`FixpipeWuToUb` |
| `sparse_consumer_vf.h` | ✅ 全部向量单元 + `LastGateVF`/`StateMirrorVF`/`ChunkUploadL1` |
| `sparse_consumer.h` | ✅ 约 620 行，GM/UB/L1/L0 绑定 + 每 chunk H→O 全流程 |
| `sparse_phase_adapter.h` / `sparse_sync.h` | ✅ 串行回退（保留） |
| `sparse_math.h` | 串行/历史用，当前未被 include |

## 6. 本轮修正记录（按类别）

### 6.0 首轮设备运行：key-64 tiling 静默失败（2026-11-02 晚）

1. **现象**：ATK case 348（dense B=3 Hk=Hv=2 T=511 chunk=64，B*Hv=6<=AIC/2）
   报 `aclnnChunkGatedDeltaRuleFwdGetWorkspaceSize failed: 169104` /
   EZ1008 "Failed to execute tiling"，plog 无任何算子侧 OP_LOGE。
2. **根因**：稀疏 tiling 的 `SparseShape` 只认 `GetStorageShape()`。`initial_state`
   （调用方张量）与 `h/w/u/v_new/g_cumsum_bht/beta_eff`（executor `AllocTensor`
   产物）按 chunk32 已实测的教训可能以**拍平成 numel 的 rank-1 storage** 到达
   tiling（见 chunk32 tiling 注释），storage 形状校验静默返回 GRAPH_FAILED。
3. **修复**（`op_host/op_tiling/arch35/chunk_64_sparse/chunk_gated_delta_rule_fwd_sparse_tiling.cpp`）：
   - `SparseShape` 改为 storage∨origin 匹配（同 chunk32 `IsShape`）；
   - 新增 `SparseShapeTolerant`：额外接受 numel 相等的 rank-1 拍平 storage，
     用于 initial_state 与全部 executor 自有/调用方输出（逻辑形状 ACLNN 已校验）；
   - 维度来源只有 q/v 的 storage（entry 仍强制 rank-4），不受影响；
   - 所有静默 `return ge::GRAPH_FAILED` 补上带上下文的 `OP_LOGE`，
     末尾加 `OP_LOGD` 摘要；下次失败可直接从 plog 定位分支。
4. **第二轮设备运行（2026-11-03 早）**：新增日志直接命中
   `Sparse key64 requires output 9`——`GetOutputShape(9)` 为 nullptr。
   **根因**：fused 分支 l0op 把可选输出 4-8（q_hat/k_hat/q_rstd/k_rstd/beta_eff）
   传 nullptr、9-13 传真实张量；GE 会把中间的空可选输出槽**打包前移**（Prepare
   l0op 注释记录过同一陷阱：nullptr 槽会让后续输出 GM 变 0x80000000），
   h/w/u/v_new/gCumsumBht 被挤到 4-8，索引 9 之后不存在。
   **修复**（`op_host/op_api/chunk_gated_delta_rule_fwd.cpp`）：fused 调用时
   为每个空可选输出槽分配 rank-1 `[1]` dummy（q_hat/k_hat/h/w/u/v_new 用
   q 的 dtype，q_rstd/k_rstd/beta_eff/g_cumsum_bht 用 FP32），与 Prepare
   l0op 的占位做法一致；特性开启时这些槽位本来就是真实张量，kernel 不会写
   dummy。非 fused（Phase6/chunk32）调用 4-13 全空、位于尾部，不受影响。
5. **第三轮设备运行（2026-11-03 上午）**：tiling 通过、kernel 启动，输出回拷
   报 `507015`（AICORE 异常）。静态审计（mode-0/2/4 路由、迭代计数、事件
   配平、UB/L1/L0 表、GM 偏移）未发现决定性缺陷，需要设备侧证据分流。
   本轮落地两项：
   - **二分调试开关**：`sparse_pipeline.h` 支持 `-DGDN_SPARSE_DEBUG_MODE=`
     （0=全流水默认；1=只跑 NT+1 次栅栏/relay；2=只 producer；3=只
     consumer），对应 §9 第 2-4 步。重建后按 1→2→3 定位到角色，再逐级
     恢复 `RunChunk` 内的 stage。
   - **空转组协议对齐**：`RunIdlePipeline` 改为与工作组完全相同的迭代形
     （栅栏 + mode-2 relay + AIC wait）。mode-2 若为组内语义则无成本，若
     该硅片把 mode-2 当全 launch AND-reduce 则可消除一类死锁。
   - **取数指引**：重跑后抓 plog 的 `sqe_type / errorStr / subErrType /
     core id / current: 0x / mte|l1|cube|vec error info`（参照知识卡
     `mix_507015_aic_aiv_dual_side_anomaly` 的四步手法与
     `aiv_crosscoreflag_subblock_507015` 的同步契约检查表），按核型与 PC
     分流后再改。
6. **第四轮（2026-11-03 上午，507015 复现）**：对 producer/consumer 做了逐事件
   配平审计（FIX_M/M_MTE1/M_FIX 全链路：Init prime → Stage2/4/5/7 每个 MMAD 后
   Fixpipe 补回 → Finish drain，全部平衡；consumer 每个 CubeMatmul 后跟
   CubeFinish 补回，同样平衡）、Fixpipe 参数对照（`FixpipeYToL1` dstStride=m*8、
   `FixpipeAToL1` dstStride=m*16，与 chunk32/solve_tri 的 L1 NZ 约定一致；
   `isChannelSplit=false` 与已验证的 L1/UB 目的地用法一致）、UB/L1/L0 逐槽边界
   （全部恰好不越界）、GM 偏移（全部在界内）。**未找到崩溃级静态缺陷**；
   507015 根因必须靠设备证据（§9 二分或 plog）。
   **实锤一个数值 bug**：`Stage2Kkt` 的 `MatmulToL0C` 误用 `transposeB=true`，
   已验证的串行 Prepare（`Stage2_AicOne`）对同一个 k̂ k̂ᵀ 用 `transposeB=false`
   （"B NZ is [n,k]: same k' tile as A, n=m"），已改回 `false`——不修则 KKT
   拿到转置错误的 B，A/W/U/o 全错（精度门会拦，不是 507015 的成因）。
7. **P0 延迟优化落地（精度无关）**：
   - host 在 `usePipeline` 时不再序列化串行 H/O workspace（~55MiB → 0，
     kernel 在 pipeline 模式不读这些区域，aclnn workspace 随之缩小）；
   - varlen 的 `td.pipeline.maxChunks` 收紧为逐序列 `max ceil(len_s/64)`
     （host 经 `GetOptionalInputTensor` 读 cu_seqlens，ValueDepend 已声明），
     短链不再空跑栅栏轮。dense 路径与 chain 位置映射不变。
8. **未动**：`sparse_tiling.h` 纯函数契约（契约测试不变）、kernel 侧
   dispatch/链解析/串行适配（审计一致，设备行为待 §9 分步验证）。

### 6.1 数值/算法

### 6.1 数值/算法

1. **A' 必须包含对角线**：`o_i` 含 token i 自身的 delta 更新，`A'[i,j]=(j<=i)`。
   原 `APrimeVF` 用 `firstRow+i`（严格下三角），会漏掉 `q_i·k_i·v_new_i`；
   现改为 `firstRow+i+1`，与 `chunk_fwd_o` Stage3（`row+1`）和 chunk32
   `GateAttnVF`（`i+1`）一致。
2. **W/U matmul 的 k 维**：`WuMatmulToL0C(A,B,m,n,kDim)` 的 `kDim` 是 A 的列数，
   应为 `kChunk=64`（A=[64,64]、B=[64,128]）；原代码传了 `kHeadDim=128`。
   已按 Prepare Stage7 的 `bt, nK, bt` 参数顺序修正。
3. **W/U 的 Fixpipe**：W/U 是 bf16、[64,128]，Prepare 的
   `FixpipeL0cToUbFp32Nd` 只接受 float 目的且固定方阵。producer 新增
   `FixpipeWuToUb`：ROW_MAJOR + `F322BF16` + `mSize=64 / nSize=128 /
   srcStride=64 / dstStride=128`，直接落 bf16 UB。
4. **producer Stage5A 去掉 GM→L1 回读**：`FixpipeAToL1` 已把同一 L0C tile 写成 L1 A，
   原来的 `CopyGmNdToL1Nz` 与之写同一 `l1A_` 且无 FIX↔MTE2 同步；另外 3510 上
   GM→L1 连续 `DataCopy` 会占用 mode-4 flagId 1/17，正好撞 `kProdFlagKkt`。
   公开 A 的 GM 写保留。
5. **mode-4 AIV flagId 用 raw 0–15**：初版 `AivFlagId` 给 AIV1 传 `base+16`，只能靠
   硬件 4-bit 截断碰巧等价；现 AIV 一律传 raw id，AIC 侧 `AicSetFix`/`AicWaitMte1`
   用 `base`/`base+16`，与 Prepare 内核一致。

### 6.2 编译/作用域（首轮 ccec 反馈 + 全量核对）

1. **`AscendC::CFG_ROW_MAJOR_UB` 不存在**：`CFG_ROW_MAJOR_UB`/`CFG_NZ_L1` 是仓库在
   `chunk_gated_delta_rule_fwd_prepare_common.h` 的 `ChunkGatedDeltaRuleFwdPrepare`
   命名空间里自定义的；consumer/producer 改用 `Prep::CFG_ROW_MAJOR_UB` /
   `Prep::CFG_NZ_L1`。
2. **`HardwareInfo` 等重定义**：prepare 的 `_common.h` 与 coefficient_generation 的
   `arch35/mem.h` 在全局命名空间各定义一份，而 `arch35.cpp` 同时包含两者；两份定义
   统一用 `GDN_SHARED_ONCHIP_TYPES_DEFINED` 二选一（所有 `GetBuffer` 都显式传 dtype）。
3. **`GDN::ChunkFwdOA5Dispatch` no match**：根因是 `const GM_ADDR`。`GM_ADDR` 是宏
   （`__gm__ uint8_t*`），`const GM_ADDR x` 展开为 pointer-to-const，不能转成
   `GM_ADDR`；`sparse_phase_adapter.h` 中 `user`/`qCompute`/`kCompute` 改为非 const。
4. **`Prep::kGdnRcpLn2` 不存在**：`kGdnChunkSize`/`kGdnRcpLn2`/`kGdnLn2`/
   `kGdnL2NormEps`/`kGdnGateClip` 是 `prepare_common.h` 命名空间**之外**的全局常量，
   不能加 `Prep::`；`CumsumScaleVF` 的 scale 改为未限定 `kGdnRcpLn2`。
5. **`OnChipBuffer`/`BufferType` 在全局命名空间**，role 直接 unqualified 使用。
6. **`sparse_barrier.h`/`sparse_sync.h` 命名空间自洽**：它们先于 `sparse_common.h`
   被包含，函数体内未限定的 `PIPE_MTE3`/`PIPE_FIX`/`PIPE_ALL`/`PIPE_V`/`PIPE_MTE1`
   需要文件内 `using namespace AscendC;`；已补上。
7. **VF 辅助函数属性**：`SparseLoadB16AsF32`/`SparseStoreB16` 从 `__simd_callee__`
   改为 `__aicore__ inline`，与 chunk32 `_vf.h` 的 `LoadCastB16`/`StoreCastBf16`
   （在 `__VEC_SCOPE__` 里直接调用）一致。
8. **清理**：删除 `Stage1Inputs` 未使用的 `rowOff`；`ubQ_[rows*kHeadDim]`/
   `ubK_[...]` 的下标显式转 `uint32_t`（`LocalTensor::operator[]` 只接受 `uint32_t`）。
9. **系统核对结论**：26 个 `Prep::` 函数/常量全部在命名空间内；prepare 调用的参数
   个数/类型、`FixpipeParamsArch3510`/`MmadParams`/`LoadData2DParamsV2`/
   `DataCopyExtParams` 字段、tiling 结构字段均已逐项核对；本目录再无
   `const GM_ADDR`、无未使用的自动变量。

## 7. 复用的数学助手（不要重推）

`chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/`：

- `_vf.h`：`L2NormK128VF`/`CumsumScaleVF`/`GateSoftplusVF`/`BetaSigmoidVF`/
  `GateLowerLVF`/`MulKktGateVF`/`PackDiagLeavesFromUb`/`VcsIdentityVF`/
  `MulReduceScatterVF32`/`ScaleRowsVF`/`ScaleRowsBetaExp2gVF`/`Identity64VF`/
  `FillVcsIdentity`
- `_matmul.h`：`LoadL1NzToL0AB`/`MmadL0ABToL0C`/`MatmulToL0C`/`WuMatmulToL0C`
- `_datacopy.h`：`CopyGmToUbElems`/`CopyUbToGmElems`/`UbNd64ToL1Nz8`/
  `UploadBf16NdToL1`/`FixpipeL0cToUbFp32Nd`/`FixpipeL0cToGmNd`/`CopyGmNdToL1Nz`/
  `UploadDiagLeavesToL1`

命名空间 `ChunkGatedDeltaRuleFwdPrepare`，role 内用 `namespace Prep = ...`。
**注意**：`LoadL1NzToL0AB` 的 A 侧固定用 `m/16` 作 `srcStride`，不适用于
`transposeA=true` 的 bf16 源；consumer 用自己实现的 `LoadL1ToL0A/B`
（`srcStride = 源行数/16`，照 chunk32）。

## 8. 验证状态

**本机已跑（2026-11-02）**

```
python3 -I tests/atk/chunk_gated_delta_rule_fwd/sparse_entry_contract.py      # passed
g++ -std=c++17 -Wall -Wextra -Werror -I. \
    tests/atk/chunk_gated_delta_rule_fwd/sparse_tiling_contract.cpp -o /tmp/stc && /tmp/stc  # passed
python3 -I tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py         # passed
```

**未跑/未通过**：CANN 编译仍在迭代（首轮 4 类诊断已修，等待下一轮 ccec；
尚无“编译通过”证据）、ATK 精度、MSS、msprof。
设备侧入口：`bash gdn-verify.sh --mode single --op chunk_gated_delta_rule_fwd`
（或直接 `bash build.sh --pkg --soc=ascend950 --vendor_name=fla_npu
--ops=chunk_gated_delta_rule_fwd`）。

## 9. 下一步（按顺序）

1. **收完 ccec 编译诊断**：继续按编译报错修 API/类型；本目录已做过作用域、参数个数、
   字段与宏陷阱核对，剩余多为 CANN 版本差异。
2. **空流水调试**：临时注掉两个 role 的 `RunChunk`，只留 NT+1 次 `AivRoleBarrier`，
   确认 mode-0 计数、重复调用与空转组不挂死。
3. **producer-only 比对**：让 consumer 组只进栅栏，把 producer 的 W/U/g_cumsum 与
   串行 Prepare 输出逐元素比对（dense/varlen、tail、G=1..4）。
4. **consumer 比对**：恢复流水，跑 ATK key-64 用例（dense/varlen、tail、norm、exp2、
   stateVFirst、allowNegEigval、初/末状态），对照 `gdn_reference.py`。
5. **收尾**：重复执行 3 次 + MSS、`msprof op_summary` Task Duration、chunk32/Phase6
   回归；可选去掉 host 侧并行的串行 workspace。

## 10. 坑与规则

- **mode 0 只覆盖同类型核**：AIV→AIV 有效，AIV→AIC 无效；这就是"AIC 不碰 GM + 组内
  mode-2 转发"的由来。
- **mode 4 的 flagId 路由**：AIV0 Set 0–15 → AIC Wait 0–15；AIV1 Set 0–15 →
  AIC Wait 16–31；AIC Set 0–15 → AIV0 Wait 0–15；AIC Set 16–31 → AIV1 Wait 0–15。
  **AIV 永远传 raw 0–15**，只有 AIC 用 +16。
- **`CrossCoreWaitFlag` 必须显式写 pipe**；默认 `PIPE_S` 在 3510 会 race。
- **flag id 占用**（3510，MIX 1:2）：`SyncAll` mode-4 占 12/13/28/29；Matmul 高阶 API
  占 mode-4 0..N+17；TSCM 占 mode-4 0..11；GM→L1 连续 `DataCopy` 占 AIC 1/17、
  AIV 1。本实现不用这些接口，producer 的 A 输出也不再走 GM→L1 回读。
- **每 flagId 未消费计数 ≤15**；每轮栅栏前 `PipeBarrier<PIPE_ALL>` 排空。varlen
  短链的长空转段（AIV 只做栅栏+转发）需重点压测。
- **两个 AIV 都必须进每一次组内握手**（OL-212），空转也要 Set。
- **L0 事件配平**：producer AIC 每 chunk 7 个 matmul，consumer AIC 每 chunk 5 个；
  `Init` prime 1 个 `M_MTE1`/`FIX_M`，`Finish()` 各 drain 1。新增 matmul 时同步改。
- **Fixpipe 半块**：`srcStride` 单位是 C0_Size，值填**源 NZ 的整块 M**（不是半块
  mSize）；第二半从 `l0C[(fullM/2)*16]` 开始；`subBlockId` 只在 `dualDstCtl=0`
  单目标模式下选 AIV，且仅 MIX 1:2 支持。
- **LoadData2DParamsV2 单位**：mStep=16 元素，kStep=32B，srcStride/dstStride=512B；
  `srcStride` = 源行数/16（转置源用 k 行数）。
- **GM↔UB 只用 `DataCopyPad`**（`CopyGmToUbElems`/`CopyUbToGmElems`）；
  `DataCopy` 只用于 UB↔UB、UB↔L1 与对齐的 UB↔GM 状态镜像。
- **UB 预算**：248 KiB/核，regbase 目标 ≤208 KiB（consumer 173 KiB、producer
  177.5 KiB），最终以 `GetCoreMemSize(UB)` 为准。
- **varlen**：所有组循环次数必须相同（全局 `max NT`），短链/空链只进栅栏，否则死锁。
- `sparse_tiling_contract.cpp` 断言 `PrepareTilingData` 字段顺序与 Prepare 结构一致，
  改 POD 时别破坏。

## 11. 待确认

- `sparse_common.h` 的 `cu_seqlens` 前缀扫描用 `GlobalTensor::GetValue`（O(numSeq)
  标量读），仓库 `FwdHResolveSequence` 同做法；序列数极大时改成一次 DataCopy 到 UB。
- 若 ccec 对 `constexpr FixpipeConfig` 作 NTTP 或 `Prep::CFG_*` 的跨命名空间使用有额外
  限制，优先回退为 `AscendC::CFG_ROW_MAJOR`（仅 L0C→GM）或按文件内局部常量处理。

## 12. 优化清单（2026-11-03 源码评审，按优先级）

MAC 账（每 chunk 每链）：producer = KKT 0.52M + Y 2×64³ 0.52M + A 2×64³ 0.52M +
W/U 2×(64×128×64) 1.05M ≈ **2.6M**；consumer = P 1.05M + O_s 1.05M + A_raw 0.52M +
D 1.05M + O_l 0.52M ≈ **4.2M**。比例 1.6:1，consumer 是流水瓶颈级。

**P0（host 侧，零 kernel 风险）—— 已于 2026-11-03 落地，见 §6.0 第 6 条**

1. ~~按 schedule 精简 workspace~~（已做：pipeline 模式不再序列化 ~55MiB 串行区）。
2. ~~varlen maxChunks 收紧~~（已做：逐序列最大值，host 读 ValueDepend cu_seqlens）。

**P1（结构拉平，需 device 验证）**

3. **A_raw=Q@Kᵀ 与 A' 下放 producer**：producer 已有 q/k/g 与 GateRows 能力；
   移动后 4.2/2.6 (1.6×) → 3.67/3.15 (≈1.17×)，稳态提速约 14%。consumer 去掉
   K 上传之外的 A_raw/A' 链（UB 省 ~20KiB，AIC 串行 matmul 5→4），A' [64,64]bf16
   8KiB/chunk 经 GM 发布（同 W/U 的 mode-0 可见性边）。K 上传保留（D 仍要用）。
4. **每轮 2 chunk 粗化**：producer 一轮产 chunk 2i/2i+1、consumer 消费 2i-2/2i-1，
   mode-0 栅栏轮数减半；L1 H 双缓冲结构与 chunk 奇偶 parity 已兼容。

**P2（组与核利用）**

5. **blockDim 降到 2×chains**：OL-226 的同驻约束（2C ≤ AIC）仍满足；空转组消失，
   栅栏扇入更小，余核留给并发任务。当前"全核启动保持与串行路径一致"是保守选择，
   需 device 验证 mode-0 参与集变化无回归。
6. **producer AIV1 分担**：Stage1/6/8 的加载与 L2Norm/ScaleRows/drain 按 32/32 行
   半分给两个 AIV（照 consumer firstRow 模式）；#3 之后 producer AIV0 变挤时收益
   更明显。

**P3（profiling 后收尾）**

7. 角色对象里 `FusedTilingData td_{}`（584B）瘦身为实际用到的标量。
8. 各 stage 末尾 `PipeBarrier<PIPE_ALL>` 按真实依赖收窄（PIPE_V/PIPE_MTE3）。
9. producer 与 consumer 负载进一步拉平的备选：producer 组带 2 链（1.5C 组）——
   需 producer 内部双链交错，复杂度高于 #3，仅在 #3 不够时考虑。

验证顺序：先 §9 bring-up 正确性，P0 随时可合；P1/P2 逐项用
`msprof op_summary` 的 Task Duration 对比。
