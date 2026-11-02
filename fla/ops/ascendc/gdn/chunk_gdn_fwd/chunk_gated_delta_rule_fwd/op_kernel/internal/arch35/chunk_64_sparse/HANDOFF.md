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
- 已知性能不均：每 chunk producer ≈2.1 M MAC、consumer ≈4.2 M MAC。跑通后可选拉平：
  把 `Q@Kᵀ`+`A'` 下放 producer，或 producer 一次带 2 条链（1.5C 组）。
- host 目前同时序列化串行布局与 pipeline scratch（多占 ~32 MiB）；流水验证通过后
  可去掉串行区。
- varlen 的 `maxChunks` 取全局 chunk 总数（host 看不到 cu_seqlens 值），短链场景会多跑
  空栅栏轮；跑通后可按 chunkIndices/cu_seqlens 优化。
- 若 ccec 对 `constexpr FixpipeConfig` 作 NTTP 或 `Prep::CFG_*` 的跨命名空间使用有额外
  限制，优先回退为 `AscendC::CFG_ROW_MAJOR`（仅 L0C→GM）或按文件内局部常量处理。
