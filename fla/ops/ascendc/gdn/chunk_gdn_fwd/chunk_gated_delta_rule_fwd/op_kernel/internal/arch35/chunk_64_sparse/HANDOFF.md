# chunk64 sparse（key 64）producer/consumer 实现交接

> 面向下一次会话。先读本文，再看 `gdn_fwd.md`、`../../../../../docs/design.md`、
> 计划文件 `~/.claude/plans/crystalline-tickling-bentley.md`。

## 1. 目标

`gdn_fwd.md` 的规格，针对 `B*Hv <= AIC/2`、`chunk_size=64`、`K=V=128`、BF16 输入 FP32 计算：

```
producer: L2Norm -> cumsum -> KK.T -> solve_tri -> recompute_wu -> W,U, g_sum
consumer: W, U -> fwdh -> v_new, h -> fwdo
```

现状（`sparse_phase_adapter.h:118-142`）只是把独立 Prepare → H → O 串起来，中间两次
`PhaseRendezvous()`，**没有重叠**。目标是把 key 64 变成 `schedule=1` 的真流水。

## 2. 已锁定的决策（不要再改）

1. **每链 2 个 MIX 组**：组 `c` = producer，组 `C+c` = consumer，`C = B*Hv`（varlen 取上界
   `(cu_seqlen-1)*Hv`）。**`blockDim` 保持 `GetCoreNumAic()`**，前 `2C` 组担角色，其余组空转。
2. **跨 role 同步只用 AIV + mode 0x0**：`AivRoleBarrier()`。AIC 不进 mode-0 栅栏；
   迭代边界由同组 AIV 用 `AivRelayToAic()`（mode 0x2, PIPE_MTE3）转发，AIC 用
   `AicWaitRelayMte1()`（PIPE_MTE1）接收。
3. **AIC 不碰 GM**：所有 GM 读写归 AIV；AIC 只读本组 AIV 写好的 L1、把结果 Fixpipe 到 UB。
   这条是"mode 0 只覆盖同类型核"的直接推论，别退回去让 AIC 读 GM。
4. **在 `chunk_64_sparse/` 重写编排/内存/同步**，数学助手复用已验证实现（见 §5）。
5. **consumer 切分**：chunk 行 32/32（AIV0 行 0-31，AIV1 32-63），state 按 K 64/64；
   O 段（A_raw/gateA/A'/O_s/O_l/o）也按行半落在两个 AIV（见 §5 第 2 条的修正：
   上一版"整块放 AIV0"与已完成的 UB 表/VF 接口冲突，本轮按半块实现）。
6. **cube 约定照 chunk32**（`chunk_gated_delta_rule_fwd_32_matmul.h`）：
   - `transposeA=true` = 源按 `[k,m]` 存；`transposeB=true` = 源按 `[k,n]` 存。
   - 半块下发**不用 `dualDstCtl`**，每个半块单独发一次 Fixpipe，`subBlockId` 选 AIV，
     L0C 源按 `mOffset*16` 偏移（NZ m-frac 步长 16×16）。见 `FixpipeHalfBf16`。

五条 matmul（`m,n,k` 与转置）：

| 结果 | A | B | m,n,k |
|---|---|---|---|
| `P = W@H` | W `[64,128]` | H `[128,128]` transposeB=false | 64,128,128 |
| `A_raw = Q@Kᵀ` | Q `[64,128]` | K `[64,128]` transposeB=true | 64,64,128 |
| `O_s = Q@H` | Q `[64,128]` | **复用 L0B 的 H** | 64,128,128 |
| `D = Kᵀ@right` | K `[64,128]` transposeA=true | right `[64,128]` | 128,128,64 |
| `O_l = A'@V_new` | A' `[64,64]` | V_new `[64,128]` | 64,128,64 |

## 3. 文件清单与状态

| 文件 | 状态 |
|---|---|
| `sparse_fused_tiling.h` | ✅ 已接线。`PipelineTilingData`（schedule/barrierFlagId/chainCount/maxChunks/producer•consumerGroups/workspace），`kSparseScheduleSerial/Pipeline`、`kSparseBarrierFlagId=7` |
| `op_host/.../chunk_64_sparse/chunk_gated_delta_rule_fwd_sparse_tiling.{h,cpp}` | ✅ 已接线。`BuildChunkGatedDeltaRuleFwdPipelinePlan` 接入选定条件；POD 填充；pipeline scratch 追加在串行布局之后 |
| `chunk_gated_delta_rule_fwd_arch35.cpp:487-500` | ✅ 按 `pipeline.schedule` 分派 |
| `sparse_pipeline.h` | ✅ `RunSparsePipeline` 已实现 NT+1 轮 producer/consumer 循环；串行回退仅用于 schedule 异常 |
| `sparse_common.h` | ✅ `ChainLoc`/`ResolveChain`/`ResolveWindow`/GM 偏移 |
| `sparse_memory.h` | ✅ producer UB 177.5K/L1 136K；consumer UB 173K/L1 152K；`static_assert` 本机 g++ 通过 |
| `sparse_barrier.h` | ✅ `AivRoleBarrier` / `AivRelayToAic` / `AicWaitRelayMte1` / mode-4 子块握手 |
| `sparse_producer.h` | ✅ 532 行，Stage0–8；补 `ChunkCount()`；修 `OnChipBuffer` 命名空间 |
| `sparse_consumer_vf.h` | ✅ 全部向量单元；修 A' 对角线（`j<=i`）、补 `LastGateVF`/`StateMirrorVF`、`ChunkUploadL1` 加半带偏移 |
| `sparse_consumer.h` | ✅ 新写。GM/UB/L1/L0 绑定 + 每 chunk 的 H→O 全流程（P/QKᵀ/Q@H/D/O_l 五个 cube，半块 Fixpipe） |
| `sparse_math.h` / `sparse_sync.h` / `sparse_phase_adapter.h` | 串行回退用，保留 |

## 4. 下一步（按顺序）

1. **device 首次编译**（必须，代码至今没进过 ccec）：`sparse_pipeline.h` 现在 include
   producer/consumer 两个 role；先在 950 上按 `bash gdn-verify.sh --mode single --op
   chunk_gated_delta_rule_fwd` 跑通编译，逐条修 CANN API/编译器诊断。
2. **空流水调试**（可选）：把 `RunProducerPipeline`/`RunConsumerPipeline` 的
   `RunChunk` 临时注掉，只留 NT+1 次 `AivRoleBarrier`，确认 mode-0 计数与重复调用不挂死。
3. **producer-only 比对**：临时让 consumer 组只进栅栏，把 producer 的 W/U/g_cumsum 与
   现串行 Prepare 输出逐元素比对（dense/varlen、tail、G=1..4）。
4. **consumer 比对**：恢复流水，跑 ATK key-64 用例（dense/varlen、tail、norm、exp2、
   stateVFirst、allowNegEigval、初/末状态），对照 `gdn_reference.py`。
5. **收尾**：MSS/重复 3 次、`msprof op_summary` Task Duration、chunk32/Phase6 回归；
   去掉 host 侧并行分配的串行 workspace（可选）。

## 5. 本轮实现的关键修正（相对上一版）

1. **A' 必须包含对角线**：`o_i` 含 token i 自身的 delta 更新，`A'[i,j]=(j<=i)`。
   `sparse_consumer_vf.h` 原实现用 `firstRow+i`（严格下三角），会漏掉
   `q_i·k_i·v_new_i`；现改为 `firstRow+i+1`，与 `chunk_fwd_o` Stage3
   (`row+1`) 和 chunk32 `GateAttnVF` (`i+1`) 一致。
2. **consumer 的 O 段按行 32/32 对半**（AIV0 行 0-31，AIV1 32-63）实现，而不是
   全部压到 AIV0。原因：`sparse_memory.h` 的 UB 表与 `sparse_consumer_vf.h` 的
   `firstRow/rows` 接口都是按行半设计的；整块放 AIV0 需要把 `OS/OL/ARaw/GateA/A'`
   全部翻倍，超 208K 目标。O 的每个 tensor 仍整块存在于 L1（两个 AIV 各传一半）。
3. **consumer 的 cube 装载器不用 `Prep::LoadL1NzToL0AB` 的 A 侧**：它会用
   `srcStride=mFracs`，在 `transposeA=true`（源按 `[k,m]` 存）时本该是
   `kFracs`。`sparse_consumer.h` 内的 `LoadL1ToL0A/B` 照 chunk32 的
   `LoadL1ToL0A/B` 实现（`srcStride = 源行数/16`）。
4. **`OnChipBuffer`/`BufferType` 在全局命名空间**（`chunk_gated_delta_rule_fwd_prepare_common.h`），
   不在 `ChunkGatedDeltaRuleFwdPrepare` 里；role 直接 unqualified 使用。
5. **consumer UB 增加 256 B 标量槽**（`kConsUbScalar`）存 gLast/alpha；
   `LastGateVF` 一次算好，避免 MTE2→S 的标量同步。
6. **mode-4 AIV flagId 统一用 raw 0-15**（知识卡复核后修正）：初版 `AivFlagId` 给
   AIV1 传 `base+16`，只能靠硬件 4-bit 截断碰巧等价；现已改成 AIV 一律传 raw id，
   AIC 侧 `AicSetFix`/`AicWaitMte1` 仍用 `base`/`base+16`，与 Prepare 内核一致。
7. **producer Stage5A 去掉 GM→L1 回读**：`FixpipeAToL1` 已把同一 L0C tile 写成 L1 A，
   原来的 `CopyGmNdToL1Nz` 与之写同一 `l1A_` 且无 FIX↔MTE2 同步；另外 3510 上
   GM→L1 连续 `DataCopy` 会占用 mode-4 flagId 1/17，正好撞 `kProdFlagKkt`。

## 5b. 首轮真机编译反馈的修正（2026-11-02）

1. **`AscendC::CFG_ROW_MAJOR_UB` 不存在**：该常量是仓库在
   `chunk_gated_delta_rule_fwd_prepare_common.h` 的 `ChunkGatedDeltaRuleFwdPrepare`
   命名空间里自定义的（`CFG_ROW_MAJOR_UB`/`CFG_NZ_L1`），不是 AscendC 符号。
   consumer/producer 一律改用 `Prep::CFG_ROW_MAJOR_UB` / `Prep::CFG_NZ_L1`。
2. **`HardwareInfo`/`BufferType`/`OnChipBuffer` 重定义**：prepare 的 `_common.h` 与
   coefficient_generation 的 `arch35/mem.h` 在全局命名空间各定义了一份，而
   `arch35.cpp` 同时包含两者。两份定义在 `GDN_SHARED_ONCHIP_TYPES_DEFINED` 宏下
   二选一（所有 `GetBuffer` 调用都显式传 dtype，默认模板实参差异不生效）。
3. **`GDN::ChunkFwdOA5Dispatch` no match**：根因是 `const GM_ADDR`。`GM_ADDR` 是宏
   （`__gm__ uint8_t*`），`const GM_ADDR x` 展开成 `const __gm__ uint8_t* x`
   （pointer-to-const），不能转成 `GM_ADDR` 形参；改为 `GM_ADDR x`。
4. **`FixpipeL0cToUbFp32Nd` bf16→float**：W/U 是 bf16 且是 [64,128]，该 helper 只接受
   float 目的且固定方阵。producer 新增 `FixpipeWuToUb`：ROW_MAJOR + F322BF16 +
   `mSize=64/nSize=128/srcStride=64/dstStride=128`，直接落 bf16 UB。
5. **W/U matmul 的 k 维写错**：`WuMatmulToL0C(A,B,m,n,kDim)` 的 kDim 是 A 的列数，
   应为 `kChunk=64`（A=[64,64]、B=[64,128]）；原代码传了 `kHeadDim=128`。已按
   Prepare Stage7 的 `bt, nK, bt` 参数顺序修正。
6. **`Prep::kGdnRcpLn2` 不存在**：`kGdnChunkSize`/`kGdnRcpLn2`/`kGdnLn2`/
   `kGdnL2NormEps`/`kGdnGateClip` 是 `prepare_common.h` 里**命名空间之外**的全局常量，
   不能加 `Prep::`；已改为未限定名（`useExp2_ ? kGdnRcpLn2 : 1.0f`）。
   同一类问题还有全局的 `HardwareInfo`/`BufferType`/`OnChipBuffer`（已不用 `Prep::`）。
7. **`sparse_barrier.h`/`sparse_sync.h` 命名空间自洽**：它们先于 `sparse_common.h` 被
   包含，函数体里未限定的 `PIPE_MTE3`/`PIPE_FIX`/`PIPE_ALL` 等需要在文件内引入
   `using namespace AscendC;`，否则单独包含时找不到；已补上。
8. **VF 辅助函数的链接属性**：`SparseLoadB16AsF32`/`SparseStoreB16` 从
   `__simd_callee__` 改为 `__aicore__ inline`，与 chunk32 `_vf.h` 的
   `LoadCastB16`/`StoreCastBf16`（在 `__VEC_SCOPE__` 里直接调用）一致。
9. **清理**：删除 producer `Stage1Inputs` 的未使用局部 `rowOff`；两个
   `ubQ_[rows * kHeadDim]`/`ubK_[...]` 的 LocalTensor 下标显式转 `uint32_t`
   （`LocalTensor::operator[]` 只接受 `uint32_t`）。

## 6. 复用的数学助手（不要重推）

`chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/`：
`_vf.h`（`L2NormK128VF`/`CumsumScaleVF`/`GateSoftplusVF`/`BetaSigmoidVF`/`GateLowerLVF`/
`MulKktGateVF`/`PackDiagLeavesFromUb`/`VcsIdentityVF`/`MulReduceScatterVF32`/
`ScaleRowsVF`/`ScaleRowsBetaExp2gVF`/`Identity64VF`/`FillVcsIdentity`）、
`_matmul.h`（`LoadL1NzToL0AB`/`MmadL0ABToL0C`/`MatmulToL0C`/`WuMatmulToL0C`）、
`_datacopy.h`（`CopyGmToUbElems`/`CopyUbToGmElems`/`UbNd64ToL1Nz8`/`UploadBf16NdToL1`/
`FixpipeL0cToUbFp32Nd`/`FixpipeL0cToGmNd`/`CopyGmNdToL1Nz`/`UploadDiagLeavesToL1`）。
命名空间 `ChunkGatedDeltaRuleFwdPrepare`，用 `namespace Prep = ...` 别名。

## 7. 已验证 / 未验证

**可跑（本机，本轮已跑）**
```
python3 -I tests/atk/chunk_gated_delta_rule_fwd/sparse_entry_contract.py      # passed
g++ -std=c++17 -Wall -Wextra -Werror -I. \
    tests/atk/chunk_gated_delta_rule_fwd/sparse_tiling_contract.cpp -o /tmp/stc && /tmp/stc  # passed
python3 -I tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py         # passed
```
**未跑**：CANN 编译、ATK 精度、MSS、msprof。设备侧：
`bash gdn-verify.sh --mode single --op chunk_gated_delta_rule_fwd`。

## 8. 坑与规则

- **mode 0 只覆盖同类型核**（`CrossCoreSetFlag` 知识卡）：AIV→AIV 有效，AIV→AIC 无效；
  这正是"AIC 不碰 GM + 组内 mode-2 转发"的由来。
- **mode 4 的 flagId 由发起核决定**（`CrossCoreWaitFlag` 卡）：
  AIV0 Set 0-15 → AIC Wait 0-15；AIV1 Set 0-15 → AIC Wait 16-31；
  AIC Set 0-15 → AIV0 Wait 0-15；AIC Set 16-31 → AIV1 Wait 0-15。
  所以 **AIV 永远传 raw 0-15**（`AivSetMte3`/`AivWaitV` 不带 +16），只有 AIC 用 +16。
  本轮修掉了初版 `AivFlagId(base+16*sub)` 依赖 4-bit 截断的写法。
- **`CrossCoreWaitFlag` 必须显式写 pipe**；默认 `PIPE_S` 在 3510 会 race。
- **flag id**：mode-0 用 7；mode-2 迭代转发用 8；mode-4 stage 用 0–6 与 9（AIV 本地 0-15，
  AIC 0-31）。避开 `SyncAll`（MIX 1:2 下 mode-4 占 12/13/28/29）、Matmul 高阶 API
  （mode-4 占 0..N+17）、TSCM（mode-4 占 0..11）、GM→L1 连续 `DataCopy`
  （mode-4 占 AIC 1/17、AIV 1）。本实现不用这些接口；producer 的 A 输出已去掉
  GM→L1 回读，只保留 fixpipe 直写 GM。
- **每个 flagId 未消费计数 ≤15**；每轮栅栏前用 `PipeBarrier<PIPE_ALL>` 排空。
  流水空转段（varlen 短链）AIV 只做栅栏+迭代转发，AIC 只消费转发；理论上 AIV 领先
  超过 15 轮会撞计数上限，需在 MSS/长空转用例里重点压测。
- **两个 AIV 都必须进每一次组内握手**（OL-212），空转也要 Set。
- **L0 事件配平**：producer AIC 每 chunk 7 个 matmul，consumer AIC 每 chunk 5 个；
  Init prime 1 个 `M_MTE1`/`FIX_M`，`Finish()` 各 drain 1。新增 matmul 时同步改。
- **Fixpipe 半块下发**：`srcStride` 单位是 C0_Size（16*sizeof(T) 字节），其值填
  **源 NZ 的整块 M**（不是半块 mSize）；第二半从 `l0C[(fullM/2)*16]` 元素偏移开始，
  与 chunk32 `FixpipeHalfBf16` 的 `16*16` 一致。`subBlockId` 只在 `dualDstCtl=0`
  单目标模式下选 AIV，且仅 MIX 1:2 支持。
- **LoadData2DParamsV2 单位**：mStep=16 元素，kStep=32B，srcStride/dstStride=512B；
  `srcStride` = 源行数/16（转置源用 k 行数），所以 consumer 不能复用
  `Prep::LoadL1NzToL0AB` 的 A 侧（它固定用 m/16）。
- **tail 处理**：两个 AIV 各用 32 行 UB 带，先把无效行清零再上传整条 32 行带，
  所以 L1 的 Q/K/W/V_new/right/A' 镜像始终完整定义；cube 维数保持 64/128，
  不依赖 L1 残留、也不需要额外清零。`state` 永远整块上传。
- **GM↔UB 只用 `DataCopyPad`**（复用 `CopyGmToUbElems`/`CopyUbToGmElems`）；
  `DataCopy` 只用于 UB↔UB、UB↔L1 与对齐的 UB↔GM 状态镜像。
- **UB 预算**：248 KiB/核，regbase 目标是 ≤208 KiB（consumer 现为 173 KiB，
  producer 177.5 KiB），最终以 `GetCoreMemSize(UB)` 为准。
- **varlen**：所有组循环次数必须相同（取全局 `max NT`），短链/空链只进栅栏，否则死锁。
- `sparse_tiling_contract.cpp` 断言 `PrepareTilingData` 字段顺序与 Prepare 结构一致，
  改 POD 时别破坏。
- **producer 的 `A` 公开输出仍由 AIC fixpipe 直写 GM**（Stage5A），这是决策 3 的已知例外：
  它不在 producer→consumer 可见性边上（consumer 不读 A），但后续若要严格"GM 只走 AIV"
  需把它改成 fixpipe 到 AIV0 UB 再落盘。

## 9. 待确认

- `sparse_common.h` 的 `cu_seqlens` 前缀扫描用 `GlobalTensor::GetValue`（O(numSeq) 标量读），
  仓库现有 `FwdHResolveSequence` 同做法；序列数极大时要改成一次 DataCopy 到 UB。
- 已知性能不均：每 chunk producer ≈2.1 M MAC、consumer ≈4.2 M MAC。跑通后可选拉平：
  把 `Q@Kᵀ`+`A'` 下放 producer，或 producer 一次带 2 条链（1.5C 组）。
- host 目前同时序列化串行布局与 pipeline scratch（多占 ~32 MiB）；流水验证通过后可去掉串行区。
- varlen 的 `maxChunks` 取的是全局 chunk 总数（host 看不到 cu_seqlens 值），
  短链场景会多跑很多空栅栏轮；跑通后可按 chunkIndices/cu_seqlens 优化。
