# CP 路径优化交接（下一个会话从这里开始）

> 本文件是 `chunk_gated_delta_rule_fwd_h` CP（TilingKey 3）实现的性能优化交接单。
> 方法论来源：工作区 skill `ascendc` 的 `references/09-performance-optim.md`
> （五维调查 → baseline → 优化 → 精度 → 性能 → 最多 3 轮迭代）与
> `profiling-analysis`（Hostbound / Computing / Communication 判型）。
> 下一个会话**先读 §0，再按 §4 的 P0 → P1 → P2 顺序执行**。

## 0. 会话启动必读

1. 仓库根 `AGENTS.md` 与 `docs/agents/{03-方案设计,04-算子开发,05-算子测试}.md`（本仓唯一流程）。
2. 本算子：`docs/design.md`（当前 CP 设计）、`docs/validation.md`（开发期记录与 §5.5 review 修复）、
   `docs/api.md`、`README.md`。
3. 实现：`op_kernel/arch35/gemm/kernel/gdn_fwd_h_cp_kernel.hpp`（821→约 850 行）、
   `op_host/chunk_gated_delta_rule_fwd_h_tiling_processor.h`、
   `op_kernel/chunk_gated_delta_rule_fwd_h.cpp`。
4. 参考实现：`op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp`（replay 同构）、
   `fla/ops/ascendc/cp/pre_process_fwd_kernel_merged/`（h+m 双链、L0C→UB、950/A2 同步经验）。
5. 优化前先读 skill 的 `references/09-performance-optim.md` 与 `04a-kernel-api.md`。

## 1. 当前状态（交接时）

- 分支 `fwdh_cp_single_kernel`；CP 实现、host 路由、文档、CPU 单测（`tests/pta/test_cp_affine.py` 6/6）已完成。
- 内置 `code-review` + 工作区 `ascendc` skill 的审查发现 2 个 blocker（summary 布局、跨 task 竞争）
  与多项 major，**均已修复但未经编译与设备验证**。
- **2026-10-29 优化轮（P1.1 + P2.6）已落到代码**：全部 5 次 GEMM 改为 L0C→UB 直连、
  q 与逐行缩放改为寄存器基向量化、`CrossCore{Set,Wait}Flag` 的裸 id 调用改走 `AscendC::`。
  详见 `validation.md` §5.7。**同样未编译、未上板。**
- 本开发机为 macOS、无 CANN：下一个会话若仍无设备，只能做代码与离线验证；设备侧执行见 §5。
- **任何优化前必须先完成 P0**：编译、安装、md5 校验、PTA/ATK 精度、基线性能。
  当前代码未编译，存在编译错误时优先修复，不得在未编译的代码上继续叠加优化。
  注意 P1.1 已经把 P0 的基线对象改掉了：若设备上发现直连版本有回归，先用
  `git revert` 回到 GM 中转版本再取基线，不要在直连版本上直接比对。

## 2. 静态调查（五维，按 `09-performance-optim.md` 格式）

### 已发现的问题（按预期收益排序）

1. **[data copy / memory] 阶段 1 的两次 GEMM 结果都经 GM 中转** — 每 chunk 写 `Pv`（BT×256 FP32）
   与 `YT`（128×256 FP32），AIV 再读回。`YT` 往返约 256 KB/chunk，是阶段 1 GM 流量的大头。
   950 支持 L0C→UB 直连（`PackedTileCopyTlaToUB` + `CopyL0CToUBMode::NO_SPLIT`；
   `SPLIT_M` 只有 3 参 `operator()`，编不过 block 的 `sendVecNum=2` 输出路径），可省掉这次写+读及一轮 flag。
   *预期收益：阶段 1 时延的可观下降（GM 字节约减半）；风险：中（UB 双槽布局、行分配）。*
2. **[pipeline] 阶段 1 是严格串行链** — AIV 准备 W/K → GEMM-A → AIV epilogue → GEMM-B →
   AIV 更新 X → 下一 chunk。W/K 的准备与 X 无关，可在上一 chunk 的 GEMM-B/更新期间提前准备
   （Wf/Kf 双缓冲）。*预期收益：隐藏 AIV 搬运与部分 GM 延迟；风险：低-中（缓冲翻倍、flag 计数不变）。*
3. **[tiling] 阶段 2 线性扫描随 R 线性增长** — rank r 做 r 次 `128×256×128` GEMM，
   R=24 时平均 12 次。Hillis-Steele log-scan 可降到 `ceil(log2 R)` 轮（5 轮）+ 每轮一次 AIC 屏障。
   CPU 单测已验证等价（`test_log_scan_equals_linear_scan`）。*预期收益：R 大时阶段 2 明显缩短；
   风险：中（每轮屏障开销、summary 双缓冲）。*
4. **[tiling] 路由门限未按实测回填** — CP 每 chunk 计算量约为链并行的 2 倍，`B*HV` 接近核数时
   可能劣于 key1/2。*预期收益：避免性能回归；风险：低（host 一行判定）。*
5. **[pipeline] task 边界屏障阻止跨 task 流水** — 修复 review blocker 时加了 AIV 全局屏障（ID 8）。
   若 summary 改为 task 奇偶双缓冲，可去掉屏障并让 task t 的 phase 3 与 task t+1 的 phase 1 重叠。
   *预期收益：每 task 隐藏一次边界；风险：高（跨 task 竞争刚修复，需严格 A/B）。*
6. **[API] q 的标量计算与逐行 `Muls`** — `ComputeRpRv`/`Phase3Vector` 每行一次标量
   `expf` + `LoadG`（GM 标量读），再逐行 `Muls`。可预先在 UB 算 q 向量并复用。
   *预期收益：小；风险：低。*
7. **[memory] summary 双写** — 阶段 1 最后 chunk 同时写 X 槽与 summary 区（128 KB/task）。
   若 phase 2 的 Y 改用独立槽，X 槽可直接作为 summary，省这次双写。*预期收益：小；风险：低。*
8. **[memory] H0 未折入 rank0 偏置** — 每 rank 一次 `B_trans @ H0`（1×128³）。
   可折进 rank0 的 Ā（`Ā_0 ← B̄_0 H0 + Ā_0`）。*预期收益：小；风险：低。*

### 已确认无问题

- **tiling 负载均衡**：`R=min(aic, chunks)`，base/rem 偏斜 ≤1 chunk，无空 rank。
- **data copy 粒度**：所有 `DataCopy` ≥ 16 KB；scratch/summary 偏移 512 B 对齐；
  `DataCopyPad` 的 stride 单位已按 API 卡核实（UB→GM `srcStride` 单位 32 B）。
- **API usage**：FP32 GEMM 关闭 HF32；`DataCopyPad` 仅用于场景初始化；无小搬运循环。
- **memory 容量**：UB 峰值 224 KB ≤ 248 KB；L1 384 KB ≤ 512 KB；L0C 128 KB ≤ 256 KB；
  scratch 180224 float < host stride 180352。
- **pipeline（replay）**：阶段 3 的 C1/V1/C2/V2 与串行 kernel 同构，已有一层 flag 流水。

## 3. 优化路线（按阶段，每项独立验证）

> 每项都遵守 `09-performance-optim.md` 的闭环：先保存 baseline → 只改一个变量 →
> 精度通过 → 同 case 测性能 → 未达标保留真实结果并记录。最多 3 轮。

### P0（前置，不做完不开始优化）

1. 编译安装：`bash build.sh --soc=ascend950 --pkg --vendor_name=fla_npu --ops=chunk_gated_delta_rule_fwd_h`
   → `./build_out/fla_npu_linux-*.run --install`
   → `python scripts/verify_libcust_opapi_md5.py --run-package <run包>` 全 `[OK]`。
2. S1 哨兵：R=2/4/8/24、C=5 与 C=2R，含 `B*HV>1`，确认无死锁、h/v_new/final_state 正确。
3. 精度：PTA `test_fwd_h.py` + `data_compare_h.py`（1e-4 / 99.99% / max<0.1）；
   ATK accuracy/determinism/mssanitizer；key1/2 回归。
4. 基线：固定 `perf_cases.jsonl`（至少覆盖 `B*HV ∈ {1,2,4,8,12}` ×
   `T ∈ {8192,32768}` × chunk32/64），用 `msprof op_summary` 的 Task Duration，
   warmup 20 / 采样 100 / 中位数；保存为 baseline。**baseline 未保存前不做任何优化。**

### P1（收益最大，按顺序）

1. ~~**L0C→UB 直连**（问题 1）~~ — **已实现，待设备验证**（2026-10-29）。
   阶段 1 的 GEMM-A/GEMM-B、阶段 2 的前缀 GEMM、阶段 3 的 C1/C2 全部改为
   `Common::BlockMmadTla` + `PackedTileCopyTlaToUB(NO_SPLIT)`；新增 `UbFree`
   握手（ID 11，AIV 用 `PIPE_V` 置位、mmad 内用 `PIPE_FIX` 等待）。
   **上板第一件事**：确认 ID 11 在 mode 0x4 下可用（0..7 已被协议占用，8/9/10 是
   Catlass 屏障，12/13/14 是 `SyncAll`），以及 `NO_SPLIT` + `subBlockId` 的 `ceil(m/2)` 行分配与
   AIV 的 `RowsPerSubBlock` 一致（`tokens` 为奇数时要重点看）。
2. **W/K 预取双缓冲**（问题 2）：`Wf/Kf` 扩为两份，AIV 在处理当前 chunk 的 epilogue 时
   准备下一 chunk；flag 计数保持不变（仍是每 chunk 一次 Operand）。
   验收：精度全过 + 阶段 1 时延下降；若无收益则回退（保留真实数据）。
3. **阶段 2 log-scan**（问题 3）：实现 `ceil(log2 R)` 轮 Hillis-Steele（summary ping-pong），
   每轮 AIC 全局屏障（ID 9 复用或新增约定），排他前缀取 `inclusive[r-1]`。
   验收：与线性扫描同 case 精度一致（CPU 已有等价测试）+ R=24 时阶段 2 缩短。
4. **路由门限回填**（问题 4）：依据 P0 基线计算 break-even，`useCp` 增加
   `batchHeads <= aicCoreNum / K`（K 由实测决定）或按 `B*HV` 分档；保留 `GDN_FWD_H_CP`。
   验收：所有现有 case 不回归；`B*HV` 大的 case 回落 key1/2。

### P2（后续，逐项 A/B）

5. **task 间双缓冲**（问题 5）：summary 按 task 奇偶双缓冲，去掉 task 末尾 AIV 屏障，
   让 phase 3 尾部与下一 task phase 1 重叠。**风险最高**：必须单独一轮、单独 A/B
   （同 case key3 新旧对比 + determinism），失败立即回退。
6. **q 向量化**（问题 6）：g/q 预载 UB、按行广播；`ComputeRpRv`/`Phase3Vector` 去掉逐行标量读。
7. **summary 单写**（问题 7）与 **H0 折入**（问题 8）：小改动，合并为一轮验证。

## 4. 每轮的验证协议（`09-performance-optim.md` Stage 4/5）

```
改前：保存 <case>_baseline_report（Task Duration、profiling 原始目录）
改后：build + install + md5 → PTA → ATK accuracy/determinism/mssanitizer
      → msprof（与 baseline 完全相同的 cases）→ 三方对比（baseline/optimized/reference）
性能数据判型：用 profiling-analysis skill（step_trace_time.csv 等）
      Hostbound（空闲>20%）→ 查下发/标量；Computing（>85%）→ 查 GEMM/向量占比；
      Communication（>10%）→ CP 无跨核通信，出现即说明屏障/flag 开销异常。
决策：提升 → 保留；不变/回退 → revert 并记录；最多 3 轮，结束后写 <op>_optim_summary.md。
```

## 5. 约束与红线

- 公开接口、支持范围、CPU 标杆保持稳定（`docs/agents/03` 的 R20/R21）。
- 不得改动 key1/2 与 arch22 实现；`TilingKey 3` 的 workspace 布局改动必须同步 host 的
  `cpSummaryStride/cpScratchStride` 与 kernel 的 `SCR_*`，并加注释。
- flag 资源：普通 0..7；8/10 为 Catlass AIV 屏障；9 为 AIC 屏障；**11 为 L0C→UB 直连槽
  （本轮新增，0x4 模式）**；12/13/14 为 `SyncAll`；15 尚未使用；
  mode 4 必须显式 `modeId=4`；单 ID 未等待 set 深度 ≤15。
- 数值契约：summary/前缀/重放状态 FP32；h/v_new 按输入 dtype 输出；v_update 低精度进 C2；
  `final_state` 为 float 时写 FP32。优化不得改变这些点（否则重跑 S0 单测与精度验收）。
- 每轮只改一个可独立验证的变量；发现 Stage/数据流/地址/同步变化时回到 `docs/agents/03`。

## 6. 关键文件与定位

| 内容 | 位置 |
| --- | --- |
| 阶段 1/2/3、flag、UB | `op_kernel/arch35/gemm/kernel/gdn_fwd_h_cp_kernel.hpp`（ProcessCube :302、Phase1Cube :319、Phase2Cube :362、Phase3Cube :384、Phase1Vector :496、UpdateRollingState :586、Phase2Vector :628、Phase3Vector :723、UpdateStateAndH :787） |
| 路由/workspace | `op_host/chunk_gated_delta_rule_fwd_h_tiling_processor.h:129-175` |
| key3 分派 | `op_kernel/chunk_gated_delta_rule_fwd_h.cpp:131-170,190-200` |
| 参考 replay / L0C→UB | `arch35/gemm/kernel/gdn_fwd_h_kernel.hpp`、`kernel_utils/tile/copy_l0c_to_ub.hpp`、`cp/pre_process_fwd_kernel_merged/op_kernel/arch35/*_cube.h` |
| CPU 数学与误差模型 | `tests/pta/test_cp_affine.py` |
| 开发期记录 | `docs/validation.md`（§3 设备清单、§5.5 review 修复、§5.6 skill 记录） |

## 7. 已知风险与回退

- `B*HV` 接近核数时 CP 可能慢于链并行 → `GDN_FWD_H_CP=0` 立即回退；按 P1.4 回填门限。
- 同步协议（0x4 配对、0x0 屏障）尚未在设备上验证 → P0.2 哨兵先行；失败回退
  `SyncAll<true>() + PipeBarrier<PIPE_ALL>`。
- workspace 比 key1/2 大（CP 约 +20 MiB @ R=24；serial 因移除死分配减少 6–12 MiB）。
- 若优化触及跨 task 复用，必须重跑 determinism 与 `B*HV>1` 的多 task 用例。
