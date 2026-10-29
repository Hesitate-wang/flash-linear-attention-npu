# ChunkGatedDeltaRuleFwdH CP Path — Development Validation

开发期记录（04 Stage S1–S6）。设计见 [design.md](design.md)。
性能优化路线与下一会话入口见 [optimization-handoff.md](optimization-handoff.md)。

## 1. 当前状态（2026-10-29）

| Stage | 状态 | 证据 |
| --- | --- | --- |
| S0 CPU 数学 | **完成** | `tests/pta/test_cp_affine.py` 6/6 通过（见 §2） |
| S1 host tiling + key3 + kernel 骨架 | 代码完成 | `tiling_processor.h` / `tiling.cpp` / `gdn_fwd_h_cp_kernel.hpp`；设备哨兵实验待执行 |
| S2 Stage 1 局部摘要 | 代码完成 | 同上；summary 与 CPU 对比待设备执行 |
| S3 Stage 2 前缀 | 代码完成 | 同上；`S_in` 对比待设备执行 |
| S4 Stage 3 重放 | 代码完成 | 同上；端到端 PTA 待设备执行 |
| S5 chunk64/128、initial/final state | 代码完成 | 路由支持；ATK 用例待补 |
| S6 性能与优化 | 未开始 | H0 折入、双缓冲、门限回填、log-scan 备选 |

**未完成设备验证前，不得声称精度或性能达标。**

## 2. 离线验证（本机，无 NPU）

`python3 tests/pta/test_cp_affine.py`（numpy，无 torch 依赖）：

| 用例 | 结论 |
| --- | --- |
| `test_rank_partition_has_no_empty_rank` | `R=min(core,C)` 对 (2,2)/(5,3)/(5,24)/(24,24)/(1024,24) 均无空 rank |
| `test_left_projection_matches_explicit_affine` | 左乘投影与显式 `B_c` 在 FP64 下逐元素一致（chunk32/64/128） |
| `test_rank_summaries_match_serial` | 段 summary/前缀/重放终值与串行递推一致（R=1/2/3/5） |
| `test_zero_initial_state` | 零初值一致 |
| `test_fp32_error_model` | 见下 |
| `test_log_scan_equals_linear_scan` | Hillis-Steele 与线性扫描等价（S6 备选） |

FP32 误差模型（32 chunk、64 维、8 rank；DUT 与 FP64 golden 均按 bf16 或 fp32
输出，按 `data_compare_h.py` 的 `|a-b|<1e-4 取绝对差，否则取相对差` 统计）：

| 输出口径 | 最坏 max rel | 通过率（<1e-4） |
| --- | --- | --- |
| bf16 输出（h[31]） | 6.1e-5 | 100% |
| fp32 输出 | ~1e-8 | 100% |

结论：在模拟的 FP32 前缀复合下，PTA 判据（99.99% 元素 ≤1e-4、max < 0.1）有余量；
真实设备仍需 `mixed_tolerance_bm` 验收。

## 3. 设备验证清单（远端执行）

```bash
# 构建 + 安装 + 校验（必须全 [OK] 才能测）
bash build.sh --soc=ascend950 --pkg --vendor_name=fla_npu \
  --ops=chunk_gated_delta_rule_fwd_h
./build_out/fla_npu_linux-*.run --install
python scripts/verify_libcust_opapi_md5.py --run-package <run包>

# S1 哨兵：R=2/4/8/24，C=5 与 C=2R，覆盖偏斜切分与跨 task 复用；
# 校验全部 task 的输出 h/v_new 与最终状态，无死锁/挂死。

# PTA 精度（chunk32，dense，K=V=128，initial/final 组合）
cd fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests/pta
python3 test_fwd_h.py <B> <T> <HK> <HV> 128 128 0 32 <init> <final> bf16 ... 
python3 data_compare_h.py bf16     # 判据 1e-4 / 99.99% / max<0.1
python3 test_cp_affine.py          # S0 复核

# ATK 正式验收（新增 CP 用例见 §4）
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -scope=mssanitizer

# 性能（msprof op_summary Task Duration(us)，warmup 20 / 采样 100 / 中位数）
# 同 shape 下 GDN_FWD_H_CP=0（key1/2）与默认（key3）各 3 轮
```

判据：
- PTA：h/v_new/final_state 全部通过 1e-4 / 99.99% / max<0.1。
- ATK：新增 CP 用例全过，既有 key1/2 用例 0 回归。
- determinism：5/5 一致；mssanitizer 无 error。
- 性能：记录每个 `B*HV` 的 key1 vs key3 的 Task Duration，标出 break-even；
  `B*HV` 接近核数时若 key3 更慢，保留 `GDN_FWD_H_CP=0` 作为默认回退的候选。

## 4. ATK 用例补充

`tests/atk/chunk_gated_delta_rule_fwd_h/` 需要新增（当前只有 chunk64/128、
dense、g-only、无 initial/final）：

- chunk32，dense，K=V=128，`B*HV ∈ {1,2,4,8,12}`；
- `T ∈ {2048, 8192, 32768}`，含 `T % chunk != 0` 的尾块；
- `use_initial_state × store_final_state` 四种组合；
- bf16/fp16 × state dtype fp32/bf16；
- 回退用例：`B*HV > 核数`、`useGk`、varlen、`GDN_FWD_H_VSPLITS>1` 应走 key1/2
  且结果正确；
- `GDN_FWD_H_CP=0` 与默认的 A/B 结果一致（同 build）。

## 5. 设计到代码追踪

| 设计项 | 代码位置 |
| --- | --- |
| CP 路由、workspace 公式 | `op_host/chunk_gated_delta_rule_fwd_h_tiling_processor.h:129-170` |
| TilingKey 3、字段下发 | `op_host/chunk_gated_delta_rule_fwd_h_tiling.cpp:52-55,172-210` |
| tiling 字段 | `op_host/chunk_gated_delta_rule_fwd_h_tiling.h:49-54`、`op_kernel/chunk_gated_delta_rule_fwd_h_struct.h:53-58` |
| key3 分派 | `op_kernel/chunk_gated_delta_rule_fwd_h.cpp:16-22,131-175,190-200` |
| Stage 1/2/3、同步、UB | `op_kernel/arch35/gemm/kernel/gdn_fwd_h_cp_kernel.hpp` |
| CPU 数学单测 | `tests/pta/test_cp_affine.py` |

## 5.5 Review 修复记录（2026-10-29）

| # | 级别 | 问题 | 修复 |
| --- | --- | --- | --- |
| 1 | blocker | summary 生产者按 `[M|S]` 行交错写（行步长 256），消费者按两块连续 128×128（行步长 128）读，rank>0 的前缀必错 | 生产者改用 `DataCopyPad` 分半写入 B/A 两个连续块（`cp_kernel.hpp` UpdateRollingState） |
| 2 | blocker | task 之间无跨 rank 同步，快 rank 的 AIV 可在下一 task 覆盖 summary | `ProcessVector` 的 task 末尾加 AIV 全局屏障 `CrossCoreBarrier<0x0, PIPE_MTE3>`（ID 8） |
| 3 | major | S_in 发布时 subBlock1 读全部 128 行，可能与 subBlock0 的最后一轮写竞争 | 每个子块只发布自己的半区，子块屏障后再写 h |
| 4 | major | `STATE_TYPE==float` 时同 dtype Cast 可能编不过 | `if constexpr` 分支直接 DataCopy |
| 10 | minor | `GDN_FWD_H_CP` 只精确匹配 "0" | `strtol` 解析，数值 0 即关闭 |
| 11 | major | CPU 测试阈值 0.999 宽于 PTA 的 99.99% | 改为 0.9999 |
| 12 | nit | `Phase3Vector` 死变量 `q` | 删除 |
| 13 | minor | `K=V=128` 硬编码无断言 | 加 `static_assert` |
| 14 | minor | design.md scratch 单位错误（176 KiB vs 704 KiB） | 修正为 704 KiB（180224 float） |
| 15 | major | api.md 缺 NT-first h 布局与 `chunks>=2` 路由条件 | 补全，并删除 “no output changes” |
| — | minor | serial 分支 `stateWorkspaceOffset` 死分配（6–12 MiB，kernel 从不读写 `gmState`） | 不再分配，字段置 0 |
| — | major | `DataCopyPad` UB→GM 的 `srcStride` 单位误用为字节（API 规定为 32B dataBlock） | summary 分半写的 `srcStride` 改为 16；GM→UB 方向经 API 卡核实单位为字节，无需改 |

## 5.6 Skill 应用记录（2026-10-29）

工作区 `.claude/skills/` 中的 `ascendc`、`ascendc-api-knowledge-query`、`profiling-analysis`
在本会话启动时未被加载；本轮按文件直接执行：

- **ascendc**（`references/10-code-review.md`）：按假设检验法复核数值操作、内存/指针
  安全、资源管理、输入校验、并发安全、算子接口、ABI 七类规则；发现并修复 §5.5 的
  summary 布局、跨 task 竞争等条目。
- **ascendc-api-knowledge-query**：平台 `950`，全文阅读并核实
  `DataCopyPad`（UB→GM 与 GM→UB 的 `srcStride`/`dstStride` 单位不同）、
  `CrossCoreSetFlag/WaitFlag`（mode 0/1/2/4 语义、mode 4 的 `AIV1 ↔ id+16` 映射、
  mode 4 必须显式传 `modeId=4`、flagId 范围 0–15/16–31）、`ReinterpretCast`
  （GlobalTensor/LocalTensor 地址与内容不变）。据此修正了 summary 分半写的
  `srcStride` 单位错误，并确认 0x4 配对与 0x0 AIV 屏障的 flagId 用法正确。
- **profiling-analysis**：需要设备侧 profiling 数据（`step_trace_time.csv` 等），
  在 §3 的性能步骤完成后按该 skill 判型（Hostbound / Computing / Communication）
  并生成报告。

保留项（有意为之，非缺陷）：
- **#5 CP 默认开启**：用户已决策“满足形状条件即走 CP”；设备验证前以本文档记录风险，
  `GDN_FWD_H_CP=0` 可即时回退。
- **#7 FP32 rolling**：重放状态全程 FP32，比串行的低精度 h 载体更接近 FP64 标杆；
  design.md §2.6 已写明，A/B 差异按 bf16 量化量级监控。
- **#8 host/kernel 常量重复**：两侧均有 “Keep in sync” 注释，stride 由 tiling 下发；
  后续可抽公共 constexpr 头。
- **#9 arch22 的 key3 空分支**：正常构建不可达（`useCp` 要求 `isAscend950`），保留为
  防御性建议。

## 5.7 优化轮次记录（2026-10-29，交接单 P1.1 + P2.6）

**未编译、未上板。** 本机为 macOS、无 CANN，SSH 节点不可达，交接单 §3 的 P0 未执行；
本轮只能做代码与静态核对。任何“已优化”的结论都必须等设备侧 P0 基线复核。

| 项 | 改动 | 预期收益 | 风险 |
| --- | --- | --- | --- |
| P1.1 L0C→UB 直连 | 阶段 1 两次 GEMM、阶段 2 前缀 GEMM、阶段 3 C1/C2 全部改成 `Common::BlockMmadTla` + `PackedTileCopyTlaToUB(NO_SPLIT)`，结果直写 UB；新增 `UbFree` 握手（ID 11） | 去掉 Pv/YT/v_work/h_work/前缀 Y 的写+读往返，阶段 1 GM 字节约 −50%，阶段 3 约 −55% | 高：新增一组跨核 flag；`NO_SPLIT` + `subBlockId` 的 `ceil(m/2)` 行分配必须与 AIV 分工一致 |
| P2.6 q 向量化 | `PrepareQ` 一次搬运整块 g、一次 `Sub`+`Exp` 得到 q 与 -q；逐行 `Muls` 换成寄存器基 `ScaleRowsKernel` / `SubScaleRowsKernel` | 每 chunk 去掉 2×(BT 次标量 `expf` + BT 次 GM 标量读 + 3×BT 次 128 元素小向量调用) | 中低：标量 `expf` 换成向量 `Exp`，末位可能不同；与串行重放的口径一致 |
| 阶段 2 偏置加 | 逐行 `Add`(128) 换成 `AddRowsKernel` | 每次前缀 GEMM 少 `rank`×64 次小向量调用 | 低 |
| 顺带修复 | `Catlass::Arch::CrossCore{Set,Wait}Flag<...>(id)` 只接受 `CrossCoreFlag&`，原代码传裸 id 无法编译；改为 `AscendC::` 同名接口 | 编译 | 低 |
| 路由缺口 | 尾块 `tokens < 16` 时 CP 把短 M 交给 Cube，而串行实现把短块交给向量侧；`useCp` 增加 `尾块 >= 16`（`kCpMinChunkTokens`），其余回退 key1/2 | 消除 `T % chunk != 0` 的未知行为 | 低（仅收窄 CP 路由，公开接口不变） |
| 审查修复 | 静态审查（独立 agent）发现 3 个 blocker，均已修：① 类里没有 `Process()`，`ChunkGatedDeltaRuleFwdHCPImpl` 调不到；② `CopyL0CToUBMode::SPLIT_M` 只有 3 参 `operator()`，而 block 的 `sendVecNum>1` 输出路径调 5 参版本 → 全部改 `NO_SPLIT`；③ Kᵀ 走 ColumnMajor GM，而 Ascend950 只有 `ColumnMajor→nZ`，原 tile copy 声明 `RowMajor`（→zN）无特化 → 新增 `CpF32KvTileCopy` / `CpInKvTileCopy` 与对应 mmad | 编译 | 低 |
| UB 假设显式化 | 补 `static_assert(sizeof(INPUT_TYPE) <= 2)` 等：UB 布局按 2 字节输入推导，fp32 输入会让几处逐阶段区域重叠 | 防止未来放开 fp32 时静默写坏 v_new | 低 |

仍然保留的已知项（未在本轮处理）：

1. **`SCR_YT` / `SCR_PV` 已不再读写**，仍按原公式保留槽位（不改 host 布局）。
   若要回收这 256 KiB/rank，需同时改 host 的 `cpScratchStride` 与 kernel 的 `SCR_*`。
2. **4 个 `BlockMmadTla` 实例共用同一段 L1/L0 起点**（每个阶段只用其一），
   初始化 flag 由各自构造函数置位、析构函数等待，计数平衡；与参考实现的 4 实例做法一致。

### 5.7.1 首次设备编译修复（2026-10-29）

设备侧首轮编译报出的 3 类错误，已修。三条都是“本仓其他 kernel 从未用过，所以没先例可抄”：

| 错误 | 原因 | 修复 |
| --- | --- | --- |
| `use of undeclared identifier 'INPUT_TYPE' / 'G_TYPE' / 'STATE_TYPE'` | UB 尺寸的 `static_assert` 写在了 `namespace GDN` 作用域，而这三个是类模板参数 | 移到 `GDNFwdHCPKernel` 类内 |
| `use of undeclared identifier 'expf'` | 设备编译单元里没有标量 `expf`；本仓所有 gate 计算都走向量 `Exp` | 新增 `ChunkDecay()`：`Duplicate` 一个 32 B 块 → `Exp` → `V_S` 同步后 `GetValue(0)` |
| `no member named 'CeilDiv' in namespace 'Catlass'` | `catlass/detail/alignment.hpp` **整个文件没有 namespace**，`RoundUp/CeilDiv/Max/Min` 都在全局作用域；且 `regbase.hpp` 在全局 `using namespace AscendC::MicroAPI`，无歧义的非限定查找也不可靠 | 在 kernel 内自定义 `CpMin` / `CpCeilDiv`，不再依赖 Catlass 的这两个名字 |

**给下一个会话的教训**：本仓 `Catlass::Min` / `Catlass::CeilDiv` 从来不存在（旧版 CP 代码里也
是错的，只是从未编译）；设备核里不要假设标量 `<math.h>` 可用。

## 6. 风险与回退

| 风险 | 触发 | 回退 |
| --- | --- | --- |
| `0x4` 配对或 `0x0` AIC 屏障死锁 | S1 哨兵挂死 | 改用 `SyncAll<true>()` + `PipeBarrier<PIPE_ALL>`（AIV 同点调用） |
| `[128,256,128]` FP32 BlockMmad 编不过/越界 | 构建失败 | 拆成 4 次 128 宽 GEMM（M/S 分开），协议不变 |
| FP32 被静默降为 HF32 | identity GEMM 误差 >1e-6 | 确认 `SetHF32Mode(false)`；必要时显式校验 |
| 精度不过 1e-4 | PTA 失败 | 缩短复合链（每 J chunk 中间 summary）→ 跨 rank 接力 |
| `B*HV` 接近核数时慢于链并行 | 性能实测 | `GDN_FWD_H_CP=0`；按实测回填路由门限 |
| workspace 变大（CP 约 +7 MiB @ R=24） | aclnn 上限 | 复用 `stateWorkspaceOffset`、压缩 scratch 槽 |
| tail/偏斜切分 flag 失衡 | S1/S4 挂死或数据错 | C=5 用例覆盖；空行仍执行完整 flag 序列 |

## 7. 性能追踪表（待填）

| case (B,T,HV,chunk) | key1/2 (us) | key3 (us) | 比值 | 备注 |
| --- | --- | --- | --- | --- |
| 1, 16384, 12, 32 | 待测 | 待测 | — | 历史基线 1247 us（链并行） |
| 1, 32768, 4, 64 | 待测 | 待测 | — | |
| … | | | | |
