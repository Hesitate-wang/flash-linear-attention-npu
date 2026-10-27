# FWDH CP32 Development Validation

## Scope and environment

Dense Ascend950 chunk32, K=V=128, B*HV<=AIC core count; chunk64/128 retain their
existing dispatch. Current development host is macOS without a configured CANN
installation. No device compilation, execution, sanitizer, or performance result
is available. This file remains a development record pending formal acceptance.

Development reference: CHUNK_DEPENDENT V1, source commit
`a14961e9dfe8c8264768d67fb447101873da4ecf`. Reused Cube/Vector separation,
producer/consumer handoffs, and tail padding. Task mapping follows the user's
explicit rank-per-core scheme: all cores cooperate on each sequential batch/head task.

## Design-to-code trace

| Requirement | Implementation | Evidence/status |
| --- | --- | --- |
| CP dispatch | Existing TilingKey 3 and useCpPath branch | Retained; compiled dispatch evidence pending |
| Rank mapping | cpRankCount=CoreNum; ceil contiguous chunk partition; sequential task loop | Source implemented; device mapping evidence pending |
| Chunk affine | ColumnMajor K view; Cube KᵀW/KᵀU; Vec diagonal subtraction | Source implemented; build and intermediate comparison pending |
| Saved prefixes | Per-chunk exclusive B/A; per-rank inclusive B/A | CPU formula comparison passes; device comparison pending |
| Prior-rank prefix | Sequential Bprior@P and Bprior@Q+Aprior with separate temporaries | CPU formula comparison passes; device comparison pending |
| Replay | Cube global affine/H0 materialization and W@H; Vec additions/subtraction and output cast | Source implemented; device comparison pending |
| Final state | Last rank materializes its inclusive summary, including empty rank | Covered by CPU formula test; device comparison pending |
| Workspace | 2*(R+C) matrix slots for summaries/records; 14*R scratch slots; 64 KiB per slot | Host/kernel formulas aligned in source; device bounds check pending |
| GM/UB events | Load uses V_MTE2/MTE2_V; Store uses V_MTE3/MTE3_V; scalar gate copied through UB | Event pairing added; target compile and device validation pending |
| Cross-core sync | mode 2 ready/done; mode 0 AIC inter-block barrier ID 9 after summary and replay | Barrier ID follows CATLASS; mode-2 flag counts and repeated-task lifecycle remain unverified; do not treat as deadlock-safe |
| Legacy regression | Non-CP tiling and kernels retained | Build and chunk64/128/varlen regression pending |

UB allocation is 4096 bytes per AIV. Operands and results use separate GM slots.
No asynchronous overlap or measured performance improvement is claimed.

## Local validation

Run:

```
python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests/pta/test_cp_affine.py
git diff --check
```

The CPU tests compare affine prefix replay with the direct serial recurrence for
none/g/gk/both gate modes, optional initial state, tail lengths, rank counts
1/2/3/8 and empty ranks. The no-gate case is mathematical coverage; the public
API continues to require at least one gate. These checks do not validate the
Catlass layout implementation, CANN API compilation, device memory visibility,
or hardware flag behavior.

## Required device validation

1. Build the Ascend950 single-op package, install it, and verify package/runtime
   consistency using the repository development procedure. Confirm key 3 and all
   legacy keys compile.
2. Compare Stage 1 Bc/Ac, saved chunk prefixes and rank summaries against CPU
   intermediates, then compare Stage 2 rank prefix and replay states.
3. Compare h/v_new/final_state with the established independent CPU reference,
   using the operator ATK README and tests/atk/README.md. Include R=1, R>C,
   C not divisible by R, tail chunk, optional initial/final state, both
   state_v_first modes, g-only/gk-only/both and supported dtype combinations.
4. Exercise multiple batch/head tasks in one invocation. Verify no next-task
   overwrite occurs before both AIC and paired AIVs pass the final barrier.
   Record rank, chunk, buffer generation, and ready/done steps in a separate
   diagnostic workspace if runtime tracing is needed; tracing must add no sync.
5. Run memory/race/synchronization sanitizers and repeated executions with
   diagnostics disabled. Terminate a kernel test that remains active for 60s,
   per repository policy. Previous CPDBG messages were removed with the old
   algorithm and must not be used as evidence for this implementation.
6. Verify rejection of unsupported chunk32 inputs, and regress chunk64/128 plus
   existing variable-length paths. Measure Task Duration under matched conditions
   before making any performance claim.

## Performance tracking

Model case: `B=1, T=16384, HK=HV=12, K=V=128, bf16, chunk32`, SoC ascend950,
baseline = same build at `vSplitCount=1`. Metric: `msprof op_summary` 的 `Task Duration(us)`。
表中旧行（"User rank-prefix scheme"）对应已删除的 CP 实现，仅作历史记录。

| Model case | Round / status | Expected critical path | Decision / variable | Precision | Before/after performance | Bottleneck change | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chunk32 | 基线 | 每 chunk 4 级串行（C1→V1→C2→V2），12 task / 24 核 | 现状 Preload 路径 | Pass | 1247 µs | AIC MTE2 68% busy，cube 6.5% | 交接文档 profiling |
| chunk32 | 放弃 | 假设吞吐受限，期望 650 | V 轴切两半（vSplits=2，24 task） | Pass | 1247 → 1384 µs | 未变；W/K 每 task 重读，总字节 +10% | 本表，待归档日志 |
| chunk32 | 采用 | 每 chunk 时长与 tile 宽度无关（2.44/2.77/2.70 µs） | 结论：V 切分默认关闭，保留 `GDN_FWD_H_VSPLITS` 钩子 | Pass | 1384 → 1274 µs | — | 同上 |
| chunk32 | 采用 | Preload epilogue 手写 SIMD 只支持 128 列宽 | 恢复 SIMD 快路径，窄列宽回落普通向量算子 | Pass | 1274 → 1253 µs | — | 同上 |
| chunk32 | 采用 | V2 每 chunk 从 GM 回读 h[c]（32KB，占总量 21%） | h[c] 在 UB 跨 chunk 常驻，仅链首载入（`taskSlotsPerCore>1` 时回退） | Pass | 1253 → 1231 µs | V2 的 h 读原本被掩盖，收益仅 1.3% | 同上 |
| chunk32 | 待实现 | 1 核 1 链时每 chunk 有 ~0.8 µs AIC MTE2 空泡（1.62 忙 / 2.40 实际） | Step1: per-stream 跨核 flag + UB 按物理槽位分配（vSplits=1 应零回归） | — | 目标 = 1231 | — | 本次改动 |
| chunk32 | 待实现 | 24 核 × 2 stream 提供空泡填充；vSplits=4 时 AIC 26KB/task | Step2: 开 vSplits=4（48 链，2 stream/核），行缩放与 h 拷贝支持 32 列 | — | 预计 ~770 µs | — | 字节模型 |
| chunk32 | 待实现 | CP 单独 ≈1000 µs（用上 24 核但空泡仍在）；CP + 2 stream ≈600-650 | Step3: 本地仿射 (B_c,A_c) + 跨 rank 前缀 + 分段 replay | — | 目标 600-650 µs | — | 字节/工作量模型 |

### 关键结论（用于后续实现）

- 每 chunk 时长的固定部分（4 级依赖 + 3 次跨核 flag + GM 往返）约 2.4 µs，与 tile 宽度基本无关；
  单纯增加核数或切 V 不会变快（vSplits=2 实测 1384）。
- AIC MTE2 每核有效带宽约 34.6 GB/s，1.62 µs/chunk 的工作量配 2.40 µs 的实际时长 → 33% 空泡。
  空泡只能由**同一核上的第二条独立链**填补。
- chunk 轴并行（仿射前缀）把 12 条链摊到 24 核，但 phase A 增加约 0.8 µs/chunk 的仿射合成工作量；
  它必须与"每核 2 条链"叠加才能到 600-650，否则约 1000 µs。

