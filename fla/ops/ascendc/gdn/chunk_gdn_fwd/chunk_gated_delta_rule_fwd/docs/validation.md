# Arch35 Chunk64 Sparse GDN Forward Validation

## Current status

The chunk64 directory uses a two-group producer/consumer pipeline, consumer-side QK/A', and producer-side W/U. The current performance candidate replaces the single-AIV0 full 64x64 inverse with two independent FP32 32x32 diagonal solves on AIV0/AIV1, followed by two FP32 sixteen-row coupling tasks and a 12 KiB per-tile exchange workspace. The public ABI and BF16 Cube contraction path are unchanged.

The development host is macOS without `ccec`, an Ascend toolkit, or a visible NPU. Ascend compilation, device numerical correctness, synchronization stress, MSS and performance therefore remain unverified. The dual-AIV inverse is not yet claimed to improve performance.

## Checks run in this session

| Check | Method | Result |
|---|---|---|
| Public ACLNN ABI | `python3 -I tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py` | Passed; 29 public parameters |
| Internal OpDef/device argument order | `python3 -I tests/atk/chunk_gated_delta_rule_fwd/sparse_entry_contract.py` | Passed for arch35 and arch22 |
| Host POD and workspace helpers | C++17 `-Wall -Wextra -Werror`: `sparse_tiling_contract.cpp` | Compiled and ran successfully |
| Source whitespace | `git diff --check` | Passed |
| Include boundary | Inspected all `chunk_64_sparse` includes | Only Ascend SDK headers and files inside this operator |
| 64x64 inverse | NumPy forward substitution versus `inv(I+raw)` | Passed at FP64 tolerance |
| Dual-AIV blocked inverse | 800 randomized FP64 matrices: `X0/X1` diagonal inverses plus `-(X1@L10)@X0` versus full inverse | Passed; max absolute error `4.441e-16`; does not validate Ascend instruction behavior |
| Regbase overload review | `UpdateMask<float>(32)` changed to `CreateMask<float, MaskPattern::VL32>()`; invalid `LoadAlign(..., mask)` third arguments removed | Source calls corrected against the Ascend 950 API declarations; Ascend rebuild remains pending |
| 64-element scan | Six-stage snapshot scan versus `cumsum` | Passed at FP64 tolerance |
| Full one-chunk dataflow | New W/U/A'/H/O equations versus token recurrence | Passed for H and O at FP64 tolerance |
| Strongly negative gate regression | FP32 reciprocal path versus direct `E(g[i]-g[j])`; lightweight `--gate-scale 100` | Reciprocal path reproduced NaN; direct-difference path stayed finite |
| Lightweight test framework | `tests/run_lightweight.sh --case all --cpu-only`, plus exp2/sigmoid/stateVFirst options | Dense, varlen and empty-sequence logs/summaries generated successfully |
| Ascend C API contract | CANN knowledge cards: CrossCoreSetFlag/CrossCoreWaitFlag, arch35 regbase selection | Mode 0/4 routing and regbase amortization reviewed |

The numerical checks validate the redesigned equations, not Ascend instruction behavior. They are not ATK device tests.

## Resource checks

- Producer UB map: 144 KiB per AIV; L1 map: 72 KiB.
- Consumer UB map: 200 KiB per AIV; L1 map: 184 KiB.
- Both UB maps have static assertions against the 208 KiB regbase budget.
- Inverse exchange workspace is `totalChunks * Hv * 12288` bytes before system workspace, with checked host arithmetic. Each chunk writes 12 KiB and reads 12 KiB of FP32 X0/X1/L10 blocks.
- Every independent ComputeMm has one Fixpipe and one ReleaseL0C. The launch primes and drains M_MTE1/FIX_M once.

## Performance tracking

| Model case | Round / status | Trigger or expected critical path | Candidate and single main variable | Accuracy | Performance before/after | Bottleneck change | Evidence or conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `(1,1,16384,128)`, BF16, chunk64, Ascend 950 | Baseline | Single AIV0 executes the full FP32 64x64 inverse | Current implementation | Device result supplied by user is not an accuracy report | 2400 us `Task Duration` / pending | Pending | Baseline value supplied by user; warm-up/sample protocol still needs to be frozen |
| Same case | Dual-AIV inverse, pending measurement | Parallel X0/X1 solves; FP32 coupling and GM exchange may trade vector time against MTE/sync time | Only the inverse task mapping and its private workspace change | Pending | 2400 us / pending | Pending | Adopt only after existing accuracy gates pass and the same-case median improves beyond repeated-run variation |
| Same case | Consumer W/U schedule, pending measurement | The current unified W/U/Q/K/g load blocks prepare and QK before the W/U rendezvous | Move the per-chunk mode-0 barrier between consumer QK/A' and W/U loading; remove the unused U L1 upload | Pending | Pending / pending | Pending | Measure independently after the inverse candidate; do not attribute a combined change to one stage |
| `(1,14,16384,128)`, BF16, chunk32 | Reference only | Fourteen independent chains on fourteen MIX groups | Different kernel and workload | Not supplied | 850 us / not applicable | Not applicable | Not an equal-work baseline; use same-shape A/B for the chunk64 candidate |

## Required Ascend 950 validation

1. Build the standalone operator with `bash build.sh --pkg --soc=ascend950 --vendor_name=fla_npu --ops=chunk_gated_delta_rule_fwd`; fix any CANN-version API diagnostics.
2. Run `bash tests/run_lightweight.sh --case all --gate-scale 100 --device 0` from the operator root for fast DUT + FP64 golden checks, then run `tests/atk/chunk_gated_delta_rule_fwd/scripts/pipeline_smoke_test.py` (dense case 348 and varlen case 187) for the full three-way comparison.
3. Cover tails 1/16/31/32/33/63/64/65 tokens, Hv/Hk=1/2/3/4, norm on/off, exp/exp2, beta sigmoid/output coupling, allowNegEigval, stateVFirst, FP32/BF16 initial/final state, A output and zero-length varlen sequences.
4. Repeat execution and run MSS to validate mode-0/mode-4 counters and GM/UB/L1/L0 lifetimes.
5. Regress chunk32 and Phase6/fallback routes.
6. Measure the same real low-parallelism workload with `msprof op_summary` `Task Duration(us)`. Compare against the pre-change implementation only after both pass the same accuracy gate; retain or revert optimizations based on measured medians.

## Main source files

- `op_kernel/internal/arch35/chunk_64_sparse/sparse_vector.h`: scan, normalization, inverse, gates, state and output VF.
- `sparse_cube.h`: LoadData/Mmad/Fixpipe helpers and L0 operand reuse.
- `sparse_producer.h`: KKT/QK, inverse, W/U/A' production.
- `sparse_consumer.h`: H recurrence and O production.
- `sparse_sync.h`: mode-0 pipeline and mode-4 directional handshakes.
- `sparse_memory.h`: UB/L1/L0 maps.
- `sparse_fused_tiling.h`: minimal shared POD.
- `op_host/op_tiling/arch35/chunk_64_sparse/chunk_gated_delta_rule_fwd_sparse_tiling.cpp`: selector, exact launch width and inverse exchange workspace.
