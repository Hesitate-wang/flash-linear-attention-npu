# Arch35 Chunk64 Sparse GDN Forward Validation

## Current status

The chunk64 directory was redesigned and implemented from scratch. It uses a two-group producer/consumer pipeline, an FP32 register triangular solve, producer-side A', four consumer GEMMs, and a dedicated A' workspace. It does not reuse the former sparse implementation or its FP32 Cube solve.

The development host is macOS without `ccec`, an Ascend toolkit, or a visible NPU. Ascend compilation, device numerical correctness, synchronization stress, MSS and performance therefore remain unverified. No performance improvement is claimed without device measurements.

## Checks run in this session

| Check | Method | Result |
|---|---|---|
| Public ACLNN ABI | `python3 -I tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py` | Passed; 29 public parameters |
| Internal OpDef/device argument order | `python3 -I tests/atk/chunk_gated_delta_rule_fwd/sparse_entry_contract.py` | Passed for arch35 and arch22 |
| Host POD and workspace helpers | C++17 `-Wall -Wextra -Werror`: `sparse_tiling_contract.cpp` | Compiled and ran successfully |
| Source whitespace | `git diff --check` | Passed |
| Include boundary | Inspected all `chunk_64_sparse` includes | Only Ascend SDK headers and files inside this operator |
| 64x64 inverse | NumPy forward substitution versus `inv(I+raw)` | Passed at FP64 tolerance |
| 64-element scan | Six-stage snapshot scan versus `cumsum` | Passed at FP64 tolerance |
| Full one-chunk dataflow | New W/U/A'/H/O equations versus token recurrence | Passed for H and O at FP64 tolerance |
| Strongly negative gate regression | FP32 reciprocal path versus direct `E(g[i]-g[j])`; lightweight `--gate-scale 100` | Reciprocal path reproduced NaN; direct-difference path stayed finite |
| Lightweight test framework | `tests/run_lightweight.sh --case all --cpu-only`, plus exp2/sigmoid/stateVFirst options | Dense, varlen and empty-sequence logs/summaries generated successfully |
| Ascend C API contract | CANN knowledge cards: CrossCoreSetFlag/CrossCoreWaitFlag, arch35 regbase selection | Mode 0/4 routing and regbase amortization reviewed |

The numerical checks validate the redesigned equations, not Ascend instruction behavior. They are not ATK device tests.

## Resource checks

- Producer UB map: 144 KiB per AIV; L1 map: 72 KiB.
- Consumer UB map: 190 KiB per AIV; L1 map: 152 KiB.
- Both UB maps have static assertions against the 208 KiB regbase budget.
- A' workspace is exactly `totalChunks * Hv * 8192` bytes before system workspace, with checked host arithmetic.
- Every ComputeMm has one Fixpipe and one ReleaseL0C. The launch primes and drains M_MTE1/FIX_M once.

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
- `op_host/op_tiling/arch35/chunk_64_sparse/chunk_gated_delta_rule_fwd_sparse_tiling.cpp`: selector, exact launch width and A' workspace.
