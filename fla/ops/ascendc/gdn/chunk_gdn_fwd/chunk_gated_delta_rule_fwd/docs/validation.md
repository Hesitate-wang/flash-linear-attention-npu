# Arch35 Chunk64 GDN Forward Development Validation

## Scope and status

The existing public ACLNN entry has an incomplete arch35 key-64 branch for BF16 Q/K/V, K=V=128 and FP32 gate/beta (optional a_log/dt_bias also FP32). The internal schema is append-only and key 64 uses a separate Prepare/H/O POD. Native low-precision Prepare inputs retain the existing three-stage route. For eligible low-parallelism calls (`B*Hv <= AIC/2`) the key-64 tiling now selects a producer/consumer schedule: producer group c builds chunk i while consumer group `chainCount + c` finishes chunk i-1, paced by one all-AIV mode-0 barrier per iteration. The serial Prepare -> H -> O composition remains as the fallback when the pipeline plan is not selected. This producer/consumer path exists only at source level; it has not been compiled by ccec or executed on Ascend 950, so it is not yet evidence of correctness, synchronization safety, or overlap.

This host is macOS without the Ascend compiler/device. Source integration and ordinary host-side contract checks are distinct from CANN compilation, numerical correctness, synchronization safety and performance.

## Contract and implementation

| Item | Current source contract |
|---|---|
| Public API | Existing ACLNN signature and Python ten-element result unchanged |
| Beta | Sigmoid follows non-null beta_eff output; allowNegEigval retains the existing requirement |
| Internal schema | Two appended inputs, ten optional outputs, five appended attributes |
| Dispatch | fused_prepare branches before Phase6 validation; key 64 has its own POD |
| Topology | PlatformAscendC::GetCoreNumAic() determines the common MIX group count; schedule mode 1 |
| Workspace | Prepare L×64 KiB; H 32 MiB + L×704 KiB; O L×64 KiB; disjoint regions plus system workspace |
| Intermediate tensors | Existing ACLNN executor allocations, not duplicated scratch records |
| Modes | Dense/packed metadata, norm, exp/exp2, stateVFirst and initial/final state carried in tiling |
| Compatibility | Key 32 and Phase6 keys 1/2/301 retain their layouts and code paths; device regression pending |

## Checks actually run

| Check | Method | Result |
|---|---|---|
| Public ACLNN and ctypes ABI | Person/.venv Python: tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py | Passed; 29 public parameters |
| Shared POD and workspace helper | C++17 with -Wall -Wextra -Werror: sparse_tiling_contract.cpp | Compiled and ran successfully |
| Source whitespace | git diff --check | Passed at host integration checkpoint |
| Internal OpDef/device argument order | sparse_entry_contract.py, including Prepare mirror field order | Passed for arch35 and arch22 |
| Producer/consumer source integration | `sparse_pipeline.h` NT+1 loop, `sparse_consumer.h` cube/vector roles, `sparse_producer.h` wiring | Source complete; no compiler or device evidence |
| Ascend C API contract review | knowledge-query/ascendc-api-knowledge-query cards: CrossCoreSetFlag/CrossCoreWaitFlag (modes 0/1/2/4, flagId ranges), FixpipeL0CToUB (FixpipeParamsArch3510, srcStride units), LoadData 2D V2, Mmad | Checked; fixed AIV1 mode-4 raw-id routing and removed a redundant GM->L1 DataCopy (mode-4 flagId 1/17) |
| Consumer UB/L1 maps | `sparse_memory.h` static_assert (consumer UB 173 KiB, L1 152 KiB) | Compile-time bounds hold; runtime `GetCoreMemSize` pending |
| CANN host and kernel compilation | Ascend 950 project build | Not run; toolchain unavailable |
| CPU golden versus device | Existing gdn_reference.py and ATK harness | Not run; device unavailable |
| Repeated synchronization and MSS | Ascend 950 | Not run |
| Legacy routes | Chunk32 and Phase6 device regressions | Not run |
| Performance | msprof op_summary Task Duration(us) | Not run; no speedup claimed |

The ABI script initially failed because it searched for the former inline exp/exp2 expression. Prepare now dispatches GateLowerLVF and ScaleRowsBetaExp2gVF by true/false template argument; the source contract check was updated to verify those actual branches and then passed.

## Remaining verification

- Compile on Ascend 950; resolve any CANN API/compiler diagnostics before runtime testing. The producer and consumer roles have never entered ccec.
- First bring-up: run the NT+1 barrier-only loop (both roles' `RunChunk` disabled) to confirm the mode-0 participant count and repeated-launch flag state, then re-enable producer-only and compare W/U/g_cumsum against the serial Prepare kernel.
- Source-level phase review: the pipeline uses mode-0 flag 7 across all AIVs, group-local mode-2 id 8 for the iteration boundary, and mode-4 ids 0-6/9 for stage handshakes. All groups, including idle ones, enter every barrier; the AIC never enters mode 0 and is released through its own AIVs. This is a source audit, not a device synchronization proof.
- Validate dense and packed tails, head ratios 1/2/3/4, both exponential conventions, norm on/off, sigmoid/output coupling, state layout and dtype, initial/final state, and every optional public output. `A'` includes the diagonal (`j <= i`), matching `chunk_fwd_o` Stage3 and chunk32's `GateAttnVF`.
- Run repeated execution and MSS, then chunk32/Phase6/fallback regressions.
- Measure a real low-parallelism workload before claiming a performance benefit. Key 64 now has a source-level producer/consumer schedule but no device or timing evidence.

## Source locations

- op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.cpp: existing public composite and fused/fallback choice.
- op_host/chunk_gated_delta_rule_fwd_def.cpp and op_host/op_api/chunk_gated_delta_rule_fwd.cpp: appended internal schema and launcher.
- op_host/op_tiling/arch35/chunk_64_sparse/chunk_gated_delta_rule_fwd_sparse_tiling.cpp: key64 tiling and workspace.
- op_kernel/internal/arch35/chunk_64_sparse/sparse_fused_tiling.h: shared host/device POD.
- op_kernel/internal/arch35/chunk_64_sparse/sparse_phase_adapter.h and sparse_sync.h: device phase composition and handoff.
- fused-entry-source-analysis.md: historical Phase6 integration analysis, not current implementation status.
