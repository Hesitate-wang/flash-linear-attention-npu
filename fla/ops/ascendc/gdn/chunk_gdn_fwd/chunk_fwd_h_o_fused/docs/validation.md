# Development validation

## Traceability

| Design item | Implementation |
| --- | --- |
| `P` producer plus `P` consumer cores | host tiling `producerCoreNum`, `consumerCoreBase`, `activeCoreNum`; local H/O schedulers |
| full H/v_new handoff | host `FillWorkspace`; kernel `handoffHWorkspaceOffset` and `handoffVWorkspaceOffset` |
| startup event initialization | `InitializePipelineSync` in the fused kernel entry |
| per-chunk H-to-O publication | local H `SignalChunkReady`; local O AIV `WaitProducerSliceReady`; reverse `cube1Done` acknowledgement gates the O AIC |
| A5 H-to-O stage boundary | arch35 H drains its events; `RunChunkFwdHOFusedA5` executes `SyncAll<false>()` before arch35 O |
| A5 scratch ownership | host `FillWorkspaceA5`; local H offsets plus `oAPrimeWorkspaceOffset` |
| no sibling private dependency | all H/O kernel, scheduler and epilogue files are operator-local |

## Static checks completed

- Host and kernel fused tiling mirrors use the same field order; host tiling
  retains the exact local-H prefix and checks the complete serialized size.
- Workspace offsets are user-workspace-relative and every region is aligned to
  512 bytes; the returned size adds the platform system workspace exactly once.
- A2 `blockDim` is `2 * B * HV`, guarded by
  `2 * B * HV < physical AIC cores`; A5 uses every physical AIC.
- Fixed-length Atlas A2 exp and Ascend 950 exp2 are accepted. A5 is constrained
  to BF16 data, BF16/FP32 gates, chunk 64, K=V=128 and HV/HK in [1,4]. Varlen
  continues to fail during tiling.
- Cross-operator include scan is empty: the fused tree does not reference a
  sibling operator path or an `internal` implementation path.
- The complete local quoted-include scan resolves against the including file,
  `op_kernel`, or `op_kernel/arch35`. The previously omitted arch35
  `block_epilogue_gdn_fwdh_regbase.hpp` is now operator-local and matches the
  standalone FwdH implementation; CMake fails during configuration if either
  it or the kernel entry source is absent.
- The operator-local O-stage structure header contains both the compact A2
  projection and the complete `ChunkFwdOTilingData` projection consumed by the
  copied Ascend 950 O implementation. `FillOTiling` initializes every field in
  the Ascend 950 projection before dispatch.
- The IB local tensor remains on the SIMD side as required by the API. In MIX
  mode the IB index space is `2 * blockDim`; paired AIV waits are aggregated by
  the reverse `cube1Done` generation before the AIC reads complete H/V tiles.
- Static task-map simulation for `(B,HV,NC)=(1,8,3)` and `(2,4,3)` confirms
  that producer `p` and consumer `P+p` enumerate identical `(b,hv,chunk)`
  tuples and use 16 of 24 mixed cores.
- Ascend 950 is present in the OpDef registration and selects the
  `__CCE_AICORE__ == 310` implementation. It runs operator-local arch35 H,
  crosses an all-core stage boundary, then runs operator-local arch35 O.
  Workspace offsets for H/v_new handoff, H scratch and O A-prime scratch are
  disjoint.
- `git diff --check` passes (line-ending conversion warnings only).

## Environment limitation and pending device evidence

This Windows workspace exposes no configured CANN environment, NPU device or
usable Python runtime, so an operator build and runtime accuracy test cannot be
completed here. The following remain mandatory on the target environment:

1. build and install the single-operator package for `ascend910b` and
   `ascend950`;
2. run a minimum fixed-length case with at least three chunks, V=128, with and
   without initial/final state and `gk`;
3. run V=256 and tail-chunk cases;
4. compare `o` and optional final state against the CPU/reference composition;
5. collect an execution trace proving producer/consumer chunk order and IB
   event reuse, then profile handoff L2 hit rate before considering buffering.
6. on Ascend 950, run `--use-exp2` and compare both fused and composed BSND
   outputs for BF16/FP32 gates, with and without initial/final state.
