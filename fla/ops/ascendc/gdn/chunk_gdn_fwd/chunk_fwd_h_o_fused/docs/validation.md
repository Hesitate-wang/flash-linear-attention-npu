# Development validation

## Traceability

| Design item | Implementation |
| --- | --- |
| `P` producer plus `P` consumer cores | host tiling `producerCoreNum`, `consumerCoreBase`, `activeCoreNum`; local H/O schedulers |
| full H/v_new handoff | host `FillWorkspace`; kernel `handoffHWorkspaceOffset` and `handoffVWorkspaceOffset` |
| startup event initialization | `InitializePipelineSync` in the fused kernel entry |
| per-chunk H-to-O publication | local H `SignalChunkReady`; local O AIV `WaitProducerSliceReady`; reverse `cube1Done` acknowledgement gates the O AIC |
| no sibling private dependency | all H/O kernel, scheduler and epilogue files are operator-local |

## Static checks completed

- Host and kernel fused tiling mirrors use the same field order; host tiling
  retains the exact local-H prefix and checks the complete serialized size.
- Workspace offsets are user-workspace-relative and every region is aligned to
  512 bytes; the returned size adds the platform system workspace exactly once.
- `blockDim` is `2 * B * HV`, guarded by `2 * B * HV < physical AIC cores`.
- Fixed-length Atlas A2 exp is the only accepted first-version mode; varlen,
  exp2 and Ascend 950 fail during tiling instead of selecting an incomplete
  kernel path.
- Cross-operator include scan is empty: the fused tree does not reference a
  sibling operator path or an `internal` implementation path.
- The IB local tensor remains on the SIMD side as required by the API. In MIX
  mode the IB index space is `2 * blockDim`; paired AIV waits are aggregated by
  the reverse `cube1Done` generation before the AIC reads complete H/V tiles.
- Static task-map simulation for `(B,HV,NC)=(1,8,3)` and `(2,4,3)` confirms
  that producer `p` and consumer `P+p` enumerate identical `(b,hv,chunk)`
  tuples and use 16 of 24 mixed cores.
- `git diff --check` passes (line-ending conversion warnings only).

## Environment limitation and pending device evidence

This Windows workspace exposes no configured CANN environment, NPU device or
usable Python runtime, so an operator build and runtime accuracy test cannot be
completed here. The following remain mandatory on the target environment:

1. build and install the single-operator package for `ascend910b`;
2. run a minimum fixed-length case with at least three chunks, V=128, with and
   without initial/final state and `gk`;
3. run V=256 and tail-chunk cases;
4. compare `o` and optional final state against the CPU/reference composition;
5. collect an execution trace proving producer/consumer chunk order and IB
   event reuse, then profile handoff L2 hit rate before considering buffering.
