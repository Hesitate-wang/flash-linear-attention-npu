# ChunkFwdHOFused

`ChunkFwdHOFused` is the operator project reserved for fusing the state update
performed by `ChunkGatedDeltaRuleFwdH` with the output calculation performed by
`ChunkFwdO`.

## Status

The host-side operator definition, combined tiling, L0/aclnn APIs, the
fixed-length Atlas A2 producer/consumer kernel, and the fixed-length Ascend 950
producer/consumer pipeline are implemented. Device build, accuracy,
execution-trace and profiling evidence are still pending.

## Reference projects

- `../chunk_gated_delta_rule_fwd_h`: state recurrence, `h`/`v_new` generation,
  final-state handling, and the corresponding tiling processor.
- `../chunk_fwd_o`: inter/intra-chunk output calculation, output layouts, GEMM
  scheduling, and output epilogue.

The public contract exposes only the final `o` and, when requested,
`final_state`. The H-stage `h` and `v_new` values are transient fused-kernel
intermediates: the O stage consumes them directly and they are not materialized
as operator outputs.

## Layout

```text
chunk_fwd_h_o_fused/
|-- docs/                  # confirmed API, design, and development validation
|-- op_host/
|   `-- op_api/            # OpDef, tiling, L0, and aclnn implementation
|-- op_kernel/
|   |-- arch35/            # all Ascend 950-only kernels, constants and layouts
|   |-- epilogue/          # A2 operator-local H/O epilogues
|   `-- gemm/              # A2 operator-local H/O schedulers and kernels
`-- tests/
    `-- pta/               # CPU reference and PTA comparison cases
```

The A2 and A5 natural-exp kernels use separate per-chunk `HReady` and `VReady`
IB synchronization between dedicated H producers and O consumers. This lets O
compute `QK * mask` and `Q * gate @ H_old` before waiting for `V_new`. A5 uses
its architecture-local Catlass O implementation, while its exp2 path keeps the
specialized arch35 O implementation with a temporary all-core handoff barrier.
Both use workspace-backed internal `h`/`v_new`. See `docs/design.md` for the
support boundaries.
