# ChunkFwdHOFused

`ChunkFwdHOFused` is the operator project reserved for fusing the state update
performed by `ChunkGatedDeltaRuleFwdH` with the output calculation performed by
`ChunkFwdO`.

## Status

The host-side operator definition, combined tiling, L0/aclnn APIs, and the
fixed-length Atlas A2 producer/consumer kernel are implemented. Device build,
accuracy, execution-trace and profiling evidence are still pending.

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
|   |-- epilogue/          # operator-local H/O epilogues
|   `-- gemm/              # operator-local H/O schedulers and kernels
`-- tests/
    `-- pta/               # CPU reference and PTA comparison cases
```

The first kernel uses full-chunk `h`/`v_new` workspace and per-chunk IB
synchronization. See `docs/design.md` for its current support boundary.
