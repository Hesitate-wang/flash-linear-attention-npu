# ChunkFwdHOFused

`ChunkFwdHOFused` is the operator project reserved for fusing the state update
performed by `ChunkGatedDeltaRuleFwdH` with the output calculation performed by
`ChunkFwdO`.

## Status

The host-side operator definition, combined tiling, L0 API, and two-stage aclnn
API are implemented. The device kernel and tests are still pending.

## Reference projects

- `../chunk_gated_delta_rule_fwd_h`: state recurrence, `h`/`v_new` generation,
  final-state handling, and the corresponding tiling processor.
- `../chunk_fwd_o`: inter/intra-chunk output calculation, output layouts, GEMM
  scheduling, and output epilogue.

The public contract exposes only the final `o` and, when requested,
`final_state`. The H-stage `h` and `v_new` values are transient fused-kernel
intermediates: the O stage consumes them directly and they are not materialized
as operator outputs.

## Planned layout

```text
chunk_fwd_h_o_fused/
|-- docs/                  # confirmed API, design, and development validation
|-- op_host/
|   `-- op_api/            # OpDef, tiling, L0, and aclnn implementation
|-- op_kernel/
|   |-- arch35/            # Ascend 950 implementation
|   |-- epilogue/          # shared epilogue policies
|   `-- gemm/              # shared GEMM schedulers and kernels
`-- tests/
    `-- pta/               # CPU reference and PTA comparison cases
```

The project is registered for host compilation. A runnable build additionally
requires the pending `op_kernel/chunk_fwd_h_o_fused.cpp` entry.
