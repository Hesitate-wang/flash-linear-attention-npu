# Ascend 950 implementation

`chunk_fwd_h_o_fused_a5.hpp` provides the common
`GDN::RunChunkFwdHOFused<InputT, TileShapes>` entry selected by the single
source-file dispatcher. It is compiled only when `CHUNK_FWD_HO_ARCH35` is
defined; its H/O kernel headers reject other architecture selections and define
`CATLASS_ARCH 3510`.
H writes internal `h` and `v_new` tensors to user workspace.
Natural-exp mode uses the architecture-local Catlass O kernel under
`arch35/gemm/kernel` (FP16/BF16, V128/V256, chunk 64/128), while exp2 mode
keeps the specialized `ChunkFwdOA5` path. H scratch, handoff tensors,
natural-exp O scratch, and exp2 A-prime scratch have disjoint host-generated
offsets.

A5-only exp2 constants and tiling projection are declared in
`chunk_fwd_o_a5_constants.h` and `chunk_fwd_o_a5_struct.h`. The shared A5 UB
contract is declared in `chunk_fwd_h_o_fused_ub_layout.h`.
