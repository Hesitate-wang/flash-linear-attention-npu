# Ascend 950 implementation

`chunk_fwd_h_o_fused_a5.hpp` provides the common
`GDN::RunChunkFwdHOFused<InputT, TileShapes>` entry selected by the single
source-file dispatcher. Its H/O kernel headers define `CATLASS_ARCH 3510`.
H writes internal `h` and `v_new` tensors to user workspace.
Natural-exp mode uses the architecture-local Catlass O kernel under
`arch35/gemm/kernel` (FP16/BF16, V128/V256, chunk 64/128), while exp2 mode
keeps the specialized `ChunkFwdOA5` path. H scratch, handoff tensors,
natural-exp O scratch, and exp2 A-prime scratch have disjoint host-generated
offsets.
