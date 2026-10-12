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

When `floor(P/2) < batch * v_heads <= P`, A5 natural-exp V=128 without `gk`
dispatches to the separate `gdn_fwd_ho_fused_kernel.hpp` MODE2 implementation.
It owns one task per MIX core, uses single-slot scratch, and coordinates the
local AIC/AIV stages through mode-`0x2` flags. MODE1 kernels and their mode-`0x0`
H/O handoff remain unchanged.

A5-only exp2 constants and tiling projection are declared in
`chunk_fwd_o_a5_constants.h` and `chunk_fwd_o_a5_struct.h`. The shared A5 UB
contract is declared in `chunk_fwd_h_o_fused_ub_layout.h`.
