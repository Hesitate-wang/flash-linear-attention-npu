# Ascend 950 implementation

`chunk_fwd_h_o_fused_a5.hpp` composes operator-local H and O paths in one
kernel launch. H writes internal `h` and `v_new` tensors to user workspace.
After an all-core stage boundary, natural-exp mode uses the generic Catlass O
path (FP16/BF16, V128/V256, chunk 64/128), while exp2 mode keeps the specialized
arch35 O path. H scratch, handoff tensors, generic O scratch, and exp2 A-prime
scratch have disjoint host-generated offsets.
