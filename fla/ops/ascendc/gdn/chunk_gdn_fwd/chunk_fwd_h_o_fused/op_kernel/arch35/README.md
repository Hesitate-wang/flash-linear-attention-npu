# Ascend 950 implementation

`chunk_fwd_h_o_fused_a5.hpp` composes operator-local copies of the established
arch35 FwdH and FwdO paths in one kernel launch. FwdH writes internal `h` and
`v_new` tensors to user workspace. After an all-core stage boundary, FwdO
consumes those tensors and writes the public output. H scratch, handoff tensors,
and O A-prime scratch have disjoint host-generated offsets.
