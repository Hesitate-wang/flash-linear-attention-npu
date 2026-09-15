# Kernel implementation staging

The fused kernel will combine the state recurrence from
`chunk_gated_delta_rule_fwd_h` and the output calculation from `chunk_fwd_o`.

The implementation must define the producer/consumer synchronization before
code is added. `h` and `v_new` use the fused operator output buffers and are
passed directly to the O phase. Shared and Ascend 950 specializations belong in
the prepared subdirectories.
