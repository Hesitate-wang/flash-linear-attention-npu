# Kernel implementation

The fused entry is `chunk_fwd_h_o_fused.cpp`. Its architecture header defines
exactly one of `CHUNK_FWD_HO_ARCH_A2` and `CHUNK_FWD_HO_ARCH35`. The dispatcher
then includes only `chunk_fwd_h_o_fused_a2.hpp` or
`arch35/chunk_fwd_h_o_fused_a5.hpp`. Architecture entry and kernel headers
reject the wrong macro at preprocessing time. A5-only kernels, tiling
projections, constants and UB layout live under `arch35/`; root-level kernel
files are common or A2 implementations.

The fixed-length pipeline uses one H producer and one O consumer per
`(batch, value-head)` task, with `activeCoreNum == 2 * producerCoreNum`, full-task
handoff workspace, and per-chunk `IBSet<false>`/`IBWait<false>` synchronization. Two
ready events provide `HReady` and `VReady` for each task. All chunks of one task
reuse the corresponding H/V slots: `IBSet` waits until
its GM event slot is zero before setting it to one, and the matching `IBWait`
clears the slot after consumption, so slot reuse needs no separate O-to-H ACK.
The IB implementation's internal `PipeBarrier<Pipe_all>` orders the GM data
transfer and ready publication, so producer call sites do not add a separate
MTE3 barrier. The initial `HReady` is published for its batch/head task without
waiting for the other tasks assigned to the same producer core.
In MIX mode the IB calls execute on the AIV lanes and use the logical AIV index
space required by the API. After `HReady`, the O-internal reverse generation of
`cube1Done` lets the AIC compute `Q * gate @ H_old` while the AIV computes
`QK * mask`; `VReady` is consumed only before releasing `AttnMask @ V_new`.

For Ascend 950, the entry selects architecture-local copies of the established
arch35 H and O implementations. Separate mixed cores run H producers and O
consumers; natural-exp handoff uses per-chunk `IBSet/IBWait`. The A5 exp2 O
path temporarily retains an all-core handoff barrier. The exp2 path supports
BF16 data, BF16/FP32 gates, `chunk=64`, `K=V=128`, `HV/HK` in `[1,4]`, and
BSND/TND output.
