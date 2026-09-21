# Kernel implementation

The fused entry is `chunk_fwd_h_o_fused.cpp`. It contains one tiling-key
dispatch path and selects either `chunk_fwd_h_o_fused_a2.hpp` or
`arch35/chunk_fwd_h_o_fused_a5.hpp` at compile time. The selected kernel
headers define `CATLASS_ARCH` as 2201 or 3510, so one source file builds the
matching architecture implementation. Supporting H/O kernels, schedulers and
epilogues are operator-local copies adapted to the fused producer/consumer
schedule; they do not include sibling operator or private `internal` paths.

The fixed-length pipeline uses `floor(activeCoreNum / 2)` double-buffered H
producers followed by paired O consumers, full-task handoff
workspace, and per-chunk `IBSet<false>`/`IBWait<false>` synchronization. Four
ready events provide `HReady` and `VReady` for each of the two task lanes. All
chunks of one task lane reuse the corresponding H/V slots: `IBSet` waits until
its GM event slot is zero before setting it to one, and the matching `IBWait`
clears the slot after consumption, so slot reuse needs no separate O-to-H ACK.
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
