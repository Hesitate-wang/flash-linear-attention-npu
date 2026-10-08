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
handoff workspace, and per-chunk mode-0 `CrossCoreSetFlag/CrossCoreWaitFlag`
rendezvous. Flags 12 and 13 provide `HReady` and `VReady`; local AIC/AIV
pipeline flags remain in the mode-2 0..9 range. On the A5 V128 preload path,
H AIV publishes VReady only after the V_new GM write and O AIV waits for it
before publishing local vec1Done to Cube3. The legacy IB workspace remains allocated for
ABI compatibility but is not initialized or accessed by natural-exp H/O.
In MIX mode the IB calls execute on the AIV lanes and use the logical AIV index
space required by the API. After `HReady`, the O-internal reverse generation of
`cube1Done` lets the AIC compute `Q * gate @ H_old` while the AIV computes
`QK * mask`; `VReady` is consumed only before releasing `AttnMask @ V_new`.

For Ascend 950, the entry selects architecture-local copies of the established
arch35 H and O implementations. Separate mixed cores run H producers and O
consumers; natural-exp handoff uses per-chunk mode-0 `H_READY/V_READY`
`CrossCoreSetFlag/CrossCoreWaitFlag`. The A5 exp2 O
path temporarily retains an all-core handoff barrier. The exp2 path supports
BF16 data, BF16/FP32 gates, `chunk=64`, `K=V=128`, `HV/HK` in `[1,4]`, and
BSND/TND output.
