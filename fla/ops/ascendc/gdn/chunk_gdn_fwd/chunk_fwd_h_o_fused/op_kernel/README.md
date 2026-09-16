# Kernel implementation

The fused entry is `chunk_fwd_h_o_fused.cpp`. Its supporting H/O kernels,
schedulers and epilogues are operator-local copies adapted to the fused
producer/consumer schedule; they do not include sibling operator or private
`internal` paths.

The initial implementation supports fixed-length Atlas A2 exp mode. It uses
`B * HV` H producers followed by `B * HV` O consumers, full-chunk handoff
workspace, and per-chunk `IBSet<false>`/`IBWait<false>` synchronization.
In MIX mode the IB calls execute on the AIV lanes and use the logical AIV index
space required by the API. Each O AIV acknowledges its completed wait through
the reverse generation of the current `cube1Done` flag; the O AIC aggregates
both acknowledgements before reading the complete H/V tiles.
