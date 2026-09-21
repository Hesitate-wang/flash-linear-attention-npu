# Ascend 950 block scheduler

Contains independent arch35 block schedulers for FwdH and natural-exp FwdO.
The FwdO scheduler owns the A5 producer/consumer core mapping, chunk-affinity
iteration, workspace addressing, and ping-pong stage state.
