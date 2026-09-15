# ChunkFwdHOFused host design

Rules version: `V2`

## Scope

This revision covers host-side composition only. Kernel Stage scheduling,
cross-core synchronization, and resident storage will be completed before the
device implementation is added.

## Host composition

The fused launch retains the FwdH tensor order and adds `q`. Duplicate FwdO
inputs (`k`, `g`, `cu_seqlens`, and `chunk_indices`) and the FwdO `v/h` inputs
are removed. FwdO consumes the `v_new/h` buffers produced by FwdH.

The L0 layer derives logical B/H/T/K/V attributes from the tensor descriptors.
For `state_v_first=true`, aclnn transposes initial state into `[K,V]`, allocates
an internal `[K,V]` `h`, and transposes state outputs back after execution. This
keeps the kernel-side H-to-O handoff in one canonical state layout. As in the
reference FwdH operator, `state_v_first` is an aclnn adapter argument rather than
an L0/kernel attribute.

## Tiling layout

The operator owns one flat `ChunkFwdHOFusedTilingData` ABI containing common
shape/mode fields plus separately named H-stage and O-stage workspace offsets.
The host macro type and kernel-side plain mirror are size-checked before launch.
No type, processor, or header from another operator is part of this ABI.

Workspace is disjoint by construction: H regions are allocated first and O
regions continue from the resulting end offset. On A5, `aPrime` is likewise
allocated after all H regions.

## Tiling key

The operator owns four explicit keys: V128/V256 crossed with exp/exp2 mode.

## Pending kernel design

The kernel design must define whether the two phases run sequentially with a
global completion barrier or use a producer/consumer schedule. Until that is
reviewed, the host contract intentionally does not claim an overlap strategy.
