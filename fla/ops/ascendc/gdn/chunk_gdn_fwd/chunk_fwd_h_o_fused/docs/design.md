# ChunkFwdHOFused design

Rules version: `V2`

## 1. Scope

The first device implementation targets the fixed-length Atlas A2 exp path.
It fuses the state recurrence and output calculation in one kernel launch while
keeping `h` and `v_new` internal. Varlen and exp2 are rejected by tiling until
they have their own reviewed scheduling and synchronization path.

Ascend 950 has a registration-only arch35 skeleton. Compile-time architecture
selection prevents the A5 compiler from including the A2 Catlass implementation.
The A5 kernel entry is intentionally empty and its host tiling route returns an
explicit not-implemented error, so no invocation can report success with
uninitialized outputs.

## 2. Core mapping and execution order

Let `P = B * HV` and `NC = ceil(T / chunk_size)`. The pipeline is selected only
when `2 * P < physical_aic_core_count`, and launches `blockDim = 2 * P` mixed
cores in batch schedule mode.

- mixed cores `[0, P)` are H producers; producer `p = b * HV + hv` owns all
  chunks of one `(b, hv)` pair in ascending chunk order;
- mixed cores `[P, 2P)` are O consumers; consumer `P + p` owns the same pair
  and also visits its chunks in ascending order;
- the corresponding AIV0/AIV1 lanes retain the established split-row work;
  because IB synchronization is a SIMD-side API, each consumer AIV waits on
  the matching producer AIV index in the MIX logical-index space.

All active AIVs first zero their own IB event slots. One startup `SyncAll` makes
that initialization visible before the core groups diverge. For every chunk,
H publishes each completed AIV slice of the current `h` and `v_new` through
`IBSet<false>`. After O's two AIVs complete their matching `IBWait<false>`, they
reuse the already-consumed `cube1Done` stream flag in the reverse direction.
The O AIC waits for the aggregated pair before its first GM-to-L1 read of the
complete `h`/`v_new` tiles. `vec1Done` independently protects the later
`attnMask` input. Two IB event IDs are selected by `chunk_index % 2`. There is
no reverse free signal because every chunk has distinct handoff storage.

The H implementation replaces its former per-wave global barriers in pipeline
mode with its existing per-mixed-core CrossCore flags. Thus consumer cores do
not have to enter H, and H/O execute concurrently after the startup barrier.

## 3. Workspace ownership

All serialized offsets are relative to `AscendC::GetUserWorkspace(workspace)`;
the total allocation returned to the framework is `system_workspace +
user_workspace_bytes`. Every region starts on a 512-byte boundary.

| Region | Shape/capacity | Owner and lifetime |
| --- | --- | --- |
| handoff H | `[B, HV, NC, K, V]`, input dtype | H writes one chunk; paired O reads it after IB wait; retained to kernel end |
| handoff v_new | `[B, HV, T, V]`, input dtype | H writes one chunk; paired O reads it after IB wait; retained to kernel end |
| IB events | `2 events * 2 AIV/core * blockDim * 8` int32 words | one eight-word slot per MIX logical AIV and event ID, as required by the IB API |
| H v/v-update scratch | `P * 2 * C * V` FP32 each | private to producer mixed core and ping/pong stage |
| optional H k-decay scratch | `P * 2 * C * K` FP32 | allocated only when `gk` is present |
| H state-update scratch | `P * 2 * K * V` FP32 | private to producer mixed core and ping/pong stage |
| O v/h scratch | `P * 2 * C * V` FP32 each | private to consumer mixed core and ping/pong stage |
| O attention/aftermask scratch | `P * 2 * C * C` FP32 each | private to consumer mixed core and ping/pong stage |
| O causal mask | `C * C` bytes | read-only after construction |

The v_new allocation is immediately followed by the IB area, and both H and O
consume its explicit serialized offset rather than relying on pointer-relative
layout. Full-chunk handoff storage is intentional for the first profiling
version. A two-slot handoff buffer is considered only if device profiling
shows poor L2 hit rate.

## 4. ABI and implementation ownership

The host macro type and kernel-side plain mirror have identical field order and
are size-checked by tiling. The prefix used by the local H implementation is
also compile-time checked with `offsetof`. H/O schedulers, epilogues and kernels
needed by this operator are owned under this operator's `op_kernel` tree. No
sibling operator or `internal` private header is included or linked.

The supported tiling keys are `1` for V=128 and `2` for V=256. Both use exp
mode and the same producer/consumer protocol.

## 5. Correctness and performance gates

Correctness requires: exact `(b,hv,chunk)` address ownership, each H AIV signal
after publishing its own slice, both O AIV waits completing before the O AIC
performs a dependent read, a tail-safe final chunk, and optional final-state
behavior identical to standalone H. Validation must cover at least two chunks
so event and reverse-flag reuse are exercised.

After accuracy passes, profiling records L2 hit rate and GM traffic for the
handoff regions. Double buffering is not introduced unless that evidence shows
the full-chunk layout misses the intended L2 reuse.
