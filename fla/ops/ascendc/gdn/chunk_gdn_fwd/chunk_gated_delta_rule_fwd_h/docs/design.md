# ChunkGatedDeltaRuleFwdH Design

## Scope and dispatch

The existing implementation remains the sole path for chunk64/128. A new Arch35
TilingKey selects the CP kernel only for dense fixed-length chunk32, `K=V=128`, and
`B*HV <= total AIC cores`. `HV` is the recurrent-state head axis. `R` is
`min(ceil(T/32), floor(totalCore/(B*HV)))`; the launch block count is `totalCore` so
every AIC participates in the mode-0 collective. The first `B*HV*R` AICs do work;
remaining AICs join the barrier without accessing workspace. Each `(b,hv)` owns one
contiguous, balanced partition of the sequence chunks.

## Affine summary

For one chunk, let `q[t] = exp(g_last-g[t])` when scalar `g` is present, otherwise
`q[t]=1`. The diagonal decay is `D[k] = exp(g_last) * exp2(gk_last[k])`, omitting
a factor when its gate is absent. With `w` and `u` as the already-prepared inputs:

```text
A_c = diag(D) - k^T @ (q[:, None] * w)
E_c = k^T @ (q[:, None] * u)
h_next = A_c @ h_prev + E_c
```

The public token output is `v_new[t] = u[t] - w[t] @ h_prev`. The internal value
used by the state update is `q[t] * v_new[t]`, so
`h_next = diag(D) @ h_prev + k.T @ (q[:, None] * v_new)`. This matches the existing
FWDH epilogue, which stores unscaled `v_new` while applying scalar-gate row scaling
to the update operand.

Each rank initializes `(M,E)=(I,0)` and composes its chunks in sequence order:
`M <- A_c @ M`, `E <- A_c @ E + E_c`. It publishes one FP32 `(M,E)` record. After
the collective barrier, each rank applies preceding records in rank order to the
sequence initial state, then executes its own chunks from that true incoming state.
The local replay writes the original chunk-start `h`, token `v_new`, and the final
rank writes optional `final_state`.

## Synchronization and workspace

The CP key uses one mixed AIC/AIV launch shape. AIVs build each chunk's `A_c` and `E_c`
with vector multiply/add operations, splitting `A_c` rows and `E_c` columns between
the two sub-blocks. For each local chunk, AIC performs the two MMADs that compose
`(M,E)`; AIV vector-copies the `M` result and adds `E_c` into the `E` result. AIC waits
for both sub-block acknowledgements before reusing scratch. AIC also performs the
sequential prefix MMADs, while AIV initializes and updates their state buffers.
AICs execute `PipeBarrier<PIPE_ALL>()` and join the mode-0 collective using
`CrossCoreSetFlag<0x0, PIPE_FIX>` and `CrossCoreWaitFlag`. A5's 1-AIC:2-AIV handoffs use
mode `0x4`: each AIV uses the local flag ID, while AIC addresses the paired AIV with
`flagId + subBlock * 16`. Affine-ready, MMAD-ready/applied, and state-ready each use
producer/consumer handshakes; AIV drains MTE3 before publishing its affine matrices.
After applying the prior-rank prefix, AIV initializes the incoming state from
`initial_state` (or zero), adds each prior rank's affine bias to the prefix MMAD result,
and copies the final ping-pong buffer into the replay state when needed. AIC publishes
the completed state to its paired AIVs.
The two AIV sub-blocks split the 128 value columns and perform local chunk replay,
writing disjoint columns of `h`, `v_new`, and `final_state`. Inactive AIC/AIV ranks
also participate in their respective synchronization phases. This remains one kernel
launch; mode 0 is the only cross-rank collective barrier.

Workspace starts after the existing library and 16 MiB prefix reservation and ends
with the existing 16 MiB suffix reservation:

```text
summary bytes = activeRanks * 2 * 128 * 128 * sizeof(float) = activeRanks * 128 KiB
scratch bytes = activeRanks * 3 * 128 * 128 * sizeof(float) = activeRanks * 192 KiB
CP total      = lib workspace + 32 MiB + activeRanks * 320 KiB
```

The scratch contains two FP32 128x128 matrix buffers and one 64 KiB operand buffer.
The operand area stores the chunk-local `E_c` matrix. AIV uses the repository's
`AscendC::Exp` and V-to-scalar event pattern to prepare gate factors, then writes `A_c`
and `E_c` to their scratch regions. The paired mode-0x4 handshakes order affine
publication, both MMAD results, and vector updates.
After the global barrier AIC reuses matrix scratch as the two KxV prefix-state buffers,
then AIV reuses it as the local state source. CP offsets and allocation are independent
from legacy allocations.

For comparison, the legacy path retains its existing allocation formula: per AIC,
two `chunkSize*V` FP32 buffers, two `K*V` FP32 buffers, and (when `gk` is present)
two `chunkSize*K` FP32 buffers, plus sequence/chunk metadata, library workspace, and
the two 16 MiB reservations.

## Risks

The CP path uses AIV UB vector operations for gate preparation and local replay, and
Cube MMAD for rank matrix composition and prefix application. Chunk affine-matrix
assembly and the affine-bias accumulation still contain scalar AIC loops and require
follow-up profiling/optimization. Device build, correctness, sanitizer, and
performance evidence are required before treating this path as production-ready.
