# ChunkFwdHOFused API

## Semantics

`ChunkFwdHOFused` executes the state recurrence of
`ChunkGatedDeltaRuleFwdH`, then passes the resulting `h` and `v_new` directly
to the `ChunkFwdO` calculation. These intermediate values are internal to the
fused execution and are not operator outputs.

## Inputs

The tensor inputs retain the FwdH order, with `q` added for the O stage:

1. `k`: required `[B, HK, T, K]`, FP16/BF16.
2. `w`: required `[B, HV, T, K]`, same dtype as `k`.
3. `u`: required `[B, HV, T, V]`, same dtype as `k`.
4. `g`: required `[B, HV, T]`, FP16/BF16/FP32. It is required because the O
   stage consumes the scalar gate.
5. `gk`: optional `[B, HV, T, K]`, same gate dtype as `g`.
6. `initial_state`: optional `[N, HV, K, V]`, or `[N, HV, V, K]` when
   `state_v_first=true`.
7. `q`: required `[B, HK, T, K]`, same shape and dtype as `k`.
8. `cu_seqlens`: optional INT64 array.
9. `chunk_indices`: optional flattened INT64 `(sequence, chunk)` pairs.

`cu_seqlens` and `chunk_indices` must be both present or both absent.

The current device implementation accepts fixed-length input only, so both
optional index inputs must currently be absent.

## Attributes

- `output_final_state: bool`
- `chunk_size: int64`, currently 64 or 128
- `scale: double`
- `use_exp2: bool`; the current Atlas A2 implementation requires `false`
- `state_v_first: bool`
- `output_layout: string`, one of `BNSD`, `BSND`, `TND`, or `NTD`

The current exp path accepts `BNSD/NTD`; `NTD` requires physical `B=1`.
`BSND/TND` remain reserved for the future exp2 implementation.

## Outputs

1. `o`: required; shape selected by `output_layout`, matching `ChunkFwdO`.
2. `final_state`: optional `[N, HV, K, V]`, with the last dimensions swapped
   when `state_v_first=true`. It is present only when
   `output_final_state=true`.

The Python adapter returns `(o, final_state)`, where `final_state` is `None`
when it is not requested. The L0 implementation always presents `[K,V]` state
layout to the fused kernel; the aclnn layer performs the required state
input/output transposes.

## Current execution constraints

The functional kernel is registered for Atlas A2 (`ascend910b` and
`ascend910_93`). Let `P = B * HV`; tiling requires `2 * P` to be strictly less
than the available AIC core count. Shapes outside that condition currently
fail rather than falling back to a sequential fused schedule.

Ascend 950 is registered as a compilation skeleton. Its host and kernel entry
points build as part of the operator package, but A5 tiling deliberately fails
with a not-implemented error; A5 is not yet part of the executable API support
domain.
