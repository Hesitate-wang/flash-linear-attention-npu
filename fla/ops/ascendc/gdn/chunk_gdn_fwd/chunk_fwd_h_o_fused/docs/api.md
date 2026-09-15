# ChunkFwdHOFused API

## Semantics

`ChunkFwdHOFused` executes the state recurrence of
`ChunkGatedDeltaRuleFwdH`, then passes the resulting `h` and `v_new` directly
to the `ChunkFwdO` calculation. The intermediate tensors are outputs of the
fused operator but are not duplicated as O-stage inputs.

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

## Attributes

- `output_final_state: bool`
- `chunk_size: int64`, currently 64 or 128
- `scale: double`
- `use_exp2: bool`, used by the O stage
- `state_v_first: bool`
- `output_layout: string`, one of `BNSD`, `BSND`, `TND`, or `NTD`

`TND` and `NTD` require physical `B=1`. The current exp2 path accepts
`BSND/TND`; the exp path accepts `BNSD/NTD`.

## Outputs

1. `h`: `[B, HV, num_chunks, K, V]`, with the last dimensions swapped when
   `state_v_first=true`.
2. `v_new`: `[B, HV, T, V]`.
3. `final_state`: optional `[N, HV, K, V]`, with the last dimensions swapped
   when `state_v_first=true`.
4. `o`: shape selected by `output_layout`, matching `ChunkFwdO`.

The L0 implementation always presents `[K,V]` state layout to the fused kernel;
the aclnn layer performs the required input/output transposes.
