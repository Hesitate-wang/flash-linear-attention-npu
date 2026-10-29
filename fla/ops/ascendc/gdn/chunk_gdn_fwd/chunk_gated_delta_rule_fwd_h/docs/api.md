# ChunkGatedDeltaRuleFwdH API

## Existing contract

The public ACLNN/Python signature is unchanged. Inputs remain head-major BNSD:
`k=[B,HK,T,K]`, `w=[B,HV,T,K]`, `u=[B,HV,T,V]`, optional `g=[B,HV,T]` and
`gk=[B,HV,T,K]`. At least one gate is required. `h` is stored NT-first as
`[B, N_c, H_v, K, V]` (varlen packed: `[1, total_N_c, H_v, K, V]`);
`initial_state`/`final_state` keep the existing `state_v_first` interpretation.

## Chunk-size support

- `chunk_size=64/128`: existing A2/A3/A5 dense and variable-length paths.
- `chunk_size=32`: Ascend 950 only; dense fixed-length only; `K=V=128`. Inputs
  outside this domain are rejected by ACLNN/host tiling.
- Ascend 950 routing: dense, g-only, `K=V=128`, `vSplits==1`, `ceil(T/chunk) >= 2`,
  a tail chunk of at least 16 tokens and `B*HV <= AIC cores` select the chunk-parallel
  path (TilingKey 3); all other
  inputs keep the chain-parallel / V-split path (TilingKey 1/2). Set
  `GDN_FWD_H_CP=0` to force the chain-parallel path for A/B or rollback. The
  routing is a device-internal choice and does not change the tensor arguments or
  their layouts relative to TilingKey 1/2.
- Gate combinations and supported input, gate, and state dtypes are unchanged;
  the chunk-parallel path accepts scalar `g` only and falls back otherwise.

The API adds no arguments or output tensors.
