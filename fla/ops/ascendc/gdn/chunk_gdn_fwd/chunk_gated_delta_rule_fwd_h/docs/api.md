# ChunkGatedDeltaRuleFwdH API

## Existing contract

The public ACLNN/Python signature is unchanged. Inputs remain head-major BNSD:
`k=[B,HK,T,K]`, `w=[B,HV,T,K]`, `u=[B,HV,T,V]`, optional `g=[B,HV,T]` and
`gk=[B,HV,T,K]`. At least one gate is required. Outputs and state layout retain the
existing `state_v_first` behavior.

## Chunk-size support

- `chunk_size=64/128`: existing A2/A3/A5 dense and variable-length paths.
- `chunk_size=32`: Ascend 950 only; dense fixed-length only; `K=V=128`; `B*HV` must
  not exceed the total AIC core count. Inputs outside this domain are rejected by
  ACLNN/host tiling and are not sent to the legacy kernel.
- Gate combinations and supported input, gate, and state dtypes are unchanged.

The API adds no arguments or output changes.
