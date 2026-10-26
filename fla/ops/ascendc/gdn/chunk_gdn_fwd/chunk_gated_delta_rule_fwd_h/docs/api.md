# ChunkGatedDeltaRuleFwdH API

## Existing contract

The public ACLNN/Python signature is unchanged. Inputs remain head-major BNSD:
`k=[B,HK,T,K]`, `w=[B,HV,T,K]`, `u=[B,HV,T,V]`, optional `g=[B,HV,T]` and
`gk=[B,HV,T,K]`. At least one gate is required. Outputs and state layout retain the
existing `state_v_first` behavior.

## Chunk-size support

- `chunk_size=64/128`: existing A2/A3/A5 dense and variable-length paths.
- `chunk_size=32`: Ascend 950 only; dense fixed-length only; `K=V=128`. It uses the
  same Cube/Vector path as the other chunk sizes and splits the value dimension across
  cores (at least 64 columns per core). Inputs outside this domain are rejected by
  ACLNN/host tiling.
- Gate combinations and supported input, gate, and state dtypes are unchanged.

The API adds no arguments or output changes.
