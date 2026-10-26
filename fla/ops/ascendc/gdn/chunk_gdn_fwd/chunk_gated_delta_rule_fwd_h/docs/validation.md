# FWDH CP32 Development Validation

## Scope and environment

Dense Ascend950 chunk32, K=V=128, B*HV<=AIC core count; chunk64/128 retain their
existing dispatch. Current development host is macOS without a configured CANN
installation. No device compilation, execution, sanitizer, or performance result
is available. This file remains a development record pending formal acceptance.

Development reference: CHUNK_DEPENDENT V1, source commit
`a14961e9dfe8c8264768d67fb447101873da4ecf`. Reused Cube/Vector separation,
producer/consumer handoffs, and tail padding. Task mapping follows the user's
explicit rank-per-core scheme: all cores cooperate on each sequential batch/head task.

## Design-to-code trace

| Requirement | Implementation | Evidence/status |
| --- | --- | --- |
| CP dispatch | Existing TilingKey 3 and useCpPath branch | Retained; compiled dispatch evidence pending |
| Rank mapping | cpRankCount=CoreNum; ceil contiguous chunk partition; sequential task loop | Source implemented; device mapping evidence pending |
| Chunk affine | ColumnMajor K view; Cube KᵀW/KᵀU; Vec diagonal subtraction | Source implemented; build and intermediate comparison pending |
| Saved prefixes | Per-chunk exclusive B/A; per-rank inclusive B/A | CPU formula comparison passes; device comparison pending |
| Prior-rank prefix | Sequential Bprior@P and Bprior@Q+Aprior with separate temporaries | CPU formula comparison passes; device comparison pending |
| Replay | Cube global affine/H0 materialization and W@H; Vec additions/subtraction and output cast | Source implemented; device comparison pending |
| Final state | Last rank materializes its inclusive summary, including empty rank | Covered by CPU formula test; device comparison pending |
| Workspace | 2*(R+C) matrix slots for summaries/records; 14*R scratch slots; 64 KiB per slot | Host/kernel formulas aligned in source; device bounds check pending |
| GM/UB events | Load uses V_MTE2/MTE2_V; Store uses V_MTE3/MTE3_V; scalar gate copied through UB | Event pairing added; target compile and device validation pending |
| Cross-core sync | mode 2 ready/done; mode 0 AIC inter-block barrier ID 9 after summary and replay | Barrier ID follows CATLASS; mode-2 flag counts and repeated-task lifecycle remain unverified; do not treat as deadlock-safe |
| Legacy regression | Non-CP tiling and kernels retained | Build and chunk64/128/varlen regression pending |

UB allocation is 4096 bytes per AIV. Operands and results use separate GM slots.
No asynchronous overlap or measured performance improvement is claimed.

## Local validation

Run:

```
python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests/pta/test_cp_affine.py
git diff --check
```

The CPU tests compare affine prefix replay with the direct serial recurrence for
none/g/gk/both gate modes, optional initial state, tail lengths, rank counts
1/2/3/8 and empty ranks. The no-gate case is mathematical coverage; the public
API continues to require at least one gate. These checks do not validate the
Catlass layout implementation, CANN API compilation, device memory visibility,
or hardware flag behavior.

## Required device validation

1. Build the Ascend950 single-op package, install it, and verify package/runtime
   consistency using the repository development procedure. Confirm key 3 and all
   legacy keys compile.
2. Compare Stage 1 Bc/Ac, saved chunk prefixes and rank summaries against CPU
   intermediates, then compare Stage 2 rank prefix and replay states.
3. Compare h/v_new/final_state with the established independent CPU reference,
   using the operator ATK README and tests/atk/README.md. Include R=1, R>C,
   C not divisible by R, tail chunk, optional initial/final state, both
   state_v_first modes, g-only/gk-only/both and supported dtype combinations.
4. Exercise multiple batch/head tasks in one invocation. Verify no next-task
   overwrite occurs before both AIC and paired AIVs pass the final barrier.
   Record rank, chunk, buffer generation, and ready/done steps in a separate
   diagnostic workspace if runtime tracing is needed; tracing must add no sync.
5. Run memory/race/synchronization sanitizers and repeated executions with
   diagnostics disabled. Terminate a kernel test that remains active for 60s,
   per repository policy. Previous CPDBG messages were removed with the old
   algorithm and must not be used as evidence for this implementation.
6. Verify rejection of unsupported chunk32 inputs, and regress chunk64/128 plus
   existing variable-length paths. Measure Task Duration under matched conditions
   before making any performance claim.

## Performance tracking

| Model case | Round / status | Expected critical path | Decision / variable | Precision | Before/after performance | Bottleneck change | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| User rank-prefix scheme | Pending device measurement | Sequential task loop, FP32 128-square compositions, GM intermediates and handshakes | All-core ranks and saved exclusive prefixes | CPU formula only | Unmeasured | Unknown | Device profiling pending |
