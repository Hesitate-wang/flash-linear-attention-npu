# FWDH CP32 Development Validation

## Scope

New dense Ascend 950 CP path: chunk32, K=V=128, B*HV<=AIC core count. Existing
chunk64/128 paths are unchanged. No Ascend/CANN environment is configured in the
current development shell (`ASCEND_HOME_PATH` is unset), so device build and execution
are pending.

## Design-to-code trace

| Requirement | Implementation | Evidence/status |
| --- | --- | --- |
| Chunk32 support and rejection | ACLNN validation and Arch35 tiling | Implemented; device/API tests pending |
| CP TilingKey and block/rank mapping | Host tiling processor; kernel entry key 3 | Implemented; compiled dispatch evidence pending |
| Affine rank summaries and prefix replay | `arch35/gdn_fwd_h_cp_kernel.hpp`; AIV gate preparation, Cube rank composition/prefix MMAD, AIV UB local replay | Implemented structurally; tiny-matrix serial-vs-CP formula test passes, device comparison pending |
| One global AIC barrier and AIC/AIV handoffs | two per-sub-block mode-2 operand-ready flags with consumed acknowledgements, mode-0 collective, then mode-2 state-ready handoff | Runtime validation pending |
| Separate CP workspace | summary, two matrix scratch buffers, and gate-scaled operand staging | Formula: 320 KiB per active rank, plus library space and 32 MiB reservations |
| Legacy regression | old tiling branch retained | Build and chunk64/128/varlen regression pending |

## Required device validation

- Build the Ascend 950 OPP and verify TilingKey 3 plus all legacy keys compile.
- Compare h, v_new, and final_state against the current CPU reference for R=1, R>1,
  non-even chunk partition, tail chunk, optional initial/final state, state_v_first,
  g-only, gk-only, g+gk, and supported dtype combinations.
- Verify ineligible chunk32 fails for non-950, varlen, non-128 dimensions, and
  B*HV>totalCore; verify chunk64/128 and existing variable-length cases still pass.
- Run race/memory/synchronization sanitizers and profile Task Duration. The current
  scalar-GM AIC/AIV implementation must not be accepted as a performance improvement
  without measurement.
- Verify both AIV sub-blocks cover the V dimension exactly once and that inactive
  ranks participate in the mode-2 handoff without touching the CP scratch region.
