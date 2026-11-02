# Arch35 GDN Chunk64 Fused Entry Source Analysis

## Status

This note records the earlier Phase6 composition analysis, not the current implementation status. Development now extends the existing internal OpDef and launcher (public ACLNN unchanged) and adds a distinct key-64 POD. The new adapter uses standalone arch35 H/O implementations rather than the mega H/O helpers analyzed below, to preserve useExp2/stateVFirst and BSND output. Device compilation and validation have not been run on this macOS host.

## Candidate composition found in source

The existing arch35 implementation includes `internal/gated_delta_rule_state_update_output/chunk_gated_delta_rule_state_update_output.cpp` into the same translation unit. That file defines internal-linkage helpers `DispatchFwdH` and `DispatchFwdO`, which can run the already-implemented H and O kernels when supplied their own H/O tilings and workspace. A serial H/O composition can use `HoPipelineContext{}` and a launch-wide synchronization between H and O, rather than enabling the existing H/O producer-consumer suffix pipeline.

This is not yet a full fused GDN entry. The ACLNN composite calls a separate Prepare operator before the main GDN op. The Prepare kernel expects `ChunkGatedDeltaRuleFwdStageTilingData`, while the Phase6 GDN entry expects a different combined H/O tiling plus state-output and coefficient trailers. Calling Prepare `Init/Process` inside the current entry requires constructing and serializing the Prepare POD in the new host tiler, reserving its per-core Y workspace, selecting the correct GM inputs/outputs and attributes, and defining a participant-complete Prepare-to-H/O visibility boundary. The existing Phase6 entry cannot be reused as the wrapper because it always runs coefficient generation and recompute before H/O.

## MIX index mapping constraints

Prepare's implementation maps AIV `GetBlockIdx()` to the AIC group using `/2`, with `GetSubBlockIdx()` identifying one of the two AIV participants. The Prepare host tiler uses the AIC core count for `blockDim`. In contrast, Phase6 explicitly normalizes AIV `GetBlockIdx()` by `GetSubBlockNum()` to obtain `coreGroup`. These are distinct APIs/launch views and must not be unified by assuming the same raw AIV index interpretation. Any new entry must retain `KERNEL_TYPE_MIX_AIC_1_2`, pass the actual launch block count expected by both phase implementations, and prove that all AIC and AIV participants enter every global barrier.

Prepare's CrossCore flag IDs and its AIV1 offset are intra-MIX-group stage events; they do not publish Prepare outputs across independent producer/consumer groups. They do not replace a Prepare-to-H/O global handoff.

## H/O source interface constraints

The source-level H/O helpers are translation-unit-local. H consumes K, W, U, FP32 chunk-local gate cumsum, optional initial state and metadata, then writes H, VNew and optional final state. O consumes Q, K, VNew, H, the same gate cumsum, and explicit O tiling including scale. Both complete helper dispatches must be run; manually selecting only their AIC or AIV portions would invalidate internal synchronization and fixed hard-event lifecycles.

The H and O mega tilings are separate plain structs. H uses `GdnMegaArch35FwdHTilingData`; O uses `GdnMegaArch35FwdOTilingData`, copied to local memory before dispatch. Their workspace recipes are different. Reuse of scratch is legal only with a complete H-to-O barrier and verified phase lifetimes. This H/O interface does not perform Q/K normalization, gate cumsum generation, beta sigmoid, coefficient solve, or W/U preparation.

## ABI and route blocker

The current main op ABI has slot 4 (`a_storage`) as rank-4 A in the Phase6 path. The chunk32 raw-input path uses slot 4 for beta and slot 5 for raw gate. The main op tiler validates slot 4 as rank-4 A before Phase6 tiling. Separately, ACLNN invokes Prepare as a sub-op, where it owns the `useQkL2norm`, `useBetaSigmoid`, `allowNegEigval`, gate mode, and optional-output contract. The current main op entry does not receive the Prepare tiling data or all Prepare output handles/attributes.

Consequently, a distinct device key alone cannot safely identify or execute the desired raw-input full-chain route. Resolving this requires a reviewed internal ABI/composite change that carries complete descriptors, Prepare attributes and requested outputs, plus a host tiling design that serializes both Prepare and H/O tilings without changing legacy layouts. That append-only internal ABI/composite change is now implemented; the paragraphs above describe the original integration gap, not an outstanding public API blocker. See `api.md` and `validation.md` for current scope and verification status.

## Workspace and validation requirements before implementation

A future unified tiling must calculate overflow-checked, aligned, non-overlapping regions for Prepare's per-core Y records, H scratch, O scratch, W/U, H/VNew, public cumsum and synchronization/workspace metadata. Host and device must agree on the base convention (user workspace after system workspace), every offset and total byte count. Existing Prepare workspace is `AICcoreCount * 64 KiB` plus system workspace. Existing H/O helpers use additional offsets and reserve regions. Reusing the H/O scratch recipe without integrating it with Prepare's workspace is invalid.

Before enablement, establish source-derived half-open live ranges for each stage; prove the actual MIX `GetBlockIdx`, `GetBlockNum`, and subblock values; define a visibility boundary after all Prepare writers; verify all participants reach H and O collectives; and validate tails, final state, output layout, normalization, gate exponential convention, beta sigmoid/output coupling, and every requested optional output. Build, numerical tests, synchronization stress/MSS, and profiling require Ascend 950 and remain unverified here.
