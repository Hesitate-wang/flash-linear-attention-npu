# Arch35 Chunk64 Fused GDN Forward Design

方案设计规则版本：`V2`

This document retains the original design baseline and review checklist. The current development implementation and selector are described in [`api.md`](api.md); actual checks are recorded in [`validation.md`](validation.md). The existing ACLNN entry now selects the private key-64 integration for eligible calls, while native low-precision inputs retain Prepare -> H -> O. This source integration is not evidence of Ascend 950 build or device correctness.

## 1. Goal

Implement a private Ascend 950 / arch35 chunk64 GDN path from optional Q/K L2Norm through FwdO, targeting low parallelism `B*Hv <= AICcores/2`. The existing Prepare kernel already performs chunk64 stages 0-7 in MIX AIC:AIV 1:2; Phase6 composes coefficient generation, H and O inside a separate single kernel entry. The ACLNN composite currently enqueues Prepare, H and O as separate launches. The new route must combine the computations in one device kernel; adding a key that aliases Phase6 is not fusion.

No user-supplied performance model case or latency target exists. No speedup or target reachability claim is made. The local macOS host has no `ccec` or `/usr/local/Ascend`, so device-only resource/build evidence is pending.

## 2. Dependency and source-derived stage contract

For each dense `(b,hv)` chain, `hk = hv / (Hv/Hk)` and the recurrent state obeys `state[c+1] <- H(W[c],U[c],state[c])`. Chunks on one chain are strictly ordered. O consumes the same chunk's H and `v_new`; it must not run before those records are published.

Prepare's source stages are: S0 resident identity/leaf initialization; S1 optional q/k L2Norm, gate transform/cumsum and beta transform; S2 KKT Cube; S3 lower gate/leaf transforms; S4 Y Cube with GM workspace round trip; S5 A/inverse coefficient Cube; S6 `vb`/`kbg` Vector inputs; S7 W/U Cube. The implementation is `ChunkGatedDeltaRuleFwdPrepareKernel::Init/Process` in `chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h`; its source map and helpers are in `_common.h`.

Existing H/O stages are called by `RunPhase6` through `DispatchFwdH` and `DispatchFwdO` in `op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp`. ACLNN currently calls `l0op::ChunkGatedDeltaRuleFwdPrepare`, `ChunkFwdH`, and `ChunkFwdO` separately in `op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.cpp`; the wrapper is one public API but not one device launch.

Numeric facts already visible in source: Prepare uses `kGdnL2NormEps=1e-6`; vector normalization math is FP32 and normalized Q/K are stored at input precision. Beta sigmoid and optional `2*sigmoid(beta)` are controlled by explicit Prepare attributes and the beta-eff output contract. Gate cumsum honors the selected exp/exp2 convention. Exact solve, BF16 rounding, state update, O scaling and output conversion order must be reconciled expression-by-expression against H/O and `tests/atk/chunk_gated_delta_rule_fwd/gdn_reference.py` before calling the new implementation numerically equivalent.

## 3. Ownership and topology

Prepare's MIX mapping is specific and source-defined: AIV block index maps to the cube group via `/2`, `GetSubBlockIdx()` selects vector lane 0/1, while AIC block index is the group index. Prepare tasks use pack size four (three for `Hv/Hk=3`); AIV0 handles local slots 0,2, AIV1 handles 1,3, and ping/pong is `(task>>1)&1`. Its intra-group flags use IDs 0..15 and AIV1's offset 16. This is not a producer/consumer group protocol.

The Prepare class requires its own `ChunkGatedDeltaRuleFwdStageTilingData`, per-core workspace and on-chip initialization. H/O use different task schedulers and Phase6 tiling. Phase6 `SyncAll` calls require all launched participants. H/O's ready records identify `(physical batch,chunk,hv)` and do not establish Prepare→H readiness. Therefore calling Prepare `Process()` then `RunPhase6()` without redesigning roles, TPipe lifecycle, barriers, workspace, and GM visibility is not a valid schedule.

A fused schedule must define one launch-wide participant topology and an owner for every stage and output. It must prove strict chain ordering, Prepare-to-H visibility, H-to-O readiness, tail/no-work behavior, slot reuse, and final join. No consumer may wait for work only it can produce; no subset may enter a global barrier. The low-parallelism predicate alone does not establish a useful producer/consumer split. The serial-phase baseline makes every participant enter both boundaries; synchronization correctness still requires Ascend compilation and device stress evidence.

The serial-phase implementation uses the standalone arch35 Prepare, H and O interfaces, not the Phase6 wrapper. A distinct POD carries all three tilings; intermediate tensors remain owned by the existing ACLNN executor, while scratch regions are disjoint. Every MIX participant must enter both phase boundaries even if its phase has no local work. This establishes a functional fusion baseline, not improved low-parallelism throughput; device validation remains pending.

CrossCoreSetFlag/CrossCoreWaitFlag mode 0 provides the all-core same-type phase boundary; mode 2 then joins the AIC and both AIVs in each MIX group. This is phase closure, not per-task producer/consumer readiness. IBSet/IBWait is not used by this adapter. Phase-local credits must be drained before wrapper flag reuse; all participants follow identical phase order. The route is integrated as development code, with Ascend 950 compilation, synchronization stress and MSS still pending.

## 4. Resource and workspace baseline

Prepare source declares L1 512 KiB, UB 248 KiB, L0A/L0B 64 KiB each and L0C 256 KiB. Its V=128 map peaks at 496 KiB L1 and 236 KiB UB. These figures describe the existing Prepare kernel only; they do not prove a fused adapter's concurrent live set fits. Chunk32 resources are not a valid substitute.

Prepare Stage4 uses 64 KiB per core for four FP32 64x64 Y records; its current host tiler requests `coreNum*64KiB` plus system workspace. Phase6 has a separate coefficient/H/O workspace and serialized trailer. The new route needs a private POD and non-overlapping workspace regions for every retained Prepare intermediate, H/vNew, O input, state as applicable, and cross-group flags. Host sizing must align and overflow-check every multiplication/addition; device offsets must match the host POD. Do not reinterpret or modify the key32 or Phase6 tiling layouts.

Before enabling the route, produce complete per-stage absolute half-open L1/UB/L0 address maps, live ranges, peaks, GM record offsets/sizes, transfer counts/bytes, and reuse conditions. The currently checked source-derived numbers are a baseline, not that fused map.

## 5. Task domain and low-parallelism candidate

Dense Prepare task identity is `(b,chunk,hv)`; recurrence identity is `(b,hv)`. `B*Hv <= AICcores/2` is only an initial selector condition. A design must also account for number of chunks, actual independent chains, role counts, work waves, tail load and whether any consumer tasks are ready without relying on a consumer-produced dependency. No user performance case was supplied, so use no invented case as a target; report a representative arithmetic case only as a scheduling illustration and do not claim measured performance.

The first supported subset should be the intersection of Prepare, H/O, and ACLNN contracts. Do not silently exclude output/state/mode requests if doing so changes current behavior: such calls may use the fused route only when every requested output and semantic option is implemented; otherwise retain the existing supported route or its current error behavior. In particular, preserve the coupling between `betaEffOutOptional`, sigmoid and `allowNegEigval`.

## 6. R01-R21 status

| Rule | Status | Evidence still needed |
|---|---|---|
| R01-R02 | Partial | Stage graph identified; a unified role/ownership schedule is not proven |
| R03-R09 | Partial | Existing Prepare map is source-derived; fused concurrent live ranges/capacity remain open |
| R10-R18 | Blocked | End-to-end transfers, record ownership, slot lifetimes and AIV mapping |
| R19 | Blocked | Minimum legal stage count after full live-set analysis |
| R20 | Partial | Candidate threshold known; unified blockDim/waves/tails and device selection pending |
| R21 | Partial | Epsilon and storage/compute boundaries known; full range/cast analysis pending |

Implementation proceeds through the existing public API as requested. The private route is development code pending Ascend compilation, accuracy, synchronization and resource validation; the checklist above records remaining evidence, not an instruction to leave dispatch disconnected.

## 7. Implementation and validation

Add a distinct fused kernel body and tiling data, rather than aliasing Phase6. Keep key 32 and Phase6 keys 1/2/301 unchanged. Reuse existing Prepare and H/O math only through interfaces whose tiling, ownership, resource lifetime and synchronization assumptions have been proven compatible. Integrate exactly one fused `ADD_TO_LAUNCHER_LIST_AICORE` operation for eligible calls; host selector, tiling key and device dispatch must agree. Preserve the public ten-element return contract and every requested optional output.

Complete all stage and ACLNN integration work before the requested consolidated test phase. On Ascend950, verify build/install, CPU golden and ATK accuracy, eligibility/fallback boundaries, tails and supported state/output modes, repeat synchronization and MSS, legacy keys, and performance using `msprof op_summary` `Task Duration(us)`. Record only actual evidence in `docs/validation.md`, then execute `operator-review`. On this Mac, source/static checks are possible but Ascend build, device accuracy, MSS and performance remain unverified.
