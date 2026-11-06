# ChunkGatedDeltaRuleFwd: arch35 chunk64 fused route

## Status

Source-level two-group producer/consumer implementation. The public ACLNN ABI, stable Python signature and ten-element return tuple are unchanged. The internal OpDef adds Prepare inputs, intermediate outputs and attributes; host/device integration uses key 64. For eligible low-parallelism calls (`B*Hv <= AIC/2`), exactly `2*chainCount` MIX groups are launched. Producer groups compute coefficients and W/U; consumer groups compute QK/A' before the W/U rendezvous and H/O after it, on the same chunk. One mode-0 all-AIV barrier per chunk is placed after consumer QK/A' and before its W/U loads. The producer inverse uses two FP32 diagonal solves on AIV0/AIV1 and one mode-1 exchange per chunk. Calls above the bound use the same self-contained roles in group-serial mode. The dual-AIV inverse and revised consumer W/U schedule have no Ascend build, accuracy, synchronization or performance validation yet, so their benefits must not be treated as proven.

## Entry and selector

The route is considered only by the existing `npu_chunk_gated_delta_rule_fwd` entry. It uses existing arguments, not chunk size alone:

- `chunk_size == 64`
- Ascend 950 / arch35
- BF16 q/k/v/output; FP32 gate and beta after the wrapper's existing casts
- optional a_log/dt_bias must be FP32 for the new route
- `K == V == 128`, `Hv % Hk == 0`, `Hv/Hk <= 4`
- no optional output or supported mode may be silently ignored

Native low-precision Prepare inputs are cast to FP32 by the existing wrapper before entering key 64. `B * Hv <= AICcores / 2` selects the two-group pipeline; other eligible key-64 shapes use group-serial execution. The host obtains real AIC cores from the platform API. Internal additions are append-only, and legacy calls use `fused_prepare=false`.

The stable Python API already exposes `use_qk_l2norm_in_kernel` and `use_beta_sigmoid_in_kernel`. They remain the selectors for those semantics; no new public argument is proposed. Existing ACLNN behavior ties beta sigmoid in Prepare to a non-null `betaEffOutOptional`; `allowNegEigval=true` requires that optional output. The fused route must preserve this coupling and produce beta_eff when requested. It must not change behavior based on chunk size.

## Inputs

The public ACLNN ABI is unchanged. The internal operator appends `a_log` and `dt_bias` inputs, ten optional intermediate outputs (`q_hat`, `k_hat`, `q_rstd`, `k_rstd`, `beta_eff`, `h`, `w`, `u`, `v_new`, `g_cumsum_bht`) and five attributes (`fused_prepare`, `use_qk_l2norm`, `use_exp2`, `use_beta_sigmoid`, `output_a`). Existing input/output/attribute indices retain their order. Key 64 branches before Phase6 validation and has its own POD; it does not reinterpret the Phase6 trailer. Chunk32's beta-in-slot-4 convention remains unchanged.

The implemented structural contract is dense head-major or packed BF16, `K=V=128`, chunk64 and `Hv/Hk` in the Prepare-supported range. Tails and optional public outputs retain the existing allocations and exports. Native low-precision gate/beta or a_log/dt_bias use the existing fallback. These are source-level eligibility rules; device numerical coverage remains pending.

The wrapper exposes `use_qk_l2norm_in_kernel` and `use_beta_sigmoid_in_kernel`. The ACLNN composite requests Prepare sigmoid when `betaEffOutOptional` is non-null; `allowNegEigval` requires that output. Any fused route must preserve this coupling, write `beta_eff` and other allocated outputs with existing semantics, and accept both enabled/disabled cases only if implemented. No new public selector or semantic inference from chunk size is approved.

## Outputs

No output ABI change. Stable Python continues returning `(o, final_state, g_cumsum, A, beta_eff, h, q_hat, k_hat, q_rstd, k_rstd)`. Every output allocated or requested by the existing wrapper must be written with the existing shape, dtype, layout, and semantics before a call is eligible for fusion. An output cannot be silently ignored; fallback must preserve the current behavior.

## Compatibility

- Chunk32 key 32 and its constraints are unchanged.
- Phase6 keys 1, 2, and 301 and their workspace layout are unchanged.
- All calls not matching the complete new predicate retain existing dispatch and validation.
- The new route requires a private tiling key and POD, with host/device predicate parity.
- Support is not declared until Ascend 950 build, accuracy, deterministic synchronization, and regression evidence exist.
