/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * A5 L1 layout for the ChunkFwdHOFused SINGLE_CORE_CHUNK_FUSED schedule
 * (executionMode = 2, blockDim = T).
 *
 * The fused AIC walks one chunk at a time and runs the FwdH and FwdO Cube
 * stages strictly serially:
 *
 *   OC1 -> HC1 -> OC2 -> OC3 -> HC2
 *
 * Two properties hold for every AIC stage and are what makes a single L1 window
 * legal here:
 *   L1. Every stage stages its own operands from GM (preSetFlags / copyGmToL1)
 *       and drains them (finalWaitFlags / waitL1Drained / waitL0Drained) before
 *       the next stage starts.
 *   L2. No L1 tensor survives a stage boundary.  All Cube-to-Vector and
 *       Vector-to-Cube exchanges of this schedule use GM workspaces:
 *         W @ h        -> H v workspace   -> HV1 (MTE2)
 *         VUpdate      -> H vUpdate ws    -> HC2 (MTE2)
 *         K^T @ VUpdate-> H h workspace   -> HV2 (MTE2)
 *         masked QK    -> O aftermask ws  -> OC3 (MTE2)
 *         Q @ h        -> O h workspace   -> OV2 (MTE2)
 *         maskedQK@V   -> O v workspace   -> OV2 (MTE2)
 *       Consequently the per-stage bases below may alias.
 *
 * L1_O_CUBE3_BASE keeps the split-mode value because BlockMmadAttenVNEW* owns
 * its L1A/L1B slots relative to that offset with the dedicated event ids.
 */

#ifndef CHUNK_FWD_HO_FUSED_A5_SINGLE_SLOT_L1_LAYOUT_H
#define CHUNK_FWD_HO_FUSED_A5_SINGLE_SLOT_L1_LAYOUT_H

#include <cstdint>

namespace GDN {

// FwdH Cube stages: HC1 stages W and h, HC2 stages K and the VUpdate tile.
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_H_STAGE_BASE = 0U;

// FwdO Cube stages: OC1 stages Q and K, OC2 stages Q and h.
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_O_CUBE1_BASE = 0U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_O_CUBE2_BASE = 0U;

// OC3 (maskedQK @ V_new) keeps the split-mode reservation and event ids.
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_O_CUBE3_BASE = 192U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_O_CUBE3_L1A_EVENT = 4U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_O_CUBE3_L1B_EVENT = 6U;

// The FwdH V-new epilogue takes an L1 VUpdate operand, but the arch35 body only
// declares it (no read and no write); see block_epilogue_gdn_fwdh_vnew.hpp.
// The fused kernel therefore passes a default-constructed tensor and claims no
// L1 address for it.  If that epilogue ever starts using the operand, this
// schedule must reserve a real window before re-enabling it.
constexpr bool CHUNK_FWD_HO_FUSED_L1_VUPDATE_OPERAND_UNUSED = true;

} // namespace GDN

#endif // CHUNK_FWD_HO_FUSED_A5_SINGLE_SLOT_L1_LAYOUT_H
