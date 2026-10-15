/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * A5 L1 layout for the ChunkFwdHOFused SINGLE_CORE_CHUNK_FUSED schedule
 * (executionMode = 2, blockDim = T).
 *
 * AIC compute order: OC1 -> HC1 -> OC2 -> HC2 -> OC3.
 * HC1/HC2 share [0,128 KiB). OC1, OC2, and OC3 use separate
 * [128,256), [256,384), and [384,512) KiB windows. L0 is shared and reused
 * only after stage drain; the L0 allocation remains shared.
 */

#ifndef CHUNK_FWD_HO_FUSED_A5_SINGLE_SLOT_L1_LAYOUT_H
#define CHUNK_FWD_HO_FUSED_A5_SINGLE_SLOT_L1_LAYOUT_H

#include <cstdint>

namespace GDN {

// FwdH Cube stages: HC1 stages W and h, HC2 stages K and the VUpdate tile.
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_H_STAGE_BASE = 0U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_H_STAGE_BYTES = 128U * 1024U;

// Each 128x128 FP16/BF16 operand pair occupies 128 KiB including both stages.
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_O_CUBE1_BASE = 128U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_O_CUBE2_BASE = 256U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_O_CUBE2_L1A_EVENT = 4U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_O_CUBE2_L1B_EVENT = 6U;

// Keep OC3 disjoint from every preceding Cube stage.
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_O_CUBE3_BASE = 384U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_O_CUBE3_L1A_EVENT = 4U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_O_CUBE3_L1B_EVENT = 6U;

static_assert(CHUNK_FWD_HO_FUSED_L1_O_CUBE2_BASE >=
                  CHUNK_FWD_HO_FUSED_L1_O_CUBE1_BASE + 128U * 1024U,
              "OC2 L1 window overlaps OC1");
static_assert(CHUNK_FWD_HO_FUSED_L1_O_CUBE3_BASE >=
                  CHUNK_FWD_HO_FUSED_L1_O_CUBE2_BASE + 128U * 1024U,
              "OC3 L1 window overlaps OC2");
static_assert(CHUNK_FWD_HO_FUSED_L1_O_CUBE1_BASE >=
                  CHUNK_FWD_HO_FUSED_L1_H_STAGE_BASE + CHUNK_FWD_HO_FUSED_L1_H_STAGE_BYTES,
              "OC1 L1 window overlaps the H window");

// The FwdH V-new epilogue takes an L1 VUpdate operand, but the arch35 body only
// declares it (no read and no write); see block_epilogue_gdn_fwdh_vnew.hpp.
// The fused kernel therefore passes a default-constructed tensor and claims no
// L1 address for it.  If that epilogue ever starts using the operand, this
// schedule must reserve a real window before re-enabling it.
constexpr bool CHUNK_FWD_HO_FUSED_L1_VUPDATE_OPERAND_UNUSED = true;

} // namespace GDN

#endif // CHUNK_FWD_HO_FUSED_A5_SINGLE_SLOT_L1_LAYOUT_H
