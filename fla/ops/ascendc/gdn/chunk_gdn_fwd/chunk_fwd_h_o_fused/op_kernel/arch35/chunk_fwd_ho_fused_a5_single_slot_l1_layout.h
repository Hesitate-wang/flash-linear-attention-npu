/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * A5 L1 layout for the ChunkFwdHOFused SINGLE_CORE_CHUNK_FUSED schedule
 * (executionMode = 2, blockDim = T).
 *
 * AIC compute order: OC1 -> HC1 -> OC2 -> HC2 -> OC3.
 * H stages and OC1 use [0,128 KiB), with L1 event ids 0..3.
 * OC2 uses [128,256 KiB), with L1 event ids 4..7. Its GM->L1
 * prefetch is issued after HC1, before HC1's resource-drain waits.
 * OC3 uses [192,320 KiB), with L1 event ids 4..7. OC2 is fully
 * drained before OC3, so their address and event reuse is intentional.
 * L0 is shared: HC1 releases its L0 events before OC2 initializes them.
 */

#ifndef CHUNK_FWD_HO_FUSED_A5_SINGLE_SLOT_L1_LAYOUT_H
#define CHUNK_FWD_HO_FUSED_A5_SINGLE_SLOT_L1_LAYOUT_H

#include <cstdint>

namespace GDN {

// FwdH Cube stages: HC1 stages W and h, HC2 stages K and the VUpdate tile.
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_H_STAGE_BASE = 0U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_H_STAGE_BYTES = 128U * 1024U;

// FwdO Cube stages: OC1 stages Q and K, OC2 stages Q and h.
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_O_CUBE1_BASE = 0U;
// OC2 is prefetched while HC1 is computing, so it must not overwrite HC1's
// two L1 stages.  Each 128x128 FP16/BF16 operand pair occupies 128 KiB.
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_O_CUBE2_BASE =
    CHUNK_FWD_HO_FUSED_L1_H_STAGE_BASE + CHUNK_FWD_HO_FUSED_L1_H_STAGE_BYTES;
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_O_CUBE2_L1A_EVENT = 4U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_O_CUBE2_L1B_EVENT = 6U;

// OC3 keeps its original address and reuses the drained OC2 L1 events.
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_O_CUBE3_BASE = 192U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_O_CUBE3_L1A_EVENT = 4U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_L1_O_CUBE3_L1B_EVENT = 6U;

static_assert(CHUNK_FWD_HO_FUSED_L1_O_CUBE2_BASE >=
                  CHUNK_FWD_HO_FUSED_L1_H_STAGE_BASE + CHUNK_FWD_HO_FUSED_L1_H_STAGE_BYTES,
              "OC2 L1 window overlaps the HC1 window");
// OC2 is fully drained before OC3 starts, so they may overlap in address
// range and reuse L1 events 4..7. Neither overlaps the H window [0,128 KiB).
static_assert(CHUNK_FWD_HO_FUSED_L1_O_CUBE3_BASE >=
                  CHUNK_FWD_HO_FUSED_L1_H_STAGE_BASE + CHUNK_FWD_HO_FUSED_L1_H_STAGE_BYTES,
              "OC3 L1 window overlaps the H window");

// The FwdH V-new epilogue takes an L1 VUpdate operand, but the arch35 body only
// declares it (no read and no write); see block_epilogue_gdn_fwdh_vnew.hpp.
// The fused kernel therefore passes a default-constructed tensor and claims no
// L1 address for it.  If that epilogue ever starts using the operand, this
// schedule must reserve a real window before re-enabling it.
constexpr bool CHUNK_FWD_HO_FUSED_L1_VUPDATE_OPERAND_UNUSED = true;

} // namespace GDN

#endif // CHUNK_FWD_HO_FUSED_A5_SINGLE_SLOT_L1_LAYOUT_H
