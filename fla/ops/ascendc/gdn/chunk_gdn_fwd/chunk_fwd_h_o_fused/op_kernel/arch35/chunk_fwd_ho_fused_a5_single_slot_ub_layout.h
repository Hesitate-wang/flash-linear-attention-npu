/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * A5 UB layout for the ChunkFwdHOFused SINGLE_CORE_CHUNK_FUSED schedule
 * (executionMode = 2, blockDim = T).
 *
 * One MIX core owns one (batch, value-head) task and walks its chunks strictly
 * serially.  Inside every chunk the FwdH and FwdO stages alternate on the same
 * core, so every buffer exists once (no ping-pong) and the FwdH Cube results go
 * straight into AIV UB through Fixpipe, exactly like the split-mode DirectUb
 * path -- the space freed by dropping the second slot is what makes that direct
 * path fit next to the FwdO buffers:
 *
 *   AIC : [chunk gate] OC1 -> HC1 -> OC2 -> [wait AIV1] -> OC3 -> HC2
 *   AIV : [mask] [wait OC1] OV1 -> [wait HC1] HV1 -> [AIV1] OV2
 *         -> [wait HC2] HV2 -> [AIVCHUNK]
 *
 * Address map (V128; every size is per AIV):
 *
 *   [0,32K)      FwdH calc                             HV1/HV2
 *   [32K,96K)    FwdH staging: u / V_new / VUpdate     HV1/HV2
 *   [96K,117K)   FwdH small: g / g_last / g_sub / g_input / share
 *   [135K,199K)  H_DIRECT: HC1 and HC2 Fixpipe target  AIC -> AIV, read in place
 *   [199K,231K)  FwdH update output staging            HV2
 *   [231K,232K)  FwdH update g_last                    HV2
 *   [0,16K)      causal mask (rebuilt per chunk)       OV1/OV2
 *   [16K,71K)    FwdO base: gbrc / gcomp / share / g   OV1/OV2
 *   [71K,103K)   FwdO V work (GM staging)              OV2
 *   [103K,135K)  FwdO H work (GM staging)              OV2
 *   [199K,239K)  FwdO OV1 scratch (a / out / out_half) OV1
 *   [199K,223K)  FwdO OV2 out scratch                  OV2
 *
 * Invariants enforced by the fused kernel:
 *   I1. HC1/HC2 write H_DIRECT through Fixpipe and the matching AIV epilogue
 *       reads that same window in place.  The handshake is the DirectUb one:
 *       the AIV presets the 0x4 free flag per slot, the Cube waits it before
 *       writing and sets the 0x4 ready flag after the write, the epilogue waits
 *       the ready flag and sets the free flag again.
 *   I2. H_DIRECT must stay disjoint from every buffer of the FwdO stage that
 *       runs concurrently with the Cube Fixpipe: HC1 runs with OV1
 *       ([0,71K) plus [199K,239K)) and HC2 runs with OV2 ([0,135K) plus
 *       [199K,223K)).  The static_asserts below pin that property.
 *   I3. FwdH clobbers UB[0,16K) with its calc buffer, so the 64 x 64 causal
 *       mask is rebuilt by the AIV before every OV1.
 *
 * Full chunks run the single-slot preload family, whose epilogue offsets are
 * taken from the constants in this header.  Partial chunks use the split-mode
 * GM epilogues with their own (legacy) offsets, so those shared files are not
 * parameterized by this layout.
 */

#ifndef CHUNK_FWD_HO_FUSED_A5_SINGLE_SLOT_UB_LAYOUT_H
#define CHUNK_FWD_HO_FUSED_A5_SINGLE_SLOT_UB_LAYOUT_H

#include <cstdint>

#include "../chunk_fwd_h_o_fused_struct.h"
#include "chunk_fwd_h_o_fused_ub_layout.h"

namespace GDN {

// ---------------------------------------------------------------------------
// Single-slot FwdH layout consumed by the fused kernel.
// ---------------------------------------------------------------------------
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_CALC_OFFSET = 0U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_CALC_BYTES = 32U * 1024U;

constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_STAGING_OFFSET = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_STAGING_BYTES = 64U * 1024U;

constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_SMALL_OFFSET = 96U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_SMALL_BYTES = 21U * 1024U;

// Initial h[0] hand-off staging written by WriteInitialHandoffState: one AIV
// lane of the state (ceil(128 / 2) rows x V x 4 B = 32 KiB on A5), i.e. the
// H staging area before the first Fixpipe lands in H_DIRECT.
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_SLOT_OFFSET =
    CHUNK_FWD_HO_FUSED_UB_H_STAGING_OFFSET;

// Cube1 (W @ h) and Cube2 (K^T @ VUpdate) both land here; the single slot and
// the strict chunk order keep the two writes and their readers apart.
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_DIRECT_OFFSET = 135U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_DIRECT_BYTES = 64U * 1024U;

constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_UPDATE_SCRATCH_OFFSET = 199U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_UPDATE_SCRATCH_BYTES = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_UPDATE_G_OFFSET = 231U * 1024U;

// FwdO regions (the legacy ping half; the O epilogues are used unchanged).
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_MASK_OFFSET = 0U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_MASK_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_BASE_OFFSET = 0U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_BASE_BYTES = 71U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_V2_VWORK_OFFSET = 71U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_V2_VWORK_BYTES = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_V2_HWORK_OFFSET = 103U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_V2_HWORK_BYTES = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_V1_A_OFFSET = 199U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_V1_A_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_V1_OUT_OFFSET = 215U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_V1_OUT_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_V1_OUT_HALF_OFFSET = 231U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_V1_OUT_HALF_BYTES = 8U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_V2_OUT_OFFSET = 199U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_V2_OUT_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_V2_OUT_HALF_OFFSET = 215U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_V2_OUT_HALF_BYTES = 8U * 1024U;

// Tail Vector fallback scratch (a tail shorter than one MMAD tile has no Cube).
// It deliberately lives inside H_DIRECT: a tail chunk never uses the DirectUb
// slot, and a partial chunk is always the last chunk of the task, so nothing
// writes H_DIRECT afterwards.
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_TAIL_INPUT_OFFSET = 166U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_TAIL_FLOAT_OFFSET = 167U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_TAIL_ACCUM_OFFSET = 168U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_TAIL_WEIGHT_INPUT_OFFSET = 169U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_TAIL_WEIGHT_FLOAT_OFFSET = 170U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_TAIL_END = 171U * 1024U;

// 0x2 domain flag ids of the fused single-core schedule.  Single slot means one
// id per dependency; the preload H path needs two extra tokens of its own and
// the FwdO side keeps its own (see the fused kernel header):
//   hVec1Done       consumed inside the preload K^T @ VUpdate MMAD
//   oAiv1Done       V-new/masked-QK token consumed by the FwdO AIC before OC3
//   preloadSlotFree consumed inside the preload Fixpipe slot handshake
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_OC1_DONE = 0;
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_HC1_DONE = 1;
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_OC2_DONE = 2;
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_OC3_DONE = 3;
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_HC2_DONE = 4;
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_O_AIV1_DONE = 5;
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_AIV_CHUNK_DONE = 6;
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_H_VEC1_DONE = 7;
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_PRELOAD_SLOT_FREE = 8;
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_HV2_DONE_FOR_OC2 = 9;
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_ID_COUNT = 10;

// DirectUb handshake ids (0x4 domain).  The fused schedule keeps a single slot,
// so ubListId is pinned to 0; the block and the epilogue also address the second
// AIV through +FLAG_STRIDE, exactly like the split-mode DirectUb path.
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_DIRECT_FREE_FLAG = 1U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_DIRECT_READY_FLAG = 6U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_DIRECT_FLAG_STRIDE = 16U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_DIRECT_VEC_NUM = 2U;

// Whole single-slot FwdH footprint, excluding the FwdO-only regions.
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_USED_MAX = 239U * 1024U;

// ---------------------------------------------------------------------------
// Checks.  Every FwdH stage view must stay disjoint from itself and from the
// FwdO stage that runs concurrently with the Cube Fixpipe writing H_DIRECT.
// ---------------------------------------------------------------------------
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_CALC_END =
    CHUNK_FWD_HO_FUSED_UB_H_CALC_OFFSET + CHUNK_FWD_HO_FUSED_UB_H_CALC_BYTES;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_STAGING_END =
    CHUNK_FWD_HO_FUSED_UB_H_STAGING_OFFSET + CHUNK_FWD_HO_FUSED_UB_H_STAGING_BYTES;

static_assert(CHUNK_FWD_HO_FUSED_UB_H_SLOT_OFFSET + 32U * 1024U <=
                  CHUNK_FWD_HO_FUSED_UB_H_STAGING_END,
              "the H staging area must hold one AIV lane of the initial h state");
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_SMALL_END =
    CHUNK_FWD_HO_FUSED_UB_H_SMALL_OFFSET + CHUNK_FWD_HO_FUSED_UB_H_SMALL_BYTES;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_DIRECT_END =
    CHUNK_FWD_HO_FUSED_UB_H_DIRECT_OFFSET + CHUNK_FWD_HO_FUSED_UB_H_DIRECT_BYTES;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_UPDATE_SCRATCH_END =
    CHUNK_FWD_HO_FUSED_UB_H_UPDATE_SCRATCH_OFFSET +
    CHUNK_FWD_HO_FUSED_UB_H_UPDATE_SCRATCH_BYTES;

// FwdH internal disjointness.
static_assert(CHUNK_FWD_HO_FUSED_UB_H_CALC_END <= CHUNK_FWD_HO_FUSED_UB_H_STAGING_OFFSET,
              "fused UB: FwdH calc overlaps the FwdH staging slot");
static_assert(CHUNK_FWD_HO_FUSED_UB_H_STAGING_END <= CHUNK_FWD_HO_FUSED_UB_H_SMALL_OFFSET,
              "fused UB: FwdH staging overlaps the FwdH small buffers");
static_assert(CHUNK_FWD_HO_FUSED_UB_H_SMALL_END <= CHUNK_FWD_HO_FUSED_UB_H_DIRECT_OFFSET,
              "fused UB: FwdH small buffers overlap H_DIRECT");
static_assert(CHUNK_FWD_HO_FUSED_UB_H_DIRECT_END <=
                  CHUNK_FWD_HO_FUSED_UB_H_UPDATE_SCRATCH_OFFSET,
              "fused UB: H_DIRECT overlaps the FwdH update scratch");
static_assert(CHUNK_FWD_HO_FUSED_UB_H_UPDATE_SCRATCH_END <=
                  CHUNK_FWD_HO_FUSED_UB_H_UPDATE_G_OFFSET,
              "fused UB: FwdH update scratch overlaps its g_last buffer");
static_assert(CHUNK_FWD_HO_FUSED_UB_H_TAIL_INPUT_OFFSET >=
                      CHUNK_FWD_HO_FUSED_UB_H_DIRECT_OFFSET &&
                  CHUNK_FWD_HO_FUSED_UB_H_TAIL_END <= CHUNK_FWD_HO_FUSED_UB_H_DIRECT_END,
              "fused UB: the tail Vector scratch must stay inside H_DIRECT");

// H_DIRECT versus the concurrently running FwdO stage.
static_assert(CHUNK_FWD_HO_FUSED_UB_H_DIRECT_OFFSET >=
                  CHUNK_FWD_HO_FUSED_UB_O_BASE_OFFSET + CHUNK_FWD_HO_FUSED_UB_O_BASE_BYTES,
              "fused UB: H_DIRECT overlaps the FwdO base region (HC1 runs with OV1)");
static_assert(CHUNK_FWD_HO_FUSED_UB_H_DIRECT_OFFSET >=
                  CHUNK_FWD_HO_FUSED_UB_O_V2_VWORK_OFFSET + CHUNK_FWD_HO_FUSED_UB_O_V2_VWORK_BYTES +
                      CHUNK_FWD_HO_FUSED_UB_O_V2_HWORK_BYTES,
              "fused UB: H_DIRECT overlaps the FwdO work slots (HC2 runs with OV2)");
static_assert(CHUNK_FWD_HO_FUSED_UB_H_DIRECT_END <= CHUNK_FWD_HO_FUSED_UB_O_V1_A_OFFSET,
              "fused UB: H_DIRECT overlaps the FwdO OV1 scratch");

// FwdO internal disjointness and capacity.
static_assert(CHUNK_FWD_HO_FUSED_UB_O_MASK_OFFSET + CHUNK_FWD_HO_FUSED_UB_O_MASK_BYTES <=
                  CHUNK_FWD_HO_FUSED_UB_O_BASE_OFFSET + CHUNK_FWD_HO_FUSED_UB_O_BASE_BYTES,
              "fused UB: FwdO mask does not fit the FwdO base region");
static_assert(CHUNK_FWD_HO_FUSED_UB_O_BASE_OFFSET + CHUNK_FWD_HO_FUSED_UB_O_BASE_BYTES <=
                  CHUNK_FWD_HO_FUSED_UB_O_V2_VWORK_OFFSET,
              "fused UB: FwdO base region overlaps the OV2 V work tile");
static_assert(CHUNK_FWD_HO_FUSED_UB_O_V2_HWORK_OFFSET + CHUNK_FWD_HO_FUSED_UB_O_V2_HWORK_BYTES <=
                  CHUNK_FWD_HO_FUSED_UB_O_V1_A_OFFSET,
              "fused UB: FwdO OV2 work slots overlap the OV1 scratch");
static_assert(CHUNK_FWD_HO_FUSED_UB_O_V1_OUT_HALF_OFFSET +
                      CHUNK_FWD_HO_FUSED_UB_O_V1_OUT_HALF_BYTES <=
                  CHUNK_FWD_HO_FUSED_UB_USED_MAX,
              "fused UB: OV1 scratch exceeds the recorded footprint");
static_assert(CHUNK_FWD_HO_FUSED_UB_USED_MAX <=
                  static_cast<uint32_t>(CHUNK_FWD_HO_FUSED_UB_DATA_LIMIT),
              "fused UB: footprint crosses the UB data limit");
static_assert(static_cast<uint32_t>(CHUNK_FWD_HO_FUSED_UB_DATA_LIMIT) +
                      static_cast<uint32_t>(CHUNK_FWD_HO_FUSED_UB_RESERVED_TAIL) <=
                  CHUNK_FWD_HO_A5_UB_CAPACITY,
              "fused UB: data limit plus reserved tail exceeds the A5 UB capacity");

} // namespace GDN

#endif // CHUNK_FWD_HO_FUSED_A5_SINGLE_SLOT_UB_LAYOUT_H
