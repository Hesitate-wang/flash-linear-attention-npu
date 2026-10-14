/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * A5 UB layout for the ChunkFwdHOFused SINGLE_CORE_CHUNK_FUSED schedule
 * (executionMode = 2, blockDim = T).
 *
 * One MIX core owns one (batch, value-head) task and walks its chunks strictly
 * serially.  Inside every chunk the FwdH and FwdO stages alternate on the same
 * core. FwdH has one Fixpipe slot; FwdO retains its internal Vector ping/pong
 * scratch, not a second chunk slot:
 *
 *   AIC : [chunk gate] OC1 -> HC1 -> OC2 -> HC2 -> [wait HV1] OC3
 *   AIV : [mask] OV1 -> [O/H fence] HV1 -> HV2 -> [H/O fence] OV2
 *         -> [O fence, chunk finished]
 *
 * Address map (V128; every size is per AIV):
 *
 *   [0,32K)      FwdH calc                             HV1/HV2
 *   [64K,80K)    FwdH u / V_new / VUpdate             HV1
 *   [80K,96K)    FwdH H input/output                  HV2
 *   [96K,100K)   FwdH g / g_input / g_last            HV1/HV2
 *   [135K,199K)  H_DIRECT: HC1 and HC2 Fixpipe target  AIC -> AIV, read in place
 *   [135K,167K)  FP32 final-state output in H_DIRECT  HV2 MTE3
 *   [0,16K)      causal mask (rebuilt per chunk)       OV1/OV2
 *   [16K,71K)    FwdO base: gbrc / gcomp / share / g   OV1/OV2
 *   [71K,103K)   FwdO V work (GM staging)              OV2
 *   [103K,135K)  FwdO H work (GM staging)              OV2
 *   [135K,167K)  FwdO V work pong (reuses H_DIRECT)   OV2
 *   [167K,199K)  FwdO H work pong (reuses H_DIRECT)   OV2
 *   [199K,239K)  FwdO OV1 scratch (a / out / out_half) OV1
 *   [199K,223K)  FwdO OV2 out scratch ping            OV2
 *   [223K,247K)  FwdO OV2 out scratch pong            OV2
 *
 * Invariants enforced by the fused kernel:
 *   I1. HC1/HC2 write H_DIRECT through Fixpipe and the matching AIV epilogue
 *       reads that same window in place. HC1/HC2 readiness uses mode-0x2
 *       flags; HV1 publishes HC2's input only after reading HC1's result.
 *   I2. H_DIRECT must stay disjoint from every buffer of the FwdO stage that
 *       runs concurrently with the Cube Fixpipe: HC1 runs with OV1
 *       ([0,71K) plus [199K,239K)). OV2 starts only after HV2 (or the final
 *       skipped-HV2 wait) and its H/O fence, so its pong views may reuse
 *       H_DIRECT. The next HC1 cannot start until OV2 retires the chunk.
 *   I3. FwdH clobbers UB[0,16K) with its calc buffer, so the 64 x 64 causal
 *       mask is rebuilt by the AIV before every OV1.
 *
 * Full and partial chunks use the same preload/epilogue path with actual
 * blockTokens. Cross-family UB reuse is protected by local stage fences.
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

// Legacy reservations, not the active HV2 views (staging + 48 KiB and small
// + 3 KiB). OV1/OV2 own this upper scratch region in the current driver.
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_UPDATE_SCRATCH_OFFSET = 199U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_UPDATE_SCRATCH_BYTES = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_UPDATE_G_OFFSET = 231U * 1024U;

// FwdO regions. OV1 aliases ping/pong; OV2 keeps both Vector scratch views.
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
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_V2_VWORK_PONG_OFFSET = 135U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_V2_HWORK_PONG_OFFSET = 167U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_V2_OUT_PONG_OFFSET = 223U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_O_V2_OUT_HALF_PONG_OFFSET = 239U * 1024U;

// Local HardEvent domains, independent of the mode-0x2/mode-0x4 flag ids.
constexpr uint32_t CHUNK_FWD_HO_FUSED_EVENT_O_BEGIN = 0U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_EVENT_O_FENCE = 3U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_EVENT_H_OUTPUT = 4U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER = 5U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_EVENT_H_INPUT = 6U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_EVENT_H_SCALAR = 7U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_EVENT_H_FENCE =
    CHUNK_FWD_HO_FUSED_EVENT_H_SCALAR;
static_assert(CHUNK_FWD_HO_FUSED_EVENT_O_FENCE < CHUNK_FWD_HO_FUSED_EVENT_H_OUTPUT &&
                  CHUNK_FWD_HO_FUSED_EVENT_H_FENCE < 8U,
              "fused Vector event domains must be disjoint and fit ids 0-7");

// Legacy address constants; the current MODE2 driver has no tail fallback.
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_TAIL_INPUT_OFFSET = 166U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_TAIL_FLOAT_OFFSET = 167U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_TAIL_ACCUM_OFFSET = 168U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_TAIL_WEIGHT_INPUT_OFFSET = 169U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_TAIL_WEIGHT_FLOAT_OFFSET = 170U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_TAIL_END = 171U * 1024U;

// 0x2 domain flag ids of the fused single-core schedule.  Single slot means one
// id per dependency; the preload H path needs two extra tokens of its own and
// the FwdO side keeps its own (see the fused kernel header):
//   hv1VUpdateReadyForHc2       consumed inside the preload K^T @ VUpdate MMAD
//   ov1MaskAndHv1VNewReadyForOc3       V-new/masked-QK token consumed by the FwdO AIC before OC3
//   hv2HReadyForHc1 consumed before HC1 preloads H
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_OC1_QK_READY = 0;
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_HC1_WH_READY = 1;
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_OC2_QH_READY = 2;
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_OC3_ATTENTION_V_READY = 3;
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_HC2_KV_UPDATE_READY = 4;
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_OV1_MASK_AND_HV1_VNEW_READY_FOR_OC3 = 5;
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_OV2_CHUNK_FINISHED = 6;
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_HV1_VUPDATE_READY_FOR_HC2 = 7;
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_HV2_H_READY_FOR_HC1 = 8;
constexpr uint16_t CHUNK_FWD_HO_FUSED_FLAG_ID_COUNT = 9;

// Legacy DirectUb handshake ids (0x4 domain). The current MODE2 preload path
// uses the mode-0x2 readiness flags above, not these reserved constants.
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_DIRECT_FREE_FLAG = 1U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_DIRECT_READY_FLAG = 6U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_DIRECT_FLAG_STRIDE = 16U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_DIRECT_VEC_NUM = 2U;

// Peak of every active view, including OV2's internal output pong.
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_USED_MAX = 247U * 1024U;

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
              "fused UB: H_DIRECT overlaps the FwdO ping work slots");
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
static_assert(CHUNK_FWD_HO_FUSED_UB_O_V2_VWORK_PONG_OFFSET ==
                  CHUNK_FWD_HO_FUSED_UB_H_DIRECT_OFFSET &&
                  CHUNK_FWD_HO_FUSED_UB_O_V2_HWORK_PONG_OFFSET +
                      CHUNK_FWD_HO_FUSED_UB_O_V2_HWORK_BYTES ==
                  CHUNK_FWD_HO_FUSED_UB_H_DIRECT_END,
              "fused UB: OV2 pong work must reuse H_DIRECT only after the H/O fence");
static_assert(CHUNK_FWD_HO_FUSED_UB_O_V2_OUT_HALF_OFFSET +
                      CHUNK_FWD_HO_FUSED_UB_O_V2_OUT_HALF_BYTES <=
                  CHUNK_FWD_HO_FUSED_UB_O_V2_OUT_PONG_OFFSET &&
                  CHUNK_FWD_HO_FUSED_UB_O_V2_OUT_PONG_OFFSET +
                      CHUNK_FWD_HO_FUSED_UB_O_V2_OUT_BYTES <=
                  CHUNK_FWD_HO_FUSED_UB_O_V2_OUT_HALF_PONG_OFFSET &&
                  CHUNK_FWD_HO_FUSED_UB_O_V2_OUT_HALF_PONG_OFFSET +
                      CHUNK_FWD_HO_FUSED_UB_O_V2_OUT_HALF_BYTES ==
                  CHUNK_FWD_HO_FUSED_UB_USED_MAX,
              "fused UB: OV2 output ping/pong views overlap or exceed the footprint");
static_assert(CHUNK_FWD_HO_FUSED_UB_USED_MAX <=
                  static_cast<uint32_t>(CHUNK_FWD_HO_FUSED_UB_DATA_LIMIT),
              "fused UB: footprint crosses the UB data limit");
static_assert(static_cast<uint32_t>(CHUNK_FWD_HO_FUSED_UB_DATA_LIMIT) +
                      static_cast<uint32_t>(CHUNK_FWD_HO_FUSED_UB_RESERVED_TAIL) <=
                  CHUNK_FWD_HO_A5_UB_CAPACITY,
              "fused UB: data limit plus reserved tail exceeds the A5 UB capacity");

} // namespace GDN

#endif // CHUNK_FWD_HO_FUSED_A5_SINGLE_SLOT_UB_LAYOUT_H
