/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * A5 UB layout for the ChunkFwdHOFused SINGLE_CORE_CHUNK_FUSED schedule
 * (executionMode = 2, blockDim = T).
 *
 * One MIX core owns one (batch, value-head) task and walks its chunks strictly
 * serially. Inside every chunk the FwdH and FwdO stages alternate on the same
 * core. FwdH has one Fixpipe slot; fused narrow OV2 processes row tiles
 * serially through one UB slot:
 *
 *   AIC : [chunk gate] OC1 -> HC1 -> OC2 -> HC2 -> [wait HV1] OC3
 *   AIV : [mask] OV1 -> [O/H fence] HV1 -> HV2 -> [H/O fence] OV2
 *         -> [O fence, chunk finished]
 *
 * Variable/lifetime layout (V128; per AIV, KiB half-open):
 *
 *   Init:  state [183,215), H handoff [215,247).
 *   OV1:   QK scratch [16,111), H_DIRECT [117,181), gate/input [181,183),
 *          mask [231,247).
 *   HV:    calc [0,32), V tile [32,48), H tile [48,64), H_DIRECT [117,181),
 *          gate/input [181,183), G_LAST [247,248).
 *   OV2:   output [72,96), A/H work [183,231), gate/input [181,183).
 *
 * Init staging is moved out of calc, mask is disjoint from calc, G_LAST has a
 * dedicated slot, and OV2 work no longer aliases H_DIRECT. Ordered phase
 * reuse remains only where the 248 KiB data limit requires it. [248,256) is
 * reserved for communication; the active high-water mark is 248 KiB.
 *
 * Invariants enforced by the fused kernel:
 *   I1. HC1/HC2 write H_DIRECT through Fixpipe and the matching AIV epilogue
 *       reads that same window in place. HC1/HC2 readiness uses mode-0x2
 *       flags; HV1 publishes HC2's input only after reading HC1's result.
 *   I2. H_DIRECT stays disjoint from all OV1 views while Cube Fixpipe and AIV
 *       QK-mask execute concurrently; OV2 has a separate work/output range.
 *       The next HC1 cannot start until OV2 retires the chunk.
 *   I3. The causal mask has a dedicated OV1 range, is initialized once per
 *       AIV, and remains read-only for all chunks.
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

// MODE2 UB is allocated by temporary variable, independent of H/O ownership.
// Each interval is half-open and per AIV. Keep phase temporaries in distinct
// ranges where the 248 KiB data limit permits; remaining aliases are ordered
// by ProcessAiv fences. OV2 serializes row tiles before reusing its output slot.
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_CALC_OFFSET = 0U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_CALC_BYTES = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_INIT_STATE_OFFSET = 183U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_INIT_STATE_BYTES = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_MASK_OFFSET = 231U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_MASK_BYTES = 16U * 1024U;

constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_INIT_H_OFFSET = 215U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_INIT_H_BYTES = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_V_TILE_OFFSET =
    CHUNK_FWD_HO_FUSED_UB_VAR_INIT_H_OFFSET;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_V_TILE_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_H_TILE_OFFSET = 48U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_H_TILE_BYTES = 16U * 1024U;

constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_QK_LEFT_OFFSET = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_QK_LEFT_BYTES = 20U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_QK_UP_OFFSET = 36U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_QK_UP_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_QK_GCOMP_OFFSET = 52U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_QK_GCOMP_BYTES = 1U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_BROADCAST_SCRATCH_OFFSET = 53U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_BROADCAST_SCRATCH_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_GATE_OFFSET = 181U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_GATE_BYTES = 1U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_G_LAST_OFFSET = 247U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_G_LAST_BYTES = 1U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_GATE_INPUT_OFFSET = 182U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_GATE_INPUT_BYTES = 1U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_QK_A_OFFSET = 71U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_QK_A_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_QK_OUT_OFFSET = 87U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_QK_OUT_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_QK_OUT_LOW_OFFSET = 103U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_QK_OUT_LOW_BYTES = 8U * 1024U;

// AIC Fixpipe writes this slot while AIV runs OV1/HV. OV2 has a separate work
// region and does not alias H_DIRECT.
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_H_DIRECT_OFFSET = 117U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_H_DIRECT_BYTES = 64U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_O_WORK_A_OFFSET = 183U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_O_WORK_A_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_O_WORK_H_OFFSET = 199U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_O_WORK_H_BYTES = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_O_GATE_OFFSET =
    CHUNK_FWD_HO_FUSED_UB_VAR_GATE_OFFSET;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_O_GATE_BYTES = 1U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_O_GATE_INPUT_OFFSET =
    CHUNK_FWD_HO_FUSED_UB_VAR_GATE_INPUT_OFFSET;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_O_GATE_INPUT_BYTES = 1U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_O_OUT_OFFSET = 72U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_O_OUT_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_O_OUT_LOW_OFFSET = 88U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_O_OUT_LOW_BYTES = 8U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_O_WORK_END =
    CHUNK_FWD_HO_FUSED_UB_VAR_O_WORK_H_OFFSET + CHUNK_FWD_HO_FUSED_UB_VAR_O_WORK_H_BYTES;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_O_OUT_END =
    CHUNK_FWD_HO_FUSED_UB_VAR_O_OUT_LOW_OFFSET + CHUNK_FWD_HO_FUSED_UB_VAR_O_OUT_LOW_BYTES;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_USED_MAX = 248U * 1024U;

// Compatibility addresses used only by retained, non-instantiated legacy
// single-slot helpers. The fused MODE2 kernel uses UB_VAR_* exclusively.
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_CALC_OFFSET = 0U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_CALC_BYTES = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_STAGING_OFFSET = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_STAGING_BYTES = 64U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_SMALL_OFFSET = 96U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_SMALL_BYTES = 21U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_SLOT_OFFSET = CHUNK_FWD_HO_FUSED_UB_H_STAGING_OFFSET;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_DIRECT_OFFSET = 135U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_DIRECT_BYTES = 64U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_UPDATE_SCRATCH_OFFSET = 199U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_UPDATE_SCRATCH_BYTES = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_H_UPDATE_G_OFFSET = 231U * 1024U;
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

// Lifetime-aware interval checks. Concurrent Cube Fixpipe/Vector regions are
// disjoint. Phase-ordered reuse is permitted only across the explicit fences.
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_H_DIRECT_END =
    CHUNK_FWD_HO_FUSED_UB_VAR_H_DIRECT_OFFSET + CHUNK_FWD_HO_FUSED_UB_VAR_H_DIRECT_BYTES;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_QK_SCRATCH_END =
    CHUNK_FWD_HO_FUSED_UB_VAR_QK_OUT_LOW_OFFSET + CHUNK_FWD_HO_FUSED_UB_VAR_QK_OUT_LOW_BYTES;
constexpr uint32_t CHUNK_FWD_HO_FUSED_UB_VAR_GATE_END =
    CHUNK_FWD_HO_FUSED_UB_VAR_GATE_INPUT_OFFSET + CHUNK_FWD_HO_FUSED_UB_VAR_GATE_INPUT_BYTES;

static_assert(CHUNK_FWD_HO_FUSED_UB_VAR_INIT_STATE_OFFSET +
                      CHUNK_FWD_HO_FUSED_UB_VAR_INIT_STATE_BYTES <=
                  CHUNK_FWD_HO_FUSED_UB_VAR_INIT_H_OFFSET &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_INIT_H_OFFSET +
                      CHUNK_FWD_HO_FUSED_UB_VAR_INIT_H_BYTES <=
                  static_cast<uint32_t>(CHUNK_FWD_HO_FUSED_UB_DATA_LIMIT),
              "initial staging views overlap or exceed the UB data limit");
static_assert(CHUNK_FWD_HO_FUSED_UB_VAR_QK_LEFT_OFFSET +
                      CHUNK_FWD_HO_FUSED_UB_VAR_QK_LEFT_BYTES <=
                  CHUNK_FWD_HO_FUSED_UB_VAR_QK_UP_OFFSET &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_QK_UP_OFFSET +
                      CHUNK_FWD_HO_FUSED_UB_VAR_QK_UP_BYTES <=
                  CHUNK_FWD_HO_FUSED_UB_VAR_QK_GCOMP_OFFSET &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_QK_GCOMP_OFFSET +
                      CHUNK_FWD_HO_FUSED_UB_VAR_QK_GCOMP_BYTES <=
                  CHUNK_FWD_HO_FUSED_UB_VAR_BROADCAST_SCRATCH_OFFSET,
              "QK broadcast temporaries overlap");
static_assert(CHUNK_FWD_HO_FUSED_UB_VAR_QK_A_OFFSET +
                      CHUNK_FWD_HO_FUSED_UB_VAR_QK_A_BYTES <=
                  CHUNK_FWD_HO_FUSED_UB_VAR_QK_OUT_OFFSET &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_QK_OUT_OFFSET +
                      CHUNK_FWD_HO_FUSED_UB_VAR_QK_OUT_BYTES <=
                  CHUNK_FWD_HO_FUSED_UB_VAR_QK_OUT_LOW_OFFSET &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_QK_SCRATCH_END <=
                  CHUNK_FWD_HO_FUSED_UB_VAR_H_DIRECT_OFFSET &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_H_DIRECT_END <=
                  CHUNK_FWD_HO_FUSED_UB_VAR_GATE_OFFSET &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_GATE_END <=
                  CHUNK_FWD_HO_FUSED_UB_VAR_MASK_OFFSET &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_MASK_OFFSET >=
                      CHUNK_FWD_HO_FUSED_UB_VAR_QK_SCRATCH_END &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_MASK_OFFSET +
                      CHUNK_FWD_HO_FUSED_UB_VAR_MASK_BYTES <=
                      CHUNK_FWD_HO_FUSED_UB_VAR_G_LAST_OFFSET,
              "OV1 scratch must be disjoint from concurrent H Fixpipe output");
static_assert(CHUNK_FWD_HO_FUSED_UB_VAR_GATE_OFFSET +
                      CHUNK_FWD_HO_FUSED_UB_VAR_GATE_BYTES <=
                  CHUNK_FWD_HO_FUSED_UB_VAR_GATE_INPUT_OFFSET &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_GATE_END <=
                  CHUNK_FWD_HO_FUSED_UB_VAR_O_WORK_A_OFFSET &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_O_WORK_END <=
                  CHUNK_FWD_HO_FUSED_UB_VAR_G_LAST_OFFSET &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_O_GATE_OFFSET ==
                      CHUNK_FWD_HO_FUSED_UB_VAR_GATE_OFFSET &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_O_GATE_INPUT_OFFSET ==
                      CHUNK_FWD_HO_FUSED_UB_VAR_GATE_INPUT_OFFSET,
              "gate temporaries overlap or shared H/O gate views diverge");
static_assert(CHUNK_FWD_HO_FUSED_UB_VAR_O_WORK_A_OFFSET +
                      CHUNK_FWD_HO_FUSED_UB_VAR_O_WORK_A_BYTES <=
                  CHUNK_FWD_HO_FUSED_UB_VAR_O_WORK_H_OFFSET &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_O_WORK_END <=
                  CHUNK_FWD_HO_FUSED_UB_VAR_MASK_OFFSET &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_O_OUT_OFFSET +
                      CHUNK_FWD_HO_FUSED_UB_VAR_O_OUT_BYTES <=
                  CHUNK_FWD_HO_FUSED_UB_VAR_O_OUT_LOW_OFFSET &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_O_OUT_LOW_OFFSET +
                      CHUNK_FWD_HO_FUSED_UB_VAR_O_OUT_LOW_BYTES ==
                  CHUNK_FWD_HO_FUSED_UB_VAR_O_OUT_END &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_O_OUT_END <=
                      CHUNK_FWD_HO_FUSED_UB_USED_MAX &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_MASK_OFFSET +
                      CHUNK_FWD_HO_FUSED_UB_VAR_MASK_BYTES <=
                      CHUNK_FWD_HO_FUSED_UB_USED_MAX &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_G_LAST_OFFSET +
                      CHUNK_FWD_HO_FUSED_UB_VAR_G_LAST_BYTES ==
                      CHUNK_FWD_HO_FUSED_UB_USED_MAX,
              "OV2 work/output buffers overlap or exceed the declared footprint");
static_assert(CHUNK_FWD_HO_FUSED_UB_VAR_H_DIRECT_OFFSET >=
                  CHUNK_FWD_HO_FUSED_UB_VAR_QK_SCRATCH_END,
              "AIC Fixpipe and OV1 scratch are live concurrently");
static_assert(CHUNK_FWD_HO_FUSED_UB_VAR_O_WORK_A_OFFSET >=
                      CHUNK_FWD_HO_FUSED_UB_VAR_GATE_END &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_O_WORK_END <=
                      CHUNK_FWD_HO_FUSED_UB_VAR_MASK_OFFSET &&
                  CHUNK_FWD_HO_FUSED_UB_VAR_O_OUT_END <=
                      CHUNK_FWD_HO_FUSED_UB_VAR_H_DIRECT_OFFSET,
              "OV2 work/output must be disjoint from gate and H_DIRECT");
static_assert(CHUNK_FWD_HO_FUSED_UB_USED_MAX <=
                  static_cast<uint32_t>(CHUNK_FWD_HO_FUSED_UB_DATA_LIMIT),
              "fused UB: footprint crosses the UB data limit");
static_assert(static_cast<uint32_t>(CHUNK_FWD_HO_FUSED_UB_DATA_LIMIT) +
                      static_cast<uint32_t>(CHUNK_FWD_HO_FUSED_UB_RESERVED_TAIL) <=
                  CHUNK_FWD_HO_A5_UB_CAPACITY,
              "fused UB: data limit plus reserved tail exceeds the A5 UB capacity");

} // namespace GDN

#endif // CHUNK_FWD_HO_FUSED_A5_SINGLE_SLOT_UB_LAYOUT_H
