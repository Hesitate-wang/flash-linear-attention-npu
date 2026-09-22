/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * A5 UB layout shared by the fused H/O kernels and their epilogues.
 */

#ifndef CHUNK_FWD_H_O_FUSED_A5_UB_LAYOUT_H
#define CHUNK_FWD_H_O_FUSED_A5_UB_LAYOUT_H

#include <cstdint>

#include "../chunk_fwd_h_o_fused_struct.h"

namespace GDN {

constexpr uint32_t CHUNK_FWD_HO_A5_UB_CAPACITY = 256U * 1024U;

// Keep the final 8 KiB outside every Cube/Fixpipe/Vector data allocation. IBSet
// and IBWait use the first 32 bytes as their local operand scratch.
constexpr uint32_t CHUNK_FWD_HO_A5_COMM_UB_OFFSET = 248U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_A5_COMM_UB_BYTES = 8U * 1024U;
constexpr uint32_t CHUNK_FWD_HO_A5_IB_LOCAL_UB_OFFSET = CHUNK_FWD_HO_A5_COMM_UB_OFFSET;
constexpr uint32_t CHUNK_FWD_HO_A5_IB_LOCAL_UB_BYTES =
    static_cast<uint32_t>(CHUNK_FWD_HO_IB_WORDS_PER_EVENT * sizeof(int32_t));

static_assert(CHUNK_FWD_HO_A5_COMM_UB_OFFSET % 32U == 0U,
              "IB communication UB must be 32-byte aligned");
static_assert(CHUNK_FWD_HO_A5_IB_LOCAL_UB_BYTES <= CHUNK_FWD_HO_A5_COMM_UB_BYTES,
              "IB local operand exceeds the reserved communication UB");
static_assert(CHUNK_FWD_HO_A5_COMM_UB_OFFSET + CHUNK_FWD_HO_A5_COMM_UB_BYTES <=
                  CHUNK_FWD_HO_A5_UB_CAPACITY,
              "IB communication UB exceeds the A5 UB capacity");

} // namespace GDN

#endif // CHUNK_FWD_H_O_FUSED_A5_UB_LAYOUT_H
