/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_EPILOGUE_GDN_FUSED_EPILOGUE_POLICIES_HPP
#define CATLASS_EPILOGUE_GDN_FUSED_EPILOGUE_POLICIES_HPP

#include <cstdint>
#include "catlass/catlass.hpp"

namespace Catlass::Epilogue {

// Separate tag: including this header alongside the existing H/O policies must
// not redefine their tags or select the legacy QK-mask specialization.
struct EpilogueAtlasGDNFusedMode2Qkmask {
    using ArchTag = Arch::Ascend950;
};

struct gdnFusedMode2UbMallocTable
{
    //chunk_size = 64, dim = 128
    // Two AIV lanes: 32 token rows or 64 state rows per lane.
    /* ov1
    oc1 -> attn
    g_left = broadcast(g, shared_mem, left), g_up = broadcast(g,shared_mem,up)
    g_left = g_left - g_up
    g_left = exp(min(g_left, 0)) * mask
    g_left = attn * g_left
    */
    static constexpr uint32_t attn_ping = 0U; // 32 * 64 * 4 = 8KB
    static constexpr uint32_t attn_pong = attn_ping;
    static constexpr uint32_t g_left_ping = 8 * 1024U;
    static constexpr uint32_t g_left_pong = g_left_ping;
    static constexpr uint32_t g_up_ping = 16 * 1024U;
    static constexpr uint32_t g_up_pong = g_up_ping;
    static constexpr uint32_t mask = 24 * 1024U;  // 32 * 32 * 4 = 4KB
    static constexpr uint32_t gUbTensorBase = 28 * 1024U;
    static constexpr uint32_t g_input_ping = gUbTensorBase;  // 64 * 1 * 2 = 128B
    static constexpr uint32_t g_input_pong = g_input_ping;
    static constexpr uint32_t g_fp32_ping = gUbTensorBase + 512U; // 64 * 1 * 4 = 256B
    static constexpr uint32_t g_fp32_pong = g_fp32_ping;
    // Broadcast workspace, then a transient 32x64 mask assembled from mask.
    static constexpr uint32_t shard_mem = 29 * 1024U; // 8KB, end at 37KB
    static constexpr uint32_t broadcast_scratch_bytes = 8 * 1024U;
    static constexpr uint32_t chunk_tokens = 64U;
    static constexpr uint32_t rows_per_aiv = 32U;
    static constexpr uint32_t head_dim = 128U;

    /*hv1
    hc1 -> v_work
    g_last = g_last - g
    v_new = U - v_work
    v_new_decay = v_new * g_last
    */
    static constexpr uint32_t v_work_ping = 37 * 1024U;  // 32 * 128 * 4 = 16KB
    static constexpr uint32_t v_work_pong = v_work_ping;
    static constexpr uint32_t u_ping = 53 * 1024U; // 32 * 128 * 2 = 8KB
    static constexpr uint32_t u_pong = u_ping;
    static constexpr uint32_t u_fp_ping = 61 * 1024U;  // 16KB
    static constexpr uint32_t u_fp_pong = u_fp_ping;
    static constexpr uint32_t v_new_ping = 77 * 1024U;
    static constexpr uint32_t v_new_pong = v_new_ping;
    static constexpr uint32_t v_new_decay_ping = 93 * 1024U;
    static constexpr uint32_t v_new_decay_pong = v_new_decay_ping; // end at 109KB

    /*hv2
    h = h * g_last, matrix * scalar
    hc2-> h_work
    h = h + h_work
    */
    static constexpr uint32_t h_ping = 109 * 1024U;  // 64 * 128 * 4 = 32KB per AIV, end at 141KB
    static constexpr uint32_t h_pong = h_ping;
    static constexpr uint32_t h_work_ping = 141 * 1024U; //
    static constexpr uint32_t h_work_pong = h_work_ping; // end at 173KB


    /*ov2
    oc2 -> o_inter
    o_inter = o_inter * g_last_scalar
    oc3 -> o_intra
    o_inter = o_inter + o_intra
    o_inter = o_inter * scale
    o_out = cast(o_inter)
    */
    static constexpr uint32_t o_inter_ping = 173 * 1024U; // 32 * 128 * 4 = 16KB, independent OC2 destination;
    static constexpr uint32_t o_inter_pong = o_inter_ping;
    static constexpr uint32_t o_intra_ping = 189 * 1024U;
    static constexpr uint32_t o_intra_pong = o_intra_ping;
    static constexpr uint32_t o_out_ping = 207 * 1024U; // 32 * 128 * 2 = 8KB
    static constexpr uint32_t o_out_pong = o_out_ping; // single slot, end at 215KB

    // DataCopy block addresses must remain 32-byte aligned.
    static_assert(attn_ping % 32U == 0 && g_left_ping % 32U == 0 &&
                  g_up_ping % 32U == 0 && mask % 32U == 0 &&
                  g_input_ping % 32U == 0 && g_fp32_ping % 32U == 0 &&
                  shard_mem % 32U == 0 && v_work_ping % 32U == 0 &&
                  u_ping % 32U == 0 && u_fp_ping % 32U == 0 &&
                  v_new_ping % 32U == 0 && v_new_decay_ping % 32U == 0 &&
                  h_ping % 32U == 0 && h_work_ping % 32U == 0 &&
                  o_inter_ping % 32U == 0 && o_intra_ping % 32U == 0 &&
                  o_out_ping % 32U == 0,
                  "MODE2 UB addresses must be 32-byte aligned");

    // OV1 and the shared gate/mask remain separate from Cube destinations.
    static_assert(attn_ping + 32U * 64U * sizeof(float) <= g_left_ping,
                  "OV1 attn overlaps left broadcast");
    static_assert(g_left_ping + 32U * 64U * sizeof(float) <= g_up_ping,
                  "OV1 left and up broadcasts overlap");
    static_assert(g_up_ping + 32U * 64U * sizeof(float) <= mask,
                  "OV1 up broadcast overlaps resident mask");
    static_assert(mask + 32U * 32U * sizeof(float) <= g_input_ping,
                  "resident mask overlaps gate input");
    // GateT may be FP32: reserve the largest supported input here.
    static_assert(g_input_ping + 64U * sizeof(float) <= g_fp32_ping,
                  "gate input overlaps FP32 gate");
    static_assert(g_fp32_ping + 64U * sizeof(float) <= shard_mem,
                  "FP32 gate overlaps Broadcast workspace");
    static_assert(shard_mem + broadcast_scratch_bytes <= v_work_ping,
                  "OV1 workspace overlaps HC1 output");

    // HV1: 32 token rows per AIV; two-byte input and cast output.
    static_assert(v_work_ping + 32U * 128U * sizeof(float) <= u_ping,
                  "HC1 output overlaps U input");
    static_assert(u_ping + 32U * 128U * 2U <= u_fp_ping,
                  "U input/cast output overlaps FP32 U");
    static_assert(u_fp_ping + 32U * 128U * sizeof(float) <= v_new_ping,
                  "FP32 U overlaps Vnew");
    static_assert(v_new_ping + 32U * 128U * sizeof(float) <= v_new_decay_ping,
                  "Vnew overlaps Vnew decay");
    static_assert(v_new_decay_ping + 32U * 128U * sizeof(float) <= h_ping,
                  "Vnew decay overlaps resident H");

    // HV2: 64 state rows per AIV. Reuse the finished decay slot for DataT H.
    static_assert(v_new_decay_ping + 64U * 128U * 2U <= h_ping,
                  "HV2 staging overlaps resident H");
    static_assert(h_ping + 64U * 128U * sizeof(float) <= h_work_ping,
                  "resident H overlaps HC2 output");
    static_assert(h_work_ping + 64U * 128U * sizeof(float) <= o_inter_ping,
                  "HC2 output overlaps OC2 output");

    // OV2 has independent destinations; the final 8 KiB output must stay
    // below the communication region beginning at 248 KiB. Gaps are allowed.
    static_assert(o_inter_ping + 32U * 128U * sizeof(float) <= o_intra_ping,
                  "OC2 and OC3 outputs overlap");
    static_assert(o_intra_ping + 32U * 128U * sizeof(float) <= o_out_ping,
                  "OC3 output overlaps cast output");
    static_assert(o_out_ping + 32U * 128U * 2U <= 248U * 1024U,
                  "MODE2 output crosses the reserved communication region");
};

}  // namespace Catlass::Epilogue

#endif  // CATLASS_EPILOGUE_GDN_FUSED_EPILOGUE_POLICIES_HPP
