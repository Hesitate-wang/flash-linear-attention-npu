/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */

#include "chunk_fwd_h_o_fused_struct.h"
#include "chunk_fwd_h_o_fused_arch.h"
#include "kernel_operator.h"

#if defined(CHUNK_FWD_HO_ARCH35)
#include "arch35/chunk_fwd_h_o_fused_a5.hpp"
#elif defined(CHUNK_FWD_HO_ARCH_A2)
#include "chunk_fwd_h_o_fused_a2.hpp"
#else
#error "Unsupported ChunkFwdHOFused architecture"
#endif

#include "lib/matmul_intf.h"

extern "C" __global__ __aicore__ void chunk_fwd_h_o_fused(
    GM_ADDR k, GM_ADDR q, GM_ADDR w, GM_ADDR u, GM_ADDR g,
    GM_ADDR gk, GM_ADDR initial_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR o, GM_ADDR final_state, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GDN::ChunkFwdHOFusedTilingData);
    GET_TILING_DATA_WITH_STRUCT(GDN::ChunkFwdHOFusedTilingData, tilingData, tiling);

    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunChunkFwdHOFused<
            DTYPE_K, Catlass::Gemm::Kernel::GDNFwdHTileShapes128>(
            k, w, u, g, gk, initial_state, q, cu_seqlens, chunk_indices,
            o, final_state, workspace, tiling, tilingData);
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunChunkFwdHOFused<
            DTYPE_K, Catlass::Gemm::Kernel::GDNFwdHTileShapes256>(
            k, w, u, g, gk, initial_state, q, cu_seqlens, chunk_indices,
            o, final_state, workspace, tiling, tilingData);
    }
}
