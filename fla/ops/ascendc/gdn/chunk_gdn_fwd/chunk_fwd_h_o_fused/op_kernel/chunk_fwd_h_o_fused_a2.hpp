/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */

#ifndef CHUNK_FWD_H_O_FUSED_A2_HPP
#define CHUNK_FWD_H_O_FUSED_A2_HPP

#if !defined(CHUNK_FWD_HO_ARCH_A2) || defined(CHUNK_FWD_HO_ARCH35)
#error "chunk_fwd_h_o_fused_a2.hpp requires CHUNK_FWD_HO_ARCH_A2"
#endif

#include "chunk_fwd_h_o_fused_struct.h"
#include "kernel_operator.h"
#include "chunk_fwd_h_o_fused_sync.h"
#include "chunk_gated_delta_rule_fwd_h_struct.h"
#include "gemm/kernel/gdn_fwd_h_kernel.hpp"
#include "gemm/kernel/gdn_fwd_h_kernel_preload.hpp"
#undef CATLASS_ARCH
#include "chunk_fwd_o_struct.h"
#include "gemm/kernel/gdn_fwd_o_kernel.hpp"

#include "lib/matmul_intf.h"
#include <cstddef>

namespace GDN {
namespace {

static_assert(offsetof(ChunkFwdHOFusedTilingData, useExp2) ==
                  sizeof(::ChunkFwdHOFusedHStageTilingData),
              "The fused tiling H prefix must match the local H kernel view");

__aicore__ inline uint32_t GetMixedCoreIdx()
{
    if ASCEND_IS_AIV {
        return AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    }
    return AscendC::GetBlockIdx();
}

__aicore__ inline void FillOTiling(
    const ChunkFwdHOFusedTilingData &src, ChunkFwdHOFusedOStageTilingData &dst)
{
    dst.shapeBatch = src.shapeBatch;
    dst.seqlen = src.seqlen;
    dst.kNumHead = src.kNumHead;
    dst.vNumHead = src.vNumHead;
    dst.kHeadDim = src.kHeadDim;
    dst.vHeadDim = src.vHeadDim;
    dst.chunkSize = src.chunkSize;
    dst.isVariedLen = src.isVariedLen;
    dst.tokenBatch = src.tokenBatch;
    dst.dataType = src.dataType;
    dst.gDataType = src.gDataType;
    dst.vWorkspaceOffset = src.oVWorkspaceOffset;
    dst.hWorkspaceOffset = src.oHWorkspaceOffset;
    dst.attnWorkspaceOffset = src.oAttnWorkspaceOffset;
    dst.aftermaskWorkspaceOffset = src.oAfterMaskWorkspaceOffset;
    dst.maskWorkspaceOffset = src.oMaskWorkspaceOffset;
    dst.scale = src.scale;
    dst.pipelineSyncWorkspaceOffset = src.pipelineSyncWorkspaceOffset;
    dst.producerCoreNum = src.producerCoreNum;
    dst.consumerCoreBase = src.consumerCoreBase;
    dst.activeCoreNum = src.activeCoreNum;
}

template <typename InputT, typename GateT, typename StateT, typename TileShapes, bool UseGk, int V_DIM>
__aicore__ inline void RunH(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR h, GM_ADDR vNew,
    GM_ADDR finalState, GM_ADDR tiling, GM_ADDR userWorkspace)
{
    using Kernel = Catlass::Gemm::Kernel::GDNFwdHKernel<
        InputT, GateT, StateT, float, TileShapes, UseGk, true, false, true>;
    Kernel kernel;
    kernel.Init(k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
                h, vNew, finalState, tiling, userWorkspace);
    kernel.Process();
}

template <typename InputT, typename GateT, typename StateT, typename TileShapes>
__aicore__ inline void RunHPreload(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR h, GM_ADDR vNew,
    GM_ADDR finalState, GM_ADDR tiling, GM_ADDR userWorkspace)
{
    using Kernel = Catlass::Gemm::Kernel::GDNFwdHKernelPreload<
        InputT, GateT, StateT, float, true>;
    Kernel kernel;
    kernel.Init(k, w, u, g, initialState, cuSeqlens, chunkIndices,
                h, vNew, finalState, tiling, userWorkspace);
    kernel.Process();
}

template <typename InputT, typename GateT, typename StateT, typename TileShapes, int V_DIM>
__aicore__ inline void DispatchHByGk(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR h, GM_ADDR vNew,
    GM_ADDR finalState, GM_ADDR tiling, GM_ADDR userWorkspace, bool useGk)
{
    if (useGk) {
        RunH<InputT, GateT, StateT, TileShapes, true, V_DIM>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
            h, vNew, finalState, tiling, userWorkspace);
    } else if constexpr (V_DIM == 128) {
        RunHPreload<InputT, GateT, StateT, TileShapes>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
            h, vNew, finalState, tiling, userWorkspace);
    } else {
        RunH<InputT, GateT, StateT, TileShapes, false, V_DIM>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
            h, vNew, finalState, tiling, userWorkspace);
    }
}

template <typename InputT, typename TileShapes, int V_DIM>
__aicore__ inline void DispatchH(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR h, GM_ADDR vNew,
    GM_ADDR finalState, GM_ADDR tiling, GM_ADDR userWorkspace,
    const ChunkFwdHOFusedTilingData &data)
{
    constexpr int64_t DTYPE_FP32 = 2;
    if (data.stateDataType == DTYPE_FP32) {
        if (data.gDataType == DTYPE_FP32) {
            DispatchHByGk<InputT, float, float, TileShapes, V_DIM>(
                k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
                h, vNew, finalState, tiling, userWorkspace, data.useGk);
        } else {
            DispatchHByGk<InputT, InputT, float, TileShapes, V_DIM>(
                k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
                h, vNew, finalState, tiling, userWorkspace, data.useGk);
        }
    } else if (data.gDataType == DTYPE_FP32) {
        DispatchHByGk<InputT, float, InputT, TileShapes, V_DIM>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
            h, vNew, finalState, tiling, userWorkspace, data.useGk);
    } else {
        DispatchHByGk<InputT, InputT, InputT, TileShapes, V_DIM>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
            h, vNew, finalState, tiling, userWorkspace, data.useGk);
    }
}

template <typename InputT, typename GateT>
__aicore__ inline void RunO(
    GM_ADDR q, GM_ADDR k, GM_ADDR vNew, GM_ADDR h, GM_ADDR g,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o, GM_ADDR userWorkspace,
    const ChunkFwdHOFusedOStageTilingData &tiling)
{
    using Kernel = Catlass::Gemm::Kernel::GDNFwdOKernel<InputT, GateT, float, true>;
    Kernel kernel;
    kernel.Init(q, k, vNew, h, g, cuSeqlens, chunkIndices, o, &tiling, userWorkspace);
    kernel.Process();
}

template <typename InputT>
__aicore__ inline void DispatchO(
    GM_ADDR q, GM_ADDR k, GM_ADDR vNew, GM_ADDR h, GM_ADDR g,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o, GM_ADDR userWorkspace,
    const ChunkFwdHOFusedTilingData &data)
{
    constexpr int64_t DTYPE_FP32 = 2;
    ChunkFwdHOFusedOStageTilingData oTiling{};
    FillOTiling(data, oTiling);
    if (data.gDataType == DTYPE_FP32) {
        RunO<InputT, float>(q, k, vNew, h, g, cuSeqlens, chunkIndices,
                            o, userWorkspace, oTiling);
    } else {
        RunO<InputT, InputT>(q, k, vNew, h, g, cuSeqlens, chunkIndices,
                             o, userWorkspace, oTiling);
    }
}

template <typename InputT, typename TileShapes, int V_DIM>
__aicore__ inline void RunChunkFwdHOFused(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initialState,
    GM_ADDR q, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
    GM_ADDR finalState, GM_ADDR workspace, GM_ADDR tiling,
    const ChunkFwdHOFusedTilingData &data)
{
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    GM_ADDR h = userWorkspace + data.handoffHWorkspaceOffset;
    GM_ADDR vNew = userWorkspace + data.handoffVWorkspaceOffset;

    if (GetMixedCoreIdx() < static_cast<uint32_t>(data.producerCoreNum)) {
        DispatchH<InputT, TileShapes, V_DIM>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
            h, vNew, finalState, tiling, userWorkspace, data);
    } else {
        DispatchO<InputT>(q, k, vNew, h, g, cuSeqlens, chunkIndices,
                          o, userWorkspace, data);
    }
}

} // namespace
} // namespace GDN

#endif // CHUNK_FWD_H_O_FUSED_A2_HPP
