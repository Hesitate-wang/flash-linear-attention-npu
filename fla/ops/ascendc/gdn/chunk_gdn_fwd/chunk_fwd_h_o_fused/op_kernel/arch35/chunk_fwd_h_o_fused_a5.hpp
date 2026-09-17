/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef CHUNK_FWD_H_O_FUSED_ARCH35_A5_HPP
#define CHUNK_FWD_H_O_FUSED_ARCH35_A5_HPP

#include "../chunk_gated_delta_rule_fwd_h_struct.h"
#include "../chunk_fwd_o_struct.h"

using ChunkGatedDeltaRuleFwdHTilingData = ChunkFwdHOFusedHStageTilingData;

#include "gemm/kernel/gdn_fwd_h_kernel.hpp"
#include "chunk_fwd_o_a5.h"
#include <cstddef>

namespace GDN::Arch35 {

static_assert(offsetof(ChunkFwdHOFusedTilingData, useExp2) ==
                  sizeof(::ChunkFwdHOFusedHStageTilingData),
              "The fused A5 tiling H prefix must match the arch35 H kernel view");

__aicore__ inline void FillOTiling(const ChunkFwdHOFusedTilingData &src,
                                   ChunkFwdOTilingData &dst)
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
    dst.vWorkspaceOffset = 0;
    dst.hWorkspaceOffset = 0;
    dst.attnWorkspaceOffset = 0;
    dst.aftermaskWorkspaceOffset = 0;
    dst.maskWorkspaceOffset = 0;
    dst.stateVFirst = 0;
    dst.outputLayout = src.outputLayout;
    dst.scale = src.scale;
    dst.chunkNum = src.chunkNum;
    dst.hvPerHk = src.hvPerHk;
    dst.taskGroupSize = src.taskGroupSize;
    dst.numChunksPerBatch = src.numChunksPerBatch;
    dst.aPrimeWorkspaceOffset = src.oAPrimeWorkspaceOffset;
}

template <typename GateT, typename StateT, bool UseGk>
__aicore__ inline void RunH(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR tiling,
    GM_ADDR userWorkspace)
{
    using Kernel = Catlass::Gemm::Kernel::GDNFwdHKernel<
        bfloat16_t, GateT, StateT, float,
        Catlass::Gemm::Kernel::GDNFwdHTileShapes128,
        UseGk, true, false>;
    Kernel kernel;
    kernel.Init(k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
                h, vNew, finalState, tiling, userWorkspace);
    kernel.Process();
}

template <typename GateT, typename StateT>
__aicore__ inline void DispatchHByGk(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR tiling,
    GM_ADDR userWorkspace, bool useGk)
{
    if (useGk) {
        RunH<GateT, StateT, true>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
            h, vNew, finalState, tiling, userWorkspace);
    } else {
        RunH<GateT, StateT, false>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
            h, vNew, finalState, tiling, userWorkspace);
    }
}

__aicore__ inline void DispatchH(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR tiling,
    GM_ADDR userWorkspace, const ChunkFwdHOFusedTilingData &data)
{
    constexpr int64_t DTYPE_FP32 = 2;
    if (data.stateDataType == DTYPE_FP32) {
        if (data.gDataType == DTYPE_FP32) {
            DispatchHByGk<float, float>(k, w, u, g, gk, initialState,
                cuSeqlens, chunkIndices, h, vNew, finalState, tiling,
                userWorkspace, data.useGk);
        } else {
            DispatchHByGk<bfloat16_t, float>(k, w, u, g, gk, initialState,
                cuSeqlens, chunkIndices, h, vNew, finalState, tiling,
                userWorkspace, data.useGk);
        }
    } else if (data.gDataType == DTYPE_FP32) {
        DispatchHByGk<float, bfloat16_t>(k, w, u, g, gk, initialState,
            cuSeqlens, chunkIndices, h, vNew, finalState, tiling,
            userWorkspace, data.useGk);
    } else {
        DispatchHByGk<bfloat16_t, bfloat16_t>(k, w, u, g, gk, initialState,
            cuSeqlens, chunkIndices, h, vNew, finalState, tiling,
            userWorkspace, data.useGk);
    }
}

template <typename GateT>
__aicore__ inline void RunO(
    GM_ADDR q, GM_ADDR k, GM_ADDR vNew, GM_ADDR h, GM_ADDR g,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
    GM_ADDR userWorkspace, const ChunkFwdOTilingData &tiling)
{
    ChunkFwdOA5Dispatch<GateT, true>(q, k, vNew, h, g, cuSeqlens,
                                    chunkIndices, o, userWorkspace, &tiling);
}

__aicore__ inline void RunChunkFwdHOFusedA5(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
    GM_ADDR initialState, GM_ADDR q, GM_ADDR cuSeqlens,
    GM_ADDR chunkIndices, GM_ADDR o, GM_ADDR finalState,
    GM_ADDR workspace, GM_ADDR tiling,
    const ChunkFwdHOFusedTilingData &data)
{
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    GM_ADDR h = userWorkspace + data.handoffHWorkspaceOffset;
    GM_ADDR vNew = userWorkspace + data.handoffVWorkspaceOffset;

    DispatchH(k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
              h, vNew, finalState, tiling, userWorkspace, data);

    // H drains its local events before returning. This global stage boundary
    // makes all H GM writes visible before any AIC/AIV starts the O stage.
    AscendC::SyncAll<false>();

    ChunkFwdOTilingData oTiling{};
    FillOTiling(data, oTiling);
    constexpr int64_t DTYPE_FP32 = 2;
    if (data.gDataType == DTYPE_FP32) {
        RunO<float>(q, k, vNew, h, g, cuSeqlens, chunkIndices, o,
                    userWorkspace, oTiling);
    } else {
        RunO<bfloat16_t>(q, k, vNew, h, g, cuSeqlens, chunkIndices, o,
                         userWorkspace, oTiling);
    }
}

} // namespace GDN::Arch35

#endif // CHUNK_FWD_H_O_FUSED_ARCH35_A5_HPP
