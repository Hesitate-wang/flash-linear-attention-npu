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
#undef CATLASS_ARCH
#include "gemm/kernel/gdn_fwd_o_kernel.hpp"
#include "chunk_fwd_o_a5.h"
#include <cstddef>

namespace GDN {

__aicore__ inline uint32_t GetMixedCoreIdx()
{
    if ASCEND_IS_AIV {
        return AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    }
    return AscendC::GetBlockIdx();
}

// Initialize every ready/ACK slot before producer and consumer cores take
// different paths; the user workspace is not guaranteed to be zeroed.
__aicore__ inline void InitializePipelineSync(
    GM_ADDR userWorkspace, const ChunkFwdHOFusedTilingData &tiling)
{
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> syncBuf;
        pipe.InitBuffer(syncBuf, CHUNK_FWD_HO_IB_WORDS_PER_EVENT * sizeof(int32_t));
        AscendC::LocalTensor<int32_t> syncLocal = syncBuf.Get<int32_t>();
        AscendC::Duplicate(syncLocal, static_cast<int32_t>(0), CHUNK_FWD_HO_IB_WORDS_PER_EVENT);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::GlobalTensor<int32_t> syncGm;
        syncGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            userWorkspace + tiling.pipelineSyncWorkspaceOffset));
        const uint32_t logicalAivNum =
            static_cast<uint32_t>(tiling.activeCoreNum) * AscendC::GetSubBlockNum();
        const uint32_t logicalAivIdx = AscendC::GetBlockIdx();
        if (logicalAivIdx < logicalAivNum) {
            for (uint32_t eventId = 0;
                 eventId < static_cast<uint32_t>(tiling.pipelineEventCount); ++eventId) {
                const uint32_t offset =
                    (eventId * logicalAivNum + logicalAivIdx) * CHUNK_FWD_HO_IB_WORDS_PER_EVENT;
                AscendC::DataCopy(syncGm[offset], syncLocal, CHUNK_FWD_HO_IB_WORDS_PER_EVENT);
            }
        }
        AscendC::SyncAll<false>();
    }
    if ASCEND_IS_AIC {
        AscendC::SyncAll<false>();
    }
}

static_assert(offsetof(ChunkFwdHOFusedTilingData, useExp2) ==
                  sizeof(::ChunkFwdHOFusedHStageTilingData),
              "The fused A5 tiling H prefix must match the arch35 H kernel view");

__aicore__ inline void FillOptimizedOTiling(const ChunkFwdHOFusedTilingData &src,
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
    dst.producerCoreNum = src.producerCoreNum;
    dst.consumerCoreBase = src.consumerCoreBase;
    dst.activeCoreNum = src.activeCoreNum;
}

__aicore__ inline void FillNaturalOTiling(
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

template <typename InputT, typename GateT, typename StateT,
          typename TileShapes, bool UseGk, bool UseExp2>
__aicore__ inline void RunH(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR tiling,
    GM_ADDR userWorkspace)
{
    using Kernel = Catlass::Gemm::Kernel::GDNFwdHKernel<
        InputT, GateT, StateT, float, TileShapes, UseGk, true, UseExp2, true, !UseExp2>;
    Kernel kernel;
    kernel.Init(k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
                h, vNew, finalState, tiling, userWorkspace);
    kernel.Process();
}

template <typename InputT, typename GateT, typename StateT,
          typename TileShapes, bool UseExp2>
__aicore__ inline void DispatchHByGk(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR tiling,
    GM_ADDR userWorkspace, bool useGk)
{
    if (useGk) {
        RunH<InputT, GateT, StateT, TileShapes, true, UseExp2>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
            h, vNew, finalState, tiling, userWorkspace);
    } else {
        RunH<InputT, GateT, StateT, TileShapes, false, UseExp2>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
            h, vNew, finalState, tiling, userWorkspace);
    }
}

template <typename InputT, typename TileShapes, bool UseExp2>
__aicore__ inline void DispatchH(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR tiling,
    GM_ADDR userWorkspace, const ChunkFwdHOFusedTilingData &data)
{
    constexpr int64_t DTYPE_FP32 = 2;
    if (data.stateDataType == DTYPE_FP32) {
        if (data.gDataType == DTYPE_FP32) {
            DispatchHByGk<InputT, float, float, TileShapes, UseExp2>(
                k, w, u, g, gk, initialState,
                cuSeqlens, chunkIndices, h, vNew, finalState, tiling,
                userWorkspace, data.useGk);
        } else {
            DispatchHByGk<InputT, InputT, float, TileShapes, UseExp2>(
                k, w, u, g, gk, initialState,
                cuSeqlens, chunkIndices, h, vNew, finalState, tiling,
                userWorkspace, data.useGk);
        }
    } else if (data.gDataType == DTYPE_FP32) {
        DispatchHByGk<InputT, float, InputT, TileShapes, UseExp2>(
            k, w, u, g, gk, initialState,
            cuSeqlens, chunkIndices, h, vNew, finalState, tiling,
            userWorkspace, data.useGk);
    } else {
        DispatchHByGk<InputT, InputT, InputT, TileShapes, UseExp2>(
            k, w, u, g, gk, initialState,
            cuSeqlens, chunkIndices, h, vNew, finalState, tiling,
            userWorkspace, data.useGk);
    }
}

template <typename GateT>
__aicore__ inline void RunOptimizedO(
    GM_ADDR q, GM_ADDR k, GM_ADDR vNew, GM_ADDR h, GM_ADDR g,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
    GM_ADDR userWorkspace, const ChunkFwdOTilingData &tiling)
{
    ChunkFwdOA5Dispatch<GateT, true>(q, k, vNew, h, g, cuSeqlens,
                                    chunkIndices, o, userWorkspace, &tiling);
}

template <typename InputT, typename GateT>
__aicore__ inline void RunNaturalO(
    GM_ADDR q, GM_ADDR k, GM_ADDR vNew, GM_ADDR h, GM_ADDR g,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
    GM_ADDR userWorkspace, const ChunkFwdHOFusedOStageTilingData &tiling)
{
    using Kernel = Catlass::Gemm::Kernel::GDNFwdOKernel<InputT, GateT, float, true>;
    Kernel kernel;
    kernel.Init(q, k, vNew, h, g, cuSeqlens, chunkIndices,
                o, &tiling, userWorkspace);
    kernel.Process();
}

template <typename InputT>
__aicore__ inline void DispatchNaturalO(
    GM_ADDR q, GM_ADDR k, GM_ADDR vNew, GM_ADDR h, GM_ADDR g,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
    GM_ADDR userWorkspace, const ChunkFwdHOFusedTilingData &data)
{
    ChunkFwdHOFusedOStageTilingData oTiling{};
    FillNaturalOTiling(data, oTiling);
    constexpr int64_t DTYPE_FP32 = 2;
    if (data.gDataType == DTYPE_FP32) {
        RunNaturalO<InputT, float>(q, k, vNew, h, g, cuSeqlens,
                                   chunkIndices, o, userWorkspace, oTiling);
    } else {
        RunNaturalO<InputT, InputT>(q, k, vNew, h, g, cuSeqlens,
                                    chunkIndices, o, userWorkspace, oTiling);
    }
}

template <typename InputT, typename TileShapes, bool UseExp2>
__aicore__ inline void RunTyped(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
    GM_ADDR initialState, GM_ADDR q, GM_ADDR cuSeqlens,
    GM_ADDR chunkIndices, GM_ADDR o, GM_ADDR finalState,
    GM_ADDR workspace, GM_ADDR tiling,
    const ChunkFwdHOFusedTilingData &data)
{
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    GM_ADDR h = userWorkspace + data.handoffHWorkspaceOffset;
    GM_ADDR vNew = userWorkspace + data.handoffVWorkspaceOffset;

    InitializePipelineSync(userWorkspace, data);
    if (GetMixedCoreIdx() < static_cast<uint32_t>(data.producerCoreNum)) {
        DispatchH<InputT, TileShapes, UseExp2>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
            h, vNew, finalState, tiling, userWorkspace, data);
        // The copied arch35 exp2 O implementation does not yet expose the
        // generic O IBWait hook. Keep its handoff ordered until that hook is
        // added; the natural-exp path remains fully chunk-pipelined.
        if constexpr (UseExp2) {
            AscendC::SyncAll<false>();
        }
        return;
    }

    if constexpr (UseExp2) {
        AscendC::SyncAll<false>();
        ChunkFwdOTilingData oTiling{};
        FillOptimizedOTiling(data, oTiling);
        constexpr int64_t DTYPE_FP32 = 2;
        if (data.gDataType == DTYPE_FP32) {
            RunOptimizedO<float>(q, k, vNew, h, g, cuSeqlens,
                                 chunkIndices, o, userWorkspace, oTiling);
        } else {
            RunOptimizedO<InputT>(q, k, vNew, h, g, cuSeqlens,
                                  chunkIndices, o, userWorkspace, oTiling);
        }
    } else {
        DispatchNaturalO<InputT>(q, k, vNew, h, g, cuSeqlens,
                                 chunkIndices, o, userWorkspace, data);
    }
}

template <typename InputT, typename TileShapes>
__aicore__ inline void RunChunkFwdHOFused(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
    GM_ADDR initialState, GM_ADDR q, GM_ADDR cuSeqlens,
    GM_ADDR chunkIndices, GM_ADDR o, GM_ADDR finalState,
    GM_ADDR workspace, GM_ADDR tiling,
    const ChunkFwdHOFusedTilingData &data)
{
    if (data.useExp2) {
        RunTyped<InputT, TileShapes, true>(
            k, w, u, g, gk, initialState, q, cuSeqlens, chunkIndices,
            o, finalState, workspace, tiling, data);
        return;
    }

    RunTyped<InputT, TileShapes, false>(
        k, w, u, g, gk, initialState, q, cuSeqlens, chunkIndices,
        o, finalState, workspace, tiling, data);
}

} // namespace GDN

#endif // CHUNK_FWD_H_O_FUSED_ARCH35_A5_HPP
