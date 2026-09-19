/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */

#include "chunk_fwd_h_o_fused_struct.h"
#include "kernel_operator.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_fwd_h_o_fused_a5.hpp"
#else
#include "chunk_gated_delta_rule_fwd_h_struct.h"
#include "gemm/kernel/gdn_fwd_h_kernel.hpp"
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

// The workspace returned by aclnn is not guaranteed to be zeroed. Every AIV
// initializes its own IB slots and all active mixed cores rendezvous once
// before producers and consumers diverge.
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
        for (uint32_t eventId = 0;
             eventId < static_cast<uint32_t>(tiling.pipelineEventCount); ++eventId) {
            const uint32_t offset =
                (eventId * logicalAivNum + logicalAivIdx) * CHUNK_FWD_HO_IB_WORDS_PER_EVENT;
            AscendC::DataCopy(syncGm[offset], syncLocal, CHUNK_FWD_HO_IB_WORDS_PER_EVENT);
        }
        AscendC::SyncAll<false>();
    }
    if ASCEND_IS_AIC {
        AscendC::SyncAll<false>();
    }
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
}

template <typename InputT, typename GateT, typename StateT, typename TileShapes, bool UseGk>
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
__aicore__ inline void DispatchHByGk(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR h, GM_ADDR vNew,
    GM_ADDR finalState, GM_ADDR tiling, GM_ADDR userWorkspace, bool useGk)
{
    if (useGk) {
        RunH<InputT, GateT, StateT, TileShapes, true>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
            h, vNew, finalState, tiling, userWorkspace);
    } else {
        RunH<InputT, GateT, StateT, TileShapes, false>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
            h, vNew, finalState, tiling, userWorkspace);
    }
}

template <typename InputT, typename TileShapes>
__aicore__ inline void DispatchH(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR h, GM_ADDR vNew,
    GM_ADDR finalState, GM_ADDR tiling, GM_ADDR userWorkspace,
    const ChunkFwdHOFusedTilingData &data)
{
    constexpr int64_t DTYPE_FP32 = 2;
    if (data.stateDataType == DTYPE_FP32) {
        if (data.gDataType == DTYPE_FP32) {
            DispatchHByGk<InputT, float, float, TileShapes>(
                k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
                h, vNew, finalState, tiling, userWorkspace, data.useGk);
        } else {
            DispatchHByGk<InputT, InputT, float, TileShapes>(
                k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
                h, vNew, finalState, tiling, userWorkspace, data.useGk);
        }
    } else if (data.gDataType == DTYPE_FP32) {
        DispatchHByGk<InputT, float, InputT, TileShapes>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
            h, vNew, finalState, tiling, userWorkspace, data.useGk);
    } else {
        DispatchHByGk<InputT, InputT, InputT, TileShapes>(
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

template <typename InputT, typename TileShapes>
__aicore__ inline void RunPipeline(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initialState,
    GM_ADDR q, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o,
    GM_ADDR finalState, GM_ADDR workspace, GM_ADDR tiling,
    const ChunkFwdHOFusedTilingData &data)
{
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    GM_ADDR h = userWorkspace + data.handoffHWorkspaceOffset;
    GM_ADDR vNew = userWorkspace + data.handoffVWorkspaceOffset;

    InitializePipelineSync(userWorkspace, data);
    if (GetMixedCoreIdx() < static_cast<uint32_t>(data.producerCoreNum)) {
        DispatchH<InputT, TileShapes>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices,
            h, vNew, finalState, tiling, userWorkspace, data);
    } else {
        DispatchO<InputT>(q, k, vNew, h, g, cuSeqlens, chunkIndices,
                          o, userWorkspace, data);
    }
}

} // namespace
} // namespace GDN
#endif

extern "C" __global__ __aicore__ void chunk_fwd_h_o_fused(
    GM_ADDR k, GM_ADDR q, GM_ADDR w, GM_ADDR u, GM_ADDR g,
    GM_ADDR gk, GM_ADDR initial_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR o, GM_ADDR final_state, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GDN::ChunkFwdHOFusedTilingData);
    GET_TILING_DATA_WITH_STRUCT(GDN::ChunkFwdHOFusedTilingData, tilingData, tiling);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GDN::Arch35::RunChunkFwdHOFusedA5(
        k, w, u, g, gk, initial_state, q, cu_seqlens, chunk_indices,
        o, final_state, workspace, tiling, tilingData);
#else
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPipeline<DTYPE_K, Catlass::Gemm::Kernel::GDNFwdHTileShapes128>(
            k, w, u, g, gk, initial_state, q, cu_seqlens, chunk_indices,
            o, final_state, workspace, tiling, tilingData);
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPipeline<DTYPE_K, Catlass::Gemm::Kernel::GDNFwdHTileShapes256>(
            k, w, u, g, gk, initial_state, q, cu_seqlens, chunk_indices,
            o, final_state, workspace, tiling, tilingData);
    }
#endif
}
