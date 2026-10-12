/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * Single-slot FwdH preload stages for ChunkFwdHOFused executionMode = 2.
 *
 * Derived from gdn_fwd_h_kernel_preload.hpp, with the whole ping-pong machinery
 * removed: there is no Process(), no task iteration and no second stream.  The
 * fused single-core driver (gdn_fwd_ho_fused_kernel.hpp) owns the chunk loop and
 * calls these four stages per chunk, alternating them with the FwdO stages:
 *
 *   StageC1  W @ h[i]  -> Fixpipe straight into H_DIRECT (AIV UB)
 *   StageC2  K^T @ VUpdate -> Fixpipe straight into H_DIRECT (AIV UB)
 *   StageV1  VUpdate / V_new on the AIV, from H_DIRECT
 *   StageV2  h[i+1] = h[i] * gate + h work on the AIV
 *
 * Every buffer exists once (see chunk_fwd_ho_fused_a5_single_slot_ub_layout.h):
 * both Fixpipe targets use the H_DIRECT window and the two epilogues read them
 * in place.  Flags are single ids of the 0x2 domain:
 *   hc1Done / hc2Done  AIC -> AIV, Fixpipe result ready for the epilogue
 *   hVec1Done          AIV -> AIC, VUpdate ready; consumed inside the KV MMAD
 *   oAiv1Done          AIV -> AIC, V_new / masked QK ready; consumed by OC3
 *   preloadSlotFree    AIV -> AIC, H_DIRECT readable again; consumed inside the
 *                      WH MMAD slot handshake
 * The driver's own chunk gate (aivChunkDone) and the slotFree token are owned
 * by the driver: it publishes both in its prologue (after the h[0] write back)
 * and after every chunk's V2 stage.  These stages never publish them.
 *
 * A partial final chunk is not handled here: the driver routes
 * blockTokens < chunkSize to the split-mode bounded tail MMAD and the AIV
 * Vector fallback.
 */

#ifndef CATLASS_GEMM_KERNEL_GDN_FWD_H_PRELOAD_SS_HPP
#define CATLASS_GEMM_KERNEL_GDN_FWD_H_PRELOAD_SS_HPP

#if !defined(CHUNK_FWD_HO_ARCH35) || defined(CHUNK_FWD_HO_ARCH_A2)
#error "A5 single-slot FwdH preload stages compiled for the wrong architecture"
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/debug.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "../../epilogue/gdn_fwd_h_epilogue_policies_mode2.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdh_vnew_preload_ss.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdh_update_preload_ss.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_preload.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "kernel_operator.h"
#include "../../../chunk_fwd_h_o_fused_struct.h"
#include "../../chunk_fwd_ho_fused_a5_single_slot_ub_layout.h"

using namespace Catlass;
using namespace tla;

namespace Catlass::Gemm::Kernel {

template <
    typename INPUT_TYPE,
    typename G_TYPE,
    typename STATE_TYPE = float,
    typename WORKSPACE_TYPE = float
>
class GDNFwdHPreloadSingleSlot {
public:
    using ArchTag = Arch::Ascend950;
    using HOffsets = Catlass::Gemm::Block::GDNFwdHOffsets;

    using DispatchPolicyTlaMulti = Gemm::MmadPingpongTlaPreload<ArchTag, true, false>;
    using L1TileShapeTla = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<128>>;
    using L0TileShapeTla = L1TileShapeTla;

    using WType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using HType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using VworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using KType = Gemm::GemmType<INPUT_TYPE, layout::ColumnMajor>;
    using HworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using VType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using GType = Gemm::GemmType<G_TYPE, layout::RowMajor>;
    using UType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using FinalStateType = Gemm::GemmType<STATE_TYPE, layout::RowMajor>;

    using TileCopyWH = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor,
        WORKSPACE_TYPE, layout::RowMajor, void,
        Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadWH = Gemm::Block::BlockMmadTla<
        DispatchPolicyTlaMulti, L1TileShapeTla, L0TileShapeTla, INPUT_TYPE,
        INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyWH>;

    using TileCopyKV = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, INPUT_TYPE, layout::ColumnMajor, INPUT_TYPE, layout::RowMajor,
        WORKSPACE_TYPE, layout::RowMajor, void,
        Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadKV = Gemm::Block::BlockMmadTla<
        DispatchPolicyTlaMulti, L1TileShapeTla, L0TileShapeTla, INPUT_TYPE,
        INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyKV>;

    using DispatchPolicyVnew = Epilogue::EpilogueAtlasGDNFwdHVnewPreloadSingleSlot;
    using EpilogueVnew = Epilogue::Block::BlockEpilogue<
        DispatchPolicyVnew, VType, GType, UType, VworkType, FinalStateType>;
    using DispatchPolicyUpdate = Epilogue::EpilogueAtlasGDNFwdHUpdatePreloadSingleSlot;
    using EpilogueUpdate = Epilogue::Block::BlockEpilogue<
        DispatchPolicyUpdate, HType, GType, HType, HworkType, FinalStateType>;

    using ElementK = INPUT_TYPE;
    using ElementW = INPUT_TYPE;
    using ElementU = INPUT_TYPE;
    using ElementG = G_TYPE;
    using ElementH = INPUT_TYPE;
    using ElementV = INPUT_TYPE;
    using ElementVWork = WORKSPACE_TYPE;
    using ElementHWork = WORKSPACE_TYPE;
    using ElementInitialState = STATE_TYPE;
    using ElementFinalState = STATE_TYPE;

    using LayoutW = Catlass::layout::RowMajor;
    using LayoutH = Catlass::layout::RowMajor;
    using LayoutV = Catlass::layout::RowMajor;
    using LayoutK = Catlass::layout::ColumnMajor;

    // Single-slot H-side flags.
    Arch::CrossCoreFlag hc1Done{GDN::CHUNK_FWD_HO_FUSED_FLAG_HC1_DONE};
    Arch::CrossCoreFlag hc2Done{GDN::CHUNK_FWD_HO_FUSED_FLAG_HC2_DONE};
    Arch::CrossCoreFlag hVec1Done{GDN::CHUNK_FWD_HO_FUSED_FLAG_H_VEC1_DONE};
    Arch::CrossCoreFlag oAiv1Done{GDN::CHUNK_FWD_HO_FUSED_FLAG_O_AIV1_DONE};
    Arch::CrossCoreFlag slotFree{GDN::CHUNK_FWD_HO_FUSED_FLAG_PRELOAD_SLOT_FREE};

    uint32_t shapeBatch{0};
    uint32_t seqlen{0};
    uint32_t kNumHead{0};
    uint32_t vNumHead{0};
    uint32_t kHeadDim{0};
    uint32_t vHeadDim{0};
    uint32_t chunkSize{0};
    uint32_t coreNum{0};
    bool useInitialState{false};
    bool storeFinalState{false};

    AscendC::GlobalTensor<ElementK> gmK;
    AscendC::GlobalTensor<ElementW> gmW;
    AscendC::GlobalTensor<ElementU> gmU;
    AscendC::GlobalTensor<ElementG> gmG;
    AscendC::GlobalTensor<ElementInitialState> gmInitialState;
    AscendC::GlobalTensor<ElementH> gmH;
    AscendC::GlobalTensor<ElementV> gmV;
    AscendC::GlobalTensor<ElementFinalState> gmFinalState;
    AscendC::GlobalTensor<ElementVWork> gmVWorkspace;
    AscendC::GlobalTensor<ElementV> gmVUpdateWorkspace;
    AscendC::GlobalTensor<ElementHWork> gmHWorkspace;

    Arch::Resource<ArchTag> resource;
    BlockMmadWH blockMmadWH;
    BlockMmadKV blockMmadKV;
    EpilogueVnew epilogueVnew;
    EpilogueUpdate epilogueUpdate;

    AscendC::LocalTensor<ElementVWork> ubVWork;
    AscendC::LocalTensor<ElementHWork> ubHUpdate;

    __aicore__ inline GDNFwdHPreloadSingleSlot()
        : blockMmadWH(resource), blockMmadKV(resource),
          epilogueVnew(resource), epilogueUpdate(resource) {}

    __aicore__ inline void Init(
        GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR initialState,
        GM_ADDR h, GM_ADDR vNew,
        GM_ADDR finalState, GM_ADDR tiling, GM_ADDR user)
    {
        const __gm__ GDN::ChunkFwdHOFusedTilingData *data =
            reinterpret_cast<const __gm__ GDN::ChunkFwdHOFusedTilingData *>(tiling);

        shapeBatch = static_cast<uint32_t>(data->shapeBatch);
        seqlen = static_cast<uint32_t>(data->seqlen);
        kNumHead = static_cast<uint32_t>(data->kNumHead);
        vNumHead = static_cast<uint32_t>(data->vNumHead);
        kHeadDim = static_cast<uint32_t>(data->kHeadDim);
        vHeadDim = static_cast<uint32_t>(data->vHeadDim);
        chunkSize = static_cast<uint32_t>(data->chunkSize);
        coreNum = static_cast<uint32_t>(data->singleCoreTaskNum);
        useInitialState = data->useInitialState;
        storeFinalState = data->storeFinalState;

        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmW.SetGlobalBuffer((__gm__ ElementW *)w);
        gmU.SetGlobalBuffer((__gm__ ElementU *)u);
        gmG.SetGlobalBuffer((__gm__ ElementG *)g);
        gmInitialState.SetGlobalBuffer((__gm__ ElementInitialState *)initialState);
        gmH.SetGlobalBuffer((__gm__ ElementH *)h);
        gmV.SetGlobalBuffer((__gm__ ElementV *)vNew);
        gmFinalState.SetGlobalBuffer((__gm__ ElementFinalState *)finalState);
        gmVWorkspace.SetGlobalBuffer(
            (__gm__ ElementVWork *)(user + data->vWorkspaceOffset));
        gmVUpdateWorkspace.SetGlobalBuffer(
            (__gm__ ElementV *)(user + data->vUpdateWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer(
            (__gm__ ElementHWork *)(user + data->hWorkspaceOffset));

        // Single slot: both Fixpipe targets share the H_DIRECT window.
        ubVWork = resource.ubBuf.template GetBufferByByte<ElementVWork>(
            GDN::CHUNK_FWD_HO_FUSED_UB_H_DIRECT_OFFSET);
        ubHUpdate = resource.ubBuf.template GetBufferByByte<ElementHWork>(
            GDN::CHUNK_FWD_HO_FUSED_UB_H_DIRECT_OFFSET);
    }

    // Cube2 and V2 are skipped for the final chunk unless the caller asked for
    // the final state; both sides must take the same decision.
    __aicore__ inline bool NeedStage2(bool isFinalState) const
    {
        return storeFinalState || !isFinalState;
    }

    // ------------------------------------------------------------------
    // AIC stages
    // ------------------------------------------------------------------
    __aicore__ inline void StageC1(const HOffsets &off)
    {
        auto wLayout = tla::MakeLayout<ElementW, LayoutW>(
            shapeBatch * kNumHead * seqlen, kHeadDim);
        auto hLayout = tla::MakeLayout<ElementH, LayoutH>(
            shapeBatch * vNumHead * numChunksOfTask() * kHeadDim, vHeadDim);
        auto vLayout = tla::MakeLayout<ElementVWork, LayoutV>(off.blockTokens, vHeadDim);
        auto tensorW = tla::MakeTensor(gmW[off.wOffset], wLayout, Catlass::Arch::PositionGM{});
        auto tensorH = tla::MakeTensor(gmH[off.hSrcOffset], hLayout, Catlass::Arch::PositionGM{});
        auto tensorV = tla::MakeTensor(ubVWork, vLayout, Catlass::Arch::PositionUB{});
        GemmCoord shape{off.blockTokens, vHeadDim, kHeadDim};
        auto tileW = GetTile(tensorW, tla::MakeCoord(0, 0),
                             tla::MakeShape(shape.m(), shape.k()));
        auto tileH = GetTile(tensorH, tla::MakeCoord(0, 0),
                             tla::MakeShape(shape.k(), shape.n()));
        // No cross-core H/O rendezvous in this schedule: readyFlagId = 0.
        blockMmadWH(tileW, tileH, tensorV, shape, slotFree, 0);
        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(hc1Done);
    }

    __aicore__ inline void StageC2(const HOffsets &off, bool needStage2)
    {
        if (needStage2) {
            auto kLayout = tla::MakeLayout<ElementK, LayoutK>(
                kHeadDim, shapeBatch * kNumHead * seqlen);
            auto vworkLayout = tla::MakeLayout<ElementV, LayoutV>(coreNum * chunkSize, vHeadDim);
            auto hworkLayout = tla::MakeLayout<ElementHWork, LayoutH>(kHeadDim, vHeadDim);
            auto tensorK = tla::MakeTensor(gmK[off.wkOffset], kLayout, Catlass::Arch::PositionGM{});
            auto tensorVwork = tla::MakeTensor(
                gmVUpdateWorkspace[off.vWorkOffset], vworkLayout, Catlass::Arch::PositionGM{});
            auto tensorHwork = tla::MakeTensor(
                ubHUpdate, hworkLayout, Catlass::Arch::PositionUB{});
            GemmCoord shape{kHeadDim, vHeadDim, off.blockTokens};
            auto tileK = GetTile(tensorK, tla::MakeCoord(0, 0),
                                 tla::MakeShape(shape.m(), shape.k()));
            auto tileVwork = GetTile(tensorVwork, tla::MakeCoord(0, 0),
                                     tla::MakeShape(shape.k(), shape.n()));
            blockMmadKV(tileK, tileVwork, tensorHwork, shape, hVec1Done);
        } else {
            Arch::CrossCoreWaitFlag(hVec1Done);
        }
        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(hc2Done);
    }

    // ------------------------------------------------------------------
    // AIV stages
    // ------------------------------------------------------------------
    __aicore__ inline void StageV1(const HOffsets &off)
    {
        epilogueVnew(
            gmV[off.uvOffset], gmVUpdateWorkspace[off.vWorkOffset],
            gmG[off.gOffset], gmU[off.uvOffset], gmVWorkspace[off.vWorkOffset],
            off.blockTokens, kHeadDim, vHeadDim,
            hc1Done, hVec1Done,
            off.isInitialState, off.isFinalState, storeFinalState);
        // The FwdO side needs its own token for masked QK -> OC3.
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(oAiv1Done);
    }

    __aicore__ inline void StageV2(const HOffsets &off, bool needStage2)
    {
        if (needStage2) {
                epilogueUpdate(
                gmH[off.hDstOffset], gmFinalState[off.finalStateOffset],
                gmInitialState[off.initialStateOffset],
                gmG[off.gOffset], gmH[off.hSrcOffset], gmHWorkspace[off.hWorkOffset],
                off.blockTokens, kHeadDim, vHeadDim,
                hc2Done, off.isInitialState, off.isFinalState, useInitialState);
        } else {
            Arch::CrossCoreWaitFlag(hc2Done);
        }
    }

private:
    __aicore__ inline uint32_t numChunksOfTask() const
    {
        return (seqlen + chunkSize - 1) / chunkSize;
    }
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_GDN_FWD_H_PRELOAD_SS_HPP
