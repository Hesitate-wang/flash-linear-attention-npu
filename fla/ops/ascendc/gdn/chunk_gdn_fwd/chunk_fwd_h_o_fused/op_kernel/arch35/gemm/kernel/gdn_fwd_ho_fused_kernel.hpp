/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * A5 fused FwdH + FwdO kernel for ChunkFwdHOFused executionMode = 2
 * (SINGLE_CORE_CHUNK_FUSED): one MIX core owns one logical task, walks its
 * chunks strictly serially and alternates the FwdH/FwdO stages inside a chunk.
 *
 *   AIC : [chunk gate] OC1 -> HC1 -> OC2 -> HC2 -> [wait AIV1] OC3
 *   AIV : [init vec2] OV1(wait OC1) -> HV1(wait HC1) -> HV2(wait HC2)
 *         -> [wait OC2, OC3] OV2 -> [AIVCHUNK]
 *
 * Every chunk runs the single-slot FwdH preload stages in this kernel:
 * both Cube stages Fixpipe straight into the
 * H_DIRECT UB window and the matching preload epilogues read it in place.
 * MMAD and Vector receive the current block's actual token count.
 *
 * 0x2 flag ids (single slot, see the single-slot UB layout header):
 *   id 0 oc1QkReady        AIC -> AIV  Q @ K^T workspace ready
 *   id 1 hc1WhReady        AIC -> AIV  W @ h (H_DIRECT) ready
 *   id 2 oc2QhReady        AIC -> AIV  Q @ h FP32 workspace ready
 *   id 3 oc3AttentionVReady        AIC -> AIV  maskedQK @ V_new FP32 workspace ready
 *   id 4 hc2KvUpdateReady        AIC -> AIV  K^T @ VUpdate (H_DIRECT) ready
 *   id 5 ov1MaskAndHv1VNewReadyForOc3      AIV -> AIC  V_new / masked QK ready, gates OC3
 *   id 6 ov2ChunkFinished   AIV -> AIC  chunk retired (driver gate)
 *   id 7 hv1VUpdateReadyForHc2      AIV -> AIC  VUpdate ready, consumed inside the KV MMAD
 *   id 8 hv2HReadyForHc1       AIV -> AIC  H_DIRECT readable, consumed inside the WH MMAD
 *
 * The AIV seeds ov2ChunkFinished and hv2HReadyForHc1 after its initial h[0]
 * write. It republishes hv2HReadyForHc1 immediately after HV2, then
 * ov2ChunkFinished after OV2 retires the chunk.
 *
 * FwdH Cube results use the shared H_DIRECT Fixpipe window; the AIV consumes
 * each result before the next H Cube stage writes it. The 64x64 causal mask
 * is rebuilt by the AIV before every OV1.
 *
 * v1 scope (enforced by host tiling): A5, natural exp, fixed length, K = V =
 * 128 and no q/k gating.
 */

#ifndef CATLASS_GEMM_KERNEL_GDN_FWD_HO_FUSED_HPP
#define CATLASS_GEMM_KERNEL_GDN_FWD_HO_FUSED_HPP

#if !defined(CHUNK_FWD_HO_ARCH35) || defined(CHUNK_FWD_HO_ARCH_A2)
#error "A5 fused FwdH/FwdO kernel compiled for the wrong architecture"
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
#include "../../epilogue/block/block_epilogue_gdn_fwdo_qkmask.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdo_output.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_preload.hpp"
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "kernel_operator.h"
#include "../../../chunk_fwd_h_o_fused_struct.h"
#include "../../../chunk_fwd_o_struct.h"
#include "../../chunk_fwd_h_o_fused_ub_layout.h"
#include "../../chunk_fwd_ho_fused_a5_single_slot_ub_layout.h"
#include "../../chunk_fwd_ho_fused_a5_single_slot_l1_layout.h"
#include "../block/block_scheduler_gdn_fwd_ho_fused.hpp"

using namespace Catlass;
using namespace tla;

namespace Catlass::Gemm::Kernel {

template <
    typename INPUT_TYPE,
    typename G_TYPE,
    typename STATE_TYPE = float,
    typename WORKSPACE_TYPE = float,
    bool kGated = false
>
class GDNFwdHOFusedKernel {
public:
    using ArchTag = Arch::Ascend950;
    using Scheduler = Catlass::Gemm::Block::BlockSchedulerGdnFwdHOFused;
    using HOffsets = Catlass::Gemm::Block::GDNFwdHOffsets;
    using OOffsets = Catlass::Gemm::Block::GDNFwdOOffsets;


    using VType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using GType = Gemm::GemmType<G_TYPE, layout::RowMajor>;

    // ------------------------------------------------------------------
    // FwdO building blocks
    // ------------------------------------------------------------------
    using AttenType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using AttenMaskedType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using OinterType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using OType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using MaskType = Gemm::GemmType<bool, layout::RowMajor>;

    using DispatchPolicyO = Gemm::MmadPingpongTlaPreload<ArchTag, true, false>;
    using L1TileShapeO = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<128>>;
    using L0TileShapeO = L1TileShapeO;

    using TileCopyQK = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::ColumnMajor,
        WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<
        DispatchPolicyO, L1TileShapeO, L0TileShapeO, INPUT_TYPE, INPUT_TYPE,
        WORKSPACE_TYPE, void, TileCopyQK>;

    // OC2 keeps its own L1 base; OC3 reuses part of that window after OC2.
    using TileCopyQH = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor,
        WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadQH = Gemm::Block::BlockMmadTla<
        DispatchPolicyO, L1TileShapeO, L0TileShapeO, INPUT_TYPE, INPUT_TYPE,
        WORKSPACE_TYPE, void, TileCopyQH>;

    using TileCopyAttenVNEW = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor,
        WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadAttenVNEW = Gemm::Block::BlockMmadTla<
        DispatchPolicyO, L1TileShapeO, L0TileShapeO, INPUT_TYPE, INPUT_TYPE,
        WORKSPACE_TYPE, void, TileCopyAttenVNEW>;

    static_assert(GDN::CHUNK_FWD_HO_FUSED_L1_O_CUBE2_BASE +
                      BlockMmadQH::L1A_TILE_SIZE * BlockMmadQH::L1A_STAGES +
                      BlockMmadQH::L1B_TILE_SIZE * BlockMmadQH::L1B_STAGES <= ArchTag::L1_SIZE,
                  "OC2 L1 allocation exceeds the architecture capacity");

    using DispatchPolicyOQkmask = Epilogue::EpilogueAtlasGDNFwdOQkmask;
    using EpilogueOQkmask = Epilogue::Block::BlockEpilogue<
        DispatchPolicyOQkmask, AttenMaskedType, GType, AttenType, MaskType>;
    using DispatchPolicyOOutput = Epilogue::EpilogueAtlasGDNFwdOOutput;
    using EpilogueOOutput = Epilogue::Block::BlockEpilogue<
        DispatchPolicyOOutput, OType, GType, OinterType, OinterType>;

    using ElementK = INPUT_TYPE;
    using ElementW = INPUT_TYPE;
    using ElementU = INPUT_TYPE;
    using ElementG = G_TYPE;
    using ElementH = INPUT_TYPE;
    using ElementV = INPUT_TYPE;
    using ElementVUpdate = INPUT_TYPE;
    using ElementVWork = WORKSPACE_TYPE;
    using ElementHWork = WORKSPACE_TYPE;
    using ElementInitialState = STATE_TYPE;
    using ElementFinalState = STATE_TYPE;
    using ElementQ = INPUT_TYPE;
    using ElementAtten = WORKSPACE_TYPE;
    using ElementAttenMasked = INPUT_TYPE;
    using ElementOinter = WORKSPACE_TYPE;
    using ElementVNEW = INPUT_TYPE;
    using ElementO = INPUT_TYPE;
    using ElementMask = bool;

    using LayoutH = Catlass::layout::RowMajor;
    using LayoutW = Catlass::layout::RowMajor;
    using LayoutV = Catlass::layout::RowMajor;
    using LayoutVNEW = typename VType::Layout;
    using LayoutK = Catlass::layout::ColumnMajor;
    using LayoutQ = Catlass::layout::RowMajor;
    using LayoutAtten = Catlass::layout::RowMajor;
    using LayoutOinter = Catlass::layout::RowMajor;

    using BlockMmadWH = Gemm::Block::BlockMmadTla<
        DispatchPolicyO, L1TileShapeO, L0TileShapeO, INPUT_TYPE, INPUT_TYPE,
        WORKSPACE_TYPE, void,
        Catlass::Gemm::Tile::PackedTileCopyTlaToUB<
            ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor,
            WORKSPACE_TYPE, layout::RowMajor, void,
            Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>>;
    using BlockMmadKV = Gemm::Block::BlockMmadTla<
        DispatchPolicyO, L1TileShapeO, L0TileShapeO, INPUT_TYPE, INPUT_TYPE,
        WORKSPACE_TYPE, void,
        Catlass::Gemm::Tile::PackedTileCopyTlaToUB<
            ArchTag, INPUT_TYPE, layout::ColumnMajor, INPUT_TYPE, layout::RowMajor,
            WORKSPACE_TYPE, layout::RowMajor, void,
            Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>>;
    using VworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using HworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using FinalStateType = Gemm::GemmType<STATE_TYPE, layout::RowMajor>;
    using EpilogueVnew = Epilogue::Block::BlockEpilogue<
        Epilogue::EpilogueAtlasGDNFwdHVnewPreloadSingleSlot,
        VType, GType, VType, VworkType, FinalStateType>;
    using EpilogueUpdate = Epilogue::Block::BlockEpilogue<
        Epilogue::EpilogueAtlasGDNFwdHUpdatePreloadSingleSlot,
        VType, GType, VType, HworkType, FinalStateType>;

    static constexpr uint32_t kMaskRows = 64;
    static constexpr uint32_t kMaskRowElems = 64;

    // ------------------------------------------------------------------
    // Tiling-derived state
    // ------------------------------------------------------------------
    uint32_t shapeBatch{0};
    uint32_t seqlen{0};
    uint32_t kNumHead{0};
    uint32_t vNumHead{0};
    uint32_t kHeadDim{0};
    uint32_t vHeadDim{0};
    uint32_t chunkSize{0};
    uint32_t numChunks{0};
    float scale{1.0f};
    bool useInitialState{false};
    bool storeFinalState{false};
    uint32_t taskNum{0};
    int64_t vWorkspaceOffset{0};
    int64_t vUpdateWorkspaceOffset{0};
    int64_t hWorkspaceOffset{0};
    int64_t oVWorkspaceOffset{0};
    int64_t oHWorkspaceOffset{0};
    int64_t oAttnWorkspaceOffset{0};
    int64_t oAfterMaskWorkspaceOffset{0};
    int64_t oMaskWorkspaceOffset{0};

    AscendC::GlobalTensor<ElementK> gmK;
    AscendC::GlobalTensor<ElementW> gmW;
    AscendC::GlobalTensor<ElementU> gmU;
    AscendC::GlobalTensor<ElementG> gmG;
    AscendC::GlobalTensor<ElementInitialState> gmInitialState;
    AscendC::GlobalTensor<ElementFinalState> gmFinalState;
    AscendC::GlobalTensor<ElementQ> gmQ;
    AscendC::GlobalTensor<ElementV> gmVNew;
    AscendC::GlobalTensor<ElementH> gmH;
    AscendC::GlobalTensor<ElementO> gmO;
    AscendC::GlobalTensor<ElementMask> gmMask;
    AscendC::GlobalTensor<ElementVWork> gmVWorkspace;
    AscendC::GlobalTensor<ElementV> gmVUpdateWorkspace;
    AscendC::GlobalTensor<ElementHWork> gmHWorkspace;
    AscendC::GlobalTensor<ElementAtten> gmAttnWorkspace;
    AscendC::GlobalTensor<ElementAttenMasked> gmAftermaskWorkspace;
    AscendC::GlobalTensor<ElementOinter> gmOVWorkspace;
    AscendC::GlobalTensor<ElementOinter> gmOHWorkspace;

    Scheduler scheduler;
    Arch::Resource<ArchTag> resource;

    // Single-slot 0x2 flag ids (see the single-slot UB layout header).
    Arch::CrossCoreFlag oc1QkReady{GDN::CHUNK_FWD_HO_FUSED_FLAG_OC1_QK_READY};
    Arch::CrossCoreFlag oc2QhReady{GDN::CHUNK_FWD_HO_FUSED_FLAG_OC2_QH_READY};
    Arch::CrossCoreFlag oc3AttentionVReady{GDN::CHUNK_FWD_HO_FUSED_FLAG_OC3_ATTENTION_V_READY};
    Arch::CrossCoreFlag ov2ChunkFinished{GDN::CHUNK_FWD_HO_FUSED_FLAG_OV2_CHUNK_FINISHED};

    Arch::CrossCoreFlag hc1WhReady{GDN::CHUNK_FWD_HO_FUSED_FLAG_HC1_WH_READY};
    Arch::CrossCoreFlag hc2KvUpdateReady{GDN::CHUNK_FWD_HO_FUSED_FLAG_HC2_KV_UPDATE_READY};
    Arch::CrossCoreFlag hv1VUpdateReadyForHc2{GDN::CHUNK_FWD_HO_FUSED_FLAG_HV1_VUPDATE_READY_FOR_HC2};
    Arch::CrossCoreFlag ov1MaskAndHv1VNewReadyForOc3{
        GDN::CHUNK_FWD_HO_FUSED_FLAG_OV1_MASK_AND_HV1_VNEW_READY_FOR_OC3};
    Arch::CrossCoreFlag hv2HReadyForHc1{GDN::CHUNK_FWD_HO_FUSED_FLAG_HV2_H_READY_FOR_HC1};

    BlockMmadWH blockMmadWH;
    BlockMmadKV blockMmadKV;
    EpilogueVnew epilogueVnew;
    EpilogueUpdate epilogueUpdate;
    AscendC::LocalTensor<ElementVWork> ubVWork;
    AscendC::LocalTensor<ElementHWork> ubHUpdate;

    __aicore__ inline GDNFwdHOFusedKernel()
        : blockMmadWH(resource), blockMmadKV(resource),
          epilogueVnew(resource), epilogueUpdate(resource) {}

    __aicore__ inline void Init(
        GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR initialState,
        GM_ADDR q, GM_ADDR hHandoff, GM_ADDR vNewHandoff, GM_ADDR finalState,
        GM_ADDR o, GM_ADDR tiling, GM_ADDR user)
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
        numChunks = static_cast<uint32_t>(data->numChunksPerBatch);
        scale = data->scale;
        useInitialState = data->useInitialState;
        storeFinalState = data->storeFinalState;
        taskNum = static_cast<uint32_t>(data->singleCoreTaskNum);
        vWorkspaceOffset = data->vWorkspaceOffset;
        vUpdateWorkspaceOffset = data->vUpdateWorkspaceOffset;
        hWorkspaceOffset = data->hWorkspaceOffset;
        oVWorkspaceOffset = data->oVWorkspaceOffset;
        oHWorkspaceOffset = data->oHWorkspaceOffset;
        oAttnWorkspaceOffset = data->oAttnWorkspaceOffset;
        oAfterMaskWorkspaceOffset = data->oAfterMaskWorkspaceOffset;
        oMaskWorkspaceOffset = data->oMaskWorkspaceOffset;

        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmW.SetGlobalBuffer((__gm__ ElementW *)w);
        gmU.SetGlobalBuffer((__gm__ ElementU *)u);
        gmG.SetGlobalBuffer((__gm__ ElementG *)g);
        gmInitialState.SetGlobalBuffer((__gm__ ElementInitialState *)initialState);
        gmFinalState.SetGlobalBuffer((__gm__ ElementFinalState *)finalState);
        gmQ.SetGlobalBuffer((__gm__ ElementQ *)q);
        gmH.SetGlobalBuffer((__gm__ ElementH *)hHandoff);
        gmVNew.SetGlobalBuffer((__gm__ ElementV *)vNewHandoff);
        gmO.SetGlobalBuffer((__gm__ ElementO *)o);
        gmMask.SetGlobalBuffer((__gm__ ElementMask *)(user + oMaskWorkspaceOffset));
        gmVWorkspace.SetGlobalBuffer((__gm__ ElementVWork *)(user + vWorkspaceOffset));
        gmVUpdateWorkspace.SetGlobalBuffer((__gm__ ElementV *)(user + vUpdateWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer((__gm__ ElementHWork *)(user + hWorkspaceOffset));
        gmAttnWorkspace.SetGlobalBuffer((__gm__ ElementAtten *)(user + oAttnWorkspaceOffset));
        gmAftermaskWorkspace.SetGlobalBuffer(
            (__gm__ ElementAttenMasked *)(user + oAfterMaskWorkspaceOffset));
        gmOVWorkspace.SetGlobalBuffer((__gm__ ElementOinter *)(user + oVWorkspaceOffset));
        gmOHWorkspace.SetGlobalBuffer((__gm__ ElementOinter *)(user + oHWorkspaceOffset));

        ubVWork = resource.ubBuf.template GetBufferByByte<ElementVWork>(
            GDN::CHUNK_FWD_HO_FUSED_UB_VAR_H_DIRECT_OFFSET);
        ubHUpdate = resource.ubBuf.template GetBufferByByte<ElementHWork>(
            GDN::CHUNK_FWD_HO_FUSED_UB_VAR_H_DIRECT_OFFSET);

        uint32_t mixedCoreIdx = 0;
        if ASCEND_IS_AIV {
            mixedCoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        } else {
            mixedCoreIdx = AscendC::GetBlockIdx();
        }
        scheduler.Init(tiling, mixedCoreIdx);
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            ProcessAic();
        }
        if ASCEND_IS_AIV {
            ProcessAiv();
        }
    }

private:
    static constexpr uint32_t kSlotCount =
        static_cast<uint32_t>(GDN::CHUNK_FWD_HO_FUSED_SLOT_COUNT);

    __aicore__ inline bool NeedStage2(bool isFinalState) const
    {
        return storeFinalState || !isFinalState;
    }

    __aicore__ inline void StageC1(const HOffsets &off)
    {
        auto wLayout = tla::MakeLayout<ElementW, LayoutW>(
            shapeBatch * kNumHead * seqlen, kHeadDim);
        auto hLayout = tla::MakeLayout<ElementH, LayoutH>(
            shapeBatch * vNumHead * numChunks * kHeadDim, vHeadDim);
        auto vLayout = tla::MakeLayout<ElementVWork, LayoutV>(off.blockTokens, vHeadDim);
        auto tensorW = tla::MakeTensor(gmW[off.wOffset], wLayout, Catlass::Arch::PositionGM{});
        auto tensorH = tla::MakeTensor(gmH[off.hSrcOffset], hLayout, Catlass::Arch::PositionGM{});
        auto tensorV = tla::MakeTensor(ubVWork, vLayout, Catlass::Arch::PositionUB{});
        GemmCoord shape{off.blockTokens, vHeadDim, kHeadDim};
        auto tileW = GetTile(tensorW, tla::MakeCoord(0, 0),
                             tla::MakeShape(shape.m(), shape.k()));
        auto tileH = GetTile(tensorH, tla::MakeCoord(0, 0),
                             tla::MakeShape(shape.k(), shape.n()));
        // blockMmadWH.preSetFlags();
        blockMmadWH(tileW, tileH, tensorV, shape, hv2HReadyForHc1, 0);
        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(hc1WhReady);
    }

    __aicore__ inline void StageC2(const HOffsets &off, bool needStage2)
    {
        if (needStage2) {
            auto kLayout = tla::MakeLayout<ElementK, LayoutK>(
                kHeadDim, shapeBatch * kNumHead * seqlen);
            auto vworkLayout = tla::MakeLayout<ElementV, LayoutV>(off.blockTokens, vHeadDim);
            auto hworkLayout = tla::MakeLayout<ElementHWork, LayoutH>(kHeadDim, vHeadDim);
            auto tensorK = tla::MakeTensor(gmK[off.wkOffset], kLayout, Catlass::Arch::PositionGM{});
            auto tensorVwork = tla::MakeTensor(
                gmVUpdateWorkspace[off.vWorkOffset], vworkLayout, Catlass::Arch::PositionGM{});
            auto tensorHwork = tla::MakeTensor(ubHUpdate, hworkLayout, Catlass::Arch::PositionUB{});
            GemmCoord shape{kHeadDim, vHeadDim, off.blockTokens};
            auto tileK = GetTile(tensorK, tla::MakeCoord(0, 0),
                                 tla::MakeShape(shape.m(), shape.k()));
            auto tileVwork = GetTile(tensorVwork, tla::MakeCoord(0, 0),
                                     tla::MakeShape(shape.k(), shape.n()));
            // blockMmadKV.preSetFlags();
            blockMmadKV(tileK, tileVwork, tensorHwork, shape, hv1VUpdateReadyForHc2);
            // blockMmadKV.finalWaitFlags();
        } else {
            Arch::CrossCoreWaitFlag(hv1VUpdateReadyForHc2);
        }
        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(hc2KvUpdateReady);
    }

    __aicore__ inline void StageV1(const HOffsets &off)
    {
        epilogueVnew(
            gmVNew[off.uvOffset], gmVUpdateWorkspace[off.vWorkOffset],
            gmG[off.gOffset], gmU[off.uvOffset], gmVWorkspace[off.vWorkOffset],
            off.blockTokens, kHeadDim, vHeadDim,
            hc1WhReady, hv1VUpdateReadyForHc2,
            off.isInitialState, off.isFinalState, storeFinalState);
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(ov1MaskAndHv1VNewReadyForOc3);
    }

    __aicore__ inline void StageV2(const HOffsets &off, bool needStage2)
    {
        if (needStage2) {
            epilogueUpdate(
                gmH[off.hDstOffset], gmFinalState[off.finalStateOffset],
                gmInitialState[off.initialStateOffset], gmG[off.gOffset],
                gmH[off.hSrcOffset], gmHWorkspace[off.hWorkOffset],
                off.blockTokens, kHeadDim, vHeadDim, hc2KvUpdateReady,
                off.isInitialState, off.isFinalState, storeFinalState, useInitialState);
        } else {
            Arch::CrossCoreWaitFlag(hc2KvUpdateReady);
        }
    }

    // ------------------------------------------------------------------
    // AIC: per chunk OC1 -> HC1 -> OC2 -> HC2 -> OC3
    // ------------------------------------------------------------------
    __aicore__ inline void ProcessAic()
    {
        const uint32_t coreNum = AscendC::GetBlockNum();

        BlockMmadQK blockMmadQK(resource);
        BlockMmadQH blockMmadQH(resource, GDN::CHUNK_FWD_HO_FUSED_L1_O_CUBE2_BASE);
        BlockMmadAttenVNEW blockMmadAttenVNEW(
            resource, GDN::CHUNK_FWD_HO_FUSED_L1_O_CUBE3_BASE);
        const uint32_t totalTokens = seqlen;
        const uint32_t totalChunks = numChunks;
        auto qLayout = tla::MakeLayout<ElementQ, LayoutQ>(
            shapeBatch * kNumHead * totalTokens, kHeadDim);
        auto kLayout = tla::MakeLayout<ElementK, LayoutK>(
            kHeadDim, shapeBatch * kNumHead * totalTokens);
        auto hLayout = tla::MakeLayout<ElementH, LayoutH>(
            shapeBatch * vNumHead * totalChunks * kHeadDim, vHeadDim);
        auto vnewLayout = tla::MakeLayout<ElementVNEW, LayoutVNEW>(
            shapeBatch * vNumHead * totalTokens, vHeadDim);
        blockMmadQK.preSetFlags();
        while (scheduler.isRunning) {
            // Chunk gate: the previous chunk's AIV retired every single-slot
            // workspace and wrote h[i] back to GM.
            Arch::CrossCoreWaitFlag(ov2ChunkFinished);

            const HOffsets &h = scheduler.offsets.h;
            const OOffsets &o = scheduler.offsets.o;

            // ---- OC1: Q @ K^T -> attn workspace (GM) --------------------
            {
                auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(
                    coreNum * chunkSize * kSlotCount, o.blockTokens);
                auto tensorQ = tla::MakeTensor(gmQ[o.qkOffset], qLayout, Catlass::Arch::PositionGM{});
                auto tensorK = tla::MakeTensor(gmK[o.qkOffset], kLayout, Catlass::Arch::PositionGM{});
                auto tensorAttn = tla::MakeTensor(
                    gmAttnWorkspace[o.attnWorkOffset], attenLayout, Catlass::Arch::PositionGM{});
                GemmCoord shape{o.blockTokens, o.blockTokens, kHeadDim};
                auto tileQ = GetTile(tensorQ, tla::MakeCoord(0, 0),
                                     tla::MakeShape(shape.m(), shape.k()));
                auto tileK = GetTile(tensorK, tla::MakeCoord(0, 0),
                                     tla::MakeShape(shape.k(), shape.n()));
                auto tileAttn = GetTile(tensorAttn, tla::MakeCoord(0, 0),
                                        tla::MakeShape(shape.m(), shape.n()));
                // blockMmadQK.preSetFlags();
                blockMmadQK(tileQ, tileK, tileAttn, shape);
                // blockMmadQK.finalWaitFlags();
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(oc1QkReady);
            }

            // ---- HC1: W @ h[i] ------------------------------------------
            StageC1(h);
            // ---- OC2: Q @ h[i] -> FP32 oH workspace (GM) ----------------
            // HC1 already waited for this H before loading it.
            {
                auto tensorQ = tla::MakeTensor(gmQ[o.qkOffset], qLayout, Catlass::Arch::PositionGM{});
                auto tensorH = tla::MakeTensor(gmH[o.hOffset], hLayout, Catlass::Arch::PositionGM{});
                auto ointerLayout = tla::MakeLayout<ElementOinter, LayoutOinter>(
                    coreNum * chunkSize * kSlotCount, o.vBlockDim);
                auto tensorHWork = tla::MakeTensor(
                    gmOHWorkspace[o.hvWorkOffset], ointerLayout, Catlass::Arch::PositionGM{});
                GemmCoord shape{o.blockTokens, o.vBlockDim, kHeadDim};
                auto tileQ = GetTile(tensorQ, tla::MakeCoord(0, 0),
                                     tla::MakeShape(shape.m(), shape.k()));
                auto tileH = GetTile(tensorH, tla::MakeCoord(0, 0),
                                     tla::MakeShape(shape.k(), shape.n()));
                auto tileHWork = GetTile(tensorHWork, tla::MakeCoord(0, 0),
                                         tla::MakeShape(shape.m(), shape.n()));
                // blockMmadWH.finalWaitFlags();
                // blockMmadQH.preSetFlags();
                blockMmadQH(tileQ, tileH, tileHWork, shape);
                // blockMmadQH.finalWaitFlags();
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(oc2QhReady);
            }

            // ---- HC2: K^T @ VUpdate --------------------------------------
            // The final chunk skips stage 2 unless the final state is stored.
            StageC2(h, NeedStage2(h.isFinalState));

            // ---- OC3 waits for OV1 and V_new/HV1, then consumes both -----
            Arch::CrossCoreWaitFlag(ov1MaskAndHv1VNewReadyForOc3);
            {
                auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(
                    coreNum * chunkSize * kSlotCount, o.blockTokens);
                auto ointerLayout = tla::MakeLayout<ElementOinter, LayoutOinter>(
                    coreNum * chunkSize * kSlotCount, o.vBlockDim);
                auto tensorAttnMask = tla::MakeTensor(
                    gmAftermaskWorkspace[o.attnWorkOffset], attenLayout,
                    Catlass::Arch::PositionGM{});
                auto tensorV = tla::MakeTensor(
                    gmVNew[o.ovOffset], vnewLayout, Catlass::Arch::PositionGM{});
                auto tensorVWork = tla::MakeTensor(
                    gmOVWorkspace[o.hvWorkOffset], ointerLayout, Catlass::Arch::PositionGM{});
                GemmCoord shape{o.blockTokens, o.vBlockDim, o.blockTokens};
                auto tileAttnMask = GetTile(tensorAttnMask, tla::MakeCoord(0, 0),
                                            tla::MakeShape(shape.m(), shape.k()));
                auto tileV = GetTile(tensorV, tla::MakeCoord(0, 0),
                                     tla::MakeShape(shape.k(), shape.n()));
                auto tileVWork = GetTile(tensorVWork, tla::MakeCoord(0, 0),
                                         tla::MakeShape(shape.m(), shape.n()));
                // blockMmadAttenVNEW.preSetFlags();
                blockMmadAttenVNEW(tileAttnMask, tileV, tileVWork, shape);
                // blockMmadAttenVNEW.finalWaitFlags();
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(oc3AttentionVReady);
            }

            if (!scheduler.Next()) {
                break;
            }
        }
        blockMmadAttenVNEW.finalWaitFlags();
    }

    // ------------------------------------------------------------------
    // AIV: seed tokens, then per chunk OV1 -> HV1 -> HV2 -> OV2
    // ------------------------------------------------------------------
    template <uint32_t EventId>
    __aicore__ inline void FenceVectorUbReuse()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EventId);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EventId);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EventId);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EventId);
    }

    __aicore__ inline void FenceOToH()
    {
        FenceVectorUbReuse<GDN::CHUNK_FWD_HO_FUSED_EVENT_O_FENCE>();
    }

    __aicore__ inline void FenceHToO()
    {
        FenceVectorUbReuse<GDN::CHUNK_FWD_HO_FUSED_EVENT_H_FENCE>();
    }

    __aicore__ inline void ProcessAiv()
    {
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();

        EpilogueOQkmask epilogueQkmask(resource, true);
        EpilogueOOutput epilogueOutput(resource, true);

        AscendC::LocalTensor<float> maskUbTensor =
            resource.ubBuf.template GetBufferByByte<float>(
            GDN::CHUNK_FWD_HO_FUSED_UB_VAR_MASK_OFFSET);

        WriteInitialHandoffState(subBlockIdx, subBlockNum);
        // Only these two H tokens persist across stage calls. O owns ids 0-3.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_OUTPUT);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_INPUT);
        // Seed chunk-0 dependencies after the initial H writeback.
        // H-ready tokens recur after HV2; chunk-finished recurs after OV2.
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(ov2ChunkFinished);
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(hv2HReadyForHc1);

        uint32_t pingpongFlag = 0;  // Each O call starts at ping; OV2 may flip internally.
        while (scheduler.isRunning) {
            const HOffsets &h = scheduler.offsets.h;
            const OOffsets &o = scheduler.offsets.o;

            // Retire O's previous UB accesses before rebuilding the shared mask.
            FenceOToH();
            RebuildCausalMask(maskUbTensor);

            // ---- OV1: gate difference / exp / causal mask, masked QK -----
            Arch::CrossCoreWaitFlag(oc1QkReady);
            epilogueQkmask(
                gmAftermaskWorkspace[o.attnWorkOffset],
                gmG[o.gOffset], gmAttnWorkspace[o.attnWorkOffset], gmMask,
                chunkSize, o.blockTokens, kHeadDim, vHeadDim, pingpongFlag,
                o.batchIdx, o.headIdx, o.chunkIdx, nullptr);

            // ---- HV1: VUpdate + V_new ------------------------------------
            FenceOToH();
            StageV1(h);

            // ---- HV2: h[i+1] = h[i] * gate + h work ---------------------
            // Stage 2 is skipped for the final chunk unless the final state is
            // stored; both sides take the same decision.
            const bool needStage2 = NeedStage2(h.isFinalState);
            StageV2(h, needStage2);
            // HC1 of the next chunk is the only consumer of this token.
            if (!scheduler.IsFinalChunk()) {
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(hv2HReadyForHc1);
            }

            // ---- OV2: (exp(g) * oH + oV) * scale -> O --------------------
            // OV2 pong work reuses H_DIRECT; also fence HV1 if HV2 was skipped.
            FenceHToO();
            epilogueOutput.ProcessNarrowSplit(
                gmO[o.ovOffset], gmG[o.gOffset],
                gmOVWorkspace[o.hvWorkOffset], gmOHWorkspace[o.hvWorkOffset],
                scale, o.blockTokens, o.vBlockDim, vHeadDim, pingpongFlag,
                oc2QhReady, oc3AttentionVReady, nullptr, true);
            pingpongFlag = 0;  // single slot

            // OV2 finished: allow the next chunk to reuse all single-slot buffers.
            FenceOToH();
            // The next AIC loop iteration consumes this token; do not leave a
            // final generation pending after the task's last chunk.
            if (!scheduler.IsFinalChunk()) {
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(ov2ChunkFinished);
            }

            if (!scheduler.Next()) {
                break;
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_OUTPUT);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_INPUT);
    }

    // h[0] of the owned task: the initial state, or zeros when it is absent.
    __aicore__ inline void WriteInitialHandoffState(uint32_t subBlockIdx, uint32_t subBlockNum)
    {
        const uint32_t taskIdx = scheduler.taskIdx;
        const uint32_t batchIdx = taskIdx / vNumHead;
        const uint32_t vHeadIdx = taskIdx % vNumHead;
        const uint32_t rowsPerSubBlock = (kHeadDim + subBlockNum - 1) / subBlockNum;
        const uint32_t rowBegin = subBlockIdx * rowsPerSubBlock;
        const uint32_t rowEnd = Min(rowBegin + rowsPerSubBlock, kHeadDim);
        if (rowBegin >= rowEnd) {
            return;
        }
        const uint32_t copyElems = (rowEnd - rowBegin) * vHeadDim;
        const uint64_t stateBlockSize = static_cast<uint64_t>(kHeadDim) * vHeadDim;
        const uint64_t initialStateOffset = static_cast<uint64_t>(taskIdx) * stateBlockSize +
                                            static_cast<uint64_t>(rowBegin) * vHeadDim;
        const uint64_t hOffset =
            (static_cast<uint64_t>(batchIdx) * vNumHead * numChunks +
             static_cast<uint64_t>(vHeadIdx) * numChunks) * stateBlockSize +
            static_cast<uint64_t>(rowBegin) * vHeadDim;

        AscendC::LocalTensor<ElementInitialState> stateUb =
            resource.ubBuf.template GetBufferByByte<ElementInitialState>(
                GDN::CHUNK_FWD_HO_FUSED_UB_VAR_INIT_STATE_OFFSET);
        AscendC::LocalTensor<ElementH> hUb =
            resource.ubBuf.template GetBufferByByte<ElementH>(
                GDN::CHUNK_FWD_HO_FUSED_UB_VAR_INIT_H_OFFSET);

        if (useInitialState) {
            if constexpr (!std::is_same<ElementInitialState, ElementH>::value) {
                AscendC::DataCopy(stateUb, gmInitialState[initialStateOffset], copyElems);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
                AscendC::Cast(hUb, stateUb, AscendC::RoundMode::CAST_RINT, copyElems);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
                AscendC::DataCopy(gmH[hOffset], hUb, copyElems);
            } else {
                AscendC::DataCopy(stateUb, gmInitialState[initialStateOffset], copyElems);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
                AscendC::DataCopy(gmH[hOffset], stateUb, copyElems);
            }
        } else {
            AscendC::Duplicate(hUb, static_cast<ElementH>(0), copyElems);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
            AscendC::DataCopy(gmH[hOffset], hUb, copyElems);
        }
        // Complete the handoff write before publishing ready or reusing staging.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
    }

    // 64 x 64 lower triangular causal mask: row r holds r + 1 ones.
    __aicore__ inline void RebuildCausalMask(AscendC::LocalTensor<float> maskUbTensor)
    {
        AscendC::Duplicate<float>(maskUbTensor, 0.0f, kMaskRows * kMaskRowElems);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < kMaskRows; ++row) {
            AscendC::Duplicate<float>(maskUbTensor[row * kMaskRowElems], 1.0f, row + 1);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_GDN_FWD_HO_FUSED_HPP
