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
 * Full chunks run the single-slot FwdH preload stages
 * (gdn_fwd_h_kernel_preload_ss.hpp): both Cube stages Fixpipe straight into the
 * H_DIRECT UB window and the matching preload epilogues read it in place.  A
 * partial final chunk takes the split-mode path instead -- a bounded tail MMAD
 * to GM for 16 <= tail < chunkSize, and the AIV Vector fallback below for a
 * shorter tail -- because the preload family never handles a partial chunk.
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
 * FwdH Cube results go to GM workspaces (invariant I1 of the UB layout), so no
 * AIC-to-AIV Fixpipe target exists and the FwdO AIV stage running concurrently
 * on the same MIX core cannot be corrupted.  The 64x64 causal mask is rebuilt
 * by the AIV before every OV1 (invariant I2).
 *
 * v1 scope (enforced by host tiling): A5, natural exp, fixed length, K = V =
 * 128, no q/k gating and seqlen % chunkSize == 0, so no tail path is used.
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
#include "../../epilogue/block/block_epilogue_gdn_fwdh_vnew_gm_mode2.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdh_update_gm_mode2.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdh_vnew_preload_ss.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdh_update_preload_ss.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdo_qkmask.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdo_output.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_pipelined.hpp"
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
#include "gdn_fwd_h_kernel_preload_ss.hpp"

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


    // ------------------------------------------------------------------
    // FwdH building blocks (GM exchange, non-preload path)
    // ------------------------------------------------------------------
    using HType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using VworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using HworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using VType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using GType = Gemm::GemmType<G_TYPE, layout::RowMajor>;
    using UType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using FinalStateType = Gemm::GemmType<STATE_TYPE, layout::RowMajor>;
    using VUpdateType = Gemm::GemmType<INPUT_TYPE, layout::zN>;

    using L1TileShapeH = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<128>>;
    using L0TileShapeH = L1TileShapeH;

    using TileCopyWH = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor,
        WORKSPACE_TYPE, layout::RowMajor>;

    using TileCopyKV = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, INPUT_TYPE, layout::ColumnMajor, INPUT_TYPE, layout::zN,
        WORKSPACE_TYPE, layout::RowMajor>;


    // Tail chunks take the split-mode path instead of padding: a bounded MMAD
    // writes the working tile to GM for 16 <= tail < chunkSize, and a tail
    // shorter than one MMAD tile is computed by the AIV Vector fallback.  Both
    // variants cost exactly the tail's work at run time.
    using DispatchPolicyHTail = Gemm::MmadPingpongTlaMulti<ArchTag, true, false, 1>;
    using BlockMmadWHTail = Gemm::Block::BlockMmadTla<
        DispatchPolicyHTail, L1TileShapeH, L0TileShapeH, INPUT_TYPE, INPUT_TYPE,
        WORKSPACE_TYPE, void, TileCopyWH>;
    using BlockMmadKVTail = Gemm::Block::BlockMmadTla<
        DispatchPolicyHTail, L1TileShapeH, L0TileShapeH, INPUT_TYPE, INPUT_TYPE,
        WORKSPACE_TYPE, void, TileCopyKV>;

    using DispatchPolicyHVnew = Epilogue::EpilogueAtlasGDNFwdHVnewGmMode2;
    using DispatchPolicyHUpdate = Epilogue::EpilogueAtlasGDNFwdHUpdateGmMode2;

    // GM variants of the same epilogues for tail chunks, which never use the
    // DirectUb slot.  Both share the single-slot UB layout.
    struct GateTagGm {
        static constexpr bool value = false;
        static constexpr bool scalarGated = true;
        static constexpr bool useExp2 = false;
        static constexpr bool useDirectFp32Ub = false;
    };
    using EpilogueHVnewGm = Epilogue::Block::BlockEpilogue<
        DispatchPolicyHVnew, VType, GType, UType, VworkType, VUpdateType,
        FinalStateType, GateTagGm>;
    using EpilogueHUpdateGm = Epilogue::Block::BlockEpilogue<
        DispatchPolicyHUpdate, HType, GType, HType, HworkType, FinalStateType, GateTagGm>;

    // ------------------------------------------------------------------
    // FwdO building blocks
    // ------------------------------------------------------------------
    using AttenType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using AttenMaskedType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using OinterType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using OType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using MaskType = Gemm::GemmType<bool, layout::RowMajor>;

    using DispatchPolicyO = Gemm::MmadPingpongTlaGdnFwdO<ArchTag, true, false>;
    using L1TileShapeO = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<128>>;
    using L0TileShapeO = L1TileShapeO;

    using TileCopyQK = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::ColumnMajor,
        WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadQK = Gemm::Block::BlockMmadTlaPipelined<
        DispatchPolicyO, L1TileShapeO, L0TileShapeO, INPUT_TYPE, INPUT_TYPE,
        WORKSPACE_TYPE, void, TileCopyQK>;

    // OC2 owns a separate L1 window so Q/H prefetch can overlap HC1 MMAD.
    using TileCopyQH = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor,
        WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadQH = Gemm::Block::BlockMmadTlaPipelined<
        DispatchPolicyO, L1TileShapeO, L0TileShapeO, INPUT_TYPE, INPUT_TYPE,
        WORKSPACE_TYPE, void, TileCopyQH>;

    using TileCopyAttenVNEW = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor,
        WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadAttenVNEW = Gemm::Block::BlockMmadTlaPipelined<
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

    using LayoutW = Catlass::layout::RowMajor;
    using LayoutH = Catlass::layout::RowMajor;
    using LayoutV = Catlass::layout::RowMajor;
    using LayoutVNEW = typename VType::Layout;
    using LayoutK = Catlass::layout::ColumnMajor;
    using LayoutVUpdate = typename VUpdateType::Layout;
    using LayoutQ = Catlass::layout::RowMajor;
    using LayoutAtten = Catlass::layout::RowMajor;
    using LayoutOinter = Catlass::layout::RowMajor;

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

    // Single-slot 0x2 flag ids (see the single-slot UB layout header).  The
    // FwdH preload stages own hc1WhReady / hc2KvUpdateReady / hv1VUpdateReadyForHc2 / ov1MaskAndHv1VNewReadyForOc3 /
    // hv2HReadyForHc1; this driver owns the FwdO tokens and the chunk gate.
    Arch::CrossCoreFlag oc1QkReady{GDN::CHUNK_FWD_HO_FUSED_FLAG_OC1_QK_READY};
    Arch::CrossCoreFlag oc2QhReady{GDN::CHUNK_FWD_HO_FUSED_FLAG_OC2_QH_READY};
    Arch::CrossCoreFlag oc3AttentionVReady{GDN::CHUNK_FWD_HO_FUSED_FLAG_OC3_ATTENTION_V_READY};
    Arch::CrossCoreFlag ov2ChunkFinished{GDN::CHUNK_FWD_HO_FUSED_FLAG_OV2_CHUNK_FINISHED};

    // Full chunks run the single-slot FwdH preload stages; partial chunks use
    // the split-mode bounded tail MMAD plus the AIV Vector fallback.
    GDNFwdHPreloadSingleSlot<INPUT_TYPE, G_TYPE, STATE_TYPE, WORKSPACE_TYPE> hPreload;

    __aicore__ inline GDNFwdHOFusedKernel() {}

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

        uint32_t mixedCoreIdx = 0;
        if ASCEND_IS_AIV {
            mixedCoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        } else {
            mixedCoreIdx = AscendC::GetBlockIdx();
        }
        scheduler.Init(tiling, mixedCoreIdx);
        hPreload.Init(k, w, u, g, initialState, hHandoff, vNewHandoff,
                      finalState, tiling, user);
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
    // MMAD / Fixpipe granularity: a chunk tail shorter than this would need the
    // split-mode Vector fallback, which host tiling rejects for now.
    static constexpr uint32_t kMinCubeTokens = 16U;

    // ------------------------------------------------------------------
    // AIC: per chunk OC1 -> HC1 -> OC2 -> HC2 -> OC3
    // ------------------------------------------------------------------
    __aicore__ inline void ProcessAic()
    {
        const uint32_t coreNum = AscendC::GetBlockNum();

        BlockMmadQK blockMmadQK(resource);
        BlockMmadQH blockMmadQH(
            resource, GDN::CHUNK_FWD_HO_FUSED_L1_O_CUBE2_BASE,
            GDN::CHUNK_FWD_HO_FUSED_L1_O_CUBE2_L1A_EVENT,
            GDN::CHUNK_FWD_HO_FUSED_L1_O_CUBE2_L1B_EVENT);
        BlockMmadAttenVNEW blockMmadAttenVNEW(
            resource, GDN::CHUNK_FWD_HO_FUSED_L1_O_CUBE3_BASE,
            GDN::CHUNK_FWD_HO_FUSED_L1_O_CUBE3_L1A_EVENT,
            GDN::CHUNK_FWD_HO_FUSED_L1_O_CUBE3_L1B_EVENT);
        // Full chunks run the single-slot FwdH preload stages inside hPreload;
        // partial chunks (blockTokens < chunkSize) use the bounded tail MMAD.
        BlockMmadWHTail blockMmadWHTail(resource);
        BlockMmadKVTail blockMmadKVTail(resource);

        const uint32_t totalTokens = seqlen;
        const uint32_t totalChunks = numChunks;
        auto qLayout = tla::MakeLayout<ElementQ, LayoutQ>(
            shapeBatch * kNumHead * totalTokens, kHeadDim);
        auto kLayout = tla::MakeLayout<ElementK, LayoutK>(
            kHeadDim, shapeBatch * kNumHead * totalTokens);
        auto wLayout = tla::MakeLayout<ElementW, LayoutW>(
            shapeBatch * kNumHead * totalTokens, kHeadDim);
        auto hLayout = tla::MakeLayout<ElementH, LayoutH>(
            shapeBatch * vNumHead * totalChunks * kHeadDim, vHeadDim);
        auto vnewLayout = tla::MakeLayout<ElementVNEW, LayoutVNEW>(
            shapeBatch * vNumHead * totalTokens, vHeadDim);
        auto hworkLayout = tla::MakeLayout<ElementHWork, LayoutH>(kHeadDim, vHeadDim);

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
                blockMmadQK.preSetFlags();
                blockMmadQK(tileQ, tileK, tileAttn, shape);
                blockMmadQK.waitL1Drained();
                blockMmadQK.waitL0Drained();
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(oc1QkReady);
            }

            // ---- HC1: W @ h[i] ------------------------------------------
            // full chunk  : Fixpipe direct into H_DIRECT (AIV UB)
            // 16 <= tail  : bounded tail MMAD into the GM v workspace
            // tail < 16   : no Cube; the AIV Vector fallback computes it
            {
                const bool tailVectorPath = h.blockTokens < kMinCubeTokens;
                const bool tailMmadPath = !tailVectorPath && (h.blockTokens < chunkSize);
                if (tailVectorPath) {
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(hPreload.hc1WhReady);
                } else if (tailMmadPath) {
                    auto vLayout = tla::MakeLayout<ElementVWork, LayoutV>(h.blockTokens, h.vBlockDim);
                    auto tensorW = tla::MakeTensor(gmW[h.wOffset], wLayout, Catlass::Arch::PositionGM{});
                    auto tensorH = tla::MakeTensor(gmH[h.hSrcOffset], hLayout, Catlass::Arch::PositionGM{});
                    auto tensorV = tla::MakeTensor(
                        gmVWorkspace[h.vWorkOffset], vLayout, Catlass::Arch::PositionGM{});
                    GemmCoord shape{h.blockTokens, h.vBlockDim, kHeadDim};
                    auto tileW = GetTile(tensorW, tla::MakeCoord(0, 0),
                                         tla::MakeShape(shape.m(), shape.k()));
                    auto tileH = GetTile(tensorH, tla::MakeCoord(0, 0),
                                         tla::MakeShape(shape.k(), shape.n()));
                    auto tileV = GetTile(tensorV, tla::MakeCoord(0, 0),
                                         tla::MakeShape(shape.m(), shape.n()));
                    blockMmadWHTail.preSetFlags();
                    blockMmadWHTail(tileW, tileH, tileV, shape, EmptyClass{}, true);
                    blockMmadWHTail.finalWaitFlags();
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(hPreload.hc1WhReady);
                } else {
                    hPreload.StageC1(h);
                }
            }

            // ---- OC2: Q @ h[i] -> FP32 oH workspace (GM) ----------------
            // HC1 already waited for this H before loading it. Tail HC1
            // paths are protected by the chunk gate. No second H-ready wait.
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
                // The OC2 operands use a disjoint L1 window, so launch their
                // GM->L1 copy immediately after HC1 is issued.  Only the L0
                // handoff waits for HC1 below.
                blockMmadQH.preSetL1Flags();
                blockMmadQH.copyGmToL1(tileQ, tileH, shape);
                if (h.blockTokens == chunkSize) {
                    hPreload.WaitC1Compute();
                }
                blockMmadQH.preSetL0Flags();
                blockMmadQH.executeCompute(tileHWork, shape);
                blockMmadQH.finalWaitFlags();
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(oc2QhReady);
            }

            // ---- HC2: K^T @ VUpdate --------------------------------------
            // full chunk : single-slot preload stage (Fixpipe into H_DIRECT)
            // 16 <= tail : bounded tail MMAD into the GM h workspace
            // tail < 16  : no Cube; the AIV Vector fallback computes it
            // The final chunk skips stage 2 unless the final state is stored.
            {
                const bool tailVectorPath = h.blockTokens < kMinCubeTokens;
                const bool tailMmadPath = !tailVectorPath && (h.blockTokens < chunkSize);
                const bool needStage2 = hPreload.NeedStage2(h.isFinalState);
                if (!tailVectorPath && !tailMmadPath) {
                    hPreload.StageC2(h, needStage2);
                } else {
                    Arch::CrossCoreWaitFlag(hPreload.hv1VUpdateReadyForHc2);
                    if (needStage2 && tailMmadPath) {
                        auto tensorK = tla::MakeTensor(gmK[h.wkOffset], kLayout, Catlass::Arch::PositionGM{});
                        auto vUpdateLayout = tla::MakeLayout<ElementVUpdate, LayoutVUpdate>(
                            h.blockTokens, h.vBlockDim);
                        auto tensorVUpdate = tla::MakeTensor(
                            gmVUpdateWorkspace[h.vWorkOffset], vUpdateLayout,
                            Catlass::Arch::PositionGM{});
                        auto tensorHWork = tla::MakeTensor(
                            gmHWorkspace[h.hWorkOffset], hworkLayout, Catlass::Arch::PositionGM{});
                        GemmCoord shape{kHeadDim, h.vBlockDim, h.blockTokens};
                        auto tileK = GetTile(tensorK, tla::MakeCoord(0, 0),
                                             tla::MakeShape(shape.m(), shape.k()));
                        auto tileVUpdate = GetTile(tensorVUpdate, tla::MakeCoord(0, 0),
                                                   tla::MakeShape(shape.k(), shape.n()));
                        auto tileHWork = GetTile(tensorHWork, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(shape.m(), shape.n()));
                        blockMmadKVTail.preSetFlags();
                        blockMmadKVTail(tileK, tileVUpdate, tileHWork, shape, EmptyClass{}, true);
                        blockMmadKVTail.finalWaitFlags();
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(hPreload.hc2KvUpdateReady);
                }
            }

            // ---- OC3 waits for OV1 and V_new/HV1, then consumes both -----
            Arch::CrossCoreWaitFlag(hPreload.ov1MaskAndHv1VNewReadyForOc3);
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
                blockMmadAttenVNEW.preSetL1Flags();
                blockMmadAttenVNEW.copyGmToL1BOnly(tileV, shape);
                blockMmadAttenVNEW.copyGmToL1AOnly(tileAttnMask, shape);
                blockMmadAttenVNEW.preSetL0Flags();
                blockMmadAttenVNEW.executeCompute(tileVWork, shape);
                blockMmadAttenVNEW.finalWaitFlags();
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(oc3AttentionVReady);
            }

            if (!scheduler.Next()) {
                break;
            }
        }
    }

    // ------------------------------------------------------------------
    // AIV: seed tokens, then per chunk OV1 -> HV1 -> HV2 -> OV2
    // ------------------------------------------------------------------
    __aicore__ inline void ProcessAiv()
    {
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();

        EpilogueHVnewGm epilogueVnewGm(resource);
        EpilogueHUpdateGm epilogueUpdateGm(resource);
        EpilogueOQkmask epilogueQkmask(resource);
        EpilogueOOutput epilogueOutput(resource);

        AscendC::LocalTensor<float> maskUbTensor =
            resource.ubBuf.template GetBufferByByte<float>(
                GDN::CHUNK_FWD_HO_FUSED_UB_O_MASK_OFFSET);

        // FwdH epilogue event presets.  Only the ping set exists: the fused
        // schedule keeps one slot, so the pong events are never produced.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);

        WriteInitialHandoffState(subBlockIdx, subBlockNum);
        // Seed chunk-0 dependencies after the initial H writeback.
        // H-ready tokens recur after HV2; chunk-finished recurs after OV2.
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(ov2ChunkFinished);
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(hPreload.hv2HReadyForHc1);

        uint32_t pingpongFlag = 0;  // forced: single slot always uses the ping half
        while (scheduler.isRunning) {
            const HOffsets &h = scheduler.offsets.h;
            const OOffsets &o = scheduler.offsets.o;

            // Invariant I2: the FwdH stages own UB[0,32K), so the causal mask
            // is rebuilt before every OV1 that reads it.
            RebuildCausalMask(maskUbTensor);

            // ---- OV1: gate difference / exp / causal mask, masked QK -----
            Arch::CrossCoreWaitFlag(oc1QkReady);
            epilogueQkmask(
                gmAftermaskWorkspace[o.attnWorkOffset],
                gmG[o.gOffset], gmAttnWorkspace[o.attnWorkOffset], gmMask,
                chunkSize, o.blockTokens, kHeadDim, vHeadDim, pingpongFlag,
                o.batchIdx, o.headIdx, o.chunkIdx, nullptr);

            // ---- HV1: VUpdate + V_new ------------------------------------
            // Full chunks run the single-slot preload stage, which publishes
            // hv1VUpdateReadyForHc2 (consumed inside the KV MMAD) and ov1MaskAndHv1VNewReadyForOc3 (consumed
            // by this driver before OC3).  Partial chunks run the GM epilogue
            // on the bounded tail MMAD, or on the Vector fallback when the tail
            // is shorter than one MMAD tile.
            const bool tailChunk = h.blockTokens < chunkSize;
            const bool tailVectorPath = h.blockTokens < kMinCubeTokens;
            if (!tailChunk) {
                hPreload.StageV1(h);
            } else {
                if (tailVectorPath) {
                    Arch::CrossCoreWaitFlag(hPreload.hc1WhReady);
                    ComputeTailVWorkspace(h, EVENT_ID3);
                }
                epilogueVnewGm(
                    gmVNew[h.uvOffset], gmVUpdateWorkspace[h.vWorkOffset],
                    AscendC::LocalTensor<ElementVUpdate>(),
                    gmG[h.gOffset], gmU[h.uvOffset], gmVWorkspace[h.vWorkOffset],
                    gmG[h.gOffset], gmK[h.wkOffset], gmK[h.wkOffset],
                    h.blockTokens, kHeadDim, h.vBlockDim, vHeadDim,
                    hPreload.hc1WhReady, hPreload.hv1VUpdateReadyForHc2,
                    h.isInitialState, h.isFinalState, storeFinalState,
                    false, true, tailVectorPath, 0ULL, 0ULL);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(hPreload.ov1MaskAndHv1VNewReadyForOc3);
            }

            // ---- HV2: h[i+1] = h[i] * gate + h work ---------------------
            // Stage 2 is skipped for the final chunk unless the final state is
            // stored; both sides take the same decision.
            const bool needStage2 = hPreload.NeedStage2(h.isFinalState);
            if (!tailChunk) {
                hPreload.StageV2(h, needStage2);
            } else if (needStage2) {
                if (tailVectorPath) {
                    Arch::CrossCoreWaitFlag(hPreload.hc2KvUpdateReady);
                    ComputeTailHWorkspace(h, EVENT_ID3);
                }
                epilogueUpdateGm(
                    gmH[h.hDstOffset], gmFinalState[h.finalStateOffset],
                    gmG[h.gOffset], gmH[h.hSrcOffset], gmHWorkspace[h.hWorkOffset],
                    gmG[h.gOffset], gmInitialState[h.initialStateOffset],
                    h.blockTokens, kHeadDim, h.vBlockDim, vHeadDim,
                    hPreload.hc2KvUpdateReady,
                    h.isInitialState, h.isFinalState, storeFinalState, useInitialState,
                    true, tailVectorPath, 0ULL, 0ULL);
            } else {
                Arch::CrossCoreWaitFlag(hPreload.hc2KvUpdateReady);
            }
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(hPreload.hv2HReadyForHc1);

            // ---- OV2: (exp(g) * oH + oV) * scale -> O --------------------
            epilogueOutput.ProcessNarrowSplit(
                gmO[o.ovOffset], gmG[o.gOffset],
                gmOVWorkspace[o.hvWorkOffset], gmOHWorkspace[o.hvWorkOffset],
                scale, o.blockTokens, o.vBlockDim, vHeadDim, pingpongFlag,
                oc2QhReady, oc3AttentionVReady, nullptr);
            pingpongFlag = 0;  // single slot

            // OV2 finished: allow the next chunk to reuse all single-slot buffers.
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(ov2ChunkFinished);

            if (!scheduler.Next()) {
                break;
            }
        }

        // Drain the FwdH epilogue events (ping set only).
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
    }

    // Tail Vector fallback, ported from the split-mode FwdH kernel.  A partial
    // chunk shorter than one MMAD tile has no Cube, so the AIV computes the
    // same working tile with Vector ops, writing it to the GM workspace that the
    // GM epilogue variant consumes.  The helpers borrow the V_MTE2 free token of
    // the ping slot and restore it before returning.
    __aicore__ inline void ComputeTailVWorkspace(
        const HOffsets &offsets, uint32_t tailEventId)
    {
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        const uint32_t rowsPerSubBlock = CeilDiv(offsets.blockTokens, subBlockNum);
        const uint32_t rowBegin = subBlockIdx * rowsPerSubBlock;
        const uint32_t rowEnd = Min(rowBegin + rowsPerSubBlock, offsets.blockTokens);
        if (rowBegin >= rowEnd) {
            return;
        }
        AscendC::ResetMask();
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(tailEventId);

        AscendC::LocalTensor<ElementH> inputUb =
            resource.ubBuf.template GetBufferByByte<ElementH>(
                GDN::CHUNK_FWD_HO_FUSED_UB_H_TAIL_INPUT_OFFSET);
        AscendC::LocalTensor<float> floatUb =
            resource.ubBuf.template GetBufferByByte<float>(
                GDN::CHUNK_FWD_HO_FUSED_UB_H_TAIL_FLOAT_OFFSET);
        AscendC::LocalTensor<float> accumUb =
            resource.ubBuf.template GetBufferByByte<float>(
                GDN::CHUNK_FWD_HO_FUSED_UB_H_TAIL_ACCUM_OFFSET);
        AscendC::LocalTensor<ElementW> weightInputUb =
            resource.ubBuf.template GetBufferByByte<ElementW>(
                GDN::CHUNK_FWD_HO_FUSED_UB_H_TAIL_WEIGHT_INPUT_OFFSET);
        AscendC::LocalTensor<float> weightFloatUb =
            resource.ubBuf.template GetBufferByByte<float>(
                GDN::CHUNK_FWD_HO_FUSED_UB_H_TAIL_WEIGHT_FLOAT_OFFSET);

        for (uint32_t tokenRow = rowBegin; tokenRow < rowEnd; ++tokenRow) {
            AscendC::DataCopy(
                weightInputUb,
                gmW[offsets.wOffset + tokenRow * kHeadDim],
                kHeadDim);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
            AscendC::Cast(
                weightFloatUb, weightInputUb, AscendC::RoundMode::CAST_NONE,
                kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(tailEventId);

            AscendC::Duplicate(accumUb, 0.0f, offsets.vBlockDim);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t kIdx = 0; kIdx < kHeadDim; ++kIdx) {
                AscendC::DataCopy(
                    inputUb, gmH[offsets.hSrcOffset + kIdx * vHeadDim],
                    offsets.vBlockDim);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                AscendC::Cast(
                    floatUb, inputUb, AscendC::RoundMode::CAST_NONE,
                    offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                const float weight = weightFloatUb.GetValue(kIdx);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(tailEventId);
                AscendC::Muls(floatUb, floatUb, weight, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(accumUb, accumUb, floatUb, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(tailEventId);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(tailEventId);
            AscendC::DataCopy(
                gmVWorkspace[offsets.vWorkOffset + tokenRow * offsets.vBlockDim],
                accumUb, offsets.vBlockDim);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
    }

    // H work of a sub-tile tail.  The fused schedule has no q/k gating yet, so
    // the weight is the user K tensor itself (the split-mode helper reads the
    // gated K decay workspace, which only exists for the gated path).
    __aicore__ inline void ComputeTailHWorkspace(
        const HOffsets &offsets, uint32_t tailEventId)
    {
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        const uint32_t rowsPerSubBlock = CeilDiv(kHeadDim, subBlockNum);
        const uint32_t rowBegin = subBlockIdx * rowsPerSubBlock;
        const uint32_t rowEnd = Min(rowBegin + rowsPerSubBlock, kHeadDim);
        AscendC::ResetMask();
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(tailEventId);

        AscendC::LocalTensor<ElementV> inputUb =
            resource.ubBuf.template GetBufferByByte<ElementV>(
                GDN::CHUNK_FWD_HO_FUSED_UB_H_TAIL_INPUT_OFFSET);
        AscendC::LocalTensor<float> floatUb =
            resource.ubBuf.template GetBufferByByte<float>(
                GDN::CHUNK_FWD_HO_FUSED_UB_H_TAIL_FLOAT_OFFSET);
        AscendC::LocalTensor<float> accumUb =
            resource.ubBuf.template GetBufferByByte<float>(
                GDN::CHUNK_FWD_HO_FUSED_UB_H_TAIL_ACCUM_OFFSET);
        AscendC::LocalTensor<ElementK> weightInputUb =
            resource.ubBuf.template GetBufferByByte<ElementK>(
                GDN::CHUNK_FWD_HO_FUSED_UB_H_TAIL_WEIGHT_INPUT_OFFSET);
        AscendC::LocalTensor<float> weightFloatUb =
            resource.ubBuf.template GetBufferByByte<float>(
                GDN::CHUNK_FWD_HO_FUSED_UB_H_TAIL_WEIGHT_FLOAT_OFFSET);

        for (uint32_t kRow = rowBegin; kRow < rowEnd; ++kRow) {
            AscendC::Duplicate(accumUb, 0.0f, offsets.vBlockDim);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t tokenRow = 0; tokenRow < offsets.blockTokens; ++tokenRow) {
                AscendC::DataCopy(
                    weightInputUb,
                    gmK[offsets.wkOffset + tokenRow * kHeadDim],
                    kHeadDim);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                AscendC::Cast(
                    weightFloatUb, weightInputUb, AscendC::RoundMode::CAST_NONE,
                    kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_S>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(tailEventId);
                AscendC::DataCopy(
                    inputUb,
                    gmVUpdateWorkspace[offsets.vWorkOffset + tokenRow * offsets.vBlockDim],
                    offsets.vBlockDim);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(tailEventId);
                AscendC::Cast(
                    floatUb, inputUb, AscendC::RoundMode::CAST_NONE,
                    offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                const float weight = weightFloatUb.GetValue(kRow);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(tailEventId);
                AscendC::Muls(floatUb, floatUb, weight, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(accumUb, accumUb, floatUb, offsets.vBlockDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(tailEventId);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(tailEventId);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(tailEventId);
            AscendC::DataCopy(
                gmHWorkspace[offsets.hWorkOffset + kRow * offsets.vBlockDim],
                accumUb, offsets.vBlockDim);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(tailEventId);
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
                GDN::CHUNK_FWD_HO_FUSED_UB_H_CALC_OFFSET);
        AscendC::LocalTensor<ElementH> hUb =
            resource.ubBuf.template GetBufferByByte<ElementH>(
                GDN::CHUNK_FWD_HO_FUSED_UB_H_SLOT_OFFSET);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        if (useInitialState) {
            if constexpr (!std::is_same<ElementInitialState, ElementH>::value) {
                AscendC::DataCopy(stateUb, gmInitialState[initialStateOffset], copyElems);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Cast(hUb, stateUb, AscendC::RoundMode::CAST_RINT, copyElems);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopy(gmH[hOffset], hUb, copyElems);
            } else {
                AscendC::DataCopy(stateUb, gmInitialState[initialStateOffset], copyElems);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                AscendC::DataCopy(gmH[hOffset], stateUb, copyElems);
            }
        } else {
            AscendC::Duplicate(hUb, static_cast<ElementH>(0), copyElems);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(gmH[hOffset], hUb, copyElems);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
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
