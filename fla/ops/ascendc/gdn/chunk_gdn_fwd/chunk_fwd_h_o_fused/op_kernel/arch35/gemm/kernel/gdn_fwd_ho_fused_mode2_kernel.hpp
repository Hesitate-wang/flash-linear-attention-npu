/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef CATLASS_GDN_FWD_HO_FUSED_MODE2_KERNEL_HPP
#define CATLASS_GDN_FWD_HO_FUSED_MODE2_KERNEL_HPP

#include <type_traits>
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "../block/block_scheduler_gdn_fwd_ho_fused_mode2.hpp"
#include "../block/block_mmad_gdn_fused_mode2.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fused_qkmask.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fused_vnew.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fused_update.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fused_output.hpp"

namespace Catlass::Gemm::Kernel {

// Mode2 uses one complete task per MIX core and exactly two AIV lanes.
// The two switches select Cube->Vec and Vec->Cube transport independently.
template<class DataT, class GateT, class StateT = float,
         bool CubeToUB = true, bool VecToL1 = false>
class GDNFwdHOFusedMode2Kernel {
    using ArchTag = Arch::Ascend950;
    using Ub = Epilogue::gdnFusedMode2UbMallocTable;
    using Scheduler = Block::BlockSchedulerGdnFwdHOFusedMode2;
    using Chunk = Block::GdnFusedMode2Chunk;
    using Data = GemmType<DataT, layout::RowMajor>;
    using Gate = GemmType<GateT, layout::RowMajor>;
    using Work = GemmType<float, layout::RowMajor>;
    using QkMask = Epilogue::Block::BlockEpilogue<
        Epilogue::EpilogueAtlasGDNFusedMode2Qkmask, Data, Gate, Work, Work>;
    static constexpr uint32_t hL1 = 64U * 1024U;
    static constexpr uint32_t maskL1 = 96U * 1024U;
    static constexpr uint32_t vnewL1 = 104U * 1024U;
    static constexpr uint32_t decayL1 = 120U * 1024U;
    static_assert(sizeof(DataT) == 2);
    static_assert(hL1 + 128U * 128U * sizeof(DataT) <= maskL1);
    static_assert(maskL1 + 64U * 64U * sizeof(DataT) <= vnewL1);
    static_assert(vnewL1 + 64U * 128U * sizeof(DataT) <= decayL1);
    static_assert(decayL1 + 64U * 128U * sizeof(DataT) <= ArchTag::L1_SIZE);
    // 0x2: Cube producers 0..4, Vector producers 5..8. Every generation
    // has exactly one matching consumer, including the final chunk token.
    Arch::CrossCoreFlag oc1Ready = 0;
    Arch::CrossCoreFlag hc1Ready = 1;
    Arch::CrossCoreFlag oc2Ready = 2;
    Arch::CrossCoreFlag oc3Ready = 3;
    Arch::CrossCoreFlag hc2Ready = 4;
    Arch::CrossCoreFlag vnewReady = 5;
    Arch::CrossCoreFlag chunkFinished = 6;
    Arch::CrossCoreFlag decayReady = 7;
    Arch::CrossCoreFlag hReady = 8;

public:
    CATLASS_DEVICE void Init(
        GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR initialState,
        GM_ADDR q, GM_ADDR h, GM_ADDR vnew, GM_ADDR finalState, GM_ADDR o,
        GM_ADDR tiling, GM_ADDR workspace)
    {
        const auto *t = reinterpret_cast<const __gm__ GDN::ChunkFwdHOFusedTilingData *>(tiling);
        useInitialState_ = t->useInitialState;
        storeFinalState_ = t->storeFinalState;
        scale_ = t->scale;
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(k));
        w_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(w));
        u_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(u));
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(g));
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(q));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(h));
        vnew_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(vnew));
        initial_.SetGlobalBuffer(reinterpret_cast<__gm__ StateT *>(initialState));
        final_.SetGlobalBuffer(reinterpret_cast<__gm__ StateT *>(finalState));
        o_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(o));
        wh_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + t->vWorkspaceOffset));
        update_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + t->hWorkspaceOffset));
        decay_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(workspace + t->vUpdateWorkspaceOffset));
        attn_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + t->oAttnWorkspaceOffset));
        masked_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(workspace + t->oAfterMaskWorkspaceOffset));
        inter_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + t->oHWorkspaceOffset));
        intra_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + t->oVWorkspaceOffset));
        uint32_t core;
        if ASCEND_IS_AIV {
            core = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        } else {
            core = AscendC::GetBlockIdx();
        }
        scheduler_.Init(t, core);
    }

    CATLASS_DEVICE void Process()
    {
        if (!scheduler_.isRunning) return;
        if ASCEND_IS_AIC { ProcessCube(); }
        if ASCEND_IS_AIV { ProcessVector(); }
    }

private:
    template<class LayoutA, class LayoutB, uint32_t M, uint32_t N, uint32_t K,
             bool ExternalA = false, bool ExternalB = false>
    CATLASS_DEVICE void Mmad(
        AscendC::GlobalTensor<DataT> a, AscendC::GlobalTensor<DataT> b,
        AscendC::GlobalTensor<float> gmDst, uint32_t ubOffset,
        AscendC::LocalTensor<DataT> l1A = {}, AscendC::LocalTensor<DataT> l1B = {})
    {
        auto ta = tla::MakeTensor(a, tla::MakeLayout<DataT, LayoutA>(M, K), Arch::PositionGM{});
        auto tb = tla::MakeTensor(b, tla::MakeLayout<DataT, LayoutB>(K, N), Arch::PositionGM{});
        Block::BlockMmadGdnFusedMode2<DataT, LayoutA, LayoutB, M, N, K,
            CubeToUB, ExternalA, ExternalB> block(resource_, l1A, l1B);
        if constexpr (CubeToUB) {
            auto tc = tla::MakeTensor(resource_.ubBuf.template GetBufferByByte<float>(ubOffset),
                tla::MakeLayout<float, layout::RowMajor>(M, N), Arch::PositionUB{});
            block(ta, tb, tc, {M, N, K});
        } else {
            auto tc = tla::MakeTensor(gmDst, tla::MakeLayout<float, layout::RowMajor>(M, N),
                Arch::PositionGM{});
            block(ta, tb, tc, {M, N, K});
        }
    }

    CATLASS_DEVICE void ProcessCube()
    {
        // Every Mode2 block uses one stage and the same A/B/C event IDs.
        // Seed these shared free-slot tokens once; each Mmad passes them on.
        Block::BlockMmadGdnFusedMode2<DataT, layout::RowMajor, layout::ColumnMajor,
            64, 64, 128, CubeToUB> pipeline(resource_);
        pipeline.preSetFlags();
        auto l1H = resource_.l1Buf.template GetBufferByByte<DataT>(hL1);
        auto l1Mask = resource_.l1Buf.template GetBufferByByte<DataT>(maskL1);
        auto l1Vnew = resource_.l1Buf.template GetBufferByByte<DataT>(vnewL1);
        auto l1Decay = resource_.l1Buf.template GetBufferByByte<DataT>(decayL1);
        do {
            const Chunk &c = scheduler_.offsets;
            // This closes reuse of all previous-generation Cube destinations.
            // The first chunk has no previous OV2 token to consume.
            
            Mmad<layout::RowMajor, layout::ColumnMajor, 64, 64, 128>(q_[c.qk], k_[c.qk],
                attn_[c.attnWork], Ub::attn_ping);
            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(oc1Ready);

            Arch::CrossCoreWaitFlag(hReady);
            Mmad<layout::RowMajor, layout::RowMajor, 64, 128, 128, false, VecToL1>(w_[c.w], h_[c.hInput],
                wh_[c.tokenWork], Ub::v_work_ping, {}, l1H);
            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(hc1Ready);
            // OC2 must consume this H before HC2 makes HV2 runnable.
            if (!c.first) Arch::CrossCoreWaitFlag(chunkFinished);
            Mmad<layout::RowMajor, layout::RowMajor, 64, 128, 128, false, VecToL1>(q_[c.qk], h_[c.hInput],
                inter_[c.tokenWork], Ub::o_inter_ping, {}, l1H);
            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(oc2Ready);

            Arch::CrossCoreWaitFlag(decayReady);
            if (!c.last || storeFinalState_) {
                Mmad<layout::ColumnMajor, layout::RowMajor, 128, 128, 64, false, VecToL1>(k_[c.qk], decay_[c.tokenWork],
                    update_[c.stateWork], Ub::h_work_ping, {}, l1Decay);
            }
            // Publish a token even when the last update was skipped.
            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(hc2Ready);
            Arch::CrossCoreWaitFlag(vnewReady);
            Mmad<std::conditional_t<VecToL1, layout::nZ, layout::RowMajor>,
                layout::RowMajor, 64, 128, 64, VecToL1, VecToL1>(masked_[c.attnWork], vnew_[c.uv],
                intra_[c.tokenWork], Ub::o_intra_ping, l1Mask, l1Vnew);
            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(oc3Ready);
        } while (scheduler_.Next());
        // OV2 always publishes, including the last generation. Do not leave
        // a stale completion token at kernel exit.
        Arch::CrossCoreWaitFlag(chunkFinished);
        pipeline.finalWaitFlags();
    }

    CATLASS_DEVICE void InitializeState()
    {
        constexpr uint32_t elements = 64U * 128U;
        const Chunk &c = scheduler_.offsets;
        const uint32_t row = AscendC::GetSubBlockIdx() * 64U;
        const uint32_t offset = row * 128U;
        auto state = resource_.ubBuf.template GetBufferByByte<float>(Ub::h_ping);
        auto cast = resource_.ubBuf.template GetBufferByByte<DataT>(Ub::v_new_decay_ping);
        AscendC::ResetMask();
        if (!useInitialState_) {
            AscendC::Duplicate(state, 0.0f, elements);
        } else {
            if constexpr (std::is_same_v<StateT, float>) {
                AscendC::DataCopy(state, initial_[c.state + offset], elements);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID5);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID5);
            } else {
                AscendC::DataCopy(cast, initial_[c.state + offset], elements);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID5);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID5);
                AscendC::Cast(state, cast, AscendC::RoundMode::CAST_NONE, elements);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(cast, state, AscendC::RoundMode::CAST_RINT, elements);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
        if constexpr (VecToL1) {
            Epilogue::Block::mode2_fixed64_detail::CopyRowsToL1Zn<128, 128, 64>(
                resource_.l1Buf.template GetBufferByByte<DataT>(hL1), cast, row);
        } else {
            AscendC::DataCopy(h_[c.hInput + offset], cast, elements);
        }
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(hReady);
    }

    CATLASS_DEVICE void ProcessVector()
    {
        auto mask = resource_.ubBuf.template GetBufferByByte<float>(Ub::mask);
        AscendC::ResetMask();
        AscendC::Duplicate(mask, 0.0f, 32U * 32U);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < 32; ++row) {
            AscendC::Duplicate(mask[row * 32], 1.0f, row + 1);
        }
        AscendC::PipeBarrier<PIPE_V>();
        InitializeState();
        Epilogue::Block::mode2_fixed64_detail::GateBuffer<GateT> gate(resource_);
        QkMask ov1(resource_, mask, gate.Tensor(), CubeToUB, VecToL1,
            resource_.l1Buf.template GetBufferByByte<DataT>(maskL1));
        Epilogue::Block::GdnFusedMode2Vnew<DataT, GateT> hv1(resource_, gate.Tensor(), CubeToUB, VecToL1,
            resource_.l1Buf.template GetBufferByByte<DataT>(vnewL1),
            resource_.l1Buf.template GetBufferByByte<DataT>(decayL1));
        Epilogue::Block::GdnFusedMode2Update<DataT, GateT, StateT> hv2(resource_, gate.Tensor(), CubeToUB, VecToL1,
            resource_.l1Buf.template GetBufferByByte<DataT>(hL1));
        Epilogue::Block::GdnFusedMode2Output<DataT, GateT> ov2(resource_, gate.Tensor(), CubeToUB, false);
        do {
            const Chunk &c = scheduler_.offsets;
            if (!c.first) {
                // The next MTE2 gate load must not pass the previous chunk's
                // Vector reads. OV2's MTE3_V wait also retires prior DMA sources.
                // AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID7);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID7);
            }
            gate.Prepare(g_[c.gate]);
            ov1(masked_[c.attnWork], attn_[c.attnWork], oc1Ready);
            hv1(vnew_[c.uv], decay_[c.tokenWork], u_[c.uv], wh_[c.tokenWork],
                hc1Ready, decayReady, vnewReady);
            hv2(h_[c.last ? c.hInput : c.hOutput], final_[c.state], update_[c.stateWork],
                hc2Ready, hReady, c.last, storeFinalState_);
            ov2(o_[c.uv], inter_[c.tokenWork], intra_[c.tokenWork], scale_,
                oc2Ready, oc3Ready, chunkFinished);
        } while (scheduler_.Next());
        // Also covers a final state copy without a next-chunk H-ready.
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID7);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
    }

    Arch::Resource<ArchTag> resource_;
    Scheduler scheduler_;
    bool useInitialState_{false}, storeFinalState_{false};
    float scale_{1.0f};
    AscendC::GlobalTensor<DataT> k_, w_, u_, q_, h_, vnew_, decay_, masked_, o_;
    AscendC::GlobalTensor<GateT> g_;
    AscendC::GlobalTensor<StateT> initial_, final_;
    AscendC::GlobalTensor<float> wh_, update_, attn_, inter_, intra_;
};
} // namespace Catlass::Gemm::Kernel
#endif
