/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef CATLASS_EPILOGUE_GDN_FUSED_UPDATE_HPP
#define CATLASS_EPILOGUE_GDN_FUSED_UPDATE_HPP

#include "block_epilogue_gdn_fused_mode2_common.hpp"

namespace Catlass::Epilogue::Block {
namespace mode2_update_detail {
__simd_vf__ inline void ScaleState(
    __ubuf__ float *h, __ubuf__ float *gate)
{
    using namespace AscendC::MicroAPI;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> last, h0, h1;
    LoadAlign<float, LoadDist::DIST_BRC_B32>(last, gate + 63);
    Exp(last, last, all);
    // State rows are K=128, not the chunk's 64 token rows: 64 per AIV.
    for (uint16_t row = 0; row < 64; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * 128;
        LoadAlign(h0, h + offset);
        LoadAlign(h1, h + offset + 64);
        Mul(h0, h0, last, all);
        Mul(h1, h1, last, all);
        StoreAlign(h + offset, h0, all);
        StoreAlign(h + offset + 64, h1, all);
    }
}
} // namespace mode2_update_detail

template <class DataT, class GateT, class StateT = float>
class GdnFusedMode2Update {
public:
    using Ub = gdnFusedMode2UbMallocTable;
    using ArchTag = Arch::Ascend950;
    static_assert(sizeof(DataT) == 2, "HV2 H must be FP16/BF16");
    static_assert(std::is_same_v<StateT, float> || std::is_same_v<StateT, DataT>,
                  "state dtype must be FP32 or the H dtype");
    static constexpr uint32_t kElements = 64 * 128;
    // HV1 has completed: reuse its decay slot for DataT staging (16 KiB).
    static_assert(Ub::v_new_decay_ping + kElements * sizeof(DataT) <= Ub::h_ping);
    static_assert(Ub::h_ping + kElements * sizeof(float) <= Ub::h_work_ping);
    static_assert(Ub::h_work_ping + kElements * sizeof(float) <= Ub::o_inter_ping,
                  "HC2 output overlaps OC2 output");

    // gate is the caller's FP32 [64] view, prepared once for this chunk.
    CATLASS_DEVICE GdnFusedMode2Update(Arch::Resource<ArchTag> &resource,
                                   AscendC::LocalTensor<float> gate,
                                   bool enableL0C2UB = false,
                                   bool enableUB2L1 = false,
                                   AscendC::LocalTensor<DataT> l1H = {})
        : l1H_(l1H), enableL0C2UB_(enableL0C2UB),
          enableUB2L1_(enableUB2L1), gate_(gate)
    {
        staging_ = resource.ubBuf.template GetBufferByByte<DataT>(Ub::v_new_decay_ping);
        h_ = resource.ubBuf.template GetBufferByByte<float>(Ub::h_ping);
        hWork_ = resource.ubBuf.template GetBufferByByte<float>(Ub::h_work_ping);
    }

    // Direct HC2 input: h_work_ping (141 KiB), SPLIT_M, 64x128 FP32 per AIV.
    // L1 H output: full 128x128 zN origin. finalState is always a GM output.
    // Driver initializes FP32 h_ping once before the chunk loop. Every chunk
    // updates that resident state in place. DataT outputs are Cube operands;
    // casting them must not round the persistent FP32 state.
    // HV1 must have returned before staging_ is used. All matrices below are
    // full 128x128 state origins; gate_ holds the current 64-token chunk.
    // This entry publishes hReadyForHc1 after the next H has been copied out.
    // Caller manages slot release and publishes chunkFinished after OV2.
    CATLASS_DEVICE void operator()(
        AscendC::GlobalTensor<DataT> hOutput,
        AscendC::GlobalTensor<StateT> finalState,
        AscendC::GlobalTensor<float> hWorkInput,
        Arch::CrossCoreFlag hc2Ready,
        Arch::CrossCoreFlag hReadyForHc1,
        bool isFinalState,
        bool storeFinalState)
    {
        // Driver still produces a matching HC2 token when MMAD is skipped.
        if (isFinalState && !storeFinalState) {
            Arch::CrossCoreWaitFlag(hc2Ready);
            return;
        }
        const uint32_t offset = AscendC::GetSubBlockIdx() * kElements;
        AscendC::ResetMask();
        // Decaying the previous state does not depend on HC2 h_work.
        AscendC::VF_CALL<mode2_update_detail::ScaleState>(
            reinterpret_cast<__ubuf__ float *>(h_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gate_.GetPhyAddr()));
        AscendC::PipeBarrier<PIPE_V>();
        Arch::CrossCoreWaitFlag(hc2Ready);
        if (!enableL0C2UB_) {
            AscendC::DataCopy(hWork_, hWorkInput[offset], kElements);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID5);
        }
        AscendC::Add(h_, h_, hWork_, kElements);
        AscendC::PipeBarrier<PIPE_V>();
        const bool needsRoundedH = !isFinalState || !std::is_same_v<StateT, float>;
        if (needsRoundedH) {
            AscendC::Cast(staging_, h_, AscendC::RoundMode::CAST_RINT, kElements);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
        if constexpr (std::is_same_v<StateT, float>) {
            if (isFinalState && storeFinalState) {
                AscendC::DataCopy(finalState[offset], h_, kElements);
            }
        }
        if (!isFinalState) {
            if (enableUB2L1_) {
                mode2_fixed64_detail::CopyRowsToL1Zn<128, 128, 64>(
                    l1H_, staging_, AscendC::GetSubBlockIdx() * 64);
            } else {
                AscendC::DataCopy(hOutput[offset], staging_, kElements);
            }
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(hReadyForHc1);
        } else {
            if constexpr (std::is_same_v<StateT, DataT>) {
                AscendC::DataCopy(finalState[offset], staging_, kElements);
            }
        }
        // Across chunks, MTE3 H-ready -> HC1 -> HV1 -> HC2 orders reuse of
        // h_ and staging_; no per-call local drain is required. The driver
        // must fence the final output before reinitializing UB for a new task.
        // AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);
        // AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);
        // AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        // AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);

    }
private:
    AscendC::LocalTensor<DataT> l1H_;
    bool enableL0C2UB_, enableUB2L1_;
    AscendC::LocalTensor<float> gate_;
    AscendC::LocalTensor<DataT> staging_;
    AscendC::LocalTensor<float> h_, hWork_;
};
} // namespace Catlass::Epilogue::Block
#endif
