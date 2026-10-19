/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef CATLASS_EPILOGUE_GDN_FUSED_VNEW_HPP
#define CATLASS_EPILOGUE_GDN_FUSED_VNEW_HPP

#include "block_epilogue_gdn_fused_mode2_common.hpp"

namespace Catlass::Epilogue::Block {
namespace mode2_vnew_detail {
// Independent of HC1: prepare the 64 relative decays in one register.
__simd_vf__ inline void PrepareRelativeDecay(
    __ubuf__ float *relativeDecay, __ubuf__ float *gate)
{
    using namespace AscendC::MicroAPI;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> last, current, relative;
    LoadAlign<float, LoadDist::DIST_BRC_B32>(last, gate + 63);
    LoadAlign(current, gate);
    Sub(relative, last, current, all);
    Exp(relative, relative, all);
    StoreAlign(relativeDecay, relative, all);
}

// Compute only the HC2 operand. Keep u_fp32 and WH intact for the later
// un-decayed Vnew pass; the subtraction here must not be rounded before Mul.
__simd_vf__ inline void ComputeVnewDecay(
    __ubuf__ float *vUpdate,
    __ubuf__ float *u, __ubuf__ float *wh, __ubuf__ float *relativeDecay, uint32_t rowBegin)
{
    using namespace AscendC::MicroAPI;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> relative, u0, u1, w0, w1, v0, v1, d0, d1;
    for (uint16_t row = 0; row < 32; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * 128;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(relative, relativeDecay + rowBegin + row);
        LoadAlign(u0, u + offset);
        LoadAlign(u1, u + offset + 64);
        LoadAlign(w0, wh + offset);
        LoadAlign(w1, wh + offset + 64);
        Sub(v0, u0, w0, all);
        Sub(v1, u1, w1, all);
        Mul(d0, v0, relative, all);
        Mul(d1, v1, relative, all);
        StoreAlign(vUpdate + offset, d0, all);
        StoreAlign(vUpdate + offset + 64, d1, all);
    }
}
} // namespace mode2_vnew_detail

// Full chunk=64, K=V=128, exactly two AIV lanes, natural exp, one slot.
// Direct HC1 input: Fixpipe SPLIT_M into v_work_ping (37 KiB), 32x128 FP32 per AIV.
template <class DataT, class GateT>
class GdnFusedMode2Vnew {
public:
    using Ub = gdnFusedMode2UbMallocTable;
    using ArchTag = Arch::Ascend950;
    static_assert(sizeof(DataT) == 2, "HV1 data must be FP16/BF16");
    static constexpr uint32_t kElements = 32 * 128;
    // OV1 must have returned: reuse 256 B of its Broadcast/mask workspace.
    static_assert(Ub::shard_mem + 64U * sizeof(float) <= Ub::v_work_ping);
    static_assert(Ub::v_work_ping + kElements * sizeof(float) <= Ub::u_ping);
    static_assert(Ub::u_ping + kElements * sizeof(DataT) <= Ub::u_fp_ping);
    static_assert(Ub::u_fp_ping + kElements * sizeof(float) <= Ub::v_new_ping);
    static_assert(Ub::v_new_ping + kElements * sizeof(float) <= Ub::v_new_decay_ping);
    static_assert(Ub::v_new_decay_ping + kElements * sizeof(float) <= Ub::h_ping);

    // gate is the caller's FP32 [64] view, prepared once for this chunk.
    CATLASS_DEVICE GdnFusedMode2Vnew(Arch::Resource<ArchTag> &resource,
                                   AscendC::LocalTensor<float> gate,
                                   bool enableL0C2UB = false,
                                   bool enableUB2L1 = false,
                                   AscendC::LocalTensor<DataT> l1Vnew = {},
                                   AscendC::LocalTensor<DataT> l1VUpdate = {})
        : l1Vnew_(l1Vnew), l1VUpdate_(l1VUpdate),
          enableL0C2UB_(enableL0C2UB), enableUB2L1_(enableUB2L1), gate_(gate)
    {
        relativeDecay_ = resource.ubBuf.template GetBufferByByte<float>(Ub::shard_mem);
        wh_ = resource.ubBuf.template GetBufferByByte<float>(Ub::v_work_ping);
        u_ = resource.ubBuf.template GetBufferByByte<DataT>(Ub::u_ping);
        uFp32_ = resource.ubBuf.template GetBufferByByte<float>(Ub::u_fp_ping);
        vNew_ = resource.ubBuf.template GetBufferByByte<float>(Ub::v_new_ping);
        vUpdate_ = resource.ubBuf.template GetBufferByByte<float>(Ub::v_new_decay_ping);
        // Raw u is dead after its FP32 cast. Reuse only that 8 KiB slot for
        // both sequential outputs; u_fp32 must survive the early decay output.
        castOutput_ = resource.ubBuf.template GetBufferByByte<DataT>(Ub::u_ping);
    }

    // GM arguments are full 64x128 chunk origins. With direct input, whInput
    // is unused; HC1 ready must order Fixpipe completion. L1 output uses two
    // exclusive 64x128 zN tiles, and the corresponding GM outputs are unused.
    // OV1 must have released scratch and Cube must not overwrite WH early.
    // The two notifications are published here, in order: decay -> HC2,
    // then Vnew -> OC3. The caller must not publish duplicate ready tokens.
    // The early HC2 notification does NOT release WH/u_fp32: they stay live
    // until the Vnew pass ends. Each ready is published on MTE3 after its DMA.
    // OV1's mask must already be ready before the OC3 notification is used.
    CATLASS_DEVICE void operator()(
        AscendC::GlobalTensor<DataT> vNewOutput,
        AscendC::GlobalTensor<DataT> vUpdateOutput,
        AscendC::GlobalTensor<DataT> uInput,
        AscendC::GlobalTensor<float> whInput,
        Arch::CrossCoreFlag hc1Ready,
        Arch::CrossCoreFlag vnewDecayReadyForHc2,
        Arch::CrossCoreFlag vnewReadyForOc3)
    {
        const uint32_t rowBegin = AscendC::GetSubBlockIdx() * 32;
        const uint32_t offset = rowBegin * 128;
        AscendC::ResetMask();
        AscendC::DataCopy(u_, uInput[offset], kElements);
        AscendC::VF_CALL<mode2_vnew_detail::PrepareRelativeDecay>(
            reinterpret_cast<__ubuf__ float *>(relativeDecay_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gate_.GetPhyAddr()));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
        AscendC::Cast(uFp32_, u_, AscendC::RoundMode::CAST_NONE, kElements);
        AscendC::PipeBarrier<PIPE_V>();
        Arch::CrossCoreWaitFlag(hc1Ready);
        if (!enableL0C2UB_) {
            AscendC::DataCopy(wh_, whInput[offset], kElements);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
        }
        // First produce the HC2 operand and let Cube start consuming it.
        AscendC::VF_CALL<mode2_vnew_detail::ComputeVnewDecay>(
            reinterpret_cast<__ubuf__ float *>(vUpdate_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(uFp32_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(wh_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(relativeDecay_.GetPhyAddr()), rowBegin);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(castOutput_, vUpdate_, AscendC::RoundMode::CAST_RINT, kElements);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
        if (enableUB2L1_) {
            mode2_fixed64_detail::CopyRowsToL1Zn<64, 128, 32>(l1VUpdate_, castOutput_, rowBegin);
        } else {
            AscendC::DataCopy(vUpdateOutput[offset], castOutput_, kElements);
        }
        // Both output casts share castOutput_. Protect the same-chunk V write
        // from the decay DMA; defer the wait until after the independent Sub.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        // AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        // AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vnewDecayReadyForHc2);

        // Only after decay is published, produce the OC3 operand. Recompute
        // the unrounded subtraction; do not derive Vnew from rounded decay.
        AscendC::Sub(vNew_, uFp32_, wh_, kElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::Cast(castOutput_, vNew_, AscendC::RoundMode::CAST_RINT, kElements);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
        if (enableUB2L1_) {
            mode2_fixed64_detail::CopyRowsToL1Zn<64, 128, 32>(l1Vnew_, castOutput_, rowBegin);
        } else {
            AscendC::DataCopy(vNewOutput[offset], castOutput_, kElements);
        }
        // AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        // AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        // AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        // AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vnewReadyForOc3);
    }
private:
    AscendC::LocalTensor<DataT> l1Vnew_, l1VUpdate_;
    bool enableL0C2UB_, enableUB2L1_;
    AscendC::LocalTensor<float> gate_;
    AscendC::LocalTensor<float> wh_, uFp32_, vNew_, vUpdate_, relativeDecay_;
    AscendC::LocalTensor<DataT> u_, castOutput_;
};
} // namespace Catlass::Epilogue::Block
#endif
