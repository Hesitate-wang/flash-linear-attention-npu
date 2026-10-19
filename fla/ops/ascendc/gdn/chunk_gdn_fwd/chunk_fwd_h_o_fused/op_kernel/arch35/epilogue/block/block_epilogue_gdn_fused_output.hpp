/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef CATLASS_EPILOGUE_GDN_FUSED_OUTPUT_HPP
#define CATLASS_EPILOGUE_GDN_FUSED_OUTPUT_HPP

#include "block_epilogue_gdn_fused_mode2_common.hpp"

namespace Catlass::Epilogue::Block {
namespace mode2_output_detail {
// Independent of both OC2 and OC3; keep raw gate read-only.
__simd_vf__ inline void PrepareDecay(
    __ubuf__ float *decay, __ubuf__ float *gate)
{
    using namespace AscendC::MicroAPI;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> value;
    LoadAlign(value, gate);
    Exp(value, value, all);
    StoreAlign(decay, value, all);
}

// Depends only on OC2, so finish this before waiting for OC3.
__simd_vf__ inline void ScaleInter(
    __ubuf__ float *inter, __ubuf__ float *decay, uint32_t rowBegin)
{
    using namespace AscendC::MicroAPI;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> value, h0, h1;
    for (uint16_t row = 0; row < 32; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * 128;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(value, decay + rowBegin + row);
        LoadAlign(h0, inter + offset);
        LoadAlign(h1, inter + offset + 64);
        Mul(h0, h0, value, all);
        Mul(h1, h1, value, all);
        StoreAlign(inter + offset, h0, all);
        StoreAlign(inter + offset + 64, h1, all);
    }
}
} // namespace mode2_output_detail

template <class DataT, class GateT>
class GdnFusedMode2Output {
public:
    using Ub = gdnFusedMode2UbMallocTable;
    using ArchTag = Arch::Ascend950;
    static_assert(sizeof(DataT) == 2, "OV2 output must be FP16/BF16");
    static constexpr uint32_t kElements = 32 * 128;
    // OV1/HV1 have returned; reuse 256 B of the same temporary workspace.
    static_assert(Ub::shard_mem + 64U * sizeof(float) <= Ub::v_work_ping);
    static_assert(Ub::o_inter_ping + kElements * sizeof(float) <= Ub::o_intra_ping);
    static_assert(Ub::o_intra_ping + kElements * sizeof(float) <= Ub::o_out_ping);
    static_assert(Ub::o_out_ping + kElements * sizeof(DataT) <= 248U * 1024U,
                  "OV2 output crosses the reserved communication region");

    // gate is the caller's FP32 [64] view, prepared once for this chunk.
    CATLASS_DEVICE GdnFusedMode2Output(Arch::Resource<ArchTag> &resource,
                                   AscendC::LocalTensor<float> gate,
                                   bool enableL0C2UB = true,
                                   bool enableUB2L1 = false,
                                   AscendC::LocalTensor<DataT> l1Output = {})
        : l1Output_(l1Output), enableL0C2UB_(enableL0C2UB),
          enableUB2L1_(enableUB2L1), gate_(gate)
    {
        decay_ = resource.ubBuf.template GetBufferByByte<float>(Ub::shard_mem);
        inter_ = resource.ubBuf.template GetBufferByByte<float>(Ub::o_inter_ping);
        intra_ = resource.ubBuf.template GetBufferByByte<float>(Ub::o_intra_ping);
        output_ = resource.ubBuf.template GetBufferByByte<DataT>(Ub::o_out_ping);
    }

    // Direct OC2/OC3 input: Fixpipe SPLIT_M to o_inter_ping (173 KiB) and
    // o_intra_ping (189 KiB), each 32x128 FP32 per AIV. Cube must wait for
    // the previous chunk's readers before reusing these destinations.
    // L1 output: an exclusive full 64x128 zN tile; GM output is then unused.
    // Direct mode skips both GM inputs; their ready tokens are still consumed.
    // All GM arguments start at the current chunk, not at a pre-sliced lane.
    // Publish chunkFinished internally on MTE3 after the GM/L1 output.
    CATLASS_DEVICE void operator()(
        AscendC::GlobalTensor<DataT> output,
        AscendC::GlobalTensor<float> interInput,
        AscendC::GlobalTensor<float> intraInput,
        float scale, Arch::CrossCoreFlag oc2Ready, Arch::CrossCoreFlag oc3Ready,
        Arch::CrossCoreFlag chunkFinished)
    {
        const uint32_t rowBegin = AscendC::GetSubBlockIdx() * 32;
        const uint32_t offset = rowBegin * 128;
        AscendC::ResetMask();
        AscendC::VF_CALL<mode2_output_detail::PrepareDecay>(
            reinterpret_cast<__ubuf__ float *>(decay_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gate_.GetPhyAddr()));
        AscendC::PipeBarrier<PIPE_V>();
        Arch::CrossCoreWaitFlag(oc2Ready);
        if (!enableL0C2UB_) {
            AscendC::DataCopy(inter_, interInput[offset], kElements);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
        }
        AscendC::VF_CALL<mode2_output_detail::ScaleInter>(
            reinterpret_cast<__ubuf__ float *>(inter_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(decay_.GetPhyAddr()), rowBegin);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID7);
        Arch::CrossCoreWaitFlag(oc3Ready);
        if (!enableL0C2UB_) {
            AscendC::DataCopy(intra_, intraInput[offset], kElements);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
        }
        AscendC::Add(inter_, inter_, intra_, kElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(inter_, inter_, scale, kElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(output_, inter_, AscendC::RoundMode::CAST_RINT, kElements);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
        if (enableUB2L1_) {
            mode2_fixed64_detail::CopyRowsToL1Zn<64, 128, 32>(l1Output_, output_, rowBegin);
        } else {
            AscendC::DataCopy(output[offset], output_, kElements);
        }
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(chunkFinished);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        // AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        // AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    }
private:
    AscendC::LocalTensor<DataT> l1Output_;
    bool enableL0C2UB_, enableUB2L1_;
    AscendC::LocalTensor<float> gate_;
    AscendC::LocalTensor<float> inter_, intra_, decay_;
    AscendC::LocalTensor<DataT> output_;
};
} // namespace Catlass::Epilogue::Block
#endif
