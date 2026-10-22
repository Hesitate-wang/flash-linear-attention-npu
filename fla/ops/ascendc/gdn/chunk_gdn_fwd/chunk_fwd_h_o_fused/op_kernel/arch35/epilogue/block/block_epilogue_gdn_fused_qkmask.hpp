/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef CATLASS_EPILOGUE_BLOCK_GDN_FUSED_MODE2_QKMASK_HPP
#define CATLASS_EPILOGUE_BLOCK_GDN_FUSED_MODE2_QKMASK_HPP

#include <type_traits>
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "kernel_operator.h"
#include "block_epilogue_gdn_fused_mode2_common.hpp"

namespace Catlass::Epilogue::Block {
namespace mode2_qkmask_detail {

// Prepare gate decay independently of the OC1 result. expandedMask is the temporary
// [32,64] mask in the now-idle Broadcast workspace, not a second resident mask.
__simd_vf__ inline void PrepareGateMask(
    __ubuf__ float *left, __ubuf__ float *up,
    __ubuf__ float *expandedMask)
{
    using namespace AscendC::MicroAPI;
    RegTensor<float> rowGate, colGate, value, causal, zero;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    Duplicate(zero, 0.0f, all);
    for (uint16_t row = 0; row < 32; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * 64;
        LoadAlign(rowGate, left + offset);
        LoadAlign(colGate, up + offset);
        LoadAlign(causal, expandedMask + offset);
        Sub(value, rowGate, colGate, all);
        // Match the existing natural-exp QK-mask path. Invalid positions are
        // replaced before Exp and selected to exact zero afterward.
        Mins(value, value, 0.0f, all);
        MaskReg valid;
        Compare<float, AscendC::CMPMODE::GT>(valid, causal, zero, all);
        Select(value, value, zero, valid);
        Exp(value, value, all);
        Select(value, value, zero, valid);
        StoreAlign(left + offset, value, all);
    }
}

} // namespace mode2_qkmask_detail

template <class AOutputType_, class GInputType_, class AInputType_, class MaskInputType_>
class BlockEpilogue<EpilogueAtlasGDNFusedMode2Qkmask,
                    AOutputType_, GInputType_, AInputType_, MaskInputType_> {
public:
    using Ub = gdnFusedMode2UbMallocTable;
    using ArchTag = Arch::Ascend950;
    using AElementOutput = typename AOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using AElementInput = typename AInputType_::Element;
    static_assert(std::is_same_v<AElementInput, float>, "QK input must be FP32");
    static_assert(sizeof(AElementOutput) == 2, "masked QK output must be FP16/BF16");
    static_assert(sizeof(GElementInput) == 2 || std::is_same_v<GElementInput, float>,
                  "gate input must be FP16/BF16/FP32");

    // Requires exactly two AIV lanes, chunk size 64, K=V=128, natural exp.
    // The caller releases the prior chunk before entry; OC1 ready is waited here.
    // Caller initializes mask once before the chunk loop, and prepares gate once
    // per chunk. Both tensors are read-only here and shared with later stages.
    CATLASS_DEVICE BlockEpilogue(Arch::Resource<ArchTag> &resource,
                                AscendC::LocalTensor<float> mask,
                                AscendC::LocalTensor<float> gate,
                                bool enableL0C2UB = false,
                                bool enableUB2L1 = false,
                                AscendC::LocalTensor<AElementOutput> l1Output = {})
        : mask_(mask), gate_(gate), l1Output_(l1Output),
          enableL0C2UB_(enableL0C2UB), enableUB2L1_(enableUB2L1)
    {
        tileInput_ = resource.ubBuf.template GetBufferByByte<uint16_t>(Ub::nz_convert_src);
        tileTransposed_ = resource.ubBuf.template GetBufferByByte<uint16_t>(Ub::nz_convert_dst);
        attn_ = resource.ubBuf.template GetBufferByByte<float>(Ub::attn_ping);
        left_ = resource.ubBuf.template GetBufferByByte<float>(Ub::g_left_ping);
        up_ = resource.ubBuf.template GetBufferByByte<float>(Ub::g_up_ping);
        scratch_ = resource.ubBuf.template GetBufferByByte<uint8_t>(Ub::shard_mem);
        attn_.SetSize(32 * 64);
        left_.SetSize(32 * 64);
        up_.SetSize(32 * 64);
        scratch_.SetSize(Ub::broadcast_scratch_bytes);
        expandedMask_ = resource.ubBuf.template GetBufferByByte<float>(Ub::shard_mem);
        // After the VF has consumed up_, use that same 8 KiB for the 4 KiB cast.
        output_ = resource.ubBuf.template GetBufferByByte<AElementOutput>(Ub::g_up_ping);
    }

    // enableL0C2UB: OC1 Fixpipe must deliver 32x64 FP32 row-major per AIV
    // to Ub::attn_ping (0 B), using SPLIT_M and stride 64. OC1 ready must
    // order completion of those writes. The GM attn argument is unused.
    // enableUB2L1: caller supplies an exclusive 64x64 nZ L1 tile; its GM
    // output argument is unused. Cube must consume that same layout directly.
    // Arguments refer to the beginning of the current chunk, not to an AIV tile.
    // This fixed-shape entry deliberately has no tail, stage or ping/pong branch.
    CATLASS_DEVICE void operator()(
        AscendC::GlobalTensor<AElementOutput> maskedQk,
        AscendC::GlobalTensor<AElementInput> attnInput,
        Arch::CrossCoreFlag oc1Ready)
    {
        constexpr uint32_t elements = 32 * 64;
        const uint32_t lane = AscendC::GetSubBlockIdx();
        const uint32_t rowBegin = lane * 32;
        AscendC::ResetMask();
        uint32_t dstShape[2] = {32, 64};
        uint32_t upShape[2] = {1, 64};
        uint32_t leftShape[2] = {32, 1};
        AscendC::Broadcast<float, 2, 0>(up_, gate_, dstShape, upShape, scratch_);
        AscendC::PipeBarrier<PIPE_V>(); // Both Broadcast calls reuse scratch_.
        AscendC::Broadcast<float, 2, 1>(left_, gate_[rowBegin], dstShape, leftShape, scratch_);
        AscendC::PipeBarrier<PIPE_V>();

        // [[T,0],[1,T]]: both lanes reuse the same resident 32x32 triangle T.
        // Reuse the finished Broadcast scratch for a full-width transient mask.
        AscendC::Duplicate(expandedMask_, lane == 0 ? 0.0f : 1.0f, elements);
        AscendC::PipeBarrier<PIPE_V>();
        // Copy 32 floats per row; dst row = 8 blocks, src row = 4 blocks.
        AscendC::Copy(expandedMask_[lane * 32], mask_, 32, 32, {1, 1, 8, 4});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::VF_CALL<mode2_qkmask_detail::PrepareGateMask>(
            reinterpret_cast<__ubuf__ float *>(left_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(up_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(expandedMask_.GetPhyAddr()));
        AscendC::PipeBarrier<PIPE_V>();
        // Gate broadcast, exp and causal masking are independent of OC1.
        // Consume ready only when its QK output is about to be read.
        Arch::CrossCoreWaitFlag(oc1Ready);
        if (!enableL0C2UB_) {
            AscendC::DataCopy(attn_, attnInput[rowBegin * 64], elements);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }
        AscendC::Mul(left_, attn_, left_, elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(output_, left_, AscendC::RoundMode::CAST_RINT, elements);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        if (enableUB2L1_) {
            CopyOutputToL1(rowBegin);
        } else {
            AscendC::DataCopy(maskedQk[rowBegin * 64], output_, elements);
        }
        // Output DMA may still be pending; the driver orders reuse across chunks.
        // AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        // AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        // AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        // AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

private:
    // Fixed 64x64 nZ output. Each AIV writes its own two 16-row tiles
    // across all four column tiles; the lane row ranges are disjoint.
    CATLASS_DEVICE void CopyOutputToL1(uint32_t rowBegin)
    {
        mode2_fixed64_detail::CopyRowsToL1Nz<64, 64, 32>(
            l1Output_, output_, tileInput_, tileTransposed_, rowBegin);
    }

    AscendC::LocalTensor<float> attn_, left_, up_, mask_, gate_, expandedMask_;
    AscendC::LocalTensor<uint8_t> scratch_;
    AscendC::LocalTensor<AElementOutput> output_;
    AscendC::LocalTensor<AElementOutput> l1Output_;
    AscendC::LocalTensor<uint16_t> tileInput_, tileTransposed_;
    bool enableL0C2UB_;
    bool enableUB2L1_;

};

} // namespace Catlass::Epilogue::Block
#endif // CATLASS_EPILOGUE_BLOCK_GDN_FUSED_MODE2_QKMASK_HPP
