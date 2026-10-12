/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_VNEW_PRELOAD_SS_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_VNEW_PRELOAD_SS_HPP
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "../gdn_fwd_h_epilogue_policies_mode2.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "../../../chunk_fwd_h_o_fused_sync.h"
#include "block_epilogue_gdn_fwdh_regbase.hpp"
#include "../../chunk_fwd_ho_fused_a5_single_slot_ub_layout.h"



namespace Catlass::Epilogue::Block {

namespace detail {
template <bool USE_EXP2>
static __simd_vf__ inline void GateExpRegbaseMode2(__ubuf__ float *gate, uint32_t count)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t FP32_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr float LN2 = 0.6931471805599453f;
    RegTensor<float> gateReg;
    RegTensor<float> lastReg;
    MaskReg mask;
    LoadAlign<float, LoadDist::DIST_BRC_B32>(lastReg, gate + (count - 1));
    uint32_t remaining = count;
    uint32_t offset = 0;
    while (remaining > 0) {
        mask = UpdateMask<float>(remaining);
        LoadKdaFloat(gateReg, gate + offset);
        Sub(gateReg, lastReg, gateReg, mask);
        if constexpr (USE_EXP2) {
            Muls(gateReg, gateReg, LN2, mask);
        }
        Exp(gateReg, gateReg, mask);
        StoreKdaFloat(gate + offset, gateReg, mask);
        offset += FP32_PER_REG;
    }
}
} // namespace detail

template <
    class VOutputType_,
    class GInputType_,
    class UInputType_,
    class WSInputType_,
    class FinalStateType_
>
class BlockEpilogue <
    EpilogueAtlasGDNFwdHVnewPreloadSingleSlot,
    VOutputType_,
    GInputType_,
    UInputType_,
    WSInputType_,
    FinalStateType_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasGDNFwdHVnewPreloadSingleSlot;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using VElementOutput = typename VOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using UElementInput = typename UInputType_::Element;
    using WSElementInput = typename WSInputType_::Element;
    using FinalStateElement = typename FinalStateType_::Element;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {

        constexpr uint32_t CALC_BUF_OFFSET = GDN::CHUNK_FWD_HO_FUSED_UB_H_CALC_OFFSET;
        constexpr uint32_t SLOT_BUF_0_OFFSET = GDN::CHUNK_FWD_HO_FUSED_UB_H_DIRECT_OFFSET;
        constexpr uint32_t SLOT_BUF_1_OFFSET = GDN::CHUNK_FWD_HO_FUSED_UB_H_STAGING_OFFSET + 16U * 1024U;
        constexpr uint32_t SLOT_BUF_2_OFFSET = GDN::CHUNK_FWD_HO_FUSED_UB_H_STAGING_OFFSET + 32U * 1024U;
        constexpr uint32_t SLOT_BUF_3_OFFSET = GDN::CHUNK_FWD_HO_FUSED_UB_H_STAGING_OFFSET + 48U * 1024U;
        constexpr uint32_t SLOT_G_BUF_OFFSET = GDN::CHUNK_FWD_HO_FUSED_UB_H_SMALL_OFFSET;
        constexpr uint32_t SLOT_G_SUB_BUF_OFFSET = GDN::CHUNK_FWD_HO_FUSED_UB_H_SMALL_OFFSET + 2U * 1024U;
        constexpr uint32_t SLOT_G_INPUT_BUF_OFFSET = GDN::CHUNK_FWD_HO_FUSED_UB_H_SMALL_OFFSET + 3U * 1024U;

        calcUbTensor = resource.ubBuf.template GetBufferByByte<float>(CALC_BUF_OFFSET);

        uUbTensor = resource.ubBuf.template GetBufferByByte<UElementInput>(SLOT_BUF_2_OFFSET);
        wsUbTensor = resource.ubBuf.template GetBufferByByte<float>(SLOT_BUF_0_OFFSET);
        gUbTensor = resource.ubBuf.template GetBufferByByte<float>(SLOT_G_BUF_OFFSET);
        gLastUbTensor = resource.ubBuf.template GetBufferByByte<float>(SLOT_G_SUB_BUF_OFFSET);
        gInputUbTensor = resource.ubBuf.template GetBufferByByte<GElementInput>(SLOT_G_SUB_BUF_OFFSET);
        vNewOutputUbTensor = resource.ubBuf.template GetBufferByByte<VElementOutput>(SLOT_BUF_2_OFFSET);
        vNewDecayUbTensor = resource.ubBuf.template GetBufferByByte<VElementOutput>(SLOT_BUF_2_OFFSET);



    }

    CATLASS_DEVICE
    ~BlockEpilogue() {}

    __simd_vf__ inline void Vec1CalcVF(
        __ubuf__ VElementOutput* vNewAddr, __ubuf__ VElementOutput* vOutAddr,
        __ubuf__  float* vSrcAddr, __ubuf__ float* uSrcAddr, __ubuf__ float* gSrcAddr,
        uint32_t mActualThisSubBlock, uint32_t nvActual
    ) {
        constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() * 2 / sizeof(float);
        uint16_t repeatOuterTimes = mActualThisSubBlock;
        uint16_t repeatInnerTimes = nvActual / oneRepeatSize;

        static constexpr AscendC::Reg::CastTrait castTraitFloatToHalfZero = {
            AscendC::Reg::RegLayout::ZERO,
            AscendC::Reg::SatMode::NO_SAT,
            AscendC::Reg::MaskMergeMode::MERGING,
            AscendC::RoundMode::CAST_RINT
        };
        static constexpr AscendC::Reg::CastTrait castTraitFloatToHalfOne = {
            AscendC::Reg::RegLayout::ONE,
            AscendC::Reg::SatMode::NO_SAT,
            AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT
        };

        AscendC::Reg::RegTensor<float> vReg0;
        AscendC::Reg::RegTensor<float> vReg1;
        AscendC::Reg::RegTensor<float> uReg0;
        AscendC::Reg::RegTensor<float> uReg1;
        AscendC::Reg::RegTensor<float> gReg;
        AscendC::Reg::RegTensor<float> subReg0;
        AscendC::Reg::RegTensor<float> subReg1;
        AscendC::Reg::RegTensor<float> mulReg0;
        AscendC::Reg::RegTensor<float> mulReg1;
        AscendC::Reg::RegTensor<VElementOutput> castReg;
        AscendC::Reg::MaskReg maskFull32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskFull16 = AscendC::Reg::CreateMask<half, AscendC::Reg::MaskPattern::ALL>();

        for (uint16_t outIdx = 0; outIdx < repeatOuterTimes; ++outIdx) {
            for (uint16_t inIdx = 0; inIdx < repeatInnerTimes; ++inIdx) {
                uint32_t loadOffset = (outIdx * repeatInnerTimes + inIdx) * oneRepeatSize;
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(vReg0, vReg1, vSrcAddr + loadOffset);
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(uReg0, uReg1, uSrcAddr + loadOffset);
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(gReg, gSrcAddr + outIdx);
                AscendC::Reg::Sub(subReg0, uReg0, vReg0, maskFull32);
                AscendC::Reg::Sub(subReg1, uReg1, vReg1, maskFull32);
                AscendC::Reg::Mul(mulReg0, subReg0, gReg, maskFull32);
                AscendC::Reg::Mul(mulReg1, subReg1, gReg, maskFull32);
                AscendC::Reg::Cast<VElementOutput, float, castTraitFloatToHalfOne>(castReg, mulReg1, maskFull32);
                AscendC::Reg::Cast<VElementOutput, float, castTraitFloatToHalfZero>(castReg, mulReg0, maskFull32);
                __ubuf__ VElementOutput* storeAddr = vNewAddr + loadOffset;
                AscendC::Reg::StoreAlign(storeAddr, castReg, maskFull16);
            }
        }

    }

    __simd_vf__ inline void Vec1PostVF(
        __ubuf__ VElementOutput* vOutAddr, __ubuf__  float* vSrcAddr, __ubuf__ float* uSrcAddr,
        uint32_t mActualThisSubBlock, uint32_t nvActual
    ) {
        constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() * 2 / sizeof(float);
        uint16_t repeatOuterTimes = mActualThisSubBlock;
        uint16_t repeatInnerTimes = nvActual / oneRepeatSize;

        static constexpr AscendC::Reg::CastTrait castTraitFloatToHalfZero = {
            AscendC::Reg::RegLayout::ZERO,
            AscendC::Reg::SatMode::NO_SAT,
            AscendC::Reg::MaskMergeMode::MERGING,
            AscendC::RoundMode::CAST_RINT
        };
        static constexpr AscendC::Reg::CastTrait castTraitFloatToHalfOne = {
            AscendC::Reg::RegLayout::ONE,
            AscendC::Reg::SatMode::NO_SAT,
            AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT
        };

        AscendC::Reg::RegTensor<float> vReg0;
        AscendC::Reg::RegTensor<float> vReg1;
        AscendC::Reg::RegTensor<float> uReg0;
        AscendC::Reg::RegTensor<float> uReg1;
        AscendC::Reg::RegTensor<float> subReg0;
        AscendC::Reg::RegTensor<float> subReg1;
        AscendC::Reg::RegTensor<VElementOutput> castReg;
        AscendC::Reg::MaskReg maskFull32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskFull16 = AscendC::Reg::CreateMask<half, AscendC::Reg::MaskPattern::ALL>();

        for (uint16_t outIdx = 0; outIdx < repeatOuterTimes; ++outIdx) {
            for (uint16_t inIdx = 0; inIdx < repeatInnerTimes; ++inIdx) {
                uint32_t loadOffset = (outIdx * repeatInnerTimes + inIdx) * oneRepeatSize;
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(vReg0, vReg1, vSrcAddr + loadOffset);
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(uReg0, uReg1, uSrcAddr + loadOffset);
                AscendC::Reg::Sub(subReg0, uReg0, vReg0, maskFull32);
                AscendC::Reg::Sub(subReg1, uReg1, vReg1, maskFull32);
                AscendC::Reg::Cast<VElementOutput, float, castTraitFloatToHalfOne>(castReg, subReg1, maskFull32);
                AscendC::Reg::Cast<VElementOutput, float, castTraitFloatToHalfZero>(castReg, subReg0, maskFull32);
                __ubuf__ VElementOutput* storeAddr = vOutAddr + loadOffset;
                AscendC::Reg::StoreAlign(storeAddr, castReg, maskFull16);
            }
        }
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<VElementOutput> vnewOutput,
        AscendC::GlobalTensor<VElementOutput> vnewdecayOutput,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::GlobalTensor<UElementInput> uInput,
        AscendC::GlobalTensor<float> wsInput,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vHeadDim,
        Arch::CrossCoreFlag cube1Done,
        Arch::CrossCoreFlag vec1Done,
        bool isInitialState,
        bool isFinalState,
        bool storeFinalState
    )
    {
        uint32_t mActual = chunkSize;
        uint32_t nkActual = kHeadDim;
        uint32_t nvActual = vHeadDim;

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t nOffset = 0;
        int64_t offsetK = mOffset * nvActual + nOffset;
        int64_t offsetD = 0;

        uint32_t gbrcStart, gbrcRealStart, gbrcReptime, gbrcEffStart, gbrcEffEnd;
        if(subBlockIdx==0)
        {
            gbrcStart = 0;
            gbrcRealStart = 0;
            gbrcReptime = (mActualThisSubBlock + 8 - 1) / 8;

        }
        else
        {
            gbrcStart = mActualPerSubBlock;
            gbrcRealStart = gbrcStart & ~15;
            gbrcReptime = (mActual - gbrcRealStart + 8 - 1) / 8;
        }
        gbrcEffStart = gbrcStart-gbrcRealStart;
        gbrcEffEnd = gbrcEffStart + mActualThisSubBlock;
        uint32_t dstShape_[2] = {gbrcReptime*8, nvActual};
        uint32_t srcShape_[2] = {gbrcReptime*8, 1};

        AscendC::ResetMask();

        AscendC::GlobalTensor<VElementOutput> vnewOutputThisSubBlock = vnewOutput[offsetK];
        AscendC::GlobalTensor<VElementOutput> vnewdecayOutputThisSubBlock = vnewdecayOutput[offsetK];
        AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;
        AscendC::GlobalTensor<UElementInput> uInputThisSubBlock = uInput[offsetK];
        AscendC::GlobalTensor<float> wsInputThisSubBlock = wsInput[offsetK];

        // The shared V output slot is reused once for VUpdate and again for
        // V_new.  The previous chunk's MTE3 read must finish before this chunk
        // starts writing that slot.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::DataCopy(uUbTensor, uInputThisSubBlock, mActualThisSubBlock * nvActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::Cast(calcUbTensor, uUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nvActual);

        if constexpr(std::is_same<GElementInput, float>::value) {
            AscendC::DataCopyParams gUbParams{1, (uint16_t)(mActual * sizeof(float)), 0, 0};
            AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(gUbTensor, gInputThisSubBlock, gUbParams, gUbPadParams); // copy g
        } else {
            AscendC::DataCopyParams gUbParams{1, (uint16_t)(mActual * sizeof(half)), 0, 0};
            AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(gInputUbTensor, gInputThisSubBlock, gUbParams, gUbPadParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
        if constexpr(!std::is_same<GElementInput, float>::value) {
            AscendC::Cast(gUbTensor, gInputUbTensor, AscendC::RoundMode::CAST_NONE, mActual);
        }

        // Single regbase pass: g_last (broadcast from the last valid element) - g
        // then exp.  Replaces the scalar read-back, Duplicate, Sub and Exp UB
        // sequence with its three PipeBarriers.  The preload family uses the
        // natural exponential, so the LN2 scaling of the fused schedule is off.
        AscendC::VF_CALL<detail::GateExpRegbaseMode2<false>>(
            reinterpret_cast<__ubuf__ float *>(gUbTensor.GetPhyAddr()), mActual);
        AscendC::PipeBarrier<PIPE_V>();

        Arch::CrossCoreWaitFlag(cube1Done);

        Vec1CalcVF(
            (__ubuf__ VElementOutput*)vNewDecayUbTensor.GetPhyAddr(), (__ubuf__ VElementOutput*)vNewOutputUbTensor.GetPhyAddr(),
            (__ubuf__ float*)wsUbTensor.GetPhyAddr(), (__ubuf__ float*)calcUbTensor.GetPhyAddr(), (__ubuf__ float*)gUbTensor[mOffset].GetPhyAddr(),
            mActualThisSubBlock, nvActual
        );


        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::DataCopy(vnewdecayOutputThisSubBlock, vNewDecayUbTensor, mActualThisSubBlock * nvActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);

        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vec1Done);

        Vec1PostVF(
            (__ubuf__ VElementOutput*)vNewOutputUbTensor.GetPhyAddr(), (__ubuf__ float*)wsUbTensor.GetPhyAddr(), (__ubuf__ float*)calcUbTensor.GetPhyAddr(),
            mActualThisSubBlock, nvActual
        );
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::DataCopy(vnewOutputThisSubBlock, vNewOutputUbTensor, mActualThisSubBlock * nvActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        // Single-slot fused schedule: H and O share one core, so V_new readiness
        // is carried by the driver's 0x2 tokens (hVec1Done / oAiv1Done) and no
        // mode-0 cross-core H/O rendezvous happens here.

    }

private:

    AscendC::LocalTensor<float> calcUbTensor;

    AscendC::LocalTensor<UElementInput> uUbTensor;
    AscendC::LocalTensor<float> wsUbTensor;
    AscendC::LocalTensor<float> gUbTensor;
    AscendC::LocalTensor<float> gLastUbTensor;
    AscendC::LocalTensor<GElementInput> gInputUbTensor;
    AscendC::LocalTensor<VElementOutput> vNewOutputUbTensor;
    AscendC::LocalTensor<VElementOutput> vNewDecayUbTensor;



};
}

#endif
