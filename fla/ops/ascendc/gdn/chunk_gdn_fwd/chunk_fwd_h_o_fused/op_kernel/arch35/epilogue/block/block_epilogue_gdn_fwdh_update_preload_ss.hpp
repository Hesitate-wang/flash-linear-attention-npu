/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_UPDATE_PRELOAD_SS_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_UPDATE_PRELOAD_SS_HPP
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "../gdn_fwd_h_epilogue_policies_mode2.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "../../../chunk_fwd_h_o_fused_sync.h"
#include "../../chunk_fwd_ho_fused_a5_single_slot_ub_layout.h"

namespace Catlass::Epilogue::Block {

template <
    class HOutputType_,
    class GInputType_,
    class HInputType_,
    class HUpdateInputType_,
    class FinalStateType_
>
class BlockEpilogue <
    EpilogueAtlasGDNFwdHUpdatePreloadSingleSlot,
    HOutputType_,
    GInputType_,
    HInputType_,
    HUpdateInputType_,
    FinalStateType_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasGDNFwdHUpdatePreloadSingleSlot;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using HElementOutput = typename HOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using HElementInput = typename HInputType_::Element;
    using HUpdateElementInput = typename HUpdateInputType_::Element;
    using FinalStateElement = typename FinalStateType_::Element;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {

        constexpr uint32_t CALC_BUF_OFFSET = GDN::CHUNK_FWD_HO_FUSED_UB_VAR_CALC_OFFSET;
        constexpr uint32_t SLOT_BUF_0_OFFSET = GDN::CHUNK_FWD_HO_FUSED_UB_VAR_H_DIRECT_OFFSET;
        constexpr uint32_t SLOT_BUF_2_OFFSET = GDN::CHUNK_FWD_HO_FUSED_UB_VAR_V_TILE_OFFSET;
        constexpr uint32_t SLOT_BUF_3_OFFSET = GDN::CHUNK_FWD_HO_FUSED_UB_VAR_H_TILE_OFFSET;
        constexpr uint32_t SLOT_G_BUF_OFFSET = GDN::CHUNK_FWD_HO_FUSED_UB_VAR_GATE_OFFSET;
        constexpr uint32_t SLOT_G_SUB_BUF_OFFSET = GDN::CHUNK_FWD_HO_FUSED_UB_VAR_G_LAST_OFFSET;
        constexpr uint32_t SLOT_G_INPUT_BUF_OFFSET = GDN::CHUNK_FWD_HO_FUSED_UB_VAR_G_LAST_OFFSET;


        calcUbTensor = resource.ubBuf.template GetBufferByByte<float>(CALC_BUF_OFFSET);

        hUpdateUbTensor = resource.ubBuf.template GetBufferByByte<float>(SLOT_BUF_0_OFFSET);
        hUbTensor = resource.ubBuf.template GetBufferByByte<HElementOutput>(SLOT_BUF_3_OFFSET);
        glastUbTensor = resource.ubBuf.template GetBufferByByte<float>(SLOT_G_INPUT_BUF_OFFSET);


    }

    CATLASS_DEVICE
    ~BlockEpilogue() {}

    __simd_vf__ inline void Vec2PreVF(
        __ubuf__  float* dstAddr, __ubuf__ HElementOutput* srcAddr, float muls,
        uint32_t count, uint32_t oneRepeatSize, uint16_t repeatOuterTimes, uint16_t repeatInnerTimes
    ) {
        static constexpr AscendC::Reg::CastTrait castTraitHalfToFloatZero = {
            AscendC::Reg::RegLayout::ZERO,
            AscendC::Reg::SatMode::NO_SAT,
            AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_NONE
        };
        static constexpr AscendC::Reg::CastTrait castTraitHalfToFloatOne = {
            AscendC::Reg::RegLayout::ONE,
            AscendC::Reg::SatMode::NO_SAT,
            AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_NONE
        };

        AscendC::Reg::RegTensor<HElementOutput> srcReg;
        AscendC::Reg::RegTensor<float> castReg0;
        AscendC::Reg::RegTensor<float> castReg1;
        AscendC::Reg::RegTensor<float> mulsReg0;
        AscendC::Reg::RegTensor<float> mulsReg1;
        AscendC::Reg::MaskReg maskFull32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskFull16 = AscendC::Reg::CreateMask<half, AscendC::Reg::MaskPattern::ALL>();

        for (uint16_t outIdx = 0; outIdx < repeatOuterTimes; ++outIdx) {
            for (uint16_t inIdx = 0; inIdx < repeatInnerTimes; ++inIdx) {
                uint32_t loadOffset = (outIdx * repeatInnerTimes + inIdx) * oneRepeatSize;
                AscendC::Reg::LoadAlign(srcReg, srcAddr + loadOffset);
                AscendC::Reg::Cast<float, HElementOutput, castTraitHalfToFloatZero>(castReg0, srcReg, maskFull32);
                AscendC::Reg::Cast<float, HElementOutput, castTraitHalfToFloatOne>(castReg1, srcReg, maskFull32);
                AscendC::Reg::Muls(mulsReg0, castReg0, muls, maskFull32);
                AscendC::Reg::Muls(mulsReg1, castReg1, muls, maskFull32);
                __ubuf__ float* storeAddr = dstAddr + loadOffset;
                AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_INTLV_B32>(storeAddr, mulsReg0, mulsReg1, maskFull32);
            }
        }
    }

    // dst = src * scalar + dst, one regbase pass (FP32 final-state path).
    __simd_vf__ inline void Vec2ScaleAddFp32VF(
        __ubuf__ float* dstAddr, __ubuf__ float* srcAddr, float scalar, uint32_t count)
    {
        constexpr uint32_t FP32_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        AscendC::Reg::RegTensor<float> srcReg;
        AscendC::Reg::RegTensor<float> dstReg;
        AscendC::Reg::MaskReg mask;
        uint32_t remaining = count;
        uint32_t offset = 0;
        while (remaining > 0) {
            mask = AscendC::Reg::UpdateMask<float>(remaining);
            AscendC::Reg::LoadAlign(srcReg, srcAddr + offset);
            AscendC::Reg::LoadAlign(dstReg, dstAddr + offset);
            AscendC::Reg::Muls(srcReg, srcReg, scalar, mask);
            AscendC::Reg::Add(dstReg, dstReg, srcReg, mask);
            AscendC::Reg::StoreAlign(dstAddr + offset, dstReg, mask);
            offset += FP32_PER_REG;
        }
    }

    __simd_vf__ inline void Vec2CalcVF(
        __ubuf__ HElementOutput* ndAddr, __ubuf__  float* src0Addr, __ubuf__ float* src1Addr,
        uint32_t count, uint32_t oneRepeatSize, uint16_t repeatOuterTimes, uint16_t repeatInnerTimes,
        float scalar
    ) {
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

        AscendC::Reg::RegTensor<float> srcReg00;
        AscendC::Reg::RegTensor<float> srcReg01;
        AscendC::Reg::RegTensor<float> srcReg10;
        AscendC::Reg::RegTensor<float> srcReg11;
        AscendC::Reg::RegTensor<float> addReg0;
        AscendC::Reg::RegTensor<float> addReg1;
        AscendC::Reg::RegTensor<HElementOutput> castReg;
        AscendC::Reg::MaskReg maskFull32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskFull16 = AscendC::Reg::CreateMask<half, AscendC::Reg::MaskPattern::ALL>();

        for (uint16_t outIdx = 0; outIdx < repeatOuterTimes; ++outIdx) {
            for (uint16_t inIdx = 0; inIdx < repeatInnerTimes; ++inIdx) {
                uint32_t loadOffset = (outIdx * repeatInnerTimes + inIdx) * oneRepeatSize;
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(srcReg00, srcReg01, src0Addr + loadOffset);
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(srcReg10, srcReg11, src1Addr + loadOffset);
                // exp(g_last) scaling folded in: two register multiplies instead
                // of a separate full-tile UB Muls pass with its own barrier.
                AscendC::Reg::Muls(srcReg00, srcReg00, scalar, maskFull32);
                AscendC::Reg::Muls(srcReg01, srcReg01, scalar, maskFull32);
                AscendC::Reg::Add(addReg0, srcReg00, srcReg10, maskFull32);
                AscendC::Reg::Add(addReg1, srcReg01, srcReg11, maskFull32);
                AscendC::Reg::Cast<HElementOutput, float, castTraitFloatToHalfOne>(castReg, addReg1, maskFull32);
                AscendC::Reg::Cast<HElementOutput, float, castTraitFloatToHalfZero>(castReg, addReg0, maskFull32);
                __ubuf__ HElementOutput* storeAddr = ndAddr + loadOffset;
                AscendC::Reg::StoreAlign(storeAddr, castReg, maskFull16);
            }
        }
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<HElementOutput> hOutput,
        AscendC::GlobalTensor<FinalStateElement> finalState,
        AscendC::GlobalTensor<FinalStateElement> initialState,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::GlobalTensor<HElementInput> hInput,
        AscendC::GlobalTensor<float> hUpdateInput,
        uint32_t blockTokens,
        uint32_t kHeadDim,
        uint32_t vHeadDim,
        Arch::CrossCoreFlag hc2KvUpdateReady,
        bool isInitialState,
        bool isFinalState,
        bool storeFinalState,
        bool useInitialState
    )
    {
        uint32_t mActual = kHeadDim;
        uint32_t nActual = vHeadDim;
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t nOffset = 0;
        int64_t offsetH = mOffset * nActual + nOffset;


        constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() * 2 / sizeof(float);
        uint16_t repeatOuterTimes = mActualThisSubBlock;
        uint16_t repeatInnerTimes = vHeadDim / oneRepeatSize;

        AscendC::ResetMask();

        AscendC::GlobalTensor<HElementOutput> hOutputThisSubBlock = hOutput[offsetH];
        AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;
        AscendC::GlobalTensor<HElementInput> hInputThisSubBlock = hInput[offsetH];
        AscendC::GlobalTensor<FinalStateElement> finalStateThisSubBlock = finalState[offsetH];

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_OUTPUT);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_INPUT);

        AscendC::DataCopy(hUbTensor, hInputThisSubBlock, mActualThisSubBlock * nActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);

        GElementInput gLastVal = gInputThisSubBlock.GetValue(blockTokens - 1);
        float gLastFloat = 0.0f;
        if constexpr(std::is_same<GElementInput, float>::value) {
            gLastFloat = gLastVal;
        } else if constexpr(std::is_same<GElementInput, half>::value) {
            gLastFloat = (float)gLastVal;
        } else if constexpr(std::is_same<GElementInput, bfloat16_t>::value) {
            gLastFloat = AscendC::ToFloat(gLastVal);
        }
        glastUbTensor.SetValue(0, gLastFloat);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_SCALAR);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_SCALAR);
        AscendC::Exp(glastUbTensor, glastUbTensor, 1);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_SCALAR);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_SCALAR);
        float muls = glastUbTensor.GetValue(0);

        Arch::CrossCoreWaitFlag(hc2KvUpdateReady);

        if constexpr (std::is_same<FinalStateElement, float>::value) {
            if (storeFinalState) {
                if (isInitialState && !useInitialState) {
                    AscendC::Duplicate(calcUbTensor, 0.0f, mActualThisSubBlock * nActual);
                    AscendC::PipeBarrier<PIPE_V>();
                } else {
                    if (isInitialState) {
                        AscendC::DataCopy(calcUbTensor, initialState[offsetH],
                                          mActualThisSubBlock * nActual);
                    } else {
                        AscendC::DataCopy(calcUbTensor, finalStateThisSubBlock,
                                          mActualThisSubBlock * nActual);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
                }
            } else {
                AscendC::Cast(calcUbTensor, hUbTensor, AscendC::RoundMode::CAST_NONE,
                              mActualThisSubBlock * nActual);
                AscendC::PipeBarrier<PIPE_V>();
            }
        } else {
            AscendC::Cast(calcUbTensor, hUbTensor, AscendC::RoundMode::CAST_NONE,
                          mActualThisSubBlock * nActual);
            AscendC::PipeBarrier<PIPE_V>();
        }

        if constexpr (std::is_same<FinalStateElement, float>::value) {
            if (storeFinalState) {
                Vec2ScaleAddFp32VF(
                    (__ubuf__ float*)hUpdateUbTensor.GetPhyAddr(),
                    (__ubuf__ float*)calcUbTensor.GetPhyAddr(), muls,
                    mActualThisSubBlock * nActual);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
                AscendC::DataCopy(finalStateThisSubBlock, hUpdateUbTensor,
                                  mActualThisSubBlock * nActual);
                if (!isFinalState) {
                    AscendC::Cast(hUbTensor, hUpdateUbTensor,
                                  AscendC::RoundMode::CAST_RINT,
                                  mActualThisSubBlock * nActual);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
                    AscendC::DataCopy(hOutputThisSubBlock, hUbTensor,
                                      mActualThisSubBlock * nActual);
                }
            } else {
                Vec2CalcVF(
                    (__ubuf__ HElementOutput*)hUbTensor.GetPhyAddr(),
                    (__ubuf__ float*)calcUbTensor.GetPhyAddr(),
                    (__ubuf__ float*)hUpdateUbTensor.GetPhyAddr(),
                    mActualThisSubBlock * nActual, oneRepeatSize,
                    repeatOuterTimes, repeatInnerTimes, muls);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
                AscendC::DataCopy(hOutputThisSubBlock, hUbTensor,
                                  mActualThisSubBlock * nActual);
            }
        } else {
            if (isFinalState) {
                Vec2CalcVF(
                    (__ubuf__ HElementOutput*)hUbTensor.GetPhyAddr(),
                    (__ubuf__ float*)calcUbTensor.GetPhyAddr(),
                    (__ubuf__ float*)hUpdateUbTensor.GetPhyAddr(),
                    mActualThisSubBlock * nActual, oneRepeatSize,
                    repeatOuterTimes, repeatInnerTimes, muls);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
                AscendC::DataCopy(finalStateThisSubBlock, hUbTensor, mActualThisSubBlock * nActual);
            } else {
                Vec2CalcVF(
                    (__ubuf__ HElementOutput*)hUbTensor.GetPhyAddr(),
                    (__ubuf__ float*)calcUbTensor.GetPhyAddr(),
                    (__ubuf__ float*)hUpdateUbTensor.GetPhyAddr(),
                    mActualThisSubBlock * nActual, oneRepeatSize,
                    repeatOuterTimes, repeatInnerTimes, muls);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_TRANSFER);
                AscendC::DataCopy(hOutputThisSubBlock, hUbTensor,
                                  mActualThisSubBlock * nActual);
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_INPUT);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_INPUT);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_INPUT);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(GDN::CHUNK_FWD_HO_FUSED_EVENT_H_OUTPUT);


    }

private:

    AscendC::LocalTensor<float> calcUbTensor;

    AscendC::LocalTensor<float> hUpdateUbTensor;
    AscendC::LocalTensor<HElementOutput> hUbTensor;
    AscendC::LocalTensor<float> glastUbTensor;


};
}

#endif
