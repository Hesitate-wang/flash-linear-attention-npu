/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_UPDATE_PRELOAD_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_UPDATE_PRELOAD_HPP
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "../gdn_fwd_h_epilogue_policies.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Block {

template <
    class HOutputType_,
    class GInputType_,
    class HInputType_,
    class HUpdateInputType_,
    class FinalStateType_
>
class BlockEpilogue <
    EpilogueAtlasGDNFwdHUpdatePreload,
    HOutputType_,
    GInputType_,
    HInputType_,
    HUpdateInputType_,
    FinalStateType_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasGDNFwdHUpdatePreload;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using HElementOutput = typename HOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using HElementInput = typename HInputType_::Element;
    using HUpdateElementInput = typename HUpdateInputType_::Element;
    using FinalStateElement = typename FinalStateType_::Element;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {

        constexpr uint32_t CALC_BUF_OFFSET = 0;
        constexpr uint32_t PING_BUF_0_OFFSET = 32 * 1024;
        constexpr uint32_t PING_BUF_1_OFFSET = 48 * 1024;
        constexpr uint32_t PING_BUF_2_OFFSET = 64 * 1024;
        constexpr uint32_t PING_BUF_3_OFFSET = 80 * 1024;
        constexpr uint32_t PONG_BUF_0_OFFSET = 96 * 1024;
        constexpr uint32_t PONG_BUF_1_OFFSET = 112 * 1024;
        constexpr uint32_t PONG_BUF_2_OFFSET = 128 * 1024;
        constexpr uint32_t PONG_BUF_3_OFFSET = 144 * 1024;
        constexpr uint32_t PING_G_BUF_OFFSET = 160 * 1024;
        constexpr uint32_t PONG_G_BUF_OFFSET = 161 * 1024;
        constexpr uint32_t PING_G_SUB_BUF_OFFSET = 162 * 1024;
        constexpr uint32_t PONG_G_SUB_BUF_OFFSET = 163 * 1024;
        constexpr uint32_t PING_G_INPUT_BUF_OFFSET = 164 * 1024;
        constexpr uint32_t PONG_G_INPUT_BUF_OFFSET = 165 * 1024;
        constexpr uint32_t SHARE_BUF_OFFSET = 166 * 1024;


        calcUbTensor = resource.ubBuf.template GetBufferByByte<float>(CALC_BUF_OFFSET);

        hUpdateUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_BUF_0_OFFSET);
        hUbTensor_ping = resource.ubBuf.template GetBufferByByte<HElementOutput>(PING_BUF_3_OFFSET);
        glastUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_G_INPUT_BUF_OFFSET);

        hUpdateUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_BUF_0_OFFSET);
        hUbTensor_pong = resource.ubBuf.template GetBufferByByte<HElementOutput>(PONG_BUF_3_OFFSET);
        glastUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_G_INPUT_BUF_OFFSET);

    }

    CATLASS_DEVICE
    ~BlockEpilogue() {}

    // 一次遍历完成 (calc + hUpdate) -> half。列宽能被一个 repeat 整除时才能用（vHeadDim=128 的
    // 默认配置走这里，与 V 切片改造前逐字一致）；V 切片后的窄列宽回落到普通向量算子。
    __simd_vf__ inline void Vec2CalcVF(
        __ubuf__ HElementOutput* ndAddr, __ubuf__  float* src0Addr, __ubuf__ float* src1Addr,
        uint32_t count, uint32_t oneRepeatSize, uint16_t repeatOuterTimes, uint16_t repeatInnerTimes
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
                AscendC::Reg::Add(addReg0, srcReg00, srcReg10, maskFull32);
                AscendC::Reg::Add(addReg1, srcReg01, srcReg11, maskFull32);
                AscendC::Reg::Cast<HElementOutput, float, castTraitFloatToHalfOne>(castReg, addReg1, maskFull32);
                AscendC::Reg::Cast<HElementOutput, float, castTraitFloatToHalfZero>(castReg, addReg0, maskFull32);
                __ubuf__ HElementOutput* storeAddr = ndAddr + loadOffset;
                AscendC::Reg::StoreAlign(storeAddr, castReg, maskFull16);
            }
        }
    }

    // h[i+1] = h[i] * exp(g_last) + h_work 的收尾：宽列走融合 SIMD，窄列走 Add + Cast。
    CATLASS_DEVICE
    void ComputeHNextHalf(
        AscendC::LocalTensor<HElementOutput> hUbTensor,
        AscendC::LocalTensor<float> calcUbTensor,
        AscendC::LocalTensor<float> hUpdateUbTensor,
        uint32_t rows,
        uint32_t cols)
    {
        constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() * 2 / sizeof(float);
        uint32_t count = rows * cols;
        if (cols % oneRepeatSize == 0) {
            Vec2CalcVF(
                (__ubuf__ HElementOutput*)hUbTensor.GetPhyAddr(),
                (__ubuf__ float*)calcUbTensor.GetPhyAddr(),
                (__ubuf__ float*)hUpdateUbTensor.GetPhyAddr(),
                count, oneRepeatSize,
                static_cast<uint16_t>(rows), static_cast<uint16_t>(cols / oneRepeatSize));
            return;
        }
        AscendC::Add<float>(hUpdateUbTensor, calcUbTensor, hUpdateUbTensor, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(hUbTensor, hUpdateUbTensor, AscendC::RoundMode::CAST_RINT, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // 按列切片搬运：rows 行、每行 cols 个元素，源/目的行跨度为 stride。
    // cols == stride 时退化为一次连续 DataCopy（vSplits == 1 的路径与旧实现一致）。
    template <typename Element>
    CATLASS_DEVICE
    void CopyGmToUb(
        AscendC::LocalTensor<Element> dst,
        AscendC::GlobalTensor<Element> src,
        uint32_t rows,
        uint32_t cols,
        uint32_t srcStride)
    {
        if (cols == srcStride) {
            AscendC::DataCopy(dst, src, rows * cols);
            return;
        }
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(Element)),
            static_cast<uint32_t>((srcStride - cols) * sizeof(Element)),
            0,
            0};
        AscendC::DataCopyPadExtParams<Element> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(dst, src, copyParams, padParams);
    }

    template <typename Element>
    CATLASS_DEVICE
    void CopyUbToGm(
        AscendC::GlobalTensor<Element> dst,
        AscendC::LocalTensor<Element> src,
        uint32_t rows,
        uint32_t cols,
        uint32_t dstStride)
    {
        if (cols == dstStride) {
            AscendC::DataCopy(dst, src, rows * cols);
            return;
        }
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(Element)),
            0,
            static_cast<uint32_t>((dstStride - cols) * sizeof(Element)),
            0};
        AscendC::DataCopyPad(dst, src, copyParams);
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<HElementOutput> hOutput,
        AscendC::GlobalTensor<FinalStateElement> finalState,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::GlobalTensor<HElementInput> hInput,
        AscendC::GlobalTensor<float> hUpdateInput,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vBlockDim,
        uint32_t vHeadDim,
        Arch::CrossCoreFlag cube2Done,
        bool needHLoad,
        bool isFinalState,
        bool isPing
    )
    {
        uint32_t mActual = kHeadDim;
        // 本任务只处理 V 轴的 vBlockDim 列，h / final_state 的行跨度仍是 vHeadDim。
        uint32_t nActual = vBlockDim;
        uint32_t stateStride = vHeadDim;
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t elementCount = mActualThisSubBlock * nActual;
        int64_t offsetH = static_cast<int64_t>(mOffset) * stateStride;
        // needHLoad=false 时 h[c] 直接复用本 UB 缓冲里上一个 chunk 写下的 h（见调用点：
        // isInitialState 或 stream 槽位不稳定时才置 true）。h 输出/最终状态照旧写 GM。

        AscendC::ResetMask();

        AscendC::GlobalTensor<HElementOutput> hOutputThisSubBlock = hOutput[offsetH];
        AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;
        AscendC::GlobalTensor<HElementInput> hInputThisSubBlock = hInput[offsetH];
        AscendC::GlobalTensor<FinalStateElement> finalStateThisSubBlock = finalState[offsetH];

        uint32_t pingpongFlag = isPing ? 0 : pongBaseEvent;
        AscendC::LocalTensor<float> hUpdateUbTensor = isPing ? hUpdateUbTensor_ping : hUpdateUbTensor_pong;
        AscendC::LocalTensor<HElementOutput> hUbTensor = isPing ? hUbTensor_ping : hUbTensor_pong;
        AscendC::LocalTensor<float> glastUbTensor = isPing ? glastUbTensor_ping : glastUbTensor_pong;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
        if (needHLoad) {
            CopyGmToUb(hUbTensor, hInputThisSubBlock, mActualThisSubBlock, nActual, stateStride);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
        }

        GElementInput gLastVal = gInputThisSubBlock.GetValue(chunkSize-1);
        float gLastFloat = 0.0f;
        if constexpr(std::is_same<GElementInput, float>::value) {
            gLastFloat = gLastVal;
        } else if constexpr(std::is_same<GElementInput, half>::value) {
            gLastFloat = (float)gLastVal;
        } else if constexpr(std::is_same<GElementInput, bfloat16_t>::value) {
            gLastFloat = AscendC::ToFloat(gLastVal);
        }
        glastUbTensor.SetValue(0, gLastFloat);

        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);
        AscendC::Exp(glastUbTensor, glastUbTensor, 1);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + pingpongFlag);
        float muls = glastUbTensor.GetValue(0);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);
        if (needHLoad) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
        }
        AscendC::Cast(calcUbTensor, hUbTensor, AscendC::RoundMode::CAST_NONE, elementCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(calcUbTensor, calcUbTensor, muls, elementCount);
        AscendC::PipeBarrier<PIPE_V>();

        Arch::CrossCoreWaitFlag(cube2Done);

        // h_update 直接来自 Cube C2 写入的 UB（与 hUpdateUbTensor 同址），不再从 GM 读回。
        if (isFinalState) {
            if constexpr(std::is_same<FinalStateElement, float>::value) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
                AscendC::Add<float>(hUpdateUbTensor, calcUbTensor, hUpdateUbTensor, elementCount);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                CopyUbToGm(finalStateThisSubBlock, hUpdateUbTensor, mActualThisSubBlock, nActual, stateStride);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pingpongFlag);
            } else {
                ComputeHNextHalf(hUbTensor, calcUbTensor, hUpdateUbTensor, mActualThisSubBlock, nActual);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                CopyUbToGm(finalStateThisSubBlock, hUbTensor, mActualThisSubBlock, nActual, stateStride);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
            }
        } else {
            ComputeHNextHalf(hUbTensor, calcUbTensor, hUpdateUbTensor, mActualThisSubBlock, nActual);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
            CopyUbToGm(hOutputThisSubBlock, hUbTensor, mActualThisSubBlock, nActual, stateStride);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
        }


    }

private:
    uint32_t pongBaseEvent = 4;

    AscendC::LocalTensor<float> calcUbTensor;

    AscendC::LocalTensor<float> hUpdateUbTensor_ping;
    AscendC::LocalTensor<HElementOutput> hUbTensor_ping;
    AscendC::LocalTensor<float> glastUbTensor_ping;

    AscendC::LocalTensor<float> hUpdateUbTensor_pong;
    AscendC::LocalTensor<HElementOutput> hUbTensor_pong;
    AscendC::LocalTensor<float> glastUbTensor_pong;

};
}

#endif