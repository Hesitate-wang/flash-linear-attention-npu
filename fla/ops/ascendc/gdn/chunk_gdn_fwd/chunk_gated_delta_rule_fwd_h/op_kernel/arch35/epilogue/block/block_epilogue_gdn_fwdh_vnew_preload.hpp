/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_VNEW_PRELOAD_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_VNEW_PRELOAD_HPP
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "../gdn_fwd_h_epilogue_policies.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "block_epilogue_gdn_fwdh_regbase.hpp"



namespace Catlass::Epilogue::Block {

template <
    class VOutputType_,
    class GInputType_,
    class UInputType_,
    class WSInputType_,
    class FinalStateType_
>
class BlockEpilogue <
    EpilogueAtlasGDNFwdHVnewPreload,
    VOutputType_,
    GInputType_,
    UInputType_,
    WSInputType_,
    FinalStateType_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasGDNFwdHVnewPreload;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using VElementOutput = typename VOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using UElementInput = typename UInputType_::Element;
    using WSElementInput = typename WSInputType_::Element;
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

        uUbTensor_ping = resource.ubBuf.template GetBufferByByte<UElementInput>(PING_BUF_1_OFFSET);
        wsUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_BUF_0_OFFSET);
        gUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_G_BUF_OFFSET);
        gLastUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_G_SUB_BUF_OFFSET);
        gInputUbTensor_ping = resource.ubBuf.template GetBufferByByte<GElementInput>(PING_G_SUB_BUF_OFFSET);
        vNewOutputUbTensor_ping = resource.ubBuf.template GetBufferByByte<VElementOutput>(PING_BUF_2_OFFSET);
        vNewDecayUbTensor_ping = resource.ubBuf.template GetBufferByByte<VElementOutput>(PING_BUF_2_OFFSET);

        uUbTensor_pong = resource.ubBuf.template GetBufferByByte<UElementInput>(PONG_BUF_1_OFFSET);
        wsUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_BUF_0_OFFSET);
        gUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_G_BUF_OFFSET);
        gLastUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_G_SUB_BUF_OFFSET);
        gInputUbTensor_pong = resource.ubBuf.template GetBufferByByte<GElementInput>(PONG_G_SUB_BUF_OFFSET);
        vNewOutputUbTensor_pong = resource.ubBuf.template GetBufferByByte<VElementOutput>(PONG_BUF_2_OFFSET);
        vNewDecayUbTensor_pong = resource.ubBuf.template GetBufferByByte<VElementOutput>(PONG_BUF_2_OFFSET);

        shareBuffer_ = resource.ubBuf.template GetBufferByByte<uint8_t>(SHARE_BUF_OFFSET);

    }

    CATLASS_DEVICE
    ~BlockEpilogue() {}

    // 一次遍历算出 v_new_decay = (u - v_work) * exp(g_last - g)。列宽能被一个 repeat 整除时才能用
    // （vHeadDim=128 的默认配置走这里，与 V 切片改造前逐字一致）；窄列宽回落到普通向量算子。
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

    // 一次遍历算出 v_new = u - v_work。
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

    // 把 src 的第一个元素广播成 count 个元素写到 dst：用寄存器广播代替"V→S 读标量→S→V"的往返。
    // src 只读一个 32B 块（会读过尾，调用方需保证仍在 UB 内）。
    __simd_vf__ inline void BroadcastLastElement(
        __ubuf__ float* dstAddr, __ubuf__ float* srcAddr, uint32_t count)
    {
        constexpr uint32_t fp32PerReg = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        AscendC::Reg::RegTensor<float> brcReg;
        AscendC::Reg::MaskReg maskFull = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(brcReg, srcAddr);
        for (uint32_t offset = 0; offset < count; offset += fp32PerReg) {
            AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(dstAddr + offset, brcReg, maskFull);
        }
    }

    // 逐行乘 exp(g_last - g_t)：matrix 是 rows x cols 的紧凑 UB 块，rowScale 从 rowScaleOffset 行开始。
    // 列宽是 64（一个 FP32 寄存器）的整数倍时走 dual-issue VF；窄列（V 切 4 份时 32 列）逐行 Muls，
    // 因为 VF 用整寄存器访存，32 列会读写到下一行。
    CATLASS_DEVICE
    void ApplyRowScale(
        AscendC::LocalTensor<float> matrix,
        AscendC::LocalTensor<float> rowScale,
        uint32_t rowScaleOffset,
        uint32_t rows,
        uint32_t cols,
        uint32_t pingpongFlag)
    {
        constexpr uint32_t fp32PerReg = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        if (cols % fp32PerReg == 0) {
            __ubuf__ float *matrixAddr = reinterpret_cast<__ubuf__ float *>(matrix.GetPhyAddr());
            __ubuf__ float *rowScaleAddr = reinterpret_cast<__ubuf__ float *>(rowScale.GetPhyAddr());
            AscendC::VF_CALL<detail::ApplyRowScaleDualIssue>(
                matrixAddr, rowScaleAddr, rowScaleOffset,
                static_cast<uint16_t>(rows), static_cast<uint16_t>(cols));
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }
        // Exp 之后 g 由 V 写，标量读取前需要一次 V→S 同步。
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + pingpongFlag);
        for (uint32_t row = 0; row < rows; ++row) {
            float scale = rowScale.GetValue(rowScaleOffset + row);
            AscendC::Muls(matrix[row * cols], matrix[row * cols], scale, cols);
        }
        AscendC::PipeBarrier<PIPE_V>();
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
        uint32_t vBlockDim,
        uint32_t vHeadDim,
        Arch::CrossCoreFlag cube1Done,
        Arch::CrossCoreFlag vec1Done,
        bool isInitialState,
        bool isFinalState,
        bool storeFinalState,
        bool isPing
    )
    {
        uint32_t mActual = chunkSize;
        uint32_t nkActual = kHeadDim;
        // 本任务只处理 V 轴的 vBlockDim 列：u / v_new 的行跨度是 vHeadDim，
        // v_work workspace 自身按 vBlockDim 紧凑排列。
        uint32_t nvActual = vBlockDim;
        uint32_t inputStride = vHeadDim;

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t elementCount = mActualThisSubBlock * nvActual;
        // 当前场景内部一定连续
        // k [B, H, T, D]
        // g [B, H, T]
        // 在外部offset的基础上进一步offset
        // 当前asset kdim == vHeadDim
        int64_t offsetK = static_cast<int64_t>(mOffset) * inputStride;
        // v_work workspace 按 vBlockDim 紧凑排列，行偏移不带 vHeadDim
        int64_t offsetWorkspace = static_cast<int64_t>(mOffset) * nvActual;

        AscendC::ResetMask();

        AscendC::GlobalTensor<VElementOutput> vnewOutputThisSubBlock = vnewOutput[offsetK];
        AscendC::GlobalTensor<VElementOutput> vnewdecayOutputThisSubBlock = vnewdecayOutput[offsetWorkspace];
        AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;
        AscendC::GlobalTensor<UElementInput> uInputThisSubBlock = uInput[offsetK];

        uint32_t pingpongFlag = isPing ? 0 : pongBaseEvent;
        AscendC::LocalTensor<UElementInput> uUbTensor = isPing ? uUbTensor_ping : uUbTensor_pong;
        AscendC::LocalTensor<float> wsUbTensor = isPing ? wsUbTensor_ping : wsUbTensor_pong;
        AscendC::LocalTensor<float> gUbTensor = isPing ? gUbTensor_ping : gUbTensor_pong;
        AscendC::LocalTensor<float> gLastUbTensor = isPing ? gLastUbTensor_ping : gLastUbTensor_pong;
        AscendC::LocalTensor<GElementInput> gInputUbTensor = isPing ? gInputUbTensor_ping : gInputUbTensor_pong;
        AscendC::LocalTensor<VElementOutput> vNewOutputUbTensor = isPing ? vNewOutputUbTensor_ping : vNewOutputUbTensor_pong;
        AscendC::LocalTensor<VElementOutput> vNewDecayUbTensor = isPing ? vNewDecayUbTensor_ping : vNewDecayUbTensor_pong;


        // u 与 g 的载入互不依赖：先一起下发，再分别等完成，避免两次 DMA 延迟串行。
        // u 独占 PING/PONG_BUF_1，不再与 v_new 输出缓冲同址，载入不必等上一 chunk 的搬出。
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag); // u 缓冲可复用
        CopyGmToUb(uUbTensor, uInputThisSubBlock, mActualThisSubBlock, nvActual, inputStride); // mte2 u
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pingpongFlag); // wait last g
        if constexpr(std::is_same<GElementInput, float>::value) {
            AscendC::DataCopyParams gUbParams{1, (uint16_t)(mActual * sizeof(float)), 0, 0};
            AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(gUbTensor, gInputThisSubBlock, gUbParams, gUbPadParams); // copy g
        } else {
            AscendC::DataCopyParams gUbParams{1, (uint16_t)(mActual * sizeof(half)), 0, 0};
            AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(gInputUbTensor, gInputThisSubBlock, gUbParams, gUbPadParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag); // set u
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag); // wait u
        AscendC::Cast(calcUbTensor, uUbTensor, AscendC::RoundMode::CAST_NONE, elementCount); // cast u
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag); // u 读过，下次可覆盖
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3 + pingpongFlag);
        if constexpr(!std::is_same<GElementInput, float>::value) {
            AscendC::Cast(gUbTensor, gInputUbTensor, AscendC::RoundMode::CAST_NONE, mActual);
        }

        // g 的最后一个元素用寄存器广播展开，省掉 V→S→V 的标量往返。
        BroadcastLastElement(
            (__ubuf__ float*)gLastUbTensor.GetPhyAddr(),
            (__ubuf__ float*)gUbTensor[mActual - 1].GetPhyAddr(), mActual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sub<float>(gUbTensor, gLastUbTensor, gUbTensor, mActual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Exp(gUbTensor, gUbTensor, mActual);
        AscendC::PipeBarrier<PIPE_V>();

        Arch::CrossCoreWaitFlag(cube1Done);

        if (storeFinalState && isInitialState && std::is_same<FinalStateElement, float>::value) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pingpongFlag);
        }

        // v_work 由 Cube C1 直接写进 UB（与 wsUbTensor 同址），不再从 GM 读回。
        // v_new = u - v_work，v_new_decay = (u - v_work) * exp(g_last - g)。
        constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() * 2 / sizeof(float);
        bool useSimdVf = (nvActual % oneRepeatSize) == 0;
        if (useSimdVf) {
            Vec1CalcVF(
                (__ubuf__ VElementOutput*)vNewDecayUbTensor.GetPhyAddr(), (__ubuf__ VElementOutput*)vNewOutputUbTensor.GetPhyAddr(),
                (__ubuf__ float*)wsUbTensor.GetPhyAddr(), (__ubuf__ float*)calcUbTensor.GetPhyAddr(),
                (__ubuf__ float*)gUbTensor[mOffset].GetPhyAddr(),
                mActualThisSubBlock, nvActual);
        } else {
            AscendC::Sub<float>(wsUbTensor, calcUbTensor, wsUbTensor, elementCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Copy(calcUbTensor, wsUbTensor, elementCount);
            AscendC::PipeBarrier<PIPE_V>();
            ApplyRowScale(calcUbTensor, gUbTensor, mOffset, mActualThisSubBlock, nvActual, pingpongFlag);
            AscendC::Cast(vNewDecayUbTensor, calcUbTensor, AscendC::RoundMode::CAST_RINT, elementCount);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pingpongFlag);

        // v_work workspace 按 vBlockDim 紧凑排列，行跨度就是 nvActual。
        CopyUbToGm(vnewdecayOutputThisSubBlock, vNewDecayUbTensor, mActualThisSubBlock, nvActual, nvActual);

        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vec1Done);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1 + pingpongFlag);
        // v_new 输出缓冲在本次调用末尾才写；这里才等上一 chunk 的搬出完成，
        // 让开头两次载入不必等它。
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pingpongFlag);
        if (useSimdVf) {
            Vec1PostVF(
                (__ubuf__ VElementOutput*)vNewOutputUbTensor.GetPhyAddr(),
                (__ubuf__ float*)wsUbTensor.GetPhyAddr(), (__ubuf__ float*)calcUbTensor.GetPhyAddr(),
                mActualThisSubBlock, nvActual);
        } else {
            AscendC::Cast(vNewOutputUbTensor, wsUbTensor, AscendC::RoundMode::CAST_RINT, elementCount);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
        CopyUbToGm(vnewOutputThisSubBlock, vNewOutputUbTensor, mActualThisSubBlock, nvActual, inputStride);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pingpongFlag);

    }

private:
    uint32_t pongBaseEvent = 4;

    AscendC::LocalTensor<float> calcUbTensor;

    AscendC::LocalTensor<UElementInput> uUbTensor_ping;
    AscendC::LocalTensor<float> wsUbTensor_ping;
    AscendC::LocalTensor<float> gUbTensor_ping;
    AscendC::LocalTensor<float> gLastUbTensor_ping;
    AscendC::LocalTensor<GElementInput> gInputUbTensor_ping;
    AscendC::LocalTensor<VElementOutput> vNewOutputUbTensor_ping;
    AscendC::LocalTensor<VElementOutput> vNewDecayUbTensor_ping;

    AscendC::LocalTensor<UElementInput> uUbTensor_pong;
    AscendC::LocalTensor<float> wsUbTensor_pong;
    AscendC::LocalTensor<float> gUbTensor_pong;
    AscendC::LocalTensor<float> gLastUbTensor_pong;
    AscendC::LocalTensor<GElementInput> gInputUbTensor_pong;
    AscendC::LocalTensor<VElementOutput> vNewOutputUbTensor_pong;
    AscendC::LocalTensor<VElementOutput> vNewDecayUbTensor_pong;

    AscendC::LocalTensor<uint8_t> shareBuffer_;

};
}

#endif