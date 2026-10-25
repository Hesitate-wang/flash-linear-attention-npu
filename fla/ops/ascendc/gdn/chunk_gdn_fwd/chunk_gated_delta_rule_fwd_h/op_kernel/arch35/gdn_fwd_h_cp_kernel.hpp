/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

#ifndef GDN_FWD_H_CP_KERNEL_HPP
#define GDN_FWD_H_CP_KERNEL_HPP

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include <type_traits>
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "kernel_operator.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "../chunk_gated_delta_rule_fwd_h_struct.h"

namespace GDN {

static constexpr uint16_t kCpAicSummaryAllDoneFlag = 6;
static constexpr uint16_t kCpAicStateReadyFlag = 7;
static constexpr uint16_t kCpAivOperandReadyFlag = 8;
static constexpr uint16_t kCpAicOperandConsumedFlag = 10;
static constexpr float kCpLn2 = 0.6931471805599453f;

template <typename T>
__aicore__ inline float CpToFloat(T value)
{
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        return AscendC::ToFloat(value);
    } else {
        return static_cast<float>(value);
    }
}

template <typename DataT, typename GateT, typename StateT>
class GDNFwdHCPKernel {
    using CpL1TileShape = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<128>>;
    using CpL0TileShape = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<32>>;
    using CpArchTag = Catlass::Arch::Ascend950;
    using CpDispatch = Catlass::Gemm::MmadPingpongTlaMulti<CpArchTag, false, false, 2, false, 2, 2, 1, 1>;
    using CpTileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
        CpArchTag, float, Catlass::layout::RowMajor, float, Catlass::layout::RowMajor,
        float, Catlass::layout::RowMajor>;
    using CpBlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        CpDispatch, CpL1TileShape, CpL0TileShape,
        float, float, float, void, CpTileCopy>;

public:
    __aicore__ inline void Init(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initialState,
                                GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR tiling, GM_ADDR workspace)
    {
        auto *t = reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingDataRaw *>(tiling);
        batch_ = t->shapeBatch;
        seqlen_ = t->seqlen;
        hk_ = t->kNumHead;
        hv_ = t->vNumHead;
        ranks_ = t->cpRankCount;
        chunks_ = (seqlen_ + 31) / 32;
        useG_ = t->useG;
        useGk_ = t->useGk;
        useInitial_ = t->useInitialState;
        storeFinal_ = t->storeFinalState;

        k_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(k));
        w_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(w));
        u_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(u));
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(g));
        gk_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(gk));
        initial_.SetGlobalBuffer(reinterpret_cast<__gm__ StateT *>(initialState));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(h));
        vNew_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(vNew));
        final_.SetGlobalBuffer(reinterpret_cast<__gm__ StateT *>(finalState));

        uint64_t rankId = AscendC::GetBlockIdx();
        if ASCEND_IS_AIV {
            rankId /= AscendC::GetSubBlockNum();
        }
        activeRankCount_ = static_cast<uint64_t>(batch_) * hv_ * ranks_;
        active_ = rankId < activeRankCount_;
        if (!active_) {
            return;
        }
        const uint64_t chainId = rankId / ranks_;
        rank_ = rankId % ranks_;
        batchIdx_ = chainId / hv_;
        hvIdx_ = chainId % hv_;
        const int64_t base = chunks_ / ranks_;
        const int64_t rem = chunks_ % ranks_;
        chunkBegin_ = rank_ * base + (rank_ < rem ? rank_ : rem);
        chunkCount_ = base + (rank_ < rem ? 1 : 0);
        khIdx_ = hvIdx_ / (hv_ / hk_);

        const uint64_t chainRank = chainId * ranks_ + rank_;
        const uint64_t summaryOffset = static_cast<uint64_t>(t->cpSummaryWorkspaceOffset) / sizeof(float);
        const uint64_t scratchOffset = static_cast<uint64_t>(t->cpScratchWorkspaceOffset) / sizeof(float);
        const uint64_t matrixElems = 128U * 128U;
        const uint64_t stateElems = 128U * 128U;
        chainSummaryBase_ = summaryOffset + chainId * ranks_ * (matrixElems + stateElems);
        summaryBase_ = chainSummaryBase_ + rank_ * (matrixElems + stateElems);
        scratchBase_ = scratchOffset + chainRank * (3U * matrixElems);
        summary_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
        scratch_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
        operand_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));

        if ASCEND_IS_AIV {
            pipe_.InitBuffer(vecBuf_, 48 * 1024);
            mte2ToV_ = pipe_.AllocEventID<AscendC::HardEvent::MTE2_V>();
            vToMte3_ = pipe_.AllocEventID<AscendC::HardEvent::V_MTE3>();
            mte3ToV_ = pipe_.AllocEventID<AscendC::HardEvent::MTE3_V>();
            vToS_ = pipe_.AllocEventID<AscendC::HardEvent::V_S>();
        }

        summaryM_ = summaryBase_;
        summaryE_ = summaryBase_ + matrixElems;
        tempA_ = scratchBase_;
        tempM_ = scratchBase_ + matrixElems;
        operandBase_ = scratchBase_ + 2U * matrixElems;
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            if (active_) {
                BuildLocalSummary();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(kCpAicSummaryAllDoneFlag);
            AscendC::CrossCoreWaitFlag<0x0, PIPE_FIX>(kCpAicSummaryAllDoneFlag);
            if (active_) {
                BuildIncomingState();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kCpAicStateReadyFlag);
        }

        if ASCEND_IS_AIV {
            if (active_) {
                for (int64_t localChunk = 0; localChunk < chunkCount_; ++localChunk) {
                    PrepareSummaryOperands(chunkBegin_ + localChunk);
                    const uint16_t subBlock = AscendC::GetSubBlockIdx();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(kCpAivOperandReadyFlag + subBlock);
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(kCpAicOperandConsumedFlag + subBlock);
                }
            }
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(kCpAicStateReadyFlag);
            if (active_) {
                const uint32_t subBlock = AscendC::GetSubBlockIdx();
                const uint32_t subBlockCount = AscendC::GetSubBlockNum();
                const int64_t valuesPerSubBlock = 128 / subBlockCount;
                const int64_t valueBegin = subBlock * valuesPerSubBlock;
                RunLocalChunks(valueBegin, valuesPerSubBlock);
            }
            if (active_) {
                pipe_.ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2ToV_);
                pipe_.ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3_);
                pipe_.ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToV_);
                pipe_.ReleaseEventID<AscendC::HardEvent::V_S>(vToS_);
            }
        }
    }

private:
    __aicore__ inline uint64_t KOffset(int64_t token, int64_t dim) const
    {
        return ((static_cast<uint64_t>(batchIdx_ * hk_ + khIdx_) * seqlen_ + token) * 128 + dim);
    }

    __aicore__ inline uint64_t HvOffset(int64_t token, int64_t dim, int64_t headDim) const
    {
        return ((static_cast<uint64_t>(batchIdx_ * hv_ + hvIdx_) * seqlen_ + token) * headDim + dim);
    }

    __aicore__ inline uint64_t OutputHOffset(int64_t chunk, int64_t row, int64_t col) const
    {
        return ((((static_cast<uint64_t>(batchIdx_) * chunks_ + chunk) * hv_ + hvIdx_) * 128 + row) * 128 + col);
    }

    __aicore__ inline uint64_t StateOffset(int64_t row, int64_t col) const
    {
        return (static_cast<uint64_t>(batchIdx_) * hv_ + hvIdx_) * 16384 + row * 128 + col;
    }

    __aicore__ inline float ScalarDecay(int64_t tokenLast, int64_t key) const
    {
        (void)tokenLast;
        return operand_.GetValue(operandBase_ + 64U * 128U + key);
    }

    __aicore__ inline float ExpScalar(float value)
    {
        auto scalar = vecBuf_.GetWithOffset<float>(1, 35072);
        AscendC::Duplicate(scalar, value, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(scalar, scalar, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(vToS_);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(vToS_);
        return scalar.GetValue(0);
    }

    __aicore__ inline AscendC::LocalTensor<float> PrepareDecay(int64_t tokenLast)
    {
        constexpr uint32_t kStateBytes = 128U * 64U * sizeof(float);
        auto decayUb = vecBuf_.GetWithOffset<float>(128, kStateBytes);
        if (useGk_) {
            if constexpr (std::is_same<GateT, float>::value) {
                AscendC::DataCopy(decayUb, gk_[HvOffset(tokenLast, 0, 128)], 128);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
            } else {
                auto gateUb = vecBuf_.GetWithOffset<GateT>(128, kStateBytes + 512);
                AscendC::DataCopy(gateUb, gk_[HvOffset(tokenLast, 0, 128)], 128);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
                AscendC::Cast(decayUb, gateUb, AscendC::RoundMode::CAST_NONE, 128);
            }
            AscendC::Muls(decayUb, decayUb, kCpLn2, 128);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(decayUb, decayUb, 128);
        } else {
            AscendC::Duplicate(decayUb, 1.0f, 128);
        }
        if (useG_) {
            const uint64_t gOffset = (static_cast<uint64_t>(batchIdx_) * hv_ + hvIdx_) * seqlen_ + tokenLast;
            const float gate = CpToFloat<GateT>(g_.GetValue(gOffset));
            const float scale = ExpScalar(gate);
            AscendC::Muls(decayUb, decayUb, scale, 128);
        }
        AscendC::PipeBarrier<PIPE_V>();
        return decayUb;
    }

    __aicore__ inline float VFactor(int64_t token, int64_t tokenLast)
    {
        if (!useG_) {
            return 1.0f;
        }
        const uint64_t base = (static_cast<uint64_t>(batchIdx_) * hv_ + hvIdx_) * seqlen_;
        const float gLast = CpToFloat<GateT>(g_.GetValue(base + tokenLast));
        const float gNow = CpToFloat<GateT>(g_.GetValue(base + token));
        return ExpScalar(gLast - gNow);
    }

    __aicore__ inline void BuildLocalSummary()
    {
        constexpr int64_t K = 128;
        constexpr int64_t V = 128;
        for (int64_t i = 0; i < K * K; ++i) {
            summary_.SetValue(summaryM_ + i, (i / K == i % K) ? 1.0f : 0.0f);
            scratch_.SetValue(tempA_ + i, 0.0f);
            scratch_.SetValue(tempM_ + i, 0.0f);
        }
        for (int64_t i = 0; i < K * V; ++i) {
            summary_.SetValue(summaryE_ + i, 0.0f);
        }

        for (int64_t localChunk = 0; localChunk < chunkCount_; ++localChunk) {
            const int64_t chunk = chunkBegin_ + localChunk;
            const int64_t tokenBegin = chunk * 32;
            const int64_t tokenEnd = (tokenBegin + 32 < seqlen_) ? tokenBegin + 32 : seqlen_;
            const int64_t tokenLast = tokenEnd - 1;
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(kCpAivOperandReadyFlag);
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(kCpAivOperandReadyFlag + 1);
            for (int64_t i = 0; i < K; ++i) {
                const float d = ScalarDecay(tokenLast, i);
                for (int64_t j = 0; j < K; ++j) {
                    float a = i == j ? d : 0.0f;
                    for (int64_t token = tokenBegin; token < tokenEnd; ++token) {
                        const float kval = CpToFloat<DataT>(k_.GetValue(KOffset(token, i)));
                        const int64_t row = token - tokenBegin;
                        const uint64_t wOperand = operandBase_ + row * K + j;
                        const float wval = operand_.GetValue(wOperand);
                        a -= kval * wval;
                    }
                    scratch_.SetValue(tempA_ + i * K + j, a);
                }
            }

            CpMmad(scratch_[tempA_], summary_[summaryM_], scratch_[tempM_]);
            for (int64_t i = 0; i < K * K; ++i) {
                summary_.SetValue(summaryM_ + i, scratch_.GetValue(tempM_ + i));
            }

            CpMmad(scratch_[tempA_], summary_[summaryE_], scratch_[tempM_]);
            for (int64_t v = 0; v < V; ++v) {
                for (int64_t i = 0; i < K; ++i) {
                    float value = 0.0f;
                    for (int64_t token = tokenBegin; token < tokenEnd; ++token) {
                        const int64_t row = token - tokenBegin;
                        const uint64_t uOperand = operandBase_ + 32U * K + row * V + v;
                        value += CpToFloat<DataT>(k_.GetValue(KOffset(token, i))) * operand_.GetValue(uOperand);
                    }
                    summary_.SetValue(summaryE_ + i * V + v,
                                      scratch_.GetValue(tempM_ + i * V + v) + value);
                }
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kCpAicOperandConsumedFlag);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kCpAicOperandConsumedFlag + 1);
        }
    }

    __aicore__ inline void BuildIncomingState()
    {
        constexpr int64_t K = 128;
        constexpr int64_t V = 128;
        uint64_t stateA = tempA_;
        uint64_t stateB = tempM_;
        for (int64_t i = 0; i < K * V; ++i) {
            const float init = useInitial_ ? CpToFloat<StateT>(initial_.GetValue(StateOffset(i / V, i % V))) : 0.0f;
            scratch_.SetValue(stateA + i, init);
        }
        for (int64_t prior = 0; prior < rank_; ++prior) {
            const uint64_t record = chainSummaryBase_ + static_cast<uint64_t>(prior) * 32768;
            CpMmad(summary_[record], scratch_[stateA], scratch_[stateB]);
            for (int64_t i = 0; i < K * V; ++i) {
                scratch_.SetValue(stateB + i,
                                  scratch_.GetValue(stateB + i) + summary_.GetValue(record + 16384 + i));
            }
            const uint64_t swap = stateA;
            stateA = stateB;
            stateB = swap;
        }
        for (int64_t i = 0; i < K * V; ++i) {
            scratch_.SetValue(tempA_ + i, scratch_.GetValue(stateA + i));
        }
    }

    __aicore__ inline void PrepareSummaryOperands(int64_t chunk)
    {
        constexpr int64_t K = 128;
        constexpr int64_t V = 128;
        constexpr int64_t columnsPerSubBlock = 64;
        const int64_t subBlock = AscendC::GetSubBlockIdx();
        const int64_t colBegin = subBlock * columnsPerSubBlock;
        const int64_t tokenBegin = chunk * 32;
        const int64_t tokenEnd = (tokenBegin + 32 < seqlen_) ? tokenBegin + 32 : seqlen_;
        const int64_t tokenLast = tokenEnd - 1;
        constexpr uint32_t kStateBytes = 128U * 64U * sizeof(float);
        auto inputUb = vecBuf_.GetWithOffset<float>(64, kStateBytes + 512);
        auto inputDataUb = vecBuf_.GetWithOffset<DataT>(64, kStateBytes + 2816);
        auto decayUb = PrepareDecay(tokenLast);
        if (subBlock == 0) {
            AscendC::DataCopy(operand_[operandBase_ + 64U * 128U], decayUb, 128);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
        for (int64_t token = tokenBegin; token < tokenEnd; ++token) {
            const int64_t row = token - tokenBegin;
            const float factor = VFactor(token, tokenLast);
            const uint64_t wOffset = HvOffset(token, colBegin, K);
            const uint64_t uOffset = HvOffset(token, colBegin, V);
            AscendC::DataCopy(inputDataUb, w_[wOffset], columnsPerSubBlock);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
            AscendC::Cast(inputUb, inputDataUb, AscendC::RoundMode::CAST_NONE, columnsPerSubBlock);
            AscendC::Muls(inputUb, inputUb, factor, columnsPerSubBlock);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
            AscendC::DataCopy(operand_[operandBase_ + row * K + colBegin], inputUb, columnsPerSubBlock);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);

            AscendC::DataCopy(inputDataUb, u_[uOffset], columnsPerSubBlock);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
            AscendC::Cast(inputUb, inputDataUb, AscendC::RoundMode::CAST_NONE, columnsPerSubBlock);
            AscendC::Muls(inputUb, inputUb, factor, columnsPerSubBlock);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
            AscendC::DataCopy(operand_[operandBase_ + 32U * V + row * V + colBegin], inputUb,
                              columnsPerSubBlock);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
    }

    __aicore__ inline void CpMmad(AscendC::GlobalTensor<float> lhs, AscendC::GlobalTensor<float> rhs,
                                  AscendC::GlobalTensor<float> dst)
    {
        auto layout = tla::MakeLayout<float, Catlass::layout::RowMajor>(128, 128);
        auto lhsTensor = tla::MakeTensor(lhs, layout, Catlass::Arch::PositionGM{});
        auto rhsTensor = tla::MakeTensor(rhs, layout, Catlass::Arch::PositionGM{});
        auto dstTensor = tla::MakeTensor(dst, layout, Catlass::Arch::PositionGM{});
        auto lhsTile = tla::GetTile(lhsTensor, tla::MakeCoord(0, 0), tla::MakeShape(128, 128));
        auto rhsTile = tla::GetTile(rhsTensor, tla::MakeCoord(0, 0), tla::MakeShape(128, 128));
        auto dstTile = tla::GetTile(dstTensor, tla::MakeCoord(0, 0), tla::MakeShape(128, 128));
        CpBlockMmad blockMmad(resource_);
        blockMmad.preSetFlags();
        blockMmad(lhsTile, rhsTile, dstTile, Catlass::GemmCoord{128, 128, 128});
        blockMmad.finalWaitFlags();
    }

    __aicore__ inline void RunLocalChunks(int64_t valueBegin, int64_t valueCount)
    {
        constexpr int64_t K = 128;
        constexpr int64_t V = 128;
        constexpr uint32_t kStateBytes = K * 64 * sizeof(float);
        auto stateUb = vecBuf_.Get<float>();
        auto accumUb = vecBuf_.GetWithOffset<float>(64, kStateBytes);
        auto tempUb = vecBuf_.GetWithOffset<float>(64, kStateBytes + 256);
        auto inputUb = vecBuf_.GetWithOffset<float>(64, kStateBytes + 512);
        auto weightUb = vecBuf_.GetWithOffset<float>(K, kStateBytes + 768);
        auto keyUb = vecBuf_.GetWithOffset<float>(K, kStateBytes + 1280);
        auto outputUb = vecBuf_.GetWithOffset<DataT>(64, kStateBytes + 1792);
        auto finalUb = vecBuf_.GetWithOffset<StateT>(64, kStateBytes + 1920);
        auto weightInputUb = vecBuf_.GetWithOffset<DataT>(K, kStateBytes + 2304);
        auto keyInputUb = vecBuf_.GetWithOffset<DataT>(K, kStateBytes + 2560);
        auto inputDataUb = vecBuf_.GetWithOffset<DataT>(64, kStateBytes + 2816);
        const uint64_t stateCol = static_cast<uint64_t>(valueBegin);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
        for (int64_t row = 0; row < K; ++row) {
            AscendC::DataCopy(stateUb[row * valueCount], scratch_[tempA_ + row * V + stateCol], valueCount);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);

        for (int64_t localChunk = 0; localChunk < chunkCount_; ++localChunk) {
            const int64_t chunk = chunkBegin_ + localChunk;
            const int64_t tokenBegin = chunk * 32;
            const int64_t tokenEnd = (tokenBegin + 32 < seqlen_) ? tokenBegin + 32 : seqlen_;
            const int64_t tokenLast = tokenEnd - 1;
            for (int64_t row = 0; row < K; ++row) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
                AscendC::Cast(outputUb, stateUb[row * valueCount], AscendC::RoundMode::CAST_RINT, valueCount);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
                AscendC::DataCopy(h_[OutputHOffset(chunk, row, valueBegin)], outputUb, valueCount);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
            }
            auto decayUb = PrepareDecay(tokenLast);
            for (int64_t row = 0; row < K; ++row) {
                const float decay = decayUb.GetValue(row);
                AscendC::Muls(stateUb[row * valueCount], stateUb[row * valueCount], decay, valueCount);
                AscendC::PipeBarrier<PIPE_V>();
            }
            for (int64_t token = tokenBegin; token < tokenEnd; ++token) {
                const uint64_t hvToken = HvOffset(token, 0, V);
                AscendC::DataCopy(weightInputUb, w_[hvToken], K);
                AscendC::DataCopy(keyInputUb, k_[KOffset(token, 0)], K);
                AscendC::DataCopy(inputDataUb, u_[hvToken + stateCol], valueCount);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
                AscendC::Cast(weightUb, weightInputUb, AscendC::RoundMode::CAST_NONE, K);
                AscendC::Cast(keyUb, keyInputUb, AscendC::RoundMode::CAST_NONE, K);
                AscendC::Cast(inputUb, inputDataUb, AscendC::RoundMode::CAST_NONE, valueCount);
                AscendC::Duplicate(accumUb, 0.0f, valueCount);
                AscendC::PipeBarrier<PIPE_V>();
                for (int64_t row = 0; row < K; ++row) {
                    const float weight = weightUb.GetValue(row);
                    AscendC::Muls(tempUb, stateUb[row * valueCount], -weight, valueCount);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add<float>(accumUb, accumUb, tempUb, valueCount);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Sub<float>(tempUb, inputUb, accumUb, valueCount);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
                AscendC::Cast(outputUb, tempUb, AscendC::RoundMode::CAST_RINT, valueCount);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
                const uint64_t outputOffset =
                    (static_cast<uint64_t>(batchIdx_) * hv_ + hvIdx_) * seqlen_ * V + token * V + stateCol;
                AscendC::DataCopy(vNew_[outputOffset], outputUb, valueCount);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);

                AscendC::Cast(inputUb, outputUb, AscendC::RoundMode::CAST_NONE, valueCount);
                AscendC::PipeBarrier<PIPE_V>();
                const float factor = VFactor(token, tokenLast);
                AscendC::Muls(inputUb, inputUb, factor, valueCount);
                AscendC::PipeBarrier<PIPE_V>();
                for (int64_t row = 0; row < K; ++row) {
                    const float key = keyUb.GetValue(row);
                    AscendC::Muls(tempUb, inputUb, key, valueCount);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add<float>(stateUb[row * valueCount], stateUb[row * valueCount], tempUb, valueCount);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
        }
        if (storeFinal_ && rank_ == ranks_ - 1) {
            for (int64_t row = 0; row < K; ++row) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
                AscendC::Cast(finalUb, stateUb[row * valueCount], AscendC::RoundMode::CAST_RINT, valueCount);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
                AscendC::DataCopy(final_[StateOffset(row, valueBegin)], finalUb, valueCount);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
    }

    int64_t batch_ = 0;
    int64_t seqlen_ = 0;
    int64_t hk_ = 0;
    int64_t hv_ = 0;
    int64_t ranks_ = 0;
    int64_t chunks_ = 0;
    int64_t batchIdx_ = 0;
    int64_t hvIdx_ = 0;
    int64_t khIdx_ = 0;
    int64_t rank_ = 0;
    int64_t chunkBegin_ = 0;
    int64_t chunkCount_ = 0;
    bool useG_ = false;
    bool useGk_ = false;
    bool useInitial_ = false;
    bool storeFinal_ = false;
    bool active_ = false;
    uint64_t activeRankCount_ = 0;
    uint64_t summaryBase_ = 0;
    uint64_t chainSummaryBase_ = 0;
    uint64_t scratchBase_ = 0;
    uint64_t summaryM_ = 0;
    uint64_t summaryE_ = 0;
    uint64_t tempA_ = 0;
    uint64_t tempM_ = 0;
    uint64_t operandBase_ = 0;
    AscendC::GlobalTensor<DataT> k_, w_, u_, h_, vNew_;
    AscendC::GlobalTensor<GateT> g_, gk_;
    AscendC::GlobalTensor<StateT> initial_, final_;
    AscendC::GlobalTensor<float> summary_, scratch_;
    AscendC::GlobalTensor<float> operand_;
    Catlass::Arch::Resource<CpArchTag> resource_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> vecBuf_;
    AscendC::TEventID mte2ToV_;
    AscendC::TEventID vToMte3_;
    AscendC::TEventID mte3ToV_;
    AscendC::TEventID vToS_;
};

template <typename DataT, typename GateT, typename StateT>
__aicore__ inline void ChunkGatedDeltaRuleFwdHCPImpl(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initialState,
    GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR tiling, GM_ADDR workspace)
{
    GDNFwdHCPKernel<DataT, GateT, StateT> kernel;
    kernel.Init(k, w, u, g, gk, initialState, h, vNew, finalState, tiling, workspace);
    kernel.Process();
}

} // namespace GDN

#endif
