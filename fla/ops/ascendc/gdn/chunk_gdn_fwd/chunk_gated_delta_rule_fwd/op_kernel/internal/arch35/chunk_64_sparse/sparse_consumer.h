/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_CHUNK64_SPARSE_CONSUMER_H
#define GDN_CHUNK64_SPARSE_CONSUMER_H

#include "sparse_common.h"
#include "sparse_cube.h"
#include "sparse_memory.h"
#include "sparse_sync.h"
#include "sparse_vector.h"

namespace GdnChunk64Sparse {

constexpr uint16_t kConsInputs = 0;
constexpr uint16_t kConsP = 1;
constexpr uint16_t kConsVn = 2;
constexpr uint16_t kConsD = 4;
constexpr uint16_t kConsOL = 5;
constexpr uint16_t kConsQk = 6;

class ConsumerRole {
public:
    __aicore__ inline void Init(const PhaseArgs &args, const FusedTilingData &td)
    {
        td_ = td;
        sub_ = GetSubBlockIdx();
        gmQ_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.q));
        gmK_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.k));
        gmW_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.w));
        gmW_.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
        gmU_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.u));
        gmU_.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
        gmG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.rawG));
        gmH_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.h));
        gmO_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.o));
        if (td.hasALog != 0) {
            gmALog_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.aLog));
        }
        if (td.hasDtBias != 0) {
            gmDtBias_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.dtBias));
        }
        if (args.initialState != nullptr) {
            hasInitial_ = true;
            if (td.stateFp32 != 0) {
                gmInitF_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.initialState));
            } else {
                gmInitB_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.initialState));
            }
        }
        if (args.finalState != nullptr) {
            if (td.stateFp32 != 0) {
                gmFinalF_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.finalState));
            } else {
                gmFinalB_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.finalState));
            }
        }
        InitBuffers();
        if ASCEND_IS_AIV {
            GenerateScanIndices(ubShift_);
        }
    }

    __aicore__ inline void BeginChain(const PhaseArgs &args, uint32_t chain)
    {
        loc_ = ResolveChain(td_, args.cuSeqlens, chain);
        if ASCEND_IS_AIV {
            InitState();
        }
    }

    __aicore__ inline int64_t ChunkCount() const
    {
        return loc_.chunks;
    }

    __aicore__ inline void RunChunk(int64_t chunk)
    {
        window_ = ResolveWindow(loc_, chunk);
        if ASCEND_IS_AIV {
            RunAiv(chunk);
        }
        if ASCEND_IS_AIC {
            RunCube(chunk);
        }
    }

    __aicore__ inline void EndChain()
    {
        if ASCEND_IS_AIV {
            if (td_.storeFinalState != 0) {
                const int64_t offset = StateOffset();
                if (td_.stateFp32 != 0) {
                    F32PanelsToNd(ubD_, ubStateF_);
                    SetFlag<HardEvent::V_MTE3>(0);
                    WaitFlag<HardEvent::V_MTE3>(0);
                    DataCopy(gmFinalF_[offset], ubD_, kStateElems / 2);
                } else {
                    Bf16PanelsToNd(ubInitB_, ubStateB_);
                    SetFlag<HardEvent::V_MTE3>(0);
                    WaitFlag<HardEvent::V_MTE3>(0);
                    DataCopy(gmFinalB_[offset], ubInitB_, kStateElems / 2);
                }
            }
        }
    }

private:
    __aicore__ inline void InitBuffers()
    {
        OnChipBuffer buf;
        ubStateF_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kCUbStateF);
        ubStateB_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbStateB);
        ubD_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kCUbD);
        ubInitB_ = ubD_.ReinterpretCast<bfloat16_t>();
        ubP_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kCUbP);
        ubQRstd_ = ubP_;
        ubKRstd_ = ubP_[32];
        ubOSB_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbOS);
        ubU_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbU);
        ubVNew_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbVNew);
        ubRight_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbRight);
        ubW_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbW);
        ubQ_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbQ);
        ubK_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbK);
        if (td_.useNorm != 0) {
            ubQHat_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbQHat);
            ubKHat_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbKHat);
        } else {
            ubQHat_ = ubQ_;
            ubKHat_ = ubK_;
        }
        // QK overwrites QHat only after QHat has been uploaded to L1.
        ubQk_ = ubQHat_.ReinterpretCast<float>();
        ubAPrime_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbAPrime);
        ubO_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbO);
        ubG_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kCUbGates);
        ubEnergy_ = ubG_[64];
        ubShift_ = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(kCUbShift);
        ubBetaDummy_ = ubG_[128];

        l1W_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kCL1W);
        l1U_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kCL1U);
        l1QHat_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kCL1QHat);
        l1QExp_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kCL1QExp);
        l1K_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kCL1K);
        l1H_[0] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kCL1H0);
        l1H_[1] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kCL1H1);
        l1VNew_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kCL1VNew);
        l1Right_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kCL1Right);
        l1APrime_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kCL1APrime);
        l0A0_ = buf.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(kL0Bank0);
        l0A1_ = buf.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(kL0Bank1);
        l0B0_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0Bank0);
        l0B1_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0Bank1);
        l0C0_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C0);
        l0C1_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C1);
        l0C3_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C3);
    }

    __aicore__ inline int64_t StateOffset() const
    {
        return (loc_.seq * td_.valueHeads + loc_.hv) * kStateElems + sub_ * (kStateElems / 2);
    }

    __aicore__ inline void InitState()
    {
        if (hasInitial_) {
            if (td_.stateFp32 != 0) {
                DataCopy(ubD_, gmInitF_[StateOffset()], kStateElems / 2);
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                F32NdToPanels(ubStateF_, ubD_);
                StateMirrorVF(ubStateF_, ubStateB_);
            } else {
                DataCopy(ubInitB_, gmInitB_[StateOffset()], kStateElems / 2);
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                Bf16NdToPanels(ubStateB_, ubInitB_);
                Cast(ubStateF_, ubStateB_, RoundMode::CAST_NONE, kStateElems / 2);
            }
        } else {
            Duplicate(ubStateF_, 0.0f, kStateElems / 2);
            Duplicate(ubStateB_, static_cast<bfloat16_t>(0), kStateElems / 2);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
    }

    __aicore__ inline void RunAiv(int64_t chunk)
    {
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        const int64_t remaining = window_.rows - static_cast<int64_t>(firstRow);
        const uint32_t rows = remaining <= 0 ? 0U
            : static_cast<uint32_t>(remaining < kHalfRows ? remaining : kHalfRows);
        const int64_t tile = TileIndex(loc_, chunk, td_.chunksPerBatch, td_.varlen != 0);
        if (td_.writeH != 0) {
            Bf16PanelsToNd(ubInitB_, ubStateB_);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopy(gmH_[OffsetStateTile(tile, loc_.hv, td_.valueHeads) + sub_ * (kStateElems / 2)],
                     ubInitB_, kStateElems / 2);
        }
        LoadInputs(chunk, firstRow, rows);
        AivPublish(kConsInputs);

        AivWait(kConsQk);
        if (td_.useExp2 != 0) {
            APrimeBandVF<true>(ubQk_, ubAPrime_, ubG_, ubEnergy_, gateRatioSafe_, firstRow);
        } else {
            APrimeBandVF<false>(ubQk_, ubAPrime_, ubG_, ubEnergy_, gateRatioSafe_, firstRow);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        UploadNdToL1(l1APrime_, ubAPrime_, kHalfRows, kChunk, kChunk, firstRow);

        AivWait(kConsP);
        SetFlag<HardEvent::V_S>(1);
        WaitFlag<HardEvent::V_S>(1);
        const int64_t lastIndex = window_.rows - 1;
        const float gLast = ubG_.GetValue(lastIndex);
        const float lastEnergy = ubEnergy_.GetValue(lastIndex);
        if (td_.useExp2 != 0) {
            VNewRightVF<true>(ubP_, ubU_, ubVNew_, ubRight_, ubG_, ubEnergy_,
                              gLast, lastEnergy, gateRatioSafe_, firstRow);
        } else {
            VNewRightVF<false>(ubP_, ubU_, ubVNew_, ubRight_, ubG_, ubEnergy_,
                               gLast, lastEnergy, gateRatioSafe_, firstRow);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        const int64_t token = window_.tokenStart + firstRow;
        UploadNdToL1(l1VNew_, ubVNew_, kHalfRows, kDim, kChunk, firstRow);
        UploadNdToL1(l1Right_, ubRight_, kHalfRows, kDim, kChunk, firstRow);
        AivPublish(kConsVn);

        if (td_.reserved == 4) {
            AivWait(kConsOL);
            ubP_.SetValue(0, ubEnergy_.GetValue(firstRow));
            ubP_.SetValue(1, td_.scale);
            ubP_.SetValue(2, ubG_.GetValue(firstRow));
            ubP_.SetValue(3, lastEnergy);
            SetFlag<HardEvent::S_V>(0);
            WaitFlag<HardEvent::S_V>(0);
            Cast(ubO_, ubP_, RoundMode::CAST_RINT, 4);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
        } else {
            AivWaitMte3(kConsOL);
        }
        if (rows != 0) {
            DataCopyExtParams params{};
            params.blockCount = static_cast<uint16_t>(rows);
            params.blockLen = kDim * sizeof(bfloat16_t);
            params.srcStride = 0;
            params.dstStride = (td_.valueHeads - 1) * kDim * sizeof(bfloat16_t);
            params.rsv = 0;
            DataCopyPad(gmO_[OffsetOutput(loc_.batch, token, loc_.hv, td_.valueHeads, td_.tokens)],
                        ubO_, params);
        }

        AivWait(kConsD);
        StateUpdateVF(ubStateF_, ubStateB_, ubD_, lastEnergy);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
    }

    __aicore__ inline void LoadInputs(int64_t chunk, uint32_t firstRow, uint32_t rows)
    {
        if (rows < static_cast<uint32_t>(kHalfRows)) {
            Duplicate(ubW_, static_cast<bfloat16_t>(0), kHalfRows * kDim);
            Duplicate(ubU_, static_cast<bfloat16_t>(0), kHalfRows * kDim);
            Duplicate(ubQ_, static_cast<bfloat16_t>(0), kHalfRows * kDim);
            Duplicate(ubK_, static_cast<bfloat16_t>(0), kHalfRows * kDim);
        }
        if (window_.rows < kChunk) {
            Duplicate(ubG_, 0.0f, kChunk);
        }
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        const int64_t token = window_.tokenStart + firstRow;
        const uint32_t elems = rows * static_cast<uint32_t>(kDim);
        CopyGmToUb(ubW_, gmW_, OffsetBhtd(loc_.batch, loc_.hv, token, td_.valueHeads, td_.tokens), elems);
        CopyGmToUb(ubU_, gmU_, OffsetBhtd(loc_.batch, loc_.hv, token, td_.valueHeads, td_.tokens), elems);
        CopyGmToUb(ubQ_, gmQ_, OffsetBhtd(loc_.batch, loc_.kh, token, td_.keyHeads, td_.tokens), elems);
        CopyGmToUb(ubK_, gmK_, OffsetBhtd(loc_.batch, loc_.kh, token, td_.keyHeads, td_.tokens), elems);
        CopyGmToUb(ubG_, gmG_, OffsetBht(loc_.batch, loc_.hv, window_.tokenStart,
                                          td_.valueHeads, td_.tokens),
                   static_cast<uint32_t>(window_.rows));
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        float aLog = 0.0f;
        float dtBias = 0.0f;
        if (td_.hasALog != 0) {
            aLog = gmALog_.GetValue(loc_.hv);
        }
        if (td_.hasDtBias != 0) {
            dtBias = gmDtBias_.GetValue(loc_.hv);
        }
        PrepareGateBetaVF(ubG_, ubBetaDummy_, aLog, dtBias, td_.gateSoftplus != 0, false, 1.0f);
        if (td_.useExp2 != 0) {
            CumsumEnergyVF<true>(ubG_, ubEnergy_, ubShift_);
        } else {
            CumsumEnergyVF<false>(ubG_, ubEnergy_, ubShift_);
        }
        SetFlag<HardEvent::V_S>(1);
        WaitFlag<HardEvent::V_S>(1);
        gateRatioSafe_ = GateRatioSafe(ubG_, static_cast<uint32_t>(window_.rows));
        if (td_.useNorm != 0) {
            NormalizeRowsVF(ubQ_, ubQHat_, ubQRstd_);
            NormalizeRowsVF(ubK_, ubKHat_, ubKRstd_);
        }
        QExpVF(ubQHat_, ubOSB_, ubEnergy_, firstRow);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        UploadNdToL1(l1W_, ubW_, kHalfRows, kDim, kChunk, firstRow);
        UploadNdToL1(l1U_, ubU_, kHalfRows, kDim, kChunk, firstRow);
        UploadNdToL1(l1QHat_, ubQHat_, kHalfRows, kDim, kChunk, firstRow);
        UploadNdToL1(l1QExp_, ubOSB_, kHalfRows, kDim, kChunk, firstRow);
        UploadNdToL1(l1K_, ubKHat_, kHalfRows, kDim, kChunk, firstRow);
        UploadNzStateToL1(l1H_[chunk & 1], ubStateB_,
                          sub_ * static_cast<uint32_t>(kHalfDim));
    }

    __aicore__ inline void RunCube(int64_t chunk)
    {
        AicWaitBoth(kConsInputs);
        ComputeMm(l1QHat_, l1K_, l0A0_, l0B1_, l0C1_, 64, 64, 128, false, false);
        FixF32Halves(ubQk_, l0C1_, 64, 64);
        ReleaseL0C();
        AicPublishToBoth(kConsQk);

        ComputeMm(l1W_, l1H_[chunk & 1], l0A0_, l0B0_, l0C0_, 64, 128, 128,
                  false, td_.stateVFirst == 0);
        FixF32Halves(ubP_, l0C0_, 64, 128);
        ReleaseL0C();
        AicPublishToBoth(kConsP);

        ComputeMmReuseB(l1QExp_, l0A0_, l0B0_, l0C0_, 64, 128, 128, false, true);

        AicWaitBoth(kConsVn);
        ComputeMm(l1APrime_, l1VNew_, l0A1_, l0B1_, l0C0_, 64, 128, 64,
                  false, true, false);
        FixB16ScaledHalves(ubO_, l0C0_, 64, 128, td_.scale);
        ReleaseL0C();
        AicPublishToBoth(kConsOL);

        if (td_.stateVFirst != 0) {
            ComputeMm(l1Right_, l1K_, l0A0_, l0B0_, l0C3_, 128, 128, 64, true, true);
        } else {
            ComputeMm(l1K_, l1Right_, l0A0_, l0B0_, l0C3_, 128, 128, 64, true, true);
        }
        FixF32NzHalves(ubD_, l0C3_, 128, 128);
        ReleaseL0C();
        AicPublishToBoth(kConsD);
    }

    FusedTilingData td_{};
    ChainLoc loc_{};
    ChunkWindow window_{};
    uint32_t sub_{0};
    bool hasInitial_{false};
    bool gateRatioSafe_{false};

    GlobalTensor<bfloat16_t> gmQ_, gmK_, gmW_, gmU_, gmH_, gmO_;
    GlobalTensor<bfloat16_t> gmInitB_, gmFinalB_;
    GlobalTensor<float> gmG_, gmInitF_, gmFinalF_, gmALog_, gmDtBias_;

    LocalTensor<float> ubStateF_, ubD_, ubP_, ubQk_, ubQRstd_, ubKRstd_;
    LocalTensor<bfloat16_t> ubStateB_, ubU_, ubVNew_, ubRight_, ubW_, ubQ_, ubK_;
    LocalTensor<bfloat16_t> ubQHat_, ubKHat_, ubAPrime_, ubO_, ubInitB_, ubOSB_;
    LocalTensor<float> ubG_, ubEnergy_, ubBetaDummy_;
    LocalTensor<uint32_t> ubShift_;
    LocalTensor<bfloat16_t> l1W_, l1U_, l1QHat_, l1QExp_, l1K_, l1H_[2];
    LocalTensor<bfloat16_t> l1VNew_, l1Right_, l1APrime_;
    LocalTensor<bfloat16_t> l0A0_, l0A1_, l0B0_, l0B1_;
    LocalTensor<float> l0C0_, l0C1_, l0C3_;
};

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_CONSUMER_H
