/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_CHUNK64_SPARSE_PRODUCER_H
#define GDN_CHUNK64_SPARSE_PRODUCER_H

#include "sparse_common.h"
#include "sparse_cube.h"
#include "sparse_memory.h"
#include "sparse_sync.h"
#include "sparse_vector.h"

namespace GdnChunk64Sparse {

constexpr uint16_t kProdInputs = 0;
constexpr uint16_t kProdKkt = 1;
constexpr uint16_t kProdQk = 2;
constexpr uint16_t kProdA = 3;
constexpr uint16_t kProdWu = 4;

class ProducerRole {
public:
    __aicore__ inline void Init(const PhaseArgs &args, const FusedTilingData &td)
    {
        td_ = td;
        sub_ = GetSubBlockIdx();
        gmQ_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.q));
        gmK_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.k));
        gmV_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.v));
        gmG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.rawG));
        gmBeta_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.beta));
        gmGOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.gCumsum));
        gmBetaEff_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.betaEff));
        gmW_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.w));
        gmU_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.u));
        if (td.useNorm != 0) {
            gmQHat_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.qHat));
            gmKHat_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.kHat));
            gmQRstd_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.qRstd));
            gmKRstd_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.kRstd));
        }
        if (td.outputA != 0) {
            gmA_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.a));
        }
        if (td.hasALog != 0) {
            gmALog_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.aLog));
        }
        if (td.hasDtBias != 0) {
            gmDtBias_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.dtBias));
        }
        gmAPrime_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            GetUserWorkspace(args.workspace) + td.workspaceOffset));
        InitBuffers();
        if ASCEND_IS_AIV {
            GenerateScanIndices(ubShift_);
        }
    }

    __aicore__ inline void BeginChain(const PhaseArgs &args, uint32_t chain)
    {
        loc_ = ResolveChain(td_, args.cuSeqlens, chain);
    }

    __aicore__ inline int64_t ChunkCount() const
    {
        return loc_.chunks;
    }

    __aicore__ inline void RunChunk(int64_t chunk)
    {
        window_ = ResolveWindow(loc_, chunk);
        if ASCEND_IS_AIV {
            Prepare(chunk);
            if (sub_ == 0) {
                AivWait(kProdKkt);
                BuildA(chunk);
            }
            AivPublish(kProdA);
            AivWait(kProdQk);
            PublishAPrime(chunk);
            AivWaitMte3(kProdWu);
            DrainWu();
        }
        if ASCEND_IS_AIC {
            RunCube();
        }
    }

private:
    __aicore__ inline void InitBuffers()
    {
        OnChipBuffer buf;
        ubQ_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbQ);
        ubK_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbK);
        ubV_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbV);
        ubQHat_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbQHat);
        ubKHat_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbKHat);
        ubVb_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbVb);
        ubKbg_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbKbg);
        ubW_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbW);
        ubU_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbU);
        ubQk_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbQk);
        ubKkt_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbKkt);
        ubL_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbL);
        ubA_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbA);
        ubABf_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbABf);
        ubAPrime_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbAPrime);
        ubG_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbGates);
        ubEnergy_ = ubG_[64];
        ubBeta_ = ubG_[192];
        ubQRstd_ = ubG_[256];
        ubKRstd_ = ubG_[288];
        ubShift_ = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(kPUbShift);

        l1K_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kPL1K);
        l1Q_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kPL1Q);
        l1Vb_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kPL1Vb);
        l1Kbg_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kPL1Kbg);
        l1A_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kPL1A);
        l0A0_ = buf.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(kL0Bank0);
        l0A1_ = buf.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(kL0Bank1);
        l0B0_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0Bank0);
        l0B1_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0Bank1);
        l0C0_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C0);
        l0C1_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C1);
        l0C2_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C2);
        l0C3_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C3);
    }

    __aicore__ inline void Prepare(int64_t chunk)
    {
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        const int64_t remaining = window_.rows - static_cast<int64_t>(firstRow);
        const uint32_t rows = remaining <= 0 ? 0U
            : static_cast<uint32_t>(remaining < kHalfRows ? remaining : kHalfRows);
        if (rows < static_cast<uint32_t>(kHalfRows)) {
            Duplicate(ubQ_, static_cast<bfloat16_t>(0), kHalfRows * kDim);
            Duplicate(ubK_, static_cast<bfloat16_t>(0), kHalfRows * kDim);
            Duplicate(ubV_, static_cast<bfloat16_t>(0), kHalfRows * kDim);
        }
        if (window_.rows < kChunk) {
            Duplicate(ubG_, 0.0f, kChunk);
            Duplicate(ubBeta_, 0.0f, kChunk);
        }
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        const int64_t token = window_.tokenStart + firstRow;
        const uint32_t elems = rows * static_cast<uint32_t>(kDim);
        CopyGmToUb(ubQ_, gmQ_, OffsetBhtd(loc_.batch, loc_.kh, token, td_.keyHeads, td_.tokens), elems);
        CopyGmToUb(ubK_, gmK_, OffsetBhtd(loc_.batch, loc_.kh, token, td_.keyHeads, td_.tokens), elems);
        CopyGmToUb(ubV_, gmV_, OffsetBhtd(loc_.batch, loc_.hv, token, td_.valueHeads, td_.tokens), elems);
        const int64_t gateOffset = OffsetBht(loc_.batch, loc_.hv, window_.tokenStart,
                                             td_.valueHeads, td_.tokens);
        CopyGmToUb(ubG_, gmG_, gateOffset, static_cast<uint32_t>(window_.rows));
        CopyGmToUb(ubBeta_, gmBeta_, gateOffset, static_cast<uint32_t>(window_.rows));
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
        PrepareGateBetaVF(ubG_, ubBeta_, aLog, dtBias, td_.gateSoftplus != 0,
                          td_.betaSigmoid != 0, td_.allowNegEigval != 0 ? 2.0f : 1.0f);
        if (td_.useExp2 != 0) {
            CumsumEnergyVF<true>(ubG_, ubEnergy_, ubShift_);
        } else {
            CumsumEnergyVF<false>(ubG_, ubEnergy_, ubShift_);
        }
        if (td_.useNorm != 0) {
            NormalizeRowsVF(ubQ_, ubQHat_, ubQRstd_);
            NormalizeRowsVF(ubK_, ubKHat_, ubKRstd_);
        } else {
            DataCopy(ubQHat_, ubQ_, kHalfRows * kDim);
            DataCopy(ubKHat_, ubK_, kHalfRows * kDim);
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
        }
        ScaleVbVF(ubV_, ubVb_, ubBeta_, firstRow);
        ScaleKbgVF(ubKHat_, ubKbg_, ubBeta_, ubEnergy_, firstRow);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        if (td_.useNorm != 0) {
            const int64_t qkOut = OffsetBhtd(loc_.batch, loc_.kh, token, td_.keyHeads, td_.tokens);
            const int64_t rstdOut = OffsetBht(loc_.batch, loc_.kh, token, td_.keyHeads, td_.tokens);
            CopyUbToGm(gmQHat_, qkOut, ubQHat_, elems);
            CopyUbToGm(gmKHat_, qkOut, ubKHat_, elems);
            CopyUbToGm(gmQRstd_, rstdOut, ubQRstd_, rows);
            CopyUbToGm(gmKRstd_, rstdOut, ubKRstd_, rows);
        }
        if (sub_ == 0) {
            CopyUbToGm(gmGOut_, gateOffset, ubG_, static_cast<uint32_t>(window_.rows));
            if (td_.betaSigmoid != 0) {
                CopyUbToGm(gmBetaEff_, gateOffset, ubBeta_, static_cast<uint32_t>(window_.rows));
            }
        }
        UploadNdToL1(l1K_, ubKHat_, kHalfRows, kDim, kChunk, firstRow);
        UploadNdToL1(l1Q_, ubQHat_, kHalfRows, kDim, kChunk, firstRow);
        UploadNdToL1(l1Vb_, ubVb_, kHalfRows, kDim, kChunk, firstRow);
        UploadNdToL1(l1Kbg_, ubKbg_, kHalfRows, kDim, kChunk, firstRow);
        AivPublish(kProdInputs);
    }

    __aicore__ inline void BuildA(int64_t chunk)
    {
        if (td_.useExp2 != 0) {
            Inverse64VF<true>(ubKkt_, ubL_, ubA_, ubG_, ubBeta_);
        } else {
            Inverse64VF<false>(ubKkt_, ubL_, ubA_, ubG_, ubBeta_);
        }
        Cast(ubABf_, ubA_, RoundMode::CAST_RINT, kMatrixElems);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        UploadNdToL1(l1A_, ubABf_, kChunk, kChunk, kChunk, 0);
        if (td_.outputA != 0) {
            CopyUbToGm(gmA_, OffsetBhtd(loc_.batch, loc_.hv, window_.tokenStart,
                                        td_.valueHeads, td_.tokens, kChunk),
                       ubABf_, static_cast<uint32_t>(window_.rows * kChunk));
        }
    }

    __aicore__ inline void PublishAPrime(int64_t chunk)
    {
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        if (td_.useExp2 != 0) {
            APrimeBandVF<true>(ubQk_, ubAPrime_, ubG_, firstRow);
        } else {
            APrimeBandVF<false>(ubQk_, ubAPrime_, ubG_, firstRow);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        const int64_t tile = TileIndex(loc_, chunk, td_.chunksPerBatch, td_.varlen != 0);
        const int64_t offset = (tile * td_.valueHeads + loc_.hv) * kMatrixElems + firstRow * kChunk;
        CopyUbToGm(gmAPrime_, offset, ubAPrime_, kHalfRows * kChunk);
        if (td_.reserved == 1 && td_.outputA != 0) {
            const int64_t remaining = window_.rows - static_cast<int64_t>(firstRow);
            const uint32_t rows = remaining <= 0 ? 0U
                : static_cast<uint32_t>(remaining < kHalfRows ? remaining : kHalfRows);
            CopyUbToGm(gmA_, OffsetBhtd(loc_.batch, loc_.hv, window_.tokenStart + firstRow,
                                        td_.valueHeads, td_.tokens, kChunk),
                       ubAPrime_, rows * static_cast<uint32_t>(kChunk));
        }
    }

    __aicore__ inline void DrainWu()
    {
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        const int64_t remaining = window_.rows - static_cast<int64_t>(firstRow);
        const uint32_t rows = remaining <= 0 ? 0U
            : static_cast<uint32_t>(remaining < kHalfRows ? remaining : kHalfRows);
        const int64_t token = window_.tokenStart + firstRow;
        const int64_t offset = OffsetBhtd(loc_.batch, loc_.hv, token, td_.valueHeads, td_.tokens);
        CopyUbToGm(gmW_, offset, ubW_, rows * static_cast<uint32_t>(kDim));
        CopyUbToGm(gmU_, offset, ubU_, rows * static_cast<uint32_t>(kDim));
    }

    __aicore__ inline void RunCube()
    {
        AicWaitBoth(kProdInputs);
        ComputeMm(l1K_, l1K_, l0A0_, l0B0_, l0C0_, 64, 64, 128, false, false);
        FixF32(ubKkt_, l0C0_, 64, 64);
        ReleaseL0C();
        AicPublishToAiv0(kProdKkt);

        ComputeMm(l1Q_, l1K_, l0A1_, l0B1_, l0C1_, 64, 64, 128, false, false);
        FixF32Halves(ubQk_, l0C1_, 64, 64);
        ReleaseL0C();
        AicPublishToBoth(kProdQk);

        AicWaitBoth(kProdA);
        ComputeMm(l1A_, l1Kbg_, l0A0_, l0B0_, l0C2_, 64, 128, 64, false, true);
        FixB16Halves(ubW_, l0C2_, 64, 128);
        ReleaseL0C();
        ComputeMmReuseA(l1Vb_, l0A0_, l0B1_, l0C3_, 64, 128, 64, true);
        FixB16Halves(ubU_, l0C3_, 64, 128);
        ReleaseL0C();
        AicPublishToBoth(kProdWu);
    }

    FusedTilingData td_{};
    ChainLoc loc_{};
    ChunkWindow window_{};
    uint32_t sub_{0};

    GlobalTensor<bfloat16_t> gmQ_, gmK_, gmV_, gmQHat_, gmKHat_, gmW_, gmU_, gmA_, gmAPrime_;
    GlobalTensor<float> gmG_, gmBeta_, gmGOut_, gmBetaEff_, gmQRstd_, gmKRstd_, gmALog_, gmDtBias_;

    LocalTensor<bfloat16_t> ubQ_, ubK_, ubV_, ubQHat_, ubKHat_, ubVb_, ubKbg_, ubW_, ubU_;
    LocalTensor<bfloat16_t> ubABf_, ubAPrime_;
    LocalTensor<float> ubQk_, ubKkt_, ubL_, ubA_;
    LocalTensor<float> ubG_, ubEnergy_, ubBeta_, ubQRstd_, ubKRstd_;
    LocalTensor<uint32_t> ubShift_;
    LocalTensor<bfloat16_t> l1K_, l1Q_, l1Vb_, l1Kbg_, l1A_;
    LocalTensor<bfloat16_t> l0A0_, l0A1_, l0B0_, l0B1_;
    LocalTensor<float> l0C0_, l0C1_, l0C2_, l0C3_;
};

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_PRODUCER_H
