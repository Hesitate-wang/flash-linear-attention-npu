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
constexpr uint16_t kProdA = 3;
constexpr uint16_t kProdW = 4;
constexpr uint16_t kProdInverse = 5;
constexpr uint16_t kProdCoupled = 6;
constexpr uint16_t kProdK = 8;

class ProducerRole {
public:
    __aicore__ inline void Init(const PhaseArgs &args, const FusedTilingData &td)
    {
        td_ = td;
        sub_ = GetSubBlockIdx();
        gmK_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.k));
        gmV_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.v));
        gmG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.rawG));
        gmBeta_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.beta));
        gmGOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.gCumsum));
        hasBetaEffOut_ = args.betaEff != nullptr;
        if (hasBetaEffOut_) {
            gmBetaEff_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.betaEff));
        }
        gmW_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.w));
        gmW_.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
        gmU_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.u));
        gmU_.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
        gmInverse_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            GetUserWorkspace(args.workspace) + td.workspaceOffset));
        gmInverse_.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
        gmGOut_.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
        if (td.useNorm != 0) {
            gmKHat_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.kHat));
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
            PrepareKGate(chunk);
            AivWait(kProdKkt);
            BuildA(chunk);
            AivPublish(kProdA);
            AivWaitMte3(kProdW);
            DrainWu();
        }
        if ASCEND_IS_AIC {
            RunCube(chunk);
        }
    }

private:
    __aicore__ inline void InitBuffers()
    {
        OnChipBuffer buf;
        ubK_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbK);
        ubV_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbV);
        ubKHat_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbKHat);
        ubVb_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbVb);
        ubKbg_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbKbg);
        ubW_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbW);
        ubU_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbU);
        ubKkt_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbKkt);
        ubPublicB16_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbQHat);
        ubL_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbL);
        ubA_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbA);
        ubABf_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbABf);
        ubG_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbGates);
        ubEnergy_ = ubG_[64];
        ubBeta_ = ubG_[192];
        ubKRstd_ = ubG_[288];
        ubShift_ = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(kPUbShift);

        l1K_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kPL1K);
        l1InvX0_ = buf.GetBuffer<BufferType::ASCEND_CB, float>(kPL1InvX0);
        l1InvX1_ = buf.GetBuffer<BufferType::ASCEND_CB, float>(kPL1InvX1);
        l1InvL10_ = buf.GetBuffer<BufferType::ASCEND_CB, float>(kPL1InvL10);
        l1InvT_ = buf.GetBuffer<BufferType::ASCEND_CB, float>(kPL1InvT);
        l1Vb_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kPL1Vb);
        l1Kbg_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kPL1Kbg);
        l1A_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kPL1A);
        l0A0_ = buf.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(kL0Bank0);
        l0A1_ = buf.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(kL0Bank1);
        l0B0_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0Bank0);
        l0B1_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0Bank1);
        l0Af_ = buf.GetBuffer<BufferType::ASCEND_L0A, float>(kL0Bank0);
        l0Bf_ = buf.GetBuffer<BufferType::ASCEND_L0B, float>(kL0Bank0);
        l0C0_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C0);
        l0C2_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C2);
        l0C3_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C3);
    }

    __aicore__ inline void PrepareKGate(int64_t)
    {
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        const int64_t remaining = window_.rows - static_cast<int64_t>(firstRow);
        const uint32_t rows = remaining <= 0 ? 0U
            : static_cast<uint32_t>(remaining < kHalfRows ? remaining : kHalfRows);
        if (rows < static_cast<uint32_t>(kHalfRows)) {
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
        const int64_t gateOffset = OffsetBht(loc_.batch, loc_.hv, window_.tokenStart,
                                             td_.valueHeads, td_.tokens);

        // K is the only Cube input needed before inverse.
        CopyGmToUb(ubK_, gmK_, OffsetBhtd(loc_.batch, loc_.kh, token, td_.keyHeads, td_.tokens), elems);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        if (td_.useNorm != 0) {
            NormalizeRowsNzVF<true, true>(ubK_, ubKHat_, ubPublicB16_, ubKRstd_);
        } else {
            NormalizeRowsNzVF<false, true>(ubK_, ubKHat_, ubPublicB16_, ubKRstd_);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyNzToL1(l1K_, ubKHat_, kHalfRows, kDim, kChunk, firstRow);
        AivPublish(kProdK);

        // Gate/beta are the only remaining inputs of inverse. KKT overlaps
        // this load/wait/compute stage while Q/V stay off the critical path.
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
    }

    __aicore__ inline void PrepareQvAfterInverse(int64_t)
    {
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        const int64_t remaining = window_.rows - static_cast<int64_t>(firstRow);
        const uint32_t rows = remaining <= 0 ? 0U
            : static_cast<uint32_t>(remaining < kHalfRows ? remaining : kHalfRows);
        const int64_t token = window_.tokenStart + firstRow;
        const uint32_t elems = rows * static_cast<uint32_t>(kDim);
        const int64_t gateOffset = OffsetBht(loc_.batch, loc_.hv, window_.tokenStart,
                                             td_.valueHeads, td_.tokens);
        const int64_t qkOut = OffsetBhtd(loc_.batch, loc_.kh, token, td_.keyHeads, td_.tokens);
        const int64_t rstdOut = OffsetBht(loc_.batch, loc_.kh, token, td_.keyHeads, td_.tokens);

        if (td_.useNorm != 0) {
            CopyUbToGm(gmKHat_, qkOut, ubPublicB16_, elems);
            CopyUbToGm(gmKRstd_, rstdOut, ubKRstd_, rows);
        }

        CopyGmToUb(ubV_, gmV_, OffsetBhtd(loc_.batch, loc_.hv, token, td_.valueHeads, td_.tokens), elems);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        ScaleKbgNzVF(ubPublicB16_, ubKbg_, ubBeta_, ubEnergy_, firstRow);
        ScaleVbNzVF(ubV_, ubVb_, ubBeta_, firstRow);
        if (sub_ == 0) {
            CopyUbToGm(gmGOut_, gateOffset, ubG_, static_cast<uint32_t>(window_.rows));
            if (td_.betaSigmoid != 0 && hasBetaEffOut_) {
                CopyUbToGm(gmBetaEff_, gateOffset, ubBeta_, static_cast<uint32_t>(window_.rows));
            }
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyNzToL1(l1Vb_, ubVb_, kHalfRows, kDim, kChunk, firstRow);
        CopyNzToL1(l1Kbg_, ubKbg_, kHalfRows, kDim, kChunk, firstRow);
        AivPublish(kProdInputs);
    }

    __aicore__ inline void PublishInverse(int64_t chunk)
    {
        const int64_t tile = TileIndex(loc_, chunk, td_.chunksPerBatch, td_.varlen != 0);
        const int64_t record = tile * td_.valueHeads + loc_.hv;
        const int64_t base = record * 3 * 1024;
        if (sub_ == 0) {
            CopyUbToGm(gmInverse_, base, ubA_, 1024);
        } else {
            CopyUbToGm(gmInverse_, base + 1024, ubA_, 1024);
            CopyUbToGm(gmInverse_, base + 2048, ubA_[2048], 1024);
        }
        AivPublish(kProdInverse);
    }

    __aicore__ inline void WritePublicABlock(uint32_t rows, uint32_t rowOffset,
                                             uint32_t colOffset, LocalTensor<bfloat16_t> src)
    {
        if (rows == 0) {
            return;
        }
        DataCopyExtParams params{};
        params.blockCount = static_cast<uint16_t>(rows);
        params.blockLen = 32 * sizeof(bfloat16_t);
        params.srcStride = 0;
        params.dstStride = 32 * sizeof(bfloat16_t);
        params.rsv = 0;
        const int64_t offset = OffsetBhtd(loc_.batch, loc_.hv, window_.tokenStart,
                                          td_.valueHeads, td_.tokens, kChunk) +
                               rowOffset * kChunk + colOffset;
        DataCopyPad(gmA_[offset], src, params);
    }

    __aicore__ inline void WritePublicA()
    {
        const uint32_t topRows = static_cast<uint32_t>(
            window_.rows < kHalfRows ? window_.rows : kHalfRows);
        const int64_t bottomRemaining = window_.rows - kHalfRows;
        const uint32_t bottomRows = bottomRemaining <= 0 ? 0U
            : static_cast<uint32_t>(bottomRemaining < kHalfRows ? bottomRemaining : kHalfRows);
        if (sub_ == 0) {
            WritePublicABlock(topRows, 0, 0, ubABf_);
            WritePublicABlock(topRows, 0, 32, ubABf_[1024]);
        } else {
            WritePublicABlock(bottomRows, 32, 32, ubABf_);
        }
    }

    __aicore__ inline void BuildA(int64_t chunk)
    {
        SetFlag<HardEvent::V_S>(1);
        WaitFlag<HardEvent::V_S>(1);
        const bool useEnergyRatio = GateRatioSafe(ubG_, static_cast<uint32_t>(window_.rows));
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        if (td_.useExp2 != 0) {
            Inverse32DiagonalVF<true>(ubKkt_, ubL_, ubA_, ubG_, ubBeta_, ubEnergy_,
                                      useEnergyRatio, firstRow);
        } else {
            Inverse32DiagonalVF<false>(ubKkt_, ubL_, ubA_, ubG_, ubBeta_, ubEnergy_,
                                       useEnergyRatio, firstRow);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        PublishInverse(chunk);
        PrepareQvAfterInverse(chunk);

        Cast(ubABf_, ubA_, RoundMode::CAST_RINT, 1024);
        if (sub_ == 0) {
            Duplicate(ubABf_[1024], static_cast<bfloat16_t>(0), 1024);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        if (sub_ == 0) {
            UploadNdToL1(l1A_, ubABf_, 32, 32, kChunk, 0, 0);
            UploadNdToL1(l1A_, ubABf_[1024], 32, 32, kChunk, 0, 32);
        } else {
            UploadNdToL1(l1A_, ubABf_, 32, 32, kChunk, 32, 32);
        }
        if (td_.outputA != 0) {
            WritePublicA();
        }
        AivWait(kProdCoupled);
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

    __aicore__ inline void ComputeCoupling(int64_t chunk)
    {
        const int64_t tile = TileIndex(loc_, chunk, td_.chunksPerBatch, td_.varlen != 0);
        const int64_t record = tile * td_.valueHeads + loc_.hv;
        const int64_t base = record * 3 * 1024;
        CopyGmNdToL1NzF32(l1InvX1_, gmInverse_[base + 1024]);
        CopyGmNdToL1NzF32(l1InvL10_, gmInverse_[base + 2048]);
        ComputeMmF32(l1InvX1_, l1InvL10_, l0Af_, l0Bf_, l0C0_);
        FixF32L1(l1InvT_, l0C0_);
        SetFlag<HardEvent::FIX_MTE1>(0);
        WaitFlag<HardEvent::FIX_MTE1>(0);
        ReleaseL0C();

        CopyGmNdToL1NzF32(l1InvX0_, gmInverse_[base]);
        ComputeMmF32(l1InvT_, l1InvX0_, l0Af_, l0Bf_, l0C0_);
        FixB16L1(l1A_[32 * 16], l0C0_);
        if (td_.outputA != 0) {
            const int64_t offset = OffsetBhtd(loc_.batch, loc_.hv, window_.tokenStart,
                                              td_.valueHeads, td_.tokens, kChunk) +
                                   32 * kChunk;
            const int64_t remaining = window_.rows - 32;
            const uint32_t rows = remaining <= 0 ? 0U
                : static_cast<uint32_t>(remaining < 32 ? remaining : 32);
            FixB16Gm(gmA_[offset], l0C0_, rows);
        }
        ReleaseL0C();
        AicPublishToBoth(kProdCoupled);
    }

    __aicore__ inline void RunCube(int64_t chunk)
    {
        AicWaitBoth(kProdK);
        ComputeMm(l1K_, l1K_, l0A0_, l0B0_, l0C0_, 64, 64, 128, false, false);
        FixF32Halves(ubKkt_, l0C0_, 64, 64);
        ReleaseL0C();
        AicPublishToBoth(kProdKkt);

        AicWaitBoth(kProdInverse);
        ComputeCoupling(chunk);
        AicWaitBoth(kProdInputs);
        AicWaitBoth(kProdA);
        ComputeMm(l1A_, l1Kbg_, l0A0_, l0B0_, l0C2_, 64, 128, 64, false, true);
        FixB16Halves(ubW_, l0C2_, 64, 128);
        ReleaseL0C();
        ComputeMmReuseA(l1Vb_, l0A0_, l0B1_, l0C3_, 64, 128, 64, true);
        FixB16Halves(ubU_, l0C3_, 64, 128);
        ReleaseL0C();
        AicPublishToBoth(kProdW);
    }

    FusedTilingData td_{};
    ChainLoc loc_{};
    ChunkWindow window_{};
    uint32_t sub_{0};
    bool hasBetaEffOut_{false};

    GlobalTensor<bfloat16_t> gmK_, gmV_, gmKHat_, gmW_, gmU_, gmA_;
    GlobalTensor<float> gmG_, gmBeta_, gmGOut_, gmBetaEff_, gmKRstd_, gmInverse_;
    GlobalTensor<float> gmALog_, gmDtBias_;

    LocalTensor<bfloat16_t> ubK_, ubV_, ubKHat_, ubVb_, ubKbg_, ubW_, ubU_;
    LocalTensor<bfloat16_t> ubABf_;
    LocalTensor<float> ubKkt_, ubL_, ubA_;
    LocalTensor<bfloat16_t> ubPublicB16_;
    LocalTensor<float> ubG_, ubEnergy_, ubBeta_, ubKRstd_;
    LocalTensor<uint32_t> ubShift_;
    LocalTensor<bfloat16_t> l1K_, l1Vb_, l1Kbg_, l1A_;
    LocalTensor<float> l1InvX0_, l1InvX1_, l1InvL10_, l1InvT_;
    LocalTensor<bfloat16_t> l0A0_, l0A1_, l0B0_, l0B1_;
    LocalTensor<float> l0Af_, l0Bf_;
    LocalTensor<float> l0C0_, l0C2_, l0C3_;
};

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_PRODUCER_H
