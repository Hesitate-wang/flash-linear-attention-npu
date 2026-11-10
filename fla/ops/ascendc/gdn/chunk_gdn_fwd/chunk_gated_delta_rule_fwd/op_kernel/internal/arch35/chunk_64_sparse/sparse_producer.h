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

constexpr uint16_t kProdKkt = 1;
constexpr uint16_t kProdA = 3;
constexpr uint16_t kProdW = 4;
constexpr uint16_t kProdInverse = 5;
constexpr uint16_t kProdK = 8;

class ProducerRole {
public:
    __aicore__ inline void Init(const PhaseArgs &args, const FusedTilingData &td)
    {
        td_ = td;
        sub_ = GetSubBlockIdx();
        gmK_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.k));
        gmG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.rawG));
        gmBeta_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.beta));
        gmT_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.a));
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

    __aicore__ inline void Run()
    {
        if ASCEND_IS_AIV {
            if (loc_.chunks > 0) {
                window_ = ResolveWindow(loc_, 0);
                PrepareGate(0);
            }
            for (int64_t chunk = 0; chunk < loc_.chunks; ++chunk) {
                window_ = ResolveWindow(loc_, chunk);
                AivWait(kProdKkt);
                BuildInverse(chunk);
                if (chunk + 1 < loc_.chunks) {
                    window_ = ResolveWindow(loc_, chunk + 1);
                    PrepareGate(chunk + 1);
                }
            }
            if (td_.schedule != kSparseSchedulePipeline && loc_.chunks > 0) {
                AivWaitMte3(kProdW);
            }
        }
        if ASCEND_IS_AIC {
            const int64_t iterations = td_.schedule == kSparseSchedulePipeline ? td_.maxChunks : loc_.chunks;
            for (int64_t iteration = 0; iteration < iterations; ++iteration) {
                if (iteration < loc_.chunks) {
                    window_ = ResolveWindow(loc_, iteration);
                    RunCube(iteration);
                    if (td_.schedule != kSparseSchedulePipeline && iteration + 1 == loc_.chunks) {
                        AicPublishToBoth(kProdW);
                    }
                }
                if (td_.schedule == kSparseSchedulePipeline) {
                    PublishIterationAic<PIPE_M>();
                }
            }
        }
    }

private:
    __aicore__ inline void InitBuffers()
    {
        OnChipBuffer buf;
        for (uint32_t slot = 0; slot < 2; ++slot) {
            const uint32_t ubBase = slot * kPUbSlotBytes;
            ubK_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(ubBase + kPUbK);
            ubV_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(ubBase + kPUbV);
            ubKHat_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(ubBase + kPUbKHat);
            ubPublicB16_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(ubBase + kPUbPublic);
            ubVb_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(ubBase + kPUbVb);
            ubKbg_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(ubBase + kPUbKbg);
            ubG_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbGates + slot * 2 * kKiB);
            ubEnergy_[slot] = ubG_[slot][64];
            ubBeta_[slot] = ubG_[slot][192];
            ubKRstd_[slot] = ubG_[slot][288];

            const uint32_t l1Base = slot * kPL1SlotBytes;
            l1K_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(l1Base + kPL1K);
            l1Vb_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(l1Base + kPL1Vb);
            l1Kbg_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(l1Base + kPL1Kbg);
        }
        ubKkt_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbKkt);
        ubL_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbL);
        ubX_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbX);
        ubABf_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbABf);
        ubZeroBf_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbZero);
        ubShift_ = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(kPUbShift);
        l1InvX0_ = buf.GetBuffer<BufferType::ASCEND_CB, float>(kPL1InvX0);
        l1InvX1_ = buf.GetBuffer<BufferType::ASCEND_CB, float>(kPL1InvX1);
        l1InvL10_ = buf.GetBuffer<BufferType::ASCEND_CB, float>(kPL1InvL10);
        l1InvT_ = buf.GetBuffer<BufferType::ASCEND_CB, float>(kPL1InvT);
        l0A0_ = buf.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(kL0Bank0);
        l0B0_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0Bank0);
        l0B1_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0Bank1);
        l0Af_ = buf.GetBuffer<BufferType::ASCEND_L0A, float>(kL0Bank0);
        l0Bf_ = buf.GetBuffer<BufferType::ASCEND_L0B, float>(kL0Bank0);
        l0C0_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C0);
        l0C2_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C2);
        l0C3_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C3);
    }

    __aicore__ inline void PrepareGate(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        const int64_t remaining = window_.rows - static_cast<int64_t>(firstRow);
        const uint32_t rows = remaining <= 0 ? 0U
            : static_cast<uint32_t>(remaining < kHalfRows ? remaining : kHalfRows);
        if (rows < static_cast<uint32_t>(kHalfRows)) {
            Duplicate(ubK_[slot], static_cast<bfloat16_t>(0), kHalfRows * kDim);
        }
        if (window_.rows < kChunk) {
            Duplicate(ubG_[slot], 0.0f, kChunk);
            Duplicate(ubBeta_[slot], 0.0f, kChunk);
        }
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        const int64_t token = window_.tokenStart + firstRow;
        const uint32_t elems = rows * static_cast<uint32_t>(kDim);
        const int64_t gateOffset = OffsetBht(loc_.batch, loc_.hv, window_.tokenStart,
                                             td_.valueHeads, td_.tokens);
        // Strict load/wait/compute stages. KKT is released before the next
        // gate/beta load starts; V is loaded only after gate processing.
        CopyGmToUb(ubK_[slot], gmK_,
                   OffsetBhtd(loc_.batch, loc_.kh, token, td_.keyHeads, td_.tokens), elems);
        SetFlag<HardEvent::MTE2_V>(0);
        PrepareK(chunk);
        CopyGmToUb(ubG_[slot], gmG_, gateOffset, static_cast<uint32_t>(window_.rows));
        CopyGmToUb(ubBeta_[slot], gmBeta_, gateOffset, static_cast<uint32_t>(window_.rows));
        SetFlag<HardEvent::MTE2_V>(1);
        WaitFlag<HardEvent::MTE2_V>(1);
        float aLog = 0.0f;
        float dtBias = 0.0f;
        if (td_.hasALog != 0) {
            aLog = gmALog_.GetValue(loc_.hv);
        }
        if (td_.hasDtBias != 0) {
            dtBias = gmDtBias_.GetValue(loc_.hv);
        }
        PrepareGateBetaVF(ubG_[slot], ubBeta_[slot], aLog, dtBias, td_.gateSoftplus != 0,
                          td_.betaSigmoid != 0, td_.allowNegEigval != 0 ? 2.0f : 1.0f);
        if (td_.useExp2 != 0) {
            CumsumEnergyVF<true>(ubG_[slot], ubEnergy_[slot], ubShift_);
        } else {
            CumsumEnergyVF<false>(ubG_[slot], ubEnergy_[slot], ubShift_);
        }
    }

    __aicore__ inline void PrepareK(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        WaitFlag<HardEvent::MTE2_V>(0);
        if (td_.useNorm != 0) {
            NormalizeRowsNzVF<true, true>(ubK_[slot], ubKHat_[slot], ubPublicB16_[slot], ubKRstd_[slot]);
        } else {
            NormalizeRowsNzVF<false, true>(ubK_[slot], ubKHat_[slot], ubPublicB16_[slot], ubKRstd_[slot]);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyNzToL1(l1K_[slot], ubKHat_[slot], kHalfRows, kDim, kChunk, firstRow);
        AivPublish(kProdK);
    }

    __aicore__ inline void PublishInverse()
    {
        if (sub_ == 0) {
            CopyUbNdToL1NzF32(l1InvX0_, ubX_);
        } else {
            CopyUbNdToL1NzF32(l1InvX1_, ubX_);
            // L10 is a complete 32x32 leaf, not a local triangular matrix.
            CopyUbNdToL1NzF32(l1InvL10_, ubX_[2048]);
        }
        AivPublish(kProdInverse);
    }

    __aicore__ inline void WriteTBlock(uint32_t rows, uint32_t rowOffset,
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
        DataCopyPad(gmT_[offset], src, params);
    }

    __aicore__ inline void BuildInverse(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        // The previous inverse/T copies must release ubX_/ubABf_ before reuse.
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::V_S>(1);
        WaitFlag<HardEvent::V_S>(1);
        const bool useEnergyRatio = GateRatioSafe(ubG_[slot], ubL_, static_cast<uint32_t>(window_.rows));
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        if (td_.useExp2 != 0) {
            Inverse32DiagonalVF<true>(ubKkt_, ubL_, ubX_, ubG_[slot], ubBeta_[slot],
                                      ubEnergy_[slot], useEnergyRatio, firstRow);
        } else {
            Inverse32DiagonalVF<false>(ubKkt_, ubL_, ubX_, ubG_[slot], ubBeta_[slot],
                                       ubEnergy_[slot], useEnergyRatio, firstRow);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        PublishInverse();

        Cast(ubABf_, ubX_, RoundMode::CAST_RINT, 1024);
        if (sub_ == 0) {
            Duplicate(ubZeroBf_, static_cast<bfloat16_t>(0), 1024);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        if (sub_ == 0) {
            const uint32_t topRows = static_cast<uint32_t>(
                window_.rows < kHalfRows ? window_.rows : kHalfRows);
            WriteTBlock(topRows, 0, 0, ubABf_);
            WriteTBlock(topRows, 0, 32, ubZeroBf_);
        } else {
            const int64_t bottomRemaining = window_.rows - kHalfRows;
            const uint32_t bottomRows = bottomRemaining <= 0 ? 0U
                : static_cast<uint32_t>(bottomRemaining < kHalfRows ? bottomRemaining : kHalfRows);
            WriteTBlock(bottomRows, 32, 32, ubABf_);
        }
        AivPublish(kProdA);
    }

    __aicore__ inline void ComputeCoupling(int64_t chunk)
    {
        ComputeMmF32(l1InvX1_, l1InvL10_, l0Af_, l0Bf_, l0C0_);
        FixF32L1Matrix(l1InvT_, l0C0_);
        SetFlag<HardEvent::FIX_MTE1>(0);
        WaitFlag<HardEvent::FIX_MTE1>(0);
        ReleaseL0C();

        ComputeMmF32(l1InvT_, l1InvX0_, l0Af_, l0Bf_, l0C0_);
        const int64_t offset = OffsetBhtd(loc_.batch, loc_.hv, window_.tokenStart,
                                          td_.valueHeads, td_.tokens, kChunk) + 32 * kChunk + 32;
        const int64_t remaining = window_.rows - 32;
        const uint32_t rows = remaining <= 0 ? 0U
            : static_cast<uint32_t>(remaining < 32 ? remaining : 32);
        FixB16GmNd(gmT_[offset], l0C0_, rows, 32, 64, 32);
        ReleaseL0C();
    }

    __aicore__ inline void RunCube(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        AicWaitBoth(kProdK);
        ComputeMm(l1K_[slot], l1K_[slot], l0A0_, l0B0_, l0C0_, 64, 64, 128, false, false);
        FixF32Halves(ubKkt_, l0C0_, 64, 64);
        ReleaseL0C();
        AicPublishToBoth(kProdKkt);

        AicWaitBoth(kProdInverse);
        ComputeCoupling(chunk);
        AicWaitBoth<PIPE_FIX>(kProdA);
    }

    FusedTilingData td_{};
    ChainLoc loc_{};
    ChunkWindow window_{};
    uint32_t sub_{0};

    GlobalTensor<bfloat16_t> gmK_, gmT_;
    GlobalTensor<float> gmG_, gmBeta_;
    GlobalTensor<float> gmALog_, gmDtBias_;

    LocalTensor<bfloat16_t> ubK_[2], ubV_[2], ubKHat_[2], ubPublicB16_[2], ubVb_[2], ubKbg_[2];
    LocalTensor<float> ubKkt_, ubL_, ubX_;
    LocalTensor<bfloat16_t> ubABf_, ubZeroBf_;
    LocalTensor<float> ubG_[2], ubEnergy_[2], ubBeta_[2], ubKRstd_[2];
    LocalTensor<uint32_t> ubShift_;
    LocalTensor<bfloat16_t> l1K_[2], l1Vb_[2], l1Kbg_[2];
    LocalTensor<float> l1InvX0_, l1InvX1_, l1InvL10_, l1InvT_;
    LocalTensor<bfloat16_t> l0A0_, l0B0_, l0B1_;
    LocalTensor<float> l0Af_, l0Bf_;
    LocalTensor<float> l0C0_, l0C2_, l0C3_;
};

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_PRODUCER_H
