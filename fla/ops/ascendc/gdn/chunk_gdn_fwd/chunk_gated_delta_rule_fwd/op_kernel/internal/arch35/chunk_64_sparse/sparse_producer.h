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
        gmU_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.u));
        gmW_.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
        gmU_.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
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
        if ASCEND_IS_AIV {
            InitIdentity();
        }
    }

    __aicore__ inline int64_t ChunkCount() const
    {
        return loc_.chunks;
    }

    // AIV prepares chunk c+1 in the alternate slot while AIC evaluates the
    // current chunk's Neumann series and W/U. No Vector triangular solve remains.
    __aicore__ inline void Run()
    {
        if ASCEND_IS_AIV {
            for (int64_t chunk = 0; chunk < loc_.chunks; ++chunk) {
                window_ = ResolveWindow(loc_, chunk);
                PrepareInput(chunk);
                AivWait(kProdKkt);
                BuildNeumannInput(chunk);
            }
            if (td_.schedule != kSparseSchedulePipeline && loc_.chunks > 0) {
                // Serial publishes the whole chain only after the final AIC GM store.
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
            ubKkt_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, float>(ubBase + kPUbKkt);
            ubP_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(ubBase + kPUbP);
            ubS_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(ubBase + kPUbS);
            ubG_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, float>(ubBase + kPUbGates);
            ubEnergy_[slot] = ubG_[slot][64];
            ubBeta_[slot] = ubG_[slot][192];
            ubKRstd_[slot] = ubG_[slot][288];

            const uint32_t l1Base = slot * kPL1SlotBytes;
            l1K_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(l1Base + kPL1K);
            l1Vb_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(l1Base + kPL1Vb);
            l1Kbg_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(l1Base + kPL1Kbg);
            l1P_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(l1Base + kPL1P);
            l1S_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(l1Base + kPL1S);
        }
        ubShift_ = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(kPUbShift);
        l1Identity_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kPL1Identity);
        l0A0_ = buf.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(kL0Bank0);
        l0B0_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0Bank0);
        l0B1_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0Bank1);
        l0C0_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C0);
        l0C1_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C1);
        l0C2_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C2);
        l0C3_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C3);
    }

    __aicore__ inline void InitIdentity()
    {
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        Duplicate(ubS_[0], static_cast<bfloat16_t>(0), kHalfRows * kChunk);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < static_cast<uint32_t>(kHalfRows); ++row) {
            ubS_[0].SetValue(row * kChunk + firstRow + row, static_cast<bfloat16_t>(1.0f));
        }
        SetFlag<HardEvent::S_MTE3>(0);
        WaitFlag<HardEvent::S_MTE3>(0);
        UploadNdToL1(l1Identity_, ubS_[0], kHalfRows, kChunk, kChunk, firstRow);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        outputLive_[0] = false;
        outputLive_[1] = false;
    }

    __aicore__ inline void PrepareInput(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        if (outputLive_[slot]) {
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
            outputLive_[slot] = false;
        }
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        const int64_t remaining = window_.rows - static_cast<int64_t>(firstRow);
        const uint32_t rows = remaining <= 0 ? 0U
            : static_cast<uint32_t>(remaining < kHalfRows ? remaining : kHalfRows);
        if (rows < static_cast<uint32_t>(kHalfRows)) {
            Duplicate(ubK_[slot], static_cast<bfloat16_t>(0), kHalfRows * kDim);
            Duplicate(ubV_[slot], static_cast<bfloat16_t>(0), kHalfRows * kDim);
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

        CopyGmToUb(ubK_[slot], gmK_,
                   OffsetBhtd(loc_.batch, loc_.kh, token, td_.keyHeads, td_.tokens), elems);
        SetFlag<HardEvent::MTE2_V>(0);
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

        CopyGmToUb(ubG_[slot], gmG_, gateOffset, static_cast<uint32_t>(window_.rows));
        CopyGmToUb(ubBeta_[slot], gmBeta_, gateOffset, static_cast<uint32_t>(window_.rows));
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
        PrepareGateBetaVF(ubG_[slot], ubBeta_[slot], aLog, dtBias, td_.gateSoftplus != 0,
                          td_.betaSigmoid != 0, td_.allowNegEigval != 0 ? 2.0f : 1.0f);
        if (td_.useExp2 != 0) {
            CumsumEnergyVF<true>(ubG_[slot], ubEnergy_[slot], ubShift_);
        } else {
            CumsumEnergyVF<false>(ubG_[slot], ubEnergy_[slot], ubShift_);
        }

        const int64_t qkOut = OffsetBhtd(loc_.batch, loc_.kh, token, td_.keyHeads, td_.tokens);
        const int64_t rstdOut = OffsetBht(loc_.batch, loc_.kh, token, td_.keyHeads, td_.tokens);
        if (td_.useNorm != 0) {
            CopyUbToGm(gmKHat_, qkOut, ubPublicB16_[slot], elems);
            CopyUbToGm(gmKRstd_, rstdOut, ubKRstd_[slot], rows);
        }
        CopyGmToUb(ubV_[slot], gmV_,
                   OffsetBhtd(loc_.batch, loc_.hv, token, td_.valueHeads, td_.tokens), elems);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        ScaleKbgNzVF(ubPublicB16_[slot], ubKbg_[slot], ubBeta_[slot], ubEnergy_[slot], firstRow);
        ScaleVbNzVF(ubV_[slot], ubVb_[slot], ubBeta_[slot], firstRow);
        if (sub_ == 0) {
            CopyUbToGm(gmGOut_, gateOffset, ubG_[slot], static_cast<uint32_t>(window_.rows));
            if (td_.betaSigmoid != 0 && hasBetaEffOut_) {
                CopyUbToGm(gmBetaEff_, gateOffset, ubBeta_[slot], static_cast<uint32_t>(window_.rows));
            }
        }
        outputLive_[slot] = td_.useNorm != 0 || sub_ == 0;
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyNzToL1(l1Vb_[slot], ubVb_[slot], kHalfRows, kDim, kChunk, firstRow);
        CopyNzToL1(l1Kbg_[slot], ubKbg_[slot], kHalfRows, kDim, kChunk, firstRow);
        AivPublish(kProdInputs);
    }

    __aicore__ inline void BuildNeumannInput(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        SetFlag<HardEvent::V_S>(1);
        WaitFlag<HardEvent::V_S>(1);
        const bool useEnergyRatio = GateRatioSafe(ubG_[slot], static_cast<uint32_t>(window_.rows));
        if (td_.useExp2 != 0) {
            BuildNeumannInputVF<true>(ubKkt_[slot], ubP_[slot], ubS_[slot], ubG_[slot], ubBeta_[slot],
                                      ubEnergy_[slot], useEnergyRatio, firstRow);
        } else {
            BuildNeumannInputVF<false>(ubKkt_[slot], ubP_[slot], ubS_[slot], ubG_[slot], ubBeta_[slot],
                                       ubEnergy_[slot], useEnergyRatio, firstRow);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        UploadNdToL1(l1P_[slot], ubP_[slot], kHalfRows, kChunk, kChunk, firstRow);
        UploadNdToL1(l1S_[slot], ubS_[slot], kHalfRows, kChunk, kChunk, firstRow);
        AivPublish(kProdA);
    }

    __aicore__ inline void RunCube(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        AicWaitBoth(kProdK);
        ComputeMm(l1K_[slot], l1K_[slot], l0A0_, l0B0_, l0C0_, 64, 64, 128, false, false);
        FixF32Halves(ubKkt_[slot], l0C0_, 64, 64);
        ReleaseL0C();
        AicPublishToBoth(kProdKkt);

        AicWaitBoth(kProdInputs);
        AicWaitBoth(kProdA);
        ComputeMm(l1S_[slot], l1Identity_, l0A0_, l0B0_, l0C0_, 64, 64, 64, false, true);
        FixB16L1Matrix(l1S_[slot], l0C0_, 64, 64, 64);
        ReleaseL0C();
        for (uint32_t stage = 0; stage < 5; ++stage) {
            ComputeMm(l1P_[slot], l1P_[slot], l0A0_, l0B0_, l0C1_, 64, 64, 64, false, true);
            FixB16L1Matrix(l1P_[slot], l0C1_, 64, 64, 64);
            ReleaseL0C();
            // S stays in L0C. Retire P's L1 store before accumulating P@S into S.
            WaitFlag<HardEvent::FIX_M>(0);
            ComputeMm(l1P_[slot], l1S_[slot], l0A0_, l0B0_, l0C0_, 64, 64, 64,
                      false, true, false);
            FixB16L1Matrix(l1S_[slot], l0C0_, 64, 64, 64);
            if (stage == 4 && td_.outputA != 0) {
                const int64_t offset = OffsetBhtd(loc_.batch, loc_.hv, window_.tokenStart,
                                                  td_.valueHeads, td_.tokens, kChunk);
                FixB16GmNd(gmA_[offset], l0C0_, static_cast<uint32_t>(window_.rows), kChunk, kChunk);
            }
            ReleaseL0C();
        }

        const int64_t offset = OffsetBhtd(loc_.batch, loc_.hv, window_.tokenStart,
                                          td_.valueHeads, td_.tokens);
        const uint32_t rows = static_cast<uint32_t>(window_.rows);
        ComputeMm(l1S_[slot], l1Kbg_[slot], l0A0_, l0B0_, l0C2_, 64, 128, 64, false, true);
        FixB16GmNd(gmW_[offset], l0C2_, rows, kDim, kDim);
        ReleaseL0C();
        ComputeMmReuseA(l1Vb_[slot], l0A0_, l0B1_, l0C3_, 64, 128, 64, true);
        FixB16GmNd(gmU_[offset], l0C3_, rows, kDim, kDim);
        ReleaseL0C();
    }

    FusedTilingData td_{};
    ChainLoc loc_{};
    ChunkWindow window_{};
    uint32_t sub_{0};
    bool hasBetaEffOut_{false};
    bool outputLive_[2]{false, false};

    GlobalTensor<bfloat16_t> gmK_, gmV_, gmKHat_, gmW_, gmU_, gmA_;
    GlobalTensor<float> gmG_, gmBeta_, gmGOut_, gmBetaEff_, gmKRstd_;
    GlobalTensor<float> gmALog_, gmDtBias_;

    LocalTensor<bfloat16_t> ubK_[2], ubV_[2], ubKHat_[2], ubPublicB16_[2];
    LocalTensor<bfloat16_t> ubVb_[2], ubKbg_[2], ubP_[2], ubS_[2];
    LocalTensor<float> ubKkt_[2], ubG_[2], ubEnergy_[2], ubBeta_[2], ubKRstd_[2];
    LocalTensor<uint32_t> ubShift_;
    LocalTensor<bfloat16_t> l1K_[2], l1Vb_[2], l1Kbg_[2], l1P_[2], l1S_[2], l1Identity_;
    LocalTensor<bfloat16_t> l0A0_, l0B0_, l0B1_;
    LocalTensor<float> l0C0_, l0C1_, l0C2_, l0C3_;
};

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_PRODUCER_H
