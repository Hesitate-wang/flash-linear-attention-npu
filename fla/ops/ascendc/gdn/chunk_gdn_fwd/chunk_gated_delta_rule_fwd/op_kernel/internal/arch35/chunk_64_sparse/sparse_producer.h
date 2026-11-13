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
constexpr uint16_t kProdQ = 9;
constexpr uint16_t kProdQK = 10;
constexpr uint16_t kProdAPrime = 11;

class ProducerRole {
public:
    __aicore__ inline void Init(const PhaseArgs &args, const FusedTilingData &td)
    {
        td_ = td;
        sub_ = GetSubBlockIdx();
        gmK_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.k));
        gmQ_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.q));
        gmG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.rawG));
        gmBeta_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.beta));
        gmT_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.a));
        gmAPrime_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            GetUserWorkspace(args.workspace) + td.workspaceOffset));
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
                const int64_t tailRows = ResolveWindow(loc_, loc_.chunks - 1).rows;
                const int64_t fullChunks = loc_.chunks - (tailRows < kChunk ? 1 : 0);
                window_ = ResolveWindow(loc_, 0);
                if (fullChunks == 0) {
                    LoadTailInputs(0);
                } else {
                    LoadFullInputs(0);
                }
                PrepareLoadedInputs(0);
                if (loc_.chunks > 1) {
                    // Keep one prepared input and one in-flight input before
                    // entering the load(i+2)/prepare(i+1) steady pipeline.
                    window_ = ResolveWindow(loc_, 1);
                    if (fullChunks > 1) {
                        LoadFullInputs(1);
                    } else {
                        LoadTailInputs(1);
                    }
                }

                int64_t chunk = 0;
                for (; chunk + 2 < fullChunks; ++chunk) {
                    PrepareLoadedInputs(chunk + 1);
                    window_ = ResolveWindow(loc_, chunk + 2);
                    LoadFullInputs(chunk + 2);
                    AivWait(kProdKkt);
                    BuildCurrentChunk(chunk);
                }
                // At most two chunks remain. Their next load, if any, is the
                // chain's single tail chunk; full-chunk iterations stay above.
                for (; chunk < loc_.chunks; ++chunk) {
                    if (chunk + 1 < loc_.chunks) {
                        PrepareLoadedInputs(chunk + 1);
                    }
                    if (chunk + 2 < loc_.chunks) {
                        window_ = ResolveWindow(loc_, chunk + 2);
                        LoadTailInputs(chunk + 2);
                    }
                    AivWait(kProdKkt);
                    BuildCurrentChunk(chunk);
                }
                BuildAPrime(loc_.chunks - 1);
            }
            if (td_.schedule != kSparseSchedulePipeline && loc_.chunks > 0) {
                AivWaitMte3(kProdW);
            }
        }
        if ASCEND_IS_AIC {
            const int64_t iterations = td_.schedule == kSparseSchedulePipeline ? td_.maxChunks : loc_.chunks;
            for (int64_t iteration = 0; iteration < iterations; ++iteration) {
                if (td_.schedule == kSparseSchedulePipeline && iteration >= 2) {
                    // Drain generation i-2 before its parity is reused below.
                    WaitIterationAic<PIPE_M>(iteration);
                    WaitAPrimeAic<PIPE_MTE3>(iteration);
                }
                if (iteration < loc_.chunks) {
                    window_ = ResolveWindow(loc_, iteration);
                    RunKkt(iteration);
                }
                const bool hasPrevious = iteration > 0 && iteration - 1 < loc_.chunks;
                if (hasPrevious) {
                    window_ = ResolveWindow(loc_, iteration - 1);
                    RunCube(iteration - 1);
                }
                if (td_.schedule == kSparseSchedulePipeline && iteration > 0) {
                    SetIterationAic(iteration - 1);
                }
                if (hasPrevious) {
                    RunQK(iteration - 1);
                    // APrime(i-1) is generated by AIV after inverse(i).
                    if (td_.schedule == kSparseSchedulePipeline) {
                        AicWaitBoth<PIPE_MTE3>(kProdAPrime);
                    } else {
                        AicWaitBoth<PIPE_MTE2>(kProdAPrime);
                    }
                }
                if (td_.schedule == kSparseSchedulePipeline && iteration > 0) {
                    // Shorter chains still join every shifted generation.
                    SetAPrimeAic(iteration - 1);
                }
            }
            if (loc_.chunks == iterations && loc_.chunks > 0) {
                // The shifted loop closes coupling/QK at i-1. Only a chain that
                // reaches the global iteration bound needs this final compute;
                // shorter chains already closed their last chunk inside the loop.
                const int64_t chunk = loc_.chunks - 1;
                window_ = ResolveWindow(loc_, chunk);
                RunCube(chunk);
                if (td_.schedule == kSparseSchedulePipeline) {
                    SetIterationAic(chunk);
                } else {
                    AicPublishToBoth(kProdW);
                    // The serial Consumer reads T through this AIC's MTE2.
                    SetFlag<HardEvent::FIX_MTE2>(0);
                    WaitFlag<HardEvent::FIX_MTE2>(0);
                }
                RunQK(chunk);
                if (td_.schedule == kSparseSchedulePipeline) {
                    AicWaitBoth<PIPE_MTE3>(kProdAPrime);
                } else {
                    AicWaitBoth<PIPE_MTE2>(kProdAPrime);
                }
            }
            if (td_.schedule == kSparseSchedulePipeline) {
                // Close the last global generation, using a dummy publication
                // for shorter chains and the real final APrime for full chains.
                if (loc_.chunks != iterations) {
                    SetIterationAic(iterations - 1);
                }
                SetAPrimeAic(iterations - 1);
                if (iterations >= 2) {
                    WaitIterationAic<PIPE_M>(iterations - 2);
                    WaitAPrimeAic<PIPE_MTE3>(iterations - 2);
                }
                if (iterations >= 1) {
                    WaitIterationAic<PIPE_M>(iterations - 1);
                    WaitAPrimeAic<PIPE_MTE3>(iterations - 1);
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
            ubKbg_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(ubBase + kPUbKbg);
            ubKkt_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, float>(
                kPUbKkt + slot * kPUbKktSlotBytes);
            ubQ_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(
                kPUbQ + slot * 8 * kKiB);
            ubQHat_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(
                kPUbQHat + slot * 8 * kKiB);
            ubRatio_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, float>(
                kPUbRatio + slot * kPUbRatioSlotBytes);

            const uint32_t l1Base = slot * kPL1SlotBytes;
            l1K_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(l1Base + kPL1K);
            l1Vb_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(l1Base + kPL1Vb);
            l1Kbg_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(l1Base + kPL1Kbg);
            l1InvX0_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, float>(
                kPL1InvX0 + slot * kPL1InvSlotBytes);
            l1InvX1_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, float>(
                kPL1InvX1 + slot * kPL1InvSlotBytes);
            l1InvL10_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, float>(
                kPL1InvL10 + slot * kPL1InvSlotBytes);
            l1InvT_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, float>(
                kPL1InvT + slot * kPL1InvSlotBytes);
            l1QHat_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(
                kPL1QHat + slot * kPL1QSlotBytes);
        }
        l1K_[2] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(2 * kPL1SlotBytes + kPL1K);
        l1QHat_[2] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(
            kPL1QHat + 2 * kPL1QSlotBytes);
        ubRatio_[2] = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbRatioThird);
        for (uint32_t gateSlot = 0; gateSlot < 3; ++gateSlot) {
            ubG_[gateSlot] = buf.GetBuffer<BufferType::ASCEND_UB, float>(
                kPUbGates + gateSlot * 2 * kKiB);
            ubEnergy_[gateSlot] = ubG_[gateSlot][64];
            ubBeta_[gateSlot] = ubG_[gateSlot][192];
            ubKRstd_[gateSlot] = ubG_[gateSlot][288];
        }
        ubQk_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbQk);
        ubAPrime_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbAPrime);
        ubL_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbL);
        ubX_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbX);
        ubABf_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbABf);
        ubZeroBf_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kPUbZero);
        ubShift_ = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(kPUbShift);
        l0A0_ = buf.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(kL0Bank0);
        l0B0_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0Bank0);
        l0B1_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0Bank1);
        l0Af_ = buf.GetBuffer<BufferType::ASCEND_L0A, float>(kL0Bank0);
        l0Bf_ = buf.GetBuffer<BufferType::ASCEND_L0B, float>(kL0Bank0);
        l0C0_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C0);
        l0C1_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C1);
        l0C2_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C2);
        l0C3_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C3);
        l0CKkt_[0] = l0C0_;
        l0CKkt_[1] = l0C2_;
    }

    __aicore__ inline void BuildCurrentChunk(int64_t chunk)
    {
        window_ = ResolveWindow(loc_, chunk);
        BuildInverse(chunk);
        if (chunk > 0) {
            BuildAPrime(chunk - 1);
        }
    }

    __aicore__ inline void PrepareLoadedInputs(int64_t chunk)
    {
        window_ = ResolveWindow(loc_, chunk);
        PrepareK(chunk);
        PrepareGateBeta(chunk);
        PrepareQInputs(chunk);
    }

    __aicore__ inline void DrainRawInputUpload()
    {
        if (td_.useNorm == 0) {
            // UploadNdToL1 reads raw Q asynchronously. Drain the previous
            // upload before tail clearing or reusing that UB for new MTE2 data.
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }
    }

    __aicore__ inline void IssueInputs(int64_t chunk, uint32_t rows)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t gateSlot = static_cast<uint32_t>(chunk % 3);
        const int32_t eventBase = static_cast<int32_t>((chunk & 1) * 3);
        const uint32_t firstRow = sub_ * kHalfRows;
        const int64_t token = window_.tokenStart + firstRow;
        const uint32_t elems = rows * static_cast<uint32_t>(kDim);
        const int64_t gateOffset = OffsetBht(loc_.batch, loc_.hv, window_.tokenStart,
                                             td_.valueHeads, td_.tokens);
        // K, G/BETA and Q have independent completion events. Their consumers
        // wait only after the overlapped inverse and QKMask work.
        CopyGmToUb(ubK_[slot], gmK_,
                   OffsetBhtd(loc_.batch, loc_.kh, token, td_.keyHeads, td_.tokens), elems);
        SetFlag<HardEvent::MTE2_V>(eventBase);
        CopyGmToUb(ubG_[gateSlot], gmG_, gateOffset, static_cast<uint32_t>(window_.rows));
        CopyGmToUb(ubBeta_[gateSlot], gmBeta_, gateOffset, static_cast<uint32_t>(window_.rows));
        SetFlag<HardEvent::MTE2_V>(eventBase + 1);
        CopyGmToUb(ubQ_[slot], gmQ_,
                   OffsetBhtd(loc_.batch, loc_.kh, token, td_.keyHeads, td_.tokens), elems);
        SetFlag<HardEvent::MTE2_V>(eventBase + 2);
    }

    __aicore__ inline void LoadFullInputs(int64_t chunk)
    {
        // K/Q slot reuse is already ordered by AivWait(kProdKkt), and QKMask
        // consumes the ratio band rather than the previous raw gate. No extra
        // Vector fence is needed on the full-chunk path.
        DrainRawInputUpload();
        IssueInputs(chunk, kHalfRows);
    }

    __aicore__ inline void LoadTailInputs(int64_t chunk)
    {
        DrainRawInputUpload();
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t gateSlot = static_cast<uint32_t>(chunk % 3);
        const uint32_t firstRow = sub_ * kHalfRows;
        const int64_t remaining = window_.rows - static_cast<int64_t>(firstRow);
        const uint32_t rows = remaining <= 0 ? 0U
            : static_cast<uint32_t>(remaining < kHalfRows ? remaining : kHalfRows);
        // This path runs only for the chain's single tail chunk. K/Q are
        // per-AIV 32-row halves; G/BETA remain global 64-row inputs.
        if (rows < kHalfRows) {
            Duplicate(ubK_[slot], static_cast<bfloat16_t>(0), kHalfRows * kDim);
            Duplicate(ubQ_[slot], static_cast<bfloat16_t>(0), kHalfRows * kDim);
        }
        Duplicate(ubG_[gateSlot], 0.0f, kChunk);
        Duplicate(ubBeta_[gateSlot], 0.0f, kChunk);
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        IssueInputs(chunk, rows);
    }

    __aicore__ inline void PrepareGateBeta(int64_t chunk)
    {
        const uint32_t gateSlot = static_cast<uint32_t>(chunk % 3);
        const uint32_t ratioSlot = static_cast<uint32_t>(chunk % 3);
        const int32_t eventBase = static_cast<int32_t>((chunk & 1) * 3);
        const uint32_t firstRow = sub_ * kHalfRows;
        WaitFlag<HardEvent::MTE2_V>(eventBase + 1);
        float aLog = 0.0f;
        float dtBias = 0.0f;
        if (td_.hasALog != 0) {
            aLog = gmALog_.GetValue(loc_.hv);
        }
        if (td_.hasDtBias != 0) {
            dtBias = gmDtBias_.GetValue(loc_.hv);
        }
        PrepareGateBetaVF(ubG_[gateSlot], ubBeta_[gateSlot], aLog, dtBias, td_.gateSoftplus != 0,
                          td_.betaSigmoid != 0, td_.allowNegEigval != 0 ? 2.0f : 1.0f);
        if (td_.useExp2 != 0) {
            CumsumEnergyVF<true>(ubG_[gateSlot], ubEnergy_[gateSlot], ubShift_);
            const bool useEnergyRatio = GateRatioSafe(
                ubG_[gateSlot], ubL_, static_cast<uint32_t>(window_.rows));
            GateRatioBandVF<true>(ubRatio_[ratioSlot], ubG_[gateSlot], ubEnergy_[gateSlot],
                                  useEnergyRatio, firstRow);
        } else {
            CumsumEnergyVF<false>(ubG_[gateSlot], ubEnergy_[gateSlot], ubShift_);
            const bool useEnergyRatio = GateRatioSafe(
                ubG_[gateSlot], ubL_, static_cast<uint32_t>(window_.rows));
            GateRatioBandVF<false>(ubRatio_[ratioSlot], ubG_[gateSlot], ubEnergy_[gateSlot],
                                   useEnergyRatio, firstRow);
        }
    }

    __aicore__ inline void PrepareQInputs(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t l1Slot = static_cast<uint32_t>(chunk % 3);
        const int32_t eventBase = static_cast<int32_t>((chunk & 1) * 3);
        const uint32_t firstRow = sub_ * kHalfRows;
        WaitFlag<HardEvent::MTE2_V>(eventBase + 2);
        if (td_.useNorm != 0) {
            NormalizeRowsNzVF<true, true>(ubQ_[slot], ubQHat_[slot], ubQ_[slot], ubKRstd_[slot]);
        }
        SetFlag<HardEvent::V_MTE3>(1);
        WaitFlag<HardEvent::V_MTE3>(1);
        if (td_.useNorm != 0) {
            CopyNzToL1(l1QHat_[l1Slot], ubQHat_[slot], kHalfRows, kDim, kChunk, firstRow);
        } else {
            UploadNdToL1(l1QHat_[l1Slot], ubQ_[slot], kHalfRows, kDim, kChunk, firstRow);
        }
        AivPublish(kProdQ);
    }

    __aicore__ inline void PrepareK(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t l1Slot = static_cast<uint32_t>(chunk % 3);
        const int32_t eventBase = static_cast<int32_t>((chunk & 1) * 3);
        const uint32_t firstRow = sub_ * kHalfRows;
        WaitFlag<HardEvent::MTE2_V>(eventBase);
        if (td_.useNorm != 0) {
            NormalizeRowsNzVF<true, true>(ubK_[slot], ubKHat_[slot], ubPublicB16_[slot], ubKRstd_[slot]);
        } else {
            NormalizeRowsNzVF<false, true>(ubK_[slot], ubKHat_[slot], ubPublicB16_[slot], ubKRstd_[slot]);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyNzToL1(l1K_[l1Slot], ubKHat_[slot], kHalfRows, kDim, kChunk, firstRow);
        AivPublish(kProdK);
    }

    __aicore__ inline void PublishInverse(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        if (sub_ == 0) {
            CopyUbNdToL1NzF32(l1InvX0_[slot], ubX_);
        } else {
            CopyUbNdToL1NzF32(l1InvX1_[slot], ubX_);
            // L10 is a complete 32x32 leaf, not a local triangular matrix.
            CopyUbNdToL1NzF32(l1InvL10_[slot], ubX_[2048]);
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
        const uint32_t ratioSlot = static_cast<uint32_t>(chunk % 3);
        // The previous inverse/T copies must release ubX_/ubABf_ before reuse.
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::V_S>(1);
        WaitFlag<HardEvent::V_S>(1);
        const uint32_t firstRow = sub_ * kHalfRows;
        Inverse32DiagonalVF(ubKkt_[slot], ubL_, ubX_, ubRatio_[ratioSlot],
                            ubBeta_[ratioSlot], firstRow);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        PublishInverse(chunk);

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
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        LocalTensor<float> l0C = l0CKkt_[slot];
        ComputeMmF32(l1InvX1_[slot], l1InvL10_[slot], l0Af_, l0Bf_, l0C);
        FixF32L1Matrix(l1InvT_[slot], l0C);
        SetFlag<HardEvent::FIX_MTE1>(0);
        WaitFlag<HardEvent::FIX_MTE1>(0);
        ReleaseL0C();

        ComputeMmF32(l1InvT_[slot], l1InvX0_[slot], l0Af_, l0Bf_, l0C);
        // X10 is the lower-left block. X1 is independently written to the
        // lower-right block by AIV1, so the two stores must not overlap.
        const int64_t offset = OffsetBhtd(loc_.batch, loc_.hv, window_.tokenStart,
                                          td_.valueHeads, td_.tokens, kChunk) + 32 * kChunk;
        const int64_t remaining = window_.rows - 32;
        const uint32_t rows = remaining <= 0 ? 0U
            : static_cast<uint32_t>(remaining < 32 ? remaining : 32);
        FixB16GmNd(gmT_[offset], l0C, rows, 32, 64, 32);
        ReleaseL0C();
    }

    __aicore__ inline void RunQK(int64_t chunk)
    {
        const uint32_t l1Slot = static_cast<uint32_t>(chunk % 3);
        AicWaitBoth(kProdQ);
        ComputeMm(l1QHat_[l1Slot], l1K_[l1Slot], l0A0_, l0B1_, l0C1_, 64, 64, 128,
                  false, false);
        FixF32Halves(ubQk_, l0C1_, 64, 64);
        ReleaseL0C();
        AicPublishToBoth(kProdQK);
    }

    __aicore__ inline void BuildAPrime(int64_t chunk)
    {
        window_ = ResolveWindow(loc_, chunk);
        const uint32_t ratioSlot = static_cast<uint32_t>(chunk % 3);
        const uint32_t firstRow = sub_ * kHalfRows;
        AivWait(kProdQK);
        APrimeBandNzVF(ubQk_, ubAPrime_, ubRatio_[ratioSlot], firstRow);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        // Each AIV writes its 32-row NZ band into the full 64-row GM tile.
        DataCopyExtParams params{};
        params.blockCount = static_cast<uint16_t>(kChunk / 16);
        params.blockLen = kHalfRows * 16 * sizeof(bfloat16_t);
        params.srcStride = 0;
        params.dstStride = (kChunk - kHalfRows) * 16 * sizeof(bfloat16_t);
        params.rsv = 0;
        const int64_t tile = TileIndex(loc_, chunk, td_.chunksPerBatch, td_.varlen != 0);
        const int64_t offset = OffsetMatrixTile(tile, loc_.hv, td_.valueHeads) + firstRow * 16;
        DataCopyPad(gmAPrime_[offset], ubAPrime_, params);
        AivPublish(kProdAPrime);
    }

    __aicore__ inline void RunKkt(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t l1Slot = static_cast<uint32_t>(chunk % 3);
        AicWaitBoth(kProdK);
        ComputeMm(l1K_[l1Slot], l1K_[l1Slot], l0A0_, l0B0_, l0CKkt_[slot], 64, 64, 128,
                  false, false);
        FixF32Halves(ubKkt_[slot], l0CKkt_[slot], 64, 64);
        ReleaseL0C();
        AicPublishToBoth(kProdKkt);
    }

    __aicore__ inline void RunCube(int64_t chunk)
    {
        AicWaitBoth(kProdInverse);
        ComputeCoupling(chunk);
        AicWaitBoth<PIPE_FIX>(kProdA);
    }

    FusedTilingData td_{};
    ChainLoc loc_{};
    ChunkWindow window_{};
    uint32_t sub_{0};

    GlobalTensor<bfloat16_t> gmK_, gmQ_, gmT_, gmAPrime_;
    GlobalTensor<float> gmG_, gmBeta_;
    GlobalTensor<float> gmALog_, gmDtBias_;

    LocalTensor<bfloat16_t> ubK_[2], ubV_[2], ubKHat_[2], ubPublicB16_[2], ubKbg_[2];
    LocalTensor<bfloat16_t> ubQ_[2], ubQHat_[2], ubAPrime_;
    LocalTensor<float> ubKkt_[2], ubL_, ubX_, ubQk_, ubRatio_[3];
    LocalTensor<bfloat16_t> ubABf_, ubZeroBf_;
    LocalTensor<float> ubG_[3], ubEnergy_[3], ubBeta_[3], ubKRstd_[3];
    LocalTensor<uint32_t> ubShift_;
    LocalTensor<bfloat16_t> l1K_[3], l1Vb_[2], l1Kbg_[2], l1QHat_[3];
    LocalTensor<float> l1InvX0_[2], l1InvX1_[2], l1InvL10_[2], l1InvT_[2];
    LocalTensor<bfloat16_t> l0A0_, l0B0_, l0B1_;
    LocalTensor<float> l0Af_, l0Bf_;
    LocalTensor<float> l0C0_, l0C1_, l0C2_, l0C3_, l0CKkt_[2];
};

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_PRODUCER_H
