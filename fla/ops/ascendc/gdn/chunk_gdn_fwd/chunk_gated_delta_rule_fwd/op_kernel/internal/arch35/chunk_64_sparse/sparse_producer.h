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
constexpr uint16_t kProdK = 8;
constexpr uint16_t kProdQ = 9;
constexpr uint16_t kProdQK = 10;
constexpr uint16_t kProdAPrime = 11;
constexpr int32_t kProdInputMte3EventBase = 0;
constexpr int32_t kProdOutputMte3EventBase = 2;

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
            InitIdentity64();
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
            const int64_t first = static_cast<int64_t>(sub_);
            if (first < loc_.chunks) {
                // The first pair is prepared before the loop. Its successor is
                // already loaded so round i can preprocess i+1 immediately.
                LoadInputs(first);
                PrepareLoadedInputs(first);
                if (first + 2 < loc_.chunks) {
                    LoadInputs(first + 2);
                }
                for (int64_t chunk = first; chunk < loc_.chunks; chunk += 2) {
                    if (chunk + 2 < loc_.chunks) {
                        PrepareLoadedInputs(chunk + 2);
                    }
                    if (chunk + 4 < loc_.chunks) {
                        // Independent MTE2 work for the next round's preprocess.
                        LoadInputs(chunk + 4);
                    }
                    AivWait(kProdKkt);
                    BuildInverse(chunk);
                    BuildAPrime(chunk);
                }
                const int64_t last = first + ((loc_.chunks - 1 - first) >> 1) * 2;
                if (last - 2 >= first) {
                    const uint32_t previousSlot = static_cast<uint32_t>(((last - 2) >> 1) & 1);
                    WaitFlag<HardEvent::MTE3_V>(kProdOutputMte3EventBase + previousSlot);
                }
                const uint32_t lastSlot = static_cast<uint32_t>((last >> 1) & 1);
                // At most the last two output events remain unconsumed. Earlier
                // events were consumed by their same-slot preprocess.
                WaitFlag<HardEvent::MTE3_V>(kProdOutputMte3EventBase + lastSlot);
            }
        }
        if ASCEND_IS_AIC {
            const int64_t iterations = td_.maxChunks;
            const int64_t pairs = (iterations + 1) / 2;
            for (int64_t pair = 0; pair < pairs; ++pair) {
                for (uint32_t lane = 0; lane < 2; ++lane) {
                    const int64_t iteration = 2 * pair + lane;
                    if (iteration < iterations && iteration >= 2) {
                        WaitIterationAic<PIPE_M>(iteration);
                        WaitAPrimeAic<PIPE_MTE3>(iteration);
                    }
                }
                for (uint32_t lane = 0; lane < 2; ++lane) {
                    const int64_t chunk = 2 * pair + lane;
                    if (chunk < loc_.chunks) {
                        RunKkt(chunk);
                    }
                }
                for (uint32_t lane = 0; lane < 2; ++lane) {
                    const int64_t chunk = 2 * pair + lane;
                    if (chunk < loc_.chunks) {
                        RunQK(chunk);
                    }
                }
                for (uint32_t lane = 0; lane < 2; ++lane) {
                    const int64_t chunk = 2 * pair + lane;
                    if (chunk < iterations) {
                        if (chunk < loc_.chunks) {
                            AicWaitOwner<PIPE_FIX>(chunk, kProdA);
                        }
                        // Short and empty chains still join every global generation.
                        SetIterationAic(chunk);
                    }
                }
                for (uint32_t lane = 0; lane < 2; ++lane) {
                    const int64_t chunk = 2 * pair + lane;
                    if (chunk < iterations) {
                        if (chunk < loc_.chunks) {
                            // Backpressure also releases the single KKT/QK UB slots.
                            AicWaitOwner<PIPE_FIX>(chunk, kProdAPrime);
                        }
                        SetAPrimeAic(chunk);
                    }
                }
            }
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

private:
    __aicore__ inline void InitBuffers()
    {
        OnChipBuffer buf;
        ubKkt_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbKkt);
        ubQk_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbQk);
        for (uint32_t slot = 0; slot < 3; ++slot) {
            l1K_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(
                kPL1K + slot * kPL1SlotBytes);
            l1QHat_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(
                kPL1QHat + slot * kPL1QSlotBytes);
        }

        if ASCEND_IS_AIV {
            for (uint32_t slot = 0; slot < 2; ++slot) {
                const uint32_t inputOffset = slot * kPUbInputSlotBytes;
                ubK_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(
                    kPUbK + inputOffset);
                ubQ_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(
                    kPUbQ + inputOffset);
                ubKHat_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(
                    kPUbKHat + inputOffset);
                ubQHat_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(
                    kPUbQHat + inputOffset);
                ubRatio_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, float>(
                    kPUbRatio + slot * kPUbRatioSlotBytes);

                // Inverse and QKMask use the opposite slot from preprocess.
                // L is dead after substitution, so its two 8 KiB halves hold
                // the BF16 T and APrime outputs without additional UB storage.
                ubL_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, float>(
                    kPUbKHat + inputOffset);
                ubX_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, float>(
                    kPUbQHat + inputOffset);
                ubABf_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(
                    kPUbKHat + inputOffset);
                ubAPrime_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(
                    kPUbKHat + inputOffset + 8 * kKiB);
            }
            for (uint32_t gateSlot = 0; gateSlot < 3; ++gateSlot) {
                ubG_[gateSlot] = buf.GetBuffer<BufferType::ASCEND_UB, float>(
                    kPUbGates + gateSlot * kPUbGateSlotBytes);
                ubEnergy_[gateSlot] = ubG_[gateSlot][64];
                ubBeta_[gateSlot] = ubG_[gateSlot][192];
                ubKRstd_[gateSlot] = ubG_[gateSlot][256];
                ubEnergyRecip_[gateSlot] = ubG_[gateSlot][320];
            }
            ubShift_ = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(kPUbShift);
            ubScratch_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbScratch);
            ubEye_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kPUbEye);
        }

        if ASCEND_IS_AIC {
            l0A0_ = buf.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(kL0Bank0);
            l0B0_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0Bank0);
            l0B1_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0Bank1);
            l0C0_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C0);
            l0C1_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C1);
        }
    }

    __aicore__ inline void InitIdentity64()
    {
        Duplicate(ubEye_, 0.0f, kMatrixElems);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        for (uint32_t row = 0; row < kChunk; ++row) {
            ubEye_.SetValue(row * (kChunk + 1), 1.0f);
        }
    }

    __aicore__ inline void PrepareLoadedInputs(int64_t chunk)
    {
        const int64_t pair = chunk >> 1;
        const uint32_t slot = static_cast<uint32_t>(pair & 1);
        if (pair >= 2) {
            // Wait only for the output from pair-2 that owns this same slot.
            // The opposite slot's preprocess/inverse remains independent.
            WaitFlag<HardEvent::MTE3_V>(kProdOutputMte3EventBase + slot);
        }
        PrepareK(chunk);
        PrepareGateBeta(chunk);
        PrepareQInputs(chunk);
        // Publish this slot's K/Q L1 reads without draining them here. The
        // matching inverse waits this event before reusing the UB as L/X.
        SetFlag<HardEvent::MTE3_V>(kProdInputMte3EventBase + slot);
    }

    __aicore__ inline void IssueInputs(int64_t chunk, uint32_t rows)
    {
        const ChunkWindow window = WindowOf(loc_, chunk);
        const int64_t pair = chunk >> 1;
        const uint32_t slot = static_cast<uint32_t>(pair & 1);
        const uint32_t gateSlot = static_cast<uint32_t>(pair % 3);
        const int32_t eventBase = static_cast<int32_t>(slot * 3);
        const uint32_t elems = rows * static_cast<uint32_t>(kDim);
        const int64_t gateOffset = OffsetBht(loc_.batch, loc_.hv, window.tokenStart,
                                             td_.valueHeads, td_.tokens);
        CopyGmToUb(ubK_[slot], gmK_,
                   OffsetBhtd(loc_.batch, loc_.kh, window.tokenStart,
                              td_.keyHeads, td_.tokens), elems);
        SetFlag<HardEvent::MTE2_V>(eventBase);
        CopyGmToUb(ubG_[gateSlot], gmG_, gateOffset, rows);
        CopyGmToUb(ubBeta_[gateSlot], gmBeta_, gateOffset, rows);
        SetFlag<HardEvent::MTE2_V>(eventBase + 1);
        CopyGmToUb(ubQ_[slot], gmQ_,
                   OffsetBhtd(loc_.batch, loc_.kh, window.tokenStart,
                              td_.keyHeads, td_.tokens), elems);
        SetFlag<HardEvent::MTE2_V>(eventBase + 2);
    }

    __aicore__ inline void LoadInputs(int64_t chunk)
    {
        const ChunkWindow window = WindowOf(loc_, chunk);
        const int64_t pair = chunk >> 1;
        const uint32_t slot = static_cast<uint32_t>(pair & 1);
        const uint32_t gateSlot = static_cast<uint32_t>(pair % 3);
        const uint32_t rows = static_cast<uint32_t>(window.rows);
        if (rows < kChunk) {
            Duplicate(ubK_[slot], static_cast<bfloat16_t>(0), kChunkElems);
            Duplicate(ubQ_[slot], static_cast<bfloat16_t>(0), kChunkElems);
            Duplicate(ubG_[gateSlot], 0.0f, kChunk);
            Duplicate(ubBeta_[gateSlot], 0.0f, kChunk);
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }
        IssueInputs(chunk, rows);
    }

    __aicore__ inline void PrepareGateBeta(int64_t chunk)
    {
        const ChunkWindow window = WindowOf(loc_, chunk);
        const int64_t pair = chunk >> 1;
        const uint32_t gateSlot = static_cast<uint32_t>(pair % 3);
        const uint32_t ratioSlot = static_cast<uint32_t>(pair & 1);
        const int32_t eventBase = static_cast<int32_t>(ratioSlot * 3);
        WaitFlag<HardEvent::MTE2_V>(eventBase + 1);
        float aLog = 0.0f;
        float dtBias = 0.0f;
        if (td_.hasALog != 0) {
            aLog = gmALog_.GetValue(loc_.hv);
        }
        if (td_.hasDtBias != 0) {
            dtBias = gmDtBias_.GetValue(loc_.hv);
        }
        if (td_.gateSoftplus != 0 || td_.betaSigmoid != 0) {
            PrepareGateBetaVF(ubG_[gateSlot], ubBeta_[gateSlot], aLog, dtBias,
                              td_.gateSoftplus != 0, td_.betaSigmoid != 0,
                              td_.allowNegEigval != 0 ? 2.0f : 1.0f);
        }
        const bool useEnergyRatio = td_.useExp2 != 0
            ? PrepareCumsumAndCheckRatio<true>(gateSlot, window.rows)
            : PrepareCumsumAndCheckRatio<false>(gateSlot, window.rows);
        if (td_.useExp2 != 0) {
            GateRatioVF<true>(ubRatio_[ratioSlot], ubG_[gateSlot], ubEnergy_[gateSlot],
                              ubEnergyRecip_[gateSlot], useEnergyRatio);
        } else {
            GateRatioVF<false>(ubRatio_[ratioSlot], ubG_[gateSlot], ubEnergy_[gateSlot],
                               ubEnergyRecip_[gateSlot], useEnergyRatio);
        }
    }

    template <bool UseExp2>
    __aicore__ inline bool PrepareCumsumAndCheckRatio(uint32_t gateSlot, int64_t rows)
    {
        CumsumEnergyVF<UseExp2>(ubG_[gateSlot], ubEnergy_[gateSlot],
                                ubEnergyRecip_[gateSlot], ubShift_);
        return GateRatioSafe(ubG_[gateSlot], ubScratch_, static_cast<uint32_t>(rows));
    }

    __aicore__ inline void PrepareQInputs(int64_t chunk)
    {
        const int64_t pair = chunk >> 1;
        const uint32_t slot = static_cast<uint32_t>(pair & 1);
        const uint32_t gateSlot = static_cast<uint32_t>(pair % 3);
        const uint32_t l1Slot = static_cast<uint32_t>(pair % 3);
        const int32_t eventBase = static_cast<int32_t>(slot * 3);
        WaitFlag<HardEvent::MTE2_V>(eventBase + 2);
        if (td_.useNorm != 0) {
            NormalizeRowsNzVF<true>(ubQ_[slot], ubQHat_[slot], ubKRstd_[gateSlot]);
        }
        SetFlag<HardEvent::V_MTE3>(1);
        WaitFlag<HardEvent::V_MTE3>(1);
        if (td_.useNorm != 0) {
            CopyNzToL1(l1QHat_[l1Slot], ubQHat_[slot], kChunk, kDim, kChunk, 0);
        } else {
            UploadNdToL1(l1QHat_[l1Slot], ubQ_[slot], kChunk, kDim, kChunk, 0);
        }
        AivPublish(kProdQ);
    }

    __aicore__ inline void PrepareK(int64_t chunk)
    {
        const int64_t pair = chunk >> 1;
        const uint32_t slot = static_cast<uint32_t>(pair & 1);
        const uint32_t gateSlot = static_cast<uint32_t>(pair % 3);
        const uint32_t l1Slot = static_cast<uint32_t>(pair % 3);
        const int32_t eventBase = static_cast<int32_t>(slot * 3);
        WaitFlag<HardEvent::MTE2_V>(eventBase);
        if (td_.useNorm != 0) {
            NormalizeRowsNzVF<true>(ubK_[slot], ubKHat_[slot], ubKRstd_[gateSlot]);
        } else {
            NormalizeRowsNzVF<false>(ubK_[slot], ubKHat_[slot], ubKRstd_[gateSlot]);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyNzToL1(l1K_[l1Slot], ubKHat_[slot], kChunk, kDim, kChunk, 0);
        AivPublish(kProdK);
    }

    __aicore__ inline void WriteT(int64_t chunk, uint32_t rows, LocalTensor<bfloat16_t> src)
    {
        if (rows == 0) {
            return;
        }
        const ChunkWindow window = WindowOf(loc_, chunk);
        DataCopyExtParams params{};
        params.blockCount = static_cast<uint16_t>(rows);
        params.blockLen = kChunk * sizeof(bfloat16_t);
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        const int64_t offset = OffsetBhtd(loc_.batch, loc_.hv, window.tokenStart,
                                          td_.valueHeads, td_.tokens, kChunk);
        DataCopyPad(gmT_[offset], src, params);
    }

    __aicore__ inline void BuildInverse(int64_t chunk)
    {
        const ChunkWindow window = WindowOf(loc_, chunk);
        const int64_t pair = chunk >> 1;
        const uint32_t slot = static_cast<uint32_t>(pair & 1);
        const uint32_t gateSlot = static_cast<uint32_t>(pair % 3);
        // Wait only for this chunk's preprocess L1 uploads before reusing its
        // normalized K/Q UB as L/X. Do not drain the next pair's uploads.
        WaitFlag<HardEvent::MTE3_V>(kProdInputMte3EventBase + slot);
        Inverse64VF(ubKkt_, ubL_[slot], ubX_[slot], ubRatio_[slot], ubBeta_[gateSlot], ubEye_);
        Cast(ubABf_[slot], ubX_[slot], RoundMode::CAST_RINT, kMatrixElems);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        WriteT(chunk, static_cast<uint32_t>(window.rows), ubABf_[slot]);
        AivPublish(kProdA);
    }

    __aicore__ inline void BuildAPrime(int64_t chunk)
    {
        const int64_t pair = chunk >> 1;
        const uint32_t slot = static_cast<uint32_t>(pair & 1);
        AivWait(kProdQK);
        APrimeNzVF(ubQk_, ubAPrime_[slot], ubRatio_[slot]);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        const int64_t tile = TileIndex(loc_, chunk, td_.chunksPerBatch, td_.varlen != 0);
        const int64_t offset = OffsetMatrixTile(tile, loc_.hv, td_.valueHeads);
        DataCopy(gmAPrime_[offset], ubAPrime_[slot], kMatrixElems);
        // This MTE3 event covers both T and APrime in the slot. It is consumed
        // by preprocess two pairs later, or by the chain-tail drain.
        SetFlag<HardEvent::MTE3_V>(kProdOutputMte3EventBase + slot);
        AivPublish(kProdAPrime);
    }

    __aicore__ inline void RunQK(int64_t chunk)
    {
        const uint32_t l1Slot = static_cast<uint32_t>((chunk >> 1) % 3);
        AicWaitOwner(chunk, kProdQ);
        ComputeMm(l1QHat_[l1Slot], l1K_[l1Slot], l0A0_, l0B1_, l0C1_, 64, 64, 128,
                  false, false);
        FixF32(ubQk_, l0C1_, 64, 64, (chunk & 1) != 0);
        ReleaseL0C();
        AicPublishToOwner(chunk, kProdQK);
    }

    __aicore__ inline void RunKkt(int64_t chunk)
    {
        const uint32_t l1Slot = static_cast<uint32_t>((chunk >> 1) % 3);
        AicWaitOwner(chunk, kProdK);
        ComputeMm(l1K_[l1Slot], l1K_[l1Slot], l0A0_, l0B0_, l0C0_, 64, 64, 128,
                  false, false);
        FixF32(ubKkt_, l0C0_, 64, 64, (chunk & 1) != 0);
        ReleaseL0C();
        AicPublishToOwner(chunk, kProdKkt);
    }

    FusedTilingData td_{};
    ChainLoc loc_{};
    uint32_t sub_{0};

    GlobalTensor<bfloat16_t> gmK_, gmQ_, gmT_, gmAPrime_;
    GlobalTensor<float> gmG_, gmBeta_;
    GlobalTensor<float> gmALog_, gmDtBias_;

    LocalTensor<bfloat16_t> ubK_[2], ubKHat_[2];
    LocalTensor<bfloat16_t> ubQ_[2], ubQHat_[2];
    LocalTensor<bfloat16_t> ubABf_[2], ubAPrime_[2];
    LocalTensor<float> ubKkt_, ubQk_, ubL_[2], ubX_[2], ubRatio_[2];
    LocalTensor<float> ubG_[3], ubEnergy_[3], ubBeta_[3], ubKRstd_[3], ubEnergyRecip_[3];
    LocalTensor<float> ubScratch_, ubEye_;
    LocalTensor<uint32_t> ubShift_;
    LocalTensor<bfloat16_t> l1K_[3], l1QHat_[3];
    LocalTensor<bfloat16_t> l0A0_, l0B0_, l0B1_;
    LocalTensor<float> l0C0_, l0C1_;
};

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_PRODUCER_H
