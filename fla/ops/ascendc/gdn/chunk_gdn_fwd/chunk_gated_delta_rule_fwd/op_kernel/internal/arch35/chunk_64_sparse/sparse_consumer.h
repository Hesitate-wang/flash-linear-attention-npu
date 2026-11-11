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

constexpr uint16_t kConsQKInputs = 0;
constexpr uint16_t kConsP = 1;
constexpr uint16_t kConsRight = 2;
constexpr uint16_t kConsWU = 3;
constexpr uint16_t kConsD = 4;
constexpr uint16_t kConsOL = 5;
constexpr uint16_t kConsQk = 6;
constexpr uint16_t kConsAPrime = 9;
constexpr uint16_t kConsS = 10;

class ConsumerRole {
public:
    __aicore__ inline void Init(const PhaseArgs &args, const FusedTilingData &td)
    {
        td_ = td;
        sub_ = GetSubBlockIdx();
        gmQ_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.q));
        gmK_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.k));
        gmV_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.v));
        gmT_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.a));
        gmG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.rawG));
        gmBeta_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.beta));
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

    __aicore__ inline void RunPrologue()
    {
        if (loc_.chunks == 0) {
            return;
        }
        window_ = ResolveWindow(loc_, 0);
        if ASCEND_IS_AIV {
            RunAivPrologue();
        }
        if ASCEND_IS_AIC {
            RunCubeQkt(0);
        }
    }

    __aicore__ inline void RunPrepare(int64_t chunk)
    {
        window_ = ResolveWindow(loc_, chunk);
        if ASCEND_IS_AIV {
            RunAivPrepare(chunk);
        }
        if ASCEND_IS_AIC {
            RunCubePrepare(chunk);
        }
    }

    __aicore__ inline void RunAfterWu(int64_t chunk)
    {
        window_ = ResolveWindow(loc_, chunk);
        if ASCEND_IS_AIV {
            RunAivAfterWu(chunk);
        }
        if ASCEND_IS_AIC {
            RunCubeAfterWu(chunk);
        }
    }

    __aicore__ inline void RunChunk(int64_t chunk)
    {
        RunPrepare(chunk);
        RunAfterWu(chunk);
    }

    __aicore__ inline void EndChain()
    {
        if ASCEND_IS_AIV {
            if (td_.storeFinalState != 0) {
                const int64_t offset = StateOffset();
                if (td_.stateFp32 != 0) {
                    DataCopy(gmFinalF_[offset], ubStateF_, kStateElems / 2);
                } else {
                    // The recurrent BF16 mirror is tight NZ for H uploads, but
                    // public final_state is ND. ubD_ is dead after StateAddD.
                    Bf16PanelsToNd(ubInitB_, ubStateB_);
                    SetFlag<HardEvent::V_MTE3>(0);
                    WaitFlag<HardEvent::V_MTE3>(0);
                    DataCopy(gmFinalB_[offset], ubInitB_, kStateElems / 2);
                }
            }
            // Serial mode reuses this UB for the next chain's Producer. Do not
            // let its Vector/MTE2 writes overtake the final-state MTE3 read.
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
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
        ubDecay_[0] = ubP_;
        ubDecay_[1] = ubP_[64];
        ubScratch_ = ubP_[256];
        ubOSB_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbOS);
        ubU_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbU);
        ubVNew_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbVNew);
        ubRight_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbRight);
        ubW_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbW);
        ubQ_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbQ);
        ubK_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbK);
        ubV_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbV);
        if (td_.useNorm != 0) {
            ubQHat_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbQHat);
            ubKHat_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbKHat);
        } else {
            ubQHat_ = ubQ_;
            ubKHat_ = ubK_;
        }
        ubQk_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kCUbQk);
        ubAPrime_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbAPrime);
        ubO_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbO);
        ubG_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kCUbGates);
        ubEnergy_ = ubG_[64];
        ubQRstd_ = ubG_[256];
        ubKRstd_ = ubG_[288];
        ubShift_ = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(kCUbShift);
        ubBeta_ = ubG_[128];

        l1QHat_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kCL1QHat);
        l1H_[0] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kCL1H0);
        l1H_[1] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kCL1H1);
        l1VNew_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kCL1VNew);
        l1Right_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kCL1Right);
        l1T_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kCL1T);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            l1W_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kCL1W + slot * kCL1Slot16);
            l1U_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kCL1U + slot * kCL1Slot16);
            l1QExp_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kCL1QExp + slot * kCL1Slot16);
            l1K_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kCL1K + slot * kCL1Slot16);
            l1APrime_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(
                kCL1APrime + slot * kCL1APrimeSlotBytes);
        }
        l0A0_ = buf.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(kL0Bank0);
        l0A1_ = buf.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(kL0Bank1);
        l0B0_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0Bank0);
        l0B1_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0Bank1);
        l0C0_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C0);
        l0C1_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C1);
        l0C2_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C2);
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
                DataCopy(ubStateF_, gmInitF_[StateOffset()], kStateElems / 2);
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                StateMirrorVF(ubStateF_, ubStateB_);
            } else {
                DataCopy(ubInitB_, gmInitB_[StateOffset()], kStateElems / 2);
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                StateWidenVF(ubInitB_, ubStateF_, ubStateB_);
            }
        } else {
            Duplicate(ubStateF_, 0.0f, kStateElems / 2);
            Duplicate(ubStateB_, static_cast<bfloat16_t>(0), kStateElems / 2);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
    }

    __aicore__ inline void RunAivPrologue()
    {
        PrepareNext(0);
        PrepareNextVbKbg(0);
        BuildAPrime(0);
    }

    __aicore__ inline void RunAivPrepare(int64_t chunk)
    {
        if (chunk + 1 < loc_.chunks) {
            PrepareNext(chunk + 1);
        }
        PrepareCurrentState(chunk);
        if (chunk + 1 < loc_.chunks) {
            // Place the next Vb/Kbg between the current S stage and the next
            // APrime, overlapping the current QS/WU Cube work.
            PrepareNextVbKbg(chunk + 1);
            BuildAPrime(chunk + 1);
        }
    }

    __aicore__ inline void PrepareCurrentState(int64_t chunk)
    {
        window_ = ResolveWindow(loc_, chunk);
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        if (window_.rows < kChunk) {
            // LoadT reads only valid rows. Clear the padded L1 rows before the
            // kConsS publication, which precedes this chunk's mode-0/LoadT.
            Duplicate(ubAPrime_, static_cast<bfloat16_t>(0), kHalfRows * kChunk);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            CopyNzToL1(l1T_, ubAPrime_, kHalfRows, kChunk, kChunk, firstRow);
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
        }
        // Cube keeps using the old S in L1H, so the FP32 UB state can be
        // decayed immediately after this upload without changing QS/W@S.
        CopyNzToL1(l1H_[slot], ubStateB_, kHalfDim, kDim, kDim,
                   sub_ * static_cast<uint32_t>(kHalfDim));
        AivPublish(kConsS);
        StateDecayVF(ubStateF_, chunkDecay_[slot]);
    }

    __aicore__ inline void BuildAPrime(int64_t chunk)
    {
        window_ = ResolveWindow(loc_, chunk);
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        AivWait(kConsQk);
        if (td_.useExp2 != 0) {
            APrimeBandNzVF<true>(ubQk_, ubAPrime_, ubG_, ubEnergy_, gateRatioSafe_, firstRow);
        } else {
            APrimeBandNzVF<false>(ubQk_, ubAPrime_, ubG_, ubEnergy_, gateRatioSafe_, firstRow);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyNzToL1(l1APrime_[slot], ubAPrime_, kHalfRows, kChunk, kChunk, firstRow);
        AivPublish(kConsAPrime);
    }

    __aicore__ inline void RunAivAfterWu(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        const int64_t remaining = window_.rows - static_cast<int64_t>(firstRow);
        const uint32_t rows = remaining <= 0 ? 0U
            : static_cast<uint32_t>(remaining < kHalfRows ? remaining : kHalfRows);
        AivWait(kConsP);
        SetFlag<HardEvent::V_S>(1);
        WaitFlag<HardEvent::V_S>(1);
        const bool directRightNz = (td_.reserved & kSparseDebugNdVnRight) == 0;
        // decay was prepared with this chunk's QExp, so v_new only needs a
        // multiply and NZ store before the right upload.
        RightNzVF(ubVNew_, ubRight_, ubDecay_[slot], firstRow, directRightNz);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        const int64_t token = window_.tokenStart + firstRow;
        if (directRightNz) {
            CopyNzToL1(l1Right_, ubRight_, kHalfRows, kDim, kChunk, firstRow);
        } else {
            UploadNdToL1(l1Right_, ubRight_, kHalfRows, kDim, kChunk, firstRow);
        }
        AivPublish(kConsRight);

        if (td_.reserved == 4) {
            AivWait(kConsOL);
            ubScratch_.SetValue(0, debugEnergy_[slot]);
            ubScratch_.SetValue(1, td_.scale);
            ubScratch_.SetValue(2, debugG_[slot]);
            ubScratch_.SetValue(3, chunkDecay_[slot]);
            SetFlag<HardEvent::S_V>(0);
            WaitFlag<HardEvent::S_V>(0);
            Cast(ubO_, ubScratch_, RoundMode::CAST_RINT, 4);
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
        StateAddDVF(ubStateF_, ubStateB_, ubD_);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
    }

    __aicore__ inline void PrepareNext(int64_t chunk)
    {
        window_ = ResolveWindow(loc_, chunk);
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
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

        // Q/K are the first complete unit. Publish them before gate, QExp,
        // decay and Vb/Kbg so the AIC can start QKT immediately.
        CopyGmToUb(ubQ_, gmQ_, OffsetBhtd(loc_.batch, loc_.kh, token, td_.keyHeads, td_.tokens), elems);
        SetFlag<HardEvent::MTE2_V>(1);
        WaitFlag<HardEvent::MTE2_V>(1);
        if (td_.useNorm != 0) {
            NormalizeRowsNzVF<true, true>(ubQ_, ubQHat_, ubQ_, ubQRstd_);
        }
        CopyGmToUb(ubK_, gmK_, OffsetBhtd(loc_.batch, loc_.kh, token, td_.keyHeads, td_.tokens), elems);
        SetFlag<HardEvent::MTE2_V>(2);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        if (td_.useNorm != 0) {
            CopyNzToL1(l1QHat_, ubQHat_, kHalfRows, kDim, kChunk, firstRow);
        } else {
            UploadNdToL1(l1QHat_, ubQHat_, kHalfRows, kDim, kChunk, firstRow);
        }

        WaitFlag<HardEvent::MTE2_V>(2);
        if (td_.useNorm != 0) {
            NormalizeRowsNzVF<true, true>(ubK_, ubKHat_, ubK_, ubKRstd_);
        }
        SetFlag<HardEvent::V_MTE3>(1);
        WaitFlag<HardEvent::V_MTE3>(1);
        if (td_.useNorm != 0) {
            CopyNzToL1(l1K_[slot], ubKHat_, kHalfRows, kDim, kChunk, firstRow);
        } else {
            UploadNdToL1(l1K_[slot], ubKHat_, kHalfRows, kDim, kChunk, firstRow);
        }
        AivPublish(kConsQKInputs);

        CopyGmToUb(ubG_, gmG_, OffsetBht(loc_.batch, loc_.hv, window_.tokenStart,
                                          td_.valueHeads, td_.tokens),
                   static_cast<uint32_t>(window_.rows));
        CopyGmToUb(ubBeta_, gmBeta_, OffsetBht(loc_.batch, loc_.hv, window_.tokenStart,
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
        PrepareGateBetaVF(ubG_, ubBeta_, aLog, dtBias, td_.gateSoftplus != 0,
                          td_.betaSigmoid != 0, td_.allowNegEigval != 0 ? 2.0f : 1.0f);
        if (td_.useExp2 != 0) {
            CumsumEnergyVF<true>(ubG_, ubEnergy_, ubShift_);
        } else {
            CumsumEnergyVF<false>(ubG_, ubEnergy_, ubShift_);
        }

        // V overlaps gate/QExp/decay. Only Vb waits for its MTE2 event.
        CopyGmToUb(ubV_, gmV_, OffsetBhtd(loc_.batch, loc_.hv, token, td_.valueHeads, td_.tokens), elems);
        SetFlag<HardEvent::MTE2_V>(3);
        SetFlag<HardEvent::V_S>(1);
        WaitFlag<HardEvent::V_S>(1);
        gateRatioSafe_ = GateRatioSafe(ubG_, ubScratch_, static_cast<uint32_t>(window_.rows));
        const int64_t lastIndex = window_.rows - 1;
        const float gLast = ubG_.GetValue(lastIndex);
        const float lastEnergy = ubEnergy_.GetValue(lastIndex);
        chunkDecay_[slot] = lastEnergy;
        debugEnergy_[slot] = ubEnergy_.GetValue(firstRow);
        debugG_[slot] = ubG_.GetValue(firstRow);
        if (td_.useExp2 != 0) {
            DecayVF<true>(ubDecay_[slot], ubG_, ubEnergy_, gLast, lastEnergy,
                          gateRatioSafe_, firstRow);
        } else {
            DecayVF<false>(ubDecay_[slot], ubG_, ubEnergy_, gLast, lastEnergy,
                           gateRatioSafe_, firstRow);
        }
        QExpNzVF(ubQ_, ubOSB_, ubEnergy_, firstRow);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyNzToL1(l1QExp_[slot], ubOSB_, kHalfRows, kDim, kChunk, firstRow);
    }

    __aicore__ inline void PrepareNextVbKbg(int64_t chunk)
    {
        window_ = ResolveWindow(loc_, chunk);
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        ScaleKbgNzVF(ubK_, ubW_, ubBeta_, ubEnergy_, firstRow, true);
        WaitFlag<HardEvent::MTE2_V>(3);
        ScaleVbNzVF(ubV_, ubU_, ubBeta_, firstRow);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyNzToL1(l1U_[slot], ubU_, kHalfRows, kDim, kChunk, firstRow);
        CopyNzToL1(l1W_[slot], ubW_, kHalfRows, kDim, kChunk, firstRow);
        AivPublish(kConsWU);
    }

    __aicore__ inline void LoadT()
    {
        Nd2NzParams params{};
        params.ndNum = 1;
        params.nValue = static_cast<uint32_t>(window_.rows);
        params.dValue = 64;
        params.srcDValue = 64;
        params.srcNdMatrixStride = 0;
        params.dstNzC0Stride = 64;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 0;
        const int64_t offset = OffsetBhtd(loc_.batch, loc_.hv, window_.tokenStart,
                                          td_.valueHeads, td_.tokens, kChunk);
        DataCopy(l1T_, gmT_[offset], params);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
    }

    __aicore__ inline void RunCubeQkt(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        AicWaitBoth(kConsQKInputs);
        ComputeMm(l1QHat_, l1K_[slot], l0A0_, l0B1_, l0C1_, 64, 64, 128, false, false);
        FixF32Halves(ubQk_, l0C1_, 64, 64);
        ReleaseL0C();
        AicPublishToBoth(kConsQk);
    }

    __aicore__ inline void RunCubePrepare(int64_t chunk)
    {
        if (chunk + 1 < loc_.chunks) {
            RunCubeQkt(chunk + 1);
        }
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        // QExp was prepared with the next-chunk inputs in the previous round;
        // only the current S publication remains on the QS critical path.
        AicWaitBoth(kConsS);
        ComputeMm(l1QExp_[slot], l1H_[slot], l0A0_, l0B0_, l0C3_, 64, 128, 128,
                  false, td_.stateVFirst == 0);
        // No Fixpipe follows QS. This is only the FIX_M event priming required
        // by the next independent L0C2 GEMM; the QS accumulator stays live.
        ReleaseL0C();
    }

    __aicore__ inline void RunCubeAfterWu(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        LoadT();
        AicWaitBoth(kConsWU);
        // Kbg is negated: W = -trueW. U starts in L0C and W@S accumulates
        // into it, so the final accumulator is U - trueW@S = v_new.
        ComputeMm(l1T_, l1W_[slot], l0A0_, l0B1_, l0C2_, 64, 128, 64, false, true);
        FixB16L1Matrix(l1W_[slot], l0C2_, 64, 128, 64);
        ReleaseL0C();

        ComputeMm(l1T_, l1U_[slot], l0A0_, l0B1_, l0C0_, 64, 128, 64, false, true);
        // B0 still holds S from the early QS GEMM; only W needs an MTE1 load.
        ComputeMmReuseB(l1W_[slot], l0A0_, l0B0_, l0C0_, 64, 128, 128, false, false);
        // O reads the AIC-local L1 copy. AIV reads only the independent UB copy
        // to produce right=v_new_decay, leaving ubU_ free for the next Vb.
        FixB16L1Matrix(l1VNew_, l0C0_, 64, 128, 64);
        FixB16Halves(ubVNew_, l0C0_, 64, 128);
        ReleaseL0C();
        AicPublishToBoth(kConsP);

        AicWaitBoth(kConsAPrime);
        // Cube-local publication of L1 vNew; kConsP is only the AIC->AIV
        // publication of UB v_new and is not used as the O dependency.
        SetFlag<HardEvent::FIX_MTE1>(0);
        WaitFlag<HardEvent::FIX_MTE1>(0);
        // Consume the v_new Fixpipe event before O's ReleaseL0C.
        WaitFlag<HardEvent::FIX_M>(0);
        ComputeMm(l1APrime_[slot], l1VNew_, l0A1_, l0B1_, l0C3_, 64, 128, 64,
                  false, true, false);
        FixB16ScaledHalves(ubO_, l0C3_, 64, 128, td_.scale);
        ReleaseL0C();
        AicPublishToBoth(kConsOL);

        // O overlaps the AIV right computation. Only D waits for right.
        AicWaitBoth(kConsRight);
        if (td_.stateVFirst != 0) {
            ComputeMm(l1Right_, l1K_[slot], l0A0_, l0B0_, l0C3_, 128, 128, 64, true, true);
        } else {
            ComputeMm(l1K_[slot], l1Right_, l0A0_, l0B0_, l0C3_, 128, 128, 64, true, true);
        }
        FixF32Halves(ubD_, l0C3_, 128, 128);
        ReleaseL0C();
        AicPublishToBoth(kConsD);
    }

    FusedTilingData td_{};
    ChainLoc loc_{};
    ChunkWindow window_{};
    uint32_t sub_{0};
    bool hasInitial_{false};
    bool gateRatioSafe_{false};
    float chunkDecay_[2]{};
    float debugEnergy_[2]{};
    float debugG_[2]{};

    GlobalTensor<bfloat16_t> gmQ_, gmK_, gmV_, gmT_, gmO_;
    GlobalTensor<bfloat16_t> gmInitB_, gmFinalB_;
    GlobalTensor<float> gmG_, gmBeta_, gmInitF_, gmFinalF_, gmALog_, gmDtBias_;

    LocalTensor<float> ubStateF_, ubD_, ubP_, ubQk_, ubQRstd_, ubKRstd_, ubScratch_;
    LocalTensor<float> ubDecay_[2];
    LocalTensor<bfloat16_t> ubStateB_, ubU_, ubVNew_, ubRight_, ubW_, ubQ_, ubK_, ubV_;
    LocalTensor<bfloat16_t> ubQHat_, ubKHat_, ubAPrime_, ubO_, ubInitB_, ubOSB_;
    LocalTensor<float> ubG_, ubEnergy_, ubBeta_;
    LocalTensor<uint32_t> ubShift_;
    LocalTensor<bfloat16_t> l1W_[2], l1U_[2], l1QHat_, l1QExp_[2], l1K_[2], l1H_[2];
    LocalTensor<bfloat16_t> l1VNew_, l1Right_, l1APrime_[2], l1T_;
    LocalTensor<bfloat16_t> l0A0_, l0A1_, l0B0_, l0B1_;
    LocalTensor<float> l0C0_, l0C1_, l0C2_, l0C3_;
};

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_CONSUMER_H
