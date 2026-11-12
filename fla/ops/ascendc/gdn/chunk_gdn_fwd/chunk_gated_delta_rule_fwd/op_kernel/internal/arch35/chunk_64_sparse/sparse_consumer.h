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

constexpr uint16_t kConsP = 6; // Continuous GM->L1 DataCopy reserves mode-4 flags 1/17.
constexpr uint16_t kConsRight = 2;
constexpr uint16_t kConsW = 3;
constexpr uint16_t kConsD = 4;
constexpr uint16_t kConsOL = 5;
constexpr uint16_t kConsS = 10;
constexpr uint16_t kConsU = 11;

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
        gmAPrime_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            GetUserWorkspace(args.workspace) + td.workspaceOffset));
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
        // Non-empty chains stage initial-state load/conversion inside the
        // prologue so it overlaps Q/K/V. Empty chains have no such pipeline,
        // but EndChain still needs a valid initial/final state.
        if ASCEND_IS_AIV {
            if (loc_.chunks == 0) {
                InitState();
            }
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
            RunCubeQs(0);
        }
    }

    __aicore__ inline void RunPrepare(int64_t chunk)
    {
        window_ = ResolveWindow(loc_, chunk);
        if ASCEND_IS_AIV {
            RunAivPrepare(chunk);
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
            // let its Vector or MTE2 writes overtake the final-state MTE3 read.
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
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
        ubZeroBf_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbZero);
        ubO_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbO);
        ubG_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kCUbGates);
        ubEnergy_ = ubG_[64];
        ubShift_ = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(kCUbShift);
        ubBeta_ = ubG_[128];

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

    __aicore__ inline void LoadInitialState()
    {
        if (hasInitial_) {
            if (td_.stateFp32 != 0) {
                DataCopy(ubStateF_, gmInitF_[StateOffset()], kStateElems / 2);
            } else {
                DataCopy(ubInitB_, gmInitB_[StateOffset()], kStateElems / 2);
            }
            // Initial-state MTE2 is independent of Q/K/V events 1/2/3.
            SetFlag<HardEvent::MTE2_V>(4);
        } else {
            Duplicate(ubStateF_, 0.0f, kStateElems / 2);
            Duplicate(ubStateB_, static_cast<bfloat16_t>(0), kStateElems / 2);
        }
    }

    __aicore__ inline void ConvertInitialState()
    {
        if (hasInitial_) {
            WaitFlag<HardEvent::MTE2_V>(4);
            if (td_.stateFp32 != 0) {
                StateMirrorVF(ubStateF_, ubStateB_);
            } else {
                StateWidenVF(ubInitB_, ubStateF_, ubStateB_);
            }
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
    }

    __aicore__ inline void RunAivPrologue()
    {
        // In serial mode this AIV has just finished the Producer role. Its
        // last MTE3 reads must complete before Consumer Vector work reuses UB.
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        // H is independent of Q/K/V preparation. Publish it first so QS can
        // start immediately after QExp is ready, then overlap QKT with Vb.
        PrepareNextGate(0, true);
        PrepareNextQAndQExp(0);
        ConvertInitialState(); //格式cast
        PrepareCurrentState(0, false);
        // chunkDecay_ is produced by PrepareNextQAndQExp. H was already published,
        // so this decay cannot change the S used by QS/W@S.
        StateDecayVF(ubStateF_, chunkDecay_[0]);
        PrepareNextKAndKbg(0);
        PrepareNextVb(0);
        PrepareRightDecay(0);
    }

    __aicore__ inline void RunAivPrepare(int64_t chunk)
    {
        if (chunk + 1 < loc_.chunks) {
            PrepareNextGate(chunk + 1);
            PrepareNextQAndQExp(chunk + 1);
        }
    }

    __aicore__ inline void PrepareCurrentState(int64_t chunk, bool applyDecay = true)
    {
        window_ = ResolveWindow(loc_, chunk);
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        if (window_.rows < kChunk) {
            // LoadT reads only valid rows. Clear the padded L1 rows before the
            // kConsS publication, which precedes this chunk's mode-0/LoadT.
            Duplicate(ubZeroBf_, static_cast<bfloat16_t>(0), kHalfRows * kChunk);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            CopyNzToL1(l1T_, ubZeroBf_, kHalfRows, kChunk, kChunk, firstRow);
        }
        // Cube keeps using the old S in L1H, so the FP32 UB state can be
        // decayed immediately after this upload without changing QS/W@S.
        CopyNzToL1(l1H_[slot], ubStateB_, kHalfDim, kDim, kDim,
                   sub_ * static_cast<uint32_t>(kHalfDim));
        // PrepareNextGate publishes gate/energy on V_S event 1. Consume it
        // before the scalar read; PrepareRightDecay reuses the same values later.
        WaitFlag<HardEvent::V_S>(1);
        chunkDecay_[slot] = ubEnergy_.GetValue(window_.rows - 1);
        AivPublish(kConsS);
        if (applyDecay) {
            StateDecayVF(ubStateF_, chunkDecay_[slot]);
        }
    }

    __aicore__ inline void RunAivAfterWu(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        const int64_t remaining = window_.rows - static_cast<int64_t>(firstRow);
        const uint32_t rows = remaining <= 0 ? 0U
            : static_cast<uint32_t>(remaining < kHalfRows ? remaining : kHalfRows);
        AivWait(kConsP);
        // PrepareRightDecay published ubDecay immediately after writing it.
        WaitFlag<HardEvent::V_S>(2);
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

        AivWaitMte3(kConsOL);
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

        if (chunk + 1 < loc_.chunks) {
            PrepareNextKAndKbg(chunk + 1);
        }
        AivWait(kConsD);
        StateAddDVF(ubStateF_, ubStateB_, ubD_);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        if (chunk + 1 < loc_.chunks) {
            // Publish the next H before its QS, then leave Vb as the final
            // independent Vector stage of this round.
            PrepareCurrentState(chunk + 1);
            PrepareNextVb(chunk + 1);
            PrepareRightDecay(chunk + 1);
        }
    }

    __aicore__ inline void PrepareNextGate(int64_t chunk, bool includeInitial = false)
    {
        window_ = ResolveWindow(loc_, chunk);
        if (td_.useNorm == 0) {
            // Without Norm, ubKHat_ aliases ubK_ and UploadNdToL1 reads that
            // raw UB asynchronously. Drain it before tail clearing or reloading.
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }
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

        // Publish gate/energy completion without blocking the Vector pipeline.
        // The matching wait is placed only when Scalar gate reads are needed.
        SetFlag<HardEvent::V_S>(1);

        // Issue Q/K/V while gate/cumsum and the scalar-safe checks retire.
        // Their consumers below only wait for the corresponding event.
        CopyGmToUb(ubQ_, gmQ_, OffsetBhtd(loc_.batch, loc_.kh, token, td_.keyHeads, td_.tokens), elems);
        SetFlag<HardEvent::MTE2_V>(1);
        CopyGmToUb(ubK_, gmK_, OffsetBhtd(loc_.batch, loc_.kh, token, td_.keyHeads, td_.tokens), elems);
        SetFlag<HardEvent::MTE2_V>(2);
        CopyGmToUb(ubV_, gmV_, OffsetBhtd(loc_.batch, loc_.hv, token, td_.valueHeads, td_.tokens), elems);
        SetFlag<HardEvent::MTE2_V>(3);
        if (includeInitial) {
            LoadInitialState();
        }
    }

    __aicore__ inline void PrepareNextQAndQExp(int64_t chunk)
    {
        window_ = ResolveWindow(loc_, chunk);
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        WaitFlag<HardEvent::MTE2_V>(1);
        if (td_.useNorm != 0) {
            QNormExpNzVF(ubQ_, ubQHat_, ubOSB_, ubEnergy_, firstRow);
        } else {
            QExpNzVF(ubQ_, ubOSB_, ubEnergy_, firstRow);
        }
        // QExp does not depend on right's decay. Upload it first so the
        // following scalar checks and decay work overlap this MTE3 transfer.
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyNzToL1(l1QExp_[slot], ubOSB_, kHalfRows, kDim, kChunk, firstRow);
    }

    __aicore__ inline void PrepareRightDecay(int64_t chunk)
    {
        window_ = ResolveWindow(loc_, chunk);
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        // This is needed only by the next steady right stage, so defer it
        // after Q/K/Kbg/Vb and let it overlap the AIC's QS work. Gate/energy
        // V_S event 1 was already consumed by PrepareCurrentState.
        gateRatioSafe_ = GateRatioSafe(ubG_, ubScratch_, static_cast<uint32_t>(window_.rows));
        const float gLast = ubG_.GetValue(window_.rows - 1);
        const float lastEnergy = chunkDecay_[slot];
        if (td_.useExp2 != 0) {
            DecayVF<true>(ubDecay_[slot], ubG_, ubEnergy_, gLast, lastEnergy,
                          gateRatioSafe_, firstRow);
        } else {
            DecayVF<false>(ubDecay_[slot], ubG_, ubEnergy_, gLast, lastEnergy,
                           gateRatioSafe_, firstRow);
        }
        // Publish decay now rather than draining all Vector work at its use.
        SetFlag<HardEvent::V_S>(2);
    }

    __aicore__ inline void PrepareNextKAndKbg(int64_t chunk)
    {
        window_ = ResolveWindow(loc_, chunk);
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        WaitFlag<HardEvent::MTE2_V>(2);
        if (td_.useNorm != 0) {
            KNormKbgNzVF(ubK_, ubKHat_, ubW_, ubBeta_, ubEnergy_, firstRow);
        } else {
            ScaleKbgNzVF(ubK_, ubW_, ubBeta_, ubEnergy_, firstRow, true);
        }
        SetFlag<HardEvent::V_MTE3>(1);
        WaitFlag<HardEvent::V_MTE3>(1);
        if (td_.useNorm != 0) {
            CopyNzToL1(l1K_[slot], ubKHat_, kHalfRows, kDim, kChunk, firstRow);
        } else {
            UploadNdToL1(l1K_[slot], ubKHat_, kHalfRows, kDim, kChunk, firstRow);
        }
        CopyNzToL1(l1W_[slot], ubW_, kHalfRows, kDim, kChunk, firstRow);
        AivPublish(kConsW);
    }

    __aicore__ inline void PrepareNextVb(int64_t chunk)
    {
        window_ = ResolveWindow(loc_, chunk);
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t firstRow = sub_ * static_cast<uint32_t>(kHalfRows);
        WaitFlag<HardEvent::MTE2_V>(3);
        ScaleVbNzVF(ubV_, ubU_, ubBeta_, firstRow);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyNzToL1(l1U_[slot], ubU_, kHalfRows, kDim, kChunk, firstRow);
        AivPublish(kConsU);
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

    __aicore__ inline void LoadAPrime(int64_t chunk)
    {
        if (td_.schedule == kSparseSchedulePipeline) {
            // Mode 0 requires every AIC to set as well as wait. Delay this
            // publish until O so T, W and v_new do not wait for APrime.
            PublishAPrimeAic<PIPE_MTE2>(chunk);
        }
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const int64_t tile = TileIndex(loc_, chunk, td_.chunksPerBatch, td_.varlen != 0);
        const int64_t offset = OffsetMatrixTile(tile, loc_.hv, td_.valueHeads);
        // The Producer writes a complete 64x64 NZ tile, so no ND conversion or
        // per-half copies are needed here.
        DataCopy(l1APrime_[slot], gmAPrime_[offset], kMatrixElems);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
    }

    __aicore__ inline void RunCubeQs(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        AicWaitBoth(kConsS);
        ComputeMm(l1QExp_[slot], l1H_[slot], l0A0_, l0B0_, l0C3_, 64, 128, 128,
                  false, td_.stateVFirst == 0);
        // Keep QS in L0C3 for the next O and S in B0 for the next W@S. This
        // SetFlag only primes the next independent T@Kbg operation.
        ReleaseL0C();
    }

    __aicore__ inline void RunCubeAfterWu(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        LoadT();
        AicWaitBoth(kConsW);
        // Kbg is negated, so W=T@Kbg=-trueW. The previous QS already loaded
        // this chunk's S into B0; T@Kbg uses B1 and preserves it.
        ComputeMm(l1T_, l1W_[slot], l0A0_, l0B1_, l0C2_, 64, 128, 64, false, true);
        FixB16L1Matrix(l1W_[slot], l0C2_, 64, 128, 64);
        SetFlag<HardEvent::FIX_MTE1>(0);
        WaitFlag<HardEvent::FIX_MTE1>(0);
        ReleaseL0C();
        ComputeMmReuseB(l1W_[slot], l0A0_, l0B0_, l0C0_, 64, 128, 128, false, true);

        AicWaitBoth(kConsU);
        ComputeMm(l1T_, l1U_[slot], l0A0_, l0B1_, l0C0_, 64, 128, 64, false, true, false);
        FixB16L1Matrix(l1VNew_, l0C0_, 64, 128, 64);
        FixB16Halves(ubVNew_, l0C0_, 64, 128);
        ReleaseL0C();
        AicPublishToBoth(kConsP);

        // O accumulates into the QS retained from the previous round. APrime
        // has an independent mode-0 publication and is loaded only here.
        LoadAPrime(chunk);
        SetFlag<HardEvent::FIX_MTE1>(0);
        WaitFlag<HardEvent::FIX_MTE1>(0);
        WaitFlag<HardEvent::FIX_M>(0);
        ComputeMm(l1APrime_[slot], l1VNew_, l0A1_, l0B1_, l0C3_, 64, 128, 64,
                  false, true, false);
        FixB16ScaledHalves(ubO_, l0C3_, 64, 128, td_.scale);
        ReleaseL0C();
        AicPublishToBoth(kConsOL);

        AicWaitBoth(kConsRight);
        if (td_.stateVFirst != 0) {
            ComputeMm(l1Right_, l1K_[slot], l0A0_, l0B0_, l0C3_, 128, 128, 64, true, true);
        } else {
            ComputeMm(l1K_[slot], l1Right_, l0A0_, l0B0_, l0C3_, 128, 128, 64, true, true);
        }
        FixF32Halves(ubD_, l0C3_, 128, 128);
        ReleaseL0C();
        AicPublishToBoth(kConsD);

        // The following QS is retained for the next O, while its B0 copy of S
        // is reused by the next W@S.
        if (chunk + 1 < loc_.chunks) {
            RunCubeQs(chunk + 1);
        }
    }

    FusedTilingData td_{};
    ChainLoc loc_{};
    ChunkWindow window_{};
    uint32_t sub_{0};
    bool hasInitial_{false};
    bool gateRatioSafe_{false};
    float chunkDecay_[2]{};

    GlobalTensor<bfloat16_t> gmQ_, gmK_, gmV_, gmT_, gmO_, gmAPrime_;
    GlobalTensor<bfloat16_t> gmInitB_, gmFinalB_;
    GlobalTensor<float> gmG_, gmBeta_, gmInitF_, gmFinalF_, gmALog_, gmDtBias_;

    LocalTensor<float> ubStateF_, ubD_, ubP_, ubScratch_;
    LocalTensor<float> ubDecay_[2];
    LocalTensor<bfloat16_t> ubStateB_, ubU_, ubVNew_, ubRight_, ubW_, ubQ_, ubK_, ubV_;
    LocalTensor<bfloat16_t> ubQHat_, ubKHat_, ubZeroBf_, ubO_, ubInitB_, ubOSB_;
    LocalTensor<float> ubG_, ubEnergy_, ubBeta_;
    LocalTensor<uint32_t> ubShift_;
    LocalTensor<bfloat16_t> l1W_[2], l1U_[2], l1QExp_[2], l1K_[2], l1H_[2];
    LocalTensor<bfloat16_t> l1VNew_, l1Right_, l1APrime_[2], l1T_;
    LocalTensor<bfloat16_t> l0A0_, l0A1_, l0B0_, l0B1_;
    LocalTensor<float> l0C0_, l0C2_, l0C3_;
};

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_CONSUMER_H
