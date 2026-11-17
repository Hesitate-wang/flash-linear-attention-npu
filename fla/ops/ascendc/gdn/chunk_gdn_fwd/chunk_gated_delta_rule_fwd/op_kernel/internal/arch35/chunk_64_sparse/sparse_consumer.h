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

constexpr uint16_t kConsRight = 2;
constexpr uint16_t kConsW = 3;
constexpr uint16_t kConsD = 4;
constexpr uint16_t kConsOL = 5;
constexpr uint16_t kConsP = 6; // Continuous GM->L1 DataCopy reserves mode-4 flags 1/17.
constexpr uint16_t kConsD1 = 7;
constexpr uint16_t kConsRight1 = 8;
constexpr uint16_t kConsP1 = 9;
constexpr uint16_t kConsS = 10;
constexpr uint16_t kConsU = 11;
constexpr uint16_t kConsOL1 = 12;
constexpr uint16_t kConsS1 = 13;

class ConsumerRole {
public:
    __aicore__ inline void Init(const PhaseArgs &args, const FusedTilingData &td)
    {
        td_ = td;
        sub_ = GetSubBlockIdx();
        gmV_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.v));
        gmT_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.a));
        gmAPrime_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            GetUserWorkspace(args.workspace) + td.workspaceOffset));
        gmNormQ_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            GetUserWorkspace(args.workspace) + td.normalizedQKOffset));
        gmNormK_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            GetUserWorkspace(args.workspace) + td.normalizedQKOffset +
            td.totalChunks * td.keyHeads * kChunkElems * sizeof(bfloat16_t)));
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
            // Prime both K UB slots before the chunk loop. The first loads
            // consume these flags; later loads consume the prior L1 uploads.
            SetFlag<HardEvent::MTE3_MTE2>(1);
            SetFlag<HardEvent::MTE3_MTE2>(2);
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
        if ASCEND_IS_AIV {
            RunAivPrologue();
        }
        if ASCEND_IS_AIC {
            RunCubeQs(0);
        }
    }

    __aicore__ inline void RunPrepare(int64_t chunk)
    {
        if ASCEND_IS_AIV {
            RunAivPrepare(chunk);
        }
    }

    __aicore__ inline void RunAfterWu(int64_t chunk)
    {
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

    __aicore__ inline void JoinNormalizedQKGeneration(int64_t chunk)
    {
        if ASCEND_IS_AIV {
            if ((chunk & 1) == 0) {
                WaitNormalizedQK(chunk);
            }
        }
    }

    __aicore__ inline void EndChain()
    {
        if ASCEND_IS_AIV {
            for (uint32_t slot = 0; slot < 2; ++slot) {
                WaitFlag<HardEvent::MTE3_MTE2>(1 + slot);
            }
            WaitUbRelease(ubOSBArmed_, 1);
            WaitUbRelease(ubWArmed_, 2);
            WaitUbRelease(ubUArmed_, 3);
            WaitUbRelease(ubRightArmed_, 4);
            WaitUbRelease(ubStateBArmed_, 5);
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
            // Drain the final-state source before the kernel exits.
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
        // These UB values cross the AIC Fixpipe/AIV Vector boundary.
        ubD_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kCUbD);
        ubVNew_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbVNew);
        ubO_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbO);

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

        if ASCEND_IS_AIV {
            ubStateF_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kCUbStateF);
            ubStateB_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbStateB);
            ubInitB_ = ubD_.ReinterpretCast<bfloat16_t>();
            ubP_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kCUbP);
            ubDecay_[0] = ubP_;
            ubDecay_[1] = ubP_[64];
            ubScratch_ = ubP_[256];
            ubOSB_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbOS);
            ubU_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbU);
            ubRight_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbRight);
            ubW_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbW);
            ubQ_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbQ);
            ubK_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbK);
            ubV_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbV);
            ubQHat_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbQHat);
            ubKHat_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbKHat);
            ubZeroBf_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kCUbZero);
            for (uint32_t gateSlot = 0; gateSlot < 3; ++gateSlot) {
                ubG_[gateSlot] = buf.GetBuffer<BufferType::ASCEND_UB, float>(
                    kCUbGates + gateSlot * kCUbGateSlotBytes);
                ubEnergy_[gateSlot] = ubG_[gateSlot][64];
                ubBeta_[gateSlot] = ubG_[gateSlot][128];
                ubEnergyRecip_[gateSlot] = ubG_[gateSlot][320];
            }
            ubShift_ = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(kCUbShift);
        }

        if ASCEND_IS_AIC {
            l0A0_ = buf.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(kL0Bank0);
            l0A1_ = buf.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(kL0Bank1);
            l0B0_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0Bank0);
            l0B1_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0Bank1);
            l0C0_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C0);
            l0C2_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C2);
            l0C3_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C3);
        }
    }

    __aicore__ inline int64_t StateOffset() const
    {
        return (loc_.seq * td_.valueHeads + loc_.hv) * kStateElems + sub_ * (kStateElems / 2);
    }

    __aicore__ inline void WaitUbRelease(bool &armed, int32_t eventId)
    {
        if (armed) {
            WaitFlag<HardEvent::MTE3_V>(eventId);
            armed = false;
        }
    }

    __aicore__ inline void SetUbRelease(bool &armed, int32_t eventId)
    {
        SetFlag<HardEvent::MTE3_V>(eventId);
        armed = true;
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

    __aicore__ inline void LoadInitialState(int32_t eventId = 2)
    {
        if (hasInitial_) {
            if (td_.stateFp32 != 0) {
                DataCopy(ubStateF_, gmInitF_[StateOffset()], kStateElems / 2);
            } else {
                DataCopy(ubInitB_, gmInitB_[StateOffset()], kStateElems / 2);
            }
            // Initial-state MTE2 uses event 2; Q/K/V and gate use 0/1, 4/5,
            // 3 and 0 respectively.
            SetFlag<HardEvent::MTE2_V>(eventId);
        } else {
            Duplicate(ubStateF_, 0.0f, kStateElems / 2);
            Duplicate(ubStateB_, static_cast<bfloat16_t>(0), kStateElems / 2);
        }
    }

    __aicore__ inline void ConvertInitialState(int32_t eventId = 2)
    {
        if (hasInitial_) {
            WaitFlag<HardEvent::MTE2_V>(eventId);
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
        // H preparation starts after the previous chain's EndChain drain.
        // H is independent of Q/K/V preparation. Publish it first so QS can
        // start immediately after QExp is ready, then overlap QKT with Vb.
        IssueGateInputs(0);
        if (loc_.chunks > 1) {
            IssueGateInputs(1);
        }
        SetFlag<HardEvent::MTE2_V>(0);
        PrepareLoadedGate(0, true);
        PrepareNextVb(0);
        PrepareNormalizedQK(0);
        ConvertInitialState();
        PrepareCurrentState(0, false);
        // chunkDecay_ is produced from the gate energy. H was already published,
        // so this decay cannot change the S used by QS/W@S.
        StateDecayVF(ubStateF_, chunkDecay_[0]);
        PrepareRightDecay(0);
    }

    __aicore__ inline void RunAivPrepare(int64_t chunk)
    {
        if (chunk + 1 < loc_.chunks) {
            // Publish the already-loaded gate before issuing newer MTE2 work.
            SetFlag<HardEvent::MTE2_V>(0);
            if (chunk + 2 < loc_.chunks) {
                IssueGateInputs(chunk + 2);
            }
            PrepareLoadedGate(chunk + 1);
            PrepareNormalizedQKLoad(chunk + 1);
            PrepareNextVb(chunk + 1);
        }
    }

    __aicore__ inline void PrepareCurrentStateHalf(int64_t chunk, uint32_t half, bool applyDecay)
    {
        const ChunkWindow window = WindowOf(loc_, chunk);
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t gateSlot = static_cast<uint32_t>(chunk % 3);
        LocalTensor<bfloat16_t> hPanel = l1H_[slot][half * (kStateElems / 2)];
        LocalTensor<bfloat16_t> statePanel = ubStateB_[half * kHalfDim * kHalfDim];
        CopyNzToL1(hPanel, statePanel, kHalfDim, kHalfDim, kDim,
                   sub_ * static_cast<uint32_t>(kHalfDim));
        if (half != 0) {
            // MTE3 is ordered, so the second-half release covers both H panels.
            SetUbRelease(ubStateBArmed_, 5);
        }
        if (half == 0) {
            // PrepareLoadedGate publishes gate/energy on V_S event 1.
            WaitFlag<HardEvent::V_S>(1);
            chunkDecay_[slot] = ubEnergy_[gateSlot].GetValue(window.rows - 1);
            AivPublish(kConsS);
        } else {
            AivPublish(kConsS1);
        }
        if (applyDecay) {
            StateDecay64VF(ubStateF_, chunkDecay_[slot], half * kHalfDim);
        }
    }

    __aicore__ inline void PrepareCurrentState(int64_t chunk, bool applyDecay = true)
    {
        const ChunkWindow window = WindowOf(loc_, chunk);
        const uint32_t firstRow = sub_ * kHalfRows;
        if (window.rows < kChunk) {
            // LoadT reads only valid rows. Clear the padded L1 rows before the
            // kConsS publication, which precedes this chunk's mode-0/LoadT.
            Duplicate(ubZeroBf_, static_cast<bfloat16_t>(0), kHalfRows * kChunk);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            CopyNzToL1(l1T_, ubZeroBf_, kHalfRows, kChunk, kChunk, firstRow);
        }
        PrepareCurrentStateHalf(chunk, 0, applyDecay);
        PrepareCurrentStateHalf(chunk, 1, applyDecay);
    }

    __aicore__ inline void StoreOutputHalf(int64_t token, uint32_t rows, uint32_t half)
    {
        if (rows == 0) {
            return;
        }
        DataCopyExtParams params{};
        params.blockCount = static_cast<uint16_t>(rows);
        params.blockLen = kHalfDim * sizeof(bfloat16_t);
        params.srcStride = 0;
        params.dstStride = (td_.valueHeads - 1) * kDim * sizeof(bfloat16_t);
        params.rsv = 0;
        DataCopyPad(gmO_[OffsetOutput(loc_.batch, token, loc_.hv,
                                      td_.valueHeads, td_.tokens) + half * kHalfDim],
                    ubO_[half * kHalfRows * kHalfDim], params);
    }

    __aicore__ inline void RunAivAfterWu(int64_t chunk)
    {
        const ChunkWindow window = WindowOf(loc_, chunk);
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t firstRow = sub_ * kHalfRows;
        const int64_t remaining = window.rows - static_cast<int64_t>(firstRow);
        const uint32_t rows = remaining <= 0 ? 0U
            : static_cast<uint32_t>(remaining < kHalfRows ? remaining : kHalfRows);
        AivWait(kConsP);
        // PrepareRightDecay published ubDecay immediately after writing it.
        WaitFlag<HardEvent::V_S>(2);
        WaitUbRelease(ubRightArmed_, 4);
        Right64NzVF(ubVNew_, ubRight_, ubDecay_[slot], firstRow);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyNzToL1(l1Right_, ubRight_, kHalfRows, kHalfDim, kChunk, firstRow);
        AivPublish(kConsRight);
        const int64_t token = window.tokenStart + firstRow;
        AivWaitMte3(kConsOL);
        StoreOutputHalf(token, rows, 0);

        AivWait(kConsP1);
        LocalTensor<bfloat16_t> vNewC2 = ubVNew_[kHalfRows * kHalfDim];
        LocalTensor<bfloat16_t> rightC2 = ubRight_[kHalfRows * kHalfDim];
        Right64NzVF(vNewC2, rightC2, ubDecay_[slot], firstRow);
        SetFlag<HardEvent::V_MTE3>(1);
        WaitFlag<HardEvent::V_MTE3>(1);
        LocalTensor<bfloat16_t> l1RightC2 = l1Right_[kChunk * kHalfDim];
        CopyNzToL1(l1RightC2, rightC2, kHalfRows, kHalfDim, kChunk, firstRow);
        SetUbRelease(ubRightArmed_, 4);
        AivPublish(kConsRight1);
        AivWaitMte3(kConsOL1);
        StoreOutputHalf(token, rows, 1);
        if (chunk + 1 < loc_.chunks) {
            // Right is published before next Q/K Vector work; the GM loads
            // were already issued after the next gate was prepared.
            ConsumeNormalizedQK(chunk + 1);
        }

        AivWait(kConsD);
        WaitUbRelease(ubStateBArmed_, 5);
        StateAddD64VF(ubStateF_, ubStateB_, ubD_, 0);
        if (chunk + 1 < loc_.chunks) {
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            PrepareCurrentStateHalf(chunk + 1, 0, true);
        }
        AivWait(kConsD1);
        LocalTensor<float> dC2 = ubD_[kHalfDim * kHalfDim];
        StateAddD64VF(ubStateF_, ubStateB_, dC2, kHalfDim);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        if (chunk + 1 < loc_.chunks) {
            PrepareCurrentStateHalf(chunk + 1, 1, true);
            PrepareRightDecay(chunk + 1);
        }
    }

    __aicore__ inline void WaitNormalizedQK(int64_t chunk)
    {
        const int64_t pair = chunk >> 1;
        // Both Consumer AIVs join once per even chunk. The odd chunk relies on
        // this same-AIV generation wait and needs no additional synchronization.
        PublishMode0Flag<PIPE_MTE3, PIPE_MTE2>(IterationNormalizedQKFlag(pair));
    }

    __aicore__ inline int64_t NormalizedQKChunkOffset(int64_t chunk) const
    {
        const int64_t tile = TileIndex(loc_, chunk, td_.chunksPerBatch, td_.varlen != 0);
        return (tile * td_.keyHeads + loc_.kh) * kChunkElems;
    }

    __aicore__ inline void CopyNormalizedGmToUb(LocalTensor<bfloat16_t> dst,
                                                GlobalTensor<bfloat16_t> src,
                                                int64_t offset, uint32_t firstRow)
    {
        // Eight NZ panels: each source panel has 64 rows, while each Consumer
        // AIV receives a contiguous 32-row band in UB.
        DataCopyParams params{8, kHalfRows * 16 * sizeof(bfloat16_t) / 32,
                              (kChunk - kHalfRows) * 16 * sizeof(bfloat16_t) / 32, 0};
        DataCopy(dst, src[offset + firstRow * 16], params);
    }

    __aicore__ inline void IssueNormalizedQKLoad(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t firstRow = sub_ * kHalfRows;
        LocalTensor<bfloat16_t> kBuffer = slot == 0 ? ubKHat_ : ubK_;
        LocalTensor<bfloat16_t> qBuffer = slot == 0 ? ubQHat_ : ubQ_;
        // Wait for this slot's primed flag or previous K L1 upload before the
        // new GM load overwrites its UB. The other slot remains independent.
        WaitFlag<HardEvent::MTE3_MTE2>(1 + slot);
        const int64_t offset = NormalizedQKChunkOffset(chunk);
        CopyNormalizedGmToUb(kBuffer, gmNormK_, offset, firstRow);
        SetFlag<HardEvent::MTE2_MTE3>(slot);
        SetFlag<HardEvent::MTE2_V>(slot);
        CopyNormalizedGmToUb(qBuffer, gmNormQ_, offset, firstRow);
        // K and Q have independent Vector readiness events. Q uses 4/5;
        // initial-state loading uses 2 and V uses 3.
        SetFlag<HardEvent::MTE2_V>(4 + slot);
    }

    __aicore__ inline void ConsumeNormalizedQK(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t gateSlot = static_cast<uint32_t>(chunk % 3);
        const uint32_t firstRow = sub_ * kHalfRows;
        LocalTensor<bfloat16_t> kBuffer = slot == 0 ? ubKHat_ : ubK_;
        LocalTensor<bfloat16_t> qBuffer = slot == 0 ? ubQHat_ : ubQ_;
        WaitFlag<HardEvent::MTE2_MTE3>(slot);
        CopyNzToL1(l1K_[slot], kBuffer, kHalfRows, kDim, kChunk, firstRow);
        SetFlag<HardEvent::MTE3_MTE2>(1 + slot);

        // The odd chunk consumes the load issued by its even predecessor. K's
        // L1 upload and Kbg Vector work do not wait for Q.
        WaitFlag<HardEvent::MTE2_V>(slot);
        WaitUbRelease(ubWArmed_, 2);
        KHatKbgNzVF(kBuffer, ubW_, ubBeta_[gateSlot], ubEnergy_[gateSlot], firstRow);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyNzToL1(l1W_[slot], ubW_, kHalfRows, kDim, kChunk, firstRow);
        SetUbRelease(ubWArmed_, 2);
        AivPublish(kConsW);

        WaitFlag<HardEvent::MTE2_V>(4 + slot);
        WaitUbRelease(ubOSBArmed_, 1);
        QHatExpNzVF(qBuffer, ubOSB_, ubEnergy_[gateSlot], firstRow);
        SetFlag<HardEvent::V_MTE3>(1);
        WaitFlag<HardEvent::V_MTE3>(1);
        CopyNzToL1(l1QExp_[slot], ubOSB_, kHalfRows, kDim, kChunk, firstRow);
        SetUbRelease(ubOSBArmed_, 1);
    }

    __aicore__ inline void PrepareNormalizedQKLoad(int64_t chunk)
    {
        if ((chunk & 1) == 0) {
            // At each even chunk, issue both chunks' GM transfers back-to-back.
            WaitNormalizedQK(chunk);
            IssueNormalizedQKLoad(chunk);
            if (chunk + 1 < loc_.chunks) {
                IssueNormalizedQKLoad(chunk + 1);
            }
        }
    }

    __aicore__ inline void PrepareNormalizedQK(int64_t chunk)
    {
        PrepareNormalizedQKLoad(chunk);
        ConsumeNormalizedQK(chunk);
    }

    __aicore__ inline void IssueGateInputs(int64_t chunk)
    {
        const ChunkWindow window = WindowOf(loc_, chunk);
        const uint32_t gateSlot = static_cast<uint32_t>(chunk % 3);
        if (window.rows < kChunk) {
            Duplicate(ubG_[gateSlot], 0.0f, kChunk);
            Duplicate(ubBeta_[gateSlot], 0.0f, kChunk);
        }
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        CopyGmToUb(ubG_[gateSlot], gmG_, OffsetBht(loc_.batch, loc_.hv, window.tokenStart,
                                                   td_.valueHeads, td_.tokens),
                   static_cast<uint32_t>(window.rows));
        CopyGmToUb(ubBeta_[gateSlot], gmBeta_, OffsetBht(loc_.batch, loc_.hv, window.tokenStart,
                                                         td_.valueHeads, td_.tokens),
                   static_cast<uint32_t>(window.rows));
    }

    __aicore__ inline void PrepareLoadedGate(int64_t chunk, bool includeInitial = false)
    {
        const ChunkWindow window = WindowOf(loc_, chunk);
        const uint32_t gateSlot = static_cast<uint32_t>(chunk % 3);
        const uint32_t firstRow = sub_ * kHalfRows;
        const int64_t remaining = window.rows - static_cast<int64_t>(firstRow);
        const uint32_t rows = remaining <= 0 ? 0U
            : static_cast<uint32_t>(remaining < kHalfRows ? remaining : kHalfRows);
        if (rows < kHalfRows) {
            Duplicate(ubV_, static_cast<bfloat16_t>(0), kHalfRows * kDim);
        }
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        const int64_t token = window.tokenStart + firstRow;
        // V still overlaps gate processing. Gate i+2 was issued before this
        // function started computing the already-loaded gate i+1.
        CopyGmToUb(ubV_, gmV_, OffsetBhtd(loc_.batch, loc_.hv, token, td_.valueHeads, td_.tokens),
                   rows * static_cast<uint32_t>(kDim));
        SetFlag<HardEvent::MTE2_V>(3);
        WaitFlag<HardEvent::MTE2_V>(0);
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
        if (td_.useExp2 != 0) {
            CumsumEnergyVF<true>(ubG_[gateSlot], ubEnergy_[gateSlot], ubEnergyRecip_[gateSlot], ubShift_);
        } else {
            CumsumEnergyVF<false>(ubG_[gateSlot], ubEnergy_[gateSlot], ubEnergyRecip_[gateSlot], ubShift_);
        }

        SetFlag<HardEvent::V_S>(1);
        if (includeInitial) {
            LoadInitialState();
        }
    }

    __aicore__ inline void PrepareRightDecay(int64_t chunk)
    {
        const ChunkWindow window = WindowOf(loc_, chunk);
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t gateSlot = static_cast<uint32_t>(chunk % 3);
        const uint32_t firstRow = sub_ * kHalfRows;
        // Gate/energy V_S event 1 was consumed by PrepareCurrentStateHalf.
        // PrepareRightDecay runs before this gate slot is reused.
        gateRatioSafe_ = GateRatioSafe(ubG_[gateSlot], ubScratch_, static_cast<uint32_t>(window.rows));
        const float gLast = ubG_[gateSlot].GetValue(window.rows - 1);
        const float lastEnergy = chunkDecay_[slot];
        if (td_.useExp2 != 0) {
            DecayVF<true>(ubDecay_[slot], ubG_[gateSlot], ubEnergyRecip_[gateSlot], gLast, lastEnergy,
                          gateRatioSafe_, firstRow);
        } else {
            DecayVF<false>(ubDecay_[slot], ubG_[gateSlot], ubEnergyRecip_[gateSlot], gLast, lastEnergy,
                           gateRatioSafe_, firstRow);
        }
        // Publish decay now rather than draining all Vector work at its use.
        SetFlag<HardEvent::V_S>(2);
    }

    __aicore__ inline void PrepareNextVb(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t gateSlot = static_cast<uint32_t>(chunk % 3);
        const uint32_t firstRow = sub_ * kHalfRows;
        WaitFlag<HardEvent::MTE2_V>(3);
        WaitUbRelease(ubUArmed_, 3);
        ScaleVbNzVF(ubV_, ubU_, ubBeta_[gateSlot], firstRow);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyNzToL1(l1U_[slot], ubU_, kHalfRows, kDim, kChunk, firstRow);
        SetUbRelease(ubUArmed_, 3);
        AivPublish(kConsU);
    }

    __aicore__ inline void LoadT(int64_t chunk)
    {
        const ChunkWindow window = WindowOf(loc_, chunk);
        Nd2NzParams params{};
        params.ndNum = 1;
        params.nValue = static_cast<uint32_t>(window.rows);
        params.dValue = 64;
        params.srcDValue = 64;
        params.srcNdMatrixStride = 0;
        params.dstNzC0Stride = 64;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 0;
        const int64_t offset = OffsetBhtd(loc_.batch, loc_.hv, window.tokenStart,
                                          td_.valueHeads, td_.tokens, kChunk);
        DataCopy(l1T_, gmT_[offset], params);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
    }

    __aicore__ inline void LoadAPrime(int64_t chunk)
    {
        // Mode 0 requires every AIC to set as well as wait. Delay this
        // publish until O so T, W and v_new do not wait for APrime.
        PublishAPrimeAic<PIPE_MTE2>(chunk);
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
        const bool transposeB = td_.stateVFirst == 0;
        AicWaitBoth(kConsS);
        ComputeMm(l1QExp_[slot], l1H_[slot], l0A0_, l0B0_, l0C3_, 64, 64, 128,
                  false, transposeB);
        AicWaitBoth(kConsS1);
        LocalTensor<bfloat16_t> hC2 = l1H_[slot][kStateElems / 2];
        ComputeMm(l1QExp_[slot], hC2, l0A0_, l0B0_, l0C3_[kChunk * kHalfDim],
                  64, 64, 128, false, transposeB, true, false);
        // C1/C2 remain in L0C3 for the corresponding O panels. One release
        // covers both Matrix results; H is reloaded from L1 for W@S later.
        ReleaseL0C();
    }

    __aicore__ inline void RunCubeAfterWu(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const bool transposeState = td_.stateVFirst == 0;
        LoadT(chunk);
        AicWaitBoth(kConsW);
        // Kbg is negated, so W=T@Kbg=-trueW.
        ComputeMmAndPrefetchB(l1T_, l1W_[slot], l1U_[slot], l0A0_, l0B1_, l0B0_, l0C2_,
                              64, 128, 64, 64, false, true, true);
        FixB16L1Matrix(l1W_[slot], l0C2_, 64, 128, 64);

        // U initializes each accumulator while W's L1 store is in flight.
        // W@S then accumulates into the corresponding U panel.
        AicWaitBoth(kConsU);
        IssueMm(l0A0_, l0B0_, l0C0_, 64, 64, 64, true, false);
        SetFlag<HardEvent::FIX_MTE1>(0);
        WaitFlag<HardEvent::FIX_MTE1>(0);
        ComputeMm(l1W_[slot], l1H_[slot], l0A0_, l0B0_, l0C0_, 64, 64, 128,
                  false, transposeState, false);
        FixB16Halves(ubVNew_, l0C0_, 64, 64);
        AicPublishToBoth(kConsP);
        FixB16L1Matrix(l1VNew_, l0C0_, 64, 64, 64);
        SetFlag<HardEvent::FIX_MTE1>(0);

        LocalTensor<bfloat16_t> hC2 = l1H_[slot][kStateElems / 2];
        LocalTensor<bfloat16_t> uC2 = l1U_[slot][kChunk * kHalfDim];
        ComputeMm(l1T_, uC2, l0A0_, l0B1_, l0C0_[kChunk * kHalfDim],
                  64, 64, 64, false, true, true, false);
        ComputeMm(l1W_[slot], hC2, l0A0_, l0B0_, l0C0_[kChunk * kHalfDim],
                  64, 64, 128, false, transposeState, false);
        LocalTensor<bfloat16_t> l1VNewC2 = l1VNew_[kChunk * kHalfDim];
        FixB16Halves(ubVNew_[kHalfRows * kHalfDim], l0C0_[kChunk * kHalfDim], 64, 64);
        AicPublishToBoth(kConsP1);
        FixB16L1Matrix(l1VNewC2, l0C0_[kChunk * kHalfDim], 64, 64, 64);
        SetFlag<HardEvent::FIX_MTE1>(1);
        ReleaseL0C();

        // O accumulates into the matching QS panel retained from RunCubeQs.
        LoadAPrime(chunk);
        WaitFlag<HardEvent::FIX_MTE1>(0);
        AicWaitBoth(kConsRight);
        LocalTensor<bfloat16_t> dA = td_.stateVFirst != 0 ? l1Right_ : l1K_[slot];
        LocalTensor<bfloat16_t> dB = td_.stateVFirst != 0 ? l1K_[slot] : l1Right_;
        ComputeMmAndPrefetchB(l1APrime_[slot], l1VNew_, dB, l0A1_, l0B1_, l0B0_, l0C3_,
                              64, 64, 64, 64, false, true, true, false);
        FixB16ScaledHalves(ubO_, l0C3_, 64, 64, td_.scale);
        AicPublishToBoth(kConsOL);

        // D-C1 reuses its prefetched B and prefetches VNew-C2 for O-C2.
        WaitFlag<HardEvent::FIX_MTE1>(1);
        WaitFlag<HardEvent::M_MTE1>(0);
        LoadA(l0A0_, dA, 128, 64, true);
        IssueMmAndPrefetchB(l0A0_, l0B0_, l0B1_, l1VNewC2, l0C2_,
                            128, 64, 64, 64, true, true, false);
        FixF32Halves(ubD_, l0C2_, 128, 64);
        AicPublishToBoth(kConsD);

        // APrime and VNew-C2 are already in L0A/L0B for O-C2. Prefetch D-C2's
        // second B panel while O-C2 is in flight.
        AicWaitBoth(kConsRight1);
        WaitFlag<HardEvent::M_MTE1>(0);
        IssueMmAndPrefetchB(l0A1_, l0B1_, l0B0_, dB[kHalfDim * kHalfDim],
                            l0C3_[kChunk * kHalfDim], 64, 64, 64, 64, true, false);
        FixB16ScaledHalves(ubO_[kHalfRows * kHalfDim], l0C3_[kChunk * kHalfDim],
                           64, 64, td_.scale);
        AicPublishToBoth(kConsOL1);

        IssueMm(l0A0_, l0B0_, l0C2_[kStateElems / 2], 128, 64, 64, true, false);
        FixF32Halves(ubD_[kHalfDim * kHalfDim], l0C2_[kStateElems / 2], 128, 64);
        // Consume the VNew accumulator release only after all O/D Matrix work.
        WaitFlag<HardEvent::FIX_M>(0);
        // One release after D-C2 covers both O panels and both D panels.
        ReleaseL0C();
        AicPublishToBoth(kConsD1);

        if (chunk + 1 < loc_.chunks) {
            RunCubeQs(chunk + 1);
        }
    }

    FusedTilingData td_{};
    ChainLoc loc_{};
    uint32_t sub_{0};
    bool hasInitial_{false};
    bool gateRatioSafe_{false};
    bool ubOSBArmed_{false};
    bool ubWArmed_{false};
    bool ubUArmed_{false};
    bool ubRightArmed_{false};
    bool ubStateBArmed_{false};
    float chunkDecay_[2]{};

    GlobalTensor<bfloat16_t> gmV_, gmT_, gmO_, gmAPrime_, gmNormQ_, gmNormK_;
    GlobalTensor<bfloat16_t> gmInitB_, gmFinalB_;
    GlobalTensor<float> gmG_, gmBeta_, gmInitF_, gmFinalF_, gmALog_, gmDtBias_;

    LocalTensor<float> ubStateF_, ubD_, ubP_, ubScratch_;
    LocalTensor<float> ubDecay_[2];
    LocalTensor<bfloat16_t> ubStateB_, ubU_, ubVNew_, ubRight_, ubW_, ubQ_, ubK_, ubV_;
    LocalTensor<bfloat16_t> ubQHat_, ubKHat_, ubZeroBf_, ubO_, ubInitB_, ubOSB_;
    LocalTensor<float> ubG_[3], ubEnergy_[3], ubBeta_[3], ubEnergyRecip_[3];
    LocalTensor<uint32_t> ubShift_;
    LocalTensor<bfloat16_t> l1W_[2], l1U_[2], l1QExp_[2], l1K_[2], l1H_[2];
    LocalTensor<bfloat16_t> l1VNew_, l1Right_, l1APrime_[2], l1T_;
    LocalTensor<bfloat16_t> l0A0_, l0A1_, l0B0_, l0B1_;
    LocalTensor<float> l0C0_, l0C2_, l0C3_;
};

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_CONSUMER_H
