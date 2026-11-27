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
constexpr uint16_t kConsD1 = kConsD; // Ordered C1/C2 generations share one mode-4 channel.
constexpr uint16_t kConsRight1 = 8;
constexpr uint16_t kConsP1 = 9;
constexpr uint16_t kConsS = 10;
constexpr uint16_t kConsU = 11;
constexpr uint16_t kConsS1 = 0;
constexpr uint16_t kConsQExp = 7; // Continuous GM->L1 DataCopy reserves mode-4 flags 1/17.
// V ready events per parity slot; K uses 0/1, initial state 2, Q 4/5.
constexpr int32_t kConsVEventIds[2] = {3, 6};
// Gate/beta readiness gets a dedicated event: a blanket publish would cover
// every prior MTE2 transfer (V, normalized K/Q), not just the gate loads.
constexpr int32_t kConsGateEvent = 7;
// V->MTE2 release guarding each ubV_ slot against its next load. Each parity
// slot owns an ID: consecutive SetFlag on one pipe+ID without an intervening
// wait is undefined behavior, and the chain tail's last two releases would
// land back-to-back on a single shared ID.
constexpr int32_t kConsVReleaseEventIds[2] = {1, 2};
// V->MTE2 release guarding each normalized-Q UB slot (ubQHat_/ubQ_) against
// the chunk+2 load. This used to ride IssueGateInputs' V_MTE2(0) self-drain,
// which preceded the QK loads; with the loads moved ahead of it the slot
// needs its own release, one ID per parity for the same reason as V.
constexpr int32_t kConsQReleaseEventIds[2] = {3, 4};
// MTE1->MTE2 releases guarding l1APrime_ against the next pair's GM loads.
// T uses IDs 0/1; APrime owns independent IDs because its LoadA is consumed
// much later in RunCubeAfterWu.
constexpr int32_t kConsAPrimeReleaseEventIds[2] = {2, 3};

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
            // Prime this consumer's normalized K/Q credits for pairs 0/1 on
            // both channels. Without them the first two rounds never gather
            // all AIV sets. From then on each WaitNormalized* returns the
            // pair+2 credit. Round 1 exists only when there is a second
            // global pair; priming it otherwise would leave an orphan set.
            // An empty chain runs no chunk at all, so it stays out of the
            // generation exchange entirely: a prime here would strand a credit
            // on the producer, which publishes nothing for this chain either.
            if (loc_.chunks > 0) {
                const int64_t globalPairs = (td_.maxChunks + 1) / 2;
                SetMode0Flag<PIPE_MTE2>(IterationNormalizedKFlag(0));
                SetMode0Flag<PIPE_MTE2>(IterationNormalizedQFlag(0));
                if (globalPairs >= 2) {
                    SetMode0Flag<PIPE_MTE2>(IterationNormalizedKFlag(1));
                    SetMode0Flag<PIPE_MTE2>(IterationNormalizedQFlag(1));
                }
            } else {
                InitState();
            }
        }
        GDN_SPARSE_TRACE(td_, 30, chain, static_cast<int64_t>(loc_.chunks));
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
    }

    __aicore__ inline void RunPrepare(int64_t chunk)
    {
        if ASCEND_IS_AIV {
            RunAivPrepare(chunk);
        }
        if ASCEND_IS_AIC {
            RunCubePrepare(chunk);
            RunCubeQs(chunk);
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
                WaitNormalizedK(chunk);
                WaitNormalizedQ(chunk);
            }
        }
    }

    __aicore__ inline void EndChain()
    {
        if ASCEND_IS_AIV {
            GDN_SPARSE_TRACE(td_, 33, 0, 0);
            for (uint32_t slot = 0; slot < 2; ++slot) {
                WaitFlag<HardEvent::MTE3_MTE2>(1 + slot);
            }
            GDN_SPARSE_TRACE(td_, 34, 0, loc_.chunks);
            // Drain the V slot releases of the last two ComputeNextVb calls;
            // their IssueV waits never came. Each release sits on its own
            // parity slot's ID, so drain by parity. The same applies to the
            // normalized-Q slot releases of the last two ConsumeNormalizedQ
            // calls.
            const int64_t releaseDrainBegin = loc_.chunks >= 2 ? loc_.chunks - 2 : 0;
            for (int64_t chunk = releaseDrainBegin; chunk < loc_.chunks; ++chunk) {
                const uint32_t parity = static_cast<uint32_t>(chunk & 1);
                WaitFlag<HardEvent::V_MTE2>(kConsVReleaseEventIds[parity]);
                WaitFlag<HardEvent::V_MTE2>(kConsQReleaseEventIds[parity]);
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
            GDN_SPARSE_TRACE(td_, 35, 0, 0);
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
        for (uint32_t slot = 0; slot < 2; ++slot) {
            l1T_[slot] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(
                kCL1T + slot * kCL1TSlotBytes);
        }
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
            for (uint32_t slot = 0; slot < 2; ++slot) {
                ubV_[slot] = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(
                    kCUbV + slot * kCUbVSlotBytes);
            }
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
            // bank1 holds QExp first, APrime behind it; they never overlap.
            l0APrime_ = l0A1_[kL0APrimeOffset];
            l0B0_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0B0);
            l0B1_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0B1);
            l0B2_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0B2);
            l0B3_ = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kL0B3);
            l0C0_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C0);
            l0C1_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C1);
            l0C2_ = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0C2);
            l0CQS_[0] = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0CQS0);
            l0CQS_[1] = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kL0CQS1);
        }
    }

    // QS/O accumulator and its release event are selected by chunk parity: the
    // Fixpipe of chunk i's O therefore never gates chunk i+1's QS.
    __aicore__ inline LocalTensor<float> QsAccumulator(int64_t chunk) const
    {
        return l0CQS_[static_cast<uint32_t>(chunk & 1)];
    }

    __aicore__ inline int32_t QsReleaseEvent(int64_t chunk) const
    {
        return kL0CReleaseEventIds[2 + 3 * static_cast<int32_t>(chunk & 1)];
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
        GDN_SPARSE_TRACE(td_, 31, 0, 0);
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
        GDN_SPARSE_TRACE(td_, 32, 0, 0);
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

    // Debug-only: force the Vector queue to drain so a stall inside it becomes
    // visible as a missing marker. Compiled out (a single scalar compare) when
    // the trace channel is off.
    __aicore__ inline void DrainVectorProbe(uint32_t code)
    {
        if (td_.reserved != kSparseDebugTrace) {
            return;
        }
        SetFlag<HardEvent::V_S>(3);
        WaitFlag<HardEvent::V_S>(3);
        GDN_SPARSE_TRACE(td_, code, 0, 0);
    }

    __aicore__ inline void RunAivPrologue()
    {
        // H preparation starts after the previous chain's EndChain drain.
        // H is independent of Q/K/V preparation. Publish it first so QS can
        // start immediately after QExp is ready, then overlap QKT with Vb.
        IssueGateInputs(0);
        // V runs one iteration ahead of its Vb computation, like the gates.
        IssueV(0);
        // The gate readiness event is a single ID: consume gate 0's
        // publication before gate 1's IssueGateInputs sets it again, so the
        // ID never sees back-to-back sets.
        PrepareLoadedGate(0, true);
        DrainVectorProbe(90);
        if (loc_.chunks > 1) {
            IssueGateInputs(1);
            IssueV(1);
        }
        PrepareNextVb(0);
        DrainVectorProbe(91);
        PrepareNormalizedQK(0);
        DrainVectorProbe(92);
        ConvertInitialState();
        DrainVectorProbe(93);
        PrepareCurrentState(0, false);
        DrainVectorProbe(94);
        // chunkDecay_ is produced from the gate energy. H was already published,
        // so this decay cannot change the S used by QS/W@S.
        StateDecayVF(ubStateF_, chunkDecay_[0]);
        DrainVectorProbe(95);
        PrepareRightDecay(0);
        DrainVectorProbe(96);
    }

    __aicore__ inline void RunAivPrepare(int64_t chunk)
    {
        if (chunk + 1 < loc_.chunks) {
            // Consume this gate's publication before IssueGateInputs reuses
            // the single event ID, keeping the ID in strict set/wait
            // alternation. Gate chunk+2 still lands a full iteration early.
            PrepareLoadedGate(chunk + 1);
            // Normalized Q/K feeds this iteration's Kbg computation, so its
            // GM loads go ahead of the gate/V prefetch: MTE2 is in-order and
            // ComputeNormalizedK's ready flag covers only up to the K copy.
            // The mode-0 credit sets also fire earlier, unblocked by the
            // prefetch transfers.
            PrepareNormalizedQKLoad(chunk + 1);
            if (chunk + 2 < loc_.chunks) {
                IssueGateInputs(chunk + 2);
                IssueV(chunk + 2);
            }
            ComputeNextVb(chunk + 1);
            ComputeNormalizedK(chunk + 1);
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
            CopyNzToL1(l1T_[0], ubZeroBf_, kHalfRows, kChunk, kChunk, firstRow);
        }
        PrepareCurrentStateHalf(chunk, 0, applyDecay);
        PrepareCurrentStateHalf(chunk, 1, applyDecay);
    }

    __aicore__ inline void StoreOutput(int64_t token, uint32_t rows)
    {
        if (rows == 0) {
            return;
        }
        DataCopyExtParams params{};
        params.blockCount = static_cast<uint16_t>(rows);
        params.blockLen = kDim * sizeof(bfloat16_t);
        params.srcStride = 0;
        params.dstStride = (td_.valueHeads - 1) * kDim * sizeof(bfloat16_t);
        params.rsv = 0;
        DataCopyPad(gmO_[OffsetOutput(loc_.batch, token, loc_.hv, td_.valueHeads, td_.tokens)],
                    ubO_, params);
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
        if (chunk + 1 < loc_.chunks) {
            // Current Right is published before next-chunk L1 uploads. Vb/Kbg
            // were computed in Prepare; QExp overlaps the current state update.
            UploadNextVb(chunk + 1);
            UploadNormalizedK(chunk + 1);
            ConsumeNormalizedQ(chunk + 1);
        }

        AivWait(kConsD);
        WaitUbRelease(ubStateBArmed_, 5);
        StateAddD64VF(ubStateF_, ubStateB_, ubD_, 0);
        if (chunk + 1 < loc_.chunks) {
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            PrepareCurrentStateHalf(chunk + 1, 0, true);
            PrepareRightDecay(chunk + 1);
        }
        AivWait(kConsD1);
        LocalTensor<float> dC2 = ubD_[kHalfDim * kHalfDim];
        StateAddD64VF(ubStateF_, ubStateB_, dC2, kHalfDim);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        if (chunk + 1 < loc_.chunks) {
            PrepareCurrentStateHalf(chunk + 1, 1, true);
        }
        AivWaitMte3(kConsOL);
        StoreOutput(window.tokenStart + firstRow, rows);
    }

    __aicore__ inline void WaitNormalizedK(int64_t chunk)
    {
        const int64_t pair = chunk >> 1;
        // Credit protocol on a single pipe: receive generation pair, then
        // return the pair+2 credit on the same flag ID (IDs alternate with
        // period 2, so pair+2 aliases pair's ID). The MTE2 set stays behind
        // the wait in the load stream, so the credit is strictly posterior
        // to the receipt and the producer's drain tracks load progress.
        // Pairs 0/1 are primed in BeginChain; rounds at or beyond the last
        // global generation get no credit — nobody publishes them.
        WaitMode0Flag<PIPE_MTE2>(IterationNormalizedKFlag(pair));
        if (pair + 2 < (td_.maxChunks + 1) / 2) {
            SetMode0Flag<PIPE_MTE2>(IterationNormalizedKFlag(pair + 2));
        }
    }

    __aicore__ inline void WaitNormalizedQ(int64_t chunk)
    {
        const int64_t pair = chunk >> 1;
        WaitMode0Flag<PIPE_MTE2>(IterationNormalizedQFlag(pair));
        if (pair + 2 < (td_.maxChunks + 1) / 2) {
            SetMode0Flag<PIPE_MTE2>(IterationNormalizedQFlag(pair + 2));
        }
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
        if ((chunk & 1) == 0) {
            // Join K and Q independently so the K GM read can overlap Q compute.
            WaitNormalizedK(chunk);
        }
        CopyNormalizedGmToUb(kBuffer, gmNormK_, offset, firstRow);
        SetFlag<HardEvent::MTE2_MTE3>(slot);
        SetFlag<HardEvent::MTE2_V>(slot);
        if ((chunk & 1) == 0) {
            WaitNormalizedQ(chunk);
        }
        if (chunk >= 2) {
            // The slot's previous reader is QHatExpNzVF(chunk-2) on V.
            WaitFlag<HardEvent::V_MTE2>(kConsQReleaseEventIds[slot]);
        }
        CopyNormalizedGmToUb(qBuffer, gmNormQ_, offset, firstRow);
        // K and Q have independent Vector readiness events. Q uses 4/5;
        // initial-state loading uses 2 and V uses 3.
        SetFlag<HardEvent::MTE2_V>(4 + slot);
    }

    __aicore__ inline void ComputeNormalizedK(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t gateSlot = static_cast<uint32_t>(chunk % 3);
        const uint32_t firstRow = sub_ * kHalfRows;
        LocalTensor<bfloat16_t> kBuffer = slot == 0 ? ubKHat_ : ubK_;
        WaitFlag<HardEvent::MTE2_V>(slot);
        WaitUbRelease(ubWArmed_, 2);
        KHatKbgNzVF(kBuffer, ubW_, ubBeta_[gateSlot], ubEnergy_[gateSlot], firstRow);
    }

    __aicore__ inline void UploadNormalizedK(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t firstRow = sub_ * kHalfRows;
        LocalTensor<bfloat16_t> kBuffer = slot == 0 ? ubKHat_ : ubK_;
        WaitFlag<HardEvent::MTE2_MTE3>(slot);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyNzToL1(l1K_[slot], kBuffer, kHalfRows, kDim, kChunk, firstRow);
        SetFlag<HardEvent::MTE3_MTE2>(1 + slot);
        CopyNzToL1(l1W_[slot], ubW_, kHalfRows, kDim, kChunk, firstRow);
        SetUbRelease(ubWArmed_, 2);
        AivPublish(kConsW);
    }

    __aicore__ inline void ConsumeNormalizedK(int64_t chunk)
    {
        ComputeNormalizedK(chunk);
        UploadNormalizedK(chunk);
    }

    __aicore__ inline void ConsumeNormalizedQ(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t gateSlot = static_cast<uint32_t>(chunk % 3);
        const uint32_t firstRow = sub_ * kHalfRows;
        LocalTensor<bfloat16_t> qBuffer = slot == 0 ? ubQHat_ : ubQ_;
        WaitFlag<HardEvent::MTE2_V>(4 + slot);
        WaitUbRelease(ubOSBArmed_, 1);
        QHatExpNzVF(qBuffer, ubOSB_, ubEnergy_[gateSlot], firstRow);
        // Release the Q UB slot to the chunk+2 load; the load no longer trails
        // IssueGateInputs' V drain, so this release is the only guard.
        SetFlag<HardEvent::V_MTE2>(kConsQReleaseEventIds[slot]);
        SetFlag<HardEvent::V_MTE3>(1);
        WaitFlag<HardEvent::V_MTE3>(1);
        CopyNzToL1(l1QExp_[slot], ubOSB_, kHalfRows, kDim, kChunk, firstRow);
        SetUbRelease(ubOSBArmed_, 1);
        AivPublish(kConsQExp);
    }

    __aicore__ inline void ConsumeNormalizedQK(int64_t chunk)
    {
        ConsumeNormalizedK(chunk);
        ConsumeNormalizedQ(chunk);
    }

    __aicore__ inline void PrepareNormalizedQKLoad(int64_t chunk)
    {
        if ((chunk & 1) == 0) {
            // At each even chunk, issue both chunks' GM transfers back-to-back.
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
        // Publish exactly this chunk's gate/beta loads, nothing newer.
        SetFlag<HardEvent::MTE2_V>(kConsGateEvent);
    }

    // V loads run one iteration ahead of their ComputeNextVb consumption:
    // iteration i issues chunk i+2 into the parity slot that chunk i's Vb
    // computation has just released.
    __aicore__ inline void IssueV(int64_t chunk)
    {
        const ChunkWindow window = WindowOf(loc_, chunk);
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t firstRow = sub_ * kHalfRows;
        const int64_t remaining = window.rows - static_cast<int64_t>(firstRow);
        const uint32_t rows = remaining <= 0 ? 0U
            : static_cast<uint32_t>(remaining < kHalfRows ? remaining : kHalfRows);
        if (chunk >= 2) {
            // The slot's previous reader is ComputeNextVb(chunk-2) on V.
            WaitFlag<HardEvent::V_MTE2>(kConsVReleaseEventIds[slot]);
        }
        if (rows < kHalfRows) {
            Duplicate(ubV_[slot], static_cast<bfloat16_t>(0), kHalfRows * kDim);
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }
        const int64_t token = window.tokenStart + firstRow;
        CopyGmToUb(ubV_[slot], gmV_,
                   OffsetBhtd(loc_.batch, loc_.hv, token, td_.valueHeads, td_.tokens),
                   rows * static_cast<uint32_t>(kDim));
        SetFlag<HardEvent::MTE2_V>(kConsVEventIds[slot]);
    }

    __aicore__ inline void PrepareLoadedGate(int64_t chunk, bool includeInitial = false)
    {
        const uint32_t gateSlot = static_cast<uint32_t>(chunk % 3);
        // V is issued separately one iteration early; only this chunk's
        // gate/beta loads are consumed here.
        WaitFlag<HardEvent::MTE2_V>(kConsGateEvent);
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

    __aicore__ inline void ComputeNextVb(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t gateSlot = static_cast<uint32_t>(chunk % 3);
        const uint32_t firstRow = sub_ * kHalfRows;
        WaitFlag<HardEvent::MTE2_V>(kConsVEventIds[slot]);
        WaitUbRelease(ubUArmed_, 3);
        ScaleVbNzVF(ubV_[slot], ubU_, ubBeta_[gateSlot], firstRow);
        // Release the slot to the chunk+2 load issued next iteration.
        SetFlag<HardEvent::V_MTE2>(kConsVReleaseEventIds[slot]);
    }

    __aicore__ inline void UploadNextVb(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const uint32_t firstRow = sub_ * kHalfRows;
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyNzToL1(l1U_[slot], ubU_, kHalfRows, kDim, kChunk, firstRow);
        SetUbRelease(ubUArmed_, 3);
        AivPublish(kConsU);
    }

    __aicore__ inline void PrepareNextVb(int64_t chunk)
    {
        ComputeNextVb(chunk);
        UploadNextVb(chunk);
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
        DataCopy(l1T_[chunk & 1], gmT_[offset], params);
    }

    __aicore__ inline void LoadAPrime(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const int64_t tile = TileIndex(loc_, chunk, td_.chunksPerBatch, td_.varlen != 0);
        const int64_t offset = OffsetMatrixTile(tile, loc_.hv, td_.valueHeads);
        // The Producer writes a complete 64x64 NZ tile, so no ND conversion or
        // per-half copies are needed here.
        DataCopy(l1APrime_[slot], gmAPrime_[offset], kMatrixElems);
    }

    __aicore__ inline void RunCubeQs(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        const bool transposeB = td_.stateVFirst == 0;
        LocalTensor<float> qs = QsAccumulator(chunk);
        AicWaitBoth(kConsQExp);
        AicWaitBoth(kConsS);
        // QExp goes to bank1 and H to its own B slots, so neither load waits for
        // the T@Vb mmad still in flight. Both were retired by the chunk-head
        // M_MTE1 drain, so no operand release wait is needed here either.
        LoadA(l0A1_, l1QExp_[slot], 64, 128, false);
        LoadB(l0B2_, l1H_[slot], 64, 128, transposeB);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        // Consume the same-parity O Fixpipe release (two chunks back) before
        // reinitializing this parity's accumulator.
        WaitFlag<HardEvent::FIX_M>(QsReleaseEvent(chunk));
        IssueMm(l0A1_, l0B2_, qs, 64, 64, 128, true);

        AicWaitBoth(kConsS1);
        LocalTensor<bfloat16_t> hC2 = l1H_[slot][kStateElems / 2];
        LoadB(l0B3_, hC2, 64, 128, transposeB);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        IssueMm(l0A1_, l0B3_, qs[kChunk * kHalfDim], 64, 64, 128, true);
        // M orders C2 after C1 and the W@S mmads behind both, so QS needs no
        // completion wait of its own; the O accumulation consumes the result.
    }

    __aicore__ inline void RunCubePrepare(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        if ((chunk & 1) == 0) {
            // Load both lanes' T tiles back-to-back on independent MTE2_MTE1
            // flags so this lane's T@Kbg never waits for the other lane's
            // load. Each slot's previous LoadA retired two chunks earlier.
            if (chunk >= 2) {
                WaitFlag<HardEvent::MTE1_MTE2>(0);
                WaitFlag<HardEvent::MTE1_MTE2>(1);
            }
            LoadT(chunk);
            SetFlag<HardEvent::MTE2_MTE1>(0);
            if (chunk + 1 < loc_.chunks) {
                LoadT(chunk + 1);
                SetFlag<HardEvent::MTE2_MTE1>(1);
            }
            // The APrime join and loads follow T: T never waits for APrime.
            // APrime uses dedicated flag IDs 2/3; reusing the T IDs would put
            // two sets on one ID back-to-back, which is undefined behavior.
            // Their wait sits at the O consumption point in RunCubeAfterWu,
            // so the transfers hide under U-WS and the v_new-decay work.
            // Before each parity slot's GM load, retire the previous pair's
            // LoadA for that same slot. Waiting per slot avoids making the
            // next even load trail the previous odd lane unnecessarily.
            JoinAPrimeAicCredit<PIPE_MTE2>(chunk >> 1, (td_.maxChunks + 1) / 2);
            if (chunk >= 2) {
                WaitFlag<HardEvent::MTE1_MTE2>(kConsAPrimeReleaseEventIds[chunk & 1]);
            }
            LoadAPrime(chunk);
            SetFlag<HardEvent::MTE2_MTE1>(2);
            if (chunk + 1 < loc_.chunks) {
                if (chunk >= 2) {
                    WaitFlag<HardEvent::MTE1_MTE2>(kConsAPrimeReleaseEventIds[(chunk + 1) & 1]);
                }
                LoadAPrime(chunk + 1);
                SetFlag<HardEvent::MTE2_MTE1>(3);
            }
        }
        AicWaitBoth(kConsW);
        // T@Kbg and T@Vb do not depend on H. Compute them before QS, using
        // l0C2/l0C0 independently of the previous D panels in l0C1.
        WaitFlag<HardEvent::M_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(slot);
        LoadA(l0A0_, l1T_[slot], 64, 64, false);
        SetFlag<HardEvent::MTE1_MTE2>(slot);
        LoadB(l0B0_, l1W_[slot], 128, 64, true);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::FIX_M>(kL0CReleaseEventIds[1]);
        IssueMm(l0A0_, l0B0_, l0C2_, 64, 128, 64, true);
        LoadB(l0B1_, l1U_[slot], 128, 64, true);
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        FixB16L1Matrix(l1W_[slot], l0C2_, 64, 128, 64);
        SetFlag<HardEvent::FIX_M>(kL0CReleaseEventIds[1]);
        // Publish W as soon as its Fixpipe is ordered; U and QS are independent.
        SetFlag<HardEvent::FIX_MTE1>(0);

        AicWaitBoth(kConsU);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::FIX_M>(kL0CReleaseEventIds[0]);
        IssueMm(l0A0_, l0B1_, l0C0_, 64, 128, 64, true);
        // T@Vb keeps bank0's T and the l0C0 accumulator live until W@S
        // accumulates onto them. Its completion is waited in RunCubeAfterWu, so
        // the QS issue runs while this mmad is still in the queue.
        SetFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::M_FIX>(0);
    }

    __aicore__ inline void RunCubeAfterWu(int64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 1);
        LocalTensor<float> qs = QsAccumulator(chunk);
        // T@Vb held bank0's T and the l0C0 accumulator; its completion is only
        // needed now, so QS was issued underneath it.
        WaitFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_MTE1>(0);
        // This wait pairs with the W Fixpipe in RunCubePrepare, not with QS.
        WaitFlag<HardEvent::FIX_MTE1>(0);
        LoadA(l0A0_, l1W_[slot], 64, 128, false);
        // H's first half is still in its own B slot from RunCubeQs; W@S reuses
        // it instead of reloading.
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        IssueMm(l0A0_, l0B2_, l0C0_, 64, 64, 128, false);
        SetFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        FixB16Halves(ubVNew_, l0C0_, 64, 64);
        AicPublishToBoth(kConsP);
        FixB16L1Matrix(l1VNew_, l0C0_, 64, 64, 64);
        SetFlag<HardEvent::FIX_MTE1>(0);

        WaitFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        IssueMm(l0A0_, l0B3_, l0C0_[kChunk * kHalfDim], 64, 64, 128, false);
        SetFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        LocalTensor<bfloat16_t> l1VNewC2 = l1VNew_[kChunk * kHalfDim];
        FixB16Halves(ubVNew_[kHalfRows * kHalfDim], l0C0_[kChunk * kHalfDim], 64, 64);
        AicPublishToBoth(kConsP1);
        FixB16L1Matrix(l1VNewC2, l0C0_[kChunk * kHalfDim], 64, 64, 64);
        SetFlag<HardEvent::FIX_MTE1>(1);
        SetFlag<HardEvent::FIX_M>(kL0CReleaseEventIds[0]);

        // Start D-C1 with Right-C1 only; C2 is loaded after its own ready event.
        AicWaitBoth(kConsRight);
        LocalTensor<bfloat16_t> dA = td_.stateVFirst != 0 ? l1Right_ : l1K_[slot];
        LocalTensor<bfloat16_t> dB = td_.stateVFirst != 0 ? l1K_[slot] : l1Right_;
        WaitFlag<HardEvent::M_MTE1>(0);
        LoadA(l0A0_, dA, 128, 64, true);
        LoadB(l0B0_, dB, 64, 64, true);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::FIX_M>(kL0CReleaseEventIds[3]);
        IssueMm(l0A0_, l0B0_, l0C1_, 128, 64, 64, true);
        SetFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        FixF32Halves(ubD_, l0C1_, 128, 64);
        SetFlag<HardEvent::FIX_M>(kL0CReleaseEventIds[3]);
        AicPublishToBoth(kConsD);

        // APrime sits behind QExp in bank1, so this load never invalidates the
        // QS left operand. Once Right-C2 arrives, load its B and queue D-C2
        // before consuming D-C1's release for the VNew prefetch.
        // APrime was joined and loaded in RunCubePrepare behind the T loads on
        // dedicated flag IDs 2/3, only consumed here after U-WS and the
        // v_new-decay work covered the transfer.
        WaitFlag<HardEvent::MTE2_MTE1>(2 + slot);
        LoadA(l0APrime_, l1APrime_[slot], 64, 64, false);
        SetFlag<HardEvent::MTE1_MTE2>(kConsAPrimeReleaseEventIds[slot]);
        AicWaitBoth(kConsRight1);
        LoadB(l0B1_, dB[kHalfDim * kHalfDim], 64, 64, true);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::FIX_M>(kL0CReleaseEventIds[4]);
        IssueMm(l0A0_, l0B1_, l0C1_[kStateElems / 2], 128, 64, 64, true);
        WaitFlag<HardEvent::M_MTE1>(0);
        WaitFlag<HardEvent::FIX_MTE1>(0);
        WaitFlag<HardEvent::FIX_MTE1>(1);
        LoadB(l0B3_, l1VNew_, 128, 64, true);
        SetFlag<HardEvent::M_MTE1>(1);
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        FixF32Halves(ubD_[kHalfDim * kHalfDim], l0C1_[kStateElems / 2], 128, 64);
        SetFlag<HardEvent::FIX_M>(kL0CReleaseEventIds[4]);
        AicPublishToBoth(kConsD1);

        WaitFlag<HardEvent::M_MTE1>(1);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        IssueMm(l0APrime_, l0B3_, qs, 64, 128, 64, false);
        SetFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        FixB16ScaledHalves(ubO_, qs, 64, 128, td_.scale);
        SetFlag<HardEvent::FIX_M>(QsReleaseEvent(chunk));
        AicPublishToBoth(kConsOL);
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
    LocalTensor<bfloat16_t> ubStateB_, ubU_, ubVNew_, ubRight_, ubW_, ubQ_, ubK_;
    LocalTensor<bfloat16_t> ubV_[2];
    LocalTensor<bfloat16_t> ubQHat_, ubKHat_, ubZeroBf_, ubO_, ubInitB_, ubOSB_;
    LocalTensor<float> ubG_[3], ubEnergy_[3], ubBeta_[3], ubEnergyRecip_[3];
    LocalTensor<uint32_t> ubShift_;
    LocalTensor<bfloat16_t> l1W_[2], l1U_[2], l1QExp_[2], l1K_[2], l1H_[2];
    LocalTensor<bfloat16_t> l1VNew_, l1Right_, l1APrime_[2], l1T_[2];
    LocalTensor<bfloat16_t> l0A0_, l0A1_, l0APrime_;
    LocalTensor<bfloat16_t> l0B0_, l0B1_, l0B2_, l0B3_;
    LocalTensor<float> l0C0_, l0C1_, l0C2_, l0CQS_[2];
};

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_CONSUMER_H
