/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 *
 * Producer role: L2Norm -> cumsum -> KK^T -> solve_tri -> recompute_wu ->
 * W, U, g_sum for one chain, one chunk per pipeline iteration.
 *
 * Every minimum unit is its own member function so a stage can be replayed or
 * compared against the serial Prepare kernel on device:
 *   Stage0Aux      resident I_vcs / I_64 / leaf zeros
 *   Stage1Inputs   q/k L2Norm, gate softplus + chunk cumsum, beta sigmoid
 *   Stage2Kkt      K K^T                       (AIC)
 *   Stage3Solve    -L, diagonal leaves, 32-leaf block inverse (AIV0)
 *   Stage4Y        Y = I + LeafLeft @ (-L)     (AIC)
 *   Stage5A        A = LeafLeft + Y @ LeafRight(AIC)
 *   Stage6WuInputs vb = v*beta, kbg = k_hat*beta*exp2(g) (AIV0)
 *   Stage7Wu       W = A @ kbg, U = A @ vb     (AIC)
 *   Stage8Drain    W/U UB -> GM
 *
 * AIV0 owns every vector stage; AIV1 only mirrors the handshakes because the
 * one-chunk map holds a single task. The AIC never touches GM: W/U are fixpiped
 * into AIV0's UB and drained from there, so the producer -> consumer edge is
 * AIV -> AIV and is covered by the mode-0 role barrier.
 */
#ifndef GDN_CHUNK64_SPARSE_PRODUCER_H
#define GDN_CHUNK64_SPARSE_PRODUCER_H

#include "sparse_barrier.h"
#include "sparse_common.h"
#include "sparse_fused_tiling.h"
#include "sparse_memory.h"
#include "sparse_prep_lib.h"

namespace GdnChunk64Sparse {

namespace Prep = ChunkGatedDeltaRuleFwdPrepare;

// Intra-group handshake ids. AIV1 answers at id + 16 (kSubBlockFlagOffset).
constexpr uint16_t kProdFlagInputs = 0;   // AIV -> AIC: k_hat (and gate) in L1/UB
constexpr uint16_t kProdFlagKkt = 1;      // AIC -> AIV: KK^T in UB
constexpr uint16_t kProdFlagSolve = 2;    // AIV -> AIC: -L, leaves, inverse in L1
constexpr uint16_t kProdFlagWuIn = 3;     // AIV -> AIC: vb/kbg in L1
constexpr uint16_t kProdFlagWu = 4;       // AIC -> AIV: W/U in UB

// Local event ids (one chunk in flight, so a single bank is enough).
constexpr int32_t kProdEvtUb = 0;
constexpr int32_t kProdEvtPack = 1;

class ProducerRole {
public:
    // One-time launch setup. Per-chain state resolves in BeginChain so the
    // same role instance can walk several chains in group-serial mode.
    __aicore__ inline void Init(const PhaseArgs &args, const FusedTilingData &td)
    {
        td_ = td;

        b_ = td.prepare.inputBatchSize;
        hk_ = td.prepare.queryKeyHeadCount;
        hv_ = td.prepare.valueHeadCount;
        t_ = td.prepare.sequenceTokenLength;
        ch_ = td.prepare.tokensPerChunk > 0 ? td.prepare.tokensPerChunk : kChunk;
        useNorm_ = td.prepare.enableQueryKeyL2NormInKernel != 0;
        useGate_ = td.prepare.enableFusedGateSoftplus != 0;
        useBetaSigmoid_ = td.prepare.enableBetaSigmoid != 0;
        hasDtBias_ = td.prepare.hasGateDtBias != 0;
        allowNegEigval_ = td.prepare.scaleBetaByTwoWhenNegEigval != 0;
        useExp2_ = td.prepare.useExp2ForGateCumsum != 0;
        outputA_ = td.prepare.hasAOutput != 0;
        hasQHat_ = td.prepare.hasQueryHatGmOutput != 0;
        hasKHat_ = td.prepare.hasKeyHatGmOutput != 0;
        varlen_ = td.prepare.isVariableLengthPacked != 0;

        gmQ_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.q));
        gmK_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.k));
        gmV_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.v));
        gmG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.rawG));
        gmBeta_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.beta));
        hasALog_ = args.aLog != nullptr ? 1 : 0;
        if (hasALog_ != 0) {
            gmALog_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.aLog));
        }
        if (args.dtBias != nullptr) {
            gmDt_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.dtBias));
        }
        gmGOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.gCumsum));
        gmBetaEff_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.betaEff));
        if (useNorm_) {
            gmQHat_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.qHat));
            gmKHat_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.kHat));
            gmQRstd_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.qRstd));
            gmKRstd_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.kRstd));
            gmKHatSrc_ = gmKHat_;
        } else {
            gmKHatSrc_ = gmK_;
        }
        gmW_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.w));
        gmU_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.u));
        if (outputA_) {
            gmA_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.a));
        }
        InitOnChip();
        if ASCEND_IS_AIC {
            // Prime the L0 event credits once per launch; every Mmad/Fixpipe
            // below pairs Set with the next Wait.
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        }
    }

    __aicore__ inline void BeginChain(const PhaseArgs &args, uint32_t chainIdx)
    {
        chainIdx_ = chainIdx;
        loc_ = ResolveChain(td_, args.cuSeqlens, chainIdx);
    }

    // Return the fixed L0 event ids to their primed state so a later launch or
    // the serial fallback cannot inherit a stale credit.
    __aicore__ inline void Finish()
    {
        if ASCEND_IS_AIC {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline int64_t ChunkCount() const
    {
        return loc_.chunks;
    }

    __aicore__ inline void RunChunk(int64_t localChunk)
    {
        const ChunkWindow window = ResolveWindow(loc_, localChunk);
        if ASCEND_IS_AIV {
            Stage0Aux();
            if (GetSubBlockIdx() == 0) {
                Stage1Inputs(window);
                Stage3Solve(window);
                Stage6WuInputs(window);
                Stage8Drain(window);
            } else {
                // Mirror every handshake without local work.
                AivSetMte3(kProdFlagInputs);
                AivWaitV(kProdFlagKkt);
                AivSetMte3(kProdFlagSolve);
                AivSetMte3(kProdFlagWuIn);
                AivWaitV(kProdFlagWu);
            }
        }
        if ASCEND_IS_AIC {
            Stage2Kkt();
            Stage4Y(window);
            Stage5A(window);
            Stage7Wu(window);
        }
    }

private:
    __aicore__ inline void InitOnChip()
    {
        OnChipBuffer buf;
        ubK_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kProdUbK);
        ubQHat_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kProdUbQHat);
        ubKHat_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kProdUbKHat);
        ubQ_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kProdUbQ);
        ubKkt_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kProdUbKkt);
        ubL_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kProdUbL);
        ubLPacked_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kProdUbLPacked);
        ubResVcs_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kProdUbResVcs);
        ubIVcs_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kProdUbIVcs);
        ubW_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kProdUbW);
        ubU_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kProdUbU);
        ubVb_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kProdUbVb);
        ubScale_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kProdUbScale);
        ubVcsIdx_ = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(kProdUbVcsIdx);
        // g' [64], beta [64], rstd [64] + [64] share the 1 KiB scale slot.
        ubGPrime_ = ubScale_;
        ubBeta_ = ubScale_[64];
        ubQRstd_ = ubScale_[128];
        ubKRstd_ = ubScale_[192];

        l1KHat_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kProdL1KHat);
        l1NegL_ = buf.GetBuffer<BufferType::ASCEND_CB, float>(kProdL1NegL);
        l1LeafL_ = buf.GetBuffer<BufferType::ASCEND_CB, float>(kProdL1LeafL);
        l1LeafR_ = buf.GetBuffer<BufferType::ASCEND_CB, float>(kProdL1LeafR);
        l1Y_ = buf.GetBuffer<BufferType::ASCEND_CB, float>(kProdL1Y);
        l1A_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kProdL1A);
        l1Kbg_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kProdL1Kbg);
        l1Vb_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kProdL1Vb);
        l1I_ = buf.GetBuffer<BufferType::ASCEND_CB, float>(kProdL1I);

        for (uint32_t b = 0; b < 2; ++b) {
            l0A_[b] = buf.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(kProdL0Pair[b]);
            l0B_[b] = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kProdL0Pair[b]);
            l0Af_[b] = buf.GetBuffer<BufferType::ASCEND_L0A, float>(kProdL0Pair[b]);
            l0Bf_[b] = buf.GetBuffer<BufferType::ASCEND_L0B, float>(kProdL0Pair[b]);
        }
        for (uint32_t t = 0; t < 4; ++t) {
            l0CTask_[t] = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kProdL0CTile[t]);
        }
    }

    // ========================= Stage 0 =========================
    __aicore__ inline void Stage0Aux()
    {
        if (auxReady_ != 0) {
            return;
        }
        if (GetSubBlockIdx() == 0) {
            Prep::FillVcsIdentity(ubIVcs_, ubVcsIdx_);
            AscendC::Duplicate(ubKkt_, 0.0f, static_cast<int32_t>(kMatrixElems));
            SetFlag<AscendC::HardEvent::V_MTE3>(kProdEvtUb);
            WaitFlag<AscendC::HardEvent::V_MTE3>(kProdEvtUb);
            Prep::UbToL1Fp32(l1LeafL_, ubKkt_, static_cast<uint32_t>(kChunk));
            Prep::UbToL1Fp32(l1LeafR_, ubKkt_, static_cast<uint32_t>(kChunk));
            Prep::Identity64VF(ubKkt_);
            SetFlag<AscendC::HardEvent::V_MTE3>(kProdEvtUb);
            WaitFlag<AscendC::HardEvent::V_MTE3>(kProdEvtUb);
            Prep::UbNd64ToL1Nz8(l1I_, ubKkt_);
            SetFlag<AscendC::HardEvent::MTE3_V>(kProdEvtUb);
            WaitFlag<AscendC::HardEvent::MTE3_V>(kProdEvtUb);
        }
        auxReady_ = 1;
    }

    // ========================= Stage 1 =========================
    // Rows past the window are zeroed so the cube K dimension stays valid.
    __aicore__ inline void Stage1Inputs(const ChunkWindow &window)
    {
        const uint32_t rows = static_cast<uint32_t>(window.rows);

        SetFlag<AscendC::HardEvent::V_MTE2>(kProdEvtUb);
        WaitFlag<AscendC::HardEvent::V_MTE2>(kProdEvtUb);
        if (rows < static_cast<uint32_t>(kChunk)) {
            AscendC::Duplicate(ubQ_[static_cast<uint32_t>(rows * kHeadDim)], static_cast<bfloat16_t>(0),
                               static_cast<int32_t>((kChunk - window.rows) * kHeadDim));
            AscendC::Duplicate(ubK_[static_cast<uint32_t>(rows * kHeadDim)], static_cast<bfloat16_t>(0),
                               static_cast<int32_t>((kChunk - window.rows) * kHeadDim));
        }
        Prep::CopyGmToUbElems(ubQ_, gmQ_[OffsetBhtd(loc_.batch, loc_.kh, window.tokenStart, hk_, t_, kHeadDim)],
                              rows * static_cast<uint32_t>(kHeadDim));
        Prep::CopyGmToUbElems(ubK_,
                              gmK_[OffsetBhtd(loc_.batch, loc_.kh, window.tokenStart, hk_, t_, kHeadDim)],
                              rows * static_cast<uint32_t>(kHeadDim));
        SetFlag<AscendC::HardEvent::MTE2_V>(kProdEvtUb);
        WaitFlag<AscendC::HardEvent::MTE2_V>(kProdEvtUb);

        if (useNorm_) {
            Prep::L2NormK128VF<bfloat16_t>(ubQ_, ubQHat_, ubQRstd_, static_cast<uint32_t>(kChunk));
            Prep::L2NormK128VF<bfloat16_t>(ubK_, ubKHat_, ubKRstd_, static_cast<uint32_t>(kChunk));
        } else {
            // k_hat is k; q_hat is not consumed by O in this mode either.
            AscendC::DataCopy(ubKHat_, ubK_, static_cast<int32_t>(kChunkElems));
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(kProdEvtUb);
        WaitFlag<AscendC::HardEvent::V_MTE3>(kProdEvtUb);
        if (useNorm_ && hasQHat_) {
            Prep::CopyUbToGmElems(
                gmQHat_[OffsetBhtd(loc_.batch, loc_.kh, window.tokenStart, hk_, t_, kHeadDim)],
                ubQHat_, rows * static_cast<uint32_t>(kHeadDim));
            Prep::CopyUbToGmElems(gmQRstd_[OffsetBht(loc_.batch, loc_.kh, window.tokenStart, hk_, t_)],
                                  ubQRstd_, rows);
        }
        if (useNorm_ && hasKHat_) {
            Prep::CopyUbToGmElems(
                gmKHat_[OffsetBhtd(loc_.batch, loc_.kh, window.tokenStart, hk_, t_, kHeadDim)],
                ubKHat_, rows * static_cast<uint32_t>(kHeadDim));
            Prep::CopyUbToGmElems(gmKRstd_[OffsetBht(loc_.batch, loc_.kh, window.tokenStart, hk_, t_)],
                                  ubKRstd_, rows);
        }
        Prep::UploadBf16NdToL1<bfloat16_t>(l1KHat_, ubKHat_, static_cast<uint32_t>(kHeadDim));
        SetFlag<AscendC::HardEvent::MTE3_V>(kProdEvtUb);
        WaitFlag<AscendC::HardEvent::MTE3_V>(kProdEvtUb);

        Stage1Gate(window, rows);
        // Publish k_hat in L1 only after every vector write above retired.
        AscendC::PipeBarrier<PIPE_ALL>();
        AivSetMte3(kProdFlagInputs);
    }

    __aicore__ inline void Stage1Gate(const ChunkWindow &window, uint32_t rows)
    {
        const int64_t gOff = OffsetBht(loc_.batch, loc_.hv, window.tokenStart, hv_, t_);
        if (rows < static_cast<uint32_t>(kChunk)) {
            AscendC::Duplicate(ubGPrime_, 0.0f, static_cast<int32_t>(kChunk));
            AscendC::Duplicate(ubBeta_, 0.0f, static_cast<int32_t>(kChunk));
        }
        Prep::CopyGmToUbElems(ubGPrime_, gmG_[gOff], rows);
        Prep::CopyGmToUbElems(ubBeta_, gmBeta_[gOff], rows);
        SetFlag<AscendC::HardEvent::MTE2_V>(kProdEvtUb);
        WaitFlag<AscendC::HardEvent::MTE2_V>(kProdEvtUb);

        if (useGate_) {
            float aLog = 0.0f;
            float dtBias = 0.0f;
            if (hasALog_ != 0) {
                aLog = gmALog_.GetValue(loc_.hv);
            }
            if (hasDtBias_ != 0) {
                dtBias = gmDt_.GetValue(loc_.hv);
            }
            Prep::GateSoftplusVF<float>(ubGPrime_, ubGPrime_, aLog, dtBias,
                                        static_cast<uint32_t>(kChunk));
        }
        // kGdnRcpLn2 is a global constant in prepare_common.h (outside the
        // ChunkGatedDeltaRuleFwdPrepare namespace), so it must stay unqualified.
        Prep::CumsumScaleVF(ubGPrime_, useExp2_ ? kGdnRcpLn2 : 1.0f,
                            static_cast<uint32_t>(kChunk));
        if (useBetaSigmoid_) {
            Prep::BetaSigmoidVF<float>(ubBeta_, ubBeta_, allowNegEigval_ ? 2.0f : 1.0f,
                                       static_cast<uint32_t>(kChunk));
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(kProdEvtUb);
        WaitFlag<AscendC::HardEvent::V_MTE3>(kProdEvtUb);
        Prep::CopyUbToGmElems(gmGOut_[gOff], ubGPrime_, rows);
        if (useBetaSigmoid_) {
            Prep::CopyUbToGmElems(gmBetaEff_[gOff], ubBeta_, rows);
        }
        SetFlag<AscendC::HardEvent::MTE3_V>(kProdEvtUb);
        WaitFlag<AscendC::HardEvent::MTE3_V>(kProdEvtUb);
    }

    // ========================= Stage 2 (AIC) =========================
    // KKT = k̂ k̂ᵀ. The verified serial Prepare calls this with transposeB=false
    // ("B NZ is [n,k]: same k̂ tile as A, n=m"); transposeB=true would read a
    // wrongly transposed B and silently corrupt the gate matrix.
    __aicore__ inline void Stage2Kkt()
    {
        AicWaitMte1(kProdFlagInputs);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        Prep::MatmulToL0C<bfloat16_t>(l1KHat_, l1KHat_, l0A_[0], l0B_[0], l0CTask_[0],
                                      static_cast<int32_t>(kChunk), static_cast<int32_t>(kChunk),
                                      static_cast<int32_t>(kHeadDim), true, false, false, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        Prep::FixpipeL0cToUbFp32Nd(ubKkt_, l0CTask_[0], static_cast<uint32_t>(kChunk), 0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        AicSetFix(kProdFlagKkt);
    }

    // ========================= Stage 3 (AIV0) =========================
    __aicore__ inline void Stage3Solve(const ChunkWindow &)
    {
        if (useExp2_) {
            Prep::GateLowerLVF<true>(ubGPrime_, ubBeta_, ubL_);
        } else {
            Prep::GateLowerLVF<false>(ubGPrime_, ubBeta_, ubL_);
        }
        AscendC::DataCopy(ubResVcs_, ubIVcs_, static_cast<int32_t>(Prep::kVcsPackedElems32));
        AivWaitV(kProdFlagKkt);
        Prep::MulKktGateVF(ubKkt_, ubL_);
        SetFlag<AscendC::HardEvent::V_MTE3>(kProdEvtUb);
        WaitFlag<AscendC::HardEvent::V_MTE3>(kProdEvtUb);
        Prep::PackDiagLeavesFromUb(ubLPacked_, ubL_);
        SetFlag<AscendC::HardEvent::MTE3_V>(kProdEvtPack);
        WaitFlag<AscendC::HardEvent::MTE3_V>(kProdEvtPack);
        Prep::UbNd64ToL1Nz8(l1NegL_, ubL_);
        Prep::MulReduceScatterVF32(ubResVcs_, ubLPacked_, ubResVcs_, ubVcsIdx_);
        SetFlag<AscendC::HardEvent::V_MTE3>(kProdEvtUb);
        WaitFlag<AscendC::HardEvent::V_MTE3>(kProdEvtUb);
        Prep::UploadDiagLeavesToL1(l1LeafR_, l1LeafL_, ubResVcs_);
        AscendC::PipeBarrier<PIPE_ALL>();
        AivSetMte3(kProdFlagSolve);
    }

    // ========================= Stage 4 (AIC) =========================
    // Y = I + LeafLeft @ (-L), kept in L1 (the single-chunk map removes the
    // serial Prepare's GM NZ round trip).
    __aicore__ inline void Stage4Y(const ChunkWindow &)
    {
        AicWaitMte1(kProdFlagSolve);
        const int32_t bt = static_cast<int32_t>(kChunk);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        Prep::LoadL1NzToL0AB<float>(l1I_, l1I_, l0Af_[0], l0Bf_[0], bt, bt, bt, true, false, 0);
        Prep::MmadL0ABToL0C<float>(l0Af_[0], l0Bf_[0], l0CTask_[0], bt, bt, bt, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        // LeafLeft @ (-L) accumulates onto I.
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        Prep::MatmulToL0C<float>(l1LeafL_, l1NegL_, l0Af_[1], l0Bf_[1], l0CTask_[0],
                                 bt, bt, bt, false, true, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        FixpipeYToL1();
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(0);
    }

    __aicore__ inline void FixpipeYToL1()
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> p;
        p.nSize = static_cast<uint16_t>(kChunk);
        p.mSize = static_cast<uint16_t>(kChunk);
        p.srcStride = static_cast<uint16_t>(kChunk);
        p.dstStride = static_cast<uint16_t>(kChunk * 8);
        p.quantPre = QuantMode_t::NoQuant;
        p.unitFlag = 0;
        p.isChannelSplit = false;
        AscendC::Fixpipe<float, float, Prep::CFG_NZ_L1>(l1Y_, l0CTask_[0], p);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(0);
    }

    // ========================= Stage 5 (AIC) =========================
    __aicore__ inline void Stage5A(const ChunkWindow &window)
    {
        const int32_t bt = static_cast<int32_t>(kChunk);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        Prep::MatmulToL0C<float>(l1I_, l1LeafL_, l0Af_[0], l0Bf_[0], l0CTask_[0],
                                 bt, bt, bt, true, false, false, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        Prep::MatmulToL0C<float>(l1Y_, l1LeafR_, l0Af_[1], l0Bf_[1], l0CTask_[0],
                                 bt, bt, bt, false, false, false, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        if (outputA_) {
            Prep::FixpipeL0cToGmNd<bfloat16_t>(
                gmA_[OffsetBhtd(loc_.batch, loc_.hv, window.tokenStart, hv_, t_, kChunk)],
                l0CTask_[0], static_cast<uint32_t>(window.rows),
                static_cast<uint32_t>(kChunk), static_cast<uint32_t>(kChunk));
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(0);
            // FixpipeAToL1 below publishes the same L0C tile to L1 for Stage 7.
            // No GM round trip: it would race with that Fixpipe on l1A_ and, on
            // 3510, the GM->L1 DataCopy path reserves mode-4 flagIds 1/17.
        }
        FixpipeAToL1();
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(0);
    }

    __aicore__ inline void FixpipeAToL1()
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> p;
        p.nSize = static_cast<uint16_t>(kChunk);
        p.mSize = static_cast<uint16_t>(kChunk);
        p.srcStride = static_cast<uint16_t>(kChunk);
        p.dstStride = static_cast<uint16_t>(kChunk * 16);
        p.quantPre = QuantMode_t::F322BF16;
        p.unitFlag = 0;
        p.isChannelSplit = false;
        AscendC::Fixpipe<bfloat16_t, float, Prep::CFG_NZ_L1>(l1A_, l0CTask_[0], p);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(0);
    }

    // ========================= Stage 6 (AIV0) =========================
    __aicore__ inline void Stage6WuInputs(const ChunkWindow &window)
    {
        const uint32_t rows = static_cast<uint32_t>(window.rows);
        // vb = v * beta, kbg = k_hat * beta * exp2(g).
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kProdEvtUb);
        WaitFlag<AscendC::HardEvent::V_MTE2>(kProdEvtUb);
        if (rows < static_cast<uint32_t>(kChunk)) {
            AscendC::Duplicate(ubQ_, static_cast<bfloat16_t>(0),
                               static_cast<int32_t>((kChunk - window.rows) * kHeadDim));
        }
        Prep::CopyGmToUbElems(ubQ_, gmV_[OffsetBhtd(loc_.batch, loc_.hv, window.tokenStart, hv_, t_, kHeadDim)],
                              rows * static_cast<uint32_t>(kHeadDim));
        SetFlag<AscendC::HardEvent::MTE2_V>(kProdEvtUb);
        WaitFlag<AscendC::HardEvent::MTE2_V>(kProdEvtUb);
        Prep::ScaleRowsVF<bfloat16_t>(ubQ_, ubVb_, ubBeta_, static_cast<uint32_t>(kChunk),
                                      static_cast<uint32_t>(kHeadDim));
        Prep::UploadBf16NdToL1<bfloat16_t>(l1Vb_, ubVb_, static_cast<uint32_t>(kHeadDim));
        if (useExp2_) {
            Prep::ScaleRowsBetaExp2gVF<bfloat16_t, true>(ubKHat_, ubK_, ubBeta_, ubGPrime_,
                                                         static_cast<uint32_t>(kChunk));
        } else {
            Prep::ScaleRowsBetaExp2gVF<bfloat16_t, false>(ubKHat_, ubK_, ubBeta_, ubGPrime_,
                                                          static_cast<uint32_t>(kChunk));
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(kProdEvtUb);
        WaitFlag<AscendC::HardEvent::V_MTE3>(kProdEvtUb);
        Prep::UploadBf16NdToL1<bfloat16_t>(l1Kbg_, ubK_, static_cast<uint32_t>(kHeadDim));
        AscendC::PipeBarrier<PIPE_ALL>();
        AivSetMte3(kProdFlagWuIn);
    }

    // ========================= Stage 7 (AIC) =========================
    // W/U are [64,128] and live in UB as BF16 for the GM drain. Prepare's
    // FixpipeL0cToUbFp32Nd is square (nSize = mSize = chunkSize) and takes a
    // float destination, so W/U need this bf16 NZ2ND variant with explicit m/n.
    __aicore__ inline void FixpipeWuToUb(LocalTensor<bfloat16_t> ub)
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> p;
        p.nSize = static_cast<uint16_t>(kHeadDim);
        p.mSize = static_cast<uint16_t>(kChunk);
        p.srcStride = static_cast<uint16_t>(kChunk);
        p.dstStride = static_cast<uint32_t>(kHeadDim);
        p.quantPre = QuantMode_t::F322BF16;
        p.dualDstCtl = 0;
        p.isChannelSplit = false;
        p.unitFlag = 0;
        p.params.ndNum = 1;
        p.params.srcNdStride = 0;
        p.params.dstNdStride = 0;
        p.subBlockId = false;
        AscendC::Fixpipe<bfloat16_t, float, Prep::CFG_ROW_MAJOR_UB>(ub, l0CTask_[0], p);
    }

    __aicore__ inline void Stage7Wu(const ChunkWindow &)
    {
        AicWaitMte1(kProdFlagWuIn);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        Prep::WuMatmulToL0C<bfloat16_t>(l1A_, l1Kbg_, l0A_[0], l0B_[0], l0CTask_[0],
                                        static_cast<int32_t>(kChunk), static_cast<int32_t>(kHeadDim),
                                        static_cast<int32_t>(kChunk), 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        FixpipeWuToUb(ubW_);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        Prep::WuMatmulToL0C<bfloat16_t>(l1A_, l1Vb_, l0A_[0], l0B_[0], l0CTask_[0],
                                        static_cast<int32_t>(kChunk), static_cast<int32_t>(kHeadDim),
                                        static_cast<int32_t>(kChunk), 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        FixpipeWuToUb(ubU_);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        AicSetFix(kProdFlagWu);
    }

    // ========================= Stage 8 (AIV0) =========================
    __aicore__ inline void Stage8Drain(const ChunkWindow &window)
    {
        AivWaitV(kProdFlagWu);
        const uint32_t elems = static_cast<uint32_t>(window.rows * kHeadDim);
        Prep::CopyUbToGmElems(gmW_[OffsetBhtd(loc_.batch, loc_.hv, window.tokenStart, hv_, t_, kHeadDim)],
                              ubW_, elems);
        Prep::CopyUbToGmElems(gmU_[OffsetBhtd(loc_.batch, loc_.hv, window.tokenStart, hv_, t_, kHeadDim)],
                              ubU_, elems);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kProdEvtUb);
        WaitFlag<AscendC::HardEvent::MTE3_V>(kProdEvtUb);
    }

    FusedTilingData td_{};
    ChainLoc loc_{};
    uint32_t chainIdx_{0};
    int64_t b_{0}, hk_{0}, hv_{0}, t_{0}, ch_{0};
    int64_t useNorm_{0}, useGate_{0}, useBetaSigmoid_{0}, hasDtBias_{0};
    int64_t allowNegEigval_{0}, useExp2_{0}, outputA_{0}, hasQHat_{0}, hasKHat_{0}, varlen_{0};
    int64_t hasALog_{0};
    int64_t auxReady_{0};

    AscendC::GlobalTensor<bfloat16_t> gmQ_, gmK_, gmV_, gmQHat_, gmKHat_, gmKHatSrc_;
    AscendC::GlobalTensor<float> gmG_, gmBeta_, gmALog_, gmDt_, gmGOut_, gmBetaEff_, gmQRstd_, gmKRstd_;
    AscendC::GlobalTensor<bfloat16_t> gmW_, gmU_, gmA_;

    AscendC::LocalTensor<bfloat16_t> ubQ_, ubK_, ubQHat_, ubKHat_, ubVb_, ubW_, ubU_;
    AscendC::LocalTensor<float> ubKkt_, ubL_, ubLPacked_, ubResVcs_, ubIVcs_, ubScale_;
    AscendC::LocalTensor<float> ubGPrime_, ubBeta_, ubQRstd_, ubKRstd_;
    AscendC::LocalTensor<uint32_t> ubVcsIdx_;

    AscendC::LocalTensor<bfloat16_t> l1KHat_, l1A_, l1Kbg_, l1Vb_;
    AscendC::LocalTensor<float> l1NegL_, l1LeafL_, l1LeafR_, l1Y_, l1I_;

    AscendC::LocalTensor<bfloat16_t> l0A_[2], l0B_[2];
    AscendC::LocalTensor<float> l0Af_[2], l0Bf_[2], l0CTask_[4];
};

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_PRODUCER_H
