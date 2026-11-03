/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 *
 * Consumer role: H -> O for one chain, one chunk per pipeline iteration.
 *
 *   LoadInputs     U/W/q_hat/k_hat/g -> UB, gate rows -> UB
 *   UploadInputs   W/Q/K + H (state mirror) -> L1
 *   CubeP          P = W @ H                 (AIC)
 *   VNewRight      v_new = U - P, right = gateScale * v_new
 *   CubeQk         A_raw = Q @ K^T           (AIC)
 *   CubeOS         O_s = Q @ H               (AIC, reuses L0B H)
 *   CubeD          D = K^T @ right           (AIC; right^T @ K when stateVFirst)
 *   APrime         gateA, A' = tril(A_raw * gateA), -> L1
 *   CubeOL         O_l = A' @ V_new          (AIC)
 *   Output         o = (O_s * gateO + O_l) * scale -> GM
 *   StateUpdate    R = alpha * R + D         (K half per AIV), h/v_new -> GM
 *
 * The two AIVs split chunk rows 32/32 for v_new, right and the whole O
 * section, and split the recurrent state along K 64/64 (or V for
 * stateVFirst). Every handshake is a group-local mode-4 flag; the AIC never
 * touches GM, so the producer -> consumer publication edge remains AIV -> AIV
 * and is covered by the mode-0 role barrier in sparse_pipeline.h.
 */
#ifndef GDN_CHUNK64_SPARSE_CONSUMER_H
#define GDN_CHUNK64_SPARSE_CONSUMER_H

#include "kernel_operator.h"
#include "sparse_barrier.h"
#include "sparse_common.h"
#include "sparse_consumer_vf.h"
#include "sparse_fused_tiling.h"
#include "sparse_memory.h"
#include "sparse_prep_lib.h"

namespace GdnChunk64Sparse {

namespace Prep = ChunkGatedDeltaRuleFwdPrepare;

// Intra-group handshake ids (AIV1 answers at id + kSubBlockFlagOffset).
constexpr uint16_t kConsFlagInputs = 0;  // AIV -> AIC: W/Q/K/H in L1
constexpr uint16_t kConsFlagP = 1;       // AIC -> AIV: P row half in UB
constexpr uint16_t kConsFlagQk = 2;      // AIC -> AIV: A_raw row half in UB
constexpr uint16_t kConsFlagOS = 3;      // AIC -> AIV: O_s row half in UB
constexpr uint16_t kConsFlagVn = 4;      // AIV -> AIC: V_new/right in L1
constexpr uint16_t kConsFlagD = 5;       // AIC -> AIV: D state half in UB
constexpr uint16_t kConsFlagAPrime = 6;  // AIV -> AIC: A' in L1
constexpr uint16_t kConsFlagO = 9;       // AIC -> AIV: O_l row half in UB

// Local event ids (one chunk in flight).
constexpr int32_t kConsEvtUb = 0;
constexpr int32_t kConsEvtScalar = 2;
constexpr int32_t kConsEvtCube = 0;

// Cube loaders. srcStride is the source row count in 16-row fractals, which
// differs between [m,k] and [k,m] sources; this is the chunk32 convention
// (verified there) rather than Prepare's fixed-M variant.
__aicore__ inline void LoadL1ToL0A(LocalTensor<bfloat16_t> l1A, LocalTensor<bfloat16_t> l0A,
                                   int32_t m, int32_t k, bool transposeA)
{
    const uint16_t mFracs = static_cast<uint16_t>(m / 16);
    const uint16_t kFracs = static_cast<uint16_t>(k / 16);
    AscendC::LoadData2DParamsV2 p;
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.sid = 0;
    if (transposeA) {
        p.mStep = kFracs;
        p.kStep = mFracs;
        p.srcStride = kFracs;
        p.dstStride = mFracs;
        p.ifTranspose = true;
    } else {
        p.mStep = mFracs;
        p.kStep = kFracs;
        p.srcStride = mFracs;
        p.dstStride = mFracs;
        p.ifTranspose = false;
    }
    AscendC::LoadData(l0A, l1A, p);
}

__aicore__ inline void LoadL1ToL0B(LocalTensor<bfloat16_t> l1B, LocalTensor<bfloat16_t> l0B,
                                   int32_t n, int32_t k, bool transposeB)
{
    const uint16_t bMFracs = transposeB ? static_cast<uint16_t>(k / 16)
                                        : static_cast<uint16_t>(n / 16);
    const uint16_t bKFracs = transposeB ? static_cast<uint16_t>(n / 16)
                                        : static_cast<uint16_t>(k / 16);
    const uint16_t bDst = transposeB ? static_cast<uint16_t>(n / 16) : bMFracs;
    AscendC::LoadData2DParamsV2 p;
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.mStep = bMFracs;
    p.kStep = bKFracs;
    p.srcStride = bMFracs;
    p.dstStride = bDst;
    p.ifTranspose = transposeB;
    p.sid = 0;
    AscendC::LoadData(l0B, l1B, p);
}

__aicore__ inline void LoadL1ToL0AB(LocalTensor<bfloat16_t> l1A, LocalTensor<bfloat16_t> l1B,
                                    LocalTensor<bfloat16_t> l0A, LocalTensor<bfloat16_t> l0B,
                                    int32_t m, int32_t n, int32_t k,
                                    bool transposeA, bool transposeB)
{
    LoadL1ToL0A(l1A, l0A, m, k, transposeA);
    LoadL1ToL0B(l1B, l0B, n, k, transposeB);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kConsEvtCube);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kConsEvtCube);
}

// L0C fp32 NZ -> one AIV's row-major ND UB. mSize is the per-AIV half; the
// source keeps the full M stride, and the second half starts at mFrac =
// (fullM/2)/16, i.e. (fullM/2)*16 floats into the first column fractal.
__aicore__ inline void FixpipeHalfFp32Ub(LocalTensor<float> ub, LocalTensor<float> l0C,
                                         uint32_t fullM, uint32_t n)
{
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> p;
    p.nSize = static_cast<uint16_t>(n);
    p.mSize = static_cast<uint16_t>(fullM / 2);
    p.srcStride = static_cast<uint16_t>(fullM);
    p.dstStride = static_cast<uint32_t>(n);
    p.quantPre = QuantMode_t::NoQuant;
    p.dualDstCtl = 0;
    p.isChannelSplit = false;
    p.unitFlag = 0;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    p.subBlockId = false;
    AscendC::Fixpipe<float, float, Prep::CFG_ROW_MAJOR_UB>(ub, l0C, p);
    p.subBlockId = true;
    AscendC::Fixpipe<float, float, Prep::CFG_ROW_MAJOR_UB>(
        ub, l0C[static_cast<uint32_t>(fullM / 2) * 16U], p);
}

class ConsumerRole {
public:
    // One-time launch setup. Per-chain state and the initial-state mirror
    // resolve in BeginChain so the same role instance can walk several
    // chains in group-serial mode.
    __aicore__ inline void Init(const PhaseArgs &args, const FusedTilingData &td)
    {
        td_ = td;

        hk_ = td.prepare.queryKeyHeadCount;
        hv_ = td.prepare.valueHeadCount;
        t_ = td.prepare.sequenceTokenLength;
        const int64_t chunk = td.prepare.tokensPerChunk > 0 ? td.prepare.tokensPerChunk : kChunk;
        chunksPerBatch_ = (t_ + chunk - 1) / chunk;
        useNorm_ = td.prepare.enableQueryKeyL2NormInKernel != 0;
        useExp2_ = td.useExp2 != 0;
        stateVFirst_ = td.stateVFirst != 0;
        stateFp32_ = td.stateDtype == 0;
        storeFinal_ = td.h.storeFinalState != 0;
        scale_ = td.o.scale;

        if (useNorm_) {
            gmQ_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.qHat));
            gmK_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.kHat));
        } else {
            gmQ_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.q));
            gmK_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.k));
        }
        gmVNew_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.vNew));
        gmW_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.w));
        gmU_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.u));
        gmG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.gCumsum));
        gmO_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.o));
        gmH_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.h));
        if (args.initialState != nullptr) {
            hasInitial_ = 1;
            if (stateFp32_) {
                gmInitF_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.initialState));
            } else {
                gmInitB_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.initialState));
            }
        }
        if (args.finalState != nullptr) {
            if (stateFp32_) {
                gmFinalF_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.finalState));
            } else {
                gmFinalB_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.finalState));
            }
        }

        InitOnChip();
        if ASCEND_IS_AIC {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kConsEvtCube);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kConsEvtCube);
        }
    }

    // Per-chain start: resolve the chain and load its initial state mirror.
    __aicore__ inline void BeginChain(const PhaseArgs &args, uint32_t chainIdx)
    {
        chainIdx_ = chainIdx;
        loc_ = ResolveChain(td_, args.cuSeqlens, chainIdx);
        if ASCEND_IS_AIV {
            InitState();
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
            const uint32_t firstRow = GetSubBlockIdx() * static_cast<uint32_t>(kHalfRows);
            const uint32_t rows = static_cast<uint32_t>(window.rows);
            const uint32_t halfRows = static_cast<uint32_t>(kHalfRows);
            const uint32_t nRows = rows > firstRow ? rows - firstRow : 0U;
            RunChunkAiv(localChunk, window, firstRow, nRows < halfRows ? nRows : halfRows);
        }
        if ASCEND_IS_AIC {
            RunChunkAic(localChunk);
        }
    }

    // Final state for the current chain. The last state update already wrote
    // the GM h record for its own chunk start; only the post-last-chunk state
    // is left, and an empty chain matches the standalone H kernel by not
    // writing.
    __aicore__ inline void EndChain()
    {
        if ASCEND_IS_AIV {
            if (loc_.chunks > 0 && storeFinal_) {
                StoreFinalState();
            }
        }
    }

    // L0 event credits, once at launch end after every chain drained.
    __aicore__ inline void Finish()
    {
        if ASCEND_IS_AIC {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kConsEvtCube);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kConsEvtCube);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    __aicore__ inline void InitOnChip()
    {
        OnChipBuffer buf;
        ubStateF_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kConsUbStateF);
        ubStateB_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kConsUbStateB);
        ubD_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kConsUbD);
        ubP_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kConsUbP);
        ubU_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kConsUbU);
        ubVNew_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kConsUbVNew);
        ubRight_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kConsUbRight);
        ubOS_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kConsUbOS);
        ubOL_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kConsUbOL);
        ubARaw_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kConsUbARaw);
        ubGateA_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kConsUbGateA);
        ubAPrime_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kConsUbAPrime);
        ubG_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kConsUbGates);
        ubGateScale_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kConsUbGates + 256);
        ubGateO_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kConsUbGates + 512);
        ubScalar_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(kConsUbScalar);
        // Staging aliases. W shares the P result slot, Q the O_s slot and K the
        // O_l slot: each is uploaded to L1 before the fixpipe that overwrites
        // it, and the mode-2 input relay drains MTE3 in between.
        ubWStage_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kConsUbP);
        ubQStage_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kConsUbOS);
        ubKStage_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kConsUbOL);
        ubInitStage_ = buf.GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kConsUbD);

        l1W_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kConsL1W);
        l1Q_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kConsL1Q);
        l1K_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kConsL1K);
        l1H_[0] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kConsL1H[0]);
        l1H_[1] = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kConsL1H[1]);
        l1VNew_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kConsL1VNew);
        l1Right_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kConsL1Right);
        l1APrime_ = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kConsL1APrime);

        l0A_[0] = buf.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(kConsL0Pair[0]);
        l0A_[1] = buf.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(kConsL0Pair[1]);
        l0B_[0] = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kConsL0Pair[0]);
        l0B_[1] = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(kConsL0Wide);
        for (uint32_t t = 0; t < 4; ++t) {
            l0C_[t] = buf.GetBuffer<BufferType::ASCEND_L0C, float>(kConsL0CTile[t]);
        }
    }

    // Initial state lives in this AIV's state half: FP32 dtype goes straight
    // into stateF and is mirrored to BF16; BF16 dtype is staged through ubD.
    // With no initial state the recurrence starts from zeros.
    __aicore__ inline void InitState()
    {
        const int64_t stateElem = StateElem();
        if (hasInitial_ != 0) {
            if (stateFp32_) {
                AscendC::DataCopy(ubStateF_, gmInitF_[stateElem],
                                  static_cast<uint32_t>(kStateElems / 2));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kConsEvtUb);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kConsEvtUb);
                StateMirrorVF(ubStateF_, ubStateB_, static_cast<uint32_t>(kHalfDim));
            } else {
                AscendC::DataCopy(ubInitStage_, gmInitB_[stateElem],
                                  static_cast<uint32_t>(kStateElems / 2));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kConsEvtUb);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kConsEvtUb);
                StateLoadVF(ubStateF_, ubStateB_, ubInitStage_, static_cast<uint32_t>(kHalfDim));
            }
        } else {
            AscendC::Duplicate(ubStateF_, 0.0f, static_cast<int32_t>(kStateElems / 2));
            AscendC::Duplicate(ubStateB_, static_cast<bfloat16_t>(0),
                               static_cast<int32_t>(kStateElems / 2));
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kConsEvtUb);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kConsEvtUb);
    }

    __aicore__ inline int64_t StateElem() const
    {
        return (loc_.seq * hv_ + loc_.hv) * kStateElems +
               static_cast<int64_t>(GetSubBlockIdx()) * (kStateElems / 2);
    }

    __aicore__ inline int64_t GlobalChunk(int64_t localChunk) const
    {
        return td_.prepare.isVariableLengthPacked != 0 ? loc_.chunkBase + localChunk : localChunk;
    }

    // ========================= AIV =========================
    __aicore__ inline void RunChunkAiv(int64_t localChunk, const ChunkWindow &window,
                                       uint32_t firstRow, uint32_t nRows)
    {
        StoreH(localChunk);
        LoadInputs(window, firstRow, nRows);
        const float alpha = ComputeGates(window);
        UploadInputs(firstRow, localChunk);

        AivRelayToAic(kConsFlagInputs);
        AivWaitV(kConsFlagP);
        StageVNewRight(window, firstRow, nRows);
        AivRelayToAic(kConsFlagVn);

        AivWaitV(kConsFlagQk);
        StageAPrime(firstRow, nRows);
        AivRelayToAic(kConsFlagAPrime);

        AivWaitV(kConsFlagOS);
        GateOutputVF(ubOS_, ubGateO_, firstRow, nRows);
        AivWaitV(kConsFlagO);
        StageOutput(window, firstRow, nRows);

        AivWaitV(kConsFlagD);
        StateUpdateVF(ubStateF_, ubD_, ubStateB_, alpha, static_cast<uint32_t>(kHalfDim));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kConsEvtUb);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kConsEvtUb);
    }

    // h[chunk] is the state before this chunk. Both AIVs write their own half
    // in the physical order selected by stateVFirst.
    __aicore__ inline void StoreH(int64_t localChunk)
    {
        const int64_t off = OffsetHChunk(td_.prepare.isVariableLengthPacked != 0, loc_.batch,
                                         GlobalChunk(localChunk), loc_.hv, hv_, chunksPerBatch_);
        AscendC::DataCopy(gmH_[off + static_cast<int64_t>(GetSubBlockIdx()) * (kStateElems / 2)],
                          ubStateB_, static_cast<uint32_t>(kStateElems / 2));
    }

    // Raw rows past the valid window are zeroed first, so every L1 band upload
    // publishes zeros and the cube may keep its full 64-row dimensions. Only
    // the tail is zeroed: the head comes from the MTE2 copies, and the two
    // ranges are disjoint, so the V zero-fill and the MTE2 copy do not race.
    __aicore__ inline void LoadInputs(const ChunkWindow &window, uint32_t firstRow, uint32_t nRows)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kConsEvtUb);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kConsEvtUb);
        if (nRows < static_cast<uint32_t>(kHalfRows)) {
            const int32_t tail = static_cast<int32_t>((kHalfRows - nRows) * kHeadDim);
            AscendC::Duplicate(ubWStage_[static_cast<int32_t>(nRows * kHeadDim)],
                               static_cast<bfloat16_t>(0), tail);
            AscendC::Duplicate(ubQStage_[static_cast<int32_t>(nRows * kHeadDim)],
                               static_cast<bfloat16_t>(0), tail);
            AscendC::Duplicate(ubKStage_[static_cast<int32_t>(nRows * kHeadDim)],
                               static_cast<bfloat16_t>(0), tail);
            AscendC::Duplicate(ubU_[static_cast<int32_t>(nRows * kHeadDim)],
                               static_cast<bfloat16_t>(0), tail);
        }
        if (window.rows < kChunk) {
            AscendC::Duplicate(ubG_[static_cast<int32_t>(window.rows)], 0.0f,
                               static_cast<int32_t>(kChunk - window.rows));
        }

        const int64_t token = window.tokenStart + firstRow;
        const uint32_t elems = nRows * static_cast<uint32_t>(kHeadDim);
        Prep::CopyGmToUbElems(
            ubWStage_, gmW_[OffsetBhtd(loc_.batch, loc_.hv, token, hv_, t_, kHeadDim)], elems);
        Prep::CopyGmToUbElems(
            ubU_, gmU_[OffsetBhtd(loc_.batch, loc_.hv, token, hv_, t_, kHeadDim)], elems);
        Prep::CopyGmToUbElems(
            ubQStage_, gmQ_[OffsetBhtd(loc_.batch, loc_.kh, token, hk_, t_, kHeadDim)], elems);
        Prep::CopyGmToUbElems(
            ubKStage_, gmK_[OffsetBhtd(loc_.batch, loc_.kh, token, hk_, t_, kHeadDim)], elems);
        Prep::CopyGmToUbElems(ubG_, gmG_[OffsetBht(loc_.batch, loc_.hv, window.tokenStart, hv_, t_)],
                              static_cast<uint32_t>(window.rows));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kConsEvtUb);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kConsEvtUb);
    }

    // gateScale[i] = E(g_last - g_i), gateO[i] = E(g_i); alpha = E(g_last) is
    // returned for the state decay. The padded tail keeps the last valid
    // cumsum at rows - 1, so a short chunk still uses the right g_last.
    __aicore__ inline float ComputeGates(const ChunkWindow &window)
    {
        const uint32_t rows = static_cast<uint32_t>(window.rows);
        if (useExp2_) {
            LastGateVF<true>(ubG_, ubScalar_, rows - 1U);
        } else {
            LastGateVF<false>(ubG_, ubScalar_, rows - 1U);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(kConsEvtScalar);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(kConsEvtScalar);
        const float gLast = ubScalar_.GetValue(0);
        const float alpha = ubScalar_.GetValue(1);
        if (useExp2_) {
            GateRowsVF<true>(ubG_, ubGateScale_, ubGateO_, gLast);
        } else {
            GateRowsVF<false>(ubG_, ubGateScale_, ubGateO_, gLast);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kConsEvtUb);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kConsEvtUb);
        return alpha;
    }

    __aicore__ inline void UploadInputs(uint32_t firstRow, int64_t localChunk)
    {
        const int32_t parity = static_cast<int32_t>(localChunk & 1);
        ChunkUploadL1(l1W_, ubWStage_, static_cast<uint32_t>(kHeadDim), firstRow,
                      static_cast<uint32_t>(kHalfRows));
        ChunkUploadL1(l1Q_, ubQStage_, static_cast<uint32_t>(kHeadDim), firstRow,
                      static_cast<uint32_t>(kHalfRows));
        ChunkUploadL1(l1K_, ubKStage_, static_cast<uint32_t>(kHeadDim), firstRow,
                      static_cast<uint32_t>(kHalfRows));
        StateUploadL1(l1H_[parity], ubStateB_, static_cast<uint32_t>(kHeadDim),
                      GetSubBlockIdx() * static_cast<uint32_t>(kHalfDim),
                      static_cast<uint32_t>(kHalfDim));
    }

    // v_new = U - P and right = gateScale * v_new for this AIV's row band.
    __aicore__ inline void StageVNewRight(const ChunkWindow &window, uint32_t firstRow, uint32_t nRows)
    {
        if (nRows < static_cast<uint32_t>(kHalfRows)) {
            AscendC::Duplicate(ubVNew_[static_cast<int32_t>(nRows * kHeadDim)],
                               static_cast<bfloat16_t>(0),
                               static_cast<int32_t>((kHalfRows - nRows) * kHeadDim));
            AscendC::Duplicate(ubRight_[static_cast<int32_t>(nRows * kHeadDim)],
                               static_cast<bfloat16_t>(0),
                               static_cast<int32_t>((kHalfRows - nRows) * kHeadDim));
        }
        VNewRightVF(ubP_, ubU_, ubVNew_, ubRight_, ubGateScale_, firstRow, nRows);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kConsEvtUb);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kConsEvtUb);

        const int64_t token = window.tokenStart + firstRow;
        Prep::CopyUbToGmElems(
            gmVNew_[OffsetBhtd(loc_.batch, loc_.hv, token, hv_, t_, kHeadDim)], ubVNew_,
            nRows * static_cast<uint32_t>(kHeadDim));
        ChunkUploadL1(l1VNew_, ubVNew_, static_cast<uint32_t>(kHeadDim), firstRow,
                      static_cast<uint32_t>(kHalfRows));
        ChunkUploadL1(l1Right_, ubRight_, static_cast<uint32_t>(kHeadDim), firstRow,
                      static_cast<uint32_t>(kHalfRows));
    }

    // gateA and A' for this AIV's row band. A' columns past the diagonal are
    // masked inside the VF; columns past the valid rows are zero because the
    // K band in L1 was published with zero tails.
    __aicore__ inline void StageAPrime(uint32_t firstRow, uint32_t nRows)
    {
        if (nRows < static_cast<uint32_t>(kHalfRows)) {
            AscendC::Duplicate(ubGateA_[static_cast<int32_t>(nRows * kChunk)], 0.0f,
                               static_cast<int32_t>((kHalfRows - nRows) * kChunk));
            AscendC::Duplicate(ubAPrime_[static_cast<int32_t>(nRows * kChunk)],
                               static_cast<bfloat16_t>(0),
                               static_cast<int32_t>((kHalfRows - nRows) * kChunk));
        }
        if (useExp2_) {
            GateMatrixVF<true>(ubGateA_, ubG_, firstRow, nRows);
        } else {
            GateMatrixVF<false>(ubGateA_, ubG_, firstRow, nRows);
        }
        APrimeVF(ubAPrime_, ubARaw_, ubGateA_, firstRow, nRows);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kConsEvtUb);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kConsEvtUb);
        ChunkUploadL1(l1APrime_, ubAPrime_, static_cast<uint32_t>(kChunk), firstRow,
                      static_cast<uint32_t>(kHalfRows));
    }

    // o = (O_s' + O_l) * scale, written straight to the BSND public output.
    __aicore__ inline void StageOutput(const ChunkWindow &window, uint32_t firstRow, uint32_t nRows)
    {
        if (nRows == 0) {
            return;
        }
        OFuseVF(ubU_, ubOS_, ubOL_, scale_, nRows);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kConsEvtUb);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kConsEvtUb);
        const int64_t token = window.tokenStart + firstRow;
        DataCopyExtParams p;
        p.blockCount = static_cast<uint16_t>(nRows);
        p.blockLen = static_cast<uint32_t>(kHeadDim * sizeof(bfloat16_t));
        p.srcStride = 0;
        p.dstStride = static_cast<uint32_t>((hv_ - 1) * kHeadDim * sizeof(bfloat16_t));
        p.rsv = 0;
        AscendC::DataCopyPad(gmO_[OffsetONd(loc_.batch, token, loc_.hv, hv_, t_)], ubU_, p);
    }

    __aicore__ inline void StoreFinalState()
    {
        const int64_t stateElem = StateElem();
        if (stateFp32_) {
            AscendC::DataCopy(gmFinalF_[stateElem], ubStateF_,
                              static_cast<uint32_t>(kStateElems / 2));
        } else {
            AscendC::DataCopy(gmFinalB_[stateElem], ubStateB_,
                              static_cast<uint32_t>(kStateElems / 2));
        }
    }

    // ========================= AIC =========================
    __aicore__ inline void RunChunkAic(int64_t localChunk)
    {
        const int32_t parity = static_cast<int32_t>(localChunk & 1);
        AicWaitRelayMte1(kConsFlagInputs);

        // P = W @ H. H is [K,V] physically, or [V,K] when stateVFirst.
        CubeMatmul(l1W_, l1H_[parity], l0A_[0], l0B_[0], l0C_[0], 64, 128, 128,
                   false, stateVFirst_ == 0);
        CubeFinish(l0C_[0], ubP_, 64, 128);
        AicSetFix(kConsFlagP);

        // O_s = Q @ H. H stays in L0B from the P product; only Q reloads.
        CubeLoadA(l1Q_, l0A_[0], 64, 128, false);
        CubeMmad(l0A_[0], l0B_[0], l0C_[2], 64, 128, 128);
        CubeFinish(l0C_[2], ubOS_, 64, 128);
        AicSetFix(kConsFlagOS);

        // A_raw = Q @ K^T. K is [n=64, k=128], so transposeB stays false.
        CubeMatmul(l1Q_, l1K_, l0A_[1], l0B_[1], l0C_[1], 64, 64, 128, false, false);
        CubeFinish(l0C_[1], ubARaw_, 64, 64);
        AicSetFix(kConsFlagQk);

        AicWaitRelayMte1(kConsFlagVn);

        // D = K^T @ right, or right^T @ K for the [V,K] state layout.
        if (stateVFirst_) {
            CubeMatmul(l1Right_, l1K_, l0A_[0], l0B_[0], l0C_[3], 128, 128, 64, true, true);
        } else {
            CubeMatmul(l1K_, l1Right_, l0A_[0], l0B_[0], l0C_[3], 128, 128, 64, true, true);
        }
        CubeFinish(l0C_[3], ubD_, 128, 128);
        AicSetFix(kConsFlagD);

        AicWaitRelayMte1(kConsFlagAPrime);

        // O_l = A' @ V_new. A' is [64,64], V_new [k=64, n=128].
        CubeMatmul(l1APrime_, l1VNew_, l0A_[0], l0B_[0], l0C_[0], 64, 128, 64, false, true);
        CubeFinish(l0C_[0], ubOL_, 64, 128);
        AicSetFix(kConsFlagO);
    }

    // One full matmul with the single L0C event bank the producer uses: every
    // Mmad consumes one M_MTE1 and one FIX_M credit and every Fixpipe returns
    // them, so the five products per chunk stay balanced with Finish().
    __aicore__ inline void CubeMatmul(LocalTensor<bfloat16_t> l1A, LocalTensor<bfloat16_t> l1B,
                                      LocalTensor<bfloat16_t> l0A, LocalTensor<bfloat16_t> l0B,
                                      LocalTensor<float> l0C, int32_t m, int32_t n, int32_t k,
                                      bool transposeA, bool transposeB)
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kConsEvtCube);
        LoadL1ToL0AB(l1A, l1B, l0A, l0B, m, n, k, transposeA, transposeB);
        CubeMmad(l0A, l0B, l0C, m, n, k);
    }

    __aicore__ inline void CubeLoadA(LocalTensor<bfloat16_t> l1A, LocalTensor<bfloat16_t> l0A,
                                     int32_t m, int32_t k, bool transposeA)
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kConsEvtCube);
        LoadL1ToL0A(l1A, l0A, m, k, transposeA);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kConsEvtCube);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kConsEvtCube);
    }

    __aicore__ inline void CubeMmad(LocalTensor<bfloat16_t> l0A, LocalTensor<bfloat16_t> l0B,
                                    LocalTensor<float> l0C, int32_t m, int32_t n, int32_t k)
    {
        Prep::MmadL0ABToL0C<bfloat16_t>(l0A, l0B, l0C, m, n, k, true, kConsEvtCube);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kConsEvtCube);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kConsEvtCube);
    }

    __aicore__ inline void CubeFinish(LocalTensor<float> l0C, LocalTensor<float> ub,
                                      uint32_t fullM, uint32_t n)
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kConsEvtCube);
        FixpipeHalfFp32Ub(ub, l0C, fullM, n);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kConsEvtCube);
    }

    FusedTilingData td_{};
    ChainLoc loc_{};
    uint32_t chainIdx_{0};
    int64_t hk_{0}, hv_{0}, t_{0};
    int64_t chunksPerBatch_{0};
    int64_t useNorm_{0}, useExp2_{0}, stateVFirst_{0}, stateFp32_{0};
    int64_t storeFinal_{0}, hasInitial_{0};
    float scale_{1.0f};

    AscendC::GlobalTensor<bfloat16_t> gmQ_, gmK_, gmW_, gmU_, gmVNew_, gmO_, gmH_;
    AscendC::GlobalTensor<bfloat16_t> gmInitB_, gmFinalB_;
    AscendC::GlobalTensor<float> gmG_, gmInitF_, gmFinalF_;

    AscendC::LocalTensor<float> ubStateF_, ubD_, ubP_, ubOS_, ubOL_, ubARaw_, ubGateA_;
    AscendC::LocalTensor<float> ubG_, ubGateScale_, ubGateO_, ubScalar_;
    AscendC::LocalTensor<bfloat16_t> ubStateB_, ubU_, ubVNew_, ubRight_, ubAPrime_;
    AscendC::LocalTensor<bfloat16_t> ubWStage_, ubQStage_, ubKStage_, ubInitStage_;

    AscendC::LocalTensor<bfloat16_t> l1W_, l1Q_, l1K_, l1VNew_, l1Right_, l1APrime_;
    AscendC::LocalTensor<bfloat16_t> l1H_[2];

    AscendC::LocalTensor<bfloat16_t> l0A_[2], l0B_[2];
    AscendC::LocalTensor<float> l0C_[4];
};

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_CONSUMER_H
