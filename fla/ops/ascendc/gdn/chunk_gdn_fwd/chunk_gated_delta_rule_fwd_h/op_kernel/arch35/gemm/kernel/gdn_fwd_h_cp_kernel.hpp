/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

/*!
 * \file gdn_fwd_h_cp_kernel.hpp
 * \brief Chunk-parallel (CP) path of chunk_gated_delta_rule_fwd_h (TilingKey 3).
 *
 * One task = (batch, value head) at a time; all ranks split the chunk axis.
 *
 *   phase 1  each rank folds its own chunks into an affine summary (B_bar, A_bar)
 *            by rolling X = [M | S]:
 *                [P | v_work] = W_c @ X
 *                [Rp | Rv]    = [-q*P | q*(U - v_work)]
 *                [T | S_work] = K_c^T @ [Rp | Rv]
 *                X' = D*X + [T | S_work]
 *            (left-projection rewrite of M' = D*M - (q*K)^T @ (W@M));
 *   barrier  AIC-wide rendezvous, every rank publishes [B_bar | A_bar];
 *   phase 2  rank r replays summaries 0..r-1 (ascending left multiplication)
 *                [B_trans | S_local] = B_bar_p @ [B_trans | S_local] + [0 | A_bar_p]
 *            initialised with [I | H0], so the S columns end up as
 *                S_in = B_trans @ H0 + S_local;
 *   phase 3  replay of the local chunks from S_in with the serial rounding points
 *            (h is the low-precision carrier, v_update is low precision for C2).
 *
 * Every GEMM writes its L0C accumulator straight into UB
 * (`Common::Tile::PackedTileCopyTlaToUB` with `CopyL0CToUBMode::NO_SPLIT`): the Fixpipe
 * splits the M rows across the two AIV sub-blocks, so each sub-block finds its own
 * rows at UB offset 0 with the tile row pitch.  The AIV epilogue consumes the result
 * in place and only the operands the Cube still has to read through L1 (Wf/Kf,
 * [Rp|Rv], Y) travel through GM.  This removes the Pv / YT / v_work / h_work /
 * prefix-Y write+read round trips, i.e. most of the non-operand GM traffic.
 *
 * Direct-UB slot ownership (flag 11, AIV -> AIC): the AIV releases the slot with
 * `CrossCoreSetFlag<0x4, PIPE_V>` once the vector reads of the previous tile retired;
 * the mmad waits for it inside its own L0C->UB output loop (`PIPE_FIX`).  The slot is
 * single buffered on purpose: phase 1 and phase 3 are strict chains
 * (X -> GEMM-A -> RpRv -> GEMM-B -> X'), so a second slot has nothing to overlap with,
 * and releasing it once before every GEMM keeps the flag count balanced with the mmad.
 *
 * Row scaling / row subtraction / the phase-2 bias add are register-base loops
 * (`CpDetail`), so a tile costs rows * (V_DIM / FP32_PER_REG) vector issues instead of
 * three tiny calls plus a scalar expf and a GM scalar read per row.
 *
 * All summaries, the prefix and the rolling state are FP32; the FP32 GEMM uses
 * `Common::MmadPingpong<Ascend950, ..., USE_HF32_MODE=false>` (native IEEE FP32).
 *
 * Workspace (host cpScratchStride must match):
 *   summary[rank] : [B_bar(128x128) | A_bar(128x128)] FP32
 *   scratch[rank] : X[M|S] | H0/S_in | YT | Wf | Kf/v_update | Pv | RpRv
 *                   (SCR_YT / SCR_PV are no longer accessed; the slots stay reserved so
 *                    the host layout formula is unchanged.)
 */

#ifndef GDN_FWD_H_CP_KERNEL_HPP
#define GDN_FWD_H_CP_KERNEL_HPP

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla.hpp"
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "kernel_operator.h"
#include "../../epilogue/block/block_epilogue_gdn_fwdh_regbase.hpp"
#include "../../../chunk_gated_delta_rule_fwd_h_struct.h"

namespace GDN {

using _0 = tla::Int<0>;
using _1 = tla::Int<1>;
using _32 = tla::Int<32>;
using _64 = tla::Int<64>;
using _128 = tla::Int<128>;
using _256 = tla::Int<256>;

// Paired AIC <-> AIV flags (mode 0x4 on Ascend950: each AIV sub-block owns id / id+16).
// Phases are separated by a full AIC barrier and each phase drains its own flags, so
// ids may be reused across phases (they are counters, not direction-bound). Phase 3
// reuses 0..3; replay-ready and task-done intentionally share id 4: both are AIV->AIC
// and AIC waits exactly once for each of them per task.
static constexpr uint16_t kCpFlagOperand = 0;     // AIV -> AIC: W/K staged        | phase3 C1 done
static constexpr uint16_t kCpFlagGemmA = 1;       // AIC -> AIV: [P|v_work] in UB  | phase3 v_update ready
static constexpr uint16_t kCpFlagRpRv = 2;        // AIV -> AIC: [Rp|Rv] ready     | phase3 C2 done
static constexpr uint16_t kCpFlagGemmB = 3;       // AIC -> AIV: [T|S_work] in UB  | phase3 h written
static constexpr uint16_t kCpFlagXUpdated = 4;    // AIV -> AIC: X written | replay ready | task done
static constexpr uint16_t kCpFlagYInit = 5;       // AIV -> AIC: prefix state [I|H0] ready
static constexpr uint16_t kCpFlagPrefixGemm = 6;  // AIC -> AIV: one prefix GEMM done
static constexpr uint16_t kCpFlagPrefixAdd = 7;   // AIV -> AIC: bias added, Y written
// Direct L0C -> UB slot. 0..7 are the hand-off protocol above, 8/9/10 are the Catlass
// block barriers and 12/13/14 belong to SyncAll, so 11 is the first free id.
static constexpr uint16_t kCpFlagUbFree = 11;     // AIV -> AIC: direct-UB slot recycled
static constexpr uint16_t kCpFlagReplayReady = 4;
static constexpr uint16_t kCpFlagTaskDone = 4;
static constexpr uint16_t kCpFlagC1Done = 0;
static constexpr uint16_t kCpFlagVUpdate = 1;
static constexpr uint16_t kCpFlagC2Done = 2;
static constexpr uint16_t kCpFlagHDone = 3;
static constexpr uint16_t kCpSubFlagStride = 16;

// The two AIVs of a physical core split every M row range.  The AIC has no sub-block
// view, so the split has to be a compile-time constant matching KERNEL_TYPE_MIX_AIC_1_2.
static constexpr uint32_t kCpVecNum = 2;

// UB layout per AIV sub-block (Catlass Resource::ubBuf hands out 248 KiB).  Regions may
// alias across phases and across a GEMM round trip, but never while both are live:
// the worst case below is 224 KiB (phase 1).
static constexpr uint32_t UB_DIRECT = 0;              // GEMM result tile, pitch 256 (phase 1/2) or 128 (phase 3)
// phase 1
static constexpr uint32_t UB_P1_UIN = 64 * 1024;      // 16 KiB  raw U rows, INPUT_TYPE
static constexpr uint32_t UB_P1_X = 64 * 1024;        // 64 KiB  rolling X tile / [I|0] staging, FP32
static constexpr uint32_t UB_P1_IN_W = 128 * 1024;    // 16 KiB  W input rows
static constexpr uint32_t UB_P1_WF = 144 * 1024;      // 32 KiB  Wf, FP32
static constexpr uint32_t UB_P1_IN_K = 176 * 1024;    // 16 KiB  K input rows
static constexpr uint32_t UB_P1_KF = 192 * 1024;      // 32 KiB  Kf, FP32
// phase 2
static constexpr uint32_t UB_P2_Y = 64 * 1024;        // 64 KiB  [I|H0] / S_in staging, FP32
static constexpr uint32_t UB_P2_H0IN = 128 * 1024;    // 64 KiB  initial_state rows, STATE_TYPE
static constexpr uint32_t UB_P2_A = 128 * 1024;       // 32 KiB  A_bar tile, FP32
static constexpr uint32_t UB_P2_OUT = 192 * 1024;     // 16 KiB  h[chunkBegin] staging, INPUT_TYPE
// phase 3
static constexpr uint32_t UB_P3_UIN = 64 * 1024;      // 16 KiB  raw U rows, INPUT_TYPE
static constexpr uint32_t UB_P3_VU = 144 * 1024;      // 16 KiB  v_update output, INPUT_TYPE
static constexpr uint32_t UB_P3_OUT = 160 * 1024;     // 16 KiB  v_new / h output staging, INPUT_TYPE
static constexpr uint32_t UB_P3_S = 64 * 1024;        // 32 KiB  rolling S, FP32
static constexpr uint32_t UB_P3_ST = 128 * 1024;      // 16 KiB  final_state staging, STATE_TYPE
// shared gate scratch: 64 rows of G_TYPE / FP32 plus q and -q
static constexpr uint32_t UB_G_IN = 240 * 1024;       // 512 B
static constexpr uint32_t UB_G_F = 240 * 1024 + 512;  // 512 B
static constexpr uint32_t UB_Q = 241 * 1024;          // 512 B
static constexpr uint32_t UB_Q_NEG = 241 * 1024 + 512;
// The size labels above assume the op's registered dtypes: INPUT_TYPE is bf16/fp16 (2 B)
// and STATE_TYPE is float/bf16 (<= 4 B). A wider INPUT_TYPE would make a few of the
// per-phase regions overlap, so fail the build rather than corrupt UB silently.
static_assert(sizeof(INPUT_TYPE) <= 2, "CP UB layout assumes a 2-byte input dtype.");
static_assert(sizeof(STATE_TYPE) <= 4, "CP UB layout assumes STATE_TYPE is at most FP32.");
static_assert(sizeof(G_TYPE) <= 4, "CP UB gate scratch assumes G_TYPE is at most FP32.");

// ---------------------------------------------------------------------------
// Register-base helpers for row scaling / row subtraction on a strided tile.
// ApplyRowScaleDualIssue in block_epilogue_gdn_fwdh_regbase.hpp assumes a packed
// [rows, cols] matrix; the CP tile is [rows, 256] holding two logical halves, so the
// same loop is reused with an explicit row pitch for the destination.
namespace CpDetail {

// matrix[row * stride + c] *= scale[row]
static __simd_vf__ inline void ScaleRowsKernel(
    __ubuf__ float *matrix, uint32_t stride, __ubuf__ float *scale,
    uint16_t rows, uint16_t cols)
{
    constexpr uint16_t FP32_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> matrixReg0;
    RegTensor<float> matrixReg1;
    RegTensor<float> scaleReg0;
    RegTensor<float> scaleReg1;
    MaskReg mask0;
    MaskReg mask1;

    uint16_t row = 0;
    for (; row + 1 < rows; row += 2) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg0, scale + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg1, scale + row + 1);
        uint32_t remaining0 = cols;
        uint32_t remaining1 = cols;
        uint16_t colLoops = static_cast<uint16_t>((cols + FP32_PER_REG - 1) / FP32_PER_REG);
        for (uint16_t colLoop = 0; colLoop < colLoops; ++colLoop) {
            uint32_t col = static_cast<uint32_t>(colLoop) * FP32_PER_REG;
            mask0 = UpdateMask<float>(remaining0);
            mask1 = UpdateMask<float>(remaining1);
            LoadAlign(matrixReg0, matrix + static_cast<uint32_t>(row) * stride + col);
            LoadAlign(matrixReg1, matrix + static_cast<uint32_t>(row + 1) * stride + col);
            Mul(matrixReg0, matrixReg0, scaleReg0, mask0);
            Mul(matrixReg1, matrixReg1, scaleReg1, mask1);
            StoreAlign(matrix + static_cast<uint32_t>(row) * stride + col, matrixReg0, mask0);
            StoreAlign(matrix + static_cast<uint32_t>(row + 1) * stride + col, matrixReg1, mask1);
        }
    }
    if (row < rows) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg0, scale + row);
        uint32_t remaining = cols;
        uint16_t colLoops = static_cast<uint16_t>((cols + FP32_PER_REG - 1) / FP32_PER_REG);
        for (uint16_t colLoop = 0; colLoop < colLoops; ++colLoop) {
            uint32_t col = static_cast<uint32_t>(colLoop) * FP32_PER_REG;
            mask0 = UpdateMask<float>(remaining);
            LoadAlign(matrixReg0, matrix + static_cast<uint32_t>(row) * stride + col);
            Mul(matrixReg0, matrixReg0, scaleReg0, mask0);
            StoreAlign(matrix + static_cast<uint32_t>(row) * stride + col, matrixReg0, mask0);
        }
    }
}

// dst[row * stride + c] = scale[row] * (u[row * cols + c] - dst[row * stride + c])
template <typename T>
static __simd_vf__ inline void SubScaleRowsKernel(
    __ubuf__ float *dst, uint32_t stride, __ubuf__ T *u, __ubuf__ float *scale,
    uint16_t rows, uint16_t cols)
{
    constexpr uint16_t FP32_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> dstReg0;
    RegTensor<float> dstReg1;
    RegTensor<float> uReg0;
    RegTensor<float> uReg1;
    RegTensor<float> scaleReg0;
    RegTensor<float> scaleReg1;
    MaskReg mask0;
    MaskReg mask1;

    uint16_t row = 0;
    for (; row + 1 < rows; row += 2) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg0, scale + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg1, scale + row + 1);
        uint32_t remaining0 = cols;
        uint32_t remaining1 = cols;
        uint16_t colLoops = static_cast<uint16_t>((cols + FP32_PER_REG - 1) / FP32_PER_REG);
        for (uint16_t colLoop = 0; colLoop < colLoops; ++colLoop) {
            uint32_t col = static_cast<uint32_t>(colLoop) * FP32_PER_REG;
            mask0 = UpdateMask<float>(remaining0);
            mask1 = UpdateMask<float>(remaining1);
            LoadAlign(dstReg0, dst + static_cast<uint32_t>(row) * stride + col);
            LoadAlign(dstReg1, dst + static_cast<uint32_t>(row + 1) * stride + col);
            Catlass::Epilogue::Block::detail::LoadKdaAsFloat<T>(
                uReg0, u + static_cast<uint32_t>(row) * cols + col, mask0);
            Catlass::Epilogue::Block::detail::LoadKdaAsFloat<T>(
                uReg1, u + static_cast<uint32_t>(row + 1) * cols + col, mask1);
            Sub(dstReg0, uReg0, dstReg0, mask0);
            Sub(dstReg1, uReg1, dstReg1, mask1);
            Mul(dstReg0, dstReg0, scaleReg0, mask0);
            Mul(dstReg1, dstReg1, scaleReg1, mask1);
            StoreAlign(dst + static_cast<uint32_t>(row) * stride + col, dstReg0, mask0);
            StoreAlign(dst + static_cast<uint32_t>(row + 1) * stride + col, dstReg1, mask1);
        }
    }
    if (row < rows) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg0, scale + row);
        uint32_t remaining = cols;
        uint16_t colLoops = static_cast<uint16_t>((cols + FP32_PER_REG - 1) / FP32_PER_REG);
        for (uint16_t colLoop = 0; colLoop < colLoops; ++colLoop) {
            uint32_t col = static_cast<uint32_t>(colLoop) * FP32_PER_REG;
            mask0 = UpdateMask<float>(remaining);
            LoadAlign(dstReg0, dst + static_cast<uint32_t>(row) * stride + col);
            Catlass::Epilogue::Block::detail::LoadKdaAsFloat<T>(
                uReg0, u + static_cast<uint32_t>(row) * cols + col, mask0);
            Sub(dstReg0, uReg0, dstReg0, mask0);
            Mul(dstReg0, dstReg0, scaleReg0, mask0);
            StoreAlign(dst + static_cast<uint32_t>(row) * stride + col, dstReg0, mask0);
        }
    }
}

// dst[row * stride + c] += src[row * cols + c]
static __simd_vf__ inline void AddRowsKernel(
    __ubuf__ float *dst, uint32_t stride, __ubuf__ float *src,
    uint16_t rows, uint16_t cols)
{
    constexpr uint16_t FP32_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> dstReg0;
    RegTensor<float> dstReg1;
    RegTensor<float> srcReg0;
    RegTensor<float> srcReg1;
    MaskReg mask0;
    MaskReg mask1;

    uint16_t row = 0;
    for (; row + 1 < rows; row += 2) {
        uint32_t remaining0 = cols;
        uint32_t remaining1 = cols;
        uint16_t colLoops = static_cast<uint16_t>((cols + FP32_PER_REG - 1) / FP32_PER_REG);
        for (uint16_t colLoop = 0; colLoop < colLoops; ++colLoop) {
            uint32_t col = static_cast<uint32_t>(colLoop) * FP32_PER_REG;
            mask0 = UpdateMask<float>(remaining0);
            mask1 = UpdateMask<float>(remaining1);
            LoadAlign(dstReg0, dst + static_cast<uint32_t>(row) * stride + col);
            LoadAlign(dstReg1, dst + static_cast<uint32_t>(row + 1) * stride + col);
            LoadAlign(srcReg0, src + static_cast<uint32_t>(row) * cols + col);
            LoadAlign(srcReg1, src + static_cast<uint32_t>(row + 1) * cols + col);
            Add(dstReg0, dstReg0, srcReg0, mask0);
            Add(dstReg1, dstReg1, srcReg1, mask1);
            StoreAlign(dst + static_cast<uint32_t>(row) * stride + col, dstReg0, mask0);
            StoreAlign(dst + static_cast<uint32_t>(row + 1) * stride + col, dstReg1, mask1);
        }
    }
    if (row < rows) {
        uint32_t remaining = cols;
        uint16_t colLoops = static_cast<uint16_t>((cols + FP32_PER_REG - 1) / FP32_PER_REG);
        for (uint16_t colLoop = 0; colLoop < colLoops; ++colLoop) {
            uint32_t col = static_cast<uint32_t>(colLoop) * FP32_PER_REG;
            mask0 = UpdateMask<float>(remaining);
            LoadAlign(dstReg0, dst + static_cast<uint32_t>(row) * stride + col);
            LoadAlign(srcReg0, src + static_cast<uint32_t>(row) * cols + col);
            Add(dstReg0, dstReg0, srcReg0, mask0);
            StoreAlign(dst + static_cast<uint32_t>(row) * stride + col, dstReg0, mask0);
        }
    }
}

} // namespace CpDetail

template <typename T>
__aicore__ inline float CpToFloat(T value)
{
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        return AscendC::ToFloat(value);
    } else {
        return static_cast<float>(value);
    }
}

template <typename INPUT_TYPE, typename G_TYPE, typename STATE_TYPE>
class GDNFwdHCPKernel {
public:
    using ArchTag = Catlass::Arch::Ascend950;

    // ---- Direct L0C -> UB GEMM: every result lands in UB, no GM round trip ----
    // The dispatch policy mirrors the reference replay kernel (no unit flag, HF32 off), so
    // the Fixpipe handshake is the plain FIX_M / cross-core pair.
    using CpDirectPolicy = Common::MmadPingpong<ArchTag, false, false, 1>;

    // [128, 256, 128] tiles for the phase-1 / phase-2 FP32 GEMM.
    // `NO_SPLIT` is the mode that carries the (dst, src, ubRowNum, beginSubBlockIdx,
    // sendVecNum) output overload the block's `sendVecNum > 1` path calls; SPLIT_M only
    // ships the 3-argument form. The row split across the two AIVs comes from the explicit
    // `subBlockId` each Fixpipe is given, which is what the reference replay kernel uses too.
    using CpF32L1 = tla::Shape<_128, _256, _128>;
    using CpF32L0 = tla::Shape<_128, _256, _32>;
    using CpF32TileCopy = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, float, Catlass::layout::RowMajor, float, Catlass::layout::RowMajor,
        float, Catlass::layout::RowMajor, void,
        Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
    using CpF32Mmad = Common::BlockMmadTla<
        CpDirectPolicy, CpF32L1, CpF32L0, float, float, float, void, CpF32TileCopy>;
    // The K^T operand is a ColumnMajor GM view, and Ascend950 only has ColumnMajor->nZ for
    // the GM->L1 copy, so that GEMM needs its own tile copy (RowMajor tag -> zN would have
    // no specialization). Same tile shapes, only the GM A tag differs.
    using CpF32KvTileCopy = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, float, Catlass::layout::ColumnMajor, float, Catlass::layout::RowMajor,
        float, Catlass::layout::RowMajor, void,
        Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
    using CpF32KvMmad = Common::BlockMmadTla<
        CpDirectPolicy, CpF32L1, CpF32L0, float, float, float, void, CpF32KvTileCopy>;

    // [128, 128, 128] tiles for the phase-3 replay (input-dtype operands, FP32 accumulator).
    using CpInL1 = tla::Shape<_128, _128, _128>;
    using CpInL0 = tla::Shape<_128, _128, _64>;
    using CpInTileCopy = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, INPUT_TYPE, Catlass::layout::RowMajor, INPUT_TYPE, Catlass::layout::RowMajor,
        float, Catlass::layout::RowMajor, void,
        Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
    using CpInMmad = Common::BlockMmadTla<
        CpDirectPolicy, CpInL1, CpInL0, INPUT_TYPE, INPUT_TYPE, float, void, CpInTileCopy>;
    using CpInKvTileCopy = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, INPUT_TYPE, Catlass::layout::ColumnMajor, INPUT_TYPE, Catlass::layout::RowMajor,
        float, Catlass::layout::RowMajor, void,
        Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
    using CpInKvMmad = Common::BlockMmadTla<
        CpDirectPolicy, CpInL1, CpInL0, INPUT_TYPE, INPUT_TYPE, float, void, CpInKvTileCopy>;

    static constexpr uint32_t K_DIM = 128;   // CP path requires K = V = 128
    static constexpr uint32_t V_DIM = 128;
    static constexpr uint32_t ROW_DIM = 2 * V_DIM;
    static_assert(K_DIM == 128 && V_DIM == 128,
                  "CP path layout constants assume K = V = 128; update the route first.");

    static constexpr uint32_t SCR_X = 0;                             // [128, 256] M|S
    static constexpr uint32_t SCR_H0 = SCR_X + K_DIM * ROW_DIM;      // [128, 128] H0 / S_in
    static constexpr uint32_t SCR_YT = SCR_H0 + K_DIM * V_DIM;       // reserved (unused)
    static constexpr uint32_t SCR_BASE = SCR_YT + K_DIM * ROW_DIM;   // per-chunk operands
    static constexpr uint32_t SCR_WF = SCR_BASE;                     // [BT, 128] FP32
    static constexpr uint32_t SCR_KF = SCR_WF + K_DIM * K_DIM;       // [BT, 128] FP32 | v_update
    static constexpr uint32_t SCR_PV = SCR_KF + K_DIM * K_DIM;       // reserved (unused)
    static constexpr uint32_t SCR_RPRV = SCR_PV + K_DIM * ROW_DIM;   // [BT, 256] FP32

    uint32_t batch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t chunkSize;
    bool useInitialState;
    bool storeFinalState;
    uint32_t totalChunks;
    uint32_t rankCount;
    uint32_t rank;
    uint32_t chunkBegin;
    uint32_t chunkEnd;
    uint32_t headGroups;
    uint32_t taskCount;
    int64_t cpSummaryOffset;
    int64_t cpScratchOffset;
    int64_t cpSummaryStride;
    int64_t cpScratchStride;

    AscendC::GlobalTensor<INPUT_TYPE> gmK;
    AscendC::GlobalTensor<INPUT_TYPE> gmW;
    AscendC::GlobalTensor<INPUT_TYPE> gmU;
    AscendC::GlobalTensor<G_TYPE> gmG;
    AscendC::GlobalTensor<STATE_TYPE> gmInitialState;
    AscendC::GlobalTensor<INPUT_TYPE> gmH;
    AscendC::GlobalTensor<INPUT_TYPE> gmVNew;
    AscendC::GlobalTensor<STATE_TYPE> gmFinalState;
    AscendC::GlobalTensor<float> gmSummary;   // [rank][B_bar | A_bar]
    AscendC::GlobalTensor<float> gmScratch;   // [rank][X | H0 | YT | Wf | Kf | Pv | RpRv]

    Catlass::Arch::Resource<ArchTag> resource;

    __aicore__ inline GDNFwdHCPKernel() {}

    __aicore__ inline void Init(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR inital_state,
                                GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state, GM_ADDR tiling,
                                GM_ADDR user)
    {
        __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict t =
            reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);
        batch = static_cast<uint32_t>(t->shapeBatch);
        seqlen = static_cast<uint32_t>(t->seqlen);
        kNumHead = static_cast<uint32_t>(t->kNumHead);
        vNumHead = static_cast<uint32_t>(t->vNumHead);
        chunkSize = static_cast<uint32_t>(t->chunkSize);
        useInitialState = t->useInitialState;
        storeFinalState = t->storeFinalState;
        totalChunks = (seqlen + chunkSize - 1) / chunkSize;
        rankCount = static_cast<uint32_t>(t->cpRankCount);
        cpSummaryOffset = t->cpSummaryOffset;
        cpScratchOffset = t->cpScratchOffset;
        cpSummaryStride = t->cpSummaryStride;
        cpScratchStride = t->cpScratchStride;
        headGroups = vNumHead / kNumHead;
        taskCount = batch * vNumHead;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        if ASCEND_IS_AIV {
            blockIdx /= AscendC::GetSubBlockNum();
        }
        rank = blockIdx;
        const uint32_t base = totalChunks / rankCount;
        const uint32_t rem = totalChunks % rankCount;
        chunkBegin = rank * base + Catlass::Min(rank, rem);
        chunkEnd = chunkBegin + base + (rank < rem ? 1U : 0U);

        gmK.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_TYPE *>(k));
        gmW.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_TYPE *>(w));
        gmU.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_TYPE *>(u));
        gmG.SetGlobalBuffer(reinterpret_cast<__gm__ G_TYPE *>(g));
        gmInitialState.SetGlobalBuffer(reinterpret_cast<__gm__ STATE_TYPE *>(inital_state));
        gmH.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_TYPE *>(h));
        gmVNew.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_TYPE *>(v_new));
        gmFinalState.SetGlobalBuffer(reinterpret_cast<__gm__ STATE_TYPE *>(final_state));
        gmSummary.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + cpSummaryOffset));
        gmScratch.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + cpScratchOffset));
    }

    // ------------------------------------------------------------------ helpers
    __aicore__ inline uint64_t ScratchBase() const
    {
        return static_cast<uint64_t>(rank) * static_cast<uint64_t>(cpScratchStride);
    }

    __aicore__ inline uint64_t SummaryBase(uint32_t r) const
    {
        return static_cast<uint64_t>(r) * static_cast<uint64_t>(cpSummaryStride);
    }

    __aicore__ inline uint64_t KOffset(uint32_t b, uint32_t hk, uint64_t token) const
    {
        return ((static_cast<uint64_t>(b) * kNumHead + hk) * seqlen + token) * K_DIM;
    }

    __aicore__ inline uint64_t HvOffset(uint32_t b, uint32_t hv, uint64_t token, uint32_t dim) const
    {
        return ((static_cast<uint64_t>(b) * vNumHead + hv) * seqlen + token) * dim;
    }

    __aicore__ inline uint64_t HOffset(uint32_t b, uint32_t hv, uint32_t chunk) const
    {
        return ((static_cast<uint64_t>(b) * totalChunks + chunk) * vNumHead + hv) * K_DIM * V_DIM;
    }

    __aicore__ inline uint64_t StateOffset(uint32_t b, uint32_t hv) const
    {
        return (static_cast<uint64_t>(b) * vNumHead + hv) * K_DIM * V_DIM;
    }

    __aicore__ inline uint32_t ChunkTokens(uint32_t chunk) const
    {
        return Catlass::Min(chunkSize, seqlen - chunk * chunkSize);
    }

    __aicore__ inline float LoadG(uint32_t b, uint32_t hv, uint64_t token) const
    {
        return CpToFloat<G_TYPE>(gmG.GetValue(HvOffset(b, hv, token, 1)));
    }

    // The Catlass::Arch cross-core wrappers only accept a CrossCoreFlag object, so the
    // id-based forms below call the AscendC intrinsics directly.
    __aicore__ inline void AivSetToAic(uint16_t id)
    {
        AscendC::PipeBarrier<PIPE_MTE3>();
        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(id);
    }

    __aicore__ inline void AivSetUbFree()
    {
        // Ordered on PIPE_V: the slot may only be recycled once the vector reads of the
        // previous tile retired.
        AscendC::CrossCoreSetFlag<0x4, PIPE_V>(kCpFlagUbFree);
    }

    __aicore__ inline void AivWaitFromAic(uint16_t id)
    {
        AscendC::CrossCoreWaitFlag<0x4, PIPE_S>(id);
    }

    __aicore__ inline void AicWaitFromAiv(uint16_t id)
    {
        AscendC::CrossCoreWaitFlag<0x4, PIPE_S>(id);
        AscendC::CrossCoreWaitFlag<0x4, PIPE_S>(static_cast<uint16_t>(id + kCpSubFlagStride));
    }

    __aicore__ inline void AivSubBlockBarrier()
    {
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
    }

    // Rows each AIV sub-block owns when the Fixpipe splits an M-row tile.
    __aicore__ inline uint32_t RowsPerSubBlock(uint32_t rows) const
    {
        return Catlass::CeilDiv(rows, kCpVecNum);
    }

    // One mmad call that hands the L0C accumulator to the two AIV sub-blocks through the
    // shared UB slot. `rowsPerSubBlock` must match RowsPerSubBlock of the tile's M size.
    template <typename MmadT, typename TensorA, typename TensorB, typename UbTensor>
    __aicore__ inline void DirectMmad(MmadT &mmad, TensorA &blockA, TensorB &blockB,
                                      UbTensor *ubList, const Catlass::GemmCoord &shape,
                                      uint32_t rowsPerSubBlock, uint16_t readyFlag)
    {
        uint32_t ubListId = 0;
        mmad(blockA, blockB, ubList, shape, rowsPerSubBlock, 0,
             static_cast<uint64_t>(kCpFlagUbFree), static_cast<uint64_t>(readyFlag), ubListId,
             static_cast<uint8_t>(kCpVecNum), 1);
    }

    // Fills the single-slot UB destination list for one GEMM result tile.
    template <typename UbTensor>
    __aicore__ inline void FillUbList(UbTensor &tensor, UbTensor *list)
    {
        for (uint32_t i = 0; i < CpF32Mmad::MAX_CUBE_VEC_SYNC_NUM; ++i) {
            list[i] = tensor;
        }
    }

    // q[i] = exp(g_last - g[i]) for i in [rowBegin, rowBegin + rows), plus -q.
    // The gate rows are loaded once per chunk instead of one GM scalar read per row, and
    // the exponential runs as one vector op like in the serial replay kernel.
    __aicore__ inline void PrepareQ(uint32_t b, uint32_t hv, uint64_t tokenBase, uint32_t rowBegin,
                                    uint32_t rows, float gLast,
                                    const AscendC::LocalTensor<float> &ubQ,
                                    const AscendC::LocalTensor<float> &ubQNeg)
    {
        auto ubGIn = resource.ubBuf.template GetBufferByByte<G_TYPE>(UB_G_IN);
        auto ubGf = resource.ubBuf.template GetBufferByByte<float>(UB_G_F);
        AscendC::DataCopyExtParams gParams{
            1, static_cast<uint32_t>(rows * sizeof(G_TYPE)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<G_TYPE> gPad{false, 0, 0, 0};
        AscendC::DataCopyPad(ubGIn, gmG[HvOffset(b, hv, tokenBase + rowBegin, 1)], gParams, gPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        AscendC::LocalTensor<float> gFloat = ubGf;
        if constexpr (!std::is_same<G_TYPE, float>::value) {
            AscendC::Cast(gFloat, ubGIn, AscendC::RoundMode::CAST_NONE, rows);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            gFloat = ubGIn.template ReinterpretCast<float>();
        }
        AscendC::Duplicate(ubQ, gLast, rows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(ubQ, ubQ, gFloat, rows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(ubQ, ubQ, rows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(ubQNeg, ubQ, -1.0f, rows);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ---------------------------------------------------------------------------
    // Bulk copy helpers: one DataCopyPad for a contiguous half of the K rows when the
    // destination has the 2*V_DIM float row pitch of X (M columns at +0, S at +V_DIM).
    __aicore__ inline void CopyPadToXHalf(uint32_t rowBegin, uint32_t rows, uint32_t colOffset,
                                          const AscendC::LocalTensor<float> &src)
    {
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(V_DIM * sizeof(float)),
            0,
            static_cast<uint32_t>(V_DIM * sizeof(float)),
            0};
        AscendC::DataCopyPad(gmScratch[ScratchBase() + SCR_X + rowBegin * ROW_DIM + colOffset],
                             src, params);
    }

    // ---------------------------------------------------------------------------
    // Writes X = [I | 0] (phase 1). Sub-block 0 owns the M columns, sub-block 1 the S
    // columns; each covers all K rows with a single DataCopyPad.
    __aicore__ inline void InitRollingState()
    {
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        auto ubMat = resource.ubBuf.template GetBufferByByte<float>(UB_P1_X);
        const uint32_t rows = K_DIM;
        AscendC::Duplicate(ubMat, 0.0f, rows * V_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        if (subBlock == 0) {
            for (uint32_t i = 0; i < rows; ++i) {
                ubMat.SetValue(static_cast<int64_t>(i) * V_DIM + i, 1.0f);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        CopyPadToXHalf(0, rows, (subBlock == 0) ? 0 : V_DIM, ubMat);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    // ------------------------------------------------------------------ process
    __aicore__ inline void Process()
    {
        AscendC::SyncAll<false>();
        if ASCEND_IS_AIC {
            ProcessCube();
        }
        if ASCEND_IS_AIV {
            ProcessVector();
        }
    }

    // ================================================================== AIC
    __aicore__ inline void ProcessCube()
    {
        CpF32Mmad mmadF32(resource, 0);
        CpF32KvMmad mmadF32Kv(resource, 0);
        CpInMmad mmadIn(resource, 0);
        CpInKvMmad mmadInKv(resource, 0);
        for (uint32_t task = 0; task < taskCount; ++task) {
            const uint32_t b = task / vNumHead;
            const uint32_t hv = task % vNumHead;
            const uint32_t hk = hv / headGroups;
            Phase1Cube(mmadF32, mmadF32Kv);
            AscendC::PipeBarrier<PIPE_ALL>();
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_FIX>();
            Phase2Cube(mmadF32);
            Phase3Cube(mmadIn, mmadInKv, b, hv, hk);
            AicWaitFromAiv(kCpFlagTaskDone);
        }
    }

    __aicore__ inline void Phase1Cube(CpF32Mmad &mmad, CpF32KvMmad &mmadKv)
    {
        const uint64_t scr = ScratchBase();
        auto wfLayout = tla::MakeLayout<float, Catlass::layout::RowMajor>(chunkSize, K_DIM);
        auto xLayout = tla::MakeLayout<float, Catlass::layout::RowMajor>(K_DIM, ROW_DIM);
        auto kfLayout = tla::MakeLayout<float, Catlass::layout::ColumnMajor>(K_DIM, chunkSize);
        auto rpLayout = tla::MakeLayout<float, Catlass::layout::RowMajor>(chunkSize, ROW_DIM);
        auto ubDirect = resource.ubBuf.template GetBufferByByte<float>(UB_DIRECT);
        for (uint32_t chunk = chunkBegin; chunk < chunkEnd; ++chunk) {
            const uint32_t tokens = ChunkTokens(chunk);
            AicWaitFromAiv(kCpFlagOperand);

            // [P | v_work] = Wf @ X, straight into the UB slot.
            auto aLayout = tla::MakeLayout<float, Catlass::layout::RowMajor>(tokens, ROW_DIM);
            auto tensorUbA = tla::MakeTensor(ubDirect, aLayout, Catlass::Arch::PositionUB{});
            using UbTensorA = decltype(tensorUbA);
            UbTensorA ubListA[CpF32Mmad::MAX_CUBE_VEC_SYNC_NUM];
            FillUbList(tensorUbA, ubListA);

            auto tensorWf = tla::MakeTensor(gmScratch[scr + SCR_WF], wfLayout, Catlass::Arch::PositionGM{});
            auto tensorX = tla::MakeTensor(gmScratch[scr + SCR_X], xLayout, Catlass::Arch::PositionGM{});
            Catlass::GemmCoord shapeA{tokens, ROW_DIM, K_DIM};
            auto blockWf = tla::GetTile(tensorWf, tla::MakeCoord(0, 0), tla::MakeShape(shapeA.m(), shapeA.k()));
            auto blockX = tla::GetTile(tensorX, tla::MakeCoord(0, 0), tla::MakeShape(shapeA.k(), shapeA.n()));
            DirectMmad(mmad, blockWf, blockX, ubListA, shapeA, RowsPerSubBlock(tokens), kCpFlagGemmA);

            AicWaitFromAiv(kCpFlagRpRv);

            // [T | S_work] = Kf^T @ [Rp|Rv], straight into the same UB slot.
            auto bOutLayout = tla::MakeLayout<float, Catlass::layout::RowMajor>(K_DIM, ROW_DIM);
            auto tensorUbB = tla::MakeTensor(ubDirect, bOutLayout, Catlass::Arch::PositionUB{});
            using UbTensorB = decltype(tensorUbB);
            UbTensorB ubListB[CpF32Mmad::MAX_CUBE_VEC_SYNC_NUM];
            FillUbList(tensorUbB, ubListB);

            auto tensorKf = tla::MakeTensor(gmScratch[scr + SCR_KF], kfLayout, Catlass::Arch::PositionGM{});
            auto tensorRpRv = tla::MakeTensor(gmScratch[scr + SCR_RPRV], rpLayout, Catlass::Arch::PositionGM{});
            Catlass::GemmCoord shapeB{K_DIM, ROW_DIM, tokens};
            auto blockKf = tla::GetTile(tensorKf, tla::MakeCoord(0, 0), tla::MakeShape(shapeB.m(), shapeB.k()));
            auto blockRpRv = tla::GetTile(tensorRpRv, tla::MakeCoord(0, 0), tla::MakeShape(shapeB.k(), shapeB.n()));
            DirectMmad(mmadKv, blockKf, blockRpRv, ubListB, shapeB, RowsPerSubBlock(K_DIM), kCpFlagGemmB);

            AicWaitFromAiv(kCpFlagXUpdated);
        }
    }

    __aicore__ inline void Phase2Cube(CpF32Mmad &mmad)
    {
        AicWaitFromAiv(kCpFlagYInit);
        const uint64_t scr = ScratchBase();
        auto bLayout = tla::MakeLayout<float, Catlass::layout::RowMajor>(K_DIM, K_DIM);
        auto yLayout = tla::MakeLayout<float, Catlass::layout::RowMajor>(K_DIM, ROW_DIM);
        auto ubDirect = resource.ubBuf.template GetBufferByByte<float>(UB_DIRECT);
        auto outLayout = tla::MakeLayout<float, Catlass::layout::RowMajor>(K_DIM, ROW_DIM);
        auto tensorUb = tla::MakeTensor(ubDirect, outLayout, Catlass::Arch::PositionUB{});
        using UbTensor = decltype(tensorUb);
        UbTensor ubList[CpF32Mmad::MAX_CUBE_VEC_SYNC_NUM];
        FillUbList(tensorUb, ubList);
        for (uint32_t prior = 0; prior < rank; ++prior) {
            auto tensorB = tla::MakeTensor(gmSummary[SummaryBase(prior)], bLayout, Catlass::Arch::PositionGM{});
            auto tensorY = tla::MakeTensor(gmScratch[scr + SCR_X], yLayout, Catlass::Arch::PositionGM{});
            Catlass::GemmCoord shape{K_DIM, ROW_DIM, K_DIM};
            auto blockB = tla::GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
            auto blockY = tla::GetTile(tensorY, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
            DirectMmad(mmad, blockB, blockY, ubList, shape, RowsPerSubBlock(K_DIM), kCpFlagPrefixGemm);
            AicWaitFromAiv(kCpFlagPrefixAdd);
        }
    }

    __aicore__ inline void Phase3Cube(CpInMmad &mmad, CpInKvMmad &mmadKv, uint32_t b, uint32_t hv,
                                      uint32_t hk)
    {
        AicWaitFromAiv(kCpFlagReplayReady);
        const uint64_t scr = ScratchBase();
        auto wLayout = tla::MakeLayout<INPUT_TYPE, Catlass::layout::RowMajor>(
            static_cast<uint64_t>(batch) * vNumHead * seqlen, K_DIM);
        auto hLayout = tla::MakeLayout<INPUT_TYPE, Catlass::layout::RowMajor>(
            static_cast<uint64_t>(batch) * totalChunks * vNumHead * K_DIM, V_DIM);
        auto kLayout = tla::MakeLayout<INPUT_TYPE, Catlass::layout::ColumnMajor>(
            K_DIM, static_cast<uint64_t>(batch) * kNumHead * seqlen);
        auto vuLayout = tla::MakeLayout<INPUT_TYPE, Catlass::layout::RowMajor>(chunkSize, V_DIM);
        auto ubDirect = resource.ubBuf.template GetBufferByByte<float>(UB_DIRECT);
        for (uint32_t chunk = chunkBegin; chunk < chunkEnd; ++chunk) {
            const uint32_t tokens = ChunkTokens(chunk);
            const uint64_t tokenBase = static_cast<uint64_t>(chunk) * chunkSize;

            // C1: v_work = W_c @ h[c], straight into UB.
            auto c1Layout = tla::MakeLayout<float, Catlass::layout::RowMajor>(tokens, V_DIM);
            auto tensorUbC1 = tla::MakeTensor(ubDirect, c1Layout, Catlass::Arch::PositionUB{});
            using UbTensorC1 = decltype(tensorUbC1);
            UbTensorC1 ubListC1[CpInMmad::MAX_CUBE_VEC_SYNC_NUM];
            FillUbList(tensorUbC1, ubListC1);

            auto tensorW = tla::MakeTensor(gmW[HvOffset(b, hv, tokenBase, K_DIM)], wLayout,
                                           Catlass::Arch::PositionGM{});
            auto tensorH = tla::MakeTensor(gmH[HOffset(b, hv, chunk)], hLayout,
                                           Catlass::Arch::PositionGM{});
            Catlass::GemmCoord shapeC1{tokens, V_DIM, K_DIM};
            auto blockW = tla::GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(shapeC1.m(), shapeC1.k()));
            auto blockH = tla::GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(shapeC1.k(), shapeC1.n()));
            DirectMmad(mmad, blockW, blockH, ubListC1, shapeC1, RowsPerSubBlock(tokens), kCpFlagC1Done);

            AicWaitFromAiv(kCpFlagVUpdate);

            const bool isLastChunk = (chunk + 1 == totalChunks);
            const bool needStage2 = storeFinalState || !isLastChunk;
            if (needStage2) {
                // C2: h_work = K_c^T @ v_update, straight into UB.
                auto c2Layout = tla::MakeLayout<float, Catlass::layout::RowMajor>(K_DIM, V_DIM);
                auto tensorUbC2 = tla::MakeTensor(ubDirect, c2Layout, Catlass::Arch::PositionUB{});
                using UbTensorC2 = decltype(tensorUbC2);
                UbTensorC2 ubListC2[CpInMmad::MAX_CUBE_VEC_SYNC_NUM];
                FillUbList(tensorUbC2, ubListC2);

                auto gmVu = gmScratch[scr + SCR_KF].template ReinterpretCast<INPUT_TYPE>();
                auto tensorK = tla::MakeTensor(gmK[KOffset(b, hk, tokenBase)], kLayout,
                                               Catlass::Arch::PositionGM{});
                auto tensorVu = tla::MakeTensor(gmVu, vuLayout, Catlass::Arch::PositionGM{});
                Catlass::GemmCoord shapeC2{K_DIM, V_DIM, tokens};
                auto blockK = tla::GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(shapeC2.m(), shapeC2.k()));
                auto blockVu = tla::GetTile(tensorVu, tla::MakeCoord(0, 0), tla::MakeShape(shapeC2.k(), shapeC2.n()));
                DirectMmad(mmadKv, blockK, blockVu, ubListC2, shapeC2, RowsPerSubBlock(K_DIM), kCpFlagC2Done);
            }
            AicWaitFromAiv(kCpFlagHDone);
        }
    }

    // ================================================================== AIV
    __aicore__ inline void ProcessVector()
    {
        for (uint32_t task = 0; task < taskCount; ++task) {
            const uint32_t b = task / vNumHead;
            const uint32_t hv = task % vNumHead;
            Phase1Vector(b, hv);
            Phase2Vector(b, hv);
            Phase3Vector(b, hv);
            AivSetToAic(kCpFlagTaskDone);
            // Task boundary rendezvous across all AIV sub-blocks: a fast rank must not
            // start the next task (and overwrite its summary / scratch) while a slow rank
            // is still reading them. ID 8 is Catlass AIV_INTER_BLOCK_BARRIER.
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        }
    }

    __aicore__ inline void Phase1Vector(uint32_t b, uint32_t hv)
    {
        InitRollingState();
        for (uint32_t chunk = chunkBegin; chunk < chunkEnd; ++chunk) {
            const uint32_t tokens = ChunkTokens(chunk);
            const uint32_t rowsPerSubBlock = RowsPerSubBlock(tokens);
            const uint32_t rowBegin = AscendC::GetSubBlockIdx() * rowsPerSubBlock;
            const uint32_t rowEnd = Catlass::Min(rowBegin + rowsPerSubBlock, tokens);
            PrepareOperands(b, hv, chunk, rowBegin, rowEnd);
            AivSetToAic(kCpFlagOperand);
            AivSetUbFree();   // slot free for GEMM-A

            AivWaitFromAic(kCpFlagGemmA);
            ComputeRpRv(b, hv, chunk, rowBegin, rowEnd);
            AivSetToAic(kCpFlagRpRv);
            AivSetUbFree();   // slot free for GEMM-B

            AivWaitFromAic(kCpFlagGemmB);
            UpdateRollingState(b, hv, chunk, chunk + 1 == chunkEnd);
            AivSetToAic(kCpFlagXUpdated);
        }
    }

    __aicore__ inline void PrepareOperands(uint32_t b, uint32_t hv, uint32_t chunk,
                                           uint32_t rowBegin, uint32_t rowEnd)
    {
        const uint64_t scr = ScratchBase();
        const uint64_t tokenBase = static_cast<uint64_t>(chunk) * chunkSize;
        const uint32_t rows = rowEnd - rowBegin;
        if (rows == 0) {
            return;
        }
        auto ubInW = resource.ubBuf.template GetBufferByByte<INPUT_TYPE>(UB_P1_IN_W);
        auto ubWf = resource.ubBuf.template GetBufferByByte<float>(UB_P1_WF);
        auto ubInK = resource.ubBuf.template GetBufferByByte<INPUT_TYPE>(UB_P1_IN_K);
        auto ubKf = resource.ubBuf.template GetBufferByByte<float>(UB_P1_KF);

        AscendC::DataCopy(ubInW, gmW[HvOffset(b, hv, tokenBase + rowBegin, K_DIM)], rows * K_DIM);
        AscendC::DataCopy(ubInK, gmK[KOffset(b, hv / headGroups, tokenBase + rowBegin)], rows * K_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(ubWf, ubInW, AscendC::RoundMode::CAST_NONE, rows * K_DIM);
        AscendC::Cast(ubKf, ubInK, AscendC::RoundMode::CAST_NONE, rows * K_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(gmScratch[scr + SCR_WF + rowBegin * K_DIM], ubWf, rows * K_DIM);
        AscendC::DataCopy(gmScratch[scr + SCR_KF + rowBegin * K_DIM], ubKf, rows * K_DIM);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    // [Rp | Rv] = [-q*P | q*(U - v_work)]. The GEMM-A row is already in the UB slot and
    // holds this sub-block's rows [0, rows); P is the first V_DIM columns, v_work the
    // second. Both halves are scaled by a register-base loop, so the whole tile costs
    // rows * (V_DIM / FP32_PER_REG) vector issues instead of three tiny calls per row.
    __aicore__ inline void ComputeRpRv(uint32_t b, uint32_t hv, uint32_t chunk,
                                       uint32_t rowBegin, uint32_t rowEnd)
    {
        const uint32_t rows = rowEnd - rowBegin;
        if (rows == 0) {
            return;
        }
        const uint64_t scr = ScratchBase();
        const uint64_t tokenBase = static_cast<uint64_t>(chunk) * chunkSize;
        const uint32_t tokens = ChunkTokens(chunk);

        auto ubPv = resource.ubBuf.template GetBufferByByte<float>(UB_DIRECT);
        auto ubUIn = resource.ubBuf.template GetBufferByByte<INPUT_TYPE>(UB_P1_UIN);
        auto ubQ = resource.ubBuf.template GetBufferByByte<float>(UB_Q);
        auto ubQNeg = resource.ubBuf.template GetBufferByByte<float>(UB_Q_NEG);

        PrepareQ(b, hv, tokenBase, rowBegin, rows, LoadG(b, hv, tokenBase + tokens - 1), ubQ, ubQNeg);

        AscendC::DataCopy(ubUIn, gmU[HvOffset(b, hv, tokenBase + rowBegin, V_DIM)], rows * V_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        __ubuf__ float *pvAddr = reinterpret_cast<__ubuf__ float *>(ubPv.GetPhyAddr());
        __ubuf__ INPUT_TYPE *uAddr = reinterpret_cast<__ubuf__ INPUT_TYPE *>(ubUIn.GetPhyAddr());
        __ubuf__ float *qAddr = reinterpret_cast<__ubuf__ float *>(ubQ.GetPhyAddr());
        __ubuf__ float *qNegAddr = reinterpret_cast<__ubuf__ float *>(ubQNeg.GetPhyAddr());
        // Rv = q * (U - v_work) on the S half, Rp = -q * P on the M half.
        AscendC::VF_CALL<CpDetail::SubScaleRowsKernel<INPUT_TYPE>>(
            pvAddr + V_DIM, ROW_DIM, uAddr, qAddr, static_cast<uint16_t>(rows),
            static_cast<uint16_t>(V_DIM));
        AscendC::VF_CALL<CpDetail::ScaleRowsKernel>(
            pvAddr, ROW_DIM, qNegAddr, static_cast<uint16_t>(rows), static_cast<uint16_t>(V_DIM));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(gmScratch[scr + SCR_RPRV + rowBegin * ROW_DIM], ubPv, rows * ROW_DIM);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    // X' = D*X + [T | S_work]; on the last local chunk the result is also published as
    // the rank summary ([B_bar | A_bar]). Half of the K rows is owned by each sub-block.
    __aicore__ inline void UpdateRollingState(uint32_t b, uint32_t hv, uint32_t chunk, bool publish)
    {
        const uint64_t scr = ScratchBase();
        const uint32_t tokens = ChunkTokens(chunk);
        const uint64_t tokenBase = static_cast<uint64_t>(chunk) * chunkSize;
        const float decay = expf(LoadG(b, hv, tokenBase + tokens - 1));
        const uint32_t begin = AscendC::GetSubBlockIdx() * (K_DIM / 2);
        const uint32_t rows = K_DIM / 2;
        auto ubX = resource.ubBuf.template GetBufferByByte<float>(UB_P1_X);
        auto ubT = resource.ubBuf.template GetBufferByByte<float>(UB_DIRECT);

        AscendC::DataCopy(ubX, gmScratch[scr + SCR_X + begin * ROW_DIM], rows * ROW_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Muls(ubX, ubX, decay, rows * ROW_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float>(ubX, ubX, ubT, rows * ROW_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(gmScratch[scr + SCR_X + begin * ROW_DIM], ubX, rows * ROW_DIM);
        if (publish) {
            // Summary layout is two contiguous 128x128 blocks: B_bar at [0, 16384)
            // and A_bar at [16384, 32768), both row-major with a 128-float pitch.
            // ubX interleaves [M | S] per row, so copy the halves separately.
            // DataCopyPad UB->GM: srcStride is in 32-byte dataBlocks (the gap between
            // the end of one block and the start of the next), dstStride is in bytes.
            AscendC::DataCopyExtParams halfParams{
                static_cast<uint16_t>(rows),
                static_cast<uint32_t>(V_DIM * sizeof(float)),
                static_cast<int64_t>((ROW_DIM - V_DIM) * sizeof(float) / 32),
                0, 0};
            AscendC::DataCopyPad(gmSummary[SummaryBase(rank) + begin * V_DIM], ubX, halfParams);
            AscendC::DataCopyPad(
                gmSummary[SummaryBase(rank) + K_DIM * K_DIM + begin * V_DIM],
                ubX[V_DIM], halfParams);
        }
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void Phase2Vector(uint32_t b, uint32_t hv)
    {
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        const uint64_t scr = ScratchBase();
        const uint32_t begin = subBlock * (K_DIM / 2);
        const uint32_t rows = K_DIM / 2;
        auto ubMat = resource.ubBuf.template GetBufferByByte<float>(UB_P2_Y);
        auto ubIn = resource.ubBuf.template GetBufferByByte<STATE_TYPE>(UB_P2_H0IN);

        // Y = [I | H0]; sub-block 0 owns the M columns, sub-block 1 the S columns.
        // Each sub-block covers all K rows of its own column block.
        if (subBlock == 0) {
            AscendC::Duplicate(ubMat, 0.0f, K_DIM * V_DIM);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < K_DIM; ++i) {
                ubMat.SetValue(static_cast<int64_t>(i) * V_DIM + i, 1.0f);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        } else {
            if (useInitialState) {
                if constexpr (std::is_same<STATE_TYPE, float>::value) {
                    // Same dtype: copy straight into the FP32 staging row block.
                    AscendC::DataCopy(ubMat, gmInitialState[StateOffset(b, hv)], K_DIM * V_DIM);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                } else {
                    AscendC::DataCopy(ubIn, gmInitialState[StateOffset(b, hv)], K_DIM * V_DIM);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::Cast(ubMat, ubIn, AscendC::RoundMode::CAST_NONE, K_DIM * V_DIM);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            } else {
                AscendC::Duplicate(ubMat, 0.0f, K_DIM * V_DIM);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        CopyPadToXHalf(0, K_DIM, (subBlock == 0) ? 0 : V_DIM, ubMat);
        AscendC::PipeBarrier<PIPE_MTE3>();
        AivSetToAic(kCpFlagYInit);

        // Prefix: [B_trans | S_local] = B_bar_p @ Y + [0 | A_bar_p]. The GEMM lands in the
        // UB slot, so the bias add runs in place on this sub-block's rows.
        auto ubT = resource.ubBuf.template GetBufferByByte<float>(UB_DIRECT);
        auto ubA = resource.ubBuf.template GetBufferByByte<float>(UB_P2_A);
        for (uint32_t prior = 0; prior < rank; ++prior) {
            AivSetUbFree();   // slot free for the prefix GEMM
            AivWaitFromAic(kCpFlagPrefixGemm);
            AscendC::DataCopy(ubA, gmSummary[SummaryBase(prior) + K_DIM * K_DIM + begin * V_DIM],
                              rows * V_DIM);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            __ubuf__ float *tAddr = reinterpret_cast<__ubuf__ float *>(ubT[V_DIM].GetPhyAddr());
            __ubuf__ float *aAddr = reinterpret_cast<__ubuf__ float *>(ubA.GetPhyAddr());
            AscendC::VF_CALL<CpDetail::AddRowsKernel>(
                tAddr, ROW_DIM, aAddr, static_cast<uint16_t>(rows), static_cast<uint16_t>(V_DIM));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(gmScratch[scr + SCR_X + begin * ROW_DIM], ubT, rows * ROW_DIM);
            AscendC::PipeBarrier<PIPE_MTE3>();
            AivSetToAic(kCpFlagPrefixAdd);
        }

        // S_in lives in the S columns of Y. Each sub-block publishes its own half of
        // the K rows into SCR_H0 (the phase-3 rolling state); the sub-blocks join
        // before h[chunkBegin] is seeded, so no sub-block reads rows still being
        // written by the other one.
        {
            AscendC::DataCopyExtParams params{
                static_cast<uint16_t>(rows),
                static_cast<uint32_t>(V_DIM * sizeof(float)),
                static_cast<uint32_t>(V_DIM * sizeof(float)),
                0, 0};
            AscendC::DataCopyPad(ubMat, gmScratch[scr + SCR_X + begin * ROW_DIM + V_DIM], params,
                                 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::DataCopy(gmScratch[scr + SCR_H0 + begin * V_DIM], ubMat, rows * V_DIM);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
        AivSubBlockBarrier();

        auto ubOut = resource.ubBuf.template GetBufferByByte<INPUT_TYPE>(UB_P2_OUT);
        AscendC::DataCopy(ubMat, gmScratch[scr + SCR_H0 + begin * V_DIM], rows * V_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(ubOut, ubMat, AscendC::RoundMode::CAST_RINT, rows * V_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(gmH[HOffset(b, hv, chunkBegin) + begin * V_DIM], ubOut, rows * V_DIM);
        AivSetToAic(kCpFlagReplayReady);
    }

    __aicore__ inline void Phase3Vector(uint32_t b, uint32_t hv)
    {
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        const uint64_t scr = ScratchBase();
        auto ubVw = resource.ubBuf.template GetBufferByByte<float>(UB_DIRECT);
        auto ubIn = resource.ubBuf.template GetBufferByByte<INPUT_TYPE>(UB_P3_UIN);
        auto ubVu = resource.ubBuf.template GetBufferByByte<INPUT_TYPE>(UB_P3_VU);
        auto ubOut = resource.ubBuf.template GetBufferByByte<INPUT_TYPE>(UB_P3_OUT);
        auto ubQ = resource.ubBuf.template GetBufferByByte<float>(UB_Q);
        auto ubQNeg = resource.ubBuf.template GetBufferByByte<float>(UB_Q_NEG);
        auto gmVu = gmScratch[scr + SCR_KF].template ReinterpretCast<INPUT_TYPE>();
        for (uint32_t chunk = chunkBegin; chunk < chunkEnd; ++chunk) {
            const uint32_t tokens = ChunkTokens(chunk);
            const uint64_t tokenBase = static_cast<uint64_t>(chunk) * chunkSize;
            const uint32_t rowsPerSubBlock = RowsPerSubBlock(tokens);
            const uint32_t rowBegin = subBlock * rowsPerSubBlock;
            const uint32_t rowEnd = Catlass::Min(rowBegin + rowsPerSubBlock, tokens);

            AivSetUbFree();   // slot free for C1
            AivWaitFromAic(kCpFlagC1Done);
            const uint32_t rows = rowEnd - rowBegin;
            if (rows > 0) {
                PrepareQ(b, hv, tokenBase, rowBegin, rows, LoadG(b, hv, tokenBase + tokens - 1),
                         ubQ, ubQNeg);
                AscendC::DataCopy(ubIn, gmU[HvOffset(b, hv, tokenBase + rowBegin, V_DIM)], rows * V_DIM);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                __ubuf__ float *vAddr = reinterpret_cast<__ubuf__ float *>(ubVw.GetPhyAddr());
                __ubuf__ INPUT_TYPE *uAddr = reinterpret_cast<__ubuf__ INPUT_TYPE *>(ubIn.GetPhyAddr());
                __ubuf__ float *qAddr = reinterpret_cast<__ubuf__ float *>(ubQ.GetPhyAddr());
                // v_new = U - v_work (un-decayed, written at input precision).
                AscendC::VF_CALL<Catlass::Epilogue::Block::detail::ComputeVNewRegbaseDualIssue<INPUT_TYPE>>(
                    vAddr, uAddr, rows * V_DIM);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(ubOut, ubVw, AscendC::RoundMode::CAST_RINT, rows * V_DIM);
                AscendC::PipeBarrier<PIPE_V>();
                // v_update = q * v_new feeds C2.
                AscendC::VF_CALL<Catlass::Epilogue::Block::detail::ApplyRowScaleDualIssue>(
                    vAddr, qAddr, 0, static_cast<uint16_t>(rows), static_cast<uint16_t>(V_DIM));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(ubVu, ubVw, AscendC::RoundMode::CAST_RINT, rows * V_DIM);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopy(gmVNew[HvOffset(b, hv, tokenBase + rowBegin, V_DIM)], ubOut,
                                  rows * V_DIM);
                AscendC::DataCopy(gmVu[rowBegin * V_DIM], ubVu, rows * V_DIM);
            }
            AscendC::PipeBarrier<PIPE_MTE3>();
            AivSetToAic(kCpFlagVUpdate);

            const bool isLastChunk = (chunk + 1 == totalChunks);
            const bool needStage2 = storeFinalState || !isLastChunk;
            if (needStage2) {
                AivSetUbFree();   // slot free for C2
                AivWaitFromAic(kCpFlagC2Done);
                UpdateStateAndH(b, hv, chunk, isLastChunk);
            }
            AivSetToAic(kCpFlagHDone);
        }
    }

    // S_next = D*S + h_work; write h[chunk+1] inside this rank's range and, on the last
    // global chunk, final_state. Half of the K rows is owned by each sub-block.
    __aicore__ inline void UpdateStateAndH(uint32_t b, uint32_t hv, uint32_t chunk, bool isLastChunk)
    {
        const uint64_t scr = ScratchBase();
        const uint32_t tokens = ChunkTokens(chunk);
        const uint64_t tokenBase = static_cast<uint64_t>(chunk) * chunkSize;
        const float decay = expf(LoadG(b, hv, tokenBase + tokens - 1));
        const uint32_t begin = AscendC::GetSubBlockIdx() * (K_DIM / 2);
        const uint32_t rows = K_DIM / 2;
        auto ubS = resource.ubBuf.template GetBufferByByte<float>(UB_P3_S);
        auto ubHw = resource.ubBuf.template GetBufferByByte<float>(UB_DIRECT);
        auto ubOut = resource.ubBuf.template GetBufferByByte<INPUT_TYPE>(UB_P3_OUT);
        auto ubState = resource.ubBuf.template GetBufferByByte<STATE_TYPE>(UB_P3_ST);

        AscendC::DataCopy(ubS, gmScratch[scr + SCR_H0 + begin * V_DIM], rows * V_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Muls(ubS, ubS, decay, rows * V_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float>(ubS, ubS, ubHw, rows * V_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        if (chunk + 1 < chunkEnd) {
            AscendC::Cast(ubOut, ubS, AscendC::RoundMode::CAST_RINT, rows * V_DIM);
            AscendC::PipeBarrier<PIPE_V>();
        } else if (isLastChunk && storeFinalState) {
            if constexpr (!std::is_same<STATE_TYPE, float>::value) {
                AscendC::Cast(ubState, ubS, AscendC::RoundMode::CAST_RINT, rows * V_DIM);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(gmScratch[scr + SCR_H0 + begin * V_DIM], ubS, rows * V_DIM);
        if (chunk + 1 < chunkEnd) {
            AscendC::DataCopy(gmH[HOffset(b, hv, chunk + 1) + begin * V_DIM], ubOut, rows * V_DIM);
        } else if (isLastChunk && storeFinalState) {
            if constexpr (std::is_same<STATE_TYPE, float>::value) {
                AscendC::DataCopy(gmFinalState[StateOffset(b, hv) + begin * V_DIM], ubS,
                                  rows * V_DIM);
            } else {
                AscendC::DataCopy(gmFinalState[StateOffset(b, hv) + begin * V_DIM], ubState,
                                  rows * V_DIM);
            }
        }
        AscendC::PipeBarrier<PIPE_MTE3>();
    }
};

template <typename INPUT_TYPE, typename G_TYPE, typename STATE_TYPE>
__aicore__ inline void ChunkGatedDeltaRuleFwdHCPImpl(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g,
                                                     GM_ADDR inital_state, GM_ADDR h, GM_ADDR v_new,
                                                     GM_ADDR final_state, GM_ADDR tiling, GM_ADDR user)
{
    GDNFwdHCPKernel<INPUT_TYPE, G_TYPE, STATE_TYPE> kernel;
    kernel.Init(k, w, u, g, inital_state, h, v_new, final_state, tiling, user);
    kernel.Process();
}

} // namespace GDN

#endif // GDN_FWD_H_CP_KERNEL_HPP
