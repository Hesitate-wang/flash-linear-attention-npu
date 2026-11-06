/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 *
 * FP32 register-resident math for the chunk64 implementation.
 */
#ifndef GDN_CHUNK64_SPARSE_VECTOR_H
#define GDN_CHUNK64_SPARSE_VECTOR_H

#include "sparse_common.h"

namespace GdnChunk64Sparse {
using namespace AscendC::MicroAPI;

constexpr CastTrait kSparseCastB16F32 = {
    RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr CastTrait kSparseCastF32B16 = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait kSparseCastF32B16Hi = {
    RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

__aicore__ inline void LoadBf16(RegTensor<float> &dst, __ubuf__ bfloat16_t *src,
                                MaskReg mask)
{
    RegTensor<bfloat16_t> raw;
    LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(raw, src);
    Cast<float, bfloat16_t, kSparseCastB16F32>(dst, raw, mask);
}

__aicore__ inline void StoreBf16(__ubuf__ bfloat16_t *dst, RegTensor<float> &src,
                                 MaskReg mask)
{
    RegTensor<bfloat16_t> packed;
    Cast<bfloat16_t, float, kSparseCastF32B16>(packed, src, mask);
    StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(dst, packed, mask);
}

__aicore__ inline void GenerateScanIndices(LocalTensor<uint32_t> &shift)
{
    __ubuf__ uint32_t *dst = reinterpret_cast<__ubuf__ uint32_t *>(shift.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<uint32_t, MaskPattern::ALL>();
        RegTensor<int32_t> index, source, zero, distance;
        Arange<int32_t, IndexOrder::INCREASE_ORDER>(index, 0);
        Duplicate(zero, 0, all);
        for (uint16_t stage = 0; stage < 6; ++stage) {
            const int32_t d = 1 << stage;
            Duplicate(distance, d, all);
            Sub(source, index, distance, all);
            MaskReg valid;
            CompareScalar<int32_t, CMPMODE::GE>(valid, index, d, all);
            Select(source, source, zero, valid);
            StoreAlign(dst + static_cast<uint32_t>(stage) * 64,
                       (RegTensor<uint32_t> &)source, all);
        }
    }
}

__aicore__ inline void PrepareGateBetaVF(LocalTensor<float> &g,
                                         LocalTensor<float> &beta,
                                         float aLog, float dtBias,
                                         bool softplus, bool sigmoid,
                                         float betaScale)
{
    __ubuf__ float *gAddr = reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr());
    __ubuf__ float *bAddr = reinterpret_cast<__ubuf__ float *>(beta.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> x, ax, e, one, out;
        LoadAlign(x, gAddr);
        if (softplus) {
            Adds(x, x, dtBias, all);
            Abs(ax, x, all);
            Muls(ax, ax, -1.0f, all);
            Exp(e, ax, all);
            Adds(e, e, 1.0f, all);
            Ln(out, e, all);
            Duplicate(one, 0.0f, all);
            Max(ax, x, one, all);
            Add(out, out, ax, all);
            Duplicate(one, aLog, all);
            Exp(one, one, all);
            Muls(one, one, -1.0f, all);
            Mul(out, out, one, all);
            StoreAlign(gAddr, out, all);
        }
        if (sigmoid) {
            LoadAlign(x, bAddr);
            Muls(x, x, -1.0f, all);
            Exp(e, x, all);
            Duplicate(one, 1.0f, all);
            Add(e, e, one, all);
            Div(out, one, e, all);
            Muls(out, out, betaScale, all);
            StoreAlign(bAddr, out, all);
        }
    }
}

template <bool UseExp2>
__aicore__ inline void CumsumEnergyVF(LocalTensor<float> &g,
                                      LocalTensor<float> &energy,
                                      LocalTensor<uint32_t> &shift)
{
    __ubuf__ float *gAddr = reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr());
    __ubuf__ float *eAddr = reinterpret_cast<__ubuf__ float *>(energy.GetPhyAddr());
    __ubuf__ uint32_t *sAddr = reinterpret_cast<__ubuf__ uint32_t *>(shift.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> value, gathered;
        RegTensor<uint32_t> index;
        LoadAlign(value, gAddr);
        if constexpr (UseExp2) {
            Muls(value, value, kSparseRcpLn2, all);
        }
        for (uint16_t stage = 0; stage < 6; ++stage) {
            LoadAlign(index, sAddr + static_cast<uint32_t>(stage) * 64);
            Gather(gathered, value, index);
            uint32_t lowerCount = 1U << stage;
            MaskReg lower = UpdateMask<float>(lowerCount);
            MaskReg active;
            Xor(active, all, lower, all);
            Add<float, MaskMergeMode::MERGING>(value, value, gathered, active);
        }
        StoreAlign(gAddr, value, all);
        if constexpr (UseExp2) {
            Muls(value, value, kSparseLn2, all);
        }
        Exp(value, value, all);
        StoreAlign(eAddr, value, all);
    }
}

template <bool UseExp2>
__aicore__ inline void EnergyVF(LocalTensor<float> &g,
                                LocalTensor<float> &energy)
{
    __ubuf__ float *gAddr = reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr());
    __ubuf__ float *eAddr = reinterpret_cast<__ubuf__ float *>(energy.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> value;
        LoadAlign(value, gAddr);
        if constexpr (UseExp2) {
            Muls(value, value, kSparseLn2, all);
        }
        Exp(value, value, all);
        StoreAlign(eAddr, value, all);
    }
}

// Both the energies and every possible pairwise ratio must stay in the
// normal FP32 range. NaN fails all comparisons and selects the safe path.
__aicore__ inline bool GateRatioSafe(LocalTensor<float> &g, uint32_t rows)
{
    float lo = g.GetValue(0);
    float hi = lo;
    for (uint32_t row = 1; row < rows; ++row) {
        const float value = g.GetValue(row);
        lo = value < lo ? value : lo;
        hi = value > hi ? value : hi;
    }
    return lo >= -kSparseExpSafeLimit && hi <= kSparseExpSafeLimit &&
           hi - lo <= kSparseExpSafeLimit;
}

// Normalize a 32-row band. The whole band is handled in one vector scope;
// padded rows are zeros and their reciprocal norms are never exported.
__aicore__ inline void NormalizeRowsVF(LocalTensor<bfloat16_t> &input,
                                       LocalTensor<bfloat16_t> &output,
                                       LocalTensor<float> &rstd)
{
    __ubuf__ bfloat16_t *src = reinterpret_cast<__ubuf__ bfloat16_t *>(input.GetPhyAddr());
    __ubuf__ bfloat16_t *dst = reinterpret_cast<__ubuf__ bfloat16_t *>(output.GetPhyAddr());
    __ubuf__ float *norm = reinterpret_cast<__ubuf__ float *>(rstd.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        MaskReg scalar = CreateMask<float, MaskPattern::VL1>();
        RegTensor<float> x0, x1, sum0, sum1, one, scale;
        Duplicate(one, 1.0f, all);
        for (uint16_t row = 0; row < 32; ++row) {
            const uint32_t offset = static_cast<uint32_t>(row) * 128;
            LoadBf16(x0, src + offset, all);
            LoadBf16(x1, src + offset + 64, all);
            Mul(sum0, x0, x0, all);
            Mul(sum1, x1, x1, all);
            Reduce<Reg::ReduceType::SUM>(sum0, sum0, all);
            Reduce<Reg::ReduceType::SUM>(sum1, sum1, all);
            Add(sum0, sum0, sum1, scalar);
            Adds(sum0, sum0, kSparseEps, scalar);
            Sqrt(sum0, sum0, scalar);
            Div(sum0, one, sum0, scalar);
            Duplicate(scale, sum0, all);
            Mul(x0, x0, scale, all);
            Mul(x1, x1, scale, all);
            StoreBf16(dst + offset, x0, all);
            StoreBf16(dst + offset + 64, x1, all);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(norm + row, sum0, scalar);
        }
    }
}

__aicore__ inline void ScaleVbVF(LocalTensor<bfloat16_t> &v,
                                 LocalTensor<bfloat16_t> &out,
                                 LocalTensor<float> &beta,
                                 uint32_t firstRow)
{
    __ubuf__ bfloat16_t *src = reinterpret_cast<__ubuf__ bfloat16_t *>(v.GetPhyAddr());
    __ubuf__ bfloat16_t *dst = reinterpret_cast<__ubuf__ bfloat16_t *>(out.GetPhyAddr());
    __ubuf__ float *b = reinterpret_cast<__ubuf__ float *>(beta.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> x0, x1, scale;
        for (uint16_t row = 0; row < 32; ++row) {
            const uint32_t offset = static_cast<uint32_t>(row) * 128;
            LoadBf16(x0, src + offset, all);
            LoadBf16(x1, src + offset + 64, all);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scale, b + firstRow + row);
            Mul(x0, x0, scale, all);
            Mul(x1, x1, scale, all);
            StoreBf16(dst + offset, x0, all);
            StoreBf16(dst + offset + 64, x1, all);
        }
    }
}

__aicore__ inline void ScaleKbgVF(LocalTensor<bfloat16_t> &k,
                                  LocalTensor<bfloat16_t> &out,
                                  LocalTensor<float> &beta,
                                  LocalTensor<float> &energy,
                                  uint32_t firstRow)
{
    __ubuf__ bfloat16_t *src = reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr());
    __ubuf__ bfloat16_t *dst = reinterpret_cast<__ubuf__ bfloat16_t *>(out.GetPhyAddr());
    __ubuf__ float *b = reinterpret_cast<__ubuf__ float *>(beta.GetPhyAddr());
    __ubuf__ float *e = reinterpret_cast<__ubuf__ float *>(energy.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> x0, x1, scale, be;
        for (uint16_t row = 0; row < 32; ++row) {
            const uint32_t offset = static_cast<uint32_t>(row) * 128;
            LoadBf16(x0, src + offset, all);
            LoadBf16(x1, src + offset + 64, all);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(be, b + firstRow + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scale, e + firstRow + row);
            Mul(scale, scale, be, all);
            Mul(x0, x0, scale, all);
            Mul(x1, x1, scale, all);
            StoreBf16(dst + offset, x0, all);
            StoreBf16(dst + offset + 64, x1, all);
        }
    }
}

// Each AIV owns one 32x32 diagonal inverse. The lower band still has 64
// columns; AIV1 additionally keeps a compact L10 copy at a[2048].
template <bool UseExp2>
__aicore__ inline void Inverse32DiagonalVF(LocalTensor<float> &kkt,
                                           LocalTensor<float> &lower,
                                           LocalTensor<float> &a,
                                           LocalTensor<float> &g,
                                           LocalTensor<float> &beta,
                                           LocalTensor<float> &energy,
                                           bool useEnergyRatio,
                                           uint32_t firstRow)
{
    __ubuf__ float *kk = reinterpret_cast<__ubuf__ float *>(kkt.GetPhyAddr());
    __ubuf__ float *l = reinterpret_cast<__ubuf__ float *>(lower.GetPhyAddr());
    __ubuf__ float *x = reinterpret_cast<__ubuf__ float *>(a.GetPhyAddr());
    __ubuf__ float *gAddr = reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr());
    __ubuf__ float *b = reinterpret_cast<__ubuf__ float *>(beta.GetPhyAddr());
    __ubuf__ float *eAddr = reinterpret_cast<__ubuf__ float *>(energy.GetPhyAddr());
    const uint32_t diagonalCol = firstRow;
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        MaskReg half = CreateMask<float, MaskPattern::VL32>();
        RegTensor<float> gJ, gI, bi, diff, gate, score, zero, clipLo, clipHi, expLo, expHi;
        Duplicate(zero, 0.0f, all);
        Duplicate(clipLo, -50.0f, all);
        Duplicate(clipHi, 50.0f, all);
        Duplicate(expLo, 1.92874985e-22f, all);
        Duplicate(expHi, 5.18470553e21f, all);
        if (useEnergyRatio) {
            LoadAlign(gJ, eAddr);
        } else {
            LoadAlign(gJ, gAddr);
        }
        for (uint16_t row = 0; row < 32; ++row) {
            const uint32_t globalRow = firstRow + row;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(bi, b + globalRow);
            if (useEnergyRatio) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(gI, eAddr + globalRow);
                Div(gate, gI, gJ, all);
                Max(gate, gate, expLo, all);
                Min(gate, gate, expHi, all);
            } else {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(gI, gAddr + globalRow);
                Sub(diff, gI, gJ, all);
                Max(diff, diff, clipLo, all);
                Min(diff, diff, clipHi, all);
                if constexpr (UseExp2) {
                    Muls(diff, diff, kSparseLn2, all);
                }
                Exp(gate, diff, all);
            }
            Mul(gate, gate, bi, all);
            Muls(gate, gate, -1.0f, all);
            LoadAlign(score, kk + static_cast<uint32_t>(row) * 64);
            Mul(gate, gate, score, all);
            MaskReg lowerMask = UpdateMask<float>(globalRow);
            Select(gate, gate, zero, lowerMask);
            StoreAlign(l + static_cast<uint32_t>(row) * 64, gate, all);
            if (firstRow != 0) {
                StoreAlign(x + 2048 + static_cast<uint32_t>(row) * 32, gate, half);
            }
        }
    }
    PipeBarrier<PIPE_V>();
#define SPARSE_SOLVE_PREV(dst, rr)                                                                                     \
    LoadAlign<float, LoadDist::DIST_BRC_B32>(coefficient,                                                             \
        l + (base + (rr)) * 64 + diagonalCol + col);                                                                    \
    MulAddDst(dst, previous, coefficient, all);
#define SPARSE_SOLVE_LOCAL(dst, src, rr, cc)                                                                           \
    LoadAlign<float, LoadDist::DIST_BRC_B32>(coefficient,                                                             \
        l + (base + (rr)) * 64 + diagonalCol + base + (cc));                                                            \
    MulAddDst(dst, src, coefficient, all);
    for (uint16_t group = 0; group < 4; ++group) {
        const uint32_t base = static_cast<uint32_t>(group) * 8;
        __VEC_SCOPE__
        {
            MaskReg all = CreateMask<float, MaskPattern::VL32>();
            RegTensor<float> y0, y1, y2, y3, y4, y5, y6, y7;
            RegTensor<float> previous, coefficient, zero, one;
            RegTensor<int32_t> lanes;
            Duplicate(zero, 0.0f, all);
            Duplicate(one, 1.0f, all);
            Arange<int32_t, IndexOrder::INCREASE_ORDER>(lanes, 0);
            MaskReg diagonal;
            CompareScalar<int32_t, CMPMODE::EQ>(diagonal, lanes, static_cast<int32_t>(base + 0), all);
            Select(y0, one, zero, diagonal);
            CompareScalar<int32_t, CMPMODE::EQ>(diagonal, lanes, static_cast<int32_t>(base + 1), all);
            Select(y1, one, zero, diagonal);
            CompareScalar<int32_t, CMPMODE::EQ>(diagonal, lanes, static_cast<int32_t>(base + 2), all);
            Select(y2, one, zero, diagonal);
            CompareScalar<int32_t, CMPMODE::EQ>(diagonal, lanes, static_cast<int32_t>(base + 3), all);
            Select(y3, one, zero, diagonal);
            CompareScalar<int32_t, CMPMODE::EQ>(diagonal, lanes, static_cast<int32_t>(base + 4), all);
            Select(y4, one, zero, diagonal);
            CompareScalar<int32_t, CMPMODE::EQ>(diagonal, lanes, static_cast<int32_t>(base + 5), all);
            Select(y5, one, zero, diagonal);
            CompareScalar<int32_t, CMPMODE::EQ>(diagonal, lanes, static_cast<int32_t>(base + 6), all);
            Select(y6, one, zero, diagonal);
            CompareScalar<int32_t, CMPMODE::EQ>(diagonal, lanes, static_cast<int32_t>(base + 7), all);
            Select(y7, one, zero, diagonal);
            for (uint16_t col = 0; col < base; ++col) {
                LoadAlign(previous, x + static_cast<uint32_t>(col) * 32);
                SPARSE_SOLVE_PREV(y0, 0)
                SPARSE_SOLVE_PREV(y1, 1)
                SPARSE_SOLVE_PREV(y2, 2)
                SPARSE_SOLVE_PREV(y3, 3)
                SPARSE_SOLVE_PREV(y4, 4)
                SPARSE_SOLVE_PREV(y5, 5)
                SPARSE_SOLVE_PREV(y6, 6)
                SPARSE_SOLVE_PREV(y7, 7)
            }
            SPARSE_SOLVE_LOCAL(y1, y0, 1, 0)
            SPARSE_SOLVE_LOCAL(y2, y0, 2, 0)
            SPARSE_SOLVE_LOCAL(y2, y1, 2, 1)
            SPARSE_SOLVE_LOCAL(y3, y0, 3, 0)
            SPARSE_SOLVE_LOCAL(y3, y1, 3, 1)
            SPARSE_SOLVE_LOCAL(y3, y2, 3, 2)
            SPARSE_SOLVE_LOCAL(y4, y0, 4, 0)
            SPARSE_SOLVE_LOCAL(y4, y1, 4, 1)
            SPARSE_SOLVE_LOCAL(y4, y2, 4, 2)
            SPARSE_SOLVE_LOCAL(y4, y3, 4, 3)
            SPARSE_SOLVE_LOCAL(y5, y0, 5, 0)
            SPARSE_SOLVE_LOCAL(y5, y1, 5, 1)
            SPARSE_SOLVE_LOCAL(y5, y2, 5, 2)
            SPARSE_SOLVE_LOCAL(y5, y3, 5, 3)
            SPARSE_SOLVE_LOCAL(y5, y4, 5, 4)
            SPARSE_SOLVE_LOCAL(y6, y0, 6, 0)
            SPARSE_SOLVE_LOCAL(y6, y1, 6, 1)
            SPARSE_SOLVE_LOCAL(y6, y2, 6, 2)
            SPARSE_SOLVE_LOCAL(y6, y3, 6, 3)
            SPARSE_SOLVE_LOCAL(y6, y4, 6, 4)
            SPARSE_SOLVE_LOCAL(y6, y5, 6, 5)
            SPARSE_SOLVE_LOCAL(y7, y0, 7, 0)
            SPARSE_SOLVE_LOCAL(y7, y1, 7, 1)
            SPARSE_SOLVE_LOCAL(y7, y2, 7, 2)
            SPARSE_SOLVE_LOCAL(y7, y3, 7, 3)
            SPARSE_SOLVE_LOCAL(y7, y4, 7, 4)
            SPARSE_SOLVE_LOCAL(y7, y5, 7, 5)
            SPARSE_SOLVE_LOCAL(y7, y6, 7, 6)
            StoreAlign(x + (base + 0) * 32, y0, all);
            StoreAlign(x + (base + 1) * 32, y1, all);
            StoreAlign(x + (base + 2) * 32, y2, all);
            StoreAlign(x + (base + 3) * 32, y3, all);
            StoreAlign(x + (base + 4) * 32, y4, all);
            StoreAlign(x + (base + 5) * 32, y5, all);
            StoreAlign(x + (base + 6) * 32, y6, all);
            StoreAlign(x + (base + 7) * 32, y7, all);
        }
        PipeBarrier<PIPE_V>();
    }
#undef SPARSE_SOLVE_PREV
#undef SPARSE_SOLVE_LOCAL
}

// X10 = -X1 @ L10 @ X0. Each AIV evaluates sixteen rows in FP32:
// first tmp = X1[rowBegin:rowBegin+16] @ L10, then -tmp @ X0.
// a[0:1024] holds the local diagonal inverse, a[1024:1536] tmp,
// a[1536:2048] X10, and AIV1's a[2048:3072] holds its local L10.
__aicore__ inline void InverseCoupling16VF(LocalTensor<float> &a,
                                           LocalTensor<float> &exchange,
                                           uint32_t sub)
{
    __ubuf__ float *aAddr = reinterpret_cast<__ubuf__ float *>(a.GetPhyAddr());
    __ubuf__ float *peer = reinterpret_cast<__ubuf__ float *>(exchange.GetPhyAddr());
    __ubuf__ float *x0 = sub == 0 ? aAddr : peer;
    __ubuf__ float *x1 = sub == 1 ? aAddr : peer;
    __ubuf__ float *l10 = sub == 1 ? aAddr + 2048 : peer + 1024;
    __ubuf__ float *tmp = aAddr + 1024;
    __ubuf__ float *out = aAddr + 1536;
    const uint32_t rowBegin = sub * 16;
    __VEC_SCOPE__
    {
        MaskReg half = CreateMask<float, MaskPattern::VL32>();
        RegTensor<float> acc, row, coefficient;
        for (uint16_t r = 0; r < 16; ++r) {
            Duplicate(acc, 0.0f, half);
            for (uint16_t t = 0; t < 32; ++t) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(
                    coefficient, x1 + (rowBegin + r) * 32 + t);
                LoadAlign(row, l10 + static_cast<uint32_t>(t) * 32);
                MulAddDst(acc, row, coefficient, half);
            }
            StoreAlign(tmp + static_cast<uint32_t>(r) * 32, acc, half);
        }
    }
    PipeBarrier<PIPE_V>();
    __VEC_SCOPE__
    {
        MaskReg half = CreateMask<float, MaskPattern::VL32>();
        RegTensor<float> acc, row, coefficient;
        for (uint16_t r = 0; r < 16; ++r) {
            Duplicate(acc, 0.0f, half);
            for (uint16_t t = 0; t < 32; ++t) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(
                    coefficient, tmp + static_cast<uint32_t>(r) * 32 + t);
                LoadAlign(row, x0 + static_cast<uint32_t>(t) * 32);
                MulAddDst(acc, row, coefficient, half);
            }
            Muls(acc, acc, -1.0f, half);
            StoreAlign(out + static_cast<uint32_t>(r) * 32, acc, half);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <bool UseExp2>
__aicore__ inline void APrimeBandVF(LocalTensor<float> &qk,
                                    LocalTensor<bfloat16_t> &out,
                                    LocalTensor<float> &g,
                                    LocalTensor<float> &energy,
                                    bool useEnergyRatio,
                                    uint32_t firstRow)
{
    __ubuf__ float *score = reinterpret_cast<__ubuf__ float *>(qk.GetPhyAddr());
    __ubuf__ bfloat16_t *dst = reinterpret_cast<__ubuf__ bfloat16_t *>(out.GetPhyAddr());
    __ubuf__ float *gAddr = reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr());
    __ubuf__ float *eAddr = reinterpret_cast<__ubuf__ float *>(energy.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> gJ, gI, diff, gate, raw, zero;
        Duplicate(zero, 0.0f, all);
        if (useEnergyRatio) {
            LoadAlign(gJ, eAddr);
        } else {
            LoadAlign(gJ, gAddr);
        }
        for (uint16_t row = 0; row < 32; ++row) {
            const uint32_t globalRow = firstRow + row;
            if (useEnergyRatio) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(gI, eAddr + globalRow);
                Div(gate, gI, gJ, all);
            } else {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(gI, gAddr + globalRow);
                Sub(diff, gI, gJ, all);
                if constexpr (UseExp2) {
                    Muls(diff, diff, kSparseLn2, all);
                }
                Exp(gate, diff, all);
            }
            LoadAlign(raw, score + static_cast<uint32_t>(row) * 64);
            Mul(gate, gate, raw, all);
            uint32_t lowerCount = globalRow + 1U;
            MaskReg lower = UpdateMask<float>(lowerCount);
            Select(gate, gate, zero, lower);
            StoreBf16(dst + static_cast<uint32_t>(row) * 64, gate, all);
        }
    }
}

__aicore__ inline void QExpVF(LocalTensor<bfloat16_t> &q,
                              LocalTensor<bfloat16_t> &out,
                              LocalTensor<float> &energy,
                              uint32_t firstRow)
{
    __ubuf__ bfloat16_t *src = reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr());
    __ubuf__ bfloat16_t *dst = reinterpret_cast<__ubuf__ bfloat16_t *>(out.GetPhyAddr());
    __ubuf__ float *e = reinterpret_cast<__ubuf__ float *>(energy.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> x0, x1, scale;
        for (uint16_t row = 0; row < 32; ++row) {
            const uint32_t offset = static_cast<uint32_t>(row) * 128;
            LoadBf16(x0, src + offset, all);
            LoadBf16(x1, src + offset + 64, all);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scale, e + firstRow + row);
            Mul(x0, x0, scale, all);
            Mul(x1, x1, scale, all);
            StoreBf16(dst + offset, x0, all);
            StoreBf16(dst + offset + 64, x1, all);
        }
    }
}

template <bool UseExp2>
__aicore__ inline void VNewRightVF(LocalTensor<float> &p,
                                   LocalTensor<bfloat16_t> &u,
                                   LocalTensor<bfloat16_t> &vNew,
                                   LocalTensor<bfloat16_t> &right,
                                   LocalTensor<float> &g,
                                   LocalTensor<float> &energy,
                                   float gLast, float lastEnergy,
                                   bool useEnergyRatio, uint32_t firstRow)
{
    __ubuf__ float *pAddr = reinterpret_cast<__ubuf__ float *>(p.GetPhyAddr());
    __ubuf__ bfloat16_t *uAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(u.GetPhyAddr());
    __ubuf__ bfloat16_t *vAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(vNew.GetPhyAddr());
    __ubuf__ bfloat16_t *rAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(right.GetPhyAddr());
    __ubuf__ float *gAddr = reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr());
    __ubuf__ float *eAddr = reinterpret_cast<__ubuf__ float *>(energy.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> p0, p1, u0, u1, scale, last, diff;
        Duplicate(last, useEnergyRatio ? lastEnergy : gLast, all);
        for (uint16_t row = 0; row < 32; ++row) {
            const uint32_t offset = static_cast<uint32_t>(row) * 128;
            LoadAlign(p0, pAddr + offset);
            LoadAlign(p1, pAddr + offset + 64);
            LoadBf16(u0, uAddr + offset, all);
            LoadBf16(u1, uAddr + offset + 64, all);
            Sub(u0, u0, p0, all);
            Sub(u1, u1, p1, all);
            StoreBf16(vAddr + offset, u0, all);
            StoreBf16(vAddr + offset + 64, u1, all);
            if (useEnergyRatio) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(scale, eAddr + firstRow + row);
                Div(scale, last, scale, all);
            } else {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(scale, gAddr + firstRow + row);
                Sub(diff, last, scale, all);
                if constexpr (UseExp2) {
                    Muls(diff, diff, kSparseLn2, all);
                }
                Exp(scale, diff, all);
            }
            Mul(u0, u0, scale, all);
            Mul(u1, u1, scale, all);
            StoreBf16(rAddr + offset, u0, all);
            StoreBf16(rAddr + offset + 64, u1, all);
        }
    }
}

__aicore__ inline void StoreFlat128(__ubuf__ bfloat16_t *dst,
                                    RegTensor<float> &even,
                                    RegTensor<float> &odd,
                                    MaskReg &m16, MaskReg &m128)
{
    RegTensor<bfloat16_t> lo, hi;
    Cast<bfloat16_t, float, kSparseCastF32B16>(lo, even, m16);
    Cast<bfloat16_t, float, kSparseCastF32B16Hi>(hi, odd, m16);
    Or((RegTensor<uint16_t> &)lo, (RegTensor<uint16_t> &)lo,
       (RegTensor<uint16_t> &)hi, m16);
    StoreAlign(dst, lo, m128);
}

__aicore__ inline void StoreState128(__ubuf__ float *fp32,
                                     __ubuf__ bfloat16_t *bf16,
                                     RegTensor<float> &even,
                                     RegTensor<float> &odd,
                                     MaskReg &all, MaskReg &m16, MaskReg &m128)
{
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(fp32, even, odd, all);
    StoreFlat128(bf16, even, odd, m16, m128);
}

__aicore__ inline void StateUpdateVF(LocalTensor<float> &state,
                                     LocalTensor<bfloat16_t> &stateBf,
                                     LocalTensor<float> &d, float decay)
{
    __ubuf__ float *s = reinterpret_cast<__ubuf__ float *>(state.GetPhyAddr());
    __ubuf__ bfloat16_t *b = reinterpret_cast<__ubuf__ bfloat16_t *>(stateBf.GetPhyAddr());
    __ubuf__ float *dd = reinterpret_cast<__ubuf__ float *>(d.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> s0, s1, d0, d1, alpha;
        Duplicate(alpha, decay, all);
        for (uint16_t row = 0; row < 64; ++row) {
            const uint32_t offset = static_cast<uint32_t>(row) * 128;
            LoadAlign(s0, s + offset);
            LoadAlign(s1, s + offset + 64);
            LoadAlign(d0, dd + offset);
            LoadAlign(d1, dd + offset + 64);
            Mul(s0, s0, alpha, all);
            Mul(s1, s1, alpha, all);
            Add(s0, s0, d0, all);
            Add(s1, s1, d1, all);
            StoreAlign(s + offset, s0, all);
            StoreAlign(s + offset + 64, s1, all);
            StoreBf16(b + offset, s0, all);
            StoreBf16(b + offset + 64, s1, all);
        }
    }
}

// Public state is ND. Convert only at chain boundaries; each chunk keeps NZ.
__aicore__ inline void Bf16NdToPanels(LocalTensor<bfloat16_t> nz,
                                      LocalTensor<bfloat16_t> nd)
{
    __ubuf__ bfloat16_t *nAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(nd.GetPhyAddr());
    __ubuf__ bfloat16_t *zAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(nz.GetPhyAddr());
    __VEC_SCOPE__
    {
        uint32_t n128 = 128;
        MaskReg m128 = UpdateMask<bfloat16_t>(n128);
        for (uint16_t row = 0; row < 64; ++row) {
            RegTensor<bfloat16_t> value;
            LoadAlign(value, nAddr + static_cast<uint32_t>(row) * kDim);
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(
                zAddr + static_cast<uint32_t>(row) * 16, value, 64, m128);
        }
    }
}

__aicore__ inline void Bf16PanelsToNd(LocalTensor<bfloat16_t> nd,
                                      LocalTensor<bfloat16_t> nz)
{
    __ubuf__ bfloat16_t *nAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(nd.GetPhyAddr());
    __ubuf__ bfloat16_t *zAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(nz.GetPhyAddr());
    __VEC_SCOPE__
    {
        uint32_t n128 = 128;
        MaskReg m128 = UpdateMask<bfloat16_t>(n128);
        for (uint16_t row = 0; row < 64; ++row) {
            RegTensor<bfloat16_t> value;
            LoadAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(
                value, zAddr + static_cast<uint32_t>(row) * 16, 64, m128);
            StoreAlign(nAddr + static_cast<uint32_t>(row) * kDim, value, m128);
        }
    }
}

__aicore__ inline void F32NdToPanels(LocalTensor<float> nz,
                                     LocalTensor<float> nd)
{
    __ubuf__ float *nAddr = reinterpret_cast<__ubuf__ float *>(nd.GetPhyAddr());
    __ubuf__ float *zAddr = reinterpret_cast<__ubuf__ float *>(nz.GetPhyAddr());
    __VEC_SCOPE__
    {
        uint32_t n16 = 16;
        MaskReg m16 = UpdateMask<float>(n16);
        for (uint16_t row = 0; row < 64; ++row) {
            for (uint16_t panel = 0; panel < 8; ++panel) {
                RegTensor<float> value;
                const uint32_t src = static_cast<uint32_t>(row) * kDim + panel * 16;
                const uint32_t dst = panel * (64 * 16) + static_cast<uint32_t>(row) * 16;
                LoadAlign(value, nAddr + src);
                StoreAlign(zAddr + dst, value, m16);
            }
        }
    }
}

__aicore__ inline void F32PanelsToNd(LocalTensor<float> nd,
                                     LocalTensor<float> nz)
{
    __ubuf__ float *nAddr = reinterpret_cast<__ubuf__ float *>(nd.GetPhyAddr());
    __ubuf__ float *zAddr = reinterpret_cast<__ubuf__ float *>(nz.GetPhyAddr());
    __VEC_SCOPE__
    {
        uint32_t n16 = 16;
        MaskReg m16 = UpdateMask<float>(n16);
        for (uint16_t row = 0; row < 64; ++row) {
            for (uint16_t panel = 0; panel < 8; ++panel) {
                RegTensor<float> value;
                const uint32_t src = panel * (64 * 16) + static_cast<uint32_t>(row) * 16;
                const uint32_t dst = static_cast<uint32_t>(row) * kDim + panel * 16;
                LoadAlign(value, zAddr + src);
                StoreAlign(nAddr + dst, value, m16);
            }
        }
    }
}

__aicore__ inline void StateWidenVF(LocalTensor<bfloat16_t> &input,
                                    LocalTensor<float> &state,
                                    LocalTensor<bfloat16_t> &stateBf)
{
    __ubuf__ bfloat16_t *src = reinterpret_cast<__ubuf__ bfloat16_t *>(input.GetPhyAddr());
    __ubuf__ float *s = reinterpret_cast<__ubuf__ float *>(state.GetPhyAddr());
    __ubuf__ bfloat16_t *b = reinterpret_cast<__ubuf__ bfloat16_t *>(stateBf.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> v0, v1;
        for (uint16_t row = 0; row < 64; ++row) {
            const uint32_t offset = static_cast<uint32_t>(row) * 128;
            LoadBf16(v0, src + offset, all);
            LoadBf16(v1, src + offset + 64, all);
            StoreAlign(s + offset, v0, all);
            StoreAlign(s + offset + 64, v1, all);
            StoreBf16(b + offset, v0, all);
            StoreBf16(b + offset + 64, v1, all);
        }
    }
}

__aicore__ inline void StateMirrorVF(LocalTensor<float> &state,
                                     LocalTensor<bfloat16_t> &stateBf)
{
    __ubuf__ float *s = reinterpret_cast<__ubuf__ float *>(state.GetPhyAddr());
    __ubuf__ bfloat16_t *b = reinterpret_cast<__ubuf__ bfloat16_t *>(stateBf.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> v0, v1;
        for (uint16_t row = 0; row < 64; ++row) {
            const uint32_t offset = static_cast<uint32_t>(row) * 128;
            LoadAlign(v0, s + offset);
            LoadAlign(v1, s + offset + 64);
            StoreBf16(b + offset, v0, all);
            StoreBf16(b + offset + 64, v1, all);
        }
    }
}

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_VECTOR_H
