/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 *
 * FP32 register-resident math for the chunk64 implementation.
 */
#ifndef GDN_CHUNK64_SPARSE_VECTOR_H
#define GDN_CHUNK64_SPARSE_VECTOR_H

#include "sparse_common.h"
#include "../chunk_gated_delta_rule_fwd_chunk32/chunk_gated_delta_rule_fwd_32_vf.h"

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

// Cast ZERO leaves each BF16 in the low half of a 32-bit lane. DeInterleave
// packs those 64 values before the strided block store; DATA_BLOCK_COPY alone
// does not perform DIST_PACK_B32 packing.
__aicore__ inline void StoreBf16Nz64(__ubuf__ bfloat16_t *nz, uint32_t row,
                                     RegTensor<float> &src, MaskReg all, MaskReg m64,
                                     uint32_t totalRows = kHalfRows)
{
    RegTensor<bfloat16_t> packed, values, unused;
    Cast<bfloat16_t, float, kSparseCastF32B16>(packed, src, all);
    DeInterleave(values, unused, packed, packed);
    StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(
        nz + row * 16, values, totalRows, m64);
}

// Store two contiguous FP32 halves as one 128-column tight-NZ row. Each
// 64-column group occupies four panels of totalRows*16 BF16 elements.
__aicore__ inline void StoreBf16Nz128(__ubuf__ bfloat16_t *nz, uint32_t row,
                                      RegTensor<float> &lo, RegTensor<float> &hi,
                                      MaskReg all, MaskReg m64, uint32_t totalRows)
{
    StoreBf16Nz64(nz, row, lo, all, m64, totalRows);
    StoreBf16Nz64(nz + totalRows * 64, row, hi, all, m64, totalRows);
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
        if (softplus) {
            LoadAlign(x, gAddr);
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
                                      LocalTensor<float> &energyRecip,
                                      LocalTensor<uint32_t> &shift)
{
    __ubuf__ float *gAddr = reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr());
    __ubuf__ float *eAddr = reinterpret_cast<__ubuf__ float *>(energy.GetPhyAddr());
    __ubuf__ float *rAddr = reinterpret_cast<__ubuf__ float *>(energyRecip.GetPhyAddr());
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
        RegTensor<float> one, reciprocal;
        Duplicate(one, 1.0f, all);
        Div(reciprocal, one, value, all);
        StoreAlign(rAddr, reciprocal, all);
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
// scratch only needs its first two floats and is reused by the following stage.
__aicore__ inline bool GateRatioSafe(LocalTensor<float> &g, LocalTensor<float> &scratch,
                                     uint32_t rows)
{
    __ubuf__ float *gAddr = reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr());
    __ubuf__ float *scratchAddr = reinterpret_cast<__ubuf__ float *>(scratch.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg mask = UpdateMask<float>(rows);
        MaskReg scalar = CreateMask<float, MaskPattern::VL1>();
        RegTensor<float> values, lo, hi;
        LoadAlign(values, gAddr);
        Reduce<Reg::ReduceType::MIN>(lo, values, mask);
        Reduce<Reg::ReduceType::MAX>(hi, values, mask);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scratchAddr, lo, scalar);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scratchAddr + 1, hi, scalar);
    }
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
    const float lo = scratch.GetValue(0);
    const float hi = scratch.GetValue(1);
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

// Store the same even/odd registers in the original ND row order.
__aicore__ inline void StoreNdEvenOdd(__ubuf__ bfloat16_t *dst, uint32_t row,
                                      RegTensor<float> &even, RegTensor<float> &odd,
                                      MaskReg &m16, MaskReg &m128)
{
    RegTensor<bfloat16_t> lo, hi;
    Cast<bfloat16_t, float, GdnFwd32::kCastB322B16>(lo, even, m16);
    Cast<bfloat16_t, float, GdnFwd32::kCastB322B16Hi>(hi, odd, m16);
    Or((RegTensor<uint16_t> &)lo, (RegTensor<uint16_t> &)lo, (RegTensor<uint16_t> &)hi, m16);
    StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(dst + row * 128, lo, 1, m128);
}

__aicore__ inline void StoreNzEvenOdd64(__ubuf__ bfloat16_t *nz, uint32_t row,
                                        RegTensor<float> &even, RegTensor<float> &odd,
                                        MaskReg &m16, MaskReg &m128)
{
    RegTensor<bfloat16_t> lo, hi;
    Cast<bfloat16_t, float, GdnFwd32::kCastB322B16>(lo, even, m16);
    Cast<bfloat16_t, float, GdnFwd32::kCastB322B16Hi>(hi, odd, m16);
    Or((RegTensor<uint16_t> &)lo, (RegTensor<uint16_t> &)lo, (RegTensor<uint16_t> &)hi, m16);
    StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(nz + row * 16, lo, kChunk, m128);
}

template <bool Normalize>
__aicore__ inline void NormalizeRowsNzVF(LocalTensor<bfloat16_t> &input,
                                         LocalTensor<bfloat16_t> &nzOutput,
                                         LocalTensor<float> &rstd)
{
    __ubuf__ bfloat16_t *src = reinterpret_cast<__ubuf__ bfloat16_t *>(input.GetPhyAddr());
    __ubuf__ bfloat16_t *nz = reinterpret_cast<__ubuf__ bfloat16_t *>(nzOutput.GetPhyAddr());
    __ubuf__ float *norm = reinterpret_cast<__ubuf__ float *>(rstd.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        MaskReg scalar = CreateMask<float, MaskPattern::VL1>();
        MaskReg m16 = CreateMask<uint16_t, MaskPattern::ALL>();
        uint32_t n128 = 128;
        MaskReg m128 = UpdateMask<bfloat16_t>(n128);
        RegTensor<float> even, odd, sum0, sum1, one, scale;
        Duplicate(one, 1.0f, all);
        for (uint16_t row = 0; row < kChunk; ++row) {
            GdnFwd32::LoadEvenOdd(src + static_cast<uint32_t>(row) * 128, even, odd, m16);
            if constexpr (Normalize) {
                Mul(sum0, even, even, all);
                Mul(sum1, odd, odd, all);
                Add(sum0, sum0, sum1, all);
                Reduce<Reg::ReduceType::SUM>(sum0, sum0, all);
                Adds(sum0, sum0, kSparseEps, scalar);
                Sqrt(sum0, sum0, scalar);
                Div(sum0, one, sum0, scalar);
                Duplicate(scale, sum0, all);
                Mul(even, even, scale, all);
                Mul(odd, odd, scale, all);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(norm + row, sum0, scalar);
            }
            StoreNzEvenOdd64(nz, row, even, odd, m16, m128);
        }
    }
}

// Two-row software pipeline shared by Q/QExp and K/Kbg. Independent reduction
// work from the next pair hides the current pair's Sqrt/Div latency.
#define GDN64_L2_REDUCE(addr, base, ee0, oo0, ee1, oo1, m0, m1)                                                        \
    GdnFwd32::LoadEvenOdd((addr) + static_cast<uint32_t>(base) * kPair, ee0, oo0, m16);                                         \
    GdnFwd32::LoadEvenOdd((addr) + static_cast<uint32_t>(base) * kPair + kRow, ee1, oo1, m16);                                  \
    Mul(s0, ee0, ee0, pregAll);                                                                                       \
    Mul(s1, oo0, oo0, pregAll);                                                                                       \
    Add(s0, s0, s1, pregAll);                                                                                         \
    Mul(s1, ee1, ee1, pregAll);                                                                                       \
    Mul(brc0, oo1, oo1, pregAll);                                                                                     \
    Add(s1, s1, brc0, pregAll);                                                                                       \
    Reduce<Reg::ReduceType::SUM>(m0, s0, pregAll);                                                                   \
    Reduce<Reg::ReduceType::SUM>(m1, s1, pregAll);                                                                   \
    Adds(m0, m0, kSparseEps, pregMerge);                                                                              \
    Adds(m1, m1, kSparseEps, pregMerge);

#define GDN64_L2_APPLY(ee0, oo0, ee1, oo1, m0, m1)                                                                    \
    Div(r0, one, m0, pregMerge);                                                                                      \
    Div(r1, one, m1, pregMerge);                                                                                      \
    Duplicate(brc0, r0, pregAll);                                                                                     \
    Duplicate(brc1, r1, pregAll);                                                                                     \
    Mul(ee0, ee0, brc0, pregAll);                                                                                     \
    Mul(oo0, oo0, brc0, pregAll);                                                                                     \
    Mul(ee1, ee1, brc1, pregAll);                                                                                     \
    Mul(oo1, oo1, brc1, pregAll);

#define GDN64_L2_ROWS(xAddr, nzAddr)                                                                                   \
    {                                                                                                                  \
        RegTensor<float> e0, o0, e1, o1, e2, o2, e3, o3, s0, s1, mean0, mean1, mean2, mean3, r0, r1, brc0, brc1;      \
        GDN64_L2_REDUCE(xAddr, 0, e0, o0, e1, o1, mean0, mean1)                                                        \
        for (uint16_t i = 0; i < nEven - 1; ++i) {                                                                     \
            Sqrt(mean0, mean0, pregMerge);                                                                             \
            Sqrt(mean1, mean1, pregMerge);                                                                             \
            GDN64_L2_REDUCE(xAddr, i + 1, e2, o2, e3, o3, mean2, mean3)                                                \
            GDN64_L2_APPLY(e0, o0, e1, o1, mean0, mean1)                                                               \
            GdnFwd32::StoreNzEvenOdd(nzAddr, static_cast<uint32_t>(i) * 2, e0, o0, m16, m128);                         \
            GdnFwd32::StoreNzEvenOdd(nzAddr, static_cast<uint32_t>(i) * 2 + 1, e1, o1, m16, m128);                     \
            GDN64_L2_TAIL                                                                                              \
            e0 = e2;                                                                                                   \
            o0 = o2;                                                                                                   \
            e1 = e3;                                                                                                   \
            o1 = o3;                                                                                                   \
            mean0 = mean2;                                                                                             \
            mean1 = mean3;                                                                                             \
        }                                                                                                              \
        {                                                                                                              \
            uint16_t i = static_cast<uint16_t>(nEven - 1);                                                             \
            Sqrt(mean0, mean0, pregMerge);                                                                             \
            Sqrt(mean1, mean1, pregMerge);                                                                             \
            GDN64_L2_APPLY(e0, o0, e1, o1, mean0, mean1)                                                               \
            GdnFwd32::StoreNzEvenOdd(nzAddr, static_cast<uint32_t>(i) * 2, e0, o0, m16, m128);                         \
            GdnFwd32::StoreNzEvenOdd(nzAddr, static_cast<uint32_t>(i) * 2 + 1, e1, o1, m16, m128);                     \
            GDN64_L2_TAIL                                                                                              \
        }                                                                                                              \
    }

__aicore__ inline void QNormExpNzVF(LocalTensor<bfloat16_t> &q,
                                    LocalTensor<bfloat16_t> &qNz,
                                    LocalTensor<bfloat16_t> &qExpNz,
                                    LocalTensor<float> &energy,
                                    uint32_t firstRow)
{
    constexpr uint32_t kRow = 128;
    constexpr uint32_t kPair = 256;
    __ubuf__ bfloat16_t *qAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr());
    __ubuf__ bfloat16_t *nzAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(qNz.GetPhyAddr());
    __ubuf__ bfloat16_t *qeAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(qExpNz.GetPhyAddr());
    __ubuf__ float *eAddr = reinterpret_cast<__ubuf__ float *>(energy.GetPhyAddr());
    const uint16_t nEven = static_cast<uint16_t>(kHalfRows >> 1);
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        MaskReg m16 = CreateMask<uint16_t, MaskPattern::ALL>();
        uint32_t n128 = 128;
        MaskReg m128 = UpdateMask<bfloat16_t>(n128);
        RegTensor<float> one;
        Duplicate(one, 1.0f, pregMerge);
#define GDN64_L2_TAIL                                                                                                  \
    {                                                                                                                  \
        RegTensor<float> g0, g1;                                                                                       \
        LoadAlign<float, LoadDist::DIST_BRC_B32>(g0, eAddr + firstRow + i * 2);                                       \
        LoadAlign<float, LoadDist::DIST_BRC_B32>(g1, eAddr + firstRow + i * 2 + 1);                                   \
        Mul(e0, e0, g0, pregAll);                                                                                     \
        Mul(o0, o0, g0, pregAll);                                                                                     \
        Mul(e1, e1, g1, pregAll);                                                                                     \
        Mul(o1, o1, g1, pregAll);                                                                                     \
        GdnFwd32::StoreNzEvenOdd(qeAddr, static_cast<uint32_t>(i) * 2, e0, o0, m16, m128);                             \
        GdnFwd32::StoreNzEvenOdd(qeAddr, static_cast<uint32_t>(i) * 2 + 1, e1, o1, m16, m128);                         \
    }
        GDN64_L2_ROWS(qAddr, nzAddr)
#undef GDN64_L2_TAIL
    }
}

__aicore__ inline void KNormKbgNzVF(LocalTensor<bfloat16_t> &k,
                                    LocalTensor<bfloat16_t> &kNz,
                                    LocalTensor<bfloat16_t> &kbgNz,
                                    LocalTensor<float> &beta,
                                    LocalTensor<float> &energy,
                                    uint32_t firstRow)
{
    constexpr uint32_t kRow = 128;
    constexpr uint32_t kPair = 256;
    __ubuf__ bfloat16_t *kAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr());
    __ubuf__ bfloat16_t *nzAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(kNz.GetPhyAddr());
    __ubuf__ bfloat16_t *kbgAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(kbgNz.GetPhyAddr());
    __ubuf__ float *bAddr = reinterpret_cast<__ubuf__ float *>(beta.GetPhyAddr());
    __ubuf__ float *eAddr = reinterpret_cast<__ubuf__ float *>(energy.GetPhyAddr());
    const uint16_t nEven = static_cast<uint16_t>(kHalfRows >> 1);
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        MaskReg m16 = CreateMask<uint16_t, MaskPattern::ALL>();
        uint32_t n128 = 128;
        MaskReg m128 = UpdateMask<bfloat16_t>(n128);
        RegTensor<float> one;
        Duplicate(one, 1.0f, pregMerge);
#define GDN64_L2_TAIL                                                                                                  \
    {                                                                                                                  \
        RegTensor<float> g0, g1, be0, be1, sc0, sc1;                                                                  \
        LoadAlign<float, LoadDist::DIST_BRC_B32>(g0, eAddr + firstRow + i * 2);                                       \
        LoadAlign<float, LoadDist::DIST_BRC_B32>(g1, eAddr + firstRow + i * 2 + 1);                                   \
        LoadAlign<float, LoadDist::DIST_BRC_B32>(be0, bAddr + firstRow + i * 2);                                      \
        LoadAlign<float, LoadDist::DIST_BRC_B32>(be1, bAddr + firstRow + i * 2 + 1);                                  \
        Mul(sc0, be0, g0, pregAll);                                                                                   \
        Mul(sc1, be1, g1, pregAll);                                                                                   \
        Muls(sc0, sc0, -1.0f, pregAll);                                                                               \
        Muls(sc1, sc1, -1.0f, pregAll);                                                                               \
        Mul(e0, e0, sc0, pregAll);                                                                                    \
        Mul(o0, o0, sc0, pregAll);                                                                                    \
        Mul(e1, e1, sc1, pregAll);                                                                                    \
        Mul(o1, o1, sc1, pregAll);                                                                                    \
        GdnFwd32::StoreNzEvenOdd(kbgAddr, static_cast<uint32_t>(i) * 2, e0, o0, m16, m128);                            \
        GdnFwd32::StoreNzEvenOdd(kbgAddr, static_cast<uint32_t>(i) * 2 + 1, e1, o1, m16, m128);                        \
    }
        GDN64_L2_ROWS(kAddr, nzAddr)
#undef GDN64_L2_TAIL
    }
}

#undef GDN64_L2_ROWS
#undef GDN64_L2_APPLY
#undef GDN64_L2_REDUCE

__aicore__ inline void ScaleVbNzVF(LocalTensor<bfloat16_t> &v,
                                   LocalTensor<bfloat16_t> &out,
                                   LocalTensor<float> &beta,
                                   uint32_t firstRow)
{
    __ubuf__ bfloat16_t *src = reinterpret_cast<__ubuf__ bfloat16_t *>(v.GetPhyAddr());
    __ubuf__ bfloat16_t *nz = reinterpret_cast<__ubuf__ bfloat16_t *>(out.GetPhyAddr());
    __ubuf__ float *b = reinterpret_cast<__ubuf__ float *>(beta.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        MaskReg m16 = CreateMask<uint16_t, MaskPattern::ALL>();
        uint32_t n128 = 128;
        MaskReg m128 = UpdateMask<bfloat16_t>(n128);
        RegTensor<float> even, odd, scale;
        for (uint16_t row = 0; row < 32; ++row) {
            GdnFwd32::LoadEvenOdd(src + static_cast<uint32_t>(row) * 128, even, odd, m16);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scale, b + firstRow + row);
            Mul(even, even, scale, all);
            Mul(odd, odd, scale, all);
            GdnFwd32::StoreNzEvenOdd(nz, row, even, odd, m16, m128);
        }
    }
}

__aicore__ inline void ScaleKbgNzVF(LocalTensor<bfloat16_t> &k,
                                    LocalTensor<bfloat16_t> &out,
                                    LocalTensor<float> &beta,
                                    LocalTensor<float> &energy,
                                    uint32_t firstRow, bool negate = false)
{
    __ubuf__ bfloat16_t *src = reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr());
    __ubuf__ bfloat16_t *nz = reinterpret_cast<__ubuf__ bfloat16_t *>(out.GetPhyAddr());
    __ubuf__ float *b = reinterpret_cast<__ubuf__ float *>(beta.GetPhyAddr());
    __ubuf__ float *e = reinterpret_cast<__ubuf__ float *>(energy.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        MaskReg m16 = CreateMask<uint16_t, MaskPattern::ALL>();
        uint32_t n128 = 128;
        MaskReg m128 = UpdateMask<bfloat16_t>(n128);
        RegTensor<float> even, odd, scale, be;
        for (uint16_t row = 0; row < 32; ++row) {
            GdnFwd32::LoadEvenOdd(src + static_cast<uint32_t>(row) * 128, even, odd, m16);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(be, b + firstRow + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scale, e + firstRow + row);
            Mul(scale, scale, be, all);
            Mul(even, even, scale, all);
            Mul(odd, odd, scale, all);
            if (negate) {
                Muls(even, even, -1.0f, all);
                Muls(odd, odd, -1.0f, all);
            }
            GdnFwd32::StoreNzEvenOdd(nz, row, even, odd, m16, m128);
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

// Build the full 64x64 pairwise gate-ratio matrix once per chunk. Inverse and
// QKMask consume the same FP32 ratios before the next pair overwrites the slot.
template <bool UseExp2>
__aicore__ inline void GateRatioVF(LocalTensor<float> &ratio,
                                   LocalTensor<float> &g,
                                   LocalTensor<float> &energy,
                                   LocalTensor<float> &energyRecip,
                                   bool useEnergyRatio)
{
    __ubuf__ float *r = reinterpret_cast<__ubuf__ float *>(ratio.GetPhyAddr());
    __ubuf__ float *gAddr = reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr());
    __ubuf__ float *eAddr = reinterpret_cast<__ubuf__ float *>(energy.GetPhyAddr());
    __ubuf__ float *rcpAddr = reinterpret_cast<__ubuf__ float *>(energyRecip.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> invJ, gI, diff, value;
        if (useEnergyRatio) {
            LoadAlign(invJ, rcpAddr);
        } else {
            LoadAlign(invJ, gAddr);
        }
        for (uint16_t row = 0; row < kChunk; ++row) {
            if (useEnergyRatio) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(gI, eAddr + row);
                Mul(value, gI, invJ, all);
            } else {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(gI, gAddr + row);
                Sub(diff, gI, invJ, all);
                if constexpr (UseExp2) {
                    Muls(diff, diff, kSparseLn2, all);
                }
                Exp(value, diff, all);
            }
            StoreAlign(r + static_cast<uint32_t>(row) * 64, value, all);
        }
    }
}

// A=(I-L)^-1 for one complete 64x64 chunk. The owner AIV performs forward
// substitution directly on 64-lane vectors; no diagonal split or Cube coupling
// is involved. X starts as identity and each group reuses all earlier X rows.
__aicore__ inline void Inverse64VF(LocalTensor<float> &kkt,
                                   LocalTensor<float> &lower,
                                   LocalTensor<float> &a,
                                   LocalTensor<float> &ratio,
                                   LocalTensor<float> &beta,
                                   LocalTensor<float> &eye)
{
    __ubuf__ float *kk = reinterpret_cast<__ubuf__ float *>(kkt.GetPhyAddr());
    __ubuf__ float *l = reinterpret_cast<__ubuf__ float *>(lower.GetPhyAddr());
    __ubuf__ float *x = reinterpret_cast<__ubuf__ float *>(a.GetPhyAddr());
    __ubuf__ float *r = reinterpret_cast<__ubuf__ float *>(ratio.GetPhyAddr());
    __ubuf__ float *b = reinterpret_cast<__ubuf__ float *>(beta.GetPhyAddr());
    __ubuf__ float *eyeAddr = reinterpret_cast<__ubuf__ float *>(eye.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> bi, gate, score, zero, expLo, expHi;
        Duplicate(zero, 0.0f, all);
        Duplicate(expLo, 1.92874985e-22f, all);
        Duplicate(expHi, 5.18470553e21f, all);
        for (uint16_t row = 0; row < kChunk; ++row) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(bi, b + row);
            LoadAlign(gate, r + static_cast<uint32_t>(row) * 64);
            Max(gate, gate, expLo, all);
            Min(gate, gate, expHi, all);
            Mul(gate, gate, bi, all);
            Muls(gate, gate, -1.0f, all);
            LoadAlign(score, kk + static_cast<uint32_t>(row) * 64);
            Mul(gate, gate, score, all);
            uint32_t lowerCount = row;
            MaskReg lowerMask = UpdateMask<float>(lowerCount);
            Select(gate, gate, zero, lowerMask);
            StoreAlign(l + static_cast<uint32_t>(row) * 64, gate, all);
        }
    }
    PipeBarrier<PIPE_V>();
#define SPARSE_SOLVE_PREV(dst, rr)                                                                                     \
    LoadAlign<float, LoadDist::DIST_BRC_B32>(coefficient,                                                             \
        l + (base + (rr)) * 64 + col);                                                                                 \
    MulAddDst(dst, previous, coefficient, all);
#define SPARSE_SOLVE_LOCAL(dst, src, rr, cc)                                                                           \
    LoadAlign<float, LoadDist::DIST_BRC_B32>(coefficient,                                                             \
        l + (base + (rr)) * 64 + base + (cc));                                                                         \
    MulAddDst(dst, src, coefficient, all);
    for (uint16_t group = 0; group < 8; ++group) {
        const uint32_t base = static_cast<uint32_t>(group) * 8;
        __VEC_SCOPE__
        {
            MaskReg all = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> y0, y1, y2, y3, y4, y5, y6, y7;
            RegTensor<float> previous, coefficient;
            LoadAlign(y0, eyeAddr + (base + 0) * 64);
            LoadAlign(y1, eyeAddr + (base + 1) * 64);
            LoadAlign(y2, eyeAddr + (base + 2) * 64);
            LoadAlign(y3, eyeAddr + (base + 3) * 64);
            LoadAlign(y4, eyeAddr + (base + 4) * 64);
            LoadAlign(y5, eyeAddr + (base + 5) * 64);
            LoadAlign(y6, eyeAddr + (base + 6) * 64);
            LoadAlign(y7, eyeAddr + (base + 7) * 64);
            for (uint16_t col = 0; col < base; ++col) {
                LoadAlign(previous, x + static_cast<uint32_t>(col) * 64);
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
            StoreAlign(x + (base + 0) * 64, y0, all);
            StoreAlign(x + (base + 1) * 64, y1, all);
            StoreAlign(x + (base + 2) * 64, y2, all);
            StoreAlign(x + (base + 3) * 64, y3, all);
            StoreAlign(x + (base + 4) * 64, y4, all);
            StoreAlign(x + (base + 5) * 64, y5, all);
            StoreAlign(x + (base + 6) * 64, y6, all);
            StoreAlign(x + (base + 7) * 64, y7, all);
        }
        PipeBarrier<PIPE_V>();
    }
#undef SPARSE_SOLVE_PREV
#undef SPARSE_SOLVE_LOCAL
}

// Build the two full-chunk Neumann inputs for one AIV's 32-row band.
// P is the strict-lower L in (I-L)^-1; S starts at I+L. The inverse itself is
// evaluated entirely by full 64x64 Cube GEMMs in the producer.
template <bool UseExp2>
__aicore__ inline void BuildNeumannInputVF(LocalTensor<float> &kkt,
                                           LocalTensor<bfloat16_t> &p,
                                           LocalTensor<bfloat16_t> &s,
                                           LocalTensor<float> &g,
                                           LocalTensor<float> &beta,
                                           LocalTensor<float> &energy,
                                           bool useEnergyRatio,
                                           uint32_t firstRow)
{
    __ubuf__ float *kk = reinterpret_cast<__ubuf__ float *>(kkt.GetPhyAddr());
    __ubuf__ bfloat16_t *pAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(p.GetPhyAddr());
    __ubuf__ bfloat16_t *sAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(s.GetPhyAddr());
    __ubuf__ float *gAddr = reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr());
    __ubuf__ float *b = reinterpret_cast<__ubuf__ float *>(beta.GetPhyAddr());
    __ubuf__ float *eAddr = reinterpret_cast<__ubuf__ float *>(energy.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> gJ, gI, bi, diff, gate, score, sum, zero, one;
        RegTensor<int32_t> lanes;
        Duplicate(zero, 0.0f, all);
        Duplicate(one, 1.0f, all);
        Arange<int32_t, IndexOrder::INCREASE_ORDER>(lanes, 0);
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
            } else {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(gI, gAddr + globalRow);
                Sub(diff, gI, gJ, all);
                if constexpr (UseExp2) {
                    Muls(diff, diff, kSparseLn2, all);
                }
                Exp(gate, diff, all);
            }
            Mul(gate, gate, bi, all);
            Muls(gate, gate, -1.0f, all);
            LoadAlign(score, kk + static_cast<uint32_t>(row) * 64);
            Mul(gate, gate, score, all);
            uint32_t lowerCount = globalRow;
            MaskReg lower = UpdateMask<float>(lowerCount);
            Select(gate, gate, zero, lower);
            StoreBf16(pAddr + static_cast<uint32_t>(row) * 64, gate, all);
            MaskReg diagonal;
            CompareScalar<int32_t, CMPMODE::EQ>(diagonal, lanes, static_cast<int32_t>(globalRow), all);
            Select(sum, one, gate, diagonal);
            StoreBf16(sAddr + static_cast<uint32_t>(row) * 64, sum, all);
        }
    }
}

__aicore__ inline void APrimeNzVF(LocalTensor<float> &qk,
                                  LocalTensor<bfloat16_t> &out,
                                  LocalTensor<float> &ratio)
{
    __ubuf__ float *score = reinterpret_cast<__ubuf__ float *>(qk.GetPhyAddr());
    __ubuf__ bfloat16_t *nz = reinterpret_cast<__ubuf__ bfloat16_t *>(out.GetPhyAddr());
    __ubuf__ float *r = reinterpret_cast<__ubuf__ float *>(ratio.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        uint32_t n64 = 64;
        MaskReg m64 = UpdateMask<bfloat16_t>(n64);
        RegTensor<float> gate, raw, zero;
        Duplicate(zero, 0.0f, all);
        for (uint16_t row = 0; row < kChunk; ++row) {
            LoadAlign(raw, score + static_cast<uint32_t>(row) * 64);
            LoadAlign(gate, r + static_cast<uint32_t>(row) * 64);
            Mul(gate, gate, raw, all);
            uint32_t lowerCount = row + 1U;
            MaskReg lower = UpdateMask<float>(lowerCount);
            Select(gate, gate, zero, lower);
            StoreBf16Nz64(nz, row, gate, all, m64, kChunk);
        }
    }
}

__aicore__ inline void QExpNzVF(LocalTensor<bfloat16_t> &q,
                                LocalTensor<bfloat16_t> &out,
                                LocalTensor<float> &energy,
                                uint32_t firstRow)
{
    __ubuf__ bfloat16_t *src = reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr());
    __ubuf__ bfloat16_t *nz = reinterpret_cast<__ubuf__ bfloat16_t *>(out.GetPhyAddr());
    __ubuf__ float *e = reinterpret_cast<__ubuf__ float *>(energy.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        MaskReg m16 = CreateMask<uint16_t, MaskPattern::ALL>();
        uint32_t n128 = 128;
        MaskReg m128 = UpdateMask<bfloat16_t>(n128);
        RegTensor<float> even, odd, scale;
        for (uint16_t row = 0; row < 32; ++row) {
            GdnFwd32::LoadEvenOdd(src + static_cast<uint32_t>(row) * 128, even, odd, m16);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scale, e + firstRow + row);
            Mul(even, even, scale, all);
            Mul(odd, odd, scale, all);
            GdnFwd32::StoreNzEvenOdd(nz, row, even, odd, m16, m128);
        }
    }
}

template <bool UseExp2>
__aicore__ inline void DecayVF(LocalTensor<float> &decay,
                               LocalTensor<float> &g,
                               LocalTensor<float> &energyRecip,
                               float gLast, float lastEnergy,
                               bool useEnergyRatio, uint32_t firstRow)
{
    __ubuf__ float *dAddr = reinterpret_cast<__ubuf__ float *>(decay.GetPhyAddr());
    __ubuf__ float *gAddr = reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr());
    __ubuf__ float *rAddr = reinterpret_cast<__ubuf__ float *>(energyRecip.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        MaskReg scalar = CreateMask<float, MaskPattern::VL1>();
        RegTensor<float> scale, last, diff;
        Duplicate(last, useEnergyRatio ? lastEnergy : gLast, all);
        for (uint16_t row = 0; row < 32; ++row) {
            if (useEnergyRatio) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(scale, rAddr + firstRow + row);
                Mul(scale, last, scale, all);
            } else {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(scale, gAddr + firstRow + row);
                Sub(diff, last, scale, all);
                if constexpr (UseExp2) {
                    Muls(diff, diff, kSparseLn2, all);
                }
                Exp(scale, diff, all);
            }
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dAddr + firstRow + row, scale, scalar);
        }
    }
}

__aicore__ inline void RightNzVF(LocalTensor<bfloat16_t> &u,
                                 LocalTensor<bfloat16_t> &right,
                                 LocalTensor<float> &decay,
                                 uint32_t firstRow, bool directNz)
{
    __ubuf__ bfloat16_t *uAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(u.GetPhyAddr());
    __ubuf__ bfloat16_t *rAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(right.GetPhyAddr());
    __ubuf__ float *dAddr = reinterpret_cast<__ubuf__ float *>(decay.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        MaskReg m16 = CreateMask<uint16_t, MaskPattern::ALL>();
        uint32_t n128 = 128;
        MaskReg m128 = UpdateMask<bfloat16_t>(n128);
        RegTensor<float> u0, u1, scale;
        for (uint16_t row = 0; row < 32; ++row) {
            const uint32_t offset = static_cast<uint32_t>(row) * 128;
            GdnFwd32::LoadEvenOdd(uAddr + offset, u0, u1, m16);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scale, dAddr + firstRow + row);
            Mul(u0, u0, scale, all);
            Mul(u1, u1, scale, all);
            if (directNz) {
                GdnFwd32::StoreNzEvenOdd(rAddr, row, u0, u1, m16, m128);
            } else {
                StoreNdEvenOdd(rAddr, row, u0, u1, m16, m128);
            }
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

__aicore__ inline void StateDecayVF(LocalTensor<float> &state, float decay)
{
    __ubuf__ float *s = reinterpret_cast<__ubuf__ float *>(state.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> s0, s1, alpha;
        Duplicate(alpha, decay, all);
        for (uint16_t row = 0; row < 64; ++row) {
            const uint32_t offset = static_cast<uint32_t>(row) * 128;
            LoadAlign(s0, s + offset);
            LoadAlign(s1, s + offset + 64);
            Mul(s0, s0, alpha, all);
            Mul(s1, s1, alpha, all);
            StoreAlign(s + offset, s0, all);
            StoreAlign(s + offset + 64, s1, all);
        }
    }
}

__aicore__ inline void StateAddDVF(LocalTensor<float> &state,
                                   LocalTensor<bfloat16_t> &stateBf,
                                   LocalTensor<float> &d)
{
    __ubuf__ float *s = reinterpret_cast<__ubuf__ float *>(state.GetPhyAddr());
    __ubuf__ bfloat16_t *b = reinterpret_cast<__ubuf__ bfloat16_t *>(stateBf.GetPhyAddr());
    __ubuf__ float *dd = reinterpret_cast<__ubuf__ float *>(d.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        uint32_t n64 = 64;
        MaskReg m64 = UpdateMask<bfloat16_t>(n64);
        RegTensor<float> s0, s1, d0, d1;
        for (uint16_t row = 0; row < 64; ++row) {
            const uint32_t offset = static_cast<uint32_t>(row) * 128;
            LoadAlign(s0, s + offset);
            LoadAlign(s1, s + offset + 64);
            LoadAlign(d0, dd + offset);
            LoadAlign(d1, dd + offset + 64);
            Add(s0, s0, d0, all);
            Add(s1, s1, d1, all);
            StoreAlign(s + offset, s0, all);
            StoreAlign(s + offset + 64, s1, all);
            StoreBf16Nz128(b, row, s0, s1, all, m64, static_cast<uint32_t>(kHalfDim));
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
        uint32_t n64 = 64;
        MaskReg m64 = UpdateMask<bfloat16_t>(n64);
        RegTensor<float> v0, v1;
        for (uint16_t row = 0; row < 64; ++row) {
            const uint32_t offset = static_cast<uint32_t>(row) * 128;
            LoadBf16(v0, src + offset, all);
            LoadBf16(v1, src + offset + 64, all);
            StoreAlign(s + offset, v0, all);
            StoreAlign(s + offset + 64, v1, all);
            StoreBf16Nz128(b, row, v0, v1, all, m64, static_cast<uint32_t>(kHalfDim));
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
        uint32_t n64 = 64;
        MaskReg m64 = UpdateMask<bfloat16_t>(n64);
        RegTensor<float> v0, v1;
        for (uint16_t row = 0; row < 64; ++row) {
            const uint32_t offset = static_cast<uint32_t>(row) * 128;
            LoadAlign(v0, s + offset);
            LoadAlign(v1, s + offset + 64);
            StoreBf16Nz128(b, row, v0, v1, all, m64, static_cast<uint32_t>(kHalfDim));
        }
    }
}

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_VECTOR_H
