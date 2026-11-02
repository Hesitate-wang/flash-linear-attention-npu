/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_CHUNK64_SPARSE_MATH_H
#define GDN_CHUNK64_SPARSE_MATH_H

#include "kernel_operator.h"

namespace GdnChunk64Sparse {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr int32_t VL = 64;

constexpr CastTrait kCastB162B32 = {
    RegLayout::ZERO,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};
constexpr CastTrait kCastB322B16 = {
    RegLayout::ZERO,
    SatMode::NO_SAT,
    MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

template <typename T>
__aicore__ inline void LoadCastB16(__ubuf__ T *src, RegTensor<float> &dst, MaskReg &mask)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign(dst, src);
    } else {
        RegTensor<T> xB16;
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(xB16, src);
        Cast<float, T, kCastB162B32>(dst, xB16, mask);
    }
}

template <typename T>
__aicore__ inline void StoreCastB16(__ubuf__ T *dst, RegTensor<float> &src, MaskReg &mask)
{
    if constexpr (IsSameType<T, float>::value) {
        StoreAlign(dst, src, mask);
    } else {
        RegTensor<T> yB16;
        Cast<T, float, kCastB322B16>(yB16, src, mask);
        StoreAlign<T, StoreDist::DIST_PACK_B32>(dst, yB16, mask);
    }
}

template <typename T>
__aicore__ inline void LoadTwoVLFp32(__ubuf__ T *addr, uint32_t offset0, uint32_t offset1,
                                     RegTensor<float> &a, RegTensor<float> &b, MaskReg &pregLoop)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> aB16, bB16;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(aB16, addr + offset0);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(bB16, addr + offset1);
        Cast<float, half, kCastB162B32>(a, aB16, pregLoop);
        Cast<float, half, kCastB162B32>(b, bB16, pregLoop);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> aB16, bB16;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(aB16, addr + offset0);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(bB16, addr + offset1);
        Cast<float, bfloat16_t, kCastB162B32>(a, aB16, pregLoop);
        Cast<float, bfloat16_t, kCastB162B32>(b, bB16, pregLoop);
    } else {
        LoadAlign(a, addr + offset0);
        LoadAlign(b, addr + offset1);
    }
}

// Row-wise L2 normalization for K=128, preserving Prepare's cast and rounding order.
template <typename T>
__aicore__ inline void L2NormK128VF(LocalTensor<T> &xLocal, LocalTensor<T> &yLocal,
                                    LocalTensor<float> &rstdLocal, uint32_t curRows,
                                    float eps = 1e-6f)
{
    constexpr uint32_t K = 2 * VL;
    constexpr uint32_t kPair = 2 * K;
    __ubuf__ T *xAddr = reinterpret_cast<__ubuf__ T *>(xLocal.GetPhyAddr());
    __ubuf__ T *yAddr = reinterpret_cast<__ubuf__ T *>(yLocal.GetPhyAddr());
    __ubuf__ float *rstdAddr = reinterpret_cast<__ubuf__ float *>(rstdLocal.GetPhyAddr());
    const uint16_t nRows = static_cast<uint16_t>(curRows);
    const uint16_t nEven = nRows >> 1;
    const uint16_t nTail = nRows & 1;

    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        RegTensor<float> one;
        Duplicate(one, 1.0f, pregMerge);

        for (uint16_t i = 0; i < nEven; i++) {
            RegTensor<float> x0a, x0b, x1a, x1b, s0a, s0b, s1a, s1b;
            RegTensor<float> mean0, mean1, r0, r1, brc0, brc1, y0a, y0b, y1a, y1b;
            LoadTwoVLFp32<T>(xAddr, i * kPair, i * kPair + VL, x0a, x0b, pregAll);
            LoadTwoVLFp32<T>(xAddr, i * kPair + K, i * kPair + K + VL, x1a, x1b, pregAll);
            Mul(s0a, x0a, x0a, pregAll);
            Mul(s0b, x0b, x0b, pregAll);
            Mul(s1a, x1a, x1a, pregAll);
            Mul(s1b, x1b, x1b, pregAll);
            Add(s0a, s0a, s0b, pregAll);
            Add(s1a, s1a, s1b, pregAll);
            Reduce<AscendC::Reg::ReduceType::SUM>(mean0, s0a, pregAll);
            Reduce<AscendC::Reg::ReduceType::SUM>(mean1, s1a, pregAll);
            Adds(mean0, mean0, eps, pregMerge);
            Adds(mean1, mean1, eps, pregMerge);
            Sqrt(mean0, mean0, pregMerge);
            Sqrt(mean1, mean1, pregMerge);
            Div(r0, one, mean0, pregMerge);
            Div(r1, one, mean1, pregMerge);
            Duplicate(brc0, r0, pregAll);
            Duplicate(brc1, r1, pregAll);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr + i * 2, r0, pregMerge);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr + i * 2 + 1, r1, pregMerge);
            Mul(y0a, x0a, brc0, pregAll);
            Mul(y0b, x0b, brc0, pregAll);
            Mul(y1a, x1a, brc1, pregAll);
            Mul(y1b, x1b, brc1, pregAll);
            StoreCastB16<T>(yAddr + i * kPair, y0a, pregAll);
            StoreCastB16<T>(yAddr + i * kPair + VL, y0b, pregAll);
            StoreCastB16<T>(yAddr + i * kPair + K, y1a, pregAll);
            StoreCastB16<T>(yAddr + i * kPair + K + VL, y1b, pregAll);
        }
        for (uint16_t t = 0; t < nTail; t++) {
            const uint32_t row = static_cast<uint32_t>(nEven) * 2 + t;
            RegTensor<float> xa, xb, sa, sb, vMean, rstdReg, brc, y0, y1;
            LoadTwoVLFp32<T>(xAddr, row * K, row * K + VL, xa, xb, pregAll);
            Mul(sa, xa, xa, pregAll);
            Mul(sb, xb, xb, pregAll);
            Add(sa, sa, sb, pregAll);
            Reduce<AscendC::Reg::ReduceType::SUM>(vMean, sa, pregAll);
            Adds(vMean, vMean, eps, pregMerge);
            Sqrt(vMean, vMean, pregMerge);
            Div(rstdReg, one, vMean, pregMerge);
            Duplicate(brc, rstdReg, pregAll);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr + row, rstdReg, pregMerge);
            Mul(y0, xa, brc, pregAll);
            Mul(y1, xb, brc, pregAll);
            StoreCastB16<T>(yAddr + row * K, y0, pregAll);
            StoreCastB16<T>(yAddr + row * K + VL, y1, pregAll);
        }
    }
}

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_MATH_H
