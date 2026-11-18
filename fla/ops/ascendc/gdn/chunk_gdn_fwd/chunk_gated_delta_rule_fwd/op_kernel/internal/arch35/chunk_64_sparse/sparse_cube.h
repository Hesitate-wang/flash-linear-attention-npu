/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_CHUNK64_SPARSE_CUBE_H
#define GDN_CHUNK64_SPARSE_CUBE_H

#include "sparse_common.h"

namespace GdnChunk64Sparse {

constexpr FixpipeConfig kSparseFixUb = {CO2Layout::ROW_MAJOR, true};
constexpr FixpipeConfig kSparseFixUbNz = {CO2Layout::NZ, true};
constexpr FixpipeConfig kSparseFixL1 = {CO2Layout::NZ, false};
constexpr FixpipeConfig kSparseFixGm = {CO2Layout::ROW_MAJOR, false};

__aicore__ inline void LoadA(LocalTensor<bfloat16_t> l0, LocalTensor<bfloat16_t> l1,
                             int32_t m, int32_t k, bool transpose)
{
    LoadData2DParamsV2 p{};
    p.mStep = static_cast<uint16_t>((transpose ? k : m) / 16);
    p.kStep = static_cast<uint16_t>((transpose ? m : k) / 16);
    p.srcStride = static_cast<uint16_t>((transpose ? k : m) / 16);
    p.dstStride = static_cast<uint16_t>(m / 16);
    p.ifTranspose = transpose;
    p.sid = 0;
    LoadData(l0, l1, p);
}

__aicore__ inline void LoadB(LocalTensor<bfloat16_t> l0, LocalTensor<bfloat16_t> l1,
                             int32_t n, int32_t k, bool transpose)
{
    LoadData2DParamsV2 p{};
    p.mStep = static_cast<uint16_t>((transpose ? k : n) / 16);
    p.kStep = static_cast<uint16_t>((transpose ? n : k) / 16);
    p.srcStride = p.mStep;
    p.dstStride = static_cast<uint16_t>(transpose ? n / 16 : p.mStep);
    p.ifTranspose = transpose;
    p.sid = 0;
    LoadData(l0, l1, p);
}

// Pure Matrix issue. Operand readiness and releases are orchestrated by the
// Producer/Consumer pipeline around this primitive.
__aicore__ inline void IssueMm(LocalTensor<bfloat16_t> l0A,
                               LocalTensor<bfloat16_t> l0B,
                               LocalTensor<float> l0C,
                               int32_t m, int32_t n, int32_t k, bool initC = true)
{
    MmadParams p{};
    p.m = m;
    p.n = n;
    p.k = k;
    p.cmatrixInitVal = initC;
    p.cmatrixSource = false;
    p.unitFlag = 0;
    Mmad(l0C, l0A, l0B, p);
}

__aicore__ inline void FixB16L1(LocalTensor<bfloat16_t> l1, LocalTensor<float> l0C)
{
    FixpipeParamsArch3510<CO2Layout::NZ> p{};
    p.nSize = 32;
    p.mSize = 32;
    p.srcStride = 32;
    p.dstStride = 64 * 16;
    p.quantPre = QuantMode_t::F322BF16;
    p.isChannelSplit = false;
    p.unitFlag = 0;
    Fixpipe<bfloat16_t, float, kSparseFixL1>(l1, l0C, p);
}

// Full BF16 Neumann matrix in L1. L0C keeps the FP32 accumulator live so the
// next S update can accumulate with initC=false after this store retires.
__aicore__ inline void FixB16L1Matrix(LocalTensor<bfloat16_t> l1, LocalTensor<float> l0C,
                                      uint32_t m, uint32_t n, uint32_t totalRows)
{
    FixpipeParamsArch3510<CO2Layout::NZ> p{};
    p.nSize = static_cast<uint16_t>(n);
    p.mSize = static_cast<uint16_t>(m);
    p.srcStride = static_cast<uint16_t>(m);
    p.dstStride = totalRows * 16;
    p.quantPre = QuantMode_t::F322BF16;
    p.isChannelSplit = false;
    p.unitFlag = 0;
    Fixpipe<bfloat16_t, float, kSparseFixL1>(l1, l0C, p);
}

// AIC direct ND output for public A/W/U. Only valid rows are written for tails.
__aicore__ inline void FixB16GmNd(GlobalTensor<bfloat16_t> gm, LocalTensor<float> l0C,
                                  uint32_t rows, uint32_t cols, uint32_t dstStride,
                                  uint32_t srcStride = 64)
{
    if (rows == 0) {
        return;
    }
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> p{};
    p.nSize = static_cast<uint16_t>(cols);
    p.mSize = static_cast<uint16_t>(rows);
    p.srcStride = static_cast<uint16_t>(srcStride);
    p.dstStride = dstStride;
    p.quantPre = QuantMode_t::F322BF16;
    p.dualDstCtl = 0;
    p.subBlockId = 0;
    p.isChannelSplit = false;
    p.unitFlag = 0;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    Fixpipe<bfloat16_t, float, kSparseFixGm>(gm, l0C, p);
}

__aicore__ inline void FixF32(LocalTensor<float> ub, LocalTensor<float> l0C,
                              uint32_t m, uint32_t n, bool aiv1 = false)
{
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> p{};
    p.nSize = static_cast<uint16_t>(n);
    p.mSize = static_cast<uint16_t>(m);
    p.srcStride = static_cast<uint16_t>(m);
    p.dstStride = n;
    p.quantPre = QuantMode_t::NoQuant;
    p.dualDstCtl = 0;
    p.isChannelSplit = false;
    p.params.ndNum = 1;
    p.subBlockId = aiv1;
    Fixpipe<float, float, kSparseFixUb>(ub, l0C, p);
}

__aicore__ inline void FixF32Halves(LocalTensor<float> ub, LocalTensor<float> l0C,
                                    uint32_t fullM, uint32_t n)
{
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> p{};
    p.nSize = static_cast<uint16_t>(n);
    p.mSize = static_cast<uint16_t>(fullM / 2);
    p.srcStride = static_cast<uint16_t>(fullM);
    p.dstStride = n;
    p.quantPre = QuantMode_t::NoQuant;
    p.dualDstCtl = 0;
    p.isChannelSplit = false;
    p.params.ndNum = 1;
    p.subBlockId = false;
    Fixpipe<float, float, kSparseFixUb>(ub, l0C, p);
    p.subBlockId = true;
    Fixpipe<float, float, kSparseFixUb>(ub, l0C[(fullM / 2) * 16], p);
}

__aicore__ inline void FixF32NzHalves(LocalTensor<float> ub, LocalTensor<float> l0C,
                                      uint32_t fullM, uint32_t n)
{
    FixpipeParamsArch3510<CO2Layout::NZ> p{};
    p.nSize = static_cast<uint16_t>(n);
    p.mSize = static_cast<uint16_t>(fullM / 2);
    p.srcStride = static_cast<uint16_t>(fullM);
    p.dstStride = (fullM / 2) * 16;
    p.quantPre = QuantMode_t::NoQuant;
    p.dualDstCtl = 0;
    p.isChannelSplit = false;
    p.subBlockId = false;
    Fixpipe<float, float, kSparseFixUbNz>(ub, l0C, p);
    p.subBlockId = true;
    Fixpipe<float, float, kSparseFixUbNz>(ub, l0C[(fullM / 2) * 16], p);
}

__aicore__ inline void FixB16Halves(LocalTensor<bfloat16_t> ub,
                                    LocalTensor<float> l0C,
                                    uint32_t fullM, uint32_t n)
{
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> p{};
    p.nSize = static_cast<uint16_t>(n);
    p.mSize = static_cast<uint16_t>(fullM / 2);
    p.srcStride = static_cast<uint16_t>(fullM);
    p.dstStride = n;
    p.quantPre = QuantMode_t::F322BF16;
    p.dualDstCtl = 0;
    p.isChannelSplit = false;
    p.params.ndNum = 1;
    p.subBlockId = false;
    Fixpipe<bfloat16_t, float, kSparseFixUb>(ub, l0C, p);
    p.subBlockId = true;
    Fixpipe<bfloat16_t, float, kSparseFixUb>(ub, l0C[(fullM / 2) * 16], p);
}

__aicore__ inline void FixB16ScaledHalves(LocalTensor<bfloat16_t> ub,
                                          LocalTensor<float> l0C,
                                          uint32_t fullM, uint32_t n,
                                          float scale)
{
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> p{};
    p.nSize = static_cast<uint16_t>(n);
    p.mSize = static_cast<uint16_t>(fullM / 2);
    p.srcStride = static_cast<uint16_t>(fullM);
    p.dstStride = n;
    p.quantPre = QuantMode_t::QF322BF16_PRE;
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = scale;
    p.deqScalar = static_cast<uint64_t>(bits.u);
    p.dualDstCtl = 0;
    p.isChannelSplit = false;
    p.params.ndNum = 1;
    p.subBlockId = false;
    Fixpipe<bfloat16_t, float, kSparseFixUb>(ub, l0C, p);
    p.subBlockId = true;
    Fixpipe<bfloat16_t, float, kSparseFixUb>(ub, l0C[(fullM / 2) * 16], p);
}

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_CUBE_H
