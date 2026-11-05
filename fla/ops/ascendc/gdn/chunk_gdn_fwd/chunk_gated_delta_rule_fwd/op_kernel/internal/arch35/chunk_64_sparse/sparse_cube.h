/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_CHUNK64_SPARSE_CUBE_H
#define GDN_CHUNK64_SPARSE_CUBE_H

#include "sparse_common.h"

namespace GdnChunk64Sparse {

constexpr FixpipeConfig kSparseFixUb = {CO2Layout::ROW_MAJOR, true};

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

__aicore__ inline void IssueMm(LocalTensor<bfloat16_t> l0A,
                               LocalTensor<bfloat16_t> l0B,
                               LocalTensor<float> l0C,
                               int32_t m, int32_t n, int32_t k, bool initC = true)
{
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    if (initC) {
        WaitFlag<HardEvent::FIX_M>(0);
    }
    MmadParams p{};
    p.m = m;
    p.n = n;
    p.k = k;
    p.cmatrixInitVal = initC;
    p.cmatrixSource = false;
    p.unitFlag = 0;
    Mmad(l0C, l0A, l0B, p);
    SetFlag<HardEvent::M_MTE1>(0);
    SetFlag<HardEvent::M_FIX>(0);
    WaitFlag<HardEvent::M_FIX>(0);
}

// One independent BF16 GEMM. The caller fixpipes L0C and then calls
// ReleaseL0C. Event zero is primed once by the pipeline entry.
__aicore__ inline void ComputeMm(LocalTensor<bfloat16_t> l1A,
                                 LocalTensor<bfloat16_t> l1B,
                                 LocalTensor<bfloat16_t> l0A,
                                 LocalTensor<bfloat16_t> l0B,
                                 LocalTensor<float> l0C,
                                 int32_t m, int32_t n, int32_t k,
                                 bool transposeA, bool transposeB, bool initC = true)
{
    WaitFlag<HardEvent::M_MTE1>(0);
    LoadA(l0A, l1A, m, k, transposeA);
    LoadB(l0B, l1B, n, k, transposeB);
    IssueMm(l0A, l0B, l0C, m, n, k, initC);
}

// Reuse the B operand left in L0 by the immediately preceding GEMM.
__aicore__ inline void ComputeMmReuseB(LocalTensor<bfloat16_t> l1A,
                                       LocalTensor<bfloat16_t> l0A,
                                       LocalTensor<bfloat16_t> l0B,
                                       LocalTensor<float> l0C,
                                       int32_t m, int32_t n, int32_t k,
                                       bool transposeA, bool initC = true)
{
    WaitFlag<HardEvent::M_MTE1>(0);
    LoadA(l0A, l1A, m, k, transposeA);
    IssueMm(l0A, l0B, l0C, m, n, k, initC);
}

// Reuse the A operand left in L0 by the immediately preceding GEMM.
__aicore__ inline void ComputeMmReuseA(LocalTensor<bfloat16_t> l1B,
                                       LocalTensor<bfloat16_t> l0A,
                                       LocalTensor<bfloat16_t> l0B,
                                       LocalTensor<float> l0C,
                                       int32_t m, int32_t n, int32_t k,
                                       bool transposeB, bool initC = true)
{
    WaitFlag<HardEvent::M_MTE1>(0);
    LoadB(l0B, l1B, n, k, transposeB);
    IssueMm(l0A, l0B, l0C, m, n, k, initC);
}

__aicore__ inline void ReleaseL0C()
{
    SetFlag<HardEvent::FIX_M>(0);
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
