/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 *
 * Consumer vector units, one function per minimum step. All of them are
 * row-ranged: AIV0 owns chunk rows 0..31, AIV1 rows 32..63. State-shaped
 * buffers are keyed by K instead (AIV0 rows 0..63, AIV1 64..127).
 *
 * Numeric contract follows gdn_reference.py:242-249 and the validated kernels:
 *   right[i,:] = exp(g_last - g_i) * v_new[i,:]
 *   gateA[i,j] = exp(g_i - g_j)
 *   A'[i,j]    = (j <= i) ? A_raw[i,j] * gateA[i,j] : 0
 *   R_next     = exp(g_last) * R + D
 *   o[i,:]     = (O_s[i,:]*exp(g_i) + O_l[i,:]) * scale
 */
#ifndef GDN_CHUNK64_SPARSE_CONSUMER_VF_H
#define GDN_CHUNK64_SPARSE_CONSUMER_VF_H

#include "kernel_operator.h"
#include "sparse_common.h"

namespace GdnChunk64Sparse {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr float kSparseLn2 = 0.6931471825f;
constexpr int32_t kVfVl = 64;

constexpr CastTrait kSparseB16ToF32 = {
    RegLayout::ZERO,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};
constexpr CastTrait kSparseF32ToB16 = {
    RegLayout::ZERO,
    SatMode::NO_SAT,
    MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

__simd_callee__ inline void SparseLoadB16AsF32(RegTensor<float> &dst, __ubuf__ bfloat16_t *src,
                                               MaskReg &mask32)
{
    RegTensor<bfloat16_t> raw;
    LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(raw, src);
    Cast<float, bfloat16_t, kSparseB16ToF32>(dst, raw, mask32);
}

__simd_callee__ inline void SparseStoreB16(__ubuf__ bfloat16_t *dst, RegTensor<float> &src,
                                           MaskReg &mask32)
{
    RegTensor<bfloat16_t> packed;
    Cast<bfloat16_t, float, kSparseF32ToB16>(packed, src, mask32);
    StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(dst, packed, mask32);
}

// gateScale[i] = E(g_last - g_i), gateO[i] = E(g_i) over the full chunk. Both
// AIVs compute the whole 64-lane vectors and use their own row range.
template <bool UseExp2>
__aicore__ inline void GateRowsVF(LocalTensor<float> &g, LocalTensor<float> &gateScale,
                                  LocalTensor<float> &gateO, float gLast)
{
    __ubuf__ float *gAddr = reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr());
    __ubuf__ float *scaleAddr = reinterpret_cast<__ubuf__ float *>(gateScale.GetPhyAddr());
    __ubuf__ float *oAddr = reinterpret_cast<__ubuf__ float *>(gateO.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg mask = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> gVec, last, diff, scale, expo;
        LoadAlign(gVec, gAddr);
        Duplicate(last, gLast, mask);
        Sub(diff, last, gVec, mask);
        if constexpr (UseExp2) {
            Muls(diff, diff, kSparseLn2, mask);
        }
        Exp(scale, diff, mask);
        if constexpr (UseExp2) {
            Muls(gVec, gVec, kSparseLn2, mask);
        }
        Exp(expo, gVec, mask);
        StoreAlign(scaleAddr, scale, mask);
        StoreAlign(oAddr, expo, mask);
    }
}

// gateA[i,j] = E(g_i - g_j) for the local rows [firstRow, firstRow+rows).
template <bool UseExp2>
__aicore__ inline void GateMatrixVF(LocalTensor<float> &gateA, LocalTensor<float> &g,
                                    uint32_t firstRow, uint32_t rows)
{
    __ubuf__ float *dst = reinterpret_cast<__ubuf__ float *>(gateA.GetPhyAddr());
    __ubuf__ float *gAddr = reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr());
    const uint16_t nRows = static_cast<uint16_t>(rows);
    __VEC_SCOPE__
    {
        MaskReg mask = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> gVec, gi, diff, gate;
        LoadAlign(gVec, gAddr);
        for (uint16_t i = 0; i < nRows; i++) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(gi, gAddr + firstRow + i);
            Sub(diff, gi, gVec, mask);
            if constexpr (UseExp2) {
                Muls(diff, diff, kSparseLn2, mask);
            }
            Exp(gate, diff, mask);
            StoreAlign(dst + static_cast<uint32_t>(i) * static_cast<uint32_t>(kChunk), gate, mask);
        }
    }
}

// v_new = U - P; right = gateScale * v_new. Both stored BF16; v_new also feeds
// the public output, right feeds the state matmul through L1.
__aicore__ inline void VNewRightVF(LocalTensor<float> &p, LocalTensor<bfloat16_t> &u,
                                   LocalTensor<bfloat16_t> &vNew, LocalTensor<bfloat16_t> &right,
                                   LocalTensor<float> &gateScale, uint32_t firstRow, uint32_t rows)
{
    __ubuf__ float *pAddr = reinterpret_cast<__ubuf__ float *>(p.GetPhyAddr());
    __ubuf__ bfloat16_t *uAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(u.GetPhyAddr());
    __ubuf__ bfloat16_t *vAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(vNew.GetPhyAddr());
    __ubuf__ bfloat16_t *rAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(right.GetPhyAddr());
    __ubuf__ float *scaleAddr = reinterpret_cast<__ubuf__ float *>(gateScale.GetPhyAddr());
    const uint16_t nRows = static_cast<uint16_t>(rows);
    __VEC_SCOPE__
    {
        MaskReg mask = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> pf0, pf1, uf0, uf1, vf0, vf1, s, r0, r1;
        for (uint16_t i = 0; i < nRows; i++) {
            const uint32_t row = static_cast<uint32_t>(i);
            const uint32_t off = row * static_cast<uint32_t>(kHeadDim);
            LoadAlign(pf0, pAddr + off);
            LoadAlign(pf1, pAddr + off + kVfVl);
            SparseLoadB16AsF32(uf0, uAddr + off, mask);
            SparseLoadB16AsF32(uf1, uAddr + off + kVfVl, mask);
            Sub(vf0, uf0, pf0, mask);
            Sub(vf1, uf1, pf1, mask);
            SparseStoreB16(vAddr + off, vf0, mask);
            SparseStoreB16(vAddr + off + kVfVl, vf1, mask);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(s, scaleAddr + firstRow + row);
            Mul(r0, vf0, s, mask);
            Mul(r1, vf1, s, mask);
            SparseStoreB16(rAddr + off, r0, mask);
            SparseStoreB16(rAddr + off + kVfVl, r1, mask);
        }
    }
}

// O_s' = O_s * gateO, in place.
__aicore__ inline void GateOutputVF(LocalTensor<float> &oS, LocalTensor<float> &gateO,
                                    uint32_t firstRow, uint32_t rows)
{
    __ubuf__ float *oAddr = reinterpret_cast<__ubuf__ float *>(oS.GetPhyAddr());
    __ubuf__ float *gAddr = reinterpret_cast<__ubuf__ float *>(gateO.GetPhyAddr());
    const uint16_t nRows = static_cast<uint16_t>(rows);
    __VEC_SCOPE__
    {
        MaskReg mask = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> o0, o1, s;
        for (uint16_t i = 0; i < nRows; i++) {
            const uint32_t off = static_cast<uint32_t>(i) * static_cast<uint32_t>(kHeadDim);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(s, gAddr + firstRow + i);
            LoadAlign(o0, oAddr + off);
            LoadAlign(o1, oAddr + off + kVfVl);
            Mul(o0, o0, s, mask);
            Mul(o1, o1, s, mask);
            StoreAlign(oAddr + off, o0, mask);
            StoreAlign(oAddr + off + kVfVl, o1, mask);
        }
    }
}

// A'[i,j] = (j <= i) ? A_raw[i,j] * gateA[i,j] : 0, BF16 for the cube. The
// diagonal is included: o_i contains the token-i delta update, so the j == i
// term q_i . k_i * v_new_i belongs to A'. This matches the production O
// kernel (chunk_fwd_o Stage3 uses row + 1 lanes) and chunk32's GateAttnVF.
__aicore__ inline void APrimeVF(LocalTensor<bfloat16_t> &aPrime, LocalTensor<float> &aRaw,
                                LocalTensor<float> &gateA, uint32_t firstRow, uint32_t rows)
{
    __ubuf__ bfloat16_t *dst = reinterpret_cast<__ubuf__ bfloat16_t *>(aPrime.GetPhyAddr());
    __ubuf__ float *rawAddr = reinterpret_cast<__ubuf__ float *>(aRaw.GetPhyAddr());
    __ubuf__ float *gateAddr = reinterpret_cast<__ubuf__ float *>(gateA.GetPhyAddr());
    const uint16_t nRows = static_cast<uint16_t>(rows);
    __VEC_SCOPE__
    {
        MaskReg mask = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> raw, gate, prod, zero;
        Duplicate(zero, 0.0f, mask);
        for (uint16_t i = 0; i < nRows; i++) {
            const uint32_t off = static_cast<uint32_t>(i) * static_cast<uint32_t>(kChunk);
            uint32_t count = firstRow + static_cast<uint32_t>(i) + 1U;
            MaskReg lower = UpdateMask<float>(count);
            LoadAlign(raw, rawAddr + off);
            LoadAlign(gate, gateAddr + off);
            Mul(prod, raw, gate, mask);
            Select(prod, prod, zero, lower);
            SparseStoreB16(dst + off, prod, mask);
        }
    }
}

// gLast = g[lastIndex] and alpha = E(gLast) as two scalars. The chunk cumsum
// is zero-padded past the last valid row, so the last token's cumsum still
// sits at lastIndex (not necessarily kChunk - 1) on a short tail.
template <bool UseExp2>
__aicore__ inline void LastGateVF(LocalTensor<float> &g, LocalTensor<float> &out, uint32_t lastIndex)
{
    __ubuf__ float *gAddr = reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr());
    __ubuf__ float *outAddr = reinterpret_cast<__ubuf__ float *>(out.GetPhyAddr());
    __VEC_SCOPE__
    {
        MaskReg one = CreateMask<float, MaskPattern::VL1>();
        RegTensor<float> last, alpha;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(last, gAddr + lastIndex);
        if constexpr (UseExp2) {
            Muls(alpha, last, kSparseLn2, one);
            Exp(alpha, alpha, one);
        } else {
            Exp(alpha, last, one);
        }
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(outAddr, last, one);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(outAddr + 1, alpha, one);
    }
}

// o = (O_s' + O_l) * scale, BF16 into the row-half output buffer.
__aicore__ inline void OFuseVF(LocalTensor<bfloat16_t> &o, LocalTensor<float> &oS,
                               LocalTensor<float> &oL, float scale, uint32_t rows)
{
    __ubuf__ bfloat16_t *dst = reinterpret_cast<__ubuf__ bfloat16_t *>(o.GetPhyAddr());
    __ubuf__ float *sAddr = reinterpret_cast<__ubuf__ float *>(oS.GetPhyAddr());
    __ubuf__ float *lAddr = reinterpret_cast<__ubuf__ float *>(oL.GetPhyAddr());
    const uint16_t nRows = static_cast<uint16_t>(rows);
    __VEC_SCOPE__
    {
        MaskReg mask = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> s0, s1, l0, l1, out0, out1;
        for (uint16_t i = 0; i < nRows; i++) {
            const uint32_t off = static_cast<uint32_t>(i) * static_cast<uint32_t>(kHeadDim);
            LoadAlign(s0, sAddr + off);
            LoadAlign(s1, sAddr + off + kVfVl);
            LoadAlign(l0, lAddr + off);
            LoadAlign(l1, lAddr + off + kVfVl);
            Add(out0, s0, l0, mask);
            Add(out1, s1, l1, mask);
            Muls(out0, out0, scale, mask);
            Muls(out1, out1, scale, mask);
            SparseStoreB16(dst + off, out0, mask);
            SparseStoreB16(dst + off + kVfVl, out1, mask);
        }
    }
}

// R = alpha * R + D over this AIV's K half, then refresh the BF16 cube image.
__aicore__ inline void StateUpdateVF(LocalTensor<float> &stateF, LocalTensor<float> &d,
                                     LocalTensor<bfloat16_t> &stateB, float alpha, uint32_t rows)
{
    __ubuf__ float *stateAddr = reinterpret_cast<__ubuf__ float *>(stateF.GetPhyAddr());
    __ubuf__ float *dAddr = reinterpret_cast<__ubuf__ float *>(d.GetPhyAddr());
    __ubuf__ bfloat16_t *bAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(stateB.GetPhyAddr());
    const uint16_t nRows = static_cast<uint16_t>(rows);
    __VEC_SCOPE__
    {
        MaskReg mask = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> s0, s1, dd0, dd1, alphaReg;
        Duplicate(alphaReg, alpha, mask);
        for (uint16_t i = 0; i < nRows; i++) {
            const uint32_t off = static_cast<uint32_t>(i) * static_cast<uint32_t>(kHeadDim);
            LoadAlign(s0, stateAddr + off);
            LoadAlign(s1, stateAddr + off + kVfVl);
            LoadAlign(dd0, dAddr + off);
            LoadAlign(dd1, dAddr + off + kVfVl);
            Mul(s0, s0, alphaReg, mask);
            Mul(s1, s1, alphaReg, mask);
            Add(s0, s0, dd0, mask);
            Add(s1, s1, dd1, mask);
            StoreAlign(stateAddr + off, s0, mask);
            StoreAlign(stateAddr + off + kVfVl, s1, mask);
            SparseStoreB16(bAddr + off, s0, mask);
            SparseStoreB16(bAddr + off + kVfVl, s1, mask);
        }
    }
}

// Initial state load: BF16 (or FP32) GM row into the FP32 resident image and
// its BF16 mirror, for this AIV's K half.
__aicore__ inline void StateLoadVF(LocalTensor<float> &stateF, LocalTensor<bfloat16_t> &stateB,
                                   LocalTensor<bfloat16_t> &staging, uint32_t rows)
{
    __ubuf__ bfloat16_t *src = reinterpret_cast<__ubuf__ bfloat16_t *>(staging.GetPhyAddr());
    __ubuf__ float *dstF = reinterpret_cast<__ubuf__ float *>(stateF.GetPhyAddr());
    __ubuf__ bfloat16_t *dstB = reinterpret_cast<__ubuf__ bfloat16_t *>(stateB.GetPhyAddr());
    const uint16_t nRows = static_cast<uint16_t>(rows);
    __VEC_SCOPE__
    {
        MaskReg mask = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> v0, v1;
        for (uint16_t i = 0; i < nRows; i++) {
            const uint32_t off = static_cast<uint32_t>(i) * static_cast<uint32_t>(kHeadDim);
            SparseLoadB16AsF32(v0, src + off, mask);
            SparseLoadB16AsF32(v1, src + off + kVfVl, mask);
            StoreAlign(dstF + off, v0, mask);
            StoreAlign(dstF + off + kVfVl, v1, mask);
            SparseStoreB16(dstB + off, v0, mask);
            SparseStoreB16(dstB + off + kVfVl, v1, mask);
        }
    }
}

// FP32 initial state already lives in stateF: build the BF16 cube mirror.
// Called once per chain, before the first chunk's L1 upload.
__aicore__ inline void StateMirrorVF(LocalTensor<float> &stateF, LocalTensor<bfloat16_t> &stateB,
                                     uint32_t rows)
{
    __ubuf__ float *srcF = reinterpret_cast<__ubuf__ float *>(stateF.GetPhyAddr());
    __ubuf__ bfloat16_t *dstB = reinterpret_cast<__ubuf__ bfloat16_t *>(stateB.GetPhyAddr());
    const uint16_t nRows = static_cast<uint16_t>(rows);
    __VEC_SCOPE__
    {
        MaskReg mask = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> v0, v1;
        for (uint16_t i = 0; i < nRows; i++) {
            const uint32_t off = static_cast<uint32_t>(i) * static_cast<uint32_t>(kHeadDim);
            LoadAlign(v0, srcF + off);
            LoadAlign(v1, srcF + off + kVfVl);
            SparseStoreB16(dstB + off, v0, mask);
            SparseStoreB16(dstB + off + kVfVl, v1, mask);
        }
    }
}

// Copy the BF16 state image into the L1 cube layout. NZ keeps 16-row fractals
// inside a column fractal, so a K half starts at rowOffset*16 of every column
// fractal and stays contiguous for the whole 64-row band.
__aicore__ inline void StateUploadL1(LocalTensor<bfloat16_t> &l1, LocalTensor<bfloat16_t> &stateB,
                                     uint32_t nzRows, uint32_t rowOffset, uint32_t rows)
{
    constexpr uint32_t kNFrac = static_cast<uint32_t>(kHeadDim) / 16;
    const uint16_t srcGap = static_cast<uint16_t>(kNFrac - 1);
    for (uint32_t fc = 0; fc < kNFrac; ++fc) {
        DataCopy(l1[static_cast<int32_t>(fc) * static_cast<int32_t>(nzRows * 16) +
                     static_cast<int32_t>(rowOffset * 16)],
                 stateB[static_cast<int32_t>(fc * 16)],
                 DataCopyParams(static_cast<uint16_t>(rows), 1, srcGap, 0));
    }
}

// BF16 cube tile upload for the [64,cols] consumer tensors (W, Q, K, V_new,
// right, A'). Both AIVs write their own 32-row band at rowOffset = 32*sub, so
// the L1 image is fully defined even when nRows is short, because the staging
// buffers are zero-filled past the valid rows first. cols is 128 or 64.
__aicore__ inline void ChunkUploadL1(LocalTensor<bfloat16_t> &l1, LocalTensor<bfloat16_t> &ub,
                                     uint32_t cols, uint32_t rowOffset, uint32_t rows)
{
    constexpr uint32_t kNzRows = static_cast<uint32_t>(kChunk);
    const uint32_t nFrac = cols / 16;
    const uint16_t srcGap = static_cast<uint16_t>(nFrac - 1);
    for (uint32_t fc = 0; fc < nFrac; ++fc) {
        DataCopy(l1[static_cast<int32_t>(fc) * static_cast<int32_t>(kNzRows * 16) +
                     static_cast<int32_t>(rowOffset * 16)],
                 ub[static_cast<int32_t>(fc * 16)],
                 DataCopyParams(static_cast<uint16_t>(rows), 1, srcGap, 0));
    }
}

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_CONSUMER_VF_H
