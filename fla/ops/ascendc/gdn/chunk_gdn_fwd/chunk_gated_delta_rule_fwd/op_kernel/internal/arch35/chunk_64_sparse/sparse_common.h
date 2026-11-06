/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_CHUNK64_SPARSE_COMMON_H
#define GDN_CHUNK64_SPARSE_COMMON_H

#include "kernel_operator.h"
#include "sparse_fused_tiling.h"
#include "../../gdn_onchip_types.h"

namespace GdnChunk64Sparse {
using namespace AscendC;

constexpr int64_t kChunk = 64;
constexpr int64_t kHalfRows = 32;
constexpr int64_t kDim = 128;
constexpr int64_t kHalfDim = 64;
constexpr int64_t kChunkElems = kChunk * kDim;
constexpr int64_t kMatrixElems = kChunk * kChunk;
constexpr int64_t kStateElems = kDim * kDim;
constexpr float kSparseEps = 1e-6f;
constexpr float kSparseLn2 = 0.6931471805599453f;
constexpr float kSparseRcpLn2 = 1.4426950408889634f;
constexpr float kSparseExpSafeLimit = 80.0f;

struct PhaseArgs {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR rawG;
    GM_ADDR beta;
    GM_ADDR aLog;
    GM_ADDR dtBias;
    GM_ADDR cuSeqlens;
    GM_ADDR chunkIndices;
    GM_ADDR gCumsum;
    GM_ADDR w;
    GM_ADDR u;
    GM_ADDR a;
    GM_ADDR qHat;
    GM_ADDR kHat;
    GM_ADDR qRstd;
    GM_ADDR kRstd;
    GM_ADDR betaEff;
    GM_ADDR h;
    GM_ADDR vNew;
    GM_ADDR o;
    GM_ADDR initialState;
    GM_ADDR finalState;
    GM_ADDR workspace;
};

struct ChainLoc {
    int64_t batch;
    int64_t seq;
    int64_t hv;
    int64_t kh;
    int64_t chunkBase;
    int64_t chunks;
    int64_t tokenBegin;
    int64_t tokenEnd;
};

struct ChunkWindow {
    int64_t tokenStart;
    int64_t rows;
};

__aicore__ inline int64_t OffsetBhtd(int64_t batch, int64_t head, int64_t token,
                                     int64_t heads, int64_t tokens, int64_t dim = kDim)
{
    return ((batch * heads + head) * tokens + token) * dim;
}

__aicore__ inline int64_t OffsetBht(int64_t batch, int64_t head, int64_t token,
                                    int64_t heads, int64_t tokens)
{
    return (batch * heads + head) * tokens + token;
}

__aicore__ inline int64_t TileIndex(const ChainLoc &loc, int64_t localChunk,
                                    int64_t chunksPerBatch, bool varlen)
{
    return varlen ? loc.chunkBase + localChunk : loc.batch * chunksPerBatch + localChunk;
}

__aicore__ inline int64_t OffsetStateTile(int64_t tile, int64_t head, int64_t heads)
{
    return (tile * heads + head) * kStateElems;
}

__aicore__ inline int64_t OffsetOutput(int64_t batch, int64_t token, int64_t head,
                                       int64_t heads, int64_t tokens)
{
    return ((batch * tokens + token) * heads + head) * kDim;
}

__aicore__ inline ChainLoc ResolveChain(const FusedTilingData &td, GM_ADDR cuSeqlens,
                                        uint32_t chainIdx)
{
    ChainLoc loc{};
    loc.hv = static_cast<int64_t>(chainIdx) % td.valueHeads;
    const int64_t sequence = static_cast<int64_t>(chainIdx) / td.valueHeads;
    loc.kh = loc.hv / (td.valueHeads / td.keyHeads);
    loc.batch = sequence;
    loc.seq = sequence;
    if (td.varlen != 0) {
        GlobalTensor<int64_t> cu;
        cu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
        loc.batch = 0;
        int64_t prefix = 0;
        for (int64_t s = 0; s < sequence; ++s) {
            const int64_t len = cu.GetValue(s + 1) - cu.GetValue(s);
            prefix += len > 0 ? (len + kChunk - 1) / kChunk : 0;
        }
        loc.tokenBegin = cu.GetValue(sequence);
        loc.tokenEnd = cu.GetValue(sequence + 1);
        loc.chunkBase = prefix;
        loc.chunks = loc.tokenEnd > loc.tokenBegin
                         ? (loc.tokenEnd - loc.tokenBegin + kChunk - 1) / kChunk
                         : 0;
    } else {
        loc.chunkBase = sequence * td.chunksPerBatch;
        loc.chunks = td.chunksPerBatch;
        loc.tokenBegin = 0;
        loc.tokenEnd = td.tokens;
    }
    return loc;
}

__aicore__ inline ChunkWindow ResolveWindow(const ChainLoc &loc, int64_t localChunk)
{
    ChunkWindow window{};
    window.tokenStart = loc.tokenBegin + localChunk * kChunk;
    const int64_t remain = loc.tokenEnd - window.tokenStart;
    window.rows = remain <= 0 ? 0 : (remain < kChunk ? remain : kChunk);
    return window;
}

__aicore__ inline uint32_t GroupIndex()
{
    uint32_t group = GetBlockIdx();
    if ASCEND_IS_AIV {
        const uint32_t subNum = GetSubBlockNum();
        if (subNum != 0) {
            group /= subNum;
        }
    }
    return group;
}

template <typename T>
__aicore__ inline void CopyGmToUb(LocalTensor<T> dst, GlobalTensor<T> src,
                                  int64_t offset, uint32_t count)
{
    if (count == 0) {
        return;
    }
    const uint32_t bytes = count * static_cast<uint32_t>(sizeof(T));
    if (bytes >= 32 && (bytes & 31U) == 0 &&
        ((static_cast<uint64_t>(offset) * sizeof(T)) & 31U) == 0) {
        DataCopy(dst, src[offset], count);
        return;
    }
    const uint32_t padBytes = (32U - (bytes & 31U)) & 31U;
    DataCopyParams params{1, static_cast<uint16_t>(bytes), 0, 0};
    DataCopyPadParams pad{padBytes != 0, 0,
                          static_cast<uint8_t>(padBytes / sizeof(T)), 0};
    DataCopyPad(dst, src[offset], params, pad);
}

template <typename T>
__aicore__ inline void CopyUbToGm(GlobalTensor<T> dst, int64_t offset,
                                  LocalTensor<T> src, uint32_t count)
{
    if (count == 0) {
        return;
    }
    const uint32_t bytes = count * static_cast<uint32_t>(sizeof(T));
    if (bytes >= 32 && (bytes & 31U) == 0 &&
        ((static_cast<uint64_t>(offset) * sizeof(T)) & 31U) == 0) {
        DataCopy(dst[offset], src, count);
        return;
    }
    DataCopyExtParams params{};
    params.blockCount = 1;
    params.blockLen = bytes;
    params.srcStride = 0;
    params.dstStride = 0;
    params.rsv = 0;
    DataCopyPad(dst[offset], src, params);
}

__aicore__ inline void UploadNzStateToL1(LocalTensor<bfloat16_t> l1,
                                         LocalTensor<bfloat16_t> ub,
                                         uint32_t rowOffset)
{
    // Eight contiguous local panels; skip the other AIV's 64 rows in L1.
    DataCopy(l1[rowOffset * 16], ub, DataCopyParams(8, 64, 0, 64));
}

__aicore__ inline void UploadNdToL1(LocalTensor<bfloat16_t> l1,
                                    LocalTensor<bfloat16_t> ub,
                                    uint32_t rows, uint32_t cols,
                                    uint32_t totalRows, uint32_t rowOffset)
{
    if (rows == kHalfDim && totalRows == kDim) {
        // The full H conversion would need a 30KB temporary UB image. Keep
        // its panel path; all 32/64-row operands use the single-call ND2NZ.
        const uint32_t panels = cols / 16;
        const uint16_t srcGap = static_cast<uint16_t>(panels - 1);
        for (uint32_t panel = 0; panel < panels; ++panel) {
            DataCopy(l1[panel * totalRows * 16 + rowOffset * 16], ub[panel * 16],
                     DataCopyParams(static_cast<uint16_t>(rows), 1, srcGap, 0));
        }
        return;
    }
    Nd2NzParams params{};
    params.ndNum = 1;
    params.nValue = rows;
    params.dValue = cols;
    params.srcNdMatrixStride = 0;
    params.srcDValue = cols;
    params.dstNzC0Stride = totalRows;
    params.dstNzNStride = 1;
    params.dstNzMatrixStride = 0;
    DataCopy(l1[rowOffset * 16], ub, params);
}

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_COMMON_H
