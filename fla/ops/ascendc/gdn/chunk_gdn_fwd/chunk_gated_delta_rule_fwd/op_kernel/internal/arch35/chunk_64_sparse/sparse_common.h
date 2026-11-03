/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 *
 * Shared constants and scalar helpers for the chunk64 producer/consumer path.
 * The stage modules own the math; this header owns chain identity, GM offsets
 * and the tail rules both roles must agree on.
 */
#ifndef GDN_CHUNK64_SPARSE_COMMON_H
#define GDN_CHUNK64_SPARSE_COMMON_H

#include "kernel_operator.h"
#include "sparse_fused_tiling.h"

namespace GdnChunk64Sparse {

using namespace AscendC;

// GM argument bundle shared by every sparse role.
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

constexpr int64_t kChunk = 64;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kChunkElems = kChunk * kHeadDim;
constexpr int64_t kMatrixElems = kChunk * kChunk;
constexpr int64_t kStateElems = kHeadDim * kHeadDim;
constexpr int64_t kHalfRows = kChunk / 2;
constexpr int64_t kHalfDim = kHeadDim / 2;

constexpr uint32_t kBytesBf16Chunk = static_cast<uint32_t>(kChunkElems * sizeof(bfloat16_t));
constexpr uint32_t kBytesBf16Matrix = static_cast<uint32_t>(kMatrixElems * sizeof(bfloat16_t));
constexpr uint32_t kBytesFp32Matrix = static_cast<uint32_t>(kMatrixElems * sizeof(float));
constexpr uint32_t kBytesFp32Chunk = static_cast<uint32_t>(kChunkElems * sizeof(float));
constexpr uint32_t kBytesFp32State = static_cast<uint32_t>(kStateElems * sizeof(float));
constexpr uint32_t kBytesBf16State = static_cast<uint32_t>(kStateElems * sizeof(bfloat16_t));
constexpr uint32_t kBytesFp32HalfChunk = static_cast<uint32_t>(kHalfRows * kHeadDim * sizeof(float));
constexpr uint32_t kBytesBf16HalfChunk = static_cast<uint32_t>(kHalfRows * kHeadDim * sizeof(bfloat16_t));
constexpr uint32_t kBytesVec = static_cast<uint32_t>(kChunk * sizeof(float));

// GM row index helpers. Inputs are BNSD [B,H,T,D]; h/v_new are chunk-major
// with the head inside the chunk axis; o is token-major BSND.
__aicore__ inline int64_t OffsetBhtd(int64_t batch, int64_t head, int64_t token, int64_t heads,
                                     int64_t seq, int64_t dim)
{
    return ((batch * heads + head) * seq + token) * dim;
}

__aicore__ inline int64_t OffsetBht(int64_t batch, int64_t head, int64_t token, int64_t heads,
                                    int64_t seq)
{
    return (batch * heads + head) * seq + token;
}

// Dense: h[(b*NT + c)*Hv + hv]. Varlen: h[globalChunk*Hv + hv].
__aicore__ inline int64_t OffsetHChunk(bool variedLen, int64_t batch, int64_t globalChunk,
                                       int64_t head, int64_t heads, int64_t chunksPerBatch)
{
    const int64_t chunkIndex = variedLen ? globalChunk : batch * chunksPerBatch + globalChunk;
    return (chunkIndex * heads + head) * kStateElems;
}

__aicore__ inline int64_t OffsetONd(int64_t batch, int64_t token, int64_t head, int64_t heads,
                                    int64_t seq)
{
    return ((batch * seq + token) * heads + head) * kHeadDim;
}

// One chain = one (sequence, value head). `chunkBase` is the global chunk index
// of this sequence's first chunk; `chunks` is how many chunks it owns.
struct ChainLoc {
    int64_t batch;      // BNSD dim-0 (0 for packed varlen)
    int64_t seq;        // sequence index for cu_seqlens / initial state
    int64_t hv;         // value head
    int64_t kh;         // key head (hv / hvPerHk)
    int64_t chunkBase;  // global chunk index of local chunk 0
    int64_t chunks;     // local chunk count
    int64_t tokenBegin; // first token of the sequence (absolute on T)
    int64_t tokenEnd;   // last token (exclusive) of the sequence
};

// Resolve chain `chainIdx` from the same cu_seqlens scan the standalone H
// kernel performs (chunk_indices is derived, not trusted).
__aicore__ inline ChainLoc ResolveChain(const FusedTilingData &td, GM_ADDR cuSeqlens,
                                        uint32_t chainIdx)
{
    const int64_t hvTotal = td.prepare.valueHeadCount;
    const int64_t hvPerHk = td.prepare.valueHeadsPerQueryKeyHead > 0
                                ? td.prepare.valueHeadsPerQueryKeyHead
                                : 1;
    ChainLoc loc{};
    loc.hv = hvTotal > 0 ? static_cast<int64_t>(chainIdx) % hvTotal : 0;
    const int64_t sequence = hvTotal > 0 ? static_cast<int64_t>(chainIdx) / hvTotal : 0;
    loc.kh = loc.hv / hvPerHk;
    loc.batch = sequence;
    loc.seq = sequence;
    if (td.prepare.isVariableLengthPacked != 0 && cuSeqlens != nullptr) {
        GlobalTensor<int64_t> gmCu;
        gmCu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
        loc.batch = 0;
        int64_t prefix = 0;
        for (int64_t s = 0; s < sequence; ++s) {
            const int64_t length = gmCu.GetValue(s + 1) - gmCu.GetValue(s);
            prefix += length <= 0 ? 0 : (length + kChunk - 1) / kChunk;
        }
        const int64_t begin = gmCu.GetValue(sequence);
        const int64_t end = gmCu.GetValue(sequence + 1);
        loc.chunkBase = prefix;
        loc.chunks = end > begin ? (end - begin + kChunk - 1) / kChunk : 0;
        loc.tokenBegin = begin;
        loc.tokenEnd = end;
    } else {
        const int64_t chunksPerBatch = td.pipeline.maxChunks;
        loc.chunkBase = sequence * chunksPerBatch;
        loc.chunks = chunksPerBatch;
        loc.tokenBegin = 0;
        loc.tokenEnd = td.prepare.sequenceTokenLength;
    }
    return loc;
}

// Token window of local chunk `localChunk` of `loc`. Valid rows may be short in
// the last chunk; every stage zero-fills the remainder.
struct ChunkWindow {
    int64_t tokenStart;
    int64_t rows;
};

__aicore__ inline ChunkWindow ResolveWindow(const ChainLoc &loc, int64_t localChunk)
{
    ChunkWindow window{};
    window.tokenStart = loc.tokenBegin + localChunk * kChunk;
    const int64_t remain = loc.tokenEnd - window.tokenStart;
    window.rows = remain < 0 ? 0 : (remain > kChunk ? kChunk : remain);
    return window;
}

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_COMMON_H
