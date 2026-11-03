/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_CHUNK64_SPARSE_FUSED_TILING_H
#define GDN_CHUNK64_SPARSE_FUSED_TILING_H

#include <cstdint>

namespace GdnChunk64Sparse {

// Self-contained copies of the plain H/O tiling PODs (previously included
// from the chunk_fwd_h / chunk_fwd_o operators; the operator must not include
// headers from other operators). Only the fields the sparse roles consume
// are kept; the layout is unchanged from the original definitions.
struct ChunkFwdHPlainTilingData {
    int64_t batch;
    int64_t seqlen;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    bool useInitialState;
    bool storeFinalState;
    int64_t isVariedLen;
    int64_t shapeBatch;
    int64_t tokenBatch;
    int64_t vWorkspaceOffset;
    int64_t vUpdateWorkspaceOffset;
    int64_t kDecayWorkspaceOffset;
    int64_t hWorkspaceOffset;
};

enum ChunkFwdOOutputLayout : int64_t {
    CHUNK_FWD_O_LAYOUT_BNSD = 0,
    CHUNK_FWD_O_LAYOUT_BSND = 1,
    CHUNK_FWD_O_LAYOUT_TND = 2,
    CHUNK_FWD_O_LAYOUT_NTD = 3,
};

struct ChunkFwdOTilingData {
    int64_t shapeBatch;
    int64_t seqlen;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    int64_t isVariedLen;
    int64_t tokenBatch;
    int64_t dataType;
    int64_t gDataType;
    int64_t vWorkspaceOffset;
    int64_t hWorkspaceOffset;
    int64_t attnWorkspaceOffset;
    int64_t aftermaskWorkspaceOffset;
    int64_t maskWorkspaceOffset;
    int64_t stateVFirst;
    int64_t outputLayout;
    float scale;
    int64_t chunkNum;
    int64_t hvPerHk;
    int64_t taskGroupSize;
    int64_t numChunksPerBatch;
    int64_t aPrimeWorkspaceOffset;
};

// Mirrors Prepare's field contract without including device-only headers.
struct PrepareTilingData {
    int64_t inputBatchSize;
    int64_t queryKeyHeadCount;
    int64_t valueHeadCount;
    int64_t sequenceTokenLength;
    int64_t queryKeyHeadDim;
    int64_t valueHeadDim;
    int64_t valueHeadsPerQueryKeyHead;
    int64_t tokensPerChunk;
    int64_t packedSequenceCount;
    int64_t isVariableLengthPacked;
    int64_t hasChunkIndexTable;
    int64_t enableQueryKeyL2NormInKernel;
    int64_t enableFusedGateSoftplus;
    int64_t enableBetaSigmoid;
    int64_t hasGateDtBias;
    int64_t hasQueryHatGmOutput;
    int64_t hasKeyHatGmOutput;
    int64_t scaleBetaByTwoWhenNegEigval;
    int64_t useExp2ForGateCumsum;
    int64_t queryKeyStorageDtype;
    int64_t gateStorageDtype;
    int64_t betaStorageDtype;
    int64_t totalChunkTileCount;
    int64_t hasAOutput;
};

// Key-64 schedule selector. Serial keeps the existing Prepare -> H -> O
// composition; Pipeline is the low-parallelism producer/consumer path.
constexpr uint32_t kSparseScheduleSerial = 0;
constexpr uint32_t kSparseSchedulePipeline = 1;

// Mode-0 flag id for the producer/consumer barrier. Stage-local handshakes use
// 0..6 (AIV1 at id+16), so 7 stays free; SyncAll owns 11..14 and must not be
// mixed in.
constexpr uint32_t kSparseBarrierFlagId = 7;

// Producer/consumer tiling. `chainCount` chunks of one (b,hv) chain run on the
// producer group `c`; the same chunks are consumed one iteration later by the
// consumer group `chainCount + c`. Both counts are host-side upper bounds:
// the device trims them against cu_seqlens / chunk_indices before the loop.
struct PipelineTilingData {
    // Role-local scratch. Zero when the fused on-chip map needs no GM spill.
    uint64_t workspaceOffset;
    uint64_t workspaceBytes;
    // kSparseScheduleSerial | kSparseSchedulePipeline.
    uint32_t schedule;
    // Mode-0 (all-AIC / all-AIV) barrier id shared by both roles.
    uint32_t barrierFlagId;
    // Upper bound on independent (b,hv) chains.
    uint32_t chainCount;
    // Upper bound on chunks in one chain.
    uint32_t maxChunks;
    // Producer groups [0, chainCount), consumer groups [chainCount, 2*chainCount).
    uint32_t producerGroups;
    uint32_t consumerGroups;
    // Bring-up bisect switch, host-filled from the GDN_SPARSE_DEBUG_MODE
    // environment variable so no rebuild is needed: 0 = full pipeline,
    // 1 = barrier/relay only, 2 = producer only, 3 = consumer only.
    uint32_t debugMode;
    uint32_t reserved;
};

struct FusedTilingData {
    PrepareTilingData prepare;
    ChunkFwdHPlainTilingData h;
    ChunkFwdOTilingData o;
    PipelineTilingData pipeline;
    uint64_t hWorkspaceOffset;
    uint64_t oWorkspaceOffset;
    uint64_t workspaceBytes;
    uint32_t launchGroups;
    uint32_t useExp2;
    uint32_t stateVFirst;
    uint32_t stateDtype;
};

} // namespace GdnChunk64Sparse
#endif // GDN_CHUNK64_SPARSE_FUSED_TILING_H
