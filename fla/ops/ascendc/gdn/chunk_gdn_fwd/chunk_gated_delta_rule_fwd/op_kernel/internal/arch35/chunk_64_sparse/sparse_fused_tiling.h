/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 *
 * Minimal host/device contract for the chunk64 two-group implementation.
 */
#ifndef GDN_CHUNK64_SPARSE_FUSED_TILING_H
#define GDN_CHUNK64_SPARSE_FUSED_TILING_H

#include <cstdint>

namespace GdnChunk64Sparse {

constexpr uint32_t kSparseScheduleSerial = 0;
constexpr uint32_t kSparseSchedulePipeline = 1;
constexpr uint32_t kSparseBarrierFlagId = 7;
constexpr uint32_t kSparseInverseBytesPerTile = 4 * 32 * 32 * sizeof(float);

struct FusedTilingData {
    int64_t batch;
    int64_t sequenceCount;
    int64_t keyHeads;
    int64_t valueHeads;
    int64_t tokens;
    int64_t totalChunks;
    int64_t chunksPerBatch;
    int64_t workspaceOffset;
    int64_t workspaceBytes;
    float scale;
    uint32_t chainCount;
    uint32_t maxChunks;
    uint32_t launchGroups;
    uint32_t schedule;
    uint32_t useNorm;
    uint32_t gateSoftplus;
    uint32_t betaSigmoid;
    uint32_t allowNegEigval;
    uint32_t useExp2;
    uint32_t outputA;
    uint32_t storeFinalState;
    uint32_t stateVFirst;
    uint32_t stateFp32;
    uint32_t varlen;
    uint32_t hasALog;
    uint32_t hasDtBias;
    uint32_t reserved;
    uint32_t writeH;
};

static_assert(sizeof(FusedTilingData) % sizeof(uint64_t) == 0,
              "sparse tiling must preserve 64-bit alignment");

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_FUSED_TILING_H
