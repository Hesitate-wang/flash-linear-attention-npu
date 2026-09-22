/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Ascend 950 exp2 O-stage tiling projection.
 */

#ifndef CHUNK_FWD_H_O_FUSED_A5_O_STAGE_STRUCT_H
#define CHUNK_FWD_H_O_FUSED_A5_O_STAGE_STRUCT_H

#include <cstdint>

namespace GDN {

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
    int64_t producerCoreNum;
    int64_t consumerCoreBase;
    int64_t activeCoreNum;
};

} // namespace GDN

#endif // CHUNK_FWD_H_O_FUSED_A5_O_STAGE_STRUCT_H
