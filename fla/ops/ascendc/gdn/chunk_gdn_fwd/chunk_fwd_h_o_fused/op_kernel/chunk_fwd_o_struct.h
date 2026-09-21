/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_fwd_o_struct.h
 * \brief Operator-local O-stage projection of the fused tiling data.
 */

#ifndef CHUNK_FWD_H_O_FUSED_O_STAGE_STRUCT_H
#define CHUNK_FWD_H_O_FUSED_O_STAGE_STRUCT_H

#include <cstdint>

namespace GDN {

struct ChunkFwdHOFusedOStageTilingData {
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
    float scale;
    int64_t pipelineSyncWorkspaceOffset;
    int64_t producerCoreNum;
    int64_t consumerCoreBase;
    int64_t activeCoreNum;
};

// Operator-local projection consumed by the copied Ascend 950 ChunkFwdO
// implementation. Keep this layout aligned with ChunkFwdO's kernel struct;
// FillOTiling in chunk_fwd_h_o_fused_a5.hpp adapts the fused tiling payload.
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

#endif // CHUNK_FWD_H_O_FUSED_O_STAGE_STRUCT_H
