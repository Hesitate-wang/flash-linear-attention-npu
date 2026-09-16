/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#pragma once

#include <register/tilingdata_base.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(ChunkFwdHOFusedTilingData)
// Keep this prefix byte-for-byte compatible with the fused kernel's local H view.
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, seqlen);
TILING_DATA_FIELD_DEF(int64_t, kNumHead);
TILING_DATA_FIELD_DEF(int64_t, vNumHead);
TILING_DATA_FIELD_DEF(int64_t, kHeadDim);
TILING_DATA_FIELD_DEF(int64_t, vHeadDim);
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
TILING_DATA_FIELD_DEF(bool, useInitialState);
TILING_DATA_FIELD_DEF(bool, storeFinalState);
TILING_DATA_FIELD_DEF(int64_t, dataType);
TILING_DATA_FIELD_DEF(int64_t, gDataType);
TILING_DATA_FIELD_DEF(int64_t, stateDataType);
TILING_DATA_FIELD_DEF(int64_t, isVariedLen);
TILING_DATA_FIELD_DEF(int64_t, shapeBatch);
TILING_DATA_FIELD_DEF(int64_t, tokenBatch);
TILING_DATA_FIELD_DEF(bool, useG);
TILING_DATA_FIELD_DEF(bool, useGk);
TILING_DATA_FIELD_DEF(int64_t, vWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, vUpdateWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, kDecayWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, hWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, numSeqWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, numChunksWorkspaceOffset);

TILING_DATA_FIELD_DEF(bool, useExp2);
TILING_DATA_FIELD_DEF(int64_t, outputLayout);
TILING_DATA_FIELD_DEF(float, scale);
TILING_DATA_FIELD_DEF(int64_t, chunkNum);
TILING_DATA_FIELD_DEF(int64_t, numChunksPerBatch);
TILING_DATA_FIELD_DEF(int64_t, hvPerHk);
TILING_DATA_FIELD_DEF(int64_t, taskGroupSize);
TILING_DATA_FIELD_DEF(int64_t, producerCoreNum);
TILING_DATA_FIELD_DEF(int64_t, consumerCoreBase);
TILING_DATA_FIELD_DEF(int64_t, activeCoreNum);
TILING_DATA_FIELD_DEF(int64_t, handoffHWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, handoffVWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, pipelineSyncWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, pipelineEventCount);
TILING_DATA_FIELD_DEF(int64_t, oVWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, oHWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, oAttnWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, oAfterMaskWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, oMaskWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, oAPrimeWorkspaceOffset);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkFwdHOFused, ChunkFwdHOFusedTilingData)

struct ChunkFwdHOFusedCompileInfo {};
} // namespace optiling
