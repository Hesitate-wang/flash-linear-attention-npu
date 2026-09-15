/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#pragma once

#include <register/tilingdata_base.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(ChunkFwdHOFusedTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, shapeBatch);
TILING_DATA_FIELD_DEF(int64_t, tokenBatch);
TILING_DATA_FIELD_DEF(int64_t, seqlen);
TILING_DATA_FIELD_DEF(int64_t, kNumHead);
TILING_DATA_FIELD_DEF(int64_t, vNumHead);
TILING_DATA_FIELD_DEF(int64_t, kHeadDim);
TILING_DATA_FIELD_DEF(int64_t, vHeadDim);
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
TILING_DATA_FIELD_DEF(int64_t, chunkNum);
TILING_DATA_FIELD_DEF(int64_t, numChunksPerBatch);
TILING_DATA_FIELD_DEF(int64_t, hvPerHk);
TILING_DATA_FIELD_DEF(int64_t, taskGroupSize);
TILING_DATA_FIELD_DEF(int64_t, dataType);
TILING_DATA_FIELD_DEF(int64_t, gDataType);
TILING_DATA_FIELD_DEF(int64_t, stateDataType);
TILING_DATA_FIELD_DEF(int64_t, isVariedLen);
TILING_DATA_FIELD_DEF(bool, useInitialState);
TILING_DATA_FIELD_DEF(bool, storeFinalState);
TILING_DATA_FIELD_DEF(bool, useGk);
TILING_DATA_FIELD_DEF(bool, useExp2);
TILING_DATA_FIELD_DEF(int64_t, outputLayout);
TILING_DATA_FIELD_DEF(float, scale);
TILING_DATA_FIELD_DEF(int64_t, hVWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, hVUpdateWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, hKDecayWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, hStateWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, hNumSeqWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, hNumChunksWorkspaceOffset);
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
