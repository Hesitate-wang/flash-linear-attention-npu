/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef CHUNK_FWD_H_O_FUSED_STRUCT_H
#define CHUNK_FWD_H_O_FUSED_STRUCT_H

#include <cstdint>

namespace GDN {

enum class ChunkFwdHOFusedDtype : int64_t { FP16 = 0, BF16 = 1, FP32 = 2 };
enum class ChunkFwdHOFusedOutputLayout : int64_t { BNSD = 0, BSND = 1, TND = 2, NTD = 3 };
enum class ChunkFwdHOFusedTilingKey : uint64_t {
    V128_EXP = 1,
    V256_EXP = 2,
    V128_EXP2 = 3,
    V256_EXP2 = 4,
};

// Plain kernel-side mirror of op_host/chunk_fwd_h_o_fused_tiling.h.
struct ChunkFwdHOFusedTilingData {
    int64_t batch;
    int64_t shapeBatch;
    int64_t tokenBatch;
    int64_t seqlen;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    int64_t chunkNum;
    int64_t numChunksPerBatch;
    int64_t hvPerHk;
    int64_t taskGroupSize;
    int64_t dataType;
    int64_t gDataType;
    int64_t stateDataType;
    int64_t isVariedLen;
    bool useInitialState;
    bool storeFinalState;
    bool useGk;
    bool useExp2;
    int64_t outputLayout;
    float scale;
    int64_t hVWorkspaceOffset;
    int64_t hVUpdateWorkspaceOffset;
    int64_t hKDecayWorkspaceOffset;
    int64_t hStateWorkspaceOffset;
    int64_t hNumSeqWorkspaceOffset;
    int64_t hNumChunksWorkspaceOffset;
    int64_t oVWorkspaceOffset;
    int64_t oHWorkspaceOffset;
    int64_t oAttnWorkspaceOffset;
    int64_t oAfterMaskWorkspaceOffset;
    int64_t oMaskWorkspaceOffset;
    int64_t oAPrimeWorkspaceOffset;
};

} // namespace GDN
#endif
