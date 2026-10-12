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
enum class ChunkFwdHOFusedTilingKey : uint64_t { V128_EXP = 1, V256_EXP = 2 };
enum class ChunkFwdHOFusedExecutionMode : int64_t {
    // T FwdH producers + T FwdO consumers on split physical cores, two on-chip
    // slots per core (ping-pong), cross-core H/O handoff per chunk.
    SPLIT_CORE_PIPELINE = 0,
    // One MIX core per logical task (blockDim = T): FwdH and FwdO run on the
    // same core, every chunk is processed serially and inside one chunk the
    // FwdH/FwdO stages alternate on the single on-chip slot (no ping-pong).
    SINGLE_CORE_CHUNK_FUSED = 2,
};

// SINGLE_CORE_CHUNK_FUSED schedule constants.
constexpr int64_t CHUNK_FWD_HO_FUSED_SLOT_COUNT = 1;
constexpr int64_t CHUNK_FWD_HO_FUSED_UB_DATA_LIMIT = 248 * 1024;
constexpr int64_t CHUNK_FWD_HO_FUSED_UB_RESERVED_TAIL = 8 * 1024;
constexpr int64_t CHUNK_FWD_HO_AIV_PER_MIXED_CORE = 2;
constexpr int64_t CHUNK_FWD_HO_TASK_LANES_PER_CORE = 1;
constexpr int64_t CHUNK_FWD_HO_H_READY_EVENT_BASE = 0;
constexpr int64_t CHUNK_FWD_HO_V_READY_EVENT_BASE = CHUNK_FWD_HO_TASK_LANES_PER_CORE;
constexpr int64_t CHUNK_FWD_HO_READY_EVENT_COUNT = 2 * CHUNK_FWD_HO_TASK_LANES_PER_CORE;
constexpr int64_t CHUNK_FWD_HO_IB_EVENT_COUNT = CHUNK_FWD_HO_READY_EVENT_COUNT;
constexpr int64_t CHUNK_FWD_HO_IB_WORDS_PER_EVENT = 8;

// Plain kernel-side mirror of op_host/chunk_fwd_h_o_fused_tiling.h.
struct ChunkFwdHOFusedTilingData {
    int64_t batch;
    int64_t seqlen;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    bool useInitialState;
    bool storeFinalState;
    int64_t dataType;
    int64_t gDataType;
    int64_t stateDataType;
    int64_t isVariedLen;
    int64_t shapeBatch;
    int64_t tokenBatch;
    bool useG;
    bool useGk;
    int64_t vWorkspaceOffset;
    int64_t vUpdateWorkspaceOffset;
    int64_t kDecayWorkspaceOffset;
    int64_t hWorkspaceOffset;
    int64_t numSeqWorkspaceOffset;
    int64_t numChunksWorkspaceOffset;
    bool useExp2;
    int64_t outputLayout;
    float scale;
    int64_t chunkNum;
    int64_t numChunksPerBatch;
    int64_t hvPerHk;
    int64_t taskGroupSize;
    int64_t producerCoreNum;
    int64_t consumerCoreBase;
    int64_t activeCoreNum;
    int64_t handoffHWorkspaceOffset;
    int64_t handoffVWorkspaceOffset;
    int64_t pipelineSyncWorkspaceOffset;
    int64_t pipelineEventCount;
    int64_t oVWorkspaceOffset;
    int64_t oHWorkspaceOffset;
    int64_t oAttnWorkspaceOffset;
    int64_t oAfterMaskWorkspaceOffset;
    int64_t oMaskWorkspaceOffset;
    int64_t oAPrimeWorkspaceOffset;
    // 0: split H/O core pipeline; 2: one MIX core per logical task with the
    // chunk-serial fused H/O schedule.
    int64_t executionMode;
    // Number of logical tasks owned by the single-core schedule (T = B * HV).
    int64_t singleCoreTaskNum;
};

} // namespace GDN
#endif
