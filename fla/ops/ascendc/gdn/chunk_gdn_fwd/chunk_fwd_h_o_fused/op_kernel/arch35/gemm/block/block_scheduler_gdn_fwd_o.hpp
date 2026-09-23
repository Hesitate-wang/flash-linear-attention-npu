/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_GEMM_SCHEDULER_GDN_FWD_O_ARCH35_HPP
#define CATLASS_GEMM_SCHEDULER_GDN_FWD_O_ARCH35_HPP

#include "../../../chunk_fwd_o_struct.h"

constexpr uint32_t GDN_FWD_O_PING_PONG_STAGES = 2;
constexpr uint32_t GDN_FWD_HO_CONSUMERS_PER_PAIR = 1;

namespace Catlass::Gemm::Block {


struct GDNFwdOOffsets {
    int64_t qkOffset;
    int64_t ovOffset;
    int64_t hOffset;
    int64_t gOffset;
    int64_t attnWorkOffset;
    int64_t hvWorkOffset;
    uint32_t vBlockOffset;
    uint32_t vBlockDim;
    bool isFinalState;
    uint32_t blockTokens;
    // for debug
    uint32_t batchIdx;
    uint32_t headIdx;
    uint32_t chunkIdx;

};

struct BlockSchedulerGdnFwdO {
    uint32_t shapeBatch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    uint32_t vHeadDim;
    uint32_t chunkSize;
    uint32_t isVariedLen;
    uint32_t tokenBatch;
    uint32_t numChunks{0};
    uint32_t vBlockSize{128};

    uint32_t taskIdx;
    uint32_t cubeCoreIdx;
    uint32_t cubeCoreNum;
    uint32_t workspaceCoreIdx{0};
    uint32_t taskNum;
    uint32_t headGroups;

    bool isRunning;
    bool chunkPipeline{false};
    bool taskAffinity{false};
    bool processNewTask {true};
    bool firstLoop {true};
    bool lastLoop {false};
    GDNFwdOOffsets offsets[GDN_FWD_O_PING_PONG_STAGES];
    int32_t currStage{GDN_FWD_O_PING_PONG_STAGES - 1};

    uint32_t baseTaskIdx;
    uint32_t chunkIdx;
    uint32_t headInnerIdx;
    uint32_t vHeadIdx;
    uint32_t kHeadIdx;
    uint32_t shapeBatchIdx;
    uint32_t tokenBatchIdx;

    uint32_t batchChunkIdx;
    uint32_t batchChunkStartIdx;
    uint32_t tokenOffset;
    uint32_t batchChunks;
    uint32_t batchTokens;
    uint32_t pipelineHeadIdx{0};
    uint32_t pipelineLaneIdx{0};
    uint32_t pipelineHTaskBase{0};
    uint32_t pipelineChunkIdx{0};
    uint32_t pipelineLaneCount{0};
    uint32_t pipelineTaskStride{0};
    uint32_t pipelineStageCount{GDN_FWD_O_PING_PONG_STAGES};
    uint32_t initialStageCount{0};

    AscendC::GlobalTensor<int64_t> gmSeqlen;
    AscendC::GlobalTensor<int64_t> gmChunkOffsets;

    Arch::CrossCoreFlag cube1Done[GDN_FWD_O_PING_PONG_STAGES] = {0, 1};
    Arch::CrossCoreFlag vec1Done[GDN_FWD_O_PING_PONG_STAGES] = {2, 3};
    Arch::CrossCoreFlag cube3Done[GDN_FWD_O_PING_PONG_STAGES] = {4, 5};
    Arch::CrossCoreFlag vec2Done[GDN_FWD_O_PING_PONG_STAGES] = {6, 7};

    CATLASS_DEVICE
    BlockSchedulerGdnFwdO() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, const GDN::ChunkFwdHOFusedOStageTilingData *tilingData,
              uint32_t coreIdx, uint32_t coreNum, bool enableChunkPipeline = false,
              bool enableTaskAffinity = false) {
        shapeBatch = tilingData->shapeBatch;
        seqlen = tilingData->seqlen;
        kNumHead = tilingData->kNumHead;
        vNumHead = tilingData->vNumHead;
        kHeadDim = tilingData->kHeadDim;
        vHeadDim = tilingData->vHeadDim;
        chunkSize = tilingData->chunkSize;
        isVariedLen = tilingData->isVariedLen;
        tokenBatch = tilingData->tokenBatch;

        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmChunkOffsets.SetGlobalBuffer((__gm__ int64_t *)chunk_offsets);

        if (isVariedLen) {
            for (uint32_t b = 1; b <= tokenBatch; b++) {
                numChunks += (gmSeqlen.GetValue(b) - gmSeqlen.GetValue(b - 1) + chunkSize - 1) / chunkSize;
            }
        } else {
            numChunks = (seqlen + chunkSize - 1) / chunkSize;
        }

        cubeCoreIdx = coreIdx;
        cubeCoreNum = coreNum;
        vBlockSize = vHeadDim;
        taskNum = shapeBatch * numChunks * vNumHead;
        headGroups = vNumHead / kNumHead;
        chunkPipeline = enableChunkPipeline;
        taskAffinity = enableTaskAffinity;
        if (chunkPipeline) {
            const uint32_t producerCoreNum = tilingData->producerCoreNum;
            const uint32_t consumerCoreBegin = tilingData->consumerCoreBase;
            const uint32_t consumerCoreEnd = consumerCoreBegin + producerCoreNum;
            if (cubeCoreIdx < consumerCoreBegin || cubeCoreIdx >= consumerCoreEnd) {
                taskIdx = taskNum;
                initialStageCount = 0;
                isRunning = false;
            } else {
                const uint32_t consumerIdx = cubeCoreIdx - consumerCoreBegin;
                pipelineLaneIdx = 0;
                pipelineChunkIdx = 0;
                pipelineHTaskBase = consumerIdx * GDN_FWD_O_PING_PONG_STAGES;
                pipelineTaskStride = producerCoreNum * GDN_FWD_O_PING_PONG_STAGES;
                workspaceCoreIdx = consumerIdx;
                const uint32_t hTaskNum = shapeBatch * vNumHead;
                const uint32_t remainingTasks = hTaskNum > pipelineHTaskBase
                                                    ? hTaskNum - pipelineHTaskBase
                                                    : 0;
                // Keep two pipeline slots even when only one head is valid.
                // The slots represent reusable chunk buffers, not head lanes.
                pipelineStageCount = GDN_FWD_O_PING_PONG_STAGES;
                pipelineLaneCount = remainingTasks < GDN_FWD_O_PING_PONG_STAGES
                                        ? remainingTasks
                                        : GDN_FWD_O_PING_PONG_STAGES;
                initialStageCount = remainingTasks < GDN_FWD_O_PING_PONG_STAGES
                                        ? remainingTasks
                                        : GDN_FWD_O_PING_PONG_STAGES;
                currStage = pipelineStageCount - 1;
                isRunning = pipelineHTaskBase < shapeBatch * vNumHead;
            }
        } else if (taskAffinity) {
            taskIdx = 0;
            initialStageCount = GDN_FWD_O_PING_PONG_STAGES;
            isRunning = taskNum > 0 && cubeCoreNum > 0;
        } else {
            taskIdx = cubeCoreIdx * GDN_FWD_O_PING_PONG_STAGES;
            const uint32_t remainingTasks = taskNum > taskIdx ? taskNum - taskIdx : 0;
            initialStageCount = remainingTasks < GDN_FWD_O_PING_PONG_STAGES
                                    ? remainingTasks
                                    : GDN_FWD_O_PING_PONG_STAGES;
            isRunning = taskIdx < taskNum;
        }

    }

    CATLASS_DEVICE
    uint32_t GetCompactSequenceIdx(uint32_t rawSequenceIdx) const {
        uint32_t compactSequenceIdx = 0;
        for (uint32_t sequence = 0; sequence < rawSequenceIdx; ++sequence) {
            compactSequenceIdx += gmSeqlen.GetValue(sequence + 1) > gmSeqlen.GetValue(sequence);
        }
        return compactSequenceIdx;
    }

    CATLASS_DEVICE
    bool IsTaskOwnedByCore(uint32_t candidateTaskIdx) const {
        const uint32_t candidateBatchIdx = candidateTaskIdx / (numChunks * vNumHead);
        const uint32_t candidateChunkIdx =
            (candidateTaskIdx - candidateBatchIdx * numChunks * vNumHead) / vNumHead;
        const uint32_t candidateHeadIdx = candidateTaskIdx % vNumHead;
        const uint32_t hBatchIdx = isVariedLen
                                       ? GetCompactSequenceIdx(gmChunkOffsets.GetValue(2 * candidateChunkIdx))
                                       : candidateBatchIdx;
        const uint32_t hTaskIdx = hBatchIdx * vNumHead + candidateHeadIdx;
        // FwdH assigns task wave * coreNum + coreIdx, so every dependent O
        // chunk must stay on hTaskIdx % coreNum. The old two-tasks-per-core
        // formula no longer matched FwdH after its wave scheduler was added.
        return cubeCoreNum > 0 && (hTaskIdx % cubeCoreNum) == cubeCoreIdx;
    }

    // Dense task-affinity ownership depends only on batch/head, not chunk.
    // Jump directly to the next owned head while preserving the original
    // ascending task order.  Varlen keeps the guarded scan above because its
    // compact batch mapping is read from GM.
    CATLASS_DEVICE
    uint32_t FindNextOwnedDenseTask(uint32_t candidateTaskIdx) const {
        if (candidateTaskIdx >= taskNum || cubeCoreNum == 0 || vNumHead == 0 || numChunks == 0) {
            return taskNum;
        }
        while (candidateTaskIdx < taskNum) {
            const uint32_t batchChunk = candidateTaskIdx / vNumHead;
            const uint32_t batch = batchChunk / numChunks;
            const uint32_t headBegin = candidateTaskIdx - batchChunk * vNumHead;
            const uint32_t ownerBase = (batch * vNumHead) % cubeCoreNum;
            uint32_t head = (cubeCoreIdx + cubeCoreNum - ownerBase) % cubeCoreNum;
            if (head < headBegin) {
                const uint32_t distance = headBegin - head;
                head += ((distance + cubeCoreNum - 1) / cubeCoreNum) * cubeCoreNum;
            }
            if (head < vNumHead) {
                return batchChunk * vNumHead + head;
            }
            candidateTaskIdx = (batchChunk + 1) * vNumHead;
        }
        return taskNum;
    }

    CATLASS_DEVICE
    void InitTask() {
        uint32_t curTaskIdx;
        if (chunkPipeline) {
            const uint32_t hTaskNum = shapeBatch * vNumHead;
            while (pipelineHTaskBase < hTaskNum && pipelineLaneCount == 0) {
                pipelineHTaskBase += pipelineTaskStride;
                pipelineLaneIdx = 0;
                pipelineChunkIdx = 0;
                const uint32_t remainingTasks = hTaskNum > pipelineHTaskBase
                                                    ? hTaskNum - pipelineHTaskBase
                                                    : 0;
                pipelineLaneCount = remainingTasks < GDN_FWD_O_PING_PONG_STAGES
                                        ? remainingTasks
                                        : GDN_FWD_O_PING_PONG_STAGES;
            }
            if (unlikely(pipelineHTaskBase >= hTaskNum)) {
                isRunning = false;
                currStage = (currStage + 1) % pipelineStageCount;
                return;
            }
            const uint32_t hTaskIdx = pipelineHTaskBase + pipelineLaneIdx;
            const uint32_t pipelineBatchIdx = hTaskIdx / vNumHead;
            pipelineHeadIdx = hTaskIdx % vNumHead;
            curTaskIdx = pipelineBatchIdx * numChunks * vNumHead +
                         pipelineChunkIdx * vNumHead + pipelineHeadIdx;
            pipelineLaneIdx += 1;
            if (pipelineLaneIdx == pipelineLaneCount) {
                pipelineLaneIdx = 0;
                pipelineChunkIdx += 1;
                if (pipelineChunkIdx == numChunks) {
                    pipelineChunkIdx = 0;
                    pipelineHTaskBase += pipelineTaskStride;
                    const uint32_t remainingTasks = hTaskNum > pipelineHTaskBase
                                                        ? hTaskNum - pipelineHTaskBase
                                                        : 0;
                    pipelineLaneCount = remainingTasks < GDN_FWD_O_PING_PONG_STAGES
                                            ? remainingTasks
                                            : GDN_FWD_O_PING_PONG_STAGES;
                }
            }
        } else if (taskAffinity) {
            taskIdx = FindNextOwnedDenseTask(taskIdx);
            if (unlikely(taskIdx >= taskNum)) {
                isRunning = false;
                currStage = (currStage + 1) % GDN_FWD_O_PING_PONG_STAGES;
                return;
            }
            curTaskIdx = taskIdx++;
        } else {
            if (processNewTask) {
                headInnerIdx = 0;
                baseTaskIdx = taskIdx;
            } else {
                headInnerIdx = (headInnerIdx + 1) % GDN_FWD_O_PING_PONG_STAGES;
            }
            curTaskIdx = baseTaskIdx + headInnerIdx;
            if (unlikely(curTaskIdx >= taskNum)) {
                isRunning = false;
                processNewTask = true;
                currStage = (currStage + 1) % GDN_FWD_O_PING_PONG_STAGES;
                return;
            }
        }

        shapeBatchIdx = curTaskIdx / (numChunks * vNumHead);
        chunkIdx = (curTaskIdx - shapeBatchIdx * numChunks * vNumHead) / vNumHead;
        vHeadIdx = curTaskIdx % vNumHead;
        kHeadIdx = vHeadIdx / headGroups;
        tokenBatchIdx = isVariedLen ? gmChunkOffsets.GetValue(2 * chunkIdx) : 0;
        batchChunkIdx = isVariedLen ? gmChunkOffsets.GetValue(2 * chunkIdx + 1) : chunkIdx;
        batchChunkStartIdx = chunkIdx - batchChunkIdx;
        tokenOffset = isVariedLen ? gmSeqlen.GetValue(tokenBatchIdx) : 0;
        batchTokens = isVariedLen ? (gmSeqlen.GetValue(tokenBatchIdx + 1) - tokenOffset) : seqlen;
        uint32_t vBlockOffset = 0;
        uint32_t vBlockDim = vBlockSize;
        const int64_t tokenStart = static_cast<int64_t>(tokenOffset) +
                                   static_cast<int64_t>(batchChunkIdx) * chunkSize;
        const int64_t qkRowOffset = (static_cast<int64_t>(shapeBatchIdx) * kNumHead + kHeadIdx) * seqlen +
                                    tokenStart;
        const int64_t ovRowOffset = (static_cast<int64_t>(shapeBatchIdx) * vNumHead + vHeadIdx) * seqlen +
                                    tokenStart;
        const int64_t hBlockOffset = (static_cast<int64_t>(shapeBatchIdx) * vNumHead * numChunks +
                                      static_cast<int64_t>(vHeadIdx) * numChunks + chunkIdx) *
                                     kHeadDim;
        const int64_t workStageOffset =
            static_cast<int64_t>(chunkPipeline ? workspaceCoreIdx : cubeCoreIdx) *
                GDN_FWD_O_PING_PONG_STAGES + currStage;
        offsets[currStage].qkOffset = qkRowOffset * kHeadDim;
        offsets[currStage].ovOffset = ovRowOffset * vHeadDim + vBlockOffset;
        offsets[currStage].hOffset = hBlockOffset * vHeadDim + vBlockOffset;
        offsets[currStage].gOffset = ovRowOffset;
        offsets[currStage].attnWorkOffset = workStageOffset * chunkSize * chunkSize;
        offsets[currStage].hvWorkOffset = workStageOffset * chunkSize * vBlockSize;
        offsets[currStage].vBlockOffset = vBlockOffset;
        offsets[currStage].vBlockDim = vBlockDim;
        offsets[currStage].isFinalState = chunkIdx == (numChunks - 1) || (isVariedLen && gmChunkOffsets.GetValue(2 * chunkIdx + 3) == 0);
        offsets[currStage].blockTokens = offsets[currStage].isFinalState ? (batchTokens - batchChunkIdx * chunkSize) : chunkSize;
        offsets[currStage].batchIdx = shapeBatchIdx;
        offsets[currStage].headIdx = vHeadIdx;
        offsets[currStage].chunkIdx = chunkIdx;

        if (!chunkPipeline && !taskAffinity) {
            processNewTask = headInnerIdx == GDN_FWD_O_PING_PONG_STAGES - 1;
            if (processNewTask) {
                taskIdx += GDN_FWD_O_PING_PONG_STAGES * cubeCoreNum;
            }
        }

        currStage = (currStage + 1) % pipelineStageCount;
    }

    CATLASS_DEVICE
    uint32_t GetCurStageId() const {
        return (currStage + pipelineStageCount - 1) % pipelineStageCount;
    }

    CATLASS_DEVICE
    uint32_t GetInitialStageCount() const {
        return initialStageCount;
    }

    CATLASS_DEVICE
    uint32_t GetPrevStageId() const {
        return pipelineStageCount == 1
                   ? 0
                   : (currStage + pipelineStageCount - 2) % pipelineStageCount;
    }


};

struct BlockSchedulerGdnFwdOCube : public BlockSchedulerGdnFwdO {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdOCube() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, const GDN::ChunkFwdHOFusedOStageTilingData *tilingData,
              bool enableChunkPipeline = false, bool enableTaskAffinity = false) {
        BlockSchedulerGdnFwdO::Init(cu_seqlens, chunk_offsets, tilingData, AscendC::GetBlockIdx(),
                                    AscendC::GetBlockNum(), enableChunkPipeline, enableTaskAffinity);
    }

    CATLASS_DEVICE
    bool NeedProcessCube1() {
        return true;
    }

    CATLASS_DEVICE
    GDNFwdOOffsets& GetCube1Offsets() {
        return offsets[GetCurStageId()];
    }

    CATLASS_DEVICE
    GemmCoord GetCube1Shape() {
        GDNFwdOOffsets& cube1Offsets = GetCube1Offsets();
        return GemmCoord{cube1Offsets.blockTokens, cube1Offsets.blockTokens, kHeadDim};
    }

    CATLASS_DEVICE
    bool NeedProcessCube23() {
        if (unlikely(firstLoop)) {
            firstLoop = false;
            return false;
        }
        return true;
    }

    CATLASS_DEVICE
    GDNFwdOOffsets& GetCube23Offsets() {
        return offsets[GetPrevStageId()];
    }

    CATLASS_DEVICE
    GemmCoord GetCube2Shape() {
        GDNFwdOOffsets& cube2Offsets = GetCube23Offsets();
        return GemmCoord{cube2Offsets.blockTokens, cube2Offsets.vBlockDim, kHeadDim};
    }

    CATLASS_DEVICE
    GemmCoord GetCube3Shape() {
        GDNFwdOOffsets& cube2Offsets = GetCube23Offsets();
        return GemmCoord{cube2Offsets.blockTokens, cube2Offsets.vBlockDim, cube2Offsets.blockTokens};
    }

};

struct BlockSchedulerGdnFwdOVec : public BlockSchedulerGdnFwdO {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdOVec() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, const GDN::ChunkFwdHOFusedOStageTilingData *tilingData,
              bool enableChunkPipeline = false, bool enableTaskAffinity = false) {
        BlockSchedulerGdnFwdO::Init(cu_seqlens, chunk_offsets, tilingData,
                                    AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(), AscendC::GetBlockNum(),
                                    enableChunkPipeline, enableTaskAffinity);
    }

    CATLASS_DEVICE
    bool NeedProcessVec1() {
        return isRunning;
    }

    CATLASS_DEVICE
    bool NeedProcessVec2() {
        if (unlikely(firstLoop)) {
            firstLoop = false;
            return false;
        }
        return true;
    }

    CATLASS_DEVICE
    GDNFwdOOffsets& GetVec1Offsets() {
        return offsets[GetCurStageId()];
    }

    CATLASS_DEVICE
    GDNFwdOOffsets& GetVec2Offsets() {
        return offsets[GetPrevStageId()];
    }

};

}  // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_SCHEDULER_GDN_FWD_O_ARCH35_HPP
