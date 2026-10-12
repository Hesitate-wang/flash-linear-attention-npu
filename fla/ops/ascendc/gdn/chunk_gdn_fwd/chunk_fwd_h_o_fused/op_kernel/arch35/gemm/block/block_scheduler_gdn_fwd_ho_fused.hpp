/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_GEMM_SCHEDULER_GDN_FWD_HO_FUSED_HPP
#define CATLASS_GEMM_SCHEDULER_GDN_FWD_HO_FUSED_HPP

#include "../../../chunk_fwd_h_o_fused_struct.h"
#include "block_scheduler_gdn_fwd_h.hpp"
#include "block_scheduler_gdn_fwd_o.hpp"

namespace Catlass::Gemm::Block {

// Per-chunk offsets of the fused single-core schedule.  Both views describe the
// same (task, chunk) pair, so the FwdH and FwdO stages of one chunk can be
// issued back to back from a single driver.
struct GDNFwdHOFusedChunkOffsets {
    GDNFwdHOffsets h;
    GDNFwdOOffsets o;
    bool isFinalChunk{false};
};

// One MIX core owns one (batch, value-head) task and walks its chunks strictly
// serially.  Every workspace uses a single slot per core, so the split-mode
// ping-pong stride ((coreIdx * STAGES) + slot) collapses to coreIdx.
struct BlockSchedulerGdnFwdHOFused {
    uint32_t shapeBatch{0};
    uint32_t seqlen{0};
    uint32_t kNumHead{0};
    uint32_t vNumHead{0};
    uint32_t kHeadDim{0};
    uint32_t vHeadDim{0};
    uint32_t chunkSize{0};
    uint32_t numChunks{0};
    uint32_t headGroups{1};
    bool storeFinalState{false};
    bool useInitialState{false};

    uint32_t coreIdx{0};
    uint32_t taskNum{0};
    uint32_t taskIdx{0};
    uint32_t chunkIdx{0};
    bool isRunning{false};

    GDNFwdHOFusedChunkOffsets offsets;

    CATLASS_DEVICE
    BlockSchedulerGdnFwdHOFused() {}

    CATLASS_DEVICE
    void Init(GM_ADDR tiling, uint32_t mixedCoreIdx)
    {
        const __gm__ GDN::ChunkFwdHOFusedTilingData *tilingData =
            reinterpret_cast<const __gm__ GDN::ChunkFwdHOFusedTilingData *>(tiling);

        shapeBatch = static_cast<uint32_t>(tilingData->shapeBatch);
        seqlen = static_cast<uint32_t>(tilingData->seqlen);
        kNumHead = static_cast<uint32_t>(tilingData->kNumHead);
        vNumHead = static_cast<uint32_t>(tilingData->vNumHead);
        kHeadDim = static_cast<uint32_t>(tilingData->kHeadDim);
        vHeadDim = static_cast<uint32_t>(tilingData->vHeadDim);
        chunkSize = static_cast<uint32_t>(tilingData->chunkSize);
        numChunks = static_cast<uint32_t>(tilingData->numChunksPerBatch);
        storeFinalState = tilingData->storeFinalState;
        useInitialState = tilingData->useInitialState;
        headGroups = vNumHead / (kNumHead == 0 ? 1U : kNumHead);

        coreIdx = mixedCoreIdx;
        taskNum = static_cast<uint32_t>(tilingData->singleCoreTaskNum);
        taskIdx = mixedCoreIdx;
        chunkIdx = 0;
        isRunning = (seqlen != 0) && (chunkSize != 0) && (taskIdx < taskNum);
        if (isRunning) {
            UpdateOffsets();
        }
    }

    // Advances to the next chunk of the owned task.  Returns false once the
    // task's chunk range is exhausted; every caller must stop issuing stages
    // then.  The returned offsets always describe the current chunk, because
    // this schedule keeps a single slot and no stage lag.
    CATLASS_DEVICE
    bool Next()
    {
        if (unlikely(!isRunning)) {
            return false;
        }
        ++chunkIdx;
        if (chunkIdx >= numChunks) {
            isRunning = false;
            return false;
        }
        UpdateOffsets();
        return true;
    }

    CATLASS_DEVICE
    const GDNFwdHOffsets &GetHOffsets() const { return offsets.h; }

    CATLASS_DEVICE
    const GDNFwdOOffsets &GetOOffsets() const { return offsets.o; }

    CATLASS_DEVICE
    bool IsFinalChunk() const { return offsets.isFinalChunk; }

private:
    CATLASS_DEVICE
    void UpdateOffsets()
    {
        const uint32_t batchIdx = taskIdx / vNumHead;
        const uint32_t vHeadIdx = taskIdx % vNumHead;
        const uint32_t kHeadIdx = vHeadIdx / headGroups;
        const uint32_t tokenStart = chunkIdx * chunkSize;
        const bool isInitialChunk = (chunkIdx == 0);
        const bool isFinalChunk = (chunkIdx + 1 == numChunks);
        offsets.isFinalChunk = isFinalChunk;

        const uint32_t blockTokens =
            isFinalChunk ? (seqlen - tokenStart) : chunkSize;
        const uint32_t stateBlockSize = kHeadDim * vHeadDim;
        const uint32_t hChunkOffset =
            (batchIdx * vNumHead * numChunks + vHeadIdx * numChunks + chunkIdx) *
            stateBlockSize;

        // ---- FwdH view -------------------------------------------------
        GDNFwdHOffsets &h = offsets.h;
        h.isInitialState = isInitialChunk;
        h.isFinalState = isFinalChunk;
        h.initialStateOffset = (batchIdx * vNumHead + vHeadIdx) * stateBlockSize;
        h.finalStateOffset = h.initialStateOffset;
        h.hSrcOffset = hChunkOffset;
        h.hDstOffset = hChunkOffset + stateBlockSize;
        if (storeFinalState && isFinalChunk) {
            h.hDstOffset = h.hSrcOffset;
        }
        h.uvOffset = (batchIdx * vNumHead * seqlen + vHeadIdx * seqlen + tokenStart) * vHeadDim;
        h.wkOffset = (batchIdx * kNumHead * seqlen + kHeadIdx * seqlen + tokenStart) * kHeadDim;
        h.wOffset = (batchIdx * vNumHead * seqlen + vHeadIdx * seqlen + tokenStart) * kHeadDim;
        h.gOffset = batchIdx * vNumHead * seqlen + vHeadIdx * seqlen + tokenStart;
        h.gkOffset = h.gOffset * kHeadDim;
        // Single slot per core: no ping-pong stride, no stream id.
        h.hWorkOffset = coreIdx * kHeadDim * vHeadDim;
        h.vWorkOffset = coreIdx * chunkSize * vHeadDim;
        h.kDecayWorkOffset = coreIdx * chunkSize * kHeadDim;
        h.vBlockOffset = 0;
        h.vBlockDim = vHeadDim;
        h.blockTokens = blockTokens;
        h.streamId = 0;
        h.batchIdx = batchIdx;
        h.headIdx = vHeadIdx;
        h.chunkIdx = chunkIdx;

        // ---- FwdO view -------------------------------------------------
        GDNFwdOOffsets &o = offsets.o;
        o.qkOffset =
            ((static_cast<int64_t>(batchIdx) * kNumHead + kHeadIdx) * seqlen + tokenStart) * kHeadDim;
        o.ovOffset =
            ((static_cast<int64_t>(batchIdx) * vNumHead + vHeadIdx) * seqlen + tokenStart) * vHeadDim;
        o.hOffset = hChunkOffset;
        o.gOffset = (batchIdx * vNumHead + vHeadIdx) * seqlen + tokenStart;
        o.attnWorkOffset = static_cast<int64_t>(coreIdx) * chunkSize * chunkSize;
        o.hvWorkOffset = static_cast<int64_t>(coreIdx) * chunkSize * vHeadDim;
        o.vBlockOffset = 0;
        o.vBlockDim = vHeadDim;
        o.isFinalState = isFinalChunk;
        o.blockTokens = blockTokens;
        o.batchIdx = batchIdx;
        o.headIdx = vHeadIdx;
        o.chunkIdx = chunkIdx;
    }
};

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_SCHEDULER_GDN_FWD_HO_FUSED_HPP
