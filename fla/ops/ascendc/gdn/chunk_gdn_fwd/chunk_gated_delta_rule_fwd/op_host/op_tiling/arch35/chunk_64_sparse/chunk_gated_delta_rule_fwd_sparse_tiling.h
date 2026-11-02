/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_CHUNK64_SPARSE_TILING_H
#define GDN_CHUNK64_SPARSE_TILING_H

#include <cstdint>
#include <limits>

namespace optiling {

struct ChunkGatedDeltaRuleFwdSparseTilingData {
    uint32_t batch = 0;
    uint32_t keyHeads = 0;
    uint32_t valueHeads = 0;
    uint32_t tokens = 0;
    uint32_t chunks = 0;
    uint32_t aicCores = 0;
    uint32_t producerGroups = 0;
    uint32_t consumerGroups = 0;
    uint64_t normalizedOffset = 0;
    uint64_t coefficientOffset = 0;
    uint64_t stateOffset = 0;
    uint64_t outputOffset = 0;
    uint64_t workspaceBytes = 0;
};

struct ChunkGatedDeltaRuleFwdSparseWorkspace {
    uint64_t normalizedBytes;
    uint64_t cumsumBytes;
    uint64_t stateBytes;
    uint64_t outputBytes;
    uint64_t prepareBytes;
    uint64_t hBytes;
    uint64_t oBytes;
    uint64_t totalBytes;
};

struct ChunkGatedDeltaRuleFwdSparsePlan {
    uint64_t launchGroups;
    uint64_t physicalBatch;
    uint64_t tokens;
    uint64_t totalChunks;
    uint64_t keyHeads;
    uint64_t valueHeads;
    uint64_t headDim;
    uint64_t hvPerHk;
    uint64_t prepareOffset;
    uint64_t hOffset;
    uint64_t oOffset;
    uint64_t normalizedOffset;
    uint64_t cumsumOffset;
    uint64_t stateOffset;
    uint64_t outputOffset;
    uint64_t workspaceBytes;
};

inline bool IsChunkGatedDeltaRuleFwdSparseLowParallelism(
    uint64_t batch, uint64_t valueHeads, uint64_t aicCores)
{
    if (batch == 0 || valueHeads == 0 || aicCores == 0 ||
        batch > std::numeric_limits<uint64_t>::max() / valueHeads) {
        return false;
    }
    return batch * valueHeads <= aicCores / 2;
}

inline bool CheckedAddU64(uint64_t lhs, uint64_t rhs, uint64_t *result)
{
    if (result == nullptr || rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

inline bool CheckedMulU64(uint64_t lhs, uint64_t rhs, uint64_t *result)
{
    if (result == nullptr || (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

inline bool AlignSparseWorkspace(uint64_t value, uint64_t alignment, uint64_t *result)
{
    if (alignment == 0 || result == nullptr) {
        return false;
    }
    const uint64_t remainder = value % alignment;
    return remainder == 0 ? (*result = value, true)
                          : CheckedAddU64(value, alignment - remainder, result);
}

inline bool PlaceSparseWorkspaceRegion(uint64_t size, uint64_t alignment,
                                      uint64_t *cursor, uint64_t *offset)
{
    if (cursor == nullptr || offset == nullptr ||
        !AlignSparseWorkspace(*cursor, alignment, offset)) {
        return false;
    }
    return CheckedAddU64(*offset, size, cursor);
}

// Producer/consumer plan for the low-parallelism path. One chain = one (b,hv)
// pair; its chunks are produced by group c and consumed one iteration later by
// group chainCount + c. Both counts are upper bounds; the device trims them.
struct ChunkGatedDeltaRuleFwdPipelinePlan {
    uint32_t chainCount;
    uint32_t maxChunks;
    uint32_t producerGroups;
    uint32_t consumerGroups;
    uint64_t workspaceOffset;
    uint64_t workspaceBytes;
    uint64_t totalWorkspaceBytes;
};

// Each producer group may spill one 64 KiB intermediate tile; consumer groups
// keep their recurrent state on chip.
inline constexpr uint64_t kChunkGatedDeltaRuleFwdPipelineBytesPerChain = 64 * 1024;
inline constexpr uint64_t kChunkGatedDeltaRuleFwdPipelineMinWorkspace = 1024 * 1024;

inline bool BuildChunkGatedDeltaRuleFwdPipelinePlan(
    uint64_t batch, uint64_t valueHeads, uint64_t tokens,
    uint64_t sequenceCount, uint64_t chunkCount, bool isVarlen,
    uint64_t aicCores, uint64_t alignment,
    ChunkGatedDeltaRuleFwdPipelinePlan *plan)
{
    constexpr uint64_t kChunk = 64;
    if (plan == nullptr || alignment == 0 || valueHeads == 0 || aicCores == 0) {
        return false;
    }
    const uint64_t chainsPerSequence = isVarlen ? sequenceCount : batch;
    uint64_t chains = 0;
    if (chainsPerSequence == 0 || !CheckedMulU64(chainsPerSequence, valueHeads, &chains) ||
        chains > std::numeric_limits<uint32_t>::max() ||
        !IsChunkGatedDeltaRuleFwdSparseLowParallelism(chainsPerSequence, valueHeads, aicCores)) {
        return false;
    }
    uint64_t maxChunks = chunkCount;
    if (!isVarlen) {
        if (tokens == 0) {
            return false;
        }
        maxChunks = tokens / kChunk + (tokens % kChunk != 0);
    }
    if (maxChunks == 0 || maxChunks > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    uint64_t scratch = 0;
    if (!CheckedMulU64(chains, kChunkGatedDeltaRuleFwdPipelineBytesPerChain, &scratch)) {
        return false;
    }
    if (scratch < kChunkGatedDeltaRuleFwdPipelineMinWorkspace) {
        scratch = kChunkGatedDeltaRuleFwdPipelineMinWorkspace;
    }
    uint64_t cursor = 0;
    if (!PlaceSparseWorkspaceRegion(scratch, alignment, &cursor, &plan->workspaceOffset)) {
        return false;
    }
    plan->chainCount = static_cast<uint32_t>(chains);
    plan->maxChunks = static_cast<uint32_t>(maxChunks);
    plan->producerGroups = static_cast<uint32_t>(chains);
    plan->consumerGroups = static_cast<uint32_t>(chains);
    plan->workspaceBytes = scratch;
    plan->totalWorkspaceBytes = cursor;
    return true;
}

inline bool BuildChunkGatedDeltaRuleFwdSparseWorkspace(
    uint64_t launchGroups, uint64_t physicalBatch, uint64_t tokens,
    uint64_t totalChunks, uint64_t keyHeads, uint64_t valueHeads,
    uint64_t headDim, uint64_t inputElementBytes, uint64_t alignment,
    ChunkGatedDeltaRuleFwdSparsePlan *plan,
    ChunkGatedDeltaRuleFwdSparseWorkspace *workspace)
{
    constexpr uint64_t kChunk = 64;
    constexpr uint64_t kRoundHeadSlots = 4;
    constexpr uint64_t kPrepareBytesPerGroup = 64 * 1024;
    constexpr uint64_t kHReserveBytes = 16 * 1024 * 1024;
    constexpr uint64_t kOAprimeBytesPerGroup = 64 * 1024;
    if (plan == nullptr || workspace == nullptr || launchGroups == 0 ||
        physicalBatch == 0 || tokens == 0 || totalChunks == 0 ||
        keyHeads == 0 || valueHeads == 0 ||
        headDim != 128 || valueHeads % keyHeads != 0 ||
        valueHeads / keyHeads > 4 || inputElementBytes != 2 || alignment == 0) {
        return false;
    }

    uint64_t normalizedElements = 0;
    uint64_t normalizedBytes = 0;
    uint64_t cumsumElements = 0;
    uint64_t cumsumBytes = 0;
    uint64_t stateElements = 0;
    uint64_t stateBytes = 0;
    uint64_t outputElements = 0;
    uint64_t outputBytes = 0;
    uint64_t prepareBytes = 0;
    uint64_t hBytes = 0;
    uint64_t oBytes = 0;
    uint64_t tmp = 0;
    uint64_t vScratchPerGroup = 0;
    uint64_t updateScratchPerGroup = 0;
    uint64_t matrixScratchPerGroup = 0;
    uint64_t hScratchPerGroup = 0;

    if (!CheckedMulU64(physicalBatch, keyHeads, &tmp) ||
        !CheckedMulU64(tmp, tokens, &normalizedElements) ||
        !CheckedMulU64(normalizedElements, headDim, &normalizedElements) ||
        !CheckedMulU64(normalizedElements, inputElementBytes, &normalizedBytes) ||
        !CheckedMulU64(physicalBatch, valueHeads, &cumsumElements) ||
        !CheckedMulU64(cumsumElements, tokens, &cumsumElements) ||
        !CheckedMulU64(cumsumElements, sizeof(float), &cumsumBytes) ||
        !CheckedMulU64(totalChunks, valueHeads, &stateElements) ||
        !CheckedMulU64(stateElements, headDim, &stateElements) ||
        !CheckedMulU64(stateElements, headDim, &stateElements) ||
        !CheckedMulU64(stateElements, inputElementBytes, &stateBytes) ||
        !CheckedMulU64(physicalBatch, valueHeads, &outputElements) ||
        !CheckedMulU64(outputElements, tokens, &outputElements) ||
        !CheckedMulU64(outputElements, headDim, &outputElements) ||
        !CheckedMulU64(outputElements, inputElementBytes, &outputBytes) ||
        !CheckedMulU64(launchGroups, kPrepareBytesPerGroup, &prepareBytes)) {
        return false;
    }

    // Reserve each nested phase from the same user-workspace base. H offsets
    // include the existing library and 16 MiB leading/trailing reservations.
    uint64_t hScratch = 0;
    if (!CheckedMulU64(kRoundHeadSlots * kChunk * headDim, sizeof(float), &tmp) ||
        !AlignSparseWorkspace(tmp, 512, &vScratchPerGroup) ||
        !CheckedMulU64(kRoundHeadSlots * kChunk * headDim, inputElementBytes, &tmp) ||
        !AlignSparseWorkspace(tmp, 512, &updateScratchPerGroup) ||
        !CheckedMulU64(kRoundHeadSlots * headDim * headDim, sizeof(float), &tmp) ||
        !AlignSparseWorkspace(tmp, 512, &matrixScratchPerGroup) ||
        !CheckedAddU64(vScratchPerGroup, updateScratchPerGroup, &hScratchPerGroup) ||
        !CheckedAddU64(hScratchPerGroup, matrixScratchPerGroup, &hScratchPerGroup) ||
        !CheckedAddU64(hScratchPerGroup, matrixScratchPerGroup, &hScratchPerGroup) ||
        !CheckedMulU64(launchGroups, hScratchPerGroup, &hScratch) ||
        !CheckedAddU64(hScratch, 2 * kHReserveBytes, &hBytes) ||
        !CheckedMulU64(launchGroups, kOAprimeBytesPerGroup, &oBytes)) {
        return false;
    }

    uint64_t cursor = 0;
    if (!PlaceSparseWorkspaceRegion(prepareBytes, alignment, &cursor, &plan->prepareOffset) ||
        !PlaceSparseWorkspaceRegion(hBytes, alignment, &cursor, &plan->hOffset) ||
        !PlaceSparseWorkspaceRegion(oBytes, alignment, &cursor, &plan->oOffset) ||
        !PlaceSparseWorkspaceRegion(normalizedBytes, alignment, &cursor, &plan->normalizedOffset) ||
        !PlaceSparseWorkspaceRegion(cumsumBytes, alignment, &cursor, &plan->cumsumOffset) ||
        !PlaceSparseWorkspaceRegion(stateBytes, alignment, &cursor, &plan->stateOffset) ||
        !PlaceSparseWorkspaceRegion(outputBytes, alignment, &cursor, &plan->outputOffset)) {
        return false;
    }

    plan->launchGroups = launchGroups;
    plan->physicalBatch = physicalBatch;
    plan->tokens = tokens;
    plan->totalChunks = totalChunks;
    plan->keyHeads = keyHeads;
    plan->valueHeads = valueHeads;
    plan->headDim = headDim;
    plan->hvPerHk = valueHeads / keyHeads;
    plan->workspaceBytes = cursor;
    workspace->normalizedBytes = normalizedBytes;
    workspace->cumsumBytes = cumsumBytes;
    workspace->stateBytes = stateBytes;
    workspace->outputBytes = outputBytes;
    workspace->prepareBytes = prepareBytes;
    workspace->hBytes = hBytes;
    workspace->oBytes = oBytes;
    workspace->totalBytes = cursor;
    return true;
}

} // namespace optiling

#endif // GDN_CHUNK64_SPARSE_TILING_H
