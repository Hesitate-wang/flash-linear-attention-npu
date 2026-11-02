/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "chunk_gated_delta_rule_fwd_sparse_tiling.h"
#include "../../../../op_kernel/internal/arch35/chunk_64_sparse/sparse_fused_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"
#include "securec.h"
#include <cstddef>
#include <initializer_list>
#include <limits>

namespace optiling {
namespace {
template <typename Attrs>
bool SparseBool(const Attrs *attrs, size_t index)
{
    const bool *value = attrs->template GetAttrPointer<bool>(index);
    return value != nullptr && *value;
}

bool SparseShape(const gert::StorageShape *shape, std::initializer_list<int64_t> dims)
{
    if (shape == nullptr || shape->GetStorageShape().GetDimNum() != dims.size()) {
        return false;
    }
    size_t index = 0;
    for (int64_t dim : dims) {
        if (shape->GetStorageShape().GetDim(index++) != dim) {
            return false;
        }
    }
    return true;
}
} // namespace

ge::graphStatus TilingForChunkGatedDeltaRuleFwdSparse(gert::TilingContext *context)
{
    const auto *attrs = context->GetAttrs();
    const auto *q = context->GetOptionalInputShape(0);
    const auto *v = context->GetOptionalInputShape(2);
    if (attrs == nullptr || q == nullptr || v == nullptr ||
        q->GetStorageShape().GetDimNum() != 4 || v->GetStorageShape().GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    const auto &qs = q->GetStorageShape();
    const int64_t batch = qs.GetDim(0);
    const int64_t hk = qs.GetDim(1);
    const int64_t tokens = qs.GetDim(2);
    const int64_t hv = v->GetStorageShape().GetDim(1);
    const int64_t *chunkSize = attrs->GetAttrPointer<int64_t>(1);
    if (batch <= 0 || hk <= 0 || hv <= 0 || tokens <= 0 || hk > hv || hv % hk != 0 || hv / hk > 4 ||
        qs.GetDim(3) != 128 || chunkSize == nullptr || *chunkSize != 64 ||
        !SparseShape(context->GetOptionalInputShape(1), {batch, hk, tokens, 128}) ||
        !SparseShape(v, {batch, hv, tokens, 128}) ||
        !SparseShape(context->GetOptionalInputShape(3), {batch, hv, tokens}) ||
        !SparseShape(context->GetOptionalInputShape(5), {batch, hv, tokens})) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index : {size_t(0), size_t(1), size_t(2)}) {
        const auto *desc = context->GetInputDesc(index);
        if (desc == nullptr || desc->GetDataType() != ge::DT_BF16) {
            return ge::GRAPH_FAILED;
        }
    }
    for (size_t index : {size_t(3), size_t(5), size_t(10), size_t(11)}) {
        const auto *desc = context->GetOptionalInputDesc(index);
        if ((index < 6 && desc == nullptr) || (desc != nullptr && desc->GetDataType() != ge::DT_FLOAT)) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto *qkvLayout = attrs->GetAttrPointer<int64_t>(5);
    const auto *oLayout = attrs->GetAttrPointer<int64_t>(6);
    if ((qkvLayout != nullptr && *qkvLayout != 0) ||
        (oLayout != nullptr && *oLayout != 1)) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index : {size_t(10), size_t(11)}) {
        const auto *shape = context->GetOptionalInputShape(index);
        if (shape != nullptr && !SparseShape(shape, {hv})) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto *cu = context->GetOptionalInputShape(8);
    const auto *indices = context->GetOptionalInputShape(9);
    const bool varlen = cu != nullptr;
    int64_t sequenceCount = batch;
    uint64_t chunkCount = 0;
    if (varlen) {
        if (batch != 1 || indices == nullptr || cu->GetStorageShape().GetDimNum() != 1 ||
            cu->GetStorageShape().GetDim(0) < 2) {
            return ge::GRAPH_FAILED;
        }
        sequenceCount = cu->GetStorageShape().GetDim(0) - 1;
        const auto &is = indices->GetStorageShape();
        if (is.GetDimNum() == 1 && is.GetDim(0) > 0 && is.GetDim(0) % 2 == 0) {
            chunkCount = static_cast<uint64_t>(is.GetDim(0) / 2);
        } else if (is.GetDimNum() == 2 && is.GetDim(0) > 0 && is.GetDim(1) == 2) {
            chunkCount = static_cast<uint64_t>(is.GetDim(0));
        } else {
            return ge::GRAPH_FAILED;
        }
        for (size_t index : {size_t(8), size_t(9)}) {
            const auto *desc = context->GetOptionalInputDesc(index);
            if (desc == nullptr || desc->GetDataType() != ge::DT_INT64) {
                return ge::GRAPH_FAILED;
            }
        }
    } else if (indices != nullptr ||
               !CheckedMulU64(static_cast<uint64_t>(batch),
                              static_cast<uint64_t>(tokens / 64 + (tokens % 64 != 0)), &chunkCount)) {
        return ge::GRAPH_FAILED;
    }
    const auto *initialShape = context->GetOptionalInputShape(7);
    if (initialShape != nullptr && !SparseShape(initialShape, {sequenceCount, hv, 128, 128})) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t groups = platform.GetCoreNumAic();
    if (groups == 0) {
        return ge::GRAPH_FAILED;
    }
    GdnChunk64Sparse::FusedTilingData td{};
    auto &p = td.prepare;
    p.inputBatchSize = batch;
    p.queryKeyHeadCount = hk;
    p.valueHeadCount = hv;
    p.sequenceTokenLength = tokens;
    p.queryKeyHeadDim = 128;
    p.valueHeadDim = 128;
    p.valueHeadsPerQueryKeyHead = hv / hk;
    p.tokensPerChunk = 64;
    p.packedSequenceCount = sequenceCount;
    p.isVariableLengthPacked = varlen;
    p.hasChunkIndexTable = varlen;
    p.enableQueryKeyL2NormInKernel = SparseBool(attrs, 10);
    p.enableFusedGateSoftplus = context->GetOptionalInputShape(10) != nullptr ||
                               context->GetOptionalInputShape(11) != nullptr;
    p.enableBetaSigmoid = SparseBool(attrs, 12);
    p.hasGateDtBias = context->GetOptionalInputShape(11) != nullptr;
    p.hasQueryHatGmOutput = p.enableQueryKeyL2NormInKernel;
    p.hasKeyHatGmOutput = p.enableQueryKeyL2NormInKernel;
    p.scaleBetaByTwoWhenNegEigval = SparseBool(attrs, 8);
    p.useExp2ForGateCumsum = SparseBool(attrs, 11);
    p.queryKeyStorageDtype = static_cast<int64_t>(ge::DT_BF16);
    p.gateStorageDtype = static_cast<int64_t>(ge::DT_FLOAT);
    p.betaStorageDtype = static_cast<int64_t>(ge::DT_FLOAT);
    uint64_t tileCount = 0;
    if (!CheckedMulU64(chunkCount, static_cast<uint64_t>(hv), &tileCount) ||
        tileCount > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return ge::GRAPH_FAILED;
    }
    p.totalChunkTileCount = static_cast<int64_t>(tileCount);
    p.hasAOutput = SparseBool(attrs, 13);
    const bool norm = p.enableQueryKeyL2NormInKernel != 0;
    for (size_t index : {size_t(9), size_t(10), size_t(11), size_t(12), size_t(13)}) {
        if (context->GetOutputShape(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    if (!SparseShape(context->GetOutputShape(10), {batch, hv, tokens, 128}) ||
        !SparseShape(context->GetOutputShape(11), {batch, hv, tokens, 128}) ||
        !SparseShape(context->GetOutputShape(12), {batch, hv, tokens, 128}) ||
        !SparseShape(context->GetOutputShape(13), {batch, hv, tokens}) ||
        (p.enableBetaSigmoid && !SparseShape(context->GetOutputShape(8), {batch, hv, tokens}))) {
        return ge::GRAPH_FAILED;
    }
    const auto *hShape = context->GetOutputShape(9);
    if (hShape->GetStorageShape().GetDimNum() != 5 ||
        hShape->GetStorageShape().GetDim(0) != batch ||
        hShape->GetStorageShape().GetDim(1) != static_cast<int64_t>(varlen ? chunkCount : chunkCount / batch) ||
        hShape->GetStorageShape().GetDim(2) != hv ||
        hShape->GetStorageShape().GetDim(3) != 128 ||
        hShape->GetStorageShape().GetDim(4) != 128) {
        return ge::GRAPH_FAILED;
    }
    if (norm) {
        if (!SparseShape(context->GetOutputShape(4), {batch, hk, tokens, 128}) ||
            !SparseShape(context->GetOutputShape(5), {batch, hk, tokens, 128}) ||
            !SparseShape(context->GetOutputShape(6), {batch, hk, tokens}) ||
            !SparseShape(context->GetOutputShape(7), {batch, hk, tokens})) {
            return ge::GRAPH_FAILED;
        }
        for (size_t index : {size_t(4), size_t(5), size_t(6), size_t(7)}) {
            if (context->GetOutputShape(index) == nullptr) {
                return ge::GRAPH_FAILED;
            }
        }
    }
    auto &h = td.h;
    h.batch = sequenceCount;
    h.seqlen = tokens;
    h.kNumHead = hk;
    h.vNumHead = hv;
    h.kHeadDim = 128;
    h.vHeadDim = 128;
    h.chunkSize = 64;
    h.useInitialState = context->GetOptionalInputShape(7) != nullptr;
    h.storeFinalState = SparseBool(attrs, 0);
    h.isVariedLen = varlen;
    h.shapeBatch = batch;
    h.tokenBatch = varlen ? sequenceCount : 1;
    auto &o = td.o;
    o.shapeBatch = batch;
    o.seqlen = tokens;
    o.kNumHead = hk;
    o.vNumHead = hv;
    o.kHeadDim = 128;
    o.vHeadDim = 128;
    o.chunkSize = 64;
    o.isVariedLen = varlen;
    o.tokenBatch = varlen ? sequenceCount : 1;
    o.dataType = 1;
    o.gDataType = 2;
    o.stateVFirst = SparseBool(attrs, 7);
    o.outputLayout = GDN::CHUNK_FWD_O_LAYOUT_BSND;
    const float *scale = attrs->GetAttrPointer<float>(2);
    o.scale = scale == nullptr ? 1.0f : *scale;
    o.chunkNum = chunkCount;
    o.hvPerHk = hv / hk;
    o.taskGroupSize = hv / hk == 3 ? 3 : 4;
    o.numChunksPerBatch = varlen ? 0 : tokens / 64 + (tokens % 64 != 0);
    o.aPrimeWorkspaceOffset = 0;

    // Low-parallelism selector: with B*Hv <= AIC/2 every chain gets a producer
    // and a consumer group, and the launch keeps the physical MIX width so the
    // mode-0 participant set matches the serial path.
    ChunkGatedDeltaRuleFwdPipelinePlan pipeline{};
    const bool usePipeline = BuildChunkGatedDeltaRuleFwdPipelinePlan(
        static_cast<uint64_t>(batch), static_cast<uint64_t>(hv),
        static_cast<uint64_t>(tokens), static_cast<uint64_t>(sequenceCount),
        chunkCount, varlen, static_cast<uint64_t>(groups), 512, &pipeline);
    td.pipeline.schedule = usePipeline ? GdnChunk64Sparse::kSparseSchedulePipeline
                                       : GdnChunk64Sparse::kSparseScheduleSerial;
    td.pipeline.barrierFlagId = GdnChunk64Sparse::kSparseBarrierFlagId;
    td.pipeline.chainCount = usePipeline ? pipeline.chainCount : 0;
    td.pipeline.maxChunks = usePipeline ? pipeline.maxChunks : 0;
    td.pipeline.producerGroups = usePipeline ? pipeline.producerGroups : 0;
    td.pipeline.consumerGroups = usePipeline ? pipeline.consumerGroups : 0;

    // Serial baseline: Prepare scratch, then the standalone H and O recipes.
    // H scratch is deliberately not sized by the reduced active count.
    uint64_t cursor = static_cast<uint64_t>(groups) * 64 * 1024;
    td.hWorkspaceOffset = cursor;
    uint64_t hCursor = 16ULL * 1024 * 1024;
    h.vWorkspaceOffset = hCursor;
    hCursor += static_cast<uint64_t>(groups) * 128 * 1024;
    h.vUpdateWorkspaceOffset = hCursor;
    hCursor += static_cast<uint64_t>(groups) * 64 * 1024;
    h.kDecayWorkspaceOffset = hCursor;
    hCursor += static_cast<uint64_t>(groups) * 256 * 1024;
    h.hWorkspaceOffset = hCursor;
    hCursor += static_cast<uint64_t>(groups) * 256 * 1024 + 16ULL * 1024 * 1024;
    cursor += hCursor;
    td.oWorkspaceOffset = cursor;
    cursor += static_cast<uint64_t>(groups) * 64 * 1024;

    // The pipeline scratch is appended after the serial recipe until the fused
    // role map stops relying on the fallback; the device picks the region it
    // needs from `pipeline.schedule`.
    if (usePipeline) {
        uint64_t pipelineOffset = 0;
        if (!PlaceSparseWorkspaceRegion(pipeline.totalWorkspaceBytes, 512, &cursor,
                                        &pipelineOffset)) {
            return ge::GRAPH_FAILED;
        }
        td.pipeline.workspaceOffset = pipelineOffset;
        td.pipeline.workspaceBytes = pipeline.totalWorkspaceBytes;
    } else {
        td.pipeline.workspaceOffset = 0;
        td.pipeline.workspaceBytes = 0;
    }
    td.workspaceBytes = cursor;
    td.launchGroups = groups;
    td.useExp2 = p.useExp2ForGateCumsum;
    td.stateVFirst = o.stateVFirst;
    const auto *initialDesc = context->GetOptionalInputDesc(7);
    td.stateDtype = initialDesc == nullptr ? static_cast<uint32_t>(ge::DT_FLOAT)
                                         : static_cast<uint32_t>(initialDesc->GetDataType());
    if (td.stateDtype != static_cast<uint32_t>(ge::DT_FLOAT) &&
        td.stateDtype != static_cast<uint32_t>(ge::DT_BF16)) {
        return ge::GRAPH_FAILED;
    }
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    uint64_t totalWorkspace = 0;
    if (raw == nullptr || raw->GetCapacity() < sizeof(td) || workspace == nullptr ||
        !CheckedAddU64(cursor, platform.GetLibApiWorkSpaceSize(), &totalWorkspace) ||
        totalWorkspace > std::numeric_limits<size_t>::max() ||
        memcpy_s(raw->GetData(), raw->GetCapacity(), &td, sizeof(td)) != EOK) {
        return ge::GRAPH_FAILED;
    }
    raw->SetDataSize(sizeof(td));
    workspace[0] = static_cast<size_t>(totalWorkspace);
    context->SetTilingKey(64);
    context->SetBlockDim(groups);
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
