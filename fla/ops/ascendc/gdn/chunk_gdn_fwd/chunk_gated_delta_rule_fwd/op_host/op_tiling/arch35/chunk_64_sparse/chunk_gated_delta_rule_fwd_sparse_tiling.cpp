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
#include <cstdlib>
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
    if (shape == nullptr) {
        return false;
    }
    const auto matches = [&dims](const gert::Shape &candidate) {
        if (candidate.GetDimNum() != dims.size()) {
            return false;
        }
        size_t index = 0;
        for (int64_t dim : dims) {
            if (candidate.GetDim(index++) != dim) {
                return false;
            }
        }
        return true;
    };
    return matches(shape->GetStorageShape()) || matches(shape->GetOriginShape());
}

int64_t SparseNumel(std::initializer_list<int64_t> dims)
{
    int64_t numel = 1;
    for (int64_t dim : dims) {
        numel *= dim;
    }
    return numel;
}

bool SparseShapeTolerant(const gert::StorageShape *shape, std::initializer_list<int64_t> dims)
{
    if (SparseShape(shape, dims)) {
        return true;
    }
    if (shape == nullptr) {
        return false;
    }
    const gert::Shape &storage = shape->GetStorageShape();
    return storage.GetDimNum() == 1 && storage.GetDim(0) == SparseNumel(dims);
}
} // namespace

ge::graphStatus TilingForChunkGatedDeltaRuleFwdSparse(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("ChunkGatedDeltaRuleFwd", "Invalid sparse tiling context."),
                return ge::GRAPH_FAILED);
    const auto *attrs = context->GetAttrs();
    const auto *q = context->GetOptionalInputShape(0);
    const auto *v = context->GetOptionalInputShape(2);
    OP_CHECK_IF(attrs == nullptr || q == nullptr || v == nullptr ||
                    q->GetStorageShape().GetDimNum() != 4 || v->GetStorageShape().GetDimNum() != 4,
                OP_LOGE("ChunkGatedDeltaRuleFwd", "Sparse key64 requires attrs and rank-4 q/v."),
                return ge::GRAPH_FAILED);
    const auto &qs = q->GetStorageShape();
    const int64_t batch = qs.GetDim(0);
    const int64_t hk = qs.GetDim(1);
    const int64_t tokens = qs.GetDim(2);
    const int64_t hv = v->GetStorageShape().GetDim(1);
    const int64_t *chunkSize = attrs->GetAttrPointer<int64_t>(1);
    OP_CHECK_IF(batch <= 0 || hk <= 0 || hv <= 0 || tokens <= 0 || hk > hv || hv % hk != 0 ||
                    hv / hk > 4 || qs.GetDim(3) != 128 || chunkSize == nullptr || *chunkSize != 64,
                OP_LOGE("ChunkGatedDeltaRuleFwd",
                        "Sparse key64 requires B/Hk/Hv/T>0, Hv/Hk<=4, K=V=128, chunk=64."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!SparseShape(context->GetOptionalInputShape(1), {batch, hk, tokens, 128}) ||
                    !SparseShape(v, {batch, hv, tokens, 128}) ||
                    !SparseShape(context->GetOptionalInputShape(3), {batch, hv, tokens}) ||
                    !SparseShape(context->GetOptionalInputShape(5), {batch, hv, tokens}),
                OP_LOGE("ChunkGatedDeltaRuleFwd", "Sparse key64 input storage mismatch."),
                return ge::GRAPH_FAILED);
    for (size_t index : {size_t(0), size_t(1), size_t(2)}) {
        const auto *desc = context->GetInputDesc(index);
        OP_CHECK_IF(desc == nullptr || desc->GetDataType() != ge::DT_BF16,
                    OP_LOGE("ChunkGatedDeltaRuleFwd", "Sparse key64 requires BF16 q/k/v."),
                    return ge::GRAPH_FAILED);
    }
    for (size_t index : {size_t(3), size_t(5), size_t(10), size_t(11)}) {
        const auto *desc = context->GetOptionalInputDesc(index);
        OP_CHECK_IF((index < 6 && desc == nullptr) ||
                        (desc != nullptr && desc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE("ChunkGatedDeltaRuleFwd", "Sparse key64 gate/beta must be FP32."),
                    return ge::GRAPH_FAILED);
    }
    const auto *qkvLayout = attrs->GetAttrPointer<int64_t>(5);
    const auto *oLayout = attrs->GetAttrPointer<int64_t>(6);
    OP_CHECK_IF((qkvLayout != nullptr && *qkvLayout != 0) ||
                    (oLayout != nullptr && *oLayout != 1),
                OP_LOGE("ChunkGatedDeltaRuleFwd", "Sparse key64 requires BNSD input and BSND output."),
                return ge::GRAPH_FAILED);
    for (size_t index : {size_t(10), size_t(11)}) {
        const auto *shape = context->GetOptionalInputShape(index);
        OP_CHECK_IF(shape != nullptr && !SparseShape(shape, {hv}),
                    OP_LOGE("ChunkGatedDeltaRuleFwd", "Invalid optional gate input %zu.", index),
                    return ge::GRAPH_FAILED);
    }

    const auto *cu = context->GetOptionalInputShape(8);
    const auto *indices = context->GetOptionalInputShape(9);
    const bool varlen = cu != nullptr;
    int64_t sequenceCount = batch;
    uint64_t chunkCount = 0;
    if (varlen) {
        OP_CHECK_IF(batch != 1 || indices == nullptr || cu->GetStorageShape().GetDimNum() != 1 ||
                        cu->GetStorageShape().GetDim(0) < 2,
                    OP_LOGE("ChunkGatedDeltaRuleFwd", "Invalid varlen metadata."),
                    return ge::GRAPH_FAILED);
        sequenceCount = cu->GetStorageShape().GetDim(0) - 1;
        const auto &is = indices->GetStorageShape();
        if (is.GetDimNum() == 1 && is.GetDim(0) > 0 && is.GetDim(0) % 2 == 0) {
            chunkCount = static_cast<uint64_t>(is.GetDim(0) / 2);
        } else if (is.GetDimNum() == 2 && is.GetDim(0) > 0 && is.GetDim(1) == 2) {
            chunkCount = static_cast<uint64_t>(is.GetDim(0));
        } else {
            OP_LOGE("ChunkGatedDeltaRuleFwd", "chunk_indices must contain (seq,chunk) pairs.");
            return ge::GRAPH_FAILED;
        }
        for (size_t index : {size_t(8), size_t(9)}) {
            const auto *desc = context->GetOptionalInputDesc(index);
            OP_CHECK_IF(desc == nullptr || desc->GetDataType() != ge::DT_INT64,
                        OP_LOGE("ChunkGatedDeltaRuleFwd", "Varlen metadata must be INT64."),
                        return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(indices != nullptr ||
                        !CheckedMulU64(static_cast<uint64_t>(batch),
                                       static_cast<uint64_t>((tokens + 63) / 64), &chunkCount),
                    OP_LOGE("ChunkGatedDeltaRuleFwd", "Invalid dense chunk metadata."),
                    return ge::GRAPH_FAILED);
    }
    const auto *initialShape = context->GetOptionalInputShape(7);
    OP_CHECK_IF(initialShape != nullptr &&
                    !SparseShapeTolerant(initialShape, {sequenceCount, hv, 128, 128}),
                OP_LOGE("ChunkGatedDeltaRuleFwd", "Invalid initial_state shape."),
                return ge::GRAPH_FAILED);

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t groups = platform.GetCoreNumAic();
    OP_CHECK_IF(groups == 0,
                OP_LOGE("ChunkGatedDeltaRuleFwd", "Sparse key64 requires AIC cores."),
                return ge::GRAPH_FAILED);

    uint64_t tileCount = 0;
    OP_CHECK_IF(!CheckedMulU64(chunkCount, static_cast<uint64_t>(hv), &tileCount) ||
                    tileCount > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                OP_LOGE("ChunkGatedDeltaRuleFwd", "Sparse tile count overflows."),
                return ge::GRAPH_FAILED);
    const bool norm = SparseBool(attrs, 10);
    const bool betaSigmoid = SparseBool(attrs, 12);
    for (size_t index : {size_t(9), size_t(10), size_t(11), size_t(12), size_t(13)}) {
        OP_CHECK_IF(context->GetOutputShape(index) == nullptr,
                    OP_LOGE("ChunkGatedDeltaRuleFwd", "Missing fused output %zu.", index),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(!SparseShapeTolerant(context->GetOutputShape(10), {batch, hv, tokens, 128}) ||
                    !SparseShapeTolerant(context->GetOutputShape(11), {batch, hv, tokens, 128}) ||
                    !SparseShapeTolerant(context->GetOutputShape(12), {batch, hv, tokens, 128}) ||
                    !SparseShapeTolerant(context->GetOutputShape(13), {batch, hv, tokens}) ||
                    (betaSigmoid && !SparseShapeTolerant(context->GetOutputShape(8), {batch, hv, tokens})),
                OP_LOGE("ChunkGatedDeltaRuleFwd", "Fused intermediate output shape mismatch."),
                return ge::GRAPH_FAILED);
    const int64_t hChunks = varlen ? static_cast<int64_t>(chunkCount)
                                   : static_cast<int64_t>(chunkCount) / batch;
    OP_CHECK_IF(!SparseShapeTolerant(context->GetOutputShape(9), {batch, hChunks, hv, 128, 128}),
                OP_LOGE("ChunkGatedDeltaRuleFwd", "Invalid h output shape."),
                return ge::GRAPH_FAILED);
    if (norm) {
        OP_CHECK_IF(!SparseShapeTolerant(context->GetOutputShape(4), {batch, hk, tokens, 128}) ||
                        !SparseShapeTolerant(context->GetOutputShape(5), {batch, hk, tokens, 128}) ||
                        !SparseShapeTolerant(context->GetOutputShape(6), {batch, hk, tokens}) ||
                        !SparseShapeTolerant(context->GetOutputShape(7), {batch, hk, tokens}),
                    OP_LOGE("ChunkGatedDeltaRuleFwd", "Invalid norm output shape."),
                    return ge::GRAPH_FAILED);
    }

    ChunkGatedDeltaRuleFwdPipelinePlan plan{};
    const bool pipeline = BuildChunkGatedDeltaRuleFwdPipelinePlan(
        static_cast<uint64_t>(batch), static_cast<uint64_t>(hv),
        static_cast<uint64_t>(tokens), static_cast<uint64_t>(sequenceCount),
        chunkCount, varlen, static_cast<uint64_t>(groups), 512, &plan);
    uint64_t totalChains = 0;
    OP_CHECK_IF(!CheckedMulU64(static_cast<uint64_t>(varlen ? sequenceCount : batch),
                               static_cast<uint64_t>(hv), &totalChains) ||
                    totalChains == 0 || totalChains > std::numeric_limits<uint32_t>::max() ||
                    (pipeline && totalChains > groups / 2),
                OP_LOGE("ChunkGatedDeltaRuleFwd", "Invalid chain count."),
                return ge::GRAPH_FAILED);

    GdnChunk64Sparse::FusedTilingData td{};
    td.batch = batch;
    td.sequenceCount = sequenceCount;
    td.keyHeads = hk;
    td.valueHeads = hv;
    td.tokens = tokens;
    td.totalChunks = static_cast<int64_t>(chunkCount);
    td.chunksPerBatch = varlen ? 0 : (tokens + 63) / 64;
    td.chainCount = static_cast<uint32_t>(totalChains);
    td.maxChunks = pipeline ? plan.maxChunks : 0;
    td.launchGroups = pipeline ? 2U * td.chainCount : groups;
    td.schedule = pipeline ? GdnChunk64Sparse::kSparseSchedulePipeline
                           : GdnChunk64Sparse::kSparseScheduleSerial;
    td.useNorm = norm;
    td.gateSoftplus = context->GetOptionalInputShape(10) != nullptr ||
                      context->GetOptionalInputShape(11) != nullptr;
    td.betaSigmoid = betaSigmoid;
    td.allowNegEigval = SparseBool(attrs, 8);
    td.useExp2 = SparseBool(attrs, 11);
    td.outputA = SparseBool(attrs, 13);
    td.storeFinalState = SparseBool(attrs, 0);
    td.stateVFirst = SparseBool(attrs, 7);
    td.varlen = varlen;
    td.hasALog = context->GetOptionalInputShape(10) != nullptr;
    td.hasDtBias = context->GetOptionalInputShape(11) != nullptr;
    td.reserved = std::getenv("GDN_SPARSE_DEBUG_APRIME") == nullptr ? 0U : 1U;
    const float *scale = attrs->GetAttrPointer<float>(2);
    td.scale = scale == nullptr ? 1.0f : *scale;
    const auto *initialDesc = context->GetOptionalInputDesc(7);
    OP_CHECK_IF(initialDesc != nullptr && initialDesc->GetDataType() != ge::DT_FLOAT &&
                    initialDesc->GetDataType() != ge::DT_BF16,
                OP_LOGE("ChunkGatedDeltaRuleFwd", "initial_state must be FP32 or BF16."),
                return ge::GRAPH_FAILED);
    td.stateFp32 = initialDesc == nullptr || initialDesc->GetDataType() == ge::DT_FLOAT;

    if (pipeline && varlen) {
        const auto *cuTensor = context->GetOptionalInputTensor(8);
        const int64_t *cuData = cuTensor == nullptr ? nullptr : cuTensor->GetData<int64_t>();
        if (cuData != nullptr) {
            uint32_t perChainMax = 0;
            for (int64_t seq = 0; seq < sequenceCount; ++seq) {
                const int64_t length = cuData[seq + 1] - cuData[seq];
                const uint32_t chunks = length <= 0 ? 0U : static_cast<uint32_t>((length + 63) / 64);
                perChainMax = chunks > perChainMax ? chunks : perChainMax;
            }
            if (perChainMax > 0 && perChainMax < td.maxChunks) {
                td.maxChunks = perChainMax;
            }
        }
    }

    // A' is the only cross-role tensor not present in the public schema.
    uint64_t aPrimeBytes = 0;
    uint64_t cursor = 0;
    uint64_t workspaceOffset = 0;
    OP_CHECK_IF(!CheckedMulU64(tileCount,
                               static_cast<uint64_t>(GdnChunk64Sparse::kSparseAPrimeBytesPerTile),
                               &aPrimeBytes) ||
                    !PlaceSparseWorkspaceRegion(aPrimeBytes, 512, &cursor, &workspaceOffset) ||
                    workspaceOffset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
                    cursor > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                OP_LOGE("ChunkGatedDeltaRuleFwd", "A' workspace sizing overflows."),
                return ge::GRAPH_FAILED);
    td.workspaceOffset = static_cast<int64_t>(workspaceOffset);
    td.workspaceBytes = static_cast<int64_t>(cursor);

    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    uint64_t totalWorkspace = 0;
    OP_CHECK_IF(raw == nullptr || raw->GetCapacity() < sizeof(td) || workspace == nullptr ||
                    !CheckedAddU64(cursor, platform.GetLibApiWorkSpaceSize(), &totalWorkspace) ||
                    totalWorkspace > std::numeric_limits<size_t>::max(),
                OP_LOGE("ChunkGatedDeltaRuleFwd", "Sparse tiling/workspace allocation failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(memcpy_s(raw->GetData(), raw->GetCapacity(), &td, sizeof(td)) != EOK,
                OP_LOGE("ChunkGatedDeltaRuleFwd", "Sparse tiling serialization failed."),
                return ge::GRAPH_FAILED);
    raw->SetDataSize(sizeof(td));
    workspace[0] = static_cast<size_t>(totalWorkspace);
    context->SetTilingKey(64);
    context->SetBlockDim(td.launchGroups);
    context->SetScheduleMode(1);
    OP_LOGD("ChunkGatedDeltaRuleFwd",
            "Sparse key64: B=%ld Hk=%ld Hv=%ld T=%ld chains=%u maxChunks=%u groups=%u A'=%llu.",
            batch, hk, hv, tokens, td.chainCount, td.maxChunks, td.launchGroups,
            static_cast<unsigned long long>(aPrimeBytes));
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
