/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "chunk_gated_delta_rule_fwd_32_tiling_data.h"
#include "../chunk_64_sparse/chunk_gated_delta_rule_fwd_sparse_tiling.h"
#include "../../../../op_kernel/internal/arch35/chunk_64_sparse/sparse_fused_tiling.h"

#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include <algorithm>
#include <limits>

namespace optiling {
namespace {

constexpr size_t INPUT_Q = 0;
constexpr size_t INPUT_K = 1;
constexpr size_t INPUT_V = 2;
// The public low-level op reserves input 3 for FP32 beta. Chunk32 keeps its
// fixed BF16 beta in the otherwise-unused a_storage slot to preserve that ABI.
constexpr size_t INPUT_BETA = 4;
constexpr size_t INPUT_RAW_G = 5;
constexpr size_t INPUT_INITIAL_STATE = 7;
constexpr size_t INPUT_CU_SEQLENS = 8;
constexpr size_t INPUT_CHUNK_INDICES = 9;
constexpr size_t OUTPUT_O = 0;
constexpr size_t OUTPUT_FINAL_STATE = 1;
constexpr size_t ATTR_CHUNK_SIZE = 1;
constexpr size_t ATTR_SCALE = 2;
constexpr size_t ATTR_STATE_V_FIRST = 7;
constexpr size_t ATTR_ALLOW_NEG_EIGVAL = 8;

constexpr int64_t CHUNK_SIZE = 32;
constexpr int64_t HEAD_DIM = 128;
constexpr uint64_t TILING_KEY = 32;

bool IsShape(const gert::StorageShape *shape, std::initializer_list<int64_t> dims)
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

} // namespace

ge::graphStatus TilingForChunkGatedDeltaRuleFwdChunk32(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr || context->GetAttrs() == nullptr,
                OP_LOGE("ChunkGatedDeltaRuleFwd", "Invalid chunk32 tiling context."),
                return ge::GRAPH_FAILED);

    const int64_t *chunkSize = context->GetAttrs()->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE);
    const double *scale = context->GetAttrs()->GetAttrPointer<double>(ATTR_SCALE);
    OP_CHECK_IF(chunkSize == nullptr || *chunkSize != CHUNK_SIZE || scale == nullptr,
                OP_LOGE(context->GetNodeName(), "Chunk32 tiling requires chunk_size=32 and a scale attribute."),
                return ge::GRAPH_FAILED);

    const auto *qShape = context->GetOptionalInputShape(INPUT_Q);
    const auto *kShape = context->GetOptionalInputShape(INPUT_K);
    const auto *vShape = context->GetOptionalInputShape(INPUT_V);
    const auto *betaShape = context->GetOptionalInputShape(INPUT_BETA);
    const auto *gShape = context->GetOptionalInputShape(INPUT_RAW_G);
    const auto *oShape = context->GetOutputShape(OUTPUT_O);
    const auto *stateShape = context->GetOutputShape(OUTPUT_FINAL_STATE);
    const auto *qDesc = context->GetInputDesc(INPUT_Q);
    const auto *kDesc = context->GetInputDesc(INPUT_K);
    const auto *vDesc = context->GetInputDesc(INPUT_V);
    const auto *betaDesc = context->GetInputDesc(INPUT_BETA);
    const auto *gDesc = context->GetInputDesc(INPUT_RAW_G);
    const auto *stateDesc = context->GetOutputDesc(OUTPUT_FINAL_STATE);
    OP_CHECK_IF(qShape == nullptr || kShape == nullptr || vShape == nullptr ||
                    betaShape == nullptr || gShape == nullptr || oShape == nullptr ||
                    stateShape == nullptr || qDesc == nullptr || kDesc == nullptr ||
                    vDesc == nullptr || betaDesc == nullptr || gDesc == nullptr ||
                    stateDesc == nullptr,
                OP_LOGE(context->GetNodeName(), "Chunk32 requires valid q/k/v/beta/g and output descriptors."),
                return ge::GRAPH_FAILED);

    const gert::Shape &q = qShape->GetStorageShape();
    const gert::Shape &v = vShape->GetStorageShape();
    OP_CHECK_IF(q.GetDimNum() != 4 || v.GetDimNum() != 4,
                OP_LOGE(context->GetNodeName(), "Chunk32 q/k/v must use BNSD rank-4 storage."),
                return ge::GRAPH_FAILED);
    const int64_t batch = q.GetDim(0);
    const int64_t keyHeads = q.GetDim(1);
    const int64_t tokens = q.GetDim(2);
    const int64_t keyDim = q.GetDim(3);
    const int64_t valueHeads = v.GetDim(1);
    const int64_t valueDim = v.GetDim(3);
    OP_CHECK_IF(batch <= 0 || keyHeads <= 0 || valueHeads <= 0 || tokens <= 0 ||
                    valueHeads % keyHeads != 0 || keyDim != HEAD_DIM || valueDim != HEAD_DIM,
                OP_LOGE(context->GetNodeName(),
                        "Chunk32 requires positive B/H/T, Hv divisible by Hk, and K=V=128."),
                return ge::GRAPH_FAILED);
    // A caller-owned contiguous o can arrive with its storage collapsed to
    // numel. The logical BNSD layout is already checked in ACLNN.
    const int64_t oNumel = batch * valueHeads * tokens * valueDim;
    const gert::Shape &oStorage = oShape->GetStorageShape();
    const bool oFlat = oStorage.GetDimNum() == 1 && oStorage.GetDim(0) == oNumel;
    OP_CHECK_IF(!IsShape(kShape, {batch, keyHeads, tokens, keyDim}) ||
                    !IsShape(vShape, {batch, valueHeads, tokens, valueDim}) ||
                    !IsShape(betaShape, {batch, valueHeads, tokens}) ||
                    !IsShape(gShape, {batch, valueHeads, tokens}) ||
                    !(IsShape(oShape, {batch, valueHeads, tokens, valueDim}) || oFlat),
                OP_LOGE(context->GetNodeName(), "Chunk32 input or output storage shape is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qDesc->GetDataType() != ge::DT_BF16 ||
                    kDesc->GetDataType() != ge::DT_BF16 ||
                    vDesc->GetDataType() != ge::DT_BF16 ||
                    betaDesc->GetDataType() != ge::DT_BF16 ||
                    gDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(context->GetNodeName(), "Chunk32 requires BF16 q/k/v/beta and FP32 g."),
                return ge::GRAPH_FAILED);

    const auto *cuShape = context->GetOptionalInputShape(INPUT_CU_SEQLENS);
    const auto *chunkShape = context->GetOptionalInputShape(INPUT_CHUNK_INDICES);
    const auto *cuDesc = context->GetOptionalInputDesc(INPUT_CU_SEQLENS);
    const auto *chunkDesc = context->GetOptionalInputDesc(INPUT_CHUNK_INDICES);
    const bool hasCu = cuShape != nullptr && cuDesc != nullptr;
    const bool hasChunks = chunkShape != nullptr && chunkDesc != nullptr;
    OP_CHECK_IF(hasCu != hasChunks,
                OP_LOGE(context->GetNodeName(), "Chunk32 varlen metadata must be paired."),
                return ge::GRAPH_FAILED);
    int64_t stateBatch = batch;
    if (hasCu) {
        const gert::Shape &cu = cuShape->GetStorageShape();
        const gert::Shape &chunks = chunkShape->GetStorageShape();
        OP_CHECK_IF(batch != 1 || cuDesc->GetDataType() != ge::DT_INT64 ||
                        chunkDesc->GetDataType() != ge::DT_INT64 ||
                        cu.GetDimNum() != 1 || cu.GetDim(0) < 2 ||
                        chunks.GetDimNum() != 1 || chunks.GetDim(0) < 0 ||
                        chunks.GetDim(0) % 2 != 0,
                    OP_LOGE(context->GetNodeName(), "Chunk32 varlen metadata descriptor is invalid."),
                    return ge::GRAPH_FAILED);
        stateBatch = cu.GetDim(0) - 1;
    }
    // ACLNN validates the logical state shape. Executor-owned/output tensors
    // may reach tiling with a flattened storage descriptor, so do not repeat
    // that logical-shape check against StorageShape here.

    uint32_t hasInitial = 0;
    uint32_t initialFp32 = 0;
    const auto *initialShape = context->GetOptionalInputShape(INPUT_INITIAL_STATE);
    const auto *initialDesc = context->GetOptionalInputDesc(INPUT_INITIAL_STATE);
    if (initialShape != nullptr || initialDesc != nullptr) {
        OP_CHECK_IF(initialShape == nullptr || initialDesc == nullptr ||
                        (initialDesc->GetDataType() != ge::DT_BF16 &&
                         initialDesc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE(context->GetNodeName(), "Chunk32 initial state descriptor is invalid."),
                    return ge::GRAPH_FAILED);
        hasInitial = 1;
        initialFp32 = initialDesc->GetDataType() == ge::DT_FLOAT ? 1U : 0U;
    }
    const ge::DataType stateDtype = stateDesc->GetDataType();
    if (hasInitial == 0 || initialFp32 != 0) {
        OP_CHECK_IF(stateDtype != ge::DT_FLOAT,
                    OP_LOGE(context->GetNodeName(),
                            "Chunk32 final state must be FP32 when initial state is absent or FP32."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(stateDtype != ge::DT_BF16,
                    OP_LOGE(context->GetNodeName(),
                            "Chunk32 final state must be BF16 when initial state is BF16."),
                    return ge::GRAPH_FAILED);
    }

    auto *tiling = context->GetTilingData<GdnChunk64Sparse::FusedTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    GdnChunk64Sparse::FusedTilingData &td = *tiling;
    td = {};

    uint64_t chunkCount = 0;
    if (hasCu) {
        chunkCount = static_cast<uint64_t>(chunkShape->GetStorageShape().GetDim(0) / 2);
    } else if (!CheckedMulU64(static_cast<uint64_t>(batch),
                              static_cast<uint64_t>((tokens + CHUNK_SIZE - 1) / CHUNK_SIZE),
                              &chunkCount)) {
        OP_LOGE(context->GetNodeName(), "Chunk32 chunk count overflow.");
        return ge::GRAPH_FAILED;
    }
    const platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aicNum = std::max<uint32_t>(static_cast<uint32_t>(platform.GetCoreNumAic()), 1U);
    uint64_t totalChains = 0;
    OP_CHECK_IF(chunkCount == 0 ||
                    !CheckedMulU64(static_cast<uint64_t>(stateBatch),
                                   static_cast<uint64_t>(valueHeads), &totalChains) ||
                    totalChains == 0 || totalChains > std::numeric_limits<uint32_t>::max() ||
                    totalChains > aicNum / 2,
                OP_LOGE(context->GetNodeName(),
                        "Chunk32 Producer-only benchmark requires two MIX groups per chain."),
                return ge::GRAPH_FAILED);

    td.batch = batch;
    td.sequenceCount = stateBatch;
    td.keyHeads = keyHeads;
    td.valueHeads = valueHeads;
    td.tokens = tokens;
    td.totalChunks = static_cast<int64_t>(chunkCount);
    td.chunksPerBatch = hasCu ? 0 : (tokens + CHUNK_SIZE - 1) / CHUNK_SIZE;
    td.scale = static_cast<float>(*scale);
    td.chainCount = static_cast<uint32_t>(totalChains);
    td.maxChunks = hasCu ? static_cast<uint32_t>(chunkCount)
                         : static_cast<uint32_t>((tokens + CHUNK_SIZE - 1) / CHUNK_SIZE);
    td.launchGroups = 2U * td.chainCount;
    td.chunkSize = CHUNK_SIZE;
    td.useNorm = 1;
    td.gateSoftplus = 0;
    td.betaSigmoid = 1;
    const bool *allowNeg = context->GetAttrs()->GetAttrPointer<bool>(ATTR_ALLOW_NEG_EIGVAL);
    td.allowNegEigval = allowNeg != nullptr && *allowNeg ? 1U : 0U;
    td.useExp2 = 0;
    td.outputA = 0;
    td.storeFinalState = 0;
    const bool *stateVFirst = context->GetAttrs()->GetAttrPointer<bool>(ATTR_STATE_V_FIRST);
    td.stateVFirst = stateVFirst != nullptr && *stateVFirst ? 1U : 0U;
    td.stateFp32 = initialFp32 != 0 || hasInitial == 0;
    td.varlen = hasCu ? 1U : 0U;
    td.hasALog = 0;
    td.hasDtBias = 0;
    td.reserved = 0;
    td.writeH = 0;

    if (hasCu) {
        const auto *cuTensor = context->GetOptionalInputTensor(INPUT_CU_SEQLENS);
        const int64_t *cuData = cuTensor == nullptr ? nullptr : cuTensor->GetData<int64_t>();
        if (cuData != nullptr) {
            uint32_t perChainMax = 0;
            for (int64_t seq = 0; seq < stateBatch; ++seq) {
                const int64_t length = cuData[seq + 1] - cuData[seq];
                const uint32_t chunks = length <= 0 ? 0U
                    : static_cast<uint32_t>((length + CHUNK_SIZE - 1) / CHUNK_SIZE);
                perChainMax = chunks > perChainMax ? chunks : perChainMax;
            }
            if (perChainMax > 0 && perChainMax < td.maxChunks) {
                td.maxChunks = perChainMax;
            }
        }
    }

    uint64_t aPrimeBytes = 0;
    uint64_t tBytes = 0;
    uint64_t normalizedTensorBytes = 0;
    OP_CHECK_IF(!CheckedMulU64(chunkCount, static_cast<uint64_t>(valueHeads), &aPrimeBytes) ||
                    !CheckedMulU64(aPrimeBytes, CHUNK_SIZE * CHUNK_SIZE * sizeof(uint16_t), &aPrimeBytes) ||
                    !CheckedMulU64(chunkCount, static_cast<uint64_t>(valueHeads), &tBytes) ||
                    !CheckedMulU64(tBytes, CHUNK_SIZE * CHUNK_SIZE * sizeof(uint16_t), &tBytes) ||
                    !CheckedMulU64(chunkCount, static_cast<uint64_t>(keyHeads), &normalizedTensorBytes) ||
                    !CheckedMulU64(normalizedTensorBytes, CHUNK_SIZE * HEAD_DIM * sizeof(uint16_t),
                                   &normalizedTensorBytes),
                OP_LOGE(context->GetNodeName(), "Chunk32 benchmark workspace size overflow."),
                return ge::GRAPH_FAILED);
    uint64_t cursor = 0;
    td.workspaceOffset = cursor;
    td.internalTOffset = aPrimeBytes;
    td.normalizedQKOffset = td.internalTOffset + tBytes;
    uint64_t normalizedBytes = 0;
    OP_CHECK_IF(!CheckedMulU64(normalizedTensorBytes, 2, &normalizedBytes) ||
                    !CheckedAddU64(td.normalizedQKOffset, normalizedBytes, &cursor),
                OP_LOGE(context->GetNodeName(), "Chunk32 benchmark workspace allocation failed."),
                return ge::GRAPH_FAILED);
    td.workspaceBytes = cursor;

    size_t *workspace = context->GetWorkspaceSizes(1);
    uint64_t totalWorkspace = 0;
    OP_CHECK_IF(workspace == nullptr ||
                    !CheckedAddU64(cursor, platform.GetLibApiWorkSpaceSize(), &totalWorkspace) ||
                    totalWorkspace > std::numeric_limits<size_t>::max(),
                OP_LOGE(context->GetNodeName(), "Chunk32 benchmark workspace is invalid."),
                return ge::GRAPH_FAILED);
    workspace[0] = static_cast<size_t>(totalWorkspace);
    context->SetTilingKey(TILING_KEY);
    context->SetBlockDim(td.launchGroups);
    context->SetScheduleMode(1);
    OP_LOGD(context->GetNodeName(),
            "Chunk32 Producer-only: chains=%u maxChunks=%u groups=%u APrimeOffset=%ld TOffset=%ld QKOffset=%ld userWs=%ld totalWs=%ld.",
            td.chainCount, td.maxChunks, td.launchGroups,
            static_cast<int64_t>(td.workspaceOffset), static_cast<int64_t>(td.internalTOffset),
            static_cast<int64_t>(td.normalizedQKOffset), static_cast<int64_t>(td.workspaceBytes),
            static_cast<int64_t>(totalWorkspace));
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
