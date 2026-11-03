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

// Executor-owned and caller output tensors may reach tiling with the storage
// descriptor collapsed to the flat element count (same lesson as the chunk32
// path). ACLNN already validated the logical shape, so accept a rank-1 storage
// whose element count matches in addition to the storage/origin forms.
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
                OP_LOGE("ChunkGatedDeltaRuleFwd",
                        "Sparse key64 requires attrs and rank-4 q/v storage shapes."),
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
                        "Sparse key64 requires positive B/Hk/Hv/T, Hk<=Hv, Hv%Hk==0, Hv/Hk<=4, "
                        "K=128 and chunk_size=64; got B=%ld Hk=%ld Hv=%ld T=%ld K=%ld.",
                        batch, hk, hv, tokens, qs.GetDim(3)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!SparseShape(context->GetOptionalInputShape(1), {batch, hk, tokens, 128}) ||
                    !SparseShape(v, {batch, hv, tokens, 128}) ||
                    !SparseShape(context->GetOptionalInputShape(3), {batch, hv, tokens}) ||
                    !SparseShape(context->GetOptionalInputShape(5), {batch, hv, tokens}),
                OP_LOGE("ChunkGatedDeltaRuleFwd",
                        "Sparse key64 input storage mismatch: k/v=[%ld,%ld,%ld,128], beta/g=[%ld,%ld,%ld].",
                        batch, hk, tokens, batch, hv, tokens),
                return ge::GRAPH_FAILED);
    for (size_t index : {size_t(0), size_t(1), size_t(2)}) {
        const auto *desc = context->GetInputDesc(index);
        OP_CHECK_IF(desc == nullptr || desc->GetDataType() != ge::DT_BF16,
                    OP_LOGE("ChunkGatedDeltaRuleFwd",
                            "Sparse key64 requires BF16 q/k/v; input %zu violates the contract.",
                            index),
                    return ge::GRAPH_FAILED);
    }
    for (size_t index : {size_t(3), size_t(5), size_t(10), size_t(11)}) {
        const auto *desc = context->GetOptionalInputDesc(index);
        OP_CHECK_IF((index < 6 && desc == nullptr) ||
                        (desc != nullptr && desc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE("ChunkGatedDeltaRuleFwd",
                            "Sparse key64 requires FP32 beta/raw_g (and FP32 a_log/dt_bias when "
                            "present); input %zu violates the contract.",
                            index),
                    return ge::GRAPH_FAILED);
    }
    const auto *qkvLayout = attrs->GetAttrPointer<int64_t>(5);
    const auto *oLayout = attrs->GetAttrPointer<int64_t>(6);
    OP_CHECK_IF((qkvLayout != nullptr && *qkvLayout != 0) ||
                    (oLayout != nullptr && *oLayout != 1),
                OP_LOGE("ChunkGatedDeltaRuleFwd",
                        "Sparse key64 requires qkv_layout=0 and o_layout=1(BSND)."),
                return ge::GRAPH_FAILED);
    for (size_t index : {size_t(10), size_t(11)}) {
        const auto *shape = context->GetOptionalInputShape(index);
        OP_CHECK_IF(shape != nullptr && !SparseShape(shape, {hv}),
                    OP_LOGE("ChunkGatedDeltaRuleFwd",
                            "Sparse key64 optional input %zu must have shape [Hv=%ld].", index, hv),
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
                    OP_LOGE("ChunkGatedDeltaRuleFwd",
                            "Sparse key64 varlen requires physical B=1, paired chunk_indices and "
                            "cu_seqlens with at least two entries."),
                    return ge::GRAPH_FAILED);
        sequenceCount = cu->GetStorageShape().GetDim(0) - 1;
        const auto &is = indices->GetStorageShape();
        if (is.GetDimNum() == 1 && is.GetDim(0) > 0 && is.GetDim(0) % 2 == 0) {
            chunkCount = static_cast<uint64_t>(is.GetDim(0) / 2);
        } else if (is.GetDimNum() == 2 && is.GetDim(0) > 0 && is.GetDim(1) == 2) {
            chunkCount = static_cast<uint64_t>(is.GetDim(0));
        } else {
            OP_LOGE("ChunkGatedDeltaRuleFwd",
                    "Sparse key64 chunk_indices must be (seq,chunk) pairs.");
            return ge::GRAPH_FAILED;
        }
        for (size_t index : {size_t(8), size_t(9)}) {
            const auto *desc = context->GetOptionalInputDesc(index);
            OP_CHECK_IF(desc == nullptr || desc->GetDataType() != ge::DT_INT64,
                        OP_LOGE("ChunkGatedDeltaRuleFwd",
                                "Sparse key64 varlen metadata must be INT64; input %zu violates it.",
                                index),
                        return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(indices != nullptr ||
                        !CheckedMulU64(static_cast<uint64_t>(batch),
                                       static_cast<uint64_t>(tokens / 64 + (tokens % 64 != 0)),
                                       &chunkCount),
                    OP_LOGE("ChunkGatedDeltaRuleFwd",
                            "Sparse key64 dense mode forbids chunk_indices without cu_seqlens."),
                    return ge::GRAPH_FAILED);
    }
    const auto *initialShape = context->GetOptionalInputShape(7);
    OP_CHECK_IF(initialShape != nullptr &&
                    !SparseShapeTolerant(initialShape, {sequenceCount, hv, 128, 128}),
                OP_LOGE("ChunkGatedDeltaRuleFwd",
                        "Sparse key64 initial_state must be [%ld,%ld,128,128] (flat storage accepted).",
                        sequenceCount, hv),
                return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t groups = platform.GetCoreNumAic();
    OP_CHECK_IF(groups == 0,
                OP_LOGE("ChunkGatedDeltaRuleFwd", "Sparse key64 requires a nonzero AIC count."),
                return ge::GRAPH_FAILED);
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
    OP_CHECK_IF(!CheckedMulU64(chunkCount, static_cast<uint64_t>(hv), &tileCount) ||
                    tileCount > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                OP_LOGE("ChunkGatedDeltaRuleFwd", "Sparse key64 chunk tile count overflows."),
                return ge::GRAPH_FAILED);
    p.totalChunkTileCount = static_cast<int64_t>(tileCount);
    p.hasAOutput = SparseBool(attrs, 13);
    const bool norm = p.enableQueryKeyL2NormInKernel != 0;
    for (size_t index : {size_t(9), size_t(10), size_t(11), size_t(12), size_t(13)}) {
        OP_CHECK_IF(context->GetOutputShape(index) == nullptr,
                    OP_LOGE("ChunkGatedDeltaRuleFwd",
                            "Sparse key64 requires output %zu (h/w/u/v_new/g_cumsum_bht).", index),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(!SparseShapeTolerant(context->GetOutputShape(10), {batch, hv, tokens, 128}) ||
                    !SparseShapeTolerant(context->GetOutputShape(11), {batch, hv, tokens, 128}) ||
                    !SparseShapeTolerant(context->GetOutputShape(12), {batch, hv, tokens, 128}) ||
                    !SparseShapeTolerant(context->GetOutputShape(13), {batch, hv, tokens}) ||
                    (p.enableBetaSigmoid &&
                     !SparseShapeTolerant(context->GetOutputShape(8), {batch, hv, tokens})),
                OP_LOGE("ChunkGatedDeltaRuleFwd",
                        "Sparse key64 output storage mismatch: w/u/v_new=[%ld,%ld,%ld,128], "
                        "g_cumsum_bht/beta_eff=[%ld,%ld,%ld] (flat storage accepted).",
                        batch, hv, tokens, batch, hv, tokens),
                return ge::GRAPH_FAILED);
    const auto *hShape = context->GetOutputShape(9);
    const int64_t hChunks =
        varlen ? static_cast<int64_t>(chunkCount) : static_cast<int64_t>(chunkCount) / batch;
    OP_CHECK_IF(!SparseShapeTolerant(hShape, {batch, hChunks, hv, 128, 128}),
                OP_LOGE("ChunkGatedDeltaRuleFwd",
                        "Sparse key64 h output must be [%ld,%ld,%ld,128,128] (flat storage accepted).",
                        batch, hChunks, hv),
                return ge::GRAPH_FAILED);
    if (norm) {
        OP_CHECK_IF(!SparseShapeTolerant(context->GetOutputShape(4), {batch, hk, tokens, 128}) ||
                        !SparseShapeTolerant(context->GetOutputShape(5), {batch, hk, tokens, 128}) ||
                        !SparseShapeTolerant(context->GetOutputShape(6), {batch, hk, tokens}) ||
                        !SparseShapeTolerant(context->GetOutputShape(7), {batch, hk, tokens}),
                    OP_LOGE("ChunkGatedDeltaRuleFwd",
                            "Sparse key64 norm outputs must be q_hat/k_hat=[%ld,%ld,%ld,128] and "
                            "q_rstd/k_rstd=[%ld,%ld,%ld] (flat storage accepted).",
                            batch, hk, tokens, batch, hk, tokens),
                    return ge::GRAPH_FAILED);
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
    o.outputLayout = GdnChunk64Sparse::CHUNK_FWD_O_LAYOUT_BSND;
    const float *scale = attrs->GetAttrPointer<float>(2);
    o.scale = scale == nullptr ? 1.0f : *scale;
    o.chunkNum = chunkCount;
    o.hvPerHk = hv / hk;
    o.taskGroupSize = hv / hk == 3 ? 3 : 4;
    o.numChunksPerBatch = varlen ? 0 : tokens / 64 + (tokens % 64 != 0);
    o.aPrimeWorkspaceOffset = 0;

    // Low-parallelism selector: with B*Hv <= AIC/2 every chain gets a producer
    // and a consumer group, and the launch keeps the physical MIX width so the
    // mode-0 participant set matches the serial path. Above that bound the
    // group-serial schedule gives every group every launchGroups-th chain.
    ChunkGatedDeltaRuleFwdPipelinePlan pipeline{};
    const bool usePipeline = BuildChunkGatedDeltaRuleFwdPipelinePlan(
        static_cast<uint64_t>(batch), static_cast<uint64_t>(hv),
        static_cast<uint64_t>(tokens), static_cast<uint64_t>(sequenceCount),
        chunkCount, varlen, static_cast<uint64_t>(groups), 512, &pipeline);
    td.pipeline.schedule = usePipeline ? GdnChunk64Sparse::kSparseSchedulePipeline
                                       : GdnChunk64Sparse::kSparseScheduleSerial;
    td.pipeline.barrierFlagId = GdnChunk64Sparse::kSparseBarrierFlagId;
    uint64_t totalChains = 0;
    OP_CHECK_IF(!CheckedMulU64(static_cast<uint64_t>(varlen ? sequenceCount : batch),
                               static_cast<uint64_t>(hv), &totalChains) ||
                    totalChains > std::numeric_limits<uint32_t>::max(),
                OP_LOGE("ChunkGatedDeltaRuleFwd", "Sparse key64 chain count overflows."),
                return ge::GRAPH_FAILED);
    td.pipeline.chainCount = static_cast<uint32_t>(totalChains);
    td.pipeline.maxChunks = usePipeline ? pipeline.maxChunks : 0;
    td.pipeline.producerGroups = usePipeline ? pipeline.producerGroups : 0;
    td.pipeline.consumerGroups = usePipeline ? pipeline.consumerGroups : 0;
    // Bring-up bisect switch: GDN_SPARSE_DEBUG_MODE=1/2/3 runs barrier-only /
    // producer-only / consumer-only on device without a rebuild.
    if (const char *debugEnv = std::getenv("GDN_SPARSE_DEBUG_MODE")) {
        td.pipeline.debugMode = static_cast<uint32_t>(std::atoi(debugEnv));
    }

    // Both schedules exchange every intermediate through the aclnn output
    // tensors and never touch a workspace recipe: only the pipeline scratch
    // (currently unused by the roles, kept for the fused on-chip map) is
    // reserved below. The historical serial H/O recipe workspace is gone
    // together with the external-kernel serial fallback.
    uint64_t cursor = 0;

    // Varlen: every group iterates maxChunks + 1 rounds and chains shorter than
    // maxChunks only pace the barrier. Tighten the bound to the largest
    // per-sequence chunk count; cu_seqlens is ValueDepend, so the host may read
    // it here (the Phase-5/6 suffix tiling reads it the same way).
    if (usePipeline && varlen) {
        const auto *cuTensor = context->GetOptionalInputTensor(8);
        const int64_t *cuData = cuTensor == nullptr ? nullptr : cuTensor->GetData<int64_t>();
        if (cuData != nullptr) {
            uint64_t perChainMax = 0;
            for (int64_t seq = 0; seq < sequenceCount; ++seq) {
                const int64_t length = cuData[seq + 1] - cuData[seq];
                const uint64_t seqChunks =
                    length <= 0 ? 0 : static_cast<uint64_t>((length + 63) / 64);
                perChainMax = seqChunks > perChainMax ? seqChunks : perChainMax;
            }
            if (perChainMax > 0 && perChainMax < td.pipeline.maxChunks) {
                td.pipeline.maxChunks = static_cast<uint32_t>(perChainMax);
            }
        }
    }

    // The pipeline scratch is appended after the serial recipe until the fused
    // role map stops relying on the fallback; the device picks the region it
    // needs from `pipeline.schedule`.
    if (usePipeline) {
        uint64_t pipelineOffset = 0;
        OP_CHECK_IF(!PlaceSparseWorkspaceRegion(pipeline.totalWorkspaceBytes, 512, &cursor,
                                                &pipelineOffset),
                    OP_LOGE("ChunkGatedDeltaRuleFwd",
                            "Sparse key64 pipeline workspace placement overflows."),
                    return ge::GRAPH_FAILED);
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
    OP_CHECK_IF(td.stateDtype != static_cast<uint32_t>(ge::DT_FLOAT) &&
                    td.stateDtype != static_cast<uint32_t>(ge::DT_BF16),
                OP_LOGE("ChunkGatedDeltaRuleFwd",
                        "Sparse key64 initial_state must be FP32 or BF16."),
                return ge::GRAPH_FAILED);
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    uint64_t totalWorkspace = 0;
    OP_CHECK_IF(raw == nullptr || raw->GetCapacity() < sizeof(td) || workspace == nullptr ||
                    !CheckedAddU64(cursor, platform.GetLibApiWorkSpaceSize(), &totalWorkspace) ||
                    totalWorkspace > std::numeric_limits<size_t>::max(),
                OP_LOGE("ChunkGatedDeltaRuleFwd",
                        "Sparse key64 tiling buffer or workspace accounting failed: capacity=%lu, "
                        "tilingBytes=%zu.",
                        raw == nullptr ? 0UL : static_cast<unsigned long>(raw->GetCapacity()),
                        sizeof(td)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(memcpy_s(raw->GetData(), raw->GetCapacity(), &td, sizeof(td)) != EOK,
                OP_LOGE("ChunkGatedDeltaRuleFwd", "Sparse key64 tiling serialization failed."),
                return ge::GRAPH_FAILED);
    raw->SetDataSize(sizeof(td));
    workspace[0] = static_cast<size_t>(totalWorkspace);
    context->SetTilingKey(64);
    context->SetBlockDim(groups);
    context->SetScheduleMode(1);
    OP_LOGD("ChunkGatedDeltaRuleFwd",
            "Sparse key64 tiling: B=%ld Hk=%ld Hv=%ld T=%ld varlen=%d chains=%u maxChunks=%u "
            "pipeline=%d groups=%u workspace=%llu.",
            batch, hk, hv, tokens, varlen ? 1 : 0, td.pipeline.chainCount, td.pipeline.maxChunks,
            usePipeline ? 1 : 0, groups, static_cast<unsigned long long>(totalWorkspace));
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
