/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "chunk_fwd_h_o_fused_tiling.h"

#include "../op_kernel/chunk_fwd_o_a5_constants.h"
#include "../op_kernel/chunk_fwd_h_o_fused_struct.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstring>
#include <initializer_list>
#include <limits>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include <register/op_impl_registry.h>
#include <register/register.h>
namespace optiling {
namespace {

constexpr size_t INPUT_K = 0;
constexpr size_t INPUT_Q = 1;
constexpr size_t INPUT_W = 2;
constexpr size_t INPUT_U = 3;
constexpr size_t INPUT_G = 4;
constexpr size_t INPUT_GK = 5;
constexpr size_t INPUT_INITIAL_STATE = 6;
constexpr size_t INPUT_CU_SEQLENS = 7;
constexpr size_t INPUT_CHUNK_INDICES = 8;

constexpr size_t OUTPUT_O = 0;
constexpr size_t OUTPUT_FINAL_STATE = 1;

constexpr size_t ATTR_OUTPUT_FINAL_STATE = 0;
constexpr size_t ATTR_CHUNK_SIZE = 1;
constexpr size_t ATTR_SCALE = 2;
constexpr size_t ATTR_USE_EXP2 = 3;
constexpr size_t ATTR_OUTPUT_LAYOUT = 4;
constexpr size_t ATTR_LOGICAL_BATCH = 5;
constexpr size_t ATTR_LOGICAL_SEQLEN = 6;
constexpr size_t ATTR_LOGICAL_K_HEADS = 7;
constexpr size_t ATTR_LOGICAL_V_HEADS = 8;
constexpr size_t ATTR_LOGICAL_K_DIM = 9;
constexpr size_t ATTR_LOGICAL_V_DIM = 10;

constexpr int64_t DIM_B = 0;
constexpr int64_t DIM_H = 1;
constexpr int64_t DIM_T = 2;
constexpr int64_t DIM_D = 3;
constexpr int64_t SUPPORTED_K = 128;
constexpr int64_t SUPPORTED_V128 = 128;
constexpr int64_t SUPPORTED_V256 = 256;
constexpr int64_t CHUNK_64 = 64;
constexpr int64_t CHUNK_128 = 128;
constexpr size_t WORKSPACE_ALIGNMENT = 512;
constexpr size_t PING_PONG_STAGES = 2;
constexpr size_t INPUT_ELEMENT_BYTES = sizeof(uint16_t);

size_t AlignUp(size_t value)
{
    return (value + WORKSPACE_ALIGNMENT - 1) / WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT;
}

bool CheckedMul(std::initializer_list<size_t> factors, size_t &result)
{
    result = 1;
    for (size_t factor : factors) {
        if (factor != 0 && result > std::numeric_limits<size_t>::max() / factor) {
            return false;
        }
        result *= factor;
    }
    return true;
}

bool AllocateRegion(size_t bytes, size_t &offset, int64_t &regionOffset)
{
    if (offset > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        bytes > std::numeric_limits<size_t>::max() - (WORKSPACE_ALIGNMENT - 1)) {
        return false;
    }
    regionOffset = static_cast<int64_t>(offset);
    const size_t alignedBytes = AlignUp(bytes);
    if (offset > std::numeric_limits<size_t>::max() - alignedBytes) {
        return false;
    }
    offset += alignedBytes;
    return true;
}

bool IsRank(const gert::StorageShape *shape, size_t rank)
{
    return shape != nullptr && shape->GetStorageShape().GetDimNum() == rank;
}

bool IsSupportedInputType(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}

bool IsSupportedGateType(ge::DataType dtype, ge::DataType inputType)
{
    return dtype == ge::DT_FLOAT || dtype == inputType;
}

int64_t DtypeToEnum(ge::DataType dtype)
{
    if (dtype == ge::DT_BF16) {
        return static_cast<int64_t>(GDN::ChunkFwdHOFusedDtype::BF16);
    }
    if (dtype == ge::DT_FLOAT16) {
        return static_cast<int64_t>(GDN::ChunkFwdHOFusedDtype::FP16);
    }
    return static_cast<int64_t>(GDN::ChunkFwdHOFusedDtype::FP32);
}

ge::graphStatus ValidateTensorContracts(gert::TilingContext *context, int64_t batch,
                                        int64_t seqlen, int64_t kHeads, int64_t vHeads,
                                        int64_t kDim, int64_t vDim, int64_t outputLayout)
{
    const auto *kShapePtr = context->GetOptionalInputShape(INPUT_K);
    const auto *wShapePtr = context->GetOptionalInputShape(INPUT_W);
    const auto *uShapePtr = context->GetOptionalInputShape(INPUT_U);
    const auto *gShapePtr = context->GetOptionalInputShape(INPUT_G);
    const auto *qShapePtr = context->GetOptionalInputShape(INPUT_Q);
    const auto *oShapePtr = context->GetOutputShape(OUTPUT_O);
    OP_CHECK_IF(!IsRank(kShapePtr, 4) || !IsRank(wShapePtr, 4) || !IsRank(uShapePtr, 4) ||
                    !IsRank(gShapePtr, 3) || !IsRank(qShapePtr, 4) || oShapePtr == nullptr,
                OP_LOGE(context->GetNodeName(), "Invalid required input/output rank."),
                return ge::GRAPH_FAILED);

    const gert::Shape k = kShapePtr->GetStorageShape();
    const gert::Shape w = wShapePtr->GetStorageShape();
    const gert::Shape u = uShapePtr->GetStorageShape();
    const gert::Shape g = gShapePtr->GetStorageShape();
    const gert::Shape q = qShapePtr->GetStorageShape();
    OP_CHECK_IF(k.GetDim(DIM_B) != batch || k.GetDim(DIM_H) != kHeads ||
                    k.GetDim(DIM_T) != seqlen || k.GetDim(DIM_D) != kDim ||
                    q.GetDim(DIM_B) != batch || q.GetDim(DIM_H) != kHeads ||
                    q.GetDim(DIM_T) != seqlen || q.GetDim(DIM_D) != kDim,
                OP_LOGE(context->GetNodeName(), "q/k shapes do not match the logical B/H/T/K attributes."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(w.GetDim(DIM_B) != batch || w.GetDim(DIM_H) != vHeads ||
                    w.GetDim(DIM_T) != seqlen || w.GetDim(DIM_D) != kDim ||
                    u.GetDim(DIM_B) != batch || u.GetDim(DIM_H) != vHeads ||
                    u.GetDim(DIM_T) != seqlen || u.GetDim(DIM_D) != vDim ||
                    g.GetDim(0) != batch || g.GetDim(1) != vHeads || g.GetDim(2) != seqlen,
                OP_LOGE(context->GetNodeName(), "w/u/g shapes do not match B/HV/T/K/V."),
                return ge::GRAPH_FAILED);

    const auto *gkShapePtr = context->GetOptionalInputShape(INPUT_GK);
    OP_CHECK_IF(gkShapePtr != nullptr && !IsRank(gkShapePtr, 4),
                OP_LOGE(context->GetNodeName(), "gk must be rank 4."),
                return ge::GRAPH_FAILED);
    if (gkShapePtr != nullptr) {
        const gert::Shape gk = gkShapePtr->GetStorageShape();
        OP_CHECK_IF(gk.GetDim(DIM_B) != batch || gk.GetDim(DIM_H) != vHeads ||
                        gk.GetDim(DIM_T) != seqlen || gk.GetDim(DIM_D) != kDim,
                    OP_LOGE(context->GetNodeName(), "gk must have shape [B,HV,T,K]."),
                    return ge::GRAPH_FAILED);
    }

    const gert::Shape o = oShapePtr->GetStorageShape();
    bool validO = false;
    if (outputLayout == static_cast<int64_t>(GDN::ChunkFwdHOFusedOutputLayout::BNSD)) {
        validO = o.GetDimNum() == 4 && o.GetDim(0) == batch && o.GetDim(1) == vHeads &&
                 o.GetDim(2) == seqlen && o.GetDim(3) == vDim;
    } else if (outputLayout == static_cast<int64_t>(GDN::ChunkFwdHOFusedOutputLayout::BSND)) {
        validO = o.GetDimNum() == 4 && o.GetDim(0) == batch && o.GetDim(1) == seqlen &&
                 o.GetDim(2) == vHeads && o.GetDim(3) == vDim;
    } else if (outputLayout == static_cast<int64_t>(GDN::ChunkFwdHOFusedOutputLayout::TND)) {
        validO = batch == 1 && o.GetDimNum() == 3 && o.GetDim(0) == seqlen &&
                 o.GetDim(1) == vHeads && o.GetDim(2) == vDim;
    } else {
        validO = batch == 1 && o.GetDimNum() == 3 && o.GetDim(0) == vHeads &&
                 o.GetDim(1) == seqlen && o.GetDim(2) == vDim;
    }
    OP_CHECK_IF(!validO, OP_LOGE(context->GetNodeName(), "o shape does not match output_layout."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillWorkspace(ChunkFwdHOFusedTilingData &tiling, size_t systemWorkspace,
                              size_t producerCoreNum, size_t consumerCoreNum, size_t taskNum, bool useGk,
                              size_t &workspaceSize)
{
    // All offsets consumed by the kernel are relative to GetUserWorkspace().
    size_t offset = 0;
    size_t bytes = 0;
    int64_t regionOffset = 0;

    // Full, non-reused handoff tensors. H layout is [B, HV, NC, K, V] and
    // v_new layout is [B, HV, T, V], matching the established H/O addressing.
    OP_CHECK_IF(!CheckedMul({taskNum, static_cast<size_t>(tiling.get_numChunksPerBatch()),
                             static_cast<size_t>(tiling.get_kHeadDim()),
                             static_cast<size_t>(tiling.get_vHeadDim()), INPUT_ELEMENT_BYTES}, bytes) ||
                    !AllocateRegion(bytes, offset, regionOffset),
                , return ge::GRAPH_FAILED);
    tiling.set_handoffHWorkspaceOffset(regionOffset);
    OP_CHECK_IF(!CheckedMul({taskNum, static_cast<size_t>(tiling.get_seqlen()),
                             static_cast<size_t>(tiling.get_vHeadDim()), INPUT_ELEMENT_BYTES}, bytes) ||
                    !AllocateRegion(bytes, offset, regionOffset),
                , return ge::GRAPH_FAILED);
    tiling.set_handoffVWorkspaceOffset(regionOffset);

    // The local kernels derive the IB area as aligned(v_new end), so keep it
    // immediately after v_new. IB uses one slot per logical AIV and event.
    OP_CHECK_IF(offset > static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                , return ge::GRAPH_FAILED);
    tiling.set_pipelineSyncWorkspaceOffset(static_cast<int64_t>(offset));
    OP_CHECK_IF(!CheckedMul({static_cast<size_t>(tiling.get_activeCoreNum()),
                             static_cast<size_t>(GDN::CHUNK_FWD_HO_AIV_PER_MIXED_CORE),
                             static_cast<size_t>(GDN::CHUNK_FWD_HO_IB_EVENT_COUNT),
                             static_cast<size_t>(GDN::CHUNK_FWD_HO_IB_WORDS_PER_EVENT),
                             sizeof(int32_t)}, bytes) ||
                    !AllocateRegion(bytes, offset, regionOffset),
                , return ge::GRAPH_FAILED);

    OP_CHECK_IF(!CheckedMul({producerCoreNum, static_cast<size_t>(tiling.get_chunkSize()),
                             static_cast<size_t>(tiling.get_vHeadDim()), sizeof(float), PING_PONG_STAGES}, bytes) ||
                    !AllocateRegion(bytes, offset, regionOffset),
                , return ge::GRAPH_FAILED);
    tiling.set_vWorkspaceOffset(regionOffset);
    OP_CHECK_IF(!AllocateRegion(bytes, offset, regionOffset), , return ge::GRAPH_FAILED);
    tiling.set_vUpdateWorkspaceOffset(regionOffset);
    OP_CHECK_IF(offset > static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                , return ge::GRAPH_FAILED);
    tiling.set_kDecayWorkspaceOffset(static_cast<int64_t>(offset));
    if (useGk) {
        OP_CHECK_IF(!CheckedMul({producerCoreNum, static_cast<size_t>(tiling.get_chunkSize()),
                                 static_cast<size_t>(tiling.get_kHeadDim()), sizeof(float), PING_PONG_STAGES}, bytes) ||
                        !AllocateRegion(bytes, offset, regionOffset),
                    , return ge::GRAPH_FAILED);
        tiling.set_kDecayWorkspaceOffset(regionOffset);
    }
    OP_CHECK_IF(!CheckedMul({producerCoreNum, static_cast<size_t>(tiling.get_kHeadDim()),
                             static_cast<size_t>(tiling.get_vHeadDim()), sizeof(float), PING_PONG_STAGES}, bytes) ||
                    !AllocateRegion(bytes, offset, regionOffset),
                , return ge::GRAPH_FAILED);
    tiling.set_hWorkspaceOffset(regionOffset);
    OP_CHECK_IF(!CheckedMul({static_cast<size_t>(tiling.get_batch() + 1), sizeof(int64_t)}, bytes) ||
                    !AllocateRegion(bytes, offset, regionOffset),
                , return ge::GRAPH_FAILED);
    tiling.set_numSeqWorkspaceOffset(regionOffset);
    OP_CHECK_IF(!AllocateRegion(bytes, offset, regionOffset),
                , return ge::GRAPH_FAILED);
    tiling.set_numChunksWorkspaceOffset(regionOffset);

    OP_CHECK_IF(!CheckedMul({consumerCoreNum, static_cast<size_t>(tiling.get_chunkSize()),
                             static_cast<size_t>(tiling.get_vHeadDim()), sizeof(float), PING_PONG_STAGES}, bytes) ||
                    !AllocateRegion(bytes, offset, regionOffset),
                , return ge::GRAPH_FAILED);
    tiling.set_oVWorkspaceOffset(regionOffset);
    OP_CHECK_IF(!AllocateRegion(bytes, offset, regionOffset), , return ge::GRAPH_FAILED);
    tiling.set_oHWorkspaceOffset(regionOffset);
    OP_CHECK_IF(!CheckedMul({consumerCoreNum, static_cast<size_t>(tiling.get_chunkSize()),
                             static_cast<size_t>(tiling.get_chunkSize()), sizeof(float), PING_PONG_STAGES}, bytes) ||
                    !AllocateRegion(bytes, offset, regionOffset),
                , return ge::GRAPH_FAILED);
    tiling.set_oAttnWorkspaceOffset(regionOffset);
    OP_CHECK_IF(!AllocateRegion(bytes, offset, regionOffset), , return ge::GRAPH_FAILED);
    tiling.set_oAfterMaskWorkspaceOffset(regionOffset);
    OP_CHECK_IF(!CheckedMul({static_cast<size_t>(tiling.get_chunkSize()),
                             static_cast<size_t>(tiling.get_chunkSize())}, bytes) ||
                    !AllocateRegion(bytes, offset, regionOffset),
                , return ge::GRAPH_FAILED);
    tiling.set_oMaskWorkspaceOffset(regionOffset);
    tiling.set_oAPrimeWorkspaceOffset(0);
    OP_CHECK_IF(systemWorkspace > std::numeric_limits<size_t>::max() - offset,
                , return ge::GRAPH_FAILED);
    workspaceSize = systemWorkspace + offset;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillWorkspaceA5(ChunkFwdHOFusedTilingData &tiling, size_t systemWorkspace,
                                size_t producerCoreNum, size_t consumerCoreNum, size_t taskNum, bool useGk,
                                bool useOptimizedO, size_t &workspaceSize)
{
    size_t offset = 0;
    size_t bytes = 0;
    int64_t regionOffset = 0;

    OP_CHECK_IF(!CheckedMul({taskNum, static_cast<size_t>(tiling.get_numChunksPerBatch()),
                             static_cast<size_t>(tiling.get_kHeadDim()),
                             static_cast<size_t>(tiling.get_vHeadDim()), INPUT_ELEMENT_BYTES}, bytes) ||
                    !AllocateRegion(bytes, offset, regionOffset),
                , return ge::GRAPH_FAILED);
    tiling.set_handoffHWorkspaceOffset(regionOffset);
    OP_CHECK_IF(!CheckedMul({taskNum, static_cast<size_t>(tiling.get_seqlen()),
                             static_cast<size_t>(tiling.get_vHeadDim()), INPUT_ELEMENT_BYTES}, bytes) ||
                    !AllocateRegion(bytes, offset, regionOffset),
                , return ge::GRAPH_FAILED);
    tiling.set_handoffVWorkspaceOffset(regionOffset);

    OP_CHECK_IF(!CheckedMul({producerCoreNum, static_cast<size_t>(tiling.get_chunkSize()),
                             static_cast<size_t>(tiling.get_vHeadDim()), sizeof(float), PING_PONG_STAGES}, bytes) ||
                    !AllocateRegion(bytes, offset, regionOffset),
                , return ge::GRAPH_FAILED);
    tiling.set_vWorkspaceOffset(regionOffset);
    OP_CHECK_IF(!AllocateRegion(bytes, offset, regionOffset), , return ge::GRAPH_FAILED);
    tiling.set_vUpdateWorkspaceOffset(regionOffset);

    tiling.set_kDecayWorkspaceOffset(static_cast<int64_t>(offset));
    if (useGk) {
        OP_CHECK_IF(!CheckedMul({producerCoreNum, static_cast<size_t>(tiling.get_chunkSize()),
                                 static_cast<size_t>(tiling.get_kHeadDim()), sizeof(float), PING_PONG_STAGES}, bytes) ||
                        !AllocateRegion(bytes, offset, regionOffset),
                    , return ge::GRAPH_FAILED);
        tiling.set_kDecayWorkspaceOffset(regionOffset);
    }

    OP_CHECK_IF(!CheckedMul({producerCoreNum, static_cast<size_t>(tiling.get_kHeadDim()),
                             static_cast<size_t>(tiling.get_vHeadDim()), sizeof(float), PING_PONG_STAGES}, bytes) ||
                    !AllocateRegion(bytes, offset, regionOffset),
                , return ge::GRAPH_FAILED);
    tiling.set_hWorkspaceOffset(regionOffset);

    OP_CHECK_IF(!CheckedMul({static_cast<size_t>(tiling.get_batch() + 1), sizeof(int64_t)}, bytes) ||
                    !AllocateRegion(bytes, offset, regionOffset),
                , return ge::GRAPH_FAILED);
    tiling.set_numSeqWorkspaceOffset(regionOffset);
    OP_CHECK_IF(!AllocateRegion(bytes, offset, regionOffset), , return ge::GRAPH_FAILED);
    tiling.set_numChunksWorkspaceOffset(regionOffset);

    OP_CHECK_IF(!CheckedMul({static_cast<size_t>(tiling.get_activeCoreNum()),
                             static_cast<size_t>(GDN::CHUNK_FWD_HO_AIV_PER_MIXED_CORE),
                             static_cast<size_t>(GDN::CHUNK_FWD_HO_IB_EVENT_COUNT),
                             static_cast<size_t>(GDN::CHUNK_FWD_HO_IB_WORDS_PER_EVENT),
                             sizeof(int32_t)}, bytes) ||
                    !AllocateRegion(bytes, offset, regionOffset),
                , return ge::GRAPH_FAILED);
    tiling.set_pipelineSyncWorkspaceOffset(regionOffset);
    if (useOptimizedO) {
        OP_CHECK_IF(!CheckedMul({consumerCoreNum,
                                 static_cast<size_t>(GDN::CHUNK_FWD_O_APRIME_WORKSPACE_BYTES)}, bytes) ||
                        !AllocateRegion(bytes, offset, regionOffset),
                    , return ge::GRAPH_FAILED);
        tiling.set_oAPrimeWorkspaceOffset(regionOffset);
        tiling.set_oVWorkspaceOffset(0);
        tiling.set_oHWorkspaceOffset(0);
        tiling.set_oAttnWorkspaceOffset(0);
        tiling.set_oAfterMaskWorkspaceOffset(0);
        tiling.set_oMaskWorkspaceOffset(0);
    } else {
        OP_CHECK_IF(!CheckedMul({consumerCoreNum, static_cast<size_t>(tiling.get_chunkSize()),
                                 static_cast<size_t>(tiling.get_vHeadDim()), sizeof(float),
                                 PING_PONG_STAGES}, bytes) ||
                        !AllocateRegion(bytes, offset, regionOffset),
                    , return ge::GRAPH_FAILED);
        tiling.set_oVWorkspaceOffset(regionOffset);
        OP_CHECK_IF(!AllocateRegion(bytes, offset, regionOffset), , return ge::GRAPH_FAILED);
        tiling.set_oHWorkspaceOffset(regionOffset);
        OP_CHECK_IF(!CheckedMul({consumerCoreNum, static_cast<size_t>(tiling.get_chunkSize()),
                                 static_cast<size_t>(tiling.get_chunkSize()), sizeof(float),
                                 PING_PONG_STAGES}, bytes) ||
                        !AllocateRegion(bytes, offset, regionOffset),
                    , return ge::GRAPH_FAILED);
        tiling.set_oAttnWorkspaceOffset(regionOffset);
        OP_CHECK_IF(!AllocateRegion(bytes, offset, regionOffset), , return ge::GRAPH_FAILED);
        tiling.set_oAfterMaskWorkspaceOffset(regionOffset);
        OP_CHECK_IF(!CheckedMul({static_cast<size_t>(tiling.get_chunkSize()),
                                 static_cast<size_t>(tiling.get_chunkSize())}, bytes) ||
                        !AllocateRegion(bytes, offset, regionOffset),
                    , return ge::GRAPH_FAILED);
        tiling.set_oMaskWorkspaceOffset(regionOffset);
        tiling.set_oAPrimeWorkspaceOffset(0);
    }
    OP_CHECK_IF(systemWorkspace > std::numeric_limits<size_t>::max() - offset,
                , return ge::GRAPH_FAILED);
    workspaceSize = systemWorkspace + offset;
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus Tiling4ChunkFwdHOFused(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkFwdHOFused start.");
    ChunkFwdHOFusedTilingData tiling;
    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const bool *outputFinalStatePtr = attrs->GetAttrPointer<bool>(ATTR_OUTPUT_FINAL_STATE);
    const int64_t *chunkSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE);
    const double *scalePtr = attrs->GetAttrPointer<double>(ATTR_SCALE);
    const bool *useExp2Ptr = attrs->GetAttrPointer<bool>(ATTR_USE_EXP2);
    const char *outputLayoutStr = attrs->GetStr(ATTR_OUTPUT_LAYOUT);
    const int64_t *batchPtr = attrs->GetAttrPointer<int64_t>(ATTR_LOGICAL_BATCH);
    const int64_t *seqlenPtr = attrs->GetAttrPointer<int64_t>(ATTR_LOGICAL_SEQLEN);
    const int64_t *kHeadsPtr = attrs->GetAttrPointer<int64_t>(ATTR_LOGICAL_K_HEADS);
    const int64_t *vHeadsPtr = attrs->GetAttrPointer<int64_t>(ATTR_LOGICAL_V_HEADS);
    const int64_t *kDimPtr = attrs->GetAttrPointer<int64_t>(ATTR_LOGICAL_K_DIM);
    const int64_t *vDimPtr = attrs->GetAttrPointer<int64_t>(ATTR_LOGICAL_V_DIM);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputFinalStatePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, chunkSizePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, scalePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, batchPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, seqlenPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, kHeadsPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, vHeadsPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, kDimPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, vDimPtr);

    const bool outputFinalState = *outputFinalStatePtr;
    const bool useExp2 = useExp2Ptr != nullptr && *useExp2Ptr;
    const int64_t batch = *batchPtr;
    const int64_t seqlen = *seqlenPtr;
    const int64_t kHeads = *kHeadsPtr;
    const int64_t vHeads = *vHeadsPtr;
    const int64_t kDim = *kDimPtr;
    const int64_t vDim = *vDimPtr;
    const int64_t chunkSize = *chunkSizePtr;
    OP_CHECK_IF(batch <= 0 || seqlen <= 0 || kHeads <= 0 || vHeads <= 0 ||
                    vHeads % kHeads != 0 || kDim != SUPPORTED_K ||
                    (vDim != SUPPORTED_V128 && vDim != SUPPORTED_V256) ||
                    (chunkSize != CHUNK_64 && chunkSize != CHUNK_128),
                OP_LOGE(context->GetNodeName(),
                        "Require positive B/H/T, HV divisible by HK, K=128, V=128/256, chunk=64/128."),
                return ge::GRAPH_FAILED);

    const auto *cuShape = context->GetOptionalInputShape(INPUT_CU_SEQLENS);
    const auto *chunkIndicesShape = context->GetOptionalInputShape(INPUT_CHUNK_INDICES);
    OP_CHECK_IF((cuShape == nullptr) != (chunkIndicesShape == nullptr),
                OP_LOGE(context->GetNodeName(), "cu_seqlens and chunk_indices must appear together."),
                return ge::GRAPH_FAILED);
    const bool isVarlen = cuShape != nullptr;
    OP_CHECK_IF(isVarlen,
                OP_LOGE(context->GetNodeName(),
                        "The first H/O core-pipeline implementation supports fixed-length input only."),
                return ge::GRAPH_FAILED);
    const int64_t tokenBatch = isVarlen ? cuShape->GetStorageShape().GetDim(0) - 1 : 1;
    const int64_t chunkNum = isVarlen
                                 ? chunkIndicesShape->GetStorageShape().GetDim(0) / 2
                                 : batch * ((seqlen + chunkSize - 1) / chunkSize);
    OP_CHECK_IF(tokenBatch <= 0 || chunkNum <= 0 ||
                    (isVarlen && chunkIndicesShape->GetStorageShape().GetDim(0) % 2 != 0),
                OP_LOGE(context->GetNodeName(), "Invalid varlen sequence/chunk metadata shape."),
                return ge::GRAPH_FAILED);

    int64_t outputLayout = -1;
    const char *layout = outputLayoutStr == nullptr ? "BNSD" : outputLayoutStr;
    if (std::strcmp(layout, "BNSD") == 0) {
        outputLayout = static_cast<int64_t>(GDN::ChunkFwdHOFusedOutputLayout::BNSD);
    } else if (std::strcmp(layout, "BSND") == 0) {
        outputLayout = static_cast<int64_t>(GDN::ChunkFwdHOFusedOutputLayout::BSND);
    } else if (std::strcmp(layout, "TND") == 0) {
        outputLayout = static_cast<int64_t>(GDN::ChunkFwdHOFusedOutputLayout::TND);
    } else if (std::strcmp(layout, "NTD") == 0) {
        outputLayout = static_cast<int64_t>(GDN::ChunkFwdHOFusedOutputLayout::NTD);
    } else {
        OP_LOGE(context->GetNodeName(), "Unsupported output_layout: %s.", layout);
        return ge::GRAPH_FAILED;
    }
    const bool validModeLayout = useExp2
                                     ? outputLayout == static_cast<int64_t>(GDN::ChunkFwdHOFusedOutputLayout::BSND) ||
                                           outputLayout == static_cast<int64_t>(GDN::ChunkFwdHOFusedOutputLayout::TND)
                                     : outputLayout == static_cast<int64_t>(GDN::ChunkFwdHOFusedOutputLayout::BNSD) ||
                                           outputLayout == static_cast<int64_t>(GDN::ChunkFwdHOFusedOutputLayout::NTD);
    OP_CHECK_IF(!validModeLayout,
                OP_LOGE(context->GetNodeName(), "exp2 supports BSND/TND; exp supports BNSD/NTD."),
                return ge::GRAPH_FAILED);

    const auto *qDesc = context->GetInputDesc(INPUT_Q);
    const auto *kDesc = context->GetInputDesc(INPUT_K);
    const auto *wDesc = context->GetInputDesc(INPUT_W);
    const auto *uDesc = context->GetInputDesc(INPUT_U);
    const auto *gDesc = context->GetInputDesc(INPUT_G);
    OP_CHECK_NULL_WITH_CONTEXT(context, qDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, kDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, wDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, uDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gDesc);
    const ge::DataType inputType = kDesc->GetDataType();
    const ge::DataType gateType = gDesc->GetDataType();
    OP_CHECK_IF(!IsSupportedInputType(inputType) || qDesc->GetDataType() != inputType ||
                    wDesc->GetDataType() != inputType || uDesc->GetDataType() != inputType ||
                    !IsSupportedGateType(gateType, inputType),
                OP_LOGE(context->GetNodeName(), "Invalid q/k/w/u or g dtype combination."),
                return ge::GRAPH_FAILED);
    const auto *gkDesc = context->GetOptionalInputDesc(INPUT_GK);
    OP_CHECK_IF(gkDesc != nullptr && gkDesc->GetDataType() != gateType,
                OP_LOGE(context->GetNodeName(), "gk dtype must match g dtype."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(useExp2 != (gkDesc != nullptr),
                OP_LOGE(context->GetNodeName(),
                        "use_exp2 is the KDA mode and must match the presence of gk."),
                return ge::GRAPH_FAILED);
    const auto *initialStateDesc = context->GetOptionalInputDesc(INPUT_INITIAL_STATE);
    const auto *oDesc = context->GetOutputDesc(OUTPUT_O);
    OP_CHECK_NULL_WITH_CONTEXT(context, oDesc);
    OP_CHECK_IF(oDesc->GetDataType() != inputType,
                OP_LOGE(context->GetNodeName(), "o dtype must match q/k/w/u."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(initialStateDesc != nullptr && initialStateDesc->GetDataType() != inputType &&
                    initialStateDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(context->GetNodeName(), "initial_state dtype must match inputs or be float32."),
                return ge::GRAPH_FAILED);
    const int64_t sequenceCount = isVarlen ? tokenBatch : batch;
    const auto *initialStateShape = context->GetOptionalInputShape(INPUT_INITIAL_STATE);
    if (initialStateShape != nullptr) {
        OP_CHECK_IF(!IsRank(initialStateShape, 4),
                    OP_LOGE(context->GetNodeName(), "initial_state must be rank 4."),
                    return ge::GRAPH_FAILED);
        const gert::Shape state = initialStateShape->GetStorageShape();
        OP_CHECK_IF(state.GetDim(0) != sequenceCount || state.GetDim(1) != vHeads ||
                        state.GetDim(2) != kDim || state.GetDim(3) != vDim,
                    OP_LOGE(context->GetNodeName(), "Internal initial_state must be [N,HV,K,V]."),
                    return ge::GRAPH_FAILED);
    }
    const auto *finalStateShape = context->GetOutputShape(OUTPUT_FINAL_STATE);
    const auto *finalStateDesc = context->GetOutputDesc(OUTPUT_FINAL_STATE);
    if (outputFinalState) {
        OP_CHECK_IF(!IsRank(finalStateShape, 4) || finalStateDesc == nullptr,
                    OP_LOGE(context->GetNodeName(), "final_state output is required and must be rank 4."),
                    return ge::GRAPH_FAILED);
        const gert::Shape state = finalStateShape->GetStorageShape();
        const ge::DataType expectedStateType = initialStateDesc == nullptr
                                                   ? ge::DT_FLOAT
                                                   : initialStateDesc->GetDataType();
        OP_CHECK_IF(state.GetDim(0) != sequenceCount || state.GetDim(1) != vHeads ||
                        state.GetDim(2) != kDim || state.GetDim(3) != vDim ||
                        finalStateDesc->GetDataType() != expectedStateType,
                    OP_LOGE(context->GetNodeName(), "final_state shape or dtype is invalid."),
                    return ge::GRAPH_FAILED);
    }
    const int64_t stateType = initialStateDesc == nullptr
                                  ? static_cast<int64_t>(GDN::ChunkFwdHOFusedDtype::FP32)
                                  : DtypeToEnum(initialStateDesc->GetDataType());

    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const NpuArch npuArch = platform.GetCurNpuArch();
    OP_CHECK_IF(npuArch != NpuArch::DAV_2201 && npuArch != NpuArch::DAV_3510,
                OP_LOGE(context->GetNodeName(), "ChunkFwdHOFused supports Atlas A2 and A5 only."),
                return ge::GRAPH_FAILED);
    const bool isA5 = npuArch == NpuArch::DAV_3510;
    if (isA5 && useExp2) {
        OP_CHECK_IF(inputType != ge::DT_BF16 ||
                        (gateType != ge::DT_FLOAT && gateType != ge::DT_BF16) ||
                        chunkSize != GDN::CHUNK_FWD_O_A5_BT ||
                        kDim != GDN::CHUNK_FWD_O_A5_K || vDim != GDN::CHUNK_FWD_O_A5_V ||
                        vHeads / kHeads < 1 || vHeads / kHeads > 4,
                    OP_LOGE(context->GetNodeName(),
                            "A5 exp2 path requires BF16 data, BF16/FP32 gates, "
                            "chunk=64, K=V=128 and HV/HK in [1,4]."),
                    return ge::GRAPH_FAILED);
    } else if (!isA5) {
        OP_CHECK_IF(useExp2,
                    OP_LOGE(context->GetNodeName(),
                            "The A2 H/O core-pipeline implementation supports exp mode only."),
                    return ge::GRAPH_FAILED);
    }

    size_t taskNum = 0;
    OP_CHECK_IF(!CheckedMul({static_cast<size_t>(batch), static_cast<size_t>(vHeads)}, taskNum) ||
                    taskNum > static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                OP_LOGE(context->GetNodeName(), "B * HV overflows the core-pipeline task count."),
                return ge::GRAPH_FAILED);
    const size_t physicalCoreNum = platform.GetCoreNumAic();
    OP_CHECK_IF(physicalCoreNum == 0,
                OP_LOGE(context->GetNodeName(), "No AIC core is available."),
                return ge::GRAPH_FAILED);
    size_t activeCoreNum = 0;
    if (taskNum <= physicalCoreNum) {
        activeCoreNum = taskNum;
    } else {
        // Keep the saturated-core path explicit for its follow-up specialization.
        activeCoreNum = physicalCoreNum;
    }
    const size_t producerCoreNum = activeCoreNum / 2;
    OP_CHECK_IF(producerCoreNum == 0,
                OP_LOGE(context->GetNodeName(),
                        "The core pipeline requires at least two active AIC cores, got %zu.", activeCoreNum),
                return ge::GRAPH_FAILED);
    const size_t consumerCoreNum = activeCoreNum - producerCoreNum;

    OP_CHECK_IF(ValidateTensorContracts(context, batch, seqlen, kHeads, vHeads, kDim, vDim,
                                        outputLayout) != ge::GRAPH_SUCCESS,
                , return ge::GRAPH_FAILED);

    tiling.set_batch(isVarlen ? tokenBatch : batch);
    tiling.set_seqlen(seqlen);
    tiling.set_kNumHead(kHeads);
    tiling.set_vNumHead(vHeads);
    tiling.set_kHeadDim(kDim);
    tiling.set_vHeadDim(vDim);
    tiling.set_chunkSize(chunkSize);
    tiling.set_useInitialState(initialStateDesc != nullptr);
    tiling.set_storeFinalState(outputFinalState);
    tiling.set_dataType(DtypeToEnum(inputType));
    tiling.set_gDataType(DtypeToEnum(gateType));
    tiling.set_stateDataType(stateType);
    tiling.set_isVariedLen(isVarlen ? 1 : 0);
    tiling.set_shapeBatch(batch);
    tiling.set_tokenBatch(tokenBatch);
    tiling.set_useG(true);
    tiling.set_useGk(gkDesc != nullptr);
    tiling.set_useExp2(useExp2);
    tiling.set_outputLayout(outputLayout);
    tiling.set_scale(static_cast<float>(*scalePtr));
    tiling.set_chunkNum(chunkNum);
    tiling.set_numChunksPerBatch((seqlen + chunkSize - 1) / chunkSize);
    tiling.set_hvPerHk(vHeads / kHeads);
    tiling.set_taskGroupSize(vHeads / kHeads == 3 ? 3 : 4);
    tiling.set_producerCoreNum(static_cast<int64_t>(producerCoreNum));
    tiling.set_consumerCoreBase(static_cast<int64_t>(producerCoreNum));
    tiling.set_activeCoreNum(static_cast<int64_t>(activeCoreNum));
    tiling.set_pipelineEventCount(GDN::CHUNK_FWD_HO_IB_EVENT_COUNT);

    OP_CHECK_IF(tiling.GetDataSize() != sizeof(GDN::ChunkFwdHOFusedTilingData),
                OP_LOGE(context->GetNodeName(), "Host/kernel fused tiling size mismatch: %zu vs %zu.",
                        tiling.GetDataSize(), sizeof(GDN::ChunkFwdHOFusedTilingData)),
                return ge::GRAPH_FAILED);

    size_t workspaceSize = 0;
    const ge::graphStatus workspaceStatus = isA5
        ? FillWorkspaceA5(tiling, platform.GetLibApiWorkSpaceSize(), producerCoreNum,
                          consumerCoreNum, taskNum, gkDesc != nullptr, useExp2, workspaceSize)
        : FillWorkspace(tiling, platform.GetLibApiWorkSpaceSize(), producerCoreNum,
                        consumerCoreNum, taskNum,
                        gkDesc != nullptr, workspaceSize);
    OP_CHECK_IF(workspaceStatus != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Workspace calculation overflow."),
                return ge::GRAPH_FAILED);

    auto *rawTiling = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTiling);
    auto *rawTilingData = rawTiling->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    OP_CHECK_IF(rawTiling->GetCapacity() < tiling.GetDataSize(),
                OP_LOGE(context->GetNodeName(), "Raw tiling capacity is insufficient: %zu < %zu.",
                        rawTiling->GetCapacity(), tiling.GetDataSize()),
                return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(rawTilingData, rawTiling->GetCapacity());
    rawTiling->SetDataSize(tiling.GetDataSize());

    uint64_t tilingKey = static_cast<uint64_t>(vDim == SUPPORTED_V256
                                                   ? GDN::ChunkFwdHOFusedTilingKey::V256_EXP
                                                   : GDN::ChunkFwdHOFusedTilingKey::V128_EXP);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(static_cast<uint32_t>(activeCoreNum));
    OP_CHECK_IF(context->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to enable mixed-core batch schedule mode."),
                return ge::GRAPH_FAILED);
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = workspaceSize;
    OP_LOGD(context->GetNodeName(), "ChunkFwdHOFused tiling complete: key=%lu workspace=%zu.",
            tilingKey, workspaceSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkFwdHOFused(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkFwdHOFused)
    .Tiling(Tiling4ChunkFwdHOFused)
    .TilingParse<ChunkFwdHOFusedCompileInfo>(TilingPrepareForChunkFwdHOFused);

} // namespace optiling
