/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "chunk_fwd_h_o_fused.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include <string>

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkFwdHOFused);

namespace {
const aclTensor *ConvertIntArray(const aclIntArray *value, aclOpExecutor *executor)
{
    if (value == nullptr) {
        return nullptr;
    }
    const aclTensor *tensor = executor->ConvertToTensor(value, DataType::DT_INT64);
    if (tensor != nullptr) {
        auto *mutableTensor = const_cast<aclTensor *>(tensor);
        mutableTensor->SetStorageFormat(Format::FORMAT_ND);
        mutableTensor->SetViewFormat(Format::FORMAT_ND);
        mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
    }
    return tensor;
}
} // namespace

const std::array<const aclTensor *, 4> ChunkFwdHOFused(
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *g,
    const aclTensor *gkOptional,
    const aclTensor *initialStateOptional,
    const aclTensor *q,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool outputFinalState,
    int64_t chunkSize,
    double scale,
    bool useExp2,
    const char *outputLayout,
    const aclTensor *hOut,
    const aclTensor *vNewOut,
    const aclTensor *finalStateOut,
    const aclTensor *oOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkFwdHOFused, k, w, u, g, gkOptional, initialStateOptional, q,
           cuSeqlensOptional, chunkIndicesOptional, outputFinalState, chunkSize, scale,
           useExp2, outputLayout, hOut, vNewOut, finalStateOut, oOut);
    const aclTensor *actualCuSeqlens = ConvertIntArray(cuSeqlensOptional, executor);
    const aclTensor *actualChunkIndices = ConvertIntArray(chunkIndicesOptional, executor);
    if ((cuSeqlensOptional != nullptr && actualCuSeqlens == nullptr) ||
        (chunkIndicesOptional != nullptr && actualChunkIndices == nullptr)) {
        return {nullptr, nullptr, nullptr, nullptr};
    }

    const auto &kShape = k->GetViewShape();
    const auto &uShape = u->GetViewShape();
    const int64_t logicalBatch = kShape.GetDim(0);
    const int64_t logicalKHeads = kShape.GetDim(1);
    const int64_t logicalSeqlen = kShape.GetDim(2);
    const int64_t logicalKDim = kShape.GetDim(3);
    const int64_t logicalVHeads = uShape.GetDim(1);
    const int64_t logicalVDim = uShape.GetDim(3);
    const std::string outputLayoutStr(outputLayout == nullptr ? "BNSD" : outputLayout);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkFwdHOFused,
        OP_INPUT(k, w, u, g, gkOptional, initialStateOptional, q,
                 actualCuSeqlens, actualChunkIndices),
        OP_OUTPUT(hOut, vNewOut, finalStateOut, oOut),
        OP_ATTR(outputFinalState, chunkSize, scale, useExp2, outputLayoutStr,
                logicalBatch, logicalSeqlen, logicalKHeads, logicalVHeads,
                logicalKDim, logicalVDim));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }
    return {hOut, vNewOut, finalStateOut, oOut};
}
} // namespace l0op
