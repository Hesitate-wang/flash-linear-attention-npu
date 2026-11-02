/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "chunk_gated_delta_rule_fwd.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ChunkGatedDeltaRuleFwd);

namespace {

const aclTensor *ConvertMetadata(const aclIntArray *values, aclOpExecutor *executor)
{
    if (values == nullptr) {
        return nullptr;
    }
    const aclTensor *tensor = executor->ConvertToTensor(values, DataType::DT_INT64);
    if (tensor != nullptr) {
        auto *mutableTensor = const_cast<aclTensor *>(tensor);
        mutableTensor->SetStorageFormat(Format::FORMAT_ND);
        mutableTensor->SetViewFormat(Format::FORMAT_ND);
        mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
    }
    return tensor;
}

} // namespace

const std::array<const aclTensor *, 4> ChunkGatedDeltaRuleFwd(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *beta,
    const aclTensor *aStorage,
    const aclTensor *rawG,
    const aclTensor *gkOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool outputFinalState,
    int64_t chunkSize,
    double scale,
    bool outputGCumsum,
    const aclTensor *oOut,
    const aclTensor *finalStateOut,
    const aclTensor *gCumsumBthOut,
    const aclTensor *aOut,
    int64_t rawGLayout,
    int64_t qkvLayout,
    int64_t oLayout,
    bool stateVFirst,
    bool allowNegEigval,
    aclOpExecutor *executor,
    const ChunkGatedDeltaRuleFwdPrepareArgs *prepare)
{
    L0_DFX(ChunkGatedDeltaRuleFwd, q, k, v, beta, aStorage, rawG, gkOptional, initialStateOptional,
           cuSeqlensOptional, chunkIndicesOptional, outputFinalState, chunkSize, scale, outputGCumsum,
           oOut, finalStateOut, gCumsumBthOut, aOut, rawGLayout, qkvLayout, oLayout, stateVFirst,
           allowNegEigval);
    const aclTensor *cuSeqlens = ConvertMetadata(cuSeqlensOptional, executor);
    const aclTensor *chunkIndices = ConvertMetadata(chunkIndicesOptional, executor);
    if ((cuSeqlensOptional != nullptr && cuSeqlens == nullptr) ||
        (chunkIndicesOptional != nullptr && chunkIndices == nullptr)) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Convert Phase 6 metadata failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }

    const bool fusedPrepare = prepare != nullptr;
    const bool useQkL2norm = fusedPrepare && prepare->useQkL2norm;
    const bool useExp2 = fusedPrepare && prepare->useExp2;
    const bool useBetaSigmoid = fusedPrepare && prepare->useBetaSigmoid;
    const bool outputA = fusedPrepare && prepare->outputA;
    const aclTensor *aLog = fusedPrepare ? prepare->aLog : nullptr;
    const aclTensor *dtBias = fusedPrepare ? prepare->dtBias : nullptr;
    const aclTensor *qHat = fusedPrepare ? prepare->qHat : nullptr;
    const aclTensor *kHat = fusedPrepare ? prepare->kHat : nullptr;
    const aclTensor *qRstd = fusedPrepare ? prepare->qRstd : nullptr;
    const aclTensor *kRstd = fusedPrepare ? prepare->kRstd : nullptr;
    const aclTensor *betaEff = fusedPrepare ? prepare->betaEff : nullptr;
    const aclTensor *h = fusedPrepare ? prepare->h : nullptr;
    const aclTensor *w = fusedPrepare ? prepare->w : nullptr;
    const aclTensor *u = fusedPrepare ? prepare->u : nullptr;
    const aclTensor *vNew = fusedPrepare ? prepare->vNew : nullptr;
    const aclTensor *gCumsumBht = fusedPrepare ? prepare->gCumsumBht : nullptr;
    const aclError ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkGatedDeltaRuleFwd,
        OP_INPUT(q, k, v, beta, aStorage, rawG, gkOptional, initialStateOptional,
                 cuSeqlens, chunkIndices, aLog, dtBias),
        OP_OUTPUT(oOut, finalStateOut, gCumsumBthOut, aOut,
                  qHat, kHat, qRstd, kRstd, betaEff, h, w, u, vNew, gCumsumBht),
        OP_ATTR(outputFinalState, chunkSize, scale, outputGCumsum, rawGLayout, qkvLayout, oLayout,
                stateVFirst, allowNegEigval, fusedPrepare, useQkL2norm, useExp2,
                useBetaSigmoid, outputA));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE ChunkGatedDeltaRuleFwd failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }
    return {oOut, finalStateOut, gCumsumBthOut, aOut};
}

} // namespace l0op
