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
    if (fusedPrepare) {
        // GE packs null optional outputs into earlier slots, which would shift
        // h/w/u/v_new/g_cumsum_bht off their declared indices 9..13 (the Prepare
        // l0op documents the same pitfall). Occupy every null optional slot
        // with a dummy; the kernel only writes these outputs when the matching
        // feature flag is on, and then the caller supplies real tensors.
        op::Shape dummyShape;
        dummyShape.SetDimNum(1);
        dummyShape.SetDim(0, 1);
        const DataType ioDtype = q->GetDataType();
        const aclTensor **slots[] = {&qHat, &kHat, &h, &w, &u, &vNew};
        for (const aclTensor **slot : slots) {
            if (*slot == nullptr) {
                *slot = executor->AllocTensor(dummyShape, ioDtype, Format::FORMAT_ND);
            }
        }
        const aclTensor **fp32Slots[] = {&qRstd, &kRstd, &betaEff, &gCumsumBht};
        for (const aclTensor **slot : fp32Slots) {
            if (*slot == nullptr) {
                *slot = executor->AllocTensor(dummyShape, DataType::DT_FLOAT, Format::FORMAT_ND);
            }
        }
        if (qHat == nullptr || kHat == nullptr || qRstd == nullptr || kRstd == nullptr ||
            betaEff == nullptr || h == nullptr || w == nullptr || u == nullptr ||
            vNew == nullptr || gCumsumBht == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create optional output placeholder failed.");
            return {nullptr, nullptr, nullptr, nullptr};
        }
    }
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
