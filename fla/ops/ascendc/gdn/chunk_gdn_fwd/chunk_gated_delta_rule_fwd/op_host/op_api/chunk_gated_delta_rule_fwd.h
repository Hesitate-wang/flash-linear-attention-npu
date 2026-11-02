/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef OP_API_INC_CHUNK_GATED_DELTA_RULE_FWD_H
#define OP_API_INC_CHUNK_GATED_DELTA_RULE_FWD_H

#include "aclnn/aclnn_base.h"
#include <array>

namespace l0op {

struct ChunkGatedDeltaRuleFwdPrepareArgs {
    const aclTensor *aLog;
    const aclTensor *dtBias;
    const aclTensor *qHat;
    const aclTensor *kHat;
    const aclTensor *qRstd;
    const aclTensor *kRstd;
    const aclTensor *betaEff;
    const aclTensor *h;
    const aclTensor *w;
    const aclTensor *u;
    const aclTensor *vNew;
    const aclTensor *gCumsumBht;
    bool useQkL2norm;
    bool useExp2;
    bool useBetaSigmoid;
    bool outputA;
};

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
    const ChunkGatedDeltaRuleFwdPrepareArgs *prepare = nullptr);

} // namespace l0op

#endif // OP_API_INC_CHUNK_GATED_DELTA_RULE_FWD_H
