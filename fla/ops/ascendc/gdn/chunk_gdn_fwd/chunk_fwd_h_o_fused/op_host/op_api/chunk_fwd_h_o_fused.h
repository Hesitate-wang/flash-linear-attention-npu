/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef OP_API_INC_LEVEL0_OP_CHUNK_FWD_H_O_FUSED_H
#define OP_API_INC_LEVEL0_OP_CHUNK_FWD_H_O_FUSED_H

#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 2> ChunkFwdHOFused(
    const aclTensor *k,
    const aclTensor *q,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *g,
    const aclTensor *gkOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool outputFinalState,
    int64_t chunkSize,
    double scale,
    bool useExp2,
    const char *outputLayout,
    const aclTensor *oOut,
    const aclTensor *finalStateOut,
    aclnnStatus *status,
    aclOpExecutor *executor);
} // namespace l0op
#endif
