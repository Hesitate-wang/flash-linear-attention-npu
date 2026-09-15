/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "aclnn_chunk_fwd_h_o_fused.h"
#include "chunk_fwd_h_o_fused.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/transdata.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include <algorithm>
#include <cstring>
#include <vector>

using namespace op;

namespace {

struct ChunkFwdHOFusedParams {
    const aclTensor *k;
    const aclTensor *w;
    const aclTensor *u;
    const aclTensor *g;
    const aclTensor *gkOptional;
    const aclTensor *initialStateOptional;
    const aclTensor *q;
    const aclIntArray *cuSeqlensOptional;
    const aclIntArray *chunkIndicesOptional;
    bool outputFinalState;
    int64_t chunkSize;
    double scale;
    bool useExp2;
    bool stateVFirst;
    const char *outputLayout;
    const aclTensor *hOut;
    const aclTensor *vNewOut;
    const aclTensor *finalStateOut;
    const aclTensor *oOut;
};

op::Shape MakeShape(std::initializer_list<int64_t> dims)
{
    op::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

op::Shape SwapLastTwo(const op::Shape &input)
{
    op::Shape output;
    const size_t rank = input.GetDimNum();
    for (size_t index = 0; index < rank; ++index) {
        if (index + 2 == rank) {
            output.AppendDim(input.GetDim(rank - 1));
        } else if (index + 1 == rank) {
            output.AppendDim(input.GetDim(rank - 2));
        } else {
            output.AppendDim(input.GetDim(index));
        }
    }
    return output;
}

const aclTensor *TransposeLastTwo(const aclTensor *input, aclOpExecutor *executor)
{
    const size_t rank = input->GetViewShape().GetDimNum();
    std::vector<int64_t> permutation(rank);
    for (size_t index = 0; index < rank; ++index) {
        permutation[index] = static_cast<int64_t>(index);
    }
    std::swap(permutation[rank - 2], permutation[rank - 1]);
    const aclIntArray *perm = executor->AllocIntArray(permutation.data(), permutation.size());
    CHECK_RET(perm != nullptr, nullptr);
    return l0op::Transpose(input, perm, executor);
}

aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckRequired(const ChunkFwdHOFusedParams &params)
{
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(params.w != nullptr, ACLNN_ERR_PARAM_NULLPTR, "w must not be nullptr.");
    CHECK_COND(params.u != nullptr, ACLNN_ERR_PARAM_NULLPTR, "u must not be nullptr.");
    CHECK_COND(params.g != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "g must not be nullptr because the fused O stage consumes it.");
    CHECK_COND(params.q != nullptr, ACLNN_ERR_PARAM_NULLPTR, "q must not be nullptr.");
    CHECK_COND(params.hOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "hOut must not be nullptr.");
    CHECK_COND(params.vNewOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "vNewOut must not be nullptr.");
    CHECK_COND(params.oOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "oOut must not be nullptr.");
    CHECK_COND(!params.outputFinalState || params.finalStateOut != nullptr,
               ACLNN_ERR_PARAM_NULLPTR,
               "finalStateOut must be provided when outputFinalState is true.");
    CHECK_COND((params.cuSeqlensOptional == nullptr) == (params.chunkIndicesOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional and chunkIndicesOptional must be both present or both absent.");
    CHECK_COND(params.chunkSize == 64 || params.chunkSize == 128,
               ACLNN_ERR_PARAM_INVALID, "chunkSize must be 64 or 128.");
    return ACLNN_SUCCESS;
}

aclnnStatus CheckShapes(const ChunkFwdHOFusedParams &params)
{
    const auto &k = params.k->GetViewShape();
    const auto &w = params.w->GetViewShape();
    const auto &u = params.u->GetViewShape();
    const auto &q = params.q->GetViewShape();
    const auto &g = params.g->GetViewShape();
    CHECK_COND(k.GetDimNum() == 4 && w.GetDimNum() == 4 && u.GetDimNum() == 4 &&
                   q.GetDimNum() == 4 && g.GetDimNum() == 3,
               ACLNN_ERR_PARAM_INVALID, "k/w/u/q must be rank 4 and g must be rank 3.");
    CHECK_COND(q.GetDim(0) == k.GetDim(0) && q.GetDim(1) == k.GetDim(1) &&
                   q.GetDim(2) == k.GetDim(2) && q.GetDim(3) == k.GetDim(3),
               ACLNN_ERR_PARAM_INVALID, "q and k must have the same shape.");
    CHECK_COND(w.GetDim(0) == k.GetDim(0) && u.GetDim(0) == k.GetDim(0) &&
                   w.GetDim(1) == u.GetDim(1) && w.GetDim(2) == k.GetDim(2) &&
                   u.GetDim(2) == k.GetDim(2) && w.GetDim(3) == k.GetDim(3),
               ACLNN_ERR_PARAM_INVALID, "w/u must match k in B/T and their H/K/V roles.");
    CHECK_COND(u.GetDim(1) >= k.GetDim(1) && u.GetDim(1) % k.GetDim(1) == 0,
               ACLNN_ERR_PARAM_INVALID, "HV must be divisible by HK.");
    CHECK_COND(g.GetDim(0) == u.GetDim(0) && g.GetDim(1) == u.GetDim(1) &&
                   g.GetDim(2) == u.GetDim(2),
               ACLNN_ERR_PARAM_INVALID, "g must have shape [B, HV, T].");
    if (params.gkOptional != nullptr) {
        const auto &gk = params.gkOptional->GetViewShape();
        CHECK_COND(gk.GetDimNum() == 4 && gk.GetDim(0) == k.GetDim(0) &&
                       gk.GetDim(1) == u.GetDim(1) && gk.GetDim(2) == k.GetDim(2) &&
                       gk.GetDim(3) == k.GetDim(3),
                   ACLNN_ERR_PARAM_INVALID, "gk must have shape [B, HV, T, K].");
    }

    const int64_t batch = k.GetDim(0);
    const int64_t hv = u.GetDim(1);
    const int64_t tokens = k.GetDim(2);
    const int64_t kDim = k.GetDim(3);
    const int64_t vDim = u.GetDim(3);
    const int64_t sequences = params.cuSeqlensOptional == nullptr
                                  ? batch
                                  : static_cast<int64_t>(params.cuSeqlensOptional->Size()) - 1;
    const int64_t chunks = params.chunkIndicesOptional == nullptr
                               ? (tokens + params.chunkSize - 1) / params.chunkSize
                               : static_cast<int64_t>(params.chunkIndicesOptional->Size()) / 2;
    CHECK_COND(params.chunkIndicesOptional == nullptr || params.chunkIndicesOptional->Size() % 2 == 0,
               ACLNN_ERR_PARAM_INVALID, "chunkIndicesOptional must contain flattened pairs.");

    const auto &h = params.hOut->GetViewShape();
    CHECK_COND(h.GetDimNum() == 5 && h.GetDim(0) == batch && h.GetDim(1) == hv && h.GetDim(2) == chunks,
               ACLNN_ERR_PARAM_INVALID, "hOut must have shape [B, HV, chunks, K, V].");
    const int64_t hK = params.stateVFirst ? h.GetDim(4) : h.GetDim(3);
    const int64_t hV = params.stateVFirst ? h.GetDim(3) : h.GetDim(4);
    CHECK_COND(hK == kDim && hV == vDim, ACLNN_ERR_PARAM_INVALID,
               "hOut state dimensions do not match K/V.");
    const auto &vNew = params.vNewOut->GetViewShape();
    CHECK_COND(vNew.GetDimNum() == 4 && vNew.GetDim(0) == batch && vNew.GetDim(1) == hv &&
                   vNew.GetDim(2) == tokens && vNew.GetDim(3) == vDim,
               ACLNN_ERR_PARAM_INVALID, "vNewOut must have shape [B, HV, T, V].");

    const aclTensor *states[] = {params.initialStateOptional, params.finalStateOut};
    for (const aclTensor *stateTensor : states) {
        if (stateTensor == nullptr) {
            continue;
        }
        const auto &state = stateTensor->GetViewShape();
        CHECK_COND(state.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID,
                   "initial/final state must be rank 4.");
        const int64_t stateK = params.stateVFirst ? state.GetDim(3) : state.GetDim(2);
        const int64_t stateV = params.stateVFirst ? state.GetDim(2) : state.GetDim(3);
        CHECK_COND(state.GetDim(0) == sequences &&
                       state.GetDim(1) == hv && stateK == kDim && stateV == vDim,
                   ACLNN_ERR_PARAM_INVALID, "initial/final state shape does not match N/HV/K/V.");
    }

    const char *layout = params.outputLayout == nullptr ? "BNSD" : params.outputLayout;
    const auto &o = params.oOut->GetViewShape();
    bool validO = false;
    if (std::strcmp(layout, "BNSD") == 0) {
        validO = o.GetDimNum() == 4 && o.GetDim(0) == batch && o.GetDim(1) == hv &&
                 o.GetDim(2) == tokens && o.GetDim(3) == vDim;
    } else if (std::strcmp(layout, "BSND") == 0) {
        validO = o.GetDimNum() == 4 && o.GetDim(0) == batch && o.GetDim(1) == tokens &&
                 o.GetDim(2) == hv && o.GetDim(3) == vDim;
    } else if (std::strcmp(layout, "TND") == 0) {
        validO = batch == 1 && o.GetDimNum() == 3 && o.GetDim(0) == tokens &&
                 o.GetDim(1) == hv && o.GetDim(2) == vDim;
    } else if (std::strcmp(layout, "NTD") == 0) {
        validO = batch == 1 && o.GetDimNum() == 3 && o.GetDim(0) == hv &&
                 o.GetDim(1) == tokens && o.GetDim(2) == vDim;
    }
    CHECK_COND(validO, ACLNN_ERR_PARAM_INVALID, "oOut shape does not match outputLayout.");
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtypes(const ChunkFwdHOFusedParams &params)
{
    const DataType inputType = params.k->GetDataType();
    CHECK_COND(inputType == DataType::DT_FLOAT16 || inputType == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "input dtype must be float16 or bfloat16.");
    CHECK_COND(params.q->GetDataType() == inputType && params.w->GetDataType() == inputType &&
                   params.u->GetDataType() == inputType && params.hOut->GetDataType() == inputType &&
                   params.vNewOut->GetDataType() == inputType && params.oOut->GetDataType() == inputType,
               ACLNN_ERR_PARAM_INVALID, "q/k/w/u/h/vNew/o must have the same dtype.");
    const DataType gateType = params.g->GetDataType();
    CHECK_COND(gateType == DataType::DT_FLOAT || gateType == inputType,
               ACLNN_ERR_PARAM_INVALID, "g dtype must be float32 or match input dtype.");
    CHECK_COND(params.gkOptional == nullptr || params.gkOptional->GetDataType() == gateType,
               ACLNN_ERR_PARAM_INVALID, "gk dtype must match g dtype.");
    if (params.outputFinalState) {
        const DataType stateType = params.initialStateOptional == nullptr
                                       ? DataType::DT_FLOAT
                                       : params.initialStateOptional->GetDataType();
        CHECK_COND(params.finalStateOut->GetDataType() == stateType,
                   ACLNN_ERR_PARAM_INVALID, "finalStateOut dtype does not match state dtype.");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus MakeInputsContiguous(ChunkFwdHOFusedParams &params, aclOpExecutor *executor)
{
    const aclTensor **required[] = {&params.k, &params.w, &params.u, &params.g, &params.q};
    for (const aclTensor **tensor : required) {
        CHECK_RET(MakeContiguous(*tensor, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }
    if (params.gkOptional != nullptr) {
        CHECK_RET(MakeContiguous(params.gkOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }
    if (params.initialStateOptional != nullptr) {
        CHECK_RET(MakeContiguous(params.initialStateOptional, executor) == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnChunkFwdHOFusedGetWorkspaceSize(
    const aclTensor *k, const aclTensor *w, const aclTensor *u, const aclTensor *g,
    const aclTensor *gkOptional, const aclTensor *initialStateOptional, const aclTensor *q,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    bool outputFinalState, int64_t chunkSize, double scale, bool useExp2, bool stateVFirst,
    const char *outputLayout, const aclTensor *hOut, const aclTensor *vNewOut,
    const aclTensor *finalStateOut, const aclTensor *oOut, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    ChunkFwdHOFusedParams params{k, w, u, g, gkOptional, initialStateOptional, q,
                                  cuSeqlensOptional, chunkIndicesOptional, outputFinalState,
                                  chunkSize, scale, useExp2, stateVFirst, outputLayout,
                                  hOut, vNewOut, finalStateOut, oOut};
    L2_DFX_PHASE_1(aclnnChunkFwdHOFused,
                   DFX_IN(k, w, u, g, gkOptional, initialStateOptional, q, cuSeqlensOptional,
                          chunkIndicesOptional, outputFinalState, chunkSize, scale, useExp2,
                          stateVFirst, outputLayout),
                   DFX_OUT(hOut, vNewOut, finalStateOut, oOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    aclOpExecutor *executorPtr = uniqueExecutor.get();
    CHECK_RET(CheckRequired(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapes(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypes(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeInputsContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    const auto &kShape = params.k->GetViewShape();
    const auto &uShape = params.u->GetViewShape();
    const int64_t sequences = params.cuSeqlensOptional == nullptr
                                  ? kShape.GetDim(0)
                                  : static_cast<int64_t>(params.cuSeqlensOptional->Size()) - 1;
    const aclTensor *initialStateCompute = params.initialStateOptional;
    const aclTensor *hCompute = params.hOut;
    const aclTensor *finalStateCompute = params.finalStateOut;
    if (params.stateVFirst && initialStateCompute != nullptr) {
        initialStateCompute = TransposeLastTwo(initialStateCompute, executorPtr);
        CHECK_RET(initialStateCompute != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (params.stateVFirst) {
        hCompute = executorPtr->AllocTensor(SwapLastTwo(params.hOut->GetViewShape()),
                                            params.hOut->GetDataType(), Format::FORMAT_ND);
        CHECK_RET(hCompute != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (!params.outputFinalState) {
        const DataType stateType = initialStateCompute == nullptr
                                       ? DataType::DT_FLOAT
                                       : initialStateCompute->GetDataType();
        finalStateCompute = executorPtr->AllocTensor(
            MakeShape({sequences, uShape.GetDim(1), kShape.GetDim(3), uShape.GetDim(3)}),
            stateType, Format::FORMAT_ND);
    } else if (params.stateVFirst) {
        finalStateCompute = executorPtr->AllocTensor(SwapLastTwo(params.finalStateOut->GetViewShape()),
                                                     params.finalStateOut->GetDataType(), Format::FORMAT_ND);
    }
    CHECK_RET(finalStateCompute != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::ChunkFwdHOFused(
        params.k, params.w, params.u, params.g, params.gkOptional, initialStateCompute, params.q,
        params.cuSeqlensOptional, params.chunkIndicesOptional, params.outputFinalState,
        params.chunkSize, params.scale, params.useExp2, params.outputLayout,
        hCompute, params.vNewOut, finalStateCompute, params.oOut, executorPtr);
    CHECK_RET(result[0] != nullptr && result[1] != nullptr && result[3] != nullptr,
              ACLNN_ERR_PARAM_NULLPTR);

    const aclTensor *hResult = result[0];
    if (params.stateVFirst) {
        hResult = TransposeLastTwo(hResult, executorPtr);
        CHECK_RET(hResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(l0op::ViewCopy(hResult, params.hOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[1], params.vNewOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (params.outputFinalState) {
        const aclTensor *finalResult = result[2];
        CHECK_RET(finalResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (params.stateVFirst) {
            finalResult = TransposeLastTwo(finalResult, executorPtr);
            CHECK_RET(finalResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        CHECK_RET(l0op::ViewCopy(finalResult, params.finalStateOut, executorPtr) != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(l0op::ViewCopy(result[3], params.oOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkFwdHOFused(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkFwdHOFused);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ChunkFwdHOFused launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
