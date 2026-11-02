/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_CHUNK64_SPARSE_PHASE_ADAPTER_H
#define GDN_CHUNK64_SPARSE_PHASE_ADAPTER_H

#include "../../../../../chunk_gated_delta_rule_fwd_prepare/op_kernel/arch35/chunk_gated_delta_rule_fwd_prepare.h"
#include "../../../../../chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h"
#include "../../../../../chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_vec.h"
#include "../../../../../chunk_fwd_o/op_kernel/arch35/chunk_fwd_o_a5.h"
#include "sparse_fused_tiling.h"
#include "sparse_sync.h"

namespace GdnChunk64Sparse {

struct PhaseArgs {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR rawG;
    GM_ADDR beta;
    GM_ADDR aLog;
    GM_ADDR dtBias;
    GM_ADDR cuSeqlens;
    GM_ADDR chunkIndices;
    GM_ADDR gCumsum;
    GM_ADDR w;
    GM_ADDR u;
    GM_ADDR a;
    GM_ADDR qHat;
    GM_ADDR kHat;
    GM_ADDR qRstd;
    GM_ADDR kRstd;
    GM_ADDR betaEff;
    GM_ADDR h;
    GM_ADDR vNew;
    GM_ADDR o;
    GM_ADDR initialState;
    GM_ADDR finalState;
    GM_ADDR workspace;
};

__aicore__ inline GDN::FwdHRuntimeTiling LoadHTiling(const ChunkFwdHPlainTilingData &td)
{
    GDN::FwdHRuntimeTiling result{};
    result.batch = td.batch;
    result.seqlen = td.seqlen;
    result.kNumHead = td.kNumHead;
    result.vNumHead = td.vNumHead;
    result.kHeadDim = td.kHeadDim;
    result.vHeadDim = td.vHeadDim;
    result.chunkSize = td.chunkSize;
    result.shapeBatch = td.shapeBatch;
    result.tokenBatch = td.tokenBatch;
    result.vWorkspaceOffset = td.vWorkspaceOffset;
    result.vUpdateWorkspaceOffset = td.vUpdateWorkspaceOffset;
    result.kDecayWorkspaceOffset = td.kDecayWorkspaceOffset;
    result.hWorkspaceOffset = td.hWorkspaceOffset;
    result.useInitialState = td.useInitialState;
    result.storeFinalState = td.storeFinalState;
    result.isVariedLen = td.isVariedLen;
    return result;
}

template <bool UseExp2, bool StateFp32, bool StateVFirst>
__aicore__ inline void RunH(const GDN::FwdHKernelArgs &args)
{
    using Policy = GDN::FwdHCompilePolicy<GDN::FwdHGateMode::SCALAR_G, UseExp2, StateFp32>;
    if ASCEND_IS_AIC {
        const bool hasCubeWork = args.tiling.useInitialState || args.tiling.storeFinalState ||
            args.tiling.seqlen > static_cast<uint32_t>(GDN::FWD_H_CHUNK);
        if (hasCubeWork) {
            GDN::ChunkFwdHCubeArch35<Policy, StateVFirst> cube;
            cube.Init(args);
            cube.Process();
        }
    }
    if ASCEND_IS_AIV {
        GDN::ChunkFwdHVecArch35<float, Policy, StateVFirst> vec;
        vec.Init(args);
        vec.Process();
    }
}

template <bool UseExp2, bool StateVFirst>
__aicore__ inline void RunHO(const PhaseArgs &args, const FusedTilingData &td)
{
    // GM_ADDR is a macro (`__gm__ uint8_t*`), so `const GM_ADDR x` would mean
    // pointer-to-const and would not convert to a plain GM_ADDR parameter.
    GM_ADDR user = AscendC::GetUserWorkspace(args.workspace);
    const bool normalize = td.prepare.enableQueryKeyL2NormInKernel != 0;
    GM_ADDR qCompute = normalize ? args.qHat : args.q;
    GM_ADDR kCompute = normalize ? args.kHat : args.k;
    GDN::FwdHKernelArgs hArgs{};
    hArgs.k = kCompute;
    hArgs.w = args.w;
    hArgs.u = args.u;
    hArgs.g = args.gCumsum;
    hArgs.initialState = args.initialState;
    hArgs.cuSeqlens = args.cuSeqlens;
    hArgs.chunkIndices = args.chunkIndices;
    hArgs.h = args.h;
    hArgs.vNew = args.vNew;
    hArgs.finalState = args.finalState;
    hArgs.workspace = user + td.hWorkspaceOffset;
    hArgs.tiling = LoadHTiling(td.h);
    if (td.stateDtype == 0) {
        RunH<UseExp2, true, StateVFirst>(hArgs);
    } else {
        RunH<UseExp2, false, StateVFirst>(hArgs);
    }
    PhaseRendezvous();
    GDN::ChunkFwdOA5Dispatch<float, UseExp2>(
        qCompute, kCompute, args.vNew, args.h, args.gCumsum,
        args.cuSeqlens, args.chunkIndices, args.o,
        user + td.oWorkspaceOffset, &td.o);
}

__aicore__ inline void RunSerialPhases(const PhaseArgs &args, const FusedTilingData &td)
{
    {
        ChunkGatedDeltaRuleFwdPrepare::ChunkGatedDeltaRuleFwdPrepareKernel<
            float, float, float> prepare;
        prepare.Init(args.q, args.k, args.v, args.rawG, args.beta, args.aLog, args.dtBias,
                     args.cuSeqlens, args.chunkIndices, args.gCumsum, args.w, args.u, args.a,
                     args.qHat, args.kHat, args.qRstd, args.kRstd, args.betaEff,
                     args.workspace, td.prepare);
        prepare.Process();
        prepare.FinishForFusion();
    }
    PhaseRendezvous();
    if (td.useExp2 != 0) {
        if (td.stateVFirst != 0) {
            RunHO<true, true>(args, td);
        } else {
            RunHO<true, false>(args, td);
        }
    } else if (td.stateVFirst != 0) {
        RunHO<false, true>(args, td);
    } else {
        RunHO<false, false>(args, td);
    }
}

} // namespace GdnChunk64Sparse
#endif // GDN_CHUNK64_SPARSE_PHASE_ADAPTER_H
