/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_CHUNK64_SPARSE_PIPELINE_H
#define GDN_CHUNK64_SPARSE_PIPELINE_H

#include "sparse_common.h"
#include "sparse_consumer.h"
#include "sparse_producer.h"
#include "sparse_sync.h"

namespace GdnChunk64Sparse {

__aicore__ inline void RunProducer(const PhaseArgs &args, const FusedTilingData &td,
                                   uint32_t chain)
{
    ProducerRole role;
    role.Init(args, td);
    role.BeginChain(args, chain);
    role.Run();
}

__aicore__ inline void RunConsumer(const PhaseArgs &args, const FusedTilingData &td,
                                   uint32_t chain)
{
    ConsumerRole role;
    role.Init(args, td);
    role.BeginChain(args, chain);
    role.RunPrologue();
    const int64_t chunks = role.ChunkCount();
    if ASCEND_IS_AIV {
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            role.RunChunk(chunk);
        }
    }
    if ASCEND_IS_AIC {
        for (int64_t iteration = 0; iteration < td.maxChunks; ++iteration) {
            if (iteration < chunks) {
                role.RunPrepare(iteration);
            }
            // T is complete after the producer's mode-0. The next operation is
            // the Consumer's GM->L1 T copy, so block MTE2 rather than only FIX.
            PublishIterationAic<PIPE_MTE2>(iteration);
            if (iteration < chunks) {
                role.RunAfterWu(iteration);
            } else {
                // Shorter and empty chains still join every global mode-0
                // generation; active chains publish at O inside LoadAPrime.
                // No MTE2 load follows an idle iteration, so block MTE3 to
                // prevent a later Set from reusing this flag before the wait.
                PublishAPrimeAic<PIPE_MTE3>(iteration);
            }
        }
    }
    role.EndChain();
}

__aicore__ inline void RunSparsePipeline(const PhaseArgs &args, const FusedTilingData &td)
{
    if ASCEND_IS_AIC {
        SetHF32Mode(false);
        SetFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::FIX_M>(0);
    }
    const uint32_t group = GroupIndex();
    if (group < td.chainCount) {
        RunProducer(args, td, group);
    } else {
        RunConsumer(args, td, group - td.chainCount);
    }
    if ASCEND_IS_AIV {
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
    if ASCEND_IS_AIC {
        WaitFlag<HardEvent::M_MTE1>(0);
        WaitFlag<HardEvent::FIX_M>(0);
    }
}

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_PIPELINE_H
