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
        for (int64_t iteration = 0; iteration < td.maxChunks; ++iteration) {
            if (iteration < chunks) {
                role.RunChunk(iteration);
            } else {
                role.JoinNormalizedQKGeneration(iteration);
            }
        }
    }
    if ASCEND_IS_AIC {
        const int64_t globalPairs = (td.maxChunks + 1) / 2;
        // Prime T/APrime generations 0/1; every join below returns the
        // pair+2 credit instead of publishing its own generation.
        PrimeTAPrimeCreditsAic(globalPairs);
        for (int64_t iteration = 0; iteration < td.maxChunks; ++iteration) {
            // T and APrime mode-0 generations are pair-granular: join once per
            // pair, before the even chunk's GM->L1 T copy in RunCubePrepare.
            if ((iteration & 1) == 0) {
                JoinIterationAicCredit<PIPE_MTE2>(iteration >> 1, globalPairs);
            }
            if (iteration < chunks) {
                role.RunPrepare(iteration);
            }
            if (iteration < chunks) {
                role.RunAfterWu(iteration);
            } else if ((iteration & 1) == 0) {
                // Shorter and empty chains still join every global mode-0
                // generation; active chains join in RunCubePrepare behind
                // the T loads. No MTE2 load follows an idle iteration, so
                // the wait and its credit set both ride MTE3.
                JoinAPrimeAicCredit<PIPE_MTE3>(iteration >> 1, globalPairs);
            }
        }
        // Drain the local MTE1->MTE2 guards of the final l1T_ reads. An even
        // chunk count leaves both parity flags pending, an odd one only the
        // even slot's.
        if (chunks >= 2) {
            WaitFlag<HardEvent::MTE1_MTE2>(0);
            if ((chunks & 1) == 0) {
                WaitFlag<HardEvent::MTE1_MTE2>(1);
            }
        } else if (chunks == 1) {
            WaitFlag<HardEvent::MTE1_MTE2>(0);
        }
    }
    role.EndChain();
}

__aicore__ inline void RunSparsePipeline(const PhaseArgs &args, const FusedTilingData &td)
{
    if ASCEND_IS_AIC {
        SetHF32Mode(false);
        SetFlag<HardEvent::M_MTE1>(0);
        for (int32_t region = 0; region < kL0CReleaseEventCount; ++region) {
            SetFlag<HardEvent::FIX_M>(kL0CReleaseEventIds[region]);
        }
    }
    const uint32_t group = GroupIndex();
    // Pair producer and consumer of one chain on adjacent cores (2g, 2g+1)
    // instead of splitting all producers before all consumers: the pair's
    // GM exchange (T, APrime, normalized Q/K) then stays within the same
    // L2 die/bank neighborhood instead of crossing the device.
    const uint32_t chain = group >> 1;
    if ((group & 1U) == 0U) {
        RunProducer(args, td, chain);
    } else {
        RunConsumer(args, td, chain);
    }
    if ASCEND_IS_AIV {
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
    if ASCEND_IS_AIC {
        WaitFlag<HardEvent::M_MTE1>(0);
        for (int32_t region = 0; region < kL0CReleaseEventCount; ++region) {
            WaitFlag<HardEvent::FIX_M>(kL0CReleaseEventIds[region]);
        }
    }
}

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_PIPELINE_H
