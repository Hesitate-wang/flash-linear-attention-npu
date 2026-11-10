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
            // Producer arrives after T@Kb/T@Vb and their GM Fixpipe. Once all
            // AICs arrive, release this group's AIVs to load W/U from GM.
            PublishIterationAic<PIPE_FIX>();
            if (iteration < chunks) {
                role.RunAfterWu(iteration);
            }
        }
    }
    role.EndChain();
}

__aicore__ inline void RunSerial(const PhaseArgs &args, const FusedTilingData &td,
                                 uint32_t group)
{
    ProducerRole producer;
    ConsumerRole consumer;
    producer.Init(args, td);
    consumer.Init(args, td);
    for (uint32_t chain = group; chain < td.chainCount; chain += td.launchGroups) {
        producer.BeginChain(args, chain);
        const int64_t chunks = producer.ChunkCount();
        producer.Run();
        if (chunks == 0) {
            consumer.BeginChain(args, chain);
            consumer.EndChain();
            continue;
        }
        PublishChainInGroup();
        consumer.BeginChain(args, chain);
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            consumer.RunChunk(chunk);
        }
        consumer.EndChain();
    }
}

__aicore__ inline void RunSparsePipeline(const PhaseArgs &args, const FusedTilingData &td)
{
    if ASCEND_IS_AIC {
        SetHF32Mode(false);
        SetFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::FIX_M>(0);
    }
    const uint32_t group = GroupIndex();
    if (td.schedule == kSparseSchedulePipeline && td.chainCount != 0 &&
        td.maxChunks != 0 && td.launchGroups == 2U * td.chainCount) {
        if (group < td.chainCount) {
            RunProducer(args, td, group);
        } else if (group < td.launchGroups) {
            RunConsumer(args, td, group - td.chainCount);
        }
    } else {
        RunSerial(args, td, group);
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
