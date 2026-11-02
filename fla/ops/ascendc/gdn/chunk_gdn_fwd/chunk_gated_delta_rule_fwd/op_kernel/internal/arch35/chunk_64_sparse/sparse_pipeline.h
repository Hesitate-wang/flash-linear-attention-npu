/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 *
 * Producer/consumer entry for the chunk64 low-parallelism schedule.
 *
 * Chain c owns producer group c and consumer group chainCount + c. Iteration i
 * lets the producer build chunk i while the consumer finishes chunk i-1; one
 * AivRoleBarrier() per iteration publishes the produced chunk and paces the
 * pair. Only the AIVs take part in that barrier: they own all GM traffic, and
 * each AIV relays the boundary to its own AIC inside the MIX core.
 *
 * Groups outside [0, 2*chainCount) take part in the barrier only, so the
 * mode-0 participant set is identical on every iteration and no group can
 * starve the collective.
 */
#ifndef GDN_CHUNK64_SPARSE_PIPELINE_H
#define GDN_CHUNK64_SPARSE_PIPELINE_H

#include "sparse_barrier.h"
#include "sparse_common.h"
#include "sparse_fused_tiling.h"
#include "sparse_memory.h"
#include "sparse_phase_adapter.h"
#include "sparse_producer.h"
#include "sparse_consumer.h"

namespace GdnChunk64Sparse {

// Group-local mode-2 iteration boundary: both AIVs of a MIX group release
// their own AIC after the global barrier. Distinct from every mode-4 stage id.
constexpr uint16_t kSparseIterationFlag = 8;

__aicore__ inline uint32_t SparseGroupIndex()
{
    uint32_t group = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        const uint32_t subNum = AscendC::GetSubBlockNum();
        if (subNum > 0) {
            group /= subNum;
        }
    }
    return group;
}

__aicore__ inline void RunProducerPipeline(const PhaseArgs &args, const FusedTilingData &td,
                                           uint32_t groupIdx)
{
    ProducerRole role;
    role.Init(args, td, groupIdx);
    const int64_t chunks = role.ChunkCount();
    const int64_t iterations = static_cast<int64_t>(td.pipeline.maxChunks) + 1;
    for (int64_t i = 0; i < iterations; ++i) {
        if (i < chunks) {
            role.RunChunk(i);
        }
        AivRoleBarrier(td.pipeline.barrierFlagId);
        AivRelayToAic(kSparseIterationFlag);
        AicWaitRelayMte1(kSparseIterationFlag);
    }
    role.Finish();
}

__aicore__ inline void RunConsumerPipeline(const PhaseArgs &args, const FusedTilingData &td,
                                           uint32_t chainIdx)
{
    ConsumerRole role;
    role.Init(args, td, chainIdx);
    const int64_t chunks = role.ChunkCount();
    const int64_t iterations = static_cast<int64_t>(td.pipeline.maxChunks) + 1;
    for (int64_t i = 0; i < iterations; ++i) {
        // The consumer trails the producer by one chunk: iteration i finishes
        // chunk i-1, whose W/U/q_hat/k_hat/g were published by iteration i-1.
        if (i >= 1 && i - 1 < chunks) {
            role.RunChunk(i - 1);
        }
        AivRoleBarrier(td.pipeline.barrierFlagId);
        AivRelayToAic(kSparseIterationFlag);
        AicWaitRelayMte1(kSparseIterationFlag);
    }
    role.Finish();
}

// Idle and future groups: AIVs only, to keep the mode-0 set complete.
__aicore__ inline void RunIdlePipeline(const FusedTilingData &td)
{
    const int64_t iterations = static_cast<int64_t>(td.pipeline.maxChunks) + 1;
    for (int64_t i = 0; i < iterations; ++i) {
        AivRoleBarrier(td.pipeline.barrierFlagId);
    }
}

__aicore__ inline void RunSparsePipeline(const PhaseArgs &args, const FusedTilingData &td)
{
    const uint32_t chains = td.pipeline.chainCount;
    if (chains == 0 || td.pipeline.maxChunks == 0 || td.launchGroups == 0 ||
        2U * chains > td.launchGroups || td.pipeline.schedule != kSparseSchedulePipeline) {
        RunSerialPhases(args, td);
        return;
    }
    const uint32_t groupIdx = SparseGroupIndex();
    if (groupIdx < chains) {
        RunProducerPipeline(args, td, groupIdx);
    } else if (groupIdx < 2U * chains) {
        RunConsumerPipeline(args, td, groupIdx - chains);
    } else {
        RunIdlePipeline(td);
    }
}

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_PIPELINE_H
