/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 *
 * Schedule entry for the chunk64 sparse roles.
 *
 * Pipeline schedule (kSparseSchedulePipeline, B*Hv <= AIC/2): chain c owns
 * producer group c and consumer group chainCount + c. Iteration i lets the
 * producer build chunk i while the consumer finishes chunk i-1; one
 * AivRoleBarrier() per iteration publishes the produced chunk and paces the
 * pair. Only the AIVs take part in that barrier: they own all GM traffic,
 * and each AIV relays the boundary to its own AIC inside the MIX core.
 * Groups outside [0, 2*chainCount) take part in the barrier only, so the
 * mode-0 participant set is identical on every iteration.
 *
 * Group-serial schedule (kSparseScheduleSerial, any chain count): each group
 * owns every launchGroups-th chain outright. Per chain the producer role
 * builds every chunk, one mode-1 rendezvous publishes them inside the MIX
 * core, then the consumer role drains the chain. Chains are independent, so
 * there is no mode-0 barrier at all.
 */
#ifndef GDN_CHUNK64_SPARSE_PIPELINE_H
#define GDN_CHUNK64_SPARSE_PIPELINE_H

#include "sparse_barrier.h"
#include "sparse_common.h"
#include "sparse_fused_tiling.h"
#include "sparse_memory.h"
#include "sparse_producer.h"
#include "sparse_consumer.h"

namespace GdnChunk64Sparse {

// Group-local mode-2 iteration boundary: both AIVs of a MIX group release
// their own AIC after the global barrier. Distinct from every mode-4 stage id.
constexpr uint16_t kSparseIterationFlag = 8;
// Intra-core AIV rendezvous publishing one chain's produced tensors from the
// producer half to the consumer half in group-serial mode. Mode 1 ids are
// independent of the mode-2/mode-4 ids used inside the roles.
constexpr uint16_t kSparseChainHandoffFlag = 15;

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

// ---------------------------------------------------------------------------
// Producer/consumer pipeline (schedule = kSparseSchedulePipeline)
// ---------------------------------------------------------------------------
__aicore__ inline void RunProducerPipeline(const PhaseArgs &args, const FusedTilingData &td,
                                           uint32_t groupIdx)
{
    ProducerRole role;
    role.Init(args, td);
    role.BeginChain(args, groupIdx);
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
    role.Init(args, td);
    role.BeginChain(args, chainIdx);
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
    role.EndChain();
    role.Finish();
}

// Idle and future groups: mirror the working groups' full iteration shape
// (barrier + group-local relay + AIC wait) instead of the barrier alone. The
// relay is a no-op cost when mode 2 is group-local, and it keeps the launch
// deadlock-free if this silicon AND-reduces mode 2 across the whole launch.
__aicore__ inline void RunIdlePipeline(const FusedTilingData &td)
{
    const int64_t iterations = static_cast<int64_t>(td.pipeline.maxChunks) + 1;
    for (int64_t i = 0; i < iterations; ++i) {
        AivRoleBarrier(td.pipeline.barrierFlagId);
        AivRelayToAic(kSparseIterationFlag);
        AicWaitRelayMte1(kSparseIterationFlag);
    }
}

// ---------------------------------------------------------------------------
// Group-serial schedule (schedule = kSparseScheduleSerial)
// ---------------------------------------------------------------------------
// Producer published this chain's W/U/g_cumsum through AIV0's MTE3. Both AIVs
// rendezvous before the consumer half reads them: the Set reports each AIV's
// pipe state, and mode 1 releases only after both arrive.
__aicore__ inline void AivChainHandoff()
{
    if ASCEND_IS_AIV {
        AscendC::PipeBarrier<PIPE_ALL>();
        GDN_SPARSE_XCORE_SET(0x1, PIPE_MTE3, kSparseChainHandoffFlag);
        GDN_SPARSE_XCORE_WAIT(0x1, PIPE_MTE2, kSparseChainHandoffFlag);
    }
}

__aicore__ inline void RunGroupSerial(const PhaseArgs &args, const FusedTilingData &td,
                                      uint32_t groupIdx)
{
    const uint32_t totalChains = td.pipeline.chainCount;
    const uint32_t groups = td.launchGroups == 0 ? 1 : td.launchGroups;
    ProducerRole producer;
    producer.Init(args, td);
    ConsumerRole consumer;
    consumer.Init(args, td);
    for (uint32_t chain = groupIdx; chain < totalChains; chain += groups) {
        producer.BeginChain(args, chain);
        const int64_t chunks = producer.ChunkCount();
        if (chunks <= 0) {
            continue;
        }
        for (int64_t i = 0; i < chunks; ++i) {
            producer.RunChunk(i);
        }
        AivChainHandoff();
        consumer.BeginChain(args, chain);
        for (int64_t i = 0; i < chunks; ++i) {
            consumer.RunChunk(i);
        }
        consumer.EndChain();
    }
    producer.Finish();
    consumer.Finish();
}

// Bring-up bisect switch, driven at runtime by the host-filled
// td.pipeline.debugMode (GDN_SPARSE_DEBUG_MODE env). 0 (default): full
// pipeline; 1: barrier/relay only; 2: producer only; 3: consumer only.
// Mode 2/3 leave the partner role's outputs unwritten, so they isolate the
// crashing side rather than produce correct numbers.
__aicore__ inline void RunSparsePipeline(const PhaseArgs &args, const FusedTilingData &td)
{
    const uint32_t chains = td.pipeline.chainCount;
    const uint32_t groupIdx = SparseGroupIndex();
    const uint32_t debugMode = td.pipeline.debugMode;
    if (td.pipeline.schedule == kSparseSchedulePipeline && chains != 0 &&
        td.pipeline.maxChunks != 0 && td.launchGroups != 0 && 2U * chains <= td.launchGroups) {
        if (groupIdx < chains) {
            if (debugMode == 1 || debugMode == 3) {
                RunIdlePipeline(td);
            } else {
                RunProducerPipeline(args, td, groupIdx);
            }
        } else if (groupIdx < 2U * chains) {
            if (debugMode == 1 || debugMode == 2) {
                RunIdlePipeline(td);
            } else {
                RunConsumerPipeline(args, td, groupIdx - chains);
            }
        } else {
            RunIdlePipeline(td);
        }
        return;
    }
    RunGroupSerial(args, td, groupIdx);
}

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_PIPELINE_H
