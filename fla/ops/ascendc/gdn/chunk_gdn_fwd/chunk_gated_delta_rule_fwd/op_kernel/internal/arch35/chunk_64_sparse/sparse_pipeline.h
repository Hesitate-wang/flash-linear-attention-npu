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
    GDN_SPARSE_TRACE(td, 1, chain, 0);
    ProducerRole role;
    role.Init(args, td);
    role.BeginChain(args, chain);
    role.Run();
    GDN_SPARSE_TRACE(td, 2, chain, 0);
}

__aicore__ inline void RunConsumer(const PhaseArgs &args, const FusedTilingData &td,
                                   uint32_t chain)
{
    ConsumerRole role;
    role.Init(args, td);
    role.BeginChain(args, chain);
    GDN_SPARSE_TRACE(td, 11, chain, static_cast<int64_t>(role.ChunkCount()));
    role.RunPrologue();
    const int64_t chunks = role.ChunkCount();
    if ASCEND_IS_AIV {
        // An empty chain joins the generation exchange without running a chunk:
        // mode-0 flags are an all-AIV barrier, so it must post and wait exactly
        // like its siblings even though it moves no data.
        for (int64_t iteration = 0; iteration < td.maxChunks; ++iteration) {
            if (iteration < chunks) {
                role.RunChunk(iteration);
                GDN_SPARSE_TRACE(td, 42, chain, iteration);
            } else {
                role.JoinNormalizedQKGeneration(iteration);
            }
        }
        GDN_SPARSE_TRACE(td, 13, chain, 0);
    }
    if ASCEND_IS_AIC {
        // Same contract on the cube side: an empty chain primes, joins and
        // drains every generation without running a chunk.
        const int64_t globalPairs = (td.maxChunks + 1) / 2;
        // Prime T/APrime generations 0/1; every join below returns the
        // pair+2 credit instead of publishing its own generation.
        PrimeTAPrimeCreditsAic(globalPairs);
        for (int64_t iteration = 0; iteration < td.maxChunks; ++iteration) {
            // T and APrime mode-0 generations are pair-granular: join once
            // per pair, before the even chunk's GM->L1 T copy in
            // RunCubePrepare.
            if ((iteration & 1) == 0) {
                JoinIterationAicCredit<PIPE_MTE2>(iteration >> 1, globalPairs);
            }
            if (iteration < chunks) {
                role.RunPrepare(iteration);
                GDN_SPARSE_TRACE(td, 40, chain, iteration);
            }
            if (iteration < chunks) {
                role.RunAfterWu(iteration);
                GDN_SPARSE_TRACE(td, 41, chain, iteration);
            } else if ((iteration & 1) == 0) {
                // Shorter chains still join every global mode-0 generation;
                // active chains join in RunCubePrepare behind the T loads.
                // No MTE2 load follows an idle iteration, so the wait and
                // its credit set ride the Fixpipe queue: MTE3 does not
                // exist on the Cube, and a wait issued there never retires.
                JoinAPrimeAicCredit<PIPE_FIX>(iteration >> 1, globalPairs);
            }
        }
        // Drain the local MTE1->MTE2 guards of the final l1T_/l1APrime_
        // reads. An even chunk count leaves both parity flags pending. An
        // odd one leaves the even slot's pair plus the last odd chunk's
        // APrime release: the following even chunk that would consume it
        // does not exist, and the T-slot flag of that odd chunk was
        // already consumed at the final even chunk's head.
        if (chunks >= 2) {
            WaitFlag<HardEvent::MTE1_MTE2>(0);
            WaitFlag<HardEvent::MTE1_MTE2>(kConsAPrimeReleaseEventIds[0]);
            if ((chunks & 1) == 0) {
                WaitFlag<HardEvent::MTE1_MTE2>(1);
                WaitFlag<HardEvent::MTE1_MTE2>(kConsAPrimeReleaseEventIds[1]);
            } else {
                // Without this drain the flag survives chain exit and the
                // next chain's first MTE1_MTE2(3) wait would see a stale set.
                WaitFlag<HardEvent::MTE1_MTE2>(kConsAPrimeReleaseEventIds[1]);
            }
        } else if (chunks == 1) {
            WaitFlag<HardEvent::MTE1_MTE2>(0);
            WaitFlag<HardEvent::MTE1_MTE2>(kConsAPrimeReleaseEventIds[0]);
        }
        GDN_SPARSE_TRACE(td, 14, chain, 0);
    }
    role.EndChain();
    GDN_SPARSE_TRACE(td, 16, chain, 0);
}

// A pair left without a chain in a trailing pass still posts and joins every
// global mode-0 generation. Those flags are barriers over every AIC / every
// AIV, so a silent pair would keep the active pairs' waits from ever retiring.
// Only the data movement is skipped: the set/wait sequence below matches, step
// for step, what a chain whose chunk count is zero executes.
__aicore__ inline void RunIdlePair(const FusedTilingData &td, uint32_t group)
{
    const int64_t globalPairs = (td.maxChunks + 1) / 2;
    const bool isProducer = (group & 1U) == 0U;
    if ASCEND_IS_AIV {
        if (isProducer) {
            for (int64_t pair = 0; pair < globalPairs; ++pair) {
                if (pair >= 2) {
                    WaitMode0Flag<PIPE_MTE3>(IterationNormalizedKFlag(pair - 2));
                    WaitMode0Flag<PIPE_MTE3>(IterationNormalizedQFlag(pair - 2));
                }
                SetMode0Flag<PIPE_MTE3>(IterationNormalizedKFlag(pair));
                SetMode0Flag<PIPE_MTE3>(IterationNormalizedQFlag(pair));
            }
            if (globalPairs >= 2) {
                WaitMode0Flag<PIPE_MTE3>(IterationNormalizedKFlag(globalPairs - 2));
                WaitMode0Flag<PIPE_MTE3>(IterationNormalizedQFlag(globalPairs - 2));
            }
            if (globalPairs >= 1) {
                WaitMode0Flag<PIPE_MTE3>(IterationNormalizedKFlag(globalPairs - 1));
                WaitMode0Flag<PIPE_MTE3>(IterationNormalizedQFlag(globalPairs - 1));
            }
        } else {
            SetMode0Flag<PIPE_MTE2>(IterationNormalizedKFlag(0));
            SetMode0Flag<PIPE_MTE2>(IterationNormalizedQFlag(0));
            if (globalPairs >= 2) {
                SetMode0Flag<PIPE_MTE2>(IterationNormalizedKFlag(1));
                SetMode0Flag<PIPE_MTE2>(IterationNormalizedQFlag(1));
            }
            for (int64_t iteration = 0; iteration < td.maxChunks; ++iteration) {
                if ((iteration & 1) != 0) {
                    continue;
                }
                const int64_t pair = iteration >> 1;
                WaitMode0Flag<PIPE_MTE2>(IterationNormalizedKFlag(pair));
                if (pair + 2 < globalPairs) {
                    SetMode0Flag<PIPE_MTE2>(IterationNormalizedKFlag(pair + 2));
                }
                WaitMode0Flag<PIPE_MTE2>(IterationNormalizedQFlag(pair));
                if (pair + 2 < globalPairs) {
                    SetMode0Flag<PIPE_MTE2>(IterationNormalizedQFlag(pair + 2));
                }
            }
        }
    }
    if ASCEND_IS_AIC {
        const int64_t pairs = (td.maxChunks + 1) / 2;
        if (isProducer) {
            for (int64_t pair = 0; pair < pairs; ++pair) {
                if (pair >= 2) {
                    WaitIterationAic<PIPE_MTE2>(pair);
                    WaitAPrimeAic<PIPE_FIX>(pair);
                }
                SetIterationAic<PIPE_MTE2>(pair);
                SetAPrimeAic(pair);
            }
            if (pairs >= 2) {
                WaitIterationAic<PIPE_MTE2>(pairs - 2);
                WaitAPrimeAic<PIPE_FIX>(pairs - 2);
            }
            if (pairs >= 1) {
                WaitIterationAic<PIPE_MTE2>(pairs - 1);
                WaitAPrimeAic<PIPE_FIX>(pairs - 1);
            }
        } else {
            PrimeTAPrimeCreditsAic(globalPairs);
            for (int64_t iteration = 0; iteration < td.maxChunks; ++iteration) {
                if ((iteration & 1) != 0) {
                    continue;
                }
                JoinIterationAicCredit<PIPE_MTE2>(iteration >> 1, globalPairs);
                JoinAPrimeAicCredit<PIPE_FIX>(iteration >> 1, globalPairs);
            }
            if (pairs >= 2) {
                WaitIterationAic<PIPE_MTE2>(pairs - 2);
                WaitAPrimeAic<PIPE_FIX>(pairs - 2);
            }
            if (pairs >= 1) {
                WaitIterationAic<PIPE_MTE2>(pairs - 1);
                WaitAPrimeAic<PIPE_FIX>(pairs - 1);
            }
        }
    }
    GDN_SPARSE_TRACE(td, 22, 0, 0);
}

__aicore__ inline void RunSparsePipeline(const PhaseArgs &args, const FusedTilingData &td)
{
    // Chain c is served by the neighbouring pair (2c, 2c+1) so its GM exchange
    // stays inside one L2 neighbourhood. A chain set larger than the launched
    // pairs is covered by successive passes over the same pairs: pair s runs
    // chain s, s+pairs, s+2*pairs, ... Every pass starts and ends with the
    // pipeline flags balanced, so the next chain sees a clean machine.
    const uint32_t group = GroupIndex();
    const uint32_t slot = group >> 1;
    const uint32_t pairs = td.launchGroups > 1U ? (td.launchGroups >> 1) : 1U;
    for (uint32_t base = 0; base < td.chainCount; base += pairs) {
        const uint32_t chain = base + slot;
        if ASCEND_IS_AIC {
            SetHF32Mode(false);
            SetFlag<HardEvent::M_MTE1>(0);
            for (int32_t region = 0; region < kL0CReleaseEventCount; ++region) {
                SetFlag<HardEvent::FIX_M>(kL0CReleaseEventIds[region]);
            }
        }
        GDN_SPARSE_TRACE(td, 20, chain, 0);
        if (chain >= td.chainCount) {
            // A trailing pass leaves this pair without a chain. It still joins
            // every global generation; only the data movement is skipped.
            RunIdlePair(td, group);
        } else if ((group & 1U) == 0U) {
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
}

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_PIPELINE_H
