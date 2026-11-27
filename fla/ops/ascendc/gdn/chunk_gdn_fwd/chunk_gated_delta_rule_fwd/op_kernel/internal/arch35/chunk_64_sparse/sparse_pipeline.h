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
    GDN_SPARSE_TRACE(td, 10, chain, 0);
    role.BeginChain(args, chain);
    GDN_SPARSE_TRACE(td, 11, chain, static_cast<int64_t>(role.ChunkCount()));
    role.RunPrologue();
    const int64_t chunks = role.ChunkCount();
    GDN_SPARSE_TRACE(td, 12, chain, chunks);
    if ASCEND_IS_AIV {
        // An empty chain owns no chunk and stays out of the generation
        // exchange, so its join-only loop is skipped as well.
        if (chunks > 0) {
            for (int64_t iteration = 0; iteration < td.maxChunks; ++iteration) {
                if (iteration < chunks) {
                    role.RunChunk(iteration);
                    GDN_SPARSE_TRACE(td, 42, chain, iteration);
                } else {
                    role.JoinNormalizedQKGeneration(iteration);
                }
            }
        }
        GDN_SPARSE_TRACE(td, 13, chain, 0);
    }
    if ASCEND_IS_AIC {
        // Same contract on the cube side: an empty chain primes no generation,
        // joins none and drains none.
        if (chunks > 0) {
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
        }
        GDN_SPARSE_TRACE(td, 14, chain, 0);
    }
    GDN_SPARSE_TRACE(td, 15, chain, 0);
    role.EndChain();
    GDN_SPARSE_TRACE(td, 16, chain, 0);
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
        if (chain >= td.chainCount) {
            // Both roles of the pair take the same branch, so a trailing short
            // pass leaves the idle pairs fully silent.
            continue;
        }
        if ASCEND_IS_AIC {
            SetHF32Mode(false);
            SetFlag<HardEvent::M_MTE1>(0);
            for (int32_t region = 0; region < kL0CReleaseEventCount; ++region) {
                SetFlag<HardEvent::FIX_M>(kL0CReleaseEventIds[region]);
            }
        }
        GDN_SPARSE_TRACE(td, 20, chain, 0);
        if ((group & 1U) == 0U) {
            RunProducer(args, td, chain);
        } else {
            RunConsumer(args, td, chain);
        }
        GDN_SPARSE_TRACE(td, 21, chain, 0);
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
