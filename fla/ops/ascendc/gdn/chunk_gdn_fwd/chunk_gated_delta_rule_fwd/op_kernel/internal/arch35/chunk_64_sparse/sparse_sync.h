/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_CHUNK64_SPARSE_SYNC_H
#define GDN_CHUNK64_SPARSE_SYNC_H

#include "kernel_operator.h"
#include "sparse_fused_tiling.h"

namespace GdnChunk64Sparse {
using namespace AscendC;

constexpr uint16_t kAiv1FlagOffset = 16;
// AIC mode-0 T/APrime channels. AIC and AIV have independent mode-0 counters,
// so equal numeric IDs on different execution units do not share a channel.
constexpr uint16_t kSparseBarrierFlagIds[2] = {7, 12};
constexpr uint16_t kSparseAPrimeBarrierFlagIds[2] = {13, 14};
// Producer-internal mode-2 pair-completion flags: each AIV sets after its
// UB->GM store of the pair's lane, so one AIC wait covers both lanes. The IDs
// avoid every mode-0 flag in use on either execution-unit domain.
constexpr uint16_t kProdTPairDoneFlag = 2;
constexpr uint16_t kProdAPairDoneFlag = 4;
// AIV mode-0 normalized K/Q readiness channels, indexed by pair parity.
// Credit protocol: the producer publishes pair p's data with a set, gated on
// the drain of pair p-2. The consumer primes pairs 0/1 at chain start, then
// each wait(pair) on its MTE2 load stream is followed by a same-pipe set
// returning the pair+2 credit — producers therefore track the consumer's
// load progress (up to 4 pairs ahead) instead of its compute progress.
constexpr uint16_t kSparseNormalizedKFlagIds[2] = {10, 11};
constexpr uint16_t kSparseNormalizedQFlagIds[2] = {14, 15};

// Independent L0C accumulator release event IDs. Prime and drain every ID at
// the pipeline boundary; each role waits only before reusing its own region.
// Slots 2 and 5 are the two per-parity QS/O accumulators; every other slot
// guards a single region.
constexpr int32_t kL0CReleaseEventCount = 6;
constexpr int32_t kL0CReleaseEventIds[kL0CReleaseEventCount] = {0, 1, 2, 3, 4, 5};

__aicore__ inline uint16_t IterationNormalizedKFlag(int64_t pair)
{
    return kSparseNormalizedKFlagIds[pair & 1];
}

__aicore__ inline uint16_t IterationNormalizedQFlag(int64_t pair)
{
    return kSparseNormalizedQFlagIds[pair & 1];
}

template <pipe_t SetPipe>
__aicore__ inline void SetMode0Flag(uint16_t flagId)
{
    CrossCoreSetFlag<0x0, SetPipe>(flagId);
}

template <pipe_t WaitPipe>
__aicore__ inline void WaitMode0Flag(uint16_t flagId)
{
    CrossCoreWaitFlag<0x0, WaitPipe>(flagId);
}

__aicore__ inline uint16_t IterationBarrierFlag(int64_t iteration)
{
    return kSparseBarrierFlagIds[iteration & 1];
}

__aicore__ inline uint16_t IterationAPrimeFlag(int64_t iteration)
{
    return kSparseAPrimeBarrierFlagIds[iteration & 1];
}

// Producer-only: the FIX-pipe set stays ordered behind the kProdAPairDoneFlag
// FIX wait (both AIVs' APrime stores retired). Consumers must not use this
// wrapper; their publish is the templated one below.
__aicore__ inline void SetAPrimeAic(int64_t iteration)
{
    if ASCEND_IS_AIC {
        SetMode0Flag<PIPE_FIX>(IterationAPrimeFlag(iteration));
    }
}

template <pipe_t WaitPipe>
__aicore__ inline void WaitAPrimeAic(int64_t iteration)
{
    if ASCEND_IS_AIC {
        WaitMode0Flag<WaitPipe>(IterationAPrimeFlag(iteration));
    }
}

// Consumer credit join: wait for generation pair on this pipe, then return
// the pair+2 credit on the same pipe (flag IDs alternate with period 2, so
// pair+2 aliases pair's ID). The credit is strictly posterior to the
// receipt; credits stop at the last global generation, leaving no orphans.
template <pipe_t Pipe>
__aicore__ inline void JoinAPrimeAicCredit(int64_t pair, int64_t globalPairs)
{
    WaitAPrimeAic<Pipe>(pair);
    if (pair + 2 < globalPairs) {
        if ASCEND_IS_AIC {
            SetMode0Flag<Pipe>(IterationAPrimeFlag(pair + 2));
        }
    }
}

// T and APrime mode-0 generations are pair-granular, run as a credit
// protocol: the producer publishes a generation after its AIVs' stores and
// drains the generation from two pairs earlier before reusing the flag ID.
// The consumer primes generations 0/1 at chain start, then each wait(pair)
// is followed by a same-pipe set returning the pair+2 credit — the producer
// therefore tracks the consumer's load progress (up to 4 pairs ahead)
// instead of its compute progress. A generation only has to cover the GM
// loads themselves: the T/APrime GM regions are per-chunk unique and the L1
// slot reuse is guarded locally by MTE1_MTE2, so the barrier retires with
// the loads, not with the consumer's mmad/fix backlog.
// Consumer AIC primes generations 0/1 of both barriers. Round 1 exists only
// when there is a second global pair; priming it otherwise orphans the set.
__aicore__ inline void PrimeTAPrimeCreditsAic(int64_t globalPairs)
{
    if ASCEND_IS_AIC {
        SetMode0Flag<PIPE_MTE2>(IterationBarrierFlag(0));
        SetMode0Flag<PIPE_MTE2>(IterationAPrimeFlag(0));
        if (globalPairs >= 2) {
            SetMode0Flag<PIPE_MTE2>(IterationBarrierFlag(1));
            SetMode0Flag<PIPE_MTE2>(IterationAPrimeFlag(1));
        }
    }
}

template <pipe_t SetPipe = PIPE_FIX>
__aicore__ inline void SetIterationAic(int64_t iteration)
{
    if ASCEND_IS_AIC {
        SetMode0Flag<SetPipe>(IterationBarrierFlag(iteration));
    }
}

template <pipe_t WaitPipe>
__aicore__ inline void WaitIterationAic(int64_t iteration)
{
    if ASCEND_IS_AIC {
        WaitMode0Flag<WaitPipe>(IterationBarrierFlag(iteration));
    }
}

// Consumer credit join: wait for generation pair on this pipe, then return
// the pair+2 credit on the same pipe (IDs alternate with period 2, so pair+2
// aliases pair's ID). Same contract as JoinAPrimeAicCredit above.
template <pipe_t Pipe>
__aicore__ inline void JoinIterationAicCredit(int64_t pair, int64_t globalPairs)
{
    WaitIterationAic<Pipe>(pair);
    if (pair + 2 < globalPairs) {
        SetIterationAic<Pipe>(pair + 2);
    }
}

__aicore__ inline void AivPublish(uint16_t id)
{
    CrossCoreSetFlag<0x4, PIPE_MTE3>(id);
}

__aicore__ inline void AivWait(uint16_t id)
{
    CrossCoreWaitFlag<0x4, PIPE_V>(id);
}

__aicore__ inline void AivWaitMte3(uint16_t id)
{
    CrossCoreWaitFlag<0x4, PIPE_MTE3>(id);
}

__aicore__ inline void AicPublishToOwner(int64_t chunk, uint16_t id)
{
    const bool odd = (chunk & 1) != 0;
    CrossCoreSetFlag<0x4, PIPE_FIX>(static_cast<uint16_t>(id + (odd ? kAiv1FlagOffset : 0)));
}

template <pipe_t WaitPipe = PIPE_MTE1>
__aicore__ inline void AicWaitOwner(int64_t chunk, uint16_t id)
{
    const bool odd = (chunk & 1) != 0;
    CrossCoreWaitFlag<0x4, WaitPipe>(static_cast<uint16_t>(id + (odd ? kAiv1FlagOffset : 0)));
}

__aicore__ inline void AicPublishToBoth(uint16_t id)
{
    CrossCoreSetFlag<0x4, PIPE_FIX>(id);
    CrossCoreSetFlag<0x4, PIPE_FIX>(static_cast<uint16_t>(id + kAiv1FlagOffset));
}

template <pipe_t WaitPipe = PIPE_MTE1>
__aicore__ inline void AicWaitBoth(uint16_t id)
{
    CrossCoreWaitFlag<0x4, WaitPipe>(id);
    CrossCoreWaitFlag<0x4, WaitPipe>(static_cast<uint16_t>(id + kAiv1FlagOffset));
}

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_SYNC_H
