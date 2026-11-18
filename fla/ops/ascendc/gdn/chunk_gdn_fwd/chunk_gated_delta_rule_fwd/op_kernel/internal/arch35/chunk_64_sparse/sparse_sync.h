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
// AIV mode-0 normalized K/Q readiness channels, indexed by pair parity.
constexpr uint16_t kSparseNormalizedKFlagIds[2] = {10, 11};
constexpr uint16_t kSparseNormalizedQFlagIds[2] = {14, 15};

// Independent L0C accumulator release event IDs. Prime and drain every ID at
// the pipeline boundary; each role waits only before reusing its own region.
constexpr int32_t kL0CReleaseEventCount = 5;
constexpr int32_t kL0CReleaseEventIds[kL0CReleaseEventCount] = {0, 1, 2, 3, 4};

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

template <pipe_t SetPipe, pipe_t WaitPipe>
__aicore__ inline void PublishMode0Flag(uint16_t flagId)
{
    SetMode0Flag<SetPipe>(flagId);
    WaitMode0Flag<WaitPipe>(flagId);
}

__aicore__ inline uint16_t IterationBarrierFlag(int64_t iteration)
{
    return kSparseBarrierFlagIds[iteration & 1];
}

__aicore__ inline uint16_t IterationAPrimeFlag(int64_t iteration)
{
    return kSparseAPrimeBarrierFlagIds[iteration & 1];
}

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

template <pipe_t WaitPipe>
__aicore__ inline void PublishAPrimeAic(int64_t iteration)
{
    SetAPrimeAic(iteration);
    WaitAPrimeAic<WaitPipe>(iteration);
}

// Producer sets this after the current T is complete but delays the matching
// wait until the same parity flag is reused two iterations later. The producer
// joins on its idle MTE2 pipe so its issue stream never stalls; the consumer
// keeps the default FIX join that transitively protects its single L1 T buffer.
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

// // Consumer still publishes and drains each T generation immediately.
template <pipe_t WaitPipe>
__aicore__ inline void PublishIterationAic(int64_t iteration)
{
    SetIterationAic(iteration);
    WaitIterationAic<WaitPipe>(iteration);
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
