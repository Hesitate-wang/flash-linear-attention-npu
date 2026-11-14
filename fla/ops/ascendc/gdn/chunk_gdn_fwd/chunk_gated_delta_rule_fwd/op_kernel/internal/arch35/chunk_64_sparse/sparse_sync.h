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
constexpr uint16_t kSparseBarrierFlagIdAlt = 12;
constexpr uint16_t kSparseAPrimeBarrierFlagId = 13;
constexpr uint16_t kSparseAPrimeBarrierFlagIdAlt = 14;

__aicore__ inline uint16_t IterationBarrierFlag(int64_t iteration)
{
    return (iteration & 1) != 0 ? kSparseBarrierFlagIdAlt : kSparseBarrierFlagId;
}

__aicore__ inline uint16_t IterationAPrimeFlag(int64_t iteration)
{
    return (iteration & 1) != 0 ? kSparseAPrimeBarrierFlagIdAlt : kSparseAPrimeBarrierFlagId;
}

__aicore__ inline void SetAPrimeAic(int64_t iteration)
{
    // if ASCEND_IS_AIC {
    //     CrossCoreSetFlag<0x0, PIPE_MTE3>(IterationAPrimeFlag(iteration));
    // }
}

template <pipe_t WaitPipe>
__aicore__ inline void WaitAPrimeAic(int64_t iteration)
{
    // if ASCEND_IS_AIC {
    //     CrossCoreWaitFlag<0x0, WaitPipe>(IterationAPrimeFlag(iteration));
    // }
}

template <pipe_t WaitPipe>
__aicore__ inline void PublishAPrimeAic(int64_t iteration)
{
    SetAPrimeAic(iteration);
    WaitAPrimeAic<WaitPipe>(iteration);
}

// Producer sets this after the current T is complete but delays the matching
// wait until the same parity flag is reused two iterations later.
__aicore__ inline void SetIterationAic(int64_t iteration)
{
    // if ASCEND_IS_AIC {
    //     CrossCoreSetFlag<0x0, PIPE_FIX>(IterationBarrierFlag(iteration));
    // }
}

template <pipe_t WaitPipe>
__aicore__ inline void WaitIterationAic(int64_t iteration)
{
    // if ASCEND_IS_AIC {
    //     CrossCoreWaitFlag<0x0, WaitPipe>(IterationBarrierFlag(iteration));
    // }
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

__aicore__ inline void AivWaitMte2(uint16_t id)
{
    CrossCoreWaitFlag<0x4, PIPE_MTE2>(id);
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
