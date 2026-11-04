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

// The launch contains exactly 2*chainCount MIX groups. There are no idle
// participants: every launched AIV reaches this barrier once per iteration.
__aicore__ inline void PublishIteration()
{
    if ASCEND_IS_AIV {
        PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<0x0, PIPE_MTE3>(kSparseBarrierFlagId);
        CrossCoreWaitFlag<0x0, PIPE_MTE2>(kSparseBarrierFlagId);
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

__aicore__ inline void AicPublishToAiv0(uint16_t id)
{
    CrossCoreSetFlag<0x4, PIPE_FIX>(id);
}

__aicore__ inline void AicPublishToBoth(uint16_t id)
{
    CrossCoreSetFlag<0x4, PIPE_FIX>(id);
    CrossCoreSetFlag<0x4, PIPE_FIX>(static_cast<uint16_t>(id + kAiv1FlagOffset));
}

__aicore__ inline void AicWaitBoth(uint16_t id)
{
    CrossCoreWaitFlag<0x4, PIPE_MTE1>(id);
    CrossCoreWaitFlag<0x4, PIPE_MTE1>(static_cast<uint16_t>(id + kAiv1FlagOffset));
}

__aicore__ inline void PublishChainInGroup()
{
    if ASCEND_IS_AIV {
        PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<0x1, PIPE_MTE3>(15);
        CrossCoreWaitFlag<0x1, PIPE_MTE2>(15);
    }
}

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_SYNC_H
