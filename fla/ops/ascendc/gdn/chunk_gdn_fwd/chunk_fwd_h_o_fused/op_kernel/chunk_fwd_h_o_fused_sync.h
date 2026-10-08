/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */

#ifndef CHUNK_FWD_H_O_FUSED_SYNC_H
#define CHUNK_FWD_H_O_FUSED_SYNC_H

#include "kernel_operator.h"

namespace GDN {

#if !defined(CHUNK_FWD_HO_ARCH_A2) && !defined(CHUNK_FWD_HO_ARCH35)
#error "Include chunk_fwd_h_o_fused_arch.h before chunk_fwd_h_o_fused_sync.h"
#endif
#if defined(CHUNK_FWD_HO_ARCH_A2) && defined(CHUNK_FWD_HO_ARCH35)
#error "chunk_fwd_h_o_fused_sync.h received multiple architecture macros"
#endif

// H_READY is an AIC rendezvous: seven producer/consumer pairs contribute at
// most 14 Set operations to one flag. V_READY is an AIV rendezvous: matching
// relative AIV block indices are grouped six at a time, so one flag receives
// at most 12 Set operations from the H and O sides.
constexpr uint32_t CHUNK_FWD_HO_H_READY_GROUP_SIZE = 7;
constexpr uint32_t CHUNK_FWD_HO_H_READY_GROUP_COUNT = 3;
constexpr uint32_t CHUNK_FWD_HO_V_READY_GROUP_SIZE = 6;
constexpr uint32_t CHUNK_FWD_HO_MAX_TASK_PAIRS =
    CHUNK_FWD_HO_H_READY_GROUP_SIZE * CHUNK_FWD_HO_H_READY_GROUP_COUNT;
constexpr uint32_t CHUNK_FWD_HO_MAX_AIV_PARTICIPANTS =
    2 * CHUNK_FWD_HO_MAX_TASK_PAIRS;
constexpr uint32_t CHUNK_FWD_HO_V_READY_GROUP_COUNT =
    (CHUNK_FWD_HO_MAX_AIV_PARTICIPANTS + CHUNK_FWD_HO_V_READY_GROUP_SIZE - 1) /
    CHUNK_FWD_HO_V_READY_GROUP_SIZE;
// Mode-0 has an independent flag namespace. H_READY keeps its existing
// interleaved 0/2/4 mapping, while V_READY maps consecutive AIV groups from 0.
constexpr uint16_t CHUNK_FWD_HO_H_READY_FLAG_BASE = 0;
constexpr uint16_t CHUNK_FWD_HO_V_READY_FLAG_BASE = 0;
static_assert(CHUNK_FWD_HO_H_READY_FLAG_BASE +
                  2 * (CHUNK_FWD_HO_H_READY_GROUP_COUNT - 1) < 16,
              "H_READY mode-0 flags exceed the 16-entry flag range");
static_assert(CHUNK_FWD_HO_V_READY_FLAG_BASE +
                  CHUNK_FWD_HO_V_READY_GROUP_COUNT - 1 < 16,
              "H/O mode-0 flags exceed the 16-entry flag range");

// H_READY receives a MIX pair index. V_READY receives a side-relative AIV
// block index, identical on the producer and consumer sides.
__aicore__ inline uint16_t ChunkFwdHOReadyFlag(int32_t eventId, uint32_t participantIdx)
{
    const bool isVReady = eventId >= CHUNK_FWD_HO_V_READY_EVENT_BASE;
    const uint32_t group = participantIdx /
        (isVReady ? CHUNK_FWD_HO_V_READY_GROUP_SIZE
                  : CHUNK_FWD_HO_H_READY_GROUP_SIZE);
    return static_cast<uint16_t>(isVReady
        ? CHUNK_FWD_HO_V_READY_FLAG_BASE + group
        : CHUNK_FWD_HO_H_READY_FLAG_BASE + 2 * group);
}

struct ActiveChunkFwdHOSync {
    __aicore__ static inline void Set(int32_t eventId, uint32_t pairId)
    {
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(
            ChunkFwdHOReadyFlag(eventId, pairId));
    }

    __aicore__ static inline void Wait(int32_t eventId, uint32_t pairId)
    {
        AscendC::CrossCoreWaitFlag<0x0, PIPE_MTE3>(
            ChunkFwdHOReadyFlag(eventId, pairId));
    }

    __aicore__ static inline void AicHReadySetWait(uint32_t pairId)
    {
        Set(CHUNK_FWD_HO_H_READY_EVENT_BASE, pairId);
        Wait(CHUNK_FWD_HO_H_READY_EVENT_BASE, pairId);
    }

    __aicore__ static inline void AicVReadySetWait(uint32_t pairId)
    {
        Set(CHUNK_FWD_HO_V_READY_EVENT_BASE, pairId);
        Wait(CHUNK_FWD_HO_V_READY_EVENT_BASE, pairId);
    }
};

} // namespace GDN

#endif // CHUNK_FWD_H_O_FUSED_SYNC_H
