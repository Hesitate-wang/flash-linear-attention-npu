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

// Mode-0 flags are shared by paired H/O mixed cores. Seven task pairs use one
// phase flag; H_READY retains the AIC rendezvous, while the A5 V128 V_READY
// handoff is published and consumed by the corresponding AIV lanes.
constexpr uint32_t CHUNK_FWD_HO_FLAG_GROUP_SIZE = 7;
constexpr uint32_t CHUNK_FWD_HO_FLAG_GROUP_COUNT = 3;
constexpr uint32_t CHUNK_FWD_HO_MAX_TASK_PAIRS =
    CHUNK_FWD_HO_FLAG_GROUP_SIZE * CHUNK_FWD_HO_FLAG_GROUP_COUNT;
// Scheduler flags occupy IDs 0..9 (the O scheduler uses five ping-pong
// pairs). Keep the mode-0 cross-core rendezvous in a separate ID range.
constexpr uint16_t CHUNK_FWD_HO_H_READY_FLAG_BASE = 10;
constexpr uint16_t CHUNK_FWD_HO_V_READY_FLAG_BASE = 11;
static_assert(CHUNK_FWD_HO_V_READY_FLAG_BASE +
                  2 * (CHUNK_FWD_HO_FLAG_GROUP_COUNT - 1) < 16,
              "H/O mode-0 flags exceed the 16-entry flag range");

// Logical H/V event IDs share one mode-0 flag per handoff phase.
__aicore__ inline uint16_t ChunkFwdHOReadyFlag(int32_t eventId, uint32_t pairId)
{
    const uint32_t group = pairId / CHUNK_FWD_HO_FLAG_GROUP_SIZE;
    return static_cast<uint16_t>((eventId >= CHUNK_FWD_HO_V_READY_EVENT_BASE
        ? CHUNK_FWD_HO_V_READY_FLAG_BASE : CHUNK_FWD_HO_H_READY_FLAG_BASE) +
        2 * group);
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
