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

// These are mode-0 inter-core flags.  They are deliberately outside the
// 0..9 mode-2 AIC/AIV flag range used by the H/O local schedulers.
constexpr uint16_t CHUNK_FWD_HO_H_READY_FLAG = 12;
constexpr uint16_t CHUNK_FWD_HO_V_READY_FLAG = 13;

// eventId retains the old H/V lane encoding for call-site stability:
// H lanes occupy [H_BASE, V_BASE), V lanes occupy [V_BASE, READY_EVENT_COUNT).
// Do not compare against H_BASE directly, since that misclassifies H lanes
// other than lane zero as V_READY when more than one task lane is enabled.
__aicore__ inline uint16_t ChunkFwdHOReadyFlag(int32_t eventId)
{
    return eventId >= CHUNK_FWD_HO_V_READY_EVENT_BASE
        ? CHUNK_FWD_HO_V_READY_FLAG : CHUNK_FWD_HO_H_READY_FLAG;
}

struct ActiveChunkFwdHOSync {
    __aicore__ static inline void Set(int32_t eventId)
    {
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(
            ChunkFwdHOReadyFlag(eventId));
    }

    __aicore__ static inline void Wait(int32_t eventId)
    {
        AscendC::CrossCoreWaitFlag<0x0, PIPE_MTE2>(
            ChunkFwdHOReadyFlag(eventId));
    }
};

struct ChunkFwdHOProducerReadySignal {
    int32_t eventId{0};
    bool enabled{false};

    __aicore__ inline void Publish() const
    {
        if (!enabled) {
            return;
        }
        ActiveChunkFwdHOSync::Set(eventId);
        ActiveChunkFwdHOSync::Wait(eventId);
    }
};

// Consumer-side handoff used by the A5 O epilogue.  The epilogue invokes this
// after issuing independent Vector work, allowing that work to cover the
// cross-core V_new wait.
struct ChunkFwdHOConsumerReadyWait {
    int32_t eventId{0};
    bool enabled{false};

    __aicore__ inline void Wait() const
    {
        if (!enabled) {
            return;
        }
        ActiveChunkFwdHOSync::Set(eventId);
        ActiveChunkFwdHOSync::Wait(eventId);
    }

    __aicore__ inline void Participate() const
    {
        if (!enabled) {
            return;
        }
        Wait();
    }
};

} // namespace GDN

#endif // CHUNK_FWD_H_O_FUSED_SYNC_H
