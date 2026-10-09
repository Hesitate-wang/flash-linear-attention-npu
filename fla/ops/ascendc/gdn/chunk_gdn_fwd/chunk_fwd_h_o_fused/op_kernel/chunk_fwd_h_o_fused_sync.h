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
constexpr uint16_t CHUNK_FWD_HO_H_READY_FLAG_BASE = 10;
constexpr uint16_t CHUNK_FWD_HO_V_READY_FLAG_BASE = 10;

// H_READY receives a MIX pair index. V_READY receives a side-relative AIV
// block index, identical on the producer and consumer sides.

struct ActiveChunkFwdHOSync {
    __aicore__ static inline void Set(uint16_t flagId)
    {
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(flagId);
    }

    __aicore__ static inline void Wait(uint16_t flagId)
    {
        AscendC::CrossCoreWaitFlag<0x0, PIPE_MTE3>(flagId);
    }

    __aicore__ static inline void AicHReadySetWait()
    {
        Set(CHUNK_FWD_HO_H_READY_FLAG_BASE);
        Wait(CHUNK_FWD_HO_H_READY_FLAG_BASE);
    }

    __aicore__ static inline void AicVReadySetWait()
    {
        Set(CHUNK_FWD_HO_V_READY_FLAG_BASE);
        Wait(CHUNK_FWD_HO_V_READY_FLAG_BASE);
    }
};

} // namespace GDN

#endif // CHUNK_FWD_H_O_FUSED_SYNC_H
