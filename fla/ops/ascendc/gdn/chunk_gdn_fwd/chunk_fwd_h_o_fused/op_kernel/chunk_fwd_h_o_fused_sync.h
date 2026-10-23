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

// Every active H/O AIC or AIV joins the same rendezvous for each chunk round.
constexpr uint16_t CHUNK_FWD_HO_H_READY_FLAG = 12;
constexpr uint16_t CHUNK_FWD_HO_V_READY_FLAG = 13;

struct ActiveChunkFwdHOSync {
    __aicore__ static inline void AicHReadySetWait()
    {
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(CHUNK_FWD_HO_H_READY_FLAG);
        AscendC::CrossCoreWaitFlag<0x0, PIPE_MTE3>(CHUNK_FWD_HO_H_READY_FLAG);
    }

    __aicore__ static inline void AicVReadySetWait()
    {
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(CHUNK_FWD_HO_V_READY_FLAG);
        AscendC::CrossCoreWaitFlag<0x0, PIPE_MTE3>(CHUNK_FWD_HO_V_READY_FLAG);
    }
};

} // namespace GDN

#endif // CHUNK_FWD_H_O_FUSED_SYNC_H
