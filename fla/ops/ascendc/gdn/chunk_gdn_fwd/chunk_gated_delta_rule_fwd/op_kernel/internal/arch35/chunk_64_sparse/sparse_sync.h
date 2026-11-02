/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_CHUNK64_SPARSE_SYNC_H
#define GDN_CHUNK64_SPARSE_SYNC_H

#include "kernel_operator.h"

namespace GdnChunk64Sparse {

constexpr uint16_t kPhaseAicAllFlag = 12;
constexpr uint16_t kPhaseAivAllFlag = 13;
constexpr uint16_t kPhaseLocalMixFlag = 14;

// Call after the phase's stores and internal mode-2/mode-4 handshakes drain.
// Every participant must enter each collective in identical phase order.
__aicore__ inline void PhaseRendezvous()
{
    AscendC::PipeBarrier<PIPE_ALL>();
    if ASCEND_IS_AIC {
        AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(kPhaseAicAllFlag);
        AscendC::CrossCoreWaitFlag<0x0, PIPE_S>(kPhaseAicAllFlag);
    }
    if ASCEND_IS_AIV {
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(kPhaseAivAllFlag);
        AscendC::CrossCoreWaitFlag<0x0, PIPE_S>(kPhaseAivAllFlag);
    }

    if ASCEND_IS_AIC {
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kPhaseLocalMixFlag);
        AscendC::CrossCoreWaitFlag<0x2, PIPE_S>(kPhaseLocalMixFlag);
    } else if ASCEND_IS_AIV {
        AscendC::CrossCoreWaitFlag<0x2, PIPE_S>(kPhaseLocalMixFlag);
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(kPhaseLocalMixFlag);
    }
}

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_SYNC_H
