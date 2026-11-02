/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_CHUNK64_SPARSE_BARRIER_H
#define GDN_CHUNK64_SPARSE_BARRIER_H

#include "kernel_operator.h"
#include "sparse_fused_tiling.h"

namespace GdnChunk64Sparse {

// Mode 0 is the only CrossCore mode that spans AI Cores: every AIC (or every
// AIV) of the launch sets the same id and waits for the full set. It therefore
// paces the producer/consumer roles; per-chain point-to-point handshakes would
// need GM ready slots and IBSet/IBWait instead.
constexpr uint8_t kBarrierMode = 0x0;

// Cross-group publication is carried by the AIVs only: on both sides the AIV
// is the sole GM gateway, so the producer's MTE3 stores and the consumer's
// MTE2 loads are the same core type and mode 0 orders them. The wait is pinned
// to MTE2 because the next thing the consumer AIV does with published data is a
// GM load; the implicit PIPE_S wait races on arch35.
//
// Every launched AIV must call this once per iteration, in the same order,
// including groups with no work, or the collective never completes. The AIC
// does not enter it: the cube never reads GM, and the iteration boundary
// reaches it through the intra-group relay below.
__aicore__ inline void AivRoleBarrier(uint32_t flagId)
{
    if ASCEND_IS_AIV {
        const uint16_t id = static_cast<uint16_t>(flagId);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::CrossCoreSetFlag<kBarrierMode, PIPE_MTE3>(id);
        AscendC::CrossCoreWaitFlag<kBarrierMode, PIPE_MTE2>(id);
    }
}

// AIV -> its own AIC, one MIX core: mode 2 AND-reduces both subblocks, so the
// cube starts only after both vector halves have published their L1 tiles.
__aicore__ inline void AivRelayToAic(uint16_t base)
{
    if ASCEND_IS_AIV {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(base);
    }
}

__aicore__ inline void AicWaitRelayMte1(uint16_t base)
{
    if ASCEND_IS_AIC {
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE1>(base);
    }
}

// ---------------------------------------------------------------------------
// Intra-group handshakes. Mode 4 addresses one AIV at a time. The documented
// 3510 routing (see CrossCoreWaitFlag, flagId range) is:
//   AIV0 Set id 0..15  -> AIC Wait id 0..15
//   AIV1 Set id 0..15  -> AIC Wait id 16..31
//   AIC  Set id 0..15  -> AIV0 Wait id 0..15
//   AIC  Set id 16..31 -> AIV1 Wait id 0..15
// So an AIV always passes the raw 0..15 id and the sub-block identity is
// carried by the issuing core; only the AIC uses the +16 half. Passing
// base+16 from AIV1 would be out of its local range (truncated to 4 bits) and
// is rejected here instead of relying on that truncation. Both subblocks must
// reach every handshake even when one has no local work (OL-212: an idle
// subblock never Sets and the AIC waits forever).
// ---------------------------------------------------------------------------
constexpr uint16_t kSubBlockFlagOffset = 16;

__aicore__ inline void AivSetMte3(uint16_t base)
{
    AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(base);
}

__aicore__ inline void AivSetFix(uint16_t base)
{
    AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(base);
}

__aicore__ inline void AivWaitMte2(uint16_t base)
{
    AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE2>(base);
}

__aicore__ inline void AivWaitV(uint16_t base)
{
    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(base);
}

__aicore__ inline void AicSetFix(uint16_t base)
{
    AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(base);
    AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(
        static_cast<uint16_t>(base + kSubBlockFlagOffset));
}

__aicore__ inline void AicWaitMte1(uint16_t base)
{
    AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(base);
    AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(
        static_cast<uint16_t>(base + kSubBlockFlagOffset));
}

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_BARRIER_H
