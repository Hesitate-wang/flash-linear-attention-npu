/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */

#ifndef CHUNK_FWD_H_O_FUSED_SYNC_H
#define CHUNK_FWD_H_O_FUSED_SYNC_H

#include "kernel_operator.h"

namespace GDN {

enum class ChunkFwdHOSyncArch : uint32_t {
    ATLAS_A2,
    ASCEND_950,
};

template <ChunkFwdHOSyncArch Arch>
struct ChunkFwdHOSync;

#if defined(CHUNK_FWD_HO_ARCH_A2)

// A2 keeps the CANN-provided protocol, including its architecture-specific
// stubs and full-pipe barriers.
template <>
struct ChunkFwdHOSync<ChunkFwdHOSyncArch::ATLAS_A2> {
    template <bool IsAivOnly>
    __aicore__ static inline void Set(
        const AscendC::GlobalTensor<int32_t> &gmWorkspace,
        const AscendC::LocalTensor<int32_t> &ubWorkspace,
        int32_t blockIdx,
        int32_t eventId)
    {
        AscendC::IBSet<IsAivOnly>(
            gmWorkspace, ubWorkspace, blockIdx, eventId);
    }

    template <bool IsAivOnly>
    __aicore__ static inline void Wait(
        const AscendC::GlobalTensor<int32_t> &gmWorkspace,
        const AscendC::LocalTensor<int32_t> &ubWorkspace,
        int32_t blockIdx,
        int32_t eventId)
    {
        AscendC::IBWait<IsAivOnly>(
            gmWorkspace, ubWorkspace, blockIdx, eventId);
    }
};

using ActiveChunkFwdHOSync =
    ChunkFwdHOSync<ChunkFwdHOSyncArch::ATLAS_A2>;

#elif defined(CHUNK_FWD_HO_ARCH35)

// Ascend 950 specialization. Only the producer protocol is changed: the
// leading PIPE_ALL barrier in the stock IBSet is omitted. The polling loop and
// the trailing barrier are intentionally kept identical to the stock protocol.
template <>
struct ChunkFwdHOSync<ChunkFwdHOSyncArch::ASCEND_950> {
    template <bool IsAivOnly>
    __aicore__ static inline void SetNoPreBarrier(
        const AscendC::GlobalTensor<int32_t> &gmWorkspace,
        const AscendC::LocalTensor<int32_t> &ubWorkspace,
        int32_t blockIdx,
        int32_t eventId)
    {
        if ASCEND_IS_AIC {
            return;
        }

        int32_t blockNum = AscendC::GetBlockNum();
        if (!IsAivOnly) {
            blockNum *= 2;
        }

        constexpr int32_t syncWords = 32 / sizeof(int32_t);
        auto localSyncGm =
            gmWorkspace[blockNum * syncWords * eventId + blockIdx * syncWords];

        // Contract: the caller has already established the required producer
        // data visibility. This UB tile must not alias live Vector operands.
        while (true) {
            AscendC::DataCopy(ubWorkspace, localSyncGm, syncWords);
            AscendC::TEventID mte2ToScalar =
                AscendC::GetTPipePtr()->FetchEventID(
                    AscendC::HardEvent::MTE2_S);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(mte2ToScalar);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(mte2ToScalar);

            if (ubWorkspace.GetValue(0) == 0) {
                ubWorkspace.SetValue(0, 1);
                AscendC::TEventID scalarToMte3 =
                    AscendC::GetTPipePtr()->FetchEventID(
                        AscendC::HardEvent::S_MTE3);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3);
                AscendC::DataCopy(localSyncGm, ubWorkspace, syncWords);
                break;
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <bool IsAivOnly>
    __aicore__ static inline void Set(
        const AscendC::GlobalTensor<int32_t> &gmWorkspace,
        const AscendC::LocalTensor<int32_t> &ubWorkspace,
        int32_t blockIdx,
        int32_t eventId)
    {
        SetNoPreBarrier<IsAivOnly>(
            gmWorkspace, ubWorkspace, blockIdx, eventId);
    }

    // Preferred entry after an asynchronous producer GM write. The caller
    // owns eventIdMte3ToMte2 and remains responsible for its allocation and
    // release; this narrow dependency replaces the removed leading PIPE_ALL.
    template <bool IsAivOnly>
    __aicore__ static inline void SetAfterMte3(
        const AscendC::GlobalTensor<int32_t> &gmWorkspace,
        const AscendC::LocalTensor<int32_t> &ubWorkspace,
        int32_t blockIdx,
        int32_t eventId,
        AscendC::TEventID eventIdMte3ToMte2)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
            eventIdMte3ToMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
            eventIdMte3ToMte2);
        SetNoPreBarrier<IsAivOnly>(
            gmWorkspace, ubWorkspace, blockIdx, eventId);
    }

    template <bool IsAivOnly>
    __aicore__ static inline void Wait(
        const AscendC::GlobalTensor<int32_t> &gmWorkspace,
        const AscendC::LocalTensor<int32_t> &ubWorkspace,
        int32_t blockIdx,
        int32_t eventId)
    {
        AscendC::IBWait<IsAivOnly>(
            gmWorkspace, ubWorkspace, blockIdx, eventId);
    }
};

using ActiveChunkFwdHOSync =
    ChunkFwdHOSync<ChunkFwdHOSyncArch::ASCEND_950>;

#else
#error "Include chunk_fwd_h_o_fused_arch.h before chunk_fwd_h_o_fused_sync.h"
#endif

struct ChunkFwdHOProducerReadySignal {
    AscendC::GlobalTensor<int32_t> gmWorkspace;
    AscendC::LocalTensor<int32_t> ubWorkspace;
    int32_t blockIdx{0};
    int32_t eventId{0};
    bool enabled{false};

    template <bool IsAivOnly>
    __aicore__ inline void Publish() const
    {
        if (!enabled) {
            return;
        }
        ActiveChunkFwdHOSync::template Set<IsAivOnly>(
            gmWorkspace, ubWorkspace, blockIdx, eventId);
    }
};

} // namespace GDN

#endif // CHUNK_FWD_H_O_FUSED_SYNC_H
