/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef CATLASS_GDN_FWD_HO_FUSED_MODE2_SCHEDULER_HPP
#define CATLASS_GDN_FWD_HO_FUSED_MODE2_SCHEDULER_HPP

#include <cstdint>
#include "catlass/catlass.hpp"
#include "../../../chunk_fwd_h_o_fused_struct.h"

namespace Catlass::Gemm::Block {

struct GdnFusedMode2Chunk {
    uint64_t qk{0}, w{0}, uv{0}, gate{0};
    uint64_t hInput{0}, hOutput{0}, state{0};
    uint64_t attnWork{0}, tokenWork{0}, stateWork{0};
    bool first{true}, last{false};
};

// One task per MIX core; H has 128 rows, token matrices have 64 rows.
class BlockSchedulerGdnFwdHOFusedMode2 {
public:
    uint64_t task{0};
    uint32_t chunk{0}, chunks{0};
    bool isRunning{false};
    GdnFusedMode2Chunk offsets;

    CATLASS_DEVICE static bool Supports(const __gm__ GDN::ChunkFwdHOFusedTilingData *t)
    {
        return t->executionMode == 2 && !t->isVariedLen && !t->useExp2 && !t->useGk &&
               t->useG && t->chunkSize == 64 && t->kHeadDim == 128 && t->vHeadDim == 128 &&
               t->seqlen > 0 && t->seqlen % 64 == 0 &&
               t->numChunksPerBatch == t->seqlen / 64 &&
               t->kNumHead > 0 && t->vNumHead >= t->kNumHead && t->vNumHead % t->kNumHead == 0;
    }

    CATLASS_DEVICE void Init(const __gm__ GDN::ChunkFwdHOFusedTilingData *t, uint32_t core)
    {
        task = core;
        chunks = static_cast<uint32_t>(t->numChunksPerBatch);
        tokens_ = static_cast<uint64_t>(t->seqlen);
        kHeads_ = static_cast<uint64_t>(t->kNumHead);
        vHeads_ = static_cast<uint64_t>(t->vNumHead);
        core_ = core;
        chunk = 0;
        isRunning = Supports(t) && task < static_cast<uint64_t>(t->singleCoreTaskNum) &&
                    task < static_cast<uint64_t>(t->shapeBatch) * vHeads_;
        if (isRunning) Update();
    }

    CATLASS_DEVICE bool Next()
    {
        if (!isRunning) return false;
        ++chunk;
        isRunning = chunk < chunks;
        if (isRunning) Update();
        return isRunning;
    }

private:
    uint64_t tokens_{0}, kHeads_{0}, vHeads_{0}, core_{0};

    CATLASS_DEVICE void Update()
    {
        const uint64_t batch = task / vHeads_;
        const uint64_t head = task % vHeads_;
        const uint64_t kHead = head / (vHeads_ / kHeads_);
        const uint64_t start = static_cast<uint64_t>(chunk) * 64U;
        offsets.qk = ((batch * kHeads_ + kHead) * tokens_ + start) * 128U;
        offsets.w = ((batch * vHeads_ + head) * tokens_ + start) * 128U;
        offsets.uv = offsets.w;
        offsets.gate = (batch * vHeads_ + head) * tokens_ + start;
        offsets.state = task * 128U * 128U;
        offsets.hInput = (task * chunks + chunk) * 128U * 128U;
        // No next-state handoff is written after the final chunk.
        offsets.hOutput = offsets.hInput + 128U * 128U;
        offsets.attnWork = core_ * 64U * 64U;
        offsets.tokenWork = core_ * 64U * 128U;
        offsets.stateWork = core_ * 128U * 128U;
        offsets.first = chunk == 0;
        offsets.last = chunk + 1 == chunks;
    }
};
} // namespace Catlass::Gemm::Block
#endif
