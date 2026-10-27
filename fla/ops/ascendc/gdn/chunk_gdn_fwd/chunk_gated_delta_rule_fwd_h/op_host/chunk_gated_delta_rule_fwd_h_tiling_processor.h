/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_fwd_h_tiling_processor.h
 * \brief Tiling computation decoupled from gert::TilingContext, reusable by both the
 *        aclnn tiling entry and the fast kernel launch C++ extension.
 *
 * The caller is responsible for resolving framework-specific information (shapes, dtypes,
 * platform core number, lib-api workspace size) into the plain context struct below. The
 * processor then fills the plain ChunkGatedDeltaRuleFwdHTilingDataRaw together with the block
 * dim and the total workspace size, mirroring exactly the original Tiling4ChunkGatedDeltaRuleFwdH.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_H_TILING_PROCESSOR_H
#define CHUNK_GATED_DELTA_RULE_FWD_H_TILING_PROCESSOR_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>   // std::getenv / std::atoll（GDN_FWD_H_VSPLITS 调优钩子）
#include <algorithm>

#include "../op_kernel/chunk_gated_delta_rule_fwd_h_struct.h"

namespace optiling {

// dtype enum convention shared with the kernel: 0 - fp16, 1 - bf16, 2 - fp32
static constexpr int64_t GDN_FWD_H_DTYPE_FP16 = 0;
static constexpr int64_t GDN_FWD_H_DTYPE_BF16 = 1;
static constexpr int64_t GDN_FWD_H_DTYPE_FP32 = 2;

static constexpr size_t GDN_FWD_H_WORKSPACE_RSV_BYTE = 16 * 1024 * 1024;
static constexpr size_t GDN_FWD_H_GM_ALIGN = 512;
static constexpr int64_t GDN_FWD_H_PING_PONG_STAGES = 2;

// Plain, framework-agnostic inputs needed to compute the tiling.
struct ChunkGatedDeltaRuleFwdHTilingContext {
    // shapes
    int64_t seqlen;        // k.dim(2)
    int64_t kNumHead;      // k.dim(1)
    int64_t kHeadDim;      // k.dim(3)
    int64_t vNumHead;      // u.dim(1)
    int64_t vHeadDim;      // u.dim(3)
    int64_t shapeBatchDim; // k.dim(0)
    // variable length
    bool hasCuSeqlens;
    int64_t cuSeqlensDim0; // length of cu_seqlens (only used when hasCuSeqlens)
    // dtypes (use GDN_FWD_H_DTYPE_*)
    int64_t dataType;      // input (k/w/u) dtype: fp16 or bf16
    int64_t gDataType;     // g dtype
    bool useInitialState;
    int64_t stateDataType; // initial/final state dtype
    bool useG;
    bool useGk;
    bool isAscend950;
    // attrs
    bool storeFinalState;
    int64_t chunkSize;
    // platform
    uint32_t aicCoreNum;
    size_t libApiWorkSpaceSize;
};

class ChunkGatedDeltaRuleFwdHTilingProcessor {
public:
    explicit ChunkGatedDeltaRuleFwdHTilingProcessor(const ChunkGatedDeltaRuleFwdHTilingContext &ctx) : ctx_(ctx) {}

    // Fills the plain tiling struct, the block dim and the total workspace size.
    void Process(::ChunkGatedDeltaRuleFwdHTilingDataRaw &tiling, uint32_t &blockDim, size_t &workspaceSize) const
    {
        int64_t isVariedLen;
        int64_t shapeBatch;
        int64_t tokenBatch;
        int64_t batch;

        if (!ctx_.hasCuSeqlens) {
            isVariedLen = 0;
            shapeBatch = ctx_.shapeBatchDim;
            tokenBatch = 1;
            batch = shapeBatch;
        } else {
            isVariedLen = 1;
            shapeBatch = 1;
            tokenBatch = ctx_.cuSeqlensDim0 - 1;
            batch = tokenBatch;
        }

        const int64_t aicCoreNum = static_cast<int64_t>(ctx_.aicCoreNum);
        const int64_t chunkSize = ctx_.chunkSize;
        const int64_t kHeadDim = ctx_.kHeadDim;
        const int64_t vHeadDim = ctx_.vHeadDim;

        // Value-dim split: independent chunks share nothing along V, so a chain's V axis can be
        // cut across cores without any extra communication.
        //
        // 实测结论：Preload 路径上开 V 切分是净亏，默认关闭。
        //   * W/K 的读取量不随 V 变小，每个 task 都要整份读，task 数翻倍后这两项总量翻倍；
        //     chunk32 每 task 152KB -> 2x84KB，总 GM 字节 +10%。
        //   * 每 chunk 时间是"4 级串行依赖 + 跨核 flag + GM 往返"的固定延迟（约 2.4us），
        //     与 tile 宽度基本无关（chunk64 2.77us / chunk32 2.44us / chunk32 半 V 2.70us），
        //     所以把 12 个 task 拆成 24 个并不会让每个 chunk 变快。
        //   * 实测：B=1,T=16384,HV=12 时 1247us -> 1384us。
        // 保留 GDN_FWD_H_VSPLITS 环境变量（1/2/...）用于继续 A/B 和后续多 stream 方案。
        const int64_t batchHeads = (isVariedLen ? tokenBatch : shapeBatch) * ctx_.vNumHead;
        int64_t vSplits = 1;
        if (const char *forcedVSplits = std::getenv("GDN_FWD_H_VSPLITS")) {
            vSplits = std::atoll(forcedVSplits);
        }
        if (vSplits < 1 || vHeadDim % vSplits != 0 || vSplits > vHeadDim / 64) {
            vSplits = 1;
        }
        tiling.vSplitCount = vSplits;
        // chunk64/128 keep the historical blockDim so their scheduling is untouched.
        // vSplits == 1 时 chunk32 也保持历史 blockDim（多出的核没有 task，只是空跑）。
        blockDim = ctx_.aicCoreNum;
        if (chunkSize == 32 && vSplits > 1) {
            blockDim = static_cast<uint32_t>(std::min<int64_t>(aicCoreNum, batchHeads * vSplits));
        }

        size_t workspaceOffset = ctx_.libApiWorkSpaceSize;
        workspaceOffset += GDN_FWD_H_WORKSPACE_RSV_BYTE;

        tiling.vWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(aicCoreNum * chunkSize * vHeadDim * static_cast<int64_t>(sizeof(float)) * GDN_FWD_H_PING_PONG_STAGES));

        tiling.vUpdateWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(aicCoreNum * chunkSize * vHeadDim * static_cast<int64_t>(sizeof(float)) * GDN_FWD_H_PING_PONG_STAGES));

        tiling.kDecayWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        if (ctx_.useGk) {
            workspaceOffset += AlignUp(static_cast<size_t>(aicCoreNum * chunkSize * kHeadDim * static_cast<int64_t>(sizeof(float)) * GDN_FWD_H_PING_PONG_STAGES));
        }

        tiling.hWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(aicCoreNum * kHeadDim * vHeadDim * static_cast<int64_t>(sizeof(float)) * GDN_FWD_H_PING_PONG_STAGES));

        tiling.numSeqWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>((tokenBatch + 1) * static_cast<int64_t>(sizeof(int64_t))));

        tiling.numChunksWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>((tokenBatch + 1) * static_cast<int64_t>(sizeof(int64_t))));

        // Contiguous per-core state buffers for the next-chunk state that Cube reads.
        // Layout: [core][stream][parity][K][vBlockSize], parity ping-pongs so the
        // Cube read of chunk c and the Vector write of chunk c+1 never overlap.
        // Sized with sizeof(float) as an upper bound (actual element is 2 bytes).
        tiling.stateWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(aicCoreNum * GDN_FWD_H_PING_PONG_STAGES *
                                                       GDN_FWD_H_PING_PONG_STAGES * kHeadDim * vHeadDim *
                                                       static_cast<int64_t>(sizeof(float))));

        workspaceOffset += GDN_FWD_H_WORKSPACE_RSV_BYTE;
        workspaceSize = workspaceOffset;

        FillCommonTiling(tiling, batch, isVariedLen, shapeBatch, tokenBatch, chunkSize);
    }

private:
    void FillCommonTiling(::ChunkGatedDeltaRuleFwdHTilingDataRaw &tiling, int64_t batch, int64_t isVariedLen,
                          int64_t shapeBatch, int64_t tokenBatch, int64_t chunkSize) const
    {
        tiling.batch = batch;
        tiling.seqlen = ctx_.seqlen;
        tiling.kNumHead = ctx_.kNumHead;
        tiling.vNumHead = ctx_.vNumHead;
        tiling.kHeadDim = ctx_.kHeadDim;
        tiling.vHeadDim = ctx_.vHeadDim;
        tiling.chunkSize = chunkSize;
        tiling.useInitialState = ctx_.useInitialState;
        tiling.storeFinalState = ctx_.storeFinalState;
        tiling.dataType = ctx_.dataType;
        tiling.gDataType = ctx_.gDataType;
        tiling.stateDataType = ctx_.stateDataType;
        tiling.isVariedLen = isVariedLen;
        tiling.shapeBatch = shapeBatch;
        tiling.tokenBatch = tokenBatch;
        tiling.useG = ctx_.useG;
        tiling.useGk = ctx_.useGk;
    }

    // Mirrors the original "(x + GM_ALIGN) / GM_ALIGN * GM_ALIGN" alignment.
    static size_t AlignUp(size_t x)
    {
        return (x + GDN_FWD_H_GM_ALIGN) / GDN_FWD_H_GM_ALIGN * GDN_FWD_H_GM_ALIGN;
    }

    const ChunkGatedDeltaRuleFwdHTilingContext &ctx_;
};

} // namespace optiling

#endif // CHUNK_GATED_DELTA_RULE_FWD_H_TILING_PROCESSOR_H
