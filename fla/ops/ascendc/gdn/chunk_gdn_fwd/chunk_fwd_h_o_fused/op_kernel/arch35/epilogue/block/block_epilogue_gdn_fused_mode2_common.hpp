/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef CATLASS_EPILOGUE_GDN_FUSED_MODE2_COMMON_HPP
#define CATLASS_EPILOGUE_GDN_FUSED_MODE2_COMMON_HPP

#include <type_traits>
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "kernel_operator.h"
#include "../gdn_fused_epilogue_policies.hpp"

namespace Catlass::Epilogue::Block::mode2_fixed64_detail {
using Ub = gdnFusedMode2UbMallocTable;

// Fixed-shape 16-bit row-major UB -> zN L1. The destination is the full
// matrix origin; each AIV writes only its own rows. Caller owns ready/free.
template <uint32_t FullRows, uint32_t Columns, uint32_t Rows, class T>
CATLASS_DEVICE void CopyRowsToL1Zn(
    AscendC::LocalTensor<T> dst, AscendC::LocalTensor<T> src, uint32_t rowBegin)
{
    static_assert(sizeof(T) == 2 && Columns % 16 == 0);
    static_assert(FullRows == 2 * Rows);
    constexpr uint32_t c0 = 16;
    AscendC::DataCopyParams params{Rows, 1, Columns / c0 - 1, 0};
    for (uint32_t panel = 0; panel < Columns / c0; ++panel) {
        AscendC::DataCopy(dst[panel * FullRows * c0 + rowBegin * c0],
                         src[panel * c0], params);
    }
}

template <class GateT>
class GateBuffer {
public:
    static_assert(sizeof(GateT) == 2 || std::is_same_v<GateT, float>,
                  "MODE2 gate must be FP16/BF16/FP32");
    CATLASS_DEVICE explicit GateBuffer(Arch::Resource<Arch::Ascend950> &resource)
    {
        input_ = resource.ubBuf.template GetBufferByByte<GateT>(Ub::g_input_ping);
        fp32_ = resource.ubBuf.template GetBufferByByte<float>(Ub::g_fp32_ping);
        input_.SetSize(64);
        fp32_.SetSize(64);
    }

    // Driver calls once per chunk, after the previous chunk has released UB.
    // Event 7 belongs exclusively to this preparation; every epilogue only reads
    // Tensor(). Do not prepare the next chunk until all current readers finish.
    CATLASS_DEVICE void Prepare(AscendC::GlobalTensor<GateT> src)
    {
        AscendC::ResetMask();
        if constexpr (std::is_same_v<GateT, float>) {
            AscendC::DataCopy(fp32_, src, 64);
        } else {
            AscendC::DataCopy(input_, src, 64);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
        if constexpr (!std::is_same_v<GateT, float>) {
            AscendC::Cast(fp32_, input_, AscendC::RoundMode::CAST_NONE, 64);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    CATLASS_DEVICE AscendC::LocalTensor<float> Tensor() const { return fp32_; }
private:
    AscendC::LocalTensor<GateT> input_;
    AscendC::LocalTensor<float> fp32_;
};

static_assert(Ub::chunk_tokens == 64 && Ub::rows_per_aiv == 32 && Ub::head_dim == 128,
              "these epilogues implement only full 64/128 MODE2 tiles");
static_assert(Ub::mask + 32U * 32U * sizeof(float) <= Ub::g_input_ping);
static_assert(Ub::g_input_ping + 64U * sizeof(float) <= Ub::g_fp32_ping);
static_assert(Ub::g_fp32_ping + 64U * sizeof(float) <= Ub::shard_mem);
} // namespace Catlass::Epilogue::Block::mode2_fixed64_detail
#endif
