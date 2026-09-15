# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import math

import pytest
import torch

from fla_npu.ops import ascendc as ascendc_ops


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def test_chunk_fwd_h_o_fused_public_names_are_exported():
    assert hasattr(ascendc_ops, "npu_chunk_fwd_h_o_fused")
    assert hasattr(ascendc_ops, "chunk_fwd_h_o_fused")


@pytest.mark.skipif(not _npu_available(), reason="NPU device not found")
def test_chunk_fwd_h_o_fused_matches_public_h_then_o_composition():
    torch.manual_seed(20260915)
    dtype = torch.bfloat16
    batch, k_heads, v_heads, tokens, k_dim, v_dim = 1, 1, 1, 64, 128, 128
    chunk_size = 64
    scale = 1.0 / math.sqrt(k_dim)

    k = (torch.randn(batch, k_heads, tokens, k_dim, dtype=dtype) * 0.02).npu()
    q = (torch.randn(batch, k_heads, tokens, k_dim, dtype=dtype) * 0.02).npu()
    w = (torch.randn(batch, v_heads, tokens, k_dim, dtype=dtype) * 0.02).npu()
    u = (torch.randn(batch, v_heads, tokens, v_dim, dtype=dtype) * 0.02).npu()
    g = torch.zeros(batch, v_heads, tokens, dtype=torch.float32).npu()

    expected_h, expected_v_new, expected_final = ascendc_ops.npu_chunk_fwd_h(
        k,
        w,
        u,
        g=g,
        output_final_state=True,
        chunk_size=chunk_size,
        state_v_first=False,
    )
    expected_o = ascendc_ops.npu_chunk_fwd_o(
        q,
        k,
        expected_v_new,
        expected_h,
        scale,
        g=g,
        chunk_size=chunk_size,
        transpose_state_layout=False,
        use_exp2=False,
        output_layout="BNSD",
    )

    actual_o, actual_final = ascendc_ops.npu_chunk_fwd_h_o_fused(
        k,
        w,
        u,
        g,
        q,
        output_final_state=True,
        chunk_size=chunk_size,
        scale=scale,
        use_exp2=False,
        state_v_first=False,
        output_layout="BNSD",
    )
    torch.npu.synchronize()

    for name, actual, expected in (
        ("o", actual_o, expected_o),
        ("final_state", actual_final, expected_final),
    ):
        torch.testing.assert_close(
            actual.cpu().float(),
            expected.cpu().float(),
            rtol=5e-3,
            atol=5e-3,
            msg=f"{name} mismatch",
        )
