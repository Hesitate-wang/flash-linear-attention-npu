#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Tianjin University, Ltd.

import pytest
import torch
import torch_npu

import ascend_ops


SYNC_WORKSPACE_ELEMENTS = 8  # 32 bytes, required by IBSet/IBWait.
MATRIX_SHAPE = (8, 8)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_ib_set_wait_copies_matrix_between_two_logical_cores():
    matrix_elements = MATRIX_SHAPE[0] * MATRIX_SHAPE[1]

    # One zero-initialized GM workspace: [IB sync area | matrix A | matrix B].
    workspace = torch.zeros(
        SYNC_WORKSPACE_ELEMENTS + 2 * matrix_elements,
        dtype=torch.int32,
        device="npu",
    )

    torch.ops.ascend_ops.ib_set_wait_test(workspace, matrix_elements)
    torch.npu.synchronize()

    workspace_cpu = workspace.cpu()
    matrix_a = workspace_cpu[
        SYNC_WORKSPACE_ELEMENTS : SYNC_WORKSPACE_ELEMENTS + matrix_elements
    ].reshape(MATRIX_SHAPE)
    matrix_b = workspace_cpu[
        SYNC_WORKSPACE_ELEMENTS + matrix_elements :
    ].reshape(MATRIX_SHAPE)

    assert torch.equal(matrix_a, torch.ones_like(matrix_a)), "logical core 0 did not write A to ones"
    assert torch.equal(matrix_b, matrix_a), "logical core 1 did not copy A to B after IBWait"
