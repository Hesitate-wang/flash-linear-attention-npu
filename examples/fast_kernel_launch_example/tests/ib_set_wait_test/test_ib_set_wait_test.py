#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Tianjin University, Ltd.

import pytest
import torch
import torch_npu

import ascend_ops


LOGICAL_BLOCK_NUM = 2
SYNC_WORDS_PER_BLOCK = 8  # One 32-byte IB slot per logical block.
SYNC_WORKSPACE_ELEMENTS = LOGICAL_BLOCK_NUM * SYNC_WORDS_PER_BLOCK
MATRIX_SHAPE = (8, 8)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_ib_set_wait_synchronizes_two_aic_cores():
    matrix_elements = MATRIX_SHAPE[0] * MATRIX_SHAPE[1]

    # Zeroed GM workspace: [IB sync area | producer matrix A | consumer matrix B].
    # AIC core 0 writes A to one, signals, and AIC core 1 copies A to B after
    # waiting. Keeping both matrices initially zero prevents a false positive.
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

    assert torch.equal(matrix_a, torch.ones_like(matrix_a)), "AIC core 0 corrupted matrix A"
    assert torch.equal(matrix_b, matrix_a), "AIC core 1 did not copy A to B after IBWait"
