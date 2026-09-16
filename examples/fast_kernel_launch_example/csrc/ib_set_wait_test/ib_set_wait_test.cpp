/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "acl/acl.h"
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "kernel_operator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace ascend_ops {
namespace IBSetWaitTest {

constexpr int64_t SYNC_WORKSPACE_ELEMENTS = 8;
constexpr int64_t SYNC_WORKSPACE_BYTES = SYNC_WORKSPACE_ELEMENTS * sizeof(int32_t);
constexpr int64_t LOGICAL_BLOCK_NUM = 2;
constexpr int64_t MAX_PAYLOAD_ELEMENTS = 8192;

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("ib_set_wait_test(Tensor(a!) workspace, int matrix_elements) -> Tensor(a!)");
}

void CheckArguments(const torch::Tensor &workspace, int64_t matrixElements)
{
    TORCH_CHECK(workspace.scalar_type() == torch::kInt32, "workspace must have dtype torch.int32.");
    TORCH_CHECK(workspace.dim() == 1, "workspace must be a one-dimensional tensor.");
    TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous.");
    TORCH_CHECK(matrixElements > 0, "matrix_elements must be positive.");
    TORCH_CHECK(matrixElements % 8 == 0, "matrix_elements must be a multiple of 8 (32-byte aligned).");
    TORCH_CHECK(matrixElements <= MAX_PAYLOAD_ELEMENTS, "matrix_elements exceeds the test payload limit.");
    TORCH_CHECK(
        workspace.numel() == SYNC_WORKSPACE_ELEMENTS + 2 * matrixElements,
        "workspace must contain 8 synchronization elements followed by matrices A and B.");
}

torch::Tensor ib_set_wait_test_meta(const torch::Tensor &workspace, int64_t matrixElements)
{
    CheckArguments(workspace, matrixElements);
    return workspace;
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("ib_set_wait_test", ib_set_wait_test_meta);
}

extern "C" __global__ __aicore__ void ib_set_wait_test_kernel(GM_ADDR workspace, uint32_t matrixElements)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    AscendC::TPipe pipe;
    // IBSet/IBWait require a 32-byte local tensor in a Vector position even
    // though this experiment launches AICs only. Whether this is accepted and
    // operational is exactly what the test is intended to determine.
    AscendC::TBuf<AscendC::TPosition::VECCALC> syncBuffer;
    pipe.InitBuffer(syncBuffer, SYNC_WORKSPACE_BYTES);
    auto syncLocal = syncBuffer.Get<int32_t>();

    auto *workspacePtr = reinterpret_cast<__gm__ int32_t *>(workspace);
    AscendC::GlobalTensor<int32_t> syncGm;
    AscendC::GlobalTensor<int32_t> matrixAGm;
    AscendC::GlobalTensor<int32_t> matrixBGm;
    syncGm.SetGlobalBuffer(workspacePtr, SYNC_WORKSPACE_ELEMENTS);
    matrixAGm.SetGlobalBuffer(workspacePtr + SYNC_WORKSPACE_ELEMENTS, matrixElements);
    matrixBGm.SetGlobalBuffer(workspacePtr + SYNC_WORKSPACE_ELEMENTS + matrixElements, matrixElements);

    const uint32_t blockIdx = AscendC::GetBlockIdx();
    if (blockIdx == 0) {
        // Produce the payload entirely on AIC so a successful consumer result
        // depends on the AIC-to-AIC synchronization rather than host ordering.
        for (uint32_t index = 0; index < matrixElements; ++index) {
            matrixAGm.SetValue(index, static_cast<int32_t>(1));
        }
        AscendC::IBSet<false>(syncGm, syncLocal, 0, 0);
        return;
    }

    if (blockIdx == 1) {
        AscendC::IBWait<false>(syncGm, syncLocal, 0, 0);
        for (uint32_t index = 0; index < matrixElements; ++index) {
            matrixBGm.SetValue(index, matrixAGm.GetValue(index));
        }
    }
}

torch::Tensor ib_set_wait_test_npu(const torch::Tensor &workspace, int64_t matrixElements)
{
    CheckArguments(workspace, matrixElements);
    TORCH_CHECK(workspace.device().type() == c10::DeviceType::PrivateUse1, "workspace must be on an NPU device.");

    const c10::OptionalDeviceGuard guard(workspace.device());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto workspacePtr = (GM_ADDR)workspace.data_ptr();
    const uint32_t matrixElementsU32 = static_cast<uint32_t>(matrixElements);
    auto aclCall = [=]() -> int {
        ib_set_wait_test_kernel<<<LOGICAL_BLOCK_NUM, nullptr, stream>>>(workspacePtr, matrixElementsU32);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("IBSetWaitTest", aclCall);
    return workspace;
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("ib_set_wait_test", ib_set_wait_test_npu);
}

} // namespace IBSetWaitTest
} // namespace ascend_ops
