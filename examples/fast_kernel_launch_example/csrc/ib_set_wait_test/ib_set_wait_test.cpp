/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace ascend_ops {
namespace IBSetWaitTest {

constexpr int64_t SYNC_WORKSPACE_ELEMENTS = 8;
constexpr int64_t SYNC_WORKSPACE_BYTES = SYNC_WORKSPACE_ELEMENTS * sizeof(int32_t);
constexpr int64_t LOGICAL_BLOCK_NUM = 2;

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

__global__ __aicore__ void ib_set_wait_test_kernel(GM_ADDR workspace, uint32_t matrixElements)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> syncQueue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inputQueueA;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inputQueueB;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outputQueue;

    pipe.InitBuffer(syncQueue, 1, SYNC_WORKSPACE_BYTES);
    pipe.InitBuffer(inputQueueA, 1, matrixElements * sizeof(int32_t));
    pipe.InitBuffer(inputQueueB, 1, matrixElements * sizeof(int32_t));
    pipe.InitBuffer(outputQueue, 1, matrixElements * sizeof(int32_t));

    auto *workspacePtr = reinterpret_cast<__gm__ int32_t *>(workspace);
    AscendC::GlobalTensor<int32_t> syncGm;
    AscendC::GlobalTensor<int32_t> matrixAGm;
    AscendC::GlobalTensor<int32_t> matrixBGm;
    syncGm.SetGlobalBuffer(workspacePtr, SYNC_WORKSPACE_ELEMENTS);
    matrixAGm.SetGlobalBuffer(workspacePtr + SYNC_WORKSPACE_ELEMENTS, matrixElements);
    matrixBGm.SetGlobalBuffer(workspacePtr + SYNC_WORKSPACE_ELEMENTS + matrixElements, matrixElements);

    const uint32_t blockIdx = AscendC::GetBlockIdx();
    if (blockIdx == 0) {
        auto matrixA = outputQueue.AllocTensor<int32_t>();
        AscendC::Duplicate(matrixA, static_cast<int32_t>(1), matrixElements);
        outputQueue.EnQue(matrixA);
        matrixA = outputQueue.DeQue<int32_t>();
        AscendC::DataCopy(matrixAGm, matrixA, matrixElements);
        outputQueue.FreeTensor(matrixA);

        auto syncLocal = syncQueue.AllocTensor<int32_t>();
        AscendC::IBSet(syncGm, syncLocal, 0, 0);
        syncQueue.FreeTensor(syncLocal);
        return;
    }

    if (blockIdx == 1) {
        auto syncLocal = syncQueue.AllocTensor<int32_t>();
        AscendC::IBWait(syncGm, syncLocal, 0, 0);
        syncQueue.FreeTensor(syncLocal);

        auto matrixA = inputQueueA.AllocTensor<int32_t>();
        auto matrixB = inputQueueB.AllocTensor<int32_t>();
        AscendC::DataCopy(matrixA, matrixAGm, matrixElements);
        AscendC::DataCopy(matrixB, matrixBGm, matrixElements);
        inputQueueA.EnQue(matrixA);
        inputQueueB.EnQue(matrixB);

        matrixA = inputQueueA.DeQue<int32_t>();
        matrixB = inputQueueB.DeQue<int32_t>();
        auto result = outputQueue.AllocTensor<int32_t>();
        AscendC::Add(result, matrixA, matrixB, matrixElements);
        outputQueue.EnQue(result);
        inputQueueA.FreeTensor(matrixA);
        inputQueueB.FreeTensor(matrixB);

        result = outputQueue.DeQue<int32_t>();
        AscendC::DataCopy(matrixBGm, result, matrixElements);
        outputQueue.FreeTensor(result);
    }
}

torch::Tensor ib_set_wait_test_npu(const torch::Tensor &workspace, int64_t matrixElements)
{
    CheckArguments(workspace, matrixElements);
    TORCH_CHECK(workspace.device().type() == c10::DeviceType::PrivateUse1, "workspace must be on an NPU device.");

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(ascendcPlatform->GetCoreNumAiv() >= LOGICAL_BLOCK_NUM, "at least two AIV cores are required.");
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint64_t requiredUb = SYNC_WORKSPACE_BYTES + 3ULL * matrixElements * sizeof(int32_t);
    TORCH_CHECK(requiredUb <= ubSize, "matrix is too large for the kernel's UB queues.");

    const c10::OptionalDeviceGuard guard(workspace.device());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto workspacePtr = reinterpret_cast<GM_ADDR>(workspace.data_ptr());
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
