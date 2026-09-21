# ChunkFwdHOFused 设计

本文定义当前实现的 Host tiling、kernel、workspace 和同步约束。公开接口与 TilingData 字段顺序以 `api.md` 和 `chunk_fwd_h_o_fused_struct.h` 为准。

## 1. 架构

定义任务数 `T = shapeBatch * vNumHead`、活动 MIX core 数 `A = min(T, physicalCoreNum)`、producer 数 `R = floor(A / 2)`。A2 (`DAV_2201`) 与 A5 (`DAV_3510`) 使用相同的 core 分区：前 `R` 个 core 是 H producer，从 `consumerCoreBase=R` 开始是 O consumer。每个配对 core 通过两个 ping-pong stream 处理两个 `(batch, value-head)` 任务；`T > A` 时以 `2R` 为步长继续处理后续波次。`A` 为奇数时最后一个未配对 consumer core 保持空闲。

| SoC | Tiling key | blockDim | H/O 调度 | 同步 |
| --- | --- | --- | --- | --- |
| Atlas A2 | V128/V256 | `min(T, physicalCoreNum)` | 双任务 producer H + consumer O | 每 chunk `IBSet/IBWait` |
| Ascend 950 A5 | V128/V256 | `min(T, physicalCoreNum)` | 双任务 producer H + consumer O | 每 chunk `IBSet/IBWait` |

入口先按编译目标选择 A2 或 A5 架构实现，再通过 key 1/2 选择 V128/V256 的已注册 kernel object；key 不表示 exp/exp2。`useExp2` 只在已选定的架构实现内部选择指数模式，A5 在 `useExp2=false` 时仍使用 `arch35` FwdO；dtype、layout 和 head 维度同样在 object 内继续选择模板路径。

## 2. 执行流程

H 阶段先计算 `h` 与 `v_new`，O 阶段消费它们：

```text
v_new = u - w @ h_before
h_after = h_before * gate_end + k^T @ (v_new * gate_delta)
o = scale * (q @ h_before + masked(q @ k^T) @ v_new)
```

`GetMixedCoreIdx() < producerCoreNum` 时只运行 H，否则只运行 O。A2 与 A5 都不得采用“全核 H、全局同步、全核 O”的顺序执行模型。

H producer `p in [0,R)` 负责 `2p`、`2p+1` 以及后续相隔 `2R` 的 `(batch, value-head)` 任务，写入按完整任务索引编址的 `handoffH`、`handoffV` 和可选 `final_state`。每个 producer AIV 使用 4 个 ready event：`HReady[0..1]` 和 `VReady[0..1]` 分别隔离两个 task lane。首 chunk 的 `HReady` 在初始状态写完后发布；后续 chunk 的 `HReady` 在前一 chunk 的 Vec2 写完 `h_after` 后发布；`VReady` 在当前 chunk 的 Vec1 写完 `v_new` 后发布。同一 task 的所有 chunk 分别复用同一个 H/V ready slot；`IBSet` 仅在对应 GM event slot 为 0 时将其置 1，因此 H 发布下一代同类事件前会自然等待 O 消费前一代事件。

O consumer `R+p` 消费 producer `p` 发布的同一任务序列。`QK` 不依赖 H 阶段；O AIV 在 `QK` 完成且 `HReady` 到达后反向放行 O AIC，使 `masked(QK)` 与 `Q * gate @ h_before` 并行推进。只有 `masked(QK) @ v_new` 需要继续等待 `VReady`。每次 `IBWait` 消费事件后将对应 GM slot 清零，从而直接向下一次 `IBSet` 归还该 slot，不需要额外的 O 到 H ACK。handoff 数据按任务/chunk 独立编址，不依赖 event slot 保护数据覆盖。A2 按 wave 顺序消费每个任务的全部 chunk；A5 自然指数路径按两个 stream 的同序 chunk 交错消费，以匹配各自 H scheduler 的发布次序。固定长度自然指数场景下 H/O 阶段边界不允许插入 `SyncAll`。

当前 arch35 exp2 专用 O 的底层 cube/vector 类尚未暴露 handoff 的 IBWait 接口，因此实现阶段暂时在 exp2 分支保留 H/O 边界同步作为兼容保护；该分支在补齐专用 O 的 IBWait 后必须删除该同步，恢复与自然指数路径相同的 chunk pipeline。

## 3. Host tiling 约束

Host 设置 `activeCoreNum=min(T, physicalCoreNum)`、`producerCoreNum=floor(activeCoreNum/2)`、`consumerCoreBase=producerCoreNum` 和 `scheduleMode=1`。当 `T > physicalCoreNum` 时保留全物理 core 的多波次调度路径；当前实现仅接受定长输入，且至少需要两个活动 AIC core。提供 `cuSeqlens` 或 `chunkIndices` 时返回失败。

## 4. TilingData ABI

以下字段顺序不可改变：

```cpp
BEGIN_TILING_DATA_DEF(ChunkFwdHOFusedTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, seqlen);
TILING_DATA_FIELD_DEF(int64_t, kNumHead);
TILING_DATA_FIELD_DEF(int64_t, vNumHead);
TILING_DATA_FIELD_DEF(int64_t, kHeadDim);
TILING_DATA_FIELD_DEF(int64_t, vHeadDim);
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
TILING_DATA_FIELD_DEF(bool, useInitialState);
TILING_DATA_FIELD_DEF(bool, storeFinalState);
TILING_DATA_FIELD_DEF(int64_t, dataType);
TILING_DATA_FIELD_DEF(int64_t, gDataType);
TILING_DATA_FIELD_DEF(int64_t, stateDataType);
TILING_DATA_FIELD_DEF(int64_t, isVariedLen);
TILING_DATA_FIELD_DEF(int64_t, shapeBatch);
TILING_DATA_FIELD_DEF(int64_t, tokenBatch);
TILING_DATA_FIELD_DEF(bool, useG);
TILING_DATA_FIELD_DEF(bool, useGk);
TILING_DATA_FIELD_DEF(int64_t, vWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, vUpdateWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, kDecayWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, hWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, numSeqWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, numChunksWorkspaceOffset);
TILING_DATA_FIELD_DEF(bool, useExp2);
TILING_DATA_FIELD_DEF(int64_t, outputLayout);
TILING_DATA_FIELD_DEF(float, scale);
TILING_DATA_FIELD_DEF(int64_t, chunkNum);
TILING_DATA_FIELD_DEF(int64_t, numChunksPerBatch);
TILING_DATA_FIELD_DEF(int64_t, hvPerHk);
TILING_DATA_FIELD_DEF(int64_t, taskGroupSize);
TILING_DATA_FIELD_DEF(int64_t, producerCoreNum);
TILING_DATA_FIELD_DEF(int64_t, consumerCoreBase);
TILING_DATA_FIELD_DEF(int64_t, activeCoreNum);
TILING_DATA_FIELD_DEF(int64_t, handoffHWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, handoffVWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, pipelineSyncWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, pipelineEventCount);
TILING_DATA_FIELD_DEF(int64_t, oVWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, oHWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, oAttnWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, oAfterMaskWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, oMaskWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, oAPrimeWorkspaceOffset);
END_TILING_DATA_DEF;
```

H 只解释从 `batch` 到 `numChunksWorkspaceOffset` 的前缀；O 使用显式 stage projection，不得把完整 TilingData 强转为另一种布局。`offsetof(useExp2)` 必须继续与 H stage struct 大小一致。

## 5. Workspace

所有 offset 相对 `AscendC::GetUserWorkspace(workspace)`，每段按 512 字节对齐。`E` 为输入元素字节数，`F=4`，`S=2`：

| 区域 | 大小 |
| --- | --- |
| handoff H | `T * NC * K * V * E` |
| handoff V | `T * tokens * V * E` |
| pipeline sync | `A * 2 * 4 * 8 * 4`（每个 task lane 各一个 HReady 和 VReady） |
| H v/v-update | 各 `R * C * V * F * S` |
| H k-decay（有 GK） | `R * C * K * F * S` |
| H state | `R * K * V * F * S` |
| O v/h | 各 `(A-R) * C * V * F * S` |
| O attention/after-mask | 各 `(A-R) * C * C * F * S` |
| O mask | `C * C` |
| A5 exp2 A-prime | `(A-R) * CHUNK_FWD_O_APRIME_WORKSPACE_BYTES` |

O 和 A-prime 临时区按 consumer-local index 编址；H 临时区按 producer-local index 编址。`pipelineSyncWorkspaceOffset` 必须指向实际分配的同步区，不能置零。

## 6. 接口与验证

公开 ACLNN/L0 输入、输出、属性顺序和 kernel ABI 顺序保持现有实现不变。修改后至少静态验证：TilingData 字段顺序/类型/大小一致；A2/A5 key 1/2 均有对应 entry；`activeCoreNum=min(T, physicalCoreNum)` 且 `producerCoreNum=floor(activeCoreNum/2)`；handoff workspace 使用任务数 `T`，core-local workspace 使用对应 producer/consumer 数；A5 自然指数路径无 H/O 边界 `SyncAll` 并逐 chunk 执行 IBWait。
