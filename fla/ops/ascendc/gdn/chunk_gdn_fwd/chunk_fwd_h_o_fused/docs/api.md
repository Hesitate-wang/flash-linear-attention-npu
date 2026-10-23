# ChunkFwdHOFused 接口说明

## 算子语义

`ChunkFwdHOFused` 先执行 `ChunkGatedDeltaRuleFwdH` 的状态递推，再将得到的
`h` 和 `v_new` 直接传给 `ChunkFwdO` 计算。`h` 和 `v_new` 仅在融合算子内部
流转，不作为算子输出。

## 输入

注册接口和 AICore kernel 的张量输入统一采用以下顺序：

1. `k`：必选，形状为 `[B, HK, T, K]`，数据类型为 FP16 或 BF16。
2. `q`：必选，形状为 `[B, HK, T, K]`，形状和数据类型均与 `k` 相同。
3. `w`：必选，形状为 `[B, HV, T, K]`，数据类型与 `k` 相同。
4. `u`：必选，形状为 `[B, HV, T, V]`，数据类型与 `k` 相同。
5. `g`：必选，形状为 `[B, HV, T]`，数据类型为 FP16、BF16 或 FP32。O 阶段
   需要使用该标量门控，因此此输入不能为空。
6. `gk`：可选，形状为 `[B, HV, T, K]`，数据类型与 `g` 相同。
7. `initial_state`：可选；当 `state_v_first=false` 时，形状为
   `[N, HV, K, V]`；当 `state_v_first=true` 时，形状为 `[N, HV, V, K]`。
8. `cu_seqlens`：可选的 INT64 数组。
9. `chunk_indices`：可选的 INT64 一维数组，内容为展平后的
   `(sequence, chunk)` 索引对。

`cu_seqlens` 和 `chunk_indices` 必须同时提供或同时不提供。

当前设备侧实现仅支持定长输入，因此目前这两个可选索引输入都必须为空。

## 属性

- `output_final_state: bool`
- `chunk_size: int64`，当前 mode1 仅支持 32
- `scale: double`
- `use_exp2: bool`；当前 mode1 仅支持 `false`
- `state_v_first: bool`
- `output_layout: string`，可取 `BNSD`、`BSND`、`TND` 或 `NTD`

当前 mode1 的自然指数路径支持 `BNSD/NTD`，其中 `NTD` 要求物理批次 `B=1`。
`BSND/TND` 与 exp2 路径不在当前支持范围内。

## 输出

1. `o`：必选；形状由 `output_layout` 决定，与 `ChunkFwdO` 保持一致。
2. `final_state`：可选，基础形状为 `[N, HV, K, V]`；当
   `state_v_first=true` 时交换最后两个维度。仅当
   `output_final_state=true` 时返回该输出。

Python 适配层返回 `(o, final_state)`；未请求最终状态时，`final_state` 为
`None`。L0 实现始终以 `[K, V]` 状态布局调用融合 kernel；所需的状态输入、
输出转置由 ACLNN 层完成。

## 当前执行约束

当前仅支持定长 mode1：令 `P = B * HV`、`TOTAL_CORE` 为可用物理 AIC 核数，
要求 `P <= min(floor(TOTAL_CORE / 2), 21)`，并令 `blockDim = 2 * P`；21
是当前跨核事件分组 ABI 的任务对上限。超出该核数约束的形状直接失败，不回退
到其他调度模式。q/k/w/u 使用 FP16 或 BF16，
门控使用 FP32 或输入类型，`K=V=128`、`chunk_size=32`，使用自然指数，
输出布局为 BNSD 或 NTD。当前不支持 exp2、V=256 或 chunk 64/128。
