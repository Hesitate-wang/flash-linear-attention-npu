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
- `chunk_size: int64`，当前支持 64 或 128
- `scale: double`
- `use_exp2: bool`；Atlas A2 要求为 `false`，Ascend 950 要求为 `true`
- `state_v_first: bool`
- `output_layout: string`，可取 `BNSD`、`BSND`、`TND` 或 `NTD`

自然指数路径支持 `BNSD/NTD`，其中 `NTD` 要求物理批次 `B=1`。Ascend 950
的 exp2 路径支持 `BSND/TND`。

## 输出

1. `o`：必选；形状由 `output_layout` 决定，与 `ChunkFwdO` 保持一致。
2. `final_state`：可选，基础形状为 `[N, HV, K, V]`；当
   `state_v_first=true` 时交换最后两个维度。仅当
   `output_final_state=true` 时返回该输出。

Python 适配层返回 `(o, final_state)`；未请求最终状态时，`final_state` 为
`None`。L0 实现始终以 `[K, V]` 状态布局调用融合 kernel；所需的状态输入、
输出转置由 ACLNN 层完成。

## 当前执行约束

功能 kernel 已为 Atlas A2（`ascend910b` 和 `ascend910_93`）及 Ascend 950
注册。在 A2 上，令 `P = B * HV`，tiling 要求 `2 * P` 严格小于可用 AIC
核数。不满足该条件的形状会直接失败，不会回退到串行融合调度。

Ascend 950 支持定长 exp2 路径，其约束为：q/k/w/u 使用 BF16，门控张量使用
BF16 或 FP32，`chunk_size=64`，`K=V=128`，且 `HV/HK` 位于 `[1,4]`。
输出布局为 BSND 或 TND。H 阶段保持独立 FwdH 的自然指数语义；`use_exp2`
只选择 FwdO 的指数计算和布局路径。
