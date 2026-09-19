# ChunkFwdHOFused 方案设计

方案设计规则版本：`V2`

## 1. 目标与范围

设备侧实现面向 Atlas A2 的定长自然指数路径和 Ascend 950 的定长 exp2 输出
路径。两种实现都在一次 kernel 下发中融合状态递推与输出计算，并将 `h` 和
`v_new` 保留为内部数据。当前不支持变长输入，检测到变长输入时直接拒绝。

Ascend 950 使用算子目录内复制的既有 arch35 H 和 O 实现。编译期架构选择确保
A5 编译器不会包含 A2 的生产者/消费者实现。A5 依次执行 H 和 O，保持独立算子
组合时“H 阶段采用自然指数、O 阶段采用 exp2”的语义。

## 2. 核映射与执行顺序

令 `P = B * HV`，`NC = ceil(T / chunk_size)`。仅当
`2 * P < physical_aic_core_count` 时选择流水实现，并以 batch 调度模式启动
`blockDim = 2 * P` 个 MIX 核。

- MIX 核 `[0, P)` 是 H 生产者；生产者 `p = b * HV + hv` 按 chunk 升序负责
  一个 `(b, hv)` 对的全部 chunk。
- MIX 核 `[P, 2P)` 是 O 消费者；消费者 `P + p` 负责相同的 `(b, hv)` 对，
  同样按 chunk 升序遍历。
- 对应的 AIV0/AIV1 通道沿用既有的分行处理方式。IB 同步是 SIMD 侧接口，
  因此每个消费者 AIV 都在 MIX 逻辑索引空间内等待对应的生产者 AIV 索引。

所有活跃 AIV 首先将自己的 IB 事件槽清零。启动阶段执行一次 `SyncAll`，确保
核组分流前所有核都能观察到初始化结果。对于每个 chunk，H 通过
`IBSet<false>` 发布当前 `h` 和 `v_new` 中由各 AIV 完成的切片。O 的两个 AIV
完成对应的 `IBWait<false>` 后，反向复用已经消费完毕的 `cube1Done` 流水标志。
O 的 AIC 在首次把完整 `h`/`v_new` tile 从 GM 搬入 L1 前，等待这两个 AIV 的
聚合确认。`vec1Done` 独立保护随后使用的 `attnMask` 输入。IB 事件编号由
`chunk_index % 2` 选择。由于每个 chunk 都有独立的交接存储区，因此不需要
反向发送空闲信号。

在流水模式下，H 实现使用既有的逐 MIX 核 CrossCore 标志替代原先逐 wave 的
全局同步。因此，消费者核不需要进入 H；启动同步完成后，H 和 O 可以并发执行。

## 3. Workspace 所有权

所有序列化偏移都相对于 `AscendC::GetUserWorkspace(workspace)`；框架获得的
总分配量为 `system_workspace + user_workspace_bytes`。每个区域都从 512 字节
对齐地址开始。

| 区域 | 形状或容量 | 所有者与生命周期 |
| --- | --- | --- |
| H 交接区 | `[B, HV, NC, K, V]`，输入数据类型 | H 写入一个 chunk；配对的 O 在 IB 等待完成后读取；保留到 kernel 结束 |
| `v_new` 交接区 | `[B, HV, T, V]`，输入数据类型 | H 写入一个 chunk；配对的 O 在 IB 等待完成后读取；保留到 kernel 结束 |
| IB 事件区 | `2 events * 2 AIV/core * blockDim * 8` 个 int32 | 按 IB 接口要求，为每个 MIX 逻辑 AIV 和事件编号分配一个八字槽 |
| H 的 v/v-update 临时区 | 两块，各为 `P * 2 * C * V` 个 FP32 | 生产者 MIX 核私有，供 ping/pong 阶段使用 |
| H 的可选 k-decay 临时区 | `P * 2 * C * K` 个 FP32 | 仅存在 `gk` 时分配 |
| H 的 state-update 临时区 | `P * 2 * K * V` 个 FP32 | 生产者 MIX 核私有，供 ping/pong 阶段使用 |
| O 的 v/h 临时区 | 两块，各为 `P * 2 * C * V` 个 FP32 | 消费者 MIX 核私有，供 ping/pong 阶段使用 |
| O 的 attention/aftermask 临时区 | 两块，各为 `P * 2 * C * C` 个 FP32 | 消费者 MIX 核私有，供 ping/pong 阶段使用 |
| O 的因果掩码 | `C * C` 字节 | 构造完成后只读 |

`v_new` 分配区后紧接 IB 区域。H 和 O 都使用显式序列化偏移访问这两个区域，
不依赖指针之间的相对布局。首个 profiling 版本有意为所有 chunk 保留完整交接
存储。只有设备 profiling 表明 L2 命中率不理想时，才考虑改为双槽交接缓冲区。

在 A5 上，所有物理 MIX 核都参与两个阶段。其 workspace 包含完整的 H/`v_new`
交接区、每核 H ping/pong 临时区、可选 k-decay 临时区、每核 H update 临时区、
序列元数据，以及每个物理 AIC 对应的一块
`CHUNK_FWD_O_APRIME_WORKSPACE_BYTES` 区域。所有区域按 512 字节对齐，且互不
重叠。H kernel 返回前会清空本地事件；随后 AIC 和 AIV 执行
`SyncAll<false>()`，再由 O 读取交接张量。A5 使用
`blockDim=physical_aic_core_count`。

## 4. ABI 与实现归属

Host 侧宏类型和 kernel 侧普通镜像结构具有完全相同的字段顺序，tiling 会检查
两者大小。算子内 H 实现使用的结构前缀也通过 `offsetof` 在编译期检查。本算子
所需的 H/O 调度器、收尾模块和 kernel 全部归属当前算子的 `op_kernel` 目录树，
不包含或链接同级算子的文件及 `internal` 私有头文件。

A2 支持的 tiling key 为：V=128 时使用 `1`，V=256 时使用 `2`。A5 当前评审
范围仅包含 V=128，因此使用 key `1`。

## 5. 正确性与性能门禁

正确性要求如下：准确分配 `(b,hv,chunk)` 地址所有权；每个 H AIV 发布自己的
切片后再发送信号；O 的两个 AIV 等待均完成后，O AIC 才能执行依赖这些数据的
读取；最后一个 chunk 必须安全处理尾块；可选最终状态的行为必须与独立 H 算子
一致。验证至少覆盖两个 chunk，以实际执行事件和反向标志的复用。

精度通过后，profiling 需要记录交接区域的 L2 命中率和 GM 流量。只有这些证据
表明完整 chunk 布局未达到预期的 L2 复用效果时，才引入双缓冲。
