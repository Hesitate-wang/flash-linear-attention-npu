# 开发阶段验证记录

## 设计追踪

| 设计项 | 实现位置 |
| --- | --- |
| `P` 个生产者核和 `P` 个消费者核 | Host tiling 中的 `producerCoreNum`、`consumerCoreBase`、`activeCoreNum`，以及算子内 H/O 调度器 |
| 完整 H/`v_new` 交接区 | Host 的 `FillWorkspace`；kernel 的 `handoffHWorkspaceOffset` 和 `handoffVWorkspaceOffset` |
| 启动事件初始化 | 融合 kernel 入口中的 `InitializePipelineSync` |
| 逐 chunk 的 H 到 O 发布 | 算子内 H 的 `SignalChunkReady`；算子内 O AIV 的 `WaitProducerSliceReady`；反向 `cube1Done` 确认用于约束 O AIC |
| A5 的 H 到 O 阶段边界 | arch35 H 清空事件；`RunChunkFwdHOFusedA5` 在执行 arch35 O 前调用 `SyncAll<false>()` |
| A5 临时区所有权 | Host 的 `FillWorkspaceA5`；算子内 H 偏移和 `oAPrimeWorkspaceOffset` |
| 不依赖同级算子的私有实现 | 所有 H/O kernel、调度器和收尾文件都位于当前算子目录内 |

## 已完成的静态检查

- Host 和 kernel 的融合 tiling 镜像使用相同字段顺序；Host tiling 保留与算子内
  H 完全一致的结构前缀，并检查完整序列化结构的大小。宏生成的 Host TilingData
  采用本地构造，在所有字段和 workspace offset 填充完成后显式 `SaveToBuffer` 到
  raw tiling buffer，并设置实际数据长度；不再将 raw buffer 解释为未构造的
  TilingData 管理对象，因此首个 `set_batch` 不会访问空的内部存储。
- Workspace 偏移均相对于用户 workspace，所有区域都按 512 字节对齐；返回的
  workspace 大小只累加一次平台系统 workspace。
- A2 的 `blockDim` 为 `2 * B * HV`，并受
  `2 * B * HV < physical AIC cores` 约束；A5 使用全部物理 AIC。
- 已接受 Atlas A2 的定长自然指数路径和 Ascend 950 的定长 exp2 路径。A5
  约束为：数据使用 BF16，门控使用 BF16/FP32，chunk 为 64，K=V=128，且
  HV/HK 位于 `[1,4]`。变长输入仍在 tiling 阶段失败。
- 跨算子 include 扫描结果为空：融合算子目录树未引用同级算子路径或
  `internal` 实现路径。
- 所有使用双引号的本地 include 都能从当前文件目录、`op_kernel` 或
  `op_kernel/arch35` 解析。此前缺失的 arch35
  `block_epilogue_gdn_fwdh_regbase.hpp` 现已位于当前算子目录，并与独立 FwdH
  实现一致；若该文件或 kernel 入口源文件缺失，CMake 会在配置阶段失败。
- 当前算子的 O 阶段结构头文件同时包含紧凑的 A2 投影，以及复制后的 Ascend
  950 O 实现所使用的完整 `ChunkFwdOTilingData` 投影。分发前，`FillOTiling`
  会初始化 Ascend 950 投影的全部字段。
- ACLNN 前置检查在 Atlas A2 上接受自然指数和 BNSD/NTD，在 Ascend 950 上
  接受 exp2 和 BSND/TND。L0 下发失败时保留原始状态码，不再统一改写为
  `ACLNN_ERR_PARAM_NULLPTR`（161001）。
- 按接口要求，IB 本地张量保留在 SIMD 侧。在 MIX 模式下，IB 索引空间为
  `2 * blockDim`；O AIC 读取完整 H/V tile 前，通过反向 `cube1Done` 的代次
  聚合两个配对 AIV 的等待结果。
- 对 `(B,HV,NC)=(1,8,3)` 和 `(2,4,3)` 进行的静态任务映射模拟表明，生产者
  `p` 和消费者 `P+p` 会枚举完全相同的 `(b,hv,chunk)` 元组，并使用 24 个
  MIX 核中的 16 个。
- OpDef 已注册 Ascend 950，并选择 `__CCE_AICORE__ == 310` 实现。该路径先
  执行当前算子目录内的 arch35 H，跨越一次全核阶段边界，再执行当前算子目录
  内的 arch35 O。H/`v_new` 交接区、H 临时区和 O A-prime 临时区互不重叠。
- `git diff --check` 已通过，仅存在行尾转换警告。

## 环境限制与待补充的设备证据

当前 Windows 工作区没有已配置的 CANN 环境、NPU 设备或可用的 Python 运行
环境，因此无法在这里完成算子构建和运行时精度测试。仍须在目标环境完成：

1. 分别为 `ascend910b` 和 `ascend950` 构建并安装单算子包。
2. 运行至少包含三个 chunk、V=128 的最小定长用例，覆盖有无初始状态、最终
   状态和 `gk` 的组合。
3. 运行 V=256 和尾块场景。
4. 将 `o` 和可选最终状态与 CPU/参考组合实现进行比较。
5. 采集执行 trace，证明生产者/消费者的 chunk 顺序和 IB 事件复用正确；在考虑
   缓冲方案前，先对交接区域的 L2 命中率进行 profiling。
6. 在 Ascend 950 上启用 `--use-exp2`，针对 BF16/FP32 门控、有无初始状态和
   最终状态的组合，比较融合实现与独立组合实现的 BSND 输出。
