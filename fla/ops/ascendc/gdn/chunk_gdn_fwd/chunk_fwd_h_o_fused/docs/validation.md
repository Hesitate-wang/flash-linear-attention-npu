# 开发阶段验证记录

## 设计追踪

| 设计项 | 实现位置 |
| --- | --- |
| `R=ceil(T/2)` 对生产者/消费者核 | Host tiling 检查 `R<=floor(P/2)`，并设置相等的 producer/consumer 数、`consumerCoreBase=R` 和 `activeCoreNum=2R` |
| 完整 H/`v_new` 交接区 | Host 的 `FillWorkspace`；kernel 的 `handoffHWorkspaceOffset` 和 `handoffVWorkspaceOffset` |
| 启动事件初始化 | A2/A5 统一入口 `RunChunkFwdHOFused` 在 H/O 分流前调用各自的 `InitializePipelineSync` |
| 逐 chunk 的 H 到 O 发布 | H 的 `SignalProducerSliceReady` 按 task lane 分别发布 `HReady` 和 `VReady`，共 4 个 event；同一 task 的 chunk 串行复用对应 slot，O 的 `WaitProducerSliceReady` 消费并清零；`cube1Done` 在 `HReady` 后放行 O AIC 的 `QH_old` |
| A5 的 H 到 O 阶段边界 | 自然指数路径按 chunk 执行 `IBSet/IBWait`；exp2 专用 O 暂以 `SyncAll<false>()` 保护交接 |
| A5 临时区所有权 | Host 的 `FillWorkspaceA5`；自然指数通用 O offset 或 exp2 的 `oAPrimeWorkspaceOffset` |
| 不依赖同级算子的私有实现 | 所有 H/O kernel、调度器和收尾文件都位于当前算子目录内 |
| FwdO 架构隔离 | A2 实现在 `op_kernel/gemm/kernel`；A5 kernel、配套 epilogue、exp2 tiling/constants 和 UB layout 全部位于 `op_kernel/arch35` |

## 已完成的静态检查

- Host 和 kernel 的融合 tiling 镜像使用相同字段顺序；Host tiling 保留与算子内
  H 完全一致的结构前缀，并检查完整序列化结构的大小。宏生成的 Host TilingData
  采用本地构造，在所有字段和 workspace offset 填充完成后显式 `SaveToBuffer` 到
  raw tiling buffer，并设置实际数据长度；不再将 raw buffer 解释为未构造的
  TilingData 管理对象，因此首个 `set_batch` 不会访问空的内部存储。
- Workspace 偏移均相对于用户 workspace，所有区域都按 512 字节对齐；返回的
  workspace 大小只累加一次平台系统 workspace。
- A2 和 A5 的当前路径均要求 `ceil(B * HV / 2) <= floor(physical AIC cores / 2)`，
  `blockDim=2*ceil(B*HV/2)`。producer 与 consumer 数量相等且一一配对；单任务
  启动一对 core，奇数任务的最后一个 task lane 允许为空。核数不足的 saturated-core
  路径尚未实现，tiling 会明确返回失败。
- 已接受 Atlas A2 和 Ascend 950 的定长自然指数路径。两者均支持 FP16/BF16、
  FP32 或输入类型门控、chunk 64/128、K=128、V=128/256 及 BNSD/NTD。
  Ascend 950 的 exp2 专用路径仍限制为 BF16、BF16/FP32 门控、chunk 64、
  K=V=128、HV/HK 位于 `[1,4]` 及 BSND/TND。变长输入仍在 tiling 阶段失败。
- 跨算子 include 扫描结果为空：融合算子目录树未引用同级算子路径或
  `internal` 实现路径。
- 所有使用双引号的本地 include 都能从当前文件目录、`op_kernel`、
  `op_kernel/arch35` 或公共 kernel include 路径解析。此前缺失的 arch35
  `block_epilogue_gdn_fwdh_regbase.hpp` 现已位于当前算子目录，并与独立 FwdH
  实现一致；若该文件或 kernel 入口源文件缺失，CMake 会在配置阶段失败。
- 当前算子的根目录 O 阶段结构头只包含自然指数路径共用的 O 投影；Ascend 950
  exp2 专用 O 使用的完整 `ChunkFwdOTilingData` 已拆至
  `op_kernel/arch35/chunk_fwd_o_a5_struct.h`。分发前，
  `FillNaturalOTiling` 或 `FillOptimizedOTiling` 会初始化对应投影的全部字段。
- ACLNN 前置检查在 Atlas A2 和 Ascend 950 上接受自然指数与 BNSD/NTD，
  在 Ascend 950 上另接受 exp2 与 BSND/TND。L0 下发失败时保留原始状态码，不再统一改写为
  `ACLNN_ERR_PARAM_NULLPTR`（161001）。
- 按接口要求，IB 本地张量保留在 SIMD 侧。在 MIX 模式下，IB 索引空间为
  `2 * blockDim`；O AIC 读取完整 `H_old` tile 前，通过反向 `cube1Done` 的代次
  聚合两个配对 AIV 的 `HReady` 等待结果；`V_new` 由后续 `VReady` 单独约束。
- A2 和 A5 均在 H/O 分流前由每个逻辑 AIV 清零自身的全部 IB event slot，
  随后由全部活动 AIC/AIV 执行一次启动 rendezvous；A5 不依赖未初始化的
  ACLNN user workspace 作为 `IBSet/IBWait` 同步状态。Host 同步区容量按
  `activeCoreNum * AIV_PER_MIXED_CORE * eventCount * wordsPerEvent` 分配。
- 自然指数路径为每个 producer AIV 分配 4 个 ready event，每个 task lane 分别拥有
  `HReady` 和 `VReady`。首 chunk 的 H 来自初始化，后续 `HReady(i+1)` 由 Vec2(i)
  发布，`VReady(i)` 由 Vec1(i) 发布；同一 task 的所有 chunk 串行复用对应 slot。
  `IBSet` 只在 GM slot 为 0 时置 1，`IBWait` 消费后将 slot 清零，因此下一代同类
  事件的发布由接口自身反压。
- 已对 `(T,P,A,NC)=(1,32,2,3)`、`(3,32,4,4)` 和 `(9,32,10,5)` 执行
  静态任务/event 映射检查。生产者 `p` 和消费者 `R+p` 对每个
  `(task,chunk)` 得到相同的 producer AIV、task lane、`0..1` HReady 和 `2..3` VReady，
  并完整覆盖全部任务。该检查覆盖单任务、奇数任务以及 A5 最后一对只有 lane0
  有效的逐 chunk 推进；另检查 `(T,P)=(33,32)` 和 `(5,3)` 被 Host 拒绝。
- OpDef 已注册 Ascend 950，并选择 `__CCE_AICORE__ == 310` 实现。该路径由
  arch35 H producer 和 O consumer 并行推进；自然指数使用逐 chunk IB 交接，
  exp2 专用 O 暂时保留全核交接屏障。设备侧按 dtype 和 V 维度显式选择 H 模板，
  A5 入口为 Host 下发的 key 1/2 分别声明同为 MIX 1:2 的 kernel task，避免
  object 查找落到不存在的 default key；各路径所需 workspace 区域互不重叠。
- `git diff --check` 已通过，仅存在行尾转换警告。

## 架构文件组织

- `chunk_fwd_h_o_fused_arch.h` 是唯一读取 `__CCE_AICORE__` 的设备架构选择点，
  生成 `CHUNK_FWD_HO_ARCH_A2` 或 `CHUNK_FWD_HO_ARCH35`；统一 dispatcher 只包含
  对应架构入口。
- A2/A5 入口及两套 H/O kernel 头均校验架构宏，错误架构的头文件被包含时会在
  预处理阶段失败，避免同一编译单元混入另一架构实现。
- 相对 include 静态扫描通过；移动后的 A5 constants/tiling 头均从 `arch35/`
  解析，根目录不存在旧的 A5 constants 文件。

## A5 IB 通信 UB 隔离修复

- A5 UB 尾部 `[248 KiB, 256 KiB)` 已声明为 IB 通信专用区；`IBSet/IBWait`
  的 32-byte local tensor 固定使用 `248 KiB`，不再使用会落入 O H-pong
  Fixpipe 槽的 `188 KiB`。
- H/O kernel 共用同一布局常量；QK-mask 和 output epilogue 的编译期上界检查
  已改为通信区起点，数据 tensor 若越过 `248 KiB` 将编译失败。
- 静态检查确认通信区按 32 bytes 对齐，终点为 `256 KiB`，IB local tensor
  终点为 `248 KiB + 32 bytes`；arch35 下不再存在旧的
  `HO_PIPELINE_SYNC_UB_OFFSET` 或 `188 * 1024`。
- `git diff --check` 通过。当前环境没有 CANN/NPU，尚未完成设备编译和运行验证；
  上板时需关闭调试 `PRINTF`，覆盖多 chunk、V=128/256 的重复压力测试并确认
  不再出现 IB 等待超时。

## IB 发布屏障精简

- A2/A5 的 `SignalProducerSliceReadyAfterMte3` 包装函数已删除，四个 V/H ready
  发布点直接调用 `SignalProducerSliceReady`。
- 初始 H ready 发布前的外部 `PipeBarrier<PIPE_MTE3>` 也已删除；数据可见性由
  `IBSet/IBWait` 内部搬入、搬出前后的 `PipeBarrier<Pipe_all>` 保证。

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
7. 在 Ascend 950 上以自然指数分别运行 FP16/BF16、V=128/256、chunk=64/128、
   BNSD/NTD，并比较融合实现、独立组合实现和 CPU 标杆。
