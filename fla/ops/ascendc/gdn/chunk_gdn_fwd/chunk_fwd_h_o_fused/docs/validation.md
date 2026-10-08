# 开发阶段验证记录

## 设计追踪

| 设计项 | 实现位置 |
| --- | --- |
| `T` 对生产者/消费者核 | Host tiling 检查 `T<=floor(P/2)`，并设置相等的 producer/consumer 数、`consumerCoreBase=T` 和 `activeCoreNum=2T` |
| 完整 H/`v_new` 交接区 | Host 的 `FillWorkspace`；kernel 的 `handoffHWorkspaceOffset` 和 `handoffVWorkspaceOffset` |
| 启动事件初始化 | A2/A5 统一入口不再初始化同步区或执行额外启动屏障；自然指数路径不访问 IB event table |
| 逐 chunk 的 H 到 O 发布 | mode `0x0` 的 `H_READY=12`、`V_READY=13` 按 chunk 执行全体 AIV `Set→Wait`；本核 `cube1Done/cube2Done/vec1Done` 继续使用 mode `0x2` |
| A5 的 H 到 O 阶段边界 | 自然指数路径按 chunk 执行 CrossCore H/V rendezvous；exp2 专用 O 仍使用既有 `SyncAll<false>()` 阶段边界 |
| A5 临时区所有权 | Host 的 `FillWorkspaceA5`；自然指数通用 O offset 或 exp2 的 `oAPrimeWorkspaceOffset` |
| 不依赖同级算子的私有实现 | 所有 H/O kernel、调度器和收尾文件都位于当前算子目录内 |
| FwdO 架构隔离 | A2 实现在 `op_kernel/gemm/kernel`；A5 kernel、配套 epilogue、exp2 tiling/constants 和 UB layout 全部位于 `op_kernel/arch35` |
| A5 自然指数 FwdO Vec1/HReady 重排 | qkmask 两条分支均先发射 gate/causal-mask Vector，再在 QK MTE2 前执行 H_READY Set/Wait 并立即 ACK；`chunk=128` 仅首段握手。待设备检查三 chunk 精度、IB/flag 代次以及相同条件下的 profiling 总耗时 |
| A5 FwdO V128 两段 Vec2 | 新增 `cube2Done[0/1]` 事件 8/9；Cube2 完成后先对整片 H FP32 乘 `exp(g)` 并驻留 UB，Cube3 完成后再 Add、Muls、Cast、写 O。V256 沿用旧融合路径。待设备验证 V128 chunk64/128、两种 gate dtype、三 chunk slot 回绕的精度、同步代次和总耗时 |

## 已完成的静态检查

- Host 和 kernel 的融合 tiling 镜像使用相同字段顺序；Host tiling 保留与算子内
  H 完全一致的结构前缀，并检查完整序列化结构的大小。宏生成的 Host TilingData
  采用本地构造，在所有字段和 workspace offset 填充完成后显式 `SaveToBuffer` 到
  raw tiling buffer，并设置实际数据长度；不再将 raw buffer 解释为未构造的
  TilingData 管理对象，因此首个 `set_batch` 不会访问空的内部存储。
- Workspace 偏移均相对于用户 workspace，所有区域都按 512 字节对齐；返回的
  workspace 大小只累加一次平台系统 workspace。
- A2 和 A5 的当前路径均要求 `B * HV <= floor(physical AIC cores / 2)`，
  `blockDim=2*B*HV`。producer 与 consumer 数量相等且一一配对；每个逻辑 MIX core
  只负责一个 `(batch, value-head)` 任务。核数不足的 saturated-core
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
  ACLNN user workspace 中的旧 IB 同步区仅作 ABI 保留，不再作为自然指数 H/O 同步状态。Host 同步区容量按
  `activeCoreNum * AIV_PER_MIXED_CORE * eventCount * wordsPerEvent` 分配。
- 自然指数路径为每个 producer AIV 分配 2 个 ready event，分别表示
  `HReady` 和 `VReady`。首 chunk 的 H 来自初始化，后续 `HReady(i+1)` 由 Vec2(i)
  发布，`VReady(i)` 由 Vec1(i) 发布；同一 task 的所有 chunk 串行复用对应 slot。
  mode-0 flag 12/13 按 H_READY→V_READY 代次由所有活动 AIV 执行 Set→Wait；旧 IB
  slot 不再参与自然指数路径。
- 已对 `(T,P,A,NC)=(1,32,2,3)`、`(3,32,6,4)` 和 `(9,32,18,5)` 执行
  静态任务/event 映射检查。生产者 `p` 和消费者 `T+p` 对每个
  `(task,chunk)` 得到相同的 producer AIV、固定 task lane 0、event 0 的 HReady 和
  event 1 的 VReady，并完整覆盖全部任务；另检查 `(T,P)=(17,32)` 和 `(5,3)`
  被 Host 拒绝。
- OpDef 已注册 Ascend 950，并选择 `__CCE_AICORE__ == 310` 实现。该路径由
  arch35 H producer 和 O consumer 并行推进；自然指数使用逐 chunk mode-0 H_READY/V_READY 交接，
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

- A5 UB 尾部 `[248 KiB, 256 KiB)` 作为旧 IB 协议的兼容保留区；自然指数路径不访问它。
  若后续启用旧协议，其 32-byte local tensor 固定使用 `248 KiB`，不再使用会落入 O H-pong
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

- HReady/VReady 的跨核发布和等待统一由 H/O 两侧 AIC 代理完成；AIV 仅保留
  cube/vector 本核事件，已删除旧的 ready signal 兼容对象及其传递链。
- 初始 H ready 发布前的外部 `PipeBarrier<PIPE_MTE3>` 也已删除；数据可见性由
  CrossCore Set/Wait 的 mode-0 语义与本核 MTE3/MTE2 pipe 顺序共同保证。
- A2/A5 的启动事件清零由 `Duplicate` 在 `PIPE_V` 生成本地零值，再通过成对的
  `SetFlag/WaitFlag<HardEvent::V_MTE3>` 交给 UB→GM `DataCopy`；初始化路径不再
  单独使用 `PipeBarrier<PIPE_V>()`。
- 所有初始化 `DataCopy` 提交后，再通过 `SetFlag/WaitFlag<HardEvent::MTE3_MTE2>`
  等待异步 MTE3 写回完成，之后进入自然指数 H/O 的 mode-0 chunk 握手；exp2 路径保留其阶段边界同步。
- FwdO chunk-pipeline 的计算 workspace 固定使用两个 stage；两个 stage 的
  `vec2Done` free token 均在启动时预置。H/`v_new` 交接地址仍按与 FwdH 相同的
  `(batch, head, chunk)` 映射计算，不引入 stage 偏移；只有 O 的 attention/HV
  计算 workspace 按 stage 分配独立物理区域。

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

## A5 H 主路径编译期分离（待设备验证）

- `GDNFwdHPath` 已分为 `DirectUb`、`BoundedGm` 和 `StandardGm`；路径选择只保留在
  A5 H 入口，Cube/Vector 主循环与两个 epilogue 使用编译期常量。
- Direct 路径限定 `chunkSize >= 16`，原有小于 16 token 的 Vector fallback 统一由
  `StandardGm` specialization 承担，避免 Direct specialization 内继续逐 task 判断。
- 静态检查确认不再把运行时 `useDirectFp32Ub` 传入 Vec1/Vec2 epilogue，TilingData
  和 workspace ABI 未修改，`git diff --check` 通过。
- 目标环境需要分别覆盖：Direct（V128/chunk64）、Standard（V256/chunk64、
  V128/chunk128、小于16 token）和 Bounded（非整除尾块或变长内部验证）路径；
  每条路径均需比较 CPU/compose 精度并确认 ready/free 事件无悬空。

## DirectUb Vec2 整 sub-block 搬运（待设备验证）

- 在 `DirectUb && !useGk` specialization 中，Vec2 已与独立
  FwdH preload 路径对齐：每个 AIV 对自己的 `64 x 128` state 半片执行一次 MTE2、
  一次完整 Vector 计算，并按既有输出语义统一写回；未请求 final state 时只有一次
  MTE3，不再按 16 行拆成四轮搬运。
- `DirectUb` 的入口条件已经保证定长、无尾、`K=V=128`；新增快路径不改变
  TilingData、workspace、Fixpipe ready/free flag 或 HReady/vec2Done 的代次。
- `useGk=true`、StandardGm 和 BoundedGm 仍使用原 16 行通用路径。final-state 分支
  使用相同的 64 行计算粒度，但保留 fused 原有的 FP32 驻留及中间/最终写回语义。
- 静态容量核算：每个 AIV 的完整半片分别需要 32 KiB FP32 calc、32 KiB FP32
  update 和 16 KiB BF16/FP16 输出，均落在原有 UB 区域内且不侵入 248 KiB IB 区。
- 目标环境需用默认三 chunk case 比较 CPU 与 composed 输出，并通过指令 trace
  确认每个有效 Vec2、每个 AIV 只有一次 state UB-to-GM MTE3 搬运。

## H/O ready 同步精简与静态配对检查（待设备验证）

- H_READY 保持每七个 MIX task pair 一组并映射到 mode-0 flag `0/2/4`；
  V_READY 改为每六个本侧相对 AIV block index 一组并映射到 flag `0..6`。
  两类 ready 都从 flag ID 0 开始映射。
- A5 两条 H kernel 删除未调用的发布包装函数，并复用同一个 ready signal
  构造函数生成下一 chunk 的 HReady；同步封装删除未调用的 `Participate()` 别名。
- 当前 `TASK_LANES_PER_CORE=1`。A5 V128 preload 的 H AIV 直接使用原始
  `blockIdx`，O AIV 使用 `blockIdx - producerCoreNum * subBlockNum`。静态枚举
  `1..21` 个 task pair，确认两侧 V_READY flag 完全一致，单个 V flag 最多
  12 次 Set；H_READY 单 flag 最多 14 次 Set。静态检查同时确认 H 初始状态或前一 chunk Vec2
  发布 HReady，O 当前 chunk Vec1 等待；H 当前 chunk Vec1 发布 VReady，O 当前
  chunk Vec2 后等待。末 chunk 不生成下一轮 HReady，单 chunk 的 VReady 由收尾路径
  发布。H/O 数据按相同 task 和 chunk 编址，mode-0 flag 本身是全局集合握手。
- 本环境没有 CANN/NPU，尚未构建和运行。目标设备需覆盖单 chunk、三 chunk、
  DirectUb/StandardGm、V128/V256，并采集 Set/Wait 代次及精度结果。

## AIC/AIV ready 分工（待设备验证）

- H_READY 继续由 H/O 两侧 AIC 执行双向 `Set/Wait`；A5 V128 preload 的
  V_READY 由 H/O 两侧对应 AIV 执行双向 `Set/Wait`。
- H_READY 每七对任务一组，每轮最多 14 次 Set；V_READY 每六个相对 AIV
  编号一组，每轮最多 12 次 Set。当前 flag 范围覆盖 Host 的 21 个 task pair
  上限。
- A5 chunk pipeline 的 O scheduler 与 H scheduler 使用同一单 task 串行模型：每个
  producer/consumer pair 固定使用唯一 active stream/slot，当前 chunk 完成并发布
  `vec2Done` 后才推进到下一 chunk；不为同一 task 伪造第二个并行 stream。
