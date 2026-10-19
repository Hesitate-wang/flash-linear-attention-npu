# 开发阶段验证记录

## 2026-10-17 MODE2 OV1/HV2 与 HV1/OV2 分组复用

- MODE2 UB 按执行重叠关系重排：OV1 与 HV1 保持独立，OV1 的 `[71,103) KiB`
  在 HV2 开始前复用为 H update scratch；HV1 的 H_DIRECT `[117,181) KiB`
  在 H→O 交接后复用为 OV2 A/H work `[117,165) KiB`。
- AIV 顺序改为 `OV1 -> HV1 -> O→H fence -> HV2 -> H→O fence -> OV2`。
  删除 OV1→HV1 的整段 fence；保留 HV2 覆盖 OV1 区域前、OV2 覆盖 HV1 区域前，
  以及 OV2→下一 chunk OV1 的 fence。
- 已更新布局常量、H update epilogue 引用和静态布局回归。设备编译、精度和性能结果待 A5 环境验证。

## 2026-10-17 MODE2 mask 与 HV2 g_last 地址冲突修正

- 复核发现 `epilogueUpdate` 的旧 `glast` 地址 `[231,232) KiB` 位于 MODE2
  永久 causal mask `[231,247) KiB` 内。此前每 chunk 重建 mask 会掩盖该覆盖，
  mask 改为单次初始化后会在 HV2 之后破坏 mask。
- MODE2 `epilogueUpdate` 现在使用 `[247,248) KiB` 的专用单值临时区；MODE1/旧路径
  继续使用原地址。布局静态断言已覆盖该区域。
- AIC `ov2ChunkFinished` 曾位于 HC1 之后；因 OV2 复用 H_DIRECT，已移到每轮
  OC1/HC1 之前，源码回归已验证该顺序。

## 2026-10-16 MODE2 causal mask 单次初始化

- MODE2 mask 已有专用 UB 区间 `[231,247) KiB`，QK-mask epilogue 只读取该 64x64
  下三角矩阵。将 `RebuildCausalMask` 从 chunk 循环内移至循环前，每个 AIV 实例只构造
  一次；保留每 chunk 阶段 fence 和其他数据同步。
- 源码回归检查初始化位于循环前且调用仅一次。性能收益未在 A5 设备上 profiling；本机无
  CANN/`ccec` 和 NPU，未执行设备构建或精度验收。

## 2026-10-15 MODE2 UB 临时变量统一布局

- MODE2 fused UB 从 H/O 阶段分区改为变量/生存期布局；共享 `gate`/`G_LAST` 地址，
  mask、初始化 staging、QK scratch、H_DIRECT 与 OV2 work/output buffer 按阶段交接复用。
- MODE2 OV2 work/output 改为单份 UB，分段开始前等待上一段 MTE3 写回完成；普通 O
  路径保持原 ping/pong。H_DIRECT 仅在 H→O fence 后允许 OV2 work 复用，OV1 并发期仍与
  Fixpipe/QK scratch 分离。最高活动地址为 183 KiB，低于 `[248,256) KiB` 通信保留区。
- 普通 A5 O epilogue 默认仍使用原 layout；仅 fused MODE2 通过显式构造参数使用新布局。
  MODE1 调用点、计算路径与共享 O epilogue 行为未改。
- `python3 -m unittest discover -s tests -p test_chunk_fwd_ho_mode2_sync.py -v`：6 项通过，
  覆盖事件模型、交接次序、变量容量/alias、单缓冲写回等待及 footprint 上界。
- `clang++ -std=c++17 -x c++ -fsyntax-only` 对 fused UB 布局头检查通过；
  `git diff --check` 通过。该检查不编译 Ascend kernel。
- 本机无 CANN/`ccec` 和 A5 设备；设备构建、ATK 精度、MODE1 设备回归及多 chunk 超时
  验收未执行，不据此宣称设备死锁风险已经验证消除。

## 2026-10-15 AIC chunk gate 顺序修正

- 静态依赖检查发现，若 AIC 将 `ov2ChunkFinished` 等待放在 HC1 后，会形成环：AIC
  等待 AIV 的 OV2 完成，AIV 的 OV2 又等待 AIC 后续产生的 OC2/OC3 ready。
  HC1 还会与上一轮复用 `H_DIRECT` 的 OV2 并发。
- 已将等待恢复到 AIC 每轮入口、OC1/HC1 之前；新增源码顺序回归，约束 gate 先于
  OC1、HC1，且每轮只消费一次该 flag。
- 这是静态依赖图结论；本机无 CANN/`ccec` 和 A5 设备，尚未完成设备编译及多 chunk
  超时验证。

## 2026-10-14 MODE2 H/O Local Event 隔离与 UB 交接

- H single-slot epilogue 的 local ID0-3 映射为具名 ID4-7；O 仍使用 ID0-3。
  driver 只 preset/drain `MTE3_V(4)`、`MTE3_MTE2(6)`，初始化 ID5 全部闭环。
- O→H 使用 ID3、H→O 使用 ID7 的私有 fence，配对 MTE2/V/MTE3 依赖；
  覆盖 OV1→HV1、HV2/跳过→OV2、OV2→chunk-finished 及下一 mask 重建。
  未改共享 O/MODE1 epilogue、host tiling、公开 ABI、实际 shape 或数学公式。
- HV1 的 HC2-ready 通知移到最后一次 H_DIRECT 读取之后；flag ID、消费者及
  每 chunk 生产次数不变。OV2 pong work 仅在 H fence 后复用 H_DIRECT。
- 当时按旧布局记录的 HV2 低精度 staging `[80,96)` KiB 与 O 输出 pong 247 KiB
  已由 2026-10-15 变量生命周期布局取代；当前 MODE2 fused 活动区终点为 183 KiB。
- 离线命令：`python3 -m unittest discover -s tests -p test_chunk_fwd_ho_mode2_sync.py -v`。
  8 项通过：事件域隔离、初始化分支闭环、错误事件拒绝、多 chunk/最终状态跳过/
  空行 token 状态、阶段顺序与 H_DIRECT 释放、尾块不发布下一 chunk 的 CrossCore token、
  UB 区间/实际尾块跨度。
  事件模型读取真实 Set/Wait 和分支，以逻辑 token 检查重复 Set/无生产者 Wait；
  H 条件使用保守分支超集；MODE2 OV2 事件模型按单缓冲逐 32 行片段等待写回，
  普通 O 路径仍使用原 ping/pong。
  UB 检查结合区间与源码交接顺序，不是硬件队列调度或跨核协议仿真。
- `clang++ -std=c++17 -x c++ -fsyntax-only` 编译 single-slot UB 布局头通过，
  布局容量/区间静态断言成立；`git diff --check` 通过。此命令未编译 Ascend kernel。
- 本机无 CANN/`ccec`、Ascend NPU；无法运行设备编译、ATK 精度、MODE1 设备回归
  或 60 秒超时验收。当前状态仅为代码修改与离线检查通过，不能认定死锁已获设备验证。

设备验收待办：A5 FP16/BF16、支持的 gate/state dtype、initial/final 四组合、
单/多 chunk、chunk64/128 和尾块 1/31/32/33/63/65/127、MODE2 核数边界；比较
O、H、V_new、final state，并重复多 chunk 检查事件代次。单次 kernel 超过 60 秒
判失败并终止、记录配置；同时回归 MODE1。Cube 现有一次 preset/drain 生命周期
保持工作区版本，本次未对其作硬件验证。

## 2026-10-14 MODE2 统一 preload 与 H stage 内联

- `gdn_fwd_ho_fused_kernel.hpp` 直接持有 H 的 MMAD、epilogue、UB tensor 和
  stage 方法，不再包含或实例化 `GDNFwdHPreloadSingleSlot`；原 helper 文件保留。
- 五个矩阵统一使用 `MmadPingpongTlaPreload`。公共 BlockMmad 新增
  `operator()(A, B, C, actualShape)`，共享主体以 `EmptyClass` 在编译期去除
  `vecDone` 等待；原带 flag/readyFlagId/bias 的签名保持不变。
- OC1/OC2/OC3 使用无 flag 重载。driver 保留 chunk gate 和 OC3 的 ready wait；
  HC1 完整 drain 后 OC2 才 preset 默认 L1/L0 events，取消原 HC1/OC2 分离预取重叠。
- H 和 O 的 shape 使用实际 `blockTokens`，H 向量 stage 继续传实际长度；未增加
  尾块专用路径，未修改公开接口、host tiling 或 MODE1 调度。
- 静态搜索确认目标 kernel 不再引用 helper、pipelined MMAD 或分段预取 API；
  `git diff --check` 通过。本机无 CANN/`ccec`；Clang 检查在缺少 Catlass 头文件处
  停止，不能证明设备编译或精度通过。待 A5 构建及首/多 chunk、实际尾块、
  initial/final state 组合的设备精度与同步回归。
- 公共 preload header 的原签名调用方包括本算子 A5 MODE1 preload、旧通用 H
  preload，以及 `chunk_gated_delta_rule_fwd_h` 的通用/A5 preload；原调用语义保留，
  这些模板仍需随公共 header 改动进行设备编译回归。

## MODE2 saturated-core path

- A5 natural-exp, fixed-length, V=128 and no-gk only: `floor(P/2) < T <= P`
  selects one MIX core per task (`blockDim=T`); `T<=floor(P/2)` retains MODE1.
- Dedicated MODE2 kernel/helper/epilogue files contain the single-slot path;
  MODE1 H/O kernels and shared MODE1 epilogues remain unchanged.
- The local mode-`0x2` graph seeds `hv2HReadyForHc1` before chunk 0.
  Non-final HV2 republishes it; the final chunk omits both next-chunk tokens
  (`hv2HReadyForHc1`, `ov2ChunkFinished`). OC2 follows HC1 and needs no second
  H-ready wait; OV2 waits for OC2/OC3. Device checks still required: CANN build, one-/multi-chunk CPU and
  composed precision, initial/final-state combinations, and T boundary cases.
- Local environment has no `ccec`/CANN toolchain or Ascend device; only static
  source checks are available here.
- MODE2 compile-error review: defined `LayoutVNEW`; replaced the mismatched
  preload-policy GM aliases with private GM tail epilogues; checked all six
  driver/helper epilogue template arities and all four H epilogue call arities
  against their declarations. The GM tail variants use one shared slot and
  omit the MODE1 V-ready rendezvous. Added direct includes for the preload
  helper's tile-copy and H-offset types and for private epilogue policies.
  Local type-alias, relative-include, GDN-constant and whitespace checks pass;
  these checks do not substitute for a CANN kernel build.

## 设计追踪

| 设计项 | 实现位置 |
| --- | --- |
| `T` 对生产者/消费者核 | Host tiling 对 MODE1 检查 `T<=floor(P/2)`，对受限 A5 MODE2 检查 `floor(P/2)<T<=P` 并设置 `activeCoreNum=T` |
| 完整 H/`v_new` 交接区 | Host 的 `FillWorkspace`；kernel 的 `handoffHWorkspaceOffset` 和 `handoffVWorkspaceOffset` |
| 启动事件初始化 | A2/A5 统一入口不再初始化同步区或执行额外启动屏障；自然指数路径不访问 IB event table |
| 逐 chunk 的 H 到 O 发布 | mode `0x0` 的 `H_READY=12`、`V_READY=13` 按 chunk 执行全体 AIV `Set→Wait`；本核 `hc1WhReady/hc2KvUpdateReady/hv1VUpdateReadyForHc2` 继续使用 mode `0x2` |
| A5 的 H 到 O 阶段边界 | 自然指数路径按 chunk 执行 CrossCore H/V rendezvous；exp2 专用 O 仍使用既有 `SyncAll<false>()` 阶段边界 |
| A5 临时区所有权 | Host 的 `FillWorkspaceA5`；自然指数通用 O offset 或 exp2 的 `oAPrimeWorkspaceOffset` |
| 不依赖同级算子的私有实现 | 所有 H/O kernel、调度器和收尾文件都位于当前算子目录内 |
| FwdO 架构隔离 | A2 实现在 `op_kernel/gemm/kernel`；A5 kernel、配套 epilogue、exp2 tiling/constants 和 UB layout 全部位于 `op_kernel/arch35` |
| A5 自然指数 FwdO Vec1/HReady 重排 | qkmask 两条分支均先发射 gate/causal-mask Vector，再在 QK MTE2 前执行 H_READY Set/Wait 并立即 ACK；`chunk=128` 仅首段握手。待设备检查三 chunk 精度、IB/flag 代次以及相同条件下的 profiling 总耗时 |
| A5 FwdO V128 两段 Vec2 | 新增 `hc2KvUpdateReady[0/1]` 事件 8/9；Cube2 完成后先对整片 H FP32 乘 `exp(g)` 并驻留 UB，Cube3 完成后再 Add、Muls、Cast、写 O。V256 沿用旧融合路径。待设备验证 V128 chunk64/128、两种 gate dtype、三 chunk slot 回绕的精度、同步代次和总耗时 |

## 已完成的静态检查

- Host 和 kernel 的融合 tiling 镜像使用相同字段顺序；Host tiling 保留与算子内
  H 完全一致的结构前缀，并检查完整序列化结构的大小。宏生成的 Host TilingData
  采用本地构造，在所有字段和 workspace offset 填充完成后显式 `SaveToBuffer` 到
  raw tiling buffer，并设置实际数据长度；不再将 raw buffer 解释为未构造的
  TilingData 管理对象，因此首个 `set_batch` 不会访问空的内部存储。
- Workspace 偏移均相对于用户 workspace，所有区域都按 512 字节对齐；返回的
  workspace 大小只累加一次平台系统 workspace。
- A2 和 A5 MODE1 要求 `B * HV <= floor(physical AIC cores / 2)`，
  `blockDim=2*B*HV`。A5 MODE2 在限定的 saturated range 使用 `blockDim=B*HV`，
  每个逻辑 MIX core 负责一个 `(batch, value-head)` 任务，临时 workspace 只分配一槽。
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
  `2 * blockDim`；O AIC 读取完整 `H_old` tile 前，通过反向 `hc1WhReady` 的代次
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

## 完整 chunk OC2 预取调整

- OC2 的 GM→L1 位于 HC1 发射之后、HC1 资源回收等待之前；等待仅在完整 chunk 调用。
- HC1/HC2 保留原内部缓冲配置，补齐 preSetFlags/finalWaitFlags 生命周期。OC2 预取后回收 HC1 事件，再初始化 OC2 L0 事件；HC2 的事件在 OC3 之前回收。
- H L1 为 [0,128 KiB)，OC2 为 [128,256 KiB)，OC3 为 [192,320 KiB)。H 使用事件 0–3，OC2/OC3 在阶段结束后复用事件 4–7。
- 本地完成源码顺序、完整 chunk 分支保护及 diff 格式检查；CANN 编译、设备精度和重叠收益仍待验证。


## 2026-10-19 固定 64/128 的独立 MODE2 OV1 构件

- 用户指定 `gdnFusedMode2UbMallocTable`；实现独立 policy、include guard 和模板 specialization，
  避免与现有 O policy/epilogue 重定义。表内 constexpr 成员改为 static constexpr，单槽别名一致。
- 基于既有 QK-mask 的数学语义，限制 full chunk=64、K=V=128、两 AIV，按一份 T32
  组装两个 lane 的因果 mask。参考角色：既有 arch35 OV1 的广播、自然指数、cast 与事件 API；
  没有修改 CPU 标杆或公开接口。详细数据流与集成约束见 design.md 新增小节。
- 仅独立构件，未改 driver/host；原整算子 ATK 不能验证这个尚未实例化的新路径。
- 验证计划：静态布局/编译 policy、CPU 象限分解与完整 64×64 因果参考对照，现有 MODE2
  本地同步回归；目标 CANN 编译后运行 FP16/BF16/FP32 gate × 两种输出 dtype、两个 AIV、
  多 chunk、自定义 gate 差值、对角线/上三角精确零、A5 内存检查。验证 8 KiB Broadcast
  workspace 的目标 SDK 最小要求，接入整表后再跑全部受影响阶段与原场景回归。
- 本机没有 ccec/CANN/A5 NPU：设备编译、Broadcast workspace 需求、设备精度/同步和内存
  验收未执行，构件尚未达到算子正式验收条件。

本轮本地结果：

- policy 表使用最小 Catlass 类型桩通过 `clang++ -std=c++17 -fsyntax-only`，并验证 gate、
  workspace、HC1、H/H-work 与单槽输出的地址静态断言。该结果仅验证普通 C++ 与常量，
  不验证 AscendC API 或设备代码。
- CPU 独立模拟两 AIV 的 T32→32×64 拼接、Copy 步长及门控公式，64 组随机 gate/QK，
  合计 262144 个元素，与完整 64×64 因果参考及 FP16 舍入逐元素一致。
  这不是新 epilogue 的 NPU 精度验收。
- `python3 -m unittest discover -s tests -p test_chunk_fwd_ho_mode2_sync.py -v`：8 项中
  7 项通过，`test_aic_chunk_gate_precedes_all_chunk_work` 失败。当前 driver 的
  `ov2ChunkFinished` 等待在 HC1 之后；逐字节确认 driver 与 HEAD 相同，因此为本次未修改
  路径的既有失败。本轮未调整 driver 或该测试的期望值。
- `git diff --check` 通过；两个新增文件另外检查行尾空白。


## 2026-10-19 固定 64/128 独立构件的 mask/gate 准备

- 新 OV1 构造函数接收外部 mask/gate LocalTensor；内部不再初始化 mask 或搬运 gate。
  调用方在 chunk 循环前初始化 T32，每 chunk 在前一代读者释放后执行一次 GateBuffer::Prepare。
- 新 HV1/HV2/OV2 共享只读 FP32 gate，删除各自重复的 GM gate load/cast；HV2 两份输出合并一次同步，
  OV2 移除 gate cast 被删除后多余的 V 屏障。旧 driver、旧 epilogue 和内存表未改。
- `python3 tests/check_gdn_fused_mode2_epilogues_cpu.py`：ASan/UBSan CPU 仿真通过实际四个新头文件，
  六种数据/gate组合、两个lane、三chunk的OV1/HV1/OV2及48组HV2状态场景；检查mask/gate不被改写，
  每chunk只搬运一次gate且至多转换一次。同步CPU桩不验证真实CANN签名和异步流水。
- 新构件仍未接入 driver。本机无 CANN/NPU，设备构建、ATK精度及跨核 ready/free 验证待集成后执行。


## 2026-10-19 固定 64/128 OV1 内部等待 OC1

- OV1 新增传入的 `oc1Ready`，内部在 QK DataCopy 前执行 CrossCoreWaitFlag。
- CPU 仿真为每个 lane/chunk 的 QK 读取加入 guard：未消费传入 ready 时读取直接失败，
  并要求每次调用恰好等待该 flag 一次。六种 dtype 组合与原有状态场景回归通过。
- 调用方示例同步删除外部 OC1 等待；旧 driver 未接入新构件，真实设备同步验证仍待执行。


## 2026-10-19 OV1 延后等待与 gate/mask RegBase 拆分

- 对齐现有 fwdo_qkmask 顺序：Broadcast→mask组装→PrepareGateMask VF→V屏障→OC1 wait→QK搬入→MTE2_V→Mul→cast。
- RegBase 函数移除 attn 参数及 QK 乘法，只负责独立的 gate/mask 准备。
- CPU 仿真增加等待点的完整 gate 衰减矩阵核对，要求独立计算已完成才等待OC1，且等待后才可读取QK。
  原六种dtype、两个lane、三chunk及48组状态场景一并回归；设备验证仍待执行。


## 2026-10-19 HV1/HV2/OV2 等待位置与 RegBase 边界

- HV1拆分PrepareRelativeDecay/ComputeVnew，gate指数和u转换提前到HC1等待前；
  HV2将ScaleState提前到HC2等待前，等待后仅Add；OV2拆分PrepareDecay/ScaleInter，
  分别提前到OC2/OC3等待前，等待后仅Add/Muls/cast。函数只接收本阶段所依赖的参数。
- HV1/OV2先后复用OV1已结束的29KiB工作区开头256B；保留末尾搬出等待和跨核token配对。
  HV2无输入搬运的隐式零初态分支跳过空MTE2_V。
- CPU仿真增加HC1/HC2/OC2/OC3等待点的完整独立计算结果检查及提前读取保护，
  六种dtype、双lane、三chunk与48组状态场景通过ASan/UBSan；设备构建与异步同步/性能未验证。


## 2026-10-19 OV1 Fixpipe输入 / L1输出配置

- 构造参数enableL0C2UB/enableUB2L1分别控制跳过QK的GM搬运和改写L1，默认均false。
  L1为调用方提供的64x64 zN，两个lane分别写32行；未改Cube或driver。
- CPU仿真通过四条路径、六种dtype、两lane和三chunk，直连使用空GM参数，检查不搬运/不发QK的MTE2_V；
  L1检查布局、边界、另一lane保留及搬出等待。原HV1/HV2/OV2精度和等待位置检查一并通过。
- 不能用同步CPU桩证明真实Fixpipe/UB→L1签名和异步行为；CANN/NPU验证与driver配套待执行。


## 2026-10-19 其余三个Vec构件的Fixpipe/GM与L1/GM路径

- HV1/HV2/OV2新增两个默认false的构造开关及L1输出tensor；HV1/HV2补充GM Cube结果输入参数。
  独立计算仍提前，ready后按需搬入或直接使用UB；L1输出复用私有CopyRowsToL1Zn助手。
- HV2保留final_state GM语义；L1 H跨chunk保留于h_ping，下次使用按DataT舍入后扩宽，FP32状态链保持原语义。
- `python3 tests/check_gdn_fused_mode2_epilogues_cpu.py`：四个epilogue的4种路径×6种dtype×2个lane×3个chunk通过；
  HV2有192组状态场景，覆盖初态、末态、保存开关、FP32/DataT状态及L1 H的保留递推。
  验证空的未使用GM参数、按需MTE2_V、输出完成等待、L1布局/边界/另一lane不变和独立计算先于ready等待。
- 无CANN/NPU，设备API、异步Fixpipe/UB→L1、精度与性能未验收。Cube/driver未接入；
  OC2/OC3写37/53KiB前的HV1反向free及L1消费者释放必须由调用方配套，不能由单向ready推断。


## 2026-10-19 HV1 vnewDecay / vnew分段计算与通知

- ComputeVnewDecay只生产HC2的decay。搬出完成发vnewDecayReadyForHc2，再Sub计算vnew，搬出完成发vnewReadyForOc3。
  两次各闭合搬出事件；两次cast仅复用53KiB原始u，保留61KiB的u_fp32供第二段使用。
- `python3 tests/check_gdn_fused_mode2_epilogues_cpu.py --skip-qkmask`通过：六种dtype、四条路径、双lane、三chunk，
  检查通知顺序、通知时搬出完成、HC2通知时vnew尚未计算、u_fp32未被覆盖及HC2独立h_work写入；
  HV2的192组状态场景回归通过。
- 默认全量命令失败于当前qkmask被注释的搬出等待与原断言不符。未修改该文件；skip选项明确只验证HV1/HV2/OV2。
- 两份输出flag由新HV1内部发布，旧driver仍未接入。实际Cube消费/双AIV代次/异步重叠须设备验证，不能以CPU桩替代。


## 2026-10-19 局部同步原地展开

- 删除三个同步包装函数，将vnew/update/output及GateBuffer::Prepare共15处调用展开为原地SetFlag/WaitFlag。
  事件类型、编号、条件和调用顺序保持不变，未改qkmask。
- `python3 tests/check_gdn_fused_mode2_epilogues_cpu.py --skip-qkmask`通过原有事件计数、计算结果、两段通知及192组状态场景检查。
  qkmask的独立改动仍按上一条记录隔离；本次为等价代码展开，未新增设备验证结论。


## 2026-10-19 局部事件改用EVENT_ID常量

- 删除HV1/HV2/OV2的kEvent别名，分别直接使用EVENT_ID4/5/6；GateBuffer准备使用EVENT_ID7。
  OV1原EVENT_ID0保持，事件分配和顺序不变，跨核flag仍由调用方传入。
- 源码扫描确认本组所有局部SetFlag/WaitFlag均使用EVENT_ID常量。
  CPU仿真入口使用--skip-qkmask隔离既有QK改动，检查原事件编号和通知顺序。


## 2026-10-19 HV2初态移交driver

- 新HV2删除initialState/useInitialState参数与初态清零/搬入/转换；首chunk消费driver已准备的FP32 h_ping。
  后续chunk保持FP32 state、GM DataT与L1保留H的恢复和舍入规则。
- CPU仿真在chunk循环前初始化各AIV h_ping，首chunk hInput传空；验证首chunk没有状态输入DMA，
  在HC2等待点核对衰减已完成，回归四条路径、六种dtype、双lane、192组状态场景。
- `python3 tests/check_gdn_fused_mode2_epilogues_cpu.py --update-only`通过，启用ASan/UBSan，仅验证本次HV2修改。
- 当前HV1也存在搬出等待被注释的修改，`--skip-qkmask`运行在HV1事件计数断言处失败；本次保留该修改，未宣称全量通过。
- 当前独立构件未接入旧driver，循环前准备示例已更新；设备验证仍待执行。


## 2026-10-19 HV2跨chunk常驻FP32 state

- 删除hInput/isInitialState参数及全部chunk间状态恢复、DataT往返转换，所有chunk直接更新h_ping。
  driver初态加载独立于storeFinalState，FP32初态直接加载，DataT初态仅在循环前扩宽一次。
- CPU参考改为FP32连续递推，逐chunk核对UB保留的未舍入结果；破坏GM副本验证后续chunk不读回state。
- 验证命令：`python3 tests/check_gdn_fused_mode2_epilogues_cpu.py --update-only`；
  192组三chunk场景、六种dtype组合、四种输入输出路径、双AIV，启用ASan/UBSan。
  该记录替代上一节中的chunk间恢复规则。其余epilogue的独立修改与设备验证限制保持。


## 2026-10-19 HV2内部发布HC1 ready

- 新增hReadyForHc1入参，非末chunk在GM/L1输出及搬出等待后内部set，末chunk不set。
- CPU仿真在set回调核对输出内容、HC2已等待、MTE3搬出等待已执行、每AIV恰好一次通知，
  并检查末chunk无通知（保存/不保存均覆盖）。
- 命令：`python3 tests/check_gdn_fused_mode2_epilogues_cpu.py --update-only`，
  覆盖192组三chunk状态场景、四种输入输出路径、六种dtype组合及双AIV，启用ASan/UBSan。
  同步CPU仿真不验证真实跨核代次或双AIV汇合，仍需driver接入后的设备验证。


## 2026-10-19 按AIV分块重新计算H内存偏移

- h_ping=[109,141)KiB，h_work_ping=[141,173)KiB；OV2的37/53/69KiB复用偏移不变。
- 更新内存表、容量断言、HV2直连注释和设计说明；测试的模拟Fixpipe目的引用表中偏移自动更新。
- 当前用户已将HV2 ready移至H搬出后的MTE3队列，本轮保留该位置；测试在set时检查输出及V_MTE3，
  返回后继续检查MTE3_V/MTE3_MTE2完成，避免要求set必须位于本地drain之后。
- 命令：`python3 tests/check_gdn_fused_mode2_epilogues_cpu.py --update-only`，192组三chunk场景，
  启用ASan/UBSan；实际HC2 Fixpipe地址接线及异步流水仍待driver接入后的设备验证。


## 2026-10-19 完善独立OV2布局的断言

- 内存表补齐OV1、gate/mask、HV1、HV2和OV2各边界及32B对齐检查，保留用户的207KiB输出起点。
  OV2移除依赖77KiB v_new_ping的旧断言，HV2改为检查h_work不得覆盖o_inter。
- 临时C++编译检查：当前布局通过；地址未对齐、HC2/OC2重叠、OC2/OC3重叠、OC3/cast重叠、
  输出进入通信保留区五类错误偏移均被预期static_assert拒绝。
- `--skip-qkmask`已通过全部相关模板编译；运行仍在既有HV1搬出等待计数检查失败，未修改该同步逻辑。
- `python3 tests/check_gdn_fused_mode2_epilogues_cpu.py --update-only`用于HV2的192组三chunk回归，
  启用ASan/UBSan。未执行CANN/NPU验证。


## 2026-10-19 HV2去除逐次本地drain的强制断言

- CPU仿真不再要求每次HV2调用执行MTE3_V/MTE3_MTE2；保留V_MTE3、HC2等待、
  GM/L1输出内容及非末chunk恰好一次MTE3 H-ready通知的检查。
- 依赖MTE3 H-ready→HC1→HV1→HC2→下一HV2的跨chunk链保护源UB复用；
  末task输出完成到新task初态写入的同步仍由driver保证，不由同步CPU桩推断。
- 验证命令：`python3 tests/check_gdn_fused_mode2_epilogues_cpu.py --update-only`，
  192组三chunk场景、六种dtype、四条输入输出路径、双AIV，启用ASan/UBSan。
  本次仅调整HV2对应检查，未改HV1两次同chunk cast槽复用的断言。


## 2026-10-19 OV2内部发布完成信号

- OV2新增chunkFinished参数，GM/L1输出之后内部在MTE3上发布一次；所有调用示例同步更新。
- 新增`--output-only`入口，独立验证OV2，避免受既有HV1同chunk暂存复用检查影响。
  在set回调检查输出内容、OC2/OC3等待顺序、V_MTE3与恰好一次通知，覆盖六种dtype、四条路径、
  三chunk及双AIV，启用ASan/UBSan。同步桩不验证实际设备异步顺序或跨task token管理。


## 2026-10-19 Mode2新schedule/driver联调

- 新建独立scheduler、私有单tile Cube块和driver，固定64/128整chunk入口接新实现，其他场景回退原driver。
- 默认Fixpipe→UB、Vec→GM（整体复查后按用户要求调整），模板保留四种组合；按当前UB表与实际L1布局连接全部生产者/消费者。
- 初态和mask循环前初始化，H FP32常驻；9个跨核flag按chunk平衡，最后OV2 token在Cube退出前消费。
- HV1同chunk共享cast槽恢复一组MTE3_V set/wait，wait推迟到第二次cast前，保留Sub重叠；
  HV2逐chunkdrain保持删除。原epilogue测试调整过时的本地drain计数检查，保留必要的ready、内容和顺序检查。
- `python3 tests/check_gdn_fused_mode2_epilogues_cpu.py`：四epilogue全量CPU仿真通过，
  六种Data/Gate组合、四条路径、双AIV、三chunk及192组HV2状态场景，ASan/UBSan。
- `python3 tests/check_gdn_fused_mode2_driver_cpu.py`：实际scheduler/driver/Cube块/epilogue的
  三线程CPU联调，覆盖四种搬运组合、1/3 chunks、8种Input/Gate/State组合与初态/终态开关，共256场景；
  独立参考核对O/final_state，所有flag无残留；scheduler额外检查多batch/GQA、inactive core、大于32位偏移与fallback条件。
- 当前为开发验证：没有CANN编译器、Catlass vendor头和NPU；不能据此宣称SDK构建通过、
  实际异步同步正确或达到性能目标。正式ATK/Example/ST、mssanitizer与设备profiling待目标环境执行。


## 2026-10-19 Mode2公共Cube preload复用修正

- 原Mode2私有块仅借用TileMmad类型，未调用公共BlockMmadTla流水。现改为继承并实际调用
  common/kernel_utils/block/block_mmad_pingpong_tla_preload.hpp的operator()、preSetFlags和finalWaitFlags。
- 私有适配只绑定Vec生产的L1操作数和跳过其GM→L1搬运；五个Stage使用各自精确M/N/K，
  单槽L1/L0配置；公共头和旧driver均未修改。
- python3 tests/check_gdn_fused_mode2_driver_cpu.py：256场景通过，ASan/UBSan；本轮测试直接包含
  实际公共preload头，仅模拟设备/TLA基础类型与tile原语。核对O/state、四条搬运组合、跨核token收支，
  并检查Cube本地事件重复set、空wait、退出残留及B/C槽preset/release计数。
- 无CANN/NPU环境；CPU事件检查不等价于硬件流水/可见性验证，SDK编译和设备精度/同步检查尚待执行。


## 2026-10-19 整体复查结果

- driver默认CubeToUB=true、VecToL1=false；通过不传这两个参数的真实模板实例验证默认选择。
  默认Cube输入不从GM workspace重复搬入Vec，GM H/Vnew快照与独立参考一致。
- 修正非首chunk gate/U的复用边界：在gate.Prepare之前用V_MTE2 EVENT_ID7约束下一批MTE2。
  保留OV2已有MTE3_V，不恢复HV1/HV2内部被去除的逐次drain。
- HV2仅在末chunk写FP32 final_state，清理嵌套重复判断；修正UB表与OV1异步返回的过时注释。
- 核对入口条件、host workspace、GQA与batch偏移、各UB/L1区间、五阶段精确M/N/K、
  公共preload调用、九个跨核信号的首末chunk计数、末chunk跳过更新及退出回收。
- python3 tests/check_gdn_fused_mode2_driver_cpu.py：304场景通过，ASan/UBSan。
  其中256个既有组合、16个默认模板场景、32个多batch/GQA场景。
  多batch/GQA采用B=2/HK=2/HV=8，对16个task分别执行GM/L1路径，逐task三个线程；
  不模拟不同物理Cube核心并发。核对O/state、GM H/Vnew快照和未选task的输出保护，
  并核对跨核token、本地事件收支、直连输入跳过GM搬运及跨chunk V→MTE2依赖。
- python3 tests/check_gdn_fused_mode2_epilogues_cpu.py：四epilogue全量通过，6种Data/Gate组合、
  四路径、双AIV与3 chunks、192组HV2状态场景；新增检查非末chunk不写final_state。
- git diff --check通过；本地无ccec/CANN/NPU。CPU仿真不覆盖真实硬件流水、跨核可见性、
  vendor头类型匹配、性能和正式ATK/ST验收，不将本轮结果记为设备验收通过。


## 2026-10-19 用户移动EVENT_ID7后的复查

- 重新读取确认：OV2在ScaleInter及PIPE_V barrier之后set V_MTE2 EVENT_ID7，
  driver非首chunk入口wait，循环退出再wait一次。N个chunk对应N次set和N次wait，无首轮空等或末轮残留。
- 最终MTE3_V/MTE3_MTE2仍保留；早期V_MTE2仅释放gate/scratch，不替代最终输出完成依赖。
  Cube仍在chunk入口等待chunkFinished，Vec仍按完整OV2后进入下一轮，尚未实现跨chunk OV1/HV1交错执行。
- 算子源码未改；仅同步CPU测试的HardEvent枚举和N次set计数，并对Vec EVENT_ID7增加重复set、空wait和退出残留检查。
- driver 304场景及四epilogue全量CPU验证重新通过，ASan/UBSan；不代表设备异步流水验收。
