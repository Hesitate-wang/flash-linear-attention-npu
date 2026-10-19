# 固定 64/128 的 MODE2 HV1、HV2、OV2 独立构件

设计规则 V2。本组为用户要求新建的独立构件，本轮调整新 OV1/HV/O 构件与公共辅助头。
既有 driver、旧 HV/O epilogue 和内存表保持原样；设计和验证记录同步更新。本文件同时记录本轮方案与开发期验证，
不替代整算子的正式设备验收。

## 边界与映射

chunk=64，K=V=128，自然指数，两个 AIV，单槽，无 gk 门控；DataT 为 FP16/BF16，
GateT 为 FP16/BF16/FP32，StateT 为 FP32 或 DataT。公开 API、host、dispatcher 不变。
HV1/OV2 lane s 处理 token 行 [32s,32s+32)，HV2 处理状态行 [64s,64s+64)。
构件以当前 chunk/状态矩阵的 GM 起点接收参数，内部加 lane 行偏移。
完整 chunk 的支持条件由将来接入的调用方校验，不接受尾块或其它 shape。

## 逐 Stage 设计

| Stage | FP32 公式 | 输入 | 输出 |
| --- | --- | --- | --- |
| HV1 | v_new=u-WH；v_update=v_new*exp(g[63]-g[row]) | HC1 直连 v_work；GM u；共享 FP32 UB gate | GM DataT v_new/v_update |
| HV2 | h_next=h_prev*exp(g[63])+h_work | HC2 直连 h_work；GM h/可选 FP32 state；共享 FP32 UB gate | GM DataT h_next；可选 StateT final_state |
| OV2 | o=(o_inter*exp(g[row])+o_intra)*scale | GM OC2/OC3 FP32 结果；共享 FP32 UB gate | GM DataT o |

RegBase 函数按数据依赖拆分：HV1 为 PrepareRelativeDecay / ComputeVnewDecay / 外层Sub，HV2 为 ScaleState，
OV2 为 PrepareDecay / ScaleInter；等待后仅执行依赖 Cube 结果的操作。FP32 乘法和加法分开，
不改成 FMA。HV1 的 v_update 从未舍入的 FP32 v_new 计算，与现有 preload 公式相同。
HV2 保持 storeFinalState+FP32 的状态链：首 chunk 使用 FP32 initial_state 或零，后续
chunk 使用上一代 FP32 final_state；未启用该路径时从 DataT h_prev 转 FP32。
末 chunk 且不保存 final_state 时只消费 HC2 ready，不读未产生的 h_work。
OV2 的 exp(g[row]) 是每个 token 一个值，不能误用 exp(g[63])。

## UB 资源、生命周期与复用

全部地址只引用既有 gdnFusedMode2UbMallocTable。单位 KiB，每 AIV：

| Stage | 实际使用区间 | 说明 |
| --- | --- | --- |
| 共同 | g_input [28,28.25)，g_fp32 [28.5,28.75) | 每 Stage 各搬一次64个 gate；16位输入仅用前128B |
| HV1 | v_work [37,53)，u [53,61)，u_fp [61,77)，v_new [77,93)，decay [93,109) | 每 lane32×128；两个结果先保持FP32 |
| HV1 cast | DataT decay / v_new 顺序共用 [53,61) | 仅复用已结束的原始u；u_fp [61,77)保留到第二段Sub结束 |
| HV2 | DataT staging [93,109)，h_fp32 [109,141)，h_work [141,173) | 每 lane64×128；staging复用已结束HV1的decay槽 |
| OV2 | o_inter [37,53)，o_intra [53,69)，o_out [69,77) | 每 lane32×128；VF原地更新o_inter后独立cast |

HV2 的 h/h_work 槽各32KiB，按每AIV的64×128 FP32连续排列，结束于173KiB。
OV2复用37/53/69KiB的区域，不随h_work的压缩移动。
DataT staging 为16KiB，与 h_fp32 和 h_work 不重叠，避免把16位矩阵原地扩宽到FP32。
常驻 OV1 mask [24,28) 不被这三个构件覆盖。所有区间结束地址低于248KiB通信保留边界。
HV1的HC1 FP32输入16KiB、HV2的HC2 FP32输入32KiB可选择Fixpipe直连UB或GM搬入；
OV2的OC2/OC3输入各16KiB同样可选择。各构件两个开关默认false，GM示例见下。
每 lane GM 搬运：HV1输入8KiB u、输出各8KiB v_new/update；
HV2在GM输入模式读取32KiB h_work，state常驻UB；输出16KiB h和/或16/32KiB state；
OV2输入两份16KiB Cube结果，输出8KiB o。

## 同步、地址所有权与调用约束

共同辅助头封装数据搬运事件及由调用方管理的 gate 准备，不改公共ABI或旧实现。
每个 AIV 在循环前初始化一次 T32 mask（24KiB，4KiB），只向 OV1 传入其 LocalTensor。
每 chunk 在复用许可到达后调用一次 gate.Prepare，搬入64个gate并按需转FP32；
OV1/HV1/HV2/OV2 都绑定同一个只读 Tensor，末尾所有读者完成后才可准备下一chunk。
HV2 先准备并衰减旧状态，等待 HC2 后相加，再按需 cast，最后搬出 FP32 state / DataT H，
统一一次 V_MTE3 和末尾 MTE3_V/MTE3_MTE2 等待。
局部事件每次都Set→Wait闭合，返回时MTE3、V及MTE2均无本构件遗留读写。
构件接收并等待对应Cube ready；HV1内部依次发布decayReadyForHc2和vnewReadyForOc3，
其余输出ready/free仍归调用方。两个HV1 ready在外部不得重复发布。

1. 调用HV1前HC1写目标改为37KiB；HV1等待hc1Ready后读取，返回时最后一次v_work读取及
   decay搬出完成后内部先发HC2 ready，随后计算v_new并搬出再发OC3 ready；
   仅HV1完整返回才允许复用WH/u_fp32，第一份ready不是输入free。
2. HC2的目的必须为141KiB，两个AIV按SPLIT_M分配64行。HV2先确认HV1返回，
   再用93KiB作为staging；等待hc2Ready后读取h_work，输出完成后返回。
3. HV2内部在GM/L1输出完成后发布hReadyForHc1，供下一chunk HC1读取；这不等于当前OV2已经完成。
4. OV2在HV1返回后才能写37/53KiB。等待oc2Ready/oc3Ready后才搬相应GM输入，
   计算并搬出O后，由OV2内部在MTE3上发布chunkFinished。
5. 下一chunk复用本表前必须等待chunkFinished；尤其不得让HV1与前chunk的OV2同时使用37KiB。
   一份ready/free信号的每个代次均需匹配消费，最终chunk不多发无人等待的代次。
6. 这些独立构件未接入driver；不能与旧HC1/HC2地址或旧epilogue资源混用。

有限合法输入沿用当前算子值域；累计非正gate确保相对指数非正，避免正向溢出。
不新增clip、不提前舍入；设备上需检查FP16/BF16转换及极小指数下溢的原精度规则。

## 验证计划与记录

参考：既有arch35 single-slot preload HV1/HV2及窄OV2，保留公式和FP32 state行为；
开发参考CHUNK_DEPENDENT/V1（来源a14961e9dfe8c8264768d67fb447101873da4ecf）。
验证包括普通C++语法/地址、各lane公式/状态链、首末chunk、无initial state与FP32 final_state，
以及已有文件指纹不变。目标CANN设备编译、A5 dtype组合精度、跨chunk同步及内存检查
必须在新driver接入后执行；本机无CANN/NPU，不声称设备已验收。


## 新文件与调用方式

- `block_epilogue_gdn_fused_vnew.hpp`：`GdnFusedMode2Vnew<DataT,GateT>`。
- `block_epilogue_gdn_fused_update.hpp`：`GdnFusedMode2Update<DataT,GateT,StateT>`。
- `block_epilogue_gdn_fused_output.hpp`：`GdnFusedMode2Output<DataT,GateT>`。
- `block_epilogue_gdn_fused_mode2_common.hpp`：调用方 gate.Prepare/Tensor、CopyRowsToL1Zn 和闭合的局部搬运事件。
- 局部事件号OV1=0、HV1=4、HV2=5、OV2=6、gate准备=7；新调用方不得对这些事件额外预置未消费token。

```cpp
using HV1 = Catlass::Epilogue::Block::GdnFusedMode2Vnew<INPUT_TYPE, G_TYPE>;
using HV2 = Catlass::Epilogue::Block::GdnFusedMode2Update<INPUT_TYPE, G_TYPE, STATE_TYPE>;
using OV2 = Catlass::Epilogue::Block::GdnFusedMode2Output<INPUT_TYPE, G_TYPE>;
using Ub = Catlass::Epilogue::gdnFusedMode2UbMallocTable;
using GateBuffer = Catlass::Epilogue::Block::mode2_fixed64_detail::GateBuffer<G_TYPE>;
GateBuffer gate(resource);
auto mask = resource.ubBuf.template GetBufferByByte<float>(Ub::mask);
mask.SetSize(32 * 32);
// Each AIV prepares its own identical, persistent T32 once, before the loop.
AscendC::ResetMask();
AscendC::Duplicate(mask, 0.0f, 32 * 32);
AscendC::PipeBarrier<PIPE_V>();
for (uint32_t row = 0; row < 32; ++row) {
    AscendC::Duplicate(mask[row * 32], 1.0f, row + 1);
}
AscendC::PipeBarrier<PIPE_V>();
// OV1 is the new Qkmask BlockEpilogue specialization, using the same UB table.
OV1 ov1(resource, mask, gate.Tensor());
HV1 hv1(resource, gate.Tensor());
HV2 hv2(resource, gate.Tensor());
OV2 ov2(resource, gate.Tensor());

for (/* each full chunk */) {
    // Wait for all consumers of the previous chunk before reusing this UB.
    // Resolve the current chunk's GM origins and state flags here.
    // h_ping was initialized by the driver before this loop (see below).
    gate.Prepare(gmG); // one GM load, at most one FP32 cast, local event 7
    // OV1 consumes OC1 ready internally before its QK GM load.
    ov1(gmMaskedQk, gmQk, oc1Ready);
    hv1(gmVNew, gmVUpdate, gmU, gmWH, hc1Ready, decayReadyForHc2, vnewReadyForOc3);
    // Both ready tokens were published inside HV1; do not publish them again.
    hv2(gmHNext, gmFinalState, gmHWork, hc2Ready, hReadyForHc1,
        isFinalChunk, storeFinalState);
    // HV2 published hReadyForHc1 internally; do not publish it again.
    ov2(gmO, gmOHWorkspace, gmOVWorkspace, scale, oc2Ready, oc3Ready, chunkFinished);
    // OV2 published chunkFinished internally; do not publish it again.
}
```

这不是完整driver示例：Cube侧发布OC1/HC1/OC2/HC2/OC3 ready，消费HV1 ready与chunkFinished，
所有代次仍须完整配对。独立构件无权自行选择调用方的flag ID，故不在内部发送free信号。

## 2026-10-19 本轮本地验证

- 新增 `tests/check_gdn_fused_mode2_epilogues_cpu.py`，编译并执行实际 OV1/HV1/HV2/OV2 及公共准备头文件，
  同步CPU桩仅替代Ascend API、寄存器和UB；开启AddressSanitizer/UndefinedBehaviorSanitizer，
  并禁止FMA合并。该检查不验证SDK API签名或设备流水语义。
- 六种DataT/GateT组合通过HV1、OV2公式对照；48组HV2三chunk场景覆盖FP32/DataT state、
  initial state有无、final state存储开关和两个lane；检查常驻OV1 mask的4KiB canary不变。
- 真实CANN编译、A5精度、内存及异步ready/free尚未执行；新源码未接入driver。

## 2026-10-19 mask/gate 所有权与冗余检查

- 删除 OV1 内部 InitializeMask、内部 mask 地址获取及 gate GM 搬运/cast。
- gate 每 chunk 只由调用方准备一次；HV1/HV2/OV2 不再重复 load/cast；OV2 删除原先仅等待 gate cast 的 PIPE_V 屏障。
- HV2 对两份独立输出只做一组 V→MTE3 和末尾 MTE3→V/MTE2 等待，FP32 状态更新与舍入语义保持不变。
- 保留 Broadcast 间工作区复用、临时 mask 组装、VF→cast 的 V 屏障及各阶段返回前的搬出完成等待。
  临时全宽 mask 与 Broadcast 共用29KiB工作区，故不能像常驻 T32 一样只初始化一次。
- CPU 仿真通过六种 dtype 组合、两个 AIV、三 chunk 的 OV1/HV1/OV2，核对 T32 只初始化一次且内容不变，
  gate 每 chunk 只搬入一次并至多转换一次且只读；48组 HV2 三 chunk 状态场景回归通过。
- 这些独立构件尚未接入原 driver；上面的调用示例给出新 driver 的初始化与循环位置，未修改旧路径。

## 2026-10-19 OV1 内部等待 OC1 ready

OV1 的 operator 新增 `Arch::CrossCoreFlag oc1Ready` 参数，在搬入 QK 前执行
`Arch::CrossCoreWaitFlag(oc1Ready)`。调用方传入该 chunk 的匹配信号，不再在外部重复消费。
Cube 应在对应 QK GM 写回完成后发布 ready。上一 chunk 的 workspace 释放仍由 driver 保证。
CPU 仿真对 QK 首次读取加入 guard，检查每次 OV1 正确消费传入 flag 一次且先等待再读取；
该检查不替代设备异步同步验收。

## 2026-10-19 OV1 独立计算提前与 RegBase 拆分

参照 `block_epilogue_gdn_fwdo_qkmask.hpp`：先完成两个 Broadcast、临时 mask 组装及
`PrepareGateMask(left, up, expandedMask)`，该 VF 只计算 gate 差值、Mins、Exp 和 causal 置零，
不接收 attn。随后才等待 OC1、搬入 QK，闭合 MTE2_V 后执行 `AscendC::Mul`，再 cast/搬出。
CPU 仿真在等待点核对完整衰减矩阵已经准备完成，并保留读取 QK 前必须等待的 guard。

## 2026-10-19 其余构件的依赖边界检查

| 构件 | 等待前完成 | 等待后完成 | RegBase 参数边界 |
| --- | --- | --- | --- |
| HV1/vnew | 发射 u 搬运；PrepareRelativeDecay 计算 exp(g[63]-g[:])；等待 u 搬运并 cast | 等 HC1 后，ComputeVnewDecay读取WH计算未舍入的decay，搬出并通知HC2，再Sub计算v_new、搬出并通知OC3 | 前者仅相对衰减和 raw gate；后者仅 u/WH/预计算衰减及输出，无 raw gate/Exp |
| HV2/update | 读取或初始化旧 h；ScaleState 计算 h*exp(g[63]) | 等 HC2 后 Add h_work，再 cast/写出 | ScaleState 不接收 h_work；Add 在外层 |
| OV2/output | PrepareDecay 计算 exp(g[:])；等 OC2 并搬入 inter 后 ScaleInter | 等 OC3 后才搬入 intra、Add、Muls(scale)、cast/写出 | PrepareDecay 不接收 inter/intra；ScaleInter 只接收 inter/衰减，不接收 intra/scale |

HV1 和 OV2 分别复用29KiB工作区开头256B存储64个FP32衰减，生命周期依次为
OV1 Broadcast/临时mask→HV1 relativeDecay→OV2 decay；不得同时执行。两个准备函数各用
一个64元素寄存器完成全部gate的Exp，后续按行寄存器广播，不额外物化32×128广播矩阵。
各阶段结束前仍完成其输出搬出；只移动等待位置，不改变 ready/free 的生产者、代次和返回保证。
HV2 的末块不存状态分支仍只消费 HC2 token；隐式零 FP32 初始状态没有输入 DMA，删除该分支的空 MTE2_V 同步。
公共 GateBuffer::Prepare 每 chunk 只执行一次，mask 初始化在循环外，未发现还需重复搬运或初始化的工作。

验证在 HC1 等待点核对 relativeDecay 和 u_fp32 已完成，在 HC2 等待点核对旧状态衰减已完成，
在 OC2 等待点核对 exp(g)，在 OC3 等待点核对 inter 衰减已完成；同时禁止 ready 之前读取
WH/h_work/GM inter/intra。六种 dtype、两个 lane、三 chunk 与48组状态场景通过 ASan/UBSan CPU仿真。
真实设备编译、异步流水正确性和性能收益仍待 NPU 验证。

## 2026-10-19 OV1 可选 Fixpipe 输入与 UB→L1 输出

新构造函数在 resource/mask/gate 后依次接收 `enableL0C2UB`、`enableUB2L1`，
两者默认false保持GM路径，最后接收L1输出LocalTensor（仅L1输出开启时必传）：

```cpp
// l1MaskedQk must denote the full 64x64 zN tile, not a lane-sliced pointer.
OV1 ov1(resource, mask, gate.Tensor(), true, true, l1MaskedQk);
ov1(gmMaskedQk, gmQk, oc1Ready);
```

| 输入开关 | 输出开关 | 路径 |
| --- | --- | --- |
| false | false | QK GM→UB，结果UB→GM |
| true | false | OC1 Fixpipe→UB，结果UB→GM |
| false | true | QK GM→UB，结果UB→L1 |
| true | true | OC1 Fixpipe→UB，结果UB→L1 |

gate广播、mask组装与PrepareGateMask始终先完成，再等待OC1。true输入分支不访问GM QK参数，
不执行QK的MTE2搬运或MTE2_V；false分支在OC1等待后执行这两步。
输出true不访问GM输出参数，false不访问L1参数；未使用的GM参数允许为未绑定GlobalTensor。
所有路径仍在返回前执行MTE3_V/MTE3_MTE2，调用方随后才可发布结果ready或允许输入UB复用。

直连输入契约：Cube将64x64 FP32 QK按SPLIT_M分给两个AIV，每lane32x64、行距64，
Fixpipe目的为每个AIV的 `Ub::attn_ping=0`，8KiB；OC1 ready在这些写入完成后发布。
该0..8KiB区域与提前执行的gate/mask计算分离。

L1输出契约：调用方提供共享且独占的64x64 zN tile（8KiB），元素偏移为
`(col/16)*64*16 + row*16 + col%16`。每lane写自身32行的4个16列panel，
复用参考fwdo_qkmask中按列panel的DataCopy方案，blockCount=32、blockLen=1、srcGap=3、dstGap=0。
后续Cube应按同样64x64物理布局直接读取；128x128 L1槽的物理行距不同，不能直接传入。
Cube消费完L1后才能复用，两个AIV输出完成的聚合ready由driver负责。

本轮未修改Cube/driver：使用true开关前必须配套上述输入地址、输出布局和双向ready/free。
CPU仿真覆盖4种组合×6种dtype×2个lane×3个chunk；直连分支传空GM参数，
检查没有GM读取与MTE2_V，验证L1布局、边界canary、另一lane未被覆盖及输出搬出等待。
真实Fixpipe、UB→L1 API和跨核设备行为仍需CANN/NPU验证。

## 2026-10-19 HV1/HV2/OV2 可选直连完善

三个构造函数在resource/gate后统一接收两个bool：enableL0C2UB、enableUB2L1，均默认false；
随后接收调用方预留的完整L1目标tensor，不能传lane切片。true输入跳过对应Cube结果的GM搬运/MTE2_V，
仍消费ready；false在ready后主动搬入一次。true输出写L1，false写GM，所有输出搬完才返回。

| 构件 | true输入的Fixpipe目的（每AIV） | true输出的L1物理布局 | false输入新增GM参数 |
| --- | --- | --- | --- |
| HV1 | v_work_ping=37KiB，32x128 FP32，stride128 | v_new与v_update各64x128 zN、各16KiB，两块独占槽 | whInput，64x128 FP32起点 |
| HV2 | h_work_ping=141KiB，64x128 FP32，stride128 | 非末chunk H为128x128 zN、32KiB | hWorkInput，128x128 FP32起点 |
| OV2 | o_inter_ping=37KiB、o_intra_ping=53KiB，各32x128 FP32，stride128 | O为64x128 zN、16KiB | 既有interInput/intraInput |

L1通用助手CopyRowsToL1Zn按`(col/16)*FullRows*16+row*16+col%16`寻址，
每AIV写半数行，DataCopy blockLen=1、srcGap=7、dstGap=0。
完整64行与128行物理布局不能互换。L1空间由driver预留，未修改既有L1内存表。

```cpp
// Both HV1 results feed Cube consumers via separate L1 slots.
HV1 hv1(resource, gate.Tensor(), true, true, l1Vnew, l1VUpdate);
HV2 hv2(resource, gate.Tensor(), true, true, l1HNext);
// Final O normally goes to GM; passing true,true,l1O is also supported.
OV2 ov2(resource, gate.Tensor(), true, false);
hv1({}, {}, gmU, {}, hc1Ready, decayReadyForHc2, vnewReadyForOc3);
hv2({}, gmFinalState, {}, hc2Ready, hReadyForHc1,
    isFinalChunk, storeFinalState);
ov2(gmO, {}, {}, scale, oc2Ready, oc3Ready, chunkFinished);
```

HV2的H下一chunk路径与final_state路径分开：enableUB2L1仅改变非末chunk的H输出；
按需保存的final_state仍按原dtype写GM，末chunk且不保存仍只消费HC2 token。
h_ping的FP32结果在GM/L1两种输出路径下都跨chunk保留、原地更新。DataT H副本供Cube使用，
不再读回或舍入Vec的FP32 state；final_state也只作为输出，不参与恢复。
每AIV必须沿用同一UB资源，h_ping不得被其它stage覆盖；切换task时由driver重新初始化。

独立计算顺序保持HV1 PrepareRelativeDecay/u cast→HC1 wait、HV2 ScaleState→HC2 wait、
OV2 PrepareDecay→OC2 wait→ScaleInter→OC3 wait。准备完成之前不提前等待，ready之前不读取Cube输入。
所有未被选择的GM输入/输出可传未绑定tensor；HV2只在GM输入模式读取hWorkInput，final_state按需写出。

直连模式对生产者增加明确的地址释放约束：OC2/OC3写37/53KiB前必须等HV1彻底返回并释放相应区域，
不能仅依靠OV2内部ready等待，因为Fixpipe可能提前覆盖HV1的WH/u。HC2写141KiB和HC1写37KiB也必须
遵守各自上一代读者的free；L1消费者必须等待两个AIV输出完成，消费后才允许下一代覆盖。
这些新构件尚未接入Cube/driver，不改变旧路径；使用开关前必须配套相同地址、布局和ready/free。

本轮CPU仿真通过4种路径×6种dtype×2个AIV×3个chunk；HV2扩展至192组状态场景，
验证空GM参数、条件MTE2_V、搬出等待、L1边界和另一lane保持、跨chunk常驻FP32状态链、
末块/无状态跳过及各等待点独立计算完成。同步CPU桩不验证真实SDK签名或异步流水，
CANN构建、NPU精度和性能仍待driver接入后验证。

## 2026-10-19 HV1 decay优先、分段通知

HV1调用在hc1Ready之后新增两个必传输出flag：vnewDecayReadyForHc2和vnewReadyForOc3。
每个AIV依次执行以下完整步骤，GM与L1输出路径保持相同顺序：

1. 先准备relativeDecay、搬入/cast u，再等待HC1；GM fallback随后搬入WH。
2. ComputeVnewDecay只将`(u_fp32-WH)*exp(g[63]-g[row])`写入decay FP32槽，
   不物化或写出v_new；Sub中间值保持FP32，不能先舍入。
3. Cast decay至已结束的u槽[53,61)KiB，搬出到GM vUpdate或L1 vUpdate。
   闭合V_MTE3和MTE3_V/MTE3_MTE2后，通过`CrossCoreSetFlag<0x2,PIPE_MTE3>`通知HC2。
4. 再用AscendC::Sub单独计算v_new=u_fp32-WH，Cast至同一[53,61)KiB，搬出到GM或L1 v_new。
   再次完成搬出等待后，发布OC3 ready。

第一段不能复用u_fp32的[61,77)KiB，否则会覆盖第二段输入。WH [37,53)KiB和u_fp32
在第一份HC2 ready后仍存活；该ready只表示decay可读，绝不能作为这些输入区域的free。
HC2可使用独立的h_work [141,173)KiB目的运行；覆盖37/53KiB的OC2/OC3 Fixpipe仍须等HV1释放。
OC3还依赖OV1 maskedQK，调用方必须保证OV1结果已就绪。两个AIV应按driver协议匹配消费，
最终chunk的HC2跳过路径也必须消费对应decay token；返回后不得额外重复发布这两个ready。

验证命令：`python3 tests/check_gdn_fused_mode2_epilogues_cpu.py --skip-qkmask`。
覆盖六种dtype、四种路径、两个lane、三chunk，分别在HC2/OC3通知点核对结果已搬出，
要求HC2通知时v_new槽与目的尚未改动、第二次Sub必须在HC2通知之后执行，
u_fp32保持完整，并模拟HC2在第一份通知后写独立h_work区域。HV2的192组状态场景一并回归通过。
全量脚本未通过：当前qkmask末尾的MTE3_V/MTE3_MTE2已被注释，与原断言不一致；
本次不修改该文件，新增显式skip选项仅用于隔离HV1/HV2/OV2，不声称全量验证通过。
真实CANN/NPU编译与异步重叠仍待验证；旧driver尚未接入本新构件。

## 2026-10-19 同步代码原地展开

删除公共头中的三种局部事件包装函数，所有调用处（包括GateBuffer::Prepare）直接写出
AscendC::SetFlag/WaitFlag。事件类型、编号、条件分支和顺序保持不变，便于沿计算和搬运代码阅读。

## 2026-10-19 局部事件使用AscendC命名

所有新构件的局部SetFlag/WaitFlag直接使用AscendC EVENT_ID常量：OV1=EVENT_ID0、
HV1=EVENT_ID4、HV2=EVENT_ID5、OV2=EVENT_ID6、GateBuffer准备=EVENT_ID7。
删除kEvent别名及GateBuffer中的裸数字7，原事件分配和顺序保持不变。
跨核ready继续使用调用方传入的CrossCoreFlag，属于独立的信号参数。

## 2026-10-19 初始state由driver在循环前准备

HV2不再接收initialState、useInitialState、hInput、isInitialState参数，也不清零、搬入或恢复state。
所有chunk直接使用h_ping的FP32内容，先原地衰减，再等待HC2并累加h_work。
DataT输出由另一个staging地址承接；Vec的FP32递推不依赖storeFinalState或输出位置。
调用方在每个task的chunk循环前、每个AIV中准备一次，例如：

```cpp
using Ub = Catlass::Epilogue::gdnFusedMode2UbMallocTable;
constexpr uint32_t elements = 64 * 128;
const uint32_t offset = AscendC::GetSubBlockIdx() * elements;
auto h = resource.ubBuf.template GetBufferByByte<float>(Ub::h_ping);
auto staging = resource.ubBuf.template GetBufferByByte<INPUT_TYPE>(Ub::v_new_decay_ping);
AscendC::ResetMask();
if (!useInitialState) {
    AscendC::Duplicate(h, 0.0f, elements);
} else {
    if constexpr (std::is_same_v<STATE_TYPE, float>) {
        AscendC::DataCopy(h, gmInitialState[offset], elements);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID5);
    } else {
        // STATE_TYPE is DataT. Only this initial load needs expansion to FP32.
        AscendC::DataCopy(staging, gmInitialState[offset], elements);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID5);
        AscendC::Cast(h, staging, AscendC::RoundMode::CAST_NONE, elements);
    }
}
AscendC::PipeBarrier<PIPE_V>();
// Keep h_ping intact until the first HV2; initialize Cube's H operand separately.
// The loop uses: hv2(gmHNext, gmFinalState, gmHWork, hc2Ready, hReadyForHc1,
//                   isFinalChunk, storeFinalState);
```

driver同时负责首个HC1/OC2所用H的GM/L1初态及ready，不可只初始化Vec h_ping。
现有driver尚未接入这些独立构件，本轮仅调整新HV2与调用契约，并给出循环前准备示例。
CPU仿真在循环前准备FP32初态，每chunk核对未舍入的UB结果，并破坏GM副本以检查后续chunk不读回state。
此处明确采用Vec常驻FP32递推，替代早期示例中的DataT舍入后恢复方式；旧实现未修改。


## 2026-10-19 HV2负责发布H ready

HV2调用新增hReadyForHc1参数。非末chunk完成H的GM/L1搬出及本地搬出等待后，
通过`Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(hReadyForHc1)`通知下一HC1。
末chunk不发布该token，包括保存最终state和直接跳过两种情况；调用方不再重复set。
每个AIV发布自身完成信号，Cube侧沿用相匹配的跨核等待协议；ready不替代目的槽覆盖前的free。


## 2026-10-19 按每AIV状态分块收紧UB分配

H整体为128×128，两AIV各处理64×128。每AIV h_ping=[109,141)KiB，
h_work_ping=[141,173)KiB；原来的h_work起点173KiB改为141KiB，总占用结束从237KiB降到173KiB。
staging仍复用[93,109)KiB；37KiB同时承接OC2直连输出，不额外引入与HV2暂存的重叠。
OV2沿用o_inter=[37,53)、o_intra=[53,69)、o_out=[69,77)KiB。
新driver接入时HC2 Fixpipe目的也必须引用同一内存表的新h_work_ping地址。


## 2026-10-19 OV2生产者完成通知

OV2调用末尾新增chunkFinished参数。GM/L1输出指令之后，内部执行
`Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(chunkFinished)`，调用方不再重复set。
每次调用均发布一次，包含末chunk；driver需配套管理flag消费和task边界，避免遗留token。
保留现有V_MTE3与MTE3_V，未恢复被注释的MTE3_MTE2。


## 2026-10-19 新schedule与driver接入

固定64整chunk的新路径现已接入`RunSingleCoreChunkFused`。不满足新路径条件时仍进入旧driver，
未改host ABI或workspace分配。新文件：

- `gemm/block/block_scheduler_gdn_fwd_ho_fused_mode2.hpp`：uint64偏移、GQA分组、首末chunk标记、单槽workspace。
- `gemm/block/block_mmad_gdn_fused_mode2.hpp`：公共preload BlockMmadTla的外部L1输入适配，独立GM/L1输入和GM/UB输出。
- `gemm/kernel/gdn_fwd_ho_fused_mode2_kernel.hpp`：初始化、五Cube/四Vec调度及退出同步。

以上相对路径以`op_kernel/arch35`为根。模板开关`CubeToUB`和`VecToL1`独立，当前默认分别为true和false。
最终O及final_state始终输出GM，H/Vnew是内部值，L1路径不再物化其GM副本。
H在Vec中始终FP32原地递推，Cube读取转换后的DataT副本。

| 阶段 | 输入依赖 | 结果位置（当前默认） | 发布信号 | 消费者 |
| --- | --- | --- | --- | --- |
| driver初态 | initial_state或零 | UB H与GM H | hReady | 首HC1 |
| OC1 | Q、K | UB 0KiB | oc1Ready | OV1 |
| HC1 | hReady、W、H | UB 37KiB | hc1Ready | HV1 |
| OC2 | Q、同代H | UB 173KiB | oc2Ready | OV2 |
| OV1 | gate、mask、oc1Ready | GM masked QK | 由后续HV1的MTE3 vnewReady一并覆盖 | OC3 |
| HV1 decay | hc1Ready、U、gate | GM decay | decayReady | HC2 |
| HC2 | decayReady、K | UB 141KiB | hc2Ready | HV2 |
| HV1 Vnew | U、WH | GM Vnew | vnewReady | OC3 |
| HV2 | hc2Ready、常驻H | UB H与GM H | 非末chunk hReady | 下一HC1 |
| OC3 | vnewReady、masked QK、Vnew | UB 189KiB | oc3Ready | OV2 |
| OV2 | oc2Ready、oc3Ready | GM O | chunkFinished | 下一chunk入口或末尾退出 |

L1 KiB区间：scratch=[0,64)，H=[64,96)，masked QK=[96,104)，
Vnew=[104,120)，decay=[120,136)。L1布局以实际M/N/K构造，匹配Vec的FullRows，
不能用128×128的layout代替64×64的masked QK。Cube块逐次调用公共preSetFlags/operator()/finalWaitFlags，闭合scratch/L0事件后才复用同一资源。

Cube顺序为OC1→HC1→OC2→HC2→OC3；Vec顺序为OV1→HV1→HV2→OV2。
Cube从第二chunk起先等待上一OV2完成，末尾额外等待最后一代完成，故OV2发布的token全部被消费。
初态只发布H-ready，不伪造OV2完成；H-ready共发布numChunks次，包含初始化与numChunks-1次HV2。
末chunk不保存state时HC2仍消费decayReady并发布hc2Ready，HV2仅消费该token。

HV1两次cast共用53KiB，不能用跨chunk链证明同chunk复用安全。因此第一段搬出后set MTE3_V，
独立Sub之后、第二次cast之前wait；其余被注释的drain未恢复。HV2仍无逐chunk本地drain。
driver末尾以EVENT_ID5等待MTE3完成，覆盖最后state输出和初始化边界。

测试入口：`python3 tests/check_gdn_fused_mode2_driver_cpu.py`，以三个host线程执行实际driver、
实际公共preload BlockMmadTla、Mode2 L1适配和实际epilogue；只替换设备API、TLA基础类型、TileCopy和TileMmad为CPU实现。
覆盖四种搬运组合、首末chunk、初态/终态开关、公开dispatch可达dtype组合，核对O/state参考值与信号收支。
多batch/GQA在CPU联调中逐task验证，64位大偏移由scheduler检查覆盖；不宣称这些shape已运行设备整算子。
当前无CANN/NPU，真实SDK类型匹配、跨核可见性、mssanitizer和性能必须在设备补验。


## 2026-10-19 Cube改为复用公共preload实现

原Mode2块只复用了TileMmad类型，自己编排搬运和本地事件，现已移除这部分实现。
五个阶段均通过继承的operator()实际执行
`common/kernel_utils/block/block_mmad_pingpong_tla_preload.hpp`。
Mode2适配只绑定公共块的L1输入槽，并对外部L1操作数跳过GM→L1；普通GM输入仍由公共块搬运。
不修改公共头文件或旧driver。

各阶段精确tile为OC1=64/64/128、HC1与OC2=64/128/128、HC2=128/128/64、
OC3=64/128/64（顺序M/N/K）。外部L1的zN行距必须与Vec写入的FullRows一致。
L1A/L1B/L0A/L0B/L0C阶段数均为1，与单槽资源规划一致；关闭GM resident缓存。
每次调用前preset、调用后drain，使不同形状实例共享scratch/L0时事件成对回收。
跨核ready仍由driver在对应阶段后发布。

CPU联调现直接包含仓库的公共preload头，而非用stub替换该头。
256场景通过；Cube本地事件额外检查重复set、无set的wait和退出残留，
并核对公共块B/C槽的preset/release次数。设备异步行为仍需CANN/NPU验证。


## 2026-10-19 整体复查及默认路径调整

按用户要求将driver模板默认设为CubeToUB=true、VecToL1=false。
默认五个Cube阶段Fixpipe写UB；OV1/HV1/HV2的Cube输入操作数写GM，公共preload再搬入L1。
可选VecToL1=true路径保留。

非首chunk在gate.Prepare之前增加V_MTE2 EVENT_ID7：防止新的gate DMA提前覆盖
上一chunk仍由Vector读取的数据。OV2已有的MTE3_V位于该V→MTE2依赖之前，
也覆盖跨chunk复用的MTE3源；没有恢复HV1/HV2内被移除的逐次drain。
HV2的FP32 final_state仅在末chunk写出；移除重复的isFinalState判断。
修正OV1异步返回及UB表中过时的地址复用注释。

CPU联调扩展到304场景：原256场景、16个实际默认模板实例场景，
以及B=2/HK=2/HV=8的16个task分别走GM/L1两条输出路径共32场景。
逐task启动一个Cube与两个Vec线程，不模拟不同Cube核心并发；核对O/state、GM H/Vnew
快照和其它task区域未被修改，检查直连输入确实跳过GM搬运。
所有测试仍属于CPU开发验证，设备编译及异步流水需在CANN/NPU环境补验。
