# ChunkFwdHOFused 方案设计

方案设计规则版本：`V2`

本文描述当前仓库实现。公开接口契约以 [`api.md`](api.md) 为准；本文中的函数原型、
参数顺序、TilingData 字段顺序、workspace 分配和架构分支均与当前源码一致。

## 1. 目标、计算语义与实现边界

`ChunkFwdHOFused` 在一次 AICore kernel 下发中完成 H 状态递推和 O 输出计算。
中间张量 `h`、`v_new` 位于 user workspace，不作为公开输出。

令 `B/T/HK/HV` 分别为 batch、序列长度、K/Q head 数和 value head 数，
`K=128`，`V in {128,256}`，`C=chunkSize in {64,128}`，`NC=ceil(T/C)`，
`R=HV/HK`，`P=B*HV`，`hk=floor(hv/R)`。chunk 有效区间记为 `[s,e)`。

H 阶段实现：

```text
h[b,hv,chunk] = state_before_chunk
v_new[s:e] = u[s:e] - w[s:e] @ state_before_chunk
state_after_chunk = state_before_chunk * exp(g[e-1])
                  + k[s:e]^T @ (v_new[s:e] * exp(g[e-1] - g[s:e]))
```

O 阶段实现：

```text
A = q[s:e] @ k[s:e]^T
A = tril(A * gate_fn(g[s:e,None] - g[None,s:e]))
state_term = (q[s:e] * gate_fn(g[s:e])) @ h[b,hv,chunk]
o[s:e] = scale * (state_term + A @ v_new[s:e])
```

当前实现只接受定长输入；只要 `cu_seqlens` 和 `chunk_indices` 存在，Host tiling
立即返回 `GRAPH_FAILED`。

### 1.1 架构分支

| 架构 | Host 识别值 | `blockDim` | H/O 关系 | 同步方式 |
| --- | --- | --- | --- | --- |
| Atlas A2 | `DAV_2201` | `2*P` | 前 `P` 个 MIX 核运行 H，后 `P` 个运行 O | 启动时清空 IB 区并 `SyncAll`；逐 chunk `IBSet/IBWait` |
| Ascend 950 | `DAV_3510` | `physicalCoreNum` | 所有核先完整运行 H，再完整运行 O | H 返回后 `SyncAll<false>()` |

A2 要求 `2*P < physicalCoreNum`，并设置 `scheduleMode=1`。A5 使用全部物理 AIC。

### 1.2 当前未闭合的范围

Host tiling 中针对 SoC 的 `useExp2/dtype/C/K/V/R` 校验当前被注释，所以代码实际
放行范围宽于 `api.md` 声明的设备路径：

- A2 的 O-stage 投影不携带 `useExp2` 和 `outputLayout`，有效设计范围仍是自然指数
  的 `BNSD/NTD` 路径。
- A5 的 arch35 模板固定面向 BF16、`C=64`、`K=V=128`，Host 当前仍可能放行
  FP16、`C=128` 或 `V=256`。
- `V=256` 会生成 `TilingKey=2`，但 A5 入口不按 TilingKey 分派，仍进入固定的
  arch35 路径。

这些是当前实现缺口，不代表新增支持范围。

## 2. Stage 与设备执行路径

### 2.1 Stage 0：Host 构造执行任务和 tiling

ACLNN 第一阶段完成参数校验、连续化、可选状态转置和 executor 创建。L0 将
`aclIntArray` 转为 INT64 tensor；未请求 `final_state` 时创建 shape `[0]` 的内部
占位 tensor，然后调用 `ADD_TO_LAUNCHER_LIST_AICORE`。

Host tiling 依次完成：属性和逻辑 shape 校验、定长/layout/dtype/storage shape
校验、架构识别、核数计算、TilingData 填充、workspace 计算，以及 TilingKey、
blockDim、schedule mode 和 workspace size 设置。

### 2.2 Stage 1：A2 H 生产者

逻辑 MIX 核 `p in [0,P)` 固定负责 `b=p/HV, hv=p%HV` 的全部 chunk。H kernel
使用两个 stream slot；`vWorkspaceOffset`、`vUpdateWorkspaceOffset`、
`kDecayWorkspaceOffset` 和 `hWorkspaceOffset` 是其双流水临时区。每个 chunk 写：

- `handoffH[b,hv,chunk,:,:]`；
- `handoffV[b,hv,s:e,:]`；
- 可选 `final_state[b,hv,:,:]`。

两个 AIV 分别完成自己的切片后，以 `eventId=chunk%2` 调用 `IBSet<false>`。

### 2.3 Stage 2：A2 O 消费者

逻辑 MIX 核 `P+p` 固定消费生产者 `p` 的全部 chunk，workspace 索引使用
`workspaceCoreIdx=p`。每个 chunk 的内部流水为：

1. Cube1：`Q @ K^T` 写 attention workspace。
2. Vector1：AIV 等待 H 的 `IBWait`，完成 gate、causal mask 和指数处理。
3. Cube2：`Q @ H` 写 O 的 H 临时区。
4. Cube3：masked attention `@ v_new` 写 O 的 V 临时区。
5. Vector2：合并 Cube2/Cube3，乘 `scale` 并写出 `o`。

O 使用两个 ping/pong slot。`cube1Done/vec1Done/cube3Done/vec2Done` 使用
CrossCore flag `0..7`。两个消费者 AIV 完成 `IBWait` 后反向发布 `cube1Done`，
AIC 在读取完整 H/V tile 前等待聚合结果。

### 2.4 Stage 1A/2A：A5 顺序执行

A5 的 `RunChunkFwdHOFusedA5` 先调用 arch35 H，将 `h` 和 `v_new` 写入完整交接区；
H 返回后所有 AIC/AIV 执行 `SyncAll<false>()`，随后构造 `ChunkFwdOTilingData`
并调用 arch35 O。A5 不使用 A2 的 IB 区和 O 临时区；O 只使用每物理核一块
A-prime workspace。

## 3. TilingData 定义和 ABI

### 3.1 枚举与常量

```cpp
namespace GDN {
enum class ChunkFwdHOFusedDtype : int64_t { FP16 = 0, BF16 = 1, FP32 = 2 };
enum class ChunkFwdHOFusedOutputLayout : int64_t {
    BNSD = 0, BSND = 1, TND = 2, NTD = 3
};
enum class ChunkFwdHOFusedTilingKey : uint64_t { V128_EXP = 1, V256_EXP = 2 };
constexpr int64_t CHUNK_FWD_HO_AIV_PER_MIXED_CORE = 2;
constexpr int64_t CHUNK_FWD_HO_IB_EVENT_COUNT = 2;
constexpr int64_t CHUNK_FWD_HO_IB_WORDS_PER_EVENT = 8;
}
```

枚举名中的 `EXP` 是现有代码名称，不表示 Host 已完成 SoC 的 exp/exp2 限制。

### 3.2 Host 序列化结构

以下字段顺序是 ABI，不能重排：

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

REGISTER_TILING_DATA_CLASS(ChunkFwdHOFused, ChunkFwdHOFusedTilingData)
```

kernel 侧 `GDN::ChunkFwdHOFusedTilingData` 是逐字段同序的普通 C++ 镜像。Host
在栈上构造宏生成的 `ChunkFwdHOFusedTilingData`，完成字段和 workspace offset
填充后，通过 `SaveToBuffer` 序列化到 `context->GetRawTilingData()`，再设置实际
数据长度。Host 同时检查
`tiling.GetDataSize() == sizeof(GDN::ChunkFwdHOFusedTilingData)`；该检查只能发现
总大小不一致，不能发现总大小相同但字段顺序不同。

### 3.3 字段来源和消费方

| 字段组 | Host 来源 | 主要消费方 |
| --- | --- | --- |
| `batch` 至 `chunkSize` | L0 逻辑属性 | H/O scheduler |
| `useInitialState,storeFinalState` | descriptor / attr | H kernel |
| `dataType,gDataType,stateDataType` | input descriptor | H/O dtype dispatch |
| `isVariedLen,shapeBatch,tokenBatch` | optional shape | H/O scheduler |
| `useG,useGk` | `true` / optional descriptor | H dispatch |
| H workspace offsets | `FillWorkspace*` | H kernel |
| `useExp2,outputLayout,scale` | attributes | A5 O；A2 O 只投影 `scale` |
| chunk/head/core 派生字段 | shape 和 platform | A5 O、入口分流和调度 |
| handoff/sync offsets | `FillWorkspace*` | fused 入口、A2 IB |
| O workspace offsets | `FillWorkspace` | A2 O kernel |
| `oAPrimeWorkspaceOffset` | `FillWorkspaceA5` | arch35 O |

`consumerCoreBase` 当前会写入，但 A2 入口实际以
`GetMixedCoreIdx() < producerCoreNum` 分流，没有直接读取该字段。

### 3.4 H 前缀和 O 投影

H kernel 将 tiling GM 指针解释为 `ChunkFwdHOFusedHStageTilingData*`，因此完整结构
从 `batch` 到 `numChunksWorkspaceOffset` 必须构成严格前缀：

```cpp
static_assert(offsetof(ChunkFwdHOFusedTilingData, useExp2) ==
              sizeof(ChunkFwdHOFusedHStageTilingData));
```

O 不直接重解释完整结构：A2 投影为 `ChunkFwdHOFusedOStageTilingData`，A5 投影
为 `ChunkFwdOTilingData`。这两个结构不是完整 TilingData 的 ABI 镜像，必须逐字段
赋值，不能用裸指针转换代替。

A2 O-stage 投影的实际定义为：

```cpp
struct ChunkFwdHOFusedOStageTilingData {
    int64_t shapeBatch;
    int64_t seqlen;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    int64_t isVariedLen;
    int64_t tokenBatch;
    int64_t dataType;
    int64_t gDataType;
    int64_t vWorkspaceOffset;
    int64_t hWorkspaceOffset;
    int64_t attnWorkspaceOffset;
    int64_t aftermaskWorkspaceOffset;
    int64_t maskWorkspaceOffset;
    float scale;
    int64_t pipelineSyncWorkspaceOffset;
};
```

A5 arch35 O-stage 投影的实际定义为：

```cpp
struct ChunkFwdOTilingData {
    int64_t shapeBatch;
    int64_t seqlen;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    int64_t isVariedLen;
    int64_t tokenBatch;
    int64_t dataType;
    int64_t gDataType;
    int64_t vWorkspaceOffset;
    int64_t hWorkspaceOffset;
    int64_t attnWorkspaceOffset;
    int64_t aftermaskWorkspaceOffset;
    int64_t maskWorkspaceOffset;
    int64_t stateVFirst;
    int64_t outputLayout;
    float scale;
    int64_t chunkNum;
    int64_t hvPerHk;
    int64_t taskGroupSize;
    int64_t numChunksPerBatch;
    int64_t aPrimeWorkspaceOffset;
};
```

A2 `FillOTiling` 把 `oV/oH/oAttn/oAfterMask/oMask` 映射到上述通用 offset；A5
`FillOTiling` 将这些通用临时 offset 和 `stateVFirst` 置零，只传递
`outputLayout`、chunk/head 派生字段及 `oAPrimeWorkspaceOffset`。

## 4. Host tiling 接口和索引

### 4.1 固定索引

```cpp
// Inputs
K=0, Q=1, W=2, U=3, G=4, GK=5, INITIAL_STATE=6,
CU_SEQLENS=7, CHUNK_INDICES=8

// Outputs
O=0, FINAL_STATE=1

// Attributes
OUTPUT_FINAL_STATE=0, CHUNK_SIZE=1, SCALE=2, USE_EXP2=3,
OUTPUT_LAYOUT=4, LOGICAL_BATCH=5, LOGICAL_SEQLEN=6,
LOGICAL_K_HEADS=7, LOGICAL_V_HEADS=8, LOGICAL_K_DIM=9,
LOGICAL_V_DIM=10
```

这些索引必须匹配 OpDef、L0 的 `OP_INPUT/OP_OUTPUT/OP_ATTR` 顺序和已安装 OPP。

### 4.2 Tiling 函数

```cpp
ge::graphStatus Tiling4ChunkFwdHOFused(gert::TilingContext *context);
ge::graphStatus TilingPrepareForChunkFwdHOFused(
    gert::TilingParseContext *context);

IMPL_OP_OPTILING(ChunkFwdHOFused)
    .Tiling(Tiling4ChunkFwdHOFused)
    .TilingParse<ChunkFwdHOFusedCompileInfo>(TilingPrepareForChunkFwdHOFused);
```

辅助接口：

```cpp
ge::graphStatus ValidateTensorContracts(
    gert::TilingContext *context, int64_t batch, int64_t seqlen,
    int64_t kHeads, int64_t vHeads, int64_t kDim, int64_t vDim,
    int64_t outputLayout);

ge::graphStatus FillWorkspace(
    ChunkFwdHOFusedTilingData &tiling, size_t systemWorkspace,
    size_t producerCoreNum, bool useGk, size_t &workspaceSize);

ge::graphStatus FillWorkspaceA5(
    ChunkFwdHOFusedTilingData &tiling, size_t systemWorkspace,
    size_t physicalCoreNum, size_t taskNum, bool useGk,
    size_t &workspaceSize);
```

## 5. Workspace 布局

全部 offset 相对于 `AscendC::GetUserWorkspace(workspace)`，每个区域起点按 512
bytes 对齐。最终返回 `systemWorkspace + alignedUserWorkspace`。以下 `E=2`、
`F=4`、`S=2` 分别表示输入元素字节数、FP32 字节数和 ping/pong slot 数。

### 5.1 A2 分配顺序

| 字段 | 原始字节数 |
| --- | --- |
| `handoffHWorkspaceOffset` | `P*NC*K*V*E` |
| `handoffVWorkspaceOffset` | `P*T*V*E` |
| `pipelineSyncWorkspaceOffset` | `activeCoreNum*2*2*8*4` |
| `vWorkspaceOffset` | `P*C*V*F*S` |
| `vUpdateWorkspaceOffset` | `P*C*V*F*S` |
| `kDecayWorkspaceOffset` | `P*C*K*F*S`，仅有 `gk` 时分配 |
| `hWorkspaceOffset` | `P*K*V*F*S` |
| `numSeqWorkspaceOffset` | `(batch+1)*sizeof(int64_t)` |
| `numChunksWorkspaceOffset` | `(batch+1)*sizeof(int64_t)` |
| `oVWorkspaceOffset` | `P*C*V*F*S` |
| `oHWorkspaceOffset` | `P*C*V*F*S` |
| `oAttnWorkspaceOffset` | `P*C*C*F*S` |
| `oAfterMaskWorkspaceOffset` | `P*C*C*F*S` |
| `oMaskWorkspaceOffset` | `C*C` |

`oAPrimeWorkspaceOffset=0`。没有 `gk` 时 `kDecayWorkspaceOffset` 指向当前位置但不
推进 offset，H 的非 GK 模板不消费该区域。

### 5.2 A5 分配顺序

| 字段 | 原始字节数 |
| --- | --- |
| `handoffHWorkspaceOffset` | `P*NC*K*V*E` |
| `handoffVWorkspaceOffset` | `P*T*V*E` |
| `vWorkspaceOffset` | `physicalCoreNum*C*V*F*S` |
| `vUpdateWorkspaceOffset` | 同上 |
| `kDecayWorkspaceOffset` | `physicalCoreNum*C*K*F*S`，仅有 `gk` 时分配 |
| `hWorkspaceOffset` | `physicalCoreNum*K*V*F*S` |
| `numSeqWorkspaceOffset` | `(batch+1)*sizeof(int64_t)` |
| `numChunksWorkspaceOffset` | `(batch+1)*sizeof(int64_t)` |
| `oAPrimeWorkspaceOffset` | `physicalCoreNum*CHUNK_FWD_O_APRIME_WORKSPACE_BYTES` |

A5 将 `pipelineSyncWorkspaceOffset` 和全部 A2 O workspace offset 置零。

## 6. 函数接口和参数顺序

### 6.1 ACLNN 两阶段接口

```cpp
aclnnStatus aclnnChunkFwdHOFusedGetWorkspaceSize(
    const aclTensor *k, const aclTensor *q, const aclTensor *w,
    const aclTensor *u, const aclTensor *g, const aclTensor *gkOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool outputFinalState, int64_t chunkSize, double scale,
    bool useExp2, bool stateVFirst, const char *outputLayout,
    const aclTensor *oOut, const aclTensor *finalStateOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnChunkFwdHOFused(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```

第一阶段构建 executor/workspace；第二阶段通过 `CommonOpExecutorRun` 执行任务。

### 6.2 L0 接口与 launcher

```cpp
const std::array<const aclTensor *, 2> ChunkFwdHOFused(
    const aclTensor *k, const aclTensor *q, const aclTensor *w,
    const aclTensor *u, const aclTensor *g, const aclTensor *gkOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool outputFinalState, int64_t chunkSize, double scale, bool useExp2,
    const char *outputLayout, const aclTensor *oOut,
    const aclTensor *finalStateOut, aclnnStatus *status,
    aclOpExecutor *executor);
```

```cpp
ADD_TO_LAUNCHER_LIST_AICORE(
    ChunkFwdHOFused,
    OP_INPUT(k, q, w, u, g, gkOptional, initialStateOptional,
             actualCuSeqlens, actualChunkIndices),
    OP_OUTPUT(oOut, finalStateOutKernel),
    OP_ATTR(outputFinalState, chunkSize, scale, useExp2, outputLayoutStr,
            logicalBatch, logicalSeqlen, logicalKHeads, logicalVHeads,
            logicalKDim, logicalVDim));
```

### 6.3 AICore kernel ABI

```cpp
extern "C" __global__ __aicore__ void chunk_fwd_h_o_fused(
    GM_ADDR k, GM_ADDR q, GM_ADDR w, GM_ADDR u, GM_ADDR g,
    GM_ADDR gk, GM_ADDR initial_state, GM_ADDR cu_seqlens,
    GM_ADDR chunk_indices, GM_ADDR o, GM_ADDR final_state,
    GM_ADDR workspace, GM_ADDR tiling);
```

kernel ABI 与 `OP_INPUT/OP_OUTPUT` 完全一致。内部 helper 为复用 H 实现改成
`k,w,u,g,gk,initialState,q,...` 顺序，这只是内部顺序，不能反向修改公开 ABI。

入口先执行：

```cpp
REGISTER_TILING_DEFAULT(GDN::ChunkFwdHOFusedTilingData);
GET_TILING_DATA_WITH_STRUCT(
    GDN::ChunkFwdHOFusedTilingData, tilingData, tiling);
```

A5 随后调用 `RunChunkFwdHOFusedA5`；A2 根据 key `1/2` 选择 V128/V256 的
`RunPipeline` 模板。

## 7. 注册、构建和一致性约束

运行时关联链：

```text
OP_TYPE_REGISTER(ChunkFwdHOFused)
  -> OP_ADD(ChunkFwdHOFused)
  -> REGISTER_TILING_DATA_CLASS(ChunkFwdHOFused, ...)
  -> IMPL_OP_OPTILING(ChunkFwdHOFused)
  -> opFile/opInterface = chunk_fwd_h_o_fused
  -> kernel symbol chunk_fwd_h_o_fused
```

由于连续大写 `HO` 默认会折叠成 `chunk_fwd_ho_fused`，OpDef 显式设置：

```cpp
.ExtendCfgInfo("opFile.value", "chunk_fwd_h_o_fused")
.ExtendCfgInfo("opInterface.value", "chunk_fwd_h_o_fused")
```

ACLNN/L0 SO、OpDef、tiling SO、kernel JSON/O 和 `binary_info_config.json` 必须来自
同一次构建和安装。必须逐项保持：

1. 九个输入、两个输出和十一个属性的顺序一致。
2. Host TilingData 与 kernel 镜像的字段顺序、类型、对齐和总大小一致。
3. H 前缀的 `offsetof(useExp2)` 断言成立。
4. `opFile/opInterface` 指向实际的 `chunk_fwd_h_o_fused` 符号。
5. A2 的 IB/O workspace 和 A5 的 A-prime workspace 与架构分支一致。

修改任一 ABI 字段、输入顺序或 workspace offset 后，必须同步修改 Host 定义、
kernel 镜像、H 前缀/O 投影、OpDef、L0 launcher、测试和本文档。
