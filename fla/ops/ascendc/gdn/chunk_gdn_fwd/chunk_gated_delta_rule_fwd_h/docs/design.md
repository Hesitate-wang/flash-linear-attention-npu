# ChunkGatedDeltaRuleFwdH Design

方案设计规则版本：V2。公开接口见 [api.md](api.md)。

## 1. 目标

实现 chunk32 的 rank 并行 affine prefix；chunk64/128 沿用现有实现。
按用户指定的分核方案，每个 `batch * vhead` task 顺序执行，全部 AIC core
作为该 task 的 rank；每个 AIC 配对两个 AIV。这里采用用户指定的 chunk 轴
分核，覆盖通用规则 R20 的 head 轴分核默认值。

定义 `C=ceil(T/32)`、`R=CoreNum`、`L=ceil(C/R)`。
rank r 负责 `[min(r*L,C),min((r+1)*L,C))`；空 rank 不访问输入或输出 chunk，
但初始化 identity summary 并参加全局同步。例如 C=5,R=4，分区为
[0,2)、[2,4)、[4,5)、[5,5)。task 映射为 b=task/HV、hv=task%HV、
hk=hv/(HV/HK)。AIV 的 block id 除以 subblock 数后得到相同 rank。

令 q[t]=exp(g_last-g[t])，D[k]=exp(g_last)*exp2(gk_last[k])，缺失 gate
贡献单位因子。每 chunk 的 affine 为：

```
Bc = diag(D) - (q[:,None]*K).T @ W
Ac = (q[:,None]*K).T @ U
H_next = Bc @ H_start + Ac
v_new = U - W @ H_start
```

上述运算 FP32 累加。K 的转置通过 ColumnMajor logical view 表达；物理存储
始终为 [32,128] RowMajor。无效 tail 行在任何 MMAD 前全部补零，gate 只从
有效 token 读取。exp 的输入范围继承现有 CPU 标杆的 gate 值域；本变换不新增
指数因子。长链 affine 重新结合会改变浮点误差，必须通过设备精度验收。

## 2. Stage 完整详设

### 存储与生命周期

一个矩阵槽 S=128*128*sizeof(float)=64 KiB，所有矩阵为 FP32 GM RowMajor，
只有读取 Kᵀ 时用 ColumnMajor。整个调用按 task 顺序复用以下区域：

- summary 区：先 R 个 rank record，再 C 个 chunk record；每 record 为 B、A 两槽。
  总量 `2*(R+C)*S`。chunk 保存 exclusive prefix，即 chunk 开始前的局部 affine；
  rank 保存 inclusive 最终 summary。第一 barrier 前发布，第二 barrier 后释放。
- scratch 区：每 rank 14 槽，总量 `14*R*S`。槽 0/1/2 为 padded K/W/U；
  3/4 为 Bc/Ac；5/6 为 rolling B/A；7/8 为乘积 tmpB/tmpA；9/10 为 rank
  prefix B/A；11 为 H0；12 为实际 state；13 为 projection（只使用前 32 行）。
  D 向量仅驻留在 AIV UB，通过标量广播构造各对角行，不占 GM scratch。
  输入、输出矩阵不原位重叠；每个 Cube 完成通知和 Vec 完成确认后才能复用。
- UB 每 AIV `[0,512)` FP32 行 a、`[512,1024)` FP32 行 b、
  `[1024,1536)` typed 输入/输出行、`[1536,2048)` FP32 decay，
  `[2048,2080)` gate 标量/exp，分配 4096 bytes。按行批量 DataCopyPad，
  GM↔UB 搬运使用对应 MTE event 等待后才访问或复用缓冲区；
  AIV0/1 各处理连续的 64 个矩阵行，或连续 16 个 token 行，互不重叠。
- L1/L0 由 Catlass preload 模板分配；L1 tile [128,128,128]、L0 tile
  [128,128,32]，L1 A/B 各 2 stages，L0 A/B 各 2 stages，L0C 1 stage。
  FP32 L1 A+B 峰值 256 KiB，L0A/B 各 32 KiB、L0C 64 KiB。
  各 GEMM 的资源生命周期以 preSetFlags/finalWaitFlags 闭合。
  由于所有中间矩阵的完整 prefix 同时存活，本实现采用 GM 中转；性能与
  设备容量检查仍待实测，不声明流水重叠或性能提升。

### Stage 0：Vector，初始化与 padded operands

初始化 rolling (B,A)=(I,0)。每 chunk 在更新前将其保存为 exclusive record。
读取有效 token 的 K/W/U，将 K 逐元素乘 q，写入 padded [32,128] 槽。
尾部无效 token 不读取 GM，写零。两 AIV 在 MTE3 完成后共同发布 ready。

### Stage 1：Cube/Vector 交替，rank 内顺序合成

Cube `P=(q*K).T @ W`（M,N,K=128,128,32），Fixpipe 写槽 3；
Vector `Bc=diag(D)-P`，Cube `Ac=(q*K).T @ U` 写槽 4。
Cube `tmpB=Bc@B`、`tmpA=Bc@A`，均为 128 方阵；
Vector `A=tmpA+Ac`、`B=tmpB`。最后 Vector 将 rolling B/A 写 rank summary。
所有 rank（包括空 rank）共同参加 mode-0 barrier，发布本 task 的全部 summary。

### Stage 2：Cube/Vector 交替，前序 rank 合成

Vector 初始化 rank prefix `(P,Q)=(I,0)` 及 H0（输入转换 FP32 或零）。
按 prior=0..rank-1：Cube `tmpB=Bprior@P`、`tmpA=Bprior@Q`，
Vector `P=tmpB`、`Q=tmpA+Aprior`。使用独立临时槽，避免原位 alias。

### Stage 3：Cube/Vector 交替，chunk replay

对 exclusive record `(Bl,Al)`：Cube `tmpB=Bl@P`、`tmpA=Bl@Q`，
Vector `tmpA+=Al`；Cube `H=tmpB@H0`，Vector `H+=tmpA` 并转换写 h。
重新准备 padded W/U，Cube `projection=W@H`（32,128,128），
Vector 写 `v_new=U-projection` 的有效行。h/v_new 的 dtype 舍入不反馈到 FP32
prefix。若请求 final_state，由最后 rank 用自己的 inclusive summary 代替
exclusive record，执行同一状态计算，写 final_state；空末 rank 的 summary
是 (I,0)，因此仍能正确写最终状态。

### 同步与搬运

当前 mode-2 rendezvous 使用 READY/DONE ID，mode-0 AIC inter-block barrier 使用
CATLASS 定义的 ID 9；各 ID 的
core 配对、硬件计数上限和重复 task 生命周期尚未在 Ascend950 上验证，因此该同步
协议仍是待验证实现，不视为已证明无死锁。GM↔UB 的 Load/Store helper 以
V_MTE2/MTE2_V 和 V_MTE3/MTE3_V event 闭合本地搬运与计算依赖。

```
for task:
    Vec initialize; VecFence
    for local chunk:
        Vec save prefix/prepare; VecFence
        Cube GEMM; set done; AIV wait done
        Vec optional elementwise; VecFence
        # 每个 GEMM 都使用上述 done/ack 闭环
    Vec publish rank summary; VecFence
    AIC all-rank barrier
    Vec initialize rank prefix/H0; VecFence
    for prior rank: Cube/Vec compose with done/ack
    for local chunk: Cube/Vec materialize and project with done/ack
    last rank: optional Cube/Vec final_state with done/ack
    AIC all-rank barrier
```

Load/Store helper 在启动新 DMA 前等待对应方向的上一事件，完成 DMA 后再等待反向
event，保证 UB 源/目的可复用。跨核 READY/DONE 的生产、消费及 mode-0 屏障细节仍须
以 Ascend950 多轮运行记录验证；在取得记录前，不据此声称跨核写入可见或 flag 无累积。
空 rank 执行初始化、summary
发布和两个 barrier；省略它没有对应 chunk 的 GEMM。第二 barrier 防止快 rank
进入下一 task 覆盖仍被其它 rank 读取的 summary。最后任务也消费全部 flag。

每 chunk Stage 0 读取 K/W/U 各一次，replay 再读取 W/U 各一次；g/gk 只读取有效
位置。summary 在 GM 保存至整个 task 完成；prefix 与中间结果按上述槽显式搬运。
当前采用默认 cache 策略。重复搬运、串行 task 和逐 GEMM 握手的性能代价待
Ascend950 的 Task Duration 测量；不将 CPU 公式验证等同设备验收。
