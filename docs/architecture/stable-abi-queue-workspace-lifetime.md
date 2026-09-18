# 队列下发下的 workspace 生命周期（2026-09-16，910B3）

`7f12b4ec`（把 launch 交给 torch_npu 任务队列）之后，队列条目同时持有 aclnn
descriptor 和 workspace tensor。本文件记录为什么把它改成 vLLM 的写法（workspace
留在调用线程的帧内），怎么验证的，以及实测结果。

## 1. 问题

单形状、同进程、2000 次 recurrent 调用后，`torch.npu.memory_reserved()` 会随调用
持续上升；同样形状走内联下发（`FLA_NPU_STABLE_LAUNCH=inline`）或走 vLLM 的
`_C_ascend` custom op 都是 0。

两边的差异只有一处：workspace 的**释放发生在哪个线程**。

## 2. 为什么这么改

vLLM 的 `EXEC_NPU_CMD`（`csrc/aclnn_torch_adapter/op_api_common.h`）是权威参考：

```cpp
auto workspace_tensor = at::empty({workspace_size}, options.dtype(kByte));   // 局部
workspace_addr = const_cast<void *>(workspace_tensor.storage().data());
auto acl_call = [converted_params, workspace_addr, ...]() -> int { ... };    // 只捕裸指针
at_npu::native::OpCommand cmd; cmd.Name(...); cmd.SetCustomHandler(acl_call); cmd.Run();
```

`workspace_tensor` 是宏内局部变量，`Run()` 一返回（还在调用线程上）就析构；闭包只
捕指针。我们的版本把同一个张量 `std::move` 进队列条目，于是释放被推迟到消费者线程
执行该条目的时刻——这就是本次要改掉的东西。

顺序性不因此受损：workspace 提前回到 caching allocator 并不等于它会在这条流上被
提前使用，因为本次 launch 和之后所有算子进的是**同一个队列**，队列本身保证提交顺序；
这也正是 vLLM 线上一直在跑的路径。

改动（3 个文件）：

- `csrc/include/stable/exec.h`：`QueuedLaunch` 只保留 `held`；`enqueue_launch` 去掉
  `workspace` 形参、去掉 `state->workspace.emplace(...)`；被拒（外部 stream 等）时
  不再需要把 workspace 搬回去——它本来就在调用方帧内。
- `csrc/src/stable_recurrent_gdr.cpp`、`csrc/src/stable_recurrent_kda.cpp`：两个手写
  adapter 的 `enqueue_launch(...)` 调用去掉 `workspace`。

`7f12b4ec` 当初写成条目持有，是为了"提交之前 storage 不能回到 allocator"这条保守的
存活期论证；代价是把释放搬到消费者线程，而收益并不存在。

## 3. 怎么测的

环境：221 容器 `fla-vllm14340-e2e`（CANN 9.2.0、torch/torch_npu 2.12、python 3.12），
卡 6，容器内就地编译。

同一棵树编两份 launcher，**只差上述 3 个文件**：

| 树 | 内容 |
| --- | --- |
| `ab_ws_base` | 559 HEAD（队列条目持有 workspace） |
| `ab_ws` | + 本补丁（调用线程释放） |

两侧共用同一份 OPP（从已安装 wheel 复制进树内；该版本的 `fla_npu.__init__` 拒绝
symlink 出去的 `opp`），因此设备侧 kernel 完全相同。

探针：

- `probes/mem_footprint_probe.py --child --arm fla --op {recurrent,conv1d_update}
  --batch 8 --warmup 20 --loop 2000`：单次峰值、单次 live 增量、2000 次循环后的
  `allocated` / `reserved` 增量。
- `probes/host_timing_probe.py`：每个算子 200 次，报 p50 / p10 / min（计时区间内不
  synchronize）。整组序列见 `ab_queue_ws.sh`。

另外三组对照（安装包自带的 launcher，同一探针）：队列 / `FLA_NPU_STABLE_LAUNCH=inline`
/ vLLM 的 `_C_ascend` custom op。

> 234 上该版本的 recurrent 起不来（卡在 op 上，和主线程早先遇到的一致），因此整套
> 测试放在 221；221 当时服务已停，未启停任何服务。

## 4. 结果

### 4.1 同源 A/B：recurrent 2000 次后的 reserved 增量（MiB）

| 轮次 | 未修复（条目持有） | 修复后（调用线程释放） |
| --- | --- | --- |
| 1 | +288.0 | **0.0** |
| 2 | +4122.0 | **0.0** |
| 3 | +414.0 | **0.0** |

每轮两臂的 `single_peak_delta_mib` 都是 16.033、`single_live_delta_mib` 和
`loop_allocated_delta_mib` 都是 0.0；修复臂的 reserved 轨迹在 500/1000/1500/2000
处恒为同一个值。

### 4.2 同源 A/B：host p50（第二次采样的稳态，µs）

| 轮次 | 算子 | 未修复 | 修复后 |
| --- | --- | --- | --- |
| 1 | recurrent | 80.2 | 102.4 |
| 1 | conv1d_update | 53.8 | 59.0 |
| 2 | recurrent | 75.3 | 75.7 |
| 2 | conv1d_update | 60.0 | 53.4 |

同一进程内第一组采样含预热，绝对值偏高；稳态两组之间没有可辨认的差异（p10/min 同样
在噪声内）。也就是说这次改动不付 host 成本的代价。

### 4.3 安装包三臂对照：recurrent 2000 次

| 路径 | reserved 增量 | allocated 增量 | 单次峰值 |
| --- | --- | --- | --- |
| fla（队列） | +288.0 MiB | 0 | 16.033 MiB |
| fla（`FLA_NPU_STABLE_LAUNCH=inline`） | 0 | 0 | 16.033 MiB |
| vLLM `_C_ascend` custom | 0 | 0 | 16.033 MiB |

`conv1d_update` 三臂全 0（该算子 workspace 为 0 字节）。

> 同一进程内 queue 与 inline 的 host p50 没有差别（conv1d 60.1 / 64.1 µs，recurrent
> 80.5 / 83.4 µs）：内联要付的队列 barrier 只在真实 worker 里有负载时才有代价，
> 算子级探针复现不出来，这里不能用来比较两种下发的 host 成本。

## 5. 解读

- 爬升全部落在 `reserved` 上，`allocated` 恒为 0：不是 Python 侧引用泄漏，而是
  caching allocator 里有一批块在一段时间内无法复用/合并——释放从调用线程挪到消费者
  线程之后出现的现象。torch_npu 的 `free` 仍按块记录的 device/stream 入池，具体
  触发路径没有在源码层面完全坐实，本文件只作观测结论。
- **它不是"有界的小效应"**：三轮 288 / 4122 / 414 MiB，差一个数量级，说明它与消费者
  线程的时序、分配器当时的状态耦合。按有界处理会低估最坏情况。
- 修复把这一项清零，同时保留队列下发；host 成本无退化。
- 边界：这一项**解释不了服务级 OOM 的全量**。inline 臂在服务级同样 OOM，而算子级
  inline 的 reserved 增量是 0，两者对不上；建议在 worker 里打印
  `fla_npu.ops.ascendc._stable._enqueues_launch()` 确认那一臂的 inline 是否真的生效。

## 6. 复现

```bash
# 两份树各编一次（容器内，约 40 s）
docker exec fla-vllm14340-e2e bash -lc \
  "cd /mnt/share/weights/vllm-ascend/fla-e2e-14340/ab_ws/fla_npu && \
   python3 csrc/build_stable.py --out ../libfla_npu_stable.so"

# 整组 A/B（内存 + host，两轮）
bash /mnt/share/weights/vllm-ascend/fla-e2e-14340/ab_queue_ws.sh
```

单臂：

```bash
docker exec fla-vllm14340-e2e bash -lc '
  export ASCEND_RT_VISIBLE_DEVICES=6
  export PYTHONPATH=/mnt/share/weights/vllm-ascend/fla-e2e-14340/ab_ws/fla_npu:/vllm-workspace/vllm-ascend-custom
  export FLA_NPU_STABLE_LIB=/mnt/share/weights/vllm-ascend/fla-e2e-14340/ab_ws/libfla_npu_stable.so
  python3 /mnt/share/weights/vllm-ascend/fla-e2e-14340/probes/mem_footprint_probe.py \
    --child --arm fla --op recurrent --batch 8 --warmup 20 --loop 2000'
```

## 7. 结论与待办

- 采纳：队列条目只持有 descriptor，workspace 由调用线程帧释放（与 `EXEC_NPU_CMD`
  一致）。回退开关保持不变：`FLA_NPU_STABLE_LAUNCH=inline`。
- 559 与 26.9.0 两条线的这 3 个文件 LF 归一后一致，同一份补丁两边都适用。
- 待办：服务级 OOM 还需要第二个来源；先用 `_enqueues_launch()` 确认 inline 臂是否
  真的走内联，再决定下一步是查 glue 里的引用，还是把本项泄漏当作主要嫌疑人重跑服务级
  A/B。
