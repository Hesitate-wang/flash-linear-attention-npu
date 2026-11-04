# Lightweight chunk_gated_delta_rule_fwd tests

默认运行 **融合 DUT + CPU FP64 golden**，不依赖 ATK。可通过独立选项增加或替换对照路径：

- `--compare-cpu`：显式启用 CPU FP64 golden；
- `--compare-chain`：启用 Prepare → FwdH → FwdO 的 NPU 拼接对照；
- 两者同时指定：执行 DUT、CPU、NPU 拼接三路对齐；
- 仅指定 `--compare-chain`：跳过 CPU golden，只比较 DUT 与 NPU 拼接。

不指定任何 compare 选项时保持默认的 DUT+CPU 轻量模式。Prepare → FwdH → FwdO 拼接只在显式指定 `--compare-chain` 时运行。

## 环境

需要：

- Ascend NPU 和 CANN；
- PyTorch、torch_npu；
- 已安装包含当前算子的 `fla_npu`；
- 使用 `--compare-chain` 时，还需安装 `chunk_gated_delta_rule_fwd_prepare`、`chunk_gated_delta_rule_fwd_h` 和 `chunk_fwd_o`；
- 已加载 CANN 和 `fla_npu_transformer` vendor 环境。

不需要 TensorFlow，也不需要 ATK。`--cpu-only` 仅需要 PyTorch。

## 快速运行

```bash
# 默认 dense-tail：DUT + CPU FP64 golden
bash tests/run_lightweight.sh --device 0

# 只比较 DUT 与 Prepare → FwdH → FwdO，不运行 CPU golden
bash tests/run_lightweight.sh --compare-chain --device 0

# DUT、CPU golden、NPU 拼接三路对齐
bash tests/run_lightweight.sh --compare-cpu --compare-chain --device 0

# varlen：0..17、17..80、80..129
bash tests/run_lightweight.sh --case varlen --device 0

# 含空序列和重复边界的 varlen
bash tests/run_lightweight.sh --case varlen-empty --device 0

# 三个小用例依次运行；每个用例有独立日志目录
bash tests/run_lightweight.sh --case all --device 0

# 本机无 NPU 时，只验证输入、golden、日志与 summary
bash tests/run_lightweight.sh --case all --cpu-only
```

常用覆盖选项：

```bash
bash tests/run_lightweight.sh \
  --tokens 129 --key-heads 1 --value-heads 2 \
  --use-exp2 --beta-sigmoid --state-v-first \
  --repeat 3 --warmup 1 --save-tensors --device 0
```

- `--no-l2norm`：关闭默认开启的核内 Q/K L2Norm。
- `--no-initial-state` / `--no-final-state`：关闭初始/最终状态。
- `--initial-dtype bf16`：使用 BF16 初态和末态。
- `--allow-neg-eigval`：要求同时指定 `--beta-sigmoid`。
- `--cu-seqlens 0,31,96,129`：覆盖自定义变长边界。
- `--diff-thd/--pct-thd/--max-diff-hd`：混合容差，默认 `0.001/0.0001/0.1`，即绝对误差小于 `0.001` 直接通过、相对误差阈值为 `0.001`、至少 99.99% 元素通过、最大混合误差小于 0.1。
- `--seed`：固定输入随机种子，默认 19。
- `--gate-scale`：负 gate 的幅值，默认 0.05；设为 100 可覆盖累积极负时的 exp 下溢场景。
- `--dump-aprime`：调试模式，将 A′ 写入公开 A 输出槽并逐元素比较；此时 A 不再表示逆系数。
- `--dump-ol`：调试模式，绕过 gate/scale/输出融合，将原始 O_l 写入公开 o 并逐元素比较。
- `--dump-os`：调试模式，将原始 O_s 写入公开 o 并逐元素比较。
- `--dump-scalars`：调试模式，在 o 的前四个元素导出 `energy/scale/g/lastEnergy`，用于单 token 标量核对。

## 比较范围

始终比较：

- `o`：统一为公开 BSND；
- `g_cumsum`；
- `A`；
- `h`：逐 chunk 的状态快照，同时额外记录每个 tile 的误差；
- 请求时的 `final_state`。

按开关额外比较 `beta_eff/q_hat/k_hat/q_rstd/k_rstd`。启用 `--compare-chain` 时，DUT 还会与 Prepare → FwdH → FwdO 的对应输出比较，指标单独记录为 `compare_chain` 和 `chain_metrics`。Prepare 在 NPU 上独立完成 L2Norm 和 beta sigmoid，无需 CPU 预计算 hats。

## 日志

每次运行生成 `tests/logs/<timestamp>_<case>/`：

- `run.log`：同时输出到终端和文件；
- `summary.json`：状态、完整配置、环境信息索引、耗时、输出统计和全部误差；
- `inputs.pt`、`dut_outputs.pt`、`golden_outputs.pt`：指定 `--save-tensors` 时保存；
- `chain_outputs.pt`：同时指定 `--compare-chain --save-tensors` 时保存。

日志包含：

- Python、PyTorch、torch_npu、平台和 Git commit；
- CANN/OPP 相关环境变量；
- 输入和全部非空输出的 shape、dtype、min/max/mean/std、finite；
- golden 和每次 DUT 的耗时；
- 每个比较输出的 `max_abs/mean_abs/RMSE/max_rel/mean_rel`；
- 超差元素数量/比例、通过率、最大混合误差、最差元素位置及 DUT/golden 数值；
- DUT 和参考输出中任意 NaN/Inf 都直接判定失败；
- NPU 同步结果、异常完整 traceback 和最终状态。

`summary.json.status` 可能为 `passed`、`failed` 或 `cpu-only`。`cpu-only` 不代表设备 DUT 已执行或通过。
