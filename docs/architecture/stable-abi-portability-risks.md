# Stable-ABI 交付件的可移植性风险登记

这份文档只回答一个问题：**我们现在的产物，在"别处编、这里用"和"这里编、这里用"两种情况下，会在哪些地方出问题。**

涉及三样东西，它们的可移植性来源完全不同：

| 组成部分 | 形态 | 可移植性由什么决定 |
| --- | --- | --- |
| `libfla_npu_stable.so`（launcher） | 一个普通共享库，由 `torch.ops.load_library()` 加载 | 符号面（torch）+ C++ 运行时（libstdc++）+ host 架构 |
| Python 包（`fla_npu/**`） | 纯 Python + ctypes 参考路径 | Python 版本 + CANN/OPP 环境 |
| wheel 内的 OPP（`fla_npu/opp/vendors/fla_npu_transformer`） | 目标 SoC 的 kernel + `libcust_opapi.so` | SoC + host 架构 + CANN 版本 |

结论先说：**torch 版本这一轴已经基本免疫**（一份产物在 2.7.1 / 2.9 / 2.12 三个端点上全量 parity 通过），
**真正还没有闭环的是构建环境带来的两条轴（libstdc++ 下限、CANN/OPP 组合）和 SoC/架构这类"装错就废"的轴。**

---

## 1. 速览

| 编号 | 风险 | 触发场景 | 严重度 | 现有防护 | 缺口 |
| --- | --- | --- | --- | --- | --- |
| A1 | libstdc++（GLIBCXX）下限被构建机抬高 | A | 高 | `stable_abi_audit.py --lib` 断言 `max_glibcxx` | 未挂 CI；发版容器未固定 |
| A2 | glibc（`GLIBC_x.y`）下限被构建机抬高 | A | 中 | 无（audit 只打印） | 没有上限断言 |
| A3 | 注册入口 / 新增运行时符号 | A | 低 | `--lib` 断言入口符号 + 符号白名单 | 已覆盖 |
| A4 | torch 版本范围 | A | 低 | wheel 声明 `torch>=2.7.1`；加载失败有清晰报错 | 2.8 / 2.10 / 2.11 运行时未跑 |
| A5 | torch_npu 补丁级版本 | A | 中高（正确性） | `setup.py` 有最低版本表 | 默认构建**不执行**该检查；运行期也没有校验 |
| A6 | CANN / OPP ABI 漂移 | A、B | 中高 | `op_abi_validate.py`、`op_abi_parity.py`、`verify_libcust_opapi_md5.py` | 跨 CANN 版本矩阵未做 |
| A7 | host 架构不匹配 | A | 已收敛 | wheel tag 由 `py3-none-any` 改为 `py3-none-<platform>` | 已发出的旧包仍是 `any` |
| A8 | SoC 不匹配（910B / 950） | A | 高 | 部分算子按 `device_name` 显式拒绝 | 没有"这个包是为哪个 SoC 编的"全局校验 |
| A9 | Python 版本 | A | 低 | `python_requires>=3.9`，tag 为 `py3` | 3.9 / 3.12 / 3.13 未实测 |
| A10 | CANN / OPP 环境变量被覆盖 | A、B | 中 | 包内 OPP + `fla_npu_opp_env.pth` | 无冲突检测 |
| A11 | 加载失败后静默退 ctypes | A | 中 | 每进程告警一次 + `FLA_NPU_STABLE_TRACE` | 客户可能忽略 warning |
| A12 | wheel 命名变更 | A | 低 | — | 写死旧名的流水线/文档要同步 |
| B1 | 自编产物与构建机绑定 | B | 高 | 无 | 拷到别的机器可能加载不了 |
| B2 | 构建机依赖清单 | B | 低 | 缺 torch 时给出可读报错 | — |
| B3 | vendor 头被绕过（回落到构建 torch） | B | 低 | `vendor_stable_headers.py --check --torch` | 未挂 CI |
| B4 | OPP 来源混用（本地编 vs 包内） | B | 中高 | `op_abi_parity.py`、`op_abi_validate.py` | 需要人主动跑 |
| B5 | `_stable_hash.py` 与 `.so` 不同步 | B | 低 | 加载时比对并报错 | 已覆盖 |
| B6 | `FLA_NPU_STABLE_LIB` 泄漏 | B | 中 | 无 | 会静默使用旧产物 |
| C1 | 同名包只能装一个 | A、B | 中 | — | 多 SoC 并存要独立 venv |
| C2 | launcher 与 ctypes 必须位级一致 | A、B | 高 | `regression_stable_full.py` + 双机基线 | 基线要按 SoC 分别维护 |
| C3 | 多流 / 多线程 | A、B | 已收敛 | `test_stable_stream_interleaving.py` | — |

---

## 2. 场景 A：一个环境编 wheel，另一个环境用

### A1. libstdc++（GLIBCXX）下限 —— 目前最值得担心的一条

**现象**：目标机上加载失败，报的是加载器信息：

```
ImportError: /…/fla_npu/libfla_npu_stable.so: version `GLIBCXX_3.4.32' not found
```

**触发条件**（三条同时成立）：

1. 构建机的 C++ 工具链比目标机新。实测两端：241 = Ubuntu 24.04 / g++ 13.3 / 系统 libstdc++ 上限 `3.4.33`；
   221 = Ubuntu 22.04 / g++ 11.4 / 上限 `3.4.30`。
2. 编译期生成的符号引用（例如 GCC 13 头带来的 `_ZSt21ios_base_library_initv`）在链接时**从系统 libstdc++** 解析到，
   于是引用被标上该库的版本。
3. 运行进程里解析到的那份 libstdc++ 比它旧。

**实测证据**：

- 同一份源码、同一个 g++，只换 `-L` 指向的 torch 目录：`fzy-t27`（torch 2.7.1+cpu）→ 要求 `GLIBCXX_3.4.32`；
  `fzy-t29`（torch 2.9.0+cu128）→ 无 GLIBCXX 要求。
- 原因是 `_ZSt21ios_base_library_initv` 在 torch 2.7.1 的库里**没有**定义（0 个），在 2.9 的库里**有**（2 个）。
  从 torch 自己的 .so 解析到的是无版本符号，于是产物对它没有任何版本要求。
- 出货产物（221 上 GCC 11 编）要求 `≤ 3.4.21` + `GLIBC_2.2.5`；在 conda 的 2.7.1 环境里编出来的那份要求 `3.4.32`。

**这条轴不只属于 launcher —— 包的真实下限由 OPP 决定。** 在同一个安装态包里逐个量（aarch64）：

| 产物 | GLIBCXX 上限 | 谁编的 |
| --- | --- | --- |
| `libfla_npu_stable.so`（我们的 launcher） | 3.4.21 | 我们 |
| `libfla_npu_thin.so`（旧的 pybind 方案，留作对照） | 3.4.21 | 我们 |
| OPP `libcust_opapi.so` / `liboptiling.so` / `libcust_opmaster_rt2.0.so` | **3.4.29** | 我们（同一容器） |
| OPP `libcust_opsproto_rt2.0.so` / `libes_transformer_cust.so` | 3.4.18 / 3.4.11 | 我们 |
| CANN `libopapi.so` / CANN 目录内最高 | 3.4.18 / 3.4.26 | CANN 官方 |
| 目标机 libstdc++（Ubuntu 22.04） | 3.4.30 | — |

两个结论：

1. **这条轴在 ctypes 方案里同样存在**——那时我们没有 C++ 的 host 层，但 OPP 里的 C++ 产物一直是编译产物，
   而且它的要求（3.4.29）比今天的 launcher（3.4.21）**还高**。换适配方案并不能去掉这条轴。
2. **今天整包的下限是 OPP 的 3.4.29，不是 launcher 的 3.4.21。** 即使把 launcher 降到 3.4.19，
   包在 Ubuntu 20.04（上限 3.4.27）上照样装不起来。要真正降下限，必须同时降 OPP 的构建工具链。

`ci/Dockerfile` 的基础镜像是 `cann:9.1.0-910b-ubuntu22.04-py3.12-devel`，Ubuntu 22.04 + GCC 11 的
libstdc++ 上限恰好是 3.4.29，所以**走 CI 编出来的产物天然就是这条水位**（目标机留一级余量）；
真正危险的是人手在更新的机器（例如 Ubuntu 24.04 / GCC 13 的 241）上编产物再发出去。

**这条下限不是本次改动引入的：已发布的 26.6.0 包就已经在同一个水位上。** 量一个 26.6.0 的正式
wheel（`flash_linear_attention_npu-26.6.0-910b.aarch64-py3-none-any.whl`）：

```
fla_npu/custom_aclnn_extension_lib.cpython-311-aarch64-linux-gnu.so      3.4.21
fla_npu/opp/.../op_api/lib/libcust_opapi.so                              3.4.29
fla_npu/opp/.../op_api/lib/libopapi.so                                   3.4.29
fla_npu/opp/.../op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so    3.4.29
fla_npu/opp/.../op_tiling/liboptiling.so                                 3.4.29
fla_npu/opp/.../op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so     3.4.18
```

也就是说：这条轴在客户手里**已经存在至少一个发布周期**，且没有触发过问题——反过来可以推断
**实际部署的机器都在 3.4.29 之上**（等价于 Ubuntu 22.04+ / GCC 11+）。
所以它的定位不是"今天会不会炸"，而是两件事：

1. **防止下限继续上漂**：本次在 Ubuntu 24.04 上编 launcher 就得到了 3.4.32；如果哪天有人用
   GCC 14 / 24.04 的镜像重建 OPP，下限会抬到 3.4.33 之上，Ubuntu 22.04 就开始出问题。
2. **把支持矩阵写实**：本包要求 `libstdc++ ≥ 3.4.29`，即 Ubuntu 22.04+ / openEuler 22.03+ /
   GCC 11+；CANN 9.1 自己的库只需要 ≤ 3.4.26，所以顶着这条线的是我们的 OPP，不是 CANN。

顺带一个判断依据：这类失败是**硬报错**（OPP 加载不了会直接报 aclnn 加载失败），不会被静默吞掉，
所以"没遇到过"在这里是相当强的证据，而不只是"没人注意"。

**为什么"内部测不出来"**：`3.4.32` 这批产物在构建机上当然能加载（构建机就有新库），
而客户用 conda python 时进程里也是 conda 自带的 `libstdc++ 6.0.34`（提供到 `3.4.34`），也可能恰好不报；
换成系统 python 或更老的镜像就报。**同一台机器、同一个包，换个 python 入口结论就变。**

**防护**：`tools/stable_abi_audit.py --lib` 断言 `GLIBCXX` 上限（阈值在 `tools/stable_abi_symbols.json` 的 `max_glibcxx`）。
**缺口**：

- 该断言只看**一个** launcher，不看包内其余 `.so`——而真正顶着下限的是 OPP 的那三个文件，需要把"取包内最大值"作为判据；
- 未挂 CI；`max_glibcxx` 现在写 3.4.21（launcher 的水位），与"整包下限 3.4.29"不是一回事，两者要对齐；
- 发版容器没有写进流程（`ci/Dockerfile` 已经是 22.04，但"不要用更新的机器编产物"目前只是口头约定）。

### A2. glibc 下限

同一条逻辑，走 `GLIBC_x.y` 符号。当前产物只要 `GLIBC_2.2.5`（很老，不是问题），
但换成更新/更旧的构建机同样可能改变它。audit 目前**只打印** NEEDED 与版本，没有对 `GLIBC_` 设上限——这是缺口。

### A3. 注册入口与新增运行时符号（已覆盖）

产物必须引用 `aoti_torch_library_impl`（2.7.1 ~ 2.12 都导出），不得引用 `torch_library_impl`（只有 ≥2.10 有）。
引入新的 `aoti_torch_*` 会抬高运行时下限，因此 audit 要求它出现在 `stable_abi_symbols.json` 里。
实测：六版本编译产物引用 38 个符号，在 2.7.1（导出 278 个同类）上 0 缺失。

### A4. torch 版本范围

一份产物实测覆盖 2.7.1 / 2.9 / 2.12 三个端点（全量 parity 284 / 272 / 284 通过）。
低于 2.7.1 时加载失败，`_stable.load()` 会给出"需要 torch >= 2.7.1"的可读报错；wheel 元数据也声明了 `torch>=2.7.1`。
**缺口**：2.8 / 2.10 / 2.11 只验证了"能编"，没跑运行时。

### A5. torch_npu 补丁级版本（正确性风险）

`setup.py` 里维护了一张按 torch 家族的 torch_npu 最低版本表（2.7.1→post5、2.8→post5、2.9→post3、2.10→post2、2.11→rc3、2.12→rc1），
对应的是 torch_npu 侧 GDN 的 stream 修复。但：

- 该检查在 `if build_legacy_extension:` 分支里，**默认构建（不带 legacy 扩展）不执行**；
- 即便执行，也是**构建期**检查，装 wheel 的客户永远不会跑到；
- 运行期没有对应的版本校验，wheel 元数据只声明 `torch_npu>=2.7.1`。

也就是说：客户装一个"补丁级偏低的 torch_npu"，可能既不报错也不降级，而是拿到不对的结果。
**需要确认当前调用路径是否还会落到 torch_npu 的 GDN 实现**：如果会，这条要升级为高危并补运行期校验；如果不会，这张表应当标注为历史遗留。

### A6. CANN / OPP ABI 漂移

launcher 用 `dlopen` 加载两份库：CANN 的 `libopapi.so` 和我们的 `libcust_opapi.so`（包内 OPP）。
`aclnn*` 的符号名与参数槽位必须与**运行期 CANN** 一致；历史上踩过"12 槽 vs 18 槽"的旧 ABI（causal_conv1d）。
**防护**：`tools/op_abi_validate.py`（用 OPP 头对拍）、`tools/op_abi_parity.py`、`scripts/verify_libcust_opapi_md5.py`。
**缺口**：跨 CANN 版本（同 SoC 不同 CANN）矩阵未做。

### A7. host 架构

原先是 `py3-none-any`，pip 会把 aarch64 的包装到 x86_64 上，然后在 dlopen 时才失败。
现在 tag 是 `py3-none-<platform>`（`linux_aarch64` / `linux_x86_64`），pip 会正确拒绝。
**缺口**：更早发出去的 `any` 包没有这层保护。

### A8. SoC 不匹配（910B / 950）

wheel 文件名里的 `910b` / `950` 只是 build tag，**pip 不校验**。装错 SoC 的后果是逐算子报错
（例如 950-only 的 `chunk_gated_delta_rule_bwd_finalize` 会按 `device_name` 明确拒绝）或结果异常。
包内只有零散的 SoC 分支，没有"这个包是为哪个 SoC 编的"全局校验。
**建议**：运行期做一次 SoC 与包标签的比对，不一致直接报错——比让客户在某个算子上偶发精度问题好得多。

### A9. Python 版本

launcher 是普通共享库、不含 CPython 扩展，tag 为 `py3`，`python_requires>=3.9` 让 pip 拒绝 3.8。
实测 3.10（241）与 3.11（221）可用。**缺口**：3.9 / 3.12 / 3.13 未实测。

### A10. 环境变量

生效的开关：`ASCEND_CUSTOM_OPP_PATH`（CANN 发现我们的 OPP）、`FLA_NPU_OP_API_LIB`、`LD_LIBRARY_PATH`、
`FLA_NPU_STABLE_LIB`（覆盖包内 launcher）。客户环境里如果先 `source` 了别的 `set_env`，
或者残留了指向旧目录的 `ASCEND_CUSTOM_OPP_PATH`，表现会是"找不到算子/加载到错版本"。
**缺口**：没有冲突检测（例如发现 `ASCEND_CUSTOM_OPP_PATH` 指向的不是本包 OPP 时给出提示）。

### A11. 静默降级

launcher 加载失败时，所有算子回退到 ctypes 参考实现：结果正确，但失去 host 侧加速。
现在每个进程告警一次，并在 `FLA_NPU_STABLE_TRACE=1` 时把"launcher 不可用"与"该算子没带"区分开。
**缺口**：warning 容易被客户日志淹没，表现为"装了但没变快"的工单。

### A12. wheel 命名变更

`flash_linear_attention_npu-<ver>-<buildtag>-py3-none-any.whl` →
`flash_linear_attention_npu-<ver>-<buildtag>-py3-none-<platform>.whl`。
仓内的名字预测脚本已同步（`scripts/fla_npu_artifacts.py wheel-filename`），流水线与安装文档需要一并核对。

---

## 3. 场景 B：自己编、自己用

### B1. 产物与构建机绑定（最高的一条）

自编自用掩盖了 A1：在自己机器上编、自己机器上跑，GLIBCXX 永远不会不匹配。
**一旦把这个 `.so` 拷到别的机器（很常见：先在构建机验证，再分发到测试机/客户机），就变成场景 A 的风险。**
实测：conda 2.7.1 环境里编出来的 `.so` 要求 `GLIBCXX_3.4.32`，而 221 的系统 libstdc++ 上限是 `3.4.30`。
**建议**：任何"编完拷走"的流程，都要先在目标机跑一次 `stable_abi_audit.py --lib`。

### B2. 构建机依赖清单

编 launcher 需要：`g++`、可导入的 `torch`（只为拿 `torch/lib` 路径）、bash。
编译期**不需要** torch 的头文件（vendor 头是完整闭包，实测两个 TU 在不给任何 torch include 路径时可编过）。
缺 torch 时现在的报错是明确的（`SystemExit` + "需要 torch >= 2.7.1"）。
如果要在没装 torch 的机器上编，需要提供 lib 目录替代方案（目前没有这个开关）。

### B3. vendor 头被绕过

如果 include 顺序被改动、或 vendor 树被删/被改，编译会回落到构建机的 torch 头，产物就重新跟构建机版本绑定（回到当初的问题）。
**防护**：`tools/vendor_stable_headers.py --check`（逐字节比对）与 `--check --torch`（闭包必须全部来自 vendor）。
**缺口**：未挂 CI，需要人主动跑。

### B4. OPP 来源混用

源码树里自己编的 OPP（`build.sh --pkg` 产物）与 wheel 里带的那份可能不是同一版本；
用 `ASCEND_CUSTOM_OPP_PATH` 指向本地 OPP 时，算子集/参数槽位可能与 launcher 期望的不一致。
**防护**：`tools/op_abi_parity.py`（三方顺序对齐）、`tools/op_abi_validate.py`（对照 `aclnn_*.h`）。

### B5. build stamp 必须同步

`.so` 内嵌源码哈希（`fla_npu_stable_source_hash()`），Python 侧 `_stable_hash.py` 记录期望值；
手工只拷 `.so` 不更新 `_stable_hash.py` 会在加载时明确报错（这是好事，不要绕过）。

### B6. `FLA_NPU_STABLE_LIB` 泄漏

`_lib_path()` 优先返回 `FLA_NPU_STABLE_LIB`，其次才是包内那份。
长期开着这个变量指向一个旧 `.so`，会静默使用旧产物（性能/行为都可能不符合预期），且不带任何提示。

### B7. 构建残留

同一目录重复编、或从 `_C_thin` 时代留下的产物混在一起，可能被误打包。wheel 组装侧有清理逻辑，源码树侧没有。

### B8. 自编自用不覆盖发布侧风险

平台标签、wheel 内容、安装矩阵、CANN 组合——这些只有走"编 wheel → 干净环境安装 → 跑 smoke"才能发现。
`tests/stable_abi/` 是设备回归，`scripts/check_packaged_wheel_api.py` 与 `check_install_workflows.py` 是安装侧检查。

---

## 4. 两种场景共有

**C1 同名包只能装一个。** `flash-linear-attention-npu` 同名互覆盖，多 SoC / 多版本并存必须用独立 venv。

**C2 launcher 与 ctypes 必须位级一致。** 否则"客户 A 走 launcher、客户 B 因加载失败走 ctypes"会得到不同结果。
基线 `tests/stable_abi/stable_scenarios.json` 按 SoC 分开（`Ascend910B3`、`Ascend950PR_9579`），改动场景集必须两台都重录。

**C3 多流 / 多线程。** vLLM 场景下曾因进程级 stream 缓存崩溃，已修并留下 `test_stable_stream_interleaving.py`
（含"故意装回缓存必须失败"的负例）。

**C4 加载顺序。** launcher 由 `torch.ops.load_library()` 加载，必须在 torch 初始化之后；fork 出来的子进程要重新加载。

---

## 5. 风险 ↔ 门禁对照

| 能自动发现 | 靠哪条 |
| --- | --- |
| GLIBCXX 超限、入口符号错、新增符号、RPATH、NEEDED | `tools/stable_abi_audit.py --lib` |
| 编译期重新依赖 torch 头 | `tools/stable_abi_audit.py --vendor-syntax-only` |
| vendor 树被改、闭包不闭合 | `tools/vendor_stable_headers.py --check [--torch]` |
| 平台标签、wheel 元数据 | `tests/test_wheel_environment.py`（**这条已进 CI**） |
| 算子语义与 ctypes 的位级一致 | `tests/stable_abi/regression_stable_full.py`（需要 NPU） |
| `.so` 与 Python 侧版本不匹配 | 加载时的 build stamp 比对 |

**只能靠人的**：构建容器是否换了、发版机 libstdc++ 是否变新、CANN/OPP 组合是否换过、装到了哪个 SoC、客户日志里的降级告警。

---

## 6. 自检清单

发布前（在**发版机**上，对**刚编出的产物**）：

```bash
python torch_custom/fla_npu/tools/stable_abi_audit.py \
    --lib torch_custom/fla_npu/fla_npu/libfla_npu_stable.so     # 符号面 / GLIBCXX / NEEDED / RPATH
python torch_custom/fla_npu/tools/stable_abi_audit.py --vendor-syntax-only
python torch_custom/fla_npu/tools/vendor_stable_headers.py --check
python -m unittest tests.test_stable_gates tests.test_wheel_environment
```

拷到目标机之后（**第一件事**）：

```bash
readelf -V <libfla_npu_stable.so> | grep -o 'GLIBCXX_[0-9.]*' | sort -uV | tail -1   # 与目标机 libstdc++ 上限比对
strings /usr/lib/<arch>/libstdc++.so.6 | grep -o 'GLIBCXX_3\.4\.[0-9]*' | sort -uV | tail -1
```

装完包之后：

```bash
FLA_NPU_STABLE_TRACE=1 python -c "import fla_npu, torch; print(fla_npu.ops.ascendc.BACKENDS)"
# 期望：每个算子都是 stable；出现 ctypes 就要看告警原因
```

---

## 7. 还没有闭环的项

1. torch 2.8 / 2.10 / 2.11 的**运行时**验证（目前只有编译）。
2. Python 3.9 / 3.12 / 3.13 的实测。
3. `GLIBC_`（C 库）上限断言。
4. **包内 `.so` 的 GLIBCXX 上限断言**：现在只查 launcher，OPP 的三个 3.4.29 文件没有门禁覆盖；
   判据应当是"包内所有 `.so` 的最大值 ≤ 目标机水位"，而不是单个文件。
5. 六版本产物的 6×6 交叉矩阵（X 编 → Y 跑）。
6. 发版容器的固定，以及 CI 里挂上 audit（`--lib` 那条会与容器绑定）。
7. SoC 与包标签的运行期校验。
8. torch_npu 补丁级版本的运行期校验（或明确它已不相关）。
9. 在真正的老 libstdc++ 环境上复现一次客户侧的加载失败，把报错形态钉死。
10. 非 conda 的系统 python 加载"要求 3.4.32 的产物"的实测。
