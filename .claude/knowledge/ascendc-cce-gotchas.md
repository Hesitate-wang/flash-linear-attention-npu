# AscendC 编译/运行踩坑记录

本文件是本地自定义知识库的条目集，索引见 [README.md](README.md)。

---

## ASC-0001 UpdateMask 不接受整型字面量

- **首次记录**：2026-11-28，`chunk_gated_delta_rule_fwd` 的 chunk64 sparse
  `h` 导出辅助函数 `Bf16PanelsToNd64`。
- **架构/版本**：arch35（Ascend 950），CANN 提供的 AscendC MicroAPI。

### 现象

```cpp
MaskReg m64 = UpdateMask<bfloat16_t>(64);
```

编译报错：

```
no matching function call for this
```

同样的写法如果宽度来自 `uint32_t` 变量，则可以正常编译：

```cpp
uint32_t n64 = 64;
MaskReg m64 = UpdateMask<bfloat16_t>(n64);   // OK
```

### 原因

本仓库使用的 `UpdateMask<T>()` 重载只匹配 `uint32_t` 实参，未定型的
`int` 字面量在模板/重载择选中不参与匹配。规则可以固化为：
**传给 `UpdateMask` 的宽度必须是 `uint32_t` 变量或显式 `static_cast<uint32_t>`，
`int`、`int64_t`、`constexpr` 宽度表达式都不保证匹配。**

仓库里所有可编译的调用都遵循“先把宽度放进 `uint32_t` 变量，再调用
`UpdateMask`”的写法，例如：

```cpp
uint32_t n128 = 128;
MaskReg m128 = UpdateMask<bfloat16_t>(n128);
MaskReg lower = UpdateMask<float>(lowerCount);   // lowerCount 本身就是 uint32_t
```

`CreateMask<T, MaskPattern::ALL>()` / `CreateMask<T, MaskPattern::VL32>()`
不受此限制，可以直接使用。

### 修复模式

任选其一：

```cpp
// 1) 与仓库现有风格一致
uint32_t n64 = 64;
MaskReg m64 = UpdateMask<bfloat16_t>(n64);

// 2) 显式转换
MaskReg m64 = UpdateMask<bfloat16_t>(static_cast<uint32_t>(64));
```

### 自检命令

提交前扫描 kernel 目录：

```bash
# 1) 直接传字面量：一定有问题
grep -rnE "UpdateMask<[^>]+>\([0-9]+\)" \
  fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel

# 2) 传 int64_t / constexpr 常量（如 kChunk）：先转 uint32_t 再传
grep -rnE "UpdateMask<[^>]+>\((kChunk|kDim|kBt|64|128)" \
  fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel
```

两条都应无输出（`CreateMask<T, MaskPattern::...>` 不在此列）。

### 关联

- 本仓库示例：`op_kernel/internal/arch35/chunk_64_sparse/sparse_vector.h`
  （`NormalizeRowsNzVF`、`APrimeNzVF`、`Right64NzVF` 等均使用 `uint32_t` 变量）。
- 该问题在历史分支上出现过多次，属于“看起来应该能编译”的隐式转换陷阱，
  新增任何 `m16/m64/m128/lanes` 掩码时都应先查此条。
