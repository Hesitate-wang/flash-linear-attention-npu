# 本地自定义知识库

本目录记录 `flash-linear-attention-npu` 仓库中反复出现的编译、运行与调试问题，
用于在改 kernel 前快速对照，避免重复踩坑。它独立于 `cannbot-knowledge`
（受治理的只读知识仓），只保存本仓库实践中的结论。

## 条目索引

| 编号 | 文件 | 主题 |
|---|---|---|
| ASC-0001 | [ascendc-cce-gotchas.md](ascendc-cce-gotchas.md#asc-0001-updatemask-不接受整型字面量) | `UpdateMask<T>()` 不接受整型字面量 |

## 记录约定

- 编号：`ASC-XXXX`，按新增顺序递增。
- 每条至少包含：首次记录日期、现象（原始报错）、原因、最小修复模式、自检命令、关联文件。
- 只记录**本仓库可复现**的结论；与具体 CANN 版本强相关时注明版本/架构（如 arch35 / Ascend 950）。
- 修改 kernel 前后可用自检命令快速扫描同类问题。
