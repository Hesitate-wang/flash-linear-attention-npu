#!/usr/bin/env python3
"""Chunk-64 producer/consumer 流水单 case 冒烟测试。

默认从冻结矩阵抽取密集（非 varlen）case 348（B=3、Hk=Hv=2、
T=511、8 个 chunk、63 token 尾块、6 条 chain、初/末状态），写成一
case JSON 后复用仓内 ATK 入口。需要 varlen 覆盖时传 `--case-id 187`：

    accuracy     scripts/run_double_benchmark.sh
                 DUT / 六 ACLNN benchmark / FP64 CPU golden
    performance  ATK performance_device + op_summary Task Duration(us)

调用前请像 tests/atk/run_test_cpu.sh 一样 source CANN、ATK 与 fla_npu
vendor 环境；本脚本只拼装命令、校验结果，不修改环境。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OP_DIR = SCRIPT_DIR.parent
DEFAULT_CASE_JSON = OP_DIR / "atk_chunk_gated_delta_rule_fwd.json"
DEFAULT_CASE_ID = 348
DEFAULT_DEVICE = 0
DOUBLE_BENCHMARK = SCRIPT_DIR / "run_double_benchmark.sh"
EXECUTOR = "executor_chunk_gated_delta_rule_fwd.py"

SPARSE_DTYPE = "bf16"
SPARSE_CHUNK = 64
SPARSE_HEAD_DIM = 128
# Ascend 950 系列最小的常规 AI Core 数量（28/32/36）；4 核 950z 另行传入。
DEFAULT_MIN_AIC_CORES = 28
OP_MARKERS = ("chunkgateddeltarulefwd", "chunk_gated_delta_rule_fwd")


def _inputs(case: dict) -> dict[str, dict]:
    return {item["name"]: item for item in case["inputs"]}


def _attr(case: dict, name: str):
    return _inputs(case)[name]["range_values"]


def pipeline_chains(case: dict) -> int:
    """返回该 case 的流水线 chain 数：dense 为 B*Hv，varlen 为 seq*Hv。"""
    inputs = _inputs(case)
    q = inputs["q"]["shape"]
    v = inputs["v"]["shape"]
    is_varlen = bool(_attr(case, "is_varlen"))
    spec = str(_attr(case, "cu_seqlens_spec"))
    sequences = None
    if is_varlen and spec not in ("", "none"):
        values = [int(value) for value in spec.split(",") if value.strip()]
        if len(values) < 2 or values[0] != 0 or values[-1] != q[2]:
            raise ValueError(f"cu_seqlens_spec 非法：{spec}")
        sequences = len(values) - 1
    return (sequences if sequences is not None else q[0]) * v[1]


def validate_pipeline_case(case: dict, min_aic_cores: int) -> int:
    """确认 case 会进入新稀疏 kernel，且 chain 数满足流水线选择条件。"""
    inputs = _inputs(case)
    q = inputs["q"]["shape"]
    k = inputs["k"]["shape"]
    v = inputs["v"]["shape"]
    dtype = str(_attr(case, "qkv_dtype")).lower()
    chunk = int(_attr(case, "chunk_size"))
    if dtype != SPARSE_DTYPE:
        raise ValueError(f"新稀疏 kernel 只支持 bf16，当前 qkv_dtype={dtype}")
    if chunk != SPARSE_CHUNK:
        raise ValueError(f"新稀疏 kernel 只支持 chunk_size=64，当前 {chunk}")
    if q[3] != SPARSE_HEAD_DIM or k[3] != SPARSE_HEAD_DIM or v[3] != SPARSE_HEAD_DIM:
        raise ValueError(
            "新稀疏 kernel 要求 K=V=128，当前 "
            f"q={q} k={k} v={v}"
        )
    if v[1] % q[1] != 0 or v[1] // q[1] > 4:
        raise ValueError(f"要求 Hv % Hk == 0 且 Hv/Hk <= 4，当前 Hk={q[1]} Hv={v[1]}")
    chains = pipeline_chains(case)
    if 2 * chains > min_aic_cores:
        raise ValueError(
            f"chains={chains}，需要 2*chains <= AIC={min_aic_cores}；"
            "否则会走串行回退，测不到流水线"
        )
    return chains


def describe_case(case: dict) -> str:
    inputs = _inputs(case)
    q = inputs["q"]["shape"]
    v = inputs["v"]["shape"]
    scenario = str(_attr(case, "scenario"))
    varlen = bool(_attr(case, "is_varlen"))
    spec = str(_attr(case, "cu_seqlens_spec"))
    chunk = int(_attr(case, "chunk_size"))
    chunks = (len(spec.split(",")) - 1) if varlen and spec not in ("", "none") else None
    return (
        f"[pipeline-smoke] case={case.get('id')} scenario={scenario} "
        f"B={q[0]} Hk={q[1]} Hv={v[1]} T={q[2]} K={q[3]} V={v[3]} "
        f"chunk={chunk} varlen={varlen} sequences={chunks} "
        f"chains={pipeline_chains(case)}"
    )


def write_one_case(case: dict, path: Path) -> None:
    """把单条 case 重编号为 0 后写入独立 JSON。"""
    one = json.loads(json.dumps(case))
    one["id"] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([one], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _run_streaming(cmd: list[str], cwd: Path, env: dict[str, str]) -> tuple[int, str]:
    print("[pipeline-smoke] $ " + " ".join(str(part) for part in cmd), flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        lines.append(line)
    return process.wait(), "".join(lines)


def run_accuracy(
    case_json: Path, device: int, atk_bin: str, result_dir: Path
) -> dict:
    env = os.environ.copy()
    env["GDN_ATK_CASE_JSON"] = str(case_json)
    env["GDN_ATK_RESULT_DIR"] = str(result_dir)
    env["ATK_BIN"] = atk_bin
    # 角色合同要求 dut/benchmark/golden 三个执行角色都出现在主进程日志里。
    # 多进程模式下第二个 NPU 节点在独立 worker 中执行，它的 role 行不会被
    # 收进 atk_task.log；run_matrix.sh 的每个分片也是强制 single-process 的。
    env["GDN_ATK_SINGLE_PROCESS"] = "1"
    env.setdefault("GDN_ATK_MAX_TASK", "1")
    rc, output = _run_streaming(
        ["bash", str(DOUBLE_BENCHMARK), str(device)], OP_DIR, env
    )
    matches = re.findall(r"双标杆精度测试完成：(.+)", output)
    if rc != 0 or not matches:
        raise RuntimeError(
            f"双标杆执行失败（rc={rc}），日志尾部：\n{output[-2000:]}"
        )
    run_dir = Path(matches[-1].strip())
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    roles = json.loads(
        (run_dir / "runtime_role_contract.json").read_text(encoding="utf-8")
    )
    problems: list[str] = []
    if not summary.get("passed"):
        problems.extend(summary.get("errors") or ["summary.passed=false"])
    if int(summary.get("expected_cases", -1)) != 1:
        problems.append(f"expected_cases={summary.get('expected_cases')} != 1")
    case_ids = sorted(
        int(value) for value in (summary.get("statistic") or {}).get("case_ids", [])
    )
    if case_ids != [0]:
        problems.append(f"case_ids={case_ids} != [0]")
    if not roles.get("passed"):
        problems.extend(roles.get("failures") or ["runtime_role_contract.passed=false"])
    if problems:
        raise RuntimeError("精度门禁失败：" + "; ".join(problems))
    print(
        f"[pipeline-smoke] 精度通过：{run_dir} "
        f"(dut/benchmark/golden 三路各执行一次)"
    )
    return {"run_dir": run_dir, "summary": summary, "roles": roles}


def find_op_summary(output_root: Path) -> Path:
    patterns = (
        "**/mindstudio_profiler_output/op_summary_*.csv",
        "**/op_summary*.csv",
    )
    found: dict[Path, Path] = {}
    for pattern in patterns:
        for path in output_root.glob(pattern):
            found[path.resolve()] = path
    if not found:
        raise RuntimeError(f"未找到 op_summary CSV：{output_root}")
    return max(found.values(), key=lambda path: path.stat().st_mtime)


def parse_task_durations(csv_path: Path) -> tuple[list[float], set[str]]:
    """读取 op_summary，返回目标融合 kernel 的正数 Task Duration。"""
    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = [field or "" for field in (reader.fieldnames or [])]
        name_field = next(
            (field for field in ("OP Type", "Op Name", "Kernel Name", "OP Name") if field in fields),
            None,
        )
        duration_field = next(
            (field for field in fields if field.strip().lower().startswith("task duration")),
            None,
        )
        if name_field is None or duration_field is None:
            raise RuntimeError(
                f"op_summary 缺少 OP Type/Task Duration 列：{fields}"
            )
        durations: list[float] = []
        names: set[str] = set()
        for row in reader:
            name = (row.get(name_field) or "").strip()
            if name:
                names.add(name)
            if not any(marker in name.lower() for marker in OP_MARKERS):
                continue
            raw = (row.get(duration_field) or "").strip().rstrip(",")
            try:
                value = float(raw)
            except ValueError:
                continue
            if value > 0:
                durations.append(value)
    return durations, names


def run_performance(
    case_json: Path, device: int, atk_bin: str, output_root: Path
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        atk_bin,
        "node",
        "--name", "npu_dut",
        "--backend", "npu",
        "--devices", str(device),
        "--output_path", str(output_root),
        "task",
        "-c", str(case_json),
        "--task", "performance_device",
        "-p", str(OP_DIR / EXECUTOR),
        "--save_data", "profile",
        "-sp",
        "-to", "2000",
    ]
    rc, output = _run_streaming(cmd, OP_DIR, os.environ.copy())
    if rc != 0:
        raise RuntimeError(
            f"performance_device 执行失败（rc={rc}），日志尾部：\n{output[-2000:]}"
        )
    csv_path = find_op_summary(output_root)
    durations, names = parse_task_durations(csv_path)
    if not durations:
        raise RuntimeError(
            f"未找到融合 kernel 的 Task Duration；CSV={csv_path}，"
            f"OP Type={sorted(names)}"
        )
    result = {
        "csv": str(csv_path),
        "samples": len(durations),
        "min_us": min(durations),
        "median_us": statistics.median(durations),
        "max_us": max(durations),
    }
    print(
        "[pipeline-smoke] 融合 kernel Task Duration(us)："
        f"samples={result['samples']} "
        f"min={result['min_us']:.3f} "
        f"median={result['median_us']:.3f} "
        f"max={result['max_us']:.3f}"
    )
    print(
        "[pipeline-smoke] 只统计 OP Type 含 ChunkGatedDeltaRuleFwd 的行；"
        "Cast/Transpose 等 wrapper kernel 不计入"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scope",
        choices=("accuracy", "performance", "all"),
        default="all",
        help="执行阶段，默认 all",
    )
    parser.add_argument("--device", type=int, default=DEFAULT_DEVICE, help="NPU 设备号")
    parser.add_argument("--case-id", type=int, default=DEFAULT_CASE_ID, help="冻结矩阵 case id")
    parser.add_argument(
        "--case-json", type=Path, default=DEFAULT_CASE_JSON, help="用例矩阵 JSON"
    )
    parser.add_argument(
        "--atk", default=os.environ.get("ATK_BIN", "atk"), help="ATK 命令"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="单 case JSON 与结果的输出目录（默认 atk_output/pipeline_smoke_<时间戳>）",
    )
    parser.add_argument(
        "--min-aic-cores",
        type=int,
        default=DEFAULT_MIN_AIC_CORES,
        help="设备 AIC 核数下限，用于确认 2*chains <= AIC；950z（4 核）传 4",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只校验用例并生成单 case JSON，不调用 ATK",
    )
    args = parser.parse_args()

    try:
        cases = json.loads(args.case_json.read_text(encoding="utf-8"))
        case = next(
            (item for item in cases if int(item.get("id", -1)) == args.case_id),
            None,
        )
        if case is None:
            raise ValueError(f"{args.case_json} 中找不到 case {args.case_id}")
        validate_pipeline_case(case, args.min_aic_cores)
        print(describe_case(case))

        work_dir = args.work_dir or (
            OP_DIR / "atk_output" / f"pipeline_smoke_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        case_json = work_dir / "one_case.json"
        write_one_case(case, case_json)
        print(f"[pipeline-smoke] 单 case JSON：{case_json}")

        if args.check_only:
            print("[pipeline-smoke] --check-only：用例检查通过，未调用 ATK")
            return 0
        if shutil.which(args.atk) is None and not Path(args.atk).is_file():
            raise RuntimeError(f"找不到 ATK 命令：{args.atk}")

        if args.scope in ("accuracy", "all"):
            run_accuracy(case_json, args.device, args.atk, work_dir / "accuracy")
        if args.scope in ("performance", "all"):
            run_performance(
                case_json, args.device, args.atk, work_dir / "performance"
            )
        print(f"[pipeline-smoke] PASS：结果目录 {work_dir}")
        return 0
    except Exception as exc:
        print(f"[pipeline-smoke] FAIL：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
