#!/usr/bin/env python3
"""Lightweight GDN forward test: one fused DUT plus a CPU FP64 golden.

No ATK and no six-operator benchmark. Run from any working directory after the
CANN, torch_npu and fla_npu environments have been configured.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
from compare_contract import data_compare

OPERATOR_DIR = TEST_DIR.parent
REPO_ROOT = OPERATOR_DIR.parents[5]
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("dense-tail", "varlen", "varlen-empty", "all"),
                        default="dense-tail")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--key-heads", type=int, default=1)
    parser.add_argument("--value-heads", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, choices=(32, 64), default=64)
    parser.add_argument("--cu-seqlens", default=None,
                        help="Comma-separated offsets; overrides the selected varlen case")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--gate-scale", type=float, default=0.05,
                        help="Magnitude of negative raw gates; large values stress exp underflow")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--no-l2norm", action="store_true")
    parser.add_argument("--use-exp2", action="store_true")
    parser.add_argument("--beta-sigmoid", action="store_true")
    parser.add_argument("--allow-neg-eigval", action="store_true")
    parser.add_argument("--state-v-first", action="store_true")
    parser.add_argument("--no-initial-state", action="store_true")
    parser.add_argument("--no-final-state", action="store_true")
    parser.add_argument("--initial-dtype", choices=tuple(DTYPES), default="fp32")
    parser.add_argument("--diff-thd", type=float, default=1e-3,
                        help="Mixed absolute/relative error threshold")
    parser.add_argument("--pct-thd", type=float, default=1e-4,
                        help="Allowed failed-element fraction; default 0.0001 = 0.01%")
    parser.add_argument("--max-diff-hd", type=float, default=0.1,
                        help="Maximum allowed mixed error")
    parser.add_argument("--save-tensors", action="store_true")
    parser.add_argument("--compare-cpu", action="store_true",
                        help="Compare DUT with the CPU FP64 golden")
    parser.add_argument("--compare-chain", action="store_true",
                        help="Compare DUT with the six-ACLNN NPU operator chain")
    parser.add_argument("--performance", action="store_true",
                        help="Skip all comparisons and only measure DUT execution")
    parser.add_argument("--dump-aprime", action="store_true",
                        help="Debug: export A' through the public A slot and compare it")
    parser.add_argument("--dump-ol", action="store_true",
                        help="Debug: export raw O_l through o and compare it")
    parser.add_argument("--dump-os", action="store_true",
                        help="Debug: export raw O_s through o and compare it")
    parser.add_argument("--dump-scalars", action="store_true",
                        help="Debug: export energy/scale/g/lastEnergy in the first four outputs")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Only validate input/golden/log generation; do not touch the NPU")
    parser.add_argument("--log-dir", type=Path, default=TEST_DIR / "logs")
    args = parser.parse_args()
    if args.batch <= 0 or args.key_heads <= 0 or args.value_heads <= 0:
        parser.error("batch/key-heads/value-heads must be positive")
    if args.value_heads % args.key_heads != 0:
        parser.error("value-heads must be divisible by key-heads")
    if args.repeat <= 0 or args.warmup < 0:
        parser.error("repeat must be positive and warmup must be nonnegative")
    if args.diff_thd <= 0 or args.max_diff_hd <= 0 or not 0 <= args.pct_thd <= 1:
        parser.error("diff-thd/max-diff-hd must be positive and pct-thd must be in [0,1]")
    if args.allow_neg_eigval and not args.beta_sigmoid and args.chunk_size != 32:
        parser.error("allow-neg-eigval requires --beta-sigmoid")
    if args.dump_aprime and args.dump_ol:
        parser.error("--dump-aprime and --dump-ol are mutually exclusive")
    if args.dump_os and (args.dump_aprime or args.dump_ol):
        parser.error("--dump-os cannot be combined with other dump modes")
    if args.dump_scalars and (args.dump_aprime or args.dump_ol or args.dump_os):
        parser.error("--dump-scalars cannot be combined with other dump modes")
    if args.cpu_only and args.compare_chain:
        parser.error("--cpu-only cannot be combined with --compare-chain")
    if args.performance and (args.compare_cpu or args.compare_chain or args.cpu_only):
        parser.error("--performance cannot be combined with comparison modes")
    if args.dump_aprime or args.dump_ol or args.dump_os or args.dump_scalars:
        parser.error("debug dump modes are disabled; only o/final_state are compared")
    if args.chunk_size == 32:
        if args.compare_chain:
            parser.error("chunk_size=32 does not support the Prepare reference chain")
        if args.use_exp2 or args.no_l2norm:
            parser.error("chunk_size=32 requires use_exp2=false and kernel L2Norm")
        if args.dump_aprime or args.dump_ol or args.dump_os or args.dump_scalars:
            parser.error("chunk64 debug dump modes are unavailable for chunk_size=32")
    return args


def case_config(args: argparse.Namespace, name: str) -> dict:
    config = copy.deepcopy(vars(args))
    config["case"] = name
    config["log_dir"] = str(args.log_dir)
    if name == "dense-tail":
        config["tokens"] = args.tokens or 65
        config["cu_seqlens"] = None
    else:
        config["tokens"] = args.tokens or 129
        if args.cu_seqlens is not None:
            config["cu_seqlens"] = args.cu_seqlens
        elif name == "varlen":
            config["cu_seqlens"] = "0,17,80,129"
        else:
            config["cu_seqlens"] = "0,0,65,65,129"
    if config["cu_seqlens"] is not None:
        config["batch"] = 1
    config["scale"] = args.scale if args.scale is not None else 128.0 ** -0.5
    debug_compare = args.dump_aprime or args.dump_ol or args.dump_os or args.dump_scalars
    if args.performance:
        config["compare_cpu"] = False
        config["compare_chain"] = False
    elif args.cpu_only or debug_compare:
        config["compare_cpu"] = True
        config["compare_chain"] = False
    elif not args.compare_cpu and not args.compare_chain:
        config["compare_cpu"] = True
        config["compare_chain"] = False
    else:
        config["compare_cpu"] = args.compare_cpu
        config["compare_chain"] = args.compare_chain
    config["l2norm"] = not args.no_l2norm
    if config["chunk_size"] == 32:
        config["beta_sigmoid"] = True
    config["initial_state"] = not args.no_initial_state
    config["final_state"] = not args.no_final_state
    return config


def parse_cu(config: dict) -> tuple[int, ...] | None:
    text = config["cu_seqlens"]
    if text is None:
        return None
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if len(values) < 2 or values[0] != 0 or values[-1] != config["tokens"]:
        raise ValueError(f"invalid cu_seqlens={values}, expected 0..T={config['tokens']}")
    if any(left > right for left, right in zip(values, values[1:])):
        raise ValueError(f"cu_seqlens must be nondecreasing: {values}")
    return values


def canonical_chunk_indices(cu: tuple[int, ...] | None, chunk_size: int) -> list[int] | None:
    if cu is None:
        return None
    result = []
    for sequence, (begin, end) in enumerate(zip(cu, cu[1:])):
        for local_chunk in range((end - begin + chunk_size - 1) // chunk_size):
            result.extend((sequence, local_chunk))
    return result


def sequence_ranges(config: dict, cu: tuple[int, ...] | None):
    if cu is None:
        for batch in range(config["batch"]):
            yield batch, batch, 0, config["tokens"]
    else:
        for sequence, (begin, end) in enumerate(zip(cu, cu[1:])):
            yield 0, sequence, begin, end


def make_inputs(config: dict, cu: tuple[int, ...] | None) -> dict:
    generator = torch.Generator().manual_seed(config["seed"])
    b, hk, hv, t = config["batch"], config["key_heads"], config["value_heads"], config["tokens"]
    q = (0.125 * torch.randn((b, hk, t, 128), generator=generator)).to(torch.bfloat16).contiguous()
    k = (0.125 * torch.randn((b, hk, t, 128), generator=generator)).to(torch.bfloat16).contiguous()
    v = (0.125 * torch.randn((b, hv, t, 128), generator=generator)).to(torch.bfloat16).contiguous()
    g = (-config["gate_scale"] * torch.rand((b, t, hv), generator=generator)).float().contiguous()
    beta = torch.rand((b, t, hv), generator=generator).to(torch.bfloat16).contiguous()
    sequence_count = len(cu) - 1 if cu is not None else b
    initial = None
    if config["initial_state"]:
        values = torch.arange(sequence_count * hv * 128 * 128, dtype=torch.float32)
        initial = ((values.remainder(257) - 128) * 1e-4).reshape(sequence_count, hv, 128, 128)
        initial = initial.to(DTYPES[config["initial_dtype"]]).contiguous()
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta, "initial_state": initial}


def effective_inputs(inputs: dict, config: dict) -> dict:
    q, k = inputs["q"], inputs["k"]
    if config["l2norm"]:
        q_rstd = torch.rsqrt(q.float().square().sum(dim=-1) + 1e-6)
        k_rstd = torch.rsqrt(k.float().square().sum(dim=-1) + 1e-6)
        q_hat = (q.float() * q_rstd.unsqueeze(-1)).to(torch.bfloat16)
        k_hat = (k.float() * k_rstd.unsqueeze(-1)).to(torch.bfloat16)
    else:
        q_rstd = k_rstd = None
        q_hat, k_hat = q, k
    beta = inputs["beta"].float()
    if config["beta_sigmoid"]:
        beta = torch.sigmoid(beta)
        if config["allow_neg_eigval"]:
            beta = beta * 2.0
    g = inputs["g"].float()
    g_scaled = g * (1.0 / math.log(2.0)) if config["use_exp2"] else g
    return {"q": q_hat, "k": k_hat, "q_rstd": q_rstd, "k_rstd": k_rstd,
            "beta": beta, "g": g, "g_scaled": g_scaled}


def golden(inputs: dict, config: dict, cu: tuple[int, ...] | None) -> dict:
    effective = effective_inputs(inputs, config)
    q = effective["q"].double()
    k = effective["k"].double()
    v = inputs["v"].double()
    beta = effective["beta"].double()
    g = effective["g"].double()
    g_scaled = effective["g_scaled"].double()
    b, hv, t = config["batch"], config["value_heads"], config["tokens"]
    chunk = config["chunk_size"]
    ratio = hv // config["key_heads"]
    sequence_count = len(cu) - 1 if cu is not None else b
    o = torch.zeros((b, hv, t, 128), dtype=torch.float64)
    vnew_output = torch.zeros((b, hv, t, 128), dtype=torch.float64)
    ol_output = torch.zeros((b, hv, t, 128), dtype=torch.float64)
    os_output = torch.zeros((b, hv, t, 128), dtype=torch.float64)
    final = torch.zeros((sequence_count, hv, 128, 128), dtype=torch.float64)
    g_cumsum = torch.zeros((b, t, hv), dtype=torch.float64)
    a_output = torch.zeros((b, hv, t, chunk), dtype=torch.float64)
    attention_output = torch.zeros((b, hv, t, chunk), dtype=torch.float64)
    if cu is None:
        total_chunks = b * ((t + chunk - 1) // chunk)
        h_output = torch.zeros((b, (t + chunk - 1) // chunk, hv, 128, 128), dtype=torch.bfloat16)
    else:
        total_chunks = sum((end - begin + chunk - 1) // chunk for begin, end in zip(cu, cu[1:]))
        h_output = torch.zeros((1, total_chunks, hv, 128, 128), dtype=torch.bfloat16)
    initial = inputs["initial_state"]

    for batch, sequence, begin, end in sequence_ranges(config, cu):
        chunk_base = 0 if cu is None else sum(
            (cu[index + 1] - cu[index] + chunk - 1) // chunk for index in range(sequence))
        for head in range(hv):
            key_head = head // ratio
            if initial is None:
                state = torch.zeros((128, 128), dtype=torch.float64)
            else:
                state = initial[sequence, head].double().clone()
                if config["state_v_first"]:
                    state = state.transpose(0, 1).contiguous()
            for local_chunk, chunk_begin in enumerate(range(begin, end, chunk)):
                tile = local_chunk if cu is None else chunk_base + local_chunk
                state_image = state.transpose(0, 1) if config["state_v_first"] else state
                h_output[batch, tile, head] = state_image.to(torch.bfloat16)
                chunk_end = min(chunk_begin + chunk, end)
                os_output[batch, head, chunk_begin:chunk_end] = (
                    q[batch, key_head, chunk_begin:chunk_end] @ state)
                for token in range(chunk_begin, chunk_end):
                    decay = torch.exp2(g[batch, token, head]) if config["use_exp2"] else torch.exp(g[batch, token, head])
                    state = state * decay
                    key = k[batch, key_head, token]
                    delta = (v[batch, head, token] - key @ state) * beta[batch, token, head]
                    vnew_output[batch, head, token] = delta
                    state = state + torch.outer(key, delta)
                    o[batch, head, token] = (q[batch, key_head, token] @ state) * config["scale"]
            if config["state_v_first"]:
                state = state.transpose(0, 1).contiguous()
            final[sequence, head] = state

        for chunk_begin in range(begin, end, chunk):
            chunk_end = min(chunk_begin + chunk, end)
            for head in range(hv):
                key_head = head // ratio
                gate = torch.cumsum(g_scaled[batch, chunk_begin:chunk_end, head], dim=0)
                g_cumsum[batch, chunk_begin:chunk_end, head] = gate
                score = k[batch, key_head, chunk_begin:chunk_end] @ k[batch, key_head, chunk_begin:chunk_end].T
                difference = gate[:, None] - gate[None, :]
                gate_scale = torch.exp2(difference) if config["use_exp2"] else torch.exp(difference)
                raw = score * gate_scale * beta[batch, chunk_begin:chunk_end, head, None]
                raw = torch.tril(raw, diagonal=-1)
                size = chunk_end - chunk_begin
                inverse = torch.linalg.inv(torch.eye(size, dtype=torch.float64) + raw)
                a_output[batch, head, chunk_begin:chunk_end, :size] = inverse
                attention = q[batch, key_head, chunk_begin:chunk_end] @ k[batch, key_head, chunk_begin:chunk_end].T
                attention = torch.tril(attention * gate_scale, diagonal=0)
                attention_output[batch, head, chunk_begin:chunk_end, :size] = attention
                ol_output[batch, head, chunk_begin:chunk_end] = (
                    attention @ vnew_output[batch, head, chunk_begin:chunk_end])

    result = {
        "o": (os_output if config["dump_os"] else (ol_output if config["dump_ol"] else o)).to(torch.bfloat16).transpose(1, 2).contiguous(),
        "g_cumsum": g_cumsum.float(),
        "A": attention_output.to(torch.bfloat16) if config["dump_aprime"] else a_output.to(torch.bfloat16),
        "h": h_output,
    }
    if config["final_state"]:
        result["final_state"] = final.to(DTYPES[config["initial_dtype"]] if initial is not None else torch.float32)
    if config["beta_sigmoid"]:
        result["beta_eff"] = effective["beta"].float()
    if config["l2norm"]:
        result.update({"q_hat": effective["q"], "k_hat": effective["k"],
                       "q_rstd": effective["q_rstd"], "k_rstd": effective["k_rstd"]})
    return result


def tensor_stats(tensor: torch.Tensor) -> dict:
    value = tensor.detach().cpu()
    flat = value.float().flatten()
    stats = {"shape": list(value.shape), "dtype": str(value.dtype), "numel": value.numel()}
    if flat.numel() == 0:
        return stats
    stats.update({
        "finite": bool(torch.isfinite(flat).all()),
        "min": float(flat.min()), "max": float(flat.max()),
        "mean": float(flat.mean()), "std": float(flat.std(unbiased=False)),
    })
    return stats


def make_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"gdn-light-{time.time_ns()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "run.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def log_environment(logger: logging.Logger, config: dict) -> None:
    logger.info("python=%s", sys.version.replace("\n", " "))
    logger.info("platform=%s", platform.platform())
    logger.info("torch=%s", torch.__version__)
    try:
        commit = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        commit = f"unavailable: {exc}"
    logger.info("repo=%s commit=%s", REPO_ROOT, commit)
    for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH"):
        logger.info("env %s=%s", name, os.environ.get(name, "<unset>"))
    logger.info("config=%s", json.dumps(config, ensure_ascii=False, sort_keys=True))


def run_one(base_args: argparse.Namespace, name: str) -> bool:
    config = case_config(base_args, name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(config["log_dir"]) / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = make_logger(run_dir)
    summary = {"case": name, "status": "failed", "log_dir": str(run_dir), "config": config}
    started = time.perf_counter()
    try:
        log_environment(logger, config)
        if not config["cpu_only"]:
            import torch_npu
            from fla_npu.ops import ascendc
            logger.info("torch_npu=%s path=%s", getattr(torch_npu, "__version__", "unknown"), torch_npu.__file__)
            logger.info("ascendc path=%s", ascendc.__file__)
            logger.info("NPU set_device started: %d", config["device"])
            torch.npu.set_device(config["device"])
            logger.info("NPU set_device finished")
        cu = parse_cu(config)
        inputs = make_inputs(config, cu)
        for tensor_name, tensor in inputs.items():
            if tensor is not None:
                logger.info("input %s %s", tensor_name, json.dumps(tensor_stats(tensor)))

        expected = {}
        if config["compare_cpu"]:
            logger.info("CPU golden started")
            golden_started = time.perf_counter()
            expected = golden(inputs, config, cu)
            summary["golden_seconds"] = time.perf_counter() - golden_started
            logger.info("CPU golden finished seconds=%.6f", summary["golden_seconds"])
            if config["chunk_size"] == 32:
                for name in ("g_cumsum", "A", "h"):
                    expected.pop(name, None)
            expected = {name: tensor for name, tensor in expected.items()
                        if name == "o" or (name == "final_state" and config["final_state"])}
        else:
            logger.info("CPU golden skipped (compare_cpu=false)")
        if config["cpu_only"]:
            golden_stats = {name: tensor_stats(tensor) for name, tensor in expected.items()}
            summary["golden_output_stats"] = golden_stats
            for tensor_name, stats in golden_stats.items():
                logger.info("golden output %s %s", tensor_name, json.dumps(stats))
            if config["save_tensors"]:
                torch.save(inputs, run_dir / "inputs.pt")
                torch.save(expected, run_dir / "golden_outputs.pt")
            summary["status"] = "cpu-only"
            summary["total_seconds"] = time.perf_counter() - started
            with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            logger.info("result=cpu-only total_seconds=%.6f summary=%s",
                        summary["total_seconds"], run_dir / "summary.json")
            for handler in list(logger.handlers):
                handler.flush()
                handler.close()
                logger.removeHandler(handler)
            return True

        device = torch.device(f"npu:{config['device']}")
        logger.info("input H2D copy started")
        npu_inputs = {key: (None if value is None else value.to(device).contiguous())
                      for key, value in inputs.items()}
        torch.npu.synchronize()
        logger.info("input H2D copy finished")
        if config["state_v_first"] and npu_inputs["initial_state"] is not None:
            logger.info("initial-state transpose started")
            npu_inputs["initial_state"] = npu_inputs["initial_state"].transpose(-1, -2).contiguous()
            torch.npu.synchronize()
            logger.info("initial-state transpose finished")
        chunk_indices = canonical_chunk_indices(cu, config["chunk_size"])
        if config["dump_aprime"]:
            os.environ["GDN_SPARSE_DEBUG_APRIME"] = "1"
            logger.warning("A' debug export enabled: public A now contains A', not the inverse coefficient")
        else:
            os.environ.pop("GDN_SPARSE_DEBUG_APRIME", None)
        if config["dump_ol"]:
            os.environ["GDN_SPARSE_DEBUG_OL"] = "1"
            logger.warning("O_l debug export enabled: public o now contains raw O_l")
        else:
            os.environ.pop("GDN_SPARSE_DEBUG_OL", None)
        if config["dump_os"]:
            os.environ["GDN_SPARSE_DEBUG_OS"] = "1"
            logger.warning("O_s debug export enabled: public o now contains raw O_s")
        else:
            os.environ.pop("GDN_SPARSE_DEBUG_OS", None)
        if config["dump_scalars"]:
            os.environ["GDN_SPARSE_DEBUG_SCALARS"] = "1"
            logger.warning("scalar debug export enabled: first four o values are energy/scale/g/lastEnergy")
        else:
            os.environ.pop("GDN_SPARSE_DEBUG_SCALARS", None)

        def invoke():
            common = dict(
                initial_state=npu_inputs["initial_state"],
                output_final_state=config["final_state"],
                cu_seqlens=None if cu is None else list(cu),
                chunk_indices=chunk_indices,
                scale=config["scale"],
                allow_neg_eigval=config["allow_neg_eigval"],
                state_v_first=config["state_v_first"],
                return_intermediate_states=False,
                layout="BNSD",
            )
            if config["chunk_size"] == 32:
                from fla_npu.ops.ascendc import _aclnn_ctypes
                return _aclnn_ctypes.npu_chunk_gated_delta_rule_fwd(
                    npu_inputs["q"], npu_inputs["k"], npu_inputs["v"],
                    npu_inputs["g"], npu_inputs["beta"],
                    chunk_size=32, use_exp2=False,
                    use_qk_l2norm_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True,
                    disable_recompute=False, **common)
            return ascendc.chunk_gated_delta_rule_fwd(
                npu_inputs["q"], npu_inputs["k"], npu_inputs["v"],
                npu_inputs["g"], npu_inputs["beta"],
                chunk_size=64, use_exp2=config["use_exp2"],
                use_qk_l2norm_in_kernel=config["l2norm"],
                use_beta_sigmoid_in_kernel=config["beta_sigmoid"],
                disable_recompute=False, **common)

        for index in range(config["warmup"]):
            logger.info("warmup %d started", index)
            invoke()
            torch.npu.synchronize()
            logger.info("warmup %d finished", index)

        durations = []
        outputs = None
        for index in range(config["repeat"]):
            logger.info("DUT iteration %d started", index)
            iteration_started = time.perf_counter()
            outputs = invoke()
            torch.npu.synchronize()
            elapsed = time.perf_counter() - iteration_started
            durations.append(elapsed)
            logger.info("DUT iteration %d finished seconds=%.6f", index, elapsed)
        summary["dut_seconds"] = {
            "min": min(durations), "median": statistics.median(durations),
            "mean": statistics.mean(durations), "max": max(durations), "all": durations,
        }
        names = ("o", "final_state", "g_cumsum", "A", "beta_eff", "h",
                 "q_hat", "k_hat", "q_rstd", "k_rstd")
        actual = dict(zip(names, outputs))
        if config["dump_scalars"]:
            scalar_values = actual["o"].detach().cpu().float().flatten()[:4].tolist()
            raw_gate = float(inputs["g"][0, 0, 0])
            expected_gate = 2.0 ** raw_gate if config["use_exp2"] else math.exp(raw_gate)
            expected_scalars = [expected_gate, config["scale"], raw_gate, expected_gate]
            summary["scalar_debug"] = {"actual": scalar_values, "expected": expected_scalars}
            logger.info("scalar debug actual=%s expected=%s", scalar_values, expected_scalars)
        output_stats = {}
        for tensor_name, tensor in actual.items():
            if tensor is not None and (tensor_name == "o" or tensor_name == "final_state"):
                output_stats[tensor_name] = tensor_stats(tensor)
                logger.info("DUT output %s %s", tensor_name, json.dumps(output_stats[tensor_name]))
        summary["output_stats"] = output_stats

        metrics = {}
        if config["compare_cpu"]:
            for tensor_name, expected_tensor in expected.items():
                if actual.get(tensor_name) is None:
                    raise RuntimeError(f"DUT output {tensor_name} is required but returned None")
                metric = data_compare(actual[tensor_name], expected_tensor, config,
                                      logger, label=f"cpu.{tensor_name}")
                metrics[tensor_name] = metric
                logger.info("compare %s %s", tensor_name, json.dumps(metric))
        summary["metrics"] = metrics

        chain_metrics = {}
        chain_outputs = None
        if config["compare_chain"]:
            from npu_chain_reference import run_npu_chain
            chain_outputs = run_npu_chain(ascendc, inputs, config, cu, chunk_indices, logger)
            chain_timings = chain_outputs.pop("_timings")
            chain_outputs = {name: tensor for name, tensor in chain_outputs.items()
                             if name == "o" or (name == "final_state" and config["final_state"])}
            summary["chain_timings"] = chain_timings
            chain_stats = {}
            for tensor_name, tensor in chain_outputs.items():
                if tensor is not None:
                    chain_stats[tensor_name] = tensor_stats(tensor)
                    logger.info("chain output %s %s", tensor_name, json.dumps(chain_stats[tensor_name]))
            summary["chain_output_stats"] = chain_stats
            for tensor_name, chain_tensor in chain_outputs.items():
                if actual.get(tensor_name) is None:
                    raise RuntimeError(f"DUT output {tensor_name} is required for chain comparison")
                metric = data_compare(actual[tensor_name], chain_tensor, config,
                                      logger, label=f"chain.{tensor_name}")
                chain_metrics[tensor_name] = metric
                logger.info("compare_chain %s %s", tensor_name, json.dumps(metric))
        summary["chain_metrics"] = chain_metrics
        all_metrics = list(metrics.values()) + list(chain_metrics.values())
        summary["status"] = "passed" if all(metric["passed"] for metric in all_metrics) else "failed"
        if config["save_tensors"]:
            torch.save(inputs, run_dir / "inputs.pt")
            torch.save({key: (None if value is None else value.cpu()) for key, value in actual.items()},
                       run_dir / "dut_outputs.pt")
            if config["compare_cpu"]:
                torch.save(expected, run_dir / "golden_outputs.pt")
            if chain_outputs is not None:
                torch.save({key: (None if value is None else value.cpu()) for key, value in chain_outputs.items()},
                           run_dir / "chain_outputs.pt")
            logger.info("tensors saved under %s", run_dir)
    except Exception as exc:
        summary["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        logger.error("test exception: %s", exc)
        logger.error("traceback:\n%s", traceback.format_exc())
        try:
            torch.npu.synchronize()
            logger.info("post-exception NPU synchronize succeeded")
        except Exception as sync_exc:
            logger.error("post-exception NPU synchronize failed: %s", sync_exc)
    summary["total_seconds"] = time.perf_counter() - started
    summary["log_dir"] = str(run_dir)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    logger.info("result=%s total_seconds=%.6f summary=%s",
                summary["status"], summary["total_seconds"], run_dir / "summary.json")
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
    return summary["status"] == "passed"


def main() -> int:
    args = parse_args()
    cases = ("dense-tail", "varlen", "varlen-empty") if args.case == "all" else (args.case,)
    passed = True
    for name in cases:
        passed = run_one(args, name) and passed
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
