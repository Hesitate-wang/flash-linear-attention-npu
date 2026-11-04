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
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--save-tensors", action="store_true")
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
    if args.allow_neg_eigval and not args.beta_sigmoid:
        parser.error("allow-neg-eigval requires --beta-sigmoid")
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
    config["l2norm"] = not args.no_l2norm
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


def canonical_chunk_indices(cu: tuple[int, ...] | None) -> list[int] | None:
    if cu is None:
        return None
    result = []
    for sequence, (begin, end) in enumerate(zip(cu, cu[1:])):
        for local_chunk in range((end - begin + 63) // 64):
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
    ratio = hv // config["key_heads"]
    sequence_count = len(cu) - 1 if cu is not None else b
    o = torch.zeros((b, hv, t, 128), dtype=torch.float64)
    final = torch.zeros((sequence_count, hv, 128, 128), dtype=torch.float64)
    g_cumsum = torch.zeros((b, t, hv), dtype=torch.float64)
    a_output = torch.zeros((b, hv, t, 64), dtype=torch.float64)
    if cu is None:
        total_chunks = b * ((t + 63) // 64)
        h_output = torch.zeros((b, (t + 63) // 64, hv, 128, 128), dtype=torch.bfloat16)
    else:
        total_chunks = sum((end - begin + 63) // 64 for begin, end in zip(cu, cu[1:]))
        h_output = torch.zeros((1, total_chunks, hv, 128, 128), dtype=torch.bfloat16)
    initial = inputs["initial_state"]

    for batch, sequence, begin, end in sequence_ranges(config, cu):
        chunk_base = 0 if cu is None else sum(
            (cu[index + 1] - cu[index] + 63) // 64 for index in range(sequence))
        for head in range(hv):
            key_head = head // ratio
            if initial is None:
                state = torch.zeros((128, 128), dtype=torch.float64)
            else:
                state = initial[sequence, head].double().clone()
                if config["state_v_first"]:
                    state = state.transpose(0, 1).contiguous()
            for local_chunk, chunk_begin in enumerate(range(begin, end, 64)):
                tile = local_chunk if cu is None else chunk_base + local_chunk
                state_image = state.transpose(0, 1) if config["state_v_first"] else state
                h_output[batch, tile, head] = state_image.to(torch.bfloat16)
                chunk_end = min(chunk_begin + 64, end)
                for token in range(chunk_begin, chunk_end):
                    decay = torch.exp2(g[batch, token, head]) if config["use_exp2"] else torch.exp(g[batch, token, head])
                    state = state * decay
                    key = k[batch, key_head, token]
                    delta = (v[batch, head, token] - key @ state) * beta[batch, token, head]
                    state = state + torch.outer(key, delta)
                    o[batch, head, token] = (q[batch, key_head, token] @ state) * config["scale"]
            if config["state_v_first"]:
                state = state.transpose(0, 1).contiguous()
            final[sequence, head] = state

        for chunk_begin in range(begin, end, 64):
            chunk_end = min(chunk_begin + 64, end)
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

    result = {
        "o": o.to(torch.bfloat16).transpose(1, 2).contiguous(),
        "g_cumsum": g_cumsum.float(),
        "A": a_output.to(torch.bfloat16),
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


def error_metrics(actual: torch.Tensor, expected: torch.Tensor, rtol: float, atol: float) -> dict:
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: actual={tuple(actual.shape)}, expected={tuple(expected.shape)}")
    actual = actual.detach().cpu().float()
    expected = expected.detach().cpu().float()
    difference = (actual - expected).abs()
    denominator = expected.abs().clamp_min(1e-6)
    relative = difference / denominator
    mismatches = ~torch.isclose(actual, expected, rtol=rtol, atol=atol)
    worst = int(difference.flatten().argmax()) if difference.numel() else 0
    return {
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean()) if difference.numel() else 0.0,
        "rmse": float(difference.square().mean().sqrt()) if difference.numel() else 0.0,
        "max_rel": float(relative.max()) if relative.numel() else 0.0,
        "mean_rel": float(relative.mean()) if relative.numel() else 0.0,
        "mismatch_count": int(mismatches.sum()),
        "mismatch_ratio": float(mismatches.float().mean()) if mismatches.numel() else 0.0,
        "worst_flat_index": worst,
        "worst_actual": float(actual.flatten()[worst]) if actual.numel() else 0.0,
        "worst_expected": float(expected.flatten()[worst]) if expected.numel() else 0.0,
        "passed": not bool(mismatches.any()),
    }


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
        cu = parse_cu(config)
        inputs = make_inputs(config, cu)
        for tensor_name, tensor in inputs.items():
            if tensor is not None:
                logger.info("input %s %s", tensor_name, json.dumps(tensor_stats(tensor)))

        logger.info("CPU golden started")
        golden_started = time.perf_counter()
        expected = golden(inputs, config, cu)
        summary["golden_seconds"] = time.perf_counter() - golden_started
        logger.info("CPU golden finished seconds=%.6f", summary["golden_seconds"])
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

        import torch_npu
        from fla_npu.ops import ascendc
        logger.info("torch_npu=%s path=%s", getattr(torch_npu, "__version__", "unknown"), torch_npu.__file__)
        logger.info("ascendc path=%s", ascendc.__file__)
        torch.npu.set_device(config["device"])
        device = torch.device(f"npu:{config['device']}")
        npu_inputs = {key: (None if value is None else value.to(device).contiguous())
                      for key, value in inputs.items()}
        if config["state_v_first"] and npu_inputs["initial_state"] is not None:
            npu_inputs["initial_state"] = npu_inputs["initial_state"].transpose(-1, -2).contiguous()
        chunk_indices = canonical_chunk_indices(cu)

        def invoke():
            return ascendc.chunk_gated_delta_rule_fwd(
                npu_inputs["q"], npu_inputs["k"], npu_inputs["v"], npu_inputs["g"], npu_inputs["beta"],
                initial_state=npu_inputs["initial_state"], output_final_state=config["final_state"],
                chunk_size=64, cu_seqlens=None if cu is None else list(cu), chunk_indices=chunk_indices,
                scale=config["scale"], use_exp2=config["use_exp2"],
                use_qk_l2norm_in_kernel=config["l2norm"],
                use_beta_sigmoid_in_kernel=config["beta_sigmoid"],
                allow_neg_eigval=config["allow_neg_eigval"], state_v_first=config["state_v_first"],
                disable_recompute=True, return_intermediate_states=True, layout="BNSD")

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
        output_stats = {}
        for tensor_name, tensor in actual.items():
            if tensor is not None:
                output_stats[tensor_name] = tensor_stats(tensor)
                logger.info("DUT output %s %s", tensor_name, json.dumps(output_stats[tensor_name]))
        summary["output_stats"] = output_stats

        metrics = {}
        for tensor_name, expected_tensor in expected.items():
            if actual.get(tensor_name) is None:
                raise RuntimeError(f"DUT output {tensor_name} is required but returned None")
            metric = error_metrics(actual[tensor_name], expected_tensor, config["rtol"], config["atol"])
            metrics[tensor_name] = metric
            logger.info("compare %s %s", tensor_name, json.dumps(metric))
            if tensor_name == "h":
                chunk_metrics = []
                for batch_index in range(actual["h"].shape[0]):
                    for tile_index in range(actual["h"].shape[1]):
                        chunk_metric = error_metrics(actual["h"][batch_index, tile_index],
                                                     expected_tensor[batch_index, tile_index],
                                                     config["rtol"], config["atol"])
                        chunk_metrics.append({"batch": batch_index, "tile": tile_index,
                                              "max_abs": chunk_metric["max_abs"],
                                              "mismatch_count": chunk_metric["mismatch_count"],
                                              "passed": chunk_metric["passed"]})
                        logger.info("compare h batch=%d tile=%d %s", batch_index, tile_index,
                                    json.dumps(chunk_metrics[-1]))
                summary["h_chunk_metrics"] = chunk_metrics
        summary["metrics"] = metrics
        summary["status"] = "passed" if all(metric["passed"] for metric in metrics.values()) else "failed"
        if config["save_tensors"]:
            torch.save(inputs, run_dir / "inputs.pt")
            torch.save({key: (None if value is None else value.cpu()) for key, value in actual.items()},
                       run_dir / "dut_outputs.pt")
            torch.save(expected, run_dir / "golden_outputs.pt")
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
