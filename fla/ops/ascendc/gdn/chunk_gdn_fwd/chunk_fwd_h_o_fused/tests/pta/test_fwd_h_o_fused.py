# Copyright (c) 2026 Tianjin University, Ltd.
# CANN Open Software License Agreement Version 2.0.

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch_npu

from fla_npu.ops import ascendc


@dataclass
class Case:
    q: torch.Tensor
    k: torch.Tensor
    w: torch.Tensor
    u: torch.Tensor
    g: torch.Tensor
    initial_state: Optional[torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate ChunkFwdHOFused and optionally compare it with H -> O."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--runtime",
        choices=("direct", "legacy"),
        default="direct",
        help="Use the decoupled ctypes path or the opt-in torch.ops.npu extension.",
    )
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=192)
    parser.add_argument("--k-heads", type=int, default=1)
    parser.add_argument("--v-heads", type=int, default=1)
    parser.add_argument("--value-dim", type=int, choices=(128, 256), default=128)
    parser.add_argument("--chunk-size", type=int, choices=(64, 128), default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--gate-dtype", choices=("input", "float32"), default="float32")
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--initial-state", action="store_true")
    parser.add_argument("--output-final-state", action="store_true")
    parser.add_argument(
        "--use-exp2",
        action="store_true",
        help="Use the A5 exp2 path instead of the natural-exp path.",
    )
    parser.add_argument(
        "--output-layout",
        choices=("auto", "BNSD", "NTD", "BSND", "TND"),
        default="auto",
        help="Output layout; auto selects BNSD for exp and BSND for exp2.",
    )
    parser.add_argument(
        "--compare-composed",
        action="store_true",
        help="Also run chunk_gated_delta_rule_fwd_h followed by chunk_fwd_o.",
    )
    parser.add_argument(
        "--use-actual-input",
        "--use_actual_input",
        dest="use_actual_input",
        action="store_true",
        help="Load q/k/w/u/g and optional initial_state from --data-path.",
    )
    parser.add_argument(
        "--use-actual-output",
        "--use_actual_output",
        dest="use_actual_output",
        action="store_true",
        help="Load the expected o/final_state from --data-path.",
    )
    parser.add_argument(
        "--data-path",
        "--data_path",
        type=str,
        default=None,
        help="A torch.save/.pt file containing the actual input/output tensors.",
    )
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch <= 0 or args.tokens <= 0 or args.k_heads <= 0 or args.v_heads <= 0:
        raise ValueError("batch, tokens and head counts must be positive")
    if (args.use_actual_input or args.use_actual_output) and not args.data_path:
        raise ValueError("--data-path is required with --use-actual-input/--use-actual-output")
    if args.data_path and not os.path.isfile(args.data_path):
        raise FileNotFoundError(args.data_path)
    if args.v_heads < args.k_heads or args.v_heads % args.k_heads != 0:
        raise ValueError("v-heads must be divisible by k-heads")
    output_layout = resolve_output_layout(args)
    valid_layouts = ("BSND", "TND") if args.use_exp2 else ("BNSD", "NTD")
    if output_layout not in valid_layouts:
        raise ValueError(
            f"use_exp2={args.use_exp2} requires output layout in {valid_layouts}"
        )
    if output_layout in ("NTD", "TND") and args.batch != 1:
        raise ValueError("NTD/TND output requires --batch 1")
    if args.use_exp2:
        if args.dtype != "bfloat16":
            raise ValueError("the A5 exp2 path requires --dtype bfloat16")
        if args.value_dim != 128 or args.chunk_size != 64:
            raise ValueError("the A5 exp2 path requires --value-dim 128 --chunk-size 64")
        if args.v_heads // args.k_heads > 4:
            raise ValueError("the A5 exp2 path requires v-heads/k-heads <= 4")


def resolve_output_layout(args: argparse.Namespace) -> str:
    if args.output_layout != "auto":
        return args.output_layout
    return "BSND" if args.use_exp2 else "BNSD"


def make_case(args: argparse.Namespace) -> Case:
    input_dtype = getattr(torch, args.dtype)
    gate_dtype = torch.float32 if args.gate_dtype == "float32" else input_dtype
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    b, t, hk, hv, k_dim, v_dim = (
        args.batch,
        args.tokens,
        args.k_heads,
        args.v_heads,
        128,
        args.value_dim,
    )

    def normal(shape: Tuple[int, ...], magnitude: float = 0.02) -> torch.Tensor:
        values = torch.randn(shape, generator=generator, dtype=torch.float32)
        return (values * magnitude).to(input_dtype)

    g = torch.empty((b, hv, t), dtype=torch.float32)
    for begin in range(0, t, args.chunk_size):
        end = min(begin + args.chunk_size, t)
        increments = -torch.rand((b, hv, end - begin), generator=generator) * 0.01
        g[..., begin:end] = increments.cumsum(dim=-1)
    g = g.to(gate_dtype)

    initial_state = None
    if args.initial_state:
        initial_state = normal((b, hv, k_dim, v_dim), magnitude=0.01).float()

    return Case(
        q=normal((b, hk, t, k_dim)),
        k=normal((b, hk, t, k_dim)),
        w=normal((b, hv, t, k_dim)),
        u=normal((b, hv, t, v_dim)),
        g=g,
        initial_state=initial_state,
    )


def _load_pt(path: str) -> dict:
    """Load a tensor dictionary while supporting older torch versions."""
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"expected a tensor dictionary in {path}, got {type(data)!r}")
    return data


def _get_tensor(data: dict, *names: str, required: bool = True) -> Optional[torch.Tensor]:
    for name in names:
        value = data.get(name)
        if value is not None:
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            return value.detach().cpu()
    if required:
        raise KeyError(f"none of {names!r} is present in the actual-data file")
    return None


def _canonical_sequence_tensor(
    value: torch.Tensor,
    name: str,
    batch: int,
    heads: int,
    tokens: int,
    dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Accept [B,H,T,D] and the reference-test [B,T,H,D] layout."""
    if value.ndim != 4:
        raise ValueError(f"{name} must be rank 4, got shape {tuple(value.shape)}")
    if value.shape[1] == tokens and value.shape[2] != tokens:
        value = value.permute(0, 2, 1, 3)
    if value.shape[0] < batch or value.shape[1] < heads or value.shape[2] < tokens:
        raise ValueError(
            f"{name} shape {tuple(value.shape)} cannot provide "
            f"[{batch}, {heads}, {tokens}, {dim}]"
        )
    if value.shape[3] != dim:
        raise ValueError(f"{name} last dimension must be {dim}, got {value.shape[3]}")
    return value[:batch, :heads, :tokens, :].to(dtype).contiguous()


def _canonical_gate(
    value: torch.Tensor, batch: int, heads: int, tokens: int, dtype: torch.dtype
) -> torch.Tensor:
    """Accept [B,H,T] and [B,T,H] gate layouts."""
    if value.ndim != 3:
        raise ValueError(f"g must be rank 3, got shape {tuple(value.shape)}")
    if value.shape[1] == tokens and value.shape[2] != tokens:
        value = value.permute(0, 2, 1)
    if value.shape[0] < batch or value.shape[1] < heads or value.shape[2] < tokens:
        raise ValueError(
            f"g shape {tuple(value.shape)} cannot provide [{batch}, {heads}, {tokens}]"
        )
    return value[:batch, :heads, :tokens].to(dtype).contiguous()


def _canonical_initial_state(
    value: torch.Tensor, batch: int, heads: int, k_dim: int, v_dim: int, dtype: torch.dtype
) -> torch.Tensor:
    # The fused API consumes [N, HV, K, V].  H/O reference files may retain a
    # singleton token-batch dimension as [B, HV, 1, K, V].
    if value.ndim == 5:
        if value.shape[2] == 1:
            value = value[:, :, 0]
        elif value.shape[0] == 1 and value.shape[2] == batch:
            value = value[0].permute(1, 0, 2, 3)
        else:
            raise ValueError(f"unsupported initial_state shape {tuple(value.shape)}")
    if value.ndim != 4:
        raise ValueError(f"initial_state must be rank 4/5, got {tuple(value.shape)}")
    if value.shape[0] < batch or value.shape[1] < heads:
        raise ValueError(
            f"initial_state shape {tuple(value.shape)} cannot provide "
            f"[{batch}, {heads}, {k_dim}, {v_dim}]"
        )
    if value.shape[2:] != (k_dim, v_dim):
        raise ValueError(
            f"initial_state trailing shape must be {(k_dim, v_dim)}, got {tuple(value.shape[2:])}"
        )
    return value[:batch, :heads].to(dtype).contiguous()


def load_actual_case(args: argparse.Namespace) -> Case:
    data = _load_pt(args.data_path)
    input_dtype = getattr(torch, args.dtype)
    gate_dtype = torch.float32 if args.gate_dtype == "float32" else input_dtype
    q = _canonical_sequence_tensor(
        _get_tensor(data, "q", "query"), "q", args.batch, args.k_heads, args.tokens, 128, input_dtype
    )
    k = _canonical_sequence_tensor(
        _get_tensor(data, "k", "key"), "k", args.batch, args.k_heads, args.tokens, 128, input_dtype
    )
    w = _canonical_sequence_tensor(
        _get_tensor(data, "w"), "w", args.batch, args.v_heads, args.tokens, 128, input_dtype
    )
    u = _canonical_sequence_tensor(
        _get_tensor(data, "u", "v"),
        "u",
        args.batch,
        args.v_heads,
        args.tokens,
        args.value_dim,
        input_dtype,
    )
    g = _canonical_gate(
        _get_tensor(data, "g", "gate"), args.batch, args.v_heads, args.tokens, gate_dtype
    )
    initial_state = None
    if args.initial_state:
        initial_state = _canonical_initial_state(
            _get_tensor(data, "initial_state"),
            args.batch,
            args.v_heads,
            128,
            args.value_dim,
            torch.float32,
        )
    return Case(q=q, k=k, w=w, u=u, g=g, initial_state=initial_state)


def _canonical_actual_output(
    value: torch.Tensor, args: argparse.Namespace, output_layout: str
) -> torch.Tensor:
    """Normalize an actual output into the selected operator output layout."""
    value = value.detach().cpu()
    if value.ndim == 4:
        if value.shape[1] == args.tokens and value.shape[2] != args.tokens:
            value = value.permute(0, 2, 1, 3)
        if value.shape[0] < args.batch or value.shape[1] < args.v_heads or value.shape[2] < args.tokens:
            raise ValueError(f"o shape {tuple(value.shape)} is smaller than the requested case")
        value = value[: args.batch, : args.v_heads, : args.tokens, : args.value_dim].contiguous()
        if output_layout == "BNSD":
            return value
        if output_layout == "BSND":
            return value.permute(0, 2, 1, 3).contiguous()
        if args.batch != 1:
            raise ValueError(f"{output_layout} actual output requires batch=1")
        value = value.squeeze(0)
        return value if output_layout == "NTD" else value.permute(1, 0, 2).contiguous()
    if value.ndim == 3 and args.batch == 1:
        if output_layout == "NTD":
            if value.shape[0] == args.tokens and value.shape[1] == args.v_heads:
                value = value.permute(1, 0, 2)
            return value[: args.v_heads, : args.tokens, : args.value_dim].contiguous()
        if output_layout == "TND":
            if value.shape[0] == args.v_heads and value.shape[1] == args.tokens:
                value = value.permute(1, 0, 2)
            return value[: args.tokens, : args.v_heads, : args.value_dim].contiguous()
    raise ValueError(f"unsupported actual output shape {tuple(value.shape)} for {output_layout}")


def load_actual_output(args: argparse.Namespace, output_layout: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    data = _load_pt(args.data_path)
    actual_o = _canonical_actual_output(
        _get_tensor(data, "o", "ref_o", "output"), args, output_layout
    )
    actual_final = _get_tensor(data, "final_state", "ref_final_state", required=False)
    if actual_final is not None:
        actual_final = _canonical_initial_state(
            actual_final,
            args.batch,
            args.v_heads,
            128,
            args.value_dim,
            torch.float32,
        )
    return actual_o, actual_final


def cpu_reference(
    case: Case, chunk_size: int, scale: float, use_exp2: bool, output_layout: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = case.q.float()
    k = case.k.float()
    w = case.w.float()
    u = case.u.float()
    g = case.g.float()
    input_dtype = case.k.dtype
    b, hk, tokens, k_dim = k.shape
    hv, value_dim = u.shape[1], u.shape[3]
    chunks = math.ceil(tokens / chunk_size)
    head_ratio = hv // hk

    h = torch.empty((b, hv, chunks, k_dim, value_dim), dtype=input_dtype)
    v_new = torch.empty_like(case.u)
    final_state = torch.empty((b, hv, k_dim, value_dim), dtype=torch.float32)

    for batch_idx in range(b):
        for v_head in range(hv):
            state = (
                torch.zeros((k_dim, value_dim), dtype=torch.float32)
                if case.initial_state is None
                else case.initial_state[batch_idx, v_head].float().clone()
            )
            k_head = v_head // head_ratio
            for chunk_idx, begin in enumerate(range(0, tokens, chunk_size)):
                end = min(begin + chunk_size, tokens)
                h[batch_idx, v_head, chunk_idx] = state.to(input_dtype)
                current_v = u[batch_idx, v_head, begin:end] - (
                    w[batch_idx, v_head, begin:end] @ state
                )
                v_new[batch_idx, v_head, begin:end] = current_v.to(input_dtype)
                gate = g[batch_idx, v_head, begin:end]
                gate_last = gate[-1]
                state = state * torch.exp(gate_last) + (
                    k[batch_idx, k_head, begin:end].transpose(0, 1)
                    @ (current_v * torch.exp(gate_last - gate).unsqueeze(-1))
                )
            final_state[batch_idx, v_head] = state

    output = torch.empty_like(case.u)
    for batch_idx in range(b):
        for v_head in range(hv):
            k_head = v_head // head_ratio
            for chunk_idx, begin in enumerate(range(0, tokens, chunk_size)):
                end = min(begin + chunk_size, tokens)
                q_chunk = q[batch_idx, k_head, begin:end]
                k_chunk = k[batch_idx, k_head, begin:end]
                v_chunk = v_new[batch_idx, v_head, begin:end].float()
                gate = g[batch_idx, v_head, begin:end]
                attention = q_chunk @ k_chunk.transpose(0, 1)
                gate_fn = torch.exp2 if use_exp2 else torch.exp
                attention *= gate_fn(gate[:, None] - gate[None, :])
                attention = torch.tril(attention)
                state_term = (
                    q_chunk * gate_fn(gate).unsqueeze(-1)
                ) @ h[batch_idx, v_head, chunk_idx].float()
                output[batch_idx, v_head, begin:end] = (
                    (state_term + attention @ v_chunk) * scale
                ).to(input_dtype)
    if output_layout == "BSND":
        output = output.permute(0, 2, 1, 3).contiguous()
    elif output_layout == "NTD":
        output = output.squeeze(0).contiguous()
    elif output_layout == "TND":
        output = output.permute(0, 2, 1, 3).squeeze(0).contiguous()
    return output, final_state


def to_npu(case: Case, device: int) -> Case:
    target = torch.device(f"npu:{device}")
    return Case(
        q=case.q.to(target),
        k=case.k.to(target),
        w=case.w.to(target),
        u=case.u.to(target),
        g=case.g.to(target),
        initial_state=(
            None if case.initial_state is None else case.initial_state.to(target)
        ),
    )


def data_compare(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
) -> None:
    """Compare using the legacy PTA mixed absolute/symmetric-relative rule.

    ``atol`` is the per-element difference threshold.  ``rtol`` is retained
    as the allowed failure fraction to keep the existing CLI/API stable; this
    mirrors ``data_compare_h.py`` where ``pct_thd`` is the allowed failure
    fraction.  A result is accepted when at least ``(1 - rtol)`` of elements
    pass and the maximum normalized error is below the legacy 0.1 cap.
    """
    actual_cpu = actual.detach().cpu().float()
    expected_cpu = expected.detach().cpu().float()
    if actual_cpu.shape != expected_cpu.shape:
        raise AssertionError(
            f"{name}: shape mismatch, actual={tuple(actual_cpu.shape)}, "
            f"expected={tuple(expected_cpu.shape)}"
        )

    actual_flat = actual_cpu.reshape(-1)
    expected_flat = expected_cpu.reshape(-1)
    abs_diff = (actual_flat - expected_flat).abs()
    finite = torch.isfinite(actual_flat) & torch.isfinite(expected_flat)

    # Match data_compare_h.py::cal_relative_diff_np.  The floor is derived
    # from the fp16 quantum and prevents tiny reference values from producing
    # meaningless relative-error spikes.
    denominator_floor = (2.0 ** -14) / atol
    denominator = torch.maximum(
        torch.maximum(actual_flat.abs(), expected_flat.abs()),
        torch.full_like(expected_flat, denominator_floor),
    ) + 1e-9
    mixed_diff = torch.where(abs_diff < atol, abs_diff, abs_diff / denominator)

    # Exact matches are omitted by the reference implementation as well.
    compared = finite & (abs_diff > 0)
    passed = finite & ((abs_diff == 0) | (mixed_diff <= atol))

    failed_indices = torch.nonzero(~passed, as_tuple=False).flatten().tolist()
    failed_count = int((~passed).sum().item())
    max_abs = float(abs_diff.max().item()) if abs_diff.numel() else 0.0
    failed_diff = mixed_diff[compared & (mixed_diff > atol)]
    max_relative = float(failed_diff.max().item()) if failed_diff.numel() else 0.0
    element_count = actual_flat.numel()
    fulfill_percent = (
        100.0 * (element_count - failed_count) / element_count
        if element_count
        else 100.0
    )
    required_percent = (1.0 - rtol) * 100.0
    passed = (
        fulfill_percent >= required_percent
        and max_relative < 0.1
        and bool(torch.all(finite))
    )

    # Match the existing PTA data_compare format.  For a clean result print
    # only the beginning/end of the flattened data; for a failing result print
    # only failing rows, capped at 50 entries.
    if failed_indices:
        display_indices = failed_indices[:50]
        title = f"{name} error rows ({len(display_indices)}/{len(failed_indices)})"
        insert_gap = False
    else:
        if element_count <= 40:
            display_indices = list(range(element_count))
            insert_gap = False
        else:
            display_indices = list(range(20)) + list(range(element_count - 20, element_count))
            insert_gap = True
        title = name

    print("-" * 95)
    print(f"{title}: shape={tuple(actual_cpu.shape)}, elements={actual_flat.numel()}")
    print("Loop\tExpectOut\tRealOut\tFpDiff\tRateDiff")
    print("-" * 95)
    for display_position, index in enumerate(display_indices):
        if insert_gap and display_position == 20:
            print("...\t...\t...\t...\t...")
        expected_value = float(expected_flat[index])
        actual_value = float(actual_flat[index])
        abs_value = abs(expected_value - actual_value)
        # Keep the reference helper's interpretation: small absolute errors
        # are displayed as absolute values, otherwise display relative error.
        rate_value = abs_value if abs_value < atol else abs_value / (
            max(abs(expected_value), abs(actual_value), denominator_floor) + 1e-9
        )
        print(
            f"{index + 1:08d}\t{expected_value:.7f}\t{actual_value:.7f}\t"
            f"{abs_value:.7f}\t{rate_value:.7f}"
        )
    if failed_indices and len(failed_indices) > len(display_indices):
        print(f"... {len(failed_indices) - len(display_indices)} more error rows omitted ...")
    print("-" * 95)
    print(
        f"{name}: failed={failed_count}/{actual_flat.numel()}, "
        f"fulfill={fulfill_percent:.6f}%, required={required_percent:.6f}%, "
        f"max_abs={max_abs:.6e}, max_relative={max_relative:.6e}"
    )
    print("-" * 120)
    if not passed:
        raise AssertionError(
            f"{name}: fulfill={fulfill_percent:.6f}% (required {required_percent:.6f}%), "
            f"max_relative={max_relative:.6e} (limit 1.000000e-1)"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("No available NPU device")

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    scale = args.scale if args.scale is not None else 1.0 / math.sqrt(128)
    output_layout = resolve_output_layout(args)

    cpu_case = load_actual_case(args) if args.use_actual_input else make_case(args)
    cpu_o, cpu_final = cpu_reference(
        cpu_case, args.chunk_size, scale, args.use_exp2, output_layout
    )
    actual_o = actual_final = None
    if args.use_actual_output:
        actual_o, actual_final = load_actual_output(args, output_layout)
    npu_case = to_npu(cpu_case, args.device)

    if args.runtime == "legacy":
        import fla_npu

        fla_npu.load_legacy_torch_ops()
        fused_op = torch.ops.npu.npu_chunk_fwd_h_o_fused
    else:
        fused_op = ascendc.chunk_fwd_h_o_fused

    fused_o, fused_final = fused_op(
        npu_case.k,
        npu_case.w,
        npu_case.u,
        npu_case.g,
        npu_case.q,
        initial_state=npu_case.initial_state,
        output_final_state=args.output_final_state,
        chunk_size=args.chunk_size,
        scale=scale,
        use_exp2=args.use_exp2,
        state_v_first=False,
        output_layout=output_layout,
    )
    torch.npu.synchronize()
    # Keep the CPU-generated comparison when it was generated from the same
    # input.  If only an external output was supplied, the generated random
    # input is unrelated to that output and must not be compared with it.
    if actual_o is None or args.use_actual_input:
        data_compare("fused.o vs cpu", fused_o, cpu_o, args.rtol, args.atol)
    elif args.use_actual_output:
        print("fused.o vs cpu: skipped (actual output supplied without actual input)")
    if actual_o is not None:
        data_compare("fused.o vs actual output", fused_o, actual_o, args.rtol, args.atol)
    if args.output_final_state:
        assert fused_final is not None
        if not args.use_actual_output or args.use_actual_input:
            data_compare("fused.final_state vs cpu", fused_final, cpu_final, args.rtol, args.atol)
        elif args.use_actual_output:
            print(
                "fused.final_state vs cpu: skipped "
                "(actual output supplied without actual input)"
            )
        if actual_final is not None:
            data_compare(
                "fused.final_state vs actual output",
                fused_final,
                actual_final,
                args.rtol,
                args.atol,
            )

    if args.compare_composed:
        h, composed_v, composed_final = ascendc.chunk_gated_delta_rule_fwd_h(
            npu_case.k,
            npu_case.w,
            npu_case.u,
            npu_case.g,
            initial_state=npu_case.initial_state,
            output_final_state=args.output_final_state,
            chunk_size=args.chunk_size,
            state_v_first=False,
        )
        composed_o = ascendc.chunk_fwd_o(
            npu_case.q,
            npu_case.k,
            composed_v,
            h,
            scale,
            g=npu_case.g,
            chunk_size=args.chunk_size,
            transpose_state_layout=False,
            use_exp2=args.use_exp2,
            output_layout=output_layout,
        )
        torch.npu.synchronize()
        data_compare("fused.o vs composed.o", fused_o, composed_o, args.rtol, args.atol)
        if args.output_final_state:
            assert fused_final is not None and composed_final is not None
            data_compare(
                "fused.final_state vs composed.final_state",
                fused_final,
                composed_final,
                args.rtol,
                args.atol,
            )

    print("PASS: ChunkFwdHOFused validation completed")


if __name__ == "__main__":
    main()
