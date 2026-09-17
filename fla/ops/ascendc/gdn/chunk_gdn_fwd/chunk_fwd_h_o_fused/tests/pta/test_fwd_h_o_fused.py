# Copyright (c) 2026 Tianjin University, Ltd.
# CANN Open Software License Agreement Version 2.0.

from __future__ import annotations

import argparse
import math
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
        help="Use the A5 exp2/BSND path instead of the A2 exp/BNSD path.",
    )
    parser.add_argument(
        "--compare-composed",
        action="store_true",
        help="Also run chunk_gated_delta_rule_fwd_h followed by chunk_fwd_o.",
    )
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch <= 0 or args.tokens <= 0 or args.k_heads <= 0 or args.v_heads <= 0:
        raise ValueError("batch, tokens and head counts must be positive")
    if args.v_heads < args.k_heads or args.v_heads % args.k_heads != 0:
        raise ValueError("v-heads must be divisible by k-heads")
    if args.use_exp2:
        if args.dtype != "bfloat16":
            raise ValueError("the A5 exp2 path requires --dtype bfloat16")
        if args.value_dim != 128 or args.chunk_size != 64:
            raise ValueError("the A5 exp2 path requires --value-dim 128 --chunk-size 64")
        if args.v_heads // args.k_heads > 4:
            raise ValueError("the A5 exp2 path requires v-heads/k-heads <= 4")


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


def cpu_reference(
    case: Case, chunk_size: int, scale: float, use_exp2: bool
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
    if use_exp2:
        output = output.permute(0, 2, 1, 3).contiguous()
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


def assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
) -> None:
    actual_cpu = actual.detach().cpu().float()
    expected_cpu = expected.detach().cpu().float()
    max_abs = (actual_cpu - expected_cpu).abs().max().item()
    print(f"{name}: shape={tuple(actual.shape)}, max_abs={max_abs:.6e}")
    torch.testing.assert_close(actual_cpu, expected_cpu, rtol=rtol, atol=atol, msg=name)


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("No available NPU device")

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    scale = args.scale if args.scale is not None else 1.0 / math.sqrt(128)

    cpu_case = make_case(args)
    expected_o, expected_final = cpu_reference(
        cpu_case, args.chunk_size, scale, args.use_exp2
    )
    npu_case = to_npu(cpu_case, args.device)

    fused_o, fused_final = ascendc.chunk_fwd_h_o_fused(
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
        output_layout="BSND" if args.use_exp2 else "BNSD",
    )
    torch.npu.synchronize()
    assert_close("fused.o vs cpu", fused_o, expected_o, args.rtol, args.atol)
    if args.output_final_state:
        assert fused_final is not None
        assert_close(
            "fused.final_state vs cpu",
            fused_final,
            expected_final,
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
            output_layout="BSND" if args.use_exp2 else "BNSD",
        )
        torch.npu.synchronize()
        assert_close("fused.o vs composed.o", fused_o, composed_o, args.rtol, args.atol)
        if args.output_final_state:
            assert fused_final is not None and composed_final is not None
            assert_close(
                "fused.final_state vs composed.final_state",
                fused_final,
                composed_final,
                args.rtol,
                args.atol,
            )

    print("PASS: ChunkFwdHOFused validation completed")


if __name__ == "__main__":
    main()
