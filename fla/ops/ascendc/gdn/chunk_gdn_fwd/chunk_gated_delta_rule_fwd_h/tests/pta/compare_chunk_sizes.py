#!/usr/bin/env python3
"""比较 chunk_size=32 与 64 的算子输出是否一致（同源 WY 输入）。

为什么不能直接拿同一份 (k, w, u, g) 跑两次：

    fwd_h 的输入 w/u 是 WY 变换后的结果，变换依赖 chunk_size：
        M[t,s] = beta_t * (k_t·k_s) * exp(g_t - g_s),  s < t
        A      = inv(I + M)
        w      = A @ (beta_t * exp(g_t) * k_t)
        u      = A @ (beta_t * v_t)
    且 g 是 chunk 内局部 cumsum（chunk_local_cumsum 语义）。
    同一份 w/u 喂给 32 和 64 会得到两个不同的函数，比较无意义。

正确的交叉验证：

    1. 固定 (k, v, beta, 原始 gate 增量)；
    2. 各自按目标 chunk_size 生成 g 并做 WY，得到各自的 w/u；
    3. 两次调用的 v_new、final_state 应逐元素一致；
       h 只能比较共享边界：h32[2i] == h64[i]。

用法（Ascend 950，dense 定长，K=V=128）:

    python3 compare_chunk_sizes.py --T 640 --K 128 --V 128 --B 1 --HK 1 --HV 1 --device 2
    python3 compare_chunk_sizes.py --T 16384 --K 128 --V 128 --B 1 --HK 2 --HV 14 --device 2
"""

import argparse
import sys

import torch

from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd_h


def local_gate(inc: torch.Tensor, chunk: int) -> torch.Tensor:
    """逐 chunk 做 cumsum，得到 chunk 内局部累计 gate。"""
    g = torch.zeros_like(inc)
    for s in range(0, inc.numel(), chunk):
        e = min(s + chunk, inc.numel())
        g[s:e] = -torch.cumsum(inc[s:e], 0)
    return g


def wy_inputs(k, v, beta, g, chunk):
    """按 chunk 粒度做 WY 变换：返回 w [B,HV,T,K]、u [B,HV,T,V]。"""
    B, HK, T, K = k.shape
    HV, V = v.shape[1], v.shape[3]
    group = max(HV // HK, 1)
    w = torch.zeros((B, HV, T, K), dtype=torch.float32)
    u = torch.zeros((B, HV, T, V), dtype=torch.float32)
    for b in range(B):
        for h in range(HV):
            hk = h // group
            for s in range(0, T, chunk):
                e = min(s + chunk, T)
                n = e - s
                kc = k[b, hk, s:e].float()
                vc = v[b, h, s:e].float()
                gc = g[b, h, s:e].float()
                bc = beta[b, h, s:e].float()
                m = (kc @ kc.t()) * torch.exp(gc[:, None] - gc[None, :]) * bc[:, None]
                m = torch.tril(m, -1)
                a = torch.linalg.inv(torch.eye(n) + m)
                w[b, h, s:e] = a @ ((bc * torch.exp(gc))[:, None] * kc)
                u[b, h, s:e] = a @ (bc[:, None] * vc)
    return w, u


def build_problem(args, chunk):
    """同一 seed 下按指定 chunk 生成 g 和 w/u；返回 NPU 上的算子输入。"""
    gen = torch.Generator().manual_seed(args.seed)
    k = torch.randn((args.B, args.HK, args.T, args.K), generator=gen) * 0.05
    v = torch.randn((args.B, args.HV, args.T, args.V), generator=gen) * 0.05
    beta = torch.rand((args.B, args.HV, args.T), generator=gen) * 0.8 + 0.1
    inc = torch.rand((args.B, args.HV, args.T), generator=gen) * 0.01 + 0.001
    g = torch.stack([
        torch.stack([local_gate(inc[b, h], chunk) for h in range(args.HV)])
        for b in range(args.B)
    ])
    w, u = wy_inputs(k, v, beta, g, chunk)
    dtype = getattr(torch, args.dtype)
    return (k.to(dtype).npu(), w.to(dtype).npu(), u.to(dtype).npu(),
            g.to(torch.float32).npu())


def run_op(args, chunk):
    k, w, u, g = build_problem(args, chunk)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k, w, u, g=g, output_final_state=True, chunk_size=chunk)
    return h.float().cpu(), v_new.float().cpu(), final_state.float().cpu()


def report(name, x, y, rtol):
    d = (x - y).abs()
    scale = y.abs().max().clamp_min(1e-6).item()
    norm = d.max().item() / scale
    ok = norm <= rtol
    print("  %-22s max|diff|=%.3e  max|diff|/scale=%.3e  %s"
          % (name, d.max().item(), norm, "PASS" if ok else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=640)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--V", type=int, default=128)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--HK", type=int, default=1)
    ap.add_argument("--HV", type=int, default=1)
    ap.add_argument("--chunk-a", type=int, default=32)
    ap.add_argument("--chunk-b", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--rtol", type=float, default=2e-2,
                    help="按 max|diff|/max|ref| 归一化的阈值")
    args = ap.parse_args()

    if args.T % args.chunk_b != 0:
        print("T 必须是 chunk-b 的整数倍，才能对齐 h 的共享边界")
        return 2

    print("case: B=%d HK=%d HV=%d T=%d K=%d V=%d dtype=%s"
          % (args.B, args.HK, args.HV, args.T, args.K, args.V, args.dtype))
    h_a, v_a, f_a = run_op(args, args.chunk_a)
    h_b, v_b, f_b = run_op(args, args.chunk_b)
    print("  h slices: chunk%d=%d  chunk%d=%d" % (args.chunk_a, h_a.shape[1],
                                                  args.chunk_b, h_b.shape[1]))

    ok = True
    ok &= report("v_new", v_a, v_b, args.rtol)
    ok &= report("final_state", f_a, f_b, args.rtol)
    # chunk-a=32 的偶数边界与 chunk-b=64 的全部边界对应
    aligned = h_a[:, : 2 * h_b.shape[1] : 2]
    ok &= report("h (aligned boundaries)", aligned, h_b, args.rtol)

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
