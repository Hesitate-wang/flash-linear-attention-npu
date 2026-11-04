#!/usr/bin/env python3
"""Optional six-ACLNN NPU reference chain for the lightweight GDN test."""

from __future__ import annotations

import math
import time

import torch


def _expand_qk(q, k, value_heads: int):
    key_heads = q.shape[1]
    if key_heads == value_heads:
        return q, k
    ratio = value_heads // key_heads
    return (q.repeat_interleave(ratio, dim=1).contiguous(),
            k.repeat_interleave(ratio, dim=1).contiguous())


def run_npu_chain(ascendc, inputs: dict, effective: dict, config: dict,
                  cu: tuple[int, ...] | None, chunk_indices: list[int] | None,
                  logger) -> dict:
    """Run the public operator chain without ATK and return common outputs."""
    device = f"npu:{config['device']}"
    q = effective["q"].to(device).contiguous()
    k = effective["k"].to(device).contiguous()
    v = inputs["v"].to(device).contiguous()
    q, k = _expand_qk(q, k, config["value_heads"])
    # The public chain consumes natural-exp gates. For an exp2 DUT, multiply
    # its raw gate by ln2 here and scale only the exported cumsum back.
    g_natural = inputs["g"].float()
    if config["use_exp2"]:
        g_natural = g_natural * math.log(2.0)
    g_natural = g_natural.to(device).contiguous()
    beta = effective["beta"].to(device).contiguous()
    initial = inputs["initial_state"]
    if initial is not None:
        initial = initial.to(device).contiguous()
        if config["state_v_first"]:
            initial = initial.transpose(-1, -2).contiguous()
    cu_list = None if cu is None else list(cu)
    timings = {}

    def stage(name, invoke):
        logger.info("chain stage %s started", name)
        started = time.perf_counter()
        result = invoke()
        torch.npu.synchronize()
        timings[name] = time.perf_counter() - started
        logger.info("chain stage %s finished seconds=%.6f", name, timings[name])
        return result

    g_head_first = stage("chunk_local_cumsum", lambda: ascendc.chunk_local_cumsum(
        g_natural.transpose(1, 2), chunk_size=64, cu_seqlens=cu_list,
        chunk_indices_out=chunk_indices, head_first=True, output_dtype="float32"))
    beta_head_first = beta.transpose(1, 2).contiguous().float()
    a_raw = stage("chunk_scaled_dot_kkt", lambda: ascendc.chunk_scaled_dot_kkt(
        k=k, g=g_head_first, beta=beta_head_first, cu_seqlens=cu_list,
        chunk_indices=chunk_indices, chunk_size=64))

    def solve():
        if cu_list is None:
            return ascendc.solve_tri(a_raw.to(q.dtype), layout="bhtd")
        token_first = a_raw.transpose(1, 2).contiguous().squeeze(0)
        token_first = ascendc.solve_tri(token_first.to(q.dtype), cu_seqlens=cu_list,
                                        chunk_indices=chunk_indices, layout="tnd")
        return token_first.unsqueeze(0).transpose(1, 2).contiguous()

    a = stage("solve_tri", solve)
    w, u = stage("recompute_w_u_fwd", lambda: ascendc.recompute_w_u_fwd(
        k, v, beta_head_first, a, 64, g=g_head_first, gk=None,
        cu_seqlens=cu_list, chunk_indices=chunk_indices))
    h, v_new, final_state = stage("chunk_gated_delta_rule_fwd_h", lambda: ascendc.chunk_gated_delta_rule_fwd_h(
        k, w, u, g=g_head_first, gk=None, initial_state=initial,
        output_final_state=config["final_state"], chunk_size=64,
        cu_seqlens=cu_list, chunk_indices=chunk_indices,
        state_v_first=config["state_v_first"]))
    o = stage("chunk_fwd_o", lambda: ascendc.chunk_fwd_o(
        q, k, v_new, h, config["scale"], g=g_head_first, g_gamma=None,
        cu_seqlens=cu_list, chunk_indices=chunk_indices, chunk_size=64,
        transpose_state_layout=config["state_v_first"]))
    g_export = g_head_first.transpose(1, 2).contiguous()
    if config["use_exp2"]:
        g_export = g_export / math.log(2.0)
    result = {"o": o.transpose(1, 2).contiguous(), "g_cumsum": g_export,
              "A": a, "h": h}
    if config["final_state"]:
        result["final_state"] = final_state
    result["_timings"] = timings
    return result
