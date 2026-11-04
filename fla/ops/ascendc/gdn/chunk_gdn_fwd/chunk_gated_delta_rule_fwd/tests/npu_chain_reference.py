#!/usr/bin/env python3
"""Optional Prepare -> FwdH -> FwdO NPU reference chain."""

from __future__ import annotations

import time

import torch


def run_npu_chain(ascendc, inputs: dict, config: dict,
                  cu: tuple[int, ...] | None, chunk_indices: list[int] | None,
                  logger) -> dict:
    """Run the three public operators and return outputs shared with the DUT."""
    device = f"npu:{config['device']}"
    q = inputs["q"].to(device).contiguous()
    k = inputs["k"].to(device).contiguous()
    v = inputs["v"].to(device).contiguous()
    # Prepare uses BHT for gate/beta; the fused public wrapper uses BTH.
    g = inputs["g"].transpose(1, 2).to(device).contiguous()
    beta = inputs["beta"].float().transpose(1, 2).to(device).contiguous()
    logger.info("chain input layouts q=%s/%s k=%s/%s v=%s/%s g=%s/%s beta=%s/%s",
                tuple(q.shape), q.dtype, tuple(k.shape), k.dtype, tuple(v.shape), v.dtype,
                tuple(g.shape), g.dtype, tuple(beta.shape), beta.dtype)
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

    prepared = stage("chunk_gated_delta_rule_fwd_prepare", lambda: ascendc.chunk_gated_delta_rule_fwd_prepare(
        q, k, v, g, beta, chunk_size=64,
        use_qk_l2norm_in_kernel=config["l2norm"], use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=config["beta_sigmoid"],
        allow_neg_eigval=config["allow_neg_eigval"], use_exp2=config["use_exp2"],
        cu_seqlens=cu_list, chunk_indices=chunk_indices, output_a=True))
    q_hat, k_hat, q_rstd, k_rstd, beta_eff, g_cumsum, w, u, a = prepared

    h, v_new, final_state = stage("chunk_gated_delta_rule_fwd_h", lambda: ascendc.chunk_gated_delta_rule_fwd_h(
        k_hat, w, u,
        g=None if config["use_exp2"] else g_cumsum,
        gk=g_cumsum if config["use_exp2"] else None,
        initial_state=initial, output_final_state=config["final_state"], chunk_size=64,
        cu_seqlens=cu_list, chunk_indices=chunk_indices,
        state_v_first=config["state_v_first"]))
    o = stage("chunk_fwd_o", lambda: ascendc.chunk_fwd_o(
        q_hat, k_hat, v_new, h, config["scale"], g=g_cumsum, g_gamma=None,
        cu_seqlens=cu_list, chunk_indices=chunk_indices, chunk_size=64,
        transpose_state_layout=config["state_v_first"], use_exp2=config["use_exp2"]))

    result = {"o": o.transpose(1, 2).contiguous(),
              "g_cumsum": g_cumsum.transpose(1, 2).contiguous(),
              "A": a, "h": h}
    if config["final_state"]:
        result["final_state"] = final_state
    if config["beta_sigmoid"]:
        result["beta_eff"] = beta_eff.transpose(1, 2).contiguous()
    if config["l2norm"]:
        result.update({"q_hat": q_hat, "k_hat": k_hat,
                       "q_rstd": q_rstd, "k_rstd": k_rstd})
    result["_timings"] = timings
    return result
