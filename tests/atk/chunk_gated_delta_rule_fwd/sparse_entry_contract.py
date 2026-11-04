#!/usr/bin/env python3
"""Check the private fused schema and both device entry argument orders."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OP = ROOT / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd"
INPUTS = ["q", "k", "v", "beta", "a_storage", "raw_g", "gk", "initial_state",
          "cu_seqlens", "chunk_indices", "a_log", "dt_bias"]
OUTPUTS = ["o", "final_state", "g_cumsum_bth", "A", "q_hat", "k_hat", "q_rstd",
           "k_rstd", "beta_eff", "h", "w", "u", "v_new", "g_cumsum_bht"]
ATTRS = ["output_final_state", "chunk_size", "scale", "output_g_cumsum", "raw_g_layout",
         "qkv_layout", "o_layout", "state_v_first", "allow_neg_eigval", "fused_prepare",
         "use_qk_l2norm", "use_exp2", "use_beta_sigmoid", "output_a"]


def main():
    definition = (OP / "op_host/chunk_gated_delta_rule_fwd_def.cpp").read_text()
    for kind, expected in [("Input", INPUTS), ("Output", OUTPUTS), ("Attr", ATTRS)]:
        actual = re.findall(rf'this->{kind}\("([^"]+)"\)', definition)
        assert actual == expected, (kind, actual, expected)
    for arch in ("arch35", "arch22"):
        source = (OP / f"op_kernel/internal/{arch}/chunk_gated_delta_rule_fwd_{arch}.cpp").read_text()
        match = re.search(r"void\s+chunk_gated_delta_rule_fwd\s*\((.*?)\)\s*\{", source, re.S)
        assert match, arch
        actual = re.findall(r"GM_ADDR\s+(\w+)", match.group(1))
        assert actual == INPUTS + OUTPUTS + ["workspace", "tiling"], (arch, actual)
    mirror = (OP / "op_kernel/internal/arch35/chunk_64_sparse/sparse_fused_tiling.h").read_text()
    assert "struct FusedTilingData" in mirror
    assert "PrepareTilingData" not in mirror
    assert "ChunkFwdHPlainTilingData" not in mirror
    sparse_tiling = (OP / "op_host/op_tiling/arch35/chunk_64_sparse/chunk_gated_delta_rule_fwd_sparse_tiling.cpp").read_text()
    assert "GetAttrPointer<double>(2)" in sparse_tiling
    assert "GetAttrPointer<float>(2)" not in sparse_tiling
    public_header = (OP / "op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.h").read_text()
    assert "fused_prepare" not in public_header
    assert "ChunkGatedDeltaRuleFwdPrepareArgs" not in public_header
    print("sparse entry schema and device ABI: passed")


if __name__ == "__main__":
    main()
