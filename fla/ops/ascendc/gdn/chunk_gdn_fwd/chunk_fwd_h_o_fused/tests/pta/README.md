# PTA test

`test_fwd_h_o_fused.py` uses the direct, decoupled Python interface by default:

```python
from fla_npu.ops import ascendc
```

The default case runs the fused operator and compares `o` with an independent
FP32 CPU reference. It uses three chunks so that producer/consumer event reuse
is exercised.

```bash
python test_fwd_h_o_fused.py
```

To validate the optional legacy PTA dispatcher registration, first build the
wheel with `FLA_NPU_BUILD_LEGACY_EXTENSION=1`, then run:

```bash
python test_fwd_h_o_fused.py --runtime legacy
```

This path loads the extension with `fla_npu.load_legacy_torch_ops()` and calls
`torch.ops.npu.npu_chunk_fwd_h_o_fused`. The default path remains decoupled
from the PyTorch/torch_npu C++ ABI.

Add `--compare-composed` to also run
`ascendc.chunk_gated_delta_rule_fwd_h` followed by
`ascendc.chunk_fwd_o`, then compare that result with the fused result:

```bash
python test_fwd_h_o_fused.py --compare-composed
```

Final-state and initial-state coverage can be enabled independently:

```bash
python test_fwd_h_o_fused.py \
  --compare-composed \
  --initial-state \
  --output-final-state
```

On Ascend 950, enable the exp2/BSND path and optionally compare it with the
standalone arch35 H then O composition:

```bash
python test_fwd_h_o_fused.py --use-exp2 --compare-composed
```

The Ascend 950 natural-exp path now covers the same functional matrix as A2.
These commands exercise the added FP16, V=256, chunk=128, and NTD branches:

```bash
python test_fwd_h_o_fused.py --dtype float16 --compare-composed
python test_fwd_h_o_fused.py --value-dim 256 --compare-composed
python test_fwd_h_o_fused.py --tokens 384 --chunk-size 128 --compare-composed
python test_fwd_h_o_fused.py --output-layout NTD --compare-composed
```

Useful shape and dtype switches include `--tokens`, `--chunk-size`,
`--value-dim`, `--dtype`, `--gate-dtype`, `--batch`, `--k-heads`, and
`--v-heads`. Atlas A2 accepts exp mode with `K=128`, `V=128/256`, chunk size
`64/128`, and requires `2 * batch * v_heads < physical AIC core count`.
Ascend 950 natural-exp mode accepts the same matrix. Ascend 950 `--use-exp2`
still requires BF16 data, BF16/FP32 gates, `K=V=128`, chunk size 64, and
`v_heads/k_heads` in `[1,4]`. Use `--output-layout` to select the layout allowed
by each exponent mode.
