# PTA test

`test_fwd_h_o_fused.py` uses the direct, decoupled Python interface:

```python
from fla_npu.ops import ascendc
```

The default case runs the fused operator and compares `o` with an independent
FP32 CPU reference. It uses three chunks so that producer/consumer event reuse
is exercised.

```bash
python test_fwd_h_o_fused.py
```

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

Useful shape and dtype switches include `--tokens`, `--chunk-size`,
`--value-dim`, `--dtype`, `--gate-dtype`, `--batch`, `--k-heads`, and
`--v-heads`. The fused implementation currently accepts fixed-length Atlas A2
exp mode only, with `K=128`, `V=128/256`, and chunk size `64/128`. The target
must also satisfy `2 * batch * v_heads < physical AIC core count`.
