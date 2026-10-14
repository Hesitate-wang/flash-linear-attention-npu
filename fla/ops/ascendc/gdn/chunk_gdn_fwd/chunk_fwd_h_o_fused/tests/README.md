# Tests

The PTA test in `pta/test_fwd_h_o_fused.py` covers the supported fixed-length
fused path against a CPU reference. Its `--compare-composed` switch additionally
compares the fused result with the public H-then-O operator composition.

See `pta/README.md` for commands and supported parameters.

## Offline MODE2 Event/UB Regression

Run from the repository root, without CANN or an NPU:

```bash
python3 -m unittest discover -s tests -p test_chunk_fwd_ho_mode2_sync.py -v
```

This source-driven logical model checks local event ownership, persistent H
tokens, initialization, multi-chunk/final-state branches, empty token lanes,
32-row O ping/pong segments, UB bounds and stage handoff order. It rejects
duplicate Set and Wait without a producer. It does not simulate hardware
pipeline execution or cross-core timing, and is not device deadlock/precision
acceptance. Device validation requirements are recorded in `../docs/validation.md`.
