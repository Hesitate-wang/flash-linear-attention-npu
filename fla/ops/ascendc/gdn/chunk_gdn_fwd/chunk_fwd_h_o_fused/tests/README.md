# Tests

The PTA test in `pta/test_fwd_h_o_fused.py` covers the supported fixed-length
fused path against a CPU reference. Its `--compare-composed` switch additionally
compares the fused result with the public H-then-O operator composition.

See `pta/README.md` for commands and supported parameters.
