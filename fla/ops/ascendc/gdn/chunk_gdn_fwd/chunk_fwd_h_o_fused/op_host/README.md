# Host implementation staging

This directory contains the `ChunkFwdHOFused` OpDef, combined tiling, and API
implementation.

The host implementation owns a single fused tiling type, validates the current
fixed-length Atlas A2/A5 paths, chooses `min(B * HV, physical AIC cores)` active mixed cores, assigns half of them as double-buffered H producers, and
allocates full H/v_new handoff storage plus disjoint H/O scratch regions. It
does not include or link private implementation files from another operator.
