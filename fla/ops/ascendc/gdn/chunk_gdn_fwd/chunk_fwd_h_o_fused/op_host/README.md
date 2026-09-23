# Host implementation staging

This directory contains the `ChunkFwdHOFused` OpDef, combined tiling, and API
implementation.

The host implementation owns a single fused tiling type, validates the current
fixed-length Atlas A2/A5 paths, chooses paired H/O mixed cores with
`taskNum=B * HV` after checking that two cores per task fit, and
allocates full H/v_new handoff storage plus disjoint H/O scratch regions. It
does not include or link private implementation files from another operator.
