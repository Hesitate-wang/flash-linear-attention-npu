# Host implementation staging

This directory contains the `ChunkFwdHOFused` OpDef, combined tiling, and API
implementation.

The host implementation owns a single fused tiling type, validates the current
fixed-length Atlas A2 path, chooses `2 * B * HV` active mixed cores, and
allocates full H/v_new handoff storage plus disjoint H/O scratch regions. It
does not include or link private implementation files from another operator.
