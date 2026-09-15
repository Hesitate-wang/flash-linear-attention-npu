# Host implementation staging

This directory contains the `ChunkFwdHOFused` OpDef, combined tiling, and API
implementation.

The host implementation owns a single fused tiling type, validates both phases,
and allocates disjoint H-stage and O-stage workspace regions. It does not include
or link private implementation files from another operator.
