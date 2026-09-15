# Documentation staging

This directory contains:

- `api.md`: the single source of truth for inputs, outputs, attributes, shapes,
  dtypes, layouts, optional values, error behavior, and supported hardware.
- `design.md`: the reviewed fused-stage design and resource allocation.
- `validation.md`: temporary development-time experiments and results; remove it
  after final conclusions have been moved to the operator and ATK READMEs.

The public interface exposes only `o` and the optional `final_state`; `h` and
`v_new` remain internal values passed directly between the two phases.
