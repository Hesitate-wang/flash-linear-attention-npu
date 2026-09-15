# Development validation

## Reference traceability

| Area | Reference role | Locally owned behavior |
| --- | --- | --- |
| H API and state layout | Existing state-update behavior | Input order, shape and dtype checks, optional final state, aclnn transposes |
| H tiling | Existing state-update behavior | Logical shape, gate modes, workspace and block dimension implemented locally |
| O API and layout | `chunk_fwd_o/op_host` | `q`, scale, `use_exp2`, output layouts and output validation |
| O tiling | Existing output behavior | Shape validation, workspace and architecture-specific path implemented locally |
| Fused tiling ABI | Fusion requirement | One operator-owned tiling type and disjoint workspace offsets |

## Checks completed

- Repository-local whitespace and required-file checks: passed.
- OpDef, L0, aclnn, and tiling parameter order inspected for consistency.
- Public output order is consistently `o`, optional `final_state`; no `h` or
  `v_new` output descriptor/allocation remains in the fused host or adapter.
- Cross-operator source include scan: passed; no sibling operator header,
  processor, internal directory, CMake dependency, or private include remains.
- Host/kernel tiling mirror check: all 35 fields have identical order.
- No runnable build was attempted because the fused device entry is not yet
  present; the operator cannot execute until kernel development is complete.

## Pending

- Host compilation in a configured CANN environment.
- Kernel tiling parser and entry implementation.
- CPU reference, minimum precision case, fixed/varlen regression, and formal ATK
  coverage.
