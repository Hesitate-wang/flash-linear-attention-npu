# Arch35 Chunk64 Sparse GDN Forward Validation

## Current implementation

The Producer now uses full 64x64 BF16 Neumann GEMMs with FP32 accumulation, two Producer slots, AIC direct W/U GM stores, and AIC mode-0 synchronization. The former dual-32-row inverse, GM inverse exchange and GM Channel Split are not part of the active implementation.

The development host is macOS without `ccec`, `ascendc++`, an Ascend toolkit, or a visible NPU. Ascend compilation, device correctness, synchronization stress and the 30 us Producer target therefore remain unverified here.

## Checks completed after the rewrite

| Check | Method | Result |
|---|---|---|
| Host POD and pipeline helpers | C++17 `-Wall -Wextra -Werror`: `sparse_tiling_contract.cpp` | Compiled and ran successfully |
| Public ACLNN ABI | `aclnn_abi_contract.py` | Passed; 29 public parameters |
| Internal OpDef/device argument order | `sparse_entry_contract.py` | Passed |
| Neumann doubling | Twenty random strict-lower matrices at each size 2/4/8/16/32/64 versus `inv(I-L)` | Passed; chunk64 uses five power and five S-update GEMMs |
| Source whitespace | `git diff --check` | Passed |
| Inverse workspace audit | Search active Producer/host tiling for `gmInverse`, GM Channel Split and inverse workspace uses | No active inverse exchange workspace; POD fields remain zero for ABI stability |

These checks do not validate Ascend instruction behavior or device performance.

## Resource checks

- Producer UB: 138 KiB per AIV, below the 208 KiB regbase budget.
- Producer L1: 136 KiB per MIX group, below the 512 KiB hardware budget.
- Consumer maps are unchanged.
- Two Producer slots own all live Vector/KKT/Neumann inputs. Neumann P/S are reused only within the current slot.
- Serial publishes `kProdW` only after the last chunk; pipeline uses paired AIC mode-0 calls on every iteration.

## Required Ascend 950 validation

1. Build with `bash build.sh --pkg --soc=ascend950 --vendor_name=fla_npu --ops=chunk_gated_delta_rule_fwd` and fix any CANN-version diagnostics.
2. Reproduce the current 58 us Producer baseline with the user's mode-0 performance experiment, then measure the rewritten Producer with warmup 5 and 20 samples. Record all Task Duration samples and min/median/mean/max; the target is **median <=30 us**. Do not substitute Consumer or full-pipeline time.
3. Restore normal AIC mode-0 and run the lightweight `o/final_state` checks and ATK cases 348/187. Include public A/W/U, T=64/640, a non-64-aligned tail and strongly negative gates.
4. Cover varlen empty sequences, norm on/off, exp/exp2, beta sigmoid/output coupling, stateVFirst and FP32/BF16 states.
5. Run synchronization stress and MSS. In particular check the AIC mode-0 flag, Consumer GM-ready hand-off, double-slot reuse, Neumann L0C accumulation and serial final-W publication.
6. Regress high-chain serial shapes because `ProducerRole` is shared by serial and pipeline modes.
7. Measure T=16384 after correctness and synchronization pass. The full-pipeline result is additional evidence, not a replacement for the Producer 30 us gate.

## Performance tracking

| Case | Baseline | Candidate | Accuracy | Performance | Status |
|---|---|---|---|---|---|
| User's frozen single-chain chunk64 workload | Producer 58 us; Consumer 90 us; chunk32 full fusion 40 us | Full 64 Neumann + two-slot Producer + AIC direct W/U | Device validation pending | Producer target 30 us; no device result recorded | Implementation complete, Ascend build/measurement pending |

Earlier validation entries for the 32-row inverse and GM Channel Split describe superseded implementations and are not evidence for this version.
