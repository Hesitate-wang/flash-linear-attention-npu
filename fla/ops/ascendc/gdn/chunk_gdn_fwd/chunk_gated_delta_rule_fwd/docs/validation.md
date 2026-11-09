# Arch35 Chunk64 Sparse GDN Forward Validation

## Current implementation

The active Producer uses one parity owner for packed 32x64 VCS, alternating AIV preparation, two native-FP32 coupling GEMMs, and AIC direct W/U GM stores. Gate/beta are recomputed locally by each AIV; there is no L1-to-UB or GM gate-sharing path.

The former full 64x64 Neumann implementation is no longer active. The previous Neumann profile (`aic_time=72.62 us`, `aiv_time=68.91 us`) is retained only as evidence that this approach was slower, not as a result for the packed VCS implementation.

The development host is macOS without `ccec`, `ascendc++`, an Ascend toolkit, or a visible NPU. Device compilation, correctness and the 30 us Producer target remain unverified here.

## Checks completed

| Check | Method | Result |
|---|---|---|
| Host POD and pipeline helpers | C++17 `-Wall -Wextra -Werror`: `sparse_tiling_contract.cpp` | Compiled and ran successfully |
| Public ACLNN ABI | `aclnn_abi_contract.py` | Passed; 29 public parameters |
| Internal OpDef/device argument order | `sparse_entry_contract.py` | Passed |
| Packed VCS | Independent left/right forward substitution for 4/8/16/32-row diagonal blocks | Passed against full `inv(I-L)` |
| Lower-left coupling | `X1@L10@X0` versus the complete inverse | Passed |
| UpdateMask audit | All new mask counts use independent writable `uint32_t` variables | No const/rvalue `UpdateMask` call in the new VCS |
| Source whitespace | `git diff --check` | Passed |

These checks do not validate Ascend instruction behavior or device performance.

## Resource and lifetime checks

- Producer UB: 150 KiB per AIV, below the 208 KiB budget.
- Producer L1: 120 KiB per MIX group, below the 512 KiB budget.
- The parity owner receives one complete KKT; the other AIV does not wait for it.
- X0/X1/L10 are published before coupling; diagonal A blocks are published before W/U.
- Two K/Kbg/Vb slots are drained before reuse. Public intermediate stores are removed; only W/U and O/final_state are written.
- Serial publishes `kProdW` only after the final chunk. Pipeline uses paired AIC mode-0 calls on every iteration.

## Required Ascend 950 validation

1. Build with `bash build.sh --pkg --soc=ascend950 --vendor_name=fla_npu --ops=chunk_gated_delta_rule_fwd` and fix any CANN-version diagnostics.
2. Reproduce the original 58 us Producer baseline and compare packed VCS with the rejected Neumann profile. Use the same workload, warmup 5 and 20 samples, and record all Task Duration samples and min/median/mean/max. The target is **Producer median <=30 us**.
3. Restore normal AIC mode-0 and run lightweight `o/final_state` and ATK cases 348/187. Include W/U, T=64/640, a non-64-aligned tail and strongly negative gates. Public A/g_cumsum/KHat/KRstd/H/qHat/qRstd are intentionally not written.
4. Cover varlen empty sequences, norm on/off, exp/exp2, beta sigmoid/output coupling, stateVFirst and FP32/BF16 states.
5. Run synchronization stress and MSS, focusing on parity-owner KKT delivery, inverse/A hand-offs, gate recomputation, two-slot reuse and serial final-W publication.
6. Regress high-chain serial shapes and T=16384. Full-pipeline results do not replace the standalone Producer target.
