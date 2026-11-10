# Arch35 Chunk64 Sparse GDN Forward Validation

## Current implementation

Producer computes KKT, packed 32x64 VCS and native-FP32 lower-left coupling, then publishes only the complete inverse T. It no longer computes Kbg/Vb or W/U. Consumer preloads negative Kbg and Vb into L1, then computes W/U and `U-W@S` directly in L0C, followed by O and D.

The internal A/T buffer is always allocated at full `[B,Hv,T,64]` size, including when `aOutOptional` is null. It is not a public output. Public A/g_cumsum/KHat/KRstd/H/qHat/qRstd are not written; only the internal T, final O and final_state are used.

The development host is macOS without `ccec`, `ascendc++`, an Ascend toolkit, or a visible NPU. Device compilation, correctness and the 30 us Producer target remain unverified here.

## Checks completed

| Check | Method | Result |
|---|---|---|
| Host POD and pipeline helpers | C++17 `-Wall -Wextra -Werror`: `sparse_tiling_contract.cpp` | Compiled and ran successfully |
| Public ACLNN ABI | `aclnn_abi_contract.py` | Passed; 29 public parameters |
| Internal OpDef/device argument order | `sparse_entry_contract.py` | Passed |
| Packed VCS and lower-left coupling | Independent left/right substitution and `X1@L10@X0` versus the complete inverse | Passed |
| Negative-Kbg identity | `U + (-trueW)@S` equals `U - trueW@S` | Same L0C accumulation semantics |
| Source whitespace | `git diff --check` | Passed |
| Test scripts | `py_compile` and lightweight `--cpu-only` | Passed |

These checks do not validate Ascend instruction behavior or device performance.

## Resource and lifetime checks

- Ascend 950 L1 capacity confirmed from the platform knowledge cards: **512 KiB per AI Core**.
- Consumer UB: 208 KiB per AIV, within the 208 KiB regbase budget.
- Consumer L1: 192 KiB per MIX group, within the 512 KiB hardware budget.
- Kbg and Vb are prepared in L1 while Producer computes the inverse.
- U remains in L0C while W@S accumulates, producing vnew without a separate UB subtraction.
- QExp@S and A'@vnew accumulate in L0C before the final output Fixpipe.
- Serial publishes only the final T. Pipeline uses paired AIC mode-0 calls on every iteration.

## Required Ascend 950 validation

1. Build with `bash build.sh --pkg --soc=ascend950 --vendor_name=fla_npu --ops=chunk_gated_delta_rule_fwd` and fix any CANN-version diagnostics.
2. Measure Producer and Consumer separately with the original workload, warmup 5 and 20 samples. Record all Task Duration samples and min/median/mean/max. The Producer target is **median <=30 us**, but Consumer and full-pipeline time must also be reported.
3. Restore normal AIC mode-0 and run lightweight `o/final_state` and ATK cases 348/187. Include T=64/640, a non-64-aligned tail and strongly negative gates.
4. Cover varlen empty sequences, norm on/off, exp/exp2, beta sigmoid, stateVFirst and FP32/BF16 states.
5. Run synchronization stress and MSS, focusing on T readiness, negative Kbg/Vb L1 lifetime, U-W@S accumulation, vnew/right hand-off and serial final-T publication.
6. Regress high-chain serial shapes and T=16384. Full-pipeline results do not replace the standalone Producer target.
