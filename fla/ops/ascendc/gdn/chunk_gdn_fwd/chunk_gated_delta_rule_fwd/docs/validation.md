# Arch35 Chunk64 Sparse GDN Forward Validation

## Current implementation

Producer AIV0/AIV1 own complete even/odd 64-row chunks. Each round preprocesses the owner's next chunk, issues independent MTE2 loads for the following round, performs one complete FP32 64x64 Vector inverse and executes QKMask. Producer AIC executes KKT and QKT for the even/odd pair. There are no 32x32 diagonal solves, Cube coupling operations or serial scheduling paths.

Producer always performs Q/K L2Norm and publishes BF16 NZ QHat/KHat to independent per-chunk GM records. Consumer waits once per even/odd pair, then preloads the normalized inputs instead of repeating L2Norm. It still computes QExp/Kbg locally from gate energy and beta.

A persistent 64x64 identity matrix is built once during Producer initialization and loaded by every inverse group. Preprocess and inverse use opposite local-pair buffer slots. The Producer UB layout occupies 216 KiB per AIV within a 248 KiB budget; Producer L1 occupies 96 KiB. Consumer prepares negative Kbg/Vb, recomputes W/U, fuses `U-W@S`, then computes O and the recurrent state update.

The internal A/T buffer remains allocated at full `[B,Hv,T,64]` size, including when `aOutOptional` is null. Only valid T rows are written for tails; APrime uses a complete 64x64 NZ workspace tile. Normalized Q/K use independent BF16 NZ records sized as `totalChunks * Hk * 64 * 128` elements per tensor.

## Checks completed on the development host

| Check | Method | Result |
|---|---|---|
| Producer UB/L1 offsets and capacity | C++17 `-Wall -Wextra -Werror` static assertions | Passed; 216 KiB UB, 96 KiB L1, 248 KiB UB budget |
| Host/device tiling POD alignment | Same C++ static contract | Passed |
| Undeclared-identifier static audit | All Producer member references, 75 Producer/L1/UB constants, new local call targets and deleted-symbol references | No missing declaration or deleted-symbol use found; not a substitute for device compilation |
| Pipeline host plan | C++ dense/varlen plans, odd tails, excessive chains and empty input | Passed; eligible plans formed two groups per chain and ineligible plans returned false |
| Complete grouped 64x64 forward substitution | 100 random strict-lower matrices versus `numpy.linalg.inv(I-L)` | Passed; maximum absolute error `1.388e-16` in FP64 reference evaluation |
| Gate-ratio inverse and QKMask formulas | 20 random cases each for exp/exp2; inverse and inclusive causal mask references | Passed |
| Parity buffer slot schedule | Both owners, chunk counts 1 through 64 | Current/preprocess ratio slots and current/preprocess/load gate slots passed |
| Odd, short and empty chain schedule | Every chain length for each `maxChunks` from 1 through 64 | Each valid chunk executed once; global generations exactly covered `0..maxChunks-1` |
| Slot-decoupled MTE3 events | Both owners, chunk counts 1 through 64, plus explicit chunk4/inverse2 boundary | No consecutive Set without Wait, no Wait without Set, no unconsumed tail event; opposite-slot preprocess remained in flight |
| Normalized Q/K pair protocol | Every chain length for `maxChunks` 1 through 32 | Four AIV participants per pair, including short/empty chains; four eventual Waits and no unmatched flag reuse |
| BF16 NZ exchange layout | Both 32-row half bands, QExp/Kbg scaling reconstruction, Q/K workspace offsets | Passed |
| Even/odd normalized prefetch | Every chain length from 1 through 64 | Both chunks issued at the even step with distinct MTE2 events; per-slot K release and tail waits fully paired |
| Lightweight test scripts | Python 3.12 `py_compile` | Passed |
| CPU-only lightweight cases | `dense-tail`, `varlen`, `varlen-empty` | CPU golden generation and logging completed; no device DUT was run |
| Comparison helper | `compare_contract.py` | Passed |
| Source whitespace | `git diff --check` | Passed |

These checks validate the host layout, reference algorithm and existing CPU utilities. They do not validate Ascend instruction behavior, generated device code, cross-core synchronization, numerical output or performance.

## Environment limitation

The development host is macOS without `ccec`, `ascendc++`, an Ascend toolkit or a visible NPU. Device compilation, correctness and the Producer performance target remain unverified here.

## Required Ascend 950 validation

1. Build with `bash build.sh --pkg --soc=ascend950 --vendor_name=fla_npu --ops=chunk_gated_delta_rule_fwd` and fix any CANN-version diagnostics.
2. Run lightweight `o/final_state` comparisons for dense T=64/65/128/129/640 and varlen with empty sequences. Include an odd `maxChunks`, a single-chunk chain and chains of different lengths in one launch.
3. Cover norm on/off, exp/exp2, beta sigmoid, strongly negative gates, stateVFirst and FP32/BF16 initial states.
4. Run synchronization stress and MSS, focusing on owner-specific mode-4 flag routing, single KKT/QK UB backpressure, complete T readiness, APrime completion, preprocess/inverse UB aliases, three-slot gate/L1 lifetimes and odd-pair generation counts.
5. Verify that key-64 tiling launches exactly `2*chainCount` groups and rejects shapes that cannot form the pipeline; no serial fallback should remain.
6. Measure Producer and Consumer separately with warmup 5 and 20 samples. Record min/median/mean/max and compare against the previous implementation without treating host-side checks as performance evidence.

## 2026-11-23 bare-run O precision localization

- Fixed failure: `dense-tail`, `B=1`, `Hk=Hv=16`, `T=8192`, chunk 64, BF16 Q/K/V/O, FP32 initial/final state, seed 19, device 6, no warmup, one iteration, NPU-chain comparison with the existing `0.001/0.0001/0.1` tolerance. The bare run failed O with 218,998 mismatches and maximum mixed error 0.306; final_state passed. Historical passes under msprof are not bare-run evidence.
- Saved DUT/chain outputs showed that all errors above `1e-3` occurred in odd chunks 1, 3 and 125. Reconstructing the attention coefficient error with the reference chain's `v_new/q_hat/k_hat/g_cumsum` showed that chunk 1 consumed chunk 3's APrime, chunk 3 consumed chunk 5's, and chunk 125 consumed chunk 127's; representative per-head fits were within approximately 1% to 5%.
- Root cause: `l1APrime_[2]` had no local MTE1-to-MTE2 release. A later pair's GM-to-L1 copy could overwrite a parity slot before the previous pair's `LoadA` consumed it. T already had the corresponding protection.
- Fix design: APrime `LoadA` publishes MTE1_MTE2 IDs 2/3 by parity; each next-pair GM-to-L1 copy waits only for its matching slot release; the AIC tail drains the final one or two releases. The mode-0 GM readiness protocol is unchanged.
- Build/static status: a single-operator build/install completed before the user took over all subsequent build and device execution. Its `libcust_opapi.so` MD5 matched the installed library; fresh kernel-object comparison was skipped because the expected build kernel directory was absent. The source was subsequently narrowed to per-slot waits, so a new build is required. `git diff --check` and issue-corpus coverage passed; generic static review reports pre-existing host DFX-MESSAGE findings outside this change.
- Required regression: rebuild and install the current source, verify the installed artifact, repeat the identical bare case, then cover Hv=12, short odd/even chains and varlen tails. No post-fix device result is available; the user performs all further build and device execution.
