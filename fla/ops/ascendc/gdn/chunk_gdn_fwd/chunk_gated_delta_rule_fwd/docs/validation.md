# Arch35 Chunk64 Sparse GDN Forward Validation

## Current implementation

Producer AIV0/AIV1 own complete even/odd 64-row chunks. Each round preprocesses the owner's next chunk, issues independent MTE2 loads for the following round, performs one complete FP32 64x64 Vector inverse and executes QKMask. Producer AIC executes KKT and QKT for the even/odd pair. There are no 32x32 diagonal solves, Cube coupling operations or serial scheduling paths.

For `useNorm=1`, Producer also publishes BF16 NZ QHat/KHat to independent per-chunk GM records. Consumer waits once per even/odd pair, then preloads the normalized inputs instead of repeating L2Norm. It still computes QExp/Kbg locally from gate energy and beta. The no-norm path remains unchanged.

Chunk32 is a Producer-only benchmark and instantiates the same `ProducerRole<ChunkSize,BetaT>` as chunk64. Shared vector and pipeline code is parameterized only by row count/stride; BF16 beta is converted to the same FP32 gate representation. Consumer groups are idle, so O/final_state are not correctness outputs in this mode.

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
| Shared chunk32/64 benchmark algorithm | Twenty random inverse checks at each size, 8-row groups; chunk32 workspace offsets | Passed; no alternate chunk32 high-level inverse implementation is used |
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
7. For the chunk32 benchmark, use the same B/H/K/V/gate/beta configuration as chunk64 and compare both total Producer time and time per chunk. Keep Consumer groups idle and do not interpret public O/final_state as valid outputs.
