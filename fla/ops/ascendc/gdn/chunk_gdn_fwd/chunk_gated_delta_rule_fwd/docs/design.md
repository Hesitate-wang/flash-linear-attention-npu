# Arch35 Chunk64 Sparse GDN Forward Design

## 1. Scope

This implementation targets Ascend 950 / arch35, BF16 Q/K/V, FP32 gate/beta, `K=V=128`, and low parallelism `B*Hv <= AIC/2`. The public ACLNN ABI and ten-element Python result are unchanged.

The current Producer replaces the former dual-32-row triangular solve, GM inverse exchange, and native-FP32 Channel Split path with full 64x64 Neumann GEMMs and a two-slot Producer pipeline. Its standalone performance target is 30 us on the frozen user workload; that target is not claimed until the Ascend 950 measurement is recorded.

## 2. Producer dataflow

Each chain has two Producer slots. Slot `c&1` owns the K/V/gate/beta inputs, normalized/scaled matrices, KKT result, and Neumann input matrices for chunk `c`.

For each chunk:

1. AIV loads and prepares K, gate and beta, then publishes K early for KKT. It also prepares Vb and Kbg and publishes the complete input slot.
2. AIC computes KKT and publishes the two 32-row FP32 bands to the AIVs.
3. AIV converts its KKT/gate/beta band into two full-chunk BF16 inputs: strict-lower `P=L` and `S=I+L`. It then immediately prepares the next chunk in the other slot.
4. AIC evaluates the inverse and W/U for the current chunk while the other AIV slot is being prepared.

The overlap is between current-chunk AIC Cube work and next-chunk Vector preparation. KKT and Neumann GEMMs still execute serially on the single AIC; the design does not assume two chunks can use that AIC simultaneously.

## 3. Full 64x64 Neumann inverse

For unit-lower-triangular `M=I-L`, with strict-lower and nilpotent `L`:

```text
S0 = I + L
P0 = L
for k = 0..4:
    P(k+1) = P(k) @ P(k)
    S(k+1) = S(k) + P(k+1) @ S(k)
T = S5 = (I-L)^-1 = I + L + ... + L^63
```

There are five power-squaring GEMMs and five S-update GEMMs. S0 already contains the first two Neumann terms. One initial `S0@I` GEMM materializes S0 in the L0C accumulator before those updates. All matrices are full 64x64; no 32x32 inverse blocks or Vector forward substitution remain.

GEMM operands are BF16 with FP32 accumulation. P is written back to L1 after each squaring. S remains as an FP32 L0C accumulator and is also written to L1 as the next GEMM's BF16 operand. The S update uses `initC=false`; the preceding Fixpipe must retire before L0C is overwritten.

After the fifth update, T is used directly for public A when requested and for `W=T@Kb` and `U=T@Vb`. No inverse workspace or GM intermediate matrix is used.

## 4. W/U and cross-group synchronization

AIC writes W/U directly from L0C to GM with BF16 ND Fixpipe, including only valid rows for tails. The former `FixB16Halves` plus AIV `DrainWu` path is removed.

Pipeline mode uses AIC mode-0 flag 7:

- Producer AIC arrives after the W/U GM Fixpipe, including empty iterations for shorter chains.
- Consumer AIC arrives after its QK phase and the same number of iterations.
- After the barrier, Consumer AIC publishes `kConsWuGmReady` through mode 4. Consumer AIV waits this flag before loading W/U; W's L1-ready flag remains `kConsW`.

AIV no longer executes a second mode-0 barrier. Other mode-4 handshakes and the kernel-exit drains remain.

Serial mode does not use AIC mode-0. AIC publishes `kProdW` only after the final chunk, and AIV waits it before the existing mode-1 `PublishChainInGroup`. This preserves the whole-chain hand-off without allowing the mode-4 counter to accumulate once per chunk.

## 5. Memory layout

Producer UB is per AIV:

- Two 68 KiB slots containing K, V, normalized/public K, Vb, Kbg, KKT, P, S and gate/norm temporaries.
- One shared 2 KiB scan-index region.
- Total: 138 KiB, below the 208 KiB regbase budget.

Producer L1 is per MIX group:

- Each 64 KiB slot contains K, Vb, Kbg, P and S.
- Two slots plus one shared 8 KiB identity matrix: 136 KiB, below the 512 KiB hardware budget.

Neumann uses the current slot's P/S throughout. L0C banks separately hold S, P, W and U/KKT as required. Consumer memory maps are unchanged.

## 6. Output and workspace

Public KHat/g_cumsum/beta_eff retain their existing layouts. Double slots are drained before reuse so asynchronous public writes cannot overwrite a live UB source.

Public A is written directly from the final Neumann accumulator when `outputA` is enabled. W/U use the existing BHTD addresses. `disable_recompute=true` therefore retains the A output cost in Producer measurements.

The former X0/X1/L10/T workspace is removed. `workspaceOffset/workspaceBytes` remain in the shared tiling POD for ABI stability and are zero; only the platform library workspace is requested.
