# Arch35 Chunk64 Sparse GDN Forward Design

## 1. Scope

This is a clean chunk64 implementation for Ascend 950 / arch35. It targets BF16 Q/K/V, FP32 gate/beta, `K=V=128`, and low parallelism `B*Hv <= AIC/2`. The public ACLNN ABI and ten-element Python result are unchanged. The implementation is self-contained under `chunk_gated_delta_rule_fwd`; it does not include another operator's headers or call standalone Prepare/H/O operators.

The measured low-parallelism model case is `B=1,Hv=1,T=16384,K=V=128`; its current fused-kernel `Task Duration` baseline is 2400 us. The comparison `(1,14,16384,128)` chunk32 case takes 850 us but has fourteen independent chains, so it is not an equal-work baseline. An absolute performance target and the final warm-up/sample protocol remain to be frozen with the device measurements.

## 2. Dataflow

For each `(sequence, value-head)` chain, two MIX groups execute a chunk-synchronous pre/post pipeline with the W/U rendezvous placed after consumer QK/A'.

Producer, per chunk:

1. Two AIVs first process only the inverse dependencies: K load/wait/normalize, then gate/beta load/wait/transform. K normalization writes tight NZ directly in UB; after the K NZ copy reaches L1, `kProdK` releases AIC KKT while gate/beta processing continues. Producer no longer loads or normalizes Q. Public KHat, V loading, and Vb/Kbg scaling are deferred until after `PublishInverse`, so they overlap the AIC coupling GEMMs. Public KHat retains ND and uses an independent buffer from the FP32 coupling temporary T.
2. AIC computes `KKT=K@K^T` in parallel with the remaining AIV preparation, then sends the upper/lower 32-row bands to AIV0/AIV1 respectively. Vb and Kbg are also produced directly in NZ, so their L1 path is a contiguous copy rather than an ND-to-NZ panel transform.
3. The two AIVs compute the diagonal inverses in FP32. For strictly-lower `L=-raw` and `M=I-L=[[M00,0],[-L10,M11]]`, AIV0 computes `X0=M00^-1` and AIV1 computes `X1=M11^-1` independently. AIC then evaluates `T=X1@L10` and `X10=T@X0` with two native FP32 32x32 GEMMs; all inverse arithmetic remains FP32 until the final BF16 cast.
4. AIC computes `W=A@kbg` and `U=A@vb`. U reuses A already loaded in L0A by W.
5. W/U and requested Prepare outputs are published through GM.

Consumer, on the same chunk before and after the W/U rendezvous:

1. Load Q/K and raw gate; recompute gate cumsum, energies and optional L2 normalization. Upload Q/K/QExp and the current BF16 state mirror to L1. W/U are not loaded in this phase.
2. AIC computes `QK=Q@K^T`; both AIVs form their own `A'` row band from QK and gate energies.
3. Enter the per-chunk mode-0 rendezvous. The producer has completed the current chunk's W/U, so the consumer delays its W/U dependency until after QK/A'.
4. Load W/U, upload W to L1 and publish it to AIC. U remains in UB for the vector epilogue; it is not copied to L1.
5. AIC computes `P=W@H` and `O_s=Q@H`. AIV computes `v_new=U-P` and `right=E(g_last-g)*v_new`, then publishes both to L1.
6. AIC computes `D=K^T@right` (or `right^T@K` for stateVFirst) and `O_l=A'@v_new`.
7. AIV writes `o=(E(g)*O_s+O_l)*scale`, updates the FP32 state with `H=E(g_last)*H+D`, and refreshes its BF16 cube mirror.
8. After the current chunk's H/O work, the consumer asynchronously writes public q_hat/q_rstd in ND form. No wait is inserted before QK, W/U, the mode-0 barrier, or H/O. The following chunk's Prepare waits only for the previous GM reads to retire before reusing q_hat/q_rstd buffers, and the kernel drains the final writes at exit.

Per chain, the current main Cube MAC count is approximately 1.57M in producer and 4.19M in consumer. Consumer-side QK/A' is concurrent with producer work; only W/U cross the role boundary. The inverse merge adds two logical 32x32 matrix products, implemented by the two AIVs in FP32 rather than by extra BF16 Cube operations.

## 3. Schedule and synchronization

Pipeline mode launches exactly `2*chainCount` MIX groups, so there are no idle groups in the mode-0 collective. Chain `c` uses producer group `c` and consumer group `chainCount+c`.

Iteration `i`:

- producer builds W/U for chunk `i` when it exists;
- in parallel, consumer loads Q/K/g and builds QK/A' for the same chunk `i`;
- every launched AIV executes one mode-0 barrier, including short varlen chains;
- after the barrier, the consumer loads W/U and runs H/O for chunk `i`, while the producer may start chunk `i+1`.

The mode-0 Set uses PIPE_MTE3 and Wait blocks PIPE_MTE2. This publishes producer W/U before the consumer's post-QK W/U loads. AIC does not enter mode 0; its QK phase waits for the Q/K inputs through mode 4, and its W@H phase waits for a separate W-ready mode-4 flag after the barrier.

Mode-4 flags are directional. AIV0/AIV1 use raw IDs 0..15. AIC waits IDs `id` and `id+16` for a two-AIV input and publishes to either AIV0 only or both AIVs as required.

The inverse uses per-chunk mode-4 handshakes. AIV0 writes X0; AIV1 writes X1 and L10. After the PIPE_MTE3 publication, AIC reads all three FP32 blocks from GM to L1 and computes the first coupling GEMM. T is written directly from L0C to its L1 C0=8 NZ slot through `FixpipeL0CToL1`, followed by an explicit FIX_MTE1 barrier before the second GEMM's MTE1 LoadData. AIC then computes the second GEMM and publishes X10 through FIX. No UB staging, AIV format-conversion task, peer-AIV GM read, or mode-1 inverse rendezvous is used. MTE2_MTE1 covers the GM-to-L1 input path before the native FP32 GEMMs, FIX_MTE1 covers T between the two GEMMs, and the coupled flag closes the L1 lifetime.

Above the low-parallelism bound, each group runs producer then consumer for whole chains. A mode-1 AIV rendezvous publishes the chain inside that group; no mode-0 barrier is used.

## 4. Inverse exchange workspace

Host reserves `totalChunks * Hv * 3 * 32 * 32 * sizeof(float)` bytes, 512-byte aligned, in user workspace. Its tile index is:

- dense: `(batch*chunksPerBatch + chunk)*Hv + hv`;
- varlen: `(sequenceChunkPrefix + chunk)*Hv + hv`.

Each tile contains three aligned 4 KiB FP32 blocks: X0, X1 and L10. AIC reads all 12 KiB from GM to L1 per chunk; there are no peer-AIV reads. T occupies a separate FP32 L1 slot and X10 is fixed directly into the BF16 A L1 block and, when requested, the public A output. The workspace is not an output and is not part of the public ABI.

The previous A' workspace is no longer reserved: QK/A' are recomputed in the consumer and never cross the role boundary.

## 5. On-chip maps

| Resource | Producer | Consumer |
|---|---:|---:|
| UB per AIV | 144 KiB | 200 KiB |
| L1 per MIX group | 72 KiB | 184 KiB |
| L0A/L0B | two 32 KiB banks | two 32 KiB banks |
| L0C | four 64 KiB banks | four 64 KiB banks |

Both UB maps are below the 208 KiB regbase budget. `sparse_memory.h` contains compile-time bound assertions. The inverse reuses the existing KKT/L/A/ABf regions and the otherwise-unused producer QK/A' UB regions for exchange and coupling temporaries; it does not increase the static UB map.

Each AIV writes only its owned 32x32 L1 blocks: X0 and the zero upper-right block on AIV0, X1 on AIV1, and sixteen X10 rows on each AIV. No zero-filled block may overwrite the other AIV's result. Row bands are fully defined: invalid Q/K/V/W/U rows are zero-filled before upload.

State is split into contiguous 64-row halves. For `[K,V]`, each AIV owns 64 K rows; for `[V,K]`, each owns 64 V rows. FP32 state remains resident across chunks, while the BF16 mirror is used for Cube and public H output. Initial/final state may be FP32 or BF16 according to the existing API contract.

## 6. Output and HBM traffic

Only cross-role data is written to HBM unconditionally: W/U, gate cumsum, and normalized q/k when L2Norm is enabled. These are public or consumer inputs and cannot remain producer-local. The inverse exchange writes 12 KiB and reads 12 KiB per chunk; T and the AIV peer traffic have been eliminated.

Consumer-local `v_new` is uploaded directly from UB to L1 and is never written to GM. H is written only when `hOutOptional` is explicitly requested; A is written only when `aOutOptional` is explicitly requested. Otherwise their internal output slots contain rank-1 dummies and the corresponding stores are skipped. Public q_hat/k_hat/rstd and beta_eff retain the existing stable ABI behavior.

## 7. Numeric order

- Gate softplus, beta sigmoid, L2 epsilon, exp/exp2 selection, scale and output casts follow the existing wrapper/reference semantics.
- Prefix sum uses six register snapshot gather/add stages.
- Each diagonal inverse preserves the FP32 forward-substitution order of the 32-row solver. The coupling block evaluates `X1@L10@X0` as `(X1@L10)@X0` with native FP32 Cube GEMMs, which is algebraically equivalent to the former full 64-row substitution for `(I-L)^-1` but has a different FP32 FMA association for X10. There is no BF16 quantization of X0/X1/L10 before the coupling computation.
- Lower/attention gates and `right` evaluate `E(g[i]-g[j])` directly when the reciprocal path is unsafe. They never form `1/E(g[j])` in that mode; this avoids `0*Inf` when a strongly negative cumulative gate underflows. A' includes the diagonal.
- Recurrence state and vector epilogues remain FP32; A/W/U/A'/v_new/right/H cube inputs are BF16.

CPU checks for the independent inverse, scan, and full one-chunk dataflow are recorded in `validation.md`; these do not replace device accuracy tests. The dual-AIV inverse is a performance candidate until the same model case passes the existing accuracy gate and improves the frozen `Task Duration` baseline beyond run-to-run variation.
