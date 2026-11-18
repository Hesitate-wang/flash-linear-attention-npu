# Arch35 Chunk64 Sparse GDN Forward Design

## 1. Scope

This implementation targets Ascend 950 / arch35, BF16 Q/K/V, FP32 gate/beta, `K=V=128`, and low parallelism `B*Hv <= AIC/2`. The public ACLNN ABI and ten-element Python result are unchanged. Every chain has one Producer group and one Consumer group; there is no serial scheduling mode.

The Producer computes KKT, complete 64x64 Vector inverses, QKT and masked attention APrime. It does not compute Kbg/Vb or W/U. Consumer recomputes W/U and fuses `U-W@S` directly in L0C. Internal T, normalized Q/K and APrime workspace records, final O and requested final_state reach GM.

## 2. Producer: parity-owned full chunks

AIV0 owns even chunks and AIV1 owns odd chunks. Each owner processes all 64 rows of its chunk; there is no upper/lower 32-row split. For pair iteration `i`:

```text
AIV0: preprocess(2i+2), load(2i+4), inverse(2i),   QKMask(2i)
AIV1: preprocess(2i+3), load(2i+5), inverse(2i+1), QKMask(2i+1)
AIC:  KKT(2i), KKT(2i+1), QKT(2i), QKT(2i+1)
```

Invalid chunk indices are skipped. Before iteration zero, each active AIV loads and preprocesses its first chunk and loads its second chunk. `preprocess` covers normalized K, gate/beta cumulative ratios and normalized Q. Raw K/G/beta/Q movement is independent MTE2 work for the following round's preprocess.

For a chunk owned in pair `r`, preprocess occurs in pair `r-1`; KKT, inverse, QKT and QKMask all complete in pair `r`. KKT and QKT Fixpipe the complete 64x64 FP32 result only to the owner AIV. AIC waits for the corresponding APrime publication before advancing, which releases the single KKT/QK UB targets and provides backpressure.

## 3. Complete 64x64 Vector inverse

For unit-lower-triangular `M=I-L`, the owner AIV builds the complete strict-lower FP32 matrix `L` from KKT, beta and the shared gate-ratio matrix, then evaluates:

```text
X[row] = I[row] + sum(L[row,col] * X[col], col < row)
T = cast_bf16(X)
```

Forward substitution runs entirely on 64-lane Vector registers in eight groups of eight rows. Earlier X rows are read from UB. No diagonal blocks, L10 leaf, FP32 Cube GEMM or coupling intermediate exist.

A complete 64x64 FP32 identity matrix is initialized once in `ubEye_` during `ProducerRole::Init`. Every inverse group loads its eight initial rows from this persistent buffer; identity elements are not regenerated with Compare/Select inside the inverse loop.

The owner writes all valid rows of the complete T directly to the internal `args.a` allocation. T is not a public output. QKMask uses the same gate-ratio matrix as the inverse and writes one complete 64x64 BF16 NZ APrime tile.

## 4. Buffer lifetime and reuse

Buffer indices use the owner-local pair index `chunk/2`, not global `chunk&1`:

- Raw K/Q and normalized K/Q use two slots selected by `pair&1`.
- K/Q in L1 use three slots selected by `pair%3`, protecting current KKT/QKT operands from the next preprocess.
- Gate/beta records use three slots selected by `pair%3`: current inverse, next preprocess and the load for the following preprocess.
- Full 64x64 FP32 gate ratios use two slots selected by `pair&1`; inverse and QKMask consume the current slot while preprocess fills the other.

Normalized K/Q UB is reused as inverse workspace in the opposite pipeline phase. Preprocess fills slot `(i+1)&1`, while inverse and QKMask use slot `i&1`. The FP32 lower matrix occupies normalized-K storage and the inverse X occupies normalized-Q storage. After substitution, L is dead, so its two 8 KiB halves hold BF16 T and APrime.

MTE3-to-Vector events are slot-specific rather than full-pipe drains. Input events 0/1 are set after each preprocess's K/Q L1 uploads and consumed only by that chunk's inverse. Output events 2/3 are set after T/APrime stores and consumed by preprocess two pairs later, before the same slot is reused. A chain tail waits only the final one or two unconsumed output events. Consequently, inverse in one slot does not drain preprocess uploads in the other slot.

Producer UB occupies 216 KiB per AIV, including the persistent 16 KiB identity matrix, within the 248 KiB budget. Producer L1 occupies 96 KiB for three full K slots and three full Q slots. No FP32 inverse records remain in L1.

## 5. Normalized Q/K exchange

Producer preprocess always performs L2Norm and writes each full 64x128 normalized Q/K tile to independent BF16 NZ GM workspace records at `normalizedQKOffset`; K storage follows Q storage. There is one record per global chunk and key head, so no Consumer acknowledgement is required before Producer reuses its on-chip slots.

Readiness is an AIV mode-0 generation per pair. Both Producer AIVs set the same flag after their even/odd Q/K GM stores, using alternating flag IDs 14 and 15. Producer waits only before reusing the pair-2 flag generation and drains at most the final two generations. Short and empty chains publish dummy generations.

Consumer first prepares G/Beta and Vb. At each even chunk, both Consumer AIVs join and wait for that pair, then issue the even and odd QHat/KHat GM transfers back-to-back into two UB slots with distinct MTE2 completion events. The even chunk is consumed first; the following odd chunk consumes its already-issued transfer without another pair wait or GM issue. Invalid even chunks join without loading. Consumer never repeats L2Norm; it only applies gate energy and beta to the normalized inputs.

K and Q have independent MTE2-to-Vector readiness events. K's UB-to-L1 MTE3 is issued before Kbg without waiting for Q. Kbg and QExp use separate V_MTE3 events for their W/QExp uploads, and W/OSB releases are tracked separately. K's input upload is awaited only before that UB is overwritten or at chain teardown.

## 6. Consumer: prepare in L1, then fuse W/U and FwdO

While Producer computes KKT/inverse/QKT, Consumer prepares:

```text
Kbg = -normalizedK * beta * energy
Vb  = V * beta
```

Kbg is negative so Cube accumulation computes `U-trueW@S` without a separate Vector subtraction. Once T is ready, Consumer AIC loads its valid GM rows and performs:

```text
W = T @ Kbg                  // W = -trueW
U = T @ Vb                   // keep U in L0C
accumulate W @ S into U      // vnew = U - trueW@S
O = APrime @ vnew + QExp @ S // accumulate in L0C
D = K^T @ (vnew * decay)
```

Gate/Beta use three modulo-3 UB slots. The prologue loads gates 0 and 1; iteration i issues gate i+2 before computing the already-loaded gate i+1. Gate readiness is published before the i+2 and V MTE2 work, so the gate-compute wait covers neither newer transfer. The third slot preserves the gate lifetime through PrepareRightDecay while future loads are in flight. Next-chunk Q/K GM loads remain immediately after gate preparation, with even chunks issuing the even/odd pair back-to-back. Next-chunk Vb and Kbg are computed during Prepare, but their U/K/W uploads are deferred until both current Right panels have been published, so the ordered UB-to-L1 transfers overlap current D rather than delaying its operands; only QExp(i+1) is deferred until the current full O is written.

All state-dependent Consumer data use two 64-column panels. T@Kbg and the full U=T@Vb run in RunCubePrepare before QS and independently of H; Each L0C accumulator has an independent FIX_M release, consumed only before the next GEMM that reinitializes the same region. The steady Cube order is QS-C1, QS-C2, W@S-C1, W@S-C2, D-C1, D-C2, then one full APrime@VNew. QS-C1 waits QExp and H-C1; QS-C2 then loads H-C2 into l0B1, which W@S-C2 reuses. D-C1 waits only Right-C1; APrime loads independently into l0A1, then D-C2 waits Right-C2, loads its own B and prefetches full VNew, so the full O reuses APrime/VNew without another L1-to-L0 load. VNew initializes its full 64x128 accumulator with one U GEMM and accumulates W@S by panel. Full Vb is prefetched into l0B0 during T@Kbg. D uses two 32 KiB regions of l0C1, W uses l0C2, and QS/O use l0C3, so the next round's W/U remains independent of the previous D Fixpipe. AIV computes next Vb/Kbg during Prepare, computes and publishes both current Right panels before the corresponding next-chunk U/K/W uploads, overlaps those uploads with current D, applies D-C1 and uploads H-C1, applies D-C2 and uploads H-C2, stores the full O, and finally prepares next QExp. FP32 state decay and the BF16 mirror follow the same column order, with existing AIV row partitioning unchanged. VNew-C1/C2 L1 readiness is awaited before the full O; VNew, W, O and the two D panels release their accumulators independently for the next chunk.

## 7. Synchronization

Cube helpers only issue loads, Mmad and Fixpipe. ProducerRole and ConsumerRole explicitly orchestrate all MTE1_M, M_MTE1, M_FIX, FIX_M and FIX_MTE1 events; compute primitives do not hide data movement or synchronization. QExp has its own mode-4 AIV-to-AIC ready event, consumed before QS-C1 loads it. FIX_M releases are region-specific: 0 for U/VNew, 1 for W, 2 for QS/O, 3 for D-C1 and 4 for D-C2. Each is primed at pipeline entry, set after its region's final Fixpipe, awaited before the next initialization of that region and drained at pipeline exit. QS-C2 publishes one M_MTE1(0) release for both ordered QS GEMMs. D-C1 uses M_MTE1(0) to release its B bank for the VNew prefetch, while D-C2 publishes M_MTE1(1) for the final O.

Mode-4 ownership is chunk-relative:

- `kProdK` / `kProdQ`: the owner has published full normalized K/Q to L1; AIC waits only for that owner.
- `kProdKkt` / `kProdQK`: AIC has delivered a full 64x64 result to the owner AIV.
- `kProdA`: the owner's complete T store has been issued through MTE3.
- `kProdAPrime`: the owner's complete APrime store has been issued through MTE3 and provides per-pair backpressure.

Short and empty Producer chains still join every valid global AIC generation. When `maxChunks` is odd, the nonexistent last lane does not create an extra generation. Consumer T and APrime mode-0 hand-offs remain separate.

## 8. Numeric order

KKT, QKT, gate processing, gate ratios and the complete inverse remain FP32. Inverse clamping is applied only while constructing L; the shared QKMask ratio remains unmodified. T and APrime are BF16. Consumer W/U and subsequent Cube operations use BF16 inputs with FP32 accumulation. Kbg's negation changes only representation: `U + (-trueW)@S` is algebraically identical to `U - trueW@S`.
