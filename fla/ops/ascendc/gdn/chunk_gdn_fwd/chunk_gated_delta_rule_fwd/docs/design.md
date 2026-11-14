# Arch35 Chunk64 Sparse GDN Forward Design

## 1. Scope

This implementation targets Ascend 950 / arch35, BF16 Q/K/V, FP32 gate/beta, `K=V=128`, and low parallelism `B*Hv <= AIC/2`. The public ACLNN ABI and ten-element Python result are unchanged. Every chain has one Producer group and one Consumer group; there is no serial scheduling mode.

The Producer computes KKT, complete 64x64 Vector inverses, QKT and masked attention APrime. It does not compute Kbg/Vb or W/U. Consumer recomputes W/U and fuses `U-W@S` directly in L0C. Only the internal T, APrime workspace, final O and requested final_state reach GM.

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

## 5. Consumer: prepare in L1, then fuse W/U and FwdO

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

`FixF32Halves` sends vnew to the two Consumer AIVs for the decayed right computation. D returns to AIV, which performs the FP32 state update and refreshes the BF16 state mirror for the next chunk.

## 6. Synchronization

Mode-4 ownership is chunk-relative:

- `kProdK` / `kProdQ`: the owner has published full normalized K/Q to L1; AIC waits only for that owner.
- `kProdKkt` / `kProdQK`: AIC has delivered a full 64x64 result to the owner AIV.
- `kProdA`: the owner's complete T store has been issued through MTE3.
- `kProdAPrime`: the owner's complete APrime store has been issued through MTE3 and provides per-pair backpressure.

Short and empty Producer chains still join every valid global AIC generation. When `maxChunks` is odd, the nonexistent last lane does not create an extra generation. Consumer T and APrime mode-0 hand-offs remain separate.

## 7. Numeric order

KKT, QKT, gate processing, gate ratios and the complete inverse remain FP32. Inverse clamping is applied only while constructing L; the shared QKMask ratio remains unmodified. T and APrime are BF16. Consumer W/U and subsequent Cube operations use BF16 inputs with FP32 accumulation. Kbg's negation changes only representation: `U + (-trueW)@S` is algebraically identical to `U - trueW@S`.
