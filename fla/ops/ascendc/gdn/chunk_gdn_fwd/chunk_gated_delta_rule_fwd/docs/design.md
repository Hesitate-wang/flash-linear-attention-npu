# Arch35 Chunk64 Sparse GDN Forward Design

## 1. Scope

This implementation targets Ascend 950 / arch35, BF16 Q/K/V, FP32 gate/beta, `K=V=128`, and low parallelism `B*Hv <= AIC/2`. The public ACLNN ABI and ten-element Python result are unchanged.

The Producer uses a packed 32x64 Vector forward substitution (VCS), two native-FP32 coupling GEMMs, two input slots, and alternating AIV ownership. The full 64x64 Neumann implementation has been removed from the active path because its GEMM/Fixpipe critical path increased latency.

## 2. Packed VCS and coupling

For unit-lower-triangular `M=I-L`:

```text
M = [ M00   0  ]       M^-1 = [ X0   0  ]
    [ -L10  M11 ]              [ X10  X1 ]

X0  = M00^-1
X1  = M11^-1
X10 = X1 @ L10 @ X0
```

One owner AIV receives the complete FP32 KKT. `InversePair32x64VF` packs the two independent diagonal solves into one 32x64 workspace:

```text
X = [ X0 | X1 ]
```

The left 32 lanes solve X0 and the right 32 lanes solve X1. Each forward-substitution FMA therefore works on both diagonal blocks. L10 is generated separately and is not part of the diagonal VCS.

The owner publishes X0/X1/L10 as FP32 records. AIC computes `T=X1@L10` and `X10=T@X0` with two native-FP32 32x32 GEMMs. The diagonal A blocks are uploaded by the owner, the lower-left block by AIC. Public A retains the same four-block layout.

## 3. Alternating AIV ownership and overlap

Chunk `c` uses owner `c&1`:

1. Both AIVs initially prepare gate/beta, normalized K and Kbg/Vb for chunk 0.
2. The owner waits for its complete KKT and executes packed VCS. The other AIV prepares gate/beta for chunk `c+1`.
3. After publishing the inverse, the owner also recomputes its local gate/beta copy for chunk `c+1`. Both AIVs then perform their own 32-row L2Norm(K), Kbg and Vb work while AIC performs coupling and W/U.
4. AIC computes the next KKT after the current W/U. Ownership alternates and the other AIV prepares the following gate/beta.

Each AIV keeps a local gate/beta copy for its own VCS and its 32-row scaling work. The duplicate work is only 64 elements per chunk. No L1-to-UB gate transfer and no GM gate roundtrip are used. Public g_cumsum/beta_eff are written only by AIV0.

K/Kbg/Vb use two slots. The current chunk's coupling and W/U overlap the other slot's Vector preparation; a single AIC still executes its own GEMMs serially.

## 4. W/U and synchronization

AIC writes W/U directly from L0C to GM with BF16 ND Fixpipe. Pipeline mode-0 remains on AIC after W/U. Consumer AIC arrives after QK and releases its AIVs through `kConsWuGmReady`; W's L1-ready flag remains `kConsW`.

Producer-specific mode-4 flags are:

- `kProdK`: both normalized K halves are ready for KKT.
- `kProdKkt`: the complete KKT is delivered to the parity owner.
- `kProdInverse`: the owner's X0/X1/L10 GM records are ready for coupling.
- `kProdInputs`: both Kbg/Vb halves are ready for W/U.
- `kProdA`: the owner's diagonal A blocks are ready in L1.

Ready flags are published before public g_cumsum/beta_eff/KHat/KRstd/A writes, so the MTE3 publication only covers data required by KKT/coupling/WU. Public writes retire on slot reuse or at kernel exit. K, V, gate and beta GM loads are issued back-to-back with independent MTE2 events and consumed in gate, K, V order. X1/L10 share one MTE2-to-MTE1 wait, and T/X0 share another.

Serial mode does not use mode-0. AIC publishes `kProdW` only after the final chunk, and AIV waits it before the existing mode-1 whole-chain hand-off.

## 5. Memory and workspace

Producer UB per AIV:

- Two 48 KiB input slots: K, V, normalized/public K, Vb and Kbg.
- One 16 KiB complete KKT, 8 KiB packed L, 8 KiB packed X and 4 KiB L10.
- Diagonal/zero BF16 staging, two 2 KiB gate slots and shared scan indices.
- Total: 150 KiB, below the 208 KiB regbase budget.

Producer L1 per MIX group:

- Two 48 KiB K/Vb/Kbg slots.
- One 8 KiB complete BF16 A and four 4 KiB native-FP32 coupling records.
- Total: 120 KiB, below the 512 KiB hardware budget.

The per-tile GM workspace contains X0, X1, L10 and T, each a 32x32 FP32 record. It is indexed by the same dense/varlen tile mapping as before. Public A/W/U retain their existing BHTD addresses and valid-row tail handling.

## 6. Numeric order

KKT, gate processing and both diagonal VCS blocks remain FP32. Coupling also remains native FP32 until the final BF16 cast. W/U Cube inputs and public A are BF16. This removes Neumann's extra full-chunk GEMM sequence without moving the diagonal solve or coupling to BF16 arithmetic.
