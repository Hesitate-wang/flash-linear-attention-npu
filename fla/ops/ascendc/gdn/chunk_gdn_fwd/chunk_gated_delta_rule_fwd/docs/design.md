# Arch35 Chunk64 Sparse GDN Forward Design

## 1. Scope

This implementation targets Ascend 950 / arch35, BF16 Q/K/V, FP32 gate/beta, `K=V=128`, and low parallelism `B*Hv <= AIC/2`. The public ACLNN ABI and ten-element Python result are unchanged.

The Producer computes only KKT, the packed inverse and its lower-left coupling. It no longer computes Kbg/Vb or W/U. Consumer recomputes W/U and fuses `U-W@S` directly in L0C. The full 64x64 Neumann implementation is not active.

Ascend 950 provides 512 KiB L1, 64 KiB L0A, 64 KiB L0B and 256 KiB L0C per AI Core. The Consumer keeps T, Kbg, Vb, W, U, H, vnew and right in L1/L0C wherever practical instead of making repeated GM roundtrips.

## 2. Producer: packed inverse only

For unit-lower-triangular `M=I-L`:

```text
M = [ M00   0  ]       M^-1 = [ X0   0  ]
    [ -L10  M11 ]              [ X10  X1 ]

X0  = M00^-1
X1  = M11^-1
X10 = X1 @ L10 @ X0
```

One owner AIV receives the complete FP32 KKT. `InversePair32x64VF` packs both diagonal solves into one 32x64 workspace, using the left 32 lanes for X0 and right 32 lanes for X1. L10 is generated separately.

The owner publishes X0/X1/L10 as FP32 records. AIC computes `T=X1@L10` and `X10=T@X0` with two native-FP32 32x32 GEMMs. The owner writes the diagonal blocks of the complete inverse T, and AIC writes its lower-left block. T uses the internal `args.a` buffer and is not a public output. That buffer is allocated at full `[B,Hv,T,64]` size even when `aOutOptional` is null.

Producer does not load V, compute Kbg/Vb, or execute `T@Kbg` / `T@Vb`. There is no Producer W/U GM store.

## 3. Alternating AIV preparation

Chunk `c` uses owner `c&1`:

1. Both AIVs initially prepare gate/beta and normalized K for chunk 0.
2. The owner waits for its complete KKT and executes packed VCS. The other AIV prepares gate/beta for chunk `c+1`.
3. After publishing the inverse, the owner recomputes its local gate/beta copy for chunk `c+1`; both perform their own 32-row L2Norm(K).
4. AIC computes the next KKT after the current inverse is published. Ownership alternates.

Each AIV keeps a local gate/beta copy for VCS and scaling. Duplicate work is only 64 elements per chunk; no L1-to-UB or GM gate sharing is used. Public intermediate outputs are disabled.

## 4. Consumer: prepare in L1, then fuse W/U and FwdO

While Producer computes KKT/inverse, Consumer prepares:

```text
Kbg = -normalizedK * beta * energy   // negative by construction
Vb  = V * beta
```

Kbg and Vb are written from UB to L1. Kbg is negative so the later Cube accumulation computes `U-trueW@S` without a separate Vector subtraction.

Consumer also computes QK and masked attention A' and leaves it in L1.

Once T is ready, Consumer AIC performs:

```text
T GM -> L1
W = T @ Kbg                  // W = -trueW
U = T @ Vb                   // keep U in L0C
accumulate W @ S into U      // vnew = U - trueW@S
O = A' @ vnew + QExp @ S     // accumulate in L0C
D = K^T @ (vnew * decay)
```

`FixF32Halves` sends vnew to AIV for the decayed right computation. A'@vnew accumulates into the QExp@S accumulator and is scaled during the final BF16 output Fixpipe. D returns to AIV, which performs the FP32 state update and refreshes the BF16 state mirror for the next chunk.

## 5. Synchronization

Pipeline mode-0 remains on AIC after the complete T has been written. Consumer AIC arrives after QK and then loads T itself; the former Consumer W/U GM-ready hand-off is removed.

Producer-specific mode-4 flags are:

- `kProdK`: both normalized K halves are ready for KKT.
- `kProdKkt`: the complete KKT is delivered to the parity owner.
- `kProdInverse`: X0/X1/L10 are ready for coupling.
- `kProdA`: the diagonal blocks of T are ready.

Consumer-specific flags are:

- `kConsP`: vnew is ready in UB after U-W@S accumulation.
- `kConsVn`: vnew/right are ready in L1 for O_l and D.
- `kConsOL`: the final O accumulator is ready.
- `kConsD`: D is ready for the state update.

Serial mode does not use mode-0. AIC publishes `kProdW` only after the final T, and AIV waits it before the existing mode-1 whole-chain hand-off.

## 6. Memory

Producer UB uses two 48 KiB K/input slots plus KKT, packed L/X/L10 and diagonal staging. Producer L1 uses two K slots and native-FP32 coupling records. The per-tile workspace contains packed X, L10 and T.

Consumer UB is 208 KiB per AIV, within the 208 KiB regbase budget. Consumer L1 is 192 KiB per MIX group, within the 512 KiB hardware budget. Existing regions are reused:

- `l1W_`: negative Kbg, then W.
- `l1U_`: Vb.
- `l1T_`: the complete inverse.
- `l1H_[2]`: alternating state mirrors.
- `l1APrime_`: masked attention.
- `l1VNew_` and `l1Right_`: vnew and decayed vnew.

Public A/g_cumsum/KHat/KRstd/H/qHat/qRstd are not written. Only the internal T, final O and final_state reach GM.

## 7. Numeric order

KKT, gate processing and diagonal VCS remain FP32. Coupling remains native FP32 until the final BF16 cast. T, Kbg, Vb, W/U and subsequent Cube inputs are BF16 with FP32 accumulation. Kbg's negation changes only its representation: `U + (-trueW)@S` is algebraically identical to `U - trueW@S`.
