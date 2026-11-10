# Arch35 Chunk64 Sparse GDN Forward Design

## 1. Scope

This implementation targets Ascend 950 / arch35, BF16 Q/K/V, FP32 gate/beta, `K=V=128`, and low parallelism `B*Hv <= AIC/2`. The public ACLNN ABI and ten-element Python result are unchanged.

The Producer computes only KKT, two independent 32x32 diagonal inverses and their lower-left coupling. It no longer computes Kbg/Vb or W/U. Consumer recomputes W/U and fuses `U-W@S` directly in L0C. The packed 32x64 VCS and full 64x64 Neumann implementations are not active.

Ascend 950 provides 512 KiB L1, 64 KiB L0A, 64 KiB L0B and 256 KiB L0C per AI Core. The Consumer keeps T, Kbg, Vb, W, U, H, vnew and right in L1/L0C wherever practical instead of making repeated GM roundtrips.

## 2. Producer: split diagonal inverses

For unit-lower-triangular `M=I-L`:

```text
M = [ M00   0  ]       M^-1 = [ X0   0  ]
    [ -L10  M11 ]              [ X10  X1 ]

X0  = M00^-1
X1  = M11^-1
X10 = X1 @ L10 @ X0
```

AIC sends KKT rows 0:32 to AIV0 and rows 32:64 to AIV1. AIV0 constructs L00 and solves X0; AIV1 constructs L11 and solves X1. AIV1 also generates L10 from KKT rows 32:64 and columns 0:32. L10 is a complete 32x32 leaf: every local element satisfies the global causal condition, so no local triangular mask is applied.

Each AIV keeps its FP32 diagonal inverse as ND in UB and uploads it to native-FP32 NZ L1 with four strided MTE3 copies. AIV1 uploads the complete L10 leaf in the same way. AIC computes `P=X1@L10`, Fixpipes P directly from FP32 L0C to FP32 NZ L1, and computes `X10=P@X0`. No inverse operand or intermediate coupling result reaches GM. The AIVs write the diagonal blocks of the complete inverse T, and AIC writes its lower-left block. T uses the internal `args.a` buffer and is not a public output. That buffer is allocated at full `[B,Hv,T,64]` size even when `aOutOptional` is null.

Producer does not load V, compute Kbg/Vb, or execute `T@Kbg` / `T@Vb`. There is no Producer W/U GM store.

## 3. Dual-AIV preparation and inverse

For every chunk:

1. Both AIVs prepare gate/beta and their own 32-row normalized K band.
2. AIC computes KKT and delivers the two 32x64 row bands to the corresponding AIVs.
3. Both AIVs wait for KKT and execute independent 32x32 VCS solves. AIV1 additionally publishes the complete L10 leaf.
4. After publishing their inverse blocks, both AIVs prepare the next chunk's gate/beta and normalized K. AIC performs the lower-left coupling.

Each AIV keeps a local gate/beta copy for VCS. Duplicate work is only 64 elements per chunk; no L1-to-UB or GM gate sharing is used. Public intermediate outputs are disabled.

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
- `kProdKkt`: both 32x64 KKT row bands are delivered to their AIVs.
- `kProdInverse`: X0, X1 and the complete L10 are ready in FP32 NZ L1 for coupling.
- `kProdA`: the diagonal blocks of T are ready.

Consumer-specific flags are:

- `kConsP`: vnew is ready in UB after U-W@S accumulation.
- `kConsVn`: vnew/right are ready in L1 for O_l and D.
- `kConsOL`: the final O accumulator is ready.
- `kConsD`: D is ready for the state update.

Serial mode does not use mode-0. AIC publishes `kProdW` only after the final T, and AIV waits it before the existing mode-1 whole-chain hand-off.

## 6. Memory

Producer UB uses two 48 KiB K/input slots plus per-AIV KKT row bands, diagonal L/X storage and the complete L10 leaf. Producer L1 uses two K slots and native-FP32 X0/X1/L10/P records. The former per-tile inverse workspace is removed; its tiling fields remain zero for ABI compatibility.

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
