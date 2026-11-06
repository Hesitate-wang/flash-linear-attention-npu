# Arch35 Chunk64 Sparse GDN Forward Design

## 1. Scope

This is a clean chunk64 implementation for Ascend 950 / arch35. It targets BF16 Q/K/V, FP32 gate/beta, `K=V=128`, and low parallelism `B*Hv <= AIC/2`. The public ACLNN ABI and ten-element Python result are unchanged. The implementation is self-contained under `chunk_gated_delta_rule_fwd`; it does not include another operator's headers or call standalone Prepare/H/O operators.

The measured low-parallelism model case is `B=1,Hv=1,T=16384,K=V=128`; its current fused-kernel `Task Duration` baseline is 2400 us. The comparison `(1,14,16384,128)` chunk32 case takes 850 us but has fourteen independent chains, so it is not an equal-work baseline. An absolute performance target and the final warm-up/sample protocol remain to be frozen with the device measurements.

## 2. Dataflow

For each `(sequence, value-head)` chain, two MIX groups execute a two-chunk skewed pipeline.

Producer, per chunk:

1. Two AIVs load independent 32-row Q/K/V bands. They compute gate/beta transforms, a 64-element Sklansky prefix sum, exp/exp2 energies, optional L2 normalization, `vb=v*beta`, and `kbg=k_hat*beta*E(g)`.
2. AIC computes `KKT=K@K^T` and sends the upper/lower 32-row bands to AIV0/AIV1 respectively.
3. The two AIVs compute the inverse in FP32. For `L=I-raw=[[L00,0],[L10,L11]]`, AIV0 computes `X0=L00^-1` and AIV1 computes `X1=L11^-1` independently. Each then computes sixteen rows of `X10=-X1@L10@X0`; all inverse arithmetic remains FP32 until the final BF16 cast.
4. AIC computes `W=A@kbg` and `U=A@vb`. U reuses A already loaded in L0A by W.
5. W/U and requested Prepare outputs are published through GM.

Consumer, two chunks behind:

1. Load W/U/Q/K and raw gate; recompute gate cumsum, energies and optional L2 normalization. Upload the current BF16 state mirror and inputs to L1.
2. AIC computes `QK=Q@K^T`; both AIVs form their own `A'` row band from QK and gate energies.
3. AIC computes `P=W@H` and `O_s=Q@H`. The second product reuses H already loaded in L0B by P.
4. AIV computes `v_new=U-P` and `right=E(g_last-g)*v_new`, then publishes both to L1.
5. AIC computes `D=K^T@right` (or `right^T@K` for stateVFirst) and `O_l=A'@v_new`.
6. AIV writes `o=(E(g)*O_s+O_l)*scale`, updates the FP32 state with `H=E(g_last)*H+D`, and refreshes its BF16 cube mirror.

Per chain, the current main Cube MAC count is approximately 1.57M in producer and 4.19M in consumer. Consumer-side QK/A' is concurrent with producer work; only W/U cross the role boundary. The inverse merge adds two logical 32x32 matrix products, implemented by the two AIVs in FP32 rather than by extra BF16 Cube operations.

## 3. Schedule and synchronization

Pipeline mode launches exactly `2*chainCount` MIX groups, so there are no idle groups in the mode-0 collective. Chain `c` uses producer group `c` and consumer group `chainCount+c`.

Iteration `i`:

- producer builds chunk `i` when it exists;
- consumer builds chunk `i-2` when it exists;
- every launched AIV executes one mode-0 barrier after each odd iteration and the final drain, including short varlen chains.

The mode-0 Set uses PIPE_MTE3 and Wait blocks PIPE_MTE2. This publishes producer GM writes before consumer GM reads. AIC does not enter mode 0; its next chunk cannot start until its own AIVs publish that chunk's L1 inputs through mode 4.

Mode-4 flags are directional. AIV0/AIV1 use raw IDs 0..15. AIC waits IDs `id` and `id+16` for a two-AIV input and publishes to either AIV0 only or both AIVs as required.

The inverse uses one additional per-chunk mode-1 AIV rendezvous. AIV0 writes X0; AIV1 writes X1 and L10. The PIPE_MTE3 Set publishes these FP32 workspace blocks, and the PIPE_MTE2 Wait allows each AIV to read the blocks required for its coupling rows. The flag is consumed before the next chunk, so no credit leaks across iterations. A local V_MTE2 barrier prevents the next chunk's peer-buffer load from overwriting data still read by the previous coupling task, and MTE2_V covers the current load before Vector use.

Above the low-parallelism bound, each group runs producer then consumer for whole chains. A mode-1 AIV rendezvous publishes the chain inside that group; no mode-0 barrier is used.

## 4. Inverse exchange workspace

Host reserves `totalChunks * Hv * 3 * 32 * 32 * sizeof(float)` bytes, 512-byte aligned, in user workspace. Its tile index is:

- dense: `(batch*chunksPerBatch + chunk)*Hv + hv`;
- varlen: `(sequenceChunkPrefix + chunk)*Hv + hv`.

Each tile contains three aligned 4 KiB FP32 blocks: X0, X1 and L10. AIV0 reads X1/L10 (8 KiB), and AIV1 reads X0 (4 KiB); writes total 12 KiB and reads total 12 KiB per chunk. Blocks within a tile are disjoint, and different chains/chunks never share a tile. The workspace is not an output and is not part of the public ABI.

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

Only cross-role data is written to HBM unconditionally: W/U, gate cumsum, and normalized q/k when L2Norm is enabled. These are public or consumer inputs and cannot remain producer-local. The inverse exchange adds the 24 KiB per-chunk GM traffic described above.

Consumer-local `v_new` is uploaded directly from UB to L1 and is never written to GM. H is written only when `hOutOptional` is explicitly requested; A is written only when `aOutOptional` is explicitly requested. Otherwise their internal output slots contain rank-1 dummies and the corresponding stores are skipped. Public q_hat/k_hat/rstd and beta_eff retain the existing stable ABI behavior.

## 7. Numeric order

- Gate softplus, beta sigmoid, L2 epsilon, exp/exp2 selection, scale and output casts follow the existing wrapper/reference semantics.
- Prefix sum uses six register snapshot gather/add stages.
- Each diagonal inverse preserves the FP32 forward-substitution order of the 32-row solver. The coupling block evaluates `-X1@L10@X0` as `-(X1@L10)@X0`, which is algebraically equivalent to the former full 64-row substitution but has a different FP32 FMA association for X10. There is no BF16 quantization of X0/X1/L10 before the coupling computation.
- Lower/attention gates and `right` evaluate `E(g[i]-g[j])` directly when the reciprocal path is unsafe. They never form `1/E(g[j])` in that mode; this avoids `0*Inf` when a strongly negative cumulative gate underflows. A' includes the diagonal.
- Recurrence state and vector epilogues remain FP32; A/W/U/A'/v_new/right/H cube inputs are BF16.

CPU checks for the independent inverse, scan, and full one-chunk dataflow are recorded in `validation.md`; these do not replace device accuracy tests. The dual-AIV inverse is a performance candidate until the same model case passes the existing accuracy gate and improves the frozen `Task Duration` baseline beyond run-to-run variation.
