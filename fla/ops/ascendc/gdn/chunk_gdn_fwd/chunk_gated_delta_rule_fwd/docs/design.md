# Arch35 Chunk64 Sparse GDN Forward Design

## 1. Scope

This is a clean chunk64 implementation for Ascend 950 / arch35. It targets BF16 Q/K/V, FP32 gate/beta, `K=V=128`, and low parallelism `B*Hv <= AIC/2`. The public ACLNN ABI and ten-element Python result are unchanged. The implementation is self-contained under `chunk_gated_delta_rule_fwd`; it does not include another operator's headers or call standalone Prepare/H/O operators.

Device compilation and performance are not claimed by this document; actual checks are listed in `validation.md`.

## 2. Dataflow

For each `(sequence, value-head)` chain, two MIX groups execute a one-chunk skewed pipeline.

Producer, per chunk:

1. Two AIVs load independent 32-row Q/K/V bands. They compute gate/beta transforms, a 64-element Sklansky prefix sum, exp/exp2 energies, optional L2 normalization, `vb=v*beta`, and `kbg=k_hat*beta*E(g)`.
2. AIC computes `KKT=K@K^T` and `QK=Q@K^T`.
3. AIV0 constructs `-raw` and computes `A=(I+raw)^-1` in FP32 registers. Eight rows stay live per VF scope; eight scopes cover the 64 rows. No FP32 Cube solve or FP32 L1 NZ path is used.
4. AIC computes `W=A@kbg` and `U=A@vb`. U reuses A already loaded in L0A by W.
5. Both AIVs compute their own `A'` row band from QK and the gate energies. W/U/A' and requested Prepare outputs are published through GM.

Consumer, one chunk behind:

1. Load W/U/Q/K/A' and gate cumsum; upload the current BF16 state mirror and inputs to L1.
2. AIC computes `P=W@H` and `O_s=Q@H`. The second product reuses H already loaded in L0B by P.
3. AIV computes `v_new=U-P` and `right=E(g_last-g)*v_new`, then publishes both to L1.
4. AIC computes `D=K^T@right` (or `right^T@K` for stateVFirst) and `O_l=A'@v_new`.
5. AIV writes `o=(E(g)*O_s+O_l)*scale`, updates the FP32 state with `H=E(g_last)*H+D`, and refreshes its BF16 cube mirror.

Per chain, the Cube MAC count is approximately 2.10M in producer and 3.67M in consumer. QK/A' and the FP32 triangular solve are not repeated in consumer.

## 3. Schedule and synchronization

Pipeline mode launches exactly `2*chainCount` MIX groups, so there are no idle groups in the mode-0 collective. Chain `c` uses producer group `c` and consumer group `chainCount+c`.

Iteration `i`:

- producer builds chunk `i` when it exists;
- consumer builds chunk `i-1` when it exists;
- every launched AIV executes one mode-0 barrier, including short varlen chains.

The mode-0 Set uses PIPE_MTE3 after a full local pipe barrier, and Wait blocks PIPE_MTE2. This publishes producer GM writes before consumer GM reads. AIC does not enter mode 0; its next chunk cannot start until its own AIVs publish that chunk's L1 inputs through mode 4.

Mode-4 flags are directional. AIV0/AIV1 use raw IDs 0..15. AIC waits IDs `id` and `id+16` for a two-AIV input and publishes to either AIV0 only or both AIVs as required. KKT is sent only to AIV0, avoiding unconsumed AIV1 credits.

Above the low-parallelism bound, each group runs producer then consumer for whole chains. A mode-1 AIV rendezvous publishes the chain inside that group; no mode-0 barrier is used.

## 4. A' workspace

A' is not part of the public schema. Host reserves `totalChunks * Hv * 64 * 64 * sizeof(bf16)` bytes, 512-byte aligned, in user workspace. Its tile index is identical to the H tile index:

- dense: `(batch*chunksPerBatch + chunk)*Hv + hv`;
- varlen: `(sequenceChunkPrefix + chunk)*Hv + hv`.

Each AIV writes/reads a 32-row, 4 KiB band. The mode-0 barrier is the cross-group visibility edge. No other serial H/O workspace is retained.

## 5. On-chip maps

| Resource | Producer | Consumer |
|---|---:|---:|
| UB per AIV | 144 KiB | 190 KiB |
| L1 per MIX group | 72 KiB | 152 KiB |
| L0A/L0B | two 32 KiB banks | two 32 KiB banks |
| L0C | four 64 KiB banks | four 64 KiB banks |

Both UB maps are below the 208 KiB regbase budget. `sparse_memory.h` contains compile-time bound assertions. Row bands are always fully defined: invalid Q/K/V/W/U rows are zero-filled before upload; A' workspace is fully written for each tile.

State is split into contiguous 64-row halves. For `[K,V]`, each AIV owns 64 K rows; for `[V,K]`, each owns 64 V rows. FP32 state remains resident across chunks, while the BF16 mirror is used for Cube and public H output. Initial/final state may be FP32 or BF16 according to the existing API contract.

## 6. Numeric order

- Gate softplus, beta sigmoid, L2 epsilon, exp/exp2 selection, scale and output casts follow the existing wrapper/reference semantics.
- Prefix sum uses six register snapshot gather/add stages.
- The inverse computes `x[i]=e[i]+sum(t<i, -raw[i,t]*x[t])`, equivalent to `(I+raw)^-1`.
- Energy ratios `E(g[i])/E(g[j])` form lower/attention gates. A' includes the diagonal.
- Recurrence state and vector epilogues remain FP32; A/W/U/A'/v_new/right/H cube inputs are BF16.

CPU checks for the independent inverse, scan, and full one-chunk dataflow are recorded in `validation.md`; these do not replace device accuracy tests.
