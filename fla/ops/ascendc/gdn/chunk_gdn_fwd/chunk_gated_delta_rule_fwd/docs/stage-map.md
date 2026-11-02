# Arch35 Chunk64 Stage Map

This file records the source-derived stage map used by the future single-launch kernel. It is not a public support declaration.

## Global domain

For dense BNSD, `work = ((b * NT + c) * Hv + hv)`, `hk = hv / (Hv/Hk)`, `BT=64`, `M=min(BT,T-c*BT)`. A full pack contains four work items. In MIX AIC:AIV 1:2, AIC owns the cube work and AIV0/AIV1 own alternating task slices. AIV0 handles task indices `0,2`; AIV1 handles `1,3`; ping/pong is `(task >> 1) & 1`. The state chain is ordered by `(b,hv,c)`.

## Stage 0: Vector, resident identity

AIV creates the 64x64 VCS identity and zeroes the two leaf buffers. AIV-owned leaf slots are disjoint. AIV0 additionally paints the shared 64x64 identity into L1 NZ. This stage has no GM input and must complete before the first pack uses L1 identity.

## Stage 1: Vector, normalize and gate

For each work item, load q/k `[M,128]`, raw gate and beta `[M]`. In FP32, compute `q_hat=q*rstd_q`, `k_hat=k*rstd_k`, with `rstd=1/sqrt(mean(x*x)+1e-6)` when enabled. Compute gate transform, chunk cumsum and optional `beta_eff=sigmoid(beta)` (or `2*sigmoid(beta)` for `allow_neg_eigval`). Invalid tail rows are zero-filled before vector math. The K source is written once to L1 NZ; q_hat/k_hat/rstd and beta_eff are written to their requested GM outputs. Stage 1 publishes per-task AIC readiness.

## Stage 2: Cube, KKT

For the K owner of each `(b,c,hk)`, compute `KKT = k_hat @ k_hat^T`, shape `[64,64]`, BF16 inputs and FP32 accumulation. L0A/L0B use the two parity banks; L0C is dumped to the owning AIV UB. GVA siblings receive the KKT-ready event and never recompute it.

## Stage 3: Vector, lower gate and leaves

Compute the lower-triangular gate matrix from cumsum and beta, multiply `KKT` elementwise, pack diagonal leaves and write `-L` and leaf matrices to L1. Copy the VCS identity to UB and perform the 32-leaf reduction/scatter. Publish L1 readiness to Stage 4.

## Stage 4: Cube, Y

For each task compute `Y = I + LeafLeft @ (-L)` in FP32. The identity product is prefetched into per-task L0C. Results are written through a per-task GM workspace slot and copied back to L1 because the L1 live set cannot hold all producer results without colliding with Stage 5/7 buffers. Stage 4 frees the L1 `-L` alias and publishes the slot for next-pack Stage 1 reuse.

## Stage 5: Cube, inverse coefficient

Compute `A = LeafLeft + Y @ LeafRight`, shape `[64,64]`, FP32 accumulation and BF16 storage. Full chunks optionally write A to GM and retain NZ in L1; tail rows remain masked. Stage 5 publishes A readiness to Stage 6/7.

## Stage 6: Vector, W/U inputs

Load V and beta-scale rows to form `vb=v*beta_eff`; load k_hat and form `kbg=k_hat*beta_eff*exp(g_cumsum)` (or `exp2` equivalent). Write both as L1 NZ. Tail rows are zero-filled. Stage 6 publishes readiness to Stage 7.

## Stage 7: Cube, W/U

Compute `W = A @ kbg` and `U = A @ vb`, shape `[M,128]`, FP32 accumulation and BF16 output. V=256 uses two GM writes for U halves; V=128 retains U in UB until the next pack drains it. W/U are then available to the H stage.

## H/O boundary

H consumes W/U and the prior state, and serializes each `(b,hv)` chain over chunks. It produces `h` and `v_new`. O may read a current chunk only after its H/v_new record is published. Existing H/O ready records have two 32-byte slots per `(physical batch,chunk,hv)` task and two AIV notifications. A future fused kernel must preserve this identity and must not reuse a slot until both O waits have consumed it.

## Resource baseline

The source-derived physical budgets are L1 512 KiB, UB 248 KiB, L0A 64 KiB, L0B 64 KiB, L0C 256 KiB. Prepare uses two UB ping/pong task slots and four L1 task slots. Stage 4 workspace stores one FP32 64x64 record per task slot. All GM records are 32-byte aligned. The exact fused-kernel workspace must add H/v_new and ready records after the Prepare live ranges; it must be computed by host tiling with overflow checks.

## Cross-core rules

Prepare's Stage 1–7 flags are intra-MIX-group CrossCore flags and cannot synchronize producer groups. For any new group boundary, use `IBSet<false>`/`IBWait<false>`, initialize every 32-byte flag before first use, use the same producer block index and event ID on both sides, and size flags with `eventStride=GetBlockNum()*2`. A ready event must have one producer and one consumer; empty/tail work is masked but still participates only when both sides have a defined record.

## Required implementation split

The eventual single launch must either (a) run all Prepare stages on the producer group and H/O on consumer groups with a proved queue per chain, or (b) assign complete chains to groups and use consumer subblocks only for independent vector halves. It must not split a single recurrence chain at an unready stage. Until one of these schedules is implemented and tested on 950, the public ACLNN route remains Prepare -> H -> O.
