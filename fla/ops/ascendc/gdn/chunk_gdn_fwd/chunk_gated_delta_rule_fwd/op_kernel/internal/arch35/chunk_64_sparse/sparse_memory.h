/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 *
 * On-chip maps for the chunk64 producer/consumer roles.
 *
 * The serial Prepare keeps four tasks in flight and therefore needs 236 KiB of
 * UB and 496 KiB of L1. A role here works on exactly one chunk at a time and
 * only its own stage set, so both maps shrink and stay explicit: every region
 * below is a half-open byte range checked by the static asserts at the bottom.
 *
 * UB is per AIV (248 KiB each), L1 is per AI Core (512 KiB, shared by its two
 * AIVs), L0A/L0B are 64 KiB and L0C is 256 KiB.
 */
#ifndef GDN_CHUNK64_SPARSE_MEMORY_H
#define GDN_CHUNK64_SPARSE_MEMORY_H

#include <cstdint>

namespace GdnChunk64Sparse {

constexpr uint32_t kKb = 1024;
constexpr uint32_t kUbBytes = 248 * kKb;

// ---------------------------------------------------------------------------
// Producer (one chunk in flight). Raw q and raw v share the first slot: raw q
// dies once q_hat is written, and raw v only appears in Stage 6 afterwards.
// Raw k likewise hands its slot to kbg.
// ---------------------------------------------------------------------------
constexpr uint32_t kProdUbQ = 0;                              // bf16 [64,128] raw q, then raw v
constexpr uint32_t kProdUbK = kProdUbQ + 16 * kKb;            // bf16 [64,128] raw k, then kbg
constexpr uint32_t kProdUbQHat = kProdUbK + 16 * kKb;         // bf16 [64,128]
constexpr uint32_t kProdUbKHat = kProdUbQHat + 16 * kKb;      // bf16 [64,128]
constexpr uint32_t kProdUbKkt = kProdUbKHat + 16 * kKb;       // fp32 [64,64]
constexpr uint32_t kProdUbL = kProdUbKkt + 16 * kKb;          // fp32 [64,64] full gate matrix
constexpr uint32_t kProdUbLPacked = kProdUbL + 16 * kKb;      // fp32 2 x [32,64] leaves
constexpr uint32_t kProdUbResVcs = kProdUbLPacked + 16 * kKb; // fp32 2 x [32,32]
constexpr uint32_t kProdUbIVcs = kProdUbResVcs + 8 * kKb;     // fp32 2 x [32,32] identity
constexpr uint32_t kProdUbW = kProdUbIVcs + 8 * kKb;          // bf16 [64,128]
constexpr uint32_t kProdUbU = kProdUbW + 16 * kKb;            // bf16 [64,128]
constexpr uint32_t kProdUbVb = kProdUbU + 16 * kKb;           // bf16 [64,128]
constexpr uint32_t kProdUbScale = kProdUbVb + 16 * kKb;       // fp32 [64] g', [64] beta, [128] rstd
constexpr uint32_t kProdUbVcsIdx = kProdUbScale + 2 * 512;    // uint32 scatter index
constexpr uint32_t kProdUbEnd = kProdUbVcsIdx + 512;

static_assert(kProdUbEnd <= kUbBytes, "producer UB map exceeds 248 KiB");
// The regbase/SIMT reservation is a target, not a hard cap: the shipped chunk32
// fused kernel runs at 224 KiB. Keep the producer under it.
static_assert(kProdUbEnd <= 224 * kKb, "producer UB map exceeds the 224 KiB budget");

constexpr uint32_t kProdL1KHat = 0;                              // bf16 NZ [64,128]
constexpr uint32_t kProdL1NegL = kProdL1KHat + 16 * kKb;         // fp32 NZ [64,64]
constexpr uint32_t kProdL1LeafL = kProdL1NegL + 16 * kKb;        // fp32 NZ [64,64]
constexpr uint32_t kProdL1LeafR = kProdL1LeafL + 16 * kKb;       // fp32 NZ [64,64]
constexpr uint32_t kProdL1Y = kProdL1LeafR + 16 * kKb;           // fp32 NZ [64,64]
constexpr uint32_t kProdL1A = kProdL1Y + 16 * kKb;               // bf16 NZ [64,64]
constexpr uint32_t kProdL1Kbg = kProdL1A + 8 * kKb;              // bf16 NZ [64,128]
constexpr uint32_t kProdL1Vb = kProdL1Kbg + 16 * kKb;            // bf16 NZ [64,128]
constexpr uint32_t kProdL1I = kProdL1Vb + 16 * kKb;              // fp32 NZ [64,64] resident
constexpr uint32_t kProdL1End = kProdL1I + 16 * kKb;

static_assert(kProdL1End <= 512 * kKb, "producer L1 map exceeds 512 KiB");

// L0A/L0B are 64 KiB each; two 16 KiB slots cover every producer matmul.
constexpr uint32_t kProdL0Slot = 16 * kKb;
constexpr uint32_t kProdL0Pair[2] = {0, 16 * kKb};
// L0C: four 16 KiB result tiles on a 64 KiB stride, the same convention the
// serial Prepare uses so a tile never straddles a 64 KiB bank.
constexpr uint32_t kProdL0CStride = 64 * kKb;
constexpr uint32_t kProdL0CTile[4] = {0, 64 * kKb, 128 * kKb, 192 * kKb};

// ---------------------------------------------------------------------------
// Consumer. The two AIVs split the chunk-row axis (AIV0 rows 0..31, AIV1 rows
// 32..63) and the recurrent state along K (AIV0 K rows 0..63, AIV1 64..127),
// the same convention chunk32 uses. The cube computes full tiles and fixes them
// out with dualDstCtl split-by-M, so every row-shaped vector buffer below holds
// 32 rows and every state-shaped buffer 64 K rows.
// The AIC never touches GM: the AIV is the only GM gateway, which keeps the
// producer->consumer publication edge AIV->AIV (covered by the mode-0 barrier).
// ---------------------------------------------------------------------------
constexpr uint32_t kConsUbStateF = 0;                              // fp32 [64,128] K half, resident
constexpr uint32_t kConsUbStateB = kConsUbStateF + 32 * kKb;       // bf16 [64,128] K half
constexpr uint32_t kConsUbD = kConsUbStateB + 16 * kKb;            // fp32 [64,128] K half of k^T@right
constexpr uint32_t kConsUbP = kConsUbD + 32 * kKb;                 // fp32 [32,128] W@H row half
constexpr uint32_t kConsUbU = kConsUbP + 16 * kKb;                 // bf16 [32,128] row half
constexpr uint32_t kConsUbVNew = kConsUbU + 8 * kKb;               // bf16 [32,128] row half
constexpr uint32_t kConsUbRight = kConsUbVNew + 8 * kKb;           // bf16 [32,128] row half
constexpr uint32_t kConsUbOS = kConsUbRight + 8 * kKb;             // fp32 [32,128] O_s row half
constexpr uint32_t kConsUbOL = kConsUbOS + 16 * kKb;               // fp32 [32,128] O_l row half
constexpr uint32_t kConsUbARaw = kConsUbOL + 16 * kKb;             // fp32 [32,64] Q@K^T row half
constexpr uint32_t kConsUbGateA = kConsUbARaw + 8 * kKb;           // fp32 [32,64] gateA row half
constexpr uint32_t kConsUbAPrime = kConsUbGateA + 8 * kKb;         // bf16 [32,64] row half
constexpr uint32_t kConsUbGates = kConsUbAPrime + 4 * kKb;         // fp32 g[64], gateScale[64], gateO[64]
constexpr uint32_t kConsUbScalar = kConsUbGates + 3 * 256;         // fp32 gLast, alpha
constexpr uint32_t kConsUbEnd = kConsUbScalar + 256;

static_assert(kConsUbEnd <= kUbBytes, "consumer UB map exceeds 248 KiB");
static_assert(kConsUbEnd <= 208 * kKb, "consumer UB map misses the 208 KiB regbase target");

constexpr uint32_t kConsL1W = 0;                                  // bf16 NZ [64,128]
constexpr uint32_t kConsL1Q = kConsL1W + 16 * kKb;                // bf16 NZ [64,128]
constexpr uint32_t kConsL1K = kConsL1Q + 16 * kKb;                // bf16 NZ [64,128]
// H is ping-ponged by chunk parity so the next upload cannot land before the
// current chunk's Q@H mmad; one image serves both W@H and Q@H of its chunk.
constexpr uint32_t kConsL1H[2] = {kConsL1K + 16 * kKb, kConsL1K + 48 * kKb};
constexpr uint32_t kConsL1VNew = kConsL1H[1] + 32 * kKb;          // bf16 NZ [64,128]
constexpr uint32_t kConsL1Right = kConsL1VNew + 16 * kKb;         // bf16 NZ [64,128]
constexpr uint32_t kConsL1APrime = kConsL1Right + 16 * kKb;       // bf16 NZ [64,64]
constexpr uint32_t kConsL1End = kConsL1APrime + 8 * kKb;

static_assert(kConsL1End <= 512 * kKb, "consumer L1 map exceeds 512 KiB");

constexpr uint32_t kConsL0Slot = 16 * kKb;
constexpr uint32_t kConsL0Pair[2] = {0, 16 * kKb};
constexpr uint32_t kConsL0Wide = 32 * kKb;  // [128,128] bf16 or [64,128] fp32
constexpr uint32_t kConsL0CStride = 64 * kKb;
constexpr uint32_t kConsL0CTile[4] = {0, 64 * kKb, 128 * kKb, 192 * kKb};

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_MEMORY_H
