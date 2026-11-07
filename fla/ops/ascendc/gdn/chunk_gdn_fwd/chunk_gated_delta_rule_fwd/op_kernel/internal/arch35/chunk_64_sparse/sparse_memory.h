/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_CHUNK64_SPARSE_MEMORY_H
#define GDN_CHUNK64_SPARSE_MEMORY_H

#include <cstdint>

namespace GdnChunk64Sparse {
constexpr uint32_t kKiB = 1024;
constexpr uint32_t kUbBudget = 208 * kKiB;

// Producer UB, per AIV. Row-shaped tensors contain a 32-row band. KKT/L/A hold
// that AIV's diagonal inverse inputs and FP32 temporaries. Q and its public
// output are no longer produced here; kPUbQHat is reused only for public KHat ND.
constexpr uint32_t kPUbQ = 0; // Unused after removing producer Q.
constexpr uint32_t kPUbK = 8 * kKiB;
constexpr uint32_t kPUbV = 16 * kKiB;
constexpr uint32_t kPUbQHat = 24 * kKiB; // Independent 8 KiB public KHat ND buffer.
constexpr uint32_t kPUbKHat = 32 * kKiB;
constexpr uint32_t kPUbVb = 40 * kKiB;
constexpr uint32_t kPUbKbg = 48 * kKiB;
constexpr uint32_t kPUbW = 56 * kKiB;
constexpr uint32_t kPUbU = 64 * kKiB;
constexpr uint32_t kPUbQk = 72 * kKiB; // Unused after moving T to direct L0C-to-L1 Fixpipe.
constexpr uint32_t kPUbKkt = 80 * kKiB;
constexpr uint32_t kPUbL = 96 * kKiB;
constexpr uint32_t kPUbA = 112 * kKiB;
constexpr uint32_t kPUbABf = 128 * kKiB;
constexpr uint32_t kPUbAPrime = 136 * kKiB; // Unused 4 KiB slot; not large enough for KHat ND.
constexpr uint32_t kPUbGates = 140 * kKiB;
constexpr uint32_t kPUbShift = 142 * kKiB;
constexpr uint32_t kPUbEnd = 144 * kKiB;
static_assert(kPUbEnd <= kUbBudget);

constexpr uint32_t kPL1K = 0;
constexpr uint32_t kPL1InvX0 = 16 * kKiB;
constexpr uint32_t kPL1InvX1 = 20 * kKiB;
constexpr uint32_t kPL1InvL10 = 24 * kKiB;
constexpr uint32_t kPL1InvT = 28 * kKiB;
constexpr uint32_t kPL1Vb = 32 * kKiB;
constexpr uint32_t kPL1Kbg = 48 * kKiB;
constexpr uint32_t kPL1A = 64 * kKiB;
constexpr uint32_t kPL1End = 72 * kKiB;
static_assert(kPL1End <= 512 * kKiB);

// Consumer UB, per AIV.
constexpr uint32_t kCUbStateF = 0;
constexpr uint32_t kCUbStateB = 32 * kKiB;
constexpr uint32_t kCUbD = 48 * kKiB;
constexpr uint32_t kCUbP = 80 * kKiB;
constexpr uint32_t kCUbOS = 96 * kKiB;
constexpr uint32_t kCUbQHat = 112 * kKiB;
constexpr uint32_t kCUbKHat = 120 * kKiB;
constexpr uint32_t kCUbU = 128 * kKiB;
constexpr uint32_t kCUbVNew = 136 * kKiB;
constexpr uint32_t kCUbRight = 144 * kKiB;
constexpr uint32_t kCUbW = 152 * kKiB;
constexpr uint32_t kCUbQ = 160 * kKiB;
constexpr uint32_t kCUbK = 168 * kKiB;
constexpr uint32_t kCUbAPrime = 176 * kKiB;
constexpr uint32_t kCUbO = 180 * kKiB;
constexpr uint32_t kCUbGates = 188 * kKiB;
constexpr uint32_t kCUbShift = 190 * kKiB;
constexpr uint32_t kCUbQk = 192 * kKiB;
constexpr uint32_t kCUbEnd = 200 * kKiB;
static_assert(kCUbEnd <= kUbBudget);

constexpr uint32_t kCL1W = 0;
constexpr uint32_t kCL1U = 16 * kKiB;
constexpr uint32_t kCL1QHat = 32 * kKiB;
constexpr uint32_t kCL1QExp = 48 * kKiB;
constexpr uint32_t kCL1K = 64 * kKiB;
constexpr uint32_t kCL1H0 = 80 * kKiB;
constexpr uint32_t kCL1H1 = 112 * kKiB;
constexpr uint32_t kCL1VNew = 144 * kKiB;
constexpr uint32_t kCL1Right = 160 * kKiB;
constexpr uint32_t kCL1APrime = 176 * kKiB;
constexpr uint32_t kCL1End = 184 * kKiB;
static_assert(kCL1End <= 512 * kKiB);

constexpr uint32_t kL0Bank0 = 0;
constexpr uint32_t kL0Bank1 = 32 * kKiB;
constexpr uint32_t kL0C0 = 0;
constexpr uint32_t kL0C1 = 64 * kKiB;
constexpr uint32_t kL0C2 = 128 * kKiB;
constexpr uint32_t kL0C3 = 192 * kKiB;

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_MEMORY_H
