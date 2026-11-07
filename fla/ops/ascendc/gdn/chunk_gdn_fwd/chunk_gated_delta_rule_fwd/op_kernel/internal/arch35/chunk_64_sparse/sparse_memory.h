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

// Producer double-buffer layout, per AIV. Each slot owns all Vector inputs and
// the KKT/Neumann input band. W/U no longer consume UB: AIC writes them to GM.
constexpr uint32_t kPUbSlotBytes = 68 * kKiB;
constexpr uint32_t kPUbK = 0;
constexpr uint32_t kPUbV = 8 * kKiB;
constexpr uint32_t kPUbKHat = 16 * kKiB;
constexpr uint32_t kPUbPublic = 24 * kKiB;
constexpr uint32_t kPUbVb = 32 * kKiB;
constexpr uint32_t kPUbKbg = 40 * kKiB;
constexpr uint32_t kPUbKkt = 48 * kKiB;
constexpr uint32_t kPUbP = 56 * kKiB;
constexpr uint32_t kPUbS = 60 * kKiB;
constexpr uint32_t kPUbGates = 64 * kKiB;
constexpr uint32_t kPUbShift = 2 * kPUbSlotBytes;
constexpr uint32_t kPUbEnd = kPUbShift + 2 * kKiB;
static_assert(kPUbEnd <= kUbBudget);

// Two full producer slots. P/S are complete 64x64 BF16 matrices in L1; each
// AIV writes one 32-row band. Neumann reuses the current slot throughout.
constexpr uint32_t kPL1SlotBytes = 64 * kKiB;
constexpr uint32_t kPL1K = 0;
constexpr uint32_t kPL1Vb = 16 * kKiB;
constexpr uint32_t kPL1Kbg = 32 * kKiB;
constexpr uint32_t kPL1P = 48 * kKiB;
constexpr uint32_t kPL1S = 56 * kKiB;
constexpr uint32_t kPL1Identity = 2 * kPL1SlotBytes;
constexpr uint32_t kPL1End = kPL1Identity + 8 * kKiB;
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
