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

// Two input slots per AIV. Normalized K is uploaded to L1 for Cube; each AIV
// also keeps its own ND half and gate/beta copy for the later scaling pass.
constexpr uint32_t kPUbSlotBytes = 48 * kKiB;
constexpr uint32_t kPUbK = 0;
constexpr uint32_t kPUbV = 8 * kKiB;
constexpr uint32_t kPUbKHat = 16 * kKiB;
constexpr uint32_t kPUbPublic = 24 * kKiB;
constexpr uint32_t kPUbVb = 32 * kKiB;
constexpr uint32_t kPUbKbg = 40 * kKiB;

// The parity owner receives one complete KKT and packs both diagonal solves
// into a single 32x64 FP32 VCS workspace.
constexpr uint32_t kPUbKkt = 2 * kPUbSlotBytes;
constexpr uint32_t kPUbL = kPUbKkt + 16 * kKiB;
constexpr uint32_t kPUbX = kPUbL + 8 * kKiB;
constexpr uint32_t kPUbL10 = kPUbX + 8 * kKiB;
constexpr uint32_t kPUbABf = kPUbL10 + 4 * kKiB;
constexpr uint32_t kPUbZero = kPUbABf + 8 * kKiB;
constexpr uint32_t kPUbGates = kPUbZero + 4 * kKiB;
constexpr uint32_t kPUbShift = kPUbGates + 4 * kKiB;
constexpr uint32_t kPUbEnd = kPUbShift + 2 * kKiB;
static_assert(kPUbEnd <= kUbBudget);

// Two K/Vb/Kbg slots, one complete BF16 inverse, and native-FP32 coupling
// records. No Neumann P/S or identity storage remains.
constexpr uint32_t kPL1SlotBytes = 48 * kKiB;
constexpr uint32_t kPL1K = 0;
constexpr uint32_t kPL1Vb = 16 * kKiB;
constexpr uint32_t kPL1Kbg = 32 * kKiB;
constexpr uint32_t kPL1A = 2 * kPL1SlotBytes;
constexpr uint32_t kPL1InvX0 = kPL1A + 8 * kKiB;
constexpr uint32_t kPL1InvX1 = kPL1InvX0 + 4 * kKiB;
constexpr uint32_t kPL1InvL10 = kPL1InvX1 + 4 * kKiB;
constexpr uint32_t kPL1InvT = kPL1InvL10 + 4 * kKiB;
constexpr uint32_t kPL1End = kPL1InvT + 4 * kKiB;
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
