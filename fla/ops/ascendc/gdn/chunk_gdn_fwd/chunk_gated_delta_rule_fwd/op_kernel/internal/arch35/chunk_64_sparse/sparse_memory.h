/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_CHUNK64_SPARSE_MEMORY_H
#define GDN_CHUNK64_SPARSE_MEMORY_H

#include <cstdint>

namespace GdnChunk64Sparse {
constexpr uint32_t kKiB = 1024;
constexpr uint32_t kUbBudget = 248 * kKiB;

// Each AIV owns one full 64-row chunk per pair iteration. Raw K/Q and their
// normalized NZ forms use local-pair double buffering. The normalized buffers
// are reused by inverse/QKMask in the opposite phase: preprocess fills slot
// (i+1)&1 while inverse and QKMask consume slot i&1.
constexpr uint32_t kPUbInputSlotBytes = 16 * kKiB;
constexpr uint32_t kPUbK = 0;
constexpr uint32_t kPUbQ = kPUbK + 2 * kPUbInputSlotBytes;
constexpr uint32_t kPUbKHat = kPUbQ + 2 * kPUbInputSlotBytes;
constexpr uint32_t kPUbQHat = kPUbKHat + 2 * kPUbInputSlotBytes;
constexpr uint32_t kPUbKkt = kPUbQHat + 2 * kPUbInputSlotBytes;
constexpr uint32_t kPUbKktBytes = 16 * kKiB;
constexpr uint32_t kPUbQk = kPUbKkt + kPUbKktBytes;
constexpr uint32_t kPUbQkBytes = 16 * kKiB;
constexpr uint32_t kPUbRatio = kPUbQk + kPUbQkBytes;
constexpr uint32_t kPUbRatioSlotBytes = 16 * kKiB;
constexpr uint32_t kPUbGates = kPUbRatio + 2 * kPUbRatioSlotBytes;
// Three gate slots cover the current inverse, next preprocess and the load
// issued for the following preprocess. Each slot holds 64 G/energy/beta/rstd
// values plus 64 reciprocal-energy values.
constexpr uint32_t kPUbGateSlotBytes = 2 * kKiB;
constexpr uint32_t kPUbShift = kPUbGates + 3 * kPUbGateSlotBytes;
constexpr uint32_t kPUbScratch = kPUbShift + 6 * 64 * sizeof(uint32_t);
constexpr uint32_t kPUbEnd = kPUbShift + 2 * kKiB;
constexpr uint32_t kPUbEye = kPUbEnd;
constexpr uint32_t kPUbEndWithEye = kPUbEye + 16 * kKiB;
static_assert(kPUbEndWithEye <= kUbBudget);

// Three full-chunk K/Q L1 slots protect the current KKT/QKT pair from the next
// preprocess. No FP32 inverse or coupling records remain in L1.
constexpr uint32_t kPL1SlotBytes = 16 * kKiB;
constexpr uint32_t kPL1K = 0;
constexpr uint32_t kPL1QHat = kPL1K + 3 * kPL1SlotBytes;
constexpr uint32_t kPL1QSlotBytes = kPL1SlotBytes;
constexpr uint32_t kPL1End = kPL1QHat + 3 * kPL1QSlotBytes;
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
// v_new uses this UB only for right generation; O uses a separate L0C->L1
// Fixpipe. Keeping it independent from ubU_ avoids a next-Vb WAR hazard.
constexpr uint32_t kCUbVNew = 136 * kKiB;
constexpr uint32_t kCUbRight = 144 * kKiB;
constexpr uint32_t kCUbW = 152 * kKiB;
constexpr uint32_t kCUbQ = 160 * kKiB;
constexpr uint32_t kCUbK = 168 * kKiB;
constexpr uint32_t kCUbZero = 176 * kKiB;
constexpr uint32_t kCUbO = 180 * kKiB;
constexpr uint32_t kCUbGates = 188 * kKiB;
constexpr uint32_t kCUbShift = 190 * kKiB;
constexpr uint32_t kCUbV = 200 * kKiB;
constexpr uint32_t kCUbEnd = 208 * kKiB;
static_assert(kCUbEnd <= kUbBudget);

// Consumer L1 uses chunk-parity slots for values whose next-chunk upload can
// overlap the current Cube consumer. H, vNew, right and T keep their existing
// lifetimes; H is already double-buffered.
constexpr uint32_t kCL1Slot16 = 16 * kKiB;
constexpr uint32_t kCL1W = 0;
constexpr uint32_t kCL1U = 32 * kKiB;
constexpr uint32_t kCL1QExp = 80 * kKiB;
constexpr uint32_t kCL1K = 112 * kKiB;
constexpr uint32_t kCL1H0 = 144 * kKiB;
constexpr uint32_t kCL1H1 = 176 * kKiB;
constexpr uint32_t kCL1VNew = 208 * kKiB;
constexpr uint32_t kCL1Right = 224 * kKiB;
constexpr uint32_t kCL1APrime = 240 * kKiB;
constexpr uint32_t kCL1APrimeSlotBytes = 8 * kKiB;
constexpr uint32_t kCL1T = 256 * kKiB;
constexpr uint32_t kCL1End = 264 * kKiB;
static_assert(kCL1End <= 512 * kKiB);

constexpr uint32_t kL0Bank0 = 0;
constexpr uint32_t kL0Bank1 = 32 * kKiB;
constexpr uint32_t kL0C0 = 0;
constexpr uint32_t kL0C1 = 64 * kKiB;
constexpr uint32_t kL0C2 = 128 * kKiB;
constexpr uint32_t kL0C3 = 192 * kKiB;

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_MEMORY_H
