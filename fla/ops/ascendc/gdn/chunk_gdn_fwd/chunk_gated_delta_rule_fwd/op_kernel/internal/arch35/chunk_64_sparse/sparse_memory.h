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
// T and APrime own a private pair-parity region instead of borrowing the dead
// halves of the normalized K slot. Splitting them off leaves the K/Q slots
// guarded only by the same-generation input event, and moves the
// out-generation drain into BuildInverse, two pairs after the stores were
// issued. Each slot holds T in the first 8 KiB and APrime in the second.
constexpr uint32_t kPUbOut = kPUbEye + 16 * kKiB;
constexpr uint32_t kPUbOutSlotBytes = 16 * kKiB;
constexpr uint32_t kPUbEndFinal = kPUbOut + 2 * kPUbOutSlotBytes;
static_assert(kPUbEndFinal <= kUbBudget);

// Six full-chunk K/Q L1 slots: each lane of a pair owns a private slot so the
// odd AIV's upload never overwrites the even chunk still feeding KKT/QK. A
// slot is reused three pairs later, after the pair-2 output event confirms the
// Cube side retired all reads of that generation.
constexpr uint32_t kPL1SlotBytes = 16 * kKiB;
constexpr uint32_t kPL1SlotCount = 6;
constexpr uint32_t kPL1K = 0;
constexpr uint32_t kPL1QHat = kPL1K + kPL1SlotCount * kPL1SlotBytes;
constexpr uint32_t kPL1QSlotBytes = kPL1SlotBytes;
constexpr uint32_t kPL1End = kPL1QHat + kPL1SlotCount * kPL1QSlotBytes;
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
constexpr uint32_t kCUbGateSlotBytes = 2 * kKiB;
constexpr uint32_t kCUbShift = kCUbGates + 3 * kCUbGateSlotBytes;
// V is double-buffered by chunk parity: iteration i issues the chunk i+2 load
// while ComputeNextVb consumes the chunk i+1 buffer.
constexpr uint32_t kCUbV = 200 * kKiB;
constexpr uint32_t kCUbVSlotBytes = 8 * kKiB;
static_assert(kCUbShift + 6 * 64 * sizeof(uint32_t) <= kCUbV);
constexpr uint32_t kCUbEnd = kCUbV + 2 * kCUbVSlotBytes;
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
// T keeps one slot per pair lane: the even chunk loads both tiles back-to-back
// after a single pair-granular mode-0 join, like the normalized K/Q loads.
constexpr uint32_t kCL1T = 256 * kKiB;
constexpr uint32_t kCL1TSlotBytes = 8 * kKiB;
constexpr uint32_t kCL1End = kCL1T + 2 * kCL1TSlotBytes;
static_assert(kCL1End <= 512 * kKiB);

// L0A: two 32 KiB banks. Bank0 carries T, the corrected W (T@W result, also
// the W@S left operand) and the D left operand; bank1 carries QExp in its first
// 16 KiB and APrime in the second. QExp and APrime coexist, so the QS operand
// load never waits for the T@Vb mmad and never invalidates APrime.
constexpr uint32_t kL0Bank0 = 0;
constexpr uint32_t kL0Bank1 = 32 * kKiB;
constexpr uint32_t kL0APrimeOffset = static_cast<uint32_t>(kChunk * kDim); // elements inside bank1

// L0B: four 16 KiB slots so the T@W / T@Vb operands (W, Vb), the two H halves
// (consumed by both QS and W@S) and the D operands never share a slot. dB/dB2/
// vNew reuse a slot only after its previous mmad retired.
constexpr uint32_t kL0BSlotBytes = 16 * kKiB;
constexpr uint32_t kL0B0 = 0;                  // W (T@W B), then dB (D-C1 B)
constexpr uint32_t kL0B1 = kL0BSlotBytes;      // Vb (T@Vb B), then dB2 (D-C2 B)
constexpr uint32_t kL0B2 = 2 * kL0BSlotBytes;  // H first half (QS C1, W@S C1)
constexpr uint32_t kL0B3 = 3 * kL0BSlotBytes;  // H second half (QS C2, W@S C2), then vNew
static_assert(kL0B3 + kL0BSlotBytes <= 64 * kKiB);

// L0C: 256 KiB repacked. QS/O keep one 32 KiB accumulator per chunk parity so
// QS never waits for the previous chunk's O Fixpipe. kL0C0/kL0C1 are also read
// by the Producer role, which uses only 16 KiB of each; the new spacing keeps
// its two regions disjoint (0-16 KiB and 32-48 KiB).
constexpr uint32_t kL0C0 = 0;                // T@Vb + W@S accumulator (32 KiB)
constexpr uint32_t kL0C1 = 32 * kKiB;        // D halves (2 x 32 KiB)
constexpr uint32_t kL0C2 = 96 * kKiB;        // T@W scratch (32 KiB)
constexpr uint32_t kL0CQS0 = 128 * kKiB;     // QS/O accumulator, even chunks
constexpr uint32_t kL0CQS1 = 160 * kKiB;     // QS/O accumulator, odd chunks
static_assert(kL0C2 - kL0C1 >= 64 * kKiB);   // D keeps two 128x64 panels
static_assert(kL0CQS1 + 32 * kKiB <= 256 * kKiB);

} // namespace GdnChunk64Sparse

#endif // GDN_CHUNK64_SPARSE_MEMORY_H
