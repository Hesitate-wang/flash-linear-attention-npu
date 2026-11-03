/**
 * Hardware buffer bases for the fused GDN forward. One LocalTensor per memory
 * level, byte 0 is the hardware base. Tile addresses live in the kernel.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_32_COMMON_H
#define CHUNK_GATED_DELTA_RULE_FWD_32_COMMON_H

#include "kernel_operator.h"

// The on-chip buffer types are defined exactly once for the whole operator in
// op_kernel/internal/gdn_onchip_types.h (global namespace); re-export them
// here so existing GdnFwd32-unqualified uses keep working.
#include "../../gdn_onchip_types.h"

namespace GdnFwd32 {

using ::HardwareInfo;
using ::BufferType;
using ::OnChipBuffer;

using namespace AscendC;

constexpr int kBt = 32;
constexpr int kDim = 128;
constexpr float kEps = 1e-6f;
constexpr int32_t kVl = 64;

constexpr AscendC::FixpipeConfig kFixUbNz2Nd = {AscendC::CO2Layout::ROW_MAJOR, true};
constexpr AscendC::FixpipeConfig kFixUbNz = {AscendC::CO2Layout::NZ, true};

}  // namespace GdnFwd32

#endif
