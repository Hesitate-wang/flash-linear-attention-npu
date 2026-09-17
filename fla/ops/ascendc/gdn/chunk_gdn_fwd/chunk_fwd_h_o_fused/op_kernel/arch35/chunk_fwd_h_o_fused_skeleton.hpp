/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef CHUNK_FWD_H_O_FUSED_ARCH35_SKELETON_HPP
#define CHUNK_FWD_H_O_FUSED_ARCH35_SKELETON_HPP

namespace GDN::Arch35 {

// Registration-only A5 entry. The host tiling route rejects execution until
// the architecture-specific schedule and computation have been implemented.
__aicore__ inline void RunChunkFwdHOFusedSkeleton(
    const ChunkFwdHOFusedTilingData &tiling)
{
    (void)tiling;
}

} // namespace GDN::Arch35

#endif
