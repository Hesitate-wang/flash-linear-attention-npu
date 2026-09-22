/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */

#ifndef CHUNK_FWD_H_O_FUSED_ARCH_H
#define CHUNK_FWD_H_O_FUSED_ARCH_H

#if !defined(__CCE_AICORE__)
#error "ChunkFwdHOFused architecture selection requires __CCE_AICORE__"
#elif __CCE_AICORE__ == 310
#define CHUNK_FWD_HO_ARCH35 1
#elif __CCE_AICORE__ == 220
#define CHUNK_FWD_HO_ARCH_A2 1
#else
#error "ChunkFwdHOFused supports only A2 (__CCE_AICORE__=220) and A5 (__CCE_AICORE__=310)"
#endif

#endif // CHUNK_FWD_H_O_FUSED_ARCH_H
