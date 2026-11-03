/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

#ifndef MEM_H
#define MEM_H

// The on-chip buffer types are defined exactly once for the whole operator in
// op_kernel/internal/gdn_onchip_types.h (global namespace); this header keeps
// its historical include point for the solve_tri sources.
#include "../../../gdn_onchip_types.h"

#endif  // MEM_H
