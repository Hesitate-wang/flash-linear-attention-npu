# Copyright (c) 2026 Tianjin University, Ltd.
# CANN Open Software License Agreement Version 2.0.

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE chunk_fwd_h_o_fused_def.cpp)
endif()

add_modules_sources(OPTYPE chunk_fwd_h_o_fused ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkFwdHOFused
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)
