# Copyright (c) 2026 Tianjin University, Ltd.
# CANN Open Software License Agreement Version 2.0.

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE chunk_fwd_ho_fused_def.cpp)
endif()

# CANN collapses the consecutive uppercase letters in ChunkFwdHOFused to "ho"
# when generating the host/proto basename.
add_modules_sources(OPTYPE chunk_fwd_ho_fused ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkFwdHOFused
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)

if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME ChunkFwdHOFused
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
    )
endif()
