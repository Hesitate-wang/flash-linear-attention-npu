# Copyright (c) 2026 Tianjin University, Ltd.
# CANN Open Software License Agreement Version 2.0.

add_op_to_compiled_list()

# CANN collapses the consecutive uppercase letters in ChunkFwdHOFused for the
# host/proto basename. Some toolchain versions also use that collapsed basename
# as opFile despite the explicit OpAICoreConfig override, so keep both names
# mapped to the source tree. An undefined opFile directory is silently skipped
# by add_bin_compile_target and produces no AICore .o.
get_filename_component(_chunk_fwd_h_o_fused_op_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
if(NOT EXISTS "${_chunk_fwd_h_o_fused_op_dir}/op_kernel/chunk_fwd_h_o_fused.cpp")
    message(FATAL_ERROR "ChunkFwdHOFused kernel source directory is incomplete: ${_chunk_fwd_h_o_fused_op_dir}")
endif()
set(chunk_fwd_h_o_fused_dir "${_chunk_fwd_h_o_fused_op_dir}" CACHE INTERNAL
    "Source directory for the public ChunkFwdHOFused opFile" FORCE)
set(chunk_fwd_ho_fused_dir "${_chunk_fwd_h_o_fused_op_dir}" CACHE INTERNAL
    "Source directory alias for the CANN-collapsed ChunkFwdHOFused opFile" FORCE)

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
