add_op_to_compiled_list()

# 本算子自包含：kernel/host 不引用其他算子目录的头文件或实现（chunk_64_sparse
# 使用内部副本 sparse_prep_lib.h，串行回退已由组内串行编排取代），因此不再需要
# 把 chunk_gated_delta_rule_fwd_prepare / chunk_fwd_h / chunk_fwd_o 的 op_host
# 拉入过滤构建，也不再需要指向算子目录之外的 -I 路径。

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_gated_delta_rule_fwd_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_gated_delta_rule_fwd ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkGatedDeltaRuleFwd
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            # 临时实验（507015 排查）：屏蔽 chunk_64_sparse 全部 CrossCore 同步，
            # 用于判定超时是否由跨核同步协议导致。输出必然错误（竞态），
            # 实验结束必须删除本行。详见 chunk_64_sparse/HANDOFF.md §9。
            -DGDN_SPARSE_DEBUG_NO_CROSSCORE_SYNC
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/coefficient_generation
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/coefficient_generation/gated_delta_rule_solve_tri
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/operators/chunk_fwd_o/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/operators/recompute_w_u_fwd/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch22/operators/chunk_kkt_solve_tri/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch22/operators/chunk_scaled_dot_kkt/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch22/operators/solve_tri/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch22/operators/chunk_gated_delta_rule_fwd_h/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch22/operators/chunk_fwd_o/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch22/operators/recompute_w_u_fwd/op_kernel
)
