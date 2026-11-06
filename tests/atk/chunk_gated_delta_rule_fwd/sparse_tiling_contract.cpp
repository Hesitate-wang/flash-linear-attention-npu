#include "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_host/op_tiling/arch35/chunk_64_sparse/chunk_gated_delta_rule_fwd_sparse_tiling.h"

#include "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch35/chunk_64_sparse/sparse_fused_tiling.h"
#include <type_traits>
#include <cstddef>
#include <cassert>
#include <cstdint>
#include <limits>

int main()
{
    using namespace optiling;
    static_assert(std::is_standard_layout<GdnChunk64Sparse::FusedTilingData>::value);
    static_assert(std::is_trivially_copyable<GdnChunk64Sparse::FusedTilingData>::value);
    static_assert(sizeof(GdnChunk64Sparse::FusedTilingData) % sizeof(uint64_t) == 0);
    static_assert(offsetof(GdnChunk64Sparse::FusedTilingData, batch) == 0);
    static_assert(GdnChunk64Sparse::kSparseInverseBytesPerTile == 3 * 32 * 32 * sizeof(float));
    assert(IsChunkGatedDeltaRuleFwdSparseLowParallelism(1, 8, 16));
    assert(!IsChunkGatedDeltaRuleFwdSparseLowParallelism(1, 9, 16));
    assert(!IsChunkGatedDeltaRuleFwdSparseLowParallelism(0, 1, 16));
    assert(!IsChunkGatedDeltaRuleFwdSparseLowParallelism(
        std::numeric_limits<uint64_t>::max(), 2, 16));

    ChunkGatedDeltaRuleFwdSparsePlan plan{};
    ChunkGatedDeltaRuleFwdSparseWorkspace workspace{};
    assert(BuildChunkGatedDeltaRuleFwdSparseWorkspace(
        16, 2, 65, 4, 2, 4, 128, 2, 512, &plan, &workspace));
    assert(plan.launchGroups == 16);
    assert(plan.hvPerHk == 2);
    assert(plan.prepareOffset == 0);
    assert(plan.hOffset >= workspace.prepareBytes);
    assert(plan.oOffset >= plan.hOffset + workspace.hBytes);
    assert(plan.normalizedOffset >= plan.oOffset + workspace.oBytes);
    assert(plan.cumsumOffset >= plan.normalizedOffset + workspace.normalizedBytes);
    assert(workspace.cumsumBytes == 2ULL * 4 * 65 * sizeof(float));
    assert(plan.stateOffset >= plan.cumsumOffset + workspace.cumsumBytes);
    assert(plan.outputOffset >= plan.stateOffset + workspace.stateBytes);
    assert(plan.workspaceBytes == workspace.totalBytes);
    assert(workspace.prepareBytes == 16ULL * 64 * 1024);
    assert(workspace.hBytes == 16ULL * 704 * 1024 + 32ULL * 1024 * 1024);
    assert(workspace.oBytes == 16ULL * 64 * 1024);

    ChunkGatedDeltaRuleFwdSparsePlan invalidPlan{};
    ChunkGatedDeltaRuleFwdSparseWorkspace invalidWorkspace{};
    assert(!BuildChunkGatedDeltaRuleFwdSparseWorkspace(
        0, 2, 65, 4, 2, 4, 128, 2, 512, &invalidPlan, &invalidWorkspace));
    assert(!BuildChunkGatedDeltaRuleFwdSparseWorkspace(
        16, 2, 65, 4, 3, 4, 128, 2, 512, &invalidPlan, &invalidWorkspace));
    assert(!BuildChunkGatedDeltaRuleFwdSparseWorkspace(
        16, 2, 65, std::numeric_limits<uint64_t>::max(), 2, 4,
        128, 2, 512, &invalidPlan, &invalidWorkspace));

    uint64_t cursor = std::numeric_limits<uint64_t>::max() - 3;
    uint64_t offset = 0;
    assert(!PlaceSparseWorkspaceRegion(8, 1, &cursor, &offset));
    assert(!AlignSparseWorkspace(1, 0, &offset));

    static_assert(GdnChunk64Sparse::kSparseScheduleSerial !=
                  GdnChunk64Sparse::kSparseSchedulePipeline);
    // The role barrier must stay clear of the stage-local 0..6 ids and of the
    // SyncAll window 11..14.
    static_assert(GdnChunk64Sparse::kSparseBarrierFlagId == 7);

    ChunkGatedDeltaRuleFwdPipelinePlan pipe{};
    assert(BuildChunkGatedDeltaRuleFwdPipelinePlan(2, 4, 130, 2, 6, false, 16, 512, &pipe));
    assert(pipe.chainCount == 8);
    assert(pipe.producerGroups == 8);
    assert(pipe.consumerGroups == 8);
    assert(pipe.maxChunks == 3);
    assert(pipe.workspaceOffset == 0);
    assert(pipe.workspaceBytes == kChunkGatedDeltaRuleFwdPipelineMinWorkspace);
    assert(pipe.totalWorkspaceBytes == pipe.workspaceOffset + pipe.workspaceBytes);

    // Chains above AIC/2 stay on the serial baseline.
    ChunkGatedDeltaRuleFwdPipelinePlan rejected{};
    assert(!BuildChunkGatedDeltaRuleFwdPipelinePlan(3, 4, 130, 3, 6, false, 16, 512, &rejected));
    assert(!BuildChunkGatedDeltaRuleFwdPipelinePlan(2, 4, 130, 2, 6, false, 16, 0, &rejected));
    assert(!BuildChunkGatedDeltaRuleFwdPipelinePlan(2, 4, 130, 2, 6, false, 16, 512, nullptr));
    assert(!BuildChunkGatedDeltaRuleFwdPipelinePlan(0, 4, 130, 0, 6, false, 16, 512, &rejected));
    assert(!BuildChunkGatedDeltaRuleFwdPipelinePlan(2, 0, 130, 2, 6, false, 16, 512, &rejected));

    // Varlen bounds chains by the sequence count and chunks by chunk_indices.
    ChunkGatedDeltaRuleFwdPipelinePlan varlen{};
    assert(BuildChunkGatedDeltaRuleFwdPipelinePlan(1, 4, 0, 2, 5, true, 16, 512, &varlen));
    assert(varlen.chainCount == 8);
    assert(varlen.maxChunks == 5);
    assert(!BuildChunkGatedDeltaRuleFwdPipelinePlan(1, 4, 0, 3, 5, true, 16, 512, &rejected));
    return 0;
}
