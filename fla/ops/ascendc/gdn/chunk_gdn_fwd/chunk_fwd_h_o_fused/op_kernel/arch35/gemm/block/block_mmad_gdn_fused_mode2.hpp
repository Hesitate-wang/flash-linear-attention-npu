/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef CATLASS_GDN_FUSED_MODE2_MMAD_HPP
#define CATLASS_GDN_FUSED_MODE2_MMAD_HPP

#include <type_traits>
#include "catlass/gemm/block/block_mmad.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_preload.hpp"
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"

namespace Catlass::Gemm::Block {

// Vec has already populated the bound L1 operand before the driver's ready wait.
struct GdnFusedMode2KeepL1 {
    template<class Dst, class Src>
    CATLASS_DEVICE void operator()(Dst &, Src &) {}
};

template<class DataT, class LayoutA, class LayoutB, bool ToUB, bool ExternalA, bool ExternalB>
struct GdnFusedMode2TileCopy : std::conditional_t<ToUB,
    Tile::PackedTileCopyTlaToUB<Arch::Ascend950, DataT, LayoutA, DataT, LayoutB,
        float, layout::RowMajor, void, Tile::CopyL0CToUBMode::SPLIT_M>,
    Tile::PackedTileCopyTla<Arch::Ascend950, DataT, LayoutA, DataT, LayoutB,
        float, layout::RowMajor>> {
    using Base = std::conditional_t<ToUB,
        Tile::PackedTileCopyTlaToUB<Arch::Ascend950, DataT, LayoutA, DataT, LayoutB,
            float, layout::RowMajor, void, Tile::CopyL0CToUBMode::SPLIT_M>,
        Tile::PackedTileCopyTla<Arch::Ascend950, DataT, LayoutA, DataT, LayoutB,
            float, layout::RowMajor>>;
    template<class Tensor>
    using CopyGmToL1A = std::conditional_t<ExternalA, GdnFusedMode2KeepL1,
        typename Base::template CopyGmToL1A<Tensor>>;
    template<class Tensor>
    using CopyGmToL1B = std::conditional_t<ExternalB, GdnFusedMode2KeepL1,
        typename Base::template CopyGmToL1B<Tensor>>;
};

template<class DataT, class LayoutA, class LayoutB, uint32_t M, uint32_t N, uint32_t K,
         bool ToUB, bool ExternalA = false, bool ExternalB = false>
class BlockMmadGdnFusedMode2 : public BlockMmadTla<
    MmadPingpongTlaPreload<Arch::Ascend950, false, false, 1, false, 1, 1, 1, 1>,
    tla::Shape<tla::Int<M>, tla::Int<N>, tla::Int<K>>,
    tla::Shape<tla::Int<M>, tla::Int<N>, tla::Int<K>>,
    DataT, DataT, float, void,
    GdnFusedMode2TileCopy<DataT, LayoutA, LayoutB, ToUB, ExternalA, ExternalB>> {
    using Copy = GdnFusedMode2TileCopy<DataT, LayoutA, LayoutB, ToUB, ExternalA, ExternalB>;
    using Shape = tla::Shape<tla::Int<M>, tla::Int<N>, tla::Int<K>>;
    using Base = BlockMmadTla<
        MmadPingpongTlaPreload<Arch::Ascend950, false, false, 1, false, 1, 1, 1, 1>,
        Shape, Shape, DataT, DataT, float, void, Copy>;
    static_assert(sizeof(DataT) == 2, "Mode2 Cube operands must be FP16/BF16");
    // All shapes share the same event IDs, so their A/B scratch slots must
    // also share fixed, disjoint addresses across successive Mmad calls.
    static constexpr uint32_t L1_A_OFFSET = 0U;
    static constexpr uint32_t L1_B_OFFSET = 16U * 1024U;
    static constexpr uint32_t L1_SCRATCH_END = 64U * 1024U;
    static_assert(L1_A_OFFSET + M * K * sizeof(DataT) <= L1_B_OFFSET,
                  "Mode2 A scratch exceeds its fixed L1 slot");
    static_assert(L1_B_OFFSET + K * N * sizeof(DataT) <= L1_SCRATCH_END,
                  "Mode2 B scratch overlaps persistent L1 operands");
    static_assert(!ExternalA || std::is_same_v<typename Copy::LayoutTagL1A, layout::zN>,
                  "Vec L1 A must match CopyRowsToL1Zn");
    static_assert(!ExternalB || std::is_same_v<typename Copy::LayoutTagL1B, layout::zN>,
                  "Vec L1 B must match CopyRowsToL1Zn");
public:
    CATLASS_DEVICE explicit BlockMmadGdnFusedMode2(Arch::Resource<Arch::Ascend950> &resource,
        AscendC::LocalTensor<DataT> externalA = {}, AscendC::LocalTensor<DataT> externalB = {})
        : Base(resource)
    {
        this->l1ATensorList[0] = resource.l1Buf.template GetBufferByByte<DataT>(L1_A_OFFSET);
        this->l1BTensorList[0] = resource.l1Buf.template GetBufferByByte<DataT>(L1_B_OFFSET);
        if constexpr (ExternalA) this->l1ATensorList[0] = externalA;
        if constexpr (ExternalB) this->l1BTensorList[0] = externalB;
    }
    using Base::operator();
};

} // namespace Catlass::Gemm::Block
#endif
