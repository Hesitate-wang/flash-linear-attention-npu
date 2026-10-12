/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef CATLASS_EPILOGUE_GDN_FWD_H_EPILOGUE_POLICIES_MODE2_HPP
#define CATLASS_EPILOGUE_GDN_FWD_H_EPILOGUE_POLICIES_MODE2_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Epilogue {

struct EpilogueAtlasGDNFwdHVnewPreloadSingleSlot {
    using ArchTag = Arch::Ascend950;
};

struct EpilogueAtlasGDNFwdHUpdatePreloadSingleSlot {
    using ArchTag = Arch::Ascend950;
};

} // namespace Catlass::Epilogue

#endif // CATLASS_EPILOGUE_GDN_FWD_H_EPILOGUE_POLICIES_MODE2_HPP
