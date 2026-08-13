#pragma once

#include "isa/record_lowering.h"
#include "prims/collective_data_v1_prim.h"

#include <cstddef>
#include <cstdint>
#include <memory>

// Materializes one canonical local-data action from the validated
// whole-artifact collective product.  The action is selected by its core
// stream index rather than accepted as a free-standing ExternalRecord, so a
// public REDUCE_COMPUTE record cannot bypass graph validation.
//
// The returned Prim is owned by the caller and is never inserted into any
// process-global Prim or address table.  This mapping does not make the
// artifact executable.
std::unique_ptr<Collective_data_v1_prim>
MaterializeIsaV1CollectiveDataAction(
    const IsaV1CollectiveArtifactLowering &lowering,
    uint16_t executing_core, std::size_t action_stream_index);
