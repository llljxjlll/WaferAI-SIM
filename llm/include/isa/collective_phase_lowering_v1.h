#pragma once

#include "isa/record_lowering.h"
#include "prims/collective_phase_barrier_v1_prim.h"

#include <cstddef>
#include <cstdint>
#include <memory>

// Materializes one canonical internal phase barrier from the validated
// whole-artifact collective action stream.  Only POSTED_BARRIER and
// COMPLETE_BARRIER are accepted.  The returned Prim is caller-owned and does
// not make the artifact executable.
std::unique_ptr<Collective_phase_barrier_v1_prim>
MaterializeIsaV1CollectivePhaseBarrier(
    const IsaV1CollectiveArtifactLowering &lowering,
    uint16_t executing_core, std::size_t action_stream_index);
