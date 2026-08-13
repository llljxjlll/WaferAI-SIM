#pragma once

#include "isa/record_lowering.h"
#include "prims/dte_endpoint_prims.h"

#include <cstddef>
#include <cstdint>
#include <memory>

// Materializes one internal child endpoint from the validated whole-artifact
// collective product.  action_stream_index is an index in executing_core's
// canonical action stream; free-standing/keyed ExternalRecords are not
// accepted by this API.
//
// Only POST_RECEIVE and ISSUE_SEND create endpoint Prims.  Wait and barrier
// actions remain orchestration operations.  The returned strict P2P Prim is
// owned by the caller, is never inserted into a process-global table, and
// does not by itself make the artifact executable.
std::unique_ptr<Dte_endpoint_prim_base>
MaterializeIsaV1CollectiveChildEndpoint(
    const IsaV1CollectiveArtifactLowering &lowering,
    uint16_t executing_core, std::size_t action_stream_index);
