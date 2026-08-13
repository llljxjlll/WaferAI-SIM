#pragma once

#include "dte/coll_config.h"
#include "dte/coll_types.h"

#include <cstddef>
#include <cstdint>
#include <string>

// P7 keeps the semantic collective plan independent from its NoC acceleration
// choices.  This pure planner resolves the two orthogonal profile switches to
// explicit backends, after applying operation applicability and capability
// gates.  It never silently substitutes another backend.

enum class IsaV1ProfileBroadcastBackend : uint8_t {
    NOT_APPLICABLE = 0,
    BYPASS = 1,
    UNICAST = 2,
    MULTICAST = 3
};

enum class IsaV1ProfileReduceBackend : uint8_t {
    NOT_APPLICABLE = 0,
    BYPASS = 1,
    ENDPOINT = 2,
    DCA_OFFLOAD = 3
};

enum class IsaV1ProfileRejectReason : uint8_t {
    NONE = 0,
    MULTICAST_UNAVAILABLE = 1,
    DCA_UNAVAILABLE = 2,
    REDUCE_SCATTER_DCA_UNSUPPORTED = 3
};

struct IsaV1CollectiveProfileCapabilities {
    bool multicast = false;
    bool dca = false;
    // D-11 is deliberately independent from generic DCA availability.  It is
    // false until the multi-destination ReduceScatter DCA path is validated.
    bool reduce_scatter_dca = false;
};

struct IsaV1CollectiveProfileRequest {
    CollOp op = CollOp::P2P;
    size_t group_size = 0;
    NocCollProfile profile = NocCollProfile::BASELINE;
    IsaV1CollectiveProfileCapabilities capabilities;
};

struct IsaV1CollectiveProfileDecision {
    bool accepted = false;
    CollOp op = CollOp::P2P;
    size_t group_size = 0;
    NocCollProfile requested_profile = NocCollProfile::BASELINE;
    IsaV1ProfileBroadcastBackend broadcast_backend =
        IsaV1ProfileBroadcastBackend::NOT_APPLICABLE;
    IsaV1ProfileReduceBackend reduce_backend =
        IsaV1ProfileReduceBackend::NOT_APPLICABLE;
    bool endpoint_reduce_compute = false;
    size_t multicast_tree_count = 0;
    size_t dca_reduce_tree_count = 0;
    bool requires_multicast = false;
    bool requires_dca = false;
    IsaV1ProfileRejectReason reject_reason =
        IsaV1ProfileRejectReason::NONE;
    // Stable key=value representation for production trace integration.
    std::string trace;

    bool operator==(const IsaV1CollectiveProfileDecision &other) const;
};

const char *IsaV1ProfileBroadcastBackendName(
    IsaV1ProfileBroadcastBackend backend);
const char *IsaV1ProfileReduceBackendName(
    IsaV1ProfileReduceBackend backend);
const char *IsaV1ProfileRejectReasonName(IsaV1ProfileRejectReason reason);

// group_size==0 and invalid enum values are malformed contracts and throw
// invalid_argument.  Missing capabilities are a valid planning result:
// accepted is false and reject_reason is exact.  group_size==1 always returns
// accepted BYPASS and does not require acceleration capabilities.
IsaV1CollectiveProfileDecision PlanIsaV1CollectiveProfile(
    const IsaV1CollectiveProfileRequest &request);
