#include "dte/coll_profile_v1.h"

#include <sstream>
#include <stdexcept>
#include <tuple>

namespace {

const char *OpName(CollOp op) {
    switch (op) {
    case CollOp::P2P: return "p2p";
    case CollOp::SCATTER: return "scatter";
    case CollOp::GATHER: return "gather";
    case CollOp::BROADCAST: return "broadcast";
    case CollOp::ALLTOALL: return "alltoall";
    case CollOp::ALLGATHER: return "allgather";
    case CollOp::REDUCE: return "reduce";
    case CollOp::REDUCESCATTER: return "reducescatter";
    case CollOp::ALLREDUCE: return "allreduce";
    }
    throw std::invalid_argument("invalid ISA-v1 collective profile op");
}

const char *ProfileName(NocCollProfile profile) {
    switch (profile) {
    case NocCollProfile::BASELINE: return "baseline";
    case NocCollProfile::BROADCAST_ONLY: return "broadcast_only";
    case NocCollProfile::REDUCE_ONLY: return "reduce_only";
    case NocCollProfile::REDUCE_BROADCAST:
        return "reduce_broadcast";
    }
    throw std::invalid_argument("invalid ISA-v1 collective profile");
}

bool ProfileRequestsMulticast(NocCollProfile profile) {
    switch (profile) {
    case NocCollProfile::BASELINE:
    case NocCollProfile::REDUCE_ONLY: return false;
    case NocCollProfile::BROADCAST_ONLY:
    case NocCollProfile::REDUCE_BROADCAST: return true;
    }
    throw std::invalid_argument("invalid ISA-v1 collective profile");
}

bool ProfileRequestsDca(NocCollProfile profile) {
    switch (profile) {
    case NocCollProfile::BASELINE:
    case NocCollProfile::BROADCAST_ONLY: return false;
    case NocCollProfile::REDUCE_ONLY:
    case NocCollProfile::REDUCE_BROADCAST: return true;
    }
    throw std::invalid_argument("invalid ISA-v1 collective profile");
}

bool UsesBroadcastSide(CollOp op) {
    switch (op) {
    case CollOp::BROADCAST:
    case CollOp::ALLGATHER:
    case CollOp::ALLREDUCE: return true;
    case CollOp::P2P:
    case CollOp::SCATTER:
    case CollOp::GATHER:
    case CollOp::ALLTOALL:
    case CollOp::REDUCE:
    case CollOp::REDUCESCATTER: return false;
    }
    throw std::invalid_argument("invalid ISA-v1 collective profile op");
}

bool UsesReduceSide(CollOp op) {
    switch (op) {
    case CollOp::REDUCE:
    case CollOp::REDUCESCATTER:
    case CollOp::ALLREDUCE: return true;
    case CollOp::P2P:
    case CollOp::SCATTER:
    case CollOp::GATHER:
    case CollOp::BROADCAST:
    case CollOp::ALLTOALL:
    case CollOp::ALLGATHER: return false;
    }
    throw std::invalid_argument("invalid ISA-v1 collective profile op");
}

size_t MulticastTreeCount(CollOp op, size_t group_size) {
    switch (op) {
    case CollOp::BROADCAST: return 1;
    case CollOp::ALLGATHER:
    case CollOp::ALLREDUCE: return group_size;
    default: return 0;
    }
}

size_t DcaTreeCount(CollOp op, size_t group_size) {
    switch (op) {
    case CollOp::REDUCE: return 1;
    case CollOp::REDUCESCATTER:
    case CollOp::ALLREDUCE: return group_size;
    default: return 0;
    }
}

std::string Trace(const IsaV1CollectiveProfileDecision &decision,
                  bool group_size_one_bypass) {
    std::ostringstream out;
    out << "profile=" << ProfileName(decision.requested_profile)
        << " op=" << OpName(decision.op)
        << " group_size=" << decision.group_size
        << " status=" << (decision.accepted ? "accepted" : "rejected")
        << " broadcast="
        << IsaV1ProfileBroadcastBackendName(decision.broadcast_backend)
        << " reduce="
        << IsaV1ProfileReduceBackendName(decision.reduce_backend)
        << " endpoint_reduce_compute="
        << (decision.endpoint_reduce_compute ? "true" : "false")
        << " multicast_trees=" << decision.multicast_tree_count
        << " dca_trees=" << decision.dca_reduce_tree_count
        << " requires_multicast="
        << (decision.requires_multicast ? "true" : "false")
        << " requires_dca="
        << (decision.requires_dca ? "true" : "false")
        << " reason=";
    if (group_size_one_bypass)
        out << "group_size_one_bypass";
    else if (decision.reject_reason == IsaV1ProfileRejectReason::NONE)
        out << "accepted";
    else
        out << IsaV1ProfileRejectReasonName(decision.reject_reason);
    return out.str();
}

} // namespace

bool IsaV1CollectiveProfileDecision::operator==(
    const IsaV1CollectiveProfileDecision &other) const {
    return std::tie(accepted, op, group_size, requested_profile,
                    broadcast_backend, reduce_backend,
                    endpoint_reduce_compute, multicast_tree_count,
                    dca_reduce_tree_count, requires_multicast, requires_dca,
                    reject_reason, trace) ==
           std::tie(other.accepted, other.op, other.group_size,
                    other.requested_profile, other.broadcast_backend,
                    other.reduce_backend, other.endpoint_reduce_compute,
                    other.multicast_tree_count,
                    other.dca_reduce_tree_count,
                    other.requires_multicast, other.requires_dca,
                    other.reject_reason, other.trace);
}

const char *IsaV1ProfileBroadcastBackendName(
    IsaV1ProfileBroadcastBackend backend) {
    switch (backend) {
    case IsaV1ProfileBroadcastBackend::NOT_APPLICABLE:
        return "not_applicable";
    case IsaV1ProfileBroadcastBackend::BYPASS: return "bypass";
    case IsaV1ProfileBroadcastBackend::UNICAST: return "unicast";
    case IsaV1ProfileBroadcastBackend::MULTICAST: return "multicast";
    }
    throw std::invalid_argument(
        "invalid ISA-v1 collective broadcast backend");
}

const char *IsaV1ProfileReduceBackendName(
    IsaV1ProfileReduceBackend backend) {
    switch (backend) {
    case IsaV1ProfileReduceBackend::NOT_APPLICABLE:
        return "not_applicable";
    case IsaV1ProfileReduceBackend::BYPASS: return "bypass";
    case IsaV1ProfileReduceBackend::ENDPOINT: return "endpoint";
    case IsaV1ProfileReduceBackend::DCA_OFFLOAD: return "dca_offload";
    }
    throw std::invalid_argument("invalid ISA-v1 collective reduce backend");
}

const char *IsaV1ProfileRejectReasonName(
    IsaV1ProfileRejectReason reason) {
    switch (reason) {
    case IsaV1ProfileRejectReason::NONE: return "none";
    case IsaV1ProfileRejectReason::MULTICAST_UNAVAILABLE:
        return "multicast_unavailable";
    case IsaV1ProfileRejectReason::DCA_UNAVAILABLE:
        return "dca_unavailable";
    case IsaV1ProfileRejectReason::REDUCE_SCATTER_DCA_UNSUPPORTED:
        return "reduce_scatter_dca_unsupported";
    }
    throw std::invalid_argument(
        "invalid ISA-v1 collective profile reject reason");
}

IsaV1CollectiveProfileDecision PlanIsaV1CollectiveProfile(
    const IsaV1CollectiveProfileRequest &request) {
    // Validate both enums before the N=1 fast path so malformed contracts
    // cannot hide behind bypass semantics.
    static_cast<void>(OpName(request.op));
    static_cast<void>(ProfileName(request.profile));
    if (request.group_size == 0)
        throw std::invalid_argument(
            "ISA-v1 collective profile group_size must be positive");

    IsaV1CollectiveProfileDecision result;
    result.op = request.op;
    result.group_size = request.group_size;
    result.requested_profile = request.profile;

    if (request.group_size == 1) {
        result.accepted = true;
        result.broadcast_backend = IsaV1ProfileBroadcastBackend::BYPASS;
        result.reduce_backend = IsaV1ProfileReduceBackend::BYPASS;
        result.trace = Trace(result, true);
        return result;
    }

    const bool use_broadcast = UsesBroadcastSide(request.op);
    const bool use_reduce = UsesReduceSide(request.op);

    if (use_broadcast) {
        result.broadcast_backend = ProfileRequestsMulticast(request.profile)
                                       ? IsaV1ProfileBroadcastBackend::MULTICAST
                                       : IsaV1ProfileBroadcastBackend::UNICAST;
        result.requires_multicast =
            result.broadcast_backend ==
            IsaV1ProfileBroadcastBackend::MULTICAST;
        if (result.requires_multicast)
            result.multicast_tree_count =
                MulticastTreeCount(request.op, request.group_size);
    }

    if (use_reduce) {
        result.reduce_backend = ProfileRequestsDca(request.profile)
                                    ? IsaV1ProfileReduceBackend::DCA_OFFLOAD
                                    : IsaV1ProfileReduceBackend::ENDPOINT;
        result.requires_dca =
            result.reduce_backend == IsaV1ProfileReduceBackend::DCA_OFFLOAD;
        result.endpoint_reduce_compute =
            result.reduce_backend == IsaV1ProfileReduceBackend::ENDPOINT;
        if (result.requires_dca)
            result.dca_reduce_tree_count =
                DcaTreeCount(request.op, request.group_size);
    }

    // Stable rejection precedence: multicast capability, the D-11
    // ReduceScatter-specific gate, then generic DCA capability.  This keeps
    // combined-profile diagnostics deterministic and makes a closed D-11
    // gate visible even when generic DCA is also absent.
    if (result.requires_multicast && !request.capabilities.multicast) {
        result.reject_reason =
            IsaV1ProfileRejectReason::MULTICAST_UNAVAILABLE;
    } else if (request.op == CollOp::REDUCESCATTER &&
               result.requires_dca &&
               !request.capabilities.reduce_scatter_dca) {
        result.reject_reason =
            IsaV1ProfileRejectReason::REDUCE_SCATTER_DCA_UNSUPPORTED;
    } else if (result.requires_dca && !request.capabilities.dca) {
        result.reject_reason = IsaV1ProfileRejectReason::DCA_UNAVAILABLE;
    }

    result.accepted =
        result.reject_reason == IsaV1ProfileRejectReason::NONE;
    result.trace = Trace(result, false);
    return result;
}
