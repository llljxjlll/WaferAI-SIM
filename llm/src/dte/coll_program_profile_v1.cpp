#include "dte/coll_program_profile_v1.h"

#include "dte/p2p_payload.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace {

size_t CheckedAdd(size_t left, size_t right, const char *message) {
    if (right > std::numeric_limits<size_t>::max() - left)
        throw std::overflow_error(message);
    return left + right;
}

size_t CheckedMultiply(size_t left, size_t right, const char *message) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
        throw std::overflow_error(message);
    return left * right;
}

void ReserveBounded(size_t &total, size_t additional, size_t limit,
                    const char *message) {
    const size_t candidate = CheckedAdd(total, additional, message);
    if (candidate > limit) throw std::overflow_error(message);
    total = candidate;
}

size_t PotentialConflictEdges(size_t tree_count) {
    if (tree_count < 2) return 0;
    return CheckedMultiply(tree_count, tree_count - 1,
                           "ISA-v1 profile conflict edge count overflows") /
           2;
}

size_t EstimateTreeEntryCount(const IsaV1MeshShape &shape, uint16_t root,
                              const std::vector<uint16_t> &group) {
    if (group.size() == 1) return 0;
    std::set<uint16_t> routers;
    for (uint16_t target : group) {
        if (target == root) continue;
        const auto path = BuildIsaV1XFirstPath(shape, root, target);
        for (const IsaV1RouterOutput &resource : path)
            routers.insert(resource.router_id);
    }
    return routers.size();
}

std::pair<NocCollBroadcastBackend, NocCollReduceBackend> ExpectedBackends(
    NocCollProfile profile) {
    switch (profile) {
    case NocCollProfile::BASELINE:
        return {NocCollBroadcastBackend::UNICAST,
                NocCollReduceBackend::ENDPOINT};
    case NocCollProfile::BROADCAST_ONLY:
        return {NocCollBroadcastBackend::MULTICAST,
                NocCollReduceBackend::ENDPOINT};
    case NocCollProfile::REDUCE_ONLY:
        return {NocCollBroadcastBackend::UNICAST,
                NocCollReduceBackend::DCA_OFFLOAD};
    case NocCollProfile::REDUCE_BROADCAST:
        return {NocCollBroadcastBackend::MULTICAST,
                NocCollReduceBackend::DCA_OFFLOAD};
    }
    throw std::invalid_argument("invalid ISA-v1 production profile");
}

void ValidateConfig(const IsaV1CollectiveProfileImageConfig &config) {
    const auto expected = ExpectedBackends(config.noc.profile);
    if (config.noc.broadcast_backend != expected.first ||
        config.noc.reduce_backend != expected.second)
        throw std::invalid_argument(
            "ISA-v1 named profile/backend configuration mismatch");
    if (!config.noc.enabled && config.noc.profile != NocCollProfile::BASELINE)
        throw std::invalid_argument(
            "ISA-v1 disabled collective fabric cannot request acceleration");
    if (config.noc.reduce_backend ==
        NocCollReduceBackend::LEGACY_ROUTER_ALU)
        throw std::invalid_argument(
            "ISA-v1 program image forbids legacy Router ALU fallback");
    if (config.max_trees_per_batch == 0 ||
        (config.max_trees_per_batch != static_cast<size_t>(-1) &&
         config.max_trees_per_batch > kNocCollMaxTreesPerBatch) ||
        config.entries_per_router == 0 || config.max_profile_plans == 0 ||
        config.max_total_trees == 0 ||
        config.max_total_tree_entries == 0 ||
        config.max_total_conflict_edges == 0 ||
        config.max_total_batches == 0 || config.max_derived_bytes == 0)
        throw std::invalid_argument(
            "ISA-v1 profile tree capacities must be positive");
    for (const auto &occupied : config.occupied_entries_by_router)
        if (occupied.second > config.entries_per_router)
            throw std::invalid_argument(
                "ISA-v1 existing tree occupancy exceeds Router capacity");
    // Validate the shape even when baseline builds no trees.
    if (config.mesh.grid_x == 0 || config.mesh.grid_y == 0 ||
        config.mesh.die_count == 0)
        throw std::invalid_argument(
            "ISA-v1 profile image mesh dimensions must be positive");
}

bool UsesMulticast(const IsaV1CollectiveProfileDecision &decision) {
    return decision.broadcast_backend ==
           IsaV1ProfileBroadcastBackend::MULTICAST;
}

bool UsesDca(const IsaV1CollectiveProfileDecision &decision) {
    return decision.reduce_backend ==
           IsaV1ProfileReduceBackend::DCA_OFFLOAD;
}

bool GroupSpansDies(const IsaV1CollectivePlan &plan,
                    const IsaV1MeshShape &mesh) {
    if (plan.group.empty())
        throw std::invalid_argument(
            "ISA-v1 accelerated collective group cannot be empty");
    const uint64_t cores_per_die = CheckedMultiply(
        mesh.grid_x, mesh.grid_y,
        "ISA-v1 mesh cores-per-die overflows");
    const uint64_t first_die = plan.group.front() / cores_per_die;
    return std::any_of(
        plan.group.begin(), plan.group.end(),
        [cores_per_die, first_die](uint16_t core) {
            return core / cores_per_die != first_die;
        });
}

std::vector<uint16_t> Roots(const IsaV1CollectivePlan &plan,
                            bool multicast, bool dca) {
    const bool every_rank =
        (multicast && (plan.op == CollOp::ALLGATHER ||
                       plan.op == CollOp::ALLREDUCE)) ||
        (dca && (plan.op == CollOp::REDUCESCATTER ||
                 plan.op == CollOp::ALLREDUCE));
    if (every_rank) {
        std::vector<uint16_t> result(plan.group.size());
        for (size_t rank = 0; rank < result.size(); ++rank)
            result[rank] = static_cast<uint16_t>(rank);
        return result;
    }
    return {plan.root_rank};
}

bool RootUsesMulticast(const IsaV1CollectivePlan &plan,
                       const IsaV1CollectiveProfileDecision &decision,
                       uint16_t root_rank) {
    if (!UsesMulticast(decision)) return false;
    if (plan.op == CollOp::BROADCAST) return root_rank == plan.root_rank;
    return plan.op == CollOp::ALLGATHER || plan.op == CollOp::ALLREDUCE;
}

bool RootUsesDca(const IsaV1CollectivePlan &plan,
                 const IsaV1CollectiveProfileDecision &decision,
                 uint16_t root_rank) {
    if (!UsesDca(decision)) return false;
    if (plan.op == CollOp::REDUCE) return root_rank == plan.root_rank;
    return plan.op == CollOp::REDUCESCATTER ||
           plan.op == CollOp::ALLREDUCE;
}

void ValidateDcaDataContract(const IsaV1CollectivePlan &plan,
                             const NocCollectiveConfig &config) {
    if (config.dca.value_mode != NocCollValueMode::INTEGER_EXACT)
        throw std::invalid_argument(
            "ISA-v1 DCA program image requires integer_exact byte mode");
    if (plan.reduce_targets.empty())
        throw std::invalid_argument(
            "ISA-v1 DCA plan has no byte-visible reduce targets");
    constexpr uint64_t kWire24 = (1ULL << 24) - 1;
    constexpr uint64_t kWire12 = (1ULL << 12) - 1;
    constexpr uint64_t kWire16 = (1ULL << 16) - 1;
    if (plan.key.group_id > kWire24 ||
        plan.key.collective_id > kWire24 || plan.key.epoch > kWire24)
        throw std::invalid_argument(
            "ISA-v1 DCA CollectiveKey exceeds STREAM_V2 wire width");
    for (uint16_t core : plan.group)
        if (core > kWire12)
            throw std::invalid_argument(
                "ISA-v1 DCA source core exceeds STREAM_V2 source_id width");
    const uint64_t chunks =
        plan.length_bytes / kDteEndpointP2pMaxBytes +
        (plan.length_bytes % kDteEndpointP2pMaxBytes != 0);
    if (chunks == 0 || chunks > kWire24)
        throw std::invalid_argument(
            "ISA-v1 DCA chunk stream_id exceeds STREAM_V2 wire width");
    for (const IsaV1ReduceTarget &target : plan.reduce_targets) {
        if ((target.dtype != CollDType::UINT8 &&
             target.dtype != CollDType::INT32 &&
             target.dtype != CollDType::INT64) ||
            target.reduce_op == CollReduceOp::NONE)
            throw std::invalid_argument(
                "ISA-v1 DCA plan requires UINT8/INT32/INT64 SUM/MAX");
        config.dca.ValidateForDtype(target.dtype);
        const uint64_t dtype_bytes = CollDTypeBits(target.dtype) / 8;
        const uint64_t chunk_bytes = std::min<uint64_t>(
            plan.length_bytes, kDteEndpointP2pMaxBytes);
        const uint64_t elements = chunk_bytes / dtype_bytes;
        const uint64_t physical_flits =
            chunk_bytes / P2P_PAYLOAD_FRAGMENT_BYTES +
            (chunk_bytes % P2P_PAYLOAD_FRAGMENT_BYTES != 0);
        const uint64_t lanes =
            config.dca.vector_bits / CollDTypeBits(target.dtype);
        const uint64_t vector_beats =
            elements / lanes + (elements % lanes != 0);
        if (elements == 0 || elements > kWire24 ||
            physical_flits == 0 || physical_flits > kWire16 ||
            vector_beats == 0 || vector_beats > kWire16)
            throw std::invalid_argument(
                "ISA-v1 DCA chunk exceeds STREAM_V2 geometry width");
    }
}

bool SameSchedule(const IsaV1TreeSchedule &left,
                  const IsaV1TreeSchedule &right) {
    return left.conflict_graph == right.conflict_graph &&
           left.batches == right.batches &&
           left.peak_entries_by_router == right.peak_entries_by_router &&
           left.serial_fallback == right.serial_fallback;
}

std::vector<CollReduceTreeNode> BuildReduceNodes(
    const IsaV1CollectiveTreeTopology &topology,
    const NocCollDcaConfig &dca) {
    std::vector<CollReduceTreeNode> nodes;
    nodes.reserve(topology.entries.size());
    const std::set<uint16_t> members(topology.group.begin(),
                                     topology.group.end());
    for (const IsaV1CollectiveTreeEntry &entry : topology.entries) {
        uint8_t expected = static_cast<uint8_t>(
            entry.output_mask & ~(1U << CENTER));
        if (members.count(entry.router_id) != 0)
            expected = static_cast<uint8_t>(expected | (1U << CENTER));
        if (expected == 0)
            throw std::logic_error(
                "ISA-v1 DCA topology node has no expected input");
        const uint64_t fanin = static_cast<uint64_t>(
            __builtin_popcount(static_cast<unsigned>(expected)));
        if (fanin > dca.header_fifo_depth ||
            fanin > dca.operand_fifo_depth)
            throw std::invalid_argument(
                "ISA-v1 DCA FIFO is smaller than X-first tree fan-in");
        nodes.push_back({expected, entry.ingress});
    }
    return nodes;
}

} // namespace

bool IsaV1CollectiveAcceleratedTree::operator==(
    const IsaV1CollectiveAcceleratedTree &other) const {
    if (plan_index != other.plan_index || root_rank != other.root_rank ||
        multicast != other.multicast || dca_reduce != other.dca_reduce ||
        !(topology == other.topology) ||
        reduce_nodes.size() != other.reduce_nodes.size())
        return false;
    for (size_t index = 0; index < reduce_nodes.size(); ++index)
        if (reduce_nodes[index].expected_inputs !=
                other.reduce_nodes[index].expected_inputs ||
            reduce_nodes[index].parent_output !=
                other.reduce_nodes[index].parent_output)
            return false;
    return true;
}

bool IsaV1CollectiveProfilePlanImage::operator==(
    const IsaV1CollectiveProfilePlanImage &other) const {
    return plan_index == other.plan_index && key == other.key &&
           decision == other.decision &&
           suppress_endpoint_reduce_compute ==
               other.suppress_endpoint_reduce_compute &&
           requires_dca_payload_executor ==
               other.requires_dca_payload_executor &&
           trees == other.trees &&
           SameSchedule(tree_schedule, other.tree_schedule) &&
           max_trees_per_batch == other.max_trees_per_batch;
}

const IsaV1CollectiveProfilePlanImage *
IsaV1CollectiveProfileProgramImage::FindPlan(
    uint32_t plan_index) const noexcept {
    const auto found = std::lower_bound(
        plans_.begin(), plans_.end(), plan_index,
        [](const IsaV1CollectiveProfilePlanImage &plan, uint32_t index) {
            return plan.plan_index < index;
        });
    return found != plans_.end() && found->plan_index == plan_index
               ? &*found
               : nullptr;
}

IsaV1CollectiveProfileProgramImage
BuildIsaV1CollectiveProfileProgramImage(
    const IsaV1CollectiveProgramImage &base,
    const IsaV1CollectiveProfileImageConfig &config) {
    ValidateConfig(config);
    IsaV1CollectiveProfileProgramImage result;
    result.base_generation_ = base.Generation();
    result.base_cookie_ = base.Cookie();
    result.profile_ = config.noc.profile;

    uint32_t next_tree_id = 1;
    const auto &plans = base.Lowering().plans;
    if (plans.size() > config.max_profile_plans)
        throw std::overflow_error(
            "ISA-v1 profile plan capacity is exhausted");
    ReserveBounded(
        result.derived_bytes_,
        CheckedMultiply(plans.size(),
                        sizeof(IsaV1CollectiveProfilePlanImage),
                        "ISA-v1 profile plan bytes overflow"),
        config.max_derived_bytes,
        "ISA-v1 profile derived-byte capacity is exhausted");
    for (size_t plan_index = 0; plan_index < plans.size(); ++plan_index) {
        if (plan_index > std::numeric_limits<uint32_t>::max())
            throw std::overflow_error(
                "ISA-v1 profile plan index exceeds u32");
        const IsaV1CollectivePlan &plan = plans[plan_index];
        IsaV1CollectiveProfilePlanImage profile_plan;
        profile_plan.plan_index = static_cast<uint32_t>(plan_index);
        profile_plan.key = plan.key;
        profile_plan.max_trees_per_batch = config.max_trees_per_batch;
        profile_plan.decision = PlanIsaV1CollectiveProfile(
            {plan.op, plan.group.size(), config.noc.profile,
             config.capabilities});
        if (!profile_plan.decision.accepted)
            throw std::runtime_error(
                std::string("ISA-v1 requested collective backend rejected: ") +
                IsaV1ProfileRejectReasonName(
                    profile_plan.decision.reject_reason));
        ReserveBounded(
            result.derived_bytes_, profile_plan.decision.trace.size(),
            config.max_derived_bytes,
            "ISA-v1 profile derived-byte capacity is exhausted");

        const bool multicast = UsesMulticast(profile_plan.decision);
        const bool dca = UsesDca(profile_plan.decision);
        if ((multicast || dca) && GroupSpansDies(plan, config.mesh))
            throw std::runtime_error(
                "ISA-v1 accelerated collective cannot cross dies");
        profile_plan.suppress_endpoint_reduce_compute = dca;
        profile_plan.requires_dca_payload_executor = dca;
        if (dca) ValidateDcaDataContract(plan, config.noc);

        if (multicast || dca) {
            for (uint16_t root_rank : Roots(plan, multicast, dca)) {
                if (root_rank >= plan.group.size())
                    throw std::logic_error(
                        "ISA-v1 profile tree root rank is invalid");
                if (next_tree_id > std::numeric_limits<uint16_t>::max())
                    throw std::overflow_error(
                        "ISA-v1 profile image exhausts non-zero tree IDs");
                ReserveBounded(
                    result.tree_count_, 1, config.max_total_trees,
                    "ISA-v1 profile tree capacity is exhausted");
                const size_t estimated_entries = EstimateTreeEntryCount(
                    config.mesh, plan.group[root_rank], plan.group);
                ReserveBounded(
                    result.tree_entry_count_, estimated_entries,
                    config.max_total_tree_entries,
                    "ISA-v1 profile tree-entry capacity is exhausted");
                size_t tree_bytes = sizeof(IsaV1CollectiveAcceleratedTree);
                tree_bytes = CheckedAdd(
                    tree_bytes,
                    CheckedMultiply(plan.group.size(), sizeof(uint16_t),
                                    "ISA-v1 profile group bytes overflow"),
                    "ISA-v1 profile tree bytes overflow");
                tree_bytes = CheckedAdd(
                    tree_bytes,
                    CheckedMultiply(estimated_entries,
                                    sizeof(IsaV1CollectiveTreeEntry),
                                    "ISA-v1 profile entry bytes overflow"),
                    "ISA-v1 profile tree bytes overflow");
                if (dca)
                    tree_bytes = CheckedAdd(
                        tree_bytes,
                        CheckedMultiply(estimated_entries,
                                        sizeof(CollReduceTreeNode),
                                        "ISA-v1 profile DCA bytes overflow"),
                        "ISA-v1 profile tree bytes overflow");
                ReserveBounded(
                    result.derived_bytes_, tree_bytes,
                    config.max_derived_bytes,
                    "ISA-v1 profile derived-byte capacity is exhausted");
                IsaV1CollectiveAcceleratedTree tree;
                tree.plan_index = static_cast<uint32_t>(plan_index);
                tree.root_rank = root_rank;
                tree.multicast = RootUsesMulticast(
                    plan, profile_plan.decision, root_rank);
                tree.dca_reduce = RootUsesDca(
                    plan, profile_plan.decision, root_rank);
                tree.topology = BuildIsaV1XFirstCollectiveTree(
                    config.mesh, static_cast<uint16_t>(next_tree_id++),
                    plan.group[root_rank], plan.group);
                if (tree.topology.entries.size() != estimated_entries)
                    throw std::logic_error(
                        "ISA-v1 topology preflight entry count diverged");
                if (tree.dca_reduce)
                    tree.reduce_nodes =
                        BuildReduceNodes(tree.topology, config.noc.dca);
                if (!tree.multicast && !tree.dca_reduce)
                    throw std::logic_error(
                        "ISA-v1 profile constructed an unused tree");
                profile_plan.trees.push_back(std::move(tree));
            }
        }

        if (!profile_plan.trees.empty()) {
            const size_t potential_edges =
                PotentialConflictEdges(profile_plan.trees.size());
            ReserveBounded(
                result.potential_conflict_edge_count_, potential_edges,
                config.max_total_conflict_edges,
                "ISA-v1 profile conflict-edge capacity is exhausted");
            size_t schedule_bytes = CheckedMultiply(
                potential_edges, sizeof(std::pair<uint16_t, uint16_t>),
                "ISA-v1 profile conflict bytes overflow");
            schedule_bytes = CheckedAdd(
                schedule_bytes,
                CheckedMultiply(profile_plan.trees.size(),
                                sizeof(IsaV1TreeBatch),
                                "ISA-v1 profile batch bytes overflow"),
                "ISA-v1 profile schedule bytes overflow");
            ReserveBounded(
                result.derived_bytes_, schedule_bytes,
                config.max_derived_bytes,
                "ISA-v1 profile derived-byte capacity is exhausted");

            std::vector<IsaV1TreeScheduleInput> inputs;
            inputs.reserve(profile_plan.trees.size());
            for (const auto &tree : profile_plan.trees)
                inputs.push_back(IsaV1TreeScheduleInputFromTopology(
                    tree.topology));
            IsaV1TreeScheduleOptions options;
            options.max_trees_per_batch = config.max_trees_per_batch;
            options.entries_per_router = config.entries_per_router;
            options.occupied_entries_by_router =
                config.occupied_entries_by_router;
            profile_plan.tree_schedule =
                ScheduleIsaV1CollectiveTrees(inputs, options);
            ReserveBounded(
                result.batch_count_,
                profile_plan.tree_schedule.batches.size(),
                config.max_total_batches,
                "ISA-v1 profile batch capacity is exhausted");
        }

        for (const auto &tree : profile_plan.trees) {
            if (tree.multicast) ++result.multicast_tree_count_;
            if (tree.dca_reduce) ++result.dca_tree_count_;
        }
        result.plans_.push_back(std::move(profile_plan));
    }
    return result;
}
