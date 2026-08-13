#include "dte/coll_tree_registry_bridge_v1.h"

#include "dte/coll_innetwork_reduce.h"
#include "dte/coll_multicast.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

bool SameSchedule(const IsaV1TreeSchedule &left,
                  const IsaV1TreeSchedule &right) {
    return left.conflict_graph == right.conflict_graph &&
           left.batches == right.batches &&
           left.peak_entries_by_router == right.peak_entries_by_router &&
           left.serial_fallback == right.serial_fallback;
}

} // namespace

struct IsaV1CollectiveTreeRegistryBridge::Impl {
    struct RegisteredPlan {
        IsaV1CollectiveProfilePlanImage image;
        std::map<uint16_t, IsaV1CollectiveAcceleratedTree> trees;
    };

    explicit Impl(IsaV1CollectiveTreeBatchRuntimeConfig config)
        : runtime(std::move(config)) {}

    IsaV1CollectiveTreeBatchRuntime runtime;
    std::map<CollectiveKey, RegisteredPlan> plans;
    std::map<uint16_t, CollectiveKey> tree_owners;
    IsaV1TreeRegistryBridgeStats stats;
    size_t owned_multicast = 0;
    size_t owned_reduce = 0;

    RegisteredPlan &FindPlan(const CollectiveKey &key) {
        const auto found = plans.find(key);
        if (found == plans.end())
            throw std::runtime_error(
                "ISA-v1 production tree bridge key is unknown or stale");
        return found->second;
    }

    void EraseOwnedTree(const IsaV1CollectiveAcceleratedTree &tree,
                        bool verify) {
        const uint16_t tree_id = tree.topology.tree_id;
        if (tree.multicast) {
            const size_t erased = EraseCollectiveTree(tree_id);
            if (verify && erased != tree.topology.entries.size())
                throw std::logic_error(
                    "ISA-v1 production multicast erase count mismatch");
            stats.multicast_entries_erased += erased;
            if (erased > owned_multicast)
                throw std::logic_error(
                    "ISA-v1 production multicast ownership underflow");
            owned_multicast -= erased;
        }
        if (tree.dca_reduce) {
            const size_t erased = EraseCollectiveReduceTree(tree_id);
            if (verify && erased != tree.reduce_nodes.size())
                throw std::logic_error(
                    "ISA-v1 production reduce erase count mismatch");
            stats.reduce_nodes_erased += erased;
            if (erased > owned_reduce)
                throw std::logic_error(
                    "ISA-v1 production reduce ownership underflow");
            owned_reduce -= erased;
        }
    }

    std::vector<uint16_t> ActiveTreesFor(const CollectiveKey &key) const {
        const std::vector<uint16_t> active = runtime.ProgrammedTreeIds();
        if (active.empty()) return {};
        const auto owner = tree_owners.find(active.front());
        if (owner == tree_owners.end())
            throw std::logic_error(
                "ISA-v1 production bridge active tree has no owner");
        return owner->second == key ? active : std::vector<uint16_t>{};
    }

    void RemovePlan(const CollectiveKey &key) {
        const auto found = plans.find(key);
        if (found == plans.end())
            throw std::logic_error(
                "ISA-v1 production bridge lost completed plan");
        for (const auto &tree : found->second.trees)
            tree_owners.erase(tree.first);
        plans.erase(found);
    }
};

bool IsaV1TreeRegistryBridgeResidual::Empty() const {
    return schedule.Empty() && registered_profile_plans == 0 &&
           registered_tree_images == 0 && owned_multicast_entries == 0 &&
           owned_reduce_nodes == 0;
}

IsaV1CollectiveTreeRegistryBridge::IsaV1CollectiveTreeRegistryBridge(
    IsaV1CollectiveTreeBatchRuntimeConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

IsaV1CollectiveTreeRegistryBridge::~IsaV1CollectiveTreeRegistryBridge() =
    default;

void IsaV1CollectiveTreeRegistryBridge::RegisterPlan(
    const IsaV1CollectiveProfilePlanImage &plan) {
    if (plan.trees.empty())
        throw std::invalid_argument(
            "ISA-v1 production bridge cannot register a backend-free plan");
    if (impl_->plans.count(plan.key) != 0)
        throw std::runtime_error(
            "ISA-v1 production bridge duplicate plan key");

    Impl::RegisteredPlan candidate;
    candidate.image = plan;
    std::vector<IsaV1CollectiveTreeTopology> topologies;
    topologies.reserve(plan.trees.size());
    for (const IsaV1CollectiveAcceleratedTree &tree : plan.trees) {
        const uint16_t tree_id = tree.topology.tree_id;
        if (tree_id == 0 || (!tree.multicast && !tree.dca_reduce) ||
            (tree.dca_reduce &&
             tree.reduce_nodes.size() != tree.topology.entries.size()))
            throw std::invalid_argument(
                "ISA-v1 production bridge tree image is malformed");
        if (impl_->tree_owners.count(tree_id) != 0 ||
            CollectiveTreeEntryCountForTree(tree_id) != 0 ||
            CollectiveReduceNodeCountForTree(tree_id) != 0)
            throw std::runtime_error(
                "ISA-v1 production bridge tree_id collision");
        if (!candidate.trees.emplace(tree_id, tree).second)
            throw std::invalid_argument(
                "ISA-v1 production bridge duplicate tree_id");
        topologies.push_back(tree.topology);
    }

    const IsaV1TreeSchedule actual = impl_->runtime.RegisterSchedule(
        plan.key, topologies, plan.max_trees_per_batch);
    if (!SameSchedule(actual, plan.tree_schedule)) {
        impl_->runtime.Abort(plan.key);
        throw std::logic_error(
            "ISA-v1 production bridge schedule differs from immutable image");
    }
    impl_->plans.emplace(plan.key, std::move(candidate));
    for (const auto &tree : plan.trees)
        impl_->tree_owners.emplace(tree.topology.tree_id, plan.key);
}

void IsaV1CollectiveTreeRegistryBridge::RegisterImage(
    const IsaV1CollectiveProfileProgramImage &image) {
    std::vector<CollectiveKey> registered;
    try {
        for (const auto &plan : image.Plans()) {
            if (plan.trees.empty()) continue;
            RegisterPlan(plan);
            registered.push_back(plan.key);
        }
    } catch (...) {
        for (auto it = registered.rbegin(); it != registered.rend(); ++it)
            Abort(*it);
        throw;
    }
}

void IsaV1CollectiveTreeRegistryBridge::BeginBatch(
    const CollectiveKey &key, uint16_t batch_index) {
    Impl::RegisteredPlan &plan = impl_->FindPlan(key);
    const auto &batches = plan.image.tree_schedule.batches;
    if (batch_index >= batches.size())
        throw std::runtime_error(
            "ISA-v1 production bridge batch index is outside schedule");
    const std::vector<uint16_t> tree_ids = batches[batch_index].tree_ids;
    for (uint16_t tree_id : tree_ids)
        if (CollectiveTreeEntryCountForTree(tree_id) != 0 ||
            CollectiveReduceNodeCountForTree(tree_id) != 0)
            throw std::runtime_error(
                "ISA-v1 production bridge tree became resident before its batch");

    impl_->runtime.BeginBatch(key, batch_index);
    std::vector<uint16_t> programmed;
    try {
        for (uint16_t tree_id : tree_ids) {
            const auto found = plan.trees.find(tree_id);
            if (found == plan.trees.end())
                throw std::logic_error(
                    "ISA-v1 production batch references unknown image tree");
            const auto &tree = found->second;
            programmed.push_back(tree_id);
            if (tree.multicast) {
                for (const auto &entry : tree.topology.entries) {
                    ProgramCollectiveTreeEntry(
                        {tree_id, entry.router_id,
                         static_cast<uint8_t>(entry.ingress)},
                        entry.output_mask);
                    ++impl_->owned_multicast;
                    ++impl_->stats.multicast_entries_programmed;
                }
                ValidateCollectiveTree(tree_id, tree.topology.root,
                                       tree.topology.group);
            }
            if (tree.dca_reduce) {
                for (size_t index = 0;
                     index < tree.topology.entries.size(); ++index) {
                    ProgramCollectiveReduceNode(
                        tree_id, tree.topology.entries[index].router_id,
                        tree.reduce_nodes[index]);
                    ++impl_->owned_reduce;
                    ++impl_->stats.reduce_nodes_programmed;
                }
            }
        }
    } catch (...) {
        for (auto it = programmed.rbegin(); it != programmed.rend(); ++it)
            impl_->EraseOwnedTree(plan.trees.at(*it), false);
        for (uint16_t tree_id : tree_ids) {
            EraseCollectiveTree(tree_id);
            EraseCollectiveReduceTree(tree_id);
        }
        impl_->runtime.Abort(key);
        impl_->RemovePlan(key);
        throw;
    }
    ++impl_->stats.batches_programmed;
    impl_->stats.owned_multicast_entries_peak = std::max(
        impl_->stats.owned_multicast_entries_peak,
        impl_->owned_multicast);
    impl_->stats.owned_reduce_nodes_peak = std::max(
        impl_->stats.owned_reduce_nodes_peak, impl_->owned_reduce);
}

void IsaV1CollectiveTreeRegistryBridge::MarkTreeComplete(
    const CollectiveKey &key, uint16_t batch_index, uint16_t tree_id) {
    impl_->runtime.MarkTreeComplete(key, batch_index, tree_id);
}

void IsaV1CollectiveTreeRegistryBridge::EndBatch(
    const CollectiveKey &key, uint16_t batch_index) {
    Impl::RegisteredPlan &plan = impl_->FindPlan(key);
    const std::vector<uint16_t> active =
        impl_->runtime.ProgrammedTreeIds();
    const size_t schedules_before =
        impl_->runtime.Residual().registered_schedules;
    impl_->runtime.EndBatch(key, batch_index);
    for (uint16_t tree_id : active)
        impl_->EraseOwnedTree(plan.trees.at(tree_id), true);
    ++impl_->stats.batches_released;
    if (impl_->runtime.Residual().registered_schedules + 1 ==
        schedules_before)
        impl_->RemovePlan(key);
}

void IsaV1CollectiveTreeRegistryBridge::Abort(const CollectiveKey &key) {
    Impl::RegisteredPlan &plan = impl_->FindPlan(key);
    const std::vector<uint16_t> active =
        impl_->ActiveTreesFor(key);
    impl_->runtime.Abort(key);
    for (uint16_t tree_id : active)
        impl_->EraseOwnedTree(plan.trees.at(tree_id), true);
    impl_->RemovePlan(key);
    ++impl_->stats.batches_aborted;
}

IsaV1TreeRegistryBridgeResidual
IsaV1CollectiveTreeRegistryBridge::Residual() const {
    IsaV1TreeRegistryBridgeResidual result;
    result.schedule = impl_->runtime.Residual();
    result.registered_profile_plans = impl_->plans.size();
    result.registered_tree_images = impl_->tree_owners.size();
    result.owned_multicast_entries = impl_->owned_multicast;
    result.owned_reduce_nodes = impl_->owned_reduce;
    return result;
}

const IsaV1TreeRegistryBridgeStats &
IsaV1CollectiveTreeRegistryBridge::Stats() const noexcept {
    return impl_->stats;
}

const IsaV1CollectiveTreeBatchRuntime &
IsaV1CollectiveTreeRegistryBridge::BatchRuntime() const noexcept {
    return impl_->runtime;
}
