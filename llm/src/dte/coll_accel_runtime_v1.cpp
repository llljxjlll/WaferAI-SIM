#include "dte/coll_accel_runtime_v1.h"

#include "dte/endpoint_contract.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

uint64_t CheckedAddressAdd(uint64_t base, uint64_t offset) {
    if (base > std::numeric_limits<uint64_t>::max() - offset)
        throw std::overflow_error(
            "ISA-v1 acceleration SRAM address overflows u64");
    return base + offset;
}

uint64_t CheckedRankOffset(uint16_t rank, uint64_t length) {
    if (rank != 0 &&
        length > std::numeric_limits<uint64_t>::max() / rank)
        throw std::overflow_error(
            "ISA-v1 acceleration rank-major offset overflows u64");
    return static_cast<uint64_t>(rank) * length;
}

const IsaV1ReduceTarget &ReduceTarget(
    const IsaV1CollectivePlan &plan, uint16_t rank) {
    const auto found = std::find_if(
        plan.reduce_targets.begin(), plan.reduce_targets.end(),
        [rank](const IsaV1ReduceTarget &target) {
            return target.rank == rank;
        });
    if (found == plan.reduce_targets.end())
        throw std::invalid_argument(
            "ISA-v1 multicast all-reduce rank has no reduce target");
    return *found;
}

std::string TreeIds(const std::vector<uint16_t> &ids) {
    std::ostringstream result;
    for (size_t index = 0; index < ids.size(); ++index) {
        if (index != 0) result << ',';
        result << ids[index];
    }
    return result.str();
}

size_t PeakEntriesPerRouter(const IsaV1TreeBatchRuntimeStats &stats) {
    size_t peak = 0;
    for (const auto &entry : stats.peak_entries_by_router)
        peak = std::max(peak, static_cast<size_t>(entry.second));
    return peak;
}

const char *ReleaseReasonName(IsaV1TreeBatchReleaseReason reason) {
    switch (reason) {
    case IsaV1TreeBatchReleaseReason::NONE: return "none";
    case IsaV1TreeBatchReleaseReason::BATCH_COMPLETE:
        return "batch_complete";
    case IsaV1TreeBatchReleaseReason::SCHEDULE_COMPLETE:
        return "schedule_complete";
    case IsaV1TreeBatchReleaseReason::ABORT: return "abort";
    }
    throw std::logic_error("unknown ISA-v1 tree release reason");
}

IsaV1TreeBatchTraceEvent LatestBatchEvent(
    const IsaV1CollectiveTreeBatchRuntime &runtime,
    IsaV1TreeBatchTraceEventKind expected) {
    static thread_local std::vector<IsaV1TreeBatchTraceEvent> snapshot;
    snapshot = runtime.TraceEvents();
    if (snapshot.empty() || snapshot.back().kind != expected)
        throw std::logic_error(
            "ISA-v1 production batch trace lost its lifecycle event");
    return snapshot.back();
}

} // namespace

uint64_t IsaV1CollectiveMulticastSourceAddress(
    const IsaV1CollectivePlan &plan,
    const IsaV1CollectiveAcceleratedTree &tree, uint64_t chunk_offset) {
    if (tree.root_rank >= plan.rank_records.size() ||
        chunk_offset >= plan.length_bytes)
        throw std::invalid_argument(
            "ISA-v1 multicast source rank/chunk is invalid");
    uint64_t base =
        plan.rank_records.at(tree.root_rank).send.base_address_bytes;
    if (plan.op == CollOp::ALLREDUCE && tree.dca_reduce)
        base = ReduceTarget(plan, tree.root_rank).result_address_bytes;
    return CheckedAddressAdd(base, chunk_offset);
}

uint64_t IsaV1CollectiveMulticastDestinationAddress(
    const IsaV1CollectivePlan &plan,
    const IsaV1CollectiveAcceleratedTree &tree, uint16_t target_rank,
    uint64_t chunk_offset) {
    if (tree.root_rank >= plan.rank_records.size() ||
        target_rank >= plan.rank_records.size() ||
        target_rank == tree.root_rank || chunk_offset >= plan.length_bytes)
        throw std::invalid_argument(
            "ISA-v1 multicast destination rank/chunk is invalid");
    uint64_t base =
        plan.rank_records.at(target_rank).receive.base_address_bytes;
    if (plan.op == CollOp::ALLGATHER) {
        base = CheckedAddressAdd(
            base, CheckedRankOffset(tree.root_rank, plan.length_bytes));
    } else if (plan.op == CollOp::ALLREDUCE) {
        const IsaV1ReduceTarget &target = ReduceTarget(plan, target_rank);
        base = tree.dca_reduce
                   ? target.result_address_bytes
                   : CheckedAddressAdd(
                         target.staging_address_bytes,
                         CheckedRankOffset(tree.root_rank,
                                           plan.length_bytes));
    }
    return CheckedAddressAdd(base, chunk_offset);
}

bool IsaV1CollectiveAccelerationRuntimeResidual::Empty() const noexcept {
    return plans == 0 && active_batches == 0 && active_trees == 0 &&
           live_sessions == 0 && pending_multicast_acks == 0 &&
           pending_dca_acks == 0;
}

IsaV1CollectiveAccelerationRuntime::IsaV1CollectiveAccelerationRuntime(
    std::shared_ptr<const IsaV1CollectiveProgramImage> base,
    std::shared_ptr<const IsaV1CollectiveProfileProgramImage> profile,
    std::shared_ptr<IsaV1CollectiveTreeRegistryBridge> bridge,
    uint32_t first_session)
    : base_(std::move(base)), profile_(std::move(profile)),
      bridge_(std::move(bridge)), next_session_(first_session) {
    if (!base_ || !profile_ || !bridge_ ||
        profile_->BaseGeneration() != base_->Generation() ||
        profile_->BaseCookie() != base_->Cookie() || first_session == 0 ||
        first_session > std::numeric_limits<uint16_t>::max())
        throw std::invalid_argument(
            "ISA-v1 acceleration runtime image/bridge contract mismatch");
    const auto &semantic = base_->Lowering().plans;
    uint64_t required_sessions = 0;
    for (const auto &image : profile_->Plans()) {
        if (image.trees.empty()) continue;
        if (image.plan_index >= semantic.size() ||
            !(semantic[image.plan_index].key == image.key) ||
            image.tree_schedule.batches.empty())
            throw std::invalid_argument(
                "ISA-v1 acceleration runtime profile plan is malformed");
        const IsaV1CollectivePlan &plan = semantic[image.plan_index];
        PlanState state;
        state.key = image.key;
        state.plan_index = image.plan_index;
        state.conflict_edges = image.tree_schedule.conflict_graph.edges.size();
        state.batches = image.tree_schedule.batches;
        state.expected_observers.insert(plan.group.begin(), plan.group.end());
        const uint64_t chunks64 =
            plan.length_bytes / kDteEndpointP2pMaxBytes +
            (plan.length_bytes % kDteEndpointP2pMaxBytes != 0);
        if (chunks64 == 0 || chunks64 > std::numeric_limits<uint32_t>::max())
            throw std::overflow_error(
                "ISA-v1 acceleration runtime chunk count exceeds u32");
        for (const auto &image_tree : image.trees) {
            if (image_tree.root_rank >= plan.group.size())
                throw std::invalid_argument(
                    "ISA-v1 acceleration tree root rank is invalid");
            TreeState tree;
            tree.tree_id = image_tree.topology.tree_id;
            tree.multicast = image_tree.multicast;
            tree.dca = image_tree.dca_reduce;
            tree.root_core = plan.group[image_tree.root_rank];
            tree.chunk_count = static_cast<uint32_t>(chunks64);
            if (tree.multicast) {
                if (required_sessions >
                    std::numeric_limits<uint64_t>::max() - chunks64)
                    throw std::overflow_error(
                        "ISA-v1 multicast session preflight overflows");
                required_sessions += chunks64;
                for (uint16_t core : plan.group)
                    if (core != tree.root_core)
                        tree.multicast_targets.insert(core);
            }
            if (tree.dca)
                tree.dca_sources.insert(plan.group.begin(), plan.group.end());
            if (!state.trees.emplace(tree.tree_id, std::move(tree)).second)
                throw std::invalid_argument(
                    "ISA-v1 acceleration runtime duplicates a tree");
        }
        if (!plans_.emplace(state.key, std::move(state)).second)
            throw std::invalid_argument(
                "ISA-v1 acceleration runtime duplicates a plan key");
    }
    const uint64_t available_sessions =
        static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) -
        first_session + 1;
    if (required_sessions > available_sessions)
        throw std::overflow_error(
            "ISA-v1 multicast session image exceeds non-reuse space");
}

IsaV1CollectiveAccelerationRuntime::PlanState &
IsaV1CollectiveAccelerationRuntime::FindPlan(const CollectiveKey &key) {
    auto found = plans_.find(key);
    if (found == plans_.end())
        throw std::runtime_error(
            "ISA-v1 acceleration plan key is unknown or retired");
    return found->second;
}

const IsaV1CollectiveAccelerationRuntime::PlanState &
IsaV1CollectiveAccelerationRuntime::FindPlan(
    const CollectiveKey &key) const {
    auto found = plans_.find(key);
    if (found == plans_.end())
        throw std::runtime_error(
            "ISA-v1 acceleration plan key is unknown or retired");
    return found->second;
}

IsaV1CollectiveAccelerationRuntime::TreeState &
IsaV1CollectiveAccelerationRuntime::FindTree(PlanState &plan,
                                              uint16_t tree_id) {
    auto found = plan.trees.find(tree_id);
    if (found == plan.trees.end())
        throw std::invalid_argument(
            "ISA-v1 acceleration tree is outside the plan");
    return found->second;
}

const IsaV1CollectiveAccelerationRuntime::TreeState &
IsaV1CollectiveAccelerationRuntime::FindTree(
    const PlanState &plan, uint16_t tree_id) const {
    auto found = plan.trees.find(tree_id);
    if (found == plan.trees.end())
        throw std::invalid_argument(
            "ISA-v1 acceleration tree is outside the plan");
    return found->second;
}

bool IsaV1CollectiveAccelerationRuntime::EnsureActive(
    PlanState &plan, uint16_t tree_id) {
    if (active_)
        return active_->key == plan.key &&
               active_->tree_ids.count(tree_id) != 0;
    if (plan.next_batch >= plan.batches.size())
        throw std::logic_error(
            "ISA-v1 acceleration plan has no next batch");
    const IsaV1TreeBatch &batch = plan.batches[plan.next_batch];
    if (batch.batch_index != plan.next_batch)
        throw std::logic_error(
            "ISA-v1 acceleration batch indices are not canonical");
    const std::set<uint16_t> ids(batch.tree_ids.begin(),
                                 batch.tree_ids.end());
    if (ids.count(tree_id) == 0) return false;
    bridge_->BeginBatch(plan.key, batch.batch_index);
    const auto &batch_runtime = bridge_->BatchRuntime();
    const auto event = LatestBatchEvent(
        batch_runtime, IsaV1TreeBatchTraceEventKind::PROGRAM);
    std::cout << "[P7_TREE_BATCH] event=begin plan=" << plan.plan_index
              << " key=" << plan.key.group_id << ':'
              << plan.key.collective_id << ':' << plan.key.epoch
              << " batch=" << batch.batch_index
              << " tree_ids=" << TreeIds(event.tree_ids)
              << " trees=" << event.tree_ids.size()
              << " programmed=" << event.tree_ids.size()
              << " erased=0 conflicts=" << plan.conflict_edges
              << " peak_entries="
              << PeakEntriesPerRouter(batch_runtime.Stats())
              << " capacity="
              << batch_runtime.EntriesPerRouterCapacity()
              << " occupancy_after=" << event.managed_occupancy_after
              << " release_reason=none" << std::endl;
    active_ = ActiveBatch{plan.key, batch.batch_index, ids};
    ++stats_.batches_started;
    return true;
}

uint16_t IsaV1CollectiveAccelerationRuntime::AllocateSession() {
    if (next_session_ > std::numeric_limits<uint16_t>::max())
        throw std::overflow_error(
            "ISA-v1 multicast session space is exhausted; reuse is forbidden");
    ++stats_.sessions_allocated;
    return static_cast<uint16_t>(next_session_++);
}

std::optional<uint16_t>
IsaV1CollectiveAccelerationRuntime::TryAcquireMulticast(
    const CollectiveKey &key, uint16_t tree_id, uint32_t chunk_id) {
    PlanState &plan = FindPlan(key);
    if (plan.complete) return std::nullopt;
    TreeState &tree = FindTree(plan, tree_id);
    if (!tree.multicast || chunk_id >= tree.chunk_count)
        throw std::invalid_argument(
            "ISA-v1 multicast acquisition contract is invalid");
    for (uint16_t target : tree.multicast_targets)
        if (tree.multicast_posts.count({chunk_id, target}) == 0)
            return std::nullopt;
    if (chunk_id != 0)
        for (uint16_t target : tree.multicast_targets)
            if (tree.multicast_acks.count({chunk_id - 1, target}) == 0)
                return std::nullopt;
    if (!EnsureActive(plan, tree_id) || (tree.dca && !DcaDone(tree)))
        return std::nullopt;
    auto found = tree.multicast_sessions.find(chunk_id);
    if (found != tree.multicast_sessions.end()) return found->second;
    const uint16_t session = AllocateSession();
    tree.multicast_sessions.emplace(chunk_id, session);
    return session;
}

void IsaV1CollectiveAccelerationRuntime::RegisterMulticastPost(
    const CollectiveKey &key, uint16_t tree_id, uint32_t chunk_id,
    uint16_t target_core) {
    PlanState &plan = FindPlan(key);
    TreeState &tree = FindTree(plan, tree_id);
    if (!tree.multicast || chunk_id >= tree.chunk_count ||
        tree.multicast_targets.count(target_core) == 0 ||
        !tree.multicast_posts.emplace(chunk_id, target_core).second)
        throw std::runtime_error(
            "ISA-v1 multicast post is invalid or duplicate");
}

void IsaV1CollectiveAccelerationRuntime::RegisterDcaRootReady(
    const CollectiveKey &key, uint16_t tree_id, uint32_t chunk_id,
    uint16_t root_core) {
    PlanState &plan = FindPlan(key);
    TreeState &tree = FindTree(plan, tree_id);
    if (!tree.dca || chunk_id >= tree.chunk_count ||
        root_core != tree.root_core)
        throw std::runtime_error(
            "ISA-v1 DCA root-ready post is invalid");
    auto &chunk = tree.dca_chunks[chunk_id];
    if (chunk.root_ready)
        throw std::runtime_error("ISA-v1 duplicate DCA root-ready post");
    chunk.root_ready = true;
}

bool IsaV1CollectiveAccelerationRuntime::TryAcquireDca(
    const CollectiveKey &key, uint16_t tree_id, uint32_t chunk_id,
    uint16_t source_core) {
    PlanState &plan = FindPlan(key);
    if (plan.complete) return false;
    TreeState &tree = FindTree(plan, tree_id);
    if (!tree.dca || chunk_id >= tree.chunk_count ||
        tree.dca_sources.count(source_core) == 0)
        throw std::invalid_argument(
            "ISA-v1 DCA acquisition references a non-DCA tree");
    auto found = tree.dca_chunks.find(chunk_id);
    if (found == tree.dca_chunks.end() || !found->second.root_ready ||
        !EnsureActive(plan, tree_id))
        return false;
    if (chunk_id != 0) {
        const auto previous = tree.dca_chunks.find(chunk_id - 1);
        if (previous == tree.dca_chunks.end() || !previous->second.ack)
            return false;
    }
    found->second.sources.insert(source_core);
    return true;
}

bool IsaV1CollectiveAccelerationRuntime::MulticastDone(
    const TreeState &tree) const {
    if (!tree.multicast) return true;
    const uint64_t expected = static_cast<uint64_t>(tree.chunk_count) *
                              tree.multicast_targets.size();
    return tree.multicast_sessions.size() == tree.chunk_count &&
           tree.multicast_acks.size() == expected;
}

bool IsaV1CollectiveAccelerationRuntime::DcaDone(
    const TreeState &tree) const {
    if (!tree.dca) return true;
    if (tree.dca_chunks.size() != tree.chunk_count) return false;
    for (uint32_t chunk_id = 0; chunk_id < tree.chunk_count; ++chunk_id) {
        const auto found = tree.dca_chunks.find(chunk_id);
        if (found == tree.dca_chunks.end() || !found->second.root_ready ||
            found->second.sources != tree.dca_sources || !found->second.ack)
            return false;
    }
    return true;
}

void IsaV1CollectiveAccelerationRuntime::AcknowledgeMulticast(
    const CollectiveKey &key, uint16_t tree_id, uint32_t chunk_id,
    uint16_t target_core) {
    PlanState &plan = FindPlan(key);
    TreeState &tree = FindTree(plan, tree_id);
    if (!active_ || !(active_->key == key) ||
        active_->tree_ids.count(tree_id) == 0 ||
        tree.multicast_sessions.count(chunk_id) == 0 ||
        tree.multicast_targets.count(target_core) == 0)
        throw std::runtime_error(
            "ISA-v1 multicast ACK is stale, unknown, or unissued");
    if (!tree.multicast_acks.emplace(chunk_id, target_core).second)
        throw std::runtime_error("ISA-v1 duplicate multicast target ACK");
    ++stats_.multicast_acks;
    MaybeCompleteTree(plan, tree);
}

void IsaV1CollectiveAccelerationRuntime::AcknowledgeDca(
    const CollectiveKey &key, uint16_t tree_id, uint32_t chunk_id,
    uint16_t root_core) {
    PlanState &plan = FindPlan(key);
    TreeState &tree = FindTree(plan, tree_id);
    auto chunk = tree.dca_chunks.find(chunk_id);
    if (!active_ || !(active_->key == key) ||
        active_->tree_ids.count(tree_id) == 0 || !tree.dca ||
        chunk == tree.dca_chunks.end() || !chunk->second.root_ready ||
        chunk->second.sources != tree.dca_sources ||
        root_core != tree.root_core || chunk->second.ack)
        throw std::runtime_error(
            "ISA-v1 DCA ACK is stale, duplicate, or has the wrong root");
    chunk->second.ack = true;
    ++stats_.dca_acks;
    MaybeCompleteTree(plan, tree);
}

void IsaV1CollectiveAccelerationRuntime::MaybeCompleteTree(
    PlanState &plan, TreeState &tree) {
    if (tree.marked_complete || !MulticastDone(tree) || !DcaDone(tree))
        return;
    if (!active_ || !(active_->key == plan.key) ||
        active_->tree_ids.count(tree.tree_id) == 0)
        throw std::logic_error(
            "ISA-v1 completed tree is not in the active batch");
    bridge_->MarkTreeComplete(plan.key, active_->batch_index, tree.tree_id);
    tree.marked_complete = true;
    ++stats_.trees_completed;
    MaybeCompleteBatch(plan);
}

void IsaV1CollectiveAccelerationRuntime::MaybeCompleteBatch(
    PlanState &plan) {
    if (!active_ || !(active_->key == plan.key)) return;
    for (uint16_t tree_id : active_->tree_ids)
        if (!FindTree(plan, tree_id).marked_complete) return;
    const uint16_t batch = active_->batch_index;
    bridge_->EndBatch(plan.key, batch);
    const auto &batch_runtime = bridge_->BatchRuntime();
    const auto event = LatestBatchEvent(
        batch_runtime, IsaV1TreeBatchTraceEventKind::RELEASE);
    std::cout << "[P7_TREE_BATCH] event=end plan=" << plan.plan_index
              << " key=" << plan.key.group_id << ':'
              << plan.key.collective_id << ':' << plan.key.epoch
              << " batch=" << batch
              << " tree_ids=" << TreeIds(event.tree_ids)
              << " trees=" << event.tree_ids.size()
              << " programmed=0 erased=" << event.tree_ids.size()
              << " conflicts=" << plan.conflict_edges
              << " peak_entries="
              << PeakEntriesPerRouter(batch_runtime.Stats())
              << " capacity="
              << batch_runtime.EntriesPerRouterCapacity()
              << " occupancy_after=" << event.managed_occupancy_after
              << " release_reason="
              << ReleaseReasonName(event.release_reason) << std::endl;
    active_.reset();
    ++plan.next_batch;
    ++stats_.batches_completed;
    if (plan.next_batch == plan.batches.size()) {
        plan.complete = true;
        completed_plans_.insert(plan.key);
        const auto residual = bridge_->Residual();
        const size_t total_residual =
            residual.schedule.registered_schedules +
            residual.schedule.registered_trees +
            residual.schedule.planned_entries +
            residual.schedule.active_batches +
            residual.schedule.programmed_trees +
            residual.schedule.programmed_entries +
            residual.schedule.completed_trees +
            residual.registered_profile_plans +
            residual.registered_tree_images +
            residual.owned_multicast_entries +
            residual.owned_reduce_nodes;
        std::cout << "[P7_TREE_DRAIN] plan=" << plan.plan_index
                  << " key=" << plan.key.group_id << ':'
                  << plan.key.collective_id << ':' << plan.key.epoch
                  << " tree_entries="
                  << residual.owned_multicast_entries
                  << " reduce_nodes=" << residual.owned_reduce_nodes
                  << " schedule_entries="
                  << residual.schedule.programmed_entries
                  << " residual=" << total_residual << std::endl;
    }
}

bool IsaV1CollectiveAccelerationRuntime::MulticastComplete(
    const CollectiveKey &key, uint16_t tree_id) const {
    return MulticastDone(FindTree(FindPlan(key), tree_id));
}

bool IsaV1CollectiveAccelerationRuntime::DcaSourcesReady(
    const CollectiveKey &key, uint16_t tree_id, uint32_t chunk_id) const {
    const TreeState &tree = FindTree(FindPlan(key), tree_id);
    if (!tree.dca || chunk_id >= tree.chunk_count)
        throw std::invalid_argument(
            "ISA-v1 DCA readiness references a non-DCA chunk");
    const auto found = tree.dca_chunks.find(chunk_id);
    return found != tree.dca_chunks.end() && found->second.root_ready &&
           found->second.sources == tree.dca_sources;
}

bool IsaV1CollectiveAccelerationRuntime::DcaComplete(
    const CollectiveKey &key, uint16_t tree_id) const {
    return DcaDone(FindTree(FindPlan(key), tree_id));
}

bool IsaV1CollectiveAccelerationRuntime::TreeComplete(
    const CollectiveKey &key, uint16_t tree_id) const {
    return FindTree(FindPlan(key), tree_id).marked_complete;
}

bool IsaV1CollectiveAccelerationRuntime::PlanComplete(
    const CollectiveKey &key) const noexcept {
    return completed_plans_.count(key) != 0;
}

void IsaV1CollectiveAccelerationRuntime::ObservePlanComplete(
    const CollectiveKey &key, uint16_t core) {
    auto found = plans_.find(key);
    if (found == plans_.end() || !found->second.complete ||
        completed_plans_.count(key) == 0)
        throw std::runtime_error(
            "ISA-v1 acceleration completion observation is stale or early");
    PlanState &plan = found->second;
    if (plan.expected_observers.count(core) == 0 ||
        !plan.completion_observers.insert(core).second)
        throw std::runtime_error(
            "ISA-v1 acceleration completion observer is invalid or duplicate");
    if (plan.completion_observers == plan.expected_observers)
        plans_.erase(found);
}

void IsaV1CollectiveAccelerationRuntime::Abort(
    const CollectiveKey &key) {
    FindPlan(key);
    if (active_ && !(active_->key == key))
        throw std::runtime_error(
            "ISA-v1 cannot abort an inactive plan while another batch owns tables");
    bridge_->Abort(key);
    if (active_) active_.reset();
    plans_.erase(key);
}

IsaV1CollectiveAccelerationRuntimeResidual
IsaV1CollectiveAccelerationRuntime::Residual() const {
    IsaV1CollectiveAccelerationRuntimeResidual result;
    result.plans = plans_.size();
    result.active_batches = active_ ? 1 : 0;
    result.active_trees = active_ ? active_->tree_ids.size() : 0;
    for (const auto &plan : plans_)
        for (const auto &entry : plan.second.trees) {
            const TreeState &tree = entry.second;
            result.live_sessions += tree.multicast_sessions.size();
            if (tree.multicast) {
                const size_t expected = tree.chunk_count *
                                        tree.multicast_targets.size();
                if (expected >= tree.multicast_acks.size())
                    result.pending_multicast_acks +=
                        expected - tree.multicast_acks.size();
            }
            if (tree.dca)
                for (uint32_t chunk = 0; chunk < tree.chunk_count; ++chunk) {
                    const auto found = tree.dca_chunks.find(chunk);
                    if (found == tree.dca_chunks.end() ||
                        !found->second.ack)
                        ++result.pending_dca_acks;
                }
        }
    return result;
}
