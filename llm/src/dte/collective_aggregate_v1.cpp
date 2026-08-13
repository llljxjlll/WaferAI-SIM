#include "dte/collective_aggregate_v1.h"

#include "isa/record_lowering.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace {

void Require(bool condition, const std::string &message) {
    if (!condition) throw std::invalid_argument(message);
}

bool IsSendAction(IsaV1ActionKind kind) {
    return kind == IsaV1ActionKind::ISSUE_SEND ||
           kind == IsaV1ActionKind::WAIT_SEND ||
           kind == IsaV1ActionKind::WAIT_TRANSPORT_RETIRE;
}

bool IsReceiveAction(IsaV1ActionKind kind) {
    return kind == IsaV1ActionKind::POST_RECEIVE ||
           kind == IsaV1ActionKind::WAIT_RECEIVE;
}

struct ExpectedAction {
    size_t plan_index = 0;
    CollectiveKey key;
    IsaV1Action action;
};

using ChildDescriptorKey = std::pair<size_t, uint32_t>;

} // namespace

bool CollectiveAggregateLocalWorkHandle::operator==(
    const CollectiveAggregateLocalWorkHandle &other) const noexcept {
    return std::tie(id, public_token, key, kind, plan_index, item_index) ==
           std::tie(other.id, other.public_token, other.key, other.kind,
                    other.plan_index, other.item_index);
}

CollectiveAggregateRuntime::CollectiveAggregateRuntime(
    uint16_t local_core, CollectiveAggregateCapacity capacity)
    : local_core_(local_core), capacity_(capacity) {
    if (capacity.max_aggregates == 0)
        throw std::invalid_argument(
            "collective aggregate capacity must be non-zero");
}

void CollectiveAggregateRuntime::RegisterArtifact(
    const IsaV1CollectiveArtifactLowering &lowering,
    const std::set<uint32_t> &reserved_tokens) {
    AggregateMap candidate_aggregates = aggregates_;
    std::map<uint32_t, Child> candidate_children = children_;
    std::map<uint64_t, LocalWork> candidate_local_work = local_work_;
    std::set<uint32_t> candidate_reserved = reserved_tokens_;
    uint64_t candidate_next_work_id = next_local_work_id_;

    for (uint32_t token : reserved_tokens) {
        if (token == 0) continue;
        Require(candidate_aggregates.count(token) == 0,
                "ordinary DTE token collides with a collective public token");
        Require(candidate_children.count(token) == 0,
                "ordinary DTE token collides with a collective child token");
        if (candidate_reserved.count(token) == 0 &&
            candidate_reserved.size() == capacity_.max_reserved_tokens)
            throw std::overflow_error(
                "collective aggregate reserved-token capacity exhausted");
        candidate_reserved.insert(token);
    }
    if (candidate_reserved.size() > capacity_.max_reserved_tokens)
        throw std::overflow_error(
            "collective aggregate reserved-token capacity exhausted");

    std::map<std::pair<size_t, bool>, uint32_t> endpoint_tokens;
    std::map<size_t, uint16_t> local_ranks;
    std::set<uint32_t> new_public_tokens;
    auto add_aggregate = [&](size_t plan_index, bool receive,
                             uint32_t token, const CollectiveKey &key) {
        Require(token != 0,
                "collective aggregate public token must be non-zero");
        Require(candidate_reserved.count(token) == 0,
                "collective public token collides with ordinary DTE token");
        Require(candidate_children.count(token) == 0,
                "collective public token collides with a child token");
        if (candidate_aggregates.size() == capacity_.max_aggregates)
            throw std::overflow_error(
                "collective aggregate capacity exhausted");
        Require(candidate_aggregates.emplace(
                    token, Aggregate{key, false, {}, {}}).second,
                "collective public token is not unique on one core");
        endpoint_tokens.emplace(std::make_pair(plan_index, receive), token);
        new_public_tokens.insert(token);
    };

    for (size_t plan_index = 0; plan_index < lowering.plans.size();
         ++plan_index) {
        const IsaV1CollectivePlan &plan = lowering.plans[plan_index];
        Require(plan.rank_records.size() == plan.group.size(),
                "collective plan rank-record count is inconsistent");
        Require(plan.actions_by_rank.size() == plan.group.size(),
                "collective plan action-rank count is inconsistent");
        const auto rank =
            std::find(plan.group.begin(), plan.group.end(), local_core_);
        if (rank == plan.group.end()) continue;
        Require(std::find(rank + 1, plan.group.end(), local_core_) ==
                    plan.group.end(),
                "collective plan contains the local core twice");
        const uint16_t rank_index = static_cast<uint16_t>(
            std::distance(plan.group.begin(), rank));
        local_ranks.emplace(plan_index, rank_index);
        const IsaV1RankRecordContract &contract =
            plan.rank_records[rank_index];
        if (contract.send.present) {
            Require(contract.send.asynchronous,
                    "collective SEND aggregate must be asynchronous");
            add_aggregate(plan_index, false, contract.send.token, plan.key);
        } else {
            Require(contract.send.token == 0,
                    "absent collective SEND carries a token");
        }
        if (contract.receive.present) {
            Require(contract.receive.asynchronous,
                    "collective RECEIVE aggregate must be asynchronous");
            add_aggregate(plan_index, true, contract.receive.token, plan.key);
        } else {
            Require(contract.receive.token == 0,
                    "absent collective RECEIVE carries a token");
        }
    }

    std::map<ChildDescriptorKey, const IsaV1LoweredCollectiveChild *>
        child_descriptors;
    std::set<uint32_t> artifact_internal_tokens;
    for (const IsaV1LoweredCollectiveChild &child : lowering.children) {
        Require(child.plan_index < lowering.plans.size(),
                "collective child references an unknown plan");
        const IsaV1CollectivePlan &plan = lowering.plans[child.plan_index];
        Require(child.child_index < plan.child_flows.size(),
                "collective child index is out of range");
        Require(child_descriptors.emplace(
                    ChildDescriptorKey{child.plan_index, child.child_index},
                    &child).second,
                "collective child descriptor is duplicated");
        Require(child.source_internal_token != 0 &&
                    child.destination_internal_token != 0 &&
                    child.source_internal_token !=
                        child.destination_internal_token,
                "collective child internal tokens are invalid");
        Require(artifact_internal_tokens.insert(
                    child.source_internal_token).second &&
                    artifact_internal_tokens.insert(
                    child.destination_internal_token).second,
                "collective child token belongs to more than one endpoint");
    }
    size_t expected_descriptor_count = 0;
    for (size_t plan_index = 0; plan_index < lowering.plans.size();
         ++plan_index) {
        const IsaV1CollectivePlan &plan = lowering.plans[plan_index];
        if (plan.child_flows.size() >
            std::numeric_limits<size_t>::max() - expected_descriptor_count)
            throw std::overflow_error(
                "collective child descriptor count overflows size_t");
        expected_descriptor_count += plan.child_flows.size();
        for (uint32_t child_index = 0;
             child_index < plan.child_flows.size(); ++child_index)
            Require(child_descriptors.count(
                        {plan_index, child_index}) == 1,
                    "collective plan child is missing its lowered descriptor");
    }
    Require(child_descriptors.size() == expected_descriptor_count,
            "collective lowered child descriptor count is inconsistent");

    std::set<uint32_t> expected_local_children;
    auto add_child = [&](uint32_t public_token, uint32_t internal_token) {
        const auto aggregate = candidate_aggregates.find(public_token);
        Require(aggregate != candidate_aggregates.end() &&
                    new_public_tokens.count(public_token) != 0,
                "collective child references an unknown local aggregate");
        Require(candidate_reserved.count(internal_token) == 0,
                "collective child token collides with ordinary DTE token");
        Require(candidate_aggregates.count(internal_token) == 0,
                "collective child token collides with a public token");
        if (candidate_children.size() == capacity_.max_child_tokens)
            throw std::overflow_error(
                "collective child-token capacity exhausted");
        const auto inserted = candidate_children.emplace(
            internal_token, Child{public_token, false, false, false});
        Require(inserted.second,
                "collective child token belongs to more than one aggregate");
        aggregate->second.child_tokens.insert(internal_token);
        expected_local_children.insert(internal_token);
    };

    for (size_t plan_index = 0; plan_index < lowering.plans.size();
         ++plan_index) {
        const IsaV1CollectivePlan &plan = lowering.plans[plan_index];
        for (uint32_t child_index = 0;
             child_index < plan.child_flows.size(); ++child_index) {
            const IsaV1ChildFlow &flow = plan.child_flows[child_index];
            const IsaV1LoweredCollectiveChild &child =
                *child_descriptors.at({plan_index, child_index});
            if (flow.source_core == local_core_) {
                const auto endpoint = endpoint_tokens.find(
                    {plan_index, false});
                Require(endpoint != endpoint_tokens.end() &&
                            endpoint->second == flow.source_public_token,
                        "collective child SEND public token is inconsistent");
                add_child(flow.source_public_token,
                          child.source_internal_token);
            }
            if (flow.destination_core == local_core_) {
                const auto endpoint = endpoint_tokens.find(
                    {plan_index, true});
                Require(endpoint != endpoint_tokens.end() &&
                            endpoint->second ==
                                flow.destination_public_token,
                        "collective child RECEIVE public token is inconsistent");
                add_child(flow.destination_public_token,
                          child.destination_internal_token);
            }
        }
    }

    std::vector<ExpectedAction> expected_actions;
    for (const auto &[plan_index, rank] : local_ranks) {
        const IsaV1CollectivePlan &plan = lowering.plans[plan_index];
        for (const IsaV1Action &action : plan.actions_by_rank[rank])
            expected_actions.push_back({plan_index, plan.key, action});
    }
    const IsaV1CoreCollectiveActionStream *local_stream = nullptr;
    for (const IsaV1CoreCollectiveActionStream &stream :
         lowering.core_actions) {
        if (stream.core_id != local_core_) continue;
        Require(local_stream == nullptr,
                "collective lowering has duplicate local action streams");
        local_stream = &stream;
    }
    if (expected_actions.empty()) {
        Require(local_stream == nullptr || local_stream->actions.empty(),
                "collective lowering has unexpected local actions");
    } else {
        Require(local_stream != nullptr &&
                    local_stream->actions.size() == expected_actions.size(),
                "collective local action stream is incomplete");
    }

    std::set<std::tuple<size_t, CollectiveAggregateLocalWorkKind,
                        uint32_t, uint32_t>>
        local_work_memberships;
    auto add_local_work = [&](uint32_t public_token,
                              const CollectiveKey &key,
                              CollectiveAggregateLocalWorkKind kind,
                              size_t plan_index, uint32_t item_index) {
        const auto aggregate = candidate_aggregates.find(public_token);
        Require(aggregate != candidate_aggregates.end() &&
                    aggregate->second.key == key &&
                    new_public_tokens.count(public_token) != 0,
                "collective local work references an unknown aggregate");
        Require(local_work_memberships.emplace(
                    plan_index, kind, item_index, public_token).second,
                "collective local-work membership is duplicated");
        if (candidate_next_work_id == 0)
            throw std::overflow_error(
                "collective local-work ID space is exhausted");
        if (candidate_local_work.size() ==
            capacity_.max_local_work_items)
            throw std::overflow_error(
                "collective local-work capacity exhausted");
        CollectiveAggregateLocalWorkHandle handle;
        handle.id = candidate_next_work_id;
        handle.public_token = public_token;
        handle.key = key;
        handle.kind = kind;
        handle.plan_index = plan_index;
        handle.item_index = item_index;
        LocalWork work{handle, false, false};
        Require(candidate_local_work.emplace(handle.id, work).second,
                "collective local-work ID is duplicated");
        aggregate->second.local_work_ids.insert(handle.id);
        candidate_next_work_id =
            handle.id == std::numeric_limits<uint64_t>::max()
                ? 0
                : handle.id + 1;
    };

    std::set<uint32_t> observed_local_children;
    if (local_stream != nullptr) {
        for (size_t index = 0; index < expected_actions.size(); ++index) {
            const IsaV1LoweredCollectiveAction &lowered =
                local_stream->actions[index];
            const ExpectedAction &expected = expected_actions[index];
            Require(lowered.plan_index == expected.plan_index &&
                        lowered.key == expected.key &&
                        lowered.action == expected.action &&
                        lowered.action.core == local_core_,
                    "collective local action stream is not canonical");
            const IsaV1CollectivePlan &plan =
                lowering.plans[lowered.plan_index];
            const IsaV1Action &action = lowered.action;
            if (IsSendAction(action.kind) ||
                IsReceiveAction(action.kind)) {
                Require(action.item_index < plan.child_flows.size(),
                        "collective child action index is out of range");
                const IsaV1ChildFlow &flow =
                    plan.child_flows[action.item_index];
                const IsaV1LoweredCollectiveChild &child =
                    *child_descriptors.at(
                        {lowered.plan_index, action.item_index});
                const bool send = IsSendAction(action.kind);
                const uint32_t expected_internal =
                    send ? child.source_internal_token
                         : child.destination_internal_token;
                const uint32_t expected_public =
                    send ? flow.source_public_token
                         : flow.destination_public_token;
                Require((send ? flow.source_core
                              : flow.destination_core) == local_core_ &&
                            lowered.internal_token == expected_internal &&
                            lowered.public_aggregate_token ==
                                expected_public,
                        "collective child action token mapping is inconsistent");
                observed_local_children.insert(expected_internal);
                continue;
            }
            Require(lowered.internal_token == 0 &&
                        lowered.public_aggregate_token == 0,
                    "non-child collective action carries endpoint tokens");
            if (action.kind == IsaV1ActionKind::LOCAL_COPY) {
                Require(action.item_index < plan.local_copies.size(),
                        "collective local-copy action index is out of range");
                const IsaV1LocalCopy &copy =
                    plan.local_copies[action.item_index];
                Require(copy.core == local_core_ &&
                            copy.rank == action.rank,
                        "collective local-copy action target is inconsistent");
                const IsaV1RankRecordContract &contract =
                    plan.rank_records[action.rank];
                Require(contract.send.present && contract.receive.present,
                        "collective local copy lacks endpoint aggregates");
                add_local_work(
                    contract.send.token, plan.key,
                    CollectiveAggregateLocalWorkKind::LOCAL_COPY,
                    lowered.plan_index, action.item_index);
                add_local_work(
                    contract.receive.token, plan.key,
                    CollectiveAggregateLocalWorkKind::LOCAL_COPY,
                    lowered.plan_index, action.item_index);
            } else if (action.kind ==
                       IsaV1ActionKind::REDUCE_COMPUTE) {
                Require(action.item_index < plan.reduce_targets.size(),
                        "collective reduction action index is out of range");
                const IsaV1ReduceTarget &target =
                    plan.reduce_targets[action.item_index];
                Require(target.core == local_core_ &&
                            target.rank == action.rank,
                        "collective reduction action target is inconsistent");
                const IsaV1RankRecordContract &contract =
                    plan.rank_records[action.rank];
                Require(contract.receive.present,
                        "collective reduction lacks a RECEIVE aggregate");
                add_local_work(
                    contract.receive.token, plan.key,
                    CollectiveAggregateLocalWorkKind::REDUCE_COMPUTE,
                    lowered.plan_index, action.item_index);
            } else {
                Require(action.kind ==
                                IsaV1ActionKind::POSTED_BARRIER ||
                            action.kind ==
                                IsaV1ActionKind::COMPLETE_BARRIER,
                        "collective action kind is invalid");
                Require(action.item_index == kIsaV1NoItem,
                        "collective barrier carries an item index");
            }
        }
    }
    Require(observed_local_children == expected_local_children,
            "collective action stream does not cover every local child token");

    for (uint32_t token : new_public_tokens) {
        const Aggregate &aggregate = candidate_aggregates.at(token);
        Require(!aggregate.child_tokens.empty() ||
                    !aggregate.local_work_ids.empty(),
                "zero-child collective aggregate lacks a local-work gate");
    }
    if (candidate_aggregates.size() > capacity_.max_aggregates)
        throw std::overflow_error(
            "collective aggregate capacity exhausted");
    if (candidate_children.size() > capacity_.max_child_tokens)
        throw std::overflow_error(
            "collective child-token capacity exhausted");
    if (candidate_local_work.size() > capacity_.max_local_work_items)
        throw std::overflow_error(
            "collective local-work capacity exhausted");

    aggregates_.swap(candidate_aggregates);
    children_.swap(candidate_children);
    local_work_.swap(candidate_local_work);
    reserved_tokens_.swap(candidate_reserved);
    next_local_work_id_ = candidate_next_work_id;
}

CollectiveAggregateRuntime::AggregateMap::iterator
CollectiveAggregateRuntime::RequireAggregate(uint32_t public_token) {
    const auto found = aggregates_.find(public_token);
    if (found == aggregates_.end())
        throw std::out_of_range("unknown collective aggregate token");
    return found;
}

CollectiveAggregateRuntime::AggregateMap::const_iterator
CollectiveAggregateRuntime::RequireAggregate(uint32_t public_token) const {
    const auto found = aggregates_.find(public_token);
    if (found == aggregates_.end())
        throw std::out_of_range("unknown collective aggregate token");
    return found;
}

bool CollectiveAggregateRuntime::Ready(
    const Aggregate &aggregate) const {
    if (!aggregate.begun) return false;
    for (uint32_t token : aggregate.child_tokens) {
        const auto child = children_.find(token);
        if (child == children_.end())
            throw std::logic_error(
                "collective aggregate lost a child token");
        if (!child->second.local_complete ||
            !child->second.transport_retired)
            return false;
    }
    for (uint64_t id : aggregate.local_work_ids) {
        const auto work = local_work_.find(id);
        if (work == local_work_.end())
            throw std::logic_error(
                "collective aggregate lost a local-work gate");
        if (!work->second.complete) return false;
    }
    return true;
}

void CollectiveAggregateRuntime::Begin(uint32_t public_token) {
    auto aggregate = RequireAggregate(public_token);
    if (aggregate->second.begun)
        throw std::invalid_argument("collective aggregate BEGIN is duplicate");
    for (uint32_t token : aggregate->second.child_tokens) {
        const auto child = children_.find(token);
        if (child == children_.end() || child->second.started)
            throw std::logic_error(
                "collective aggregate child start state is inconsistent");
    }
    for (uint64_t id : aggregate->second.local_work_ids) {
        const auto work = local_work_.find(id);
        if (work == local_work_.end() || work->second.started)
            throw std::logic_error(
                "collective aggregate local-work start state is inconsistent");
    }
    aggregate->second.begun = true;
    for (uint32_t token : aggregate->second.child_tokens)
        children_.at(token).started = true;
    for (uint64_t id : aggregate->second.local_work_ids)
        local_work_.at(id).started = true;
}

void CollectiveAggregateRuntime::MarkChildLocalComplete(
    uint32_t internal_token) {
    const auto child = children_.find(internal_token);
    if (child == children_.end())
        throw std::out_of_range("unknown collective child token");
    if (!child->second.started)
        throw std::invalid_argument(
            "collective child completed before aggregate BEGIN");
    if (child->second.local_complete)
        throw std::invalid_argument(
            "collective child local completion is duplicate");
    child->second.local_complete = true;
}

void CollectiveAggregateRuntime::MarkChildTransportRetired(
    uint32_t internal_token) {
    const auto child = children_.find(internal_token);
    if (child == children_.end())
        throw std::out_of_range("unknown collective child token");
    if (!child->second.started)
        throw std::invalid_argument(
            "collective child retired before aggregate BEGIN");
    if (child->second.transport_retired)
        throw std::invalid_argument(
            "collective child transport retirement is duplicate");
    child->second.transport_retired = true;
}

void CollectiveAggregateRuntime::MarkLocalWorkComplete(
    const CollectiveAggregateLocalWorkHandle &handle) {
    const auto work = local_work_.find(handle.id);
    if (work == local_work_.end())
        throw std::out_of_range("unknown collective local-work handle");
    if (!(work->second.handle == handle))
        throw std::invalid_argument(
            "collective local-work handle metadata is stale");
    if (!work->second.started)
        throw std::invalid_argument(
            "collective local work completed before aggregate BEGIN");
    if (work->second.complete)
        throw std::invalid_argument(
            "collective local-work completion is duplicate");
    work->second.complete = true;
}

void CollectiveAggregateRuntime::Retire(
    AggregateMap::iterator aggregate) {
    for (uint32_t token : aggregate->second.child_tokens)
        children_.erase(token);
    for (uint64_t id : aggregate->second.local_work_ids)
        local_work_.erase(id);
    aggregates_.erase(aggregate);
}

bool CollectiveAggregateRuntime::TryWait(uint32_t public_token) {
    auto aggregate = RequireAggregate(public_token);
    if (!Ready(aggregate->second)) return false;
    Retire(aggregate);
    return true;
}

bool CollectiveAggregateRuntime::TryFence() {
    for (const auto &[token, aggregate] : aggregates_) {
        (void)token;
        if (!Ready(aggregate)) return false;
    }
    aggregates_.clear();
    children_.clear();
    local_work_.clear();
    return true;
}

void CollectiveAggregateRuntime::Cancel(uint32_t public_token) {
    auto aggregate = RequireAggregate(public_token);
    if (aggregate->second.begun)
        throw std::invalid_argument(
            "collective aggregate CANCEL after child start is forbidden");
    for (uint32_t token : aggregate->second.child_tokens) {
        const auto child = children_.find(token);
        if (child == children_.end() || child->second.started)
            throw std::logic_error(
                "collective aggregate CANCEL child state is inconsistent");
    }
    for (uint64_t id : aggregate->second.local_work_ids) {
        const auto work = local_work_.find(id);
        if (work == local_work_.end() || work->second.started)
            throw std::logic_error(
                "collective aggregate CANCEL local-work state is inconsistent");
    }
    Retire(aggregate);
}

CollectiveAggregatePhase CollectiveAggregateRuntime::Poll(
    uint32_t public_token) const {
    const auto aggregate = RequireAggregate(public_token);
    if (!aggregate->second.begun)
        return CollectiveAggregatePhase::REGISTERED;
    return Ready(aggregate->second) ? CollectiveAggregatePhase::READY
                                    : CollectiveAggregatePhase::ACTIVE;
}

std::vector<uint32_t> CollectiveAggregateRuntime::ChildTokens(
    uint32_t public_token) const {
    const auto aggregate = RequireAggregate(public_token);
    return {aggregate->second.child_tokens.begin(),
            aggregate->second.child_tokens.end()};
}

std::vector<CollectiveAggregateLocalWorkHandle>
CollectiveAggregateRuntime::LocalWorks(uint32_t public_token) const {
    const auto aggregate = RequireAggregate(public_token);
    std::vector<CollectiveAggregateLocalWorkHandle> result;
    result.reserve(aggregate->second.local_work_ids.size());
    for (uint64_t id : aggregate->second.local_work_ids)
        result.push_back(local_work_.at(id).handle);
    return result;
}

bool CollectiveAggregateRuntime::HasPublicToken(
    uint32_t public_token) const noexcept {
    return aggregates_.count(public_token) != 0;
}

bool CollectiveAggregateRuntime::HasChildToken(
    uint32_t internal_token) const noexcept {
    return children_.count(internal_token) != 0;
}

CollectiveAggregateResidual
CollectiveAggregateRuntime::Residual() const noexcept {
    return {aggregates_.size(), children_.size(), local_work_.size(),
            reserved_tokens_.size()};
}
