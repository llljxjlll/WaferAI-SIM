#include "dte/collective_executor_v1.h"

#include "isa/collective_child_endpoint_v1.h"
#include "isa/collective_data_lowering_v1.h"
#include "isa/collective_phase_lowering_v1.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void Require(bool condition, const std::string &message) {
    if (!condition) throw std::invalid_argument(message);
}

bool IsWaitKind(IsaV1CollectiveActionImplementationKind kind) {
    return kind == IsaV1CollectiveActionImplementationKind::WAIT_RECEIVE ||
           kind == IsaV1CollectiveActionImplementationKind::WAIT_SEND ||
           kind ==
               IsaV1CollectiveActionImplementationKind::WAIT_TRANSPORT_RETIRE;
}

} // namespace

CollectiveWaveAdmissionCapacityV1
CollectiveWaveAdmissionCapacityForImageV1(
    const IsaV1CollectiveProgramImage &image,
    uint32_t per_core_sessions,
    uint64_t per_core_receive_bytes) {
    Require(per_core_sessions != 0 && per_core_receive_bytes != 0,
            "collective executor wave capacity must be positive");
    const auto &summary = image.AdmissionCapacity();
    Require(summary.waves != 0 && summary.wave_demands != 0 &&
                summary.max_endpoint_sessions_per_core_wave <=
                    per_core_sessions &&
                summary.max_receive_bytes_per_core_wave <=
                    per_core_receive_bytes,
            "collective image demand exceeds production wave capacity");
    CollectiveWaveAdmissionCapacityV1 capacity;
    capacity.max_plans = image.Lowering().plans.size();
    capacity.max_registered_waves = summary.waves;
    capacity.max_wave_demands = summary.wave_demands;
    capacity.max_pending_waves = summary.waves;
    capacity.max_active_waves = summary.waves;
    capacity.cores.reserve(image.Cores().size());
    for (const IsaV1CollectiveCoreProgramImage &core : image.Cores())
        capacity.cores.push_back(
            {core.core_id, per_core_sessions, per_core_receive_bytes});
    Require(capacity.max_plans != 0 && !capacity.cores.empty(),
            "collective image has no executable wave participants");
    return capacity;
}

bool CollectiveExecutorResidualV1::ActiveEmpty() const noexcept {
    return images == 0 && plans == 0 && runnable_plans == 0 &&
           remaining_actions == 0 && inflight_actions == 0 &&
           required_issue_sites == accepted_issue_sites &&
           arrived_waves == departed_waves &&
           aggregate.ActiveEmpty() && final_gate.Empty();
}

CollectiveLaunchV1Role CollectiveExecutorV1::LaunchRole(
    IsaV1CollectiveRecordRole role) {
    switch (role) {
    case IsaV1CollectiveRecordRole::SEND:
        return CollectiveLaunchV1Role::ISSUE_SEND;
    case IsaV1CollectiveRecordRole::RECEIVE:
        return CollectiveLaunchV1Role::ISSUE_RECEIVE;
    case IsaV1CollectiveRecordRole::REDUCE_COMPUTE:
        return CollectiveLaunchV1Role::DECLARE_REDUCE_COMPUTE;
    }
    throw std::invalid_argument("collective executor issue role is invalid");
}

void CollectiveExecutorV1::Configure(
    std::shared_ptr<const IsaV1CollectiveProgramImage> image,
    uint16_t local_core,
    CollectiveWaveAdmissionCoordinatorV1 *coordinator) {
    Require(!configured_ && !image_ && !aggregate_ && !final_gate_,
            "collective executor is already configured");
    Require(image != nullptr,
            "collective executor image is null");
    Require(coordinator != nullptr,
            "collective executor wave coordinator is null");

    const CollectiveProgramImageIdentityV1 identity =
        CollectiveProgramImageIdentity(*image);
    Require(identity.generation != 0 && identity.cookie != 0,
            "collective executor image identity is invalid");
    Require(coordinator->HasImage() &&
                coordinator->Identity() == identity,
            "collective executor coordinator owns another image");

    const IsaV1CollectiveCoreProgramImage *core_image =
        image->FindCore(local_core);
    Require(core_image != nullptr,
            "collective executor local core is absent from image");

    const IsaV1CoreCollectiveActionStream *core_stream = nullptr;
    for (const IsaV1CoreCollectiveActionStream &stream :
         image->Lowering().core_actions) {
        if (stream.core_id != local_core) continue;
        Require(core_stream == nullptr,
                "collective executor local action stream is duplicated");
        core_stream = &stream;
    }
    Require((core_stream != nullptr || core_image->actions.empty()) &&
                (core_stream == nullptr ||
                 core_stream->actions.size() == core_image->actions.size()),
            "collective executor local action image is incomplete");

    std::map<uint32_t, PlanState> candidate_plans;
    std::size_t expected_begin = 0;
    for (const IsaV1CollectiveActionRange &range :
         core_image->action_ranges) {
        Require(range.plan_index < image->Lowering().plans.size() &&
                    range.begin == expected_begin &&
                    range.count != 0 &&
                    static_cast<std::size_t>(range.begin) + range.count <=
                        core_image->actions.size(),
                "collective executor action range is non-canonical");
        const IsaV1CollectivePlan &plan =
            image->Lowering().plans[range.plan_index];
        Require(range.rank < plan.group.size() &&
                    plan.group[range.rank] == local_core,
                "collective executor action range rank is inconsistent");
        Require(candidate_plans.emplace(range.plan_index,
                                        PlanState{range})
                    .second,
                "collective executor plan range is duplicated");
        expected_begin += range.count;
    }
    Require(expected_begin == core_image->actions.size() &&
                (candidate_plans.empty() ==
                 core_image->actions.empty()),
            "collective executor action ranges do not cover the core stream");
    for (std::size_t index = 0; index < core_image->actions.size(); ++index) {
        const IsaV1CollectiveActionImplementation &implementation =
            core_image->actions[index];
        const IsaV1LoweredCollectiveAction &lowered =
            core_stream->actions[index];
        Require(implementation.core_id == local_core &&
                    implementation.action_stream_index == index &&
                    implementation.plan_index == lowered.plan_index,
                "collective executor action implementation is inconsistent");
    }

    std::map<uint32_t, IsaV1CollectiveIssueSite> candidate_sites;
    std::map<uint32_t, uint32_t> candidate_tokens;
    for (const IsaV1CollectiveIssueSite &site : image->IssueSites()) {
        if (site.core_id != local_core) continue;
        auto plan = candidate_plans.find(site.plan_index);
        Require(plan != candidate_plans.end() &&
                    site.key ==
                        image->Lowering().plans[site.plan_index].key,
                "collective executor issue site references another plan");
        Require(candidate_sites.emplace(site.record_index, site).second,
                "collective executor issue record is duplicated");
        Require(plan->second.required_records.insert(site.record_index).second,
                "collective executor plan issue site is duplicated");
        if (site.role != IsaV1CollectiveRecordRole::REDUCE_COMPUTE) {
            Require(site.public_token != 0 &&
                        plan->second.public_tokens.insert(
                            site.public_token).second &&
                        candidate_tokens.emplace(site.public_token,
                                                 site.plan_index).second,
                    "collective executor public token is duplicated");
        } else {
            Require(site.public_token == 0,
                    "collective executor reduce declaration carries a token");
        }
    }
    for (auto &[plan_index, plan] : candidate_plans) {
        (void)plan_index;
        Require(plan.required_records.empty() ==
                    plan.public_tokens.empty(),
                "collective executor passive plan has partial issue state");
        // A group participant with no external issue record still owns
        // barrier actions. It is locally ready from program start and joins
        // the wave only after issue-bearing peers arrive.
        if (plan.required_records.empty())
            plan.runnable = true;
    }

    const auto &capacity = core_image->capacity.aggregate;
    auto candidate_aggregate =
        std::make_unique<CollectiveAggregateRuntime>(
            local_core,
            CollectiveAggregateCapacity{
                std::max<std::size_t>(1, capacity.aggregate_tokens),
                capacity.child_tokens,
                capacity.local_work_items,
                capacity.ordinary_reserved_tokens});
    candidate_aggregate->RegisterArtifact(
        image->Lowering(), core_image->ordinary_reserved_tokens);
    auto candidate_gate =
        std::make_unique<CollectiveAggregateFinalPhaseGateV1>(
            local_core,
            CollectiveFinalPhaseGateCapacityV1{
                std::max<std::size_t>(1, core_image->capacity.plans),
                std::max<std::size_t>(1, capacity.aggregate_tokens)});
    candidate_gate->RegisterImage(*image, *candidate_aggregate);

    image_ = std::move(image);
    core_image_ = core_image;
    empty_core_stream_ = {local_core, {}};
    core_stream_ = core_stream != nullptr ? core_stream :
                                             &empty_core_stream_;
    coordinator_ = coordinator;
    aggregate_ = std::move(candidate_aggregate);
    final_gate_ = std::move(candidate_gate);
    plans_.swap(candidate_plans);
    sites_by_record_.swap(candidate_sites);
    token_to_plan_.swap(candidate_tokens);
    identity_ = identity;
    local_core_ = local_core;
    next_action_id_ = 1;
    next_plan_cursor_ = 0;
    inflight_.reset();
    aborted_ = false;
    locally_retired_ = false;
    configured_ = true;
}

void CollectiveExecutorV1::RequireReady() const {
    Require(configured_ && !locally_retired_ && !aborted_ &&
                image_ != nullptr && core_image_ != nullptr &&
                core_stream_ != nullptr && coordinator_ != nullptr &&
                aggregate_ != nullptr && final_gate_ != nullptr,
            "collective executor is not active");
}

bool CollectiveExecutorV1::AcceptLaunch(
    const Collective_launch_v1_prim &launch) {
    RequireReady();
    launch.Validate();
    Require(launch.image_generation == identity_.generation &&
                launch.expected_core == local_core_,
            "collective executor launch image/core mismatch");
    const auto expected = sites_by_record_.find(
        launch.external_record_index);
    Require(expected != sites_by_record_.end(),
            "collective executor launch record is unknown");
    const IsaV1CollectiveIssueSite &site = expected->second;
    Require(launch.plan_index == site.plan_index &&
                launch.role == LaunchRole(site.role) &&
                launch.key == site.key &&
                launch.public_token == site.public_token,
            "collective executor launch metadata mismatch");

    PlanState &plan = plans_.at(site.plan_index);
    Require(plan.accepted_records.count(site.record_index) == 0,
            "collective executor launch is duplicate");
    const bool activates =
        plan.accepted_records.size() + 1 ==
        plan.required_records.size();
    if (activates) {
        Require(!plan.runnable,
                "collective executor plan launch is duplicate");
        for (uint32_t token : plan.public_tokens)
            Require(aggregate_->Poll(token) ==
                        CollectiveAggregatePhase::REGISTERED,
                    "collective executor aggregate was already begun");
    }

    plan.accepted_records.insert(site.record_index);
    if (!activates) return false;
    try {
        for (uint32_t token : plan.public_tokens)
            aggregate_->Begin(token);
        plan.runnable = true;
    } catch (...) {
        Abort();
        throw;
    }
    return true;
}

const IsaV1LoweredCollectiveAction &
CollectiveExecutorV1::LoweredAction(uint32_t stream_index) const {
    if (core_stream_ == nullptr ||
        stream_index >= core_stream_->actions.size())
        throw std::out_of_range(
            "collective executor action stream index is invalid");
    return core_stream_->actions[stream_index];
}

bool CollectiveExecutorV1::EnsureWaveActive(
    uint32_t plan_index, const IsaV1Action &action) {
    PlanState &plan = plans_.at(plan_index);
    if (plan.arrived_waves.count(action.wave_index) == 0) {
        coordinator_->Arrive(identity_, plan_index, action.wave_index,
                             local_core_);
        plan.arrived_waves.insert(action.wave_index);
    }
    return coordinator_->Poll(identity_, plan_index,
                              action.wave_index) ==
           CollectiveWaveAdmissionStatusV1::ACTIVE;
}

CollectiveExecutorActionV1 CollectiveExecutorV1::Materialize(
    uint32_t plan_index, uint32_t stream_index,
    const IsaV1LoweredCollectiveAction &lowered,
    const IsaV1CollectiveActionImplementation &implementation) {
    Require(next_action_id_ != 0 &&
                next_action_id_ != std::numeric_limits<uint64_t>::max(),
            "collective executor action ID space is exhausted");
    CollectiveExecutorActionV1 result;
    result.action_id = next_action_id_++;
    result.plan_index = plan_index;
    result.action_stream_index = stream_index;
    result.wave_index = lowered.action.wave_index;
    result.phase_id = lowered.action.phase_id;
    result.canonical_kind = lowered.action.kind;
    result.internal_token = lowered.internal_token;
    result.public_token = lowered.public_aggregate_token;

    Inflight inflight;
    inflight.action_id = result.action_id;
    inflight.plan_index = plan_index;
    inflight.stream_index = stream_index;
    inflight.kind = lowered.action.kind;
    inflight.wave_index = lowered.action.wave_index;
    inflight.phase_id = lowered.action.phase_id;
    inflight.internal_token = lowered.internal_token;
    inflight.public_token = lowered.public_aggregate_token;

    switch (implementation.kind) {
    case IsaV1CollectiveActionImplementationKind::CHILD_ENDPOINT_PRIM:
        result.payload_kind =
            CollectiveExecutorPayloadKindV1::ENDPOINT_PRIM;
        result.prim = MaterializeIsaV1CollectiveChildEndpoint(
            image_->Lowering(), local_core_, stream_index);
        break;
    case IsaV1CollectiveActionImplementationKind::LOCAL_DATA_PRIM: {
        result.payload_kind =
            CollectiveExecutorPayloadKindV1::LOCAL_DATA_PRIM;
        result.prim = MaterializeIsaV1CollectiveDataAction(
            image_->Lowering(), local_core_, stream_index);
        const CollectiveAggregateLocalWorkKind expected_kind =
            lowered.action.kind == IsaV1ActionKind::LOCAL_COPY
                ? CollectiveAggregateLocalWorkKind::LOCAL_COPY
                : CollectiveAggregateLocalWorkKind::REDUCE_COMPUTE;
        const PlanState &plan = plans_.at(plan_index);
        for (uint32_t token : plan.public_tokens) {
            for (const CollectiveAggregateLocalWorkHandle &work :
                 aggregate_->LocalWorks(token)) {
                if (work.plan_index == plan_index &&
                    work.item_index == lowered.action.item_index &&
                    work.kind == expected_kind)
                    inflight.local_work.push_back(work);
            }
        }
        Require(!inflight.local_work.empty(),
                "collective executor data action has no aggregate work");
        break;
    }
    case IsaV1CollectiveActionImplementationKind::PHASE_BARRIER_PRIM:
        result.payload_kind =
            CollectiveExecutorPayloadKindV1::PHASE_BARRIER_PRIM;
        result.prim = MaterializeIsaV1CollectivePhaseBarrier(
            image_->Lowering(), local_core_, stream_index);
        break;
    case IsaV1CollectiveActionImplementationKind::WAIT_RECEIVE:
        result.payload_kind = CollectiveExecutorPayloadKindV1::WAIT;
        result.wait = CollectiveExecutorWaitV1{
            CollectiveExecutorWaitKindV1::RECEIVE_LOCAL_AND_TRANSPORT,
            lowered.internal_token, lowered.public_aggregate_token};
        break;
    case IsaV1CollectiveActionImplementationKind::WAIT_SEND:
        result.payload_kind = CollectiveExecutorPayloadKindV1::WAIT;
        result.wait = CollectiveExecutorWaitV1{
            CollectiveExecutorWaitKindV1::SEND_LOCAL,
            lowered.internal_token, lowered.public_aggregate_token};
        break;
    case IsaV1CollectiveActionImplementationKind::WAIT_TRANSPORT_RETIRE:
        result.payload_kind = CollectiveExecutorPayloadKindV1::WAIT;
        result.wait = CollectiveExecutorWaitV1{
            CollectiveExecutorWaitKindV1::TRANSPORT_RETIRE,
            lowered.internal_token, lowered.public_aggregate_token};
        break;
    }
    Require((IsWaitKind(implementation.kind) && result.wait.has_value() &&
             !result.prim) ||
                (!IsWaitKind(implementation.kind) &&
                 !result.wait.has_value() && result.prim),
            "collective executor action payload is inconsistent");
    inflight_ = std::move(inflight);
    return result;
}

std::optional<CollectiveExecutorActionV1>
CollectiveExecutorV1::NextAction() {
    RequireReady();
    Require(!inflight_.has_value(),
            "collective executor already has an in-flight action");
    if (plans_.empty()) return std::nullopt;

    try {
        for (std::size_t offset = 0; offset < plans_.size(); ++offset) {
            const std::size_t index =
                (next_plan_cursor_ + offset) % plans_.size();
            auto plan_it = plans_.begin();
            std::advance(plan_it, index);
            PlanState &plan = plan_it->second;
            if (!plan.runnable || plan.complete) continue;
            Require(plan.cursor < plan.range.count,
                    "collective executor runnable plan lost its action");
            const uint32_t stream_index =
                plan.range.begin + plan.cursor;
            const IsaV1LoweredCollectiveAction &lowered =
                LoweredAction(stream_index);
            const IsaV1CollectiveActionImplementation &implementation =
                core_image_->actions.at(stream_index);
            Require(lowered.plan_index == plan_it->first &&
                        implementation.plan_index == plan_it->first &&
                        implementation.action_stream_index == stream_index,
                    "collective executor next action is non-canonical");
            if (!EnsureWaveActive(plan_it->first, lowered.action))
                continue;
            next_plan_cursor_ = (index + 1) % plans_.size();
            return Materialize(plan_it->first, stream_index, lowered,
                               implementation);
        }
        return std::nullopt;
    } catch (...) {
        Abort();
        throw;
    }
}

void CollectiveExecutorV1::AdvanceInflight() {
    Require(inflight_.has_value(),
            "collective executor has no in-flight action");
    PlanState &plan = plans_.at(inflight_->plan_index);
    Require(plan.cursor < plan.range.count &&
                plan.range.begin + plan.cursor ==
                    inflight_->stream_index,
            "collective executor action cursor is inconsistent");
    ++plan.cursor;
    if (plan.cursor == plan.range.count)
        plan.complete = true;
    inflight_.reset();
}

void CollectiveExecutorV1::ActionComplete(uint64_t action_id) {
    RequireReady();
    Require(inflight_.has_value() &&
                inflight_->action_id == action_id,
            "collective executor action completion is stale");
    const IsaV1CollectiveActionImplementation &implementation =
        core_image_->actions.at(inflight_->stream_index);
    Require(!IsWaitKind(implementation.kind),
            "collective executor WAIT requires WaitComplete");
    try {
        if (implementation.kind ==
            IsaV1CollectiveActionImplementationKind::LOCAL_DATA_PRIM) {
            for (const CollectiveAggregateLocalWorkHandle &work :
                 inflight_->local_work)
                aggregate_->MarkLocalWorkComplete(work);
        }
        if (inflight_->kind == IsaV1ActionKind::COMPLETE_BARRIER) {
            PlanState &plan = plans_.at(inflight_->plan_index);
            coordinator_->Depart(identity_, inflight_->plan_index,
                                 inflight_->wave_index, local_core_);
            plan.departed_waves.insert(inflight_->wave_index);
            const IsaV1CollectivePlan &lowered_plan =
                image_->Lowering().plans.at(inflight_->plan_index);
            if (inflight_->wave_index + 1 ==
                lowered_plan.waves.size()) {
                final_gate_->MarkFinalComplete(
                    identity_, inflight_->plan_index, lowered_plan.key,
                    local_core_, plan.range.rank, inflight_->phase_id);
            }
        }
        AdvanceInflight();
    } catch (...) {
        Abort();
        throw;
    }
}

bool CollectiveExecutorV1::WaitComplete(
    uint64_t action_id, bool local_complete,
    bool transport_retired) {
    RequireReady();
    Require(inflight_.has_value() &&
                inflight_->action_id == action_id,
            "collective executor wait completion is stale");
    const IsaV1CollectiveActionImplementation &implementation =
        core_image_->actions.at(inflight_->stream_index);
    Require(IsWaitKind(implementation.kind) &&
                inflight_->internal_token != 0 &&
                inflight_->public_token != 0,
            "collective executor action is not a typed WAIT");

    bool ready = false;
    switch (implementation.kind) {
    case IsaV1CollectiveActionImplementationKind::WAIT_RECEIVE:
        ready = local_complete && transport_retired;
        break;
    case IsaV1CollectiveActionImplementationKind::WAIT_SEND:
        ready = local_complete;
        break;
    case IsaV1CollectiveActionImplementationKind::WAIT_TRANSPORT_RETIRE:
        ready = transport_retired;
        break;
    default:
        break;
    }
    if (!ready) return false;

    try {
        if (implementation.kind ==
            IsaV1CollectiveActionImplementationKind::WAIT_RECEIVE) {
            aggregate_->MarkChildLocalComplete(
                inflight_->internal_token);
            aggregate_->MarkChildTransportRetired(
                inflight_->internal_token);
        } else if (implementation.kind ==
                   IsaV1CollectiveActionImplementationKind::WAIT_SEND) {
            aggregate_->MarkChildLocalComplete(
                inflight_->internal_token);
        } else {
            aggregate_->MarkChildTransportRetired(
                inflight_->internal_token);
        }
        AdvanceInflight();
        return true;
    } catch (...) {
        Abort();
        throw;
    }
}

void CollectiveExecutorV1::MaybeRetireLocal() {
    if (locally_retired_ || !configured_) return;
    const bool plans_complete = std::all_of(
        plans_.begin(), plans_.end(),
        [](const auto &entry) { return entry.second.complete; });
    const CollectiveFinalPhaseGateResidualV1 gate =
        final_gate_->Residual();
    if (!plans_complete || gate.public_tokens != 0 ||
        gate.plans != 0 || inflight_.has_value())
        return;
    final_gate_->RetireImage(identity_);
    locally_retired_ = true;
}

bool CollectiveExecutorV1::TryWait(uint32_t public_token) {
    RequireReady();
    Require(token_to_plan_.count(public_token) == 1,
            "collective executor WAIT token is unknown");
    try {
        const bool complete = final_gate_->TryWait(
            identity_, public_token, *aggregate_);
        if (complete) MaybeRetireLocal();
        return complete;
    } catch (...) {
        Abort();
        throw;
    }
}

bool CollectiveExecutorV1::TryFence() {
    RequireReady();
    try {
        const bool complete =
            final_gate_->TryFence(identity_, *aggregate_);
        if (complete) MaybeRetireLocal();
        return complete;
    } catch (...) {
        Abort();
        throw;
    }
}

void CollectiveExecutorV1::Cancel(uint32_t public_token) {
    RequireReady();
    Require(HasPublicToken(public_token),
            "collective executor CANCEL token is unknown");
    Require(aggregate_->Poll(public_token) ==
                CollectiveAggregatePhase::REGISTERED,
            "collective executor CANCEL after plan BEGIN is forbidden");
    try {
        final_gate_->Cancel(identity_, public_token, *aggregate_);
    } catch (...) {
        Abort();
        throw;
    }
    // A collective plan is a whole-artifact operation. Cancelling any one of
    // its not-yet-issued public endpoints terminates this local image and
    // reclaims shared wave admission, rather than leaving its peer roles
    // permanently waiting for a partial plan.
    Abort();
}

bool CollectiveExecutorV1::HasPublicToken(
    uint32_t public_token) const noexcept {
    return configured_ && !aborted_ && aggregate_ != nullptr &&
           token_to_plan_.count(public_token) == 1 &&
           aggregate_->HasPublicToken(public_token);
}

bool CollectiveExecutorV1::HasRunnable() const noexcept {
    if (!configured_ || locally_retired_ || aborted_) return false;
    return std::any_of(plans_.begin(), plans_.end(),
                       [](const auto &entry) {
                           return entry.second.runnable &&
                                  !entry.second.complete;
                       });
}

void CollectiveExecutorV1::Abort() noexcept {
    if (aborted_) return;
    if (coordinator_ != nullptr && coordinator_->HasImage() &&
        coordinator_->Identity() == identity_) {
        try {
            coordinator_->AbortImage(identity_);
        } catch (...) {
        }
    }
    configured_ = false;
    locally_retired_ = true;
    aborted_ = true;
    inflight_.reset();
    plans_.clear();
    sites_by_record_.clear();
    token_to_plan_.clear();
    aggregate_.reset();
    final_gate_.reset();
    image_.reset();
    core_image_ = nullptr;
    core_stream_ = nullptr;
    empty_core_stream_ = {};
    coordinator_ = nullptr;
    identity_ = {};
    next_plan_cursor_ = 0;
}

CollectiveExecutorResidualV1
CollectiveExecutorV1::Residual() const noexcept {
    CollectiveExecutorResidualV1 residual;
    residual.images =
        configured_ && !locally_retired_ ? 1 : 0;
    residual.aborted = aborted_ ? 1 : 0;
    residual.inflight_actions = inflight_.has_value() ? 1 : 0;
    for (const auto &[plan_index, plan] : plans_) {
        (void)plan_index;
        residual.required_issue_sites +=
            plan.required_records.size();
        residual.accepted_issue_sites +=
            plan.accepted_records.size();
        residual.arrived_waves += plan.arrived_waves.size();
        residual.departed_waves += plan.departed_waves.size();
        if (plan.complete) {
            ++residual.completed_plans;
        } else {
            ++residual.plans;
            residual.remaining_actions +=
                plan.range.count - plan.cursor;
            if (plan.runnable) ++residual.runnable_plans;
        }
    }
    if (aggregate_) residual.aggregate = aggregate_->Residual();
    if (final_gate_) residual.final_gate = final_gate_->Residual();
    return residual;
}

bool CollectiveExecutorV1::Drained() const noexcept {
    return Residual().ActiveEmpty();
}
