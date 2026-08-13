#include "dte/collective_final_phase_gate_v1.h"

#include <algorithm>
#include <limits>
#include <vector>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void Require(bool condition, const std::string &message) {
    if (!condition) throw std::invalid_argument(message);
}

std::size_t RankForCore(const IsaV1CollectivePlan &plan,
                        uint16_t core) {
    const auto found =
        std::find(plan.group.begin(), plan.group.end(), core);
    if (found == plan.group.end())
        throw std::out_of_range(
            "collective final phase local core is absent from plan");
    Require(std::find(found + 1, plan.group.end(), core) ==
                plan.group.end(),
            "collective final phase plan contains duplicate core");
    return static_cast<std::size_t>(found - plan.group.begin());
}

} // namespace

bool CollectiveFinalPhaseGateResidualV1::Empty() const noexcept {
    return images == 0 && plans == 0 && public_tokens == 0 &&
           completed_plans == 0;
}

CollectiveAggregateFinalPhaseGateV1::
CollectiveAggregateFinalPhaseGateV1(
    uint16_t local_core,
    CollectiveFinalPhaseGateCapacityV1 capacity)
    : local_core_(local_core), capacity_(capacity) {
    Require(capacity_.max_plans != 0 &&
                capacity_.max_public_tokens != 0,
            "collective final phase gate capacity must be positive");
}

void CollectiveAggregateFinalPhaseGateV1::RegisterImage(
    const IsaV1CollectiveProgramImage &image,
    const CollectiveAggregateRuntime &aggregate) {
    Require(!registered_,
            "collective final phase gate already owns an image");
    Require(aggregate.LocalCore() == local_core_,
            "collective final phase aggregate belongs to another core");
    const CollectiveProgramImageIdentityV1 identity =
        CollectiveProgramImageIdentity(image);
    Require(identity.generation != 0 && identity.cookie != 0,
            "collective final phase image identity is invalid");

    std::map<uint32_t, PlanGate> candidate_plans;
    std::map<uint32_t, TokenGate> candidate_tokens;
    for (std::size_t plan_index = 0;
         plan_index < image.Lowering().plans.size(); ++plan_index) {
        const IsaV1CollectivePlan &plan =
            image.Lowering().plans[plan_index];
        const auto core =
            std::find(plan.group.begin(), plan.group.end(), local_core_);
        if (core == plan.group.end()) continue;
        Require(plan_index <= std::numeric_limits<uint32_t>::max() &&
                    !plan.waves.empty(),
                "collective final phase plan metadata is invalid");
        const std::size_t rank = RankForCore(plan, local_core_);
        Require(rank < plan.rank_records.size(),
                "collective final phase rank contract is missing");
        PlanGate gate;
        gate.key = plan.key;
        gate.rank = static_cast<uint16_t>(rank);
        gate.final_phase_id = plan.waves.back().complete_phase_id;
        Require(candidate_plans
                    .emplace(static_cast<uint32_t>(plan_index), gate)
                    .second,
                "collective final phase plan is duplicated");
    }

    for (const IsaV1CollectiveIssueSite &site :
         image.IssueSites()) {
        if (site.core_id != local_core_ ||
            site.role ==
                IsaV1CollectiveRecordRole::REDUCE_COMPUTE)
            continue;
        const auto plan = candidate_plans.find(site.plan_index);
        Require(plan != candidate_plans.end() &&
                    site.key == plan->second.key &&
                    site.public_token != 0,
                "collective final phase issue site is inconsistent");
        Require(candidate_tokens
                    .emplace(site.public_token,
                             TokenGate{site.plan_index})
                    .second,
                "collective final phase public token is duplicated");
        ++plan->second.outstanding_tokens;
    }
    Require(candidate_plans.size() <= capacity_.max_plans &&
                candidate_tokens.size() <=
                    capacity_.max_public_tokens,
            "collective final phase gate capacity is exhausted");
    Require(aggregate.Residual().aggregates ==
                candidate_tokens.size(),
            "collective final phase aggregate token count does not match image");
    for (const auto &[token, gate] : candidate_tokens) {
        (void)gate;
        Require(aggregate.HasPublicToken(token),
                "collective final phase aggregate is missing an image token");
    }

    plans_.swap(candidate_plans);
    tokens_.swap(candidate_tokens);
    identity_ = identity;
    registered_ = true;
}

void CollectiveAggregateFinalPhaseGateV1::RequireIdentity(
    const CollectiveProgramImageIdentityV1 &identity) const {
    Require(registered_ && identity == identity_,
            "collective final phase image identity is stale or unknown");
}

const CollectiveAggregateFinalPhaseGateV1::TokenGate &
CollectiveAggregateFinalPhaseGateV1::RequireToken(
    uint32_t public_token) const {
    const auto token = tokens_.find(public_token);
    if (token == tokens_.end())
        throw std::out_of_range(
            "collective final phase public token is unknown");
    return token->second;
}

void CollectiveAggregateFinalPhaseGateV1::RequireAggregateBinding(
    const CollectiveAggregateRuntime &aggregate) const {
    Require(aggregate.LocalCore() == local_core_ &&
                aggregate.Residual().aggregates == tokens_.size(),
            "collective final phase aggregate binding is inconsistent");
    for (const auto &[token, gate] : tokens_) {
        (void)gate;
        Require(aggregate.HasPublicToken(token),
                "collective final phase aggregate lost a public token");
    }
}

void CollectiveAggregateFinalPhaseGateV1::MarkFinalComplete(
    const CollectiveProgramImageIdentityV1 &identity,
    uint32_t plan_index, const CollectiveKey &key,
    uint16_t executing_core, uint16_t rank, uint16_t phase_id) {
    RequireIdentity(identity);
    const auto found = plans_.find(plan_index);
    if (found == plans_.end())
        throw std::out_of_range(
            "collective final phase plan is unknown");
    PlanGate &plan = found->second;
    Require(executing_core == local_core_ && rank == plan.rank &&
                key == plan.key &&
                phase_id == plan.final_phase_id,
            "collective final phase completion metadata is inconsistent");
    Require(!plan.final_complete,
            "collective final phase completion is duplicate");
    plan.final_complete = true;
    if (plan.outstanding_tokens == 0)
        plans_.erase(found);
}

CollectiveFinalPhaseGateStatusV1
CollectiveAggregateFinalPhaseGateV1::Poll(
    const CollectiveProgramImageIdentityV1 &identity,
    uint32_t public_token,
    const CollectiveAggregateRuntime &aggregate) const {
    RequireIdentity(identity);
    RequireAggregateBinding(aggregate);
    const TokenGate &token = RequireToken(public_token);
    const CollectiveAggregatePhase aggregate_phase =
        aggregate.Poll(public_token);
    if (aggregate_phase != CollectiveAggregatePhase::READY)
        return CollectiveFinalPhaseGateStatusV1::WAITING_AGGREGATE;
    const auto plan = plans_.find(token.plan_index);
    if (plan == plans_.end())
        throw std::logic_error(
            "collective final phase token lost its plan");
    return plan->second.final_complete
               ? CollectiveFinalPhaseGateStatusV1::READY
               : CollectiveFinalPhaseGateStatusV1::
                     WAITING_FINAL_PHASE;
}

void CollectiveAggregateFinalPhaseGateV1::ConsumeToken(
    uint32_t public_token) {
    const auto token = tokens_.find(public_token);
    if (token == tokens_.end())
        throw std::logic_error(
            "collective final phase consumed an unknown token");
    const auto plan = plans_.find(token->second.plan_index);
    if (plan == plans_.end() ||
        plan->second.outstanding_tokens == 0)
        throw std::logic_error(
            "collective final phase token accounting is inconsistent");
    --plan->second.outstanding_tokens;
    tokens_.erase(token);
    if (plan->second.outstanding_tokens == 0)
        plans_.erase(plan);
}

bool CollectiveAggregateFinalPhaseGateV1::TryWait(
    const CollectiveProgramImageIdentityV1 &identity,
    uint32_t public_token,
    CollectiveAggregateRuntime &aggregate) {
    if (Poll(identity, public_token, aggregate) !=
        CollectiveFinalPhaseGateStatusV1::READY)
        return false;
    if (!aggregate.TryWait(public_token)) return false;
    ConsumeToken(public_token);
    return true;
}

bool CollectiveAggregateFinalPhaseGateV1::TryFence(
    const CollectiveProgramImageIdentityV1 &identity,
    CollectiveAggregateRuntime &aggregate) {
    RequireIdentity(identity);
    RequireAggregateBinding(aggregate);
    for (const auto &[plan_index, plan] : plans_) {
        (void)plan_index;
        if (!plan.final_complete) return false;
    }
    if (!aggregate.TryFence()) return false;
    tokens_.clear();
    plans_.clear();
    return true;
}

void CollectiveAggregateFinalPhaseGateV1::Cancel(
    const CollectiveProgramImageIdentityV1 &identity,
    uint32_t public_token,
    CollectiveAggregateRuntime &aggregate) {
    RequireIdentity(identity);
    RequireAggregateBinding(aggregate);
    (void)RequireToken(public_token);
    aggregate.Cancel(public_token);
    ConsumeToken(public_token);
}

void CollectiveAggregateFinalPhaseGateV1::AbortRegistered(
    const CollectiveProgramImageIdentityV1 &identity,
    CollectiveAggregateRuntime &aggregate) {
    RequireIdentity(identity);
    RequireAggregateBinding(aggregate);
    for (const auto &[token, gate] : tokens_) {
        (void)gate;
        Require(aggregate.Poll(token) ==
                    CollectiveAggregatePhase::REGISTERED,
                "collective final phase cannot abort an active aggregate");
    }
    std::vector<uint32_t> tokens;
    tokens.reserve(tokens_.size());
    for (const auto &[token, gate] : tokens_) {
        (void)gate;
        tokens.push_back(token);
    }
    for (uint32_t token : tokens)
        aggregate.Cancel(token);
    tokens_.clear();
    plans_.clear();
}

void CollectiveAggregateFinalPhaseGateV1::RetireImage(
    const CollectiveProgramImageIdentityV1 &identity) {
    RequireIdentity(identity);
    Require(tokens_.empty() && plans_.empty(),
            "collective final phase image still has aggregate gates");
    registered_ = false;
    identity_ = {};
}

CollectiveFinalPhaseGateResidualV1
CollectiveAggregateFinalPhaseGateV1::Residual() const noexcept {
    CollectiveFinalPhaseGateResidualV1 residual;
    residual.images = registered_ ? 1 : 0;
    residual.plans = plans_.size();
    residual.public_tokens = tokens_.size();
    for (const auto &[plan_index, plan] : plans_) {
        (void)plan_index;
        if (plan.final_complete) ++residual.completed_plans;
    }
    return residual;
}
