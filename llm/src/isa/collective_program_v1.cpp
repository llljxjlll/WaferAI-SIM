#include "isa/collective_program_v1.h"

#include "isa/collective_child_endpoint_v1.h"
#include "isa/collective_data_lowering_v1.h"
#include "isa/collective_phase_lowering_v1.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace {

using Wire = std::vector<sc_bv<128>>;

void Require(bool condition, const std::string &message) {
    if (!condition) throw RecordLoweringError(message);
}

template <class T>
uint32_t CheckedU32(T value, const std::string &message) {
    if (value > static_cast<T>(std::numeric_limits<uint32_t>::max()))
        throw RecordLoweringError(message);
    return static_cast<uint32_t>(value);
}

uint64_t CheckedAdd(uint64_t left, uint64_t right,
                    const std::string &message) {
    if (left > std::numeric_limits<uint64_t>::max() - right)
        throw RecordLoweringError(message);
    return left + right;
}

void ValidateConfig(const IsaV1CollectiveProgramImageConfig &config) {
    Require(config.total_cores != 0 && config.cores_per_die != 0 &&
                config.total_cores % config.cores_per_die == 0 &&
                config.total_cores <= UINT16_MAX + 1U,
            "ISA-v1 collective image topology is invalid");
    Require(config.generation != 0,
            "ISA-v1 collective image generation must be non-zero");
    const auto &limits = config.limits;
    Require(limits.max_plans != 0 && limits.max_children != 0 &&
                limits.max_cores != 0 && limits.max_actions != 0 &&
                limits.max_issue_sites != 0 && limits.max_waves != 0 &&
                limits.max_wave_demands != 0 &&
                limits.max_aggregate_tokens_per_core != 0 &&
                limits.max_child_tokens_per_core != 0 &&
                limits.max_local_work_items_per_core != 0 &&
                limits.max_endpoint_sessions_per_core_wave != 0 &&
                limits.max_receive_bytes_per_core_wave != 0,
            "ISA-v1 collective image limits must be positive");
}

bool IsExternalCollective(const ExternalRecord &record) {
    if (std::holds_alternative<ReduceComputeOperands>(record.operands))
        return true;
    if (const auto *send =
            std::get_if<DteSendOperands>(&record.operands))
        return send->mode != DteSendMode::P2P || send->group_id != 0;
    if (const auto *receive =
            std::get_if<DteRecvOperands>(&record.operands))
        return receive->mode != DteRecvMode::P2P ||
               receive->group_id != 0;
    return false;
}

CollectiveKey KeyOf(const ExternalRecord &record) {
    if (const auto *send =
            std::get_if<DteSendOperands>(&record.operands))
        return {static_cast<uint32_t>(send->group_id),
                static_cast<uint32_t>(send->collective_id),
                static_cast<uint32_t>(send->epoch)};
    if (const auto *receive =
            std::get_if<DteRecvOperands>(&record.operands))
        return {static_cast<uint32_t>(receive->group_id),
                static_cast<uint32_t>(receive->collective_id),
                static_cast<uint32_t>(receive->epoch)};
    const auto &compute =
        std::get<ReduceComputeOperands>(record.operands);
    return {static_cast<uint32_t>(compute.group_id),
            static_cast<uint32_t>(compute.collective_id),
            static_cast<uint32_t>(compute.epoch)};
}

IsaV1CollectiveRecordRole RoleOf(const ExternalRecord &record) {
    if (std::holds_alternative<DteSendOperands>(record.operands))
        return IsaV1CollectiveRecordRole::SEND;
    if (std::holds_alternative<DteRecvOperands>(record.operands))
        return IsaV1CollectiveRecordRole::RECEIVE;
    return IsaV1CollectiveRecordRole::REDUCE_COMPUTE;
}

uint32_t TokenOf(const ExternalRecord &record) {
    if (const auto *send =
            std::get_if<DteSendOperands>(&record.operands))
        return static_cast<uint32_t>(send->token);
    if (const auto *receive =
            std::get_if<DteRecvOperands>(&record.operands))
        return static_cast<uint32_t>(receive->token);
    return 0;
}

std::size_t RankForCore(const IsaV1CollectivePlan &plan, uint16_t core) {
    const auto first = std::find(plan.group.begin(), plan.group.end(), core);
    Require(first != plan.group.end(),
            "ISA-v1 collective issue core is absent from its plan group");
    Require(std::find(first + 1, plan.group.end(), core) == plan.group.end(),
            "ISA-v1 collective plan group contains a duplicate core");
    return static_cast<std::size_t>(first - plan.group.begin());
}

bool ExpectedCompute(const IsaV1CollectivePlan &plan, std::size_t rank) {
    if (plan.op == CollOp::REDUCE) return rank == plan.root_rank;
    return plan.op == CollOp::REDUCESCATTER ||
           plan.op == CollOp::ALLREDUCE;
}

const IsaV1LoweredCollectiveChild &FindChild(
    const IsaV1CollectiveArtifactLowering &lowering,
    std::size_t plan_index, uint32_t child_index) {
    const IsaV1LoweredCollectiveChild *found = nullptr;
    for (const IsaV1LoweredCollectiveChild &child : lowering.children) {
        if (child.plan_index != plan_index ||
            child.child_index != child_index)
            continue;
        Require(found == nullptr,
                "ISA-v1 collective image child descriptor is duplicated");
        found = &child;
    }
    Require(found != nullptr,
            "ISA-v1 collective image child descriptor is missing");
    return *found;
}

const IsaV1Wave &FindWave(const IsaV1CollectivePlan &plan,
                          uint16_t wave_index) {
    const IsaV1Wave *found = nullptr;
    for (const IsaV1Wave &wave : plan.waves) {
        if (wave.wave_index != wave_index) continue;
        Require(found == nullptr,
                "ISA-v1 collective image wave index is duplicated");
        found = &wave;
    }
    Require(found != nullptr,
            "ISA-v1 collective image action references an unknown wave");
    return *found;
}

void ValidateRuntimeWait(
    const IsaV1CollectiveArtifactLowering &lowering,
    const IsaV1LoweredCollectiveAction &lowered,
    uint16_t executing_core) {
    Require(lowered.plan_index < lowering.plans.size(),
            "ISA-v1 collective runtime wait plan index is out of range");
    const IsaV1CollectivePlan &plan = lowering.plans[lowered.plan_index];
    const IsaV1Action &action = lowered.action;
    Require(lowered.key == plan.key && action.core == executing_core &&
                action.rank < plan.group.size() &&
                plan.group[action.rank] == executing_core,
            "ISA-v1 collective runtime wait plan/key/core is inconsistent");
    Require(action.item_index < plan.child_flows.size(),
            "ISA-v1 collective runtime wait child index is out of range");
    const IsaV1ChildFlow &flow = plan.child_flows[action.item_index];
    const IsaV1LoweredCollectiveChild &child = FindChild(
        lowering, lowered.plan_index, action.item_index);
    const IsaV1Wave &wave = FindWave(plan, action.wave_index);
    Require(flow.wave_index == action.wave_index &&
                action.phase_id == wave.complete_phase_id,
            "ISA-v1 collective runtime wait phase is non-canonical");
    const bool receive = action.kind == IsaV1ActionKind::WAIT_RECEIVE;
    const bool send = action.kind == IsaV1ActionKind::WAIT_SEND ||
                      action.kind ==
                          IsaV1ActionKind::WAIT_TRANSPORT_RETIRE;
    Require(receive || send,
            "ISA-v1 collective runtime typed action is not a wait");
    Require(action.rank ==
                    (receive ? flow.destination_rank : flow.source_rank) &&
                action.core ==
                    (receive ? flow.destination_core : flow.source_core) &&
                lowered.internal_token ==
                    (receive ? child.destination_internal_token
                             : child.source_internal_token) &&
                lowered.public_aggregate_token ==
                    (receive ? flow.destination_public_token
                             : flow.source_public_token) &&
                lowered.internal_token != 0 &&
                lowered.public_aggregate_token != 0,
            "ISA-v1 collective runtime wait endpoint/token mapping is inconsistent");
    Require(action.rank < plan.actions_by_rank.size() &&
                std::count(plan.actions_by_rank[action.rank].begin(),
                           plan.actions_by_rank[action.rank].end(),
                           action) == 1,
            "ISA-v1 collective runtime wait is not unique in its plan");
}

Wire ValidateChildPrimRoundTrip(
    const IsaV1CollectiveArtifactLowering &lowering, uint16_t core,
    std::size_t action_index) {
    auto prim = MaterializeIsaV1CollectiveChildEndpoint(
        lowering, core, action_index);
    Wire wire = prim->serialize();
    if (dynamic_cast<Dte_send_endpoint_prim *>(prim.get()) != nullptr) {
        Dte_send_endpoint_prim decoded;
        decoded.deserialize(wire);
        Require(decoded.serialize() == wire,
                "ISA-v1 collective image SEND strict roundtrip changed wire");
    } else {
        auto *receive =
            dynamic_cast<Dte_recv_endpoint_prim *>(prim.get());
        Require(receive != nullptr,
                "ISA-v1 collective image endpoint materializer returned an unknown Prim");
        Dte_recv_endpoint_prim decoded;
        decoded.deserialize(wire);
        Require(decoded.serialize() == wire,
                "ISA-v1 collective image RECV strict roundtrip changed wire");
    }
    return wire;
}

Wire ValidateDataPrimRoundTrip(
    const IsaV1CollectiveArtifactLowering &lowering, uint16_t core,
    std::size_t action_index) {
    auto prim = MaterializeIsaV1CollectiveDataAction(
        lowering, core, action_index);
    Wire wire = prim->serialize();
    Collective_data_v1_prim decoded;
    decoded.deserialize(wire);
    Require(decoded.serialize() == wire,
            "ISA-v1 collective image data strict roundtrip changed wire");
    return wire;
}

Wire ValidatePhasePrimRoundTrip(
    const IsaV1CollectiveArtifactLowering &lowering, uint16_t core,
    std::size_t action_index) {
    auto prim = MaterializeIsaV1CollectivePhaseBarrier(
        lowering, core, action_index);
    Wire wire = prim->serialize();
    Collective_phase_barrier_v1_prim decoded;
    decoded.deserialize(wire);
    Require(decoded.serialize() == wire,
            "ISA-v1 collective image phase strict roundtrip changed wire");
    return wire;
}

class CookieBuilder final {
public:
    void Add8(uint8_t value) { AddByte(value); }
    void Add16(uint16_t value) { AddInteger(value, 2); }
    void Add32(uint32_t value) { AddInteger(value, 4); }
    void Add64(uint64_t value) { AddInteger(value, 8); }
    void AddSize(std::size_t value) {
        Add64(static_cast<uint64_t>(value));
    }
    void AddKey(const CollectiveKey &key) {
        Add32(key.group_id);
        Add32(key.collective_id);
        Add32(key.epoch);
    }
    void AddWire(const Wire &wire) {
        AddSize(wire.size());
        for (const sc_bv<128> &segment : wire) {
            for (unsigned int byte = 0; byte < 16; ++byte)
                AddByte(static_cast<uint8_t>(
                    segment.range(byte * 8 + 7, byte * 8).to_uint64()));
        }
    }
    uint64_t Value() const noexcept { return value_; }

private:
    void AddInteger(uint64_t value, unsigned int bytes) {
        for (unsigned int index = 0; index < bytes; ++index)
            AddByte(static_cast<uint8_t>(value >> (index * 8)));
    }
    void AddByte(uint8_t value) {
        value_ ^= value;
        value_ *= 1099511628211ULL;
    }

    uint64_t value_ = 1469598103934665603ULL;
};

bool SameLimits(const IsaV1CollectiveProgramImageLimits &left,
                const IsaV1CollectiveProgramImageLimits &right) {
    return std::tie(
               left.max_plans, left.max_children, left.max_cores,
               left.max_actions, left.max_issue_sites, left.max_waves,
               left.max_wave_demands,
               left.max_aggregate_tokens_per_core,
               left.max_child_tokens_per_core,
               left.max_local_work_items_per_core,
               left.max_reserved_tokens_per_core,
               left.max_endpoint_sessions_per_core_wave,
               left.max_receive_bytes_per_core_wave) ==
           std::tie(
               right.max_plans, right.max_children, right.max_cores,
               right.max_actions, right.max_issue_sites, right.max_waves,
               right.max_wave_demands,
               right.max_aggregate_tokens_per_core,
               right.max_child_tokens_per_core,
               right.max_local_work_items_per_core,
               right.max_reserved_tokens_per_core,
               right.max_endpoint_sessions_per_core_wave,
               right.max_receive_bytes_per_core_wave);
}

} // namespace

bool IsaV1CollectiveActionImplementation::operator==(
    const IsaV1CollectiveActionImplementation &other) const noexcept {
    return std::tie(core_id, action_stream_index, plan_index, kind) ==
           std::tie(other.core_id, other.action_stream_index,
                    other.plan_index, other.kind);
}

bool IsaV1CollectiveActionRange::operator==(
    const IsaV1CollectiveActionRange &other) const noexcept {
    return std::tie(plan_index, rank, begin, count) ==
           std::tie(other.plan_index, other.rank, other.begin, other.count);
}

bool IsaV1CollectiveIssueSite::operator==(
    const IsaV1CollectiveIssueSite &other) const noexcept {
    return std::tie(core_id, record_index, plan_index, role, public_token,
                    key) ==
           std::tie(other.core_id, other.record_index, other.plan_index,
                    other.role, other.public_token, other.key);
}

bool IsaV1CollectiveWaveDemand::operator==(
    const IsaV1CollectiveWaveDemand &other) const noexcept {
    return std::tie(plan_index, wave_index, rank, core_id,
                    endpoint_sessions, receive_bytes) ==
           std::tie(other.plan_index, other.wave_index, other.rank,
                    other.core_id, other.endpoint_sessions,
                    other.receive_bytes);
}

bool IsaV1CollectiveAggregateImageCapacity::operator==(
    const IsaV1CollectiveAggregateImageCapacity &other) const noexcept {
    return std::tie(aggregate_tokens, child_tokens, local_work_items,
                    ordinary_reserved_tokens) ==
           std::tie(other.aggregate_tokens, other.child_tokens,
                    other.local_work_items,
                    other.ordinary_reserved_tokens);
}

bool IsaV1CollectiveCoreImageCapacity::operator==(
    const IsaV1CollectiveCoreImageCapacity &other) const noexcept {
    return std::tie(aggregate, plans, actions, wave_demands,
                    max_wave_endpoint_sessions,
                    max_wave_receive_bytes) ==
           std::tie(other.aggregate, other.plans, other.actions,
                    other.wave_demands,
                    other.max_wave_endpoint_sessions,
                    other.max_wave_receive_bytes);
}

bool IsaV1CollectiveCoreProgramImage::operator==(
    const IsaV1CollectiveCoreProgramImage &other) const noexcept {
    return std::tie(core_id, action_ranges, actions,
                    ordinary_reserved_tokens, capacity) ==
           std::tie(other.core_id, other.action_ranges, other.actions,
                    other.ordinary_reserved_tokens, other.capacity);
}

bool IsaV1CollectiveAdmissionImageCapacity::operator==(
    const IsaV1CollectiveAdmissionImageCapacity &other) const noexcept {
    return std::tie(waves, wave_demands,
                    max_endpoint_sessions_per_core_wave,
                    max_receive_bytes_per_core_wave) ==
           std::tie(other.waves, other.wave_demands,
                    other.max_endpoint_sessions_per_core_wave,
                    other.max_receive_bytes_per_core_wave);
}

const IsaV1CollectiveCoreProgramImage *
IsaV1CollectiveProgramImage::FindCore(uint16_t core_id) const noexcept {
    const auto found = std::lower_bound(
        cores_.begin(), cores_.end(), core_id,
        [](const IsaV1CollectiveCoreProgramImage &core, uint16_t id) {
            return core.core_id < id;
        });
    return found != cores_.end() && found->core_id == core_id
               ? &*found
               : nullptr;
}

bool IsaV1CollectiveProgramImage::operator==(
    const IsaV1CollectiveProgramImage &other) const {
    return generation_ == other.generation_ && cookie_ == other.cookie_ &&
           lowering_ == other.lowering_ && cores_ == other.cores_ &&
           issue_sites_ == other.issue_sites_ &&
           wave_demands_ == other.wave_demands_ &&
           admission_capacity_ == other.admission_capacity_ &&
           SameLimits(limits_, other.limits_);
}

IsaV1CollectiveProgramImage BuildIsaV1CollectiveProgramImage(
    const ProgramArtifact &relocated_artifact,
    const IsaV1CollectiveArtifactLowering &lowering,
    const IsaV1CollectiveProgramImageConfig &config) {
    ValidateConfig(config);
    const IsaV1CollectiveArtifactLowering rebuilt =
        LowerIsaV1CollectiveArtifact(
            relocated_artifact, config.total_cores,
            config.cores_per_die, config.planner_capacity);
    Require(rebuilt == lowering,
            "ISA-v1 collective image artifact/lowering mismatch");
    Require(!lowering.plans.empty(),
            "ISA-v1 collective image requires at least one plan");
    Require(lowering.plans.size() <= config.limits.max_plans &&
                lowering.plans.size() <=
                    std::numeric_limits<uint32_t>::max(),
            "ISA-v1 collective image plan capacity is exhausted");
    Require(lowering.children.size() <= config.limits.max_children,
            "ISA-v1 collective image child capacity is exhausted");

    IsaV1CollectiveProgramImage image;
    image.generation_ = config.generation;
    image.lowering_ = lowering;
    image.limits_ = config.limits;

    std::map<CollectiveKey, uint32_t> plan_by_key;
    for (std::size_t index = 0; index < lowering.plans.size(); ++index) {
        const IsaV1CollectivePlan &plan = lowering.plans[index];
        Require(plan_by_key.emplace(
                    plan.key,
                    CheckedU32(index,
                               "ISA-v1 collective image plan index overflows u32"))
                    .second,
                "ISA-v1 collective image plan key is duplicated");
    }

    using RoleKey =
        std::tuple<uint32_t, uint16_t, IsaV1CollectiveRecordRole>;
    std::set<RoleKey> seen_roles;
    std::map<uint16_t, std::set<uint32_t>> ordinary_issue_tokens;
    std::map<uint16_t, std::set<uint32_t>> control_token_references;
    std::set<uint16_t> artifact_cores;
    for (const ProgramCore &artifact_core : relocated_artifact.cores) {
        Require(artifact_core.core_id <= UINT16_MAX,
                "ISA-v1 collective image core ID exceeds u16");
        const uint16_t core = static_cast<uint16_t>(artifact_core.core_id);
        Require(artifact_cores.insert(core).second,
                "ISA-v1 collective image artifact core is duplicated");
        for (std::size_t record_index = 0;
             record_index < artifact_core.records.size(); ++record_index) {
            const ExternalRecord &record = artifact_core.records[record_index];
            if (IsExternalCollective(record)) {
                const CollectiveKey key = KeyOf(record);
                const auto plan_entry = plan_by_key.find(key);
                Require(plan_entry != plan_by_key.end(),
                        "ISA-v1 collective image issue site has no plan");
                const uint32_t plan_index = plan_entry->second;
                const IsaV1CollectivePlan &plan =
                    lowering.plans[plan_index];
                const std::size_t rank = RankForCore(plan, core);
                const IsaV1CollectiveRecordRole role = RoleOf(record);
                Require(seen_roles.emplace(plan_index, core, role).second,
                        "ISA-v1 collective image issue role is duplicated");
                const uint32_t token = TokenOf(record);
                if (role == IsaV1CollectiveRecordRole::SEND) {
                    Require(rank < plan.rank_records.size() &&
                                plan.rank_records[rank].send.present &&
                                plan.rank_records[rank].send.token == token,
                            "ISA-v1 collective SEND issue site disagrees with its plan");
                } else if (role ==
                           IsaV1CollectiveRecordRole::RECEIVE) {
                    Require(rank < plan.rank_records.size() &&
                                plan.rank_records[rank].receive.present &&
                                plan.rank_records[rank].receive.token == token,
                            "ISA-v1 collective RECEIVE issue site disagrees with its plan");
                } else {
                    Require(token == 0 && ExpectedCompute(plan, rank),
                            "ISA-v1 collective REDUCE_COMPUTE issue site is not a reduction target");
                }
                image.issue_sites_.push_back(
                    {core,
                     CheckedU32(record_index,
                                "ISA-v1 collective issue record index overflows u32"),
                     plan_index, role, token, key});
                continue;
            }

            if (const auto *issue =
                    std::get_if<DteIssueOperands>(&record.operands)) {
                ordinary_issue_tokens[core].insert(
                    static_cast<uint32_t>(issue->token));
            } else if (const auto *send =
                           std::get_if<DteSendOperands>(&record.operands)) {
                if (send->completion == EndpointCompletion::ASYNC)
                    ordinary_issue_tokens[core].insert(
                        static_cast<uint32_t>(send->token));
            } else if (const auto *receive =
                           std::get_if<DteRecvOperands>(&record.operands)) {
                if (receive->completion == EndpointCompletion::ASYNC)
                    ordinary_issue_tokens[core].insert(
                        static_cast<uint32_t>(receive->token));
            } else if (const auto *token =
                           std::get_if<TokenOperands>(&record.operands)) {
                control_token_references[core].insert(
                    static_cast<uint32_t>(token->token));
            }
        }
    }
    Require(image.issue_sites_.size() <= config.limits.max_issue_sites,
            "ISA-v1 collective image issue-site capacity is exhausted");
    std::sort(image.issue_sites_.begin(), image.issue_sites_.end(),
              [](const IsaV1CollectiveIssueSite &left,
                 const IsaV1CollectiveIssueSite &right) {
                  return std::tie(left.core_id, left.record_index,
                                  left.plan_index, left.role) <
                         std::tie(right.core_id, right.record_index,
                                  right.plan_index, right.role);
              });

    std::map<uint16_t, std::set<uint32_t>> public_tokens;
    for (std::size_t plan_index = 0; plan_index < lowering.plans.size();
         ++plan_index) {
        const IsaV1CollectivePlan &plan = lowering.plans[plan_index];
        Require(plan.rank_records.size() == plan.group.size(),
                "ISA-v1 collective image rank-record table is incomplete");
        for (std::size_t rank = 0; rank < plan.group.size(); ++rank) {
            const uint16_t core = plan.group[rank];
            artifact_cores.insert(core);
            const IsaV1RankRecordContract &contract =
                plan.rank_records[rank];
            if (contract.send.present) {
                Require(seen_roles.count(
                            {static_cast<uint32_t>(plan_index), core,
                             IsaV1CollectiveRecordRole::SEND}) == 1,
                        "ISA-v1 collective image is missing a SEND issue site");
                public_tokens[core].insert(contract.send.token);
            }
            if (contract.receive.present) {
                Require(seen_roles.count(
                            {static_cast<uint32_t>(plan_index), core,
                             IsaV1CollectiveRecordRole::RECEIVE}) == 1,
                        "ISA-v1 collective image is missing a RECEIVE issue site");
                public_tokens[core].insert(contract.receive.token);
            }
            Require((seen_roles.count(
                         {static_cast<uint32_t>(plan_index), core,
                          IsaV1CollectiveRecordRole::REDUCE_COMPUTE}) == 1) ==
                        ExpectedCompute(plan, rank),
                    "ISA-v1 collective image reduction issue-site membership is incomplete");
        }
    }
    for (const auto &[core, tokens] : ordinary_issue_tokens) {
        for (uint32_t token : tokens)
            Require(public_tokens[core].count(token) == 0,
                    "ISA-v1 collective public token collides with an ordinary DTE/P2P token");
    }

    std::size_t total_waves = 0;
    for (std::size_t plan_index = 0; plan_index < lowering.plans.size();
         ++plan_index) {
        const IsaV1CollectivePlan &plan = lowering.plans[plan_index];
        Require(plan.waves.size() <=
                    config.limits.max_waves -
                        std::min(config.limits.max_waves, total_waves),
                "ISA-v1 collective image wave capacity is exhausted");
        total_waves += plan.waves.size();
        for (const IsaV1Wave &wave : plan.waves) {
            for (std::size_t rank = 0; rank < plan.group.size(); ++rank) {
                IsaV1CollectiveWaveDemand demand;
                demand.plan_index = static_cast<uint32_t>(plan_index);
                demand.wave_index = wave.wave_index;
                demand.rank = static_cast<uint16_t>(rank);
                demand.core_id = plan.group[rank];
                for (uint32_t child_index : wave.child_indices) {
                    Require(child_index < plan.child_flows.size(),
                            "ISA-v1 collective image wave child index is out of range");
                    const IsaV1ChildFlow &flow =
                        plan.child_flows[child_index];
                    if (flow.source_rank == rank) {
                        Require(demand.endpoint_sessions != UINT32_MAX,
                                "ISA-v1 collective image endpoint-session demand overflows u32");
                        ++demand.endpoint_sessions;
                    }
                    if (flow.destination_rank == rank) {
                        Require(demand.endpoint_sessions != UINT32_MAX,
                                "ISA-v1 collective image endpoint-session demand overflows u32");
                        ++demand.endpoint_sessions;
                        demand.receive_bytes = CheckedAdd(
                            demand.receive_bytes, flow.length_bytes,
                            "ISA-v1 collective image receive-byte demand overflows u64");
                    }
                }
                Require(demand.endpoint_sessions <=
                            config.limits
                                .max_endpoint_sessions_per_core_wave &&
                            demand.receive_bytes <=
                                config.limits
                                    .max_receive_bytes_per_core_wave,
                        "ISA-v1 collective image wave exceeds admission capacity");
                image.wave_demands_.push_back(demand);
            }
        }
    }
    Require(total_waves <= config.limits.max_waves,
            "ISA-v1 collective image wave capacity is exhausted");
    Require(image.wave_demands_.size() <=
                config.limits.max_wave_demands,
            "ISA-v1 collective image wave-demand capacity is exhausted");

    std::map<std::pair<uint16_t, uint32_t>, Wire> action_wires;
    std::map<uint16_t, const IsaV1CoreCollectiveActionStream *> streams;
    for (const IsaV1CoreCollectiveActionStream &stream :
         lowering.core_actions) {
        Require(streams.emplace(stream.core_id, &stream).second,
                "ISA-v1 collective image core action stream is duplicated");
        artifact_cores.insert(stream.core_id);
    }
    Require(artifact_cores.size() <= config.limits.max_cores,
            "ISA-v1 collective image core capacity is exhausted");

    std::size_t total_actions = 0;
    for (uint16_t core_id : artifact_cores) {
        IsaV1CollectiveCoreProgramImage core;
        core.core_id = core_id;
        core.ordinary_reserved_tokens = ordinary_issue_tokens[core_id];
        for (uint32_t token : control_token_references[core_id]) {
            if (public_tokens[core_id].count(token) == 0)
                core.ordinary_reserved_tokens.insert(token);
        }
        Require(core.ordinary_reserved_tokens.size() <=
                    config.limits.max_reserved_tokens_per_core,
                "ISA-v1 collective image ordinary-token capacity is exhausted");

        const auto stream_entry = streams.find(core_id);
        std::size_t local_work_items = 0;
        if (stream_entry != streams.end()) {
            const auto &stream = *stream_entry->second;
            Require(stream.actions.size() <=
                        std::numeric_limits<uint32_t>::max(),
                    "ISA-v1 collective image core action count exceeds u32");
            std::size_t begin = 0;
            while (begin < stream.actions.size()) {
                const std::size_t plan_index =
                    stream.actions[begin].plan_index;
                Require(plan_index < lowering.plans.size(),
                        "ISA-v1 collective image action plan index is invalid");
                std::size_t end = begin + 1;
                while (end < stream.actions.size() &&
                       stream.actions[end].plan_index == plan_index)
                    ++end;
                const IsaV1CollectivePlan &plan =
                    lowering.plans[plan_index];
                const std::size_t rank = RankForCore(plan, core_id);
                Require(std::find_if(
                            stream.actions.begin() + end,
                            stream.actions.end(),
                            [plan_index](
                                const IsaV1LoweredCollectiveAction &action) {
                                return action.plan_index == plan_index;
                            }) == stream.actions.end(),
                        "ISA-v1 collective image plan action range is not contiguous");
                core.action_ranges.push_back(
                    {CheckedU32(plan_index,
                                "ISA-v1 collective image plan index overflows u32"),
                     static_cast<uint16_t>(rank),
                     CheckedU32(begin,
                                "ISA-v1 collective action range begin overflows u32"),
                     CheckedU32(end - begin,
                                "ISA-v1 collective action range count overflows u32")});
                begin = end;
            }

            for (std::size_t action_index = 0;
                 action_index < stream.actions.size(); ++action_index) {
                const IsaV1LoweredCollectiveAction &lowered =
                    stream.actions[action_index];
                IsaV1CollectiveActionImplementation implementation;
                implementation.core_id = core_id;
                implementation.action_stream_index = CheckedU32(
                    action_index,
                    "ISA-v1 collective action stream index overflows u32");
                implementation.plan_index = CheckedU32(
                    lowered.plan_index,
                    "ISA-v1 collective action plan index overflows u32");
                Wire wire;
                switch (lowered.action.kind) {
                case IsaV1ActionKind::POST_RECEIVE:
                case IsaV1ActionKind::ISSUE_SEND:
                    implementation.kind =
                        IsaV1CollectiveActionImplementationKind::
                            CHILD_ENDPOINT_PRIM;
                    wire = ValidateChildPrimRoundTrip(
                        lowering, core_id, action_index);
                    break;
                case IsaV1ActionKind::LOCAL_COPY:
                case IsaV1ActionKind::REDUCE_COMPUTE:
                    implementation.kind =
                        IsaV1CollectiveActionImplementationKind::
                            LOCAL_DATA_PRIM;
                    wire = ValidateDataPrimRoundTrip(
                        lowering, core_id, action_index);
                    local_work_items +=
                        lowered.action.kind == IsaV1ActionKind::LOCAL_COPY
                            ? 2
                            : 1;
                    break;
                case IsaV1ActionKind::POSTED_BARRIER:
                case IsaV1ActionKind::COMPLETE_BARRIER:
                    implementation.kind =
                        IsaV1CollectiveActionImplementationKind::
                            PHASE_BARRIER_PRIM;
                    wire = ValidatePhasePrimRoundTrip(
                        lowering, core_id, action_index);
                    break;
                case IsaV1ActionKind::WAIT_RECEIVE:
                    implementation.kind =
                        IsaV1CollectiveActionImplementationKind::
                            WAIT_RECEIVE;
                    ValidateRuntimeWait(lowering, lowered, core_id);
                    break;
                case IsaV1ActionKind::WAIT_SEND:
                    implementation.kind =
                        IsaV1CollectiveActionImplementationKind::WAIT_SEND;
                    ValidateRuntimeWait(lowering, lowered, core_id);
                    break;
                case IsaV1ActionKind::WAIT_TRANSPORT_RETIRE:
                    implementation.kind =
                        IsaV1CollectiveActionImplementationKind::
                            WAIT_TRANSPORT_RETIRE;
                    ValidateRuntimeWait(lowering, lowered, core_id);
                    break;
                }
                core.actions.push_back(implementation);
                if (!wire.empty())
                    action_wires.emplace(
                        std::make_pair(core_id,
                                       implementation.action_stream_index),
                        std::move(wire));
            }
        }

        core.capacity.plans = core.action_ranges.size();
        core.capacity.actions = core.actions.size();
        core.capacity.aggregate.aggregate_tokens =
            public_tokens[core_id].size();
        core.capacity.aggregate.child_tokens =
            stream_entry == streams.end()
                ? 0
                : [&]() {
                      std::set<uint32_t> tokens;
                      for (const auto &action :
                           stream_entry->second->actions)
                          if (action.internal_token != 0)
                              tokens.insert(action.internal_token);
                      return tokens.size();
                  }();
        core.capacity.aggregate.local_work_items = local_work_items;
        core.capacity.aggregate.ordinary_reserved_tokens =
            core.ordinary_reserved_tokens.size();
        for (const IsaV1CollectiveWaveDemand &demand :
             image.wave_demands_) {
            if (demand.core_id != core_id) continue;
            ++core.capacity.wave_demands;
            core.capacity.max_wave_endpoint_sessions = std::max(
                core.capacity.max_wave_endpoint_sessions,
                demand.endpoint_sessions);
            core.capacity.max_wave_receive_bytes = std::max(
                core.capacity.max_wave_receive_bytes,
                demand.receive_bytes);
        }
        Require(core.capacity.aggregate.aggregate_tokens <=
                    config.limits.max_aggregate_tokens_per_core &&
                    core.capacity.aggregate.child_tokens <=
                        config.limits.max_child_tokens_per_core &&
                    core.capacity.aggregate.local_work_items <=
                        config.limits.max_local_work_items_per_core,
                "ISA-v1 collective image aggregate capacity is exhausted");
        total_actions += core.actions.size();
        Require(total_actions <= config.limits.max_actions,
                "ISA-v1 collective image action capacity is exhausted");
        image.cores_.push_back(std::move(core));
    }

    image.admission_capacity_.waves = total_waves;
    image.admission_capacity_.wave_demands = image.wave_demands_.size();
    for (const IsaV1CollectiveWaveDemand &demand : image.wave_demands_) {
        image.admission_capacity_.max_endpoint_sessions_per_core_wave =
            std::max(
                image.admission_capacity_
                    .max_endpoint_sessions_per_core_wave,
                demand.endpoint_sessions);
        image.admission_capacity_.max_receive_bytes_per_core_wave =
            std::max(
                image.admission_capacity_.max_receive_bytes_per_core_wave,
                demand.receive_bytes);
    }

    std::set<uint32_t> all_reserved;
    for (const auto &core : image.cores_)
        all_reserved.insert(core.ordinary_reserved_tokens.begin(),
                            core.ordinary_reserved_tokens.end());
    for (const IsaV1LoweredCollectiveChild &child : lowering.children) {
        Require(all_reserved.count(child.source_internal_token) == 0 &&
                    all_reserved.count(child.destination_internal_token) == 0,
                "ISA-v1 collective child token collides with an ordinary token");
    }

    CookieBuilder cookie;
    cookie.Add32(0x50364534U); // "P6E4"
    cookie.Add32(config.total_cores);
    cookie.Add32(config.cores_per_die);
    cookie.AddSize(lowering.plans.size());
    for (const IsaV1CollectivePlan &plan : lowering.plans) {
        cookie.Add8(static_cast<uint8_t>(plan.op));
        cookie.AddKey(plan.key);
        cookie.Add16(plan.root_rank);
        cookie.Add64(plan.length_bytes);
        cookie.Add32(plan.logical_fsm_id_base);
        cookie.AddSize(plan.group.size());
        for (uint16_t core : plan.group) cookie.Add16(core);
    }
    cookie.AddSize(image.issue_sites_.size());
    for (const IsaV1CollectiveIssueSite &site : image.issue_sites_) {
        cookie.Add16(site.core_id);
        cookie.Add32(site.record_index);
        cookie.Add32(site.plan_index);
        cookie.Add8(static_cast<uint8_t>(site.role));
        cookie.Add32(site.public_token);
        cookie.AddKey(site.key);
    }
    cookie.AddSize(image.wave_demands_.size());
    for (const IsaV1CollectiveWaveDemand &demand : image.wave_demands_) {
        cookie.Add32(demand.plan_index);
        cookie.Add16(demand.wave_index);
        cookie.Add16(demand.rank);
        cookie.Add16(demand.core_id);
        cookie.Add32(demand.endpoint_sessions);
        cookie.Add64(demand.receive_bytes);
    }
    for (const IsaV1CollectiveCoreProgramImage &core : image.cores_) {
        cookie.Add16(core.core_id);
        cookie.AddSize(core.action_ranges.size());
        for (const IsaV1CollectiveActionRange &range :
             core.action_ranges) {
            cookie.Add32(range.plan_index);
            cookie.Add16(range.rank);
            cookie.Add32(range.begin);
            cookie.Add32(range.count);
        }
        cookie.AddSize(core.actions.size());
        for (const IsaV1CollectiveActionImplementation &implementation :
             core.actions) {
            cookie.Add32(implementation.action_stream_index);
            cookie.Add32(implementation.plan_index);
            cookie.Add8(static_cast<uint8_t>(implementation.kind));
            const auto wire = action_wires.find(
                {core.core_id, implementation.action_stream_index});
            if (wire != action_wires.end()) {
                cookie.Add8(1);
                cookie.AddWire(wire->second);
            } else {
                cookie.Add8(0);
                const auto &action = lowering.core_actions[
                    static_cast<std::size_t>(
                        std::find_if(
                            lowering.core_actions.begin(),
                            lowering.core_actions.end(),
                            [&](const IsaV1CoreCollectiveActionStream &s) {
                                return s.core_id == core.core_id;
                            }) - lowering.core_actions.begin())]
                                         .actions[
                                             implementation
                                                 .action_stream_index];
                cookie.Add8(static_cast<uint8_t>(action.action.kind));
                cookie.Add16(action.action.rank);
                cookie.Add16(action.action.core);
                cookie.Add16(action.action.wave_index);
                cookie.Add16(action.action.phase_id);
                cookie.Add32(action.action.item_index);
                cookie.Add32(action.internal_token);
                cookie.Add32(action.public_aggregate_token);
            }
        }
        cookie.AddSize(core.ordinary_reserved_tokens.size());
        for (uint32_t token : core.ordinary_reserved_tokens)
            cookie.Add32(token);
    }
    image.cookie_ = cookie.Value();
    Require(image.cookie_ != 0,
            "ISA-v1 collective image cookie unexpectedly equals zero");
    return image;
}
