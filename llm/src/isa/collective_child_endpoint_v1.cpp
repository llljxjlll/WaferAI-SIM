#include "isa/collective_child_endpoint_v1.h"

#include "dte/endpoint_contract.h"

#include <algorithm>
#include <limits>
#include <string>

namespace {

void Require(bool condition, const std::string &message) {
    if (!condition) throw RecordLoweringError(message);
}

uint64_t CheckedAdd(uint64_t left, uint64_t right,
                    const std::string &message) {
    if (left > std::numeric_limits<uint64_t>::max() - right)
        throw RecordLoweringError(message);
    return left + right;
}

const IsaV1CoreCollectiveActionStream &FindCoreStream(
    const IsaV1CollectiveArtifactLowering &lowering, uint16_t core) {
    const IsaV1CoreCollectiveActionStream *found = nullptr;
    for (const IsaV1CoreCollectiveActionStream &stream :
         lowering.core_actions) {
        if (stream.core_id != core) continue;
        Require(found == nullptr,
                "ISA-v1 child endpoint lowering has duplicate core streams");
        found = &stream;
    }
    Require(found != nullptr,
            "ISA-v1 child endpoint lowering has no executing-core stream");
    return *found;
}

std::size_t RankForCore(const IsaV1CollectivePlan &plan, uint16_t core) {
    const auto first = std::find(plan.group.begin(), plan.group.end(), core);
    Require(first != plan.group.end(),
            "ISA-v1 child endpoint core is absent from its plan group");
    Require(std::find(first + 1, plan.group.end(), core) == plan.group.end(),
            "ISA-v1 child endpoint plan group contains duplicate cores");
    return static_cast<std::size_t>(first - plan.group.begin());
}

void ValidateCoreStreamShape(
    const IsaV1CollectiveArtifactLowering &lowering,
    const IsaV1CoreCollectiveActionStream &stream) {
    std::size_t cursor = 0;
    for (std::size_t plan_index = 0; plan_index < lowering.plans.size();
         ++plan_index) {
        const IsaV1CollectivePlan &plan = lowering.plans[plan_index];
        const auto rank_it =
            std::find(plan.group.begin(), plan.group.end(), stream.core_id);
        if (rank_it == plan.group.end()) continue;
        Require(std::find(rank_it + 1, plan.group.end(), stream.core_id) ==
                    plan.group.end(),
                "ISA-v1 child endpoint plan group contains duplicate cores");
        const std::size_t rank =
            static_cast<std::size_t>(rank_it - plan.group.begin());
        Require(plan.actions_by_rank.size() == plan.group.size(),
                "ISA-v1 child endpoint plan action-rank table is incomplete");
        for (const IsaV1Action &expected : plan.actions_by_rank[rank]) {
            Require(cursor < stream.actions.size(),
                    "ISA-v1 child endpoint core action stream is truncated");
            const IsaV1LoweredCollectiveAction &actual =
                stream.actions[cursor++];
            Require(actual.plan_index == plan_index &&
                        actual.key == plan.key && actual.action == expected,
                    "ISA-v1 child endpoint core action stream is non-canonical");
        }
    }
    Require(cursor == stream.actions.size(),
            "ISA-v1 child endpoint core action stream has extra actions");
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
                "ISA-v1 child endpoint descriptor is duplicated");
        found = &child;
    }
    Require(found != nullptr,
            "ISA-v1 child endpoint descriptor is missing");
    return *found;
}

void ValidateInternalTokenOwnership(
    const IsaV1CollectiveArtifactLowering &lowering,
    const IsaV1LoweredCollectiveChild &selected) {
    Require(selected.source_internal_token != 0 &&
                selected.destination_internal_token != 0 &&
                selected.source_internal_token !=
                    selected.destination_internal_token,
            "ISA-v1 child endpoint internal tokens are not distinct/non-zero");
    std::size_t source_owners = 0;
    std::size_t destination_owners = 0;
    for (const IsaV1LoweredCollectiveChild &child : lowering.children) {
        source_owners += child.source_internal_token ==
                             selected.source_internal_token;
        source_owners += child.destination_internal_token ==
                             selected.source_internal_token;
        destination_owners += child.source_internal_token ==
                                  selected.destination_internal_token;
        destination_owners += child.destination_internal_token ==
                                  selected.destination_internal_token;
    }
    Require(source_owners == 1 && destination_owners == 1,
            "ISA-v1 child endpoint internal token has multiple owners");

    for (const IsaV1CollectivePlan &plan : lowering.plans) {
        for (const IsaV1RankRecordContract &rank : plan.rank_records) {
            const uint32_t send = rank.send.present ? rank.send.token : 0;
            const uint32_t receive =
                rank.receive.present ? rank.receive.token : 0;
            Require((send == 0 ||
                     (send != selected.source_internal_token &&
                      send != selected.destination_internal_token)) &&
                        (receive == 0 ||
                         (receive != selected.source_internal_token &&
                          receive != selected.destination_internal_token)),
                    "ISA-v1 child endpoint internal token collides with a public token");
        }
    }
}

void ValidateAbsoluteAddress(const SramAddressOperand &address,
                             uint64_t expected,
                             const std::string &which) {
    Require(address.kind == SramAddressKind::ABSOLUTE,
            "ISA-v1 child endpoint " + which +
                " must use an absolute SRAM address");
    Require(address.absolute_address_bytes == expected &&
                address.region_symbol_index == 0 &&
                address.region_offset_bytes == 0,
            "ISA-v1 child endpoint " + which +
                " address does not match its child flow");
}

const IsaV1Wave &ValidateFlowAndWave(
    const IsaV1CollectivePlan &plan, uint32_t child_index,
    const IsaV1ChildFlow &flow) {
    Require(plan.group.size() >= 2 &&
                plan.actions_by_rank.size() == plan.group.size() &&
                plan.rank_records.size() == plan.group.size(),
            "ISA-v1 child endpoint plan rank tables are inconsistent");
    Require(plan.root_rank < plan.group.size(),
            "ISA-v1 child endpoint plan root rank is invalid");
    Require(flow.source_rank < plan.group.size() &&
                flow.destination_rank < plan.group.size() &&
                flow.source_rank != flow.destination_rank &&
                plan.group[flow.source_rank] == flow.source_core &&
                plan.group[flow.destination_rank] == flow.destination_core,
            "ISA-v1 child endpoint flow rank/core pair is invalid");
    Require(flow.fsm_id != 0 && flow.length_bytes != 0 &&
                flow.length_bytes <= kDteEndpointP2pMaxBytes,
            "ISA-v1 child endpoint flow fsm/length is invalid");
    Require(child_index <=
                std::numeric_limits<uint32_t>::max() -
                    plan.logical_fsm_id_base &&
                flow.fsm_id == plan.logical_fsm_id_base + child_index,
            "ISA-v1 child endpoint flow fsm is not canonical");
    Require(flow.chunk_offset_bytes <= plan.length_bytes &&
                flow.length_bytes <=
                    plan.length_bytes - flow.chunk_offset_bytes,
            "ISA-v1 child endpoint flow chunk is outside plan length");

    const IsaV1RankRecordContract &source =
        plan.rank_records[flow.source_rank];
    const IsaV1RankRecordContract &destination =
        plan.rank_records[flow.destination_rank];
    Require(source.send.present && source.send.asynchronous &&
                source.send.token != 0 && destination.receive.present &&
                destination.receive.asynchronous &&
                destination.receive.token != 0,
            "ISA-v1 child endpoint lacks asynchronous public records");
    Require(flow.source_public_token == source.send.token &&
                flow.destination_public_token == destination.receive.token,
            "ISA-v1 child endpoint public token does not match rank records");
    Require(flow.source_address_bytes ==
                CheckedAdd(source.send.base_address_bytes,
                           flow.source_offset_bytes,
                           "ISA-v1 child endpoint source address overflows") &&
                flow.destination_address_bytes ==
                    CheckedAdd(destination.receive.base_address_bytes,
                               flow.destination_offset_bytes,
                               "ISA-v1 child endpoint destination address overflows"),
            "ISA-v1 child endpoint address provenance is inconsistent");

    const IsaV1Wave *wave = nullptr;
    for (const IsaV1Wave &candidate : plan.waves) {
        if (candidate.wave_index != flow.wave_index) continue;
        Require(wave == nullptr,
                "ISA-v1 child endpoint wave index is duplicated");
        wave = &candidate;
    }
    Require(wave != nullptr,
            "ISA-v1 child endpoint flow wave is missing");
    Require(wave->posted_phase_id != wave->complete_phase_id &&
                std::count(wave->child_indices.begin(),
                           wave->child_indices.end(), child_index) == 1,
            "ISA-v1 child endpoint wave membership/phase is invalid");
    return *wave;
}

IsaV1Action ExpectedAction(IsaV1ActionKind kind,
                           const IsaV1ChildFlow &flow,
                           const IsaV1Wave &wave, uint32_t child_index) {
    IsaV1Action expected;
    expected.kind = kind;
    if (kind == IsaV1ActionKind::ISSUE_SEND) {
        expected.rank = flow.source_rank;
        expected.core = flow.source_core;
    } else {
        expected.rank = flow.destination_rank;
        expected.core = flow.destination_core;
    }
    expected.wave_index = flow.wave_index;
    expected.phase_id = wave.posted_phase_id;
    expected.item_index = child_index;
    return expected;
}

const IsaV1LoweredCollectiveAction &ValidateEndpointAction(
    const IsaV1CollectiveArtifactLowering &lowering,
    std::size_t plan_index, const IsaV1CollectivePlan &plan,
    const IsaV1LoweredCollectiveChild &child,
    const IsaV1ChildFlow &flow, const IsaV1Wave &wave,
    IsaV1ActionKind kind) {
    const IsaV1Action expected =
        ExpectedAction(kind, flow, wave, child.child_index);
    Require(expected.rank < plan.actions_by_rank.size() &&
                std::count(plan.actions_by_rank[expected.rank].begin(),
                           plan.actions_by_rank[expected.rank].end(),
                           expected) == 1,
            "ISA-v1 child endpoint action is not unique in its plan");

    const IsaV1CoreCollectiveActionStream &stream =
        FindCoreStream(lowering, expected.core);
    ValidateCoreStreamShape(lowering, stream);
    const IsaV1LoweredCollectiveAction *found = nullptr;
    for (const IsaV1LoweredCollectiveAction &candidate : stream.actions) {
        if (candidate.plan_index != plan_index ||
            candidate.action.kind != kind ||
            candidate.action.item_index != child.child_index)
            continue;
        Require(found == nullptr,
                "ISA-v1 child endpoint action identity is duplicated");
        found = &candidate;
    }
    Require(found != nullptr && found->key == plan.key &&
                found->action == expected,
            "ISA-v1 child endpoint action does not match its canonical pair");

    const bool send = kind == IsaV1ActionKind::ISSUE_SEND;
    Require(found->internal_token ==
                    (send ? child.source_internal_token
                          : child.destination_internal_token) &&
                found->public_aggregate_token ==
                    (send ? flow.source_public_token
                          : flow.destination_public_token),
            "ISA-v1 child endpoint action token mapping is inconsistent");
    return *found;
}

void FillCommon(Dte_endpoint_prim_base &prim,
                const IsaV1ChildFlow &flow, uint32_t token,
                uint16_t peer_core) {
    prim.completion = DteEndpointCompletion::ASYNC;
    prim.datatype = DteEndpointDataType::UINT8;
    prim.reduce_op = DteEndpointReduceOp::NONE;
    prim.fsm_id = flow.fsm_id;
    prim.token = token;
    prim.length_bytes = flow.length_bytes;
    prim.peer_core = peer_core;
    prim.expected_sources = 0;
    prim.tree_id = 0;
    prim.group_id = 0;
    prim.collective_id = 0;
    prim.epoch = 0;
}

} // namespace

std::unique_ptr<Dte_endpoint_prim_base>
MaterializeIsaV1CollectiveChildEndpoint(
    const IsaV1CollectiveArtifactLowering &lowering,
    uint16_t executing_core, std::size_t action_stream_index) {
    const IsaV1CoreCollectiveActionStream &executing_stream =
        FindCoreStream(lowering, executing_core);
    ValidateCoreStreamShape(lowering, executing_stream);
    Require(action_stream_index < executing_stream.actions.size(),
            "ISA-v1 child endpoint action stream index is out of range");
    const IsaV1LoweredCollectiveAction &selected =
        executing_stream.actions[action_stream_index];
    Require(selected.action.kind == IsaV1ActionKind::POST_RECEIVE ||
                selected.action.kind == IsaV1ActionKind::ISSUE_SEND,
            "ISA-v1 child endpoint materializer accepts only POST_RECEIVE or ISSUE_SEND");
    Require(selected.plan_index < lowering.plans.size(),
            "ISA-v1 child endpoint action plan index is out of range");
    const IsaV1CollectivePlan &plan = lowering.plans[selected.plan_index];
    Require(plan.key.group_id != 0 && selected.key == plan.key,
            "ISA-v1 child endpoint action key does not match a collective plan");
    Require(selected.action.core == executing_core &&
                RankForCore(plan, executing_core) == selected.action.rank,
            "ISA-v1 child endpoint action rank/core is inconsistent");
    Require(selected.action.item_index < plan.child_flows.size(),
            "ISA-v1 child endpoint action item is out of range");

    const uint32_t child_index = selected.action.item_index;
    const IsaV1ChildFlow &flow = plan.child_flows[child_index];
    const IsaV1LoweredCollectiveChild &child =
        FindChild(lowering, selected.plan_index, child_index);
    Require(child.source_space == EndpointSourceSpace::SRAM,
            "ISA-v1 child endpoint baseline supports SRAM source space only");
    ValidateInternalTokenOwnership(lowering, child);
    ValidateAbsoluteAddress(child.source, flow.source_address_bytes,
                            "source");
    ValidateAbsoluteAddress(child.destination,
                            flow.destination_address_bytes,
                            "destination");
    const IsaV1Wave &wave = ValidateFlowAndWave(plan, child_index, flow);
    const IsaV1LoweredCollectiveAction &send = ValidateEndpointAction(
        lowering, selected.plan_index, plan, child, flow, wave,
        IsaV1ActionKind::ISSUE_SEND);
    const IsaV1LoweredCollectiveAction &receive = ValidateEndpointAction(
        lowering, selected.plan_index, plan, child, flow, wave,
        IsaV1ActionKind::POST_RECEIVE);
    Require(&selected == (selected.action.kind == IsaV1ActionKind::ISSUE_SEND
                              ? &send
                              : &receive),
            "ISA-v1 child endpoint selected action is not its unique endpoint pair");

    std::unique_ptr<Dte_endpoint_prim_base> prim;
    if (selected.action.kind == IsaV1ActionKind::ISSUE_SEND) {
        auto send_prim = std::make_unique<Dte_send_endpoint_prim>();
        send_prim->mode = DteEndpointSendMode::P2P;
        send_prim->source_space = DteEndpointSourceSpace::SRAM;
        send_prim->source.kind = DteEndpointAddressKind::ABSOLUTE;
        send_prim->source.absolute_address_bytes =
            flow.source_address_bytes;
        FillCommon(*send_prim, flow, child.source_internal_token,
                   flow.destination_core);
        prim = std::move(send_prim);
    } else {
        auto receive_prim = std::make_unique<Dte_recv_endpoint_prim>();
        receive_prim->mode = DteEndpointRecvMode::P2P;
        receive_prim->destination.kind =
            DteEndpointAddressKind::ABSOLUTE;
        receive_prim->destination.absolute_address_bytes =
            flow.destination_address_bytes;
        FillCommon(*receive_prim, flow,
                   child.destination_internal_token, flow.source_core);
        prim = std::move(receive_prim);
    }

    try {
        if (auto *send_prim =
                dynamic_cast<Dte_send_endpoint_prim *>(prim.get()))
            send_prim->Validate();
        else
            dynamic_cast<Dte_recv_endpoint_prim &>(*prim).Validate();
    } catch (const std::exception &error) {
        throw RecordLoweringError(
            std::string("ISA-v1 child endpoint Prim validation: ") +
            error.what());
    }
    return prim;
}
