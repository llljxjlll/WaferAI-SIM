#include "isa/collective_child_endpoint_v1.h"
#include "isa/collective_child_endpoint_v1_selftest.h"

#include "dte/coll_plan_v1.h"
#include "utils/prim_utils.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Suite {
    int checks = 0;
    int failures = 0;

    void Check(bool condition, const std::string &name) {
        ++checks;
        if (condition) return;
        ++failures;
        std::cerr << "[COLLECTIVE CHILD ENDPOINT V1] FAIL: "
                  << name << '\n';
    }

    template <class F>
    void Rejects(F &&fn, const std::string &name) {
        bool rejected = false;
        try {
            fn();
        } catch (const RecordLoweringError &) {
            rejected = true;
        } catch (...) {
        }
        Check(rejected, name);
    }
};

class StrictWireScope {
public:
    StrictWireScope()
        : previous_(prim_wire::LegacyCompatibilityEnabled()) {
        prim_wire::SetLegacyCompatibility(false);
    }
    ~StrictWireScope() {
        prim_wire::SetLegacyCompatibility(previous_);
    }

private:
    bool previous_;
};

IsaV1CollectiveSpec MakeRemoteP2pSpec() {
    IsaV1CollectiveSpec spec;
    spec.tx_kind = CollTxKind::UNICAST;
    spec.rx_kind = CollRxKind::UNICAST;
    spec.key = {7, 11, 13};
    spec.group = {2, 9};
    spec.root_rank = 0;
    spec.p2p_source_rank = 0;
    spec.p2p_destination_rank = 1;
    spec.length_bytes = 32;
    spec.logical_fsm_id_base = 0x1200;
    spec.rank_records.resize(2);
    spec.rank_records[0].send = {true, true, 101, 0x1000};
    spec.rank_records[1].receive = {true, true, 202, 0x4000};
    return spec;
}

IsaV1CollectiveArtifactLowering MakeLowering() {
    IsaV1CollectiveArtifactLowering lowering;
    lowering.plans.push_back(
        PlanIsaV1Collective(MakeRemoteP2pSpec()));
    const IsaV1CollectivePlan &plan = lowering.plans.front();
    for (uint32_t index = 0; index < plan.child_flows.size(); ++index) {
        const IsaV1ChildFlow &flow = plan.child_flows[index];
        IsaV1LoweredCollectiveChild child;
        child.plan_index = 0;
        child.child_index = index;
        child.source_internal_token = 0xfffffff0U - index * 2;
        child.destination_internal_token = 0xffffffefU - index * 2;
        child.source_space = EndpointSourceSpace::SRAM;
        child.source.kind = SramAddressKind::ABSOLUTE;
        child.source.absolute_address_bytes = flow.source_address_bytes;
        child.destination.kind = SramAddressKind::ABSOLUTE;
        child.destination.absolute_address_bytes =
            flow.destination_address_bytes;
        lowering.children.push_back(child);
    }
    for (std::size_t rank = 0; rank < plan.group.size(); ++rank) {
        IsaV1CoreCollectiveActionStream stream;
        stream.core_id = plan.group[rank];
        for (const IsaV1Action &action : plan.actions_by_rank[rank]) {
            IsaV1LoweredCollectiveAction lowered;
            lowered.plan_index = 0;
            lowered.key = plan.key;
            lowered.action = action;
            const bool send =
                action.kind == IsaV1ActionKind::ISSUE_SEND ||
                action.kind == IsaV1ActionKind::WAIT_SEND ||
                action.kind == IsaV1ActionKind::WAIT_TRANSPORT_RETIRE;
            const bool receive =
                action.kind == IsaV1ActionKind::POST_RECEIVE ||
                action.kind == IsaV1ActionKind::WAIT_RECEIVE;
            if (send || receive) {
                const auto &child = lowering.children[action.item_index];
                const auto &flow = plan.child_flows[action.item_index];
                lowered.internal_token =
                    send ? child.source_internal_token
                         : child.destination_internal_token;
                lowered.public_aggregate_token =
                    send ? flow.source_public_token
                         : flow.destination_public_token;
            }
            stream.actions.push_back(lowered);
        }
        lowering.core_actions.push_back(std::move(stream));
    }
    return lowering;
}

IsaV1CoreCollectiveActionStream &CoreStream(
    IsaV1CollectiveArtifactLowering &lowering, uint16_t core) {
    const auto found = std::find_if(
        lowering.core_actions.begin(), lowering.core_actions.end(),
        [&](const IsaV1CoreCollectiveActionStream &stream) {
            return stream.core_id == core;
        });
    if (found == lowering.core_actions.end())
        throw std::logic_error("test core stream is missing");
    return *found;
}

const IsaV1CoreCollectiveActionStream &CoreStream(
    const IsaV1CollectiveArtifactLowering &lowering, uint16_t core) {
    const auto found = std::find_if(
        lowering.core_actions.begin(), lowering.core_actions.end(),
        [&](const IsaV1CoreCollectiveActionStream &stream) {
            return stream.core_id == core;
        });
    if (found == lowering.core_actions.end())
        throw std::logic_error("test core stream is missing");
    return *found;
}

std::size_t FindAction(const IsaV1CollectiveArtifactLowering &lowering,
                       uint16_t core, IsaV1ActionKind kind) {
    const auto &stream = CoreStream(lowering, core);
    const auto found = std::find_if(
        stream.actions.begin(), stream.actions.end(),
        [&](const IsaV1LoweredCollectiveAction &action) {
            return action.action.kind == kind;
        });
    if (found == stream.actions.end())
        throw std::logic_error("test action is missing");
    return static_cast<std::size_t>(found - stream.actions.begin());
}

bool SameCommon(const Dte_endpoint_prim_base &left,
                const Dte_endpoint_prim_base &right) {
    return left.completion == right.completion &&
           left.datatype == right.datatype &&
           left.reduce_op == right.reduce_op &&
           left.fsm_id == right.fsm_id && left.token == right.token &&
           left.length_bytes == right.length_bytes &&
           left.peer_core == right.peer_core &&
           left.expected_sources == right.expected_sources &&
           left.tree_id == right.tree_id &&
           left.group_id == right.group_id &&
           left.collective_id == right.collective_id &&
           left.epoch == right.epoch;
}

bool SameAddress(const DteEndpointSramAddress &left,
                 const DteEndpointSramAddress &right) {
    return left.kind == right.kind &&
           left.absolute_address_bytes == right.absolute_address_bytes &&
           left.region == right.region &&
           left.region_offset_bytes == right.region_offset_bytes;
}

void TestMaterializationAndWire(Suite &suite) {
    StrictWireScope strict;
    const auto lowering = MakeLowering();
    const IsaV1ChildFlow &flow = lowering.plans[0].child_flows[0];
    const auto &child = lowering.children[0];
    const std::size_t send_index =
        FindAction(lowering, 2, IsaV1ActionKind::ISSUE_SEND);
    const std::size_t receive_index =
        FindAction(lowering, 9, IsaV1ActionKind::POST_RECEIVE);

    auto send_base = MaterializeIsaV1CollectiveChildEndpoint(
        lowering, 2, send_index);
    auto *send = dynamic_cast<Dte_send_endpoint_prim *>(send_base.get());
    suite.Check(send != nullptr && !lowering.executable &&
                    send->prim_context == nullptr &&
                    send->mode == DteEndpointSendMode::P2P &&
                    send->completion == DteEndpointCompletion::ASYNC &&
                    send->datatype == DteEndpointDataType::UINT8 &&
                    send->reduce_op == DteEndpointReduceOp::NONE &&
                    send->fsm_id == flow.fsm_id &&
                    send->token == child.source_internal_token &&
                    send->length_bytes == flow.length_bytes &&
                    send->peer_core == flow.destination_core &&
                    send->expected_sources == 0 && send->tree_id == 0 &&
                    send->group_id == 0 && send->collective_id == 0 &&
                    send->epoch == 0 &&
                    send->source_space == DteEndpointSourceSpace::SRAM &&
                    send->source.kind ==
                        DteEndpointAddressKind::ABSOLUTE &&
                    send->source.absolute_address_bytes ==
                        flow.source_address_bytes &&
                    send->source.region.empty() &&
                    send->source.region_offset_bytes == 0,
                "ISSUE_SEND maps every field to an untracked strict P2P Prim");
    if (send != nullptr) {
        const auto wire = send->serialize();
        Dte_send_endpoint_prim decoded;
        decoded.deserialize(wire);
        suite.Check(SameCommon(*send, decoded) &&
                        send->mode == decoded.mode &&
                        send->source_space == decoded.source_space &&
                        SameAddress(send->source, decoded.source) &&
                        wire == decoded.serialize(),
                    "DTE_SEND strict wire roundtrip preserves every field");
    }

    auto receive_base = MaterializeIsaV1CollectiveChildEndpoint(
        lowering, 9, receive_index);
    auto *receive =
        dynamic_cast<Dte_recv_endpoint_prim *>(receive_base.get());
    suite.Check(receive != nullptr && !lowering.executable &&
                    receive->prim_context == nullptr &&
                    receive->mode == DteEndpointRecvMode::P2P &&
                    receive->completion == DteEndpointCompletion::ASYNC &&
                    receive->datatype == DteEndpointDataType::UINT8 &&
                    receive->reduce_op == DteEndpointReduceOp::NONE &&
                    receive->fsm_id == flow.fsm_id &&
                    receive->token == child.destination_internal_token &&
                    receive->length_bytes == flow.length_bytes &&
                    receive->peer_core == flow.source_core &&
                    receive->expected_sources == 0 &&
                    receive->tree_id == 0 && receive->group_id == 0 &&
                    receive->collective_id == 0 && receive->epoch == 0 &&
                    receive->destination.kind ==
                        DteEndpointAddressKind::ABSOLUTE &&
                    receive->destination.absolute_address_bytes ==
                        flow.destination_address_bytes &&
                    receive->destination.region.empty() &&
                    receive->destination.region_offset_bytes == 0,
                "POST_RECEIVE maps every field to an untracked strict P2P Prim");
    if (receive != nullptr) {
        const auto wire = receive->serialize();
        Dte_recv_endpoint_prim decoded;
        decoded.deserialize(wire);
        suite.Check(SameCommon(*receive, decoded) &&
                        receive->mode == decoded.mode &&
                        SameAddress(receive->destination,
                                    decoded.destination) &&
                        wire == decoded.serialize(),
                    "DTE_RECV strict wire roundtrip preserves every field");
    }
}

void TestActionGates(Suite &suite) {
    const auto lowering = MakeLowering();
    for (const auto &entry :
         std::vector<std::pair<uint16_t, IsaV1ActionKind>>{
             {2, IsaV1ActionKind::WAIT_SEND},
             {9, IsaV1ActionKind::WAIT_RECEIVE},
             {2, IsaV1ActionKind::WAIT_TRANSPORT_RETIRE}}) {
        suite.Rejects(
            [&] {
                (void)MaterializeIsaV1CollectiveChildEndpoint(
                    lowering, entry.first,
                    FindAction(lowering, entry.first, entry.second));
            },
            "WAIT action is orchestration-only and cannot materialize an endpoint");
    }
    suite.Rejects(
        [&] {
            (void)MaterializeIsaV1CollectiveChildEndpoint(
                lowering, 2, CoreStream(lowering, 2).actions.size());
        },
        "out-of-range action stream index is rejected");
    suite.Rejects(
        [&] {
            (void)MaterializeIsaV1CollectiveChildEndpoint(lowering, 77, 0);
        },
        "unknown executing core is rejected");
}

void TestEnvelopeAndPairRejections(Suite &suite) {
    const auto base = MakeLowering();
    const std::size_t send_index =
        FindAction(base, 2, IsaV1ActionKind::ISSUE_SEND);
    auto rejects_send = [&](IsaV1CollectiveArtifactLowering bad,
                            const std::string &name) {
        suite.Rejects(
            [&] {
                (void)MaterializeIsaV1CollectiveChildEndpoint(
                    bad, 2, send_index);
            }, name);
    };

    {
        auto bad = base;
        bad.core_actions.push_back(CoreStream(bad, 2));
        rejects_send(std::move(bad), "duplicate core stream is rejected");
    }
    {
        auto bad = base;
        CoreStream(bad, 2).actions[send_index].plan_index = 1;
        rejects_send(std::move(bad), "out-of-range action plan is rejected");
    }
    {
        auto bad = base;
        CoreStream(bad, 2).actions[send_index].key.epoch++;
        rejects_send(std::move(bad), "action key mismatch is rejected");
    }
    {
        auto bad = base;
        CoreStream(bad, 2).actions[send_index].action.core = 3;
        rejects_send(std::move(bad), "action core mismatch is rejected");
    }
    {
        auto bad = base;
        CoreStream(bad, 2).actions[send_index].action.rank = 1;
        rejects_send(std::move(bad), "action rank mismatch is rejected");
    }
    {
        auto bad = base;
        CoreStream(bad, 2).actions[send_index].action.wave_index++;
        rejects_send(std::move(bad), "action wave mismatch is rejected");
    }
    {
        auto bad = base;
        CoreStream(bad, 2).actions[send_index].action.phase_id++;
        rejects_send(std::move(bad), "action phase mismatch is rejected");
    }
    {
        auto bad = base;
        CoreStream(bad, 2).actions[send_index].action.item_index++;
        rejects_send(std::move(bad), "action item mismatch is rejected");
    }
    {
        auto bad = base;
        const std::size_t receive_index =
            FindAction(bad, 9, IsaV1ActionKind::POST_RECEIVE);
        CoreStream(bad, 9).actions[receive_index].internal_token++;
        rejects_send(std::move(bad),
                     "paired receive token mismatch is rejected");
    }
    {
        auto bad = base;
        const std::size_t receive_index =
            FindAction(bad, 9, IsaV1ActionKind::POST_RECEIVE);
        CoreStream(bad, 9).actions[receive_index].action.phase_id++;
        rejects_send(std::move(bad),
                     "non-canonical paired receive action is rejected");
    }
}

void TestDescriptorAndFlowRejections(Suite &suite) {
    const auto base = MakeLowering();
    const std::size_t send_index =
        FindAction(base, 2, IsaV1ActionKind::ISSUE_SEND);
    auto rejects_send = [&](IsaV1CollectiveArtifactLowering bad,
                            const std::string &name) {
        suite.Rejects(
            [&] {
                (void)MaterializeIsaV1CollectiveChildEndpoint(
                    bad, 2, send_index);
            }, name);
    };

    {
        auto bad = base;
        bad.children.clear();
        rejects_send(std::move(bad), "missing child descriptor is rejected");
    }
    {
        auto bad = base;
        bad.children.push_back(bad.children.front());
        rejects_send(std::move(bad), "duplicate child descriptor is rejected");
    }
    {
        auto bad = base;
        bad.children[0].source_internal_token =
            bad.children[0].destination_internal_token;
        rejects_send(std::move(bad), "shared child direction token is rejected");
    }
    {
        auto bad = base;
        bad.children[0].source_internal_token =
            bad.plans[0].rank_records[0].send.token;
        CoreStream(bad, 2).actions[send_index].internal_token =
            bad.children[0].source_internal_token;
        const std::size_t wait_index =
            FindAction(bad, 2, IsaV1ActionKind::WAIT_SEND);
        CoreStream(bad, 2).actions[wait_index].internal_token =
            bad.children[0].source_internal_token;
        const std::size_t retire_index = FindAction(
            bad, 2, IsaV1ActionKind::WAIT_TRANSPORT_RETIRE);
        CoreStream(bad, 2).actions[retire_index].internal_token =
            bad.children[0].source_internal_token;
        rejects_send(std::move(bad),
                     "internal/public token collision is rejected");
    }
    {
        auto bad = base;
        bad.children[0].source_space = EndpointSourceSpace::HBM;
        rejects_send(std::move(bad), "HBM child source is rejected by baseline");
    }
    {
        auto bad = base;
        bad.children[0].source.kind = SramAddressKind::REGION;
        bad.children[0].source.absolute_address_bytes = 0;
        bad.children[0].source.region_symbol_index = 1;
        rejects_send(std::move(bad), "region child source is rejected by baseline");
    }
    {
        auto bad = base;
        bad.children[0].destination.absolute_address_bytes++;
        rejects_send(std::move(bad), "child destination/flow mismatch is rejected");
    }
    {
        auto bad = base;
        bad.plans[0].child_flows[0].fsm_id = 0;
        rejects_send(std::move(bad), "zero child fsm is rejected");
    }
    {
        auto bad = base;
        bad.plans[0].child_flows[0].source_core = 9;
        rejects_send(std::move(bad), "child rank/core mismatch is rejected");
    }
    {
        auto bad = base;
        bad.plans[0].child_flows[0].source_public_token++;
        rejects_send(std::move(bad), "child/rank public token mismatch is rejected");
    }
    {
        auto bad = base;
        bad.plans[0].rank_records[0].send.base_address_bytes++;
        rejects_send(std::move(bad), "child address provenance mismatch is rejected");
    }
    {
        auto bad = base;
        bad.plans[0].waves[0].child_indices.clear();
        rejects_send(std::move(bad), "child wave membership mismatch is rejected");
    }
}

} // namespace

int RunIsaV1CollectiveChildEndpointSelfTest() {
    Suite suite;
    TestMaterializationAndWire(suite);
    TestActionGates(suite);
    TestEnvelopeAndPairRejections(suite);
    TestDescriptorAndFlowRejections(suite);
    if (suite.failures == 0) {
        std::cout << "[COLLECTIVE CHILD ENDPOINT V1] " << suite.checks
                  << " checks passed\n";
    }
    return suite.failures;
}

#ifdef ISA_V1_COLLECTIVE_CHILD_ENDPOINT_SELFTEST_MAIN
int sc_main(int, char **) {
    return RunIsaV1CollectiveChildEndpointSelfTest() == 0 ? 0 : 1;
}
#endif
