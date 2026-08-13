#include "dte/collective_aggregate_v1_selftest.h"

#include "dte/collective_aggregate_v1.h"
#include "isa/record_lowering.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

class Checks {
public:
    void Check(bool condition, const std::string &name) {
        ++result.checks;
        if (!condition) result.failures.push_back(name);
    }

    template <class F>
    void Reject(const std::string &name, F &&fn) {
        ++result.checks;
        try {
            fn();
            result.failures.push_back(name + ": unexpectedly accepted");
        } catch (const std::exception &) {
        }
    }

    CollectiveAggregateV1SelfTestResult result;
};

CollectiveAggregateCapacity Capacity(
    size_t aggregates = 32, size_t children = 128,
    size_t local_work = 64, size_t reserved = 64) {
    return {aggregates, children, local_work, reserved};
}

IsaV1EndpointRecord Endpoint(uint32_t token, uint64_t address) {
    IsaV1EndpointRecord endpoint;
    endpoint.present = true;
    endpoint.asynchronous = true;
    endpoint.token = token;
    endpoint.base_address_bytes = address;
    return endpoint;
}

IsaV1CollectivePlan RemoteP2p(uint32_t epoch, uint32_t send_token,
                              uint32_t receive_token,
                              uint32_t fsm_id = 0x1000) {
    IsaV1CollectiveSpec spec;
    spec.tx_kind = CollTxKind::UNICAST;
    spec.rx_kind = CollRxKind::UNICAST;
    spec.key = {7, 9, epoch};
    spec.group = {0, 1};
    spec.p2p_source_rank = 0;
    spec.p2p_destination_rank = 1;
    spec.length_bytes = 16;
    spec.logical_fsm_id_base = fsm_id;
    spec.rank_records.resize(2);
    spec.rank_records[0].send = Endpoint(send_token, 0x1000);
    spec.rank_records[1].receive = Endpoint(receive_token, 0x2000);
    return PlanIsaV1Collective(spec);
}

IsaV1CollectivePlan LocalP2p(uint32_t send_token,
                             uint32_t receive_token) {
    IsaV1CollectiveSpec spec;
    spec.tx_kind = CollTxKind::UNICAST;
    spec.rx_kind = CollRxKind::UNICAST;
    spec.key = {8, 3, 0};
    spec.group = {0};
    spec.p2p_source_rank = 0;
    spec.p2p_destination_rank = 0;
    spec.length_bytes = 16;
    spec.logical_fsm_id_base = 0x2000;
    spec.rank_records.resize(1);
    spec.rank_records[0].send = Endpoint(send_token, 0x3000);
    spec.rank_records[0].receive = Endpoint(receive_token, 0x4000);
    return PlanIsaV1Collective(spec);
}

IsaV1CollectivePlan N1Reduce(uint32_t send_token,
                             uint32_t receive_token) {
    IsaV1CollectiveSpec spec;
    spec.tx_kind = CollTxKind::UNICAST;
    spec.rx_kind = CollRxKind::REDUCE;
    spec.key = {8, 4, 0};
    spec.group = {0};
    spec.root_rank = 0;
    spec.length_bytes = 16;
    spec.dtype = CollDType::INT32;
    spec.reduce_op = CollReduceOp::SUM;
    spec.expected_sources = 0;
    spec.logical_fsm_id_base = 0x2100;
    spec.rank_records.resize(1);
    spec.rank_records[0].send = Endpoint(send_token, 0x5000);
    spec.rank_records[0].receive = Endpoint(receive_token, 0x6000);
    spec.rank_records[0].result_address_bytes = 0x7000;
    return PlanIsaV1Collective(spec);
}

IsaV1CollectivePlan Reduce2() {
    IsaV1CollectiveSpec spec;
    spec.tx_kind = CollTxKind::UNICAST;
    spec.rx_kind = CollRxKind::REDUCE;
    spec.key = {8, 5, 0};
    spec.group = {0, 1};
    spec.root_rank = 0;
    spec.length_bytes = 16;
    spec.dtype = CollDType::INT32;
    spec.reduce_op = CollReduceOp::SUM;
    spec.expected_sources = 1;
    spec.logical_fsm_id_base = 0x2200;
    spec.rank_records.resize(2);
    spec.rank_records[0].send = Endpoint(35, 0x8000);
    spec.rank_records[0].receive = Endpoint(36, 0x9000);
    spec.rank_records[0].result_address_bytes = 0xa000;
    spec.rank_records[1].send = Endpoint(37, 0xb000);
    return PlanIsaV1Collective(spec);
}

IsaV1CollectivePlan Scatter3() {
    IsaV1CollectiveSpec spec;
    spec.tx_kind = CollTxKind::SCATTER;
    spec.rx_kind = CollRxKind::UNICAST;
    spec.key = {10, 1, 0};
    spec.group = {0, 1, 2};
    spec.root_rank = 0;
    spec.length_bytes = 16;
    spec.logical_fsm_id_base = 0x3000;
    spec.rank_records.resize(3);
    spec.rank_records[0].send = Endpoint(41, 0x8000);
    spec.rank_records[0].receive = Endpoint(42, 0x9000);
    spec.rank_records[1].receive = Endpoint(43, 0xa000);
    spec.rank_records[2].receive = Endpoint(44, 0xb000);
    return PlanIsaV1Collective(spec);
}

IsaV1CollectiveArtifactLowering MakeLowering(
    std::vector<IsaV1CollectivePlan> plans,
    uint32_t first_internal_token = 1000) {
    IsaV1CollectiveArtifactLowering lowering;
    lowering.plans = std::move(plans);
    std::vector<std::vector<size_t>> child_lookup(
        lowering.plans.size());
    uint32_t next_token = first_internal_token;
    for (size_t plan_index = 0; plan_index < lowering.plans.size();
         ++plan_index) {
        const IsaV1CollectivePlan &plan = lowering.plans[plan_index];
        for (uint32_t child_index = 0;
             child_index < plan.child_flows.size(); ++child_index) {
            if (next_token > UINT32_MAX - 2)
                throw std::overflow_error("selftest child token overflow");
            IsaV1LoweredCollectiveChild child;
            child.plan_index = plan_index;
            child.child_index = child_index;
            child.source_internal_token = next_token++;
            child.destination_internal_token = next_token++;
            child_lookup[plan_index].push_back(lowering.children.size());
            lowering.children.push_back(child);
        }
    }

    std::map<uint16_t, std::vector<IsaV1LoweredCollectiveAction>> by_core;
    for (size_t plan_index = 0; plan_index < lowering.plans.size();
         ++plan_index) {
        const IsaV1CollectivePlan &plan = lowering.plans[plan_index];
        for (const auto &rank_actions : plan.actions_by_rank) {
            for (const IsaV1Action &action : rank_actions) {
                IsaV1LoweredCollectiveAction lowered;
                lowered.plan_index = plan_index;
                lowered.key = plan.key;
                lowered.action = action;
                const bool send =
                    action.kind == IsaV1ActionKind::ISSUE_SEND ||
                    action.kind == IsaV1ActionKind::WAIT_SEND ||
                    action.kind ==
                        IsaV1ActionKind::WAIT_TRANSPORT_RETIRE;
                const bool receive =
                    action.kind == IsaV1ActionKind::POST_RECEIVE ||
                    action.kind == IsaV1ActionKind::WAIT_RECEIVE;
                if (send || receive) {
                    const IsaV1LoweredCollectiveChild &child =
                        lowering.children[child_lookup[plan_index]
                                                   [action.item_index]];
                    const IsaV1ChildFlow &flow =
                        plan.child_flows[action.item_index];
                    lowered.internal_token =
                        send ? child.source_internal_token
                             : child.destination_internal_token;
                    lowered.public_aggregate_token =
                        send ? flow.source_public_token
                             : flow.destination_public_token;
                }
                by_core[action.core].push_back(std::move(lowered));
            }
        }
    }
    for (auto &entry : by_core)
        lowering.core_actions.push_back(
            {entry.first, std::move(entry.second)});
    return lowering;
}

void CompleteAggregate(CollectiveAggregateRuntime &runtime,
                       uint32_t public_token) {
    runtime.Begin(public_token);
    for (uint32_t child : runtime.ChildTokens(public_token)) {
        runtime.MarkChildLocalComplete(child);
        runtime.MarkChildTransportRetired(child);
    }
    for (const auto &work : runtime.LocalWorks(public_token))
        runtime.MarkLocalWorkComplete(work);
}

void CheckTwoGateWaitAndStale(Checks &checks) {
    const auto lowering = MakeLowering({RemoteP2p(0, 11, 12)});
    CollectiveAggregateRuntime runtime(0, Capacity());
    runtime.RegisterArtifact(lowering, {77});
    const std::vector<uint32_t> children = runtime.ChildTokens(11);
    checks.Check(children.size() == 1 &&
                     runtime.Poll(11) ==
                         CollectiveAggregatePhase::REGISTERED &&
                     runtime.Residual().reserved_tokens == 1,
                 "registration derives one deterministic local child");
    checks.Reject("child completion before BEGIN", [&] {
        runtime.MarkChildLocalComplete(children.front());
    });
    runtime.Begin(11);
    checks.Reject("duplicate BEGIN", [&] { runtime.Begin(11); });
    runtime.MarkChildTransportRetired(children.front());
    checks.Reject("duplicate child transport retirement", [&] {
        runtime.MarkChildTransportRetired(children.front());
    });
    checks.Check(!runtime.TryWait(11) &&
                     runtime.Poll(11) == CollectiveAggregatePhase::ACTIVE,
                 "transport retirement alone cannot release public token");
    runtime.MarkChildLocalComplete(children.front());
    checks.Check(runtime.Poll(11) == CollectiveAggregatePhase::READY,
                 "local and transport child gates make aggregate ready");
    checks.Reject("duplicate child local completion", [&] {
        runtime.MarkChildLocalComplete(children.front());
    });
    checks.Check(runtime.TryWait(11) &&
                     runtime.Residual().ActiveEmpty(),
                 "successful WAIT consumes aggregate child state");
    checks.Reject("stale child completion after WAIT", [&] {
        runtime.MarkChildTransportRetired(children.front());
    });
    checks.Reject("stale public WAIT", [&] { (void)runtime.TryWait(11); });
    checks.Reject("unknown public token", [&] { (void)runtime.Poll(999); });
}

void CheckCancelAndAtomicity(Checks &checks) {
    const auto lowering = MakeLowering({RemoteP2p(0, 21, 22)}, 2000);
    CollectiveAggregateRuntime runtime(0, Capacity());
    runtime.RegisterArtifact(lowering);
    const uint32_t child = runtime.ChildTokens(21).front();
    runtime.Cancel(21);
    checks.Check(runtime.Residual().ActiveEmpty(),
                 "CANCEL before BEGIN removes an aggregate atomically");
    checks.Reject("duplicate stale CANCEL", [&] { runtime.Cancel(21); });

    runtime.RegisterArtifact(lowering);
    runtime.Begin(21);
    const CollectiveAggregateResidual before = runtime.Residual();
    checks.Reject("CANCEL after child start", [&] { runtime.Cancel(21); });
    checks.Check(runtime.Residual().aggregates == before.aggregates &&
                     runtime.HasChildToken(child) &&
                     runtime.Poll(21) == CollectiveAggregatePhase::ACTIVE,
                 "failed CANCEL leaves every child and aggregate intact");
    runtime.MarkChildLocalComplete(child);
    runtime.MarkChildTransportRetired(child);
    checks.Check(runtime.TryWait(21),
                 "aggregate remains usable after rejected CANCEL");
}

void CheckFenceEpochsAndConcurrency(Checks &checks) {
    IsaV1CollectivePlan distinct =
        RemoteP2p(0, 55, 56, 0x4200);
    distinct.key.collective_id = 10;
    const auto lowering = MakeLowering(
        {RemoteP2p(0, 51, 52, 0x4000),
         RemoteP2p(1, 53, 54, 0x4100), distinct},
        3000);
    CollectiveAggregateRuntime runtime(0, Capacity());
    runtime.RegisterArtifact(lowering);
    checks.Check(runtime.Residual().aggregates == 3,
                 "consecutive epochs and a distinct collective coexist");
    CompleteAggregate(runtime, 51);
    runtime.Begin(53);
    const std::vector<uint32_t> epoch1_children = runtime.ChildTokens(53);
    runtime.MarkChildLocalComplete(epoch1_children.front());
    CompleteAggregate(runtime, 55);
    checks.Check(!runtime.TryFence() &&
                     runtime.Residual().aggregates == 3 &&
                     runtime.HasPublicToken(51) &&
                     runtime.HasPublicToken(53) &&
                     runtime.HasPublicToken(55),
                 "FENCE failure is all-or-nothing across concurrent collectives");
    runtime.MarkChildTransportRetired(epoch1_children.front());
    checks.Check(runtime.TryFence() && runtime.Residual().ActiveEmpty(),
                 "FENCE retires all ready aggregates in token order");
    checks.Check(runtime.TryFence(), "empty FENCE succeeds");
}

void CheckZeroChildLocalWork(Checks &checks) {
    const auto local = MakeLowering({LocalP2p(31, 32)}, 4000);
    CollectiveAggregateRuntime first(0, Capacity());
    CollectiveAggregateRuntime second(0, Capacity());
    first.RegisterArtifact(local);
    second.RegisterArtifact(local);
    CollectiveAggregateRuntime zero_child_capacity(
        0, {2, 0, 2, 0});
    zero_child_capacity.RegisterArtifact(local);
    checks.Check(first.ChildTokens(31).empty() &&
                     first.ChildTokens(32).empty() &&
                     first.LocalWorks(31).size() == 1 &&
                     first.LocalWorks(32).size() == 1 &&
                     first.LocalWorks(31) == second.LocalWorks(31) &&
                     zero_child_capacity.Residual().child_tokens == 0,
                 "N=1 local copy gets deterministic explicit gates per token");
    const auto send_work = first.LocalWorks(31).front();
    checks.Reject("local work before BEGIN", [&] {
        first.MarkLocalWorkComplete(send_work);
    });
    first.Begin(31);
    first.MarkLocalWorkComplete(send_work);
    checks.Check(first.TryWait(31) && first.HasPublicToken(32),
                 "one local-copy public token retires independently");
    checks.Reject("stale local work after WAIT", [&] {
        first.MarkLocalWorkComplete(send_work);
    });
    CompleteAggregate(first, 32);
    checks.Check(first.TryWait(32) && first.Residual().ActiveEmpty(),
                 "second local-copy token uses its own explicit gate");

    const auto reduce = MakeLowering({N1Reduce(33, 34)}, 4100);
    CollectiveAggregateRuntime reduce_runtime(0, Capacity());
    reduce_runtime.RegisterArtifact(reduce);
    checks.Check(reduce_runtime.ChildTokens(34).empty() &&
                     reduce_runtime.LocalWorks(34).size() == 1 &&
                     reduce_runtime.LocalWorks(34).front().kind ==
                         CollectiveAggregateLocalWorkKind::LOCAL_COPY,
                 "N=1 reduction bypass has a zero-child local-work gate");
    CompleteAggregate(reduce_runtime, 33);
    CompleteAggregate(reduce_runtime, 34);
    checks.Check(reduce_runtime.TryFence(),
                 "N=1 reduction public tokens complete only after local work");

    const auto reduce2 = MakeLowering({Reduce2()}, 4200);
    CollectiveAggregateRuntime reduce2_runtime(0, Capacity());
    reduce2_runtime.RegisterArtifact(reduce2);
    const std::vector<uint32_t> reduce_children =
        reduce2_runtime.ChildTokens(36);
    const auto reduce_works = reduce2_runtime.LocalWorks(36);
    checks.Check(reduce_children.size() == 1 &&
                     reduce_works.size() == 2 &&
                     std::count_if(
                         reduce_works.begin(), reduce_works.end(),
                         [](const auto &work) {
                             return work.kind ==
                                    CollectiveAggregateLocalWorkKind::
                                        REDUCE_COMPUTE;
                         }) == 1,
                 "Reduce RECEIVE maps child, local-copy, and compute gates");
    reduce2_runtime.Begin(36);
    reduce2_runtime.MarkChildLocalComplete(reduce_children.front());
    reduce2_runtime.MarkChildTransportRetired(reduce_children.front());
    for (const auto &work : reduce_works)
        if (work.kind ==
            CollectiveAggregateLocalWorkKind::LOCAL_COPY)
            reduce2_runtime.MarkLocalWorkComplete(work);
    checks.Check(!reduce2_runtime.TryWait(36),
                 "Reduce aggregate waits for strict compute local work");
    for (const auto &work : reduce_works)
        if (work.kind ==
            CollectiveAggregateLocalWorkKind::REDUCE_COMPUTE)
            reduce2_runtime.MarkLocalWorkComplete(work);
    checks.Check(reduce2_runtime.TryWait(36),
                 "Reduce aggregate releases after compute completion");
    reduce2_runtime.Cancel(35);
    checks.Check(reduce2_runtime.Residual().ActiveEmpty(),
                 "remaining diagonal SEND aggregate can cancel pre-BEGIN");
}

void CheckMultiChildAndMembership(Checks &checks) {
    const auto scatter = MakeLowering({Scatter3()}, 5000);
    CollectiveAggregateRuntime runtime(0, Capacity());
    runtime.RegisterArtifact(scatter);
    const std::vector<uint32_t> children = runtime.ChildTokens(41);
    checks.Check(children.size() == 2 &&
                     runtime.LocalWorks(41).size() == 1 &&
                     runtime.ChildTokens(42).empty(),
                 "multi-child SEND and diagonal RECEIVE memberships separate");
    runtime.Begin(41);
    for (const auto &work : runtime.LocalWorks(41))
        runtime.MarkLocalWorkComplete(work);
    for (uint32_t child : children)
        runtime.MarkChildLocalComplete(child);
    checks.Check(!runtime.TryWait(41),
                 "all local completions still wait for every transport retire");
    runtime.MarkChildTransportRetired(children.front());
    checks.Check(!runtime.TryWait(41),
                 "one retired child cannot release a multi-child token");
    runtime.MarkChildTransportRetired(children.back());
    checks.Check(runtime.TryWait(41),
                 "multi-child token releases after both gates on every child");
    CompleteAggregate(runtime, 42);
    checks.Check(runtime.TryWait(42) && runtime.Residual().ActiveEmpty(),
                 "diagonal local aggregate drains independently");

    auto duplicate_child = scatter;
    duplicate_child.children[1].source_internal_token =
        duplicate_child.children[0].source_internal_token;
    CollectiveAggregateRuntime rejected(0, Capacity());
    checks.Reject("one child token cannot belong to two endpoints", [&] {
        rejected.RegisterArtifact(duplicate_child);
    });
    checks.Check(rejected.Residual().ActiveEmpty(),
                 "duplicate child-token registration rolls back fully");
}

void CheckTokenConflictsAndRollback(Checks &checks) {
    const auto valid = MakeLowering({RemoteP2p(0, 61, 62)}, 6000);
    const uint32_t internal = valid.children.front().source_internal_token;
    CollectiveAggregateRuntime public_conflict(0, Capacity());
    checks.Reject("ordinary P2P/public aggregate token conflict", [&] {
        public_conflict.RegisterArtifact(valid, {61});
    });
    checks.Check(public_conflict.Residual().ActiveEmpty() &&
                     public_conflict.Residual().reserved_tokens == 0,
                 "public-token conflict does not retain reserved state");

    CollectiveAggregateRuntime child_conflict(0, Capacity());
    checks.Reject("ordinary P2P/internal child token conflict", [&] {
        child_conflict.RegisterArtifact(valid, {internal});
    });
    checks.Check(child_conflict.Residual().ActiveEmpty() &&
                     child_conflict.Residual().reserved_tokens == 0,
                 "child-token conflict rolls back reserved state");

    auto duplicate_public = MakeLowering({LocalP2p(63, 66)}, 6100);
    duplicate_public.plans[0].rank_records[0].receive.token = 63;
    CollectiveAggregateRuntime duplicate(0, Capacity());
    checks.Reject("public aggregate token must be unique", [&] {
        duplicate.RegisterArtifact(duplicate_public);
    });
    checks.Check(duplicate.Residual().ActiveEmpty(),
                 "duplicate public-token failure leaves no residue");

    CollectiveAggregateRuntime runtime(0, Capacity());
    runtime.RegisterArtifact(valid);
    const CollectiveAggregateResidual before = runtime.Residual();
    auto malformed = MakeLowering(
        {RemoteP2p(1, 64, 65, 0x5000)}, 6200);
    auto action = std::find_if(
        malformed.core_actions.front().actions.begin(),
        malformed.core_actions.front().actions.end(),
        [](const IsaV1LoweredCollectiveAction &value) {
            return value.internal_token != 0;
        });
    action->public_aggregate_token = 999;
    checks.Reject("malformed action mapping", [&] {
        runtime.RegisterArtifact(malformed);
    });
    checks.Check(runtime.Residual().aggregates == before.aggregates &&
                     runtime.Residual().child_tokens ==
                         before.child_tokens &&
                     runtime.HasPublicToken(61) &&
                     !runtime.HasPublicToken(64),
                 "failed concurrent registration preserves prior aggregate");
}

void CheckCapacityOverflow(Checks &checks) {
    const auto local = MakeLowering({LocalP2p(71, 72)}, 7000);
    CollectiveAggregateRuntime aggregate_limit(
        0, Capacity(1, 8, 8, 8));
    checks.Reject("aggregate capacity overflow", [&] {
        aggregate_limit.RegisterArtifact(local);
    });
    checks.Check(aggregate_limit.Residual().ActiveEmpty(),
                 "aggregate overflow rolls back");

    const auto scatter = MakeLowering({Scatter3()}, 7100);
    CollectiveAggregateRuntime child_limit(
        0, Capacity(8, 1, 8, 8));
    checks.Reject("child-token capacity overflow", [&] {
        child_limit.RegisterArtifact(scatter);
    });
    checks.Check(child_limit.Residual().ActiveEmpty(),
                 "child-token overflow rolls back");

    CollectiveAggregateRuntime work_limit(
        0, Capacity(8, 8, 1, 8));
    checks.Reject("local-work capacity overflow", [&] {
        work_limit.RegisterArtifact(local);
    });
    checks.Check(work_limit.Residual().ActiveEmpty(),
                 "local-work overflow rolls back");

    const auto remote = MakeLowering({RemoteP2p(0, 73, 74)}, 7200);
    CollectiveAggregateRuntime reserved_limit(
        0, Capacity(8, 8, 8, 1));
    checks.Reject("reserved-token capacity overflow", [&] {
        reserved_limit.RegisterArtifact(remote, {80, 81});
    });
    checks.Check(reserved_limit.Residual().ActiveEmpty() &&
                     reserved_limit.Residual().reserved_tokens == 0,
                 "reserved-token overflow rolls back");
    checks.Reject("zero aggregate capacity", [&] {
        CollectiveAggregateRuntime invalid(0, {0, 1, 1, 1});
        (void)invalid;
    });
}

} // namespace

CollectiveAggregateV1SelfTestResult
CheckCollectiveAggregateV1Runtime() {
    Checks checks;
    CheckTwoGateWaitAndStale(checks);
    CheckCancelAndAtomicity(checks);
    CheckFenceEpochsAndConcurrency(checks);
    CheckZeroChildLocalWork(checks);
    CheckMultiChildAndMembership(checks);
    CheckTokenConflictsAndRollback(checks);
    CheckCapacityOverflow(checks);
    return std::move(checks.result);
}

int RunCollectiveAggregateV1SelfTest() {
    const CollectiveAggregateV1SelfTestResult result =
        CheckCollectiveAggregateV1Runtime();
    if (result.passed()) {
        std::cout << "ISA-v1 collective aggregate runtime selftest passed ("
                  << result.checks << " checks)\n";
        return 0;
    }
    std::cerr << "ISA-v1 collective aggregate runtime selftest failed ("
              << result.failures.size() << "/" << result.checks
              << ")\n";
    for (const std::string &failure : result.failures)
        std::cerr << "  - " << failure << '\n';
    return 1;
}

#ifdef COLLECTIVE_AGGREGATE_V1_SELFTEST_MAIN
int sc_main(int, char **) { return RunCollectiveAggregateV1SelfTest(); }
#endif
