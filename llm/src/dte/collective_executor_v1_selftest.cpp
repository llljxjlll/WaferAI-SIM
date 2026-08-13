#include "dte/collective_executor_v1.h"

#include "collective_wave_runtime_v1_test_fixture.h"
#include "dte/endpoint_contract.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using collective_wave_runtime_v1_test::BuildImage;
using collective_wave_runtime_v1_test::Cell;
using collective_wave_runtime_v1_test::StrictWireScope;

struct Suite {
    int checks = 0;
    int failures = 0;

    void Check(bool condition, const std::string &name) {
        ++checks;
        if (condition) return;
        ++failures;
        std::cerr << "[COLLECTIVE EXECUTOR V1] FAIL: " << name << '\n';
    }

    template <class F>
    void Rejects(F &&fn, const std::string &name) {
        bool rejected = false;
        try {
            fn();
        } catch (const std::exception &) {
            rejected = true;
        }
        Check(rejected, name);
    }
};

Collective_launch_v1_prim LaunchFor(
    const IsaV1CollectiveProgramImage &image,
    const IsaV1CollectiveIssueSite &site) {
    Collective_launch_v1_prim launch;
    launch.role = site.role == IsaV1CollectiveRecordRole::SEND
                      ? CollectiveLaunchV1Role::ISSUE_SEND
                      : site.role == IsaV1CollectiveRecordRole::RECEIVE
                            ? CollectiveLaunchV1Role::ISSUE_RECEIVE
                            : CollectiveLaunchV1Role::
                                  DECLARE_REDUCE_COMPUTE;
    launch.image_generation = image.Generation();
    launch.plan_index = site.plan_index;
    launch.external_record_index = site.record_index;
    launch.expected_core = site.core_id;
    launch.key = site.key;
    launch.public_token = site.public_token;
    return launch;
}

struct Harness {
    std::shared_ptr<const IsaV1CollectiveProgramImage> image;
    std::unique_ptr<CollectiveWaveAdmissionCoordinatorV1> coordinator;
    std::map<uint16_t, std::unique_ptr<CollectiveExecutorV1>> executors;

    explicit Harness(
        IsaV1CollectiveProgramImage value,
        uint32_t sessions = 64,
        uint64_t receive_bytes = kDteEndpointP2pMaxBytes,
        std::size_t max_active_override = 0)
        : image(std::make_shared<const IsaV1CollectiveProgramImage>(
              std::move(value))) {
        auto capacity = CollectiveWaveAdmissionCapacityForImageV1(
            *image, sessions, receive_bytes);
        if (max_active_override != 0)
            capacity.max_active_waves = max_active_override;
        coordinator =
            std::make_unique<CollectiveWaveAdmissionCoordinatorV1>(
                std::move(capacity));
        coordinator->RegisterImage(*image);
        for (const auto &core : image->Cores()) {
            auto executor = std::make_unique<CollectiveExecutorV1>();
            executor->Configure(image, core.core_id, coordinator.get());
            executors.emplace(core.core_id, std::move(executor));
        }
    }

    CollectiveExecutorV1 &At(uint16_t core) {
        return *executors.at(core);
    }

    void AcceptAll() {
        for (const IsaV1CollectiveIssueSite &site :
             image->IssueSites())
            At(site.core_id).AcceptLaunch(LaunchFor(*image, site));
    }

    std::vector<uint32_t> Tokens(uint16_t core) const {
        std::vector<uint32_t> result;
        for (const auto &site : image->IssueSites())
            if (site.core_id == core && site.public_token != 0)
                result.push_back(site.public_token);
        return result;
    }
};

struct DriveObservation {
    bool receive_two_gate = false;
    bool send_local_gate = false;
    bool transport_gate = false;
    bool final_gate_blocked = false;
    std::map<std::pair<uint16_t, uint32_t>, uint32_t> next_stream;
    std::map<uint16_t, std::vector<uint32_t>> plan_order;
};

void CompleteAction(Harness &harness, uint16_t core,
                    CollectiveExecutorActionV1 action,
                    DriveObservation *observation) {
    CollectiveExecutorV1 &executor = harness.At(core);
    if (observation != nullptr) {
        const auto key = std::make_pair(core, action.plan_index);
        const auto found = observation->next_stream.find(key);
        if (found == observation->next_stream.end()) {
            const auto *core_image = harness.image->FindCore(core);
            const auto range = std::find_if(
                core_image->action_ranges.begin(),
                core_image->action_ranges.end(),
                [&](const IsaV1CollectiveActionRange &value) {
                    return value.plan_index == action.plan_index;
                });
            if (range == core_image->action_ranges.end())
                throw std::logic_error("test plan range is missing");
            observation->next_stream.emplace(key, range->begin + 1);
            if (action.action_stream_index != range->begin)
                throw std::logic_error("executor action order starts late");
        } else {
            if (action.action_stream_index != found->second)
                throw std::logic_error("executor action order is non-canonical");
            ++found->second;
        }
        observation->plan_order[core].push_back(action.plan_index);
    }

    if (action.prim) {
        const auto wire = action.prim->serialize();
        if (wire.empty())
            throw std::logic_error(
                "executor materialized an empty strict Prim wire");
    }

    if (action.wait.has_value()) {
        switch (action.wait->kind) {
        case CollectiveExecutorWaitKindV1::
                 RECEIVE_LOCAL_AND_TRANSPORT:
            if (observation != nullptr &&
                !observation->receive_two_gate) {
                const bool first = executor.WaitComplete(
                    action.action_id, false, true);
                const bool second = executor.WaitComplete(
                    action.action_id, true, false);
                if (first || second)
                    throw std::logic_error(
                        "receive wait accepted one completion gate");
                observation->receive_two_gate = true;
            }
            if (!executor.WaitComplete(action.action_id, true, true))
                throw std::logic_error("receive wait did not complete");
            return;
        case CollectiveExecutorWaitKindV1::SEND_LOCAL:
            if (observation != nullptr &&
                !observation->send_local_gate) {
                if (executor.WaitComplete(action.action_id, false, true))
                    throw std::logic_error(
                        "send wait ignored local completion");
                observation->send_local_gate = true;
            }
            if (!executor.WaitComplete(action.action_id, true, false))
                throw std::logic_error("send local wait did not complete");
            return;
        case CollectiveExecutorWaitKindV1::TRANSPORT_RETIRE:
            if (observation != nullptr &&
                !observation->transport_gate) {
                if (executor.WaitComplete(action.action_id, true, false))
                    throw std::logic_error(
                        "transport wait ignored retirement");
                observation->transport_gate = true;
            }
            if (!executor.WaitComplete(action.action_id, false, true))
                throw std::logic_error("transport wait did not complete");
            return;
        }
    }

    if (action.canonical_kind ==
            IsaV1ActionKind::COMPLETE_BARRIER &&
        observation != nullptr &&
        !observation->final_gate_blocked) {
        const auto tokens = harness.Tokens(core);
        if (!tokens.empty() && executor.TryWait(tokens.front()))
            throw std::logic_error(
                "public token retired before final barrier");
        observation->final_gate_blocked = true;
    }
    executor.ActionComplete(action.action_id);
}

void Drive(Harness &harness, DriveObservation *observation = nullptr) {
    constexpr std::size_t kStepLimit = 100000;
    for (std::size_t step = 0; step < kStepLimit; ++step) {
        bool progress = false;
        bool actions_done = true;
        for (auto &[core, executor] : harness.executors) {
            if (executor->Residual().remaining_actions != 0)
                actions_done = false;
            auto action = executor->NextAction();
            if (!action.has_value()) continue;
            CompleteAction(harness, core, std::move(*action),
                           observation);
            progress = true;
        }
        if (actions_done) break;
        if (!progress && step + 1 == kStepLimit)
            throw std::logic_error("collective executor made no progress");
    }
    for (auto &[core, executor] : harness.executors) {
        (void)core;
        if (!executor->TryFence())
            throw std::logic_error(
                "collective executor final FENCE did not complete");
        if (!executor->Drained())
            throw std::logic_error(
                "collective executor did not locally drain");
    }
    harness.coordinator->RetireImage(
        CollectiveProgramImageIdentity(*harness.image));
    if (!harness.coordinator->Residual().Empty())
        throw std::logic_error(
            "collective executor shared wave state did not drain");
}

void TestNineGrid(Suite &suite) {
    const std::vector<Cell> cells{
        {CollTxKind::UNICAST, CollRxKind::UNICAST},
        {CollTxKind::SCATTER, CollRxKind::UNICAST},
        {CollTxKind::BROADCAST, CollRxKind::UNICAST},
        {CollTxKind::UNICAST, CollRxKind::GATHER},
        {CollTxKind::SCATTER, CollRxKind::GATHER},
        {CollTxKind::BROADCAST, CollRxKind::GATHER},
        {CollTxKind::UNICAST, CollRxKind::REDUCE},
        {CollTxKind::SCATTER, CollRxKind::REDUCE},
        {CollTxKind::BROADCAST, CollRxKind::REDUCE},
    };
    for (std::size_t n : {std::size_t{1}, std::size_t{2},
                          std::size_t{4}}) {
        for (const Cell &cell : cells) {
            Harness harness(BuildImage({cell}, n));
            harness.AcceptAll();
            Drive(harness);
            bool drained = true;
            for (const auto &[core, executor] : harness.executors) {
                (void)core;
                drained = drained && executor->Drained();
            }
            suite.Check(drained,
                        "nine-grid executor N=" + std::to_string(n) +
                            " tx=" +
                            std::to_string(static_cast<int>(cell.tx)) +
                            " rx=" +
                            std::to_string(static_cast<int>(cell.rx)));
        }
    }
}

IsaV1CollectiveProgramImage ImageWithOrdinaryOnlyCore() {
    ProgramArtifact artifact =
        collective_wave_runtime_v1_test::Artifact(
            {{CollTxKind::UNICAST, CollRxKind::UNICAST}}, 2);
    artifact.cores.push_back(ProgramCore{2, {}});
    artifact.envelope.active_cores.push_back(2);
    artifact.envelope.expected_ack_cores.push_back(2);
    IsaV1CollectiveProgramImageConfig config;
    config.total_cores = 8;
    config.cores_per_die = 8;
    config.generation = 0x1234f00d;
    const auto lowering = LowerIsaV1CollectiveArtifact(
        artifact, config.total_cores, config.cores_per_die,
        config.planner_capacity);
    return BuildIsaV1CollectiveProgramImage(
        artifact, lowering, config);
}

void TestMarkersAndTransaction(Suite &suite) {
    Harness harness(BuildImage(
        {{CollTxKind::UNICAST, CollRxKind::UNICAST}}, 1));
    const std::vector<IsaV1CollectiveIssueSite> sites =
        harness.image->IssueSites();
    suite.Check(sites.size() >= 2,
                "marker test has split send/receive issue roles");
    const Collective_launch_v1_prim valid =
        LaunchFor(*harness.image, sites.front());
    const auto before = harness.At(0).Residual();

    auto reject_unchanged = [&](Collective_launch_v1_prim bad,
                                const std::string &name) {
        suite.Rejects([&] { harness.At(0).AcceptLaunch(bad); },
                      name);
        const auto after = harness.At(0).Residual();
        suite.Check(after.accepted_issue_sites ==
                        before.accepted_issue_sites &&
                        !harness.At(0).HasRunnable(),
                    name + " is transactional");
    };
    auto bad = valid;
    ++bad.image_generation;
    reject_unchanged(bad, "wrong marker generation");
    bad = valid;
    ++bad.expected_core;
    reject_unchanged(bad, "wrong marker core");
    bad = valid;
    ++bad.key.group_id;
    reject_unchanged(bad, "wrong marker key");
    bad = valid;
    ++bad.external_record_index;
    reject_unchanged(bad, "unknown marker record");
    bad = valid;
    ++bad.public_token;
    reject_unchanged(bad, "wrong marker token");
    bad = valid;
    bad.role = bad.role == CollectiveLaunchV1Role::ISSUE_SEND
                   ? CollectiveLaunchV1Role::ISSUE_RECEIVE
                   : CollectiveLaunchV1Role::ISSUE_SEND;
    reject_unchanged(bad, "wrong marker role");

    suite.Check(!harness.At(0).AcceptLaunch(valid) &&
                    !harness.At(0).HasRunnable() &&
                    !harness.At(0).NextAction().has_value(),
                "missing marker keeps plan non-runnable");
    suite.Rejects(
        [&] { harness.At(0).AcceptLaunch(valid); },
        "duplicate marker is rejected");
    bool activated = false;
    for (std::size_t index = 1; index < sites.size(); ++index)
        activated =
            harness.At(0).AcceptLaunch(
                LaunchFor(*harness.image, sites[index])) ||
            activated;
    suite.Check(activated && harness.At(0).HasRunnable(),
                "all split marker roles begin one plan exactly once");
    Drive(harness);

    Harness mixed(ImageWithOrdinaryOnlyCore());
    suite.Check(mixed.executors.count(2) == 1 &&
                    !mixed.At(2).HasRunnable() &&
                    !mixed.At(2).NextAction().has_value(),
                "ordinary-only active core accepts shared image without actions");
    mixed.AcceptAll();
    Drive(mixed);
    suite.Check(mixed.At(2).Drained(),
                "ordinary-only active core drains an empty local image");
}

void TestCapacityAndConcurrentPlans(Suite &suite) {
    IsaV1PlannerCapacity planner;
    planner.max_child_bytes = kDteEndpointP2pMaxBytes;
    planner.max_receive_bytes_per_rank_per_wave =
        kDteEndpointP2pMaxBytes;
    planner.max_sessions_per_rank_per_wave = 3;
    const IsaV1CollectiveProgramImage production =
        BuildImage({{CollTxKind::SCATTER, CollRxKind::GATHER}},
                   4, planner);
    const auto capacity =
        CollectiveWaveAdmissionCapacityForImageV1(
            production, 3, kDteEndpointP2pMaxBytes);
    bool bounded = capacity.max_registered_waves > 1;
    for (const auto &demand : production.WaveDemands())
        bounded = bounded && demand.endpoint_sessions <= 3 &&
                  demand.receive_bytes <=
                      kDteEndpointP2pMaxBytes;
    suite.Check(bounded,
                "capacity helper preserves production multi-wave bounds");
    suite.Rejects(
        [&] {
            (void)CollectiveWaveAdmissionCapacityForImageV1(
                production, 1, kDteEndpointP2pMaxBytes);
        },
        "capacity helper rejects insufficient sessions");
    suite.Rejects(
        [&] {
            (void)CollectiveWaveAdmissionCapacityForImageV1(
                production, 3, 1);
        },
        "capacity helper rejects insufficient receive bytes");

    Harness concurrent(
        BuildImage({
            {CollTxKind::BROADCAST, CollRxKind::GATHER},
            {CollTxKind::SCATTER, CollRxKind::REDUCE}},
                   2));
    concurrent.AcceptAll();
    DriveObservation observation;
    Drive(concurrent, &observation);
    bool saw_both = false;
    for (const auto &[core, order] : observation.plan_order) {
        (void)core;
        if (std::find(order.begin(), order.end(), 0) != order.end() &&
            std::find(order.begin(), order.end(), 1) != order.end())
            saw_both = true;
    }
    suite.Check(saw_both,
                "two runnable plans interleave without duplicate BEGIN");

    Harness fair(
        BuildImage({
            {CollTxKind::BROADCAST, CollRxKind::GATHER},
            {CollTxKind::SCATTER, CollRxKind::GATHER}},
                   2),
        64, kDteEndpointP2pMaxBytes, 1);
    fair.AcceptAll();
    suite.Check(!fair.At(0).NextAction().has_value(),
                "first core forms both plan waves");
    auto first = fair.At(1).NextAction();
    suite.Check(first.has_value() && first->plan_index == 0,
                "oldest complete wave becomes active first");
    CompleteAction(fair, 1, std::move(*first), nullptr);
    auto next = fair.At(1).NextAction();
    if (next.has_value())
        CompleteAction(fair, 1, std::move(*next), nullptr);
    suite.Check(
        fair.coordinator->Poll(fair.At(0).Identity(), 1, 0) ==
            CollectiveWaveAdmissionStatusV1::PENDING,
        "younger wave waits behind active oldest ticket");
    Drive(fair);
}

void TestWaitFinalCancelAbort(Suite &suite) {
    Harness gates(BuildImage(
        {{CollTxKind::BROADCAST, CollRxKind::REDUCE}}, 2));
    gates.AcceptAll();
    DriveObservation observation;
    Drive(gates, &observation);
    suite.Check(observation.receive_two_gate &&
                    observation.send_local_gate &&
                    observation.transport_gate &&
                    observation.final_gate_blocked,
                "typed waits and final phase enforce all completion gates");

    Harness cancel(BuildImage(
        {{CollTxKind::UNICAST, CollRxKind::UNICAST}}, 1));
    const auto cancel_sites = cancel.image->IssueSites();
    const uint32_t token = cancel_sites.front().public_token;
    suite.Check(cancel.At(0).HasPublicToken(token) &&
                    !cancel.At(0).HasRunnable(),
                "collective token is queryable before plan BEGIN");
    cancel.At(0).Cancel(token);
    suite.Check(cancel.At(0).Drained() &&
                    cancel.At(0).Residual().aborted == 1 &&
                    !cancel.coordinator->HasImage(),
                "pre-BEGIN CANCEL atomically aborts and drains image");

    Harness active(BuildImage(
        {{CollTxKind::UNICAST, CollRxKind::UNICAST}}, 1));
    active.AcceptAll();
    const uint32_t active_token =
        active.image->IssueSites().front().public_token;
    suite.Rejects(
        [&] { active.At(0).Cancel(active_token); },
        "CANCEL after aggregate BEGIN is rejected");
    suite.Check(active.At(0).HasPublicToken(active_token) &&
                    active.At(0).HasRunnable() &&
                    active.coordinator->HasImage(),
                "active CANCEL rejection preserves runnable image");
    Drive(active);

    Harness aborted(BuildImage(
        {{CollTxKind::UNICAST, CollRxKind::UNICAST}}, 2));
    aborted.AcceptAll();
    auto action = aborted.At(0).NextAction();
    suite.Check(!action.has_value(),
                "partial wave does not expose an action before ACTIVE");
    aborted.At(0).Abort();
    aborted.At(0).Abort();
    suite.Check(aborted.At(0).Drained() &&
                    aborted.At(0).Residual().aborted == 1 &&
                    !aborted.coordinator->HasImage(),
                "explicit abort is idempotent and reclaims shared wave state");
}

} // namespace

int RunCollectiveExecutorV1SelfTest() {
    StrictWireScope strict;
    Suite suite;
    TestNineGrid(suite);
    TestMarkersAndTransaction(suite);
    TestCapacityAndConcurrentPlans(suite);
    TestWaitFinalCancelAbort(suite);
    std::cout << "[COLLECTIVE EXECUTOR V1] "
              << (suite.failures == 0 ? "PASS" : "FAIL")
              << " (" << suite.checks << " checks)\n";
    return suite.failures;
}

#ifdef COLLECTIVE_EXECUTOR_V1_SELFTEST_MAIN
int sc_main(int, char **) {
    return RunCollectiveExecutorV1SelfTest();
}
#endif
