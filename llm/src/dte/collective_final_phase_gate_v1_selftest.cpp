#include "dte/collective_final_phase_gate_v1.h"
#include "dte/collective_final_phase_gate_v1_selftest.h"

#include "collective_wave_runtime_v1_test_fixture.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
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
        std::cerr << "[COLLECTIVE FINAL PHASE GATE V1] FAIL: "
                  << name << '\n';
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

CollectiveAggregateCapacity AggregateCapacity() {
    return {64, 256, 256, 64};
}

CollectiveFinalPhaseGateCapacityV1 GateCapacity() {
    return {64, 128};
}

void CompleteAggregate(CollectiveAggregateRuntime &aggregate,
                       uint32_t public_token) {
    aggregate.Begin(public_token);
    for (uint32_t child :
         aggregate.ChildTokens(public_token)) {
        aggregate.MarkChildLocalComplete(child);
        aggregate.MarkChildTransportRetired(child);
    }
    for (const auto &work :
         aggregate.LocalWorks(public_token))
        aggregate.MarkLocalWorkComplete(work);
}

struct Bound {
    IsaV1CollectiveProgramImage image;
    CollectiveAggregateRuntime aggregate;
    CollectiveAggregateFinalPhaseGateV1 gate;
    CollectiveProgramImageIdentityV1 identity;

    Bound(IsaV1CollectiveProgramImage value, uint16_t core)
        : image(std::move(value)),
          aggregate(core, AggregateCapacity()),
          gate(core, GateCapacity()),
          identity(CollectiveProgramImageIdentity(image)) {
        const auto *core_image = image.FindCore(core);
        if (core_image == nullptr)
            throw std::logic_error("test core image is missing");
        aggregate.RegisterArtifact(
            image.Lowering(),
            core_image->ordinary_reserved_tokens);
        gate.RegisterImage(image, aggregate);
    }
};

uint32_t TokenFor(const IsaV1CollectiveProgramImage &image,
                  uint16_t core, uint32_t plan,
                  IsaV1CollectiveRecordRole role) {
    for (const auto &site : image.IssueSites())
        if (site.core_id == core && site.plan_index == plan &&
            site.role == role)
            return site.public_token;
    throw std::logic_error("test token is missing");
}

void CompleteFinal(Bound &bound, uint16_t core,
                   uint32_t plan_index) {
    const IsaV1CollectivePlan &plan =
        bound.image.Lowering().plans.at(plan_index);
    const auto rank =
        static_cast<uint16_t>(std::find(
            plan.group.begin(), plan.group.end(), core) -
                              plan.group.begin());
    bound.gate.MarkFinalComplete(
        bound.identity, plan_index, plan.key, core, rank,
        plan.waves.back().complete_phase_id);
}

void TestFinalPhaseBlocksReady(Suite &suite) {
    Bound bound(BuildImage(
                    {{CollTxKind::BROADCAST, CollRxKind::REDUCE}},
                    4),
                1);
    const uint32_t send = TokenFor(
        bound.image, 1, 0,
        IsaV1CollectiveRecordRole::SEND);
    const uint32_t receive = TokenFor(
        bound.image, 1, 0,
        IsaV1CollectiveRecordRole::RECEIVE);
    CompleteAggregate(bound.aggregate, send);
    CompleteAggregate(bound.aggregate, receive);

    suite.Check(bound.aggregate.Poll(send) ==
                        CollectiveAggregatePhase::READY &&
                    bound.gate.Poll(
                        bound.identity, send,
                        bound.aggregate) ==
                        CollectiveFinalPhaseGateStatusV1::
                            WAITING_FINAL_PHASE &&
                    !bound.gate.TryWait(
                        bound.identity, send,
                        bound.aggregate),
                "aggregate READY cannot retire before final COMPLETE phase");

    CompleteFinal(bound, 1, 0);
    suite.Check(bound.gate.Poll(
                    bound.identity, send,
                    bound.aggregate) ==
                        CollectiveFinalPhaseGateStatusV1::READY &&
                    bound.gate.TryWait(
                        bound.identity, send,
                        bound.aggregate),
                "final COMPLETE releases one ready aggregate");
    suite.Check(bound.gate.TryWait(
                    bound.identity, receive,
                    bound.aggregate),
                "one final plan phase releases every local public token");
    suite.Check(bound.gate.Residual().plans == 0 &&
                    bound.gate.Residual().public_tokens == 0 &&
                    bound.aggregate.Residual().ActiveEmpty(),
                "waits drain aggregate and final-phase membership");

    bound.gate.RetireImage(bound.identity);
    suite.Check(bound.gate.Residual().Empty(),
                "drained gate retires image identity");
}

void TestFenceAndMetadata(Suite &suite) {
    Bound bound(BuildImage(
                    {{CollTxKind::SCATTER, CollRxKind::GATHER},
                     {CollTxKind::BROADCAST,
                      CollRxKind::UNICAST}},
                    4),
                1);
    std::vector<uint32_t> tokens;
    for (const auto &site : bound.image.IssueSites())
        if (site.core_id == 1 &&
            site.role !=
                IsaV1CollectiveRecordRole::REDUCE_COMPUTE)
            tokens.push_back(site.public_token);
    for (uint32_t token : tokens)
        CompleteAggregate(bound.aggregate, token);

    CompleteFinal(bound, 1, 0);
    suite.Check(!bound.gate.TryFence(
                    bound.identity, bound.aggregate),
                "fence waits for every plan final COMPLETE");
    CompleteFinal(bound, 1, 1);
    suite.Check(bound.gate.TryFence(
                    bound.identity, bound.aggregate) &&
                    bound.aggregate.Residual().ActiveEmpty() &&
                    bound.gate.Residual().public_tokens == 0,
                "fence consumes all aggregates only after all final phases");
    bound.gate.RetireImage(bound.identity);

    Bound metadata(BuildImage(
                       {{CollTxKind::SCATTER,
                         CollRxKind::GATHER}},
                       4),
                   1);
    const auto &plan = metadata.image.Lowering().plans[0];
    auto stale = metadata.identity;
    ++stale.generation;
    suite.Rejects(
        [&] { metadata.gate.MarkFinalComplete(
                  stale, 0, plan.key, 1, 1,
                  plan.waves.back().complete_phase_id); },
        "stale generation is rejected");
    suite.Rejects(
        [&] { metadata.gate.MarkFinalComplete(
                  metadata.identity, 0, plan.key, 2, 1,
                  plan.waves.back().complete_phase_id); },
        "wrong executing core is rejected");
    suite.Rejects(
        [&] { metadata.gate.MarkFinalComplete(
                  metadata.identity, 0, plan.key, 1, 0,
                  plan.waves.back().complete_phase_id); },
        "wrong rank is rejected");
    suite.Rejects(
        [&] { metadata.gate.MarkFinalComplete(
                  metadata.identity, 0, plan.key, 1, 1,
                  plan.waves.back().posted_phase_id); },
        "posted/non-final phase cannot open final gate");
    auto wrong_key = plan.key;
    ++wrong_key.epoch;
    suite.Rejects(
        [&] { metadata.gate.MarkFinalComplete(
                  metadata.identity, 0, wrong_key, 1, 1,
                  plan.waves.back().complete_phase_id); },
        "wrong collective key is rejected");
    CompleteFinal(metadata, 1, 0);
    suite.Rejects(
        [&] { CompleteFinal(metadata, 1, 0); },
        "duplicate final completion is rejected");

    Bound idle(BuildImage(
                   {{CollTxKind::UNICAST, CollRxKind::UNICAST}},
                   4),
               2);
    suite.Check(idle.gate.Residual().plans == 1 &&
                    idle.gate.Residual().public_tokens == 0,
                "idle keyed-P2P rank retains a phase-only plan gate");
    CompleteFinal(idle, 2, 0);
    suite.Check(idle.gate.TryFence(idle.identity, idle.aggregate) &&
                    idle.gate.Residual().plans == 0,
                "idle rank drains after its final COMPLETE phase");
    idle.gate.RetireImage(idle.identity);
}

void TestCancelAbortAndRegistration(Suite &suite) {
    Bound bound(BuildImage(
                    {{CollTxKind::SCATTER, CollRxKind::GATHER}},
                    4),
                1);
    const uint32_t send = TokenFor(
        bound.image, 1, 0,
        IsaV1CollectiveRecordRole::SEND);
    const auto before = bound.gate.Residual();
    bound.gate.Cancel(bound.identity, send, bound.aggregate);
    suite.Check(bound.gate.Residual().public_tokens + 1 ==
                        before.public_tokens &&
                    !bound.aggregate.HasPublicToken(send),
                "registered token cancel drains both runtimes");

    bound.gate.AbortRegistered(
        bound.identity, bound.aggregate);
    suite.Check(bound.gate.Residual().public_tokens == 0 &&
                    bound.aggregate.Residual().ActiveEmpty(),
                "registered-image abort drains every remaining aggregate");
    bound.gate.RetireImage(bound.identity);

    Bound active(BuildImage(
                    {{CollTxKind::SCATTER, CollRxKind::GATHER}},
                    4),
                 1);
    const uint32_t token = TokenFor(
        active.image, 1, 0,
        IsaV1CollectiveRecordRole::SEND);
    active.aggregate.Begin(token);
    const auto residual = active.gate.Residual();
    suite.Rejects(
        [&] { active.gate.AbortRegistered(
                  active.identity, active.aggregate); },
        "active aggregate force-abort is explicitly rejected");
    suite.Check(active.gate.Residual().public_tokens ==
                        residual.public_tokens &&
                    active.aggregate.HasPublicToken(token),
                "failed abort is transactional");

    const auto image = BuildImage(
        {{CollTxKind::SCATTER, CollRxKind::GATHER}}, 4);
    CollectiveAggregateRuntime empty(1, AggregateCapacity());
    CollectiveAggregateFinalPhaseGateV1 gate(
        1, GateCapacity());
    suite.Rejects(
        [&] { gate.RegisterImage(image, empty); },
        "gate registration requires pre-registered aggregate tokens");
    suite.Check(gate.Residual().Empty(),
                "failed gate registration exposes no partial identity");

    auto low = GateCapacity();
    low.max_public_tokens = 1;
    CollectiveAggregateRuntime aggregate(1, AggregateCapacity());
    aggregate.RegisterArtifact(image.Lowering());
    CollectiveAggregateFinalPhaseGateV1 bounded(1, low);
    suite.Rejects(
        [&] { bounded.RegisterImage(image, aggregate); },
        "public-token capacity failure rejects transactionally");
    suite.Check(bounded.Residual().Empty(),
                "capacity failure leaves final gate empty");
}

} // namespace

int RunCollectiveFinalPhaseGateV1SelfTest() {
    StrictWireScope strict;
    Suite suite;
    TestFinalPhaseBlocksReady(suite);
    TestFenceAndMetadata(suite);
    TestCancelAbortAndRegistration(suite);
    if (suite.failures == 0) {
        std::cout << "[COLLECTIVE FINAL PHASE GATE V1] PASS ("
                  << suite.checks << " checks)\n";
        return 0;
    }
    std::cerr << "[COLLECTIVE FINAL PHASE GATE V1] FAIL ("
              << suite.failures << "/" << suite.checks
              << " checks failed)\n";
    return suite.failures;
}

#ifdef COLLECTIVE_FINAL_PHASE_GATE_V1_SELFTEST_MAIN
int sc_main(int, char **) {
    return RunCollectiveFinalPhaseGateV1SelfTest();
}
#endif
