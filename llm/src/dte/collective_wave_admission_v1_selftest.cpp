#include "dte/collective_wave_admission_v1.h"
#include "dte/collective_wave_admission_v1_selftest.h"

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
        std::cerr << "[COLLECTIVE WAVE ADMISSION V1] FAIL: "
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

CollectiveWaveAdmissionCapacityV1 Capacity(
    std::size_t cores, uint32_t sessions = 8,
    uint64_t bytes = 4096) {
    CollectiveWaveAdmissionCapacityV1 capacity;
    capacity.max_plans = 16;
    capacity.max_registered_waves = 64;
    capacity.max_wave_demands = 256;
    capacity.max_pending_waves = 64;
    capacity.max_active_waves = 16;
    for (std::size_t core = 0; core < cores; ++core)
        capacity.cores.push_back(
            {static_cast<uint16_t>(core), sessions, bytes});
    return capacity;
}

void ArriveAll(CollectiveWaveAdmissionCoordinatorV1 &runtime,
               CollectiveProgramImageIdentityV1 identity,
               uint32_t plan, uint16_t wave,
               const std::vector<uint16_t> &cores) {
    for (uint16_t core : cores)
        (void)runtime.Arrive(identity, plan, wave, core);
}

void DepartAll(CollectiveWaveAdmissionCoordinatorV1 &runtime,
               CollectiveProgramImageIdentityV1 identity,
               uint32_t plan, uint16_t wave,
               const std::vector<uint16_t> &cores) {
    for (uint16_t core : cores)
        runtime.Depart(identity, plan, wave, core);
}

void TestLifecycle(Suite &suite) {
    const auto image = BuildImage(
        {{CollTxKind::BROADCAST, CollRxKind::REDUCE}}, 4);
    const auto identity = CollectiveProgramImageIdentity(image);
    CollectiveWaveAdmissionCoordinatorV1 runtime(Capacity(4));
    runtime.RegisterImage(image);
    const auto registered = runtime.Residual();
    suite.Check(registered.images == 1 &&
                    registered.waves ==
                        image.AdmissionCapacity().waves &&
                    registered.arrivals == 0 &&
                    registered.active_endpoint_sessions == 0,
                "transactional registration stores bounded canonical waves");

    const auto first = runtime.Arrive(identity, 0, 0, 3);
    suite.Check(first == CollectiveWaveAdmissionStatusV1::FORMING &&
                    runtime.Residual().forming_waves == 1,
                "partial arrival forms no resource-holding wave");
    suite.Rejects(
        [&] { (void)runtime.Arrive(identity, 0, 0, 3); },
        "duplicate arrival is rejected atomically");
    for (uint16_t core : {uint16_t{1}, uint16_t{0},
                          uint16_t{2}})
        (void)runtime.Arrive(identity, 0, 0, core);
    suite.Check(runtime.Poll(identity, 0, 0) ==
                        CollectiveWaveAdmissionStatusV1::ACTIVE &&
                    runtime.Residual().active_waves == 1 &&
                    runtime.Residual().active_endpoint_sessions > 0 &&
                    runtime.Residual().active_receive_bytes > 0,
                "all arrivals atomically reserve whole-wave resources");

    runtime.Depart(identity, 0, 0, 2);
    suite.Check(runtime.Poll(identity, 0, 0) ==
                        CollectiveWaveAdmissionStatusV1::ACTIVE,
                "partial departure retains all wave resources");
    for (uint16_t core : {uint16_t{0}, uint16_t{3},
                          uint16_t{1}})
        runtime.Depart(identity, 0, 0, core);
    suite.Check(runtime.Poll(identity, 0, 0) ==
                        CollectiveWaveAdmissionStatusV1::COMPLETE &&
                    runtime.Residual().active_endpoint_sessions == 0 &&
                    runtime.Residual().active_receive_bytes == 0,
                "last departure releases the atomic reservation");
    suite.Rejects(
        [&] { runtime.Depart(identity, 0, 0, 1); },
        "late duplicate departure is rejected");

    runtime.RetireImage(identity);
    suite.Check(runtime.Residual().Empty(),
                "normal image retirement drains residual");
}

void TestPredecessorAndFairness(Suite &suite) {
    IsaV1PlannerCapacity planner;
    planner.max_sessions_per_rank_per_wave = 1;
    const auto image = BuildImage(
        {{CollTxKind::BROADCAST, CollRxKind::REDUCE},
         {CollTxKind::SCATTER, CollRxKind::GATHER}},
        4, planner);
    CollectiveWaveAdmissionCapacityV1 capacity =
        Capacity(4, 2, 64);
    capacity.max_active_waves = 1;
    CollectiveWaveAdmissionCoordinatorV1 runtime(capacity);
    runtime.RegisterImage(image);
    const auto identity = CollectiveProgramImageIdentity(image);
    const std::vector<uint16_t> cores{0, 1, 2, 3};

    suite.Check(image.Lowering().plans[0].waves.size() > 1 &&
                    image.Lowering().plans[1].waves.size() > 1,
                "fixture creates multi-wave competing plans");
    ArriveAll(runtime, identity, 0, 1, cores);
    ArriveAll(runtime, identity, 1, 0, cores);
    suite.Check(runtime.Poll(identity, 0, 1) ==
                        CollectiveWaveAdmissionStatusV1::PENDING &&
                    runtime.Poll(identity, 1, 0) ==
                        CollectiveWaveAdmissionStatusV1::ACTIVE,
                "blocked successor cannot deadlock an eligible later plan");

    ArriveAll(runtime, identity, 0, 0, cores);
    suite.Check(runtime.Poll(identity, 0, 0) ==
                        CollectiveWaveAdmissionStatusV1::PENDING,
                "active-wave bound queues an older eligible ticket");
    DepartAll(runtime, identity, 1, 0, cores);
    suite.Check(runtime.Poll(identity, 0, 0) ==
                        CollectiveWaveAdmissionStatusV1::ACTIVE,
                "oldest eligible ticket wins deterministic admission");
    DepartAll(runtime, identity, 0, 0, cores);
    suite.Check(runtime.Poll(identity, 0, 1) ==
                        CollectiveWaveAdmissionStatusV1::ACTIVE,
                "successor is admitted immediately after predecessor completes");

    DepartAll(runtime, identity, 0, 1, cores);
    for (std::size_t plan = 0;
         plan < image.Lowering().plans.size(); ++plan) {
        for (std::size_t wave = 0;
             wave < image.Lowering().plans[plan].waves.size();
             ++wave) {
            const auto status = runtime.Poll(
                identity, static_cast<uint32_t>(plan),
                static_cast<uint16_t>(wave));
            if (status ==
                CollectiveWaveAdmissionStatusV1::REGISTERED) {
                ArriveAll(runtime, identity,
                          static_cast<uint32_t>(plan),
                          static_cast<uint16_t>(wave), cores);
            }
            if (runtime.Poll(identity,
                             static_cast<uint32_t>(plan),
                             static_cast<uint16_t>(wave)) ==
                CollectiveWaveAdmissionStatusV1::ACTIVE)
                DepartAll(runtime, identity,
                          static_cast<uint32_t>(plan),
                          static_cast<uint16_t>(wave), cores);
        }
    }
    runtime.RetireImage(identity);
    suite.Check(runtime.Residual().Empty(),
                "cross-plan multi-wave schedule fully drains");
}

void TestRegistrationAbortAndBounds(Suite &suite) {
    const auto image = BuildImage(
        {{CollTxKind::SCATTER, CollRxKind::GATHER}}, 4);
    const auto identity = CollectiveProgramImageIdentity(image);
    CollectiveWaveAdmissionCoordinatorV1 runtime(Capacity(4));
    runtime.RegisterImage(image);
    const auto before = runtime.Residual();

    suite.Rejects(
        [&] { runtime.RegisterImage(image); },
        "duplicate image registration is rejected");
    suite.Check(runtime.Residual().waves == before.waves,
                "failed registration preserves existing image");
    auto stale = identity;
    ++stale.cookie;
    suite.Rejects(
        [&] { (void)runtime.Poll(stale, 0, 0); },
        "stale cookie is rejected");
    suite.Rejects(
        [&] { (void)runtime.Arrive(identity, 99, 0, 0); },
        "unknown plan is rejected");
    suite.Rejects(
        [&] { (void)runtime.Arrive(identity, 0, 0, 7); },
        "nonparticipant core is rejected");
    runtime.AbortImage(identity);
    suite.Check(runtime.Residual().Empty(),
                "registered/forming image abort drains all state");

    CollectiveWaveAdmissionCoordinatorV1 active(Capacity(4));
    active.RegisterImage(image);
    ArriveAll(active, identity, 0, 0, {0, 1, 2, 3});
    active.AbortImage(identity);
    suite.Check(active.Residual().Empty(),
                "active abort reclaims atomic reservations");

    auto too_small = Capacity(4);
    too_small.max_registered_waves = 1;
    too_small.max_wave_demands = 1;
    CollectiveWaveAdmissionCoordinatorV1 bounded(too_small);
    suite.Rejects(
        [&] { bounded.RegisterImage(image); },
        "registration capacity failure rejects transactionally");
    suite.Check(bounded.Residual().Empty(),
                "failed registration exposes no partial image");

    auto no_core = Capacity(3);
    CollectiveWaveAdmissionCoordinatorV1 missing_core(no_core);
    suite.Rejects(
        [&] { missing_core.RegisterImage(image); },
        "missing core capacity rejects registration");

    auto low_bytes = Capacity(4, 8, 1);
    CollectiveWaveAdmissionCoordinatorV1 bytes(low_bytes);
    suite.Rejects(
        [&] { bytes.RegisterImage(image); },
        "per-core receive-byte bound is enforced");
}

} // namespace

int RunCollectiveWaveAdmissionV1SelfTest() {
    StrictWireScope strict;
    Suite suite;
    TestLifecycle(suite);
    TestPredecessorAndFairness(suite);
    TestRegistrationAbortAndBounds(suite);
    if (suite.failures == 0) {
        std::cout << "[COLLECTIVE WAVE ADMISSION V1] PASS ("
                  << suite.checks << " checks)\n";
        return 0;
    }
    std::cerr << "[COLLECTIVE WAVE ADMISSION V1] FAIL ("
              << suite.failures << "/" << suite.checks
              << " checks failed)\n";
    return suite.failures;
}

#ifdef COLLECTIVE_WAVE_ADMISSION_V1_SELFTEST_MAIN
int sc_main(int, char **) {
    return RunCollectiveWaveAdmissionV1SelfTest();
}
#endif
