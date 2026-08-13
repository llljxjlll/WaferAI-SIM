#include "dte/coll_accel_runtime_v1_selftest.h"

#include "dte/coll_accel_runtime_v1.h"
#include "dte/coll_innetwork_reduce.h"
#include "dte/coll_multicast.h"
#include "collective_wave_runtime_v1_test_fixture.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

using collective_wave_runtime_v1_test::BuildImage;
using collective_wave_runtime_v1_test::Cell;

struct Suite {
    int checks = 0;
    int failures = 0;
    void Check(bool value, const std::string &name) {
        ++checks;
        if (value) return;
        ++failures;
        std::cerr << "[ISA V1 ACCEL RUNTIME] FAIL: " << name << '\n';
    }
    template <class Error = std::exception, class Function>
    void Throws(Function &&function, const std::string &name) {
        bool threw = false;
        try { function(); } catch (const Error &) { threw = true; }
        Check(threw, name);
    }
};

class MeshGuard {
public:
    MeshGuard()
        : x_(GRID_X), y_(GRID_Y), size_(GRID_SIZE), dies_(DIE_COUNT),
          per_die_(CORES_PER_DIE), total_(TOTAL_CORES) {
        GRID_X = 4; GRID_Y = 2; GRID_SIZE = 8;
        DIE_COUNT = 1; CORES_PER_DIE = 8; TOTAL_CORES = 8;
    }
    ~MeshGuard() {
        GRID_X = x_; GRID_Y = y_; GRID_SIZE = size_;
        DIE_COUNT = dies_; CORES_PER_DIE = per_die_; TOTAL_CORES = total_;
    }
private:
    int x_, y_, size_, dies_, per_die_, total_;
};

IsaV1CollectiveProfileImageConfig Profile(NocCollProfile profile,
                                           size_t batch_limit = 1) {
    IsaV1CollectiveProfileImageConfig config;
    config.mesh = {4, 2, 1};
    config.noc.enabled = true;
    config.noc.profile = profile;
    config.noc.dca.value_mode = NocCollValueMode::INTEGER_EXACT;
    config.capabilities = {true, true, true};
    config.max_trees_per_batch = batch_limit;
    if (profile == NocCollProfile::BROADCAST_ONLY) {
        config.noc.broadcast_backend = NocCollBroadcastBackend::MULTICAST;
        config.noc.reduce_backend = NocCollReduceBackend::ENDPOINT;
    } else if (profile == NocCollProfile::REDUCE_BROADCAST) {
        config.noc.broadcast_backend = NocCollBroadcastBackend::MULTICAST;
        config.noc.reduce_backend = NocCollReduceBackend::DCA_OFFLOAD;
    } else {
        throw std::logic_error("self-test requests an unsupported profile");
    }
    return config;
}

struct Fixture {
    std::shared_ptr<const IsaV1CollectiveProgramImage> base;
    std::shared_ptr<const IsaV1CollectiveProfileProgramImage> profile;
    std::shared_ptr<IsaV1CollectiveTreeRegistryBridge> bridge;
};

Fixture Make(const std::vector<Cell> &cells, size_t ranks,
             NocCollProfile profile) {
    ResetCollectiveFabric();
    ResetCollectiveReduceFabric();
    auto base = std::make_shared<const IsaV1CollectiveProgramImage>(
        BuildImage(cells, ranks));
    auto profile_image =
        std::make_shared<const IsaV1CollectiveProfileProgramImage>(
            BuildIsaV1CollectiveProfileProgramImage(
                *base, Profile(profile)));
    IsaV1CollectiveTreeBatchRuntimeConfig bridge_config;
    bridge_config.max_registered_schedules = profile_image->Plans().size();
    bridge_config.max_registered_trees = profile_image->TreeCount();
    bridge_config.max_planned_entries = profile_image->TreeEntryCount();
    bridge_config.max_trace_events = profile_image->BatchCount() * 2;
    auto bridge = std::make_shared<IsaV1CollectiveTreeRegistryBridge>(
        bridge_config);
    bridge->RegisterImage(*profile_image);
    return {std::move(base), std::move(profile_image), std::move(bridge)};
}

Fixture Make(const Cell &cell, size_t ranks, NocCollProfile profile) {
    return Make(std::vector<Cell>{cell}, ranks, profile);
}

void ObserveAll(IsaV1CollectiveAccelerationRuntime &runtime,
                const IsaV1CollectivePlan &plan) {
    for (uint16_t core : plan.group)
        runtime.ObservePlanComplete(plan.key, core);
}

void TestMulticastAckGate(Suite &suite) {
    auto fixture = Make(
        {CollTxKind::BROADCAST, CollRxKind::UNICAST}, 3,
        NocCollProfile::BROADCAST_ONLY);
    IsaV1CollectiveAccelerationRuntime runtime(
        fixture.base, fixture.profile, fixture.bridge);
    const auto &plan = fixture.profile->Plans().front();
    const auto &tree_image = plan.trees.front();
    const uint16_t tree = tree_image.topology.tree_id;
    const auto &semantic = fixture.base->Lowering().plans.front();
    std::vector<uint16_t> targets;
    for (uint16_t core : semantic.group)
        if (core != semantic.group[tree_image.root_rank]) {
            targets.push_back(core);
            runtime.RegisterMulticastPost(plan.key, tree, 0, core);
        }
    suite.Throws<std::runtime_error>(
        [&] {
            runtime.RegisterMulticastPost(
                plan.key, tree, 0, targets.front());
        },
        "multicast target POST is exactly once");
    const auto session = runtime.TryAcquireMulticast(plan.key, tree, 0);
    suite.Check(session == 1 && CollectiveTreeEntryCountForTree(tree) != 0,
                "acquire begins the real batch and allocates session one");
    suite.Throws<std::runtime_error>(
        [&] { runtime.AcknowledgeMulticast(plan.key, tree, 0, 7); },
        "unknown multicast target ACK is rejected");
    runtime.AcknowledgeMulticast(plan.key, tree, 0, targets.front());
    suite.Throws<std::runtime_error>(
        [&] {
            runtime.AcknowledgeMulticast(
                plan.key, tree, 0, targets.front());
        },
        "duplicate multicast target ACK is rejected");
    suite.Check(CollectiveTreeEntryCountForTree(tree) != 0,
                "tree remains resident until every target commits");
    runtime.AcknowledgeMulticast(plan.key, tree, 0, targets.back());
    suite.Check(runtime.PlanComplete(plan.key) &&
                    !runtime.TryAcquireMulticast(plan.key, tree, 0) &&
                    runtime.Residual().plans == 1 &&
                    fixture.bridge->Residual().Empty() &&
                    CollectiveTreeEntryCountForTree(tree) == 0,
                "last ACK drains the tree but retains the plan for observers");
    runtime.ObservePlanComplete(plan.key, semantic.group.at(0));
    runtime.ObservePlanComplete(plan.key, semantic.group.at(1));
    suite.Check(runtime.Residual().plans == 1,
                "late final Worker can still observe a completed plan");
    runtime.ObservePlanComplete(plan.key, semantic.group.at(2));
    suite.Check(runtime.Residual().Empty(),
                "last participating Worker observation retires the plan");
    suite.Throws<std::runtime_error>(
        [&] { runtime.TryAcquireMulticast(plan.key, tree, 0); },
        "acquisition after all observers is genuinely stale");
    suite.Throws<std::runtime_error>(
        [&] { runtime.ObservePlanComplete(plan.key, semantic.group.at(2)); },
        "completion observation is exactly once");
}

void TestCombinedOrderingAndBatches(Suite &suite) {
    auto fixture = Make(
        {CollTxKind::BROADCAST, CollRxKind::REDUCE}, 2,
        NocCollProfile::REDUCE_BROADCAST);
    IsaV1CollectiveAccelerationRuntime runtime(
        fixture.base, fixture.profile, fixture.bridge);
    const auto &plan = fixture.profile->Plans().front();
    const auto &first = plan.trees.at(0);
    const auto &second = plan.trees.at(1);
    runtime.RegisterMulticastPost(
        plan.key, first.topology.tree_id, 0, 1);
    runtime.RegisterDcaRootReady(
        plan.key, first.topology.tree_id, 0, 0);
    suite.Check(runtime.TryAcquireDca(
                    plan.key, first.topology.tree_id, 0, 0) &&
                    !runtime.DcaSourcesReady(
                        plan.key, first.topology.tree_id, 0) &&
                    runtime.TryAcquireDca(
                        plan.key, first.topology.tree_id, 0, 1) &&
                    runtime.DcaSourcesReady(
                        plan.key, first.topology.tree_id, 0) &&
                    !runtime.TryAcquireMulticast(
                        plan.key, first.topology.tree_id, 0),
                "combined tree gates multicast until DCA SRAM write ACK");
    suite.Throws<std::runtime_error>(
        [&] { runtime.AcknowledgeDca(
            plan.key, first.topology.tree_id, 0, 1); },
        "DCA ACK from a non-root core is rejected");
    runtime.AcknowledgeDca(plan.key, first.topology.tree_id, 0, 0);
    const auto session1 = runtime.TryAcquireMulticast(
        plan.key, first.topology.tree_id, 0);
    runtime.AcknowledgeMulticast(
        plan.key, first.topology.tree_id, 0, 1);
    suite.Check(session1 == 1 &&
                    CollectiveTreeEntryCountForTree(
                        first.topology.tree_id) == 0 &&
                    !runtime.TryAcquireDca(
                        plan.key, first.topology.tree_id, 0, 0),
                "completed batch erases old tree and refuses stale acquisition");
    // The stale call above returns false only if the next batch is active for
    // another tree; it must not reprogram the retired first tree.
    runtime.RegisterMulticastPost(
        plan.key, second.topology.tree_id, 0, 0);
    runtime.RegisterDcaRootReady(
        plan.key, second.topology.tree_id, 0, 1);
    suite.Check(runtime.TryAcquireDca(
                    plan.key, second.topology.tree_id, 0, 0) &&
                    runtime.TryAcquireDca(
                        plan.key, second.topology.tree_id, 0, 1),
                "next canonical batch starts after prior tree drain");
    runtime.AcknowledgeDca(plan.key, second.topology.tree_id, 0, 1);
    const auto session2 = runtime.TryAcquireMulticast(
        plan.key, second.topology.tree_id, 0);
    runtime.AcknowledgeMulticast(
        plan.key, second.topology.tree_id, 0, 0);
    const auto &semantic = fixture.base->Lowering().plans.front();
    suite.Check(session2 == 2 && runtime.PlanComplete(plan.key) &&
                    fixture.bridge->Residual().Empty() &&
                    runtime.Stats().batches_started == 2 &&
                    runtime.Stats().batches_completed == 2,
                "sessions never reuse and both real batches drain exactly");
    ObserveAll(runtime, semantic);
    suite.Check(runtime.Residual().Empty(),
                "combined plan retires after all Worker observations");
}

void TestSessionPreflight(Suite &suite) {
    auto exact = Make(
        {CollTxKind::BROADCAST, CollRxKind::UNICAST}, 2,
        NocCollProfile::BROADCAST_ONLY);
    IsaV1CollectiveAccelerationRuntime exact_runtime(
        exact.base, exact.profile, exact.bridge, UINT16_MAX);
    const auto &exact_plan = exact.profile->Plans().front();
    const auto &exact_tree_image = exact_plan.trees.front();
    const uint16_t exact_tree = exact_tree_image.topology.tree_id;
    const auto &exact_semantic = exact.base->Lowering().plans.front();
    const uint16_t exact_target = exact_semantic.group[
        exact_tree_image.root_rank == 0 ? 1 : 0];
    exact_runtime.RegisterMulticastPost(
        exact_plan.key, exact_tree, 0, exact_target);
    suite.Check(exact_runtime.TryAcquireMulticast(
                    exact_plan.key, exact_tree, 0) == UINT16_MAX,
                "last legal session value is usable exactly once");
    exact_runtime.AcknowledgeMulticast(
        exact_plan.key, exact_tree, 0, exact_target);
    ObserveAll(exact_runtime, exact_semantic);

    auto overflow = Make(
        {CollTxKind::BROADCAST, CollRxKind::GATHER}, 2,
        NocCollProfile::BROADCAST_ONLY);
    suite.Throws<std::overflow_error>(
        [&] {
            IsaV1CollectiveAccelerationRuntime rejected(
                overflow.base, overflow.profile, overflow.bridge,
                UINT16_MAX);
        },
        "session limit plus one rejects before any tree programming");
    suite.Check(CollectiveTreeEntryCount() == 0 &&
                    CollectiveReduceNodeCount() == 0 &&
                    overflow.bridge->Residual().schedule.active_batches == 0,
                "failed session preflight is production-registry atomic");
    overflow.bridge->Abort(overflow.profile->Plans().front().key);
}

void CompleteCombinedPlan(
    Suite &suite, IsaV1CollectiveAccelerationRuntime &runtime,
    const IsaV1CollectiveProfilePlanImage &image,
    const IsaV1CollectivePlan &semantic) {
    for (const auto &tree : image.trees) {
        const uint16_t tree_id = tree.topology.tree_id;
        const uint16_t root = semantic.group.at(tree.root_rank);
        for (uint16_t core : semantic.group)
            if (core != root)
                runtime.RegisterMulticastPost(
                    image.key, tree_id, 0, core);
        runtime.RegisterDcaRootReady(image.key, tree_id, 0, root);
        bool all_sources = true;
        for (uint16_t core : semantic.group)
            all_sources = runtime.TryAcquireDca(
                              image.key, tree_id, 0, core) &&
                          all_sources;
        suite.Check(all_sources && runtime.DcaSourcesReady(
                                      image.key, tree_id, 0),
                    "four Workers acquire each canonical DCA batch");
        runtime.AcknowledgeDca(image.key, tree_id, 0, root);
        suite.Check(runtime.TryAcquireMulticast(
                        image.key, tree_id, 0).has_value(),
                    "combined batch starts multicast after DCA ACK");
        for (uint16_t core : semantic.group)
            if (core != root)
                runtime.AcknowledgeMulticast(
                    image.key, tree_id, 0, core);
    }
}

void TestFourWorkerTwoPlanLateRetirement(Suite &suite) {
    auto fixture = Make(
        {{CollTxKind::BROADCAST, CollRxKind::REDUCE},
         {CollTxKind::BROADCAST, CollRxKind::REDUCE}},
        4, NocCollProfile::REDUCE_BROADCAST);
    IsaV1CollectiveAccelerationRuntime runtime(
        fixture.base, fixture.profile, fixture.bridge);
    const auto &images = fixture.profile->Plans();
    const auto &semantics = fixture.base->Lowering().plans;
    suite.Check(images.size() == 2 &&
                    images.at(0).tree_schedule.batches.size() == 4 &&
                    images.at(1).tree_schedule.batches.size() == 4,
                "two four-Worker plans each use four canonical batches");

    CompleteCombinedPlan(suite, runtime, images.at(0), semantics.at(0));
    for (size_t rank = 0; rank + 1 < semantics.at(0).group.size(); ++rank)
        runtime.ObservePlanComplete(
            images.at(0).key, semantics.at(0).group.at(rank));
    suite.Check(runtime.Residual().plans == 2,
                "late observer keeps first plan queryable across next plan");

    CompleteCombinedPlan(suite, runtime, images.at(1), semantics.at(1));
    ObserveAll(runtime, semantics.at(1));
    runtime.ObservePlanComplete(
        images.at(0).key, semantics.at(0).group.back());
    suite.Check(runtime.Residual().Empty() &&
                    fixture.bridge->Residual().Empty() &&
                    runtime.Stats().batches_completed == 8,
                "two back-to-back plans retire only after all four Workers");
}

void TestAllReduceMulticastByteLayout(Suite &suite) {
    auto endpoint = Make(
        {CollTxKind::BROADCAST, CollRxKind::REDUCE}, 4,
        NocCollProfile::BROADCAST_ONLY);
    IsaV1CollectivePlan endpoint_plan =
        endpoint.base->Lowering().plans.front();
    // Exercise the production N=4, L=64 rank-major byte contract directly.
    endpoint_plan.length_bytes = 64;
    const auto &endpoint_image = endpoint.profile->Plans().front();
    bool endpoint_layout = true;
    for (const auto &tree : endpoint_image.trees) {
        endpoint_layout = endpoint_layout &&
            IsaV1CollectiveMulticastSourceAddress(
                endpoint_plan, tree, 0) ==
                endpoint_plan.rank_records.at(tree.root_rank)
                    .send.base_address_bytes;
        for (uint16_t rank = 0; rank < endpoint_plan.group.size(); ++rank) {
            if (rank == tree.root_rank) continue;
            const auto target = std::find_if(
                endpoint_plan.reduce_targets.begin(),
                endpoint_plan.reduce_targets.end(),
                [rank](const IsaV1ReduceTarget &value) {
                    return value.rank == rank;
                });
            endpoint_layout = endpoint_layout &&
                target != endpoint_plan.reduce_targets.end() &&
                IsaV1CollectiveMulticastDestinationAddress(
                    endpoint_plan, tree, rank, 0) ==
                    target->staging_address_bytes +
                        static_cast<uint64_t>(tree.root_rank) * 64;
        }
    }
    suite.Check(endpoint_layout,
                "N4 L64 endpoint all-reduce uses source-rank staging slots");

    auto combined = Make(
        {CollTxKind::BROADCAST, CollRxKind::REDUCE}, 4,
        NocCollProfile::REDUCE_BROADCAST);
    IsaV1CollectivePlan combined_plan =
        combined.base->Lowering().plans.front();
    combined_plan.length_bytes = 64;
    const auto &combined_image = combined.profile->Plans().front();
    bool result_layout = true;
    for (const auto &tree : combined_image.trees) {
        const auto root_target = std::find_if(
            combined_plan.reduce_targets.begin(),
            combined_plan.reduce_targets.end(),
            [&](const IsaV1ReduceTarget &value) {
                return value.rank == tree.root_rank;
            });
        result_layout = result_layout &&
            root_target != combined_plan.reduce_targets.end() &&
            IsaV1CollectiveMulticastSourceAddress(
                combined_plan, tree, 0) ==
                root_target->result_address_bytes;
        for (uint16_t rank = 0; rank < combined_plan.group.size(); ++rank) {
            if (rank == tree.root_rank) continue;
            const auto target = std::find_if(
                combined_plan.reduce_targets.begin(),
                combined_plan.reduce_targets.end(),
                [rank](const IsaV1ReduceTarget &value) {
                    return value.rank == rank;
                });
            result_layout = result_layout &&
                target != combined_plan.reduce_targets.end() &&
                IsaV1CollectiveMulticastDestinationAddress(
                    combined_plan, tree, rank, 0) ==
                    target->result_address_bytes;
        }
    }
    suite.Check(result_layout,
                "N4 L64 DCA all-reduce broadcasts committed result bytes");
}

} // namespace

int RunIsaV1CollectiveAccelerationRuntimeSelfTest() {
    MeshGuard mesh;
    Suite suite;
    TestMulticastAckGate(suite);
    TestCombinedOrderingAndBatches(suite);
    TestSessionPreflight(suite);
    TestFourWorkerTwoPlanLateRetirement(suite);
    TestAllReduceMulticastByteLayout(suite);
    ResetCollectiveFabric();
    ResetCollectiveReduceFabric();
    std::cout << "ISA v1 collective acceleration runtime self-test: "
              << (suite.failures == 0
                      ? "PASS"
                      : "FAILURES=" + std::to_string(suite.failures))
              << " (" << suite.checks << " checks)\n";
    return suite.failures;
}
