#include "dte/coll_program_profile_v1_selftest.h"

#include "dte/coll_program_profile_v1.h"
#include "dte/coll_multicast.h"
#include "dte/coll_tree_registry_bridge_v1.h"
#include "collective_wave_runtime_v1_test_fixture.h"

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using collective_wave_runtime_v1_test::BuildImage;
using collective_wave_runtime_v1_test::Cell;

class MeshGlobalsGuard {
public:
    MeshGlobalsGuard()
        : grid_x_(GRID_X), grid_y_(GRID_Y), grid_size_(GRID_SIZE),
          die_count_(DIE_COUNT), cores_per_die_(CORES_PER_DIE),
          total_cores_(TOTAL_CORES) {
        GRID_X = 4;
        GRID_Y = 2;
        GRID_SIZE = 8;
        DIE_COUNT = 1;
        CORES_PER_DIE = 8;
        TOTAL_CORES = 8;
    }
    ~MeshGlobalsGuard() {
        GRID_X = grid_x_;
        GRID_Y = grid_y_;
        GRID_SIZE = grid_size_;
        DIE_COUNT = die_count_;
        CORES_PER_DIE = cores_per_die_;
        TOTAL_CORES = total_cores_;
    }
private:
    int grid_x_;
    int grid_y_;
    int grid_size_;
    int die_count_;
    int cores_per_die_;
    int total_cores_;
};

struct Suite {
    int checks = 0;
    int failures = 0;

    void Check(bool condition, const std::string &name) {
        ++checks;
        if (condition) return;
        ++failures;
        std::cerr << "[ISA V1 PROFILE IMAGE] FAIL: " << name << '\n';
    }

    template <class Error = std::exception, class Function>
    void Throws(Function &&function, const std::string &name) {
        bool threw = false;
        try {
            function();
        } catch (const Error &) {
            threw = true;
        }
        Check(threw, name);
    }
};

NocCollectiveConfig Noc(NocCollProfile profile) {
    NocCollectiveConfig config;
    config.enabled = true;
    config.profile = profile;
    switch (profile) {
    case NocCollProfile::BASELINE:
        config.broadcast_backend = NocCollBroadcastBackend::UNICAST;
        config.reduce_backend = NocCollReduceBackend::ENDPOINT;
        break;
    case NocCollProfile::BROADCAST_ONLY:
        config.broadcast_backend = NocCollBroadcastBackend::MULTICAST;
        config.reduce_backend = NocCollReduceBackend::ENDPOINT;
        break;
    case NocCollProfile::REDUCE_ONLY:
        config.broadcast_backend = NocCollBroadcastBackend::UNICAST;
        config.reduce_backend = NocCollReduceBackend::DCA_OFFLOAD;
        break;
    case NocCollProfile::REDUCE_BROADCAST:
        config.broadcast_backend = NocCollBroadcastBackend::MULTICAST;
        config.reduce_backend = NocCollReduceBackend::DCA_OFFLOAD;
        break;
    }
    config.dca.value_mode = NocCollValueMode::INTEGER_EXACT;
    return config;
}

IsaV1CollectiveProfileImageConfig Config(NocCollProfile profile,
                                         size_t batch_limit = 2) {
    IsaV1CollectiveProfileImageConfig config;
    config.mesh = {4, 2, 1};
    config.noc = Noc(profile);
    config.capabilities = {true, true, true};
    config.max_trees_per_batch = batch_limit;
    return config;
}

std::vector<Cell> NineCells() {
    return {{CollTxKind::UNICAST, CollRxKind::UNICAST},
            {CollTxKind::SCATTER, CollRxKind::UNICAST},
            {CollTxKind::BROADCAST, CollRxKind::UNICAST},
            {CollTxKind::UNICAST, CollRxKind::GATHER},
            {CollTxKind::SCATTER, CollRxKind::GATHER},
            {CollTxKind::BROADCAST, CollRxKind::GATHER},
            {CollTxKind::UNICAST, CollRxKind::REDUCE},
            {CollTxKind::SCATTER, CollRxKind::REDUCE},
            {CollTxKind::BROADCAST, CollRxKind::REDUCE}};
}

const IsaV1CollectiveProfilePlanImage &PlanFor(
    const IsaV1CollectiveProfileProgramImage &image, CollOp op) {
    const auto found = std::find_if(
        image.Plans().begin(), image.Plans().end(),
        [op](const auto &plan) { return plan.decision.op == op; });
    if (found == image.Plans().end())
        throw std::logic_error("self-test profile plan is missing");
    return *found;
}

void TestFourProfiles(Suite &suite) {
    const auto base = BuildImage(NineCells(), 4);
    struct Expected {
        NocCollProfile profile;
        size_t multicast;
        size_t dca;
    };
    const std::array<Expected, 4> expected{{
        {NocCollProfile::BASELINE, 0, 0},
        {NocCollProfile::BROADCAST_ONLY, 9, 0},
        {NocCollProfile::REDUCE_ONLY, 0, 9},
        {NocCollProfile::REDUCE_BROADCAST, 9, 9},
    }};
    for (const Expected &cell : expected) {
        const auto image = BuildIsaV1CollectiveProfileProgramImage(
            base, Config(cell.profile));
        suite.Check(image.BaseGeneration() == base.Generation() &&
                        image.BaseCookie() == base.Cookie() &&
                        image.Plans().size() == NineCells().size(),
                    "profile is layered over unchanged P6 semantic image");
        suite.Check(image.MulticastTreeCount() == cell.multicast &&
                        image.DcaTreeCount() == cell.dca,
                    "four-way profile selects exact orthogonal tree counts");
        const auto &scatter = PlanFor(image, CollOp::SCATTER);
        suite.Check(scatter.trees.empty() &&
                        scatter.decision.broadcast_backend ==
                            IsaV1ProfileBroadcastBackend::NOT_APPLICABLE,
                    "Scatter never enters multicast tree programming");
        const auto &reduce = PlanFor(image, CollOp::REDUCE);
        suite.Check(reduce.suppress_endpoint_reduce_compute ==
                        (cell.dca != 0) &&
                        reduce.requires_dca_payload_executor ==
                            (cell.dca != 0),
                    "DCA explicitly replaces endpoint REDUCE_COMPUTE");
    }
}

void TestGatesAndDeterminism(Suite &suite) {
    const auto reduce = BuildImage(
        {{CollTxKind::UNICAST, CollRxKind::REDUCE}}, 4);
    auto no_dca = Config(NocCollProfile::REDUCE_ONLY);
    no_dca.capabilities.dca = false;
    suite.Throws<std::runtime_error>(
        [&] { (void)BuildIsaV1CollectiveProfileProgramImage(reduce, no_dca); },
        "requested DCA never silently falls back to endpoint");

    const auto rs = BuildImage(
        {{CollTxKind::SCATTER, CollRxKind::REDUCE}}, 4);
    auto closed_rs = Config(NocCollProfile::REDUCE_ONLY);
    closed_rs.capabilities.reduce_scatter_dca = false;
    suite.Throws<std::runtime_error>(
        [&] { (void)BuildIsaV1CollectiveProfileProgramImage(rs, closed_rs); },
        "ReduceScatter DCA has an independent closed gate");

    auto timing_only = Config(NocCollProfile::REDUCE_ONLY);
    timing_only.noc.dca.value_mode = NocCollValueMode::TIMING_ONLY;
    suite.Throws<std::invalid_argument>(
        [&] {
            (void)BuildIsaV1CollectiveProfileProgramImage(reduce,
                                                          timing_only);
        },
        "DCA profile rejects timing-only instead of claiming byte correctness");

    auto mismatch = Config(NocCollProfile::BROADCAST_ONLY);
    mismatch.noc.broadcast_backend = NocCollBroadcastBackend::UNICAST;
    suite.Throws<std::invalid_argument>(
        [&] { (void)BuildIsaV1CollectiveProfileProgramImage(rs, mismatch); },
        "named profile/backend mismatch fails fast");

    auto exhausted = Config(NocCollProfile::BROADCAST_ONLY);
    exhausted.occupied_entries_by_router[1] =
        exhausted.entries_per_router;
    const auto broadcast = BuildImage(
        {{CollTxKind::BROADCAST, CollRxKind::UNICAST}}, 4);
    auto cross_die = Config(NocCollProfile::BROADCAST_ONLY);
    cross_die.mesh = {2, 1, 2};
    suite.Throws<std::runtime_error>(
        [&] {
            (void)BuildIsaV1CollectiveProfileProgramImage(broadcast,
                                                          cross_die);
        },
        "accelerated group spanning two dies is rejected at image load");
    const auto same_die_broadcast = BuildImage(
        {{CollTxKind::BROADCAST, CollRxKind::UNICAST}}, 2);
    const auto same_die = BuildIsaV1CollectiveProfileProgramImage(
        same_die_broadcast, cross_die);
    suite.Check(same_die.MulticastTreeCount() == 1,
                "multi-die platform accepts a group confined to one die");
    suite.Throws<std::runtime_error>(
        [&] {
            (void)BuildIsaV1CollectiveProfileProgramImage(broadcast,
                                                          exhausted);
        },
        "existing Router occupancy participates in profile batch capacity");

    const auto allreduce = BuildImage(
        {{CollTxKind::BROADCAST, CollRxKind::REDUCE}}, 4);
    const auto left = BuildIsaV1CollectiveProfileProgramImage(
        allreduce, Config(NocCollProfile::REDUCE_BROADCAST, 1));
    const auto right = BuildIsaV1CollectiveProfileProgramImage(
        allreduce, Config(NocCollProfile::REDUCE_BROADCAST, 1));
    suite.Check(left.Plans() == right.Plans() && left.TreeCount() == 4,
                "tree IDs, X-first paths and schedules are deterministic");
}

void TestCapacityBounds(Suite &suite) {
    ResetCollectiveFabric();
    ResetCollectiveReduceFabric();
    const auto base = BuildImage(
        {{CollTxKind::BROADCAST, CollRxKind::GATHER}}, 4);
    const auto reference = BuildIsaV1CollectiveProfileProgramImage(
        base, Config(NocCollProfile::BROADCAST_ONLY, 1));
    suite.Check(reference.TreeCount() == 4 &&
                    reference.TreeEntryCount() != 0 &&
                    reference.PotentialConflictEdgeCount() == 6 &&
                    reference.BatchCount() == 4 &&
                    reference.DerivedBytes() != 0,
                "profile image reports every bounded derived dimension");
    suite.Throws<std::invalid_argument>(
        [&] {
            (void)BuildIsaV1CollectiveProfileProgramImage(
                base, Config(NocCollProfile::BROADCAST_ONLY,
                             kNocCollMaxTreesPerBatch + 1));
        },
        "production max_trees_per_batch limit plus one rejects");

    auto exact = Config(NocCollProfile::BROADCAST_ONLY, 1);
    exact.max_total_trees = reference.TreeCount();
    exact.max_total_tree_entries = reference.TreeEntryCount();
    exact.max_total_conflict_edges =
        reference.PotentialConflictEdgeCount();
    exact.max_total_batches = reference.BatchCount();
    exact.max_derived_bytes = reference.DerivedBytes();
    const auto exact_image =
        BuildIsaV1CollectiveProfileProgramImage(base, exact);
    suite.Check(exact_image.TreeCount() == reference.TreeCount() &&
                    exact_image.TreeEntryCount() ==
                        reference.TreeEntryCount() &&
                    exact_image.DerivedBytes() == reference.DerivedBytes(),
                "all limits accept the exact boundary");

    auto one_extra_tree = exact;
    one_extra_tree.max_total_trees = reference.TreeCount() - 1;
    suite.Throws<std::overflow_error>(
        [&] {
            (void)BuildIsaV1CollectiveProfileProgramImage(base,
                                                          one_extra_tree);
        },
        "limit+1 tree is rejected before topology construction");
    auto one_extra_entry = exact;
    one_extra_entry.max_total_tree_entries =
        reference.TreeEntryCount() - 1;
    suite.Throws<std::overflow_error>(
        [&] {
            (void)BuildIsaV1CollectiveProfileProgramImage(base,
                                                          one_extra_entry);
        },
        "limit+1 tree entry is rejected by topology preflight");
    auto one_extra_byte = exact;
    one_extra_byte.max_derived_bytes = reference.DerivedBytes() - 1;
    suite.Throws<std::overflow_error>(
        [&] {
            (void)BuildIsaV1CollectiveProfileProgramImage(base,
                                                          one_extra_byte);
        },
        "limit+1 conservative derived byte is rejected");
    auto one_extra_edge = exact;
    one_extra_edge.max_total_conflict_edges =
        reference.PotentialConflictEdgeCount() - 1;
    suite.Throws<std::overflow_error>(
        [&] {
            (void)BuildIsaV1CollectiveProfileProgramImage(base,
                                                          one_extra_edge);
        },
        "limit+1 potential conflict edge is rejected before scheduling");
    suite.Check(CollectiveTreeEntryCount() == 0 &&
                    CollectiveReduceNodeCount() == 0,
                "all profile capacity failures are registry-atomic");
}

void TestProductionRegistryBridge(Suite &suite) {
    MeshGlobalsGuard mesh_globals;
    static_cast<void>(mesh_globals);
    ResetCollectiveFabric();
    ResetCollectiveReduceFabric();
    const auto base = BuildImage(
        {{CollTxKind::BROADCAST, CollRxKind::GATHER}}, 4);
    const auto image = BuildIsaV1CollectiveProfileProgramImage(
        base, Config(NocCollProfile::BROADCAST_ONLY, 1));
    const auto &plan = image.Plans().front();
    suite.Check(plan.trees.size() == 4 &&
                    plan.tree_schedule.batches.size() == 4,
                "AllGather K=1 has four non-resident batches");

    IsaV1CollectiveTreeRegistryBridge bridge;
    bridge.RegisterPlan(plan);
    suite.Check(CollectiveTreeEntryCount() == 0 &&
                    bridge.Residual().schedule.active_batches == 0,
                "registration does not preload future source trees");
    for (const IsaV1TreeBatch &batch : plan.tree_schedule.batches) {
        bridge.BeginBatch(plan.key, batch.batch_index);
        const uint16_t tree_id = batch.tree_ids.front();
        const auto found = std::find_if(
            plan.trees.begin(), plan.trees.end(),
            [tree_id](const auto &tree) {
                return tree.topology.tree_id == tree_id;
            });
        suite.Check(found != plan.trees.end() &&
                        CollectiveTreeEntryCount() ==
                            found->topology.entries.size() &&
                        CollectiveTreeEntryCountForTree(tree_id) ==
                            found->topology.entries.size(),
                    "BeginBatch programs exactly the current production tree");
        for (uint16_t id : batch.tree_ids)
            bridge.MarkTreeComplete(plan.key, batch.batch_index, id);
        bridge.EndBatch(plan.key, batch.batch_index);
        suite.Check(CollectiveTreeEntryCount() == 0,
                    "EndBatch erases current tree before next batch");
    }
    suite.Check(bridge.Residual().Empty() &&
                    bridge.Stats().batches_programmed == 4 &&
                    bridge.Stats().batches_released == 4 &&
                    bridge.Stats().multicast_entries_programmed ==
                        bridge.Stats().multicast_entries_erased,
                "AllGather production registry and bridge fully drain");

    const auto ar_base = BuildImage(
        {{CollTxKind::BROADCAST, CollRxKind::REDUCE}}, 4);
    const auto ar_image = BuildIsaV1CollectiveProfileProgramImage(
        ar_base, Config(NocCollProfile::REDUCE_BROADCAST, 1));
    const auto &ar = ar_image.Plans().front();
    IsaV1CollectiveTreeRegistryBridge combined;
    combined.RegisterPlan(ar);
    const auto &first = ar.tree_schedule.batches.front();
    combined.BeginBatch(ar.key, first.batch_index);
    suite.Check(CollectiveTreeEntryCount() != 0 &&
                    CollectiveReduceNodeCount() != 0,
                "combined profile really programs multicast and DCA registries");
    suite.Throws<std::runtime_error>(
        [&] { combined.EndBatch(ar.key, first.batch_index); },
        "release before tree completion is rejected while entries stay live");
    suite.Check(CollectiveTreeEntryCount() != 0 &&
                    CollectiveReduceNodeCount() != 0,
                "failed early release preserves the active production batch");
    combined.Abort(ar.key);
    suite.Check(combined.Residual().Empty() &&
                    CollectiveTreeEntryCount() == 0 &&
                    CollectiveReduceNodeCount() == 0,
                "Abort erases both production registries and all ownership");

    ProgramCollectiveTreeEntry({ar.trees.front().topology.tree_id, 0, CENTER},
                               1U << EAST);
    IsaV1CollectiveTreeRegistryBridge collision;
    suite.Throws<std::runtime_error>(
        [&] { collision.RegisterPlan(ar); },
        "pre-existing production tree ID collision is never reused");
    suite.Check(CollectiveTreeEntryCount() == 1 &&
                    collision.Residual().Empty(),
                "collision rejection is transactional");
    ResetCollectiveFabric();
    ResetCollectiveReduceFabric();
}

} // namespace

int RunIsaV1CollectiveProgramProfileSelfTest() {
    Suite suite;
    TestFourProfiles(suite);
    TestGatesAndDeterminism(suite);
    TestCapacityBounds(suite);
    TestProductionRegistryBridge(suite);
    if (suite.failures == 0)
        std::cout << "[ISA V1 PROFILE IMAGE] PASS: " << suite.checks
                  << " checks\n";
    return suite.failures;
}

#ifdef ISA_V1_COLL_PROGRAM_PROFILE_SELFTEST_MAIN
int main() { return RunIsaV1CollectiveProgramProfileSelfTest(); }
#endif
