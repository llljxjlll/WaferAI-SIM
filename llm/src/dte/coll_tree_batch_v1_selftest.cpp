#include "dte/coll_tree_batch_v1_selftest.h"

#include "dte/coll_tree_batch_v1.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Suite {
    int checks = 0;
    int failures = 0;

    void Check(bool condition, const std::string &name) {
        ++checks;
        if (condition) return;
        ++failures;
        std::cerr << "[ISA V1 TREE BATCH] FAIL: " << name << '\n';
    }

    template <class E = std::exception, class F>
    void Throws(F &&fn, const std::string &name) {
        bool threw = false;
        try {
            fn();
        } catch (const E &) {
            threw = true;
        }
        Check(threw, name);
    }
};

std::vector<IsaV1CollectiveTreeTopology> AllGatherTrees(
    const IsaV1MeshShape &shape, const std::vector<uint16_t> &group,
    uint16_t first_tree_id) {
    std::vector<IsaV1CollectiveTreeTopology> trees;
    for (size_t source = 0; source < group.size(); ++source)
        trees.push_back(BuildIsaV1XFirstCollectiveTree(
            shape, static_cast<uint16_t>(first_tree_id + source),
            group[source], group));
    return trees;
}

void CompleteBatch(IsaV1CollectiveTreeBatchRuntime &runtime,
                   const CollectiveKey &key, const IsaV1TreeBatch &batch) {
    runtime.BeginBatch(key, batch.batch_index);
    for (uint16_t tree : batch.tree_ids)
        runtime.MarkTreeComplete(key, batch.batch_index, tree);
    runtime.EndBatch(key, batch.batch_index);
}

} // namespace

int RunIsaV1CollectiveTreeBatchSelfTest() {
    Suite suite;
    const IsaV1MeshShape shape{4, 4, 1};
    const std::vector<uint16_t> group{0, 3, 12, 15};
    auto allgather = AllGatherTrees(shape, group, 100);
    std::reverse(allgather.begin(), allgather.end());
    const CollectiveKey epoch0{7, 9, 0};

    IsaV1CollectiveTreeBatchRuntime runtime;
    suite.Check(runtime.EntriesPerRouterCapacity() ==
                    kIsaV1CollectiveTreeEntriesPerRouter,
                "runtime exposes the exact production Router capacity");
    const auto schedule = runtime.RegisterSchedule(epoch0, allgather, 1);
    suite.Check(schedule.batches.size() == 4 &&
                    schedule.batches[0].tree_ids ==
                        std::vector<uint16_t>{100} &&
                    runtime.ProgrammedTreeIds().empty(),
                "AllGather input is canonicalized and registration is non-resident");
    suite.Check(runtime.Residual().registered_schedules == 1 &&
                    runtime.Residual().registered_trees == 4 &&
                    runtime.Residual().active_batches == 0,
                "registered AllGather plan has bounded non-programmed residual");

    runtime.BeginBatch(epoch0, 0);
    suite.Check(runtime.ProgrammedTreeIds() == std::vector<uint16_t>{100} &&
                    !runtime.IsTreeProgrammed(101) &&
                    !runtime.IsTreeProgrammed(102) &&
                    !runtime.IsTreeProgrammed(103),
                "only current AllGather source tree is resident");
    const auto before_early_release = runtime.ProgrammedEntries();
    suite.Throws<std::runtime_error>(
        [&] { runtime.EndBatch(epoch0, 0); },
        "batch cannot release before its tree completes");
    suite.Check(runtime.ProgrammedEntries() == before_early_release,
                "failed early release is strongly non-mutating");
    suite.Throws<std::runtime_error>(
        [&] { runtime.MarkTreeComplete(epoch0, 0, 101); },
        "completion for a future AllGather source is rejected");
    runtime.MarkTreeComplete(epoch0, 0, 100);
    suite.Throws<std::runtime_error>(
        [&] { runtime.MarkTreeComplete(epoch0, 0, 100); },
        "duplicate tree completion is rejected");
    suite.Throws<std::runtime_error>(
        [&] { runtime.EndBatch(epoch0, 1); },
        "release batch mismatch is rejected");
    runtime.EndBatch(epoch0, 0);
    suite.Check(runtime.ProgrammedEntries().empty() &&
                    runtime.Residual().active_batches == 0,
                "batch release erases every current source entry");
    suite.Throws<std::runtime_error>(
        [&] { runtime.BeginBatch(epoch0, 0); },
        "released batch cannot be replayed");
    suite.Throws<std::runtime_error>(
        [&] { runtime.BeginBatch(epoch0, 2); },
        "future batch cannot skip required batch synchronization");

    for (size_t i = 1; i < schedule.batches.size(); ++i) {
        runtime.BeginBatch(epoch0, static_cast<uint16_t>(i));
        const auto resident = runtime.ProgrammedTreeIds();
        suite.Check(resident == schedule.batches[i].tree_ids &&
                        resident.size() == 1,
                    "each AllGather batch programs exactly its scheduled source");
        for (uint16_t tree : schedule.batches[i].tree_ids)
            runtime.MarkTreeComplete(epoch0, static_cast<uint16_t>(i), tree);
        runtime.EndBatch(epoch0, static_cast<uint16_t>(i));
    }
    suite.Check(runtime.Residual().Empty(),
                "final AllGather batch retires schedule with residual zero");
    suite.Throws<std::runtime_error>(
        [&] { runtime.Abort(epoch0); },
        "retired exact epoch is stale for lifecycle calls");

    const auto stats = runtime.Stats();
    suite.Check(stats.batches_programmed == 4 &&
                    stats.batches_released == 4 &&
                    stats.trees_programmed == 4 &&
                    stats.trees_erased == 4 &&
                    stats.entries_programmed == stats.entries_erased &&
                    stats.release_batch_complete == 3 &&
                    stats.release_schedule_complete == 1,
                "program/erase and release-reason statistics are exact");
    const auto trace = runtime.TraceEvents();
    suite.Check(trace.size() == 8 &&
                    trace.front().kind ==
                        IsaV1TreeBatchTraceEventKind::PROGRAM &&
                    trace.front().tree_ids == std::vector<uint16_t>{100} &&
                    trace.back().release_reason ==
                        IsaV1TreeBatchReleaseReason::SCHEDULE_COMPLETE &&
                    trace.back().managed_occupancy_after == 0,
                "trace records exact program/release lifecycle and final reason");

    suite.Throws<std::runtime_error>(
        [&] { (void)runtime.RegisterSchedule(epoch0, allgather, 1); },
        "retired epoch cannot be registered again");
    const CollectiveKey epoch1{7, 9, 1};
    auto epoch1_trees = AllGatherTrees(shape, group, 100);
    const auto epoch1_schedule = runtime.RegisterSchedule(epoch1, epoch1_trees, 2);
    for (const auto &batch : epoch1_schedule.batches)
        CompleteBatch(runtime, epoch1, batch);
    suite.Check(runtime.Residual().Empty(),
                "next epoch reuses released tree ids and drains to zero");

    IsaV1CollectiveTreeBatchRuntimeConfig boundary_config;
    boundary_config.reserved_entries_by_router[0] = 63;
    IsaV1CollectiveTreeBatchRuntime boundary(boundary_config);
    const CollectiveKey boundary_key{10, 10, 0};
    const auto boundary_tree =
        BuildIsaV1XFirstCollectiveTree(shape, 200, 0, {0, 1});
    const auto boundary_schedule =
        boundary.RegisterSchedule(boundary_key, {boundary_tree});
    boundary.BeginBatch(boundary_key, 0);
    suite.Check(boundary.Stats().peak_entries_by_router.at(0) == 64 &&
                    boundary.Stats().total_occupancy_peak == 65,
                "63 reserved plus current root reaches exact per-Router 64 boundary");
    boundary.MarkTreeComplete(boundary_key, 0, 200);
    boundary.EndBatch(boundary_key, 0);
    suite.Check(boundary.Residual().Empty() &&
                    boundary_schedule.batches.size() == 1,
                "64-boundary schedule releases all managed state");

    boundary_config.reserved_entries_by_router[0] = 64;
    IsaV1CollectiveTreeBatchRuntime full(boundary_config);
    suite.Throws<std::runtime_error>(
        [&] { (void)full.RegisterSchedule({11, 11, 0}, {boundary_tree}); },
        "64 reserved entries reject one additional tree at registration");
    suite.Check(full.Residual().Empty(),
                "capacity rejection leaves registration atomically empty");
    boundary_config.reserved_entries_by_router[0] = 65;
    suite.Throws<std::invalid_argument>(
        [&] { IsaV1CollectiveTreeBatchRuntime invalid(boundary_config); },
        "reserved occupancy above 64 is rejected at construction");

    IsaV1CollectiveTreeBatchRuntime multi;
    const CollectiveKey group_a{20, 1, 0};
    const CollectiveKey group_b{21, 1, 0};
    const auto tree_a =
        BuildIsaV1XFirstCollectiveTree(shape, 300, 0, {0, 1});
    const auto tree_b =
        BuildIsaV1XFirstCollectiveTree(shape, 301, 4, {4, 5});
    multi.RegisterSchedule(group_a, {tree_a});
    multi.RegisterSchedule(group_b, {tree_b});
    multi.BeginBatch(group_a, 0);
    const auto group_a_entries = multi.ProgrammedEntries();
    suite.Throws<std::runtime_error>(
        [&] { multi.BeginBatch(group_b, 0); },
        "conservative runtime forbids a second globally active batch");
    suite.Check(multi.ProgrammedEntries() == group_a_entries,
                "failed concurrent begin cannot disturb active group");
    multi.Abort(group_b);
    suite.Check(multi.ProgrammedEntries() == group_a_entries &&
                    multi.IsTreeProgrammed(300),
                "aborting inactive group preserves exact active group state");
    multi.MarkTreeComplete(group_a, 0, 300);
    multi.EndBatch(group_a, 0);
    suite.Check(multi.Residual().Empty(),
                "multiple group schedules retire with global residual zero");

    IsaV1CollectiveTreeBatchRuntime collision;
    collision.RegisterSchedule({30, 1, 0}, {tree_a});
    const auto collision_before = collision.Residual();
    auto colliding_tree = tree_b;
    colliding_tree.tree_id = tree_a.tree_id;
    suite.Throws<std::runtime_error>(
        [&] {
            (void)collision.RegisterSchedule({31, 1, 0}, {colliding_tree});
        },
        "live tree_id collision across groups is rejected");
    suite.Check(collision.Residual().registered_schedules ==
                    collision_before.registered_schedules &&
                    collision.Residual().registered_trees ==
                        collision_before.registered_trees,
                "tree collision rejection is registration-atomic");
    collision.Abort({30, 1, 0});

    IsaV1CollectiveTreeBatchRuntime aborting;
    const CollectiveKey abort_key{40, 2, 3};
    aborting.RegisterSchedule(abort_key, {tree_a});
    aborting.BeginBatch(abort_key, 0);
    aborting.Abort(abort_key);
    suite.Check(aborting.Residual().Empty() &&
                    aborting.Stats().schedules_aborted == 1 &&
                    aborting.Stats().release_abort == 1 &&
                    aborting.Stats().entries_programmed ==
                        aborting.Stats().entries_erased &&
                    aborting.TraceEvents().back().release_reason ==
                        IsaV1TreeBatchReleaseReason::ABORT,
                "active abort precisely erases entries and records reason");
    suite.Throws<std::runtime_error>(
        [&] { (void)aborting.RegisterSchedule(abort_key, {tree_a}); },
        "aborted epoch is stale and cannot be reused");

    auto invalid_tree = tree_a;
    invalid_tree.entries[0].ingress = EAST;
    IsaV1CollectiveTreeBatchRuntime invalid_image_runtime;
    suite.Throws<std::invalid_argument>(
        [&] {
            (void)invalid_image_runtime.RegisterSchedule(
                {50, 1, 0}, {invalid_tree});
        },
        "malformed root ingress is rejected before mutation");
    suite.Check(invalid_image_runtime.Residual().Empty(),
                "invalid topology registration leaves residual zero");

    IsaV1CollectiveTreeBatchRuntimeConfig bounded_config;
    bounded_config.max_registered_schedules = 1;
    bounded_config.max_registered_trees = 1;
    bounded_config.max_planned_entries = tree_a.entries.size();
    bounded_config.max_trace_events = 2;
    IsaV1CollectiveTreeBatchRuntime bounded(bounded_config);
    bounded.RegisterSchedule({60, 1, 0}, {tree_a});
    suite.Throws<std::overflow_error>(
        [&] { (void)bounded.RegisterSchedule({61, 1, 0}, {tree_b}); },
        "registered schedule bound is enforced without mutation");
    bounded.BeginBatch({60, 1, 0}, 0);
    bounded.MarkTreeComplete({60, 1, 0}, 0, 300);
    bounded.EndBatch({60, 1, 0}, 0);
    auto reused = tree_a;
    reused.tree_id = 302;
    bounded.RegisterSchedule({61, 1, 0}, {reused});
    bounded.BeginBatch({61, 1, 0}, 0);
    bounded.Abort({61, 1, 0});
    suite.Check(bounded.TraceEvents().size() == 2 &&
                    bounded.Stats().trace_events_dropped == 2 &&
                    bounded.Residual().Empty(),
                "trace ring and all runtime registries remain strictly bounded");

    if (suite.failures == 0)
        std::cout << "[ISA V1 TREE BATCH] PASS: " << suite.checks
                  << " checks\n";
    return suite.failures;
}

#ifdef ISA_V1_COLL_TREE_BATCH_SELFTEST_MAIN
int main() { return RunIsaV1CollectiveTreeBatchSelfTest(); }
#endif
