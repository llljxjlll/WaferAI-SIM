#include "dte/coll_topology_v1_selftest.h"

#include "dte/coll_topology_v1.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
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
        std::cerr << "[ISA V1 COLL TOPOLOGY] FAIL: " << name << '\n';
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

IsaV1RouterOutput R(uint16_t router, Directions output) {
    return {router, output};
}

IsaV1TreeScheduleInput Tree(
    uint16_t id, std::vector<IsaV1RouterOutput> resources,
    std::vector<uint16_t> routers, bool proven = true) {
    return {id, proven, std::move(resources), std::move(routers)};
}

size_t BatchOf(const IsaV1TreeSchedule &schedule, uint16_t tree_id) {
    for (size_t i = 0; i < schedule.batches.size(); ++i)
        if (std::find(schedule.batches[i].tree_ids.begin(),
                      schedule.batches[i].tree_ids.end(), tree_id) !=
            schedule.batches[i].tree_ids.end())
            return i;
    throw std::logic_error("tree missing from test schedule");
}

std::vector<IsaV1TreeScheduleInput> GraphTrees(
    uint16_t count, const std::vector<std::pair<uint16_t, uint16_t>> &edges) {
    std::vector<IsaV1TreeScheduleInput> trees;
    for (uint16_t i = 0; i < count; ++i)
        trees.push_back(Tree(static_cast<uint16_t>(i + 1), {},
                             {static_cast<uint16_t>(100 + i)}));
    uint16_t resource = 0;
    for (const auto &edge : edges) {
        const auto shared = R(static_cast<uint16_t>(1000 + resource), EAST);
        trees.at(edge.first).resources.push_back(shared);
        trees.at(edge.second).resources.push_back(shared);
        trees.at(edge.first).entry_routers.push_back(shared.router_id);
        trees.at(edge.second).entry_routers.push_back(shared.router_id);
        ++resource;
    }
    return trees;
}

void CheckBatchesSafe(Suite &suite,
                      const std::vector<IsaV1TreeScheduleInput> &trees,
                      const IsaV1TreeSchedule &schedule,
                      const std::string &name) {
    std::map<uint16_t, std::vector<IsaV1RouterOutput>> resources;
    for (const auto &tree : trees) resources[tree.tree_id] = tree.resources;
    bool safe = true;
    for (const auto &batch : schedule.batches) {
        std::vector<IsaV1RouterOutput> used;
        for (uint16_t tree_id : batch.tree_ids) {
            auto next = resources.at(tree_id);
            std::sort(next.begin(), next.end());
            for (const auto &resource : next)
                if (std::find(used.begin(), used.end(), resource) != used.end())
                    safe = false;
                else
                    used.push_back(resource);
        }
    }
    suite.Check(safe, name);
}

} // namespace

int RunIsaV1CollectiveTopologySelfTest() {
    Suite suite;
    const IsaV1MeshShape shape{4, 4, 1};

    suite.Check(BuildIsaV1XFirstPath(shape, 0, 3) ==
                    std::vector<IsaV1RouterOutput>{
                        R(0, EAST), R(1, EAST), R(2, EAST), R(3, CENTER)},
                "same-Y path is X-first and terminates at CENTER");
    suite.Check(BuildIsaV1XFirstPath(shape, 0, 12) ==
                    std::vector<IsaV1RouterOutput>{
                        R(0, NORTH), R(4, NORTH), R(8, NORTH), R(12, CENTER)},
                "same-X path follows Y and terminates at CENTER");
    suite.Check(BuildIsaV1XFirstPath(shape, 0, 15) ==
                    std::vector<IsaV1RouterOutput>{
                        R(0, EAST), R(1, EAST), R(2, EAST), R(3, NORTH),
                        R(7, NORTH), R(11, NORTH), R(15, CENTER)},
                "cross-X-Y path completes all X hops before Y hops");
    suite.Check(BuildIsaV1XFirstPath(shape, 15, 0) ==
                    std::vector<IsaV1RouterOutput>{
                        R(15, WEST), R(14, WEST), R(13, WEST), R(12, SOUTH),
                        R(8, SOUTH), R(4, SOUTH), R(0, CENTER)},
                "reverse path remains X-first with reversed directions");
    suite.Check(BuildIsaV1XFirstPath(shape, 6, 6) ==
                    std::vector<IsaV1RouterOutput>{R(6, CENTER)},
                "zero-hop path exposes only destination CENTER");

    const auto topology = BuildIsaV1XFirstCollectiveTree(
        shape, 17, 0, {15, 12, 3, 0});
    const std::vector<IsaV1CollectiveTreeEntry> expected_entries{
        {0, CENTER, static_cast<uint8_t>((1U << EAST) | (1U << NORTH))},
        {1, WEST, static_cast<uint8_t>(1U << EAST)},
        {2, WEST, static_cast<uint8_t>(1U << EAST)},
        {3, WEST, static_cast<uint8_t>((1U << NORTH) | (1U << CENTER))},
        {4, SOUTH, static_cast<uint8_t>(1U << NORTH)},
        {7, SOUTH, static_cast<uint8_t>(1U << NORTH)},
        {8, SOUTH, static_cast<uint8_t>(1U << NORTH)},
        {11, SOUTH, static_cast<uint8_t>(1U << NORTH)},
        {12, SOUTH, static_cast<uint8_t>(1U << CENTER)},
        {15, SOUTH, static_cast<uint8_t>(1U << CENTER)}};
    suite.Check(topology.group == std::vector<uint16_t>({0, 3, 12, 15}) &&
                    topology.entries == expected_entries,
                "tree entries exactly encode production ingress/output bitmap");
    suite.Check(topology == BuildIsaV1XFirstCollectiveTree(
                                shape, 17, 0, {0, 3, 12, 15}),
                "group input ordering cannot change canonical topology");
    const auto topology_resources =
        IsaV1CollectiveTreeOutputResources(topology);
    suite.Check(topology_resources.size() == 12 &&
                    std::binary_search(topology_resources.begin(),
                                       topology_resources.end(), R(0, EAST)) &&
                    std::binary_search(topology_resources.begin(),
                                       topology_resources.end(), R(3, CENTER)) &&
                    std::binary_search(topology_resources.begin(),
                                       topology_resources.end(), R(15, CENTER)),
                "router-output resources are derived from production entries");
    suite.Check(BuildIsaV1XFirstCollectiveTree(shape, 18, 5, {5}).entries.empty(),
                "single-member multicast programs no tree entry");

    suite.Throws<std::invalid_argument>(
        [&] { (void)BuildIsaV1XFirstCollectiveTree(shape, 0, 0, {0, 1}); },
        "tree_id zero is rejected");
    suite.Throws<std::invalid_argument>(
        [&] { (void)BuildIsaV1XFirstCollectiveTree(shape, 1, 0, {1, 2}); },
        "root missing from group is rejected");
    suite.Throws<std::invalid_argument>(
        [&] { (void)BuildIsaV1XFirstCollectiveTree(shape, 1, 0, {0, 1, 1}); },
        "duplicate group member is rejected");
    suite.Throws<std::invalid_argument>(
        [&] {
            (void)BuildIsaV1XFirstCollectiveTree({4, 4, 2}, 1, 0,
                                                 {0, 16});
        },
        "cross-die collective topology is rejected");
    suite.Throws<std::invalid_argument>(
        [&] { (void)BuildIsaV1XFirstPath(shape, 0, 16); },
        "out-of-shape path endpoint is rejected");

    const auto no_conflict = std::vector<IsaV1TreeScheduleInput>{
        Tree(2, {R(4, NORTH)}, {4}), Tree(1, {R(0, EAST)}, {0})};
    const auto parallel = ScheduleIsaV1CollectiveTrees(no_conflict);
    suite.Check(parallel.batches.size() == 1 &&
                    parallel.batches[0].tree_ids ==
                        std::vector<uint16_t>({1, 2}),
                "disjoint trees share a canonical batch");

    const auto edge_conflict = std::vector<IsaV1TreeScheduleInput>{
        Tree(1, {R(0, EAST)}, {0}), Tree(2, {R(0, EAST)}, {0})};
    const auto separated = ScheduleIsaV1CollectiveTrees(edge_conflict);
    suite.Check(separated.conflict_graph.edges ==
                    std::vector<std::pair<uint16_t, uint16_t>>{{1, 2}} &&
                    separated.batches.size() == 2,
                "shared Router output creates an edge and separate batches");

    const auto point_only = std::vector<IsaV1TreeScheduleInput>{
        Tree(1, {R(0, EAST)}, {0}), Tree(2, {R(0, NORTH)}, {0})};
    suite.Check(ScheduleIsaV1CollectiveTrees(point_only).batches.size() == 1,
                "same Router with different outputs has no link conflict");
    IsaV1TreeScheduleOptions point_capacity;
    point_capacity.entries_per_router = 1;
    const auto capacity_split =
        ScheduleIsaV1CollectiveTrees(point_only, point_capacity);
    suite.Check(capacity_split.batches.size() == 2 &&
                    capacity_split.peak_entries_by_router.at(0) == 1,
                "Router point capacity splits otherwise independent outputs");

    const auto chain = GraphTrees(3, {{0, 1}, {1, 2}});
    const auto chain_schedule = ScheduleIsaV1CollectiveTrees(chain);
    suite.Check(chain_schedule.batches.size() == 2 &&
                    BatchOf(chain_schedule, 1) == BatchOf(chain_schedule, 3) &&
                    BatchOf(chain_schedule, 1) != BatchOf(chain_schedule, 2),
                "chain conflict graph uses two legal colors");
    const auto cycle = GraphTrees(5, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}});
    const auto cycle_schedule = ScheduleIsaV1CollectiveTrees(cycle);
    suite.Check(cycle_schedule.batches.size() == 3,
                "odd cycle deterministic greedy coloring uses three colors");
    const auto complete = GraphTrees(
        4, {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}});
    suite.Check(ScheduleIsaV1CollectiveTrees(complete).batches.size() == 4,
                "complete conflict graph is fully serialized");
    CheckBatchesSafe(suite, chain, chain_schedule,
                     "every programmed chain batch has disjoint resources");
    CheckBatchesSafe(suite, cycle, cycle_schedule,
                     "every programmed cycle batch has disjoint resources");

    auto shuffled = cycle;
    std::reverse(shuffled.begin(), shuffled.end());
    for (auto &tree : shuffled) {
        std::reverse(tree.resources.begin(), tree.resources.end());
        std::reverse(tree.entry_routers.begin(), tree.entry_routers.end());
    }
    const auto shuffled_schedule = ScheduleIsaV1CollectiveTrees(shuffled);
    suite.Check(shuffled_schedule.conflict_graph == cycle_schedule.conflict_graph &&
                    shuffled_schedule.batches == cycle_schedule.batches,
                "tree/resource input permutations preserve schedule exactly");

    IsaV1TreeScheduleOptions k1;
    k1.max_trees_per_batch = 1;
    IsaV1TreeScheduleOptions k2;
    k2.max_trees_per_batch = 2;
    IsaV1TreeScheduleOptions k_many;
    k_many.max_trees_per_batch = 99;
    suite.Check(ScheduleIsaV1CollectiveTrees(no_conflict, k1).batches.size() == 2 &&
                    ScheduleIsaV1CollectiveTrees(no_conflict, k2).batches.size() == 1 &&
                    ScheduleIsaV1CollectiveTrees(no_conflict, k_many).batches.size() == 1,
                "K=1, K=2 and K>=N are hard deterministic upper bounds");

    IsaV1TreeScheduleOptions boundary;
    boundary.occupied_entries_by_router[0] = 63;
    const auto at_64 = ScheduleIsaV1CollectiveTrees(
        {Tree(7, {R(0, EAST)}, {0})}, boundary);
    suite.Check(at_64.peak_entries_by_router.at(0) == 64,
                "63 resident plus one batch tree reaches exact 64-entry boundary");
    boundary.occupied_entries_by_router[0] = 64;
    suite.Throws<std::runtime_error>(
        [&] {
            (void)ScheduleIsaV1CollectiveTrees(
                {Tree(7, {R(0, EAST)}, {0})}, boundary);
        },
        "64 resident entries reject another tree without overprovision");
    boundary.occupied_entries_by_router[0] = 65;
    suite.Throws<std::invalid_argument>(
        [&] { (void)ScheduleIsaV1CollectiveTrees({}, boundary); },
        "pre-existing occupancy above 64 is rejected");

    const auto unknown = std::vector<IsaV1TreeScheduleInput>{
        Tree(3, {}, {3}, false), Tree(1, {R(1, EAST)}, {1}),
        Tree(2, {R(2, NORTH)}, {2})};
    const auto conservative = ScheduleIsaV1CollectiveTrees(unknown);
    suite.Check(conservative.serial_fallback &&
                    conservative.conflict_graph.edges ==
                        std::vector<std::pair<uint16_t, uint16_t>>{{1, 3}, {2, 3}} &&
                    conservative.batches.size() == 2 &&
                    BatchOf(conservative, 3) != BatchOf(conservative, 1) &&
                    BatchOf(conservative, 3) != BatchOf(conservative, 2),
                "unproven path conservatively conflicts with every other tree");
    IsaV1TreeScheduleOptions force_serial;
    force_serial.force_serial = true;
    suite.Check(ScheduleIsaV1CollectiveTrees(no_conflict, force_serial)
                        .batches.size() == 2,
                "explicit serial fallback overrides proven independence");

    suite.Throws<std::invalid_argument>(
        [&] {
            auto duplicate = no_conflict;
            duplicate[1].tree_id = duplicate[0].tree_id;
            (void)ScheduleIsaV1CollectiveTrees(duplicate);
        },
        "duplicate tree ids are rejected before scheduling");
    suite.Throws<std::invalid_argument>(
        [&] {
            (void)ScheduleIsaV1CollectiveTrees(
                {Tree(1, {R(0, EAST)}, {})});
        },
        "resource without table entry demand is rejected");
    suite.Throws<std::invalid_argument>(
        [&] {
            (void)ScheduleIsaV1CollectiveTrees(
                {Tree(1, {}, {}, false)});
        },
        "unproven path without capacity demand is rejected");
    IsaV1TreeScheduleOptions bad_k;
    bad_k.max_trees_per_batch = 0;
    suite.Throws<std::invalid_argument>(
        [&] { (void)ScheduleIsaV1CollectiveTrees(no_conflict, bad_k); },
        "K=0 is rejected");

    const auto production_input = IsaV1TreeScheduleInputFromTopology(topology);
    suite.Check(production_input.tree_id == topology.tree_id &&
                    production_input.resources == topology_resources &&
                    production_input.entry_routers.size() ==
                        topology.entries.size(),
                "scheduler input is losslessly derived from production image");

    if (suite.failures == 0)
        std::cout << "[ISA V1 COLL TOPOLOGY] PASS: " << suite.checks
                  << " checks\n";
    return suite.failures;
}

#ifdef ISA_V1_COLL_TOPOLOGY_SELFTEST_MAIN
int main() { return RunIsaV1CollectiveTopologySelfTest(); }
#endif
