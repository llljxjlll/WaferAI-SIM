#pragma once

#include "defs/enums.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

// The P7 topology and scheduler are deliberately pure: callers supply the
// mesh shape and current table occupancy, and the result contains no runtime
// registry or SystemC state.  The production helper can therefore program the
// exact entries returned here, while static conflict analysis consumes the
// output resources derived from those same entries.

inline constexpr uint16_t kIsaV1CollectiveTreeEntriesPerRouter = 64;

struct IsaV1MeshShape {
    uint16_t grid_x = 0;
    uint16_t grid_y = 0;
    uint16_t die_count = 0;
};

struct IsaV1RouterOutput {
    uint16_t router_id = 0;
    Directions output = CENTER;

    bool operator==(const IsaV1RouterOutput &other) const;
    bool operator<(const IsaV1RouterOutput &other) const;
};

struct IsaV1CollectiveTreeEntry {
    uint16_t router_id = 0;
    Directions ingress = CENTER;
    uint8_t output_mask = 0;

    bool operator==(const IsaV1CollectiveTreeEntry &other) const;
};

struct IsaV1CollectiveTreeTopology {
    uint16_t tree_id = 0;
    uint16_t root = 0;
    std::vector<uint16_t> group;
    std::vector<IsaV1CollectiveTreeEntry> entries;

    bool operator==(const IsaV1CollectiveTreeTopology &other) const;
};

// Returns the production X-first XY route, including the destination CENTER
// output.  src==dst therefore has the one-resource path (src,CENTER).
// Cross-die or out-of-shape endpoints are rejected.
std::vector<IsaV1RouterOutput> BuildIsaV1XFirstPath(
    const IsaV1MeshShape &shape, uint16_t src, uint16_t dst);

// Builds the canonical multicast programming image for one same-die group.
// A one-member multicast has no entries, matching the production no-op.
// tree_id zero, duplicate/missing root members and cross-die groups fail fast.
IsaV1CollectiveTreeTopology BuildIsaV1XFirstCollectiveTree(
    const IsaV1MeshShape &shape, uint16_t tree_id, uint16_t root,
    const std::vector<uint16_t> &group);

// Expands every bit in entries.output_mask.  This is the only resource set
// used by the conflict graph, so static analysis cannot diverge from the
// entries that production programs.
std::vector<IsaV1RouterOutput> IsaV1CollectiveTreeOutputResources(
    const IsaV1CollectiveTreeTopology &tree);

struct IsaV1TreeScheduleInput {
    uint16_t tree_id = 0;
    // false means output conflicts cannot be proven; the scheduler conflicts
    // this tree with every other tree and records a serial fallback.
    bool path_proven = true;
    std::vector<IsaV1RouterOutput> resources;
    // One item is one table entry.  X-first multicast trees contain at most
    // one entry per router, but the scheduler validates this explicitly.
    std::vector<uint16_t> entry_routers;
};

IsaV1TreeScheduleInput IsaV1TreeScheduleInputFromTopology(
    const IsaV1CollectiveTreeTopology &tree);

struct IsaV1TreeConflictGraph {
    std::vector<uint16_t> tree_ids;
    // Canonical (smaller tree_id, larger tree_id) edges in sorted order.
    std::vector<std::pair<uint16_t, uint16_t>> edges;

    bool operator==(const IsaV1TreeConflictGraph &other) const;
};

IsaV1TreeConflictGraph BuildIsaV1TreeConflictGraph(
    const std::vector<IsaV1TreeScheduleInput> &trees);

struct IsaV1TreeScheduleOptions {
    // K is a hard upper bound, not a target.  Zero is invalid.
    size_t max_trees_per_batch = static_cast<size_t>(-1);
    uint16_t entries_per_router =
        kIsaV1CollectiveTreeEntriesPerRouter;
    // Entries held by unrelated live trees.  They remain resident throughout
    // every scheduled batch.
    std::map<uint16_t, uint16_t> occupied_entries_by_router;
    bool force_serial = false;
};

struct IsaV1TreeBatch {
    uint16_t batch_index = 0;
    std::vector<uint16_t> tree_ids;
    std::map<uint16_t, uint16_t> entry_demand_by_router;

    bool operator==(const IsaV1TreeBatch &other) const;
};

struct IsaV1TreeSchedule {
    IsaV1TreeConflictGraph conflict_graph;
    std::vector<IsaV1TreeBatch> batches;
    // Includes unrelated occupied entries.  Since batches are program/erase
    // scoped, this is max(current + one-batch demand), never a cumulative sum.
    std::map<uint16_t, uint16_t> peak_entries_by_router;
    bool serial_fallback = false;
};

// Deterministic earliest-fit greedy coloring in ascending tree_id order,
// constrained by resource edges, K and per-router remaining capacity.
// A tree that cannot fit even in an empty scheduled batch is rejected.
IsaV1TreeSchedule ScheduleIsaV1CollectiveTrees(
    const std::vector<IsaV1TreeScheduleInput> &trees,
    const IsaV1TreeScheduleOptions &options = IsaV1TreeScheduleOptions{});
