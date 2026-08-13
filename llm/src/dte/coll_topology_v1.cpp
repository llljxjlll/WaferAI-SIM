#include "dte/coll_topology_v1.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>

namespace {

uint32_t CoresPerDie(const IsaV1MeshShape &shape) {
    if (shape.grid_x == 0 || shape.grid_y == 0 || shape.die_count == 0)
        throw std::invalid_argument("ISA-v1 mesh dimensions must be positive");
    const uint32_t per_die =
        static_cast<uint32_t>(shape.grid_x) * shape.grid_y;
    const uint64_t total = static_cast<uint64_t>(per_die) * shape.die_count;
    if (total > static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) +
                    1ULL)
        throw std::overflow_error("ISA-v1 mesh exceeds u16 router id space");
    return per_die;
}

uint32_t ValidateCore(const IsaV1MeshShape &shape, uint16_t core) {
    const uint32_t per_die = CoresPerDie(shape);
    const uint32_t total = per_die * shape.die_count;
    if (core >= total)
        throw std::invalid_argument("ISA-v1 collective core is outside mesh");
    return per_die;
}

uint8_t OutputBit(Directions direction) {
    if (direction < WEST || direction >= DIRECTIONS)
        throw std::invalid_argument("ISA-v1 tree direction is invalid");
    return static_cast<uint8_t>(1U << static_cast<unsigned>(direction));
}

Directions Opposite(Directions direction) {
    switch (direction) {
    case WEST: return EAST;
    case EAST: return WEST;
    case NORTH: return SOUTH;
    case SOUTH: return NORTH;
    case CENTER:
    case DIRECTIONS:
    case HOST:
        break;
    }
    throw std::invalid_argument("ISA-v1 tree link has no opposite direction");
}

Directions StepDirection(const IsaV1MeshShape &shape, uint16_t from,
                         uint16_t to) {
    if (to + 1U == from) return WEST;
    if (to == from + 1U) return EAST;
    if (static_cast<uint32_t>(to) + shape.grid_x == from) return SOUTH;
    if (to == static_cast<uint32_t>(from) + shape.grid_x) return NORTH;
    throw std::invalid_argument(
        "ISA-v1 collective tree contains a non-neighbor edge");
}

std::vector<IsaV1TreeScheduleInput> NormalizeTrees(
    const std::vector<IsaV1TreeScheduleInput> &trees) {
    std::vector<IsaV1TreeScheduleInput> normalized = trees;
    for (auto &tree : normalized) {
        if (tree.tree_id == 0)
            throw std::invalid_argument("ISA-v1 scheduled tree_id zero is reserved");
        for (const auto &resource : tree.resources)
            (void)OutputBit(resource.output);
        std::sort(tree.resources.begin(), tree.resources.end());
        if (std::adjacent_find(tree.resources.begin(), tree.resources.end()) !=
            tree.resources.end())
            throw std::invalid_argument(
                "ISA-v1 tree has duplicate output resources");
        std::sort(tree.entry_routers.begin(), tree.entry_routers.end());
        if (std::adjacent_find(tree.entry_routers.begin(),
                               tree.entry_routers.end()) !=
            tree.entry_routers.end())
            throw std::invalid_argument(
                "ISA-v1 tree has duplicate entries on one router");
        for (const auto &resource : tree.resources)
            if (!std::binary_search(tree.entry_routers.begin(),
                                    tree.entry_routers.end(),
                                    resource.router_id))
                throw std::invalid_argument(
                    "ISA-v1 tree resource has no matching table entry");
        if (!tree.path_proven && tree.entry_routers.empty())
            throw std::invalid_argument(
                "ISA-v1 unproven tree requires explicit entry demand");
    }
    std::sort(normalized.begin(), normalized.end(),
              [](const auto &a, const auto &b) {
                  return a.tree_id < b.tree_id;
              });
    for (size_t i = 1; i < normalized.size(); ++i)
        if (normalized[i - 1].tree_id == normalized[i].tree_id)
            throw std::invalid_argument("ISA-v1 scheduled tree_id collision");
    return normalized;
}

bool SharesResource(const IsaV1TreeScheduleInput &a,
                    const IsaV1TreeScheduleInput &b) {
    if (!a.path_proven || !b.path_proven) return true;
    size_t i = 0;
    size_t j = 0;
    while (i < a.resources.size() && j < b.resources.size()) {
        if (a.resources[i] == b.resources[j]) return true;
        if (a.resources[i] < b.resources[j])
            ++i;
        else
            ++j;
    }
    return false;
}

bool ContainsEdge(const IsaV1TreeConflictGraph &graph, uint16_t a,
                  uint16_t b) {
    if (b < a) std::swap(a, b);
    return std::binary_search(graph.edges.begin(), graph.edges.end(),
                              std::make_pair(a, b));
}

std::map<uint16_t, uint16_t> EntryDemand(
    const IsaV1TreeScheduleInput &tree) {
    std::map<uint16_t, uint16_t> result;
    for (uint16_t router : tree.entry_routers) result.emplace(router, 1);
    return result;
}

bool FitsCapacity(const IsaV1TreeBatch &batch,
                  const IsaV1TreeScheduleInput &tree,
                  const IsaV1TreeScheduleOptions &options) {
    for (uint16_t router : tree.entry_routers) {
        uint32_t demand = 1;
        auto batch_it = batch.entry_demand_by_router.find(router);
        if (batch_it != batch.entry_demand_by_router.end())
            demand += batch_it->second;
        auto occupied_it = options.occupied_entries_by_router.find(router);
        if (occupied_it != options.occupied_entries_by_router.end())
            demand += occupied_it->second;
        if (demand > options.entries_per_router) return false;
    }
    return true;
}

} // namespace

bool IsaV1RouterOutput::operator==(const IsaV1RouterOutput &other) const {
    return std::tie(router_id, output) ==
           std::tie(other.router_id, other.output);
}

bool IsaV1RouterOutput::operator<(const IsaV1RouterOutput &other) const {
    return std::tie(router_id, output) <
           std::tie(other.router_id, other.output);
}

bool IsaV1CollectiveTreeEntry::operator==(
    const IsaV1CollectiveTreeEntry &other) const {
    return std::tie(router_id, ingress, output_mask) ==
           std::tie(other.router_id, other.ingress, other.output_mask);
}

bool IsaV1CollectiveTreeTopology::operator==(
    const IsaV1CollectiveTreeTopology &other) const {
    return std::tie(tree_id, root, group, entries) ==
           std::tie(other.tree_id, other.root, other.group, other.entries);
}

std::vector<IsaV1RouterOutput> BuildIsaV1XFirstPath(
    const IsaV1MeshShape &shape, uint16_t src, uint16_t dst) {
    const uint32_t per_die = ValidateCore(shape, src);
    ValidateCore(shape, dst);
    if (src / per_die != dst / per_die)
        throw std::invalid_argument(
            "ISA-v1 X-first collective path cannot cross dies");

    std::vector<IsaV1RouterOutput> path;
    uint16_t current = src;
    const uint32_t die_base = (src / per_die) * per_die;
    const uint32_t dst_local = dst - die_base;
    const uint32_t dst_x = dst_local % shape.grid_x;
    const uint32_t dst_y = dst_local / shape.grid_x;
    while ((current - die_base) % shape.grid_x != dst_x) {
        const uint32_t current_x = (current - die_base) % shape.grid_x;
        const Directions direction = current_x < dst_x ? EAST : WEST;
        path.push_back({current, direction});
        current = static_cast<uint16_t>(
            direction == EAST ? current + 1U : current - 1U);
    }
    while ((current - die_base) / shape.grid_x != dst_y) {
        const uint32_t current_y = (current - die_base) / shape.grid_x;
        const Directions direction = current_y < dst_y ? NORTH : SOUTH;
        path.push_back({current, direction});
        current = static_cast<uint16_t>(
            direction == NORTH ? current + shape.grid_x
                               : current - shape.grid_x);
    }
    path.push_back({dst, CENTER});
    return path;
}

IsaV1CollectiveTreeTopology BuildIsaV1XFirstCollectiveTree(
    const IsaV1MeshShape &shape, uint16_t tree_id, uint16_t root,
    const std::vector<uint16_t> &group) {
    if (tree_id == 0)
        throw std::invalid_argument("ISA-v1 collective tree_id zero is reserved");
    const uint32_t per_die = ValidateCore(shape, root);
    if (group.empty())
        throw std::invalid_argument("ISA-v1 collective tree group is empty");

    IsaV1CollectiveTreeTopology tree;
    tree.tree_id = tree_id;
    tree.root = root;
    tree.group = group;
    std::sort(tree.group.begin(), tree.group.end());
    if (std::adjacent_find(tree.group.begin(), tree.group.end()) !=
        tree.group.end())
        throw std::invalid_argument(
            "ISA-v1 collective tree group contains duplicate cores");
    if (!std::binary_search(tree.group.begin(), tree.group.end(), root))
        throw std::invalid_argument(
            "ISA-v1 collective tree root is not in group");
    for (uint16_t core : tree.group) {
        ValidateCore(shape, core);
        if (core / per_die != root / per_die)
            throw std::invalid_argument(
                "ISA-v1 collective tree cannot cross dies");
    }
    if (tree.group.size() == 1) return tree;

    std::map<uint16_t, uint16_t> parent;
    std::map<uint16_t, uint8_t> outputs;
    outputs[root] = 0;
    for (uint16_t target : tree.group) {
        if (target == root) continue;
        const auto path = BuildIsaV1XFirstPath(shape, root, target);
        uint16_t current = root;
        for (const auto &resource : path) {
            if (resource.output == CENTER) {
                outputs[resource.router_id] |= OutputBit(CENTER);
                break;
            }
            const uint16_t next = static_cast<uint16_t>(
                resource.output == WEST
                    ? current - 1U
                    : resource.output == EAST
                          ? current + 1U
                          : resource.output == NORTH
                                ? current + shape.grid_x
                                : current - shape.grid_x);
            const auto inserted = parent.emplace(next, current);
            if (!inserted.second && inserted.first->second != current)
                throw std::runtime_error(
                    "ISA-v1 X-first collective tree has multiple parents");
            outputs[current] |= OutputBit(resource.output);
            current = next;
        }
    }

    tree.entries.reserve(outputs.size());
    for (const auto &node : outputs) {
        Directions ingress = CENTER;
        if (node.first != root) {
            auto parent_it = parent.find(node.first);
            if (parent_it == parent.end())
                throw std::logic_error(
                    "ISA-v1 collective tree node has no parent");
            ingress = Opposite(
                StepDirection(shape, parent_it->second, node.first));
        }
        if (node.second == 0)
            throw std::logic_error(
                "ISA-v1 collective tree entry has no outputs");
        if ((node.second & OutputBit(ingress)) != 0)
            throw std::logic_error(
                "ISA-v1 collective tree reflects to its ingress");
        tree.entries.push_back({node.first, ingress, node.second});
    }
    return tree;
}

std::vector<IsaV1RouterOutput> IsaV1CollectiveTreeOutputResources(
    const IsaV1CollectiveTreeTopology &tree) {
    std::vector<IsaV1RouterOutput> resources;
    for (const auto &entry : tree.entries) {
        if (entry.output_mask == 0 ||
            (entry.output_mask & ~static_cast<uint8_t>((1U << DIRECTIONS) - 1U)) !=
                0)
            throw std::invalid_argument(
                "ISA-v1 collective tree entry has invalid outputs");
        for (int direction = WEST; direction < DIRECTIONS; ++direction)
            if ((entry.output_mask & (1U << direction)) != 0)
                resources.push_back(
                    {entry.router_id, static_cast<Directions>(direction)});
    }
    std::sort(resources.begin(), resources.end());
    if (std::adjacent_find(resources.begin(), resources.end()) !=
        resources.end())
        throw std::invalid_argument(
            "ISA-v1 collective tree duplicates an output resource");
    return resources;
}

IsaV1TreeScheduleInput IsaV1TreeScheduleInputFromTopology(
    const IsaV1CollectiveTreeTopology &tree) {
    if (tree.tree_id == 0)
        throw std::invalid_argument("ISA-v1 topology tree_id zero is reserved");
    IsaV1TreeScheduleInput input;
    input.tree_id = tree.tree_id;
    input.resources = IsaV1CollectiveTreeOutputResources(tree);
    input.entry_routers.reserve(tree.entries.size());
    for (const auto &entry : tree.entries)
        input.entry_routers.push_back(entry.router_id);
    std::sort(input.entry_routers.begin(), input.entry_routers.end());
    if (std::adjacent_find(input.entry_routers.begin(),
                           input.entry_routers.end()) !=
        input.entry_routers.end())
        throw std::invalid_argument(
            "ISA-v1 topology has multiple entries on one router");
    return input;
}

bool IsaV1TreeConflictGraph::operator==(
    const IsaV1TreeConflictGraph &other) const {
    return std::tie(tree_ids, edges) == std::tie(other.tree_ids, other.edges);
}

IsaV1TreeConflictGraph BuildIsaV1TreeConflictGraph(
    const std::vector<IsaV1TreeScheduleInput> &trees) {
    const auto normalized = NormalizeTrees(trees);
    IsaV1TreeConflictGraph graph;
    graph.tree_ids.reserve(normalized.size());
    for (const auto &tree : normalized) graph.tree_ids.push_back(tree.tree_id);
    for (size_t i = 0; i < normalized.size(); ++i)
        for (size_t j = i + 1; j < normalized.size(); ++j)
            if (SharesResource(normalized[i], normalized[j]))
                graph.edges.emplace_back(normalized[i].tree_id,
                                         normalized[j].tree_id);
    return graph;
}

bool IsaV1TreeBatch::operator==(const IsaV1TreeBatch &other) const {
    return std::tie(batch_index, tree_ids, entry_demand_by_router) ==
           std::tie(other.batch_index, other.tree_ids,
                    other.entry_demand_by_router);
}

IsaV1TreeSchedule ScheduleIsaV1CollectiveTrees(
    const std::vector<IsaV1TreeScheduleInput> &trees,
    const IsaV1TreeScheduleOptions &options) {
    if (options.max_trees_per_batch == 0)
        throw std::invalid_argument("ISA-v1 tree batch K must be positive");
    if (options.entries_per_router == 0)
        throw std::invalid_argument(
            "ISA-v1 collective tree table capacity must be positive");
    for (const auto &occupied : options.occupied_entries_by_router)
        if (occupied.second > options.entries_per_router)
            throw std::invalid_argument(
                "ISA-v1 existing tree occupancy exceeds capacity");

    const auto normalized = NormalizeTrees(trees);
    IsaV1TreeSchedule result;
    result.conflict_graph = BuildIsaV1TreeConflictGraph(normalized);
    result.serial_fallback = options.force_serial ||
        std::any_of(normalized.begin(), normalized.end(),
                    [](const auto &tree) { return !tree.path_proven; });

    for (const auto &tree : normalized) {
        IsaV1TreeBatch empty;
        if (!FitsCapacity(empty, tree, options))
            throw std::runtime_error(
                "ISA-v1 tree cannot fit remaining per-router capacity");

        size_t selected = result.batches.size();
        if (!options.force_serial) {
            for (size_t b = 0; b < result.batches.size(); ++b) {
                const auto &batch = result.batches[b];
                if (batch.tree_ids.size() >= options.max_trees_per_batch)
                    continue;
                bool conflict = false;
                for (uint16_t other : batch.tree_ids)
                    if (ContainsEdge(result.conflict_graph, tree.tree_id,
                                     other)) {
                        conflict = true;
                        break;
                    }
                if (!conflict && FitsCapacity(batch, tree, options)) {
                    selected = b;
                    break;
                }
            }
        }
        if (selected == result.batches.size()) {
            if (result.batches.size() >=
                static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1U)
                throw std::overflow_error(
                    "ISA-v1 tree batch index exceeds u16");
            IsaV1TreeBatch batch;
            batch.batch_index = static_cast<uint16_t>(result.batches.size());
            result.batches.push_back(std::move(batch));
        }
        auto &batch = result.batches[selected];
        batch.tree_ids.push_back(tree.tree_id);
        const auto demand = EntryDemand(tree);
        for (const auto &entry : demand)
            ++batch.entry_demand_by_router[entry.first];
    }

    result.peak_entries_by_router = options.occupied_entries_by_router;
    for (const auto &batch : result.batches) {
        if (batch.tree_ids.empty() ||
            batch.tree_ids.size() > options.max_trees_per_batch)
            throw std::logic_error("ISA-v1 tree scheduler produced invalid K");
        for (size_t i = 0; i < batch.tree_ids.size(); ++i)
            for (size_t j = i + 1; j < batch.tree_ids.size(); ++j)
                if (ContainsEdge(result.conflict_graph, batch.tree_ids[i],
                                 batch.tree_ids[j]))
                    throw std::logic_error(
                        "ISA-v1 tree scheduler produced a conflicting batch");
        for (const auto &demand : batch.entry_demand_by_router) {
            uint32_t occupancy = demand.second;
            auto occupied = options.occupied_entries_by_router.find(demand.first);
            if (occupied != options.occupied_entries_by_router.end())
                occupancy += occupied->second;
            if (occupancy > options.entries_per_router)
                throw std::logic_error(
                    "ISA-v1 tree scheduler silently exceeded capacity");
            auto &peak = result.peak_entries_by_router[demand.first];
            peak = std::max<uint16_t>(peak,
                static_cast<uint16_t>(occupancy));
        }
    }
    return result;
}
