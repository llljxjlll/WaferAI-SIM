#include "dte/coll_multicast.h"
#include "defs/spec.h"

#include <map>
#include <mutex>
#include <set>

namespace {
std::mutex g_tree_mutex;
std::map<CollectiveTreeKey, uint8_t> g_tree_entries;
using LinkKey = std::tuple<uint16_t, uint16_t, uint8_t>;
std::map<LinkKey, std::pair<uint64_t, uint64_t>> g_link_stats;
std::map<std::pair<uint16_t, uint8_t>, std::pair<uint64_t, uint64_t>>
    g_shared_link_stats;
}

void ResetCollectiveFabric() {
    std::lock_guard<std::mutex> lock(g_tree_mutex);
    g_tree_entries.clear();
    g_link_stats.clear();
    g_shared_link_stats.clear();
}

void ProgramCollectiveTreeEntry(const CollectiveTreeKey &key, uint8_t outputs) {
    CollectiveTreeTable validation(1);
    validation.Program(key, outputs);
    std::lock_guard<std::mutex> lock(g_tree_mutex);
    auto it = g_tree_entries.find(key);
    if (it != g_tree_entries.end() && it->second != outputs)
        throw std::runtime_error("conflicting production collective tree entry");
    if (it == g_tree_entries.end()) {
        size_t per_router = 0;
        for (const auto &entry : g_tree_entries)
            if (entry.first.router_id == key.router_id) ++per_router;
        if (per_router >= COLL_TREE_ENTRIES_PER_ROUTER)
            throw std::runtime_error(
                "production collective tree table capacity exhausted");
    }
    g_tree_entries[key] = outputs;
}

size_t EraseCollectiveTree(uint16_t tree_id) {
    std::lock_guard<std::mutex> lock(g_tree_mutex);
    size_t erased = 0;
    for (auto it = g_tree_entries.begin(); it != g_tree_entries.end();) {
        if (it->first.tree_id == tree_id) {
            it = g_tree_entries.erase(it);
            ++erased;
        } else {
            ++it;
        }
    }
    return erased;
}

uint8_t LookupCollectiveTreeEntry(const CollectiveTreeKey &key) {
    std::lock_guard<std::mutex> lock(g_tree_mutex);
    auto it = g_tree_entries.find(key);
    if (it == g_tree_entries.end())
        throw std::runtime_error("production collective tree entry missing");
    return it->second;
}

size_t CollectiveTreeEntryCount() {
    std::lock_guard<std::mutex> lock(g_tree_mutex);
    return g_tree_entries.size();
}

void ValidateCollectiveTree(uint16_t tree_id, uint16_t root,
                            const std::vector<uint16_t> &targets) {
    std::lock_guard<std::mutex> lock(g_tree_mutex);
    std::map<uint16_t, std::vector<uint16_t>> edges;
    std::set<uint16_t> delivered;
    for (const auto &entry : g_tree_entries) {
        if (entry.first.tree_id != tree_id) continue;
        const uint16_t router = entry.first.router_id;
        for (int d = 0; d < DIRECTIONS; ++d) {
            if (!(entry.second & (1u << d))) continue;
            if (d == CENTER) { delivered.insert(router); continue; }
            int next = router;
            if (d == WEST) --next;
            else if (d == EAST) ++next;
            else if (d == NORTH) next += GRID_X;
            else if (d == SOUTH) next -= GRID_X;
            if (next < 0 || next >= TOTAL_CORES)
                throw std::runtime_error("collective tree edge leaves mesh");
            edges[router].push_back(static_cast<uint16_t>(next));
        }
    }
    std::set<uint16_t> visiting, visited;
    std::function<void(uint16_t)> dfs = [&](uint16_t node) {
        if (visiting.count(node)) throw std::runtime_error("collective tree contains a cycle");
        if (visited.count(node)) return;
        visiting.insert(node);
        for (uint16_t child : edges[node]) dfs(child);
        visiting.erase(node); visited.insert(node);
    };
    dfs(root);
    for (uint16_t target : targets)
        if (target != root && (!visited.count(target) || !delivered.count(target)))
            throw std::runtime_error("collective tree does not cover target group");
}

void RecordCollectiveForkAttempt(uint16_t tree_id, uint16_t router_id,
                                 uint8_t outputs, bool committed) {
    std::lock_guard<std::mutex> lock(g_tree_mutex);
    for (uint8_t d = 0; d < DIRECTIONS; ++d) {
        if (!(outputs & (1u << d))) continue;
        auto &stat = g_link_stats[{tree_id, router_id, d}];
        if (committed) ++stat.first;
        else ++stat.second;
    }
}

std::vector<CollFabricLinkStat> CollectiveFabricLinkStats() {
    std::lock_guard<std::mutex> lock(g_tree_mutex);
    std::vector<CollFabricLinkStat> out;
    out.reserve(g_link_stats.size());
    for (const auto &entry : g_link_stats) {
        CollFabricLinkStat stat;
        stat.tree_id = std::get<0>(entry.first);
        stat.router_id = std::get<1>(entry.first);
        stat.output = std::get<2>(entry.first);
        stat.committed_flits = entry.second.first;
        stat.stalled_attempts = entry.second.second;
        out.push_back(stat);
    }
    return out;
}

void RecordCollectiveSharedOutput(uint16_t router_id, uint8_t output,
                                  bool collective) {
    std::lock_guard<std::mutex> lock(g_tree_mutex);
    auto &stat = g_shared_link_stats[{router_id, output}];
    if (collective) ++stat.second;
    else ++stat.first;
}

std::vector<CollSharedLinkStat> CollectiveSharedLinkStats() {
    std::lock_guard<std::mutex> lock(g_tree_mutex);
    std::vector<CollSharedLinkStat> out;
    out.reserve(g_shared_link_stats.size());
    for (const auto &entry : g_shared_link_stats) {
        CollSharedLinkStat stat;
        stat.router_id = entry.first.first;
        stat.output = entry.first.second;
        stat.normal_flits = entry.second.first;
        stat.collective_flits = entry.second.second;
        out.push_back(stat);
    }
    return out;
}
