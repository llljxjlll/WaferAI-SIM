#include "dte/coll_innetwork_reduce.h"

#include <map>
#include <mutex>

namespace {
std::mutex g_reduce_tree_mutex;
std::map<std::pair<uint16_t, uint16_t>, CollReduceTreeNode> g_reduce_nodes;
}

void ResetCollectiveReduceFabric() {
    std::lock_guard<std::mutex> lock(g_reduce_tree_mutex);
    g_reduce_nodes.clear();
}

void ProgramCollectiveReduceNode(uint16_t tree_id, uint16_t router_id,
                                 const CollReduceTreeNode &node) {
    if (tree_id == 0 || node.expected_inputs == 0 ||
        node.parent_output < WEST || node.parent_output > CENTER)
        throw std::invalid_argument("invalid production reduce-tree node");
    std::lock_guard<std::mutex> lock(g_reduce_tree_mutex);
    const auto key = std::make_pair(tree_id, router_id);
    auto it = g_reduce_nodes.find(key);
    if (it != g_reduce_nodes.end() &&
        (it->second.expected_inputs != node.expected_inputs ||
         it->second.parent_output != node.parent_output))
        throw std::runtime_error("conflicting production reduce-tree node");
    if (it == g_reduce_nodes.end()) {
        size_t per_router = 0;
        for (const auto &entry : g_reduce_nodes)
            if (entry.first.second == router_id) ++per_router;
        if (per_router >= COLL_REDUCE_NODES_PER_ROUTER)
            throw std::runtime_error(
                "production reduce-tree table capacity exhausted");
    }
    g_reduce_nodes[key] = node;
}

size_t EraseCollectiveReduceTree(uint16_t tree_id) {
    std::lock_guard<std::mutex> lock(g_reduce_tree_mutex);
    size_t erased = 0;
    for (auto it = g_reduce_nodes.begin(); it != g_reduce_nodes.end();) {
        if (it->first.first == tree_id) {
            it = g_reduce_nodes.erase(it);
            ++erased;
        } else {
            ++it;
        }
    }
    return erased;
}

size_t CollectiveReduceNodeCount() {
    std::lock_guard<std::mutex> lock(g_reduce_tree_mutex);
    return g_reduce_nodes.size();
}

CollReduceTreeNode LookupCollectiveReduceNode(uint16_t tree_id,
                                              uint16_t router_id) {
    std::lock_guard<std::mutex> lock(g_reduce_tree_mutex);
    auto it = g_reduce_nodes.find({tree_id, router_id});
    if (it == g_reduce_nodes.end())
        throw std::runtime_error("production reduce-tree node missing");
    return it->second;
}
