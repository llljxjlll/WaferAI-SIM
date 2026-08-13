#pragma once

#include "dte/coll_config.h"
#include "dte/coll_innetwork_reduce.h"
#include "dte/coll_profile_v1.h"
#include "dte/coll_topology_v1.h"
#include "isa/collective_program_v1.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

// P7 acceleration metadata is a second immutable image layered on the P6
// semantic image.  It never rewrites the P6 plan or silently substitutes a
// backend: an unavailable requested backend rejects image construction.
struct IsaV1CollectiveAcceleratedTree {
    uint32_t plan_index = 0;
    uint16_t root_rank = 0;
    bool multicast = false;
    bool dca_reduce = false;
    IsaV1CollectiveTreeTopology topology;
    std::vector<CollReduceTreeNode> reduce_nodes;

    bool operator==(const IsaV1CollectiveAcceleratedTree &other) const;
};

struct IsaV1CollectiveProfilePlanImage {
    uint32_t plan_index = 0;
    CollectiveKey key;
    IsaV1CollectiveProfileDecision decision;
    // True means the P7 executor must replace, rather than execute, every P6
    // endpoint REDUCE_COMPUTE action belonging to this plan.
    bool suppress_endpoint_reduce_compute = false;
    bool requires_dca_payload_executor = false;
    std::vector<IsaV1CollectiveAcceleratedTree> trees;
    IsaV1TreeSchedule tree_schedule;
    size_t max_trees_per_batch = static_cast<size_t>(-1);

    bool operator==(const IsaV1CollectiveProfilePlanImage &other) const;
};

inline constexpr size_t kIsaV1DefaultMaxProfilePlans = 65536;
inline constexpr size_t kIsaV1DefaultMaxProfileTrees = 65535;
inline constexpr size_t kIsaV1DefaultMaxProfileTreeEntries = 1048576;
inline constexpr size_t kIsaV1DefaultMaxProfileConflictEdges = 1048576;
inline constexpr size_t kIsaV1DefaultMaxProfileBatches = 65535;
inline constexpr size_t kIsaV1DefaultMaxProfileDerivedBytes =
    256ULL * 1024ULL * 1024ULL;

struct IsaV1CollectiveProfileImageConfig {
    IsaV1MeshShape mesh;
    NocCollectiveConfig noc;
    IsaV1CollectiveProfileCapabilities capabilities;
    size_t max_trees_per_batch = static_cast<size_t>(-1);
    uint16_t entries_per_router = kIsaV1CollectiveTreeEntriesPerRouter;
    std::map<uint16_t, uint16_t> occupied_entries_by_router;
    size_t max_profile_plans = kIsaV1DefaultMaxProfilePlans;
    size_t max_total_trees = kIsaV1DefaultMaxProfileTrees;
    size_t max_total_tree_entries =
        kIsaV1DefaultMaxProfileTreeEntries;
    size_t max_total_conflict_edges =
        kIsaV1DefaultMaxProfileConflictEdges;
    size_t max_total_batches = kIsaV1DefaultMaxProfileBatches;
    size_t max_derived_bytes = kIsaV1DefaultMaxProfileDerivedBytes;
};

class IsaV1CollectiveProfileProgramImage final {
public:
    uint64_t BaseGeneration() const noexcept { return base_generation_; }
    uint64_t BaseCookie() const noexcept { return base_cookie_; }
    NocCollProfile Profile() const noexcept { return profile_; }
    const std::vector<IsaV1CollectiveProfilePlanImage> &Plans() const
        noexcept { return plans_; }
    size_t TreeCount() const noexcept { return tree_count_; }
    size_t MulticastTreeCount() const noexcept {
        return multicast_tree_count_;
    }
    size_t DcaTreeCount() const noexcept { return dca_tree_count_; }
    size_t TreeEntryCount() const noexcept { return tree_entry_count_; }
    size_t PotentialConflictEdgeCount() const noexcept {
        return potential_conflict_edge_count_;
    }
    size_t BatchCount() const noexcept { return batch_count_; }
    size_t DerivedBytes() const noexcept { return derived_bytes_; }

    const IsaV1CollectiveProfilePlanImage *FindPlan(
        uint32_t plan_index) const noexcept;

private:
    friend IsaV1CollectiveProfileProgramImage
    BuildIsaV1CollectiveProfileProgramImage(
        const IsaV1CollectiveProgramImage &,
        const IsaV1CollectiveProfileImageConfig &);

    uint64_t base_generation_ = 0;
    uint64_t base_cookie_ = 0;
    NocCollProfile profile_ = NocCollProfile::BASELINE;
    std::vector<IsaV1CollectiveProfilePlanImage> plans_;
    size_t tree_count_ = 0;
    size_t multicast_tree_count_ = 0;
    size_t dca_tree_count_ = 0;
    size_t tree_entry_count_ = 0;
    size_t potential_conflict_edge_count_ = 0;
    size_t batch_count_ = 0;
    size_t derived_bytes_ = 0;
};

// The config backends must exactly match its named four-way profile.  Legacy
// Router ALU, timing-only/FP DCA and unvalidated ReduceScatter+DCA are rejected
// before any production registry mutation.
IsaV1CollectiveProfileProgramImage
BuildIsaV1CollectiveProfileProgramImage(
    const IsaV1CollectiveProgramImage &base,
    const IsaV1CollectiveProfileImageConfig &config);
