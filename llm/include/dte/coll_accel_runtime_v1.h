#pragma once

#include "dte/coll_program_profile_v1.h"
#include "dte/coll_tree_registry_bridge_v1.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>

// Shared production coordinator for the P7 accelerated backends.  It is the
// only owner of batch transitions and 16-bit wire sessions.  A tree is never
// erased until every required multicast target has acknowledged a CRC-valid
// SRAM commit and/or its DCA root has acknowledged the real result write.
struct IsaV1CollectiveAccelerationRuntimeStats {
    uint64_t batches_started = 0;
    uint64_t batches_completed = 0;
    uint64_t sessions_allocated = 0;
    uint64_t multicast_acks = 0;
    uint64_t dca_acks = 0;
    uint64_t trees_completed = 0;
};

struct IsaV1CollectiveAccelerationRuntimeResidual {
    size_t plans = 0;
    size_t active_batches = 0;
    size_t active_trees = 0;
    size_t live_sessions = 0;
    size_t pending_multicast_acks = 0;
    size_t pending_dca_acks = 0;

    bool Empty() const noexcept;
};

// Resolves the immutable P6/P7 SRAM contract for one multicast chunk.  The
// endpoint-reduce profile uses rank-major staging; a DCA-completed all-reduce
// broadcasts the already reduced result instead.
uint64_t IsaV1CollectiveMulticastSourceAddress(
    const IsaV1CollectivePlan &plan,
    const IsaV1CollectiveAcceleratedTree &tree, uint64_t chunk_offset);
uint64_t IsaV1CollectiveMulticastDestinationAddress(
    const IsaV1CollectivePlan &plan,
    const IsaV1CollectiveAcceleratedTree &tree, uint16_t target_rank,
    uint64_t chunk_offset);

class IsaV1CollectiveAccelerationRuntime final {
public:
    IsaV1CollectiveAccelerationRuntime(
        std::shared_ptr<const IsaV1CollectiveProgramImage> base,
        std::shared_ptr<const IsaV1CollectiveProfileProgramImage> profile,
        std::shared_ptr<IsaV1CollectiveTreeRegistryBridge> bridge,
        uint32_t first_session = 1);

    IsaV1CollectiveAccelerationRuntime(
        const IsaV1CollectiveAccelerationRuntime &) = delete;
    IsaV1CollectiveAccelerationRuntime &operator=(
        const IsaV1CollectiveAccelerationRuntime &) = delete;

    // Starts the next canonical batch when possible.  Nullopt means another
    // plan/batch currently owns the production tables or this tree belongs to
    // a later batch. Sessions are monotonically allocated and never reused.
    std::optional<uint16_t> TryAcquireMulticast(
        const CollectiveKey &key, uint16_t tree_id, uint32_t chunk_id);
    void RegisterMulticastPost(const CollectiveKey &key, uint16_t tree_id,
                               uint32_t chunk_id, uint16_t target_core);
    void RegisterDcaRootReady(const CollectiveKey &key, uint16_t tree_id,
                              uint32_t chunk_id, uint16_t root_core);
    bool TryAcquireDca(const CollectiveKey &key, uint16_t tree_id,
                       uint32_t chunk_id, uint16_t source_core);

    void AcknowledgeMulticast(const CollectiveKey &key, uint16_t tree_id,
                              uint32_t chunk_id, uint16_t target_core);
    void AcknowledgeDca(const CollectiveKey &key, uint16_t tree_id,
                        uint32_t chunk_id, uint16_t root_core);

    bool MulticastComplete(const CollectiveKey &key,
                           uint16_t tree_id) const;
    bool DcaSourcesReady(const CollectiveKey &key, uint16_t tree_id,
                         uint32_t chunk_id) const;
    bool DcaComplete(const CollectiveKey &key, uint16_t tree_id) const;
    bool TreeComplete(const CollectiveKey &key, uint16_t tree_id) const;
    bool PlanComplete(const CollectiveKey &key) const noexcept;
    // A completed plan remains queryable until every semantic participant has
    // performed its local cleanup.  The last unique observer retires it.
    void ObservePlanComplete(const CollectiveKey &key, uint16_t core);
    void Abort(const CollectiveKey &key);

    IsaV1CollectiveAccelerationRuntimeResidual Residual() const;
    const IsaV1CollectiveAccelerationRuntimeStats &Stats() const noexcept {
        return stats_;
    }

private:
    struct TreeState {
        struct DcaChunkState {
            bool root_ready = false;
            std::set<uint16_t> sources;
            bool ack = false;
        };
        uint16_t tree_id = 0;
        bool multicast = false;
        bool dca = false;
        uint16_t root_core = 0;
        uint32_t chunk_count = 0;
        std::set<uint16_t> multicast_targets;
        std::set<uint16_t> dca_sources;
        std::set<std::pair<uint32_t, uint16_t>> multicast_posts;
        std::map<uint32_t, uint16_t> multicast_sessions;
        std::set<std::pair<uint32_t, uint16_t>> multicast_acks;
        std::map<uint32_t, DcaChunkState> dca_chunks;
        bool marked_complete = false;
    };

    struct PlanState {
        CollectiveKey key;
        uint32_t plan_index = 0;
        size_t conflict_edges = 0;
        std::vector<IsaV1TreeBatch> batches;
        std::map<uint16_t, TreeState> trees;
        std::set<uint16_t> expected_observers;
        std::set<uint16_t> completion_observers;
        size_t next_batch = 0;
        bool complete = false;
    };

    struct ActiveBatch {
        CollectiveKey key;
        uint16_t batch_index = 0;
        std::set<uint16_t> tree_ids;
    };

    PlanState &FindPlan(const CollectiveKey &key);
    const PlanState &FindPlan(const CollectiveKey &key) const;
    TreeState &FindTree(PlanState &plan, uint16_t tree_id);
    const TreeState &FindTree(const PlanState &plan,
                              uint16_t tree_id) const;
    bool EnsureActive(PlanState &plan, uint16_t tree_id);
    bool MulticastDone(const TreeState &tree) const;
    bool DcaDone(const TreeState &tree) const;
    void MaybeCompleteTree(PlanState &plan, TreeState &tree);
    void MaybeCompleteBatch(PlanState &plan);
    uint16_t AllocateSession();

    std::shared_ptr<const IsaV1CollectiveProgramImage> base_;
    std::shared_ptr<const IsaV1CollectiveProfileProgramImage> profile_;
    std::shared_ptr<IsaV1CollectiveTreeRegistryBridge> bridge_;
    std::map<CollectiveKey, PlanState> plans_;
    std::set<CollectiveKey> completed_plans_;
    std::optional<ActiveBatch> active_;
    uint32_t next_session_ = 1;
    IsaV1CollectiveAccelerationRuntimeStats stats_;
};
