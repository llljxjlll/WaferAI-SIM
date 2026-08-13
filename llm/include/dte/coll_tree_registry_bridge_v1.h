#pragma once

#include "dte/coll_program_profile_v1.h"
#include "dte/coll_tree_batch_v1.h"

#include <cstddef>
#include <cstdint>
#include <memory>

struct IsaV1TreeRegistryBridgeStats {
    uint64_t batches_programmed = 0;
    uint64_t batches_released = 0;
    uint64_t batches_aborted = 0;
    uint64_t multicast_entries_programmed = 0;
    uint64_t multicast_entries_erased = 0;
    uint64_t reduce_nodes_programmed = 0;
    uint64_t reduce_nodes_erased = 0;
    size_t owned_multicast_entries_peak = 0;
    size_t owned_reduce_nodes_peak = 0;
};

struct IsaV1TreeRegistryBridgeResidual {
    IsaV1TreeBatchRuntimeResidual schedule;
    size_t registered_profile_plans = 0;
    size_t registered_tree_images = 0;
    size_t owned_multicast_entries = 0;
    size_t owned_reduce_nodes = 0;

    bool Empty() const;
};

// Transactional adapter between the pure batch runtime and the registries
// consumed by production Routers/DCA. BeginBatch really programs the tables;
// EndBatch and Abort really erase those exact tree IDs. Existing production
// tree IDs are hard collisions, never idempotent/silent reuse.
class IsaV1CollectiveTreeRegistryBridge {
public:
    explicit IsaV1CollectiveTreeRegistryBridge(
        IsaV1CollectiveTreeBatchRuntimeConfig config = {});
    ~IsaV1CollectiveTreeRegistryBridge();

    IsaV1CollectiveTreeRegistryBridge(
        const IsaV1CollectiveTreeRegistryBridge &) = delete;
    IsaV1CollectiveTreeRegistryBridge &operator=(
        const IsaV1CollectiveTreeRegistryBridge &) = delete;

    void RegisterPlan(const IsaV1CollectiveProfilePlanImage &plan);
    void RegisterImage(const IsaV1CollectiveProfileProgramImage &image);
    void BeginBatch(const CollectiveKey &key, uint16_t batch_index);
    void MarkTreeComplete(const CollectiveKey &key, uint16_t batch_index,
                          uint16_t tree_id);
    void EndBatch(const CollectiveKey &key, uint16_t batch_index);
    void Abort(const CollectiveKey &key);

    IsaV1TreeRegistryBridgeResidual Residual() const;
    const IsaV1TreeRegistryBridgeStats &Stats() const noexcept;
    const IsaV1CollectiveTreeBatchRuntime &BatchRuntime() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
