#pragma once

#include "dte/coll_topology_v1.h"
#include "dte/coll_types.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

inline constexpr size_t kIsaV1DefaultMaxRegisteredTreeSchedules = 1024;
inline constexpr size_t kIsaV1DefaultMaxRegisteredTrees = 65535;
inline constexpr size_t kIsaV1DefaultMaxPlannedTreeEntries = 1048576;
inline constexpr size_t kIsaV1DefaultMaxTreeBatchTraceEvents = 4096;

struct IsaV1CollectiveTreeBatchRuntimeConfig {
    size_t max_registered_schedules =
        kIsaV1DefaultMaxRegisteredTreeSchedules;
    size_t max_registered_trees = kIsaV1DefaultMaxRegisteredTrees;
    size_t max_planned_entries = kIsaV1DefaultMaxPlannedTreeEntries;
    size_t max_trace_events = kIsaV1DefaultMaxTreeBatchTraceEvents;
    uint16_t entries_per_router =
        kIsaV1CollectiveTreeEntriesPerRouter;
    // Capacity reserved by production entries not owned by this runtime.
    // These entries are never returned as ProgrammedEntries(), but are
    // included in capacity validation and total-occupancy peaks.
    std::map<uint16_t, uint16_t> reserved_entries_by_router;
};

enum class IsaV1TreeBatchTraceEventKind : uint8_t {
    PROGRAM = 0,
    RELEASE = 1
};

enum class IsaV1TreeBatchReleaseReason : uint8_t {
    NONE = 0,
    BATCH_COMPLETE = 1,
    SCHEDULE_COMPLETE = 2,
    ABORT = 3
};

struct IsaV1TreeBatchTraceEvent {
    uint64_t sequence = 0;
    IsaV1TreeBatchTraceEventKind kind =
        IsaV1TreeBatchTraceEventKind::PROGRAM;
    IsaV1TreeBatchReleaseReason release_reason =
        IsaV1TreeBatchReleaseReason::NONE;
    CollectiveKey key;
    uint16_t batch_index = 0;
    std::vector<uint16_t> tree_ids;
    size_t entry_count = 0;
    size_t managed_occupancy_after = 0;
    size_t total_occupancy_after = 0;
};

struct IsaV1TreeBatchRuntimeStats {
    uint64_t batches_programmed = 0;
    uint64_t batches_released = 0;
    uint64_t schedules_aborted = 0;
    uint64_t trees_programmed = 0;
    uint64_t trees_erased = 0;
    uint64_t entries_programmed = 0;
    uint64_t entries_erased = 0;
    uint64_t release_batch_complete = 0;
    uint64_t release_schedule_complete = 0;
    uint64_t release_abort = 0;
    size_t managed_occupancy_peak = 0;
    size_t total_occupancy_peak = 0;
    std::map<uint16_t, uint16_t> peak_entries_by_router;
    uint64_t trace_events_dropped = 0;
};

struct IsaV1TreeBatchRuntimeResidual {
    size_t registered_schedules = 0;
    size_t registered_trees = 0;
    size_t planned_entries = 0;
    size_t active_batches = 0;
    size_t programmed_trees = 0;
    size_t programmed_entries = 0;
    size_t completed_trees = 0;

    bool Empty() const;
};

struct IsaV1ProgrammedTreeEntry {
    uint16_t tree_id = 0;
    IsaV1CollectiveTreeEntry entry;

    bool operator==(const IsaV1ProgrammedTreeEntry &other) const;
};

// A bounded, pure batch-level lifecycle for the production tree image.
// The conservative v1 runtime allows at most one globally active batch.  It
// therefore never depends on runtime lock arbitration for progress: after a
// successful BeginBatch, ProgrammedEntries contains exactly that batch; after
// EndBatch or Abort, all entries owned by that batch have been erased.
class IsaV1CollectiveTreeBatchRuntime {
public:
    explicit IsaV1CollectiveTreeBatchRuntime(
        IsaV1CollectiveTreeBatchRuntimeConfig config = {});
    ~IsaV1CollectiveTreeBatchRuntime();

    IsaV1CollectiveTreeBatchRuntime(
        const IsaV1CollectiveTreeBatchRuntime &) = delete;
    IsaV1CollectiveTreeBatchRuntime &operator=(
        const IsaV1CollectiveTreeBatchRuntime &) = delete;

    // Registration is strongly transactional for all contract/capacity
    // errors.  The schedule is recomputed from topology entries rather than
    // accepting caller-provided resource metadata.
    IsaV1TreeSchedule RegisterSchedule(
        const CollectiveKey &key,
        const std::vector<IsaV1CollectiveTreeTopology> &trees,
        size_t max_trees_per_batch = static_cast<size_t>(-1));

    void BeginBatch(const CollectiveKey &key, uint16_t batch_index);
    void MarkTreeComplete(const CollectiveKey &key, uint16_t batch_index,
                          uint16_t tree_id);
    void EndBatch(const CollectiveKey &key, uint16_t batch_index);

    // Aborts an exact live key.  If its batch is active, only entries owned by
    // that key are erased.  The epoch is retired and cannot be registered
    // again.  Unknown/stale keys are rejected.
    void Abort(const CollectiveKey &key);

    bool IsTreeProgrammed(uint16_t tree_id) const;
    std::vector<uint16_t> ProgrammedTreeIds() const;
    std::vector<IsaV1ProgrammedTreeEntry> ProgrammedEntries() const;
    IsaV1TreeBatchRuntimeResidual Residual() const;
    const IsaV1TreeBatchRuntimeStats &Stats() const;
    std::vector<IsaV1TreeBatchTraceEvent> TraceEvents() const;
    uint16_t EntriesPerRouterCapacity() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
