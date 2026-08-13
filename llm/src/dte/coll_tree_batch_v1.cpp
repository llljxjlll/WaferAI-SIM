#include "dte/coll_tree_batch_v1.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace {

using CollectiveSeriesKey = std::pair<uint32_t, uint32_t>;

struct ProgrammedEntryKey {
    uint16_t tree_id = 0;
    uint16_t router_id = 0;
    Directions ingress = CENTER;

    bool operator<(const ProgrammedEntryKey &other) const {
        return std::tie(tree_id, router_id, ingress) <
               std::tie(other.tree_id, other.router_id, other.ingress);
    }
};

size_t CheckedAdd(size_t a, size_t b, const char *message) {
    if (b > std::numeric_limits<size_t>::max() - a)
        throw std::overflow_error(message);
    return a + b;
}

size_t SumOccupancy(const std::map<uint16_t, uint16_t> &occupancy) {
    size_t total = 0;
    for (const auto &entry : occupancy)
        total = CheckedAdd(total, entry.second,
                           "ISA-v1 tree occupancy sum overflows");
    return total;
}

void ValidateTopologyImage(const IsaV1CollectiveTreeTopology &tree) {
    if (tree.tree_id == 0)
        throw std::invalid_argument("ISA-v1 batch tree_id zero is reserved");
    if (tree.group.empty() ||
        !std::is_sorted(tree.group.begin(), tree.group.end()) ||
        std::adjacent_find(tree.group.begin(), tree.group.end()) !=
            tree.group.end() ||
        !std::binary_search(tree.group.begin(), tree.group.end(), tree.root))
        throw std::invalid_argument(
            "ISA-v1 batch tree has invalid canonical group/root");
    if (tree.entries.empty())
        throw std::invalid_argument(
            "ISA-v1 batch runtime forbids an empty multicast tree");

    uint16_t previous_router = 0;
    bool first = true;
    bool saw_root = false;
    for (const auto &entry : tree.entries) {
        if (!first && entry.router_id <= previous_router)
            throw std::invalid_argument(
                "ISA-v1 batch tree entries are not canonical/unique");
        first = false;
        previous_router = entry.router_id;
        if (entry.ingress < WEST || entry.ingress >= DIRECTIONS)
            throw std::invalid_argument(
                "ISA-v1 batch tree ingress is invalid");
        if (entry.output_mask == 0 ||
            (entry.output_mask &
             ~static_cast<uint8_t>((1U << DIRECTIONS) - 1U)) != 0 ||
            (entry.output_mask & (1U << entry.ingress)) != 0)
            throw std::invalid_argument(
                "ISA-v1 batch tree output bitmap is invalid");
        if (entry.router_id == tree.root) {
            if (entry.ingress != CENTER)
                throw std::invalid_argument(
                    "ISA-v1 batch tree root ingress must be CENTER");
            saw_root = true;
        } else if (entry.ingress == CENTER) {
            throw std::invalid_argument(
                "ISA-v1 batch non-root ingress cannot be CENTER");
        }
    }
    if (!saw_root)
        throw std::invalid_argument(
            "ISA-v1 batch tree image has no root entry");
    (void)IsaV1CollectiveTreeOutputResources(tree);
}

} // namespace

struct IsaV1CollectiveTreeBatchRuntime::Impl {
    struct ScheduleState {
        CollectiveKey key;
        std::map<uint16_t, IsaV1CollectiveTreeTopology> trees;
        IsaV1TreeSchedule schedule;
        size_t next_batch = 0;
        size_t planned_entries = 0;
    };

    struct ActiveBatch {
        CollectiveKey key;
        uint16_t batch_index = 0;
        std::set<uint16_t> completed_trees;
    };

    explicit Impl(IsaV1CollectiveTreeBatchRuntimeConfig value)
        : config(std::move(value)) {
        if (config.max_registered_schedules == 0 ||
            config.max_registered_trees == 0 ||
            config.max_planned_entries == 0 || config.max_trace_events == 0 ||
            config.entries_per_router == 0)
            throw std::invalid_argument(
                "ISA-v1 tree batch runtime capacities must be positive");
        for (const auto &reserved : config.reserved_entries_by_router)
            if (reserved.second > config.entries_per_router)
                throw std::invalid_argument(
                    "ISA-v1 reserved tree occupancy exceeds Router capacity");
        stats.total_occupancy_peak =
            SumOccupancy(config.reserved_entries_by_router);
        stats.peak_entries_by_router = config.reserved_entries_by_router;
    }

    IsaV1CollectiveTreeBatchRuntimeConfig config;
    std::map<CollectiveKey, ScheduleState> schedules;
    std::map<CollectiveSeriesKey, CollectiveKey> live_series;
    std::map<CollectiveSeriesKey, uint32_t> retired_epoch;
    std::map<uint16_t, CollectiveKey> tree_owners;
    std::map<ProgrammedEntryKey, uint8_t> programmed_entries;
    std::set<uint16_t> programmed_trees;
    std::unique_ptr<ActiveBatch> active;
    size_t registered_tree_count = 0;
    size_t planned_entry_count = 0;
    IsaV1TreeBatchRuntimeStats stats;
    std::deque<IsaV1TreeBatchTraceEvent> trace;
    uint64_t next_trace_sequence = 0;

    const ScheduleState &FindSchedule(const CollectiveKey &key) const {
        const auto found = schedules.find(key);
        if (found == schedules.end())
            throw std::runtime_error(
                "ISA-v1 tree batch key is unknown or stale");
        return found->second;
    }

    ScheduleState &FindSchedule(const CollectiveKey &key) {
        const auto found = schedules.find(key);
        if (found == schedules.end())
            throw std::runtime_error(
                "ISA-v1 tree batch key is unknown or stale");
        return found->second;
    }

    const IsaV1TreeBatch &FindBatch(const ScheduleState &state,
                                    uint16_t batch_index) const {
        if (batch_index >= state.schedule.batches.size())
            throw std::runtime_error(
                "ISA-v1 tree batch index is outside schedule");
        const auto &batch = state.schedule.batches[batch_index];
        if (batch.batch_index != batch_index)
            throw std::logic_error(
                "ISA-v1 tree schedule has non-canonical batch indices");
        return batch;
    }

    size_t ReservedTotal() const {
        return SumOccupancy(config.reserved_entries_by_router);
    }

    void PushTrace(IsaV1TreeBatchTraceEvent event) {
        event.sequence = next_trace_sequence++;
        if (trace.size() == config.max_trace_events) {
            trace.pop_front();
            ++stats.trace_events_dropped;
        }
        trace.push_back(std::move(event));
    }

    void UpdatePeaks() {
        stats.managed_occupancy_peak =
            std::max(stats.managed_occupancy_peak, programmed_entries.size());
        const size_t total =
            CheckedAdd(ReservedTotal(), programmed_entries.size(),
                       "ISA-v1 total tree occupancy overflows");
        stats.total_occupancy_peak =
            std::max(stats.total_occupancy_peak, total);
        std::map<uint16_t, uint16_t> current =
            config.reserved_entries_by_router;
        for (const auto &entry : programmed_entries) {
            auto &count = current[entry.first.router_id];
            if (count == std::numeric_limits<uint16_t>::max())
                throw std::overflow_error(
                    "ISA-v1 per-Router tree occupancy overflows u16");
            ++count;
        }
        for (const auto &entry : current) {
            if (entry.second > config.entries_per_router)
                throw std::logic_error(
                    "ISA-v1 programmed tree occupancy exceeds capacity");
            auto &peak = stats.peak_entries_by_router[entry.first];
            peak = std::max(peak, entry.second);
        }
    }

    void RetireSchedule(const CollectiveKey &key) {
        auto found = schedules.find(key);
        if (found == schedules.end())
            throw std::logic_error("ISA-v1 retiring an unknown tree schedule");
        const CollectiveSeriesKey series{key.group_id, key.collective_id};
        const size_t tree_count = found->second.trees.size();
        const size_t entry_count = found->second.planned_entries;
        for (const auto &tree : found->second.trees) {
            auto owner = tree_owners.find(tree.first);
            if (owner == tree_owners.end() || !(owner->second == key))
                throw std::logic_error(
                    "ISA-v1 tree owner registry is corrupt");
        }
        for (const auto &tree : found->second.trees)
            tree_owners.erase(tree.first);
        schedules.erase(found);
        live_series.erase(series);
        auto retired = retired_epoch.find(series);
        if (retired == retired_epoch.end())
            retired_epoch.emplace(series, key.epoch);
        else
            retired->second = std::max(retired->second, key.epoch);
        registered_tree_count -= tree_count;
        planned_entry_count -= entry_count;
    }
};

bool IsaV1TreeBatchRuntimeResidual::Empty() const {
    return registered_schedules == 0 && registered_trees == 0 &&
           planned_entries == 0 && active_batches == 0 &&
           programmed_trees == 0 && programmed_entries == 0 &&
           completed_trees == 0;
}

bool IsaV1ProgrammedTreeEntry::operator==(
    const IsaV1ProgrammedTreeEntry &other) const {
    return tree_id == other.tree_id && entry == other.entry;
}

IsaV1CollectiveTreeBatchRuntime::IsaV1CollectiveTreeBatchRuntime(
    IsaV1CollectiveTreeBatchRuntimeConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

IsaV1CollectiveTreeBatchRuntime::~IsaV1CollectiveTreeBatchRuntime() = default;

IsaV1TreeSchedule IsaV1CollectiveTreeBatchRuntime::RegisterSchedule(
    const CollectiveKey &key,
    const std::vector<IsaV1CollectiveTreeTopology> &trees,
    size_t max_trees_per_batch) {
    if (trees.empty())
        throw std::invalid_argument(
            "ISA-v1 tree batch schedule must contain trees");
    if (impl_->schedules.size() >= impl_->config.max_registered_schedules)
        throw std::overflow_error(
            "ISA-v1 registered tree schedule capacity exhausted");

    const CollectiveSeriesKey series{key.group_id, key.collective_id};
    const auto retired = impl_->retired_epoch.find(series);
    if (retired != impl_->retired_epoch.end() && key.epoch <= retired->second)
        throw std::runtime_error(
            "ISA-v1 tree schedule epoch is stale");
    if (impl_->live_series.count(series) != 0)
        throw std::runtime_error(
            "ISA-v1 tree schedule series already has a live epoch");
    if (impl_->schedules.count(key) != 0)
        throw std::runtime_error(
            "ISA-v1 duplicate tree batch schedule key");

    Impl::ScheduleState candidate;
    candidate.key = key;
    std::vector<IsaV1TreeScheduleInput> inputs;
    inputs.reserve(trees.size());
    for (const auto &tree : trees) {
        ValidateTopologyImage(tree);
        if (impl_->tree_owners.count(tree.tree_id) != 0)
            throw std::runtime_error(
                "ISA-v1 live tree_id collision across schedules");
        if (!candidate.trees.emplace(tree.tree_id, tree).second)
            throw std::invalid_argument(
                "ISA-v1 duplicate tree_id inside schedule");
        candidate.planned_entries = CheckedAdd(
            candidate.planned_entries, tree.entries.size(),
            "ISA-v1 planned tree entry count overflows");
        inputs.push_back(IsaV1TreeScheduleInputFromTopology(tree));
    }
    if (CheckedAdd(impl_->registered_tree_count, candidate.trees.size(),
                   "ISA-v1 registered tree count overflows") >
        impl_->config.max_registered_trees)
        throw std::overflow_error(
            "ISA-v1 registered tree capacity exhausted");
    if (CheckedAdd(impl_->planned_entry_count, candidate.planned_entries,
                   "ISA-v1 planned entry count overflows") >
        impl_->config.max_planned_entries)
        throw std::overflow_error(
            "ISA-v1 planned tree entry capacity exhausted");

    IsaV1TreeScheduleOptions options;
    options.max_trees_per_batch = max_trees_per_batch;
    options.entries_per_router = impl_->config.entries_per_router;
    options.occupied_entries_by_router =
        impl_->config.reserved_entries_by_router;
    candidate.schedule = ScheduleIsaV1CollectiveTrees(inputs, options);
    if (candidate.schedule.batches.empty())
        throw std::logic_error(
            "ISA-v1 non-empty tree schedule produced no batches");

    const IsaV1TreeSchedule result = candidate.schedule;
    impl_->schedules.emplace(key, std::move(candidate));
    impl_->live_series.emplace(series, key);
    for (const auto &tree : trees) impl_->tree_owners.emplace(tree.tree_id, key);
    impl_->registered_tree_count += trees.size();
    impl_->planned_entry_count +=
        impl_->schedules.at(key).planned_entries;
    return result;
}

void IsaV1CollectiveTreeBatchRuntime::BeginBatch(
    const CollectiveKey &key, uint16_t batch_index) {
    Impl::ScheduleState &state = impl_->FindSchedule(key);
    if (impl_->active)
        throw std::runtime_error(
            "ISA-v1 another collective tree batch is already active");
    if (!impl_->programmed_entries.empty() || !impl_->programmed_trees.empty())
        throw std::logic_error(
            "ISA-v1 inactive tree runtime retains programmed state");
    if (state.next_batch != batch_index)
        throw std::runtime_error(
            "ISA-v1 tree batch begin is stale or out of order");
    const IsaV1TreeBatch &batch = impl_->FindBatch(state, batch_index);

    std::map<ProgrammedEntryKey, uint8_t> candidate_entries;
    std::set<uint16_t> candidate_trees;
    std::map<uint16_t, uint16_t> occupancy =
        impl_->config.reserved_entries_by_router;
    for (uint16_t tree_id : batch.tree_ids) {
        const auto topology = state.trees.find(tree_id);
        if (topology == state.trees.end())
            throw std::logic_error(
                "ISA-v1 scheduled batch references an unknown tree");
        if (!candidate_trees.insert(tree_id).second)
            throw std::logic_error(
                "ISA-v1 scheduled batch duplicates a tree");
        for (const auto &entry : topology->second.entries) {
            const ProgrammedEntryKey entry_key{
                tree_id, entry.router_id, entry.ingress};
            if (!candidate_entries.emplace(entry_key, entry.output_mask).second)
                throw std::logic_error(
                    "ISA-v1 tree batch duplicates a programmed entry");
            auto &count = occupancy[entry.router_id];
            if (count >= impl_->config.entries_per_router)
                throw std::runtime_error(
                    "ISA-v1 tree batch exceeds Router table capacity");
            ++count;
        }
    }

    auto active = std::make_unique<Impl::ActiveBatch>();
    active->key = key;
    active->batch_index = batch_index;
    impl_->programmed_entries = std::move(candidate_entries);
    impl_->programmed_trees = std::move(candidate_trees);
    impl_->active = std::move(active);
    ++impl_->stats.batches_programmed;
    impl_->stats.trees_programmed += batch.tree_ids.size();
    impl_->stats.entries_programmed += impl_->programmed_entries.size();
    impl_->UpdatePeaks();

    IsaV1TreeBatchTraceEvent event;
    event.kind = IsaV1TreeBatchTraceEventKind::PROGRAM;
    event.key = key;
    event.batch_index = batch_index;
    event.tree_ids = batch.tree_ids;
    event.entry_count = impl_->programmed_entries.size();
    event.managed_occupancy_after = impl_->programmed_entries.size();
    event.total_occupancy_after = CheckedAdd(
        impl_->ReservedTotal(), impl_->programmed_entries.size(),
        "ISA-v1 trace occupancy overflows");
    impl_->PushTrace(std::move(event));
}

void IsaV1CollectiveTreeBatchRuntime::MarkTreeComplete(
    const CollectiveKey &key, uint16_t batch_index, uint16_t tree_id) {
    (void)impl_->FindSchedule(key);
    if (!impl_->active || !(impl_->active->key == key) ||
        impl_->active->batch_index != batch_index)
        throw std::runtime_error(
            "ISA-v1 tree completion key/batch mismatch");
    if (impl_->programmed_trees.count(tree_id) == 0)
        throw std::runtime_error(
            "ISA-v1 completion references a tree outside active batch");
    if (!impl_->active->completed_trees.insert(tree_id).second)
        throw std::runtime_error(
            "ISA-v1 duplicate tree completion");
}

void IsaV1CollectiveTreeBatchRuntime::EndBatch(
    const CollectiveKey &key, uint16_t batch_index) {
    Impl::ScheduleState &state = impl_->FindSchedule(key);
    if (!impl_->active || !(impl_->active->key == key) ||
        impl_->active->batch_index != batch_index)
        throw std::runtime_error(
            "ISA-v1 tree release key/batch mismatch");
    const IsaV1TreeBatch &batch = impl_->FindBatch(state, batch_index);
    if (impl_->active->completed_trees.size() != batch.tree_ids.size())
        throw std::runtime_error(
            "ISA-v1 tree batch cannot release before every tree completes");
    for (uint16_t tree_id : batch.tree_ids)
        if (impl_->active->completed_trees.count(tree_id) == 0 ||
            impl_->programmed_trees.count(tree_id) == 0)
            throw std::logic_error(
                "ISA-v1 active tree completion/program state is corrupt");

    const size_t erased_entries = impl_->programmed_entries.size();
    const size_t erased_trees = impl_->programmed_trees.size();
    impl_->programmed_entries.clear();
    impl_->programmed_trees.clear();
    impl_->active.reset();
    ++state.next_batch;
    const bool final = state.next_batch == state.schedule.batches.size();
    const IsaV1TreeBatchReleaseReason reason =
        final ? IsaV1TreeBatchReleaseReason::SCHEDULE_COMPLETE
              : IsaV1TreeBatchReleaseReason::BATCH_COMPLETE;
    ++impl_->stats.batches_released;
    impl_->stats.trees_erased += erased_trees;
    impl_->stats.entries_erased += erased_entries;
    if (final)
        ++impl_->stats.release_schedule_complete;
    else
        ++impl_->stats.release_batch_complete;

    IsaV1TreeBatchTraceEvent event;
    event.kind = IsaV1TreeBatchTraceEventKind::RELEASE;
    event.release_reason = reason;
    event.key = key;
    event.batch_index = batch_index;
    event.tree_ids = batch.tree_ids;
    event.entry_count = erased_entries;
    event.managed_occupancy_after = impl_->programmed_entries.size();
    event.total_occupancy_after = impl_->ReservedTotal();
    impl_->PushTrace(std::move(event));
    if (final) impl_->RetireSchedule(key);
}

void IsaV1CollectiveTreeBatchRuntime::Abort(const CollectiveKey &key) {
    Impl::ScheduleState &state = impl_->FindSchedule(key);
    uint16_t batch_index = 0;
    std::vector<uint16_t> tree_ids;
    size_t erased_entries = 0;
    size_t erased_trees = 0;
    if (impl_->active && impl_->active->key == key) {
        batch_index = impl_->active->batch_index;
        tree_ids.assign(impl_->programmed_trees.begin(),
                        impl_->programmed_trees.end());
        erased_entries = impl_->programmed_entries.size();
        erased_trees = impl_->programmed_trees.size();
        impl_->programmed_entries.clear();
        impl_->programmed_trees.clear();
        impl_->active.reset();
    } else {
        if (state.next_batch > std::numeric_limits<uint16_t>::max())
            throw std::logic_error(
                "ISA-v1 abort batch index exceeds u16");
        batch_index = static_cast<uint16_t>(state.next_batch);
    }
    ++impl_->stats.schedules_aborted;
    ++impl_->stats.release_abort;
    impl_->stats.trees_erased += erased_trees;
    impl_->stats.entries_erased += erased_entries;

    IsaV1TreeBatchTraceEvent event;
    event.kind = IsaV1TreeBatchTraceEventKind::RELEASE;
    event.release_reason = IsaV1TreeBatchReleaseReason::ABORT;
    event.key = key;
    event.batch_index = batch_index;
    event.tree_ids = std::move(tree_ids);
    event.entry_count = erased_entries;
    event.managed_occupancy_after = impl_->programmed_entries.size();
    event.total_occupancy_after = CheckedAdd(
        impl_->ReservedTotal(), impl_->programmed_entries.size(),
        "ISA-v1 abort trace occupancy overflows");
    impl_->PushTrace(std::move(event));
    impl_->RetireSchedule(key);
}

bool IsaV1CollectiveTreeBatchRuntime::IsTreeProgrammed(
    uint16_t tree_id) const {
    return impl_->programmed_trees.count(tree_id) != 0;
}

std::vector<uint16_t>
IsaV1CollectiveTreeBatchRuntime::ProgrammedTreeIds() const {
    return {impl_->programmed_trees.begin(), impl_->programmed_trees.end()};
}

std::vector<IsaV1ProgrammedTreeEntry>
IsaV1CollectiveTreeBatchRuntime::ProgrammedEntries() const {
    std::vector<IsaV1ProgrammedTreeEntry> result;
    result.reserve(impl_->programmed_entries.size());
    for (const auto &entry : impl_->programmed_entries)
        result.push_back({entry.first.tree_id,
                          {entry.first.router_id, entry.first.ingress,
                           entry.second}});
    return result;
}

IsaV1TreeBatchRuntimeResidual
IsaV1CollectiveTreeBatchRuntime::Residual() const {
    IsaV1TreeBatchRuntimeResidual result;
    result.registered_schedules = impl_->schedules.size();
    result.registered_trees = impl_->registered_tree_count;
    result.planned_entries = impl_->planned_entry_count;
    result.active_batches = impl_->active ? 1 : 0;
    result.programmed_trees = impl_->programmed_trees.size();
    result.programmed_entries = impl_->programmed_entries.size();
    result.completed_trees =
        impl_->active ? impl_->active->completed_trees.size() : 0;
    return result;
}

const IsaV1TreeBatchRuntimeStats &
IsaV1CollectiveTreeBatchRuntime::Stats() const {
    return impl_->stats;
}

std::vector<IsaV1TreeBatchTraceEvent>
IsaV1CollectiveTreeBatchRuntime::TraceEvents() const {
    return {impl_->trace.begin(), impl_->trace.end()};
}

uint16_t
IsaV1CollectiveTreeBatchRuntime::EntriesPerRouterCapacity() const noexcept {
    return impl_->config.entries_per_router;
}
