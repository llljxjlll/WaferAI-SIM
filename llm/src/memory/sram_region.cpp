#include "memory/sram/sram_region.h"
#include "trace/Event_engine.h"

#include <algorithm>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace sram {
namespace {

Initiator ParseInitiator(const std::string &name) {
    if (name == "compute") return Initiator::kCompute;
    if (name == "dte") return Initiator::kDte;
    if (name == "lsu") return Initiator::kLsu;
    if (name == "noc_rx") return Initiator::kNocRx;
    if (name == "legacy") return Initiator::kLegacy;
    throw std::invalid_argument("unknown SRAM initiator '" + name + "'");
}

AllocatorKind ParseAllocator(const std::string &name) {
    if (name == "fixed") return AllocatorKind::kFixed;
    if (name == "block") return AllocatorKind::kBlock;
    throw std::invalid_argument("unknown SRAM allocator '" + name + "'");
}

PortConfig ParsePort(const nlohmann::json &parent, const char *field,
                     PortConfig fallback) {
    if (!parent.contains(field)) return fallback;
    const auto &node = parent.at(field);
    PortConfig result = fallback;
    if (node.is_number_unsigned() || node.is_number_integer()) {
        result.count = node.get<uint32_t>();
        return result;
    }
    result.count = node.value("count", result.count);
    result.width_bits = node.value("width_bits", result.width_bits);
    return result;
}

void ParseInitiatorPorts(const nlohmann::json &sram_json, const char *name,
                         InitiatorPortConfig *target) {
    if (!sram_json.contains("ports")) return;
    const auto &ports = sram_json.at("ports");
    if (!ports.contains(name)) return;
    const auto &node = ports.at(name);
    target->read = ParsePort(node, "read", target->read);
    target->write = ParsePort(node, "write", target->write);
}

std::vector<Initiator> DefaultAccess() {
    return {Initiator::kCompute, Initiator::kDte, Initiator::kLsu,
            Initiator::kNocRx, Initiator::kLegacy};
}

std::string RegionLabel(const RegionConfig &region) {
    std::ostringstream os;
    os << "SRAM region '" << region.name << "'";
    return os.str();
}

} // namespace

const char *ToString(Initiator initiator) {
    switch (initiator) {
    case Initiator::kCompute: return "compute";
    case Initiator::kDte: return "dte";
    case Initiator::kLsu: return "lsu";
    case Initiator::kNocRx: return "noc_rx";
    case Initiator::kLegacy: return "legacy";
    }
    return "unknown";
}

const char *ToString(Command command) {
    switch (command) {
    case Command::kRead: return "read";
    case Command::kWrite: return "write";
    case Command::kClear: return "clear";
    }
    return "unknown";
}

Config ParseConfig(const nlohmann::json &memory_json) {
    Config result;
    result.capacity_bytes = memory_json.value("sram_size", uint64_t{0});

    const nlohmann::json empty = nlohmann::json::object();
    const auto &node = memory_json.contains("sram") ? memory_json.at("sram")
                                                     : empty;
    result.capacity_bytes = node.value("capacity_bytes", result.capacity_bytes);
    result.allocation_alignment_bytes =
        node.value("allocation_alignment_bytes", uint64_t{1});
    result.bank_count = node.value("bank_count", uint32_t{1});
    result.bank_interleave_bytes =
        node.value("bank_interleave_bytes", uint64_t{16});
    result.read_base_latency_cycles =
        node.value("read_base_latency_cycles", uint32_t{1});
    result.write_base_latency_cycles =
        node.value("write_base_latency_cycles", uint32_t{1});
    result.queue_depth = node.value("queue_depth", uint32_t{16});
    result.real_data_path = node.value("real_data_path", false);
    result.manual_regions = node.value("manual_regions", false);
    result.manual_memory_schedule =
        node.value("manual_memory_schedule", false);

    const auto &lsu_node = memory_json.contains("lsu")
                               ? memory_json.at("lsu")
                               : (node.contains("lsu") ? node.at("lsu")
                                                       : empty);
    result.lsu_queue_depth =
        lsu_node.value("queue_depth", result.lsu_queue_depth);
    result.lsu_max_outstanding =
        lsu_node.value("max_outstanding", result.lsu_max_outstanding);
    result.lsu_issue_latency_ns =
        lsu_node.value("issue_latency_ns", result.lsu_issue_latency_ns);
    const auto &dte_memory = node.contains("dte_memory")
                                 ? node.at("dte_memory")
                                 : empty;
    result.dte_memory_workers =
        dte_memory.value("workers", result.dte_memory_workers);
    result.dte_memory_queue_depth =
        dte_memory.value("queue_depth", result.dte_memory_queue_depth);

    ParseInitiatorPorts(node, "compute", &result.compute);
    ParseInitiatorPorts(node, "dte", &result.dte);
    ParseInitiatorPorts(node, "lsu", &result.lsu);
    ParseInitiatorPorts(node, "noc_rx", &result.noc_rx);
    ParseInitiatorPorts(node, "legacy", &result.legacy);

    if (node.contains("regions")) {
        if (!node.at("regions").is_array())
            throw std::invalid_argument("memory.sram.regions must be an array");
        for (const auto &item : node.at("regions")) {
            RegionConfig region;
            region.name = item.at("name").get<std::string>();
            region.base_bytes = item.at("base_bytes").get<uint64_t>();
            region.size_bytes = item.at("size_bytes").get<uint64_t>();
            region.allocator =
                ParseAllocator(item.value("allocator", std::string("fixed")));
            region.spillable = item.value("spillable", false);
            if (item.contains("access")) {
                if (!item.at("access").is_array())
                    throw std::invalid_argument(RegionLabel(region) +
                                                " access must be an array");
                for (const auto &access : item.at("access"))
                    region.access.push_back(
                        ParseInitiator(access.get<std::string>()));
            } else {
                region.access = DefaultAccess();
            }
            result.regions.push_back(std::move(region));
        }
    }

    if (result.regions.empty()) {
        RegionConfig legacy;
        legacy.name = "legacy";
        legacy.size_bytes = result.capacity_bytes;
        legacy.allocator = AllocatorKind::kBlock;
        legacy.spillable = true;
        legacy.access = DefaultAccess();
        result.regions.push_back(std::move(legacy));
    }

    ValidateConfig(result);
    return result;
}

void ValidateConfig(const Config &config) {
    if (config.capacity_bytes == 0)
        throw std::invalid_argument("SRAM capacity must be non-zero");
    if (config.allocation_alignment_bytes == 0)
        throw std::invalid_argument("SRAM allocation alignment must be non-zero");
    if (config.bank_count == 0)
        throw std::invalid_argument("SRAM bank_count must be non-zero");
    if (config.bank_interleave_bytes == 0)
        throw std::invalid_argument(
            "SRAM bank_interleave_bytes must be non-zero");
    if (config.queue_depth == 0)
        throw std::invalid_argument("SRAM queue_depth must be non-zero");
    if (config.lsu_queue_depth == 0 || config.lsu_max_outstanding == 0)
        throw std::invalid_argument(
            "LSU queue_depth and max_outstanding must be non-zero");
    if (config.lsu_max_outstanding > config.lsu_queue_depth)
        throw std::invalid_argument(
            "LSU max_outstanding cannot exceed queue_depth");
    if (config.dte_memory_workers == 0 ||
        config.dte_memory_queue_depth == 0)
        throw std::invalid_argument(
            "DTE memory workers and queue_depth must be non-zero");

    const InitiatorPortConfig ports[] = {
        config.compute, config.dte, config.lsu, config.noc_rx, config.legacy};
    for (const auto &p : ports) {
        if (p.read.count == 0 || p.write.count == 0 ||
            p.read.width_bits == 0 || p.write.width_bits == 0)
            throw std::invalid_argument(
                "SRAM port count and width_bits must be non-zero");
        if ((p.read.width_bits % 8) != 0 || (p.write.width_bits % 8) != 0)
            throw std::invalid_argument(
                "SRAM port width_bits must be byte-addressable");
    }

    std::set<std::string> names;
    std::vector<ByteRange> ranges;
    ranges.reserve(config.regions.size());
    for (const auto &region : config.regions) {
        if (region.name.empty())
            throw std::invalid_argument("SRAM region name must not be empty");
        if (!names.insert(region.name).second)
            throw std::invalid_argument("duplicate SRAM region name '" +
                                        region.name + "'");
        if (region.size_bytes == 0)
            throw std::invalid_argument(RegionLabel(region) +
                                        " size must be non-zero");
        if (region.base_bytes % config.allocation_alignment_bytes != 0 ||
            region.size_bytes % config.allocation_alignment_bytes != 0)
            throw std::invalid_argument(RegionLabel(region) +
                                        " is not allocation-aligned");
        ByteRange range{region.base_bytes, region.size_bytes};
        if (range.End() > config.capacity_bytes)
            throw std::out_of_range(RegionLabel(region) +
                                    " exceeds SRAM capacity");
        if (region.access.empty())
            throw std::invalid_argument(RegionLabel(region) +
                                        " has no permitted initiator");
        for (const auto &existing : ranges) {
            if (Overlaps(range, existing))
                throw std::invalid_argument(RegionLabel(region) +
                                            " overlaps another region");
        }
        ranges.push_back(range);
    }
}

ConfigRegistry &ConfigRegistry::Instance() {
    static ConfigRegistry registry;
    return registry;
}

void ConfigRegistry::Configure(const nlohmann::json &hardware_json,
                               int total_cores) {
    if (total_cores <= 0)
        throw std::invalid_argument("SRAM registry requires at least one core");
    const auto memory = hardware_json.contains("memory")
                            ? hardware_json.at("memory")
                            : nlohmann::json::object();
    global_ = ParseConfig(memory);
    per_core_.clear();

    if (hardware_json.contains("cores")) {
        const auto &cores = hardware_json.at("cores");
        if (!cores.is_array())
            throw std::invalid_argument("hardware cores must be an array");
        for (size_t index = 0; index < cores.size(); ++index) {
            const auto &core = cores.at(index);
            if (!core.contains("sram") && !core.contains("lsu")) continue;
            const int core_id = core.value("id", static_cast<int>(index));
            if (core_id < 0 || core_id >= total_cores)
                throw std::out_of_range("per-core SRAM override id is invalid");
            nlohmann::json per_memory = memory;
            if (core.contains("sram")) {
                if (!per_memory.contains("sram"))
                    per_memory["sram"] = nlohmann::json::object();
                per_memory["sram"].merge_patch(core.at("sram"));
            }
            if (core.contains("lsu"))
                per_memory["lsu"] = core.at("lsu");
            per_core_[core_id] = ParseConfig(per_memory);
        }
    }
    configured_ = true;
}

const Config &ConfigRegistry::ForCore(int core_id) const {
    if (!configured_)
        throw std::logic_error("SRAM config registry is not configured");
    const auto it = per_core_.find(core_id);
    return it == per_core_.end() ? global_ : it->second;
}

void ConfigRegistry::ResetForTest() {
    global_ = Config{};
    per_core_.clear();
    configured_ = false;
}

RegionTable::RegionTable(Config config, Event_engine *event_engine,
                         int core_id)
    : config_(std::move(config)), event_engine_(event_engine),
      core_id_(core_id) {
    ValidateConfig(config_);
    for (size_t i = 0; i < config_.regions.size(); ++i) {
        const auto &region = config_.regions[i];
        name_to_id_.emplace(region.name, static_cast<int>(i));
        if (region.allocator == AllocatorKind::kBlock)
            free_spans_[static_cast<int>(i)].push_back(
                FreeSpan{0, region.size_bytes});
    }
}

int RegionTable::RegionId(std::string_view name) const {
    const auto it = name_to_id_.find(std::string(name));
    if (it == name_to_id_.end())
        throw std::out_of_range("unknown SRAM region '" + std::string(name) +
                                "'");
    return it->second;
}

const RegionConfig &RegionTable::Region(std::string_view name) const {
    return Region(RegionId(name));
}

const RegionConfig &RegionTable::Region(int region_id) const {
    if (region_id < 0 || static_cast<size_t>(region_id) >= config_.regions.size())
        throw std::out_of_range("invalid SRAM region id");
    return config_.regions[region_id];
}

bool RegionTable::Allows(const RegionConfig &region,
                         Initiator initiator) const {
    return std::find(region.access.begin(), region.access.end(), initiator) !=
           region.access.end();
}

ResolvedRange RegionTable::Resolve(std::string_view name, uint64_t offset,
                                   uint64_t size_bytes, Initiator initiator,
                                   Command command) const {
    const int region_id = RegionId(name);
    const auto &region = Region(region_id);
    if (!Allows(region, initiator))
        throw std::invalid_argument("SRAM " + std::string(ToString(initiator)) +
                                    " cannot " + ToString(command) + " region '" +
                                    region.name + "'");
    const ByteRange relative{offset, size_bytes};
    if (relative.Empty())
        throw std::invalid_argument("SRAM resolve size must be non-zero");
    if (relative.End() > region.size_bytes)
        throw std::out_of_range("SRAM access exceeds region '" + region.name +
                                "'");
    if (region.base_bytes > std::numeric_limits<uint64_t>::max() - offset)
        throw std::overflow_error("resolved SRAM address overflows uint64_t");
    return {region.base_bytes + offset, size_bytes, region_id};
}

ResolvedRange RegionTable::LocateAbsolute(uint64_t address,
                                          uint64_t size_bytes) const {
    const ByteRange requested{address, size_bytes};
    if (requested.Empty())
        throw std::invalid_argument("SRAM access size must be non-zero");
    const uint64_t end = requested.End();
    for (size_t i = 0; i < config_.regions.size(); ++i) {
        const auto &region = config_.regions[i];
        const uint64_t region_end =
            ByteRange{region.base_bytes, region.size_bytes}.End();
        if (address >= region.base_bytes && end <= region_end)
            return {address, size_bytes, static_cast<int>(i)};
    }
    throw std::out_of_range(
        "absolute SRAM range is outside or crosses configured regions");
}

ResolvedRange RegionTable::ResolveAbsolute(uint64_t address,
                                           uint64_t size_bytes,
                                           Initiator initiator,
                                           Command command) const {
    const auto resolved = LocateAbsolute(address, size_bytes);
    const auto &region = Region(resolved.region_id);
    if (!Allows(region, initiator))
        throw std::invalid_argument(
            "SRAM " + std::string(ToString(initiator)) + " cannot " +
            ToString(command) + " region '" + region.name + "'");
    return resolved;
}

Allocation RegionTable::Allocate(std::string_view region_name,
                                 uint64_t size_bytes, std::string label,
                                 AllocationLifetime lifetime) {
    if (size_bytes == 0)
        throw std::invalid_argument("SRAM allocation size must be non-zero");
    const int region_id = RegionId(region_name);
    const auto &region = Region(region_id);
    if (region.allocator != AllocatorKind::kBlock)
        throw std::invalid_argument("fixed SRAM region cannot allocate blocks");
    const uint64_t aligned =
        AlignUp(size_bytes, config_.allocation_alignment_bytes);
    auto &spans = free_spans_.at(region_id);
    for (size_t span_index = 0; span_index < spans.size(); ++span_index) {
        const FreeSpan original = spans[span_index];
        const uint64_t start =
            AlignUp(original.offset, config_.allocation_alignment_bytes);
        const uint64_t padding = start - original.offset;
        if (padding > original.size_bytes ||
            aligned > original.size_bytes - padding)
            continue;
        const uint64_t old_end = original.offset + original.size_bytes;
        const uint64_t alloc_end = start + aligned;
        const FreeSpan before{original.offset, padding};
        const FreeSpan after{alloc_end, old_end - alloc_end};
        spans.erase(spans.begin() + span_index);
        size_t insert_index = span_index;
        if (before.size_bytes != 0) {
            spans.insert(spans.begin() + insert_index, before);
            ++insert_index;
        }
        if (after.size_bytes != 0)
            spans.insert(spans.begin() + insert_index, after);

        Allocation allocation;
        allocation.id = next_allocation_id_++;
        allocation.range = {region.base_bytes + start, aligned, region_id};
        allocation.label = std::move(label);
        allocation.lifetime = lifetime;
        allocations_.emplace(allocation.id, allocation);
        TraceLifecycle("SRAM_region_alloc", "B", allocation.id,
                       allocation.range.address, allocation.range.size_bytes,
                       allocation.label);
        TraceLifecycle("SRAM_region_alloc", "E", allocation.id,
                       allocation.range.address, allocation.range.size_bytes,
                       allocation.label);
        return allocation;
    }
    throw std::bad_alloc();
}

Allocation RegionTable::AllocateAt(
    std::string_view region_name, uint64_t offset_bytes,
    uint64_t size_bytes, std::string label, AllocationLifetime lifetime) {
    if (size_bytes == 0)
        throw std::invalid_argument("SRAM allocation size must be non-zero");
    const int region_id = RegionId(region_name);
    const auto &region = Region(region_id);
    if (region.allocator != AllocatorKind::kBlock)
        throw std::invalid_argument("fixed SRAM region cannot allocate blocks");
    if (offset_bytes % config_.allocation_alignment_bytes != 0)
        throw std::invalid_argument("SRAM fixed allocation offset is not aligned");
    const uint64_t aligned =
        AlignUp(size_bytes, config_.allocation_alignment_bytes);
    const ByteRange wanted{offset_bytes, aligned};
    if (wanted.End() > region.size_bytes)
        throw std::out_of_range("SRAM fixed allocation exceeds region");
    auto &spans = free_spans_.at(region_id);
    for (size_t index = 0; index < spans.size(); ++index) {
        const FreeSpan original = spans[index];
        const uint64_t original_end = original.offset + original.size_bytes;
        if (offset_bytes < original.offset || wanted.End() > original_end)
            continue;
        const FreeSpan before{original.offset,
                              offset_bytes - original.offset};
        const FreeSpan after{wanted.End(), original_end - wanted.End()};
        spans.erase(spans.begin() + index);
        size_t insert = index;
        if (before.size_bytes != 0) {
            spans.insert(spans.begin() + insert, before);
            ++insert;
        }
        if (after.size_bytes != 0)
            spans.insert(spans.begin() + insert, after);
        Allocation allocation;
        allocation.id = next_allocation_id_++;
        allocation.range = {region.base_bytes + offset_bytes, aligned,
                            region_id};
        allocation.label = std::move(label);
        allocation.lifetime = lifetime;
        allocations_.emplace(allocation.id, allocation);
        TraceLifecycle("SRAM_region_alloc", "B", allocation.id,
                       allocation.range.address, allocation.range.size_bytes,
                       allocation.label);
        TraceLifecycle("SRAM_region_alloc", "E", allocation.id,
                       allocation.range.address, allocation.range.size_bytes,
                       allocation.label);
        return allocation;
    }
    throw std::bad_alloc();
}

const Allocation &RegionTable::ResizeAllocation(uint64_t allocation_id,
                                                  uint64_t size_bytes) {
    auto it = allocations_.find(allocation_id);
    if (it == allocations_.end())
        throw std::invalid_argument("unknown SRAM allocation");
    if (size_bytes == 0)
        throw std::invalid_argument("SRAM allocation size must be non-zero");
    Allocation &allocation = it->second;
    const uint64_t aligned =
        AlignUp(size_bytes, config_.allocation_alignment_bytes);
    const uint64_t old_size = allocation.range.size_bytes;
    if (aligned == old_size) return allocation;
    const auto &region = Region(allocation.range.region_id);
    const uint64_t offset = allocation.range.address - region.base_bytes;
    if (offset + aligned > region.size_bytes)
        throw std::out_of_range("resized SRAM allocation exceeds region");
    auto &spans = free_spans_.at(allocation.range.region_id);
    if (aligned < old_size) {
        spans.push_back({offset + aligned, old_size - aligned});
        allocation.range.size_bytes = aligned;
        MergeFreeSpans(allocation.range.region_id);
        return allocation;
    }
    const uint64_t grow_begin = offset + old_size;
    const uint64_t grow_end = offset + aligned;
    for (size_t index = 0; index < spans.size(); ++index) {
        const FreeSpan original = spans[index];
        const uint64_t original_end = original.offset + original.size_bytes;
        if (grow_begin < original.offset || grow_end > original_end) continue;
        const FreeSpan before{original.offset, grow_begin - original.offset};
        const FreeSpan after{grow_end, original_end - grow_end};
        spans.erase(spans.begin() + index);
        size_t insert = index;
        if (before.size_bytes != 0) {
            spans.insert(spans.begin() + insert, before);
            ++insert;
        }
        if (after.size_bytes != 0) spans.insert(spans.begin() + insert, after);
        allocation.range.size_bytes = aligned;
        return allocation;
    }
    throw std::bad_alloc();
}

void RegionTable::MergeFreeSpans(int region_id) {
    auto &spans = free_spans_.at(region_id);
    std::sort(spans.begin(), spans.end(),
              [](const FreeSpan &a, const FreeSpan &b) {
                  return a.offset < b.offset;
              });
    std::vector<FreeSpan> merged;
    for (const auto &span : spans) {
        if (merged.empty() ||
            merged.back().offset + merged.back().size_bytes < span.offset) {
            merged.push_back(span);
        } else {
            const uint64_t end = std::max(
                merged.back().offset + merged.back().size_bytes,
                span.offset + span.size_bytes);
            merged.back().size_bytes = end - merged.back().offset;
        }
    }
    spans = std::move(merged);
}

void RegionTable::Free(uint64_t allocation_id,
                       AllocationLifetime completed_lifetime) {
    const auto it = allocations_.find(allocation_id);
    if (it == allocations_.end())
        throw std::invalid_argument("unknown or already freed SRAM allocation");
    const Allocation allocation = it->second;
    if (static_cast<uint8_t>(allocation.lifetime) >
        static_cast<uint8_t>(completed_lifetime))
        throw std::runtime_error(
            "SRAM allocation lifetime has not ended");
    if (range_busy_probe_ &&
        range_busy_probe_(
            {allocation.range.address, allocation.range.size_bytes}))
        throw std::runtime_error(
            "cannot free SRAM allocation with outstanding accesses");
    const auto &region = Region(allocation.range.region_id);
    free_spans_.at(allocation.range.region_id)
        .push_back({allocation.range.address - region.base_bytes,
                    allocation.range.size_bytes});
    TraceLifecycle("SRAM_region_free", "B", allocation.id,
                   allocation.range.address, allocation.range.size_bytes,
                   allocation.label);
    allocations_.erase(it);
    MergeFreeSpans(allocation.range.region_id);
    TraceLifecycle("SRAM_region_free", "E", allocation.id,
                   allocation.range.address, allocation.range.size_bytes,
                   allocation.label);
}

void RegionTable::TraceLifecycle(
    const std::string &event_name, const char *phase, uint64_t allocation_id,
    uint64_t address, uint64_t size_bytes, const std::string &label) const {
    if (event_engine_ == nullptr) return;
    event_engine_->add_event(
        "SRAM_region_" + std::to_string(core_id_), event_name, phase,
        Trace_event_util("allocation=" + std::to_string(allocation_id) +
                         " address=" + std::to_string(address) +
                         " bytes=" + std::to_string(size_bytes) +
                         " label=" + label),
        SC_ZERO_TIME, static_cast<unsigned>(allocation_id));
}

const Allocation &RegionTable::FindAllocation(uint64_t allocation_id) const {
    const auto it = allocations_.find(allocation_id);
    if (it == allocations_.end())
        throw std::out_of_range("unknown SRAM allocation");
    return it->second;
}

void RegionTable::RenameAllocation(uint64_t allocation_id, std::string label) {
    const auto it = allocations_.find(allocation_id);
    if (it == allocations_.end())
        throw std::out_of_range("unknown SRAM allocation");
    it->second.label = std::move(label);
}

} // namespace sram
