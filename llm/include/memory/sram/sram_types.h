#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sram {

enum class Initiator : uint8_t {
    kCompute = 0,
    kDte,
    kLsu,
    kNocRx,
    kLegacy,
};

enum class Command : uint8_t { kRead = 0, kWrite, kClear };

enum class AllocatorKind : uint8_t { kFixed = 0, kBlock };

enum class AllocationLifetime : uint8_t {
    kTask = 0,
    kLayer,
    kPersistent,
};

struct ByteRange {
    uint64_t address = 0;
    uint64_t size_bytes = 0;

    uint64_t End() const {
        if (size_bytes > std::numeric_limits<uint64_t>::max() - address)
            throw std::overflow_error("SRAM byte range overflows uint64_t");
        return address + size_bytes;
    }

    bool Empty() const { return size_bytes == 0; }
};

inline bool Overlaps(const ByteRange &lhs, const ByteRange &rhs) {
    return !lhs.Empty() && !rhs.Empty() && lhs.address < rhs.End() &&
           rhs.address < lhs.End();
}

inline uint64_t AlignUp(uint64_t value, uint64_t alignment) {
    if (alignment == 0)
        throw std::invalid_argument("SRAM alignment must be non-zero");
    const uint64_t remainder = value % alignment;
    if (remainder == 0)
        return value;
    const uint64_t delta = alignment - remainder;
    if (value > std::numeric_limits<uint64_t>::max() - delta)
        throw std::overflow_error("SRAM aligned value overflows uint64_t");
    return value + delta;
}

struct PortConfig {
    uint32_t count = 1;
    uint32_t width_bits = 128;
};

struct InitiatorPortConfig {
    PortConfig read;
    PortConfig write;
};

struct RegionConfig {
    std::string name;
    uint64_t base_bytes = 0;
    uint64_t size_bytes = 0;
    AllocatorKind allocator = AllocatorKind::kFixed;
    bool spillable = false;
    std::vector<Initiator> access;
};

struct Config {
    uint64_t capacity_bytes = 0;
    uint64_t allocation_alignment_bytes = 1;
    uint32_t bank_count = 1;
    uint64_t bank_interleave_bytes = 16;
    uint32_t read_base_latency_cycles = 1;
    uint32_t write_base_latency_cycles = 1;
    uint32_t queue_depth = 16;
    bool real_data_path = false;
    bool manual_regions = false;
    bool manual_memory_schedule = false;
    uint32_t lsu_queue_depth = 8;
    uint32_t lsu_max_outstanding = 2;
    uint64_t lsu_issue_latency_ns = 0;
    uint32_t dte_memory_workers = 2;
    uint32_t dte_memory_queue_depth = 16;
    InitiatorPortConfig compute;
    InitiatorPortConfig dte;
    InitiatorPortConfig lsu;
    InitiatorPortConfig noc_rx;
    InitiatorPortConfig legacy;
    std::vector<RegionConfig> regions;
};

struct ResolvedRange {
    uint64_t address = 0;
    uint64_t size_bytes = 0;
    int region_id = -1;
};

struct Allocation {
    uint64_t id = 0;
    ResolvedRange range;
    std::string label;
    AllocationLifetime lifetime = AllocationLifetime::kTask;
};

const char *ToString(Initiator initiator);
const char *ToString(Command command);

} // namespace sram
