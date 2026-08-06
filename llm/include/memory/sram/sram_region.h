#pragma once

#include "memory/sram/sram_types.h"
#include <functional>
#include <nlohmann/json.hpp>
#include <string_view>
#include <unordered_map>

class Event_engine;

namespace sram {

Config ParseConfig(const nlohmann::json &memory_json);
void ValidateConfig(const Config &config);

class ConfigRegistry {
  public:
    static ConfigRegistry &Instance();

    void Configure(const nlohmann::json &hardware_json, int total_cores);
    const Config &ForCore(int core_id) const;
    void ResetForTest();

  private:
    Config global_;
    std::unordered_map<int, Config> per_core_;
    bool configured_ = false;
};

class RegionTable {
  public:
    explicit RegionTable(Config config, Event_engine *event_engine = nullptr,
                         int core_id = -1);

    const Config &config() const { return config_; }
    const RegionConfig &Region(std::string_view name) const;
    const RegionConfig &Region(int region_id) const;
    int RegionId(std::string_view name) const;

    ResolvedRange Resolve(std::string_view name, uint64_t offset,
                          uint64_t size_bytes, Initiator initiator,
                          Command command) const;
    ResolvedRange ResolveAbsolute(uint64_t address, uint64_t size_bytes,
                                  Initiator initiator,
                                  Command command) const;
    ResolvedRange LocateAbsolute(uint64_t address,
                                 uint64_t size_bytes) const;

    Allocation Allocate(
        std::string_view region_name, uint64_t size_bytes,
        std::string label = {},
        AllocationLifetime lifetime = AllocationLifetime::kTask);
    Allocation AllocateAt(
        std::string_view region_name, uint64_t offset_bytes,
        uint64_t size_bytes, std::string label = {},
        AllocationLifetime lifetime = AllocationLifetime::kTask);
    const Allocation &ResizeAllocation(uint64_t allocation_id,
                                       uint64_t size_bytes);
    void Free(uint64_t allocation_id,
              AllocationLifetime completed_lifetime =
                  AllocationLifetime::kTask);
    const Allocation &FindAllocation(uint64_t allocation_id) const;
    void RenameAllocation(uint64_t allocation_id, std::string label);
    void SetRangeBusyProbe(std::function<bool(ByteRange)> probe) {
        range_busy_probe_ = std::move(probe);
    }
    void TraceLifecycle(const std::string &event_name, const char *phase,
                        uint64_t allocation_id, uint64_t address,
                        uint64_t size_bytes, const std::string &label) const;

  private:
    struct FreeSpan {
        uint64_t offset = 0;
        uint64_t size_bytes = 0;
    };

    bool Allows(const RegionConfig &region, Initiator initiator) const;
    void MergeFreeSpans(int region_id);

    Config config_;
    std::unordered_map<std::string, int> name_to_id_;
    std::unordered_map<int, std::vector<FreeSpan>> free_spans_;
    std::unordered_map<uint64_t, Allocation> allocations_;
    uint64_t next_allocation_id_ = 1;
    std::function<bool(ByteRange)> range_busy_probe_;
    Event_engine *event_engine_ = nullptr;
    int core_id_ = -1;
};

} // namespace sram
