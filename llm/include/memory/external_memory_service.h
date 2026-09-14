#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace external_memory {

inline constexpr const char *kExternalDmaRequestSchemaVersion =
    "npusim.external_dma_request/v1alpha1";

enum class TransferDirection {
    kExternalToHbm,
    kHbmToExternal,
};

struct ExternalCapacityConfig {
    std::string id;
    std::string owner_ref;
    uint64_t base_address = 0;
    uint64_t capacity_bytes = 0;
};

struct HbmCapacityConfig {
    std::string id;
    uint64_t owner_die_id = 0;
    uint64_t base_address = 0;
    uint64_t capacity_bytes = 0;
};

struct LinkConfig {
    std::string id;
    std::string external_capacity_ref;
    uint64_t ingress_die_id = 0;
    uint64_t bytes_per_cycle = 0;
    uint64_t latency_cycles = 0;
    uint64_t queue_depth = 0;
    uint64_t max_outstanding = 0;
};

struct ConnectionConfig {
    std::string id;
    std::string link_ref;
    std::string hbm_capacity_ref;
    uint64_t target_die_id = 0;
    std::vector<uint64_t> route_die_ids;
    uint64_t route_latency_cycles = 0;
    std::optional<uint64_t> route_bytes_per_cycle;
};

struct FabricConfig {
    std::vector<ExternalCapacityConfig> external_capacities;
    std::vector<HbmCapacityConfig> hbm_capacities;
    std::vector<LinkConfig> links;
    std::vector<ConnectionConfig> connections;
};

struct TransferRequest {
    std::string id;
    std::string connection_ref;
    TransferDirection direction = TransferDirection::kExternalToHbm;
    uint64_t external_address = 0;
    uint64_t hbm_address = 0;
    uint64_t size_bytes = 0;
    uint64_t issue_cycle = 0;
    std::string schema_version = kExternalDmaRequestSchemaVersion;
};

struct TransferCompletion {
    std::string request_ref;
    std::string link_ref;
    TransferDirection direction = TransferDirection::kExternalToHbm;
    uint64_t start_cycle = 0;
    uint64_t completion_cycle = 0;
    uint64_t external_service_cycles = 0;
    uint64_t route_service_cycles = 0;
    uint64_t queue_stall_cycles = 0;
    uint64_t payload_bytes = 0;
};

struct TransferStats {
    uint64_t submitted_requests = 0;
    uint64_t completed_requests = 0;
    uint64_t external_read_bytes = 0;
    uint64_t external_write_bytes = 0;
    uint64_t hbm_read_bytes = 0;
    uint64_t hbm_write_bytes = 0;
    uint64_t link_busy_cycles = 0;
    uint64_t queue_stall_cycles = 0;
    uint64_t max_queue_occupancy = 0;
    uint64_t max_outstanding = 0;
    uint64_t makespan_cycles = 0;
    uint64_t pending_requests = 0;
};

struct TransferReport {
    std::vector<TransferRequest> requests;
    std::vector<TransferCompletion> completions;
    TransferStats stats;
};

void ValidateFabricConfig(const FabricConfig &config);

class SparseMemoryBacking {
public:
    SparseMemoryBacking(uint64_t base_address, uint64_t capacity_bytes);

    void Write(uint64_t address, const std::vector<uint8_t> &payload);
    std::vector<uint8_t> Read(uint64_t address, uint64_t size_bytes) const;
    uint64_t NonzeroByteCount() const;

private:
    void ValidateRange(uint64_t address, uint64_t size_bytes) const;

    uint64_t base_address_ = 0;
    uint64_t capacity_bytes_ = 0;
    std::map<uint64_t, uint8_t> nonzero_bytes_;
};

class ExternalMemoryService {
public:
    explicit ExternalMemoryService(FabricConfig config);

    void SeedExternal(const std::string &capacity_ref, uint64_t address,
                      const std::vector<uint8_t> &payload);
    void SeedHbm(const std::string &capacity_ref, uint64_t address,
                 const std::vector<uint8_t> &payload);
    std::vector<uint8_t> PeekExternal(const std::string &capacity_ref,
                                      uint64_t address,
                                      uint64_t size_bytes) const;
    std::vector<uint8_t> PeekHbm(const std::string &capacity_ref,
                                 uint64_t address,
                                 uint64_t size_bytes) const;

    TransferReport Execute(const std::vector<TransferRequest> &requests);
    const FabricConfig &Config() const { return config_; }

private:
    FabricConfig config_;
    std::map<std::string, SparseMemoryBacking> external_backings_;
    std::map<std::string, SparseMemoryBacking> hbm_backings_;
};

} // namespace external_memory
