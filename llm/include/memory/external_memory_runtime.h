#pragma once

#include "memory/external_memory_service.h"
#include "memory/hbm_backend.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <systemc>
#include <vector>

namespace external_memory {

struct RuntimeCompletion {
    std::string request_ref;
    int status = 0;
    std::string error;
    sc_core::sc_time submitted_at = sc_core::SC_ZERO_TIME;
    sc_core::sc_time started_at = sc_core::SC_ZERO_TIME;
    sc_core::sc_time completed_at = sc_core::SC_ZERO_TIME;
    sc_core::sc_time external_service_time = sc_core::SC_ZERO_TIME;
    sc_core::sc_time hbm_service_time = sc_core::SC_ZERO_TIME;
    uint64_t payload_bytes = 0;
};

struct RuntimeStats {
    uint64_t submitted_requests = 0;
    uint64_t completed_requests = 0;
    uint64_t failed_requests = 0;
    uint64_t external_read_bytes = 0;
    uint64_t external_write_bytes = 0;
    uint64_t hbm_read_bytes = 0;
    uint64_t hbm_write_bytes = 0;
    uint64_t max_outstanding = 0;
    sc_core::sc_time external_service_time = sc_core::SC_ZERO_TIME;
    sc_core::sc_time hbm_service_time = sc_core::SC_ZERO_TIME;
    sc_core::sc_time queue_stall_time = sc_core::SC_ZERO_TIME;
};

class ExternalMemoryRuntimeBridge : public sc_core::sc_module {
public:
    ExternalMemoryRuntimeBridge(
        const sc_core::sc_module_name &name, FabricConfig config,
        std::map<std::string, HBMBackend *> hbm_backends,
        sc_core::sc_time cycle_time);
    ~ExternalMemoryRuntimeBridge() override;

    void SeedExternal(const std::string &capacity_ref, uint64_t address,
                      const std::vector<uint8_t> &payload);
    std::vector<uint8_t> ProbeExternal(const std::string &capacity_ref,
                                       uint64_t address,
                                       uint64_t size_bytes) const;

    void Submit(const TransferRequest &request);
    std::optional<RuntimeCompletion> Poll(
        const std::string &request_ref) const;
    RuntimeCompletion Wait(const std::string &request_ref);
    const sc_core::sc_event &CompletionEvent() const;
    const RuntimeStats &Stats() const;
    uint64_t Outstanding() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace external_memory
