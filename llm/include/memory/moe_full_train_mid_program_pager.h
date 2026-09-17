#pragma once

#include "memory/external_dma_program.h"
#include "memory/external_memory_runtime.h"

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace external_memory {

struct MoeFullTrainPagerSpan {
    std::string state_ref;
    std::string state_abi_id;
    std::string source_allocation_ref;
    std::string group;
    uint64_t source_hbm_address = 0;
    uint64_t external_address = 0;
    uint64_t hbm_address = 0;
    uint64_t size_bytes = 0;
};

struct MoeFullTrainPagerEvent {
    uint64_t index = 0;
    uint64_t step_index = 0;
    uint64_t linked_record_index = 0;
    std::string fragment_id;
    uint64_t fragment_record_index = 0;
    std::string kind;
    std::string state_ref;
    std::string state_abi_id;
    uint64_t source_hbm_address = 0;
    uint64_t external_address = 0;
    uint64_t hbm_address = 0;
    uint64_t size_bytes = 0;
};

// Real LSU hooks serialize each 128B weight-page reuse. Route traces remain
// resident HBM ProgramIO seeds, and optimizer state is external authority.
class MoeFullTrainMidProgramPager {
public:
    MoeFullTrainMidProgramPager(
        const sc_core::sc_module_name &name,
        const std::filesystem::path &sidecar,
        std::vector<std::string> manifest_texts,
        std::map<HbmEndpoint, HBMBackend *> backends,
        sc_core::sc_time cycle_time);

    void BeforeLoad(uint64_t core_id, uint64_t hbm_address,
                    uint64_t size_bytes);
    void AfterLoad(uint64_t core_id, uint64_t hbm_address,
                   uint64_t size_bytes);
    void AfterStore(uint64_t core_id, uint64_t hbm_address,
                    uint64_t size_bytes);
    void CompleteStep(uint64_t step);
    uint64_t Pending() const;
    uint64_t CompletedEvents() const { return completed_events_; }
    uint64_t RouteLoads() const { return route_loads_; }
    const RuntimeStats &Stats() const;
    const std::string &SourceRef() const { return source_ref_; }
    const std::string &AuthorityDigest() const { return authority_digest_; }

private:
    void Transfer(const MoeFullTrainPagerEvent &event,
                  TransferDirection direction);
    std::vector<uint8_t> ProbeParameters() const;
    const MoeFullTrainPagerEvent &NextEvent(uint64_t core_id,
                                            uint64_t address,
                                            uint64_t size) const;

    std::string source_ref_;
    std::string external_capacity_ref_;
    std::string connection_ref_;
    std::unique_ptr<ExternalMemoryRuntimeBridge> runtime_;
    std::vector<MoeFullTrainPagerSpan> spans_;
    std::vector<MoeFullTrainPagerEvent> events_;
    std::optional<uint64_t> awaiting_load_;
    std::set<std::string> loaded_states_;
    sc_core::sc_time cycle_time_;
    uint64_t next_event_ = 0;
    uint64_t completed_events_ = 0;
    uint64_t route_loads_ = 0;
    std::string authority_digest_;
};

} // namespace external_memory
