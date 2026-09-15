#pragma once

#include "memory/external_dma_program.h"
#include "memory/dense_adamw_versioned_role_contract.h"

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace external_memory {

struct DenseAdamwPagerSpan {
    std::string state_ref;
    std::string state_abi_id;
    std::string source_allocation_ref;
    std::string kind;
    uint64_t external_address = 0;
    uint64_t hbm_address = 0;
    uint64_t size_bytes = 0;
};

struct DenseAdamwPagerEvent {
    uint64_t step_index = 0;
    uint64_t linked_record_index = 0;
    std::string kind;
    std::string state_ref;
    uint64_t external_address = 0;
    uint64_t hbm_address = 0;
    uint64_t size_bytes = 0;
};

class DenseAdamwMidProgramPager {
public:
    DenseAdamwMidProgramPager(
        const sc_core::sc_module_name &name,
        const std::filesystem::path &sidecar,
        std::vector<std::string> manifest_texts,
        std::map<HbmEndpoint, HBMBackend *> backends,
        sc_core::sc_time cycle_time);

    void BeforeLoad(uint64_t hbm_address, uint64_t size_bytes);
    void AfterStore(uint64_t hbm_address, uint64_t size_bytes);
    std::string ProbeInitialAuthority() const;
    void CompleteStep(uint64_t index);
    uint64_t CompletedEvents() const { return next_event_; }
    uint64_t ExternalAuthorityProbes() const { return external_probes_; }
    const std::string &AuthorityDigest() const { return authority_digest_; }
    uint64_t Pending() const;
    const RuntimeStats &Stats() const;
    const std::string &SourceRef() const { return source_ref_; }

private:
    void Transfer(const DenseAdamwPagerEvent &event,
                  TransferDirection direction);
    const DenseAdamwPagerEvent &NextEvent(
        const char *kind, uint64_t address, uint64_t size_bytes) const;

    std::string source_ref_;
    std::string connection_ref_;
    std::string external_capacity_ref_;
    std::unique_ptr<ExternalMemoryRuntimeBridge> runtime_;
    std::unique_ptr<DenseAdamwVersionedRoleContract> role_contract_;
    std::vector<DenseAdamwPagerSpan> spans_;
    std::vector<DenseAdamwPagerEvent> events_;
    std::map<std::string, bool> resident_;
    std::string authority_digest_;
    std::vector<DmaExternalProbe> probes_;
    sc_core::sc_time cycle_time_;
    uint64_t next_event_ = 0;
    uint64_t external_probes_ = 0;
};

} // namespace external_memory
