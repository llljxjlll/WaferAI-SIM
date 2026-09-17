#pragma once

#include "memory/external_memory_runtime.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace external_memory {

struct FullDenseAdamwStateSpan {
    std::string state_ref;
    std::string source_abi_id;
    std::string paged_abi_id;
    std::string kind;
    uint64_t external_address = 0;
    uint64_t hbm_address = 0;
    uint64_t size_bytes = 0;
    std::vector<uint8_t> seed;
};

struct FullDenseAdamwDmaEvent {
    uint64_t index = 0;
    uint64_t program_record_index = 0;
    uint64_t step = 0;
    std::string kind;
    std::string state_ref;
    uint64_t external_address = 0;
    uint64_t hbm_address = 0;
    uint64_t size_bytes = 0;
};

// One physical core executes the complete 520-fragment, two-step program.
// Each LSU operation completes before the following record: the HBM slot is
// therefore time-aliased only between fully completed external DMA requests.
class FullDenseAdamwPager {
public:
    FullDenseAdamwPager(const sc_core::sc_module_name &name,
                        const std::filesystem::path &sidecar,
                        const std::vector<uint8_t> &program_bytes,
                        HBMBackend *hbm,
                        sc_core::sc_time cycle_time);

    void BeforeLoad(uint64_t hbm_address, uint64_t size_bytes);
    void AfterStore(uint64_t hbm_address, uint64_t size_bytes);
    void RequireComplete() const;
    uint64_t CompletedEvents() const { return next_event_; }
    uint64_t StateCount() const { return spans_.size(); }
    uint64_t HbmCapacity() const { return hbm_capacity_; }
    uint64_t ExternalCapacity() const { return external_capacity_; }
    uint64_t SlotBytes() const { return slot_bytes_; }
    uint64_t Outstanding() const;
    const RuntimeStats &Stats() const;
    const std::string &SourceManifestDigest() const { return source_digest_; }
    const std::string &PagedManifestDigest() const { return paged_digest_; }
    const std::string &InitialAuthorityDigest() const { return initial_digest_; }
    const std::string &FinalAuthorityDigest() const { return final_digest_; }

private:
    const FullDenseAdamwDmaEvent &Next(const char *kind,
                                       uint64_t address,
                                       uint64_t size) const;
    void Transfer(const FullDenseAdamwDmaEvent &event,
                  TransferDirection direction);
    std::string ProbeAuthority(uint64_t version);

    std::unique_ptr<ExternalMemoryRuntimeBridge> runtime_;
    std::vector<FullDenseAdamwStateSpan> spans_;
    std::vector<FullDenseAdamwDmaEvent> events_;
    std::string source_digest_;
    std::string paged_digest_;
    std::string initial_digest_;
    std::string final_digest_;
    std::string external_ref_;
    std::string connection_ref_;
    sc_core::sc_time cycle_time_;
    uint64_t hbm_capacity_ = 0;
    uint64_t external_capacity_ = 0;
    uint64_t slot_bytes_ = 0;
    uint64_t next_event_ = 0;
};

} // namespace external_memory
