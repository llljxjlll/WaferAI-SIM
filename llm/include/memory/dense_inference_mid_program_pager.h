#pragma once

#include "memory/external_memory_runtime.h"

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace external_memory {

struct DenseInferencePagerSpan {
    std::string kind;
    std::string state_ref;
    uint64_t source_hbm_address = 0;
    uint64_t external_address = 0;
    uint64_t hbm_address = 0;
    uint64_t size_bytes = 0;
    uint64_t die_id = 0;
    uint64_t runtime_core_id = 0;
    std::string connection_ref;
};

struct DenseInferencePagerEvent {
    uint64_t event_index = 0;
    uint64_t segment_index = 0;
    uint64_t linked_record_index = 0;
    std::string fragment_id;
    uint64_t fragment_record_index = 0;
    std::string kind;
    std::string state_ref;
    std::string state_abi_id;
    uint64_t source_hbm_address = 0;
    uint64_t external_address = 0;
    uint64_t hbm_address = 0;
    uint64_t lsu_address = 0;
    uint64_t lsu_size_bytes = 0;
    uint64_t dma_size_bytes = 0;
    uint64_t die_id = 0;
    uint64_t runtime_core_id = 0;
    std::string connection_ref;
};

// Source-signed Prefill/Decode/Decode runtime for 1x1 and TP4 rectangles.
// Each Die has one time-multiplexed weight slot and four stable KV slots.
class DenseInferenceMidProgramPager {
public:
    DenseInferenceMidProgramPager(
        const sc_core::sc_module_name &name,
        const std::filesystem::path &sidecar,
        std::vector<std::string> manifest_texts,
        std::map<std::pair<uint64_t, uint64_t>, HBMBackend *> backends,
        sc_core::sc_time cycle_time);

    void BeforeLoad(uint64_t runtime_core_id, uint64_t hbm_address,
                    uint64_t size_bytes);
    void AfterLoad(uint64_t runtime_core_id, uint64_t hbm_address,
                   uint64_t size_bytes);
    void AfterStore(uint64_t runtime_core_id, uint64_t hbm_address,
                    uint64_t size_bytes);
    void CompleteSegment(uint64_t segment_index);
    std::string ProbeInitialKvAuthority() const;

    uint64_t CompletedEvents() const { return completed_events_; }
    uint64_t ExternalKvProbes() const { return external_kv_probes_; }
    uint64_t Pending() const;
    uint64_t Pinned() const;
    uint64_t Dirty() const { return dirty_; }
    const RuntimeStats &Stats() const;
    const std::string &KvAuthorityDigest() const { return kv_digest_; }
    uint64_t KvAuthorityBytes() const { return kv_bytes_; }
    const std::string &SourceRef() const { return source_ref_; }

private:
    const DenseInferencePagerEvent &NextEvent(
        uint64_t runtime_core_id, const char *kind,
        uint64_t address, uint64_t size_bytes) const;
    void Transfer(const DenseInferencePagerEvent &event,
                  TransferDirection direction);
    std::vector<uint8_t> ProbeKvPages(uint64_t page_bytes) const;

    std::string source_ref_;
    std::string connection_ref_;
    std::string external_capacity_ref_;
    std::unique_ptr<ExternalMemoryRuntimeBridge> runtime_;
    std::vector<DenseInferencePagerSpan> weights_;
    std::vector<DenseInferencePagerSpan> kv_pages_;
    std::vector<DenseInferencePagerEvent> events_;
    std::map<std::pair<uint64_t, uint64_t>, bool> kv_pinned_;
    std::map<uint64_t, bool> weight_pinned_;
    std::map<uint64_t, bool> awaiting_after_load_;
    std::map<uint64_t, std::vector<uint64_t>> event_indices_by_core_;
    std::map<uint64_t, uint64_t> next_event_by_core_;
    uint64_t dirty_ = 0;
    uint64_t completed_events_ = 0;
    uint64_t external_kv_probes_ = 0;
    uint64_t kv_bytes_ = 0;
    std::string kv_digest_;
    uint64_t kv_page_initial_bytes_ = 128;
    uint64_t kv_page_step_bytes_ = 32;
    sc_core::sc_time cycle_time_;
};

} // namespace external_memory
