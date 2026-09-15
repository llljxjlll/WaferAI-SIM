#pragma once

#include "memory/external_memory_runtime.h"

#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace external_memory {

struct MoeInferencePagerSpan {
    uint64_t die_id = 0;
    std::string kind;
    std::string source_state_ref;
    uint64_t source_hbm_address = 0;
    uint64_t external_address = 0;
    uint64_t hbm_address = 0;
    uint64_t size_bytes = 0;
};

struct MoeInferencePagerEvent {
    uint64_t index = 0;
    uint64_t segment_index = 0;
    uint64_t runtime_core_id = 0;
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
};

// Fixed full-model 1x2/EP2 MoE Prefill+2Decode. One actual shared external
// link services both physical HBM homes. Each core consumes its source-signed
// LSU gates in linked order; SystemC may interleave the two core streams.
class MoeInferenceMidProgramPager {
public:
    MoeInferenceMidProgramPager(
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
    uint64_t KvAuthorityBytes() const { return kv_bytes_; }
    const std::string &KvAuthorityDigest() const { return kv_digest_; }
    const std::string &ParameterAuthorityDigest() const { return parameter_digest_; }
    const std::string &SourceRef() const { return source_ref_; }
    const RuntimeStats &Stats() const;

private:
    const MoeInferencePagerEvent &NextEvent(uint64_t core_id,
                                            uint64_t address,
                                            uint64_t size_bytes) const;
    void Transfer(const MoeInferencePagerEvent &event,
                  TransferDirection direction);
    std::vector<uint8_t> ProbeKvPages(uint64_t page_bytes) const;
    std::vector<uint8_t> ProbeParameters() const;

    std::string source_ref_;
    std::string external_capacity_ref_;
    std::map<uint64_t, std::string> connection_by_die_;
    std::unique_ptr<ExternalMemoryRuntimeBridge> runtime_;
    std::vector<MoeInferencePagerSpan> parameters_;
    std::vector<MoeInferencePagerSpan> kv_pages_;
    std::vector<MoeInferencePagerEvent> events_;
    std::map<uint64_t, std::vector<uint64_t>> event_indices_by_core_;
    std::map<uint64_t, size_t> next_by_core_{{0, 0}, {16, 0}};
    std::map<uint64_t, uint64_t> awaiting_load_;
    std::map<uint64_t, bool> weight_pinned_{{0, false}, {1, false}};
    std::map<uint64_t, bool> kv_pinned_;
    std::set<std::pair<uint64_t, uint64_t>> expert_awaiting_store_;
    uint64_t dirty_ = 0;
    uint64_t completed_events_ = 0;
    uint64_t external_kv_probes_ = 0;
    uint64_t kv_bytes_ = 0;
    std::string kv_digest_;
    std::string parameter_digest_;
    std::string initial_parameter_digest_;
    sc_core::sc_time cycle_time_;
};

} // namespace external_memory
