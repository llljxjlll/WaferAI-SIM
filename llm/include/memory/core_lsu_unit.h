#pragma once

#include "memory/hbm_byte_transport.h"
#include "memory/sram/sram_access_unit.h"
#include <map>
#include <memory>
#include <systemc.h>
#include <sysc/kernel/sc_dynamic_processes.h>

class Event_engine;

namespace sram {

using LsuToken = uint64_t;

enum class LsuDirection : uint8_t {
    kHbmToSram = 0,
    kSramToHbm,
};

enum class LsuTokenStatus : uint8_t {
    kQueued = 0,
    kRunning,
    kComplete,
    kCancelled,
    kFailed,
};

struct LsuDescriptor {
    LsuDirection direction = LsuDirection::kHbmToSram;
    uint64_t hbm_addr = 0;
    uint64_t sram_addr = 0;
    uint64_t size_bytes = 0;
    std::vector<uint8_t> byte_enable;
};

struct LsuTraceRecord {
    LsuToken token = 0;
    LsuDirection direction = LsuDirection::kHbmToSram;
    uint64_t hbm_addr = 0;
    uint64_t sram_addr = 0;
    uint64_t size_bytes = 0;
    sc_time issue_time = SC_ZERO_TIME;
    sc_time run_time = SC_ZERO_TIME;
    sc_time completion_time = SC_ZERO_TIME;
};

struct LsuStats {
    uint64_t issued = 0;
    uint64_t completed = 0;
    uint64_t cancelled = 0;
    uint64_t failed = 0;
    uint64_t hbm_read_bytes = 0;
    uint64_t hbm_write_bytes = 0;
    uint64_t sram_read_bytes = 0;
    uint64_t sram_write_bytes = 0;
    uint64_t issue_latency_ns = 0;
    uint64_t peak_outstanding = 0;
    uint64_t peak_running = 0;
};

class CoreLsuUnit : public sc_module {
  public:
    SC_HAS_PROCESS(CoreLsuUnit);
    CoreLsuUnit(sc_module_name name, RegionTable &regions,
                AccessUnit &sram_access, HbmByteTransport &hbm,
                uint32_t queue_depth = 16,
                uint32_t max_outstanding = 2,
                uint64_t issue_latency_ns = 0,
                Event_engine *event_engine = nullptr,
                int core_id = -1);

    LsuToken Issue(const LsuDescriptor &descriptor,
                   LsuToken requested_token = 0);
    LsuToken IssueLoad(uint64_t hbm_addr, uint64_t sram_addr,
                       uint64_t size_bytes);
    LsuToken IssueStore(uint64_t sram_addr, uint64_t hbm_addr,
                        uint64_t size_bytes,
                        std::vector<uint8_t> byte_enable = {});
    LsuToken IssueLoadRegion(uint64_t hbm_addr, std::string_view region,
                             uint64_t offset, uint64_t size_bytes);
    LsuToken IssueStoreRegion(std::string_view region, uint64_t offset,
                              uint64_t hbm_addr, uint64_t size_bytes,
                              std::vector<uint8_t> byte_enable = {});

    void Wait(LsuToken token);
    bool Poll(LsuToken token) const;
    bool Cancel(LsuToken token);
    void Fence();
    void Load(uint64_t hbm_addr, uint64_t sram_addr, uint64_t size_bytes);
    void Store(uint64_t sram_addr, uint64_t hbm_addr, uint64_t size_bytes,
               std::vector<uint8_t> byte_enable = {});

    size_t OutstandingCount() const { return records_.size(); }
    uint32_t max_outstanding() const { return max_outstanding_; }
    const LsuStats &stats() const { return stats_; }
    const std::vector<LsuTraceRecord> &trace() const { return trace_; }

  private:
    struct TokenRecord {
        LsuToken token = 0;
        LsuDescriptor descriptor;
        LsuTokenStatus status = LsuTokenStatus::kQueued;
        bool cancel_requested = false;
        std::exception_ptr error;
        sc_event done;
        uint64_t hazard_lease = 0;
        sc_time issue_time = SC_ZERO_TIME;
        sc_time run_time = SC_ZERO_TIME;
    };

    void Validate(const LsuDescriptor &descriptor) const;
    std::shared_ptr<TokenRecord> Find(LsuToken token) const;
    void Complete(const std::shared_ptr<TokenRecord> &record,
                  LsuTokenStatus status);
    void TraceStage(const char *stage, const char *phase, LsuToken token,
                    const std::string &extra = {}) const;
    void Worker();

    RegionTable &regions_;
    AccessUnit &sram_access_;
    HbmByteTransport &hbm_;
    sc_fifo<std::shared_ptr<TokenRecord>> queue_;
    std::map<LsuToken, std::shared_ptr<TokenRecord>> records_;
    uint32_t queue_depth_ = 0;
    uint32_t max_outstanding_ = 0;
    uint64_t issue_latency_ns_ = 0;
    uint32_t running_ = 0;
    LsuToken next_token_ = 1;
    LsuStats stats_;
    std::vector<LsuTraceRecord> trace_;
    Event_engine *event_engine_ = nullptr;
    int core_id_ = -1;
};

} // namespace sram
