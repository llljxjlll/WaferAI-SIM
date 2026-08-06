#pragma once

#include "dte/dte_types.h"
#include "memory/hbm_byte_transport.h"
#include "memory/sram/sram_access_unit.h"
#include <deque>
#include <map>
#include <memory>
#include <systemc.h>
#include <sysc/kernel/sc_dynamic_processes.h>
#include <vector>

class Event_engine;

struct DteMemoryStats {
    uint64_t issued = 0;
    uint64_t completed = 0;
    uint64_t cancelled = 0;
    uint64_t failed = 0;
    uint64_t hbm_read_bytes = 0;
    uint64_t hbm_write_bytes = 0;
    uint64_t sram_read_bytes = 0;
    uint64_t sram_write_bytes = 0;
    uint64_t sram_copy_bytes = 0;
    uint64_t peak_outstanding = 0;
    uint64_t peak_running = 0;
};

struct DteMemoryTraceRecord {
    uint32_t token = 0;
    DteDir direction = DteDir::DRAM_TO_SPM;
    uint64_t source_addr = 0;
    uint64_t destination_addr = 0;
    uint64_t size_bytes = 0;
    sc_time issue_time = SC_ZERO_TIME;
    sc_time run_time = SC_ZERO_TIME;
    sc_time completion_time = SC_ZERO_TIME;
};

class DteMemoryBridge : public sc_module {
  public:
    SC_HAS_PROCESS(DteMemoryBridge);
    DteMemoryBridge(sc_module_name name, sram::RegionTable &regions,
                    sram::AccessUnit &sram_access,
                    sram::HbmByteTransport &hbm, uint32_t queue_depth = 16,
                    uint32_t workers = 2,
                    Event_engine *event_engine = nullptr,
                    int core_id = -1);

    void Validate(uint32_t token, DteDir direction, uint64_t hbm_addr,
                  uint64_t sram_addr, uint64_t size_bytes) const;
    void Reserve(uint32_t token, DteDir direction, uint64_t hbm_addr,
                 uint64_t sram_addr, uint64_t size_bytes);
    void Commit(uint32_t token);
    void Abort(uint32_t token);
    void Issue(uint32_t token, DteDir direction, uint64_t hbm_addr,
               uint64_t sram_addr, uint64_t size_bytes);
    void Wait(uint32_t token);
    bool Poll(uint32_t token) const;
    bool CanCancel(uint32_t token) const;
    void Cancel(uint32_t token);
    void Release(uint32_t token);

    size_t OutstandingCount() const { return records_.size(); }
    const DteMemoryStats &stats() const { return stats_; }
    const std::vector<DteMemoryTraceRecord> &trace() const { return trace_; }

  private:
    enum class Status : uint8_t {
        kReserved = 0,
        kQueued,
        kRunning,
        kComplete,
        kCancelled,
        kFailed,
    };
    struct Record {
        uint32_t token = 0;
        DteDir direction = DteDir::DRAM_TO_SPM;
        uint64_t hbm_addr = 0;
        uint64_t sram_addr = 0;
        uint64_t size_bytes = 0;
        Status status = Status::kReserved;
        std::exception_ptr error;
        sc_event done;
        std::vector<uint64_t> hazard_leases;
        sc_time issue_time = SC_ZERO_TIME;
        sc_time run_time = SC_ZERO_TIME;
    };

    std::shared_ptr<Record> Find(uint32_t token) const;
    void TraceStage(const char *stage, const char *phase, uint32_t token,
                    const std::string &extra = {}) const;
    void Worker();

    sram::RegionTable &regions_;
    sram::AccessUnit &sram_access_;
    sram::HbmByteTransport &hbm_;
    std::map<uint32_t, std::shared_ptr<Record>> records_;
    std::deque<std::shared_ptr<Record>> pending_;
    sc_event pending_changed_;
    uint32_t queue_depth_ = 0;
    uint32_t worker_count_ = 0;
    uint32_t reserved_ = 0;
    uint32_t running_ = 0;
    DteMemoryStats stats_;
    std::vector<DteMemoryTraceRecord> trace_;
    Event_engine *event_engine_ = nullptr;
    int core_id_ = -1;
};
