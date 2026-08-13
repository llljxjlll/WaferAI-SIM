#pragma once

#include "memory/sram/sram_region.h"
#include "memory/sram/sram_storage.h"
#include <array>
#include <deque>
#include <map>
#include <memory>
#include <systemc.h>

class Event_engine;

namespace sram {

struct Request {
    Initiator initiator = Initiator::kCompute;
    Command command = Command::kRead;
    uint64_t address = 0;
    uint64_t size_bytes = 0;
    std::vector<uint8_t> payload;
    std::vector<uint8_t> byte_enable;
    uint64_t hazard_lease = 0;
};

struct Response {
    std::vector<uint8_t> payload;
    sc_time issue_time = SC_ZERO_TIME;
    sc_time service_begin = SC_ZERO_TIME;
    sc_time completion_time = SC_ZERO_TIME;
};

struct AccessCounter {
    uint64_t requests = 0;
    uint64_t bytes = 0;
    uint64_t queue_wait_cycles = 0;
    uint64_t service_cycles = 0;
    uint64_t hazard_stalls = 0;
    uint64_t bank_stalls = 0;
    uint64_t port_stalls = 0;
};

struct ResourceCounter {
    uint64_t beats = 0;
    uint64_t bytes = 0;
    uint64_t service_cycles = 0;
    uint64_t stall_cycles = 0;
};

struct SramTraceRecord {
    uint64_t request_id = 0;
    Initiator initiator = Initiator::kCompute;
    Command command = Command::kRead;
    uint64_t address = 0;
    uint64_t size_bytes = 0;
    sc_time issue_time = SC_ZERO_TIME;
    sc_time service_begin = SC_ZERO_TIME;
    sc_time completion_time = SC_ZERO_TIME;
};

struct AccessStats {
    std::array<std::array<AccessCounter, 3>, 5> by_initiator_command{};
    std::vector<uint64_t> bank_requests;
    std::vector<ResourceCounter> banks;
    std::array<std::array<ResourceCounter, 3>, 5> ports{};
    uint64_t peak_queued = 0;
};

class AccessUnit : public sc_module {
  public:
    SC_HAS_PROCESS(AccessUnit);
    AccessUnit(sc_module_name name, RegionTable &regions, Storage &storage,
               Event_engine *event_engine = nullptr, int core_id = -1);
    ~AccessUnit() override;

    Response Access(const Request &request);
    uint64_t DeclareRangeLease(Initiator initiator, Command command,
                               uint64_t address, uint64_t size_bytes,
                               uint64_t group_id = 0);
    void WaitRangeLease(uint64_t lease_id);
    void ReleaseRangeLease(uint64_t lease_id);
    uint32_t BankForAddress(uint64_t address) const;
    std::vector<uint32_t> BanksForRange(uint64_t address,
                                        uint64_t size_bytes) const;
    bool IsRangeBusy(ByteRange range) const;

    // Test-only direct storage access. These calls reject while SystemC is
    // running and do not consume ports, timing, statistics, or trace state.
    void DebugSeed(uint64_t address,
                   const std::vector<uint8_t> &payload);
    DebugSnapshot DebugPeek(uint64_t address,
                            uint64_t size_bytes) const;
    void DebugRestore(uint64_t address,
                      const DebugSnapshot &snapshot);

    const AccessStats &stats() const { return stats_; }
    const RegionTable &regions() const { return regions_; }
    const std::vector<SramTraceRecord> &trace() const { return trace_; }
    uint64_t outstanding() const { return admitted_; }

  private:
    struct ActiveAccess {
        Request request;
        sc_event done;
        uint64_t lease_id = 0;
        uint64_t group_id = 0;
        bool hazard_only = false;
        std::vector<std::shared_ptr<ActiveAccess>> dependencies;
    };

    struct Beat {
        uint64_t address = 0;
        uint64_t size_bytes = 0;
        uint32_t bank = 0;
    };

    struct BeatWaiter {
        Initiator initiator = Initiator::kCompute;
        Command command = Command::kRead;
        uint32_t bank = 0;
        uint64_t size_bytes = 0;
        uint64_t sequence = 0;
        uint64_t service_cycles = 1;
        sc_time enqueue_time = SC_ZERO_TIME;
        bool granted = false;
        bool saw_bank_block = false;
        bool saw_port_block = false;
        sc_event granted_event;
    };

    const InitiatorPortConfig &Ports(Initiator initiator) const;
    const PortConfig &Port(Initiator initiator, Command command) const;
    bool HasHazard(const Request &request, const ActiveAccess &active) const;
    std::vector<Beat> BuildBeats(const Request &request) const;
    std::shared_ptr<BeatWaiter> AcquireBeat(const Beat &beat,
                                            const Request &request,
                                            uint64_t trace_request_id);
    void TraceStage(const char *stage, const char *phase,
                    uint64_t trace_request_id,
                    const std::string &extra = {}) const;
    void ReleaseBeat(const std::shared_ptr<BeatWaiter> &waiter);
    void TryGrantBeats();
    void RemoveActive(const std::shared_ptr<ActiveAccess> &active);

    RegionTable &regions_;
    Storage &storage_;
    std::vector<std::shared_ptr<ActiveAccess>> active_;
    std::deque<std::shared_ptr<BeatWaiter>> pending_beats_;
    std::vector<bool> bank_busy_;
    std::array<std::array<uint32_t, 3>, 5> port_users_{};
    sc_event capacity_available_;
    uint64_t admitted_ = 0;
    uint64_t next_lease_id_ = 1;
    uint64_t next_beat_sequence_ = 1;
    uint64_t next_trace_id_ = 1;
    size_t rr_next_initiator_ = 0;
    std::map<uint64_t, std::shared_ptr<ActiveAccess>> leases_;
    AccessStats stats_;
    std::vector<SramTraceRecord> trace_;
    Event_engine *event_engine_ = nullptr;
    int core_id_ = -1;
};

} // namespace sram
