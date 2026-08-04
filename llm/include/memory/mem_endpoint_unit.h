#pragma once

#include "memory/hbm_backend.h"
#include "memory/hbm_mem_wire.h"

#include <deque>
#include <memory>
#include <systemc>
#include <vector>

struct MemEndpointStats {
    uint64_t requests = 0;
    uint64_t bytes = 0;
    uint64_t queue_stalls = 0;
    uint64_t completed = 0;
    sc_core::sc_time queue_wait = sc_core::SC_ZERO_TIME;
    sc_core::sc_time response_time = sc_core::SC_ZERO_TIME;
};

class MemEndpointUnit : public sc_core::sc_module {
public:
    int stack_id;
    int channel_id;

    SC_HAS_PROCESS(MemEndpointUnit);
    MemEndpointUnit(const sc_core::sc_module_name &n, int stack_id,
                    int channel_id, HBMBackend &backend,
                    int queue_depth = 8, int max_outstanding = 8);

    MemMsg HandleRequest(const MemMsg &req);
    int QueueDepthNow() const { return (int)queue_.size() + in_flight_; }
    int InFlightCount() const { return in_flight_; }
    sc_core::sc_time BusyTime() const { return backend_.Stats().service_time; }
    const MemEndpointStats &Stats() const { return stats_; }

private:
    struct PendingEntry {
        MemMsg req;
        MemMsg resp;
        sc_core::sc_event done;
        sc_core::sc_time arrival = sc_core::SC_ZERO_TIME;
        sc_core::sc_time issued = sc_core::SC_ZERO_TIME;
        sc_core::sc_time due = sc_core::SC_ZERO_TIME;
        sc_core::sc_time backend_service = sc_core::SC_ZERO_TIME;
        int backend_status = 0;
        std::string backend_error;
        std::shared_ptr<HBMBackendTransaction> backend_tx;
    };

    void DispatchLoop();
    void CompletionLoop();
    void ScheduleCompletion(const std::shared_ptr<PendingEntry> &entry,
                            sc_core::sc_time delay,
                            sc_core::sc_time service_time, int status,
                            const std::string &error);

    HBMBackend &backend_;
    int queue_depth_;
    int max_outstanding_;
    int in_flight_ = 0;
    std::deque<std::shared_ptr<PendingEntry>> queue_;
    std::vector<std::shared_ptr<PendingEntry>> scheduled_;
    sc_core::sc_event dispatch_event_;
    sc_core::sc_event space_available_event_;
    sc_core::sc_event_queue completion_events_;
    MemEndpointStats stats_;
};
