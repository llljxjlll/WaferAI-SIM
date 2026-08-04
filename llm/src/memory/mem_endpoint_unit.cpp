#include "memory/mem_endpoint_unit.h"

#include <algorithm>
#include <stdexcept>

using namespace sc_core;

MemEndpointUnit::MemEndpointUnit(const sc_module_name &n, int stack_id,
                                 int channel_id, HBMBackend &backend,
                                 int queue_depth, int max_outstanding)
    : sc_module(n), stack_id(stack_id), channel_id(channel_id),
      backend_(backend), queue_depth_(queue_depth),
      max_outstanding_(max_outstanding) {
    if (queue_depth_ <= 0 || max_outstanding_ <= 0)
        throw std::runtime_error(
            "MemEndpointUnit: queue depth/max outstanding must be > 0");
    SC_THREAD(DispatchLoop);
    SC_THREAD(CompletionLoop);
}

MemMsg MemEndpointUnit::HandleRequest(const MemMsg &req) {
    if (req.message_type != MemMessageType::kRequest || req.txid < 0 ||
        req.length_bytes <= 0 || req.stack_id != stack_id ||
        req.channel_id != channel_id)
        throw std::runtime_error("MemEndpointUnit: malformed/misrouted request");
    if (req.command == MemCommand::kWrite &&
        (int)req.payload.size() != req.length_bytes)
        throw std::runtime_error("MemEndpointUnit: write payload mismatch");
    if (!req.byte_enable.empty() &&
        (int)req.byte_enable.size() != req.length_bytes)
        throw std::runtime_error("MemEndpointUnit: byte-enable mismatch");

    while ((int)queue_.size() >= queue_depth_) {
        stats_.queue_stalls++;
        wait(space_available_event_);
    }
    auto entry = std::make_shared<PendingEntry>();
    entry->req = req;
    entry->arrival = sc_time_stamp();
    queue_.push_back(entry);
    stats_.requests++;
    stats_.bytes += req.length_bytes;
    dispatch_event_.notify(SC_ZERO_TIME);
    wait(entry->done);
    return entry->resp;
}

void MemEndpointUnit::ScheduleCompletion(
    const std::shared_ptr<PendingEntry> &entry, sc_time delay,
    sc_time service_time, int status, const std::string &error) {
    entry->due = sc_time_stamp() + delay;
    entry->backend_service = service_time;
    entry->backend_status = status;
    entry->backend_error = error;
    scheduled_.push_back(entry);
    completion_events_.notify(delay);
}

void MemEndpointUnit::DispatchLoop() {
    while (true) {
        while (!queue_.empty() && in_flight_ < max_outstanding_) {
            auto entry = queue_.front();
            queue_.pop_front();
            space_available_event_.notify(SC_ZERO_TIME);
            entry->issued = sc_time_stamp();
            stats_.queue_wait += entry->issued - entry->arrival;
            in_flight_++;

            auto tx = std::make_shared<HBMBackendTransaction>();
            tx->command = entry->req.command;
            tx->address = entry->req.address;
            tx->payload = entry->req.command == MemCommand::kWrite
                              ? entry->req.payload
                              : std::vector<uint8_t>(entry->req.length_bytes);
            tx->byte_enable = entry->req.byte_enable;
            tx->submitted = sc_time_stamp();
            tx->complete = [this, entry](sc_time delay, sc_time service,
                                         int status,
                                         const std::string &error) {
                ScheduleCompletion(entry, delay, service, status, error);
            };
            entry->backend_tx = tx;
            backend_.Submit(tx);
        }
        wait(dispatch_event_);
    }
}

void MemEndpointUnit::CompletionLoop() {
    while (true) {
        wait(completion_events_.default_event());
        sc_time now = sc_time_stamp();
        std::vector<std::shared_ptr<PendingEntry>> ready;
        auto it = scheduled_.begin();
        while (it != scheduled_.end()) {
            if ((*it)->due <= now) {
                ready.push_back(*it);
                it = scheduled_.erase(it);
            } else {
                ++it;
            }
        }
        for (auto &entry : ready) {
            const MemMsg &req = entry->req;
            MemMsg &resp = entry->resp;
            resp.txid = req.txid;
            resp.message_type = MemMessageType::kResponse;
            resp.source_core = req.source_core;
            resp.home_die = req.home_die;
            resp.stack_id = req.stack_id;
            resp.channel_id = req.channel_id;
            resp.address = req.address;
            resp.command = req.command;
            resp.status = entry->backend_status;
            if (req.command == MemCommand::kRead && resp.status == 0) {
                resp.payload = entry->backend_tx->payload;
                resp.byte_enable.assign(resp.payload.size(), 0xff);
                resp.length_bytes = req.length_bytes;
            } else {
                resp.length_bytes = 0;
            }
            in_flight_--;
            stats_.completed++;
            stats_.response_time += now - entry->arrival;
            dispatch_event_.notify(SC_ZERO_TIME);
            entry->done.notify(SC_ZERO_TIME);
        }
    }
}
