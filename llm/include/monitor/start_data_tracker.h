#pragma once

#include "common/msg.h"
#include "systemc.h"
#include <cstdint>
#include <map>
#include <string>
#include <tuple>

enum class StartDataStage {
    ENQUEUED = 0,
    INJECTED,
    ROUTER_ACCEPTED,
    DELIVERED,
    CONSUMED,
    COMPLETED,
};

struct StartDataStageCounts {
    uint64_t enqueued = 0;
    uint64_t injected = 0;
    uint64_t router_accepted = 0;
    uint64_t delivered = 0;
    uint64_t consumed = 0;
    uint64_t completed = 0;
};

// Behavioral NoC represents an entire S_DATA payload with one physical
// message. Track the physical lifecycle explicitly so host injection cannot be
// confused with endpoint delivery or completion.
class StartDataTracker {
public:
    static StartDataTracker &Instance();

    void Reset();
    void Record(StartDataStage stage, const Msg &msg);

    const StartDataStageCounts &Totals() const { return totals_; }
    sc_event &CompletedEvent() { return completed_event_; }
    std::string Summary() const;
    std::string OutstandingSummary() const;

private:
    using Key = std::tuple<int, int, int>; // destination, tag, sequence

    StartDataStageCounts totals_;
    std::map<Key, StartDataStageCounts> per_message_;
    sc_event completed_event_;
};

void ResetStartDataTracking();
void RecordStartDataStage(StartDataStage stage, const Msg &msg);
