#pragma once

#include "memory/sram/sram_access_unit.h"

class Event_engine;

namespace sram {

struct ComputeTraceRecord {
    uint64_t tile_id = 0;
    ResolvedRange input;
    uint64_t compute_cycles = 0;
    sc_time issue_time = SC_ZERO_TIME;
    sc_time compute_begin = SC_ZERO_TIME;
    sc_time completion_time = SC_ZERO_TIME;
};

struct ComputeTimelineStats {
    uint64_t tiles = 0;
    uint64_t compute_cycles = 0;
    uint64_t sram_read_bytes = 0;
};

class ComputeTimeline {
  public:
    explicit ComputeTimeline(AccessUnit &access,
                             Event_engine *event_engine = nullptr,
                             int core_id = -1)
        : access_(access), event_engine_(event_engine), core_id_(core_id) {}

    std::vector<uint8_t> RunTile(const ResolvedRange &input,
                                 uint64_t compute_cycles);
    void RunCycles(uint64_t compute_cycles);
    const ComputeTimelineStats &stats() const { return stats_; }
    const std::vector<ComputeTraceRecord> &trace() const { return trace_; }

  private:
    AccessUnit &access_;
    ComputeTimelineStats stats_;
    uint64_t next_tile_id_ = 1;
    std::vector<ComputeTraceRecord> trace_;
    Event_engine *event_engine_ = nullptr;
    int core_id_ = -1;
};

} // namespace sram
