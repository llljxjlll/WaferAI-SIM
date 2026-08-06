#include "memory/sram/compute_timeline.h"

#include "macros/macros.h"
#include "trace/Event_engine.h"

namespace sram {

std::vector<uint8_t> ComputeTimeline::RunTile(const ResolvedRange &input,
                                               uint64_t compute_cycles) {
    ComputeTraceRecord trace;
    trace.tile_id = next_tile_id_++;
    trace.input = input;
    trace.compute_cycles = compute_cycles;
    trace.issue_time = sc_time_stamp();
    const auto &region = access_.regions().Region(input.region_id);
    const uint64_t region_offset = input.address - region.base_bytes;
    if (event_engine_ != nullptr)
        event_engine_->add_event(
            "Compute_" + std::to_string(core_id_), "Compute_tile", "B",
            Trace_event_util("tile=" + std::to_string(trace.tile_id) +
                             " region=" + std::to_string(input.region_id) +
                             " offset=" + std::to_string(region_offset) +
                             " bytes=" + std::to_string(input.size_bytes)),
            SC_ZERO_TIME, static_cast<unsigned>(trace.tile_id));

    Request request;
    request.initiator = Initiator::kCompute;
    request.command = Command::kRead;
    request.address = input.address;
    request.size_bytes = input.size_bytes;
    auto payload = access_.Access(request).payload;
    stats_.sram_read_bytes += input.size_bytes;
    trace.compute_begin = sc_time_stamp();
    RunCycles(compute_cycles);
    ++stats_.tiles;
    trace.completion_time = sc_time_stamp();
    trace_.push_back(trace);
    if (event_engine_ != nullptr)
        event_engine_->add_event(
            "Compute_" + std::to_string(core_id_), "Compute_tile", "E",
            Trace_event_util("tile=" + std::to_string(trace.tile_id) +
                             " region=" + std::to_string(input.region_id) +
                             " offset=" + std::to_string(region_offset) +
                             " bytes=" + std::to_string(input.size_bytes)),
            SC_ZERO_TIME, static_cast<unsigned>(trace.tile_id));
    return payload;
}

void ComputeTimeline::RunCycles(uint64_t compute_cycles) {
    if (compute_cycles != 0)
        wait(sc_time(static_cast<double>(compute_cycles * CYCLE), SC_NS));
    stats_.compute_cycles += compute_cycles;
}

} // namespace sram
