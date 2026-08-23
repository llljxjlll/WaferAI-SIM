#include "workercore/moe_swizzle_runtime_capture.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace {

struct TickInterval {
    uint64_t start = 0;
    uint64_t end = 0;
};

uint64_t CheckedAdd(uint64_t lhs, uint64_t rhs, const char *what) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
        throw std::overflow_error(what);
    return lhs + rhs;
}

std::vector<TickInterval> UnionIntervals(std::vector<TickInterval> values) {
    values.erase(
        std::remove_if(values.begin(), values.end(),
                       [](const TickInterval &value) {
                           return value.start == value.end;
                       }),
        values.end());
    std::sort(values.begin(), values.end(),
              [](const TickInterval &lhs, const TickInterval &rhs) {
                  return std::tie(lhs.start, lhs.end) <
                         std::tie(rhs.start, rhs.end);
              });
    std::vector<TickInterval> result;
    for (const TickInterval &value : values) {
        if (value.start > value.end)
            throw std::invalid_argument("runtime interval is reversed");
        if (result.empty() || value.start > result.back().end) {
            result.push_back(value);
        } else {
            result.back().end = std::max(result.back().end, value.end);
        }
    }
    return result;
}

std::vector<TickInterval> IntersectIntervals(
    const std::vector<TickInterval> &lhs,
    const std::vector<TickInterval> &rhs) {
    std::vector<TickInterval> result;
    size_t left = 0;
    size_t right = 0;
    while (left < lhs.size() && right < rhs.size()) {
        const uint64_t start = std::max(lhs[left].start, rhs[right].start);
        const uint64_t end = std::min(lhs[left].end, rhs[right].end);
        if (start < end) result.push_back({start, end});
        if (lhs[left].end < rhs[right].end)
            ++left;
        else
            ++right;
    }
    return result;
}

uint64_t DurationCycles(const std::vector<TickInterval> &values,
                        uint64_t ticks_per_cycle) {
    uint64_t ticks = 0;
    for (const TickInterval &value : values)
        ticks = CheckedAdd(ticks, value.end - value.start,
                           "runtime interval duration overflows u64");
    return ticks / ticks_per_cycle;
}

bool IsCompute(MoeSwizzleRuntimeIntervalKind kind) {
    return kind == MoeSwizzleRuntimeIntervalKind::MATMUL ||
           kind == MoeSwizzleRuntimeIntervalKind::LOCAL_REDUCE ||
           kind == MoeSwizzleRuntimeIntervalKind::SWIGLU_GROUP;
}

bool IsSha256(const std::string &value) {
    return value.size() == 64 &&
           std::all_of(value.begin(), value.end(), [](unsigned char ch) {
               return std::isdigit(ch) || (ch >= 'a' && ch <= 'f');
           });
}

const char *CalibrationKindName(MoeSwizzleCalibrationKind kind) {
    switch (kind) {
    case MoeSwizzleCalibrationKind::GROUP_GEMM: return "group_gemm";
    case MoeSwizzleCalibrationKind::SWIGLU_GROUP: return "swiglu_group";
    case MoeSwizzleCalibrationKind::DTE_LAUNCH: return "dte_launch";
    case MoeSwizzleCalibrationKind::DTE_SYNC: return "dte_sync";
    case MoeSwizzleCalibrationKind::DTE_HOP: return "dte_hop";
    case MoeSwizzleCalibrationKind::SESSION_OPEN: return "session_open";
    case MoeSwizzleCalibrationKind::SESSION_RETIRE: return "session_retire";
    case MoeSwizzleCalibrationKind::LOCAL_COPY: return "local_copy";
    case MoeSwizzleCalibrationKind::SRAM_ALLOC: return "sram_alloc";
    case MoeSwizzleCalibrationKind::SRAM_BIND: return "sram_bind";
    case MoeSwizzleCalibrationKind::SRAM_FREE: return "sram_free";
    case MoeSwizzleCalibrationKind::EVENT_SET: return "event_set";
    case MoeSwizzleCalibrationKind::EVENT_WAIT: return "event_wait";
    case MoeSwizzleCalibrationKind::TERMINAL_DONE: return "terminal_done";
    }
    throw std::invalid_argument("unknown MoE Swizzle calibration kind");
}

MoeSwizzleRuntimeIntervalKind CalibrationIntervalKind(
    MoeSwizzleCalibrationKind kind) {
    switch (kind) {
    case MoeSwizzleCalibrationKind::GROUP_GEMM:
        return MoeSwizzleRuntimeIntervalKind::MATMUL;
    case MoeSwizzleCalibrationKind::SWIGLU_GROUP:
        return MoeSwizzleRuntimeIntervalKind::SWIGLU_GROUP;
    case MoeSwizzleCalibrationKind::DTE_LAUNCH:
        return MoeSwizzleRuntimeIntervalKind::DTE_LAUNCH;
    case MoeSwizzleCalibrationKind::DTE_SYNC:
        return MoeSwizzleRuntimeIntervalKind::DTE_SYNC;
    case MoeSwizzleCalibrationKind::SESSION_OPEN:
        return MoeSwizzleRuntimeIntervalKind::SESSION_OPEN;
    case MoeSwizzleCalibrationKind::SESSION_RETIRE:
        return MoeSwizzleRuntimeIntervalKind::SESSION_RETIRE;
    case MoeSwizzleCalibrationKind::LOCAL_COPY:
        return MoeSwizzleRuntimeIntervalKind::LOCAL_COPY_FIXED;
    case MoeSwizzleCalibrationKind::SRAM_ALLOC:
        return MoeSwizzleRuntimeIntervalKind::SRAM_ALLOC_FIXED;
    case MoeSwizzleCalibrationKind::SRAM_BIND:
        return MoeSwizzleRuntimeIntervalKind::SRAM_BIND;
    case MoeSwizzleCalibrationKind::SRAM_FREE:
        return MoeSwizzleRuntimeIntervalKind::SRAM_FREE_FIXED;
    case MoeSwizzleCalibrationKind::EVENT_SET:
        return MoeSwizzleRuntimeIntervalKind::EVENT_SET_FIXED;
    case MoeSwizzleCalibrationKind::EVENT_WAIT:
        return MoeSwizzleRuntimeIntervalKind::EVENT_WAIT_FIXED;
    case MoeSwizzleCalibrationKind::TERMINAL_DONE:
        return MoeSwizzleRuntimeIntervalKind::TERMINAL_DONE_FIXED;
    case MoeSwizzleCalibrationKind::DTE_HOP:
        break;
    }
    throw std::invalid_argument("DTE hop is not a WorkerCore interval");
}

std::map<uint16_t, std::vector<TickInterval>> EndpointIntervals(
    std::vector<P2pEndpointLifetimeEvent> events,
    const std::map<uint16_t, uint16_t> &runtime_core_to_die,
    uint64_t window_ticks) {
    std::sort(events.begin(), events.end(),
              [](const P2pEndpointLifetimeEvent &lhs,
                 const P2pEndpointLifetimeEvent &rhs) {
                  return std::tie(lhs.simulation_ticks, lhs.delta_cycle,
                                  lhs.local_sequence, lhs.local_core,
                                  lhs.direction, lhs.delta) <
                         std::tie(rhs.simulation_ticks, rhs.delta_cycle,
                                  rhs.local_sequence, rhs.local_core,
                                  rhs.direction, rhs.delta);
              });
    std::map<uint16_t, int64_t> active;
    std::map<uint16_t, uint64_t> starts;
    std::map<uint16_t, std::vector<TickInterval>> result;
    for (const auto &event : events) {
        if (runtime_core_to_die.find(event.local_core) ==
            runtime_core_to_die.end())
            throw std::invalid_argument(
                "endpoint event core lacks an exact manifest binding");
        if (event.simulation_ticks > window_ticks ||
            (event.delta != 1 && event.delta != -1))
            throw std::invalid_argument("endpoint lifetime event is invalid");
        int64_t &count = active[event.local_core];
        if (event.delta == 1) {
            if (count == 0) starts[event.local_core] = event.simulation_ticks;
            ++count;
        } else {
            if (count <= 0)
                throw std::invalid_argument(
                    "endpoint lifetime retirement has no open session");
            --count;
            if (count == 0) {
                result[event.local_core].push_back(
                    {starts.at(event.local_core), event.simulation_ticks});
                starts.erase(event.local_core);
            }
        }
    }
    for (const auto &[core, count] : active) {
        (void)core;
        if (count != 0)
            throw std::invalid_argument(
                "endpoint lifetime capture has unretired sessions");
    }
    return result;
}

} // namespace

void FinalizeMoeSwizzlePendingFixedIntervals(
    uint16_t runtime_core, uint64_t start_ticks, uint64_t end_ticks,
    std::vector<MoeSwizzleRuntimeIntervalKind> *pending,
    std::vector<MoeSwizzleRuntimeInterval> *completed) {
    if (pending == nullptr || completed == nullptr || pending->empty())
        throw std::invalid_argument(
            "pending fixed interval finalization requires nonempty storage");
    if (end_ticks <= start_ticks)
        throw std::logic_error(
            "fixed calibration record interval is not positive");
    for (MoeSwizzleRuntimeIntervalKind kind : *pending)
        completed->push_back(
            {runtime_core, kind, start_ticks, end_ticks});
    pending->clear();
}

MoeSwizzleRuntimeIntervalMarkers BuildMoeSwizzleRuntimeIntervalMarkers(
    const std::vector<MoeSwizzleRuntimeInterval> &record_intervals,
    const std::vector<P2pEndpointLifetimeEvent> &endpoint_events,
    const std::map<uint16_t, uint16_t> &runtime_core_to_die,
    uint16_t die_count, uint64_t window_ticks, uint64_t ticks_per_cycle,
    const MoeSwizzleRuntimeManifestCounts &counts) {
    if (die_count == 0 || window_ticks == 0 || ticks_per_cycle == 0 ||
        window_ticks % ticks_per_cycle != 0)
        throw std::invalid_argument("runtime marker observation window is invalid");
    std::vector<bool> dies(die_count, false);
    for (const auto &[core, die] : runtime_core_to_die) {
        (void)core;
        if (die >= die_count)
            throw std::invalid_argument("manifest core binding has invalid die");
        dies[die] = true;
    }
    if (std::find(dies.begin(), dies.end(), false) != dies.end())
        throw std::invalid_argument("manifest does not bind every marker die");

    std::map<uint16_t, std::vector<TickInterval>> compute_by_core;
    std::map<uint16_t, std::vector<TickInterval>> dte_by_core =
        EndpointIntervals(endpoint_events, runtime_core_to_die, window_ticks);
    std::map<MoeSwizzleRuntimeIntervalKind, std::vector<TickInterval>> setup;
    uint64_t captured_group_gemm = 0;
    uint64_t captured_group_gemm_setup = 0;
    uint64_t captured_group_gemm_execution = 0;
    uint64_t captured_local_dte = 0;
    uint64_t captured_event_records = 0;
    for (const MoeSwizzleRuntimeInterval &value : record_intervals) {
        if (runtime_core_to_die.find(value.runtime_core) ==
            runtime_core_to_die.end())
            throw std::invalid_argument(
                "runtime interval core lacks an exact manifest binding");
        if (value.start_ticks > value.end_ticks ||
            value.end_ticks > window_ticks)
            throw std::invalid_argument("runtime interval is outside the window");
        const TickInterval interval{value.start_ticks, value.end_ticks};
        if (IsCompute(value.kind)) compute_by_core[value.runtime_core].push_back(interval);
        if (value.kind == MoeSwizzleRuntimeIntervalKind::MATMUL)
            ++captured_group_gemm;
        if (value.kind == MoeSwizzleRuntimeIntervalKind::GROUP_GEMM_SETUP)
            ++captured_group_gemm_setup;
        if (value.kind == MoeSwizzleRuntimeIntervalKind::GROUP_GEMM_EXECUTION)
            ++captured_group_gemm_execution;
        if (value.kind == MoeSwizzleRuntimeIntervalKind::LOCAL_DTE) {
            dte_by_core[value.runtime_core].push_back(interval);
            ++captured_local_dte;
        }
        if (value.kind == MoeSwizzleRuntimeIntervalKind::EVENT_CONTROL)
            ++captured_event_records;
        if (value.kind == MoeSwizzleRuntimeIntervalKind::MATMUL ||
            value.kind == MoeSwizzleRuntimeIntervalKind::GROUP_GEMM_SETUP ||
            value.kind == MoeSwizzleRuntimeIntervalKind::SRAM_LIFECYCLE ||
            value.kind == MoeSwizzleRuntimeIntervalKind::SRAM_BIND ||
            value.kind == MoeSwizzleRuntimeIntervalKind::EVENT_CONTROL)
            setup[value.kind].push_back(interval);
    }
    uint64_t endpoint_launches = 0;
    for (const auto &event : endpoint_events) {
        if (event.delta == 1)
            endpoint_launches = CheckedAdd(
                endpoint_launches, 1,
                "endpoint launch count overflows u64");
    }
    const uint64_t captured_dte = CheckedAdd(
        captured_local_dte, endpoint_launches,
        "DTE launch count overflows u64");
    if (captured_group_gemm != counts.group_gemm_primitives ||
        captured_group_gemm_setup != counts.group_gemm_primitives ||
        captured_group_gemm_execution != counts.group_gemm_primitives ||
        captured_dte != counts.dte_launch_count ||
        captured_event_records != counts.event_record_count) {
        std::ostringstream message;
        message << "runtime interval capture count disagrees with validated "
                   "artifact: matmul=" << captured_group_gemm << "/"
                << counts.group_gemm_primitives << " dte=" << captured_dte
                << "/" << counts.dte_launch_count << " event="
                << captured_event_records << "/" << counts.event_record_count;
        throw std::invalid_argument(message.str());
    }

    std::vector<TickInterval> global_compute;
    std::vector<TickInterval> global_dte;
    MoeSwizzleRuntimeIntervalMarkers result;
    for (uint16_t die = 0; die < die_count; ++die) {
        std::vector<TickInterval> compute;
        std::vector<TickInterval> dte;
        for (const auto &[core, bound_die] : runtime_core_to_die) {
            if (bound_die != die) continue;
            const auto compute_it = compute_by_core.find(core);
            if (compute_it != compute_by_core.end())
                compute.insert(compute.end(), compute_it->second.begin(),
                               compute_it->second.end());
            const auto dte_it = dte_by_core.find(core);
            if (dte_it != dte_by_core.end())
                dte.insert(dte.end(), dte_it->second.begin(), dte_it->second.end());
        }
        compute = UnionIntervals(std::move(compute));
        dte = UnionIntervals(std::move(dte));
        const auto overlap = IntersectIntervals(compute, dte);
        global_compute.insert(global_compute.end(), compute.begin(), compute.end());
        global_dte.insert(global_dte.end(), dte.begin(), dte.end());
        result.overlap.push_back(
            {die, false, DurationCycles(compute, ticks_per_cycle),
             DurationCycles(dte, ticks_per_cycle),
             DurationCycles(overlap, ticks_per_cycle),
             window_ticks / ticks_per_cycle});
    }
    global_compute = UnionIntervals(std::move(global_compute));
    global_dte = UnionIntervals(std::move(global_dte));
    const auto global_overlap = IntersectIntervals(global_compute, global_dte);
    result.overlap.push_back(
        {die_count, true, DurationCycles(global_compute, ticks_per_cycle),
         DurationCycles(global_dte, ticks_per_cycle),
         DurationCycles(global_overlap, ticks_per_cycle),
         window_ticks / ticks_per_cycle});
    result.setup = {
        counts.group_gemm_primitives,
        DurationCycles(
            UnionIntervals(setup[MoeSwizzleRuntimeIntervalKind::GROUP_GEMM_SETUP]),
            ticks_per_cycle),
        DurationCycles(UnionIntervals(setup[MoeSwizzleRuntimeIntervalKind::MATMUL]),
                       ticks_per_cycle),
        counts.dte_launch_count,
        counts.physical_root_count,
        counts.event_record_count,
        DurationCycles(UnionIntervals(setup[MoeSwizzleRuntimeIntervalKind::SRAM_LIFECYCLE]),
                       ticks_per_cycle),
        DurationCycles(UnionIntervals(setup[MoeSwizzleRuntimeIntervalKind::SRAM_BIND]),
                       ticks_per_cycle),
        DurationCycles(UnionIntervals(setup[MoeSwizzleRuntimeIntervalKind::EVENT_CONTROL]),
                       ticks_per_cycle),
    };
    return result;
}

MoeSwizzleCalibrationMarker BuildMoeSwizzleCalibrationMarker(
    const MoeSwizzleCalibrationRequest &request,
    const std::vector<MoeSwizzleRuntimeInterval> &record_intervals,
    const std::vector<uint64_t> &directional_data_hop_busy_cycles,
    const MoeSwizzleRuntimeManifestCounts &manifest_counts,
    uint64_t ticks_per_cycle) {
    if (request.sample_index >= 3 || request.repeat_index >= 2 ||
        ticks_per_cycle == 0)
        throw std::invalid_argument(
            "calibration sample/repeat/tick contract is invalid");
    const bool group = request.kind == MoeSwizzleCalibrationKind::GROUP_GEMM;
    const bool swiglu =
        request.kind == MoeSwizzleCalibrationKind::SWIGLU_GROUP;
    const bool shaped = group || swiglu;
    if (shaped != request.shape.has_value() ||
        (shaped && std::any_of(request.shape->begin(), request.shape->end(),
                               [](uint64_t value) { return value == 0; })))
        throw std::invalid_argument(
            "GroupGEMM and SWIGLU_GROUP calibrations require a positive shape");
    if (swiglu) {
        const uint64_t m = (*request.shape)[0];
        const uint64_t intermediate = (*request.shape)[1];
        const uint64_t flattened = (*request.shape)[2];
        if (intermediate != 32 ||
            m > std::numeric_limits<uint64_t>::max() / intermediate ||
            flattened != m * intermediate ||
            flattened > std::numeric_limits<uint64_t>::max() / 4)
            throw std::invalid_argument(
                "SWIGLU_GROUP shape must be exact (M,32,M*32)");
        if (manifest_counts.compute_record_count != 1 ||
            manifest_counts.swiglu_primitives != 1 ||
            manifest_counts.swiglu_fp16_primitives != 1 ||
            manifest_counts.swiglu_runtime_core != request.runtime_core ||
            manifest_counts.swiglu_flattened_elements != flattened ||
            manifest_counts.swiglu_input_bytes != 4 * flattened ||
            manifest_counts.swiglu_output_bytes != 2 * flattened ||
            manifest_counts.dte_record_count != 0 ||
            manifest_counts.endpoint_session_count != 0)
            throw std::invalid_argument(
                "isolated SWIGLU_GROUP manifest/workload closure is not exact");
        const auto forbidden = [](MoeSwizzleRuntimeIntervalKind kind) {
            return kind == MoeSwizzleRuntimeIntervalKind::DTE_LAUNCH ||
                   kind == MoeSwizzleRuntimeIntervalKind::DTE_SYNC ||
                   kind == MoeSwizzleRuntimeIntervalKind::SESSION_OPEN ||
                   kind == MoeSwizzleRuntimeIntervalKind::SESSION_RETIRE ||
                   kind == MoeSwizzleRuntimeIntervalKind::LOCAL_DTE;
        };
        if (std::any_of(record_intervals.begin(), record_intervals.end(),
                        [&](const MoeSwizzleRuntimeInterval &item) {
                            return forbidden(item.kind);
                        }) ||
            std::any_of(directional_data_hop_busy_cycles.begin(),
                        directional_data_hop_busy_cycles.end(),
                        [](uint64_t cycles) { return cycles != 0; }))
            throw std::invalid_argument(
                "isolated SWIGLU_GROUP forbids DTE/session/link activity");
    }
    for (const std::string *digest :
         {&request.tool_sha256, &request.hardware_sha256,
          &request.simulation_sha256, &request.mapping_sha256})
        if (!IsSha256(*digest))
            throw std::invalid_argument(
                "calibration provenance must be lowercase SHA-256");

    uint64_t cycles = 0;
    if (request.kind == MoeSwizzleCalibrationKind::DTE_HOP) {
        std::vector<uint64_t> positive;
        std::copy_if(directional_data_hop_busy_cycles.begin(),
                     directional_data_hop_busy_cycles.end(),
                     std::back_inserter(positive),
                     [](uint64_t value) { return value != 0; });
        if (positive.size() != 1)
            throw std::invalid_argument(
                "isolated DTE hop requires exactly one busy directed link");
        cycles = positive.front();
    } else {
        const MoeSwizzleRuntimeIntervalKind expected =
            CalibrationIntervalKind(request.kind);
        std::vector<const MoeSwizzleRuntimeInterval *> matches;
        for (const auto &interval : record_intervals)
            if (interval.runtime_core == request.runtime_core &&
                interval.kind == expected)
                matches.push_back(&interval);
        if (matches.size() != 1)
            throw std::invalid_argument(
                "isolated calibration requires exactly one matching interval");
        const auto &value = *matches.front();
        if (value.end_ticks <= value.start_ticks ||
            (value.end_ticks - value.start_ticks) % ticks_per_cycle != 0)
            throw std::invalid_argument(
                "calibration interval is nonpositive or off-cycle");
        cycles = (value.end_ticks - value.start_ticks) / ticks_per_cycle;
    }
    if (cycles == 0)
        throw std::invalid_argument("calibration cycles must be positive");
    return {request, cycles};
}

std::string FormatMoeSwizzleCalibrationMarker(
    const MoeSwizzleCalibrationMarker &marker) {
    const auto &request = marker.request;
    std::ostringstream output;
    output << "[MOE_SWIZZLE_CALIBRATION] kind="
           << CalibrationKindName(request.kind)
           << " sample=" << static_cast<unsigned>(request.sample_index)
           << " repeat=" << static_cast<unsigned>(request.repeat_index)
           << " cycles=" << marker.cycles << " shape=";
    if (request.shape.has_value())
        output << (*request.shape)[0] << "x" << (*request.shape)[1]
               << "x" << (*request.shape)[2];
    else
        output << "none";
    output << " dtype=" << (request.shape.has_value() ? "fp16" : "none")
           << " tool_sha256=" << request.tool_sha256
           << " hardware_sha256=" << request.hardware_sha256
           << " simulation_sha256=" << request.simulation_sha256
           << " mapping_sha256=" << request.mapping_sha256;
    return output.str();
}

std::string FormatMoeSwizzleOverlapMarker(
    const MoeSwizzleOverlapMarker &marker) {
    std::ostringstream output;
    output << "[MOE_SWIZZLE_OVERLAP] scope="
           << (marker.global ? "global" : "die") << " die=";
    if (marker.global) output << "all"; else output << marker.die;
    output << " compute_cycles=" << marker.compute_cycles
           << " dte_cycles=" << marker.dte_cycles
           << " compute_dte_cycles=" << marker.compute_dte_cycles
           << " window_cycles=" << marker.window_cycles;
    return output.str();
}

std::string FormatMoeSwizzleSetupMarker(
    const MoeSwizzleSetupMarker &marker) {
    std::ostringstream output;
    output << "[MOE_SWIZZLE_SETUP] group_gemm_primitives="
           << marker.group_gemm_primitives
           << " group_gemm_setup_cycles="
           << marker.group_gemm_setup_cycles
           << " matmul_total_cycles=" << marker.matmul_total_cycles
           << " dte_launch_count=" << marker.dte_launch_count
           << " physical_root_count=" << marker.physical_root_count
           << " event_record_count=" << marker.event_record_count
           << " sram_lifecycle_cycles=" << marker.sram_lifecycle_cycles
           << " bind_cycles=" << marker.bind_cycles
           << " event_control_cycles=" << marker.event_control_cycles;
    return output.str();
}
