#pragma once

#include "dte/p2p_session_runtime.h"

#include <cstdint>
#include <array>
#include <map>
#include <optional>
#include <string>
#include <vector>

enum class MoeSwizzleRuntimeIntervalKind : uint8_t {
    MATMUL = 0,
    LOCAL_REDUCE = 1,
    LOCAL_DTE = 2,
    SRAM_LIFECYCLE = 3,
    SRAM_BIND = 4,
    EVENT_CONTROL = 5,
    GROUP_GEMM_SETUP = 6,
    GROUP_GEMM_EXECUTION = 7,
    DTE_LAUNCH = 8,
    DTE_SYNC = 9,
    SESSION_OPEN = 10,
    SESSION_RETIRE = 11,
    LOCAL_COPY_FIXED = 12,
    SRAM_ALLOC_FIXED = 13,
    SRAM_FREE_FIXED = 14,
    EVENT_SET_FIXED = 15,
    EVENT_WAIT_FIXED = 16,
    TERMINAL_DONE_FIXED = 17,
    SWIGLU_GROUP = 18,
};

struct MoeSwizzleRuntimeInterval {
    uint16_t runtime_core = 0;
    MoeSwizzleRuntimeIntervalKind kind =
        MoeSwizzleRuntimeIntervalKind::MATMUL;
    uint64_t start_ticks = 0;
    uint64_t end_ticks = 0;
};

void FinalizeMoeSwizzlePendingFixedIntervals(
    uint16_t runtime_core, uint64_t start_ticks, uint64_t end_ticks,
    std::vector<MoeSwizzleRuntimeIntervalKind> *pending,
    std::vector<MoeSwizzleRuntimeInterval> *completed);

struct MoeSwizzleRuntimeManifestCounts {
    uint64_t group_gemm_primitives = 0;
    uint64_t dte_launch_count = 0;
    uint64_t physical_root_count = 0;
    uint64_t event_record_count = 0;
    uint64_t compute_record_count = 0;
    uint64_t swiglu_primitives = 0;
    uint64_t swiglu_fp16_primitives = 0;
    uint64_t swiglu_runtime_core = 0;
    uint64_t swiglu_flattened_elements = 0;
    uint64_t swiglu_input_bytes = 0;
    uint64_t swiglu_output_bytes = 0;
    uint64_t dte_record_count = 0;
    uint64_t endpoint_session_count = 0;
};

struct MoeSwizzleOverlapMarker {
    // die==die_count denotes the global wall-clock union, not a physical die.
    uint16_t die = 0;
    bool global = false;
    uint64_t compute_cycles = 0;
    uint64_t dte_cycles = 0;
    uint64_t compute_dte_cycles = 0;
    uint64_t window_cycles = 0;
};

struct MoeSwizzleSetupMarker {
    uint64_t group_gemm_primitives = 0;
    uint64_t group_gemm_setup_cycles = 0;
    uint64_t matmul_total_cycles = 0;
    uint64_t dte_launch_count = 0;
    uint64_t physical_root_count = 0;
    uint64_t event_record_count = 0;
    uint64_t sram_lifecycle_cycles = 0;
    uint64_t bind_cycles = 0;
    uint64_t event_control_cycles = 0;
};

struct MoeSwizzleRuntimeIntervalMarkers {
    std::vector<MoeSwizzleOverlapMarker> overlap;
    MoeSwizzleSetupMarker setup;
};

enum class MoeSwizzleCalibrationKind : uint8_t {
    GROUP_GEMM = 0,
    DTE_LAUNCH = 1,
    DTE_SYNC = 2,
    DTE_HOP = 3,
    SESSION_OPEN = 4,
    SESSION_RETIRE = 5,
    LOCAL_COPY = 6,
    SRAM_ALLOC = 7,
    SRAM_BIND = 8,
    SRAM_FREE = 9,
    EVENT_SET = 10,
    EVENT_WAIT = 11,
    TERMINAL_DONE = 12,
    SWIGLU_GROUP = 13,
};

struct MoeSwizzleCalibrationRequest {
    MoeSwizzleCalibrationKind kind = MoeSwizzleCalibrationKind::GROUP_GEMM;
    uint16_t runtime_core = 0;
    uint8_t sample_index = 0;
    uint8_t repeat_index = 0;
    std::optional<std::array<uint64_t, 3>> shape;
    std::string tool_sha256;
    std::string hardware_sha256;
    std::string simulation_sha256;
    std::string mapping_sha256;
};

struct MoeSwizzleCalibrationMarker {
    MoeSwizzleCalibrationRequest request;
    uint64_t cycles = 0;
};

MoeSwizzleRuntimeIntervalMarkers BuildMoeSwizzleRuntimeIntervalMarkers(
    const std::vector<MoeSwizzleRuntimeInterval> &record_intervals,
    const std::vector<P2pEndpointLifetimeEvent> &endpoint_events,
    const std::map<uint16_t, uint16_t> &runtime_core_to_die,
    uint16_t die_count, uint64_t window_ticks, uint64_t ticks_per_cycle,
    const MoeSwizzleRuntimeManifestCounts &counts);

std::string FormatMoeSwizzleOverlapMarker(
    const MoeSwizzleOverlapMarker &marker);
std::string FormatMoeSwizzleSetupMarker(
    const MoeSwizzleSetupMarker &marker);
MoeSwizzleCalibrationMarker BuildMoeSwizzleCalibrationMarker(
    const MoeSwizzleCalibrationRequest &request,
    const std::vector<MoeSwizzleRuntimeInterval> &record_intervals,
    const std::vector<uint64_t> &directional_data_hop_busy_cycles,
    const MoeSwizzleRuntimeManifestCounts &manifest_counts,
    uint64_t ticks_per_cycle);
std::string FormatMoeSwizzleCalibrationMarker(
    const MoeSwizzleCalibrationMarker &marker);

int RunMoeSwizzleRuntimeCaptureSelfTest();
