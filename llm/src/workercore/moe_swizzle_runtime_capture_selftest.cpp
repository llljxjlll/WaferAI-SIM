#include "workercore/moe_swizzle_runtime_capture.h"

#include <iostream>
#include <stdexcept>

namespace {

template <class F>
bool Throws(F &&function) {
    try {
        function();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

} // namespace

int RunMoeSwizzleRuntimeCaptureSelfTest() {
    int tests = 0;
    int failures = 0;
    auto check = [&](bool condition, const char *message) {
        ++tests;
        if (!condition) {
            ++failures;
            std::cerr << "[MOE_SWIZZLE_RUNTIME_CAPTURE_SELFTEST] FAIL: "
                      << message << "\n";
        }
    };
    const std::map<uint16_t, uint16_t> bindings{
        {0, 0}, {4, 0}, {1, 1}, {2, 2}, {3, 3}};
    const std::vector<MoeSwizzleRuntimeInterval> intervals{
        {0, MoeSwizzleRuntimeIntervalKind::MATMUL, 10, 30},
        {0, MoeSwizzleRuntimeIntervalKind::MATMUL, 20, 40},
        {0, MoeSwizzleRuntimeIntervalKind::GROUP_GEMM_SETUP, 8, 10},
        {0, MoeSwizzleRuntimeIntervalKind::GROUP_GEMM_SETUP, 18, 20},
        {1, MoeSwizzleRuntimeIntervalKind::GROUP_GEMM_SETUP, 48, 50},
        {0, MoeSwizzleRuntimeIntervalKind::GROUP_GEMM_EXECUTION, 10, 30},
        {0, MoeSwizzleRuntimeIntervalKind::GROUP_GEMM_EXECUTION, 20, 40},
        {1, MoeSwizzleRuntimeIntervalKind::GROUP_GEMM_EXECUTION, 50, 60},
        {4, MoeSwizzleRuntimeIntervalKind::LOCAL_DTE, 15, 25},
        {1, MoeSwizzleRuntimeIntervalKind::MATMUL, 50, 60},
        {1, MoeSwizzleRuntimeIntervalKind::LOCAL_DTE, 60, 70},
        {0, MoeSwizzleRuntimeIntervalKind::SRAM_LIFECYCLE, 1, 4},
        {4, MoeSwizzleRuntimeIntervalKind::SRAM_LIFECYCLE, 2, 5},
        {0, MoeSwizzleRuntimeIntervalKind::SRAM_BIND, 5, 7},
        {0, MoeSwizzleRuntimeIntervalKind::EVENT_CONTROL, 7, 9},
        {0, MoeSwizzleRuntimeIntervalKind::EVENT_CONTROL, 7, 9},
        {0, MoeSwizzleRuntimeIntervalKind::EVENT_CONTROL, 7, 9},
        {0, MoeSwizzleRuntimeIntervalKind::EVENT_CONTROL, 7, 9},
        {0, MoeSwizzleRuntimeIntervalKind::EVENT_CONTROL, 7, 9},
    };
    const MoeSwizzleRuntimeManifestCounts counts{3, 4, 7, 5};
    const std::vector<P2pEndpointLifetimeEvent> endpoint_events{
        {80, 0, 0, 2, P2pEndpointDirection::TX, 1},
        {80, 0, 1, 2, P2pEndpointDirection::TX, -1},
        {80, 0, 0, 3, P2pEndpointDirection::RX, 1},
        {80, 0, 1, 3, P2pEndpointDirection::RX, -1},
    };
    const auto result = BuildMoeSwizzleRuntimeIntervalMarkers(
        intervals, endpoint_events, bindings, 4, 100, 1, counts);
    check(result.overlap.size() == 5,
          "four die markers plus one global marker are mandatory");
    check(result.overlap[0].compute_cycles == 30 &&
              result.overlap[0].dte_cycles == 10 &&
              result.overlap[0].compute_dte_cycles == 10,
          "same-die cross-core union/intersection is exact");
    check(result.overlap[1].compute_dte_cycles == 0,
          "zero overlap is valid and must remain observable");
    check(result.overlap[2].compute_cycles == 0 &&
              result.overlap[3].dte_cycles == 0,
          "inactive dies still emit zero-valued markers");
    check(result.overlap[4].global &&
              result.overlap[4].compute_dte_cycles == 10,
          "global wall-clock union is distinct from per-die accounting");
    check(result.setup.group_gemm_setup_cycles == 6 &&
              result.setup.matmul_total_cycles == 40 &&
              result.setup.sram_lifecycle_cycles == 4 &&
              result.setup.bind_cycles == 2 &&
              result.setup.event_control_cycles == 2,
          "setup cycles use interval union and never record-count constants");
    check(result.setup.group_gemm_primitives == 3 &&
              result.setup.dte_launch_count == 4 &&
              result.setup.physical_root_count == 7 &&
              result.setup.event_record_count == 5,
          "validated manifest counts remain exact");
    std::vector<P2pEndpointLifetimeEvent> unretired{
        {10, 0, 0, 0, P2pEndpointDirection::TX, 1}};
    check(Throws([&] {
              (void)BuildMoeSwizzleRuntimeIntervalMarkers(
                  {}, unretired, bindings, 4, 100, 1, counts);
          }),
          "missing endpoint retirement fails closed");
    auto wrong_core = intervals;
    wrong_core.push_back(
        {9, MoeSwizzleRuntimeIntervalKind::MATMUL, 0, 1});
    check(Throws([&] {
              (void)BuildMoeSwizzleRuntimeIntervalMarkers(
                  wrong_core, endpoint_events, bindings, 4, 100, 1, counts);
          }),
          "capture without manifest binding fails closed");
    check(FormatMoeSwizzleOverlapMarker(result.overlap.front()) ==
              "[MOE_SWIZZLE_OVERLAP] scope=die die=0 compute_cycles=30 "
              "dte_cycles=10 compute_dte_cycles=10 window_cycles=100",
          "overlap marker grammar is exact");
    check(FormatMoeSwizzleSetupMarker(result.setup) ==
              "[MOE_SWIZZLE_SETUP] group_gemm_primitives=3 "
              "group_gemm_setup_cycles=6 matmul_total_cycles=40 "
              "dte_launch_count=4 "
              "physical_root_count=7 event_record_count=5 "
              "sram_lifecycle_cycles=4 bind_cycles=2 "
              "event_control_cycles=2",
          "setup marker grammar is exact");
    const std::string digest(64, 'a');
    MoeSwizzleCalibrationRequest calibration{
        MoeSwizzleCalibrationKind::GROUP_GEMM, 1, 2, 1,
        std::array<uint64_t, 3>{16, 16, 16},
        digest, digest, digest, digest};
    const auto group_marker = BuildMoeSwizzleCalibrationMarker(
        calibration, intervals, {}, counts, 1);
    check(group_marker.cycles == 10 &&
              FormatMoeSwizzleCalibrationMarker(group_marker) ==
                  "[MOE_SWIZZLE_CALIBRATION] kind=group_gemm sample=2 "
                  "repeat=1 cycles=10 shape=16x16x16 dtype=fp16 "
                  "tool_sha256=" + digest + " hardware_sha256=" + digest +
                  " simulation_sha256=" + digest + " mapping_sha256=" + digest,
          "isolated GroupGEMM exporter uses one real runtime interval");
    calibration.kind = MoeSwizzleCalibrationKind::DTE_HOP;
    calibration.shape.reset();
    const auto hop_marker = BuildMoeSwizzleCalibrationMarker(
        calibration, intervals, {0, 7, 0}, counts, 1);
    check(hop_marker.cycles == 7,
          "isolated DTE hop exporter requires one busy physical link");
    auto duplicate = intervals;
    duplicate.push_back(
        {1, MoeSwizzleRuntimeIntervalKind::MATMUL, 70, 80});
    calibration.kind = MoeSwizzleCalibrationKind::GROUP_GEMM;
    calibration.shape = std::array<uint64_t, 3>{16, 16, 16};
    check(Throws([&] {
              (void)BuildMoeSwizzleCalibrationMarker(
                  calibration, duplicate, {}, counts, 1);
          }),
          "duplicate matching calibration intervals fail closed");
    calibration.kind = MoeSwizzleCalibrationKind::SRAM_BIND;
    calibration.runtime_core = 0;
    calibration.shape.reset();
    const auto bind_marker = BuildMoeSwizzleCalibrationMarker(
        calibration, intervals, {}, counts, 1);
    check(bind_marker.cycles == 2,
          "isolated SRAM_BIND exporter consumes one fixed runtime interval");
    auto duplicate_bind = intervals;
    duplicate_bind.push_back(
        {0, MoeSwizzleRuntimeIntervalKind::SRAM_BIND, 40, 42});
    check(Throws([&] {
              (void)BuildMoeSwizzleCalibrationMarker(
                  calibration, duplicate_bind, {}, counts, 1);
          }),
          "duplicate SRAM_BIND runtime intervals fail closed");
    std::vector<MoeSwizzleRuntimeIntervalKind> pending_terminal{
        MoeSwizzleRuntimeIntervalKind::TERMINAL_DONE_FIXED};
    std::vector<MoeSwizzleRuntimeInterval> completed_terminal;
    FinalizeMoeSwizzlePendingFixedIntervals(
        3, 10, 14, &pending_terminal, &completed_terminal);
    check(pending_terminal.empty() && completed_terminal.size() == 1 &&
              completed_terminal.front().runtime_core == 3 &&
              completed_terminal.front().kind ==
                  MoeSwizzleRuntimeIntervalKind::TERMINAL_DONE_FIXED &&
              completed_terminal.front().start_ticks == 10 &&
              completed_terminal.front().end_ticks == 14,
          "terminal fixed interval finalizes once at the real completion time");
    calibration.kind = MoeSwizzleCalibrationKind::TERMINAL_DONE;
    calibration.runtime_core = 3;
    calibration.shape.reset();
    const auto terminal_marker = BuildMoeSwizzleCalibrationMarker(
        calibration, completed_terminal, {}, counts, 1);
    check(terminal_marker.cycles == 4,
          "isolated TERMINAL_DONE consumes the finalized real interval");
    std::vector<MoeSwizzleRuntimeIntervalKind> invalid_terminal{
        MoeSwizzleRuntimeIntervalKind::TERMINAL_DONE_FIXED};
    check(Throws([&] {
              FinalizeMoeSwizzlePendingFixedIntervals(
                  3, 10, 10, &invalid_terminal, &completed_terminal);
          }) && invalid_terminal.size() == 1,
          "nonpositive terminal completion fails without mutating pending state");
    const std::vector<MoeSwizzleRuntimeInterval> swiglu_intervals{
        {1, MoeSwizzleRuntimeIntervalKind::SWIGLU_GROUP, 80, 88}};
    MoeSwizzleRuntimeManifestCounts swiglu_counts{};
    swiglu_counts.compute_record_count = 1;
    swiglu_counts.swiglu_primitives = 1;
    swiglu_counts.swiglu_fp16_primitives = 1;
    swiglu_counts.swiglu_runtime_core = 1;
    swiglu_counts.swiglu_flattened_elements = 64;
    swiglu_counts.swiglu_input_bytes = 256;
    swiglu_counts.swiglu_output_bytes = 128;
    calibration.kind = MoeSwizzleCalibrationKind::SWIGLU_GROUP;
    calibration.runtime_core = 1;
    calibration.shape = std::array<uint64_t, 3>{2, 32, 64};
    const auto swiglu_marker = BuildMoeSwizzleCalibrationMarker(
        calibration, swiglu_intervals, std::vector<uint64_t>(8, 0),
        swiglu_counts, 1);
    check(swiglu_marker.cycles == 8 &&
              FormatMoeSwizzleCalibrationMarker(swiglu_marker) ==
                  "[MOE_SWIZZLE_CALIBRATION] kind=swiglu_group sample=2 "
                  "repeat=1 cycles=8 shape=2x32x64 dtype=fp16 "
                  "tool_sha256=" + digest + " hardware_sha256=" + digest +
                  " simulation_sha256=" + digest + " mapping_sha256=" + digest,
          "isolated SWIGLU_GROUP closes one real FP16 record and interval");
    auto wrong_swiglu_shape = calibration;
    wrong_swiglu_shape.shape = std::array<uint64_t, 3>{2, 32, 65};
    check(Throws([&] {
              (void)BuildMoeSwizzleCalibrationMarker(
                  wrong_swiglu_shape, swiglu_intervals,
                  std::vector<uint64_t>(8, 0), swiglu_counts, 1);
          }),
          "SWIGLU_GROUP flattened shape mismatch fails closed");
    auto wrong_swiglu_compute = swiglu_counts;
    wrong_swiglu_compute.compute_record_count = 2;
    check(Throws([&] {
              (void)BuildMoeSwizzleCalibrationMarker(
                  calibration, swiglu_intervals,
                  std::vector<uint64_t>(8, 0), wrong_swiglu_compute, 1);
          }),
          "SWIGLU_GROUP second compute record fails closed");
    auto wrong_swiglu_dtype = swiglu_counts;
    wrong_swiglu_dtype.swiglu_fp16_primitives = 0;
    check(Throws([&] {
              (void)BuildMoeSwizzleCalibrationMarker(
                  calibration, swiglu_intervals,
                  std::vector<uint64_t>(8, 0), wrong_swiglu_dtype, 1);
          }),
          "SWIGLU_GROUP non-FP16 record fails closed");
    auto swiglu_with_dte = swiglu_counts;
    swiglu_with_dte.dte_record_count = 1;
    check(Throws([&] {
              (void)BuildMoeSwizzleCalibrationMarker(
                  calibration, swiglu_intervals,
                  std::vector<uint64_t>(8, 0), swiglu_with_dte, 1);
          }),
          "SWIGLU_GROUP artifact carrying DTE records fails closed");
    check(Throws([&] {
              (void)BuildMoeSwizzleCalibrationMarker(
                  calibration, swiglu_intervals,
                  std::vector<uint64_t>{0, 0, 1, 0, 0, 0, 0, 0},
                  swiglu_counts, 1);
          }),
          "SWIGLU_GROUP physical link activity fails closed");
    std::cout << "[MOE_SWIZZLE_RUNTIME_CAPTURE_SELFTEST] "
              << (failures == 0 ? "PASS" : "FAIL") << " "
              << (tests - failures) << "/" << tests << "\n";
    return failures == 0 ? 0 : 1;
}
