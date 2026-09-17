#include "isa/opcode.h"

#include "assert.h"
#include "defs/global.h"
#include "defs/spec.h"
#include "die/d2d_link.h"
#include "die/port.h"
#include "frontend/program_io.h"
#include "frontend/program_finalizer.h"
#include "isa/isa_v1_selftest.h"
#include "isa/p5_memory_probe.h"
#include "isa/p5_memory_probe_selftest.h"
#include "isa/p8_double_buffer_probe.h"
#include "isa/prim_manifest.h"
#include "utils/prim_utils.h"
#include "memory/hbm_r0_selftest.h"
#include "memory/hbm_r1_selftest.h"
#include "memory/hbm_r2_selftest.h"
#include "memory/hbm_r3_selftest.h"
#include "memory/hbm_r4_selftest.h"
#include "memory/external_dma_program.h"
#include "memory/dense_adamw_mid_program_pager.h"
#include "memory/dense_inference_mid_program_pager.h"
#include "memory/moe_inference_mid_program_pager.h"
#include "memory/sram/sram_selftest.h"
#include "dte/dte_async.h"
#include "dte/dte_control_core.h"
#include "dte/dte_unit.h"
#include "dte/coll_runtime.h"
#include "dte/p2p_payload.h"
#include "dte/p2p_payload_selftest.h"
#include "dte/p2p_session_runtime_selftest.h"
#include "dte/sync_runtime_selftest.h"
#include "dte/coll_multicast.h"
#include "dte/coll_innetwork_reduce.h"
#include "monitor/monitor.h"
#include "monitor/config_helper_program.h"
#include "monitor/config_helper_program_sequence.h"
#include "monitor/watchdog.h"
#include "monitor/start_data_tracker.h"
#include "monitor/workload_rendezvous_selftest.h"
#include "prims/collective_data_v1_prim_selftest.h"
#include "prims/collective_phase_barrier_v1_prim_selftest.h"
#include "router/router.h"
#include "systemc.h"
#include "trace/Event_engine.h"
#include "utils/print_utils.h"
#include "utils/router_utils.h"
#include "utils/config_preflight.h"
#include "utils/simple_flags.h"
#include "utils/system_utils.h"
#include "workercore/workercore.h"
#include "workercore/moe_swizzle_runtime_capture.h"
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <map>
#include <optional>
#include <numeric>
#include <algorithm>
#include <set>

// 假设 json.hpp 文件在当前目录或包含路径中
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include <SFML/Graphics.hpp>
using namespace std;

#ifndef NPUSIM_SOURCE_ROOT
#define NPUSIM_SOURCE_ROOT "."
#endif

Define_bool_opt("--help", g_flag_help, false, "show these help information");

Define_bool_opt("--isa-v1-selftest", g_flag_isa_v1_selftest, false,
                "run ISA v1 manifest, codec, graph, and lowering self-tests and exit");

Define_bool_opt("--collective-data-v1-prim-selftest",
                g_flag_collective_data_v1_prim_selftest, false,
                "run ID58 collective data SystemC self-test and exit");

Define_bool_opt("--collective-phase-barrier-v1-prim-selftest",
                g_flag_collective_phase_barrier_v1_prim_selftest, false,
                "run ID59 collective phase barrier SystemC self-test and exit");

Define_bool_opt("--sync-runtime-selftest", g_flag_sync_runtime_selftest, false,
                "run GROUP_SYNC and EVENT runtime self-test and exit");

Define_bool_opt("--p2p-payload-selftest", g_flag_p2p_payload_selftest, false,
                "run P2P payload codec/reassembly self-test and exit");

Define_bool_opt("--p2p-session-selftest", g_flag_p2p_session_selftest, false,
                "run P2P endpoint session runtime self-test and exit");

Define_bool_opt("--moe-swizzle-runtime-markers",
                g_flag_moe_swizzle_runtime_markers, false,
                "emit manifest-bound dedicated MoE Swizzle runtime markers");
Define_bool_opt("--moe-swizzle-runtime-capture-selftest",
                g_flag_moe_swizzle_runtime_capture_selftest, false,
                "run MoE Swizzle runtime interval aggregation self-test");
Define_string_opt("--moe-swizzle-calibration-kind",
                  g_flag_moe_swizzle_calibration_kind, std::string{},
                  "emit one isolated MoE Swizzle calibration sample");
Define_int64_opt("--moe-swizzle-calibration-core",
                 g_flag_moe_swizzle_calibration_core, 0,
                 "runtime core carrying the isolated calibration primitive");
Define_int64_opt("--moe-swizzle-calibration-sample",
                 g_flag_moe_swizzle_calibration_sample, 0,
                 "isolated calibration sample index in [0,2]");
Define_int64_opt("--moe-swizzle-calibration-repeat",
                 g_flag_moe_swizzle_calibration_repeat, 0,
                 "isolated calibration repeat index in [0,1]");
Define_string_opt("--moe-swizzle-calibration-shape",
                  g_flag_moe_swizzle_calibration_shape, std::string("none"),
                  "GroupGEMM/SWIGLU_GROUP MxNxK shape; fixed samples require none");
Define_string_opt("--moe-swizzle-calibration-tool-sha256",
                  g_flag_moe_swizzle_calibration_tool_sha256, std::string{},
                  "matching npusim tool SHA-256");
Define_string_opt("--moe-swizzle-calibration-hardware-sha256",
                  g_flag_moe_swizzle_calibration_hardware_sha256,
                  std::string{}, "matching hardware config SHA-256");
Define_string_opt("--moe-swizzle-calibration-simulation-sha256",
                  g_flag_moe_swizzle_calibration_simulation_sha256,
                  std::string{}, "matching simulation config SHA-256");
Define_string_opt("--moe-swizzle-calibration-mapping-sha256",
                  g_flag_moe_swizzle_calibration_mapping_sha256,
                  std::string{}, "matching mapping config SHA-256");

Define_bool_opt("--p5-memory-probe-selftest",
                g_flag_p5_memory_probe_selftest, false,
                "run P5 memory probe foundation self-test and exit");

Define_bool_opt("--program-one-shot", g_flag_program_one_shot, false,
                "execute a Program artifact once without primitive refill");

Define_string_opt("--linked-manifest-sequence",
                  g_flag_linked_manifest_sequence, std::string{},
                  "comma-separated linked manifests for --program-sequence");
Define_string_opt("--program-sequence", g_flag_program_sequence,
                  std::string{},
                  "comma-separated Program artifacts executed in one instance");
Define_string_opt("--program-io-sequence", g_flag_program_io_sequence,
                  std::string{},
                  "comma-separated ProgramIo sidecars for --program-sequence");
Define_bool_opt("--moe-forward-sequence", g_flag_moe_forward_sequence, false,
                "two EP1 MoE forward programs only; no backward or SGD claim");
Define_bool_opt("--moe-router-sgd-partial-sequence",
                g_flag_moe_router_sgd_partial_sequence, false,
                "two EP1 MoE partial programs with one native router SGD; not full training");
Define_bool_opt("--moe-layer1-sgd-partial-sequence",
                g_flag_moe_layer1_sgd_partial_sequence, false,
                "two EP1 MoE partial programs with four router/expert SGD writes; not full training");
Define_string_opt("--dense-adamw-paged-runtime",
                  g_flag_dense_adamw_paged_runtime, std::string{},
                  "source-signed per-StateABI external DMA for two Dense AdamW steps");
Define_string_opt("--dense-inference-paged-runtime",
                  g_flag_dense_inference_paged_runtime, std::string{},
                  "source-signed blocking parameter/KV DMA for Prefill+2Decode");
Define_string_opt("--moe-inference-paged-runtime",
                  g_flag_moe_inference_paged_runtime, std::string{},
                  "source-signed blocking full MoE EP2 expert/weight/KV DMA");
Define_string_opt(
    "--external-dma-binding", g_flag_external_dma_binding, std::string{},
    "typed external DMA startup binding for --program-sequence");
Define_string_opt("--linked-manifest", g_flag_linked_manifest,
                  std::string{},
                  "linked Program manifest JSON required by --program-io");
Define_string_opt("--program-io", g_flag_program_io, std::string{},
                  "symbol-aware ProgramIo host SRAM sidecar");
// Register the longer --program-io spelling before --program.  The legacy
// simple_flags parser accepts '-' as an inline-value separator and otherwise
// treats --program-io as --program=io.
Define_string_opt("--program", g_flag_program, std::string{},
                  "Program Format v1 artifact file");
Define_string_opt("--p5-memory-probe", g_flag_p5_memory_probe,
                  std::string{},
                  "test-only P5 memory preload/post-run probe sidecar");
Define_string_opt("--p6-memory-probe", g_flag_p6_memory_probe,
                  std::string{},
                  "test-only P6 multi-core/multi-region probe sidecar");
Define_string_opt("--p8-double-buffer-probe",
                  g_flag_p8_double_buffer_probe, std::string{},
                  "test-only P8 HBM/SRAM double-buffer probe sidecar");
Define_string_opt("--workload-config", g_flag_workload_config, std::string{},
                  "legacy JSON workload config file");
Define_string_opt("--hardware-config", g_flag_hardware_config,
                  std::string(NPUSIM_SOURCE_ROOT) +
                      "/llm/test/default/hardware.json",
                  "hardware config file");
Define_string_opt("--simulation-config", g_flag_simulation_config,
                  std::string(NPUSIM_SOURCE_ROOT) +
                      "/llm/test/default/simulation.json",
                  "simulation config file");
Define_string_opt("--mapping-config", g_flag_mapping_config,
                  std::string(NPUSIM_SOURCE_ROOT) +
                      "/llm/test/default/mapping.spec",
                  "mapping config file");

Define_int64_opt("--trace-window", g_flag_trace_window, 2, "Trace window size");

Define_bool_opt("--d2d-v0-selftest", g_flag_d2d_v0_selftest, false,
                "run D2D V0 pure-function self-test and exit");

Define_bool_opt("--d2d-link-selftest", g_flag_d2d_link_selftest, false,
                "run D2D V1 SystemC link self-test (drives packets) and exit");

Define_bool_opt("--dte-v0-selftest", g_flag_dte_v0_selftest, false,
                "run DTE V0 payload/resource SystemC self-test and exit");

Define_bool_opt("--dte-control-core-selftest",
                g_flag_dte_control_core_selftest, false,
                "run configurable DTE control-core self-test and exit");

Define_bool_opt("--coll-v0-selftest", g_flag_coll_v0_selftest, false,
                "run NoC collective V0 contract self-test and exit");

Define_bool_opt("--coll-v1-selftest", g_flag_coll_v1_selftest, false,
                "run NoC collective V1 planner/barrier self-test and exit");

Define_bool_opt("--coll-v2-selftest", g_flag_coll_v2_selftest, false,
                "run NoC collective V2 finite Gather reorder self-test and exit");

Define_bool_opt("--coll-v3-selftest", g_flag_coll_v3_selftest, false,
                "run NoC collective V3 Tier0 reduction self-test and exit");

Define_bool_opt("--coll-v4-selftest", g_flag_coll_v4_selftest, false,
                "run NoC collective V4 multicast contract self-test and exit");

Define_bool_opt("--coll-v5-selftest", g_flag_coll_v5_selftest, false,
                "run NoC collective V5 in-network reduce contract self-test and exit");

Define_bool_opt("--coll-v6-selftest", g_flag_coll_v6_selftest, false,
                "run NoC collective V6 integration/lifecycle self-test and exit");

Define_bool_opt("--coll-r0-selftest", g_flag_coll_r0_selftest, false,
                "run NoC collective refactor R0 contract self-test and exit");

Define_bool_opt("--coll-r1-selftest", g_flag_coll_r1_selftest, false,
                "run NoC collective refactor R1 config self-test and exit");

Define_bool_opt("--coll-r2-selftest", g_flag_coll_r2_selftest, false,
                "run NoC collective refactor R2 DCA ComputePool self-test and exit");

Define_bool_opt("--coll-r3-selftest", g_flag_coll_r3_selftest, false,
                "run NoC collective refactor R3 stream/state self-test and exit");

Define_bool_opt("--coll-r4-selftest", g_flag_coll_r4_selftest, false,
                "run NoC collective refactor R4 Router stream engine self-test and exit");

Define_bool_opt("--coll-r5-selftest", g_flag_coll_r5_selftest, false,
                "run NoC collective refactor R5 shared vector/FP self-test and exit");

Define_bool_opt("--coll-r6-selftest", g_flag_coll_r6_selftest, false,
                "run NoC collective refactor R6 reduce-only self-test and exit");

Define_bool_opt("--coll-r7-selftest", g_flag_coll_r7_selftest, false,
                "run NoC collective refactor R7 reduce+broadcast self-test and exit");

Define_bool_opt("--coll-r8-selftest", g_flag_coll_r8_selftest, false,
                "run NoC collective refactor R8 performance-oracle self-test and exit");

Define_bool_opt("--hbm-r0-selftest", g_flag_hbm_r0_selftest, false,
                "run distributed HBM R0 config/topology self-test and exit");

Define_bool_opt("--hbm-r1-selftest", g_flag_hbm_r1_selftest, false,
                "run distributed HBM R1 synthetic-request self-test and exit");

Define_bool_opt("--hbm-r2-selftest", g_flag_hbm_r2_selftest, false,
                "run distributed HBM R2 bandwidth/queueing self-test and exit");

Define_bool_opt("--hbm-r3-selftest", g_flag_hbm_r3_selftest, false,
                "run distributed HBM R3 DRAMSys-backend self-test and exit");
Define_bool_opt("--hbm-r4-selftest", g_flag_hbm_r4_selftest, false,
                "run distributed HBM R4 Router/NUMA integration self-test and exit");
Define_bool_opt("--sram-r0-selftest", g_flag_sram_r0_selftest, false,
                "run SRAM R0 configuration contract self-test and exit");
Define_bool_opt("--sram-r1-selftest", g_flag_sram_r1_selftest, false,
                "run SRAM R1 storage/region self-test and exit");
Define_bool_opt("--sram-r2-selftest", g_flag_sram_r2_selftest, false,
                "run SRAM R2 arbitration/hazard self-test and exit");
Define_bool_opt("--sram-r3-selftest", g_flag_sram_r3_selftest, false,
                "run SRAM R3 LSU HBM round-trip self-test and exit");
Define_bool_opt("--sram-r4-selftest", g_flag_sram_r4_selftest, false,
                "run SRAM R4 DTE HBM round-trip self-test and exit");
Define_bool_opt("--sram-r5-selftest", g_flag_sram_r5_selftest, false,
                "run SRAM R5 manual double-buffer self-test and exit");
Define_bool_opt("--sram-r6-selftest", g_flag_sram_r6_selftest, false,
                "run SRAM R6 compatibility migration self-test and exit");
Define_bool_opt("--hbm-contention-experiment",
                g_flag_hbm_contention_experiment, false,
                "run distributed HBM port/contention experiment and exit");
Define_string_opt("--hbm-experiment-port", g_flag_hbm_experiment_port, "N0",
                  "HBM experiment attachment port: N0..N3/S0..S3/E0..E3/W0..W3");
Define_int64_opt("--hbm-experiment-cores", g_flag_hbm_experiment_cores, 1,
                 "HBM experiment concurrent core count (1..8)");
Define_int64_opt("--hbm-experiment-pairs", g_flag_hbm_experiment_pairs, 8,
                 "HBM experiment write/read pairs per core");
Define_int64_opt("--hbm-experiment-bytes", g_flag_hbm_experiment_bytes, 1024,
                 "HBM experiment bytes per read or write");

Define_bool_opt("--dte-v3-selftest", g_flag_dte_v3_selftest, false,
                "run DTE V3a async token/dependency SystemC self-test and exit");

Define_bool_opt("--dte-v3b-selftest", g_flag_dte_v3b_selftest, false,
                "run DTE V3b aggregation SystemC self-test and exit");

Define_bool_opt("--dte-v4-selftest", g_flag_dte_v4_selftest, false,
                "run DTE V4 resource SystemC self-test and exit");

Define_bool_opt("--workload-rendezvous-selftest",
                g_flag_workload_rendezvous_selftest, false,
                "run workload rendezvous validation self-test and exit");

namespace {
const std::string kDefaultWorkloadConfig =
    std::string(NPUSIM_SOURCE_ROOT) + "/llm/test/default/workload.json";

constexpr uintmax_t kMaxLinkedProgramManifestBytes = uintmax_t{256} << 20;

constexpr bool FitsPublicFileLimit(uintmax_t size, uintmax_t limit) {
    return size <= limit;
}

static_assert(FitsPublicFileLimit(kMaxProgramFileBytes,
                                  kMaxProgramFileBytes));
static_assert(!FitsPublicFileLimit(kMaxProgramFileBytes + 1,
                                   kMaxProgramFileBytes));
static_assert(FitsPublicFileLimit(kMaxLinkedProgramManifestBytes,
                                  kMaxLinkedProgramManifestBytes));
static_assert(!FitsPublicFileLimit(kMaxLinkedProgramManifestBytes + 1,
                                   kMaxLinkedProgramManifestBytes));

std::vector<uint8_t> ReadProgramFile(const std::string &path) {
    const std::filesystem::path input(path);
    if (!std::filesystem::exists(input))
        throw std::runtime_error("program artifact does not exist: " + path);
    if (!std::filesystem::is_regular_file(input))
        throw std::runtime_error(
            "program artifact is not a regular file: " + path);
    const uintmax_t size = std::filesystem::file_size(input);
    if (size > kMaxProgramFileBytes)
        throw std::runtime_error("program artifact exceeds 64 MiB: " + path);
    std::ifstream stream(input, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open program artifact: " + path);
    std::vector<uint8_t> bytes(static_cast<std::size_t>(size));
    if (!bytes.empty() &&
        !stream.read(reinterpret_cast<char *>(bytes.data()), bytes.size()))
        throw std::runtime_error("cannot read program artifact: " + path);
    return bytes;
}

MoeSwizzleCalibrationKind ParseMoeSwizzleCalibrationKind(
    const std::string &value) {
    static const std::map<std::string, MoeSwizzleCalibrationKind> kinds{
        {"group_gemm", MoeSwizzleCalibrationKind::GROUP_GEMM},
        {"swiglu_group", MoeSwizzleCalibrationKind::SWIGLU_GROUP},
        {"dte_launch", MoeSwizzleCalibrationKind::DTE_LAUNCH},
        {"dte_sync", MoeSwizzleCalibrationKind::DTE_SYNC},
        {"dte_hop", MoeSwizzleCalibrationKind::DTE_HOP},
        {"session_open", MoeSwizzleCalibrationKind::SESSION_OPEN},
        {"session_retire", MoeSwizzleCalibrationKind::SESSION_RETIRE},
        {"local_copy", MoeSwizzleCalibrationKind::LOCAL_COPY},
        {"sram_alloc", MoeSwizzleCalibrationKind::SRAM_ALLOC},
        {"sram_bind", MoeSwizzleCalibrationKind::SRAM_BIND},
        {"sram_free", MoeSwizzleCalibrationKind::SRAM_FREE},
        {"event_set", MoeSwizzleCalibrationKind::EVENT_SET},
        {"event_wait", MoeSwizzleCalibrationKind::EVENT_WAIT},
        {"terminal_done", MoeSwizzleCalibrationKind::TERMINAL_DONE},
    };
    const auto found = kinds.find(value);
    if (found == kinds.end())
        throw std::invalid_argument(
            "unknown --moe-swizzle-calibration-kind");
    return found->second;
}

std::optional<std::array<uint64_t, 3>> ParseMoeSwizzleCalibrationShape(
    const std::string &value) {
    if (value == "none") return std::nullopt;
    std::array<uint64_t, 3> result{};
    std::istringstream input(value);
    char first = 0;
    char second = 0;
    if (!(input >> result[0] >> first >> result[1] >> second >> result[2]) ||
        first != 'x' || second != 'x' || input.peek() != EOF ||
        std::any_of(result.begin(), result.end(),
                    [](uint64_t item) { return item == 0; }))
        throw std::invalid_argument(
            "--moe-swizzle-calibration-shape must be none or positive MxNxK");
    return result;
}

std::string ReadRegularTextFile(
    const std::string &description, const std::string &path,
    uintmax_t max_bytes = kMaxProgramFileBytes) {
    const std::filesystem::path input(path);
    if (!std::filesystem::exists(input))
        throw std::runtime_error(description + " does not exist: " + path);
    if (!std::filesystem::is_regular_file(input))
        throw std::runtime_error(description + " is not a regular file: " +
                                 path);
    const uintmax_t size = std::filesystem::file_size(input);
    if (!FitsPublicFileLimit(size, max_bytes))
        throw std::runtime_error(
            description + " exceeds " +
            std::to_string(max_bytes >> 20) + " MiB: " + path);
    std::ifstream stream(input, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open " + description + ": " + path);
    std::string text(static_cast<std::size_t>(size), '\0');
    if (!text.empty() && !stream.read(text.data(), text.size()))
        throw std::runtime_error("cannot read " + description + ": " + path);
    return text;
}

std::vector<std::string> SplitSequencePaths(const std::string &value,
                                            const std::string &flag) {
    std::vector<std::string> result;
    std::size_t begin = 0;
    while (begin <= value.size()) {
        const std::size_t end = value.find(',', begin);
        const std::string item = value.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        if (item.empty())
            throw std::runtime_error(flag + " contains an empty path");
        result.push_back(item);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    if (result.size() < 2)
        throw std::runtime_error(flag + " requires at least two paths");
    return result;
}

struct DenseSequenceKvRange {
    std::string id;
    frontend::StateKindDto kind = frontend::StateKindDto::KV_KEY;
    uint64_t die_id = 0;
    uint64_t address = 0;
    uint64_t size_bytes = 0;
};

std::vector<DenseSequenceKvRange> DenseKvRanges(
    const std::string &manifest_text) {
    const frontend::LinkedProgramManifestDto manifest =
        frontend::ProgramArtifactFinalizer::Parse(manifest_text);
    std::map<std::string, DenseSequenceKvRange> unique;
    for (const frontend::LinkedFragmentDto &linked : manifest.fragments) {
        const frontend::CommandFragmentDto *fragment =
            std::get_if<frontend::CommandFragmentDto>(&linked);
        if (fragment == nullptr)
            fragment = &std::get<frontend::RegionManifestDto>(linked).fragment;
        for (const frontend::StateAbiDto &abi : fragment->state_abi) {
            if (abi.kind != frontend::StateKindDto::KV_KEY &&
                abi.kind != frontend::StateKindDto::KV_VALUE)
                continue;
            DenseSequenceKvRange range{
                abi.id, abi.kind, abi.die_id, abi.address, abi.size_bytes};
            auto [it, inserted] = unique.emplace(abi.id, range);
            if (!inserted &&
                (it->second.kind != range.kind ||
                 it->second.die_id != range.die_id ||
                 it->second.address != range.address ||
                 it->second.size_bytes != range.size_bytes))
                throw std::runtime_error(
                    "Dense sequence manifest has conflicting KV StateABI");
        }
    }
    std::vector<DenseSequenceKvRange> result;
    for (const auto &entry : unique) result.push_back(entry.second);
    std::sort(result.begin(), result.end(),
              [](const DenseSequenceKvRange &left,
                 const DenseSequenceKvRange &right) {
                  return std::tie(left.kind, left.die_id, left.address) <
                         std::tie(right.kind, right.die_id, right.address);
              });
    return result;
}

std::vector<DenseSequenceKvRange> DenseTrainingStateRanges(
    const std::string &manifest_text) {
    const frontend::LinkedProgramManifestDto manifest =
        frontend::ProgramArtifactFinalizer::Parse(manifest_text);
    std::map<std::string, DenseSequenceKvRange> unique;
    for (const frontend::LinkedFragmentDto &linked : manifest.fragments) {
        const frontend::CommandFragmentDto *fragment =
            std::get_if<frontend::CommandFragmentDto>(&linked);
        if (fragment == nullptr)
            fragment = &std::get<frontend::RegionManifestDto>(linked).fragment;
        // Composite inference can include MoE expert retention STATE_STOREs.
        // Those weights are read/write StateABIs, but they are not a Dense
        // optimizer family; preserve them in the executable and sidecar.
        if (manifest.producer_pass == "moe_full_model_region_linker" &&
            fragment->producer_pass == "flexible_moe_production_lowering")
            continue;
        for (const frontend::StateAbiDto &abi : fragment->state_abi) {
            if (abi.kind != frontend::StateKindDto::TRAINABLE_PARAMETER &&
                abi.kind != frontend::StateKindDto::OPTIMIZER_MASTER &&
                abi.kind != frontend::StateKindDto::OPTIMIZER_MOMENT1 &&
                abi.kind != frontend::StateKindDto::OPTIMIZER_MOMENT2 &&
                abi.kind != frontend::StateKindDto::OPTIMIZER_STEP)
                continue;
            DenseSequenceKvRange range{
                abi.id, abi.kind, abi.die_id, abi.address, abi.size_bytes};
            auto [it, inserted] = unique.emplace(abi.id, range);
            if (!inserted &&
                (it->second.kind != range.kind ||
                 it->second.die_id != range.die_id ||
                 it->second.address != range.address ||
                 it->second.size_bytes != range.size_bytes))
                throw std::runtime_error(
                    "Dense training sequence has conflicting StateABI");
        }
    }
    std::vector<DenseSequenceKvRange> result;
    for (const auto &entry : unique) result.push_back(entry.second);
    std::sort(result.begin(), result.end(),
              [](const DenseSequenceKvRange &left,
                 const DenseSequenceKvRange &right) {
                  return std::tie(left.die_id, left.address, left.id) <
                         std::tie(right.die_id, right.address, right.id);
              });
    return result;
}

struct ExternalAuthoritySpan {
    DenseSequenceKvRange state;
    std::vector<uint8_t> seed;
};

std::vector<ExternalAuthoritySpan> ExternalTrainingAuthority(
    const external_memory::ExternalDmaProgram &program,
    const std::vector<DenseSequenceKvRange> &states) {
    if (states.empty() || program.external_seeds.empty() ||
        program.external_probes.empty())
        throw std::runtime_error("external training StateABI source is empty");
    std::vector<ExternalAuthoritySpan> spans;
    bool nonzero = false;
    for (const auto &state : states) {
        std::vector<ExternalAuthoritySpan> candidates;
        for (const auto &descriptor : program.descriptors) {
            if (descriptor.direction !=
                    external_memory::TransferDirection::kExternalToHbm ||
                state.address < descriptor.hbm_address ||
                state.size_bytes > descriptor.size_bytes ||
                state.address - descriptor.hbm_address >
                    descriptor.size_bytes - state.size_bytes)
                continue;
            const auto connection = std::find_if(
                program.fabric.connections.begin(),
                program.fabric.connections.end(),
                [&](const auto &item) {
                    return item.id == descriptor.connection_ref &&
                           item.target_die_id == state.die_id;
                });
            if (connection == program.fabric.connections.end()) continue;
            const auto link = std::find_if(
                program.fabric.links.begin(), program.fabric.links.end(),
                [&](const auto &item) { return item.id == connection->link_ref; });
            if (link == program.fabric.links.end())
                throw std::runtime_error("external StateABI link disappeared");
            const uint64_t external_address = descriptor.external_address +
                state.address - descriptor.hbm_address;
            for (const auto &seed : program.external_seeds) {
                if (seed.external_capacity_ref != link->external_capacity_ref ||
                    external_address < seed.address ||
                    state.size_bytes > seed.payload.size() ||
                    external_address - seed.address >
                        seed.payload.size() - state.size_bytes)
                    continue;
                const auto begin = seed.payload.begin() +
                    static_cast<std::ptrdiff_t>(external_address - seed.address);
                ExternalAuthoritySpan candidate{
                    state, std::vector<uint8_t>(
                        begin, begin + static_cast<std::ptrdiff_t>(state.size_bytes))};
                const bool final_writeback = std::any_of(
                    program.descriptors.begin(), program.descriptors.end(),
                    [&](const auto &write) {
                        if (write.direction !=
                                external_memory::TransferDirection::kHbmToExternal ||
                            state.address < write.hbm_address ||
                            state.size_bytes > write.size_bytes ||
                            state.address - write.hbm_address >
                                write.size_bytes - state.size_bytes ||
                            write.external_address + state.address -
                                write.hbm_address != external_address)
                            return false;
                        const auto target = std::find_if(
                            program.fabric.connections.begin(),
                            program.fabric.connections.end(),
                            [&](const auto &item) {
                                return item.id == write.connection_ref &&
                                       item.target_die_id == state.die_id &&
                                       item.link_ref == connection->link_ref;
                            });
                        return target != program.fabric.connections.end();
                    });
                const bool final_probe = std::any_of(
                    program.external_probes.begin(), program.external_probes.end(),
                    [&](const auto &probe) {
                        return probe.external_capacity_ref ==
                                link->external_capacity_ref &&
                               external_address >= probe.address &&
                               state.size_bytes <= probe.expected_payload.size() &&
                               external_address - probe.address <=
                                   probe.expected_payload.size() - state.size_bytes;
                    });
                if (final_writeback && final_probe)
                    candidates.push_back(std::move(candidate));
            }
        }
        if (candidates.size() != 1)
            throw std::runtime_error(
                "external training StateABI lacks exactly one signed "
                "seeded E2H and same-authority H2E/probe: " + state.id);
        nonzero |= std::any_of(candidates.front().seed.begin(),
                               candidates.front().seed.end(),
                               [](uint8_t value) { return value != 0; });
        spans.push_back(std::move(candidates.front()));
    }
    if (!nonzero)
        throw std::runtime_error(
            "external training restore lacks a nonzero payload witness");
    return spans;
}

uint64_t InspectExternalTrainingHbm(
    HBMRuntime &hbm, const std::vector<ExternalAuthoritySpan> &spans,
    bool expect_restored) {
    uint64_t present = 0;
    for (const auto &span : spans) {
        const auto snapshot = hbm.DebugPeek(
            span.state.address, static_cast<int>(span.state.die_id),
            span.state.size_bytes);
        if (snapshot.payload.size() != span.seed.size())
            throw std::runtime_error("external StateABI HBM snapshot truncated");
        for (const auto &chunk : snapshot.chunks)
            for (const uint8_t bit : chunk.backend.present)
                present += bit != 0;
        if (expect_restored && snapshot.payload != span.seed)
            throw std::runtime_error(
                "external DMA restored wrong physical HBM StateABI payload: " +
                span.state.id);
    }
    const uint64_t bytes = std::accumulate(
        spans.begin(), spans.end(), uint64_t{0},
        [](uint64_t total, const auto &span) {
            return total + span.state.size_bytes;
        });
    if (present != (expect_restored ? bytes : 0))
        throw std::runtime_error(expect_restored
            ? "external DMA did not physically restore every StateABI byte"
            : "external training StateABI was already present in HBM before DMA");
    return bytes;
}
void ValidateDenseKvContinuity(
    const std::vector<std::vector<DenseSequenceKvRange>> &segments) {
    if (segments.size() < 2 || segments[0].empty())
        throw std::runtime_error("Dense sequence requires nonempty KV layouts");
    for (std::size_t segment = 1; segment < segments.size(); ++segment) {
        if (segments[segment].size() != segments[0].size())
            throw std::runtime_error(
                "Dense sequence KV StateABI cardinality changed");
        for (std::size_t index = 0; index < segments[0].size(); ++index) {
            const auto &first = segments[0][index];
            const auto &next = segments[segment][index];
            if (first.kind != next.kind || first.die_id != next.die_id ||
                first.address != next.address ||
                segments[segment - 1][index].size_bytes >= next.size_bytes)
                throw std::runtime_error(
                    "Dense sequence KV address/extent continuity failed");
        }
    }
}

void ValidateDenseTrainingStateContinuity(
    const std::vector<std::vector<DenseSequenceKvRange>> &segments) {
    if (segments.size() != 2 || segments[0].empty() ||
        segments[1].size() != segments[0].size())
        throw std::runtime_error(
            "Dense training sequence requires two matching state layouts");
    for (std::size_t index = 0; index < segments[0].size(); ++index) {
        const auto &first = segments[0][index];
        const auto &next = segments[1][index];
        if (first.kind != next.kind || first.die_id != next.die_id ||
            first.address != next.address ||
            first.size_bytes != next.size_bytes)
            throw std::runtime_error(
                "Dense training sequence state layout continuity failed");
    }
}

struct MoeForwardProgramWitness {
    std::set<std::string> route_state_refs;
    std::size_t trainable_states = 0;
    std::size_t records = 0;
};

MoeForwardProgramWitness ValidateMoeForwardProgram(
    const std::string &manifest_text,
    const std::vector<DenseSequenceKvRange> &ranges) {
    const frontend::LinkedProgramManifestDto manifest =
        frontend::ProgramArtifactFinalizer::Parse(manifest_text);
    const std::map<std::string, std::size_t> required_fragments{
        {"coarse_lowering", 20}, {"state_dma_lowering", 21},
        {"moe_full_train_router_lowering", 2},
        {"moe_full_train_route_freeze_lowering", 2},
        {"moe_full_train_dispatch_lowering", 2},
        {"moe_full_train_expert_lowering", 2},
        {"moe_full_train_combine_lowering", 2},
    };
    const std::map<Opcode, std::size_t> required_records{
        {Opcode::SRAM_ALLOC_AT, 57}, {Opcode::SRAM_FREE, 57},
        {Opcode::SRAM_BIND, 32}, {Opcode::LSU_LOAD, 21},
        {Opcode::MATMUL, 13}, {Opcode::RMSNORM, 5},
        {Opcode::RESIDUAL, 4}, {Opcode::DTE_ISSUE, 4},
        {Opcode::DTE_WAIT, 4}, {Opcode::ATTENTION_EXACT, 2},
        {Opcode::SWIGLU, 2}, {Opcode::MOE_SCORE_WEIGHTED_FORWARD, 2},
        {Opcode::ROPE_QK_EXACT, 2}, {Opcode::EMBEDDING_LOOKUP, 1},
        {Opcode::CROSS_ENTROPY_FORWARD, 1},
    };
    std::map<std::string, std::size_t> fragments;
    std::map<Opcode, std::size_t> records;
    std::map<std::string, frontend::StateKindDto> states;
    for (const frontend::LinkedFragmentDto &linked : manifest.fragments) {
        const frontend::CommandFragmentDto *fragment =
            std::get_if<frontend::CommandFragmentDto>(&linked);
        if (fragment == nullptr)
            fragment = &std::get<frontend::RegionManifestDto>(linked).fragment;
        ++fragments[fragment->producer_pass];
        for (const frontend::StateAbiDto &state : fragment->state_abi) {
            const auto [it, inserted] = states.emplace(state.id, state.kind);
            if (!inserted && it->second != state.kind)
                throw std::runtime_error(
                    "MoE forward sequence has conflicting StateABI kind");
        }
        for (const auto &stream : fragment->core_streams)
            for (const auto &record : stream.records) ++records[record.opcode];
    }
    MoeForwardProgramWitness result;
    result.records = 207;
    for (const auto &[ref, kind] : states) {
        if (kind == frontend::StateKindDto::MOE_STATIC_ROUTE)
            result.route_state_refs.insert(ref);
        else if (kind == frontend::StateKindDto::TRAINABLE_PARAMETER)
            ++result.trainable_states;
        else
            throw std::runtime_error(
                "MoE forward sequence has optimizer or unsupported StateABI");
    }
    if (manifest.producer_pass != "manifest_linker" ||
        fragments != required_fragments || records != required_records ||
        result.trainable_states != 19 || ranges.size() != 19 ||
        result.route_state_refs.size() != 2 ||
        manifest.core_streams.size() != 1)
        throw std::runtime_error(
            "MoE forward sequence requires exact EP1 two-layer forward-only records and states");
    return result;
}

MoeForwardProgramWitness ValidateMoeSgdPartialProgram(
    const std::string &manifest_text,
    const std::vector<DenseSequenceKvRange> &ranges,
    bool layer1_parameter_sgd) {
    const frontend::LinkedProgramManifestDto manifest =
        frontend::ProgramArtifactFinalizer::Parse(manifest_text);
    std::map<Opcode, std::size_t> records;
    std::map<std::string, frontend::StateKindDto> states;
    std::map<std::string, const frontend::StateAbiDto *> state_abis;
    for (const frontend::LinkedFragmentDto &linked : manifest.fragments) {
        const frontend::CommandFragmentDto *fragment =
            std::get_if<frontend::CommandFragmentDto>(&linked);
        if (fragment == nullptr)
            fragment = &std::get<frontend::RegionManifestDto>(linked).fragment;
        for (const frontend::StateAbiDto &state : fragment->state_abi) {
            const auto [it, inserted] = states.emplace(state.id, state.kind);
            state_abis.emplace(state.id, &state);
            if (!inserted && it->second != state.kind)
                throw std::runtime_error(
                    "MoE partial sequence has conflicting StateABI kind");
        }
        for (const auto &stream : fragment->core_streams)
            for (const auto &record : stream.records) ++records[record.opcode];
    }
    MoeForwardProgramWitness result;
    result.records = layer1_parameter_sgd ? 340 : 254;
    for (const auto &[ref, kind] : states) {
        if (kind == frontend::StateKindDto::MOE_STATIC_ROUTE)
            result.route_state_refs.insert(ref);
        else if (kind == frontend::StateKindDto::TRAINABLE_PARAMETER)
            ++result.trainable_states;
        else
            throw std::runtime_error(
                "MoE partial sequence has unsupported optimizer StateABI");
    }
    std::size_t total_records = 0;
    for (const auto &[opcode, count] : records) total_records += count;
    if (layer1_parameter_sgd) {
        std::set<std::string> stored_state_refs;
        std::vector<uint64_t> stored_sizes;
        for (const auto &binding : manifest.state_operand_bindings) {
            const auto fragment = std::find_if(
                manifest.fragments.begin(), manifest.fragments.end(),
                [&](const frontend::LinkedFragmentDto &linked) {
                    const auto *item =
                        std::get_if<frontend::CommandFragmentDto>(&linked);
                    if (item == nullptr)
                        item = &std::get<frontend::RegionManifestDto>(linked).fragment;
                    return item->id == binding.fragment_id;
                });
            if (fragment == manifest.fragments.end())
                throw std::runtime_error(
                    "MoE layer1 SGD STORE references unknown fragment");
            const auto *item =
                std::get_if<frontend::CommandFragmentDto>(&*fragment);
            if (item == nullptr)
                item = &std::get<frontend::RegionManifestDto>(*fragment).fragment;
            const auto stream = std::find_if(
                item->core_streams.begin(), item->core_streams.end(),
                [&](const auto &entry) {
                    return entry.logical_core == binding.logical_core;
                });
            if (stream == item->core_streams.end() ||
                binding.fragment_record_index >= stream->records.size())
                throw std::runtime_error(
                    "MoE layer1 SGD STORE references unknown core record");
            if (stream->records[binding.fragment_record_index].opcode !=
                Opcode::LSU_STORE)
                continue;
            const auto state = state_abis.find(binding.state_abi_id);
            if (state == state_abis.end() ||
                binding.operand_id != SemanticOperandId::HBM_ADDRESS ||
                state->second->kind !=
                    frontend::StateKindDto::TRAINABLE_PARAMETER ||
                state->second->access != frontend::StateAccessDto::READ_WRITE ||
                state->second->dtype != frontend::BufferDTypeDto::FP16 ||
                state->second->die_id != 0 ||
                !stored_state_refs.insert(state->second->state_ref).second)
                throw std::runtime_error(
                    "MoE layer1 SGD STORE lacks unique FP16 trainable StateABI");
            stored_sizes.push_back(state->second->size_bytes);
        }
        std::sort(stored_sizes.begin(), stored_sizes.end());
        if (stored_sizes != std::vector<uint64_t>{8, 64, 64, 64})
            throw std::runtime_error(
                "MoE layer1 SGD STORE has wrong four-state byte closure");
    }
    if (manifest.producer_pass != "manifest_linker" ||
        total_records != result.records || manifest.core_streams.size() != 1 ||
        result.trainable_states != 19 || result.route_state_refs.size() != 2 ||
        ranges.size() != 19 ||
        records[Opcode::GEMM_WEIGHT_WGRAD_TIMING] !=
            (layer1_parameter_sgd ? 5 : 2) ||
        records[Opcode::MOE_SCORE_WEIGHT_BACKWARD] != 1 ||
        records[Opcode::CROSS_ENTROPY_BACKWARD] != 1 ||
        records[Opcode::SGD_UPDATE] != (layer1_parameter_sgd ? 4 : 1) ||
        records[Opcode::LSU_STORE] != (layer1_parameter_sgd ? 4 : 1) ||
        (layer1_parameter_sgd &&
         (records[Opcode::GEMM_DX_TIMING] != 5 ||
          records[Opcode::NORM_GAMMA_WGRAD_TIMING] != 2 ||
          records[Opcode::RMSNORM_BACKWARD_TIMING] != 2 ||
          records[Opcode::RESIDUAL] != 6 ||
          records[Opcode::SWIGLU_BACKWARD_TIMING] != 1 ||
          records[Opcode::LOCAL_REDUCE] != 1)))
        throw std::runtime_error(
            "MoE partial SGD sequence needs exact physical backward/update/store witness");
    return result;
}

struct DenseTrainingProgramWitness {
    std::size_t matmul_records = 0;
    std::size_t sgd_records = 0;
    std::size_t adamw_records = 0;
    std::size_t load_records = 0;
    std::size_t store_records = 0;
    std::size_t trainable_states = 0;
    std::size_t optimizer_states = 0;
};

DenseTrainingProgramWitness DenseTrainingWitness(
    const std::string &manifest_text,
    const std::vector<DenseSequenceKvRange> &ranges) {
    const frontend::LinkedProgramManifestDto manifest =
        frontend::ProgramArtifactFinalizer::Parse(manifest_text);
    DenseTrainingProgramWitness result;
    std::map<frontend::StateKindDto, std::size_t> state_kinds;
    for (const DenseSequenceKvRange &range : ranges) ++state_kinds[range.kind];
    result.trainable_states =
        state_kinds[frontend::StateKindDto::TRAINABLE_PARAMETER];
    result.optimizer_states = ranges.size() - result.trainable_states;
    const bool adamw = result.optimizer_states != 0;
    if (adamw) {
        // DP1 gate/up are two logical parameters in each of two layers but
        // occupy only one packed physical FP16 trainable carrier per layer.
        if (result.trainable_states != 15 || result.optimizer_states != 68)
            throw std::runtime_error(
                "Dense AdamW DP1 requires 15 trainable carriers and 17 logical four-state groups");
        const std::vector<std::pair<frontend::StateKindDto, std::string>> roles{
            {frontend::StateKindDto::OPTIMIZER_MASTER,
             "optimizer.adamw.master."},
            {frontend::StateKindDto::OPTIMIZER_MOMENT1,
             "optimizer.adamw.m."},
            {frontend::StateKindDto::OPTIMIZER_MOMENT2,
             "optimizer.adamw.v."},
            {frontend::StateKindDto::OPTIMIZER_STEP,
             "optimizer.adamw.step."},
        };
        std::optional<std::set<std::string>> parameter_refs;
        for (const auto &[kind, prefix] : roles) {
            std::set<std::string> names;
            for (const frontend::LinkedFragmentDto &linked : manifest.fragments) {
                const frontend::CommandFragmentDto *fragment =
                    std::get_if<frontend::CommandFragmentDto>(&linked);
                if (fragment == nullptr)
                    fragment = &std::get<frontend::RegionManifestDto>(linked).fragment;
                for (const frontend::StateAbiDto &state : fragment->state_abi) {
                    if (state.kind != kind) continue;
                    if (state.state_ref.rfind(prefix, 0) != 0 ||
                        !names.emplace(state.state_ref.substr(prefix.size())).second)
                        throw std::runtime_error(
                            "Dense AdamW has repeated/malformed optimizer StateABI");
                }
            }
            if (names.size() != 17 ||
                (parameter_refs.has_value() && names != *parameter_refs))
                throw std::runtime_error(
                    "Dense AdamW four states must cover the same 17 logical parameters");
            parameter_refs = std::move(names);
        }
    }
    for (const frontend::LinkedFragmentDto &linked : manifest.fragments) {
        const frontend::CommandFragmentDto *fragment =
            std::get_if<frontend::CommandFragmentDto>(&linked);
        if (fragment == nullptr)
            fragment = &std::get<frontend::RegionManifestDto>(linked).fragment;
        for (const auto &stream : fragment->core_streams) {
            for (const auto &record : stream.records) {
                if (record.opcode == Opcode::MATMUL)
                    ++result.matmul_records;
                else if (record.opcode == Opcode::SGD_UPDATE)
                    ++result.sgd_records;
                else if (record.opcode == Opcode::ADAMW_UPDATE)
                    ++result.adamw_records;
                else if (record.opcode == Opcode::LSU_LOAD)
                    ++result.load_records;
                else if (record.opcode == Opcode::LSU_STORE)
                    ++result.store_records;
            }
        }
    }
    if (!result.trainable_states ||
        result.matmul_records < result.trainable_states ||
        (adamw &&
         (result.sgd_records != 0 ||
          result.adamw_records != result.optimizer_states / 4 ||
          result.load_records != ranges.size() ||
          result.store_records != ranges.size())) ||
        (!adamw &&
         (result.adamw_records != 0 ||
          result.sgd_records != result.trainable_states ||
          result.store_records != result.trainable_states)))
        throw std::runtime_error(
            "Dense training sequence lacks exact WGRAD/optimizer/state load-store coverage");
    return result;
}

struct DenseSequenceBoundary {
    std::size_t bytes = 0;
    std::string digest;
};

DenseSequenceBoundary ReadDenseStateBoundary(
    HBMRuntime &runtime,
    const std::vector<DenseSequenceKvRange> &ranges) {
    std::vector<uint8_t> aggregate;
    for (const auto &range : ranges) {
        const HBMRuntimeDebugSnapshot snapshot = runtime.DebugPeek(
            range.address, static_cast<int>(range.die_id), range.size_bytes);
        const bool present = std::all_of(
            snapshot.chunks.begin(), snapshot.chunks.end(),
            [](const HBMRuntimeDebugChunkSnapshot &chunk) {
                return chunk.backend.present.size() ==
                           chunk.backend.payload.size() &&
                       std::all_of(chunk.backend.present.begin(),
                                   chunk.backend.present.end(),
                                   [](uint8_t value) { return value != 0; });
            });
        if (!present)
            throw std::runtime_error(
                "Dense sequence state contains unwritten bytes");
        aggregate.insert(aggregate.end(), snapshot.payload.begin(),
                         snapshot.payload.end());
    }
    return DenseSequenceBoundary{
        aggregate.size(), frontend::program_io::Sha256Hex(aggregate)};
}

void PrintDenseKvBoundary(
    HBMRuntime &runtime, std::size_t segment,
    const std::vector<DenseSequenceKvRange> &ranges) {
    const DenseSequenceBoundary boundary =
        ReadDenseStateBoundary(runtime, ranges);
    std::cout << "[DENSE_SEQUENCE_KV] index=" << segment
              << " bytes=" << boundary.bytes
              << " digest=" << boundary.digest
              << " pass=1" << std::endl;
}

std::string PrintDenseTrainingStateBoundary(
    HBMRuntime &runtime, std::size_t version,
    const std::vector<DenseSequenceKvRange> &ranges,
    const std::optional<std::string> &previous_digest) {
    const DenseSequenceBoundary boundary =
        ReadDenseStateBoundary(runtime, ranges);
    const bool changed = previous_digest.has_value() &&
                         *previous_digest != boundary.digest;
    std::cout << "[DENSE_TRAINING_SEQUENCE_STATE] version=" << version
              << " bytes=" << boundary.bytes
              << " digest=" << boundary.digest
              << " content_changed=" << (changed ? 1 : 0)
              << " functional=0 pass=1" << std::endl;
    return boundary.digest;
}

std::string PrintMoeRouterSgdPartialStateBoundary(
    HBMRuntime &runtime, std::size_t version,
    const std::vector<DenseSequenceKvRange> &ranges,
    const std::optional<std::string> &previous_digest) {
    const DenseSequenceBoundary boundary = ReadDenseStateBoundary(runtime, ranges);
    const bool changed = previous_digest.has_value() &&
                         *previous_digest != boundary.digest;
    std::cout << "[MOE_ROUTER_SGD_PARTIAL_STATE] version=" << version
              << " bytes=" << boundary.bytes
              << " digest=" << boundary.digest
              << " content_changed=" << (changed ? 1 : 0)
              << " functional=0 full_training=0 pass=1" << std::endl;
    return boundary.digest;
}

std::string PrintMoeLayer1SgdPartialStateBoundary(
    HBMRuntime &runtime, std::size_t version,
    const std::vector<DenseSequenceKvRange> &ranges,
    const std::optional<std::string> &previous_digest) {
    const DenseSequenceBoundary boundary = ReadDenseStateBoundary(runtime, ranges);
    const bool changed = previous_digest.has_value() &&
                         *previous_digest != boundary.digest;
    std::cout << "[MOE_LAYER1_SGD_PARTIAL_STATE] version=" << version
              << " bytes=" << boundary.bytes
              << " digest=" << boundary.digest
              << " content_changed=" << (changed ? 1 : 0)
              << " functional=0 full_training=0 pass=1" << std::endl;
    return boundary.digest;
}

const char *ProgramIoModeName(frontend::program_io::Mode mode) {
    switch (mode) {
    case frontend::program_io::Mode::TIMING:
        return "timing";
    case frontend::program_io::Mode::FUNCTIONAL:
        return "functional";
    }
    return "unknown";
}

void PrintProgramIoStatus(const std::string &phase, const std::string &mode,
                          std::size_t initialization_count,
                          std::size_t probe_count,
                          const std::string &checksum, bool passed) {
    std::cout << "[PROGRAM_IO] phase=" << phase << " mode=" << mode
              << " initializations=" << initialization_count
              << " probes=" << probe_count << " checksum=" << checksum
              << " pass=" << (passed ? 1 : 0) << "\n";
}

void ValidateIsaV1StartupInvariants() {
    std::string error;
    if (!ValidateOpcodeManifest(&error))
        throw std::logic_error("invalid external opcode manifest: " + error);
    error.clear();
    if (!ValidatePrimManifest(&error))
        throw std::logic_error("invalid internal Prim manifest: " + error);

    const std::vector<int> ids =
        PrimFactory::getInstance().registeredIds();
    if (ids.size() != kPrimManifestSize)
        throw std::logic_error(
            "PrimFactory registration count disagrees with frozen manifest");
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] != static_cast<int>(i + 1))
            throw std::logic_error(
                "PrimFactory registrations are not the frozen contiguous IDs");
    }
}

void CollectD2DLinkUnits(const std::vector<sc_object *> &objects,
                         std::vector<D2DLinkUnit *> &links) {
    for (sc_object *object : objects) {
        if (auto *link = dynamic_cast<D2DLinkUnit *>(object))
            links.push_back(link);
        CollectD2DLinkUnits(object->get_child_objects(), links);
    }
}

std::string MoeSwizzleDirection(Directions direction) {
    switch (direction) {
    case EAST:
        return "x+";
    case WEST:
        return "x-";
    case NORTH:
        return "y+";
    case SOUTH:
        return "y-";
    default:
        throw std::runtime_error(
            "MoE Swizzle port-time link direction is not physical");
    }
}

std::vector<D2DDirectionalDataServiceTrace>
CollectMoeSwizzlePortTimeTraces() {
    if (g_d2d_links.size() != 8)
        throw std::runtime_error(
            "MoE Swizzle port-time requires exact eight 2x2 directed links");
    std::vector<D2DLinkUnit *> units;
    CollectD2DLinkUnits(sc_get_top_level_objects(), units);
    if (units.size() != g_d2d_links.size())
        throw std::runtime_error(
            "MoE Swizzle port-time link instance count drifted");
    std::set<int> indices;
    std::vector<D2DDirectionalDataServiceTrace> traces;
    traces.reserve(units.size());
    for (const D2DLinkUnit *unit : units) {
        if (unit->link_idx < 0 ||
            unit->link_idx >= static_cast<int>(g_d2d_links.size()) ||
            !indices.insert(unit->link_idx).second)
            throw std::runtime_error(
                "MoE Swizzle port-time link index is missing or duplicated");
        const D2DLink &link = g_d2d_links.at(unit->link_idx);
        if (link.local_die < 0 || link.local_die > 3 ||
            link.remote_die < 0 || link.remote_die > 3 ||
            link.local_port < 0 ||
            link.local_port >= static_cast<int>(g_die_ports.ports.size()))
            throw std::runtime_error(
                "MoE Swizzle port-time physical link binding is invalid");
        traces.push_back(D2DDirectionalDataServiceTrace{
            static_cast<uint16_t>(link.local_die),
            static_cast<uint16_t>(link.remote_die),
            MoeSwizzleDirection(g_die_ports.ports.at(link.local_port).dir),
            unit->DataServiceIntervalsComplete(),
            unit->DataServiceIntervals()});
    }
    return traces;
}

MoeSwizzleRuntimeManifestCounts CountMoeSwizzleRuntimeManifest(
    const ProgramArtifact &artifact,
    const frontend::LinkedProgramManifestDto &manifest) {
    MoeSwizzleRuntimeManifestCounts result;
    for (const ProgramCore &core : artifact.cores) {
        for (const ExternalRecord &record : core.records) {
            const OpcodeManifestEntry *entry = LookupOpcode(record.opcode);
            if (entry == nullptr)
                throw std::runtime_error(
                    "calibration manifest contains an unknown opcode");
            if (entry->category == OpcodeCategory::COMPUTE)
                ++result.compute_record_count;
            switch (record.opcode) {
            case Opcode::MATMUL:
            case Opcode::MOE_MATMUL:
                ++result.group_gemm_primitives;
                break;
            case Opcode::SWIGLU: {
                ++result.swiglu_primitives;
                const auto *operands =
                    std::get_if<ComputeOperands>(&record.operands);
                if (operands == nullptr || operands->parameters.size() != 1)
                    throw std::runtime_error(
                        "SWIGLU calibration record operands are not canonical");
                if (result.swiglu_primitives != 1)
                    break;
                if (operands->datatype == ExternalDataType::FP16)
                    ++result.swiglu_fp16_primitives;
                result.swiglu_runtime_core = core.core_id;
                result.swiglu_flattened_elements = operands->parameters[0];
                result.swiglu_input_bytes =
                    result.swiglu_flattened_elements * 4;
                result.swiglu_output_bytes =
                    result.swiglu_flattened_elements * 2;
                break;
            }
            case Opcode::DTE_SEND:
            case Opcode::DTE_RECV:
                ++result.dte_launch_count;
                ++result.dte_record_count;
                ++result.endpoint_session_count;
                break;
            case Opcode::DTE_ISSUE:
                ++result.dte_launch_count;
                ++result.dte_record_count;
                break;
            case Opcode::EVENT_SET:
            case Opcode::EVENT_WAIT:
                ++result.event_record_count;
                break;
            default:
                break;
            }
        }
    }
    for (const frontend::LinkedFragmentDto &linked : manifest.fragments) {
        const frontend::CommandFragmentDto *fragment =
            std::get_if<frontend::CommandFragmentDto>(&linked);
        if (fragment == nullptr)
            fragment = &std::get<frontend::RegionManifestDto>(linked).fragment;
        for (const frontend::BufferAbiDto &buffer : fragment->buffer_abi) {
            if (!buffer.alias_of.has_value()) ++result.physical_root_count;
        }
    }
    return result;
}

struct DenseSequenceProgramWitness {
    std::size_t compute_records = 0;
    std::size_t lsu_load_records = 0;
};

DenseSequenceProgramWitness CountDenseSequenceProgramRecords(
    const ProgramArtifact &artifact) {
    DenseSequenceProgramWitness result;
    for (const ProgramCore &core : artifact.cores) {
        for (const ExternalRecord &record : core.records) {
            const OpcodeManifestEntry *entry = LookupOpcode(record.opcode);
            if (entry == nullptr)
                throw std::runtime_error(
                    "Dense sequence contains an unknown opcode");
            if (entry->category == OpcodeCategory::COMPUTE)
                ++result.compute_records;
            if (record.opcode == Opcode::LSU_LOAD)
                ++result.lsu_load_records;
        }
    }
    if (result.compute_records == 0 || result.lsu_load_records == 0)
        throw std::runtime_error(
            "Dense sequence lacks compute or parameter-load records");
    return result;
}

class ExternalDmaStartupCoordinator : public sc_module {
public:
    SC_HAS_PROCESS(ExternalDmaStartupCoordinator);

    ExternalDmaStartupCoordinator(
        const sc_module_name &name,
        external_memory::ExternalDmaProgramExecutor &executor,
        external_memory::ExternalDmaRuntimePhaseMode phase_mode)
        : sc_module(name), executor_(executor), phase_mode_(phase_mode) {
        SC_THREAD(Run);
    }

    const sc_event &ReadyEvent() const { return ready_; }
    const std::optional<external_memory::ExternalDmaProgramExecution> &
    Execution() const { return execution_; }
    const std::string &Error() const { return error_; }
    void ReleaseFinalWriteback() { executor_.ReleaseFinalWriteback(); }
    bool BringInReady() const { return bring_in_ready_; }
    void ReleaseComputeAfterRestore() {
        if (!bring_in_ready_ || compute_released_)
            throw std::runtime_error(
                "external authority compute released before bring-in/restore");
        compute_released_ = true;
        compute_gate_.notify(SC_ZERO_TIME);
    }

private:
    void Run() {
        try {
            if (phase_mode_ == external_memory::ExternalDmaRuntimePhaseMode::
                                   kBringInThenFinalWriteback) {
                const auto stats = executor_.WaitForBringIn();
                if (stats.submitted_requests == 0 ||
                    stats.submitted_requests != stats.completed_requests ||
                    stats.failed_requests != 0 ||
                    stats.external_write_bytes != 0 ||
                    stats.hbm_read_bytes != 0)
                    throw std::runtime_error(
                        "external DMA bring-in phase did not complete and drain");
                std::cout
                    << "[EXTERNAL_DMA_READY] program="
                    << executor_.ProgramRef()
                    << " completed=" << stats.completed_requests
                    << " external_read_bytes=" << stats.external_read_bytes
                    << " hbm_write_bytes=" << stats.hbm_write_bytes
                    << " pending=0" << std::endl;
                // Pause with workers still blocked on ReadyEvent: the host
                // inspects the actual HBM backing before compute may start.
                bring_in_ready_ = true;
                sc_pause();
                wait(compute_gate_);
                ready_.notify(SC_ZERO_TIME);
                execution_ = executor_.Wait();
                ValidateFinalExecution();
                sc_pause();
                return;
            }
            execution_ = executor_.Wait();
            ValidateFinalExecution();
            std::cout
                << "[EXTERNAL_DMA_READY] program="
                << execution_->program_ref
                << " completed=" << execution_->stats.completed_requests
                << " external_read_bytes="
                << execution_->stats.external_read_bytes
                << " hbm_write_bytes=" << execution_->stats.hbm_write_bytes
                << " pending=0" << std::endl;
            ready_.notify(SC_ZERO_TIME);
        } catch (const std::exception &error) {
            error_ = error.what();
            sc_stop();
        }
    }

    void ValidateFinalExecution() const {
        if (!execution_.has_value() || !execution_->completed ||
            !execution_->error.empty() || execution_->pending_requests != 0 ||
            std::any_of(
                execution_->probes.begin(), execution_->probes.end(),
                [](const external_memory::DmaProbeResult &probe) {
                    return !probe.matched;
                }))
            throw std::runtime_error(
                "external DMA program did not complete and drain");
    }

    external_memory::ExternalDmaProgramExecutor &executor_;
    external_memory::ExternalDmaRuntimePhaseMode phase_mode_;
    std::optional<external_memory::ExternalDmaProgramExecution> execution_;
    std::string error_;
    sc_event ready_;
    sc_event compute_gate_;
    bool bring_in_ready_ = false;
    bool compute_released_ = false;
};
} // namespace

int sc_main(int argc, char *argv[]) {
    clock_t start = clock();

    srand((unsigned)time(NULL));
    std::cout.setf(std::ios::unitbuf);

    // 解析参数
    simple_flags::parse_args(argc, argv);
    if (!simple_flags::get_unknown_flags().empty()) {
        string content;
        for (auto it : simple_flags::get_unknown_flags()) {
            content += "'" + it + "', ";
        }
        content.resize(content.size() - 2); // remove last ', '
        content.append(".");
        LOG_ERROR(CONFIG) << "Unknown option(s): " << content;
        return -1;
    }

    if (g_flag_help) {
        simple_flags::print_args_info();
        return 0;
    }

    const bool adamw_paged = !g_flag_dense_adamw_paged_runtime.empty();
    const bool inference_paged =
        !g_flag_dense_inference_paged_runtime.empty();
    const bool moe_inference_paged =
        !g_flag_moe_inference_paged_runtime.empty();
    if (static_cast<int>(adamw_paged) +
            static_cast<int>(inference_paged) +
            static_cast<int>(moe_inference_paged) > 1) {
        LOG_ERROR(CONFIG) << "only one source-signed mid-program pager may be bound";
        return 2;
    }
    const bool sequence_any = !g_flag_program_sequence.empty() ||
                              !g_flag_linked_manifest_sequence.empty() ||
                              !g_flag_program_io_sequence.empty() ||
                              adamw_paged || inference_paged ||
                              moe_inference_paged ||
                              g_flag_moe_forward_sequence ||
                              g_flag_moe_router_sgd_partial_sequence ||
                              g_flag_moe_layer1_sgd_partial_sequence;
    if (sequence_any &&
        (g_flag_program_sequence.empty() ||
         g_flag_linked_manifest_sequence.empty() ||
         (adamw_paged ? !g_flag_program_io_sequence.empty()
                      : g_flag_program_io_sequence.empty()))) {
        LOG_ERROR(CONFIG) << "--program-sequence and --linked-manifest-sequence "
                             "require exactly one of --program-io-sequence "
                             "or --dense-adamw-paged-runtime";
        return 2;
    }
    const bool sequence_mode = sequence_any;
    const bool moe_forward_sequence = g_flag_moe_forward_sequence;
    const bool moe_router_sgd_partial_sequence =
        g_flag_moe_router_sgd_partial_sequence;
    const bool moe_layer1_sgd_partial_sequence =
        g_flag_moe_layer1_sgd_partial_sequence;
    const bool moe_partial_sgd_sequence =
        moe_router_sgd_partial_sequence || moe_layer1_sgd_partial_sequence;
    if (static_cast<int>(moe_forward_sequence) +
        static_cast<int>(moe_router_sgd_partial_sequence) +
        static_cast<int>(moe_layer1_sgd_partial_sequence) > 1) {
        LOG_ERROR(CONFIG) << "MoE forward/router/layer1 SGD partial sequence modes are exclusive";
        return 2;
    }
    if ((moe_forward_sequence || moe_partial_sgd_sequence) &&
        (adamw_paged || inference_paged || moe_inference_paged ||
         !g_flag_external_dma_binding.empty())) {
        LOG_ERROR(CONFIG) << "MoE forward-only sequence forbids pager and external DMA";
        return 2;
    }
    const bool external_dma_requested =
        !g_flag_external_dma_binding.empty();
    if (external_dma_requested && !sequence_mode) {
        LOG_ERROR(CONFIG)
            << "--external-dma-binding requires --program-sequence";
        return 2;
    }
    if ((adamw_paged || inference_paged || moe_inference_paged) &&
        external_dma_requested) {
        LOG_ERROR(CONFIG) << "on-demand paged DMA cannot share an all-state "
                             "startup/final-writeback phase";
        return 2;
    }
    if (sequence_mode &&
        (!g_flag_program.empty() || !g_flag_linked_manifest.empty() ||
         !g_flag_program_io.empty() || g_flag_program_one_shot)) {
        LOG_ERROR(CONFIG) << "Dense program sequence flags are mutually "
                             "exclusive with single-program flags";
        return 2;
    }

    const bool program_io_any =
        !g_flag_linked_manifest.empty() || !g_flag_program_io.empty();
    if (program_io_any &&
        (g_flag_linked_manifest.empty() || g_flag_program_io.empty())) {
        LOG_ERROR(CONFIG)
            << "--linked-manifest and --program-io must be provided together";
        PrintProgramIoStatus("preflight", "unresolved", 0, 0,
                             "unavailable", false);
        return 2;
    }
    const bool program_io_requested = program_io_any;
    const bool moe_swizzle_calibration_requested =
        !g_flag_moe_swizzle_calibration_kind.empty();
    if (g_flag_moe_swizzle_runtime_markers && !program_io_requested) {
        LOG_ERROR(CONFIG)
            << "--moe-swizzle-runtime-markers requires validated "
               "--linked-manifest and --program-io inputs";
        return 2;
    }
    if (moe_swizzle_calibration_requested &&
        (!g_flag_moe_swizzle_runtime_markers || !program_io_requested ||
         g_flag_moe_swizzle_calibration_core < 0 ||
         g_flag_moe_swizzle_calibration_core > UINT16_MAX ||
         g_flag_moe_swizzle_calibration_sample < 0 ||
         g_flag_moe_swizzle_calibration_sample > 2 ||
         g_flag_moe_swizzle_calibration_repeat < 0 ||
         g_flag_moe_swizzle_calibration_repeat > 1)) {
        LOG_ERROR(CONFIG)
            << "isolated MoE Swizzle calibration requires runtime markers, "
               "validated ProgramIo, core uint16, sample [0,2], repeat [0,1]";
        return 2;
    }
    if (!moe_swizzle_calibration_requested &&
        (g_flag_moe_swizzle_calibration_shape != "none" ||
         !g_flag_moe_swizzle_calibration_tool_sha256.empty() ||
         !g_flag_moe_swizzle_calibration_hardware_sha256.empty() ||
         !g_flag_moe_swizzle_calibration_simulation_sha256.empty() ||
         !g_flag_moe_swizzle_calibration_mapping_sha256.empty())) {
        LOG_ERROR(CONFIG)
            << "calibration metadata flags require "
               "--moe-swizzle-calibration-kind";
        return 2;
    }
    const unsigned memory_probe_count =
        (!g_flag_p5_memory_probe.empty() ? 1U : 0U) +
        (!g_flag_p6_memory_probe.empty() ? 1U : 0U) +
        (!g_flag_p8_double_buffer_probe.empty() ? 1U : 0U) +
        (program_io_requested ? 1U : 0U);
    if (memory_probe_count > 1) {
        LOG_ERROR(CONFIG)
            << "ProgramIo and P5, P6, and P8 memory probes are mutually "
               "exclusive";
        if (program_io_requested)
            PrintProgramIoStatus("preflight", "unresolved", 0, 0,
                                 "unavailable", false);
        return 2;
    }

    if (program_io_requested && g_flag_program.empty()) {
        LOG_ERROR(CONFIG) << "--program-io requires --program";
        PrintProgramIoStatus("preflight", "unresolved", 0, 0,
                             "unavailable", false);
        return 2;
    }

    if (g_flag_program_one_shot && g_flag_program.empty()) {
        LOG_ERROR(CONFIG) << "--program-one-shot requires --program";
        return 2;
    }

    if (!g_flag_p5_memory_probe.empty()) {
        if (g_flag_program.empty()) {
            LOG_ERROR(CONFIG)
                << "--p5-memory-probe requires --program";
            return 2;
        }
        const std::filesystem::path probe_path(g_flag_p5_memory_probe);
        if (!std::filesystem::exists(probe_path) ||
            !std::filesystem::is_regular_file(probe_path)) {
            LOG_ERROR(CONFIG)
                << "P5 memory probe sidecar does not exist or is not a "
                   "regular file: "
                << probe_path.string();
            return 2;
        }
    }

    if (!g_flag_p6_memory_probe.empty()) {
        if (g_flag_program.empty()) {
            LOG_ERROR(CONFIG)
                << "--p6-memory-probe requires --program";
            return 2;
        }
        const std::filesystem::path probe_path(g_flag_p6_memory_probe);
        if (!std::filesystem::exists(probe_path) ||
            !std::filesystem::is_regular_file(probe_path)) {
            LOG_ERROR(CONFIG)
                << "P6 memory probe sidecar does not exist or is not a "
                   "regular file: "
                << probe_path.string();
            return 2;
        }
    }

    if (!g_flag_p8_double_buffer_probe.empty()) {
        if (g_flag_program.empty()) {
            LOG_ERROR(CONFIG)
                << "--p8-double-buffer-probe requires --program";
            return 2;
        }
        const std::filesystem::path probe_path(
            g_flag_p8_double_buffer_probe);
        if (!std::filesystem::exists(probe_path) ||
            !std::filesystem::is_regular_file(probe_path)) {
            LOG_ERROR(CONFIG)
                << "P8 double-buffer probe sidecar does not exist or is not "
                   "a regular file: "
                << probe_path.string();
            return 2;
        }
    }

    try {
        ValidateIsaV1StartupInvariants();
    } catch (const std::exception &error) {
        LOG_ERROR(CONFIG) << "ISA v1 startup validation failed: "
                          << error.what();
        return 2;
    }

    if (g_flag_isa_v1_selftest) {
        int fails = RunIsaV1SelfTest();
        return fails == 0 ? 0 : 1;
    }

    if (g_flag_collective_data_v1_prim_selftest) {
        int fails = RunCollectiveDataV1PrimSelfTest();
        return fails == 0 ? 0 : 1;
    }

    if (g_flag_collective_phase_barrier_v1_prim_selftest) {
        int fails = RunCollectivePhaseBarrierV1PrimSelfTest();
        return fails == 0 ? 0 : 1;
    }

    if (g_flag_sync_runtime_selftest) {
        int fails = RunSyncRuntimeSelfTest();
        return fails == 0 ? 0 : 1;
    }

    if (g_flag_p2p_payload_selftest) {
        int fails = RunP2pPayloadSelfTest();
        return fails == 0 ? 0 : 1;
    }

    if (g_flag_p2p_session_selftest) {
        int fails = RunP2pSessionRuntimeSelfTest();
        return fails == 0 ? 0 : 1;
    }

    if (g_flag_moe_swizzle_runtime_capture_selftest)
        return RunMoeSwizzleRuntimeCaptureSelfTest();

    if (g_flag_p5_memory_probe_selftest) {
        int fails = RunP5MemoryProbeSelfTest();
        return fails == 0 ? 0 : 1;
    }

    // D2D V0 L0 自测：纯函数（编址/端点/矩形拓扑/端口校验），不建仿真
    if (g_flag_d2d_v0_selftest) {
        int fails = RunD2DV0SelfTest();
        return fails == 0 ? 0 : 1;
    }
    // D2D V1 link 自测：SystemC testbench 驱动真实包穿过 D2DLinkUnit
    if (g_flag_d2d_link_selftest) {
        int fails = RunD2DLinkSelfTest();
        return fails == 0 ? 0 : 1;
    }
    // DTE V0：bit/payload 纯函数 + 有界 channel/shared-bus SystemC 自测。
    if (g_flag_dte_v0_selftest) {
        int fails = RunDTEV0SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_dte_control_core_selftest) {
        int fails = RunDteControlCoreSelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_coll_v0_selftest) {
        int fails = RunCollV0SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_coll_v1_selftest) {
        int fails = RunCollV1SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_coll_v2_selftest) {
        int fails = RunCollV2SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_coll_v3_selftest) {
        int fails = RunCollV3SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_coll_v4_selftest) {
        int fails = RunCollV4SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_coll_v5_selftest) {
        int fails = RunCollV5SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_coll_v6_selftest) {
        int fails = RunCollV6SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_coll_r0_selftest) {
        int fails = RunCollR0SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_coll_r1_selftest) {
        int fails = RunCollR1SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_coll_r2_selftest) {
        int fails = RunCollR2SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_coll_r3_selftest) {
        int fails = RunCollR3SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_coll_r4_selftest) {
        int fails = RunCollR4SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_coll_r5_selftest) {
        int fails = RunCollR5SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_coll_r6_selftest) {
        int fails = RunCollR6SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_coll_r7_selftest) {
        int fails = RunCollR7SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_hbm_r0_selftest) {
        int fails = RunHbmR0SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_hbm_r1_selftest) {
        int fails = RunHbmR1SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_hbm_r2_selftest) {
        int fails = RunHbmR2SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_hbm_r3_selftest) {
        int fails = RunHbmR3SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_hbm_r4_selftest) {
        int fails = RunHbmR4SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_sram_r0_selftest) {
        int fails = RunSramR0SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_sram_r1_selftest) {
        int fails = RunSramR1SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_sram_r2_selftest) {
        int fails = RunSramR2SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_sram_r3_selftest) {
        int fails = RunSramR3SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_sram_r4_selftest) {
        int fails = RunSramR4SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_sram_r5_selftest) {
        int fails = RunSramR5SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_sram_r6_selftest) {
        int fails = RunSramR6SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_hbm_contention_experiment) {
        int fails = RunHbmContentionExperiment(
            g_flag_hbm_experiment_port,
            static_cast<int>(g_flag_hbm_experiment_cores),
            static_cast<int>(g_flag_hbm_experiment_pairs),
            static_cast<int>(g_flag_hbm_experiment_bytes));
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_coll_r8_selftest) {
        int fails = RunCollR8SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_dte_v3_selftest) {
        int fails = RunDTEV3SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_dte_v3b_selftest) {
        int fails = RunDTEV3bSelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_dte_v4_selftest) {
        int fails = RunDTEV4SelfTest();
        return fails == 0 ? 0 : 1;
    }
    if (g_flag_workload_rendezvous_selftest) {
        int fails = RunWorkloadRendezvousSelfTest();
        return fails == 0 ? 0 : 1;
    }

    const bool program_mode = !g_flag_program.empty() || sequence_mode;
    const bool p5_memory_probe_requested =
        !g_flag_p5_memory_probe.empty();
    if (p5_memory_probe_requested && !program_mode) {
        LOG_ERROR(CONFIG)
            << "--p5-memory-probe requires --program";
        return 2;
    }
    const bool p6_memory_probe_requested =
        !g_flag_p6_memory_probe.empty();
    if (p6_memory_probe_requested && !program_mode) {
        LOG_ERROR(CONFIG)
            << "--p6-memory-probe requires --program";
        return 2;
    }
    const bool p8_double_buffer_probe_requested =
        !g_flag_p8_double_buffer_probe.empty();
    if (p8_double_buffer_probe_requested && !program_mode) {
        LOG_ERROR(CONFIG)
            << "--p8-double-buffer-probe requires --program";
        return 2;
    }
    if (program_mode && !g_flag_workload_config.empty()) {
        LOG_ERROR(CONFIG)
            << "--program and --workload-config are mutually exclusive";
        return 2;
    }
    if (!program_mode && g_flag_workload_config.empty())
        g_flag_workload_config = kDefaultWorkloadConfig;

    std::vector<uint8_t> program_bytes;
    std::vector<std::vector<uint8_t>> sequence_program_bytes;
    std::vector<std::string> sequence_manifest_texts;
    std::vector<DenseSequenceProgramWitness>
        sequence_program_witnesses;
    std::vector<std::vector<DenseSequenceKvRange>> sequence_kv_ranges;
    std::vector<std::vector<DenseSequenceKvRange>>
        sequence_training_state_ranges;
    std::vector<DenseTrainingProgramWitness>
        sequence_training_witnesses;
    bool dense_training_sequence = false;
    std::vector<MoeForwardProgramWitness> sequence_moe_forward_witnesses;
    std::vector<frontend::program_io::ResolvedContract>
        sequence_program_io_resolved;
    std::optional<p5_probe::Spec> p5_memory_probe_spec;
    std::optional<p6_probe::Spec> p6_memory_probe_spec;
    std::optional<p8_double_buffer_probe::Spec>
        p8_double_buffer_probe_spec;
    std::optional<frontend::program_io::ResolvedContract>
        program_io_resolved;
    std::optional<std::map<uint16_t, uint16_t>>
        moe_swizzle_runtime_core_to_die;
    std::optional<MoeSwizzleRuntimeManifestCounts>
        moe_swizzle_runtime_manifest_counts;
    std::optional<external_memory::ExternalDmaRuntimeBinding>
        external_dma_binding;
    std::optional<external_memory::ExternalDmaProgram>
        external_dma_program;
    std::vector<ExternalAuthoritySpan> external_authority_spans;
    try {
        if (program_mode) {
            ValidatePlatformConfigInputs(
                g_flag_hardware_config, g_flag_simulation_config,
                g_flag_mapping_config);
            if (sequence_mode) {
                const auto program_paths = SplitSequencePaths(
                    g_flag_program_sequence, "--program-sequence");
                const auto manifest_paths = SplitSequencePaths(
                    g_flag_linked_manifest_sequence,
                    "--linked-manifest-sequence");
                const auto sidecar_paths = adamw_paged
                    ? std::vector<std::string>{}
                    : SplitSequencePaths(g_flag_program_io_sequence,
                                         "--program-io-sequence");
                if (manifest_paths.size() != program_paths.size() ||
                    (!adamw_paged &&
                     sidecar_paths.size() != program_paths.size()) ||
                    (adamw_paged && program_paths.size() != 2) ||
                    ((moe_forward_sequence || moe_partial_sgd_sequence) &&
                     program_paths.size() != 2) ||
                    (inference_paged && program_paths.size() != 3) ||
                    (moe_inference_paged && program_paths.size() != 3))
                    throw std::runtime_error(
                        "Program sequence path lists or strict two-step shape drifted");
                for (std::size_t index = 0;
                     index < program_paths.size(); ++index) {
                    auto bytes = ReadProgramFile(program_paths[index]);
                    const ProgramArtifact decoded =
                        DecodeProgramArtifact(bytes);
                    sequence_program_witnesses.push_back(
                        CountDenseSequenceProgramRecords(decoded));
                    const std::string manifest = ReadRegularTextFile(
                        "Dense sequence linked Program manifest",
                        manifest_paths[index], kMaxLinkedProgramManifestBytes);
                    const ProgramArtifact finalized =
                        frontend::ProgramArtifactFinalizer{}.Finalize(
                            frontend::ProgramArtifactFinalizer::Parse(manifest));
                    if (EncodeProgramArtifact(finalized) != bytes)
                        throw std::runtime_error(
                            "Dense sequence manifest/artifact closure failed");
                    sequence_program_bytes.push_back(std::move(bytes));
                    sequence_manifest_texts.push_back(manifest);
                    const auto kv_ranges = DenseKvRanges(manifest);
                    const auto training_ranges =
                        DenseTrainingStateRanges(manifest);
                    if (index == 0) {
                        if (kv_ranges.empty() == training_ranges.empty())
                            throw std::runtime_error(
                                "Program sequence must contain exactly one "
                                "supported persistent-state family");
                        dense_training_sequence = !training_ranges.empty() &&
                            !moe_forward_sequence && !moe_partial_sgd_sequence;
                    }
                    if (moe_forward_sequence || moe_partial_sgd_sequence) {
                        if (!kv_ranges.empty() || training_ranges.empty())
                            throw std::runtime_error(
                                "MoE forward-only sequence state family changed");
                        sequence_training_state_ranges.push_back(training_ranges);
                        sequence_moe_forward_witnesses.push_back(
                            moe_partial_sgd_sequence
                                ? ValidateMoeSgdPartialProgram(
                                      manifest, training_ranges,
                                      moe_layer1_sgd_partial_sequence)
                                : ValidateMoeForwardProgram(
                                      manifest, training_ranges));
                    } else if (dense_training_sequence) {
                        if (!kv_ranges.empty() || training_ranges.empty())
                            throw std::runtime_error(
                                "Dense training sequence state family changed");
                        sequence_training_state_ranges.push_back(
                            training_ranges);
                        sequence_training_witnesses.push_back(
                            DenseTrainingWitness(manifest, training_ranges));
                    } else {
                        if (kv_ranges.empty() || !training_ranges.empty())
                            throw std::runtime_error(
                                "Dense inference sequence state family changed");
                        sequence_kv_ranges.push_back(kv_ranges);
                    }
                    if (!adamw_paged) {
                        const std::string sidecar = ReadRegularTextFile(
                            "Dense sequence ProgramIo sidecar",
                            sidecar_paths[index]);
                        if (moe_forward_sequence ||
                            moe_partial_sgd_sequence) {
                            const nlohmann::json io = nlohmann::json::parse(sidecar);
                            if (!io.is_object() ||
                                io.value("producer_pass", std::string{}) !=
                                    "source_bound_full_moe_route_program_io")
                                throw std::runtime_error(
                                    "MoE forward sequence requires source-bound route ProgramIO");
                        }
                        sequence_program_io_resolved.push_back(
                            frontend::program_io::ParseAndResolve(
                                sidecar, manifest,
                                sequence_program_bytes.back()));
                    }
                }
                if (moe_partial_sgd_sequence) {
                    const auto &ranges = sequence_training_state_ranges.front();
                    const auto &initializations =
                        sequence_program_io_resolved.front().initializations;
                    for (const auto &range : ranges) {
                        std::size_t matches = 0;
                        for (const auto &entry : initializations) {
                            if (!entry.hbm_range.has_value() ||
                                entry.hbm_range->die_id != range.die_id ||
                                entry.hbm_range->address_bytes != range.address ||
                                entry.hbm_range->size_bytes != range.size_bytes)
                                continue;
                            ++matches;
                            if (!std::any_of(entry.bytes.begin(), entry.bytes.end(),
                                             [](uint8_t value) { return value != 0; }))
                                throw std::runtime_error(
                                    "MoE partial router SGD has zero trainable state seed");
                        }
                        if (matches != 1)
                            throw std::runtime_error(
                                "MoE partial router SGD needs one nonzero seed per trainable StateABI");
                    }
                }
                if (adamw_paged &&
                    (!dense_training_sequence ||
                     std::any_of(sequence_training_witnesses.begin(),
                                 sequence_training_witnesses.end(),
                                 [](const auto &item) {
                                     return item.adamw_records != 17 ||
                                            item.optimizer_states != 68;
                                 })))
                    throw std::runtime_error(
                        "paged external DMA requires actual two-step 17-parameter Dense AdamW");
                if ((inference_paged || moe_inference_paged) &&
                    dense_training_sequence)
                    throw std::runtime_error(
                        "paged inference DMA requires actual three-segment KV sequence");
                if (moe_forward_sequence || moe_partial_sgd_sequence) {
                    ValidateDenseTrainingStateContinuity(
                        sequence_training_state_ranges);
                    if (sequence_manifest_texts[0] == sequence_manifest_texts[1] ||
                        sequence_program_bytes[0] == sequence_program_bytes[1] ||
                        sequence_moe_forward_witnesses[0].route_state_refs ==
                            sequence_moe_forward_witnesses[1].route_state_refs)
                        throw std::runtime_error(
                            "MoE forward sequence replayed the first source/route/program");
                } else if (dense_training_sequence)
                    ValidateDenseTrainingStateContinuity(
                        sequence_training_state_ranges);
                else
                    ValidateDenseKvContinuity(sequence_kv_ranges);
                if (external_dma_requested) {
                    const std::filesystem::path binding_path(
                        g_flag_external_dma_binding);
                    external_dma_binding =
                        external_memory::LoadExternalDmaRuntimeBinding(
                            binding_path);
                    external_dma_program =
                        external_memory::LoadExternalDmaProgram(
                            binding_path.parent_path() /
                                external_dma_binding->program_relative_path,
                            external_dma_binding->expected_source);
                    if (external_dma_binding->phase_mode ==
                            external_memory::ExternalDmaRuntimePhaseMode::
                                kBringInThenFinalWriteback &&
                        !dense_training_sequence)
                        throw std::runtime_error(
                            "external DMA final writeback phase requires a "
                            "Dense training sequence");
                    if (external_dma_binding->phase_mode ==
                            external_memory::ExternalDmaRuntimePhaseMode::
                                kBringInThenFinalWriteback) {
                        for (const auto &io : sequence_program_io_resolved) {
                            if (std::any_of(io.initializations.begin(),
                                            io.initializations.end(),
                                            [](const auto &entry) {
                                                return std::holds_alternative<
                                                    frontend::program_io::HbmTarget>(
                                                    entry.source.target);
                                            }))
                                throw std::runtime_error(
                                    "external training ProgramIO preloaded HBM StateABI "
                                    "before authoritative E2H restore");
                        }
                        external_authority_spans = ExternalTrainingAuthority(
                            *external_dma_program,
                            sequence_training_state_ranges.front());
                    }
                }
            } else {
                program_bytes = ReadProgramFile(g_flag_program);
            }
            const ProgramArtifact decoded_program = DecodeProgramArtifact(
                sequence_mode ? sequence_program_bytes.front() : program_bytes);
            if (program_io_requested) {
                const std::string manifest = ReadRegularTextFile(
                    "linked Program manifest", g_flag_linked_manifest,
                    kMaxLinkedProgramManifestBytes);
                const std::string sidecar = ReadRegularTextFile(
                    "ProgramIo sidecar", g_flag_program_io);
                program_io_resolved =
                    frontend::program_io::ParseAndResolve(
                        sidecar, manifest, program_bytes);
                if (g_flag_moe_swizzle_runtime_markers) {
                    const frontend::LinkedProgramManifestDto parsed =
                        frontend::ProgramArtifactFinalizer::Parse(manifest);
                    const ProgramArtifact finalized =
                        frontend::ProgramArtifactFinalizer{}.Finalize(parsed);
                    if (EncodeProgramArtifact(finalized) != program_bytes)
                        throw std::runtime_error(
                            "MoE Swizzle marker manifest/artifact bytes drifted");
                    std::map<uint16_t, uint16_t> exact;
                    std::set<uint16_t> dies;
                    for (const auto &binding : parsed.core_bindings) {
                        if (binding.runtime_core_id > UINT16_MAX ||
                            binding.logical_core.die_id > 3)
                            throw std::runtime_error(
                                "MoE Swizzle marker core binding exceeds 2x2 mesh");
                        const uint16_t runtime_core =
                            static_cast<uint16_t>(binding.runtime_core_id);
                        const uint16_t die = static_cast<uint16_t>(
                            binding.logical_core.die_id);
                        if (!exact.emplace(runtime_core, die).second)
                            throw std::runtime_error(
                                "MoE Swizzle marker runtime core binding conflicts");
                        dies.insert(die);
                    }
                    if (!moe_swizzle_calibration_requested &&
                        dies != std::set<uint16_t>{0, 1, 2, 3})
                        throw std::runtime_error(
                            "MoE Swizzle marker manifest must bind all 2x2 dies");
                    moe_swizzle_runtime_core_to_die = std::move(exact);
                    moe_swizzle_runtime_manifest_counts =
                        CountMoeSwizzleRuntimeManifest(decoded_program, parsed);
                }
                PrintProgramIoStatus(
                    "resolved",
                    ProgramIoModeName(program_io_resolved->mode),
                    program_io_resolved->initializations.size(),
                    program_io_resolved->output_probes.size(),
                    program_io_resolved->program_artifact_sha256, true);
            }
            if (p5_memory_probe_requested) {
                const std::filesystem::path probe_path(
                    g_flag_p5_memory_probe);
                if (!std::filesystem::exists(probe_path))
                    throw std::runtime_error(
                        "P5 memory probe sidecar does not exist: " +
                        probe_path.string());
                if (!std::filesystem::is_regular_file(probe_path))
                    throw std::runtime_error(
                        "P5 memory probe sidecar is not a regular file: " +
                        probe_path.string());
                p5_memory_probe_spec = p5_probe::Load(probe_path);
            }
            if (p6_memory_probe_requested) {
                const std::filesystem::path probe_path(
                    g_flag_p6_memory_probe);
                p6_memory_probe_spec = p6_probe::Load(probe_path);
            }
            if (p8_double_buffer_probe_requested) {
                const std::filesystem::path probe_path(
                    g_flag_p8_double_buffer_probe);
                p8_double_buffer_probe_spec =
                    p8_double_buffer_probe::Load(probe_path);
            }
        } else {
            ValidateConfigInputs(
                g_flag_workload_config, g_flag_hardware_config,
                g_flag_simulation_config, g_flag_mapping_config);
        }
    } catch (const std::exception &error) {
        LOG_ERROR(CONFIG) << "Configuration preflight failed: "
                          << error.what();
        if (program_io_requested)
            PrintProgramIoStatus("resolve", "unresolved", 0, 0,
                                 "unavailable", false);
        return 2;
    }

    // 清理所有上一次运行后产生的log文件
    DeleteCoreLogFiles();
    DeleteMemoryLogFiles();

    // 收集所有配置文件，统一解析。Program v1 固定使用 dataflow，
    // 不读取或伪造 workload JSON。
    std::unique_ptr<config_helper_program> program_helper;
    std::unique_ptr<config_helper_program_sequence> sequence_helper;
    try {
        if (program_mode) {
            SYSTEM_MODE = SIM_DATAFLOW;
            InitPlatform(g_flag_hardware_config, g_flag_simulation_config,
                         g_flag_mapping_config);
            if (sequence_mode)
                sequence_helper =
                    std::make_unique<config_helper_program_sequence>(
                        sequence_program_bytes, false,
                        external_dma_binding.has_value() &&
                            external_dma_binding->phase_mode ==
                                external_memory::ExternalDmaRuntimePhaseMode::
                                    kBringInThenFinalWriteback);
            else
                program_helper =
                    std::make_unique<config_helper_program>(
                        program_bytes, !g_flag_program_one_shot);
            const uint64_t loaded_capabilities = sequence_mode
                ? DecodeProgramArtifact(sequence_program_bytes.front()).capabilities
                : program_helper->artifact().capabilities;
            std::cout << "Loaded Program Format " << kProgramFormatMajor
                      << "." << kProgramFormatMinor << ", ISA "
                      << kProgramIsaMajor << "." << kProgramIsaMinor
                      << ", capabilities=0x" << std::hex
                      << loaded_capabilities << std::dec
                      << "\n";
        } else {
            InitGrid(g_flag_workload_config, g_flag_hardware_config,
                     g_flag_simulation_config, g_flag_mapping_config);
        }
        InitGlobalMembers();
        InitializeMemorySpec();
    } catch (const std::exception &error) {
        LOG_ERROR(CONFIG) << "Configuration initialization failed: "
                          << error.what();
        return 2;
    }

    // init_dram_areas();
    // initialize_cache_structures();

    Event_engine *event_engine =
        new Event_engine("event-engine", g_flag_trace_window);
    std::unique_ptr<Monitor> monitor;
    if (program_mode)
        monitor = std::make_unique<Monitor>(
            "monitor", event_engine,
            sequence_mode
                ? static_cast<config_helper_base *>(sequence_helper.get())
                : static_cast<config_helper_base *>(program_helper.get()));
    else
        monitor = std::make_unique<Monitor>(
            "monitor", event_engine, g_flag_workload_config.c_str());

    std::unique_ptr<external_memory::ExternalDmaProgramExecutor>
        external_dma_executor;
    std::unique_ptr<external_memory::DenseAdamwMidProgramPager>
        adamw_mid_program_pager;
    std::unique_ptr<external_memory::DenseInferenceMidProgramPager>
        inference_mid_program_pager;
    bool inference_rect_paged = false;
    std::unique_ptr<external_memory::MoeInferenceMidProgramPager>
        moe_inference_mid_program_pager;
    std::unique_ptr<ExternalDmaStartupCoordinator>
        external_dma_coordinator;
    if (external_dma_program.has_value()) {
        try {
            if (monitor->hbmRuntime == nullptr)
                throw std::runtime_error(
                    "external DMA startup requires distributed HBM runtime");
            std::map<external_memory::HbmEndpoint, HBMBackend *> backends;
            for (const auto &binding :
                 external_dma_program->backend_bindings) {
                HBMRuntimeInstance *instance = monitor->hbmRuntime->Find(
                    static_cast<int>(binding.stack_id),
                    static_cast<int>(binding.channel_id));
                if (instance == nullptr || !instance->backend)
                    throw std::runtime_error(
                        "external DMA binding has no matching HBM backend");
                const auto endpoint = std::make_pair(
                    binding.stack_id, binding.channel_id);
                if (!backends.emplace(endpoint, instance->backend.get()).second)
                    throw std::runtime_error(
                        "external DMA binding repeats an HBM endpoint");
            }
            external_dma_executor = std::make_unique<
                external_memory::ExternalDmaProgramExecutor>(
                    "external_dma_startup_executor",
                    *external_dma_program, std::move(backends),
                    sc_time(CYCLE, SC_NS),
                    external_dma_binding->phase_mode);
            external_dma_coordinator =
                std::make_unique<ExternalDmaStartupCoordinator>(
                    "external_dma_startup_coordinator",
                    *external_dma_executor,
                    external_dma_binding->phase_mode);
            monitor->GateStartupUntil(
                external_dma_coordinator->ReadyEvent());
        } catch (const std::exception &error) {
            LOG_ERROR(CONFIG)
                << "External DMA startup initialization failed: "
                << error.what();
            return 2;
        }
    }
    if (adamw_paged) {
        try {
            if (monitor->hbmRuntime == nullptr ||
                monitor->workerCores[0] == nullptr ||
                !monitor->workerCores[0]->lsu_memory)
                throw std::runtime_error(
                    "paged Dense AdamW requires real die0 HBM and core0 LSU");
            auto *physical = monitor->hbmRuntime->Find(0, 0);
            if (physical == nullptr || !physical->backend)
                throw std::runtime_error(
                    "paged Dense AdamW requires source die0 HBM backend");
            std::map<external_memory::HbmEndpoint, HBMBackend *> backends{
                {{0, 0}, physical->backend.get()}};
            adamw_mid_program_pager = std::make_unique<
                external_memory::DenseAdamwMidProgramPager>(
                    "dense_adamw_mid_program_pager",
                    std::filesystem::path(g_flag_dense_adamw_paged_runtime),
                    sequence_manifest_texts, std::move(backends),
                    sc_time(CYCLE, SC_NS));
            monitor->workerCores[0]->lsu_memory->SetDenseAdamwPager(
                adamw_mid_program_pager.get());
            std::cout << "[DENSE_ADAMW_PAGED_BINDING] source="
                      << adamw_mid_program_pager->SourceRef()
                      << " state_abi=83 events=332 hbm_capacity=36864"
                      << " workspace_end=9248 pass=1" << std::endl;
        } catch (const std::exception &error) {
            LOG_ERROR(CONFIG) << "Dense AdamW paged DMA binding failed: "
                              << error.what();
            return 2;
        }
    }
    if (inference_paged) {
        try {
            if (monitor->hbmRuntime == nullptr)
                throw std::runtime_error(
                    "paged Dense inference requires distributed HBM runtime");
            std::map<std::pair<uint64_t, uint64_t>, HBMBackend *> backends;
            std::vector<uint64_t> runtime_cores;
            for (uint64_t die = 0; die < 4; ++die) {
                auto *physical = monitor->hbmRuntime->Find(die, 0);
                if (physical == nullptr || !physical->backend) {
                    if (die == 0)
                        throw std::runtime_error(
                            "paged Dense inference requires physical die0 HBM backend");
                    break;
                }
                const uint64_t core = die * 4;
                if (monitor->workerCores[core] == nullptr ||
                    !monitor->workerCores[core]->lsu_memory)
                    throw std::runtime_error(
                        "paged Dense inference lacks a physical Die-local core0 LSU");
                backends.emplace(std::make_pair(die, 0),
                                 physical->backend.get());
                runtime_cores.push_back(core);
            }
            if (backends.size() != 1 && backends.size() != 4)
                throw std::runtime_error(
                    "paged Dense inference requires exactly one or four HBM homes");
            inference_rect_paged = backends.size() == 4;
            inference_mid_program_pager = std::make_unique<
                external_memory::DenseInferenceMidProgramPager>(
                    "dense_inference_mid_program_pager",
                    std::filesystem::path(g_flag_dense_inference_paged_runtime),
                    sequence_manifest_texts, std::move(backends),
                    sc_time(CYCLE, SC_NS));
            for (const uint64_t core : runtime_cores)
                monitor->workerCores[core]->lsu_memory->SetDenseInferencePager(
                    inference_mid_program_pager.get());
            std::cout << "[DENSE_INFERENCE_PAGED_BINDING] source="
                      << inference_mid_program_pager->SourceRef()
                      << " parameter_states=" << (inference_rect_paged ? 60 : 15)
                      << " kv_pages=" << (inference_rect_paged ? 16 : 4)
                      << " events=" << (inference_rect_paged ? 260 : 65)
                      << " hbm_capacity_per_die=12288 weight_slot_base=1600"
                      << " highest_state_end="
                      << (inference_rect_paged ? 11328 : 10560)
                      << " pass=1" << std::endl;
        } catch (const std::exception &error) {
            LOG_ERROR(CONFIG) << "Dense inference paged DMA binding failed: "
                              << error.what();
            return 2;
        }
    }

    if (moe_inference_paged) {
        try {
            const std::filesystem::path sidecar_path(
                g_flag_moe_inference_paged_runtime);
            std::ifstream sidecar_stream(sidecar_path);
            if (!sidecar_stream)
                throw std::runtime_error("MoE paged sidecar cannot be opened");
            nlohmann::json sidecar_json;
            sidecar_stream >> sidecar_json;
            const auto schema = sidecar_json.at("schema_version").get<std::string>();
            const bool sparse_v6 = schema ==
                "wafer_frontend.moe_inference_paged_runtime/v6alpha1";
            const uint64_t active_die_count = sparse_v6 ? 100 :
                schema == "wafer_frontend.moe_inference_paged_runtime/v5alpha1" ? 100 :
                schema == "wafer_frontend.moe_inference_paged_runtime/v4alpha1" ? 9 :
                schema == "wafer_frontend.moe_inference_paged_runtime/v3alpha1" ? 6 :
                schema == "wafer_frontend.moe_inference_paged_runtime/v2alpha1" ? 4 : 2;
            const uint64_t die_count = sparse_v6
                ? sidecar_json.at("physical_die_count").get<uint64_t>()
                : active_die_count;
            if (sparse_v6 && !((DIE_X == 11 && DIE_Y == 11 && die_count == 121) ||
                               (DIE_X == 12 && DIE_Y == 12 && die_count == 144)))
                throw std::runtime_error("v6 sparse physical Die geometry changed");
            if (DIE_X <= 0 || DIE_Y <= 0 || DIE_COUNT != static_cast<int>(die_count) ||
                DIE_X * DIE_Y != static_cast<int>(die_count))
                throw std::runtime_error("paged MoE inference physical Die mesh differs from EP");
            if (monitor->hbmRuntime == nullptr)
                throw std::runtime_error("paged MoE inference requires physical HBM runtime");
            std::map<std::pair<uint64_t, uint64_t>, HBMBackend *> backends;
            for (uint64_t die = 0; die < die_count; ++die) {
                const uint64_t core_id = die * 4;
                if (core_id >= TOTAL_CORES ||
                    monitor->workerCores[core_id] == nullptr ||
                    (die < active_die_count &&
                     !monitor->workerCores[core_id]->lsu_memory))
                    throw std::runtime_error(
                        "paged MoE inference requires each physical EP core LSU");
                auto *home = monitor->hbmRuntime->Find(die, 0);
                if (!home || !home->backend)
                    throw std::runtime_error(
                        "paged MoE inference requires each real HBM backend");
                backends.emplace(std::make_pair(die, 0), home->backend.get());
            }
            moe_inference_mid_program_pager = std::make_unique<
                external_memory::MoeInferenceMidProgramPager>(
                    "moe_inference_mid_program_pager", sidecar_path,
                    sequence_manifest_texts, std::move(backends),
                    sc_time(CYCLE, SC_NS));
            if (moe_inference_mid_program_pager->ActiveDieCount() != active_die_count)
                throw std::runtime_error("MoE pager version/active Die count differs");
            for (uint64_t die = 0; die < active_die_count; ++die)
                monitor->workerCores[die * 4]->lsu_memory->SetMoeInferencePager(
                    moe_inference_mid_program_pager.get());
            std::cout << "[MOE_INFERENCE_PAGED_BINDING] source="
                      << moe_inference_mid_program_pager->SourceRef()
                      << " mesh=" << DIE_Y << "x" << DIE_X << " ep=" << active_die_count
                      << " physical_weights="
                      << moe_inference_mid_program_pager->WeightPageCount()
                      << " kv_pages=4 lsu_gates="
                      << moe_inference_mid_program_pager->ExpectedEvents()
                      << " hbm_capacity_per_die=" << (active_die_count == 100 ? 2048 : 1024)
                      << " workspace_end=464 highest_relative_state_end="
                      << (active_die_count == 100 ? 1600 : 960)
                      << " pass=1" << std::endl;
        } catch (const std::exception &error) {
            LOG_ERROR(CONFIG) << "MoE inference paged DMA binding failed: "
                              << error.what();
            return 2;
        }
    }

    std::optional<p5_probe::Applied> p5_memory_probe_applied;
    if (p5_memory_probe_spec.has_value()) {
        try {
            p5_probe::Bindings bindings;
            for (int core = 0; core < TOTAL_CORES; ++core) {
                WorkerCore *worker = monitor->workerCores[core];
                if (worker == nullptr || !worker->sram_access)
                    throw std::runtime_error(
                        "P5 memory probe found a missing core SRAM AccessUnit");
                bindings.sram_by_core.emplace(
                    static_cast<uint32_t>(core),
                    worker->sram_access.get());
            }
            bindings.hbm_runtime = monitor->hbmRuntime;
            bindings.current_die_for_core = [](uint32_t core) {
                if (core >= static_cast<uint32_t>(TOTAL_CORES))
                    throw p5_probe::Error(
                        "P5 memory probe core is outside the platform");
                return DieOfGlobal(static_cast<int>(core));
            };
            p5_memory_probe_applied = p5_probe::ApplyBeforeSimulation(
                *p5_memory_probe_spec, bindings);
        } catch (const std::exception &error) {
            LOG_ERROR(CONFIG)
                << "P5 memory probe pre-simulation apply failed: "
                << error.what();
            return 2;
        }
    }

    std::optional<p6_probe::Applied> p6_memory_probe_applied;
    if (p6_memory_probe_spec.has_value()) {
        try {
            p5_probe::Bindings bindings;
            for (int core = 0; core < TOTAL_CORES; ++core) {
                WorkerCore *worker = monitor->workerCores[core];
                if (worker == nullptr || !worker->sram_access)
                    throw std::runtime_error(
                        "P6 memory probe found a missing core SRAM AccessUnit");
                bindings.sram_by_core.emplace(
                    static_cast<uint32_t>(core),
                    worker->sram_access.get());
            }
            p6_memory_probe_applied =
                p6_probe::ApplyBeforeSimulation(
                    *p6_memory_probe_spec, bindings);
        } catch (const std::exception &error) {
            LOG_ERROR(CONFIG)
                << "P6 memory probe pre-simulation apply failed: "
                << error.what();
            return 2;
        }
    }

    std::optional<p8_double_buffer_probe::Applied>
        p8_double_buffer_probe_applied;
    if (p8_double_buffer_probe_spec.has_value()) {
        try {
            p8_double_buffer_probe::Bindings bindings;
            for (int core = 0; core < TOTAL_CORES; ++core) {
                WorkerCore *worker = monitor->workerCores[core];
                if (worker == nullptr || !worker->sram_access)
                    throw std::runtime_error(
                        "P8 probe found a missing core SRAM AccessUnit");
                bindings.sram_by_core.emplace(
                    static_cast<uint32_t>(core),
                    worker->sram_access.get());
            }
            bindings.hbm_runtime = monitor->hbmRuntime;
            bindings.current_die_for_core = [](uint32_t core) {
                if (core >= static_cast<uint32_t>(TOTAL_CORES))
                    throw p8_double_buffer_probe::Error(
                        "P8 probe core is outside the platform");
                return DieOfGlobal(static_cast<int>(core));
            };
            p8_double_buffer_probe_applied =
                p8_double_buffer_probe::ApplyBeforeSimulation(
                    *p8_double_buffer_probe_spec, bindings);
        } catch (const std::exception &error) {
            LOG_ERROR(CONFIG)
                << "P8 double-buffer probe pre-simulation apply failed: "
                << error.what();
            return 2;
        }
    }

    std::optional<frontend::program_io::Applied> program_io_applied;
    std::optional<frontend::program_io::Bindings> sequence_program_io_bindings;
    std::optional<frontend::program_io::Applied>
        sequence_program_io_applied;
    if (program_io_resolved.has_value()) {
        try {
            frontend::program_io::Bindings bindings;
            for (int core = 0; core < TOTAL_CORES; ++core) {
                WorkerCore *worker = monitor->workerCores[core];
                if (worker == nullptr || !worker->sram_access)
                    throw std::runtime_error(
                        "ProgramIo found a missing core SRAM AccessUnit");
                bindings.sram_by_runtime_core.emplace(
                    static_cast<uint32_t>(core),
                    worker->sram_access.get());
            }
            bindings.hbm_runtime = monitor->hbmRuntime;
            program_io_applied =
                frontend::program_io::ApplyBeforeSimulation(
                    *program_io_resolved, bindings);
            PrintProgramIoStatus(
                "applied", ProgramIoModeName(program_io_resolved->mode),
                program_io_resolved->initializations.size(),
                program_io_resolved->output_probes.size(),
                program_io_resolved->program_artifact_sha256, true);
        } catch (const std::exception &error) {
            PrintProgramIoStatus(
                "apply", ProgramIoModeName(program_io_resolved->mode),
                program_io_resolved->initializations.size(),
                program_io_resolved->output_probes.size(), "unavailable",
                false);
            LOG_ERROR(CONFIG)
                << "ProgramIo pre-simulation apply failed: " << error.what();
            return 2;
        }
    }
    if (sequence_mode && !adamw_paged) {
        try {
            frontend::program_io::Bindings bindings;
            for (int core = 0; core < TOTAL_CORES; ++core) {
                WorkerCore *worker = monitor->workerCores[core];
                if (worker == nullptr || !worker->sram_access)
                    throw std::runtime_error(
                        "Dense sequence ProgramIo found a missing SRAM AccessUnit");
                bindings.sram_by_runtime_core.emplace(
                    static_cast<uint32_t>(core),
                    worker->sram_access.get());
            }
            bindings.hbm_runtime = monitor->hbmRuntime;
            sequence_program_io_bindings = bindings;
            sequence_program_io_applied =
                frontend::program_io::ApplyBeforeSequenceSegment(
                    sequence_program_io_resolved.front(), bindings, false);
        } catch (const std::exception &error) {
            LOG_ERROR(CONFIG)
                << "Dense sequence ProgramIo initial apply failed: "
                << error.what();
            return 2;
        }
    }

    std::optional<std::string> dense_training_state_digest;
    if (sequence_mode && moe_partial_sgd_sequence) {
        dense_training_state_digest = moe_layer1_sgd_partial_sequence
            ? PrintMoeLayer1SgdPartialStateBoundary(
                  *monitor->hbmRuntime, 0,
                  sequence_training_state_ranges.front(), std::nullopt)
            : PrintMoeRouterSgdPartialStateBoundary(
                  *monitor->hbmRuntime, 0,
                  sequence_training_state_ranges.front(), std::nullopt);
    }
    if (sequence_mode && dense_training_sequence) {
        if (adamw_paged) {
            dense_training_state_digest =
                adamw_mid_program_pager->ProbeInitialAuthority();
            std::cout << "[DENSE_TRAINING_SEQUENCE_STATE] version=0 bytes=32100"
                      << " digest=" << *dense_training_state_digest
                      << " content_changed=0 functional=0 pass=1"
                      << " authority=external" << std::endl;
        } else if (external_authority_spans.empty()) {
            dense_training_state_digest = PrintDenseTrainingStateBoundary(
                *monitor->hbmRuntime, 0,
                sequence_training_state_ranges.front(), std::nullopt);
        }
    }
    if (!external_authority_spans.empty()) {
        const uint64_t bytes = InspectExternalTrainingHbm(
            *monitor->hbmRuntime, external_authority_spans, false);
        std::cout << "[EXTERNAL_AUTHORITY_PRELOAD] state_abis="
                  << external_authority_spans.size()
                  << " hbm_initializations=0 present_bytes=0"
                  << " source_bytes=" << bytes << " pass=1" << std::endl;
    }
    if (sequence_mode && inference_paged) {
        const std::string digest =
            inference_mid_program_pager->ProbeInitialKvAuthority();
        std::cout << "[DENSE_INFERENCE_PAGED_KV] version=0 bytes=0"
                  << " digest=" << digest
                  << " authority=external functional=0 pass=1" << std::endl;
    }
    if (sequence_mode && moe_inference_paged) {
        const std::string digest =
            moe_inference_mid_program_pager->ProbeInitialKvAuthority();
        std::cout << "[MOE_INFERENCE_PAGED_KV] version=0 bytes=0"
                  << " digest=" << digest
                  << " authority=external functional=0 pass=1" << std::endl;
    }

    sc_trace_file *tf = sc_create_vcd_trace_file("Cchip_1");
    sc_clock clk("clk", CYCLE, SC_NS);

    if (sequence_mode) {
        if (!external_authority_spans.empty()) {
            // Only the external bridge runs; workers still wait for the
            // host-controlled startup event until physical restore is proven.
            sc_start();
            if (!external_dma_coordinator->Error().empty())
                throw std::runtime_error("external DMA bring-in failed: " +
                                         external_dma_coordinator->Error());
            if (!external_dma_coordinator->BringInReady() ||
                sequence_helper->completed_segments() != 0)
                throw std::runtime_error(
                    "external restore did not pause before any training compute");
            const uint64_t bytes = InspectExternalTrainingHbm(
                *monitor->hbmRuntime, external_authority_spans, true);
            std::cout << "[EXTERNAL_AUTHORITY_RESTORED] state_abis="
                      << external_authority_spans.size()
                      << " payload_bytes=" << bytes
                      << " matched=1 pending=0 pass=1" << std::endl;
            dense_training_state_digest = PrintDenseTrainingStateBoundary(
                *monitor->hbmRuntime, 0,
                sequence_training_state_ranges.front(), std::nullopt);
            external_dma_coordinator->ReleaseComputeAfterRestore();
        }
        for (std::size_t expected = 0;
             expected < sequence_program_bytes.size(); ++expected) {
            sc_start();
            if (external_dma_coordinator &&
                !external_dma_coordinator->Error().empty())
                throw std::runtime_error(
                    "external DMA startup failed: " +
                    external_dma_coordinator->Error());
            if (sequence_helper->completed_segments() != expected + 1)
                throw std::runtime_error(
                    "Dense sequence paused outside an exact segment boundary");
            if (adamw_paged) {
                adamw_mid_program_pager->CompleteStep(expected);
                std::cout << "[DENSE_ADAMW_EXTERNAL_PROGRAM_IO] index="
                          << expected << " probes=83"
                          << " physical_state_bytes=32100"
                          << " pending=" << adamw_mid_program_pager->Pending()
                          << " pass=1 functional=0" << std::endl;
            } else {
                if (inference_paged)
                    inference_mid_program_pager->CompleteSegment(expected);
                if (moe_inference_paged)
                    moe_inference_mid_program_pager->CompleteSegment(expected);
                const frontend::program_io::Result io_result =
                    frontend::program_io::VerifyAfterSimulation(
                        *sequence_program_io_applied);
                if (!io_result.Passed())
                    throw std::runtime_error(
                        "Dense sequence ProgramIo segment verification failed");
                std::cout << "[DENSE_SEQUENCE_PROGRAM_IO] index=" << expected
                          << " probes=" << io_result.probes.size()
                          << " pass=1" << std::endl;
                if (inference_paged)
                    std::cout << "[DENSE_INFERENCE_PAGED_EXTERNAL_PROGRAM_IO]"
                              << " index=" << expected
                              << " kv_probes=" << (inference_rect_paged ? 16 : 4)
                              << " kv_bytes="
                              << inference_mid_program_pager->KvAuthorityBytes()
                              << " pending="
                              << inference_mid_program_pager->Pending()
                              << " functional=0 pass=1" << std::endl;
                if (moe_inference_paged)
                    std::cout << "[MOE_INFERENCE_PAGED_EXTERNAL_PROGRAM_IO]"
                              << " index=" << expected
                              << " logits_probes=" << io_result.probes.size()
                              << " kv_probes=4 kv_bytes="
                              << moe_inference_mid_program_pager->KvAuthorityBytes()
                              << " parameter_immutable=1 pending="
                              << moe_inference_mid_program_pager->Pending()
                              << " functional=0 pass=1" << std::endl;
            }
            std::cout
                << "[DENSE_SEQUENCE_COMPUTE] index=" << expected
                << " records="
                << sequence_program_witnesses[expected].compute_records
                << " lsu_loads="
                << sequence_program_witnesses[expected].lsu_load_records
                << " status=done" << std::endl;
            if (moe_forward_sequence) {
                const auto &witness = sequence_moe_forward_witnesses[expected];
                std::cout << "[MOE_FORWARD_SEQUENCE_STEP] index=" << expected
                          << " records=" << witness.records
                          << " trainable_states=" << witness.trainable_states
                          << " route_states=" << witness.route_state_refs.size()
                          << " backward=0 sgd=0 functional=0 pass=1"
                          << std::endl;
            } else if (moe_partial_sgd_sequence) {
                const auto before = *dense_training_state_digest;
                dense_training_state_digest = moe_layer1_sgd_partial_sequence
                    ? PrintMoeLayer1SgdPartialStateBoundary(
                          *monitor->hbmRuntime, expected + 1,
                          sequence_training_state_ranges[expected],
                          dense_training_state_digest)
                    : PrintMoeRouterSgdPartialStateBoundary(
                          *monitor->hbmRuntime, expected + 1,
                          sequence_training_state_ranges[expected],
                          dense_training_state_digest);
                const auto &witness = sequence_moe_forward_witnesses[expected];
                std::cout << (moe_layer1_sgd_partial_sequence
                                  ? "[MOE_LAYER1_SGD_PARTIAL_SEQUENCE_STEP] index="
                                  : "[MOE_ROUTER_SGD_PARTIAL_SEQUENCE_STEP] index=")
                          << expected << " input_version=" << expected
                          << " output_version=" << expected + 1
                          << " trainable_states=" << witness.trainable_states
                          << " route_states=" << witness.route_state_refs.size()
                          << " records=" << witness.records
                          << " sgd=" << (moe_layer1_sgd_partial_sequence ? 4 : 1)
                          << " store=" << (moe_layer1_sgd_partial_sequence ? 4 : 1)
                          << " state_digest_before=" << before
                          << " state_digest_after=" << *dense_training_state_digest
                          << " full_training=0 functional=0 pass=1"
                          << std::endl;
            } else if (dense_training_sequence) {
                if (adamw_paged) {
                    const std::string current =
                        adamw_mid_program_pager->AuthorityDigest();
                    const bool changed = current != *dense_training_state_digest;
                    dense_training_state_digest = current;
                    std::cout << "[DENSE_TRAINING_SEQUENCE_STATE] version="
                              << expected + 1 << " bytes=32100 digest="
                              << current << " content_changed="
                              << (changed ? 1 : 0)
                              << " functional=0 pass=1 authority=external"
                              << std::endl;
                } else {
                    dense_training_state_digest =
                        PrintDenseTrainingStateBoundary(
                            *monitor->hbmRuntime, expected + 1,
                            sequence_training_state_ranges[expected],
                            dense_training_state_digest);
                }
                const auto &witness =
                    sequence_training_witnesses[expected];
                std::cout
                    << "[DENSE_TRAINING_SEQUENCE_STEP] index=" << expected
                    << " input_version=" << expected
                    << " output_version=" << expected + 1
                    << " trainable_states="
                    << witness.trainable_states
                    << " matmul_records=" << witness.matmul_records
                    << " sgd_records=" << witness.sgd_records
                    << " store_records=" << witness.store_records
                    << " functional=0 pass=1" << std::endl;
                if (witness.adamw_records != 0)
                    std::cout
                        << "[DENSE_ADAMW_SEQUENCE_STEP] index=" << expected
                        << " input_version=" << expected
                        << " output_version=" << expected + 1
                        << " trainable_states=" << witness.trainable_states
                        << " optimizer_states=" << witness.optimizer_states
                        << " adamw_records=" << witness.adamw_records
                        << " load_records=" << witness.load_records
                        << " store_records=" << witness.store_records
                        << " functional=0 pass=1" << std::endl;
            } else {
                if (inference_paged) {
                    const uint64_t kv_bytes =
                        inference_mid_program_pager->KvAuthorityBytes();
                    const auto &digest =
                        inference_mid_program_pager->KvAuthorityDigest();
                    std::cout << "[DENSE_SEQUENCE_KV] index=" << expected
                              << " bytes=" << kv_bytes
                              << " digest=" << digest
                              << " pass=1 authority=external" << std::endl;
                    std::cout << "[DENSE_INFERENCE_PAGED_KV] version="
                              << expected + 1 << " bytes=" << kv_bytes
                              << " digest=" << digest
                              << " authority=external functional=0 pass=1"
                              << std::endl;
                } else if (moe_inference_paged) {
                    const auto bytes =
                        moe_inference_mid_program_pager->KvAuthorityBytes();
                    const auto &digest =
                        moe_inference_mid_program_pager->KvAuthorityDigest();
                    std::cout << "[DENSE_SEQUENCE_KV] index=" << expected
                              << " bytes=" << bytes << " digest=" << digest
                              << " pass=1 authority=external" << std::endl;
                    std::cout << "[MOE_INFERENCE_PAGED_KV] version="
                              << expected + 1 << " bytes=" << bytes
                              << " digest=" << digest
                              << " authority=external functional=0 pass=1"
                              << std::endl;
                } else {
                    PrintDenseKvBoundary(
                        *monitor->hbmRuntime, expected,
                        sequence_kv_ranges[expected]);
                }
            }
            if (!adamw_paged &&
                expected + 1 < sequence_program_bytes.size()) {
                sequence_program_io_applied =
                    frontend::program_io::ApplyBeforeSequenceSegment(
                        sequence_program_io_resolved[expected + 1],
                        *sequence_program_io_bindings, true);
                if (moe_partial_sgd_sequence) {
                    const DenseSequenceBoundary next_input =
                        ReadDenseStateBoundary(
                            *monitor->hbmRuntime,
                            sequence_training_state_ranges[expected + 1]);
                    if (next_input.digest != *dense_training_state_digest)
                        throw std::runtime_error(
                            "MoE partial step input does not preserve completed prior HBM state");
                    std::cout << (moe_layer1_sgd_partial_sequence
                                      ? "[MOE_LAYER1_SGD_PARTIAL_INPUT] index="
                                      : "[MOE_ROUTER_SGD_PARTIAL_INPUT] index=")
                              << expected + 1 << " prior_store_completed=1"
                              << " same_hbm_state=1 digest="
                              << next_input.digest << " pass=1" << std::endl;
                }
            }
        }
        if (!sequence_helper->final_complete())
            throw std::runtime_error(
                "Dense sequence did not reach its final one-shot drain");
        if (external_dma_binding.has_value() &&
            external_dma_binding->phase_mode ==
                external_memory::ExternalDmaRuntimePhaseMode::
                    kBringInThenFinalWriteback) {
            external_dma_coordinator->ReleaseFinalWriteback();
            sc_start();
            if (!external_dma_coordinator->Error().empty())
                throw std::runtime_error(
                    "external DMA final writeback failed: " +
                    external_dma_coordinator->Error());
        }
        if (external_dma_coordinator) {
            const auto &execution = external_dma_coordinator->Execution();
            if (!execution.has_value() || !execution->completed ||
                execution->pending_requests != 0 ||
                std::any_of(
                    execution->probes.begin(), execution->probes.end(),
                    [](const external_memory::DmaProbeResult &probe) {
                        return !probe.matched;
                    }))
                throw std::runtime_error(
                    "external DMA final drain verification failed");
            std::cout
                << "[EXTERNAL_DMA_DRAIN] probes="
                << execution->probes.size()
                << " external_read_bytes="
                << execution->stats.external_read_bytes
                << " external_write_bytes="
                << execution->stats.external_write_bytes
                << " hbm_read_bytes=" << execution->stats.hbm_read_bytes
                << " hbm_write_bytes=" << execution->stats.hbm_write_bytes
                << " pending=0" << std::endl;
        }
        if (adamw_paged) {
            const auto &stats = adamw_mid_program_pager->Stats();
            if (adamw_mid_program_pager->CompletedEvents() != 332 ||
                adamw_mid_program_pager->ExternalAuthorityProbes() != 166 ||
                adamw_mid_program_pager->Pending() != 0 ||
                stats.submitted_requests != 332 ||
                stats.completed_requests != 332 ||
                stats.failed_requests != 0 ||
                stats.external_read_bytes != 64200 ||
                stats.external_write_bytes != 64200 ||
                stats.hbm_read_bytes != 64200 ||
                stats.hbm_write_bytes != 64200)
                throw std::runtime_error(
                    "paged Dense AdamW actual DMA traffic/drain disagreed with 83-state byte oracle");
            std::cout << "[DENSE_ADAMW_PAGED_DMA_DRAIN] events=332"
                      << " probes=166 submitted=" << stats.submitted_requests
                      << " completed=" << stats.completed_requests
                      << " external_read_bytes=" << stats.external_read_bytes
                      << " external_write_bytes=" << stats.external_write_bytes
                      << " hbm_read_bytes=" << stats.hbm_read_bytes
                      << " hbm_write_bytes=" << stats.hbm_write_bytes
                      << " pending=0 dirty=0 pinned=0 pass=1" << std::endl;
        }
        if (inference_paged) {
            const auto &stats = inference_mid_program_pager->Stats();
            const uint64_t expected_events = inference_rect_paged ? 260 : 65;
            const uint64_t expected_probes = inference_rect_paged ? 48 : 12;
            const uint64_t expected_reads = inference_rect_paged ? 334592 : 162112;
            const uint64_t expected_writes = inference_rect_paged ? 15360 : 1920;
            if (inference_mid_program_pager->CompletedEvents() != expected_events ||
                inference_mid_program_pager->ExternalKvProbes() != expected_probes ||
                inference_mid_program_pager->Pending() != 0 ||
                inference_mid_program_pager->Dirty() != 0 ||
                inference_mid_program_pager->Pinned() != 0 ||
                stats.submitted_requests != expected_events ||
                stats.completed_requests != expected_events ||
                stats.failed_requests != 0 ||
                stats.external_read_bytes != expected_reads ||
                stats.hbm_write_bytes != expected_reads ||
                stats.external_write_bytes != expected_writes ||
                stats.hbm_read_bytes != expected_writes)
                throw std::runtime_error(
                    "paged Dense inference real shared DMA/StateABI drain disagreed with byte oracle");
            std::cout << "[DENSE_INFERENCE_PAGED_DMA_DRAIN] events="
                      << expected_events << " kv_probes=" << expected_probes
                      << " submitted=" << stats.submitted_requests
                      << " completed=" << stats.completed_requests
                      << " external_read_bytes=" << stats.external_read_bytes
                      << " external_write_bytes=" << stats.external_write_bytes
                      << " hbm_read_bytes=" << stats.hbm_read_bytes
                      << " hbm_write_bytes=" << stats.hbm_write_bytes
                      << " pending=0 dirty=0 pinned=0 pass=1" << std::endl;
        }
        if (moe_inference_paged) {
            const auto &stats = moe_inference_mid_program_pager->Stats();
            const auto events = moe_inference_mid_program_pager->ExpectedEvents();
            const auto reads = moe_inference_mid_program_pager->ExpectedReadBytes();
            const auto writes = moe_inference_mid_program_pager->ExpectedWriteBytes();
            if (moe_inference_mid_program_pager->CompletedEvents() != events ||
                moe_inference_mid_program_pager->ExternalKvProbes() != 12 ||
                moe_inference_mid_program_pager->Pending() != 0 ||
                moe_inference_mid_program_pager->Dirty() != 0 ||
                moe_inference_mid_program_pager->Pinned() != 0 ||
                stats.submitted_requests != events ||
                stats.completed_requests != events ||
                stats.failed_requests != 0 ||
                stats.external_read_bytes != reads ||
                stats.hbm_write_bytes != reads ||
                stats.external_write_bytes != writes ||
                stats.hbm_read_bytes != writes)
                throw std::runtime_error(
                    "full MoE inference shared DMA/StateABI drain disagreed with signed byte oracle");
            if (moe_inference_mid_program_pager->ActiveDieCount() >= 4) {
                if (moe_inference_mid_program_pager->AdmissionWaitedEvents() == 0 ||
                    moe_inference_mid_program_pager->AdmissionWaitCycles() == 0)
                    throw std::runtime_error(
                        "multi-Die EP cores did not exercise bounded two-request DMA admission");
                std::cout << "[MOE_INFERENCE_PAGED_ADMISSION_DRAIN] waited_events="
                          << moe_inference_mid_program_pager->AdmissionWaitedEvents()
                          << " wait_cycles="
                          << moe_inference_mid_program_pager->AdmissionWaitCycles()
                          << " capacity=2 pass=1" << std::endl;
            }
            if (DIE_COUNT > static_cast<int>(moe_inference_mid_program_pager->ActiveDieCount())) {
                for (uint64_t die = moe_inference_mid_program_pager->ActiveDieCount();
                     die < static_cast<uint64_t>(DIE_COUNT); ++die) {
                    auto *home = monitor->hbmRuntime->Find(die, 0);
                    if (!home || !home->backend)
                        throw std::runtime_error("sparse idle physical HBM backend missing");
                    const auto &idle = home->backend->Stats();
                    if (idle.requests || idle.reads || idle.writes || idle.bytes ||
                        idle.completed || idle.failed)
                        throw std::runtime_error("sparse idle physical HBM saw DMA traffic");
                    std::cout << "[MOE_INFERENCE_PAGED_IDLE_DIE] die=" << die
                              << " hbm_requests=0 reads=0 writes=0 bytes=0"
                              << " completed=0 failed=0 pass=1" << std::endl;
                }
            }
            std::cout << "[MOE_INFERENCE_PAGED_DMA_DRAIN] events=" << events
                      << " kv_probes=12 submitted=" << stats.submitted_requests
                      << " completed=" << stats.completed_requests
                      << " external_read_bytes=" << stats.external_read_bytes
                      << " external_write_bytes=" << stats.external_write_bytes
                      << " hbm_read_bytes=" << stats.hbm_read_bytes
                      << " hbm_write_bytes=" << stats.hbm_write_bytes
                      << " pending=0 dirty=0 pinned=0 pass=1" << std::endl;
        }
    } else {
        sc_start();
    }

    const uint64_t makespan_cycles =
        sc_time_stamp().value() / sc_time(CYCLE, SC_NS).value();
    std::cout << "[SIM_RESULT] makespan_cycles=" << makespan_cycles << "\n";
    if (g_flag_moe_swizzle_runtime_markers &&
        !moe_swizzle_calibration_requested) {
        try {
            if (!moe_swizzle_runtime_core_to_die.has_value())
                throw std::logic_error(
                    "MoE Swizzle runtime marker lost validated core bindings");
            std::vector<P2pEndpointLifetimeEvent> lifetime_events;
            std::vector<MoeSwizzleRuntimeInterval> runtime_intervals;
            std::map<uint16_t, uint64_t> runtime_core_session_capacity;
            for (int core = 0; core < TOTAL_CORES; ++core) {
                WorkerCore *worker = monitor->workerCores[core];
                if (worker == nullptr || worker->executor == nullptr)
                    continue;
                if (!worker->executor->P2pLifetimeEventsComplete())
                    throw std::runtime_error(
                        "MoE Swizzle P2P lifetime event capture is incomplete");
                const auto &one = worker->executor->P2pLifetimeEvents();
                lifetime_events.insert(
                    lifetime_events.end(), one.begin(), one.end());
                if (!worker->executor->MoeSwizzleRuntimeIntervalsComplete())
                    throw std::runtime_error(
                        "MoE Swizzle runtime interval capture is incomplete");
                const auto &core_intervals =
                    worker->executor->MoeSwizzleRuntimeIntervals();
                runtime_intervals.insert(runtime_intervals.end(),
                                         core_intervals.begin(),
                                         core_intervals.end());
            }
            for (const auto &[runtime_core, die] :
                 *moe_swizzle_runtime_core_to_die) {
                (void)die;
                if (runtime_core >= TOTAL_CORES)
                    throw std::runtime_error(
                        "MoE Swizzle manifest core is outside runtime");
                WorkerCore *worker = monitor->workerCores[runtime_core];
                if (worker == nullptr || worker->executor == nullptr ||
                    worker->executor->P2pMaxSessions() != 3)
                    throw std::runtime_error(
                        "MoE Swizzle endpoint capacity per core must be exact 3");
                runtime_core_session_capacity.emplace(
                    runtime_core, worker->executor->P2pMaxSessions());
            }
            const auto markers = ReplayMoeSwizzleSessionMarkers(
                lifetime_events, *moe_swizzle_runtime_core_to_die,
                runtime_core_session_capacity, 4);
            for (const MoeSwizzleSessionMarker &marker : markers)
                std::cout << FormatMoeSwizzleSessionMarker(marker) << "\n";
            const auto port_markers = BuildMoeSwizzlePortTimeMarkers(
                CollectMoeSwizzlePortTimeTraces(), makespan_cycles);
            for (const MoeSwizzlePortTimeMarker &marker : port_markers)
                std::cout << FormatMoeSwizzlePortTimeMarker(marker) << "\n";
            if (!moe_swizzle_runtime_manifest_counts.has_value())
                throw std::logic_error(
                    "MoE Swizzle runtime marker lost validated manifest counts");
            const auto interval_markers =
                BuildMoeSwizzleRuntimeIntervalMarkers(
                    runtime_intervals, lifetime_events,
                    *moe_swizzle_runtime_core_to_die, 4,
                    sc_time_stamp().value(),
                    sc_time(CYCLE, SC_NS).value(),
                    *moe_swizzle_runtime_manifest_counts);
            for (const MoeSwizzleOverlapMarker &marker :
                 interval_markers.overlap)
                std::cout << FormatMoeSwizzleOverlapMarker(marker) << "\n";
            std::cout << FormatMoeSwizzleSetupMarker(interval_markers.setup)
                      << "\n";
        } catch (const std::exception &error) {
            LOG_ERROR(SYSTEM)
                << "MoE Swizzle runtime marker export failed: "
                << error.what();
            return 2;
        }
    }
    if (moe_swizzle_calibration_requested) {
        try {
            std::vector<MoeSwizzleRuntimeInterval> intervals;
            for (int core = 0; core < TOTAL_CORES; ++core) {
                WorkerCore *worker = monitor->workerCores[core];
                if (worker == nullptr || worker->executor == nullptr)
                    continue;
                if (!worker->executor->MoeSwizzleCalibrationIntervalsComplete())
                    throw std::runtime_error(
                        "isolated calibration interval capture is incomplete");
                const auto &one =
                    worker->executor->MoeSwizzleRuntimeIntervals();
                intervals.insert(intervals.end(), one.begin(), one.end());
            }
            std::vector<uint64_t> hop_cycles;
            for (const auto &marker : BuildMoeSwizzlePortTimeMarkers(
                     CollectMoeSwizzlePortTimeTraces(), makespan_cycles))
                hop_cycles.push_back(marker.busy_cycles);
            MoeSwizzleCalibrationRequest request{
                ParseMoeSwizzleCalibrationKind(
                    g_flag_moe_swizzle_calibration_kind),
                static_cast<uint16_t>(g_flag_moe_swizzle_calibration_core),
                static_cast<uint8_t>(g_flag_moe_swizzle_calibration_sample),
                static_cast<uint8_t>(g_flag_moe_swizzle_calibration_repeat),
                ParseMoeSwizzleCalibrationShape(
                    g_flag_moe_swizzle_calibration_shape),
                g_flag_moe_swizzle_calibration_tool_sha256,
                g_flag_moe_swizzle_calibration_hardware_sha256,
                g_flag_moe_swizzle_calibration_simulation_sha256,
                g_flag_moe_swizzle_calibration_mapping_sha256,
            };
            std::cout << FormatMoeSwizzleCalibrationMarker(
                             BuildMoeSwizzleCalibrationMarker(
                                 request, intervals, hop_cycles,
                                 *moe_swizzle_runtime_manifest_counts,
                                 sc_time(CYCLE, SC_NS).value()))
                      << "\n";
        } catch (const std::exception &error) {
            LOG_ERROR(SYSTEM)
                << "isolated MoE Swizzle calibration export failed: "
                << error.what();
            return 2;
        }
    }
    if (program_mode) {
        const ProgramArtifact *runtime_artifact = sequence_mode
            ? (sequence_helper ? &sequence_helper->artifact() : nullptr)
            : (program_helper ? &program_helper->artifact() : nullptr);
        if (runtime_artifact == nullptr)
            throw std::logic_error(
                "Program runtime lacks its decoded artifact");
        for (const auto &program_core :
             runtime_artifact->cores) {
            if (program_core.core_id >=
                static_cast<uint64_t>(TOTAL_CORES))
                throw std::logic_error(
                    "Program runtime core is outside the platform");
            const int core = static_cast<int>(program_core.core_id);
            WorkerCore *worker = monitor->workerCores[core];
            if (worker == nullptr || !worker->lsu_memory ||
                !worker->dte_memory_bridge)
                throw std::logic_error(
                    "Program runtime core lacks memory engines");
            const sram::LsuStats &lsu = worker->lsu_memory->stats();
            std::cout
                << "[PROGRAM_MEMORY] core=" << core
                << " lsu_issued=" << lsu.issued
                << " lsu_completed=" << lsu.completed
                << " lsu_hbm_read_bytes=" << lsu.hbm_read_bytes
                << " lsu_hbm_write_bytes=" << lsu.hbm_write_bytes
                << " lsu_sram_read_bytes=" << lsu.sram_read_bytes
                << " lsu_sram_write_bytes=" << lsu.sram_write_bytes
                << " lsu_residual="
                << worker->lsu_memory->OutstandingCount()
                << " dte_residual="
                << worker->dte_memory_bridge->OutstandingCount() << "\n";
        }
    }

    bool p5_memory_probe_failed = false;
    if (p5_memory_probe_applied.has_value()) {
        try {
            const p5_probe::Result result =
                p5_probe::VerifyAfterSimulation(
                    *p5_memory_probe_applied);
            std::cout
                << "[P5 MEMORY PROBE] scenario=" << result.scenario
                << " source_initialized="
                << (result.source_initialized ? 1 : 0)
                << " payload_bytes=" << result.payload_bytes
                << " expected_checksum=" << result.expected_checksum
                << " destination_checksum="
                << result.destination_checksum
                << " payload_match=" << (result.payload_match ? 1 : 0)
                << " sentinels_intact="
                << (result.sentinels_intact ? 1 : 0) << "\n";
            p5_memory_probe_failed =
                !result.source_initialized || !result.payload_match ||
                !result.sentinels_intact ||
                result.destination_checksum != result.expected_checksum;
        } catch (const std::exception &error) {
            p5_memory_probe_failed = true;
            LOG_ERROR(SYSTEM)
                << "P5 memory probe post-simulation verify failed: "
                << error.what();
        }
    }

    bool p6_memory_probe_failed = false;
    if (p6_memory_probe_applied.has_value()) {
        try {
            const p6_probe::Result result =
                p6_probe::VerifyAfterSimulation(
                    *p6_memory_probe_applied);
            for (const p6_probe::SourceResult &source : result.sources) {
                std::cout
                    << "[P6 MEMORY SOURCE] scenario=" << result.scenario
                    << " core=" << source.core
                    << " region=" << source.region
                    << " checksum=" << source.checksum
                    << " source_initialized="
                    << (source.source_initialized ? 1 : 0)
                    << " payload_match="
                    << (source.payload_match ? 1 : 0)
                    << " sentinels_intact="
                    << (source.sentinels_intact ? 1 : 0) << "\n";
            }
            for (const p6_probe::VerificationResult &verification :
                 result.verifications) {
                std::cout
                    << "[P6 MEMORY PROBE] scenario=" << result.scenario
                    << " core=" << verification.core
                    << " region=" << verification.region
                    << " checksum=" << verification.checksum
                    << " payload_match="
                    << (verification.payload_match ? 1 : 0)
                    << " sentinels_intact="
                    << (verification.sentinels_intact ? 1 : 0) << "\n";
            }
            const auto image = program_helper
                ? program_helper->collective_program_image()
                : nullptr;
            uint64_t action_count = 0;
            uint64_t child_count = 0;
            uint64_t wave_count = 0;
            if (image) {
                for (const auto &core : image->Cores()) {
                    if (action_count > UINT64_MAX - core.actions.size())
                        throw std::overflow_error(
                            "P6 collective action statistic overflows u64");
                    action_count += core.actions.size();
                }
                child_count = image->Lowering().children.size();
                for (const auto &plan : image->Lowering().plans) {
                    if (wave_count > UINT64_MAX - plan.waves.size())
                        throw std::overflow_error(
                            "P6 collective wave statistic overflows u64");
                    wave_count += plan.waves.size();
                }
            }
            std::cout
                << "[P6 COLLECTIVE STATS] scenario=" << result.scenario
                << " child_count=" << child_count
                << " action_count=" << action_count
                << " wave_count=" << wave_count << "\n";

            uint64_t aggregate_residual = 0;
            uint64_t endpoint_residual = 0;
            if (!program_helper)
                throw std::logic_error(
                    "P6 memory probe requires a Program helper");
            for (const auto &core : program_helper->artifact().cores) {
                const WorkerCoreExecutor *executor =
                    monitor->workerCores[core.core_id]->executor;
                aggregate_residual +=
                    executor->CollectiveProgramResidual();
                endpoint_residual += executor->P2pEndpointResidual();
            }
            const CollectiveBarrierRuntimeResidual barrier =
                CollectiveBarrierRuntimeResidualState();
            const uint64_t barrier_residual =
                barrier.active_states + barrier.arrived_ranks +
                barrier.departed_ranks + barrier.waiting_ranks;
            const uint64_t timing_residual =
                P2pSharedTimingSidebandRuntime::Residual();
            std::cout
                << "[P6 COLLECTIVE DRAIN] scenario=" << result.scenario
                << " aggregate=" << aggregate_residual
                << " admission=" << aggregate_residual
                << " barrier=" << barrier_residual
                << " endpoint=" << endpoint_residual
                << " timing=" << timing_residual << "\n";
            if (aggregate_residual != 0 || endpoint_residual != 0 ||
                barrier_residual != 0 || timing_residual != 0)
                p6_memory_probe_failed = true;
            p6_memory_probe_failed =
                p6_memory_probe_failed || !result.Passed();
        } catch (const std::exception &error) {
            p6_memory_probe_failed = true;
            LOG_ERROR(SYSTEM)
                << "P6 memory probe post-simulation verify failed: "
                << error.what();
        }
    }

    bool p8_double_buffer_probe_failed = false;
    if (p8_double_buffer_probe_applied.has_value()) {
        try {
            const p8_double_buffer_probe::Result result =
                p8_double_buffer_probe::VerifyAfterSimulation(
                    *p8_double_buffer_probe_applied);
            for (const p8_double_buffer_probe::RangeResult &range :
                 result.ranges) {
                std::cout
                    << "[P8 DOUBLE BUFFER PROBE] scenario="
                    << result.scenario
                    << " space="
                    << (range.space == p8_double_buffer_probe::Space::kSram
                            ? "SRAM" : "HBM")
                    << " core=" << range.core
                    << " region=" << range.region
                    << " absolute_address_bytes="
                    << range.absolute_address_bytes
                    << " payload_bytes=" << range.payload_bytes
                    << " expected_checksum=" << range.expected_checksum
                    << " checksum=" << range.checksum
                    << " payload_match="
                    << (range.payload_match ? 1 : 0)
                    << " sentinels_intact="
                    << (range.sentinels_intact ? 1 : 0) << "\n";
            }

            WorkerCore *worker = monitor->workerCores[0];
            if (worker == nullptr || !worker->lsu_memory ||
                !worker->dte_memory_bridge || !worker->executor)
                throw std::runtime_error(
                    "P8 probe found missing core-0 memory engines");
            const sram::LsuStats &lsu = worker->lsu_memory->stats();
            const DteMemoryStats &dte =
                worker->dte_memory_bridge->stats();
            std::cout
                << "[P8 DOUBLE BUFFER STATS] core=0"
                << " lsu_issued=" << lsu.issued
                << " lsu_completed=" << lsu.completed
                << " lsu_hbm_read_bytes=" << lsu.hbm_read_bytes
                << " lsu_hbm_write_bytes=" << lsu.hbm_write_bytes
                << " lsu_sram_read_bytes=" << lsu.sram_read_bytes
                << " lsu_sram_write_bytes=" << lsu.sram_write_bytes
                << " lsu_residual="
                << worker->lsu_memory->OutstandingCount()
                << " dte_issued=" << dte.issued
                << " dte_completed=" << dte.completed
                << " dte_hbm_read_bytes=" << dte.hbm_read_bytes
                << " dte_hbm_write_bytes=" << dte.hbm_write_bytes
                << " dte_sram_read_bytes=" << dte.sram_read_bytes
                << " dte_sram_write_bytes=" << dte.sram_write_bytes
                << " dte_residual="
                << worker->dte_memory_bridge->OutstandingCount()
                << " dte_tracker_residual="
                << worker->executor->DteOutstandingCount() << "\n";
            p8_double_buffer_probe_failed = !result.Passed() ||
                worker->lsu_memory->OutstandingCount() != 0 ||
                worker->dte_memory_bridge->OutstandingCount() != 0 ||
                worker->executor->DteOutstandingCount() != 0;
        } catch (const std::exception &error) {
            p8_double_buffer_probe_failed = true;
            LOG_ERROR(SYSTEM)
                << "P8 double-buffer probe post-simulation verify failed: "
                << error.what();
        }
    }

    bool program_io_failed = false;
    if (program_io_applied.has_value()) {
        try {
            const frontend::program_io::Result result =
                frontend::program_io::VerifyAfterSimulation(
                    *program_io_applied);
            if (result.probes.size() !=
                program_io_applied->contract.output_probes.size())
                throw std::logic_error(
                    "ProgramIo probe/result count changed");
            std::string aggregate;
            std::size_t probe_index = 0;
            for (const frontend::program_io::ProbeResult &probe :
                 result.probes) {
                const frontend::program_io::ResolvedOutputProbe &resolved_probe =
                    program_io_applied->contract.output_probes[probe_index++];
                const bool hbm = resolved_probe.hbm_range.has_value();
                aggregate += probe.probe_id;
                aggregate += probe.actual_sha256;
                const bool passed =
                    probe.exact_match && probe.all_bytes_valid;
                if (hbm)
                    std::cout
                        << "[PROGRAM_IO_PROBE] id=" << probe.probe_id
                        << " die=" << resolved_probe.hbm_range->die_id;
                else
                    std::cout
                        << "[PROGRAM_IO_PROBE] id=" << probe.probe_id
                        << " core=" << probe.runtime_core_id;
                std::cout
                    << " address=" << probe.absolute_address_bytes
                    << " bytes=" << probe.length_bytes
                    << " expected_checksum=" << probe.expected_sha256
                    << " checksum=" << probe.actual_sha256
                    << " valid=" << (probe.all_bytes_valid ? 1 : 0)
                    << " exact=" << (probe.exact_match ? 1 : 0)
                    << " pass=" << (passed ? 1 : 0) << "\n";
            }
            const std::string aggregate_checksum =
                frontend::program_io::Sha256Hex(aggregate);
            program_io_failed = !result.Passed();
            PrintProgramIoStatus(
                "verify", ProgramIoModeName(program_io_applied->contract.mode),
                program_io_applied->contract.initializations.size(),
                result.probes.size(), aggregate_checksum,
                !program_io_failed);
        } catch (const std::exception &error) {
            program_io_failed = true;
            PrintProgramIoStatus(
                "verify", ProgramIoModeName(program_io_applied->contract.mode),
                program_io_applied->contract.initializations.size(),
                program_io_applied->contract.output_probes.size(),
                "unavailable", false);
            LOG_ERROR(SYSTEM)
                << "ProgramIo post-simulation verify failed: "
                << error.what();
        }
    }

    // 运行结束后 dump D2D 端口/链路统计（V0b-2A：无 C2C 端口时恒为 0，供 runner 断言）
    {
        long in = 0, out = 0, busy = 0, stall = 0;
        for (auto &p : g_die_ports.ports) {
            in += p.stats.in_pkts;
            out += p.stats.out_pkts;
            busy += p.stats.busy_cycles;
            stall += p.stats.stall_cycles;
        }
        // V1-b：D2D link 单元实际穿越的包数（idle 时为 0）
        in += g_d2d_link_in_pkts;
        out += g_d2d_link_out_pkts;
        LOG_INFO(SYSTEM) << "[D2D] in_pkts=" << in << " out_pkts=" << out
                         << " busy_cycles=" << busy << " stall_cycles=" << stall;
        // c3 端到端证据：分别证明握手双向控制包与 DATA 都实际穿越 Link，且交付数守恒。
        LOG_INFO(SYSTEM)
            << "[D2D_TYPE] request_in=" << g_d2d_link_in_by_type[REQUEST]
            << " request_out=" << g_d2d_link_out_by_type[REQUEST]
            << " ack_in=" << g_d2d_link_in_by_type[ACK]
            << " ack_out=" << g_d2d_link_out_by_type[ACK]
            << " data_in=" << g_d2d_link_in_by_type[DATA]
            << " data_out=" << g_d2d_link_out_by_type[DATA];
        if (g_d2d_cfg.backend == BACKEND_BEHAVIORAL) {
            LOG_INFO(SYSTEM)
                << "[D2D_BEHA] data_flows=" << g_d2d_behavioral_stats.data_flows
                << " logical_data_packets="
                << g_d2d_behavioral_stats.logical_data_packets
                << " service_cycles=" << g_d2d_behavioral_stats.service_cycles
                << " fixed_cycles=" << g_d2d_behavioral_stats.fixed_cycles
                << " total_d2d_cycles="
                << (g_d2d_behavioral_stats.service_cycles +
                    g_d2d_behavioral_stats.fixed_cycles);
        }
        if (g_d2d_cfg.v5_multiport) {
            std::string loads;
            const auto &v = V5DynamicPortLoads();
            for (size_t i = 0; i < v.size(); ++i) {
                if (i) loads += ",";
                loads += std::to_string(v[i]);
            }
            LOG_INFO(SYSTEM)
                << "[V5_DYNAMIC] selections=" << V5DynamicSelections()
                << " releases=" << V5DynamicReleases()
                << " active=" << V5DynamicActivePins()
                << " loads=" << loads;
        }
        // V1-d2 DATA 逐包完整性：in/out 两侧 pkts/seqhash/csum 相等提供链路无丢/重/
        // 乱序/损坏的强证据；序号是 base-agnostic 连续区间（当前生产从 1 开始），唯一
        // is_end 必须落在 maxseq。cycle span 用于 V1-d3 验证 latency 不改变稳态包间距。
        LOG_INFO(SYSTEM)
            << "[D2D_DATA] in_pkts=" << g_d2d_data_in.pkts
            << " out_pkts=" << g_d2d_data_out.pkts
            << " in_seqhash=" << g_d2d_data_in.seqhash
            << " out_seqhash=" << g_d2d_data_out.seqhash
            << " in_csum=" << g_d2d_data_in.csum
            << " out_csum=" << g_d2d_data_out.csum
            << " out_inorder=" << (g_d2d_data_out.inorder ? 1 : 0)
            << " out_minseq=" << g_d2d_data_out.minseq
            << " out_maxseq=" << g_d2d_data_out.maxseq
            << " out_endseq=" << g_d2d_data_out.endseq
            << " out_end_count=" << g_d2d_data_out.end_count
            << " out_end_length=" << g_d2d_data_out.end_length
            << " in_first_cycle=" << g_d2d_data_in.first_cycle
            << " in_last_cycle=" << g_d2d_data_in.last_cycle
            << " out_first_cycle=" << g_d2d_data_out.first_cycle
            << " out_last_cycle=" << g_d2d_data_out.last_cycle;
        // V5-b/c：striping 的序号空间按 subflow 独立，从 1 重新开始。逐 link 分桶
        // 可同时证明选口、每条子流的 q/r 配额、顺序、唯一尾包和 wire payload 守恒。
        for (const auto &kv : g_v5_subflow_stats) {
            const auto &[idx, source, tag, subflow] = kv.first;
            const V5SubflowStat &st = kv.second;
            LOG_INFO(SYSTEM)
                << "[V5_SUBFLOW] idx=" << idx << " source=" << source
                << " tag=" << tag << " subflow=" << subflow
                << " in=" << st.in_pkts << " out=" << st.out_pkts
                << " in_seqhash=" << st.in_seqhash
                << " out_seqhash=" << st.out_seqhash
                << " in_csum=" << st.in_csum
                << " out_csum=" << st.out_csum
                << " inorder=" << (st.out_inorder ? 1 : 0)
                << " minseq=" << st.out_minseq
                << " maxseq=" << st.out_maxseq
                << " endseq=" << st.out_endseq
                << " ends=" << st.out_end_count
                << " end_length=" << st.out_end_length;
        }

        // V2-b 多跳证据：每个包每跨一次 link，在落点 die 的入口被重新 pin 一次。
        // same>0 说明存在「新旧 exit_port 数值相同」的重写（如 3×1 直线 E→E），这类情形
        // 光看路由结果无法证明重写发生，必须靠本计数。
        LOG_INFO(SYSTEM) << "[D2D_REPIN] total=" << g_d2d_repin_total
                         << " changed=" << g_d2d_repin_changed
                         << " same=" << g_d2d_repin_same;
        // V2-c：逐条**有向** link 的分类型计数。据此可精确断言「经过了哪几条 link、方向序列
        // 是什么、每条各多少包」——全局 [D2D_TYPE] 只有总数，无法区分路径。只打印有流量的 link。
        // 顺序必须与 enums.h 的 Directions 一致：WEST=0, EAST=1, NORTH=2, SOUTH=3, CENTER=4
        static const char *DIRNAME[] = {"W", "E", "N", "S", "C"};
        for (size_t i = 0; i < g_d2d_link_stats.size(); i++) {
            const D2DLinkStat &st = g_d2d_link_stats[i];
            long tot = 0;
            for (int t = 0; t < MSG_TYPE_NUM; t++)
                tot += st.in_by_type[t] + st.out_by_type[t];
            if (tot == 0)
                continue; // 未被使用的 link 不打印，保持输出简洁
            int d = (int)st.dir;
            LOG_INFO(SYSTEM)
                << "[D2D_LINK] idx=" << i << " die" << st.local_die << "->die"
                << st.remote_die << " dir="
                << ((d >= 0 && d < 5) ? DIRNAME[d] : "?")
                << " req_in=" << st.in_by_type[REQUEST]
                << " req_out=" << st.out_by_type[REQUEST]
                << " ack_in=" << st.in_by_type[ACK]
                << " ack_out=" << st.out_by_type[ACK]
                << " data_in=" << st.in_by_type[DATA]
                << " data_out=" << st.out_by_type[DATA];
        }
        // V3-d：生产 bounded_saf 的每条有向 link 多级流水线证据。峰值分别对应
        // whole-flow SAF / link inflight / 远端 RX；stall 分类用于瓶颈和背压归因。
        std::function<void(const std::vector<sc_object *> &)> bounded_dump =
            [&](const std::vector<sc_object *> &objs) {
                for (auto *o : objs) {
                    if (auto *link = dynamic_cast<D2DLinkUnit *>(o);
                        link && link->bound.enabled && link->bound.whole_flow_saf) {
                        LOG_INFO(SYSTEM)
                            << "[D2D_BOUND] idx=" << link->link_idx
                            << " saf_peak=" << link->SafOccMax()
                            << " inflight_peak=" << link->InflightOccMax()
                            << " rx_peak=" << link->RxOccMax()
                            << " saf_full=" << link->SafFullCycles()
                            << " inflight_full=" << link->InflightFullCycles()
                            << " rx_full=" << link->RxFullCycles()
                            << " port_stall=" << link->PortRateStall()
                            << " link_stall=" << link->LinkRateStall()
                            << " inflight_stall=" << link->RateStall()
                            << " rx_stall=" << link->RxBackpressureStall()
                            << " downstream_stall=" << link->DownstreamStall()
                            << " group_stall=" << link->LinkGroupStall();
                    }
                    bounded_dump(o->get_child_objects());
                }
            };
        bounded_dump(sc_get_top_level_objects());
        LOG_INFO(SYSTEM) << "[SAF] reserved_packets="
                         << WholeFlowSafReservedPackets()
                         << " group_reserved_packets=" << WholeFlowSafGroupReservedPackets();

        // V2-c：每 die 的 router 入口包数。中间 die >0 证明包确实穿越了该 die 的 NoC。
        {
            std::string per_die, per_mesh;
            for (size_t i = 0; i < g_die_router_pkts.size(); i++)
                per_die += (i ? "," : "") + std::to_string(g_die_router_pkts[i]);
            for (size_t i = 0; i < g_die_mesh_pkts.size(); i++)
                per_mesh += (i ? "," : "") + std::to_string(g_die_mesh_pkts[i]);
            LOG_INFO(SYSTEM) << "[DIE_ACT] router_pkts=" << per_die
                             << " mesh_pkts=" << per_mesh;
            std::string noc_send, noc_stall;
            for (size_t i = 0; i < g_die_noc_sends.size(); ++i)
                noc_send += (i ? "," : "") + std::to_string(g_die_noc_sends[i]);
            for (size_t i = 0; i < g_die_noc_stalls.size(); ++i)
                noc_stall += (i ? "," : "") + std::to_string(g_die_noc_stalls[i]);
            LOG_INFO(SYSTEM) << "[NOC_ACT] sends=" << noc_send
                             << " stalls=" << noc_stall
                             << " d2d_source_stalls=" << g_d2d_source_stalls;
        }
        {
            std::string sig;
            for (const auto &kv : g_flow_done_cycle)
                sig += (sig.empty() ? "" : ",") +
                       std::to_string(std::get<0>(kv.first)) + ":" +
                       std::to_string(std::get<1>(kv.first)) + ":" +
                       std::to_string(std::get<2>(kv.first)) + "@" +
                       std::to_string(kv.second);
            LOG_INFO(SYSTEM) << "[FLOW_DONE] " << sig;
            LOG_INFO(SYSTEM) << "[SAF_ADMIT] success="
                             << g_saf_admission_successes
                             << " reject=" << g_saf_admission_rejects;
        }
    }

    {
        for (const auto &stat : CollectiveFabricLinkStats()) {
            LOG_INFO(SYSTEM)
                << "[COLL_LINK] tree=" << stat.tree_id
                << " router=" << stat.router_id
                << " output=" << static_cast<unsigned>(stat.output)
                << " flits=" << stat.committed_flits
                << " stalls=" << stat.stalled_attempts;
        }
        for (const auto &stat : CollectiveSharedLinkStats()) {
            LOG_INFO(SYSTEM)
                << "[COLL_SHARED] router=" << stat.router_id
                << " output=" << static_cast<unsigned>(stat.output)
                << " normal_flits=" << stat.normal_flits
                << " collective_flits=" << stat.collective_flits;
        }
        size_t endpoint_residual = 0, dte_tokens = 0, event_residual = 0;
        std::vector<WorkerCoreExecutor *> p2p_active_cores;
        std::function<void(const std::vector<sc_object *> &)> coll_drain =
            [&](const std::vector<sc_object *> &objs) {
            for (auto *o : objs) {
                if (auto *core = dynamic_cast<WorkerCoreExecutor *>(o)) {
                    const bool dedicated =
                        core->dte_control_core != nullptr;
                    DteControlCoreStatistics controller_stats;
                    DteControlResidual controller_residual;
                    if (dedicated) {
                        controller_stats =
                            core->dte_control_core->statistics();
                        controller_residual =
                            core->dte_control_core->Residual();
                    }
                    std::cout
                        << "[DTE_CTRL_STATS] core=" << core->cid
                        << " mode="
                        << (dedicated ? "dual_dte_dedicated"
                                      : "legacy_shared")
                        << " submitted=" << controller_stats.enqueued
                        << " dispatched=" << controller_stats.dispatched
                        << " completed=" << controller_stats.completed
                        << " queue_stalls="
                        << controller_stats.queue_stalls
                        << " max_queue_occupancy="
                        << controller_stats.max_queue_occupancy
                        << " queued="
                        << controller_residual.queued_commands
                        << " outstanding="
                        << (controller_residual.inflight_commands +
                            controller_residual.active_transfers +
                            controller_residual.logical_tokens +
                            controller_residual.pending_notifications)
                        << "\n";
                    if (core->dte != nullptr) {
                        const auto &dte_stats = core->dte->statistics();
                        std::cout
                            << "[DTE_STATS] core=" << core->cid
                            << " issued=" << dte_stats.physical_issued
                            << " completed=" << dte_stats.completed
                            << " cancelled=" << dte_stats.cancelled
                            << " backpressure_stalls="
                            << dte_stats.backpressure_stalls
                            << " pending=" << core->dte->PendingCount()
                            << " active=" << core->dte->ActiveCount()
                            << " inflight=" << core->dte->InflightCount()
                            << " max_active=" << core->dte->MaxActiveCount()
                            << "\n";
                    }
                    const auto endpoint =
                        core->CollectiveEndpointResidualState();
                    endpoint_residual += endpoint.Total();
                    if (endpoint.Total() != 0) {
                        LOG_INFO(SYSTEM)
                            << "[COLL_ENDPOINT_RESIDUAL] core=" << core->cid
                            << " data=" << endpoint.collective_data
                            << " reduce=" << endpoint.collective_reduce
                            << " stream_routes=" << endpoint.stream_routes
                            << " stream_sessions=" << endpoint.stream_sessions
                            << " stream_assembler=" << endpoint.stream_assembler
                            << " multicast_posts=" << endpoint.multicast_posts
                            << " multicast_routes=" << endpoint.multicast_routes
                            << " serialized_wires=" << endpoint.serialized_wires
                            << " multicast_reassembly="
                            << endpoint.multicast_reassembly
                            << " core_vector_sessions="
                            << endpoint.core_vector_sessions
                            << " p2p=" << endpoint.p2p;
                    }
                    dte_tokens += core->DteOutstandingCount();
                    event_residual += core->EventResidual();
                    const auto &p2p = core->P2pStats();
                    if (p2p.source_read_bytes != 0 || p2p.wire_bytes != 0 ||
                        p2p.wire_fragments != 0 ||
                        p2p.noc_rx_write_bytes != 0 ||
                        p2p.tx_local_completions != 0 ||
                        p2p.rx_local_completions != 0 ||
                        p2p.admission_requests_sent != 0 ||
                        p2p.admission_requests_received != 0 ||
                        p2p.admission_acks_sent != 0 ||
                        p2p.admission_acks_received != 0 ||
                        p2p.duplicate_requests_suppressed != 0 ||
                        p2p.request_conflicts_rejected != 0 ||
                        p2p.request_aborts != 0 ||
                        p2p.completion_acks_sent != 0 ||
                        p2p.completion_acks_received != 0)
                        p2p_active_cores.push_back(core);
                }
                if (auto *router = dynamic_cast<RouterUnit *>(o)) {
                    if (router->reduce_stream_engine) {
                        const auto &stream =
                            router->reduce_stream_engine->Stats();
                        const auto &dca =
                            router->reduce_stream_engine->DcaStats();
                        LOG_INFO(SYSTEM)
                            << "[COLL_STREAM] router=" << router->rid
                            << " headers_in=" << stream.headers_in
                            << " data_in=" << stream.data_in
                            << " headers_out=" << stream.headers_out
                            << " data_out=" << stream.data_out
                            << " assembler_stalls="
                            << stream.assembler_backpressure
                            << " issue_stalls=" << stream.issue_backpressure
                            << " egress_stalls=" << stream.egress_backpressure;
                        LOG_INFO(SYSTEM)
                            << "[COLL_DCA] router=" << router->rid
                            << " core_issues=" << dca.core_issued
                            << " dca_issues=" << dca.dca_issued
                            << " completions=" << dca.completions
                            << " core_stalls=" << dca.core_wait_cycles
                            << " dca_stalls=" << dca.dca_wait_cycles
                            << " submit_stalls="
                            << dca.issue_queue_backpressure
                            << " inflight_peak=" << dca.inflight_peak
                            << " result_stalls="
                            << dca.result_backpressure_cycles;
                    }
                }
                coll_drain(o->get_child_objects());
            }
        };
        coll_drain(sc_get_top_level_objects());
        const size_t timing_residual =
            P2pSharedTimingSidebandRuntime::Residual();
        endpoint_residual += timing_residual;
        for (const WorkerCoreExecutor *core : p2p_active_cores) {
            const auto &p2p = core->P2pStats();
            LOG_INFO(SYSTEM)
                << "[P5 P2P STATS] core=" << core->cid
                << " source_read_bytes=" << p2p.source_read_bytes
                << " sram_source_read_bytes="
                << p2p.sram_source_read_bytes
                << " hbm_source_read_bytes="
                << p2p.hbm_source_read_bytes
                << " wire_bytes=" << p2p.wire_bytes
                << " wire_fragments=" << p2p.wire_fragments
                << " noc_rx_write_bytes=" << p2p.noc_rx_write_bytes
                << " tx_local_completions="
                << p2p.tx_local_completions
                << " rx_local_completions="
                << p2p.rx_local_completions
                << " admission_requests_sent="
                << p2p.admission_requests_sent
                << " admission_requests_received="
                << p2p.admission_requests_received
                << " admission_acks_sent="
                << p2p.admission_acks_sent
                << " admission_acks_received="
                << p2p.admission_acks_received
                << " duplicate_requests_suppressed="
                << p2p.duplicate_requests_suppressed
                << " request_conflicts_rejected="
                << p2p.request_conflicts_rejected
                << " request_aborts=" << p2p.request_aborts
                << " completion_acks_sent="
                << p2p.completion_acks_sent
                << " completion_acks_received="
                << p2p.completion_acks_received
                << " source_checksum=" << p2p.last_source_checksum
                << " wire_checksum=" << p2p.last_wire_checksum
                << " destination_checksum="
                << p2p.last_destination_checksum;
            const size_t p2p_residual =
                core->P2pEndpointResidual() +
                static_cast<size_t>(WholeFlowSafReservedPackets()) +
                static_cast<size_t>(WholeFlowSafGroupReservedPackets());
            LOG_INFO(SYSTEM)
                << "[P5 P2P DRAIN] core=" << core->cid
                << " residual=" << p2p_residual;
        }
        LOG_INFO(SYSTEM)
            << "[P5 P2P TIMING DRAIN] residual=" << timing_residual;
        LOG_INFO(SYSTEM)
            << "[COLL_DRAIN] tree_entries=" << CollectiveTreeEntryCount()
            << " reduce_nodes=" << CollectiveReduceNodeCount()
            << " barriers=" << CollectiveBarrierStateCount()
            << " gather=" << CollectiveGatherReorderStateCount()
            << " reduce_rx=" << CollectiveReduceRxStateCount()
            << " endpoints=" << endpoint_residual
            << " dte_tokens=" << dte_tokens
            << " event=" << event_residual;
    }

    // 结束态 drain 不变量（V1 验收）：遍历 SystemC 层级，累加所有 RouterUnit 的残留
    // （未释放的 in/out lock ref + 各方向 data/ctrl buffer + host buffer）。仿真正常结束时
    // 应为 0——非 0 说明有锁泄漏或滞留包（如尾包丢失 / 别名导致的 ref 未归零）。
    {
        std::function<long(const std::vector<sc_object *> &)> resid =
            [&](const std::vector<sc_object *> &objs) -> long {
            long r = 0;
            for (auto *o : objs) {
                if (auto *ru = dynamic_cast<RouterUnit *>(o))
                    r += ru->residual();
                r += resid(o->get_child_objects());
            }
            return r;
        };
        LOG_INFO(SYSTEM) << "[DRAIN] router_residual="
                         << resid(sc_get_top_level_objects());
        std::function<bool(const std::vector<sc_object *> &, bool)> credit_ok =
            [&](const std::vector<sc_object *> &objs, bool data) -> bool {
            for (auto *o : objs) {
                if (auto *ru = dynamic_cast<RouterUnit *>(o)) {
                    bool ok = data ? ru->D2DDataCreditsBalanced()
                                   : ru->D2DCtrlCreditsBalanced();
                    if (!ok)
                        return false;
                }
                if (!credit_ok(o->get_child_objects(), data))
                    return false;
            }
            return true;
        };
        LOG_INFO(SYSTEM) << "[CREDIT] data_balanced="
                         << (credit_ok(sc_get_top_level_objects(), true) ? 1 : 0)
                         << " ctrl_balanced="
                         << (credit_ok(sc_get_top_level_objects(), false) ? 1 : 0);
    }
    {
        std::function<long(const std::vector<sc_object *> &)> resid =
            [&](const std::vector<sc_object *> &objs) -> long {
            long r = 0;
            for (auto *o : objs) {
                if (auto *link = dynamic_cast<D2DLinkUnit *>(o))
                    r += link->residual();
                r += resid(o->get_child_objects());
            }
            return r;
        };
        LOG_INFO(SYSTEM) << "[DRAIN] d2d_link_residual="
                         << resid(sc_get_top_level_objects()) +
                                D2DBehavioralFlowResidual() + V5DynamicActivePins();
    }

    LOG_INFO(SYSTEM) << "[START_DATA] "
                     << StartDataTracker::Instance().Summary();

    // output_lock_ref 峰值：>=2 证明同 tag 多流共享同一把锁（多发一聚合，tag-only 核心语义）。
    LOG_INFO(SYSTEM) << "[LOCK] max_output_ref=" << g_max_output_lock_ref;

    // HOST lane 接收统计（V1-pre 3b-2b）：每 lane DONE/ACK 数 + 错配数
    // （消息应到 HostLaneOfCore(source_)；mismatch>0 说明路由送错 lane）。
    {
        long done_tot = 0, ack_tot = 0;
        std::string per_lane;
        for (size_t i = 0; i < g_host_lane_done.size(); i++) {
            done_tot += g_host_lane_done[i];
            ack_tot += g_host_lane_ack[i];
            per_lane += (i ? "," : "") + std::to_string(g_host_lane_done[i]);
        }
        LOG_INFO(SYSTEM) << "[HOSTLANE] done_total=" << done_tot
                         << " ack_total=" << ack_tot
                         << " mismatch=" << g_host_lane_mismatch
                         << " per_lane_done=" << per_lane;

        // 多重集签名（src:count）——严格证明无丢包/重复需比对多重集而非总数
        std::string dsig, asig;
        for (auto &kv : g_host_done_src) // source:count
            dsig += std::to_string(kv.first) + ":" + std::to_string(kv.second) + ",";
        for (auto &kv : g_host_ack_sig) // source:tag:count
            asig += std::to_string(kv.first.first) + ":" +
                    std::to_string(kv.first.second) + ":" +
                    std::to_string(kv.second) + ",";
        LOG_INFO(SYSTEM) << "[HOSTSIG] done=" << dsig << " ack=" << asig;
    }

    // destroy_dram_areas();
    // destroy_cache_structures();
    // event_engine->dump_traced_file();
    sc_close_vcd_trace_file(tf);

    SystemCleanup();
    CloseLogFiles();

    clock_t end = clock();

    if (correct_exit) {
        LOG_INFO(SYSTEM) << "Total Real-time Cost: "
                         << (double)(end - start) / CLOCKS_PER_SEC << "s";
    } else {
        LOG_WARN(SYSTEM) << "Simulation terminated abnormally";
    }

    ofstream outfile("simulation_result_df_pd.txt", ios::app);
    if (outfile.is_open()) {
        outfile << "Total Real-time Cost: "
                << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
        outfile.close();
    } else {
        LOG_ERROR(SYSTEM) << "Unable to open file for writing timestamp";
    }
    delete event_engine;
    // V2-d2：协议 watchdog 判定停顿 ⇒ 仿真器**主动**非零退出（不依赖测试框架超时）。
    if (g_protocol_stall_detected) {
        LOG_ERROR(SYSTEM) << "[PROTO_WAIT] simulation aborted by protocol "
                             "progress watchdog at cycle "
                          << g_protocol_stall_cycle;
        return 3;
    }
    if (p5_memory_probe_failed)
        return 4;
    if (p6_memory_probe_failed)
        return 5;
    if (p8_double_buffer_probe_failed)
        return 6;
    if (program_io_failed)
        return 7;
    return 0;
}
