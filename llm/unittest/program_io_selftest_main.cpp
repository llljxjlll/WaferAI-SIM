#include "frontend/program_finalizer.h"
#include "frontend/program_io.h"
#include "defs/spec.h"
#include "die/port.h"
#include "memory/behavioral_hbm_backend.h"
#include "memory/hbm_address_map.h"
#include "memory/hbm_runtime.h"
#include "isa/program_format.h"
#include "memory/sram/sram_region.h"
#include "memory/sram/sram_storage.h"
#include "nlohmann/json.hpp"
#include "trace/Event_engine.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

// The isolated selftest never installs an Event_engine on its AccessUnits.
// Supplying the sole referenced trace hook keeps this target independent from
// the legacy trace writer and its unrelated warning surface.
void Event_engine::add_event(string, string, string, Trace_event_util, sc_time,
                             unsigned, string) {}

int DIE_COUNT = 1;
std::unordered_map<std::string, HBMProfile> g_hbm_profiles;
std::vector<HBMStackConfig> g_hbm_stacks;
std::vector<HBMChannelConfig> g_hbm_channels;

namespace frontend::program_io {
void RunExactMoeCalibrationDynamicRootByteClosureSelfTest();
void RunExactFourStreamUnfusedTerminalReuseSelfTest();

struct HBMRuntimeSelfTestPeer {
    static void Add(HBMRuntime *runtime, int stack_id, int channel_id,
                    std::unique_ptr<HBMBackend> backend) {
        runtime->instances_.push_back(
            {stack_id, channel_id, std::move(backend), nullptr});
    }
};
} // namespace frontend::program_io

namespace {

using Json = nlohmann::json;
namespace io = frontend::program_io;

void Require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void ExpectFailure(Function &&function, const std::string &name) {
    try {
        function();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error("expected failure: " + name);
}

struct HbmTopologyGuard {
    int die_count = DIE_COUNT;
    std::unordered_map<std::string, HBMProfile> profiles =
        g_hbm_profiles;
    std::vector<HBMStackConfig> stacks = g_hbm_stacks;
    std::vector<HBMChannelConfig> channels = g_hbm_channels;
    AddressPolicyConfig address_policy = g_address_policy;

    ~HbmTopologyGuard() {
        DIE_COUNT = die_count;
        g_hbm_profiles = std::move(profiles);
        g_hbm_stacks = std::move(stacks);
        g_hbm_channels = std::move(channels);
        g_address_policy = std::move(address_policy);
    }
};

void ConfigureBehavioralHbm() {
    DIE_COUNT = 1;
    HBMProfile profile;
    profile.channels_per_stack = 1;
    profile.pseudo_channels_per_channel = 1;
    g_hbm_profiles = {{"program_io", profile}};
    HBMStackConfig stack;
    stack.stack_id = 0;
    stack.compute_die_id = 0;
    stack.profile = "program_io";
    stack.capacity_bytes = 1024;
    stack.backend_kind = HBMBackendKind::kBehavioral;
    g_hbm_stacks = {stack};
    g_hbm_channels = {{0, 0, 0}};
    g_address_policy = AddressPolicyConfig{};
    g_address_policy.active = true;
    g_address_policy.mode = AddressPolicyMode::kNumaLocalInterleave;
    g_address_policy.home_ranges = {{0, 0, 1024}};
    g_address_policy.stack_interleave_bytes = 64;
    g_address_policy.channel_interleave_bytes = 64;
    g_address_policy.pseudo_channel_interleave_bytes = 64;
    ValidateAddressPolicy();
}

class FailingAddressBackend : public BehavioralHBMBackend {
public:
    FailingAddressBackend(const BehavioralHBMBackendConfig &config,
                          uint64_t fail_address)
        : BehavioralHBMBackend(config), fail_address_(fail_address) {}

    void DebugSeed(uint64_t address,
                   const std::vector<uint8_t> &payload) override {
        if (address >= fail_address_)
            throw std::runtime_error("injected HBM seed failure");
        BehavioralHBMBackend::DebugSeed(address, payload);
    }

private:
    uint64_t fail_address_ = 0;
};

class RejectingBackend : public HBMBackend {
public:
    void Submit(const std::shared_ptr<HBMBackendTransaction> &) override {}
    const HBMBackendStats &Stats() const override { return stats_; }

private:
    HBMBackendStats stats_;
};

bool SameStats(const HBMBackendStats &left,
               const HBMBackendStats &right) {
    return left.requests == right.requests && left.reads == right.reads &&
           left.writes == right.writes && left.bytes == right.bytes &&
           left.completed == right.completed &&
           left.failed == right.failed &&
           left.service_time == right.service_time;
}

std::string ReadText(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open input file: " + path);
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

std::vector<uint8_t> ReadBytes(const std::string &path) {
    const std::string text = ReadText(path);
    return std::vector<uint8_t>(text.begin(), text.end());
}

void WriteBytes(const std::string &path, const std::vector<uint8_t> &bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open output file: " + path);
    output.write(reinterpret_cast<const char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("cannot write output file: " + path);
}

std::string StableId(const std::string &kind, const std::string &schema,
                     const Json &semantic) {
    const Json identity{{"kind", kind},
                        {"schema_version", schema},
                        {"semantic_key", semantic}};
    return kind + "_" + io::Sha256Hex(identity.dump()).substr(0, 16);
}

void RestableChild(Json &value, const std::string &kind,
                   const std::string &schema) {
    Json semantic = value;
    semantic.erase("id");
    value["id"] = StableId(kind, schema, semantic);
}

void RestableContract(Json &value) {
    Json semantic = value;
    semantic.erase("schema_version");
    semantic.erase("producer_pass");
    semantic.erase("id");
    value["id"] = StableId(
        "program_io_contract", std::string(io::kSchemaVersion), semantic);
}

Json Blob(const std::vector<uint8_t> &bytes, const std::string &base64) {
    Json semantic{{"bytes_base64", base64},
                  {"length_bytes", bytes.size()},
                  {"sha256", io::Sha256Hex(bytes)}};
    Json result = semantic;
    result["id"] = StableId(
        "program_blob", "wafer_frontend.program_blob/v1alpha1", semantic);
    return result;
}

Json Slice(const std::string &value) {
    return {{"value_id", value},
            {"offset", Json::array({0})},
            {"shape", Json::array({2})}};
}

Json HbmTargetJson(const std::string &symbol, uint64_t index,
                   const std::string &name) {
    return {{"kind", "hbm"},
            {"program_symbol_ref", symbol},
            {"finalized_symbol_index", index},
            {"expected_symbol_name", name},
            {"state_abi_id", "state_abi"},
            {"state_ref", "state"},
            {"hbm_binding_ref", "hbm_binding"}};
}

Json Initialization(const Json &blob) {
    Json target{{"kind", "sram"},
                {"runtime_core_id", 0},
                {"program_symbol_ref", "label_input"},
                {"finalized_symbol_index", 1},
                {"expected_symbol_name", "input_label"},
                {"buffer_abi_id", "abi_input"},
                {"storage_id", "storage_input"},
                {"value_id", "value_input"},
                {"tensor_slice", Slice("value_input")},
                {"dtype", "fp16"},
                {"layout", "row_major"}};
    Json semantic{{"target", target},
                  {"offset_bytes", 0},
                  {"length_bytes", 4},
                  {"blob_ref", blob.at("id")},
                  {"purpose", "activation"}};
    Json result = semantic;
    result["id"] = StableId(
        "program_sram_initialization",
        "wafer_frontend.program_sram_initialization/v1alpha2", semantic);
    return result;
}

Json Probe(const Json &blob) {
    Json target{{"kind", "sram"},
                {"runtime_core_id", 0},
                {"program_symbol_ref", "label_output"},
                {"finalized_symbol_index", 2},
                {"expected_symbol_name", "output_label"},
                {"buffer_abi_id", "abi_output"},
                {"storage_id", "storage_output"},
                {"value_id", "value_output"},
                {"tensor_slice", Slice("value_output")},
                {"dtype", "fp16"},
                {"layout", "row_major"}};
    Json semantic{{"target", target},
                  {"offset_bytes", 0},
                  {"length_bytes", 4},
                  {"blob_ref", blob.at("id")},
                  {"comparison", "exact_bytes/v1"},
                  {"capture", "after_program/v1"}};
    Json result = semantic;
    result["id"] = StableId(
        "program_output_probe",
        "wafer_frontend.program_output_probe/v1alpha2", semantic);
    return result;
}

Json Sidecar() {
    const Json input = Blob({1, 2, 3, 4}, "AQIDBA==");
    const Json output = Blob({9, 8, 7, 6}, "CQgHBg==");
    Json blobs = Json::array({input, output});
    std::sort(blobs.begin(), blobs.end(), [](const Json &left,
                                             const Json &right) {
        return left.at("id").get<std::string>() <
               right.at("id").get<std::string>();
    });
    Json semantic{{"mode", "functional"},
                  {"source_linked_manifest_id", "manifest_id"},
                  {"source_linked_manifest_digest", std::string(64, 'c')},
                  {"program_artifact_sha256", std::string(64, 'a')},
                  {"blobs", blobs},
                  {"initializations", Json::array({Initialization(input)})},
                  {"output_probes", Json::array({Probe(output)})}};
    Json result = semantic;
    result["schema_version"] = io::kSchemaVersion;
    result["producer_pass"] = "program_io_cpp_selftest";
    result["id"] = StableId("program_io_contract", std::string(io::kSchemaVersion),
                            semantic);
    return result;
}


void TestTaggedTargets() {
    const Json sidecar = Sidecar();
    const io::Contract parsed = io::Parse(sidecar.dump());
    Require(std::holds_alternative<io::SramTarget>(
                parsed.initializations[0].target) &&
                std::holds_alternative<io::SramTarget>(
                    parsed.output_probes[0].target),
            "v1alpha2 SRAM target tag was not preserved");

    Json hbm = sidecar;
    hbm["initializations"][0]["target"] =
        HbmTargetJson("state_load", 3, "state_load");
    hbm["initializations"][0]["purpose"] = "state";
    RestableChild(
        hbm["initializations"][0], "program_sram_initialization",
        "wafer_frontend.program_sram_initialization/v1alpha2");
    hbm["output_probes"][0]["target"] =
        HbmTargetJson("state_store", 4, "state_store");
    RestableChild(
        hbm["output_probes"][0], "program_output_probe",
        "wafer_frontend.program_output_probe/v1alpha2");
    RestableContract(hbm);
    const io::Contract parsed_hbm = io::Parse(hbm.dump());
    Require(std::holds_alternative<io::HbmTarget>(
                parsed_hbm.initializations[0].target) &&
                std::holds_alternative<io::HbmTarget>(
                    parsed_hbm.output_probes[0].target),
            "v1alpha2 HBM target tag was not preserved");

    Json old_version = sidecar;
    old_version["schema_version"] =
        "wafer_frontend.program_io_contract/v1alpha1";
    ExpectFailure([&] { (void)io::Parse(old_version.dump()); },
                  "old v1alpha1 contract");

    Json flat = sidecar;
    const Json old_target = flat["initializations"][0]["target"];
    flat["initializations"][0].erase("target");
    for (const auto &field : old_target.items())
        flat["initializations"][0][field.key()] = field.value();
    ExpectFailure([&] { (void)io::Parse(flat.dump()); }, "old flat target");


    Json physical = hbm;
    physical["initializations"][0]["target"]["address"] = 4096;
    ExpectFailure([&] { (void)io::Parse(physical.dump()); },
                  "HBM physical address in sidecar");
}
void TestStrictParser() {
    const Json sidecar = Sidecar();
    const io::Contract parsed = io::Parse(sidecar.dump());
    Require(parsed.mode == io::Mode::FUNCTIONAL && parsed.blobs.size() == 2 &&
                parsed.initializations.size() == 1 &&
                parsed.output_probes.size() == 1,
            "strict parser did not preserve the canonical ProgramIo contract");
    Require(io::Sha256Hex(std::string_view{}) ==
                "e3b0c44298fc1c149afbf4c8996fb924"
                "27ae41e4649b934ca495991b7852b855",
            "ProgramIo SHA-256 is not canonical");
    Require(io::Sha256Hex(std::vector<uint8_t>{}) ==
                "e3b0c44298fc1c149afbf4c8996fb924"
                "27ae41e4649b934ca495991b7852b855",
            "ProgramIo empty artifact SHA-256 is not canonical");

    Json unknown = sidecar;
    unknown["unknown"] = 0;
    ExpectFailure([&] { (void)io::Parse(unknown.dump()); }, "unknown field");

    Json bad_blob = sidecar;
    bad_blob["blobs"][0]["bytes_base64"] = "AQIDBA==\n";
    ExpectFailure([&] { (void)io::Parse(bad_blob.dump()); },
                  "non-canonical base64");

    const std::string duplicate = sidecar.dump();
    ExpectFailure(
        [&] {
            (void)io::Parse(duplicate.substr(0, duplicate.size() - 1) +
                            ",\"mode\":\"functional\"}");
        },
        "duplicate object key");

    Json unstable = sidecar;
    unstable["id"] = "program_io_contract_impostor";
    ExpectFailure([&] { (void)io::Parse(unstable.dump()); }, "unstable id");
}

void TestInt32SramDType() {
    Json sidecar = Sidecar();
    for (const char *field : {"initializations", "output_probes"}) {
        Json &entry = sidecar[field][0];
        entry["target"]["dtype"] = "int32";
        entry["target"]["tensor_slice"]["shape"] = Json::array({1});
        RestableChild(
            entry,
            std::string(field) == "initializations"
                ? "program_sram_initialization"
                : "program_output_probe",
            std::string(field) == "initializations"
                ? "wafer_frontend.program_sram_initialization/v1alpha2"
                : "wafer_frontend.program_output_probe/v1alpha2");
    }
    RestableContract(sidecar);
    const io::Contract parsed = io::Parse(sidecar.dump());
    Require(std::get<io::SramTarget>(parsed.initializations[0].target).dtype ==
                io::DType::INT32 &&
                std::get<io::SramTarget>(parsed.output_probes[0].target).dtype ==
                    io::DType::INT32,
            "INT32 SRAM initialization/probe dtype was not preserved");

    Json unknown = sidecar;
    unknown["initializations"][0]["target"]["dtype"] = "bf16";
    RestableChild(
        unknown["initializations"][0], "program_sram_initialization",
        "wafer_frontend.program_sram_initialization/v1alpha2");
    RestableContract(unknown);
    ExpectFailure([&] { (void)io::Parse(unknown.dump()); },
                  "unknown SRAM dtype");

    Json hbm = Sidecar();
    hbm["initializations"][0]["target"] =
        HbmTargetJson("state_load", 3, "state_load");
    hbm["initializations"][0]["target"]["dtype"] = "int32";
    hbm["initializations"][0]["purpose"] = "state";
    RestableChild(
        hbm["initializations"][0], "program_sram_initialization",
        "wafer_frontend.program_sram_initialization/v1alpha2");
    RestableContract(hbm);
    ExpectFailure([&] { (void)io::Parse(hbm.dump()); },
                  "INT32 HBM target field");
}

sram::Config MemoryConfig(bool payload) {
    sram::Config config;
    config.capacity_bytes = 1024;
    config.allocation_alignment_bytes = 16;
    config.bank_count = 2;
    config.bank_interleave_bytes = 16;
    config.real_data_path = payload;
    config.manual_regions = true;
    config.manual_memory_schedule = true;
    config.regions.push_back(
        {"comm", 0, 1024, sram::AllocatorKind::kBlock, false,
         {sram::Initiator::kCompute, sram::Initiator::kDte,
          sram::Initiator::kLsu, sram::Initiator::kNocRx,
          sram::Initiator::kLegacy}});
    return config;
}

io::ResolvedInitialization ResolvedInput(uint32_t core, uint64_t address,
                                         std::vector<uint8_t> bytes) {
    io::Initialization source;
    source.id = "init_" + std::to_string(core);
    io::SramTarget target;
    target.runtime_core_id = core;
    source.target = target;
    source.length_bytes = bytes.size();
    return {source, "region", "comm", 0, 1024, address, std::move(bytes),
            std::nullopt};
}

io::ResolvedOutputProbe ResolvedProbe(uint32_t core, uint64_t address,
                                     std::vector<uint8_t> bytes) {
    io::OutputProbe source;
    source.id = "probe_" + std::to_string(core);
    io::SramTarget target;
    target.runtime_core_id = core;
    source.target = target;
    source.length_bytes = bytes.size();
    return {source, "region", "comm", 0, 1024, address, std::move(bytes),
            std::nullopt};
}

io::HbmTarget HbmTarget(uint64_t address) {
    io::HbmTarget target;
    target.program_symbol_ref = "hbm_" + std::to_string(address);
    target.expected_symbol_name = target.program_symbol_ref;
    target.state_abi_id = "state_abi_" + std::to_string(address);
    target.state_ref = "state_" + std::to_string(address);
    target.hbm_binding_ref = "binding_" + std::to_string(address);
    return target;
}

io::ResolvedInitialization ResolvedHbmInput(
    uint64_t address, std::vector<uint8_t> bytes) {
    io::Initialization source;
    source.id = "hbm_init_" + std::to_string(address);
    source.target = HbmTarget(address);
    source.length_bytes = bytes.size();
    const uint64_t size = bytes.size();
    const io::HbmTarget &target = std::get<io::HbmTarget>(source.target);
    return {source, target.program_symbol_ref, target.expected_symbol_name,
            address, size, address, std::move(bytes),
            io::ResolvedHbmRange{0, address, size}};
}

io::ResolvedOutputProbe ResolvedHbmProbe(
    uint64_t address, std::vector<uint8_t> bytes) {
    io::OutputProbe source;
    source.id = "hbm_probe_" + std::to_string(address);
    source.target = HbmTarget(address);
    source.length_bytes = bytes.size();
    const uint64_t size = bytes.size();
    const io::HbmTarget &target = std::get<io::HbmTarget>(source.target);
    return {source, target.program_symbol_ref, target.expected_symbol_name,
            address, size, address, std::move(bytes),
            io::ResolvedHbmRange{0, address, size}};
}

void TestSeedAndProbe() {
    sram::RegionTable regions(MemoryConfig(true));
    sram::Storage storage(1024, true);
    sram::AccessUnit access("program_io_access", regions, storage);
    io::ResolvedContract resolved;
    resolved.id = "contract";
    resolved.mode = io::Mode::FUNCTIONAL;
    resolved.initializations.push_back(ResolvedInput(0, 64, {1, 2, 3, 4}));
    resolved.output_probes.push_back(ResolvedProbe(0, 128, {9, 8, 7, 6}));
    io::Bindings bindings;
    bindings.sram_by_runtime_core.emplace(0, &access);

    const io::Applied applied = io::ApplyBeforeSimulation(resolved, bindings);
    const sram::DebugSnapshot input = access.DebugPeek(64, 4);
    Require(input.payload == std::vector<uint8_t>({1, 2, 3, 4}) &&
                std::all_of(input.valid.begin(), input.valid.end(),
                            [](uint8_t value) { return value != 0; }),
            "ProgramIo did not seed exact BORROWED bytes and validity");
    const io::Result before = io::VerifyAfterSimulation(applied);
    Require(!before.Passed() && !before.probes[0].all_bytes_valid,
            "ProgramIo accepted an unwritten OWNED output");
    access.DebugSeed(128, {9, 8, 7, 6});
    const io::Result after = io::VerifyAfterSimulation(applied);
    Require(after.Passed() && after.probes.size() == 1,
            "ProgramIo did not accept an exact valid OWNED output");
    access.DebugSeed(128, {9, 8, 7, 5});
    const io::Result mismatch = io::VerifyAfterSimulation(applied);
    Require(!mismatch.Passed() && mismatch.probes[0].all_bytes_valid &&
                !mismatch.probes[0].exact_match,
            "ProgramIo failed to distinguish validity from exact bytes");

    sram::RegionTable no_payload_regions(MemoryConfig(false));
    sram::Storage no_payload_storage(1024, false);
    sram::AccessUnit no_payload(
        "program_io_no_payload", no_payload_regions, no_payload_storage);
    const sram::DebugSnapshot original = access.DebugPeek(32, 4);
    io::ResolvedContract rollback = resolved;
    rollback.initializations = {
        ResolvedInput(0, 32, {7, 7, 7, 7}),
        ResolvedInput(1, 48, {8, 8, 8, 8}),
    };
    io::Bindings rollback_bindings = bindings;
    rollback_bindings.sram_by_runtime_core.emplace(1, &no_payload);
    ExpectFailure(
        [&] {
            (void)io::ApplyBeforeSimulation(rollback, rollback_bindings);
        },
        "transactional seed rollback");
    const sram::DebugSnapshot restored = access.DebugPeek(32, 4);
    Require(restored.payload == original.payload &&
                restored.valid == original.valid,
            "failed ProgramIo apply did not restore bytes and validity");
}


void TestHbmSeedProbeAndRollback() {
    HbmTopologyGuard topology_guard;
    ConfigureBehavioralHbm();
    BehavioralHBMBackendConfig hbm_config;
    hbm_config.bandwidth_GBps = 1.0;

    HBMRuntime runtime;
    io::HBMRuntimeSelfTestPeer::Add(
        &runtime, 0, 0,
        std::make_unique<BehavioralHBMBackend>(hbm_config));
    HBMBackend *backend = runtime.Find(0, 0)->backend.get();
    const HBMBackendStats stats_before = backend->Stats();

    sram::RegionTable regions(MemoryConfig(true));
    sram::Storage storage(1024, true);
    sram::AccessUnit access(
        "program_io_hbm_access", regions, storage);
    io::ResolvedContract resolved;
    resolved.id = "hbm_contract";
    resolved.mode = io::Mode::FUNCTIONAL;
    resolved.initializations = {
        ResolvedInput(0, 32, {1, 2, 3, 4}),
        ResolvedHbmInput(0, {5, 6, 7, 8}),
    };
    resolved.output_probes = {
        ResolvedProbe(0, 128, {9, 8, 7, 6}),
        ResolvedHbmProbe(16, {0, 0, 0, 0}),
    };
    io::Bindings bindings;
    bindings.sram_by_runtime_core.emplace(0, &access);
    bindings.hbm_runtime = &runtime;

    const io::Applied applied =
        io::ApplyBeforeSimulation(resolved, bindings);
    const HBMRuntimeDebugSnapshot seeded =
        runtime.DebugPeek(0, 0, 4);
    Require(seeded.payload == std::vector<uint8_t>({5, 6, 7, 8}) &&
                std::all_of(
                    seeded.chunks[0].backend.present.begin(),
                    seeded.chunks[0].backend.present.end(),
                    [](uint8_t byte) { return byte != 0; }),
            "ProgramIo did not seed exact behavioral HBM bytes/presence");
    const io::Result absent = io::VerifyAfterSimulation(applied);
    Require(!absent.Passed() && absent.probes.size() == 2 &&
                absent.probes[1].exact_match &&
                !absent.probes[1].all_bytes_valid,
            "HBM probe did not distinguish absent zero bytes from exact bytes");

    access.DebugSeed(128, {9, 8, 7, 6});
    runtime.DebugSeed(16, 0, {0, 0, 0, 0});
    const io::Result exact = io::VerifyAfterSimulation(applied);
    Require(exact.Passed() && exact.probes[1].all_bytes_valid,
            "ProgramIo rejected exact present HBM output bytes");
    runtime.DebugSeed(16, 0, {0, 0, 0, 1});
    const io::Result mismatch = io::VerifyAfterSimulation(applied);
    Require(!mismatch.Passed() &&
                mismatch.probes[1].all_bytes_valid &&
                !mismatch.probes[1].exact_match,
            "HBM probe did not distinguish present mismatch from absence");
    Require(SameStats(stats_before, backend->Stats()),
            "ProgramIo HBM debug operations changed timing statistics");

    const sram::DebugSnapshot missing_before = access.DebugPeek(256, 4);
    io::ResolvedContract missing_runtime = resolved;
    missing_runtime.initializations = {
        ResolvedInput(0, 256, {4, 4, 4, 4}),
        ResolvedHbmInput(32, {3, 3, 3, 3}),
    };
    io::Bindings missing_bindings;
    missing_bindings.sram_by_runtime_core.emplace(0, &access);
    ExpectFailure(
        [&] {
            (void)io::ApplyBeforeSimulation(
                missing_runtime, missing_bindings);
        },
        "missing HBMRuntime preflight");
    const sram::DebugSnapshot missing_after = access.DebugPeek(256, 4);
    Require(missing_after.payload == missing_before.payload &&
                missing_after.valid == missing_before.valid,
            "missing HBMRuntime preflight mutated earlier SRAM");

    HBMRuntime failing_runtime;
    io::HBMRuntimeSelfTestPeer::Add(
        &failing_runtime, 0, 0,
        std::make_unique<FailingAddressBackend>(hbm_config, 64));
    const sram::DebugSnapshot rollback_sram_before =
        access.DebugPeek(300, 4);
    const HBMRuntimeDebugSnapshot rollback_hbm_before =
        failing_runtime.DebugPeek(0, 0, 4);
    io::ResolvedContract rollback = resolved;
    rollback.initializations = {
        ResolvedInput(0, 300, {7, 7, 7, 7}),
        ResolvedHbmInput(0, {8, 8, 8, 8}),
        ResolvedHbmInput(64, {9, 9, 9, 9}),
    };
    rollback.output_probes = {
        ResolvedHbmProbe(128, {1, 1, 1, 1}),
    };
    io::Bindings rollback_bindings;
    rollback_bindings.sram_by_runtime_core.emplace(0, &access);
    rollback_bindings.hbm_runtime = &failing_runtime;
    ExpectFailure(
        [&] {
            (void)io::ApplyBeforeSimulation(
                rollback, rollback_bindings);
        },
        "mixed SRAM/HBM rollback");
    const sram::DebugSnapshot rollback_sram_after =
        access.DebugPeek(300, 4);
    const HBMRuntimeDebugSnapshot rollback_hbm_after =
        failing_runtime.DebugPeek(0, 0, 4);
    Require(rollback_sram_after.payload == rollback_sram_before.payload &&
                rollback_sram_after.valid == rollback_sram_before.valid &&
                rollback_hbm_after.payload == rollback_hbm_before.payload &&
                rollback_hbm_after.chunks[0].backend.present ==
                    rollback_hbm_before.chunks[0].backend.present,
            "mixed ProgramIo rollback did not restore SRAM/HBM presence");

    HBMRuntime rejecting_runtime;
    io::HBMRuntimeSelfTestPeer::Add(
        &rejecting_runtime, 0, 0,
        std::make_unique<RejectingBackend>());
    io::Bindings rejecting_bindings = bindings;
    rejecting_bindings.hbm_runtime = &rejecting_runtime;
    ExpectFailure(
        [&] {
            (void)io::ApplyBeforeSimulation(
                missing_runtime, rejecting_bindings);
        },
        "non-behavioral HBM debug access");
}
int RunSelftest() {
    TestStrictParser();
    TestTaggedTargets();
    TestInt32SramDType();
    TestSeedAndProbe();
    TestHbmSeedProbeAndRollback();
    io::RunExactFourStreamUnfusedTerminalReuseSelfTest();
    io::RunExactMoeCalibrationDynamicRootByteClosureSelfTest();
    std::cout << "ProgramIo C++ selftest: 7/7 PASS\n";
    return 0;
}

} // namespace

int sc_main(int argc, char **argv) {
    try {
        if (argc == 1) return RunSelftest();
        if (argc == 4 && std::string(argv[1]) == "--finalize") {
            const std::string manifest = ReadText(argv[2]);
            WriteBytes(argv[3],
                       frontend::ProgramArtifactFinalizer{}.FinalizeEncoded(
                           manifest));
            return 0;
        }
        if (argc == 5 && std::string(argv[1]) == "--resolve") {
            const std::string manifest = ReadText(argv[2]);
            const std::vector<uint8_t> artifact = ReadBytes(argv[3]);
            const std::string sidecar = ReadText(argv[4]);
            const io::ResolvedContract resolved =
                io::ParseAndResolve(sidecar, manifest, artifact);
            std::cout << "ProgramIo resolved id=" << resolved.id
                      << " initializations=" << resolved.initializations.size()
                      << " probes=" << resolved.output_probes.size() << "\n";
            return 0;
        }
        std::cerr << "usage: " << argv[0]
                  << " [--finalize MANIFEST ARTIFACT | "
                     "--resolve MANIFEST ARTIFACT SIDECAR]\n";
        return 2;
    } catch (const std::exception &error) {
        std::cerr << "ProgramIo C++ selftest failed: " << error.what() << "\n";
        return 1;
    }
}
