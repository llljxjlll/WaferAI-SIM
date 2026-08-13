#include "isa/p5_memory_probe_selftest.h"

#include "isa/p5_memory_probe.h"
#include "defs/spec.h"
#include "die/port.h"
#include "memory/behavioral_hbm_backend.h"
#include "memory/hbm_address_map.h"
#include "memory/sram/sram_region.h"
#include "memory/sram/sram_storage.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <unordered_map>

namespace p5_probe {

struct HBMRuntimeSelfTestPeer {
    static void Add(HBMRuntime *runtime, int stack_id, int channel_id,
                    std::unique_ptr<HBMBackend> backend) {
        HBMRuntimeInstance instance;
        instance.stack_id = stack_id;
        instance.channel_id = channel_id;
        instance.backend = std::move(backend);
        runtime->instances_.push_back(std::move(instance));
    }
};

} // namespace p5_probe

namespace {

template <typename Exception, typename Function>
bool Throws(Function &&function) {
    try {
        function();
    } catch (const Exception &) {
        return true;
    } catch (...) {
    }
    return false;
}

void Check(bool condition, const char *message, int *failures) {
    if (condition) return;
    ++*failures;
    std::cerr << "[P5 MEMORY PROBE] FAIL: " << message << "\n";
}

nlohmann::json ProbeJson() {
    return {
        {"version", 1},
        {"scenario", "selftest_sram_9"},
        {"source",
         {{"space", "SRAM"},
          {"core", 0},
          {"region", "input"},
          {"offset_bytes", 3},
          {"length_bytes", 9},
          {"pattern",
           {{"kind", "affine_u8_v1"},
            {"seed", 17},
            {"multiplier", 37},
            {"quarter_step", 1}}}}},
        {"destination",
         {{"core", 1},
          {"region", "comm"},
          {"region_size_bytes", 64},
          {"payload_offset_bytes", 5},
          {"payload_length_bytes", 9},
          {"prefill_byte", 0xa5},
          {"verify_all_bytes_outside_payload", true}}},
        {"expected_payload_checksum", UINT32_C(0x92211d7b)},
    };
}

sram::Config ProbeSramConfig() {
    return sram::ParseConfig(
        {{"sram_size", 128},
         {"sram",
          {{"capacity_bytes", 128},
           {"real_data_path", true},
           {"regions",
            {{{"name", "input"},
              {"base_bytes", 0},
              {"size_bytes", 64},
              {"allocator", "fixed"},
              {"access", {"dte"}}},
             {{"name", "comm"},
              {"base_bytes", 64},
              {"size_bytes", 64},
              {"allocator", "fixed"},
              {"access", {"noc_rx"}}}}}}}});
}

struct TopologyGuard {
    int die_count = DIE_COUNT;
    std::unordered_map<std::string, HBMProfile> profiles = g_hbm_profiles;
    std::vector<HBMStackConfig> stacks = g_hbm_stacks;
    std::vector<HBMChannelConfig> channels = g_hbm_channels;
    AddressPolicyConfig address_policy = g_address_policy;

    ~TopologyGuard() {
        DIE_COUNT = die_count;
        g_hbm_profiles = std::move(profiles);
        g_hbm_stacks = std::move(stacks);
        g_hbm_channels = std::move(channels);
        g_address_policy = std::move(address_policy);
    }
};

class FailingSeedBackend : public BehavioralHBMBackend {
public:
    explicit FailingSeedBackend(
        const BehavioralHBMBackendConfig &config)
        : BehavioralHBMBackend(config) {}

    void DebugSeed(
        uint64_t, const std::vector<uint8_t> &) override {
        throw std::runtime_error(
            "injected second-backend seed failure");
    }
};

class RejectingBackend : public HBMBackend {
public:
    void Submit(
        const std::shared_ptr<HBMBackendTransaction> &) override {}
    const HBMBackendStats &Stats() const override { return stats_; }

private:
    HBMBackendStats stats_;
};

} // namespace

void RunP6MemoryProbeChecks(int *failures);

int RunP5MemoryProbeSelfTest() {
    int failures = 0;
    const p5_probe::Spec spec = p5_probe::Parse(ProbeJson());
    const std::vector<uint8_t> payload = p5_probe::BuildPayload(spec);
    Check(payload.size() == 9 &&
              p5_probe::Crc32c(payload) == UINT32_C(0x92211d7b),
          "strict parser freezes affine_u8_v1 and CRC32C", &failures);

    nlohmann::json unknown = ProbeJson();
    unknown["source"]["unexpected"] = 1;
    Check(Throws<p5_probe::Error>(
              [&] { (void)p5_probe::Parse(unknown); }),
          "strict parser rejects unknown source fields", &failures);
    nlohmann::json bad_checksum = ProbeJson();
    bad_checksum["expected_payload_checksum"] = 1;
    Check(Throws<p5_probe::Error>(
              [&] { (void)p5_probe::Parse(bad_checksum); }),
          "strict parser rejects a non-canonical checksum", &failures);

    const sram::Config config = ProbeSramConfig();
    sram::RegionTable source_regions(config);
    sram::RegionTable destination_regions(config);
    sram::Storage source_storage(config.capacity_bytes);
    sram::Storage destination_storage(config.capacity_bytes);
    sram::AccessUnit source_access(
        "p5_probe_source_access", source_regions, source_storage);
    sram::AccessUnit destination_access(
        "p5_probe_destination_access", destination_regions,
        destination_storage);

    const sram::DebugSnapshot source_initial =
        source_access.DebugPeek(3, payload.size());
    p5_probe::Bindings missing_second;
    missing_second.sram_by_core.emplace(0, &source_access);
    Check(Throws<p5_probe::Error>([&] {
              (void)p5_probe::ApplyBeforeSimulation(
                  spec, missing_second);
          }),
          "missing second SRAM binding fails before commit", &failures);
    const sram::DebugSnapshot source_after_failure =
        source_access.DebugPeek(3, payload.size());
    Check(source_after_failure.payload == source_initial.payload &&
              source_after_failure.valid == source_initial.valid,
          "second-binding failure preserves first binding payload and validity",
          &failures);

    p5_probe::Bindings bindings;
    bindings.sram_by_core.emplace(0, &source_access);
    bindings.sram_by_core.emplace(1, &destination_access);
    const p5_probe::Applied applied =
        p5_probe::ApplyBeforeSimulation(spec, bindings);
    const sram::DebugSnapshot seeded_source =
        source_access.DebugPeek(3, payload.size());
    Check(seeded_source.payload == payload &&
              std::all_of(
                  seeded_source.valid.begin(), seeded_source.valid.end(),
                  [](uint8_t value) { return value == 1; }),
          "ApplyBeforeSimulation seeds non-zero source bytes", &failures);

    const sram::DebugSnapshot prefilled_destination =
        destination_access.DebugPeek(64, 64);
    bool prefill_matches = true;
    for (size_t index = 0;
         index < prefilled_destination.payload.size(); ++index) {
        const uint8_t expected =
            index >= 5 && index < 5 + payload.size()
                ? static_cast<uint8_t>(payload[index - 5] ^ UINT8_C(0xff))
                : UINT8_C(0xa5);
        if (prefilled_destination.payload[index] != expected ||
            prefilled_destination.valid[index] != 1) {
            prefill_matches = false;
            break;
        }
    }
    Check(prefill_matches,
          "ApplyBeforeSimulation pre-fills outside sentinels and per-byte "
          "payload anti-patterns",
          &failures);

    destination_access.DebugSeed(
        64 + 5,
        std::vector<uint8_t>(payload.begin(), payload.end() - 1));
    const p5_probe::Result partial =
        p5_probe::VerifyAfterSimulation(applied);
    Check(!partial.payload_match && partial.sentinels_intact,
          "a partial destination write cannot pass payload verification",
          &failures);

    destination_access.DebugSeed(64 + 5, payload);
    const p5_probe::Result success =
        p5_probe::VerifyAfterSimulation(applied);
    Check(success.source_initialized && success.payload_match &&
              success.sentinels_intact &&
              success.destination_checksum ==
                  spec.expected_payload_checksum,
          "VerifyAfterSimulation proves payload and outside sentinels",
          &failures);

    destination_access.DebugSeed(64, {0x5a});
    const p5_probe::Result corrupted =
        p5_probe::VerifyAfterSimulation(applied);
    Check(corrupted.payload_match && !corrupted.sentinels_intact,
          "sentinel corruption is independently observable", &failures);

    nlohmann::json collision_json = ProbeJson();
    collision_json["scenario"] = "selftest_sram_1_collision";
    collision_json["source"]["length_bytes"] = 1;
    collision_json["source"]["pattern"]["seed"] = 0xa5;
    collision_json["source"]["pattern"]["multiplier"] = 1;
    collision_json["destination"]["payload_length_bytes"] = 1;
    collision_json["expected_payload_checksum"] =
        UINT32_C(0xc5c7f2eb);
    const p5_probe::Spec collision_spec =
        p5_probe::Parse(collision_json);
    const std::vector<uint8_t> collision_payload =
        p5_probe::BuildPayload(collision_spec);
    const p5_probe::Applied collision_applied =
        p5_probe::ApplyBeforeSimulation(collision_spec, bindings);
    const p5_probe::Result collision_no_transfer =
        p5_probe::VerifyAfterSimulation(collision_applied);
    Check(collision_payload == std::vector<uint8_t>({0xa5}) &&
              !collision_no_transfer.payload_match &&
              collision_no_transfer.sentinels_intact,
          "length-one payload equal to outside sentinel cannot pass without "
          "a transfer",
          &failures);
    destination_access.DebugSeed(64 + 5, collision_payload);
    const p5_probe::Result collision_success =
        p5_probe::VerifyAfterSimulation(collision_applied);
    Check(collision_success.payload_match &&
              collision_success.sentinels_intact &&
              collision_success.destination_checksum ==
                  collision_spec.expected_payload_checksum,
          "length-one payload equal to outside sentinel passes after transfer",
          &failures);

    BehavioralHBMBackendConfig hbm_config;
    hbm_config.bandwidth_GBps = 1.0;
    BehavioralHBMBackend hbm(hbm_config);
    const HBMDebugSnapshot empty = hbm.DebugPeek(7, 3);
    hbm.DebugSeed(7, {1, 2, 3});
    Check(hbm.DebugPeek(7, 3).payload ==
              std::vector<uint8_t>({1, 2, 3}),
          "behavioral HBM debug seed uses production backing", &failures);
    hbm.DebugRestore(empty);
    const HBMDebugSnapshot restored = hbm.DebugPeek(7, 3);
    Check(restored.payload == std::vector<uint8_t>({0, 0, 0}) &&
              restored.present == std::vector<uint8_t>({0, 0, 0}),
          "behavioral HBM restore preserves absent backing entries",
          &failures);

    {
        TopologyGuard topology_guard;
        DIE_COUNT = 1;
        HBMProfile profile;
        profile.channels_per_stack = 2;
        profile.pseudo_channels_per_channel = 1;
        g_hbm_profiles = {{"p5_probe", profile}};
        HBMStackConfig stack;
        stack.stack_id = 0;
        stack.compute_die_id = 0;
        stack.profile = "p5_probe";
        stack.capacity_bytes = 64;
        g_hbm_stacks = {stack};
        g_hbm_channels = {{0, 0, 0}, {0, 1, 1}};
        g_address_policy = AddressPolicyConfig{};
        g_address_policy.active = true;
        g_address_policy.mode =
            AddressPolicyMode::kNumaLocalInterleave;
        g_address_policy.home_ranges = {{0, 0, 64}};
        g_address_policy.stack_interleave_bytes = 64;
        g_address_policy.channel_interleave_bytes = 2;
        g_address_policy.pseudo_channel_interleave_bytes = 2;

        HBMRuntime runtime;
        p5_probe::HBMRuntimeSelfTestPeer::Add(
            &runtime, 0, 0,
            std::make_unique<BehavioralHBMBackend>(hbm_config));
        p5_probe::HBMRuntimeSelfTestPeer::Add(
            &runtime, 0, 1,
            std::make_unique<FailingSeedBackend>(hbm_config));
        Check(Throws<std::runtime_error>([&] {
                  runtime.DebugSeed(0, 0, {9, 8, 7, 6});
              }),
              "second HBM binding failure is propagated", &failures);
        const HBMDebugSnapshot first_after_failure =
            runtime.Find(0, 0)->backend->DebugPeek(0, 2);
        Check(first_after_failure.payload ==
                  std::vector<uint8_t>({0, 0}) &&
                  first_after_failure.present ==
                      std::vector<uint8_t>({0, 0}),
              "second HBM binding failure rolls back first binding",
              &failures);
    }

    RejectingBackend rejecting;
    Check(Throws<std::logic_error>(
              [&] { (void)rejecting.DebugPeek(0, 1); }) &&
              Throws<std::logic_error>(
                  [&] { rejecting.DebugSeed(0, {1}); }),
          "non-behavioral HBM backend rejects debug access by default",
          &failures);

    RunP6MemoryProbeChecks(&failures);

    if (failures == 0) {
        std::cout << "[P5 MEMORY PROBE] PASS\n";
        std::cout << "[P6 MEMORY PROBE] PASS\n";
    }
    return failures;
}
nlohmann::json P6ProbeJson() {
    auto initialization = [](uint32_t core, const char *encoded,
                             const std::vector<uint8_t> &bytes) {
        return nlohmann::json{
            {"space", "SRAM"}, {"core", core}, {"region", "p6_input"},
            {"region_size_bytes", 64}, {"offset_bytes", 3},
            {"length_bytes", bytes.size()}, {"prefill_byte", 0xa5},
            {"pattern",
             {{"kind", "affine_u8_v1"}, {"seed", 17 + core},
              {"multiplier", 3 + core * 2}, {"quarter_step", 1},
              {"reduction_boundary_patch", false}}},
            {"bytes_base64", encoded},
            {"checksum", p5_probe::Crc32c(bytes)},
        };
    };
    auto verification = [](uint32_t core, const char *region,
                           const char *encoded,
                           const std::vector<uint8_t> &bytes) {
        return nlohmann::json{
            {"core", core}, {"region", region},
            {"region_size_bytes", 64}, {"payload_offset_bytes", 5},
            {"payload_length_bytes", bytes.size()},
            {"expected_bytes_base64", encoded},
            {"expected_checksum", p5_probe::Crc32c(bytes)},
            {"prefill_byte", 0xa5},
            {"verify_all_bytes_outside_payload", true},
        };
    };
    return {
        {"version", 1},
        {"format", "p6-p5-compatible-multi-region-v1"},
        {"scenario", "p6_probe_selftest"},
        {"initializations",
         {initialization(0, "AQIDBA==", {1, 2, 3, 4}),
          initialization(1, "BQYHCA==", {5, 6, 7, 8})}},
        {"verifications",
         {verification(0, "p6_staging", "CQoLDA==", {9, 10, 11, 12}),
          verification(1, "p6_result", "DQ4PEA==", {13, 14, 15, 16})}},
    };
}

sram::Config P6ProbeSramConfig() {
    return sram::ParseConfig(
        {{"sram_size", 192},
         {"sram",
          {{"capacity_bytes", 192}, {"real_data_path", true},
           {"regions",
            {{{"name", "p6_input"}, {"base_bytes", 0}, {"size_bytes", 64},
              {"allocator", "fixed"}, {"access", {"dte"}}},
             {{"name", "p6_staging"}, {"base_bytes", 64},
              {"size_bytes", 64}, {"allocator", "fixed"},
              {"access", {"noc_rx"}}},
             {{"name", "p6_result"}, {"base_bytes", 128},
              {"size_bytes", 64}, {"allocator", "fixed"},
              {"access", {"noc_rx"}}}}}}}});
}

void CheckP6Snapshot(const sram::DebugSnapshot &actual,
                     const sram::DebugSnapshot &expected,
                     const char *message, int *failures) {
    Check(actual.payload == expected.payload &&
              actual.valid == expected.valid,
          message, failures);
}

void RunP6MemoryProbeChecks(int *failures) {
    const nlohmann::json json = P6ProbeJson();
    const p6_probe::Spec spec = p6_probe::Parse(json);
    Check(spec.initializations.size() == 2 &&
              spec.verifications.size() == 2,
          "P6 parser accepts multi-core/multi-region schema", failures);

    nlohmann::json invalid = json;
    invalid["unexpected"] = 1;
    Check(Throws<p6_probe::Error>(
              [&] { (void)p6_probe::Parse(invalid); }),
          "P6 parser rejects unknown fields", failures);
    invalid = json;
    invalid["initializations"][0]["bytes_base64"] = "AQIDBA=";
    Check(Throws<p6_probe::Error>(
              [&] { (void)p6_probe::Parse(invalid); }),
          "P6 parser rejects non-canonical base64", failures);
    invalid = json;
    invalid["verifications"][0]["expected_checksum"] = 1;
    Check(Throws<p6_probe::Error>(
              [&] { (void)p6_probe::Parse(invalid); }),
          "P6 parser rejects checksum mismatch", failures);
    invalid = json;
    invalid["verifications"][0]["region"] = "p6_input";
    Check(Throws<p6_probe::Error>(
              [&] { (void)p6_probe::Parse(invalid); }),
          "P6 parser rejects duplicate core/region ownership", failures);
    invalid = json;
    invalid["initializations"][0]["offset_bytes"] = 62;
    Check(Throws<p6_probe::Error>(
              [&] { (void)p6_probe::Parse(invalid); }),
          "P6 parser rejects payload outside its region", failures);
    invalid = json;
    invalid["verifications"][0]["region_size_bytes"] =
        p6_probe::kMaxSnapshotBytes + 1;
    Check(Throws<p6_probe::Error>(
              [&] { (void)p6_probe::Parse(invalid); }),
          "P6 parser rejects bounded-snapshot overflow", failures);

    const std::vector<uint8_t> expected_after = {9, 2, 3, 4};
    nlohmann::json transformed_json = json;
    transformed_json["initializations"][0]
                    ["expected_after_bytes_base64"] = "CQIDBA==";
    transformed_json["initializations"][0]
                    ["expected_after_checksum"] =
        p5_probe::Crc32c(expected_after);
    const p6_probe::Spec transformed_spec =
        p6_probe::Parse(transformed_json);
    Check(transformed_spec.initializations[0].has_expected_after &&
              transformed_spec.initializations[0].expected_after_bytes ==
                  expected_after &&
              transformed_spec.initializations[0].expected_after_checksum ==
                  p5_probe::Crc32c(expected_after) &&
              !transformed_spec.initializations[1].has_expected_after,
          "P6 parser accepts a canonical expected-after pair without changing old entries",
          failures);
    invalid = transformed_json;
    invalid["initializations"][0].erase(
        "expected_after_checksum");
    Check(Throws<p6_probe::Error>(
              [&] { (void)p6_probe::Parse(invalid); }),
          "P6 parser rejects an unpaired expected-after payload", failures);
    invalid = transformed_json;
    invalid["initializations"][0].erase(
        "expected_after_bytes_base64");
    Check(Throws<p6_probe::Error>(
              [&] { (void)p6_probe::Parse(invalid); }),
          "P6 parser rejects an unpaired expected-after checksum", failures);
    invalid = transformed_json;
    invalid["initializations"][0]["expected_after_bytes_base64"] =
        "CQIDBA=";
    Check(Throws<p6_probe::Error>(
              [&] { (void)p6_probe::Parse(invalid); }),
          "P6 parser rejects non-canonical expected-after base64", failures);
    invalid = transformed_json;
    invalid["initializations"][0]["expected_after_checksum"] = 1;
    Check(Throws<p6_probe::Error>(
              [&] { (void)p6_probe::Parse(invalid); }),
          "P6 parser rejects an expected-after checksum mismatch", failures);
    invalid = transformed_json;
    invalid["initializations"][0]["expected_after_bytes_base64"] =
        "CQID";
    invalid["initializations"][0]["expected_after_checksum"] =
        p5_probe::Crc32c({9, 2, 3});
    Check(Throws<p6_probe::Error>(
              [&] { (void)p6_probe::Parse(invalid); }),
          "P6 parser bounds expected-after bytes to the declared length", failures);
    invalid = transformed_json;
    invalid["initializations"][0]["expected_after_crc"] = 1;
    Check(Throws<p6_probe::Error>(
              [&] { (void)p6_probe::Parse(invalid); }),
          "P6 parser keeps exact-key validation with expected-after", failures);

    const sram::Config config = P6ProbeSramConfig();
    sram::RegionTable regions0(config);
    sram::RegionTable regions1(config);
    sram::Storage storage0(config.capacity_bytes);
    sram::Storage storage1(config.capacity_bytes);
    sram::AccessUnit access0("p6_probe_access_0", regions0, storage0);
    sram::AccessUnit access1("p6_probe_access_1", regions1, storage1);
    for (uint64_t base : {UINT64_C(0), UINT64_C(64), UINT64_C(128)}) {
        access0.DebugSeed(base, std::vector<uint8_t>(64, 0x31));
        access1.DebugSeed(base, std::vector<uint8_t>(64, 0x72));
    }
    const sram::DebugSnapshot before0_input = access0.DebugPeek(0, 64);
    const sram::DebugSnapshot before0_staging = access0.DebugPeek(64, 64);
    const sram::DebugSnapshot before0_result = access0.DebugPeek(128, 64);
    const sram::DebugSnapshot before1_input = access1.DebugPeek(0, 64);
    const sram::DebugSnapshot before1_staging = access1.DebugPeek(64, 64);
    const sram::DebugSnapshot before1_result = access1.DebugPeek(128, 64);

    p5_probe::Bindings missing;
    missing.sram_by_core.emplace(0, &access0);
    Check(Throws<p6_probe::Error>([&] {
              (void)p6_probe::ApplyBeforeSimulation(spec, missing);
          }),
          "P6 resolves all bindings before any commit", failures);
    CheckP6Snapshot(access0.DebugPeek(0, 64), before0_input,
                    "P6 resolve failure leaves earlier core unchanged",
                    failures);

    p5_probe::Bindings bindings;
    bindings.sram_by_core.emplace(0, &access0);
    bindings.sram_by_core.emplace(1, &access1);
    Check(Throws<std::runtime_error>([&] {
              (void)p6_probe::ApplyBeforeSimulation(
                  spec, bindings, [](size_t index) {
                      if (index == 2)
                          throw std::runtime_error(
                              "injected P6 seed failure");
                  });
          }),
          "P6 propagates mid-transaction seed failure", failures);
    CheckP6Snapshot(access0.DebugPeek(0, 64), before0_input,
                    "P6 rollback restores core-0 input", failures);
    CheckP6Snapshot(access0.DebugPeek(64, 64), before0_staging,
                    "P6 rollback restores core-0 staging", failures);
    CheckP6Snapshot(access0.DebugPeek(128, 64), before0_result,
                    "P6 rollback restores core-0 result", failures);
    CheckP6Snapshot(access1.DebugPeek(0, 64), before1_input,
                    "P6 rollback restores core-1 input", failures);
    CheckP6Snapshot(access1.DebugPeek(64, 64), before1_staging,
                    "P6 rollback restores core-1 staging", failures);
    CheckP6Snapshot(access1.DebugPeek(128, 64), before1_result,
                    "P6 rollback restores core-1 result", failures);

    Check(Throws<std::runtime_error>([&] {
              (void)p6_probe::ApplyBeforeSimulation(
                  transformed_spec, bindings, [](size_t index) {
                      if (index == 2)
                          throw std::runtime_error(
                              "injected expected-after seed failure");
                  });
          }),
          "P6 expected-after metadata preserves transactional failure",
          failures);
    CheckP6Snapshot(access0.DebugPeek(0, 64), before0_input,
                    "P6 expected-after rollback restores initialized bytes",
                    failures);
    CheckP6Snapshot(access0.DebugPeek(64, 64), before0_staging,
                    "P6 expected-after rollback restores later regions",
                    failures);

    const p6_probe::Applied transformed_applied =
        p6_probe::ApplyBeforeSimulation(transformed_spec, bindings);
    Check(access0.DebugPeek(3, 4).payload ==
              std::vector<uint8_t>({1, 2, 3, 4}),
          "P6 Apply seeds bytes_base64 rather than expected-after bytes",
          failures);
    const p6_probe::Result before_transform =
        p6_probe::VerifyAfterSimulation(transformed_applied);
    Check(!before_transform.sources[0].payload_match &&
              before_transform.sources[0].expected_checksum ==
                  p5_probe::Crc32c(expected_after) &&
              before_transform.sources[0].checksum ==
                  p5_probe::Crc32c({1, 2, 3, 4}),
          "P6 unchanged initialization fails an explicit expected-after oracle",
          failures);
    access0.DebugSeed(3, expected_after);
    const p6_probe::Result after_transform =
        p6_probe::VerifyAfterSimulation(transformed_applied);
    Check(after_transform.sources[0].payload_match &&
              after_transform.sources[0].sentinels_intact &&
              after_transform.sources[0].checksum ==
                  p5_probe::Crc32c(expected_after),
          "P6 exact expected-after transformation passes source verification",
          failures);

    const p6_probe::Applied applied =
        p6_probe::ApplyBeforeSimulation(spec, bindings);
    const p6_probe::Result untouched =
        p6_probe::VerifyAfterSimulation(applied);
    Check(untouched.sources.size() == 2 &&
              untouched.verifications.size() == 2 &&
              untouched.sources[0].payload_match &&
              untouched.sources[1].payload_match &&
              !untouched.verifications[0].payload_match &&
              !untouched.verifications[1].payload_match &&
              !untouched.Passed(),
          "P6 verifies sources and rejects output anti-patterns", failures);

    access0.DebugSeed(64 + 5, {9, 10, 11, 12});
    access1.DebugSeed(128 + 5, {13, 14, 15, 16});
    const p6_probe::Result success =
        p6_probe::VerifyAfterSimulation(applied);
    Check(success.Passed(), "P6 verifies all payloads and sentinels",
          failures);

    access0.DebugSeed(3, {0xff});
    const p6_probe::Result bad_source =
        p6_probe::VerifyAfterSimulation(applied);
    Check(!bad_source.Passed() &&
              !bad_source.sources[0].payload_match &&
              bad_source.verifications[0].payload_match,
          "P6 independently observes source corruption", failures);
    access0.DebugSeed(3, {1});
    access1.DebugSeed(128, {0x5a});
    const p6_probe::Result bad_sentinel =
        p6_probe::VerifyAfterSimulation(applied);
    Check(!bad_sentinel.Passed() &&
              bad_sentinel.verifications[1].payload_match &&
              !bad_sentinel.verifications[1].sentinels_intact,
          "P6 independently observes result sentinel corruption",
          failures);
}
