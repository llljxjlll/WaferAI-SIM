#pragma once

#include "memory/hbm_runtime.h"
#include "memory/sram/sram_access_unit.h"
#include "nlohmann/json.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace p5_probe {

class Error : public std::invalid_argument {
public:
    explicit Error(const std::string &message)
        : std::invalid_argument(message) {}
};

enum class SourceSpace {
    kSram,
    kHbm,
};

struct AffinePattern {
    uint8_t seed = 0;
    uint8_t multiplier = 0;
    uint8_t quarter_step = 1;
};

struct SourceSpec {
    SourceSpace space = SourceSpace::kSram;
    uint32_t core = 0;
    std::string region;
    uint64_t offset_bytes = 0;
    uint64_t absolute_address_bytes = 0;
    uint64_t length_bytes = 0;
    AffinePattern pattern;
};

struct DestinationSpec {
    uint32_t core = 0;
    std::string region;
    uint64_t region_size_bytes = 0;
    uint64_t payload_offset_bytes = 0;
    uint64_t payload_length_bytes = 0;
    uint8_t prefill_byte = 0;
    bool verify_all_bytes_outside_payload = false;
};

struct Spec {
    uint32_t version = 1;
    std::string scenario;
    SourceSpec source;
    DestinationSpec destination;
    uint32_t expected_payload_checksum = 0;
};

struct Bindings {
    std::map<uint32_t, sram::AccessUnit *> sram_by_core;
    HBMRuntime *hbm_runtime = nullptr;
    std::function<int(uint32_t)> current_die_for_core;
};

struct Applied {
    Spec spec;
    std::vector<uint8_t> expected_payload;
    uint32_t expected_payload_checksum = 0;
    bool source_initialized = false;
    sram::AccessUnit *destination_access = nullptr;
    uint64_t destination_region_address = 0;
};

struct Result {
    std::string scenario;
    bool source_initialized = false;
    uint64_t payload_bytes = 0;
    uint32_t expected_checksum = 0;
    uint32_t destination_checksum = 0;
    bool payload_match = false;
    bool sentinels_intact = false;
};

Spec Parse(const nlohmann::json &json);
Spec Load(const std::filesystem::path &path);
std::vector<uint8_t> BuildPayload(const Spec &spec);
uint32_t Crc32c(const std::vector<uint8_t> &payload) noexcept;

Applied ApplyBeforeSimulation(const Spec &spec,
                              const Bindings &bindings);
Result VerifyAfterSimulation(const Applied &applied);

} // namespace p5_probe

namespace p6_probe {

inline constexpr std::size_t kMaxEntries = 4096;
inline constexpr uint64_t kMaxSnapshotBytes = uint64_t{64} << 20;

class Error : public std::invalid_argument {
public:
    explicit Error(const std::string &message)
        : std::invalid_argument(message) {}
};

struct PatternSpec {
    std::string kind;
    uint8_t seed = 0;
    uint8_t multiplier = 0;
    uint8_t quarter_step = 1;
    bool reduction_boundary_patch = false;
};

struct InitializationSpec {
    uint32_t core = 0;
    std::string region;
    uint64_t region_size_bytes = 0;
    uint64_t offset_bytes = 0;
    uint64_t length_bytes = 0;
    uint8_t prefill_byte = 0;
    PatternSpec pattern;
    std::vector<uint8_t> expected_bytes;
    uint32_t checksum = 0;
    bool has_expected_after = false;
    std::vector<uint8_t> expected_after_bytes;
    uint32_t expected_after_checksum = 0;
};

struct VerificationSpec {
    uint32_t core = 0;
    std::string region;
    uint64_t region_size_bytes = 0;
    uint64_t payload_offset_bytes = 0;
    uint64_t payload_length_bytes = 0;
    std::vector<uint8_t> expected_bytes;
    uint32_t expected_checksum = 0;
    uint8_t prefill_byte = 0;
    bool verify_all_bytes_outside_payload = false;
};

struct Spec {
    uint32_t version = 1;
    std::string format;
    std::string scenario;
    std::vector<InitializationSpec> initializations;
    std::vector<VerificationSpec> verifications;
};

struct AppliedInitialization {
    InitializationSpec spec;
    sram::AccessUnit *access = nullptr;
    uint64_t region_address = 0;
};

struct AppliedVerification {
    VerificationSpec spec;
    sram::AccessUnit *access = nullptr;
    uint64_t region_address = 0;
};

struct Applied {
    Spec spec;
    std::vector<AppliedInitialization> initializations;
    std::vector<AppliedVerification> verifications;
};

struct SourceResult {
    uint32_t core = 0;
    std::string region;
    uint64_t payload_bytes = 0;
    uint32_t expected_checksum = 0;
    uint32_t checksum = 0;
    bool source_initialized = false;
    bool payload_match = false;
    bool sentinels_intact = false;
};

struct VerificationResult {
    uint32_t core = 0;
    std::string region;
    uint64_t payload_bytes = 0;
    uint32_t expected_checksum = 0;
    uint32_t checksum = 0;
    bool payload_match = false;
    bool sentinels_intact = false;
};

struct Result {
    std::string scenario;
    std::vector<SourceResult> sources;
    std::vector<VerificationResult> verifications;

    bool Passed() const noexcept;
};

using BeforeSeedHook = std::function<void(std::size_t)>;

Spec Parse(const nlohmann::json &json);
Spec Load(const std::filesystem::path &path);

Applied ApplyBeforeSimulation(
    const Spec &spec, const p5_probe::Bindings &bindings,
    const BeforeSeedHook &before_seed = {});
Result VerifyAfterSimulation(const Applied &applied);

} // namespace p6_probe
