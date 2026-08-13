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

namespace p8_double_buffer_probe {

class Error : public std::invalid_argument {
public:
    explicit Error(const std::string &message)
        : std::invalid_argument(message) {}
};

enum class Space : uint8_t { kSram = 0, kHbm = 1 };

struct Pattern {
    uint8_t seed = 0;
    uint8_t multiplier = 0;
    uint8_t quarter_step = 1;
};

struct Range {
    Space space = Space::kSram;
    uint32_t core = 0;
    std::string region;
    uint64_t region_size_bytes = 0;
    uint64_t absolute_address_bytes = 0;
    uint64_t payload_offset_bytes = 0;
    uint64_t payload_length_bytes = 0;
    uint8_t prefill_byte = 0;
    bool verify_all_bytes_outside_payload = false;
    Pattern pattern;
    uint32_t expected_checksum = 0;
};

struct Spec {
    uint32_t version = 1;
    std::string scenario;
    std::vector<Range> initializations;
    std::vector<Range> verifications;
};

struct Bindings {
    std::map<uint32_t, sram::AccessUnit *> sram_by_core;
    HBMRuntime *hbm_runtime = nullptr;
    std::function<int(uint32_t)> current_die_for_core;
};

struct AppliedRange {
    Range spec;
    std::vector<uint8_t> expected_payload;
    sram::AccessUnit *sram_access = nullptr;
    uint64_t sram_region_address = 0;
};

struct Applied {
    Spec spec;
    Bindings bindings;
    std::vector<AppliedRange> verifications;
};

struct RangeResult {
    Space space = Space::kSram;
    uint32_t core = 0;
    std::string region;
    uint64_t absolute_address_bytes = 0;
    uint64_t payload_bytes = 0;
    uint32_t expected_checksum = 0;
    uint32_t checksum = 0;
    bool payload_match = false;
    bool sentinels_intact = false;
};

struct Result {
    std::string scenario;
    std::vector<RangeResult> ranges;

    bool Passed() const noexcept;
};

Spec Parse(const nlohmann::json &json);
Spec Load(const std::filesystem::path &path);
std::vector<uint8_t> BuildPayload(const Range &range);
uint32_t Crc32c(const std::vector<uint8_t> &payload) noexcept;
Applied ApplyBeforeSimulation(const Spec &spec, const Bindings &bindings);
Result VerifyAfterSimulation(const Applied &applied);

} // namespace p8_double_buffer_probe
