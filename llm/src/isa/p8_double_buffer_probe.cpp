#include "isa/p8_double_buffer_probe.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <optional>
#include <set>

namespace p8_double_buffer_probe {
namespace {

using Json = nlohmann::json;
constexpr uint64_t kMaxRangeBytes = uint64_t{8} << 20;

void Require(bool condition, const std::string &message) {
    if (!condition) throw Error(message);
}

void RequireBoundary(const char *operation) {
    if (sc_core::sc_is_running())
        throw Error(std::string(operation) +
                    " is forbidden while simulation is running");
}

void RequireKeys(const Json &node, const std::set<std::string> &keys,
                 const char *field) {
    if (!node.is_object())
        throw Error(std::string(field) + " must be an object");
    std::set<std::string> actual;
    for (auto it = node.begin(); it != node.end(); ++it)
        actual.insert(it.key());
    if (actual != keys)
        throw Error(std::string(field) +
                    " fields do not match the P8 v1 canonical schema");
}

uint64_t Unsigned(const Json &node, const char *field) {
    if (!node.is_number_unsigned() && !node.is_number_integer())
        throw Error(std::string(field) + " must be an unsigned integer");
    if (!node.is_number_unsigned() && node.get<int64_t>() < 0)
        throw Error(std::string(field) + " must be an unsigned integer");
    try {
        return node.get<uint64_t>();
    } catch (const Json::exception &) {
        throw Error(std::string(field) + " is outside uint64");
    }
}

uint32_t U32(const Json &node, const char *field) {
    const uint64_t value = Unsigned(node, field);
    if (value > UINT32_MAX)
        throw Error(std::string(field) + " is outside uint32");
    return static_cast<uint32_t>(value);
}

uint8_t U8(const Json &node, const char *field) {
    const uint64_t value = Unsigned(node, field);
    if (value > UINT8_MAX)
        throw Error(std::string(field) + " is outside uint8");
    return static_cast<uint8_t>(value);
}

std::string Token(const Json &node, const char *field) {
    if (!node.is_string())
        throw Error(std::string(field) + " must be a string");
    const std::string value = node.get<std::string>();
    if (value.empty() || value.size() > 128)
        throw Error(std::string(field) + " must contain 1..128 bytes");
    for (const unsigned char ch : value) {
        if (!std::isalnum(ch) && ch != '_' && ch != '-' && ch != '.')
            throw Error(std::string(field) +
                        " contains a non-canonical character");
    }
    return value;
}

Space ParseSpace(const Json &node, const char *field) {
    const std::string value = Token(node, field);
    if (value == "SRAM") return Space::kSram;
    if (value == "HBM") return Space::kHbm;
    throw Error(std::string(field) + " must be SRAM or HBM");
}

uint64_t CheckedEnd(uint64_t begin, uint64_t size,
                    const char *field) {
    if (size == 0 || size > kMaxRangeBytes)
        throw Error(std::string(field) +
                    " size must be in [1,8 MiB]");
    if (begin > std::numeric_limits<uint64_t>::max() - size)
        throw Error(std::string(field) + " range overflows");
    return begin + size;
}

Range ParseRange(const Json &node, const char *field) {
    RequireKeys(node,
                {"absolute_address_bytes", "core", "expected_checksum",
                 "pattern", "payload_length_bytes", "payload_offset_bytes",
                 "prefill_byte", "region", "region_size_bytes", "space",
                 "verify_all_bytes_outside_payload"},
                field);
    RequireKeys(node.at("pattern"),
                {"kind", "multiplier", "quarter_step", "seed"},
                "range.pattern");
    Require(Token(node.at("pattern").at("kind"), "pattern.kind") ==
                "affine_u8_v1",
            "pattern.kind must be affine_u8_v1");

    Range result;
    result.space = ParseSpace(node.at("space"), "range.space");
    result.core = U32(node.at("core"), "range.core");
    if (!node.at("region").is_string())
        throw Error("range.region must be a string");
    result.region = node.at("region").get<std::string>();
    result.region_size_bytes =
        Unsigned(node.at("region_size_bytes"), "range.region_size_bytes");
    result.absolute_address_bytes = Unsigned(
        node.at("absolute_address_bytes"),
        "range.absolute_address_bytes");
    result.payload_offset_bytes = Unsigned(
        node.at("payload_offset_bytes"), "range.payload_offset_bytes");
    result.payload_length_bytes = Unsigned(
        node.at("payload_length_bytes"), "range.payload_length_bytes");
    result.prefill_byte = U8(node.at("prefill_byte"),
                             "range.prefill_byte");
    if (!node.at("verify_all_bytes_outside_payload").is_boolean())
        throw Error("range.verify_all_bytes_outside_payload must be boolean");
    result.verify_all_bytes_outside_payload =
        node.at("verify_all_bytes_outside_payload").get<bool>();
    result.pattern.seed =
        U8(node.at("pattern").at("seed"), "pattern.seed");
    result.pattern.multiplier = U8(
        node.at("pattern").at("multiplier"), "pattern.multiplier");
    result.pattern.quarter_step = U8(
        node.at("pattern").at("quarter_step"), "pattern.quarter_step");
    result.expected_checksum = U32(
        node.at("expected_checksum"), "range.expected_checksum");

    CheckedEnd(0, result.region_size_bytes, "range container");
    const uint64_t payload_end = CheckedEnd(
        result.payload_offset_bytes, result.payload_length_bytes,
        "range payload");
    Require(payload_end <= result.region_size_bytes,
            "range payload exceeds its container");
    if (result.space == Space::kSram) {
        Require(!result.region.empty() && result.region.size() <= 64,
                "SRAM range requires a 1..64 byte region name");
        Require(result.absolute_address_bytes == 0,
                "SRAM range must have absolute_address_bytes=0");
    } else {
        Require(result.region.empty(), "HBM range.region must be empty");
        (void)CheckedEnd(result.absolute_address_bytes,
                         result.region_size_bytes, "HBM container");
    }
    Require(result.pattern.multiplier != 0 ||
                result.pattern.quarter_step != 0,
            "range pattern must not be constant");
    const std::vector<uint8_t> payload = BuildPayload(result);
    Require(std::any_of(payload.begin(), payload.end(),
                        [](uint8_t value) { return value != 0; }),
            "range pattern must contain non-zero bytes");
    Require(Crc32c(payload) == result.expected_checksum,
            "range expected_checksum disagrees with its pattern");
    return result;
}

void ValidateSpec(const Spec &spec) {
    Require(spec.version == 1, "P8 probe version must be 1");
    Require(!spec.scenario.empty(), "P8 probe scenario must not be empty");
    Require(!spec.initializations.empty(),
            "P8 probe needs at least one initialization");
    Require(!spec.verifications.empty(),
            "P8 probe needs at least one verification");
}

sram::AccessUnit *AccessFor(const Bindings &bindings, uint32_t core) {
    const auto it = bindings.sram_by_core.find(core);
    if (it == bindings.sram_by_core.end() || it->second == nullptr)
        throw Error("P8 probe has no SRAM binding for core " +
                    std::to_string(core));
    return it->second;
}

uint64_t RegionBase(sram::AccessUnit &access, const Range &range) {
    const sram::RegionConfig &region = access.regions().Region(range.region);
    if (region.size_bytes != range.region_size_bytes)
        throw Error("P8 probe sidecar/hardware SRAM region size mismatch");
    return region.base_bytes;
}

int CurrentDie(const Bindings &bindings, uint32_t core) {
    if (!bindings.current_die_for_core)
        throw Error("P8 HBM probe requires a core-to-die binding");
    return bindings.current_die_for_core(core);
}

std::vector<uint8_t> Container(const Range &range) {
    std::vector<uint8_t> result(
        static_cast<size_t>(range.region_size_bytes), range.prefill_byte);
    const std::vector<uint8_t> payload = BuildPayload(range);
    std::copy(payload.begin(), payload.end(),
              result.begin() +
                  static_cast<size_t>(range.payload_offset_bytes));
    return result;
}

AppliedRange ResolveVerification(const Range &range,
                                 const Bindings &bindings) {
    AppliedRange result;
    result.spec = range;
    result.expected_payload = BuildPayload(range);
    if (range.space == Space::kSram) {
        result.sram_access = AccessFor(bindings, range.core);
        result.sram_region_address = RegionBase(*result.sram_access, range);
    } else if (bindings.hbm_runtime == nullptr) {
        throw Error("P8 HBM probe requires an HBMRuntime binding");
    }
    return result;
}

} // namespace

std::vector<uint8_t> BuildPayload(const Range &range) {
    std::vector<uint8_t> result(
        static_cast<size_t>(range.payload_length_bytes));
    for (size_t index = 0; index < result.size(); ++index) {
        result[index] = static_cast<uint8_t>(
            static_cast<uint64_t>(range.pattern.seed) +
            static_cast<uint64_t>(range.pattern.multiplier) * index +
            static_cast<uint64_t>(range.pattern.quarter_step) *
                (index >> 2));
    }
    return result;
}

uint32_t Crc32c(const std::vector<uint8_t> &payload) noexcept {
    uint32_t checksum = UINT32_C(0xffffffff);
    for (const uint8_t value : payload) {
        checksum ^= value;
        for (int bit = 0; bit < 8; ++bit)
            checksum = (checksum >> 1) ^
                ((checksum & 1) ? UINT32_C(0x82f63b78) : 0U);
    }
    return ~checksum;
}

bool Result::Passed() const noexcept {
    return !ranges.empty() &&
        std::all_of(ranges.begin(), ranges.end(),
                    [](const RangeResult &range) {
                        return range.payload_match &&
                               range.sentinels_intact &&
                               range.checksum == range.expected_checksum;
                    });
}

Spec Parse(const Json &json) {
    try {
        RequireKeys(json,
                    {"format", "initializations", "scenario",
                     "verifications", "version"},
                    "P8 probe");
        Require(Token(json.at("format"), "format") ==
                    "p8_double_buffer_probe_v1",
                "P8 probe format is invalid");
        Spec result;
        result.version = U32(json.at("version"), "version");
        result.scenario = Token(json.at("scenario"), "scenario");
        if (!json.at("initializations").is_array() ||
            !json.at("verifications").is_array())
            throw Error("P8 probe range collections must be arrays");
        for (const Json &entry : json.at("initializations"))
            result.initializations.push_back(
                ParseRange(entry, "initialization"));
        for (const Json &entry : json.at("verifications"))
            result.verifications.push_back(
                ParseRange(entry, "verification"));
        ValidateSpec(result);
        return result;
    } catch (const Error &) {
        throw;
    } catch (const Json::exception &error) {
        throw Error(std::string("invalid P8 probe JSON: ") + error.what());
    }
}

Spec Load(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input)
        throw Error("cannot open P8 probe sidecar: " + path.string());
    try {
        Json json;
        input >> json;
        input >> std::ws;
        if (input.peek() != std::char_traits<char>::eof())
            throw Error("P8 probe sidecar contains trailing tokens");
        if (input.bad())
            throw Error("failed while reading P8 probe sidecar");
        return Parse(json);
    } catch (const Error &) {
        throw;
    } catch (const Json::exception &error) {
        throw Error(std::string("invalid P8 probe JSON: ") + error.what());
    }
}

Applied ApplyBeforeSimulation(const Spec &spec,
                              const Bindings &bindings) {
    RequireBoundary("P8 ApplyBeforeSimulation");
    ValidateSpec(spec);
    struct Rollback {
        Space space = Space::kSram;
        sram::AccessUnit *access = nullptr;
        uint64_t address = 0;
        int current_die = -1;
        std::optional<sram::DebugSnapshot> sram;
        std::optional<HBMRuntimeDebugSnapshot> hbm;
    };
    std::vector<Rollback> rollbacks;
    rollbacks.reserve(spec.initializations.size());
    try {
        for (const Range &range : spec.initializations) {
            const std::vector<uint8_t> bytes = Container(range);
            Rollback rollback;
            rollback.space = range.space;
            if (range.space == Space::kSram) {
                rollback.access = AccessFor(bindings, range.core);
                rollback.address = RegionBase(*rollback.access, range);
                rollback.sram = rollback.access->DebugPeek(
                    rollback.address, range.region_size_bytes);
                rollbacks.push_back(std::move(rollback));
                rollbacks.back().access->DebugSeed(
                    rollbacks.back().address, bytes);
            } else {
                if (bindings.hbm_runtime == nullptr)
                    throw Error(
                        "P8 HBM probe requires an HBMRuntime binding");
                rollback.address = range.absolute_address_bytes;
                rollback.current_die = CurrentDie(bindings, range.core);
                rollback.hbm = bindings.hbm_runtime->DebugPeek(
                    rollback.address, rollback.current_die,
                    range.region_size_bytes);
                rollbacks.push_back(std::move(rollback));
                bindings.hbm_runtime->DebugSeed(
                    rollbacks.back().address,
                    rollbacks.back().current_die, bytes);
            }
        }
    } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        try {
            for (auto it = rollbacks.rbegin();
                 it != rollbacks.rend(); ++it) {
                if (it->space == Space::kSram)
                    it->access->DebugRestore(it->address, *it->sram);
                else
                    bindings.hbm_runtime->DebugRestore(*it->hbm);
            }
        } catch (const std::exception &rollback) {
            throw std::runtime_error(
                std::string("P8 probe rollback failed: ") +
                rollback.what());
        }
        std::rethrow_exception(failure);
    }

    Applied result;
    result.spec = spec;
    result.bindings = bindings;
    for (const Range &range : spec.verifications)
        result.verifications.push_back(
            ResolveVerification(range, bindings));
    return result;
}

Result VerifyAfterSimulation(const Applied &applied) {
    RequireBoundary("P8 VerifyAfterSimulation");
    ValidateSpec(applied.spec);
    Require(applied.verifications.size() ==
                applied.spec.verifications.size(),
            "P8 applied probe verification count changed");
    Result result;
    result.scenario = applied.spec.scenario;
    for (const AppliedRange &applied_range : applied.verifications) {
        const Range &range = applied_range.spec;
        std::vector<uint8_t> bytes;
        if (range.space == Space::kSram) {
            const sram::DebugSnapshot snapshot =
                applied_range.sram_access->DebugPeek(
                    applied_range.sram_region_address,
                    range.region_size_bytes);
            bytes = snapshot.payload;
        } else {
            bytes = applied.bindings.hbm_runtime->DebugPeek(
                range.absolute_address_bytes,
                CurrentDie(applied.bindings, range.core),
                range.region_size_bytes).payload;
        }
        const size_t begin =
            static_cast<size_t>(range.payload_offset_bytes);
        const size_t end = begin +
            static_cast<size_t>(range.payload_length_bytes);
        const std::vector<uint8_t> payload(bytes.begin() + begin,
                                           bytes.begin() + end);
        bool sentinels = true;
        if (range.verify_all_bytes_outside_payload) {
            for (size_t index = 0; index < bytes.size(); ++index) {
                if (index >= begin && index < end) continue;
                if (bytes[index] != range.prefill_byte) {
                    sentinels = false;
                    break;
                }
            }
        }
        RangeResult one;
        one.space = range.space;
        one.core = range.core;
        one.region = range.region;
        one.absolute_address_bytes = range.absolute_address_bytes;
        one.payload_bytes = payload.size();
        one.expected_checksum = range.expected_checksum;
        one.checksum = Crc32c(payload);
        one.payload_match = payload == applied_range.expected_payload;
        one.sentinels_intact = sentinels;
        result.ranges.push_back(std::move(one));
    }
    return result;
}

} // namespace p8_double_buffer_probe
