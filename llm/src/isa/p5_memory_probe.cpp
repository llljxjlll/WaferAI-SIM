#include "isa/p5_memory_probe.h"

#include "dte/endpoint_contract.h"

#include <algorithm>
#include <cctype>
#include <exception>
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#include <sstream>

namespace p5_probe {
namespace {

using Json = nlohmann::json;

void RequireDebugBoundary(const char *operation) {
    if (sc_core::sc_is_running())
        throw Error(std::string(operation) +
                    " is forbidden while simulation is running");
}

void RequireKeys(const Json &node, const std::set<std::string> &expected,
                 const char *field) {
    if (!node.is_object())
        throw Error(std::string(field) + " must be an object");
    std::set<std::string> actual;
    for (auto it = node.begin(); it != node.end(); ++it)
        actual.insert(it.key());
    if (actual != expected)
        throw Error(std::string(field) +
                    " fields do not match the v1 canonical schema");
}

uint64_t Unsigned(const Json &node, const char *field) {
    if (!node.is_number_unsigned() && !node.is_number_integer())
        throw Error(std::string(field) +
                    " must be an unsigned integer");
    if (node.is_number_integer() && node.get<int64_t>() < 0)
        throw Error(std::string(field) +
                    " must be an unsigned integer");
    try {
        return node.get<uint64_t>();
    } catch (const Json::exception &) {
        throw Error(std::string(field) +
                    " is outside the uint64 range");
    }
}

uint32_t U32(const Json &node, const char *field) {
    const uint64_t value = Unsigned(node, field);
    if (value > UINT32_MAX)
        throw Error(std::string(field) +
                    " is outside the uint32 range");
    return static_cast<uint32_t>(value);
}

uint8_t U8(const Json &node, const char *field) {
    const uint64_t value = Unsigned(node, field);
    if (value > UINT8_MAX)
        throw Error(std::string(field) +
                    " is outside the uint8 range");
    return static_cast<uint8_t>(value);
}

std::string Token(const Json &node, const char *field) {
    if (!node.is_string())
        throw Error(std::string(field) + " must be a string");
    std::string value = node.get<std::string>();
    if (value.empty() || value.size() > 128)
        throw Error(std::string(field) +
                    " must contain 1..128 characters");
    for (const unsigned char ch : value) {
        if (!std::isalnum(ch) && ch != '_' && ch != '-' && ch != '.')
            throw Error(std::string(field) +
                        " contains a non-canonical character");
    }
    return value;
}

uint64_t CheckedEnd(uint64_t begin, uint64_t size,
                    const char *field) {
    if (size == 0)
        throw Error(std::string(field) +
                    " size must be non-zero");
    if (begin > std::numeric_limits<uint64_t>::max() - size)
        throw Error(std::string(field) + " range overflows");
    return begin + size;
}

std::vector<uint8_t> BuildPayloadRaw(const Spec &spec) {
    std::vector<uint8_t> result(
        static_cast<size_t>(spec.source.length_bytes));
    for (size_t index = 0; index < result.size(); ++index) {
        result[index] = static_cast<uint8_t>(
            spec.source.pattern.seed +
            spec.source.pattern.multiplier * index +
            spec.source.pattern.quarter_step * (index >> 2));
    }
    return result;
}

void ValidateSpec(const Spec &spec) {
    if (spec.version != 1)
        throw Error("P5 memory probe version must be 1");
    Json scenario = spec.scenario;
    (void)Token(scenario, "scenario");
    if (spec.source.core == spec.destination.core)
        throw Error("P5 memory probe source and destination cores must differ");
    if (spec.source.length_bytes == 0 ||
        spec.source.length_bytes > kDteEndpointP2pMaxBytes ||
        spec.source.length_bytes >
            std::numeric_limits<size_t>::max())
        throw Error("P5 memory probe source length is outside P2P bounds");
    if (spec.destination.payload_length_bytes !=
        spec.source.length_bytes)
        throw Error("P5 memory probe source/destination lengths differ");
    if (spec.destination.region_size_bytes == 0 ||
        spec.destination.region_size_bytes >
            std::numeric_limits<size_t>::max())
        throw Error("P5 memory probe destination region size is invalid");
    CheckedEnd(spec.destination.payload_offset_bytes,
               spec.destination.payload_length_bytes,
               "destination payload");
    if (spec.destination.payload_offset_bytes +
            spec.destination.payload_length_bytes >
        spec.destination.region_size_bytes)
        throw Error("P5 memory probe destination payload exceeds region");
    if (!spec.destination.verify_all_bytes_outside_payload)
        throw Error("P5 memory probe must verify all destination sentinels");
    if (spec.source.pattern.multiplier == 0 ||
        spec.source.pattern.quarter_step != 1)
        throw Error("P5 memory probe affine_u8_v1 parameters are non-canonical");
    if (spec.source.space == SourceSpace::kSram) {
        if (spec.source.region.empty())
            throw Error("P5 SRAM probe source requires a region");
        Json region = spec.source.region;
        (void)Token(region, "source.region");
        CheckedEnd(spec.source.offset_bytes, spec.source.length_bytes,
                   "SRAM source");
        if (spec.source.absolute_address_bytes != 0)
            throw Error("P5 SRAM probe source forbids absolute_address_bytes");
    } else {
        if (!spec.source.region.empty())
            throw Error("P5 HBM probe source forbids a region");
        if (spec.source.offset_bytes !=
            spec.source.absolute_address_bytes)
            throw Error("P5 HBM source offset must equal absolute address");
        CheckedEnd(spec.source.absolute_address_bytes,
                   spec.source.length_bytes, "HBM source");
    }
    if (spec.destination.region.empty())
        throw Error("P5 probe destination requires a region");
    Json destination_region = spec.destination.region;
    (void)Token(destination_region, "destination.region");

    const std::vector<uint8_t> payload = BuildPayloadRaw(spec);
    if (!std::any_of(payload.begin(), payload.end(),
                     [](uint8_t value) { return value != 0; }))
        throw Error("P5 memory probe payload must contain non-zero data");
    if (spec.expected_payload_checksum == 0 ||
        Crc32c(payload) != spec.expected_payload_checksum)
        throw Error("P5 memory probe expected checksum is not canonical");
}

sram::AccessUnit *AccessForCore(const Bindings &bindings,
                                uint32_t core,
                                const char *role) {
    const auto it = bindings.sram_by_core.find(core);
    if (it == bindings.sram_by_core.end() || it->second == nullptr)
        throw Error(std::string("P5 memory probe missing ") + role +
                    " SRAM binding for core " + std::to_string(core));
    return it->second;
}

uint64_t RegionBase(sram::AccessUnit &access,
                    const std::string &name,
                    uint64_t expected_size,
                    const char *role) {
    const sram::RegionConfig &region = access.regions().Region(name);
    if (expected_size != 0 && region.size_bytes != expected_size)
        throw Error(std::string("P5 memory probe ") + role +
                    " region size disagrees with hardware");
    CheckedEnd(region.base_bytes, region.size_bytes, role);
    return region.base_bytes;
}

} // namespace

uint32_t Crc32c(const std::vector<uint8_t> &payload) noexcept {
    uint32_t checksum = UINT32_MAX;
    for (const uint8_t value : payload) {
        checksum ^= value;
        for (int bit = 0; bit < 8; ++bit)
            checksum = (checksum >> 1) ^
                       ((checksum & 1U) ? UINT32_C(0x82f63b78) : 0U);
    }
    return ~checksum;
}

std::vector<uint8_t> BuildPayload(const Spec &spec) {
    ValidateSpec(spec);
    return BuildPayloadRaw(spec);
}

Spec Parse(const Json &json) {
    try {
        RequireKeys(
            json,
            {"destination", "expected_payload_checksum", "scenario",
             "source", "version"},
            "root");
        Spec spec;
        spec.version = U32(json.at("version"), "version");
        spec.scenario = Token(json.at("scenario"), "scenario");

        const Json &source = json.at("source");
        if (!source.is_object() || !source.contains("space") ||
            !source.at("space").is_string())
            throw Error("source.space must be SRAM or HBM");
        const std::string space = source.at("space").get<std::string>();
        if (space == "SRAM") {
            RequireKeys(
                source,
                {"core", "length_bytes", "offset_bytes", "pattern",
                 "region", "space"},
                "source");
            spec.source.space = SourceSpace::kSram;
            spec.source.region =
                Token(source.at("region"), "source.region");
        } else if (space == "HBM") {
            RequireKeys(
                source,
                {"absolute_address_bytes", "core", "length_bytes",
                 "offset_bytes", "pattern", "space"},
                "source");
            spec.source.space = SourceSpace::kHbm;
            spec.source.absolute_address_bytes = Unsigned(
                source.at("absolute_address_bytes"),
                "source.absolute_address_bytes");
        } else {
            throw Error("source.space must be SRAM or HBM");
        }
        spec.source.core = U32(source.at("core"), "source.core");
        spec.source.offset_bytes =
            Unsigned(source.at("offset_bytes"), "source.offset_bytes");
        spec.source.length_bytes =
            Unsigned(source.at("length_bytes"), "source.length_bytes");

        const Json &pattern = source.at("pattern");
        RequireKeys(pattern,
                    {"kind", "multiplier", "quarter_step", "seed"},
                    "source.pattern");
        if (!pattern.at("kind").is_string() ||
            pattern.at("kind").get<std::string>() != "affine_u8_v1")
            throw Error("source.pattern.kind must be affine_u8_v1");
        spec.source.pattern.seed =
            U8(pattern.at("seed"), "source.pattern.seed");
        spec.source.pattern.multiplier =
            U8(pattern.at("multiplier"), "source.pattern.multiplier");
        spec.source.pattern.quarter_step =
            U8(pattern.at("quarter_step"), "source.pattern.quarter_step");

        const Json &destination = json.at("destination");
        RequireKeys(
            destination,
            {"core", "payload_length_bytes", "payload_offset_bytes",
             "prefill_byte", "region", "region_size_bytes",
             "verify_all_bytes_outside_payload"},
            "destination");
        spec.destination.core =
            U32(destination.at("core"), "destination.core");
        spec.destination.region =
            Token(destination.at("region"), "destination.region");
        spec.destination.region_size_bytes = Unsigned(
            destination.at("region_size_bytes"),
            "destination.region_size_bytes");
        spec.destination.payload_offset_bytes = Unsigned(
            destination.at("payload_offset_bytes"),
            "destination.payload_offset_bytes");
        spec.destination.payload_length_bytes = Unsigned(
            destination.at("payload_length_bytes"),
            "destination.payload_length_bytes");
        spec.destination.prefill_byte = U8(
            destination.at("prefill_byte"),
            "destination.prefill_byte");
        if (!destination.at("verify_all_bytes_outside_payload").is_boolean())
            throw Error(
                "destination.verify_all_bytes_outside_payload must be boolean");
        spec.destination.verify_all_bytes_outside_payload =
            destination.at("verify_all_bytes_outside_payload").get<bool>();
        spec.expected_payload_checksum = U32(
            json.at("expected_payload_checksum"),
            "expected_payload_checksum");
        ValidateSpec(spec);
        return spec;
    } catch (const Error &) {
        throw;
    } catch (const Json::exception &error) {
        throw Error(std::string("invalid P5 memory probe JSON: ") +
                    error.what());
    }
}

Spec Load(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input)
        throw Error("cannot open P5 memory probe sidecar: " +
                    path.string());
    try {
        Json json;
        input >> json;
        input >> std::ws;
        if (input.peek() != std::char_traits<char>::eof())
            throw Error(
                "P5 memory probe sidecar contains trailing tokens");
        if (input.bad())
            throw Error("failed while reading P5 memory probe sidecar");
        return Parse(json);
    } catch (const Error &) {
        throw;
    } catch (const Json::exception &error) {
        throw Error(std::string("invalid P5 memory probe JSON: ") +
                    error.what());
    }
}

Applied ApplyBeforeSimulation(const Spec &spec,
                              const Bindings &bindings) {
    RequireDebugBoundary("P5 ApplyBeforeSimulation");
    ValidateSpec(spec);
    std::vector<uint8_t> payload = BuildPayloadRaw(spec);
    std::vector<uint8_t> destination_prefill(
        static_cast<size_t>(spec.destination.region_size_bytes),
        spec.destination.prefill_byte);
    // Keep the region exterior uniform while making every unwritten payload
    // byte observable, including when the expected byte equals the sentinel.
    const size_t payload_begin = static_cast<size_t>(
        spec.destination.payload_offset_bytes);
    std::transform(
        payload.begin(), payload.end(),
        destination_prefill.begin() + payload_begin,
        [](uint8_t value) {
            return static_cast<uint8_t>(value ^ UINT8_C(0xff));
        });

    sram::AccessUnit *destination = AccessForCore(
        bindings, spec.destination.core, "destination");
    const uint64_t destination_base = RegionBase(
        *destination, spec.destination.region,
        spec.destination.region_size_bytes, "destination");
    const sram::DebugSnapshot destination_before =
        destination->DebugPeek(
            destination_base, spec.destination.region_size_bytes);

    sram::AccessUnit *source_sram = nullptr;
    uint64_t source_sram_address = 0;
    std::optional<sram::DebugSnapshot> source_sram_before;
    std::optional<HBMRuntimeDebugSnapshot> source_hbm_before;
    int source_current_die = -1;

    if (spec.source.space == SourceSpace::kSram) {
        source_sram = AccessForCore(
            bindings, spec.source.core, "source");
        if (source_sram == destination)
            throw Error("P5 memory probe SRAM bindings must be distinct");
        const uint64_t source_base = RegionBase(
            *source_sram, spec.source.region, 0, "source");
        const sram::RegionConfig &source_region =
            source_sram->regions().Region(spec.source.region);
        if (spec.source.length_bytes > source_region.size_bytes ||
            spec.source.offset_bytes >
                source_region.size_bytes - spec.source.length_bytes)
            throw Error("P5 memory probe source exceeds hardware region");
        source_sram_address =
            source_base + spec.source.offset_bytes;
        source_sram_before = source_sram->DebugPeek(
            source_sram_address, spec.source.length_bytes);
    } else {
        if (bindings.hbm_runtime == nullptr)
            throw Error("P5 HBM probe requires an HBMRuntime binding");
        if (!bindings.current_die_for_core)
            throw Error("P5 HBM probe requires a core-to-die binding");
        source_current_die =
            bindings.current_die_for_core(spec.source.core);
        source_hbm_before = bindings.hbm_runtime->DebugPeek(
            spec.source.absolute_address_bytes, source_current_die,
            spec.source.length_bytes);
    }

    bool source_commit_started = false;
    bool destination_commit_started = false;
    try {
        source_commit_started = true;
        if (source_sram != nullptr)
            source_sram->DebugSeed(source_sram_address, payload);
        else
            bindings.hbm_runtime->DebugSeed(
                spec.source.absolute_address_bytes,
                source_current_die, payload);

        destination_commit_started = true;
        destination->DebugSeed(destination_base, destination_prefill);

        if (source_sram != nullptr) {
            const sram::DebugSnapshot readback =
                source_sram->DebugPeek(
                    source_sram_address, payload.size());
            if (readback.payload != payload ||
                !std::all_of(
                    readback.valid.begin(), readback.valid.end(),
                    [](uint8_t value) { return value != 0; }))
                throw std::runtime_error(
                    "P5 SRAM source seed readback mismatch");
        } else {
            const HBMRuntimeDebugSnapshot readback =
                bindings.hbm_runtime->DebugPeek(
                    spec.source.absolute_address_bytes,
                    source_current_die, payload.size());
            if (readback.payload != payload)
                throw std::runtime_error(
                    "P5 HBM source seed readback mismatch");
        }
    } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        try {
            if (destination_commit_started)
                destination->DebugRestore(
                    destination_base, destination_before);
            if (source_commit_started) {
                if (source_sram != nullptr)
                    source_sram->DebugRestore(
                        source_sram_address, *source_sram_before);
                else
                    bindings.hbm_runtime->DebugRestore(
                        *source_hbm_before);
            }
        } catch (const std::exception &rollback) {
            throw std::runtime_error(
                std::string("P5 memory probe rollback failed: ") +
                rollback.what());
        }
        std::rethrow_exception(failure);
    }

    Applied applied;
    applied.spec = spec;
    applied.expected_payload = std::move(payload);
    applied.expected_payload_checksum =
        spec.expected_payload_checksum;
    applied.source_initialized = true;
    applied.destination_access = destination;
    applied.destination_region_address = destination_base;
    return applied;
}

Result VerifyAfterSimulation(const Applied &applied) {
    RequireDebugBoundary("P5 VerifyAfterSimulation");
    ValidateSpec(applied.spec);
    if (applied.destination_access == nullptr ||
        applied.expected_payload != BuildPayloadRaw(applied.spec) ||
        applied.expected_payload_checksum !=
            applied.spec.expected_payload_checksum ||
        !applied.source_initialized)
        throw Error("P5 applied memory probe state is malformed");

    const sram::DebugSnapshot destination =
        applied.destination_access->DebugPeek(
            applied.destination_region_address,
            applied.spec.destination.region_size_bytes);
    const size_t payload_begin = static_cast<size_t>(
        applied.spec.destination.payload_offset_bytes);
    const size_t payload_size = static_cast<size_t>(
        applied.spec.destination.payload_length_bytes);
    const size_t payload_end = payload_begin + payload_size;

    const bool payload_valid = std::all_of(
        destination.valid.begin() + payload_begin,
        destination.valid.begin() + payload_end,
        [](uint8_t value) { return value != 0; });
    const bool payload_match =
        payload_valid &&
        std::equal(
            applied.expected_payload.begin(),
            applied.expected_payload.end(),
            destination.payload.begin() + payload_begin);
    bool sentinels_intact = true;
    for (size_t index = 0; index < destination.payload.size(); ++index) {
        if (index >= payload_begin && index < payload_end)
            continue;
        if (!destination.valid[index] ||
            destination.payload[index] !=
                applied.spec.destination.prefill_byte) {
            sentinels_intact = false;
            break;
        }
    }

    std::vector<uint8_t> actual_payload(
        destination.payload.begin() + payload_begin,
        destination.payload.begin() + payload_end);
    Result result;
    result.scenario = applied.spec.scenario;
    result.source_initialized = applied.source_initialized;
    result.payload_bytes = payload_size;
    result.expected_checksum = applied.expected_payload_checksum;
    result.destination_checksum = Crc32c(actual_payload);
    result.payload_match = payload_match;
    result.sentinels_intact = sentinels_intact;
    return result;
}

} // namespace p5_probe

namespace p6_probe {
namespace {

using Json = nlohmann::json;
using RegionKey = std::pair<uint32_t, std::string>;

void Require(bool condition, const std::string &message) {
    if (!condition) throw Error(message);
}

void RequireDebugBoundary(const char *operation) {
    if (sc_core::sc_is_running())
        throw Error(std::string(operation) +
                    " is forbidden while simulation is running");
}

void RequireKeys(const Json &node, const std::set<std::string> &expected,
                 const char *field) {
    if (!node.is_object())
        throw Error(std::string(field) + " must be an object");
    std::set<std::string> actual;
    for (auto it = node.begin(); it != node.end(); ++it)
        actual.insert(it.key());
    if (actual != expected)
        throw Error(std::string(field) +
                    " fields do not match the P6 v1 canonical schema");
}

uint64_t Unsigned(const Json &node, const char *field) {
    if (!node.is_number_unsigned() && !node.is_number_integer())
        throw Error(std::string(field) +
                    " must be an unsigned integer");
    if (!node.is_number_unsigned() && node.get<int64_t>() < 0)
        throw Error(std::string(field) +
                    " must be an unsigned integer");
    try {
        return node.get<uint64_t>();
    } catch (const Json::exception &) {
        throw Error(std::string(field) +
                    " is outside the uint64 range");
    }
}

uint32_t U32(const Json &node, const char *field) {
    const uint64_t value = Unsigned(node, field);
    if (value > UINT32_MAX)
        throw Error(std::string(field) +
                    " is outside the uint32 range");
    return static_cast<uint32_t>(value);
}

uint8_t U8(const Json &node, const char *field) {
    const uint64_t value = Unsigned(node, field);
    if (value > UINT8_MAX)
        throw Error(std::string(field) +
                    " is outside the uint8 range");
    return static_cast<uint8_t>(value);
}

std::string Token(const Json &node, const char *field) {
    if (!node.is_string())
        throw Error(std::string(field) + " must be a string");
    std::string value = node.get<std::string>();
    if (value.empty() || value.size() > 128)
        throw Error(std::string(field) +
                    " must contain 1..128 characters");
    for (const unsigned char ch : value) {
        if (!std::isalnum(ch) && ch != '_' && ch != '-' && ch != '.')
            throw Error(std::string(field) +
                        " contains a non-canonical character");
    }
    return value;
}

uint64_t CheckedEnd(uint64_t begin, uint64_t size,
                    const char *field) {
    if (size == 0)
        throw Error(std::string(field) +
                    " size must be non-zero");
    if (begin > std::numeric_limits<uint64_t>::max() - size)
        throw Error(std::string(field) + " range overflows");
    return begin + size;
}

uint64_t CheckedTotal(uint64_t current, uint64_t amount,
                      const char *field) {
    if (amount > kMaxSnapshotBytes ||
        current > kMaxSnapshotBytes - amount)
        throw Error(std::string(field) +
                    " exceeds the P6 bounded snapshot capacity");
    return current + amount;
}

int Base64Value(unsigned char value) {
    if (value >= 'A' && value <= 'Z') return value - 'A';
    if (value >= 'a' && value <= 'z') return value - 'a' + 26;
    if (value >= '0' && value <= '9') return value - '0' + 52;
    if (value == '+') return 62;
    if (value == '/') return 63;
    return -1;
}

std::string EncodeBase64(const std::vector<uint8_t> &bytes) {
    static constexpr char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve(((bytes.size() + 2) / 3) * 4);
    for (size_t offset = 0; offset < bytes.size(); offset += 3) {
        const size_t remaining = bytes.size() - offset;
        const uint32_t word =
            static_cast<uint32_t>(bytes[offset]) << 16 |
            (remaining > 1
                 ? static_cast<uint32_t>(bytes[offset + 1]) << 8
                 : 0U) |
            (remaining > 2 ? bytes[offset + 2] : 0U);
        result.push_back(kAlphabet[(word >> 18) & 63]);
        result.push_back(kAlphabet[(word >> 12) & 63]);
        result.push_back(remaining > 1
                             ? kAlphabet[(word >> 6) & 63]
                             : '=');
        result.push_back(remaining > 2 ? kAlphabet[word & 63] : '=');
    }
    return result;
}

std::vector<uint8_t> DecodeBase64(const Json &node,
                                  const char *field) {
    if (!node.is_string())
        throw Error(std::string(field) + " must be a base64 string");
    const std::string encoded = node.get<std::string>();
    if (encoded.empty() || encoded.size() % 4 != 0)
        throw Error(std::string(field) +
                    " base64 length is not canonical");
    const uint64_t max_encoded =
        ((kMaxSnapshotBytes + 2) / 3) * 4;
    if (encoded.size() > max_encoded)
        throw Error(std::string(field) +
                    " base64 exceeds bounded capacity");

    std::vector<uint8_t> result;
    result.reserve(encoded.size() / 4 * 3);
    for (size_t offset = 0; offset < encoded.size(); offset += 4) {
        const bool last = offset + 4 == encoded.size();
        const bool pad2 = encoded[offset + 2] == '=';
        const bool pad3 = encoded[offset + 3] == '=';
        if (pad2 && !pad3)
            throw Error(std::string(field) +
                        " base64 padding is malformed");
        if ((!last && (pad2 || pad3)) ||
            encoded[offset] == '=' || encoded[offset + 1] == '=')
            throw Error(std::string(field) +
                        " base64 padding is misplaced");
        const int a = Base64Value(encoded[offset]);
        const int b = Base64Value(encoded[offset + 1]);
        const int c = pad2 ? 0 : Base64Value(encoded[offset + 2]);
        const int d = pad3 ? 0 : Base64Value(encoded[offset + 3]);
        if (a < 0 || b < 0 || c < 0 || d < 0)
            throw Error(std::string(field) +
                        " base64 contains an invalid character");
        const uint32_t word =
            static_cast<uint32_t>(a) << 18 |
            static_cast<uint32_t>(b) << 12 |
            static_cast<uint32_t>(c) << 6 |
            static_cast<uint32_t>(d);
        result.push_back(static_cast<uint8_t>(word >> 16));
        if (!pad2)
            result.push_back(static_cast<uint8_t>(word >> 8));
        if (!pad3)
            result.push_back(static_cast<uint8_t>(word));
    }
    if (EncodeBase64(result) != encoded)
        throw Error(std::string(field) +
                    " base64 is not in canonical padded form");
    return result;
}

PatternSpec ParsePattern(const Json &node) {
    RequireKeys(node,
                {"kind", "multiplier", "quarter_step",
                 "reduction_boundary_patch", "seed"},
                "initialization.pattern");
    PatternSpec pattern;
    pattern.kind = Token(node.at("kind"),
                         "initialization.pattern.kind");
    pattern.seed = U8(node.at("seed"),
                      "initialization.pattern.seed");
    pattern.multiplier = U8(
        node.at("multiplier"),
        "initialization.pattern.multiplier");
    pattern.quarter_step = U8(
        node.at("quarter_step"),
        "initialization.pattern.quarter_step");
    if (!node.at("reduction_boundary_patch").is_boolean())
        throw Error(
            "initialization.pattern.reduction_boundary_patch must be boolean");
    pattern.reduction_boundary_patch =
        node.at("reduction_boundary_patch").get<bool>();
    return pattern;
}

void ValidateSpec(const Spec &spec) {
    Require(spec.version == 1,
            "P6 memory probe version must be 1");
    Require(spec.format == "p6-p5-compatible-multi-region-v1",
            "P6 memory probe format is unsupported");
    Json scenario = spec.scenario;
    (void)Token(scenario, "scenario");
    Require(!spec.initializations.empty() &&
                spec.initializations.size() <= kMaxEntries,
            "P6 memory probe initialization count is outside bounds");
    Require(!spec.verifications.empty() &&
                spec.verifications.size() <= kMaxEntries,
            "P6 memory probe verification count is outside bounds");
    Require(spec.initializations.size() <=
                kMaxEntries - spec.verifications.size(),
            "P6 memory probe total entry count exceeds capacity");

    uint64_t snapshot_bytes = 0;
    std::set<RegionKey> regions;
    for (const InitializationSpec &entry : spec.initializations) {
        Json region = entry.region;
        (void)Token(region, "initialization.region");
        Require(entry.region_size_bytes != 0 &&
                    entry.region_size_bytes <=
                        std::numeric_limits<size_t>::max(),
                "P6 initialization region size is invalid");
        const uint64_t end = CheckedEnd(
            entry.offset_bytes, entry.length_bytes,
            "P6 initialization payload");
        Require(end <= entry.region_size_bytes,
                "P6 initialization payload exceeds its region");
        Require(entry.expected_bytes.size() ==
                    entry.length_bytes,
                "P6 initialization decoded byte length differs from declaration");
        Require(p5_probe::Crc32c(entry.expected_bytes) ==
                    entry.checksum,
                "P6 initialization checksum is not canonical");
        if (entry.has_expected_after) {
            Require(entry.expected_after_bytes.size() ==
                        entry.length_bytes,
                    "P6 initialization expected-after decoded byte length differs from declaration");
            Require(p5_probe::Crc32c(entry.expected_after_bytes) ==
                        entry.expected_after_checksum,
                    "P6 initialization expected-after checksum is not canonical");
        } else {
            Require(entry.expected_after_bytes.empty() &&
                        entry.expected_after_checksum == 0,
                    "P6 initialization expected-after state is malformed");
        }
        Require(entry.pattern.kind == "affine_u8_v1" &&
                    entry.pattern.multiplier != 0 &&
                    entry.pattern.quarter_step == 1,
                "P6 initialization pattern is non-canonical");
        Require(regions.emplace(entry.core, entry.region).second,
                "P6 memory probe region appears more than once");
        snapshot_bytes = CheckedTotal(
            snapshot_bytes, entry.region_size_bytes,
            "P6 initialization snapshots");
    }
    for (const VerificationSpec &entry : spec.verifications) {
        Json region = entry.region;
        (void)Token(region, "verification.region");
        Require(entry.region_size_bytes != 0 &&
                    entry.region_size_bytes <=
                        std::numeric_limits<size_t>::max(),
                "P6 verification region size is invalid");
        const uint64_t end = CheckedEnd(
            entry.payload_offset_bytes,
            entry.payload_length_bytes,
            "P6 verification payload");
        Require(end <= entry.region_size_bytes,
                "P6 verification payload exceeds its region");
        Require(entry.expected_bytes.size() ==
                    entry.payload_length_bytes,
                "P6 verification decoded byte length differs from declaration");
        Require(p5_probe::Crc32c(entry.expected_bytes) ==
                    entry.expected_checksum,
                "P6 verification checksum is not canonical");
        Require(entry.verify_all_bytes_outside_payload,
                "P6 verification must check all outside sentinels");
        Require(regions.emplace(entry.core, entry.region).second,
                "P6 memory probe region appears more than once");
        snapshot_bytes = CheckedTotal(
            snapshot_bytes, entry.region_size_bytes,
            "P6 verification snapshots");
    }
}

sram::AccessUnit *AccessForCore(
    const p5_probe::Bindings &bindings, uint32_t core,
    const char *role) {
    const auto found = bindings.sram_by_core.find(core);
    if (found == bindings.sram_by_core.end() ||
        found->second == nullptr)
        throw Error(std::string("P6 memory probe missing ") + role +
                    " SRAM binding for core " + std::to_string(core));
    return found->second;
}

uint64_t RegionBase(sram::AccessUnit &access,
                    const std::string &name,
                    uint64_t expected_size,
                    const char *role) {
    const sram::RegionConfig &region =
        access.regions().Region(name);
    if (region.size_bytes != expected_size)
        throw Error(std::string("P6 memory probe ") + role +
                    " region size disagrees with hardware");
    CheckedEnd(region.base_bytes, region.size_bytes, role);
    return region.base_bytes;
}

std::vector<uint8_t> InitializationImage(
    const InitializationSpec &entry) {
    std::vector<uint8_t> image(
        static_cast<size_t>(entry.region_size_bytes),
        entry.prefill_byte);
    std::copy(entry.expected_bytes.begin(),
              entry.expected_bytes.end(),
              image.begin() +
                  static_cast<size_t>(entry.offset_bytes));
    return image;
}

std::vector<uint8_t> VerificationImage(
    const VerificationSpec &entry) {
    std::vector<uint8_t> image(
        static_cast<size_t>(entry.region_size_bytes),
        entry.prefill_byte);
    std::transform(
        entry.expected_bytes.begin(), entry.expected_bytes.end(),
        image.begin() +
            static_cast<size_t>(entry.payload_offset_bytes),
        [](uint8_t value) {
            return static_cast<uint8_t>(value ^ UINT8_C(0xff));
        });
    return image;
}

struct ResolvedRegion {
    sram::AccessUnit *access = nullptr;
    uint64_t base = 0;
    sram::DebugSnapshot before;
    std::vector<uint8_t> image;
};

bool PayloadValidAndEqual(
    const sram::DebugSnapshot &snapshot, size_t begin,
    const std::vector<uint8_t> &expected) {
    const size_t end = begin + expected.size();
    return std::all_of(
               snapshot.valid.begin() + begin,
               snapshot.valid.begin() + end,
               [](uint8_t value) { return value != 0; }) &&
           std::equal(expected.begin(), expected.end(),
                      snapshot.payload.begin() + begin);
}

bool OutsideSentinels(
    const sram::DebugSnapshot &snapshot, size_t begin,
    size_t size, uint8_t sentinel) {
    const size_t end = begin + size;
    for (size_t index = 0; index < snapshot.payload.size();
         ++index) {
        if (index >= begin && index < end) continue;
        if (!snapshot.valid[index] ||
            snapshot.payload[index] != sentinel)
            return false;
    }
    return true;
}

} // namespace

bool Result::Passed() const noexcept {
    return std::all_of(
               sources.begin(), sources.end(),
               [](const SourceResult &entry) {
                   return entry.source_initialized &&
                          entry.payload_match &&
                          entry.sentinels_intact &&
                          entry.checksum == entry.expected_checksum;
               }) &&
           std::all_of(
               verifications.begin(), verifications.end(),
               [](const VerificationResult &entry) {
                   return entry.payload_match &&
                          entry.sentinels_intact &&
                          entry.checksum == entry.expected_checksum;
               });
}

Spec Parse(const Json &json) {
    try {
        RequireKeys(
            json,
            {"format", "initializations", "scenario",
             "verifications", "version"},
            "root");
        Spec spec;
        spec.version = U32(json.at("version"), "version");
        spec.format = Token(json.at("format"), "format");
        spec.scenario = Token(json.at("scenario"), "scenario");

        const Json &initializations = json.at("initializations");
        if (!initializations.is_array())
            throw Error("initializations must be an array");
        if (initializations.size() > kMaxEntries)
            throw Error("initialization count exceeds P6 capacity");
        for (size_t index = 0; index < initializations.size();
             ++index) {
            const Json &node = initializations[index];
            const bool has_expected_after_bytes =
                node.contains("expected_after_bytes_base64");
            const bool has_expected_after_checksum =
                node.contains("expected_after_checksum");
            if (has_expected_after_bytes !=
                has_expected_after_checksum)
                throw Error(
                    "P6 initialization expected-after fields must appear together");
            std::set<std::string> initialization_keys = {
                "bytes_base64", "checksum", "core",
                "length_bytes", "offset_bytes", "pattern",
                "prefill_byte", "region",
                "region_size_bytes", "space"};
            if (has_expected_after_bytes) {
                initialization_keys.insert(
                    "expected_after_bytes_base64");
                initialization_keys.insert(
                    "expected_after_checksum");
            }
            RequireKeys(node, initialization_keys,
                        "initialization");
            if (!node.at("space").is_string() ||
                node.at("space").get<std::string>() != "SRAM")
                throw Error(
                    "P6 initialization.space must be SRAM");
            InitializationSpec entry;
            entry.core = U32(node.at("core"),
                             "initialization.core");
            entry.region = Token(node.at("region"),
                                 "initialization.region");
            entry.region_size_bytes = Unsigned(
                node.at("region_size_bytes"),
                "initialization.region_size_bytes");
            entry.offset_bytes = Unsigned(
                node.at("offset_bytes"),
                "initialization.offset_bytes");
            entry.length_bytes = Unsigned(
                node.at("length_bytes"),
                "initialization.length_bytes");
            entry.prefill_byte = U8(
                node.at("prefill_byte"),
                "initialization.prefill_byte");
            entry.pattern = ParsePattern(node.at("pattern"));
            entry.expected_bytes = DecodeBase64(
                node.at("bytes_base64"),
                "initialization.bytes_base64");
            entry.checksum = U32(
                node.at("checksum"),
                "initialization.checksum");
            if (has_expected_after_bytes) {
                entry.has_expected_after = true;
                entry.expected_after_bytes = DecodeBase64(
                    node.at("expected_after_bytes_base64"),
                    "initialization.expected_after_bytes_base64");
                entry.expected_after_checksum = U32(
                    node.at("expected_after_checksum"),
                    "initialization.expected_after_checksum");
            }
            spec.initializations.push_back(std::move(entry));
        }

        const Json &verifications = json.at("verifications");
        if (!verifications.is_array())
            throw Error("verifications must be an array");
        if (verifications.size() > kMaxEntries)
            throw Error("verification count exceeds P6 capacity");
        for (size_t index = 0; index < verifications.size();
             ++index) {
            const Json &node = verifications[index];
            RequireKeys(
                node,
                {"core", "expected_bytes_base64",
                 "expected_checksum", "payload_length_bytes",
                 "payload_offset_bytes", "prefill_byte",
                 "region", "region_size_bytes",
                 "verify_all_bytes_outside_payload"},
                "verification");
            VerificationSpec entry;
            entry.core = U32(node.at("core"),
                             "verification.core");
            entry.region = Token(node.at("region"),
                                 "verification.region");
            entry.region_size_bytes = Unsigned(
                node.at("region_size_bytes"),
                "verification.region_size_bytes");
            entry.payload_offset_bytes = Unsigned(
                node.at("payload_offset_bytes"),
                "verification.payload_offset_bytes");
            entry.payload_length_bytes = Unsigned(
                node.at("payload_length_bytes"),
                "verification.payload_length_bytes");
            entry.expected_bytes = DecodeBase64(
                node.at("expected_bytes_base64"),
                "verification.expected_bytes_base64");
            entry.expected_checksum = U32(
                node.at("expected_checksum"),
                "verification.expected_checksum");
            entry.prefill_byte = U8(
                node.at("prefill_byte"),
                "verification.prefill_byte");
            if (!node.at(
                    "verify_all_bytes_outside_payload").is_boolean())
                throw Error(
                    "verification.verify_all_bytes_outside_payload must be boolean");
            entry.verify_all_bytes_outside_payload =
                node.at(
                    "verify_all_bytes_outside_payload").get<bool>();
            spec.verifications.push_back(std::move(entry));
        }
        ValidateSpec(spec);
        return spec;
    } catch (const Error &) {
        throw;
    } catch (const Json::exception &error) {
        throw Error(std::string("invalid P6 memory probe JSON: ") +
                    error.what());
    }
}

Spec Load(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input)
        throw Error("cannot open P6 memory probe sidecar: " +
                    path.string());
    try {
        Json json;
        input >> json;
        input >> std::ws;
        if (input.peek() != std::char_traits<char>::eof())
            throw Error(
                "P6 memory probe sidecar contains trailing tokens");
        if (input.bad())
            throw Error(
                "failed while reading P6 memory probe sidecar");
        return Parse(json);
    } catch (const Error &) {
        throw;
    } catch (const Json::exception &error) {
        throw Error(std::string("invalid P6 memory probe JSON: ") +
                    error.what());
    }
}

Applied ApplyBeforeSimulation(
    const Spec &spec, const p5_probe::Bindings &bindings,
    const BeforeSeedHook &before_seed) {
    RequireDebugBoundary("P6 ApplyBeforeSimulation");
    ValidateSpec(spec);

    std::vector<ResolvedRegion> resolved;
    resolved.reserve(spec.initializations.size() +
                     spec.verifications.size());
    for (const InitializationSpec &entry :
         spec.initializations) {
        ResolvedRegion region;
        region.access = AccessForCore(
            bindings, entry.core, "initialization");
        region.base = RegionBase(
            *region.access, entry.region,
            entry.region_size_bytes, "initialization");
        region.before = region.access->DebugPeek(
            region.base, entry.region_size_bytes);
        region.image = InitializationImage(entry);
        resolved.push_back(std::move(region));
    }
    for (const VerificationSpec &entry : spec.verifications) {
        ResolvedRegion region;
        region.access = AccessForCore(
            bindings, entry.core, "verification");
        region.base = RegionBase(
            *region.access, entry.region,
            entry.region_size_bytes, "verification");
        region.before = region.access->DebugPeek(
            region.base, entry.region_size_bytes);
        region.image = VerificationImage(entry);
        resolved.push_back(std::move(region));
    }

    size_t committed = 0;
    try {
        for (size_t index = 0; index < resolved.size(); ++index) {
            if (before_seed) before_seed(index);
            resolved[index].access->DebugSeed(
                resolved[index].base, resolved[index].image);
            ++committed;
        }
        for (const ResolvedRegion &region : resolved) {
            const sram::DebugSnapshot readback =
                region.access->DebugPeek(
                    region.base, region.image.size());
            if (readback.payload != region.image ||
                !std::all_of(
                    readback.valid.begin(), readback.valid.end(),
                    [](uint8_t value) { return value != 0; }))
                throw std::runtime_error(
                    "P6 memory probe seed readback mismatch");
        }
    } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        try {
            while (committed != 0) {
                --committed;
                resolved[committed].access->DebugRestore(
                    resolved[committed].base,
                    resolved[committed].before);
            }
        } catch (const std::exception &rollback) {
            throw std::runtime_error(
                std::string("P6 memory probe rollback failed: ") +
                rollback.what());
        }
        std::rethrow_exception(failure);
    }

    Applied applied;
    applied.spec = spec;
    for (size_t index = 0; index < spec.initializations.size();
         ++index) {
        applied.initializations.push_back(
            {spec.initializations[index],
             resolved[index].access, resolved[index].base});
    }
    const size_t verification_base = spec.initializations.size();
    for (size_t index = 0; index < spec.verifications.size();
         ++index) {
        const ResolvedRegion &region =
            resolved[verification_base + index];
        applied.verifications.push_back(
            {spec.verifications[index],
             region.access, region.base});
    }
    return applied;
}

Result VerifyAfterSimulation(const Applied &applied) {
    RequireDebugBoundary("P6 VerifyAfterSimulation");
    ValidateSpec(applied.spec);
    Require(applied.initializations.size() ==
                applied.spec.initializations.size() &&
                applied.verifications.size() ==
                    applied.spec.verifications.size(),
            "P6 applied memory probe entry count is malformed");

    Result result;
    result.scenario = applied.spec.scenario;
    for (size_t index = 0; index < applied.initializations.size();
         ++index) {
        const AppliedInitialization &entry =
            applied.initializations[index];
        Require(entry.access != nullptr &&
                    entry.spec.core ==
                        applied.spec.initializations[index].core &&
                    entry.spec.region ==
                        applied.spec.initializations[index].region &&
                    entry.spec.expected_bytes ==
                        applied.spec.initializations[index].expected_bytes &&
                    entry.spec.has_expected_after ==
                        applied.spec.initializations[index].has_expected_after &&
                    entry.spec.expected_after_bytes ==
                        applied.spec.initializations[index].expected_after_bytes &&
                    entry.spec.expected_after_checksum ==
                        applied.spec.initializations[index].expected_after_checksum,
                "P6 applied initialization state is malformed");
        const std::vector<uint8_t> &expected =
            entry.spec.has_expected_after
                ? entry.spec.expected_after_bytes
                : entry.spec.expected_bytes;
        const uint32_t expected_checksum =
            entry.spec.has_expected_after
                ? entry.spec.expected_after_checksum
                : entry.spec.checksum;
        const sram::DebugSnapshot snapshot =
            entry.access->DebugPeek(
                entry.region_address,
                entry.spec.region_size_bytes);
        const size_t begin =
            static_cast<size_t>(entry.spec.offset_bytes);
        std::vector<uint8_t> actual(
            snapshot.payload.begin() + begin,
            snapshot.payload.begin() + begin + expected.size());
        SourceResult source;
        source.core = entry.spec.core;
        source.region = entry.spec.region;
        source.payload_bytes = entry.spec.length_bytes;
        source.expected_checksum = expected_checksum;
        source.checksum = p5_probe::Crc32c(actual);
        source.source_initialized = true;
        source.payload_match = PayloadValidAndEqual(
            snapshot, begin, expected);
        source.sentinels_intact = OutsideSentinels(
            snapshot, begin, expected.size(),
            entry.spec.prefill_byte);
        result.sources.push_back(std::move(source));
    }

    for (size_t index = 0; index < applied.verifications.size();
         ++index) {
        const AppliedVerification &entry =
            applied.verifications[index];
        Require(entry.access != nullptr &&
                    entry.spec.core ==
                        applied.spec.verifications[index].core &&
                    entry.spec.region ==
                        applied.spec.verifications[index].region &&
                    entry.spec.expected_bytes ==
                        applied.spec.verifications[index].expected_bytes,
                "P6 applied verification state is malformed");
        const sram::DebugSnapshot snapshot =
            entry.access->DebugPeek(
                entry.region_address,
                entry.spec.region_size_bytes);
        const size_t begin = static_cast<size_t>(
            entry.spec.payload_offset_bytes);
        std::vector<uint8_t> actual(
            snapshot.payload.begin() + begin,
            snapshot.payload.begin() + begin +
                entry.spec.expected_bytes.size());
        VerificationResult verification;
        verification.core = entry.spec.core;
        verification.region = entry.spec.region;
        verification.payload_bytes =
            entry.spec.payload_length_bytes;
        verification.expected_checksum =
            entry.spec.expected_checksum;
        verification.checksum = p5_probe::Crc32c(actual);
        verification.payload_match = PayloadValidAndEqual(
            snapshot, begin, entry.spec.expected_bytes);
        verification.sentinels_intact = OutsideSentinels(
            snapshot, begin, entry.spec.expected_bytes.size(),
            entry.spec.prefill_byte);
        result.verifications.push_back(
            std::move(verification));
    }
    return result;
}

} // namespace p6_probe
