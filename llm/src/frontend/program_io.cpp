#include "frontend/program_io.h"

#include "frontend/program_finalizer.h"
#include "isa/program_format.h"
#include "memory/hbm_runtime.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <exception>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>
#include <variant>

namespace frontend::program_io {
namespace {

using Json = nlohmann::json;

constexpr std::string_view kBlobSchema =
    "wafer_frontend.program_blob/v1alpha1";
constexpr std::string_view kInitializationSchema =
    "wafer_frontend.program_sram_initialization/v1alpha2";
constexpr std::string_view kProbeSchema =
    "wafer_frontend.program_output_probe/v1alpha2";

[[noreturn]] void Fail(const std::string &path, const std::string &message) {
    throw Error(path + ": " + message);
}

uint32_t RotateRight(uint32_t value, unsigned int shift) {
    return (value >> shift) | (value << (32 - shift));
}

std::string Sha256Impl(std::string_view input) {
    static constexpr std::array<uint32_t, 64> constants{{
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2}};
    std::array<uint32_t, 8> state{{
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}};
    if (input.size() > std::numeric_limits<uint64_t>::max() / 8)
        throw Error("SHA-256 input is too large");
    std::vector<uint8_t> bytes(input.begin(), input.end());
    const uint64_t bit_length = static_cast<uint64_t>(bytes.size()) * 8;
    bytes.push_back(0x80);
    while (bytes.size() % 64 != 56) bytes.push_back(0);
    for (int shift = 56; shift >= 0; shift -= 8)
        bytes.push_back(static_cast<uint8_t>(bit_length >> shift));
    for (std::size_t block = 0; block < bytes.size(); block += 64) {
        std::array<uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index) {
            const std::size_t offset = block + index * 4;
            words[index] = (static_cast<uint32_t>(bytes[offset]) << 24) |
                           (static_cast<uint32_t>(bytes[offset + 1]) << 16) |
                           (static_cast<uint32_t>(bytes[offset + 2]) << 8) |
                           static_cast<uint32_t>(bytes[offset + 3]);
        }
        for (std::size_t index = 16; index < words.size(); ++index) {
            const uint32_t s0 = RotateRight(words[index - 15], 7) ^
                                RotateRight(words[index - 15], 18) ^
                                (words[index - 15] >> 3);
            const uint32_t s1 = RotateRight(words[index - 2], 17) ^
                                RotateRight(words[index - 2], 19) ^
                                (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        uint32_t a = state[0];
        uint32_t b = state[1];
        uint32_t c = state[2];
        uint32_t d = state[3];
        uint32_t e = state[4];
        uint32_t f = state[5];
        uint32_t g = state[6];
        uint32_t h = state[7];
        for (std::size_t index = 0; index < words.size(); ++index) {
            const uint32_t sum1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^
                                  RotateRight(e, 25);
            const uint32_t choose = (e & f) ^ ((~e) & g);
            const uint32_t temp1 =
                h + sum1 + choose + constants[index] + words[index];
            const uint32_t sum0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^
                                  RotateRight(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }
    static constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (const uint32_t word : state)
        for (int shift = 28; shift >= 0; shift -= 4)
            result.push_back(hex[(word >> shift) & 0xf]);
    return result;
}

std::string CanonicalDump(const Json &value) {
    try {
        return value.dump(-1, ' ', false, Json::error_handler_t::strict);
    } catch (const Json::exception &error) {
        throw Error(std::string("canonical JSON encoding failed: ") +
                    error.what());
    }
}

std::string CanonicalDigest(const Json &value) {
    return Sha256Impl(CanonicalDump(value));
}

Json Without(Json value, std::initializer_list<const char *> fields) {
    for (const char *field : fields) value.erase(field);
    return value;
}

std::string StableId(std::string_view kind, std::string_view schema,
                     const Json &semantic) {
    const Json identity{{"kind", kind},
                        {"schema_version", schema},
                        {"semantic_key", semantic}};
    return std::string(kind) + "_" + CanonicalDigest(identity).substr(0, 16);
}

Json ParseStrictJson(std::string_view text, const char *what) {
    std::vector<std::set<std::string>> object_keys;
    auto callback = [&object_keys](int, Json::parse_event_t event,
                                   Json &parsed) {
        if (event == Json::parse_event_t::object_start) {
            object_keys.emplace_back();
        } else if (event == Json::parse_event_t::key) {
            if (object_keys.empty())
                throw Error("$: internal duplicate-key parser state");
            const std::string key = parsed.get<std::string>();
            if (!object_keys.back().insert(key).second)
                throw Error("$: duplicate object key '" + key + "'");
        } else if (event == Json::parse_event_t::object_end) {
            if (object_keys.empty())
                throw Error("$: internal duplicate-key parser state");
            object_keys.pop_back();
        }
        return true;
    };
    try {
        Json value = Json::parse(text.begin(), text.end(), callback, true, false);
        if (value.is_discarded()) Fail("$", "invalid JSON");
        (void)CanonicalDump(value);
        return value;
    } catch (const Error &) {
        throw;
    } catch (const std::exception &error) {
        throw Error(std::string(what) + ": invalid JSON: " + error.what());
    }
}

void ExactObject(const Json &value, const std::string &path,
                 std::initializer_list<const char *> fields) {
    if (!value.is_object()) Fail(path, "must be an object");
    std::set<std::string> expected;
    for (const char *field : fields) expected.emplace(field);
    for (auto it = value.begin(); it != value.end(); ++it)
        if (expected.erase(it.key()) == 0)
            Fail(path + "." + it.key(), "unknown field");
    if (!expected.empty()) Fail(path + "." + *expected.begin(), "missing field");
}

const Json &Field(const Json &value, const char *name,
                  const std::string &path) {
    const auto found = value.find(name);
    if (found == value.end()) Fail(path + "." + name, "missing field");
    return *found;
}

std::string String(const Json &value, const std::string &path,
                   bool allow_empty = false) {
    if (!value.is_string()) Fail(path, "must be a string");
    std::string result = value.get<std::string>();
    if (!allow_empty && result.empty()) Fail(path, "must be non-empty");
    return result;
}

uint64_t U64(const Json &value, const std::string &path) {
    if (value.is_number_unsigned()) return value.get<uint64_t>();
    if (value.is_number_integer()) {
        const int64_t signed_value = value.get<int64_t>();
        if (signed_value >= 0) return static_cast<uint64_t>(signed_value);
    }
    Fail(path, "must be an unsigned 64-bit integer");
}

uint32_t RuntimeCore(const Json &value, const std::string &path) {
    const uint64_t result = U64(value, path);
    if (result > 0xffff) Fail(path, "must fit ProgramArtifact uint16 core id");
    return static_cast<uint32_t>(result);
}

void RequireSha256(const std::string &value, const std::string &path) {
    if (value.size() != 64 ||
        !std::all_of(value.begin(), value.end(), [](unsigned char ch) {
            return std::isdigit(ch) || (ch >= 'a' && ch <= 'f');
        }))
        Fail(path, "must be a canonical lowercase SHA-256 hex digest");
}

void RequireSymbolName(const std::string &value, const std::string &path) {
    if (value.empty() || value.size() > 255 ||
        value.find('\0') != std::string::npos)
        Fail(path, "must contain 1..255 UTF-8 bytes without NUL");
}

uint64_t CheckedEnd(uint64_t offset, uint64_t length,
                    const std::string &path) {
    if (length == 0) Fail(path + ".length_bytes", "must be positive");
    if (offset > std::numeric_limits<uint64_t>::max() - length)
        Fail(path, "byte range overflows uint64");
    return offset + length;
}

std::vector<uint64_t> U64Array(const Json &value, const std::string &path,
                               bool require_nonempty) {
    if (!value.is_array()) Fail(path, "must be an array");
    if (require_nonempty && value.empty()) Fail(path, "must be non-empty");
    std::vector<uint64_t> result;
    result.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index)
        result.push_back(U64(value[index], path + "[" +
                                           std::to_string(index) + "]"));
    return result;
}

TensorSlice ParseSlice(const Json &value, const std::string &path) {
    ExactObject(value, path, {"value_id", "offset", "shape"});
    TensorSlice result;
    result.value_id = String(Field(value, "value_id", path), path + ".value_id");
    result.offset = U64Array(Field(value, "offset", path), path + ".offset", false);
    result.shape = U64Array(Field(value, "shape", path), path + ".shape", true);
    if (result.offset.size() != result.shape.size())
        Fail(path, "offset and shape must have equal non-zero rank");
    for (std::size_t index = 0; index < result.shape.size(); ++index)
        if (result.shape[index] == 0)
            Fail(path + ".shape[" + std::to_string(index) + "]",
                 "must be greater than zero");
    return result;
}

DType ParseDType(const Json &value, const std::string &path) {
    const std::string token = String(value, path);
    if (token == "fp16") return DType::FP16;
    if (token == "fp32") return DType::FP32;
    if (token == "int32") return DType::INT32;
    Fail(path, "must be fp16, fp32, or int32");
}

Mode ParseMode(const Json &value, const std::string &path) {
    const std::string token = String(value, path);
    if (token == "timing") return Mode::TIMING;
    if (token == "functional") return Mode::FUNCTIONAL;
    Fail(path, "must be timing or functional");
}

Purpose ParsePurpose(const Json &value, const std::string &path) {
    const std::string token = String(value, path);
    if (token == "activation") return Purpose::ACTIVATION;
    if (token == "weight") return Purpose::WEIGHT;
    if (token == "state") return Purpose::STATE;
    if (token == "timing_partial") return Purpose::TIMING_PARTIAL;
    Fail(path, "must be activation, weight, state, or timing_partial");
}

int Base64Value(unsigned char ch) {
    if (ch >= 'A' && ch <= 'Z') return ch - 'A';
    if (ch >= 'a' && ch <= 'z') return ch - 'a' + 26;
    if (ch >= '0' && ch <= '9') return ch - '0' + 52;
    if (ch == '+') return 62;
    if (ch == '/') return 63;
    return -1;
}

std::string EncodeBase64(const std::vector<uint8_t> &bytes) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve(((bytes.size() + 2) / 3) * 4);
    for (std::size_t index = 0; index < bytes.size(); index += 3) {
        const uint32_t a = bytes[index];
        const uint32_t b = index + 1 < bytes.size() ? bytes[index + 1] : 0;
        const uint32_t c = index + 2 < bytes.size() ? bytes[index + 2] : 0;
        const uint32_t word = (a << 16) | (b << 8) | c;
        result.push_back(alphabet[(word >> 18) & 0x3f]);
        result.push_back(alphabet[(word >> 12) & 0x3f]);
        result.push_back(index + 1 < bytes.size()
                             ? alphabet[(word >> 6) & 0x3f]
                             : '=');
        result.push_back(index + 2 < bytes.size() ? alphabet[word & 0x3f]
                                                   : '=');
    }
    return result;
}

std::vector<uint8_t> DecodeBase64(const std::string &text,
                                  const std::string &path) {
    if (text.empty() || text.size() % 4 != 0)
        Fail(path, "must be strict canonical base64");
    std::vector<uint8_t> result;
    result.reserve((text.size() / 4) * 3);
    for (std::size_t index = 0; index < text.size(); index += 4) {
        const bool last = index + 4 == text.size();
        const int a = Base64Value(static_cast<unsigned char>(text[index]));
        const int b = Base64Value(static_cast<unsigned char>(text[index + 1]));
        const bool pad2 = text[index + 2] == '=';
        const bool pad3 = text[index + 3] == '=';
        const int c = pad2 ? 0 : Base64Value(
            static_cast<unsigned char>(text[index + 2]));
        const int d = pad3 ? 0 : Base64Value(
            static_cast<unsigned char>(text[index + 3]));
        if (a < 0 || b < 0 || c < 0 || d < 0 ||
            (!last && (pad2 || pad3)) || (pad2 && !pad3))
            Fail(path, "must be strict canonical base64");
        const uint32_t word = (static_cast<uint32_t>(a) << 18) |
                              (static_cast<uint32_t>(b) << 12) |
                              (static_cast<uint32_t>(c) << 6) |
                              static_cast<uint32_t>(d);
        result.push_back(static_cast<uint8_t>(word >> 16));
        if (!pad2) result.push_back(static_cast<uint8_t>(word >> 8));
        if (!pad3) result.push_back(static_cast<uint8_t>(word));
    }
    if (EncodeBase64(result) != text)
        Fail(path, "must be strict canonical base64");
    return result;
}

Blob ParseBlob(const Json &value, const std::string &path) {
    ExactObject(value, path,
                {"id", "bytes_base64", "length_bytes", "sha256"});
    Blob result;
    result.id = String(Field(value, "id", path), path + ".id");
    const std::string encoded =
        String(Field(value, "bytes_base64", path), path + ".bytes_base64");
    const uint64_t declared =
        U64(Field(value, "length_bytes", path), path + ".length_bytes");
    if (declared == 0 || declared > std::numeric_limits<std::size_t>::max())
        Fail(path + ".length_bytes", "must be a positive host-sized length");
    result.sha256 = String(Field(value, "sha256", path), path + ".sha256");
    RequireSha256(result.sha256, path + ".sha256");
    result.bytes = DecodeBase64(encoded, path + ".bytes_base64");
    if (result.bytes.size() != declared)
        Fail(path + ".length_bytes", "decoded bytes disagree with length_bytes");
    if (Sha256Hex(result.bytes) != result.sha256)
        Fail(path + ".sha256", "decoded bytes disagree with sha256");
    const std::string expected = StableId(
        "program_blob", kBlobSchema, Without(value, {"id"}));
    if (result.id != expected)
        Fail(path + ".id", "unstable artifact id; expected '" + expected + "'");
    return result;
}

Target ParseTarget(const Json &value, const std::string &path) {
    if (!value.is_object()) Fail(path, "must be an object");
    const std::string kind =
        String(Field(value, "kind", path), path + ".kind");
    if (kind == "sram") {
        ExactObject(value, path,
                    {"kind", "runtime_core_id", "program_symbol_ref",
                     "finalized_symbol_index", "expected_symbol_name",
                     "buffer_abi_id", "storage_id", "value_id",
                     "tensor_slice", "dtype", "layout"});
        SramTarget result;
        result.runtime_core_id = RuntimeCore(
            Field(value, "runtime_core_id", path),
            path + ".runtime_core_id");
        result.program_symbol_ref = String(
            Field(value, "program_symbol_ref", path),
            path + ".program_symbol_ref");
        result.finalized_symbol_index = U64(
            Field(value, "finalized_symbol_index", path),
            path + ".finalized_symbol_index");
        result.expected_symbol_name = String(
            Field(value, "expected_symbol_name", path),
            path + ".expected_symbol_name");
        RequireSymbolName(result.expected_symbol_name,
                          path + ".expected_symbol_name");
        result.buffer_abi_id = String(
            Field(value, "buffer_abi_id", path), path + ".buffer_abi_id");
        result.storage_id = String(
            Field(value, "storage_id", path), path + ".storage_id");
        result.value_id = String(
            Field(value, "value_id", path), path + ".value_id");
        result.tensor_slice = ParseSlice(
            Field(value, "tensor_slice", path), path + ".tensor_slice");
        if (result.tensor_slice.value_id != result.value_id)
            Fail(path + ".tensor_slice.value_id", "disagrees with value_id");
        result.dtype =
            ParseDType(Field(value, "dtype", path), path + ".dtype");
        result.layout =
            String(Field(value, "layout", path), path + ".layout");
        return result;
    }
    if (kind == "hbm") {
        ExactObject(value, path,
                    {"kind", "program_symbol_ref",
                     "finalized_symbol_index", "expected_symbol_name",
                     "state_abi_id", "state_ref", "hbm_binding_ref"});
        HbmTarget result;
        result.program_symbol_ref = String(
            Field(value, "program_symbol_ref", path),
            path + ".program_symbol_ref");
        result.finalized_symbol_index = U64(
            Field(value, "finalized_symbol_index", path),
            path + ".finalized_symbol_index");
        result.expected_symbol_name = String(
            Field(value, "expected_symbol_name", path),
            path + ".expected_symbol_name");
        RequireSymbolName(result.expected_symbol_name,
                          path + ".expected_symbol_name");
        result.state_abi_id = String(
            Field(value, "state_abi_id", path), path + ".state_abi_id");
        result.state_ref = String(
            Field(value, "state_ref", path), path + ".state_ref");
        result.hbm_binding_ref = String(
            Field(value, "hbm_binding_ref", path),
            path + ".hbm_binding_ref");
        return result;
    }
    Fail(path + ".kind", "must be sram or hbm");
}

Initialization ParseInitialization(const Json &value,
                                   const std::string &path) {
    ExactObject(value, path,
                {"id", "target", "offset_bytes", "length_bytes",
                 "blob_ref", "purpose"});
    Initialization result;
    result.id = String(Field(value, "id", path), path + ".id");
    result.target =
        ParseTarget(Field(value, "target", path), path + ".target");
    result.offset_bytes =
        U64(Field(value, "offset_bytes", path), path + ".offset_bytes");
    result.length_bytes =
        U64(Field(value, "length_bytes", path), path + ".length_bytes");
    (void)CheckedEnd(result.offset_bytes, result.length_bytes, path);
    result.blob_ref = String(Field(value, "blob_ref", path), path + ".blob_ref");
    result.purpose =
        ParsePurpose(Field(value, "purpose", path), path + ".purpose");
    if (std::holds_alternative<HbmTarget>(result.target) !=
        (result.purpose == Purpose::STATE))
        Fail(path + ".purpose",
             "STATE purpose must exactly accompany an HBM target");
    const std::string expected = StableId(
        "program_sram_initialization", kInitializationSchema,
        Without(value, {"id"}));
    if (result.id != expected)
        Fail(path + ".id", "unstable artifact id; expected '" + expected + "'");
    return result;
}

OutputProbe ParseProbe(const Json &value, const std::string &path) {
    ExactObject(value, path,
                {"id", "target", "offset_bytes", "length_bytes",
                 "blob_ref", "comparison", "capture"});
    OutputProbe result;
    result.id = String(Field(value, "id", path), path + ".id");
    result.target =
        ParseTarget(Field(value, "target", path), path + ".target");
    result.offset_bytes =
        U64(Field(value, "offset_bytes", path), path + ".offset_bytes");
    result.length_bytes =
        U64(Field(value, "length_bytes", path), path + ".length_bytes");
    (void)CheckedEnd(result.offset_bytes, result.length_bytes, path);
    result.blob_ref = String(Field(value, "blob_ref", path), path + ".blob_ref");
    if (String(Field(value, "comparison", path), path + ".comparison") !=
        "exact_bytes/v1")
        Fail(path + ".comparison", "MVP only supports exact_bytes/v1");
    if (String(Field(value, "capture", path), path + ".capture") !=
        "after_program/v1")
        Fail(path + ".capture", "MVP only supports after_program/v1");
    const std::string expected = StableId(
        "program_output_probe", kProbeSchema, Without(value, {"id"}));
    if (result.id != expected)
        Fail(path + ".id", "unstable artifact id; expected '" + expected + "'");
    return result;
}

template <typename Entry>
auto EntryKey(const Entry &entry) {
    if (const auto *sram = std::get_if<SramTarget>(&entry.target))
        return std::make_tuple(
            uint8_t{0}, sram->runtime_core_id,
            sram->finalized_symbol_index, sram->program_symbol_ref,
            entry.offset_bytes, entry.length_bytes, entry.id);
    const HbmTarget &hbm = std::get<HbmTarget>(entry.target);
    return std::make_tuple(
        uint8_t{1}, uint32_t{0}, hbm.finalized_symbol_index,
        hbm.program_symbol_ref, entry.offset_bytes, entry.length_bytes,
        entry.id);
}

template <typename Entry>
auto EntryIdentity(const Entry &entry) {
    if (const auto *sram = std::get_if<SramTarget>(&entry.target))
        return std::make_tuple(
            uint8_t{0}, sram->runtime_core_id, sram->program_symbol_ref,
            entry.offset_bytes, entry.length_bytes);
    const HbmTarget &hbm = std::get<HbmTarget>(entry.target);
    return std::make_tuple(
        uint8_t{1}, uint32_t{0}, hbm.program_symbol_ref,
        entry.offset_bytes, entry.length_bytes);
}

bool SameSlice(const TensorSliceDto &left, const TensorSliceDto &right) {
    return left.value_id == right.value_id && left.offset == right.offset &&
           left.shape == right.shape;
}

bool SameLogicalCore(const LogicalCoreDto &left,
                     const LogicalCoreDto &right) {
    return left == right;
}

bool SameBufferAbi(const BufferAbiDto &left, const BufferAbiDto &right) {
    return left.id == right.id && left.schedule_id == right.schedule_id &&
           left.binding_id == right.binding_id &&
           left.value_id == right.value_id &&
           SameLogicalCore(left.logical_core, right.logical_core) &&
           SameSlice(left.tensor_slice, right.tensor_slice) &&
           left.region_ref == right.region_ref &&
           left.region_offset_bytes == right.region_offset_bytes &&
           left.size_bytes == right.size_bytes &&
           left.alignment_bytes == right.alignment_bytes &&
           left.banks == right.banks && left.storage_id == right.storage_id &&
           left.alias_of == right.alias_of &&
           left.lifetime_start == right.lifetime_start &&
           left.lifetime_end_exclusive == right.lifetime_end_exclusive &&
           left.dtype == right.dtype && left.layout == right.layout &&
           left.ownership == right.ownership;
}

bool SameStateAbi(const StateAbiDto &left, const StateAbiDto &right) {
    return left.id == right.id && left.state_ref == right.state_ref &&
           left.hbm_binding_ref == right.hbm_binding_ref &&
           left.kind == right.kind && left.lifetime == right.lifetime &&
           left.access == right.access && left.shape == right.shape &&
           left.dtype == right.dtype && left.layout == right.layout &&
           left.die_id == right.die_id && left.address == right.address &&
           left.size_bytes == right.size_bytes &&

           left.alignment_bytes == right.alignment_bytes;
}
const CommandFragmentDto &Leaf(const LinkedFragmentDto &linked) {
    if (const auto *fragment = std::get_if<CommandFragmentDto>(&linked))
        return *fragment;
    return std::get<RegionManifestDto>(linked).fragment;
}

uint64_t LiteralU64(const RelocatableRecordDto &record, const char *name,
                    const std::string &path) {
    const auto found = std::find_if(
        record.operands.begin(), record.operands.end(),
        [name](const RecordOperandDto &operand) { return operand.name == name; });
    if (found == record.operands.end() ||
        !std::holds_alternative<uint64_t>(found->literal_value))
        Fail(path + "." + name, "must be a literal uint64");
    return std::get<uint64_t>(found->literal_value);
}

struct Allocation {
    uint32_t runtime_core_id = 0;
    std::size_t label_definition_index = 0;
    const ProgramSymbolDefinitionDto *label_definition = nullptr;
    const ProgramSymbolDefinitionDto *region_definition = nullptr;
    const BufferAbiDto *abi = nullptr;
    uint64_t region_offset_bytes = 0;
    uint64_t size_bytes = 0;
    uint64_t absolute_start = 0;
};

using AllocationKey = std::pair<uint32_t, std::string>;

struct ManifestClosure {
    std::map<AllocationKey, Allocation> allocations;
    std::map<std::string, const BufferAbiDto *> abis;
    std::map<std::string, const StateAbiDto *> state_abis;
    std::map<std::string, std::set<Opcode>> state_directions;
};

ManifestClosure BuildManifestClosure(const LinkedProgramManifestDto &manifest,
                                     const ProgramArtifact &artifact) {
    ManifestClosure result;
    std::map<std::string, const StateAbiDto *> state_by_binding;
    for (const LinkedFragmentDto &linked : manifest.fragments) {
        for (const BufferAbiDto &abi : Leaf(linked).buffer_abi) {
            const auto inserted = result.abis.emplace(abi.id, &abi);
            if (!inserted.second && !SameBufferAbi(*inserted.first->second, abi))
                Fail("linked_program_manifest.fragments",
                     "conflicting shared BufferABI definition");
        }
        for (const StateAbiDto &abi : Leaf(linked).state_abi) {
            const auto inserted = result.state_abis.emplace(abi.id, &abi);
            if (!inserted.second &&
                !SameStateAbi(*inserted.first->second, abi))
                Fail("linked_program_manifest.fragments",
                     "conflicting shared StateABI definition");
            const auto by_binding =
                state_by_binding.emplace(abi.hbm_binding_ref, &abi);
            if (!by_binding.second &&
                !SameStateAbi(*by_binding.first->second, abi))
                Fail("linked_program_manifest.fragments",
                     "conflicting StateABI for one HBM binding");
            result.state_directions.try_emplace(abi.id);
        }
    }

    std::map<LogicalCoreDto, uint32_t> runtime_by_core;
    for (const CoreRuntimeBindingDto &binding : manifest.core_bindings) {
        if (binding.runtime_core_id > 0xffff)
            Fail("linked_program_manifest.core_bindings",
                 "runtime core does not fit uint16");
        runtime_by_core.emplace(binding.logical_core,
                                static_cast<uint32_t>(binding.runtime_core_id));
    }
    std::map<std::string, std::pair<std::size_t,
                                   const ProgramSymbolDefinitionDto *>> definitions;
    for (std::size_t index = 0;
         index < manifest.program_symbol_definitions.size(); ++index) {
        const ProgramSymbolDefinitionDto &definition =
            manifest.program_symbol_definitions[index];
        definitions.emplace(definition.symbol.id,
                            std::make_pair(index, &definition));
        if (index >= artifact.symbols.size())
            Fail("program_artifact.symbols", "missing manifest symbol");
        const ProgramSymbol &symbol = artifact.symbols[index];
        if (symbol.kind != definition.symbol.kind || symbol.value != definition.value ||
            symbol.size_bytes != definition.size_bytes ||
            symbol.name_string_index >= artifact.strings.size() ||
            artifact.strings[symbol.name_string_index] != definition.name)
            Fail("program_artifact.symbols",
                 "symbol/index/name/value does not match linked manifest");
    }
    if (artifact.symbols.size() != definitions.size())
        Fail("program_artifact.symbols",
             "symbol count does not match linked manifest definitions");

    using BindingKey =
        std::tuple<std::string, LogicalCoreDto, uint64_t, SemanticOperandId>;
    std::map<BindingKey, const AddressOperandBindingDto *> address_bindings;
    for (const AddressOperandBindingDto &binding :
         manifest.address_operand_bindings)
        address_bindings.emplace(
            std::make_tuple(binding.fragment_id, binding.logical_core,
                            binding.fragment_record_index, binding.operand_id),
            &binding);

    using StateRecordKey =
        std::tuple<std::string, LogicalCoreDto, uint64_t>;
    std::map<StateRecordKey, const RelocatableRecordDto *> records;
    std::set<std::string> allocated_abi_ids;
    for (const LinkedFragmentDto &linked : manifest.fragments) {
        const CommandFragmentDto &fragment = Leaf(linked);
        for (const CoreFragmentStreamDto &stream : fragment.core_streams) {
            const auto runtime = runtime_by_core.find(stream.logical_core);
            if (runtime == runtime_by_core.end())
                Fail("linked_program_manifest.core_bindings",
                     "allocation stream core has no runtime binding");
            for (std::size_t record_index = 0;
                 record_index < stream.records.size(); ++record_index) {
                const RelocatableRecordDto &record = stream.records[record_index];
                if (!records.emplace(
                        std::make_tuple(fragment.id, stream.logical_core,
                                        record_index),
                        &record).second)
                    Fail("linked_program_manifest.fragments",
                         "duplicate fragment/core/record identity");
                if (record.opcode != Opcode::SRAM_ALLOC_AT) continue;
                std::map<SemanticOperandId, const AddressRelocationDto *> relocations;
                for (const AddressRelocationDto &relocation :
                     stream.address_relocations)
                    if (relocation.record_index == record_index)
                        relocations.emplace(relocation.operand_id, &relocation);
                if (relocations.size() != 2 ||
                    relocations.count(SemanticOperandId::REGION_NAME) != 1 ||
                    relocations.count(SemanticOperandId::LABEL_SYMBOL) != 1)
                    Fail("linked_program_manifest.fragments",
                         "SRAM_ALLOC_AT requires exact region and label relocations");
                const std::string &region_ref =
                    relocations.at(SemanticOperandId::REGION_NAME)->symbol_ref;
                const std::string &label_ref =
                    relocations.at(SemanticOperandId::LABEL_SYMBOL)->symbol_ref;
                const auto region_entry = definitions.find(region_ref);
                const auto label_entry = definitions.find(label_ref);
                if (region_entry == definitions.end() ||
                    label_entry == definitions.end())
                    Fail("linked_program_manifest.program_symbol_definitions",
                         "SRAM_ALLOC_AT references an undefined symbol");
                const BindingKey label_key = std::make_tuple(
                    fragment.id, stream.logical_core, record_index,
                    SemanticOperandId::LABEL_SYMBOL);
                const BindingKey region_key = std::make_tuple(
                    fragment.id, stream.logical_core, record_index,
                    SemanticOperandId::REGION_NAME);
                const auto label_closure = address_bindings.find(label_key);
                const auto region_closure = address_bindings.find(region_key);
                if (label_closure == address_bindings.end() ||
                    label_closure->second->buffer_abi_ids.size() != 1 ||
                    label_closure->second->tensor_slices.size() != 1)
                    Fail("linked_program_manifest.address_operand_bindings",
                         "SRAM_ALLOC_AT label requires one exact BufferABI closure");
                const std::string &abi_id =
                    label_closure->second->buffer_abi_ids.front();
                const auto abi_entry = result.abis.find(abi_id);
                if (abi_entry == result.abis.end() ||
                    region_closure == address_bindings.end() ||
                    region_closure->second->buffer_abi_ids !=
                        std::vector<std::string>{abi_id} ||
                    region_closure->second->tensor_slices.size() != 1 ||
                    !SameSlice(label_closure->second->tensor_slices.front(),
                               abi_entry->second->tensor_slice) ||
                    !SameSlice(region_closure->second->tensor_slices.front(),
                               abi_entry->second->tensor_slice))
                    Fail("linked_program_manifest.address_operand_bindings",
                         "SRAM_ALLOC_AT region/label root BufferABI closures disagree");
                const BufferAbiDto &abi = *abi_entry->second;
                const ProgramSymbolDefinitionDto &region =
                    *region_entry->second.second;
                const ProgramSymbolDefinitionDto &label =
                    *label_entry->second.second;
                const uint64_t offset = LiteralU64(
                    record, "region_offset_bytes",
                    "linked_program_manifest.fragments.SRAM_ALLOC_AT");
                const uint64_t size = LiteralU64(
                    record, "size_bytes",
                    "linked_program_manifest.fragments.SRAM_ALLOC_AT");
                if (label.symbol.kind != ProgramSymbolKind::SRAM_LABEL ||
                    label.symbol.source_ref != abi.storage_id ||
                    region.symbol.kind != ProgramSymbolKind::SRAM_REGION ||
                    region.symbol.source_ref != abi.region_ref ||
                    !(abi.logical_core == stream.logical_core) ||
                    offset != abi.region_offset_bytes || size != abi.size_bytes ||
                    offset > region.size_bytes ||
                    size > region.size_bytes - offset ||
                    region.value >
                        std::numeric_limits<uint64_t>::max() -
                            region.size_bytes ||
                    region.value > std::numeric_limits<uint64_t>::max() - offset)
                    Fail("linked_program_manifest.fragments",
                         "SRAM_ALLOC_AT does not exactly preserve symbol, core, BufferABI and region span");
                const AllocationKey key{runtime->second, label.symbol.id};
                if (!result.allocations.emplace(
                        key,
                        Allocation{runtime->second, label_entry->second.first,
                                   &label, &region, &abi, offset, size,
                                   region.value + offset})
                         .second ||
                    !allocated_abi_ids.insert(abi.id).second)
                    Fail("linked_program_manifest.fragments",
                         "each core/label and BufferABI requires exactly one SRAM_ALLOC_AT");
            }
        }
    }

    std::set<std::string> storage_ids;
    for (const auto &entry : result.abis)
        storage_ids.insert(entry.second->storage_id);
    std::set<AllocationKey> expected;
    for (const ProgramSymbolDefinitionDto &definition :
         manifest.program_symbol_definitions) {
        if (definition.symbol.kind != ProgramSymbolKind::SRAM_LABEL ||
            storage_ids.count(definition.symbol.source_ref) == 0)
            continue;
        for (const LogicalCoreDto &core : definition.logical_cores) {
            const auto runtime = runtime_by_core.find(core);
            if (runtime == runtime_by_core.end())
                Fail("linked_program_manifest.core_bindings",
                     "SRAM label core has no runtime binding");
            expected.emplace(runtime->second, definition.symbol.id);
        }
    }
    std::set<AllocationKey> actual;
    for (const auto &entry : result.allocations) actual.insert(entry.first);
    if (actual != expected)
        Fail("linked_program_manifest.fragments",
             "storage-backed SRAM_LABEL definitions require exact per-core SRAM_ALLOC_AT coverage");
    std::set<std::string> all_abi_ids;
    for (const auto &entry : result.abis)
        if (entry.second->ownership != BufferOwnershipDto::ALIASED)
            all_abi_ids.insert(entry.first);
    if (allocated_abi_ids != all_abi_ids)
        Fail("linked_program_manifest.fragments",
             "every non-aliased BufferABI requires one exact SRAM_ALLOC_AT");
    for (const StateOperandBindingDto &binding :
         manifest.state_operand_bindings) {
        if (binding.operand_id != SemanticOperandId::HBM_ADDRESS)
            Fail("linked_program_manifest.state_operand_bindings",
                 "state closure only supports HBM_ADDRESS");
        const auto abi = result.state_abis.find(binding.state_abi_id);
        const auto record = records.find(std::make_tuple(
            binding.fragment_id, binding.logical_core,
            binding.fragment_record_index));
        if (abi == result.state_abis.end() || record == records.end())
            Fail("linked_program_manifest.state_operand_bindings",
                 "state binding does not resolve to exact StateABI and record");
        if (binding.logical_core.die_id != abi->second->die_id)
            Fail("linked_program_manifest.state_operand_bindings",
                 "state binding die differs from StateABI");
        if (record->second->opcode != Opcode::LSU_LOAD &&
            record->second->opcode != Opcode::LSU_STORE)
            Fail("linked_program_manifest.state_operand_bindings",
                 "HBM state binding requires a blocking LSU record");
        result.state_directions.at(binding.state_abi_id)
            .insert(record->second->opcode);
    }
    for (const auto &entry : result.state_directions)
        if (entry.second.empty())
            Fail("linked_program_manifest.state_operand_bindings",
                 "every StateABI requires a blocking LSU witness");
    return result;
}

DType SidecarDType(BufferDTypeDto dtype) {
    switch (dtype) {
    case BufferDTypeDto::FP16: return DType::FP16;
    case BufferDTypeDto::FP32: return DType::FP32;
    case BufferDTypeDto::INT32: return DType::INT32;
    }
    throw Error("unknown BufferABI dtype");
}

uint64_t DTypeBytes(DType dtype) {
    switch (dtype) {
    case DType::FP16: return 2;
    case DType::FP32:
    case DType::INT32: return 4;
    }
    throw Error("unknown ProgramIo SRAM dtype");
}

uint64_t TightTensorBytes(const TensorSlice &slice, DType dtype,
                          const std::string &path) {
    uint64_t elements = 1;
    for (uint64_t extent : slice.shape) {
        if (extent == 0 ||
            elements > std::numeric_limits<uint64_t>::max() / extent)
            Fail(path + ".shape", "tensor element count overflows uint64");
        elements *= extent;
    }
    const uint64_t element_bytes = DTypeBytes(dtype);
    if (elements > std::numeric_limits<uint64_t>::max() / element_bytes)
        Fail(path, "tensor byte span overflows uint64");
    return elements * element_bytes;
}

bool SameSidecarSlice(const TensorSlice &left, const TensorSliceDto &right) {
    return left.value_id == right.value_id && left.offset == right.offset &&
           left.shape == right.shape;
}

template <typename Entry>
const SramTarget &RequireSramTarget(const Entry &entry,
                                    const std::string &path) {
    const auto *target = std::get_if<SramTarget>(&entry.target);
    if (target == nullptr)
        Fail(path + ".target",
             "HBM target resolution requires the StateABI closure");
    return *target;
}

template <typename Entry>
const Allocation &ValidateEntry(const Entry &entry,
                                const ManifestClosure &closure,
                                const ProgramArtifact &artifact,
                                const std::string &path) {
    const SramTarget &target = RequireSramTarget(entry, path);
    const auto allocation = closure.allocations.find(
        {target.runtime_core_id, target.program_symbol_ref});
    const auto abi = closure.abis.find(target.buffer_abi_id);
    if (allocation == closure.allocations.end() || abi == closure.abis.end() ||
        allocation->second.abi != abi->second)
        Fail(path, "core/label does not resolve to the exact BufferABI allocation");
    const Allocation &result = allocation->second;
    const BufferAbiDto &buffer = *result.abi;
    if (target.finalized_symbol_index != result.label_definition_index ||
        target.expected_symbol_name != result.label_definition->name ||
        target.storage_id != buffer.storage_id ||
        target.value_id != buffer.value_id ||
        !SameSidecarSlice(target.tensor_slice, buffer.tensor_slice) ||
        target.dtype != SidecarDType(buffer.dtype) ||
        target.layout != buffer.layout)
        Fail(path,
             "symbol/index/name and BufferABI tensor metadata must match exactly");
    if (TightTensorBytes(target.tensor_slice, target.dtype,
                         path + ".target.tensor_slice") != buffer.size_bytes)
        Fail(path,
             "BufferABI size must equal its tight dense tensor byte span");
    if (target.finalized_symbol_index >= artifact.symbols.size())
        Fail(path + ".finalized_symbol_index", "exceeds ProgramArtifact symbols");
    const ProgramSymbol &symbol = artifact.symbols[target.finalized_symbol_index];
    if (symbol.kind != ProgramSymbolKind::SRAM_LABEL ||
        symbol.name_string_index >= artifact.strings.size() ||
        artifact.strings[symbol.name_string_index] != target.expected_symbol_name)
        Fail(path, "finalized ProgramArtifact symbol identity disagrees");
    const uint64_t end = CheckedEnd(entry.offset_bytes, entry.length_bytes, path);
    if (end > result.size_bytes)
        Fail(path, "byte range exceeds its SRAM_ALLOC_AT allocation");
    return result;
}

template <typename Entry>
const StateAbiDto &ValidateHbmEntry(
    const Entry &entry, const HbmTarget &target,
    const ManifestClosure &closure,
    const LinkedProgramManifestDto &manifest,
    const ProgramArtifact &artifact, const std::string &path) {
    const auto abi = closure.state_abis.find(target.state_abi_id);
    std::size_t definition_index = 0;
    const ProgramSymbolDefinitionDto *definition = nullptr;
    for (std::size_t index = 0;
         index < manifest.program_symbol_definitions.size(); ++index) {
        const ProgramSymbolDefinitionDto &candidate =
            manifest.program_symbol_definitions[index];
        if (candidate.symbol.id == target.program_symbol_ref) {
            definition_index = index;
            definition = &candidate;
            break;
        }
    }
    if (abi == closure.state_abis.end() || definition == nullptr)
        Fail(path,
             "HBM target does not resolve to StateABI and final symbol");
    const StateAbiDto &state = *abi->second;
    if (target.state_ref != state.state_ref ||
        target.hbm_binding_ref != state.hbm_binding_ref ||
        definition_index != target.finalized_symbol_index ||
        definition->name != target.expected_symbol_name ||
        definition->symbol.kind !=
            ProgramSymbolKind::ABSOLUTE_ADDRESS ||
        definition->symbol.source_ref != state.hbm_binding_ref ||
        definition->value != state.address ||
        definition->size_bytes != state.size_bytes)
        Fail(path,
             "HBM target must exactly preserve StateABI and final symbol identity");
    if (definition_index >= artifact.symbols.size())
        Fail(path + ".target.finalized_symbol_index",
             "exceeds ProgramArtifact symbols");
    const ProgramSymbol &symbol = artifact.symbols[definition_index];
    if (symbol.kind != ProgramSymbolKind::ABSOLUTE_ADDRESS ||
        symbol.value != state.address ||
        symbol.size_bytes != state.size_bytes ||
        symbol.name_string_index >= artifact.strings.size() ||
        artifact.strings[symbol.name_string_index] !=
            target.expected_symbol_name)
        Fail(path, "finalized HBM symbol identity disagrees");
    if (entry.offset_bytes != 0 ||
        entry.length_bytes != state.size_bytes)
        Fail(path, "HBM ProgramIo requires one whole-state byte range");
    return state;
}

template <typename Range>
void ValidateNonoverlap(std::vector<Range> spans, const std::string &path) {
    std::sort(spans.begin(), spans.end());
    for (std::size_t index = 1; index < spans.size(); ++index) {
        const Range &previous = spans[index - 1];
        const Range &current = spans[index];
        if (std::get<0>(previous) == std::get<0>(current) &&
            std::get<1>(previous) == std::get<1>(current) &&
            std::get<2>(current) < std::get<3>(previous))
            Fail(path, "physical byte ranges must not overlap");
    }
}

sram::AccessUnit &AccessFor(const Bindings &bindings, uint32_t core) {
    const auto found = bindings.sram_by_runtime_core.find(core);
    if (found == bindings.sram_by_runtime_core.end() || found->second == nullptr)
        throw Error("ProgramIo has no SRAM AccessUnit for runtime core " +
                    std::to_string(core));
    return *found->second;
}

template <typename ResolvedEntry>
const SramTarget &ValidateSramRuntimeRange(
    const ResolvedEntry &entry, const Bindings &bindings) {
    const SramTarget &target =
        RequireSramTarget(entry.source, "resolved ProgramIo entry");
    if (entry.hbm_range.has_value())
        throw Error("ProgramIo SRAM entry unexpectedly carries an HBM range");
    sram::AccessUnit &access = AccessFor(bindings, target.runtime_core_id);
    const sram::RegionConfig &region =
        access.regions().Region(entry.region_name);
    if (region.base_bytes != entry.region_base_bytes ||
        region.size_bytes != entry.region_size_bytes)
        throw Error("ProgramIo linked-manifest SRAM region disagrees with runtime hardware");
    const sram::ResolvedRange located = access.regions().LocateAbsolute(
        entry.absolute_address_bytes, entry.source.length_bytes);
    if (located.address != entry.absolute_address_bytes ||
        located.size_bytes != entry.source.length_bytes ||
        access.regions().Region(located.region_id).name != entry.region_name)
        throw Error("ProgramIo resolved range is not contained by its exact runtime region");
    return target;
}

template <typename ResolvedEntry>
HBMRuntimeDebugSnapshot PeekHbmRuntimeRange(
    const ResolvedEntry &entry, const Bindings &bindings) {
    const auto *target = std::get_if<HbmTarget>(&entry.source.target);
    if (target == nullptr || !entry.hbm_range.has_value())
        throw Error("ProgramIo resolved HBM entry lacks its exact range");
    const ResolvedHbmRange &range = *entry.hbm_range;
    if (entry.region_symbol_ref != target->program_symbol_ref ||
        entry.region_name != target->expected_symbol_name ||
        entry.region_base_bytes != range.address_bytes ||
        entry.region_size_bytes != range.size_bytes ||
        entry.absolute_address_bytes != range.address_bytes ||
        entry.source.offset_bytes != 0 ||
        entry.source.length_bytes != range.size_bytes ||
        range.die_id >
            static_cast<uint64_t>(std::numeric_limits<int>::max()))
        throw Error("ProgramIo resolved HBM range identity changed");
    if (bindings.hbm_runtime == nullptr)
        throw Error("ProgramIo HBM target requires an HBMRuntime binding");
    HBMRuntimeDebugSnapshot snapshot = bindings.hbm_runtime->DebugPeek(
        range.address_bytes, static_cast<int>(range.die_id),
        range.size_bytes);
    if (snapshot.physical_address != range.address_bytes ||
        snapshot.current_die != static_cast<int>(range.die_id) ||
        snapshot.payload.size() != range.size_bytes ||
        snapshot.chunks.empty())
        throw Error("ProgramIo HBMRuntime returned a malformed snapshot");
    uint64_t cursor = 0;
    for (const HBMRuntimeDebugChunkSnapshot &chunk : snapshot.chunks) {
        if (chunk.physical_offset != cursor ||
            chunk.backend.payload.empty() ||
            chunk.backend.payload.size() != chunk.backend.present.size() ||
            chunk.backend.payload.size() > snapshot.payload.size() - cursor ||
            !std::equal(
                chunk.backend.payload.begin(), chunk.backend.payload.end(),
                snapshot.payload.begin() + static_cast<std::size_t>(cursor)))
            throw Error("ProgramIo HBMRuntime snapshot chunks are malformed");
        cursor += chunk.backend.payload.size();
    }
    if (cursor != snapshot.payload.size())
        throw Error("ProgramIo HBMRuntime snapshot does not cover its payload");
    return snapshot;
}

void RequireBoundary(const char *operation) {
    if (sc_core::sc_is_running())
        throw Error(std::string(operation) +
                    " is forbidden while simulation is running");
}

} // namespace

std::string Sha256Hex(std::string_view bytes) { return Sha256Impl(bytes); }

std::string Sha256Hex(const std::vector<uint8_t> &bytes) {
    if (bytes.empty()) return Sha256Impl(std::string_view{});
    return Sha256Impl(std::string_view(
        reinterpret_cast<const char *>(bytes.data()), bytes.size()));
}

Contract Parse(std::string_view sidecar_json) {
    const Json value = ParseStrictJson(sidecar_json, "ProgramIo sidecar");
    const std::string path = "program_io_contract";
    ExactObject(value, path,
                {"schema_version", "producer_pass", "id", "mode",
                 "source_linked_manifest_id", "source_linked_manifest_digest",
                 "program_artifact_sha256", "blobs", "initializations",
                 "output_probes"});
    Contract result;
    result.schema_version =
        String(Field(value, "schema_version", path), path + ".schema_version");
    if (result.schema_version != kSchemaVersion)
        Fail(path + ".schema_version", "unsupported schema version");
    result.producer_pass =
        String(Field(value, "producer_pass", path), path + ".producer_pass");
    result.id = String(Field(value, "id", path), path + ".id");
    result.mode = ParseMode(Field(value, "mode", path), path + ".mode");
    result.source_linked_manifest_id = String(
        Field(value, "source_linked_manifest_id", path),
        path + ".source_linked_manifest_id");
    result.source_linked_manifest_digest = String(
        Field(value, "source_linked_manifest_digest", path),
        path + ".source_linked_manifest_digest");
    result.program_artifact_sha256 = String(
        Field(value, "program_artifact_sha256", path),
        path + ".program_artifact_sha256");
    RequireSha256(result.source_linked_manifest_digest,
                  path + ".source_linked_manifest_digest");
    RequireSha256(result.program_artifact_sha256,
                  path + ".program_artifact_sha256");

    const Json &blobs = Field(value, "blobs", path);
    const Json &initializations = Field(value, "initializations", path);
    const Json &probes = Field(value, "output_probes", path);
    if (!blobs.is_array() || !initializations.is_array() || !probes.is_array())
        Fail(path, "blobs, initializations, and output_probes must be arrays");
    if (blobs.empty() || initializations.empty() || probes.empty())
        Fail(path, "MVP contract requires blobs, initializations and output probes");
    for (std::size_t index = 0; index < blobs.size(); ++index)
        result.blobs.push_back(ParseBlob(
            blobs[index], path + ".blobs[" + std::to_string(index) + "]"));
    for (std::size_t index = 0; index < initializations.size(); ++index)
        result.initializations.push_back(ParseInitialization(
            initializations[index], path + ".initializations[" +
                                        std::to_string(index) + "]"));
    for (std::size_t index = 0; index < probes.size(); ++index)
        result.output_probes.push_back(ParseProbe(
            probes[index], path + ".output_probes[" +
                               std::to_string(index) + "]"));

    for (std::size_t index = 1; index < result.blobs.size(); ++index)
        if (result.blobs[index - 1].id >= result.blobs[index].id)
            Fail(path + ".blobs", "blobs must have unique canonical ids");
    for (std::size_t index = 1; index < result.initializations.size(); ++index)
        if (!(EntryKey(result.initializations[index - 1]) <
              EntryKey(result.initializations[index])))
            Fail(path + ".initializations",
                 "initializations must be unique and canonical");
    for (std::size_t index = 1; index < result.output_probes.size(); ++index)
        if (!(EntryKey(result.output_probes[index - 1]) <
              EntryKey(result.output_probes[index])))
            Fail(path + ".output_probes",
                 "output probes must be unique and canonical");

    std::map<std::string, const Blob *> blob_by_id;
    for (const Blob &blob : result.blobs) blob_by_id.emplace(blob.id, &blob);
    std::set<std::string> used_blobs;
    std::set<std::tuple<uint8_t, uint32_t, std::string, uint64_t, uint64_t>> init_ids;
    for (const Initialization &entry : result.initializations) {
        if (!init_ids.emplace(EntryIdentity(entry)).second)
            Fail(path + ".initializations", "duplicate initialization byte range");
        const auto blob = blob_by_id.find(entry.blob_ref);
        if (blob == blob_by_id.end() ||
            blob->second->bytes.size() != entry.length_bytes)
            Fail(path + ".initializations",
                 "blob must exist and exactly match initialization length");
        used_blobs.insert(entry.blob_ref);
        if (result.mode == Mode::FUNCTIONAL &&
            entry.purpose == Purpose::TIMING_PARTIAL)
            Fail(path + ".initializations",
                 "functional mode forbids timing_partial initialization");
    }
    std::set<std::tuple<uint8_t, uint32_t, std::string, uint64_t, uint64_t>> probe_ids;
    for (const OutputProbe &entry : result.output_probes) {
        if (!probe_ids.emplace(EntryIdentity(entry)).second)
            Fail(path + ".output_probes", "duplicate output-probe byte range");
        const auto blob = blob_by_id.find(entry.blob_ref);
        if (blob == blob_by_id.end() ||
            blob->second->bytes.size() != entry.length_bytes)
            Fail(path + ".output_probes",
                 "blob must exist and exactly match output-probe length");
        used_blobs.insert(entry.blob_ref);
    }
    if (used_blobs.size() != blob_by_id.size())
        Fail(path + ".blobs",
             "blobs must be used exactly by initializations/probes");
    const std::string expected = StableId(
        "program_io_contract", kSchemaVersion,
        Without(value, {"schema_version", "producer_pass", "id"}));
    if (result.id != expected)
        Fail(path + ".id", "unstable artifact id; expected '" + expected + "'");
    return result;
}

Contract Load(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw Error("cannot open ProgramIo sidecar: " + path.string());
    std::ostringstream bytes;
    bytes << input.rdbuf();
    if (input.bad()) throw Error("failed while reading ProgramIo sidecar");
    return Parse(bytes.str());
}

ResolvedContract Resolve(const Contract &contract,
                         std::string_view linked_manifest_json,
                         const std::vector<uint8_t> &artifact_bytes) {
    try {
        const Json manifest_json =
            ParseStrictJson(linked_manifest_json, "linked manifest");
        const std::string manifest_digest = CanonicalDigest(manifest_json);
        const LinkedProgramManifestDto manifest =
            ProgramArtifactFinalizer::Parse(linked_manifest_json);
        if (contract.source_linked_manifest_id != manifest.id ||
            contract.source_linked_manifest_digest != manifest_digest)
            Fail("program_io_contract.source_linked_manifest_id",
                 "source id/digest do not identify the exact linked manifest");
        const std::string actual_artifact_sha = Sha256Hex(artifact_bytes);
        if (contract.program_artifact_sha256 != actual_artifact_sha)
            Fail("program_io_contract.program_artifact_sha256",
                 "does not match the actual encoded ProgramArtifact bytes");
        const ProgramArtifact artifact = DecodeProgramArtifact(artifact_bytes);
        const ProgramArtifact expected_artifact =
            ProgramArtifactFinalizer{}.Finalize(manifest);
        if (EncodeProgramArtifact(expected_artifact) != artifact_bytes)
            Fail("program_artifact",
                 "encoded bytes are not the exact finalization of the linked manifest");
        const ManifestClosure closure = BuildManifestClosure(manifest, artifact);
        std::map<std::string, const Blob *> blobs;
        for (const Blob &blob : contract.blobs) blobs.emplace(blob.id, &blob);

        ResolvedContract result;
        result.id = contract.id;
        result.mode = contract.mode;
        result.source_linked_manifest_id = contract.source_linked_manifest_id;
        result.source_linked_manifest_digest =
            contract.source_linked_manifest_digest;
        result.program_artifact_sha256 = contract.program_artifact_sha256;

        using Span =
            std::tuple<uint32_t, std::string, uint64_t, uint64_t, std::string>;
        std::vector<Span> initialization_ranges;
        std::vector<Span> probe_ranges;
        std::map<std::string, std::vector<std::pair<uint64_t, uint64_t>>> coverage;
        for (std::size_t index = 0; index < contract.initializations.size(); ++index) {
            const Initialization &entry = contract.initializations[index];
            const std::string path = "program_io_contract.initializations[" +
                                     std::to_string(index) + "]";
            if (const auto *hbm =
                    std::get_if<HbmTarget>(&entry.target)) {
                const StateAbiDto &state = ValidateHbmEntry(
                    entry, *hbm, closure, manifest, artifact, path);
                const auto directions =
                    closure.state_directions.find(state.id);
                if (state.access == StateAccessDto::RESERVED ||
                    directions == closure.state_directions.end() ||
                    directions->second.count(Opcode::LSU_LOAD) == 0)
                    Fail(path,
                         "HBM seed requires a readable state with an LSU_LOAD");
                result.initializations.push_back(
                    {entry,
                     hbm->program_symbol_ref,
                     hbm->expected_symbol_name,
                     state.address,
                     state.size_bytes,
                     state.address,
                     blobs.at(entry.blob_ref)->bytes,
                     ResolvedHbmRange{state.die_id, state.address,
                                      state.size_bytes}});
                continue;
            }
            const SramTarget &target = RequireSramTarget(entry, path);
            const Allocation &allocation =
                ValidateEntry(entry, closure, artifact, path);
            const BufferAbiDto &abi = *allocation.abi;
            if ((entry.purpose == Purpose::ACTIVATION ||
                 entry.purpose == Purpose::WEIGHT) &&
                abi.ownership != BufferOwnershipDto::BORROWED)
                Fail(path + ".purpose",
                     "activation/weight initialization requires BORROWED BufferABI");
            if (entry.purpose == Purpose::TIMING_PARTIAL &&
                contract.mode != Mode::TIMING)
                Fail(path + ".purpose",
                     "timing_partial is only legal in timing mode");
            if (entry.purpose == Purpose::TIMING_PARTIAL &&
                abi.ownership != BufferOwnershipDto::OWNED)
                Fail(path + ".purpose",
                     "timing_partial initialization requires OWNED BufferABI");
            if (contract.mode == Mode::FUNCTIONAL &&
                abi.ownership != BufferOwnershipDto::BORROWED)
                Fail(path + ".buffer_abi_id",
                     "functional mode forbids initialization of OWNED storage");
            const uint64_t absolute =
                allocation.absolute_start + entry.offset_bytes;
            const uint64_t end = absolute + entry.length_bytes;
            initialization_ranges.emplace_back(
                target.runtime_core_id, allocation.region_definition->symbol.id,
                absolute, end, entry.id);
            coverage[abi.id].emplace_back(
                entry.offset_bytes, entry.offset_bytes + entry.length_bytes);
            result.initializations.push_back(
                {entry,
                 allocation.region_definition->symbol.id,
                 allocation.region_definition->name,
                 allocation.region_definition->value,
                 allocation.region_definition->size_bytes,
                 absolute,
                 blobs.at(entry.blob_ref)->bytes,
                 std::nullopt});
        }
        for (const auto &abi_entry : closure.abis) {
            const BufferAbiDto &abi = *abi_entry.second;
            if (abi.ownership != BufferOwnershipDto::BORROWED) continue;
            std::vector<std::pair<uint64_t, uint64_t>> intervals =
                coverage[abi.id];
            std::sort(intervals.begin(), intervals.end());
            uint64_t cursor = 0;
            for (const auto &interval : intervals) {
                if (interval.first > cursor) break;
                cursor = std::max(cursor, interval.second);
            }
            if (cursor != abi.size_bytes)
                Fail("program_io_contract.initializations",
                     "every manifested BORROWED BufferABI requires complete host initialization");
        }

        for (std::size_t index = 0; index < contract.output_probes.size(); ++index) {
            const OutputProbe &entry = contract.output_probes[index];
            const std::string path = "program_io_contract.output_probes[" +
                                     std::to_string(index) + "]";
            if (const auto *hbm =
                    std::get_if<HbmTarget>(&entry.target)) {
                const StateAbiDto &state = ValidateHbmEntry(
                    entry, *hbm, closure, manifest, artifact, path);
                const auto directions =
                    closure.state_directions.find(state.id);
                if (state.access != StateAccessDto::READ_WRITE ||
                    directions == closure.state_directions.end() ||
                    directions->second.count(Opcode::LSU_STORE) == 0)
                    Fail(path,
                         "HBM probe requires READ_WRITE state with an LSU_STORE");
                result.output_probes.push_back(
                    {entry,
                     hbm->program_symbol_ref,
                     hbm->expected_symbol_name,
                     state.address,
                     state.size_bytes,
                     state.address,
                     blobs.at(entry.blob_ref)->bytes,
                     ResolvedHbmRange{state.die_id, state.address,
                                      state.size_bytes}});
                continue;
            }
            const Allocation &allocation =
                ValidateEntry(entry, closure, artifact, path);
            const SramTarget &target = RequireSramTarget(entry, path);
            if (allocation.abi->ownership != BufferOwnershipDto::OWNED)
                Fail(path + ".buffer_abi_id",
                     "output probe requires OWNED BufferABI");
            const uint64_t absolute =
                allocation.absolute_start + entry.offset_bytes;
            const uint64_t end = absolute + entry.length_bytes;
            probe_ranges.emplace_back(
                target.runtime_core_id, allocation.region_definition->symbol.id,
                absolute, end, entry.id);
            result.output_probes.push_back(
                {entry,
                 allocation.region_definition->symbol.id,
                 allocation.region_definition->name,
                 allocation.region_definition->value,
                 allocation.region_definition->size_bytes,
                 absolute,
                 blobs.at(entry.blob_ref)->bytes,
                 std::nullopt});
            for (const auto &other_entry : closure.allocations) {
                const Allocation &other = other_entry.second;
                if (&other == &allocation ||
                    other.runtime_core_id != allocation.runtime_core_id ||
                    other.region_definition->symbol.id !=
                        allocation.region_definition->symbol.id)
                    continue;
                if (absolute < other.absolute_start + other.size_bytes &&
                    other.absolute_start < end)
                    Fail(path + ".capture",
                         "after-program probe rejects physical allocation reuse/aliasing");
            }
        }
        ValidateNonoverlap(initialization_ranges,
                           "program_io_contract.initializations");
        ValidateNonoverlap(probe_ranges,
                           "program_io_contract.output_probes");
        return result;
    } catch (const Error &) {
        throw;
    } catch (const std::exception &error) {
        throw Error(std::string("ProgramIo resolve failed: ") + error.what());
    }
}

ResolvedContract ParseAndResolve(
    std::string_view sidecar_json, std::string_view linked_manifest_json,
    const std::vector<uint8_t> &artifact_bytes) {
    return Resolve(Parse(sidecar_json), linked_manifest_json, artifact_bytes);
}

Applied ApplyBeforeSimulation(const ResolvedContract &contract,
                              const Bindings &bindings) {
    RequireBoundary("ProgramIo ApplyBeforeSimulation");
    if (contract.initializations.empty() || contract.output_probes.empty())
        throw Error("ProgramIo resolved contract requires initialization and probe entries");

    struct Rollback {
        TargetKind kind = TargetKind::SRAM;
        const ResolvedInitialization *entry = nullptr;
        sram::AccessUnit *access = nullptr;
        std::optional<sram::DebugSnapshot> sram;
        std::optional<HBMRuntimeDebugSnapshot> hbm;
    };
    std::vector<Rollback> rollbacks;
    rollbacks.reserve(contract.initializations.size());

    // Preflight every read/write route and snapshot every destination before
    // the first mutation, so a late missing/unsupported HBM backend cannot
    // leave earlier SRAM or HBM state changed.
    for (const ResolvedInitialization &entry : contract.initializations) {
        if (entry.bytes.size() != entry.source.length_bytes)
            throw Error("ProgramIo resolved initialization payload length changed");
        Rollback rollback;
        rollback.entry = &entry;
        if (std::holds_alternative<HbmTarget>(entry.source.target)) {
            rollback.kind = TargetKind::HBM;
            rollback.hbm = PeekHbmRuntimeRange(entry, bindings);
        } else {
            const SramTarget &target =
                ValidateSramRuntimeRange(entry, bindings);
            rollback.access = &AccessFor(bindings, target.runtime_core_id);
            rollback.sram = rollback.access->DebugPeek(
                entry.absolute_address_bytes, entry.source.length_bytes);
        }
        rollbacks.push_back(std::move(rollback));
    }
    for (const ResolvedOutputProbe &entry : contract.output_probes) {
        if (entry.expected_bytes.size() != entry.source.length_bytes)
            throw Error("ProgramIo resolved probe payload length changed");
        if (std::holds_alternative<HbmTarget>(entry.source.target))
            (void)PeekHbmRuntimeRange(entry, bindings);
        else
            (void)ValidateSramRuntimeRange(entry, bindings);
    }

    try {
        for (Rollback &rollback : rollbacks) {
            const ResolvedInitialization &entry = *rollback.entry;
            if (rollback.kind == TargetKind::SRAM) {
                rollback.access->DebugSeed(
                    entry.absolute_address_bytes, entry.bytes);
            } else {
                const ResolvedHbmRange &range = *entry.hbm_range;
                bindings.hbm_runtime->DebugSeed(
                    range.address_bytes, static_cast<int>(range.die_id),
                    entry.bytes);
            }
        }
    } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        try {
            for (auto it = rollbacks.rbegin();
                 it != rollbacks.rend(); ++it) {
                if (it->kind == TargetKind::SRAM)
                    it->access->DebugRestore(
                        it->entry->absolute_address_bytes, *it->sram);
                else
                    bindings.hbm_runtime->DebugRestore(*it->hbm);
            }
        } catch (const std::exception &rollback) {
            throw std::runtime_error(
                std::string("ProgramIo rollback failed: ") + rollback.what());
        }
        std::rethrow_exception(failure);
    }
    return Applied{contract, bindings};
}

bool Result::Passed() const noexcept {
    return !probes.empty() &&
           std::all_of(probes.begin(), probes.end(),
                       [](const ProbeResult &probe) {
                           return probe.exact_match && probe.all_bytes_valid;
                       });
}

Result VerifyAfterSimulation(const Applied &applied) {
    RequireBoundary("ProgramIo VerifyAfterSimulation");
    if (applied.contract.output_probes.empty())
        throw Error("ProgramIo applied contract has no output probes");
    Result result;
    result.contract_id = applied.contract.id;
    for (const ResolvedOutputProbe &probe : applied.contract.output_probes) {
        if (probe.expected_bytes.size() != probe.source.length_bytes)
            throw Error("ProgramIo resolved probe payload length changed");
        std::vector<uint8_t> actual;
        bool all_bytes_valid = false;
        uint32_t runtime_core_id = 0;
        if (std::holds_alternative<HbmTarget>(probe.source.target)) {
            const HBMRuntimeDebugSnapshot snapshot =
                PeekHbmRuntimeRange(probe, applied.bindings);
            actual = snapshot.payload;
            all_bytes_valid = std::all_of(
                snapshot.chunks.begin(), snapshot.chunks.end(),
                [](const HBMRuntimeDebugChunkSnapshot &chunk) {
                    return std::all_of(
                        chunk.backend.present.begin(),
                        chunk.backend.present.end(),
                        [](uint8_t byte) { return byte != 0; });
                });
        } else {
            const SramTarget &target =
                ValidateSramRuntimeRange(probe, applied.bindings);
            runtime_core_id = target.runtime_core_id;
            sram::AccessUnit &access =
                AccessFor(applied.bindings, target.runtime_core_id);
            const sram::DebugSnapshot snapshot = access.DebugPeek(
                probe.absolute_address_bytes, probe.source.length_bytes);
            actual = snapshot.payload;
            all_bytes_valid =
                snapshot.valid.size() == snapshot.payload.size() &&
                std::all_of(snapshot.valid.begin(), snapshot.valid.end(),
                            [](uint8_t byte) { return byte != 0; });
        }
        ProbeResult one;
        one.probe_id = probe.source.id;
        one.runtime_core_id = runtime_core_id;
        one.absolute_address_bytes = probe.absolute_address_bytes;
        one.length_bytes = probe.source.length_bytes;
        one.expected_sha256 = Sha256Hex(probe.expected_bytes);
        one.actual_sha256 = Sha256Hex(actual);
        one.exact_match = actual == probe.expected_bytes;
        one.all_bytes_valid = all_bytes_valid;
        result.probes.push_back(std::move(one));
    }
    return result;
}

} // namespace frontend::program_io
