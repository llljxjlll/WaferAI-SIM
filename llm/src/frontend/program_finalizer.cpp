#include "frontend/program_finalizer.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <initializer_list>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>

namespace frontend {
namespace {

using Json = nlohmann::json;

uint32_t RotateRight(uint32_t value, unsigned int shift) {
    return (value >> shift) | (value << (32 - shift));
}

std::string Sha256(std::string_view input) {
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
    std::vector<uint8_t> bytes(input.begin(), input.end());
    const uint64_t bit_length = static_cast<uint64_t>(bytes.size()) * 8;
    bytes.push_back(0x80);
    while (bytes.size() % 64 != 56)
        bytes.push_back(0);
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
    for (uint32_t word : state) {
        for (int shift = 28; shift >= 0; shift -= 4)
            result.push_back(hex[(word >> shift) & 0xf]);
    }
    return result;
}

std::string CanonicalDigest(const Json &value) {
    return Sha256(value.dump(-1, ' ', false,
                             Json::error_handler_t::strict));
}

Json Without(const Json &value,
             std::initializer_list<const char *> excluded) {
    Json result = value;
    for (const char *field : excluded)
        result.erase(field);
    return result;
}

std::string StableArtifactId(std::string_view kind,
                             std::string_view schema_version,
                             const Json &semantic_key) {
    const Json identity{{"kind", kind},
                        {"schema_version", schema_version},
                        {"semantic_key", semantic_key}};
    return std::string(kind) + "_" + CanonicalDigest(identity).substr(0, 16);
}

[[noreturn]] void Fail(const std::string &path, const std::string &message) {
    throw ProgramFinalizerError(path + ": " + message);
}

void ExactObject(const Json &value, const std::string &path,
                 std::initializer_list<const char *> fields) {
    if (!value.is_object())
        Fail(path, "must be an object");
    std::set<std::string> expected;
    for (const char *field : fields)
        expected.emplace(field);
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (expected.erase(it.key()) == 0)
            Fail(path + "." + it.key(), "unknown field");
    }
    if (!expected.empty())
        Fail(path + "." + *expected.begin(), "missing required field");
}

const Json &Field(const Json &value, const char *name) {
    return value.at(name);
}

uint64_t U64(const Json &value, const std::string &path) {
    if (value.is_number_unsigned())
        return value.get<uint64_t>();
    if (value.is_number_integer()) {
        const int64_t result = value.get<int64_t>();
        if (result >= 0)
            return static_cast<uint64_t>(result);
    }
    Fail(path, "must be a uint64 integer");
}

int64_t I64(const Json &value, const std::string &path) {
    if (value.type() == Json::value_t::number_integer)
        return value.get<int64_t>();
    if (value.type() == Json::value_t::number_unsigned) {
        const uint64_t result = value.get<uint64_t>();
        if (result <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            return static_cast<int64_t>(result);
    }
    Fail(path, "must be a signed int64 integer");
}

std::string String(const Json &value, const std::string &path,
                   bool allow_empty = false) {
    if (!value.is_string())
        Fail(path, "must be a string");
    std::string result = value.get<std::string>();
    if (!allow_empty && result.empty())
        Fail(path, "must be non-empty");
    return result;
}

bool Bool(const Json &value, const std::string &path) {
    if (!value.is_boolean())
        Fail(path, "must be a boolean");
    return value.get<bool>();
}

template <typename T, typename Parser>
std::vector<T> Array(const Json &value, const std::string &path,
                     Parser parser) {
    if (!value.is_array())
        Fail(path, "must be an array");
    std::vector<T> result;
    result.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index)
        result.push_back(parser(value[index], path + "[" +
                                           std::to_string(index) + "]"));
    return result;
}

std::vector<std::string> Strings(const Json &value,
                                 const std::string &path) {
    return Array<std::string>(value, path,
                              [](const Json &item, const std::string &item_path) {
                                  return String(item, item_path);
                              });
}

std::vector<uint64_t> U64s(const Json &value, const std::string &path) {
    return Array<uint64_t>(value, path,
                           [](const Json &item, const std::string &item_path) {
                               return U64(item, item_path);
                           });
}

std::optional<std::string> NullableString(const Json &value,
                                          const std::string &path) {
    if (value.is_null())
        return std::nullopt;
    return String(value, path);
}

LogicalCoreDto ParseCore(const Json &value, const std::string &path) {
    ExactObject(value, path, {"die_id", "local_core_id"});
    return {U64(Field(value, "die_id"), path + ".die_id"),
            U64(Field(value, "local_core_id"), path + ".local_core_id")};
}

FragmentKindDto ParseFragmentKind(const Json &value,
                                  const std::string &path) {
    const std::string raw = String(value, path);
    if (raw == "coarse")
        return FragmentKindDto::COARSE;
    if (raw == "standalone_collective")
        return FragmentKindDto::STANDALONE_COLLECTIVE;
    if (raw == "isa_region")
        return FragmentKindDto::ISA_REGION;
    if (raw == "state_io")
        return FragmentKindDto::STATE_IO;
    if (raw == "state_transfer")
        return FragmentKindDto::STATE_TRANSFER;
    if (raw == "moe_transfer")
        return FragmentKindDto::MOE_TRANSFER;
    if (raw == "s2_lite_rooted_ar")
        return FragmentKindDto::S2_LITE_ROOTED_AR;
    Fail(path, "unknown FragmentKind");
}

RuntimeSymbolKindDto ParseRuntimeSymbolKind(const Json &value,
                                            const std::string &path) {
    const std::string raw = String(value, path);
    if (raw == "start_tag")
        return RuntimeSymbolKindDto::START_TAG;
    if (raw == "event_tag")
        return RuntimeSymbolKindDto::EVENT_TAG;
    if (raw == "dte_token")
        return RuntimeSymbolKindDto::DTE_TOKEN;
    if (raw == "dte_fsm")
        return RuntimeSymbolKindDto::DTE_FSM;
    if (raw == "group")
        return RuntimeSymbolKindDto::GROUP;
    if (raw == "runtime_core")
        return RuntimeSymbolKindDto::RUNTIME_CORE;
    Fail(path, "unknown RuntimeSymbolKind");
}

RuntimeOperandFieldDto ParseRuntimeField(const Json &value,
                                         const std::string &path) {
    const std::string raw = String(value, path);
    if (raw == "start_tag")
        return RuntimeOperandFieldDto::START_TAG;
    if (raw == "event_tag")
        return RuntimeOperandFieldDto::EVENT_TAG;
    if (raw == "dte_token")
        return RuntimeOperandFieldDto::DTE_TOKEN;
    if (raw == "dte_fsm")
        return RuntimeOperandFieldDto::DTE_FSM;
    if (raw == "group_id")
        return RuntimeOperandFieldDto::GROUP_ID;
    if (raw == "source_core")
        return RuntimeOperandFieldDto::SOURCE_CORE;
    if (raw == "destination_core")
        return RuntimeOperandFieldDto::DESTINATION_CORE;
    if (raw == "peer_core")
        return RuntimeOperandFieldDto::PEER_CORE;
    Fail(path, "unknown RuntimeOperandField");
}

OperandKindDto ParseOperandKind(const Json &value, const std::string &path) {
    const std::string raw = String(value, path);
    if (raw == "literal")
        return OperandKindDto::LITERAL;
    if (raw == "runtime_symbol")
        return OperandKindDto::RUNTIME_SYMBOL;
    if (raw == "address_symbol")
        return OperandKindDto::ADDRESS_SYMBOL;
    Fail(path, "unknown OperandKind");
}

ProgramSymbolKind ParseProgramSymbolKind(const Json &value,
                                         const std::string &path) {
    switch (U64(value, path)) {
    case 1:
        return ProgramSymbolKind::ABSOLUTE_ADDRESS;
    case 2:
        return ProgramSymbolKind::SRAM_REGION;
    case 3:
        return ProgramSymbolKind::SRAM_LABEL;
    default:
        Fail(path, "unknown ProgramSymbolKind");
    }
}

SemanticOperandId ParseOperandId(const Json &value,
                                 const std::string &path) {
    const uint64_t raw = U64(value, path);
    if ((raw >= 1 && raw <= 12) || (raw >= 0x100 && raw <= 0x110))
        return static_cast<SemanticOperandId>(raw);
    Fail(path, "unknown SemanticOperandId");
}

Opcode ParseOpcode(const Json &value, const std::string &path) {
    const uint64_t raw = U64(value, path);
    switch (raw) {
    case 0x01:
    case 0x06:
    case 0x0c:
    case 0x0e:
    case 0x10:
    case 0x1a:
    case 0x1b:
    case 0x1c:
    case 0x1d:
    case 0x1e:
    case 0x1f:
    case 0x20:
    case 0x40:
    case 0x41:
    case 0x43:
    case 0x80:
    case 0x81:
    case 0x82:
    case 0x84:
    case 0x86:
    case 0x89:
    case 0xc0:
    case 0xc3:
    case 0xc4:
        return static_cast<Opcode>(raw);
    default:
        Fail(path, "opcode is outside the LinkedProgramManifest ABI");
    }
}

ManifestInputKindDto ParseInputKind(const Json &value,
                                    const std::string &path) {
    const std::string raw = String(value, path);
    if (raw == "s3_lite_moe") return ManifestInputKindDto::S3_LITE_MOE;
    if (raw == "s2_lite_rooted_ar")
        return ManifestInputKindDto::S2_LITE_ROOTED_AR;
    if (raw == "train_lowered_program")
        return ManifestInputKindDto::TRAIN_LOWERED_PROGRAM;
    if (raw == "ir1") return ManifestInputKindDto::IR1;
    if (raw == "fusion_plan") return ManifestInputKindDto::FUSION_PLAN;
    if (raw == "standalone_plan") return ManifestInputKindDto::STANDALONE_PLAN;
    if (raw == "ir2_projection") return ManifestInputKindDto::IR2_PROJECTION;
    if (raw == "schedule_set") return ManifestInputKindDto::SCHEDULE_SET;
    if (raw == "global_action_dag") return ManifestInputKindDto::GLOBAL_ACTION_DAG;
    if (raw == "command_fragment") return ManifestInputKindDto::COMMAND_FRAGMENT;
    if (raw == "region_manifest") return ManifestInputKindDto::REGION_MANIFEST;
    Fail(path, "unknown ManifestInputKind");
}

BufferDTypeDto ParseDType(const Json &value, const std::string &path) {
    const std::string raw = String(value, path);
    if (raw == "fp16") return BufferDTypeDto::FP16;
    if (raw == "fp32") return BufferDTypeDto::FP32;
    if (raw == "int32") return BufferDTypeDto::INT32;
    Fail(path, "unsupported BufferABI dtype");
}

BufferOwnershipDto ParseOwnership(const Json &value,
                                  const std::string &path) {
    const std::string raw = String(value, path);
    if (raw == "owned") return BufferOwnershipDto::OWNED;
    if (raw == "borrowed") return BufferOwnershipDto::BORROWED;
    if (raw == "aliased") return BufferOwnershipDto::ALIASED;
    Fail(path, "unknown BufferOwnership");
}

StateKindDto ParseStateKind(const Json &value, const std::string &path) {
    const std::string raw = String(value, path);
    if (raw == "parameter") return StateKindDto::PARAMETER;
    if (raw == "trainable_parameter")
        return StateKindDto::TRAINABLE_PARAMETER;
    if (raw == "kv_key") return StateKindDto::KV_KEY;
    if (raw == "kv_value") return StateKindDto::KV_VALUE;
    if (raw == "optimizer_reserved") return StateKindDto::OPTIMIZER_RESERVED;
    Fail(path, "unknown StateKind");
}

StateLifetimeDto ParseStateLifetime(const Json &value,
                                    const std::string &path) {
    const std::string raw = String(value, path);
    if (raw == "step") return StateLifetimeDto::STEP;
    if (raw == "persistent") return StateLifetimeDto::PERSISTENT;
    Fail(path, "unknown persistent-state lifetime");
}

StateAccessDto ParseStateAccess(const Json &value,
                                const std::string &path) {
    const std::string raw = String(value, path);
    if (raw == "read_only") return StateAccessDto::READ_ONLY;
    if (raw == "read_write") return StateAccessDto::READ_WRITE;
    if (raw == "reserved") return StateAccessDto::RESERVED;
    Fail(path, "unknown persistent-state access");
}


RuntimeSymbolDto ParseRuntimeSymbol(const Json &value,
                                    const std::string &path) {
    ExactObject(value, path, {"id", "kind", "source_ref"});
    return {String(Field(value, "id"), path + ".id"),
            ParseRuntimeSymbolKind(Field(value, "kind"), path + ".kind"),
            String(Field(value, "source_ref"), path + ".source_ref")};
}

ProgramSymbolDto ParseProgramSymbol(const Json &value,
                                    const std::string &path) {
    ExactObject(value, path, {"id", "kind", "source_ref"});
    return {String(Field(value, "id"), path + ".id"),
            ParseProgramSymbolKind(Field(value, "kind"), path + ".kind"),
            String(Field(value, "source_ref"), path + ".source_ref")};
}

LiteralValueDto ParseLiteral(const Json &value, const std::string &path) {
    if (value.is_null()) return std::monostate{};
    if (value.is_boolean()) return Bool(value, path);
    if (value.is_string()) return String(value, path);
    if (value.is_array()) return U64s(value, path);
    return U64(value, path);
}

RecordOperandDto ParseRecordOperand(const Json &value,
                                    const std::string &path) {
    ExactObject(value, path,
                {"name", "kind", "literal_value", "runtime_field",
                 "operand_id", "symbol_ref"});
    RecordOperandDto result;
    result.name = String(Field(value, "name"), path + ".name");
    result.kind = ParseOperandKind(Field(value, "kind"), path + ".kind");
    result.literal_value =
        ParseLiteral(Field(value, "literal_value"), path + ".literal_value");
    if (!Field(value, "runtime_field").is_null())
        result.runtime_field = ParseRuntimeField(Field(value, "runtime_field"),
                                                 path + ".runtime_field");
    if (!Field(value, "operand_id").is_null())
        result.operand_id =
            ParseOperandId(Field(value, "operand_id"), path + ".operand_id");
    result.symbol_ref =
        NullableString(Field(value, "symbol_ref"), path + ".symbol_ref");
    if (result.kind == OperandKindDto::LITERAL) {
        if (std::holds_alternative<std::monostate>(result.literal_value) ||
            result.runtime_field || result.operand_id || result.symbol_ref)
            Fail(path, "literal operand carries mixed or null state");
    } else if (result.kind == OperandKindDto::RUNTIME_SYMBOL) {
        if (!std::holds_alternative<std::monostate>(result.literal_value) ||
            !result.runtime_field || result.operand_id || !result.symbol_ref)
            Fail(path, "runtime-symbol operand carries mixed state");
    } else if (!std::holds_alternative<std::monostate>(result.literal_value) ||
               result.runtime_field || !result.operand_id ||
               !result.symbol_ref) {
        Fail(path, "address-symbol operand carries mixed state");
    }
    return result;
}

RelocatableRecordDto ParseRecord(const Json &value,
                                 const std::string &path) {
    ExactObject(value, path,
                {"source_global_action_id", "opcode", "operands"});
    return {String(Field(value, "source_global_action_id"),
                   path + ".source_global_action_id"),
            ParseOpcode(Field(value, "opcode"), path + ".opcode"),
            Array<RecordOperandDto>(Field(value, "operands"), path + ".operands",
                                    ParseRecordOperand)};
}

RuntimeRelocationDto ParseRuntimeRelocation(const Json &value,
                                            const std::string &path) {
    ExactObject(value, path, {"record_index", "field", "symbol_ref"});
    return {U64(Field(value, "record_index"), path + ".record_index"),
            ParseRuntimeField(Field(value, "field"), path + ".field"),
            String(Field(value, "symbol_ref"), path + ".symbol_ref")};
}

AddressRelocationDto ParseAddressRelocation(const Json &value,
                                            const std::string &path) {
    ExactObject(value, path,
                {"record_index", "operand_id", "symbol_kind", "symbol_ref",
                 "addend"});
    return {U64(Field(value, "record_index"), path + ".record_index"),
            ParseOperandId(Field(value, "operand_id"), path + ".operand_id"),
            ParseProgramSymbolKind(Field(value, "symbol_kind"),
                                   path + ".symbol_kind"),
            String(Field(value, "symbol_ref"), path + ".symbol_ref"),
            I64(Field(value, "addend"), path + ".addend")};
}

TensorSliceDto ParseTensorSlice(const Json &value,
                                const std::string &path) {
    ExactObject(value, path, {"value_id", "offset", "shape"});
    TensorSliceDto result{
        String(Field(value, "value_id"), path + ".value_id"),
        U64s(Field(value, "offset"), path + ".offset"),
        U64s(Field(value, "shape"), path + ".shape")};
    if (result.shape.empty() || result.shape.size() != result.offset.size() ||
        std::find(result.shape.begin(), result.shape.end(), 0) !=
            result.shape.end())
        Fail(path, "tensor slice requires equal non-empty rank and positive shape");
    return result;
}

BufferAbiDto ParseBufferAbi(const Json &value, const std::string &path) {
    ExactObject(value, path,
                {"id", "schedule_id", "binding_id", "value_id",
                 "logical_core", "tensor_slice", "region_ref",
                 "region_offset_bytes", "size_bytes", "alignment_bytes",
                 "banks", "storage_id", "alias_of", "lifetime_start",
                 "lifetime_end_exclusive", "dtype", "layout", "ownership"});
    BufferAbiDto result;
    result.id = String(Field(value, "id"), path + ".id");
    result.schedule_id = String(Field(value, "schedule_id"), path + ".schedule_id");
    result.binding_id = String(Field(value, "binding_id"), path + ".binding_id");
    result.value_id = String(Field(value, "value_id"), path + ".value_id");
    result.logical_core = ParseCore(Field(value, "logical_core"), path + ".logical_core");
    result.tensor_slice = ParseTensorSlice(Field(value, "tensor_slice"), path + ".tensor_slice");
    result.region_ref = String(Field(value, "region_ref"), path + ".region_ref");
    result.region_offset_bytes = U64(Field(value, "region_offset_bytes"), path + ".region_offset_bytes");
    result.size_bytes = U64(Field(value, "size_bytes"), path + ".size_bytes");
    result.alignment_bytes = U64(Field(value, "alignment_bytes"), path + ".alignment_bytes");
    result.banks = U64s(Field(value, "banks"), path + ".banks");
    result.storage_id = String(Field(value, "storage_id"), path + ".storage_id");
    result.alias_of = NullableString(Field(value, "alias_of"), path + ".alias_of");
    result.lifetime_start = U64(Field(value, "lifetime_start"), path + ".lifetime_start");
    result.lifetime_end_exclusive = U64(Field(value, "lifetime_end_exclusive"), path + ".lifetime_end_exclusive");
    result.dtype = ParseDType(Field(value, "dtype"), path + ".dtype");
    result.layout = String(Field(value, "layout"), path + ".layout");
    result.ownership = ParseOwnership(Field(value, "ownership"), path + ".ownership");
    if (result.tensor_slice.value_id != result.value_id || result.size_bytes == 0 ||
        result.alignment_bytes == 0 ||
        (result.alignment_bytes & (result.alignment_bytes - 1)) != 0 ||
        result.lifetime_start >= result.lifetime_end_exclusive)
        Fail(path, "invalid BufferABI size/alignment/lifetime/tensor closure");
    const bool aliased = result.ownership == BufferOwnershipDto::ALIASED;
    if (aliased != result.alias_of.has_value())
        Fail(path + ".alias_of", "must be present exactly for aliased ownership");
    if (std::set<uint64_t>(result.banks.begin(), result.banks.end()).size() !=
        result.banks.size())
        Fail(path + ".banks", "contains duplicate banks");
    return result;
}

StateAbiDto ParseStateAbi(const Json &value, const std::string &path) {
    ExactObject(value, path,
                {"id", "state_ref", "hbm_binding_ref", "kind",
                 "lifetime", "access", "shape", "dtype", "layout",
                 "die_id", "address", "size_bytes", "alignment_bytes"});
    StateAbiDto result;
    result.id = String(Field(value, "id"), path + ".id");
    result.state_ref = String(Field(value, "state_ref"), path + ".state_ref");
    result.hbm_binding_ref = String(Field(value, "hbm_binding_ref"),
                                    path + ".hbm_binding_ref");
    result.kind = ParseStateKind(Field(value, "kind"), path + ".kind");
    result.lifetime = ParseStateLifetime(Field(value, "lifetime"),
                                         path + ".lifetime");
    result.access = ParseStateAccess(Field(value, "access"), path + ".access");
    result.shape = U64s(Field(value, "shape"), path + ".shape");
    result.dtype = ParseDType(Field(value, "dtype"), path + ".dtype");
    if (result.dtype == BufferDTypeDto::INT32)
        Fail(path + ".dtype", "StateABI dtype must be fp16 or fp32");
    result.layout = String(Field(value, "layout"), path + ".layout");
    result.die_id = U64(Field(value, "die_id"), path + ".die_id");
    result.address = U64(Field(value, "address"), path + ".address");
    result.size_bytes = U64(Field(value, "size_bytes"), path + ".size_bytes");
    result.alignment_bytes = U64(Field(value, "alignment_bytes"),
                                 path + ".alignment_bytes");
    if (result.id.empty() || result.state_ref.empty() ||
        result.hbm_binding_ref.empty() || result.layout.empty() ||
        result.shape.empty())
        Fail(path, "StateABI identity/layout/shape must be non-empty");
    uint64_t elements = 1;
    for (uint64_t extent : result.shape) {
        if (extent == 0 ||
            elements > std::numeric_limits<uint64_t>::max() / extent)
            Fail(path + ".shape", "invalid or overflowing state shape");
        elements *= extent;
    }
    const uint64_t element_bytes =
        result.dtype == BufferDTypeDto::FP16 ? 2 : 4;
    if (elements > std::numeric_limits<uint64_t>::max() / element_bytes ||
        result.size_bytes != elements * element_bytes ||
        result.size_bytes == 0)
        Fail(path + ".size_bytes",
             "must equal product(shape) * dtype bytes");
    if (result.die_id > static_cast<uint64_t>(
                            std::numeric_limits<int32_t>::max()))
        Fail(path + ".die_id", "must fit signed 32-bit range");
    if (result.alignment_bytes == 0 ||
        (result.alignment_bytes & (result.alignment_bytes - 1)) != 0 ||
        result.address % result.alignment_bytes != 0)
        Fail(path, "invalid StateABI address alignment");
    if (result.address > std::numeric_limits<uint64_t>::max() -
                             result.size_bytes)
        Fail(path, "StateABI HBM span overflows uint64");
    if (result.kind == StateKindDto::PARAMETER) {
        if (result.lifetime != StateLifetimeDto::PERSISTENT ||
            result.access != StateAccessDto::READ_ONLY)
            Fail(path, "parameter must be PERSISTENT and READ_ONLY");
    } else if (result.kind == StateKindDto::TRAINABLE_PARAMETER) {
        if (result.lifetime != StateLifetimeDto::PERSISTENT ||
            result.access != StateAccessDto::READ_WRITE)
            Fail(path,
                 "trainable parameter must be PERSISTENT and READ_WRITE");
    } else if (result.kind == StateKindDto::KV_KEY ||
               result.kind == StateKindDto::KV_VALUE) {
        if (result.lifetime != StateLifetimeDto::PERSISTENT ||
            result.access != StateAccessDto::READ_WRITE)
            Fail(path, "KV state must be PERSISTENT and READ_WRITE");
    } else if (result.access != StateAccessDto::RESERVED) {
        Fail(path + ".access",
             "optimizer reservation cannot grant DMA access");
    }
    const std::string expected = StableArtifactId(
        "state_abi", kStateAbiSchemaVersion, Without(value, {"id"}));
    if (result.id != expected)
        Fail(path + ".id", "unstable artifact id; expected '" + expected + "'");
    return result;
}


CoreFragmentStreamDto ParseCoreFragmentStream(const Json &value,
                                              const std::string &path) {
    ExactObject(value, path,
                {"logical_core", "records", "runtime_relocations",
                 "address_relocations"});
    CoreFragmentStreamDto result{
        ParseCore(Field(value, "logical_core"), path + ".logical_core"),
        Array<RelocatableRecordDto>(Field(value, "records"), path + ".records", ParseRecord),
        Array<RuntimeRelocationDto>(Field(value, "runtime_relocations"), path + ".runtime_relocations", ParseRuntimeRelocation),
        Array<AddressRelocationDto>(Field(value, "address_relocations"), path + ".address_relocations", ParseAddressRelocation)};
    if (result.records.empty())
        Fail(path + ".records", "must be non-empty");
    return result;
}

CommandFragmentDto ParseCommandFragment(const Json &value,
                                        const std::string &path) {
    ExactObject(value, path,
                {"schema_version", "producer_pass", "id",
                 "source_global_dag_id", "kind", "claimed_action_ids",
                 "core_streams", "runtime_symbols", "program_symbols",
                 "buffer_abi", "state_abi"});
    CommandFragmentDto result;
    result.schema_version = String(Field(value, "schema_version"), path + ".schema_version");
    if (result.schema_version != kCommandFragmentSchemaVersion)
        Fail(path + ".schema_version", "unsupported schema version");
    result.producer_pass = String(Field(value, "producer_pass"), path + ".producer_pass");
    result.id = String(Field(value, "id"), path + ".id");
    result.source_global_dag_id = String(Field(value, "source_global_dag_id"), path + ".source_global_dag_id");
    result.kind = ParseFragmentKind(Field(value, "kind"), path + ".kind");
    result.claimed_action_ids = Strings(Field(value, "claimed_action_ids"), path + ".claimed_action_ids");
    result.core_streams = Array<CoreFragmentStreamDto>(Field(value, "core_streams"), path + ".core_streams", ParseCoreFragmentStream);
    result.runtime_symbols = Array<RuntimeSymbolDto>(Field(value, "runtime_symbols"), path + ".runtime_symbols", ParseRuntimeSymbol);
    result.program_symbols = Array<ProgramSymbolDto>(Field(value, "program_symbols"), path + ".program_symbols", ParseProgramSymbol);
    result.buffer_abi = Array<BufferAbiDto>(Field(value, "buffer_abi"), path + ".buffer_abi", ParseBufferAbi);
    result.state_abi = Array<StateAbiDto>(
        Field(value, "state_abi"), path + ".state_abi", ParseStateAbi);
    return result;
}

RegionManifestDto ParseRegionManifest(const Json &value,
                                      const std::string &path) {
    ExactObject(value, path,
                {"schema_version", "producer_pass", "id", "region_id",
                 "fusion_plan_id", "target_dies", "fragment"});
    RegionManifestDto result;
    result.schema_version = String(Field(value, "schema_version"), path + ".schema_version");
    if (result.schema_version != kRegionManifestSchemaVersion)
        Fail(path + ".schema_version", "unsupported schema version");
    result.producer_pass = String(Field(value, "producer_pass"), path + ".producer_pass");
    result.id = String(Field(value, "id"), path + ".id");
    result.region_id = String(Field(value, "region_id"), path + ".region_id");
    result.fusion_plan_id = String(Field(value, "fusion_plan_id"), path + ".fusion_plan_id");
    result.target_dies = U64s(Field(value, "target_dies"), path + ".target_dies");
    result.fragment = ParseCommandFragment(Field(value, "fragment"), path + ".fragment");
    return result;
}

LinkedFragmentDto ParseLinkedFragment(const Json &value,
                                      const std::string &path) {
    if (!value.is_object()) Fail(path, "must be an object");
    const auto it = value.find("schema_version");
    if (it == value.end()) Fail(path + ".schema_version", "missing required field");
    const std::string version = String(*it, path + ".schema_version");
    if (version == kCommandFragmentSchemaVersion)
        return ParseCommandFragment(value, path);
    if (version == kRegionManifestSchemaVersion)
        return ParseRegionManifest(value, path);
    Fail(path + ".schema_version", "unsupported linked fragment schema version");
}

ManifestInputDigestDto ParseDigest(const Json &value,
                                   const std::string &path) {
    ExactObject(value, path, {"kind", "artifact_id", "schema_version", "digest"});
    ManifestInputDigestDto result{
        ParseInputKind(Field(value, "kind"), path + ".kind"),
        String(Field(value, "artifact_id"), path + ".artifact_id"),
        String(Field(value, "schema_version"), path + ".schema_version"),
        String(Field(value, "digest"), path + ".digest")};
    if (result.digest.size() != 64 ||
        !std::all_of(result.digest.begin(), result.digest.end(), [](char c) {
            return std::isdigit(static_cast<unsigned char>(c)) ||
                   (c >= 'a' && c <= 'f');
        }))
        Fail(path + ".digest", "must be lowercase SHA-256 hex");
    return result;
}

CoreRuntimeBindingDto ParseCoreBinding(const Json &value,
                                       const std::string &path) {
    ExactObject(value, path,
                {"logical_core", "core_spec_ref", "runtime_core_id",
                 "sram_profile_ref"});
    CoreRuntimeBindingDto result{
        ParseCore(Field(value, "logical_core"), path + ".logical_core"),
        String(Field(value, "core_spec_ref"), path + ".core_spec_ref"),
        U64(Field(value, "runtime_core_id"), path + ".runtime_core_id"),
        String(Field(value, "sram_profile_ref"), path + ".sram_profile_ref")};
    if (result.runtime_core_id > 0xffff)
        Fail(path + ".runtime_core_id", "must fit uint16");
    return result;
}

LinkedRecordRefDto ParseLinkedRecordRef(const Json &value,
                                        const std::string &path) {
    ExactObject(value, path,
                {"fragment_id", "fragment_record_index",
                 "source_global_action_id"});
    return {String(Field(value, "fragment_id"), path + ".fragment_id"),
            U64(Field(value, "fragment_record_index"), path + ".fragment_record_index"),
            String(Field(value, "source_global_action_id"), path + ".source_global_action_id")};
}

LinkedCoreStreamDto ParseLinkedCoreStream(const Json &value,
                                          const std::string &path) {
    ExactObject(value, path, {"logical_core", "runtime_core_id", "records"});
    LinkedCoreStreamDto result{
        ParseCore(Field(value, "logical_core"), path + ".logical_core"),
        U64(Field(value, "runtime_core_id"), path + ".runtime_core_id"),
        Array<LinkedRecordRefDto>(Field(value, "records"), path + ".records", ParseLinkedRecordRef)};
    if (result.runtime_core_id > 0xffff)
        Fail(path + ".runtime_core_id", "must fit uint16");
    return result;
}

EventCreditDto ParseEventCredit(const Json &value,
                                const std::string &path) {
    ExactObject(value, path, {"symbol_ref", "count"});
    EventCreditDto result{String(Field(value, "symbol_ref"), path + ".symbol_ref"),
                          U64(Field(value, "count"), path + ".count")};
    if (result.count == 0) Fail(path + ".count", "must be non-zero");
    return result;
}

FragmentInterfaceDto ParseFragmentInterface(const Json &value,
                                            const std::string &path) {
    ExactObject(value, path,
                {"fragment_id", "runtime_imports", "runtime_exports",
                 "program_imports", "program_exports", "entry_events",
                 "exit_events"});
    return {
        String(Field(value, "fragment_id"), path + ".fragment_id"),
        Strings(Field(value, "runtime_imports"), path + ".runtime_imports"),
        Strings(Field(value, "runtime_exports"), path + ".runtime_exports"),
        Strings(Field(value, "program_imports"), path + ".program_imports"),
        Strings(Field(value, "program_exports"), path + ".program_exports"),
        Array<EventCreditDto>(Field(value, "entry_events"), path + ".entry_events", ParseEventCredit),
        Array<EventCreditDto>(Field(value, "exit_events"), path + ".exit_events", ParseEventCredit)};
}

RuntimeSymbolDefinitionDto ParseRuntimeDefinition(const Json &value,
                                                  const std::string &path) {
    ExactObject(value, path,
                {"symbol", "logical_cores", "source_action_id",
                 "destination_action_id"});
    return {
        ParseRuntimeSymbol(Field(value, "symbol"), path + ".symbol"),
        Array<LogicalCoreDto>(Field(value, "logical_cores"), path + ".logical_cores", ParseCore),
        NullableString(Field(value, "source_action_id"), path + ".source_action_id"),
        NullableString(Field(value, "destination_action_id"), path + ".destination_action_id")};
}

ProgramSymbolDefinitionDto ParseProgramDefinition(const Json &value,
                                                  const std::string &path) {
    ExactObject(value, path,
                {"symbol", "name", "value", "size_bytes",
                 "logical_cores"});
    ProgramSymbolDefinitionDto result{
        ParseProgramSymbol(Field(value, "symbol"), path + ".symbol"),
        String(Field(value, "name"), path + ".name"),
        U64(Field(value, "value"), path + ".value"),
        U64(Field(value, "size_bytes"), path + ".size_bytes"),
        Array<LogicalCoreDto>(Field(value, "logical_cores"), path + ".logical_cores", ParseCore)};
    if (result.name.find('\0') != std::string::npos || result.name.size() > 255)
        Fail(path + ".name", "must be <=255 UTF-8 bytes without NUL");
    if (result.symbol.kind == ProgramSymbolKind::SRAM_REGION &&
        (result.name.size() > 64 || result.size_bytes == 0))
        Fail(path, "SRAM region requires a <=64-byte name and non-zero size");
    if (result.symbol.kind == ProgramSymbolKind::SRAM_LABEL &&
        (result.value != 0 || result.size_bytes != 0))
        Fail(path, "SRAM label carries no address or span");
    if (result.symbol.kind == ProgramSymbolKind::ABSOLUTE_ADDRESS &&
        result.size_bytes == 0)
        Fail(path + ".size_bytes", "absolute address span must be non-zero");
    if (result.size_bytes != 0 &&
        result.value > std::numeric_limits<uint64_t>::max() -
                           (result.size_bytes - 1))
        Fail(path, "program symbol span overflows uint64");
    return result;
}

AddressOperandBindingDto ParseAddressBinding(const Json &value,
                                             const std::string &path) {
    ExactObject(value, path,
                {"fragment_id", "logical_core", "fragment_record_index",
                 "operand_id", "buffer_abi_ids", "tensor_slices"});
    AddressOperandBindingDto result{
        String(Field(value, "fragment_id"), path + ".fragment_id"),
        ParseCore(Field(value, "logical_core"), path + ".logical_core"),
        U64(Field(value, "fragment_record_index"), path + ".fragment_record_index"),
        ParseOperandId(Field(value, "operand_id"), path + ".operand_id"),
        Strings(Field(value, "buffer_abi_ids"), path + ".buffer_abi_ids"),
        Array<TensorSliceDto>(Field(value, "tensor_slices"),
                              path + ".tensor_slices", ParseTensorSlice)};
    if (result.buffer_abi_ids.empty() ||
        result.tensor_slices.size() != result.buffer_abi_ids.size())
        Fail(path, "buffer_abi_ids and tensor_slices must have equal non-zero length");
    return result;
}

StateOperandBindingDto ParseStateBinding(const Json &value,
                                         const std::string &path) {
    ExactObject(value, path,
                {"fragment_id", "logical_core", "fragment_record_index",
                 "operand_id", "state_abi_id"});
    StateOperandBindingDto result{
        String(Field(value, "fragment_id"), path + ".fragment_id"),
        ParseCore(Field(value, "logical_core"), path + ".logical_core"),
        U64(Field(value, "fragment_record_index"),
            path + ".fragment_record_index"),
        ParseOperandId(Field(value, "operand_id"), path + ".operand_id"),
        String(Field(value, "state_abi_id"), path + ".state_abi_id")};
    if (result.fragment_id.empty() || result.state_abi_id.empty() ||
        result.operand_id != SemanticOperandId::HBM_ADDRESS)
        Fail(path, "state operand binding only supports HBM_ADDRESS");
    return result;
}


LogicalCoreGroupDto ParseCoreGroup(const Json &value,
                                   const std::string &path) {
    ExactObject(value, path, {"symbol_ref", "members"});
    return {String(Field(value, "symbol_ref"), path + ".symbol_ref"),
            Array<LogicalCoreDto>(Field(value, "members"), path + ".members", ParseCore)};
}

LogicalStartEventDto ParseStartEvent(const Json &value,
                                     const std::string &path) {
    ExactObject(value, path, {"target_core", "tag_symbol_ref", "count"});
    LogicalStartEventDto result{
        ParseCore(Field(value, "target_core"), path + ".target_core"),
        String(Field(value, "tag_symbol_ref"), path + ".tag_symbol_ref"),
        U64(Field(value, "count"), path + ".count")};
    if (result.count == 0 || result.count > 0xff)
        Fail(path + ".count", "must fit non-zero uint8");
    return result;
}

ProgramControlEnvelopeDto ParseEnvelope(const Json &value,
                                        const std::string &path) {
    ExactObject(value, path,
                {"active_cores", "start_events", "terminal_cores",
                 "expected_ack_cores", "expected_done_cores",
                 "empty_core_ack_policy", "failure_policy"});
    ProgramControlEnvelopeDto result;
    result.active_cores = Array<LogicalCoreDto>(Field(value, "active_cores"), path + ".active_cores", ParseCore);
    result.start_events = Array<LogicalStartEventDto>(Field(value, "start_events"), path + ".start_events", ParseStartEvent);
    result.terminal_cores = Array<LogicalCoreDto>(Field(value, "terminal_cores"), path + ".terminal_cores", ParseCore);
    result.expected_ack_cores = Array<LogicalCoreDto>(Field(value, "expected_ack_cores"), path + ".expected_ack_cores", ParseCore);
    result.expected_done_cores = Array<LogicalCoreDto>(Field(value, "expected_done_cores"), path + ".expected_done_cores", ParseCore);
    const std::string ack = String(Field(value, "empty_core_ack_policy"), path + ".empty_core_ack_policy");
    if (ack == "exclude_empty") result.empty_core_ack_policy = EmptyCoreAckPolicy::EXCLUDE_EMPTY;
    else if (ack == "include_empty") result.empty_core_ack_policy = EmptyCoreAckPolicy::INCLUDE_EMPTY;
    else Fail(path + ".empty_core_ack_policy", "unknown policy");
    if (String(Field(value, "failure_policy"), path + ".failure_policy") != "abort_all")
        Fail(path + ".failure_policy", "ProgramArtifact v1 requires abort_all");
    result.failure_policy = ProgramFailurePolicy::ABORT_ALL;
    return result;
}

void ValidateCommandStableIds(const Json &value, const std::string &path) {
    const std::string expected = StableArtifactId(
        "command_fragment", kCommandFragmentSchemaVersion,
        Without(value, {"schema_version", "producer_pass", "id"}));
    if (String(Field(value, "id"), path + ".id") != expected)
        Fail(path + ".id", "unstable artifact id; expected '" + expected + "'");
}

void ValidateManifestStableIds(const Json &value,
                               const std::string &path) {
    const Json &fragments = Field(value, "fragments");
    if (!fragments.is_array())
        Fail(path + ".fragments", "must be an array");
    struct EmbeddedDigest {
        std::string kind;
        std::string id;
        std::string schema_version;
        std::string digest;
    };
    std::vector<EmbeddedDigest> embedded;
    for (std::size_t index = 0; index < fragments.size(); ++index) {
        const Json &linked = fragments[index];
        const std::string fragment_path = path + ".fragments[" +
                                          std::to_string(index) + "]";
        if (!linked.is_object())
            Fail(fragment_path, "must be an object");
        const std::string schema =
            String(Field(linked, "schema_version"),
                   fragment_path + ".schema_version");
        if (schema == kCommandFragmentSchemaVersion) {
            ValidateCommandStableIds(linked, fragment_path);
            embedded.push_back({
                "command_fragment",
                String(Field(linked, "id"), fragment_path + ".id"), schema,
                CanonicalDigest(linked)});
        } else if (schema == kRegionManifestSchemaVersion) {
            const Json &leaf = Field(linked, "fragment");
            ValidateCommandStableIds(leaf, fragment_path + ".fragment");
            const std::string expected = StableArtifactId(
                "region_manifest", kRegionManifestSchemaVersion,
                Without(linked, {"schema_version", "producer_pass", "id"}));
            if (String(Field(linked, "id"), fragment_path + ".id") != expected)
                Fail(fragment_path + ".id",
                     "unstable artifact id; expected '" + expected + "'");
            embedded.push_back({
                "region_manifest", expected, schema, CanonicalDigest(linked)});
            embedded.push_back({
                "command_fragment",
                String(Field(leaf, "id"), fragment_path + ".fragment.id"),
                String(Field(leaf, "schema_version"),
                       fragment_path + ".fragment.schema_version"),
                CanonicalDigest(leaf)});
        }
    }
    const Json &digests = Field(value, "input_digests");
    if (!digests.is_array())
        Fail(path + ".input_digests", "must be an array");
    for (const EmbeddedDigest &expected : embedded) {
        std::size_t matches = 0;
        for (const Json &digest : digests) {
            if (digest.is_object() && digest.value("kind", "") == expected.kind &&
                digest.value("artifact_id", "") == expected.id) {
                ++matches;
                if (digest.value("schema_version", "") !=
                        expected.schema_version ||
                    digest.value("digest", "") != expected.digest)
                    Fail(path + ".input_digests",
                         "embedded fragment digest does not match canonical content");
            }
        }
        if (matches != 1)
            Fail(path + ".input_digests",
                 "must contain exactly one canonical digest for every embedded fragment");
    }
    const std::string expected = StableArtifactId(
        "linked_program_manifest", kLinkedProgramManifestSchemaVersion,
        Without(value, {"schema_version", "producer_pass", "id"}));
    if (String(Field(value, "id"), path + ".id") != expected)
        Fail(path + ".id", "unstable artifact id; expected '" + expected + "'");
}

Json ParseStrictJson(std::string_view text) {
    std::vector<std::set<std::string>> object_keys;
    auto callback = [&object_keys](int, Json::parse_event_t event,
                                   Json &parsed) {
        if (event == Json::parse_event_t::object_start) {
            object_keys.emplace_back();
        } else if (event == Json::parse_event_t::key) {
            if (object_keys.empty())
                Fail("$", "internal duplicate-key parser state");
            const std::string key = parsed.get<std::string>();
            if (!object_keys.back().insert(key).second)
                Fail("$", "duplicate object key '" + key + "'");
        } else if (event == Json::parse_event_t::object_end) {
            if (object_keys.empty())
                Fail("$", "internal duplicate-key parser state");
            object_keys.pop_back();
        }
        return true;
    };
    try {
        Json result = Json::parse(text.begin(), text.end(), callback, true, false);
        if (result.is_discarded()) Fail("$", "invalid JSON");
        return result;
    } catch (const ProgramFinalizerError &) {
        throw;
    } catch (const std::exception &error) {
        throw ProgramFinalizerError(std::string("$: invalid JSON: ") + error.what());
    }
}

} // namespace

LinkedProgramManifestDto
ProgramArtifactFinalizer::Parse(std::string_view manifest_json) {
    try {
    const Json value = ParseStrictJson(manifest_json);
    const std::string path = "linked_program_manifest";
    ExactObject(value, path,
                {"schema_version", "producer_pass", "id", "capabilities",
                 "source_ir1_id", "source_projection_id",
                 "source_schedule_set_id", "source_global_dag_id",
                 "input_digests", "fragments", "fragment_interfaces",
                 "core_bindings", "core_streams",
                 "runtime_symbol_definitions", "program_symbol_definitions",
                 "address_operand_bindings", "state_operand_bindings",
                 "core_groups", "envelope"});
    ValidateManifestStableIds(value, path);
    LinkedProgramManifestDto result;
    result.schema_version = String(Field(value, "schema_version"), path + ".schema_version");
    if (result.schema_version != kLinkedProgramManifestSchemaVersion)
        Fail(path + ".schema_version", "unsupported schema version");
    result.producer_pass = String(Field(value, "producer_pass"), path + ".producer_pass");
    result.id = String(Field(value, "id"), path + ".id");
    result.capabilities = U64(Field(value, "capabilities"), path + ".capabilities");
    if (result.capabilities != 0)
        Fail(path + ".capabilities", "MVP capabilities must be zero");
    result.source_ir1_id = String(Field(value, "source_ir1_id"), path + ".source_ir1_id");
    result.source_projection_id = String(Field(value, "source_projection_id"), path + ".source_projection_id");
    result.source_schedule_set_id = String(Field(value, "source_schedule_set_id"), path + ".source_schedule_set_id");
    result.source_global_dag_id = String(Field(value, "source_global_dag_id"), path + ".source_global_dag_id");
    result.input_digests = Array<ManifestInputDigestDto>(Field(value, "input_digests"), path + ".input_digests", ParseDigest);
    result.fragments = Array<LinkedFragmentDto>(Field(value, "fragments"), path + ".fragments", ParseLinkedFragment);
    result.fragment_interfaces = Array<FragmentInterfaceDto>(Field(value, "fragment_interfaces"), path + ".fragment_interfaces", ParseFragmentInterface);
    result.core_bindings = Array<CoreRuntimeBindingDto>(Field(value, "core_bindings"), path + ".core_bindings", ParseCoreBinding);
    result.core_streams = Array<LinkedCoreStreamDto>(Field(value, "core_streams"), path + ".core_streams", ParseLinkedCoreStream);
    result.runtime_symbol_definitions = Array<RuntimeSymbolDefinitionDto>(Field(value, "runtime_symbol_definitions"), path + ".runtime_symbol_definitions", ParseRuntimeDefinition);
    result.program_symbol_definitions = Array<ProgramSymbolDefinitionDto>(Field(value, "program_symbol_definitions"), path + ".program_symbol_definitions", ParseProgramDefinition);
    result.address_operand_bindings = Array<AddressOperandBindingDto>(Field(value, "address_operand_bindings"), path + ".address_operand_bindings", ParseAddressBinding);
    result.state_operand_bindings = Array<StateOperandBindingDto>(
        Field(value, "state_operand_bindings"),
        path + ".state_operand_bindings", ParseStateBinding);
    result.core_groups = Array<LogicalCoreGroupDto>(Field(value, "core_groups"), path + ".core_groups", ParseCoreGroup);
    result.envelope = ParseEnvelope(Field(value, "envelope"), path + ".envelope");
    if (result.fragments.empty())
        Fail(path + ".fragments", "must be non-empty");
    return result;
    } catch (const ProgramFinalizerError &) {
        throw;
    } catch (const std::exception &error) {
        throw ProgramFinalizerError(
            std::string("linked_program_manifest: parsing failed: ") +
            error.what());
    }
}

std::string ProgramArtifactFinalizer::CanonicalManifestDigest(
    std::string_view manifest_json) {
    const Json value = ParseStrictJson(manifest_json);
    return CanonicalDigest(value);
}

std::string ProgramArtifactFinalizer::EncodedArtifactDigest(
    const std::vector<uint8_t> &encoded_artifact) {
    return Sha256(std::string_view(
        reinterpret_cast<const char *>(encoded_artifact.data()),
        encoded_artifact.size()));
}

namespace {

template <typename T>
void RequireCanonical(const std::vector<T> &values, const std::string &path) {
    if (!std::is_sorted(values.begin(), values.end()) ||
        std::adjacent_find(values.begin(), values.end()) != values.end())
        Fail(path, "must be unique and canonical");
}

template <typename T, typename Key>
void RequireCanonicalBy(const std::vector<T> &values, const std::string &path,
                        Key key) {
    for (std::size_t index = 1; index < values.size(); ++index) {
        if (!(key(values[index - 1]) < key(values[index])))
            Fail(path, "must be unique and canonical");
    }
}

bool SameProgramSymbol(const ProgramSymbolDto &left,
                       const ProgramSymbolDto &right) {
    return left.id == right.id && left.kind == right.kind &&
           left.source_ref == right.source_ref;
}

bool SameBufferAbi(const BufferAbiDto &left, const BufferAbiDto &right) {
    return left.id == right.id && left.schedule_id == right.schedule_id &&
           left.binding_id == right.binding_id &&
           left.value_id == right.value_id &&
           left.logical_core == right.logical_core &&
           left.tensor_slice.value_id == right.tensor_slice.value_id &&
           left.tensor_slice.offset == right.tensor_slice.offset &&
           left.tensor_slice.shape == right.tensor_slice.shape &&
           left.region_ref == right.region_ref &&
           left.region_offset_bytes == right.region_offset_bytes &&
           left.size_bytes == right.size_bytes &&
           left.alignment_bytes == right.alignment_bytes &&
           left.banks == right.banks &&
           left.storage_id == right.storage_id &&
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


std::string_view InputKindKey(ManifestInputKindDto kind) {
    switch (kind) {
    case ManifestInputKindDto::S3_LITE_MOE: return "s3_lite_moe";
    case ManifestInputKindDto::S2_LITE_ROOTED_AR:
        return "s2_lite_rooted_ar";
    case ManifestInputKindDto::TRAIN_LOWERED_PROGRAM:
        return "train_lowered_program";
    case ManifestInputKindDto::IR1: return "ir1";
    case ManifestInputKindDto::FUSION_PLAN: return "fusion_plan";
    case ManifestInputKindDto::STANDALONE_PLAN: return "standalone_plan";
    case ManifestInputKindDto::IR2_PROJECTION: return "ir2_projection";
    case ManifestInputKindDto::SCHEDULE_SET: return "schedule_set";
    case ManifestInputKindDto::GLOBAL_ACTION_DAG: return "global_action_dag";
    case ManifestInputKindDto::COMMAND_FRAGMENT: return "command_fragment";
    case ManifestInputKindDto::REGION_MANIFEST: return "region_manifest";
    }
    Fail("linked_program_manifest.input_digests", "unknown input kind");
}

bool ContainsCore(const std::vector<LogicalCoreDto> &cores,
                  const LogicalCoreDto &core) {
    return std::binary_search(cores.begin(), cores.end(), core);
}

const CommandFragmentDto &Leaf(const LinkedFragmentDto &fragment) {
    if (const auto *command = std::get_if<CommandFragmentDto>(&fragment))
        return *command;
    return std::get<RegionManifestDto>(fragment).fragment;
}

std::string OuterId(const LinkedFragmentDto &fragment) {
    if (const auto *command = std::get_if<CommandFragmentDto>(&fragment))
        return command->id;
    return std::get<RegionManifestDto>(fragment).id;
}

uint64_t LiteralU64(const RecordOperandDto &operand,
                    const std::string &path) {
    if (operand.kind != OperandKindDto::LITERAL ||
        !std::holds_alternative<uint64_t>(operand.literal_value))
        Fail(path, "must be a uint64 literal");
    return std::get<uint64_t>(operand.literal_value);
}

template <typename Enum>
Enum LiteralEnum(const RecordOperandDto &operand, const std::string &path) {
    const uint64_t raw = LiteralU64(operand, path);
    if (raw > std::numeric_limits<uint8_t>::max())
        Fail(path, "enum literal exceeds uint8 and would truncate");
    return static_cast<Enum>(raw);
}

bool LiteralBool(const RecordOperandDto &operand, const std::string &path) {
    if (operand.kind != OperandKindDto::LITERAL ||
        !std::holds_alternative<bool>(operand.literal_value))
        Fail(path, "must be a boolean literal");
    return std::get<bool>(operand.literal_value);
}

const std::vector<uint64_t> &LiteralU64Array(const RecordOperandDto &operand,
                                             const std::string &path) {
    if (operand.kind != OperandKindDto::LITERAL ||
        !std::holds_alternative<std::vector<uint64_t>>(
            operand.literal_value))
        Fail(path, "must be an integer-array literal");
    return std::get<std::vector<uint64_t>>(operand.literal_value);
}

void RequireLiteral(const RecordOperandDto &operand, std::string_view name,
                    const std::string &path) {
    if (operand.name != name || operand.kind != OperandKindDto::LITERAL ||
        std::holds_alternative<std::monostate>(operand.literal_value) ||
        operand.runtime_field || operand.operand_id || operand.symbol_ref)
        Fail(path, "does not match literal ABI field '" +
                       std::string(name) + "'");
}

void RequireAddress(const RecordOperandDto &operand, std::string_view name,
                    SemanticOperandId id, const std::string &path) {
    if (operand.name != name || operand.kind != OperandKindDto::ADDRESS_SYMBOL ||
        operand.operand_id != id || !operand.symbol_ref ||
        !std::holds_alternative<std::monostate>(operand.literal_value) ||
        operand.runtime_field)
        Fail(path, "does not match address ABI field '" +
                       std::string(name) + "'");
}

void RequireRuntime(const RecordOperandDto &operand, std::string_view name,
                    RuntimeOperandFieldDto field, const std::string &path) {
    if (operand.name != name || operand.kind != OperandKindDto::RUNTIME_SYMBOL ||
        operand.runtime_field != field || !operand.symbol_ref ||
        !std::holds_alternative<std::monostate>(operand.literal_value) ||
        operand.operand_id)
        Fail(path, "does not match runtime ABI field '" +
                       std::string(name) + "'");
}

struct SymbolEntry {
    const ProgramSymbolDefinitionDto *definition = nullptr;
    uint64_t index = 0;
};

struct RuntimeEntry {
    const RuntimeSymbolDefinitionDto *definition = nullptr;
    uint64_t value = 0;
};

uint64_t ApplyAddend(uint64_t value, int64_t addend,
                     const std::string &path) {
    if (addend >= 0) {
        const uint64_t positive = static_cast<uint64_t>(addend);
        if (value > std::numeric_limits<uint64_t>::max() - positive)
            Fail(path, "symbol value plus addend overflows uint64");
        return value + positive;
    }
    const uint64_t magnitude = uint64_t{0} - static_cast<uint64_t>(addend);
    if (value < magnitude)
        Fail(path, "symbol value plus addend underflows uint64");
    return value - magnitude;
}

uint64_t CheckedAdd(uint64_t left, uint64_t right,
                    const std::string &path) {
    if (left > std::numeric_limits<uint64_t>::max() - right)
        Fail(path, "byte-span addition overflows uint64");
    return left + right;
}

uint64_t CheckedMultiply(uint64_t left, uint64_t right,
                         const std::string &path) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)
        Fail(path, "byte-span multiplication overflows uint64");
    return left * right;
}

uint64_t ElementBytes(BufferDTypeDto dtype) {
    switch (dtype) {
    case BufferDTypeDto::FP16: return 2;
    case BufferDTypeDto::FP32: return 4;
    case BufferDTypeDto::INT32: return 4;
    }
    Fail("address_operand_binding.tensor_slices", "unknown BufferABI dtype");
}

struct DenseViewSpan {
    uint64_t addend = 0;
    uint64_t length = 0;
    uint64_t root_length = 0;
};

DenseViewSpan DenseRowMajorViewSpan(const TensorSliceDto &root,
                                    const TensorSliceDto &view,
                                    BufferDTypeDto dtype,
                                    const std::string &path) {
    if (root.value_id != view.value_id || root.shape.empty() ||
        root.shape.size() != root.offset.size() ||
        view.shape.size() != root.shape.size() ||
        view.offset.size() != root.shape.size())
        Fail(path, "view and root must reference one value with equal non-zero rank");

    const std::size_t rank = root.shape.size();
    std::vector<uint64_t> relative(rank);
    for (std::size_t axis = 0; axis < rank; ++axis) {
        if (root.shape[axis] == 0 || view.shape[axis] == 0)
            Fail(path, "root and view shapes must be positive");
        const uint64_t root_end = CheckedAdd(
            root.offset[axis], root.shape[axis], path + ".root");
        const uint64_t view_end = CheckedAdd(
            view.offset[axis], view.shape[axis], path + ".view");
        if (view.offset[axis] < root.offset[axis] || view_end > root_end)
            Fail(path, "view must be contained in its root backing");
        relative[axis] = view.offset[axis] - root.offset[axis];
    }

    std::vector<uint64_t> strides(rank, 1);
    for (std::size_t axis = rank - 1; axis != 0; --axis)
        strides[axis - 1] = CheckedMultiply(
            strides[axis], root.shape[axis], path + ".root.shape");

    uint64_t first = 0;
    uint64_t last = 0;
    uint64_t view_elements = 1;
    uint64_t root_elements = 1;
    for (std::size_t axis = 0; axis < rank; ++axis) {
        first = CheckedAdd(
            first, CheckedMultiply(relative[axis], strides[axis], path), path);
        const uint64_t last_coordinate = CheckedAdd(
            relative[axis], view.shape[axis] - 1, path);
        last = CheckedAdd(
            last, CheckedMultiply(last_coordinate, strides[axis], path), path);
        view_elements = CheckedMultiply(view_elements, view.shape[axis], path);
        root_elements = CheckedMultiply(root_elements, root.shape[axis], path);
    }
    if (last < first || CheckedAdd(last - first, 1, path) != view_elements)
        Fail(path, "view is not contiguous in its dense row-major root backing");
    const uint64_t element_bytes = ElementBytes(dtype);
    return {
        CheckedMultiply(first, element_bytes, path),
        CheckedMultiply(view_elements, element_bytes, path),
        CheckedMultiply(root_elements, element_bytes, path),
    };
}

uint64_t ComputeOperandBytes(const RelocatableRecordDto &record,
                             SemanticOperandId operand_id,
                             const std::string &path) {
    const std::vector<uint64_t> &parameters =
        LiteralU64Array(record.operands.back(), path + ".parameters");
    auto product = [&](std::initializer_list<uint64_t> factors) {
        uint64_t result = 1;
        for (uint64_t factor : factors)
            result = CheckedMultiply(result, factor, path);
        return result;
    };
    uint64_t elements = 0;
    if (record.opcode == Opcode::MATMUL) {
        if (parameters.size() != 4)
            Fail(path, "MATMUL parameters must have four fields");
        const uint64_t m = parameters[1];
        const uint64_t k = parameters[2];
        const uint64_t n = parameters[3];
        if (operand_id == SemanticOperandId::COMPUTE_INPUT_ADDRESS)
            elements = product({m, k});
        else if (operand_id == SemanticOperandId::COMPUTE_DATA_ADDRESS)
            elements = product({k, n});
        else
            elements = product({m, n});
    } else if (record.opcode == Opcode::ATTENTION) {
        if (parameters.size() != 5)
            Fail(path, "ATTENTION parameters must have five fields");
        const uint64_t tokens = parameters[1];
        const uint64_t qkv_width = parameters[2];
        const uint64_t heads = parameters[3];
        const uint64_t ratio = parameters[4];
        if (ratio == 0 || heads % ratio != 0)
            Fail(path, "ATTENTION head ratio is not integral");
        const uint64_t kv_heads = heads / ratio;
        const uint64_t width_factor = CheckedAdd(
            heads, CheckedMultiply(2, kv_heads, path), path);
        if (width_factor == 0 || qkv_width % width_factor != 0)
            Fail(path, "ATTENTION qkv width cannot derive head_dim");
        const uint64_t head_dim = qkv_width / width_factor;
        elements = operand_id == SemanticOperandId::COMPUTE_INPUT_ADDRESS
                       ? product({tokens, qkv_width})
                       : product({tokens, heads, head_dim});
    } else if (record.opcode == Opcode::SWIGLU) {
        if (parameters.size() != 1)
            Fail(path, "SWIGLU parameters must have one field");
        elements = operand_id == SemanticOperandId::COMPUTE_INPUT_ADDRESS
                       ? CheckedMultiply(2, parameters[0], path)
                       : parameters[0];
    } else if (record.opcode == Opcode::RESIDUAL) {
        if (parameters.size() != 1)
            Fail(path, "RESIDUAL parameters must have one field");
        elements = parameters[0];
    } else if (record.opcode == Opcode::RMSNORM) {
        if (parameters.size() != 3)
            Fail(path, "RMSNORM parameters must have three fields");
        elements = operand_id == SemanticOperandId::COMPUTE_DATA_ADDRESS
                       ? parameters[2]
                       : product({parameters[0], parameters[1], parameters[2]});
    } else {
        Fail(path, "not a compute record");
    }
    return CheckedMultiply(elements, 2, path);
}

uint64_t OperandAccessBytes(const RelocatableRecordDto &record,
                            SemanticOperandId operand_id,
                            const std::string &path) {
    switch (record.opcode) {
    case Opcode::MATMUL:
    case Opcode::ATTENTION:
    case Opcode::SWIGLU:
    case Opcode::RESIDUAL:
    case Opcode::RMSNORM:
        return ComputeOperandBytes(record, operand_id, path);
    case Opcode::ROPE_QK_EXACT: {
        if (operand_id != SemanticOperandId::COMPUTE_INPUT_ADDRESS &&
            operand_id != SemanticOperandId::COMPUTE_OUTPUT_ADDRESS)
            Fail(path, "ROPE_QK_EXACT has no such payload operand");
        return CheckedMultiply(
            2, CheckedMultiply(
                   LiteralU64(record.operands[4], path),
                   CheckedMultiply(
                       CheckedAdd(
                           LiteralU64(record.operands[8], path),
                           CheckedMultiply(2,
                               LiteralU64(record.operands[9], path), path),
                           path),
                       LiteralU64(record.operands[10], path), path), path),
            path);
    }
    case Opcode::ATTENTION_EXACT: {
        const uint64_t query_tokens = LiteralU64(record.operands[6], path);
        const uint64_t rank_heads = LiteralU64(record.operands[10], path);
        const uint64_t rank_kv_heads = LiteralU64(record.operands[11], path);
        const uint64_t head_dim = LiteralU64(record.operands[12], path);
        uint64_t heads = rank_heads;
        if (operand_id == SemanticOperandId::COMPUTE_INPUT_ADDRESS)
            heads = CheckedAdd(
                heads, CheckedMultiply(2, rank_kv_heads, path), path);
        else if (operand_id != SemanticOperandId::COMPUTE_OUTPUT_ADDRESS)
            Fail(path, "ATTENTION_EXACT has no such payload operand");
        return CheckedMultiply(
            2, CheckedMultiply(query_tokens,
                CheckedMultiply(heads, head_dim, path), path), path);
    }
    case Opcode::EMBEDDING_LOOKUP: {
        const uint64_t rank_rows = LiteralU64(record.operands[8], path);
        if (operand_id == SemanticOperandId::COMPUTE_INPUT_ADDRESS)
            return CheckedMultiply(4, rank_rows, path);
        if (operand_id == SemanticOperandId::COMPUTE_DATA_ADDRESS ||
            operand_id == SemanticOperandId::COMPUTE_OUTPUT_ADDRESS)
            return CheckedMultiply(
                2, CheckedMultiply(rank_rows,
                    LiteralU64(record.operands[11], path), path), path);
        Fail(path, "EMBEDDING_LOOKUP has no such payload operand");
    }
    case Opcode::GREEDY_SAMPLE: {
        const uint64_t samples = LiteralU64(record.operands[9], path);
        if (operand_id == SemanticOperandId::COMPUTE_INPUT_ADDRESS)
            return CheckedMultiply(
                2, CheckedMultiply(samples,
                    LiteralU64(record.operands[8], path), path), path);
        if (operand_id == SemanticOperandId::COMPUTE_OUTPUT_ADDRESS)
            return CheckedMultiply(4, samples, path);
        Fail(path, "GREEDY_SAMPLE has no such payload operand");
    }
    case Opcode::CROSS_ENTROPY_FORWARD: {
        const uint64_t rank_rows = LiteralU64(record.operands[8], path);
        if (operand_id == SemanticOperandId::COMPUTE_INPUT_ADDRESS)
            return CheckedMultiply(
                2, CheckedMultiply(rank_rows,
                    LiteralU64(record.operands[10], path), path), path);
        if (operand_id == SemanticOperandId::COMPUTE_DATA_ADDRESS ||
            operand_id == SemanticOperandId::COMPUTE_OUTPUT_ADDRESS)
            return CheckedMultiply(4, rank_rows, path);
        Fail(path, "CROSS_ENTROPY_FORWARD has no such payload operand");
    }
    case Opcode::CROSS_ENTROPY_BACKWARD: {
        const uint64_t rank_rows = LiteralU64(record.operands[11], path);
        if (operand_id == SemanticOperandId::COMPUTE_INPUT_ADDRESS ||
            operand_id == SemanticOperandId::COMPUTE_OUTPUT_ADDRESS)
            return CheckedMultiply(
                2, CheckedMultiply(rank_rows,
                    LiteralU64(record.operands[13], path), path), path);
        if (operand_id == SemanticOperandId::COMPUTE_DATA_ADDRESS)
            return CheckedMultiply(4, rank_rows, path);
        if (operand_id == SemanticOperandId::COMPUTE_AUX_ADDRESS)
            return CheckedMultiply(
                4, LiteralU64(record.operands[14], path), path);
        Fail(path, "CROSS_ENTROPY_BACKWARD has no such payload operand");
    }
    case Opcode::SGD_UPDATE: {
        const uint64_t elements = LiteralU64(record.operands[7], path);
        if (operand_id == SemanticOperandId::COMPUTE_INPUT_ADDRESS ||
            operand_id == SemanticOperandId::COMPUTE_OUTPUT_ADDRESS)
            return CheckedMultiply(2, elements, path);
        if (operand_id == SemanticOperandId::COMPUTE_DATA_ADDRESS)
            return CheckedMultiply(4, elements, path);
        Fail(path, "SGD_UPDATE has no such payload operand");
    }
    case Opcode::DTE_SEND:
        return LiteralU64(record.operands[7], path + ".length_bytes");
    case Opcode::DTE_RECV:
        return LiteralU64(record.operands[6], path + ".length_bytes");
    case Opcode::DTE_ISSUE:
        return LiteralU64(record.operands[3], path + ".size_bytes");
    case Opcode::LSU_LOAD:
    case Opcode::LSU_STORE:
        return LiteralU64(record.operands[1], path + ".size_bytes");
    case Opcode::LOCAL_REDUCE: {
        const uint64_t elements =
            LiteralU64(record.operands[7], path + ".element_count");
        const uint64_t dtype =
            LiteralU64(record.operands[0], path + ".input_dtype");
        if (dtype > 1)
            Fail(path, "LOCAL_REDUCE input_dtype is invalid");
        const uint64_t one_input =
            CheckedMultiply(elements, dtype == 1 ? 4 : 2, path);
        if (operand_id == SemanticOperandId::DESTINATION_ADDRESS)
            return one_input;
        const uint64_t count =
            LiteralU64(record.operands[6], path + ".input_count");
        const uint64_t stride =
            LiteralU64(record.operands[8], path + ".input_stride_bytes");
        if (count == 0)
            Fail(path, "LOCAL_REDUCE input_count must be positive");
        return CheckedAdd(CheckedMultiply(count - 1, stride, path),
                          one_input, path);
    }
    default:
        Fail(path, "record has no physical payload address extent");
    }
}

std::optional<BufferDTypeDto> ExpectedBufferDType(
    const RelocatableRecordDto &record, SemanticOperandId operand_id) {
    switch (record.opcode) {
    case Opcode::MATMUL:
    case Opcode::ATTENTION:
    case Opcode::SWIGLU:
    case Opcode::RESIDUAL:
    case Opcode::RMSNORM:
    case Opcode::ROPE_QK_EXACT:
    case Opcode::ATTENTION_EXACT:
        return BufferDTypeDto::FP16;
    case Opcode::EMBEDDING_LOOKUP:
        return operand_id == SemanticOperandId::COMPUTE_INPUT_ADDRESS
                   ? BufferDTypeDto::INT32
                   : BufferDTypeDto::FP16;
    case Opcode::GREEDY_SAMPLE:
        return operand_id == SemanticOperandId::COMPUTE_OUTPUT_ADDRESS
                   ? BufferDTypeDto::INT32
                   : BufferDTypeDto::FP16;
    case Opcode::CROSS_ENTROPY_FORWARD:
        if (operand_id == SemanticOperandId::COMPUTE_INPUT_ADDRESS)
            return BufferDTypeDto::FP16;
        if (operand_id == SemanticOperandId::COMPUTE_DATA_ADDRESS)
            return BufferDTypeDto::INT32;
        if (operand_id == SemanticOperandId::COMPUTE_OUTPUT_ADDRESS)
            return BufferDTypeDto::FP32;
        return std::nullopt;
    case Opcode::CROSS_ENTROPY_BACKWARD:
        if (operand_id == SemanticOperandId::COMPUTE_INPUT_ADDRESS ||
            operand_id == SemanticOperandId::COMPUTE_OUTPUT_ADDRESS)
            return BufferDTypeDto::FP16;
        if (operand_id == SemanticOperandId::COMPUTE_DATA_ADDRESS)
            return BufferDTypeDto::INT32;
        if (operand_id == SemanticOperandId::COMPUTE_AUX_ADDRESS)
            return BufferDTypeDto::FP32;
        return std::nullopt;
    case Opcode::SGD_UPDATE:
        if (operand_id == SemanticOperandId::COMPUTE_INPUT_ADDRESS ||
            operand_id == SemanticOperandId::COMPUTE_OUTPUT_ADDRESS)
            return BufferDTypeDto::FP16;
        if (operand_id == SemanticOperandId::COMPUTE_DATA_ADDRESS)
            return BufferDTypeDto::FP32;
        return std::nullopt;
    case Opcode::LOCAL_REDUCE:
        return LiteralU64(record.operands[0], "LOCAL_REDUCE.input_dtype") == 1
                   ? BufferDTypeDto::FP32
                   : BufferDTypeDto::FP16;
    default:
        return std::nullopt;
    }
}

SemanticRelocationKind RelocationKind(ProgramSymbolKind kind) {
    return static_cast<SemanticRelocationKind>(static_cast<uint8_t>(kind));
}

using RelocationMap = std::map<SemanticOperandId, AddressRelocationDto>;
using RuntimeRelocationMap =
    std::map<RuntimeOperandFieldDto, RuntimeRelocationDto>;

RuntimeSymbolKindDto RuntimeKindForField(RuntimeOperandFieldDto field) {
    switch (field) {
    case RuntimeOperandFieldDto::START_TAG:
        return RuntimeSymbolKindDto::START_TAG;
    case RuntimeOperandFieldDto::EVENT_TAG:
        return RuntimeSymbolKindDto::EVENT_TAG;
    case RuntimeOperandFieldDto::DTE_TOKEN:
        return RuntimeSymbolKindDto::DTE_TOKEN;
    case RuntimeOperandFieldDto::DTE_FSM:
        return RuntimeSymbolKindDto::DTE_FSM;
    case RuntimeOperandFieldDto::GROUP_ID:
        return RuntimeSymbolKindDto::GROUP;
    case RuntimeOperandFieldDto::SOURCE_CORE:
    case RuntimeOperandFieldDto::DESTINATION_CORE:
    case RuntimeOperandFieldDto::PEER_CORE:
        return RuntimeSymbolKindDto::RUNTIME_CORE;
    }
    Fail("runtime_relocation.field", "unknown runtime operand field");
}

RuntimeRelocationMap ValidateRuntimeRecordRelocations(
    const RelocatableRecordDto &record, uint64_t record_index,
    const CoreFragmentStreamDto &stream,
    const std::map<std::string, RuntimeEntry> &runtime_symbols,
    const LogicalCoreDto &logical_core, const std::string &path) {
    RuntimeRelocationMap result;
    for (const RuntimeRelocationDto &relocation : stream.runtime_relocations) {
        if (relocation.record_index != record_index)
            continue;
        if (!result.emplace(relocation.field, relocation).second)
            Fail(path, "duplicate runtime relocation target");
        const auto symbol = runtime_symbols.find(relocation.symbol_ref);
        if (symbol == runtime_symbols.end())
            Fail(path, "runtime relocation references an unknown symbol");
        if (symbol->second.definition->symbol.kind !=
            RuntimeKindForField(relocation.field))
            Fail(path, "runtime relocation symbol kind mismatch");
        if (symbol->second.definition->symbol.kind !=
                RuntimeSymbolKindDto::RUNTIME_CORE &&
            !ContainsCore(symbol->second.definition->logical_cores,
                          logical_core))
            Fail(path, "runtime symbol definition omits the executing core");
    }
    std::set<RuntimeOperandFieldDto> fields;
    for (const RecordOperandDto &operand : record.operands) {
        if (operand.kind != OperandKindDto::RUNTIME_SYMBOL)
            continue;
        if (!operand.runtime_field || !operand.symbol_ref)
            Fail(path, "incomplete runtime operand");
        if (!fields.insert(*operand.runtime_field).second)
            Fail(path, "duplicate runtime operand field");
        const auto relocation = result.find(*operand.runtime_field);
        if (relocation == result.end() ||
            relocation->second.symbol_ref != *operand.symbol_ref)
            Fail(path, "every runtime operand needs one exact relocation");
    }
    if (fields.size() != result.size())
        Fail(path, "runtime relocation has no matching operand");
    return result;
}

RelocationMap ValidateRecordRelocations(
    const RelocatableRecordDto &record, uint64_t record_index,
    const CoreFragmentStreamDto &stream,
    const std::map<std::string, SymbolEntry> &symbols,
    const LogicalCoreDto &logical_core, const std::string &path) {
    RelocationMap result;
    for (const AddressRelocationDto &relocation : stream.address_relocations) {
        if (relocation.record_index != record_index)
            continue;
        if (!result.emplace(relocation.operand_id, relocation).second)
            Fail(path, "duplicate address relocation target");
        const auto symbol = symbols.find(relocation.symbol_ref);
        if (symbol == symbols.end())
            Fail(path, "address relocation references an unknown symbol");
        if (symbol->second.definition->symbol.kind != relocation.symbol_kind)
            Fail(path, "address relocation symbol kind mismatch");
        if (!ContainsCore(symbol->second.definition->logical_cores,
                          logical_core))
            Fail(path, "program symbol definition omits the executing core");
    }
    std::set<SemanticOperandId> operand_ids;
    for (const RecordOperandDto &operand : record.operands) {
        if (operand.kind != OperandKindDto::ADDRESS_SYMBOL)
            continue;
        if (!operand.operand_id || !operand.symbol_ref)
            Fail(path, "incomplete address operand");
        if (!operand_ids.insert(*operand.operand_id).second)
            Fail(path, "duplicate address operand id");
        const auto relocation = result.find(*operand.operand_id);
        if (relocation == result.end() ||
            relocation->second.symbol_ref != *operand.symbol_ref)
            Fail(path, "every address operand needs one exact relocation");
    }
    if (operand_ids.size() != result.size())
        Fail(path, "address relocation has no matching operand");
    return result;
}

void AppendRelocation(ProgramArtifact &artifact, uint64_t core_index,
                      uint64_t instruction_index, SemanticOperandId operand_id,
                      const AddressRelocationDto &source,
                      const std::map<std::string, SymbolEntry> &symbols) {
    const SymbolEntry &symbol = symbols.at(source.symbol_ref);
    artifact.relocations.push_back(
        {core_index, instruction_index, static_cast<uint16_t>(operand_id),
         RelocationKind(source.symbol_kind), symbol.index, source.addend});
}

ExternalRecord FinalizeRecord(
    const RelocatableRecordDto &record, const RelocationMap &relocations,
    const RuntimeRelocationMap &runtime_relocations,
    const std::map<std::string, SymbolEntry> &symbols,
    const std::map<std::string, RuntimeEntry> &runtime_symbols,
    ProgramArtifact &artifact,
    uint64_t core_index, uint64_t instruction_index,
    const std::string &path) {
    ExternalRecord result;
    result.opcode = record.opcode;
    auto relocation = [&](SemanticOperandId id) -> const AddressRelocationDto & {
        const auto found = relocations.find(id);
        if (found == relocations.end())
            Fail(path, "missing canonical relocation");
        return found->second;
    };
    auto symbol = [&](SemanticOperandId id) -> const SymbolEntry & {
        return symbols.at(relocation(id).symbol_ref);
    };
    auto runtime_value = [&](const RecordOperandDto &operand,
                             std::string_view name,
                             RuntimeOperandFieldDto field,
                             const std::string &operand_path) -> uint64_t {
        RequireRuntime(operand, name, field, operand_path);
        const auto relocation_it = runtime_relocations.find(field);
        if (relocation_it == runtime_relocations.end() ||
            relocation_it->second.symbol_ref != *operand.symbol_ref)
            Fail(operand_path, "missing exact runtime relocation");
        return runtime_symbols.at(*operand.symbol_ref).value;
    };
    auto token_or_literal = [&](const RecordOperandDto &operand,
                                std::string_view name,
                                RuntimeOperandFieldDto field,
                                const std::string &operand_path) -> uint64_t {
        if (operand.kind == OperandKindDto::LITERAL) {
            RequireLiteral(operand, name, operand_path);
            return LiteralU64(operand, operand_path);
        }
        return runtime_value(operand, name, field, operand_path);
    };
    auto address = [&](SemanticOperandId id) -> SramAddressOperand {
        const AddressRelocationDto &source = relocation(id);
        const SymbolEntry &entry = symbol(id);
        SramAddressOperand result_address;
        if (entry.definition->symbol.kind ==
            ProgramSymbolKind::ABSOLUTE_ADDRESS) {
            result_address.kind = SramAddressKind::ABSOLUTE;
            result_address.absolute_address_bytes = ApplyAddend(
                entry.definition->value, source.addend, path);
        } else if (entry.definition->symbol.kind ==
                   ProgramSymbolKind::SRAM_REGION) {
            if (source.addend < 0)
                Fail(path, "SRAM_REGION relocation addend must be non-negative");
            result_address.kind = SramAddressKind::REGION;
            result_address.region_symbol_index = entry.index;
            result_address.region_offset_bytes =
                static_cast<uint64_t>(source.addend);
        } else {
            Fail(path, "SRAM address requires ABSOLUTE_ADDRESS or SRAM_REGION");
        }
        AppendRelocation(artifact, core_index, instruction_index, id, source,
                         symbols);
        return result_address;
    };
    auto absolute_address = [&](SemanticOperandId id) -> SramAddressOperand {
        if (symbol(id).definition->symbol.kind !=
            ProgramSymbolKind::ABSOLUTE_ADDRESS)
            Fail(path, "fixed NPU address relocation requires ABSOLUTE_ADDRESS");
        return address(id);
    };

    if (record.opcode == Opcode::SRAM_ALLOC_AT) {
        if (record.operands.size() != 7)
            Fail(path + ".operands", "SRAM_ALLOC_AT requires seven operands");
        RequireAddress(record.operands[0], "region_name",
                       SemanticOperandId::REGION_NAME, path + ".operands[0]");
        RequireAddress(record.operands[1], "label_symbol",
                       SemanticOperandId::LABEL_SYMBOL, path + ".operands[1]");
        RequireLiteral(record.operands[2], "region_offset_bytes", path + ".operands[2]");
        RequireLiteral(record.operands[3], "size_bytes", path + ".operands[3]");
        RequireLiteral(record.operands[4], "alignment_bytes", path + ".operands[4]");
        RequireLiteral(record.operands[5], "lifetime", path + ".operands[5]");
        RequireLiteral(record.operands[6], "spillable", path + ".operands[6]");
        const SymbolEntry &region = symbol(SemanticOperandId::REGION_NAME);
        const SymbolEntry &label = symbol(SemanticOperandId::LABEL_SYMBOL);
        if (region.definition->symbol.kind != ProgramSymbolKind::SRAM_REGION ||
            label.definition->symbol.kind != ProgramSymbolKind::SRAM_LABEL ||
            relocation(SemanticOperandId::REGION_NAME).addend != 0 ||
            relocation(SemanticOperandId::LABEL_SYMBOL).addend != 0)
            Fail(path, "SRAM_ALLOC_AT requires exact REGION_NAME/SRAM_REGION and LABEL_SYMBOL/SRAM_LABEL relocations with zero addends");
        const uint64_t offset = LiteralU64(record.operands[2], path + ".operands[2]");
        const uint64_t size = LiteralU64(record.operands[3], path + ".operands[3]");
        const uint64_t alignment = LiteralU64(record.operands[4], path + ".operands[4]");
        const uint64_t lifetime = LiteralU64(record.operands[5], path + ".operands[5]");
        const bool spillable = LiteralBool(record.operands[6], path + ".operands[6]");
        if (size == 0 || alignment == 0 ||
            (alignment & (alignment - 1)) != 0 || lifetime != 0)
            Fail(path, "invalid SRAM_ALLOC_AT size/alignment/lifetime");
        if (offset > region.definition->size_bytes ||
            size > region.definition->size_bytes - offset)
            Fail(path, "SRAM_ALLOC_AT span exceeds its named region");
        if (region.definition->value >
            std::numeric_limits<uint64_t>::max() - offset)
            Fail(path, "SRAM_ALLOC_AT absolute start overflows uint64");
        if ((region.definition->value + offset) % alignment != 0)
            Fail(path, "SRAM_ALLOC_AT absolute start violates alignment");
        SramAllocAtOperands operands;
        operands.region_name_string_index = region.index;
        operands.label_symbol_index = label.index;
        operands.region_offset_bytes = offset;
        operands.size_bytes = size;
        operands.alignment_bytes = alignment;
        operands.lifetime = SramLifetime::TASK;
        operands.spillable = spillable;
        result.operands = operands;
        AppendRelocation(artifact, core_index, instruction_index,
                         SemanticOperandId::REGION_NAME,
                         relocation(SemanticOperandId::REGION_NAME), symbols);
        AppendRelocation(artifact, core_index, instruction_index,
                         SemanticOperandId::LABEL_SYMBOL,
                         relocation(SemanticOperandId::LABEL_SYMBOL), symbols);
    } else if (record.opcode == Opcode::SRAM_BIND) {
        if (record.operands.size() != 18)
            Fail(path + ".operands", "SRAM_BIND requires eighteen operands");
        RequireLiteral(record.operands[0], "input_count", path + ".operands[0]");
        const uint64_t count = LiteralU64(record.operands[0], path + ".operands[0]");
        if (count < 1 || count > 3)
            Fail(path + ".operands[0]",
                 "SRAM_BIND input_count must be one, two, or three");
        SramBindOperands operands;
        operands.input_count = count;
        for (std::size_t index = 0; index < kSramBindInputLimit; ++index) {
            const std::string name = "input_label_" + std::to_string(index);
            const SemanticOperandId id = static_cast<SemanticOperandId>(
                static_cast<uint16_t>(SemanticOperandId::SRAM_BIND_INPUT_0) +
                index);
            if (index < count) {
                RequireAddress(record.operands[index + 1], name, id,
                               path + ".operands[" +
                                   std::to_string(index + 1) + "]");
                if (symbol(id).definition->symbol.kind !=
                        ProgramSymbolKind::SRAM_LABEL ||
                    relocation(id).addend != 0)
                    Fail(path, "active SRAM_BIND inputs require zero-addend SRAM_LABEL relocations");
                operands.input_symbol_indices[index] = symbol(id).index;
                AppendRelocation(artifact, core_index, instruction_index, id,
                                 relocation(id), symbols);
            } else {
                RequireLiteral(record.operands[index + 1], name,
                               path + ".operands[" +
                                   std::to_string(index + 1) + "]");
                if (LiteralU64(record.operands[index + 1], path) != 0)
                    Fail(path, "inactive SRAM_BIND slots must be literal zero");
            }
        }
        RequireAddress(record.operands[17], "output_label",
                       SemanticOperandId::SRAM_BIND_OUTPUT,
                       path + ".operands[17]");
        if (symbol(SemanticOperandId::SRAM_BIND_OUTPUT).definition->symbol.kind !=
                ProgramSymbolKind::SRAM_LABEL ||
            relocation(SemanticOperandId::SRAM_BIND_OUTPUT).addend != 0)
            Fail(path, "SRAM_BIND output requires a zero-addend SRAM_LABEL relocation");
        operands.output_symbol_index = symbol(SemanticOperandId::SRAM_BIND_OUTPUT).index;
        AppendRelocation(artifact, core_index, instruction_index,
                         SemanticOperandId::SRAM_BIND_OUTPUT,
                         relocation(SemanticOperandId::SRAM_BIND_OUTPUT), symbols);
        result.operands = operands;
    } else if (record.opcode == Opcode::ROPE_QK_EXACT) {
        if (record.operands.size() != 15)
            Fail(path + ".operands", "ROPE_QK_EXACT requires fifteen operands");
        static constexpr std::array<const char *, 15> names{{
            "datatype", "packed_layout", "input_address", "output_address",
            "logical_tokens", "tp_degree", "num_heads", "num_kv_heads",
            "rank_num_heads", "rank_num_kv_heads", "head_dim", "rotary_dim",
            "max_position_embeddings", "context_max", "rope_theta_f64_bits"}};
        for (std::size_t index = 0; index < names.size(); ++index) {
            if (index == 2 || index == 3)
                continue;
            RequireLiteral(record.operands[index], names[index],
                           path + ".operands[" + std::to_string(index) + "]");
        }
        RequireAddress(record.operands[2], names[2],
                       SemanticOperandId::COMPUTE_INPUT_ADDRESS,
                       path + ".operands[2]");
        RequireAddress(record.operands[3], names[3],
                       SemanticOperandId::COMPUTE_OUTPUT_ADDRESS,
                       path + ".operands[3]");
        RopeQkExactOperands operands;
        operands.datatype = LiteralEnum<ExternalDataType>(record.operands[0], path);
        operands.packed_layout = LiteralEnum<RopePackedLayout>(record.operands[1], path);
        operands.input = absolute_address(SemanticOperandId::COMPUTE_INPUT_ADDRESS);
        operands.output = absolute_address(SemanticOperandId::COMPUTE_OUTPUT_ADDRESS);
        operands.logical_tokens = LiteralU64(record.operands[4], path);
        operands.tp_degree = LiteralU64(record.operands[5], path);
        operands.num_heads = LiteralU64(record.operands[6], path);
        operands.num_kv_heads = LiteralU64(record.operands[7], path);
        operands.rank_num_heads = LiteralU64(record.operands[8], path);
        operands.rank_num_kv_heads = LiteralU64(record.operands[9], path);
        operands.head_dim = LiteralU64(record.operands[10], path);
        operands.rotary_dim = LiteralU64(record.operands[11], path);
        operands.max_position_embeddings = LiteralU64(record.operands[12], path);
        operands.context_max = LiteralU64(record.operands[13], path);
        operands.rope_theta_f64_bits = LiteralU64(record.operands[14], path);
        result.operands = operands;
    } else if (record.opcode == Opcode::ATTENTION_EXACT) {
        if (record.operands.size() != 18)
            Fail(path + ".operands", "ATTENTION_EXACT requires eighteen operands");
        static constexpr std::array<const char *, 18> names{{
            "datatype", "mode", "packed_layout", "causal", "input_address",
            "output_address", "query_tokens", "tp_degree", "num_heads",
            "num_kv_heads", "rank_num_heads", "rank_num_kv_heads", "head_dim",
            "context_sum", "context_max", "query_key_pairs",
            "rank_kv_read_bytes", "rank_kv_write_bytes"}};
        for (std::size_t index = 0; index < names.size(); ++index) {
            if (index == 4 || index == 5)
                continue;
            RequireLiteral(record.operands[index], names[index],
                           path + ".operands[" + std::to_string(index) + "]");
        }
        RequireAddress(record.operands[4], names[4],
                       SemanticOperandId::COMPUTE_INPUT_ADDRESS,
                       path + ".operands[4]");
        RequireAddress(record.operands[5], names[5],
                       SemanticOperandId::COMPUTE_OUTPUT_ADDRESS,
                       path + ".operands[5]");
        AttentionExactOperands operands;
        operands.datatype = LiteralEnum<ExternalDataType>(record.operands[0], path);
        operands.mode = LiteralEnum<ExactAttentionMode>(record.operands[1], path);
        operands.packed_layout = LiteralEnum<AttentionPackedLayout>(record.operands[2], path);
        operands.causal = LiteralBool(record.operands[3], path);
        operands.input = absolute_address(SemanticOperandId::COMPUTE_INPUT_ADDRESS);
        operands.output = absolute_address(SemanticOperandId::COMPUTE_OUTPUT_ADDRESS);
        operands.query_tokens = LiteralU64(record.operands[6], path);
        operands.tp_degree = LiteralU64(record.operands[7], path);
        operands.num_heads = LiteralU64(record.operands[8], path);
        operands.num_kv_heads = LiteralU64(record.operands[9], path);
        operands.rank_num_heads = LiteralU64(record.operands[10], path);
        operands.rank_num_kv_heads = LiteralU64(record.operands[11], path);
        operands.head_dim = LiteralU64(record.operands[12], path);
        operands.context_sum = LiteralU64(record.operands[13], path);
        operands.context_max = LiteralU64(record.operands[14], path);
        operands.query_key_pairs = LiteralU64(record.operands[15], path);
        operands.rank_kv_read_bytes = LiteralU64(record.operands[16], path);
        operands.rank_kv_write_bytes = LiteralU64(record.operands[17], path);
        result.operands = operands;
    } else if (record.opcode == Opcode::EMBEDDING_LOOKUP) {
        if (record.operands.size() != 12)
            Fail(path + ".operands", "EMBEDDING_LOOKUP requires twelve operands");
        static constexpr std::array<const char *, 12> names{{
            "index_datatype", "table_datatype", "output_datatype", "placement",
            "indices_address", "table_address", "output_address", "logical_rows",
            "rank_rows", "tp_degree", "vocab_size", "hidden_size"}};
        for (std::size_t index = 0; index < names.size(); ++index) {
            if (index >= 4 && index <= 6)
                continue;
            RequireLiteral(record.operands[index], names[index],
                           path + ".operands[" + std::to_string(index) + "]");
        }
        RequireAddress(record.operands[4], names[4],
                       SemanticOperandId::COMPUTE_INPUT_ADDRESS, path + ".operands[4]");
        RequireAddress(record.operands[5], names[5],
                       SemanticOperandId::COMPUTE_DATA_ADDRESS, path + ".operands[5]");
        RequireAddress(record.operands[6], names[6],
                       SemanticOperandId::COMPUTE_OUTPUT_ADDRESS, path + ".operands[6]");
        EmbeddingLookupOperands operands;
        operands.index_datatype = LiteralEnum<ExternalDataType>(record.operands[0], path);
        operands.table_datatype = LiteralEnum<ExternalDataType>(record.operands[1], path);
        operands.output_datatype = LiteralEnum<ExternalDataType>(record.operands[2], path);
        operands.placement = LiteralEnum<EmbeddingPlacement>(record.operands[3], path);
        operands.indices = absolute_address(SemanticOperandId::COMPUTE_INPUT_ADDRESS);
        operands.table = absolute_address(SemanticOperandId::COMPUTE_DATA_ADDRESS);
        operands.output = absolute_address(SemanticOperandId::COMPUTE_OUTPUT_ADDRESS);
        operands.logical_rows = LiteralU64(record.operands[7], path);
        operands.rank_rows = LiteralU64(record.operands[8], path);
        operands.tp_degree = LiteralU64(record.operands[9], path);
        operands.vocab_size = LiteralU64(record.operands[10], path);
        operands.hidden_size = LiteralU64(record.operands[11], path);
        result.operands = operands;
    } else if (record.opcode == Opcode::GREEDY_SAMPLE) {
        if (record.operands.size() != 11)
            Fail(path + ".operands", "GREEDY_SAMPLE requires eleven operands");
        static constexpr std::array<const char *, 11> names{{
            "logits_datatype", "output_datatype", "mode", "row_selection",
            "logits_address", "output_address", "tp_degree", "token_rows",
            "vocab_size", "sample_count", "comparisons"}};
        for (std::size_t index = 0; index < names.size(); ++index) {
            if (index == 4 || index == 5)
                continue;
            RequireLiteral(record.operands[index], names[index],
                           path + ".operands[" + std::to_string(index) + "]");
        }
        RequireAddress(record.operands[4], names[4],
                       SemanticOperandId::COMPUTE_INPUT_ADDRESS, path + ".operands[4]");
        RequireAddress(record.operands[5], names[5],
                       SemanticOperandId::COMPUTE_OUTPUT_ADDRESS, path + ".operands[5]");
        GreedySampleOperands operands;
        operands.logits_datatype = LiteralEnum<ExternalDataType>(record.operands[0], path);
        operands.output_datatype = LiteralEnum<ExternalDataType>(record.operands[1], path);
        operands.mode = LiteralEnum<GreedySampleMode>(record.operands[2], path);
        operands.row_selection = LiteralEnum<GreedyRowSelection>(record.operands[3], path);
        operands.logits = absolute_address(SemanticOperandId::COMPUTE_INPUT_ADDRESS);
        operands.output = absolute_address(SemanticOperandId::COMPUTE_OUTPUT_ADDRESS);
        operands.tp_degree = LiteralU64(record.operands[6], path);
        operands.token_rows = LiteralU64(record.operands[7], path);
        operands.vocab_size = LiteralU64(record.operands[8], path);
        operands.sample_count = LiteralU64(record.operands[9], path);
        operands.comparisons = LiteralU64(record.operands[10], path);
        result.operands = operands;
    } else if (record.opcode == Opcode::CROSS_ENTROPY_FORWARD) {
        if (record.operands.size() != 11)
            Fail(path + ".operands",
                 "CROSS_ENTROPY_FORWARD requires eleven operands");
        static constexpr std::array<const char *, 11> names{{
            "logits_datatype", "label_datatype", "loss_datatype",
            "reduction", "logits_address", "labels_address",
            "loss_address", "logical_rows", "rank_rows", "tp_degree",
            "vocab_size"}};
        for (std::size_t index = 0; index < names.size(); ++index) {
            if (index >= 4 && index <= 6)
                continue;
            RequireLiteral(record.operands[index], names[index],
                           path + ".operands[" + std::to_string(index) + "]");
        }
        RequireAddress(record.operands[4], names[4],
                       SemanticOperandId::COMPUTE_INPUT_ADDRESS,
                       path + ".operands[4]");
        RequireAddress(record.operands[5], names[5],
                       SemanticOperandId::COMPUTE_DATA_ADDRESS,
                       path + ".operands[5]");
        RequireAddress(record.operands[6], names[6],
                       SemanticOperandId::COMPUTE_OUTPUT_ADDRESS,
                       path + ".operands[6]");
        CrossEntropyForwardOperands operands;
        operands.logits_datatype =
            LiteralEnum<ExternalDataType>(record.operands[0], path);
        operands.label_datatype =
            LiteralEnum<ExternalDataType>(record.operands[1], path);
        operands.loss_datatype =
            LiteralEnum<ExternalDataType>(record.operands[2], path);
        operands.reduction =
            LiteralEnum<CrossEntropyReduction>(record.operands[3], path);
        operands.logits =
            absolute_address(SemanticOperandId::COMPUTE_INPUT_ADDRESS);
        operands.labels =
            absolute_address(SemanticOperandId::COMPUTE_DATA_ADDRESS);
        operands.loss =
            absolute_address(SemanticOperandId::COMPUTE_OUTPUT_ADDRESS);
        operands.logical_rows = LiteralU64(record.operands[7], path);
        operands.rank_rows = LiteralU64(record.operands[8], path);
        operands.tp_degree = LiteralU64(record.operands[9], path);
        operands.vocab_size = LiteralU64(record.operands[10], path);
        result.operands = operands;
    } else if (record.opcode == Opcode::CROSS_ENTROPY_BACKWARD) {
        if (record.operands.size() != 15)
            Fail(path + ".operands",
                 "CROSS_ENTROPY_BACKWARD requires fifteen operands");
        static constexpr std::array<const char *, 15> names{{
            "logits_datatype", "label_datatype", "upstream_datatype",
            "output_datatype", "reduction", "upstream_mode",
            "logits_address", "labels_address", "upstream_address",
            "logits_grad_address", "logical_rows", "rank_rows",
            "tp_degree", "vocab_size", "upstream_elements"}};
        for (std::size_t index = 0; index < names.size(); ++index) {
            if (index >= 6 && index <= 9)
                continue;
            RequireLiteral(record.operands[index], names[index],
                           path + ".operands[" + std::to_string(index) + "]");
        }
        RequireAddress(record.operands[6], names[6],
                       SemanticOperandId::COMPUTE_INPUT_ADDRESS,
                       path + ".operands[6]");
        RequireAddress(record.operands[7], names[7],
                       SemanticOperandId::COMPUTE_DATA_ADDRESS,
                       path + ".operands[7]");
        RequireAddress(record.operands[8], names[8],
                       SemanticOperandId::COMPUTE_AUX_ADDRESS,
                       path + ".operands[8]");
        RequireAddress(record.operands[9], names[9],
                       SemanticOperandId::COMPUTE_OUTPUT_ADDRESS,
                       path + ".operands[9]");
        CrossEntropyBackwardOperands operands;
        operands.logits_datatype =
            LiteralEnum<ExternalDataType>(record.operands[0], path);
        operands.label_datatype =
            LiteralEnum<ExternalDataType>(record.operands[1], path);
        operands.upstream_datatype =
            LiteralEnum<ExternalDataType>(record.operands[2], path);
        operands.output_datatype =
            LiteralEnum<ExternalDataType>(record.operands[3], path);
        operands.reduction =
            LiteralEnum<CrossEntropyReduction>(record.operands[4], path);
        operands.upstream_mode =
            LiteralEnum<CrossEntropyUpstreamMode>(record.operands[5], path);
        operands.logits =
            absolute_address(SemanticOperandId::COMPUTE_INPUT_ADDRESS);
        operands.labels =
            absolute_address(SemanticOperandId::COMPUTE_DATA_ADDRESS);
        operands.upstream =
            absolute_address(SemanticOperandId::COMPUTE_AUX_ADDRESS);
        operands.logits_grad =
            absolute_address(SemanticOperandId::COMPUTE_OUTPUT_ADDRESS);
        operands.logical_rows = LiteralU64(record.operands[10], path);
        operands.rank_rows = LiteralU64(record.operands[11], path);
        operands.tp_degree = LiteralU64(record.operands[12], path);
        operands.vocab_size = LiteralU64(record.operands[13], path);
        operands.upstream_elements = LiteralU64(record.operands[14], path);
        result.operands = operands;
    } else if (record.opcode == Opcode::SGD_UPDATE) {
        if (record.operands.size() != 10)
            Fail(path + ".operands", "SGD_UPDATE requires ten operands");
        static constexpr std::array<const char *, 10> names{{
            "weight_datatype", "gradient_datatype", "output_datatype",
            "rounding", "weight_address", "gradient_address",
            "updated_weight_address", "element_count",
            "learning_rate_f64_bits", "momentum_f64_bits"}};
        for (std::size_t index = 0; index < names.size(); ++index) {
            if (index >= 4 && index <= 6)
                continue;
            RequireLiteral(record.operands[index], names[index],
                           path + ".operands[" + std::to_string(index) + "]");
        }
        RequireAddress(record.operands[4], names[4],
                       SemanticOperandId::COMPUTE_INPUT_ADDRESS,
                       path + ".operands[4]");
        RequireAddress(record.operands[5], names[5],
                       SemanticOperandId::COMPUTE_DATA_ADDRESS,
                       path + ".operands[5]");
        RequireAddress(record.operands[6], names[6],
                       SemanticOperandId::COMPUTE_OUTPUT_ADDRESS,
                       path + ".operands[6]");
        if (*record.operands[4].symbol_ref != *record.operands[6].symbol_ref)
            Fail(path,
                 "SGD_UPDATE weight and updated_weight must use one exact symbol");
        SgdUpdateOperands operands;
        operands.weight_datatype =
            LiteralEnum<ExternalDataType>(record.operands[0], path);
        operands.gradient_datatype =
            LiteralEnum<ExternalDataType>(record.operands[1], path);
        operands.output_datatype =
            LiteralEnum<ExternalDataType>(record.operands[2], path);
        operands.rounding =
            LiteralEnum<OptimizerRoundingMode>(record.operands[3], path);
        operands.weight =
            absolute_address(SemanticOperandId::COMPUTE_INPUT_ADDRESS);
        operands.gradient =
            absolute_address(SemanticOperandId::COMPUTE_DATA_ADDRESS);
        operands.updated_weight =
            absolute_address(SemanticOperandId::COMPUTE_OUTPUT_ADDRESS);
        operands.element_count = LiteralU64(record.operands[7], path);
        operands.learning_rate_f64_bits =
            LiteralU64(record.operands[8], path);
        operands.momentum_f64_bits = LiteralU64(record.operands[9], path);
        result.operands = operands;
    } else if (record.opcode == Opcode::MATMUL ||
               record.opcode == Opcode::ATTENTION ||
               record.opcode == Opcode::SWIGLU ||
               record.opcode == Opcode::RESIDUAL ||
               record.opcode == Opcode::RMSNORM) {
        if (record.operands.size() != 5)
            Fail(path + ".operands", "MATMUL requires five operands");
        RequireLiteral(record.operands[0], "datatype", path + ".operands[0]");
        RequireAddress(record.operands[1], "input_address",
                       SemanticOperandId::COMPUTE_INPUT_ADDRESS,
                       path + ".operands[1]");
        const bool has_data = record.opcode == Opcode::MATMUL ||
                              record.opcode == Opcode::RESIDUAL ||
                              record.opcode == Opcode::RMSNORM;
        if (has_data)
            RequireAddress(record.operands[2], "data_address",
                           SemanticOperandId::COMPUTE_DATA_ADDRESS,
                           path + ".operands[2]");
        else {
            RequireLiteral(record.operands[2], "data_address",
                           path + ".operands[2]");
            if (LiteralU64(record.operands[2], path + ".operands[2]") != 0)
                Fail(path + ".operands[2]",
                     "manual-data compute requires literal zero data_address");
        }
        RequireAddress(record.operands[3], "output_address",
                       SemanticOperandId::COMPUTE_OUTPUT_ADDRESS,
                       path + ".operands[3]");
        RequireLiteral(record.operands[4], "parameters", path + ".operands[4]");
        const uint64_t datatype = LiteralU64(record.operands[0], path + ".operands[0]");
        const std::vector<uint64_t> &parameters =
            LiteralU64Array(record.operands[4], path + ".operands[4]");
        if (datatype != 1 || parameters.empty() ||
            std::any_of(parameters.begin(), parameters.end(), [](uint64_t item) {
                return item == 0 || item > kExternalNpuParameterMax;
            }))
            Fail(path, "compute requires FP16 and positive <=30-bit parameters");
        ComputeOperands operands;
        operands.datatype = ExternalDataType::FP16;
        operands.parameters = parameters;
        std::vector<SemanticOperandId> ids{
            SemanticOperandId::COMPUTE_INPUT_ADDRESS};
        if (has_data)
            ids.push_back(SemanticOperandId::COMPUTE_DATA_ADDRESS);
        ids.push_back(SemanticOperandId::COMPUTE_OUTPUT_ADDRESS);
        std::map<SemanticOperandId, uint64_t> resolved;
        for (SemanticOperandId id : ids) {
            if (symbol(id).definition->symbol.kind !=
                ProgramSymbolKind::ABSOLUTE_ADDRESS)
                Fail(path, "compute address relocation requires ABSOLUTE_ADDRESS");
            const uint64_t address =
                ApplyAddend(symbol(id).definition->value,
                            relocation(id).addend, path);
            if (address > 0xffff)
                Fail(path, "compute relocated address cannot fit the uint16 wire");
            resolved.emplace(id, address);
            AppendRelocation(artifact, core_index, instruction_index, id,
                             relocation(id), symbols);
        }
        operands.input_offset_bytes =
            resolved.at(SemanticOperandId::COMPUTE_INPUT_ADDRESS);
        operands.data_offset_bytes = has_data
            ? resolved.at(SemanticOperandId::COMPUTE_DATA_ADDRESS) : 0;
        operands.output_offset_bytes =
            resolved.at(SemanticOperandId::COMPUTE_OUTPUT_ADDRESS);
        result.operands = operands;
    } else if (record.opcode == Opcode::DTE_SEND) {
        if (record.operands.size() != 15)
            Fail(path + ".operands", "DTE_SEND requires fifteen operands");
        static constexpr std::array<const char *, 8> literals{{
            "mode", "source_space", "completion", "datatype", "reduce_op",
            "length_bytes", "expected_sources", "tree_id"}};
        for (std::size_t i = 0; i < 5; ++i)
            RequireLiteral(record.operands[i], literals[i],
                           path + ".operands[" + std::to_string(i) + "]");
        RequireRuntime(record.operands[5], "fsm_id",
                       RuntimeOperandFieldDto::DTE_FSM,
                       path + ".operands[5]");
        DteSendOperands operands;
        operands.mode = LiteralEnum<DteSendMode>(record.operands[0], path);
        operands.source_space = LiteralEnum<EndpointSourceSpace>(record.operands[1], path);
        operands.completion = LiteralEnum<EndpointCompletion>(record.operands[2], path);
        operands.datatype = LiteralEnum<EndpointDataType>(record.operands[3], path);
        operands.reduce_op = LiteralEnum<ReduceOperator>(record.operands[4], path);
        operands.fsm_id = runtime_value(record.operands[5], "fsm_id",
                                        RuntimeOperandFieldDto::DTE_FSM, path);
        operands.token = token_or_literal(record.operands[6], "token",
                                          RuntimeOperandFieldDto::DTE_TOKEN, path);
        RequireLiteral(record.operands[7], "length_bytes", path + ".operands[7]");
        operands.length_bytes = LiteralU64(record.operands[7], path);
        RequireAddress(record.operands[8], "source_address",
                       SemanticOperandId::SOURCE_ADDRESS, path + ".operands[8]");
        operands.source = address(SemanticOperandId::SOURCE_ADDRESS);
        operands.peer_core = runtime_value(record.operands[9], "peer_core",
                                           RuntimeOperandFieldDto::PEER_CORE, path);
        RequireLiteral(record.operands[10], "expected_sources", path + ".operands[10]");
        RequireLiteral(record.operands[11], "tree_id", path + ".operands[11]");
        operands.expected_sources = LiteralU64(record.operands[10], path);
        operands.tree_id = LiteralU64(record.operands[11], path);
        operands.group_id = token_or_literal(record.operands[12], "group_id",
                                             RuntimeOperandFieldDto::GROUP_ID, path);
        RequireLiteral(record.operands[13], "collective_id", path + ".operands[13]");
        RequireLiteral(record.operands[14], "epoch", path + ".operands[14]");
        operands.collective_id = LiteralU64(record.operands[13], path);
        operands.epoch = LiteralU64(record.operands[14], path);
        result.operands = operands;
    } else if (record.opcode == Opcode::DTE_RECV) {
        if (record.operands.size() != 14)
            Fail(path + ".operands", "DTE_RECV requires fourteen operands");
        for (std::size_t i = 0; i < 4; ++i) {
            static constexpr std::array<const char *, 4> names{{
                "mode", "completion", "datatype", "reduce_op"}};
            RequireLiteral(record.operands[i], names[i],
                           path + ".operands[" + std::to_string(i) + "]");
        }
        DteRecvOperands operands;
        operands.mode = LiteralEnum<DteRecvMode>(record.operands[0], path);
        operands.completion = LiteralEnum<EndpointCompletion>(record.operands[1], path);
        operands.datatype = LiteralEnum<EndpointDataType>(record.operands[2], path);
        operands.reduce_op = LiteralEnum<ReduceOperator>(record.operands[3], path);
        operands.fsm_id = runtime_value(record.operands[4], "fsm_id",
                                        RuntimeOperandFieldDto::DTE_FSM, path);
        operands.token = token_or_literal(record.operands[5], "token",
                                          RuntimeOperandFieldDto::DTE_TOKEN, path);
        RequireLiteral(record.operands[6], "length_bytes", path + ".operands[6]");
        operands.length_bytes = LiteralU64(record.operands[6], path);
        RequireAddress(record.operands[7], "destination_address",
                       SemanticOperandId::DESTINATION_ADDRESS, path + ".operands[7]");
        operands.destination = address(SemanticOperandId::DESTINATION_ADDRESS);
        operands.peer_core = runtime_value(record.operands[8], "peer_core",
                                           RuntimeOperandFieldDto::PEER_CORE, path);
        RequireLiteral(record.operands[9], "expected_sources", path + ".operands[9]");
        RequireLiteral(record.operands[10], "tree_id", path + ".operands[10]");
        operands.expected_sources = LiteralU64(record.operands[9], path);
        operands.tree_id = LiteralU64(record.operands[10], path);
        operands.group_id = token_or_literal(record.operands[11], "group_id",
                                             RuntimeOperandFieldDto::GROUP_ID, path);
        RequireLiteral(record.operands[12], "collective_id", path + ".operands[12]");
        RequireLiteral(record.operands[13], "epoch", path + ".operands[13]");
        operands.collective_id = LiteralU64(record.operands[12], path);
        operands.epoch = LiteralU64(record.operands[13], path);
        result.operands = operands;
    } else if (record.opcode == Opcode::LOCAL_REDUCE) {
        if (record.operands.size() != 11)
            Fail(path + ".operands", "LOCAL_REDUCE requires eleven operands");
        static constexpr std::array<const char *, 9> names{{
            "input_dtype", "accumulator_dtype", "output_dtype", "reduce_op",
            "rounding", "order", "input_count", "element_count",
            "input_stride_bytes"}};
        for (std::size_t i = 0; i < names.size(); ++i)
            RequireLiteral(record.operands[i], names[i],
                           path + ".operands[" + std::to_string(i) + "]");
        RequireAddress(record.operands[9], "source_address",
                       SemanticOperandId::SOURCE_ADDRESS, path + ".operands[9]");
        RequireAddress(record.operands[10], "destination_address",
                       SemanticOperandId::DESTINATION_ADDRESS, path + ".operands[10]");
        LocalReduceOperands operands;
        operands.input_dtype = LiteralEnum<LocalReduceDataType>(record.operands[0], path);
        operands.accumulator_dtype = LiteralEnum<LocalReduceDataType>(record.operands[1], path);
        operands.output_dtype = LiteralEnum<LocalReduceDataType>(record.operands[2], path);
        operands.reduce_op = LiteralEnum<ReduceOperator>(record.operands[3], path);
        operands.rounding = LiteralEnum<LocalReduceRoundingMode>(record.operands[4], path);
        operands.order = LiteralEnum<LocalReduceOrder>(record.operands[5], path);
        operands.input_count = LiteralU64(record.operands[6], path);
        operands.element_count = LiteralU64(record.operands[7], path);
        operands.input_stride_bytes = LiteralU64(record.operands[8], path);
        operands.source = address(SemanticOperandId::SOURCE_ADDRESS);
        operands.destination = address(SemanticOperandId::DESTINATION_ADDRESS);
        result.operands = operands;
    } else if (record.opcode == Opcode::LSU_LOAD ||
               record.opcode == Opcode::LSU_STORE) {
        if (record.operands.size() != 3)
            Fail(path + ".operands", "blocking LSU requires three operands");
        RequireAddress(record.operands[0], "hbm_address",
                       SemanticOperandId::HBM_ADDRESS, path + ".operands[0]");
        RequireLiteral(record.operands[1], "size_bytes", path + ".operands[1]");
        const SemanticOperandId local_id =
            record.opcode == Opcode::LSU_LOAD
                ? SemanticOperandId::DESTINATION_ADDRESS
                : SemanticOperandId::SOURCE_ADDRESS;
        RequireAddress(
            record.operands[2],
            record.opcode == Opcode::LSU_LOAD ? "destination_address"
                                              : "source_address",
            local_id, path + ".operands[2]");
        const AddressRelocationDto &hbm =
            relocation(SemanticOperandId::HBM_ADDRESS);
        const SymbolEntry &hbm_symbol =
            symbol(SemanticOperandId::HBM_ADDRESS);
        if (hbm_symbol.definition->symbol.kind !=
                ProgramSymbolKind::ABSOLUTE_ADDRESS)
            Fail(path,
                 "blocking LSU HBM_ADDRESS requires ABSOLUTE_ADDRESS");
        LsuOperands operands;
        operands.hbm_address_bytes = hbm_symbol.definition->value;
        operands.size_bytes = LiteralU64(record.operands[1], path);
        operands.sram = address(local_id);
        result.operands = operands;
        AppendRelocation(artifact, core_index, instruction_index,
                         SemanticOperandId::HBM_ADDRESS, hbm, symbols);
    } else if (record.opcode == Opcode::DTE_ISSUE) {
        if (record.operands.size() != 7)
            Fail(path + ".operands", "DTE_ISSUE requires seven operands");
        RequireLiteral(record.operands[0], "direction", path + ".operands[0]");
        RequireRuntime(record.operands[1], "token",
                       RuntimeOperandFieldDto::DTE_TOKEN, path + ".operands[1]");
        RequireLiteral(record.operands[2], "payload_bits", path + ".operands[2]");
        RequireLiteral(record.operands[3], "size_bytes", path + ".operands[3]");
        RequireLiteral(record.operands[4], "hbm_address", path + ".operands[4]");
        RequireAddress(record.operands[5], "source_address",
                       SemanticOperandId::SOURCE_ADDRESS, path + ".operands[5]");
        RequireAddress(record.operands[6], "destination_address",
                       SemanticOperandId::DESTINATION_ADDRESS, path + ".operands[6]");
        DteIssueOperands operands;
        operands.direction = LiteralEnum<LocalDteDirection>(record.operands[0], path);
        operands.token = runtime_value(record.operands[1], "token",
                                       RuntimeOperandFieldDto::DTE_TOKEN, path);
        operands.payload_bits = LiteralU64(record.operands[2], path);
        operands.size_bytes = LiteralU64(record.operands[3], path);
        operands.hbm_address_bytes = LiteralU64(record.operands[4], path);
        operands.source_sram = address(SemanticOperandId::SOURCE_ADDRESS);
        operands.destination_sram = address(SemanticOperandId::DESTINATION_ADDRESS);
        result.operands = operands;
    } else if (record.opcode == Opcode::DTE_WAIT) {
        if (record.operands.size() != 1)
            Fail(path + ".operands", "DTE_WAIT requires one operand");
        result.operands = TokenOperands{runtime_value(
            record.operands[0], "token", RuntimeOperandFieldDto::DTE_TOKEN,
            path + ".operands[0]")};
    } else if (record.opcode == Opcode::EVENT_SET) {
        if (record.operands.size() != 3)
            Fail(path + ".operands", "EVENT_SET requires three operands");
        result.operands = EventSetOperands{
            runtime_value(record.operands[0], "source_core",
                          RuntimeOperandFieldDto::SOURCE_CORE, path),
            runtime_value(record.operands[1], "destination_core",
                          RuntimeOperandFieldDto::DESTINATION_CORE, path),
            runtime_value(record.operands[2], "tag",
                          RuntimeOperandFieldDto::EVENT_TAG, path)};
    } else if (record.opcode == Opcode::EVENT_WAIT) {
        if (record.operands.size() != 4)
            Fail(path + ".operands", "EVENT_WAIT requires four operands");
        RequireLiteral(record.operands[3], "count", path + ".operands[3]");
        result.operands = EventWaitOperands{
            runtime_value(record.operands[0], "source_core",
                          RuntimeOperandFieldDto::SOURCE_CORE, path),
            runtime_value(record.operands[1], "destination_core",
                          RuntimeOperandFieldDto::DESTINATION_CORE, path),
            runtime_value(record.operands[2], "tag",
                          RuntimeOperandFieldDto::EVENT_TAG, path),
            LiteralU64(record.operands[3], path)};
    } else if (record.opcode == Opcode::SRAM_FREE) {
        if (record.operands.size() != 1)
            Fail(path + ".operands", "SRAM_FREE requires one operand");
        RequireAddress(record.operands[0], "symbol", SemanticOperandId::SYMBOL,
                       path + ".operands[0]");
        if (symbol(SemanticOperandId::SYMBOL).definition->symbol.kind !=
                ProgramSymbolKind::SRAM_LABEL ||
            relocation(SemanticOperandId::SYMBOL).addend != 0)
            Fail(path, "SRAM_FREE requires a zero-addend SRAM_LABEL relocation");
        result.operands = SymbolOperands{symbol(SemanticOperandId::SYMBOL).index};
        AppendRelocation(artifact, core_index, instruction_index,
                         SemanticOperandId::SYMBOL,
                         relocation(SemanticOperandId::SYMBOL), symbols);
    } else {
        Fail(path + ".opcode",
             "unsupported mixed-manifest opcode");
    }
    ValidateExternalRecord(result);
    return result;
}

std::string AddressSymbolRef(const RelocatableRecordDto &record,
                             SemanticOperandId id) {
    for (const RecordOperandDto &operand : record.operands) {
        if (operand.kind == OperandKindDto::ADDRESS_SYMBOL &&
            operand.operand_id == id && operand.symbol_ref)
            return *operand.symbol_ref;
    }
    return {};
}

std::string RuntimeSymbolRef(const RelocatableRecordDto &record,
                             RuntimeOperandFieldDto field) {
    for (const RecordOperandDto &operand : record.operands) {
        if (operand.kind == OperandKindDto::RUNTIME_SYMBOL &&
            operand.runtime_field == field && operand.symbol_ref)
            return *operand.symbol_ref;
    }
    return {};
}

struct StateTransferEndpointWitness {
    std::string fragment_id;
    std::string fsm_ref;
    uint64_t length_bytes = 0;
    bool source = false;
};

StateTransferEndpointWitness ValidateStateTransferFragment(
    const CommandFragmentDto &fragment, const std::string &path) {
    if (fragment.core_streams.size() != 1 ||
        fragment.buffer_abi.size() != 1 || !fragment.state_abi.empty())
        Fail(path,
             "STATE_TRANSFER requires one core stream, one local BufferABI, and no StateABI");
    const CoreFragmentStreamDto &stream = fragment.core_streams.front();
    if (!(fragment.buffer_abi.front().logical_core == stream.logical_core))
        Fail(path + ".buffer_abi",
             "STATE_TRANSFER BufferABI must belong to its endpoint core");
    const std::vector<RelocatableRecordDto> &records = stream.records;
    const bool legacy_source =
        records.size() == 2 && records[0].opcode == Opcode::DTE_SEND &&
        records[1].opcode == Opcode::SRAM_FREE;
    const bool sliced_source =
        records.size() == 2 && records[0].opcode == Opcode::SRAM_ALLOC_AT &&
        records[1].opcode == Opcode::DTE_SEND;
    const bool source = legacy_source || sliced_source;
    const bool destination =
        records.size() == 3 && records[0].opcode == Opcode::SRAM_ALLOC_AT &&
        records[1].opcode == Opcode::DTE_RECV &&
        records[2].opcode == Opcode::DTE_WAIT;
    if (!source && !destination)
        Fail(path + ".core_streams[0].records",
             "STATE_TRANSFER must be exact SEND+FREE, ALLOC+SEND, or ALLOC+RECV+WAIT");

    const RelocatableRecordDto &transport =
        legacy_source ? records[0] : records[1];
    if (source) {
        if (fragment.claimed_action_ids.size() != 1 ||
            records[0].source_global_action_id !=
                records[1].source_global_action_id ||
            fragment.claimed_action_ids.front() !=
                records[0].source_global_action_id)
            Fail(path + ".claimed_action_ids",
                 "STATE_TRANSFER source must claim one SEND action owning its FREE");
        if (transport.operands.size() != 15)
            Fail(path + ".core_streams[0].records[" +
                     std::string(sliced_source ? "1" : "0") + "].operands",
                 "STATE_TRANSFER SEND requires fifteen operands");
    } else {
        if (fragment.claimed_action_ids.size() != 2 ||
            records[0].source_global_action_id !=
                records[1].source_global_action_id ||
            records[1].source_global_action_id ==
                records[2].source_global_action_id ||
            std::set<std::string>{
                records[1].source_global_action_id,
                records[2].source_global_action_id} !=
                std::set<std::string>(
                    fragment.claimed_action_ids.begin(),
                    fragment.claimed_action_ids.end()))
            Fail(path + ".claimed_action_ids",
                 "STATE_TRANSFER destination must claim ordered RECV then WAIT actions");
        if (transport.operands.size() != 14)
            Fail(path + ".core_streams[0].records[1].operands",
                 "STATE_TRANSFER RECV requires fourteen operands");
    }

    const std::string operand_path =
        path + ".core_streams[0].records[" +
        std::string(legacy_source ? "0" : "1") + "].operands";
    auto require_literal = [&](std::size_t index, std::string_view name,
                               uint64_t expected) {
        RequireLiteral(transport.operands[index], name,
                       operand_path + "[" + std::to_string(index) + "]");
        if (LiteralU64(transport.operands[index],
                       operand_path + "[" + std::to_string(index) + "]") !=
            expected)
            Fail(operand_path + "[" + std::to_string(index) + "]",
                 "STATE_TRANSFER literal differs from the canonical P2P ABI");
    };

    std::string fsm_ref;
    uint64_t length_bytes = 0;
    if (source) {
        require_literal(0, "mode", 0);
        require_literal(1, "source_space", 0);
        require_literal(2, "completion", 1);
        require_literal(3, "datatype", 0);
        require_literal(4, "reduce_op", 0);
        RequireRuntime(transport.operands[5], "fsm_id",
                       RuntimeOperandFieldDto::DTE_FSM,
                       operand_path + "[5]");
        fsm_ref = *transport.operands[5].symbol_ref;
        require_literal(6, "token", 0);
        RequireLiteral(transport.operands[7], "length_bytes",
                       operand_path + "[7]");
        length_bytes = LiteralU64(transport.operands[7],
                                  operand_path + "[7]");
        RequireAddress(transport.operands[8], "source_address",
                       SemanticOperandId::SOURCE_ADDRESS,
                       operand_path + "[8]");
        RequireRuntime(transport.operands[9], "peer_core",
                       RuntimeOperandFieldDto::PEER_CORE,
                       operand_path + "[9]");
        require_literal(10, "expected_sources", 0);
        require_literal(11, "tree_id", 0);
        require_literal(12, "group_id", 0);
        require_literal(13, "collective_id", 0);
        require_literal(14, "epoch", 0);
    } else {
        require_literal(0, "mode", 0);
        require_literal(1, "completion", 0);
        require_literal(2, "datatype", 0);
        require_literal(3, "reduce_op", 0);
        RequireRuntime(transport.operands[4], "fsm_id",
                       RuntimeOperandFieldDto::DTE_FSM,
                       operand_path + "[4]");
        fsm_ref = *transport.operands[4].symbol_ref;
        RequireRuntime(transport.operands[5], "token",
                       RuntimeOperandFieldDto::DTE_TOKEN,
                       operand_path + "[5]");
        RequireLiteral(transport.operands[6], "length_bytes",
                       operand_path + "[6]");
        length_bytes = LiteralU64(transport.operands[6],
                                  operand_path + "[6]");
        RequireAddress(transport.operands[7], "destination_address",
                       SemanticOperandId::DESTINATION_ADDRESS,
                       operand_path + "[7]");
        RequireRuntime(transport.operands[8], "peer_core",
                       RuntimeOperandFieldDto::PEER_CORE,
                       operand_path + "[8]");
        require_literal(9, "expected_sources", 0);
        require_literal(10, "tree_id", 0);
        require_literal(11, "group_id", 0);
        require_literal(12, "collective_id", 0);
        require_literal(13, "epoch", 0);
        if (records[2].operands.size() != 1)
            Fail(path + ".core_streams[0].records[2].operands",
                 "STATE_TRANSFER WAIT requires one token operand");
        RequireRuntime(records[2].operands[0], "token",
                       RuntimeOperandFieldDto::DTE_TOKEN,
                       path + ".core_streams[0].records[2].operands[0]");
        if (records[2].operands[0].symbol_ref !=
            transport.operands[5].symbol_ref)
            Fail(path + ".core_streams[0].records",
                 "STATE_TRANSFER RECV and WAIT must share one runtime token");
    }
    constexpr uint64_t kMaxEndpointBytes =
        (static_cast<uint64_t>(1) << 16U) * 16U - 16U;
    if (length_bytes == 0 || length_bytes > kMaxEndpointBytes)
        Fail(operand_path,
             "STATE_TRANSFER length must be positive and fit the endpoint P2P ABI");
    return {fragment.id, fsm_ref, length_bytes, source};
}
std::vector<StateTransferEndpointWitness> ValidateStateTransferFragments(
    const CommandFragmentDto &fragment, const std::string &path) {
    const std::vector<RelocatableRecordDto> &records =
        fragment.core_streams.empty()
            ? std::vector<RelocatableRecordDto>{}
            : fragment.core_streams.front().records;
    const bool legacy_source =
        records.size() == 2 && records[0].opcode == Opcode::DTE_SEND &&
        records[1].opcode == Opcode::SRAM_FREE;
    const bool sliced_source =
        records.size() == 2 && records[0].opcode == Opcode::SRAM_ALLOC_AT &&
        records[1].opcode == Opcode::DTE_SEND;
    const bool sliced_destination =
        records.size() == 3 && records[0].opcode == Opcode::SRAM_ALLOC_AT &&
        records[1].opcode == Opcode::DTE_RECV &&
        records[2].opcode == Opcode::DTE_WAIT;
    if (legacy_source || sliced_source || sliced_destination)
        return {ValidateStateTransferFragment(fragment, path)};

    if (fragment.core_streams.size() != 1 ||
        fragment.buffer_abi.size() != 1 || !fragment.state_abi.empty())
        Fail(path,
             "STATE_TRANSFER requires one core stream, one local BufferABI, and no StateABI");
    const CoreFragmentStreamDto &stream = fragment.core_streams.front();
    if (!(fragment.buffer_abi.front().logical_core == stream.logical_core))
        Fail(path + ".buffer_abi",
             "STATE_TRANSFER BufferABI must belong to its endpoint core");
    const bool has_alloc =
        !records.empty() && records.front().opcode == Opcode::SRAM_ALLOC_AT;
    const std::size_t begin = has_alloc ? 1 : 0;
    if (begin == records.size())
        Fail(path + ".core_streams[0].records",
             "segmented STATE_TRANSFER requires transport units");

    struct SegmentedUnit {
        std::size_t transport;
        std::size_t completion;
    };
    const bool source =
        records[begin].opcode == Opcode::DTE_SEND ||
        records[begin].opcode == Opcode::EVENT_WAIT;
    const bool destination = records[begin].opcode == Opcode::DTE_RECV;
    if (!source && !destination)
        Fail(path + ".core_streams[0].records",
             "segmented STATE_TRANSFER requires SEND units or RECV/WAIT pairs");
    std::vector<SegmentedUnit> units;
    std::size_t cursor = begin;
    if (source) {
        std::vector<bool> wave_waits;
        while (cursor < records.size()) {
            const bool has_wave_wait =
                records[cursor].opcode == Opcode::EVENT_WAIT;
            if (has_wave_wait) {
                if (cursor + 1 >= records.size() ||
                    records[cursor + 1].opcode != Opcode::DTE_SEND ||
                    records[cursor].source_global_action_id !=
                        records[cursor + 1].source_global_action_id)
                    Fail(path + ".core_streams[0].records",
                         "segmented STATE_TRANSFER source wave requires EVENT_WAIT immediately before its owned SEND");
                ++cursor;
            }
            if (records[cursor].opcode != Opcode::DTE_SEND)
                Fail(path + ".core_streams[0].records",
                     "segmented STATE_TRANSFER source requires canonical SEND units");
            units.push_back({cursor, cursor});
            wave_waits.push_back(has_wave_wait);
            ++cursor;
        }
        if (units.size() < 2)
            Fail(path + ".core_streams[0].records",
                 "segmented STATE_TRANSFER source requires at least two SEND units");
        bool suffix = true;
        bool all_wait = true;
        bool transition = true;
        for (std::size_t index = 0; index < wave_waits.size(); ++index) {
            suffix &= wave_waits[index] == (index >= 3);
            all_wait &= wave_waits[index];
            transition &= wave_waits[index] ==
                          (index == 0 || index >= 3);
        }
        if (!suffix && !all_wait && !transition)
            Fail(path + ".core_streams[0].records",
                 "segmented STATE_TRANSFER source wave shape is not canonical");
    } else {
        std::vector<bool> wave_sets;
        while (cursor < records.size()) {
            if (cursor + 1 >= records.size() ||
                records[cursor].opcode != Opcode::DTE_RECV ||
                records[cursor + 1].opcode != Opcode::DTE_WAIT)
                Fail(path + ".core_streams[0].records",
                     "segmented STATE_TRANSFER destination requires canonical RECV/WAIT pairs");
            const std::size_t transport = cursor;
            const std::size_t completion = cursor + 1;
            cursor += 2;
            const bool has_wave_set =
                cursor < records.size() &&
                records[cursor].opcode == Opcode::EVENT_SET;
            if (has_wave_set) {
                if (records[cursor].source_global_action_id !=
                        records[completion].source_global_action_id)
                    Fail(path + ".core_streams[0].records",
                         "segmented STATE_TRANSFER destination wave SET must be owned by WAIT");
                ++cursor;
            }
            units.push_back({transport, completion});
            wave_sets.push_back(has_wave_set);
        }
        if (units.size() < 2)
            Fail(path + ".core_streams[0].records",
                 "segmented STATE_TRANSFER destination requires at least two RECV/WAIT pairs");
        bool prefix = true;
        bool all_set = true;
        bool transition_tail = true;
        const std::size_t count = wave_sets.size();
        for (std::size_t index = 0; index < count; ++index) {
            prefix &= wave_sets[index] == (index + 3 < count);
            all_set &= wave_sets[index];
            transition_tail &= wave_sets[index] ==
                               (index + 3 < count || index + 1 == count);
        }
        if (!prefix && !all_set && !transition_tail)
            Fail(path + ".core_streams[0].records",
                 "segmented STATE_TRANSFER destination wave shape is not canonical");
    }

    std::set<std::string> expected_claims;
    for (const SegmentedUnit &unit : units) {
        if (!expected_claims.insert(
                records[unit.transport].source_global_action_id).second)
            Fail(path + ".claimed_action_ids",
                 "segmented STATE_TRANSFER transport actions must be unique");
        if (!source &&
            !expected_claims.insert(
                records[unit.completion].source_global_action_id).second)
            Fail(path + ".claimed_action_ids",
                 "segmented STATE_TRANSFER RECV/WAIT actions must be unique");
    }
    const std::set<std::string> actual_claims(
        fragment.claimed_action_ids.begin(), fragment.claimed_action_ids.end());
    if (actual_claims != expected_claims ||
        actual_claims.size() != fragment.claimed_action_ids.size())
        Fail(path + ".claimed_action_ids",
             "segmented STATE_TRANSFER must claim exactly its transport actions");
    if (has_alloc && records.front().source_global_action_id !=
                         records[units.front().transport].source_global_action_id)
        Fail(path + ".core_streams[0].records[0]",
             "segmented STATE_TRANSFER ALLOC must be owned by its first transport action");

    std::set<std::string> fsm_refs;
    std::set<std::string> peer_refs;
    std::set<std::string> token_refs;
    std::vector<StateTransferEndpointWitness> witnesses;
    for (const SegmentedUnit &unit : units) {
        CommandFragmentDto unit_fragment = fragment;
        CoreFragmentStreamDto unit_stream = stream;
        RelocatableRecordDto alloc =
            has_alloc ? records.front() : RelocatableRecordDto{};
        alloc.source_global_action_id =
            records[unit.transport].source_global_action_id;
        alloc.opcode = Opcode::SRAM_ALLOC_AT;
        unit_stream.records = source
            ? std::vector<RelocatableRecordDto>{
                  alloc, records[unit.transport]}
            : std::vector<RelocatableRecordDto>{
                  alloc, records[unit.transport],
                  records[unit.completion]};
        unit_fragment.core_streams = {std::move(unit_stream)};
        unit_fragment.claimed_action_ids =
            source
                ? std::vector<std::string>{
                      records[unit.transport].source_global_action_id}
                : std::vector<std::string>{
                      records[unit.transport].source_global_action_id,
                      records[unit.completion].source_global_action_id};
        StateTransferEndpointWitness witness =
            ValidateStateTransferFragment(unit_fragment, path);
        const RelocatableRecordDto &transport = records[unit.transport];
        const std::string peer_ref = RuntimeSymbolRef(
            transport, RuntimeOperandFieldDto::PEER_CORE);
        const std::string token_ref = source
            ? std::string{}
            : RuntimeSymbolRef(transport, RuntimeOperandFieldDto::DTE_TOKEN);
        if (!fsm_refs.insert(witness.fsm_ref).second)
            Fail(path + ".core_streams[0].records",
                 "segmented STATE_TRANSFER DTE_FSM references must be unique");
        if (!peer_refs.insert(peer_ref).second)
            Fail(path + ".core_streams[0].records",
                 "segmented STATE_TRANSFER PEER_CORE references must be unique");
        if (!source && !token_refs.insert(token_ref).second)
            Fail(path + ".core_streams[0].records",
                 "segmented STATE_TRANSFER DTE_TOKEN references must be unique");
        witnesses.push_back(std::move(witness));
    }
    return witnesses;
}


void ValidateActionSequence(
    const std::vector<const RelocatableRecordDto *> &records,
    const std::string &path) {
    std::set<std::string> completed;
    std::set<std::string> allocated_once;
    std::set<std::string> freed_once;
    std::set<std::string> active;
    std::size_t begin = 0;
    while (begin < records.size()) {
        const std::string action = records[begin]->source_global_action_id;
        if (!completed.insert(action).second)
            Fail(path, "one action's record block is non-contiguous");
        std::size_t end = begin;
        while (end < records.size() &&
               records[end]->source_global_action_id == action)
            ++end;
        std::size_t cursor = begin;
        while (cursor < end && records[cursor]->opcode == Opcode::SRAM_ALLOC_AT) {
            const std::string label = AddressSymbolRef(
                *records[cursor], SemanticOperandId::LABEL_SYMBOL);
            if (label.empty() || !allocated_once.insert(label).second ||
                !active.insert(label).second)
                Fail(path,
                     "SRAM_ALLOC_AT label must be globally unique and inactive on its final core stream");
            ++cursor;
        }
        std::size_t suffix = end;
        while (suffix > cursor &&
               records[suffix - 1]->opcode == Opcode::SRAM_FREE)
            --suffix;
        const auto is_compute = [](Opcode opcode) {
            return opcode == Opcode::MATMUL || opcode == Opcode::ATTENTION ||
                   opcode == Opcode::SWIGLU || opcode == Opcode::RESIDUAL ||
                   opcode == Opcode::RMSNORM ||
                   opcode == Opcode::ROPE_QK_EXACT ||
                   opcode == Opcode::ATTENTION_EXACT ||
                   opcode == Opcode::EMBEDDING_LOOKUP ||
                   opcode == Opcode::GREEDY_SAMPLE ||
                   opcode == Opcode::CROSS_ENTROPY_FORWARD ||
                   opcode == Opcode::CROSS_ENTROPY_BACKWARD ||
                   opcode == Opcode::SGD_UPDATE;
        };
        bool valid_body = false;
        if (suffix == cursor + 2 &&
            records[cursor]->opcode == Opcode::SRAM_BIND &&
            is_compute(records[cursor + 1]->opcode)) {
            const Opcode compute_opcode = records[cursor + 1]->opcode;
            const uint64_t expected_inputs =
                compute_opcode == Opcode::CROSS_ENTROPY_BACKWARD ? 3 :
                (compute_opcode == Opcode::RESIDUAL ||
                 compute_opcode == Opcode::EMBEDDING_LOOKUP ||
                 compute_opcode == Opcode::CROSS_ENTROPY_FORWARD ||
                 compute_opcode == Opcode::SGD_UPDATE) ? 2 : 1;
            if (records[cursor]->operands.empty() ||
                LiteralU64(records[cursor]->operands[0], path) !=
                    expected_inputs)
                Fail(path,
                     "compute SRAM_BIND input_count does not match its exact opcode ABI");
            valid_body = true;
        }
        else if (suffix == cursor + 2 &&
                 records[cursor]->opcode == Opcode::DTE_ISSUE &&
                 records[cursor + 1]->opcode == Opcode::DTE_WAIT)
            valid_body = true;
        else if (suffix == cursor + 2 &&
                 records[cursor]->opcode == Opcode::EVENT_WAIT &&
                 records[cursor + 1]->opcode == Opcode::DTE_SEND)
            valid_body = true;
        else if (suffix == cursor + 2 &&
                 records[cursor]->opcode == Opcode::DTE_WAIT &&
                 records[cursor + 1]->opcode == Opcode::EVENT_SET)
            valid_body = true;
        else if (suffix == cursor + 1) {
            const Opcode opcode = records[cursor]->opcode;
            valid_body = opcode == Opcode::DTE_SEND ||
                         opcode == Opcode::DTE_RECV ||
                         opcode == Opcode::DTE_WAIT ||
                         opcode == Opcode::LOCAL_REDUCE ||
                         opcode == Opcode::LSU_LOAD ||
                         opcode == Opcode::LSU_STORE;
        } else if (suffix > cursor) {
            valid_body = std::all_of(
                records.begin() + static_cast<std::ptrdiff_t>(cursor),
                records.begin() + static_cast<std::ptrdiff_t>(suffix),
                [](const RelocatableRecordDto *item) {
                    return item->opcode == Opcode::EVENT_SET ||
                           item->opcode == Opcode::EVENT_WAIT;
                });
        }
        if (!valid_body)
            Fail(path, "action has a non-canonical mixed record body");
        cursor = suffix;
        while (cursor < end) {
            const std::string label = AddressSymbolRef(
                *records[cursor], SemanticOperandId::SYMBOL);
            if (label.empty() || active.erase(label) != 1 ||
                !freed_once.insert(label).second)
                Fail(path,
                     "SRAM_FREE label must be active and freed exactly once on its final core stream");
            ++cursor;
        }
        begin = end;
    }
    if (!active.empty() || allocated_once != freed_once)
        Fail(path,
             "final core stream ends with dangling TASK SRAM labels");
}

std::vector<uint64_t> RemapCores(
    const std::vector<LogicalCoreDto> &cores,
    const std::map<LogicalCoreDto, uint64_t> &runtime_ids,
    const std::string &path) {
    RequireCanonical(cores, path);
    std::vector<uint64_t> result;
    result.reserve(cores.size());
    for (const LogicalCoreDto &core : cores) {
        const auto found = runtime_ids.find(core);
        if (found == runtime_ids.end())
            Fail(path, "references a core without a runtime binding");
        result.push_back(found->second);
    }
    std::sort(result.begin(), result.end());
    if (std::adjacent_find(result.begin(), result.end()) != result.end())
        Fail(path, "runtime core remap is not unique");
    return result;
}

} // namespace

ProgramArtifact ProgramArtifactFinalizer::Finalize(
    const LinkedProgramManifestDto &manifest) const {
    try {
        if (manifest.schema_version != kLinkedProgramManifestSchemaVersion ||
            manifest.producer_pass.empty() || manifest.id.empty() ||
            manifest.source_ir1_id.empty() ||
            manifest.source_projection_id.empty() ||
            manifest.source_schedule_set_id.empty() ||
            manifest.source_global_dag_id.empty())
            Fail("linked_program_manifest", "invalid identity or schema version");
        if (manifest.capabilities != 0)
            Fail("linked_program_manifest.capabilities", "MVP capabilities must be zero");
        if (!manifest.core_groups.empty())
            Fail("linked_program_manifest.core_groups",
                 "mixed MVP does not yet support GROUP runtime symbols");
        if (manifest.fragments.empty())
            Fail("linked_program_manifest.fragments", "must be non-empty");

        RequireCanonicalBy(manifest.input_digests,
                           "linked_program_manifest.input_digests",
                           [](const ManifestInputDigestDto &item) {
                               return std::make_pair(
                                   std::string(InputKindKey(item.kind)),
                                   item.artifact_id);
                           });
        const std::map<ManifestInputKindDto,
                       std::pair<std::string, std::string>> upstream_inputs{
            {ManifestInputKindDto::IR1,
             {manifest.source_ir1_id, "wafer_frontend.ir1/v1alpha14"}},
            {ManifestInputKindDto::IR2_PROJECTION,
             {manifest.source_projection_id,
              "wafer_frontend.ir2_projection_result/v1alpha13"}},
            {ManifestInputKindDto::SCHEDULE_SET,
             {manifest.source_schedule_set_id,
              "wafer_frontend.intra_die_schedule_set/v1alpha9"}},
            {ManifestInputKindDto::GLOBAL_ACTION_DAG,
             {manifest.source_global_dag_id,
              "wafer_frontend.global_action_dag/v1alpha11"}},
        };
        const std::map<ManifestInputKindDto, std::string>
            train_lineage_schemas{
                {ManifestInputKindDto::IR1,
                 "wafer_frontend.ir1/v1alpha14"},
                {ManifestInputKindDto::IR2_PROJECTION,
                 "wafer_frontend.ir2_projection_result/v1alpha13"},
                {ManifestInputKindDto::SCHEDULE_SET,
                 "wafer_frontend.intra_die_schedule_set/v1alpha9"},
                {ManifestInputKindDto::GLOBAL_ACTION_DAG,
                 "wafer_frontend.global_action_dag/v1alpha11"},
            };
        std::set<std::tuple<ManifestInputKindDto, std::string, std::string>>
            expected_inputs;
        std::vector<const ManifestInputDigestDto *> train_inputs;
        std::vector<const ManifestInputDigestDto *> s3_lite_inputs;
        std::vector<const ManifestInputDigestDto *> rooted_ar_inputs;
        std::map<ManifestInputKindDto, std::set<std::string>>
            train_lineage_ids;
        for (const ManifestInputDigestDto &digest : manifest.input_digests) {
            if (digest.kind ==
                ManifestInputKindDto::TRAIN_LOWERED_PROGRAM) {
                train_inputs.push_back(&digest);
                continue;
            }
            if (digest.kind == ManifestInputKindDto::S3_LITE_MOE) {
                s3_lite_inputs.push_back(&digest);
                continue;
            }
            if (digest.kind == ManifestInputKindDto::S2_LITE_ROOTED_AR) {
                rooted_ar_inputs.push_back(&digest);
                continue;
            }
            const auto lineage = train_lineage_schemas.find(digest.kind);
            if (lineage != train_lineage_schemas.end())
                train_lineage_ids[digest.kind].insert(digest.artifact_id);
        }
        const std::size_t top_input_kinds =
            (!train_inputs.empty() ? 1 : 0) +
            (!s3_lite_inputs.empty() ? 1 : 0) +
            (!rooted_ar_inputs.empty() ? 1 : 0);
        if (train_inputs.size() > 1 || s3_lite_inputs.size() > 1 ||
            rooted_ar_inputs.size() > 1 || top_input_kinds > 1)
            Fail("linked_program_manifest.input_digests",
                 "train, S3-Lite and rooted-AR top-level inputs are exclusive and singular");
        const bool train_link = !train_inputs.empty();
        const bool s3_lite_link = !s3_lite_inputs.empty();
        const bool rooted_ar_link = !rooted_ar_inputs.empty();
        if (s3_lite_link) {
            const ManifestInputDigestDto &s3 = *s3_lite_inputs.front();
            if (s3.schema_version !=
                "wafer_frontend.s3_lite_moe_lowered_program/v1alpha1")
                Fail("linked_program_manifest.input_digests",
                     "S3-Lite lowered-program schema version mismatch");
            expected_inputs.emplace(s3.kind, s3.artifact_id,
                                    s3.schema_version);
            const std::map<ManifestInputKindDto, std::string> s3_schemas{
                {ManifestInputKindDto::IR1,
                 "wafer_frontend.ir1/v1alpha14"},
                {ManifestInputKindDto::IR2_PROJECTION,
                 "wafer_frontend.s3_lite_static_moe_projection/v1alpha1"},
                {ManifestInputKindDto::SCHEDULE_SET,
                 "wafer_frontend.s3_lite_static_moe_schedule/v1alpha1"},
                {ManifestInputKindDto::GLOBAL_ACTION_DAG,
                 "wafer_frontend.s3_lite_static_moe_global/v1alpha1"},
            };
            for (const auto &entry : s3_schemas) {
                const std::set<std::string> &ids =
                    train_lineage_ids[entry.first];
                if (ids.size() != 1)
                    Fail("linked_program_manifest.input_digests",
                         "S3-Lite requires one exact lineage input per stage");
                for (const ManifestInputDigestDto &digest :
                     manifest.input_digests) {
                    if (digest.kind != entry.first)
                        continue;
                    if (digest.schema_version != entry.second)
                        Fail("linked_program_manifest.input_digests",
                             "S3-Lite lineage schema version mismatch");
                    expected_inputs.emplace(digest.kind, digest.artifact_id,
                                            digest.schema_version);
                }
            }
        } else if (rooted_ar_link) {
            const ManifestInputDigestDto &rooted = *rooted_ar_inputs.front();
            if (rooted.schema_version !=
                "wafer_frontend.s2_lite_rooted_ar_lowered_program/v1alpha1")
                Fail("linked_program_manifest.input_digests",
                     "rooted-AR lowered-program schema version mismatch");
            expected_inputs.emplace(rooted.kind, rooted.artifact_id,
                                    rooted.schema_version);
            for (const auto &entry : train_lineage_schemas) {
                const std::set<std::string> &ids =
                    train_lineage_ids[entry.first];
                if (ids.size() != 2)
                    Fail("linked_program_manifest.input_digests",
                         "rooted-AR requires two exact lineage inputs per stage");
                for (const ManifestInputDigestDto &digest :
                     manifest.input_digests) {
                    if (digest.kind != entry.first)
                        continue;
                    if (digest.schema_version != entry.second)
                        Fail("linked_program_manifest.input_digests",
                             "rooted-AR lineage schema version mismatch");
                    expected_inputs.emplace(digest.kind,
                                            digest.artifact_id,
                                            digest.schema_version);
                }
            }
        } else if (train_link) {
            const ManifestInputDigestDto &train = *train_inputs.front();
            if (train.schema_version !=
                "wafer_frontend.train_lowered_program/v1alpha3")
                Fail("linked_program_manifest.input_digests",
                     "train lowered-program schema version mismatch");
            expected_inputs.emplace(train.kind, train.artifact_id,
                                    train.schema_version);
            std::size_t replica_count = 0;
            for (const auto &entry : train_lineage_schemas) {
                const std::set<std::string> &ids =
                    train_lineage_ids[entry.first];
                if (ids.empty() ||
                    (replica_count != 0 && ids.size() != replica_count))
                    Fail("linked_program_manifest.input_digests",
                         "train input requires equal non-zero IR1/projection/schedule/global digest coverage");
                replica_count = ids.size();
                for (const ManifestInputDigestDto &digest :
                     manifest.input_digests) {
                    if (digest.kind != entry.first)
                        continue;
                    if (digest.schema_version != entry.second)
                        Fail("linked_program_manifest.input_digests",
                             "train replica lineage schema version mismatch");
                    expected_inputs.emplace(digest.kind,
                                            digest.artifact_id,
                                            digest.schema_version);
                }
            }
        }
        std::size_t standalone_fragment_count = 0;
        std::set<std::string> rooted_local_dag_ids;
        std::vector<std::vector<Opcode>> rooted_overlay_sequences;
        if (!train_link && !s3_lite_link && !rooted_ar_link) {
            for (const auto &entry : upstream_inputs)
                expected_inputs.emplace(entry.first, entry.second.first,
                                        entry.second.second);
        }
        std::set<std::string> fragment_global_dag_ids;
        for (const LinkedFragmentDto &linked : manifest.fragments) {
            const CommandFragmentDto &fragment = Leaf(linked);
            fragment_global_dag_ids.insert(fragment.source_global_dag_id);
            if (rooted_ar_link) {
                if (fragment.kind == FragmentKindDto::S2_LITE_ROOTED_AR) {
                    if (fragment.producer_pass !=
                            "s2_lite_rooted_ar_lowering" ||
                        fragment.source_global_dag_id !=
                            manifest.source_global_dag_id ||
                        fragment.core_streams.size() != 1)
                        Fail("linked_program_manifest.fragments",
                             "rooted-AR overlay kind/producer/top-carrier/core-stream contract mismatch");
                    std::vector<Opcode> sequence;
                    for (const RelocatableRecordDto &record :
                         fragment.core_streams.front().records)
                        sequence.push_back(record.opcode);
                    rooted_overlay_sequences.push_back(std::move(sequence));
                } else {
                    rooted_local_dag_ids.insert(
                        fragment.source_global_dag_id);
                }
            }
            if (fragment.kind == FragmentKindDto::STANDALONE_COLLECTIVE)
                ++standalone_fragment_count;
            expected_inputs.emplace(
                ManifestInputKindDto::COMMAND_FRAGMENT, fragment.id,
                kCommandFragmentSchemaVersion);
            if (const auto *region = std::get_if<RegionManifestDto>(&linked)) {
                expected_inputs.emplace(
                    ManifestInputKindDto::REGION_MANIFEST, region->id,
                    kRegionManifestSchemaVersion);
                expected_inputs.emplace(
                    ManifestInputKindDto::FUSION_PLAN,
                    region->fusion_plan_id,
                    "wafer_frontend.fusion_plan/v1alpha10");
            }
        }
        std::size_t standalone_digest_count = 0;
        for (const ManifestInputDigestDto &digest : manifest.input_digests) {
            if (digest.kind == ManifestInputKindDto::STANDALONE_PLAN) {
                ++standalone_digest_count;
                if (digest.schema_version !=
                    "wafer_frontend.standalone_collective_plan/v1alpha9")
                    Fail("linked_program_manifest.input_digests",
                         "standalone plan schema version mismatch");
                expected_inputs.emplace(digest.kind, digest.artifact_id,
                                        digest.schema_version);
            }
        }
        if (standalone_digest_count != standalone_fragment_count)
            Fail("linked_program_manifest.input_digests",
                 "standalone plan trust anchors must bijectively match standalone fragments");
        if (train_link &&
            fragment_global_dag_ids !=
                train_lineage_ids[ManifestInputKindDto::GLOBAL_ACTION_DAG])
            Fail("linked_program_manifest.fragments",
                 "train fragments must witness every replica global DAG digest");
        if (rooted_ar_link) {
            std::vector<std::vector<Opcode>> expected_sequences{{
                {Opcode::SRAM_ALLOC_AT, Opcode::DTE_ISSUE,
                 Opcode::DTE_WAIT},
                {Opcode::DTE_SEND},
                {Opcode::SRAM_ALLOC_AT, Opcode::DTE_RECV,
                 Opcode::DTE_WAIT},
                {Opcode::LOCAL_REDUCE},
                {Opcode::DTE_SEND, Opcode::SRAM_FREE,
                 Opcode::SRAM_FREE},
                {Opcode::DTE_RECV, Opcode::DTE_WAIT},
            }};
            std::sort(rooted_overlay_sequences.begin(),
                      rooted_overlay_sequences.end());
            std::sort(expected_sequences.begin(), expected_sequences.end());
            if (rooted_local_dag_ids !=
                    train_lineage_ids[
                        ManifestInputKindDto::GLOBAL_ACTION_DAG] ||
                rooted_overlay_sequences != expected_sequences)
                Fail("linked_program_manifest.fragments",
                     "rooted-AR requires both local DAG lineages and its exact six overlay record sequences");
        }
        std::set<std::tuple<ManifestInputKindDto, std::string, std::string>>
            actual_inputs;
        for (const ManifestInputDigestDto &digest : manifest.input_digests)
            actual_inputs.emplace(digest.kind, digest.artifact_id,
                                  digest.schema_version);
        if (actual_inputs != expected_inputs)
            Fail("linked_program_manifest.input_digests",
                 "mixed manifest input closure is not exact");
        RequireCanonicalBy(manifest.fragments,
                           "linked_program_manifest.fragments", OuterId);

        std::map<std::string, const CommandFragmentDto *> fragments;
        std::map<std::pair<std::string, LogicalCoreDto>,
                 const CoreFragmentStreamDto *> fragment_streams;
        std::set<std::string> declared_program_symbols;
        std::map<std::string, RuntimeSymbolDto> declared_runtime_symbols;
        std::map<std::string, const BufferAbiDto *> known_buffer_abi;
        std::map<std::pair<std::string, std::string>, const BufferAbiDto *>
            known_buffer_by_binding;
        std::map<std::string, const StateAbiDto *> known_state_abi;
        std::map<std::string, std::vector<StateTransferEndpointWitness>>
            state_transfer_endpoints;
        std::map<std::string, std::string> state_id_by_hbm_binding;

        std::map<std::string, std::set<LogicalCoreDto>> program_symbol_uses;
        std::set<std::tuple<std::string, LogicalCoreDto, uint64_t,
                            SemanticOperandId>> buffer_relocation_keys;
        std::set<std::tuple<std::string, LogicalCoreDto, uint64_t,
                            SemanticOperandId>> state_relocation_keys;
        for (const LinkedFragmentDto &linked : manifest.fragments) {
            const CommandFragmentDto &fragment = Leaf(linked);
            const bool wrapped = std::holds_alternative<RegionManifestDto>(linked);
            const bool valid_kind = wrapped
                ? fragment.kind == FragmentKindDto::ISA_REGION
                : (fragment.kind == FragmentKindDto::COARSE ||
                   fragment.kind == FragmentKindDto::STANDALONE_COLLECTIVE ||
                   fragment.kind == FragmentKindDto::STATE_IO ||
                   fragment.kind == FragmentKindDto::STATE_TRANSFER ||
                   fragment.kind == FragmentKindDto::MOE_TRANSFER ||
                   fragment.kind == FragmentKindDto::S2_LITE_ROOTED_AR);
            const bool rooted_fragment =
                fragment.kind == FragmentKindDto::S2_LITE_ROOTED_AR;
            if (rooted_fragment !=
                (fragment.producer_pass == "s2_lite_rooted_ar_lowering"))
                Fail("linked_program_manifest.fragments",
                     "S2_LITE_ROOTED_AR kind is reserved for its exact producer");
            const bool valid_source_global_dag = rooted_ar_link
                ? (rooted_fragment
                       ? fragment.source_global_dag_id ==
                             manifest.source_global_dag_id
                       : train_lineage_ids[
                             ManifestInputKindDto::GLOBAL_ACTION_DAG]
                                 .count(fragment.source_global_dag_id) == 1)
                : train_link
                ? train_lineage_ids[ManifestInputKindDto::GLOBAL_ACTION_DAG]
                      .count(fragment.source_global_dag_id) == 1
                : fragment.source_global_dag_id ==
                      manifest.source_global_dag_id;
            if (fragment.schema_version != kCommandFragmentSchemaVersion ||
                !valid_kind || !valid_source_global_dag)
                Fail("linked_program_manifest.fragments",
                     "leaf kind/wrapping/source DAG contract mismatch");
            if (wrapped) {
                const RegionManifestDto &region =
                    std::get<RegionManifestDto>(linked);
                if (region.schema_version != kRegionManifestSchemaVersion ||
                    region.fusion_plan_id.empty() || region.region_id.empty())
                    Fail("linked_program_manifest.fragments",
                         "invalid RegionManifest identity");
                RequireCanonical(region.target_dies,
                                 "region_manifest.target_dies");
                std::vector<uint64_t> leaf_dies;
                for (const CoreFragmentStreamDto &stream : fragment.core_streams)
                    leaf_dies.push_back(stream.logical_core.die_id);
                std::sort(leaf_dies.begin(), leaf_dies.end());
                leaf_dies.erase(std::unique(leaf_dies.begin(), leaf_dies.end()),
                                leaf_dies.end());
                if (region.target_dies != leaf_dies)
                    Fail("region_manifest.target_dies",
                         "must exactly equal leaf stream die coverage");
            }
            if (!fragments.emplace(fragment.id, &fragment).second)
                Fail("linked_program_manifest.fragments", "duplicate leaf fragment id");
            RequireCanonical(fragment.claimed_action_ids,
                             "command_fragment.claimed_action_ids");
            if (fragment.claimed_action_ids.empty() || fragment.core_streams.empty())
                Fail("command_fragment", "claims and core streams must be non-empty");
            RequireCanonicalBy(fragment.runtime_symbols,
                               "command_fragment.runtime_symbols",
                               [](const RuntimeSymbolDto &item) {
                                   return item.id;
                               });
            for (const RuntimeSymbolDto &symbol : fragment.runtime_symbols) {
                const auto inserted = declared_runtime_symbols.emplace(
                    symbol.id, symbol);
                if (!inserted.second &&
                    (inserted.first->second.kind != symbol.kind ||
                     inserted.first->second.source_ref != symbol.source_ref))
                    Fail("command_fragment.runtime_symbols",
                         "conflicting shared runtime symbol declaration");
            }
            RequireCanonicalBy(fragment.program_symbols,
                               "command_fragment.program_symbols",
                               [](const ProgramSymbolDto &item) { return item.id; });
            for (const ProgramSymbolDto &symbol : fragment.program_symbols)
                declared_program_symbols.insert(symbol.id);
            RequireCanonicalBy(fragment.core_streams,
                               "command_fragment.core_streams",
                               [](const CoreFragmentStreamDto &item) {
                                   return item.logical_core;
                               });
            for (const CoreFragmentStreamDto &stream : fragment.core_streams) {
                if (!fragment_streams.emplace(
                        std::make_pair(fragment.id, stream.logical_core),
                        &stream).second)
                    Fail("command_fragment.core_streams", "duplicate core stream");
                RequireCanonicalBy(stream.address_relocations,
                                   "core_fragment_stream.address_relocations",
                                   [](const AddressRelocationDto &item) {
                                       return std::make_pair(
                                           item.record_index,
                                           static_cast<uint16_t>(item.operand_id));
                                   });
                RequireCanonicalBy(stream.runtime_relocations,
                                   "core_fragment_stream.runtime_relocations",
                                   [](const RuntimeRelocationDto &item) {
                                       return std::make_pair(
                                           item.record_index,
                                           static_cast<uint8_t>(item.field));
                                   });
                for (const RuntimeRelocationDto &relocation :
                     stream.runtime_relocations) {
                    if (relocation.record_index >= stream.records.size())
                        Fail("core_fragment_stream.runtime_relocations",
                             "dangling record index");
                }
                for (const AddressRelocationDto &relocation :
                     stream.address_relocations) {
                    if (relocation.record_index >= stream.records.size())
                        Fail("core_fragment_stream.address_relocations",
                             "dangling record index");
                    auto &keys = relocation.operand_id ==
                                         SemanticOperandId::HBM_ADDRESS
                                     ? state_relocation_keys
                                     : buffer_relocation_keys;
                    keys.emplace(fragment.id, stream.logical_core,
                                 relocation.record_index,
                                 relocation.operand_id);
                    program_symbol_uses[relocation.symbol_ref].insert(
                        stream.logical_core);
                }
                std::set<std::string> record_actions;
                for (const RelocatableRecordDto &record : stream.records)
                    record_actions.insert(record.source_global_action_id);
                if (!std::includes(fragment.claimed_action_ids.begin(),
                                   fragment.claimed_action_ids.end(),
                                   record_actions.begin(), record_actions.end()))
                    Fail("command_fragment.core_streams",
                         "record origin is not claimed by the fragment");
            }
            std::set<std::string> emitted_actions;
            for (const CoreFragmentStreamDto &stream : fragment.core_streams)
                for (const RelocatableRecordDto &record : stream.records)
                    emitted_actions.insert(record.source_global_action_id);
            if (emitted_actions !=
                std::set<std::string>(fragment.claimed_action_ids.begin(),
                                      fragment.claimed_action_ids.end()))
                Fail("command_fragment.claimed_action_ids",
                     "claims must exactly equal emitted action origins");
            RequireCanonicalBy(fragment.buffer_abi,
                               "command_fragment.buffer_abi",
                               [](const BufferAbiDto &item) { return item.id; });
            for (const BufferAbiDto &abi : fragment.buffer_abi) {
                if (abi.size_bytes == 0 || abi.alignment_bytes == 0 ||
                    (abi.alignment_bytes & (abi.alignment_bytes - 1)) != 0 ||
                    abi.region_offset_bytes % abi.alignment_bytes != 0 ||
                    abi.lifetime_start >= abi.lifetime_end_exclusive ||
                    abi.tensor_slice.value_id != abi.value_id ||
                    abi.storage_id.empty() || abi.region_ref.empty())
                    Fail("command_fragment.buffer_abi",
                         "invalid size/alignment/lifetime/tensor/storage witness");
                RequireCanonical(abi.banks, "buffer_abi.banks");
                const auto previous = known_buffer_abi.find(abi.id);
                if (previous != known_buffer_abi.end() &&
                    !SameBufferAbi(*previous->second, abi))
                    Fail("command_fragment.buffer_abi",
                         "conflicting shared BufferABI definition");
                known_buffer_abi.emplace(abi.id, &abi);
                const auto binding = known_buffer_by_binding.emplace(
                    std::make_pair(abi.schedule_id, abi.binding_id), &abi);
                if (!binding.second && !SameBufferAbi(*binding.first->second, abi))
                    Fail("command_fragment.buffer_abi",
                         "one schedule binding has conflicting BufferABI values");
            }
            if ((fragment.kind == FragmentKindDto::STATE_IO) !=
                !fragment.state_abi.empty())
                Fail("command_fragment.state_abi",
                     "STATE_IO fragments require StateABI and all other fragments forbid it");
            RequireCanonicalBy(fragment.state_abi,
                               "command_fragment.state_abi",
                               [](const StateAbiDto &item) { return item.id; });
            for (const StateAbiDto &abi : fragment.state_abi) {
                if (abi.dtype == BufferDTypeDto::INT32)
                    Fail("command_fragment.state_abi",
                         "StateABI dtype must be fp16 or fp32");
                const auto previous = known_state_abi.find(abi.id);
                if (previous != known_state_abi.end() &&
                    !SameStateAbi(*previous->second, abi))
                    Fail("command_fragment.state_abi",
                         "conflicting shared StateABI definition");
                known_state_abi.emplace(abi.id, &abi);
                const auto binding = state_id_by_hbm_binding.emplace(
                    abi.hbm_binding_ref, abi.id);
                if (!binding.second && binding.first->second != abi.id)
                    Fail("command_fragment.state_abi",
                         "one HBM binding cannot name multiple StateABI ids");
            }
            if (fragment.kind == FragmentKindDto::STATE_TRANSFER) {
                for (StateTransferEndpointWitness endpoint :
                     ValidateStateTransferFragments(
                         fragment, "command_fragment"))
                    state_transfer_endpoints[endpoint.fsm_ref].push_back(
                        std::move(endpoint));
            }
        }
        for (const auto &entry : known_buffer_abi) {
            const BufferAbiDto &abi = *entry.second;
            const bool aliased = abi.ownership == BufferOwnershipDto::ALIASED;
            if (aliased != abi.alias_of.has_value())
                Fail("command_fragment.buffer_abi",
                     "alias ownership and alias_of must agree");
            if (!aliased)
                continue;
            const auto root_it = known_buffer_by_binding.find(
                std::make_pair(abi.schedule_id, *abi.alias_of));
            if (root_it == known_buffer_by_binding.end())
                Fail("command_fragment.buffer_abi",
                     "aliased BufferABI references a missing canonical root");
            const BufferAbiDto &root = *root_it->second;
            if (root.ownership == BufferOwnershipDto::ALIASED || root.alias_of ||
                root.schedule_id != abi.schedule_id ||
                !(root.logical_core == abi.logical_core) ||
                root.region_ref != abi.region_ref ||
                root.region_offset_bytes != abi.region_offset_bytes ||
                root.size_bytes != abi.size_bytes ||
                root.alignment_bytes != abi.alignment_bytes ||
                root.banks != abi.banks || root.storage_id != abi.storage_id ||
                root.dtype != abi.dtype || root.layout != abi.layout ||
                root.tensor_slice.offset != abi.tensor_slice.offset ||
                root.tensor_slice.shape != abi.tensor_slice.shape ||
                root.lifetime_start > abi.lifetime_start ||
                root.lifetime_end_exclusive < abi.lifetime_end_exclusive)
                Fail("command_fragment.buffer_abi",
                     "aliased BufferABI must preserve one enclosing canonical root geometry");
        }
        for (const auto &entry : state_transfer_endpoints) {
            const std::vector<StateTransferEndpointWitness> &endpoints =
                entry.second;
            if (endpoints.size() != 2 ||
                endpoints[0].source == endpoints[1].source)
                Fail("linked_program_manifest.fragments",
                     "one STATE_TRANSFER DTE_FSM must close exactly one SEND/RECV leaf pair");
            const StateTransferEndpointWitness &source =
                endpoints[0].source ? endpoints[0] : endpoints[1];
            const StateTransferEndpointWitness &destination =
                endpoints[0].source ? endpoints[1] : endpoints[0];
            if (source.length_bytes != destination.length_bytes)
                Fail("linked_program_manifest.fragments",
                     "STATE_TRANSFER SEND/RECV lengths must match exactly");
        }

        std::map<std::string, SymbolEntry> symbols;
        ProgramArtifact artifact;
        artifact.capabilities = manifest.capabilities;
        RequireCanonicalBy(manifest.program_symbol_definitions,
                           "linked_program_manifest.program_symbol_definitions",
                           [](const ProgramSymbolDefinitionDto &item) {
                               return item.symbol.id;
                           });
        std::set<std::string> names;
        std::map<std::string, const ProgramSymbolDefinitionDto *>
            regions_by_ref;
        for (std::size_t index = 0;
             index < manifest.program_symbol_definitions.size(); ++index) {
            const ProgramSymbolDefinitionDto &definition =
                manifest.program_symbol_definitions[index];
            if (!names.insert(definition.name).second)
                Fail("linked_program_manifest.program_symbol_definitions",
                     "program symbol names must be globally unique");
            RequireCanonical(definition.logical_cores,
                             "program_symbol_definition.logical_cores");
            if (definition.logical_cores.empty())
                Fail("program_symbol_definition.logical_cores", "must be non-empty");
            const auto uses = program_symbol_uses.find(definition.symbol.id);
            const std::vector<LogicalCoreDto> used_cores =
                uses == program_symbol_uses.end()
                    ? std::vector<LogicalCoreDto>{}
                    : std::vector<LogicalCoreDto>(uses->second.begin(),
                                                  uses->second.end());
            if (definition.logical_cores != used_cores)
                Fail("program_symbol_definition[" +
                         definition.symbol.id + "].logical_cores",
                     "must exactly equal relocation execution scope");
            if (definition.symbol.kind == ProgramSymbolKind::SRAM_REGION) {
                if (definition.size_bytes == 0 ||
                    !regions_by_ref.emplace(definition.symbol.source_ref,
                                            &definition).second)
                    Fail("program_symbol_definition",
                         "SRAM_REGION source_ref must be unique and non-empty-sized");
            } else if (definition.symbol.kind ==
                       ProgramSymbolKind::SRAM_LABEL) {
                if (definition.value != 0 || definition.size_bytes != 0)
                    Fail("program_symbol_definition",
                         "SRAM_LABEL value and size must be zero");
            } else if (definition.size_bytes == 0) {
                Fail("program_symbol_definition",
                     "ABSOLUTE_ADDRESS span must be non-empty");
            }
            symbols.emplace(definition.symbol.id,
                            SymbolEntry{&definition, index});
            artifact.strings.push_back(definition.name);
            artifact.symbols.push_back(
                {index, definition.symbol.kind, 0, definition.value,
                 definition.size_bytes});
        }
        if (declared_program_symbols.size() != symbols.size())
            Fail("linked_program_manifest.program_symbol_definitions",
                 "definitions must exactly cover fragment declarations");
        for (const auto &entry : fragments) {
            for (const ProgramSymbolDto &declaration :
                 entry.second->program_symbols) {
                const auto definition = symbols.find(declaration.id);
                if (definition == symbols.end() ||
                    !SameProgramSymbol(
                        declaration, definition->second.definition->symbol))
                    Fail("linked_program_manifest.program_symbol_definitions",
                         "definition disagrees with fragment declaration");
            }
        }

        RequireCanonicalBy(manifest.fragment_interfaces,
                           "linked_program_manifest.fragment_interfaces",
                           [](const FragmentInterfaceDto &item) {
                               return item.fragment_id;
                           });
        std::set<std::string> interface_ids;
        std::map<std::string, std::string> program_exporter;
        std::map<std::string, std::string> runtime_exporter;
        for (const FragmentInterfaceDto &interface :
             manifest.fragment_interfaces) {
            const auto fragment = fragments.find(interface.fragment_id);
            if (fragment == fragments.end() ||
                !interface_ids.insert(interface.fragment_id).second)
                Fail("linked_program_manifest.fragment_interfaces",
                     "interfaces must bijectively cover leaf fragments");
            RequireCanonical(interface.runtime_imports, "fragment_interface.runtime_imports");
            RequireCanonical(interface.runtime_exports, "fragment_interface.runtime_exports");
            RequireCanonical(interface.program_imports, "fragment_interface.program_imports");
            RequireCanonical(interface.program_exports, "fragment_interface.program_exports");
            std::set<std::string> runtime_local;
            for (const RuntimeSymbolDto &symbol :
                 fragment->second->runtime_symbols)
                runtime_local.insert(symbol.id);
            std::set<std::string> runtime_closure(
                interface.runtime_imports.begin(),
                interface.runtime_imports.end());
            for (const std::string &id : interface.runtime_exports) {
                if (!runtime_closure.insert(id).second)
                    Fail("fragment_interface",
                         "runtime imports/exports overlap");
                if (!runtime_exporter.emplace(id, interface.fragment_id).second)
                    Fail("fragment_interface.runtime_exports",
                         "runtime symbol has multiple exporters");
            }
            if (runtime_closure != runtime_local)
                Fail("fragment_interface",
                     "runtime import/export closure differs from declarations");
            auto validate_credits = [&](const std::vector<EventCreditDto> &credits,
                                        const std::string &credit_path) {
                RequireCanonicalBy(credits, credit_path,
                                   [](const EventCreditDto &item) {
                                       return item.symbol_ref;
                                   });
                for (const EventCreditDto &credit : credits) {
                    const auto symbol = declared_runtime_symbols.find(
                        credit.symbol_ref);
                    if (credit.count == 0 ||
                        symbol == declared_runtime_symbols.end() ||
                        symbol->second.kind != RuntimeSymbolKindDto::EVENT_TAG)
                        Fail(credit_path,
                             "event credits require declared EVENT_TAG symbols");
                }
            };
            validate_credits(interface.entry_events,
                             "fragment_interface.entry_events");
            validate_credits(interface.exit_events,
                             "fragment_interface.exit_events");
            std::map<std::string, uint64_t> expected_entry;
            std::map<std::string, uint64_t> expected_exit;
            for (const CoreFragmentStreamDto &stream :
                 fragment->second->core_streams) {
                for (const RelocatableRecordDto &record : stream.records) {
                    const std::string tag = RuntimeSymbolRef(
                        record, RuntimeOperandFieldDto::EVENT_TAG);
                    if (record.opcode == Opcode::EVENT_SET) {
                        if (tag.empty())
                            Fail("fragment_interface.exit_events",
                                 "EVENT_SET is missing its tag symbol");
                        ++expected_exit[tag];
                    } else if (record.opcode == Opcode::EVENT_WAIT) {
                        if (tag.empty() || record.operands.size() != 4)
                            Fail("fragment_interface.entry_events",
                                 "EVENT_WAIT is missing its tag/count ABI");
                        expected_entry[tag] += LiteralU64(
                            record.operands[3],
                            "fragment_interface.entry_events");
                    }
                }
            }
            auto credit_map = [](const std::vector<EventCreditDto> &credits) {
                std::map<std::string, uint64_t> result;
                for (const EventCreditDto &credit : credits)
                    result.emplace(credit.symbol_ref, credit.count);
                return result;
            };
            if (credit_map(interface.entry_events) != expected_entry ||
                credit_map(interface.exit_events) != expected_exit)
                Fail("fragment_interface",
                     "event credits do not exactly match EVENT_WAIT/SET records");
            std::set<std::string> local;
            for (const ProgramSymbolDto &symbol : fragment->second->program_symbols)
                local.insert(symbol.id);
            std::set<std::string> closure(interface.program_imports.begin(),
                                          interface.program_imports.end());
            for (const std::string &id : interface.program_exports) {
                if (!closure.insert(id).second)
                    Fail("fragment_interface",
                         "program imports/exports overlap");
            }
            if (closure != local)
                Fail("fragment_interface",
                     "program import/export closure differs from declarations");
            for (const std::string &id : interface.program_exports) {
                if (!program_exporter.emplace(id, interface.fragment_id).second)
                    Fail("fragment_interface.program_exports",
                         "program symbol has multiple exporters");
            }
        }
        if (interface_ids.size() != fragments.size() ||
            program_exporter.size() != declared_program_symbols.size() ||
            runtime_exporter.size() != declared_runtime_symbols.size())
            Fail("linked_program_manifest.fragment_interfaces",
                 "interfaces/exporters do not close over all fragments and symbols");

        RequireCanonicalBy(manifest.core_bindings,
                           "linked_program_manifest.core_bindings",
                           [](const CoreRuntimeBindingDto &item) {
                               return item.logical_core;
                           });
        if (manifest.core_bindings.empty())
            Fail("linked_program_manifest.core_bindings", "must be non-empty");
        std::map<LogicalCoreDto, uint64_t> runtime_ids;
        std::set<uint64_t> unique_runtime_ids;
        for (const CoreRuntimeBindingDto &binding : manifest.core_bindings) {
            if (binding.runtime_core_id > 0xffff ||
                !runtime_ids.emplace(binding.logical_core,
                                     binding.runtime_core_id).second ||
                !unique_runtime_ids.insert(binding.runtime_core_id).second)
                Fail("linked_program_manifest.core_bindings",
                     "runtime bindings are not unique uint16 ids");
        }
        if (manifest.core_streams.size() != manifest.core_bindings.size())
            Fail("linked_program_manifest.core_streams",
                 "must bijectively follow core bindings");

        RequireCanonicalBy(manifest.runtime_symbol_definitions,
                           "linked_program_manifest.runtime_symbol_definitions",
                           [](const RuntimeSymbolDefinitionDto &item) {
                               return item.symbol.id;
                           });
        std::map<std::string, RuntimeEntry> runtime_symbols;
        std::map<std::string, uint64_t> start_tags;
        std::map<RuntimeSymbolKindDto, uint64_t> next_runtime{{
            {RuntimeSymbolKindDto::START_TAG, 0},
            {RuntimeSymbolKindDto::EVENT_TAG, 0},
            {RuntimeSymbolKindDto::DTE_TOKEN, 1},
            {RuntimeSymbolKindDto::DTE_FSM, 1},
            {RuntimeSymbolKindDto::GROUP, 1},
        }};
        std::set<std::string> defined_leaf_runtime;
        for (const RuntimeSymbolDefinitionDto &definition :
             manifest.runtime_symbol_definitions) {
            RequireCanonical(definition.logical_cores,
                             "runtime_symbol_definition.logical_cores");
            if (definition.logical_cores.empty())
                Fail("runtime_symbol_definition.logical_cores",
                     "must be non-empty");
            uint64_t value = 0;
            const auto declaration = declared_runtime_symbols.find(
                definition.symbol.id);
            if (definition.symbol.kind == RuntimeSymbolKindDto::START_TAG) {
                if (declaration != declared_runtime_symbols.end() ||
                    definition.logical_cores.size() != 1 ||
                    definition.source_action_id ||
                    definition.destination_action_id)
                    Fail("runtime_symbol_definition",
                         "START_TAG must be top-only and core-local");
            } else {
                if (declaration == declared_runtime_symbols.end() ||
                    declaration->second.kind != definition.symbol.kind ||
                    declaration->second.source_ref !=
                        definition.symbol.source_ref)
                    Fail("runtime_symbol_definition",
                         "definition disagrees with leaf declaration");
                defined_leaf_runtime.insert(definition.symbol.id);
            }
            switch (definition.symbol.kind) {
            case RuntimeSymbolKindDto::START_TAG:
            case RuntimeSymbolKindDto::EVENT_TAG:
            case RuntimeSymbolKindDto::DTE_TOKEN:
            case RuntimeSymbolKindDto::DTE_FSM:
                value = next_runtime.at(definition.symbol.kind)++;
                break;
            case RuntimeSymbolKindDto::RUNTIME_CORE: {
                if (definition.logical_cores.size() != 1 ||
                    definition.source_action_id ||
                    definition.destination_action_id)
                    Fail("runtime_symbol_definition",
                         "RUNTIME_CORE must directly name one bound core");
                const auto core = runtime_ids.find(definition.logical_cores[0]);
                if (core == runtime_ids.end())
                    Fail("runtime_symbol_definition",
                         "RUNTIME_CORE references an unbound core");
                value = core->second;
                break;
            }
            case RuntimeSymbolKindDto::GROUP:
                Fail("runtime_symbol_definition",
                     "GROUP runtime symbols are not supported in mixed MVP");
            }
            if (value > 0xffff)
                Fail("runtime_symbol_definition",
                     "runtime namespace value exceeds uint16");
            runtime_symbols.emplace(definition.symbol.id,
                                    RuntimeEntry{&definition, value});
            if (definition.symbol.kind == RuntimeSymbolKindDto::START_TAG)
                start_tags.emplace(definition.symbol.id, value);
        }
        if (defined_leaf_runtime.size() != declared_runtime_symbols.size())
            Fail("linked_program_manifest.runtime_symbol_definitions",
                 "definitions must exactly cover all leaf runtime symbols");

        std::set<std::tuple<std::string, LogicalCoreDto, uint64_t, std::string>>
            actual_record_refs;
        std::set<std::tuple<std::string, LogicalCoreDto, uint64_t, std::string>>
            expected_record_refs;
        for (const auto &entry : fragment_streams) {
            for (std::size_t index = 0; index < entry.second->records.size(); ++index)
                expected_record_refs.emplace(
                    entry.first.first, entry.first.second, index,
                    entry.second->records[index].source_global_action_id);
        }

        struct PendingCore {
            uint64_t runtime_id = 0;
            const LinkedCoreStreamDto *linked = nullptr;
        };
        std::vector<PendingCore> pending;
        for (std::size_t index = 0; index < manifest.core_streams.size(); ++index) {
            const LinkedCoreStreamDto &stream = manifest.core_streams[index];
            const CoreRuntimeBindingDto &binding = manifest.core_bindings[index];
            if (!(stream.logical_core == binding.logical_core) ||
                stream.runtime_core_id != binding.runtime_core_id)
                Fail("linked_program_manifest.core_streams",
                     "stream order/runtime id disagrees with binding");
            pending.push_back({stream.runtime_core_id, &stream});
        }
        std::sort(pending.begin(), pending.end(),
                  [](const PendingCore &left, const PendingCore &right) {
                      return left.runtime_id < right.runtime_id;
                  });

        struct RuntimeUse {
            RuntimeOperandFieldDto field;
            Opcode opcode;
            std::string action_id;
            LogicalCoreDto logical_core;
            std::string fragment_id;
            uint64_t fragment_record_index = 0;
        };
        std::map<std::string, std::vector<RuntimeUse>> runtime_uses;
        std::map<std::string, LogicalCoreDto> action_cores;
        std::map<std::string, std::set<Opcode>> action_opcodes;
        std::map<LogicalCoreDto, std::string> first_action_by_core;

        for (std::size_t core_index = 0; core_index < pending.size(); ++core_index) {
            const LinkedCoreStreamDto &linked = *pending[core_index].linked;
            ProgramCore core;
            core.core_id = pending[core_index].runtime_id;
            std::vector<const RelocatableRecordDto *> source_records;
            for (std::size_t instruction = 0;
                 instruction < linked.records.size(); ++instruction) {
                const LinkedRecordRefDto &reference = linked.records[instruction];
                const auto stream = fragment_streams.find(
                    std::make_pair(reference.fragment_id, linked.logical_core));
                if (stream == fragment_streams.end() ||
                    reference.fragment_record_index >= stream->second->records.size())
                    Fail("linked_program_manifest.core_streams",
                         "linked record references an unknown fragment/core/index");
                const RelocatableRecordDto &record =
                    stream->second->records[reference.fragment_record_index];
                action_opcodes[record.source_global_action_id].insert(
                    record.opcode);
                first_action_by_core.emplace(linked.logical_core,
                                             record.source_global_action_id);
                if (record.source_global_action_id !=
                    reference.source_global_action_id)
                    Fail("linked_program_manifest.core_streams",
                         "linked record source action mismatch");
                if (!actual_record_refs.emplace(
                        reference.fragment_id, linked.logical_core,
                        reference.fragment_record_index,
                        reference.source_global_action_id).second)
                    Fail("linked_program_manifest.core_streams",
                         "fragment record is linked more than once");
                const RelocationMap record_relocations =
                    ValidateRecordRelocations(
                        record, reference.fragment_record_index,
                        *stream->second, symbols, linked.logical_core,
                        "linked_program_manifest.core_streams.record");
                const RuntimeRelocationMap record_runtime_relocations =
                    ValidateRuntimeRecordRelocations(
                        record, reference.fragment_record_index,
                        *stream->second, runtime_symbols,
                        linked.logical_core,
                        "linked_program_manifest.core_streams.record");
                const auto action_core = action_cores.emplace(
                    record.source_global_action_id, linked.logical_core);
                if (!action_core.second &&
                    !(action_core.first->second == linked.logical_core))
                    Fail("linked_program_manifest.core_streams",
                         "one GlobalAction cannot execute on multiple logical cores");
                for (const auto &runtime_relocation :
                     record_runtime_relocations) {
                    runtime_uses[runtime_relocation.second.symbol_ref].push_back(
                        {runtime_relocation.first, record.opcode,
                         record.source_global_action_id, linked.logical_core,
                         reference.fragment_id,
                         reference.fragment_record_index});
                }
                core.records.push_back(FinalizeRecord(
                    record, record_relocations, record_runtime_relocations,
                    symbols, runtime_symbols, artifact, core_index,
                    instruction,
                    "linked_program_manifest.core_streams.record"));
                source_records.push_back(&record);
            }
            ValidateActionSequence(
                source_records, "linked_program_manifest.core_streams");
            artifact.cores.push_back(std::move(core));
        }
        if (actual_record_refs != expected_record_refs)
            Fail("linked_program_manifest.core_streams",
                 "linked streams do not cover every fragment record exactly once");

        RequireCanonicalBy(manifest.address_operand_bindings,
                           "linked_program_manifest.address_operand_bindings",
                           [](const AddressOperandBindingDto &item) {
                               return std::make_tuple(
                                   item.logical_core, item.fragment_id,
                                   item.fragment_record_index,
                                   static_cast<uint16_t>(item.operand_id));
                           });
        std::set<std::tuple<std::string, LogicalCoreDto, uint64_t,
                            SemanticOperandId>> binding_keys;
        for (const AddressOperandBindingDto &binding :
             manifest.address_operand_bindings) {
            if (binding.operand_id == SemanticOperandId::HBM_ADDRESS)
                Fail("linked_program_manifest.address_operand_bindings",
                     "HBM_ADDRESS cannot use a BufferABI closure");

            if (binding.buffer_abi_ids.empty() ||
                binding.tensor_slices.size() != binding.buffer_abi_ids.size())
                Fail("linked_program_manifest.address_operand_bindings",
                     "address witness must name equal non-zero BufferABI/view tuples");
            std::set<std::string> unique_abi_ids;
            std::vector<const BufferAbiDto *> abis;
            for (const std::string &abi_id : binding.buffer_abi_ids) {
                const auto abi = known_buffer_abi.find(abi_id);
                if (abi == known_buffer_abi.end() ||
                    !unique_abi_ids.insert(abi_id).second)
                    Fail("linked_program_manifest.address_operand_bindings",
                         "references an unknown or duplicate BufferABI");
                if (!(abi->second->logical_core == binding.logical_core))
                    Fail("linked_program_manifest.address_operand_bindings",
                         "BufferABI witness belongs to the wrong core");
                abis.push_back(abi->second);
            }
            const auto owner_fragment = fragments.find(binding.fragment_id);
            if (owner_fragment != fragments.end() &&
                owner_fragment->second->kind ==
                    FragmentKindDto::STATE_TRANSFER &&
                binding.buffer_abi_ids !=
                    std::vector<std::string>{
                        owner_fragment->second->buffer_abi.front().id})
                Fail("linked_program_manifest.address_operand_bindings",
                     "STATE_TRANSFER address and lifecycle witnesses must use its one local BufferABI");
            const auto stream = fragment_streams.find(
                std::make_pair(binding.fragment_id, binding.logical_core));
            if (stream == fragment_streams.end() ||
                binding.fragment_record_index >= stream->second->records.size())
                Fail("linked_program_manifest.address_operand_bindings",
                     "binding references an unknown record");
            const auto relocation = std::find_if(
                stream->second->address_relocations.begin(),
                stream->second->address_relocations.end(),
                [&](const AddressRelocationDto &item) {
                    return item.record_index == binding.fragment_record_index &&
                           item.operand_id == binding.operand_id;
                });
            if (relocation == stream->second->address_relocations.end())
                Fail("linked_program_manifest.address_operand_bindings",
                     "binding has no exact address relocation");
            const RelocatableRecordDto &record =
                stream->second->records[binding.fragment_record_index];
            const std::optional<BufferDTypeDto> expected_dtype =
                ExpectedBufferDType(record, binding.operand_id);
            const bool mixed_matmul_output =
                record.opcode == Opcode::MATMUL &&
                binding.operand_id ==
                    SemanticOperandId::COMPUTE_OUTPUT_ADDRESS;
            if (mixed_matmul_output && std::any_of(
                    abis.begin(), abis.end(), [&](const BufferAbiDto *abi) {
                        return abi->dtype != BufferDTypeDto::FP16 &&
                               abi->dtype != BufferDTypeDto::FP32;
                    }))
                Fail("linked_program_manifest.address_operand_bindings",
                     "MATMUL output BufferABI must be FP16 or exact FP32 WGRAD");
            if (!mixed_matmul_output && expected_dtype && std::any_of(
                    abis.begin(), abis.end(), [&](const BufferAbiDto *abi) {
                        return abi->dtype != *expected_dtype;
                    }))
                Fail("linked_program_manifest.address_operand_bindings",
                     "BufferABI dtype does not match the record operand ABI");
            const ProgramSymbolDefinitionDto &definition =
                *symbols.at(relocation->symbol_ref).definition;
            const ProgramSymbolDefinitionDto *region = nullptr;
            uint64_t next_offset = abis.front()->region_offset_bytes;
            const uint64_t span_offset = next_offset;
            std::vector<DenseViewSpan> views;
            for (std::size_t index = 0; index < abis.size(); ++index) {
                const BufferAbiDto *abi = abis[index];
                if (abi->region_offset_bytes != next_offset)
                    Fail("linked_program_manifest.address_operand_bindings",
                         "multi-buffer witness must be contiguous and ordered");
                const auto found_region = regions_by_ref.find(abi->region_ref);
                if (found_region == regions_by_ref.end() ||
                    !ContainsCore(found_region->second->logical_cores,
                                  binding.logical_core) ||
                    abi->region_offset_bytes > found_region->second->size_bytes ||
                    abi->size_bytes > found_region->second->size_bytes -
                                              abi->region_offset_bytes)
                    Fail("linked_program_manifest.address_operand_bindings",
                         "BufferABI is outside its named SRAM region/core");
                if (region && region != found_region->second)
                    Fail("linked_program_manifest.address_operand_bindings",
                         "one address operand cannot span multiple regions");
                region = found_region->second;
                if (next_offset > std::numeric_limits<uint64_t>::max() -
                                      abi->size_bytes)
                    Fail("linked_program_manifest.address_operand_bindings",
                         "BufferABI span overflows uint64");
                next_offset += abi->size_bytes;
                DenseViewSpan view = DenseRowMajorViewSpan(
                    abi->tensor_slice, binding.tensor_slices[index], abi->dtype,
                    "linked_program_manifest.address_operand_bindings.tensor_slices");
                if (view.root_length != abi->size_bytes)
                    Fail("linked_program_manifest.address_operand_bindings",
                         "BufferABI size must equal its tight dense root span");
                views.push_back(view);
            }
            const uint64_t span_size = next_offset - span_offset;
            if (definition.symbol.kind ==
                ProgramSymbolKind::ABSOLUTE_ADDRESS) {
                if (!region || region->value >
                                   std::numeric_limits<uint64_t>::max() -
                                       span_offset)
                    Fail("linked_program_manifest.address_operand_bindings",
                         "absolute region base plus offset overflows uint64");
                const bool exact_symbol_source =
                    definition.symbol.source_ref == abis.front()->binding_id ||
                    (abis.front()->alias_of &&
                     definition.symbol.source_ref == *abis.front()->alias_of);
                if (!exact_symbol_source ||
                    definition.value != region->value + span_offset ||
                    definition.size_bytes != span_size ||
                    relocation->addend < 0)
                    Fail("linked_program_manifest.address_operand_bindings",
                         "absolute symbol must preserve root base/span with a non-negative addend");

                uint64_t expected_addend = 0;
                if (abis.size() == 1) {
                    expected_addend = views.front().addend;
                } else {
                    if (record.opcode != Opcode::LOCAL_REDUCE ||
                        binding.operand_id != SemanticOperandId::SOURCE_ADDRESS ||
                        std::any_of(views.begin(), views.end(),
                                    [](const DenseViewSpan &view) {
                                        return view.addend != 0 ||
                                               view.length != view.root_length;
                                    }))
                        Fail("linked_program_manifest.address_operand_bindings",
                             "multi-buffer ABS witness is reserved for whole-root rank-major LOCAL_REDUCE inputs");
                }
                if (expected_addend > static_cast<uint64_t>(
                                           std::numeric_limits<int64_t>::max()) ||
                    relocation->addend !=
                        static_cast<int64_t>(expected_addend))
                    Fail("linked_program_manifest.address_operand_bindings",
                         "ABS relocation addend does not equal the dense view byte addend");

                const uint64_t access_bytes = OperandAccessBytes(
                    record, binding.operand_id,
                    "linked_program_manifest.address_operand_bindings");
                if (abis.size() == 1) {
                    if (access_bytes > views.front().length ||
                        expected_addend > span_size ||
                        access_bytes > span_size - expected_addend)
                        Fail("linked_program_manifest.address_operand_bindings",
                             "payload byte extent does not fit its dense view/root");
                } else {
                    const uint64_t input_count = LiteralU64(
                        record.operands[6], "LOCAL_REDUCE.input_count");
                    const uint64_t input_stride = LiteralU64(
                        record.operands[8], "LOCAL_REDUCE.input_stride_bytes");
                    if (input_count != abis.size() ||
                        std::any_of(views.begin(), views.end(),
                                    [&](const DenseViewSpan &view) {
                                        return view.length != input_stride;
                                    }) ||
                        access_bytes != span_size)
                        Fail("linked_program_manifest.address_operand_bindings",
                             "LOCAL_REDUCE source views must exactly form its rank-major payload span");
                }
            } else if (definition.symbol.kind ==
                       ProgramSymbolKind::SRAM_REGION) {
                if (record.opcode != Opcode::SRAM_ALLOC_AT ||
                    binding.operand_id != SemanticOperandId::REGION_NAME ||
                    abis.size() != 1 || relocation->addend != 0 ||
                    views.front().addend != 0 ||
                    views.front().length != views.front().root_length ||
                    definition.symbol.source_ref != abis.front()->region_ref)
                    Fail("linked_program_manifest.address_operand_bindings",
                         "SRAM region relocation must be a zero-addend root ALLOC_AT witness");
            } else {
                if (abis.size() != 1 || relocation->addend != 0 ||
                    definition.symbol.source_ref != abis.front()->storage_id)
                    Fail("linked_program_manifest.address_operand_bindings",
                         "label relocation does not match BufferABI storage");
                if ((record.opcode == Opcode::SRAM_ALLOC_AT ||
                     record.opcode == Opcode::SRAM_FREE) &&
                    (views.front().addend != 0 ||
                     views.front().length != views.front().root_length))
                    Fail("linked_program_manifest.address_operand_bindings",
                         "SRAM lifecycle label must witness the whole root backing");
            }
            if (binding.operand_id == SemanticOperandId::REGION_NAME) {
                if (record.opcode != Opcode::SRAM_ALLOC_AT ||
                    record.operands.size() != 7 || abis.size() != 1 ||
                    LiteralU64(record.operands[2], "SRAM_ALLOC_AT") !=
                        abis.front()->region_offset_bytes ||
                    LiteralU64(record.operands[3], "SRAM_ALLOC_AT") !=
                        abis.front()->size_bytes ||
                    LiteralU64(record.operands[4], "SRAM_ALLOC_AT") !=
                        abis.front()->alignment_bytes)
                    Fail("linked_program_manifest.address_operand_bindings",
                         "SRAM_ALLOC_AT payload differs from BufferABI witness");
            }
            if (!binding_keys.emplace(binding.fragment_id,
                                      binding.logical_core,
                                      binding.fragment_record_index,
                                      binding.operand_id).second)
                Fail("linked_program_manifest.address_operand_bindings",
                     "duplicate address binding key");
        }
        if (binding_keys != buffer_relocation_keys)
            Fail("linked_program_manifest.address_operand_bindings",
                 "must exactly close every address relocation");

        RequireCanonicalBy(
            manifest.state_operand_bindings,
            "linked_program_manifest.state_operand_bindings",
            [](const StateOperandBindingDto &item) {
                return std::make_tuple(
                    item.logical_core, item.fragment_id,
                    item.fragment_record_index,
                    static_cast<uint16_t>(item.operand_id));
            });
        std::set<std::tuple<std::string, LogicalCoreDto, uint64_t,
                            SemanticOperandId>> state_binding_keys;
        std::set<std::string> used_state_abi_ids;
        for (const StateOperandBindingDto &binding :
             manifest.state_operand_bindings) {
            if (binding.operand_id != SemanticOperandId::HBM_ADDRESS)
                Fail("linked_program_manifest.state_operand_bindings",
                     "StateOperandBinding only supports HBM_ADDRESS");
            const auto abi = known_state_abi.find(binding.state_abi_id);
            if (abi == known_state_abi.end())
                Fail("linked_program_manifest.state_operand_bindings",
                     "references an unknown StateABI");
            const auto stream = fragment_streams.find(
                std::make_pair(binding.fragment_id, binding.logical_core));
            if (stream == fragment_streams.end() ||
                binding.fragment_record_index >= stream->second->records.size())
                Fail("linked_program_manifest.state_operand_bindings",
                     "binding references an unknown record");
            const auto relocation = std::find_if(
                stream->second->address_relocations.begin(),
                stream->second->address_relocations.end(),
                [&](const AddressRelocationDto &item) {
                    return item.record_index ==
                               binding.fragment_record_index &&
                           item.operand_id == binding.operand_id;
                });
            if (relocation == stream->second->address_relocations.end())
                Fail("linked_program_manifest.state_operand_bindings",
                     "binding has no exact HBM relocation");
            const auto symbol = symbols.find(relocation->symbol_ref);
            if (symbol == symbols.end())
                Fail("linked_program_manifest.state_operand_bindings",
                     "HBM relocation references an unknown program symbol");
            const RelocatableRecordDto &record =
                stream->second->records[binding.fragment_record_index];
            const StateAbiDto &state = *abi->second;
            const ProgramSymbolDefinitionDto &definition =
                *symbol->second.definition;
            if (record.opcode != Opcode::LSU_LOAD &&
                record.opcode != Opcode::LSU_STORE)
                Fail("linked_program_manifest.state_operand_bindings",
                     "HBM_ADDRESS is only legal on blocking LSU records");
            if (record.operands.size() != 3)
                Fail("linked_program_manifest.state_operand_bindings",
                     "blocking LSU requires three operands");
            const uint64_t size_bytes = LiteralU64(
                record.operands[1], "blocking LSU.size_bytes");
            if (relocation->addend < 0)
                Fail("linked_program_manifest.state_operand_bindings",
                     "blocking LSU HBM addend must be non-negative");
            const uint64_t state_addend =
                static_cast<uint64_t>(relocation->addend);
            if (relocation->symbol_kind !=
                    ProgramSymbolKind::ABSOLUTE_ADDRESS ||
                definition.symbol.kind !=
                    ProgramSymbolKind::ABSOLUTE_ADDRESS ||
                definition.symbol.source_ref != state.hbm_binding_ref ||
                definition.value != state.address ||
                definition.size_bytes != state.size_bytes ||
                binding.logical_core.die_id != state.die_id)
                Fail("linked_program_manifest.state_operand_bindings",
                     "HBM relocation/definition must exactly preserve StateABI");
            if (state_addend > state.size_bytes ||
                size_bytes > state.size_bytes - state_addend)
                Fail("linked_program_manifest.state_operand_bindings",
                     "blocking LSU byte range exceeds StateABI");
            if ((record.opcode == Opcode::LSU_LOAD &&
                 state.access == StateAccessDto::RESERVED) ||
                (record.opcode == Opcode::LSU_STORE &&
                 state.access != StateAccessDto::READ_WRITE))
                Fail("linked_program_manifest.state_operand_bindings",
                     "blocking LSU direction is forbidden by StateABI access");
            if (!state_binding_keys.emplace(
                    binding.fragment_id, binding.logical_core,
                    binding.fragment_record_index, binding.operand_id).second)
                Fail("linked_program_manifest.state_operand_bindings",
                     "duplicate state binding key");
            used_state_abi_ids.insert(state.id);
        }
        if (state_binding_keys != state_relocation_keys)
            Fail("linked_program_manifest.state_operand_bindings",
                 "must exactly close every HBM address relocation");
        std::set<std::string> declared_state_abi_ids;
        for (const auto &entry : known_state_abi)
            declared_state_abi_ids.insert(entry.first);
        if (used_state_abi_ids != declared_state_abi_ids)
            Fail("linked_program_manifest.state_operand_bindings",
                 "every StateABI requires a linked HBM operand witness");

        auto require_action_core = [&](const std::optional<std::string> &action,
                                       const LogicalCoreDto &core,
                                       const std::string &endpoint_path) {
            if (!action)
                Fail(endpoint_path, "missing endpoint action");
            const auto found = action_cores.find(*action);
            if (found == action_cores.end() || !(found->second == core))
                Fail(endpoint_path,
                     "endpoint action does not execute on the declared core");
        };
        auto runtime_definition_for_use = [&]
            (const RuntimeUse &target, RuntimeOperandFieldDto field)
            -> const RuntimeSymbolDefinitionDto * {
            const RuntimeSymbolDefinitionDto *result = nullptr;
            for (const RuntimeSymbolDefinitionDto &candidate :
                 manifest.runtime_symbol_definitions) {
                const auto found = runtime_uses.find(candidate.symbol.id);
                if (found == runtime_uses.end()) continue;
                for (const RuntimeUse &use : found->second) {
                    if (use.field == field &&
                        use.fragment_id == target.fragment_id &&
                        use.logical_core == target.logical_core &&
                        use.fragment_record_index ==
                            target.fragment_record_index) {
                        if (result)
                            Fail("runtime_symbol_definition",
                                 "one record field resolves to multiple runtime definitions");
                        result = &candidate;
                    }
                }
            }
            return result;
        };
        for (const RuntimeSymbolDefinitionDto &definition :
             manifest.runtime_symbol_definitions) {
            const std::vector<RuntimeUse> &uses =
                runtime_uses[definition.symbol.id];
            if (definition.symbol.kind == RuntimeSymbolKindDto::START_TAG) {
                const auto first = first_action_by_core.find(
                    definition.logical_cores[0]);
                if (!uses.empty() || first == first_action_by_core.end() ||
                    definition.symbol.source_ref != first->second)
                    Fail("runtime_symbol_definition",
                         "START_TAG must source the first linked action on its core");
            } else if (definition.symbol.kind ==
                       RuntimeSymbolKindDto::RUNTIME_CORE) {
                if (definition.logical_cores.size() != 1 ||
                    definition.source_action_id ||
                    definition.destination_action_id || uses.empty())
                    Fail("runtime_symbol_definition",
                         "RUNTIME_CORE definition/use closure mismatch");
            } else if (definition.symbol.kind == RuntimeSymbolKindDto::DTE_FSM) {
                if (definition.logical_cores.size() != 2 || uses.size() != 2)
                    Fail("runtime_symbol_definition",
                         "DTE_FSM requires two cores and exactly SEND/RECV uses");
                const RuntimeUse *send = nullptr;
                const RuntimeUse *recv = nullptr;
                std::set<Opcode> opcodes;
                for (const RuntimeUse &use : uses) {
                    if (use.field != RuntimeOperandFieldDto::DTE_FSM)
                        Fail("runtime_symbol_definition",
                             "DTE_FSM used in the wrong operand field");
                    opcodes.insert(use.opcode);
                    if (use.opcode == Opcode::DTE_SEND) send = &use;
                    if (use.opcode == Opcode::DTE_RECV) recv = &use;
                }
                if (opcodes != std::set<Opcode>{Opcode::DTE_SEND,
                                                Opcode::DTE_RECV})
                    Fail("runtime_symbol_definition",
                         "DTE_FSM uses must be one SEND and one RECV");
                if (!send || !recv ||
                    definition.source_action_id !=
                        std::optional<std::string>(send->action_id) ||
                    definition.destination_action_id !=
                        std::optional<std::string>(recv->action_id))
                    Fail("runtime_symbol_definition",
                         "DTE_FSM endpoint actions do not exactly match SEND/RECV");
                require_action_core(definition.source_action_id,
                                    send->logical_core,
                                    "runtime_symbol_definition.source_action_id");
                require_action_core(definition.destination_action_id,
                                    recv->logical_core,
                                    "runtime_symbol_definition.destination_action_id");
                std::vector<LogicalCoreDto> endpoint_cores{
                    send->logical_core, recv->logical_core};
                std::sort(endpoint_cores.begin(), endpoint_cores.end());
                endpoint_cores.erase(
                    std::unique(endpoint_cores.begin(), endpoint_cores.end()),
                    endpoint_cores.end());
                if (definition.logical_cores != endpoint_cores)
                    Fail("runtime_symbol_definition.logical_cores",
                         "DTE_FSM cores do not exactly match endpoints");
            } else if (definition.symbol.kind ==
                       RuntimeSymbolKindDto::EVENT_TAG) {
                if (definition.logical_cores.size() != 2 || uses.size() != 2)
                    Fail("runtime_symbol_definition",
                         "EVENT_TAG requires two cores and SET/WAIT uses");
                const RuntimeUse *set = nullptr;
                const RuntimeUse *wait = nullptr;
                for (const RuntimeUse &use : uses) {
                    if (use.field != RuntimeOperandFieldDto::EVENT_TAG)
                        Fail("runtime_symbol_definition",
                             "EVENT_TAG used in the wrong operand field");
                    if (use.opcode == Opcode::EVENT_SET) set = &use;
                    if (use.opcode == Opcode::EVENT_WAIT) wait = &use;
                }
                if (!set || !wait)
                    Fail("runtime_symbol_definition",
                         "EVENT_TAG uses must be one SET and one WAIT");
                if (definition.source_action_id !=
                        std::optional<std::string>(set->action_id) ||
                    definition.destination_action_id !=
                        std::optional<std::string>(wait->action_id))
                    Fail("runtime_symbol_definition",
                         "EVENT_TAG endpoint actions do not match SET/WAIT");
                require_action_core(definition.source_action_id,
                                    set->logical_core,
                                    "runtime_symbol_definition.source_action_id");
                require_action_core(definition.destination_action_id,
                                    wait->logical_core,
                                    "runtime_symbol_definition.destination_action_id");
                std::vector<LogicalCoreDto> endpoint_cores{
                    set->logical_core, wait->logical_core};
                std::sort(endpoint_cores.begin(), endpoint_cores.end());
                endpoint_cores.erase(
                    std::unique(endpoint_cores.begin(), endpoint_cores.end()),
                    endpoint_cores.end());
                if (definition.logical_cores != endpoint_cores)
                    Fail("runtime_symbol_definition.logical_cores",
                         "EVENT_TAG cores do not exactly match endpoints");

                const auto set_opcodes = action_opcodes.find(set->action_id);
                const auto wait_opcodes = action_opcodes.find(wait->action_id);
                const bool wave_event =
                    set_opcodes != action_opcodes.end() &&
                    wait_opcodes != action_opcodes.end() &&
                    set_opcodes->second.count(Opcode::DTE_WAIT) == 1 &&
                    wait_opcodes->second.count(Opcode::DTE_SEND) == 1;
                if (wave_event) {
                    const auto wait_stream = fragment_streams.find(
                        std::make_pair(wait->fragment_id,
                                       wait->logical_core));
                    if (wait_stream == fragment_streams.end() ||
                        wait->fragment_record_index >=
                            wait_stream->second->records.size())
                        Fail("runtime_symbol_definition",
                             "wave EVENT_WAIT record is dangling");
                    const RelocatableRecordDto &wait_record =
                        wait_stream->second->records[
                            wait->fragment_record_index];
                    if (wait_record.opcode != Opcode::EVENT_WAIT ||
                        wait_record.operands.size() != 4 ||
                        LiteralU64(wait_record.operands[3],
                                   "runtime_symbol_definition.wave_count") != 1)
                        Fail("runtime_symbol_definition",
                             "state-transfer wave EVENT_WAIT count must equal one");

                    constexpr std::string_view kWaveSchema =
                        "wafer_frontend.state_transfer_wave_runtime_symbol/v1";
                    const Json binding_key{
                        {"source_global_dag_id",
                         manifest.source_global_dag_id},
                        {"source_action_id", set->action_id},
                        {"destination_action_id", wait->action_id},
                        {"capacity", 3}};
                    const std::string binding = StableArtifactId(
                        "state_transfer_wave_binding", kWaveSchema,
                        binding_key);
                    const std::string expected_event = StableArtifactId(
                        "state_transfer_wave_event", kWaveSchema,
                        Json{{"binding_ref", binding},
                             {"source_action_id", set->action_id},
                             {"destination_action_id", wait->action_id}});
                    if (definition.symbol.source_ref != binding ||
                        definition.symbol.id != expected_event)
                        Fail("runtime_symbol_definition",
                             "state-transfer wave EVENT_TAG stable identity mismatch");

                    const RuntimeSymbolDefinitionDto *set_source =
                        runtime_definition_for_use(
                            *set, RuntimeOperandFieldDto::SOURCE_CORE);
                    const RuntimeSymbolDefinitionDto *set_destination =
                        runtime_definition_for_use(
                            *set, RuntimeOperandFieldDto::DESTINATION_CORE);
                    const RuntimeSymbolDefinitionDto *wait_source =
                        runtime_definition_for_use(
                            *wait, RuntimeOperandFieldDto::SOURCE_CORE);
                    const RuntimeSymbolDefinitionDto *wait_destination =
                        runtime_definition_for_use(
                            *wait, RuntimeOperandFieldDto::DESTINATION_CORE);
                    if (!set_source || !set_destination || !wait_source ||
                        !wait_destination)
                        Fail("runtime_symbol_definition",
                             "state-transfer wave event lacks exact core definitions");
                    auto require_wave_core = [&]
                        (const RuntimeSymbolDefinitionDto &candidate,
                         std::string_view role,
                         const LogicalCoreDto &core) {
                        const std::string expected = StableArtifactId(
                            "state_transfer_wave_core", kWaveSchema,
                            Json{{"binding_ref", binding},
                                 {"role", role},
                                 {"logical_core",
                                  Json{{"die_id", core.die_id},
                                       {"local_core_id",
                                        core.local_core_id}}}});
                        if (candidate.symbol.kind !=
                                RuntimeSymbolKindDto::RUNTIME_CORE ||
                            candidate.symbol.source_ref != binding ||
                            candidate.symbol.id != expected ||
                            candidate.logical_cores !=
                                std::vector<LogicalCoreDto>{core})
                            Fail("runtime_symbol_definition",
                                 "state-transfer wave core stable identity mismatch");
                    };
                    require_wave_core(*set_source, "source",
                                      set->logical_core);
                    require_wave_core(*set_destination, "destination",
                                      wait->logical_core);
                    if (set_source->symbol.id != wait_source->symbol.id ||
                        set_destination->symbol.id !=
                            wait_destination->symbol.id)
                        Fail("runtime_symbol_definition",
                             "state-transfer wave SET/WAIT core symbols differ");
                }
            } else if (definition.symbol.kind ==
                       RuntimeSymbolKindDto::DTE_TOKEN) {
                if (definition.logical_cores.size() != 1 || uses.size() != 2 ||
                    !definition.source_action_id)
                    Fail("runtime_symbol_definition",
                         "DTE_TOKEN requires one core and two endpoint uses");
                for (const RuntimeUse &use : uses) {
                    if (use.field != RuntimeOperandFieldDto::DTE_TOKEN ||
                        !(use.logical_core == definition.logical_cores[0]))
                        Fail("runtime_symbol_definition",
                             "DTE_TOKEN field/core closure mismatch");
                }
                require_action_core(definition.source_action_id,
                                    definition.logical_cores[0],
                                    "runtime_symbol_definition.source_action_id");
                if (definition.destination_action_id)
                    require_action_core(definition.destination_action_id,
                                        definition.logical_cores[0],
                                        "runtime_symbol_definition.destination_action_id");
                const std::set<Opcode> opcodes{uses[0].opcode,
                                               uses[1].opcode};
                const bool local = opcodes ==
                    std::set<Opcode>{Opcode::DTE_ISSUE, Opcode::DTE_WAIT};
                const bool remote = opcodes ==
                    std::set<Opcode>{Opcode::DTE_RECV, Opcode::DTE_WAIT};
                if (!local && !remote)
                    Fail("runtime_symbol_definition",
                         "DTE_TOKEN uses must be ISSUE/WAIT or RECV/WAIT");
                const bool rooted_local =
                    local && rooted_ar_link &&
                    uses[0].fragment_id == uses[1].fragment_id &&
                    fragments.at(uses[0].fragment_id)->kind ==
                        FragmentKindDto::S2_LITE_ROOTED_AR;
                if (local && !rooted_local &&
                    definition.destination_action_id)
                    Fail("runtime_symbol_definition",
                         "local ISSUE/WAIT token must not name a destination action");
                if (remote && !definition.destination_action_id)
                    Fail("runtime_symbol_definition",
                         "remote RECV/WAIT token requires a destination action");
                const RuntimeUse *producer = nullptr;
                const RuntimeUse *wait = nullptr;
                for (const RuntimeUse &use : uses) {
                    if (use.opcode == Opcode::DTE_ISSUE ||
                        use.opcode == Opcode::DTE_RECV)
                        producer = &use;
                    if (use.opcode == Opcode::DTE_WAIT)
                        wait = &use;
                }
                if (!producer || !wait ||
                    definition.source_action_id !=
                        std::optional<std::string>(producer->action_id) ||
                    (remote && definition.destination_action_id !=
                        std::optional<std::string>(wait->action_id)) ||
                    (rooted_local && definition.destination_action_id !=
                        std::optional<std::string>(wait->action_id)) ||
                    (local && producer->action_id != wait->action_id))
                    Fail("runtime_symbol_definition",
                         "DTE_TOKEN endpoint actions do not match producer/WAIT");
            }
        }
        using RuntimeRecordKey =
            std::tuple<std::string, LogicalCoreDto, uint64_t>;
        std::map<RuntimeRecordKey, const RuntimeSymbolDefinitionDto *>
            fsm_by_record;
        std::map<RuntimeRecordKey, const RuntimeSymbolDefinitionDto *>
            event_by_record;
        for (const RuntimeSymbolDefinitionDto &definition :
             manifest.runtime_symbol_definitions) {
            auto index_uses = [&](
                std::map<RuntimeRecordKey,
                         const RuntimeSymbolDefinitionDto *> &index) {
                for (const RuntimeUse &use :
                     runtime_uses[definition.symbol.id]) {
                    const RuntimeRecordKey key{use.fragment_id,
                                               use.logical_core,
                                               use.fragment_record_index};
                    if (!index.emplace(key, &definition).second)
                        Fail("runtime_symbol_definition",
                             "one record has multiple endpoint symbols");
                }
            };
            if (definition.symbol.kind == RuntimeSymbolKindDto::DTE_FSM) {
                index_uses(fsm_by_record);
            } else if (definition.symbol.kind ==
                       RuntimeSymbolKindDto::EVENT_TAG) {
                index_uses(event_by_record);
            }
        }
        for (const RuntimeSymbolDefinitionDto &definition :
             manifest.runtime_symbol_definitions) {
            if (definition.symbol.kind != RuntimeSymbolKindDto::RUNTIME_CORE)
                continue;
            const LogicalCoreDto represented = definition.logical_cores[0];
            for (const RuntimeUse &use : runtime_uses[definition.symbol.id]) {
                LogicalCoreDto expected;
                const RuntimeRecordKey key{use.fragment_id, use.logical_core,
                                           use.fragment_record_index};
                if (use.field == RuntimeOperandFieldDto::PEER_CORE) {
                    const auto endpoint = fsm_by_record.find(key);
                    if (endpoint == fsm_by_record.end())
                        Fail("runtime_symbol_definition",
                             "PEER_CORE use has no matching DTE_FSM endpoint");
                    const RuntimeSymbolDefinitionDto &fsm = *endpoint->second;
                    const std::optional<std::string> peer_action =
                        fsm.source_action_id ==
                                std::optional<std::string>(use.action_id)
                            ? fsm.destination_action_id
                            : fsm.source_action_id;
                    if (!peer_action || action_cores.count(*peer_action) == 0)
                        Fail("runtime_symbol_definition",
                             "PEER_CORE cannot resolve its opposite endpoint");
                    expected = action_cores.at(*peer_action);
                } else if (use.field == RuntimeOperandFieldDto::SOURCE_CORE ||
                           use.field == RuntimeOperandFieldDto::DESTINATION_CORE) {
                    const auto endpoint = event_by_record.find(key);
                    if (endpoint == event_by_record.end())
                        Fail("runtime_symbol_definition",
                             "EVENT core use has no matching EVENT_TAG endpoint");
                    const RuntimeSymbolDefinitionDto &event = *endpoint->second;
                    const std::optional<std::string> endpoint_action =
                        use.field == RuntimeOperandFieldDto::SOURCE_CORE
                            ? event.source_action_id
                            : event.destination_action_id;
                    if (!endpoint_action ||
                        action_cores.count(*endpoint_action) == 0)
                        Fail("runtime_symbol_definition",
                             "EVENT core cannot resolve its endpoint action");
                    expected = action_cores.at(*endpoint_action);
                } else {
                    Fail("runtime_symbol_definition",
                         "RUNTIME_CORE used outside peer/event core fields");
                }
                if (!(represented == expected))
                    Fail("runtime_symbol_definition",
                         "runtime core symbol names the wrong endpoint core");
            }
        }

        const ProgramControlEnvelopeDto &source_envelope = manifest.envelope;
        RequireCanonical(source_envelope.active_cores, "envelope.active_cores");
        RequireCanonical(source_envelope.terminal_cores, "envelope.terminal_cores");
        RequireCanonical(source_envelope.expected_ack_cores, "envelope.expected_ack_cores");
        RequireCanonical(source_envelope.expected_done_cores, "envelope.expected_done_cores");
        std::vector<LogicalCoreDto> bound_cores;
        for (const CoreRuntimeBindingDto &binding : manifest.core_bindings)
            bound_cores.push_back(binding.logical_core);
        if (source_envelope.active_cores != bound_cores ||
            source_envelope.expected_done_cores !=
                source_envelope.terminal_cores)
            Fail("linked_program_manifest.envelope",
                 "active/done sets do not close over bindings/terminal cores");
        artifact.envelope.active_cores =
            RemapCores(source_envelope.active_cores, runtime_ids,
                       "envelope.active_cores");
        artifact.envelope.terminal_cores =
            RemapCores(source_envelope.terminal_cores, runtime_ids,
                       "envelope.terminal_cores");
        artifact.envelope.expected_ack_cores =
            RemapCores(source_envelope.expected_ack_cores, runtime_ids,
                       "envelope.expected_ack_cores");
        artifact.envelope.expected_done_cores =
            RemapCores(source_envelope.expected_done_cores, runtime_ids,
                       "envelope.expected_done_cores");
        artifact.envelope.empty_core_ack_policy =
            source_envelope.empty_core_ack_policy;
        artifact.envelope.failure_policy = source_envelope.failure_policy;
        std::tuple<LogicalCoreDto, std::string> previous_start;
        bool first_start = true;
        std::set<std::string> used_start_tags;
        for (const LogicalStartEventDto &event : source_envelope.start_events) {
            const auto key = std::make_tuple(event.target_core,
                                             event.tag_symbol_ref);
            if (!first_start && !(previous_start < key))
                Fail("envelope.start_events", "must be unique and canonical");
            first_start = false;
            previous_start = key;
            const auto tag = start_tags.find(event.tag_symbol_ref);
            const auto core = runtime_ids.find(event.target_core);
            const auto definition = std::find_if(
                manifest.runtime_symbol_definitions.begin(),
                manifest.runtime_symbol_definitions.end(),
                [&](const RuntimeSymbolDefinitionDto &item) {
                    return item.symbol.id == event.tag_symbol_ref;
                });
            if (tag == start_tags.end() || core == runtime_ids.end() ||
                definition == manifest.runtime_symbol_definitions.end() ||
                definition->logical_cores !=
                    std::vector<LogicalCoreDto>{event.target_core})
                Fail("envelope.start_events",
                     "event does not resolve to an exact core-local START_TAG");
            used_start_tags.insert(event.tag_symbol_ref);
            artifact.envelope.start_events.push_back(
                {core->second, tag->second, event.count});
        }
        if (used_start_tags.size() != start_tags.size())
            Fail("linked_program_manifest.runtime_symbol_definitions",
                 "contains an unreferenced top-level START_TAG");
        std::sort(artifact.envelope.start_events.begin(),
                  artifact.envelope.start_events.end(),
                  [](const ProgramStartEvent &left,
                     const ProgramStartEvent &right) {
                      return std::tie(left.target_core, left.tag) <
                             std::tie(right.target_core, right.tag);
                  });

        std::sort(artifact.relocations.begin(), artifact.relocations.end(),
                  [](const SemanticRelocation &left,
                     const SemanticRelocation &right) {
                      return std::tie(left.core_index, left.instruction_index,
                                      left.operand_id) <
                             std::tie(right.core_index, right.instruction_index,
                                      right.operand_id);
                  });
        ValidateProgramArtifact(artifact);
        return artifact;
    } catch (const ProgramFinalizerError &) {
        throw;
    } catch (const std::exception &error) {
        throw ProgramFinalizerError(
            std::string("linked_program_manifest: finalization failed: ") +
            error.what());
    }
}

ProgramArtifact ProgramArtifactFinalizer::FinalizeJson(
    std::string_view manifest_json) const {
    return Finalize(Parse(manifest_json));
}

std::vector<uint8_t> ProgramArtifactFinalizer::FinalizeEncoded(
    std::string_view manifest_json) const {
    try {
        return EncodeProgramArtifact(FinalizeJson(manifest_json));
    } catch (const ProgramFinalizerError &) {
        throw;
    } catch (const std::exception &error) {
        throw ProgramFinalizerError(
            std::string("linked_program_manifest: encoding failed: ") +
            error.what());
    }
}

} // namespace frontend
