#include "isa/record_codec_selftest.h"

#include "isa/opcode.h"
#include "isa/record_codec.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

enum class Boundary { MINIMUM, TYPICAL, MAXIMUM };

class Checks {
public:
    void Check(bool condition, const std::string &name) {
        ++result.checks;
        if (!condition)
            result.failures.push_back(name);
    }

    template <class Function>
    void Reject(const std::string &name, Function function) {
        ++result.checks;
        try {
            function();
            result.failures.push_back(name + ": unexpectedly accepted");
        } catch (const RecordCodecError &) {
        } catch (const std::exception &error) {
            result.failures.push_back(name + ": wrong exception: " +
                                      error.what());
        }
    }

    template <class Function>
    void Accept(const std::string &name, Function function) {
        ++result.checks;
        try {
            function();
        } catch (const std::exception &error) {
            result.failures.push_back(name + ": " + error.what());
        }
    }

    RecordCodecSelfTestResult result;
};

uint64_t Select(Boundary boundary, uint64_t minimum, uint64_t typical,
                uint64_t maximum) {
    if (boundary == Boundary::MINIMUM)
        return minimum;
    if (boundary == Boundary::TYPICAL)
        return typical;
    return maximum;
}

SramAddressOperand Address(Boundary boundary) {
    if (boundary == Boundary::TYPICAL)
        return {SramAddressKind::REGION, 0, 17, 19};
    return {SramAddressKind::ABSOLUTE,
            boundary == Boundary::MAXIMUM
                ? std::numeric_limits<uint64_t>::max()
                : 0,
            0, 0};
}

uint64_t DoubleBits(double value) {
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value),
                  "binary64 fixture storage must be eight bytes");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool HasPublishedComputeSemantics(Opcode opcode) {
    switch (opcode) {
    case Opcode::MATMUL:
    case Opcode::CONV:
    case Opcode::MAXPOOL:
    case Opcode::ATTENTION:
    case Opcode::GATE:
    case Opcode::MOE_MATMUL:
    case Opcode::GELU:
    case Opcode::SILU:
    case Opcode::SWIGLU:
    case Opcode::RELU:
    case Opcode::RESIDUAL:
    case Opcode::LAYERNORM:
    case Opcode::RMSNORM:
    case Opcode::ROPE:
    case Opcode::SPLIT_MATMUL:
    case Opcode::MERGE_MATMUL:
    case Opcode::DUMMY:
        return true;
    default:
        return false;
    }
}

uint64_t ComputeFixtureParameter(Opcode opcode, std::string_view name,
                                 Boundary boundary) {
    if (!HasPublishedComputeSemantics(opcode))
        return Select(boundary, 0, 17, kExternalNpuParameterMax);
    const bool zero_allowed = name == "pX" || name == "pY" ||
                              name == "is_merge" || name == "need_choose";
    if (boundary == Boundary::MINIMUM) {
        if (zero_allowed) return 0;
        if (opcode == Opcode::ATTENTION && name == "R") return 3;
        return 1;
    }
    if (name == "is_merge" || name == "need_choose") return 1;
    if (name == "dim") return 1;
    if (name == "slice") return kSramBindInputLimit;
    return 17;
}

void SetComputeParameter(ExternalRecord &record, std::string_view name,
                         uint64_t value) {
    ComputeOperands &operands = std::get<ComputeOperands>(record.operands);
    const RecordSchema &schema = *LookupRecordSchema(record.opcode);
    for (std::size_t i = 0; i < schema.parameter_count; ++i) {
        if (schema.parameter_names[i] == name) {
            operands.parameters[i] = value;
            return;
        }
    }
    throw std::logic_error("selftest compute schema is missing parameter");
}

ExternalRecord MakeRecord(const RecordSchema &schema, Boundary boundary) {
    ExternalRecord record;
    record.opcode = schema.opcode;
    const uint64_t u16 = Select(boundary, 0, 17, UINT16_MAX);
    const uint64_t u32 = Select(boundary, 0, 17, UINT32_MAX);
    const uint64_t nonzero_u32 = Select(boundary, 1, 17, UINT32_MAX);

    switch (schema.operand_kind) {
    case RecordOperandKind::COMPUTE: {
        ComputeOperands operands;
        operands.datatype = boundary == Boundary::MINIMUM
                                ? ExternalDataType::INT8
                                : ExternalDataType::FP16;
        operands.input_offset_bytes = u16;
        operands.data_offset_bytes = u16;
        operands.output_offset_bytes = u16;
        operands.parameters.reserve(schema.parameter_count);
        for (std::size_t i = 0; i < schema.parameter_count; ++i)
            operands.parameters.push_back(ComputeFixtureParameter(
                schema.opcode, schema.parameter_names[i], boundary));
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::ROPE_QK_EXACT: {
        RopeQkExactOperands operands;
        operands.input = Address(boundary);
        operands.output = Address(boundary);
        operands.logical_tokens = 8;
        operands.tp_degree = 2;
        operands.num_heads = 4;
        operands.num_kv_heads = 4;
        operands.rank_num_heads = 2;
        operands.rank_num_kv_heads = 2;
        operands.head_dim = 4;
        operands.rotary_dim = 4;
        operands.max_position_embeddings = 128;
        operands.context_max = 8;
        operands.rope_theta_f64_bits = DoubleBits(10000.0);
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::ATTENTION_EXACT: {
        AttentionExactOperands operands;
        operands.input = Address(boundary);
        operands.output = Address(boundary);
        operands.query_tokens = 8;
        operands.tp_degree = 2;
        operands.num_heads = 4;
        operands.num_kv_heads = 4;
        operands.rank_num_heads = 2;
        operands.rank_num_kv_heads = 2;
        operands.head_dim = 4;
        operands.context_sum = 8;
        operands.context_max = 8;
        operands.query_key_pairs = 36;
        operands.rank_kv_read_bytes = 0;
        operands.rank_kv_write_bytes = 256;
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::EMBEDDING_LOOKUP: {
        EmbeddingLookupOperands operands;
        operands.indices = Address(boundary);
        operands.table = Address(boundary);
        operands.output = Address(boundary);
        operands.logical_rows = 8;
        operands.rank_rows = 4;
        operands.tp_degree = 2;
        operands.vocab_size = 32;
        operands.hidden_size = 16;
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::GREEDY_SAMPLE: {
        GreedySampleOperands operands;
        operands.logits = Address(boundary);
        operands.output = Address(boundary);
        operands.tp_degree = 1;
        operands.token_rows = 8;
        operands.vocab_size = 32;
        operands.sample_count = 1;
        operands.comparisons = 31;
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::CROSS_ENTROPY_FORWARD: {
        CrossEntropyForwardOperands operands;
        operands.logits = Address(boundary);
        operands.labels = Address(boundary);
        operands.loss = Address(boundary);
        operands.logical_rows = 8;
        operands.rank_rows = 4;
        operands.tp_degree = 2;
        operands.vocab_size = 32;
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::CROSS_ENTROPY_BACKWARD: {
        CrossEntropyBackwardOperands operands;
        operands.logits = Address(boundary);
        operands.labels = Address(boundary);
        operands.upstream = Address(boundary);
        operands.logits_grad = Address(boundary);
        operands.logical_rows = 8;
        operands.rank_rows = 4;
        operands.tp_degree = 2;
        operands.vocab_size = 32;
        operands.upstream_elements = 4;
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::SGD_UPDATE: {
        SgdUpdateOperands operands;
        operands.weight = Address(boundary);
        operands.gradient = Address(boundary);
        operands.updated_weight = operands.weight;
        operands.element_count = 8;
        operands.learning_rate_f64_bits = DoubleBits(0.01);
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::DTE_SEND: {
        DteSendOperands operands;
        operands.source = Address(boundary);
        operands.fsm_id = nonzero_u32;
        operands.length_bytes =
            Select(boundary, 1, 257, kDteEndpointP2pMaxBytes);
        if (boundary == Boundary::MAXIMUM)
            operands.source_space = EndpointSourceSpace::HBM;
        if (boundary == Boundary::MINIMUM) {
            operands.mode = DteSendMode::P2P;
            operands.peer_core = 0;
        } else if (boundary == Boundary::TYPICAL) {
            operands.mode = DteSendMode::SCATTER;
            operands.completion = EndpointCompletion::SYNC;
            operands.token = 0;
            operands.group_id = 7;
            operands.collective_id = 11;
            operands.epoch = 13;
        } else {
            operands.mode = DteSendMode::BROADCAST;
            operands.token = UINT32_MAX;
            operands.tree_id = 0;
            operands.group_id = UINT32_MAX;
            operands.collective_id = UINT32_MAX - 1;
            operands.epoch = UINT32_MAX;
        }
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::DTE_RECV: {
        DteRecvOperands operands;
        operands.destination = Address(boundary);
        operands.fsm_id = nonzero_u32;
        operands.length_bytes =
            Select(boundary, 1, 257, kDteEndpointP2pMaxBytes);
        if (boundary == Boundary::MINIMUM) {
            operands.mode = DteRecvMode::P2P;
        } else if (boundary == Boundary::TYPICAL) {
            operands.mode = DteRecvMode::GATHER;
            operands.completion = EndpointCompletion::SYNC;
            operands.token = 0;
            operands.expected_sources = 3;
            operands.group_id = 7;
            operands.collective_id = 11;
            operands.epoch = 13;
        } else {
            operands.mode = DteRecvMode::REDUCE;
            operands.token = UINT32_MAX;
            operands.datatype = EndpointDataType::INT64;
            operands.reduce_op = ReduceOperator::MAX;
            operands.expected_sources = UINT16_MAX;
            operands.group_id = UINT32_MAX;
            operands.collective_id = UINT32_MAX - 1;
            operands.epoch = UINT32_MAX;
        }
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::REDUCE_COMPUTE: {
        ReduceComputeOperands operands;
        operands.datatype = boundary == Boundary::MAXIMUM
                                ? EndpointDataType::INT64
                                : EndpointDataType::UINT8;
        operands.reduce_op = boundary == Boundary::MAXIMUM
                                 ? ReduceOperator::MAX
                                 : ReduceOperator::SUM;
        operands.group_id = Select(boundary, 1, 7, UINT32_MAX);
        operands.collective_id =
            Select(boundary, 0, 11, uint64_t{UINT32_MAX} - 1);
        operands.epoch = u32;
        operands.root_rank = u16;
        operands.self_rank = u16;
        operands.element_count = Select(
            boundary, 1, 257, std::numeric_limits<uint64_t>::max() / 64);
        operands.source = Address(boundary);
        operands.destination = Address(boundary);
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::LOCAL_REDUCE: {
        LocalReduceOperands operands;
        operands.input_count =
            Select(boundary, 1, 3, UINT16_MAX);
        operands.element_count =
            boundary == Boundary::MAXIMUM
                ? std::numeric_limits<uint64_t>::max() /
                      (2 * uint64_t{UINT16_MAX})
                : Select(boundary, 1, 17, 17);
        operands.input_stride_bytes = operands.element_count * 2;
        operands.source = {
            SramAddressKind::ABSOLUTE,
            boundary == Boundary::TYPICAL ? uint64_t{0x100} : uint64_t{0},
            0, 0};
        operands.destination = {
            SramAddressKind::ABSOLUTE,
            boundary == Boundary::TYPICAL ? uint64_t{0x200} : uint64_t{0},
            0, 0};
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::LOCAL_NOC_SEND: {
        LocalNocSendOperands operands;
        operands.source = Address(boundary);
        operands.destination_core = Select(boundary, 0, 7, UINT16_MAX);
        operands.byte_count = Select(
            boundary, 1, 257, std::numeric_limits<uint64_t>::max());
        operands.event_id = Select(boundary, 1, 11, UINT32_MAX);
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::LOCAL_NOC_RECV: {
        LocalNocRecvOperands operands;
        operands.destination = Address(boundary);
        operands.source_core = Select(boundary, 0, 7, UINT16_MAX);
        operands.byte_count = Select(
            boundary, 1, 257, std::numeric_limits<uint64_t>::max());
        operands.event_id = Select(boundary, 1, 11, UINT32_MAX);
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::LOCAL_NOC_WAIT: {
        LocalNocWaitOperands operands;
        operands.event_id = Select(boundary, 1, 11, UINT32_MAX);
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::LSU: {
        LsuOperands operands;
        operands.hbm_address_bytes =
            Select(boundary, 0, 23, std::numeric_limits<uint64_t>::max());
        operands.size_bytes = Select(
            boundary, 1, 257, std::numeric_limits<uint64_t>::max());
        operands.sram = Address(boundary);
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::DTE_ISSUE: {
        DteIssueOperands operands;
        operands.token = nonzero_u32;
        operands.size_bytes = Select(
            boundary, 1, 257, std::numeric_limits<uint64_t>::max() / 8);
        operands.payload_bits = operands.size_bytes * 8;
        if (boundary == Boundary::MINIMUM) {
            operands.direction = LocalDteDirection::SPM_TO_SPM;
            operands.source_sram = Address(boundary);
            operands.destination_sram = Address(boundary);
        } else if (boundary == Boundary::TYPICAL) {
            operands.direction = LocalDteDirection::SPM_TO_DRAM;
            operands.source_sram = Address(boundary);
            operands.hbm_address_bytes = 23;
        } else {
            operands.direction = LocalDteDirection::DRAM_TO_SPM;
            operands.destination_sram = Address(boundary);
            operands.hbm_address_bytes =
                std::numeric_limits<uint64_t>::max();
        }
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::SYMBOL:
        record.operands = SymbolOperands{u32};
        break;
    case RecordOperandKind::SRAM_BIND: {
        SramBindOperands operands;
        operands.input_count = Select(boundary, 1, 3, kSramBindInputLimit);
        for (std::size_t i = 0; i < operands.input_count; ++i)
            operands.input_symbol_indices[i] =
                boundary == Boundary::MAXIMUM ? UINT32_MAX : u32 + i;
        operands.output_symbol_index =
            boundary == Boundary::MAXIMUM ? UINT32_MAX : u32 + 3;
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::SRAM_ALLOC: {
        SramAllocOperands operands;
        operands.region_name_string_index = u32;
        operands.label_symbol_index = u32;
        operands.size_bytes = Select(
            boundary, 1, 257, std::numeric_limits<uint64_t>::max());
        operands.alignment_bytes = Select(boundary, 1, 16, uint64_t{1} << 63);
        operands.lifetime = boundary == Boundary::MINIMUM
                                ? SramLifetime::TASK
                                : boundary == Boundary::TYPICAL
                                      ? SramLifetime::LAYER
                                      : SramLifetime::PERSISTENT;
        operands.spillable = boundary != Boundary::MAXIMUM;
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::SRAM_ALLOC_AT: {
        SramAllocAtOperands operands;
        operands.region_name_string_index = u32;
        operands.label_symbol_index = u32;
        operands.region_offset_bytes = Select(
            boundary, 0, 257, std::numeric_limits<uint64_t>::max());
        operands.size_bytes = Select(
            boundary, 1, 257, std::numeric_limits<uint64_t>::max());
        operands.alignment_bytes = Select(
            boundary, 1, 16, uint64_t{1} << 63);
        operands.lifetime = boundary == Boundary::MINIMUM
                                ? SramLifetime::TASK
                                : boundary == Boundary::TYPICAL
                                      ? SramLifetime::LAYER
                                      : SramLifetime::PERSISTENT;
        operands.spillable = boundary != Boundary::MAXIMUM;
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::SRAM_RESIZE:
        record.operands = SramResizeOperands{
            u32, Select(boundary, 1, 257,
                        std::numeric_limits<uint64_t>::max())};
        break;
    case RecordOperandKind::SRAM_RENAME:
        record.operands = SramRenameOperands{
            Select(boundary, 0, 17, uint64_t{UINT32_MAX} - 1),
            Select(boundary, 1, 18, UINT32_MAX)};
        break;
    case RecordOperandKind::TOKEN:
        record.operands = TokenOperands{nonzero_u32};
        break;
    case RecordOperandKind::NONE:
        record.operands = NoOperands{};
        break;
    case RecordOperandKind::EVENT_SET:
        record.operands = EventSetOperands{
            u16, Select(boundary, 0, 18, UINT16_MAX), u32};
        break;
    case RecordOperandKind::EVENT_WAIT:
        record.operands = EventWaitOperands{
            u16, Select(boundary, 0, 18, UINT16_MAX), u32, nonzero_u32};
        break;
    case RecordOperandKind::GROUP_SYNC:
        record.operands = GroupSyncOperands{
            Select(boundary, 1, 17, UINT32_MAX), u32};
        break;
    }
    return record;
}

uint64_t CapabilitiesFor(const OpcodeManifestEntry &entry) {
    return entry.required_capabilities;
}

std::string Name(const OpcodeManifestEntry &entry, Boundary boundary) {
    const char *suffix = boundary == Boundary::MINIMUM
                             ? " min"
                             : boundary == Boundary::TYPICAL ? " typical"
                                                              : " max";
    return std::string(entry.canonical_name) + suffix;
}

void WriteU32(std::vector<uint8_t> &bytes, std::size_t offset,
              uint32_t value) {
    for (std::size_t i = 0; i < 4; ++i)
        bytes[offset + i] = static_cast<uint8_t>(value >> (8 * i));
}

void CheckManifest(Checks &checks) {
    const auto &schemas = RecordSchemaManifest();
    const auto &manifest = OpcodeManifest();
    checks.Check(schemas.size() == manifest.size(), "schema count");
    for (std::size_t i = 0; i < manifest.size(); ++i) {
        const auto &schema = schemas[i];
        const auto &entry = manifest[i];
        checks.Check(schema.opcode == entry.opcode,
                     std::string(entry.canonical_name) + " schema order");
        checks.Check(LookupRecordSchema(OpcodeValue(entry.opcode)) == &schema,
                     std::string(entry.canonical_name) + " schema lookup");
        if (schema.operand_kind == RecordOperandKind::COMPUTE) {
            checks.Check(schema.payload_size == 8 + 4 * schema.parameter_count,
                         std::string(entry.canonical_name) +
                             " compute schema size");
            checks.Check(schema.parameter_count == 0 ||
                             schema.parameter_names != nullptr,
                         std::string(entry.canonical_name) +
                             " parameter names");
        }
        if (schema.operand_kind == RecordOperandKind::SRAM_BIND)
            checks.Check(schema.payload_size == 72,
                         "SRAM_BIND fixed payload size");
    }
    checks.Check(LookupRecordSchema(0) == nullptr, "INVALID has no schema");
    checks.Check(LookupRecordSchema(0x47) == nullptr,
                 "unassigned opcode has no schema");
}

void CheckBoundariesAndStream(Checks &checks) {
    std::vector<uint8_t> stream;
    std::size_t executable_count = 0;
    for (const auto &entry : OpcodeManifest()) {
        const RecordSchema &schema = *LookupRecordSchema(entry.opcode);
        const bool executable = entry.lifecycle == OpcodeLifecycle::STABLE &&
                                entry.support != OpcodeSupport::UNSUPPORTED;
        if (!executable) {
            checks.Reject(std::string(entry.canonical_name) + " reserved encode",
                          [&] {
                              EncodeExternalRecord(MakeRecord(
                                  schema, Boundary::TYPICAL),
                                  CapabilitiesFor(entry));
                          });
            continue;
        }
        ++executable_count;
        for (Boundary boundary : {Boundary::MINIMUM, Boundary::TYPICAL,
                                  Boundary::MAXIMUM}) {
            const ExternalRecord record = MakeRecord(schema, boundary);
            const uint64_t caps = CapabilitiesFor(entry);
            const std::string name = Name(entry, boundary);
            checks.Accept(name + " encode/decode", [&] {
                const std::vector<uint8_t> encoded =
                    EncodeExternalRecord(record, caps);
                checks.Check(encoded.size() ==
                                 kExternalRecordHeaderSize + schema.payload_size,
                             name + " record size");
                checks.Check(encoded[0] == OpcodeValue(entry.opcode) &&
                                 encoded[1] == kExternalRecordVersion &&
                                 encoded[2] == 0 && encoded[3] == 0,
                             name + " fixed header");
                const ExternalRecord decoded =
                    DecodeExternalRecordExact(encoded, caps);
                checks.Check(decoded.opcode == entry.opcode,
                             name + " decoded opcode");
                checks.Check(EncodeExternalRecord(decoded, caps) == encoded,
                             name + " canonical roundtrip");
            });
        }
        if (entry.support == OpcodeSupport::EXPERIMENTAL) {
            checks.Reject(std::string(entry.canonical_name) +
                              " capability encode gate",
                          [&] {
                              EncodeExternalRecord(
                                  MakeRecord(schema, Boundary::TYPICAL));
                          });
            const auto gated = EncodeExternalRecord(
                MakeRecord(schema, Boundary::TYPICAL), CapabilitiesFor(entry));
            checks.Reject(std::string(entry.canonical_name) +
                              " capability decode gate",
                          [&] { DecodeExternalRecordExact(gated); });
        }
        const auto typical = EncodeExternalRecord(
            MakeRecord(schema, Boundary::TYPICAL), CapabilitiesFor(entry));
        stream.insert(stream.end(), typical.begin(), typical.end());
    }
    checks.Check(executable_count == 51, "executable opcode count");
    const uint64_t all_caps = CapabilityBit(IsaCapability::PD_CONTEXT) |
                              CapabilityBit(IsaCapability::EXPERIMENTAL_FUSED);
    checks.Accept("record stream decode", [&] {
        const auto decoded = DecodeExternalRecordStream(stream, all_caps);
        checks.Check(decoded.size() == executable_count, "stream record count");
        checks.Check(EncodeExternalRecord(decoded.front(), all_caps).size() > 8,
                     "stream first record");
    });
    checks.Check(DecodeExternalRecordStream({}, all_caps).empty(),
                 "empty stream");
}

void CheckGoldenHex(Checks &checks) {
    struct Golden {
        Opcode opcode;
        const char *hex;
    };
    // Full-record fixtures cover every operand shape.  The all-opcode loop
    // above separately fixes each opcode byte and payload length.
    static constexpr std::array<Golden, 24> golden{{
        {Opcode::MATMUL,
         "0101000018000000010011001100110011000000110000001100000011000000"},
        {Opcode::DUMMY, "15010000080000000100110011001100"},
        {Opcode::ROPE_QK_EXACT,
         "1a010000640000000100000002000000110000000000000000000000130000000000000002000000110000000000000000000000130000000000000008000000020000000400000004000000020000000200000004000000040000008000000008000000000000000088c340"},
        {Opcode::ATTENTION_EXACT,
         "1b010000740000000100000102000000110000000000000000000000130000000000000002000000110000000000000000000000130000000000000008000000020000000400000004000000020000000200000004000000080000000000000008000000240000000000000000000000000000000001000000000000"},
        {Opcode::EMBEDDING_LOOKUP,
         "1c01000060000000020101000200000011000000000000000000000013000000000000000200000011000000000000000000000013000000000000000200000011000000000000000000000013000000000000000800000004000000020000002000000010000000"},
        {Opcode::GREEDY_SAMPLE,
         "1d0100004c00000001020000020000001100000000000000000000001300000000000000020000001100000000000000000000001300000000000000010000000800000020000000010000001f00000000000000"},
        {Opcode::CROSS_ENTROPY_FORWARD,
         "1e0100005c0000000102030002000000110000000000000000000000130000000000000002000000110000000000000000000000130000000000000002000000110000000000000000000000130000000000000008000000040000000200000020000000"},
        {Opcode::DTE_SEND,
         "40010000480000000101000011000000000000000000000001010000000000000200000011000000000000000000000013000000000000000000000000000000070000000b0000000d00000000000000"},
        {Opcode::DTE_RECV,
         "41010000480000000101000011000000000000000000000001010000000000000200000011000000000000000000000013000000000000000000030000000000070000000b0000000d00000000000000"},
        {Opcode::REDUCE_COMPUTE,
         "420100005000000000010000070000000b0000001100000011001100000000000101000000000000020000001100000000000000000000001300000000000000020000001100000000000000000000001300000000000000"},
        {Opcode::LOCAL_REDUCE,
         "4301000048000000000100010000030011000000000000002200000000000000010000000000000000010000000000000000000000000000010000000000000000020000000000000000000000000000"},
        {Opcode::LSU_LOAD,
         "800100002800000017000000000000000101000000000000020000001100000000000000000000001300000000000000"},
        {Opcode::DTE_ISSUE,
         "82010000500000000100000011000000080800000000000001010000000000001700000000000000020000001100000000000000000000001300000000000000000000000000000000000000000000000000000000000000"},
        {Opcode::SRAM_CLEAR, "830100000400000011000000"},
        {Opcode::SRAM_BIND,
         "8401000048000000030000001100000012000000130000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000014000000"},
        {Opcode::SRAM_ALLOC,
         "85010000200000001100000011000000010100000000000010000000000000000101000000000000"},
        {Opcode::SRAM_RESIZE,
         "870100001000000011000000000000000101000000000000"},
        {Opcode::SRAM_RENAME, "88010000080000001100000012000000"},
        {Opcode::SRAM_ALLOC_AT,
         "890100002800000011000000110000000101000000000000010100000000000010000000000000000101000000000000"},
        {Opcode::DTE_WAIT, "c00100000400000011000000"},
        {Opcode::DTE_FENCE, "c101000000000000"},
        {Opcode::EVENT_SET, "c3010000080000001100120011000000"},
        {Opcode::EVENT_WAIT,
         "c40100000c000000110012001100000011000000"},
        {Opcode::GROUP_SYNC, "c5010000080000001100000011000000"},
    }};
    for (const Golden &fixture : golden) {
        const OpcodeManifestEntry &entry = *LookupOpcode(fixture.opcode);
        const auto bytes = EncodeExternalRecord(
            MakeRecord(*LookupRecordSchema(fixture.opcode), Boundary::TYPICAL),
            CapabilitiesFor(entry));
        checks.Check(ExternalRecordHex(bytes) == fixture.hex,
                     std::string(entry.canonical_name) + " golden hex");
    }
}

void CheckDteEndpointSourceSpaceAndCapacity(Checks &checks) {
    ExternalRecord send = MakeRecord(
        *LookupRecordSchema(Opcode::DTE_SEND), Boundary::MINIMUM);
    auto &send_operands = std::get<DteSendOperands>(send.operands);
    checks.Check(send_operands.source_space == EndpointSourceSpace::SRAM,
                 "DTE_SEND source space defaults to SRAM");
    const std::vector<uint8_t> sram_wire = EncodeExternalRecord(send);
    checks.Check(sram_wire[kExternalRecordHeaderSize + 12] == 0,
                 "DTE_SEND default SRAM preserves zero wire bits");

    send_operands.source_space = EndpointSourceSpace::HBM;
    send_operands.source = {SramAddressKind::ABSOLUTE, 0, 0, 0};
    const std::vector<uint8_t> hbm_min_wire = EncodeExternalRecord(send);
    const ExternalRecord hbm_min_record =
        DecodeExternalRecordExact(hbm_min_wire);
    const auto &hbm_min =
        std::get<DteSendOperands>(hbm_min_record.operands);
    checks.Check(hbm_min_wire[kExternalRecordHeaderSize + 12] == 1 &&
                     hbm_min.source_space == EndpointSourceSpace::HBM &&
                     hbm_min.source.absolute_address_bytes == 0,
                 "DTE_SEND HBM minimum wire roundtrip");

    send_operands.source.absolute_address_bytes = UINT64_MAX;
    send_operands.length_bytes = kDteEndpointP2pMaxBytes;
    const std::vector<uint8_t> hbm_max_wire = EncodeExternalRecord(send);
    const ExternalRecord hbm_max_record =
        DecodeExternalRecordExact(hbm_max_wire);
    const auto &hbm_max =
        std::get<DteSendOperands>(hbm_max_record.operands);
    checks.Check(hbm_max.source_space == EndpointSourceSpace::HBM &&
                     hbm_max.source.absolute_address_bytes == UINT64_MAX &&
                     hbm_max.length_bytes == kDteEndpointP2pMaxBytes,
                 "DTE_SEND HBM address and P2P transport maxima roundtrip");

    ExternalRecord invalid = send;
    std::get<DteSendOperands>(invalid.operands).source_space =
        static_cast<EndpointSourceSpace>(2);
    checks.Reject("DTE_SEND source space enum max+1 encode",
                  [&] { EncodeExternalRecord(invalid); });

    invalid = send;
    auto &region_hbm = std::get<DteSendOperands>(invalid.operands);
    region_hbm.source_space = EndpointSourceSpace::HBM;
    region_hbm.source = Address(Boundary::TYPICAL);
    checks.Reject("DTE_SEND HBM region encode",
                  [&] { EncodeExternalRecord(invalid); });

    std::vector<uint8_t> bad_wire = hbm_max_wire;
    bad_wire[kExternalRecordHeaderSize + 12] = 2;
    checks.Reject("DTE_SEND source space enum max+1 decode",
                  [&] { DecodeExternalRecordExact(bad_wire); });
    bad_wire = hbm_max_wire;
    bad_wire[kExternalRecordHeaderSize + 13] = 1;
    checks.Reject("DTE_SEND narrowed header reserved byte",
                  [&] { DecodeExternalRecordExact(bad_wire); });

    bad_wire = EncodeExternalRecord(MakeRecord(
        *LookupRecordSchema(Opcode::DTE_SEND), Boundary::TYPICAL));
    bad_wire[kExternalRecordHeaderSize + 12] = 1;
    checks.Reject("DTE_SEND HBM region decode",
                  [&] { DecodeExternalRecordExact(bad_wire); });

    send_operands.source_space = EndpointSourceSpace::SRAM;
    send_operands.length_bytes = kDteEndpointP2pMaxBytes;
    checks.Accept("DTE_SEND P2P transport maximum", [&] {
        DecodeExternalRecordExact(EncodeExternalRecord(send));
    });
    send_operands.length_bytes = kDteEndpointP2pMaxBytes + 1;
    checks.Reject("DTE_SEND P2P transport max+1",
                  [&] { EncodeExternalRecord(send); });

    ExternalRecord recv = MakeRecord(
        *LookupRecordSchema(Opcode::DTE_RECV), Boundary::MINIMUM);
    auto &recv_operands = std::get<DteRecvOperands>(recv.operands);
    recv_operands.length_bytes = kDteEndpointP2pMaxBytes;
    checks.Accept("DTE_RECV P2P transport maximum", [&] {
        DecodeExternalRecordExact(EncodeExternalRecord(recv));
    });
    recv_operands.length_bytes = kDteEndpointP2pMaxBytes + 1;
    checks.Reject("DTE_RECV P2P transport max+1",
                  [&] { EncodeExternalRecord(recv); });
}

void CheckCollectiveBaselineContracts(Checks &checks) {
    ExternalRecord broadcast = MakeRecord(
        *LookupRecordSchema(Opcode::DTE_SEND), Boundary::MAXIMUM);
    auto &broadcast_operands = std::get<DteSendOperands>(broadcast.operands);
    broadcast_operands.tree_id = 0;
    checks.Accept("baseline BROADCAST accepts tree_id=0", [&] {
        DecodeExternalRecordExact(EncodeExternalRecord(broadcast));
    });
    DteSendOperands keyed_send = broadcast_operands;
    keyed_send.mode = DteSendMode::P2P;
    keyed_send.tree_id = 0;
    keyed_send.length_bytes = kDteEndpointP2pMaxBytes + 1;
    checks.Accept("keyed collective UNICAST SEND may exceed one child", [&] {
        EncodeExternalRecord(ExternalRecord{Opcode::DTE_SEND, keyed_send});
    });
    DteSendOperands partial_send_key = keyed_send;
    partial_send_key.group_id = 0;
    partial_send_key.peer_core = 1;
    checks.Reject("standalone SEND forbids a partial collective key", [&] {
        EncodeExternalRecord(
            ExternalRecord{Opcode::DTE_SEND, partial_send_key});
    });
    DteSendOperands keyed_send_peer = keyed_send;
    keyed_send_peer.peer_core = 1;
    checks.Reject("keyed SEND peer must be graph-derived", [&] {
        EncodeExternalRecord(
            ExternalRecord{Opcode::DTE_SEND, keyed_send_peer});
    });

    broadcast_operands.tree_id = 1;
    checks.Reject("baseline BROADCAST rejects non-zero tree without capability",
                  [&] { EncodeExternalRecord(broadcast); });

    ExternalRecord gather = MakeRecord(
        *LookupRecordSchema(Opcode::DTE_RECV), Boundary::TYPICAL);
    auto &gather_operands = std::get<DteRecvOperands>(gather.operands);
    DteRecvOperands keyed_receive = gather_operands;
    keyed_receive.mode = DteRecvMode::P2P;
    keyed_receive.expected_sources = 0;
    keyed_receive.length_bytes = kDteEndpointP2pMaxBytes + 1;
    checks.Accept("keyed collective UNICAST RECEIVE may exceed one child", [&] {
        EncodeExternalRecord(ExternalRecord{Opcode::DTE_RECV, keyed_receive});
    });
    DteRecvOperands partial_receive_key = keyed_receive;
    partial_receive_key.group_id = 0;
    partial_receive_key.peer_core = 1;
    checks.Reject("standalone RECEIVE forbids a partial collective key", [&] {
        EncodeExternalRecord(
            ExternalRecord{Opcode::DTE_RECV, partial_receive_key});
    });
    DteRecvOperands keyed_receive_peer = keyed_receive;
    keyed_receive_peer.peer_core = 1;
    checks.Reject("keyed RECEIVE peer must be graph-derived", [&] {
        EncodeExternalRecord(
            ExternalRecord{Opcode::DTE_RECV, keyed_receive_peer});
    });

    gather_operands.expected_sources = 0;
    checks.Accept("GATHER codec accepts N=1 expected_sources=0", [&] {
        const auto decoded = DecodeExternalRecordExact(
            EncodeExternalRecord(gather));
        checks.Check(std::get<DteRecvOperands>(decoded.operands)
                             .expected_sources == 0,
                     "GATHER N=1 expected_sources roundtrip");
    });

    ExternalRecord reduce = MakeRecord(
        *LookupRecordSchema(Opcode::DTE_RECV), Boundary::MAXIMUM);
    auto &reduce_operands = std::get<DteRecvOperands>(reduce.operands);
    reduce_operands.expected_sources = 0;
    checks.Accept("REDUCE codec accepts N=1 expected_sources=0", [&] {
        const auto decoded = DecodeExternalRecordExact(
            EncodeExternalRecord(reduce));
        checks.Check(std::get<DteRecvOperands>(decoded.operands)
                             .expected_sources == 0,
                     "REDUCE N=1 expected_sources roundtrip");
    });
}

void CheckMalformedRecords(Checks &checks) {
    const auto good = EncodeExternalRecord(
        MakeRecord(*LookupRecordSchema(Opcode::MATMUL), Boundary::TYPICAL));
    for (std::size_t length = 0; length < kExternalRecordHeaderSize; ++length) {
        std::vector<uint8_t> short_header(good.begin(), good.begin() + length);
        checks.Reject("short header " + std::to_string(length),
                      [&] { DecodeExternalRecordExact(short_header); });
    }
    auto bad = good;
    bad[1] = 2;
    checks.Reject("bad version", [&] { DecodeExternalRecordExact(bad); });
    bad = good;
    bad[2] = 1;
    checks.Reject("bad flags", [&] { DecodeExternalRecordExact(bad); });
    bad = good;
    WriteU32(bad, 4, 1);
    checks.Reject("short declared payload",
                  [&] { DecodeExternalRecordExact(bad); });
    bad = good;
    WriteU32(bad, 4, UINT32_MAX);
    checks.Reject("long declared payload",
                  [&] { DecodeExternalRecordExact(bad); });
    bad = good;
    bad.pop_back();
    checks.Reject("truncated payload", [&] { DecodeExternalRecordExact(bad); });
    bad = good;
    bad.push_back(0);
    checks.Reject("trailing byte", [&] { DecodeExternalRecordExact(bad); });
    checks.Check(DecodeExternalRecord(bad).next_offset == good.size(),
                 "one record stream boundary");
    for (uint8_t opcode : {uint8_t{0}, uint8_t{0x44}, uint8_t{0xf0}}) {
        bad.assign(8, 0);
        bad[0] = opcode;
        bad[1] = 1;
        checks.Reject("invalid/reserved opcode " + std::to_string(opcode),
                      [&] { DecodeExternalRecordExact(bad); });
    }
    checks.Check(ValidateOpcodeValue(256) == OpcodeValidation::UNKNOWN,
                 "wide opcode is unknown");

    bad = good;
    bad[8] = 2;
    checks.Reject("unknown compute enum",
                  [&] { DecodeExternalRecordExact(bad); });
    bad = good;
    bad[9] = 1;
    checks.Reject("compute reserved byte",
                  [&] { DecodeExternalRecordExact(bad); });
    bad = good;
    WriteU32(bad, good.size() - 4, uint32_t{1} << 30);
    checks.Reject("compute parameter high bits",
                  [&] { DecodeExternalRecordExact(bad); });

    auto issue = EncodeExternalRecord(MakeRecord(
        *LookupRecordSchema(Opcode::DTE_ISSUE), Boundary::TYPICAL));
    issue[8] = 3;
    checks.Reject("unknown DTE direction",
                  [&] { DecodeExternalRecordExact(issue); });
    issue = EncodeExternalRecord(MakeRecord(
        *LookupRecordSchema(Opcode::DTE_ISSUE), Boundary::TYPICAL));
    issue[9] = 1;
    checks.Reject("DTE reserved byte",
                  [&] { DecodeExternalRecordExact(issue); });

    auto bind = EncodeExternalRecord(MakeRecord(
        *LookupRecordSchema(Opcode::SRAM_BIND), Boundary::TYPICAL));
    bind[9] = 1;
    checks.Reject("SRAM_BIND reserved header",
                  [&] { DecodeExternalRecordExact(bind); });
    bind = EncodeExternalRecord(MakeRecord(
        *LookupRecordSchema(Opcode::SRAM_BIND), Boundary::TYPICAL));
    WriteU32(bind, 8 + 4 + 3 * 4, 1);
    checks.Reject("SRAM_BIND noncanonical unused input",
                  [&] { DecodeExternalRecordExact(bind); });
    bind = EncodeExternalRecord(MakeRecord(
        *LookupRecordSchema(Opcode::SRAM_BIND), Boundary::TYPICAL));
    bind.pop_back();
    checks.Reject("truncated SRAM_BIND payload",
                  [&] { DecodeExternalRecordExact(bind); });

    auto alloc = EncodeExternalRecord(MakeRecord(
        *LookupRecordSchema(Opcode::SRAM_ALLOC), Boundary::TYPICAL));
    alloc[8 + 25] = 2;
    checks.Reject("unknown spillable value",
                  [&] { DecodeExternalRecordExact(alloc); });
    alloc = EncodeExternalRecord(MakeRecord(
        *LookupRecordSchema(Opcode::SRAM_ALLOC), Boundary::TYPICAL));
    alloc.back() = 1;
    checks.Reject("SRAM_ALLOC reserved tail",
                  [&] { DecodeExternalRecordExact(alloc); });

    auto alloc_at = EncodeExternalRecord(MakeRecord(
        *LookupRecordSchema(Opcode::SRAM_ALLOC_AT), Boundary::TYPICAL));
    alloc_at[8 + 33] = 2;
    checks.Reject("SRAM_ALLOC_AT unknown spillable value",
                  [&] { DecodeExternalRecordExact(alloc_at); });
    alloc_at = EncodeExternalRecord(MakeRecord(
        *LookupRecordSchema(Opcode::SRAM_ALLOC_AT), Boundary::TYPICAL));
    alloc_at.back() = 1;
    checks.Reject("SRAM_ALLOC_AT reserved tail",
                  [&] { DecodeExternalRecordExact(alloc_at); });
}

void CheckPublishedComputeSemantics(Checks &checks) {
    static constexpr std::array<Opcode, 17> published{{
        Opcode::MATMUL, Opcode::CONV, Opcode::MAXPOOL, Opcode::ATTENTION,
        Opcode::GATE, Opcode::MOE_MATMUL, Opcode::GELU, Opcode::SILU,
        Opcode::SWIGLU, Opcode::RELU, Opcode::RESIDUAL, Opcode::LAYERNORM,
        Opcode::RMSNORM, Opcode::ROPE, Opcode::SPLIT_MATMUL,
        Opcode::MERGE_MATMUL, Opcode::DUMMY}};
    for (Opcode opcode : published) {
        const RecordSchema &schema = *LookupRecordSchema(opcode);
        const std::string name(LookupOpcode(opcode)->canonical_name);
        checks.Accept(name + " minimum semantic shape", [&] {
            const ExternalRecord record =
                MakeRecord(schema, Boundary::MINIMUM);
            DecodeExternalRecordExact(EncodeExternalRecord(record));
        });
        checks.Accept(name + " typical semantic shape", [&] {
            const ExternalRecord record =
                MakeRecord(schema, Boundary::TYPICAL);
            DecodeExternalRecordExact(EncodeExternalRecord(record));
        });
        if (schema.parameter_count == 0) {
            checks.Check(opcode == Opcode::DUMMY,
                         name + " parameterless schema is explicit");
            continue;
        }
        ExternalRecord missing = MakeRecord(schema, Boundary::TYPICAL);
        std::get<ComputeOperands>(missing.operands).parameters.pop_back();
        checks.Reject(name + " missing fixed-schema parameter",
                      [&] { EncodeExternalRecord(missing); });
        ExternalRecord zero = MakeRecord(schema, Boundary::TYPICAL);
        std::get<ComputeOperands>(zero.operands).parameters[0] = 0;
        checks.Reject(name + " zero primary dimension",
                      [&] { EncodeExternalRecord(zero); });
    }

    ExternalRecord record = MakeRecord(
        *LookupRecordSchema(Opcode::CONV), Boundary::TYPICAL);
    SetComputeParameter(record, "sX", 0);
    checks.Reject("CONV zero stride",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::MAXPOOL),
                        Boundary::TYPICAL);
    SetComputeParameter(record, "sY", 0);
    checks.Reject("MAXPOOL zero stride",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::ATTENTION),
                        Boundary::TYPICAL);
    SetComputeParameter(record, "R", 0);
    checks.Reject("ATTENTION zero R divisor",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::ROPE),
                        Boundary::TYPICAL);
    SetComputeParameter(record, "NH", 0);
    checks.Reject("ROPE zero NH divisor",
                  [&] { EncodeExternalRecord(record); });

    record = MakeRecord(*LookupRecordSchema(Opcode::CONV),
                        Boundary::MINIMUM);
    SetComputeParameter(record, "kY", 2);
    checks.Reject("CONV negative derived extent",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::ATTENTION),
                        Boundary::MINIMUM);
    SetComputeParameter(record, "R", 1);
    checks.Reject("ATTENTION zero derived output",
                  [&] { EncodeExternalRecord(record); });

    record = MakeRecord(*LookupRecordSchema(Opcode::GATE),
                        Boundary::MINIMUM);
    SetComputeParameter(record, "K", 2);
    checks.Reject("GATE K exceeds E_N",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::MOE_MATMUL),
                        Boundary::MINIMUM);
    SetComputeParameter(record, "K", 2);
    checks.Reject("MOE_MATMUL K exceeds E_N",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::MOE_MATMUL),
                        Boundary::TYPICAL);
    SetComputeParameter(record, "is_merge", 2);
    checks.Reject("MOE_MATMUL invalid is_merge",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::MOE_MATMUL),
                        Boundary::TYPICAL);
    SetComputeParameter(record, "need_choose", 2);
    checks.Reject("MOE_MATMUL invalid need_choose",
                  [&] { EncodeExternalRecord(record); });

    for (Opcode opcode : {Opcode::SPLIT_MATMUL,
                          Opcode::MERGE_MATMUL}) {
        record = MakeRecord(*LookupRecordSchema(opcode), Boundary::TYPICAL);
        SetComputeParameter(record, "dim", 3);
        checks.Reject(std::string(LookupOpcode(opcode)->canonical_name) +
                          " invalid dim",
                      [&] { EncodeExternalRecord(record); });
    }
    record = MakeRecord(*LookupRecordSchema(Opcode::SPLIT_MATMUL),
                        Boundary::TYPICAL);
    SetComputeParameter(record, "slice", kSramBindInputLimit + 1);
    checks.Reject("SPLIT_MATMUL too many slices",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::MERGE_MATMUL),
                        Boundary::TYPICAL);
    SetComputeParameter(record, "slice", 0);
    checks.Reject("MERGE_MATMUL zero slices",
                  [&] { EncodeExternalRecord(record); });

    record = MakeRecord(*LookupRecordSchema(Opcode::MATMUL),
                        Boundary::MINIMUM);
    std::get<ComputeOperands>(record.operands).datatype =
        ExternalDataType::FP16;
    SetComputeParameter(record, "B", kExternalNpuParameterMax);
    SetComputeParameter(record, "T", 2);
    checks.Reject("MATMUL int byte-size overflow",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::MATMUL),
                        Boundary::MINIMUM);
    std::get<ComputeOperands>(record.operands).datatype =
        ExternalDataType::INT8;
    SetComputeParameter(record, "B", kExternalNpuParameterMax);
    SetComputeParameter(record, "T", kExternalNpuParameterMax);
    SetComputeParameter(record, "C", kExternalNpuParameterMax);
    checks.Reject("MATMUL u64 product overflow",
                  [&] { EncodeExternalRecord(record); });

    record = MakeRecord(*LookupRecordSchema(Opcode::RELU),
                        Boundary::MINIMUM);
    std::get<ComputeOperands>(record.operands).datatype =
        ExternalDataType::INT8;
    SetComputeParameter(record, "N", kExternalNpuParameterMax);
    checks.Accept("RELU 30-bit parameter maximum",
                  [&] { EncodeExternalRecord(record); });

    record = MakeRecord(*LookupRecordSchema(Opcode::SWIGLU),
                        Boundary::MINIMUM);
    std::get<ComputeOperands>(record.operands).datatype =
        ExternalDataType::FP16;
    SetComputeParameter(
        record, "N",
        static_cast<uint64_t>(std::numeric_limits<int>::max()) / 4);
    checks.Accept("SWIGLU concat input FP16 byte boundary",
                  [&] { EncodeExternalRecord(record); });
    SetComputeParameter(
        record, "N",
        static_cast<uint64_t>(std::numeric_limits<int>::max()) / 4 + 1);
    checks.Reject("SWIGLU concat input FP16 byte overflow",
                  [&] { EncodeExternalRecord(record); });
    // External parameters are unsigned. A producer-side negative value has
    // high bits set and is rejected by the existing 30-bit range check.
}

void CheckTypedRejections(Checks &checks) {
    ExternalRecord record = MakeRecord(
        *LookupRecordSchema(Opcode::MATMUL), Boundary::TYPICAL);
    std::get<ComputeOperands>(record.operands).input_offset_bytes = 65536;
    checks.Reject("NPU offset max+1",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::MATMUL),
                        Boundary::TYPICAL);
    std::get<ComputeOperands>(record.operands).parameters[0] =
        kExternalNpuParameterMax + 1;
    checks.Reject("NPU parameter max+1",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::MATMUL),
                        Boundary::TYPICAL);
    std::get<ComputeOperands>(record.operands).datatype =
        ExternalDataType::INT32;
    checks.Reject("generic ComputeOperands rejects INT32",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::MATMUL),
                        Boundary::TYPICAL);
    std::get<ComputeOperands>(record.operands).datatype =
        ExternalDataType::FP32;
    checks.Reject("generic ComputeOperands rejects CE-only FP32",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::MATMUL),
                        Boundary::TYPICAL);
    record.operands = TokenOperands{1};
    checks.Reject("operand variant mismatch",
                  [&] { EncodeExternalRecord(record); });

    record = MakeRecord(*LookupRecordSchema(Opcode::LOCAL_REDUCE),
                        Boundary::TYPICAL);
    {
        auto &fp32 = std::get<LocalReduceOperands>(record.operands);
        fp32.input_dtype = LocalReduceDataType::FP32;
        fp32.accumulator_dtype = LocalReduceDataType::FP32;
        fp32.output_dtype = LocalReduceDataType::FP32;
        fp32.input_count = 2;
        fp32.element_count = 512;
        fp32.input_stride_bytes = 2048;
        fp32.source.absolute_address_bytes = 0x1000;
        fp32.destination.absolute_address_bytes = 0x3000;
    }
    checks.Accept("exact DP2 FP32 LOCAL_REDUCE round trips", [&] {
        const auto bytes = EncodeExternalRecord(record);
        const auto decoded = DecodeExternalRecordExact(bytes);
        const auto &got = std::get<LocalReduceOperands>(decoded.operands);
        checks.Check(decoded.opcode == Opcode::LOCAL_REDUCE &&
                         got.input_dtype == LocalReduceDataType::FP32 &&
                         got.accumulator_dtype == LocalReduceDataType::FP32 &&
                         got.output_dtype == LocalReduceDataType::FP32 &&
                         got.input_count == 2 && got.element_count == 512 &&
                         got.input_stride_bytes == 2048,
                     "exact DP2 FP32 LOCAL_REDUCE wire is stable");
    });
    auto bad_fp32 = record;
    std::get<LocalReduceOperands>(bad_fp32.operands).input_count = 3;
    checks.Reject("FP32 LOCAL_REDUCE rejects count other than two",
                  [&] { EncodeExternalRecord(bad_fp32); });
    bad_fp32 = record;
    std::get<LocalReduceOperands>(bad_fp32.operands).element_count = 511;
    checks.Reject("FP32 LOCAL_REDUCE rejects non-512 elements",
                  [&] { EncodeExternalRecord(bad_fp32); });
    bad_fp32 = record;
    std::get<LocalReduceOperands>(bad_fp32.operands).input_stride_bytes = 2044;
    checks.Reject("FP32 LOCAL_REDUCE rejects non-2048 stride",
                  [&] { EncodeExternalRecord(bad_fp32); });
    bad_fp32 = record;
    std::get<LocalReduceOperands>(bad_fp32.operands).reduce_op =
        ReduceOperator::MAX;
    checks.Reject("FP32 LOCAL_REDUCE rejects non-SUM",
                  [&] { EncodeExternalRecord(bad_fp32); });
    bad_fp32 = record;
    std::get<LocalReduceOperands>(bad_fp32.operands)
        .destination.absolute_address_bytes = 0x3002;
    checks.Reject("FP32 LOCAL_REDUCE rejects non-4-byte alignment",
                  [&] { EncodeExternalRecord(bad_fp32); });

    record = MakeRecord(*LookupRecordSchema(Opcode::LOCAL_REDUCE),
                        Boundary::TYPICAL);
    std::get<LocalReduceOperands>(record.operands).input_dtype =
        LocalReduceDataType::FP32;
    checks.Reject("LOCAL_REDUCE rejects mixed FP32/FP16 dtype",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::LOCAL_REDUCE),
                        Boundary::TYPICAL);
    std::get<LocalReduceOperands>(record.operands).accumulator_dtype =
        LocalReduceDataType::FP16;
    checks.Reject("LOCAL_REDUCE accumulator must be FP32",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::LOCAL_REDUCE),
                        Boundary::TYPICAL);
    std::get<LocalReduceOperands>(record.operands).output_dtype =
        LocalReduceDataType::FP32;
    checks.Reject("LOCAL_REDUCE output must be FP16",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::LOCAL_REDUCE),
                        Boundary::TYPICAL);
    std::get<LocalReduceOperands>(record.operands).reduce_op =
        ReduceOperator::MAX;
    checks.Reject("LOCAL_REDUCE op must be SUM",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::LOCAL_REDUCE),
                        Boundary::TYPICAL);
    std::get<LocalReduceOperands>(record.operands).rounding =
        static_cast<LocalReduceRoundingMode>(1);
    checks.Reject("LOCAL_REDUCE rounding must be RNE",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::LOCAL_REDUCE),
                        Boundary::TYPICAL);
    std::get<LocalReduceOperands>(record.operands).order =
        static_cast<LocalReduceOrder>(1);
    checks.Reject("LOCAL_REDUCE order must be rank-major",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::LOCAL_REDUCE),
                        Boundary::TYPICAL);
    std::get<LocalReduceOperands>(record.operands).input_count = 0;
    checks.Reject("LOCAL_REDUCE zero inputs",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::LOCAL_REDUCE),
                        Boundary::TYPICAL);
    std::get<LocalReduceOperands>(record.operands).element_count = 0;
    checks.Reject("LOCAL_REDUCE zero elements",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::LOCAL_REDUCE),
                        Boundary::TYPICAL);
    std::get<LocalReduceOperands>(record.operands).input_stride_bytes = 32;
    checks.Reject("LOCAL_REDUCE non-tight stride",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::LOCAL_REDUCE),
                        Boundary::TYPICAL);
    auto &local_source =
        std::get<LocalReduceOperands>(record.operands).source;
    local_source = {SramAddressKind::REGION, 0, 1, 0};
    checks.Reject("LOCAL_REDUCE named source rejected in V1",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::LOCAL_REDUCE),
                        Boundary::TYPICAL);
    std::get<LocalReduceOperands>(record.operands)
        .destination.absolute_address_bytes = 1;
    checks.Reject("LOCAL_REDUCE misaligned destination",
                  [&] { EncodeExternalRecord(record); });

    record = MakeRecord(*LookupRecordSchema(Opcode::SRAM_CLEAR),
                        Boundary::TYPICAL);
    std::get<SymbolOperands>(record.operands).symbol_index =
        uint64_t{UINT32_MAX} + 1;
    checks.Reject("symbol index max+1",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::SRAM_BIND),
                        Boundary::TYPICAL);
    std::get<SramBindOperands>(record.operands).input_count = 0;
    checks.Reject("SRAM_BIND zero inputs",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::SRAM_BIND),
                        Boundary::TYPICAL);
    std::get<SramBindOperands>(record.operands).input_count = 17;
    checks.Reject("SRAM_BIND too many inputs",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::SRAM_BIND),
                        Boundary::TYPICAL);
    std::get<SramBindOperands>(record.operands).input_symbol_indices[0] =
        uint64_t{UINT32_MAX} + 1;
    checks.Reject("SRAM_BIND input symbol max+1",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::SRAM_BIND),
                        Boundary::TYPICAL);
    std::get<SramBindOperands>(record.operands).output_symbol_index =
        uint64_t{UINT32_MAX} + 1;
    checks.Reject("SRAM_BIND output symbol max+1",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::SRAM_BIND),
                        Boundary::TYPICAL);
    std::get<SramBindOperands>(record.operands).input_symbol_indices[3] = 1;
    checks.Reject("SRAM_BIND typed noncanonical unused input",
                  [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::SRAM_BIND),
                        Boundary::TYPICAL);
    record.operands = SymbolOperands{1};
    checks.Reject("SRAM_BIND operand variant mismatch",
                  [&] { EncodeExternalRecord(record); });

    record = MakeRecord(*LookupRecordSchema(Opcode::EVENT_SET),
                        Boundary::TYPICAL);
    std::get<EventSetOperands>(record.operands).source_core =
        uint64_t{UINT16_MAX} + 1;
    checks.Reject("core id max+1", [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::DTE_WAIT),
                        Boundary::TYPICAL);
    std::get<TokenOperands>(record.operands).token =
        uint64_t{UINT32_MAX} + 1;
    checks.Reject("token max+1", [&] { EncodeExternalRecord(record); });

    record = MakeRecord(*LookupRecordSchema(Opcode::LSU_LOAD),
                        Boundary::TYPICAL);
    auto &address = std::get<LsuOperands>(record.operands).sram;
    address.absolute_address_bytes = 23;
    checks.Reject("absolute/region XOR", [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::LSU_LOAD),
                        Boundary::TYPICAL);
    std::get<LsuOperands>(record.operands).size_bytes = 0;
    checks.Reject("zero LSU length", [&] { EncodeExternalRecord(record); });
    record = MakeRecord(*LookupRecordSchema(Opcode::EVENT_WAIT),
                        Boundary::TYPICAL);
    std::get<EventWaitOperands>(record.operands).count = 0;
    checks.Reject("zero EVENT_WAIT count",
                  [&] { EncodeExternalRecord(record); });
    checks.Reject("offset outside input", [&] {
        DecodeExternalRecord(std::vector<uint8_t>{}, 1);
    });
}

void CheckExactStage2Rejections(Checks &checks) {
    ExternalRecord record = MakeRecord(
        *LookupRecordSchema(Opcode::ROPE_QK_EXACT), Boundary::TYPICAL);
    auto encoded = EncodeExternalRecord(record);
    encoded[10] = 1;
    checks.Reject("ROPE reserved wire bytes must be zero",
                  [&] { DecodeExternalRecord(encoded); });

    auto &rope = std::get<RopeQkExactOperands>(record.operands);
    rope.packed_layout = static_cast<RopePackedLayout>(1);
    checks.Reject("ROPE packed layout enum",
                  [&] { EncodeExternalRecord(record); });
    rope.packed_layout = RopePackedLayout::Q_K_V;
    rope.rope_theta_f64_bits = UINT64_C(0x7ff8000000000000);
    checks.Reject("ROPE theta NaN", [&] { EncodeExternalRecord(record); });
    rope.rope_theta_f64_bits = DoubleBits(10000.0);
    rope.rotary_dim = 3;
    checks.Reject("ROPE rotary dimension must be even",
                  [&] { EncodeExternalRecord(record); });
    rope.rotary_dim = 4;
    rope.rank_num_heads = 1;
    checks.Reject("ROPE head quotient",
                  [&] { EncodeExternalRecord(record); });
    rope.rank_num_heads = 2;
    rope.context_max = 129;
    checks.Reject("ROPE context coverage",
                  [&] { EncodeExternalRecord(record); });
    rope.context_max = 8;
    rope.logical_tokens = 16;
    checks.Accept("ROPE aggregate tokens may exceed per-request context",
                  [&] { EncodeExternalRecord(record); });
    rope.logical_tokens = uint64_t{UINT32_MAX} + 1;
    checks.Reject("ROPE logical token u32 overflow",
                  [&] { EncodeExternalRecord(record); });

    record = MakeRecord(*LookupRecordSchema(Opcode::ATTENTION_EXACT),
                        Boundary::TYPICAL);
    auto &attention = std::get<AttentionExactOperands>(record.operands);
    attention.query_key_pairs = 35;
    checks.Reject("ATTENTION prefill pair formula",
                  [&] { EncodeExternalRecord(record); });
    attention.query_key_pairs = 36;
    attention.rank_kv_read_bytes = 1;
    checks.Reject("ATTENTION prefill KV read formula",
                  [&] { EncodeExternalRecord(record); });
    attention.rank_kv_read_bytes = 0;
    attention.rank_kv_write_bytes = 255;
    checks.Reject("ATTENTION KV write formula",
                  [&] { EncodeExternalRecord(record); });
    attention.rank_kv_write_bytes = 256;
    attention.causal = false;
    checks.Reject("ATTENTION causal exact",
                  [&] { EncodeExternalRecord(record); });
    attention.causal = true;
    attention.mode = ExactAttentionMode::DECODE;
    attention.query_tokens = 1;
    attention.context_sum = 8;
    attention.context_max = 8;
    attention.query_key_pairs = 8;
    attention.rank_kv_read_bytes = 256;
    attention.rank_kv_write_bytes = 32;
    checks.Accept("ATTENTION decode exact",
                  [&] { EncodeExternalRecord(record); });
    attention.query_key_pairs = 7;
    checks.Reject("ATTENTION decode pair formula",
                  [&] { EncodeExternalRecord(record); });
    attention.mode = ExactAttentionMode::EXACT_PROFILE;
    attention.query_tokens = 8;
    attention.context_sum = 44;
    attention.context_max = 16;
    attention.query_key_pairs = 47;
    attention.rank_kv_read_bytes = 1280;
    attention.rank_kv_write_bytes = 255;
    checks.Reject("ATTENTION static-profile write formula",
                  [&] { EncodeExternalRecord(record); });
    attention.rank_kv_write_bytes = 256;
    checks.Accept("ATTENTION static-profile mixed exact",
                  [&] { EncodeExternalRecord(record); });
    attention.query_key_pairs = 7;
    checks.Reject("ATTENTION static-profile pair lower bound",
                  [&] { EncodeExternalRecord(record); });
    attention.query_key_pairs = 129;
    checks.Reject("ATTENTION static-profile pair upper bound",
                  [&] { EncodeExternalRecord(record); });
    attention.query_key_pairs = 47;
    attention.rank_kv_read_bytes = 1;
    checks.Reject("ATTENTION static-profile read alignment",
                  [&] { EncodeExternalRecord(record); });
    attention.rank_kv_read_bytes = 2880;
    checks.Reject("ATTENTION static-profile read upper bound",
                  [&] { EncodeExternalRecord(record); });
    attention.rank_kv_read_bytes = 1280;
    attention.mode = ExactAttentionMode::TRAIN_FORWARD;
    attention.query_tokens = 8;
    attention.context_sum = 8;
    attention.context_max = 8;
    attention.query_key_pairs = 36;
    attention.rank_kv_read_bytes = 0;
    attention.rank_kv_write_bytes = 0;
    checks.Accept("ATTENTION train-forward exact",
                  [&] { EncodeExternalRecord(record); });
    attention.rank_kv_write_bytes = 256;
    checks.Reject("ATTENTION train-forward rejects KV write",
                  [&] { EncodeExternalRecord(record); });
    attention.rank_kv_write_bytes = 0;
    attention.mode = static_cast<ExactAttentionMode>(4);
    checks.Reject("ATTENTION mode enum",
                  [&] { EncodeExternalRecord(record); });

    record = MakeRecord(*LookupRecordSchema(Opcode::EMBEDDING_LOOKUP),
                        Boundary::TYPICAL);
    auto &embedding = std::get<EmbeddingLookupOperands>(record.operands);
    embedding.index_datatype = ExternalDataType::FP16;
    checks.Reject("EMBEDDING index datatype",
                  [&] { EncodeExternalRecord(record); });
    embedding.index_datatype = ExternalDataType::INT32;
    embedding.placement = static_cast<EmbeddingPlacement>(1);
    checks.Reject("EMBEDDING placement enum",
                  [&] { EncodeExternalRecord(record); });
    embedding.placement = EmbeddingPlacement::REPLICATED;
    embedding.logical_rows = 7;
    checks.Reject("EMBEDDING TP row quotient",
                  [&] { EncodeExternalRecord(record); });
    embedding.logical_rows = 8;
    embedding.vocab_size = uint64_t{UINT32_MAX} + 1;
    checks.Reject("EMBEDDING vocab u32 overflow",
                  [&] { EncodeExternalRecord(record); });

    record = MakeRecord(*LookupRecordSchema(Opcode::GREEDY_SAMPLE),
                        Boundary::TYPICAL);
    auto &greedy = std::get<GreedySampleOperands>(record.operands);
    greedy.tp_degree = 2;
    checks.Reject("GREEDY TP exact", [&] { EncodeExternalRecord(record); });
    greedy.tp_degree = 1;
    greedy.sample_count = 0;
    checks.Reject("GREEDY nonzero sample count",
                  [&] { EncodeExternalRecord(record); });
    greedy.sample_count = 9;
    checks.Reject("GREEDY sample count within rows",
                  [&] { EncodeExternalRecord(record); });
    greedy.sample_count = 1;
    greedy.comparisons = 30;
    checks.Reject("GREEDY comparison formula",
                  [&] { EncodeExternalRecord(record); });
    greedy.comparisons = 31;
    greedy.output_datatype = ExternalDataType::FP16;
    checks.Reject("GREEDY output datatype",
                  [&] { EncodeExternalRecord(record); });

    record = MakeRecord(
        *LookupRecordSchema(Opcode::CROSS_ENTROPY_FORWARD),
        Boundary::TYPICAL);
    auto &ce = std::get<CrossEntropyForwardOperands>(record.operands);
    checks.Accept("CROSS_ENTROPY_FORWARD exact",
                  [&] { EncodeExternalRecord(record); });
    ce.loss_datatype = ExternalDataType::INT8;
    checks.Reject("CROSS_ENTROPY_FORWARD loss FP32 exact",
                  [&] { EncodeExternalRecord(record); });
    ce.loss_datatype = ExternalDataType::FP16;
    checks.Reject("CROSS_ENTROPY_FORWARD loss rejects FP16",
                  [&] { EncodeExternalRecord(record); });
    ce.loss_datatype = ExternalDataType::INT32;
    checks.Reject("CROSS_ENTROPY_FORWARD loss rejects INT32",
                  [&] { EncodeExternalRecord(record); });
    ce.loss_datatype = ExternalDataType::FP32;
    ce.reduction = static_cast<CrossEntropyReduction>(1);
    checks.Reject("CROSS_ENTROPY_FORWARD reduction exact",
                  [&] { EncodeExternalRecord(record); });
    ce.reduction = CrossEntropyReduction::NONE;
    ce.logical_rows = 7;
    checks.Reject("CROSS_ENTROPY_FORWARD TP row quotient",
                  [&] { EncodeExternalRecord(record); });
    ce.logical_rows = 8;
    ce.vocab_size = 1;
    checks.Reject("CROSS_ENTROPY_FORWARD vocab lower bound",
                  [&] { EncodeExternalRecord(record); });
    ce.vocab_size = uint64_t{UINT32_MAX} + 1;
    checks.Reject("CROSS_ENTROPY_FORWARD vocab u32 overflow",
                  [&] { EncodeExternalRecord(record); });

    record = MakeRecord(
        *LookupRecordSchema(Opcode::CROSS_ENTROPY_BACKWARD),
        Boundary::TYPICAL);
    auto &ce_backward =
        std::get<CrossEntropyBackwardOperands>(record.operands);
    checks.Accept("CROSS_ENTROPY_BACKWARD exact",
                  [&] { EncodeExternalRecord(record); });
    ce_backward.logits_datatype = ExternalDataType::FP32;
    checks.Reject("CROSS_ENTROPY_BACKWARD logits FP16 exact",
                  [&] { EncodeExternalRecord(record); });
    ce_backward.logits_datatype = ExternalDataType::FP16;
    ce_backward.label_datatype = ExternalDataType::FP16;
    checks.Reject("CROSS_ENTROPY_BACKWARD labels INT32 exact",
                  [&] { EncodeExternalRecord(record); });
    ce_backward.label_datatype = ExternalDataType::INT32;
    ce_backward.upstream_datatype = ExternalDataType::FP16;
    checks.Reject("CROSS_ENTROPY_BACKWARD upstream FP32 exact",
                  [&] { EncodeExternalRecord(record); });
    ce_backward.upstream_datatype = ExternalDataType::FP32;
    ce_backward.output_datatype = ExternalDataType::FP32;
    checks.Reject("CROSS_ENTROPY_BACKWARD output FP16 exact",
                  [&] { EncodeExternalRecord(record); });
    ce_backward.output_datatype = ExternalDataType::FP16;
    ce_backward.reduction = static_cast<CrossEntropyReduction>(1);
    checks.Reject("CROSS_ENTROPY_BACKWARD reduction NONE exact",
                  [&] { EncodeExternalRecord(record); });
    ce_backward.reduction = CrossEntropyReduction::NONE;
    ce_backward.upstream_elements = 1;
    checks.Reject("CROSS_ENTROPY_BACKWARD per-row upstream quotient",
                  [&] { EncodeExternalRecord(record); });
    ce_backward.upstream_mode = CrossEntropyUpstreamMode::SCALAR;
    checks.Accept("CROSS_ENTROPY_BACKWARD scalar ABI",
                  [&] { EncodeExternalRecord(record); });
    ce_backward.upstream_elements = 4;
    checks.Reject("CROSS_ENTROPY_BACKWARD scalar upstream quotient",
                  [&] { EncodeExternalRecord(record); });
    ce_backward.upstream_mode =
        static_cast<CrossEntropyUpstreamMode>(2);
    checks.Reject("CROSS_ENTROPY_BACKWARD upstream mode",
                  [&] { EncodeExternalRecord(record); });

    record = MakeRecord(*LookupRecordSchema(Opcode::SGD_UPDATE),
                        Boundary::TYPICAL);
    auto &sgd = std::get<SgdUpdateOperands>(record.operands);
    checks.Accept("SGD_UPDATE exact", [&] { EncodeExternalRecord(record); });
    sgd.gradient_datatype = ExternalDataType::FP16;
    checks.Reject("SGD_UPDATE gradient FP32 exact",
                  [&] { EncodeExternalRecord(record); });
    sgd.gradient_datatype = ExternalDataType::FP32;
    sgd.updated_weight.absolute_address_bytes++;
    checks.Reject("SGD_UPDATE in-place exact",
                  [&] { EncodeExternalRecord(record); });
    sgd.updated_weight = sgd.weight;
    sgd.momentum_f64_bits = DoubleBits(0.5);
    checks.Reject("SGD_UPDATE momentum exact zero",
                  [&] { EncodeExternalRecord(record); });
    sgd.momentum_f64_bits = 0;
    sgd.learning_rate_f64_bits = DoubleBits(0.0);
    checks.Reject("SGD_UPDATE learning rate positive",
                  [&] { EncodeExternalRecord(record); });
    sgd.learning_rate_f64_bits = DoubleBits(
        std::numeric_limits<double>::infinity());
    checks.Reject("SGD_UPDATE learning rate finite",
                  [&] { EncodeExternalRecord(record); });
}

} // namespace

RecordCodecSelfTestResult CheckIsaV1RecordCodec() {
    Checks checks;
    CheckManifest(checks);
    CheckBoundariesAndStream(checks);
    CheckGoldenHex(checks);
    CheckDteEndpointSourceSpaceAndCapacity(checks);
    CheckCollectiveBaselineContracts(checks);
    CheckMalformedRecords(checks);
    CheckPublishedComputeSemantics(checks);
    CheckTypedRejections(checks);
    CheckExactStage2Rejections(checks);
    return std::move(checks.result);
}

int RunIsaV1RecordCodecSelfTest() {
    const RecordCodecSelfTestResult result = CheckIsaV1RecordCodec();
    if (result.passed()) {
        std::cout << "ISA v1 record codec selftest passed (" << result.checks
                  << " checks)\n";
        return 0;
    }
    std::cerr << "ISA v1 record codec selftest failed (" << result.failures.size()
              << "/" << result.checks << ")\n";
    for (const std::string &failure : result.failures)
        std::cerr << "  " << failure << '\n';
    return 1;
}
