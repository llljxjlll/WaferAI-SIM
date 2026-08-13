#include "isa/record_codec.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

namespace {

constexpr uint32_t kComputePrefixSize = 8;
constexpr uint32_t kAddressSize = 24;
constexpr uint32_t kEndpointPayloadSize = 72;
constexpr uint32_t kReducePayloadSize = 80;
constexpr uint32_t kLsuPayloadSize = 40;
constexpr uint32_t kDteIssuePayloadSize = 80;
constexpr uint32_t kSymbolPayloadSize = 4;
constexpr uint32_t kSramBindPayloadSize = 72;
constexpr uint32_t kSramAllocPayloadSize = 32;
constexpr uint32_t kSramResizePayloadSize = 16;
constexpr uint32_t kSramRenamePayloadSize = 8;
constexpr uint32_t kTokenPayloadSize = 4;
constexpr uint32_t kEventSetPayloadSize = 8;
constexpr uint32_t kEventWaitPayloadSize = 12;
constexpr uint32_t kGroupSyncPayloadSize = 8;
constexpr uint64_t kReservedCollectiveId = 0xffffffffULL;

[[noreturn]] void Fail(const std::string &message) {
    throw RecordCodecError(message);
}

void Require(bool condition, const std::string &message) {
    if (!condition)
        Fail(message);
}

std::string HexByte(uint64_t value) {
    std::ostringstream stream;
    stream << "0x" << std::hex << std::uppercase << std::setw(2)
           << std::setfill('0') << value;
    return stream.str();
}

void RequireU16(uint64_t value, std::string_view field) {
    if (value > std::numeric_limits<uint16_t>::max())
        Fail(std::string(field) + " exceeds u16");
}

void RequireU32(uint64_t value, std::string_view field) {
    if (value > std::numeric_limits<uint32_t>::max())
        Fail(std::string(field) + " exceeds u32");
}

void AppendLittleEndian(std::vector<uint8_t> &bytes, uint64_t value,
                        std::size_t width) {
    for (std::size_t i = 0; i < width; ++i)
        bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
}

uint64_t ReadLittleEndian(const std::vector<uint8_t> &bytes,
                          std::size_t offset, std::size_t width,
                          std::string_view field) {
    if (offset > bytes.size() || width > bytes.size() - offset)
        Fail("truncated external record field: " + std::string(field));
    uint64_t value = 0;
    for (std::size_t i = 0; i < width; ++i)
        value |= uint64_t{bytes[offset + i]} << (8 * i);
    return value;
}

void RequireZero(const std::vector<uint8_t> &bytes, std::size_t offset,
                 std::size_t width, std::string_view field) {
    if (offset > bytes.size() || width > bytes.size() - offset)
        Fail("truncated external record field: " + std::string(field));
    for (std::size_t i = 0; i < width; ++i) {
        if (bytes[offset + i] != 0)
            Fail(std::string(field) + " reserved bytes must be zero");
    }
}

template <class Enum>
uint8_t EnumByte(Enum value) {
    return static_cast<uint8_t>(value);
}

template <class T>
const T &RequireOperands(const ExternalRecord &record,
                         std::string_view schema_name) {
    const T *operands = std::get_if<T>(&record.operands);
    if (operands == nullptr)
        Fail("external record operand variant does not match schema: " +
             std::string(schema_name));
    return *operands;
}

bool IsNone(const SramAddressOperand &address) {
    return address.kind == SramAddressKind::NONE;
}

void ValidateAddress(const SramAddressOperand &address, bool allow_none,
                     std::string_view field) {
    const uint8_t raw_kind = EnumByte(address.kind);
    if (raw_kind > EnumByte(SramAddressKind::REGION))
        Fail(std::string(field) + " has an invalid SRAM address kind");
    RequireU32(address.region_symbol_index,
               std::string(field) + ".region_symbol_index");
    switch (address.kind) {
    case SramAddressKind::NONE:
        Require(allow_none, std::string(field) + " must be present");
        Require(address.absolute_address_bytes == 0 &&
                    address.region_symbol_index == 0 &&
                    address.region_offset_bytes == 0,
                std::string(field) + " absent address carries data");
        return;
    case SramAddressKind::ABSOLUTE:
        Require(address.region_symbol_index == 0 &&
                    address.region_offset_bytes == 0,
                std::string(field) +
                    " must select exactly one of absolute or region+offset");
        return;
    case SramAddressKind::REGION:
        Require(address.absolute_address_bytes == 0,
                std::string(field) +
                    " must select exactly one of absolute or region+offset");
        return;
    }
}

void EncodeAddress(std::vector<uint8_t> &payload,
                   const SramAddressOperand &address) {
    payload.push_back(EnumByte(address.kind));
    AppendLittleEndian(payload, 0, 3);
    AppendLittleEndian(payload, address.region_symbol_index, 4);
    AppendLittleEndian(payload, address.absolute_address_bytes, 8);
    AppendLittleEndian(payload, address.region_offset_bytes, 8);
}

SramAddressOperand DecodeAddress(const std::vector<uint8_t> &payload,
                                 std::size_t offset,
                                 std::string_view field) {
    SramAddressOperand address;
    address.kind = static_cast<SramAddressKind>(
        ReadLittleEndian(payload, offset, 1, field));
    RequireZero(payload, offset + 1, 3,
                std::string(field) + ".reserved");
    address.region_symbol_index =
        ReadLittleEndian(payload, offset + 4, 4,
                         std::string(field) + ".region_symbol_index");
    address.absolute_address_bytes =
        ReadLittleEndian(payload, offset + 8, 8,
                         std::string(field) + ".absolute_address_bytes");
    address.region_offset_bytes =
        ReadLittleEndian(payload, offset + 16, 8,
                         std::string(field) + ".region_offset_bytes");
    return address;
}

void ValidateCompletion(EndpointCompletion completion, uint64_t token) {
    if (EnumByte(completion) > EnumByte(EndpointCompletion::SYNC))
        Fail("endpoint completion enum is invalid");
    RequireU32(token, "endpoint token");
    if (completion == EndpointCompletion::ASYNC)
        Require(token != 0, "asynchronous endpoint token must be non-zero");
    else
        Require(token == 0, "synchronous endpoint token must be zero");
}

void ValidateEndpointDataType(EndpointDataType datatype) {
    if (EnumByte(datatype) > EnumByte(EndpointDataType::INT64))
        Fail("endpoint datatype enum is invalid");
}

void ValidateReduceOperator(ReduceOperator reduce_op, bool required) {
    if (EnumByte(reduce_op) > EnumByte(ReduceOperator::MAX))
        Fail("reduce_op enum is invalid");
    if (required)
        Require(reduce_op == ReduceOperator::SUM ||
                    reduce_op == ReduceOperator::MAX,
                "reduction requires SUM or MAX");
    else
        Require(reduce_op == ReduceOperator::NONE,
                "non-reduction record forbids reduce_op");
}

void ValidateCollectiveKey(uint64_t group_id, uint64_t collective_id,
                           uint64_t epoch) {
    RequireU32(group_id, "group_id");
    RequireU32(collective_id, "collective_id");
    RequireU32(epoch, "epoch");
    Require(group_id != 0, "group_id must be non-zero");
    Require(collective_id != kReservedCollectiveId,
            "collective_id 0xFFFFFFFF is reserved for GROUP_SYNC");
}

void ValidateDteSend(const DteSendOperands &operands) {
    if (EnumByte(operands.mode) > EnumByte(DteSendMode::BROADCAST))
        Fail("DTE_SEND mode enum is invalid");
    if (EnumByte(operands.source_space) >
        EnumByte(EndpointSourceSpace::HBM))
        Fail("DTE_SEND source_space enum is invalid");
    ValidateCompletion(operands.completion, operands.token);
    ValidateEndpointDataType(operands.datatype);
    ValidateReduceOperator(operands.reduce_op, false);
    Require(operands.datatype == EndpointDataType::UINT8,
            "DTE_SEND non-reduction datatype must be UINT8");
    RequireU32(operands.fsm_id, "DTE_SEND fsm_id");
    Require(operands.fsm_id != 0, "DTE_SEND fsm_id must be non-zero");
    Require(operands.length_bytes != 0,
            "DTE_SEND length_bytes must be non-zero");
    ValidateAddress(operands.source, false, "DTE_SEND source");
    if (operands.source_space == EndpointSourceSpace::HBM)
        Require(operands.source.kind == SramAddressKind::ABSOLUTE,
                "DTE_SEND HBM source must use an absolute address");
    RequireU16(operands.peer_core, "DTE_SEND peer_core");
    RequireU16(operands.expected_sources, "DTE_SEND expected_sources");
    RequireU16(operands.tree_id, "DTE_SEND tree_id");
    Require(operands.expected_sources == 0,
            "DTE_SEND expected_sources is receive-only");
    switch (operands.mode) {
    case DteSendMode::P2P:
        Require(operands.tree_id == 0 && operands.expected_sources == 0,
                "DTE_SEND UNICAST carries invalid tree/source-count metadata");
        if (operands.group_id == 0) {
            Require(operands.length_bytes <= kDteEndpointP2pMaxBytes,
                    "DTE_SEND P2P length_bytes exceeds transport maximum");
            Require(operands.collective_id == 0 && operands.epoch == 0,
                    "DTE_SEND standalone P2P carries a partial collective key");
        } else {
            Require(operands.peer_core == 0,
                    "DTE_SEND keyed UNICAST peer is derived from the collective graph");
            ValidateCollectiveKey(operands.group_id, operands.collective_id,
                                  operands.epoch);
        }
        break;
    case DteSendMode::SCATTER:
        Require(operands.peer_core == 0 && operands.tree_id == 0,
                "DTE_SEND SCATTER carries peer/tree metadata");
        ValidateCollectiveKey(operands.group_id, operands.collective_id,
                              operands.epoch);
        break;
    case DteSendMode::BROADCAST:
        Require(operands.peer_core == 0 && operands.tree_id == 0,
                "DTE_SEND BROADCAST baseline requires no peer and tree_id=0");
        ValidateCollectiveKey(operands.group_id, operands.collective_id,
                              operands.epoch);
        break;
    }
}

void ValidateDteRecv(const DteRecvOperands &operands) {
    if (EnumByte(operands.mode) > EnumByte(DteRecvMode::REDUCE))
        Fail("DTE_RECV mode enum is invalid");
    ValidateCompletion(operands.completion, operands.token);
    ValidateEndpointDataType(operands.datatype);
    RequireU32(operands.fsm_id, "DTE_RECV fsm_id");
    Require(operands.fsm_id != 0, "DTE_RECV fsm_id must be non-zero");
    Require(operands.length_bytes != 0,
            "DTE_RECV length_bytes must be non-zero");
    ValidateAddress(operands.destination, false, "DTE_RECV destination");
    RequireU16(operands.peer_core, "DTE_RECV peer_core");
    RequireU16(operands.expected_sources, "DTE_RECV expected_sources");
    RequireU16(operands.tree_id, "DTE_RECV tree_id");
    Require(operands.tree_id == 0, "DTE_RECV must not carry a tree_id");
    switch (operands.mode) {
    case DteRecvMode::P2P:
        ValidateReduceOperator(operands.reduce_op, false);
        Require(operands.datatype == EndpointDataType::UINT8,
                "DTE_RECV P2P datatype must be UINT8");
        Require(operands.expected_sources == 0,
                "DTE_RECV UNICAST carries expected_sources");
        if (operands.group_id == 0) {
            Require(operands.length_bytes <= kDteEndpointP2pMaxBytes,
                    "DTE_RECV P2P length_bytes exceeds transport maximum");
            Require(operands.collective_id == 0 && operands.epoch == 0,
                    "DTE_RECV standalone P2P carries a partial collective key");
        } else {
            Require(operands.peer_core == 0,
                    "DTE_RECV keyed UNICAST peer is derived from the collective graph");
            ValidateCollectiveKey(operands.group_id, operands.collective_id,
                                  operands.epoch);
        }
        break;
    case DteRecvMode::GATHER:
        ValidateReduceOperator(operands.reduce_op, false);
        Require(operands.datatype == EndpointDataType::UINT8,
                "DTE_RECV GATHER datatype must be UINT8");
        Require(operands.peer_core == 0,
                "DTE_RECV GATHER must not carry a peer");
        ValidateCollectiveKey(operands.group_id, operands.collective_id,
                              operands.epoch);
        break;
    case DteRecvMode::REDUCE:
        ValidateReduceOperator(operands.reduce_op, true);
        Require(operands.peer_core == 0,
                "DTE_RECV REDUCE must not carry a peer");
        ValidateCollectiveKey(operands.group_id, operands.collective_id,
                              operands.epoch);
        break;
    }
}

void ValidateReduceCompute(const ReduceComputeOperands &operands) {
    ValidateEndpointDataType(operands.datatype);
    ValidateReduceOperator(operands.reduce_op, true);
    ValidateCollectiveKey(operands.group_id, operands.collective_id,
                          operands.epoch);
    RequireU16(operands.root_rank, "REDUCE_COMPUTE root_rank");
    RequireU16(operands.self_rank, "REDUCE_COMPUTE self_rank");
    Require(operands.element_count != 0,
            "REDUCE_COMPUTE element_count must be non-zero");
    const uint64_t bits = operands.datatype == EndpointDataType::UINT8
                              ? 8
                              : operands.datatype == EndpointDataType::INT32
                                    ? 32
                                    : 64;
    Require(operands.element_count <=
                std::numeric_limits<uint64_t>::max() / bits,
            "REDUCE_COMPUTE element_count*dtype_bits overflows u64");
    ValidateAddress(operands.source, false, "REDUCE_COMPUTE source");
    ValidateAddress(operands.destination, false,
                    "REDUCE_COMPUTE destination");
}

void ValidateLsu(const LsuOperands &operands) {
    Require(operands.size_bytes != 0, "LSU size_bytes must be non-zero");
    ValidateAddress(operands.sram, false, "LSU SRAM address");
}

void ValidateDteIssue(const DteIssueOperands &operands) {
    if (EnumByte(operands.direction) >
        EnumByte(LocalDteDirection::DRAM_TO_SPM))
        Fail("DTE_ISSUE direction enum is invalid");
    RequireU32(operands.token, "DTE_ISSUE token");
    Require(operands.token != 0, "DTE_ISSUE token must be non-zero");
    Require(operands.payload_bits != 0,
            "DTE_ISSUE payload_bits must be non-zero");
    Require(operands.size_bytes != 0,
            "DTE_ISSUE size_bytes must be non-zero");
    Require(operands.size_bytes <=
                std::numeric_limits<uint64_t>::max() / 8,
            "DTE_ISSUE size_bytes*8 overflows u64");
    Require(operands.payload_bits == operands.size_bytes * 8,
            "DTE_ISSUE payload_bits must equal size_bytes*8");
    switch (operands.direction) {
    case LocalDteDirection::SPM_TO_SPM:
        ValidateAddress(operands.source_sram, false,
                        "DTE_ISSUE source SRAM");
        ValidateAddress(operands.destination_sram, false,
                        "DTE_ISSUE destination SRAM");
        Require(operands.hbm_address_bytes == 0,
                "SPM_TO_SPM must not carry an HBM address");
        break;
    case LocalDteDirection::SPM_TO_DRAM:
        ValidateAddress(operands.source_sram, false,
                        "DTE_ISSUE source SRAM");
        ValidateAddress(operands.destination_sram, true,
                        "DTE_ISSUE destination SRAM");
        Require(IsNone(operands.destination_sram),
                "SPM_TO_DRAM must not carry a destination SRAM address");
        break;
    case LocalDteDirection::DRAM_TO_SPM:
        ValidateAddress(operands.source_sram, true,
                        "DTE_ISSUE source SRAM");
        Require(IsNone(operands.source_sram),
                "DRAM_TO_SPM must not carry a source SRAM address");
        ValidateAddress(operands.destination_sram, false,
                        "DTE_ISSUE destination SRAM");
        break;
    }
}

uint64_t CheckedAdd(uint64_t left, uint64_t right,
                    std::string_view field) {
    if (left > std::numeric_limits<uint64_t>::max() - right)
        Fail("compute derived value overflows u64: " + std::string(field));
    return left + right;
}

uint64_t CheckedMul(uint64_t left, uint64_t right,
                    std::string_view field) {
    if (left != 0 &&
        right > std::numeric_limits<uint64_t>::max() / left)
        Fail("compute derived value overflows u64: " + std::string(field));
    return left * right;
}

uint64_t Product(std::initializer_list<uint64_t> factors,
                 std::string_view field) {
    uint64_t value = 1;
    for (uint64_t factor : factors)
        value = CheckedMul(value, factor, field);
    return value;
}

uint64_t ComputeParameter(const ComputeOperands &operands,
                          const RecordSchema &schema,
                          std::string_view name) {
    for (std::size_t i = 0; i < schema.parameter_count; ++i)
        if (schema.parameter_names[i] == name)
            return operands.parameters[i];
    Fail("compute schema is missing parameter " + std::string(name));
}

void RequirePositive(uint64_t value, std::string_view field) {
    Require(value != 0, std::string(field) + " must be non-zero");
}

void RequireFitsInt(uint64_t value, std::string_view field) {
    Require(value <=
                static_cast<uint64_t>(std::numeric_limits<int>::max()),
            std::string(field) + " exceeds existing int runtime storage");
}

void ValidateNpuLayout(const ComputeOperands &operands,
                       std::initializer_list<uint64_t> inputs,
                       std::initializer_list<uint64_t> chunks,
                       std::string_view opcode_name) {
    const uint64_t data_bytes =
        operands.datatype == ExternalDataType::INT8 ? 1 : 2;
    uint64_t input_elements = 0;
    for (uint64_t value : inputs) {
        RequirePositive(value, std::string(opcode_name) + " input tensor");
        RequireFitsInt(value, std::string(opcode_name) + " input tensor");
        RequireFitsInt(CheckedMul(value, data_bytes, opcode_name),
                       std::string(opcode_name) + " input bytes");
        input_elements = CheckedAdd(input_elements, value, opcode_name);
    }
    RequireFitsInt(input_elements,
                   std::string(opcode_name) + " total input elements");
    RequireFitsInt(CheckedMul(input_elements, data_bytes, opcode_name),
                   std::string(opcode_name) + " total input bytes");

    uint64_t chunk_end = operands.data_offset_bytes;
    for (uint64_t value : chunks) {
        RequirePositive(value, std::string(opcode_name) + " data chunk");
        RequireFitsInt(value, std::string(opcode_name) + " data chunk");
        const uint64_t bytes = CheckedMul(value, data_bytes, opcode_name);
        RequireFitsInt(bytes, std::string(opcode_name) + " chunk bytes");
        chunk_end = CheckedAdd(chunk_end, bytes, opcode_name);
        RequireFitsInt(chunk_end,
                       std::string(opcode_name) + " chunk address end");
    }
}

uint64_t PositiveParameter(const ComputeOperands &operands,
                           const RecordSchema &schema,
                           std::string_view name) {
    const uint64_t value = ComputeParameter(operands, schema, name);
    RequirePositive(value, std::string(schema.parameter_names == nullptr
                                           ? "compute"
                                           : "compute parameter") +
                               " " + std::string(name));
    return value;
}

uint64_t ConvOutputExtent(uint64_t input, uint64_t padding,
                          uint64_t kernel, uint64_t stride,
                          std::string_view field) {
    const uint64_t padded = CheckedAdd(
        input, CheckedMul(2, padding, field), field);
    RequireFitsInt(padded, std::string(field) + " padded extent");
    Require(padded >= kernel,
            std::string(field) + " kernel exceeds padded input");
    const uint64_t output = (padded - kernel) / stride + 1;
    RequirePositive(output, std::string(field) + " output extent");
    RequireFitsInt(output, std::string(field) + " output extent");
    return output;
}

void ValidatePublishedComputeSemantics(const ComputeOperands &operands,
                                       const RecordSchema &schema) {
    auto p = [&](std::string_view name) {
        return PositiveParameter(operands, schema, name);
    };
    switch (schema.opcode) {
    case Opcode::MATMUL: {
        const uint64_t b = p("B"), t = p("T"), c = p("C"), oc = p("OC");
        const uint64_t input = Product({b, t, c}, "MATMUL input");
        const uint64_t output = Product({b, t, oc}, "MATMUL output");
        ValidateNpuLayout(operands, {input},
                          {CheckedMul(c, oc, "MATMUL weight"), oc, output},
                          "MATMUL");
        (void)Product({b, oc, t, c, 2}, "MATMUL vec_ops");
        return;
    }
    case Opcode::CONV:
    case Opcode::MAXPOOL: {
        const bool convolution = schema.opcode == Opcode::CONV;
        const uint64_t b = p("B"), w = p("W"), h = p("H"), c = p("C");
        const uint64_t px = ComputeParameter(operands, schema, "pX");
        const uint64_t py = ComputeParameter(operands, schema, "pY");
        const uint64_t sx = p("sX"), sy = p("sY");
        const uint64_t kx = p("kX"), ky = p("kY");
        const uint64_t ow = ConvOutputExtent(w, px, kx, sx, "compute W");
        const uint64_t oh = ConvOutputExtent(h, py, ky, sy, "compute H");
        const uint64_t input = Product({b, c, h, w}, "compute input");
        if (convolution) {
            const uint64_t f = p("F");
            const uint64_t weight =
                Product({f, c, ky, kx}, "CONV weight");
            const uint64_t output =
                Product({b, f, oh, ow}, "CONV output");
            ValidateNpuLayout(operands, {input}, {weight, f, output},
                              "CONV");
            (void)Product({b, c, ky, kx, 2, oh, ow, f},
                          "CONV exu_ops");
        } else {
            const uint64_t output =
                Product({b, c, oh, ow}, "MAXPOOL output");
            ValidateNpuLayout(operands, {input}, {output}, "MAXPOOL");
            (void)Product({b, c, oh, ow, kx, ky},
                          "MAXPOOL sfu_ops");
        }
        return;
    }
    case Opcode::ATTENTION: {
        const uint64_t b = p("B"), t = p("T"), c = p("C"), nh = p("NH");
        const uint64_t r = p("R");
        const uint64_t input = Product({b, t, c}, "ATTENTION input");
        const uint64_t temporary =
            Product({b, nh, t, t}, "ATTENTION temporary");
        const uint64_t divisor = CheckedAdd(1, 2 / r, "ATTENTION output");
        const uint64_t output = input / divisor;
        RequirePositive(output, "ATTENTION output tensor");
        ValidateNpuLayout(operands, {input}, {temporary, temporary, output},
                          "ATTENTION");
        (void)Product({b, c, t, t, 4}, "ATTENTION exu_ops");
        (void)Product({b, nh, t, t}, "ATTENTION sfu_ops");
        (void)Product({b, nh, t, t, 2}, "ATTENTION vec_ops");
        return;
    }
    case Opcode::GATE: {
        const uint64_t b = p("B"), t = p("T"), c = p("C");
        const uint64_t experts = p("E_N"), k = p("K");
        Require(k <= experts, "GATE K must not exceed E_N");
        const uint64_t input = Product({b, t, c}, "GATE input");
        const uint64_t output = Product({b, t, k}, "GATE output");
        ValidateNpuLayout(operands, {input}, {output}, "GATE");
        (void)Product({b, t, c, experts}, "GATE exu_ops");
        return;
    }
    case Opcode::MOE_MATMUL: {
        const uint64_t b = p("B"), t = p("T"), c = p("C"), oc = p("OC");
        const uint64_t k = p("K"), experts = p("E_N");
        Require(k <= experts, "MOE_MATMUL K must not exceed E_N");
        const uint64_t is_merge =
            ComputeParameter(operands, schema, "is_merge");
        const uint64_t need_choose =
            ComputeParameter(operands, schema, "need_choose");
        Require(is_merge <= 1, "MOE_MATMUL is_merge must be 0 or 1");
        Require(need_choose <= 1,
                "MOE_MATMUL need_choose must be 0 or 1");
        const uint64_t base_input = Product({b, t, c}, "MOE_MATMUL input");
        const uint64_t input = is_merge
                                   ? CheckedMul(base_input, k,
                                                "MOE_MATMUL merged input")
                                   : base_input;
        const uint64_t base_output =
            Product({b, t, oc}, "MOE_MATMUL output");
        const uint64_t output = is_merge
                                    ? base_output
                                    : CheckedMul(base_output, k,
                                                 "MOE_MATMUL split output");
        ValidateNpuLayout(
            operands, {input},
            {CheckedMul(oc, c, "MOE_MATMUL weight"), oc, output},
            "MOE_MATMUL");
        const uint64_t gemm =
            Product({b, t, c, oc, k, 2}, "MOE_MATMUL exu_ops");
        if (is_merge)
            (void)CheckedAdd(gemm, Product({b, t, oc, k},
                                           "MOE_MATMUL merge ops"),
                             "MOE_MATMUL exu_ops");
        return;
    }
    case Opcode::GELU:
    case Opcode::SILU:
    case Opcode::RELU: {
        const uint64_t n = p("N");
        ValidateNpuLayout(operands, {n}, {n}, "unary compute");
        const uint64_t scale = schema.opcode == Opcode::GELU ? 4 :
                               schema.opcode == Opcode::SILU ? 3 : 1;
        (void)CheckedMul(n, scale, "unary compute ops");
        return;
    }
    case Opcode::SWIGLU:
    case Opcode::RESIDUAL: {
        const uint64_t n = p("N");
        ValidateNpuLayout(operands, {n, n}, {n},
                          schema.opcode == Opcode::SWIGLU ? "SWIGLU"
                                                         : "RESIDUAL");
        (void)CheckedMul(n, schema.opcode == Opcode::SWIGLU ? 4 : 1,
                         "binary compute ops");
        return;
    }
    case Opcode::LAYERNORM:
    case Opcode::RMSNORM: {
        const uint64_t b = p("B"), t = p("T"), c = p("C");
        const uint64_t tensor = Product({b, t, c}, "normalization tensor");
        if (schema.opcode == Opcode::LAYERNORM)
            ValidateNpuLayout(operands, {tensor}, {c, c, tensor},
                              "LAYERNORM");
        else
            ValidateNpuLayout(operands, {tensor}, {c, tensor}, "RMSNORM");
        const uint64_t scale = CheckedAdd(
            CheckedMul(schema.opcode == Opcode::LAYERNORM ? 8 : 4, c,
                       "normalization vec scale"),
            schema.opcode == Opcode::LAYERNORM ? 3 : 1,
            "normalization vec scale");
        (void)Product({b, t, scale}, "normalization vec_ops");
        return;
    }
    case Opcode::ROPE: {
        const uint64_t b = p("B"), t = p("T"), c = p("C"), nh = p("NH");
        const uint64_t head = c / nh;
        RequirePositive(head, "ROPE C/NH");
        const uint64_t tensor = Product({b, t, c}, "ROPE tensor");
        const uint64_t sincos = Product({b, head, 2, t}, "ROPE sincos");
        ValidateNpuLayout(operands, {tensor}, {sincos, tensor}, "ROPE");
        (void)Product({3, t, c}, "ROPE vec_ops");
        return;
    }
    case Opcode::SPLIT_MATMUL:
    case Opcode::MERGE_MATMUL: {
        const uint64_t b = p("B"), t = p("T"), c = p("C");
        const uint64_t dim = p("dim"), slice = p("slice");
        Require(dim == 1 || dim == 2,
                "split/merge MATMUL dim must be 1 or 2");
        Require(slice <= kSramBindInputLimit,
                "split/merge MATMUL slice exceeds 16 inputs");
        const uint64_t tensor = Product({b, t, c}, "split/merge tensor");
        if (schema.opcode == Opcode::SPLIT_MATMUL) {
            ValidateNpuLayout(operands, {tensor}, {tensor},
                              "SPLIT_MATMUL");
        } else {
            const uint64_t total_input =
                CheckedMul(tensor, slice, "MERGE_MATMUL inputs");
            const uint64_t output = dim == 1
                                        ? tensor
                                        : CheckedMul(tensor, slice,
                                                     "MERGE_MATMUL output");
            ValidateNpuLayout(operands, {total_input}, {output},
                              "MERGE_MATMUL");
            (void)tensor;
        }
        return;
    }
    case Opcode::DUMMY:
        ValidateNpuLayout(operands, {80}, {80}, "DUMMY");
        return;
    default:
        return;
    }
}

void ValidateCompute(const ComputeOperands &operands,
                     const RecordSchema &schema) {
    if (EnumByte(operands.datatype) > EnumByte(ExternalDataType::FP16))
        Fail("compute datatype enum is invalid");
    RequireU16(operands.input_offset_bytes, "compute input_offset_bytes");
    RequireU16(operands.data_offset_bytes, "compute data_offset_bytes");
    RequireU16(operands.output_offset_bytes, "compute output_offset_bytes");
    Require(operands.parameters.size() == schema.parameter_count,
            "compute parameter count does not match opcode schema");
    for (std::size_t i = 0; i < operands.parameters.size(); ++i) {
        if (operands.parameters[i] > kExternalNpuParameterMax) {
            const std::string name = schema.parameter_names == nullptr
                                         ? std::to_string(i)
                                         : std::string(schema.parameter_names[i]);
            Fail("compute parameter exceeds 30 bits: " + name);
        }
    }
    ValidatePublishedComputeSemantics(operands, schema);
}

void ValidateSymbol(const SymbolOperands &operands) {
    RequireU32(operands.symbol_index, "symbol_index");
}

void ValidateSramBind(const SramBindOperands &operands) {
    Require(operands.input_count >= 1 &&
                operands.input_count <= kSramBindInputLimit,
            "SRAM_BIND input_count must be in [1,16]");
    for (std::size_t i = 0; i < kSramBindInputLimit; ++i) {
        RequireU32(operands.input_symbol_indices[i],
                   "SRAM_BIND input_symbol_indices[" +
                       std::to_string(i) + "]");
        if (i >= operands.input_count)
            Require(operands.input_symbol_indices[i] == 0,
                    "SRAM_BIND unused input symbol slots must be zero");
    }
    RequireU32(operands.output_symbol_index,
               "SRAM_BIND output_symbol_index");
}

void ValidateSramAlloc(const SramAllocOperands &operands) {
    RequireU32(operands.region_name_string_index,
               "SRAM_ALLOC region_name_string_index");
    RequireU32(operands.label_symbol_index,
               "SRAM_ALLOC label_symbol_index");
    Require(operands.size_bytes != 0,
            "SRAM_ALLOC size_bytes must be non-zero");
    Require(operands.alignment_bytes != 0 &&
                (operands.alignment_bytes & (operands.alignment_bytes - 1)) == 0,
            "SRAM_ALLOC alignment_bytes must be a non-zero power of two");
    if (EnumByte(operands.lifetime) > EnumByte(SramLifetime::PERSISTENT))
        Fail("SRAM_ALLOC lifetime enum is invalid");
}

void ValidateSramResize(const SramResizeOperands &operands) {
    RequireU32(operands.symbol_index, "SRAM_RESIZE symbol_index");
    Require(operands.new_size_bytes != 0,
            "SRAM_RESIZE new_size_bytes must be non-zero");
}

void ValidateSramRename(const SramRenameOperands &operands) {
    RequireU32(operands.old_symbol_index, "SRAM_RENAME old_symbol_index");
    RequireU32(operands.new_symbol_index, "SRAM_RENAME new_symbol_index");
    Require(operands.old_symbol_index != operands.new_symbol_index,
            "SRAM_RENAME source and destination symbols must differ");
}

void ValidateToken(const TokenOperands &operands) {
    RequireU32(operands.token, "DTE token");
    Require(operands.token != 0, "DTE token must be non-zero");
}

void ValidateEventSet(const EventSetOperands &operands) {
    RequireU16(operands.source_core, "EVENT_SET source_core");
    RequireU16(operands.destination_core, "EVENT_SET destination_core");
    RequireU32(operands.tag, "EVENT_SET tag");
}

void ValidateEventWait(const EventWaitOperands &operands) {
    RequireU16(operands.source_core, "EVENT_WAIT source_core");
    RequireU16(operands.destination_core, "EVENT_WAIT destination_core");
    RequireU32(operands.tag, "EVENT_WAIT tag");
    RequireU32(operands.count, "EVENT_WAIT count");
    Require(operands.count != 0, "EVENT_WAIT count must be non-zero");
}

void ValidateGroupSync(const GroupSyncOperands &operands) {
    RequireU32(operands.group_id, "GROUP_SYNC group_id");
    RequireU32(operands.sync_seq, "GROUP_SYNC sync_seq");
    Require(operands.group_id != 0, "GROUP_SYNC group_id must be non-zero");
}

constexpr std::array<std::string_view, 4> kMatmul{{"B", "T", "C", "OC"}};
constexpr std::array<std::string_view, 6> kMatmulMla{{"B", "T", "C", "OC", "NH", "DH"}};
constexpr std::array<std::string_view, 6> kMatmulPd{{"B", "T", "C", "OC", "R", "chunk"}};
constexpr std::array<std::string_view, 11> kConv{{"B", "W", "H", "C", "pX", "pY", "sX", "sY", "kX", "kY", "F"}};
constexpr std::array<std::string_view, 10> kMaxPool{{"B", "W", "H", "C", "pX", "pY", "sX", "sY", "kX", "kY"}};
constexpr std::array<std::string_view, 5> kAttention{{"B", "T", "C", "NH", "R"}};
constexpr std::array<std::string_view, 6> kAttentionPd{{"B", "T", "C", "NH", "DH", "R"}};
constexpr std::array<std::string_view, 5> kGate{{"B", "T", "C", "E_N", "K"}};
constexpr std::array<std::string_view, 8> kMoeMatmul{{"B", "T", "C", "OC", "K", "E_N", "is_merge", "need_choose"}};
constexpr std::array<std::string_view, 1> kN{{"N"}};
constexpr std::array<std::string_view, 3> kBtc{{"B", "T", "C"}};
constexpr std::array<std::string_view, 4> kRope{{"B", "T", "C", "NH"}};
constexpr std::array<std::string_view, 5> kRopePd{{"B", "T", "C", "NH", "R"}};
constexpr std::array<std::string_view, 5> kSplitMergeMatmul{{"B", "T", "C", "dim", "slice"}};
constexpr std::array<std::string_view, 0> kDummy{};
constexpr std::array<std::string_view, 4> kBatchnorm{{"B", "W", "H", "C"}};
constexpr std::array<std::string_view, 9> kSplitConv{{"W", "H", "C", "B", "pX", "pY", "S", "K", "slice"}};
constexpr std::array<std::string_view, 5> kMergeConv{{"B", "T", "C", "dim", "slice"}};
constexpr std::array<std::string_view, 8> kGemmReduceScatter{{"mode", "chunk", "tile_bytes", "comm_bytes", "compute_cycles", "reduce_cycles", "hbm_base", "participants"}};

template <std::size_t N>
constexpr RecordSchema ComputeSchema(
    Opcode opcode, const std::array<std::string_view, N> &names) {
    return {opcode, RecordOperandKind::COMPUTE,
            static_cast<uint32_t>(kComputePrefixSize + 4 * N), names.data(), N};
}

constexpr RecordSchema FixedSchema(Opcode opcode, RecordOperandKind kind,
                                   uint32_t payload_size) {
    return {opcode, kind, payload_size, nullptr, 0};
}

constexpr std::array<RecordSchema, kOpcodeManifestSize> kSchemas{{
    ComputeSchema(Opcode::MATMUL, kMatmul),
    ComputeSchema(Opcode::MATMUL_MLA, kMatmulMla),
    ComputeSchema(Opcode::MATMUL_PD, kMatmulPd),
    ComputeSchema(Opcode::CONV, kConv),
    ComputeSchema(Opcode::MAXPOOL, kMaxPool),
    ComputeSchema(Opcode::ATTENTION, kAttention),
    ComputeSchema(Opcode::ATTENTION_PD, kAttentionPd),
    ComputeSchema(Opcode::GATE, kGate),
    ComputeSchema(Opcode::MOE_MATMUL, kMoeMatmul),
    ComputeSchema(Opcode::GELU, kN),
    ComputeSchema(Opcode::SILU, kN),
    ComputeSchema(Opcode::SWIGLU, kN),
    ComputeSchema(Opcode::RELU, kN),
    ComputeSchema(Opcode::RESIDUAL, kN),
    ComputeSchema(Opcode::LAYERNORM, kBtc),
    ComputeSchema(Opcode::RMSNORM, kBtc),
    ComputeSchema(Opcode::ROPE, kRope),
    ComputeSchema(Opcode::ROPE_PD, kRopePd),
    ComputeSchema(Opcode::SPLIT_MATMUL, kSplitMergeMatmul),
    ComputeSchema(Opcode::MERGE_MATMUL, kSplitMergeMatmul),
    ComputeSchema(Opcode::DUMMY, kDummy),
    ComputeSchema(Opcode::BATCHNORM, kBatchnorm),
    ComputeSchema(Opcode::SPLIT_CONV, kSplitConv),
    ComputeSchema(Opcode::MERGE_CONV, kMergeConv),
    ComputeSchema(Opcode::GEMM_REDUCE_SCATTER, kGemmReduceScatter),
    FixedSchema(Opcode::DTE_SEND, RecordOperandKind::DTE_SEND,
                kEndpointPayloadSize),
    FixedSchema(Opcode::DTE_RECV, RecordOperandKind::DTE_RECV,
                kEndpointPayloadSize),
    FixedSchema(Opcode::REDUCE_COMPUTE, RecordOperandKind::REDUCE_COMPUTE,
                kReducePayloadSize),
    FixedSchema(Opcode::LSU_LOAD, RecordOperandKind::LSU, kLsuPayloadSize),
    FixedSchema(Opcode::LSU_STORE, RecordOperandKind::LSU, kLsuPayloadSize),
    FixedSchema(Opcode::DTE_ISSUE, RecordOperandKind::DTE_ISSUE,
                kDteIssuePayloadSize),
    FixedSchema(Opcode::SRAM_CLEAR, RecordOperandKind::SYMBOL,
                kSymbolPayloadSize),
    FixedSchema(Opcode::SRAM_BIND, RecordOperandKind::SRAM_BIND,
                kSramBindPayloadSize),
    FixedSchema(Opcode::SRAM_ALLOC, RecordOperandKind::SRAM_ALLOC,
                kSramAllocPayloadSize),
    FixedSchema(Opcode::SRAM_FREE, RecordOperandKind::SYMBOL,
                kSymbolPayloadSize),
    FixedSchema(Opcode::SRAM_RESIZE, RecordOperandKind::SRAM_RESIZE,
                kSramResizePayloadSize),
    FixedSchema(Opcode::SRAM_RENAME, RecordOperandKind::SRAM_RENAME,
                kSramRenamePayloadSize),
    FixedSchema(Opcode::DTE_WAIT, RecordOperandKind::TOKEN,
                kTokenPayloadSize),
    FixedSchema(Opcode::DTE_FENCE, RecordOperandKind::NONE, 0),
    FixedSchema(Opcode::DTE_CANCEL, RecordOperandKind::TOKEN,
                kTokenPayloadSize),
    FixedSchema(Opcode::EVENT_SET, RecordOperandKind::EVENT_SET,
                kEventSetPayloadSize),
    FixedSchema(Opcode::EVENT_WAIT, RecordOperandKind::EVENT_WAIT,
                kEventWaitPayloadSize),
    FixedSchema(Opcode::GROUP_SYNC, RecordOperandKind::GROUP_SYNC,
                kGroupSyncPayloadSize),
    FixedSchema(Opcode::DTE_POLL, RecordOperandKind::TOKEN,
                kTokenPayloadSize),
}};

void ValidateOpcodeForRecord(Opcode opcode, uint64_t enabled_capabilities) {
    const uint8_t raw = OpcodeValue(opcode);
    if (raw == OpcodeValue(Opcode::INVALID))
        Fail("external record opcode INVALID is forbidden");
    switch (ValidateOpcode(raw, enabled_capabilities)) {
    case OpcodeValidation::AVAILABLE: return;
    case OpcodeValidation::DEPRECATED:
        Fail("external record opcode is deprecated: " + HexByte(raw));
    case OpcodeValidation::GATED:
        Fail("external record opcode capability is disabled: " + HexByte(raw));
    case OpcodeValidation::UNSUPPORTED:
        Fail("external record opcode is unsupported: " + HexByte(raw));
    case OpcodeValidation::RESERVED:
        Fail("external record opcode is reserved: " + HexByte(raw));
    case OpcodeValidation::INTERNAL:
        Fail("external record opcode is internal: " + HexByte(raw));
    case OpcodeValidation::UNKNOWN:
        Fail("external record opcode is unknown: " + HexByte(raw));
    }
    Fail("external record opcode validation failed");
}

void ValidateOperandsForSchema(const ExternalRecord &record,
                               const RecordSchema &schema) {
    switch (schema.operand_kind) {
    case RecordOperandKind::COMPUTE:
        ValidateCompute(RequireOperands<ComputeOperands>(record, "compute"),
                        schema);
        return;
    case RecordOperandKind::DTE_SEND:
        ValidateDteSend(RequireOperands<DteSendOperands>(record, "DTE_SEND"));
        return;
    case RecordOperandKind::DTE_RECV:
        ValidateDteRecv(RequireOperands<DteRecvOperands>(record, "DTE_RECV"));
        return;
    case RecordOperandKind::REDUCE_COMPUTE:
        ValidateReduceCompute(RequireOperands<ReduceComputeOperands>(
            record, "REDUCE_COMPUTE"));
        return;
    case RecordOperandKind::LSU:
        ValidateLsu(RequireOperands<LsuOperands>(record, "LSU"));
        return;
    case RecordOperandKind::DTE_ISSUE:
        ValidateDteIssue(
            RequireOperands<DteIssueOperands>(record, "DTE_ISSUE"));
        return;
    case RecordOperandKind::SYMBOL:
        ValidateSymbol(RequireOperands<SymbolOperands>(record, "symbol"));
        return;
    case RecordOperandKind::SRAM_BIND:
        ValidateSramBind(
            RequireOperands<SramBindOperands>(record, "SRAM_BIND"));
        return;
    case RecordOperandKind::SRAM_ALLOC:
        ValidateSramAlloc(
            RequireOperands<SramAllocOperands>(record, "SRAM_ALLOC"));
        return;
    case RecordOperandKind::SRAM_RESIZE:
        ValidateSramResize(
            RequireOperands<SramResizeOperands>(record, "SRAM_RESIZE"));
        return;
    case RecordOperandKind::SRAM_RENAME:
        ValidateSramRename(
            RequireOperands<SramRenameOperands>(record, "SRAM_RENAME"));
        return;
    case RecordOperandKind::TOKEN:
        ValidateToken(RequireOperands<TokenOperands>(record, "token"));
        return;
    case RecordOperandKind::NONE:
        (void)RequireOperands<NoOperands>(record, "no operands");
        return;
    case RecordOperandKind::EVENT_SET:
        ValidateEventSet(
            RequireOperands<EventSetOperands>(record, "EVENT_SET"));
        return;
    case RecordOperandKind::EVENT_WAIT:
        ValidateEventWait(
            RequireOperands<EventWaitOperands>(record, "EVENT_WAIT"));
        return;
    case RecordOperandKind::GROUP_SYNC:
        ValidateGroupSync(
            RequireOperands<GroupSyncOperands>(record, "GROUP_SYNC"));
        return;
    }
    Fail("external record has an unknown operand schema");
}

std::vector<uint8_t> EncodePayload(const ExternalRecord &record,
                                   const RecordSchema &schema) {
    std::vector<uint8_t> payload;
    payload.reserve(schema.payload_size);
    switch (schema.operand_kind) {
    case RecordOperandKind::COMPUTE: {
        const auto &o = std::get<ComputeOperands>(record.operands);
        payload.push_back(EnumByte(o.datatype));
        payload.push_back(0);
        AppendLittleEndian(payload, o.input_offset_bytes, 2);
        AppendLittleEndian(payload, o.data_offset_bytes, 2);
        AppendLittleEndian(payload, o.output_offset_bytes, 2);
        for (uint64_t parameter : o.parameters)
            AppendLittleEndian(payload, parameter, 4);
        break;
    }
    case RecordOperandKind::DTE_SEND: {
        const auto &o = std::get<DteSendOperands>(record.operands);
        payload.push_back(EnumByte(o.mode));
        payload.push_back(EnumByte(o.completion));
        payload.push_back(EnumByte(o.datatype));
        payload.push_back(EnumByte(o.reduce_op));
        AppendLittleEndian(payload, o.fsm_id, 4);
        AppendLittleEndian(payload, o.token, 4);
        payload.push_back(EnumByte(o.source_space));
        AppendLittleEndian(payload, 0, 3);
        AppendLittleEndian(payload, o.length_bytes, 8);
        EncodeAddress(payload, o.source);
        AppendLittleEndian(payload, o.peer_core, 2);
        AppendLittleEndian(payload, o.expected_sources, 2);
        AppendLittleEndian(payload, o.tree_id, 2);
        AppendLittleEndian(payload, 0, 2);
        AppendLittleEndian(payload, o.group_id, 4);
        AppendLittleEndian(payload, o.collective_id, 4);
        AppendLittleEndian(payload, o.epoch, 4);
        AppendLittleEndian(payload, 0, 4);
        break;
    }
    case RecordOperandKind::DTE_RECV: {
        const auto &o = std::get<DteRecvOperands>(record.operands);
        payload.push_back(EnumByte(o.mode));
        payload.push_back(EnumByte(o.completion));
        payload.push_back(EnumByte(o.datatype));
        payload.push_back(EnumByte(o.reduce_op));
        AppendLittleEndian(payload, o.fsm_id, 4);
        AppendLittleEndian(payload, o.token, 4);
        AppendLittleEndian(payload, 0, 4);
        AppendLittleEndian(payload, o.length_bytes, 8);
        EncodeAddress(payload, o.destination);
        AppendLittleEndian(payload, o.peer_core, 2);
        AppendLittleEndian(payload, o.expected_sources, 2);
        AppendLittleEndian(payload, o.tree_id, 2);
        AppendLittleEndian(payload, 0, 2);
        AppendLittleEndian(payload, o.group_id, 4);
        AppendLittleEndian(payload, o.collective_id, 4);
        AppendLittleEndian(payload, o.epoch, 4);
        AppendLittleEndian(payload, 0, 4);
        break;
    }
    case RecordOperandKind::REDUCE_COMPUTE: {
        const auto &o = std::get<ReduceComputeOperands>(record.operands);
        payload.push_back(EnumByte(o.datatype));
        payload.push_back(EnumByte(o.reduce_op));
        AppendLittleEndian(payload, 0, 2);
        AppendLittleEndian(payload, o.group_id, 4);
        AppendLittleEndian(payload, o.collective_id, 4);
        AppendLittleEndian(payload, o.epoch, 4);
        AppendLittleEndian(payload, o.root_rank, 2);
        AppendLittleEndian(payload, o.self_rank, 2);
        AppendLittleEndian(payload, 0, 4);
        AppendLittleEndian(payload, o.element_count, 8);
        EncodeAddress(payload, o.source);
        EncodeAddress(payload, o.destination);
        break;
    }
    case RecordOperandKind::LSU: {
        const auto &o = std::get<LsuOperands>(record.operands);
        AppendLittleEndian(payload, o.hbm_address_bytes, 8);
        AppendLittleEndian(payload, o.size_bytes, 8);
        EncodeAddress(payload, o.sram);
        break;
    }
    case RecordOperandKind::DTE_ISSUE: {
        const auto &o = std::get<DteIssueOperands>(record.operands);
        payload.push_back(EnumByte(o.direction));
        AppendLittleEndian(payload, 0, 3);
        AppendLittleEndian(payload, o.token, 4);
        AppendLittleEndian(payload, o.payload_bits, 8);
        AppendLittleEndian(payload, o.size_bytes, 8);
        AppendLittleEndian(payload, o.hbm_address_bytes, 8);
        EncodeAddress(payload, o.source_sram);
        EncodeAddress(payload, o.destination_sram);
        break;
    }
    case RecordOperandKind::SYMBOL: {
        const auto &o = std::get<SymbolOperands>(record.operands);
        AppendLittleEndian(payload, o.symbol_index, 4);
        break;
    }
    case RecordOperandKind::SRAM_BIND: {
        const auto &o = std::get<SramBindOperands>(record.operands);
        payload.push_back(static_cast<uint8_t>(o.input_count));
        AppendLittleEndian(payload, 0, 3);
        for (uint64_t symbol : o.input_symbol_indices)
            AppendLittleEndian(payload, symbol, 4);
        AppendLittleEndian(payload, o.output_symbol_index, 4);
        break;
    }
    case RecordOperandKind::SRAM_ALLOC: {
        const auto &o = std::get<SramAllocOperands>(record.operands);
        AppendLittleEndian(payload, o.region_name_string_index, 4);
        AppendLittleEndian(payload, o.label_symbol_index, 4);
        AppendLittleEndian(payload, o.size_bytes, 8);
        AppendLittleEndian(payload, o.alignment_bytes, 8);
        payload.push_back(EnumByte(o.lifetime));
        payload.push_back(o.spillable ? 1 : 0);
        AppendLittleEndian(payload, 0, 6);
        break;
    }
    case RecordOperandKind::SRAM_RESIZE: {
        const auto &o = std::get<SramResizeOperands>(record.operands);
        AppendLittleEndian(payload, o.symbol_index, 4);
        AppendLittleEndian(payload, 0, 4);
        AppendLittleEndian(payload, o.new_size_bytes, 8);
        break;
    }
    case RecordOperandKind::SRAM_RENAME: {
        const auto &o = std::get<SramRenameOperands>(record.operands);
        AppendLittleEndian(payload, o.old_symbol_index, 4);
        AppendLittleEndian(payload, o.new_symbol_index, 4);
        break;
    }
    case RecordOperandKind::TOKEN: {
        const auto &o = std::get<TokenOperands>(record.operands);
        AppendLittleEndian(payload, o.token, 4);
        break;
    }
    case RecordOperandKind::NONE: break;
    case RecordOperandKind::EVENT_SET: {
        const auto &o = std::get<EventSetOperands>(record.operands);
        AppendLittleEndian(payload, o.source_core, 2);
        AppendLittleEndian(payload, o.destination_core, 2);
        AppendLittleEndian(payload, o.tag, 4);
        break;
    }
    case RecordOperandKind::EVENT_WAIT: {
        const auto &o = std::get<EventWaitOperands>(record.operands);
        AppendLittleEndian(payload, o.source_core, 2);
        AppendLittleEndian(payload, o.destination_core, 2);
        AppendLittleEndian(payload, o.tag, 4);
        AppendLittleEndian(payload, o.count, 4);
        break;
    }
    case RecordOperandKind::GROUP_SYNC: {
        const auto &o = std::get<GroupSyncOperands>(record.operands);
        AppendLittleEndian(payload, o.group_id, 4);
        AppendLittleEndian(payload, o.sync_seq, 4);
        break;
    }
    }
    Require(payload.size() == schema.payload_size,
            "internal record codec payload size disagrees with schema");
    return payload;
}

template <class T>
T DecodeEnum(const std::vector<uint8_t> &payload, std::size_t offset,
             std::string_view field) {
    return static_cast<T>(ReadLittleEndian(payload, offset, 1, field));
}

ExternalRecord DecodePayload(Opcode opcode, const RecordSchema &schema,
                             const std::vector<uint8_t> &payload) {
    ExternalRecord record;
    record.opcode = opcode;
    switch (schema.operand_kind) {
    case RecordOperandKind::COMPUTE: {
        ComputeOperands o;
        o.datatype = DecodeEnum<ExternalDataType>(payload, 0, "datatype");
        RequireZero(payload, 1, 1, "compute");
        o.input_offset_bytes = ReadLittleEndian(payload, 2, 2, "input_offset");
        o.data_offset_bytes = ReadLittleEndian(payload, 4, 2, "data_offset");
        o.output_offset_bytes = ReadLittleEndian(payload, 6, 2, "output_offset");
        for (std::size_t i = 0; i < schema.parameter_count; ++i)
            o.parameters.push_back(
                ReadLittleEndian(payload, 8 + 4 * i, 4, "compute parameter"));
        record.operands = std::move(o);
        break;
    }
    case RecordOperandKind::DTE_SEND: {
        DteSendOperands o;
        o.mode = DecodeEnum<DteSendMode>(payload, 0, "DTE_SEND mode");
        o.completion = DecodeEnum<EndpointCompletion>(payload, 1, "completion");
        o.datatype = DecodeEnum<EndpointDataType>(payload, 2, "datatype");
        o.reduce_op = DecodeEnum<ReduceOperator>(payload, 3, "reduce_op");
        o.fsm_id = ReadLittleEndian(payload, 4, 4, "fsm_id");
        o.token = ReadLittleEndian(payload, 8, 4, "token");
        o.source_space = DecodeEnum<EndpointSourceSpace>(
            payload, 12, "DTE_SEND source_space");
        RequireZero(payload, 13, 3, "DTE_SEND header");
        o.length_bytes = ReadLittleEndian(payload, 16, 8, "length_bytes");
        o.source = DecodeAddress(payload, 24, "DTE_SEND source");
        o.peer_core = ReadLittleEndian(payload, 48, 2, "peer_core");
        o.expected_sources = ReadLittleEndian(payload, 50, 2, "expected_sources");
        o.tree_id = ReadLittleEndian(payload, 52, 2, "tree_id");
        RequireZero(payload, 54, 2, "DTE_SEND routing");
        o.group_id = ReadLittleEndian(payload, 56, 4, "group_id");
        o.collective_id = ReadLittleEndian(payload, 60, 4, "collective_id");
        o.epoch = ReadLittleEndian(payload, 64, 4, "epoch");
        RequireZero(payload, 68, 4, "DTE_SEND tail");
        record.operands = std::move(o);
        break;
    }
    case RecordOperandKind::DTE_RECV: {
        DteRecvOperands o;
        o.mode = DecodeEnum<DteRecvMode>(payload, 0, "DTE_RECV mode");
        o.completion = DecodeEnum<EndpointCompletion>(payload, 1, "completion");
        o.datatype = DecodeEnum<EndpointDataType>(payload, 2, "datatype");
        o.reduce_op = DecodeEnum<ReduceOperator>(payload, 3, "reduce_op");
        o.fsm_id = ReadLittleEndian(payload, 4, 4, "fsm_id");
        o.token = ReadLittleEndian(payload, 8, 4, "token");
        RequireZero(payload, 12, 4, "DTE_RECV header");
        o.length_bytes = ReadLittleEndian(payload, 16, 8, "length_bytes");
        o.destination = DecodeAddress(payload, 24, "DTE_RECV destination");
        o.peer_core = ReadLittleEndian(payload, 48, 2, "peer_core");
        o.expected_sources = ReadLittleEndian(payload, 50, 2, "expected_sources");
        o.tree_id = ReadLittleEndian(payload, 52, 2, "tree_id");
        RequireZero(payload, 54, 2, "DTE_RECV routing");
        o.group_id = ReadLittleEndian(payload, 56, 4, "group_id");
        o.collective_id = ReadLittleEndian(payload, 60, 4, "collective_id");
        o.epoch = ReadLittleEndian(payload, 64, 4, "epoch");
        RequireZero(payload, 68, 4, "DTE_RECV tail");
        record.operands = std::move(o);
        break;
    }
    case RecordOperandKind::REDUCE_COMPUTE: {
        ReduceComputeOperands o;
        o.datatype = DecodeEnum<EndpointDataType>(payload, 0, "datatype");
        o.reduce_op = DecodeEnum<ReduceOperator>(payload, 1, "reduce_op");
        RequireZero(payload, 2, 2, "REDUCE_COMPUTE header");
        o.group_id = ReadLittleEndian(payload, 4, 4, "group_id");
        o.collective_id = ReadLittleEndian(payload, 8, 4, "collective_id");
        o.epoch = ReadLittleEndian(payload, 12, 4, "epoch");
        o.root_rank = ReadLittleEndian(payload, 16, 2, "root_rank");
        o.self_rank = ReadLittleEndian(payload, 18, 2, "self_rank");
        RequireZero(payload, 20, 4, "REDUCE_COMPUTE rank");
        o.element_count = ReadLittleEndian(payload, 24, 8, "element_count");
        o.source = DecodeAddress(payload, 32, "REDUCE_COMPUTE source");
        o.destination =
            DecodeAddress(payload, 56, "REDUCE_COMPUTE destination");
        record.operands = std::move(o);
        break;
    }
    case RecordOperandKind::LSU: {
        LsuOperands o;
        o.hbm_address_bytes = ReadLittleEndian(payload, 0, 8, "hbm_address");
        o.size_bytes = ReadLittleEndian(payload, 8, 8, "size_bytes");
        o.sram = DecodeAddress(payload, 16, "LSU SRAM address");
        record.operands = std::move(o);
        break;
    }
    case RecordOperandKind::DTE_ISSUE: {
        DteIssueOperands o;
        o.direction = DecodeEnum<LocalDteDirection>(payload, 0, "direction");
        RequireZero(payload, 1, 3, "DTE_ISSUE header");
        o.token = ReadLittleEndian(payload, 4, 4, "token");
        o.payload_bits = ReadLittleEndian(payload, 8, 8, "payload_bits");
        o.size_bytes = ReadLittleEndian(payload, 16, 8, "size_bytes");
        o.hbm_address_bytes = ReadLittleEndian(payload, 24, 8, "hbm_address");
        o.source_sram = DecodeAddress(payload, 32, "DTE_ISSUE source SRAM");
        o.destination_sram =
            DecodeAddress(payload, 56, "DTE_ISSUE destination SRAM");
        record.operands = std::move(o);
        break;
    }
    case RecordOperandKind::SYMBOL: {
        record.operands = SymbolOperands{
            ReadLittleEndian(payload, 0, 4, "symbol_index")};
        break;
    }
    case RecordOperandKind::SRAM_BIND: {
        SramBindOperands o;
        o.input_count = ReadLittleEndian(payload, 0, 1, "input_count");
        RequireZero(payload, 1, 3, "SRAM_BIND reserved");
        for (std::size_t i = 0; i < kSramBindInputLimit; ++i)
            o.input_symbol_indices[i] = ReadLittleEndian(
                payload, 4 + 4 * i, 4, "input_symbol_index");
        o.output_symbol_index =
            ReadLittleEndian(payload, 68, 4, "output_symbol_index");
        record.operands = std::move(o);
        break;
    }
    case RecordOperandKind::SRAM_ALLOC: {
        SramAllocOperands o;
        o.region_name_string_index =
            ReadLittleEndian(payload, 0, 4, "region_name_string_index");
        o.label_symbol_index =
            ReadLittleEndian(payload, 4, 4, "label_symbol_index");
        o.size_bytes = ReadLittleEndian(payload, 8, 8, "size_bytes");
        o.alignment_bytes = ReadLittleEndian(payload, 16, 8, "alignment_bytes");
        o.lifetime = DecodeEnum<SramLifetime>(payload, 24, "lifetime");
        const uint64_t raw_spillable =
            ReadLittleEndian(payload, 25, 1, "spillable");
        Require(raw_spillable <= 1, "SRAM_ALLOC spillable must be 0 or 1");
        o.spillable = raw_spillable != 0;
        RequireZero(payload, 26, 6, "SRAM_ALLOC tail");
        record.operands = std::move(o);
        break;
    }
    case RecordOperandKind::SRAM_RESIZE: {
        SramResizeOperands o;
        o.symbol_index = ReadLittleEndian(payload, 0, 4, "symbol_index");
        RequireZero(payload, 4, 4, "SRAM_RESIZE");
        o.new_size_bytes = ReadLittleEndian(payload, 8, 8, "new_size_bytes");
        record.operands = std::move(o);
        break;
    }
    case RecordOperandKind::SRAM_RENAME: {
        record.operands = SramRenameOperands{
            ReadLittleEndian(payload, 0, 4, "old_symbol_index"),
            ReadLittleEndian(payload, 4, 4, "new_symbol_index")};
        break;
    }
    case RecordOperandKind::TOKEN: {
        record.operands =
            TokenOperands{ReadLittleEndian(payload, 0, 4, "token")};
        break;
    }
    case RecordOperandKind::NONE: record.operands = NoOperands{}; break;
    case RecordOperandKind::EVENT_SET: {
        EventSetOperands o;
        o.source_core = ReadLittleEndian(payload, 0, 2, "source_core");
        o.destination_core =
            ReadLittleEndian(payload, 2, 2, "destination_core");
        o.tag = ReadLittleEndian(payload, 4, 4, "tag");
        record.operands = std::move(o);
        break;
    }
    case RecordOperandKind::EVENT_WAIT: {
        EventWaitOperands o;
        o.source_core = ReadLittleEndian(payload, 0, 2, "source_core");
        o.destination_core =
            ReadLittleEndian(payload, 2, 2, "destination_core");
        o.tag = ReadLittleEndian(payload, 4, 4, "tag");
        o.count = ReadLittleEndian(payload, 8, 4, "count");
        record.operands = std::move(o);
        break;
    }
    case RecordOperandKind::GROUP_SYNC: {
        record.operands = GroupSyncOperands{
            ReadLittleEndian(payload, 0, 4, "group_id"),
            ReadLittleEndian(payload, 4, 4, "sync_seq")};
        break;
    }
    }
    return record;
}

} // namespace

const std::array<RecordSchema, kOpcodeManifestSize> &RecordSchemaManifest()
    noexcept {
    return kSchemas;
}

const RecordSchema *LookupRecordSchema(uint8_t raw_opcode) noexcept {
    const auto it = std::lower_bound(
        kSchemas.begin(), kSchemas.end(), raw_opcode,
        [](const RecordSchema &schema, uint8_t candidate) {
            return OpcodeValue(schema.opcode) < candidate;
        });
    if (it == kSchemas.end() || OpcodeValue(it->opcode) != raw_opcode)
        return nullptr;
    return &*it;
}

void ValidateExternalRecord(const ExternalRecord &record,
                            uint64_t enabled_capabilities) {
    ValidateOpcodeForRecord(record.opcode, enabled_capabilities);
    const RecordSchema *schema = LookupRecordSchema(record.opcode);
    if (schema == nullptr)
        Fail("available external opcode has no operand schema: " +
             HexByte(OpcodeValue(record.opcode)));
    ValidateOperandsForSchema(record, *schema);
}

std::vector<uint8_t>
EncodeExternalRecord(const ExternalRecord &record,
                     uint64_t enabled_capabilities) {
    ValidateExternalRecord(record, enabled_capabilities);
    const RecordSchema &schema = *LookupRecordSchema(record.opcode);
    std::vector<uint8_t> payload = EncodePayload(record, schema);
    std::vector<uint8_t> bytes;
    bytes.reserve(kExternalRecordHeaderSize + payload.size());
    bytes.push_back(OpcodeValue(record.opcode));
    bytes.push_back(kExternalRecordVersion);
    AppendLittleEndian(bytes, kExternalRecordFlags, 2);
    AppendLittleEndian(bytes, payload.size(), 4);
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    return bytes;
}

DecodedExternalRecord
DecodeExternalRecord(const std::vector<uint8_t> &bytes, std::size_t offset,
                     uint64_t enabled_capabilities) {
    if (offset > bytes.size())
        Fail("external record offset is outside the input");
    if (bytes.size() - offset < kExternalRecordHeaderSize)
        Fail("truncated external record header");
    const uint8_t raw_opcode = bytes[offset];
    if (raw_opcode == OpcodeValue(Opcode::INVALID))
        Fail("external record opcode INVALID is forbidden");
    const uint8_t version = bytes[offset + 1];
    if (version != kExternalRecordVersion)
        Fail("unsupported external record version: " +
             std::to_string(version));
    const uint16_t flags = static_cast<uint16_t>(
        ReadLittleEndian(bytes, offset + 2, 2, "record flags"));
    if (flags != kExternalRecordFlags)
        Fail("external record flags must be zero");
    const uint32_t payload_size = static_cast<uint32_t>(
        ReadLittleEndian(bytes, offset + 4, 4, "payload_size"));
    const std::size_t payload_offset = offset + kExternalRecordHeaderSize;
    if (payload_size > bytes.size() - payload_offset)
        Fail("truncated external record payload");

    const Opcode opcode = static_cast<Opcode>(raw_opcode);
    ValidateOpcodeForRecord(opcode, enabled_capabilities);
    const RecordSchema *schema = LookupRecordSchema(raw_opcode);
    if (schema == nullptr)
        Fail("available external opcode has no operand schema: " +
             HexByte(raw_opcode));
    if (payload_size != schema->payload_size)
        Fail("external record payload_size does not match opcode schema");
    const std::size_t next_offset = payload_offset + payload_size;
    std::vector<uint8_t> payload(bytes.begin() + payload_offset,
                                 bytes.begin() + next_offset);
    ExternalRecord record = DecodePayload(opcode, *schema, payload);
    ValidateExternalRecord(record, enabled_capabilities);
    return {std::move(record), next_offset};
}

ExternalRecord
DecodeExternalRecordExact(const std::vector<uint8_t> &bytes,
                          uint64_t enabled_capabilities) {
    DecodedExternalRecord decoded =
        DecodeExternalRecord(bytes, 0, enabled_capabilities);
    if (decoded.next_offset != bytes.size())
        Fail("trailing bytes after external record");
    return std::move(decoded.record);
}

std::vector<ExternalRecord>
DecodeExternalRecordStream(const std::vector<uint8_t> &bytes,
                           uint64_t enabled_capabilities) {
    std::vector<ExternalRecord> records;
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        DecodedExternalRecord decoded =
            DecodeExternalRecord(bytes, offset, enabled_capabilities);
        Require(decoded.next_offset > offset,
                "external record decoder made no progress");
        offset = decoded.next_offset;
        records.push_back(std::move(decoded.record));
    }
    return records;
}

std::string ExternalRecordHex(const std::vector<uint8_t> &bytes) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string result;
    result.reserve(bytes.size() * 2);
    for (uint8_t byte : bytes) {
        result.push_back(kHex[byte >> 4]);
        result.push_back(kHex[byte & 0x0f]);
    }
    return result;
}
