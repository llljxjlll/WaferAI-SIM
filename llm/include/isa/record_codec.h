#pragma once

#include "dte/endpoint_contract.h"
#include "isa/opcode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

// External ISA records are a byte-oriented, little-endian ABI. They never
// contain PrimId values, sc_bv segments, or process-local label-table IDs.
inline constexpr uint8_t kExternalRecordVersion = 1;
inline constexpr uint16_t kExternalRecordFlags = 0;
inline constexpr std::size_t kExternalRecordHeaderSize = 8;
inline constexpr uint64_t kExternalNpuParameterMax = (uint64_t{1} << 30) - 1;

class RecordCodecError : public std::invalid_argument {
public:
    explicit RecordCodecError(const std::string &message)
        : std::invalid_argument(message) {}
};

enum class ExternalDataType : uint8_t { INT8 = 0, FP16 = 1 };
enum class EndpointDataType : uint8_t { UINT8 = 0, INT32 = 1, INT64 = 2 };
enum class ReduceOperator : uint8_t { NONE = 0, SUM = 1, MAX = 2 };
enum class EndpointCompletion : uint8_t { ASYNC = 0, SYNC = 1 };
enum class EndpointSourceSpace : uint8_t { SRAM = 0, HBM = 1 };
enum class DteSendMode : uint8_t { P2P = 0, SCATTER = 1, BROADCAST = 2 };
enum class DteRecvMode : uint8_t { P2P = 0, GATHER = 1, REDUCE = 2 };
enum class LocalDteDirection : uint8_t {
    SPM_TO_SPM = 0,
    SPM_TO_DRAM = 1,
    DRAM_TO_SPM = 2,
};
enum class SramAddressKind : uint8_t { NONE = 0, ABSOLUTE = 1, REGION = 2 };
enum class SramLifetime : uint8_t { TASK = 0, LAYER = 1, PERSISTENT = 2 };

// Wide integer members are intentional: callers can present max+1 values and
// receive a deterministic error before the value is narrowed into the ABI.
struct ComputeOperands {
    ExternalDataType datatype = ExternalDataType::INT8;
    uint64_t input_offset_bytes = 0;
    uint64_t data_offset_bytes = 0;
    uint64_t output_offset_bytes = 0;
    std::vector<uint64_t> parameters;
};

struct SramAddressOperand {
    SramAddressKind kind = SramAddressKind::NONE;
    uint64_t absolute_address_bytes = 0;
    uint64_t region_symbol_index = 0;
    uint64_t region_offset_bytes = 0;
};

struct DteSendOperands {
    DteSendMode mode = DteSendMode::P2P;
    EndpointSourceSpace source_space = EndpointSourceSpace::SRAM;
    EndpointCompletion completion = EndpointCompletion::ASYNC;
    EndpointDataType datatype = EndpointDataType::UINT8;
    ReduceOperator reduce_op = ReduceOperator::NONE;
    uint64_t fsm_id = 1;
    uint64_t token = 1;
    uint64_t length_bytes = 1;
    SramAddressOperand source;
    uint64_t peer_core = 0;
    uint64_t expected_sources = 0;
    uint64_t tree_id = 0;
    uint64_t group_id = 0;
    uint64_t collective_id = 0;
    uint64_t epoch = 0;
};

struct DteRecvOperands {
    DteRecvMode mode = DteRecvMode::P2P;
    EndpointCompletion completion = EndpointCompletion::ASYNC;
    EndpointDataType datatype = EndpointDataType::UINT8;
    ReduceOperator reduce_op = ReduceOperator::NONE;
    uint64_t fsm_id = 1;
    uint64_t token = 1;
    uint64_t length_bytes = 1;
    SramAddressOperand destination;
    uint64_t peer_core = 0;
    uint64_t expected_sources = 0;
    uint64_t tree_id = 0;
    uint64_t group_id = 0;
    uint64_t collective_id = 0;
    uint64_t epoch = 0;
};

struct ReduceComputeOperands {
    EndpointDataType datatype = EndpointDataType::UINT8;
    ReduceOperator reduce_op = ReduceOperator::SUM;
    uint64_t group_id = 1;
    uint64_t collective_id = 0;
    uint64_t epoch = 0;
    uint64_t root_rank = 0;
    uint64_t self_rank = 0;
    uint64_t element_count = 1;
    SramAddressOperand source;
    SramAddressOperand destination;
};

struct LsuOperands {
    uint64_t hbm_address_bytes = 0;
    uint64_t size_bytes = 1;
    SramAddressOperand sram;
};

struct DteIssueOperands {
    LocalDteDirection direction = LocalDteDirection::SPM_TO_SPM;
    uint64_t token = 1;
    uint64_t payload_bits = 8;
    uint64_t size_bytes = 1;
    uint64_t hbm_address_bytes = 0;
    SramAddressOperand source_sram;
    SramAddressOperand destination_sram;
};

struct SymbolOperands {
    uint64_t symbol_index = 0;
};

inline constexpr std::size_t kSramBindInputLimit = 16;

struct SramBindOperands {
    // Wide members are narrowed only after validation so callers receive a
    // deterministic range error instead of an implicit ABI truncation.
    uint64_t input_count = 1;
    std::array<uint64_t, kSramBindInputLimit> input_symbol_indices{};
    uint64_t output_symbol_index = 0;
};

struct SramAllocOperands {
    uint64_t region_name_string_index = 0;
    uint64_t label_symbol_index = 0;
    uint64_t size_bytes = 1;
    uint64_t alignment_bytes = 1;
    SramLifetime lifetime = SramLifetime::TASK;
    bool spillable = true;
};

struct SramResizeOperands {
    uint64_t symbol_index = 0;
    uint64_t new_size_bytes = 1;
};

struct SramRenameOperands {
    uint64_t old_symbol_index = 0;
    uint64_t new_symbol_index = 1;
};

struct TokenOperands {
    uint64_t token = 1;
};

struct NoOperands {};

struct EventSetOperands {
    uint64_t source_core = 0;
    uint64_t destination_core = 0;
    uint64_t tag = 0;
};

struct EventWaitOperands {
    uint64_t source_core = 0;
    uint64_t destination_core = 0;
    uint64_t tag = 0;
    uint64_t count = 1;
};

struct GroupSyncOperands {
    uint64_t group_id = 1;
    uint64_t sync_seq = 0;
};

using RecordOperands =
    std::variant<ComputeOperands, DteSendOperands, DteRecvOperands,
                 ReduceComputeOperands, LsuOperands, DteIssueOperands,
                 SymbolOperands, SramBindOperands, SramAllocOperands,
                 SramResizeOperands, SramRenameOperands, TokenOperands,
                 NoOperands, EventSetOperands, EventWaitOperands,
                 GroupSyncOperands>;

struct ExternalRecord {
    Opcode opcode = Opcode::INVALID;
    RecordOperands operands = NoOperands{};
};

enum class RecordOperandKind : uint8_t {
    COMPUTE,
    DTE_SEND,
    DTE_RECV,
    REDUCE_COMPUTE,
    LSU,
    DTE_ISSUE,
    SYMBOL,
    SRAM_BIND,
    SRAM_ALLOC,
    SRAM_RESIZE,
    SRAM_RENAME,
    TOKEN,
    NONE,
    EVENT_SET,
    EVENT_WAIT,
    GROUP_SYNC,
};

struct RecordSchema {
    Opcode opcode;
    RecordOperandKind operand_kind;
    uint32_t payload_size;
    const std::string_view *parameter_names;
    std::size_t parameter_count;
};

const std::array<RecordSchema, kOpcodeManifestSize> &RecordSchemaManifest()
    noexcept;
const RecordSchema *LookupRecordSchema(uint8_t raw_opcode) noexcept;
inline const RecordSchema *LookupRecordSchema(Opcode opcode) noexcept {
    return LookupRecordSchema(OpcodeValue(opcode));
}

// Throws RecordCodecError on any schema, capability, range, enum, reserved-bit,
// address-XOR, or canonical-unused-field violation.
void ValidateExternalRecord(const ExternalRecord &record,
                            uint64_t enabled_capabilities = 0);

std::vector<uint8_t>
EncodeExternalRecord(const ExternalRecord &record,
                     uint64_t enabled_capabilities = 0);

struct DecodedExternalRecord {
    ExternalRecord record;
    std::size_t next_offset = 0;
};

// Decodes exactly one record beginning at offset and returns the first byte not
// consumed. Bytes after next_offset belong to the next record in a stream.
DecodedExternalRecord
DecodeExternalRecord(const std::vector<uint8_t> &bytes,
                     std::size_t offset = 0,
                     uint64_t enabled_capabilities = 0);

// Requires the supplied byte vector to contain one record and no trailing data.
ExternalRecord
DecodeExternalRecordExact(const std::vector<uint8_t> &bytes,
                          uint64_t enabled_capabilities = 0);

// Consumes records until the input boundary. Empty streams are valid.
std::vector<ExternalRecord>
DecodeExternalRecordStream(const std::vector<uint8_t> &bytes,
                           uint64_t enabled_capabilities = 0);

std::string ExternalRecordHex(const std::vector<uint8_t> &bytes);
