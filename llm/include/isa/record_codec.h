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

enum class ExternalDataType : uint8_t {
    INT8 = 0,
    FP16 = 1,
    // Fixed Stage2 records use INT32 for token indices/results. Generic
    // ComputeOperands deliberately retains its INT8/FP16 allowlist.
    INT32 = 2,
    // Fixed CE_FORWARD uses FP32 only for its unreduced loss output.
    FP32 = 3,
};
enum class EndpointDataType : uint8_t { UINT8 = 0, INT32 = 1, INT64 = 2 };
enum class ReduceOperator : uint8_t { NONE = 0, SUM = 1, MAX = 2 };
enum class LocalReduceDataType : uint8_t { FP16 = 0, FP32 = 1 };
enum class LocalReduceRoundingMode : uint8_t { RNE = 0 };
enum class LocalReduceOrder : uint8_t { RANK_MAJOR = 0 };
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
enum class RopePackedLayout : uint8_t { Q_K_V = 0 };
enum class ExactAttentionMode : uint8_t {
    PREFILL = 0,
    DECODE = 1,
    EXACT_PROFILE = 2,
    TRAIN_FORWARD = 3,
};
enum class AttentionPackedLayout : uint8_t { Q_K_V = 0 };
enum class EmbeddingPlacement : uint8_t { REPLICATED = 0 };
enum class GreedySampleMode : uint8_t { GREEDY = 0 };
enum class GreedyRowSelection : uint8_t { LAST_PER_SEQUENCE = 0 };
enum class CrossEntropyReduction : uint8_t { NONE = 0 };
enum class CrossEntropyUpstreamMode : uint8_t {
    SCALAR = 0,
    PER_ROW = 1,
};
enum class OptimizerRoundingMode : uint8_t { RNE = 0 };

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

struct RopeQkExactOperands {
    ExternalDataType datatype = ExternalDataType::FP16;
    RopePackedLayout packed_layout = RopePackedLayout::Q_K_V;
    SramAddressOperand input;
    SramAddressOperand output;
    uint64_t logical_tokens = 1;
    uint64_t tp_degree = 1;
    uint64_t num_heads = 1;
    uint64_t num_kv_heads = 1;
    uint64_t rank_num_heads = 1;
    uint64_t rank_num_kv_heads = 1;
    uint64_t head_dim = 2;
    uint64_t rotary_dim = 2;
    uint64_t max_position_embeddings = 1;
    uint64_t context_max = 1;
    uint64_t rope_theta_f64_bits = 0;
};

struct AttentionExactOperands {
    ExternalDataType datatype = ExternalDataType::FP16;
    ExactAttentionMode mode = ExactAttentionMode::PREFILL;
    AttentionPackedLayout packed_layout = AttentionPackedLayout::Q_K_V;
    bool causal = true;
    SramAddressOperand input;
    SramAddressOperand output;
    uint64_t query_tokens = 1;
    uint64_t tp_degree = 1;
    uint64_t num_heads = 1;
    uint64_t num_kv_heads = 1;
    uint64_t rank_num_heads = 1;
    uint64_t rank_num_kv_heads = 1;
    uint64_t head_dim = 1;
    uint64_t context_sum = 1;
    uint64_t context_max = 1;
    uint64_t query_key_pairs = 1;
    uint64_t rank_kv_read_bytes = 0;
    uint64_t rank_kv_write_bytes = 4;
};

struct EmbeddingLookupOperands {
    ExternalDataType index_datatype = ExternalDataType::INT32;
    ExternalDataType table_datatype = ExternalDataType::FP16;
    ExternalDataType output_datatype = ExternalDataType::FP16;
    EmbeddingPlacement placement = EmbeddingPlacement::REPLICATED;
    SramAddressOperand indices;
    SramAddressOperand table;
    SramAddressOperand output;
    uint64_t logical_rows = 1;
    uint64_t rank_rows = 1;
    uint64_t tp_degree = 1;
    uint64_t vocab_size = 1;
    uint64_t hidden_size = 1;
};

struct GreedySampleOperands {
    ExternalDataType logits_datatype = ExternalDataType::FP16;
    ExternalDataType output_datatype = ExternalDataType::INT32;
    GreedySampleMode mode = GreedySampleMode::GREEDY;
    GreedyRowSelection row_selection =
        GreedyRowSelection::LAST_PER_SEQUENCE;
    SramAddressOperand logits;
    SramAddressOperand output;
    uint64_t tp_degree = 1;
    uint64_t token_rows = 1;
    uint64_t vocab_size = 2;
    uint64_t sample_count = 1;
    uint64_t comparisons = 1;
};

struct CrossEntropyForwardOperands {
    ExternalDataType logits_datatype = ExternalDataType::FP16;
    ExternalDataType label_datatype = ExternalDataType::INT32;
    ExternalDataType loss_datatype = ExternalDataType::FP32;
    CrossEntropyReduction reduction = CrossEntropyReduction::NONE;
    SramAddressOperand logits;
    SramAddressOperand labels;
    SramAddressOperand loss;
    uint64_t logical_rows = 1;
    uint64_t rank_rows = 1;
    uint64_t tp_degree = 1;
    uint64_t vocab_size = 2;
};

struct CrossEntropyBackwardOperands {
    ExternalDataType logits_datatype = ExternalDataType::FP16;
    ExternalDataType label_datatype = ExternalDataType::INT32;
    ExternalDataType upstream_datatype = ExternalDataType::FP32;
    ExternalDataType output_datatype = ExternalDataType::FP16;
    CrossEntropyReduction reduction = CrossEntropyReduction::NONE;
    CrossEntropyUpstreamMode upstream_mode =
        CrossEntropyUpstreamMode::PER_ROW;
    SramAddressOperand logits;
    SramAddressOperand labels;
    SramAddressOperand upstream;
    SramAddressOperand logits_grad;
    uint64_t logical_rows = 1;
    uint64_t rank_rows = 1;
    uint64_t tp_degree = 1;
    uint64_t vocab_size = 2;
    uint64_t upstream_elements = 1;
};

struct SgdUpdateOperands {
    ExternalDataType weight_datatype = ExternalDataType::FP16;
    ExternalDataType gradient_datatype = ExternalDataType::FP32;
    ExternalDataType output_datatype = ExternalDataType::FP16;
    OptimizerRoundingMode rounding = OptimizerRoundingMode::RNE;
    SramAddressOperand weight;
    SramAddressOperand gradient;
    SramAddressOperand updated_weight;
    uint64_t element_count = 1;
    uint64_t learning_rate_f64_bits = 0;
    uint64_t momentum_f64_bits = 0;
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

// Stable standalone local reduction contract.  V1 deliberately fixes the
// floating-point semantics while retaining each field in the byte ABI so a
// producer cannot silently depend on an implementation default.
struct LocalReduceOperands {
    LocalReduceDataType input_dtype = LocalReduceDataType::FP16;
    LocalReduceDataType accumulator_dtype = LocalReduceDataType::FP32;
    LocalReduceDataType output_dtype = LocalReduceDataType::FP16;
    ReduceOperator reduce_op = ReduceOperator::SUM;
    LocalReduceRoundingMode rounding = LocalReduceRoundingMode::RNE;
    LocalReduceOrder order = LocalReduceOrder::RANK_MAJOR;
    uint64_t input_count = 1;
    uint64_t element_count = 1;
    uint64_t input_stride_bytes = 2;
    SramAddressOperand source;
    SramAddressOperand destination;
};

struct LocalNocSendOperands { SramAddressOperand source; uint64_t destination_core = 0; uint64_t byte_count = 1; uint64_t event_id = 0; };
struct LocalNocRecvOperands { SramAddressOperand destination; uint64_t source_core = 0; uint64_t byte_count = 1; uint64_t event_id = 0; };
struct LocalNocWaitOperands { uint64_t event_id = 0; };

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

// Stable fixed-placement allocation ABI. This is intentionally a distinct
// opcode/operand type so the public SRAM_ALLOC v1 payload remains unchanged.
struct SramAllocAtOperands {
    uint64_t region_name_string_index = 0;
    uint64_t label_symbol_index = 0;
    uint64_t region_offset_bytes = 0;
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
    std::variant<ComputeOperands, RopeQkExactOperands,
                 AttentionExactOperands, EmbeddingLookupOperands,
                 GreedySampleOperands, CrossEntropyForwardOperands,
                 CrossEntropyBackwardOperands, SgdUpdateOperands,
                 DteSendOperands, DteRecvOperands,
                 ReduceComputeOperands, LocalReduceOperands, LsuOperands,
                 LocalNocSendOperands, LocalNocRecvOperands, LocalNocWaitOperands,
                 DteIssueOperands,
                 SymbolOperands, SramBindOperands, SramAllocOperands,
                 SramAllocAtOperands, SramResizeOperands, SramRenameOperands,
                 TokenOperands,
                 NoOperands, EventSetOperands, EventWaitOperands,
                 GroupSyncOperands>;

struct ExternalRecord {
    Opcode opcode = Opcode::INVALID;
    RecordOperands operands = NoOperands{};
};

enum class RecordOperandKind : uint8_t {
    COMPUTE,
    ROPE_QK_EXACT,
    ATTENTION_EXACT,
    EMBEDDING_LOOKUP,
    GREEDY_SAMPLE,
    CROSS_ENTROPY_FORWARD,
    CROSS_ENTROPY_BACKWARD,
    SGD_UPDATE,
    DTE_SEND,
    DTE_RECV,
    REDUCE_COMPUTE,
    LOCAL_REDUCE,
    LOCAL_NOC_SEND,
    LOCAL_NOC_RECV,
    LOCAL_NOC_WAIT,
    LSU,
    DTE_ISSUE,
    SYMBOL,
    SRAM_BIND,
    SRAM_ALLOC,
    SRAM_ALLOC_AT,
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
