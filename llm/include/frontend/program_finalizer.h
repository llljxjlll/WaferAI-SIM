#pragma once

#include "isa/program_format.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace frontend {

inline constexpr std::string_view kLinkedProgramManifestSchemaVersion =
    "wafer_frontend.linked_program_manifest/v1alpha14";
inline constexpr std::string_view kCommandFragmentSchemaVersion =
    "wafer_frontend.command_fragment/v1alpha13";
inline constexpr std::string_view kRegionManifestSchemaVersion =
    "wafer_frontend.region_manifest/v1alpha12";
inline constexpr std::string_view kStateAbiSchemaVersion = "wafer_frontend.state_abi/v1alpha1";

class ProgramFinalizerError : public std::invalid_argument {
public:
    explicit ProgramFinalizerError(const std::string &message)
        : std::invalid_argument(message) {}
};

struct LogicalCoreDto {
    uint64_t die_id = 0;
    uint64_t local_core_id = 0;

    friend bool operator==(const LogicalCoreDto &left,
                           const LogicalCoreDto &right) noexcept {
        return left.die_id == right.die_id &&
               left.local_core_id == right.local_core_id;
    }
    friend bool operator<(const LogicalCoreDto &left,
                          const LogicalCoreDto &right) noexcept {
        return left.die_id < right.die_id ||
               (left.die_id == right.die_id &&
                left.local_core_id < right.local_core_id);
    }
};

enum class FragmentKindDto {
    COARSE,
    STANDALONE_COLLECTIVE,
    ISA_REGION,
    STATE_IO,
    STATE_TRANSFER,
    MOE_TRANSFER,
    S2_LITE_ROOTED_AR,
    SWIZZLE,
    MOE_SWIZZLE,
    MOE_SWIZZLE_CALIBRATION,
    UNFUSED_COMPARISON,
};

enum class RuntimeSymbolKindDto {
    START_TAG,
    EVENT_TAG,
    DTE_TOKEN,
    DTE_FSM,
    GROUP,
    RUNTIME_CORE,
};

enum class OperandKindDto {
    LITERAL,
    RUNTIME_SYMBOL,
    ADDRESS_SYMBOL,
};

enum class RuntimeOperandFieldDto {
    START_TAG,
    EVENT_TAG,
    DTE_TOKEN,
    DTE_FSM,
    GROUP_ID,
    SOURCE_CORE,
    DESTINATION_CORE,
    PEER_CORE,
};

enum class ManifestInputKindDto {
    S3_LITE_MOE,
    S2_LITE_ROOTED_AR,
    TRAIN_LOWERED_PROGRAM,
    IR1,
    FUSION_PLAN,
    STANDALONE_PLAN,
    IR2_PROJECTION,
    SCHEDULE_SET,
    GLOBAL_ACTION_DAG,
    COMMAND_FRAGMENT,
    REGION_MANIFEST,
    SWIZZLE_DECISION,
    SWIZZLE_CANDIDATE,
    SWIZZLE_FUSION_PLAN,
    SWIZZLE_PROJECTION,
    SWIZZLE_LOWERED_PROGRAM,
    SWIZZLE_CORE_ADDRESS_ABI,
    SWIZZLE_OPERAND_ABI,
    MOE_SWIZZLE_SCALE_SPEC,
    MOE_SWIZZLE_SCALE_ORACLE,
    MOE_SWIZZLE_EXECUTION,
    MOE_SWIZZLE_DECISION,
    MOE_SWIZZLE_WORKLOAD_SELECTION,
    MOE_SWIZZLE_WORKLOAD_PROJECTION,
    MOE_SWIZZLE_WORKLOAD_STATE_ABI,
    MOE_SWIZZLE_WORKLOAD_VALUE_BRIDGE,
    MOE_SWIZZLE_WORKLOAD_ABI,
    MOE_SWIZZLE_HARDWARE_FACTS,
    MOE_SWIZZLE_OVERLAY,
    MOE_SWIZZLE_PROJECTION,
    MOE_SWIZZLE_CORE_ADDRESS_ABI,
    MOE_SWIZZLE_OPERAND_ABI,
    MOE_SWIZZLE_CALIBRATION_SOURCE,
    UNFUSED_COMPARISON_BASELINE,
    UNFUSED_COMPARISON_PLAN,
    UNFUSED_COMPARISON_PROJECTION,
    UNFUSED_COMPARISON_LOWERED,
    UNFUSED_COMPARISON_CORE_ABI,
    UNFUSED_COMPARISON_OPERAND_ABI,
};

enum class BufferDTypeDto { FP16, FP32, INT32 };
enum class BufferOwnershipDto { OWNED, BORROWED, ALIASED };
enum class StateKindDto {
    PARAMETER,
    TRAINABLE_PARAMETER,
    KV_KEY,
    KV_VALUE,
    OPTIMIZER_RESERVED,
};
enum class StateLifetimeDto { STEP, PERSISTENT };
enum class StateAccessDto { READ_ONLY, READ_WRITE, RESERVED };


using LiteralValueDto =
    std::variant<std::monostate, uint64_t, std::string, bool,
                 std::vector<uint64_t>>;

struct RuntimeSymbolDto {
    std::string id;
    RuntimeSymbolKindDto kind = RuntimeSymbolKindDto::START_TAG;
    std::string source_ref;
};

struct ProgramSymbolDto {
    std::string id;
    ProgramSymbolKind kind = ProgramSymbolKind::ABSOLUTE_ADDRESS;
    std::string source_ref;
};

struct RecordOperandDto {
    std::string name;
    OperandKindDto kind = OperandKindDto::LITERAL;
    LiteralValueDto literal_value;
    std::optional<RuntimeOperandFieldDto> runtime_field;
    std::optional<SemanticOperandId> operand_id;
    std::optional<std::string> symbol_ref;
};

struct RelocatableRecordDto {
    std::string source_global_action_id;
    Opcode opcode = Opcode::INVALID;
    std::vector<RecordOperandDto> operands;
};

struct RuntimeRelocationDto {
    uint64_t record_index = 0;
    RuntimeOperandFieldDto field = RuntimeOperandFieldDto::START_TAG;
    std::string symbol_ref;
};

struct AddressRelocationDto {
    uint64_t record_index = 0;
    SemanticOperandId operand_id =
        SemanticOperandId::COMPUTE_INPUT_ADDRESS;
    ProgramSymbolKind symbol_kind = ProgramSymbolKind::ABSOLUTE_ADDRESS;
    std::string symbol_ref;
    int64_t addend = 0;
};

struct TensorSliceDto {
    std::string value_id;
    std::vector<uint64_t> offset;
    std::vector<uint64_t> shape;
};

struct BufferAbiDto {
    std::string id;
    std::string schedule_id;
    std::string binding_id;
    std::string value_id;
    LogicalCoreDto logical_core;
    TensorSliceDto tensor_slice;
    std::string region_ref;
    uint64_t region_offset_bytes = 0;
    uint64_t size_bytes = 0;
    uint64_t alignment_bytes = 0;
    std::vector<uint64_t> banks;
    std::string storage_id;
    std::optional<std::string> alias_of;
    uint64_t lifetime_start = 0;
    uint64_t lifetime_end_exclusive = 0;
    BufferDTypeDto dtype = BufferDTypeDto::FP16;
    std::string layout;
    BufferOwnershipDto ownership = BufferOwnershipDto::OWNED;
};
struct StateAbiDto {
    std::string id;
    std::string state_ref;
    std::string hbm_binding_ref;
    StateKindDto kind = StateKindDto::PARAMETER;
    StateLifetimeDto lifetime = StateLifetimeDto::PERSISTENT;
    StateAccessDto access = StateAccessDto::READ_ONLY;
    std::vector<uint64_t> shape;
    BufferDTypeDto dtype = BufferDTypeDto::FP16;
    std::string layout;
    uint64_t die_id = 0;
    uint64_t address = 0;
    uint64_t size_bytes = 0;
    uint64_t alignment_bytes = 0;
};


struct CoreFragmentStreamDto {
    LogicalCoreDto logical_core;
    std::vector<RelocatableRecordDto> records;
    std::vector<RuntimeRelocationDto> runtime_relocations;
    std::vector<AddressRelocationDto> address_relocations;
};

struct CommandFragmentDto {
    std::string schema_version;
    std::string producer_pass;
    std::string id;
    std::string source_global_dag_id;
    FragmentKindDto kind = FragmentKindDto::COARSE;
    std::vector<std::string> claimed_action_ids;
    std::vector<CoreFragmentStreamDto> core_streams;
    std::vector<RuntimeSymbolDto> runtime_symbols;
    std::vector<ProgramSymbolDto> program_symbols;
    std::vector<BufferAbiDto> buffer_abi;
    std::vector<StateAbiDto> state_abi;
};

struct RegionManifestDto {
    std::string schema_version;
    std::string producer_pass;
    std::string id;
    std::string region_id;
    std::string fusion_plan_id;
    std::vector<uint64_t> target_dies;
    CommandFragmentDto fragment;
};

using LinkedFragmentDto =
    std::variant<CommandFragmentDto, RegionManifestDto>;

struct ManifestInputDigestDto {
    ManifestInputKindDto kind = ManifestInputKindDto::IR1;
    std::string artifact_id;
    std::string schema_version;
    std::string digest;
};

struct CoreRuntimeBindingDto {
    LogicalCoreDto logical_core;
    std::string core_spec_ref;
    uint64_t runtime_core_id = 0;
    std::string sram_profile_ref;
};

struct LinkedRecordRefDto {
    std::string fragment_id;
    uint64_t fragment_record_index = 0;
    std::string source_global_action_id;
};

struct LinkedCoreStreamDto {
    LogicalCoreDto logical_core;
    uint64_t runtime_core_id = 0;
    std::vector<LinkedRecordRefDto> records;
};

struct EventCreditDto {
    std::string symbol_ref;
    uint64_t count = 0;
};

struct FragmentInterfaceDto {
    std::string fragment_id;
    std::vector<std::string> runtime_imports;
    std::vector<std::string> runtime_exports;
    std::vector<std::string> program_imports;
    std::vector<std::string> program_exports;
    std::vector<EventCreditDto> entry_events;
    std::vector<EventCreditDto> exit_events;
};

struct RuntimeSymbolDefinitionDto {
    RuntimeSymbolDto symbol;
    std::vector<LogicalCoreDto> logical_cores;
    std::optional<std::string> source_action_id;
    std::optional<std::string> destination_action_id;
};

struct ProgramSymbolDefinitionDto {
    ProgramSymbolDto symbol;
    std::string name;
    uint64_t value = 0;
    uint64_t size_bytes = 0;
    std::vector<LogicalCoreDto> logical_cores;
};

struct AddressOperandBindingDto {
    std::string fragment_id;
    LogicalCoreDto logical_core;
    uint64_t fragment_record_index = 0;
    SemanticOperandId operand_id =
        SemanticOperandId::COMPUTE_INPUT_ADDRESS;
    std::vector<std::string> buffer_abi_ids;
    std::vector<TensorSliceDto> tensor_slices;
};

struct StateOperandBindingDto {
    std::string fragment_id;
    LogicalCoreDto logical_core;
    uint64_t fragment_record_index = 0;
    SemanticOperandId operand_id = SemanticOperandId::HBM_ADDRESS;
    std::string state_abi_id;
};


struct LogicalCoreGroupDto {
    std::string symbol_ref;
    std::vector<LogicalCoreDto> members;
};

struct LogicalStartEventDto {
    LogicalCoreDto target_core;
    std::string tag_symbol_ref;
    uint64_t count = 0;
};

struct ProgramControlEnvelopeDto {
    std::vector<LogicalCoreDto> active_cores;
    std::vector<LogicalStartEventDto> start_events;
    std::vector<LogicalCoreDto> terminal_cores;
    std::vector<LogicalCoreDto> expected_ack_cores;
    std::vector<LogicalCoreDto> expected_done_cores;
    EmptyCoreAckPolicy empty_core_ack_policy =
        EmptyCoreAckPolicy::EXCLUDE_EMPTY;
    ProgramFailurePolicy failure_policy = ProgramFailurePolicy::ABORT_ALL;
};

struct LinkedProgramManifestDto {
    std::string schema_version;
    std::string producer_pass;
    std::string id;
    uint64_t capabilities = 0;
    std::string source_ir1_id;
    std::string source_projection_id;
    std::string source_schedule_set_id;
    std::string source_global_dag_id;
    std::vector<ManifestInputDigestDto> input_digests;
    std::vector<LinkedFragmentDto> fragments;
    std::vector<FragmentInterfaceDto> fragment_interfaces;
    std::vector<CoreRuntimeBindingDto> core_bindings;
    std::vector<LinkedCoreStreamDto> core_streams;
    std::vector<RuntimeSymbolDefinitionDto> runtime_symbol_definitions;
    std::vector<ProgramSymbolDefinitionDto> program_symbol_definitions;
    std::vector<AddressOperandBindingDto> address_operand_bindings;
    std::vector<StateOperandBindingDto> state_operand_bindings;
    std::vector<LogicalCoreGroupDto> core_groups;
    ProgramControlEnvelopeDto envelope;
};

class ProgramArtifactFinalizer {
public:
    static LinkedProgramManifestDto Parse(std::string_view manifest_json);
    static std::string CanonicalManifestDigest(
        std::string_view manifest_json);
    static std::string EncodedArtifactDigest(
        const std::vector<uint8_t> &encoded_artifact);

    ProgramArtifact Finalize(const LinkedProgramManifestDto &manifest) const;
    ProgramArtifact FinalizeJson(std::string_view manifest_json) const;
    std::vector<uint8_t>
    FinalizeEncoded(std::string_view manifest_json) const;
};

} // namespace frontend
