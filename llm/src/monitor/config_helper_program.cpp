#include "monitor/config_helper_program.h"

#include "common/memory.h"
#include "dte/endpoint_contract.h"
#include "defs/spec.h"
#include "monitor/host_envelope.h"
#include "prims/dte_endpoint_prims.h"
#include "prims/collective_launch_v1_prim.h"
#include "prims/comp_prims.h"
#include "prims/norm_prims.h"
#include "prims/sram_lifecycle_prim.h"
#include "trace/Event_engine.h"
#include "utils/prim_utils.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string_view>
#include <tuple>
#include <utility>

namespace {

[[noreturn]] void Fail(const std::string &message) {
    throw ConfigHelperProgramError(message);
}

struct Utf8ByteLess {
    bool operator()(const std::string &left,
                    const std::string &right) const noexcept {
        return std::lexicographical_compare(
            left.begin(), left.end(), right.begin(), right.end(),
            [](char a, char b) {
                return static_cast<unsigned char>(a) <
                       static_cast<unsigned char>(b);
            });
    }
};

class LabelTableTransaction {
public:
    LabelTableTransaction() : original_(g_addr_label_table.table) {}

    ~LabelTableTransaction() {
        if (!committed_)
            g_addr_label_table.table.swap(original_);
    }

    void Commit() noexcept { committed_ = true; }

private:
    std::vector<std::string> original_;
    bool committed_ = false;
};

class PrimWireModeTransaction {
public:
    PrimWireModeTransaction()
        : original_(prim_wire::LegacyCompatibilityEnabled()) {
        prim_wire::SetLegacyCompatibility(false);
    }

    ~PrimWireModeTransaction() {
        if (!committed_)
            prim_wire::SetLegacyCompatibility(original_);
    }

    void Commit() noexcept { committed_ = true; }

private:
    bool original_ = false;
    bool committed_ = false;
};

uint64_t RelocatedValue(const ProgramSymbol &symbol, int64_t addend) {
    if (addend >= 0) {
        const uint64_t magnitude = static_cast<uint64_t>(addend);
        if (symbol.value > std::numeric_limits<uint64_t>::max() - magnitude)
            Fail("semantic relocation overflows u64");
        return symbol.value + magnitude;
    }
    const uint64_t magnitude = static_cast<uint64_t>(-(addend + 1)) + 1;
    if (symbol.value < magnitude)
        Fail("semantic relocation underflows u64");
    return symbol.value - magnitude;
}

uint64_t RegionOffset(int64_t addend) {
    if (addend < 0)
        Fail("SRAM_REGION relocation addend cannot be negative");
    return static_cast<uint64_t>(addend);
}

void SetAddress(SramAddressOperand &address,
                const SemanticRelocation &relocation,
                const ProgramSymbol &symbol) {
    switch (relocation.kind) {
    case SemanticRelocationKind::ABSOLUTE_ADDRESS:
        address.kind = SramAddressKind::ABSOLUTE;
        address.absolute_address_bytes =
            RelocatedValue(symbol, relocation.addend);
        address.region_symbol_index = 0;
        address.region_offset_bytes = 0;
        return;
    case SemanticRelocationKind::SRAM_REGION:
        address.kind = SramAddressKind::REGION;
        address.absolute_address_bytes = 0;
        address.region_symbol_index = relocation.symbol_index;
        // SRAM_REGION symbols carry the physical base in symbol.value. A
        // named-region operand carries only an offset within that region;
        // collective lowering adds the base exactly once when it needs a
        // canonical absolute address.
        address.region_offset_bytes = RegionOffset(relocation.addend);
        return;
    case SemanticRelocationKind::SRAM_LABEL:
        Fail("address relocation cannot target an SRAM_LABEL symbol");
    }
    Fail("unknown semantic relocation kind");
}

void RequireAbsolute(const SemanticRelocation &relocation,
                     std::string_view field) {
    if (relocation.kind != SemanticRelocationKind::ABSOLUTE_ADDRESS)
        Fail(std::string(field) + " requires ABSOLUTE_ADDRESS relocation");
}

uint64_t SymbolIndex(const SemanticRelocation &relocation,
                     std::string_view field) {
    if (relocation.addend != 0)
        Fail(std::string(field) + " symbolic relocation requires addend zero");
    return relocation.symbol_index;
}

void ApplyRelocation(ExternalRecord &record,
                     const SemanticRelocation &relocation,
                     const ProgramSymbol &symbol) {
    const auto operand =
        static_cast<SemanticOperandId>(relocation.operand_id);
    if (auto *value = std::get_if<ComputeOperands>(&record.operands)) {
        RequireAbsolute(relocation, "compute offset");
        const uint64_t relocated = RelocatedValue(symbol, relocation.addend);
        if (operand == SemanticOperandId::COMPUTE_INPUT_ADDRESS)
            value->input_offset_bytes = relocated;
        else if (operand == SemanticOperandId::COMPUTE_DATA_ADDRESS)
            value->data_offset_bytes = relocated;
        else if (operand == SemanticOperandId::COMPUTE_OUTPUT_ADDRESS)
            value->output_offset_bytes = relocated;
        else
            Fail("invalid compute relocation operand_id");
        return;
    }
    if (auto *value = std::get_if<RopeQkExactOperands>(&record.operands)) {
        RequireAbsolute(relocation, "ROPE_QK_EXACT address");
        if (operand == SemanticOperandId::COMPUTE_INPUT_ADDRESS)
            SetAddress(value->input, relocation, symbol);
        else if (operand == SemanticOperandId::COMPUTE_OUTPUT_ADDRESS)
            SetAddress(value->output, relocation, symbol);
        else
            Fail("invalid ROPE_QK_EXACT relocation operand_id");
        return;
    }
    if (auto *value =
            std::get_if<AttentionExactOperands>(&record.operands)) {
        RequireAbsolute(relocation, "ATTENTION_EXACT address");
        if (operand == SemanticOperandId::COMPUTE_INPUT_ADDRESS)
            SetAddress(value->input, relocation, symbol);
        else if (operand == SemanticOperandId::COMPUTE_OUTPUT_ADDRESS)
            SetAddress(value->output, relocation, symbol);
        else
            Fail("invalid ATTENTION_EXACT relocation operand_id");
        return;
    }
    if (auto *value =
            std::get_if<EmbeddingLookupOperands>(&record.operands)) {
        RequireAbsolute(relocation, "EMBEDDING_LOOKUP address");
        if (operand == SemanticOperandId::COMPUTE_INPUT_ADDRESS)
            SetAddress(value->indices, relocation, symbol);
        else if (operand == SemanticOperandId::COMPUTE_DATA_ADDRESS)
            SetAddress(value->table, relocation, symbol);
        else if (operand == SemanticOperandId::COMPUTE_OUTPUT_ADDRESS)
            SetAddress(value->output, relocation, symbol);
        else
            Fail("invalid EMBEDDING_LOOKUP relocation operand_id");
        return;
    }
    if (auto *value =
            std::get_if<GreedySampleOperands>(&record.operands)) {
        RequireAbsolute(relocation, "GREEDY_SAMPLE address");
        if (operand == SemanticOperandId::COMPUTE_INPUT_ADDRESS)
            SetAddress(value->logits, relocation, symbol);
        else if (operand == SemanticOperandId::COMPUTE_OUTPUT_ADDRESS)
            SetAddress(value->output, relocation, symbol);
        else
            Fail("invalid GREEDY_SAMPLE relocation operand_id");
        return;
    }
    if (auto *value =
            std::get_if<CrossEntropyForwardOperands>(&record.operands)) {
        RequireAbsolute(relocation, "CROSS_ENTROPY_FORWARD address");
        if (operand == SemanticOperandId::COMPUTE_INPUT_ADDRESS)
            SetAddress(value->logits, relocation, symbol);
        else if (operand == SemanticOperandId::COMPUTE_DATA_ADDRESS)
            SetAddress(value->labels, relocation, symbol);
        else if (operand == SemanticOperandId::COMPUTE_OUTPUT_ADDRESS)
            SetAddress(value->loss, relocation, symbol);
        else
            Fail("invalid CROSS_ENTROPY_FORWARD relocation operand_id");
        return;
    }
    if (auto *value =
            std::get_if<CrossEntropyBackwardOperands>(&record.operands)) {
        RequireAbsolute(relocation, "CROSS_ENTROPY_BACKWARD address");
        if (operand == SemanticOperandId::COMPUTE_INPUT_ADDRESS)
            SetAddress(value->logits, relocation, symbol);
        else if (operand == SemanticOperandId::COMPUTE_DATA_ADDRESS)
            SetAddress(value->labels, relocation, symbol);
        else if (operand == SemanticOperandId::COMPUTE_AUX_ADDRESS)
            SetAddress(value->upstream, relocation, symbol);
        else if (operand == SemanticOperandId::COMPUTE_OUTPUT_ADDRESS)
            SetAddress(value->logits_grad, relocation, symbol);
        else
            Fail("invalid CROSS_ENTROPY_BACKWARD relocation operand_id");
        return;
    }
    if (auto *value = std::get_if<SgdUpdateOperands>(&record.operands)) {
        RequireAbsolute(relocation, "SGD_UPDATE address");
        if (operand == SemanticOperandId::COMPUTE_INPUT_ADDRESS)
            SetAddress(value->weight, relocation, symbol);
        else if (operand == SemanticOperandId::COMPUTE_DATA_ADDRESS)
            SetAddress(value->gradient, relocation, symbol);
        else if (operand == SemanticOperandId::COMPUTE_OUTPUT_ADDRESS)
            SetAddress(value->updated_weight, relocation, symbol);
        else
            Fail("invalid SGD_UPDATE relocation operand_id");
        return;
    }
    if (auto *value = std::get_if<DteSendOperands>(&record.operands)) {
        if (value->source_space == EndpointSourceSpace::HBM) {
            if (operand != SemanticOperandId::HBM_ADDRESS)
                Fail("DTE_SEND relocation operand_id does not match "
                     "source_space");
            RequireAbsolute(relocation, "DTE_SEND HBM source");
            value->source.kind = SramAddressKind::ABSOLUTE;
            value->source.absolute_address_bytes =
                RelocatedValue(symbol, relocation.addend);
            value->source.region_symbol_index = 0;
            value->source.region_offset_bytes = 0;
            return;
        }
        if (operand != SemanticOperandId::SOURCE_ADDRESS)
            Fail("DTE_SEND relocation operand_id does not match "
                 "source_space");
        SetAddress(value->source, relocation, symbol);
        return;
    }
    if (auto *value = std::get_if<DteRecvOperands>(&record.operands)) {
        if (operand != SemanticOperandId::DESTINATION_ADDRESS)
            Fail("invalid DTE_RECV relocation operand_id");
        SetAddress(value->destination, relocation, symbol);
        return;
    }
    if (auto *value = std::get_if<ReduceComputeOperands>(&record.operands)) {
        if (operand == SemanticOperandId::SOURCE_ADDRESS)
            SetAddress(value->source, relocation, symbol);
        else if (operand == SemanticOperandId::DESTINATION_ADDRESS)
            SetAddress(value->destination, relocation, symbol);
        else
            Fail("invalid REDUCE_COMPUTE relocation operand_id");
        return;
    }
    if (auto *value = std::get_if<LocalReduceOperands>(&record.operands)) {
        RequireAbsolute(relocation, "LOCAL_REDUCE address");
        if (operand == SemanticOperandId::SOURCE_ADDRESS)
            SetAddress(value->source, relocation, symbol);
        else if (operand == SemanticOperandId::DESTINATION_ADDRESS)
            SetAddress(value->destination, relocation, symbol);
        else
            Fail("invalid LOCAL_REDUCE relocation operand_id");
        return;
    }
    if (auto *value = std::get_if<LsuOperands>(&record.operands)) {
        if (operand == SemanticOperandId::HBM_ADDRESS) {
            RequireAbsolute(relocation, "LSU HBM address");
            value->hbm_address_bytes =
                RelocatedValue(symbol, relocation.addend);
        } else if (operand == SemanticOperandId::SOURCE_ADDRESS ||
                   operand == SemanticOperandId::DESTINATION_ADDRESS) {
            SetAddress(value->sram, relocation, symbol);
        } else {
            Fail("invalid LSU relocation operand_id");
        }
        return;
    }
    if (auto *value = std::get_if<DteIssueOperands>(&record.operands)) {
        if (operand == SemanticOperandId::SOURCE_ADDRESS) {
            if (value->direction == LocalDteDirection::DRAM_TO_SPM)
                Fail("DTE_ISSUE DRAM_TO_SPM has no source SRAM relocation");
            SetAddress(value->source_sram, relocation, symbol);
        } else if (operand == SemanticOperandId::DESTINATION_ADDRESS) {
            if (value->direction == LocalDteDirection::SPM_TO_DRAM)
                Fail("DTE_ISSUE SPM_TO_DRAM has no destination SRAM relocation");
            SetAddress(value->destination_sram, relocation, symbol);
        } else if (operand == SemanticOperandId::HBM_ADDRESS) {
            if (value->direction == LocalDteDirection::SPM_TO_SPM)
                Fail("DTE_ISSUE SPM_TO_SPM has no HBM relocation");
            RequireAbsolute(relocation, "DTE HBM address");
            value->hbm_address_bytes =
                RelocatedValue(symbol, relocation.addend);
        } else
            Fail("invalid DTE_ISSUE relocation operand_id");
        return;
    }
    if (auto *value = std::get_if<SymbolOperands>(&record.operands)) {
        if (operand != SemanticOperandId::SYMBOL)
            Fail("invalid symbol relocation operand_id");
        if (relocation.kind != SemanticRelocationKind::SRAM_LABEL)
            Fail("SYMBOL relocation requires an SRAM_LABEL symbol");
        value->symbol_index = SymbolIndex(relocation, "SYMBOL");
        return;
    }
    if (auto *value = std::get_if<SramBindOperands>(&record.operands)) {
        const uint16_t raw_operand = relocation.operand_id;
        const uint16_t first_input = static_cast<uint16_t>(
            SemanticOperandId::SRAM_BIND_INPUT_0);
        const uint16_t output = static_cast<uint16_t>(
            SemanticOperandId::SRAM_BIND_OUTPUT);
        if (relocation.kind != SemanticRelocationKind::SRAM_LABEL)
            Fail("SRAM_BIND relocation requires an SRAM_LABEL symbol");
        if (raw_operand >= first_input && raw_operand < output) {
            const std::size_t slot = raw_operand - first_input;
            if (slot >= value->input_count)
                Fail("SRAM_BIND relocation targets an inactive input slot");
            value->input_symbol_indices[slot] =
                SymbolIndex(relocation, "SRAM_BIND input");
        } else if (raw_operand == output) {
            value->output_symbol_index =
                SymbolIndex(relocation, "SRAM_BIND output");
        } else {
            Fail("invalid SRAM_BIND relocation operand_id");
        }
        return;
    }
    if (auto *value = std::get_if<SramAllocOperands>(&record.operands)) {
        if (operand == SemanticOperandId::REGION_NAME) {
            if (relocation.kind != SemanticRelocationKind::SRAM_REGION)
                Fail("REGION_NAME relocation requires an SRAM_REGION symbol");
            if (relocation.addend != 0)
                Fail("REGION_NAME symbolic relocation requires addend zero");
            value->region_name_string_index = symbol.name_string_index;
        } else if (operand == SemanticOperandId::LABEL_SYMBOL) {
            if (relocation.kind != SemanticRelocationKind::SRAM_LABEL)
                Fail("LABEL_SYMBOL relocation requires an SRAM_LABEL symbol");
            value->label_symbol_index =
                SymbolIndex(relocation, "LABEL_SYMBOL");
        } else {
            Fail("invalid SRAM_ALLOC relocation operand_id");
        }
        return;
    }
    if (auto *value = std::get_if<SramAllocAtOperands>(&record.operands)) {
        if (operand == SemanticOperandId::REGION_NAME) {
            if (relocation.kind != SemanticRelocationKind::SRAM_REGION)
                Fail("REGION_NAME relocation requires an SRAM_REGION symbol");
            if (relocation.addend != 0)
                Fail("REGION_NAME symbolic relocation requires addend zero");
            value->region_name_string_index = symbol.name_string_index;
        } else if (operand == SemanticOperandId::LABEL_SYMBOL) {
            if (relocation.kind != SemanticRelocationKind::SRAM_LABEL)
                Fail("LABEL_SYMBOL relocation requires an SRAM_LABEL symbol");
            value->label_symbol_index =
                SymbolIndex(relocation, "LABEL_SYMBOL");
        } else {
            Fail("invalid SRAM_ALLOC_AT relocation operand_id");
        }
        return;
    }
    if (auto *value = std::get_if<SramResizeOperands>(&record.operands)) {
        if (operand != SemanticOperandId::SYMBOL)
            Fail("invalid SRAM_RESIZE relocation operand_id");
        if (relocation.kind != SemanticRelocationKind::SRAM_LABEL)
            Fail("SRAM_RESIZE relocation requires an SRAM_LABEL symbol");
        value->symbol_index = SymbolIndex(relocation, "SYMBOL");
        return;
    }
    if (auto *value = std::get_if<SramRenameOperands>(&record.operands)) {
        if (relocation.kind != SemanticRelocationKind::SRAM_LABEL)
            Fail("SRAM_RENAME relocation requires an SRAM_LABEL symbol");
        if (operand == SemanticOperandId::OLD_SYMBOL)
            value->old_symbol_index = SymbolIndex(relocation, "OLD_SYMBOL");
        else if (operand == SemanticOperandId::NEW_SYMBOL)
            value->new_symbol_index = SymbolIndex(relocation, "NEW_SYMBOL");
        else
            Fail("invalid SRAM_RENAME relocation operand_id");
        return;
    }
    Fail("record operand variant has no relocation mapping");
}

bool Contains(const std::vector<uint64_t> &values, uint64_t candidate) {
    return std::binary_search(values.begin(), values.end(), candidate);
}

std::vector<const ProgramStartEvent *>
StartEventsFor(const ProgramArtifact &artifact, int core_id) {
    std::vector<const ProgramStartEvent *> result;
    for (const ProgramStartEvent &event : artifact.envelope.start_events)
        if (event.target_core == static_cast<uint64_t>(core_id))
            result.push_back(&event);
    return result;
}

bool SameCoreSet(const std::set<int> &values,
                 const std::vector<uint64_t> &expected) {
    if (values.size() != expected.size()) return false;
    return std::equal(values.begin(), values.end(), expected.begin());
}

struct LifecycleState {
    SramLifetime lifetime = SramLifetime::TASK;
    bool spillable = false;
};

struct EndpointPairKey {
    uint32_t source_core = 0;
    uint32_t destination_core = 0;
    uint32_t fsm_id = 0;

    bool operator<(const EndpointPairKey &other) const noexcept {
        return std::tie(source_core, destination_core, fsm_id) <
               std::tie(other.source_core, other.destination_core,
                        other.fsm_id);
    }
};

struct EndpointPairMetadata {
    uint64_t length_bytes = 0;
    EndpointDataType datatype = EndpointDataType::UINT8;
    uint32_t executing_core = 0;
    std::size_t record_index = 0;
};

std::string EndpointPairDescription(const EndpointPairKey &key) {
    return "source=" + std::to_string(key.source_core) +
           " destination=" + std::to_string(key.destination_core) +
           " fsm_id=" + std::to_string(key.fsm_id);
}

void ValidatePlatformCoresAndGroups(const ProgramArtifact &artifact) {
    if (TOTAL_CORES <= 0 || CORES_PER_DIE <= 0)
        Fail("program helper requires initialized platform core dimensions");

    std::set<uint64_t> active_cores;
    for (const ProgramCore &core : artifact.cores) {
        if (core.core_id >= static_cast<uint64_t>(TOTAL_CORES))
            Fail("program core " + std::to_string(core.core_id) +
                 " is outside platform TOTAL_CORES=" +
                 std::to_string(TOTAL_CORES));
        active_cores.insert(core.core_id);
    }

    for (const ProgramCoreGroup &group : artifact.core_groups) {
        std::optional<int> group_die;
        for (uint64_t member : group.members) {
            if (member >= static_cast<uint64_t>(TOTAL_CORES))
                Fail("core group " + std::to_string(group.group_id) +
                     " member " + std::to_string(member) +
                     " is outside platform TOTAL_CORES=" +
                     std::to_string(TOTAL_CORES));
            if (active_cores.count(member) == 0)
                Fail("core group " + std::to_string(group.group_id) +
                     " member " + std::to_string(member) +
                     " is not an active platform program core");
            const int member_die =
                static_cast<int>(member) / CORES_PER_DIE;
            if (group_die.has_value() && *group_die != member_die)
                Fail("core group " + std::to_string(group.group_id) +
                     " spans multiple dies");
            group_die = member_die;
        }
    }
}

bool IsExternalCollectiveRecord(const ExternalRecord &record) {
    if (std::holds_alternative<ReduceComputeOperands>(record.operands))
        return true;
    if (const auto *send =
            std::get_if<DteSendOperands>(&record.operands))
        return send->mode != DteSendMode::P2P || send->group_id != 0;
    if (const auto *receive =
            std::get_if<DteRecvOperands>(&record.operands))
        return receive->mode != DteRecvMode::P2P ||
               receive->group_id != 0;
    return false;
}

bool HasExternalCollectiveRecords(const ProgramArtifact &artifact) {
    for (const ProgramCore &core : artifact.cores)
        for (const ExternalRecord &record : core.records)
            if (IsExternalCollectiveRecord(record)) return true;
    return false;
}

CollectiveLaunchV1Role LaunchRole(IsaV1CollectiveRecordRole role) {
    switch (role) {
    case IsaV1CollectiveRecordRole::SEND:
        return CollectiveLaunchV1Role::ISSUE_SEND;
    case IsaV1CollectiveRecordRole::RECEIVE:
        return CollectiveLaunchV1Role::ISSUE_RECEIVE;
    case IsaV1CollectiveRecordRole::REDUCE_COMPUTE:
        return CollectiveLaunchV1Role::DECLARE_REDUCE_COMPUTE;
    }
    Fail("collective issue site has an invalid role");
}

} // namespace

void ApplyProgramRelocations(ProgramArtifact &artifact) {
    ValidateProgramArtifact(artifact);
    for (const SemanticRelocation &relocation : artifact.relocations) {
        ExternalRecord &record =
            artifact.cores.at(static_cast<std::size_t>(relocation.core_index))
                .records.at(
                    static_cast<std::size_t>(relocation.instruction_index));
        const ProgramSymbol &symbol = artifact.symbols.at(
            static_cast<std::size_t>(relocation.symbol_index));
        ApplyRelocation(record, relocation, symbol);
    }
    // Re-run full reference/span validation after semantic relocation has
    // materialized named-region offsets. This closes the pre-relocation
    // default-address validation gap transactionally.
    ValidateProgramArtifact(artifact);
}

config_helper_program::config_helper_program(
    const std::vector<uint8_t> &artifact_bytes) {
    LoadProgram(artifact_bytes);
}

config_helper_program::config_helper_program(const ProgramArtifact &artifact) {
    LoadProgram(EncodeProgramArtifact(artifact));
}

void config_helper_program::LoadProgram(
    const std::vector<uint8_t> &artifact_bytes) {
    ProgramArtifact candidate = DecodeProgramArtifact(artifact_bytes);
    if (candidate.capabilities != 0)
        Fail("program helper disables GLOBAL/PD/experimental capabilities");
    ApplyProgramRelocations(candidate);
    ValidatePlatformCoresAndGroups(candidate);
    if (candidate.cores.empty())
        Fail("program helper requires at least one configured core");

    IsaV1CollectiveProgramImageConfig image_config;
    image_config.total_cores = static_cast<uint32_t>(TOTAL_CORES);
    image_config.cores_per_die = static_cast<uint32_t>(CORES_PER_DIE);
    image_config.generation = 1;
    image_config.planner_capacity.max_child_bytes =
        kDteEndpointP2pMaxBytes;
    image_config.planner_capacity.max_receive_bytes_per_rank_per_wave =
        kDteEndpointP2pMaxBytes;
    image_config.planner_capacity.max_sessions_per_rank_per_wave =
        static_cast<uint32_t>(MAX_BUFFER_PACKET_SIZE);
    image_config.limits.max_endpoint_sessions_per_core_wave =
        static_cast<uint32_t>(MAX_BUFFER_PACKET_SIZE);
    image_config.limits.max_receive_bytes_per_core_wave =
        kDteEndpointP2pMaxBytes;

    std::optional<IsaV1CollectiveArtifactLowering> collective_lowering;
    if (HasExternalCollectiveRecords(candidate)) {
        collective_lowering = LowerIsaV1CollectiveArtifact(
            candidate, image_config.total_cores, image_config.cores_per_die,
            image_config.planner_capacity);
        if (collective_lowering->plans.empty())
            Fail("collective records produced no whole-artifact plan");
    }

    std::vector<CoreGroupDefinition> candidate_group_definitions;
    candidate_group_definitions.reserve(candidate.core_groups.size());
    for (const ProgramCoreGroup &group : candidate.core_groups) {
        CoreGroupDefinition definition;
        definition.group_id = static_cast<uint32_t>(group.group_id);
        definition.members.reserve(group.members.size());
        for (uint64_t member : group.members)
            definition.members.push_back(static_cast<uint16_t>(member));
        candidate_group_definitions.push_back(std::move(definition));
    }
    auto candidate_group_registry =
        std::make_shared<const CoreGroupRegistry>(
            std::move(candidate_group_definitions),
            static_cast<uint32_t>(TOTAL_CORES),
            static_cast<uint32_t>(CORES_PER_DIE));

    std::vector<PreparedCore> candidate_cores;
    candidate_cores.reserve(candidate.cores.size());
    std::vector<CoreConfig> candidate_coreconfigs;
    candidate_coreconfigs.reserve(candidate.cores.size());
    const bool include_empty = candidate.envelope.empty_core_ack_policy ==
                               EmptyCoreAckPolicy::INCLUDE_EMPTY;
    LoweringContext context;
    context.enabled_capabilities = 0;
    context.resolve_symbol = [&candidate](uint32_t index)
        -> std::optional<std::string> {
        if (index >= candidate.symbols.size()) return std::nullopt;
        const ProgramSymbol &symbol = candidate.symbols[index];
        if (symbol.name_string_index >= candidate.strings.size())
            return std::nullopt;
        return candidate.strings[symbol.name_string_index];
    };
    context.resolve_string = [&candidate](uint32_t index)
        -> std::optional<std::string> {
        if (index >= candidate.strings.size()) return std::nullopt;
        return candidate.strings[index];
    };

    std::set<std::string, Utf8ByteLess> candidate_label_names;
    std::map<std::pair<uint32_t, uint16_t>, uint64_t>
        candidate_group_sync_counts;
    std::map<EventKey, uint64_t> candidate_event_sets;
    std::map<EventKey, uint64_t> candidate_event_waits;
    std::map<EndpointPairKey, std::vector<EndpointPairMetadata>>
        candidate_endpoint_sends;
    std::map<EndpointPairKey, std::vector<EndpointPairMetadata>>
        candidate_endpoint_recvs;
    using CollectiveIssueKey = std::pair<uint16_t, uint32_t>;
    std::map<CollectiveIssueKey, Collective_launch_v1_prim *>
        candidate_collective_launches;
    for (const ProgramCore &core : candidate.cores) {
        CoreConfig config;
        config.id = static_cast<int>(core.core_id);
        config.prim_copy = -1;
        config.send_global_mem = -1;
        config.loop = 1;
        candidate_coreconfigs.push_back(std::move(config));

        PreparedCore prepared;
        prepared.core_id = static_cast<int>(core.core_id);
        prepared.included = include_empty || !core.records.empty();
        const bool is_source =
            !StartEventsFor(candidate, prepared.core_id).empty();
        const bool is_terminal =
            Contains(candidate.envelope.terminal_cores, core.core_id);
        if (!prepared.included && (is_source || is_terminal))
            Fail("EXCLUDE_EMPTY core cannot be a source or terminal: " +
                 std::to_string(prepared.core_id));

        std::optional<uint32_t> pending_bind_count;
        std::optional<std::size_t> pending_bind_record;
        std::map<std::string, LifecycleState> lifecycle_labels;
        std::map<uint32_t, std::optional<uint32_t>> outstanding_dte_tokens;
        // Local TX completion can precede the peer ACK that retires the
        // transport tag, so v1 never reuses a TX fsm_id within an artifact.
        std::set<uint32_t> seen_tx_fsms;
        std::set<uint32_t> outstanding_rx_fsms;
        std::map<uint32_t, uint64_t> next_group_sync_sequence;
        for (std::size_t record_index = 0; record_index < core.records.size();
             ++record_index) {
            const ExternalRecord &record = core.records[record_index];
            if (record.opcode == Opcode::DTE_SEND) {
                const auto &operands =
                    std::get<DteSendOperands>(record.operands);
                if (operands.mode == DteSendMode::P2P &&
                    operands.group_id == 0) {
                    if (operands.peer_core == core.core_id)
                        Fail("core " +
                             std::to_string(prepared.core_id) +
                             " record " + std::to_string(record_index) +
                             " DTE_SEND P2P self peer is forbidden; use "
                             "DTE_ISSUE SPM_TO_SPM for local copy");
                    if (!Contains(candidate.envelope.active_cores,
                                  operands.peer_core))
                        Fail("core " + std::to_string(prepared.core_id) +
                             " record " + std::to_string(record_index) +
                             " DTE_SEND peer_core is not an active program "
                             "core: " +
                             std::to_string(operands.peer_core));
                    const EndpointPairKey key{
                        static_cast<uint32_t>(core.core_id),
                        static_cast<uint32_t>(operands.peer_core),
                        static_cast<uint32_t>(operands.fsm_id)};
                    const EndpointPairMetadata metadata{
                        operands.length_bytes, operands.datatype,
                        static_cast<uint32_t>(core.core_id), record_index};
                    candidate_endpoint_sends[key].push_back(metadata);
                }
            } else if (record.opcode == Opcode::DTE_RECV) {
                const auto &operands =
                    std::get<DteRecvOperands>(record.operands);
                if (operands.mode == DteRecvMode::P2P &&
                    operands.group_id == 0) {
                    if (operands.peer_core == core.core_id)
                        Fail("core " +
                             std::to_string(prepared.core_id) +
                             " record " + std::to_string(record_index) +
                             " DTE_RECV P2P self peer is forbidden; use "
                             "DTE_ISSUE SPM_TO_SPM for local copy");
                    if (!Contains(candidate.envelope.active_cores,
                                  operands.peer_core))
                        Fail("core " + std::to_string(prepared.core_id) +
                             " record " + std::to_string(record_index) +
                             " DTE_RECV peer_core is not an active program "
                             "core: " +
                             std::to_string(operands.peer_core));
                    const EndpointPairKey key{
                        static_cast<uint32_t>(operands.peer_core),
                        static_cast<uint32_t>(core.core_id),
                        static_cast<uint32_t>(operands.fsm_id)};
                    const EndpointPairMetadata metadata{
                        operands.length_bytes, operands.datatype,
                        static_cast<uint32_t>(core.core_id), record_index};
                    candidate_endpoint_recvs[key].push_back(metadata);
                }
            } else if (record.opcode == Opcode::GROUP_SYNC) {
                const auto &operands =
                    std::get<GroupSyncOperands>(record.operands);
                const uint32_t group_id =
                    static_cast<uint32_t>(operands.group_id);
                if (!candidate_group_registry->Contains(group_id))
                    Fail("core " + std::to_string(prepared.core_id) +
                         " GROUP_SYNC references unknown group " +
                         std::to_string(group_id));
                try {
                    (void)candidate_group_registry->RankOf(
                        group_id, static_cast<uint16_t>(prepared.core_id));
                } catch (const std::exception &) {
                    Fail("core " + std::to_string(prepared.core_id) +
                         " executes GROUP_SYNC for nonmember group " +
                         std::to_string(group_id));
                }
                uint64_t &next = next_group_sync_sequence[group_id];
                if (operands.sync_seq != next)
                    Fail("core " + std::to_string(prepared.core_id) +
                         " GROUP_SYNC group " + std::to_string(group_id) +
                         " has out-of-order sync_seq=" +
                         std::to_string(operands.sync_seq) +
                         ", expected=" + std::to_string(next));
                ++next;
                ++candidate_group_sync_counts[
                    {group_id, static_cast<uint16_t>(prepared.core_id)}];
            } else if (record.opcode == Opcode::EVENT_SET) {
                const auto &operands =
                    std::get<EventSetOperands>(record.operands);
                if (operands.source_core != core.core_id)
                    Fail("EVENT_SET source_core must equal its executing core");
                if (!Contains(candidate.envelope.active_cores,
                              operands.source_core) ||
                    !Contains(candidate.envelope.active_cores,
                              operands.destination_core))
                    Fail("EVENT_SET endpoints must both be active program cores");
                const EventKey key{
                    static_cast<uint16_t>(operands.source_core),
                    static_cast<uint16_t>(operands.destination_core),
                    static_cast<uint32_t>(operands.tag)};
                ++candidate_event_sets[key];
            } else if (record.opcode == Opcode::EVENT_WAIT) {
                const auto &operands =
                    std::get<EventWaitOperands>(record.operands);
                if (operands.destination_core != core.core_id)
                    Fail("EVENT_WAIT destination_core must equal its executing core");
                if (!Contains(candidate.envelope.active_cores,
                              operands.source_core) ||
                    !Contains(candidate.envelope.active_cores,
                              operands.destination_core))
                    Fail("EVENT_WAIT endpoints must both be active program cores");
                const EventKey key{
                    static_cast<uint16_t>(operands.source_core),
                    static_cast<uint16_t>(operands.destination_core),
                    static_cast<uint32_t>(operands.tag)};
                uint64_t &waits = candidate_event_waits[key];
                if (operands.count >
                    std::numeric_limits<uint64_t>::max() - waits)
                    Fail("EVENT_WAIT aggregate count overflows uint64_t");
                waits += operands.count;
            }
            LoweredPrimList lowered;
            if (IsExternalCollectiveRecord(record)) {
                auto launch =
                    std::make_unique<Collective_launch_v1_prim>();
                const auto inserted = candidate_collective_launches.emplace(
                    CollectiveIssueKey{
                        static_cast<uint16_t>(prepared.core_id),
                        static_cast<uint32_t>(record_index)},
                    launch.get());
                if (!inserted.second)
                    Fail("collective launch issue site is duplicated");
                lowered.push_back(std::move(launch));
            } else {
                lowered = LowerExternalRecord(record, context);
            }
            const OpcodeManifestEntry *entry = LookupOpcode(record.opcode);
            if (entry == nullptr)
                Fail("validated program record has no opcode manifest entry");

            if (record.opcode == Opcode::SRAM_BIND) {
                if (pending_bind_count.has_value())
                    Fail("core " + std::to_string(prepared.core_id) +
                         " has two SRAM_BIND records before a compute");
                if (lowered.size() != 1)
                    Fail("SRAM_BIND must lower to exactly one Prim");
                auto *bind = dynamic_cast<Sram_bind_oneshot *>(
                    lowered.front().get());
                if (bind == nullptr)
                    Fail("SRAM_BIND lowered to the wrong Prim type");
                pending_bind_count = bind->input_count;
                pending_bind_record = record_index;
                for (std::size_t i = 0; i < bind->input_count; ++i)
                    candidate_label_names.insert(
                        bind->datapass_label.indata[i]);
                candidate_label_names.insert(bind->datapass_label.outdata);
            } else if (entry->category == OpcodeCategory::COMPUTE) {
                if (!pending_bind_count.has_value())
                    Fail("core " + std::to_string(prepared.core_id) +
                         " compute record " + std::to_string(record_index) +
                         " is missing a preceding one-shot SRAM_BIND");
                if (lowered.size() != 1)
                    Fail("compute record must lower to exactly one Prim");
                auto *compute = dynamic_cast<NpuBase *>(lowered.front().get());
                if (compute == nullptr)
                    Fail("compute record lowered to a non-NpuBase Prim");
                if (record.opcode == Opcode::MATMUL &&
                    *pending_bind_count == 2) {
                    auto *matmul = dynamic_cast<Matmul_f *>(compute);
                    if (matmul == nullptr)
                        Fail("MATMUL lowered to the wrong Prim type");
                    matmul->enableProgramTwoInputMode();
                }
                compute->initialize();
                const std::size_t actual_inputs =
                    compute->data_size_input.size();
                if (actual_inputs != *pending_bind_count)
                    Fail("core " + std::to_string(prepared.core_id) +
                         " SRAM_BIND record " +
                         std::to_string(*pending_bind_record) +
                         " input_count=" +
                         std::to_string(*pending_bind_count) +
                         " does not match compute record " +
                         std::to_string(record_index) + " input count=" +
                         std::to_string(actual_inputs));
                pending_bind_count.reset();
                pending_bind_record.reset();
            }

            if (auto *lifecycle =
                    lowered.empty()
                        ? nullptr
                        : dynamic_cast<Sram_lifecycle *>(
                              lowered.front().get())) {
                if (!lifecycle->region_name.empty())
                    candidate_label_names.insert(lifecycle->region_name);
                candidate_label_names.insert(lifecycle->label);
                if (!lifecycle->new_label.empty())
                    candidate_label_names.insert(lifecycle->new_label);

                const std::string location =
                    "core " + std::to_string(prepared.core_id) +
                    " SRAM lifecycle record " +
                    std::to_string(record_index) + ": ";
                switch (record.opcode) {
                case Opcode::SRAM_ALLOC: {
                    const auto &operands =
                        std::get<SramAllocOperands>(record.operands);
                    const auto inserted = lifecycle_labels.emplace(
                        lifecycle->label,
                        LifecycleState{operands.lifetime,
                                       operands.spillable});
                    if (!inserted.second)
                        Fail(location + "duplicate ALLOC label " +
                             lifecycle->label);
                    break;
                }
                case Opcode::SRAM_ALLOC_AT: {
                    const auto &operands =
                        std::get<SramAllocAtOperands>(record.operands);
                    const auto inserted = lifecycle_labels.emplace(
                        lifecycle->label,
                        LifecycleState{operands.lifetime,
                                       operands.spillable});
                    if (!inserted.second)
                        Fail(location + "duplicate ALLOC label " +
                             lifecycle->label);
                    break;
                }
                case Opcode::SRAM_RESIZE:
                    if (lifecycle_labels.count(lifecycle->label) == 0)
                        Fail(location + "RESIZE references unknown label " +
                             lifecycle->label);
                    break;
                case Opcode::SRAM_RENAME: {
                    const auto found =
                        lifecycle_labels.find(lifecycle->label);
                    if (found == lifecycle_labels.end())
                        Fail(location + "RENAME references unknown label " +
                             lifecycle->label);
                    if (lifecycle_labels.count(lifecycle->new_label) != 0)
                        Fail(location + "RENAME destination already exists: " +
                             lifecycle->new_label);
                    const LifecycleState state = found->second;
                    lifecycle_labels.erase(found);
                    lifecycle_labels.emplace(lifecycle->new_label, state);
                    break;
                }
                case Opcode::SRAM_CLEAR: {
                    const auto found =
                        lifecycle_labels.find(lifecycle->label);
                    if (found == lifecycle_labels.end())
                        Fail(location + "CLEAR references unknown label " +
                             lifecycle->label);
                    if (found->second.lifetime != SramLifetime::TASK ||
                        !found->second.spillable)
                        Fail(location +
                             "CLEAR requires a spillable TASK allocation");
                    lifecycle_labels.erase(found);
                    break;
                }
                case Opcode::SRAM_FREE:
                    if (lifecycle_labels.erase(lifecycle->label) != 1)
                        Fail(location + "FREE references unknown label " +
                             lifecycle->label);
                    break;
                default:
                    break;
                }
            }

            if (auto *send =
                    lowered.empty()
                        ? nullptr
                        : dynamic_cast<Dte_send_endpoint_prim *>(
                              lowered.front().get())) {
                if (!send->source.region.empty())
                    candidate_label_names.insert(send->source.region);
            } else if (auto *recv =
                           lowered.empty()
                               ? nullptr
                               : dynamic_cast<Dte_recv_endpoint_prim *>(
                                     lowered.front().get())) {
                if (!recv->destination.region.empty())
                    candidate_label_names.insert(recv->destination.region);
            }

            if (record.opcode == Opcode::DTE_ISSUE) {
                const auto token = static_cast<uint32_t>(
                    std::get<DteIssueOperands>(record.operands).token);
                if (!outstanding_dte_tokens.emplace(token, std::nullopt)
                         .second)
                    Fail("core " + std::to_string(prepared.core_id) +
                         " DTE_ISSUE reuses outstanding token " +
                         std::to_string(token));
            } else if (record.opcode == Opcode::DTE_SEND ||
                       record.opcode == Opcode::DTE_RECV) {
                const bool is_send = record.opcode == Opcode::DTE_SEND;
                const bool aggregate_endpoint =
                    IsExternalCollectiveRecord(record);
                const EndpointCompletion completion =
                    is_send
                        ? std::get<DteSendOperands>(record.operands).completion
                        : std::get<DteRecvOperands>(record.operands).completion;
                const uint32_t fsm_id = static_cast<uint32_t>(
                    is_send
                        ? std::get<DteSendOperands>(record.operands).fsm_id
                        : std::get<DteRecvOperands>(record.operands).fsm_id);
                const uint32_t token = static_cast<uint32_t>(
                    is_send
                        ? std::get<DteSendOperands>(record.operands).token
                        : std::get<DteRecvOperands>(record.operands).token);
                if (!aggregate_endpoint) {
                    if (is_send && outstanding_rx_fsms.count(fsm_id) != 0)
                        Fail("core " + std::to_string(prepared.core_id) +
                             " DTE_SEND reuses outstanding DTE_RECV fsm_id " +
                             std::to_string(fsm_id));
                    if (is_send && !seen_tx_fsms.insert(fsm_id).second)
                        Fail("core " + std::to_string(prepared.core_id) + " " +
                             "reuses DTE_SEND fsm_id " +
                             std::to_string(fsm_id) +
                             " within one artifact before peer ACK-safe "
                             "retirement");
                    if (!is_send && seen_tx_fsms.count(fsm_id) != 0)
                        Fail("core " + std::to_string(prepared.core_id) +
                             " DTE_RECV reuses a prior DTE_SEND fsm_id " +
                             std::to_string(fsm_id) +
                             " before peer ACK-safe retirement");
                    if (!is_send && outstanding_rx_fsms.count(fsm_id) != 0)
                        Fail("core " + std::to_string(prepared.core_id) +
                             " DTE_RECV reuses outstanding fsm_id " +
                             std::to_string(fsm_id));
                }
                if (completion == EndpointCompletion::ASYNC) {
                    if (outstanding_dte_tokens.count(token) != 0)
                        Fail("core " + std::to_string(prepared.core_id) +
                             " " + (is_send ? "DTE_SEND" : "DTE_RECV") +
                             " reuses outstanding DTE token " +
                             std::to_string(token));
                    if (aggregate_endpoint || is_send) {
                        outstanding_dte_tokens.emplace(token, std::nullopt);
                    } else {
                        outstanding_rx_fsms.insert(fsm_id);
                        outstanding_dte_tokens.emplace(token, fsm_id);
                    }
                }
            } else if (record.opcode == Opcode::DTE_WAIT ||
                       record.opcode == Opcode::DTE_CANCEL) {
                const auto token = static_cast<uint32_t>(
                    std::get<TokenOperands>(record.operands).token);
                const auto outstanding = outstanding_dte_tokens.find(token);
                if (outstanding == outstanding_dte_tokens.end())
                    Fail("core " + std::to_string(prepared.core_id) + " " +
                         std::string(record.opcode == Opcode::DTE_WAIT
                                         ? "DTE_WAIT"
                                         : "DTE_CANCEL") +
                         " references an unknown or completed token " +
                         std::to_string(token));
                if (record.opcode == Opcode::DTE_CANCEL &&
                    outstanding->second.has_value())
                    Fail("core " + std::to_string(prepared.core_id) +
                         " DTE_CANCEL does not support a DTE_RECV token " +
                         std::to_string(token));
                if (outstanding->second.has_value())
                    outstanding_rx_fsms.erase(*outstanding->second);
                outstanding_dte_tokens.erase(outstanding);
            } else if (record.opcode == Opcode::DTE_FENCE) {
                outstanding_dte_tokens.clear();
                outstanding_rx_fsms.clear();
            }

            for (auto &prim : lowered)
                prepared.prims.push_back(std::move(prim));
        }
        if (pending_bind_count.has_value())
            Fail("core " + std::to_string(prepared.core_id) +
                 " has a dangling SRAM_BIND at record " +
                 std::to_string(*pending_bind_record));
        if (!outstanding_dte_tokens.empty())
            Fail("core " + std::to_string(prepared.core_id) +
                 " has dangling DTE token " +
                 std::to_string(outstanding_dte_tokens.begin()->first));
        for (const auto &[label, state] : lifecycle_labels) {
            if (state.lifetime != SramLifetime::PERSISTENT)
                Fail("core " + std::to_string(prepared.core_id) +
                     " has dangling non-persistent SRAM allocation " + label);
        }
        candidate_cores.push_back(std::move(prepared));
    }

    for (const ProgramCoreGroup &group : candidate.core_groups) {
        bool used = false;
        std::optional<uint64_t> expected;
        for (uint64_t member : group.members) {
            const uint64_t count = candidate_group_sync_counts[
                {static_cast<uint32_t>(group.group_id),
                 static_cast<uint16_t>(member)}];
            used = used || count != 0;
            if (!expected.has_value())
                expected = count;
            else if (*expected != count)
                Fail("GROUP_SYNC group " + std::to_string(group.group_id) +
                     " members execute different synchronization counts");
        }
        if (!used) continue;
        if (!expected.has_value() || *expected == 0)
            Fail("used GROUP_SYNC group has no complete member sequence");
    }

    std::set<EventKey> event_keys;
    for (const auto &entry : candidate_event_sets)
        event_keys.insert(entry.first);
    for (const auto &entry : candidate_event_waits)
        event_keys.insert(entry.first);
    for (const EventKey &key : event_keys) {
        const uint64_t sets = candidate_event_sets[key];
        const uint64_t waits = candidate_event_waits[key];
        if (sets != waits)
            Fail("EVENT credit imbalance for source=" +
                 std::to_string(key.source) + " destination=" +
                 std::to_string(key.destination) + " tag=" +
                 std::to_string(key.tag) + ": SET=" +
                 std::to_string(sets) + " WAIT=" +
                 std::to_string(waits));
    }

    for (const auto &[key, sends] : candidate_endpoint_sends) {
        const auto recvs = candidate_endpoint_recvs.find(key);
        if (recvs == candidate_endpoint_recvs.end())
            Fail("DTE_SEND P2P has no matching DTE_RECV for " +
                 EndpointPairDescription(key));
        if (sends.size() != recvs->second.size())
            Fail("DTE P2P duplicate/missing endpoint count mismatch for " +
                 EndpointPairDescription(key));
        for (std::size_t index = 0; index < sends.size(); ++index) {
            if (sends[index].length_bytes !=
                recvs->second[index].length_bytes)
                Fail("DTE P2P length mismatch for " +
                     EndpointPairDescription(key) + " occurrence=" +
                     std::to_string(index));
            if (sends[index].datatype != recvs->second[index].datatype)
                Fail("DTE P2P datatype mismatch for " +
                     EndpointPairDescription(key) + " occurrence=" +
                     std::to_string(index));
        }
    }
    for (const auto &[key, recvs] : candidate_endpoint_recvs) {
        (void)recvs;
        if (candidate_endpoint_sends.count(key) == 0)
            Fail("DTE_RECV P2P has no matching DTE_SEND for " +
                 EndpointPairDescription(key));
    }

    // START fields are wider than the existing Recv_prim wire. Reject rather
    // than truncate; BuildStartMessages emits exactly count completion packets.
    for (const ProgramStartEvent &event : candidate.envelope.start_events) {
        if (event.tag > 0xffff)
            Fail("start event tag exceeds Recv_prim 16-bit wire");
        if (event.count > 0xff)
            Fail("start event count exceeds Recv_prim 8-bit wire");
    }

    std::set<int> candidate_ack;
    std::set<int> candidate_done;
    for (uint64_t core : candidate.envelope.expected_ack_cores)
        candidate_ack.insert(static_cast<int>(core));
    for (uint64_t core : candidate.envelope.expected_done_cores)
        candidate_done.insert(static_cast<int>(core));
    if (!SameCoreSet(candidate_ack,
                     candidate.envelope.expected_ack_cores) ||
        !SameCoreSet(candidate_done,
                     candidate.envelope.expected_done_cores))
        Fail("envelope ACK/DONE core set cannot fit runtime IDs");
    if (candidate_done.empty())
        Fail("DATAFLOW program requires at least one terminal/DONE core");
    if (candidate_ack.empty())
        Fail("program helper requires at least one expected ACK core");

    std::size_t new_label_count = 0;
    for (const std::string &label : candidate_label_names) {
        if (std::find(g_addr_label_table.table.begin(),
                      g_addr_label_table.table.end(), label) ==
            g_addr_label_table.table.end())
            ++new_label_count;
    }
    const std::size_t max_label_count =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (g_addr_label_table.table.size() > max_label_count ||
        new_label_count > max_label_count - g_addr_label_table.table.size())
        Fail("program label table would exceed its internal ID range");

    PrimWireModeTransaction wire_mode_transaction;
    std::shared_ptr<const IsaV1CollectiveProgramImage>
        candidate_collective_image;
    std::shared_ptr<const IsaV1CollectiveProfileProgramImage>
        candidate_collective_profile_image;
    if (collective_lowering.has_value()) {
        candidate_collective_image =
            std::make_shared<const IsaV1CollectiveProgramImage>(
                BuildIsaV1CollectiveProgramImage(
                    candidate, *collective_lowering, image_config));
        if (candidate_collective_image->IssueSites().size() !=
            candidate_collective_launches.size())
            Fail("collective image issue-site/launch count mismatch");
        std::set<CollectiveIssueKey> assigned;
        for (const IsaV1CollectiveIssueSite &site :
             candidate_collective_image->IssueSites()) {
            const CollectiveIssueKey key{site.core_id, site.record_index};
            const auto found = candidate_collective_launches.find(key);
            if (found == candidate_collective_launches.end() ||
                found->second == nullptr || !assigned.insert(key).second)
                Fail("collective image issue site has no unique launch Prim");
            Collective_launch_v1_prim &launch = *found->second;
            launch.role = LaunchRole(site.role);
            launch.image_generation =
                candidate_collective_image->Generation();
            launch.plan_index = site.plan_index;
            launch.external_record_index = site.record_index;
            launch.expected_core = site.core_id;
            launch.key = site.key;
            launch.public_token = site.public_token;
            launch.Validate();
        }
        if (assigned.size() != candidate_collective_launches.size())
            Fail("collective launch Prim remains unassigned");

        IsaV1CollectiveProfileImageConfig profile_config;
        profile_config.mesh = {
            static_cast<uint16_t>(GRID_X),
            static_cast<uint16_t>(GRID_Y),
            static_cast<uint16_t>(DIE_COUNT)};
        profile_config.noc = SPEC_NOC_COLL_CONFIG;
        if (SPEC_NOC_COLL_CONFIG.max_trees_per_batch.has_value())
            profile_config.max_trees_per_batch =
                *SPEC_NOC_COLL_CONFIG.max_trees_per_batch;
        /* P7 production paths carry strict multicast bytes and exact integer
         * DCA bytes through real SRAM. Requested backends remain fail-fast at
         * image construction if their wire, geometry, or die constraints do
         * not fit; there is no endpoint fallback. */
        profile_config.capabilities.multicast = true;
        profile_config.capabilities.dca = true;
        profile_config.capabilities.reduce_scatter_dca = false;
        candidate_collective_profile_image =
            std::make_shared<const IsaV1CollectiveProfileProgramImage>(
                BuildIsaV1CollectiveProfileProgramImage(
                    *candidate_collective_image, profile_config));
    } else if (!candidate_collective_launches.empty()) {
        Fail("collective launch Prim exists without an image");
    }

    // All artifact, platform, sequence, envelope, and label-capacity checks are
    // complete. Intern only now, in deterministic UTF-8 byte order. The
    // transaction restores the exact prior table if any final wire validation
    // or commit operation throws.
    LabelTableTransaction label_transaction;
    for (const std::string &label : candidate_label_names) {
        const int id = g_addr_label_table.addRecord(label);
        if (id <= 0)
            Fail("program label interning returned an invalid ID");
    }

    // Force every internal wire before commit. This catches target factory or
    // codec failures without exposing a partially replaced helper.
    for (PreparedCore &core : candidate_cores) {
        if (!core.included) continue;
        Recv_prim recv_weight(RECV_WEIGHT, core.core_id, 0);
        (void)recv_weight.serialize();
        for (const ProgramStartEvent *event : StartEventsFor(candidate,
                                                              core.core_id)) {
            Recv_prim recv_start(RECV_START, static_cast<int>(event->tag),
                                 static_cast<int>(event->count));
            (void)recv_start.serialize();
        }
        for (const auto &prim : core.prims)
            (void)prim->serialize();
        if (Contains(candidate.envelope.terminal_cores, core.core_id)) {
            Send_prim done(SEND_DONE);
            (void)done.serialize();
        }
    }

    std::vector<uint8_t> candidate_bytes = artifact_bytes;

    artifact_bytes_.swap(candidate_bytes);
    artifact_ = std::move(candidate);
    prepared_cores_.swap(candidate_cores);
    expected_ack_cores_.swap(candidate_ack);
    expected_done_cores_.swap(candidate_done);
    coreconfigs.swap(candidate_coreconfigs);
    core_group_registry_.swap(candidate_group_registry);
    collective_program_image_.swap(candidate_collective_image);
    collective_profile_program_image_.swap(
        candidate_collective_profile_image);
    ack_phase_.reset();
    seen_ack_cores_.clear();
    seen_done_cores_.clear();
    g_recv_ack_cnt = 0;
    g_recv_done_cnt = 0;
    end_cores = static_cast<int>(expected_done_cores_.size());
    pipeline = 1;
    end_count_sources = 1;
    prim_wire::SetLegacyCompatibility(false);
    label_transaction.Commit();
    wire_mode_transaction.Commit();
}

std::vector<HostEnvelope> config_helper_program::BuildConfigMessages() {
    std::vector<HostEnvelope> result;
    for (PreparedCore &core : prepared_cores_) {
        if (!core.included) continue;
        std::vector<PrimBase *> stream;
        Recv_prim recv_weight(RECV_WEIGHT, core.core_id, 0);
        stream.push_back(&recv_weight);
        std::vector<std::unique_ptr<Recv_prim>> starts;
        for (const ProgramStartEvent *event :
             StartEventsFor(artifact_, core.core_id)) {
            starts.push_back(std::make_unique<Recv_prim>(
                RECV_START, static_cast<int>(event->tag),
                static_cast<int>(event->count)));
            stream.push_back(starts.back().get());
        }
        for (auto &prim : core.prims)
            stream.push_back(prim.get());
        std::unique_ptr<Send_prim> done;
        if (Contains(artifact_.envelope.terminal_cores, core.core_id)) {
            done = std::make_unique<Send_prim>(SEND_DONE);
            stream.push_back(done.get());
        }

        int sequence = 1;
        std::size_t emitted = 0;
        std::size_t total_segments = 0;
        std::vector<std::vector<sc_bv<128>>> wires;
        wires.reserve(stream.size());
        for (PrimBase *prim : stream) {
            wires.push_back(prim->serialize());
            total_segments += wires.back().size();
        }
        for (const auto &wire : wires) {
            for (std::size_t segment = 0; segment < wire.size(); ++segment) {
                Msg message(false, CONFIG, sequence++, core.core_id,
                            segment + 1 == wire.size(), wire[segment]);
                ++emitted;
                if (emitted == total_segments) {
                    message.is_end_ = true;
                    message.refill_ = true;
                }
                result.push_back({core.core_id, message});
            }
        }
    }
    return result;
}

std::vector<HostEnvelope> config_helper_program::BuildStartMessages() const {
    std::vector<HostEnvelope> result;
    for (const ProgramStartEvent &event : artifact_.envelope.start_events) {
        for (uint64_t sequence = 1; sequence <= event.count; ++sequence) {
            sc_bv<128> payload = 0;
            Msg message(true, S_DATA, static_cast<int>(sequence),
                        static_cast<int>(event.target_core), 0,
                        static_cast<int>(event.tag), 0, payload);
            message.source_ = HOST_ENDPOINT_ID;
            message.roofline_packets_ = 1;
            result.push_back(
                {static_cast<int>(event.target_core), message});
        }
    }
    return result;
}

std::vector<HostEnvelope> config_helper_program::BuildDataMessages() const {
    std::vector<HostEnvelope> result;
    for (int core : expected_ack_cores_) {
        sc_bv<128> payload = 0;
        Msg message(true, P_DATA, 1, core, 0xffff, core, 0, payload);
        message.source_ = HOST_ENDPOINT_ID;
        message.roofline_packets_ = 1;
        result.push_back({core, message});
    }
    return result;
}

void config_helper_program::fill_queue_config(std::queue<Msg> *queue) {
    LegacyHostEnqueue(BuildConfigMessages(), queue);
}

void config_helper_program::fill_queue_start(std::queue<Msg> *queue) {
    LegacyHostEnqueue(BuildStartMessages(), queue);
}

void config_helper_program::fill_queue_data(std::queue<Msg> *queue) {
    LegacyHostEnqueue(BuildDataMessages(), queue);
}

bool config_helper_program::AcceptAck(const Msg &message, int phase_id) {
    if (message.msg_type_ != ACK)
        Fail("ACK parser received a non-ACK message");
    if (expected_ack_cores_.count(message.source_) == 0)
        Fail("unexpected ACK from core " + std::to_string(message.source_));
    if (!ack_phase_.has_value()) {
        ack_phase_ = phase_id;
    } else if (*ack_phase_ != phase_id &&
               seen_ack_cores_.size() == expected_ack_cores_.size()) {
        // MemInterface flow IDs advance when host CONFIG injection drains,
        // which can precede a late core CONFIG ACK. Do not discard an
        // incomplete ACK set merely because that trace-only ID advanced. A
        // new ACK round begins only after the previous complete set.
        ack_phase_ = phase_id;
        seen_ack_cores_.clear();
    }
    if (!seen_ack_cores_.insert(message.source_).second)
        Fail("duplicate ACK from core " + std::to_string(message.source_));
    g_recv_ack_cnt = static_cast<int>(seen_ack_cores_.size());
    return seen_ack_cores_.size() == expected_ack_cores_.size();
}

bool config_helper_program::AcceptDone(const Msg &message) {
    if (message.msg_type_ != DONE)
        Fail("DONE parser received a non-DONE message");
    if (expected_done_cores_.count(message.source_) == 0)
        Fail("unexpected DONE from core " + std::to_string(message.source_));
    if (!seen_done_cores_.insert(message.source_).second)
        Fail("duplicate DONE from core " + std::to_string(message.source_));
    g_recv_done_cnt = static_cast<int>(seen_done_cores_.size());
    return seen_done_cores_.size() == expected_done_cores_.size();
}

void config_helper_program::parse_ack_msg(Event_engine *event_engine,
                                          int flow_id,
                                          sc_event *notify_event) {
    bool complete = expected_ack_cores_.empty();
    for (const Msg &message : g_temp_ack_msg)
        complete = AcceptAck(message, flow_id);
    g_temp_ack_msg.clear();
    if (complete && notify_event != nullptr)
        notify_event->notify(CYCLE, SC_NS);
    if (event_engine != nullptr && complete)
        event_engine->add_event(name(), "Program ACK set complete", "i",
                                Trace_event_util());
}

void config_helper_program::parse_done_msg(Event_engine *event_engine,
                                           sc_event *notify_event) {
    bool complete = expected_done_cores_.empty();
    for (const Msg &message : g_temp_done_msg)
        complete = AcceptDone(message);
    g_temp_done_msg.clear();
    if (complete && notify_event != nullptr)
        notify_event->notify(CYCLE, SC_NS);
    if (event_engine != nullptr && complete)
        event_engine->add_event(name(), "Program DONE set complete", "i",
                                Trace_event_util());
    if (complete)
        sc_stop();
}

void config_helper_program::generate_prims(int index) {
    if (index < 0 || static_cast<std::size_t>(index) >= prepared_cores_.size())
        Fail("program core index is out of range");
}

void config_helper_program::printSelf() {
    std::cout << "Program config helper: " << prepared_cores_.size()
              << " active cores, " << expected_ack_cores_.size()
              << " ACKs, " << expected_done_cores_.size() << " DONEs\n";
}

config_helper_program *config_helper_program::clone() const {
    auto *cloned = new config_helper_program(artifact_bytes_);
    cloned->collective_program_image_ = collective_program_image_;
    cloned->collective_profile_program_image_ =
        collective_profile_program_image_;
    return cloned;
}
