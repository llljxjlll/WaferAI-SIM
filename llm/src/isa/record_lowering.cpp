#include "isa/record_lowering.h"

#include "dte/dte_async_types.h"
#include "isa/opcode.h"
#include "prims/collective_data_v1_prim.h"
#include "prims/dte_endpoint_prims.h"
#include "prims/exact_stage2_prims.h"
#include "prims/norm_prims.h"
#include "prims/sram_lifecycle_prim.h"
#include "prims/sync_prims.h"
#include "utils/prim_utils.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <sstream>
#include <string_view>
#include <utility>

namespace {

std::string OpcodePrefix(const OpcodeManifestEntry &entry) {
    std::ostringstream stream;
    stream << "opcode 0x" << std::hex << std::uppercase << std::setw(2)
           << std::setfill('0') << static_cast<unsigned>(OpcodeValue(entry.opcode))
           << " (" << entry.canonical_name << "): ";
    return stream.str();
}

[[noreturn]] void Unavailable(const OpcodeManifestEntry &entry,
                              std::string_view reason) {
    throw LoweringUnavailableError(OpcodePrefix(entry) + std::string(reason));
}

[[noreturn]] void LoweringFailure(const OpcodeManifestEntry &entry,
                                  std::string_view reason) {
    throw RecordLoweringError(OpcodePrefix(entry) + std::string(reason));
}

bool IsProductionCompute(Opcode opcode) noexcept {
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
    // DUMMY is also a legacy build macro after Prim headers are included.
    case static_cast<Opcode>(0x15):
        return true;
    default:
        return false;
    }
}

std::unique_ptr<PrimBase> CreateUntracked(PrimId id) {
    return std::unique_ptr<PrimBase>(
        PrimFactory::getInstance().createPrim(
            static_cast<int>(PrimIdValue(id)), false, false));
}

std::vector<std::string> SchemaParameterNames(const RecordSchema &schema) {
    std::vector<std::string> result;
    result.reserve(schema.parameter_count);
    for (std::size_t i = 0; i < schema.parameter_count; ++i)
        result.emplace_back(schema.parameter_names[i]);
    return result;
}

void RequireExactParameterNames(const OpcodeManifestEntry &entry,
                                const RecordSchema &schema,
                                const NpuBase &prim) {
    std::vector<std::string> external = SchemaParameterNames(schema);
    std::vector<std::string> internal = prim.param_name;
    std::sort(external.begin(), external.end());
    std::sort(internal.begin(), internal.end());
    if (external != internal)
        LoweringFailure(entry,
                        "RecordSchema parameter names do not exactly match "
                        "the target NpuBase param_name set");
    if (std::adjacent_find(external.begin(), external.end()) != external.end())
        LoweringFailure(entry, "RecordSchema contains duplicate parameters");
}

LoweredPrimList LowerCompute(const ExternalRecord &record,
                             const OpcodeManifestEntry &entry) {
    const RecordSchema *schema = LookupRecordSchema(record.opcode);
    if (schema == nullptr || schema->operand_kind != RecordOperandKind::COMPUTE)
        LoweringFailure(entry, "compute opcode has no compute RecordSchema");
    const auto &operands = std::get<ComputeOperands>(record.operands);
    std::unique_ptr<PrimBase> base = CreateUntracked(entry.lowering.target);
    NpuBase *prim = dynamic_cast<NpuBase *>(base.get());
    if (prim == nullptr)
        LoweringFailure(entry, "target Prim is not an NpuBase");
    RequireExactParameterNames(entry, *schema, *prim);

    prim->datatype = operands.datatype == ExternalDataType::INT8 ? INT8 : FP16;
    prim->inp_offset = static_cast<int>(operands.input_offset_bytes);
    prim->data_offset = static_cast<int>(operands.data_offset_bytes);
    prim->out_offset = static_cast<int>(operands.output_offset_bytes);
    prim->param_value.clear();
    for (std::size_t i = 0; i < schema->parameter_count; ++i) {
        // ValidateExternalRecord has already proved both the u16 offsets and
        // 30-bit parameters fit their internal fields. Do not truncate again.
        prim->param_value.emplace(
            std::string(schema->parameter_names[i]),
            static_cast<int>(operands.parameters[i]));
    }

    LoweredPrimList result;
    result.push_back(std::move(base));
    return result;
}

LoweredPrimList LowerExactStage2(const ExternalRecord &record,
                                 const OpcodeManifestEntry &entry) {
    std::unique_ptr<PrimBase> base = CreateUntracked(entry.lowering.target);
    switch (record.opcode) {
    case Opcode::ROPE_QK_EXACT: {
        auto *prim = dynamic_cast<Rope_qk_exact_prim *>(base.get());
        if (prim == nullptr)
            LoweringFailure(entry, "target Prim is not Rope_qk_exact_prim");
        prim->operands = std::get<RopeQkExactOperands>(record.operands);
        prim->initialize();
        break;
    }
    case Opcode::ATTENTION_EXACT: {
        auto *prim = dynamic_cast<Attention_exact_prim *>(base.get());
        if (prim == nullptr)
            LoweringFailure(entry, "target Prim is not Attention_exact_prim");
        prim->operands = std::get<AttentionExactOperands>(record.operands);
        prim->initialize();
        break;
    }
    case Opcode::EMBEDDING_LOOKUP: {
        auto *prim = dynamic_cast<Embedding_lookup_prim *>(base.get());
        if (prim == nullptr)
            LoweringFailure(entry, "target Prim is not Embedding_lookup_prim");
        prim->operands = std::get<EmbeddingLookupOperands>(record.operands);
        prim->initialize();
        break;
    }
    case Opcode::GREEDY_SAMPLE: {
        auto *prim = dynamic_cast<Greedy_sample_prim *>(base.get());
        if (prim == nullptr)
            LoweringFailure(entry, "target Prim is not Greedy_sample_prim");
        prim->operands = std::get<GreedySampleOperands>(record.operands);
        prim->initialize();
        break;
    }
    case Opcode::CROSS_ENTROPY_FORWARD: {
        auto *prim = dynamic_cast<Cross_entropy_forward_prim *>(base.get());
        if (prim == nullptr)
            LoweringFailure(
                entry, "target Prim is not Cross_entropy_forward_prim");
        prim->operands =
            std::get<CrossEntropyForwardOperands>(record.operands);
        prim->initialize();
        break;
    }
    case Opcode::CROSS_ENTROPY_BACKWARD: {
        auto *prim = dynamic_cast<Cross_entropy_backward_prim *>(base.get());
        if (prim == nullptr)
            LoweringFailure(
                entry, "target Prim is not Cross_entropy_backward_prim");
        prim->operands =
            std::get<CrossEntropyBackwardOperands>(record.operands);
        prim->initialize();
        break;
    }
    case Opcode::SGD_UPDATE: {
        auto *prim = dynamic_cast<Sgd_update_prim *>(base.get());
        if (prim == nullptr)
            LoweringFailure(entry, "target Prim is not Sgd_update_prim");
        prim->operands = std::get<SgdUpdateOperands>(record.operands);
        prim->initialize();
        break;
    }
    default:
        LoweringFailure(entry, "unexpected exact Stage2 opcode");
    }
    LoweredPrimList result;
    result.push_back(std::move(base));
    return result;
}

LoweredPrimList LowerLocalReduce(const ExternalRecord &record,
                                 const OpcodeManifestEntry &entry) {
    const auto &operands =
        std::get<LocalReduceOperands>(record.operands);
    auto prim = std::make_unique<Collective_data_v1_prim>();
    prim->mode = CollectiveDataV1PrimMode::REDUCE;
    prim->key = {};
    prim->phase_id = 0;
    prim->source_address_bytes = operands.source.absolute_address_bytes;
    prim->destination_address_bytes =
        operands.destination.absolute_address_bytes;
    const bool fp32 = operands.input_dtype == LocalReduceDataType::FP32;
    prim->length_bytes = operands.element_count * (fp32 ? 4 : 2);
    prim->input_count = static_cast<uint16_t>(operands.input_count);
    prim->dtype = fp32 ? CollDType::FP32 : CollDType::FP16;
    prim->reduce_op = CollReduceOp::SUM;
    try {
        prim->Validate();
    } catch (const std::exception &error) {
        LoweringFailure(entry,
                        std::string("strict local reduction Prim validation: ") +
                            error.what());
    }
    LoweredPrimList result;
    result.push_back(std::move(prim));
    return result;
}

std::string ResolveSymbol(const OpcodeManifestEntry &entry,
                          const LoweringContext &context, uint64_t raw_index) {
    if (!context.resolve_symbol)
        LoweringFailure(entry, "symbol resolver is required");
    if (raw_index > std::numeric_limits<uint32_t>::max())
        LoweringFailure(entry, "symbol index exceeds u32 after validation");
    const auto value =
        context.resolve_symbol(static_cast<uint32_t>(raw_index));
    if (!value.has_value())
        LoweringFailure(entry, "symbol resolver returned unknown index " +
                                   std::to_string(raw_index));
    if (value->empty())
        LoweringFailure(entry, "symbol resolver returned an empty name");
    if (value->size() > 64)
        LoweringFailure(entry,
                        "resolved SRAM region exceeds internal 64-byte wire");
    return *value;
}

std::string ResolveLifecycleLabel(
    const OpcodeManifestEntry &entry, const LoweringContext &context,
    uint64_t raw_index) {
    if (!context.resolve_symbol)
        LoweringFailure(entry, "symbol resolver is required");
    if (raw_index > std::numeric_limits<uint32_t>::max())
        LoweringFailure(entry, "symbol index exceeds u32 after validation");
    const auto value =
        context.resolve_symbol(static_cast<uint32_t>(raw_index));
    if (!value.has_value())
        LoweringFailure(entry, "symbol resolver returned unknown index " +
                                   std::to_string(raw_index));
    if (value->empty())
        LoweringFailure(entry, "symbol resolver returned an empty name");
    if (value->size() > 255)
        LoweringFailure(entry,
                        "resolved SRAM label exceeds internal 255-byte wire");
    return *value;
}

std::string ResolveString(const OpcodeManifestEntry &entry,
                          const LoweringContext &context,
                          uint64_t raw_index) {
    if (!context.resolve_string)
        LoweringFailure(entry, "string resolver is required");
    if (raw_index > std::numeric_limits<uint32_t>::max())
        LoweringFailure(entry, "string index exceeds u32 after validation");
    const auto value =
        context.resolve_string(static_cast<uint32_t>(raw_index));
    if (!value.has_value())
        LoweringFailure(entry, "string resolver returned unknown index " +
                                   std::to_string(raw_index));
    if (value->empty())
        LoweringFailure(entry, "string resolver returned an empty name");
    if (value->size() > 64)
        LoweringFailure(entry,
                        "resolved SRAM region exceeds internal 64-byte wire");
    return *value;
}

sram::AllocationLifetime LowerLifetime(
    const OpcodeManifestEntry &entry, SramLifetime lifetime) {
    switch (lifetime) {
    case SramLifetime::TASK:
        return sram::AllocationLifetime::kTask;
    case SramLifetime::LAYER:
        return sram::AllocationLifetime::kLayer;
    case SramLifetime::PERSISTENT:
        return sram::AllocationLifetime::kPersistent;
    }
    LoweringFailure(entry, "SRAM allocation lifetime is invalid");
}

void SetLsuAddress(Lsu_mem_prim &prim, const SramAddressOperand &address,
                   const OpcodeManifestEntry &entry,
                   const LoweringContext &context) {
    if (address.kind == SramAddressKind::ABSOLUTE) {
        prim.absolute_sram = true;
        prim.sram_addr = address.absolute_address_bytes;
        prim.sram_offset = 0;
        prim.sram_region.clear();
        return;
    }
    if (address.kind != SramAddressKind::REGION)
        LoweringFailure(entry, "LSU SRAM address is absent");
    prim.absolute_sram = false;
    prim.sram_addr = 0;
    prim.sram_offset = address.region_offset_bytes;
    prim.sram_region =
        ResolveSymbol(entry, context, address.region_symbol_index);
}

LoweredPrimList LowerLsu(const ExternalRecord &record,
                         const OpcodeManifestEntry &entry,
                         const LoweringContext &context) {
    const auto &operands = std::get<LsuOperands>(record.operands);
    std::unique_ptr<PrimBase> base = CreateUntracked(PrimId::LSU_MEM);
    auto *prim = dynamic_cast<Lsu_mem_prim *>(base.get());
    if (prim == nullptr)
        LoweringFailure(entry, "LSU_MEM factory returned the wrong type");
    prim->hbm_addr = operands.hbm_address_bytes;
    prim->size_bytes = operands.size_bytes;
    prim->token = 0;
    if (record.opcode == Opcode::LSU_LOAD) {
        prim->op = LsuMemOp::LOAD_BLOCKING;
        prim->direction = sram::LsuDirection::kHbmToSram;
    } else {
        prim->op = LsuMemOp::STORE_BLOCKING;
        prim->direction = sram::LsuDirection::kSramToHbm;
    }
    SetLsuAddress(*prim, operands.sram, entry, context);
    prim->refreshPrimType();
    LoweredPrimList result;
    result.push_back(std::move(base));
    return result;
}

LoweredPrimList LowerSramBind(const ExternalRecord &record,
                              const OpcodeManifestEntry &entry,
                              const LoweringContext &context) {
    const auto &operands = std::get<SramBindOperands>(record.operands);
    std::unique_ptr<PrimBase> base =
        CreateUntracked(PrimId::SRAM_BIND_ONESHOT);
    auto *prim = dynamic_cast<Sram_bind_oneshot *>(base.get());
    if (prim == nullptr)
        LoweringFailure(entry,
                        "SRAM_BIND_ONESHOT factory returned the wrong type");

    prim->input_count = static_cast<uint32_t>(operands.input_count);
    for (std::size_t i = 0; i < prim->input_count; ++i) {
        prim->datapass_label.indata[i] = ResolveSymbol(
            entry, context, operands.input_symbol_indices[i]);
    }
    prim->datapass_label.outdata =
        ResolveSymbol(entry, context, operands.output_symbol_index);

    LoweredPrimList result;
    result.push_back(std::move(base));
    return result;
}

LoweredPrimList LowerSramLifecycle(
    const ExternalRecord &record, const OpcodeManifestEntry &entry,
    const LoweringContext &context) {
    std::unique_ptr<PrimBase> base =
        CreateUntracked(PrimId::SRAM_LIFECYCLE);
    auto *prim = dynamic_cast<Sram_lifecycle *>(base.get());
    if (prim == nullptr)
        LoweringFailure(entry,
                        "SRAM_LIFECYCLE factory returned the wrong type");

    switch (record.opcode) {
    case Opcode::SRAM_ALLOC: {
        const auto &operands =
            std::get<SramAllocOperands>(record.operands);
        prim->op = SramLifecycleOp::ALLOC;
        prim->region_name = ResolveString(
            entry, context, operands.region_name_string_index);
        prim->label = ResolveLifecycleLabel(
            entry, context, operands.label_symbol_index);
        prim->size_bytes = operands.size_bytes;
        prim->alignment_bytes = operands.alignment_bytes;
        prim->lifetime = LowerLifetime(entry, operands.lifetime);
        prim->spillable = operands.spillable;
        break;
    }
    case Opcode::SRAM_ALLOC_AT: {
        const auto &operands =
            std::get<SramAllocAtOperands>(record.operands);
        prim->op = SramLifecycleOp::ALLOC_AT;
        prim->region_name = ResolveString(
            entry, context, operands.region_name_string_index);
        prim->label = ResolveLifecycleLabel(
            entry, context, operands.label_symbol_index);
        prim->region_offset_bytes = operands.region_offset_bytes;
        prim->size_bytes = operands.size_bytes;
        prim->alignment_bytes = operands.alignment_bytes;
        prim->lifetime = LowerLifetime(entry, operands.lifetime);
        prim->spillable = operands.spillable;
        break;
    }
    case Opcode::SRAM_FREE:
    case Opcode::SRAM_CLEAR: {
        const auto &operands =
            std::get<SymbolOperands>(record.operands);
        prim->op = record.opcode == Opcode::SRAM_FREE
                       ? SramLifecycleOp::FREE
                       : SramLifecycleOp::CLEAR_TARGETED;
        prim->label = ResolveLifecycleLabel(
            entry, context, operands.symbol_index);
        break;
    }
    case Opcode::SRAM_RESIZE: {
        const auto &operands =
            std::get<SramResizeOperands>(record.operands);
        prim->op = SramLifecycleOp::RESIZE;
        prim->label = ResolveLifecycleLabel(
            entry, context, operands.symbol_index);
        prim->size_bytes = operands.new_size_bytes;
        break;
    }
    case Opcode::SRAM_RENAME: {
        const auto &operands =
            std::get<SramRenameOperands>(record.operands);
        prim->op = SramLifecycleOp::RENAME;
        prim->label = ResolveLifecycleLabel(
            entry, context, operands.old_symbol_index);
        prim->new_label = ResolveLifecycleLabel(
            entry, context, operands.new_symbol_index);
        break;
    }
    default:
        LoweringFailure(entry, "unexpected SRAM lifecycle variant");
    }

    LoweredPrimList result;
    result.push_back(std::move(base));
    return result;
}

void SetDteLocalAddress(Dte_async_prim &prim,
                        const SramAddressOperand &address,
                        const OpcodeManifestEntry &entry,
                        const LoweringContext &context) {
    if (address.kind == SramAddressKind::ABSOLUTE) {
        prim.spm_addr = address.absolute_address_bytes;
        prim.sram_region.clear();
        prim.sram_offset = 0;
        return;
    }
    if (address.kind != SramAddressKind::REGION)
        LoweringFailure(entry, "DTE local SRAM address is absent");
    prim.spm_addr = 0;
    prim.sram_region =
        ResolveSymbol(entry, context, address.region_symbol_index);
    prim.sram_offset = address.region_offset_bytes;
}

void SetDteDestinationAddress(
    Dte_async_prim &prim, const SramAddressOperand &address,
    const OpcodeManifestEntry &entry, const LoweringContext &context) {
    if (address.kind == SramAddressKind::ABSOLUTE) {
        prim.remote_addr = address.absolute_address_bytes;
        prim.destination_sram_region.clear();
        prim.destination_sram_offset = 0;
        return;
    }
    if (address.kind != SramAddressKind::REGION)
        LoweringFailure(entry, "DTE destination SRAM address is absent");
    prim.remote_addr = 0;
    prim.destination_sram_region =
        ResolveSymbol(entry, context, address.region_symbol_index);
    prim.destination_sram_offset = address.region_offset_bytes;
}

DteEndpointCompletion LowerEndpointCompletion(
    const OpcodeManifestEntry &entry, EndpointCompletion completion) {
    switch (completion) {
    case EndpointCompletion::ASYNC:
        return DteEndpointCompletion::ASYNC;
    case EndpointCompletion::SYNC:
        return DteEndpointCompletion::SYNC;
    }
    LoweringFailure(entry, "endpoint completion enum is invalid");
}

DteEndpointSourceSpace LowerEndpointSourceSpace(
    const OpcodeManifestEntry &entry, EndpointSourceSpace source_space) {
    switch (source_space) {
    case EndpointSourceSpace::SRAM:
        return DteEndpointSourceSpace::SRAM;
    case EndpointSourceSpace::HBM:
        return DteEndpointSourceSpace::HBM;
    }
    LoweringFailure(entry, "endpoint source_space enum is invalid");
}

DteEndpointDataType LowerEndpointDataType(
    const OpcodeManifestEntry &entry, EndpointDataType datatype) {
    switch (datatype) {
    case EndpointDataType::UINT8:
        return DteEndpointDataType::UINT8;
    case EndpointDataType::INT32:
        return DteEndpointDataType::INT32;
    case EndpointDataType::INT64:
        return DteEndpointDataType::INT64;
    }
    LoweringFailure(entry, "endpoint datatype enum is invalid");
}

DteEndpointReduceOp LowerEndpointReduceOp(
    const OpcodeManifestEntry &entry, ReduceOperator reduce_op) {
    switch (reduce_op) {
    case ReduceOperator::NONE:
        return DteEndpointReduceOp::NONE;
    case ReduceOperator::SUM:
        return DteEndpointReduceOp::SUM;
    case ReduceOperator::MAX:
        return DteEndpointReduceOp::MAX;
    }
    LoweringFailure(entry, "endpoint reduce_op enum is invalid");
}

void SetEndpointAddress(DteEndpointSramAddress &target,
                        const SramAddressOperand &source,
                        const OpcodeManifestEntry &entry,
                        const LoweringContext &context) {
    if (source.kind == SramAddressKind::ABSOLUTE) {
        target.kind = DteEndpointAddressKind::ABSOLUTE;
        target.absolute_address_bytes = source.absolute_address_bytes;
        target.region.clear();
        target.region_offset_bytes = 0;
        return;
    }
    if (source.kind != SramAddressKind::REGION)
        LoweringFailure(entry, "endpoint SRAM address is absent");
    target.kind = DteEndpointAddressKind::REGION;
    target.absolute_address_bytes = 0;
    target.region = ResolveSymbol(entry, context,
                                  source.region_symbol_index);
    target.region_offset_bytes = source.region_offset_bytes;
}

void SetDteSendSource(Dte_send_endpoint_prim &prim,
                      const DteSendOperands &operands,
                      const OpcodeManifestEntry &entry,
                      const LoweringContext &context) {
    prim.source_space =
        LowerEndpointSourceSpace(entry, operands.source_space);
    SetEndpointAddress(prim.source, operands.source, entry, context);
}

LoweredPrimList LowerDteSend(const ExternalRecord &record,
                             const OpcodeManifestEntry &entry,
                             const LoweringContext &context) {
    const auto &operands = std::get<DteSendOperands>(record.operands);
    if (operands.mode != DteSendMode::P2P || operands.group_id != 0)
        Unavailable(entry,
                    "collective endpoint requires whole-artifact P6 lowering");

    std::unique_ptr<PrimBase> base =
        CreateUntracked(PrimId::DTE_SEND_ENDPOINT);
    auto *prim = dynamic_cast<Dte_send_endpoint_prim *>(base.get());
    if (prim == nullptr)
        LoweringFailure(
            entry, "DTE_SEND_ENDPOINT factory returned the wrong type");
    prim->mode = DteEndpointSendMode::P2P;
    prim->completion =
        LowerEndpointCompletion(entry, operands.completion);
    prim->datatype = LowerEndpointDataType(entry, operands.datatype);
    prim->reduce_op = LowerEndpointReduceOp(entry, operands.reduce_op);
    prim->fsm_id = static_cast<uint32_t>(operands.fsm_id);
    prim->token = static_cast<uint32_t>(operands.token);
    prim->length_bytes = operands.length_bytes;
    prim->peer_core = static_cast<uint16_t>(operands.peer_core);
    prim->expected_sources =
        static_cast<uint16_t>(operands.expected_sources);
    prim->tree_id = static_cast<uint16_t>(operands.tree_id);
    prim->group_id = static_cast<uint32_t>(operands.group_id);
    prim->collective_id = static_cast<uint32_t>(operands.collective_id);
    prim->epoch = static_cast<uint32_t>(operands.epoch);
    SetDteSendSource(*prim, operands, entry, context);
    prim->Validate();

    LoweredPrimList result;
    result.push_back(std::move(base));
    return result;
}

LoweredPrimList LowerDteRecv(const ExternalRecord &record,
                             const OpcodeManifestEntry &entry,
                             const LoweringContext &context) {
    const auto &operands = std::get<DteRecvOperands>(record.operands);
    if (operands.mode != DteRecvMode::P2P || operands.group_id != 0)
        Unavailable(entry,
                    "collective endpoint requires whole-artifact P6 lowering");

    std::unique_ptr<PrimBase> base =
        CreateUntracked(PrimId::DTE_RECV_ENDPOINT);
    auto *prim = dynamic_cast<Dte_recv_endpoint_prim *>(base.get());
    if (prim == nullptr)
        LoweringFailure(
            entry, "DTE_RECV_ENDPOINT factory returned the wrong type");
    prim->mode = DteEndpointRecvMode::P2P;
    prim->completion =
        LowerEndpointCompletion(entry, operands.completion);
    prim->datatype = LowerEndpointDataType(entry, operands.datatype);
    prim->reduce_op = LowerEndpointReduceOp(entry, operands.reduce_op);
    prim->fsm_id = static_cast<uint32_t>(operands.fsm_id);
    prim->token = static_cast<uint32_t>(operands.token);
    prim->length_bytes = operands.length_bytes;
    prim->peer_core = static_cast<uint16_t>(operands.peer_core);
    prim->expected_sources =
        static_cast<uint16_t>(operands.expected_sources);
    prim->tree_id = static_cast<uint16_t>(operands.tree_id);
    prim->group_id = static_cast<uint32_t>(operands.group_id);
    prim->collective_id = static_cast<uint32_t>(operands.collective_id);
    prim->epoch = static_cast<uint32_t>(operands.epoch);
    SetEndpointAddress(prim->destination, operands.destination, entry,
                       context);
    prim->Validate();

    LoweredPrimList result;
    result.push_back(std::move(base));
    return result;
}

LoweredPrimList LowerLocalNoc(const ExternalRecord &record,
                               const OpcodeManifestEntry &entry,
                               const LoweringContext &context) {
    if (record.opcode == Opcode::LOCAL_NOC_SEND) {
        const auto &operands = std::get<LocalNocSendOperands>(record.operands);
        auto base = CreateUntracked(PrimId::DTE_SEND_ENDPOINT);
        auto *prim = dynamic_cast<Dte_send_endpoint_prim *>(base.get());
        if (prim == nullptr)
            LoweringFailure(entry, "DTE_SEND_ENDPOINT factory returned the wrong type");
        prim->mode = DteEndpointSendMode::P2P;
        prim->source_space = DteEndpointSourceSpace::SRAM;
        prim->completion = DteEndpointCompletion::SYNC;
        prim->datatype = DteEndpointDataType::UINT8;
        prim->reduce_op = DteEndpointReduceOp::NONE;
        prim->fsm_id = static_cast<uint32_t>(operands.event_id);
        prim->token = 0;
        prim->length_bytes = operands.byte_count;
        prim->peer_core = static_cast<uint16_t>(operands.destination_core);
        SetEndpointAddress(prim->source, operands.source, entry, context);
        prim->Validate();
        LoweredPrimList result;
        result.push_back(std::move(base));
        return result;
    }
    if (record.opcode == Opcode::LOCAL_NOC_RECV) {
        const auto &operands = std::get<LocalNocRecvOperands>(record.operands);
        auto base = CreateUntracked(PrimId::DTE_RECV_ENDPOINT);
        auto *prim = dynamic_cast<Dte_recv_endpoint_prim *>(base.get());
        if (prim == nullptr)
            LoweringFailure(entry, "DTE_RECV_ENDPOINT factory returned the wrong type");
        prim->mode = DteEndpointRecvMode::P2P;
        prim->completion = DteEndpointCompletion::ASYNC;
        prim->datatype = DteEndpointDataType::UINT8;
        prim->reduce_op = DteEndpointReduceOp::NONE;
        prim->fsm_id = static_cast<uint32_t>(operands.event_id);
        prim->token = static_cast<uint32_t>(operands.event_id);
        prim->length_bytes = operands.byte_count;
        prim->peer_core = static_cast<uint16_t>(operands.source_core);
        SetEndpointAddress(prim->destination, operands.destination, entry, context);
        prim->Validate();
        LoweredPrimList result;
        result.push_back(std::move(base));
        return result;
    }
    if (record.opcode == Opcode::LOCAL_NOC_WAIT) {
        const auto &operands = std::get<LocalNocWaitOperands>(record.operands);
        auto base = CreateUntracked(PrimId::DTE_ASYNC);
        auto *prim = dynamic_cast<Dte_async_prim *>(base.get());
        if (prim == nullptr)
            LoweringFailure(entry, "DTE_ASYNC factory returned the wrong type");
        prim->op = DteAsyncOp::WAIT;
        prim->token = static_cast<uint32_t>(operands.event_id);
        prim->payload_bits = 0;
        prim->refreshPrimType();
        LoweredPrimList result;
        result.push_back(std::move(base));
        return result;
    }
    LoweringFailure(entry, "unexpected local NoC opcode");
}

LoweredPrimList LowerDteIssue(const ExternalRecord &record,
                              const OpcodeManifestEntry &entry,
                              const LoweringContext &context) {
    const auto &operands = std::get<DteIssueOperands>(record.operands);
    std::unique_ptr<PrimBase> base = CreateUntracked(PrimId::DTE_ASYNC);
    auto *prim = dynamic_cast<Dte_async_prim *>(base.get());
    if (prim == nullptr)
        LoweringFailure(entry, "DTE_ASYNC factory returned the wrong type");
    prim->op = DteAsyncOp::ISSUE;
    prim->token = static_cast<uint32_t>(operands.token);
    prim->payload_bits = operands.payload_bits;
    prim->spm_size = operands.size_bytes;
    prim->remote_peer = DTE_ASYNC_INVALID_REMOTE_PEER;
    prim->address_block = 0;
    prim->destination_sram_region.clear();
    prim->destination_sram_offset = 0;

    switch (operands.direction) {
    case LocalDteDirection::SPM_TO_SPM:
        prim->direction = DteDir::SPM_TO_SPM;
        SetDteLocalAddress(*prim, operands.source_sram, entry, context);
        SetDteDestinationAddress(*prim, operands.destination_sram, entry,
                                 context);
        break;
    case LocalDteDirection::SPM_TO_DRAM:
        prim->direction = DteDir::SPM_TO_DRAM;
        SetDteLocalAddress(*prim, operands.source_sram, entry, context);
        prim->remote_addr = operands.hbm_address_bytes;
        break;
    case LocalDteDirection::DRAM_TO_SPM:
        prim->direction = DteDir::DRAM_TO_SPM;
        SetDteLocalAddress(*prim, operands.destination_sram, entry, context);
        prim->remote_addr = operands.hbm_address_bytes;
        break;
    }
    prim->refreshPrimType();
    LoweredPrimList result;
    result.push_back(std::move(base));
    return result;
}

LoweredPrimList LowerDteControl(const ExternalRecord &record,
                                const OpcodeManifestEntry &entry) {
    std::unique_ptr<PrimBase> base = CreateUntracked(PrimId::DTE_ASYNC);
    auto *prim = dynamic_cast<Dte_async_prim *>(base.get());
    if (prim == nullptr)
        LoweringFailure(entry, "DTE_ASYNC factory returned the wrong type");
    prim->payload_bits = 0;
    prim->token = 0;
    switch (record.opcode) {
    case Opcode::DTE_WAIT:
        prim->op = DteAsyncOp::WAIT;
        prim->token = static_cast<uint32_t>(
            std::get<TokenOperands>(record.operands).token);
        break;
    case Opcode::DTE_FENCE:
        prim->op = DteAsyncOp::FENCE;
        break;
    case Opcode::DTE_CANCEL:
        prim->op = DteAsyncOp::CANCEL;
        prim->token = static_cast<uint32_t>(
            std::get<TokenOperands>(record.operands).token);
        break;
    default:
        LoweringFailure(entry, "unexpected DTE control variant");
    }
    prim->refreshPrimType();
    LoweredPrimList result;
    result.push_back(std::move(base));
    return result;
}

LoweredPrimList LowerSynchronization(const ExternalRecord &record,
                                     const OpcodeManifestEntry &entry) {
    LoweredPrimList result;
    if (record.opcode == Opcode::GROUP_SYNC) {
        std::unique_ptr<PrimBase> base =
            CreateUntracked(PrimId::GROUP_SYNC);
        auto *prim = dynamic_cast<Group_sync_prim *>(base.get());
        if (prim == nullptr)
            LoweringFailure(entry,
                            "GROUP_SYNC factory returned the wrong type");
        const auto &operands =
            std::get<GroupSyncOperands>(record.operands);
        prim->group_id = static_cast<uint32_t>(operands.group_id);
        prim->sync_seq = static_cast<uint32_t>(operands.sync_seq);
        result.push_back(std::move(base));
        return result;
    }

    std::unique_ptr<PrimBase> base =
        CreateUntracked(PrimId::EVENT_CONTROL);
    auto *prim = dynamic_cast<Event_control_prim *>(base.get());
    if (prim == nullptr)
        LoweringFailure(entry,
                        "EVENT_CONTROL factory returned the wrong type");
    if (record.opcode == Opcode::EVENT_SET) {
        const auto &operands =
            std::get<EventSetOperands>(record.operands);
        prim->op = EventControlOp::SET;
        prim->source_core = static_cast<uint16_t>(operands.source_core);
        prim->destination_core =
            static_cast<uint16_t>(operands.destination_core);
        prim->tag = static_cast<uint32_t>(operands.tag);
        prim->count = 1;
    } else if (record.opcode == Opcode::EVENT_WAIT) {
        const auto &operands =
            std::get<EventWaitOperands>(record.operands);
        prim->op = EventControlOp::WAIT;
        prim->source_core = static_cast<uint16_t>(operands.source_core);
        prim->destination_core =
            static_cast<uint16_t>(operands.destination_core);
        prim->tag = static_cast<uint32_t>(operands.tag);
        prim->count = static_cast<uint32_t>(operands.count);
    } else {
        LoweringFailure(entry, "unexpected synchronization opcode");
    }
    result.push_back(std::move(base));
    return result;
}

CollTxKind LowerGraphTx(DteSendMode mode) {
    switch (mode) {
    case DteSendMode::P2P: return CollTxKind::UNICAST;
    case DteSendMode::SCATTER: return CollTxKind::SCATTER;
    case DteSendMode::BROADCAST: return CollTxKind::BROADCAST;
    }
    throw RecordLoweringError("invalid DTE_SEND mode in collective adapter");
}

CollRxKind LowerGraphRx(DteRecvMode mode) {
    switch (mode) {
    case DteRecvMode::P2P: return CollRxKind::UNICAST;
    case DteRecvMode::GATHER: return CollRxKind::GATHER;
    case DteRecvMode::REDUCE: return CollRxKind::REDUCE;
    }
    throw RecordLoweringError("invalid DTE_RECV mode in collective adapter");
}

CollDType LowerGraphDType(EndpointDataType datatype) {
    switch (datatype) {
    case EndpointDataType::UINT8: return CollDType::UINT8;
    case EndpointDataType::INT32: return CollDType::INT32;
    case EndpointDataType::INT64: return CollDType::INT64;
    }
    throw RecordLoweringError("invalid endpoint datatype in collective adapter");
}

CollReduceOp LowerGraphReduce(ReduceOperator reduce_op) {
    switch (reduce_op) {
    case ReduceOperator::NONE: return CollReduceOp::NONE;
    case ReduceOperator::SUM: return CollReduceOp::SUM;
    case ReduceOperator::MAX: return CollReduceOp::MAX;
    }
    throw RecordLoweringError("invalid reduce_op in collective adapter");
}

uint64_t GraphAddressValue(const SramAddressOperand &address,
                           const ProgramArtifact &artifact) {
    if (address.kind == SramAddressKind::ABSOLUTE)
        return address.absolute_address_bytes;
    if (address.kind != SramAddressKind::REGION)
        throw RecordLoweringError("collective endpoint address is absent");
    if (address.region_symbol_index >= artifact.symbols.size())
        throw RecordLoweringError(
            "collective endpoint region symbol index is out of range");
    const ProgramSymbol &symbol =
        artifact.symbols[address.region_symbol_index];
    if (symbol.kind != ProgramSymbolKind::SRAM_REGION)
        throw RecordLoweringError(
            "collective endpoint region symbol has the wrong kind");
    if (address.region_offset_bytes > symbol.size_bytes)
        throw RecordLoweringError(
            "collective endpoint region offset exceeds its symbol size");
    if (symbol.value > std::numeric_limits<uint64_t>::max() -
                           address.region_offset_bytes)
        throw RecordLoweringError(
            "collective endpoint region base plus offset overflows u64");
    return symbol.value + address.region_offset_bytes;
}

uint64_t CheckedGraphSpan(uint64_t count, uint64_t length_bytes,
                          const char *field) {
    if (count == 0 || length_bytes == 0)
        throw RecordLoweringError(std::string(field) +
                                  " span cannot be zero");
    if (count > std::numeric_limits<uint64_t>::max() / length_bytes)
        throw RecordLoweringError(std::string(field) +
                                  " span overflows u64");
    return count * length_bytes;
}

void ValidateGraphRegionSpan(const SramAddressOperand &address,
                             const ProgramArtifact &artifact,
                             uint64_t span_bytes,
                             const std::string &field) {
    if (span_bytes == 0)
        throw RecordLoweringError(field + " span cannot be zero");
    if (address.kind == SramAddressKind::ABSOLUTE) {
        if (address.absolute_address_bytes >
            std::numeric_limits<uint64_t>::max() - (span_bytes - 1))
            throw RecordLoweringError(field +
                                      " absolute span overflows u64");
        return;
    }
    if (address.kind != SramAddressKind::REGION ||
        address.region_symbol_index >= artifact.symbols.size())
        throw RecordLoweringError(field + " has an invalid region symbol");
    const ProgramSymbol &symbol =
        artifact.symbols[address.region_symbol_index];
    if (symbol.kind != ProgramSymbolKind::SRAM_REGION)
        throw RecordLoweringError(field + " region symbol has the wrong kind");
    if (symbol.size_bytes == 0)
        throw RecordLoweringError(field + " region symbol size is zero");
    if (address.region_offset_bytes > symbol.size_bytes ||
        span_bytes > symbol.size_bytes - address.region_offset_bytes)
        throw RecordLoweringError(field +
                                  " span exceeds its region symbol size");
    if (symbol.value > std::numeric_limits<uint64_t>::max() -
                           address.region_offset_bytes ||
        symbol.value + address.region_offset_bytes >
            std::numeric_limits<uint64_t>::max() - (span_bytes - 1))
        throw RecordLoweringError(field +
                                  " physical region span overflows u64");
}

SramAddressOperand RebaseGraphAddress(uint64_t value) {
    SramAddressOperand result;
    result.kind = SramAddressKind::ABSOLUTE;
    result.absolute_address_bytes = value;
    result.region_symbol_index = 0;
    result.region_offset_bytes = 0;
    return result;
}

using ArtifactEndpointKey = std::pair<CollectiveKey, uint16_t>;

bool SameSramAddress(const SramAddressOperand &left,
                     const SramAddressOperand &right) {
    return std::tie(left.kind, left.absolute_address_bytes,
                    left.region_symbol_index, left.region_offset_bytes) ==
           std::tie(right.kind, right.absolute_address_bytes,
                    right.region_symbol_index, right.region_offset_bytes);
}

} // namespace

bool IsaV1LoweredCollectiveChild::operator==(
    const IsaV1LoweredCollectiveChild &other) const {
    return plan_index == other.plan_index && child_index == other.child_index &&
           source_internal_token == other.source_internal_token &&
           destination_internal_token == other.destination_internal_token &&
           source_space == other.source_space &&
           SameSramAddress(source, other.source) &&
           SameSramAddress(destination, other.destination);
}

bool IsaV1LoweredCollectiveAction::operator==(
    const IsaV1LoweredCollectiveAction &other) const {
    return std::tie(plan_index, key, action, internal_token,
                    public_aggregate_token) ==
           std::tie(other.plan_index, other.key, other.action,
                    other.internal_token, other.public_aggregate_token);
}

bool IsaV1CoreCollectiveActionStream::operator==(
    const IsaV1CoreCollectiveActionStream &other) const {
    return std::tie(core_id, actions) == std::tie(other.core_id, other.actions);
}

bool IsaV1CollectiveArtifactLowering::operator==(
    const IsaV1CollectiveArtifactLowering &other) const {
    return std::tie(plans, children, core_actions, executable) ==
           std::tie(other.plans, other.children, other.core_actions,
                    other.executable);
}

IsaV1CollectiveArtifactLowering LowerIsaV1CollectiveArtifact(
    const ProgramArtifact &artifact, uint32_t total_cores,
    uint32_t cores_per_die, const IsaV1PlannerCapacity &capacity) {
    ValidateProgramArtifact(artifact);
    IsaV1CollectiveGroupRegistryView registry;
    registry.total_cores = total_cores;
    registry.cores_per_die = cores_per_die;
    for (uint64_t core : artifact.envelope.active_cores)
        registry.active_cores.push_back(static_cast<uint16_t>(core));
    for (const ProgramCoreGroup &group : artifact.core_groups) {
        IsaV1CollectiveGroupDefinition definition;
        definition.group_id = static_cast<uint32_t>(group.group_id);
        for (uint64_t member : group.members)
            definition.members.push_back(static_cast<uint16_t>(member));
        registry.groups.push_back(std::move(definition));
    }

    std::vector<IsaV1NormalizedCollectiveRecord> normalized;
    std::map<ArtifactEndpointKey, const DteSendOperands *> sends;
    std::map<ArtifactEndpointKey, const DteRecvOperands *> receives;
    std::map<ArtifactEndpointKey, const ReduceComputeOperands *> computes;
    std::set<uint32_t> reserved_tokens;
    for (const ProgramCore &core : artifact.cores) {
        for (std::size_t record_index = 0; record_index < core.records.size();
             ++record_index) {
            const ExternalRecord &record = core.records[record_index];
            if (const auto *send = std::get_if<DteSendOperands>(&record.operands)) {
                IsaV1NormalizedCollectiveRecord value;
                value.core_id = static_cast<uint16_t>(core.core_id);
                value.record_index = static_cast<uint32_t>(record_index);
                value.role = IsaV1CollectiveRecordRole::SEND;
                value.tx_mode = LowerGraphTx(send->mode);
                value.asynchronous = send->completion == EndpointCompletion::ASYNC;
                value.token = static_cast<uint32_t>(send->token);
                value.logical_fsm_id_base = static_cast<uint32_t>(send->fsm_id);
                value.length_bytes = send->length_bytes;
                value.dtype = LowerGraphDType(send->datatype);
                value.reduce_op = LowerGraphReduce(send->reduce_op);
                value.base_address_bytes =
                    GraphAddressValue(send->source, artifact);
                value.key = {static_cast<uint32_t>(send->group_id),
                             static_cast<uint32_t>(send->collective_id),
                             static_cast<uint32_t>(send->epoch)};
                value.tree_id = static_cast<uint16_t>(send->tree_id);
                value.expected_sources = static_cast<uint16_t>(send->expected_sources);
                value.peer_core = static_cast<uint16_t>(send->peer_core);
                normalized.push_back(value);
                if (send->token != 0)
                    reserved_tokens.insert(static_cast<uint32_t>(send->token));
                if (send->group_id != 0)
                    sends.emplace(ArtifactEndpointKey{value.key, value.core_id}, send);
            } else if (const auto *receive = std::get_if<DteRecvOperands>(&record.operands)) {
                IsaV1NormalizedCollectiveRecord value;
                value.core_id = static_cast<uint16_t>(core.core_id);
                value.record_index = static_cast<uint32_t>(record_index);
                value.role = IsaV1CollectiveRecordRole::RECEIVE;
                value.rx_mode = LowerGraphRx(receive->mode);
                value.asynchronous = receive->completion == EndpointCompletion::ASYNC;
                value.token = static_cast<uint32_t>(receive->token);
                value.logical_fsm_id_base = static_cast<uint32_t>(receive->fsm_id);
                value.length_bytes = receive->length_bytes;
                value.dtype = LowerGraphDType(receive->datatype);
                value.reduce_op = LowerGraphReduce(receive->reduce_op);
                value.base_address_bytes =
                    GraphAddressValue(receive->destination, artifact);
                value.key = {static_cast<uint32_t>(receive->group_id),
                             static_cast<uint32_t>(receive->collective_id),
                             static_cast<uint32_t>(receive->epoch)};
                value.tree_id = static_cast<uint16_t>(receive->tree_id);
                value.expected_sources = static_cast<uint16_t>(receive->expected_sources);
                value.peer_core = static_cast<uint16_t>(receive->peer_core);
                normalized.push_back(value);
                if (receive->token != 0)
                    reserved_tokens.insert(static_cast<uint32_t>(receive->token));
                if (receive->group_id != 0)
                    receives.emplace(ArtifactEndpointKey{value.key, value.core_id}, receive);
            } else if (const auto *compute =
                           std::get_if<ReduceComputeOperands>(&record.operands)) {
                IsaV1NormalizedCollectiveRecord value;
                value.core_id = static_cast<uint16_t>(core.core_id);
                value.record_index = static_cast<uint32_t>(record_index);
                value.role = IsaV1CollectiveRecordRole::REDUCE_COMPUTE;
                value.asynchronous = false;
                value.element_count = compute->element_count;
                value.dtype = LowerGraphDType(compute->datatype);
                value.reduce_op = LowerGraphReduce(compute->reduce_op);
                value.base_address_bytes =
                    GraphAddressValue(compute->source, artifact);
                value.result_address_bytes =
                    GraphAddressValue(compute->destination, artifact);
                value.key = {static_cast<uint32_t>(compute->group_id),
                             static_cast<uint32_t>(compute->collective_id),
                             static_cast<uint32_t>(compute->epoch)};
                value.root_rank = static_cast<uint16_t>(compute->root_rank);
                value.self_rank = static_cast<uint16_t>(compute->self_rank);
                normalized.push_back(value);
                computes.emplace(
                    ArtifactEndpointKey{value.key, value.core_id}, compute);
            } else if (const auto *issue = std::get_if<DteIssueOperands>(&record.operands)) {
                reserved_tokens.insert(static_cast<uint32_t>(issue->token));
            } else if (const auto *token = std::get_if<TokenOperands>(&record.operands)) {
                reserved_tokens.insert(static_cast<uint32_t>(token->token));
            }
        }
    }

    IsaV1CollectiveArtifactLowering result;
    try {
        result.plans = BuildIsaV1CollectiveGraph(normalized, registry, capacity);
    } catch (const std::overflow_error &error) {
        throw RecordLoweringError(std::string("ISA-v1 whole-artifact collective graph: ") + error.what());
    } catch (const std::invalid_argument &error) {
        throw RecordLoweringError(std::string("ISA-v1 whole-artifact collective graph: ") + error.what());
    }

    // Region symbols describe a physical base plus a bounded named extent.
    // Validate every semantic access before the planner-derived absolute
    // addresses discard the symbol boundary. Tight collective layouts use
    // N*L for SCATTER sources, GATHER/REDUCE destinations, and reduction
    // staging; ordinary endpoints and reduction results use L.
    for (const IsaV1CollectivePlan &plan : result.plans) {
        const uint64_t rank_count = static_cast<uint64_t>(plan.group.size());
        const uint64_t tight_span = CheckedGraphSpan(
            rank_count, plan.length_bytes, "collective tight layout");
        for (uint16_t core : plan.group) {
            const ArtifactEndpointKey key{plan.key, core};
            const auto send = sends.find(key);
            if (send != sends.end()) {
                const uint64_t span =
                    send->second->mode == DteSendMode::SCATTER
                        ? tight_span
                        : plan.length_bytes;
                ValidateGraphRegionSpan(
                    send->second->source, artifact, span,
                    "collective key(" + std::to_string(plan.key.group_id) +
                        "," + std::to_string(plan.key.collective_id) +
                        "," + std::to_string(plan.key.epoch) + ") core " +
                        std::to_string(core) + " SEND source");
            }
            const auto receive = receives.find(key);
            if (receive != receives.end()) {
                const uint64_t span =
                    receive->second->mode == DteRecvMode::P2P
                        ? plan.length_bytes
                        : tight_span;
                ValidateGraphRegionSpan(
                    receive->second->destination, artifact, span,
                    "collective key(" + std::to_string(plan.key.group_id) +
                        "," + std::to_string(plan.key.collective_id) +
                        "," + std::to_string(plan.key.epoch) + ") core " +
                        std::to_string(core) + " RECEIVE destination");
            }
            const auto compute = computes.find(key);
            if (compute != computes.end()) {
                ValidateGraphRegionSpan(
                    compute->second->source, artifact, tight_span,
                    "collective key(" + std::to_string(plan.key.group_id) +
                        "," + std::to_string(plan.key.collective_id) +
                        "," + std::to_string(plan.key.epoch) + ") core " +
                        std::to_string(core) + " REDUCE source");
                ValidateGraphRegionSpan(
                    compute->second->destination, artifact,
                    plan.length_bytes,
                    "collective key(" + std::to_string(plan.key.group_id) +
                        "," + std::to_string(plan.key.collective_id) +
                        "," + std::to_string(plan.key.epoch) + ") core " +
                        std::to_string(core) + " REDUCE destination");
            }
        }
    }

    uint32_t next_internal_token = std::numeric_limits<uint32_t>::max();
    auto allocate_token = [&]() {
        while (next_internal_token != 0 && reserved_tokens.count(next_internal_token) != 0)
            --next_internal_token;
        if (next_internal_token == 0)
            throw RecordLoweringError("ISA-v1 collective internal token namespace is exhausted");
        const uint32_t token = next_internal_token--;
        reserved_tokens.insert(token);
        return token;
    };

    std::vector<std::vector<std::size_t>> child_lookup(result.plans.size());
    for (std::size_t plan_index = 0; plan_index < result.plans.size(); ++plan_index) {
        const IsaV1CollectivePlan &plan = result.plans[plan_index];
        for (std::size_t child_index = 0; child_index < plan.child_flows.size(); ++child_index) {
            const IsaV1ChildFlow &flow = plan.child_flows[child_index];
            const auto send = sends.find({plan.key, flow.source_core});
            const auto receive = receives.find({plan.key, flow.destination_core});
            if (send == sends.end() || receive == receives.end())
                throw RecordLoweringError("ISA-v1 internal collective endpoint lookup failed");
            IsaV1LoweredCollectiveChild child;
            child.plan_index = plan_index;
            child.child_index = static_cast<uint32_t>(child_index);
            child.source_internal_token = allocate_token();
            child.destination_internal_token = allocate_token();
            child.source_space = send->second->source_space;
            child.source = RebaseGraphAddress(flow.source_address_bytes);
            child.destination = RebaseGraphAddress(
                flow.destination_address_bytes);
            child_lookup[plan_index].push_back(result.children.size());
            result.children.push_back(std::move(child));
        }
    }

    std::map<uint16_t, std::vector<IsaV1LoweredCollectiveAction>> by_core;
    for (std::size_t plan_index = 0; plan_index < result.plans.size(); ++plan_index) {
        const IsaV1CollectivePlan &plan = result.plans[plan_index];
        for (const auto &rank_actions : plan.actions_by_rank) {
            for (const IsaV1Action &action : rank_actions) {
                IsaV1LoweredCollectiveAction lowered;
                lowered.plan_index = plan_index;
                lowered.key = plan.key;
                lowered.action = action;
                const bool send_action =
                    action.kind == IsaV1ActionKind::ISSUE_SEND ||
                    action.kind == IsaV1ActionKind::WAIT_SEND ||
                    action.kind == IsaV1ActionKind::WAIT_TRANSPORT_RETIRE;
                const bool receive_action =
                    action.kind == IsaV1ActionKind::POST_RECEIVE ||
                    action.kind == IsaV1ActionKind::WAIT_RECEIVE;
                if (send_action || receive_action) {
                    if (action.item_index >= child_lookup[plan_index].size())
                        throw RecordLoweringError("ISA-v1 internal collective action child index is invalid");
                    const auto &child = result.children[
                        child_lookup[plan_index][action.item_index]];
                    const auto &flow = plan.child_flows[action.item_index];
                    lowered.internal_token = send_action
                        ? child.source_internal_token
                        : child.destination_internal_token;
                    lowered.public_aggregate_token = send_action
                        ? flow.source_public_token
                        : flow.destination_public_token;
                }
                by_core[action.core].push_back(std::move(lowered));
            }
        }
    }
    for (auto &entry : by_core)
        result.core_actions.push_back({entry.first, std::move(entry.second)});
    return result;
}

LoweredPrimList LowerExternalRecord(const ExternalRecord &record,
                                    const LoweringContext &context) {
    // Preserve RecordCodecError and its gated/unsupported/reserved wording.
    ValidateExternalRecord(record, context.enabled_capabilities);
    const OpcodeManifestEntry *entry = LookupOpcode(record.opcode);
    if (entry == nullptr)
        throw RecordLoweringError("validated opcode has no manifest entry");

    if (IsProductionCompute(record.opcode))
        return LowerCompute(record, *entry);
    switch (record.opcode) {
    case Opcode::ROPE_QK_EXACT:
    case Opcode::ATTENTION_EXACT:
    case Opcode::EMBEDDING_LOOKUP:
    case Opcode::GREEDY_SAMPLE:
    case Opcode::CROSS_ENTROPY_FORWARD:
    case Opcode::CROSS_ENTROPY_BACKWARD:
    case Opcode::SGD_UPDATE:
        return LowerExactStage2(record, *entry);
    case Opcode::LSU_LOAD:
    case Opcode::LSU_STORE:
        return LowerLsu(record, *entry, context);
    case Opcode::SRAM_BIND:
        return LowerSramBind(record, *entry, context);
    case Opcode::SRAM_ALLOC:
    case Opcode::SRAM_ALLOC_AT:
    case Opcode::SRAM_FREE:
    case Opcode::SRAM_RESIZE:
    case Opcode::SRAM_RENAME:
    case Opcode::SRAM_CLEAR:
        return LowerSramLifecycle(record, *entry, context);
    case Opcode::DTE_SEND:
        return LowerDteSend(record, *entry, context);
    case Opcode::DTE_RECV:
        return LowerDteRecv(record, *entry, context);
    case Opcode::LOCAL_NOC_SEND:
    case Opcode::LOCAL_NOC_RECV:
    case Opcode::LOCAL_NOC_WAIT:
        return LowerLocalNoc(record, *entry, context);
    case Opcode::REDUCE_COMPUTE:
        Unavailable(*entry,
                    "requires whole-artifact P6 lowering and the future strict byte-executing reduction runtime; legacy timing-only Reduce_compute_prim is forbidden");
    case Opcode::LOCAL_REDUCE:
        return LowerLocalReduce(record, *entry);
    case Opcode::DTE_ISSUE:
        return LowerDteIssue(record, *entry, context);
    case Opcode::DTE_WAIT:
    case Opcode::DTE_FENCE:
    case Opcode::DTE_CANCEL:
        return LowerDteControl(record, *entry);
    case Opcode::EVENT_SET:
    case Opcode::EVENT_WAIT:
    case Opcode::GROUP_SYNC:
        return LowerSynchronization(record, *entry);
    default:
        break;
    }

    switch (entry->lowering.kind) {
    case OpcodeLoweringKind::MODE_DISPATCH:
        Unavailable(*entry, "MODE_DISPATCH lowering is deferred");
    case OpcodeLoweringKind::NEW_THIN_PRIM:
        Unavailable(*entry, "NEW_THIN_PRIM lowering is deferred");
    case OpcodeLoweringKind::DIRECT_PRIM:
    case OpcodeLoweringKind::PRIM_VARIANT:
        Unavailable(*entry, "lowering is deferred to a later ISA stage");
    }
    Unavailable(*entry, "unknown lowering kind");
}
