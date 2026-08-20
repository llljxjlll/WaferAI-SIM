#include "isa/record_lowering_selftest.h"

#include "isa/record_lowering.h"
#include "common/memory.h"
#include "prims/collective_data_v1_prim.h"
#include "prims/dte_endpoint_prims.h"
#include "prims/exact_stage2_prims.h"
#include "prims/norm_prims.h"
#include "prims/sram_lifecycle_prim.h"
#include "prims/sync_prims.h"
#include "utils/prim_utils.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

class Checks {
public:
    void Check(bool condition, std::string name) {
        ++result.checks;
        if (!condition)
            result.failures.push_back(std::move(name));
    }

    template <typename Exception, typename Function>
    void Reject(std::string name, std::string_view message_fragment,
                Function function) {
        ++result.checks;
        try {
            function();
        } catch (const Exception &error) {
            if (std::string(error.what()).find(message_fragment) ==
                std::string::npos) {
                result.failures.push_back(
                    std::move(name) + " (wrong diagnostic: " + error.what() +
                    ")");
            }
            return;
        } catch (const std::exception &error) {
            result.failures.push_back(
                std::move(name) + " (wrong exception type: " + error.what() +
                ")");
            return;
        }
        result.failures.push_back(std::move(name) + " (accepted)");
    }

    RecordLoweringSelfTestResult result;
};

LoweringContext Context() {
    LoweringContext context;
    context.resolve_symbol = [](uint32_t index)
        -> std::optional<std::string> {
        if (index == 0xffffffffU)
            return std::nullopt;
        return "region_" + std::to_string(index);
    };
    context.resolve_string = [](uint32_t index)
        -> std::optional<std::string> {
        if (index == 0xffffffffU)
            return std::nullopt;
        return "string_" + std::to_string(index);
    };
    return context;
}

SramAddressOperand Absolute(uint64_t address) {
    SramAddressOperand result;
    result.kind = SramAddressKind::ABSOLUTE;
    result.absolute_address_bytes = address;
    return result;
}

SramAddressOperand Region(uint32_t symbol, uint64_t offset) {
    SramAddressOperand result;
    result.kind = SramAddressKind::REGION;
    result.region_symbol_index = symbol;
    result.region_offset_bytes = offset;
    return result;
}

uint64_t ParameterValue(std::string_view name, std::size_t /*index*/) {
    if (name == "is_merge" || name == "need_choose") return 0;
    if (name == "pX" || name == "pY") return 0;
    if (name == "sX" || name == "sY" || name == "kX" || name == "kY")
        return 1;
    if (name == "dim" || name == "slice") return 1;
    return 2;
}

ExternalRecord MakeRecord(const RecordSchema &schema) {
    ExternalRecord record;
    record.opcode = schema.opcode;
    switch (schema.operand_kind) {
    case RecordOperandKind::COMPUTE: {
        ComputeOperands operands;
        operands.datatype = ExternalDataType::FP16;
        operands.input_offset_bytes = 0x1234;
        operands.data_offset_bytes = 0x4567;
        operands.output_offset_bytes = 0x789a;
        operands.parameters.reserve(schema.parameter_count);
        for (std::size_t i = 0; i < schema.parameter_count; ++i)
            operands.parameters.push_back(
                ParameterValue(schema.parameter_names[i], i));
        record.operands = std::move(operands);
        break;
    }
    case RecordOperandKind::ROPE_QK_EXACT: {
        RopeQkExactOperands operands;
        operands.input = Absolute(0x1000);
        operands.output = Absolute(0x2000);
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
        operands.rope_theta_f64_bits = UINT64_C(0x40c3880000000000);
        record.operands = operands;
        break;
    }
    case RecordOperandKind::ATTENTION_EXACT: {
        AttentionExactOperands operands;
        operands.input = Absolute(0x1000);
        operands.output = Absolute(0x2000);
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
        record.operands = operands;
        break;
    }
    case RecordOperandKind::EMBEDDING_LOOKUP: {
        EmbeddingLookupOperands operands;
        operands.indices = Absolute(0x1000);
        operands.table = Absolute(0x2000);
        operands.output = Absolute(0x3000);
        operands.logical_rows = 8;
        operands.rank_rows = 4;
        operands.tp_degree = 2;
        operands.vocab_size = 32;
        operands.hidden_size = 16;
        record.operands = operands;
        break;
    }
    case RecordOperandKind::GREEDY_SAMPLE: {
        GreedySampleOperands operands;
        operands.logits = Absolute(0x1000);
        operands.output = Absolute(0x2000);
        operands.tp_degree = 1;
        operands.token_rows = 8;
        operands.vocab_size = 32;
        operands.sample_count = 1;
        operands.comparisons = 31;
        record.operands = operands;
        break;
    }
    case RecordOperandKind::CROSS_ENTROPY_FORWARD: {
        CrossEntropyForwardOperands operands;
        operands.logits = Absolute(0x1000);
        operands.labels = Absolute(0x2000);
        operands.loss = Absolute(0x3000);
        operands.logical_rows = 8;
        operands.rank_rows = 4;
        operands.tp_degree = 2;
        operands.vocab_size = 32;
        record.operands = operands;
        break;
    }
    case RecordOperandKind::CROSS_ENTROPY_BACKWARD: {
        CrossEntropyBackwardOperands operands;
        operands.logits = Absolute(0x1000);
        operands.labels = Absolute(0x2000);
        operands.upstream = Absolute(0x3000);
        operands.logits_grad = Absolute(0x4000);
        operands.logical_rows = 8;
        operands.rank_rows = 4;
        operands.tp_degree = 2;
        operands.vocab_size = 32;
        operands.upstream_elements = 4;
        record.operands = operands;
        break;
    }
    case RecordOperandKind::SGD_UPDATE: {
        SgdUpdateOperands operands;
        operands.weight = Absolute(0x1000);
        operands.gradient = Absolute(0x2000);
        operands.updated_weight = operands.weight;
        operands.element_count = 8;
        const double learning_rate = 0.01;
        std::memcpy(&operands.learning_rate_f64_bits, &learning_rate,
                    sizeof(learning_rate));
        record.operands = operands;
        break;
    }
    case RecordOperandKind::DTE_SEND: {
        DteSendOperands operands;
        operands.mode = DteSendMode::P2P;
        operands.source_space = EndpointSourceSpace::SRAM;
        operands.completion = EndpointCompletion::ASYNC;
        operands.token = 7;
        operands.fsm_id = 3;
        operands.length_bytes = 16;
        operands.source = Absolute(0x100000001ULL);
        operands.peer_core = 5;
        record.operands = operands;
        break;
    }
    case RecordOperandKind::DTE_RECV: {
        DteRecvOperands operands;
        operands.mode = DteRecvMode::P2P;
        operands.completion = EndpointCompletion::ASYNC;
        operands.token = 8;
        operands.fsm_id = 4;
        operands.length_bytes = 16;
        operands.destination = Absolute(0x200000002ULL);
        operands.peer_core = 6;
        record.operands = operands;
        break;
    }
    case RecordOperandKind::REDUCE_COMPUTE: {
        ReduceComputeOperands operands;
        operands.datatype = EndpointDataType::INT32;
        operands.reduce_op = ReduceOperator::SUM;
        operands.group_id = 9;
        operands.collective_id = 10;
        operands.epoch = 11;
        operands.root_rank = 0;
        operands.self_rank = 0;
        operands.element_count = 16;
        operands.source = Absolute(0x300000000ULL);
        operands.destination = Absolute(0x400000000ULL);
        record.operands = operands;
        break;
    }
    case RecordOperandKind::LOCAL_REDUCE: {
        LocalReduceOperands operands;
        operands.input_count = 3;
        operands.element_count = 16;
        operands.input_stride_bytes = 32;
        operands.source = Absolute(0x3000);
        operands.destination = Absolute(0x4000);
        record.operands = operands;
        break;
    }
    case RecordOperandKind::LSU: {
        LsuOperands operands;
        operands.hbm_address_bytes = 0x123456789abcdef0ULL;
        operands.size_bytes = 64;
        operands.sram = Absolute(0xfedcba9876543210ULL);
        record.operands = operands;
        break;
    }
    case RecordOperandKind::DTE_ISSUE: {
        DteIssueOperands operands;
        operands.direction = LocalDteDirection::SPM_TO_SPM;
        operands.token = 17;
        operands.payload_bits = 256;
        operands.size_bytes = 32;
        operands.source_sram = Absolute(0x100000001ULL);
        operands.destination_sram = Absolute(0x200000002ULL);
        record.operands = operands;
        break;
    }
    case RecordOperandKind::SYMBOL:
        record.operands = SymbolOperands{3};
        break;
    case RecordOperandKind::SRAM_BIND: {
        SramBindOperands operands;
        operands.input_count = 2;
        operands.input_symbol_indices[0] = 3;
        operands.input_symbol_indices[1] = 4;
        operands.output_symbol_index = 5;
        record.operands = operands;
        break;
    }
    case RecordOperandKind::SRAM_ALLOC: {
        SramAllocOperands operands;
        operands.region_name_string_index = 2;
        operands.label_symbol_index = 3;
        operands.size_bytes = 64;
        operands.alignment_bytes = 16;
        record.operands = operands;
        break;
    }
    case RecordOperandKind::SRAM_ALLOC_AT: {
        SramAllocAtOperands operands;
        operands.region_name_string_index = 2;
        operands.label_symbol_index = 3;
        operands.region_offset_bytes = 192;
        operands.size_bytes = 64;
        operands.alignment_bytes = 16;
        record.operands = operands;
        break;
    }
    case RecordOperandKind::SRAM_RESIZE:
        record.operands = SramResizeOperands{3, 128};
        break;
    case RecordOperandKind::SRAM_RENAME:
        record.operands = SramRenameOperands{3, 4};
        break;
    case RecordOperandKind::TOKEN:
        record.operands = TokenOperands{19};
        break;
    case RecordOperandKind::NONE:
        record.operands = NoOperands{};
        break;
    case RecordOperandKind::EVENT_SET:
        record.operands = EventSetOperands{1, 2, 3};
        break;
    case RecordOperandKind::EVENT_WAIT:
        record.operands = EventWaitOperands{1, 2, 3, 4};
        break;
    case RecordOperandKind::GROUP_SYNC:
        record.operands = GroupSyncOperands{3, 4};
        break;
    }
    return record;
}

bool IsP2Supported(Opcode opcode) noexcept {
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
    // Opcode::DUMMY is hidden by a legacy build macro after Prim includes.
    case static_cast<Opcode>(0x15):
    case Opcode::LSU_LOAD:
    case Opcode::LSU_STORE:
    case Opcode::DTE_SEND:
    case Opcode::DTE_RECV:
    case Opcode::LOCAL_REDUCE:
    case Opcode::DTE_ISSUE:
    case Opcode::SRAM_BIND:
    case Opcode::SRAM_CLEAR:
    case Opcode::SRAM_ALLOC:
    case Opcode::SRAM_ALLOC_AT:
    case Opcode::SRAM_FREE:
    case Opcode::SRAM_RESIZE:
    case Opcode::SRAM_RENAME:
    case Opcode::DTE_WAIT:
    case Opcode::DTE_FENCE:
    case Opcode::DTE_CANCEL:
    case Opcode::EVENT_SET:
    case Opcode::EVENT_WAIT:
    case Opcode::GROUP_SYNC:
    case Opcode::ROPE_QK_EXACT:
    case Opcode::ATTENTION_EXACT:
    case Opcode::EMBEDDING_LOOKUP:
    case Opcode::GREEDY_SAMPLE:
    case Opcode::CROSS_ENTROPY_FORWARD:
    case Opcode::CROSS_ENTROPY_BACKWARD:
    case Opcode::SGD_UPDATE:
        return true;
    default:
        return false;
    }
}

bool SameWire(const std::vector<sc_bv<128>> &left,
              const std::vector<sc_bv<128>> &right) {
    if (left.size() != right.size()) return false;
    for (std::size_t i = 0; i < left.size(); ++i)
        if (left[i] != right[i]) return false;
    return true;
}

int ExpectedCategory(Opcode opcode) {
    if (OpcodeValue(opcode) <= kComputeOpcodeLast) return COMP_PRIM;
    if (opcode == Opcode::DTE_SEND || opcode == Opcode::DTE_RECV ||
        opcode == Opcode::LOCAL_REDUCE)
        return COMM_PRIM;
    if (opcode == Opcode::LSU_LOAD || opcode == Opcode::LSU_STORE ||
        opcode == Opcode::DTE_ISSUE || opcode == Opcode::SRAM_BIND ||
        opcode == Opcode::SRAM_ALLOC || opcode == Opcode::SRAM_ALLOC_AT ||
        opcode == Opcode::SRAM_FREE ||
        opcode == Opcode::SRAM_RESIZE || opcode == Opcode::SRAM_RENAME ||
        opcode == Opcode::SRAM_CLEAR)
        return MEM_PRIM;
    return SYNC_PRIM;
}

void CheckSupportedFields(Checks &checks, const ExternalRecord &record,
                          PrimBase &base) {
    if (record.opcode == Opcode::ROPE_QK_EXACT ||
        record.opcode == Opcode::ATTENTION_EXACT ||
        record.opcode == Opcode::EMBEDDING_LOOKUP ||
        record.opcode == Opcode::GREEDY_SAMPLE ||
        record.opcode == Opcode::CROSS_ENTROPY_FORWARD ||
        record.opcode == Opcode::CROSS_ENTROPY_BACKWARD ||
        record.opcode == Opcode::SGD_UPDATE) {
        auto *exact = dynamic_cast<Exact_stage2_prim_base *>(&base);
        checks.Check(exact != nullptr && exact->exact_opcode() == record.opcode,
                     "exact Stage2 target type and opcode");
        if (exact == nullptr) return;
        ExternalRecord lowered;
        lowered.opcode = record.opcode;
        if (auto *prim = dynamic_cast<Rope_qk_exact_prim *>(exact))
            lowered.operands = prim->operands;
        else if (auto *prim = dynamic_cast<Attention_exact_prim *>(exact))
            lowered.operands = prim->operands;
        else if (auto *prim = dynamic_cast<Embedding_lookup_prim *>(exact))
            lowered.operands = prim->operands;
        else if (auto *prim = dynamic_cast<Greedy_sample_prim *>(exact))
            lowered.operands = prim->operands;
        else if (auto *prim =
                     dynamic_cast<Cross_entropy_forward_prim *>(exact))
            lowered.operands = prim->operands;
        else if (auto *prim =
                     dynamic_cast<Cross_entropy_backward_prim *>(exact))
            lowered.operands = prim->operands;
        else if (auto *prim = dynamic_cast<Sgd_update_prim *>(exact))
            lowered.operands = prim->operands;
        else {
            checks.Check(false, "exact Stage2 concrete target type");
            return;
        }
        checks.Check(EncodeExternalRecord(lowered) ==
                         EncodeExternalRecord(record),
                     "exact Stage2 operands preserved byte-for-byte");
        const std::vector<int> expected_inputs =
            (record.opcode == Opcode::EMBEDDING_LOOKUP ||
             record.opcode == Opcode::CROSS_ENTROPY_FORWARD)
                ? std::vector<int>{1, 1}
            : record.opcode == Opcode::CROSS_ENTROPY_BACKWARD
                ? std::vector<int>{1, 1, 1}
            : record.opcode == Opcode::SGD_UPDATE
                ? std::vector<int>{1, 1}
                : std::vector<int>{1};
        checks.Check(exact->data_size_input == expected_inputs &&
                         exact->data_chunk ==
                             std::vector<std::pair<std::string, int>>{
                                 {"output", 1}},
                     "exact Stage2 input/output arity initialized");
        return;
    }
    if (OpcodeValue(record.opcode) <= kComputeOpcodeLast) {
        auto *prim = dynamic_cast<NpuBase *>(&base);
        const auto &operands = std::get<ComputeOperands>(record.operands);
        const RecordSchema &schema = *LookupRecordSchema(record.opcode);
        checks.Check(prim != nullptr, "compute target is NpuBase");
        if (prim == nullptr) return;
        checks.Check(prim->datatype == FP16 &&
                         prim->inp_offset ==
                             static_cast<int>(operands.input_offset_bytes) &&
                         prim->data_offset ==
                             static_cast<int>(operands.data_offset_bytes) &&
                         prim->out_offset ==
                             static_cast<int>(operands.output_offset_bytes),
                     "compute datatype and u16 offsets preserved");
        bool parameters_match = prim->param_value.size() == schema.parameter_count;
        for (std::size_t i = 0; i < schema.parameter_count; ++i) {
            const auto found =
                prim->param_value.find(std::string(schema.parameter_names[i]));
            parameters_match = parameters_match &&
                found != prim->param_value.end() &&
                found->second == static_cast<int>(operands.parameters[i]);
        }
        checks.Check(parameters_match,
                     "compute RecordSchema parameters preserved by name");
        if (record.opcode == Opcode::SWIGLU) {
            prim->initialize();
            checks.Check(
                prim->data_size_input == std::vector<int>{4} &&
                    prim->data_chunk ==
                        std::vector<std::pair<std::string, int>>{{"output", 2}},
                "SWIGLU lowering initializes one concat input of 2N");
        }
        return;
    }
    if (record.opcode == Opcode::DTE_SEND) {
        auto *prim = dynamic_cast<Dte_send_endpoint_prim *>(&base);
        const auto &operands =
            std::get<DteSendOperands>(record.operands);
        checks.Check(prim != nullptr,
                     "DTE_SEND endpoint target type");
        if (prim == nullptr) return;
        checks.Check(
            prim->mode == DteEndpointSendMode::P2P &&
                prim->source_space == DteEndpointSourceSpace::SRAM &&
                prim->completion == DteEndpointCompletion::ASYNC &&
                prim->datatype == DteEndpointDataType::UINT8 &&
                prim->reduce_op == DteEndpointReduceOp::NONE &&
                prim->fsm_id == operands.fsm_id &&
                prim->token == operands.token &&
                prim->length_bytes == operands.length_bytes &&
                prim->peer_core == operands.peer_core &&
                prim->expected_sources == operands.expected_sources &&
                prim->tree_id == operands.tree_id &&
                prim->group_id == operands.group_id &&
                prim->collective_id == operands.collective_id &&
                prim->epoch == operands.epoch &&
                prim->source.kind == DteEndpointAddressKind::ABSOLUTE &&
                prim->source.absolute_address_bytes ==
                    operands.source.absolute_address_bytes &&
                prim->source.region.empty() &&
                prim->source.region_offset_bytes == 0,
            "DTE_SEND P2P fields and canonical metadata preserved");
        return;
    }
    if (record.opcode == Opcode::DTE_RECV) {
        auto *prim = dynamic_cast<Dte_recv_endpoint_prim *>(&base);
        const auto &operands =
            std::get<DteRecvOperands>(record.operands);
        checks.Check(prim != nullptr,
                     "DTE_RECV endpoint target type");
        if (prim == nullptr) return;
        checks.Check(
            prim->mode == DteEndpointRecvMode::P2P &&
                prim->completion == DteEndpointCompletion::ASYNC &&
                prim->datatype == DteEndpointDataType::UINT8 &&
                prim->reduce_op == DteEndpointReduceOp::NONE &&
                prim->fsm_id == operands.fsm_id &&
                prim->token == operands.token &&
                prim->length_bytes == operands.length_bytes &&
                prim->peer_core == operands.peer_core &&
                prim->expected_sources == operands.expected_sources &&
                prim->tree_id == operands.tree_id &&
                prim->group_id == operands.group_id &&
                prim->collective_id == operands.collective_id &&
                prim->epoch == operands.epoch &&
                prim->destination.kind ==
                    DteEndpointAddressKind::ABSOLUTE &&
                prim->destination.absolute_address_bytes ==
                    operands.destination.absolute_address_bytes &&
                prim->destination.region.empty() &&
                prim->destination.region_offset_bytes == 0,
            "DTE_RECV P2P fields and canonical metadata preserved");
        return;
    }
    if (record.opcode == Opcode::LOCAL_REDUCE) {
        auto *prim = dynamic_cast<Collective_data_v1_prim *>(&base);
        const auto &operands =
            std::get<LocalReduceOperands>(record.operands);
        checks.Check(
            prim != nullptr &&
                prim->mode == CollectiveDataV1PrimMode::REDUCE &&
                prim->key == CollectiveKey{} && prim->phase_id == 0 &&
                prim->source_address_bytes ==
                    operands.source.absolute_address_bytes &&
                prim->destination_address_bytes ==
                    operands.destination.absolute_address_bytes &&
                prim->length_bytes == operands.element_count * 2 &&
                prim->input_count == operands.input_count &&
                prim->dtype == CollDType::FP16 &&
                prim->reduce_op == CollReduceOp::SUM,
            "LOCAL_REDUCE maps explicit FP16/FP32/RNE/rank-major contract to key-zero strict Prim");
        return;
    }
    if (record.opcode == Opcode::LSU_LOAD ||
        record.opcode == Opcode::LSU_STORE) {
        auto *prim = dynamic_cast<Lsu_mem_prim *>(&base);
        const auto &operands = std::get<LsuOperands>(record.operands);
        checks.Check(prim != nullptr, "LSU target type");
        if (prim == nullptr) return;
        const LsuMemOp expected = record.opcode == Opcode::LSU_LOAD
                                      ? LsuMemOp::LOAD_BLOCKING
                                      : LsuMemOp::STORE_BLOCKING;
        checks.Check(prim->op == expected && prim->token == 0 &&
                         prim->hbm_addr == operands.hbm_address_bytes &&
                         prim->sram_addr ==
                             operands.sram.absolute_address_bytes &&
                         prim->size_bytes == operands.size_bytes &&
                         prim->absolute_sram,
                     "LSU blocking variant and full addresses");
        return;
    }
    if (record.opcode == Opcode::SRAM_BIND) {
        auto *prim = dynamic_cast<Sram_bind_oneshot *>(&base);
        const auto &operands = std::get<SramBindOperands>(record.operands);
        bool inputs_match = prim != nullptr &&
                            prim->input_count == operands.input_count;
        if (prim != nullptr) {
            for (std::size_t i = 0; i < operands.input_count; ++i)
                inputs_match = inputs_match &&
                    prim->datapass_label.indata[i] ==
                        "region_" + std::to_string(
                            operands.input_symbol_indices[i]);
            for (std::size_t i = operands.input_count;
                 i < kSramBindInputLimit; ++i)
                inputs_match = inputs_match &&
                    prim->datapass_label.indata[i] == UNSET_LABEL;
        }
        checks.Check(inputs_match &&
                         prim->datapass_label.outdata ==
                             "region_" + std::to_string(
                                 operands.output_symbol_index),
                     "SRAM_BIND active labels and canonical unused slots");
        return;
    }
    if (record.opcode == Opcode::GROUP_SYNC) {
        auto *prim = dynamic_cast<Group_sync_prim *>(&base);
        const auto &operands =
            std::get<GroupSyncOperands>(record.operands);
        checks.Check(prim != nullptr && prim->group_id == operands.group_id &&
                         prim->sync_seq == operands.sync_seq,
                     "GROUP_SYNC operands and target type");
        return;
    }
    if (record.opcode == Opcode::EVENT_SET ||
        record.opcode == Opcode::EVENT_WAIT) {
        auto *prim = dynamic_cast<Event_control_prim *>(&base);
        if (prim == nullptr) {
            checks.Check(false, "EVENT target type");
            return;
        }
        if (record.opcode == Opcode::EVENT_SET) {
            const auto &operands =
                std::get<EventSetOperands>(record.operands);
            checks.Check(prim->op == EventControlOp::SET &&
                             prim->source_core == operands.source_core &&
                             prim->destination_core ==
                                 operands.destination_core &&
                             prim->tag == operands.tag && prim->count == 1,
                         "EVENT_SET operands and canonical count");
        } else {
            const auto &operands =
                std::get<EventWaitOperands>(record.operands);
            checks.Check(prim->op == EventControlOp::WAIT &&
                             prim->source_core == operands.source_core &&
                             prim->destination_core ==
                                 operands.destination_core &&
                             prim->tag == operands.tag &&
                             prim->count == operands.count,
                         "EVENT_WAIT operands and count");
        }
        return;
    }
    if (record.opcode == Opcode::SRAM_CLEAR ||
        record.opcode == Opcode::SRAM_ALLOC ||
        record.opcode == Opcode::SRAM_ALLOC_AT ||
        record.opcode == Opcode::SRAM_FREE ||
        record.opcode == Opcode::SRAM_RESIZE ||
        record.opcode == Opcode::SRAM_RENAME) {
        auto *prim = dynamic_cast<Sram_lifecycle *>(&base);
        checks.Check(prim != nullptr, "SRAM lifecycle target type");
        if (prim == nullptr) return;
        bool fields = prim->region_name.empty() &&
                      prim->new_label.empty() &&
                      prim->region_offset_bytes == 0 &&
                      prim->size_bytes == 0 &&
                      prim->alignment_bytes == 0 &&
                      prim->lifetime == sram::AllocationLifetime::kTask &&
                      !prim->spillable;
        if (record.opcode == Opcode::SRAM_ALLOC) {
            const auto &operands =
                std::get<SramAllocOperands>(record.operands);
            fields = prim->op == SramLifecycleOp::ALLOC &&
                     prim->region_name == "string_" + std::to_string(
                         operands.region_name_string_index) &&
                     prim->label == "region_" + std::to_string(
                         operands.label_symbol_index) &&
                     prim->size_bytes == operands.size_bytes &&
                     prim->alignment_bytes == operands.alignment_bytes &&
                     prim->region_offset_bytes == 0 &&
                     prim->lifetime == sram::AllocationLifetime::kTask &&
                     prim->spillable == operands.spillable &&
                     prim->new_label.empty();
        } else if (record.opcode == Opcode::SRAM_ALLOC_AT) {
            const auto &operands =
                std::get<SramAllocAtOperands>(record.operands);
            fields = prim->op == SramLifecycleOp::ALLOC_AT &&
                     prim->region_name == "string_" + std::to_string(
                         operands.region_name_string_index) &&
                     prim->label == "region_" + std::to_string(
                         operands.label_symbol_index) &&
                     prim->region_offset_bytes ==
                         operands.region_offset_bytes &&
                     prim->size_bytes == operands.size_bytes &&
                     prim->alignment_bytes == operands.alignment_bytes &&
                     prim->lifetime == sram::AllocationLifetime::kTask &&
                     prim->spillable == operands.spillable &&
                     prim->new_label.empty();
        } else if (record.opcode == Opcode::SRAM_RESIZE) {
            const auto &operands =
                std::get<SramResizeOperands>(record.operands);
            fields = prim->op == SramLifecycleOp::RESIZE &&
                     prim->label == "region_" + std::to_string(
                         operands.symbol_index) &&
                     prim->size_bytes == operands.new_size_bytes &&
                     prim->region_name.empty() && prim->new_label.empty() &&
                     prim->alignment_bytes == 0 &&
                     prim->lifetime == sram::AllocationLifetime::kTask &&
                     !prim->spillable;
        } else if (record.opcode == Opcode::SRAM_RENAME) {
            const auto &operands =
                std::get<SramRenameOperands>(record.operands);
            fields = prim->op == SramLifecycleOp::RENAME &&
                     prim->label == "region_" + std::to_string(
                         operands.old_symbol_index) &&
                     prim->new_label == "region_" + std::to_string(
                         operands.new_symbol_index) &&
                     prim->region_name.empty() && prim->size_bytes == 0 &&
                     prim->alignment_bytes == 0 &&
                     prim->lifetime == sram::AllocationLifetime::kTask &&
                     !prim->spillable;
        } else {
            const auto &operands =
                std::get<SymbolOperands>(record.operands);
            fields = prim->op ==
                         (record.opcode == Opcode::SRAM_FREE
                              ? SramLifecycleOp::FREE
                              : SramLifecycleOp::CLEAR_TARGETED) &&
                     prim->label == "region_" + std::to_string(
                         operands.symbol_index) && fields;
        }
        checks.Check(fields, "SRAM lifecycle fields map canonically");
        return;
    }
    auto *prim = dynamic_cast<Dte_async_prim *>(&base);
    checks.Check(prim != nullptr, "DTE target type");
    if (prim == nullptr) return;
    if (record.opcode == Opcode::DTE_ISSUE) {
        const auto &operands = std::get<DteIssueOperands>(record.operands);
        checks.Check(prim->op == DteAsyncOp::ISSUE &&
                         prim->direction == DteDir::SPM_TO_SPM &&
                         prim->token == operands.token &&
                         prim->spm_addr ==
                             operands.source_sram.absolute_address_bytes &&
                         prim->remote_addr ==
                             operands.destination_sram.absolute_address_bytes &&
                         prim->spm_size == operands.size_bytes,
                     "DTE_ISSUE variant and full dual addresses");
    } else {
        const DteAsyncOp expected =
            record.opcode == Opcode::DTE_WAIT
                ? DteAsyncOp::WAIT
                : record.opcode == Opcode::DTE_CANCEL ? DteAsyncOp::CANCEL
                                                       : DteAsyncOp::FENCE;
        const uint64_t expected_token = record.opcode == Opcode::DTE_FENCE
                                            ? 0
                                            : std::get<TokenOperands>(
                                                  record.operands).token;
        checks.Check(prim->op == expected && prim->token == expected_token &&
                         prim->payload_bits == 0,
                     "DTE control variant and token");
    }
}

void CheckManifestMatrix(Checks &checks) {
    LoweringContext context = Context();
    std::size_t supported = 0;
    std::size_t deferred = 0;
    std::size_t gated = 0;
    std::size_t reserved = 0;
    std::size_t unsupported = 0;
    for (const OpcodeManifestEntry &entry : OpcodeManifest()) {
        const ExternalRecord record =
            MakeRecord(*LookupRecordSchema(entry.opcode));
        const std::string name(entry.canonical_name);
        const OpcodeValidation validation = ValidateOpcode(entry.opcode);
        if (validation == OpcodeValidation::RESERVED) {
            ++reserved;
            checks.Reject<RecordCodecError>(name + " reserved status",
                                            "reserved", [&] {
                LoweringContext enabled = context;
                enabled.enabled_capabilities = entry.required_capabilities;
                LowerExternalRecord(record, enabled);
            });
            continue;
        }
        if (validation == OpcodeValidation::UNSUPPORTED) {
            ++unsupported;
            checks.Reject<RecordCodecError>(name + " unsupported status",
                                            "unsupported", [&] {
                LowerExternalRecord(record, context);
            });
            continue;
        }
        if (entry.support == OpcodeSupport::EXPERIMENTAL) {
            ++gated;
            checks.Reject<RecordCodecError>(name + " gated status",
                                            "capability", [&] {
                LowerExternalRecord(record, context);
            });
            LoweringContext enabled = context;
            enabled.enabled_capabilities = entry.required_capabilities;
            checks.Reject<LoweringUnavailableError>(
                name + " enabled capability remains deferred", name, [&] {
                    LowerExternalRecord(record, enabled);
                });
            continue;
        }
        if (!IsP2Supported(entry.opcode)) {
            ++deferred;
            checks.Reject<LoweringUnavailableError>(
                name + " deferred lowering", name,
                [&] { LowerExternalRecord(record, context); });
            continue;
        }
        ++supported;
        try {
            LoweredPrimList first = LowerExternalRecord(record, context);
            LoweredPrimList second = LowerExternalRecord(record, context);
            checks.Check(first.size() == 1 && second.size() == 1,
                         name + " lowers to one owned Prim");
            if (first.size() != 1 || second.size() != 1) continue;
            CheckSupportedFields(checks, record, *first.front());
            checks.Check(PrimMainCategoryBits(first.front()->prim_type) ==
                             ExpectedCategory(entry.opcode),
                         name + " primary category");
            const auto wire = first.front()->serialize();
            checks.Check(SameWire(wire, second.front()->serialize()),
                         name + " deterministic repeated lowering");
            std::unique_ptr<PrimBase> decoded(
                PrimFactory::getInstance().createPrim(
                    static_cast<int>(wire.front().range(7, 0).to_uint()),
                    false, false));
            decoded->deserialize(wire);
            checks.Check(decoded->name == first.front()->name &&
                             PrimMainCategoryBits(decoded->prim_type) ==
                                 ExpectedCategory(entry.opcode) &&
                             SameWire(decoded->serialize(), wire),
                         name + " internal wire/factory round-trip");
        } catch (const std::exception &error) {
            checks.Check(false, name + " unexpected lowering error: " +
                                    error.what());
        }
    }
    checks.Check(supported == 43,
                 "supported opcode count including exact Stage2 records");
    checks.Check(deferred == 1, "remaining P6 deferred opcode count");
    checks.Check(gated == 4, "capability-gated opcode count");
    checks.Check(reserved == 1, "reserved opcode count");
    checks.Check(unsupported == 4, "known unsupported opcode count");
}

void CheckDteEndpointLowering(Checks &checks) {
    LoweringContext context = Context();

    DteSendOperands send;
    send.mode = DteSendMode::P2P;
    send.source_space = EndpointSourceSpace::SRAM;
    send.completion = EndpointCompletion::ASYNC;
    send.datatype = EndpointDataType::UINT8;
    send.reduce_op = ReduceOperator::NONE;
    send.fsm_id = 0x10001U;
    send.token = 0xffffffffU;
    send.length_bytes = kDteEndpointP2pMaxBytes;
    send.source = Region(7, UINT64_C(0x123456789abcdef0));
    send.peer_core = 6;
    LoweredPrimList lowered_send = LowerExternalRecord(
        ExternalRecord{Opcode::DTE_SEND, send}, context);
    auto *send_prim = dynamic_cast<Dte_send_endpoint_prim *>(
        lowered_send.front().get());
    checks.Check(
        send_prim != nullptr &&
            send_prim->fsm_id == 0x10001U &&
            send_prim->token == 0xffffffffU &&
            send_prim->source_space == DteEndpointSourceSpace::SRAM &&
            send_prim->length_bytes == kDteEndpointP2pMaxBytes &&
            send_prim->peer_core == 6 &&
            send_prim->source.kind == DteEndpointAddressKind::REGION &&
            send_prim->source.absolute_address_bytes == 0 &&
            send_prim->source.region == "region_7" &&
            send_prim->source.region_offset_bytes ==
                UINT64_C(0x123456789abcdef0),
        "DTE_SEND preserves full fields and resolves a region address");

    DteRecvOperands recv;
    recv.mode = DteRecvMode::P2P;
    recv.completion = EndpointCompletion::SYNC;
    recv.datatype = EndpointDataType::UINT8;
    recv.reduce_op = ReduceOperator::NONE;
    recv.fsm_id = 0xffffffffU;
    recv.token = 0;
    recv.length_bytes = kDteEndpointP2pMaxBytes;
    recv.destination = Region(8, UINT64_MAX);
    recv.peer_core = 5;
    LoweredPrimList lowered_recv = LowerExternalRecord(
        ExternalRecord{Opcode::DTE_RECV, recv}, context);
    auto *recv_prim = dynamic_cast<Dte_recv_endpoint_prim *>(
        lowered_recv.front().get());
    checks.Check(
        recv_prim != nullptr &&
            recv_prim->completion == DteEndpointCompletion::SYNC &&
            recv_prim->fsm_id == 0xffffffffU && recv_prim->token == 0 &&
            recv_prim->length_bytes == kDteEndpointP2pMaxBytes &&
            recv_prim->peer_core == 5 &&
            recv_prim->destination.kind ==
                DteEndpointAddressKind::REGION &&
            recv_prim->destination.absolute_address_bytes == 0 &&
            recv_prim->destination.region == "region_8" &&
            recv_prim->destination.region_offset_bytes == UINT64_MAX,
        "DTE_RECV preserves sync token zero and full region fields");

    DteSendOperands hbm = send;
    hbm.source_space = EndpointSourceSpace::HBM;
    hbm.source = Absolute(UINT64_MAX);
    LoweredPrimList hbm_lowered = LowerExternalRecord(
        ExternalRecord{Opcode::DTE_SEND, hbm}, context);
    auto *hbm_prim =
        hbm_lowered.empty()
            ? nullptr
            : dynamic_cast<Dte_send_endpoint_prim *>(
                  hbm_lowered.front().get());
    checks.Check(
        hbm_lowered.size() == 1 && hbm_prim != nullptr &&
            hbm_prim->source_space == DteEndpointSourceSpace::HBM &&
            hbm_prim->source.kind == DteEndpointAddressKind::ABSOLUTE &&
            hbm_prim->source.absolute_address_bytes == UINT64_MAX &&
            hbm_prim->source.region.empty() &&
            hbm_prim->source.region_offset_bytes == 0,
        "DTE_SEND HBM source lowers to a canonical absolute endpoint");

    DteSendOperands scatter = send;
    scatter.mode = DteSendMode::SCATTER;
    scatter.completion = EndpointCompletion::SYNC;
    scatter.token = 0;
    scatter.source = Absolute(1);
    scatter.peer_core = 0;
    scatter.group_id = 1;
    scatter.collective_id = 2;
    scatter.epoch = 3;
    checks.Reject<LoweringUnavailableError>(
        "DTE_SEND SCATTER waits for P6", "whole-artifact", [&] {
            LowerExternalRecord(ExternalRecord{Opcode::DTE_SEND, scatter},
                                context);
        });

    DteSendOperands broadcast = scatter;
    broadcast.mode = DteSendMode::BROADCAST;
    broadcast.tree_id = 0;
    checks.Reject<LoweringUnavailableError>(
        "DTE_SEND BROADCAST waits for P6", "whole-artifact", [&] {
            LowerExternalRecord(ExternalRecord{Opcode::DTE_SEND, broadcast},
                                context);
        });

    DteSendOperands keyed_send = send;
    keyed_send.peer_core = 0;
    keyed_send.group_id = 1;
    keyed_send.collective_id = 2;
    keyed_send.epoch = 3;
    checks.Reject<LoweringUnavailableError>(
        "keyed DTE_SEND UNICAST cannot bypass the graph",
        "whole-artifact", [&] {
            LowerExternalRecord(
                ExternalRecord{Opcode::DTE_SEND, keyed_send}, context);
        });

    DteRecvOperands keyed_receive = recv;
    keyed_receive.completion = EndpointCompletion::ASYNC;
    keyed_receive.token = 4;
    keyed_receive.peer_core = 0;
    keyed_receive.group_id = 1;
    keyed_receive.collective_id = 2;
    keyed_receive.epoch = 3;
    checks.Reject<LoweringUnavailableError>(
        "keyed DTE_RECV UNICAST cannot bypass the graph",
        "whole-artifact", [&] {
            LowerExternalRecord(
                ExternalRecord{Opcode::DTE_RECV, keyed_receive}, context);
        });

    DteRecvOperands gather = recv;
    gather.mode = DteRecvMode::GATHER;
    gather.destination = Absolute(2);
    gather.peer_core = 0;
    gather.expected_sources = 2;
    gather.group_id = 1;
    gather.collective_id = 2;
    gather.epoch = 3;
    checks.Reject<LoweringUnavailableError>(
        "DTE_RECV GATHER waits for P6", "whole-artifact", [&] {
            LowerExternalRecord(ExternalRecord{Opcode::DTE_RECV, gather},
                                context);
        });

    DteRecvOperands reduce = gather;
    reduce.mode = DteRecvMode::REDUCE;
    reduce.datatype = EndpointDataType::INT32;
    reduce.reduce_op = ReduceOperator::SUM;
    checks.Reject<LoweringUnavailableError>(
        "DTE_RECV REDUCE waits for P6", "whole-artifact", [&] {
            LowerExternalRecord(ExternalRecord{Opcode::DTE_RECV, reduce},
                                context);
        });
}

void CheckExactStage2Wire(Checks &checks) {
    const ExternalRecord record =
        MakeRecord(*LookupRecordSchema(Opcode::ATTENTION_EXACT));
    LoweredPrimList lowered = LowerExternalRecord(record, Context());
    checks.Check(lowered.size() == 1,
                 "exact Stage2 wire fixture lowers once");
    if (lowered.size() != 1) return;
    auto *source = dynamic_cast<Attention_exact_prim *>(lowered.front().get());
    checks.Check(source != nullptr, "exact attention wire source type");
    if (source == nullptr) return;
    const std::vector<sc_bv<128>> wire = source->serialize();
    checks.Check(wire.size() == 10,
                 "exact attention wire has frozen segment count");
    checks.Reject<std::invalid_argument>(
        "exact attention rejects legacy transport", "strict-only", [&] {
            (void)prim_wire::LegacyTransportSegments(wire, source->name);
        });

    std::vector<sc_bv<128>> bad = wire;
    bad[0].range(23, 16) = sc_bv<8>(2);
    checks.Reject<std::invalid_argument>(
        "exact attention rejects internal wire version", "version", [&] {
            Attention_exact_prim decoded;
            decoded.deserialize(bad);
        });
    bad = wire;
    bad[1].range(15, 8) = sc_bv<8>(0);
    checks.Reject<std::invalid_argument>(
        "exact attention rejects segment ordinal", "ordinal", [&] {
            Attention_exact_prim decoded;
            decoded.deserialize(bad);
        });
    bad = wire;
    bad.back().range(127, 120) = sc_bv<8>(1);
    checks.Reject<std::invalid_argument>(
        "exact attention rejects nonzero wire padding", "unused", [&] {
            Attention_exact_prim decoded;
            decoded.deserialize(bad);
        });
}

DteIssueOperands BaseIssue(LocalDteDirection direction) {
    DteIssueOperands operands;
    operands.direction = direction;
    operands.token = 0xffffffffU;
    operands.payload_bits = 512;
    operands.size_bytes = 64;
    if (direction == LocalDteDirection::SPM_TO_SPM) {
        operands.source_sram = Absolute(0x123456789abcdef0ULL);
        operands.destination_sram = Absolute(0xfedcba9876543210ULL);
    } else if (direction == LocalDteDirection::SPM_TO_DRAM) {
        operands.source_sram = Absolute(0x123456789abcdef0ULL);
        operands.hbm_address_bytes = 0xfedcba9876543210ULL;
    } else {
        operands.destination_sram = Absolute(0x123456789abcdef0ULL);
        operands.hbm_address_bytes = 0xfedcba9876543210ULL;
    }
    return operands;
}

ExternalRecord IssueRecord(DteIssueOperands operands) {
    ExternalRecord record;
    record.opcode = Opcode::DTE_ISSUE;
    record.operands = std::move(operands);
    return record;
}

ProgramArtifact CollectiveArtifact(CollTxKind tx, CollRxKind rx,
                                   std::size_t n) {
    ProgramArtifact artifact;
    artifact.core_groups = {{7, {}}};
    for (std::size_t rank = 0; rank < n; ++rank)
        artifact.core_groups[0].members.push_back(rank);
    const CollOp op = IsaV1CollectiveOp(tx, rx);
    const std::size_t root = n == 1 ? 0 : 1;
    for (std::size_t rank = 0; rank < n; ++rank) {
        ProgramCore core;
        core.core_id = rank;
        const bool send_role =
            op == CollOp::SCATTER || op == CollOp::BROADCAST
                ? rank == root
                : true;
        const bool receive_role =
            op == CollOp::GATHER || op == CollOp::REDUCE
                ? rank == root
                : true;
        const bool compute_role =
            op == CollOp::REDUCE
                ? rank == root
                : op == CollOp::REDUCESCATTER || op == CollOp::ALLREDUCE;
        if (send_role) {
            DteSendOperands send;
            send.mode = tx == CollTxKind::SCATTER
                            ? DteSendMode::SCATTER
                            : tx == CollTxKind::BROADCAST
                                  ? DteSendMode::BROADCAST
                                  : DteSendMode::P2P;
            send.completion = EndpointCompletion::ASYNC;
            send.fsm_id = 0x1000;
            send.token = 100 + rank * 4 + 1;
            send.length_bytes = 32;
            send.source = Absolute(0x10000 + rank * 0x1000);
            send.group_id = 7;
            send.collective_id = 11;
            core.records.push_back({Opcode::DTE_SEND, send});
        }
        if (receive_role) {
            DteRecvOperands receive;
            receive.mode = rx == CollRxKind::GATHER
                               ? DteRecvMode::GATHER
                               : rx == CollRxKind::REDUCE
                                     ? DteRecvMode::REDUCE
                                     : DteRecvMode::P2P;
            receive.completion = EndpointCompletion::ASYNC;
            receive.fsm_id = 0x1000;
            receive.token = 100 + rank * 4 + 2;
            receive.length_bytes = 32;
            receive.destination = Absolute(0x20000 + rank * 0x1000);
            receive.expected_sources =
                rx == CollRxKind::UNICAST
                    ? 0
                    : static_cast<uint16_t>(n - 1);
            receive.datatype = rx == CollRxKind::REDUCE
                                   ? EndpointDataType::INT32
                                   : EndpointDataType::UINT8;
            receive.reduce_op = rx == CollRxKind::REDUCE
                                    ? ReduceOperator::SUM
                                    : ReduceOperator::NONE;
            receive.group_id = 7;
            receive.collective_id = 11;
            core.records.push_back({Opcode::DTE_RECV, receive});
        }
        if (compute_role) {
            ReduceComputeOperands compute;
            compute.datatype = EndpointDataType::INT32;
            compute.reduce_op = ReduceOperator::SUM;
            compute.group_id = 7;
            compute.collective_id = 11;
            compute.root_rank = op == CollOp::REDUCE ? root : 0;
            compute.self_rank = rank;
            compute.element_count = 8;
            compute.source = Absolute(0x20000 + rank * 0x1000);
            compute.destination = Absolute(0x30000 + rank * 0x1000);
            core.records.push_back({Opcode::REDUCE_COMPUTE, compute});
        }
        artifact.cores.push_back(std::move(core));
        artifact.envelope.active_cores.push_back(rank);
        artifact.envelope.expected_ack_cores.push_back(rank);
    }
    artifact.envelope.terminal_cores = {n - 1};
    artifact.envelope.expected_done_cores = {n - 1};
    return artifact;
}

void CheckWholeArtifactCollectiveLowering(Checks &checks) {
    struct Cell { CollTxKind tx; CollRxKind rx; };
    const std::vector<Cell> cells = {
        {CollTxKind::SCATTER, CollRxKind::UNICAST},
        {CollTxKind::BROADCAST, CollRxKind::UNICAST},
        {CollTxKind::UNICAST, CollRxKind::GATHER},
        {CollTxKind::SCATTER, CollRxKind::GATHER},
        {CollTxKind::BROADCAST, CollRxKind::GATHER},
        {CollTxKind::UNICAST, CollRxKind::REDUCE},
        {CollTxKind::SCATTER, CollRxKind::REDUCE},
        {CollTxKind::BROADCAST, CollRxKind::REDUCE},
    };
    for (const Cell &cell : cells) {
        ProgramArtifact artifact = CollectiveArtifact(cell.tx, cell.rx, 4);
        const auto first = LowerIsaV1CollectiveArtifact(artifact, 8, 8);
        for (ProgramCore &core : artifact.cores)
            std::reverse(core.records.begin(), core.records.end());
        const auto reversed = LowerIsaV1CollectiveArtifact(artifact, 8, 8);
        checks.Check(first == reversed && first.plans.size() == 1 &&
                         !first.executable,
                     "whole-artifact collective lowering is deterministic and explicitly non-executable");
        checks.Check(!first.core_actions.empty(),
                     "collective plan emits deterministic per-core action streams");
        checks.Reject<RecordLoweringError>(
            "every collective mode enforces one-die membership",
            "ISA-v1 whole-artifact collective graph: collective "
            "key(7,11,0) core 0 record 0: ISA-v1 non-P2P collective "
            "group crosses dies", [&] {
                LowerIsaV1CollectiveArtifact(artifact, 8, 2);
            });
    }

    ProgramArtifact n1 = CollectiveArtifact(
        CollTxKind::UNICAST, CollRxKind::REDUCE, 1);
    const auto n1_lowered = LowerIsaV1CollectiveArtifact(n1, 8, 8);
    checks.Check(n1_lowered.plans.size() == 1 &&
                     n1_lowered.children.empty() &&
                     n1_lowered.plans[0].local_copies.size() == 1,
                 "N=1 Reduce accepts expected_sources=0 and emits no remote child");

    ProgramArtifact tokens = CollectiveArtifact(
        CollTxKind::SCATTER, CollRxKind::GATHER, 4);
    DteIssueOperands issue;
    issue.token = UINT32_MAX;
    issue.payload_bits = 8;
    issue.size_bytes = 1;
    issue.source_sram = Absolute(1);
    issue.destination_sram = Absolute(2);
    tokens.cores[0].records.push_back({Opcode::DTE_ISSUE, issue});
    const auto token_lowered = LowerIsaV1CollectiveArtifact(tokens, 8, 8);
    std::set<uint32_t> internal_tokens;
    std::set<uint32_t> reserved_tokens{UINT32_MAX};
    for (const ProgramCore &core : tokens.cores) {
        for (const ExternalRecord &record : core.records) {
            if (const auto *send = std::get_if<DteSendOperands>(&record.operands))
                reserved_tokens.insert(static_cast<uint32_t>(send->token));
            if (const auto *receive = std::get_if<DteRecvOperands>(&record.operands))
                reserved_tokens.insert(static_cast<uint32_t>(receive->token));
        }
    }
    bool unique = true;
    for (const auto &child : token_lowered.children) {
        unique = unique && child.source_internal_token != 0 &&
                 child.destination_internal_token != 0 &&
                 reserved_tokens.count(child.source_internal_token) == 0 &&
                 reserved_tokens.count(child.destination_internal_token) == 0 &&
                 internal_tokens.insert(child.source_internal_token).second &&
                 internal_tokens.insert(child.destination_internal_token).second;
    }
    for (const auto &stream : token_lowered.core_actions)
        for (const auto &action : stream.actions)
            if (action.internal_token != 0)
                unique = unique && action.internal_token !=
                                       action.public_aggregate_token;
    checks.Check(unique && token_lowered.children.size() > 4,
                 "multi-child internal tokens are unique and avoid every public/local DTE token");

    ProgramArtifact regions = CollectiveArtifact(
        CollTxKind::BROADCAST, CollRxKind::UNICAST, 2);
    regions.strings = {"collective_source", "collective_destination"};
    regions.symbols = {
        {0, ProgramSymbolKind::SRAM_REGION, 0, 0x1000, 4096},
        {1, ProgramSymbolKind::SRAM_REGION, 0, 0x2000, 4096}};
    for (ProgramCore &core : regions.cores) {
        for (ExternalRecord &record : core.records) {
            if (auto *send = std::get_if<DteSendOperands>(&record.operands))
                send->source = Region(0, 13);
            if (auto *receive = std::get_if<DteRecvOperands>(&record.operands))
                receive->destination = Region(1, 29);
        }
    }
    const auto region_lowered = LowerIsaV1CollectiveArtifact(regions, 8, 8);
    checks.Check(region_lowered.children.size() == 1 &&
                     region_lowered.children[0].source.kind ==
                         SramAddressKind::ABSOLUTE &&
                     region_lowered.children[0].source.absolute_address_bytes ==
                         0x100d &&
                     region_lowered.children[0].destination.kind ==
                         SramAddressKind::ABSOLUTE &&
                     region_lowered.children[0].destination
                             .absolute_address_bytes == 0x201d,
                 "collective child resolves region base and offset to "
                 "canonical absolute addresses");

    ProgramArtifact scatter_exact = CollectiveArtifact(
        CollTxKind::SCATTER, CollRxKind::UNICAST, 2);
    scatter_exact.strings = {"scatter_source", "scatter_destination"};
    scatter_exact.symbols = {
        {0, ProgramSymbolKind::SRAM_REGION, 0, 0x3000, 77},
        {1, ProgramSymbolKind::SRAM_REGION, 0, 0x4000, 61}};
    for (ProgramCore &core : scatter_exact.cores) {
        for (ExternalRecord &record : core.records) {
            if (auto *send = std::get_if<DteSendOperands>(&record.operands))
                send->source = Region(0, 13);
            if (auto *receive = std::get_if<DteRecvOperands>(&record.operands))
                receive->destination = Region(1, 29);
        }
    }
    checks.Check(
        LowerIsaV1CollectiveArtifact(scatter_exact, 8, 8).plans.size() == 1,
        "SCATTER accepts exact N*L source and L destination region spans");
    ProgramArtifact scatter_short = scatter_exact;
    scatter_short.symbols[0].size_bytes = 76;
    checks.Reject<RecordLoweringError>(
        "SCATTER rejects source region one byte shorter than N*L",
        "SEND source span exceeds its region symbol size", [&] {
            LowerIsaV1CollectiveArtifact(scatter_short, 8, 8);
        });

    ProgramArtifact gather_exact = CollectiveArtifact(
        CollTxKind::UNICAST, CollRxKind::GATHER, 2);
    gather_exact.strings = {"gather_source", "gather_destination"};
    gather_exact.symbols = {
        {0, ProgramSymbolKind::SRAM_REGION, 0, 0x5000, 45},
        {1, ProgramSymbolKind::SRAM_REGION, 0, 0x6000, 93}};
    for (ProgramCore &core : gather_exact.cores) {
        for (ExternalRecord &record : core.records) {
            if (auto *send = std::get_if<DteSendOperands>(&record.operands))
                send->source = Region(0, 13);
            if (auto *receive = std::get_if<DteRecvOperands>(&record.operands))
                receive->destination = Region(1, 29);
        }
    }
    checks.Check(
        LowerIsaV1CollectiveArtifact(gather_exact, 8, 8).plans.size() == 1,
        "GATHER accepts exact L source and N*L destination region spans");
    ProgramArtifact gather_short = gather_exact;
    gather_short.symbols[1].size_bytes = 92;
    checks.Reject<RecordLoweringError>(
        "GATHER rejects destination region one byte shorter than N*L",
        "RECEIVE destination span exceeds its region symbol size", [&] {
            LowerIsaV1CollectiveArtifact(gather_short, 8, 8);
        });

    ProgramArtifact reduce_result = CollectiveArtifact(
        CollTxKind::UNICAST, CollRxKind::REDUCE, 2);
    reduce_result.strings = {"reduce_source", "reduce_staging",
                             "reduce_result"};
    reduce_result.symbols = {
        {0, ProgramSymbolKind::SRAM_REGION, 0, 0x7000, 44},
        {1, ProgramSymbolKind::SRAM_REGION, 0, 0x8000, 92},
        {2, ProgramSymbolKind::SRAM_REGION, 0, 0x9000, 40}};
    for (ProgramCore &core : reduce_result.cores) {
        for (ExternalRecord &record : core.records) {
            if (auto *send = std::get_if<DteSendOperands>(&record.operands))
                send->source = Region(0, 12);
            if (auto *receive = std::get_if<DteRecvOperands>(&record.operands))
                receive->destination = Region(1, 28);
            if (auto *compute =
                    std::get_if<ReduceComputeOperands>(&record.operands)) {
                compute->source = Region(1, 28);
                compute->destination = Region(2, 8);
            }
        }
    }
    checks.Check(
        LowerIsaV1CollectiveArtifact(reduce_result, 8, 8).plans.size() == 1,
        "REDUCE accepts exact N*L staging and L result region spans");
    ProgramArtifact reduce_result_short = reduce_result;
    reduce_result_short.symbols[2].size_bytes = 39;
    checks.Reject<RecordLoweringError>(
        "REDUCE rejects result region one byte shorter than L",
        "REDUCE destination span exceeds its region symbol size", [&] {
            LowerIsaV1CollectiveArtifact(reduce_result_short, 8, 8);
        });

    ProgramArtifact absolute_overflow = CollectiveArtifact(
        CollTxKind::BROADCAST, CollRxKind::UNICAST, 2);
    for (ProgramCore &core : absolute_overflow.cores) {
        for (ExternalRecord &record : core.records) {
            if (auto *send = std::get_if<DteSendOperands>(&record.operands))
                send->source = Absolute(
                    std::numeric_limits<uint64_t>::max() - 30);
        }
    }
    checks.Reject<ProgramFormatError>(
        "collective endpoint rejects absolute span overflow",
        "source absolute span overflows u64", [&] {
            LowerIsaV1CollectiveArtifact(absolute_overflow, 8, 8);
        });

    ProgramArtifact bad_expected = CollectiveArtifact(
        CollTxKind::UNICAST, CollRxKind::GATHER, 4);
    auto &bad_receive = std::get<DteRecvOperands>(
        bad_expected.cores[1].records[1].operands);
    bad_receive.expected_sources = 0;
    checks.Reject<RecordLoweringError>(
        "N>1 expected_sources=0 is rejected by whole graph",
        "ISA-v1 whole-artifact collective graph: collective "
        "key(7,11,0) core 0 record 0: ISA-v1 RECEIVE expected_sources "
        "does not match N-1/zero", [&] {
            LowerIsaV1CollectiveArtifact(bad_expected, 8, 8);
        });

    ProgramArtifact reduce_artifact = CollectiveArtifact(
        CollTxKind::UNICAST, CollRxKind::REDUCE, 2);
    const ExternalRecord &reduce_record = reduce_artifact.cores[1].records[2];
    checks.Reject<LoweringUnavailableError>(
        "REDUCE_COMPUTE never lowers to legacy timing-only Prim",
        "legacy timing-only", [&] {
            LowerExternalRecord(reduce_record, Context());
        });
}

void CheckDteDirectionsAndResolvers(Checks &checks) {
    LoweringContext context = Context();
    for (LocalDteDirection direction : {
             LocalDteDirection::SPM_TO_SPM,
             LocalDteDirection::SPM_TO_DRAM,
             LocalDteDirection::DRAM_TO_SPM}) {
        ExternalRecord record = IssueRecord(BaseIssue(direction));
        LoweredPrimList lowered = LowerExternalRecord(record, context);
        auto *prim = dynamic_cast<Dte_async_prim *>(lowered.front().get());
        checks.Check(prim != nullptr &&
                         prim->spm_addr == 0x123456789abcdef0ULL &&
                         prim->remote_addr == 0xfedcba9876543210ULL,
                     "DTE direction preserves absolute 64-bit addresses " +
                         std::to_string(static_cast<unsigned>(direction)));
    }

    for (LocalDteDirection direction : {
             LocalDteDirection::SPM_TO_DRAM,
             LocalDteDirection::DRAM_TO_SPM}) {
        DteIssueOperands operands = BaseIssue(direction);
        if (direction == LocalDteDirection::SPM_TO_DRAM)
            operands.source_sram = Region(7, 0x123456789ULL);
        else
            operands.destination_sram = Region(7, 0x123456789ULL);
        LoweredPrimList lowered =
            LowerExternalRecord(IssueRecord(operands), context);
        auto *prim = dynamic_cast<Dte_async_prim *>(lowered.front().get());
        checks.Check(prim != nullptr && prim->spm_addr == 0 &&
                         prim->sram_region == "region_7" &&
                         prim->sram_offset == 0x123456789ULL,
                     "DTE DRAM direction resolves named SRAM region");
    }

    DteIssueOperands named_copy = BaseIssue(LocalDteDirection::SPM_TO_SPM);
    named_copy.source_sram = Region(7, 3);
    LoweredPrimList source_named =
        LowerExternalRecord(IssueRecord(named_copy), context);
    auto *source_prim =
        dynamic_cast<Dte_async_prim *>(source_named.front().get());
    checks.Check(source_prim != nullptr && source_prim->spm_addr == 0 &&
                     source_prim->sram_region == "region_7" &&
                     source_prim->sram_offset == 3 &&
                     source_prim->destination_sram_region.empty() &&
                     source_prim->remote_addr == 0xfedcba9876543210ULL,
                 "named SPM_TO_SPM source lowers independently");

    named_copy = BaseIssue(LocalDteDirection::SPM_TO_SPM);
    named_copy.destination_sram = Region(8, 5);
    LoweredPrimList destination_named =
        LowerExternalRecord(IssueRecord(named_copy), context);
    auto *destination_prim =
        dynamic_cast<Dte_async_prim *>(destination_named.front().get());
    checks.Check(destination_prim != nullptr &&
                     destination_prim->spm_addr == 0x123456789abcdef0ULL &&
                     destination_prim->remote_addr == 0 &&
                     destination_prim->destination_sram_region == "region_8" &&
                     destination_prim->destination_sram_offset == 5,
                 "named SPM_TO_SPM destination lowers independently");

    named_copy.source_sram = Region(7, 3);
    LoweredPrimList both_named =
        LowerExternalRecord(IssueRecord(named_copy), context);
    auto *both_prim = dynamic_cast<Dte_async_prim *>(both_named.front().get());
    checks.Check(both_prim != nullptr && both_prim->sram_region == "region_7" &&
                     both_prim->sram_offset == 3 &&
                     both_prim->destination_sram_region == "region_8" &&
                     both_prim->destination_sram_offset == 5,
                 "SPM_TO_SPM preserves both named regions");

    DteIssueOperands named_dram =
        BaseIssue(LocalDteDirection::SPM_TO_DRAM);
    named_dram.source_sram = Region(7, 3);
    LoweringContext missing;
    checks.Reject<RecordLoweringError>("missing symbol resolver",
                                      "resolver is required", [&] {
        LowerExternalRecord(IssueRecord(named_dram), missing);
    });
    named_dram.source_sram = Region(0xffffffffU, 3);
    checks.Reject<RecordLoweringError>("unknown symbol",
                                      "unknown index", [&] {
        LowerExternalRecord(IssueRecord(named_dram), context);
    });
}

void CheckLsuRegion(Checks &checks) {
    ExternalRecord record;
    record.opcode = Opcode::LSU_STORE;
    LsuOperands operands;
    operands.hbm_address_bytes = 0xfedcba9876543210ULL;
    operands.size_bytes = 64;
    operands.sram = Region(9, 0x123456789ULL);
    record.operands = operands;
    LoweredPrimList lowered = LowerExternalRecord(record, Context());
    auto *prim = dynamic_cast<Lsu_mem_prim *>(lowered.front().get());
    checks.Check(prim != nullptr && !prim->absolute_sram &&
                     prim->sram_region == "region_9" &&
                     prim->sram_offset == 0x123456789ULL &&
                     prim->hbm_addr == 0xfedcba9876543210ULL,
                 "LSU region resolver and full addresses");
}

void CheckValidationAndOwnership(Checks &checks) {
    const std::size_t stash_before = g_prim_stash.size();
    const std::size_t labels_before = g_addr_label_table.table.size();
    ExternalRecord bad = MakeRecord(*LookupRecordSchema(Opcode::MATMUL));
    std::get<ComputeOperands>(bad.operands).parameters[0] =
        kExternalNpuParameterMax + 1;
    checks.Reject<RecordCodecError>("validation before construction",
                                    "30 bits", [&] {
        LowerExternalRecord(bad, Context());
    });

    ExternalRecord bind = MakeRecord(*LookupRecordSchema(Opcode::SRAM_BIND));
    LoweringContext no_symbols;
    checks.Reject<RecordLoweringError>(
        "SRAM_BIND requires symbol resolver", "resolver is required", [&] {
            LowerExternalRecord(bind, no_symbols);
        });
    LoweringContext empty_symbol = Context();
    empty_symbol.resolve_symbol = [](uint32_t)
        -> std::optional<std::string> { return std::string{}; };
    checks.Reject<RecordLoweringError>(
        "SRAM_BIND rejects empty resolved labels", "empty name", [&] {
            LowerExternalRecord(bind, empty_symbol);
        });

    DteIssueOperands named = BaseIssue(LocalDteDirection::SPM_TO_DRAM);
    named.source_sram = Region(0xffffffffU, 0);
    for (int i = 0; i < 32; ++i) {
        try {
            LowerExternalRecord(IssueRecord(named), Context());
        } catch (const RecordLoweringError &) {
        }
    }
    checks.Check(g_prim_stash.size() == stash_before,
                 "exception paths do not leak into g_prim_stash");
    checks.Check(g_addr_label_table.table.size() == labels_before,
                 "lowering never mutates the label table");

    {
        LoweredPrimList owned = LowerExternalRecord(
            MakeRecord(*LookupRecordSchema(Opcode::MATMUL)), Context());
        checks.Check(owned.size() == 1 && owned.front() != nullptr,
                     "caller exclusively owns lowered Prim");
    }
    checks.Check(g_prim_stash.size() == stash_before,
                 "destroying returned ownership leaves no global Prim");
}

} // namespace

RecordLoweringSelfTestResult CheckIsaV1RecordLowering() {
    Checks checks;
    CheckManifestMatrix(checks);
    CheckExactStage2Wire(checks);
    CheckDteEndpointLowering(checks);
    CheckWholeArtifactCollectiveLowering(checks);
    CheckDteDirectionsAndResolvers(checks);
    CheckLsuRegion(checks);
    CheckValidationAndOwnership(checks);
    return std::move(checks.result);
}

int RunIsaV1RecordLoweringSelfTest() {
    const RecordLoweringSelfTestResult result = CheckIsaV1RecordLowering();
    if (result.passed()) {
        std::cout << "ISA v1 record lowering selftest passed (" << result.checks
                  << " checks)\n";
        return 0;
    }
    std::cerr << "ISA v1 record lowering selftest failed ("
              << result.failures.size() << "/" << result.checks
              << " checks failed)\n";
    for (const std::string &failure : result.failures)
        std::cerr << "  - " << failure << '\n';
    return 1;
}
