#include "prims/exact_stage2_prims.h"

#include "isa/published_npu_ops.h"
#include "utils/prim_utils.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

REGISTER_PRIM(Rope_qk_exact_prim, PrimId::ROPE_QK_EXACT);
REGISTER_PRIM(Attention_exact_prim, PrimId::ATTENTION_EXACT);
REGISTER_PRIM(Embedding_lookup_prim, PrimId::EMBEDDING_LOOKUP);
REGISTER_PRIM(Greedy_sample_prim, PrimId::GREEDY_SAMPLE);
REGISTER_PRIM(Cross_entropy_forward_prim, PrimId::CROSS_ENTROPY_FORWARD);
REGISTER_PRIM(Cross_entropy_backward_prim, PrimId::CROSS_ENTROPY_BACKWARD);
REGISTER_PRIM(Sgd_update_prim, PrimId::SGD_UPDATE);

namespace {

using Wire = vector<sc_bv<128>>;

constexpr size_t kFirstSegmentPayloadBytes = 10;
constexpr size_t kLaterSegmentPayloadBytes = 14;

void Require(bool condition, const std::string &message) {
    if (!condition) throw std::invalid_argument(message);
}

void RequireStrictTransport(const std::string &name) {
    if (prim_wire::LegacyCompatibilityEnabled())
        throw std::invalid_argument(
            name + " is strict-only and rejects legacy transport");
}

uint8_t ExpectedId(const Exact_stage2_prim_base &prim) {
    const int raw = PrimFactory::getInstance().getPrimId(prim.name);
    const OpcodeManifestEntry *opcode = LookupOpcode(prim.exact_opcode());
    if (raw <= 0 || raw > UINT8_MAX || opcode == nullptr ||
        opcode->lowering.kind != OpcodeLoweringKind::DIRECT_PRIM ||
        PrimIdValue(opcode->lowering.target) != raw)
        throw std::logic_error(
            prim.name + " factory/PrimId/opcode lowering identity mismatch");
    return static_cast<uint8_t>(raw);
}

size_t SegmentCount(size_t bytes) {
    Require(bytes >= kFirstSegmentPayloadBytes,
            "exact Stage2 Prim external record is too small");
    const size_t remaining = bytes - kFirstSegmentPayloadBytes;
    return 1 + (remaining + kLaterSegmentPayloadBytes - 1) /
                   kLaterSegmentPayloadBytes;
}

void SetByte(sc_bv<128> &segment, size_t byte_offset, uint8_t value) {
    const int low = static_cast<int>(byte_offset * 8);
    segment.range(low + 7, low) = sc_bv<8>(value);
}

uint8_t GetByte(const sc_bv<128> &segment, size_t byte_offset) {
    const int low = static_cast<int>(byte_offset * 8);
    return static_cast<uint8_t>(segment.range(low + 7, low).to_uint64());
}

PublishedNpuWork ExactWork(const ExternalRecord &record) {
    switch (record.opcode) {
    case Opcode::ROPE_QK_EXACT:
        return EvaluatePublishedNpuWork(
            std::get<RopeQkExactOperands>(record.operands));
    case Opcode::ATTENTION_EXACT:
        return EvaluatePublishedNpuWork(
            std::get<AttentionExactOperands>(record.operands));
    case Opcode::EMBEDDING_LOOKUP:
        return EvaluatePublishedNpuWork(
            std::get<EmbeddingLookupOperands>(record.operands));
    case Opcode::GREEDY_SAMPLE:
        return EvaluatePublishedNpuWork(
            std::get<GreedySampleOperands>(record.operands));
    case Opcode::CROSS_ENTROPY_FORWARD:
        return EvaluatePublishedNpuWork(
            std::get<CrossEntropyForwardOperands>(record.operands));
    case Opcode::CROSS_ENTROPY_BACKWARD:
        return EvaluatePublishedNpuWork(
            std::get<CrossEntropyBackwardOperands>(record.operands));
    case Opcode::SGD_UPDATE:
        return EvaluatePublishedNpuWork(
            std::get<SgdUpdateOperands>(record.operands));
    default:
        throw std::logic_error("exact Stage2 Prim has an unexpected opcode");
    }
}

} // namespace

Exact_stage2_prim_base::Exact_stage2_prim_base(
    const char *factory_name, Opcode opcode)
    : exact_opcode_(opcode) {
    name = factory_name;
    datatype = FP16;
    skip_input = true;
    skip_output = true;
}

void Exact_stage2_prim_base::initialize() {
    data_size_input = (exact_opcode_ == Opcode::EMBEDDING_LOOKUP ||
                       exact_opcode_ == Opcode::CROSS_ENTROPY_FORWARD)
                          ? vector<int>{1, 1}
                      : exact_opcode_ == Opcode::CROSS_ENTROPY_BACKWARD
                          ? vector<int>{1, 1, 1}
                      : exact_opcode_ == Opcode::SGD_UPDATE
                          ? vector<int>{1, 1}
                          : vector<int>{1};
    data_chunk = {{"output", 1}};
}

void Exact_stage2_prim_base::taskCore(
    TaskCoreContext &, string, u_int64_t &, u_int64_t &exu_ops,
    u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    const PublishedNpuWork work = ExactWork(ExactRecord());
    exu_ops = work.ops.exu;
    sfu_ops = work.ops.sfu;
    vec_ops = work.ops.vec;
}

Wire Exact_stage2_prim_base::serialize() {
    RequireStrictTransport(name);
    const ExternalRecord record = ExactRecord();
    Require(record.opcode == exact_opcode_,
            name + " external record opcode mismatch");
    const std::vector<uint8_t> bytes = EncodeExternalRecord(record);
    const size_t count = SegmentCount(bytes.size());
    if (count > UINT8_MAX || bytes.size() > UINT16_MAX)
        throw std::overflow_error(name + " internal wire size overflow");

    Wire wire(count);
    const uint8_t id = ExpectedId(*this);
    size_t source = 0;
    for (size_t ordinal = 0; ordinal < wire.size(); ++ordinal) {
        wire[ordinal] = 0;
        SetByte(wire[ordinal], 0, id);
        SetByte(wire[ordinal], 1, static_cast<uint8_t>(ordinal));
        size_t destination = 2;
        size_t capacity = kLaterSegmentPayloadBytes;
        if (ordinal == 0) {
            SetByte(wire[ordinal], 2, kExactStage2PrimWireVersion);
            SetByte(wire[ordinal], 3, static_cast<uint8_t>(count));
            wire[ordinal].range(47, 32) =
                sc_bv<16>(static_cast<uint16_t>(bytes.size()));
            destination = 6;
            capacity = kFirstSegmentPayloadBytes;
        }
        for (size_t slot = 0; slot < capacity && source < bytes.size();
             ++slot, ++source)
            SetByte(wire[ordinal], destination + slot, bytes[source]);
    }
    Require(source == bytes.size(), name + " internal wire packing failed");
    return wire;
}

void Exact_stage2_prim_base::deserialize(Wire wire) {
    RequireStrictTransport(name);
    Require(!wire.empty(), name + " internal wire has no segments");
    const uint8_t id = ExpectedId(*this);
    Require(GetByte(wire[0], 2) == kExactStage2PrimWireVersion,
            name + " internal wire version is unsupported");
    Require(GetByte(wire[0], 3) == wire.size(),
            name + " internal wire segment count field is inconsistent");
    const size_t byte_count = wire[0].range(47, 32).to_uint64();
    Require(SegmentCount(byte_count) == wire.size(),
            name + " internal wire byte count is inconsistent");

    std::vector<uint8_t> bytes;
    bytes.reserve(byte_count);
    for (size_t ordinal = 0; ordinal < wire.size(); ++ordinal) {
        Require(GetByte(wire[ordinal], 0) == id,
                name + " internal wire has inconsistent PrimIds");
        Require(GetByte(wire[ordinal], 1) == ordinal,
                name + " internal wire has inconsistent ordinals");
        const size_t source = ordinal == 0 ? 6 : 2;
        const size_t capacity = ordinal == 0 ? kFirstSegmentPayloadBytes
                                             : kLaterSegmentPayloadBytes;
        for (size_t slot = 0; slot < capacity; ++slot) {
            const uint8_t value = GetByte(wire[ordinal], source + slot);
            if (bytes.size() < byte_count)
                bytes.push_back(value);
            else
                Require(value == 0,
                        name + " internal wire unused bytes are non-zero");
        }
    }
    const ExternalRecord record = DecodeExternalRecordExact(bytes);
    Require(record.opcode == exact_opcode_,
            name + " internal wire opcode mismatch");
    AssignExactRecord(record);
    initialize();
}

void Exact_stage2_prim_base::printSelf() {}

Rope_qk_exact_prim::Rope_qk_exact_prim()
    : Exact_stage2_prim_base("Rope_qk_exact_prim",
                             Opcode::ROPE_QK_EXACT) {}

ExternalRecord Rope_qk_exact_prim::ExactRecord() const {
    return {Opcode::ROPE_QK_EXACT, operands};
}

void Rope_qk_exact_prim::AssignExactRecord(const ExternalRecord &record) {
    operands = std::get<RopeQkExactOperands>(record.operands);
}

Attention_exact_prim::Attention_exact_prim()
    : Exact_stage2_prim_base("Attention_exact_prim",
                             Opcode::ATTENTION_EXACT) {}

ExternalRecord Attention_exact_prim::ExactRecord() const {
    return {Opcode::ATTENTION_EXACT, operands};
}

void Attention_exact_prim::AssignExactRecord(const ExternalRecord &record) {
    operands = std::get<AttentionExactOperands>(record.operands);
}

Embedding_lookup_prim::Embedding_lookup_prim()
    : Exact_stage2_prim_base("Embedding_lookup_prim",
                             Opcode::EMBEDDING_LOOKUP) {}

ExternalRecord Embedding_lookup_prim::ExactRecord() const {
    return {Opcode::EMBEDDING_LOOKUP, operands};
}

void Embedding_lookup_prim::AssignExactRecord(const ExternalRecord &record) {
    operands = std::get<EmbeddingLookupOperands>(record.operands);
}

Greedy_sample_prim::Greedy_sample_prim()
    : Exact_stage2_prim_base("Greedy_sample_prim",
                             Opcode::GREEDY_SAMPLE) {}

ExternalRecord Greedy_sample_prim::ExactRecord() const {
    return {Opcode::GREEDY_SAMPLE, operands};
}

void Greedy_sample_prim::AssignExactRecord(const ExternalRecord &record) {
    operands = std::get<GreedySampleOperands>(record.operands);
}

Cross_entropy_forward_prim::Cross_entropy_forward_prim()
    : Exact_stage2_prim_base("Cross_entropy_forward_prim",
                             Opcode::CROSS_ENTROPY_FORWARD) {}

void Cross_entropy_forward_prim::taskCore(
    TaskCoreContext &context, string prim_name, u_int64_t &dram_time,
    u_int64_t &exu_ops, u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    Exact_stage2_prim_base::taskCore(context, std::move(prim_name), dram_time,
                                    exu_ops, sfu_ops, vec_ops);
    const PublishedNpuWork work = EvaluatePublishedNpuWork(operands);
    // Labels and the unreduced loss are both one 32-bit scalar per rank row.
    // Reuse the validated published write quotient so this observation cannot
    // drift from the executable CE work contract.
    const uint64_t label_read_bytes = work.memory_write_bytes;
    std::cout << "[TRAIN_CE] core=" << context.cid
              << " invocations=1 rank_rows=" << operands.rank_rows
              << " label_read_bytes=" << label_read_bytes
              << " loss_write_bytes=" << work.memory_write_bytes << '\n';
}

ExternalRecord Cross_entropy_forward_prim::ExactRecord() const {
    return {Opcode::CROSS_ENTROPY_FORWARD, operands};
}

void Cross_entropy_forward_prim::AssignExactRecord(
    const ExternalRecord &record) {
    operands = std::get<CrossEntropyForwardOperands>(record.operands);
}

Cross_entropy_backward_prim::Cross_entropy_backward_prim()
    : Exact_stage2_prim_base("Cross_entropy_backward_prim",
                             Opcode::CROSS_ENTROPY_BACKWARD) {}

void Cross_entropy_backward_prim::taskCore(
    TaskCoreContext &context, string prim_name, u_int64_t &dram_time,
    u_int64_t &exu_ops, u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    Exact_stage2_prim_base::taskCore(context, std::move(prim_name), dram_time,
                                    exu_ops, sfu_ops, vec_ops);
    const PublishedNpuWork work = EvaluatePublishedNpuWork(operands);
    const uint64_t logits_bytes =
        2ULL * operands.rank_rows * operands.vocab_size;
    const uint64_t labels_bytes = 4ULL * operands.rank_rows;
    const uint64_t upstream_bytes = 4ULL * operands.upstream_elements;
    std::cout << "[TRAIN_CE_BACKWARD] core=" << context.cid
              << " invocations=1 rank_rows=" << operands.rank_rows
              << " upstream_elements=" << operands.upstream_elements
              << " logits_read_bytes=" << logits_bytes
              << " label_read_bytes=" << labels_bytes
              << " upstream_read_bytes=" << upstream_bytes
              << " logits_grad_write_bytes=" << work.memory_write_bytes
              << '\n';
}

ExternalRecord Cross_entropy_backward_prim::ExactRecord() const {
    return {Opcode::CROSS_ENTROPY_BACKWARD, operands};
}

void Cross_entropy_backward_prim::AssignExactRecord(
    const ExternalRecord &record) {
    operands = std::get<CrossEntropyBackwardOperands>(record.operands);
}

Sgd_update_prim::Sgd_update_prim()
    : Exact_stage2_prim_base("Sgd_update_prim", Opcode::SGD_UPDATE) {}

void Sgd_update_prim::taskCore(
    TaskCoreContext &context, string prim_name, u_int64_t &dram_time,
    u_int64_t &exu_ops, u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    Exact_stage2_prim_base::taskCore(context, std::move(prim_name), dram_time,
                                    exu_ops, sfu_ops, vec_ops);
    const PublishedNpuWork work = EvaluatePublishedNpuWork(operands);
    std::cout << "[TRAIN_SGD] core=" << context.cid
              << " invocations=1 element_count=" << operands.element_count
              << " learning_rate_f64_bits="
              << operands.learning_rate_f64_bits
              << " sram_read_bytes=" << work.memory_read_bytes
              << " sram_write_bytes=" << work.memory_write_bytes << '\n';
}

ExternalRecord Sgd_update_prim::ExactRecord() const {
    return {Opcode::SGD_UPDATE, operands};
}

void Sgd_update_prim::AssignExactRecord(const ExternalRecord &record) {
    operands = std::get<SgdUpdateOperands>(record.operands);
}
