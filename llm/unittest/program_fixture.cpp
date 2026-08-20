#include "isa/program_format.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

enum class FixtureMode {
    BOUND_DUMMY,
    MISSING_BIND,
    WRONG_ARITY,
    P3_DIFF_CHAIN,
    P4_LIFECYCLE,
    P4_LIFECYCLE_DANGLING,
    P4_SYNC_EVENT,
    P4_SYNC_BAD_ENDPOINT,
    P8_DOUBLE_BUFFER,
    P8_PROGRAM_B,
    FRONTEND_N0_TP2_RS,
};

ExternalRecord DummyRecord() {
    ComputeOperands operands;
    operands.datatype = ExternalDataType::INT8;
    operands.input_offset_bytes = 0;
    operands.data_offset_bytes = 80;
    operands.output_offset_bytes = 80;

    ExternalRecord record;
    record.opcode = Opcode::DUMMY;
    record.operands = std::move(operands);
    return record;
}

ExternalRecord BindRecord(std::size_t input_count,
                          std::size_t output_symbol_index) {
    SramBindOperands operands;
    operands.input_count = input_count;
    for (std::size_t i = 0; i < input_count; ++i)
        operands.input_symbol_indices[i] = i;
    operands.output_symbol_index = output_symbol_index;

    ExternalRecord record;
    record.opcode = Opcode::SRAM_BIND;
    record.operands = std::move(operands);
    return record;
}

ExternalRecord ChainBindRecord(std::size_t input_symbol_index,
                               std::size_t output_symbol_index) {
    SramBindOperands operands;
    operands.input_count = 1;
    operands.input_symbol_indices[0] = input_symbol_index;
    operands.output_symbol_index = output_symbol_index;
    return {Opcode::SRAM_BIND, std::move(operands)};
}

ExternalRecord ChainComputeRecord(Opcode opcode,
                                  std::vector<uint64_t> parameters,
                                  uint64_t input_offset_bytes,
                                  uint64_t data_offset_bytes,
                                  uint64_t output_offset_bytes) {
    ComputeOperands operands;
    operands.datatype = ExternalDataType::INT8;
    operands.input_offset_bytes = input_offset_bytes;
    operands.data_offset_bytes = data_offset_bytes;
    operands.output_offset_bytes = output_offset_bytes;
    operands.parameters = std::move(parameters);
    return {opcode, std::move(operands)};
}

ProgramArtifact P3DiffChainArtifact() {
    ProgramArtifact artifact;
    artifact.strings = {
        "dram_label p3diff_input",
        "p3diff_matmul_out",
        "p3diff_attention_out",
        "p3diff_layernorm_out",
        "p3diff_result",
    };
    for (std::size_t index = 0; index < artifact.strings.size(); ++index)
        artifact.symbols.push_back(
            {index, ProgramSymbolKind::SRAM_LABEL, 0, 0, 0});

    std::vector<ExternalRecord> records;
    records.push_back(ChainBindRecord(0, 1));
    records.push_back(ChainComputeRecord(
        Opcode::MATMUL, {1, 4, 64, 64}, 0, 4096, 8192));
    records.push_back(ChainBindRecord(1, 2));
    records.push_back(ChainComputeRecord(
        Opcode::ATTENTION, {1, 4, 64, 4, 3}, 8192, 9216, 10240));
    records.push_back(ChainBindRecord(2, 3));
    records.push_back(ChainComputeRecord(
        Opcode::LAYERNORM, {1, 4, 64}, 10240, 12288, 13312));
    records.push_back(ChainBindRecord(3, 4));
    records.push_back(ChainComputeRecord(
        Opcode::GELU, {256}, 13312, 14336, 15360));

    artifact.cores = {{0, std::move(records)}};
    artifact.envelope.active_cores = {0};
    artifact.envelope.start_events = {{0, 0x31, 1}};
    artifact.envelope.terminal_cores = {0};
    artifact.envelope.expected_ack_cores = {0};
    artifact.envelope.expected_done_cores = {0};
    artifact.envelope.empty_core_ack_policy =
        EmptyCoreAckPolicy::INCLUDE_EMPTY;
    return artifact;
}

constexpr uint64_t kP8PayloadBytes = 1024;
constexpr uint64_t kP8PayloadOffset = 64;
constexpr uint64_t kP8RegionBytes = 2048;
constexpr uint64_t kP8InputRegionBytes = 4096;
constexpr uint64_t kP8DoubleABase = kP8InputRegionBytes;
constexpr uint64_t kP8DoubleBBase = kP8DoubleABase + kP8RegionBytes;
constexpr uint64_t kP8Batch0Hbm = 0x10000;
constexpr uint64_t kP8Batch1Hbm = 0x20000;
constexpr uint64_t kP8Batch2Hbm = 0x30000;
constexpr uint64_t kP8OutputHbm = 0x40000;
constexpr uint64_t kP8Token1 = 81;
constexpr uint64_t kP8Token2 = 82;

SramAddressOperand P8RegionAddress(uint64_t symbol_index) {
    SramAddressOperand address;
    address.kind = SramAddressKind::REGION;
    address.region_symbol_index = symbol_index;
    address.region_offset_bytes = kP8PayloadOffset;
    return address;
}

ExternalRecord P8LsuRecord(Opcode opcode, uint64_t hbm_base,
                           uint64_t region_symbol) {
    LsuOperands operands;
    operands.hbm_address_bytes = hbm_base + kP8PayloadOffset;
    operands.size_bytes = kP8PayloadBytes;
    operands.sram = P8RegionAddress(region_symbol);
    return {opcode, std::move(operands)};
}

ExternalRecord P8DteLoadRecord(uint64_t token, uint64_t hbm_base,
                               uint64_t region_symbol) {
    DteIssueOperands operands;
    operands.direction = LocalDteDirection::DRAM_TO_SPM;
    operands.token = token;
    operands.payload_bits = kP8PayloadBytes * 8;
    operands.size_bytes = kP8PayloadBytes;
    operands.hbm_address_bytes = hbm_base + kP8PayloadOffset;
    operands.destination_sram = P8RegionAddress(region_symbol);
    return {Opcode::DTE_ISSUE, std::move(operands)};
}

ExternalRecord P8BindRecord(uint64_t label_symbol) {
    SramBindOperands operands;
    operands.input_count = 1;
    operands.input_symbol_indices[0] = label_symbol;
    operands.output_symbol_index = label_symbol;
    return {Opcode::SRAM_BIND, std::move(operands)};
}

ExternalRecord P8MatmulRecord() {
    return ChainComputeRecord(Opcode::MATMUL, {1, 4096, 1, 1},
                              kP8PayloadOffset, kP8PayloadOffset,
                              kP8PayloadOffset);
}

ProgramArtifact P8DoubleBufferArtifact() {
    ProgramArtifact artifact;
    artifact.strings = {"double_a", "double_b",
                        "double_a_label", "double_b_label"};
    artifact.symbols = {
        {0, ProgramSymbolKind::SRAM_REGION, 0, kP8DoubleABase,
         kP8RegionBytes},
        {1, ProgramSymbolKind::SRAM_REGION, 0, kP8DoubleBBase,
         kP8RegionBytes},
        {2, ProgramSymbolKind::SRAM_LABEL, 0, 0, 0},
        {3, ProgramSymbolKind::SRAM_LABEL, 0, 0, 0},
    };
    artifact.cores = {{
        0,
        {
            P8LsuRecord(Opcode::LSU_LOAD, kP8Batch0Hbm, 0),
            P8DteLoadRecord(kP8Token1, kP8Batch1Hbm, 1),
            P8BindRecord(2),
            P8MatmulRecord(),
            ExternalRecord{Opcode::DTE_WAIT, TokenOperands{kP8Token1}},
            P8DteLoadRecord(kP8Token2, kP8Batch2Hbm, 0),
            P8BindRecord(3),
            P8MatmulRecord(),
            ExternalRecord{Opcode::DTE_WAIT, TokenOperands{kP8Token2}},
            P8LsuRecord(Opcode::LSU_STORE, kP8OutputHbm, 0),
            ExternalRecord{Opcode::DTE_FENCE, NoOperands{}},
        },
    }};
    artifact.envelope.active_cores = {0};
    artifact.envelope.start_events = {{0, 0x81, 1}};
    artifact.envelope.terminal_cores = {0};
    artifact.envelope.expected_ack_cores = {0};
    artifact.envelope.expected_done_cores = {0};
    artifact.envelope.empty_core_ack_policy =
        EmptyCoreAckPolicy::INCLUDE_EMPTY;
    return artifact;
}

void PrintP8DoubleBufferManifest() {
    std::cout << "P8_DOUBLE_BUFFER scenario=blocking_lsu_dte_double_buffer"
              << " core=0 payload_bytes=" << kP8PayloadBytes
              << " payload_offset=" << kP8PayloadOffset
              << " region_bytes=" << kP8RegionBytes
              << " batch0_hbm=" << kP8Batch0Hbm
              << " batch1_hbm=" << kP8Batch1Hbm
              << " batch2_hbm=" << kP8Batch2Hbm
              << " output_hbm=" << kP8OutputHbm
              << " token1=" << kP8Token1
              << " token2=" << kP8Token2
              << " lsu_load_count=1 lsu_store_count=1"
              << " dte_issue_count=2 dte_wait_count=2"
              << " bind_count=2 matmul_count=2 fence_count=1\n";
}


const std::string &ArtifactSymbolName(const ProgramArtifact &artifact,
                                      uint64_t symbol_index) {
    if (symbol_index >= artifact.symbols.size())
        throw std::logic_error("P3 diff symbol index is out of range");
    const ProgramSymbol &symbol = artifact.symbols[symbol_index];
    if (symbol.name_string_index >= artifact.strings.size())
        throw std::logic_error("P3 diff symbol string index is out of range");
    return artifact.strings[symbol.name_string_index];
}

std::string ManifestLabel(const std::string &label) {
    std::string result;
    for (const char value : label) {
        if (value == ' ')
            result += "%20";
        else if (value == '%')
            result += "%25";
        else
            result += value;
    }
    return result;
}

void PrintP3DiffManifest(const ProgramArtifact &artifact) {
    std::size_t compute_index = 0;
    for (const ExternalRecord &record : artifact.cores.at(0).records) {
        if (record.opcode == Opcode::SRAM_BIND) {
            const auto &bind = std::get<SramBindOperands>(record.operands);
            if (bind.input_count != 1)
                throw std::logic_error("P3 diff bind arity changed");
            std::cout << "P3_DIFF_BIND index=" << compute_index
                      << " input="
                      << ManifestLabel(ArtifactSymbolName(
                             artifact, bind.input_symbol_indices[0]))
                      << " output="
                      << ManifestLabel(ArtifactSymbolName(
                             artifact, bind.output_symbol_index))
                      << "\n";
            continue;
        }

        const RecordSchema *schema = LookupRecordSchema(record.opcode);
        const OpcodeManifestEntry *entry = LookupOpcode(record.opcode);
        if (schema == nullptr || entry == nullptr ||
            schema->operand_kind != RecordOperandKind::COMPUTE)
            throw std::logic_error("P3 diff record is not a public compute");
        const auto &compute = std::get<ComputeOperands>(record.operands);
        std::cout << "P3_DIFF_COMPUTE index=" << compute_index
                  << " opcode=" << entry->canonical_name
                  << " datatype=INT8 input_offset_bytes="
                  << compute.input_offset_bytes
                  << " data_offset_bytes=" << compute.data_offset_bytes
                  << " output_offset_bytes=" << compute.output_offset_bytes
                  << " params=";
        for (std::size_t index = 0; index < schema->parameter_count; ++index) {
            if (index != 0) std::cout << ",";
            std::cout << schema->parameter_names[index] << ":"
                      << compute.parameters.at(index);
        }
        std::cout << "\n";
        ++compute_index;
    }
}

ProgramArtifact DummyArtifact(FixtureMode mode) {
    const bool wrong_arity = mode == FixtureMode::WRONG_ARITY;

    ProgramArtifact artifact;
    artifact.strings = {"dram_label p3e_input"};
    if (wrong_arity)
        artifact.strings.push_back("dram_label p3e_input_1");
    artifact.strings.push_back("p3e_output");
    for (std::size_t i = 0; i < artifact.strings.size(); ++i)
        artifact.symbols.push_back(
            {i, ProgramSymbolKind::SRAM_LABEL, 0, 0, 0});

    std::vector<ExternalRecord> records;
    if (mode != FixtureMode::MISSING_BIND) {
        const std::size_t input_count = wrong_arity ? 2 : 1;
        records.push_back(
            BindRecord(input_count, artifact.symbols.size() - 1));
    }
    records.push_back(DummyRecord());
    artifact.cores = {{0, std::move(records)}};
    artifact.envelope.active_cores = {0};
    artifact.envelope.start_events = {{0, 1, 1}};
    artifact.envelope.terminal_cores = {0};
    artifact.envelope.expected_ack_cores = {0};
    artifact.envelope.expected_done_cores = {0};
    artifact.envelope.empty_core_ack_policy =
        EmptyCoreAckPolicy::INCLUDE_EMPTY;
    return artifact;
}

ExternalRecord LifecycleAllocRecord() {
    SramAllocOperands operands;
    operands.region_name_string_index = 0;
    operands.label_symbol_index = 1;
    operands.size_bytes = 64;
    operands.alignment_bytes = 64;
    operands.lifetime = SramLifetime::TASK;
    operands.spillable = true;
    return {Opcode::SRAM_ALLOC, std::move(operands)};
}

ExternalRecord LifecycleResizeRecord() {
    return {Opcode::SRAM_RESIZE, SramResizeOperands{1, 128}};
}

ExternalRecord LifecycleRenameRecord() {
    return {Opcode::SRAM_RENAME, SramRenameOperands{1, 2}};
}

ExternalRecord LifecycleClearRecord(bool dangling) {
    return {Opcode::SRAM_CLEAR, SymbolOperands{dangling ? 1u : 2u}};
}

SemanticRelocation LifecycleRelocation(
    uint64_t instruction, SemanticOperandId operand,
    SemanticRelocationKind kind, uint64_t symbol) {
    return {0, instruction, static_cast<uint16_t>(operand), kind, symbol, 0};
}

ProgramArtifact LifecycleArtifact(bool dangling) {
    ProgramArtifact artifact;
    artifact.strings = {"input", "p4_lifecycle_buffer",
                        "p4_lifecycle_buffer_renamed"};
    artifact.symbols = {
        {0, ProgramSymbolKind::SRAM_REGION, 0, 4096, 4096},
        {1, ProgramSymbolKind::SRAM_LABEL, 0, 0, 0},
        {2, ProgramSymbolKind::SRAM_LABEL, 0, 0, 0},
    };
    artifact.cores = {{0, {LifecycleAllocRecord(), LifecycleResizeRecord(),
                            LifecycleRenameRecord(),
                            LifecycleClearRecord(dangling)}}};
    artifact.relocations = {
        LifecycleRelocation(
            0, SemanticOperandId::REGION_NAME,
            SemanticRelocationKind::SRAM_REGION, 0),
        LifecycleRelocation(
            0, SemanticOperandId::LABEL_SYMBOL,
            SemanticRelocationKind::SRAM_LABEL, 1),
        LifecycleRelocation(
            1, SemanticOperandId::SYMBOL,
            SemanticRelocationKind::SRAM_LABEL, 1),
        LifecycleRelocation(
            2, SemanticOperandId::OLD_SYMBOL,
            SemanticRelocationKind::SRAM_LABEL, 1),
        LifecycleRelocation(
            2, SemanticOperandId::NEW_SYMBOL,
            SemanticRelocationKind::SRAM_LABEL, 2),
        LifecycleRelocation(
            3, SemanticOperandId::SYMBOL,
            SemanticRelocationKind::SRAM_LABEL, dangling ? 1 : 2),
    };
    artifact.envelope.active_cores = {0};
    artifact.envelope.start_events = {{0, 0x41, 1}};
    artifact.envelope.terminal_cores = {0};
    artifact.envelope.expected_ack_cores = {0};
    artifact.envelope.expected_done_cores = {0};
    artifact.envelope.empty_core_ack_policy =
        EmptyCoreAckPolicy::INCLUDE_EMPTY;
    return artifact;
}

ExternalRecord EventSetRecord(bool bad_endpoint) {
    EventSetOperands operands;
    operands.source_core = bad_endpoint ? 1 : 0;
    operands.destination_core = 1;
    operands.tag = 0x5045;
    return {Opcode::EVENT_SET, std::move(operands)};
}

ExternalRecord EventWaitRecord() {
    EventWaitOperands operands;
    operands.source_core = 0;
    operands.destination_core = 1;
    operands.tag = 0x5045;
    operands.count = 1;
    return {Opcode::EVENT_WAIT, std::move(operands)};
}

ExternalRecord GroupSyncRecord() {
    return {Opcode::GROUP_SYNC, GroupSyncOperands{41, 0}};
}

ProgramArtifact SyncEventArtifact(bool bad_endpoint) {
    ProgramArtifact artifact;
    artifact.core_groups = {{41, {0, 1}}};
    artifact.cores = {
        {0, {EventSetRecord(bad_endpoint), GroupSyncRecord()}},
        {1, {EventWaitRecord(), GroupSyncRecord()}},
    };
    artifact.envelope.active_cores = {0, 1};
    artifact.envelope.start_events = {{0, 0x51, 1}, {1, 0x52, 1}};
    artifact.envelope.terminal_cores = {0, 1};
    artifact.envelope.expected_ack_cores = {0, 1};
    artifact.envelope.expected_done_cores = {0, 1};
    artifact.envelope.empty_core_ack_policy =
        EmptyCoreAckPolicy::INCLUDE_EMPTY;
    return artifact;
}

struct EndpointFixtureSpec {
    std::string scenario;
    uint64_t source_core = 0;
    uint64_t destination_core = 1;
    EndpointSourceSpace source_space = EndpointSourceSpace::SRAM;
    EndpointCompletion completion = EndpointCompletion::SYNC;
    uint64_t fsm_id = 1;
    uint64_t send_token = 0;
    uint64_t recv_token = 0;
    uint64_t send_length_bytes = 129;
    uint64_t recv_length_bytes = 129;
    uint64_t source_offset_bytes = 0;
    uint64_t destination_offset_bytes = 0;
    uint64_t hbm_address_bytes = 0;
    uint64_t input_region_base_bytes = 4096;
    uint64_t input_region_size_bytes = 4096;
    uint64_t comm_region_base_bytes = 12288;
    uint64_t comm_region_size_bytes = 4096;
    bool include_recv = true;
};

bool TryP5EndpointSpec(const std::string &option,
                       EndpointFixtureSpec *result) {
    EndpointFixtureSpec spec;
    static constexpr std::array<uint64_t, 10> kSyncLengths = {
        1, 15, 16, 17, 127, 128, 129, 255, 256, 257};
    for (const uint64_t length : kSyncLengths) {
        if (option != "--p5-sram-sync-" + std::to_string(length))
            continue;
        spec.scenario = "same_die_sram_sync_" + std::to_string(length);
        spec.fsm_id = length == 129 ? 0x50000001
                                    : 0x50001000 + length;
        spec.send_length_bytes = length;
        spec.recv_length_bytes = length;
        spec.source_offset_bytes = 17;
        spec.destination_offset_bytes = 19;
        *result = std::move(spec);
        return true;
    }
    if (option == "--p5-sram-async-17") {
        spec.scenario = "same_die_sram_async_17";
        spec.completion = EndpointCompletion::ASYNC;
        spec.fsm_id = 0x50011001;
        spec.send_token = 0x51011001;
        spec.recv_token = 0x51011002;
        spec.send_length_bytes = 17;
        spec.recv_length_bytes = 17;
        spec.source_offset_bytes = 17;
        spec.destination_offset_bytes = 19;
    } else if (option == "--p5-sram-async-4k") {
        spec.scenario = "cross_die_sram_async_4k";
        spec.destination_core = 16;
        spec.completion = EndpointCompletion::ASYNC;
        spec.fsm_id = 0x50010001;
        spec.send_token = 0x51010001;
        spec.recv_token = 0x51010002;
        spec.send_length_bytes = 4096;
        spec.recv_length_bytes = 4096;
    } else if (option == "--p5-cross-die-sram-sync-128") {
        spec.scenario = "cross_die_sram_sync_128";
        spec.destination_core = 16;
        spec.fsm_id = 0x50012001;
        spec.send_length_bytes = 128;
        spec.recv_length_bytes = 128;
        spec.source_offset_bytes = 17;
        spec.destination_offset_bytes = 19;
    } else if (option == "--p5-sram-async-32k") {
        spec.scenario = "cross_die_sram_async_32k";
        spec.destination_core = 16;
        spec.completion = EndpointCompletion::ASYNC;
        spec.fsm_id = 0x50013001;
        spec.send_token = 0x51013001;
        spec.recv_token = 0x51013002;
        spec.send_length_bytes = 32768;
        spec.recv_length_bytes = 32768;
        spec.source_offset_bytes = 17;
        spec.destination_offset_bytes = 19;
        spec.input_region_size_bytes = 36864;
        spec.comm_region_base_bytes = 40960;
        spec.comm_region_size_bytes = 36864;
    } else if (option == "--p5-hbm-sync-129") {
        spec.scenario = "same_die_hbm_sync_129";
        spec.source_space = EndpointSourceSpace::HBM;
        spec.fsm_id = 0x50020001;
        spec.hbm_address_bytes = 128;
        spec.destination_offset_bytes = 257;
    } else if (option == "--p5-pair-missing") {
        spec.scenario = "pair_missing";
        spec.fsm_id = 0x50030001;
        spec.include_recv = false;
    } else if (option == "--p5-pair-mismatch") {
        spec.scenario = "pair_mismatch";
        spec.fsm_id = 0x50040001;
        spec.recv_length_bytes = 130;
    } else {
        return false;
    }
    *result = std::move(spec);
    return true;
}

ExternalRecord P5EndpointSendRecord(const EndpointFixtureSpec &spec) {
    DteSendOperands operands;
    operands.mode = DteSendMode::P2P;
    operands.source_space = spec.source_space;
    operands.completion = spec.completion;
    operands.datatype = EndpointDataType::UINT8;
    operands.reduce_op = ReduceOperator::NONE;
    operands.fsm_id = spec.fsm_id;
    operands.token = spec.send_token;
    operands.length_bytes = spec.send_length_bytes;
    if (spec.source_space == EndpointSourceSpace::HBM) {
        operands.source.kind = SramAddressKind::ABSOLUTE;
        operands.source.absolute_address_bytes = spec.hbm_address_bytes;
    } else {
        operands.source.kind = SramAddressKind::REGION;
        operands.source.region_symbol_index = 0;
        operands.source.region_offset_bytes = spec.source_offset_bytes;
    }
    operands.peer_core = spec.destination_core;
    operands.expected_sources = 0;
    operands.tree_id = 0;
    operands.group_id = 0;
    operands.collective_id = 0;
    operands.epoch = 0;
    return {Opcode::DTE_SEND, std::move(operands)};
}

ExternalRecord P5EndpointRecvRecord(const EndpointFixtureSpec &spec) {
    DteRecvOperands operands;
    operands.mode = DteRecvMode::P2P;
    operands.completion = spec.completion;
    operands.datatype = EndpointDataType::UINT8;
    operands.reduce_op = ReduceOperator::NONE;
    operands.fsm_id = spec.fsm_id;
    operands.token = spec.recv_token;
    operands.length_bytes = spec.recv_length_bytes;
    operands.destination.kind = SramAddressKind::REGION;
    operands.destination.region_symbol_index = 1;
    operands.destination.region_offset_bytes =
        spec.destination_offset_bytes;
    operands.peer_core = spec.source_core;
    operands.expected_sources = 0;
    operands.tree_id = 0;
    operands.group_id = 0;
    operands.collective_id = 0;
    operands.epoch = 0;
    return {Opcode::DTE_RECV, std::move(operands)};
}

ExternalRecord P5EndpointWaitRecord(uint64_t token) {
    return {Opcode::DTE_WAIT, TokenOperands{token}};
}

ProgramArtifact P5EndpointArtifact(const EndpointFixtureSpec &spec) {
    ProgramArtifact artifact;
    artifact.strings = {"input", "comm"};
    artifact.symbols = {
        {0, ProgramSymbolKind::SRAM_REGION, 0,
         spec.input_region_base_bytes, spec.input_region_size_bytes},
        {1, ProgramSymbolKind::SRAM_REGION, 0,
         spec.comm_region_base_bytes, spec.comm_region_size_bytes},
    };

    std::vector<ExternalRecord> source_records = {
        P5EndpointSendRecord(spec)};
    if (spec.completion == EndpointCompletion::ASYNC)
        source_records.push_back(P5EndpointWaitRecord(spec.send_token));
    std::vector<ExternalRecord> destination_records;
    if (spec.include_recv) {
        destination_records.push_back(P5EndpointRecvRecord(spec));
        if (spec.completion == EndpointCompletion::ASYNC)
            destination_records.push_back(
                P5EndpointWaitRecord(spec.recv_token));
    }

    artifact.cores = {
        {spec.source_core, std::move(source_records)},
        {spec.destination_core, std::move(destination_records)},
    };
    artifact.envelope.active_cores = {
        spec.source_core, spec.destination_core};
    artifact.envelope.start_events = {
        {spec.source_core, 0x61, 1},
        {spec.destination_core, 0x62, 1},
    };
    artifact.envelope.terminal_cores = {
        spec.source_core, spec.destination_core};
    artifact.envelope.expected_ack_cores = {
        spec.source_core, spec.destination_core};
    artifact.envelope.expected_done_cores = {
        spec.source_core, spec.destination_core};
    artifact.envelope.empty_core_ack_policy =
        EmptyCoreAckPolicy::INCLUDE_EMPTY;
    return artifact;
}

void PrintP5EndpointManifest(const EndpointFixtureSpec &spec) {
    std::cout << "P5_ENDPOINT scenario=" << spec.scenario
              << " source_core=" << spec.source_core
              << " destination_core=" << spec.destination_core
              << " source_space="
              << (spec.source_space == EndpointSourceSpace::HBM
                      ? "HBM" : "SRAM")
              << " completion="
              << (spec.completion == EndpointCompletion::SYNC
                      ? "SYNC" : "ASYNC")
              << " fsm_id=" << spec.fsm_id
              << " send_token=" << spec.send_token
              << " recv_token=" << spec.recv_token
              << " send_length_bytes=" << spec.send_length_bytes
              << " recv_length_bytes=" << spec.recv_length_bytes
              << " source="
              << (spec.source_space == EndpointSourceSpace::HBM
                      ? "hbm_absolute" : "input_region")
              << " source_offset_bytes="
              << (spec.source_space == EndpointSourceSpace::HBM
                      ? spec.hbm_address_bytes : spec.source_offset_bytes)
              << " destination=comm_region"
              << " destination_offset_bytes="
              << spec.destination_offset_bytes
              << " include_recv=" << (spec.include_recv ? 1 : 0)
              << "\n";
}
enum class P6Op {
    P2P,
    SCATTER,
    BROADCAST,
    GATHER,
    ALLTOALL,
    ALLGATHER,
    REDUCE,
    REDUCESCATTER,
    ALLREDUCE,
};

enum class P6Negative {
    NONE,
    GROUP,
    RANK,
    ROOT,
    CROSSDIE,
    COUNT,
    STRIDE,
    SCHEDULE,
};

struct P6CollectiveFixtureSpec {
    std::string scenario;
    DteSendMode tx = DteSendMode::P2P;
    DteRecvMode rx = DteRecvMode::P2P;
    EndpointDataType dtype = EndpointDataType::UINT8;
    ReduceOperator reduce_op = ReduceOperator::NONE;
    P6Negative negative = P6Negative::NONE;
    std::vector<uint64_t> cores;
    uint64_t length_bytes = 17;
    uint64_t group_id = 61;
    uint64_t collective_id = 1;
    uint64_t epoch = 1;
    uint64_t fsm_base = 0x60000000;
    uint64_t token_base = 0x61000000;
};

constexpr uint64_t kP6PayloadOffset = 16;
constexpr uint8_t kP6Sentinel = 0xa5;

const char *P6TxName(DteSendMode mode) {
    switch (mode) {
    case DteSendMode::P2P: return "unicast";
    case DteSendMode::SCATTER: return "scatter";
    case DteSendMode::BROADCAST: return "broadcast";
    }
    throw std::logic_error("unknown P6 transmit mode");
}

const char *P6RxName(DteRecvMode mode) {
    switch (mode) {
    case DteRecvMode::P2P: return "unicast";
    case DteRecvMode::GATHER: return "gather";
    case DteRecvMode::REDUCE: return "reduce";
    }
    throw std::logic_error("unknown P6 receive mode");
}

const char *P6DTypeName(EndpointDataType dtype) {
    switch (dtype) {
    case EndpointDataType::UINT8: return "UINT8";
    case EndpointDataType::INT32: return "INT32";
    case EndpointDataType::INT64: return "INT64";
    }
    throw std::logic_error("unknown P6 datatype");
}

const char *P6ReduceName(ReduceOperator op) {
    switch (op) {
    case ReduceOperator::NONE: return "NONE";
    case ReduceOperator::SUM: return "SUM";
    case ReduceOperator::MAX: return "MAX";
    }
    throw std::logic_error("unknown P6 reduction operator");
}

const char *P6NegativeName(P6Negative negative) {
    switch (negative) {
    case P6Negative::NONE: return "none";
    case P6Negative::GROUP: return "group";
    case P6Negative::RANK: return "rank";
    case P6Negative::ROOT: return "root";
    case P6Negative::CROSSDIE: return "crossdie";
    case P6Negative::COUNT: return "count";
    case P6Negative::STRIDE: return "stride";
    case P6Negative::SCHEDULE: return "schedule";
    }
    throw std::logic_error("unknown P6 negative kind");
}

P6Op P6Operation(DteSendMode tx, DteRecvMode rx) {
    if (tx == DteSendMode::P2P && rx == DteRecvMode::P2P)
        return P6Op::P2P;
    if (tx == DteSendMode::SCATTER && rx == DteRecvMode::P2P)
        return P6Op::SCATTER;
    if (tx == DteSendMode::BROADCAST && rx == DteRecvMode::P2P)
        return P6Op::BROADCAST;
    if (tx == DteSendMode::P2P && rx == DteRecvMode::GATHER)
        return P6Op::GATHER;
    if (tx == DteSendMode::SCATTER && rx == DteRecvMode::GATHER)
        return P6Op::ALLTOALL;
    if (tx == DteSendMode::BROADCAST && rx == DteRecvMode::GATHER)
        return P6Op::ALLGATHER;
    if (tx == DteSendMode::P2P && rx == DteRecvMode::REDUCE)
        return P6Op::REDUCE;
    if (tx == DteSendMode::SCATTER && rx == DteRecvMode::REDUCE)
        return P6Op::REDUCESCATTER;
    if (tx == DteSendMode::BROADCAST && rx == DteRecvMode::REDUCE)
        return P6Op::ALLREDUCE;
    throw std::logic_error("unknown P6 3x3 operation");
}

const char *P6OpName(P6Op op) {
    switch (op) {
    case P6Op::P2P: return "P2P";
    case P6Op::SCATTER: return "SCATTER";
    case P6Op::BROADCAST: return "BROADCAST";
    case P6Op::GATHER: return "GATHER";
    case P6Op::ALLTOALL: return "ALLTOALL";
    case P6Op::ALLGATHER: return "ALLGATHER";
    case P6Op::REDUCE: return "REDUCE";
    case P6Op::REDUCESCATTER: return "REDUCESCATTER";
    case P6Op::ALLREDUCE: return "ALLREDUCE";
    }
    throw std::logic_error("unknown P6 operation");
}

bool P6RootTransmit(P6Op op) {
    return op == P6Op::SCATTER || op == P6Op::BROADCAST;
}

bool P6RootReceive(P6Op op) {
    return op == P6Op::GATHER || op == P6Op::REDUCE;
}

bool P6Reduction(P6Op op) {
    return op == P6Op::REDUCE || op == P6Op::REDUCESCATTER ||
           op == P6Op::ALLREDUCE;
}

uint64_t P6DTypeBytes(EndpointDataType dtype) {
    switch (dtype) {
    case EndpointDataType::UINT8: return 1;
    case EndpointDataType::INT32: return 4;
    case EndpointDataType::INT64: return 8;
    }
    throw std::logic_error("unknown P6 datatype");
}

uint32_t Fnv1a32(const std::string &value) {
    uint32_t hash = 2166136261u;
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= 16777619u;
    }
    return hash;
}

std::vector<std::string> Split(const std::string &value, char separator) {
    std::vector<std::string> result;
    std::size_t begin = 0;
    while (true) {
        const std::size_t end = value.find(separator, begin);
        result.push_back(value.substr(begin, end - begin));
        if (end == std::string::npos) return result;
        begin = end + 1;
    }
}

uint64_t ParseP6Unsigned(const std::string &text, const char *field) {
    if (text.empty())
        throw std::invalid_argument(std::string("empty P6 ") + field);
    std::size_t consumed = 0;
    uint64_t value = 0;
    try {
        value = std::stoull(text, &consumed, 10);
    } catch (const std::exception &) {
        throw std::invalid_argument(std::string("invalid P6 ") + field);
    }
    if (consumed != text.size())
        throw std::invalid_argument(std::string("invalid P6 ") + field);
    return value;
}

P6CollectiveFixtureSpec MakeP6BaseSpec(
    DteSendMode tx, DteRecvMode rx, uint64_t n, uint64_t length,
    EndpointDataType dtype, ReduceOperator reduce_op,
    const std::string &scenario) {
    if (n != 1 && n != 2 && n != 4)
        throw std::invalid_argument("P6 group size must be 1, 2, or 4");
    if (length == 0 || length > 32768)
        throw std::invalid_argument("P6 fixture length must be 1..32768");
    const bool reduction = rx == DteRecvMode::REDUCE;
    if (reduction) {
        if (reduce_op != ReduceOperator::SUM &&
            reduce_op != ReduceOperator::MAX)
            throw std::invalid_argument(
                "P6 reduction requires SUM or MAX");
        if (length % P6DTypeBytes(dtype) != 0)
            throw std::invalid_argument(
                "P6 reduction length must be dtype aligned");
    } else if (dtype != EndpointDataType::UINT8 ||
               reduce_op != ReduceOperator::NONE) {
        throw std::invalid_argument(
            "P6 non-reduction must use UINT8/NONE");
    }

    static constexpr std::array<uint64_t, 4> kCores = {1, 3, 7, 11};
    P6CollectiveFixtureSpec spec;
    spec.scenario = scenario;
    spec.tx = tx;
    spec.rx = rx;
    spec.dtype = dtype;
    spec.reduce_op = reduce_op;
    spec.cores.assign(kCores.begin(), kCores.begin() + n);
    spec.length_bytes = length;
    const uint32_t hash = Fnv1a32(scenario);
    spec.collective_id = 1u + (hash & 0x00ffffffu);
    spec.fsm_base = 0x60000000u | (hash & 0x000ffff0u);
    spec.token_base = 0x61000000u | (hash & 0x000fff00u);
    return spec;
}

bool TryP6CollectiveSpec(const std::string &option,
                         P6CollectiveFixtureSpec *result) {
    const std::string prefix = "--p6-";
    if (option.compare(0, prefix.size(), prefix) != 0)
        return false;

    const std::string suffix = option.substr(prefix.size());
    if (suffix.compare(0, 4, "bad-") == 0) {
        const std::string kind = suffix.substr(4);
        P6CollectiveFixtureSpec spec;
        if (kind == "group") {
            spec = MakeP6BaseSpec(
                DteSendMode::SCATTER, DteRecvMode::P2P, 4, 17,
                EndpointDataType::UINT8, ReduceOperator::NONE,
                "p6_bad_group");
            spec.negative = P6Negative::GROUP;
        } else if (kind == "rank") {
            spec = MakeP6BaseSpec(
                DteSendMode::P2P, DteRecvMode::REDUCE, 4, 1024,
                EndpointDataType::INT32, ReduceOperator::SUM,
                "p6_bad_rank");
            spec.negative = P6Negative::RANK;
        } else if (kind == "root") {
            spec = MakeP6BaseSpec(
                DteSendMode::P2P, DteRecvMode::REDUCE, 4, 1024,
                EndpointDataType::INT32, ReduceOperator::SUM,
                "p6_bad_root");
            spec.negative = P6Negative::ROOT;
        } else if (kind == "crossdie") {
            spec = MakeP6BaseSpec(
                DteSendMode::SCATTER, DteRecvMode::P2P, 2, 17,
                EndpointDataType::UINT8, ReduceOperator::NONE,
                "p6_bad_crossdie");
            spec.cores = {1, 17};
            spec.negative = P6Negative::CROSSDIE;
        } else if (kind == "count") {
            spec = MakeP6BaseSpec(
                DteSendMode::P2P, DteRecvMode::GATHER, 4, 17,
                EndpointDataType::UINT8, ReduceOperator::NONE,
                "p6_bad_count");
            spec.negative = P6Negative::COUNT;
        } else if (kind == "stride") {
            spec = MakeP6BaseSpec(
                DteSendMode::BROADCAST, DteRecvMode::REDUCE, 4, 1024,
                EndpointDataType::INT32, ReduceOperator::MAX,
                "p6_bad_stride");
            spec.negative = P6Negative::STRIDE;
        } else if (kind == "schedule") {
            spec = MakeP6BaseSpec(
                DteSendMode::P2P, DteRecvMode::P2P, 2, 17,
                EndpointDataType::UINT8, ReduceOperator::NONE,
                "p6_bad_schedule");
            spec.negative = P6Negative::SCHEDULE;
        } else {
            return false;
        }
        *result = std::move(spec);
        return true;
    }

    const std::vector<std::string> parts = Split(suffix, '-');
    if (parts.size() != 6 || parts[2].size() < 2 ||
        parts[2][0] != 'n' || parts[3].size() < 2 ||
        parts[3][0] != 'l')
        return false;

    DteSendMode tx;
    if (parts[0] == "unicast")
        tx = DteSendMode::P2P;
    else if (parts[0] == "scatter")
        tx = DteSendMode::SCATTER;
    else if (parts[0] == "broadcast")
        tx = DteSendMode::BROADCAST;
    else
        return false;

    DteRecvMode rx;
    if (parts[1] == "unicast")
        rx = DteRecvMode::P2P;
    else if (parts[1] == "gather")
        rx = DteRecvMode::GATHER;
    else if (parts[1] == "reduce")
        rx = DteRecvMode::REDUCE;
    else
        return false;

    EndpointDataType dtype;
    if (parts[4] == "u8")
        dtype = EndpointDataType::UINT8;
    else if (parts[4] == "i32")
        dtype = EndpointDataType::INT32;
    else if (parts[4] == "i64")
        dtype = EndpointDataType::INT64;
    else
        return false;

    ReduceOperator reduce_op;
    if (parts[5] == "none")
        reduce_op = ReduceOperator::NONE;
    else if (parts[5] == "sum")
        reduce_op = ReduceOperator::SUM;
    else if (parts[5] == "max")
        reduce_op = ReduceOperator::MAX;
    else
        return false;

    const uint64_t n = ParseP6Unsigned(parts[2].substr(1), "group size");
    const uint64_t length =
        ParseP6Unsigned(parts[3].substr(1), "length");
    *result = MakeP6BaseSpec(
        tx, rx, n, length, dtype, reduce_op,
        "p6_" + suffix);
    return true;
}
uint64_t P6RootRank(P6Op op, uint64_t n) {
    if (P6RootTransmit(op) || P6RootReceive(op))
        return n == 1 ? 0 : 1;
    return 0;
}

bool P6WantSend(P6Op op, uint64_t rank, uint64_t root, uint64_t n) {
    if (op == P6Op::P2P) return rank == 0;
    if (P6RootTransmit(op)) return rank == root;
    (void)n;
    return true;
}

bool P6WantReceive(P6Op op, uint64_t rank, uint64_t root, uint64_t n) {
    if (op == P6Op::P2P) return rank == n - 1;
    if (P6RootReceive(op)) return rank == root;
    return true;
}

bool P6WantCompute(P6Op op, uint64_t rank, uint64_t root) {
    if (op == P6Op::REDUCE) return rank == root;
    return op == P6Op::REDUCESCATTER || op == P6Op::ALLREDUCE;
}

uint64_t P6Token(const P6CollectiveFixtureSpec &spec, uint64_t rank,
                 bool receive) {
    return spec.token_base + rank * 4 + (receive ? 2 : 1);
}

SramAddressOperand P6RegionAddress(uint64_t symbol, uint64_t offset) {
    SramAddressOperand address;
    address.kind = SramAddressKind::REGION;
    address.region_symbol_index = symbol;
    address.region_offset_bytes = offset;
    return address;
}

ExternalRecord P6SendRecord(const P6CollectiveFixtureSpec &spec,
                            uint64_t rank) {
    DteSendOperands operands;
    operands.mode = spec.tx;
    operands.source_space = EndpointSourceSpace::SRAM;
    operands.completion = EndpointCompletion::ASYNC;
    operands.datatype = EndpointDataType::UINT8;
    operands.reduce_op = ReduceOperator::NONE;
    operands.fsm_id = spec.fsm_base;
    operands.token = P6Token(spec, rank, false);
    operands.length_bytes = spec.length_bytes;
    operands.source = P6RegionAddress(0, kP6PayloadOffset);
    operands.peer_core = 0;
    operands.expected_sources = 0;
    operands.tree_id = 0;
    operands.group_id = spec.group_id;
    operands.collective_id = spec.collective_id;
    operands.epoch = spec.epoch;
    return {Opcode::DTE_SEND, std::move(operands)};
}

ExternalRecord P6ReceiveRecord(const P6CollectiveFixtureSpec &spec,
                               uint64_t rank) {
    DteRecvOperands operands;
    operands.mode = spec.rx;
    operands.completion = EndpointCompletion::ASYNC;
    operands.datatype =
        spec.rx == DteRecvMode::REDUCE ? spec.dtype
                                      : EndpointDataType::UINT8;
    operands.reduce_op =
        spec.rx == DteRecvMode::REDUCE ? spec.reduce_op
                                      : ReduceOperator::NONE;
    operands.fsm_id = spec.fsm_base;
    operands.token = P6Token(spec, rank, true);
    operands.length_bytes = spec.length_bytes;
    operands.destination = P6RegionAddress(1, kP6PayloadOffset);
    operands.peer_core = 0;
    operands.expected_sources =
        (spec.rx == DteRecvMode::GATHER ||
         spec.rx == DteRecvMode::REDUCE)
            ? spec.cores.size() - 1
            : 0;
    operands.tree_id = 0;
    operands.group_id = spec.group_id;
    operands.collective_id = spec.collective_id;
    operands.epoch = spec.epoch;
    if (spec.negative == P6Negative::COUNT)
        ++operands.expected_sources;
    return {Opcode::DTE_RECV, std::move(operands)};
}

ExternalRecord P6ComputeRecord(const P6CollectiveFixtureSpec &spec,
                               uint64_t rank, uint64_t root) {
    ReduceComputeOperands operands;
    operands.datatype = spec.dtype;
    operands.reduce_op = spec.reduce_op;
    operands.group_id = spec.group_id;
    operands.collective_id = spec.collective_id;
    operands.epoch = spec.epoch;
    operands.root_rank =
        P6Operation(spec.tx, spec.rx) == P6Op::REDUCE ? root : 0;
    operands.self_rank = rank;
    operands.element_count =
        spec.length_bytes / P6DTypeBytes(spec.dtype);
    operands.source = P6RegionAddress(1, kP6PayloadOffset);
    operands.destination = P6RegionAddress(2, kP6PayloadOffset);
    if (spec.negative == P6Negative::RANK)
        ++operands.self_rank;
    if (spec.negative == P6Negative::ROOT)
        operands.root_rank = operands.root_rank == 0 ? 1 : 0;
    if (spec.negative == P6Negative::STRIDE)
        ++operands.element_count;
    return {Opcode::REDUCE_COMPUTE, std::move(operands)};
}

uint64_t P6Align64(uint64_t value) {
    if (value > std::numeric_limits<uint64_t>::max() - 63)
        throw std::overflow_error("P6 SRAM region alignment overflows");
    return (value + 63) & ~uint64_t{63};
}

struct P6SramLayout {
    uint64_t input_base = 4096;
    uint64_t input_size = 0;
    uint64_t staging_base = 0;
    uint64_t staging_size = 0;
    uint64_t result_base = 0;
    uint64_t result_size = 0;
};

P6SramLayout P6Layout(const P6CollectiveFixtureSpec &spec) {
    const uint64_t n = spec.cores.size();
    const uint64_t source_bytes =
        spec.tx == DteSendMode::SCATTER ? n * spec.length_bytes
                                        : spec.length_bytes;
    const uint64_t staging_bytes =
        (spec.rx == DteRecvMode::GATHER ||
         spec.rx == DteRecvMode::REDUCE)
            ? n * spec.length_bytes
            : spec.length_bytes;
    P6SramLayout layout;
    layout.input_size = P6Align64(source_bytes + 2 * kP6PayloadOffset);
    layout.staging_base =
        P6Align64(layout.input_base + layout.input_size + 4096);
    layout.staging_size =
        P6Align64(staging_bytes + 2 * kP6PayloadOffset);
    layout.result_base =
        P6Align64(layout.staging_base + layout.staging_size + 4096);
    layout.result_size =
        P6Align64(spec.length_bytes + 2 * kP6PayloadOffset);
    return layout;
}

void AppendP6PlanRecords(ProgramArtifact &artifact,
                         const P6CollectiveFixtureSpec &spec) {
    const P6Op op = P6Operation(spec.tx, spec.rx);
    const uint64_t n = spec.cores.size();
    const uint64_t root = P6RootRank(op, n);
    for (uint64_t rank = 0; rank < n; ++rank) {
        auto found = std::find_if(
            artifact.cores.begin(), artifact.cores.end(),
            [&](const ProgramCore &core) {
                return core.core_id == spec.cores[rank];
            });
        if (found == artifact.cores.end())
            throw std::logic_error("P6 artifact core is missing");
        std::vector<uint64_t> waits;
        if (P6WantSend(op, rank, root, n)) {
            found->records.push_back(P6SendRecord(spec, rank));
            waits.push_back(P6Token(spec, rank, false));
        }
        if (P6WantReceive(op, rank, root, n)) {
            found->records.push_back(P6ReceiveRecord(spec, rank));
            waits.push_back(P6Token(spec, rank, true));
        }
        if (P6WantCompute(op, rank, root))
            found->records.push_back(P6ComputeRecord(spec, rank, root));
        for (uint64_t token : waits)
            found->records.push_back(
                {Opcode::DTE_WAIT, TokenOperands{token}});
        found->records.push_back(
            {Opcode::DTE_FENCE, NoOperands{}});
    }
}

ProgramArtifact P6CollectiveArtifact(
    const P6CollectiveFixtureSpec &requested) {
    P6CollectiveFixtureSpec spec = requested;
    const P6SramLayout layout = P6Layout(spec);

    ProgramArtifact artifact;
    artifact.strings = {"p6_input", "p6_staging", "p6_result"};
    artifact.symbols = {
        {0, ProgramSymbolKind::SRAM_REGION, 0,
         layout.input_base, layout.input_size},
        {1, ProgramSymbolKind::SRAM_REGION, 0,
         layout.staging_base, layout.staging_size},
        {2, ProgramSymbolKind::SRAM_REGION, 0,
         layout.result_base, layout.result_size},
    };
    artifact.core_groups = {{spec.group_id, spec.cores}};
    for (uint64_t core : spec.cores)
        artifact.cores.push_back({core, {}});
    artifact.envelope.active_cores = spec.cores;
    for (uint64_t rank = 0; rank < spec.cores.size(); ++rank)
        artifact.envelope.start_events.push_back(
            {spec.cores[rank], 0x700 + rank, 1});
    artifact.envelope.terminal_cores = spec.cores;
    artifact.envelope.expected_ack_cores = spec.cores;
    artifact.envelope.expected_done_cores = spec.cores;
    artifact.envelope.empty_core_ack_policy =
        EmptyCoreAckPolicy::INCLUDE_EMPTY;

    if (spec.negative == P6Negative::GROUP)
        ++spec.group_id;
    AppendP6PlanRecords(artifact, spec);

    if (spec.negative == P6Negative::SCHEDULE) {
        P6CollectiveFixtureSpec second = spec;
        ++second.collective_id;
        second.token_base += 0x10000;
        // Reusing the full logical fsm range across two keys is the
        // representable artifact-level scheduling conflict.
        AppendP6PlanRecords(artifact, second);
    }
    return artifact;
}
uint64_t P6WaveCount(const P6CollectiveFixtureSpec &spec) {
    constexpr uint64_t kSessionsPerRank = 3;
    constexpr uint64_t kReceiveBytesPerRank = 32768;
    const P6Op op = P6Operation(spec.tx, spec.rx);
    const uint64_t n = spec.cores.size();
    const uint64_t root = P6RootRank(op, n);
    std::vector<std::pair<uint64_t, uint64_t>> flows;
    auto add = [&](uint64_t source, uint64_t destination) {
        if (source != destination)
            flows.emplace_back(source, destination);
    };
    if (op == P6Op::P2P) {
        add(0, n - 1);
    } else if (P6RootTransmit(op)) {
        for (uint64_t destination = 0; destination < n; ++destination)
            add(root, destination);
    } else if (P6RootReceive(op)) {
        for (uint64_t source = 0; source < n; ++source)
            add(source, root);
    } else {
        for (uint64_t source = 0; source < n; ++source)
            for (uint64_t destination = 0; destination < n; ++destination)
                add(source, destination);
    }
    if (flows.empty()) return 1;

    uint64_t waves = 1;
    std::vector<uint64_t> sessions(n, 0);
    std::vector<uint64_t> receive_bytes(n, 0);
    for (const auto &flow : flows) {
        const uint64_t source = flow.first;
        const uint64_t destination = flow.second;
        auto fits = [&]() {
            return sessions[source] < kSessionsPerRank &&
                   sessions[destination] < kSessionsPerRank &&
                   receive_bytes[destination] <=
                       kReceiveBytesPerRank - spec.length_bytes;
        };
        if (!fits()) {
            ++waves;
            std::fill(sessions.begin(), sessions.end(), 0);
            std::fill(receive_bytes.begin(), receive_bytes.end(), 0);
        }
        if (!fits())
            throw std::logic_error(
                "P6 child cannot fit an empty production wave");
        ++sessions[source];
        ++sessions[destination];
        receive_bytes[destination] += spec.length_bytes;
    }
    return waves;
}

struct P6ExpectedCounts {
    uint64_t child = 0;
    uint64_t local_copy = 0;
    uint64_t reduce = 0;
    uint64_t wave = 1;
    uint64_t action = 0;
    uint64_t issue = 0;
};

P6ExpectedCounts P6Counts(const P6CollectiveFixtureSpec &spec) {
    const P6Op op = P6Operation(spec.tx, spec.rx);
    const uint64_t n = spec.cores.size();
    const uint64_t root = P6RootRank(op, n);
    P6ExpectedCounts out;
    out.wave = P6WaveCount(spec);
    if (op == P6Op::P2P) {
        out.local_copy = n == 1 ? 1 : 0;
        out.child = n == 1 ? 0 : 1;
    } else if (P6RootTransmit(op) || P6RootReceive(op)) {
        out.local_copy = 1;
        out.child = n - 1;
    } else {
        out.local_copy = n;
        out.child = n * (n - 1);
    }
    if (P6Reduction(op) && n > 1)
        out.reduce = op == P6Op::REDUCE ? 1 : n;
    out.action = out.local_copy + 5 * out.child +
                 2 * n * out.wave + out.reduce;
    for (uint64_t rank = 0; rank < n; ++rank) {
        out.issue += P6WantSend(op, rank, root, n) ? 1 : 0;
        out.issue += P6WantReceive(op, rank, root, n) ? 1 : 0;
        out.issue += P6WantCompute(op, rank, root) ? 1 : 0;
    }
    return out;
}

uint8_t P6PatternSeed(uint64_t rank) {
    return static_cast<uint8_t>(0x21 + rank * 0x31);
}

uint8_t P6PatternMultiplier(uint64_t rank) {
    return static_cast<uint8_t>(3 + rank * 2);
}

std::vector<uint8_t> P6RankPattern(
    const P6CollectiveFixtureSpec &spec, uint64_t rank,
    uint64_t bytes) {
    std::vector<uint8_t> result(static_cast<std::size_t>(bytes));
    const uint8_t seed = P6PatternSeed(rank);
    const uint8_t multiplier = P6PatternMultiplier(rank);
    for (std::size_t index = 0; index < result.size(); ++index)
        result[index] = static_cast<uint8_t>(
            seed + multiplier * index + (index >> 2));
    if (spec.rx == DteRecvMode::REDUCE) {
        const uint64_t width = P6DTypeBytes(spec.dtype);
        uint64_t boundary = 0;
        const uint64_t lane = rank & 3;
        if (spec.dtype == EndpointDataType::UINT8) {
            static constexpr std::array<uint64_t, 4> kSum = {
                250, 10, 255, 1};
            static constexpr std::array<uint64_t, 4> kMax = {
                0, 255, 127, 128};
            boundary = spec.reduce_op == ReduceOperator::SUM
                           ? kSum[lane]
                           : kMax[lane];
        } else if (spec.dtype == EndpointDataType::INT32) {
            static constexpr std::array<uint64_t, 4> kSum = {
                UINT32_C(0x7fffffff), 1,
                UINT32_C(0xffffffff), UINT32_C(0x80000000)};
            static constexpr std::array<uint64_t, 4> kMax = {
                UINT32_C(0x80000000), UINT32_C(0xffffffff),
                UINT32_C(0x7fffffff), 0};
            boundary = spec.reduce_op == ReduceOperator::SUM
                           ? kSum[lane]
                           : kMax[lane];
        } else {
            static constexpr std::array<uint64_t, 4> kSum = {
                UINT64_C(0x7fffffffffffffff), 1,
                UINT64_C(0xffffffffffffffff),
                UINT64_C(0x8000000000000000)};
            static constexpr std::array<uint64_t, 4> kMax = {
                UINT64_C(0x8000000000000000),
                UINT64_C(0xffffffffffffffff),
                UINT64_C(0x7fffffffffffffff), 0};
            boundary = spec.reduce_op == ReduceOperator::SUM
                           ? kSum[lane]
                           : kMax[lane];
        }
        for (uint64_t offset = 0; offset < bytes;
             offset += spec.length_bytes)
            for (uint64_t byte = 0; byte < width; ++byte)
                result[offset + byte] =
                    static_cast<uint8_t>(boundary >> (byte * 8));
    }
    return result;
}

uint32_t P6Crc32c(const std::vector<uint8_t> &bytes) {
    uint32_t checksum = UINT32_MAX;
    for (uint8_t value : bytes) {
        checksum ^= value;
        for (int bit = 0; bit < 8; ++bit)
            checksum = (checksum >> 1) ^
                       ((checksum & 1U) ? UINT32_C(0x82f63b78) : 0U);
    }
    return ~checksum;
}
uint64_t P6LoadLittle(const std::vector<uint8_t> &bytes,
                      uint64_t offset, uint64_t width) {
    uint64_t result = 0;
    for (uint64_t index = 0; index < width; ++index)
        result |= static_cast<uint64_t>(bytes.at(offset + index))
                  << (index * 8);
    return result;
}

void P6StoreLittle(std::vector<uint8_t> &bytes, uint64_t offset,
                   uint64_t width, uint64_t value) {
    for (uint64_t index = 0; index < width; ++index)
        bytes.at(offset + index) =
            static_cast<uint8_t>(value >> (index * 8));
}

bool P6SignedLess(uint64_t left, uint64_t right, uint64_t width) {
    if (width == 1) return left < right;
    const uint64_t sign = uint64_t{1} << (width * 8 - 1);
    if ((left & sign) != (right & sign))
        return (left & sign) != 0;
    return left < right;
}

std::vector<uint8_t> P6ReduceBytes(
    const P6CollectiveFixtureSpec &spec, uint64_t destination_rank) {
    const uint64_t n = spec.cores.size();
    const uint64_t width = P6DTypeBytes(spec.dtype);
    std::vector<uint8_t> result(spec.length_bytes);
    for (uint64_t byte = 0; byte < spec.length_bytes; byte += width) {
        uint64_t aggregate = 0;
        bool first = true;
        for (uint64_t source_rank = 0; source_rank < n; ++source_rank) {
            const std::vector<uint8_t> source = P6RankPattern(
                spec, source_rank,
                spec.tx == DteSendMode::SCATTER
                    ? n * spec.length_bytes
                    : spec.length_bytes);
            const uint64_t source_offset =
                (spec.tx == DteSendMode::SCATTER
                     ? destination_rank * spec.length_bytes
                     : 0) + byte;
            const uint64_t value =
                P6LoadLittle(source, source_offset, width);
            if (first || (spec.reduce_op == ReduceOperator::MAX &&
                          P6SignedLess(aggregate, value, width)))
                aggregate = value;
            else if (spec.reduce_op == ReduceOperator::SUM)
                aggregate += value;
            first = false;
        }
        P6StoreLittle(result, byte, width, aggregate);
    }
    return result;
}

std::vector<uint8_t> P6ExpectedRankBytes(
    const P6CollectiveFixtureSpec &spec, uint64_t destination_rank) {
    const P6Op op = P6Operation(spec.tx, spec.rx);
    const uint64_t n = spec.cores.size();
    const uint64_t root = P6RootRank(op, n);
    if (P6Reduction(op))
        return P6ReduceBytes(spec, destination_rank);

    std::vector<uint8_t> result;
    auto append = [&](uint64_t source_rank, uint64_t source_offset) {
        std::vector<uint8_t> source = P6RankPattern(
            spec, source_rank,
            spec.tx == DteSendMode::SCATTER
                ? n * spec.length_bytes
                : spec.length_bytes);
        result.insert(
            result.end(), source.begin() + source_offset,
            source.begin() + source_offset + spec.length_bytes);
    };
    if (op == P6Op::P2P) {
        if (destination_rank == n - 1)
            append(0, 0);
    } else if (P6RootTransmit(op)) {
        const uint64_t source_offset =
            spec.tx == DteSendMode::SCATTER
                ? destination_rank * spec.length_bytes
                : 0;
        append(root, source_offset);
    } else if (P6RootReceive(op)) {
        if (destination_rank == root)
            for (uint64_t source = 0; source < n; ++source)
                append(source, 0);
    } else {
        for (uint64_t source = 0; source < n; ++source) {
            const uint64_t source_offset =
                spec.tx == DteSendMode::SCATTER
                    ? destination_rank * spec.length_bytes
                    : 0;
            append(source, source_offset);
        }
    }
    return result;
}

void PrintP6CollectiveManifest(const P6CollectiveFixtureSpec &spec) {
    const P6Op op = P6Operation(spec.tx, spec.rx);
    const P6ExpectedCounts counts = P6Counts(spec);
    const P6SramLayout layout = P6Layout(spec);
    std::cout << "P6_COLLECTIVE scenario=" << spec.scenario
              << " tx=" << P6TxName(spec.tx)
              << " rx=" << P6RxName(spec.rx)
              << " op=" << P6OpName(op)
              << " n=" << spec.cores.size()
              << " length_bytes=" << spec.length_bytes
              << " dtype=" << P6DTypeName(spec.dtype)
              << " reduce_op=" << P6ReduceName(spec.reduce_op)
              << " group_id=" << spec.group_id
              << " collective_id=" << spec.collective_id
              << " epoch=" << spec.epoch
              << " fsm_base=" << spec.fsm_base
              << " negative=" << P6NegativeName(spec.negative)
              << " child_count=" << counts.child
              << " local_copy_count=" << counts.local_copy
              << " reduce_count=" << counts.reduce
              << " wave_count=" << counts.wave
              << " action_count=" << counts.action
              << " issue_count=" << counts.issue
              << " input_base=" << layout.input_base
              << " input_size=" << layout.input_size
              << " staging_base=" << layout.staging_base
              << " staging_size=" << layout.staging_size
              << " result_base=" << layout.result_base
              << " result_size=" << layout.result_size
              << " payload_offset=" << kP6PayloadOffset
              << " sentinel=" << static_cast<unsigned>(kP6Sentinel)
              << " cores=";
    for (std::size_t rank = 0; rank < spec.cores.size(); ++rank) {
        if (rank != 0) std::cout << ",";
        std::cout << spec.cores[rank];
    }
    std::cout << "\n";

    for (uint64_t rank = 0; rank < spec.cores.size(); ++rank) {
        const uint64_t source_bytes =
            spec.tx == DteSendMode::SCATTER
                ? spec.cores.size() * spec.length_bytes
                : spec.length_bytes;
        const std::vector<uint8_t> source =
            P6RankPattern(spec, rank, source_bytes);
        std::cout << "P6_SOURCE rank=" << rank
                  << " core=" << spec.cores[rank]
                  << " seed="
                  << static_cast<unsigned>(P6PatternSeed(rank))
                  << " multiplier="
                  << static_cast<unsigned>(P6PatternMultiplier(rank))
                  << " bytes=" << source.size()
                  << " checksum=" << P6Crc32c(source) << "\n";
        if (!P6Reduction(op)) {
            const std::vector<uint8_t> expected =
                P6ExpectedRankBytes(spec, rank);
            if (!expected.empty())
                std::cout << "P6_EXPECT rank=" << rank
                          << " core=" << spec.cores[rank]
                          << " region=p6_staging"
                          << " offset=" << kP6PayloadOffset
                          << " bytes=" << expected.size()
                          << " checksum=" << P6Crc32c(expected)
                          << " sentinel="
                          << static_cast<unsigned>(kP6Sentinel) << "\n";
        } else if (P6WantCompute(op, rank,
                                 P6RootRank(op, spec.cores.size()))) {
            std::cout << "P6_EXPECT rank=" << rank
                      << " core=" << spec.cores[rank]
                      << " region=p6_result"
                      << " offset=" << kP6PayloadOffset
                      << " bytes=" << spec.length_bytes
                      << " checksum="
                      << P6Crc32c(
                             P6ExpectedRankBytes(spec, rank))
                      << " sentinel="
                      << static_cast<unsigned>(kP6Sentinel) << "\n";
        }
    }
}

constexpr uint64_t kP8BGroupId = 83;
constexpr uint64_t kP8BPayloadOffset = 64;
constexpr uint64_t kP8BP2pBytes = 97;
constexpr uint64_t kP8BAllGatherBytes = 68;
constexpr uint64_t kP8BAllReduceBytes = 64;
constexpr uint64_t kP8BRegionBytes = 4096;
constexpr uint64_t kP8BComputeBase = 4096;
constexpr uint64_t kP8BP2pSourceBase = 12288;
constexpr uint64_t kP8BP2pDestinationBase = 20480;
constexpr uint64_t kP8BAllGatherInputBase = 28672;
constexpr uint64_t kP8BAllGatherStagingBase = 36864;
constexpr uint64_t kP8BAllReduceInputBase = 45056;
constexpr uint64_t kP8BAllReduceStagingBase = 53248;
constexpr uint64_t kP8BAllReduceResultBase = 61440;
constexpr uint64_t kP8BP2pFsm = 0x68000001;
constexpr uint64_t kP8BP2pSendToken = 0x68000101;
constexpr uint64_t kP8BP2pRecvToken = 0x68000102;
constexpr std::array<uint64_t, 4> kP8BCores = {1, 3, 7, 11};

ExternalRecord P8BAllocRecord(uint64_t label_symbol) {
    SramAllocOperands operands;
    operands.region_name_string_index = 7;
    operands.label_symbol_index = label_symbol;
    operands.size_bytes = 256;
    operands.alignment_bytes = 64;
    operands.lifetime = SramLifetime::PERSISTENT;
    operands.spillable = false;
    return {Opcode::SRAM_ALLOC, std::move(operands)};
}

ExternalRecord P8BGroupSyncRecord(uint64_t sequence) {
    return {Opcode::GROUP_SYNC,
            GroupSyncOperands{kP8BGroupId, sequence}};
}

ExternalRecord P8BP2pSendRecord() {
    DteSendOperands operands;
    operands.mode = DteSendMode::P2P;
    operands.source_space = EndpointSourceSpace::SRAM;
    operands.completion = EndpointCompletion::ASYNC;
    operands.datatype = EndpointDataType::UINT8;
    operands.reduce_op = ReduceOperator::NONE;
    operands.fsm_id = kP8BP2pFsm;
    operands.token = kP8BP2pSendToken;
    operands.length_bytes = kP8BP2pBytes;
    operands.source = P6RegionAddress(0, kP8BPayloadOffset);
    operands.peer_core = kP8BCores[1];
    return {Opcode::DTE_SEND, std::move(operands)};
}

ExternalRecord P8BP2pRecvRecord() {
    DteRecvOperands operands;
    operands.mode = DteRecvMode::P2P;
    operands.completion = EndpointCompletion::ASYNC;
    operands.datatype = EndpointDataType::UINT8;
    operands.reduce_op = ReduceOperator::NONE;
    operands.fsm_id = kP8BP2pFsm;
    operands.token = kP8BP2pRecvToken;
    operands.length_bytes = kP8BP2pBytes;
    operands.destination = P6RegionAddress(1, kP8BPayloadOffset);
    operands.peer_core = kP8BCores[0];
    return {Opcode::DTE_RECV, std::move(operands)};
}

P6CollectiveFixtureSpec P8BAllGatherSpec() {
    P6CollectiveFixtureSpec spec = MakeP6BaseSpec(
        DteSendMode::BROADCAST, DteRecvMode::GATHER, 4,
        kP8BAllGatherBytes, EndpointDataType::UINT8,
        ReduceOperator::NONE, "p8_program_b_allgather");
    spec.group_id = kP8BGroupId;
    spec.collective_id = 0x680101;
    spec.epoch = 1;
    spec.fsm_base = 0x68100000;
    spec.token_base = 0x68200000;
    return spec;
}

P6CollectiveFixtureSpec P8BAllReduceSpec() {
    P6CollectiveFixtureSpec spec = MakeP6BaseSpec(
        DteSendMode::BROADCAST, DteRecvMode::REDUCE, 4,
        kP8BAllReduceBytes, EndpointDataType::INT32,
        ReduceOperator::SUM, "p8_program_b_allreduce");
    spec.group_id = kP8BGroupId;
    spec.collective_id = 0x680102;
    spec.epoch = 1;
    spec.fsm_base = 0x68300000;
    spec.token_base = 0x68400000;
    return spec;
}

void AppendP8BCollectiveRecords(
    ProgramArtifact &artifact, const P6CollectiveFixtureSpec &spec,
    uint64_t source_symbol, uint64_t staging_symbol,
    uint64_t result_symbol, bool append_fence) {
    const P6Op op = P6Operation(spec.tx, spec.rx);
    const uint64_t n = spec.cores.size();
    const uint64_t root = P6RootRank(op, n);
    for (uint64_t rank = 0; rank < n; ++rank) {
        auto found = std::find_if(
            artifact.cores.begin(), artifact.cores.end(),
            [&](const ProgramCore &core) {
                return core.core_id == spec.cores[rank];
            });
        if (found == artifact.cores.end())
            throw std::logic_error("P8-B artifact core is missing");
        std::vector<uint64_t> waits;
        if (P6WantSend(op, rank, root, n)) {
            ExternalRecord send = P6SendRecord(spec, rank);
            std::get<DteSendOperands>(send.operands).source =
                P6RegionAddress(source_symbol, kP8BPayloadOffset);
            found->records.push_back(std::move(send));
            waits.push_back(P6Token(spec, rank, false));
        }
        if (P6WantReceive(op, rank, root, n)) {
            ExternalRecord receive = P6ReceiveRecord(spec, rank);
            std::get<DteRecvOperands>(receive.operands).destination =
                P6RegionAddress(staging_symbol, kP8BPayloadOffset);
            found->records.push_back(std::move(receive));
            waits.push_back(P6Token(spec, rank, true));
        }
        if (P6WantCompute(op, rank, root)) {
            ExternalRecord compute = P6ComputeRecord(spec, rank, root);
            auto &operands =
                std::get<ReduceComputeOperands>(compute.operands);
            operands.source = P6RegionAddress(staging_symbol, kP8BPayloadOffset);
            operands.destination = P6RegionAddress(result_symbol, kP8BPayloadOffset);
            found->records.push_back(std::move(compute));
        }
        for (const uint64_t token : waits)
            found->records.push_back(
                {Opcode::DTE_WAIT, TokenOperands{token}});
        if (append_fence)
            found->records.push_back({Opcode::DTE_FENCE, NoOperands{}});
    }
}

void AppendP8BSync(ProgramArtifact &artifact, uint64_t sequence) {
    for (ProgramCore &core : artifact.cores)
        core.records.push_back(P8BGroupSyncRecord(sequence));
}

void AppendP8BInitialCompute(ProgramArtifact &artifact) {
    for (ProgramCore &core : artifact.cores) {
        core.records.push_back(P8BAllocRecord(8));
        core.records.push_back(ChainBindRecord(8, 8));
        core.records.push_back(
            ChainComputeRecord(Opcode::GELU, {256}, 0, 0, 0));
    }
}

void AppendP8BFinalCompute(ProgramArtifact &artifact) {
    for (ProgramCore &core : artifact.cores) {
        core.records.push_back(
            {Opcode::SRAM_RENAME, SramRenameOperands{8, 9}});
        core.records.push_back(ChainBindRecord(9, 9));
        core.records.push_back(
            ChainComputeRecord(Opcode::GELU, {256}, 0, 0, 0));
        core.records.push_back({Opcode::SRAM_FREE, SymbolOperands{9}});
    }
}

ProgramArtifact P8ProgramBArtifact() {
    ProgramArtifact artifact;
    artifact.strings = {
        "p8b_p2p_source", "p8b_p2p_destination",
        "p8b_allgather_input", "p8b_allgather_staging",
        "p8b_allreduce_input", "p8b_allreduce_staging",
        "p8b_allreduce_result", "p8b_compute",
        "p8b_precompute_label", "p8b_final_compute_label"};
    artifact.symbols = {
        {0, ProgramSymbolKind::SRAM_REGION, 0,
         kP8BP2pSourceBase, kP8BRegionBytes},
        {1, ProgramSymbolKind::SRAM_REGION, 0,
         kP8BP2pDestinationBase, kP8BRegionBytes},
        {2, ProgramSymbolKind::SRAM_REGION, 0,
         kP8BAllGatherInputBase, kP8BRegionBytes},
        {3, ProgramSymbolKind::SRAM_REGION, 0,
         kP8BAllGatherStagingBase, kP8BRegionBytes},
        {4, ProgramSymbolKind::SRAM_REGION, 0,
         kP8BAllReduceInputBase, kP8BRegionBytes},
        {5, ProgramSymbolKind::SRAM_REGION, 0,
         kP8BAllReduceStagingBase, kP8BRegionBytes},
        {6, ProgramSymbolKind::SRAM_REGION, 0,
         kP8BAllReduceResultBase, kP8BRegionBytes},
        {7, ProgramSymbolKind::SRAM_REGION, 0,
         kP8BComputeBase, kP8BRegionBytes},
        {8, ProgramSymbolKind::SRAM_LABEL, 0, 0, 256},
        {9, ProgramSymbolKind::SRAM_LABEL, 0, 0, 256},
    };
    artifact.core_groups = {
        {kP8BGroupId, std::vector<uint64_t>(
             kP8BCores.begin(), kP8BCores.end())}};
    for (const uint64_t core : kP8BCores)
        artifact.cores.push_back({core, {}});

    AppendP8BInitialCompute(artifact);
    for (ProgramCore &core : artifact.cores) {
        if (core.core_id == kP8BCores[0]) {
            core.records.push_back(P8BP2pSendRecord());
            core.records.push_back(
                {Opcode::DTE_WAIT, TokenOperands{kP8BP2pSendToken}});
        } else if (core.core_id == kP8BCores[1]) {
            core.records.push_back(P8BP2pRecvRecord());
            core.records.push_back(
                {Opcode::DTE_WAIT, TokenOperands{kP8BP2pRecvToken}});
        }
    }
    AppendP8BSync(artifact, 0);

    const P6CollectiveFixtureSpec allgather = P8BAllGatherSpec();
    AppendP8BCollectiveRecords(artifact, allgather, 2, 3, 0, false);
    AppendP8BSync(artifact, 1);

    const P6CollectiveFixtureSpec allreduce = P8BAllReduceSpec();
    AppendP8BCollectiveRecords(artifact, allreduce, 4, 5, 6, true);
    AppendP8BSync(artifact, 2);
    AppendP8BFinalCompute(artifact);
    AppendP8BSync(artifact, 3);

    artifact.envelope.active_cores.assign(
        kP8BCores.begin(), kP8BCores.end());
    for (uint64_t rank = 0; rank < kP8BCores.size(); ++rank)
        artifact.envelope.start_events.push_back(
            {kP8BCores[rank], 0x800 + rank, 1});
    artifact.envelope.terminal_cores = artifact.envelope.active_cores;
    artifact.envelope.expected_ack_cores = artifact.envelope.active_cores;
    artifact.envelope.expected_done_cores = artifact.envelope.active_cores;
    artifact.envelope.empty_core_ack_policy =
        EmptyCoreAckPolicy::INCLUDE_EMPTY;
    return artifact;
}

void PrintP8ProgramBManifest() {
    const P6CollectiveFixtureSpec allgather = P8BAllGatherSpec();
    const P6CollectiveFixtureSpec allreduce = P8BAllReduceSpec();
    const P6ExpectedCounts ag = P6Counts(allgather);
    const P6ExpectedCounts ar = P6Counts(allreduce);
    std::cout << "P8_PROGRAM_B scenario=multicore_p2p_collectives_sync"
              << " cores=1,3,7,11 group_id=" << kP8BGroupId
              << " p2p_bytes=" << kP8BP2pBytes
              << " payload_offset=" << kP8BPayloadOffset
              << " allgather_bytes=" << kP8BAllGatherBytes
              << " allreduce_bytes=" << kP8BAllReduceBytes
              << " collective_children=" << ag.child + ar.child
              << " collective_actions=" << ag.action + ar.action
              << " collective_waves=" << ag.wave + ar.wave
              << " compute_count=8 bind_count=8 alloc_count=4"
              << " rename_count=4 free_count=4"
              << " p2p_send_count=1 p2p_recv_count=1"
              << " collective_send_count=8"
              << " collective_recv_count=8"
              << " reduce_compute_count=4"
              << " wait_count=18 fence_count=4"
              << " group_sync_count=16 group_sync_sequences=4"
              << " profiles=baseline,broadcast_only,reduce_only,"
                 "reduce_broadcast"
              << " data_base=" << kP8BP2pSourceBase
              << " compute_base=" << kP8BComputeBase
              << " region_bytes=" << kP8BRegionBytes << "\n";
}

constexpr std::array<uint64_t, 2> kFrontendN0Cores = {0, 16};
constexpr uint64_t kFrontendN0DoubleABase = 0;
constexpr uint64_t kFrontendN0DoubleBBase = 2048;
constexpr uint64_t kFrontendN0InputBase = 4096;
constexpr uint64_t kFrontendN0IntermediateBase = 8192;
constexpr uint64_t kFrontendN0CommBase = 12288;
constexpr uint64_t kFrontendN0DoubleBufferBytes = 2048;
constexpr uint64_t kFrontendN0RegionBytes = 4096;
constexpr uint64_t kFrontendN0PayloadOffset = 64;
constexpr uint64_t kFrontendN0ChunkElements = 16;
constexpr uint64_t kFrontendN0ChunkBytes =
    kFrontendN0ChunkElements * 2;
constexpr uint64_t kFrontendN0Sentinel = 0xa5;
constexpr uint64_t kFrontendN0Fsm0To16 = 0x4e100001;
constexpr uint64_t kFrontendN0Fsm16To0 = 0x4e100002;
constexpr uint64_t kFrontendN0Core0RecvToken = 0x4e200001;
constexpr uint64_t kFrontendN0Core0SendToken = 0x4e200002;
constexpr uint64_t kFrontendN0Core16RecvToken = 0x4e200003;
constexpr uint64_t kFrontendN0Core16SendToken = 0x4e200004;

enum FrontendN0Symbol : uint64_t {
    N0_DOUBLE_A = 0,
    N0_DOUBLE_B = 1,
    N0_INPUT = 2,
    N0_INTERMEDIATE = 3,
    N0_COMM = 4,
    N0_GEMM_INPUT = 5,
    N0_GEMM_OUTPUT = 6,
    N0_GEMM_WEIGHT = 7,
    N0_GEMM_BIAS = 8,
};

SramAddressOperand FrontendN0RegionAddress(uint64_t symbol,
                                           uint64_t offset) {
    SramAddressOperand address;
    address.kind = SramAddressKind::REGION;
    address.region_symbol_index = symbol;
    address.region_offset_bytes = offset;
    return address;
}

SramAddressOperand FrontendN0AbsoluteAddress(uint64_t address_bytes) {
    SramAddressOperand address;
    address.kind = SramAddressKind::ABSOLUTE;
    address.absolute_address_bytes = address_bytes;
    return address;
}

ExternalRecord FrontendN0AllocRecord(uint64_t region_string,
                                     uint64_t label_symbol) {
    SramAllocOperands operands;
    operands.region_name_string_index = region_string;
    operands.label_symbol_index = label_symbol;
    operands.size_bytes = 256;
    operands.alignment_bytes = 64;
    operands.lifetime = SramLifetime::TASK;
    operands.spillable = true;
    return {Opcode::SRAM_ALLOC, std::move(operands)};
}

ExternalRecord FrontendN0PersistentAllocRecord(uint64_t label_symbol,
                                               uint64_t size_bytes) {
    SramAllocOperands operands;
    operands.region_name_string_index = N0_INPUT;
    operands.label_symbol_index = label_symbol;
    operands.size_bytes = size_bytes;
    operands.alignment_bytes = 64;
    operands.lifetime = SramLifetime::PERSISTENT;
    operands.spillable = true;
    return {Opcode::SRAM_ALLOC, std::move(operands)};
}

ExternalRecord FrontendN0BindRecord() {
    SramBindOperands operands;
    operands.input_count = 1;
    operands.input_symbol_indices[0] = N0_GEMM_INPUT;
    operands.output_symbol_index = N0_GEMM_OUTPUT;
    return {Opcode::SRAM_BIND, std::move(operands)};
}

ExternalRecord FrontendN0MatmulRecord() {
    ComputeOperands operands;
    operands.datatype = ExternalDataType::FP16;
    operands.input_offset_bytes =
        kFrontendN0InputBase + kFrontendN0PayloadOffset;
    operands.data_offset_bytes =
        kFrontendN0InputBase + kFrontendN0PayloadOffset + 64;
    operands.output_offset_bytes =
        kFrontendN0IntermediateBase + kFrontendN0PayloadOffset;
    operands.parameters = {1, 1, 8, 8};
    return {Opcode::MATMUL, std::move(operands)};
}

ExternalRecord FrontendN0ReceiveRecord(uint64_t core) {
    const bool core0 = core == kFrontendN0Cores[0];
    DteRecvOperands operands;
    operands.mode = DteRecvMode::P2P;
    operands.completion = EndpointCompletion::ASYNC;
    operands.datatype = EndpointDataType::UINT8;
    operands.reduce_op = ReduceOperator::NONE;
    operands.fsm_id = core0 ? kFrontendN0Fsm16To0
                            : kFrontendN0Fsm0To16;
    operands.token = core0 ? kFrontendN0Core0RecvToken
                           : kFrontendN0Core16RecvToken;
    operands.length_bytes = kFrontendN0ChunkBytes;
    operands.destination = FrontendN0RegionAddress(
        N0_COMM, kFrontendN0PayloadOffset +
                     (core0 ? kFrontendN0ChunkBytes : 0));
    operands.peer_core = core0 ? kFrontendN0Cores[1]
                               : kFrontendN0Cores[0];
    return {Opcode::DTE_RECV, std::move(operands)};
}

ExternalRecord FrontendN0SendRecord(uint64_t core) {
    const bool core0 = core == kFrontendN0Cores[0];
    DteSendOperands operands;
    operands.mode = DteSendMode::P2P;
    operands.source_space = EndpointSourceSpace::SRAM;
    operands.completion = EndpointCompletion::ASYNC;
    operands.datatype = EndpointDataType::UINT8;
    operands.reduce_op = ReduceOperator::NONE;
    operands.fsm_id = core0 ? kFrontendN0Fsm0To16
                            : kFrontendN0Fsm16To0;
    operands.token = core0 ? kFrontendN0Core0SendToken
                           : kFrontendN0Core16SendToken;
    operands.length_bytes = kFrontendN0ChunkBytes;
    operands.source = FrontendN0RegionAddress(
        N0_DOUBLE_B, kFrontendN0PayloadOffset +
                         (core0 ? kFrontendN0ChunkBytes : 0));
    operands.peer_core = core0 ? kFrontendN0Cores[1]
                               : kFrontendN0Cores[0];
    return {Opcode::DTE_SEND, std::move(operands)};
}

ExternalRecord FrontendN0LocalReduceRecord(uint64_t core) {
    const bool core0 = core == kFrontendN0Cores[0];
    LocalReduceOperands operands;
    operands.input_count = 2;
    operands.element_count = kFrontendN0ChunkElements;
    operands.input_stride_bytes = kFrontendN0ChunkBytes;
    operands.source = FrontendN0AbsoluteAddress(
        kFrontendN0CommBase + kFrontendN0PayloadOffset);
    operands.destination = FrontendN0AbsoluteAddress(
        kFrontendN0DoubleABase + kFrontendN0PayloadOffset +
        (core0 ? 0 : kFrontendN0ChunkBytes));
    return {Opcode::LOCAL_REDUCE, std::move(operands)};
}

std::vector<ExternalRecord> FrontendN0CoreRecords(uint64_t core) {
    const bool core0 = core == kFrontendN0Cores[0];
    const uint64_t receive_token = core0 ? kFrontendN0Core0RecvToken
                                         : kFrontendN0Core16RecvToken;
    const uint64_t send_token = core0 ? kFrontendN0Core0SendToken
                                      : kFrontendN0Core16SendToken;
    return {
        // Matmul_f otherwise performs its legacy static-data first write at
        // address zero before assigning a preferred region. Pre-register the
        // exact eternal labels so timing-only compute cannot clobber the
        // functional double_a oracle.
        FrontendN0PersistentAllocRecord(N0_GEMM_WEIGHT, 128),
        FrontendN0PersistentAllocRecord(N0_GEMM_BIAS, 16),
        FrontendN0AllocRecord(N0_INPUT, N0_GEMM_INPUT),
        FrontendN0AllocRecord(N0_INTERMEDIATE, N0_GEMM_OUTPUT),
        FrontendN0BindRecord(),
        FrontendN0MatmulRecord(),
        {Opcode::SRAM_FREE, SymbolOperands{N0_GEMM_OUTPUT}},
        {Opcode::SRAM_FREE, SymbolOperands{N0_GEMM_INPUT}},
        FrontendN0ReceiveRecord(core),
        FrontendN0SendRecord(core),
        {Opcode::DTE_WAIT, TokenOperands{receive_token}},
        {Opcode::DTE_WAIT, TokenOperands{send_token}},
        FrontendN0LocalReduceRecord(core),
    };
}

ProgramArtifact FrontendN0Tp2RsArtifact() {
    ProgramArtifact artifact;
    artifact.strings = {
        "double_a", "double_b", "input", "intermediate", "comm",
        "frontend_n0_gemm_input", "frontend_n0_gemm_output",
        "eternal_frontend_n0_gemm_w", "eternal_frontend_n0_gemm_b"};
    artifact.symbols = {
        {0, ProgramSymbolKind::SRAM_REGION, 0,
         kFrontendN0DoubleABase, kFrontendN0DoubleBufferBytes},
        {1, ProgramSymbolKind::SRAM_REGION, 0,
         kFrontendN0DoubleBBase, kFrontendN0DoubleBufferBytes},
        {2, ProgramSymbolKind::SRAM_REGION, 0,
         kFrontendN0InputBase, kFrontendN0RegionBytes},
        {3, ProgramSymbolKind::SRAM_REGION, 0,
         kFrontendN0IntermediateBase, kFrontendN0RegionBytes},
        {4, ProgramSymbolKind::SRAM_REGION, 0,
         kFrontendN0CommBase, kFrontendN0RegionBytes},
        {5, ProgramSymbolKind::SRAM_LABEL, 0, 0, 256},
        {6, ProgramSymbolKind::SRAM_LABEL, 0, 0, 256},
        {7, ProgramSymbolKind::SRAM_LABEL, 0, 0, 128},
        {8, ProgramSymbolKind::SRAM_LABEL, 0, 0, 16},
    };
    artifact.cores = {
        {kFrontendN0Cores[0], FrontendN0CoreRecords(kFrontendN0Cores[0])},
        {kFrontendN0Cores[1], FrontendN0CoreRecords(kFrontendN0Cores[1])},
    };
    artifact.envelope.active_cores.assign(
        kFrontendN0Cores.begin(), kFrontendN0Cores.end());
    artifact.envelope.start_events = {
        {kFrontendN0Cores[0], 0x4e01, 1},
        {kFrontendN0Cores[1], 0x4e02, 1},
    };
    artifact.envelope.terminal_cores = artifact.envelope.active_cores;
    artifact.envelope.expected_ack_cores = artifact.envelope.active_cores;
    artifact.envelope.expected_done_cores = artifact.envelope.active_cores;
    artifact.envelope.empty_core_ack_policy =
        EmptyCoreAckPolicy::INCLUDE_EMPTY;
    return artifact;
}

void PrintFrontendN0Tp2RsManifest() {
    std::cout
        << "FRONTEND_N0_TP2_RS scenario=gemm_tp2_dual_owner_rs"
        << " cores=0,16 region_bytes=" << kFrontendN0RegionBytes
        << " double_buffer_bytes=" << kFrontendN0DoubleBufferBytes
        << " payload_offset=" << kFrontendN0PayloadOffset
        << " sentinel=" << kFrontendN0Sentinel
        << " double_a_base=" << kFrontendN0DoubleABase
        << " double_b_base=" << kFrontendN0DoubleBBase
        << " input_base=" << kFrontendN0InputBase
        << " intermediate_base=" << kFrontendN0IntermediateBase
        << " comm_base=" << kFrontendN0CommBase
        << " chunk_bytes=" << kFrontendN0ChunkBytes
        << " chunk_count=2 rank_count=2"
        << " MATMUL=2 P2P_SEND=2 P2P_RECV=2 WAIT=4 LOCAL_REDUCE=2"
        << " core0_owner_chunk=0 core16_owner_chunk=1"
        << " core0_staging=rank0,rank1"
        << " core16_staging=rank0,rank1"
        << " transport_dtype=UINT8 reduce_input=FP16"
        << " accumulator=FP32 reduce_output=FP16"
        << " rounding=RNE order=RANK_MAJOR\n";
}

ProgramArtifact FixtureArtifact(FixtureMode mode) {
    switch (mode) {
    case FixtureMode::BOUND_DUMMY:
    case FixtureMode::MISSING_BIND:
    case FixtureMode::WRONG_ARITY:
        return DummyArtifact(mode);
    case FixtureMode::P3_DIFF_CHAIN:
        return P3DiffChainArtifact();
    case FixtureMode::P4_LIFECYCLE:
        return LifecycleArtifact(false);
    case FixtureMode::P4_LIFECYCLE_DANGLING:
        return LifecycleArtifact(true);
    case FixtureMode::P4_SYNC_EVENT:
        return SyncEventArtifact(false);
    case FixtureMode::P4_SYNC_BAD_ENDPOINT:
        return SyncEventArtifact(true);
    case FixtureMode::P8_DOUBLE_BUFFER:
        return P8DoubleBufferArtifact();
    case FixtureMode::P8_PROGRAM_B:
        return P8ProgramBArtifact();
    case FixtureMode::FRONTEND_N0_TP2_RS:
        return FrontendN0Tp2RsArtifact();
    }
    throw std::logic_error("unknown fixture mode");
}

void WriteFile(const std::filesystem::path &path,
               const std::vector<uint8_t> &bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("cannot open output artifact: " +
                                 path.string());
    if (!bytes.empty())
        output.write(reinterpret_cast<const char *>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
    if (!output)
        throw std::runtime_error("cannot write output artifact: " +
                                 path.string());
}

} // namespace

int main(int argc, char **argv) {
    bool corrupt_crc = false;
    bool is_p5_endpoint = false;
    bool is_p6_collective = false;
    EndpointFixtureSpec p5_endpoint;
    P6CollectiveFixtureSpec p6_collective;
    FixtureMode mode = FixtureMode::BOUND_DUMMY;
    if (argc == 3) {
        const std::string option = argv[2];
        if (option == "--corrupt-crc")
            corrupt_crc = true;
        else if (option == "--missing-bind")
            mode = FixtureMode::MISSING_BIND;
        else if (option == "--wrong-arity")
            mode = FixtureMode::WRONG_ARITY;
        else if (option == "--p3-diff-chain")
            mode = FixtureMode::P3_DIFF_CHAIN;
        else if (option == "--p4-lifecycle")
            mode = FixtureMode::P4_LIFECYCLE;
        else if (option == "--p4-lifecycle-dangling")
            mode = FixtureMode::P4_LIFECYCLE_DANGLING;
        else if (option == "--p4-sync-event")
            mode = FixtureMode::P4_SYNC_EVENT;
        else if (option == "--p4-sync-bad-endpoint")
            mode = FixtureMode::P4_SYNC_BAD_ENDPOINT;
        else if (option == "--p8-double-buffer")
            mode = FixtureMode::P8_DOUBLE_BUFFER;
        else if (option == "--p8-program-b")
            mode = FixtureMode::P8_PROGRAM_B;
        else if (option == "--frontend-n0-tp2-rs")
            mode = FixtureMode::FRONTEND_N0_TP2_RS;
        else if (TryP5EndpointSpec(option, &p5_endpoint))
            is_p5_endpoint = true;
        else if (TryP6CollectiveSpec(option, &p6_collective))
            is_p6_collective = true;
        else
            argc = 0;
    }
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: npusim_program_fixture <output.npup> "
                     "[--corrupt-crc|--missing-bind|--wrong-arity|\n"
                     " --p3-diff-chain|--p4-lifecycle|\n"
                     " --p4-lifecycle-dangling|--p4-sync-event|\n"
                     " --p4-sync-bad-endpoint|\n"
                     " --p8-double-buffer|--p8-program-b|\n"
                     " --frontend-n0-tp2-rs|\n"
                     " --p5-sram-sync-{1,15,16,17,127,128,129,255,256,257}|\n"
                     " --p5-sram-async-{17,4k,32k}|\n"
                     " --p5-cross-die-sram-sync-128|--p5-hbm-sync-129|\n"
                     " --p5-pair-missing|--p5-pair-mismatch|\n"
                     " --p6-<unicast|scatter|broadcast>-"
                     "<unicast|gather|reduce>-n<1|2|4>-l<bytes>-"
                     "<u8|i32|i64>-<none|sum|max>|\n"
                     " --p6-bad-{group,rank,root,crossdie,count,"
                     "stride,schedule}]\n";
        return 2;
    }

    try {
        const ProgramArtifact artifact =
            is_p5_endpoint
                ? P5EndpointArtifact(p5_endpoint)
                : (is_p6_collective
                       ? P6CollectiveArtifact(p6_collective)
                       : FixtureArtifact(mode));
        std::vector<uint8_t> bytes = EncodeProgramArtifact(artifact);
        if (mode == FixtureMode::FRONTEND_N0_TP2_RS &&
            EncodeProgramArtifact(artifact) != bytes)
            throw std::logic_error(
                "FRONTEND_N0 repeated encode is not deterministic");
        if (corrupt_crc) {
            bytes.back() ^= 0x80;
        } else {
            const ProgramArtifact decoded = DecodeProgramArtifact(bytes);
            if (EncodeProgramArtifact(decoded) != bytes)
                throw std::logic_error(
                    "Program Format encode/decode is not stable");
        }
        WriteFile(argv[1], bytes);
        std::cout << "wrote Program Format " << kProgramFormatMajor << "."
                  << kProgramFormatMinor << " fixture: " << bytes.size()
                  << " bytes\n";
        if (mode == FixtureMode::P3_DIFF_CHAIN)
            PrintP3DiffManifest(artifact);
        if (is_p5_endpoint)
            PrintP5EndpointManifest(p5_endpoint);
        if (is_p6_collective)
            PrintP6CollectiveManifest(p6_collective);
        if (mode == FixtureMode::P8_DOUBLE_BUFFER)
            PrintP8DoubleBufferManifest();
        if (mode == FixtureMode::P8_PROGRAM_B)
            PrintP8ProgramBManifest();
        if (mode == FixtureMode::FRONTEND_N0_TP2_RS)
            PrintFrontendN0Tp2RsManifest();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "program fixture generation failed: " << error.what()
                  << "\n";
        return 1;
    }
}
