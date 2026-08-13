#include "monitor/config_helper_program_selftest.h"

#include "monitor/config_helper_program.h"
#include "dte/endpoint_contract.h"
#include "common/memory.h"
#include "defs/spec.h"
#include "prims/collective_launch_v1_prim.h"
#include "utils/prim_utils.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <array>
#include <string>
#include <string_view>
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
    void Reject(std::string name, std::string_view fragment,
                Function function) {
        ++result.checks;
        try {
            function();
        } catch (const Exception &error) {
            if (std::string(error.what()).find(fragment) == std::string::npos)
                result.failures.push_back(
                    std::move(name) + " (wrong diagnostic: " + error.what() +
                    ")");
            return;
        } catch (const std::exception &error) {
            result.failures.push_back(
                std::move(name) + " (wrong exception: " + error.what() +
                ")");
            return;
        }
        result.failures.push_back(std::move(name) + " (accepted)");
    }

    ConfigHelperProgramSelfTestResult result;
};

class NocCollectiveConfigGuard {
public:
    NocCollectiveConfigGuard() : previous_(SPEC_NOC_COLL_CONFIG) {}
    ~NocCollectiveConfigGuard() { SPEC_NOC_COLL_CONFIG = previous_; }
private:
    NocCollectiveConfig previous_;
};

class PlatformDimensionsGuard {
public:
    PlatformDimensionsGuard()
        : grid_x_(GRID_X), grid_y_(GRID_Y), grid_size_(GRID_SIZE),
          die_count_(DIE_COUNT), cores_per_die_(CORES_PER_DIE),
          total_cores_(TOTAL_CORES),
          host_endpoint_id_(HOST_ENDPOINT_ID) {
        GRID_X = GRID_Y = 2;
        GRID_SIZE = CORES_PER_DIE = 4;
        DIE_COUNT = 2;
        TOTAL_CORES = 8;
        HOST_ENDPOINT_ID = TOTAL_CORES;
    }

    ~PlatformDimensionsGuard() {
        GRID_X = grid_x_;
        GRID_Y = grid_y_;
        GRID_SIZE = grid_size_;
        DIE_COUNT = die_count_;
        CORES_PER_DIE = cores_per_die_;
        TOTAL_CORES = total_cores_;
        HOST_ENDPOINT_ID = host_endpoint_id_;
    }

private:
    int grid_x_;
    int grid_y_;
    int grid_size_;
    int die_count_;
    int cores_per_die_;
    int total_cores_;
    int host_endpoint_id_;
};

ExternalRecord MatmulRecord(uint64_t base = 1) {
    ExternalRecord record;
    record.opcode = Opcode::MATMUL;
    ComputeOperands operands;
    operands.datatype = ExternalDataType::FP16;
    operands.input_offset_bytes = base;
    operands.data_offset_bytes = base + 1;
    operands.output_offset_bytes = base + 2;
    operands.parameters = {2, 3, 4, 5};
    record.operands = operands;
    return record;
}

ExternalRecord ResidualRecord(uint64_t n = 8) {
    ExternalRecord record;
    record.opcode = Opcode::RESIDUAL;
    ComputeOperands operands;
    operands.datatype = ExternalDataType::FP16;
    operands.parameters = {n};
    record.operands = operands;
    return record;
}

ExternalRecord BindRecord(uint64_t input_count = 1,
                          uint64_t first_input = 0,
                          uint64_t output = 2) {
    ExternalRecord record;
    record.opcode = Opcode::SRAM_BIND;
    SramBindOperands operands;
    operands.input_count = input_count;
    for (std::size_t i = 0; i < input_count; ++i)
        operands.input_symbol_indices[i] = first_input + i;
    operands.output_symbol_index = output;
    record.operands = operands;
    return record;
}

std::vector<ExternalRecord> BoundMatmul(uint64_t base = 1) {
    return {BindRecord(), MatmulRecord(base)};
}

ExternalRecord FenceRecord() {
    return ExternalRecord{Opcode::DTE_FENCE, NoOperands{}};
}

ExternalRecord DteIssueRecord(uint64_t token) {
    DteIssueOperands operands;
    operands.direction = LocalDteDirection::SPM_TO_SPM;
    operands.token = token;
    operands.payload_bits = 64;
    operands.size_bytes = 8;
    operands.source_sram.kind = SramAddressKind::ABSOLUTE;
    operands.source_sram.absolute_address_bytes = 0x100;
    operands.destination_sram.kind = SramAddressKind::ABSOLUTE;
    operands.destination_sram.absolute_address_bytes = 0x200;
    return ExternalRecord{Opcode::DTE_ISSUE, operands};
}

ExternalRecord DteControlRecord(Opcode opcode, uint64_t token) {
    return ExternalRecord{opcode, TokenOperands{token}};
}

ExternalRecord AllocRecord(uint64_t label, SramLifetime lifetime,
                           bool spillable) {
    SramAllocOperands operands;
    operands.region_name_string_index = 0;
    operands.label_symbol_index = label;
    operands.size_bytes = 64;
    operands.alignment_bytes = 16;
    operands.lifetime = lifetime;
    operands.spillable = spillable;
    return ExternalRecord{Opcode::SRAM_ALLOC, operands};
}

ExternalRecord ResizeRecord(uint64_t label, uint64_t size_bytes = 32) {
    return ExternalRecord{Opcode::SRAM_RESIZE,
                          SramResizeOperands{label, size_bytes}};
}

ExternalRecord RenameRecord(uint64_t old_label, uint64_t new_label) {
    return ExternalRecord{Opcode::SRAM_RENAME,
                          SramRenameOperands{old_label, new_label}};
}

ExternalRecord LabelRecord(Opcode opcode, uint64_t label) {
    return ExternalRecord{opcode, SymbolOperands{label}};
}

ExternalRecord LsuLoadRecord() {
    LsuOperands operands;
    operands.hbm_address_bytes = 0x1000;
    operands.size_bytes = 16;
    operands.sram.kind = SramAddressKind::ABSOLUTE;
    operands.sram.absolute_address_bytes = 0x2000;
    return ExternalRecord{Opcode::LSU_LOAD, operands};
}

ExternalRecord DteSendRecord(
    uint64_t peer_core = 1, uint64_t fsm_id = 1,
    EndpointCompletion completion = EndpointCompletion::ASYNC,
    uint64_t token = 1, uint64_t length_bytes = 8) {
    ExternalRecord record;
    record.opcode = Opcode::DTE_SEND;
    DteSendOperands operands;
    operands.mode = DteSendMode::P2P;
    operands.source_space = EndpointSourceSpace::SRAM;
    operands.completion = completion;
    operands.token = token;
    operands.fsm_id = fsm_id;
    operands.length_bytes = length_bytes;
    operands.source.kind = SramAddressKind::ABSOLUTE;
    operands.source.absolute_address_bytes = 0x100;
    operands.peer_core = peer_core;
    record.operands = operands;
    return record;
}

ExternalRecord DteRecvRecord(
    uint64_t peer_core = 0, uint64_t fsm_id = 1,
    EndpointCompletion completion = EndpointCompletion::ASYNC,
    uint64_t token = 1, uint64_t length_bytes = 8) {
    DteRecvOperands operands;
    operands.mode = DteRecvMode::P2P;
    operands.completion = completion;
    operands.token = token;
    operands.fsm_id = fsm_id;
    operands.length_bytes = length_bytes;
    operands.destination.kind = SramAddressKind::ABSOLUTE;
    operands.destination.absolute_address_bytes = 0x200;
    operands.peer_core = peer_core;
    return ExternalRecord{Opcode::DTE_RECV, operands};
}

ExternalRecord DteScatterRecord() {
    ExternalRecord record = DteSendRecord(
        0, 1, EndpointCompletion::ASYNC, 31, 8);
    auto &operands = std::get<DteSendOperands>(record.operands);
    operands.mode = DteSendMode::SCATTER;
    operands.group_id = 1;
    operands.collective_id = 1;
    return record;
}

ExternalRecord DteGatherRecord() {
    ExternalRecord record = DteRecvRecord(
        0, 1, EndpointCompletion::ASYNC, 32, 8);
    auto &operands = std::get<DteRecvOperands>(record.operands);
    operands.mode = DteRecvMode::GATHER;
    operands.peer_core = 0;
    operands.expected_sources = 1;
    operands.group_id = 1;
    operands.collective_id = 1;
    return record;
}

ExternalRecord GroupSyncRecord(uint64_t group_id, uint64_t sequence) {
    return ExternalRecord{Opcode::GROUP_SYNC,
                          GroupSyncOperands{group_id, sequence}};
}

ExternalRecord EventSetRecord(uint64_t source, uint64_t destination,
                              uint64_t tag) {
    return ExternalRecord{Opcode::EVENT_SET,
                          EventSetOperands{source, destination, tag}};
}

ExternalRecord EventWaitRecord(uint64_t source, uint64_t destination,
                               uint64_t tag, uint64_t count) {
    return ExternalRecord{Opcode::EVENT_WAIT,
                          EventWaitOperands{source, destination, tag, count}};
}

ProgramArtifact Artifact(std::vector<ProgramCore> cores,
                         EmptyCoreAckPolicy policy,
                         std::vector<ProgramStartEvent> starts,
                         std::vector<uint64_t> terminals) {
    ProgramArtifact artifact;
    artifact.strings = {"p3c_input_b", "p3c_input_a", "p3c_output"};
    artifact.symbols = {
        {0, ProgramSymbolKind::SRAM_LABEL, 0, 0, 0},
        {1, ProgramSymbolKind::SRAM_LABEL, 0, 0, 0},
        {2, ProgramSymbolKind::SRAM_LABEL, 0, 0, 0},
    };
    artifact.cores = std::move(cores);
    for (const ProgramCore &core : artifact.cores)
        artifact.envelope.active_cores.push_back(core.core_id);
    artifact.envelope.start_events = std::move(starts);
    artifact.envelope.terminal_cores = terminals;
    artifact.envelope.expected_done_cores = terminals;
    artifact.envelope.empty_core_ack_policy = policy;
    if (policy == EmptyCoreAckPolicy::INCLUDE_EMPTY) {
        artifact.envelope.expected_ack_cores = artifact.envelope.active_cores;
    } else {
        for (const ProgramCore &core : artifact.cores)
            if (!core.records.empty())
                artifact.envelope.expected_ack_cores.push_back(core.core_id);
    }
    return artifact;
}

bool HelperCollectiveSendRole(CollOp op, std::size_t rank,
                              std::size_t root) {
    if (op == CollOp::P2P) return rank == 0;
    return op == CollOp::SCATTER || op == CollOp::BROADCAST
               ? rank == root
               : true;
}

bool HelperCollectiveReceiveRole(CollOp op, std::size_t rank,
                                 std::size_t root) {
    if (op == CollOp::P2P) return rank == root;
    return op == CollOp::GATHER || op == CollOp::REDUCE
               ? rank == root
               : true;
}

bool HelperCollectiveComputeRole(CollOp op, std::size_t rank,
                                 std::size_t root) {
    return op == CollOp::REDUCE
               ? rank == root
               : op == CollOp::REDUCESCATTER ||
                     op == CollOp::ALLREDUCE;
}

ProgramArtifact HelperCollectiveArtifact(CollTxKind tx, CollRxKind rx,
                                         std::size_t n) {
    ProgramArtifact artifact;
    ProgramCoreGroup group;
    group.group_id = 7;
    for (std::size_t rank = 0; rank < n; ++rank)
        group.members.push_back(rank);
    artifact.core_groups.push_back(std::move(group));

    const CollOp op = IsaV1CollectiveOp(tx, rx);
    const std::size_t root = n == 1 ? 0 : 1;
    for (std::size_t rank = 0; rank < n; ++rank) {
        ProgramCore core;
        core.core_id = rank;
        if (HelperCollectiveSendRole(op, rank, root)) {
            DteSendOperands send;
            send.mode = tx == CollTxKind::SCATTER
                            ? DteSendMode::SCATTER
                            : tx == CollTxKind::BROADCAST
                                  ? DteSendMode::BROADCAST
                                  : DteSendMode::P2P;
            send.completion = EndpointCompletion::ASYNC;
            send.fsm_id = 0x7100;
            send.token = 0x100 + rank * 4 + 1;
            send.length_bytes = 32;
            send.source.kind = SramAddressKind::ABSOLUTE;
            send.source.absolute_address_bytes =
                0x10000 + rank * 0x1000;
            send.group_id = 7;
            send.collective_id = 12;
            core.records.push_back({Opcode::DTE_SEND, send});
        }
        if (HelperCollectiveReceiveRole(op, rank, root)) {
            DteRecvOperands receive;
            receive.mode = rx == CollRxKind::GATHER
                               ? DteRecvMode::GATHER
                               : rx == CollRxKind::REDUCE
                                     ? DteRecvMode::REDUCE
                                     : DteRecvMode::P2P;
            receive.completion = EndpointCompletion::ASYNC;
            receive.fsm_id = 0x7100;
            receive.token = 0x100 + rank * 4 + 2;
            receive.length_bytes = 32;
            receive.destination.kind = SramAddressKind::ABSOLUTE;
            receive.destination.absolute_address_bytes =
                0x20000 + rank * 0x1000;
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
            receive.collective_id = 12;
            core.records.push_back({Opcode::DTE_RECV, receive});
        }
        if (HelperCollectiveComputeRole(op, rank, root)) {
            ReduceComputeOperands compute;
            compute.datatype = EndpointDataType::INT32;
            compute.reduce_op = ReduceOperator::SUM;
            compute.group_id = 7;
            compute.collective_id = 12;
            compute.root_rank = op == CollOp::REDUCE ? root : 0;
            compute.self_rank = rank;
            compute.element_count = 8;
            compute.source.kind = SramAddressKind::ABSOLUTE;
            compute.source.absolute_address_bytes =
                0x20000 + rank * 0x1000;
            compute.destination.kind = SramAddressKind::ABSOLUTE;
            compute.destination.absolute_address_bytes =
                0x30000 + rank * 0x1000;
            core.records.push_back({Opcode::REDUCE_COMPUTE, compute});
        }
        core.records.push_back(FenceRecord());
        artifact.cores.push_back(std::move(core));
        artifact.envelope.active_cores.push_back(rank);
        artifact.envelope.expected_ack_cores.push_back(rank);
    }
    artifact.envelope.terminal_cores = {n - 1};
    artifact.envelope.expected_done_cores = {n - 1};
    artifact.envelope.empty_core_ack_policy =
        EmptyCoreAckPolicy::INCLUDE_EMPTY;
    return artifact;
}

ProgramArtifact N1GatherArtifact() {
    ExternalRecord send = DteSendRecord(
        0, 0x6000, EndpointCompletion::ASYNC, 61, 8);
    auto &send_operands = std::get<DteSendOperands>(send.operands);
    send_operands.group_id = 1;
    send_operands.collective_id = 9;
    ExternalRecord receive = DteGatherRecord();
    auto &receive_operands =
        std::get<DteRecvOperands>(receive.operands);
    receive_operands.fsm_id = 0x6000;
    receive_operands.token = 62;
    receive_operands.expected_sources = 0;
    receive_operands.collective_id = 9;
    ProgramArtifact artifact = Artifact(
        {{0, {AllocRecord(3, SramLifetime::PERSISTENT, true), send,
              receive, FenceRecord()}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {0});
    artifact.strings.push_back("p6_collective_image_label");
    artifact.symbols.push_back(
        {3, ProgramSymbolKind::SRAM_LABEL, 0, 0, 0});
    artifact.core_groups = {{1, {0}}};
    return artifact;
}

bool SameMessage(const HostEnvelope &left, const HostEnvelope &right) {
    const Msg &a = left.msg;
    const Msg &b = right.msg;
    return left.dest_global_id == right.dest_global_id &&
           a.is_end_ == b.is_end_ && a.msg_type_ == b.msg_type_ &&
           a.seq_id_ == b.seq_id_ && a.des_ == b.des_ &&
           a.offset_ == b.offset_ && a.tag_id_ == b.tag_id_ &&
           a.source_ == b.source_ && a.length_ == b.length_ &&
           a.refill_ == b.refill_ && a.config_end_ == b.config_end_ &&
           a.roofline_packets_ == b.roofline_packets_ && a.data_ == b.data_;
}

bool SameMessages(const std::vector<HostEnvelope> &left,
                  const std::vector<HostEnvelope> &right) {
    if (left.size() != right.size()) return false;
    for (std::size_t i = 0; i < left.size(); ++i)
        if (!SameMessage(left[i], right[i])) return false;
    return true;
}

std::vector<HostEnvelope> ForCore(const std::vector<HostEnvelope> &messages,
                                  int core) {
    std::vector<HostEnvelope> result;
    for (const HostEnvelope &message : messages)
        if (message.dest_global_id == core)
            result.push_back(message);
    return result;
}

uint8_t PrimIdOf(const HostEnvelope &message) {
    return static_cast<uint8_t>(message.msg.data_.range(7, 0).to_uint());
}

std::vector<Collective_launch_v1_prim> DecodeCollectiveLaunches(
    const std::vector<HostEnvelope> &messages) {
    std::vector<Collective_launch_v1_prim> result;
    std::vector<sc_bv<128>> wire;
    for (const HostEnvelope &envelope : messages) {
        if (PrimIdOf(envelope) !=
            PrimIdValue(PrimId::COLLECTIVE_LAUNCH_V1)) {
            if (!wire.empty())
                throw std::logic_error(
                    "collective launch wire is interrupted");
            continue;
        }
        wire.push_back(envelope.msg.data_);
        if (!envelope.msg.config_end_) continue;
        Collective_launch_v1_prim prim;
        prim.deserialize(wire);
        if (prim.serialize() != wire)
            throw std::logic_error(
                "collective launch strict roundtrip changed wire");
        result.push_back(std::move(prim));
        wire.clear();
    }
    if (!wire.empty())
        throw std::logic_error("collective launch wire is truncated");
    return result;
}

Msg Ack(int source) { return Msg(ACK, HOST_ENDPOINT_ID, 0, source); }
Msg Done(int source) { return Msg(DONE, HOST_ENDPOINT_ID, source); }

void CheckSingleCoreFlow(Checks &checks) {
    ProgramArtifact artifact = Artifact(
        {{0, BoundMatmul()}}, EmptyCoreAckPolicy::INCLUDE_EMPTY,
        {{0, 17, 2}}, {0});
    config_helper_program helper(artifact);
    const auto config = helper.BuildConfigMessages();
    const auto start = helper.BuildStartMessages();
    const auto data = helper.BuildDataMessages();
    checks.Check(!config.empty() && config.front().dest_global_id == 0,
                 "single-core CONFIG destination");
    checks.Check(PrimIdOf(config.front()) == PrimIdValue(PrimId::RECV) &&
                     config.front().msg.data_.range(11, 8).to_uint() ==
                         RECV_WEIGHT,
                 "RECV_WEIGHT protocol prologue is first");
    checks.Check(config.size() >= 4 &&
                     PrimIdOf(config[1]) == PrimIdValue(PrimId::RECV) &&
                     config[1].msg.data_.range(11, 8).to_uint() == RECV_START &&
                     config[1].msg.data_.range(27, 12).to_uint() == 17 &&
                     config[1].msg.data_.range(35, 28).to_uint() == 2,
                 "source RECV_START is application-flow head");
    checks.Check(PrimIdOf(config.back()) == PrimIdValue(PrimId::SEND) &&
                     config.back().msg.data_.range(59, 56).to_uint() ==
                         SEND_DONE &&
                     config.back().msg.is_end_ && config.back().msg.refill_,
                 "terminal SEND_DONE is final CONFIG primitive");
    checks.Check(start.size() == 2 && start[0].msg.msg_type_ == S_DATA &&
                     start[0].msg.is_end_ && start[1].msg.is_end_ &&
                     start[0].msg.tag_id_ == 17 &&
                     start[0].msg.source_ == HOST_ENDPOINT_ID,
                 "start count emits exact completion-message count");
    checks.Check(data.size() == 1 && data[0].msg.msg_type_ == P_DATA &&
                     data[0].msg.is_end_ && data[0].msg.tag_id_ == 0 &&
                     data[0].msg.source_ == HOST_ENDPOINT_ID,
                 "P_DATA end preserves RECV_WEIGHT ACK phase");
    checks.Check(helper.expected_ack_cores() == std::set<int>{0} &&
                     helper.expected_done_cores() == std::set<int>{0},
                 "single-core expected ACK/DONE sets");
    checks.Check(helper.coreconfigs.size() == 1 &&
                     helper.coreconfigs[0].id == 0 &&
                     helper.coreconfigs[0].prim_copy == -1 &&
                     helper.coreconfigs[0].send_global_mem == -1 &&
                     helper.coreconfigs[0].loop == 1 &&
                     helper.coreconfigs[0].worklist.empty(),
                 "base CoreConfig is fully initialized at atomic commit");
    checks.Check(SameMessages(config, helper.BuildConfigMessages()) &&
                     SameMessages(start, helper.BuildStartMessages()) &&
                     SameMessages(data, helper.BuildDataMessages()),
                 "message builders are deterministic and pure");
}

void CheckEmptyCorePolicies(Checks &checks) {
    ProgramArtifact include = Artifact(
        {{0, BoundMatmul()}, {1, {}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {{0, 3, 1}}, {0});
    config_helper_program include_helper(include);
    const auto include_config = include_helper.BuildConfigMessages();
    checks.Check(!ForCore(include_config, 0).empty() &&
                     !ForCore(include_config, 1).empty(),
                 "INCLUDE_EMPTY emits CONFIG for every active core");
    checks.Check(include_helper.BuildDataMessages().size() == 2 &&
                     include_helper.expected_ack_cores() ==
                         std::set<int>({0, 1}),
                 "INCLUDE_EMPTY exact P_DATA/ACK set");

    ProgramArtifact exclude = Artifact(
        {{0, BoundMatmul()}, {1, {}}},
        EmptyCoreAckPolicy::EXCLUDE_EMPTY, {{0, 3, 1}}, {0});
    config_helper_program exclude_helper(exclude);
    const auto exclude_config = exclude_helper.BuildConfigMessages();
    checks.Check(!ForCore(exclude_config, 0).empty() &&
                     ForCore(exclude_config, 1).empty(),
                 "EXCLUDE_EMPTY omits empty-core CONFIG");
    checks.Check(exclude_helper.BuildDataMessages().size() == 1 &&
                     exclude_helper.expected_ack_cores() == std::set<int>{0},
                 "EXCLUDE_EMPTY exact P_DATA/ACK set");

    ProgramArtifact contradictory = Artifact(
        {{0, BoundMatmul()}, {1, {}}},
        EmptyCoreAckPolicy::EXCLUDE_EMPTY, {{1, 3, 1}}, {0});
    checks.Reject<ConfigHelperProgramError>(
        "excluded empty source", "cannot be a source", [&] {
            config_helper_program rejected(contradictory);
        });
    contradictory = Artifact(
        {{0, BoundMatmul()}, {1, {}}},
        EmptyCoreAckPolicy::EXCLUDE_EMPTY, {{0, 3, 1}}, {1});
    checks.Reject<ConfigHelperProgramError>(
        "excluded empty terminal", "cannot be a source or terminal", [&] {
            config_helper_program rejected(contradictory);
        });
}

ProgramArtifact RelocationArtifact() {
    ProgramArtifact artifact;
    artifact.strings = {"absolute", "region", "label_a", "label_b"};
    artifact.symbols = {
        {0, ProgramSymbolKind::ABSOLUTE_ADDRESS, 0, 100, 64},
        {1, ProgramSymbolKind::SRAM_REGION, 0, 20, 64},
        {2, ProgramSymbolKind::SRAM_LABEL, 0, 0, 0},
        {3, ProgramSymbolKind::SRAM_LABEL, 0, 0, 0},
    };
    LsuOperands load;
    load.size_bytes = 8;
    load.sram.kind = SramAddressKind::ABSOLUTE;
    LsuOperands store = load;
    ExternalRecord load_record{Opcode::LSU_LOAD, load};
    ExternalRecord store_record{Opcode::LSU_STORE, store};
    ExternalRecord clear_record{Opcode::SRAM_CLEAR, SymbolOperands{2}};
    SramAllocOperands alloc;
    alloc.region_name_string_index = 0;
    alloc.label_symbol_index = 2;
    alloc.size_bytes = 8;
    alloc.alignment_bytes = 8;
    ExternalRecord alloc_record{Opcode::SRAM_ALLOC, alloc};
    ExternalRecord rename_record{Opcode::SRAM_RENAME,
                                 SramRenameOperands{2, 3}};
    artifact.cores = {{0, {BindRecord(1, 2, 3), MatmulRecord(),
                           load_record, store_record, clear_record,
                           alloc_record, rename_record}}};
    auto relocation = [](uint64_t instruction, SemanticOperandId operand,
                         SemanticRelocationKind kind, uint64_t symbol,
                         int64_t addend) {
        return SemanticRelocation{0, instruction,
                                  static_cast<uint16_t>(operand), kind,
                                  symbol, addend};
    };
    artifact.relocations = {
        relocation(0, SemanticOperandId::SRAM_BIND_INPUT_0,
                   SemanticRelocationKind::SRAM_LABEL, 3, 0),
        relocation(0, SemanticOperandId::SRAM_BIND_OUTPUT,
                   SemanticRelocationKind::SRAM_LABEL, 2, 0),
        relocation(1, SemanticOperandId::COMPUTE_INPUT_ADDRESS,
                   SemanticRelocationKind::ABSOLUTE_ADDRESS, 0, 1),
        relocation(1, SemanticOperandId::COMPUTE_DATA_ADDRESS,
                   SemanticRelocationKind::ABSOLUTE_ADDRESS, 0, 2),
        relocation(1, SemanticOperandId::COMPUTE_OUTPUT_ADDRESS,
                   SemanticRelocationKind::ABSOLUTE_ADDRESS, 0, 3),
        relocation(2, SemanticOperandId::DESTINATION_ADDRESS,
                   SemanticRelocationKind::SRAM_REGION, 1, 5),
        relocation(2, SemanticOperandId::HBM_ADDRESS,
                   SemanticRelocationKind::ABSOLUTE_ADDRESS, 0, 6),
        relocation(3, SemanticOperandId::SOURCE_ADDRESS,
                   SemanticRelocationKind::SRAM_REGION, 1, 7),
        relocation(3, SemanticOperandId::HBM_ADDRESS,
                   SemanticRelocationKind::ABSOLUTE_ADDRESS, 0, 8),
        relocation(4, SemanticOperandId::SYMBOL,
                   SemanticRelocationKind::SRAM_LABEL, 3, 0),
        relocation(5, SemanticOperandId::REGION_NAME,
                   SemanticRelocationKind::SRAM_REGION, 1, 0),
        relocation(5, SemanticOperandId::LABEL_SYMBOL,
                   SemanticRelocationKind::SRAM_LABEL, 3, 0),
        relocation(6, SemanticOperandId::OLD_SYMBOL,
                   SemanticRelocationKind::SRAM_LABEL, 3, 0),
        relocation(6, SemanticOperandId::NEW_SYMBOL,
                   SemanticRelocationKind::SRAM_LABEL, 2, 0),
    };
    artifact.envelope.active_cores = {0};
    artifact.envelope.expected_ack_cores = {0};
    artifact.envelope.empty_core_ack_policy =
        EmptyCoreAckPolicy::INCLUDE_EMPTY;
    return artifact;
}

void CheckRelocations(Checks &checks) {
    ProgramArtifact artifact = RelocationArtifact();
    ApplyProgramRelocations(artifact);
    const auto &bind =
        std::get<SramBindOperands>(artifact.cores[0].records[0].operands);
    checks.Check(bind.input_symbol_indices[0] == 3 &&
                     bind.output_symbol_index == 2,
                 "SRAM_BIND active input/output relocations");
    const auto &compute =
        std::get<ComputeOperands>(artifact.cores[0].records[1].operands);
    checks.Check(compute.input_offset_bytes == 101 &&
                     compute.data_offset_bytes == 102 &&
                     compute.output_offset_bytes == 103,
                 "operand IDs 1/2/3 relocate compute offsets");
    const auto &load =
        std::get<LsuOperands>(artifact.cores[0].records[2].operands);
    checks.Check(load.sram.kind == SramAddressKind::REGION &&
                     load.sram.region_symbol_index == 1 &&
                     load.sram.region_offset_bytes == 5 &&
                     load.hbm_address_bytes == 106,
                 "operand IDs 5/6 relocate LSU destination/HBM");
    const auto &store =
        std::get<LsuOperands>(artifact.cores[0].records[3].operands);
    checks.Check(store.sram.kind == SramAddressKind::REGION &&
                     store.sram.region_offset_bytes == 7 &&
                     store.hbm_address_bytes == 108,
                 "operand IDs 4/6 relocate LSU source/HBM");
    checks.Check(std::get<SymbolOperands>(
                         artifact.cores[0].records[4].operands).symbol_index ==
                         3,
                 "operand ID 7 relocates SYMBOL");
    const auto &alloc = std::get<SramAllocOperands>(
        artifact.cores[0].records[5].operands);
    checks.Check(alloc.region_name_string_index == 1 &&
                     alloc.label_symbol_index == 3,
                 "operand IDs 8/9 relocate region name and label symbol");
    const auto &rename = std::get<SramRenameOperands>(
        artifact.cores[0].records[6].operands);
    checks.Check(rename.old_symbol_index == 3 &&
                     rename.new_symbol_index == 2,
                 "operand IDs 10/11 relocate old/new symbols");

    ProgramArtifact endpoint_hbm = Artifact(
        {{0, {DteSendRecord(1, 7, EndpointCompletion::SYNC, 0)}},
         {1, {DteRecvRecord(0, 7, EndpointCompletion::SYNC, 0)}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {1});
    endpoint_hbm.strings = {"endpoint_hbm"};
    endpoint_hbm.symbols = {
        {0, ProgramSymbolKind::ABSOLUTE_ADDRESS, 0, 0x4000, 64}};
    auto &hbm_send = std::get<DteSendOperands>(
        endpoint_hbm.cores[0].records[0].operands);
    hbm_send.source_space = EndpointSourceSpace::HBM;
    endpoint_hbm.relocations = {{
        0, 0, static_cast<uint16_t>(SemanticOperandId::HBM_ADDRESS),
        SemanticRelocationKind::ABSOLUTE_ADDRESS, 0, 32}};
    ApplyProgramRelocations(endpoint_hbm);
    checks.Check(
        hbm_send.source.kind == SramAddressKind::ABSOLUTE &&
            hbm_send.source.absolute_address_bytes == 0x4020 &&
            hbm_send.source.region_symbol_index == 0 &&
            hbm_send.source.region_offset_bytes == 0,
        "DTE_SEND HBM_ADDRESS relocation writes canonical absolute source");

    ProgramArtifact endpoint_sram = Artifact(
        {{0, {DteSendRecord(1, 8, EndpointCompletion::SYNC, 0)}},
         {1, {DteRecvRecord(0, 8, EndpointCompletion::SYNC, 0)}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {1});
    endpoint_sram.strings = {"endpoint_sram"};
    endpoint_sram.symbols = {
        {0, ProgramSymbolKind::SRAM_REGION, 0, 20, 64}};
    endpoint_sram.relocations = {{
        0, 0, static_cast<uint16_t>(SemanticOperandId::SOURCE_ADDRESS),
        SemanticRelocationKind::SRAM_REGION, 0, 5}};
    ApplyProgramRelocations(endpoint_sram);
    const auto &sram_send = std::get<DteSendOperands>(
        endpoint_sram.cores[0].records[0].operands);
    checks.Check(
        sram_send.source.kind == SramAddressKind::REGION &&
            sram_send.source.absolute_address_bytes == 0 &&
            sram_send.source.region_symbol_index == 0 &&
            sram_send.source.region_offset_bytes == 5,
        "DTE_SEND SRAM relocation keeps addend as region-local offset");

    ProgramArtifact negative_region = endpoint_sram;
    negative_region.relocations[0].addend = -1;
    checks.Reject<ProgramFormatError>(
        "SRAM_REGION relocation rejects negative region-local offset",
        "addend cannot be negative", [&] {
            ApplyProgramRelocations(negative_region);
        });

    ProgramArtifact absolute_overflow = endpoint_hbm;
    absolute_overflow.symbols[0].value =
        std::numeric_limits<uint64_t>::max();
    absolute_overflow.relocations[0].addend = 1;
    checks.Reject<ProgramFormatError>(
        "absolute relocation rejects u64 overflow", "overflows u64", [&] {
            ApplyProgramRelocations(absolute_overflow);
        });

    ProgramArtifact absolute_underflow = endpoint_hbm;
    absolute_underflow.symbols[0].value = 0;
    absolute_underflow.relocations[0].addend = -1;
    checks.Reject<ProgramFormatError>(
        "absolute relocation rejects u64 underflow", "underflows u64", [&] {
            ApplyProgramRelocations(absolute_underflow);
        });

    ProgramArtifact invalid_hbm = endpoint_hbm;
    invalid_hbm.symbols[0].kind = ProgramSymbolKind::SRAM_REGION;
    invalid_hbm.relocations[0].kind =
        SemanticRelocationKind::SRAM_REGION;
    checks.Reject<ProgramFormatError>(
        "DTE_SEND HBM_ADDRESS rejects SRAM_REGION relocation",
        "requires ABSOLUTE_ADDRESS kind", [&] {
            ApplyProgramRelocations(invalid_hbm);
        });

    ProgramArtifact invalid = RelocationArtifact();
    invalid.relocations = {{
        0, 5, static_cast<uint16_t>(SemanticOperandId::REGION_NAME),
        SemanticRelocationKind::SRAM_REGION, 1, 1}};
    checks.Reject<ConfigHelperProgramError>(
        "REGION_NAME addend rejected deterministically", "addend zero", [&] {
            ApplyProgramRelocations(invalid);
        });
}

void CheckSramBindLifecycle(Checks &checks) {
    ProgramArtifact valid = Artifact(
        {{0, {BindRecord(), LsuLoadRecord(), FenceRecord(), MatmulRecord(),
              BindRecord(2), ResidualRecord()}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {0});
    valid.strings = {"p3c_order_zeta", "p3c_order_alpha",
                     u8"p3c_order_标签"};
    const std::size_t labels_before = g_addr_label_table.table.size();
    config_helper_program helper(valid);
    checks.Check(!helper.BuildConfigMessages().empty(),
                 "SRAM_BIND survives MEM/SYNC and two computes each consume one bind");
    const std::vector<std::string> appended(
        g_addr_label_table.table.begin() + labels_before,
        g_addr_label_table.table.end());
    checks.Check(appended ==
                     std::vector<std::string>({"p3c_order_alpha",
                                               "p3c_order_zeta",
                                               u8"p3c_order_标签"}),
                 "SRAM_BIND labels intern in deterministic UTF-8 byte order");

    ProgramArtifact missing = Artifact(
        {{0, {MatmulRecord()}}}, EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {0});
    checks.Reject<ConfigHelperProgramError>(
        "compute without SRAM_BIND", "missing a preceding", [&] {
            config_helper_program rejected(missing);
        });

    ProgramArtifact consecutive = Artifact(
        {{0, {BindRecord(), MatmulRecord(), MatmulRecord(10)}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {0});
    checks.Reject<ConfigHelperProgramError>(
        "one bind cannot feed consecutive computes", "missing a preceding",
        [&] { config_helper_program rejected(consecutive); });

    ProgramArtifact double_bind = Artifact(
        {{0, {BindRecord(), FenceRecord(), BindRecord(), MatmulRecord()}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {0});
    checks.Reject<ConfigHelperProgramError>(
        "second bind before compute", "two SRAM_BIND", [&] {
            config_helper_program rejected(double_bind);
        });

    ProgramArtifact dangling = Artifact(
        {{0, {BindRecord(), LsuLoadRecord(), FenceRecord()}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {0});
    checks.Reject<ConfigHelperProgramError>(
        "dangling bind at core end", "dangling SRAM_BIND", [&] {
            config_helper_program rejected(dangling);
        });

    ProgramArtifact wrong_count = Artifact(
        {{0, {BindRecord(2), MatmulRecord()}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {0});
    wrong_count.strings = {"p3c_rollback_zeta", "p3c_rollback_alpha",
                           "p3c_rollback_output"};
    const auto label_table_before_failure = g_addr_label_table.table;
    checks.Reject<ConfigHelperProgramError>(
        "bind input_count mismatch", "does not match", [&] {
            helper.LoadProgram(EncodeProgramArtifact(wrong_count));
        });
    checks.Check(g_addr_label_table.table == label_table_before_failure,
                 "pre-intern validation failure does not pollute label table");

    ProgramArtifact post_intern_failure = Artifact(
        {{0, BoundMatmul()}}, EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {0});
    post_intern_failure.strings = {"p3c_postintern_input",
                                   "p3c_postintern_unused", UNSET_LABEL};
    const auto table_before_post_intern_failure = g_addr_label_table.table;
    checks.Reject<std::invalid_argument>(
        "post-intern Prim wire validation rollback", "output label is unset",
        [&] {
            helper.LoadProgram(EncodeProgramArtifact(post_intern_failure));
        });
    checks.Check(g_addr_label_table.table ==
                     table_before_post_intern_failure,
                 "failed LoadProgram rolls back labels interned for preflight");
}

ProgramArtifact MemoryArtifact(std::vector<ExternalRecord> records) {
    ProgramArtifact artifact = Artifact(
        {{0, std::move(records)}}, EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {0});
    artifact.strings = {"scratch", "p4d_label_a", "p4d_label_b",
                        "p4d_persistent"};
    artifact.symbols = {
        {1, ProgramSymbolKind::SRAM_LABEL, 0, 0, 0},
        {2, ProgramSymbolKind::SRAM_LABEL, 0, 0, 0},
        {3, ProgramSymbolKind::SRAM_LABEL, 0, 0, 0},
    };
    return artifact;
}

void CheckSramLifecyclePrograms(Checks &checks) {
    ProgramArtifact valid = MemoryArtifact({
        AllocRecord(0, SramLifetime::TASK, true), ResizeRecord(0),
        RenameRecord(0, 1), LabelRecord(Opcode::SRAM_CLEAR, 1),
        AllocRecord(0, SramLifetime::TASK, true),
        LabelRecord(Opcode::SRAM_FREE, 0)});
    config_helper_program helper(valid);
    const auto committed_messages = helper.BuildConfigMessages();
    std::size_t lifecycle_segments = 0;
    for (const HostEnvelope &message : committed_messages)
        lifecycle_segments +=
            PrimIdOf(message) == PrimIdValue(PrimId::SRAM_LIFECYCLE);
    checks.Check(
        lifecycle_segments == 24,
        "all five lifecycle variants lower to six strict 4-segment Prims");

    ProgramArtifact persistent = MemoryArtifact(
        {AllocRecord(2, SramLifetime::PERSISTENT, true)});
    config_helper_program persistent_helper(persistent);
    checks.Check(!persistent_helper.BuildConfigMessages().empty(),
                 "a declared persistent output may remain live at program end");
    const auto committed_labels = g_addr_label_table.table;

    auto RejectReload = [&](std::string name, std::string_view diagnostic,
                            std::vector<ExternalRecord> records) {
        checks.Reject<ConfigHelperProgramError>(
            std::move(name), diagnostic, [&] {
                helper.LoadProgram(
                    EncodeProgramArtifact(MemoryArtifact(std::move(records))));
            });
        checks.Check(SameMessages(committed_messages,
                                  helper.BuildConfigMessages()) &&
                         g_addr_label_table.table == committed_labels,
                     "failed lifecycle reload preserves helper and label table");
    };

    RejectReload("duplicate lifecycle ALLOC", "duplicate ALLOC",
                 {AllocRecord(0, SramLifetime::TASK, true),
                  AllocRecord(0, SramLifetime::TASK, true)});
    RejectReload("lifecycle RESIZE missing label", "unknown label",
                 {ResizeRecord(0)});
    RejectReload("lifecycle FREE missing label", "unknown label",
                 {LabelRecord(Opcode::SRAM_FREE, 0)});
    RejectReload("lifecycle CLEAR missing label", "unknown label",
                 {LabelRecord(Opcode::SRAM_CLEAR, 0)});
    RejectReload("lifecycle RENAME missing label", "unknown label",
                 {RenameRecord(0, 1)});
    RejectReload("lifecycle RENAME collision", "already exists",
                 {AllocRecord(0, SramLifetime::TASK, true),
                  AllocRecord(1, SramLifetime::TASK, true),
                  RenameRecord(0, 1)});
    RejectReload("dangling task allocation", "dangling non-persistent",
                 {AllocRecord(0, SramLifetime::TASK, true)});
    RejectReload("CLEAR non-spillable allocation", "spillable TASK",
                 {AllocRecord(0, SramLifetime::TASK, false),
                  LabelRecord(Opcode::SRAM_CLEAR, 0)});
}

void CheckDteTokenPrograms(Checks &checks) {
    ProgramArtifact valid = MemoryArtifact({
        DteIssueRecord(7), LsuLoadRecord(),
        DteControlRecord(Opcode::DTE_WAIT, 7), DteIssueRecord(8),
        DteIssueRecord(9), FenceRecord(), DteIssueRecord(10),
        DteControlRecord(Opcode::DTE_CANCEL, 10)});
    config_helper_program helper(valid);
    const auto committed_messages = helper.BuildConfigMessages();
    const auto committed_labels = g_addr_label_table.table;
    checks.Check(!committed_messages.empty(),
                 "DTE WAIT/CANCEL release tokens and FENCE drains all tokens");

    auto RejectReload = [&](std::string name, std::string_view diagnostic,
                            std::vector<ExternalRecord> records) {
        checks.Reject<ConfigHelperProgramError>(
            std::move(name), diagnostic, [&] {
                helper.LoadProgram(
                    EncodeProgramArtifact(MemoryArtifact(std::move(records))));
            });
        checks.Check(SameMessages(committed_messages,
                                  helper.BuildConfigMessages()) &&
                         g_addr_label_table.table == committed_labels,
                     "failed DTE reload preserves helper and label table");
    };

    RejectReload("duplicate outstanding DTE token", "reuses outstanding",
                 {DteIssueRecord(7), DteIssueRecord(7),
                  DteControlRecord(Opcode::DTE_WAIT, 7)});
    RejectReload("DTE_WAIT unknown token", "unknown or completed",
                 {DteControlRecord(Opcode::DTE_WAIT, 7)});
    RejectReload("DTE_CANCEL unknown token", "unknown or completed",
                 {DteControlRecord(Opcode::DTE_CANCEL, 7)});
    RejectReload("DTE_WAIT repeated after release", "unknown or completed",
                 {DteIssueRecord(7), DteControlRecord(Opcode::DTE_WAIT, 7),
                  DteControlRecord(Opcode::DTE_WAIT, 7)});
    RejectReload("DTE_WAIT after FENCE release", "unknown or completed",
                 {DteIssueRecord(7), FenceRecord(),
                  DteControlRecord(Opcode::DTE_WAIT, 7)});
    RejectReload("dangling DTE token at program end", "dangling DTE token",
                 {DteIssueRecord(7)});
}

void CheckDteEndpointPrograms(Checks &checks) {
    ProgramArtifact ordered = Artifact(
        {{0,
          {DteSendRecord(1, 1, EndpointCompletion::SYNC, 0),
           DteRecvRecord(1, 0x10001U, EndpointCompletion::SYNC, 0)}},
         {1,
          {DteRecvRecord(0, 1, EndpointCompletion::SYNC, 0),
           DteSendRecord(0, 0x10001U, EndpointCompletion::SYNC, 0)}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {1});
    config_helper_program helper(ordered);
    const auto committed_messages = helper.BuildConfigMessages();
    const auto committed_registry = helper.core_group_registry();
    const auto committed_ack = helper.expected_ack_cores();
    const auto committed_done = helper.expected_done_cores();
    std::size_t send_segments = 0;
    std::size_t recv_segments = 0;
    for (const HostEnvelope &message : committed_messages) {
        send_segments +=
            PrimIdOf(message) == PrimIdValue(PrimId::DTE_SEND_ENDPOINT);
        recv_segments +=
            PrimIdOf(message) == PrimIdValue(PrimId::DTE_RECV_ENDPOINT);
    }
    checks.Check(
        send_segments == 10 && recv_segments == 10,
        "send-first/recv-first P2P pairs preserve full-u32 fsm 0x1 and 0x10001");

    ProgramArtifact cross_die = Artifact(
        {{0, {DteSendRecord(CORES_PER_DIE, 3,
                            EndpointCompletion::SYNC, 0)}},
         {static_cast<uint64_t>(CORES_PER_DIE),
          {DteRecvRecord(0, 3, EndpointCompletion::SYNC, 0)}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {},
        {static_cast<uint64_t>(CORES_PER_DIE)});
    config_helper_program cross_die_helper(cross_die);
    checks.Check(!cross_die_helper.BuildConfigMessages().empty(),
                 "P2P endpoint pair may cross dies");

    ProgramArtifact async_control = Artifact(
        {{0,
          {DteSendRecord(1, 10, EndpointCompletion::ASYNC, 7),
           DteControlRecord(Opcode::DTE_WAIT, 7), DteIssueRecord(8),
           DteSendRecord(1, 11, EndpointCompletion::ASYNC, 7),
           FenceRecord(),
           DteSendRecord(1, 12, EndpointCompletion::ASYNC, 11),
           DteControlRecord(Opcode::DTE_CANCEL, 11)}},
         {1,
          {DteRecvRecord(0, 10, EndpointCompletion::ASYNC, 17),
           DteControlRecord(Opcode::DTE_WAIT, 17),
           DteRecvRecord(0, 11, EndpointCompletion::ASYNC, 17),
           FenceRecord(),
           DteRecvRecord(0, 12, EndpointCompletion::ASYNC, 21),
           DteControlRecord(Opcode::DTE_WAIT, 21)}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {1});
    config_helper_program async_helper(async_control);
    checks.Check(!async_helper.BuildConfigMessages().empty(),
                 "endpoint WAIT permits token reuse; FENCE drains RX while CANCEL remains TX-only");

    ProgramArtifact region_pair = Artifact(
        {{0, {DteSendRecord(1, 4, EndpointCompletion::SYNC, 0)}},
         {1, {DteRecvRecord(0, 4, EndpointCompletion::SYNC, 0)}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {1});
    region_pair.strings = {"p5c2_region_zeta", "p5c2_region_alpha"};
    region_pair.symbols = {
        {0, ProgramSymbolKind::SRAM_REGION, 0, 0, 64},
        {1, ProgramSymbolKind::SRAM_REGION, 0, 0, 64},
    };
    auto &send_region = std::get<DteSendOperands>(
        region_pair.cores[0].records[0].operands);
    send_region.source.kind = SramAddressKind::REGION;
    send_region.source.absolute_address_bytes = 0;
    send_region.source.region_symbol_index = 0;
    send_region.source.region_offset_bytes = 3;
    auto &recv_region = std::get<DteRecvOperands>(
        region_pair.cores[1].records[0].operands);
    recv_region.destination.kind = SramAddressKind::REGION;
    recv_region.destination.absolute_address_bytes = 0;
    recv_region.destination.region_symbol_index = 1;
    recv_region.destination.region_offset_bytes = 5;
    const std::size_t labels_before_region = g_addr_label_table.table.size();
    config_helper_program region_helper(region_pair);
    const std::vector<std::string> appended_regions(
        g_addr_label_table.table.begin() + labels_before_region,
        g_addr_label_table.table.end());
    checks.Check(
        !region_helper.BuildConfigMessages().empty() &&
            appended_regions ==
            std::vector<std::string>({"p5c2_region_alpha",
                                      "p5c2_region_zeta"}),
        "endpoint region symbols late-intern in deterministic UTF-8 byte order");
    const auto stable_labels = g_addr_label_table.table;

    auto CheckCommittedState = [&] {
        checks.Check(SameMessages(committed_messages,
                                  helper.BuildConfigMessages()) &&
                         g_addr_label_table.table == stable_labels &&
                         helper.core_group_registry() == committed_registry &&
                         helper.expected_ack_cores() == committed_ack &&
                         helper.expected_done_cores() == committed_done &&
                         helper.coreconfigs.size() == 2 &&
                         helper.coreconfigs[0].id == 0 &&
                         helper.coreconfigs[1].id == 1,
                     "failed endpoint reload preserves helper and preexisting labels");
    };

    ProgramArtifact rx_cancel = Artifact(
        {{0,
          {DteSendRecord(1, 13, EndpointCompletion::ASYNC, 31),
           DteControlRecord(Opcode::DTE_WAIT, 31)}},
         {1,
          {DteRecvRecord(0, 13, EndpointCompletion::ASYNC, 41),
           DteControlRecord(Opcode::DTE_CANCEL, 41)}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {1});
    checks.Reject<ConfigHelperProgramError>(
        "endpoint RX CANCEL is rejected transactionally",
        "DTE_CANCEL does not support a DTE_RECV token", [&] {
            helper.LoadProgram(EncodeProgramArtifact(rx_cancel));
        });
    CheckCommittedState();

    ProgramArtifact self_send = Artifact(
        {{0, {DteSendRecord(0, 1, EndpointCompletion::SYNC, 0)}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {0});
    checks.Reject<ConfigHelperProgramError>(
        "DTE_SEND P2P self peer is rejected before pairing",
        "DTE_SEND P2P self peer is forbidden", [&] {
            helper.LoadProgram(EncodeProgramArtifact(self_send));
        });
    CheckCommittedState();

    ProgramArtifact self_recv = Artifact(
        {{0, {DteRecvRecord(0, 1, EndpointCompletion::SYNC, 0)}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {0});
    checks.Reject<ConfigHelperProgramError>(
        "DTE_RECV P2P self peer is rejected before pairing",
        "DTE_RECV P2P self peer is forbidden", [&] {
            helper.LoadProgram(EncodeProgramArtifact(self_recv));
        });
    CheckCommittedState();

    ProgramArtifact inactive = Artifact(
        {{0, {DteSendRecord(2, 1, EndpointCompletion::SYNC, 0)}},
         {1, {}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {0});
    checks.Reject<ConfigHelperProgramError>(
        "P2P peer must be active", "not an active program core", [&] {
            helper.LoadProgram(EncodeProgramArtifact(inactive));
        });
    CheckCommittedState();

    ProgramArtifact inactive_source = Artifact(
        {{0, {}},
         {1, {DteRecvRecord(2, 1, EndpointCompletion::SYNC, 0)}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {1});
    checks.Reject<ConfigHelperProgramError>(
        "P2P receive source must be active", "DTE_RECV peer_core is not",
        [&] {
            helper.LoadProgram(EncodeProgramArtifact(inactive_source));
        });
    CheckCommittedState();

    ProgramArtifact missing_recv = Artifact(
        {{0, {DteSendRecord(1, 1, EndpointCompletion::SYNC, 0)}},
         {1, {}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {0});
    checks.Reject<ConfigHelperProgramError>(
        "P2P send requires receive", "no matching DTE_RECV", [&] {
            helper.LoadProgram(EncodeProgramArtifact(missing_recv));
        });
    CheckCommittedState();

    ProgramArtifact missing_send = Artifact(
        {{0, {}},
         {1, {DteRecvRecord(0, 1, EndpointCompletion::SYNC, 0)}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {1});
    checks.Reject<ConfigHelperProgramError>(
        "P2P receive requires send", "no matching DTE_SEND", [&] {
            helper.LoadProgram(EncodeProgramArtifact(missing_send));
        });
    CheckCommittedState();

    ProgramArtifact duplicate_send = ordered;
    duplicate_send.cores[0].records.insert(
        duplicate_send.cores[0].records.begin() + 1,
        DteSendRecord(1, 1, EndpointCompletion::SYNC, 0));
    checks.Reject<ConfigHelperProgramError>(
        "DTE_SEND sync fsm reuse is rejected", "reuses DTE_SEND fsm_id", [&] {
            helper.LoadProgram(EncodeProgramArtifact(duplicate_send));
        });
    CheckCommittedState();

    ProgramArtifact tx_wait_reuse = Artifact(
        {{0,
          {DteSendRecord(1, 20, EndpointCompletion::ASYNC, 7),
           DteControlRecord(Opcode::DTE_WAIT, 7),
           DteSendRecord(1, 20, EndpointCompletion::ASYNC, 8)}},
         {1,
          {DteRecvRecord(0, 20, EndpointCompletion::SYNC, 0),
           DteRecvRecord(0, 20, EndpointCompletion::SYNC, 0)}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {1});
    checks.Reject<ConfigHelperProgramError>(
        "DTE_SEND async fsm reuse after WAIT is rejected",
        "reuses DTE_SEND fsm_id", [&] {
            helper.LoadProgram(EncodeProgramArtifact(tx_wait_reuse));
        });
    CheckCommittedState();

    ProgramArtifact tx_fence_reuse = Artifact(
        {{0,
          {DteSendRecord(1, 21, EndpointCompletion::ASYNC, 7),
           FenceRecord(),
           DteSendRecord(1, 21, EndpointCompletion::ASYNC, 8)}},
         {1,
          {DteRecvRecord(0, 21, EndpointCompletion::SYNC, 0),
           DteRecvRecord(0, 21, EndpointCompletion::SYNC, 0)}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {1});
    checks.Reject<ConfigHelperProgramError>(
        "DTE_SEND async fsm reuse after FENCE is rejected",
        "reuses DTE_SEND fsm_id", [&] {
            helper.LoadProgram(EncodeProgramArtifact(tx_fence_reuse));
        });
    CheckCommittedState();

    ProgramArtifact mismatched_length = ordered;
    std::get<DteRecvOperands>(
        mismatched_length.cores[1].records[0].operands).length_bytes = 9;
    checks.Reject<ConfigHelperProgramError>(
        "P2P length mismatch", "length mismatch", [&] {
            helper.LoadProgram(EncodeProgramArtifact(mismatched_length));
        });
    CheckCommittedState();

    ProgramArtifact mismatched_fsm = ordered;
    std::get<DteRecvOperands>(
        mismatched_fsm.cores[1].records[0].operands).fsm_id = 2;
    checks.Reject<ConfigHelperProgramError>(
        "P2P full-u32 fsm mismatch", "no matching DTE_RECV", [&] {
            helper.LoadProgram(EncodeProgramArtifact(mismatched_fsm));
        });
    CheckCommittedState();

    ProgramArtifact invalid_datatype = ordered;
    std::get<DteRecvOperands>(
        invalid_datatype.cores[1].records[0].operands).datatype =
        EndpointDataType::INT32;
    checks.Reject<ProgramFormatError>(
        "P2P datatype mismatch rejected canonically",
        "core 1 record 0: DTE_RECV P2P datatype must be UINT8",
        [&] {
            helper.LoadProgram(EncodeProgramArtifact(invalid_datatype));
        });
    CheckCommittedState();

    ProgramArtifact hbm_source = ordered;
    auto &hbm_send = std::get<DteSendOperands>(
        hbm_source.cores[0].records[0].operands);
    hbm_send.source_space = EndpointSourceSpace::HBM;
    hbm_send.source.kind = SramAddressKind::ABSOLUTE;
    hbm_send.source.absolute_address_bytes = UINT64_MAX - 7;
    config_helper_program hbm_helper(hbm_source);
    checks.Check(
        !hbm_helper.BuildConfigMessages().empty(),
        "HBM endpoint source loads after C3b execution support");
    CheckCommittedState();

    ProgramArtifact token_collision = Artifact(
        {{0,
          {DteIssueRecord(7),
           DteSendRecord(1, 10, EndpointCompletion::ASYNC, 7)}},
         {1, {DteRecvRecord(0, 10, EndpointCompletion::SYNC, 0)}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {1});
    checks.Reject<ConfigHelperProgramError>(
        "local and endpoint DTE token collision", "outstanding DTE token",
        [&] {
            helper.LoadProgram(EncodeProgramArtifact(token_collision));
        });
    CheckCommittedState();

    ProgramArtifact fsm_collision = Artifact(
        {{0,
          {DteSendRecord(1, 10, EndpointCompletion::ASYNC, 7),
           DteSendRecord(2, 10, EndpointCompletion::ASYNC, 8)}},
         {1, {DteRecvRecord(0, 10, EndpointCompletion::SYNC, 0)}},
         {2, {DteRecvRecord(0, 10, EndpointCompletion::SYNC, 0)}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {2});
    checks.Reject<ConfigHelperProgramError>(
        "endpoint fsm reuse before completion", "reuses DTE_SEND fsm_id",
        [&] {
            helper.LoadProgram(EncodeProgramArtifact(fsm_collision));
        });
    CheckCommittedState();

    ProgramArtifact dangling_endpoint = Artifact(
        {{0, {DteSendRecord(1, 10, EndpointCompletion::ASYNC, 7)}},
         {1, {DteRecvRecord(0, 10, EndpointCompletion::SYNC, 0)}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {1});
    checks.Reject<ConfigHelperProgramError>(
        "dangling endpoint token", "dangling DTE token", [&] {
            helper.LoadProgram(EncodeProgramArtifact(dangling_endpoint));
        });
    CheckCommittedState();

    ProgramArtifact scatter = Artifact(
        {{0, {DteScatterRecord()}}}, EmptyCoreAckPolicy::INCLUDE_EMPTY, {},
        {0});
    scatter.core_groups = {{1, {0}}};
    checks.Reject<RecordLoweringError>(
        "incomplete DTE SCATTER graph is rejected", "missing SEND or RECEIVE", [&] {
            helper.LoadProgram(EncodeProgramArtifact(scatter));
        });
    CheckCommittedState();

    ProgramArtifact gather = Artifact(
        {{0, {DteGatherRecord()}}}, EmptyCoreAckPolicy::INCLUDE_EMPTY, {},
        {0});
    gather.core_groups = {{1, {0}}};
    checks.Reject<RecordLoweringError>(
        "incomplete DTE GATHER graph is rejected", "missing SEND or RECEIVE", [&] {
            helper.LoadProgram(EncodeProgramArtifact(gather));
        });
    CheckCommittedState();
}

void CheckCollectiveProgramImages(Checks &checks) {
    using Cell = std::pair<CollTxKind, CollRxKind>;
    const std::array<Cell, 9> cells{{
        {CollTxKind::UNICAST, CollRxKind::UNICAST},
        {CollTxKind::SCATTER, CollRxKind::UNICAST},
        {CollTxKind::BROADCAST, CollRxKind::UNICAST},
        {CollTxKind::UNICAST, CollRxKind::GATHER},
        {CollTxKind::SCATTER, CollRxKind::GATHER},
        {CollTxKind::BROADCAST, CollRxKind::GATHER},
        {CollTxKind::UNICAST, CollRxKind::REDUCE},
        {CollTxKind::SCATTER, CollRxKind::REDUCE},
        {CollTxKind::BROADCAST, CollRxKind::REDUCE},
    }};
    for (std::size_t n : {std::size_t{1}, std::size_t{2}}) {
        for (const Cell &cell : cells) {
            config_helper_program helper(
                HelperCollectiveArtifact(cell.first, cell.second, n));
            const auto image = helper.collective_program_image();
            const auto profile_image =
                helper.collective_profile_program_image();
            const auto launches =
                DecodeCollectiveLaunches(helper.BuildConfigMessages());
            bool markers_match = image != nullptr &&
                launches.size() == image->IssueSites().size();
            if (markers_match) {
                for (std::size_t index = 0; index < launches.size(); ++index) {
                    const auto &launch = launches[index];
                    const auto &site = image->IssueSites()[index];
                    CollectiveLaunchV1Role expected_role =
                        CollectiveLaunchV1Role::ISSUE_SEND;
                    if (site.role == IsaV1CollectiveRecordRole::RECEIVE)
                        expected_role =
                            CollectiveLaunchV1Role::ISSUE_RECEIVE;
                    else if (site.role ==
                             IsaV1CollectiveRecordRole::REDUCE_COMPUTE)
                        expected_role = CollectiveLaunchV1Role::
                            DECLARE_REDUCE_COMPUTE;
                    markers_match = markers_match &&
                        launch.role == expected_role &&
                        launch.image_generation == image->Generation() &&
                        launch.plan_index == site.plan_index &&
                        launch.external_record_index == site.record_index &&
                        launch.expected_core == site.core_id &&
                        launch.key == site.key &&
                        launch.public_token == site.public_token;
                }
            }
            checks.Check(image != nullptr && image->Generation() == 1 &&
                             image->Cookie() != 0 &&
                             image->Cores().size() == n && markers_match &&
                             profile_image != nullptr &&
                             profile_image->BaseGeneration() ==
                                 image->Generation() &&
                             profile_image->BaseCookie() == image->Cookie(),
                         "helper nine-grid N=1/2 image and ID60 issue markers");
        }
    }

    {
        NocCollectiveConfigGuard noc_guard;
        static_cast<void>(noc_guard);
        SPEC_NOC_COLL_CONFIG.enabled = true;
        SPEC_NOC_COLL_CONFIG.profile = NocCollProfile::BROADCAST_ONLY;
        SPEC_NOC_COLL_CONFIG.broadcast_backend =
            NocCollBroadcastBackend::MULTICAST;
        SPEC_NOC_COLL_CONFIG.reduce_backend =
            NocCollReduceBackend::ENDPOINT;
        SPEC_NOC_COLL_CONFIG.max_trees_per_batch = 1;
        config_helper_program accelerated(HelperCollectiveArtifact(
            CollTxKind::BROADCAST, CollRxKind::UNICAST, 2));
        const auto profile =
            accelerated.collective_profile_program_image();
        checks.Check(profile != nullptr &&
                         profile->MulticastTreeCount() == 1 &&
                         profile->Plans().front().max_trees_per_batch == 1,
                     "program helper passes explicit multicast batch K");
    }

    {
        NocCollectiveConfigGuard noc_guard;
        static_cast<void>(noc_guard);
        SPEC_NOC_COLL_CONFIG.enabled = true;
        SPEC_NOC_COLL_CONFIG.profile = NocCollProfile::REDUCE_ONLY;
        SPEC_NOC_COLL_CONFIG.broadcast_backend =
            NocCollBroadcastBackend::UNICAST;
        SPEC_NOC_COLL_CONFIG.reduce_backend =
            NocCollReduceBackend::DCA_OFFLOAD;
        checks.Reject<std::runtime_error>(
            "program helper keeps ReduceScatter DCA capability closed",
            "reduce_scatter_dca_unsupported",
            [] {
                config_helper_program rejected(HelperCollectiveArtifact(
                    CollTxKind::SCATTER, CollRxKind::REDUCE, 2));
                static_cast<void>(rejected);
            });
    }

    config_helper_program capacity_helper(HelperCollectiveArtifact(
        CollTxKind::SCATTER, CollRxKind::GATHER, 4));
    const auto capacity_image = capacity_helper.collective_program_image();
    bool demands_within_production_capacity = capacity_image != nullptr;
    if (capacity_image != nullptr) {
        for (const IsaV1CollectiveWaveDemand &demand :
             capacity_image->WaveDemands()) {
            demands_within_production_capacity =
                demands_within_production_capacity &&
                demand.endpoint_sessions <=
                    static_cast<uint32_t>(MAX_BUFFER_PACKET_SIZE) &&
                demand.receive_bytes <= kDteEndpointP2pMaxBytes;
        }
    }
    checks.Check(
        capacity_image != nullptr &&
            capacity_image->Lowering().plans.size() == 1 &&
            capacity_image->Lowering().plans[0].waves.size() > 1 &&
            capacity_image->AdmissionCapacity().waves > 1 &&
            capacity_image->AdmissionCapacity()
                    .max_endpoint_sessions_per_core_wave <=
                static_cast<uint32_t>(MAX_BUFFER_PACKET_SIZE) &&
            capacity_image->AdmissionCapacity()
                    .max_receive_bytes_per_core_wave <=
                kDteEndpointP2pMaxBytes &&
            capacity_image->Limits()
                    .max_endpoint_sessions_per_core_wave ==
                static_cast<uint32_t>(MAX_BUFFER_PACKET_SIZE) &&
            capacity_image->Limits().max_receive_bytes_per_core_wave ==
                kDteEndpointP2pMaxBytes &&
            demands_within_production_capacity,
        "N=4 helper uses production endpoint caps and emits multiple safe waves");

    ProgramArtifact n1_collective = N1GatherArtifact();
    const auto labels_before = g_addr_label_table.table;
    checks.Check(
        std::find(labels_before.begin(), labels_before.end(),
                  "p6_collective_image_label") == labels_before.end(),
        "collective success label starts uninterned");
    config_helper_program helper(n1_collective);
    const auto committed_image = helper.collective_program_image();
    const auto committed_profile_image =
        helper.collective_profile_program_image();
    const auto committed_messages = helper.BuildConfigMessages();
    const auto committed_labels = g_addr_label_table.table;
    const auto launches = DecodeCollectiveLaunches(committed_messages);
    std::vector<uint8_t> prim_order;
    for (const HostEnvelope &message : ForCore(committed_messages, 0))
        if (message.msg.config_end_)
            prim_order.push_back(PrimIdOf(message));
    const auto lifecycle = std::find(
        prim_order.begin(), prim_order.end(),
        PrimIdValue(PrimId::SRAM_LIFECYCLE));
    const auto first_launch = std::find(
        lifecycle == prim_order.end() ? prim_order.end() : lifecycle + 1,
        prim_order.end(), PrimIdValue(PrimId::COLLECTIVE_LAUNCH_V1));
    const auto second_launch = std::find(
        first_launch == prim_order.end() ? prim_order.end() : first_launch + 1,
        prim_order.end(), PrimIdValue(PrimId::COLLECTIVE_LAUNCH_V1));
    const auto fence = std::find(
        second_launch == prim_order.end() ? prim_order.end() :
                                             second_launch + 1,
        prim_order.end(), PrimIdValue(PrimId::DTE_ASYNC));
    checks.Check(committed_image != nullptr &&
                     committed_image->Lowering().plans.size() == 1 &&
                     committed_image->Lowering().plans[0].op ==
                         CollOp::GATHER &&
                     committed_image->Lowering().children.empty() &&
                     launches.size() == 2 &&
                     lifecycle != prim_order.end() &&
                     first_launch != prim_order.end() &&
                     second_launch != prim_order.end() &&
                     fence != prim_order.end(),
                 "N=1 image replaces only issue records and preserves ordinary order");
    checks.Check(
        std::find(committed_labels.begin(), committed_labels.end(),
                  "p6_collective_image_label") != committed_labels.end(),
        "collective success late-interns labels after image validation");

    std::unique_ptr<config_helper_program> clone(helper.clone());
    checks.Check(clone->collective_program_image() == committed_image &&
                     clone->collective_profile_program_image() ==
                         committed_profile_image &&
                     SameMessages(clone->BuildConfigMessages(),
                                  committed_messages),
                 "collective clone shares one immutable image and exact stream");

    ProgramArtifact invalid = n1_collective;
    invalid.strings[3] = "p6_collective_failed_label";
    invalid.cores[0].records.push_back(EventSetRecord(0, 0, 77));
    prim_wire::SetLegacyCompatibility(true);
    checks.Reject<ConfigHelperProgramError>(
        "invalid collective candidate remains transactional",
        "EVENT credit imbalance", [&] {
            helper.LoadProgram(EncodeProgramArtifact(invalid));
        });
    checks.Check(prim_wire::LegacyCompatibilityEnabled() &&
                     helper.collective_program_image() == committed_image &&
                     helper.collective_profile_program_image() ==
                         committed_profile_image &&
                     g_addr_label_table.table == committed_labels &&
                     std::find(committed_labels.begin(),
                               committed_labels.end(),
                               "p6_collective_failed_label") ==
                         committed_labels.end(),
                 "collective failure preserves image, labels, and wire mode");
    prim_wire::SetLegacyCompatibility(false);
    checks.Check(SameMessages(helper.BuildConfigMessages(),
                              committed_messages),
                 "collective failure preserves committed ID60 stream");
}

void CheckTransactionalLoadAndClone(Checks &checks) {
    const std::size_t stash_before = g_prim_stash.size();
    ProgramArtifact good = Artifact(
        {{0, BoundMatmul()}}, EmptyCoreAckPolicy::INCLUDE_EMPTY,
        {{0, 1, 1}}, {0});
    prim_wire::SetLegacyCompatibility(true);
    config_helper_program helper(good);
    checks.Check(!prim_wire::LegacyCompatibilityEnabled(),
                 "successful program commit selects strict Prim wire mode");
    const auto before = helper.BuildConfigMessages();
    const auto coreconfigs_before = helper.coreconfigs;
    const auto labels_after_commit = g_addr_label_table.table;

    ProgramArtifact unsupported = Artifact(
        {{0, {DteScatterRecord()}}}, EmptyCoreAckPolicy::INCLUDE_EMPTY,
        {}, {0});
    unsupported.core_groups = {{1, {0}}};
    const auto unsupported_bytes = EncodeProgramArtifact(unsupported);
    prim_wire::SetLegacyCompatibility(true);
    checks.Reject<RecordLoweringError>(
        "incomplete non-P2P graph aborts whole load", "missing SEND or RECEIVE", [&] {
            helper.LoadProgram(unsupported_bytes);
        });
    checks.Check(SameMessages(before, helper.BuildConfigMessages()),
                 "failed lowering leaves committed helper unchanged");
    checks.Check(prim_wire::LegacyCompatibilityEnabled() &&
                     helper.coreconfigs.size() == coreconfigs_before.size() &&
                     helper.coreconfigs[0].id == coreconfigs_before[0].id &&
                     helper.coreconfigs[0].loop == coreconfigs_before[0].loop,
                 "failed load preserves strict-mode flag and CoreConfigs");

    auto corrupt = EncodeProgramArtifact(good);
    corrupt.back() ^= 1;
    checks.Reject<ProgramFormatError>("corrupt artifact", "CRC", [&] {
        helper.LoadProgram(corrupt);
    });
    checks.Check(SameMessages(before, helper.BuildConfigMessages()),
                 "failed decode leaves committed helper unchanged");

    ProgramArtifact capability = good;
    capability.capabilities = CapabilityBit(IsaCapability::PD_CONTEXT);
    const auto capability_bytes = EncodeProgramArtifact(capability);
    checks.Reject<ConfigHelperProgramError>(
        "PD capability disabled", "disables GLOBAL/PD", [&] {
            helper.LoadProgram(capability_bytes);
        });

    std::unique_ptr<config_helper_program> clone(helper.clone());
    checks.Check(SameMessages(before, clone->BuildConfigMessages()) &&
                     clone->expected_ack_cores() ==
                         helper.expected_ack_cores(),
                 "clone reparses saved artifact bytes deterministically");
    checks.Check(g_prim_stash.size() == stash_before &&
                     g_addr_label_table.table == labels_after_commit,
                 "build/failed-load/clone add no labels beyond committed bind set");
    prim_wire::SetLegacyCompatibility(false);
}

void CheckPlatformAndLifecycleValidation(Checks &checks) {
    ProgramArtifact out_of_range = Artifact(
        {{static_cast<uint64_t>(TOTAL_CORES), {MatmulRecord()}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {},
        {static_cast<uint64_t>(TOTAL_CORES)});
    checks.Reject<ConfigHelperProgramError>(
        "program core outside platform", "outside platform TOTAL_CORES", [&] {
            config_helper_program rejected(out_of_range);
        });

    ProgramArtifact group_out_of_range = Artifact(
        {{0, BoundMatmul()}}, EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {0});
    group_out_of_range.core_groups = {
        {1, {static_cast<uint64_t>(TOTAL_CORES)}}};
    checks.Reject<ConfigHelperProgramError>(
        "group member outside platform", "outside platform TOTAL_CORES", [&] {
            config_helper_program rejected(group_out_of_range);
        });

    ProgramArtifact group_inactive = Artifact(
        {{0, BoundMatmul()}}, EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {0});
    group_inactive.core_groups = {{1, {1}}};
    checks.Reject<ConfigHelperProgramError>(
        "group member outside active program", "not an active", [&] {
            config_helper_program rejected(group_inactive);
        });

    ProgramArtifact cross_die = Artifact(
        {{0, BoundMatmul()},
         {static_cast<uint64_t>(CORES_PER_DIE), BoundMatmul(10)}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {0});
    cross_die.core_groups = {
        {1, {0, static_cast<uint64_t>(CORES_PER_DIE)}}};
    checks.Reject<ConfigHelperProgramError>(
        "group cannot span dies", "spans multiple dies", [&] {
            config_helper_program rejected(cross_die);
        });

    ProgramArtifact same_die = Artifact(
        {{0, BoundMatmul()}, {1, BoundMatmul(10)}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {0});
    same_die.core_groups = {{1, {0, 1}}};
    config_helper_program same_die_helper(same_die);
    const auto committed_registry =
        same_die_helper.core_group_registry();
    checks.Check(same_die_helper.coreconfigs.size() == 2 &&
                     same_die_helper.coreconfigs[0].id == 0 &&
                     same_die_helper.coreconfigs[1].id == 1,
                 "same-die active group and all active CoreConfigs commit");
    checks.Check(committed_registry != nullptr &&
                     committed_registry->GroupCount() == 1 &&
                     committed_registry->Members(1) ==
                         std::vector<uint16_t>({0, 1}),
                 "program helper commits one immutable runtime group registry");
    checks.Reject<ConfigHelperProgramError>(
        "failed reload preserves committed registry", "spans multiple dies",
        [&] { same_die_helper.LoadProgram(EncodeProgramArtifact(cross_die)); });
    checks.Check(same_die_helper.core_group_registry() == committed_registry &&
                     same_die_helper.core_group_registry()->Members(1) ==
                         std::vector<uint16_t>({0, 1}),
                 "failed load has no committed registry side effect");

    ProgramArtifact no_cores = Artifact(
        {}, EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {});
    checks.Reject<ConfigHelperProgramError>(
        "zero configured cores are rejected", "configured core", [&] {
            config_helper_program rejected(no_cores);
        });

    ProgramArtifact no_terminal = Artifact(
        {{0, BoundMatmul()}}, EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {});
    checks.Reject<ConfigHelperProgramError>(
        "DATAFLOW requires terminal/DONE closure", "terminal/DONE", [&] {
            config_helper_program rejected(no_terminal);
        });
}

void CheckSynchronizationPrograms(Checks &checks) {
    ProgramArtifact valid = Artifact(
        {{0, {GroupSyncRecord(7, 0), GroupSyncRecord(7, 1),
              EventSetRecord(0, 1, 0x80000001u),
              EventSetRecord(0, 1, 0x80000001u)}},
         {1, {GroupSyncRecord(7, 0), GroupSyncRecord(7, 1),
              EventWaitRecord(0, 1, 0x80000001u, 2)}}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {1});
    valid.core_groups = {{7, {0, 1}}};
    config_helper_program helper(valid);
    const auto committed_registry = helper.core_group_registry();
    const auto messages = helper.BuildConfigMessages();
    std::size_t group_prims = 0;
    std::size_t event_prims = 0;
    for (const HostEnvelope &message : messages) {
        const uint8_t id = PrimIdOf(message);
        group_prims += id == PrimIdValue(PrimId::GROUP_SYNC);
        event_prims += id == PrimIdValue(PrimId::EVENT_CONTROL);
    }
    checks.Check(group_prims == 4 && event_prims == 3 &&
                     committed_registry != nullptr &&
                     committed_registry->Members(7) ==
                         std::vector<uint16_t>({0, 1}),
                 "balanced GROUP_SYNC/EVENT program lowers to strict Prims");

    ProgramArtifact unknown = valid;
    std::get<GroupSyncOperands>(
        unknown.cores[0].records[0].operands).group_id = 8;
    checks.Reject<ProgramFormatError>(
        "unknown GROUP_SYNC group",
        "core 0 record 0 references an unknown core group", [&] {
            config_helper_program rejected(unknown);
        });

    ProgramArtifact nonmember = valid;
    nonmember.core_groups = {{7, {0}}};
    checks.Reject<ConfigHelperProgramError>(
        "nonmember GROUP_SYNC core", "nonmember", [&] {
            config_helper_program rejected(nonmember);
        });

    ProgramArtifact out_of_order = valid;
    std::get<GroupSyncOperands>(
        out_of_order.cores[0].records[0].operands).sync_seq = 1;
    checks.Reject<ConfigHelperProgramError>(
        "out-of-order GROUP_SYNC", "out-of-order", [&] {
            config_helper_program rejected(out_of_order);
        });

    ProgramArtifact unequal = valid;
    unequal.cores[1].records.erase(unequal.cores[1].records.begin() + 1);
    checks.Reject<ConfigHelperProgramError>(
        "unequal GROUP_SYNC member counts", "different synchronization counts",
        [&] { config_helper_program rejected(unequal); });

    ProgramArtifact wrong_set_executor = valid;
    std::get<EventSetOperands>(
        wrong_set_executor.cores[0].records[2].operands).source_core = 1;
    checks.Reject<ConfigHelperProgramError>(
        "EVENT_SET executing source", "source_core must equal", [&] {
            config_helper_program rejected(wrong_set_executor);
        });

    ProgramArtifact wrong_wait_executor = valid;
    std::get<EventWaitOperands>(
        wrong_wait_executor.cores[1].records[2].operands).destination_core = 0;
    checks.Reject<ConfigHelperProgramError>(
        "EVENT_WAIT executing destination", "destination_core must equal", [&] {
            config_helper_program rejected(wrong_wait_executor);
        });

    ProgramArtifact inactive_endpoint = valid;
    std::get<EventSetOperands>(
        inactive_endpoint.cores[0].records[2].operands).destination_core = 2;
    checks.Reject<ConfigHelperProgramError>(
        "EVENT inactive endpoint", "active program cores", [&] {
            config_helper_program rejected(inactive_endpoint);
        });

    ProgramArtifact missing_set = valid;
    missing_set.cores[0].records.pop_back();
    missing_set.cores[0].records.pop_back();
    checks.Reject<ConfigHelperProgramError>(
        "EVENT missing SET credits", "credit imbalance", [&] {
            config_helper_program rejected(missing_set);
        });

    ProgramArtifact extra_set = valid;
    extra_set.cores[0].records.push_back(
        EventSetRecord(0, 1, 0x80000001u));
    checks.Reject<ConfigHelperProgramError>(
        "EVENT extra SET residual", "credit imbalance", [&] {
            helper.LoadProgram(EncodeProgramArtifact(extra_set));
        });
    checks.Check(helper.core_group_registry() == committed_registry &&
                     helper.core_group_registry()->Members(7) ==
                         std::vector<uint16_t>({0, 1}),
                 "failed EVENT reload preserves committed group registry");
}

void CheckAckDoneProtocol(Checks &checks) {
    ProgramArtifact artifact = Artifact(
        {{0, BoundMatmul()}, {1, BoundMatmul(10)}},
        EmptyCoreAckPolicy::INCLUDE_EMPTY, {}, {1});
    config_helper_program helper(artifact);
    checks.Check(!helper.AcceptAck(Ack(0), 7), "partial ACK set");
    checks.Reject<ConfigHelperProgramError>("duplicate ACK", "duplicate ACK",
                                            [&] {
        helper.AcceptAck(Ack(0), 7);
    });
    checks.Reject<ConfigHelperProgramError>("unexpected ACK",
                                            "unexpected ACK", [&] {
        helper.AcceptAck(Ack(2), 7);
    });
    checks.Check(helper.AcceptAck(Ack(1), 7), "complete ACK set");
    checks.Check(!helper.AcceptAck(Ack(0), 8),
                 "new ACK phase resets duplicate tracking");

    config_helper_program late_ack_helper(artifact);
    checks.Check(!late_ack_helper.AcceptAck(Ack(0), 7),
                 "late ACK fixture starts an incomplete set");
    checks.Check(late_ack_helper.AcceptAck(Ack(1), 8),
                 "trace flow-id advance cannot discard an incomplete ACK set");
    checks.Check(helper.AcceptDone(Done(1)), "complete DONE set");
    checks.Reject<ConfigHelperProgramError>("duplicate DONE",
                                            "duplicate DONE", [&] {
        helper.AcceptDone(Done(1));
    });
    checks.Reject<ConfigHelperProgramError>("unexpected DONE",
                                            "unexpected DONE", [&] {
        helper.AcceptDone(Done(0));
    });

    config_helper_program parsed(artifact);
    parsed.g_temp_done_msg.push_back(Done(1));
    parsed.parse_done_msg(nullptr, nullptr);
    checks.Check(parsed.g_recv_done_cnt == 1,
                 "DONE parser reaches complete DATAFLOW stop path");
}

void CheckStartWireLimits(Checks &checks) {
    ProgramArtifact tag = Artifact(
        {{0, BoundMatmul()}}, EmptyCoreAckPolicy::INCLUDE_EMPTY,
        {{0, 0x10000, 1}}, {0});
    checks.Reject<ConfigHelperProgramError>("start tag narrowing",
                                            "tag exceeds", [&] {
        config_helper_program helper(tag);
    });
    ProgramArtifact count = Artifact(
        {{0, BoundMatmul()}}, EmptyCoreAckPolicy::INCLUDE_EMPTY,
        {{0, 1, 0x100}}, {0});
    checks.Reject<ConfigHelperProgramError>("start count narrowing",
                                            "count exceeds", [&] {
        config_helper_program helper(count);
    });
}

} // namespace

ConfigHelperProgramSelfTestResult CheckConfigHelperProgram() {
    PlatformDimensionsGuard platform_guard;
    Checks checks;
    CheckSingleCoreFlow(checks);
    CheckEmptyCorePolicies(checks);
    CheckRelocations(checks);
    CheckSramBindLifecycle(checks);
    CheckSramLifecyclePrograms(checks);
    CheckDteTokenPrograms(checks);
    CheckDteEndpointPrograms(checks);
    CheckCollectiveProgramImages(checks);
    CheckTransactionalLoadAndClone(checks);
    CheckPlatformAndLifecycleValidation(checks);
    CheckSynchronizationPrograms(checks);
    CheckStartWireLimits(checks);
    CheckAckDoneProtocol(checks);
    return std::move(checks.result);
}

int RunConfigHelperProgramSelfTest() {
    const ConfigHelperProgramSelfTestResult result =
        CheckConfigHelperProgram();
    if (result.passed()) {
        std::cout << "Program config helper selftest passed (" << result.checks
                  << " checks)\n";
        return 0;
    }
    std::cerr << "Program config helper selftest failed ("
              << result.failures.size() << "/" << result.checks
              << " checks failed)\n";
    for (const std::string &failure : result.failures)
        std::cerr << "  - " << failure << '\n';
    return 1;
}
