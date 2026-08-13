#pragma once

#include "isa/collective_program_v1.h"
#include "utils/prim_utils.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace collective_wave_runtime_v1_test {

class StrictWireScope {
public:
    StrictWireScope()
        : previous_(prim_wire::LegacyCompatibilityEnabled()) {
        prim_wire::SetLegacyCompatibility(false);
    }
    ~StrictWireScope() {
        prim_wire::SetLegacyCompatibility(previous_);
    }

private:
    bool previous_;
};

struct Cell {
    CollTxKind tx = CollTxKind::UNICAST;
    CollRxKind rx = CollRxKind::UNICAST;
};

inline SramAddressOperand Absolute(uint64_t address) {
    SramAddressOperand operand;
    operand.kind = SramAddressKind::ABSOLUTE;
    operand.absolute_address_bytes = address;
    return operand;
}

inline bool SendRole(CollOp op, std::size_t rank,
                     std::size_t root) {
    if (op == CollOp::P2P) return rank == 0;
    return op == CollOp::SCATTER || op == CollOp::BROADCAST
               ? rank == root
               : true;
}

inline bool ReceiveRole(CollOp op, std::size_t rank,
                        std::size_t root) {
    if (op == CollOp::P2P) return rank == root;
    return op == CollOp::GATHER || op == CollOp::REDUCE
               ? rank == root
               : true;
}

inline bool ComputeRole(CollOp op, std::size_t rank,
                        std::size_t root) {
    return op == CollOp::REDUCE
               ? rank == root
               : op == CollOp::REDUCESCATTER ||
                     op == CollOp::ALLREDUCE;
}

inline ProgramArtifact Artifact(const std::vector<Cell> &cells,
                                std::size_t n) {
    ProgramArtifact artifact;
    ProgramCoreGroup group;
    group.group_id = 7;
    for (std::size_t rank = 0; rank < n; ++rank) {
        group.members.push_back(rank);
        artifact.cores.push_back(ProgramCore{rank, {}});
        artifact.envelope.active_cores.push_back(rank);
        artifact.envelope.expected_ack_cores.push_back(rank);
    }
    artifact.core_groups.push_back(std::move(group));
    artifact.envelope.terminal_cores = {n - 1};
    artifact.envelope.expected_done_cores = {n - 1};

    for (std::size_t instance = 0;
         instance < cells.size(); ++instance) {
        const Cell &cell = cells[instance];
        const CollOp op = IsaV1CollectiveOp(cell.tx, cell.rx);
        const std::size_t root = n == 1 ? 0 : 1;
        const uint32_t collective_id =
            static_cast<uint32_t>(11 + instance);
        const uint32_t fsm =
            static_cast<uint32_t>(0x10000 + instance * 0x1000);
        const uint32_t token_base =
            static_cast<uint32_t>(1000 + instance * 100);
        const uint64_t address_base =
            0x100000ULL + instance * 0x100000ULL;
        for (std::size_t rank = 0; rank < n; ++rank) {
            ProgramCore &core = artifact.cores[rank];
            if (SendRole(op, rank, root)) {
                DteSendOperands send;
                send.mode = cell.tx == CollTxKind::SCATTER
                                ? DteSendMode::SCATTER
                                : cell.tx == CollTxKind::BROADCAST
                                      ? DteSendMode::BROADCAST
                                      : DteSendMode::P2P;
                send.completion = EndpointCompletion::ASYNC;
                send.fsm_id = fsm;
                send.token = token_base + rank * 4 + 1;
                send.length_bytes = 32;
                send.source =
                    Absolute(address_base + rank * 0x1000);
                send.group_id = 7;
                send.collective_id = collective_id;
                core.records.push_back({Opcode::DTE_SEND, send});
            }
            if (ReceiveRole(op, rank, root)) {
                DteRecvOperands receive;
                receive.mode = cell.rx == CollRxKind::GATHER
                                   ? DteRecvMode::GATHER
                                   : cell.rx == CollRxKind::REDUCE
                                         ? DteRecvMode::REDUCE
                                         : DteRecvMode::P2P;
                receive.completion = EndpointCompletion::ASYNC;
                receive.fsm_id = fsm;
                receive.token = token_base + rank * 4 + 2;
                receive.length_bytes = 32;
                receive.destination =
                    Absolute(address_base + 0x40000 +
                             rank * 0x1000);
                receive.expected_sources =
                    cell.rx == CollRxKind::UNICAST
                        ? 0
                        : static_cast<uint16_t>(n - 1);
                receive.datatype =
                    cell.rx == CollRxKind::REDUCE
                        ? EndpointDataType::INT32
                        : EndpointDataType::UINT8;
                receive.reduce_op =
                    cell.rx == CollRxKind::REDUCE
                        ? ReduceOperator::SUM
                        : ReduceOperator::NONE;
                receive.group_id = 7;
                receive.collective_id = collective_id;
                core.records.push_back(
                    {Opcode::DTE_RECV, receive});
            }
            if (ComputeRole(op, rank, root)) {
                ReduceComputeOperands compute;
                compute.datatype = EndpointDataType::INT32;
                compute.reduce_op = ReduceOperator::SUM;
                compute.group_id = 7;
                compute.collective_id = collective_id;
                compute.root_rank =
                    op == CollOp::REDUCE ? root : 0;
                compute.self_rank = rank;
                compute.element_count = 8;
                compute.source =
                    Absolute(address_base + 0x40000 +
                             rank * 0x1000);
                compute.destination =
                    Absolute(address_base + 0x80000 +
                             rank * 0x1000);
                core.records.push_back(
                    {Opcode::REDUCE_COMPUTE, compute});
            }
        }
    }
    return artifact;
}

inline IsaV1CollectiveProgramImage BuildImage(
    const std::vector<Cell> &cells, std::size_t n,
    IsaV1PlannerCapacity planner = {}) {
    const ProgramArtifact artifact = Artifact(cells, n);
    IsaV1CollectiveProgramImageConfig config;
    config.total_cores = 8;
    config.cores_per_die = 8;
    config.generation = 0x12340000ULL + cells.size() * 16 + n;
    config.planner_capacity = planner;
    const auto lowering = LowerIsaV1CollectiveArtifact(
        artifact, config.total_cores, config.cores_per_die,
        planner);
    return BuildIsaV1CollectiveProgramImage(
        artifact, lowering, config);
}

} // namespace collective_wave_runtime_v1_test
