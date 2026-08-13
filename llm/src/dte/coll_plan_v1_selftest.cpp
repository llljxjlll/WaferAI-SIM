#include "dte/coll_plan_v1.h"
#include "dte/coll_plan_v1_selftest.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct Suite {
    int checks = 0;
    int failures = 0;

    void Check(bool condition, const std::string &name) {
        ++checks;
        if (!condition) ++failures;
        std::cout << "  [" << (condition ? " ok " : "FAIL") << "] "
                  << name << std::endl;
    }

    template <class Exception, class F>
    void Throws(F &&fn, const std::string &name) {
        bool matched = false;
        try {
            fn();
        } catch (const Exception &) {
            matched = true;
        } catch (...) {
        }
        Check(matched, name);
    }
};

bool RootTransmit(CollOp op) {
    return op == CollOp::SCATTER || op == CollOp::BROADCAST;
}

bool RootReceive(CollOp op) {
    return op == CollOp::GATHER || op == CollOp::REDUCE;
}

bool Symmetric(CollOp op) {
    return op == CollOp::ALLTOALL || op == CollOp::ALLGATHER ||
           op == CollOp::REDUCESCATTER || op == CollOp::ALLREDUCE;
}

bool SendRole(CollOp op, const IsaV1CollectiveSpec &spec, size_t rank) {
    if (op == CollOp::P2P) return rank == spec.p2p_source_rank;
    if (RootTransmit(op)) return rank == spec.root_rank;
    return RootReceive(op) || Symmetric(op);
}

bool ReceiveRole(CollOp op, const IsaV1CollectiveSpec &spec, size_t rank) {
    if (op == CollOp::P2P) return rank == spec.p2p_destination_rank;
    if (RootTransmit(op)) return true;
    if (RootReceive(op)) return rank == spec.root_rank;
    return Symmetric(op);
}

bool ReduceTarget(CollOp op, const IsaV1CollectiveSpec &spec, size_t rank) {
    if (op == CollOp::REDUCE) return rank == spec.root_rank;
    return op == CollOp::REDUCESCATTER || op == CollOp::ALLREDUCE;
}

IsaV1CollectiveSpec Make(CollTxKind tx, CollRxKind rx, size_t n,
                         uint64_t length = 32) {
    static const uint16_t cores[] = {2, 7, 11, 19};
    IsaV1CollectiveSpec spec;
    spec.tx_kind = tx;
    spec.rx_kind = rx;
    spec.key = {17, 23, 5};
    spec.group.assign(cores, cores + n);
    spec.root_rank = n == 1 ? 0 : 1;
    spec.p2p_source_rank = 0;
    spec.p2p_destination_rank = static_cast<uint16_t>(n - 1);
    spec.length_bytes = length;
    spec.dtype = rx == CollRxKind::REDUCE ? CollDType::INT32
                                          : CollDType::UINT8;
    spec.reduce_op = rx == CollRxKind::REDUCE ? CollReduceOp::SUM
                                               : CollReduceOp::NONE;
    spec.tree_id = 0;
    spec.expected_sources =
        (rx == CollRxKind::GATHER || rx == CollRxKind::REDUCE)
            ? static_cast<uint16_t>(n - 1)
            : 0;
    spec.logical_fsm_id_base = 0x10000;
    spec.rank_records.resize(n);
    const CollOp op = IsaV1CollectiveOp(tx, rx);
    for (size_t rank = 0; rank < n; ++rank) {
        auto &records = spec.rank_records[rank];
        if (SendRole(op, spec, rank)) {
            records.send.present = true;
            records.send.asynchronous = true;
            records.send.token = static_cast<uint32_t>(100 + rank * 10 + 1);
            records.send.base_address_bytes = 0x10000 + rank * 0x1000;
        }
        if (ReceiveRole(op, spec, rank)) {
            records.receive.present = true;
            records.receive.asynchronous = true;
            records.receive.token =
                static_cast<uint32_t>(100 + rank * 10 + 2);
            records.receive.base_address_bytes = 0x20000 + rank * 0x1000;
        }
        if (ReduceTarget(op, spec, rank))
            records.result_address_bytes = 0x30000 + rank * 0x1000;
    }
    return spec;
}

uint64_t ExpectedRemotePairs(CollOp op, const IsaV1CollectiveSpec &spec) {
    const uint64_t n = spec.group.size();
    if (op == CollOp::P2P)
        return spec.p2p_source_rank == spec.p2p_destination_rank ? 0 : 1;
    if (RootTransmit(op) || RootReceive(op)) return n - 1;
    return n * (n - 1);
}

uint64_t ExpectedLocalCopies(CollOp op, const IsaV1CollectiveSpec &spec) {
    if (op == CollOp::P2P)
        return spec.p2p_source_rank == spec.p2p_destination_rank ? 1 : 0;
    return 1;
}

bool HasCanonicalChildren(const IsaV1CollectivePlan &plan) {
    for (size_t i = 0; i < plan.child_flows.size(); ++i) {
        const auto &flow = plan.child_flows[i];
        if (flow.fsm_id != plan.logical_fsm_id_base + i) return false;
        if (i != 0) {
            const auto &previous = plan.child_flows[i - 1];
            if (std::tie(flow.chunk_offset_bytes, flow.source_rank,
                         flow.destination_rank) <
                std::tie(previous.chunk_offset_bytes, previous.source_rank,
                         previous.destination_rank))
                return false;
        }
    }
    return true;
}

bool HasRxFirstActions(const IsaV1CollectivePlan &plan) {
    for (const auto &rank_actions : plan.actions_by_rank) {
        for (const auto &wave : plan.waves) {
            bool saw_posted_barrier = false;
            for (const auto &action : rank_actions) {
                if (action.wave_index != wave.wave_index) continue;
                if (action.kind == IsaV1ActionKind::POSTED_BARRIER)
                    saw_posted_barrier = true;
                if (action.kind == IsaV1ActionKind::ISSUE_SEND &&
                    !saw_posted_barrier)
                    return false;
            }
            if (!saw_posted_barrier) return false;
        }
    }
    return true;
}

size_t CountActions(const IsaV1CollectivePlan &plan,
                    IsaV1ActionKind kind) {
    size_t count = 0;
    for (const auto &rank : plan.actions_by_rank)
        count += std::count_if(rank.begin(), rank.end(),
                              [&](const IsaV1Action &action) {
                                  return action.kind == kind;
                              });
    return count;
}

uint64_t TotalActions(const IsaV1CollectivePlan &plan) {
    uint64_t count = 0;
    for (const auto &rank : plan.actions_by_rank) count += rank.size();
    return count;
}

uint64_t DerivedBytes(const IsaV1CollectivePlan &plan) {
    return plan.group.size() * sizeof(uint16_t) +
           plan.rank_records.size() * sizeof(IsaV1RankRecordContract) +
           plan.child_flows.size() * sizeof(IsaV1ChildFlow) +
           plan.local_copies.size() * sizeof(IsaV1LocalCopy) +
           plan.reduce_targets.size() * sizeof(IsaV1ReduceTarget) +
           plan.waves.size() * sizeof(IsaV1Wave) +
           plan.child_flows.size() * sizeof(uint32_t) +
           plan.actions_by_rank.size() *
               sizeof(std::vector<IsaV1Action>) +
           TotalActions(plan) * sizeof(IsaV1Action);
}

void TestNineGrid(Suite &suite) {
    struct Cell {
        CollTxKind tx;
        CollRxKind rx;
        CollOp op;
    };
    const std::vector<Cell> cells = {
        {CollTxKind::UNICAST, CollRxKind::UNICAST, CollOp::P2P},
        {CollTxKind::SCATTER, CollRxKind::UNICAST, CollOp::SCATTER},
        {CollTxKind::BROADCAST, CollRxKind::UNICAST, CollOp::BROADCAST},
        {CollTxKind::UNICAST, CollRxKind::GATHER, CollOp::GATHER},
        {CollTxKind::SCATTER, CollRxKind::GATHER, CollOp::ALLTOALL},
        {CollTxKind::BROADCAST, CollRxKind::GATHER, CollOp::ALLGATHER},
        {CollTxKind::UNICAST, CollRxKind::REDUCE, CollOp::REDUCE},
        {CollTxKind::SCATTER, CollRxKind::REDUCE,
         CollOp::REDUCESCATTER},
        {CollTxKind::BROADCAST, CollRxKind::REDUCE, CollOp::ALLREDUCE},
    };
    for (size_t n : {size_t(1), size_t(2), size_t(4)}) {
        for (const auto &cell : cells) {
            const auto spec = Make(cell.tx, cell.rx, n);
            const auto plan = PlanIsaV1Collective(spec);
            const bool counts =
                plan.op == cell.op &&
                plan.child_flows.size() == ExpectedRemotePairs(cell.op, spec) &&
                plan.local_copies.size() ==
                    (Symmetric(cell.op) ? n : ExpectedLocalCopies(cell.op, spec));
            suite.Check(counts,
                        "3x3 cell child/local counts N=" + std::to_string(n) +
                            " op=" + std::to_string(static_cast<int>(cell.op)));
            suite.Check(HasCanonicalChildren(plan) && HasRxFirstActions(plan),
                        "3x3 cell canonical fsm order and RX-first waves N=" +
                            std::to_string(n) + " op=" +
                            std::to_string(static_cast<int>(cell.op)));
            suite.Check(plan == PlanIsaV1Collective(spec),
                        "3x3 duplicate input is byte-for-byte stable N=" +
                            std::to_string(n) + " op=" +
                            std::to_string(static_cast<int>(cell.op)));

            const size_t expected_targets =
                n == 1 || !CollIsReduction(cell.op)
                    ? 0
                    : (cell.op == CollOp::REDUCE ? 1 : n);
            suite.Check(plan.reduce_targets.size() == expected_targets &&
                            CountActions(plan,
                                         IsaV1ActionKind::REDUCE_COMPUTE) ==
                                expected_targets,
                        "reduction targets are root-only or symmetric N=" +
                            std::to_string(n) + " op=" +
                            std::to_string(static_cast<int>(cell.op)));
        }
    }
}

void TestLayoutsAndWaves(Suite &suite) {
    auto a2a = Make(CollTxKind::SCATTER, CollRxKind::GATHER, 4, 13);
    IsaV1PlannerCapacity small;
    small.max_child_bytes = 5;
    small.max_receive_bytes_per_rank_per_wave = 5;
    small.max_sessions_per_rank_per_wave = 1;
    const auto plan = PlanIsaV1Collective(a2a, small);
    bool layout = plan.child_flows.size() == 36 && plan.local_copies.size() == 4;
    for (const auto &flow : plan.child_flows) {
        layout = layout &&
                 flow.source_offset_bytes ==
                     uint64_t(flow.destination_rank) * 13 +
                         flow.chunk_offset_bytes &&
                 flow.destination_offset_bytes ==
                     uint64_t(flow.source_rank) * 13 +
                         flow.chunk_offset_bytes &&
                 flow.length_bytes <= 5;
    }
    for (const auto &wave : plan.waves) {
        std::vector<uint32_t> sessions(4, 0);
        std::vector<uint64_t> bytes(4, 0);
        for (uint32_t index : wave.child_indices) {
            const auto &flow = plan.child_flows[index];
            ++sessions[flow.source_rank];
            ++sessions[flow.destination_rank];
            bytes[flow.destination_rank] += flow.length_bytes;
        }
        layout = layout &&
                 *std::max_element(sessions.begin(), sessions.end()) <= 1 &&
                 *std::max_element(bytes.begin(), bytes.end()) <= 5;
    }
    suite.Check(layout && HasCanonicalChildren(plan),
                "tight scatter/gather offsets, chunking, and wave capacities");

    auto ar = Make(CollTxKind::BROADCAST, CollRxKind::REDUCE, 4, 32);
    const auto ar_plan = PlanIsaV1Collective(ar);
    bool direct = ar_plan.reduce_targets.size() == 4;
    bool has_non_root_pair = false;
    for (const auto &flow : ar_plan.child_flows) {
        direct = direct && flow.source_offset_bytes == flow.chunk_offset_bytes &&
                 flow.destination_offset_bytes ==
                     uint64_t(flow.source_rank) * 32 +
                         flow.chunk_offset_bytes;
        has_non_root_pair = has_non_root_pair ||
                            (flow.source_rank != ar.root_rank &&
                             flow.destination_rank != ar.root_rank);
    }
    suite.Check(direct && has_non_root_pair,
                "AllReduce is symmetric direct-to-destination with no root transit");

    auto rs = Make(CollTxKind::SCATTER, CollRxKind::REDUCE, 4, 32);
    const auto rs_plan = PlanIsaV1Collective(rs);
    bool rs_direct = rs_plan.reduce_targets.size() == 4;
    for (const auto &flow : rs_plan.child_flows)
        rs_direct = rs_direct &&
                    flow.source_offset_bytes ==
                        uint64_t(flow.destination_rank) * 32 +
                            flow.chunk_offset_bytes &&
                    flow.destination_offset_bytes ==
                        uint64_t(flow.source_rank) * 32 +
                            flow.chunk_offset_bytes;
    suite.Check(rs_direct,
                "ReduceScatter uses symmetric slice-to-owner tight layouts");

    const auto gather = PlanIsaV1Collective(
        Make(CollTxKind::UNICAST, CollRxKind::GATHER, 4));
    suite.Check(gather.group == std::vector<uint16_t>({2, 7, 11, 19}) &&
                    gather.root_rank == 1 &&
                    std::all_of(gather.child_flows.begin(),
                                gather.child_flows.end(),
                                [&](const IsaV1ChildFlow &flow) {
                                    return flow.destination_core == 7;
                                }),
                "non-contiguous group and non-zero root remain canonical");
}

void TestContractFailures(Suite &suite) {
    suite.Throws<std::invalid_argument>([&] {
        auto spec = Make(CollTxKind::UNICAST, CollRxKind::GATHER, 4);
        spec.rank_records[0].send.asynchronous = false;
        (void)PlanIsaV1Collective(spec);
    }, "collective records must be ASYNC");
    suite.Throws<std::invalid_argument>([&] {
        auto spec = Make(CollTxKind::BROADCAST, CollRxKind::UNICAST, 4);
        spec.tree_id = 1;
        (void)PlanIsaV1Collective(spec);
    }, "baseline rejects non-zero tree_id");
    suite.Throws<std::invalid_argument>([&] {
        auto spec = Make(CollTxKind::UNICAST, CollRxKind::REDUCE, 4);
        spec.expected_sources = 4;
        (void)PlanIsaV1Collective(spec);
    }, "reduce expected_sources is exactly N-1");
    suite.Throws<std::invalid_argument>([&] {
        auto spec = Make(CollTxKind::SCATTER, CollRxKind::GATHER, 4);
        spec.rank_records[0].receive.present = false;
        (void)PlanIsaV1Collective(spec);
    }, "3x3 rank role mismatch is rejected");
    suite.Throws<std::invalid_argument>([&] {
        auto spec = Make(CollTxKind::SCATTER, CollRxKind::GATHER, 4);
        spec.rank_records[0].receive.token = spec.rank_records[0].send.token;
        (void)PlanIsaV1Collective(spec);
    }, "same-core aggregate token collision is rejected");
    suite.Throws<std::invalid_argument>([&] {
        auto spec = Make(CollTxKind::SCATTER, CollRxKind::GATHER, 4);
        spec.group[2] = spec.group[1];
        (void)PlanIsaV1Collective(spec);
    }, "duplicate group input is rejected");
    suite.Throws<std::invalid_argument>([&] {
        auto spec = Make(CollTxKind::UNICAST, CollRxKind::REDUCE, 4);
        spec.dtype = CollDType::FP32;
        (void)PlanIsaV1Collective(spec);
    }, "floating-point reduction is rejected");
    suite.Throws<std::invalid_argument>([&] {
        auto spec = Make(CollTxKind::UNICAST, CollRxKind::REDUCE, 4, 30);
        (void)PlanIsaV1Collective(spec);
    }, "reduction L must contain whole dtype elements");
    suite.Throws<std::invalid_argument>([&] {
        auto spec = Make(CollTxKind::UNICAST, CollRxKind::UNICAST, 2);
        IsaV1PlannerCapacity capacity;
        capacity.max_sessions_per_rank_per_wave = 0;
        (void)PlanIsaV1Collective(spec, capacity);
    }, "zero planner capacity is rejected");
}

void TestDerivedLimits(Suite &suite) {
    auto spec = Make(CollTxKind::SCATTER, CollRxKind::GATHER, 4, 13);
    IsaV1PlannerCapacity exact;
    exact.max_child_bytes = 5;
    exact.max_receive_bytes_per_rank_per_wave = 5;
    exact.max_sessions_per_rank_per_wave = 1;
    const auto reference = PlanIsaV1Collective(spec, exact);
    exact.max_children = reference.child_flows.size();
    exact.max_actions = TotalActions(reference);
    exact.max_waves = reference.waves.size();
    exact.max_derived_bytes = DerivedBytes(reference);
    suite.Check(PlanIsaV1Collective(spec, exact) == reference,
                "all derived capacities accept exactly their limit");

    auto over_children = exact;
    --over_children.max_children;
    suite.Throws<std::invalid_argument>([&] {
        (void)PlanIsaV1Collective(spec, over_children);
    }, "child limit+1 is rejected before materialization");
    auto over_actions = exact;
    --over_actions.max_actions;
    suite.Throws<std::invalid_argument>([&] {
        (void)PlanIsaV1Collective(spec, over_actions);
    }, "action limit+1 is rejected before materialization");
    auto over_waves = exact;
    --over_waves.max_waves;
    suite.Throws<std::invalid_argument>([&] {
        (void)PlanIsaV1Collective(spec, over_waves);
    }, "wave limit+1 is rejected before materialization");
    auto over_bytes = exact;
    --over_bytes.max_derived_bytes;
    suite.Throws<std::invalid_argument>([&] {
        (void)PlanIsaV1Collective(spec, over_bytes);
    }, "derived-byte limit+1 is rejected before materialization");

    suite.Throws<std::invalid_argument>([&] {
        auto chunked = Make(CollTxKind::UNICAST, CollRxKind::UNICAST, 2, 17);
        IsaV1PlannerCapacity capacity;
        capacity.max_child_bytes = 8;
        capacity.max_receive_bytes_per_rank_per_wave = 8;
        capacity.max_children = 2;
        (void)PlanIsaV1Collective(chunked, capacity);
    }, "chunk amplification is bounded before child allocation");
    suite.Throws<std::invalid_argument>([&] {
        auto quadratic =
            Make(CollTxKind::SCATTER, CollRxKind::GATHER, 4, 1);
        IsaV1PlannerCapacity capacity;
        capacity.max_children = 11;
        (void)PlanIsaV1Collective(quadratic, capacity);
    }, "N-squared pair amplification is bounded before child allocation");
    suite.Throws<std::invalid_argument>([&] {
        auto huge = Make(CollTxKind::UNICAST, CollRxKind::UNICAST, 2,
                         1000000000ULL);
        IsaV1PlannerCapacity capacity;
        capacity.max_child_bytes = 1;
        capacity.max_receive_bytes_per_rank_per_wave = 1;
        capacity.max_children = 1;
        (void)PlanIsaV1Collective(huge, capacity);
    }, "oversize derived input fails by capacity without bad_alloc");
}

void TestOverflowBoundaries(Suite &suite) {
    suite.Throws<std::overflow_error>([&] {
        auto spec = Make(CollTxKind::SCATTER, CollRxKind::UNICAST, 4);
        spec.rank_records[spec.root_rank].send.base_address_bytes =
            std::numeric_limits<uint64_t>::max() - 10;
        (void)PlanIsaV1Collective(spec);
    }, "source base plus tight span overflow is rejected");
    suite.Throws<std::overflow_error>([&] {
        auto spec = Make(CollTxKind::SCATTER, CollRxKind::UNICAST, 4);
        spec.length_bytes = std::numeric_limits<uint64_t>::max() / 4 + 1;
        (void)PlanIsaV1Collective(spec);
    }, "N*L multiplication overflow is rejected before materialization");
    suite.Throws<std::overflow_error>([&] {
        auto spec = Make(CollTxKind::UNICAST, CollRxKind::UNICAST, 2, 17);
        spec.logical_fsm_id_base = std::numeric_limits<uint32_t>::max();
        IsaV1PlannerCapacity capacity;
        capacity.max_child_bytes = 16;
        capacity.max_receive_bytes_per_rank_per_wave = 16;
        (void)PlanIsaV1Collective(spec, capacity);
    }, "child fsm base plus range overflow is rejected");
    suite.Throws<std::overflow_error>([&] {
        auto spec = Make(CollTxKind::UNICAST, CollRxKind::UNICAST, 2,
                         uint64_t(std::numeric_limits<uint32_t>::max()) + 1);
        IsaV1PlannerCapacity capacity;
        capacity.max_child_bytes = 1;
        capacity.max_receive_bytes_per_rank_per_wave = 1;
        (void)PlanIsaV1Collective(spec, capacity);
    }, "child count beyond u32 is rejected before allocation");

    auto max_phase = Make(CollTxKind::UNICAST, CollRxKind::UNICAST, 2, 32768);
    IsaV1PlannerCapacity one;
    one.max_child_bytes = 1;
    one.max_receive_bytes_per_rank_per_wave = 1;
    one.max_sessions_per_rank_per_wave = 1;
    one.max_waves = 32769;
    one.max_derived_bytes = std::numeric_limits<uint64_t>::max();
    const auto max_plan = PlanIsaV1Collective(max_phase, one);
    suite.Check(max_plan.waves.size() == 32768 &&
                    max_plan.waves.back().posted_phase_id == 65534 &&
                    max_plan.waves.back().complete_phase_id == 65535,
                "u16 phase boundary accepts exactly 32768 two-phase waves");
    suite.Throws<std::overflow_error>([&] {
        auto over = max_phase;
        over.length_bytes = 32769;
        (void)PlanIsaV1Collective(over, one);
    }, "u16 phase boundary rejects wave 32769");
}

} // namespace

int RunCollPlanV1SelfTest() {
    Suite suite;
    std::cout << "==== ISA-v1 collective planner self-test ====" << std::endl;
    TestNineGrid(suite);
    TestLayoutsAndWaves(suite);
    TestContractFailures(suite);
    TestDerivedLimits(suite);
    TestOverflowBoundaries(suite);
    std::cout << "ISA-v1 collective planner self-test: "
              << (suite.failures == 0
                      ? "PASS"
                      : "FAILURES=" + std::to_string(suite.failures))
              << " (" << suite.checks << " checks)" << std::endl;
    return suite.failures;
}

#ifdef COLL_PLAN_V1_SELFTEST_MAIN
int main() { return RunCollPlanV1SelfTest(); }
#endif
