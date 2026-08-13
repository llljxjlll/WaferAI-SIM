#include "isa/collective_graph_v1.h"
#include "isa/collective_graph_v1_selftest.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
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

struct ArtifactInput {
    IsaV1CollectiveGroupRegistryView registry;
    std::vector<IsaV1NormalizedCollectiveRecord> records;
};

ArtifactInput Make(CollTxKind tx, CollRxKind rx, size_t n,
                   CollectiveKey key = {7, 10, 0},
                   uint32_t fsm_base = 0x1000,
                   uint32_t token_base = 100,
                   uint32_t record_index_base = 0) {
    static const uint16_t cores[] = {2, 7, 11, 19};
    ArtifactInput input;
    input.registry.total_cores = 32;
    input.registry.cores_per_die = 32;
    input.registry.active_cores.assign(cores, cores + n);
    input.registry.groups.push_back(
        {key.group_id, std::vector<uint16_t>(cores, cores + n)});

    const CollOp op = IsaV1CollectiveOp(tx, rx);
    const size_t root = n == 1 ? 0 : 1;
    const size_t p2p_source = 0;
    const size_t p2p_destination = n - 1;
    uint32_t next_index = record_index_base;
    auto add = [&](size_t rank, IsaV1CollectiveRecordRole role) {
        IsaV1NormalizedCollectiveRecord record;
        record.core_id = cores[rank];
        record.record_index = next_index++;
        record.role = role;
        record.tx_mode = tx;
        record.rx_mode = rx;
        if (role != IsaV1CollectiveRecordRole::REDUCE_COMPUTE) {
            record.logical_fsm_id_base = fsm_base;
            record.length_bytes = 32;
        }
        record.key = key;
        record.tree_id = 0;
        if (role == IsaV1CollectiveRecordRole::SEND) {
            record.asynchronous = true;
            record.token = token_base + static_cast<uint32_t>(rank * 4 + 1);
            record.dtype = CollDType::UINT8;
            record.reduce_op = CollReduceOp::NONE;
            record.base_address_bytes = 0x10000 + rank * 0x1000;
        } else if (role == IsaV1CollectiveRecordRole::RECEIVE) {
            record.asynchronous = true;
            record.token = token_base + static_cast<uint32_t>(rank * 4 + 2);
            record.dtype = rx == CollRxKind::REDUCE ? CollDType::INT32
                                                    : CollDType::UINT8;
            record.reduce_op = rx == CollRxKind::REDUCE
                                   ? CollReduceOp::SUM
                                   : CollReduceOp::NONE;
            record.base_address_bytes = 0x20000 + rank * 0x1000;
            record.expected_sources =
                (rx == CollRxKind::GATHER || rx == CollRxKind::REDUCE)
                    ? static_cast<uint16_t>(n - 1)
                    : 0;
        } else {
            record.asynchronous = false;
            record.token = 0;
            record.dtype = CollDType::INT32;
            record.reduce_op = CollReduceOp::SUM;
            record.element_count = 8;
            record.root_rank = op == CollOp::REDUCE
                                   ? static_cast<uint16_t>(root)
                                   : 0;
            record.self_rank = static_cast<uint16_t>(rank);
            record.base_address_bytes = 0x20000 + rank * 0x1000;
            record.result_address_bytes = 0x30000 + rank * 0x1000;
        }
        input.records.push_back(record);
    };

    for (size_t rank = 0; rank < n; ++rank) {
        const bool send = op == CollOp::P2P
                              ? rank == p2p_source
                              : (RootTransmit(op) ? rank == root : true);
        const bool receive = op == CollOp::P2P
                                 ? rank == p2p_destination
                                 : (RootReceive(op) ? rank == root : true);
        const bool compute =
            op == CollOp::REDUCE
                ? rank == root
                : (op == CollOp::REDUCESCATTER || op == CollOp::ALLREDUCE);
        if (send) add(rank, IsaV1CollectiveRecordRole::SEND);
        if (receive) add(rank, IsaV1CollectiveRecordRole::RECEIVE);
        if (compute) add(rank, IsaV1CollectiveRecordRole::REDUCE_COMPUTE);
    }
    return input;
}

uint64_t RemoteCount(CollOp op, size_t n) {
    if (op == CollOp::P2P) return n == 1 ? 0 : 1;
    if (RootTransmit(op) || RootReceive(op)) return n - 1;
    return n * (n - 1);
}

bool Contains(const std::string &text, const std::string &needle) {
    return text.find(needle) != std::string::npos;
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
            auto input = Make(cell.tx, cell.rx, n);
            const auto plans =
                BuildIsaV1CollectiveGraph(input.records, input.registry);
            suite.Check(plans.size() == 1 && plans[0].op == cell.op &&
                            plans[0].child_flows.size() ==
                                RemoteCount(cell.op, n),
                        "whole-artifact 3x3 roles close N=" +
                            std::to_string(n) + " op=" +
                            std::to_string(static_cast<int>(cell.op)));

            std::reverse(input.records.begin(), input.records.end());
            const auto reversed =
                BuildIsaV1CollectiveGraph(input.records, input.registry);
            suite.Check(plans == reversed,
                        "record-order perturbation is canonical N=" +
                            std::to_string(n) + " op=" +
                            std::to_string(static_cast<int>(cell.op)));
        }
    }
}

void TestStandaloneAndMultiInstance(Suite &suite) {
    auto keyed = Make(CollTxKind::UNICAST, CollRxKind::UNICAST, 2);
    suite.Check(BuildIsaV1CollectiveGraph(keyed.records, keyed.registry).size() ==
                    1,
                "keyed P2P child records are not mistaken for standalone P2P");

    auto cross_die_p2p = Make(CollTxKind::UNICAST, CollRxKind::UNICAST, 2);
    cross_die_p2p.registry.total_cores = 16;
    cross_die_p2p.registry.cores_per_die = 8;
    cross_die_p2p.registry.active_cores = {2, 11};
    cross_die_p2p.registry.groups[0].members = {2, 11};
    for (auto &record : cross_die_p2p.records)
        if (record.core_id == 7) record.core_id = 11;
    suite.Check(BuildIsaV1CollectiveGraph(cross_die_p2p.records,
                                         cross_die_p2p.registry)
                        .size() == 1,
                "keyed P2P remains legal across dies");

    IsaV1NormalizedCollectiveRecord standalone_send;
    standalone_send.core_id = 2;
    standalone_send.record_index = 99;
    standalone_send.role = IsaV1CollectiveRecordRole::SEND;
    standalone_send.tx_mode = CollTxKind::UNICAST;
    standalone_send.asynchronous = true;
    standalone_send.token = 9;
    standalone_send.logical_fsm_id_base = 0x9000;
    standalone_send.length_bytes = 16;
    standalone_send.dtype = CollDType::UINT8;
    standalone_send.reduce_op = CollReduceOp::NONE;
    standalone_send.peer_core = 7;
    IsaV1NormalizedCollectiveRecord standalone_receive = standalone_send;
    standalone_receive.core_id = 7;
    standalone_receive.record_index = 100;
    standalone_receive.role = IsaV1CollectiveRecordRole::RECEIVE;
    standalone_receive.token = 9;
    standalone_receive.peer_core = 2;
    IsaV1CollectiveGroupRegistryView standalone_registry;
    standalone_registry.total_cores = 32;
    standalone_registry.cores_per_die = 32;
    standalone_registry.active_cores = {2, 7};
    suite.Check(BuildIsaV1CollectiveGraph(
                    {standalone_send, standalone_receive},
                    standalone_registry).empty(),
                "matched fully-zero-key P2P is excluded and reserves one fsm session");

    auto third_endpoint = standalone_receive;
    third_endpoint.record_index = 101;
    third_endpoint.token = 10;
    suite.Throws<std::invalid_argument>([&] {
        (void)BuildIsaV1CollectiveGraph(
            {standalone_send, standalone_receive, third_endpoint},
            standalone_registry);
    }, "a third standalone endpoint cannot share one P2P session identity");

    auto first = Make(CollTxKind::SCATTER, CollRxKind::GATHER, 4,
                      {7, 20, 0}, 0x1000, 100, 0);
    auto second = Make(CollTxKind::BROADCAST, CollRxKind::GATHER, 4,
                       {7, 10, 0}, 0x2000, 1000, 100);
    first.records.insert(first.records.end(), second.records.begin(),
                         second.records.end());
    std::reverse(first.records.begin(), first.records.end());
    const auto plans =
        BuildIsaV1CollectiveGraph(first.records, first.registry);
    suite.Check(plans.size() == 2 && plans[0].key.collective_id == 10 &&
                    plans[1].key.collective_id == 20,
                "multiple instances are emitted in canonical key order");

    auto token_collision = Make(CollTxKind::SCATTER, CollRxKind::GATHER, 4,
                                {7, 21, 0}, 0x3000, 100, 200);
    auto merged = Make(CollTxKind::SCATTER, CollRxKind::GATHER, 4,
                       {7, 20, 0}, 0x1000, 100, 0);
    merged.records.insert(merged.records.end(), token_collision.records.begin(),
                          token_collision.records.end());
    suite.Throws<std::invalid_argument>([&] {
        (void)BuildIsaV1CollectiveGraph(merged.records, merged.registry);
    }, "aggregate token collisions are checked across collective instances");

    auto fsm_collision = Make(CollTxKind::SCATTER, CollRxKind::GATHER, 4,
                              {7, 21, 0}, 0x1005, 1000, 200);
    merged = Make(CollTxKind::SCATTER, CollRxKind::GATHER, 4,
                  {7, 20, 0}, 0x1000, 100, 0);
    merged.records.insert(merged.records.end(), fsm_collision.records.begin(),
                          fsm_collision.records.end());
    suite.Throws<std::invalid_argument>([&] {
        (void)BuildIsaV1CollectiveGraph(merged.records, merged.registry);
    }, "logical fsm child ranges collide across collective instances");

    auto collective = Make(CollTxKind::SCATTER, CollRxKind::GATHER, 4,
                           {7, 20, 0}, 0x1000, 100, 0);
    standalone_send.core_id = 2;
    standalone_send.record_index = 500;
    standalone_send.token = 5000;
    standalone_send.logical_fsm_id_base = 0x1003;
    standalone_send.peer_core = 7;
    standalone_receive.core_id = 7;
    standalone_receive.record_index = 501;
    standalone_receive.token = 5001;
    standalone_receive.logical_fsm_id_base = 0x1003;
    standalone_receive.peer_core = 2;
    collective.records.push_back(standalone_send);
    collective.records.push_back(standalone_receive);
    suite.Throws<std::invalid_argument>([&] {
        (void)BuildIsaV1CollectiveGraph(collective.records,
                                       collective.registry);
    }, "collective fsm ranges cannot overlap standalone P2P fsm IDs");
}

void TestRoleAndRegistryFailures(Suite &suite) {
    suite.Throws<std::invalid_argument>([&] {
        auto input = Make(CollTxKind::SCATTER, CollRxKind::GATHER, 4);
        input.records.back().length_bytes += 1;
        (void)BuildIsaV1CollectiveGraph(input.records, input.registry);
    }, "records sharing a key must agree on common fields");

    suite.Throws<std::invalid_argument>([&] {
        auto input = Make(CollTxKind::BROADCAST, CollRxKind::UNICAST, 4);
        input.records.front().tree_id = 1;
        (void)BuildIsaV1CollectiveGraph(input.records, input.registry);
    }, "P6 baseline graph rejects non-zero tree_id");

    suite.Throws<std::invalid_argument>([&] {
        auto input = Make(CollTxKind::SCATTER, CollRxKind::UNICAST, 4);
        const uint16_t missing_core = input.registry.groups[0].members.back();
        const auto found = std::find_if(
            input.records.begin(), input.records.end(),
            [&](const IsaV1NormalizedCollectiveRecord &record) {
                return record.core_id == missing_core &&
                       record.role == IsaV1CollectiveRecordRole::RECEIVE;
            });
        input.records.erase(found);
        (void)BuildIsaV1CollectiveGraph(input.records, input.registry);
    }, "missing per-rank role is rejected");

    suite.Throws<std::invalid_argument>([&] {
        auto input = Make(CollTxKind::SCATTER, CollRxKind::UNICAST, 4);
        auto duplicate = input.records.front();
        duplicate.record_index = 900;
        duplicate.token = 900;
        input.records.push_back(duplicate);
        (void)BuildIsaV1CollectiveGraph(input.records, input.registry);
    }, "duplicate per-rank role is rejected");

    suite.Throws<std::invalid_argument>([&] {
        auto input = Make(CollTxKind::SCATTER, CollRxKind::UNICAST, 4);
        auto extra_root = *std::find_if(
            input.records.begin(), input.records.end(),
            [](const IsaV1NormalizedCollectiveRecord &record) {
                return record.role == IsaV1CollectiveRecordRole::SEND;
            });
        extra_root.core_id = input.registry.groups[0].members[0];
        extra_root.record_index = 901;
        extra_root.token = 901;
        extra_root.base_address_bytes = 0x18000;
        input.records.push_back(extra_root);
        (void)BuildIsaV1CollectiveGraph(input.records, input.registry);
    }, "multiple root SEND records are rejected");

    suite.Throws<std::invalid_argument>([&] {
        auto input = Make(CollTxKind::UNICAST, CollRxKind::REDUCE, 4);
        const auto found = std::find_if(
            input.records.begin(), input.records.end(),
            [](const IsaV1NormalizedCollectiveRecord &record) {
                return record.role ==
                       IsaV1CollectiveRecordRole::REDUCE_COMPUTE;
            });
        input.records.erase(found);
        (void)BuildIsaV1CollectiveGraph(input.records, input.registry);
    }, "ordinary Reduce requires exactly one root compute record");

    suite.Throws<std::invalid_argument>([&] {
        auto input = Make(CollTxKind::SCATTER, CollRxKind::REDUCE, 4);
        auto &compute = *std::find_if(
            input.records.begin(), input.records.end(),
            [](const IsaV1NormalizedCollectiveRecord &record) {
                return record.role ==
                       IsaV1CollectiveRecordRole::REDUCE_COMPUTE;
            });
        compute.base_address_bytes += 4;
        (void)BuildIsaV1CollectiveGraph(input.records, input.registry);
    }, "compute staging address must equal its RECEIVE base");

    suite.Throws<std::invalid_argument>([&] {
        auto input = Make(CollTxKind::BROADCAST, CollRxKind::REDUCE, 4);
        auto &compute = *std::find_if(
            input.records.begin(), input.records.end(),
            [](const IsaV1NormalizedCollectiveRecord &record) {
                return record.role ==
                       IsaV1CollectiveRecordRole::REDUCE_COMPUTE;
            });
        ++compute.element_count;
        (void)BuildIsaV1CollectiveGraph(input.records, input.registry);
    }, "compute element_count*dtype_bytes must equal L");

    suite.Throws<std::invalid_argument>([&] {
        auto input = Make(CollTxKind::UNICAST, CollRxKind::REDUCE, 4);
        auto &compute = *std::find_if(
            input.records.begin(), input.records.end(),
            [](const IsaV1NormalizedCollectiveRecord &record) {
                return record.role ==
                       IsaV1CollectiveRecordRole::REDUCE_COMPUTE;
            });
        compute.root_rank = 0;
        (void)BuildIsaV1CollectiveGraph(input.records, input.registry);
    }, "ordinary Reduce compute root_rank must match the unique root");

    suite.Throws<std::invalid_argument>([&] {
        auto input = Make(CollTxKind::BROADCAST, CollRxKind::REDUCE, 4);
        auto &compute = *std::find_if(
            input.records.begin(), input.records.end(),
            [](const IsaV1NormalizedCollectiveRecord &record) {
                return record.role ==
                       IsaV1CollectiveRecordRole::REDUCE_COMPUTE;
            });
        compute.self_rank = static_cast<uint16_t>(compute.self_rank + 1);
        (void)BuildIsaV1CollectiveGraph(input.records, input.registry);
    }, "symmetric reduction compute self_rank must match its core rank");

    suite.Throws<std::invalid_argument>([&] {
        auto input = Make(CollTxKind::SCATTER, CollRxKind::REDUCE, 4);
        auto &compute = *std::find_if(
            input.records.begin(), input.records.end(),
            [](const IsaV1NormalizedCollectiveRecord &record) {
                return record.role ==
                       IsaV1CollectiveRecordRole::REDUCE_COMPUTE;
            });
        compute.logical_fsm_id_base = 9;
        (void)BuildIsaV1CollectiveGraph(input.records, input.registry);
    }, "compute fsm is synthesized from RECEIVE and must be absent externally");

    suite.Throws<std::invalid_argument>([&] {
        auto input = Make(CollTxKind::UNICAST, CollRxKind::GATHER, 4);
        auto &receive = *std::find_if(
            input.records.begin(), input.records.end(),
            [](const IsaV1NormalizedCollectiveRecord &record) {
                return record.role == IsaV1CollectiveRecordRole::RECEIVE;
            });
        receive.expected_sources = 4;
        (void)BuildIsaV1CollectiveGraph(input.records, input.registry);
    }, "graph derives and enforces expected_sources=N-1");

    suite.Throws<std::invalid_argument>([&] {
        auto input = Make(CollTxKind::SCATTER, CollRxKind::GATHER, 2);
        input.registry.total_cores = 16;
        input.registry.cores_per_die = 8;
        input.registry.active_cores = {2, 11};
        input.registry.groups[0].members = {2, 11};
        for (auto &record : input.records)
            if (record.core_id == 7) record.core_id = 11;
        (void)BuildIsaV1CollectiveGraph(input.records, input.registry);
    }, "non-P2P collective groups cannot cross dies");

    suite.Throws<std::invalid_argument>([&] {
        auto input = Make(CollTxKind::SCATTER, CollRxKind::UNICAST, 4);
        input.registry.active_cores.pop_back();
        (void)BuildIsaV1CollectiveGraph(input.records, input.registry);
    }, "all collective members must be active");
}

void TestPlannerDiagnostics(Suite &suite) {
    auto input = Make(CollTxKind::SCATTER, CollRxKind::UNICAST, 4,
                      {7, 88, 3});
    auto &root_send = *std::find_if(
        input.records.begin(), input.records.end(),
        [](const IsaV1NormalizedCollectiveRecord &record) {
            return record.role == IsaV1CollectiveRecordRole::SEND;
        });
    root_send.base_address_bytes = std::numeric_limits<uint64_t>::max() - 4;
    bool contextual = false;
    try {
        (void)BuildIsaV1CollectiveGraph(input.records, input.registry);
    } catch (const std::overflow_error &error) {
        const std::string message = error.what();
        contextual = Contains(message, "key(7,88,3)") &&
                     Contains(message, "core") && Contains(message, "record");
    }
    suite.Check(contextual,
                "planner failures retain collective key/core/record context");

    bool canonical_context = false;
    try {
        auto malformed = Make(CollTxKind::BROADCAST,
                              CollRxKind::UNICAST, 4, {7, 89, 4});
        malformed.records.front().tree_id = 1;
        (void)BuildIsaV1CollectiveGraph(malformed.records,
                                       malformed.registry);
    } catch (const std::invalid_argument &error) {
        const std::string message = error.what();
        canonical_context = Contains(message, "key(7,89,4)") &&
                            Contains(message, "core") &&
                            Contains(message, "record");
    }
    suite.Check(canonical_context,
                "per-record canonical failures retain key/core/record context");

}

} // namespace

int RunCollectiveGraphV1SelfTest() {
    Suite suite;
    std::cout << "==== ISA-v1 whole-artifact collective graph self-test ===="
              << std::endl;
    TestNineGrid(suite);
    TestStandaloneAndMultiInstance(suite);
    TestRoleAndRegistryFailures(suite);
    TestPlannerDiagnostics(suite);
    std::cout << "ISA-v1 whole-artifact collective graph self-test: "
              << (suite.failures == 0
                      ? "PASS"
                      : "FAILURES=" + std::to_string(suite.failures))
              << " (" << suite.checks << " checks)" << std::endl;
    return suite.failures;
}

#ifdef COLLECTIVE_GRAPH_V1_SELFTEST_MAIN
int main() { return RunCollectiveGraphV1SelfTest(); }
#endif
