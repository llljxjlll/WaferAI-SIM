#include "isa/collective_program_v1.h"
#include "isa/collective_program_v1_selftest.h"

#include "utils/prim_utils.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <set>
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
        if (condition) return;
        ++failures;
        std::cerr << "[COLLECTIVE PROGRAM V1] FAIL: " << name << '\n';
    }

    template <class F>
    void Rejects(F &&fn, const std::string &name) {
        bool rejected = false;
        try {
            fn();
        } catch (const RecordLoweringError &) {
            rejected = true;
        } catch (...) {
        }
        Check(rejected, name);
    }
};

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

SramAddressOperand Absolute(uint64_t address) {
    SramAddressOperand operand;
    operand.kind = SramAddressKind::ABSOLUTE;
    operand.absolute_address_bytes = address;
    return operand;
}

bool SendRole(CollOp op, std::size_t rank, std::size_t root) {
    if (op == CollOp::P2P) return rank == 0;
    return op == CollOp::SCATTER || op == CollOp::BROADCAST
               ? rank == root
               : true;
}

bool ReceiveRole(CollOp op, std::size_t rank, std::size_t root) {
    if (op == CollOp::P2P) return rank == root;
    return op == CollOp::GATHER || op == CollOp::REDUCE
               ? rank == root
               : true;
}

bool ComputeRole(CollOp op, std::size_t rank, std::size_t root) {
    return op == CollOp::REDUCE
               ? rank == root
               : op == CollOp::REDUCESCATTER ||
                     op == CollOp::ALLREDUCE;
}

ProgramArtifact CollectiveArtifact(CollTxKind tx, CollRxKind rx,
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
        if (SendRole(op, rank, root)) {
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
        if (ReceiveRole(op, rank, root)) {
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
        if (ComputeRole(op, rank, root)) {
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

IsaV1CollectiveProgramImageConfig Config() {
    IsaV1CollectiveProgramImageConfig config;
    config.total_cores = 8;
    config.cores_per_die = 8;
    config.generation = 0x123456789abcdef0ULL;
    return config;
}

IsaV1CollectiveProgramImage Build(
    const ProgramArtifact &artifact,
    const IsaV1CollectiveProgramImageConfig &config = Config()) {
    const auto lowering = LowerIsaV1CollectiveArtifact(
        artifact, config.total_cores, config.cores_per_die,
        config.planner_capacity);
    return BuildIsaV1CollectiveProgramImage(artifact, lowering, config);
}

std::size_t ActionCount(
    const IsaV1CollectiveArtifactLowering &lowering) {
    std::size_t count = 0;
    for (const auto &stream : lowering.core_actions)
        count += stream.actions.size();
    return count;
}

bool SemanticIssuesEqual(
    const std::vector<IsaV1CollectiveIssueSite> &left,
    const std::vector<IsaV1CollectiveIssueSite> &right) {
    if (left.size() != right.size()) return false;
    auto key = [](const IsaV1CollectiveIssueSite &site) {
        return std::make_tuple(site.core_id, site.plan_index, site.role,
                               site.public_token, site.key);
    };
    std::vector<decltype(key(left.front()))> left_keys;
    std::vector<decltype(key(right.front()))> right_keys;
    for (const auto &site : left) left_keys.push_back(key(site));
    for (const auto &site : right) right_keys.push_back(key(site));
    std::sort(left_keys.begin(), left_keys.end());
    std::sort(right_keys.begin(), right_keys.end());
    return left_keys == right_keys;
}

void TestNineGrid(Suite &suite) {
    struct Cell { CollTxKind tx; CollRxKind rx; };
    const std::array<Cell, 9> cells = {{
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
    for (std::size_t n : {std::size_t{1}, std::size_t{2},
                          std::size_t{4}}) {
        for (const Cell &cell : cells) {
            const ProgramArtifact artifact =
                CollectiveArtifact(cell.tx, cell.rx, n);
            const auto image = Build(artifact);
            suite.Check(image.Generation() == Config().generation &&
                            image.Cookie() != 0 &&
                            image.Lowering().plans.size() == 1 &&
                            !image.Lowering().executable &&
                            image.Cores().size() == n &&
                            image.IssueSites().size() > 0 &&
                            image.AdmissionCapacity().wave_demands ==
                                image.WaveDemands().size() &&
                            image.AdmissionCapacity().waves ==
                                image.Lowering().plans[0].waves.size(),
                        "nine-grid N=1/2/4 image is bounded and remains an executable candidate");
            suite.Check(image.FindCore(0) != nullptr &&
                            image.FindCore(7) == nullptr &&
                            image.FindCore(0)->actions.size() ==
                                image.Lowering().core_actions[0].actions.size() &&
                            ActionCount(image.Lowering()) ==
                                [&]() {
                                    std::size_t count = 0;
                                    for (const auto &core : image.Cores())
                                        count += core.actions.size();
                                    return count;
                                }(),
                        "every canonical action has one typed implementation");
        }
    }
}

void TestDeterminismAndIssueSites(Suite &suite) {
    ProgramArtifact artifact = CollectiveArtifact(
        CollTxKind::BROADCAST, CollRxKind::REDUCE, 4);
    const auto first = Build(artifact);
    const auto repeated = Build(artifact);
    suite.Check(first == repeated &&
                    first.Cookie() == repeated.Cookie(),
                "identical artifact yields identical immutable image/cookie");

    ProgramArtifact record_order = artifact;
    for (ProgramCore &core : record_order.cores)
        std::reverse(core.records.begin(), core.records.end());
    const auto reordered_records = Build(record_order);
    suite.Check(first.Lowering() == reordered_records.Lowering() &&
                    first.Cores() == reordered_records.Cores() &&
                    first.WaveDemands() ==
                        reordered_records.WaveDemands() &&
                    SemanticIssuesEqual(first.IssueSites(),
                                        reordered_records.IssueSites()),
                "record order preserves semantic plans/actions/issue mapping while retaining record indices");

    ProgramArtifact ordinary = artifact;
    DteIssueOperands issue;
    issue.direction = LocalDteDirection::SPM_TO_SPM;
    issue.token = UINT32_MAX;
    issue.payload_bits = 8;
    issue.size_bytes = 1;
    issue.source_sram = Absolute(1);
    issue.destination_sram = Absolute(2);
    ordinary.cores[0].records.push_back({Opcode::DTE_ISSUE, issue});
    const auto with_ordinary = Build(ordinary);
    const auto *core = with_ordinary.FindCore(0);
    suite.Check(core != nullptr &&
                    core->ordinary_reserved_tokens ==
                        std::set<uint32_t>{UINT32_MAX} &&
                    core->capacity.aggregate.ordinary_reserved_tokens == 1,
                "ordinary issue/control tokens are reserved deterministically");
}

void TestIssueAndLoweringFailures(Suite &suite) {
    ProgramArtifact base = CollectiveArtifact(
        CollTxKind::SCATTER, CollRxKind::GATHER, 4);
    const auto lowering = LowerIsaV1CollectiveArtifact(base, 8, 8);

    ProgramArtifact missing = base;
    missing.cores[1].records.pop_back();
    suite.Rejects(
        [&] { (void)BuildIsaV1CollectiveProgramImage(
                  missing, lowering, Config()); },
        "missing external issue record cannot match the canonical lowering");

    ProgramArtifact duplicate = base;
    duplicate.cores[1].records.push_back(
        duplicate.cores[1].records.front());
    suite.Rejects(
        [&] { (void)BuildIsaV1CollectiveProgramImage(
                  duplicate, lowering, Config()); },
        "duplicate external issue record is rejected atomically");

    auto tampered = lowering;
    tampered.core_actions.front().actions.pop_back();
    suite.Rejects(
        [&] { (void)BuildIsaV1CollectiveProgramImage(
                  base, tampered, Config()); },
        "non-canonical action stream cannot bypass whole-artifact rebuilding");

    ProgramArtifact collides = base;
    DteIssueOperands issue;
    issue.direction = LocalDteDirection::SPM_TO_SPM;
    issue.token = 101;
    issue.payload_bits = 8;
    issue.size_bytes = 1;
    issue.source_sram = Absolute(1);
    issue.destination_sram = Absolute(2);
    collides.cores[0].records.push_back({Opcode::DTE_ISSUE, issue});
    suite.Rejects(
        [&] { (void)Build(collides); },
        "ordinary DTE token collision with collective aggregate is rejected");
}

void TestCapacityAndRollback(Suite &suite) {
    const ProgramArtifact artifact = CollectiveArtifact(
        CollTxKind::BROADCAST, CollRxKind::REDUCE, 4);
    const auto committed = Build(artifact);
    const uint64_t cookie = committed.Cookie();
    const auto cores = committed.Cores();
    const auto demands = committed.WaveDemands();

    auto reject_limit = [&](IsaV1CollectiveProgramImageConfig config,
                            const std::string &name) {
        suite.Rejects([&] { (void)Build(artifact, config); }, name);
        suite.Check(committed.Cookie() == cookie &&
                        committed.Cores() == cores &&
                        committed.WaveDemands() == demands,
                    name + " leaves prior image unchanged");
    };

    auto config = Config();
    config.limits.max_plans = 0;
    reject_limit(config, "zero plan capacity is rejected");
    config = Config();
    config.limits.max_issue_sites = 1;
    reject_limit(config, "issue-site capacity exhaustion is rejected");
    config = Config();
    config.limits.max_actions = 1;
    reject_limit(config, "action capacity exhaustion is rejected");
    config = Config();
    config.limits.max_waves = 1;
    config.planner_capacity.max_sessions_per_rank_per_wave = 1;
    reject_limit(config, "wave capacity exhaustion is rejected");
    config = Config();
    config.limits.max_wave_demands = 1;
    reject_limit(config, "wave-demand capacity exhaustion is rejected");
    config = Config();
    config.limits.max_aggregate_tokens_per_core = 1;
    reject_limit(config, "aggregate-token capacity exhaustion is rejected");
    config = Config();
    config.limits.max_child_tokens_per_core = 1;
    reject_limit(config, "child-token capacity exhaustion is rejected");
    config = Config();
    config.limits.max_local_work_items_per_core = 1;
    reject_limit(config, "local-work capacity exhaustion is rejected");
    config = Config();
    config.limits.max_endpoint_sessions_per_core_wave = 1;
    reject_limit(config, "endpoint-session admission exhaustion is rejected");
    config = Config();
    config.limits.max_receive_bytes_per_core_wave = 1;
    reject_limit(config, "receive-byte admission exhaustion is rejected");
    config = Config();
    config.generation = 0;
    reject_limit(config, "zero generation is rejected");
    config = Config();
    config.total_cores = UINT16_MAX + 2U;
    config.cores_per_die = 1;
    reject_limit(config, "topology/core wire overflow is rejected");
}

} // namespace

int RunIsaV1CollectiveProgramImageSelfTest() {
    StrictWireScope strict;
    Suite suite;
    TestNineGrid(suite);
    TestDeterminismAndIssueSites(suite);
    TestIssueAndLoweringFailures(suite);
    TestCapacityAndRollback(suite);
    if (suite.failures == 0) {
        std::cout << "[COLLECTIVE PROGRAM V1] PASS ("
                  << suite.checks << " checks)\n";
        return 0;
    }
    std::cerr << "[COLLECTIVE PROGRAM V1] FAIL ("
              << suite.failures << "/" << suite.checks
              << " checks failed)\n";
    return suite.failures;
}

#ifdef COLLECTIVE_PROGRAM_V1_SELFTEST_MAIN
int sc_main(int, char **) {
    return RunIsaV1CollectiveProgramImageSelfTest();
}
#endif
