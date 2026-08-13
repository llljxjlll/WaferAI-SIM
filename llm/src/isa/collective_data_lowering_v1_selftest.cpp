#include "isa/collective_data_lowering_v1.h"
#include "isa/collective_data_lowering_v1_selftest.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Suite {
    int checks = 0;
    int failures = 0;

    void Check(bool condition, const std::string &name) {
        ++checks;
        if (condition) return;
        ++failures;
        std::cerr << "[COLLECTIVE DATA LOWERING V1] FAIL: "
                  << name << '\n';
    }

    template <class F>
    void Rejects(F &&fn, const std::string &name) {
        bool matched = false;
        try {
            fn();
        } catch (const RecordLoweringError &) {
            matched = true;
        } catch (...) {
        }
        Check(matched, name);
    }
};

IsaV1CollectiveSpec MakeLocalSpec() {
    IsaV1CollectiveSpec spec;
    spec.tx_kind = CollTxKind::UNICAST;
    spec.rx_kind = CollRxKind::UNICAST;
    spec.key = {3, 5, 7};
    spec.group = {9};
    spec.root_rank = 0;
    spec.p2p_source_rank = 0;
    spec.p2p_destination_rank = 0;
    spec.length_bytes = 13;
    spec.logical_fsm_id_base = 0x10000;
    spec.rank_records.resize(1);
    auto &records = spec.rank_records[0];
    records.send = {true, true, 11, 0x100};
    records.receive = {true, true, 12, 0x200};
    return spec;
}

IsaV1CollectiveSpec MakeReduceSpec() {
    IsaV1CollectiveSpec spec;
    spec.tx_kind = CollTxKind::UNICAST;
    spec.rx_kind = CollRxKind::REDUCE;
    spec.key = {13, 17, 19};
    spec.group = {2, 7, 11};
    spec.root_rank = 1;
    spec.p2p_source_rank = 0;
    spec.p2p_destination_rank = 0;
    spec.length_bytes = 16;
    spec.dtype = CollDType::INT32;
    spec.reduce_op = CollReduceOp::MAX;
    spec.expected_sources = 2;
    spec.logical_fsm_id_base = 0x20000;
    spec.rank_records.resize(3);
    for (size_t rank = 0; rank < spec.rank_records.size(); ++rank) {
        auto &records = spec.rank_records[rank];
        records.send = {true, true,
                        static_cast<uint32_t>(100 + rank),
                        0x1000 + rank * 0x100};
    }
    auto &root = spec.rank_records[1];
    root.receive = {true, true, 201, 0x4000};
    root.result_address_bytes = 0x5000;
    return spec;
}

IsaV1CollectiveArtifactLowering WrapPlan(IsaV1CollectivePlan plan) {
    IsaV1CollectiveArtifactLowering lowering;
    lowering.plans.push_back(std::move(plan));
    const IsaV1CollectivePlan &stored = lowering.plans.front();
    for (size_t rank = 0; rank < stored.actions_by_rank.size(); ++rank) {
        IsaV1CoreCollectiveActionStream stream;
        stream.core_id = stored.group[rank];
        for (const IsaV1Action &action : stored.actions_by_rank[rank]) {
            IsaV1LoweredCollectiveAction lowered;
            lowered.plan_index = 0;
            lowered.key = stored.key;
            lowered.action = action;
            stream.actions.push_back(lowered);
        }
        lowering.core_actions.push_back(std::move(stream));
    }
    return lowering;
}

size_t FindAction(const IsaV1CollectiveArtifactLowering &lowering,
                  uint16_t core, IsaV1ActionKind kind) {
    const auto stream = std::find_if(
        lowering.core_actions.begin(), lowering.core_actions.end(),
        [&](const IsaV1CoreCollectiveActionStream &candidate) {
            return candidate.core_id == core;
        });
    if (stream == lowering.core_actions.end())
        throw std::logic_error("test core stream is missing");
    const auto action = std::find_if(
        stream->actions.begin(), stream->actions.end(),
        [&](const IsaV1LoweredCollectiveAction &candidate) {
            return candidate.action.kind == kind;
        });
    if (action == stream->actions.end())
        throw std::logic_error("test action is missing");
    return static_cast<size_t>(action - stream->actions.begin());
}

IsaV1CoreCollectiveActionStream &CoreStream(
    IsaV1CollectiveArtifactLowering &lowering, uint16_t core) {
    const auto stream = std::find_if(
        lowering.core_actions.begin(), lowering.core_actions.end(),
        [&](const IsaV1CoreCollectiveActionStream &candidate) {
            return candidate.core_id == core;
        });
    if (stream == lowering.core_actions.end())
        throw std::logic_error("test core stream is missing");
    return *stream;
}

bool SamePrimFields(const Collective_data_v1_prim &left,
                    const Collective_data_v1_prim &right) {
    return left.mode == right.mode && left.key == right.key &&
           left.phase_id == right.phase_id &&
           left.source_address_bytes == right.source_address_bytes &&
           left.destination_address_bytes ==
               right.destination_address_bytes &&
           left.length_bytes == right.length_bytes &&
           left.input_count == right.input_count &&
           left.dtype == right.dtype && left.reduce_op == right.reduce_op;
}

void TestMaterialization(Suite &suite) {
    auto local = WrapPlan(PlanIsaV1Collective(MakeLocalSpec()));
    const size_t local_index =
        FindAction(local, 9, IsaV1ActionKind::LOCAL_COPY);
    const auto local_prim =
        MaterializeIsaV1CollectiveDataAction(local, 9, local_index);
    suite.Check(!local.executable &&
                    local_prim->mode ==
                        CollectiveDataV1PrimMode::LOCAL_COPY &&
                    local_prim->key == local.plans[0].key &&
                    local_prim->phase_id == 0 &&
                    local_prim->source_address_bytes == 0x100 &&
                    local_prim->destination_address_bytes == 0x200 &&
                    local_prim->length_bytes == 13 &&
                    local_prim->input_count == 1 &&
                    local_prim->dtype == CollDType::UINT8 &&
                    local_prim->reduce_op == CollReduceOp::NONE,
                "canonical LOCAL_COPY maps to untracked N=1 ID58 fields");
    const auto local_wire = local_prim->serialize();
    Collective_data_v1_prim local_roundtrip;
    local_roundtrip.deserialize(local_wire);
    suite.Check(SamePrimFields(*local_prim, local_roundtrip),
                "LOCAL_COPY strict serialize/deserialize preserves every field");

    auto reduce = WrapPlan(PlanIsaV1Collective(MakeReduceSpec()));
    const size_t reduce_copy_index =
        FindAction(reduce, 7, IsaV1ActionKind::LOCAL_COPY);
    const auto reduce_copy =
        MaterializeIsaV1CollectiveDataAction(reduce, 7,
                                             reduce_copy_index);
    suite.Check(reduce_copy->mode ==
                        CollectiveDataV1PrimMode::LOCAL_COPY &&
                    reduce_copy->source_address_bytes == 0x1100 &&
                    reduce_copy->destination_address_bytes == 0x4010 &&
                    reduce_copy->length_bytes == 16 &&
                    reduce_copy->input_count == 1 &&
                    reduce_copy->dtype == CollDType::UINT8 &&
                    reduce_copy->reduce_op == CollReduceOp::NONE,
                "reduction diagonal maps to its tight staging slice as raw copy");
    const size_t reduce_index =
        FindAction(reduce, 7, IsaV1ActionKind::REDUCE_COMPUTE);
    const auto reduce_prim =
        MaterializeIsaV1CollectiveDataAction(reduce, 7, reduce_index);
    const IsaV1Action &reduce_action =
        CoreStream(reduce, 7).actions[reduce_index].action;
    suite.Check(!reduce.executable &&
                    reduce_prim->mode ==
                        CollectiveDataV1PrimMode::REDUCE &&
                    reduce_prim->key == reduce.plans[0].key &&
                    reduce_prim->phase_id == reduce_action.phase_id &&
                    reduce_prim->source_address_bytes == 0x4000 &&
                    reduce_prim->destination_address_bytes == 0x5000 &&
                    reduce_prim->length_bytes == 16 &&
                    reduce_prim->input_count == 3 &&
                    reduce_prim->dtype == CollDType::INT32 &&
                    reduce_prim->reduce_op == CollReduceOp::MAX,
                "canonical REDUCE_COMPUTE maps staging/result/N/dtype/op");
    const auto reduce_wire = reduce_prim->serialize();
    Collective_data_v1_prim reduce_roundtrip;
    reduce_roundtrip.deserialize(reduce_wire);
    suite.Check(SamePrimFields(*reduce_prim, reduce_roundtrip),
                "REDUCE strict serialize/deserialize preserves every field");
}

void TestEnvelopeRejections(Suite &suite) {
    const auto base = WrapPlan(PlanIsaV1Collective(MakeLocalSpec()));
    const size_t local_index =
        FindAction(base, 9, IsaV1ActionKind::LOCAL_COPY);
    const size_t barrier_index =
        FindAction(base, 9, IsaV1ActionKind::POSTED_BARRIER);
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectiveDataAction(
                  base, 9, barrier_index); },
        "non-data action kind is rejected");
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectiveDataAction(
                  base, 9, std::numeric_limits<size_t>::max()); },
        "action stream index is rejected");
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectiveDataAction(
                  base, 8, local_index); },
        "unknown executing core is rejected");

    auto bad = base;
    CoreStream(bad, 9).actions[local_index].plan_index = 1;
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectiveDataAction(
                  bad, 9, local_index); },
        "wrong plan index is rejected");
    bad = base;
    CoreStream(bad, 9).actions[local_index].key.epoch++;
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectiveDataAction(
                  bad, 9, local_index); },
        "wrong plan key is rejected");
    bad = base;
    CoreStream(bad, 9).actions[local_index].action.core = 8;
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectiveDataAction(
                  bad, 9, local_index); },
        "wrong action core is rejected");
    bad = base;
    CoreStream(bad, 9).actions[local_index].action.phase_id++;
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectiveDataAction(
                  bad, 9, local_index); },
        "wrong action phase is rejected");
    bad = base;
    CoreStream(bad, 9).actions[local_index].action.item_index++;
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectiveDataAction(
                  bad, 9, local_index); },
        "wrong action item index is rejected");
    bad = base;
    bad.plans[0].actions_by_rank[0].push_back(
        bad.plans[0].actions_by_rank[0][0]);
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectiveDataAction(
                  bad, 9, local_index); },
        "duplicate canonical plan action is rejected");
}

void TestPlanItemRejections(Suite &suite) {
    const auto local = WrapPlan(PlanIsaV1Collective(MakeLocalSpec()));
    const size_t local_index =
        FindAction(local, 9, IsaV1ActionKind::LOCAL_COPY);
    auto bad_local = local;
    bad_local.plans[0].local_copies[0].source_address_bytes++;
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectiveDataAction(
                  bad_local, 9, local_index); },
        "local-copy absolute source mismatch is rejected");
    bad_local = local;
    bad_local.plans[0].local_copies[0].length_bytes++;
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectiveDataAction(
                  bad_local, 9, local_index); },
        "local-copy L mismatch is rejected");
    bad_local = local;
    bad_local.plans[0].rank_records[0].send.base_address_bytes =
        std::numeric_limits<uint64_t>::max();
    bad_local.plans[0].local_copies[0].source_offset_bytes = 1;
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectiveDataAction(
                  bad_local, 9, local_index); },
        "local-copy absolute-address overflow is rejected");

    const auto reduce = WrapPlan(PlanIsaV1Collective(MakeReduceSpec()));
    const size_t reduce_index =
        FindAction(reduce, 7, IsaV1ActionKind::REDUCE_COMPUTE);
    auto bad_reduce = reduce;
    bad_reduce.plans[0].reduce_targets[0].input_count = 2;
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectiveDataAction(
                  bad_reduce, 7, reduce_index); },
        "reduce input_count mismatch is rejected");
    bad_reduce = reduce;
    bad_reduce.plans[0].reduce_targets[0].dtype = CollDType::FP32;
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectiveDataAction(
                  bad_reduce, 7, reduce_index); },
        "reduce unsupported dtype is rejected");
    bad_reduce = reduce;
    bad_reduce.plans[0].reduce_targets[0].reduce_op = CollReduceOp::NONE;
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectiveDataAction(
                  bad_reduce, 7, reduce_index); },
        "reduce invalid op is rejected");
    bad_reduce = reduce;
    bad_reduce.plans[0].reduce_targets[0].result_address_bytes++;
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectiveDataAction(
                  bad_reduce, 7, reduce_index); },
        "reduce result absolute address mismatch is rejected");
    bad_reduce = reduce;
    bad_reduce.plans[0].reduce_targets[0].element_count =
        std::numeric_limits<uint64_t>::max();
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectiveDataAction(
                  bad_reduce, 7, reduce_index); },
        "reduce element byte-size overflow is rejected");
}

} // namespace

int RunIsaV1CollectiveDataLoweringSelfTest() {
    Suite suite;
    TestMaterialization(suite);
    TestEnvelopeRejections(suite);
    TestPlanItemRejections(suite);
    if (suite.failures == 0) {
        std::cout << "[COLLECTIVE DATA LOWERING V1] PASS ("
                  << suite.checks << " checks)\n";
        return 0;
    }
    std::cerr << "[COLLECTIVE DATA LOWERING V1] FAIL ("
              << suite.failures << "/" << suite.checks
              << " checks failed)\n";
    return suite.failures;
}

#ifdef COLLECTIVE_DATA_LOWERING_V1_SELFTEST_MAIN
int sc_main(int, char **) {
    return RunIsaV1CollectiveDataLoweringSelfTest();
}
#endif
