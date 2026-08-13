#include "isa/collective_phase_lowering_v1.h"
#include "isa/collective_phase_lowering_v1_selftest.h"

#include "dte/coll_plan_v1.h"
#include "utils/prim_utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

struct Suite {
    int checks = 0;
    int failures = 0;

    void Check(bool condition, const std::string &name) {
        ++checks;
        if (condition) return;
        ++failures;
        std::cerr << "[COLLECTIVE PHASE LOWERING V1] FAIL: "
                  << name << '\n';
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

IsaV1CollectiveArtifactLowering MakeLowering() {
    IsaV1CollectiveSpec spec;
    spec.tx_kind = CollTxKind::UNICAST;
    spec.rx_kind = CollRxKind::UNICAST;
    spec.key = {7, 11, 13};
    spec.group = {2, 9};
    spec.root_rank = 0;
    spec.p2p_source_rank = 0;
    spec.p2p_destination_rank = 1;
    spec.length_bytes = 32;
    spec.logical_fsm_id_base = 0x1200;
    spec.rank_records.resize(2);
    spec.rank_records[0].send = {true, true, 101, 0x1000};
    spec.rank_records[1].receive = {true, true, 202, 0x4000};

    IsaV1CollectiveArtifactLowering lowering;
    lowering.plans.push_back(PlanIsaV1Collective(spec));
    const IsaV1CollectivePlan &plan = lowering.plans.front();
    for (std::size_t rank = 0; rank < plan.group.size(); ++rank) {
        IsaV1CoreCollectiveActionStream stream;
        stream.core_id = plan.group[rank];
        for (const IsaV1Action &action : plan.actions_by_rank[rank]) {
            IsaV1LoweredCollectiveAction lowered;
            lowered.plan_index = 0;
            lowered.key = plan.key;
            lowered.action = action;
            stream.actions.push_back(lowered);
        }
        lowering.core_actions.push_back(std::move(stream));
    }
    return lowering;
}

IsaV1CoreCollectiveActionStream &CoreStream(
    IsaV1CollectiveArtifactLowering &lowering, uint16_t core) {
    const auto found = std::find_if(
        lowering.core_actions.begin(), lowering.core_actions.end(),
        [core](const IsaV1CoreCollectiveActionStream &stream) {
            return stream.core_id == core;
        });
    if (found == lowering.core_actions.end())
        throw std::logic_error("test core stream is missing");
    return *found;
}

const IsaV1CoreCollectiveActionStream &CoreStream(
    const IsaV1CollectiveArtifactLowering &lowering, uint16_t core) {
    const auto found = std::find_if(
        lowering.core_actions.begin(), lowering.core_actions.end(),
        [core](const IsaV1CoreCollectiveActionStream &stream) {
            return stream.core_id == core;
        });
    if (found == lowering.core_actions.end())
        throw std::logic_error("test core stream is missing");
    return *found;
}

std::size_t FindAction(const IsaV1CollectiveArtifactLowering &lowering,
                       uint16_t core, IsaV1ActionKind kind) {
    const auto &stream = CoreStream(lowering, core);
    const auto found = std::find_if(
        stream.actions.begin(), stream.actions.end(),
        [kind](const IsaV1LoweredCollectiveAction &lowered) {
            return lowered.action.kind == kind;
        });
    if (found == stream.actions.end())
        throw std::logic_error("test action is missing");
    return static_cast<std::size_t>(found - stream.actions.begin());
}

bool SameFields(const Collective_phase_barrier_v1_prim &left,
                const Collective_phase_barrier_v1_prim &right) {
    return left.key == right.key && left.phase_id == right.phase_id &&
           left.rank == right.rank &&
           left.group_size == right.group_size &&
           left.release_tree_id == right.release_tree_id;
}

void TestMaterialization(Suite &suite) {
    StrictWireScope strict;
    const auto lowering = MakeLowering();
    for (const IsaV1ActionKind kind :
         {IsaV1ActionKind::POSTED_BARRIER,
          IsaV1ActionKind::COMPLETE_BARRIER}) {
        const std::size_t index = FindAction(lowering, 2, kind);
        const IsaV1Action &action =
            CoreStream(lowering, 2).actions[index].action;
        const auto prim = MaterializeIsaV1CollectivePhaseBarrier(
            lowering, 2, index);
        suite.Check(prim->key == lowering.plans[0].key &&
                        prim->phase_id == action.phase_id &&
                        prim->rank == 0 && prim->group_size == 2 &&
                        prim->release_tree_id == 0,
                    "canonical phase action maps to baseline ID59 fields");
        const auto wire = prim->serialize();
        Collective_phase_barrier_v1_prim decoded;
        decoded.deserialize(wire);
        suite.Check(SameFields(*prim, decoded) &&
                        decoded.serialize() == wire,
                    "ID59 strict roundtrip preserves every field");
    }
    suite.Check(!lowering.executable,
                "phase materialization does not mark lowering executable");
}

void TestEnvelopeRejections(Suite &suite) {
    const auto base = MakeLowering();
    const std::size_t barrier =
        FindAction(base, 2, IsaV1ActionKind::POSTED_BARRIER);
    const std::size_t endpoint =
        FindAction(base, 2, IsaV1ActionKind::ISSUE_SEND);
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectivePhaseBarrier(
                  base, 2, endpoint); },
        "non-barrier action is rejected");
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectivePhaseBarrier(
                  base, 2, std::numeric_limits<std::size_t>::max()); },
        "out-of-range action index is rejected");
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectivePhaseBarrier(
                  base, 8, barrier); },
        "unknown executing core is rejected");

    auto bad = base;
    bad.core_actions.push_back(CoreStream(base, 2));
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectivePhaseBarrier(
                  bad, 2, barrier); },
        "duplicate core stream is rejected");
    bad = base;
    CoreStream(bad, 2).actions.pop_back();
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectivePhaseBarrier(
                  bad, 2, barrier); },
        "truncated canonical stream is rejected");
    bad = base;
    CoreStream(bad, 2).actions.push_back(
        CoreStream(bad, 2).actions.front());
    suite.Rejects(
        [&] { (void)MaterializeIsaV1CollectivePhaseBarrier(
                  bad, 2, barrier); },
        "extra canonical stream action is rejected");

    auto reject_selected = [&](IsaV1CollectiveArtifactLowering bad,
                               const std::string &name) {
        suite.Rejects(
            [&] { (void)MaterializeIsaV1CollectivePhaseBarrier(
                      bad, 2, barrier); },
            name);
    };
    bad = base;
    CoreStream(bad, 2).actions[barrier].plan_index = 1;
    reject_selected(std::move(bad), "wrong plan index is rejected");
    bad = base;
    CoreStream(bad, 2).actions[barrier].key.epoch++;
    reject_selected(std::move(bad), "wrong plan key is rejected");
    bad = base;
    CoreStream(bad, 2).actions[barrier].action.core = 9;
    reject_selected(std::move(bad), "wrong action core is rejected");
    bad = base;
    CoreStream(bad, 2).actions[barrier].action.rank = 1;
    reject_selected(std::move(bad), "wrong action rank is rejected");
    bad = base;
    CoreStream(bad, 2).actions[barrier].action.item_index = 0;
    reject_selected(std::move(bad), "barrier item metadata is rejected");
    bad = base;
    CoreStream(bad, 2).actions[barrier].internal_token = 1;
    reject_selected(std::move(bad), "barrier internal token is rejected");
    bad = base;
    CoreStream(bad, 2).actions[barrier].public_aggregate_token = 1;
    reject_selected(std::move(bad), "barrier public token is rejected");
}

void TestWaveAndPlanRejections(Suite &suite) {
    const auto base = MakeLowering();
    const std::size_t barrier =
        FindAction(base, 2, IsaV1ActionKind::POSTED_BARRIER);
    auto rejects = [&](IsaV1CollectiveArtifactLowering bad,
                       const std::string &name) {
        suite.Rejects(
            [&] { (void)MaterializeIsaV1CollectivePhaseBarrier(
                      bad, 2, barrier); },
            name);
    };

    auto bad = base;
    bad.plans[0].waves[0].wave_index++;
    rejects(std::move(bad), "non-canonical wave index is rejected");
    bad = base;
    bad.plans[0].waves[0].posted_phase_id++;
    rejects(std::move(bad), "non-canonical posted phase is rejected");
    bad = base;
    bad.plans[0].waves[0].complete_phase_id++;
    rejects(std::move(bad), "non-canonical complete phase is rejected");
    bad = base;
    bad.plans[0].waves.clear();
    rejects(std::move(bad), "missing wave is rejected");
    bad = base;
    bad.plans[0].actions_by_rank[0].push_back(
        bad.plans[0].actions_by_rank[0][barrier]);
    rejects(std::move(bad), "duplicate canonical barrier is rejected");
    bad = base;
    bad.plans[0].group[0] = 9;
    rejects(std::move(bad), "plan core/rank mismatch is rejected");
    bad = base;
    bad.plans[0].key.group_id = 0;
    CoreStream(bad, 2).actions[barrier].key.group_id = 0;
    rejects(std::move(bad), "zero collective key is rejected");
}

} // namespace

int RunIsaV1CollectivePhaseLoweringSelfTest() {
    Suite suite;
    TestMaterialization(suite);
    TestEnvelopeRejections(suite);
    TestWaveAndPlanRejections(suite);
    if (suite.failures == 0) {
        std::cout << "[COLLECTIVE PHASE LOWERING V1] PASS ("
                  << suite.checks << " checks)\n";
        return 0;
    }
    std::cerr << "[COLLECTIVE PHASE LOWERING V1] FAIL ("
              << suite.failures << "/" << suite.checks
              << " checks failed)\n";
    return suite.failures;
}

#ifdef COLLECTIVE_PHASE_LOWERING_V1_SELFTEST_MAIN
int sc_main(int, char **) {
    return RunIsaV1CollectivePhaseLoweringSelfTest();
}
#endif
