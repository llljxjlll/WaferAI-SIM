#include "isa/collective_phase_lowering_v1.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void Require(bool condition, const std::string &message) {
    if (!condition) throw RecordLoweringError(message);
}

const IsaV1CoreCollectiveActionStream &FindCoreStream(
    const IsaV1CollectiveArtifactLowering &lowering,
    uint16_t executing_core) {
    const IsaV1CoreCollectiveActionStream *found = nullptr;
    for (const IsaV1CoreCollectiveActionStream &stream :
         lowering.core_actions) {
        if (stream.core_id != executing_core) continue;
        Require(found == nullptr,
                "ISA-v1 phase lowering has duplicate core streams");
        found = &stream;
    }
    Require(found != nullptr,
            "ISA-v1 phase lowering has no executing-core stream");
    return *found;
}

void ValidateCoreStreamShape(
    const IsaV1CollectiveArtifactLowering &lowering,
    const IsaV1CoreCollectiveActionStream &stream) {
    std::size_t cursor = 0;
    for (std::size_t plan_index = 0; plan_index < lowering.plans.size();
         ++plan_index) {
        const IsaV1CollectivePlan &plan = lowering.plans[plan_index];
        const auto rank_it =
            std::find(plan.group.begin(), plan.group.end(), stream.core_id);
        if (rank_it == plan.group.end()) continue;
        Require(std::find(rank_it + 1, plan.group.end(), stream.core_id) ==
                    plan.group.end(),
                "ISA-v1 phase lowering plan group contains duplicate cores");
        Require(plan.actions_by_rank.size() == plan.group.size(),
                "ISA-v1 phase lowering plan action-rank table is incomplete");
        const std::size_t rank = static_cast<std::size_t>(
            rank_it - plan.group.begin());
        for (const IsaV1Action &expected : plan.actions_by_rank[rank]) {
            Require(cursor < stream.actions.size(),
                    "ISA-v1 phase lowering core action stream is truncated");
            const IsaV1LoweredCollectiveAction &actual =
                stream.actions[cursor++];
            Require(actual.plan_index == plan_index &&
                        actual.key == plan.key && actual.action == expected,
                    "ISA-v1 phase lowering core action stream is non-canonical");
        }
    }
    Require(cursor == stream.actions.size(),
            "ISA-v1 phase lowering core action stream has extra actions");
}

const IsaV1Wave &ValidateWave(const IsaV1CollectivePlan &plan,
                              const IsaV1Action &action) {
    Require(!plan.waves.empty(),
            "ISA-v1 phase lowering plan has no phase waves");
    const IsaV1Wave *selected = nullptr;
    for (std::size_t index = 0; index < plan.waves.size(); ++index) {
        const IsaV1Wave &wave = plan.waves[index];
        Require(index <= std::numeric_limits<uint16_t>::max() &&
                    wave.wave_index == index,
                "ISA-v1 phase lowering wave index is non-canonical");
        Require(index <=
                    static_cast<std::size_t>(
                        std::numeric_limits<uint16_t>::max() / 2) &&
                    wave.posted_phase_id == index * 2 &&
                    wave.complete_phase_id == index * 2 + 1,
                "ISA-v1 phase lowering wave phase IDs are non-canonical");
        if (wave.wave_index != action.wave_index) continue;
        Require(selected == nullptr,
                "ISA-v1 phase lowering wave index is duplicated");
        selected = &wave;
    }
    Require(selected != nullptr,
            "ISA-v1 phase lowering action references an unknown wave");
    const uint16_t expected_phase =
        action.kind == IsaV1ActionKind::POSTED_BARRIER
            ? selected->posted_phase_id
            : selected->complete_phase_id;
    Require(action.phase_id == expected_phase,
            "ISA-v1 phase lowering action phase does not match its wave");
    return *selected;
}

} // namespace

std::unique_ptr<Collective_phase_barrier_v1_prim>
MaterializeIsaV1CollectivePhaseBarrier(
    const IsaV1CollectiveArtifactLowering &lowering,
    uint16_t executing_core, std::size_t action_stream_index) {
    const IsaV1CoreCollectiveActionStream &stream =
        FindCoreStream(lowering, executing_core);
    ValidateCoreStreamShape(lowering, stream);
    Require(action_stream_index < stream.actions.size(),
            "ISA-v1 phase lowering action stream index is out of range");
    const IsaV1LoweredCollectiveAction &lowered =
        stream.actions[action_stream_index];
    const IsaV1Action &action = lowered.action;
    Require(action.kind == IsaV1ActionKind::POSTED_BARRIER ||
                action.kind == IsaV1ActionKind::COMPLETE_BARRIER,
            "ISA-v1 phase materializer rejects non-barrier actions");
    Require(lowered.plan_index < lowering.plans.size(),
            "ISA-v1 phase lowering action plan index is out of range");
    const IsaV1CollectivePlan &plan = lowering.plans[lowered.plan_index];
    Require(plan.key.group_id != 0 && lowered.key == plan.key,
            "ISA-v1 phase lowering action key does not match its plan");
    Require(lowered.internal_token == 0 &&
                lowered.public_aggregate_token == 0 &&
                action.item_index == kIsaV1NoItem,
            "ISA-v1 phase lowering barrier carries item/token metadata");
    Require(action.core == executing_core &&
                action.rank < plan.group.size() &&
                plan.group[action.rank] == executing_core,
            "ISA-v1 phase lowering action rank/core is inconsistent");
    Require(plan.group.size() <= std::numeric_limits<uint16_t>::max() &&
                action.rank < plan.actions_by_rank.size(),
            "ISA-v1 phase lowering group/rank table exceeds its wire");
    const auto &canonical = plan.actions_by_rank[action.rank];
    Require(std::count(canonical.begin(), canonical.end(), action) == 1,
            "ISA-v1 phase lowering action is not unique in its plan");
    (void)ValidateWave(plan, action);

    auto prim =
        std::make_unique<Collective_phase_barrier_v1_prim>();
    prim->key = plan.key;
    prim->phase_id = action.phase_id;
    prim->rank = action.rank;
    prim->group_size = static_cast<uint16_t>(plan.group.size());
    prim->release_tree_id = 0;
    try {
        prim->Validate();
    } catch (const std::exception &error) {
        throw RecordLoweringError(
            std::string("ISA-v1 phase barrier Prim validation: ") +
            error.what());
    }
    return prim;
}
