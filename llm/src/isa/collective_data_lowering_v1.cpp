#include "isa/collective_data_lowering_v1.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void Require(bool condition, const std::string &message) {
    if (!condition) throw RecordLoweringError(message);
}

uint64_t CheckedAdd(uint64_t left, uint64_t right,
                    const std::string &message) {
    if (left > std::numeric_limits<uint64_t>::max() - right)
        throw RecordLoweringError(message);
    return left + right;
}

uint64_t CheckedMultiply(uint64_t left, uint64_t right,
                         const std::string &message) {
    if (left != 0 &&
        right > std::numeric_limits<uint64_t>::max() / left)
        throw RecordLoweringError(message);
    return left * right;
}

uint64_t DTypeBytes(CollDType dtype) {
    switch (dtype) {
    case CollDType::UINT8: return 1;
    case CollDType::INT32: return 4;
    case CollDType::INT64: return 8;
    case CollDType::FP32:
    case CollDType::FP16:
    case CollDType::FP8: break;
    }
    throw RecordLoweringError(
        "ISA-v1 collective data action uses an unsupported dtype");
}

const IsaV1CoreCollectiveActionStream &FindCoreStream(
    const IsaV1CollectiveArtifactLowering &lowering,
    uint16_t executing_core) {
    const IsaV1CoreCollectiveActionStream *found = nullptr;
    for (const IsaV1CoreCollectiveActionStream &stream :
         lowering.core_actions) {
        if (stream.core_id != executing_core) continue;
        Require(found == nullptr,
                "ISA-v1 collective data lowering has duplicate core streams");
        found = &stream;
    }
    Require(found != nullptr,
            "ISA-v1 collective data lowering has no executing-core stream");
    return *found;
}

const IsaV1CollectivePlan &ValidateActionEnvelope(
    const IsaV1CollectiveArtifactLowering &lowering,
    uint16_t executing_core,
    const IsaV1LoweredCollectiveAction &lowered) {
    Require(lowered.plan_index < lowering.plans.size(),
            "ISA-v1 collective data action plan index is out of range");
    const IsaV1CollectivePlan &plan = lowering.plans[lowered.plan_index];
    const IsaV1Action &action = lowered.action;
    Require(lowered.key == plan.key,
            "ISA-v1 collective data action key does not match its plan");
    Require(lowered.internal_token == 0 &&
                lowered.public_aggregate_token == 0,
            "ISA-v1 collective data action carries endpoint tokens");
    Require(action.core == executing_core,
            "ISA-v1 collective data action core does not match execution");
    Require(action.rank < plan.group.size() &&
                plan.group[action.rank] == action.core,
            "ISA-v1 collective data action rank/core does not match its plan");
    Require(action.rank < plan.actions_by_rank.size(),
            "ISA-v1 collective data action rank has no canonical stream");
    const auto &canonical = plan.actions_by_rank[action.rank];
    Require(std::count(canonical.begin(), canonical.end(), action) == 1,
            "ISA-v1 collective data action is not unique in its canonical plan stream");
    return plan;
}

std::unique_ptr<Collective_data_v1_prim> MaterializeLocalCopy(
    const IsaV1CollectivePlan &plan, const IsaV1Action &action) {
    Require(action.item_index < plan.local_copies.size(),
            "ISA-v1 local-copy item index is out of range");
    Require(action.wave_index == 0 && action.phase_id == 0,
            "ISA-v1 local-copy phase is not canonical");
    const IsaV1LocalCopy &copy = plan.local_copies[action.item_index];
    Require(copy.rank == action.rank && copy.core == action.core,
            "ISA-v1 local-copy rank/core does not match its action");
    Require(copy.length_bytes == plan.length_bytes,
            "ISA-v1 local-copy L does not match its plan");
    Require(action.rank < plan.rank_records.size(),
            "ISA-v1 local-copy rank record is missing");
    const IsaV1RankRecordContract &contract =
        plan.rank_records[action.rank];
    Require(contract.send.present && contract.receive.present,
            "ISA-v1 local-copy lacks SEND/RECEIVE records");
    Require(copy.source_address_bytes == CheckedAdd(
                contract.send.base_address_bytes,
                copy.source_offset_bytes,
                "ISA-v1 local-copy source address overflows"),
            "ISA-v1 local-copy source absolute address is inconsistent");
    const uint64_t expected_destination =
        CollIsReduction(plan.op) && plan.group.size() == 1
            ? contract.result_address_bytes
            : CheckedAdd(contract.receive.base_address_bytes,
                         copy.destination_offset_bytes,
                         "ISA-v1 local-copy destination address overflows");
    Require(copy.destination_address_bytes == expected_destination,
            "ISA-v1 local-copy destination absolute address is inconsistent");

    auto prim = std::make_unique<Collective_data_v1_prim>();
    prim->mode = CollectiveDataV1PrimMode::LOCAL_COPY;
    prim->key = plan.key;
    prim->phase_id = action.phase_id;
    prim->source_address_bytes = copy.source_address_bytes;
    prim->destination_address_bytes = copy.destination_address_bytes;
    prim->length_bytes = copy.length_bytes;
    prim->input_count = 1;
    prim->dtype = CollDType::UINT8;
    prim->reduce_op = CollReduceOp::NONE;
    return prim;
}

std::unique_ptr<Collective_data_v1_prim> MaterializeReduce(
    const IsaV1CollectivePlan &plan, const IsaV1Action &action) {
    Require(CollIsReduction(plan.op) && plan.group.size() > 1,
            "ISA-v1 reduce action does not belong to a multi-rank reduction");
    Require(action.item_index < plan.reduce_targets.size(),
            "ISA-v1 reduce item index is out of range");
    Require(!plan.waves.empty() &&
                action.wave_index == plan.waves.back().wave_index &&
                action.phase_id == plan.waves.back().complete_phase_id,
            "ISA-v1 reduce action is not in the final complete phase");
    const IsaV1ReduceTarget &target =
        plan.reduce_targets[action.item_index];
    Require(target.rank == action.rank && target.core == action.core,
            "ISA-v1 reduce target rank/core does not match its action");
    Require(action.rank < plan.rank_records.size(),
            "ISA-v1 reduce rank record is missing");
    const IsaV1RankRecordContract &contract =
        plan.rank_records[action.rank];
    Require(contract.receive.present,
            "ISA-v1 reduce target lacks a RECEIVE record");
    Require(target.staging_address_bytes ==
                contract.receive.base_address_bytes &&
                target.result_address_bytes ==
                    contract.result_address_bytes,
            "ISA-v1 reduce staging/result absolute address is inconsistent");
    Require(target.length_bytes == plan.length_bytes,
            "ISA-v1 reduce target L does not match its plan");
    Require(plan.group.size() <= std::numeric_limits<uint16_t>::max() &&
                target.input_count == plan.group.size(),
            "ISA-v1 reduce target input_count does not match group N");
    const uint64_t width = DTypeBytes(target.dtype);
    Require(CheckedMultiply(target.element_count, width,
                            "ISA-v1 reduce element byte size overflows") ==
                target.length_bytes,
            "ISA-v1 reduce element_count*dtype does not match L");
    Require(target.reduce_op == CollReduceOp::SUM ||
                target.reduce_op == CollReduceOp::MAX,
            "ISA-v1 reduce action requires SUM or MAX");

    auto prim = std::make_unique<Collective_data_v1_prim>();
    prim->mode = CollectiveDataV1PrimMode::REDUCE;
    prim->key = plan.key;
    prim->phase_id = action.phase_id;
    prim->source_address_bytes = target.staging_address_bytes;
    prim->destination_address_bytes = target.result_address_bytes;
    prim->length_bytes = target.length_bytes;
    prim->input_count = target.input_count;
    prim->dtype = target.dtype;
    prim->reduce_op = target.reduce_op;
    return prim;
}

} // namespace

std::unique_ptr<Collective_data_v1_prim>
MaterializeIsaV1CollectiveDataAction(
    const IsaV1CollectiveArtifactLowering &lowering,
    uint16_t executing_core, std::size_t action_stream_index) {
    const IsaV1CoreCollectiveActionStream &stream =
        FindCoreStream(lowering, executing_core);
    Require(action_stream_index < stream.actions.size(),
            "ISA-v1 collective data action stream index is out of range");
    const IsaV1LoweredCollectiveAction &lowered =
        stream.actions[action_stream_index];
    const IsaV1CollectivePlan &plan = ValidateActionEnvelope(
        lowering, executing_core, lowered);

    std::unique_ptr<Collective_data_v1_prim> prim;
    if (lowered.action.kind == IsaV1ActionKind::LOCAL_COPY) {
        prim = MaterializeLocalCopy(plan, lowered.action);
    } else if (lowered.action.kind == IsaV1ActionKind::REDUCE_COMPUTE) {
        prim = MaterializeReduce(plan, lowered.action);
    } else {
        throw RecordLoweringError(
            "ISA-v1 collective data materializer rejects non-data actions");
    }
    try {
        prim->Validate();
    } catch (const std::exception &error) {
        throw RecordLoweringError(
            std::string("ISA-v1 collective data Prim validation: ") +
            error.what());
    }
    return prim;
}
