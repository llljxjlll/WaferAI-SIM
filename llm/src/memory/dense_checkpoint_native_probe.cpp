#include "memory/dense_checkpoint_native_probe.h"

#include <stdexcept>

namespace dense_checkpoint {
namespace {
[[noreturn]] void Fail(const char *why) {
    throw std::invalid_argument(std::string("Dense checkpoint native probe: ") + why);
}
}

NativeCheckpointObservation VerifyNativeCheckpointEvents(
    const NativeCheckpointSignature &source,
    NativeCheckpointMode mode,
    const std::vector<NativeCheckpointEvent> &events,
    const std::vector<uint8_t> &payload,
    const std::vector<uint8_t> &present) {
    if (source.manifest_digest.empty() || source.replay_record_ref.empty() ||
        source.backward_record_ref.empty() || source.replay_input_abi_id.empty() ||
        source.replay_weight_abi_id.empty() ||
        source.replay_parameter_state_abi_id.empty() ||
        source.replay_output_abi_id.empty() ||
        source.backward_input_abi_id.empty() ||
        source.replay_input_bytes == 0 || source.replay_weight_bytes == 0 ||
        source.replay_output_bytes == 0 ||
        source.replay_input_bytes >= source.replay_output_bytes)
        Fail("source signature is incomplete or checkpoint has no saving");
    const uint64_t bytes = mode == NativeCheckpointMode::kSaveOutput
        ? source.replay_output_bytes : source.replay_input_bytes;
    const std::string &saved_abi = mode == NativeCheckpointMode::kSaveOutput
        ? source.replay_output_abi_id : source.replay_input_abi_id;
    if (source.activation_hbm_address > source.hbm_capacity_bytes ||
        bytes > source.hbm_capacity_bytes - source.activation_hbm_address)
        Fail("physical activation tape exceeds real HBM capacity");
    if (payload.size() != bytes || present.size() != bytes)
        Fail("actual HBM snapshot has wrong physical size");
    for (uint8_t bit : present)
        if (bit == 0) Fail("actual HBM activation tape is not fully written");
    if (events.size() != (mode == NativeCheckpointMode::kSaveOutput ? 4U : 5U))
        Fail("actual native event count differs from policy");
    uint64_t previous = 0;
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (i != 0 && events[i].sequence_index <= previous)
            Fail("actual native event order is not strictly increasing");
        previous = events[i].sequence_index;
    }
    const auto require_compute = [&](std::size_t i,
                                     NativeCheckpointEventKind kind,
                                     const std::string &record,
                                     const std::string &abi) {
        if (events[i].kind != kind || events[i].native_record_ref != record ||
            events[i].buffer_abi_id != abi)
            Fail("native compute event lost signed record or BufferABI");
    };
    require_compute(0, NativeCheckpointEventKind::kForwardProduce,
                    source.replay_record_ref, source.replay_output_abi_id);
    if (events[0].observed_weight_payload.size() != source.replay_weight_bytes)
        Fail("original forward did not observe exact signed parameter bytes");
    const auto require_lsu = [&](std::size_t i, NativeCheckpointEventKind kind) {
        if (events[i].kind != kind || events[i].hbm_address != source.activation_hbm_address ||
            events[i].size_bytes != bytes || events[i].buffer_abi_id != saved_abi ||
            !events[i].native_record_ref.empty())
            Fail("actual LSU transfer lost signed activation span");
    };
    require_lsu(1, NativeCheckpointEventKind::kLsuSave);
    require_lsu(2, NativeCheckpointEventKind::kLsuRestore);
    if (events[1].observed_payload != payload ||
        events[2].observed_payload != payload)
        Fail("actual HBM save/restore bytes disagree with LSU payload");
    const std::size_t backward_index =
        mode == NativeCheckpointMode::kSaveOutput ? 3 : 4;
    if (mode == NativeCheckpointMode::kSaveReplayInput)
        require_compute(3, NativeCheckpointEventKind::kNativeReplay,
                        source.replay_record_ref, source.replay_output_abi_id);
    if (mode == NativeCheckpointMode::kSaveReplayInput &&
        (events[3].observed_payload != events[2].observed_payload ||
         events[0].observed_weight_payload != events[3].observed_weight_payload))
        Fail("native replay lost restored activation or frozen weight bytes");
    require_compute(backward_index,
                    NativeCheckpointEventKind::kBackwardConsume,
                    source.backward_record_ref, source.backward_input_abi_id);
    return NativeCheckpointObservation{
        bytes, bytes,
        mode == NativeCheckpointMode::kSaveReplayInput ? 1U : 0U,
        1U, 1U,
    };
}

} // namespace dense_checkpoint
