#include "memory/dense_checkpoint_native_probe.h"
#include <cassert>
#include <stdexcept>
#include <vector>

using namespace dense_checkpoint;

int main() {
    const NativeCheckpointSignature sig{
        "manifest-sha", "linked-forward-record", "linked-backward-record",
        "hidden-abi", "weight-abi", "parameter-state-abi",
        "logits-abi", "extended-logits-abi",
        2048, 32, 64, 128, 2176,
    };
    const auto events = [](bool checkpoint) {
        std::vector<NativeCheckpointEvent> out{
            {NativeCheckpointEventKind::kForwardProduce,
             "linked-forward-record", "logits-abi", 0, 0, 1, {}, {}},
            {NativeCheckpointEventKind::kLsuSave, "",
             checkpoint ? "hidden-abi" : "logits-abi",
             2048, checkpoint ? 32U : 128U, 2, {}, {}},
            {NativeCheckpointEventKind::kLsuRestore, "",
             checkpoint ? "hidden-abi" : "logits-abi",
             2048, checkpoint ? 32U : 128U, 3, {}, {}},
        };
        if (checkpoint)
            out.push_back({NativeCheckpointEventKind::kNativeReplay,
                           "linked-forward-record", "logits-abi", 0, 0, 4, {}, {}});
        out.push_back({NativeCheckpointEventKind::kBackwardConsume,
                       "linked-backward-record", "extended-logits-abi",
                       0, 0, checkpoint ? 5U : 4U, {}, {}});
        out[0].observed_weight_payload = std::vector<uint8_t>(64, 0x5a);
        out[1].observed_payload = std::vector<uint8_t>(checkpoint ? 32 : 128,
                                                       checkpoint ? 9 : 7);
        out[2].observed_payload = out[1].observed_payload;
        if (checkpoint) {
            out[3].observed_payload = out[2].observed_payload;
            out[3].observed_weight_payload = out[0].observed_weight_payload;
        }
        return out;
    };
    const auto baseline = VerifyNativeCheckpointEvents(
        sig, NativeCheckpointMode::kSaveOutput, events(false),
        std::vector<uint8_t>(128, 7), std::vector<uint8_t>(128, 1));
    assert(baseline.saved_bytes == 128 && baseline.replay_compute_records == 0);
    const auto checkpoint = VerifyNativeCheckpointEvents(
        sig, NativeCheckpointMode::kSaveReplayInput, events(true),
        std::vector<uint8_t>(32, 9), std::vector<uint8_t>(32, 1));
    assert(checkpoint.saved_bytes == 32 && checkpoint.replay_compute_records == 1);
    const auto rejects = [](auto fn) {
        try { fn(); } catch (const std::invalid_argument &) { return; }
        assert(false && "negative native probe must fail");
    };
    auto missing_replay = events(true);
    missing_replay.erase(missing_replay.begin() + 3);
    rejects([&] { VerifyNativeCheckpointEvents(
        sig, NativeCheckpointMode::kSaveReplayInput, missing_replay,
        std::vector<uint8_t>(32, 9), std::vector<uint8_t>(32, 1)); });
    auto wrong_weight = events(true);
    wrong_weight[3].observed_weight_payload[0] ^= 1;
    rejects([&] { VerifyNativeCheckpointEvents(
        sig, NativeCheckpointMode::kSaveReplayInput, wrong_weight,
        std::vector<uint8_t>(32, 9), std::vector<uint8_t>(32, 1)); });
    auto wrong_abi = events(true);
    wrong_abi[4].buffer_abi_id = "unsigned-backward";
    rejects([&] { VerifyNativeCheckpointEvents(
        sig, NativeCheckpointMode::kSaveReplayInput, wrong_abi,
        std::vector<uint8_t>(32, 9), std::vector<uint8_t>(32, 1)); });
    rejects([&] { VerifyNativeCheckpointEvents(
        sig, NativeCheckpointMode::kSaveReplayInput, events(true),
        std::vector<uint8_t>(32, 9), std::vector<uint8_t>(32, 0)); });
    rejects([&] { VerifyNativeCheckpointEvents(
        sig, NativeCheckpointMode::kSaveOutput, events(false),
        std::vector<uint8_t>(128, 9), std::vector<uint8_t>(128, 1)); });
    auto low_hbm = sig;
    low_hbm.hbm_capacity_bytes = 2176 - 1;
    rejects([&] { VerifyNativeCheckpointEvents(
        low_hbm, NativeCheckpointMode::kSaveOutput, events(false),
        std::vector<uint8_t>(128, 7), std::vector<uint8_t>(128, 1)); });
    return 0;
}
