#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dense_checkpoint {

// Contract input is derived from the source-signed LinkedProgramManifest cut.
// This probe verifies physical events supplied by a real LSU/compute runtime;
// construction alone is never evidence that recompute ran.
struct NativeCheckpointSignature {
    std::string manifest_digest;
    std::string replay_record_ref;
    std::string backward_record_ref;
    std::string replay_input_abi_id;
    std::string replay_weight_abi_id;
    std::string replay_parameter_state_abi_id;
    std::string replay_output_abi_id;
    std::string backward_input_abi_id;
    uint64_t activation_hbm_address = 0;
    uint64_t replay_input_bytes = 0;
    uint64_t replay_weight_bytes = 0;
    uint64_t replay_output_bytes = 0;
    uint64_t hbm_capacity_bytes = 0;
};

enum class NativeCheckpointMode { kSaveOutput, kSaveReplayInput };
enum class NativeCheckpointEventKind {
    kForwardProduce, kLsuSave, kLsuRestore, kNativeReplay, kBackwardConsume
};

struct NativeCheckpointEvent {
    NativeCheckpointEventKind kind;
    std::string native_record_ref;
    std::string buffer_abi_id;
    uint64_t hbm_address = 0;
    uint64_t size_bytes = 0;
    uint64_t sequence_index = 0;
    // Bytes observed from real SRAM/LSU boundaries, not source-side estimates.
    std::vector<uint8_t> observed_payload;
    std::vector<uint8_t> observed_weight_payload;
};

struct NativeCheckpointObservation {
    uint64_t saved_bytes = 0;
    uint64_t restored_bytes = 0;
    uint64_t replay_compute_records = 0;
    uint64_t useful_forward_compute_records = 0;
    uint64_t useful_backward_compute_records = 0;
};

NativeCheckpointObservation VerifyNativeCheckpointEvents(
    const NativeCheckpointSignature &source,
    NativeCheckpointMode mode,
    const std::vector<NativeCheckpointEvent> &actual_events,
    const std::vector<uint8_t> &actual_hbm_payload,
    const std::vector<uint8_t> &actual_hbm_present);

} // namespace dense_checkpoint
