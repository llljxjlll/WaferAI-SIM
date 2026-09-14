#pragma once

#include "monitor/config_helper_program.h"

#include <cstddef>
#include <memory>
#include <vector>

// Runs multiple finalized Program artifacts through one MemInterface and one
// Monitor.  Intermediate Program DONE sets pause at a host-visible boundary.
// The final DONE set normally stops the workload, but a caller that owns a
// typed post-compute phase may request one final host-visible pause.
class config_helper_program_sequence final : public config_helper_base {
public:
    explicit config_helper_program_sequence(
        const std::vector<std::vector<uint8_t>> &artifact_bytes,
        bool refill = false, bool pause_on_final = false);

    void fill_queue_config(std::queue<Msg> *queue) override;
    void fill_queue_start(std::queue<Msg> *queue) override;
    void fill_queue_data(std::queue<Msg> *queue) override;
    void parse_ack_msg(Event_engine *event_engine, int flow_id,
                       sc_event *notify_event) override;
    void parse_done_msg(Event_engine *event_engine,
                        sc_event *notify_event) override;
    void generate_prims(int index) override;
    void printSelf() override;
    config_helper_program_sequence *clone() const override;

    std::shared_ptr<const CoreGroupRegistry>
    core_group_registry() const noexcept override;
    std::shared_ptr<const IsaV1CollectiveProgramImage>
    collective_program_image() const noexcept override;
    std::shared_ptr<const IsaV1CollectiveProfileProgramImage>
    collective_profile_program_image() const noexcept override;

    std::size_t current_segment() const noexcept { return current_segment_; }
    std::size_t completed_segments() const noexcept {
        return completed_segments_;
    }
    std::size_t segment_count() const noexcept { return artifacts_.size(); }
    bool final_complete() const noexcept {
        return completed_segments_ == artifacts_.size();
    }
    const ProgramArtifact &artifact() const noexcept {
        return current_->artifact();
    }

private:
    void SyncPublicState();
    void ValidateCompatibleProgram(const config_helper_program &candidate) const;

    std::vector<std::vector<uint8_t>> artifacts_;
    bool refill_ = false;
    bool pause_on_final_ = false;
    std::size_t current_segment_ = 0;
    std::size_t completed_segments_ = 0;
    std::unique_ptr<config_helper_program> current_;
};
