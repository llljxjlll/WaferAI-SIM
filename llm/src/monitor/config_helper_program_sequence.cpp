#include "monitor/config_helper_program_sequence.h"

#include <iostream>
#include <stdexcept>

config_helper_program_sequence::config_helper_program_sequence(
    const std::vector<std::vector<uint8_t>> &artifact_bytes, bool refill,
    bool pause_on_final)
    : artifacts_(artifact_bytes), refill_(refill),
      pause_on_final_(pause_on_final) {
    if (artifacts_.size() < 2)
        throw ConfigHelperProgramError(
            "Program sequence requires at least two artifacts");
    for (const auto &bytes : artifacts_) {
        if (bytes.empty())
            throw ConfigHelperProgramError(
                "Program sequence artifact must not be empty");
    }
    current_ = std::make_unique<config_helper_program>(artifacts_[0], refill_);
    for (std::size_t index = 1; index < artifacts_.size(); ++index) {
        config_helper_program candidate(artifacts_[index], refill_);
        ValidateCompatibleProgram(candidate);
    }
    SyncPublicState();
}

void config_helper_program_sequence::ValidateCompatibleProgram(
    const config_helper_program &candidate) const {
    if (candidate.expected_ack_cores() != current_->expected_ack_cores() ||
        candidate.expected_done_cores() != current_->expected_done_cores())
        throw ConfigHelperProgramError(
            "Program sequence active-core sets must remain identical");
}

void config_helper_program_sequence::SyncPublicState() {
    coreconfigs = current_->coreconfigs;
    source_info = current_->source_info;
    pipeline = current_->pipeline;
    end_cores = current_->end_cores;
    end_count_sources = current_->end_count_sources;
    g_recv_ack_cnt = 0;
    g_recv_done_cnt = 0;
    g_temp_ack_msg.clear();
    g_temp_done_msg.clear();
}

void config_helper_program_sequence::fill_queue_config(std::queue<Msg> *queue) {
    current_->fill_queue_config(queue);
}

void config_helper_program_sequence::fill_queue_start(std::queue<Msg> *queue) {
    current_->fill_queue_start(queue);
}

void config_helper_program_sequence::fill_queue_data(std::queue<Msg> *queue) {
    current_->fill_queue_data(queue);
}

void config_helper_program_sequence::parse_ack_msg(
    Event_engine *event_engine, int flow_id, sc_event *notify_event) {
    current_->g_temp_ack_msg = std::move(g_temp_ack_msg);
    current_->parse_ack_msg(event_engine, flow_id, notify_event);
    g_recv_ack_cnt = current_->g_recv_ack_cnt;
}

void config_helper_program_sequence::parse_done_msg(
    Event_engine *event_engine, sc_event *notify_event) {
    bool complete = current_->expected_done_cores().empty();
    for (const Msg &message : g_temp_done_msg)
        complete = current_->AcceptDone(message);
    g_temp_done_msg.clear();
    g_recv_done_cnt = current_->g_recv_done_cnt;
    if (!complete)
        return;

    ++completed_segments_;
    std::cout << "[DENSE_SEQUENCE_SEGMENT] index=" << current_segment_
              << " status=done final="
              << (completed_segments_ == artifacts_.size() ? 1 : 0)
              << std::endl;
    if (event_engine != nullptr)
        event_engine->add_event(name(), "Program sequence segment complete",
                                "i", Trace_event_util());
    if (completed_segments_ == artifacts_.size()) {
        std::cout << "[DENSE_SEQUENCE_DRAIN] segments=" << artifacts_.size()
                  << " one_shot=1" << std::endl;
        if (pause_on_final_)
            sc_pause();
        else
            sc_stop();
        return;
    }
    if (notify_event == nullptr)
        throw ConfigHelperProgramError(
            "Program sequence requires a next-config notification event");
    ++current_segment_;
    current_->LoadProgram(artifacts_[current_segment_]);
    SyncPublicState();
    notify_event->notify(CYCLE, SC_NS);
    sc_pause();
}

void config_helper_program_sequence::generate_prims(int index) {
    current_->generate_prims(index);
}

void config_helper_program_sequence::printSelf() {
    std::cout << "Program sequence helper: segment=" << current_segment_
              << "/" << artifacts_.size() << "\n";
    current_->printSelf();
}

config_helper_program_sequence *config_helper_program_sequence::clone() const {
    return new config_helper_program_sequence(
        artifacts_, refill_, pause_on_final_);
}

std::shared_ptr<const CoreGroupRegistry>
config_helper_program_sequence::core_group_registry() const noexcept {
    return current_->core_group_registry();
}

std::shared_ptr<const IsaV1CollectiveProgramImage>
config_helper_program_sequence::collective_program_image() const noexcept {
    return current_->collective_program_image();
}

std::shared_ptr<const IsaV1CollectiveProfileProgramImage>
config_helper_program_sequence::collective_profile_program_image() const noexcept {
    return current_->collective_profile_program_image();
}
