#pragma once

#include "isa/program_format.h"
#include "isa/collective_program_v1.h"
#include "isa/record_lowering.h"
#include "dte/sync_runtime.h"
#include "dte/coll_program_profile_v1.h"
#include "monitor/config_helper_base.h"
#include "monitor/host_envelope.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

class ConfigHelperProgramError : public std::runtime_error {
public:
    explicit ConfigHelperProgramError(const std::string &message)
        : std::runtime_error(message) {}
};

// Applies semantic relocations to a temporary artifact. This operation has no
// simulator-global side effects and preserves ProgramFormat operand legality.
void ApplyProgramRelocations(ProgramArtifact &artifact);

class config_helper_program : public config_helper_base {
public:
    explicit config_helper_program(const std::vector<uint8_t> &artifact_bytes);
    explicit config_helper_program(const ProgramArtifact &artifact);

    // Strong exception guarantee: the currently committed program and ACK/
    // DONE state remain unchanged if any decode, relocation, lowering, or
    // internal-wire validation step fails.
    void LoadProgram(const std::vector<uint8_t> &artifact_bytes);

    std::vector<HostEnvelope> BuildConfigMessages();
    std::vector<HostEnvelope> BuildStartMessages() const;
    std::vector<HostEnvelope> BuildDataMessages() const;

    void fill_queue_config(std::queue<Msg> *queue) override;
    void fill_queue_start(std::queue<Msg> *queue) override;
    void fill_queue_data(std::queue<Msg> *queue) override;

    bool AcceptAck(const Msg &message, int phase_id);
    bool AcceptDone(const Msg &message);
    void parse_ack_msg(Event_engine *event_engine, int flow_id,
                       sc_event *notify_event) override;
    void parse_done_msg(Event_engine *event_engine,
                        sc_event *notify_event) override;

    void generate_prims(int index) override;
    void printSelf() override;
    config_helper_program *clone() const override;

    const ProgramArtifact &artifact() const noexcept { return artifact_; }
    const std::set<int> &expected_ack_cores() const noexcept {
        return expected_ack_cores_;
    }
    const std::set<int> &expected_done_cores() const noexcept {
        return expected_done_cores_;
    }
    std::shared_ptr<const CoreGroupRegistry>
    core_group_registry() const noexcept override {
        return core_group_registry_;
    }
    std::shared_ptr<const IsaV1CollectiveProgramImage>
    collective_program_image() const noexcept override {
        return collective_program_image_;
    }
    std::shared_ptr<const IsaV1CollectiveProfileProgramImage>
    collective_profile_program_image() const noexcept override {
        return collective_profile_program_image_;
    }

private:
    struct PreparedCore {
        int core_id = -1;
        bool included = false;
        std::vector<std::unique_ptr<PrimBase>> prims;
    };

    std::vector<uint8_t> artifact_bytes_;
    ProgramArtifact artifact_;
    std::vector<PreparedCore> prepared_cores_;
    std::set<int> expected_ack_cores_;
    std::set<int> expected_done_cores_;
    std::optional<int> ack_phase_;
    std::set<int> seen_ack_cores_;
    std::set<int> seen_done_cores_;
    std::shared_ptr<const CoreGroupRegistry> core_group_registry_;
    std::shared_ptr<const IsaV1CollectiveProgramImage>
        collective_program_image_;
    std::shared_ptr<const IsaV1CollectiveProfileProgramImage>
        collective_profile_program_image_;
};
