#pragma once

#include "isa/program_format.h"
#include "isa/record_lowering.h"

#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

enum class IsaV1CollectiveActionImplementationKind : uint8_t {
    CHILD_ENDPOINT_PRIM = 0,
    LOCAL_DATA_PRIM = 1,
    PHASE_BARRIER_PRIM = 2,
    WAIT_RECEIVE = 3,
    WAIT_SEND = 4,
    WAIT_TRANSPORT_RETIRE = 5,
};

struct IsaV1CollectiveActionImplementation {
    uint16_t core_id = 0;
    uint32_t action_stream_index = 0;
    uint32_t plan_index = 0;
    IsaV1CollectiveActionImplementationKind kind =
        IsaV1CollectiveActionImplementationKind::CHILD_ENDPOINT_PRIM;

    bool operator==(
        const IsaV1CollectiveActionImplementation &other) const noexcept;
};

struct IsaV1CollectiveActionRange {
    uint32_t plan_index = 0;
    uint16_t rank = 0;
    uint32_t begin = 0;
    uint32_t count = 0;

    bool operator==(const IsaV1CollectiveActionRange &other) const noexcept;
};

struct IsaV1CollectiveIssueSite {
    uint16_t core_id = 0;
    uint32_t record_index = 0;
    uint32_t plan_index = 0;
    IsaV1CollectiveRecordRole role =
        IsaV1CollectiveRecordRole::SEND;
    uint32_t public_token = 0;
    CollectiveKey key;

    bool operator==(const IsaV1CollectiveIssueSite &other) const noexcept;
};

struct IsaV1CollectiveWaveDemand {
    uint32_t plan_index = 0;
    uint16_t wave_index = 0;
    uint16_t rank = 0;
    uint16_t core_id = 0;
    uint32_t endpoint_sessions = 0;
    uint64_t receive_bytes = 0;

    bool operator==(const IsaV1CollectiveWaveDemand &other) const noexcept;
};

struct IsaV1CollectiveAggregateImageCapacity {
    std::size_t aggregate_tokens = 0;
    std::size_t child_tokens = 0;
    std::size_t local_work_items = 0;
    std::size_t ordinary_reserved_tokens = 0;

    bool operator==(
        const IsaV1CollectiveAggregateImageCapacity &other) const noexcept;
};

struct IsaV1CollectiveCoreImageCapacity {
    IsaV1CollectiveAggregateImageCapacity aggregate;
    std::size_t plans = 0;
    std::size_t actions = 0;
    std::size_t wave_demands = 0;
    uint32_t max_wave_endpoint_sessions = 0;
    uint64_t max_wave_receive_bytes = 0;

    bool operator==(
        const IsaV1CollectiveCoreImageCapacity &other) const noexcept;
};

struct IsaV1CollectiveCoreProgramImage {
    uint16_t core_id = 0;
    std::vector<IsaV1CollectiveActionRange> action_ranges;
    std::vector<IsaV1CollectiveActionImplementation> actions;
    std::set<uint32_t> ordinary_reserved_tokens;
    IsaV1CollectiveCoreImageCapacity capacity;

    bool operator==(
        const IsaV1CollectiveCoreProgramImage &other) const noexcept;
};

struct IsaV1CollectiveAdmissionImageCapacity {
    std::size_t waves = 0;
    std::size_t wave_demands = 0;
    uint32_t max_endpoint_sessions_per_core_wave = 0;
    uint64_t max_receive_bytes_per_core_wave = 0;

    bool operator==(
        const IsaV1CollectiveAdmissionImageCapacity &other) const noexcept;
};

// Explicit loader bounds.  Defaults match the existing artifact limits and
// the baseline planner, but every dimension remains independently reducible
// for capacity/error tests and future platform profiles.
struct IsaV1CollectiveProgramImageLimits {
    std::size_t max_plans = 65536;
    std::size_t max_children = kMaxProgramRecords;
    std::size_t max_cores = kMaxProgramCores;
    std::size_t max_actions = kMaxProgramRecords;
    std::size_t max_issue_sites = kMaxProgramRecords;
    std::size_t max_waves = kMaxProgramRecords;
    std::size_t max_wave_demands = kMaxProgramRecords;
    std::size_t max_aggregate_tokens_per_core = kMaxProgramRecords;
    std::size_t max_child_tokens_per_core = kMaxProgramRecords;
    std::size_t max_local_work_items_per_core = kMaxProgramRecords;
    std::size_t max_reserved_tokens_per_core = kMaxProgramRecords;
    uint32_t max_endpoint_sessions_per_core_wave = 64;
    uint64_t max_receive_bytes_per_core_wave =
        kIsaV1DefaultMaxChildBytes;
};

struct IsaV1CollectiveProgramImageConfig {
    uint32_t total_cores = 0;
    uint32_t cores_per_die = 0;
    uint64_t generation = 1;
    IsaV1PlannerCapacity planner_capacity;
    IsaV1CollectiveProgramImageLimits limits;
};

// Immutable loader product.  Construction proves that the artifact and its
// canonical lowering agree, every action has exactly one implementation, and
// all derived state fits explicit bounds.  It is still only an executable
// candidate: Worker/runtime installation is the later executable seal.
class IsaV1CollectiveProgramImage final {
public:
    uint64_t Generation() const noexcept { return generation_; }
    uint64_t Cookie() const noexcept { return cookie_; }
    const IsaV1CollectiveArtifactLowering &Lowering() const noexcept {
        return lowering_;
    }
    const std::vector<IsaV1CollectiveCoreProgramImage> &Cores() const
        noexcept {
        return cores_;
    }
    const std::vector<IsaV1CollectiveIssueSite> &IssueSites() const noexcept {
        return issue_sites_;
    }
    const std::vector<IsaV1CollectiveWaveDemand> &WaveDemands() const
        noexcept {
        return wave_demands_;
    }
    const IsaV1CollectiveAdmissionImageCapacity &AdmissionCapacity() const
        noexcept {
        return admission_capacity_;
    }
    const IsaV1CollectiveProgramImageLimits &Limits() const noexcept {
        return limits_;
    }

    const IsaV1CollectiveCoreProgramImage *FindCore(
        uint16_t core_id) const noexcept;

    bool operator==(const IsaV1CollectiveProgramImage &other) const;

private:
    friend IsaV1CollectiveProgramImage BuildIsaV1CollectiveProgramImage(
        const ProgramArtifact &, const IsaV1CollectiveArtifactLowering &,
        const IsaV1CollectiveProgramImageConfig &);

    uint64_t generation_ = 0;
    uint64_t cookie_ = 0;
    IsaV1CollectiveArtifactLowering lowering_;
    std::vector<IsaV1CollectiveCoreProgramImage> cores_;
    std::vector<IsaV1CollectiveIssueSite> issue_sites_;
    std::vector<IsaV1CollectiveWaveDemand> wave_demands_;
    IsaV1CollectiveAdmissionImageCapacity admission_capacity_;
    IsaV1CollectiveProgramImageLimits limits_;
};

IsaV1CollectiveProgramImage BuildIsaV1CollectiveProgramImage(
    const ProgramArtifact &relocated_artifact,
    const IsaV1CollectiveArtifactLowering &lowering,
    const IsaV1CollectiveProgramImageConfig &config);
