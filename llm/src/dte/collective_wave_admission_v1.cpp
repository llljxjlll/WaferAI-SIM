#include "dte/collective_wave_admission_v1.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace {

void Require(bool condition, const std::string &message) {
    if (!condition) throw std::invalid_argument(message);
}

uint64_t CheckedAdd(uint64_t left, uint64_t right,
                    const std::string &message) {
    if (left > std::numeric_limits<uint64_t>::max() - right)
        throw std::overflow_error(message);
    return left + right;
}

} // namespace

bool CollectiveProgramImageIdentityV1::operator==(
    const CollectiveProgramImageIdentityV1 &other) const noexcept {
    return generation == other.generation && cookie == other.cookie;
}

bool CollectiveProgramImageIdentityV1::operator<(
    const CollectiveProgramImageIdentityV1 &other) const noexcept {
    return std::tie(generation, cookie) <
           std::tie(other.generation, other.cookie);
}

CollectiveProgramImageIdentityV1 CollectiveProgramImageIdentity(
    const IsaV1CollectiveProgramImage &image) noexcept {
    return {image.Generation(), image.Cookie()};
}

bool CollectiveWaveAdmissionResidualV1::Empty() const noexcept {
    return images == 0 && waves == 0 && forming_waves == 0 &&
           pending_waves == 0 && active_waves == 0 &&
           complete_waves == 0 && arrivals == 0 && departures == 0 &&
           active_endpoint_sessions == 0 && active_receive_bytes == 0;
}

CollectiveWaveAdmissionCoordinatorV1::
CollectiveWaveAdmissionCoordinatorV1(
    CollectiveWaveAdmissionCapacityV1 capacity)
    : capacity_(std::move(capacity)) {
    Require(capacity_.max_plans != 0 &&
                capacity_.max_registered_waves != 0 &&
                capacity_.max_wave_demands != 0 &&
                capacity_.max_pending_waves != 0 &&
                capacity_.max_active_waves != 0 &&
                !capacity_.cores.empty(),
            "collective wave admission capacity must be positive");
    for (const CollectiveWaveCoreCapacityV1 &core : capacity_.cores) {
        Require(core.endpoint_sessions != 0 && core.receive_bytes != 0,
                "collective wave core capacity must be positive");
        Require(core_capacity_
                    .emplace(core.core_id,
                             CoreResource{core.endpoint_sessions,
                                          core.receive_bytes})
                    .second,
                "collective wave core capacity contains a duplicate core");
        core_usage_.emplace(core.core_id, CoreResource{});
    }
}

void CollectiveWaveAdmissionCoordinatorV1::RegisterImage(
    const IsaV1CollectiveProgramImage &image) {
    Require(!registered_,
            "collective wave admission already owns an image");
    const CollectiveProgramImageIdentityV1 identity =
        CollectiveProgramImageIdentity(image);
    Require(identity.generation != 0 && identity.cookie != 0,
            "collective wave admission image identity is invalid");
    const auto &lowering = image.Lowering();
    Require(!lowering.plans.empty() &&
                lowering.plans.size() <= capacity_.max_plans,
            "collective wave admission plan capacity is exhausted");

    std::map<WaveKey, Wave> candidate;
    std::size_t demand_cursor = 0;
    std::size_t demand_count = 0;
    for (std::size_t plan_index = 0;
         plan_index < lowering.plans.size(); ++plan_index) {
        Require(plan_index <= std::numeric_limits<uint32_t>::max(),
                "collective wave admission plan index exceeds u32");
        const IsaV1CollectivePlan &plan = lowering.plans[plan_index];
        Require(plan.key.group_id != 0 && !plan.group.empty() &&
                    plan.actions_by_rank.size() == plan.group.size() &&
                    !plan.waves.empty(),
                "collective wave admission plan metadata is incomplete");
        std::set<uint16_t> unique_cores;
        for (uint16_t core : plan.group) {
            Require(unique_cores.insert(core).second,
                    "collective wave admission plan group contains a duplicate core");
            Require(core_capacity_.count(core) == 1,
                    "collective wave admission has no capacity for a plan core");
        }

        for (std::size_t wave_index = 0;
             wave_index < plan.waves.size(); ++wave_index) {
            Require(wave_index <= std::numeric_limits<uint16_t>::max(),
                    "collective wave admission wave index exceeds u16");
            const IsaV1Wave &plan_wave = plan.waves[wave_index];
            Require(plan_wave.wave_index == wave_index &&
                        plan_wave.posted_phase_id == wave_index * 2 &&
                        plan_wave.complete_phase_id ==
                            wave_index * 2 + 1,
                    "collective wave admission plan phases are non-canonical");
            Wave wave;
            wave.plan_index = static_cast<uint32_t>(plan_index);
            wave.wave_index = static_cast<uint16_t>(wave_index);
            wave.participants = plan.group;
            for (std::size_t rank = 0; rank < plan.group.size(); ++rank) {
                CoreResource expected;
                for (uint32_t child_index : plan_wave.child_indices) {
                    Require(child_index < plan.child_flows.size(),
                            "collective wave admission child index is invalid");
                    const IsaV1ChildFlow &flow =
                        plan.child_flows[child_index];
                    Require(flow.wave_index == plan_wave.wave_index,
                            "collective wave admission child wave is inconsistent");
                    if (flow.source_rank == rank) {
                        Require(expected.endpoint_sessions !=
                                    std::numeric_limits<uint32_t>::max(),
                                "collective wave endpoint demand overflows u32");
                        ++expected.endpoint_sessions;
                    }
                    if (flow.destination_rank == rank) {
                        Require(expected.endpoint_sessions !=
                                    std::numeric_limits<uint32_t>::max(),
                                "collective wave endpoint demand overflows u32");
                        ++expected.endpoint_sessions;
                        expected.receive_bytes = CheckedAdd(
                            expected.receive_bytes, flow.length_bytes,
                            "collective wave receive demand overflows u64");
                    }
                }
                Require(demand_cursor < image.WaveDemands().size(),
                        "collective wave admission image demand table is truncated");
                const IsaV1CollectiveWaveDemand &actual =
                    image.WaveDemands()[demand_cursor++];
                Require(actual.plan_index == plan_index &&
                            actual.wave_index == wave_index &&
                            actual.rank == rank &&
                            actual.core_id == plan.group[rank] &&
                            actual.endpoint_sessions ==
                                expected.endpoint_sessions &&
                            actual.receive_bytes == expected.receive_bytes,
                        "collective wave admission image demand is non-canonical");
                const CoreResource &budget =
                    core_capacity_.at(actual.core_id);
                Require(actual.endpoint_sessions <=
                            budget.endpoint_sessions &&
                            actual.receive_bytes <= budget.receive_bytes,
                        "collective wave demand exceeds runtime core capacity");
                wave.demands.emplace(actual.core_id, expected);
                ++demand_count;
            }
            Require(candidate.emplace(
                        WaveKey{wave.plan_index, wave.wave_index},
                        std::move(wave))
                        .second,
                    "collective wave admission image wave is duplicated");
        }
    }
    Require(demand_cursor == image.WaveDemands().size(),
            "collective wave admission image demand table has extra entries");
    Require(candidate.size() <= capacity_.max_registered_waves &&
                demand_count <= capacity_.max_wave_demands,
            "collective wave admission registration capacity is exhausted");
    Require(image.AdmissionCapacity().waves == candidate.size() &&
                image.AdmissionCapacity().wave_demands == demand_count,
            "collective wave admission image capacity summary is inconsistent");

    waves_.swap(candidate);
    identity_ = identity;
    registered_ = true;
    pending_waves_ = 0;
    active_waves_ = 0;
    next_ticket_ = 1;
    for (auto &[core, usage] : core_usage_) {
        (void)core;
        usage = {};
    }
}

void CollectiveWaveAdmissionCoordinatorV1::RequireIdentity(
    const CollectiveProgramImageIdentityV1 &identity) const {
    Require(registered_ && identity == identity_,
            "collective wave admission image identity is stale or unknown");
}

CollectiveWaveAdmissionCoordinatorV1::Wave &
CollectiveWaveAdmissionCoordinatorV1::RequireWave(
    uint32_t plan_index, uint16_t wave_index) {
    const auto found = waves_.find({plan_index, wave_index});
    if (found == waves_.end())
        throw std::out_of_range(
            "collective wave admission wave is unknown");
    return found->second;
}

const CollectiveWaveAdmissionCoordinatorV1::Wave &
CollectiveWaveAdmissionCoordinatorV1::RequireWave(
    uint32_t plan_index, uint16_t wave_index) const {
    const auto found = waves_.find({plan_index, wave_index});
    if (found == waves_.end())
        throw std::out_of_range(
            "collective wave admission wave is unknown");
    return found->second;
}

bool CollectiveWaveAdmissionCoordinatorV1::PredecessorComplete(
    const Wave &wave) const {
    if (wave.wave_index == 0) return true;
    return RequireWave(wave.plan_index,
                       static_cast<uint16_t>(wave.wave_index - 1))
               .status == CollectiveWaveAdmissionStatusV1::COMPLETE;
}

bool CollectiveWaveAdmissionCoordinatorV1::Fits(
    const Wave &wave) const noexcept {
    for (const auto &[core, demand] : wave.demands) {
        const auto budget = core_capacity_.find(core);
        const auto usage = core_usage_.find(core);
        if (budget == core_capacity_.end() ||
            usage == core_usage_.end() ||
            usage->second.endpoint_sessions >
                budget->second.endpoint_sessions ||
            usage->second.receive_bytes > budget->second.receive_bytes)
            return false;
        if (demand.endpoint_sessions >
                budget->second.endpoint_sessions -
                    usage->second.endpoint_sessions ||
            demand.receive_bytes > budget->second.receive_bytes -
                                       usage->second.receive_bytes)
            return false;
    }
    return true;
}

void CollectiveWaveAdmissionCoordinatorV1::Reserve(
    const Wave &wave) {
    for (const auto &[core, demand] : wave.demands) {
        CoreResource &usage = core_usage_.at(core);
        usage.endpoint_sessions += demand.endpoint_sessions;
        usage.receive_bytes += demand.receive_bytes;
    }
}

void CollectiveWaveAdmissionCoordinatorV1::Release(
    const Wave &wave) {
    for (const auto &[core, demand] : wave.demands) {
        CoreResource &usage = core_usage_.at(core);
        if (usage.endpoint_sessions < demand.endpoint_sessions ||
            usage.receive_bytes < demand.receive_bytes)
            throw std::logic_error(
                "collective wave admission resource accounting underflows");
        usage.endpoint_sessions -= demand.endpoint_sessions;
        usage.receive_bytes -= demand.receive_bytes;
    }
}

void CollectiveWaveAdmissionCoordinatorV1::Schedule() {
    while (active_waves_ < capacity_.max_active_waves) {
        Wave *oldest = nullptr;
        for (auto &[key, wave] : waves_) {
            (void)key;
            if (wave.status !=
                    CollectiveWaveAdmissionStatusV1::PENDING ||
                !PredecessorComplete(wave))
                continue;
            if (oldest == nullptr || wave.ticket < oldest->ticket)
                oldest = &wave;
        }
        if (oldest == nullptr || !Fits(*oldest)) return;
        Reserve(*oldest);
        oldest->status = CollectiveWaveAdmissionStatusV1::ACTIVE;
        --pending_waves_;
        ++active_waves_;
    }
}

CollectiveWaveAdmissionStatusV1
CollectiveWaveAdmissionCoordinatorV1::Arrive(
    const CollectiveProgramImageIdentityV1 &identity,
    uint32_t plan_index, uint16_t wave_index, uint16_t core_id) {
    RequireIdentity(identity);
    Wave &wave = RequireWave(plan_index, wave_index);
    Require(wave.status ==
                    CollectiveWaveAdmissionStatusV1::REGISTERED ||
                wave.status ==
                    CollectiveWaveAdmissionStatusV1::FORMING,
            "collective wave admission arrival is late or duplicate");
    Require(std::find(wave.participants.begin(),
                      wave.participants.end(), core_id) != wave.participants.end(),
            "collective wave admission arrival core is not a participant");
    Require(wave.arrivals.count(core_id) == 0,
            "collective wave admission arrival is duplicate");

    std::set<uint16_t> candidate = wave.arrivals;
    candidate.insert(core_id);
    const bool complete =
        candidate.size() == wave.participants.size();
    if (complete) {
        Require(pending_waves_ < capacity_.max_pending_waves,
                "collective wave admission pending capacity is exhausted");
        Require(next_ticket_ != std::numeric_limits<uint64_t>::max(),
                "collective wave admission ticket space is exhausted");
    }
    wave.arrivals.swap(candidate);
    if (!complete) {
        wave.status = CollectiveWaveAdmissionStatusV1::FORMING;
    } else {
        wave.status = CollectiveWaveAdmissionStatusV1::PENDING;
        wave.ticket = next_ticket_++;
        ++pending_waves_;
        Schedule();
    }
    return wave.status;
}

CollectiveWaveAdmissionStatusV1
CollectiveWaveAdmissionCoordinatorV1::Poll(
    const CollectiveProgramImageIdentityV1 &identity,
    uint32_t plan_index, uint16_t wave_index) const {
    RequireIdentity(identity);
    return RequireWave(plan_index, wave_index).status;
}

void CollectiveWaveAdmissionCoordinatorV1::Depart(
    const CollectiveProgramImageIdentityV1 &identity,
    uint32_t plan_index, uint16_t wave_index, uint16_t core_id) {
    RequireIdentity(identity);
    Wave &wave = RequireWave(plan_index, wave_index);
    Require(wave.status == CollectiveWaveAdmissionStatusV1::ACTIVE,
            "collective wave admission departure requires an active wave");
    Require(wave.arrivals.count(core_id) == 1,
            "collective wave admission departure core did not arrive");
    Require(wave.departures.count(core_id) == 0,
            "collective wave admission departure is duplicate");

    std::set<uint16_t> candidate = wave.departures;
    candidate.insert(core_id);
    const bool complete =
        candidate.size() == wave.participants.size();
    wave.departures.swap(candidate);
    if (!complete) return;

    Release(wave);
    wave.status = CollectiveWaveAdmissionStatusV1::COMPLETE;
    --active_waves_;
    Schedule();
}

void CollectiveWaveAdmissionCoordinatorV1::AbortImage(
    const CollectiveProgramImageIdentityV1 &identity) {
    RequireIdentity(identity);
    waves_.clear();
    registered_ = false;
    identity_ = {};
    pending_waves_ = 0;
    active_waves_ = 0;
    next_ticket_ = 1;
    for (auto &[core, usage] : core_usage_) {
        (void)core;
        usage = {};
    }
}

void CollectiveWaveAdmissionCoordinatorV1::RetireImage(
    const CollectiveProgramImageIdentityV1 &identity) {
    RequireIdentity(identity);
    for (const auto &[key, wave] : waves_) {
        (void)key;
        Require(wave.status ==
                    CollectiveWaveAdmissionStatusV1::COMPLETE,
                "collective wave admission image is not fully complete");
    }
    AbortImage(identity);
}

CollectiveWaveAdmissionResidualV1
CollectiveWaveAdmissionCoordinatorV1::Residual() const {
    CollectiveWaveAdmissionResidualV1 residual;
    residual.images = registered_ ? 1 : 0;
    residual.waves = waves_.size();
    for (const auto &[key, wave] : waves_) {
        (void)key;
        residual.arrivals += wave.arrivals.size();
        residual.departures += wave.departures.size();
        switch (wave.status) {
        case CollectiveWaveAdmissionStatusV1::REGISTERED:
            break;
        case CollectiveWaveAdmissionStatusV1::FORMING:
            ++residual.forming_waves;
            break;
        case CollectiveWaveAdmissionStatusV1::PENDING:
            ++residual.pending_waves;
            break;
        case CollectiveWaveAdmissionStatusV1::ACTIVE:
            ++residual.active_waves;
            break;
        case CollectiveWaveAdmissionStatusV1::COMPLETE:
            ++residual.complete_waves;
            break;
        }
    }
    for (const auto &[core, usage] : core_usage_) {
        (void)core;
        residual.active_endpoint_sessions = CheckedAdd(
            residual.active_endpoint_sessions,
            usage.endpoint_sessions,
            "collective wave residual session count overflows u64");
        residual.active_receive_bytes = CheckedAdd(
            residual.active_receive_bytes, usage.receive_bytes,
            "collective wave residual receive bytes overflow u64");
    }
    return residual;
}
