#pragma once

#include "isa/collective_program_v1.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

struct CollectiveProgramImageIdentityV1 {
    uint64_t generation = 0;
    uint64_t cookie = 0;

    bool operator==(const CollectiveProgramImageIdentityV1 &other) const
        noexcept;
    bool operator<(const CollectiveProgramImageIdentityV1 &other) const
        noexcept;
};

CollectiveProgramImageIdentityV1 CollectiveProgramImageIdentity(
    const IsaV1CollectiveProgramImage &image) noexcept;

struct CollectiveWaveCoreCapacityV1 {
    uint16_t core_id = 0;
    uint32_t endpoint_sessions = 0;
    uint64_t receive_bytes = 0;
};

struct CollectiveWaveAdmissionCapacityV1 {
    std::size_t max_plans = 0;
    std::size_t max_registered_waves = 0;
    std::size_t max_wave_demands = 0;
    std::size_t max_pending_waves = 0;
    std::size_t max_active_waves = 0;
    std::vector<CollectiveWaveCoreCapacityV1> cores;
};

enum class CollectiveWaveAdmissionStatusV1 : uint8_t {
    REGISTERED = 0,
    FORMING = 1,
    PENDING = 2,
    ACTIVE = 3,
    COMPLETE = 4,
};

struct CollectiveWaveAdmissionResidualV1 {
    std::size_t images = 0;
    std::size_t waves = 0;
    std::size_t forming_waves = 0;
    std::size_t pending_waves = 0;
    std::size_t active_waves = 0;
    std::size_t complete_waves = 0;
    std::size_t arrivals = 0;
    std::size_t departures = 0;
    uint64_t active_endpoint_sessions = 0;
    uint64_t active_receive_bytes = 0;

    bool Empty() const noexcept;
};

// One coordinator owns one immutable program image at a time.  All
// participants arrive before a whole wave is queued, and admission reserves
// every core's endpoint/receive-byte demand atomically.  This prevents a wave
// from holding a subset of cross-core resources while waiting for the rest.
class CollectiveWaveAdmissionCoordinatorV1 final {
public:
    explicit CollectiveWaveAdmissionCoordinatorV1(
        CollectiveWaveAdmissionCapacityV1 capacity);

    // Copies only bounded canonical wave metadata.  Validation is
    // transactional: an exception leaves the coordinator empty.
    void RegisterImage(const IsaV1CollectiveProgramImage &image);

    // Arrivals may form the next wave while the preceding wave is departing.
    // A complete wave is queued with a monotonic ticket.  Scheduling chooses
    // the oldest eligible ticket and never bypasses it for a smaller demand.
    CollectiveWaveAdmissionStatusV1 Arrive(
        const CollectiveProgramImageIdentityV1 &identity,
        uint32_t plan_index, uint16_t wave_index, uint16_t core_id);

    CollectiveWaveAdmissionStatusV1 Poll(
        const CollectiveProgramImageIdentityV1 &identity,
        uint32_t plan_index, uint16_t wave_index) const;

    // Resources are released only after every participating core departs.
    void Depart(const CollectiveProgramImageIdentityV1 &identity,
                uint32_t plan_index, uint16_t wave_index,
                uint16_t core_id);

    // Exact-identity abort reclaims forming, queued, and active reservations.
    void AbortImage(const CollectiveProgramImageIdentityV1 &identity);

    // Normal retirement requires every registered wave to be COMPLETE.
    void RetireImage(const CollectiveProgramImageIdentityV1 &identity);

    bool HasImage() const noexcept { return registered_; }
    CollectiveProgramImageIdentityV1 Identity() const noexcept {
        return identity_;
    }
    CollectiveWaveAdmissionResidualV1 Residual() const;
    const CollectiveWaveAdmissionCapacityV1 &Capacity() const noexcept {
        return capacity_;
    }

private:
    using WaveKey = std::pair<uint32_t, uint16_t>;

    struct CoreResource {
        uint32_t endpoint_sessions = 0;
        uint64_t receive_bytes = 0;
    };

    struct Wave {
        uint32_t plan_index = 0;
        uint16_t wave_index = 0;
        std::vector<uint16_t> participants;
        std::map<uint16_t, CoreResource> demands;
        CollectiveWaveAdmissionStatusV1 status =
            CollectiveWaveAdmissionStatusV1::REGISTERED;
        std::set<uint16_t> arrivals;
        std::set<uint16_t> departures;
        uint64_t ticket = 0;
    };

    Wave &RequireWave(uint32_t plan_index, uint16_t wave_index);
    const Wave &RequireWave(uint32_t plan_index,
                            uint16_t wave_index) const;
    void RequireIdentity(
        const CollectiveProgramImageIdentityV1 &identity) const;
    bool PredecessorComplete(const Wave &wave) const;
    bool Fits(const Wave &wave) const noexcept;
    void Reserve(const Wave &wave);
    void Release(const Wave &wave);
    void Schedule();

    CollectiveWaveAdmissionCapacityV1 capacity_;
    std::map<uint16_t, CoreResource> core_capacity_;
    std::map<uint16_t, CoreResource> core_usage_;
    bool registered_ = false;
    CollectiveProgramImageIdentityV1 identity_;
    std::map<WaveKey, Wave> waves_;
    std::size_t pending_waves_ = 0;
    std::size_t active_waves_ = 0;
    uint64_t next_ticket_ = 1;
};
