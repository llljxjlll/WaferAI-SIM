#pragma once

#include "dte/collective_aggregate_v1.h"
#include "dte/collective_wave_admission_v1.h"

#include <cstddef>
#include <cstdint>
#include <map>

struct CollectiveFinalPhaseGateCapacityV1 {
    std::size_t max_plans = 0;
    std::size_t max_public_tokens = 0;
};

enum class CollectiveFinalPhaseGateStatusV1 : uint8_t {
    WAITING_AGGREGATE = 0,
    WAITING_FINAL_PHASE = 1,
    READY = 2,
};

struct CollectiveFinalPhaseGateResidualV1 {
    std::size_t images = 0;
    std::size_t plans = 0;
    std::size_t public_tokens = 0;
    std::size_t completed_plans = 0;

    bool Empty() const noexcept;
};

// Per-core completion facade for CollectiveAggregateRuntime.  A public token
// can retire only when its existing aggregate is READY and the core has
// returned from the plan's final COMPLETE phase barrier.
class CollectiveAggregateFinalPhaseGateV1 final {
public:
    CollectiveAggregateFinalPhaseGateV1(
        uint16_t local_core,
        CollectiveFinalPhaseGateCapacityV1 capacity);

    // Bind after CollectiveAggregateRuntime::RegisterArtifact.  Exact token
    // counts are checked so TryFence cannot consume an unrelated aggregate.
    // Registration is transactional.
    void RegisterImage(const IsaV1CollectiveProgramImage &image,
                       const CollectiveAggregateRuntime &aggregate);

    void MarkFinalComplete(
        const CollectiveProgramImageIdentityV1 &identity,
        uint32_t plan_index, const CollectiveKey &key,
        uint16_t executing_core, uint16_t rank, uint16_t phase_id);

    CollectiveFinalPhaseGateStatusV1 Poll(
        const CollectiveProgramImageIdentityV1 &identity,
        uint32_t public_token,
        const CollectiveAggregateRuntime &aggregate) const;

    bool TryWait(const CollectiveProgramImageIdentityV1 &identity,
                 uint32_t public_token,
                 CollectiveAggregateRuntime &aggregate);
    bool TryFence(const CollectiveProgramImageIdentityV1 &identity,
                  CollectiveAggregateRuntime &aggregate);

    // Normal cancel is legal only while the underlying aggregate remains
    // REGISTERED.  AbortRegistered preflights every token before changing
    // either runtime, so it is all-or-nothing.
    void Cancel(const CollectiveProgramImageIdentityV1 &identity,
                uint32_t public_token,
                CollectiveAggregateRuntime &aggregate);
    void AbortRegistered(
        const CollectiveProgramImageIdentityV1 &identity,
        CollectiveAggregateRuntime &aggregate);

    // Successful waits/fence/cancels drain token/plan gates.  RetireImage
    // clears the remaining identity only after that drain.
    void RetireImage(const CollectiveProgramImageIdentityV1 &identity);

    bool HasImage() const noexcept { return registered_; }
    CollectiveProgramImageIdentityV1 Identity() const noexcept {
        return identity_;
    }
    CollectiveFinalPhaseGateResidualV1 Residual() const noexcept;

private:
    struct PlanGate {
        CollectiveKey key;
        uint16_t rank = 0;
        uint16_t final_phase_id = 0;
        std::size_t outstanding_tokens = 0;
        bool final_complete = false;
    };

    struct TokenGate {
        uint32_t plan_index = 0;
    };

    void RequireIdentity(
        const CollectiveProgramImageIdentityV1 &identity) const;
    const TokenGate &RequireToken(uint32_t public_token) const;
    void RequireAggregateBinding(
        const CollectiveAggregateRuntime &aggregate) const;
    void ConsumeToken(uint32_t public_token);

    uint16_t local_core_;
    CollectiveFinalPhaseGateCapacityV1 capacity_;
    bool registered_ = false;
    CollectiveProgramImageIdentityV1 identity_;
    std::map<uint32_t, PlanGate> plans_;
    std::map<uint32_t, TokenGate> tokens_;
};
