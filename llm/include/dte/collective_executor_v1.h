#pragma once

#include "dte/collective_aggregate_v1.h"
#include "dte/collective_final_phase_gate_v1.h"
#include "dte/collective_wave_admission_v1.h"
#include "isa/collective_program_v1.h"
#include "prims/base.h"
#include "prims/collective_launch_v1_prim.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <vector>

CollectiveWaveAdmissionCapacityV1
CollectiveWaveAdmissionCapacityForImageV1(
    const IsaV1CollectiveProgramImage &image,
    uint32_t per_core_sessions,
    uint64_t per_core_receive_bytes);

enum class CollectiveExecutorPayloadKindV1 : uint8_t {
    ENDPOINT_PRIM = 0,
    LOCAL_DATA_PRIM = 1,
    PHASE_BARRIER_PRIM = 2,
    WAIT = 3,
};

enum class CollectiveExecutorWaitKindV1 : uint8_t {
    RECEIVE_LOCAL_AND_TRANSPORT = 0,
    SEND_LOCAL = 1,
    TRANSPORT_RETIRE = 2,
};

struct CollectiveExecutorWaitV1 {
    CollectiveExecutorWaitKindV1 kind =
        CollectiveExecutorWaitKindV1::SEND_LOCAL;
    uint32_t internal_token = 0;
    uint32_t public_token = 0;
};

// A move-only description of one canonical action. Prim ownership is passed
// to the caller; WAIT actions instead carry an explicit completion contract.
struct CollectiveExecutorActionV1 {
    uint64_t action_id = 0;
    uint32_t plan_index = 0;
    uint32_t action_stream_index = 0;
    uint16_t wave_index = 0;
    uint16_t phase_id = 0;
    IsaV1ActionKind canonical_kind = IsaV1ActionKind::POSTED_BARRIER;
    CollectiveExecutorPayloadKindV1 payload_kind =
        CollectiveExecutorPayloadKindV1::WAIT;
    uint32_t internal_token = 0;
    uint32_t public_token = 0;
    std::unique_ptr<PrimBase> prim;
    std::optional<CollectiveExecutorWaitV1> wait;
};

struct CollectiveExecutorResidualV1 {
    std::size_t images = 0;
    std::size_t plans = 0;
    std::size_t runnable_plans = 0;
    std::size_t completed_plans = 0;
    std::size_t required_issue_sites = 0;
    std::size_t accepted_issue_sites = 0;
    std::size_t remaining_actions = 0;
    std::size_t inflight_actions = 0;
    std::size_t arrived_waves = 0;
    std::size_t departed_waves = 0;
    std::size_t aborted = 0;
    CollectiveAggregateResidual aggregate;
    CollectiveFinalPhaseGateResidualV1 final_gate;

    bool ActiveEmpty() const noexcept;
};

// Per-core orchestration for one immutable image. The shared coordinator must
// already own the same exact image identity; it remains externally owned so
// one instance can be shared by every participating core.
class CollectiveExecutorV1 final {
public:
    CollectiveExecutorV1() = default;
    CollectiveExecutorV1(const CollectiveExecutorV1 &) = delete;
    CollectiveExecutorV1 &operator=(const CollectiveExecutorV1 &) = delete;

    void Configure(
        std::shared_ptr<const IsaV1CollectiveProgramImage> image,
        uint16_t local_core,
        CollectiveWaveAdmissionCoordinatorV1 *coordinator);

    // Returns true only for the marker that completes its local plan's issue
    // set and transitions that plan to runnable.
    bool AcceptLaunch(const Collective_launch_v1_prim &launch);

    // At most one action is in flight per core. A null result means either no
    // plan is runnable or the next wave has not reached ACTIVE admission.
    std::optional<CollectiveExecutorActionV1> NextAction();

    // Completes a materialized Prim action. Endpoint completion gates are not
    // inferred here; their canonical WAIT descriptions own those transitions.
    void ActionComplete(uint64_t action_id);

    // Completes a typed WAIT only when all gates required by its kind are
    // observed. False leaves the same action in flight and changes no state.
    bool WaitComplete(uint64_t action_id, bool local_complete,
                      bool transport_retired);

    bool TryWait(uint32_t public_token);
    bool TryFence();
    void Cancel(uint32_t public_token);

    bool HasPublicToken(uint32_t public_token) const noexcept;
    bool HasInflight() const noexcept { return inflight_.has_value(); }
    bool HasRunnable() const noexcept;

    // Program/runtime errors are image-wide. Abort is idempotent locally and
    // reclaims the shared coordinator only while it still owns this identity.
    void Abort() noexcept;

    bool Drained() const noexcept;
    bool Configured() const noexcept { return configured_; }
    uint16_t LocalCore() const noexcept { return local_core_; }
    CollectiveProgramImageIdentityV1 Identity() const noexcept {
        return identity_;
    }
    CollectiveExecutorResidualV1 Residual() const noexcept;

private:
    struct PlanState {
        IsaV1CollectiveActionRange range;
        std::set<uint32_t> required_records;
        std::set<uint32_t> accepted_records;
        std::set<uint32_t> public_tokens;
        uint32_t cursor = 0;
        bool runnable = false;
        bool complete = false;
        std::set<uint16_t> arrived_waves;
        std::set<uint16_t> departed_waves;
    };

    struct Inflight {
        uint64_t action_id = 0;
        uint32_t plan_index = 0;
        uint32_t stream_index = 0;
        IsaV1ActionKind kind = IsaV1ActionKind::POSTED_BARRIER;
        uint16_t wave_index = 0;
        uint16_t phase_id = 0;
        uint32_t internal_token = 0;
        uint32_t public_token = 0;
        std::vector<CollectiveAggregateLocalWorkHandle> local_work;
    };

    static CollectiveLaunchV1Role LaunchRole(
        IsaV1CollectiveRecordRole role);
    const IsaV1LoweredCollectiveAction &LoweredAction(
        uint32_t stream_index) const;
    bool EnsureWaveActive(uint32_t plan_index,
                          const IsaV1Action &action);
    CollectiveExecutorActionV1 Materialize(
        uint32_t plan_index, uint32_t stream_index,
        const IsaV1LoweredCollectiveAction &lowered,
        const IsaV1CollectiveActionImplementation &implementation);
    void AdvanceInflight();
    void MaybeRetireLocal();
    void RequireReady() const;

    bool configured_ = false;
    bool locally_retired_ = false;
    bool aborted_ = false;
    uint16_t local_core_ = 0;
    uint64_t next_action_id_ = 1;
    std::size_t next_plan_cursor_ = 0;
    CollectiveProgramImageIdentityV1 identity_;
    std::shared_ptr<const IsaV1CollectiveProgramImage> image_;
    const IsaV1CollectiveCoreProgramImage *core_image_ = nullptr;
    IsaV1CoreCollectiveActionStream empty_core_stream_;
    const IsaV1CoreCollectiveActionStream *core_stream_ = nullptr;
    CollectiveWaveAdmissionCoordinatorV1 *coordinator_ = nullptr;
    std::unique_ptr<CollectiveAggregateRuntime> aggregate_;
    std::unique_ptr<CollectiveAggregateFinalPhaseGateV1> final_gate_;
    std::map<uint32_t, PlanState> plans_;
    std::map<uint32_t, IsaV1CollectiveIssueSite> sites_by_record_;
    std::map<uint32_t, uint32_t> token_to_plan_;
    std::optional<Inflight> inflight_;
};

int RunCollectiveExecutorV1SelfTest();
