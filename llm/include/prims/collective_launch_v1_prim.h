#pragma once

#include "dte/coll_types.h"
#include "prims/base.h"

#include <cstdint>

inline constexpr uint8_t kCollectiveLaunchV1PrimWireVersion = 1;
inline constexpr uint8_t kCollectiveLaunchV1PrimWireSegments = 4;

enum class CollectiveLaunchV1Role : uint8_t {
    ISSUE_SEND = 0,
    ISSUE_RECEIVE = 1,
    DECLARE_REDUCE_COMPUTE = 2,
};

// Strict-only internal launch descriptor. It binds one already-validated
// whole-artifact record to an immutable image generation and plan; Worker
// owns the future execution dispatch.
class Collective_launch_v1_prim final : public PrimBase {
public:
    CollectiveLaunchV1Role role = CollectiveLaunchV1Role::ISSUE_SEND;
    uint64_t image_generation = 1;
    uint32_t plan_index = 0;
    uint32_t external_record_index = 0;
    uint16_t expected_core = 0;
    CollectiveKey key;
    uint32_t public_token = 1;

    void Validate() const;
    int taskCoreDefault(TaskCoreContext &context) override;
    vector<sc_bv<128>> serialize() override;
    void deserialize(vector<sc_bv<128>> wire) override;
    void printSelf() override;

    Collective_launch_v1_prim() {
        name = "Collective_launch_v1_prim";
        setPrimMainCategory(COMM_PRIM);
    }
};
