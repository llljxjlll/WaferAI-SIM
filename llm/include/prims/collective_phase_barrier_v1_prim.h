#pragma once

#include "dte/coll_types.h"
#include "prims/base.h"

#include <cstdint>

inline constexpr uint8_t kCollectivePhaseBarrierV1WireVersion = 1;
inline constexpr uint8_t kCollectivePhaseBarrierV1WireSegments = 2;

// Strict-only internal phase barrier emitted by whole-artifact collective
// lowering.  It deliberately uses the ordinary CollectiveKey namespace;
// collective_id UINT32_MAX remains reserved for public GROUP_SYNC.
class Collective_phase_barrier_v1_prim final : public PrimBase {
public:
    CollectiveKey key;
    uint16_t phase_id = 0;
    uint16_t rank = 0;
    uint16_t group_size = 1;
    // Zero is the P6 baseline.  A non-zero value releases a pre-programmed
    // P7 multicast/reduce tree after the final barrier departure.
    uint16_t release_tree_id = 0;

    void Validate() const;
    int taskCoreDefault(TaskCoreContext &context) override;
    vector<sc_bv<128>> serialize() override;
    void deserialize(vector<sc_bv<128>> wire) override;
    void printSelf() override;

    Collective_phase_barrier_v1_prim() {
        name = "Collective_phase_barrier_v1_prim";
        setPrimMainCategory(SYNC_PRIM);
    }
};
