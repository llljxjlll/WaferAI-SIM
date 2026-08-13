#pragma once

#include "dte/coll_types.h"
#include "systemc.h"

#include <cstddef>
#include <cstdint>

inline constexpr std::size_t
    kDefaultCollectiveBarrierRuntimeCapacity = 4096;

struct CollectiveBarrierRuntimeCapacity {
    std::size_t max_active_states =
        kDefaultCollectiveBarrierRuntimeCapacity;
};

struct CollectiveBarrierRuntimeResidual {
    std::size_t active_states = 0;
    std::size_t arrived_ranks = 0;
    std::size_t departed_ranks = 0;
    std::size_t waiting_ranks = 0;
};

// Reconfiguration is legal only while no barrier is active. The default is
// deliberately bounded for legacy callers; Monitor/config integration may
// call this API before dispatch without changing the barrier Prim ABI.
void ConfigureCollectiveBarrierRuntime(
    CollectiveBarrierRuntimeCapacity capacity);
CollectiveBarrierRuntimeCapacity
CollectiveBarrierRuntimeConfiguredCapacity() noexcept;
CollectiveBarrierRuntimeResidual
CollectiveBarrierRuntimeResidualState() noexcept;

// Exact CollectiveKey abort covers every active phase for one artifact
// instance/epoch. Waiters are awakened and fail; new barriers may reuse the
// key only after the old state has been detached.
std::size_t AbortCollectiveBarrierKey(const CollectiveKey &key);
void ResetCollectiveBarrierRuntime();

void WaitCollectiveBarrier(const CollectiveKey &key, uint16_t phase_id,
                           uint16_t rank, uint16_t group_size,
                           uint16_t release_tree_id = 0);
void ProcessGatherReorderArrival(const CollDescriptor &descriptor,
                                 uint16_t phase_id);
void ProcessReduceRxArrival(const CollDescriptor &descriptor,
                            uint16_t phase_id);
size_t CollectiveBarrierStateCount();
size_t CollectiveGatherReorderStateCount();
size_t CollectiveReduceRxStateCount();

// Compatibility wrappers retained for existing V1/V6 tests and callers.
void ResetCollectiveBarrierStateForTest();
void ResetCollectiveGatherReorderStateForTest();
void ResetCollectiveReduceRxStateForTest();
