#pragma once
#include "dte/coll_types.h"
#include "systemc.h"
#include <cstdint>

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
void ResetCollectiveBarrierStateForTest();
void ResetCollectiveGatherReorderStateForTest();
void ResetCollectiveReduceRxStateForTest();
