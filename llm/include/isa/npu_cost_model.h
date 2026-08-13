#pragma once

#include <cstdint>

struct NpuOps {
    uint64_t exu = 0;
    uint64_t sfu = 0;
    uint64_t vec = 0;
};

struct NpuCostHardware {
    uint64_t exu_x_dims = 0;
    uint64_t exu_count = 0;
    uint64_t sfu_x_dims = 0;
    uint64_t vec_x_dims = 0;
    uint64_t vec_count = 0;
    float compute_utilization = 0.0F;
    uint64_t cycle_ns = 0;
};

struct NpuCostSnapshot {
    NpuOps ops;
    uint64_t exu_cycle_ns = 0;
    uint64_t sfu_cycle_ns = 0;
    uint64_t vec_cycle_ns = 0;
    uint64_t compute_cycle_ns = 0;
    uint64_t dram_time_ns = 0;
    uint64_t overlap_delay_ns = 0;
};

// The production NPU accounting uses truncated division and selects the
// slowest of EXU/SFU/VEC. This pure function is the single oracle used by the
// runtime and ISA self-tests; it does not perform memory or timing side effects.
NpuCostSnapshot CalculateNpuCost(const NpuOps &ops,
                                 const NpuCostHardware &hardware,
                                 uint64_t dram_time_ns);
