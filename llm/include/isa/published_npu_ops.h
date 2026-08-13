#pragma once

#include "isa/npu_cost_model.h"

// Simulator headers historically define DUMMY as a feature macro. Keep the
// public opcode type usable regardless of include order.
#if defined(DUMMY)
#pragma push_macro("DUMMY")
#undef DUMMY
#define NPUSIM_RESTORE_DUMMY_MACRO
#endif
#include "isa/opcode.h"
#if defined(NPUSIM_RESTORE_DUMMY_MACRO)
#pragma pop_macro("DUMMY")
#undef NPUSIM_RESTORE_DUMMY_MACRO
#endif

#include <cstdint>
#include <string>
#include <unordered_map>

using PublishedNpuParameters = std::unordered_map<std::string, int>;

struct PublishedNpuHardwareView {
    uint64_t exu_x_dims = 0;
    uint64_t exu_count = 0;
    uint64_t sfu_x_dims = 0;
    uint64_t vec_x_dims = 0;
    uint64_t vec_count = 0;
    float compute_utilization = 0.0F;
    uint64_t cycle_ns = 0;
    bool use_performance_gemm = false;
};

PublishedNpuHardwareView PublishedNpuHardwareForCore(int core_id);

NpuOps EvaluatePublishedNpuOps(
    Opcode opcode, const PublishedNpuParameters &parameters,
    const PublishedNpuHardwareView &hardware = {});
