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
#include "isa/record_codec.h"

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

struct PublishedNpuWork {
    NpuOps ops;
    uint64_t memory_read_bytes = 0;
    uint64_t memory_write_bytes = 0;
    uint64_t comparisons = 0;
};

PublishedNpuHardwareView PublishedNpuHardwareForCore(int core_id);

PublishedNpuWork EvaluatePublishedNpuWork(
    Opcode opcode, const PublishedNpuParameters &parameters,
    const PublishedNpuHardwareView &hardware = {});
PublishedNpuWork EvaluatePublishedNpuWork(
    const RopeQkExactOperands &operands);
PublishedNpuWork EvaluatePublishedNpuWork(
    const AttentionExactOperands &operands);
PublishedNpuWork EvaluatePublishedNpuWork(
    const EmbeddingLookupOperands &operands);
PublishedNpuWork EvaluatePublishedNpuWork(
    const GreedySampleOperands &operands);
PublishedNpuWork EvaluatePublishedNpuWork(
    const CrossEntropyForwardOperands &operands);
PublishedNpuWork EvaluatePublishedNpuWork(
    const CrossEntropyBackwardOperands &operands);
PublishedNpuWork EvaluatePublishedNpuWork(
    const SgdUpdateOperands &operands);

// Compatibility wrapper for the published v1 operation-count API.
NpuOps EvaluatePublishedNpuOps(
    Opcode opcode, const PublishedNpuParameters &parameters,
    const PublishedNpuHardwareView &hardware = {});
