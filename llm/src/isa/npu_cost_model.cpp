#include "isa/npu_cost_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
uint64_t CheckedMultiply(uint64_t lhs, uint64_t rhs,
                         const char *field) {
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs)
        throw std::overflow_error(std::string(field) + " exceeds uint64");
    return lhs * rhs;
}

void ValidateHardware(const NpuCostHardware &hardware) {
    if (hardware.exu_x_dims == 0 || hardware.exu_count == 0 ||
        hardware.sfu_x_dims == 0 || hardware.vec_x_dims == 0 ||
        hardware.vec_count == 0 || hardware.cycle_ns == 0)
        throw std::invalid_argument(
            "NPU cost hardware dimensions/count/cycle must be non-zero");
    if (!std::isfinite(hardware.compute_utilization) ||
        hardware.compute_utilization <= 0.0F)
        throw std::invalid_argument(
            "NPU compute utilization must be finite and positive");
}
} // namespace

NpuCostSnapshot CalculateNpuCost(const NpuOps &ops,
                                 const NpuCostHardware &hardware,
                                 uint64_t dram_time_ns) {
    ValidateHardware(hardware);

    // Preserve historical EXU truncation beyond a full hardware cycle.
    // Positive EXU work needs at least one cycle, even when a tiny tile's
    // fractional cycle rounds to zero nanoseconds. SFU/VEC retain their
    // historical integer division above one cycle and the same nonzero floor.
    const float exu_denominator =
        static_cast<float>(hardware.exu_x_dims) *
        static_cast<float>(hardware.exu_x_dims) * 2.0F *
        static_cast<float>(hardware.exu_count) *
        hardware.compute_utilization;
    if (!std::isfinite(exu_denominator) || exu_denominator <= 0.0F)
        throw std::overflow_error("NPU EXU denominator is not representable");
    const float raw_exu_cycles =
        static_cast<float>(ops.exu) / exu_denominator *
        static_cast<float>(hardware.cycle_ns);
    if (!std::isfinite(raw_exu_cycles) || raw_exu_cycles < 0.0F ||
        static_cast<long double>(raw_exu_cycles) >
            static_cast<long double>(std::numeric_limits<uint64_t>::max()))
        throw std::overflow_error("NPU EXU cycle count is not representable");

    NpuCostSnapshot result;
    result.ops = ops;
    result.exu_cycle_ns = std::max(
        static_cast<uint64_t>(raw_exu_cycles),
        ops.exu == 0 ? uint64_t{0} : hardware.cycle_ns);
    result.sfu_cycle_ns = std::max(
        CheckedMultiply(ops.sfu / hardware.sfu_x_dims, hardware.cycle_ns,
                        "NPU SFU cycle count"),
        ops.sfu == 0 ? uint64_t{0} : hardware.cycle_ns);
    result.vec_cycle_ns = std::max(
        CheckedMultiply(
            ops.vec / CheckedMultiply(hardware.vec_x_dims, hardware.vec_count,
                                      "NPU vector width"),
            hardware.cycle_ns, "NPU vector cycle count"),
        ops.vec == 0 ? uint64_t{0} : hardware.cycle_ns);
    result.compute_cycle_ns =
        std::max(result.exu_cycle_ns,
                 std::max(result.sfu_cycle_ns, result.vec_cycle_ns));
    result.dram_time_ns = dram_time_ns;
    result.overlap_delay_ns = result.compute_cycle_ns > dram_time_ns
                                  ? result.compute_cycle_ns - dram_time_ns
                                  : 0;
    return result;
}
