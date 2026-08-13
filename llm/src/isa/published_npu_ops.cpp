#include "isa/published_npu_ops.h"

#include "common/config.h"
#include "defs/const.h"
#include "defs/spec.h"
#include "utils/system_utils.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

[[noreturn]] void Invalid(const std::string &message) {
    throw std::invalid_argument("published NPU ops: " + message);
}

uint64_t Parameter(const PublishedNpuParameters &parameters,
                   const char *name) {
    const auto found = parameters.find(name);
    if (found == parameters.end())
        Invalid(std::string("missing parameter ") + name);
    if (found->second < 0)
        Invalid(std::string("negative parameter ") + name);
    return static_cast<uint64_t>(found->second);
}

uint64_t PositiveParameter(const PublishedNpuParameters &parameters,
                           const char *name) {
    const uint64_t value = Parameter(parameters, name);
    if (value == 0)
        Invalid(std::string("zero parameter ") + name);
    return value;
}

uint64_t CheckedAdd(uint64_t lhs, uint64_t rhs, const char *field) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs)
        throw std::overflow_error(std::string("published NPU ops: ") +
                                  field + " exceeds uint64");
    return lhs + rhs;
}

uint64_t CheckedMultiply(uint64_t lhs, uint64_t rhs, const char *field) {
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs)
        throw std::overflow_error(std::string("published NPU ops: ") +
                                  field + " exceeds uint64");
    return lhs * rhs;
}

uint64_t Product(std::initializer_list<uint64_t> values,
                 const char *field) {
    uint64_t result = 1;
    for (const uint64_t value : values)
        result = CheckedMultiply(result, value, field);
    return result;
}

uint64_t CeilDivide(uint64_t value, uint64_t divisor,
                    const char *field) {
    if (divisor == 0)
        Invalid(std::string(field) + " divisor is zero");
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

uint64_t ScaledUtilization(uint64_t value, float utilization,
                           const char *field) {
    if (!std::isfinite(utilization) || utilization <= 0.0F)
        Invalid("compute utilization must be finite and positive");
    const float scaled = static_cast<float>(value) * utilization;
    if (!std::isfinite(scaled) || scaled < 0.0F ||
        static_cast<long double>(scaled) >
            static_cast<long double>(std::numeric_limits<uint64_t>::max()))
        throw std::overflow_error(std::string("published NPU ops: ") +
                                  field + " exceeds uint64");
    return static_cast<uint64_t>(scaled);
}

void ValidateMatmulHardware(const PublishedNpuHardwareView &hardware) {
    if (hardware.exu_x_dims == 0 || hardware.exu_count == 0 ||
        hardware.sfu_x_dims == 0 || hardware.vec_x_dims == 0 ||
        hardware.vec_count == 0 || hardware.cycle_ns == 0)
        Invalid("MATMUL hardware dimensions/count/cycle must be non-zero");
    if (!std::isfinite(hardware.compute_utilization) ||
        hardware.compute_utilization <= 0.0F)
        Invalid("compute utilization must be finite and positive");
}

uint64_t GemmPerformanceOps(uint64_t b, uint64_t t, uint64_t c,
                            uint64_t oc, uint64_t k,
                            const PublishedNpuHardwareView &hardware,
                            bool double_ops) {
    if (hardware.exu_x_dims == 0)
        Invalid("GEMM EXU dimension must be non-zero");
    const uint64_t tile_x =
        CeilDivide(c, hardware.exu_x_dims, "GEMM C tile");
    const uint64_t tile_y =
        CeilDivide(oc, hardware.exu_x_dims, "GEMM OC tile");
    const uint64_t input_rows = Product({b, t, k}, "GEMM input rows");
    const uint64_t padded_rows =
        std::max(input_rows, hardware.exu_x_dims);
    const uint64_t array_fill = CheckedMultiply(
        hardware.exu_x_dims, 2, "GEMM array fill");
    const uint64_t performance_cycle = Product(
        {CheckedAdd(array_fill, padded_rows, "GEMM performance cycle"),
         tile_x, tile_y},
        "GEMM performance cycle");
    const uint64_t array_area = Product(
        {hardware.exu_x_dims, hardware.exu_x_dims}, "GEMM array area");
    uint64_t result = ScaledUtilization(
        CheckedMultiply(performance_cycle, array_area,
                        "GEMM performance operations"),
        hardware.compute_utilization, "GEMM performance operations");
    if (double_ops)
        result = CheckedMultiply(result, 2, "GEMM performance operations");
    return result;
}

uint64_t ConvExtent(uint64_t input, uint64_t padding, uint64_t kernel,
                    uint64_t stride, const char *field) {
    if (kernel == 0 || stride == 0)
        Invalid(std::string(field) + " kernel/stride must be non-zero");
    const uint64_t padded = CheckedAdd(
        input, CheckedMultiply(padding, 2, field), field);
    if (padded < kernel)
        Invalid(std::string(field) + " kernel exceeds padded input");
    return CheckedAdd((padded - kernel) / stride, 1, field);
}

NpuOps EvaluateMatmul(const PublishedNpuParameters &parameters,
                      const PublishedNpuHardwareView &hardware) {
    ValidateMatmulHardware(hardware);
    const uint64_t b = PositiveParameter(parameters, "B");
    const uint64_t t = PositiveParameter(parameters, "T");
    const uint64_t c = PositiveParameter(parameters, "C");
    const uint64_t oc = PositiveParameter(parameters, "OC");
    NpuOps candidates;
    candidates.exu =
        GemmPerformanceOps(b, t, c, oc, 1, hardware, true);
    candidates.vec = Product({b, oc, t, c, 2}, "MATMUL vector ops");

    NpuCostHardware cost_hardware;
    cost_hardware.exu_x_dims = hardware.exu_x_dims;
    cost_hardware.exu_count = hardware.exu_count;
    cost_hardware.sfu_x_dims = hardware.sfu_x_dims;
    cost_hardware.vec_x_dims = hardware.vec_x_dims;
    cost_hardware.vec_count = hardware.vec_count;
    cost_hardware.compute_utilization = hardware.compute_utilization;
    cost_hardware.cycle_ns = hardware.cycle_ns;
    const NpuCostSnapshot cost =
        CalculateNpuCost(candidates, cost_hardware, 0);
    if (cost.vec_cycle_ns < cost.exu_cycle_ns)
        candidates.exu = 0;
    else
        candidates.vec = 0;
    return candidates;
}

NpuOps EvaluateMoeMatmul(const PublishedNpuParameters &parameters,
                         const PublishedNpuHardwareView &hardware) {
    const uint64_t b = PositiveParameter(parameters, "B");
    const uint64_t t = PositiveParameter(parameters, "T");
    const uint64_t c = PositiveParameter(parameters, "C");
    const uint64_t oc = PositiveParameter(parameters, "OC");
    const uint64_t k = Parameter(parameters, "K");
    const uint64_t experts = PositiveParameter(parameters, "E_N");
    const uint64_t is_merge = Parameter(parameters, "is_merge");
    const uint64_t need_choose = Parameter(parameters, "need_choose");
    if (k > experts)
        Invalid("MOE_MATMUL K exceeds E_N");
    if (is_merge > 1 || need_choose > 1)
        Invalid("MOE_MATMUL boolean parameter is outside [0, 1]");

    if (hardware.use_performance_gemm) {
        NpuOps result;
        result.exu =
            GemmPerformanceOps(b, t, c, oc, k, hardware, false);
        return result;
    }

    NpuOps result;
    result.exu = Product({b, t, c, oc, k, 2}, "MOE_MATMUL EXU ops");
    if (is_merge != 0)
        result.exu = CheckedAdd(
            result.exu, Product({b, t, oc, k}, "MOE_MATMUL merge ops"),
            "MOE_MATMUL EXU ops");
    return result;
}

} // namespace

PublishedNpuHardwareView PublishedNpuHardwareForCore(int core_id) {
    CoreHWConfig *core = GetCoreHWConfig(core_id);
    if (core == nullptr || core->exu == nullptr || core->sfu == nullptr ||
        core->vec == nullptr)
        Invalid("core hardware view is incomplete");
    if (core->exu->x_dims < 0 || core->exu->count < 0 ||
        core->sfu->x_dims < 0 || core->vec->x_dims < 0 ||
        core->vec->count < 0)
        Invalid("core hardware view contains a negative field");

    PublishedNpuHardwareView result;
    result.exu_x_dims = static_cast<uint64_t>(core->exu->x_dims);
    result.exu_count = static_cast<uint64_t>(core->exu->count);
    result.sfu_x_dims = static_cast<uint64_t>(core->sfu->x_dims);
    result.vec_x_dims = static_cast<uint64_t>(core->vec->x_dims);
    result.vec_count = static_cast<uint64_t>(core->vec->count);
    result.compute_utilization = HW_COMP_UTIL;
    result.cycle_ns = CYCLE;
    result.use_performance_gemm = SPEC_USE_PERF_GEMM;
    return result;
}

NpuOps EvaluatePublishedNpuOps(Opcode opcode,
                               const PublishedNpuParameters &parameters,
                               const PublishedNpuHardwareView &hardware) {
    switch (opcode) {
    case Opcode::MATMUL:
        return EvaluateMatmul(parameters, hardware);
    case Opcode::CONV:
    case Opcode::MAXPOOL: {
        const uint64_t b = PositiveParameter(parameters, "B");
        (void)PositiveParameter(parameters, "W");
        (void)PositiveParameter(parameters, "H");
        const uint64_t c = PositiveParameter(parameters, "C");
        const uint64_t px = Parameter(parameters, "pX");
        const uint64_t py = Parameter(parameters, "pY");
        const uint64_t sx = PositiveParameter(parameters, "sX");
        const uint64_t sy = PositiveParameter(parameters, "sY");
        const uint64_t kx = PositiveParameter(parameters, "kX");
        const uint64_t ky = PositiveParameter(parameters, "kY");
        const uint64_t oh = ConvExtent(
            Parameter(parameters, "H"), py, ky, sy, "compute H");
        const uint64_t ow = ConvExtent(
            Parameter(parameters, "W"), px, kx, sx, "compute W");
        NpuOps result;
        if (opcode == Opcode::CONV) {
            const uint64_t f = PositiveParameter(parameters, "F");
            result.exu = Product({b, c, ky, kx, 2, oh, ow, f},
                                 "CONV EXU ops");
        } else {
            result.sfu = Product({b, c, oh, ow, kx, ky},
                                 "MAXPOOL SFU ops");
        }
        return result;
    }
    case Opcode::ATTENTION: {
        const uint64_t b = PositiveParameter(parameters, "B");
        const uint64_t t = PositiveParameter(parameters, "T");
        const uint64_t c = PositiveParameter(parameters, "C");
        const uint64_t nh = PositiveParameter(parameters, "NH");
        (void)PositiveParameter(parameters, "R");
        return {Product({b, c, t, t, 4}, "ATTENTION EXU ops"),
                Product({b, nh, t, t}, "ATTENTION SFU ops"),
                Product({b, nh, t, t, 2}, "ATTENTION vector ops")};
    }
    case Opcode::GATE: {
        const uint64_t b = PositiveParameter(parameters, "B");
        const uint64_t t = PositiveParameter(parameters, "T");
        const uint64_t c = PositiveParameter(parameters, "C");
        const uint64_t experts = PositiveParameter(parameters, "E_N");
        const uint64_t k = PositiveParameter(parameters, "K");
        if (k > experts) Invalid("GATE K exceeds E_N");
        return {Product({b, t, c, experts}, "GATE EXU ops"), 0, 0};
    }
    case Opcode::MOE_MATMUL:
        return EvaluateMoeMatmul(parameters, hardware);
    case Opcode::GELU:
    case Opcode::SILU:
    case Opcode::SWIGLU: {
        const uint64_t n = PositiveParameter(parameters, "N");
        const uint64_t scale = opcode == Opcode::SILU ? 3 : 4;
        return {0, n, CheckedMultiply(n, scale, "activation vector ops")};
    }
    case Opcode::RELU: {
        const uint64_t n = PositiveParameter(parameters, "N");
        return {n, 0, 0};
    }
    case Opcode::RESIDUAL: {
        const uint64_t n = PositiveParameter(parameters, "N");
        return {0, 0, n};
    }
    case Opcode::LAYERNORM:
    case Opcode::RMSNORM: {
        const uint64_t b = PositiveParameter(parameters, "B");
        const uint64_t t = PositiveParameter(parameters, "T");
        const uint64_t c = PositiveParameter(parameters, "C");
        const uint64_t rows = CheckedMultiply(b, t, "normalization rows");
        const uint64_t scale = CheckedAdd(
            CheckedMultiply(opcode == Opcode::LAYERNORM ? 8 : 4, c,
                            "normalization vector scale"),
            opcode == Opcode::LAYERNORM ? 3 : 1,
            "normalization vector scale");
        return {0, opcode == Opcode::LAYERNORM ? rows : 0,
                CheckedMultiply(rows, scale, "normalization vector ops")};
    }
    case Opcode::ROPE: {
        (void)PositiveParameter(parameters, "B");
        const uint64_t t = PositiveParameter(parameters, "T");
        const uint64_t c = PositiveParameter(parameters, "C");
        const uint64_t nh = PositiveParameter(parameters, "NH");
        if (c / nh == 0) Invalid("ROPE C/NH must be non-zero");
        return {0, 0, Product({3, t, c}, "ROPE vector ops")};
    }
    case Opcode::SPLIT_MATMUL:
    case Opcode::MERGE_MATMUL: {
        const uint64_t b = PositiveParameter(parameters, "B");
        const uint64_t t = PositiveParameter(parameters, "T");
        const uint64_t c = PositiveParameter(parameters, "C");
        const uint64_t dim = PositiveParameter(parameters, "dim");
        const uint64_t slice = PositiveParameter(parameters, "slice");
        if ((dim != 1 && dim != 2) || slice > 16)
            Invalid("split/merge MATMUL dim/slice is invalid");
        if (opcode == Opcode::SPLIT_MATMUL) return {};
        return {Product({b, t, c}, "MERGE_MATMUL EXU ops"), 0, 0};
    }
    // DUMMY is also a legacy build macro after simulator headers are included.
    case static_cast<Opcode>(0x15):
        return {10, 0, 0};
    default:
        Invalid("opcode is not a published v1 NPU cost operation");
    }
}
