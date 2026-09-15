#include "prims/gemm_weight_wgrad_timing_prim.h"

#include "utils/prim_utils.h"

#include <climits>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

REGISTER_PRIM(gemm_weight_wgrad_timing, PrimId::GEMM_WEIGHT_WGRAD_TIMING);

namespace {
constexpr uint64_t kMaxProfile = (uint64_t{1} << 30) - 1;

uint64_t CheckedMul(uint64_t lhs, uint64_t rhs, const char *field) {
    if (rhs && lhs > std::numeric_limits<uint64_t>::max() / rhs)
        throw std::overflow_error(std::string(field) + " overflow");
    return lhs * rhs;
}

uint64_t CheckedAdd(uint64_t lhs, uint64_t rhs, const char *field) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs)
        throw std::overflow_error(std::string(field) + " overflow");
    return lhs + rhs;
}

uint64_t Param(const NpuBase &prim, const char *field) {
    const auto it = prim.param_value.find(field);
    if (it == prim.param_value.end() || it->second <= 0)
        throw std::invalid_argument(prim.name + " requires positive " + field);
    const uint64_t value = static_cast<uint64_t>(it->second);
    if (value > kMaxProfile)
        throw std::invalid_argument(prim.name + " profile exceeds 30-bit wire");
    return value;
}

WeightGradBufferABI Buffer(int address, uint64_t bytes,
                           WeightGradBufferDType dtype,
                           const char *field) {
    if (address < 0 || address % 16 || bytes == 0)
        throw std::invalid_argument(std::string(field) +
                                    " requires aligned nonempty SRAM span");
    const uint64_t end = CheckedAdd(static_cast<uint64_t>(address), bytes, field);
    if (end > uint64_t{UINT16_MAX} + 1)
        throw std::invalid_argument(std::string(field) +
                                    " escapes 16-bit SRAM region");
    const uint64_t scalar_bytes = dtype == WeightGradBufferDType::FP32 ? 4 : 2;
    if (bytes % scalar_bytes || bytes / 2 > INT_MAX)
        throw std::invalid_argument(std::string(field) +
                                    " typed bytes exceed NpuBase ABI");
    return {static_cast<uint32_t>(address), bytes, dtype};
}

bool Overlap(const WeightGradBufferABI &a, const WeightGradBufferABI &b) {
    return a.sram_byte_address < b.sram_byte_address + b.bytes &&
           b.sram_byte_address < a.sram_byte_address + a.bytes;
}

void Strict(const NpuBase &prim) {
    if (prim_wire::LegacyCompatibilityEnabled())
        throw std::invalid_argument(prim.name + " requires strict Prim wire");
}
} // namespace

gemm_weight_wgrad_timing::gemm_weight_wgrad_timing() {
    name = "gemm_weight_wgrad_timing";
    datatype = FP16; // Physical output remains independent FP32 BufferABI.
    skip_input = true;
    skip_output = true;
    param_name = {"M", "N", "K"};
}

GemmWeightWGradWork gemm_weight_wgrad_timing::work() const {
    if (datatype != FP16 || param_value.size() != param_name.size())
        throw std::invalid_argument(name + " requires exact FP16 source profile");
    GemmWeightWGradWork result;
    result.m = Param(*this, "M");
    result.n = Param(*this, "N");
    result.k = Param(*this, "K");
    const uint64_t km = CheckedMul(result.k, result.m, "GEMM KxM");
    const uint64_t kn = CheckedMul(result.k, result.n, "GEMM KxN");
    const uint64_t mn = CheckedMul(result.m, result.n, "GEMM MxN");
    result.fp16_activation_read_bytes =
        CheckedMul(2, km, "GEMM FP16 activation bytes");
    result.fp16_upstream_read_bytes =
        CheckedMul(2, kn, "GEMM FP16 upstream bytes");
    const uint64_t gradient_bytes =
        CheckedMul(4, mn, "GEMM FP32 gradient bytes");
    result.activation = Buffer(inp_offset,
        result.fp16_activation_read_bytes, WeightGradBufferDType::FP16,
        "GEMM activation");
    result.upstream = Buffer(data_offset,
        result.fp16_upstream_read_bytes, WeightGradBufferDType::FP16,
        "GEMM upstream");
    result.gradient = Buffer(out_offset, gradient_bytes,
        WeightGradBufferDType::FP32, "GEMM gradient");
    if (Overlap(result.activation, result.upstream) ||
        Overlap(result.activation, result.gradient) ||
        Overlap(result.upstream, result.gradient))
        throw std::invalid_argument(name + " typed physical SRAM spans overlap");
    result.fma_ops = CheckedMul(result.k, mn, "GEMM WGRAD FMA");
    result.exu_flops = CheckedMul(2, result.fma_ops, "GEMM WGRAD EXU");
    result.fp32_accumulator_vec_ops = mn;
    result.fp32_gradient_read_modify_write_bytes =
        CheckedMul(2, gradient_bytes, "GEMM FP32 RMW bytes");
    if (result.exu_flops > std::numeric_limits<u_int64_t>::max())
        throw std::overflow_error(name + " EXU work exceeds worker ABI");
    return result;
}

void gemm_weight_wgrad_timing::initialize() {
    const auto profile = work();
    data_size_input = {
        static_cast<int>(profile.activation.bytes / 2),
        static_cast<int>(profile.upstream.bytes / 2),
    };
    data_chunk = {{"upstream", static_cast<int>(profile.upstream.bytes / 2)},
                  {"output", static_cast<int>(profile.gradient.bytes / 2)}};
}

void gemm_weight_wgrad_timing::taskCore(TaskCoreContext &, string,
                                         u_int64_t &dram, u_int64_t &exu,
                                         u_int64_t &sfu, u_int64_t &vec) {
    const auto profile = work();
    dram = sfu = 0;
    exu = profile.exu_flops;
    vec = profile.fp32_accumulator_vec_ops;
}

vector<sc_bv<128>> gemm_weight_wgrad_timing::serialize() {
    Strict(*this);
    work();
    return NpuBase::serialize();
}

void gemm_weight_wgrad_timing::deserialize(vector<sc_bv<128>> wire) {
    Strict(*this);
    NpuBase::deserialize(std::move(wire));
    work();
}
