#include "prims/gemm_input_dx_timing_prim.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace {
constexpr uint64_t kMaxProfile = (uint64_t{1} << 30) - 1;
constexpr uint64_t kSramExtent = uint64_t{1} << 16;

uint64_t CheckedMul(uint64_t a, uint64_t b, const char *field) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::overflow_error(std::string(field) + " overflow");
    return a * b;
}

bool Overlap(const GemmInputDxSramSpan &a, const GemmInputDxSramSpan &b) {
    return uint64_t{a.byte_address} < uint64_t{b.byte_address} + b.bytes &&
           uint64_t{b.byte_address} < uint64_t{a.byte_address} + a.bytes;
}

void CheckSpan(const GemmInputDxSramSpan &span, uint64_t bytes,
               GemmInputDxDType dtype, const char *role) {
    if (span.dtype != dtype || span.bytes != bytes || !bytes ||
        span.byte_address % 16 || span.byte_address > kSramExtent ||
        span.bytes > kSramExtent - span.byte_address)
        throw std::invalid_argument(std::string(role) +
                                    " typed physical SRAM span differs");
}
} // namespace

GemmInputDxTimingWork BuildGemmInputDxPhysicalWork(
    const GemmInputDxTimingTile &tile) {
    if (!tile.k || !tile.m || !tile.n ||
        tile.k > kMaxProfile || tile.m > kMaxProfile || tile.n > kMaxProfile)
        throw std::invalid_argument("GEMM dX requires positive 30-bit K/M/N");
    const uint64_t km = CheckedMul(tile.k, tile.m, "GEMM dX KxM");
    const uint64_t mn = CheckedMul(tile.m, tile.n, "GEMM dX MxN");
    const uint64_t kn = CheckedMul(tile.k, tile.n, "GEMM dX KxN");
    const uint64_t wbytes = CheckedMul(2, mn, "GEMM dX source W bytes");
    const uint64_t dybytes = CheckedMul(2, kn, "GEMM dX dY bytes");
    const uint64_t dxbytes = CheckedMul(4, km, "GEMM dX output bytes");
    CheckSpan(tile.weight, wbytes, GemmInputDxDType::FP16, "GEMM dX W");
    CheckSpan(tile.upstream, dybytes, GemmInputDxDType::FP16, "GEMM dX dY");
    CheckSpan(tile.output, dxbytes, GemmInputDxDType::FP32, "GEMM dX dX");
    if (Overlap(tile.weight, tile.upstream) ||
        Overlap(tile.weight, tile.output) ||
        Overlap(tile.upstream, tile.output))
        throw std::invalid_argument("GEMM dX independent SRAM spans overlap");
    GemmInputDxTimingWork work;
    work.tile = tile;
    work.fp16_weight_read_bytes = wbytes;
    work.fp16_upstream_read_bytes = dybytes;
    work.fp32_dx_read_modify_write_bytes = CheckedMul(2, dxbytes, "GEMM dX FP32 RMW");
    work.fma_ops = CheckedMul(tile.k, mn, "GEMM dX FMA");
    work.exu_flops = CheckedMul(2, work.fma_ops, "GEMM dX EXU");
    work.fp32_output_vec_ops = km;
    return work;
}

GemmInputDxTimingWork BuildGemmInputDxTimingWork(
    const GemmInputDxSourceWitness &source,
    const GemmInputDxTimingTile &tile) {
    const auto work = BuildGemmInputDxPhysicalWork(tile);
    if (source.forward_op_ref.empty() || source.weight_state_ref.empty() ||
        source.weight_load_state_ref != source.weight_state_ref ||
        source.loaded_state_version != source.source_state_version)
        throw std::invalid_argument("GEMM dX source StateABI/load provenance differs");
    if (source.activation_dtype != GemmInputDxDType::FP16 ||
        source.activation_rows != tile.k ||
        source.activation_hidden != tile.m ||
        source.activation_bytes != CheckedMul(2, work.fp32_output_vec_ops,
                                              "GEMM dX source X bytes") ||
        source.state_weight_dtype != GemmInputDxDType::FP16 ||
        source.state_weight_hidden != tile.m ||
        source.state_weight_output != tile.n ||
        source.state_weight_bytes != work.fp16_weight_read_bytes)
        throw std::invalid_argument("GEMM dX forward X/StateABI source geometry differs");
    return work;
}
