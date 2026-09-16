#include "prims/gemm_input_dx_timing_prim.h"

#include <iostream>
#include <stdexcept>

namespace {
template <typename F> void Reject(const char *reason, F operation) {
    try { operation(); }
    catch (const std::invalid_argument &) { return; }
    catch (const std::overflow_error &) { return; }
    throw std::runtime_error(std::string(reason) + " incorrectly accepted");
}
} // namespace

int main() {
    GemmInputDxSourceWitness source{
        "T0.layer0.gate_up", "weight.state0", "weight.state0",
        2, 2, 4, 8, 64, GemmInputDxDType::FP16,
        8, 16, 256, GemmInputDxDType::FP16,
    };
    GemmInputDxTimingTile tile{
        4, 8, 16,
        {0, 256, GemmInputDxDType::FP16},
        {512, 128, GemmInputDxDType::FP16},
        {1024, 64, GemmInputDxDType::FP16},
    };
    const auto work = BuildGemmInputDxTimingWork(source, tile);
    if (work.fp16_weight_read_bytes != 256 ||
        work.fp16_upstream_read_bytes != 128 ||
        work.fp16_dx_write_bytes != 64 ||
        work.fma_ops != 512 || work.exu_flops != 1024 ||
        work.fp16_output_vec_ops != 32)
        throw std::runtime_error("true GEMM dX source/work contract differs");
    auto bad_source = source;
    bad_source.activation_hidden = 16;
    Reject("wrong source X[K,M]", [&] { BuildGemmInputDxTimingWork(bad_source, tile); });
    bad_source = source;
    bad_source.weight_load_state_ref = "unrelated.state";
    Reject("StateABI LOAD substitution", [&] { BuildGemmInputDxTimingWork(bad_source, tile); });
    bad_source = source;
    bad_source.loaded_state_version = 1;
    Reject("stale weight version", [&] { BuildGemmInputDxTimingWork(bad_source, tile); });
    bad_source = source;
    bad_source.state_weight_bytes = 128;
    Reject("FP16-sized weight source", [&] { BuildGemmInputDxTimingWork(bad_source, tile); });
    auto bad_tile = tile;
    bad_tile.output.bytes = 128;
    Reject("legacy FP32-sized dX output", [&] { BuildGemmInputDxTimingWork(source, bad_tile); });
    bad_tile = tile;
    bad_tile.output.dtype = GemmInputDxDType::FP32;
    Reject("legacy FP32 dX substituted", [&] { BuildGemmInputDxTimingWork(source, bad_tile); });
    bad_tile = tile;
    bad_tile.upstream.byte_address = 240;
    Reject("W/dY physical overlap", [&] { BuildGemmInputDxTimingWork(source, bad_tile); });
    bad_tile = tile;
    bad_tile.output.byte_address = 65488;
    Reject("dX outside 16-bit SRAM", [&] { BuildGemmInputDxTimingWork(source, bad_tile); });
    bad_tile = tile;
    bad_tile.k = 0;
    Reject("zero rows", [&] { BuildGemmInputDxTimingWork(source, bad_tile); });
    bad_tile = tile;
    bad_tile.m = uint64_t{1} << 30;
    Reject("profile wider than 30-bit wire", [&] { BuildGemmInputDxTimingWork(source, bad_tile); });
    std::cout << "GEMM FP16 input dX NEW-only physical contract: PASS (11 checks)\n";
}
