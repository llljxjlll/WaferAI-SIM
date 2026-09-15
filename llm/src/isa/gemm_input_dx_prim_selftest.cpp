#include "isa/gemm_input_dx_prim_selftest.h"

#include "prims/gemm_input_dx_npu_prim.h"
#include "utils/prim_utils.h"

#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
struct Checks {
    int count = 0;
    int failures = 0;
    void Check(bool pass, const char *label) {
        ++count;
        if (!pass) {
            ++failures;
            std::cerr << "[GEMM_DX] FAIL " << label << '\n';
        }
    }
    template <typename F> void Reject(const char *label, F operation) {
        ++count;
        try { operation(); }
        catch (const std::invalid_argument &) { return; }
        catch (const std::overflow_error &) { return; }
        catch (const std::exception &error) {
            ++failures;
            std::cerr << "[GEMM_DX] FAIL " << label << ": "
                      << error.what() << '\n';
            return;
        }
        ++failures;
        std::cerr << "[GEMM_DX] FAIL " << label << " accepted\n";
    }
};
struct StrictGuard {
    bool prior = prim_wire::LegacyCompatibilityEnabled();
    StrictGuard() { prim_wire::SetLegacyCompatibility(false); }
    ~StrictGuard() { prim_wire::SetLegacyCompatibility(prior); }
};
gemm_input_dx_timing Specimen() {
    gemm_input_dx_timing prim;
    prim.inp_offset = 0;   // FP16 W[M,N] 256B
    prim.data_offset = 512; // FP16 dY[K,N] 128B
    prim.out_offset = 1024; // FP32 dX[K,M] 128B
    prim.param_value = {{"M", 8}, {"N", 16}, {"K", 4}};
    prim.initialize();
    return prim;
}
} // namespace

int RunGemmInputDxTimingPrimSelfTest() {
    StrictGuard strict;
    Checks checks;
    auto prim = Specimen();
    const auto work = prim.work();
    checks.Check(work.tile.weight.dtype == GemmInputDxDType::FP16 &&
                 work.tile.weight.byte_address == 0 &&
                 work.tile.weight.bytes == 256 &&
                 work.tile.upstream.dtype == GemmInputDxDType::FP16 &&
                 work.tile.upstream.byte_address == 512 &&
                 work.tile.upstream.bytes == 128 &&
                 work.tile.output.dtype == GemmInputDxDType::FP32 &&
                 work.tile.output.byte_address == 1024 &&
                 work.tile.output.bytes == 128,
                 "separate real W[M,N], dY[K,N], FP32 dX[K,M] SRAM spans");
    checks.Check(work.fp16_weight_read_bytes == 256 &&
                 work.fp16_upstream_read_bytes == 128 &&
                 work.fp32_dx_read_modify_write_bytes == 256 &&
                 work.fma_ops == 512 && work.exu_flops == 1024 &&
                 work.fp32_output_vec_ops == 32 &&
                 prim.data_size_input.size() == 2 &&
                 prim.data_chunk.back().first == "output",
                 "FP16 source, FP32 output and worker timing work exact");
    TaskCoreContext context(nullptr, nullptr, nullptr, nullptr, nullptr,
                            nullptr, nullptr, nullptr, nullptr, 0, 0);
    context.cid = 0;
    uint64_t dram = 1, exu = 1, sfu = 1, vec = 1;
    prim.taskCore(context, "", dram, exu, sfu, vec);
    checks.Check(dram == 0 && exu == 1024 && sfu == 0 && vec == 32,
                 "worker charges named FP32 input dX timing");
    const auto wire = prim.serialize();
    std::unique_ptr<PrimBase> base(PrimFactory::getInstance().createPrim(
        PrimIdValue(PrimId::GEMM_INPUT_DX_TIMING), false, false));
    auto *decoded = dynamic_cast<gemm_input_dx_timing *>(base.get());
    checks.Check(decoded != nullptr, "Prim75 factory preserves native class");
    if (decoded) {
        decoded->deserialize(wire);
        checks.Check(decoded->work().tile.output.bytes == 128 &&
                     decoded->serialize() == wire,
                     "strict Prim wire roundtrip preserves FP32 dX geometry");
        auto corrupt = wire;
        corrupt[0].range(127, 127) = 1;
        checks.Reject("reserved Prim metadata", [&] {
            decoded->deserialize(corrupt);
        });
    }
    auto bad = Specimen();
    bad.param_value["K"] = 0;
    checks.Reject("zero rank rows", [&] { bad.serialize(); });
    bad = Specimen();
    bad.data_offset = 240;
    checks.Reject("W and dY SRAM overlap", [&] { bad.serialize(); });
    bad = Specimen();
    bad.out_offset = 600;
    checks.Reject("dX and dY SRAM overlap", [&] { bad.serialize(); });
    bad = Specimen();
    bad.out_offset = 65424;
    checks.Reject("FP32 dX escapes 16-bit SRAM", [&] { bad.serialize(); });
    bad = Specimen();
    bad.param_value["EXTRA"] = 1;
    checks.Reject("unexpected profile field", [&] { bad.serialize(); });
    prim_wire::SetLegacyCompatibility(true);
    checks.Reject("legacy wire cannot masquerade as typed dX", [&] {
        auto legacy = Specimen();
        legacy.serialize();
    });
    prim_wire::SetLegacyCompatibility(false);
    std::cout << "GEMM FP32 input dX strict Prim self-test: "
              << (checks.failures ? "FAIL" : "PASS") << " ("
              << checks.count << " checks)\n";
    return checks.failures;
}
