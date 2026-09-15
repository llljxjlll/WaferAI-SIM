#include "isa/gemm_weight_wgrad_timing_prim_selftest.h"

#include "prims/gemm_weight_wgrad_timing_prim.h"
#include "utils/prim_utils.h"

#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
struct Checks {
    int count = 0;
    int failures = 0;
    void Check(bool pass, const char *name) {
        ++count;
        if (!pass) {
            ++failures;
            std::cerr << "[GEMM_WGRAD] FAIL " << name << '\n';
        }
    }
    template <typename F> void Reject(const char *name, F action) {
        ++count;
        try { action(); }
        catch (const std::invalid_argument &) { return; }
        catch (const std::overflow_error &) { return; }
        catch (const std::exception &error) {
            ++failures;
            std::cerr << "[GEMM_WGRAD] FAIL " << name << ": "
                      << error.what() << '\n';
            return;
        }
        ++failures;
        std::cerr << "[GEMM_WGRAD] FAIL " << name << " accepted\n";
    }
};

struct StrictGuard {
    bool prior = prim_wire::LegacyCompatibilityEnabled();
    StrictGuard() { prim_wire::SetLegacyCompatibility(false); }
    ~StrictGuard() { prim_wire::SetLegacyCompatibility(prior); }
};

gemm_weight_wgrad_timing Specimen() {
    gemm_weight_wgrad_timing prim;
    prim.inp_offset = 0;
    prim.data_offset = 256;
    prim.out_offset = 1024;
    prim.param_value = {{"M", 8}, {"N", 16}, {"K", 4}};
    prim.initialize();
    return prim;
}
} // namespace

int RunGemmWeightWGradTimingPrimSelfTest() {
    StrictGuard strict;
    Checks checks;
    auto prim = Specimen();
    const auto work = prim.work();
    checks.Check(work.m == 8 && work.n == 16 && work.k == 4 &&
                 work.activation.dtype == WeightGradBufferDType::FP16 &&
                 work.activation.sram_byte_address == 0 &&
                 work.activation.bytes == 64 &&
                 work.upstream.dtype == WeightGradBufferDType::FP16 &&
                 work.upstream.sram_byte_address == 256 &&
                 work.upstream.bytes == 128 &&
                 work.gradient.dtype == WeightGradBufferDType::FP32 &&
                 work.gradient.sram_byte_address == 1024 &&
                 work.gradient.bytes == 512,
                 "physical X[K,M], dY[K,N], dW[M,N] typed SRAM spans exact");
    checks.Check(work.fp16_activation_read_bytes == 64 &&
                 work.fp16_upstream_read_bytes == 128 &&
                 work.fp32_gradient_read_modify_write_bytes == 1024 &&
                 work.fma_ops == 512 && work.exu_flops == 1024 &&
                 work.fp32_accumulator_vec_ops == 128 &&
                 prim.data_size_input.size() == 2 &&
                 prim.data_chunk.back().first == "output",
                 "true FP16 source/FP32 output bytes and FMA work are independent");
    TaskCoreContext context(nullptr, nullptr, nullptr, nullptr, nullptr,
                            nullptr, nullptr, nullptr, nullptr, 0, 0);
    context.cid = 0;
    uint64_t dram = 1, exu = 1, sfu = 1, vec = 1;
    prim.taskCore(context, "", dram, exu, sfu, vec);
    checks.Check(dram == 0 && exu == 1024 && sfu == 0 && vec == 128,
                 "worker charges rank-local FP16xFP16 to FP32 FMA work");
    const auto wire = prim.serialize();
    std::unique_ptr<PrimBase> base(PrimFactory::getInstance().createPrim(
        PrimIdValue(PrimId::GEMM_WEIGHT_WGRAD_TIMING), false, false));
    auto *decoded = dynamic_cast<gemm_weight_wgrad_timing *>(base.get());
    checks.Check(decoded != nullptr, "named Prim74 factory preserves runtime type");
    if (decoded) {
        decoded->deserialize(wire);
        checks.Check(decoded->work().gradient.bytes == 512 &&
                     decoded->serialize() == wire,
                     "strict Prim wire preserves FP32 physical gradient extent");
        auto corrupt = wire;
        corrupt[0].range(127, 127) = 1;
        checks.Reject("strict Prim metadata reserved bit", [&] {
            decoded->deserialize(corrupt);
        });
    }
    auto bad = Specimen();
    bad.param_value["K"] = 0;
    checks.Reject("zero source rank rows", [&] { bad.serialize(); });
    bad = Specimen();
    bad.param_value["M"] = 9;
    bad.data_offset = 64;
    checks.Reject("FP16 X and dY SRAM overlap", [&] { bad.serialize(); });
    bad = Specimen();
    bad.out_offset = 320;
    checks.Reject("FP32 dW overlaps FP16 dY", [&] { bad.serialize(); });
    bad = Specimen();
    bad.out_offset = 65040;
    checks.Reject("FP32 dW escapes 16-bit SRAM", [&] { bad.serialize(); });
    bad = Specimen();
    bad.param_value["K"] = 10000;
    checks.Reject("FP16 X tile wider than SRAM", [&] { bad.serialize(); });
    bad = Specimen();
    bad.param_value["EXTRA"] = 1;
    checks.Reject("profile has unauthorized extra field", [&] { bad.serialize(); });
    prim_wire::SetLegacyCompatibility(true);
    checks.Reject("legacy Prim wire refuses typed FP32 WGRAD", [&] {
        auto legacy = Specimen();
        legacy.serialize();
    });
    prim_wire::SetLegacyCompatibility(false);
    std::cout << "GEMM FP32 weight WGRAD strict Prim self-test: "
              << (checks.failures == 0 ? "PASS" : "FAIL")
              << " (" << checks.count << " checks)\n";
    return checks.failures;
}
