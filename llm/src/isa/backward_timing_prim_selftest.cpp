#include "isa/backward_timing_prim_selftest.h"

#include "prims/backward_timing_prims.h"
#include "utils/prim_utils.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

struct Checks {
    int checks = 0;
    int failures = 0;

    void Check(bool condition, const char *name) {
        ++checks;
        if (!condition) {
            ++failures;
            std::cerr << "[BACKWARD_TIMING_PRIM] FAIL " << name << '\n';
        }
    }

    template <typename Function>
    void Reject(const char *name, Function function) {
        ++checks;
        try {
            function();
        } catch (const std::invalid_argument &) {
            return;
        } catch (const std::overflow_error &) {
            return;
        } catch (const std::exception &error) {
            ++failures;
            std::cerr << "[BACKWARD_TIMING_PRIM] FAIL " << name
                      << " (wrong exception: " << error.what() << ")\n";
            return;
        }
        ++failures;
        std::cerr << "[BACKWARD_TIMING_PRIM] FAIL " << name
                  << " (accepted)\n";
    }
};

struct WireModeGuard {
    bool previous = prim_wire::LegacyCompatibilityEnabled();
    WireModeGuard() { prim_wire::SetLegacyCompatibility(false); }
    ~WireModeGuard() { prim_wire::SetLegacyCompatibility(previous); }
};

norm_backward_timing Norm() {
    norm_backward_timing prim;
    prim.inp_offset = 0;
    prim.data_offset = 256;
    prim.out_offset = 512;
    prim.param_value = {{"ROWS", 2}, {"HIDDEN", 16}, {"TP", 4},
                        {"MODE", 0}};
    prim.initialize();
    return prim;
}

attention_backward_timing Attention() {
    attention_backward_timing prim;
    prim.inp_offset = 0;
    prim.data_offset = 1024;
    prim.out_offset = 2048;
    prim.param_value = {{"TOKENS", 4}, {"RANK_HEADS", 2},
                        {"RANK_KV_HEADS", 1}, {"HEAD_DIM", 8},
                        {"TP", 2}, {"SEQUENCES", 2}, {"PAIRS", 6}};
    prim.initialize();
    return prim;
}

} // namespace

int RunBackwardTimingPrimSelfTest() {
    WireModeGuard mode;
    Checks state;
    auto norm = Norm();
    const BackwardTimingWork norm_work = norm.work();
    state.Check(norm_work.forward_input_bytes == 64 &&
                    norm_work.upstream_bytes == 64 &&
                    norm_work.output_bytes == 64 &&
                    norm_work.exu_ops == 0 && norm_work.sfu_ops == 2 &&
                    norm_work.vec_ops == 194,
                "norm rank SRAM and RMS work are exact");
    TaskCoreContext context(nullptr, nullptr, nullptr, nullptr, nullptr,
                            nullptr, nullptr, nullptr, nullptr, 0, 0);
    context.cid = 0;
    uint64_t dram = 0, exu = 0, sfu = 0, vec = 0;
    norm.taskCore(context, "", dram, exu, sfu, vec);
    state.Check(dram == 0 && exu == norm_work.exu_ops &&
                    sfu == norm_work.sfu_ops && vec == norm_work.vec_ops,
                "norm primitive charges its declared work");
    const auto norm_wire = norm.serialize();
    std::unique_ptr<PrimBase> norm_base(
        PrimFactory::getInstance().createPrim(
            PrimIdValue(PrimId::NORM_BACKWARD_TIMING), false, false));
    auto *norm_decoded = dynamic_cast<norm_backward_timing *>(norm_base.get());
    state.Check(norm_decoded != nullptr, "norm PrimId resolves to named factory");
    if (norm_decoded != nullptr) {
        norm_decoded->deserialize(norm_wire);
        state.Check(norm_decoded->inp_offset == 0 &&
                        norm_decoded->data_offset == 256 &&
                        norm_decoded->out_offset == 512 &&
                        norm_decoded->work().vec_ops == norm_work.vec_ops &&
                        norm_decoded->serialize() == norm_wire,
                    "norm strict wire round-trips addresses and rank work");
        auto corrupted = norm_wire;
        corrupted[0].range(127, 127) = 1;
        state.Reject("norm reserved Prim wire bit is rejected", [&] {
            norm_decoded->deserialize(corrupted);
        });
        corrupted = norm_wire;
        // Sorted wire parameters: HIDDEN, MODE, ROWS, TP.
        corrupted[1].range(67, 38) = sc_bv<30>(2);
        state.Reject("norm wire derivative mode tampering is rejected", [&] {
            norm_decoded->deserialize(corrupted);
        });
    }
    auto bad_norm = Norm();
    bad_norm.data_offset = 32;
    state.Reject("norm upstream overlapping tape is rejected", [&] {
        bad_norm.serialize();
    });
    bad_norm = Norm();
    bad_norm.param_value["MODE"] = 2;
    state.Reject("norm unknown derivative mode is rejected", [&] {
        bad_norm.serialize();
    });

    auto attention = Attention();
    const BackwardTimingWork attention_work = attention.work();
    state.Check(attention_work.forward_input_bytes == 256 &&
                    attention_work.upstream_bytes == 128 &&
                    attention_work.output_bytes == 256 &&
                    attention_work.exu_ops == 768 &&
                    attention_work.sfu_ops == 24 &&
                    attention_work.vec_ops == 384,
                "attention rank QKV, upstream, causal pairs and work are exact");
    dram = exu = sfu = vec = 0;
    attention.taskCore(context, "", dram, exu, sfu, vec);
    state.Check(dram == 0 && exu == attention_work.exu_ops &&
                    sfu == attention_work.sfu_ops &&
                    vec == attention_work.vec_ops,
                "attention primitive charges causal profile work");
    const auto attention_wire = attention.serialize();
    std::unique_ptr<PrimBase> attention_base(
        PrimFactory::getInstance().createPrim(
            PrimIdValue(PrimId::ATTENTION_BACKWARD_TIMING), false, false));
    auto *attention_decoded =
        dynamic_cast<attention_backward_timing *>(attention_base.get());
    state.Check(attention_decoded != nullptr,
                "attention PrimId resolves to named factory");
    if (attention_decoded != nullptr) {
        attention_decoded->deserialize(attention_wire);
        state.Check(attention_decoded->inp_offset == 0 &&
                        attention_decoded->data_offset == 1024 &&
                        attention_decoded->out_offset == 2048 &&
                        attention_decoded->work().exu_ops ==
                            attention_work.exu_ops &&
                        attention_decoded->serialize() == attention_wire,
                    "attention strict wire round-trips physical rank profile");
        auto corrupted = attention_wire;
        corrupted[0].range(127, 127) = 1;
        state.Reject("attention reserved Prim wire bit is rejected", [&] {
            attention_decoded->deserialize(corrupted);
        });
        corrupted = attention_wire;
        // Sorted wire parameters: HEAD_DIM, PAIRS, RANK_HEADS, ...
        corrupted[1].range(67, 38) = sc_bv<30>(7);
        state.Reject("attention wire causal pair tampering is rejected", [&] {
            attention_decoded->deserialize(corrupted);
        });
    }
    auto bad_attention = Attention();
    bad_attention.param_value["PAIRS"] = 7;
    state.Reject("attention noncausal work profile is rejected", [&] {
        bad_attention.serialize();
    });
    bad_attention = Attention();
    bad_attention.param_value["RANK_KV_HEADS"] = 3;
    state.Reject("attention invalid rank GQA layout is rejected", [&] {
        bad_attention.serialize();
    });
    bad_attention = Attention();
    bad_attention.out_offset = 1120;
    state.Reject("attention output overlapping upstream is rejected", [&] {
        bad_attention.serialize();
    });

    std::cout << "Backward timing Prim strict wire self-test: "
              << (state.failures == 0 ? "PASS" : "FAILURES=" +
                                         std::to_string(state.failures))
              << " (" << state.checks << " checks)\n";
    return state.failures;
}
