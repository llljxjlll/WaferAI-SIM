#include "isa/weight_gradient_timing_prim_selftest.h"

#include "prims/weight_gradient_timing_prims.h"
#include "utils/prim_utils.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

struct Checks {
    int count = 0;
    int failures = 0;
    void Check(bool result, const char *label) {
        ++count;
        if (!result) {
            ++failures;
            std::cerr << "[WEIGHT_GRAD_TIMING] FAIL " << label << '\n';
        }
    }
    template <typename F> void Reject(const char *label, F action) {
        ++count;
        try { action(); }
        catch (const std::invalid_argument &) { return; }
        catch (const std::overflow_error &) { return; }
        catch (const std::exception &error) {
            ++failures;
            std::cerr << "[WEIGHT_GRAD_TIMING] FAIL " << label
                      << " wrong exception " << error.what() << '\n';
            return;
        }
        ++failures;
        std::cerr << "[WEIGHT_GRAD_TIMING] FAIL " << label
                  << " accepted\n";
    }
};

struct StrictGuard {
    bool prior = prim_wire::LegacyCompatibilityEnabled();
    StrictGuard() { prim_wire::SetLegacyCompatibility(false); }
    ~StrictGuard() { prim_wire::SetLegacyCompatibility(prior); }
};

embedding_table_wgrad_timing Embedding() {
    embedding_table_wgrad_timing prim;
    prim.inp_offset = 0;
    prim.data_offset = 1024;
    prim.out_offset = 2048;
    prim.param_value = {{"ROWS", 3}, {"LOGICAL_ROWS", 6}, {"TP", 2},
                        {"VOCAB_SIZE", 128}, {"VOCAB_START", 16},
                        {"VOCAB_ROWS", 8}, {"HIDDEN", 8},
                        {"TABLE_ADDR", 256}};
    for (unsigned i = 0; i < 16; ++i)
        prim.param_value["INDEX" + std::to_string(i / 10) +
                         std::to_string(i % 10)] = 0;
    prim.param_value["INDEX00"] = 17;
    prim.param_value["INDEX01"] = 17;
    prim.param_value["INDEX02"] = 65; // Routed to a different vocab tile.
    prim.initialize();
    return prim;
}

norm_gamma_wgrad_timing Gamma() {
    norm_gamma_wgrad_timing prim;
    prim.inp_offset = 0;
    prim.data_offset = 256;
    prim.out_offset = 512;
    prim.param_value = {{"ROWS", 2}, {"LOGICAL_ROWS", 8},
                        {"TP", 4}, {"HIDDEN", 16}, {"MODE", 0}};
    prim.initialize();
    return prim;
}

} // namespace

int RunWeightGradientTimingPrimSelfTest() {
    StrictGuard guard;
    Checks state;
    auto embedding = Embedding();
    const auto work = embedding.work();
    state.Check(work.indices.dtype == WeightGradBufferDType::INT32 &&
                    work.indices.bytes == 12 &&
                    work.table.dtype == WeightGradBufferDType::FP16 &&
                    work.table.bytes == 128 &&
                    work.upstream.bytes == 48 &&
                    work.gradient.dtype == WeightGradBufferDType::FP32 &&
                    work.gradient.bytes == 256,
                "embedding typed INT32/FP16/FP32 SRAM spans are exact");
    state.Check(work.rank_rows == 3 && work.selected_rows == 2 &&
                    work.unique_weight_rows == 1 && work.row_collisions == 1 &&
                    work.selected_weight_sram_addresses.size() == 2 &&
                    work.selected_weight_sram_addresses[0] == 272 &&
                    work.selected_weight_sram_addresses[1] == 272 &&
                    work.selected_gradient_sram_addresses[0] == 2080 &&
                    work.selected_gradient_sram_addresses[1] == 2080 &&
                    work.trace_vec_ops == 3 && work.scatter_vec_ops == 16 &&
                    work.fp32_accumulate_vec_ops == 16 &&
                    work.fp32_read_modify_write_bytes == 128,
                "real source indices route to selected FP16 and FP32 rows with collision work");
    TaskCoreContext context(nullptr, nullptr, nullptr, nullptr, nullptr,
                            nullptr, nullptr, nullptr, nullptr, 0, 0);
    context.cid = 0;
    uint64_t dram = 5, exu = 5, sfu = 5, vec = 5;
    embedding.taskCore(context, "", dram, exu, sfu, vec);
    state.Check(dram == 0 && exu == 0 && sfu == 0 && vec == 35,
                "embedding worker charges index trace then scatter accumulation");
    const auto wire = embedding.serialize();
    std::unique_ptr<PrimBase> base(PrimFactory::getInstance().createPrim(
        PrimIdValue(PrimId::EMBEDDING_TABLE_WGRAD_TIMING), false, false));
    auto *decoded = dynamic_cast<embedding_table_wgrad_timing *>(base.get());
    state.Check(decoded != nullptr, "named embedding PrimId creates runtime type");
    if (decoded) {
        decoded->deserialize(wire);
        state.Check(decoded->work().selected_gradient_sram_addresses ==
                        work.selected_gradient_sram_addresses &&
                        decoded->serialize() == wire,
                    "embedding strict wire preserves physical trace and output ABI");
        auto bad = wire;
        bad[0].range(127, 127) = 1;
        state.Reject("embedding metadata reserved bit", [&] {
            decoded->deserialize(bad);
        });
        bad = wire;
        // Sorted parameters begin HIDDEN, INDEX00, INDEX01, INDEX02.
        bad[1].range(67, 38) = sc_bv<30>(128);
        state.Reject("embedding wire nonzero source index out of vocab", [&] {
            decoded->deserialize(bad);
        });
    }
    auto bad_embedding = Embedding();
    bad_embedding.param_value["INDEX01"] = 128;
    state.Reject("global index out of vocabulary", [&] { bad_embedding.serialize(); });
    bad_embedding = Embedding();
    bad_embedding.param_value["INDEX03"] = 17;
    state.Reject("noncanonical unused trace index", [&] { bad_embedding.serialize(); });
    bad_embedding = Embedding();
    bad_embedding.param_value["VOCAB_START"] = 125;
    state.Reject("vocabulary tile exceeds full vocabulary", [&] {
        bad_embedding.serialize();
    });
    bad_embedding = Embedding();
    bad_embedding.param_value["TABLE_ADDR"] = 0;
    state.Reject("FP16 table overlaps INT32 index source", [&] {
        bad_embedding.serialize();
    });
    bad_embedding = Embedding();
    bad_embedding.param_value["VOCAB_ROWS"] = 2048;
    bad_embedding.param_value["VOCAB_SIZE"] = 4096;
    state.Reject("FP32 tile exceeds 16-bit SRAM", [&] {
        bad_embedding.serialize();
    });

    auto gamma = Gamma();
    const auto gamma_work = gamma.work();
    state.Check(gamma_work.activation.bytes == 64 &&
                    gamma_work.upstream.bytes == 64 &&
                    gamma_work.gamma_gradient.dtype == WeightGradBufferDType::FP32 &&
                    gamma_work.gamma_gradient.bytes == 64 &&
                    gamma_work.rank_rows == 2 && gamma_work.hidden == 16 &&
                    gamma_work.normalization_vec_ops == 66 &&
                    gamma_work.fp32_accumulate_vec_ops == 32 &&
                    gamma_work.sfu_ops == 2 &&
                    gamma_work.fp32_read_modify_write_bytes == 256,
                "Norm gamma uses FP16 tape/upstream and independent FP32 hidden gradient");
    dram = exu = sfu = vec = 5;
    gamma.taskCore(context, "", dram, exu, sfu, vec);
    state.Check(dram == 0 && exu == 0 && sfu == 2 && vec == 98,
                "Norm gamma worker charges rank normalization and FP32 accumulation");
    const auto gamma_wire = gamma.serialize();
    std::unique_ptr<PrimBase> gamma_base(PrimFactory::getInstance().createPrim(
        PrimIdValue(PrimId::NORM_GAMMA_WGRAD_TIMING), false, false));
    auto *gamma_decoded = dynamic_cast<norm_gamma_wgrad_timing *>(gamma_base.get());
    state.Check(gamma_decoded != nullptr, "named Norm gamma PrimId creates runtime type");
    if (gamma_decoded) {
        gamma_decoded->deserialize(gamma_wire);
        state.Check(gamma_decoded->work().gamma_gradient.sram_byte_address == 512 &&
                        gamma_decoded->serialize() == gamma_wire,
                    "Norm gamma strict wire round-trips separate FP32 address");
        auto bad = gamma_wire;
        bad[0].range(127, 127) = 1;
        state.Reject("Norm gamma metadata reserved bit", [&] {
            gamma_decoded->deserialize(bad);
        });
        bad = gamma_wire;
        // Sorted parameters begin HIDDEN, LOGICAL_ROWS, MODE, ROWS.
        bad[1].range(67, 38) = sc_bv<30>(7);
        state.Reject("Norm gamma wire invalid TP rank shape", [&] {
            gamma_decoded->deserialize(bad);
        });
    }
    auto bad_gamma = Gamma();
    bad_gamma.param_value["LOGICAL_ROWS"] = 7;
    state.Reject("Norm gamma inconsistent TP rank rows", [&] {
        bad_gamma.serialize();
    });
    bad_gamma = Gamma();
    bad_gamma.out_offset = 272;
    state.Reject("Norm gamma FP32 output overlaps upstream", [&] {
        bad_gamma.serialize();
    });
    bad_gamma = Gamma();
    bad_gamma.param_value["MODE"] = 2;
    state.Reject("Norm gamma unsupported derivative mode", [&] {
        bad_gamma.serialize();
    });
    std::cout << "Weight gradient timing Prim strict wire self-test: "
              << (state.failures == 0 ? "PASS" : "FAILURES=" +
                  std::to_string(state.failures)) << " (" << state.count
              << " checks)\n";
    return state.failures;
}
