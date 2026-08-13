#include "isa/npu_cost_model_selftest.h"

#include "isa/npu_cost_model.h"
#include "isa/published_npu_ops_selftest.h"
#include "prims/comp_prims.h"

#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace {
void Check(IsaV1SelfTestResult &result, bool condition,
           const std::string &message) {
    ++result.checks;
    if (!condition) {
        result.failures.push_back(message);
    }
}

template <typename Function>
void Reject(IsaV1SelfTestResult &result, const std::string &message,
            Function &&function) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception &) {
        rejected = true;
    }
    Check(result, rejected, message);
}

NpuCostHardware Hardware() {
    NpuCostHardware hardware;
    hardware.exu_x_dims = 4;
    hardware.exu_count = 1;
    hardware.sfu_x_dims = 4;
    hardware.vec_x_dims = 4;
    hardware.vec_count = 1;
    hardware.compute_utilization = 1.0F;
    hardware.cycle_ns = 2;
    return hardware;
}

NpuOps EvaluateProductionOps(
    NpuBase &prim, TaskCoreContext &context,
    std::unordered_map<std::string, int> parameters) {
    prim.param_value = std::move(parameters);
    prim.initialize();
    uint64_t dram_time = 0;
    NpuOps ops;
    prim.taskCore(context, "cost_oracle", dram_time, ops.exu, ops.sfu, ops.vec);
    if (dram_time != 0)
        throw std::logic_error(
            "side-effect-free compute cost oracle touched memory");
    return ops;
}
} // namespace

IsaV1SelfTestResult CheckNpuCostModelSelfTest() {
    IsaV1SelfTestResult result;
    IsaV1SelfTestResult published = CheckPublishedNpuOpsSelfTest();
    result.checks += published.checks;
    for (std::string &failure : published.failures)
        result.failures.push_back("published ops: " + std::move(failure));
    const NpuCostHardware hardware = Hardware();

    int sram_address = 0;
#if USE_NB_DRAMSYS == 1
    TaskCoreContext context(nullptr, nullptr, nullptr, nullptr, &sram_address,
                            nullptr, nullptr, nullptr, nullptr, 0, 16);
#else
    TaskCoreContext context(nullptr, nullptr, nullptr, nullptr, nullptr,
                            &sram_address, nullptr, nullptr, nullptr, 0, 0, 16);
#endif
    auto check_ops = [&](const std::string &label, NpuBase &prim,
                         std::unordered_map<std::string, int> parameters,
                         NpuOps expected) {
        const NpuOps actual =
            EvaluateProductionOps(prim, context, std::move(parameters));
        Check(result, actual.exu == expected.exu &&
                          actual.sfu == expected.sfu &&
                          actual.vec == expected.vec,
              label + " production ops match the frozen oracle");
    };
    {
        gate_forward prim;
        check_ops("GATE", prim,
                  {{"B", 2}, {"T", 3}, {"C", 4}, {"E_N", 5}, {"K", 2}},
                  {120, 0, 0});
    }
    {
        Max_pool prim;
        check_ops("MAXPOOL", prim,
                  {{"B", 1}, {"W", 4}, {"H", 4}, {"C", 2},
                   {"pX", 0}, {"pY", 0}, {"sX", 2}, {"sY", 2},
                   {"kX", 2}, {"kY", 2}},
                  {0, 32, 0});
    }
    {
        Gelu_f prim;
        check_ops("GELU", prim, {{"N", 7}}, {0, 7, 28});
    }
    {
        silu_forward prim;
        check_ops("SILU", prim, {{"N", 7}}, {0, 7, 21});
    }
    {
        swiglu_forward prim;
        check_ops("SWIGLU", prim, {{"N", 7}}, {0, 7, 28});
    }
    {
        Relu_f prim;
        check_ops("RELU", prim, {{"N", 7}}, {7, 0, 0});
    }
    {
        Residual_f prim;
        check_ops("RESIDUAL", prim, {{"N", 7}}, {0, 0, 7});
    }
    {
        Split_matmul prim;
        check_ops("SPLIT_MATMUL", prim,
                  {{"B", 2}, {"T", 3}, {"C", 4}, {"dim", 1}, {"slice", 2}},
                  {0, 0, 0});
    }
    {
        Merge_matmul prim;
        check_ops("MERGE_MATMUL", prim,
                  {{"B", 2}, {"T", 3}, {"C", 4}, {"dim", 1}, {"slice", 2}},
                  {24, 0, 0});
    }
    {
        Dummy_p prim;
        check_ops("DUMMY", prim, {}, {10, 0, 0});
    }

    NpuCostSnapshot cost =
        CalculateNpuCost(NpuOps{64, 4, 4}, hardware, 0);
    Check(result, cost.exu_cycle_ns == 4 && cost.sfu_cycle_ns == 2 &&
                      cost.vec_cycle_ns == 2 && cost.compute_cycle_ns == 4,
          "EXU dominates and max cycle is selected");
    Check(result, cost.overlap_delay_ns == 4,
          "compute-only delay equals compute cycle");

    cost = CalculateNpuCost(NpuOps{32, 20, 4}, hardware, 3);
    Check(result, cost.exu_cycle_ns == 2 && cost.sfu_cycle_ns == 10 &&
                      cost.compute_cycle_ns == 10,
          "SFU dominates and max cycle is selected");
    Check(result, cost.overlap_delay_ns == 7,
          "DRAM shorter than compute overlaps exactly once");

    cost = CalculateNpuCost(NpuOps{32, 4, 24}, hardware, 12);
    Check(result, cost.vec_cycle_ns == 12 && cost.compute_cycle_ns == 12,
          "vector unit dominates and max cycle is selected");
    Check(result, cost.overlap_delay_ns == 0,
          "DRAM equal to compute leaves no extra delay");

    cost = CalculateNpuCost(NpuOps{32, 4, 24}, hardware, 13);
    Check(result, cost.overlap_delay_ns == 0,
          "DRAM longer than compute leaves no extra delay");

    cost = CalculateNpuCost(NpuOps{31, 3, 3}, hardware, 0);
    Check(result, cost.exu_cycle_ns == 1 && cost.sfu_cycle_ns == 0 &&
                      cost.vec_cycle_ns == 0,
          "historical cost divisions truncate instead of rounding up");

    NpuCostHardware bad = hardware;
    bad.exu_x_dims = 0;
    Reject(result, "zero EXU dimension is rejected", [&] {
        (void)CalculateNpuCost({}, bad, 0);
    });
    bad = hardware;
    bad.compute_utilization = 0.0F;
    Reject(result, "zero compute utilization is rejected", [&] {
        (void)CalculateNpuCost({}, bad, 0);
    });
    bad = hardware;
    bad.vec_x_dims = std::numeric_limits<uint64_t>::max();
    bad.vec_count = 2;
    Reject(result, "vector width multiplication overflow is rejected", [&] {
        (void)CalculateNpuCost({}, bad, 0);
    });
    bad = hardware;
    bad.sfu_x_dims = 1;
    bad.cycle_ns = 2;
    Reject(result, "SFU cycle multiplication overflow is rejected", [&] {
        (void)CalculateNpuCost(
            NpuOps{0, std::numeric_limits<uint64_t>::max(), 0}, bad, 0);
    });
    return result;
}
