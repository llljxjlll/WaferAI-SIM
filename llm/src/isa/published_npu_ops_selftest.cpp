#include "isa/published_npu_ops_selftest.h"

#include "isa/published_npu_ops.h"

#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void Check(IsaV1SelfTestResult &result, bool condition,
           const std::string &message) {
    ++result.checks;
    if (!condition) result.failures.push_back(message);
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

PublishedNpuHardwareView Hardware() {
    PublishedNpuHardwareView hardware;
    hardware.exu_x_dims = 4;
    hardware.exu_count = 1;
    hardware.sfu_x_dims = 4;
    hardware.vec_x_dims = 4;
    hardware.vec_count = 1;
    hardware.compute_utilization = 1.0F;
    hardware.cycle_ns = 2;
    return hardware;
}

void CheckOps(IsaV1SelfTestResult &result, const std::string &label,
              Opcode opcode, PublishedNpuParameters parameters,
              NpuOps expected,
              PublishedNpuHardwareView hardware = {}) {
    const NpuOps actual =
        EvaluatePublishedNpuOps(opcode, parameters, hardware);
    Check(result, actual.exu == expected.exu &&
                      actual.sfu == expected.sfu &&
                      actual.vec == expected.vec,
          label + " exact published operation counts");
}

} // namespace

IsaV1SelfTestResult CheckPublishedNpuOpsSelfTest() {
    IsaV1SelfTestResult result;
    const PublishedNpuHardwareView hardware = Hardware();

    CheckOps(result, "MATMUL minimal VEC branch", Opcode::MATMUL,
             {{"B", 1}, {"T", 1}, {"C", 1}, {"OC", 1}},
             {0, 0, 2}, hardware);
    CheckOps(result, "MATMUL typical EXU branch", Opcode::MATMUL,
             {{"B", 2}, {"T", 3}, {"C", 4}, {"OC", 5}},
             {896, 0, 0}, hardware);

    CheckOps(result, "CONV minimal", Opcode::CONV,
             {{"B", 1}, {"W", 1}, {"H", 1}, {"C", 1},
              {"pX", 0}, {"pY", 0}, {"sX", 1}, {"sY", 1},
              {"kX", 1}, {"kY", 1}, {"F", 1}},
             {2, 0, 0});
    CheckOps(result, "CONV typical", Opcode::CONV,
             {{"B", 2}, {"W", 5}, {"H", 4}, {"C", 3},
              {"pX", 1}, {"pY", 0}, {"sX", 2}, {"sY", 1},
              {"kX", 3}, {"kY", 2}, {"F", 4}},
             {2592, 0, 0});

    CheckOps(result, "MAXPOOL minimal", Opcode::MAXPOOL,
             {{"B", 1}, {"W", 1}, {"H", 1}, {"C", 1},
              {"pX", 0}, {"pY", 0}, {"sX", 1}, {"sY", 1},
              {"kX", 1}, {"kY", 1}},
             {0, 1, 0});
    CheckOps(result, "MAXPOOL typical", Opcode::MAXPOOL,
             {{"B", 2}, {"W", 5}, {"H", 4}, {"C", 3},
              {"pX", 1}, {"pY", 0}, {"sX", 2}, {"sY", 1},
              {"kX", 3}, {"kY", 2}},
             {0, 324, 0});

    CheckOps(result, "ATTENTION minimal", Opcode::ATTENTION,
             {{"B", 1}, {"T", 1}, {"C", 1}, {"NH", 1}, {"R", 1}},
             {4, 1, 2});
    CheckOps(result, "ATTENTION typical", Opcode::ATTENTION,
             {{"B", 2}, {"T", 3}, {"C", 4}, {"NH", 2}, {"R", 2}},
             {288, 36, 72});

    CheckOps(result, "GATE minimal", Opcode::GATE,
             {{"B", 1}, {"T", 1}, {"C", 1}, {"E_N", 1}, {"K", 1}},
             {1, 0, 0});
    CheckOps(result, "GATE typical", Opcode::GATE,
             {{"B", 2}, {"T", 3}, {"C", 4}, {"E_N", 5}, {"K", 2}},
             {120, 0, 0});

    CheckOps(result, "MOE_MATMUL minimal", Opcode::MOE_MATMUL,
             {{"B", 1}, {"T", 1}, {"C", 1}, {"OC", 1}, {"K", 1},
              {"E_N", 1}, {"is_merge", 0}, {"need_choose", 0}},
             {2, 0, 0}, hardware);
    CheckOps(result, "MOE_MATMUL typical merge", Opcode::MOE_MATMUL,
             {{"B", 2}, {"T", 3}, {"C", 4}, {"OC", 5}, {"K", 2},
              {"E_N", 4}, {"is_merge", 1}, {"need_choose", 0}},
             {540, 0, 0}, hardware);
    PublishedNpuHardwareView perf_hardware = hardware;
    perf_hardware.use_performance_gemm = true;
    CheckOps(result, "MOE_MATMUL performance mode", Opcode::MOE_MATMUL,
             {{"B", 2}, {"T", 3}, {"C", 4}, {"OC", 5}, {"K", 2},
              {"E_N", 4}, {"is_merge", 1}, {"need_choose", 0}},
             {640, 0, 0}, perf_hardware);

    CheckOps(result, "GELU minimal", Opcode::GELU, {{"N", 1}},
             {0, 1, 4});
    CheckOps(result, "GELU typical", Opcode::GELU, {{"N", 7}},
             {0, 7, 28});
    CheckOps(result, "SILU minimal", Opcode::SILU, {{"N", 1}},
             {0, 1, 3});
    CheckOps(result, "SILU typical", Opcode::SILU, {{"N", 7}},
             {0, 7, 21});
    CheckOps(result, "SWIGLU minimal", Opcode::SWIGLU, {{"N", 1}},
             {0, 1, 4});
    CheckOps(result, "SWIGLU typical", Opcode::SWIGLU, {{"N", 7}},
             {0, 7, 28});
    CheckOps(result, "RELU minimal", Opcode::RELU, {{"N", 1}},
             {1, 0, 0});
    CheckOps(result, "RELU typical", Opcode::RELU, {{"N", 7}},
             {7, 0, 0});
    CheckOps(result, "RESIDUAL minimal", Opcode::RESIDUAL, {{"N", 1}},
             {0, 0, 1});
    CheckOps(result, "RESIDUAL typical", Opcode::RESIDUAL, {{"N", 7}},
             {0, 0, 7});

    CheckOps(result, "LAYERNORM minimal", Opcode::LAYERNORM,
             {{"B", 1}, {"T", 1}, {"C", 1}}, {0, 1, 11});
    CheckOps(result, "LAYERNORM typical", Opcode::LAYERNORM,
             {{"B", 2}, {"T", 3}, {"C", 4}}, {0, 6, 210});
    CheckOps(result, "RMSNORM minimal frozen SFU", Opcode::RMSNORM,
             {{"B", 1}, {"T", 1}, {"C", 1}}, {0, 0, 5});
    CheckOps(result, "RMSNORM typical frozen SFU", Opcode::RMSNORM,
             {{"B", 2}, {"T", 3}, {"C", 4}}, {0, 0, 102});

    CheckOps(result, "ROPE minimal", Opcode::ROPE,
             {{"B", 1}, {"T", 1}, {"C", 1}, {"NH", 1}}, {0, 0, 3});
    CheckOps(result, "ROPE typical", Opcode::ROPE,
             {{"B", 2}, {"T", 3}, {"C", 8}, {"NH", 2}}, {0, 0, 72});

    CheckOps(result, "SPLIT_MATMUL minimal", Opcode::SPLIT_MATMUL,
             {{"B", 1}, {"T", 1}, {"C", 1}, {"dim", 1}, {"slice", 1}},
             {0, 0, 0});
    CheckOps(result, "SPLIT_MATMUL typical", Opcode::SPLIT_MATMUL,
             {{"B", 2}, {"T", 3}, {"C", 4}, {"dim", 2}, {"slice", 4}},
             {0, 0, 0});
    CheckOps(result, "MERGE_MATMUL minimal", Opcode::MERGE_MATMUL,
             {{"B", 1}, {"T", 1}, {"C", 1}, {"dim", 1}, {"slice", 1}},
             {1, 0, 0});
    CheckOps(result, "MERGE_MATMUL typical", Opcode::MERGE_MATMUL,
             {{"B", 2}, {"T", 3}, {"C", 4}, {"dim", 2}, {"slice", 4}},
             {24, 0, 0});

    CheckOps(result, "DUMMY minimal", static_cast<Opcode>(0x15), {},
             {10, 0, 0});
    CheckOps(result, "DUMMY typical", static_cast<Opcode>(0x15), {},
             {10, 0, 0});

    Reject(result, "missing published parameter is rejected", [&] {
        (void)EvaluatePublishedNpuOps(Opcode::GELU, {});
    });
    Reject(result, "negative published parameter is rejected", [&] {
        (void)EvaluatePublishedNpuOps(Opcode::GELU, {{"N", -1}});
    });
    Reject(result, "zero convolution divisor is rejected", [&] {
        (void)EvaluatePublishedNpuOps(
            Opcode::MAXPOOL,
            {{"B", 1}, {"W", 1}, {"H", 1}, {"C", 1},
             {"pX", 0}, {"pY", 0}, {"sX", 0}, {"sY", 1},
             {"kX", 1}, {"kY", 1}});
    });
    Reject(result, "published operation multiplication overflow is rejected",
           [&] {
               const int large = std::numeric_limits<int>::max();
               (void)EvaluatePublishedNpuOps(
                   Opcode::GATE,
                   {{"B", large}, {"T", large}, {"C", large},
                    {"E_N", large}, {"K", 1}});
           });
    Reject(result, "zero MATMUL hardware dimension is rejected", [&] {
        PublishedNpuHardwareView bad = hardware;
        bad.exu_x_dims = 0;
        (void)EvaluatePublishedNpuOps(
            Opcode::MATMUL,
            {{"B", 1}, {"T", 1}, {"C", 1}, {"OC", 1}}, bad);
    });
    Reject(result, "non-published operation cost is rejected", [&] {
        (void)EvaluatePublishedNpuOps(Opcode::DTE_SEND, {});
    });
    return result;
}
