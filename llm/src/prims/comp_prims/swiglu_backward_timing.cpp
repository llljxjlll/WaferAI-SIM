#include "prims/base.h"

#include "prims/comp_prims.h"

REGISTER_PRIM(swiglu_backward_timing, PrimId::SWIGLU_BACKWARD_TIMING);

void swiglu_backward_timing::initialize() {
    const int n = param_value["N"];
    data_size_input = {2 * n};
    data_chunk = {{"upstream_activation_gradient", n},
                  {"output", 2 * n}};
}

void swiglu_backward_timing::taskCore(TaskCoreContext &, string,
                                     u_int64_t &, u_int64_t &exu_ops,
                                     u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    const uint64_t n = static_cast<uint64_t>(param_value["N"]);
    // Sigmoid gate and derivative, then product rule for gate and up.
    // Numerical derivatives remain outside program timing mode.
    exu_ops = 0;
    sfu_ops = 2 * n;
    vec_ops = 8 * n;
}
