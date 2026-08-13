#include "systemc.h"

#include "isa/published_npu_ops.h"

#include "memory/dram/Dcachecore.h"
#include "prims/base.h"
#include "prims/comp_prims.h"
#include "utils/memory_utils.h"
#include "utils/prim_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(Gelu_f, PrimId::GELU_F);

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

void Gelu_f::initialize() {
    auto &p = param_value;
    data_size_input = {p["N"]};
    data_chunk = {{"output", p["N"]}};
}

void Gelu_f::taskCore(TaskCoreContext &context, string prim_name,
                      u_int64_t &dram_time, u_int64_t &exu_ops,
                      u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    const NpuOps ops = EvaluatePublishedNpuOps(Opcode::GELU, param_value);
    exu_ops = ops.exu;
    sfu_ops = ops.sfu;
    vec_ops = ops.vec;
}