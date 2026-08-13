#include "systemc.h"

#include "isa/published_npu_ops.h"

#include "memory/dram/Dcachecore.h"
#include "prims/base.h"
#include "prims/comp_prims.h"
#include "utils/memory_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(Relu_f, PrimId::RELU_F);

void Relu_f::initialize() {
    auto &p = param_value;
    data_size_input = {p["N"]};
    data_chunk = {{"output", p["N"]}};
}

void Relu_f::taskCore(TaskCoreContext &context, string prim_name,
                     u_int64_t &dram_time, u_int64_t &exu_ops,
                     u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    const NpuOps ops = EvaluatePublishedNpuOps(Opcode::RELU, param_value);
    exu_ops = ops.exu;
    sfu_ops = ops.sfu;
    vec_ops = ops.vec;
}