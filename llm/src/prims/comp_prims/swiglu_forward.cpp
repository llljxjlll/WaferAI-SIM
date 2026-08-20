#include "prims/base.h"

#include "isa/published_npu_ops.h"
#include "prims/comp_prims.h"
#include "utils/memory_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(swiglu_forward, PrimId::SWIGLU_FORWARD);

void swiglu_forward::initialize() {
    auto &p = param_value;
    data_size_input = {2 * p["N"]};
    data_chunk = {{"output", p["N"]}};
}

void swiglu_forward::taskCore(TaskCoreContext &context, string prim_name,
                              u_int64_t &dram_time, u_int64_t &exu_ops,
                              u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    const NpuOps ops = EvaluatePublishedNpuOps(Opcode::SWIGLU, param_value);
    exu_ops = ops.exu;
    sfu_ops = ops.sfu;
    vec_ops = ops.vec;
}
