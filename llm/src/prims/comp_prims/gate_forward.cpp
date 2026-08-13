#include "prims/comp_prims.h"

#include "isa/published_npu_ops.h"
#include "utils/memory_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(gate_forward, PrimId::GATE_FORWARD);

void gate_forward::initialize() {
    auto &p = param_value;
    data_size_input = {p["B"] * p["T"] * p["C"]};
    data_chunk = {{"output", p["B"] * p["T"] * p["K"]}};
}

void gate_forward::taskCore(TaskCoreContext &context, string prim_name,
                           u_int64_t &dram_time, u_int64_t &exu_ops,
                           u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    const NpuOps ops = EvaluatePublishedNpuOps(Opcode::GATE, param_value);
    exu_ops = ops.exu;
    sfu_ops = ops.sfu;
    vec_ops = ops.vec;
}