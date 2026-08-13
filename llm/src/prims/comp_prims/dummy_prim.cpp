#include "systemc.h"

#include "isa/published_npu_ops.h"

#include "memory/dram/Dcachecore.h"
#include "prims/base.h"
#include "prims/comp_prims.h"
#include "utils/memory_utils.h"

REGISTER_PRIM(Dummy_p, PrimId::DUMMY_P)

void Dummy_p::initialize() {
    data_size_input = {80};
    data_chunk = {{"output", 80}};
}

void Dummy_p::taskCore(TaskCoreContext &context, string prim_name,
                      u_int64_t &dram_time, u_int64_t &exu_ops,
                      u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    const NpuOps ops = EvaluatePublishedNpuOps(
        static_cast<Opcode>(0x15), param_value);
    exu_ops = ops.exu;
    sfu_ops = ops.sfu;
    vec_ops = ops.vec;
}