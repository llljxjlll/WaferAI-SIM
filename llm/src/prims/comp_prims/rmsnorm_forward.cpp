#include "isa/published_npu_ops.h"
#include "prims/base.h"
#include "prims/comp_prims.h"
#include "utils/memory_utils.h"
#include "utils/prim_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(rmsnorm_forward, PrimId::RMSNORM_FORWARD);

void rmsnorm_forward::initialize() {
    auto &p = param_value;
    data_size_input = {p["B"] * p["T"] * p["C"]};
    data_chunk = {{"weight", p["C"]}, {"output", p["B"] * p["T"] * p["C"]}};
}

void rmsnorm_forward::taskCore(TaskCoreContext &context, string prim_name,
                               u_int64_t &dram_time, u_int64_t &exu_ops,
                               u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    // 读入weight数据
    LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                    << " read weight";

    auto label_weight = ETERNAL_PREFIX + prim_name + "_w";
    checkStaticData(context, dram_time, data_chunk_addr["weight"],
                    GetFromPairedVector(data_chunk, "weight"), label_weight);

    const NpuOps ops =
        EvaluatePublishedNpuOps(Opcode::RMSNORM, param_value);
    exu_ops = ops.exu;
    sfu_ops = ops.sfu;
    vec_ops = ops.vec;
}