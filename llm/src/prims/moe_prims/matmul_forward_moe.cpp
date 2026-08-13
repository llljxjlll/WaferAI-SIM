#include "isa/published_npu_ops.h"

#include "prims/moe_prims.h"
#include "utils/memory_utils.h"
#include "utils/prim_utils.h"
#include "utils/print_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(matmul_forward_moe, PrimId::MATMUL_FORWARD_MOE);

void matmul_forward_moe::initialize() {
    auto &p = param_value;
    data_chunk = {{"weight", p["OC"] * p["C"]}, {"bias", p["OC"]}};

    if (p["is_merge"]) {
        data_size_input = {p["B"] * p["T"] * p["C"] * p["K"]};
        data_chunk.push_back({"output", p["B"] * p["T"] * p["OC"]});
    } else {
        data_size_input = {p["B"] * p["T"] * p["C"]};
        data_chunk.push_back({"output", p["B"] * p["T"] * p["OC"] * p["K"]});
    }
}

void matmul_forward_moe::taskCore(TaskCoreContext &context, string prim_name,
                                  u_int64_t &dram_time, u_int64_t &exu_ops,
                                  u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    auto &p = param_value;
    auto &selected_experts = prim_context->selected_experts_;
    auto &selected_freq = prim_context->selected_freq_;

    if (p["E_N"] <= 0 || p["K"] < 0 || p["K"] > p["E_N"])
        throw std::invalid_argument(
            "MOE_MATMUL requires 0 <= K <= E_N and E_N > 0");

    // 判断是否需要重选专家
    if (p["need_choose"]) {
        selected_experts.clear();

        LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                        << " Selecting experts...";

        bool exp_flag[p["E_N"]];
        for (auto &b : exp_flag)
            b = false;

        for (auto e : selected_experts)
            exp_flag[e] = true;

        for (auto &e : selected_experts) {
            if (RandResult(50))
                continue; // 50%概率不重选

            exp_flag[e] = false;
            do {
                e = rand() % p["E_N"];
            } while (exp_flag[e]);
            exp_flag[e] = true;
        }

        for (int i = selected_experts.size(); i < p["K"]; i++) {
            int s_exp;
            do {
                s_exp = rand() % p["E_N"];
            } while (exp_flag[s_exp]);
            exp_flag[s_exp] = true;
            selected_experts.push_back(s_exp);
        }

        while (selected_freq.size() < static_cast<std::size_t>(p["E_N"]))
            selected_freq.push_back(0);

        for (auto e : selected_experts)
            selected_freq[e]++;

    } else {
        if (selected_experts.size() != static_cast<std::size_t>(p["K"])) {
            throw std::runtime_error(
                "MOE_MATMUL selected_experts size mismatch: " +
                std::to_string(selected_experts.size()) + " != " +
                std::to_string(p["K"]));
        }
    }

    for (auto e : selected_experts) {
        cout << "Core" << prim_context->cid <<   " selected expert: " << e << endl;
    }

    // 优先查看是否有被prefetch的专家
    std::vector<bool> checked(static_cast<std::size_t>(p["E_N"]), false);

    for (auto e : selected_experts) {
        if (e < 0 || e >= p["E_N"])
            throw std::runtime_error(
                "MOE_MATMUL selected expert is outside [0,E_N)");

        // if (std::find(prefetched_experts.begin(), prefetched_experts.end(),
        //               e) == prefetched_experts.end())
        //     continue;

        auto label_weight = ETERNAL_PREFIX + prim_name + "_w_" + to_string(e);
        checkStaticData(context, dram_time,
                        data_chunk_addr["weight"] +
                            e * GetFromPairedVector(data_chunk, "weight"),
                        GetFromPairedVector(data_chunk, "weight"),
                        label_weight);

        auto label_bias = ETERNAL_PREFIX + prim_name + "_b_" + to_string(e);
        checkStaticData(context, dram_time,
                        data_chunk_addr["bias"] +
                            e * GetFromPairedVector(data_chunk, "bias"),
                        GetFromPairedVector(data_chunk, "bias"), label_bias);

        checked[e] = true;
    }

    for (auto e : selected_experts) {
        if (checked[e])
            continue;

        auto label_weight = ETERNAL_PREFIX + prim_name + "_w_" + to_string(e);
        checkStaticData(context, dram_time,
                        data_chunk_addr["weight"] +
                            e * GetFromPairedVector(data_chunk, "weight"),
                        GetFromPairedVector(data_chunk, "weight"),
                        label_weight);

        auto label_bias = ETERNAL_PREFIX + prim_name + "_b_" + to_string(e);
        checkStaticData(context, dram_time,
                        data_chunk_addr["bias"] +
                            e * GetFromPairedVector(data_chunk, "bias"),
                        GetFromPairedVector(data_chunk, "bias"), label_bias);


        checked[e] = true;
    }

    if (SPEC_USE_PERF_GEMM) {
        ExuConfig *exu = GetCoreHWConfig(context.cid)->exu;

        uint64_t weight_tile_x = (p["C"] + exu->x_dims - 1) / exu->x_dims;
        uint64_t weight_tile_y = (p["OC"] + exu->x_dims - 1) / exu->x_dims;

        uint64_t padding_input_x = (p["T"] * p["B"] * p["K"]) > exu->x_dims
                                       ? p["T"] * p["B"] * p["K"]
                                       : exu->x_dims;

        uint64_t performance_cycle =
            (exu->x_dims + exu->x_dims + padding_input_x) * weight_tile_x *
            weight_tile_y;

        LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                        << " performance_cycle " << performance_cycle;

        int loop_input_count =
            weight_tile_y - 1; // read loop_input_count Repetitive input

        for (int loop = 0; loop < loop_input_count; loop++) {
            for (std::size_t p = 0; p < data_size_input.size(); ++p) {
                if (prim_context->datapass_label_->indata[p].find(DRAM_LABEL) ==
                    0) {

                    prefReadData(context, dram_time, data_size_input[p],
                                 prim_context->datapass_label_->indata[p]);
                }
            }
        }
    }

    const NpuOps ops = EvaluatePublishedNpuOps(
        Opcode::MOE_MATMUL, param_value,
        PublishedNpuHardwareForCore(context.cid));
    exu_ops = ops.exu;
    sfu_ops = ops.sfu;
    vec_ops = ops.vec;

    cout << "Core" << prim_context->cid << " selected experts: " << endl;
}