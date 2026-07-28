#include "nlohmann/json.hpp"
#include <SFML/Graphics.hpp>
#include <algorithm>
#include <sstream>
#include <set>
#include <tuple>
#include <stdexcept>
#include <string>

#include "common/system.h"
#include "defs/spec.h"
#include "die/port.h"
#include "dte/coll_plan.h"
#include "dte/coll_latency.h"
#include "dte/coll_multicast.h"
#include "dte/coll_innetwork_reduce.h"
#include "monitor/config_helper_core.h"
#include "monitor/host_envelope.h"
#include "monitor/workload_normalize.h"
#include "prims/base.h"
#include "utils/config_utils.h"
#include "utils/display_utils.h"
#include "utils/msg_utils.h"
#include "utils/prim_utils.h"
#include "utils/router_utils.h"
#include "utils/system_utils.h"

using json = nlohmann::json;

namespace {
CollOp ParseGlobalCollOp(const std::string &v) {
    if (v == "p2p") return CollOp::P2P;
    if (v == "scatter") return CollOp::SCATTER;
    if (v == "gather") return CollOp::GATHER;
    if (v == "broadcast") return CollOp::BROADCAST;
    if (v == "alltoall") return CollOp::ALLTOALL;
    if (v == "allgather") return CollOp::ALLGATHER;
    if (v == "reduce") return CollOp::REDUCE;
    if (v == "reducescatter") return CollOp::REDUCESCATTER;
    if (v == "allreduce") return CollOp::ALLREDUCE;
    throw std::invalid_argument("unsupported collective op: " + v);
}
CollDType ParseGlobalCollDType(const std::string &v) {
    if (v == "uint8") return CollDType::UINT8;
    if (v == "int32") return CollDType::INT32;
    if (v == "int64") return CollDType::INT64;
    if (v == "fp32") return CollDType::FP32;
    throw std::invalid_argument("unsupported collective dtype: " + v);
}
CollReduceOp ParseGlobalReduceOp(const std::string &v) {
    if (v == "none") return CollReduceOp::NONE;
    if (v == "sum") return CollReduceOp::SUM;
    if (v == "max") return CollReduceOp::MAX;
    throw std::invalid_argument("unsupported collective reduce_op: " + v);
}
CollAlgorithm GlobalCollAlgorithm(CollOp op) {
    if (op == CollOp::REDUCESCATTER)
        return CollAlgorithm::REDUCE_ROOT_SCATTER;
    if (op == CollOp::ALLREDUCE)
        return CollAlgorithm::REDUCE_ROOT_BROADCAST;
    return CollAlgorithm::DIRECT;
}
uint32_t StableGroupId(const std::vector<uint16_t> &group) {
    uint32_t hash = 2166136261u;
    for (uint16_t id : group) { hash ^= id; hash *= 16777619u; }
    return hash;
}
int CollectiveFlowTag(const CollDescriptor &d, uint16_t phase, uint16_t dst_rank) {
    uint32_t hash = 2166136261u;
    auto mix = [&](uint32_t v) { hash ^= v; hash *= 16777619u; };
    mix(d.key.group_id); mix(d.key.collective_id); mix(d.key.epoch);
    mix(phase); mix(d.group[dst_rank]);
    return static_cast<int>(COLL_TAG_BASE +
        (hash % (uint32_t(COLL_TAG_MAX) - COLL_TAG_BASE + 1u)));
}
uint16_t CollectiveTreeId(const CollDescriptor &d) {
    uint32_t hash = 2166136261u;
    auto mix = [&](uint32_t v) { hash ^= v; hash *= 16777619u; };
    mix(d.key.group_id); mix(d.key.collective_id); mix(d.key.epoch);
    return static_cast<uint16_t>(1u + hash % 0xffffu);
}
Directions OppositeDirection(Directions d) {
    if (d == WEST) return EAST;
    if (d == EAST) return WEST;
    if (d == NORTH) return SOUTH;
    if (d == SOUTH) return NORTH;
    throw std::invalid_argument("CENTER has no tree-link opposite");
}
Directions StepDirection(int from, int to) {
    if (to == from - 1) return WEST;
    if (to == from + 1) return EAST;
    if (to == from - GRID_X) return SOUTH;
    if (to == from + GRID_X) return NORTH;
    throw std::invalid_argument("collective tree contains a non-neighbor edge");
}
void ProgramBroadcastTree(const CollDescriptor &d) {
    const int root = d.group[d.root_rank];
    for (uint16_t core : d.group)
        if (DieOfGlobal(core) != DieOfGlobal(root))
            throw std::invalid_argument("Tier1/Tier2 collective cannot cross dies");
    std::map<int, int> parent;
    std::map<int, uint8_t> outputs;
    for (uint16_t target : d.group) {
        if (target == root) continue;
        int cur = root;
        while (cur % GRID_X != target % GRID_X) {
            int next = cur + ((target % GRID_X > cur % GRID_X) ? 1 : -1);
            auto inserted = parent.emplace(next, cur);
            if (!inserted.second && inserted.first->second != cur)
                throw std::runtime_error("XY collective tree has multiple parents");
            outputs[cur] |= 1u << StepDirection(cur, next); cur = next;
        }
        while (cur != target) {
            int next = cur + (target > cur ? GRID_X : -GRID_X);
            auto inserted = parent.emplace(next, cur);
            if (!inserted.second && inserted.first->second != cur)
                throw std::runtime_error("XY collective tree has multiple parents");
            outputs[cur] |= 1u << StepDirection(cur, next); cur = next;
        }
        outputs[target] |= 1u << CENTER;
    }
    const uint16_t tree = CollectiveTreeId(d);
    for (const auto &node : outputs) {
        Directions ingress = CENTER;
        if (node.first != root)
            ingress = OppositeDirection(StepDirection(parent.at(node.first),
                                                       node.first));
        ProgramCollectiveTreeEntry(
            {tree, static_cast<uint16_t>(node.first),
             static_cast<uint8_t>(ingress)}, node.second);
    }
    if (CollIsReduction(d.op)) {
        const std::set<uint16_t> members(d.group.begin(), d.group.end());
        for (const auto &node : outputs) {
            uint8_t expected = node.second & ~(1u << CENTER);
            if (members.count(static_cast<uint16_t>(node.first)))
                expected |= 1u << CENTER;
            Directions parent_output = CENTER;
            if (node.first != root)
                parent_output = StepDirection(node.first,
                                              parent.at(node.first));
            ProgramCollectiveReduceNode(
                tree, static_cast<uint16_t>(node.first),
                {expected, parent_output});
        }
    }
    ValidateCollectiveTree(tree, static_cast<uint16_t>(root), d.group);
}
void SetCollectiveFlowSize(Send_prim *prim, uint64_t bits) {
    if (bits == 0) throw std::invalid_argument("collective flow payload must be positive");
    const uint64_t raw = CollCeilDiv(bits, M_D_DATA);
    const uint64_t scale = static_cast<uint64_t>(HW_NOC_PAYLOAD_PER_CYCLE);
    if (scale == 0 || scale > 255) throw std::invalid_argument("invalid NoC payload scale");
    const uint64_t grouped = CollCeilDiv(raw, scale);
    if (grouped > M_D_FLOW_PACKETS_MAX) throw std::overflow_error("collective flow exceeds REQUEST wire capacity");
    prim->max_packet = static_cast<int>(grouped);
    prim->packet_scale = static_cast<int>(scale);
    prim->packets_in_last_group = static_cast<int>(raw % scale ? raw % scale : scale);
    prim->end_length = static_cast<int>(bits - (raw - 1) * M_D_DATA);
}
void AppendCollectiveActions(std::vector<PrimBase *> &prims, const CollDescriptor &d) {
    if (SPEC_NOC_COLL_TIER == 2 && CollIsReduction(d.op)) {
        const uint16_t tree = CollectiveTreeId(d);
        auto *tx = new Collective_data_prim();
        tx->descriptor = d; tx->tree_id = tree;
        tx->mode = Collective_data_prim::Mode::REDUCE_TX;
        prims.push_back(tx);
        if (d.self_rank == d.root_rank) {
            auto *rx = new Collective_data_prim();
            rx->descriptor = d; rx->tree_id = tree;
            rx->mode = Collective_data_prim::Mode::REDUCE_RX;
            prims.push_back(rx);
        }
        auto append_barrier = [&](uint16_t phase, bool release) {
            auto *barrier = new Collective_prim();
            barrier->descriptor = d; barrier->phase_id = phase;
            barrier->release_tree_id = release ? tree : 0;
            prims.push_back(barrier);
        };
        append_barrier(0, d.op == CollOp::REDUCE);
        if (d.op == CollOp::ALLREDUCE) {
            auto *data = new Collective_data_prim();
            data->descriptor = d; data->tree_id = tree;
            data->mode = d.self_rank == d.root_rank
                ? Collective_data_prim::Mode::BROADCAST_TX
                : Collective_data_prim::Mode::BROADCAST_RX;
            prims.push_back(data); append_barrier(1, true);
        } else if (d.op == CollOp::REDUCESCATTER) {
            const uint16_t n = static_cast<uint16_t>(d.group.size());
            for (uint16_t dst = 0; dst < n; ++dst) {
                if (dst == d.root_rank) continue;
                const auto part = CollRankCountOffset(d.count, n, dst);
                const uint64_t bits = part.first * CollDTypeBits(d.dtype);
                const int tag = CollectiveFlowTag(d, 1, dst);
                if (d.self_rank == d.root_rank) {
                    auto *req = new Send_prim(SEND_TYPE::SEND_REQ,
                                              d.group[dst], tag);
                    auto *ack = new Recv_prim(RECV_TYPE::RECV_ACK);
                    auto *data = new Send_prim(SEND_TYPE::SEND_DATA,
                                               d.group[dst], tag);
                    SetCollectiveFlowSize(req, bits);
                    SetCollectiveFlowSize(data, bits);
                    data->output_label = "collective_v5";
                    prims.push_back(req); prims.push_back(ack);
                    prims.push_back(data);
                } else if (d.self_rank == dst) {
                    prims.push_back(new Recv_prim(RECV_TYPE::RECV_DATA,
                                                  tag, 1));
                }
            }
            append_barrier(1, true);
        }
        return;
    }
    if (SPEC_NOC_COLL_TIER >= 1 && d.op == CollOp::BROADCAST) {
        auto *data = new Collective_data_prim();
        data->descriptor = d; data->tree_id = CollectiveTreeId(d);
        data->mode = d.self_rank == d.root_rank
            ? Collective_data_prim::Mode::BROADCAST_TX
            : Collective_data_prim::Mode::BROADCAST_RX;
        prims.push_back(data);
        auto *barrier = new Collective_prim();
        barrier->descriptor = d; barrier->phase_id = 0;
        barrier->release_tree_id = data->tree_id;
        prims.push_back(barrier);
        return;
    }
    for (const CollAction &action : PlanTier0Collective(d, d.self_rank)) {
        if (action.kind == CollActionKind::SEND) {
            const int dest = d.group[action.peer_rank];
            const int tag = CollectiveFlowTag(d, action.phase_id, action.peer_rank);
            auto *req = new Send_prim(SEND_TYPE::SEND_REQ, dest, tag);
            auto *ack = new Recv_prim(RECV_TYPE::RECV_ACK);
            auto *data = new Send_prim(SEND_TYPE::SEND_DATA, dest, tag);
            SetCollectiveFlowSize(req, action.payload_bits);
            SetCollectiveFlowSize(data, action.payload_bits);
            data->output_label = "collective_v1";
            prims.push_back(req); prims.push_back(ack); prims.push_back(data);
        } else if (action.kind == CollActionKind::RECV) {
            const int tag = CollectiveFlowTag(d, action.phase_id, d.self_rank);
            prims.push_back(new Recv_prim(RECV_TYPE::RECV_DATA, tag, 1));
            if (d.gather_reorder_depth != 0 &&
                (d.op == CollOp::GATHER || d.op == CollOp::ALLGATHER ||
                 d.op == CollOp::ALLTOALL)) {
                auto *arrival = new Collective_prim();
                arrival->descriptor = d;
                arrival->phase_id = action.phase_id;
                arrival->marker_kind =
                    Collective_prim::MarkerKind::GATHER_ARRIVAL;
                prims.push_back(arrival);
            }
            if (CollIsReduction(d.op) && action.phase_id < d.group.size()) {
                auto *arrival = new Collective_prim();
                arrival->descriptor = d;
                arrival->phase_id = action.phase_id;
                arrival->marker_kind =
                    Collective_prim::MarkerKind::REDUCE_ARRIVAL;
                prims.push_back(arrival);
            }
        } else if (action.kind == CollActionKind::REDUCE_COMPUTE) {
            auto *compute = new Reduce_compute_prim();
            compute->descriptor = d;
            prims.push_back(compute);
        } else {
            auto *barrier = new Collective_prim();
            barrier->descriptor = d;
            barrier->phase_id = action.phase_id;
            prims.push_back(barrier);
        }
    }
}
}


CoreConfig *config_helper_core::get_core(int id) {
    for (int i = 0; i < coreconfigs.size(); i++) {
        if (coreconfigs[i].id == id)
            return &(coreconfigs[i]);
    }

    LOG_ERROR(config_helper_core.cpp) << "Core " << id << " not found";
    return nullptr;
}

void config_helper_core::printSelf() {}

void config_helper_core::random_core() {
    // 注：此随机放置路径当前已停用（构造函数里已注释）。生产路径的重映射由
    // CoreConfigRemap（`g_core_remap` map，天然支持 global id）负责。这里按 V0b-2C1
    // 用 vector(TOTAL_CORES) 取代固定 GRID_SIZE 数组、tag 边界改 TOTAL_CORES，保持一致。
    std::vector<int> o2r(TOTAL_CORES, -1);
    std::vector<int> r2o(TOTAL_CORES, -1);

    std::srand(std::time(nullptr));
    for (auto config : coreconfigs) {
        int id = config.id;
        int rand = 0;
        do {
            rand = std::rand() % TOTAL_CORES;
        } while (r2o[rand] != -1);
        o2r[id] = rand;
        r2o[rand] = id;
    }

    // 改写（只遍历 active core set）
    for (auto &config : coreconfigs) {
        int oid = config.id;
        config.id = o2r[oid];
        if (config.prim_copy != -1)
            config.prim_copy = o2r[config.prim_copy];

        if (config.send_global_mem != -1)
            config.send_global_mem = o2r[config.send_global_mem];

        for (auto &work : config.worklist) {
            if (work.recv_tag < TOTAL_CORES)
                work.recv_tag = o2r[oid];
            for (auto &cast : work.cast) {
                if (cast.tag < TOTAL_CORES && cast.dest >= 0)
                    cast.tag = o2r[cast.tag];
                if (cast.dest < TOTAL_CORES && cast.dest >= 0)
                    cast.dest = o2r[cast.dest];
            }
        }

        config.printSelf();
    }

    // 注：跨 die 拒绝已由 PreflightValidateWorkload（原始 JSON，绘图/构造前）负责（V0b-2C0）。
    // CoreConfigRemap 阶段 work.cast 尚未填充，此处不再放二次 guard（会是死代码）。

    // 改写source
    for (auto &pair : source_info) {
        pair.first = o2r[pair.first];
    }

    // 重新绘图
    set<int> source_ids;
    for (auto &pair : source_info) {
        source_ids.emplace(pair.first);
    }

    unordered_map<int, Display::Core> cores;
    int core_id = 0;
    for (const auto &core_j : coreconfigs) {
        Display::Core core;
        core.id = core_j.id;
        core.x = core.id % GRID_X; // X 坐标
        core.y = core.id / GRID_X; // Y 坐标

        // 提取每个 core 的 dest 信息
        for (const auto &work : core_j.worklist) {
            vector<int> temp_cast;
            for (const auto &cast : work.cast) {
                if (cast.critical) {
                    int d = cast.dest;
                    temp_cast.push_back(1e5 + d);
                } else
                    temp_cast.push_back(cast.dest);
            }

            core.dests.push_back(temp_cast);
        }

        cores[core.id] = core;
        core_id++;
    }

    plot_dataflow(cores, source_ids);
}

config_helper_core::config_helper_core(string filename, int config_chip_id) {
    LOG_INFO(CONFIG) << "Loading config file " << filename;

    // 2C0/2C1：先解析一次 + 原始 id 校验（在绘图/构造/elaboration 之前）
    {
        ifstream pf(filename);
        if (pf.is_open()) {
            json praw;
            try {
                pf >> praw;
            } catch (const json::parse_error &) {
                praw = json(); // 解析错误留给下方原有逻辑报告
            }
            // 结构校验（bounds + 跨 die cast），只读启动期校验（2C1）。
            // 2B1：die>0 已可运行（per-die HOST attachment 就绪），移除原「die>0 不可运行」限制。
            // V1-c3：REQUEST/ACK/DATA 已接通，放行有精确双向 peer link 的相邻 die。
            ValidateWorkloadStructure(praw, config_chip_id, true);
        }
    }

    plot_dataflow(filename);
    ifstream jfile(filename);
    if (!jfile.is_open()) {
        LOG_ERROR(CONFIG) << "Could not open config file " << filename;
    }

    json j;
    try {
        jfile >> j;
    } catch (const json::parse_error &e) {
        std::cerr << "Parse error: " << e.what() << std::endl;
    }

    // 收集相关参数
    auto config_vars = j["vars"];
    for (auto var : config_vars.items()) {
        vtable.push_back(make_pair(var.key(), var.value()));
    }

    SetParamFromJson(config_vars, "B", &batch_size, 1);
    SetParamFromJson(config_vars, "T", &seq_len, 128);

    auto config_source = j["source"];
    end_count_sources = 0;
    for (auto source : config_source) {
        int source_loop = 0;
        bool is_end = false;
        SetParamFromJson(source, "loop", &source_loop, 1);
        SetParamFromJson(source, "is_end", &is_end, false);

        if (is_end)
            end_count_sources += source_loop;

        for (; source_loop > 0; source_loop--)
            source_info.push_back(
                make_pair(source["dest"], GetDefinedParam(source["size"])));
    }

    auto config_cores = j["chips"][config_chip_id]["cores"];
    for (int i = 0; i < config_cores.size(); i++) {
        // 调用 config_helper_base中的from_json
        CoreConfig core = config_cores[i]; // 这里不直接转化prims
        coreconfigs.push_back(core);
    }

    // if (j.contains("random") && j["random"]) {
    //     random_core();
    // }

    CoreConfigRemap(source_info, coreconfigs);

    SetParamFromJson(j, "pipeline", &pipeline, 1);

    // 检查是否需要复制原语的核，config书写要求：需要重新写明所有work的cast、recv_cnt,数量等同于需要复制的那个核的work数量
    for (int i = 0; i < coreconfigs.size(); i++) {
        if (coreconfigs[i].prim_copy != -1) {
            auto worklist_temp = coreconfigs[i].worklist;
            coreconfigs[i].worklist.clear();
            for (int j = 0; j < worklist_temp.size(); j++) {
                auto prev_job = worklist_temp[j];

                auto target_core = get_core(coreconfigs[i].prim_copy);
                auto target_work = target_core->worklist[j];
                for (int c = 0; c < prev_job.cast.size(); c++) {
                    if (c >= target_work.cast.size()) {
                        target_work.cast.push_back(prev_job.cast[c]);
                        continue;
                    }
                    if (target_work.cast[c].tag == target_work.cast[c].dest ||
                        prev_job.cast[c].tag != prev_job.cast[c].dest)
                        target_work.cast[c].tag = prev_job.cast[c].tag;
                    target_work.cast[c].dest = prev_job.cast[c].dest;
                    target_work.cast[c].loopout = prev_job.cast[c].loopout;
                    target_work.cast[c].stripe = prev_job.cast[c].stripe;
                }
                target_work.recv_cnt = prev_job.recv_cnt;
                target_work.recv_stripe = prev_job.recv_stripe;
                if (target_work.recv_tag == coreconfigs[i].prim_copy ||
                    prev_job.recv_tag != coreconfigs[i].id)
                    target_work.recv_tag = prev_job.recv_tag;
                coreconfigs[i].worklist.push_back(target_work);
            }
        }
    }

    if (j["chips"][config_chip_id].contains("collectives") &&
        !SPEC_NOC_COLL_ENABLED)
        throw std::invalid_argument("workload collectives require noc.collective.enabled=true");
    if (j["chips"][config_chip_id].contains("collectives")) {
        ResetCollectiveFabric();
        ResetCollectiveReduceFabric();
        for (const auto &config : coreconfigs) {
            for (const auto &work : config.worklist) {
                if (work.recv_cnt > 0 && work.recv_tag >= COLL_TAG_BASE)
                    throw std::invalid_argument("regular recv_tag overlaps collective-reserved tag range");
                for (const auto &cast : work.cast)
                    if (cast.dest >= 0 && cast.tag >= COLL_TAG_BASE)
                        throw std::invalid_argument("regular send tag overlaps collective-reserved tag range");
            }
        }
        const auto &decls = j["chips"][config_chip_id]["collectives"];
        std::set<std::tuple<uint32_t, uint32_t, uint32_t>> collective_keys;
        std::set<std::pair<int, int>> collective_flow_tags;
        std::set<uint16_t> collective_tree_ids;
        for (size_t ci = 0; ci < decls.size(); ++ci) {
            const auto &decl = decls[ci];
            CollDescriptor base;
            base.op = ParseGlobalCollOp(decl.at("op").get<std::string>());
            base.algorithm = GlobalCollAlgorithm(base.op);
            base.dtype = ParseGlobalCollDType(
                decl.value("dtype", std::string("uint8")));
            base.reduce_op = ParseGlobalReduceOp(
                decl.value("reduce_op", std::string("none")));
            base.group = decl.at("group").get<std::vector<uint16_t>>();
            base.key.group_id = decl.value("group_id", StableGroupId(base.group));
            base.key.collective_id = decl.value("collective_id", static_cast<uint32_t>(ci));
            base.key.epoch = decl.value("epoch", uint32_t(0));
            if (!collective_keys.insert({base.key.group_id, base.key.collective_id, base.key.epoch}).second)
                throw std::invalid_argument("duplicate collective instance key");
            base.count = decl.value("count", uint64_t(1));
            base.chunk_bits = decl.at("chunk_bits").get<uint64_t>();
            base.stride_bits = decl.value("stride_bits", base.chunk_bits);
            base.src_addr = decl.value("src_addr", uint64_t(0));
            base.dst_addr = decl.value("dst_addr", uint64_t(0));
            base.gather_reorder_depth = decl.value("gather_reorder_depth", uint32_t(0));
            if (base.gather_reorder_depth != 0 &&
                base.op != CollOp::GATHER && base.op != CollOp::ALLGATHER &&
                base.op != CollOp::ALLTOALL)
                throw std::invalid_argument(
                    "gather_reorder_depth is only valid for Gather RX operations");
            if (CollIsReduction(base.op))
                ValidateTier0ReductionDescriptor(base);
            const uint16_t root_core = decl.value("root", base.group.front());
            auto root_it = std::find(base.group.begin(), base.group.end(), root_core);
            if (root_it == base.group.end()) throw std::invalid_argument("collective root is not in group");
            base.root_rank = static_cast<uint16_t>(root_it - base.group.begin());
            if (SPEC_NOC_COLL_TIER >= 1) {
                const int root_die = DieOfGlobal(root_core);
                for (uint16_t core : base.group)
                    if (DieOfGlobal(core) != root_die)
                        throw std::invalid_argument(
                            "Tier1/Tier2 collective cannot cross dies");
            }
            if (SPEC_NOC_COLL_TIER >= 1 &&
                (base.op == CollOp::BROADCAST ||
                 (SPEC_NOC_COLL_TIER == 2 && CollIsReduction(base.op)))) {
                const uint16_t tree_id = CollectiveTreeId(base);
                if (!collective_tree_ids.insert(tree_id).second)
                    throw std::invalid_argument(
                        "collective tree_id hash collision");
                ProgramBroadcastTree(base);
            }
            for (uint16_t probe_rank = 0; probe_rank < base.group.size(); ++probe_rank) {
                CollDescriptor probe = base; probe.self_rank = probe_rank;
                ValidateCollDescriptor(probe);
                for (const auto &action : PlanTier0Collective(probe, probe_rank)) {
                    if (action.kind != CollActionKind::SEND) continue;
                    const int dest = base.group[action.peer_rank];
                    const int tag = CollectiveFlowTag(base, action.phase_id, action.peer_rank);
                    if (!collective_flow_tags.insert({dest, tag}).second)
                        throw std::invalid_argument("collective flow tag collision");
                }
            }
            const bool terminal = decl.value("terminal", false);
            for (uint16_t rank = 0; rank < base.group.size(); ++rank) {
                CoreConfig *core = get_core(base.group[rank]);
                if (!core) throw std::invalid_argument("collective group core is absent from workload");
                if (core->loop != 1)
                    throw std::invalid_argument("V1 collective requires core loop=1; express epochs as declarations");
                CollDescriptor local = base;
                local.self_rank = rank;
                ValidateCollDescriptor(local);
                CoreJob job;
                job.recv_cnt = 0; job.recv_tag = core->id; job.recv_stripe = 1;
                job.collectives.push_back(local);
                job.collective_terminal = terminal && rank == base.root_rank;
                core->worklist.push_back(job);
            }
        }
    }

    end_cores = 0;
    g_recv_ack_cnt = 0;
    g_recv_done_cnt = 0;

    for (int i = 0; i < coreconfigs.size(); i++)
        generate_prims(i);

    // 再去重新填写send的收发地址
    calculate_address(true);
    calculate_address(false);
}

std::vector<HostEnvelope> config_helper_core::BuildConfigMessages() {
    std::vector<HostEnvelope> envs;
    for (auto &config : coreconfigs) {
        auto build_msgs = [&](const vector<PrimBase *> &prims,
                              bool adjust_recv = false) {
            vector<Msg> msgs;
            msgs.reserve(prims.size());
            for (auto *prim : prims) {
                if (adjust_recv) {
                    if (auto *recv_prim = dynamic_cast<Recv_prim *>(prim)) {
                        if (recv_prim->type == RECV_TYPE::RECV_START)
                            recv_prim->type = RECV_TYPE::RECV_DATA;
                    }
                }

                auto segments = prim->serialize();
                for (int seg = 0; seg < segments.size(); seg++)
                    msgs.emplace_back(
                        Msg(false, MSG_TYPE::CONFIG, msgs.size() + 1, config.id,
                            seg == segments.size() - 1, segments[seg]));
            }
            return msgs;
        };

        // 三类循环消息
        vector<Msg> in_loop, next_loop, last_loop;
        for (auto &work : config.worklist) {
            // 第一个循环接受 start 数据包
            auto tmp_in = build_msgs(work.prims_in_loop);
            // 中间的循环 原本接受start数据包的接受data数据包
            auto tmp_next = build_msgs(work.prims_in_loop, true);
            // 最后一个循环，如果只有一个循环就是最后一个循环
            auto tmp_last = build_msgs(work.prims_last_loop);

            in_loop.insert(in_loop.end(), tmp_in.begin(), tmp_in.end());
            next_loop.insert(next_loop.end(), tmp_next.begin(), tmp_next.end());
            last_loop.insert(last_loop.end(), tmp_last.begin(), tmp_last.end());
        }

        // queue push helper
        int seq_cnt = 1;
        auto push_msg = [&](Msg m) {
            m.seq_id_ = seq_cnt++;
            envs.push_back(HostEnvelope{config.id, m});
        };

        // RECV_WEIGHT
        const int recv_weight_tag = config.worklist.empty()
            ? config.id : config.worklist.front().recv_tag;
        PrimBase *recv_weight = new Recv_prim(RECV_TYPE::RECV_WEIGHT,
                                              recv_weight_tag, 0);
        push_msg(Msg(false, MSG_TYPE::CONFIG, 0, config.id,
                     recv_weight->serialize()[0]));

        // Set_batch
        vector<Stage> batchInfo;
        for (int i = 0; i < batch_size; i++)
            batchInfo.emplace_back(i + 1, PREFILL, seq_len);
        PrimBase *set_batch = new Set_batch(batchInfo, pipeline);

        // 主循环，将pipeline视为一种循环
        for (int j = 0; j < pipeline; j++) {
            // 如果 默认的 loop = 1 其实 in_loop 和 next_loop 都不会执行
            // 这里的loop 不为 1 就是 decoding 的数量
            for (int i = 0; i < config.loop - 1; i++) {
                auto segments = set_batch->serialize();
                for (int seg = 0; seg < segments.size(); seg++)
                    push_msg(Msg(false, MSG_TYPE::CONFIG, 0, config.id,
                                 seg == segments.size() - 1, segments[seg]));

                auto &reps = (i == 0) ? in_loop : next_loop;
                for (auto m : reps)
                    push_msg(m);
            }
            // 默认执行最后一个循环
            auto segments = set_batch->serialize();
            for (int seg = 0; seg < segments.size(); seg++) {
                Msg m(false, MSG_TYPE::CONFIG, 0, config.id,
                      seg == segments.size() - 1, segments[seg]);
                if (last_loop.empty() && j == pipeline - 1 &&
                    seg == segments.size() - 1) {
                    m.refill_ = true;
                    m.is_end_ = true;
                }
                push_msg(m);
            }

            for (size_t k = 0; k < last_loop.size(); k++) {
                Msg m = last_loop[k];
                // 最后一个原语， 然后循环重填
                m.refill_ = m.is_end_ =
                    (k + 1 == last_loop.size() && j == pipeline - 1);
                push_msg(m);
            }
        }
    }
    return envs;
}

// 2B0：接口不变，内部走「信封 + legacy backend」（die0 西边 row，逐位不变）。
void config_helper_core::fill_queue_config(queue<Msg> *q) {
    LegacyHostEnqueue(BuildConfigMessages(), q);
}

void config_helper_core::generate_prims(int i) {
    CoreConfig *c = &coreconfigs[i];

    bool is_source = any_of(source_info.begin(), source_info.end(),
                            [&](auto &src) { return src.first == c->id; });

    auto add_recv = [&](vector<PrimBase *> &prims, bool start, int tag,
                        int cnt, int stripe) {
        auto *p = new Recv_prim(
            start ? RECV_TYPE::RECV_START : RECV_TYPE::RECV_DATA, tag, cnt);
        p->stripe_count = stripe;
        prims.push_back(p);
    };

    auto add_comps = [&](vector<PrimBase *> &prims,
                         const vector<PrimBase *> &works) {
        for (auto *prim : works) {
            if (dynamic_cast<Dte_async_prim *>(prim) != nullptr) {
                prims.push_back(prim);
                continue;
            }
            PrimBase *p = PrimFactory::getInstance().createPrim("Set_addr");
            auto label = p->prim_context->datapass_label_;
            if (prim->prim_type & PRIM_TYPE::COMP_PRIM) {
                for (int i = 0; i < MAX_SPLIT_NUM; i++)
                    label->indata[i] =
                        prim->prim_context->datapass_label_->indata[i];
                label->outdata = prim->prim_context->datapass_label_->outdata;
            }
            prims.push_back(p);
            prims.push_back(prim);
        }
    };

    auto add_sends = [&](vector<PrimBase *> &prims, const vector<Cast> &casts,
                         bool loopout) {
        for (auto &ca : casts) {
            // loopout pipeline的最后一个核是发给 0 核还是发给host
            if ((loopout && ca.loopout == FALSE) ||
                (!loopout && ca.loopout == TRUE))
                continue;
            auto *req = new Send_prim(SEND_TYPE::SEND_REQ, ca.dest, ca.tag);
            auto *ack = new Recv_prim(RECV_TYPE::RECV_ACK);
            auto *data = new Send_prim(SEND_TYPE::SEND_DATA, ca.dest, ca.tag);
            req->stripe_count = ca.stripe;
            ack->stripe_count = ca.stripe;
            data->stripe_count = ca.stripe;
            prims.push_back(req);
            prims.push_back(ack);
            prims.push_back(data);
        }
    };

    for (int w = 0; w < c->worklist.size(); w++) {
        auto &work = c->worklist[w];
        if (!work.collectives.empty()) {
            for (const auto &coll : work.collectives) {
                AppendCollectiveActions(work.prims_in_loop, coll);
                AppendCollectiveActions(work.prims_last_loop, coll);
            }
            if (work.collective_terminal) {
                work.prims_last_loop.push_back(new Send_prim(SEND_TYPE::SEND_DONE));
                ++end_cores;
            }
            continue;
        }
        bool is_end = judge_is_end_work(work);
        if (is_end)
            end_cores++;

        // 非最后循环
        // loop = 1 的 时候只有last_loop 循环
        add_recv(work.prims_in_loop, (is_source && w == 0), work.recv_tag,
                 work.recv_cnt, work.recv_stripe);
        add_comps(work.prims_in_loop, work.prims);
        add_sends(work.prims_in_loop, work.cast, false);

        // 最后循环
        add_recv(work.prims_last_loop, (is_source && w == 0 && c->loop == 1),
                 work.recv_tag, work.recv_cnt, work.recv_stripe);
        add_comps(work.prims_last_loop, work.prims);

        if (is_end) {
            work.prims_last_loop.push_back(new Send_prim(SEND_TYPE::SEND_DONE));
            // work.prims_last_loop.push_back(
            //     PrimFactory::getInstance().createPrim("Clear_sram"));
            continue;
        }

        // 现在不会有不是is_end 的核心有 loopout

        add_sends(work.prims_last_loop, work.cast, true);

        // if (w == c->worklist.size() - 1) {
        //     work.prims_last_loop.push_back(
        //         PrimFactory::getInstance().createPrim("Clear_sram"));
        // }
    }
}

void config_helper_core::calculate_address(bool do_loop) {
    // 自动设置 send 和 receive 的地址
    for (int i = 0; i < coreconfigs.size(); i++) {
        for (auto &work : coreconfigs[i].worklist) {
            if (!work.collectives.empty()) continue;
            // 遍历每一个核中的send原语
            vector<PrimBase *> *v = nullptr;
            if (do_loop)
                v = &(work.prims_in_loop);
            else
                v = &(work.prims_last_loop);

            int output_size = 0;
            int output_offset = 0;
            int index = 0;
            string output_label = "";
            Send_prim *pending_req = nullptr;

            if (!do_loop && judge_is_end_work(work))
                continue; // 汇节点

            // 拿到这个corejob的output size
            for (int j = v->size() - 1; j >= 0; j--) {
                auto p = (*v)[j];

                if (p->prim_type & PRIM_TYPE::COMP_PRIM) {
                    CompBase *cp = (CompBase *)p;
                    output_size = cp->out_size;
                    // output_offset = cp->out_offset;
                    output_label = cp->prim_context->datapass_label_->outdata;
                    break;
                }
            }

            vector<string> output_label_split;
            stringstream ss(output_label);
            string word;

            while (ss >> word)
                output_label_split.push_back(word);

            for (auto &prim : (*v)) {
                if (typeid(*prim) == typeid(Send_prim)) {
                    Send_prim *temp = (Send_prim *)prim;
                    if (temp->type == SEND_REQ) {
                        if (pending_req)
                            throw std::runtime_error(
                                "dataflow SEND_REQ has no intervening SEND_DATA");
                        pending_req = temp;
                        continue;
                    }
                    if (temp->type != SEND_DATA)
                        continue;

                    CalculatePacketNum(output_size, work.cast[index].weight,
                                       (prim->datatype ? 2 : 1),
                                       temp->max_packet, temp->end_length,
                                       temp->packet_scale,
                                       temp->packets_in_last_group);
                    if (!pending_req || pending_req->des_id != temp->des_id ||
                        pending_req->tag_id != temp->tag_id)
                        throw std::runtime_error(
                            "SEND_REQ/SEND_DATA pair mismatch while assigning flow_packets");
                    if (pending_req->stripe_count != temp->stripe_count)
                        throw std::runtime_error(
                            "SEND_REQ/SEND_DATA stripe mismatch while assigning flow_packets");
                    if (temp->max_packet <= 0 ||
                        (unsigned)temp->max_packet > M_D_FLOW_PACKETS_MAX)
                        throw std::runtime_error(
                            "DATA flow packet count is not encodable in REQUEST flow_packets");
                    // Send_prim wire 上 max_packet 对 SEND_REQ 是 tagged union：把后续 DATA 的 F
                    // 带到源核，源核再写入 REQUEST Msg.flow_packets_。
                    pending_req->max_packet = temp->max_packet;
                    pending_req->end_length = temp->end_length;
                    pending_req->packet_scale = temp->packet_scale;
                    pending_req->packets_in_last_group =
                        temp->packets_in_last_group;
                    if (g_d2d_cfg.mode == MODE_BOUNDED_SAF &&
                        coreconfigs[i].id / CORES_PER_DIE !=
                            temp->des_id / CORES_PER_DIE) {
                        int max_subflow_packets =
                            (temp->max_packet + temp->stripe_count - 1) /
                            temp->stripe_count;
                        if (max_subflow_packets > g_d2d_cfg.saf_buffer_depth) {
                            const char *size_name = temp->stripe_count == 1
                                                        ? "flow_packets"
                                                        : "max subflow packets";
                            throw std::runtime_error(
                                "whole-flow SAF preflight: " +
                                std::string(size_name) + " (" +
                                std::to_string(max_subflow_packets) +
                                ") exceeds saf_buffer_depth (" +
                                std::to_string(g_d2d_cfg.saf_buffer_depth) +
                                "); reject before DATA injection");
                        }
                    }
                    pending_req = nullptr;

                    temp->output_label = output_label_split.size() == 1
                                             ? output_label_split[0]
                                             : output_label_split[index];
                    index++;
                }
            }
            if (pending_req)
                throw std::runtime_error(
                    "dataflow SEND_REQ has no following SEND_DATA");
        }
    }
}

std::vector<HostEnvelope> config_helper_core::BuildStartMessages() {
    std::vector<HostEnvelope> envs;
    LOG_INFO(NETWORK) << "Config helper start START data distribution";

    for (int pipe = 0; pipe < pipeline; pipe++) {
        for (auto source : source_info) {
            // 从这里看pipeline 和 source_loop 的功能是一样的
            // start 数据包一次性都下发完成 但是可以分阶段使用
            // source loop 的循环是靠prim refill 实现的
            int i = source.first;
            int size = source.second;

            int send_offset = 0;
            for (auto config : coreconfigs) {
                if (config.id == i)
                    send_offset =
                        ((NpuBase *)config.worklist[0].prims[0])->inp_offset;
            }

            int send_size_in_bit = size * sizeof(float) * 8;
            int pkg_num = (send_size_in_bit % M_D_DATA)
                              ? (send_size_in_bit / M_D_DATA + 1)
                              : (send_size_in_bit / M_D_DATA);
            pkg_num = pkg_num % HW_NOC_PAYLOAD_PER_CYCLE
                          ? pkg_num / HW_NOC_PAYLOAD_PER_CYCLE + 1
                          : pkg_num / HW_NOC_PAYLOAD_PER_CYCLE;

            if (SPEC_USE_BEHA_NOC) {
                sc_bv<M_D_DATA> d(0x1);
                int length = M_D_DATA;
                Msg m = Msg(true, MSG_TYPE::S_DATA, 1, i, send_offset, i,
                            length, d);
                m.source_ = HOST_ENDPOINT_ID;
                m.roofline_packets_ = pkg_num;
                envs.push_back(HostEnvelope{i, m});
            } else {
                for (int j = 1; j <= pkg_num; j++) {
                    sc_bv<M_D_DATA> d(0x1);
                    int length = M_D_DATA;
                    bool is_end_packet = j == pkg_num;
                    if (is_end_packet)
                        length =
                            size * sizeof(float) - M_D_DATA * (pkg_num - 1);

                    Msg m = Msg(j == pkg_num, MSG_TYPE::S_DATA, j, i,
                                send_offset + M_D_DATA * (j - 1), i, length, d);
                    m.source_ = HOST_ENDPOINT_ID;
                    m.roofline_packets_ = 1;
                    envs.push_back(HostEnvelope{i, m});
                }
            }
        }
    }
    return envs;
}

// 2B0：接口不变，内部走「信封 + legacy backend」。
void config_helper_core::fill_queue_start(queue<Msg> *q) {
    LegacyHostEnqueue(BuildStartMessages(), q);
}

void config_helper_core::parse_ack_msg(Event_engine *event_engine, int flow_id,
                                       sc_event *notify_event) {
    event_engine->add_event(this->name(), "Waiting Recv Ack", "B",
                            Trace_event_util());

    for (auto m : g_temp_ack_msg) {
        int cid = m.source_;
        LOG_DEBUG(NETWORK) << "Config helper <- ACK <- " << cid << ", total "
                           << g_recv_ack_cnt + 1 << "/" << coreconfigs.size();

        g_recv_ack_cnt++;
    }

    g_temp_ack_msg.clear();
    event_engine->add_event(this->name(), "Waiting Recv Ack", "E",
                            Trace_event_util());


    if (g_recv_ack_cnt >= coreconfigs.size()) {
        notify_event->notify(CYCLE, SC_NS);
        g_recv_ack_cnt = 0;

        // 使用唯一的flow ID替换名称
        std::string flow_name = "flow_" + std::to_string(flow_id);
        event_engine->add_event(this->name(), "Waiting Recv Ack", "f",
                                Trace_event_util(flow_name), sc_time(0, SC_NS),
                                100, "e");
        LOG_INFO(NETWORK) << "Config helper received all ACK";
    }
}

void config_helper_core::parse_done_msg(Event_engine *event_engine,
                                        sc_event *notify_event) {
    notify_event = nullptr; // 无需触发任何信号
    event_engine->add_event(this->name(), "Waiting Core busy", "B",
                            Trace_event_util());

    for (auto m : g_temp_done_msg) {
        int cid = m.source_;
        LOG_DEBUG(NETWORK) << "Config helper <- DONE <- " << cid << ", total "
                           << g_recv_done_cnt + 1 << " / "
                           << end_cores * pipeline * max(1, end_count_sources);

        g_recv_done_cnt++;
        // g_done_msg.push_back(m);
    }
    g_temp_done_msg.clear();
    event_engine->add_event(this->name(), "Waiting Core busy", "E",
                            Trace_event_util());

    if (g_recv_done_cnt >= end_cores * pipeline * max(1, end_count_sources)) {
        LOG_INFO(SYSTEM) << "All requests finished";
        LOG_INFO(SYSTEM) << "  end_cores: " << end_cores
                         << ", total pipe: " << pipeline
                         << ", end_count_sources: " << end_count_sources;

        g_recv_done_cnt = 0;
        LOG_INFO(CATCH_TEST) << "Catch test finished";
        sc_stop();
    }
}
