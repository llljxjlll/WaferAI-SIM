#include "assert.h"
#include "defs/global.h"
#include "defs/spec.h"
#include "die/d2d_link.h"
#include "die/port.h"
#include "monitor/monitor.h"
#include "router/router.h"
#include "systemc.h"
#include "trace/Event_engine.h"
#include "utils/print_utils.h"
#include "utils/simple_flags.h"
#include "utils/system_utils.h"
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>

// 假设 json.hpp 文件在当前目录或包含路径中
#include <nlohmann/json.hpp>
#include <string>

#include <SFML/Graphics.hpp>
using namespace std;

Define_bool_opt("--help", g_flag_help, false, "show these help information");

Define_string_opt("--workload-config", g_flag_workload_config,
                  "../llm/test/workload_config/gpu/pd_serving.json",
                  "workload config file");
Define_string_opt("--hardware-config", g_flag_hardware_config,
                  "../llm/test/hardware_config/core_4x4.json",
                  "hardware config file");
Define_string_opt("--simulation-config", g_flag_simulation_config,
                  "../llm/test/simulation_config/default_spec.json",
                  "simulation config file");
Define_string_opt("--mapping-config", g_flag_mapping_config,
                  "../llm/test/mapping_config/default_mapping.txt",
                  "mapping config file");

Define_int64_opt("--trace-window", g_flag_trace_window, 2, "Trace window size");

Define_bool_opt("--d2d-v0-selftest", g_flag_d2d_v0_selftest, false,
                "run D2D V0 pure-function self-test and exit");

Define_bool_opt("--d2d-link-selftest", g_flag_d2d_link_selftest, false,
                "run D2D V1 SystemC link self-test (drives packets) and exit");

int sc_main(int argc, char *argv[]) {
    clock_t start = clock();

    srand((unsigned)time(NULL));
    std::cout.setf(std::ios::unitbuf);

    // 解析参数
    simple_flags::parse_args(argc, argv);
    if (!simple_flags::get_unknown_flags().empty()) {
        string content;
        for (auto it : simple_flags::get_unknown_flags()) {
            content += "'" + it + "', ";
        }
        content.resize(content.size() - 2); // remove last ', '
        content.append(".");
        LOG_ERROR(CONFIG) << "Unknown option(s): " << content;
        return -1;
    }

    if (g_flag_help) {
        simple_flags::print_args_info();
        return 0;
    }

    // D2D V0 L0 自测：纯函数（编址/端点/矩形拓扑/端口校验），不建仿真
    if (g_flag_d2d_v0_selftest) {
        int fails = RunD2DV0SelfTest();
        return fails == 0 ? 0 : 1;
    }
    // D2D V1 link 自测：SystemC testbench 驱动真实包穿过 D2DLinkUnit
    if (g_flag_d2d_link_selftest) {
        int fails = RunD2DLinkSelfTest();
        return fails == 0 ? 0 : 1;
    }

    // 清理所有上一次运行后产生的log文件
    DeleteCoreLogFiles();
    DeleteMemoryLogFiles();

    // 收集所有配置文件，统一解析
    InitGrid(g_flag_workload_config, g_flag_hardware_config,
             g_flag_simulation_config, g_flag_mapping_config);
    InitGlobalMembers();
    InitializeMemorySpec();

    // init_dram_areas();
    // initialize_cache_structures();

    Event_engine *event_engine =
        new Event_engine("event-engine", g_flag_trace_window);
    Monitor monitor("monitor", event_engine, g_flag_workload_config.c_str());
    sc_trace_file *tf = sc_create_vcd_trace_file("Cchip_1");
    sc_clock clk("clk", CYCLE, SC_NS);

    sc_start();

    // 运行结束后 dump D2D 端口/链路统计（V0b-2A：无 C2C 端口时恒为 0，供 runner 断言）
    {
        long in = 0, out = 0, busy = 0, stall = 0;
        for (auto &p : g_die_ports.ports) {
            in += p.stats.in_pkts;
            out += p.stats.out_pkts;
            busy += p.stats.busy_cycles;
            stall += p.stats.stall_cycles;
        }
        // V1-b：D2D link 单元实际穿越的包数（idle 时为 0）
        in += g_d2d_link_in_pkts;
        out += g_d2d_link_out_pkts;
        LOG_INFO(SYSTEM) << "[D2D] in_pkts=" << in << " out_pkts=" << out
                         << " busy_cycles=" << busy << " stall_cycles=" << stall;
        // c3 端到端证据：分别证明握手双向控制包与 DATA 都实际穿越 Link，且交付数守恒。
        LOG_INFO(SYSTEM)
            << "[D2D_TYPE] request_in=" << g_d2d_link_in_by_type[REQUEST]
            << " request_out=" << g_d2d_link_out_by_type[REQUEST]
            << " ack_in=" << g_d2d_link_in_by_type[ACK]
            << " ack_out=" << g_d2d_link_out_by_type[ACK]
            << " data_in=" << g_d2d_link_in_by_type[DATA]
            << " data_out=" << g_d2d_link_out_by_type[DATA];
    }

    // 结束态 drain 不变量（V1 验收）：遍历 SystemC 层级，累加所有 RouterUnit 的残留
    // （未释放的 in/out lock ref + 各方向 data/ctrl buffer + host buffer）。仿真正常结束时
    // 应为 0——非 0 说明有锁泄漏或滞留包（如尾包丢失 / 别名导致的 ref 未归零）。
    {
        std::function<long(const std::vector<sc_object *> &)> resid =
            [&](const std::vector<sc_object *> &objs) -> long {
            long r = 0;
            for (auto *o : objs) {
                if (auto *ru = dynamic_cast<RouterUnit *>(o))
                    r += ru->residual();
                r += resid(o->get_child_objects());
            }
            return r;
        };
        LOG_INFO(SYSTEM) << "[DRAIN] router_residual="
                         << resid(sc_get_top_level_objects());
    }

    // output_lock_ref 峰值：>=2 证明同 tag 多流共享同一把锁（多发一聚合，tag-only 核心语义）。
    LOG_INFO(SYSTEM) << "[LOCK] max_output_ref=" << g_max_output_lock_ref;

    // HOST lane 接收统计（V1-pre 3b-2b）：每 lane DONE/ACK 数 + 错配数
    // （消息应到 HostLaneOfCore(source_)；mismatch>0 说明路由送错 lane）。
    {
        long done_tot = 0, ack_tot = 0;
        std::string per_lane;
        for (size_t i = 0; i < g_host_lane_done.size(); i++) {
            done_tot += g_host_lane_done[i];
            ack_tot += g_host_lane_ack[i];
            per_lane += (i ? "," : "") + std::to_string(g_host_lane_done[i]);
        }
        LOG_INFO(SYSTEM) << "[HOSTLANE] done_total=" << done_tot
                         << " ack_total=" << ack_tot
                         << " mismatch=" << g_host_lane_mismatch
                         << " per_lane_done=" << per_lane;

        // 多重集签名（src:count）——严格证明无丢包/重复需比对多重集而非总数
        std::string dsig, asig;
        for (auto &kv : g_host_done_src) // source:count
            dsig += std::to_string(kv.first) + ":" + std::to_string(kv.second) + ",";
        for (auto &kv : g_host_ack_sig) // source:tag:count
            asig += std::to_string(kv.first.first) + ":" +
                    std::to_string(kv.first.second) + ":" +
                    std::to_string(kv.second) + ",";
        LOG_INFO(SYSTEM) << "[HOSTSIG] done=" << dsig << " ack=" << asig;
    }

    // destroy_dram_areas();
    // destroy_cache_structures();
    // event_engine->dump_traced_file();
    sc_close_vcd_trace_file(tf);

    SystemCleanup();
    CloseLogFiles();

    clock_t end = clock();

    if (correct_exit) {
        LOG_INFO(SYSTEM) << "Total Real-time Cost: "
                         << (double)(end - start) / CLOCKS_PER_SEC << "s";
    } else {
        LOG_WARN(SYSTEM) << "Simulation terminated abnormally";
    }

    ofstream outfile("simulation_result_df_pd.txt", ios::app);
    if (outfile.is_open()) {
        outfile << "Total Real-time Cost: "
                << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
        outfile.close();
    } else {
        LOG_ERROR(SYSTEM) << "Unable to open file for writing timestamp";
    }
    delete event_engine;
    return 0;
}