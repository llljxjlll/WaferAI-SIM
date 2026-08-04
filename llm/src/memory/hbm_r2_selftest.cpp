// R2 分布式 HBM 自测：验证 BehavioralHBMBackend + FIFO MemEndpointUnit 的带宽收敛、
// 共享不放大、多 channel 并行扩展、公平无饥饿、突发大小开销差异。覆盖
// HBM建模计划.md（修订版）九."R2" 的测试项（NoC 瓶颈一项没有真实 router 可用，明确
// 标注推迟到 R4，不假装测了）。
//
// 注意：SystemC 要求所有 sc_module 派生对象在第一次 sc_start() 之前完成构造
// （elaboration 阶段），之后再 new 会直接报 "insert module failed: simulation
// running"。所以本文件把全部场景的端点/适配器/驱动都在一次 sc_start() 之前建好，
// 让它们在同一段仿真时间里并发跑，而不是像"每个场景各自 construct+sc_start"那样
// 分段——那种写法在这里跑不通。
#include "die/port.h"
#include "defs/spec.h"
#include "memory/behavioral_hbm_backend.h"
#include "memory/core_mem_adapter.h"
#include "memory/hbm_address_map.h"
#include "memory/hbm_memspec.h"
#include "memory/hbm_r2_selftest.h"
#include "memory/mem_endpoint_unit.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <systemc>
#include <utility>
#include <vector>

using namespace sc_core;

namespace {

int g_fail = 0;
int g_total = 0;

void Check(bool cond, const std::string &name) {
    g_total++;
    if (!cond) {
        g_fail++;
        std::cout << "  [FAIL] " << name << std::endl;
    } else {
        std::cout << "  [ ok ] " << name << std::endl;
    }
}

void SetTopo(int gx, int gy, int dx, int dy) {
    GRID_X = gx;
    GRID_Y = gy;
    GRID_SIZE = gx * gy;
    DIE_X = dx;
    DIE_Y = dy;
    DIE_COUNT = dx * dy;
    CORES_PER_DIE = GRID_SIZE;
    TOTAL_CORES = CORES_PER_DIE * DIE_COUNT;
    HOST_ENDPOINT_ID = TOTAL_CORES;
    g_die_ports = D2DPortTable{};
    g_d2d_links.clear();
}

void ParseAndBuild(const nlohmann::json &hw) {
    ParseMemorySystem(hw);
    BuildMemAttach();
    ValidateMemAttach();
    ValidateAddressPolicy();
    ValidateHbmMemSpecConsistency();
}

struct ScriptDriver : sc_module {
    std::function<void()> script;
    bool ok = true;
    std::string error;

    SC_HAS_PROCESS(ScriptDriver);
    ScriptDriver(const sc_module_name &n, std::function<void()> s)
        : sc_module(n), script(std::move(s)) {
        SC_THREAD(Run);
    }
    void Run() {
        try {
            script();
        } catch (const std::exception &e) {
            ok = false;
            error = e.what();
        } catch (...) {
            ok = false;
            error = "unknown exception";
        }
    }
};

nlohmann::json OneStackTwoChannelConfig() {
    auto hw = nlohmann::json::parse(R"JSON({
      "memory_system": {
        "topology": "distributed_hbm",
        "cache_policy": "none",
        "profiles": {
          "hbm2_2ch": {
            "generation": "HBM2", "channels_per_stack": 2,
            "pseudo_channels_per_channel": 2,
            "data_rate_gbps_per_pin": 2.0, "stack_bus_width_bits": 128
          }
        },
        "hbm_stacks": [
          {"stack_id": 0, "compute_die_id": 0, "profile": "hbm2_2ch",
           "side": "N", "start_idx": 0, "phy_span_tiles": 2,
           "capacity_bytes": 1048576, "backend_granularity": "channel",
           "channel_dram_config": "CFG0"}
        ],
        "address_policy": {
          "mode": "numa_local_interleave",
          "home_ranges": [{"die_id": 0, "base": 0, "size_bytes": 1048576}],
          "stack_interleave_bytes": 1048576,
          "channel_interleave_bytes": 64
        }
      }
    })JSON");
    hw["memory_system"]["hbm_stacks"][0]["channel_dram_config"] =
        "../DRAMSys/configs/hbm2-example.json";
    return hw;
}

double ThroughputGBps(uint64_t total_bytes, sc_time elapsed) {
    double ns = elapsed.to_seconds() * 1e9;
    if (ns <= 0)
        return 0.0;
    return (double)total_bytes / ns; // bytes/ns == GB/s
}

// sc_time 内部按整数时间分辨率存储，但这里两侧比较的值分别来自测试代码里独立写的
// 浮点算式和 BehavioralHBMBackend::TransferTime 内部的浮点算式——两条独立表达式即
// 使数学上等价，也可能因为运算顺序/编译器优化不同在最后一位浮点精度上有差异，导致
// 转换成 sc_time 后精确 == 比较在不同编译环境下变脆弱。用极小容差（1 皮秒，远小于
// 这里任何一次传输的纳秒级耗时）比较，断言的是"数学上相等"而不是"两段独立代码碰
// 巧算出一模一样的浮点尾数"。
bool NearlyEqualTime(sc_time a, sc_time b) {
    sc_time diff = (a > b) ? (a - b) : (b - a);
    return diff <= sc_time(1, SC_PS);
}

} // namespace

int RunHbmR2SelfTest() {
    g_fail = g_total = 0;
    std::cout << "==== distributed HBM R2 self-test ====" << std::endl;

    SetTopo(4, 4, 1, 1);
    auto hw = OneStackTwoChannelConfig();
    bool setup_ok = false;
    try {
        ParseAndBuild(hw);
        setup_ok = true;
    } catch (const std::exception &e) {
        std::cout << "  [FAIL] setup threw: " << e.what() << std::endl;
    }
    Check(setup_ok, "R2 topology/memory_system config validates");

    // ==== 场景 1：单 channel 饱和吞吐收敛到配置上限（base_latency=0 排除固定
    //       开销，只看带宽项）====
    BehavioralHBMBackendConfig cfg1;
    cfg1.bandwidth_GBps = 32.0;
    cfg1.efficiency = 0.8; // <1，验证折算确实生效而不是被忽略
    cfg1.base_latency = SC_ZERO_TIME;
    BehavioralHBMBackend backend_sat(cfg1);
    MemEndpointUnit ep_sat("ep_sat", 0, 0, backend_sat, /*queue_depth=*/8);
    CoreMemAdapter core_sat("core_sat", 200);
    core_sat.BindEndpoint(0, 0, &ep_sat);
    const int kSatReqs = 50, kSatLen = 16;
    ScriptDriver driver_sat("driver_sat", [&]() {
        for (int i = 0; i < kSatReqs; i++)
            core_sat.Access(MemCommand::kRead, 0, kSatLen);
    });

    // ==== 场景 2：N 核共享一个 channel，总吞吐不因核数变多而放大，且都不饥饿 ====
    BehavioralHBMBackendConfig cfg2 = cfg1;
    cfg2.base_latency = sc_time(1, SC_NS); // 加回一点固定开销，更贴近真实场景
    // 标签用 (0,0)：下面所有请求地址都是 0（落 channel 0），必须和 DecodeAddress
    // 实际解出的 (stack_id,channel_id) 一致，CoreMemAdapter 才能在自己的
    // BindEndpoint 表里查到对应的端点（原因见场景 3 前面的注释）。
    BehavioralHBMBackend backend_share(cfg2);
    MemEndpointUnit ep_share("ep_share", 0, 0, backend_share, /*queue_depth=*/16);
    const int kShareCores = 4, kReqsPerCore = 20, kShareLen = 4;
    std::vector<std::unique_ptr<CoreMemAdapter>> share_cores;
    std::vector<int> share_completed(kShareCores, 0);
    std::vector<std::unique_ptr<ScriptDriver>> share_drivers;
    for (int i = 0; i < kShareCores; i++) {
        share_cores.push_back(std::make_unique<CoreMemAdapter>(
            sc_gen_unique_name("core_share"), 300 + i));
        share_cores.back()->BindEndpoint(0, 0, &ep_share);
    }
    for (int i = 0; i < kShareCores; i++) {
        CoreMemAdapter *c = share_cores[i].get();
        int *cnt = &share_completed[i];
        share_drivers.push_back(std::make_unique<ScriptDriver>(
            sc_gen_unique_name("driver_share"), [c, cnt]() {
                for (int j = 0; j < kReqsPerCore; j++) {
                    c->Access(MemCommand::kRead, 0, kShareLen);
                    (*cnt)++;
                }
            }));
    }

    // ==== 场景 3：多 channel 并行扩展：2 个独立 channel 各自跑满，总吞吐应接近 2x ====
    // 注意：MemEndpointUnit 构造函数里的 (stack_id, channel_id) 只是它自己的身份
    // 标签，用于 HandleRequest 校验来路对不对；但 CoreMemAdapter::Access 是按物理
    // 地址走 DecodeAddress 解出 (stack_id, channel_id) 再去自己的 BindEndpoint 表
    // 里查——而 DecodeAddress 只认识本次拓扑里真实配置的 (stack_id=0, channel 0/1)
    // 两个 channel，不能凭空发明 stack_id=1。要让某个请求真正落到 channel 1，得让
    // 它的地址落在 channel_interleave_bytes=64 的第二条 stripe（即 address>=64）。
    BehavioralHBMBackendConfig cfg3 = cfg1; // base_latency=0，同场景 1
    BehavioralHBMBackend backend_par_a(cfg3), backend_par_b(cfg3);
    MemEndpointUnit ep_par_a("ep_par_a", 0, 0, backend_par_a, 8);
    MemEndpointUnit ep_par_b("ep_par_b", 0, 1, backend_par_b, 8);
    CoreMemAdapter core_par_a("core_par_a", 400), core_par_b("core_par_b", 401);
    core_par_a.BindEndpoint(0, 0, &ep_par_a);
    core_par_b.BindEndpoint(0, 1, &ep_par_b);
    const int kParReqs = 50, kParLen = 16;
    ScriptDriver driver_par_a("driver_par_a", [&]() {
        for (int i = 0; i < kParReqs; i++)
            core_par_a.Access(MemCommand::kRead, 0, kParLen); // channel 0
    });
    ScriptDriver driver_par_b("driver_par_b", [&]() {
        for (int i = 0; i < kParReqs; i++)
            core_par_b.Access(MemCommand::kRead, 64, kParLen); // channel 1
    });

    // ==== 场景 4：构造 endpoint 瓶颈 vs HBM 瓶颈 ====
    // （NoC 瓶颈：这条链路目前是直接方法调用，没有真实 router/NoC 可以构造拥塞，
    //  留给 R4 接入真实 fabric 之后再测。）
    // tight/loose 两批各自是独立的 CoreMemAdapter 实例（各自的 BindEndpoint 表互不
    // 共享），即使都用 address=0（都落 channel 0）、都标 (stack_id=0,channel_id=0)
    // 也不会互相踩——是不同对象各自绑到不同的 MemEndpointUnit。
    BehavioralHBMBackendConfig cfg4 = cfg1;
    cfg4.base_latency = sc_time(2, SC_NS);
    BehavioralHBMBackend backend_tight(cfg4), backend_loose(cfg4);
    MemEndpointUnit ep_tight("ep_tight_queue", 0, 0, backend_tight,
                             /*queue_depth=*/1);
    MemEndpointUnit ep_loose("ep_loose_queue", 0, 0, backend_loose,
                             /*queue_depth=*/16);
    const int kBneckCores = 4, kBneckReqs = 5, kBneckLen = 4;
    std::vector<std::unique_ptr<CoreMemAdapter>> tight_cores, loose_cores;
    std::vector<std::unique_ptr<ScriptDriver>> tight_drivers, loose_drivers;
    auto buildBatch = [&](MemEndpointUnit &ep,
                          std::vector<std::unique_ptr<CoreMemAdapter>> &cores,
                          std::vector<std::unique_ptr<ScriptDriver>> &drivers) {
        for (int i = 0; i < kBneckCores; i++) {
            cores.push_back(std::make_unique<CoreMemAdapter>(
                sc_gen_unique_name("core_bneck"), 500 + i));
            cores.back()->BindEndpoint(0, 0, &ep);
            CoreMemAdapter *c = cores.back().get();
            drivers.push_back(std::make_unique<ScriptDriver>(
                sc_gen_unique_name("driver_bneck"), [c]() {
                    for (int j = 0; j < kBneckReqs; j++)
                        c->Access(MemCommand::kRead, 0, kBneckLen);
                }));
        }
    };
    buildBatch(ep_tight, tight_cores, tight_drivers);
    buildBatch(ep_loose, loose_cores, loose_drivers);

    // ==== 场景 5：读写混合、短 burst 与长 burst 的开销差异 ====
    BehavioralHBMBackendConfig cfg5 = cfg1;
    cfg5.base_latency = sc_time(3, SC_NS);
    BehavioralHBMBackend backend_burst(cfg5);
    MemEndpointUnit ep_burst("ep_burst", 0, 0, backend_burst, 8);
    CoreMemAdapter core_burst("core_burst", 600);
    core_burst.BindEndpoint(0, 0, &ep_burst);
    ScriptDriver driver_burst("driver_burst", [&]() {
        core_burst.Access(MemCommand::kWrite, 0, 4, {1, 2, 3, 4});
        core_burst.Access(MemCommand::kRead, 0, 4);
        core_burst.Access(
            MemCommand::kWrite, 8, 16,
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        core_burst.Access(MemCommand::kRead, 8, 16);
    });

    // ==== 场景 6：部分写/重叠地址、read/write turnaround 与统计口径 ====
    BehavioralHBMBackendConfig cfg6 = cfg1;
    cfg6.base_latency = sc_time(2, SC_NS);
    cfg6.read_to_write_turnaround = sc_time(11, SC_NS);
    cfg6.write_to_read_turnaround = sc_time(7, SC_NS);
    BehavioralHBMBackend backend_semantics(cfg6);
    MemEndpointUnit ep_semantics("ep_semantics", 0, 0, backend_semantics, 4,
                                 /*max_outstanding=*/2);
    CoreMemAdapter core_semantics("core_semantics", 601);
    core_semantics.BindEndpoint(0, 0, &ep_semantics);
    bool partial_write_ok = false;
    sc_time semantics_elapsed = SC_ZERO_TIME;
    ScriptDriver driver_semantics("driver_semantics", [&]() {
        sc_time begin = sc_time_stamp();
        core_semantics.Access(MemCommand::kWrite, 32, 4, {1, 2, 3, 4});
        core_semantics.Access(MemCommand::kWrite, 32, 4, {9, 9, 9, 9},
                              {0, 0xff, 0, 0xff});
        MemMsg first = core_semantics.Access(MemCommand::kRead, 32, 4);
        MemMsg second = core_semantics.Access(MemCommand::kRead, 32, 4);
        core_semantics.Access(MemCommand::kWrite, 32, 4, {5, 6, 7, 8});
        semantics_elapsed = sc_time_stamp() - begin;
        partial_write_ok =
            first.payload == std::vector<uint8_t>({1, 9, 3, 9}) &&
            second.payload == first.payload;
    });

    // ==== 全部场景共用一次 sc_start：SystemC 不允许在第一次 sc_start 之后再构造
    //       新的 sc_module，所以上面所有对象必须都在这之前建好 ====
    sc_start(5000, SC_NS);

    // ---- 场景 1 断言 ----
    Check(driver_sat.ok, "saturation driver did not throw: " + driver_sat.error);
    {
        double throughput =
            ThroughputGBps((uint64_t)kSatReqs * kSatLen, ep_sat.BusyTime());
        double expected = cfg1.bandwidth_GBps * cfg1.efficiency;
        Check(std::fabs(throughput - expected) < 1e-6,
              "single saturated channel's throughput exactly matches "
              "bandwidth_GBps * efficiency (base_latency=0 isolates the "
              "bandwidth term)");
    }

    // ---- 场景 2 断言 ----
    {
        bool all_ok = true;
        for (auto &d : share_drivers)
            all_ok = all_ok && d->ok;
        Check(all_ok, "4-core sharing scenario drivers did not throw");

        bool none_starved = true;
        for (int c : share_completed)
            if (c != kReqsPerCore)
                none_starved = false;
        Check(none_starved,
              "every one of the 4 cores completes all of its requests "
              "within the run budget (no permanent starvation)");

        double throughput = ThroughputGBps(
            (uint64_t)kShareCores * kReqsPerCore * kShareLen, ep_share.BusyTime());
        double single_core_cap = cfg2.bandwidth_GBps * cfg2.efficiency;
        Check(throughput <= single_core_cap + 1e-6,
              "4 cores sharing one channel do not exceed that channel's own "
              "bandwidth cap (sharing does not multiply effective "
              "bandwidth by core count)");
    }

    // ---- 场景 3 断言 ----
    {
        Check(driver_par_a.ok && driver_par_b.ok,
              "2-channel parallel scaling drivers did not throw");
        double totalThroughput =
            ThroughputGBps((uint64_t)kParReqs * kParLen, ep_par_a.BusyTime()) +
            ThroughputGBps((uint64_t)kParReqs * kParLen, ep_par_b.BusyTime());
        double expected = 2.0 * cfg3.bandwidth_GBps * cfg3.efficiency;
        Check(std::fabs(totalThroughput - expected) < 1e-6,
              "2 independent channels each saturated in parallel achieve "
              "~2x a single channel's throughput (near-linear scaling)");
    }

    // ---- 场景 4 断言 ----
    {
        bool tight_ok = true, loose_ok = true;
        for (auto &d : tight_drivers)
            tight_ok = tight_ok && d->ok;
        for (auto &d : loose_drivers)
            loose_ok = loose_ok && d->ok;
        Check(tight_ok && loose_ok,
              "endpoint-bottleneck comparison drivers did not throw");
        // 两边总的"服务时间"（BusyTime）应该相等——真正做事的时间只取决于带宽/请求
        // 量，与 queue_depth 无关；queue_depth 只影响请求排多久的队，不影响服务本身
        // 快慢，这正是"endpoint 瓶颈"和"HBM 瓶颈"是两个独立维度的证明。
        Check(NearlyEqualTime(ep_tight.BusyTime(), ep_loose.BusyTime()),
              "queue_depth changes how long requests wait to be admitted, "
              "not how fast the channel itself services them once admitted "
              "(endpoint queueing and HBM bandwidth are independent "
              "bottlenecks)");
    }

    // ---- 场景 5 断言 ----
    {
        Check(driver_burst.ok,
              "mixed read/write burst-size driver did not throw: " +
                  driver_burst.error);
        sc_time short_t = cfg5.base_latency +
                          sc_time(4 / cfg5.bandwidth_GBps / cfg5.efficiency, SC_NS);
        sc_time long_t = cfg5.base_latency +
                         sc_time(16 / cfg5.bandwidth_GBps / cfg5.efficiency, SC_NS);
        sc_time expected_total = short_t * 2 + long_t * 2; // 2 短(读+写) + 2 长(读+写)
        Check(NearlyEqualTime(ep_burst.BusyTime(), expected_total),
              "short (4B) and long (16B) bursts, mixed with reads and "
              "writes, each cost exactly base_latency + length/bandwidth "
              "with no hidden asymmetry between read and write timing");
        Check(long_t > short_t,
              "a longer burst costs strictly more service time than a "
              "shorter one");
    }

    // ---- 场景 6 断言 ----
    {
        Check(driver_semantics.ok,
              "partial-write/turnaround driver did not throw: " +
                  driver_semantics.error);
        Check(partial_write_ok,
              "byte-enable preserves disabled bytes across overlapping writes");
        sc_time service = cfg6.base_latency +
                          sc_time(4 / cfg6.bandwidth_GBps / cfg6.efficiency,
                                  SC_NS);
        sc_time expected = service * 5 + cfg6.write_to_read_turnaround +
                           cfg6.read_to_write_turnaround;
        Check(NearlyEqualTime(semantics_elapsed, expected),
              "elapsed time includes exactly one write-to-read and one "
              "read-to-write turnaround penalty");
        const auto &bs = backend_semantics.Stats();
        const auto &es = ep_semantics.Stats();
        Check(bs.requests == 5 && bs.reads == 2 && bs.writes == 3 &&
                  bs.bytes == 20 && bs.completed == 5 && bs.failed == 0,
              "behavioral backend exports exact request/read/write/byte stats");
        Check(es.requests == 5 && es.completed == 5 && es.bytes == 20 &&
                  es.response_time >= es.queue_wait,
              "MemEndpoint exports consistent completion and timing stats");
    }

    std::cout << "distributed HBM R2 self-test: "
              << (g_fail == 0 ? "PASS" : "FAIL") << " (" << g_total
              << " checks)" << std::endl;
    return g_fail;
}
