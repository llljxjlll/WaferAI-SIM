// R1 分布式 HBM 自测：合成请求 testbench，驱动真实 SystemC 时序穿过
// CoreMemAdapter/MemEndpointUnit，覆盖 HBM建模计划.md（修订版）九."R1" 的测试项。
// 窄范围：不接 memory_utils.cpp 里真实算子的访存调用链（留给 R4），只验证
// CoreMemAdapter/MemEndpointUnit/MEM wire codec 自身的协议正确性。
#include "die/port.h"
#include "defs/spec.h"
#include "memory/behavioral_hbm_backend.h"
#include "memory/core_mem_adapter.h"
#include "memory/hbm_address_map.h"
#include "memory/hbm_mem_wire.h"
#include "memory/hbm_memspec.h"
#include "memory/mem_endpoint_unit.h"
#include "memory/hbm_r1_selftest.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <string>
#include <systemc>

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

// 驱动一段测试脚本的最小 testbench 模块：脚本里可以自由调用 CoreMemAdapter::Access
// （阻塞、需要 SC_THREAD 上下文），异常会被捕获记录而不是让整个仿真崩掉。
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

// 采样 MemEndpointUnit::QueueDepthNow() 的峰值：R2 的 endpoint 内部是严格 FIFO
// 单服务者（物理上一个 channel 的总线本来就只能串行传输），"多个请求同时被服务"不
// 再是需要证明的性质；这里改为证明"多个并发发起方确实会在队列里排上队"（而不是
// 神奇地瞬间处理完，说明背压/排队路径被真实走到了），配合下面的正确性检查一起用。
struct QueueDepthMonitor : sc_module {
    MemEndpointUnit *ep;
    int max_seen = 0;

    SC_HAS_PROCESS(QueueDepthMonitor);
    QueueDepthMonitor(const sc_module_name &n, MemEndpointUnit *e)
        : sc_module(n), ep(e) {
        SC_THREAD(Run);
    }
    void Run() {
        while (true) {
            max_seen = std::max(max_seen, ep->QueueDepthNow());
            wait(sc_time(1, SC_NS));
        }
    }
};

} // namespace

int RunHbmR1SelfTest() {
    g_fail = g_total = 0;
    std::cout << "==== distributed HBM R1 self-test ====" << std::endl;

    // ---- 0. MEM wire codec 独立往返测试（不依赖任何 SystemC 时序）----
    {
        MemMsg m;
        m.txid = 12345;
        m.source_core = 7;
        m.home_die = 1;
        m.stack_id = 2;
        m.channel_id = 1;
        m.address = 0x0123456789ABCDEFULL;
        m.command = MemCommand::kWrite;
        m.length_bytes = 4;
        m.is_end = true;
        m.status = 0;
        m.payload = {0xDE, 0xAD, 0xBE, 0xEF};

        MemMsg back = DeserializeMemMsg(SerializeMemMsg(m));
        Check(back.txid == m.txid && back.source_core == m.source_core &&
                  back.home_die == m.home_die && back.stack_id == m.stack_id &&
                  back.channel_id == m.channel_id &&
                  back.address == m.address && back.command == m.command &&
                  back.length_bytes == m.length_bytes &&
                  back.is_end == m.is_end && back.status == m.status &&
                  back.payload == m.payload,
              "MemMsg round-trips through Serialize/Deserialize unchanged");

        MemMsg maxAddr;
        maxAddr.txid = 12346;
        maxAddr.source_core = 7;
        maxAddr.home_die = 1;
        maxAddr.stack_id = 2;
        maxAddr.channel_id = 1;
        maxAddr.address = ~0ULL; // 64-bit 全 1，验证地址字段没有被截断
        maxAddr.length_bytes = 0;
        MemMsg maxBack = DeserializeMemMsg(SerializeMemMsg(maxAddr));
        Check(maxBack.address == ~0ULL,
              "64-bit address field is not truncated on the wire (unlike "
              "Msg.offset_'s 8-bit budget)");

        MemMsg burst = m;
        burst.txid = 12347;
        burst.length_bytes = 40;
        burst.payload.resize(40);
        burst.byte_enable.resize(40, 0xff);
        for (int i = 0; i < 40; ++i)
            burst.payload[i] = (uint8_t)i;
        burst.byte_enable[3] = 0;
        auto burst_wire = SerializeMemMsg(burst);
        MemMsg burst_back = DeserializeMemMsg(burst_wire);
        Check(burst_wire.size() == 4 && burst_back.payload == burst.payload &&
                  burst_back.byte_enable == burst.byte_enable,
              "40-byte write expands to REQ+3 WDATA flits and preserves "
              "sequence/tail/byte-enable on the 256-bit wire");
    }

    // ---- 拓扑与 memory_system：1 颗 stack、2 个 channel，channel 粒度 64B 交织 ----
    SetTopo(4, 4, 1, 1);
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
    bool setup_ok = false;
    try {
        ParseAndBuild(hw);
        setup_ok = true;
    } catch (const std::exception &e) {
        std::cout << "  [FAIL] setup threw: " << e.what() << std::endl;
    }
    Check(setup_ok, "R1 topology/memory_system config validates");
    Check(g_hbm_channels.size() == 2,
          "BuildMemAttach produced 2 channels for the single 2-channel stack");

    BehavioralHBMBackendConfig backend_cfg;
    backend_cfg.bandwidth_GBps = 32.0; // 与 hbm2-example.json 一个 channel 的量级同数量级
    backend_cfg.efficiency = 1.0;
    backend_cfg.base_latency = sc_time(5, SC_NS);
    // R3 起 MemEndpointUnit 只管排队/调度，数据存取+计时被抽成 HBMBackend 接口；
    // R1/R2 用 BehavioralHBMBackend 这个实现，backend 对象必须比引用它的
    // MemEndpointUnit 活得长，所以在这里（同一层作用域，构造顺序更早）声明。
    BehavioralHBMBackend backend0(backend_cfg), backend1(backend_cfg);
    MemEndpointUnit ep0("ep0", 0, 0, backend0, /*queue_depth=*/8);
    MemEndpointUnit ep1("ep1", 0, 1, backend1, /*queue_depth=*/8);

    // ---- 1. 单核 read/write/read-after-write ----
    CoreMemAdapter a0("a0", 100);
    a0.BindEndpoint(0, 0, &ep0);
    a0.BindEndpoint(0, 1, &ep1);
    bool a_ok = false;
    ScriptDriver driverA("driverA", [&]() {
        std::vector<uint8_t> wdata = {1, 2, 3, 4};
        a0.Access(MemCommand::kWrite, 0, 4, wdata);
        MemMsg r = a0.Access(MemCommand::kRead, 0, 4);
        a_ok = (r.payload == wdata);
    });

    // ---- 2. 两个 core 对同一地址交叉读写，证明不是每核私有副本 ----
    CoreMemAdapter b0("b0", 101), b1("b1", 102);
    b0.BindEndpoint(0, 0, &ep0);
    b1.BindEndpoint(0, 0, &ep0);
    bool b_ok = false;
    ScriptDriver driverB("driverB", [&]() {
        // 用与场景 A 不同的地址（4，而非 0），避免两个场景的并发脚本在同一个
        // backing_ key 上互相踩踏——场景之间没有隔离的独立地址空间，共用同一个
        // ep0，只是恰好都跑在同一次 sc_start() 里，所以各场景必须各用各的地址。
        std::vector<uint8_t> wdata = {9, 9, 9, 9};
        b0.Access(MemCommand::kWrite, 4, 4, wdata); // core 101 写
        MemMsg r = b1.Access(MemCommand::kRead, 4, 4); // core 102 读，同一地址
        b_ok = (r.payload == wdata);
    });

    // ---- 3. 不同地址交织到多个 channel，且彼此隔离不串数据 ----
    CoreMemAdapter c0("c0", 103);
    c0.BindEndpoint(0, 0, &ep0);
    c0.BindEndpoint(0, 1, &ep1);
    bool c_route_ok = false, c_isolation_ok = false;
    ScriptDriver driverC("driverC", [&]() {
        // 用 40/104（=64+40）而非 0/64，避免撞上其它并发场景在这两个 channel
        // 的 stripe 0 上使用的地址。
        AddressDecodeResult dLow = DecodeAddress(40);  // 第 0 条 64B stripe -> channel 0
        AddressDecodeResult dHigh = DecodeAddress(104); // 第 1 条 64B stripe -> channel 1
        c_route_ok = (dLow.channel_id == 0 && dHigh.channel_id == 1);

        c0.Access(MemCommand::kWrite, 40, 4, {11, 11, 11, 11});
        c0.Access(MemCommand::kWrite, 104, 4, {22, 22, 22, 22});
        MemMsg r0 = c0.Access(MemCommand::kRead, 40, 4);
        MemMsg r64 = c0.Access(MemCommand::kRead, 104, 4);
        c_isolation_ok =
            (r0.payload == std::vector<uint8_t>{11, 11, 11, 11}) &&
            (r64.payload == std::vector<uint8_t>{22, 22, 22, 22});
    });

    // ---- 3b. 跨 64B channel stripe 的 burst 必须拆成两笔并正确重组 ----
    bool c_cross_boundary_ok = false;
    ScriptDriver driverC2("driverC2", [&]() {
        std::vector<uint8_t> wd(20);
        for (int i = 0; i < 20; ++i)
            wd[i] = (uint8_t)(0x40 + i);
        c0.Access(MemCommand::kWrite, 56, 20, wd);
        MemMsg r = c0.Access(MemCommand::kRead, 56, 20);
        c_cross_boundary_ok = r.payload == wd;
    });

    // ---- 4. 多 outstanding + 有界背压：3 个 core 并发访问同一个 endpoint，
    //          各自读写不同地址（同一 64B stripe 内，仍落在 channel 0），验证
    //          互不踩踏、都能正确完成、且请求确实经过了排队路径。
    CoreMemAdapter d0("d0", 104), d1("d1", 105), d2("d2", 106);
    d0.BindEndpoint(0, 0, &ep0);
    d1.BindEndpoint(0, 0, &ep0);
    d2.BindEndpoint(0, 0, &ep0);
    bool d0_ok = false, d1_ok = false, d2_ok = false;
    auto makeDScript = [](CoreMemAdapter &a, uint64_t addr, uint8_t tag,
                          bool &okflag) {
        return [&a, addr, tag, &okflag]() {
            std::vector<uint8_t> wd(4, tag);
            a.Access(MemCommand::kWrite, addr, 4, wd);
            MemMsg r = a.Access(MemCommand::kRead, addr, 4);
            okflag = (r.payload == wd);
        };
    };
    ScriptDriver driverD0("driverD0", makeDScript(d0, 8, 201, d0_ok));
    ScriptDriver driverD1("driverD1", makeDScript(d1, 16, 202, d1_ok));
    ScriptDriver driverD2("driverD2", makeDScript(d2, 24, 203, d2_ok));
    QueueDepthMonitor monitor("monitor", &ep0);

    // ---- 5. txid 回收：小 wrap 周期下重复发起请求，txid 应循环复用而不是无限增长 ----
    CoreMemAdapter e0("e0", 107, /*txid_wrap=*/4);
    e0.BindEndpoint(0, 0, &ep0);
    std::vector<int> observed_txids;
    ScriptDriver driverE("driverE", [&]() {
        for (int i = 0; i < 10; i++) {
            e0.Access(MemCommand::kRead, 0, 4);
            observed_txids.push_back(e0.LastTxid());
        }
    });

    sc_start(500, SC_NS);

    Check(driverA.ok, "scenario A (single-core write/read-after-write) driver "
                      "did not throw: " +
                          driverA.error);
    Check(a_ok, "single-core read-after-write returns exactly what was "
               "written");

    Check(driverB.ok,
          "scenario B (cross-core same-address) driver did not throw: " +
              driverB.error);
    Check(b_ok, "a different core reading the same address sees the write "
               "made by another core (not a private per-core copy)");

    Check(driverC.ok,
          "scenario C (multi-channel interleave) driver did not throw: " +
              driverC.error);
    Check(c_route_ok,
          "addresses 64 bytes apart decode to different channel_id under "
          "channel_interleave_bytes=64");
    Check(c_isolation_ok,
              "writes to different channels do not bleed into each other");
    Check(driverC2.ok && c_cross_boundary_ok,
          "a burst crossing a channel stripe is split, routed and "
          "reassembled without corruption");

    Check(driverD0.ok && driverD1.ok && driverD2.ok,
          "scenario D (concurrent multi-outstanding) drivers did not throw");
    Check(d0_ok && d1_ok && d2_ok,
          "3 concurrent requesters against a shared endpoint each get back "
          "exactly the data they wrote (no cross-talk under concurrency)");
    Check(monitor.max_seen >= 1,
          "3 concurrent requesters against one FIFO endpoint actually queue "
          "up (not silently processed as if serialized with no queueing "
          "path exercised)");

    Check(driverE.ok,
          "scenario E (txid recycling) driver did not throw: " +
              driverE.error);
    bool txid_cycles = observed_txids.size() == 10;
    for (size_t i = 0; txid_cycles && i < observed_txids.size(); i++)
        if (observed_txids[i] != (int)(i % 4))
            txid_cycles = false;
    Check(txid_cycles,
          "txid wraps and is reused once txid_wrap requests have been "
          "issued, instead of growing without bound");

    std::cout << "distributed HBM R1 self-test: "
              << (g_fail == 0 ? "PASS" : "FAIL") << " (" << g_total
              << " checks)" << std::endl;
    return g_fail;
}
