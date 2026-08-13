// R3 分布式 HBM 自测：验证 DRAMSysHBMBackend 通过真实 DRAMSys 时序模型读写正确、
// 时延不是零时穿越；以及把 MemEndpointUnit 的 backend 从 Behavioral 换成 DRAMSys
// 后，CoreMemAdapter 整条链路依然工作（backend 可插拔契约成立）。
#include "die/port.h"
#include "defs/spec.h"
#include "memory/core_mem_adapter.h"
#include "memory/dramsys_hbm_backend.h"
#include "memory/hbm_address_map.h"
#include "memory/hbm_memspec.h"
#include "memory/hbm_r3_selftest.h"
#include "memory/hbm_runtime.h"
#include "memory/mem_endpoint_unit.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <systemc>
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

const char *kHbm2Cfg = "../DRAMSys/configs/hbm2-example.json";

} // namespace

int RunHbmR3SelfTest() {
    g_fail = g_total = 0;
    std::cout << "==== distributed HBM R3 self-test ====" << std::endl;

    // ---- 拓扑：1 颗 stack、1 个 channel（backend_granularity=channel，与
    //       hbm2-example.json 的单 channel memspec 匹配）----
    SetTopo(4, 4, 1, 1);
    auto hw = nlohmann::json::parse(R"JSON({
      "memory_system": {
        "topology": "distributed_hbm",
        "cache_policy": "none",
        "profiles": {
          "hbm2_1ch": {
            "generation": "HBM2", "channels_per_stack": 1,
            "pseudo_channels_per_channel": 2,
            "data_rate_gbps_per_pin": 2.0, "stack_bus_width_bits": 64
          }
        },
        "hbm_stacks": [
          {"stack_id": 0, "compute_die_id": 0, "profile": "hbm2_1ch",
           "side": "N", "start_idx": 0, "phy_span_tiles": 1,
           "capacity_bytes": 1048576, "backend_granularity": "channel",
           "channel_dram_config": "CFG0"}
        ],
        "address_policy": {
          "mode": "numa_local_interleave",
          "home_ranges": [{"die_id": 0, "base": 0, "size_bytes": 1048576}],
          "stack_interleave_bytes": 1048576,
          "channel_interleave_bytes": 1048576
        }
      }
    })JSON");
    hw["memory_system"]["hbm_stacks"][0]["channel_dram_config"] = kHbm2Cfg;
    bool setup_ok = false;
    try {
        ParseAndBuild(hw);
        setup_ok = true;
    } catch (const std::exception &e) {
        std::cout << "  [FAIL] setup threw: " << e.what() << std::endl;
    }
    Check(setup_ok, "R3 topology/memory_system config validates");

    // ---- 1. DRAMSysHBMBackend 直接读写往返，验证真实 DRAMSys 时序（非零时延）----
    DRAMSysHBMBackend direct_backend("direct_backend", kHbm2Cfg);
    bool direct_rw_ok = false;
    bool partial_burst_ok = false;
    sc_time direct_write_delay = SC_ZERO_TIME, direct_read_delay = SC_ZERO_TIME;
    sc_time row_hit_delay = SC_ZERO_TIME, row_miss_delay = SC_ZERO_TIME;
    DRAMSys::DecodedAddress row_hit_decoded, row_miss_decoded;
    ScriptDriver driverDirect("driverDirect", [&]() {
        auto issue = [&](MemCommand cmd, uint64_t address,
                         std::vector<uint8_t> &payload,
                         const std::vector<uint8_t> &byte_enable = {}) {
            auto tx = std::make_shared<HBMBackendTransaction>();
            tx->command = cmd;
            tx->address = address;
            tx->payload = payload;
            tx->byte_enable = byte_enable.empty()
                                  ? std::vector<uint8_t>(payload.size(), 0xff)
                                  : byte_enable;
            sc_event done;
            int status = -1;
            sc_time begin = sc_time_stamp();
            tx->complete = [&](sc_time delay, sc_time, int s,
                               const std::string &) {
                status = s;
                done.notify(delay);
            };
            direct_backend.Submit(tx);
            wait(done);
            payload = tx->payload;
            if (status != 0)
                throw std::runtime_error("direct non-blocking HBM access failed");
            return sc_time_stamp() - begin;
        };
        std::vector<uint8_t> wdata = {0xAA, 0xBB, 0xCC, 0xDD};
        std::vector<uint8_t> wbuf = wdata;
        direct_write_delay = issue(MemCommand::kWrite, 0, wbuf);
        std::vector<uint8_t> rbuf(4);
        direct_read_delay = issue(MemCommand::kRead, 0, rbuf);
        direct_rw_ok = (rbuf == wdata);

        // Logical byte accesses are expanded to legal 32-byte HBM2 bursts.
        // Exercise an unaligned write that crosses a burst boundary and prove
        // that masked bytes on both sides remain unchanged.
        std::vector<uint8_t> sentinel(64, 0x5a);
        std::vector<uint8_t> sentinel_write = sentinel;
        issue(MemCommand::kWrite, 512, sentinel_write);
        std::vector<uint8_t> patch = {0x10, 0x20, 0x30, 0x40,
                                      0x50, 0x60, 0x70};
        std::vector<uint8_t> patch_write = patch;
        std::vector<uint8_t> patch_enable(patch.size(), 0xff);
        patch_enable[1] = 0;
        patch_enable[5] = 0;
        issue(MemCommand::kWrite, 512 + 29, patch_write, patch_enable);
        std::vector<uint8_t> sentinel_read(64);
        issue(MemCommand::kRead, 512, sentinel_read);
        for (size_t i = 0; i < patch.size(); ++i)
            if (patch_enable[i]) sentinel[29 + i] = patch[i];
        partial_burst_ok = sentinel_read == sentinel;

        // 地址由 DRAMSys 自己的 decoder 生成，避免按物理位位置猜 row/bank。
        row_hit_decoded = direct_backend.DecodeBackendAddress(0);
        row_hit_decoded.column += 1;
        uint64_t row_hit_addr =
            direct_backend.EncodeBackendAddress(row_hit_decoded);
        row_miss_decoded = row_hit_decoded;
        row_miss_decoded.row += 1;
        uint64_t row_miss_addr =
            direct_backend.EncodeBackendAddress(row_miss_decoded);
        std::vector<uint8_t> row_buf(4);
        row_hit_delay = issue(MemCommand::kRead, row_hit_addr, row_buf);
        row_miss_delay = issue(MemCommand::kRead, row_miss_addr, row_buf);
    });

    // ---- 2. 通过 CoreMemAdapter/MemEndpointUnit 换上 DRAMSys backend，整条链路
    //          依然工作（backend 可插拔契约）----
    std::unique_ptr<HBMRuntime> runtime = BuildHBMBackends();
    CoreMemAdapter core("core_dramsys", 900);
    runtime->BindAdapter(core);
    bool integrated_ok = false;
    ScriptDriver driverIntegrated("driverIntegrated", [&]() {
        std::vector<uint8_t> wdata = {1, 2, 3, 4, 5, 6, 7, 8};
        core.Access(MemCommand::kWrite, 256, 8, wdata);
        MemMsg r = core.Access(MemCommand::kRead, 256, 8);
        integrated_ok = (r.payload == wdata);
    });

    // ---- 3. 同一 instance 的多 outstanding 非阻塞事务 ----
    DRAMSysHBMBackend concurrent_backend("concurrent_backend", kHbm2Cfg);
    constexpr int kConcurrent = 8;
    int concurrent_completed = 0;
    bool concurrent_ok = true;
    sc_time concurrent_elapsed = SC_ZERO_TIME;
    ScriptDriver driverConcurrent("driverConcurrent", [&]() {
        sc_event progress;
        std::vector<std::shared_ptr<HBMBackendTransaction>> transactions;
        sc_time begin = sc_time_stamp();
        for (int i = 0; i < kConcurrent; ++i) {
            auto tx = std::make_shared<HBMBackendTransaction>();
            tx->command = MemCommand::kRead;
            tx->address = (uint64_t)i * 64;
            tx->payload.resize(64);
            tx->byte_enable.assign(64, 0xff);
            tx->complete = [&](sc_time delay, sc_time service, int status,
                               const std::string &) {
                concurrent_ok = concurrent_ok && delay == SC_ZERO_TIME &&
                                service > SC_ZERO_TIME && status == 0;
                ++concurrent_completed;
                progress.notify(SC_ZERO_TIME);
            };
            transactions.push_back(tx);
            concurrent_backend.Submit(tx);
        }
        while (concurrent_completed != kConcurrent)
            wait(progress);
        concurrent_elapsed = sc_time_stamp() - begin;
    });

    sc_start(2000, SC_NS);

    Check(driverDirect.ok,
          "direct DRAMSysHBMBackend driver did not throw: " +
              driverDirect.error);
    Check(direct_rw_ok,
          "DRAMSysHBMBackend read-after-write returns exactly what was "
          "written, through real DRAMSys storage (not a toy in-memory map)");
    Check(partial_burst_ok,
          "unaligned logical write crosses a physical HBM2 burst boundary "
          "without changing adjacent masked bytes");
    Check(direct_write_delay > SC_ZERO_TIME && direct_read_delay > SC_ZERO_TIME,
          "DRAMSysHBMBackend::Access reports a real nonzero DRAMSys access "
          "delay (not a zero-time bypass)");
    Check(direct_write_delay != sc_time(60, SC_NS) ||
              direct_read_delay != sc_time(60, SC_NS),
          "non-blocking DRAMSys timing is not the legacy fixed 60ns "
          "blocking delay");
    Check(direct_backend.Stats().completed == 7 &&
              direct_backend.Stats().bytes == 151 &&
              direct_backend.Stats().failed == 0,
          "DRAMSys backend exports logical request/byte statistics independent "
          "of physical burst expansion");
    Check(row_hit_decoded.bank == row_miss_decoded.bank &&
              row_hit_decoded.bankgroup == row_miss_decoded.bankgroup &&
              row_hit_decoded.row != row_miss_decoded.row,
          "row-hit/miss addresses target the same decoded bank but different rows");
    Check(row_hit_delay > SC_ZERO_TIME && row_miss_delay > SC_ZERO_TIME &&
              row_hit_delay != row_miss_delay,
          "same-row and different-row accesses expose distinct DRAMSys timing");
    const auto &direct_trace = direct_backend.AccessTrace();
    Check(direct_trace.size() == 7 &&
              direct_trace[3].command == MemCommand::kWrite &&
              direct_trace[3].address == 541 &&
              direct_trace[3].service > SC_ZERO_TIME &&
              direct_trace.back().status == 0 &&
              direct_trace.back().service > SC_ZERO_TIME &&
              direct_trace.back().row == row_miss_decoded.row,
          "backend exports per-access command/address/bank/row/service trace");

    Check(driverIntegrated.ok,
          "CoreMemAdapter+MemEndpointUnit+DRAMSysHBMBackend integration "
          "driver did not throw: " +
              driverIntegrated.error);
    Check(integrated_ok,
          "swapping MemEndpointUnit's backend from BehavioralHBMBackend to "
          "DRAMSysHBMBackend requires no change to CoreMemAdapter and still "
          "round-trips data correctly (HBMBackend interface is genuinely "
          "pluggable)");
    Check(runtime->Size() == 1 && runtime->Find(0, 0) != nullptr &&
              dynamic_cast<DRAMSysHBMBackend *>(
                  runtime->Find(0, 0)->backend.get()) != nullptr,
          "BuildHBMBackends creates and owns exactly one configured DRAMSys "
          "instance plus endpoint at channel granularity");

    Check(driverConcurrent.ok,
          "concurrent DRAMSys driver did not throw: " +
              driverConcurrent.error);
    Check(concurrent_ok && concurrent_completed == kConcurrent,
          "all concurrently submitted non-blocking transactions complete once");
    Check(concurrent_backend.PeakInFlightCount() > 1,
          "backend holds more than one transaction in flight after END_REQ");
    Check(concurrent_backend.AccessTrace().size() == kConcurrent &&
              concurrent_backend.Stats().completed == kConcurrent &&
              concurrent_backend.Stats().failed == 0,
          "concurrent trace and aggregate completion statistics agree");

    // ---- 3. capacity 交叉核对：R0 的 ResolveHbmMemSpec 和 R3 实际构造的
    //          DRAMSysWrapper 各自独立解析同一份 memspec，应得到同一个容量 ----
    ResolvedHbmMemSpec resolved = ResolveHbmMemSpec(kHbm2Cfg);
    Check(resolved.capacity_bytes == direct_backend.CapacityBytes(),
          "R0's ResolveHbmMemSpec (constructs MemSpecHBM2 directly, working "
          "around DRAMSys::createMemSpec being private) agrees with the "
          "capacity a real DRAMSysWrapper instance reports for the same "
          "memspec file");
    double observed_GBps = concurrent_elapsed > SC_ZERO_TIME
                               ? (double)(kConcurrent * 64) /
                                     (concurrent_elapsed.to_seconds() * 1e9)
                               : 0.0;
    Check(observed_GBps > 0.0 &&
              observed_GBps <= resolved.instance_bandwidth_GBps * 1.05,
          "finite-trace throughput is positive and does not exceed resolved "
          "instance theoretical bandwidth");

    std::cout << "distributed HBM R3 self-test: "
              << (g_fail == 0 ? "PASS" : "FAIL") << " (" << g_total
              << " checks)" << std::endl;
    return g_fail;
}
