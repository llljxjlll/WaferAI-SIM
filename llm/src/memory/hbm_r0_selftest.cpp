// R0 分布式 HBM 自测：配置解析、per-die MEM keep-out、地址映射、resolved memspec
// 交叉校验。纯配置/结构测试，不建 SystemC 仿真，不接入 executor 请求路径。
// 覆盖 notes/extensions/DRAM/HBM建模计划.md（修订版）九."R0" 列出的全部测试项。
#include "die/port.h"
#include "defs/spec.h"
#include "memory/hbm_address_map.h"
#include "memory/hbm_memspec.h"
#include "memory/hbm_r0_selftest.h"

#include <functional>
#include <iostream>
#include <string>

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

bool Throws(const std::function<void()> &f) {
    try {
        f();
    } catch (const std::runtime_error &) {
        return true;
    } catch (...) {
        return true;
    }
    return false;
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
    g_die_ports = D2DPortTable{}; // 默认无模板端口，keep-out 测试会单独注入
    g_d2d_links.clear();
}

// R0 生产路径的调用顺序（与 config_utils.cpp::ParseHardwareConfig 一致）。
void ParseAndBuild(const nlohmann::json &hw) {
    ParseMemorySystem(hw);
    BuildMemAttach();
    ValidateMemAttach();
    ValidateAddressPolicy();
    ValidateHbmMemSpecConsistency();
}

const char *kHbm2Cfg = "../DRAMSys/configs/hbm2-example.json";

} // namespace

int RunHbmR0SelfTest() {
    g_fail = g_total = 0;
    std::cout << "==== distributed HBM R0 self-test ====" << std::endl;

    // ---- 1. 合法的单 die 多 stack、本地交织配置 ----
    {
        SetTopo(4, 4, 1, 1);
        auto hw = nlohmann::json::parse(R"JSON({
          "memory_system": {
            "topology": "distributed_hbm",
            "cache_policy": "none",
            "profiles": {
              "hbm2_2ch": {
                "generation": "HBM2",
                "channels_per_stack": 2,
                "pseudo_channels_per_channel": 2,
                "data_rate_gbps_per_pin": 2.0,
                "stack_bus_width_bits": 128
              }
            },
            "hbm_stacks": [
              {"stack_id": 0, "compute_die_id": 0, "profile": "hbm2_2ch",
               "side": "N", "start_idx": 0, "phy_span_tiles": 2,
               "capacity_bytes": 1048576, "backend_granularity": "channel",
               "channel_dram_config": "CFG0", "bandwidth_cap_GBps": 1.0},
              {"stack_id": 1, "compute_die_id": 0, "profile": "hbm2_2ch",
               "side": "S", "start_idx": 0, "phy_span_tiles": 2,
               "capacity_bytes": 1048576, "backend_granularity": "channel",
               "channel_dram_config": "CFG0", "bandwidth_cap_GBps": 1.0}
            ],
            "address_policy": {
              "mode": "numa_local_interleave",
              "home_ranges": [{"die_id": 0, "base": 0, "size_bytes": 1048576}],
              "stack_interleave_bytes": 256,
              "channel_interleave_bytes": 256
            }
          }
        })JSON");
        hw["memory_system"]["hbm_stacks"][0]["channel_dram_config"] = kHbm2Cfg;
        hw["memory_system"]["hbm_stacks"][1]["channel_dram_config"] = kHbm2Cfg;

        Check(!Throws([&] { ParseAndBuild(hw); }),
              "single-die multi-stack local-interleave config validates");
        Check(g_memory_system_active, "memory_system_active set after valid config");
        Check(g_hbm_channels.size() == 4,
              "BuildMemAttach produced 2 channels per stack x 2 stacks");
        AddressDecodeResult pseudo = DecodeAddress(1024);
        Check(pseudo.pseudo_channel_id == 1,
              "pseudo-channel id is decoded from the channel-local address");
    }

    // ---- 2/3. 合法的多 die NUMA home range + 地址映射确定性 ----
    {
        SetTopo(2, 2, 2, 1);
        auto hw = nlohmann::json::parse(R"JSON({
          "memory_system": {
            "topology": "distributed_hbm",
            "cache_policy": "none",
            "profiles": {
              "hbm2_1ch": {
                "generation": "HBM2",
                "channels_per_stack": 1,
                "pseudo_channels_per_channel": 2,
                "data_rate_gbps_per_pin": 2.0,
                "stack_bus_width_bits": 64
              }
            },
            "hbm_stacks": [
              {"stack_id": 0, "compute_die_id": 0, "profile": "hbm2_1ch",
               "side": "N", "start_idx": 0, "phy_span_tiles": 1,
               "capacity_bytes": 1048576, "backend_granularity": "channel",
               "channel_dram_config": "CFG0"},
              {"stack_id": 1, "compute_die_id": 1, "profile": "hbm2_1ch",
               "side": "N", "start_idx": 0, "phy_span_tiles": 1,
               "capacity_bytes": 1048576, "backend_granularity": "channel",
               "channel_dram_config": "CFG0"}
            ],
            "address_policy": {
              "mode": "numa_local_interleave",
              "home_ranges": [
                {"die_id": 0, "base": 0, "size_bytes": 1048576},
                {"die_id": 1, "base": 1048576, "size_bytes": 1048576}
              ],
              "stack_interleave_bytes": 256,
              "channel_interleave_bytes": 256
            }
          }
        })JSON");
        hw["memory_system"]["hbm_stacks"][0]["channel_dram_config"] = kHbm2Cfg;
        hw["memory_system"]["hbm_stacks"][1]["channel_dram_config"] = kHbm2Cfg;

        Check(!Throws([&] { ParseAndBuild(hw); }),
              "two-die NUMA home-range config validates");

        AddressDecodeResult r0a, r0b, r1;
        Check(!Throws([&] { r0a = DecodeAddress(4096); }),
              "DecodeAddress resolves an address inside die-0 home range");
        Check(!Throws([&] { r0b = DecodeAddress(4096); }),
              "DecodeAddress callable a second time for the same address");
        Check(r0a.home_die == r0b.home_die && r0a.stack_id == r0b.stack_id &&
                  r0a.channel_id == r0b.channel_id &&
                  r0a.local_address == r0b.local_address,
              "address decode is deterministic: same address -> same home "
              "regardless of caller/requester");
        Check(r0a.home_die == 0, "die-0 range decodes to home_die 0");
        Check(!Throws([&] { r1 = DecodeAddress(1048576 + 4096); }),
              "DecodeAddress resolves an address inside die-1 home range");
        Check(r1.home_die == 1, "die-1 range decodes to home_die 1");
    }

    // ---- 4. range 重叠/空洞/stripe 未对齐/容量越界 ----
    {
        SetTopo(2, 2, 2, 1);
        auto base_hw = nlohmann::json::parse(R"JSON({
          "memory_system": {
            "profiles": {
              "hbm2_1ch": {"generation": "HBM2", "channels_per_stack": 1,
                          "pseudo_channels_per_channel": 2,
                          "data_rate_gbps_per_pin": 2.0,
                          "stack_bus_width_bits": 64}
            },
            "hbm_stacks": [
              {"stack_id": 0, "compute_die_id": 0, "profile": "hbm2_1ch",
               "side": "N", "start_idx": 0, "phy_span_tiles": 1,
               "capacity_bytes": 10485760, "backend_granularity": "channel",
               "channel_dram_config": "CFG0"},
              {"stack_id": 1, "compute_die_id": 1, "profile": "hbm2_1ch",
               "side": "N", "start_idx": 0, "phy_span_tiles": 1,
               "capacity_bytes": 10485760, "backend_granularity": "channel",
               "channel_dram_config": "CFG0"}
            ],
            "address_policy": {"mode": "numa_local_interleave",
                               "home_ranges": [],
                               "stack_interleave_bytes": 256,
                               "channel_interleave_bytes": 256}
          }
        })JSON");
        base_hw["memory_system"]["hbm_stacks"][0]["channel_dram_config"] = kHbm2Cfg;
        base_hw["memory_system"]["hbm_stacks"][1]["channel_dram_config"] = kHbm2Cfg;

        auto withRanges = [&](nlohmann::json ranges) {
            auto hw = base_hw;
            hw["memory_system"]["address_policy"]["home_ranges"] = ranges;
            return hw;
        };

        auto overlap = withRanges(nlohmann::json::parse(R"JSON(
            [{"die_id":0,"base":0,"size_bytes":1048576},
             {"die_id":1,"base":524288,"size_bytes":1048576}])JSON"));
        Check(Throws([&] { ParseAndBuild(overlap); }),
              "overlapping home ranges are rejected");

        auto gap = withRanges(nlohmann::json::parse(R"JSON(
            [{"die_id":0,"base":0,"size_bytes":1048576},
             {"die_id":1,"base":2097152,"size_bytes":1048576}])JSON"));
        Check(Throws([&] { ParseAndBuild(gap); }),
              "a gap between home ranges is rejected (allow_gaps defaults "
              "to false)");

        auto misaligned = withRanges(nlohmann::json::parse(R"JSON(
            [{"die_id":0,"base":0,"size_bytes":1000},
             {"die_id":1,"base":1000,"size_bytes":1048576}])JSON"));
        Check(Throws([&] { ParseAndBuild(misaligned); }),
              "home range size not aligned to channel_interleave_bytes is "
              "rejected (unaligned stripe)");

        auto oversize = withRanges(nlohmann::json::parse(R"JSON(
            [{"die_id":0,"base":0,"size_bytes":1073741824},
             {"die_id":1,"base":1073741824,"size_bytes":1048576}])JSON"));
        Check(Throws([&] { ParseAndBuild(oversize); }),
              "home range size exceeding attached stack capacity is "
              "rejected");
    }

    // ---- 5. per-die MEM/C2C/HOST keep-out 冲突与 corner 重复占用 ----
    {
        SetTopo(4, 4, 1, 1);

        // 5a：MEM 端口区间与既有 die_ports 模板端口（HOST）重叠。
        g_die_ports = D2DPortTable{};
        g_die_ports.active = true;
        D2DPort host_port;
        host_port.port_id = 0;
        host_port.side = NORTH;
        host_port.tile = 12; // TileFor(NORTH, 0) at GRID_X=GRID_Y=4
        host_port.role = ROLE_HOST;
        g_die_ports.ports.push_back(host_port);

        auto keepoutHw = nlohmann::json::parse(R"JSON({
          "memory_system": {
            "profiles": {
              "hbm2_1ch": {"generation": "HBM2", "channels_per_stack": 1,
                          "pseudo_channels_per_channel": 2,
                          "data_rate_gbps_per_pin": 2.0,
                          "stack_bus_width_bits": 64}
            },
            "hbm_stacks": [
              {"stack_id": 0, "compute_die_id": 0, "profile": "hbm2_1ch",
               "side": "N", "start_idx": 0, "phy_span_tiles": 1,
               "capacity_bytes": 1048576, "backend_granularity": "channel",
               "channel_dram_config": "CFG0"}
            ],
            "address_policy": {"mode": "numa_local_interleave",
                               "home_ranges": [],
                               "stack_interleave_bytes": 256,
                               "channel_interleave_bytes": 256}
          }
        })JSON");
        keepoutHw["memory_system"]["hbm_stacks"][0]["channel_dram_config"] =
            kHbm2Cfg;
        Check(Throws([&] { ParseAndBuild(keepoutHw); }),
              "MEM tile span overlapping an existing HOST template port is "
              "rejected");

        // 5b：corner tile 同时属于 N 边与 E 边，两颗 stack 分别声明会双重占用。
        g_die_ports = D2DPortTable{}; // 清空 5a 注入的模板端口
        auto cornerHw = nlohmann::json::parse(R"JSON({
          "memory_system": {
            "profiles": {
              "hbm2_1ch": {"generation": "HBM2", "channels_per_stack": 1,
                          "pseudo_channels_per_channel": 2,
                          "data_rate_gbps_per_pin": 2.0,
                          "stack_bus_width_bits": 64}
            },
            "hbm_stacks": [
              {"stack_id": 0, "compute_die_id": 0, "profile": "hbm2_1ch",
               "side": "N", "start_idx": 3, "phy_span_tiles": 1,
               "capacity_bytes": 1048576, "backend_granularity": "channel",
               "channel_dram_config": "CFG0"},
              {"stack_id": 1, "compute_die_id": 0, "profile": "hbm2_1ch",
               "side": "E", "start_idx": 3, "phy_span_tiles": 1,
               "capacity_bytes": 1048576, "backend_granularity": "channel",
               "channel_dram_config": "CFG0"}
            ],
            "address_policy": {"mode": "numa_local_interleave",
                               "home_ranges": [],
                               "stack_interleave_bytes": 256,
                               "channel_interleave_bytes": 256}
          }
        })JSON");
        cornerHw["memory_system"]["hbm_stacks"][0]["channel_dram_config"] =
            kHbm2Cfg;
        cornerHw["memory_system"]["hbm_stacks"][1]["channel_dram_config"] =
            kHbm2Cfg;
        Check(Throws([&] { ParseAndBuild(cornerHw); }),
              "two stacks whose spans alias the same NE corner tile are "
              "rejected (corner double occupancy)");
    }

    // ---- 6. mislabeled memspec / channel 数与实例粒度不匹配 ----
    {
        SetTopo(4, 4, 1, 1);

        // 6a：profile 声称 HBM3，但引用的 memspec 实际是 HBM2。
        auto mislabeled = nlohmann::json::parse(R"JSON({
          "memory_system": {
            "profiles": {
              "fake_hbm3": {"generation": "HBM3", "channels_per_stack": 1,
                            "pseudo_channels_per_channel": 2,
                            "data_rate_gbps_per_pin": 2.0,
                            "stack_bus_width_bits": 64}
            },
            "hbm_stacks": [
              {"stack_id": 0, "compute_die_id": 0, "profile": "fake_hbm3",
               "side": "N", "start_idx": 0, "phy_span_tiles": 1,
               "capacity_bytes": 1048576, "backend_granularity": "channel",
               "channel_dram_config": "CFG0"}
            ],
            "address_policy": {"mode": "numa_local_interleave",
                               "home_ranges": [],
                               "stack_interleave_bytes": 256,
                               "channel_interleave_bytes": 256}
          }
        })JSON");
        mislabeled["memory_system"]["hbm_stacks"][0]["channel_dram_config"] =
            kHbm2Cfg;
        Check(Throws([&] { ParseAndBuild(mislabeled); }),
              "profile.generation='HBM3' against a memspec that actually "
              "declares HBM2 is rejected (mislabeled memspec)");

        // 6b：backend_granularity=stack 但 memspec 只描述 1 个 channel，
        // 与 profile 声明的 channels_per_stack=8 不一致。
        SetTopo(4, 4, 1, 1);
        auto granMismatch = nlohmann::json::parse(R"JSON({
          "memory_system": {
            "profiles": {
              "hbm2_8ch": {"generation": "HBM2", "channels_per_stack": 8,
                          "pseudo_channels_per_channel": 2,
                          "data_rate_gbps_per_pin": 2.0,
                          "stack_bus_width_bits": 64}
            },
            "hbm_stacks": [
              {"stack_id": 0, "compute_die_id": 0, "profile": "hbm2_8ch",
               "side": "N", "start_idx": 0, "phy_span_tiles": 1,
               "capacity_bytes": 1048576, "backend_granularity": "stack",
               "channel_dram_config": "CFG0"}
            ],
            "address_policy": {"mode": "numa_local_interleave",
                               "home_ranges": [],
                               "stack_interleave_bytes": 256,
                               "channel_interleave_bytes": 256}
          }
        })JSON");
        granMismatch["memory_system"]["hbm_stacks"][0]["channel_dram_config"] =
            kHbm2Cfg;
        Check(Throws([&] { ParseAndBuild(granMismatch); }),
              "backend_granularity=stack with profile.channels_per_stack=8 "
              "against a single-channel memspec is rejected");

        // 6c：显式越界带宽/容量（超过折算到整颗 stack 的理论值，不允许任何倍数容差）。
        SetTopo(4, 4, 1, 1);
        auto bwOver = nlohmann::json::parse(R"JSON({
          "memory_system": {
            "profiles": {
              "hbm2_1ch": {"generation": "HBM2", "channels_per_stack": 1,
                          "pseudo_channels_per_channel": 2,
                          "data_rate_gbps_per_pin": 2.0,
                          "stack_bus_width_bits": 64}
            },
            "hbm_stacks": [
              {"stack_id": 0, "compute_die_id": 0, "profile": "hbm2_1ch",
               "side": "N", "start_idx": 0, "phy_span_tiles": 1,
               "capacity_bytes": 1048576, "backend_granularity": "channel",
               "channel_dram_config": "CFG0", "bandwidth_cap_GBps": 1.0e9}
            ],
            "address_policy": {"mode": "numa_local_interleave",
                               "home_ranges": [],
                               "stack_interleave_bytes": 256,
                               "channel_interleave_bytes": 256}
          }
        })JSON");
        bwOver["memory_system"]["hbm_stacks"][0]["channel_dram_config"] =
            kHbm2Cfg;
        Check(Throws([&] { ParseAndBuild(bwOver); }),
              "bandwidth_cap_GBps far exceeding the resolved memspec "
              "theoretical bandwidth is rejected with no tolerance margin");

        SetTopo(4, 4, 1, 1);
        auto capOver = nlohmann::json::parse(R"JSON({
          "memory_system": {
            "profiles": {
              "hbm2_1ch": {"generation": "HBM2", "channels_per_stack": 1,
                          "pseudo_channels_per_channel": 2,
                          "data_rate_gbps_per_pin": 2.0,
                          "stack_bus_width_bits": 64}
            },
            "hbm_stacks": [
              {"stack_id": 0, "compute_die_id": 0, "profile": "hbm2_1ch",
               "side": "N", "start_idx": 0, "phy_span_tiles": 1,
               "capacity_bytes": 1125899906842624, "backend_granularity": "channel",
               "channel_dram_config": "CFG0"}
            ],
            "address_policy": {"mode": "numa_local_interleave",
                               "home_ranges": [],
                               "stack_interleave_bytes": 256,
                               "channel_interleave_bytes": 256}
          }
        })JSON");
        capOver["memory_system"]["hbm_stacks"][0]["channel_dram_config"] =
            kHbm2Cfg;
        Check(Throws([&] { ParseAndBuild(capOver); }),
              "capacity_bytes far exceeding the resolved memspec "
              "addressable capacity is rejected");
    }

    // ---- 7. local_interleave：每个 die 可有重叠的独立地址空间 ----
    {
        SetTopo(4, 4, 2, 1);
        auto hw = nlohmann::json::parse(R"JSON({
          "memory_system": {
            "topology": "distributed_hbm", "cache_policy": "none",
            "profiles": {
              "hbm2_1ch": {"generation":"HBM2","channels_per_stack":1,
                "pseudo_channels_per_channel":2,
                "data_rate_gbps_per_pin":2.0,"stack_bus_width_bits":64}
            },
            "hbm_stacks": [
              {"stack_id":0,"compute_die_id":0,"profile":"hbm2_1ch",
               "side":"N","start_idx":1,"phy_span_tiles":1,
               "capacity_bytes":1048576,"backend_granularity":"channel",
               "channel_dram_config":"CFG0"},
              {"stack_id":1,"compute_die_id":1,"profile":"hbm2_1ch",
               "side":"N","start_idx":1,"phy_span_tiles":1,
               "capacity_bytes":1048576,"backend_granularity":"channel",
               "channel_dram_config":"CFG0"}
            ],
            "address_policy": {"mode":"local_interleave",
              "home_ranges":[
                {"die_id":0,"base":0,"size_bytes":1048576},
                {"die_id":1,"base":0,"size_bytes":1048576}],
              "stack_interleave_bytes":256,"channel_interleave_bytes":256,
              "pseudo_channel_interleave_bytes":256}
          }
        })JSON");
        for (auto &s : hw["memory_system"]["hbm_stacks"])
            s["channel_dram_config"] = kHbm2Cfg;
        Check(!Throws([&] { ParseAndBuild(hw); }),
              "local_interleave accepts overlapping per-die address ranges");
        AddressDecodeResult d0 = DecodeAddress(512, 0);
        AddressDecodeResult d1 = DecodeAddress(512, 1);
        Check(d0.home_die == 0 && d1.home_die == 1 &&
                  d0.stack_id == 0 && d1.stack_id == 1,
              "local_interleave uses current_die to disambiguate identical addresses");
        MemRouteResult local = ResolveMemRoute(1, 0, d1);
        Check(local.local && local.target_mem_tile == g_hbm_channels[1].mem_tile,
              "MemRouteTable resolves a local HBM attachment without C2C");
        Check(Throws([&] { ResolveMemRoute(0, 0, d1); }),
              "local_interleave explicitly rejects a remote-HBM route");
        Check(Throws([&] { (void)DecodeAddress(512); }),
              "local_interleave decode without current_die is rejected as ambiguous");
    }

    // ---- 8. global_interleave：全局 UMA 先跨 stack、再跨各 stack 内 channel ----
    {
        SetTopo(4, 4, 2, 1);
        auto hw = nlohmann::json::parse(R"JSON({
          "memory_system": {
            "topology":"distributed_hbm","cache_policy":"none",
            "profiles": {
              "one":{"generation":"HBM2","channels_per_stack":1,
                "pseudo_channels_per_channel":2,"data_rate_gbps_per_pin":2.0,
                "stack_bus_width_bits":64},
              "two":{"generation":"HBM2","channels_per_stack":2,
                "pseudo_channels_per_channel":2,"data_rate_gbps_per_pin":2.0,
                "stack_bus_width_bits":128}
            },
            "hbm_stacks":[
              {"stack_id":10,"compute_die_id":0,"profile":"one","side":"N",
               "start_idx":0,"phy_span_tiles":1,"capacity_bytes":1048576,
               "backend_granularity":"channel","channel_dram_config":"CFG0"},
              {"stack_id":20,"compute_die_id":1,"profile":"two","side":"N",
               "start_idx":0,"phy_span_tiles":2,"capacity_bytes":1048576,
               "backend_granularity":"channel","channel_dram_config":"CFG0"}
            ],
            "address_policy":{"mode":"global_interleave","home_ranges":[],
              "stack_interleave_bytes":64,"channel_interleave_bytes":64}
          }
        })JSON");
        for (auto &s : hw["memory_system"]["hbm_stacks"])
            s["channel_dram_config"] = kHbm2Cfg;
        Check(!Throws([&] { ParseAndBuild(hw); }),
              "global UMA accepts equal-capacity stacks with different channel counts");
        auto a0 = DecodeAddress(0), a1 = DecodeAddress(64);
        auto a2 = DecodeAddress(128), a3 = DecodeAddress(192);
        Check(a0.stack_id == 10 && a1.stack_id == 20 &&
                  a2.stack_id == 10 && a3.stack_id == 20,
              "global UMA gives equal address share to stacks, not flattened channels");
        Check(a1.channel_id == 0 && a3.channel_id == 1,
              "selected stack performs its own second-level channel interleave");
        Check(a0.home_die == 0 && a1.home_die == 1,
              "global UMA derives home die from the selected physical stack");
        Check(Throws([&] { (void)DecodeAddress(2ULL * 1048576); }),
              "global UMA rejects addresses at or beyond aggregate capacity");
    }

    // ---- 9. 聚合 MEM port 和 topology/cache schema 负例 ----
    {
        SetTopo(4, 4, 1, 1);
        auto hw = nlohmann::json::parse(R"JSON({
          "memory_system": {
            "topology":"distributed_hbm","cache_policy":"none",
            "profiles":{"p":{"generation":"HBM2","channels_per_stack":2,
              "pseudo_channels_per_channel":2,"data_rate_gbps_per_pin":2.0,
              "stack_bus_width_bits":128}},
            "hbm_stacks":[{"stack_id":0,"compute_die_id":0,"profile":"p",
              "side":"N","start_idx":1,"phy_span_tiles":1,
              "capacity_bytes":1048576,"backend_granularity":"channel",
              "port_granularity":"aggregated","channels_per_mem_port":2,
              "channel_dram_config":"CFG0"}],
            "address_policy":{"mode":"numa_local_interleave",
              "home_ranges":[{"die_id":0,"base":0,"size_bytes":1048576}],
              "stack_interleave_bytes":256,"channel_interleave_bytes":256}
          }
        })JSON");
        hw["memory_system"]["hbm_stacks"][0]["channel_dram_config"] = kHbm2Cfg;
        Check(!Throws([&] { ParseAndBuild(hw); }),
              "two channels may explicitly share one aggregated MEM port");
        Check(g_hbm_channels.size() == 2 &&
                  g_hbm_channels[0].mem_tile == g_hbm_channels[1].mem_tile,
              "aggregated port maps both channel backends to one logical tile");
        auto bad_agg = hw;
        bad_agg["memory_system"]["hbm_stacks"][0]["channels_per_mem_port"] = 3;
        Check(Throws([&] { ParseAndBuild(bad_agg); }),
              "aggregation wider than the physical channel count is rejected");

        auto bad_topology = nlohmann::json::parse(
            R"JSON({"memory_system":{"topology":"typo"}})JSON");
        Check(Throws([&] { ParseMemorySystem(bad_topology); }),
              "unknown memory topology is rejected");
        auto bad_cache = nlohmann::json::parse(
            R"JSON({"memory_system":{"topology":"distributed_hbm",
                     "cache_policy":"private_dcache"}})JSON");
        Check(Throws([&] { ParseMemorySystem(bad_cache); }),
              "unsupported distributed cache policy is rejected, not ignored");
    }

    // ---- 10. 旧配置（无 memory_system）与已注册回归保持不变 ----
    {
        SetTopo(4, 4, 1, 1);
        auto legacy = nlohmann::json::object();
        Check(!Throws([&] { ParseAndBuild(legacy); }),
              "absence of 'memory_system' key does not throw");
        Check(!g_memory_system_active,
              "memory_system stays inactive (legacy_private) when unconfigured");
        Check(g_hbm_stacks.empty() && g_hbm_channels.empty(),
              "no HBM stacks/channels are constructed under legacy_private");
        Check(!g_address_policy.active,
              "address_policy stays inactive under legacy_private");
    }

    std::cout << "distributed HBM R0 self-test: "
              << (g_fail == 0 ? "PASS" : "FAIL") << " (" << g_total
              << " checks)" << std::endl;
    return g_fail;
}
