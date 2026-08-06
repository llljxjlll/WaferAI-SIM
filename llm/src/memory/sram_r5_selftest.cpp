#include "dte/dte_async.h"
#include "dte/dte_memory_bridge.h"
#include "memory/core_lsu_unit.h"
#include "memory/sram/compute_timeline.h"
#include "memory/sram/sram_selftest.h"

#include "macros/macros.h"
#include <iostream>
#include <map>

namespace {

class R5FakeHbm : public sram::HbmByteTransport {
  public:
    std::map<uint64_t, uint8_t> bytes;
    sc_time latency = sc_time(10, SC_NS);

    std::vector<uint8_t> Read(uint64_t address,
                              uint64_t size_bytes) override {
        wait(latency);
        std::vector<uint8_t> result(size_bytes);
        for (uint64_t i = 0; i < size_bytes; ++i)
            result[i] = bytes[address + i];
        return result;
    }

    void Write(uint64_t address, const std::vector<uint8_t> &payload,
               const std::vector<uint8_t> &byte_enable) override {
        wait(latency);
        for (size_t i = 0; i < payload.size(); ++i) {
            if (!byte_enable.empty() && byte_enable[i] == 0) continue;
            bytes[address + i] = payload[i];
        }
    }

    void Seed(uint64_t address, const std::vector<uint8_t> &payload) {
        for (size_t i = 0; i < payload.size(); ++i)
            bytes[address + i] = payload[i];
    }

    std::vector<uint8_t> Peek(uint64_t address, size_t size) const {
        std::vector<uint8_t> result(size);
        for (size_t i = 0; i < size; ++i) {
            const auto it = bytes.find(address + i);
            result[i] = it == bytes.end() ? 0 : it->second;
        }
        return result;
    }
};

struct R5Bench : sc_module {
    SC_HAS_PROCESS(R5Bench);
    static constexpr uint64_t kTileBytes = 32;
    static constexpr uint64_t kTileCount = 4;
    static constexpr uint64_t kComputeCycles = 10;

    sram::Storage storage;
    sram::RegionTable regions;
    sram::AccessUnit access;
    R5FakeHbm hbm;
    sram::CoreLsuUnit lsu;
    DteMemoryBridge dte_bridge;
    DTEUnit dte;
    DteAsyncTracker dte_tracker;
    sram::ComputeTimeline timeline;
    int fails = 0;
    bool finished = false;

    explicit R5Bench(sc_module_name name)
        : sc_module(name), storage(128), regions(MakeConfig()),
          access("access", regions, storage),
          lsu("lsu", regions, access, hbm, 8),
          dte_bridge("dte_bridge", regions, access, hbm),
          dte("dte", MakeDteConfig(), 0, nullptr),
          dte_tracker("dte_tracker", dte, {}, 0, nullptr), timeline(access) {
        dte_tracker.BindMemoryBridge(&dte_bridge);
        SC_THREAD(Run);
    }

    static sram::Config MakeConfig() {
        return sram::ParseConfig(
            {{"sram_size", 128},
             {"sram",
              {{"bank_count", 2},
               {"bank_interleave_bytes", 64},
               {"manual_memory_schedule", true},
               {"regions",
                {{{"name", "double_a"},
                  {"base_bytes", 0},
                  {"size_bytes", 64},
                  {"allocator", "fixed"},
                  {"access", {"compute", "lsu", "dte"}}},
                 {{"name", "double_b"},
                  {"base_bytes", 64},
                  {"size_bytes", 64},
                  {"allocator", "fixed"},
                  {"access", {"compute", "lsu", "dte"}}}}}}}});
    }

    static DTEConfig MakeDteConfig() {
        DTEConfig config;
        config.channel_count = 1;
        config.bit_width_bits = 256;
        config.gamma_cycles = 0;
        config.tau_launch_cycles = 0;
        config.fine_grained_resources = true;
        config.command_slots_per_channel = 2;
        config.pending_queue_depth = 8;
        config.spm_read_width_bits = 256;
        config.spm_write_width_bits = 256;
        config.axi_read_width_bits = 256;
        config.axi_write_width_bits = 256;
        return config;
    }

    void Check(bool condition, const char *message) {
        if (condition) return;
        ++fails;
        std::cerr << "[SRAM R5] FAIL: " << message << std::endl;
    }

    std::vector<uint8_t> Pattern(uint8_t seed) const {
        std::vector<uint8_t> result(kTileBytes);
        for (size_t i = 0; i < result.size(); ++i)
            result[i] = static_cast<uint8_t>(seed + 3 * i);
        return result;
    }

    sram::ResolvedRange Slot(uint64_t tile) {
        return regions.Resolve((tile & 1) ? "double_b" : "double_a", 0,
                               kTileBytes, sram::Initiator::kCompute,
                               sram::Command::kRead);
    }

    void SeedTiles(uint64_t hbm_base, uint8_t seed_base) {
        for (uint64_t tile = 0; tile < kTileCount; ++tile)
            hbm.Seed(hbm_base + tile * kTileBytes,
                     Pattern(static_cast<uint8_t>(seed_base + tile)));
    }

    void Run() {
        const uint64_t blocking_hbm = 0x1000;
        SeedTiles(blocking_hbm, 0x10);
        const sc_time blocking_begin = sc_time_stamp();
        for (uint64_t tile = 0; tile < kTileCount; ++tile) {
            const auto slot = Slot(tile);
            lsu.Load(blocking_hbm + tile * kTileBytes, slot.address,
                     kTileBytes);
            const auto result = timeline.RunTile(slot, kComputeCycles);
            Check(result == Pattern(static_cast<uint8_t>(0x10 + tile)),
                  "blocking schedule preserves tile payload");
        }
        const sc_time blocking_time = sc_time_stamp() - blocking_begin;

        const uint64_t pipelined_hbm = 0x2000;
        SeedTiles(pipelined_hbm, 0x40);
        const sc_time pipelined_begin = sc_time_stamp();
        auto current = lsu.IssueLoad(pipelined_hbm, Slot(0).address,
                                     kTileBytes);
        for (uint64_t tile = 0; tile < kTileCount; ++tile) {
            lsu.Wait(current);
            sram::LsuToken next = 0;
            if (tile + 1 < kTileCount) {
                const auto next_slot = Slot(tile + 1);
                next = lsu.IssueLoad(
                    pipelined_hbm + (tile + 1) * kTileBytes,
                    next_slot.address, kTileBytes);
            }
            const auto result = timeline.RunTile(Slot(tile), kComputeCycles);
            Check(result == Pattern(static_cast<uint8_t>(0x40 + tile)),
                  "double-buffer schedule preserves tile payload");
            current = next;
        }
        const sc_time pipelined_time = sc_time_stamp() - pipelined_begin;

        const sc_time load_time =
            hbm.latency + sc_time(3 * CYCLE, SC_NS);
        const sc_time compute_time =
            sc_time((3 + kComputeCycles) * CYCLE, SC_NS);
        const sc_time expected_blocking =
            kTileCount * (load_time + compute_time);
        const sc_time expected_pipeline =
            load_time + kTileCount * compute_time;
        Check(blocking_time == expected_blocking,
              "blocking schedule matches sum(load+compute) oracle");
        Check(pipelined_time == expected_pipeline,
              "double buffer matches warmup + N*max(load,compute) oracle");
        Check(pipelined_time < blocking_time,
              "manual double buffering overlaps memory and compute");
        Check(timeline.trace().size() == 2 * kTileCount,
              "Compute_tile trace covers blocking and pipelined tiles");
        Check(timeline.stats().tiles == 2 * kTileCount &&
                  timeline.stats().sram_read_bytes ==
                      2 * kTileCount * kTileBytes,
              "ComputeTimeline accounts all tile reads");
        Check(lsu.OutstandingCount() == 0,
              "manual schedule consumes every LSU token");

        Check(regions.config().manual_memory_schedule,
              "manual_memory_schedule is parsed and visible per core");

        const uint64_t dte_hbm = 0x3000;
        const auto dte_pattern = Pattern(0x70);
        hbm.Seed(dte_hbm, dte_pattern);
        dte_tracker.IssueToken(51, kTileBytes * 8, DteDir::DRAM_TO_SPM,
                               Slot(0).address, kTileBytes,
                               DTE_ASYNC_INVALID_REMOTE_PEER, dte_hbm, 0);
        const auto dte_result = timeline.RunTile(Slot(0), kComputeCycles);
        Check(dte_result == dte_pattern,
              "DTE load overlaps callable compute and its lease gates RAW");
        dte_tracker.WaitToken(51);

        const uint64_t lsu_store_hbm = 0x4000;
        lsu.Store(Slot(0).address, lsu_store_hbm, kTileBytes);
        Check(hbm.Peek(lsu_store_hbm, kTileBytes) == dte_pattern,
              "load/compute/store schedule writes slot through LSU");

        const uint64_t dte_store_hbm = 0x5000;
        dte_tracker.IssueToken(52, kTileBytes * 8, DteDir::SPM_TO_DRAM,
                               Slot(0).address, kTileBytes,
                               DTE_ASYNC_INVALID_REMOTE_PEER, dte_store_hbm,
                               0);
        dte_tracker.WaitToken(52);
        Check(hbm.Peek(dte_store_hbm, kTileBytes) == dte_pattern,
              "DTE store is callable in the manual tile schedule");

        const uint64_t hazard_hbm = 0x6000;
        hbm.Seed(hazard_hbm, Pattern(0x90));
        hbm.Seed(hazard_hbm + kTileBytes, Pattern(0xa0));
        const sc_time hazard_begin = sc_time_stamp();
        auto overwrite_a =
            lsu.IssueLoad(hazard_hbm, Slot(1).address, kTileBytes);
        auto overwrite_b = lsu.IssueLoad(hazard_hbm + kTileBytes,
                                         Slot(1).address, kTileBytes);
        lsu.Wait(overwrite_a);
        lsu.Wait(overwrite_b);
        Check(sc_time_stamp() - hazard_begin >= 2 * load_time,
              "single-slot WAW is serialized instead of overwritten");
        Check(storage.Read(Slot(1).address, kTileBytes) == Pattern(0xa0),
              "serialized single-slot overwrite commits in issue order");
        finished = true;
        sc_stop();
    }
};

} // namespace

int RunSramR5SelfTest() {
    R5Bench bench("sram_r5_bench");
    sc_start(sc_time(10, SC_US));
    if (!bench.finished) {
        ++bench.fails;
        std::cerr << "[SRAM R5] FAIL: test thread did not reach sc_stop"
                  << std::endl;
    }
    std::cout << "[SRAM R5] " << (bench.fails == 0 ? "PASS" : "FAIL")
              << " failures=" << bench.fails << std::endl;
    return bench.fails;
}
