#include "dte/dte_async.h"
#include "dte/dte_memory_bridge.h"
#include "memory/sram/sram_selftest.h"
#include "prims/norm_prims.h"

#include <iostream>
#include <map>

namespace {

class R4FakeHbm : public sram::HbmByteTransport {
  public:
    std::map<uint64_t, uint8_t> bytes;
    sc_time latency = sc_time(9, SC_NS);

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

struct R4Bench : sc_module {
    SC_HAS_PROCESS(R4Bench);
    sram::Storage storage;
    sram::RegionTable regions;
    sram::AccessUnit access;
    R4FakeHbm hbm;
    DteMemoryBridge bridge;
    DTEUnit dte;
    DteAsyncTracker tracker;
    DteMemoryBridge rollback_bridge;
    DTEUnit rollback_dte;
    DteAsyncTracker rollback_tracker;
    DTEUnit compatibility_dte;
    DteAsyncTracker compatibility_tracker;
    int fails = 0;

    explicit R4Bench(sc_module_name name)
        : sc_module(name), storage(4096), regions(MakeSramConfig()),
          access("access", regions, storage),
          bridge("bridge", regions, access, hbm),
          dte("dte", MakeDteConfig(), 0, nullptr),
          tracker("tracker", dte, {}, 0, nullptr),
          rollback_bridge("rollback_bridge", regions, access, hbm, 1, 1),
          rollback_dte("rollback_dte", MakeDteConfig(), 2, nullptr),
          rollback_tracker("rollback_tracker", rollback_dte, {}, 2, nullptr),
          compatibility_dte("compatibility_dte", MakeDteConfig(), 1, nullptr),
          compatibility_tracker("compatibility_tracker", compatibility_dte,
                                {}, 1, nullptr) {
        tracker.BindMemoryBridge(&bridge);
        rollback_tracker.BindMemoryBridge(&rollback_bridge);
        SC_THREAD(Run);
    }

    static sram::Config MakeSramConfig() {
        return sram::ParseConfig(
            {{"sram_size", 4096},
             {"sram",
              {{"bank_count", 4},
               {"bank_interleave_bytes", 64},
               {"regions",
                {{{"name", "double_a"},
                  {"base_bytes", 0},
                  {"size_bytes", 2048},
                  {"allocator", "fixed"},
                  {"access", {"compute", "dte", "lsu"}}},
                 {{"name", "double_b"},
                  {"base_bytes", 2048},
                  {"size_bytes", 2048},
                  {"allocator", "fixed"},
                  {"access", {"compute", "dte", "lsu"}}}}}}}});
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
        std::cerr << "[SRAM R4] FAIL: " << message << std::endl;
    }

    void Run() {
        std::vector<uint8_t> pattern(32);
        for (size_t i = 0; i < pattern.size(); ++i)
            pattern[i] = static_cast<uint8_t>(0x80 + i);
        hbm.Seed(0x1000, pattern);

        const sc_time begin = sc_time_stamp();
        tracker.IssueToken(11, pattern.size() * 8, DteDir::DRAM_TO_SPM,
                           128, pattern.size(),
                           DTE_ASYNC_INVALID_REMOTE_PEER, 0x1000, 0);
        Check(!tracker.PollToken(11),
              "DTE real-memory token starts incomplete");
        sram::Request immediate_read;
        immediate_read.initiator = sram::Initiator::kLsu;
        immediate_read.command = sram::Command::kRead;
        immediate_read.address = 128;
        immediate_read.size_bytes = pattern.size();
        Check(access.Access(immediate_read).payload == pattern,
              "descriptor-lifetime lease blocks LSU RAW until DTE commit");
        tracker.WaitToken(11);
        Check(sc_time_stamp() >= begin + hbm.latency,
              "DTE token waits for real HBM data plane");
        Check(storage.Read(128, pattern.size()) == pattern,
              "DTE DRAM_TO_SPM commits HBM payload to SRAM");

        std::vector<uint8_t> transformed = pattern;
        for (auto &byte : transformed) byte ^= 0x3c;
        sram::Request write;
        write.initiator = sram::Initiator::kCompute;
        write.command = sram::Command::kWrite;
        write.address = 2048;
        write.size_bytes = transformed.size();
        write.payload = transformed;
        access.Access(write);

        tracker.IssueToken(12, transformed.size() * 8,
                           DteDir::SPM_TO_DRAM, 2048, transformed.size(),
                           DTE_ASYNC_INVALID_REMOTE_PEER, 0x2000, 0);
        tracker.WaitToken(12);
        Check(hbm.Peek(0x2000, transformed.size()) == transformed,
              "DTE SPM_TO_DRAM writes SRAM payload to HBM");

        tracker.IssueToken(13, transformed.size() * 8,
                           DteDir::SPM_TO_SPM, 2048, transformed.size(),
                           DTE_ASYNC_INVALID_REMOTE_PEER, 1024, 0);
        tracker.WaitToken(13);
        Check(storage.Read(1024, transformed.size()) == transformed,
              "DTE SPM_TO_SPM copies real bytes between SRAM regions");

        bool size_mismatch_rejected = false;
        try {
            (void)tracker.IssueToken(14, 31 * 8, DteDir::DRAM_TO_SPM,
                                     768, 32,
                                     DTE_ASYNC_INVALID_REMOTE_PEER,
                                     0x3800, 0);
        } catch (const std::invalid_argument &) {
            size_mismatch_rejected = true;
        }
        Check(size_mismatch_rejected,
              "real-memory DTE requires payload_bytes == spm_size");

        hbm.Seed(0x4000, pattern);
        hbm.Seed(0x5000, transformed);
        tracker.IssueToken(31, pattern.size() * 8,
                           DteDir::DRAM_TO_SPM, 256, pattern.size(),
                           DTE_ASYNC_INVALID_REMOTE_PEER, 0x4000, 0);
        tracker.IssueToken(32, transformed.size() * 8,
                           DteDir::DRAM_TO_SPM, 512, transformed.size(),
                           DTE_ASYNC_INVALID_REMOTE_PEER, 0x5000, 0);
        tracker.WaitToken(31);
        tracker.WaitToken(32);
        Check(storage.Read(256, pattern.size()) == pattern &&
                  storage.Read(512, transformed.size()) == transformed,
              "two DTE memory workers commit independent transfers");

        hbm.Seed(0x6000, pattern);
        rollback_tracker.IssueToken(
            41, pattern.size() * 8, DteDir::DRAM_TO_SPM, 640,
            pattern.size(), DTE_ASYNC_INVALID_REMOTE_PEER, 0x6000, 0);
        bool bridge_full_rejected = false;
        try {
            (void)rollback_tracker.IssueToken(
                42, pattern.size() * 8, DteDir::DRAM_TO_SPM, 704,
                pattern.size(), DTE_ASYNC_INVALID_REMOTE_PEER, 0x7000, 0);
        } catch (const std::runtime_error &) {
            bridge_full_rejected = true;
        }
        Check(bridge_full_rejected &&
                  rollback_tracker.OutstandingCount() == 1 &&
                  rollback_bridge.OutstandingCount() == 1,
              "bridge-full failure occurs before a second control token or "
              "memory record is created");
        rollback_tracker.WaitToken(41);
        Check(rollback_tracker.OutstandingCount() == 0 &&
                  rollback_bridge.OutstandingCount() == 0 &&
                  rollback_dte.PendingCount() == 0 &&
                  rollback_dte.ActiveCount() == 0,
              "bridge-full rollback leaves no record or DTE credit leak");

        compatibility_tracker.IssueToken(
            21, pattern.size() * 8, DteDir::DRAM_TO_SPM, 1536,
            pattern.size(), DTE_ASYNC_INVALID_REMOTE_PEER, 0x3000, 0);
        compatibility_tracker.WaitToken(21);
        Check(!storage.IsValid(1536, pattern.size()),
              "unbound tracker preserves endpoint-only V4 behavior");

        const auto &stats = bridge.stats();
        Check(stats.hbm_read_bytes == 3 * pattern.size() &&
                  stats.sram_write_bytes == 4 * pattern.size() &&
                  stats.sram_read_bytes == 2 * transformed.size() &&
                  stats.hbm_write_bytes == transformed.size() &&
                  stats.sram_copy_bytes == transformed.size(),
              "DTE bridge HBM/SRAM/copy byte statistics balance");
        Check(stats.peak_running == 2 && stats.peak_outstanding >= 2,
              "DTE memory bridge exposes configured concurrency");
        Check(bridge.trace().size() == 5,
              "DTE_mem_commit trace covers every real transfer");
        Check(bridge.OutstandingCount() == 0 &&
                  tracker.OutstandingCount() == 0,
              "DTE wait releases control and data-plane tokens");
        sc_stop();
    }
};

} // namespace

int RunSramR4SelfTest() {
    int wire_fails = 0;
    try {
        Dte_async_prim encoded;
        encoded.parseJson({{"op", "issue"},
                           {"token", 17},
                           {"direction", "DRAM_TO_SPM"},
                           {"payload_bits", 256},
                           {"hbm_addr", 0x12340000ULL},
                           {"sram_region", "double_a"},
                           {"sram_offset", 64},
                           {"spm_size", 32}});
        Dte_async_prim decoded;
        decoded.deserialize(encoded.serialize());
        if (decoded.remote_addr != encoded.remote_addr ||
            decoded.sram_region != encoded.sram_region ||
            decoded.sram_offset != encoded.sram_offset ||
            decoded.spm_size != encoded.spm_size) {
            ++wire_fails;
            std::cerr << "[SRAM R4] FAIL: Dte_async region wire round-trip"
                      << std::endl;
        }
    } catch (const std::exception &error) {
        ++wire_fails;
        std::cerr << "[SRAM R4] FAIL: Dte_async wire threw: "
                  << error.what() << std::endl;
    }

    R4Bench bench("sram_r4_bench");
    sc_start();
    const int failures = wire_fails + bench.fails;
    std::cout << "[SRAM R4] " << (failures == 0 ? "PASS" : "FAIL")
              << " failures=" << failures << std::endl;
    return failures;
}
