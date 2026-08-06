#include "memory/core_lsu_unit.h"
#include "memory/sram/sram_selftest.h"
#include "prims/norm_prims.h"

#include <iostream>
#include <map>

namespace {

class FakeHbm : public sram::HbmByteTransport {
  public:
    std::map<uint64_t, uint8_t> bytes;
    sc_time latency = sc_time(5, SC_NS);
    uint64_t read_bytes = 0;
    uint64_t write_bytes = 0;

    std::vector<uint8_t> Read(uint64_t address,
                              uint64_t size_bytes) override {
        wait(latency);
        std::vector<uint8_t> payload(size_bytes);
        for (uint64_t i = 0; i < size_bytes; ++i)
            payload[i] = bytes[address + i];
        read_bytes += size_bytes;
        return payload;
    }

    void Write(uint64_t address, const std::vector<uint8_t> &payload,
               const std::vector<uint8_t> &byte_enable) override {
        wait(latency);
        for (size_t i = 0; i < payload.size(); ++i) {
            if (!byte_enable.empty() && byte_enable[i] == 0) continue;
            bytes[address + i] = payload[i];
        }
        write_bytes += payload.size();
    }

    void Seed(uint64_t address, const std::vector<uint8_t> &payload) {
        for (size_t i = 0; i < payload.size(); ++i)
            bytes[address + i] = payload[i];
    }

    std::vector<uint8_t> Peek(uint64_t address, uint64_t size_bytes) const {
        std::vector<uint8_t> payload(size_bytes);
        for (uint64_t i = 0; i < size_bytes; ++i) {
            const auto it = bytes.find(address + i);
            payload[i] = it == bytes.end() ? 0 : it->second;
        }
        return payload;
    }
};

struct R3Bench : sc_module {
    SC_HAS_PROCESS(R3Bench);
    sram::Storage storage;
    sram::RegionTable regions;
    sram::AccessUnit access;
    FakeHbm hbm;
    sram::CoreLsuUnit lsu;
    int fails = 0;

    explicit R3Bench(sc_module_name name)
        : sc_module(name), storage(4096), regions(MakeConfig()),
          access("access", regions, storage),
          lsu("lsu", regions, access, hbm, 8, 2, 2) {
        SC_THREAD(Run);
    }

    static sram::Config MakeConfig() {
        return sram::ParseConfig(
            {{"sram_size", 4096},
             {"sram",
              {{"bank_count", 4},
               {"lsu",
                {{"queue_depth", 8},
                 {"max_outstanding", 2},
                 {"issue_latency_ns", 2}}},
               {"bank_interleave_bytes", 64},
               {"regions",
                {{{"name", "input"},
                  {"base_bytes", 0},
                  {"size_bytes", 2048},
                  {"allocator", "block"},
                  {"access", {"compute", "lsu"}}},
                 {{"name", "output"},
                  {"base_bytes", 2048},
                  {"size_bytes", 2048},
                  {"allocator", "fixed"},
                  {"access", {"compute", "lsu"}}}}}}}});
    }

    void Check(bool condition, const char *message) {
        if (condition) return;
        ++fails;
        std::cerr << "[SRAM R3] FAIL: " << message << std::endl;
    }

    void Run() {
        std::vector<uint8_t> pattern(73);
        for (size_t i = 0; i < pattern.size(); ++i)
            pattern[i] = static_cast<uint8_t>(0x31 + 7 * i);
        std::vector<uint8_t> pattern2 = pattern;
        for (auto &byte : pattern2) byte ^= 0x27;
        hbm.Seed(0x1000, pattern);
        hbm.Seed(0x1800, pattern2);

        const sc_time issue_time = sc_time_stamp();
        const auto load =
            lsu.IssueLoadRegion(0x1000, "input", 128, pattern.size());
        Check(sc_time_stamp() == issue_time + sc_time(2, SC_NS),
              "LSU issue latency is modeled independently");
        const auto load2 =
            lsu.IssueLoadRegion(0x1800, "input", 512, pattern2.size());
        Check(!lsu.Poll(load) && !lsu.Poll(load2),
              "two fresh LSU load tokens are incomplete");
        std::vector<sram::LsuToken> queued{load, load2};
        for (uint64_t index = 0; index < 6; ++index) {
            const uint64_t hbm_addr = 0x1c00 + index * 0x100;
            const uint64_t sram_offset = 768 + index * 128;
            std::vector<uint8_t> tail(16,
                static_cast<uint8_t>(0x80 + index));
            hbm.Seed(hbm_addr, tail);
            queued.push_back(lsu.IssueLoadRegion(
                hbm_addr, "input", sram_offset, tail.size()));
        }
        Check(queued.size() == 8,
              "LSU queue accepts queue_depth descriptors while two run");
        bool queue_rejected = false;
        try {
            (void)lsu.IssueLoadRegion(0x3000, "input", 1600, 16);
        } catch (const std::runtime_error &) {
            queue_rejected = true;
        }
        Check(queue_rejected,
              "LSU rejects only after queue_depth descriptors are retained");
        for (const auto token : queued) lsu.Wait(token);
        Check(sc_time_stamp() > issue_time + hbm.latency,
              "loads complete only after HBM and SRAM commit");

        sram::Request compute_read;
        compute_read.initiator = sram::Initiator::kCompute;
        compute_read.command = sram::Command::kRead;
        compute_read.address = 128;
        compute_read.size_bytes = pattern.size();
        Check(access.Access(compute_read).payload == pattern,
              "HBM payload is visible in SRAM after token completion");
        compute_read.address = 512;
        Check(access.Access(compute_read).payload == pattern2,
              "second concurrent LSU load commits independently");

        std::vector<uint8_t> transformed = pattern;
        for (auto &byte : transformed) byte ^= 0x5a;
        sram::Request compute_write;
        compute_write.initiator = sram::Initiator::kCompute;
        compute_write.command = sram::Command::kWrite;
        compute_write.address = 2048;
        compute_write.size_bytes = transformed.size();
        compute_write.payload = transformed;
        access.Access(compute_write);

        const auto store =
            lsu.IssueStoreRegion("output", 0, 0x2000, transformed.size());
        lsu.Wait(store);
        Check(hbm.Peek(0x2000, transformed.size()) == transformed,
              "SRAM non-zero payload is stored back to HBM");

        auto invalid =
            lsu.IssueStoreRegion("input", 1000, 0x3000, 16);
        bool invalid_failed = false;
        try {
            lsu.Wait(invalid);
        } catch (const std::runtime_error &) {
            invalid_failed = true;
        }
        Check(invalid_failed, "store from invalid SRAM bytes fails token");

        const auto &stats = lsu.stats();
        Check(stats.hbm_read_bytes == 2 * pattern.size() + 6 * 16 &&
                  stats.sram_write_bytes == 2 * pattern.size() + 6 * 16 &&
                  stats.hbm_write_bytes == transformed.size() &&
                  stats.sram_read_bytes == transformed.size(),
              "LSU HBM/SRAM byte statistics balance");
        Check(hbm.read_bytes == 2 * pattern.size() + 6 * 16 &&
                  hbm.write_bytes == transformed.size(),
              "transport byte statistics match completed descriptors");
        Check(stats.peak_outstanding == 8 && stats.peak_running == 2,
              "LSU queue capacity and two-worker concurrency are distinct");
        Check(stats.issue_latency_ns == 20,
              "LSU issue latency statistics cover every accepted descriptor");
        Check(lsu.trace().size() == 10,
              "LSU_hbm trace covers loads, store and failed request");
        Check(regions.config().lsu_queue_depth == 8 &&
                  regions.config().lsu_max_outstanding == 2 &&
                  regions.config().lsu_issue_latency_ns == 2,
              "LSU frozen configuration is parsed");
        Check(lsu.OutstandingCount() == 0,
              "wait consumes completed and failed tokens");
        sc_stop();
    }
};

} // namespace

int RunSramR3SelfTest() {
    int wire_fails = 0;
    try {
        Lsu_mem_prim encoded;
        encoded.parseJson({{"op", "issue"},
                           {"token", 31},
                           {"direction", "HBM_TO_SRAM"},
                           {"hbm_addr", 0x12345678ULL},
                           {"sram_region", "double_a"},
                           {"sram_offset", 256},
                           {"size_bytes", 73}});
        Lsu_mem_prim decoded;
        decoded.deserialize(encoded.serialize());
        if (decoded.op != encoded.op || decoded.token != encoded.token ||
            decoded.direction != encoded.direction ||
            decoded.hbm_addr != encoded.hbm_addr ||
            decoded.sram_region != encoded.sram_region ||
            decoded.sram_offset != encoded.sram_offset ||
            decoded.size_bytes != encoded.size_bytes) {
            ++wire_fails;
            std::cerr << "[SRAM R3] FAIL: Lsu_mem wire round-trip"
                      << std::endl;
        }
    } catch (const std::exception &error) {
        ++wire_fails;
        std::cerr << "[SRAM R3] FAIL: Lsu_mem wire threw: "
                  << error.what() << std::endl;
    }

    R3Bench bench("sram_r3_bench");
    sc_start();
    const int failures = wire_fails + bench.fails;
    std::cout << "[SRAM R3] " << (failures == 0 ? "PASS" : "FAIL")
              << " failures=" << failures << std::endl;
    return failures;
}
