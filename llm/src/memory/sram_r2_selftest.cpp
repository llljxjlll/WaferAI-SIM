#include "memory/sram/sram_access_unit.h"
#include "memory/sram/sram_selftest.h"

#include "macros/macros.h"
#include <iostream>

namespace {

struct R2Bench : sc_module {
    SC_HAS_PROCESS(R2Bench);
    sram::Storage storage;
    sram::RegionTable regions;
    sram::AccessUnit unit;
    int fails = 0;
    sc_event same_bank_start, same_bank_done_a, same_bank_done_b;
    sc_time same_bank_begin_a, same_bank_begin_b;
    sc_time same_bank_end_a, same_bank_end_b;
    sc_event different_bank_start, different_bank_done_a,
        different_bank_done_b;
    sc_time different_bank_begin_a, different_bank_begin_b;
    sc_time different_bank_end_a, different_bank_end_b;
    sc_event raw_write_start, raw_read_start, raw_write_admitted,
        raw_write_done, raw_read_done;
    std::vector<uint8_t> raw_read_payload;
    sc_time raw_write_end, raw_read_end;

    explicit R2Bench(sc_module_name name)
        : sc_module(name), storage(4096),
          regions(MakeConfig()), unit("access_unit", regions, storage) {
        SC_THREAD(Control);
        SC_THREAD(SameBankA);
        SC_THREAD(SameBankB);
        SC_THREAD(DifferentBankA);
        SC_THREAD(DifferentBankB);
        SC_THREAD(RawWriter);
        SC_THREAD(RawReader);
    }

    static sram::Config MakeConfig() {
        return sram::ParseConfig(
            {{"sram_size", 4096},
             {"sram",
              {{"bank_count", 2},
               {"bank_interleave_bytes", 64},
               {"read_base_latency_cycles", 1},
               {"write_base_latency_cycles", 1},
               {"queue_depth", 16},
               {"ports",
                {{"compute",
                  {{"read", {{"count", 2}, {"width_bits", 128}}},
                   {"write", {{"count", 2}, {"width_bits", 128}}}}},
                 {"dte",
                  {{"read", {{"count", 1}, {"width_bits", 128}}},
                   {"write", {{"count", 1}, {"width_bits", 128}}}}},
                 {"lsu",
                  {{"read", {{"count", 1}, {"width_bits", 128}}},
                   {"write", {{"count", 1}, {"width_bits", 128}}}}}}}}}});
    }

    void Check(bool condition, const char *message) {
        if (condition) return;
        ++fails;
        std::cerr << "[SRAM R2] FAIL: " << message << std::endl;
    }

    sram::Request WriteRequest(sram::Initiator initiator, uint64_t address,
                               uint8_t value) {
        sram::Request request;
        request.initiator = initiator;
        request.command = sram::Command::kWrite;
        request.address = address;
        request.size_bytes = 32;
        request.payload.assign(32, value);
        return request;
    }

    void SameBankA() {
        wait(same_bank_start);
        same_bank_begin_a = sc_time_stamp();
        unit.Access(WriteRequest(sram::Initiator::kCompute, 0, 0x11));
        same_bank_end_a = sc_time_stamp();
        same_bank_done_a.notify();
    }
    void SameBankB() {
        wait(same_bank_start);
        same_bank_begin_b = sc_time_stamp();
        unit.Access(WriteRequest(sram::Initiator::kCompute, 32, 0x22));
        same_bank_end_b = sc_time_stamp();
        same_bank_done_b.notify();
    }
    void DifferentBankA() {
        wait(different_bank_start);
        different_bank_begin_a = sc_time_stamp();
        unit.Access(WriteRequest(sram::Initiator::kCompute, 128, 0x33));
        different_bank_end_a = sc_time_stamp();
        different_bank_done_a.notify();
    }
    void DifferentBankB() {
        wait(different_bank_start);
        different_bank_begin_b = sc_time_stamp();
        unit.Access(WriteRequest(sram::Initiator::kCompute, 192, 0x44));
        different_bank_end_b = sc_time_stamp();
        different_bank_done_b.notify();
    }
    void RawWriter() {
        wait(raw_write_start);
        raw_write_admitted.notify(SC_ZERO_TIME);
        unit.Access(WriteRequest(sram::Initiator::kDte, 256, 0xa5));
        raw_write_end = sc_time_stamp();
        raw_write_done.notify();
    }
    void RawReader() {
        wait(raw_read_start);
        sram::Request request;
        request.initiator = sram::Initiator::kLsu;
        request.command = sram::Command::kRead;
        request.address = 256;
        request.size_bytes = 32;
        raw_read_payload = unit.Access(request).payload;
        raw_read_end = sc_time_stamp();
        raw_read_done.notify();
    }

    void Control() {
        same_bank_start.notify(SC_ZERO_TIME);
        wait(same_bank_done_a & same_bank_done_b);
        const sc_time same_elapsed =
            std::max(same_bank_end_a, same_bank_end_b) -
            std::min(same_bank_begin_a, same_bank_begin_b);
        Check(same_elapsed == sc_time(6 * CYCLE, SC_NS),
              "same bank accesses must serialize");

        different_bank_start.notify(SC_ZERO_TIME);
        wait(different_bank_done_a & different_bank_done_b);
        const sc_time different_elapsed =
            std::max(different_bank_end_a, different_bank_end_b) -
            std::min(different_bank_begin_a, different_bank_begin_b);
        Check(different_elapsed == sc_time(3 * CYCLE, SC_NS),
              "different banks with two ports must run in parallel");

        raw_write_start.notify(SC_ZERO_TIME);
        wait(raw_write_admitted);
        wait(SC_ZERO_TIME);
        raw_read_start.notify(SC_ZERO_TIME);
        wait(raw_write_done & raw_read_done);
        Check(raw_read_end > raw_write_end,
              "overlapping LSU read waits for DTE write commit");
        Check(raw_read_payload == std::vector<uint8_t>(32, 0xa5),
              "RAW consumer observes committed DTE payload");
        const auto &stats = unit.stats();
        const auto &lsu_read =
            stats.by_initiator_command[static_cast<size_t>(sram::Initiator::kLsu)]
                                      [static_cast<size_t>(sram::Command::kRead)];
        Check(lsu_read.hazard_stalls == 1,
              "RAW stall is attributed to the LSU read");
        Check(stats.bank_requests[0] != 0 && stats.bank_requests[1] != 0,
              "per-bank request statistics are recorded");
        Check(stats.banks[0].beats != 0 && stats.banks[0].bytes != 0 &&
                  stats.banks[0].service_cycles != 0 &&
                  stats.banks[1].beats != 0,
              "per-bank beat/byte/service statistics are recorded");
        const auto &compute_write_port =
            stats.ports[static_cast<size_t>(sram::Initiator::kCompute)]
                       [static_cast<size_t>(sram::Command::kWrite)];
        Check(compute_write_port.beats >= 8 &&
                  compute_write_port.bytes >= 128,
              "per-port beat/byte statistics reflect beat splitting");
        Check(unit.trace().size() == 6,
              "SRAM_queue trace covers every request");
        sc_stop();
    }
};

} // namespace

int RunSramR2SelfTest() {
    R2Bench bench("sram_r2_bench");
    sc_start();
    std::cout << "[SRAM R2] " << (bench.fails == 0 ? "PASS" : "FAIL")
              << " failures=" << bench.fails << std::endl;
    return bench.fails;
}
