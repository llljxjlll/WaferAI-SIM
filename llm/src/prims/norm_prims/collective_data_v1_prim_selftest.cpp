#include "prims/collective_data_v1_prim.h"
#include "prims/collective_data_v1_prim_selftest.h"

#include "common/config.h"
#include "defs/global.h"
#include "memory/sram/sram_access_unit.h"
#include "memory/sram/sram_region.h"
#include "memory/sram/sram_storage.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> Le32(uint32_t value) {
    return {static_cast<uint8_t>(value),
            static_cast<uint8_t>(value >> 8),
            static_cast<uint8_t>(value >> 16),
            static_cast<uint8_t>(value >> 24)};
}

std::vector<uint8_t> Le64(uint64_t value) {
    std::vector<uint8_t> bytes(8);
    for (size_t index = 0; index < bytes.size(); ++index)
        bytes[index] = static_cast<uint8_t>(value >> (8 * index));
    return bytes;
}

void Append(std::vector<uint8_t> *target,
            const std::vector<uint8_t> &bytes) {
    target->insert(target->end(), bytes.begin(), bytes.end());
}

sram::Config TestConfig() {
    sram::Config config;
    config.capacity_bytes = 512;
    config.allocation_alignment_bytes = 1;
    config.bank_count = 4;
    config.bank_interleave_bytes = 16;
    config.read_base_latency_cycles = 1;
    config.write_base_latency_cycles = 1;
    config.queue_depth = 16;
    config.real_data_path = true;
    sram::RegionConfig region;
    region.name = "all";
    region.base_bytes = 0;
    region.size_bytes = config.capacity_bytes;
    region.allocator = sram::AllocatorKind::kFixed;
    region.access = {sram::Initiator::kCompute};
    config.regions.push_back(region);
    return config;
}

size_t InitiatorIndex(sram::Initiator value) {
    return static_cast<size_t>(value);
}

size_t CommandIndex(sram::Command value) {
    return static_cast<size_t>(value);
}

struct CollectiveDataV1PrimBench final : sc_module {
    SC_HAS_PROCESS(CollectiveDataV1PrimBench);

    sram::Storage storage;
    sram::RegionTable regions;
    sram::AccessUnit access;
    int failures = 0;
    int checks = 0;
    CoreHWConfig *hardware = nullptr;

    explicit CollectiveDataV1PrimBench(sc_module_name name)
        : sc_module(name), storage(512, true), regions(TestConfig()),
          access("collective_data_v1_access", regions, storage) {
        hardware = new CoreHWConfig(
            0, nullptr, nullptr, new VectorConfig(2, 1), "", 0, 128);
        g_core_hw_config.emplace_back(0, hardware);
        SC_THREAD(Run);
    }

    ~CollectiveDataV1PrimBench() override {
        const auto found = std::find_if(
            g_core_hw_config.begin(), g_core_hw_config.end(),
            [&](const auto &entry) { return entry.second == hardware; });
        if (found != g_core_hw_config.end()) g_core_hw_config.erase(found);
        delete hardware;
    }

    void Check(bool condition, const std::string &name) {
        ++checks;
        if (condition) return;
        ++failures;
        std::cerr << "[COLLECTIVE DATA V1 PRIM] FAIL: " << name << '\n';
    }

    template <class Exception = std::exception, class F>
    bool Rejects(F &&fn) {
        try {
            fn();
        } catch (const Exception &) {
            return true;
        } catch (...) {
        }
        return false;
    }

    const sram::AccessCounter &Counter(sram::Command command) const {
        return access.stats().by_initiator_command
            [InitiatorIndex(sram::Initiator::kCompute)]
            [CommandIndex(command)];
    }

    void Seed(uint64_t address, const std::vector<uint8_t> &bytes) {
        storage.Write(address, bytes);
    }

    void SeedSentinel(uint64_t destination, uint64_t length,
                      uint8_t value) {
        Seed(destination - 1,
             std::vector<uint8_t>(static_cast<size_t>(length + 2), value));
    }

    bool SentinelPreserved(uint64_t destination, uint64_t length,
                           uint8_t value) const {
        return storage.Read(destination - 1, 1) ==
                   std::vector<uint8_t>({value}) &&
               storage.Read(destination + length, 1) ==
                   std::vector<uint8_t>({value});
    }

    TaskCoreContext Context() {
        static int legacy_sram_address = 0;
        TaskCoreContext context(
            nullptr, nullptr, nullptr, nullptr, &legacy_sram_address,
            nullptr, nullptr, nullptr, nullptr, uint64_t{0}, unsigned{0});
        context.cid = 0;
        context.sram_regions = &regions;
        context.sram_access = &access;
        context.sram_storage = &storage;
        return context;
    }

    void TestLocalCopy(TaskCoreContext &context) {
        const std::vector<uint8_t> pattern = {0, 1, 7, 0x80, 0xff, 4, 9};
        Seed(0x20, pattern);
        SeedSentinel(0x80, pattern.size(), 0xa5);
        const auto reads = Counter(sram::Command::kRead);
        const auto writes = Counter(sram::Command::kWrite);

        Collective_data_v1_prim prim;
        prim.mode = CollectiveDataV1PrimMode::LOCAL_COPY;
        prim.key = {1, 2, 3};
        prim.phase_id = 4;
        prim.source_address_bytes = 0x20;
        prim.destination_address_bytes = 0x80;
        prim.length_bytes = pattern.size();
        prim.input_count = 1;
        prim.dtype = CollDType::UINT8;
        prim.reduce_op = CollReduceOp::NONE;
        Check(prim.taskCoreDefault(context) == 0,
              "LOCAL_COPY has no vector delay");
        Check(storage.Read(0x80, pattern.size()) == pattern &&
                  SentinelPreserved(0x80, pattern.size(), 0xa5),
              "LOCAL_COPY writes exact L and preserves sentinels");
        Check(Counter(sram::Command::kRead).requests ==
                      reads.requests + 1 &&
                  Counter(sram::Command::kRead).bytes ==
                      reads.bytes + pattern.size() &&
                  Counter(sram::Command::kWrite).requests ==
                      writes.requests + 1 &&
                  Counter(sram::Command::kWrite).bytes ==
                      writes.bytes + pattern.size(),
              "LOCAL_COPY uses exactly one AccessUnit read and write");
    }

    void TestNegativeMax(TaskCoreContext &context) {
        std::vector<uint8_t> source;
        Append(&source, Le32(UINT32_C(0xfffffffb)));
        Append(&source, Le32(UINT32_C(0xfffffff6)));
        Append(&source, Le32(UINT32_C(0xfffffffe)));
        Append(&source, Le32(UINT32_C(0xffffffec)));
        Append(&source, Le32(UINT32_C(0xfffffff9)));
        Append(&source, Le32(4));
        Seed(0x100, source);
        SeedSentinel(0x180, 8, 0x5a);
        const auto reads = Counter(sram::Command::kRead);
        const auto writes = Counter(sram::Command::kWrite);

        Collective_data_v1_prim prim;
        prim.mode = CollectiveDataV1PrimMode::REDUCE;
        prim.key = {2, 3, 4};
        prim.phase_id = 5;
        prim.source_address_bytes = 0x100;
        prim.destination_address_bytes = 0x180;
        prim.length_bytes = 8;
        prim.input_count = 3;
        prim.dtype = CollDType::INT32;
        prim.reduce_op = CollReduceOp::MAX;
        Check(prim.taskCoreDefault(context) == 2 * CYCLE,
              "REDUCE returns ceil(elements*(N-1)/lanes)*CYCLE once");
        std::vector<uint8_t> expected;
        Append(&expected, Le32(UINT32_C(0xfffffffe)));
        Append(&expected, Le32(4));
        Check(storage.Read(0x180, 8) == expected &&
                  SentinelPreserved(0x180, 8, 0x5a),
              "signed INT32 MAX handles negatives and exact destination span");
        Check(Counter(sram::Command::kRead).requests ==
                      reads.requests + 1 &&
                  Counter(sram::Command::kRead).bytes == reads.bytes + 24 &&
                  Counter(sram::Command::kWrite).requests ==
                      writes.requests + 1 &&
                  Counter(sram::Command::kWrite).bytes == writes.bytes + 8,
              "REDUCE accounts one N*L read and one L write");
    }

    void TestWrappingAndN1(TaskCoreContext &context) {
        std::vector<uint8_t> source;
        Append(&source, Le64(std::numeric_limits<uint64_t>::max()));
        Append(&source, Le64(1));
        Seed(0x120, source);
        Collective_data_v1_prim sum;
        sum.mode = CollectiveDataV1PrimMode::REDUCE;
        sum.key = {3, 4, 5};
        sum.source_address_bytes = 0x120;
        sum.destination_address_bytes = 0x1a0;
        sum.length_bytes = 8;
        sum.input_count = 2;
        sum.dtype = CollDType::INT64;
        sum.reduce_op = CollReduceOp::SUM;
        Check(sum.taskCoreDefault(context) == CYCLE &&
                  storage.Read(0x1a0, 8) == Le64(0),
              "INT64 SUM wraps and charges one vector cycle");

        const std::vector<uint8_t> one = Le64(UINT64_C(0x8877665544332211));
        Seed(0x140, one);
        hardware->vec->x_dims = 0;
        Collective_data_v1_prim n1 = sum;
        n1.key.collective_id = 6;
        n1.source_address_bytes = 0x140;
        n1.destination_address_bytes = 0x1c0;
        n1.input_count = 1;
        n1.reduce_op = CollReduceOp::MAX;
        Check(n1.taskCoreDefault(context) == 0 &&
                  storage.Read(0x1c0, 8) == one,
              "N=1 REDUCE bypasses vector hardware as local copy");
        hardware->vec->x_dims = 2;
    }

    void TestFailureAtomicity(TaskCoreContext &context) {
        SeedSentinel(0x1d0, 8, 0xcc);
        const std::vector<uint8_t> before = storage.Read(0x1d0, 8);
        const auto writes = Counter(sram::Command::kWrite);
        Collective_data_v1_prim invalid;
        invalid.mode = CollectiveDataV1PrimMode::REDUCE;
        invalid.key = {4, 5, 6};
        invalid.source_address_bytes = 0x60;
        invalid.destination_address_bytes = 0x1d0;
        invalid.length_bytes = 6;
        invalid.input_count = 2;
        invalid.dtype = CollDType::INT32;
        invalid.reduce_op = CollReduceOp::SUM;
        Check(Rejects<std::invalid_argument>([&] {
                  (void)invalid.taskCoreDefault(context);
              }) &&
                  storage.Read(0x1d0, 8) == before &&
                  Counter(sram::Command::kWrite).requests == writes.requests,
              "validation failure performs no final write");

        invalid.length_bytes = 8;
        Check(Rejects<std::runtime_error>([&] {
                  (void)invalid.taskCoreDefault(context);
              }) &&
                  storage.Read(0x1d0, 8) == before &&
                  Counter(sram::Command::kWrite).requests == writes.requests &&
                  access.outstanding() == 0,
              "failed real-byte read performs no write and drains AccessUnit");
        invalid.source_address_bytes =
            std::numeric_limits<uint64_t>::max() - 3;
        Check(Rejects<std::overflow_error>([&] { invalid.Validate(); }),
              "source N*L address overflow rejects before execution");
    }

    void Run() {
        TaskCoreContext context = Context();
        TestLocalCopy(context);
        TestNegativeMax(context);
        TestWrappingAndN1(context);
        TestFailureAtomicity(context);
        Check(access.outstanding() == 0,
              "all collective data accesses drain without residual state");
    }
};

} // namespace

int RunCollectiveDataV1PrimSelfTest() {
    CollectiveDataV1PrimBench bench("collective_data_v1_prim_bench");
    sc_start();
    if (bench.failures == 0) {
        std::cout << "[COLLECTIVE DATA V1 PRIM] PASS (" << bench.checks
                  << " checks)\n";
        return 0;
    }
    std::cerr << "[COLLECTIVE DATA V1 PRIM] FAIL (" << bench.failures << "/"
              << bench.checks << " checks failed)\n";
    return bench.failures;
}

#ifdef COLLECTIVE_DATA_V1_PRIM_SELFTEST_MAIN
int sc_main(int, char **) { return RunCollectiveDataV1PrimSelfTest(); }
#endif
