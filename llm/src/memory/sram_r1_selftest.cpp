#include "memory/sram/sram_selftest.h"

#include "memory/sram/sram_region.h"
#include "memory/sram/sram_access_unit.h"
#include "memory/sram/sram_storage.h"
#include <iostream>

namespace {

template <typename Exception, typename F> bool Throws(F &&fn) {
    try {
        fn();
    } catch (const Exception &) {
        return true;
    } catch (...) {
    }
    return false;
}

void Check(bool condition, const char *message, int *fails) {
    if (condition) return;
    ++*fails;
    std::cerr << "[SRAM R1] FAIL: " << message << std::endl;
}

} // namespace

int RunSramR1SelfTest() {
    using namespace sram;
    int fails = 0;
    Storage core0(1024);
    Storage core1(1024);
    Storage timing_only(128, false);
    std::vector<uint8_t> pattern(37);
    for (size_t i = 0; i < pattern.size(); ++i)
        pattern[i] = static_cast<uint8_t>(3 * i + 7);
    core0.Write(100, pattern);
    Check(core0.Read(100, pattern.size()) == pattern,
          "non-zero byte pattern round-trips", &fails);
    Check(!core1.IsValid(100, pattern.size()),
          "different per-core storage instances are isolated", &fails);
    Check(Throws<std::runtime_error>([&] { core1.Read(100, pattern.size()); }),
          "invalid read fails", &fails);

    std::vector<uint8_t> patch = {0xaa, 0xbb, 0xcc, 0xdd};
    core0.Write(110, patch, {1, 0, 1, 0});
    auto readback = core0.Read(100, pattern.size());
    Check(readback[10] == 0xaa && readback[11] == pattern[11] &&
              readback[12] == 0xcc && readback[13] == pattern[13],
          "byte-enable performs partial update", &fails);
    core0.Clear(100, pattern.size());
    Check(!core0.IsValid(100, pattern.size()), "clear invalidates bytes", &fails);
    Check(Throws<std::out_of_range>([&] { core0.Write(1000, pattern); }),
          "out-of-bounds write fails", &fails);

    timing_only.Write(8, pattern);
    core0.Write(8, pattern);
    Check(timing_only.Signature(8, pattern.size()) ==
              core0.Signature(8, pattern.size()),
          "timing-only storage preserves payload signature", &fails);
    const uint64_t timing_signature =
        timing_only.Signature(8, pattern.size());
    timing_only.Write(8, std::vector<uint8_t>(pattern.size(), 0x5a));
    Check(timing_only.Signature(8, pattern.size()) != timing_signature,
          "timing-only signature changes with writes", &fails);

    const Config config = ParseConfig(
        {{"sram_size", 4096},
         {"sram",
          {{"allocation_alignment_bytes", 256},
           {"regions",
            {{{"name", "blocks"},
              {"base_bytes", 0},
              {"size_bytes", 2048},
              {"allocator", "block"},
              {"access", {"compute", "lsu"}}},
             {{"name", "fixed"},
              {"base_bytes", 2048},
              {"size_bytes", 2048},
              {"allocator", "fixed"},
              {"access", {"dte"}}}}}}}});
    RegionTable table(config);
    const auto resolved =
        table.Resolve("blocks", 256, 128, Initiator::kLsu, Command::kWrite);
    Check(resolved.address == 256 && resolved.region_id == 0,
          "region-relative range resolves to byte address", &fails);
    Check(Throws<std::invalid_argument>([&] {
              table.Resolve("fixed", 0, 16, Initiator::kLsu, Command::kRead);
          }),
          "region permission is enforced", &fails);
    Check(Throws<std::out_of_range>([&] {
              table.ResolveAbsolute(2000, 128, Initiator::kCompute,
                                    Command::kRead);
          }),
          "descriptor cannot cross a region boundary", &fails);

    const auto a = table.Allocate("blocks", 300, "a");
    const auto b = table.Allocate("blocks", 300, "b");
    Check(a.range.size_bytes == 512 && b.range.address == 512,
          "block allocations honor alignment", &fails);
    table.Free(a.id);
    const auto c = table.Allocate("blocks", 128, "c");
    Check(c.range.address == 0,
          "free span is reusable without crossing regions", &fails);

    Storage region_storage(config.capacity_bytes);
    AccessUnit region_access("r1_region_access", table, region_storage);
    const uint64_t lease = region_access.DeclareRangeLease(
        Initiator::kCompute, Command::kWrite, c.range.address,
        c.range.size_bytes);
    Check(Throws<std::runtime_error>([&] { table.Free(c.id); }),
          "allocation free rejects an outstanding range lease", &fails);
    region_access.ReleaseRangeLease(lease);
    table.Free(c.id);

    const auto persistent = table.Allocate(
        "blocks", 128, "persistent", AllocationLifetime::kPersistent);
    Check(persistent.lifetime == AllocationLifetime::kPersistent,
          "allocation records persistent lifetime", &fails);
    Check(Throws<std::runtime_error>(
              [&] { table.Free(persistent.id); }),
          "task boundary cannot free persistent allocation", &fails);
    table.Free(persistent.id, AllocationLifetime::kPersistent);
    Check(Throws<std::invalid_argument>([&] { table.Free(a.id); }),
          "double free fails", &fails);
    Check(Throws<std::invalid_argument>([&] {
              table.Allocate("fixed", 256, "bad");
          }),
          "fixed region rejects allocator requests", &fails);

    std::cout << "[SRAM R1] " << (fails == 0 ? "PASS" : "FAIL")
              << " failures=" << fails << std::endl;
    return fails;
}
