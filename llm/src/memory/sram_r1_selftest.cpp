#include "memory/sram/sram_selftest.h"

#include "memory/sram/sram_region.h"
#include "memory/sram/sram_access_unit.h"
#include "memory/sram/sram_storage.h"
#include "trace/Event_engine.h"
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

    const Config absolute_config = ParseConfig(
        {{"sram_size", 1024},
         {"sram",
          {{"allocation_alignment_bytes", 16},
           {"regions",
            {{{"name", "offset_blocks"},
              {"base_bytes", 48},
              {"size_bytes", 512},
              {"allocator", "block"},
              {"spillable", true},
              {"access", {"compute"}}}}}}}});
    RegionTable absolute_table(absolute_config);
    const auto absolute_aligned = absolute_table.Allocate(
        "offset_blocks", 1, "absolute_aligned",
        AllocationLifetime::kTask, 64);
    Check(absolute_aligned.range.address == 64 &&
              absolute_aligned.range.address % 64 == 0,
          "requested alignment applies to the absolute SRAM address", &fails);
    Check(Throws<std::invalid_argument>([&] {
              absolute_table.Allocate(
                  "offset_blocks", 1, "bad_alignment",
                  AllocationLifetime::kTask, 3);
          }),
          "requested alignment rejects non-power-of-two values", &fails);
    Check(Throws<std::invalid_argument>([&] {
              absolute_table.Allocate(
                  "offset_blocks", 1, "absolute_aligned",
                  AllocationLifetime::kTask, 64);
          }),
          "allocation rejects duplicate labels without consuming a span",
          &fails);
    Check(Throws<std::overflow_error>([&] {
              table.Allocate("blocks",
                             std::numeric_limits<uint64_t>::max(),
                             "size_overflow");
          }),
          "allocation size AlignUp overflow is rejected", &fails);

    Config overflow_config;
    overflow_config.capacity_bytes = std::numeric_limits<uint64_t>::max();
    overflow_config.regions.push_back(
        {"overflow_blocks",
         std::numeric_limits<uint64_t>::max() - 255, 255,
         AllocatorKind::kBlock, true, {Initiator::kCompute}});
    RegionTable overflow_table(overflow_config);
    const auto overflow_seed = overflow_table.AllocateAt(
        "overflow_blocks", 0, 1, "overflow_seed");
    Check(overflow_seed.range.address ==
              std::numeric_limits<uint64_t>::max() - 255,
          "near-limit allocation seed is valid", &fails);
    Check(Throws<std::overflow_error>([&] {
              overflow_table.Allocate(
                  "overflow_blocks", 1, "alignment_overflow",
                  AllocationLifetime::kTask, 256);
          }),
          "allocator-selected AlignUp overflow is rejected", &fails);

    const Config lifecycle_config = ParseConfig(
        {{"sram_size", 1024},
         {"sram",
          {{"allocation_alignment_bytes", 16},
           {"regions",
            {{{"name", "lifecycle"},
              {"base_bytes", 0},
              {"size_bytes", 1024},
              {"allocator", "block"},
              {"spillable", true},
              {"access", {"compute"}}}}}}}});
    Event_engine lifecycle_trace("r1_lifecycle_trace", 1000);
    RegionTable lifecycle_table(lifecycle_config, &lifecycle_trace, 7);
    Storage lifecycle_storage(lifecycle_config.capacity_bytes);
    AccessUnit lifecycle_access(
        "r1_lifecycle_access", lifecycle_table, lifecycle_storage);
    const auto resize_a =
        lifecycle_table.Allocate("lifecycle", 256, "resize_a");
    const auto resize_b =
        lifecycle_table.Allocate("lifecycle", 256, "resize_b");
    const uint64_t resize_lease = lifecycle_access.DeclareRangeLease(
        Initiator::kCompute, Command::kWrite, resize_a.range.address,
        resize_a.range.size_bytes);
    Check(Throws<std::runtime_error>([&] {
              lifecycle_table.ResizeAllocation(resize_a.id, 128);
          }) &&
              lifecycle_table.FindAllocation(resize_a.id).range.size_bytes ==
                  256,
          "busy resize rejects without changing allocation size", &fails);
    lifecycle_access.ReleaseRangeLease(resize_lease);
    Check(Throws<std::bad_alloc>([&] {
              lifecycle_table.ResizeAllocation(resize_a.id, 512);
          }) &&
              lifecycle_table.FindAllocation(resize_a.id).range.size_bytes ==
                  256,
          "failed grow is allocation-atomic", &fails);

    const size_t trace_before_collision =
        lifecycle_trace.Trace_event_queue_clock_engine.trace_event_queue.size();
    Check(Throws<std::invalid_argument>([&] {
              lifecycle_table.RenameAllocation(resize_a.id, "resize_b");
          }) &&
              lifecycle_table.FindAllocation(resize_a.id).label ==
                  "resize_a" &&
              lifecycle_table.FindAllocation(resize_b.id).label ==
                  "resize_b" &&
              lifecycle_trace.Trace_event_queue_clock_engine.trace_event_queue
                      .size() == trace_before_collision,
          "rename collision is metadata- and trace-atomic", &fails);
    lifecycle_table.RenameAllocation(resize_a.id, "resize_a_renamed");
    Check(lifecycle_table.FindAllocation(resize_a.id).label ==
                  "resize_a_renamed" &&
              lifecycle_trace.Trace_event_queue_clock_engine.trace_event_queue
                      .size() == trace_before_collision + 2,
          "successful rename emits paired lifecycle trace events", &fails);

    lifecycle_table.Free(resize_b.id);
    const auto &grown =
        lifecycle_table.ResizeAllocation(resize_a.id, 512);
    Check(grown.range.size_bytes == 512,
          "free after failed grow proves the span map stayed reusable", &fails);
    lifecycle_table.ResizeAllocation(resize_a.id, 128);
    const auto shrink_reuse = lifecycle_table.AllocateAt(
        "lifecycle", 128, 384, "shrink_reuse");
    Check(shrink_reuse.range.address == 128,
          "shrink returns its tail span to the allocator", &fails);
    lifecycle_table.Free(shrink_reuse.id);
    lifecycle_table.Free(resize_a.id);
    Check(Throws<std::bad_alloc>([&] {
              lifecycle_table.Allocate("lifecycle", 2048, "over_capacity");
          }),
          "allocator rejects capacity overflow without metadata", &fails);

    std::cout << "[SRAM R1] " << (fails == 0 ? "PASS" : "FAIL")
              << " failures=" << fails << std::endl;
    return fails;
}
