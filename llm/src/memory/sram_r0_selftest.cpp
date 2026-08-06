#include "memory/sram/sram_selftest.h"

#include "memory/sram/sram_region.h"
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
    std::cerr << "[SRAM R0] FAIL: " << message << std::endl;
}

} // namespace

int RunSramR0SelfTest() {
    using namespace sram;
    int fails = 0;
    const nlohmann::json memory = {
        {"sram_size", 8192},
        {"sram",
         {{"bank_count", 4},
          {"bank_interleave_bytes", 64},
          {"lsu",
           {{"queue_depth", 12},
            {"max_outstanding", 3},
            {"issue_latency_ns", 4}}},
          {"dte_memory", {{"queue_depth", 10}, {"workers", 3}}},
          {"allocation_alignment_bytes", 256},
          {"real_data_path", true},
          {"manual_regions", true},
          {"ports",
           {{"lsu",
             {{"read", {{"count", 2}, {"width_bits", 256}}},
              {"write", {{"count", 1}, {"width_bits", 128}}}}}}},
          {"regions",
           {{{"name", "input"},
             {"base_bytes", 0},
             {"size_bytes", 4096},
             {"allocator", "block"},
             {"spillable", true},
             {"access", {"compute", "dte", "lsu"}}},
            {{"name", "comm"},
             {"base_bytes", 4096},
             {"size_bytes", 4096},
             {"allocator", "fixed"},
             {"access", {"dte", "noc_rx"}}}}}}}};
    const Config config = ParseConfig(memory);
    Check(config.capacity_bytes == 8192, "capacity is parsed in bytes", &fails);
    Check(config.bank_count == 4 && config.bank_interleave_bytes == 64,
          "bank geometry is parsed", &fails);
    Check(config.lsu.read.count == 2 && config.lsu.read.width_bits == 256,
          "per-initiator ports are parsed", &fails);
    Check(config.lsu_queue_depth == 12 &&
              config.lsu_max_outstanding == 3 &&
              config.lsu_issue_latency_ns == 4 &&
              config.dte_memory_queue_depth == 10 &&
              config.dte_memory_workers == 3,
          "independent LSU and DTE-memory concurrency config is parsed",
          &fails);
    Check(config.real_data_path && config.manual_regions,
          "compatibility gates are parsed", &fails);

    const Config legacy = ParseConfig({{"sram_size", 1024}});
    Check(legacy.regions.size() == 1 && legacy.regions[0].name == "legacy" &&
              legacy.regions[0].size_bytes == 1024,
          "old configuration maps to a full legacy region", &fails);

    auto bad_overlap = memory;
    bad_overlap["sram"]["regions"][1]["base_bytes"] = 2048;
    Check(Throws<std::invalid_argument>([&] { ParseConfig(bad_overlap); }),
          "overlap is rejected", &fails);
    auto bad_bounds = memory;
    bad_bounds["sram"]["regions"][1]["size_bytes"] = 8192;
    Check(Throws<std::out_of_range>([&] { ParseConfig(bad_bounds); }),
          "out-of-capacity region is rejected", &fails);
    auto bad_alignment = memory;
    bad_alignment["sram"]["regions"][1]["base_bytes"] = 4097;
    Check(Throws<std::invalid_argument>([&] { ParseConfig(bad_alignment); }),
          "misaligned region is rejected", &fails);
    auto bad_access = memory;
    bad_access["sram"]["regions"][0]["access"] = {"gpu_magic"};
    Check(Throws<std::invalid_argument>([&] { ParseConfig(bad_access); }),
          "unknown access initiator is rejected", &fails);

    nlohmann::json hardware;
    hardware["memory"] = memory;
    hardware["cores"] = nlohmann::json::array();
    hardware["cores"].push_back(
        {{"id", 1},
         {"lsu", {{"queue_depth", 6},
                  {"max_outstanding", 2},
                  {"issue_latency_ns", 7}}},
         {"sram",
          {{"capacity_bytes", 4096},
           {"allocation_alignment_bytes", 256},
           {"regions",
            {{{"name", "private"},
              {"base_bytes", 0},
              {"size_bytes", 4096},
              {"allocator", "block"}}}}}}});
    auto &registry = ConfigRegistry::Instance();
    registry.ResetForTest();
    registry.Configure(hardware, 2);
    Check(registry.ForCore(0).capacity_bytes == 8192 &&
              registry.ForCore(1).capacity_bytes == 4096 &&
              registry.ForCore(1).regions[0].name == "private" &&
              registry.ForCore(1).lsu_queue_depth == 6 &&
              registry.ForCore(1).lsu_issue_latency_ns == 7 &&
              registry.ForCore(1).bank_count == config.bank_count,
          "per-core override replaces layout fields and inherits unspecified geometry", &fails);
    registry.ResetForTest();

    std::cout << "[SRAM R0] " << (fails == 0 ? "PASS" : "FAIL")
              << " failures=" << fails << std::endl;
    return fails;
}
