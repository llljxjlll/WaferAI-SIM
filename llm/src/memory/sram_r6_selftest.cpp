#include "memory/core_lsu_unit.h"
#include "memory/sram/compute_timeline.h"
#include "prims/base.h"
#include "memory/sram/sram_selftest.h"
#include "prims/norm_prims.h"
#include "utils/memory_utils.h"
#include "utils/system_utils.h"

#include <iostream>
#include <map>

namespace {

class ManualSchedulePrim : public NpuBase {
  public:
    bool ran = false;
    std::vector<uint8_t> observed;

    ManualSchedulePrim() { name = "ManualSchedulePrim"; }
    void initialize() override {}
    void taskCore(TaskCoreContext &context, string, u_int64_t &,
                  u_int64_t &, u_int64_t &, u_int64_t &) override {
        if (context.lsu_memory == nullptr ||
            context.compute_timeline == nullptr)
            throw std::runtime_error(
                "manual primitive requires LSU and compute timeline");
        const auto token =
            context.lsu_memory->IssueLoad(0x600, 0, 32);
        context.lsu_memory->Wait(token);
        observed = context.compute_timeline->RunTile({0, 32, 0}, 3);
        context.lsu_memory->Store(0, 0x700, 32);
        ran = true;
    }
};

class R6FakeHbm : public sram::HbmByteTransport {
  public:
    std::map<uint64_t, uint8_t> bytes;

    std::vector<uint8_t> Read(uint64_t address,
                              uint64_t size_bytes) override {
        wait(sc_time(4, SC_NS));
        std::vector<uint8_t> result(size_bytes);
        for (uint64_t i = 0; i < size_bytes; ++i)
            result[i] = bytes[address + i];
        return result;
    }

    void Write(uint64_t address, const std::vector<uint8_t> &payload,
               const std::vector<uint8_t> &byte_enable) override {
        wait(sc_time(4, SC_NS));
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

struct R6Bench : sc_module {
    SC_HAS_PROCESS(R6Bench);
    sram::Storage storage;
    sram::RegionTable regions;
    sram::AccessUnit access;
    R6FakeHbm hbm;
    sram::CoreLsuUnit lsu;
    sram::ComputeTimeline timeline;
    int fails = 0;

    explicit R6Bench(sc_module_name name)
        : sc_module(name), storage(256),
          regions(sram::ParseConfig(
              {{"sram_size", 256},
               {"sram", {{"real_data_path", true},
                          {"manual_memory_schedule", true},
                          {"regions",
                           {{{"name", "scratch"},
                             {"base_bytes", 0},
                             {"size_bytes", 64},
                             {"allocator", "block"},
                             {"spillable", true},
                             {"access", {"compute", "dte", "lsu"}}},
                            {{"name", "input"},
                             {"base_bytes", 64},
                             {"size_bytes", 64},
                             {"allocator", "block"},
                             {"spillable", true},
                             {"access", {"compute", "dte", "lsu"}}},
                            {{"name", "comm"},
                             {"base_bytes", 128},
                             {"size_bytes", 64},
                             {"allocator", "block"},
                             {"spillable", false},
                             {"access", {"compute", "dte", "lsu"}}},
                            {{"name", "protected"},
                             {"base_bytes", 192},
                             {"size_bytes", 64},
                             {"allocator", "fixed"},
                             {"spillable", false},
                             {"access", {"compute", "dte", "lsu"}}}}}}}})),
          access("access", regions, storage),
          lsu("lsu", regions, access, hbm, 4), timeline(access) {
        SC_THREAD(Run);
    }

    void Check(bool condition, const char *message) {
        if (condition) return;
        ++fails;
        std::cerr << "[SRAM R6] FAIL: " << message << std::endl;
    }

    void Run() {
        CoreHWConfig fallback_core;
        CoreHWConfig *test_core = nullptr;
        for (const auto &entry : g_core_hw_config)
            if (entry.first == 0) test_core = entry.second;
        const bool installed_fallback = test_core == nullptr;
        if (installed_fallback) {
            fallback_core.id = 0;
            g_core_hw_config.emplace_back(0, &fallback_core);
            test_core = &fallback_core;
        }
        const int original_sram_bitwidth = test_core->sram_bitwidth;
        test_core->sram_bitwidth = 512;
        std::vector<uint8_t> pattern(32);
        for (size_t i = 0; i < pattern.size(); ++i)
            pattern[i] = static_cast<uint8_t>(0x21 + 5 * i);
        hbm.Seed(0x100, pattern);

        int legacy_sram_addr = 0;
        TaskCoreContext context(
            nullptr, nullptr, nullptr, nullptr, &legacy_sram_addr, nullptr,
            nullptr, nullptr, nullptr, uint64_t{0}, unsigned{0});
        context.lsu_memory = &lsu;
        context.sram_regions = &regions;
        context.sram_access = &access;
        context.sram_storage = &storage;
        context.compute_timeline = &timeline;
        Check(LegacySramWordBytes(context) == 64 &&
                  LegacySramByteAddress(context, 1) == 64,
              "512-bit legacy SRAM pos=1 resolves to byte address 64");

        Load_prim load;
        load.dram_addr = 0x100;
        load.sram_addr = 32;
        load.size = pattern.size();
        load.taskCoreDefault(context);
        Check(storage.Read(32, pattern.size()) == pattern,
              "Load_prim compatibility alias forwards HBM to SRAM");

        Load_prim decoded_load;
        decoded_load.deserialize(load.serialize());
        Check(decoded_load.dram_addr == load.dram_addr &&
                  decoded_load.sram_addr == load.sram_addr &&
                  decoded_load.size == load.size,
              "Load_prim compatibility wire round-trip");

        Store_prim store;
        store.dram_addr = 0x200;
        store.sram_addr = 32;
        store.size = pattern.size();
        store.taskCoreDefault(context);
        Check(hbm.Peek(0x200, pattern.size()) == pattern,
              "Store_prim compatibility alias forwards SRAM to HBM");

        Store_prim decoded_store;
        decoded_store.deserialize(store.serialize());
        Check(decoded_store.dram_addr == store.dram_addr &&
                  decoded_store.sram_addr == store.sram_addr &&
                  decoded_store.size == store.size,
              "Store_prim compatibility wire round-trip");

        hbm.Seed(0x600, pattern);
        ManualSchedulePrim manual;
        manual.prim_context = std::make_shared<PrimCoreContext>(0);
        manual.data_size_input = {8};
        manual.data_chunk = {{"output", 32}};
        manual.data_byte = 1;
        manual.out_size = 32;
        manual.inp_offset = 0;
        manual.data_chunk_addr["output"] = 0;
        manual.prim_context->datapass_label_->indata[0] = "manual_input";
        manual.prim_context->datapass_label_->outdata = "manual_output";
        std::string manual_label = "manual_input";
        manual.prim_context->sram_pos_locator_->data_map[manual_label] =
            AddrPosKey(96, 8, 0x680);
        Check(manual.taskCoreDefault(context) == 0 && manual.ran &&
                  manual.observed == pattern &&
                  hbm.Peek(0x700, pattern.size()) == pattern,
              "manual NpuBase primitive controls load/compute/store in one "
              "primitive");
        Check(manual.prim_context->sram_pos_locator_->data_map.count(
                  manual_label) == 1,
              "manual schedule skips automatic input-label deletion");

        const std::vector<uint8_t> protected_pattern(32, 0x9a);
        const std::vector<uint8_t> layer_pattern(32, 0x7b);
        sram::Request seed;
        seed.initiator = sram::Initiator::kCompute;
        seed.command = sram::Command::kWrite;
        seed.size_bytes = 32;
        seed.address = 128;
        seed.payload = protected_pattern;
        access.Access(seed);
        seed.address = 64;
        seed.payload = layer_pattern;
        access.Access(seed);

        Clear_sram clear;
        clear.prim_context = std::make_shared<PrimCoreContext>(0);
        clear.prim_context->sram_pos_locator_->BindRegionTable(&regions);
        auto &labels = clear.prim_context->sram_pos_locator_->data_map;
        AddrPosKey scratch_key(0, 32, 0x100);
        scratch_key.preferred_region = "scratch";
        AddrPosKey protected_key(2, 32);
        protected_key.preferred_region = "comm";
        AddrPosKey layer_key(1, 32);
        layer_key.preferred_region = "input";
        layer_key.allocation_lifetime =
            sram::AllocationLifetime::kLayer;
        uint64_t label_time = 0;
        std::string scratch_label = "scratch_task";
        std::string protected_label = "comm_buffer";
        std::string layer_label = "layer_cache";
        clear.prim_context->sram_pos_locator_->addPair(
            scratch_label, scratch_key, context, label_time);
        clear.prim_context->sram_pos_locator_->addPair(
            protected_label, protected_key, context, label_time);
        clear.prim_context->sram_pos_locator_->addPair(
            layer_label, layer_key, context, label_time);
        const uint64_t scratch_allocation =
            labels.at(scratch_label).region_allocation_id;
        Check(scratch_allocation != 0 &&
                  labels.at(layer_label).region_allocation_id != 0 &&
                  labels.at(protected_label).region_allocation_id != 0,
              "production label insertion binds scratch/input/comm block regions");
        Check(regions.Region(labels.at(scratch_label).region_id).name ==
                  "scratch" &&
                  regions.Region(labels.at(layer_label).region_id).name ==
                  "input" &&
                  regions.Region(labels.at(protected_label).region_id).name ==
                  "comm",
              "production label roles select their configured regions");
        layer_key.size = 48;
        clear.prim_context->sram_pos_locator_->addPair(
            layer_label, layer_key, context, label_time);
        const auto &grown_layer = regions.FindAllocation(
            labels.at(layer_label).region_allocation_id);
        Check(grown_layer.range.address == 64 &&
                  grown_layer.range.size_bytes == 48,
              "production tiled label growth resizes its bound allocation");
        clear.taskCoreDefault(context);
        Check(!storage.IsValid(0, 32) && storage.IsValid(128, 32) &&
                  storage.IsValid(64, 32),
              "Clear_sram clears task data but protects non-spillable and "
              "layer data");
        Check(labels.count("scratch_task") == 0 &&
                  labels.count("comm_buffer") == 1 &&
                  labels.count("layer_cache") == 1,
              "Clear_sram label lifecycle follows region and allocation "
              "policy");
        Check(legacy_sram_addr == 3,
              "Clear_sram converts byte high-water 160 back to three words");
        const auto recycled = regions.AllocateAt(
            "scratch", 0, 32, "recycled");
        Check(recycled.range.address == 0,
              "Clear_sram frees the production region allocation");
        regions.Free(recycled.id);
        std::string renamed_protected = "comm_buffer_renamed";
        clear.prim_context->sram_pos_locator_->changePairName(
            protected_label, renamed_protected);
        const auto renamed_allocation = labels.at(renamed_protected)
                                            .region_allocation_id;
        Check(regions.FindAllocation(renamed_allocation).label ==
                  renamed_protected,
              "label rename updates its bound region allocation metadata");
        clear.prim_context->sram_pos_locator_->deletePair(renamed_protected);
        const auto recycled_comm = regions.AllocateAt(
            "comm", 0, 32, "recycled_comm");
        Check(recycled_comm.range.address == 128,
              "deletePair releases its bound region allocation");
        regions.Free(recycled_comm.id);

        context.lsu_memory = nullptr;
        Load_prim historical_empty_load;
        historical_empty_load.taskCoreDefault(context);
        Store_prim historical_empty_store;
        historical_empty_store.size = 0;
        historical_empty_store.taskCoreDefault(context);
        Check(true, "historical empty Load/Store remain no-op when disabled");

        Check(lsu.stats().hbm_read_bytes == 2 * pattern.size() &&
                  lsu.stats().hbm_write_bytes == 2 * pattern.size() &&
                  lsu.stats().sram_write_bytes == 2 * pattern.size() &&
                  lsu.stats().sram_read_bytes == 2 * pattern.size(),
              "manual and compatibility primitives share balanced LSU/HBM/SRAM paths");
        Check(lsu.OutstandingCount() == 0,
              "compatibility aliases consume all internal tokens");
        test_core->sram_bitwidth = original_sram_bitwidth;
        if (installed_fallback) g_core_hw_config.pop_back();
        sc_stop();
    }
};

} // namespace

int RunSramR6SelfTest() {
    R6Bench bench("sram_r6_bench");
    sc_start();
    std::cout << "[SRAM R6] " << (bench.fails == 0 ? "PASS" : "FAIL")
              << " failures=" << bench.fails << std::endl;
    return bench.fails;
}
