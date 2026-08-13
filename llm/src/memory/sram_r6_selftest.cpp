#include "memory/core_lsu_unit.h"
#include "memory/sram/compute_timeline.h"
#include "prims/base.h"
#include "memory/sram/sram_selftest.h"
#include "prims/norm_prims.h"
#include "prims/sram_lifecycle_prim.h"
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

    template <typename Function> bool Rejects(Function function) {
        try {
            function();
        } catch (const std::exception &) {
            return true;
        }
        return false;
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
        context.cid = 0;
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

        AddrPosKey legacy_input_key(0, 16);
        legacy_input_key.preferred_region = "scratch";
        std::string legacy_input_label = INPUT_LABEL;
        clear.prim_context->sram_pos_locator_->addPair(
            legacy_input_label, legacy_input_key, context, label_time);
        const uint64_t legacy_input_allocation =
            labels.at(legacy_input_label).region_allocation_id;
        const std::size_t allocations_before_reuse =
            regions.AllocationCount();
        Check(Rejects([&] {
                  clear.prim_context->sram_pos_locator_->changePairName(
                      legacy_input_label, renamed_protected);
              }) &&
                  labels.at(legacy_input_label).region_allocation_id ==
                      legacy_input_allocation &&
                  labels.at(renamed_protected).region_allocation_id ==
                      renamed_allocation &&
                  regions.AllocationCount() == allocations_before_reuse,
              "strict label rename rejects an existing destination atomically");
        clear.prim_context->sram_pos_locator_->changePairName(
            legacy_input_label, renamed_protected, true);
        Check(labels.count(legacy_input_label) == 0 &&
                  labels.at(renamed_protected).region_allocation_id ==
                      legacy_input_allocation &&
                  regions.FindAllocation(legacy_input_allocation).label ==
                      renamed_protected &&
                  Rejects([&] {
                      (void)regions.FindAllocation(renamed_allocation);
                  }) &&
                  regions.AllocationCount() + 1 == allocations_before_reuse,
              "legacy input-label reuse replaces an existing destination "
              "without leaking its region allocation");
        clear.prim_context->sram_pos_locator_->deletePair(renamed_protected);
        clear.prim_context->sram_pos_locator_->deletePair(layer_label);
        const auto recycled_comm = regions.AllocateAt(
            "comm", 0, 32, "recycled_comm");
        Check(recycled_comm.range.address == 128,
              "deletePair releases its bound region allocation");
        regions.Free(recycled_comm.id);
        Check(labels.empty() && regions.AllocationCount() == 0,
              "pre-lifecycle labels and allocations are fully drained");

        auto lifecycle_context = std::make_shared<PrimCoreContext>(0);
        auto RunLifecycle = [&](Sram_lifecycle &prim) {
            prim.prim_context = lifecycle_context;
            return prim.taskCoreDefault(context);
        };
        auto &lifecycle_labels =
            lifecycle_context->sram_pos_locator_->data_map;

        Sram_lifecycle spill_mismatch;
        spill_mismatch.op = SramLifecycleOp::ALLOC;
        spill_mismatch.region_name = "scratch";
        spill_mismatch.label = "p4_spill_mismatch";
        spill_mismatch.size_bytes = 8;
        spill_mismatch.alignment_bytes = 64;
        spill_mismatch.spillable = false;
        Check(Rejects([&] { RunLifecycle(spill_mismatch); }) &&
                  lifecycle_labels.empty(),
              "ALLOC rejects a spillable flag inconsistent with its region");

        Sram_lifecycle fixed_alloc = spill_mismatch;
        fixed_alloc.region_name = "protected";
        fixed_alloc.label = "p4_fixed";
        Check(Rejects([&] { RunLifecycle(fixed_alloc); }) &&
                  lifecycle_labels.empty(),
              "ALLOC rejects fixed regions without partial metadata");

        Sram_lifecycle lifecycle_alloc;
        lifecycle_alloc.op = SramLifecycleOp::ALLOC;
        lifecycle_alloc.region_name = "scratch";
        lifecycle_alloc.label = "p4_live";
        lifecycle_alloc.size_bytes = 17;
        lifecycle_alloc.alignment_bytes = 64;
        lifecycle_alloc.spillable = true;
        const sc_time alloc_time = sc_time_stamp();
        const size_t alloc_requests = access.trace().size();
        Check(RunLifecycle(lifecycle_alloc) == 0 &&
                  sc_time_stamp() == alloc_time &&
                  access.trace().size() == alloc_requests &&
                  lifecycle_labels.count("p4_live") == 1,
              "ALLOC commits locator metadata with zero modeled traffic");
        const AddrPosKey live_key = lifecycle_labels.at("p4_live");
        const auto &live_allocation =
            regions.FindAllocation(live_key.region_allocation_id);
        Check(live_key.pos == 0 && live_key.size == 17 &&
                  live_key.region_id == live_allocation.range.region_id &&
                  live_allocation.range.address == 0 &&
                  live_allocation.range.size_bytes == 17 &&
                  live_allocation.label == "p4_live",
              "ALLOC keeps locator and RegionTable metadata consistent");

        const std::vector<uint8_t> live_sentinel(48, 0xa7);
        storage.Write(0, live_sentinel);
        sram::Request lifecycle_read;
        lifecycle_read.initiator = sram::Initiator::kCompute;
        lifecycle_read.command = sram::Command::kRead;
        lifecycle_read.address = 0;
        lifecycle_read.size_bytes = 17;
        Check(access.Access(lifecycle_read).payload ==
                  std::vector<uint8_t>(17, 0xa7),
              "a lifecycle allocation is immediately readable by AccessUnit");
        Check(Rejects([&] { RunLifecycle(lifecycle_alloc); }) &&
                  lifecycle_labels.size() == 1 &&
                  regions.FindAllocation(live_key.region_allocation_id).label ==
                      "p4_live",
              "duplicate ALLOC rejects without changing existing metadata");

        lifecycle_context->sram_bind_pending_ = true;
        lifecycle_context->sram_bind_input_count_ = 1;
        lifecycle_context->sram_bind_pending_labels_.indata[0] = "p4_live";
        lifecycle_context->sram_bind_pending_labels_.outdata = "p4_live";
        Sram_lifecycle lifecycle_rename;
        lifecycle_rename.op = SramLifecycleOp::RENAME;
        lifecycle_rename.label = "p4_live";
        lifecycle_rename.new_label = "p4_renamed";
        Check(RunLifecycle(lifecycle_rename) == 0 &&
                  lifecycle_labels.count("p4_live") == 0 &&
                  lifecycle_labels.count("p4_renamed") == 1 &&
                  lifecycle_context->sram_bind_pending_labels_.indata[0] ==
                      "p4_renamed" &&
                  lifecycle_context->sram_bind_pending_labels_.outdata ==
                      "p4_renamed" &&
                  regions.FindAllocation(live_key.region_allocation_id).label ==
                      "p4_renamed",
              "RENAME atomically follows a pending one-shot SRAM_BIND");

        Sram_lifecycle lifecycle_resize;
        lifecycle_resize.op = SramLifecycleOp::RESIZE;
        lifecycle_resize.label = "p4_renamed";
        lifecycle_resize.size_bytes = 33;
        Sram_lifecycle lifecycle_free;
        lifecycle_free.op = SramLifecycleOp::FREE;
        lifecycle_free.label = "p4_renamed";
        Sram_lifecycle lifecycle_clear = lifecycle_free;
        lifecycle_clear.op = SramLifecycleOp::CLEAR_TARGETED;
        const size_t pending_requests = access.trace().size();
        Check(Rejects([&] { RunLifecycle(lifecycle_resize); }) &&
                  Rejects([&] { RunLifecycle(lifecycle_free); }) &&
                  Rejects([&] { RunLifecycle(lifecycle_clear); }) &&
                  lifecycle_labels.at("p4_renamed").size == 17 &&
                  regions.FindAllocation(live_key.region_allocation_id)
                          .range.size_bytes == 17 &&
                  access.trace().size() == pending_requests,
              "pending SRAM_BIND rejects resize/free/clear atomically");
        lifecycle_context->sram_bind_pending_ = false;
        lifecycle_context->sram_bind_input_count_ = 0;
        lifecycle_context->sram_bind_pending_labels_ = AddrDatapassLabel();

        const uint64_t lifecycle_lease = access.DeclareRangeLease(
            sram::Initiator::kCompute, sram::Command::kWrite, 0, 17);
        Check(Rejects([&] { RunLifecycle(lifecycle_resize); }) &&
                  Rejects([&] { RunLifecycle(lifecycle_free); }) &&
                  Rejects([&] { RunLifecycle(lifecycle_clear); }) &&
                  lifecycle_labels.at("p4_renamed").size == 17 &&
                  storage.IsValid(0, 17),
              "busy allocation rejects resize/free/targeted clear atomically");
        access.ReleaseRangeLease(lifecycle_lease);

        const sc_time resize_time = sc_time_stamp();
        const size_t resize_requests = access.trace().size();
        Check(RunLifecycle(lifecycle_resize) == 0 &&
                  lifecycle_labels.at("p4_renamed").size == 33 &&
                  regions.FindAllocation(live_key.region_allocation_id)
                          .range.size_bytes == 33 &&
                  sc_time_stamp() == resize_time &&
                  access.trace().size() == resize_requests,
              "RESIZE grows logical and physical metadata without traffic");
        lifecycle_resize.size_bytes = 9;
        Check(RunLifecycle(lifecycle_resize) == 0 &&
                  lifecycle_labels.at("p4_renamed").size == 9 &&
                  regions.FindAllocation(live_key.region_allocation_id)
                          .range.size_bytes == 9,
              "RESIZE shrink returns capacity and preserves label identity");

        const sc_time free_time = sc_time_stamp();
        const size_t free_requests = access.trace().size();
        Check(RunLifecycle(lifecycle_free) == 0 &&
                  lifecycle_labels.count("p4_renamed") == 0 &&
                  Rejects([&] {
                      regions.FindAllocation(live_key.region_allocation_id);
                  }) &&
                  sc_time_stamp() == free_time &&
                  access.trace().size() == free_requests &&
                  storage.Read(0, live_sentinel.size()) == live_sentinel,
              "FREE removes metadata only and leaves all SRAM bytes intact");
        Check(Rejects([&] { RunLifecycle(lifecycle_free); }),
              "double FREE and missing labels reject deterministically");

        Sram_lifecycle targeted_alloc = lifecycle_alloc;
        targeted_alloc.label = "p4_targeted";
        targeted_alloc.size_bytes = 9;
        Check(RunLifecycle(targeted_alloc) == 0 &&
                  lifecycle_labels.at("p4_targeted").pos == 0,
              "freed lifecycle capacity is reusable at the same address");
        const std::vector<uint8_t> targeted_sentinel(16, 0x5c);
        storage.Write(0, targeted_sentinel);
        Sram_lifecycle targeted_clear;
        targeted_clear.op = SramLifecycleOp::CLEAR_TARGETED;
        targeted_clear.label = "p4_targeted";
        const sc_time clear_time = sc_time_stamp();
        const size_t clear_requests = access.trace().size();
        const uint64_t targeted_id =
            lifecycle_labels.at("p4_targeted").region_allocation_id;
        Check(RunLifecycle(targeted_clear) == 0 &&
                  sc_time_stamp() > clear_time &&
                  access.trace().size() == clear_requests + 1 &&
                  lifecycle_labels.count("p4_targeted") == 0 &&
                  Rejects([&] { regions.FindAllocation(targeted_id); }) &&
                  !storage.IsValid(0, 9) && storage.IsValid(9, 7) &&
                  storage.Read(9, 7) == std::vector<uint8_t>(7, 0x5c),
              "CLEAR_TARGETED clears logical bytes then frees metadata only");

        Sram_lifecycle nonspill_alloc = lifecycle_alloc;
        nonspill_alloc.region_name = "comm";
        nonspill_alloc.label = "p4_nonspill";
        nonspill_alloc.size_bytes = 16;
        nonspill_alloc.spillable = false;
        Check(RunLifecycle(nonspill_alloc) == 0,
              "ALLOC accepts a matching non-spillable region contract");
        storage.Write(128, std::vector<uint8_t>(16, 0x39));
        Sram_lifecycle nonspill_clear;
        nonspill_clear.op = SramLifecycleOp::CLEAR_TARGETED;
        nonspill_clear.label = "p4_nonspill";
        Check(Rejects([&] { RunLifecycle(nonspill_clear); }) &&
                  lifecycle_labels.count("p4_nonspill") == 1 &&
                  storage.Read(128, 16) == std::vector<uint8_t>(16, 0x39),
              "CLEAR_TARGETED rejects non-spillable labels without data change");
        Sram_lifecycle nonspill_free;
        nonspill_free.op = SramLifecycleOp::FREE;
        nonspill_free.label = "p4_nonspill";
        RunLifecycle(nonspill_free);

        Sram_lifecycle layer_alloc = lifecycle_alloc;
        layer_alloc.label = "p4_layer";
        layer_alloc.size_bytes = 8;
        layer_alloc.lifetime = sram::AllocationLifetime::kLayer;
        RunLifecycle(layer_alloc);
        Sram_lifecycle layer_clear;
        layer_clear.op = SramLifecycleOp::CLEAR_TARGETED;
        layer_clear.label = "p4_layer";
        Check(Rejects([&] { RunLifecycle(layer_clear); }) &&
                  lifecycle_labels.count("p4_layer") == 1,
              "CLEAR_TARGETED rejects non-task lifetime labels");
        Sram_lifecycle layer_free;
        layer_free.op = SramLifecycleOp::FREE;
        layer_free.label = "p4_layer";
        RunLifecycle(layer_free);

        Sram_lifecycle full_alloc = lifecycle_alloc;
        full_alloc.label = "p4_full";
        full_alloc.size_bytes = 64;
        RunLifecycle(full_alloc);
        Sram_lifecycle capacity_alloc = lifecycle_alloc;
        capacity_alloc.label = "p4_capacity";
        capacity_alloc.size_bytes = 1;
        Check(Rejects([&] { RunLifecycle(capacity_alloc); }) &&
                  lifecycle_labels.count("p4_capacity") == 0 &&
                  lifecycle_labels.count("p4_full") == 1,
              "ALLOC capacity failure leaves existing metadata unchanged");
        Sram_lifecycle full_free;
        full_free.op = SramLifecycleOp::FREE;
        full_free.label = "p4_full";
        RunLifecycle(full_free);

        Sram_lifecycle oversized_alloc = lifecycle_alloc;
        oversized_alloc.label = "p4_oversized";
        oversized_alloc.size_bytes =
            static_cast<uint64_t>(std::numeric_limits<int>::max()) + 1;
        Check(Rejects([&] { RunLifecycle(oversized_alloc); }) &&
                  lifecycle_labels.empty(),
              "ALLOC rejects locator integer overflow before region mutation");
        Check(lifecycle_labels.empty() && regions.AllocationCount() == 0,
              "completed lifecycle program leaves no labels or allocations");

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
