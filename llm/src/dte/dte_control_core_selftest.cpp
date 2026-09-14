#include "common/config.h"
#include "defs/global.h"
#include "dte/dte_control_frontend.h"

#include "dte/dte_async.h"
#include "dte/dte_memory_bridge.h"
#include "dte/dte_unit.h"
#include "memory/sram/sram_selftest.h"
#include "macros/macros.h"
#include "utils/config_utils.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
template <typename Exception, typename Function>
bool ThrowsExpected(Function &&function) {
    try {
        function();
    } catch (const Exception &) {
        return true;
    } catch (...) {
    }
    return false;
}

DTEConfig SelfTestDteConfig() {
    DTEConfig config;
    config.channel_count = 2;
    config.bit_width_bits = 128;
    config.gamma_cycles = 1;
    config.tau_launch_cycles = 0;
    return config;
}

class FailingHbm : public sram::HbmByteTransport {
public:
    std::vector<uint8_t> Read(uint64_t, uint64_t) override {
        wait(sc_time(1, SC_NS));
        throw std::runtime_error("injected DTE memory read failure");
    }
    void Write(uint64_t, const std::vector<uint8_t> &,
               const std::vector<uint8_t> &) override {
        wait(sc_time(1, SC_NS));
        throw std::runtime_error("injected DTE memory write failure");
    }
};

class DteControlProbe : public sc_module {
public:
    SC_HAS_PROCESS(DteControlProbe);

    explicit DteControlProbe(const sc_module_name &name)
        : sc_module(name), legacy_unit("legacy_unit", SelfTestDteConfig()),
          dedicated_unit("dedicated_unit", SelfTestDteConfig()),
          aggregate_unit("aggregate_unit", SelfTestDteConfig()),
          memory_storage(256), memory_regions(MemorySramConfig()),
          memory_access("memory_access", memory_regions, memory_storage),
          memory_bridge("memory_bridge", memory_regions, memory_access,
                        failing_hbm),
          memory_unit("memory_unit", MemoryDteConfig()),
          legacy_async("legacy_async", legacy_unit),
          dedicated_async("dedicated_async", dedicated_unit),
          aggregate_async("aggregate_async", aggregate_unit,
                          AggregateConfig()),
          memory_async("memory_async", memory_unit),
          dedicated_core("dedicated_core", dedicated_unit,
                         DedicatedConfig()),
          aggregate_core("aggregate_core", aggregate_unit,
                         DedicatedConfig()),
          memory_core("memory_core", memory_unit, DedicatedConfig()),
          legacy(legacy_unit, &legacy_async),
          dedicated(dedicated_unit, dedicated_core, &dedicated_async),
          aggregate(aggregate_unit, aggregate_core, &aggregate_async),
          memory(memory_unit, memory_core, &memory_async) {
        memory.BindMemoryBridge(&memory_bridge);
        SC_THREAD(Run);
    }

    int failures = 0;
    int checks = 0;

private:
    static DteControlCoreConfig DedicatedConfig() {
        DteControlCoreConfig config;
        config.command_queue_depth = 2;
        config.dispatch_width = 1;
        config.dispatch_latency = sc_time(2, SC_NS);
        config.completion_notify_latency = sc_time(3, SC_NS);
        return config;
    }

    static DteAggregationConfig AggregateConfig() {
        DteAggregationConfig config;
        config.enabled = true;
        config.max_descriptors = 4;
        config.max_payload_bytes = 1024;
        config.timeout_cycles = 1000;
        config.address_block_bytes = 4096;
        return config;
    }

    static sram::Config MemorySramConfig() {
        return sram::ParseConfig(
            {{"sram_size", 256},
             {"sram",
              {{"bank_count", 2},
               {"bank_interleave_bytes", 64},
               {"regions",
                {{{"name", "data"},
                  {"base_bytes", 0},
                  {"size_bytes", 256},
                  {"allocator", "fixed"},
                  {"access", {"compute", "dte", "lsu"}}}}}}}});
    }

    static DTEConfig MemoryDteConfig() {
        DTEConfig config = SelfTestDteConfig();
        config.fine_grained_resources = true;
        config.command_slots_per_channel = 2;
        config.pending_queue_depth = 4;
        config.spm_read_width_bits = 128;
        config.spm_write_width_bits = 128;
        config.axi_read_width_bits = 128;
        config.axi_write_width_bits = 128;
        return config;
    }

    void Check(bool condition, const std::string &message) {
        ++checks;
        std::cout << "  [" << (condition ? " ok " : "FAIL") << "] "
                  << message << "\n";
        if (!condition)
            ++failures;
    }

    void Run() {
        const sc_time legacy_before = sc_time_stamp();
        const DteTransferHandle legacy_handle =
            legacy.Issue(256, DteDir::SPM_TO_REMOTE);
        const DteTransferSnapshot legacy_issued =
            legacy.Snapshot(legacy_handle);
        Check(sc_time_stamp() == legacy_before &&
                  legacy_issued.backend_issue_time == legacy_before,
              "legacy Issue adds no controller wait");
        Check(legacy.BitWidth() == 128 && legacy_issued.backend_valid,
              "legacy facade exposes value snapshot and DTE width");
        legacy.WaitTransmitStart(legacy_handle);
        legacy.Wait(legacy_handle);
        Check(legacy.Snapshot(legacy_handle).control_state ==
                  DteControlTransferState::COMPLETED,
              "legacy Wait observes backend completion");
        Check(legacy.Release(legacy_handle),
              "legacy facade releases completed backend transfer");

        const DteTransferHandle first =
            dedicated.Issue(256, DteDir::SPM_TO_REMOTE);
        const DteTransferHandle second =
            dedicated.Issue(256, DteDir::REMOTE_TO_SPM);
        const DteTransferSnapshot queued = dedicated.Snapshot(first);
        Check(!queued.backend_valid &&
                  queued.control_state == DteControlTransferState::QUEUED,
              "dedicated Issue returns a stable handle while queued");

        dedicated.WaitTransmitStart(first);
        const DteTransferSnapshot first_started = dedicated.Snapshot(first);
        Check(first_started.backend_valid &&
                  first_started.backend_issue_time - queued.enqueue_time >=
                      sc_time(2, SC_NS),
              "dispatch latency precedes backend issue");
        dedicated.Wait(first);
        dedicated.Wait(second);
        const DteTransferSnapshot first_done = dedicated.Snapshot(first);
        const DteTransferSnapshot second_done = dedicated.Snapshot(second);
        Check(first_done.notification_time -
                      first_done.backend_completion_time ==
                  sc_time(4, SC_NS),
              "notification latency is rounded up to the control cycle");
        Check(second_done.backend_issue_time >=
                  first_started.backend_issue_time + sc_time(CYCLE, SC_NS),
              "dispatch_width limits issue rate to one command per cycle");
        Check(dedicated_core.statistics().enqueued == 5 &&
                  dedicated_core.statistics().dispatched == 5 &&
                  dedicated_core.statistics().completed == 5 &&
                  dedicated_core.statistics().max_queue_occupancy == 2,
              "physical issue/wait operations share the command FIFO");
        Check(dedicated.Release(first) && dedicated.Release(second) &&
                  dedicated_core.OutstandingCount() == 0,
              "physical release commands retire stable handles");

        dedicated.IssueToken(10, 4096, DteDir::SPM_TO_REMOTE,
                             0x0000, 512);
        (void)dedicated.PollToken(10);
        Check(dedicated.HasToken(10),
              "POLL_TOKEN reports status without reclaiming the token");
        dedicated.WaitToken(10);
        Check(!dedicated.HasToken(10),
              "WAIT_TOKEN retires a completed logical token");

        dedicated.IssueToken(20, 8192, DteDir::SPM_TO_REMOTE,
                             0x1000, 1024);
        bool fence_done = false;
        std::exception_ptr fence_failure;
        sc_spawn([&] {
            try {
                dedicated.Fence();
                fence_done = true;
            } catch (...) {
                fence_failure = std::current_exception();
            }
        }, sc_gen_unique_name("dte-control-fence-test"));
        wait(SC_ZERO_TIME);
        dedicated.IssueToken(21, 8192, DteDir::REMOTE_TO_SPM,
                             0x3000, 1024);
        while (!fence_done && !fence_failure)
            wait(dedicated_async.StateChangedEvent());
        Check(fence_failure == nullptr && dedicated.HasToken(21),
              "FENCE watermark excludes a token issued after the fence");
        dedicated.WaitToken(21);

        const uint64_t staged_cancel = aggregate.IssueToken(
            30, 256, DteDir::SPM_TO_REMOTE, 0x0000, 32, 1, 0x1000, 1);
        Check(staged_cancel == DTE_ASYNC_INVALID_XFER_ID &&
                  !aggregate.PollToken(30),
              "staged aggregation is visible to FIFO POLL_TOKEN");
        aggregate.CancelToken(30);
        Check(!aggregate.HasToken(30) &&
                  aggregate_unit.InflightCount() == 0,
              "CANCEL_TOKEN drains a staged tail without a watcher leak");

        aggregate.IssueToken(31, 256, DteDir::SPM_TO_REMOTE,
                             0x0100, 32, 1, 0x1100, 1);
        aggregate.WaitToken(31);
        Check(!aggregate.HasToken(31) &&
                  aggregate_async.OpenGroupCount() == 0,
              "WAIT_TOKEN flushes and retires staged aggregation");

        Check(ThrowsExpected<std::invalid_argument>([&] {
                  dedicated.WaitToken(999);
              }) && dedicated_core.OutstandingCount() == 0 &&
                  dedicated_core.statistics().failed == 1,
              "FAILED command is counted, propagated and consumed");

        const auto &stats = dedicated_core.statistics();
        Check(stats.opcode_count.at(static_cast<size_t>(
                  DteControlOpcode::ISSUE_TOKEN)) >= 3 &&
                  stats.opcode_count.at(static_cast<size_t>(
                  DteControlOpcode::WAIT_TOKEN)) >= 3 &&
                  stats.opcode_count.at(static_cast<size_t>(
                  DteControlOpcode::POLL_TOKEN)) >= 1 &&
                  stats.opcode_count.at(static_cast<size_t>(
                  DteControlOpcode::FENCE)) == 1 &&
                  stats.opcode_count.at(static_cast<size_t>(
                  DteControlOpcode::RELEASE_TRANSFER)) == 2,
              "per-opcode command counters cover physical and logical FIFO");
        Check(dedicated_core.Residual().Drained() &&
                  aggregate_core.Residual().Drained(),
              "controller residual is zero after success/cancel/error paths");

        memory.IssueToken(40, 256, DteDir::DRAM_TO_SPM,
                          0, 32, DTE_ASYNC_INVALID_REMOTE_PEER,
                          0x4000, 0);
        Check(ThrowsExpected<std::runtime_error>([&] {
                  memory.WaitToken(40);
              }) && memory_async.OutstandingCount() == 0 &&
                  memory_bridge.OutstandingCount() == 0 &&
                  memory_unit.InflightCount() == 0 &&
                  memory_core.Residual().Drained(),
              "memory failure cleans bridge/token/batch/context before rethrow");
    }

    DTEUnit legacy_unit;
    DTEUnit dedicated_unit;
    DTEUnit aggregate_unit;
    sram::Storage memory_storage;
    sram::RegionTable memory_regions;
    sram::AccessUnit memory_access;
    FailingHbm failing_hbm;
    DteMemoryBridge memory_bridge;
    DTEUnit memory_unit;
    DteAsyncTracker legacy_async;
    DteAsyncTracker dedicated_async;
    DteAsyncTracker aggregate_async;
    DteAsyncTracker memory_async;
    DteControlCore dedicated_core;
    DteControlCore aggregate_core;
    DteControlCore memory_core;
    DteControlFrontend legacy;
    DteControlFrontend dedicated;
    DteControlFrontend aggregate;
    DteControlFrontend memory;
};
} // namespace

int RunDteControlCoreSelfTest() {
    int failures = 0;
    int checks = 0;
    auto Check = [&](bool condition, const std::string &message) {
        ++checks;
        std::cout << "  [" << (condition ? " ok " : "FAIL") << "] "
                  << message << "\n";
        if (!condition)
            ++failures;
    };

    CoreHWConfig legacy_default = nlohmann::json{{"id", 0}};
    Check(legacy_default.control_cores.mode ==
                  ControlCoreMode::LEGACY_SHARED &&
              legacy_default.control_cores.dte.command_queue_depth == 16 &&
              legacy_default.control_cores.dte.dispatch_width == 1 &&
              legacy_default.control_cores.dte.dispatch_latency_ns == 0 &&
              legacy_default.control_cores.dte
                      .completion_notify_latency_ns == 0,
          "hardware JSON keeps regression-safe shared-controller defaults");

    CoreHWConfig dedicated = nlohmann::json{
        {"id", 3},
        {"control_cores",
         {{"mode", "dual_dte_dedicated"},
          {"dte",
           {{"command_queue_depth", 8},
            {"dispatch_width", 2},
            {"dispatch_latency_ns", 4},
            {"completion_notify_latency_ns", 5}}}}}};
    Check(dedicated.control_cores.mode ==
                  ControlCoreMode::DUAL_DTE_DEDICATED &&
              dedicated.control_cores.dte.command_queue_depth == 8 &&
              dedicated.control_cores.dte.dispatch_width == 2 &&
              dedicated.control_cores.dte.dispatch_latency_ns == 4 &&
              dedicated.control_cores.dte.completion_notify_latency_ns == 5,
          "hardware JSON parses dedicated-controller resource parameters");

    const nlohmann::json platform = {
        {"x", 4},
        {"y", 1},
        {"memory", {{"sram_size", 33554432}}},
        {"control_cores",
         {{"mode", "dual_dte_dedicated"},
          {"dte",
           {{"command_queue_depth", 8},
            {"dispatch_width", 2},
            {"dispatch_latency_ns", 4},
            {"completion_notify_latency_ns", 5}}}}},
        {"cores",
         {{{"id", 0}},
          {{"id", 3},
           {"control_cores", {{"mode", "legacy_shared"}}}}}}};
    bool platform_parsed = true;
    try {
        ParseHardwareConfig(platform);
    } catch (const std::exception &error) {
        platform_parsed = false;
        std::cout << "  parser error: " << error.what() << "\n";
    }
    const bool inherited =
        platform_parsed && g_core_hw_config.size() == 4 &&
        g_core_hw_config[0].second->control_cores.mode ==
            ControlCoreMode::DUAL_DTE_DEDICATED &&
        g_core_hw_config[1].second->control_cores.mode ==
            ControlCoreMode::DUAL_DTE_DEDICATED &&
        g_core_hw_config[2].second->control_cores.dte.dispatch_width == 2 &&
        g_core_hw_config[2].second->control_cores.dte.dispatch_latency_ns == 4;
    Check(inherited,
          "top-level controller defaults propagate through auto-filled cores");
    Check(platform_parsed && g_core_hw_config.size() == 4 &&
              g_core_hw_config[3].second->control_cores.mode ==
                  ControlCoreMode::LEGACY_SHARED &&
              g_core_hw_config[3].second->control_cores.dte
                      .dispatch_latency_ns == 0,
          "per-core legacy mode overrides inherited dedicated resources");
    for (auto &entry : g_core_hw_config) delete entry.second;
    g_core_hw_config.clear();

    Check(ThrowsExpected<std::invalid_argument>([] {
              CoreHWConfig config = nlohmann::json{
                  {"id", 4},
                  {"control_cores", {{"mode", "unknown"}}}};
              (void)config;
          }),
          "hardware JSON rejects an unknown controller mode");
    Check(ThrowsExpected<std::invalid_argument>([] {
              CoreHWConfig config = nlohmann::json{
                  {"id", 5},
                  {"control_cores",
                   {{"mode", "dual_dte_dedicated"},
                    {"dte", {{"command_queue_depth", 0}}}}}};
              (void)config;
          }),
          "hardware JSON rejects a zero-depth command FIFO");
    Check(ThrowsExpected<std::invalid_argument>([] {
              CoreHWConfig config = nlohmann::json{
                  {"id", 6},
                  {"control_cores",
                   {{"mode", "dual_dte_dedicated"},
                    {"dte",
                     {{"command_queue_depth", 1},
                      {"dispatch_width", 2}}}}}};
              (void)config;
          }),
          "hardware JSON rejects dispatch width greater than FIFO depth");
    Check(ThrowsExpected<std::invalid_argument>([] {
              CoreHWConfig config = nlohmann::json{
                  {"id", 7},
                  {"control_cores",
                   {{"mode", "dual_dte_dedicated"},
                    {"dte", {{"dispatch_latency_ns", -1}}}}}};
              (void)config;
          }),
          "hardware JSON rejects a negative controller latency");
    Check(ThrowsExpected<std::invalid_argument>([] {
              CoreHWConfig config = nlohmann::json{
                  {"id", 8},
                  {"control_cores",
                   {{"mode", "legacy_shared"},
                    {"dte", {{"dispatch_latency_ns", 1}}}}}};
              (void)config;
          }),
          "hardware JSON rejects dedicated resources in legacy mode");

    Check(ThrowsExpected<std::invalid_argument>([] {
              DteControlCoreConfig config;
              config.command_queue_depth = 0;
              DteControlCore::ValidateConfig(config);
          }),
          "configuration rejects a zero-depth command FIFO");
    Check(ThrowsExpected<std::invalid_argument>([] {
              DteControlCoreConfig config;
              config.command_queue_depth = 1;
              config.dispatch_width = 2;
              DteControlCore::ValidateConfig(config);
          }),
          "configuration rejects dispatch width greater than FIFO depth");
    Check(DteControlCore::NormalizeLatency(sc_time(1, SC_NS)) ==
                  sc_time(2, SC_NS) &&
              DteControlCore::NormalizeLatency(sc_time(2, SC_NS)) ==
                  sc_time(2, SC_NS) &&
              DteControlCore::NormalizeLatency(sc_time(3, SC_NS)) ==
                  sc_time(4, SC_NS),
          "1/2/3 ns controller latencies ceil to 2/2/4 ns cycles");

    DteControlProbe probe("dte_control_probe");
    sc_start();
    failures += probe.failures;
    checks += probe.checks;
    std::cout << "DTE control-core self-test: "
              << (failures == 0 ? "PASS" : "FAIL") << " ("
              << checks - failures << "/" << checks << " checks)\n";
    return failures;
}
