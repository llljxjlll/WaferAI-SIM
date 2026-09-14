#include "memory/behavioral_hbm_backend.h"
#include "memory/external_dma_program.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace em = external_memory;
using Json = nlohmann::json;

namespace {

struct Driver : sc_core::sc_module {
    SC_HAS_PROCESS(Driver);
    Driver(sc_core::sc_module_name name,
           em::ExternalDmaProgramExecutor &executor)
        : sc_core::sc_module(name), executor(executor) { SC_THREAD(Run); }
    void Run() {
        execution = executor.Wait();
        sc_core::sc_stop();
    }
    em::ExternalDmaProgramExecutor &executor;
    em::ExternalDmaProgramExecution execution;
};

void WriteReport(const std::string &path,
                 const std::string &action_graph_digest,
                 const em::ExternalDmaProgramExecution &execution) {
    const bool probes = std::all_of(
        execution.probes.begin(), execution.probes.end(),
        [](const em::DmaProbeResult &item) { return item.matched; });
    const auto &stats = execution.stats;
    const Json report = {
        {"schema_version", "npusim.external_dma_runtime_report/v1alpha1"},
        {"action_graph_digest", action_graph_digest},
        {"program_ref", execution.program_ref},
        {"completed", execution.completed},
        {"pending_requests", execution.pending_requests},
        {"submitted_requests", stats.submitted_requests},
        {"completed_requests", stats.completed_requests},
        {"failed_requests", stats.failed_requests},
        {"external_read_bytes", stats.external_read_bytes},
        {"external_write_bytes", stats.external_write_bytes},
        {"hbm_read_bytes", stats.hbm_read_bytes},
        {"hbm_write_bytes", stats.hbm_write_bytes},
        {"all_probes_matched", probes},
    };
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot open runtime report");
    output << report.dump() << '\n';
    if (!output) throw std::runtime_error("cannot write runtime report");
}

} // namespace

int sc_main(int argc, char **argv) {
    try {
        if (argc != 9)
            throw std::invalid_argument(
                "usage: runner PROGRAM REPORT ACTION_GRAPH CASE REQUEST GRAPH MEMORY OFFLOAD");
        const em::ExternalDmaExpectedSource expected{
            argv[4], argv[5], argv[6], argv[7], argv[8]};
        em::ExternalDmaProgram program =
            em::LoadExternalDmaProgram(argv[1], expected);
        BehavioralHBMBackendConfig config;
        config.bandwidth_GBps = 64.0;
        config.efficiency = 1.0;
        config.base_latency = sc_core::sc_time(1, sc_core::SC_NS);
        std::map<em::HbmEndpoint,
                 std::unique_ptr<BehavioralHBMBackend>> owned;
        std::map<em::HbmEndpoint, HBMBackend *> endpoints;
        for (const auto &binding : program.backend_bindings) {
            const em::HbmEndpoint endpoint{
                binding.stack_id, binding.channel_id};
            if (owned.count(endpoint) != 0)
                throw std::invalid_argument("duplicate backend endpoint");
            auto backend = std::make_unique<BehavioralHBMBackend>(config);
            endpoints.emplace(endpoint, backend.get());
            owned.emplace(endpoint, std::move(backend));
        }
        em::ExternalDmaProgramExecutor executor(
            "external_dma_action_executor", std::move(program),
            endpoints, sc_core::sc_time(1, sc_core::SC_NS));
        Driver driver("external_dma_action_driver", executor);
        sc_core::sc_start();
        WriteReport(argv[2], argv[3], driver.execution);
        if (!driver.execution.completed || !driver.execution.error.empty() ||
            driver.execution.pending_requests != 0)
            throw std::runtime_error(
                "DMA action execution failed or did not drain: " +
                driver.execution.error);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "external DMA action runner: FAIL: "
                  << error.what() << std::endl;
        return 1;
    }
}
