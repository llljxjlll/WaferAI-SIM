#include "memory/behavioral_hbm_backend.h"
#include "memory/external_dma_program.h"

#include <fstream>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace em = external_memory;
using namespace sc_core;

namespace {

const em::ExternalDmaExpectedSource kExpected{
    "b2471dd0f579303427c04ee7c1b2a9fd49feb72653bae602bbbb0186cd91f194",
    "0c367df9258041a617a7b8596e74799a8eaf9862bbcf717c69320da62963d82f",
    "b1b92f0891cf5b46eadb3122750d0bf0e2a6c0fbb905ba681b502e502ad468b7",
    "8ae2b32a5d0991d783068ef21094805369803b5033d62e4531a15a046acb1b8c",
    "94801b295669ee70c3dc70dbaa0d812a6b7703a5181a4b6b7430172d4d0ad102"};

void Check(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

std::string Read(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read golden " + path);
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

void Write(const std::filesystem::path &path, const std::string &content) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot write " + path.string());
    output << content;
    if (!output) throw std::runtime_error("failed writing " + path.string());
}

std::string ReplaceOnce(
    std::string source, const std::string &before,
    const std::string &after) {
    const auto position = source.find(before);
    if (position == std::string::npos)
        throw std::runtime_error("selftest replacement source is absent");
    source.replace(position, before.size(), after);
    return source;
}

template <typename Operation>
void Rejects(Operation operation, const std::string &message) {
    try {
        operation();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error(message);
}

struct Driver : sc_module {
    SC_HAS_PROCESS(Driver);

    Driver(sc_module_name name, em::ExternalDmaProgramExecutor &executor)
        : sc_module(name), executor(executor) {
        SC_THREAD(Run);
    }

    void Run() {
        try {
            Check(!executor.Poll().has_value(),
                  "program completed before its runtime event");
            const auto result = executor.Wait();
            Check(result.completed && result.error.empty(),
                  "program execution failed: " + result.error);
            Check(
                result.program_ref ==
                    "external_dma_program_0a1010fc2d6460dc",
                "program identity differs from Python golden");
            Check(result.completions.size() == 2,
                  "program did not execute both lowered transfers");
            Check(
                result.completions[0].request_ref !=
                    result.completions[1].request_ref &&
                    result.completions[1].started_at >=
                        result.completions[0].completed_at,
                "blocking descriptor dependency was not enforced");
            Check(result.probes.size() == 1 &&
                      result.probes[0].matched &&
                      result.probes[0].payload ==
                          std::vector<uint8_t>(
                              {1, 7, 0, 9, 13, 0, 255, 2,
                               8, 6, 7, 5, 3, 0, 9, 4}),
                  "external output probe differs from Python sidecar");
            Check(result.stats.submitted_requests == 2 &&
                      result.stats.completed_requests == 2 &&
                      result.stats.failed_requests == 0 &&
                      result.stats.external_read_bytes == 16 &&
                      result.stats.external_write_bytes == 16 &&
                      result.stats.hbm_read_bytes == 16 &&
                      result.stats.hbm_write_bytes == 16,
                  "cross-language runtime statistics are incorrect");
            passed = true;
        } catch (const std::exception &failure) {
            error = failure.what();
        }
        sc_stop();
    }

    em::ExternalDmaProgramExecutor &executor;
    bool passed = false;
    std::string error;
};

} // namespace

int sc_main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::runtime_error(
                "expected external DMA golden path");
        const std::string content = Read(argv[1]);
        const em::ExternalDmaProgram program =
            em::LoadExternalDmaProgram(argv[1], kExpected);
        Check(program.descriptors.size() == 2,
              "Python golden descriptor count is incorrect");
        em::ExternalDmaExpectedSource wrong = kExpected;
        wrong.case_digest =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        Rejects(
            [&] { em::ParseExternalDmaProgram(content, wrong); },
            "cross-case DMA program was accepted");
        std::string unknown = content;
        const auto insertion = unknown.find('{') + 1;
        unknown.insert(insertion, "\"unknown\":0,");
        Rejects(
            [&] {
                em::ParseExternalDmaProgram(unknown, kExpected);
            },
            "unknown DMA program field was accepted");

        const std::filesystem::path binding_path =
            std::filesystem::temp_directory_path() /
            "npusim_external_dma_runtime_binding_selftest.json";
        const std::string binding =
            "{"
            "\"schema_version\":\"wafer_frontend.external_dma_runtime_binding/v1alpha1\","
            "\"id\":\"external_dma_runtime_binding_selftest\","
            "\"action_graph_digest\":\"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd\","
            "\"program_relative_path\":\"artifacts/external_dma_program.json\","
            "\"case_digest\":\"" + kExpected.case_digest + "\","
            "\"request_digest\":\"" + kExpected.request_digest + "\","
            "\"logical_graph_digest\":\"" + kExpected.logical_graph_digest + "\","
            "\"source_memory_plan_digest\":\"" +
                kExpected.source_memory_plan_digest + "\","
            "\"blocking_offload_plan_digest\":\"" +
                kExpected.blocking_offload_plan_digest + "\"}";
        Write(binding_path, binding);
        const auto runtime_binding =
            em::LoadExternalDmaRuntimeBinding(binding_path);
        Check(
            runtime_binding.id == "external_dma_runtime_binding_selftest" &&
                runtime_binding.action_graph_digest == std::string(64, 'd') &&
                runtime_binding.program_relative_path ==
                    "artifacts/external_dma_program.json" &&
                runtime_binding.expected_source.case_digest ==
                    kExpected.case_digest &&
                runtime_binding.expected_source.request_digest ==
                    kExpected.request_digest &&
                runtime_binding.expected_source.logical_graph_digest ==
                    kExpected.logical_graph_digest &&
                runtime_binding.expected_source.source_memory_plan_digest ==
                    kExpected.source_memory_plan_digest &&
                runtime_binding.expected_source.blocking_offload_plan_digest ==
                    kExpected.blocking_offload_plan_digest,
            "external DMA runtime binding fields changed during load");

        Write(binding_path, ReplaceOnce(binding, "{", "{\"unknown\":0,"));
        Rejects(
            [&] { em::LoadExternalDmaRuntimeBinding(binding_path); },
            "unknown runtime binding field was accepted");
        Write(
            binding_path,
            ReplaceOnce(
                binding,
                "\"request_digest\":",
                "\"request_digest\":\"" + kExpected.request_digest +
                    "\",\"request_digest\":"));
        Rejects(
            [&] { em::LoadExternalDmaRuntimeBinding(binding_path); },
            "duplicate runtime binding key was accepted");
        Write(
            binding_path,
            ReplaceOnce(
                binding,
                "wafer_frontend.external_dma_runtime_binding/v1alpha1",
                "wafer_frontend.external_dma_runtime_binding/v0"));
        Rejects(
            [&] { em::LoadExternalDmaRuntimeBinding(binding_path); },
            "wrong runtime binding schema was accepted");
        Write(
            binding_path,
            ReplaceOnce(
                binding, "artifacts/external_dma_program.json",
                "../external_dma_program.json"));
        Rejects(
            [&] { em::LoadExternalDmaRuntimeBinding(binding_path); },
            "traversing runtime binding program path was accepted");
        std::filesystem::remove(binding_path);

        BehavioralHBMBackendConfig config;
        config.bandwidth_GBps = 8.0;
        config.efficiency = 1.0;
        config.base_latency = sc_time(1, SC_NS);
        BehavioralHBMBackend backend(config);
        em::ExternalDmaProgramExecutor executor(
            "external_dma_program_executor", program,
            {{{0, 0}, &backend}}, sc_time(1, SC_NS));
        Driver driver("driver", executor);
        sc_start(1000, SC_NS);
        if (!driver.passed)
            throw std::runtime_error(driver.error);
        std::cout << "external DMA program selftest: PASS"
                  << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "external DMA program selftest: FAIL: "
                  << error.what() << std::endl;
        return 1;
    }
}
