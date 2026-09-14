#pragma once

#include "memory/external_memory_runtime.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <systemc>
#include <utility>
#include <vector>

namespace external_memory {

inline constexpr const char *kExternalDmaProgramSchemaVersion =
    "wafer_frontend.external_dma_program/v1alpha1";
inline constexpr const char *kExternalDmaRuntimeBindingSchemaVersion =
    "wafer_frontend.external_dma_runtime_binding/v1alpha2";

enum class ExternalDmaRuntimePhaseMode {
    kExecuteAllBeforeCompute,
    kBringInThenFinalWriteback,
};

struct DmaBackendBinding {
    std::string id;
    std::string hbm_capacity_ref;
    uint64_t owner_die_id = 0;
    uint64_t stack_id = 0;
    uint64_t channel_id = 0;
};

struct DmaDescriptor {
    std::string id;
    uint64_t sequence = 0;
    std::string operation_ref;
    std::string source_transfer_request_ref;
    std::vector<std::string> source_operation_deps;
    std::vector<std::string> depends_on;
    std::string request_schema_version;
    std::string connection_ref;
    TransferDirection direction = TransferDirection::kExternalToHbm;
    uint64_t external_address = 0;
    uint64_t hbm_address = 0;
    uint64_t size_bytes = 0;
    uint64_t planned_issue_cycle = 0;
    uint64_t planned_ready_cycle = 0;
};

struct DmaExternalSeed {
    std::string id;
    std::string external_capacity_ref;
    uint64_t address = 0;
    std::vector<uint8_t> payload;
};

struct DmaExternalProbe {
    std::string id;
    std::string external_capacity_ref;
    uint64_t address = 0;
    std::vector<uint8_t> expected_payload;
};

struct ExternalDmaProgram {
    std::string schema_version;
    std::string producer_pass;
    std::string id;
    std::string case_digest;
    std::string request_digest;
    std::string logical_graph_digest;
    std::string source_memory_plan_digest;
    std::string blocking_offload_plan_id;
    std::string blocking_offload_plan_digest;
    FabricConfig fabric;
    std::vector<DmaBackendBinding> backend_bindings;
    std::vector<DmaDescriptor> descriptors;
    std::vector<DmaExternalSeed> external_seeds;
    std::vector<DmaExternalProbe> external_probes;
};

struct ExternalDmaExpectedSource {
    std::string case_digest;
    std::string request_digest;
    std::string logical_graph_digest;
    std::string source_memory_plan_digest;
    std::string blocking_offload_plan_digest;
};

struct ExternalDmaRuntimeBinding {
    std::string id;
    std::string action_graph_digest;
    std::filesystem::path program_relative_path;
    ExternalDmaRuntimePhaseMode phase_mode =
        ExternalDmaRuntimePhaseMode::kExecuteAllBeforeCompute;
    ExternalDmaExpectedSource expected_source;
};

ExternalDmaRuntimeBinding LoadExternalDmaRuntimeBinding(
    const std::filesystem::path &path);

ExternalDmaProgram ParseExternalDmaProgram(
    std::string_view json,
    const ExternalDmaExpectedSource &expected_source);
ExternalDmaProgram LoadExternalDmaProgram(
    const std::filesystem::path &path,
    const ExternalDmaExpectedSource &expected_source);

using HbmEndpoint = std::pair<uint64_t, uint64_t>;

struct DmaProbeResult {
    std::string probe_ref;
    std::vector<uint8_t> payload;
    bool matched = false;
};

struct ExternalDmaProgramExecution {
    std::string program_ref;
    std::vector<RuntimeCompletion> completions;
    std::vector<DmaProbeResult> probes;
    RuntimeStats stats;
    uint64_t pending_requests = 0;
    bool completed = false;
    std::string error;
};

class ExternalDmaProgramExecutor : public sc_core::sc_module {
public:
    SC_HAS_PROCESS(ExternalDmaProgramExecutor);

    ExternalDmaProgramExecutor(
        const sc_core::sc_module_name &name,
        ExternalDmaProgram program,
        std::map<HbmEndpoint, HBMBackend *> hbm_backends,
        sc_core::sc_time cycle_time,
        ExternalDmaRuntimePhaseMode phase_mode =
            ExternalDmaRuntimePhaseMode::kExecuteAllBeforeCompute);
    ~ExternalDmaProgramExecutor() override;

    std::optional<ExternalDmaProgramExecution> Poll() const;
    ExternalDmaProgramExecution Wait();
    RuntimeStats WaitForBringIn();
    void ReleaseFinalWriteback();
    const std::string &ProgramRef() const noexcept { return program_.id; }
    const sc_core::sc_event &CompletionEvent() const;

private:
    void Run();

    ExternalDmaProgram program_;
    std::unique_ptr<ExternalMemoryRuntimeBridge> runtime_;
    std::optional<ExternalDmaProgramExecution> execution_;
    sc_core::sc_event done_;
    sc_core::sc_event bring_in_ready_event_;
    sc_core::sc_event final_writeback_gate_;
    sc_core::sc_time cycle_time_;
    ExternalDmaRuntimePhaseMode phase_mode_;
    RuntimeStats bring_in_stats_;
    bool bring_in_ready_ = false;
    bool final_writeback_released_ = false;
    std::string phase_error_;
};

} // namespace external_memory
