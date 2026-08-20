#pragma once

#include "memory/sram/sram_access_unit.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

class HBMRuntime;

namespace frontend::program_io {

inline constexpr std::string_view kSchemaVersion =
    "wafer_frontend.program_io_contract/v1alpha2";

class Error : public std::invalid_argument {
public:
    explicit Error(const std::string &message) : std::invalid_argument(message) {}
};

enum class Mode : uint8_t { TIMING = 0, FUNCTIONAL = 1 };
enum class Purpose : uint8_t {
    ACTIVATION = 0,
    WEIGHT = 1,
    STATE = 2,
    TIMING_PARTIAL = 3,
};
enum class TargetKind : uint8_t { SRAM = 0, HBM = 1 };
enum class DType : uint8_t { FP16 = 0, FP32 = 1, INT32 = 2 };

struct TensorSlice {
    std::string value_id;
    std::vector<uint64_t> offset;
    std::vector<uint64_t> shape;
};

struct Blob {
    std::string id;
    std::vector<uint8_t> bytes;
    std::string sha256;
};

struct SramTarget {
    TargetKind kind = TargetKind::SRAM;
    uint32_t runtime_core_id = 0;
    std::string program_symbol_ref;
    uint64_t finalized_symbol_index = 0;
    std::string expected_symbol_name;
    std::string buffer_abi_id;
    std::string storage_id;
    std::string value_id;
    TensorSlice tensor_slice;
    DType dtype = DType::FP16;
    std::string layout;
};

struct HbmTarget {
    TargetKind kind = TargetKind::HBM;
    std::string program_symbol_ref;
    uint64_t finalized_symbol_index = 0;
    std::string expected_symbol_name;
    std::string state_abi_id;
    std::string state_ref;
    std::string hbm_binding_ref;
};

using Target = std::variant<SramTarget, HbmTarget>;

struct Initialization {
    std::string id;
    Target target;
    uint64_t offset_bytes = 0;
    uint64_t length_bytes = 0;
    std::string blob_ref;
    Purpose purpose = Purpose::ACTIVATION;
};

struct OutputProbe {
    std::string id;
    Target target;
    uint64_t offset_bytes = 0;
    uint64_t length_bytes = 0;
    std::string blob_ref;
};

struct Contract {
    std::string schema_version;
    std::string producer_pass;
    std::string id;
    Mode mode = Mode::TIMING;
    std::string source_linked_manifest_id;
    std::string source_linked_manifest_digest;
    std::string program_artifact_sha256;
    std::vector<Blob> blobs;
    std::vector<Initialization> initializations;
    std::vector<OutputProbe> output_probes;
};

struct ResolvedHbmRange {
    uint64_t die_id = 0;
    uint64_t address_bytes = 0;
    uint64_t size_bytes = 0;
};

struct ResolvedInitialization {
    Initialization source;
    std::string region_symbol_ref;
    std::string region_name;
    uint64_t region_base_bytes = 0;
    uint64_t region_size_bytes = 0;
    uint64_t absolute_address_bytes = 0;
    std::vector<uint8_t> bytes;
    std::optional<ResolvedHbmRange> hbm_range;
};

struct ResolvedOutputProbe {
    OutputProbe source;
    std::string region_symbol_ref;
    std::string region_name;
    uint64_t region_base_bytes = 0;
    uint64_t region_size_bytes = 0;
    uint64_t absolute_address_bytes = 0;
    std::vector<uint8_t> expected_bytes;
    std::optional<ResolvedHbmRange> hbm_range;
};

struct ResolvedContract {
    std::string id;
    Mode mode = Mode::TIMING;
    std::string source_linked_manifest_id;
    std::string source_linked_manifest_digest;
    std::string program_artifact_sha256;
    std::vector<ResolvedInitialization> initializations;
    std::vector<ResolvedOutputProbe> output_probes;
};

// Parsing is strict: unknown/duplicate fields, unstable content-addressed IDs,
// non-canonical base64, and inconsistent SHA-256 values are rejected.
Contract Parse(std::string_view sidecar_json);
Contract Load(const std::filesystem::path &path);

// Resolve binds the sidecar to the exact linked manifest and exact encoded
// ProgramArtifact. It reuses ProgramArtifactFinalizer validation but never
// changes finalization/lowering choices.
ResolvedContract Resolve(const Contract &contract,
                         std::string_view linked_manifest_json,
                         const std::vector<uint8_t> &artifact_bytes);
ResolvedContract ParseAndResolve(std::string_view sidecar_json,
                                 std::string_view linked_manifest_json,
                                 const std::vector<uint8_t> &artifact_bytes);

struct Bindings {
    std::map<uint32_t, sram::AccessUnit *> sram_by_runtime_core;
    HBMRuntime *hbm_runtime = nullptr;
};

struct Applied {
    ResolvedContract contract;
    Bindings bindings;
};

struct ProbeResult {
    std::string probe_id;
    uint32_t runtime_core_id = 0;
    uint64_t absolute_address_bytes = 0;
    uint64_t length_bytes = 0;
    std::string expected_sha256;
    std::string actual_sha256;
    bool exact_match = false;
    bool all_bytes_valid = false;
};

struct Result {
    std::string contract_id;
    std::vector<ProbeResult> probes;

    bool Passed() const noexcept;
};

// These are boundary-only transactions. AccessUnit itself rejects use while
// SystemC is running; failures restore every previously touched byte and its
// validity bitmap before rethrowing.
Applied ApplyBeforeSimulation(const ResolvedContract &contract,
                              const Bindings &bindings);
Result VerifyAfterSimulation(const Applied &applied);

std::string Sha256Hex(std::string_view bytes);
std::string Sha256Hex(const std::vector<uint8_t> &bytes);

} // namespace frontend::program_io
