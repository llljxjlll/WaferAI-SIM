#include "frontend/program_finalizer.h"

#include "nlohmann/json.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;
using frontend::LinkedProgramManifestDto;
using frontend::LogicalCoreDto;
using frontend::ProgramArtifactFinalizer;
using frontend::RuntimeSymbolKindDto;

struct Options {
    std::filesystem::path input;
    std::optional<std::filesystem::path> output;
    std::optional<std::filesystem::path> report;
    bool validate_only = false;
};

[[noreturn]] void UsageError(const std::string &message) {
    throw std::invalid_argument(
        message +
        "\nusage: npusim_program_finalizer --input linked.json "
        "[--output program.npup] [--report report.json] [--validate-only]");
}

Options ParseOptions(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&](const char *name) -> std::filesystem::path {
            if (++index >= argc)
                UsageError(std::string(name) + " requires a path");
            return argv[index];
        };
        if (argument == "--input") {
            if (!options.input.empty()) UsageError("duplicate --input");
            options.input = value("--input");
        } else if (argument == "--output") {
            if (options.output) UsageError("duplicate --output");
            options.output = value("--output");
        } else if (argument == "--report") {
            if (options.report) UsageError("duplicate --report");
            options.report = value("--report");
        } else if (argument == "--validate-only") {
            if (options.validate_only) UsageError("duplicate --validate-only");
            options.validate_only = true;
        } else if (argument == "--help") {
            std::cout
                << "usage: npusim_program_finalizer --input linked.json "
                   "[--output program.npup] [--report report.json] "
                   "[--validate-only]\n";
            std::exit(0);
        } else {
            UsageError("unknown argument '" + argument + "'");
        }
    }
    if (options.input.empty()) UsageError("--input is required");
    if (options.validate_only && options.output)
        UsageError("--validate-only cannot be combined with --output");
    if (!options.validate_only && !options.output)
        UsageError("--output is required unless --validate-only is set");
    if (options.output && options.report &&
        *options.output == *options.report)
        UsageError("--output and --report must be different paths");
    return options;
}

std::string ReadText(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot open input '" + path.string() + "'");
    std::ostringstream result;
    result << input.rdbuf();
    if (!input.good() && !input.eof())
        throw std::runtime_error("failed to read input '" + path.string() + "'");
    return result.str();
}

void WriteAtomic(const std::filesystem::path &path, const uint8_t *data,
                 std::size_t size) {
    if (path.empty() || path.filename().empty())
        throw std::runtime_error("output path must name a file");
    const auto nonce = std::chrono::steady_clock::now()
                           .time_since_epoch().count();
    std::filesystem::path temporary = path;
    temporary += ".tmp." + std::to_string(nonce);
    try {
        {
            std::ofstream output(temporary,
                                 std::ios::binary | std::ios::trunc);
            if (!output)
                throw std::runtime_error("cannot create temporary output '" +
                                         temporary.string() + "'");
            output.write(reinterpret_cast<const char *>(data),
                         static_cast<std::streamsize>(size));
            output.flush();
            if (!output)
                throw std::runtime_error("failed to write temporary output '" +
                                         temporary.string() + "'");
        }
        std::error_code error;
        std::filesystem::rename(temporary, path, error);
        if (error)
            throw std::runtime_error("atomic rename to '" + path.string() +
                                     "' failed: " + error.message());
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

void WriteAtomic(const std::filesystem::path &path,
                 const std::vector<uint8_t> &bytes) {
    WriteAtomic(path, bytes.data(), bytes.size());
}

void WriteAtomic(const std::filesystem::path &path,
                 const std::string &text) {
    WriteAtomic(path, reinterpret_cast<const uint8_t *>(text.data()),
                text.size());
}

const char *RuntimeKindName(RuntimeSymbolKindDto kind) {
    switch (kind) {
    case RuntimeSymbolKindDto::START_TAG: return "start_tag";
    case RuntimeSymbolKindDto::EVENT_TAG: return "event_tag";
    case RuntimeSymbolKindDto::DTE_TOKEN: return "dte_token";
    case RuntimeSymbolKindDto::DTE_FSM: return "dte_fsm";
    case RuntimeSymbolKindDto::GROUP: return "group";
    case RuntimeSymbolKindDto::RUNTIME_CORE: return "runtime_core";
    }
    throw std::runtime_error("unknown runtime symbol kind");
}

Json RuntimeAssignments(const LinkedProgramManifestDto &manifest) {
    std::map<LogicalCoreDto, uint64_t> runtime_cores;
    for (const frontend::CoreRuntimeBindingDto &binding :
         manifest.core_bindings)
        runtime_cores.emplace(binding.logical_core, binding.runtime_core_id);
    std::map<RuntimeSymbolKindDto, uint64_t> next{{
        {RuntimeSymbolKindDto::START_TAG, 0},
        {RuntimeSymbolKindDto::EVENT_TAG, 0},
        {RuntimeSymbolKindDto::DTE_TOKEN, 1},
        {RuntimeSymbolKindDto::DTE_FSM, 1},
        {RuntimeSymbolKindDto::GROUP, 1},
    }};
    Json assignments = Json::array();
    for (const frontend::RuntimeSymbolDefinitionDto &definition :
         manifest.runtime_symbol_definitions) {
        uint64_t value = 0;
        if (definition.symbol.kind == RuntimeSymbolKindDto::RUNTIME_CORE) {
            value = runtime_cores.at(definition.logical_cores.at(0));
        } else {
            value = next.at(definition.symbol.kind)++;
        }
        assignments.push_back({
            {"symbol_id", definition.symbol.id},
            {"kind", RuntimeKindName(definition.symbol.kind)},
            {"value", value},
        });
    }
    return assignments;
}

Json BuildReport(const std::string &input,
                 const LinkedProgramManifestDto &manifest,
                 const std::vector<uint8_t> &encoded) {
    const ProgramArtifact artifact = DecodeProgramArtifact(encoded);
    ValidateProgramArtifact(artifact);
    std::size_t record_count = 0;
    for (const ProgramCore &core : artifact.cores)
        record_count += core.records.size();
    return {
        {"schema_version", "npusim.program_finalization_report/v1alpha1"},
        {"linked_manifest_id", manifest.id},
        {"linked_manifest_digest",
         ProgramArtifactFinalizer::CanonicalManifestDigest(input)},
        {"artifact_sha256",
         ProgramArtifactFinalizer::EncodedArtifactDigest(encoded)},
        {"artifact_bytes", encoded.size()},
        {"core_count", artifact.cores.size()},
        {"record_count", record_count},
        {"relocation_count", artifact.relocations.size()},
        {"runtime_assignments", RuntimeAssignments(manifest)},
    };
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        const std::string input = ReadText(options.input);
        const LinkedProgramManifestDto manifest =
            ProgramArtifactFinalizer::Parse(input);
        const ProgramArtifactFinalizer finalizer;
        const std::vector<uint8_t> encoded =
            finalizer.FinalizeEncoded(input);
        const Json report = BuildReport(input, manifest, encoded);
        if (options.output) WriteAtomic(*options.output, encoded);
        if (options.report)
            WriteAtomic(*options.report, report.dump(2) + "\n");
        if (options.validate_only && !options.report)
            std::cout << report.dump() << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "npusim_program_finalizer: " << error.what() << '\n';
        return 1;
    }
}
