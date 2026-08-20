#pragma once

#include "isa/record_codec.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

inline constexpr std::size_t kProgramHeaderSize = 64;
inline constexpr std::size_t kProgramSectionDescriptorSize = 40;
inline constexpr std::size_t kProgramWholeFileCrcOffset = 56;
inline constexpr uint16_t kProgramFormatMajor = 1;
inline constexpr uint16_t kProgramFormatMinor = 0;
inline constexpr uint16_t kProgramIsaMajor = 1;
inline constexpr uint16_t kProgramIsaMinor = 4;
inline constexpr uint8_t kProgramLittleEndian = 1;
inline constexpr uint32_t kProgramSectionRequired = 1u << 0;

inline constexpr uint64_t kMaxProgramFileBytes = uint64_t{64} << 20;
inline constexpr uint32_t kMaxProgramSections = 64;
inline constexpr uint32_t kMaxProgramStrings = 65536;
inline constexpr uint32_t kMaxProgramStringBytes = 255;
inline constexpr uint64_t kMaxProgramStringTableBytes = uint64_t{16} << 20;
inline constexpr uint32_t kMaxProgramSymbols = 1u << 20;
inline constexpr uint32_t kMaxProgramRelocations = 1u << 20;
inline constexpr uint32_t kMaxProgramCoreGroups = 65536;
inline constexpr uint32_t kMaxProgramCores = 65536;
inline constexpr uint32_t kMaxProgramRecords = 1u << 20;

class ProgramFormatError : public std::invalid_argument {
public:
    explicit ProgramFormatError(const std::string &message)
        : std::invalid_argument(message) {}
};

enum class ProgramSectionType : uint32_t {
    STRING_TABLE = 1,
    SYMBOL_TABLE = 2,
    SEMANTIC_RELOCATION_TABLE = 3,
    CORE_GROUP_TABLE = 4,
    CORE_PROGRAM_INDEX = 5,
    EXTERNAL_RECORD_STREAM = 6,
    CONTROL_ENVELOPE = 7,
};

enum class ProgramSymbolKind : uint8_t {
    ABSOLUTE_ADDRESS = 1,
    SRAM_REGION = 2,
    SRAM_LABEL = 3,
};

enum class SemanticRelocationKind : uint8_t {
    ABSOLUTE_ADDRESS = 1,
    SRAM_REGION = 2,
    SRAM_LABEL = 3,
};

enum class SemanticOperandId : uint16_t {
    COMPUTE_INPUT_ADDRESS = 1,
    COMPUTE_DATA_ADDRESS = 2,
    COMPUTE_OUTPUT_ADDRESS = 3,
    SOURCE_ADDRESS = 4,
    DESTINATION_ADDRESS = 5,
    HBM_ADDRESS = 6,
    SYMBOL = 7,
    REGION_NAME = 8,
    LABEL_SYMBOL = 9,
    OLD_SYMBOL = 10,
    NEW_SYMBOL = 11,
    COMPUTE_AUX_ADDRESS = 12,
    SRAM_BIND_INPUT_0 = 0x100,
    SRAM_BIND_INPUT_15 = 0x10f,
    SRAM_BIND_OUTPUT = 0x110,
};

enum class EmptyCoreAckPolicy : uint8_t {
    EXCLUDE_EMPTY = 0,
    INCLUDE_EMPTY = 1,
};

enum class ProgramFailurePolicy : uint8_t {
    ABORT_ALL = 0,
};

// Numeric members remain wide until validation so encode callers receive a
// deterministic range error rather than implicit ABI narrowing.
struct ProgramSymbol {
    uint64_t name_string_index = 0;
    ProgramSymbolKind kind = ProgramSymbolKind::ABSOLUTE_ADDRESS;
    uint64_t flags = 0;
    uint64_t value = 0;
    uint64_t size_bytes = 0;
};

struct SemanticRelocation {
    uint64_t core_index = 0;
    uint64_t instruction_index = 0;
    uint64_t operand_id =
        static_cast<uint16_t>(SemanticOperandId::COMPUTE_INPUT_ADDRESS);
    SemanticRelocationKind kind =
        SemanticRelocationKind::ABSOLUTE_ADDRESS;
    uint64_t symbol_index = 0;
    int64_t addend = 0;
};

struct ProgramCoreGroup {
    uint64_t group_id = 1;
    std::vector<uint64_t> members;
};

struct ProgramCore {
    uint64_t core_id = 0;
    std::vector<ExternalRecord> records;
};

struct ProgramStartEvent {
    uint64_t target_core = 0;
    uint64_t tag = 0;
    // Host START-message / logical-contribution multiplicity for this key.
    // Zero is never meaningful.
    uint64_t count = 1;
};

struct ProgramControlEnvelope {
    std::vector<uint64_t> active_cores;
    std::vector<ProgramStartEvent> start_events;
    std::vector<uint64_t> terminal_cores;
    std::vector<uint64_t> expected_ack_cores;
    std::vector<uint64_t> expected_done_cores;
    EmptyCoreAckPolicy empty_core_ack_policy =
        EmptyCoreAckPolicy::INCLUDE_EMPTY;
    ProgramFailurePolicy failure_policy = ProgramFailurePolicy::ABORT_ALL;
};

struct ProgramArtifact {
    uint64_t capabilities = 0;
    std::vector<std::string> strings;
    std::vector<ProgramSymbol> symbols;
    // config_helper_program applies these semantic relocations only after the
    // complete artifact has passed every cross-section check.
    std::vector<SemanticRelocation> relocations;
    std::vector<ProgramCoreGroup> core_groups;
    std::vector<ProgramCore> cores;
    ProgramControlEnvelope envelope;
};

// CRC-32C (Castagnoli), initial/final XOR 0xFFFFFFFF. For the whole artifact,
// bytes [kProgramWholeFileCrcOffset, +4) are treated as zero.
uint32_t ProgramCrc32c(const uint8_t *data, std::size_t size) noexcept;
uint32_t ProgramCrc32c(const std::vector<uint8_t> &bytes) noexcept;

// Performs all cross-section checks without mutating global simulator state.
void ValidateProgramArtifact(const ProgramArtifact &artifact);

// Canonical encoding always places the section table immediately after the
// 64-byte header and emits the seven required sections without gaps.
std::vector<uint8_t> EncodeProgramArtifact(const ProgramArtifact &artifact);

// Parses into temporary state, validates the complete object and only then
// returns it. No partially decoded artifact is observable on failure.
ProgramArtifact DecodeProgramArtifact(const std::vector<uint8_t> &bytes);
