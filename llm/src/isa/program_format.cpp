#include "isa/program_format.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

constexpr std::array<uint8_t, 8> kMagic{{'N', 'P', 'U', 'P', 'R', 'G', '1', 0}};
constexpr uint32_t kSymbolEntrySize = 24;
constexpr uint32_t kRelocationEntrySize = 24;
constexpr uint32_t kCoreIndexEntrySize = 32;
constexpr uint16_t kEnvelopeVersion = 1;
constexpr std::size_t kEnvelopeHeaderSize = 32;
constexpr std::size_t kStartEventSize = 12;
constexpr uint64_t kKnownCapabilities =
    CapabilityBit(IsaCapability::PD_CONTEXT) |
    CapabilityBit(IsaCapability::EXPERIMENTAL_FUSED);

[[noreturn]] void Fail(const std::string &message) {
    throw ProgramFormatError(message);
}

void Require(bool condition, const std::string &message) {
    if (!condition)
        Fail(message);
}

void RequireU16(uint64_t value, const std::string &field) {
    if (value > UINT16_MAX)
        Fail(field + " exceeds u16");
}

void RequireU32(uint64_t value, const std::string &field) {
    if (value > UINT32_MAX)
        Fail(field + " exceeds u32");
}

void AppendLe(std::vector<uint8_t> &bytes, uint64_t value,
              std::size_t width) {
    for (std::size_t i = 0; i < width; ++i)
        bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
}

void WriteLe(std::vector<uint8_t> &bytes, std::size_t offset, uint64_t value,
             std::size_t width) {
    Require(offset <= bytes.size() && width <= bytes.size() - offset,
            "internal program encoder write is out of bounds");
    for (std::size_t i = 0; i < width; ++i)
        bytes[offset + i] = static_cast<uint8_t>(value >> (8 * i));
}

uint64_t ReadLe(const std::vector<uint8_t> &bytes, std::size_t offset,
                std::size_t width, const std::string &field) {
    if (offset > bytes.size() || width > bytes.size() - offset)
        Fail("truncated " + field + " at file offset " +
             std::to_string(offset));
    uint64_t value = 0;
    for (std::size_t i = 0; i < width; ++i)
        value |= uint64_t{bytes[offset + i]} << (8 * i);
    return value;
}

void RequireZero(const std::vector<uint8_t> &bytes, std::size_t offset,
                 std::size_t width, const std::string &field) {
    if (offset > bytes.size() || width > bytes.size() - offset)
        Fail("truncated " + field);
    for (std::size_t i = 0; i < width; ++i) {
        if (bytes[offset + i] != 0)
            Fail(field + " reserved bytes must be zero");
    }
}

bool IsValidUtf8(const std::string &value) {
    std::size_t offset = 0;
    while (offset < value.size()) {
        const uint8_t first = static_cast<uint8_t>(value[offset]);
        if (first == 0)
            return false;
        if (first < 0x80) {
            ++offset;
            continue;
        }
        std::size_t length = 0;
        uint32_t codepoint = 0;
        uint32_t minimum = 0;
        if ((first & 0xe0) == 0xc0) {
            length = 2;
            codepoint = first & 0x1f;
            minimum = 0x80;
        } else if ((first & 0xf0) == 0xe0) {
            length = 3;
            codepoint = first & 0x0f;
            minimum = 0x800;
        } else if ((first & 0xf8) == 0xf0) {
            length = 4;
            codepoint = first & 0x07;
            minimum = 0x10000;
        } else {
            return false;
        }
        if (length > value.size() - offset)
            return false;
        for (std::size_t i = 1; i < length; ++i) {
            const uint8_t continuation =
                static_cast<uint8_t>(value[offset + i]);
            if ((continuation & 0xc0) != 0x80)
                return false;
            codepoint = (codepoint << 6) | (continuation & 0x3f);
        }
        if (codepoint < minimum || codepoint > 0x10ffff ||
            (codepoint >= 0xd800 && codepoint <= 0xdfff))
            return false;
        offset += length;
    }
    return true;
}

template <class Values>
void ValidateSortedUniqueU16(const Values &values, const std::string &field,
                             bool allow_empty) {
    Require(allow_empty || !values.empty(), field + " must not be empty");
    for (std::size_t i = 0; i < values.size(); ++i) {
        RequireU16(values[i], field);
        if (i != 0)
            Require(values[i - 1] < values[i],
                    field + " must be strictly increasing and unique");
    }
}

bool ContainsCore(const std::vector<uint64_t> &cores, uint64_t core) {
    return std::binary_search(cores.begin(), cores.end(), core);
}

bool ValidSymbolKind(ProgramSymbolKind kind) {
    const uint8_t raw = static_cast<uint8_t>(kind);
    return raw >= static_cast<uint8_t>(ProgramSymbolKind::ABSOLUTE_ADDRESS) &&
           raw <= static_cast<uint8_t>(ProgramSymbolKind::SRAM_LABEL);
}

bool ValidRelocationKind(SemanticRelocationKind kind) {
    const uint8_t raw = static_cast<uint8_t>(kind);
    return raw >=
               static_cast<uint8_t>(SemanticRelocationKind::ABSOLUTE_ADDRESS) &&
           raw <= static_cast<uint8_t>(SemanticRelocationKind::SRAM_LABEL);
}
bool ValidRelocationOperand(const ExternalRecord &record,
                            uint64_t operand_id) {
    const auto id = static_cast<SemanticOperandId>(operand_id);
    if (const auto *bind = std::get_if<SramBindOperands>(&record.operands)) {
        const uint64_t first = static_cast<uint16_t>(
            SemanticOperandId::SRAM_BIND_INPUT_0);
        const uint64_t last = static_cast<uint16_t>(
            SemanticOperandId::SRAM_BIND_INPUT_15);
        if (operand_id >= first && operand_id <= last)
            return operand_id - first < bind->input_count;
        return id == SemanticOperandId::SRAM_BIND_OUTPUT;
    }
    if (std::holds_alternative<ComputeOperands>(record.operands)) {
        return id == SemanticOperandId::COMPUTE_INPUT_ADDRESS ||
               id == SemanticOperandId::COMPUTE_DATA_ADDRESS ||
               id == SemanticOperandId::COMPUTE_OUTPUT_ADDRESS;
    }
    if (const auto *send = std::get_if<DteSendOperands>(&record.operands))
        return send->source_space == EndpointSourceSpace::HBM
                   ? id == SemanticOperandId::HBM_ADDRESS
                   : id == SemanticOperandId::SOURCE_ADDRESS;
    if (std::holds_alternative<DteRecvOperands>(record.operands))
        return id == SemanticOperandId::DESTINATION_ADDRESS;
    if (std::holds_alternative<ReduceComputeOperands>(record.operands)) {
        return id == SemanticOperandId::SOURCE_ADDRESS ||
               id == SemanticOperandId::DESTINATION_ADDRESS;
    }
    if (std::holds_alternative<LsuOperands>(record.operands)) {
        if (id == SemanticOperandId::HBM_ADDRESS)
            return true;
        return record.opcode == Opcode::LSU_LOAD
                   ? id == SemanticOperandId::DESTINATION_ADDRESS
                   : id == SemanticOperandId::SOURCE_ADDRESS;
    }
    if (const auto *issue =
            std::get_if<DteIssueOperands>(&record.operands)) {
        switch (issue->direction) {
        case LocalDteDirection::SPM_TO_SPM:
            return id == SemanticOperandId::SOURCE_ADDRESS ||
                   id == SemanticOperandId::DESTINATION_ADDRESS;
        case LocalDteDirection::SPM_TO_DRAM:
            return id == SemanticOperandId::SOURCE_ADDRESS ||
                   id == SemanticOperandId::HBM_ADDRESS;
        case LocalDteDirection::DRAM_TO_SPM:
            return id == SemanticOperandId::DESTINATION_ADDRESS ||
                   id == SemanticOperandId::HBM_ADDRESS;
        }
    }
    if (std::holds_alternative<SymbolOperands>(record.operands) ||
        std::holds_alternative<SramResizeOperands>(record.operands))
        return id == SemanticOperandId::SYMBOL;
    if (std::holds_alternative<SramAllocOperands>(record.operands))
        return id == SemanticOperandId::REGION_NAME ||
               id == SemanticOperandId::LABEL_SYMBOL;
    if (std::holds_alternative<SramRenameOperands>(record.operands)) {
        return id == SemanticOperandId::OLD_SYMBOL ||
               id == SemanticOperandId::NEW_SYMBOL;
    }
    return false;
}


void ValidateSramLabelReference(uint64_t symbol_index,
                                const ProgramArtifact &artifact,
                                const std::string &field) {
    Require(symbol_index < artifact.symbols.size(),
            field + " references an unknown SRAM label symbol");
    Require(artifact.symbols[symbol_index].kind ==
                ProgramSymbolKind::SRAM_LABEL,
            field + " references a symbol with the wrong kind");
}

void ValidateAddressReference(const SramAddressOperand &address,
                              const ProgramArtifact &artifact,
                              const std::string &field,
                              uint64_t span_bytes) {
    Require(span_bytes != 0, field + " span must be non-zero");
    if (address.kind == SramAddressKind::ABSOLUTE) {
        Require(address.absolute_address_bytes <=
                    std::numeric_limits<uint64_t>::max() - (span_bytes - 1),
                field + " absolute span overflows u64");
        return;
    }
    Require(address.kind == SramAddressKind::REGION,
            field + " has an invalid address kind");
    Require(address.region_symbol_index < artifact.symbols.size(),
            field + " references an unknown region symbol");
    const ProgramSymbol &symbol =
        artifact.symbols[address.region_symbol_index];
    Require(symbol.kind == ProgramSymbolKind::SRAM_REGION,
            field + " references a symbol with the wrong kind");
    Require(address.region_offset_bytes <= symbol.size_bytes &&
                span_bytes <= symbol.size_bytes -
                                  address.region_offset_bytes,
            field + " span exceeds its region symbol size");
}

void ValidateRecordReferences(const ExternalRecord &record,
                              const ProgramArtifact &artifact,
                              const std::set<uint32_t> &group_ids,
                              const std::string &where) {
    if (const auto *o = std::get_if<DteSendOperands>(&record.operands)) {
        ValidateAddressReference(o->source, artifact, where + " source",
                                 o->length_bytes);
        if (o->group_id != 0)
            Require(group_ids.count(static_cast<uint32_t>(o->group_id)) != 0,
                    where + " references an unknown core group");
    } else if (const auto *o =
                   std::get_if<DteRecvOperands>(&record.operands)) {
        ValidateAddressReference(o->destination, artifact,
                                 where + " destination", o->length_bytes);
        if (o->group_id != 0)
            Require(group_ids.count(static_cast<uint32_t>(o->group_id)) != 0,
                    where + " references an unknown core group");
    } else if (const auto *o =
                   std::get_if<ReduceComputeOperands>(&record.operands)) {
        ValidateAddressReference(o->source, artifact, where + " source", 1);
        ValidateAddressReference(o->destination, artifact,
                                 where + " destination", 1);
        Require(group_ids.count(static_cast<uint32_t>(o->group_id)) != 0,
                where + " references an unknown core group");
    } else if (const auto *o = std::get_if<LsuOperands>(&record.operands)) {
        ValidateAddressReference(o->sram, artifact, where + " SRAM",
                                 o->size_bytes);
    } else if (const auto *o =
                   std::get_if<DteIssueOperands>(&record.operands)) {
        if (o->direction != LocalDteDirection::DRAM_TO_SPM)
            ValidateAddressReference(o->source_sram, artifact,
                                     where + " source SRAM",
                                     o->size_bytes);
        if (o->direction != LocalDteDirection::SPM_TO_DRAM)
            ValidateAddressReference(o->destination_sram, artifact,
                                     where + " destination SRAM",
                                     o->size_bytes);
    } else if (const auto *o =
                   std::get_if<SramBindOperands>(&record.operands)) {
        for (std::size_t i = 0; i < o->input_count; ++i)
            ValidateSramLabelReference(
                o->input_symbol_indices[i], artifact,
                where + " SRAM_BIND input " + std::to_string(i));
        ValidateSramLabelReference(o->output_symbol_index, artifact,
                                   where + " SRAM_BIND output");
    } else if (const auto *o =
                   std::get_if<SymbolOperands>(&record.operands)) {
        ValidateSramLabelReference(o->symbol_index, artifact,
                                   where + " lifecycle label");
    } else if (const auto *o =
                   std::get_if<SramAllocOperands>(&record.operands)) {
        Require(o->region_name_string_index < artifact.strings.size(),
                where + " references an unknown region-name string");
        Require(artifact.strings[o->region_name_string_index].size() <= 64,
                where + " SRAM region name exceeds 64 UTF-8 bytes");
        Require(o->label_symbol_index < artifact.symbols.size(),
                where + " references an unknown label symbol");
        Require(artifact.symbols[o->label_symbol_index].kind ==
                    ProgramSymbolKind::SRAM_LABEL,
                where + " label symbol has the wrong kind");
    } else if (const auto *o =
                   std::get_if<SramResizeOperands>(&record.operands)) {
        ValidateSramLabelReference(o->symbol_index, artifact,
                                   where + " resize label");
    } else if (const auto *o =
                   std::get_if<SramRenameOperands>(&record.operands)) {
        ValidateSramLabelReference(o->old_symbol_index, artifact,
                                   where + " rename old label");
        ValidateSramLabelReference(o->new_symbol_index, artifact,
                                   where + " rename new label");
    } else if (const auto *o =
                   std::get_if<GroupSyncOperands>(&record.operands)) {
        Require(group_ids.count(static_cast<uint32_t>(o->group_id)) != 0,
                where + " references an unknown core group");
    }
}

void ValidateAddend(const ProgramSymbol &symbol, int64_t addend,
                    const std::string &where) {
    if (addend >= 0) {
        const uint64_t positive = static_cast<uint64_t>(addend);
        Require(symbol.value <=
                    std::numeric_limits<uint64_t>::max() - positive,
                where + " symbol value plus addend overflows u64");
    } else {
        const uint64_t magnitude =
            static_cast<uint64_t>(-(addend + 1)) + 1;
        Require(symbol.value >= magnitude,
                where + " symbol value plus addend underflows u64");
    }
}

struct EncodedSection {
    ProgramSectionType type;
    uint32_t count;
    uint32_t entry_size;
    std::vector<uint8_t> bytes;
    uint64_t offset = 0;
};

std::vector<uint8_t> EncodeStrings(const ProgramArtifact &artifact) {
    std::vector<uint8_t> bytes;
    for (const std::string &value : artifact.strings) {
        AppendLe(bytes, value.size(), 4);
        bytes.insert(bytes.end(), value.begin(), value.end());
    }
    return bytes;
}

std::vector<uint8_t> EncodeSymbols(const ProgramArtifact &artifact) {
    std::vector<uint8_t> bytes;
    bytes.reserve(artifact.symbols.size() * kSymbolEntrySize);
    for (const ProgramSymbol &symbol : artifact.symbols) {
        AppendLe(bytes, symbol.name_string_index, 4);
        bytes.push_back(static_cast<uint8_t>(symbol.kind));
        bytes.push_back(static_cast<uint8_t>(symbol.flags));
        AppendLe(bytes, 0, 2);
        AppendLe(bytes, symbol.value, 8);
        AppendLe(bytes, symbol.size_bytes, 8);
    }
    return bytes;
}

std::vector<uint8_t> EncodeRelocations(const ProgramArtifact &artifact) {
    std::vector<uint8_t> bytes;
    bytes.reserve(artifact.relocations.size() * kRelocationEntrySize);
    for (const SemanticRelocation &relocation : artifact.relocations) {
        AppendLe(bytes, relocation.core_index, 4);
        AppendLe(bytes, relocation.instruction_index, 4);
        AppendLe(bytes, relocation.operand_id, 2);
        bytes.push_back(static_cast<uint8_t>(relocation.kind));
        bytes.push_back(0);
        AppendLe(bytes, relocation.symbol_index, 4);
        AppendLe(bytes, static_cast<uint64_t>(relocation.addend), 8);
    }
    return bytes;
}

std::vector<uint8_t> EncodeGroups(const ProgramArtifact &artifact) {
    std::vector<uint8_t> bytes;
    for (const ProgramCoreGroup &group : artifact.core_groups) {
        AppendLe(bytes, group.group_id, 4);
        AppendLe(bytes, group.members.size(), 4);
        for (uint64_t member : group.members)
            AppendLe(bytes, member, 2);
    }
    return bytes;
}

struct EncodedCoreSections {
    std::vector<uint8_t> index;
    std::vector<uint8_t> stream;
    uint32_t record_count = 0;
};

EncodedCoreSections EncodeCores(const ProgramArtifact &artifact) {
    EncodedCoreSections encoded;
    encoded.index.reserve(artifact.cores.size() * kCoreIndexEntrySize);
    for (const ProgramCore &core : artifact.cores) {
        const uint64_t stream_offset = encoded.stream.size();
        for (const ExternalRecord &record : core.records) {
            const std::vector<uint8_t> record_bytes =
                EncodeExternalRecord(record, artifact.capabilities);
            encoded.stream.insert(encoded.stream.end(), record_bytes.begin(),
                                  record_bytes.end());
            ++encoded.record_count;
        }
        AppendLe(encoded.index, core.core_id, 2);
        AppendLe(encoded.index, 0, 2);
        AppendLe(encoded.index, core.records.size(), 4);
        AppendLe(encoded.index, stream_offset, 8);
        AppendLe(encoded.index, encoded.stream.size() - stream_offset, 8);
        AppendLe(encoded.index, 0, 8);
    }
    return encoded;
}

std::vector<uint8_t> EncodeEnvelope(const ProgramArtifact &artifact) {
    const ProgramControlEnvelope &envelope = artifact.envelope;
    std::vector<uint8_t> bytes;
    bytes.reserve(kEnvelopeHeaderSize + envelope.active_cores.size() * 2 +
                  envelope.start_events.size() * kStartEventSize +
                  envelope.terminal_cores.size() * 2 +
                  envelope.expected_ack_cores.size() * 2 +
                  envelope.expected_done_cores.size() * 2);
    AppendLe(bytes, kEnvelopeVersion, 2);
    bytes.push_back(static_cast<uint8_t>(envelope.empty_core_ack_policy));
    bytes.push_back(static_cast<uint8_t>(envelope.failure_policy));
    AppendLe(bytes, envelope.active_cores.size(), 4);
    AppendLe(bytes, envelope.start_events.size(), 4);
    AppendLe(bytes, envelope.terminal_cores.size(), 4);
    AppendLe(bytes, envelope.expected_ack_cores.size(), 4);
    AppendLe(bytes, envelope.expected_done_cores.size(), 4);
    AppendLe(bytes, 0, 8);
    for (uint64_t core : envelope.active_cores)
        AppendLe(bytes, core, 2);
    for (const ProgramStartEvent &event : envelope.start_events) {
        AppendLe(bytes, event.target_core, 2);
        AppendLe(bytes, 0, 2);
        AppendLe(bytes, event.tag, 4);
        AppendLe(bytes, event.count, 4);
    }
    for (uint64_t core : envelope.terminal_cores)
        AppendLe(bytes, core, 2);
    for (uint64_t core : envelope.expected_ack_cores)
        AppendLe(bytes, core, 2);
    for (uint64_t core : envelope.expected_done_cores)
        AppendLe(bytes, core, 2);
    return bytes;
}

uint32_t WholeFileCrc(const std::vector<uint8_t> &bytes) noexcept {
    uint32_t crc = 0xffffffffu;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        const uint8_t byte =
            i >= kProgramWholeFileCrcOffset &&
                    i < kProgramWholeFileCrcOffset + 4
                ? 0
                : bytes[i];
        crc ^= byte;
        for (unsigned bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0x82f63b78u & (0u - (crc & 1u)));
    }
    return ~crc;
}

struct SectionDescriptor {
    uint32_t raw_type = 0;
    uint32_t flags = 0;
    uint64_t offset = 0;
    uint64_t size = 0;
    uint32_t count = 0;
    uint32_t entry_size = 0;
    uint32_t crc = 0;
};

bool IsKnownSection(uint32_t raw) {
    return raw >= static_cast<uint32_t>(ProgramSectionType::STRING_TABLE) &&
           raw <= static_cast<uint32_t>(ProgramSectionType::CONTROL_ENVELOPE);
}

std::string RecordDecodeContext(const std::vector<uint8_t> &bytes,
                                std::size_t offset, uint64_t core_id,
                                uint64_t record_index) {
    std::ostringstream stream;
    stream << "core " << core_id << " record " << record_index
           << " at file offset " << offset;
    if (offset < bytes.size()) {
        const uint8_t raw_opcode = bytes[offset];
        stream << ", opcode 0x" << std::hex << std::uppercase
               << std::setw(2) << std::setfill('0')
               << static_cast<unsigned>(raw_opcode);
        if (const OpcodeManifestEntry *entry = LookupOpcode(raw_opcode))
            stream << " (" << entry->canonical_name << ")";
    }
    return stream.str();
}

const SectionDescriptor &FindSection(
    const std::vector<SectionDescriptor> &sections, ProgramSectionType type) {
    const uint32_t raw = static_cast<uint32_t>(type);
    const auto it = std::find_if(
        sections.begin(), sections.end(),
        [raw](const SectionDescriptor &section) {
            return section.raw_type == raw;
        });
    Require(it != sections.end(), "missing required program section " +
                                      std::to_string(raw));
    return *it;
}

void ValidateSectionShape(const SectionDescriptor &section) {
    const ProgramSectionType type =
        static_cast<ProgramSectionType>(section.raw_type);
    switch (type) {
    case ProgramSectionType::STRING_TABLE:
        Require(section.entry_size == 0,
                "string table entry_size must be zero");
        Require(section.count <= kMaxProgramStrings,
                "string table count exceeds limit");
        Require(section.size <= kMaxProgramStringTableBytes,
                "string table byte size exceeds limit");
        break;
    case ProgramSectionType::SYMBOL_TABLE:
        Require(section.entry_size == kSymbolEntrySize,
                "symbol table entry_size must be 24");
        Require(section.count <= kMaxProgramSymbols,
                "symbol table count exceeds limit");
        Require(section.size ==
                    uint64_t{section.count} * kSymbolEntrySize,
                "symbol table count/size mismatch");
        break;
    case ProgramSectionType::SEMANTIC_RELOCATION_TABLE:
        Require(section.entry_size == kRelocationEntrySize,
                "relocation table entry_size must be 24");
        Require(section.count <= kMaxProgramRelocations,
                "relocation count exceeds limit");
        Require(section.size ==
                    uint64_t{section.count} * kRelocationEntrySize,
                "relocation table count/size mismatch");
        break;
    case ProgramSectionType::CORE_GROUP_TABLE:
        Require(section.entry_size == 0,
                "core group table entry_size must be zero");
        Require(section.count <= kMaxProgramCoreGroups,
                "core group count exceeds limit");
        break;
    case ProgramSectionType::CORE_PROGRAM_INDEX:
        Require(section.entry_size == kCoreIndexEntrySize,
                "core index entry_size must be 32");
        Require(section.count <= kMaxProgramCores,
                "core index count exceeds limit");
        Require(section.size ==
                    uint64_t{section.count} * kCoreIndexEntrySize,
                "core index count/size mismatch");
        break;
    case ProgramSectionType::EXTERNAL_RECORD_STREAM:
        Require(section.entry_size == 0,
                "record stream entry_size must be zero");
        Require(section.count <= kMaxProgramRecords,
                "record stream count exceeds limit");
        break;
    case ProgramSectionType::CONTROL_ENVELOPE:
        Require(section.entry_size == 0,
                "control envelope entry_size must be zero");
        Require(section.count == 1,
                "control envelope count must be one");
        break;
    }
}

std::vector<std::string>
DecodeStrings(const std::vector<uint8_t> &bytes,
              const SectionDescriptor &section) {
    std::vector<std::string> strings;
    strings.reserve(section.count);
    std::size_t cursor = static_cast<std::size_t>(section.offset);
    const std::size_t end = cursor + static_cast<std::size_t>(section.size);
    for (uint32_t i = 0; i < section.count; ++i) {
        Require(end - cursor >= 4, "truncated string length");
        const uint64_t length = ReadLe(bytes, cursor, 4, "string length");
        cursor += 4;
        Require(length <= kMaxProgramStringBytes,
                "program string exceeds byte limit");
        Require(length <= end - cursor, "truncated program string");
        strings.emplace_back(
            reinterpret_cast<const char *>(bytes.data() + cursor),
            static_cast<std::size_t>(length));
        cursor += static_cast<std::size_t>(length);
    }
    Require(cursor == end, "string table has trailing bytes");
    return strings;
}

std::vector<ProgramSymbol>
DecodeSymbols(const std::vector<uint8_t> &bytes,
              const SectionDescriptor &section) {
    std::vector<ProgramSymbol> symbols;
    symbols.reserve(section.count);
    std::size_t cursor = static_cast<std::size_t>(section.offset);
    for (uint32_t i = 0; i < section.count; ++i) {
        ProgramSymbol symbol;
        symbol.name_string_index =
            ReadLe(bytes, cursor, 4, "symbol name_string_index");
        symbol.kind = static_cast<ProgramSymbolKind>(
            ReadLe(bytes, cursor + 4, 1, "symbol kind"));
        symbol.flags = ReadLe(bytes, cursor + 5, 1, "symbol flags");
        RequireZero(bytes, cursor + 6, 2, "symbol");
        symbol.value = ReadLe(bytes, cursor + 8, 8, "symbol value");
        symbol.size_bytes =
            ReadLe(bytes, cursor + 16, 8, "symbol size_bytes");
        symbols.push_back(symbol);
        cursor += kSymbolEntrySize;
    }
    return symbols;
}

int64_t DecodeI64(uint64_t raw) {
    if (raw <= static_cast<uint64_t>(INT64_MAX))
        return static_cast<int64_t>(raw);
    const uint64_t magnitude = std::numeric_limits<uint64_t>::max() - raw;
    return -1 - static_cast<int64_t>(magnitude);
}

std::vector<SemanticRelocation>
DecodeRelocations(const std::vector<uint8_t> &bytes,
                  const SectionDescriptor &section) {
    std::vector<SemanticRelocation> relocations;
    relocations.reserve(section.count);
    std::size_t cursor = static_cast<std::size_t>(section.offset);
    for (uint32_t i = 0; i < section.count; ++i) {
        SemanticRelocation relocation;
        relocation.core_index =
            ReadLe(bytes, cursor, 4, "relocation core_index");
        relocation.instruction_index =
            ReadLe(bytes, cursor + 4, 4, "relocation instruction_index");
        relocation.operand_id =
            ReadLe(bytes, cursor + 8, 2, "relocation operand_id");
        relocation.kind = static_cast<SemanticRelocationKind>(
            ReadLe(bytes, cursor + 10, 1, "relocation kind"));
        RequireZero(bytes, cursor + 11, 1, "relocation");
        relocation.symbol_index =
            ReadLe(bytes, cursor + 12, 4, "relocation symbol_index");
        relocation.addend =
            DecodeI64(ReadLe(bytes, cursor + 16, 8, "relocation addend"));
        relocations.push_back(relocation);
        cursor += kRelocationEntrySize;
    }
    return relocations;
}

std::vector<ProgramCoreGroup>
DecodeGroups(const std::vector<uint8_t> &bytes,
             const SectionDescriptor &section) {
    std::vector<ProgramCoreGroup> groups;
    groups.reserve(section.count);
    std::size_t cursor = static_cast<std::size_t>(section.offset);
    const std::size_t end = cursor + static_cast<std::size_t>(section.size);
    for (uint32_t i = 0; i < section.count; ++i) {
        Require(end - cursor >= 8, "truncated core group header");
        ProgramCoreGroup group;
        group.group_id = ReadLe(bytes, cursor, 4, "core group id");
        const uint64_t member_count =
            ReadLe(bytes, cursor + 4, 4, "core group member count");
        cursor += 8;
        Require(member_count <= kMaxProgramCores,
                "core group member count exceeds limit");
        Require(member_count <= (end - cursor) / 2,
                "truncated core group members");
        group.members.reserve(static_cast<std::size_t>(member_count));
        for (uint64_t member = 0; member < member_count; ++member) {
            group.members.push_back(
                ReadLe(bytes, cursor, 2, "core group member"));
            cursor += 2;
        }
        groups.push_back(std::move(group));
    }
    Require(cursor == end, "core group table has trailing bytes");
    return groups;
}

struct RawCoreIndex {
    uint64_t core_id = 0;
    uint64_t record_count = 0;
    uint64_t stream_offset = 0;
    uint64_t stream_size = 0;
};

std::vector<RawCoreIndex>
DecodeCoreIndex(const std::vector<uint8_t> &bytes,
                const SectionDescriptor &section) {
    std::vector<RawCoreIndex> indices;
    indices.reserve(section.count);
    std::size_t cursor = static_cast<std::size_t>(section.offset);
    for (uint32_t i = 0; i < section.count; ++i) {
        RawCoreIndex index;
        index.core_id = ReadLe(bytes, cursor, 2, "core id");
        RequireZero(bytes, cursor + 2, 2, "core index flags");
        index.record_count =
            ReadLe(bytes, cursor + 4, 4, "core record count");
        index.stream_offset =
            ReadLe(bytes, cursor + 8, 8, "core stream offset");
        index.stream_size =
            ReadLe(bytes, cursor + 16, 8, "core stream size");
        RequireZero(bytes, cursor + 24, 8, "core index");
        indices.push_back(index);
        cursor += kCoreIndexEntrySize;
    }
    return indices;
}

std::vector<uint64_t> DecodeCoreList(const std::vector<uint8_t> &bytes,
                                     std::size_t &cursor,
                                     std::size_t end, uint32_t count,
                                     const std::string &field) {
    Require(uint64_t{count} * 2 <= end - cursor,
            "truncated envelope " + field);
    std::vector<uint64_t> result;
    result.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        result.push_back(ReadLe(bytes, cursor, 2, field));
        cursor += 2;
    }
    return result;
}

ProgramControlEnvelope
DecodeEnvelope(const std::vector<uint8_t> &bytes,
               const SectionDescriptor &section) {
    Require(section.size >= kEnvelopeHeaderSize,
            "truncated control envelope header");
    const std::size_t base = static_cast<std::size_t>(section.offset);
    const std::size_t end = base + static_cast<std::size_t>(section.size);
    Require(ReadLe(bytes, base, 2, "envelope version") == kEnvelopeVersion,
            "unsupported control envelope version");
    ProgramControlEnvelope envelope;
    envelope.empty_core_ack_policy = static_cast<EmptyCoreAckPolicy>(
        ReadLe(bytes, base + 2, 1, "empty core ACK policy"));
    envelope.failure_policy = static_cast<ProgramFailurePolicy>(
        ReadLe(bytes, base + 3, 1, "failure policy"));
    const uint32_t active_count =
        static_cast<uint32_t>(ReadLe(bytes, base + 4, 4, "active core count"));
    const uint32_t source_count =
        static_cast<uint32_t>(ReadLe(bytes, base + 8, 4, "source count"));
    const uint32_t terminal_count = static_cast<uint32_t>(
        ReadLe(bytes, base + 12, 4, "terminal count"));
    const uint32_t ack_count =
        static_cast<uint32_t>(ReadLe(bytes, base + 16, 4, "ACK count"));
    const uint32_t done_count =
        static_cast<uint32_t>(ReadLe(bytes, base + 20, 4, "DONE count"));
    RequireZero(bytes, base + 24, 8, "control envelope");
    for (uint32_t count : {active_count, source_count, terminal_count,
                           ack_count, done_count})
        Require(count <= kMaxProgramCores,
                "control envelope count exceeds limit");

    std::size_t cursor = base + kEnvelopeHeaderSize;
    envelope.active_cores =
        DecodeCoreList(bytes, cursor, end, active_count, "active cores");
    Require(uint64_t{source_count} * kStartEventSize <= end - cursor,
            "truncated start event list");
    envelope.start_events.reserve(source_count);
    for (uint32_t i = 0; i < source_count; ++i) {
        ProgramStartEvent event;
        event.target_core = ReadLe(bytes, cursor, 2, "start target core");
        RequireZero(bytes, cursor + 2, 2, "start event");
        event.tag = ReadLe(bytes, cursor + 4, 4, "start tag");
        event.count = ReadLe(bytes, cursor + 8, 4, "start count");
        envelope.start_events.push_back(event);
        cursor += kStartEventSize;
    }
    envelope.terminal_cores =
        DecodeCoreList(bytes, cursor, end, terminal_count, "terminal cores");
    envelope.expected_ack_cores =
        DecodeCoreList(bytes, cursor, end, ack_count, "expected ACK cores");
    envelope.expected_done_cores =
        DecodeCoreList(bytes, cursor, end, done_count, "expected DONE cores");
    Require(cursor == end, "control envelope has trailing bytes");
    return envelope;
}

} // namespace

uint32_t ProgramCrc32c(const uint8_t *data, std::size_t size) noexcept {
    uint32_t crc = 0xffffffffu;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (unsigned bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0x82f63b78u & (0u - (crc & 1u)));
    }
    return ~crc;
}

uint32_t ProgramCrc32c(const std::vector<uint8_t> &bytes) noexcept {
    return ProgramCrc32c(bytes.data(), bytes.size());
}

void ValidateProgramArtifact(const ProgramArtifact &artifact) {
    Require((artifact.capabilities & ~kKnownCapabilities) == 0,
            "artifact declares an unknown capability bit");
    Require(artifact.strings.size() <= kMaxProgramStrings,
            "program string count exceeds limit");
    uint64_t string_bytes = 0;
    std::set<std::string> unique_strings;
    for (const std::string &value : artifact.strings) {
        Require(!value.empty(), "program strings must not be empty");
        Require(value.size() <= kMaxProgramStringBytes,
                "program string exceeds 255 UTF-8 bytes");
        Require(IsValidUtf8(value),
                "program string is not canonical UTF-8 or contains NUL");
        Require(unique_strings.insert(value).second,
                "program string table contains a duplicate value");
        const uint64_t encoded_size = 4 + value.size();
        Require(encoded_size <= kMaxProgramStringTableBytes - string_bytes,
                "program string table exceeds byte limit");
        string_bytes += encoded_size;
    }

    Require(artifact.symbols.size() <= kMaxProgramSymbols,
            "program symbol count exceeds limit");
    std::set<std::string> symbol_names;
    for (std::size_t i = 0; i < artifact.symbols.size(); ++i) {
        const ProgramSymbol &symbol = artifact.symbols[i];
        RequireU32(symbol.name_string_index, "symbol name_string_index");
        Require(symbol.name_string_index < artifact.strings.size(),
                "symbol references an unknown name string");
        Require(ValidSymbolKind(symbol.kind), "symbol kind is unknown");
        Require(symbol.flags == 0, "symbol flags are reserved in v1");
        const std::string &name = artifact.strings[symbol.name_string_index];
        Require(symbol_names.insert(name).second,
                "duplicate symbol definition: " + name);
        if (symbol.kind == ProgramSymbolKind::SRAM_REGION) {
            Require(name.size() <= 64,
                    "SRAM region symbol name exceeds 64 UTF-8 bytes");
            Require(symbol.size_bytes != 0,
                    "SRAM region symbol size must be non-zero");
            Require(symbol.value <=
                        std::numeric_limits<uint64_t>::max() -
                            (symbol.size_bytes - 1),
                    "SRAM region symbol physical span overflows u64");
        }
        if (symbol.kind == ProgramSymbolKind::SRAM_LABEL)
            Require(name.size() <= 255,
                    "SRAM label symbol name exceeds 255 UTF-8 bytes");
    }

    Require(artifact.core_groups.size() <= kMaxProgramCoreGroups,
            "core group count exceeds limit");
    std::set<uint32_t> group_ids;
    uint64_t previous_group = 0;
    for (std::size_t i = 0; i < artifact.core_groups.size(); ++i) {
        const ProgramCoreGroup &group = artifact.core_groups[i];
        RequireU32(group.group_id, "core group id");
        Require(group.group_id != 0, "core group id zero is reserved");
        if (i != 0)
            Require(previous_group < group.group_id,
                    "core groups must be ordered by group id");
        previous_group = group.group_id;
        Require(group_ids.insert(static_cast<uint32_t>(group.group_id)).second,
                "duplicate core group id");
        ValidateSortedUniqueU16(group.members, "core group members", false);
    }

    Require(artifact.cores.size() <= kMaxProgramCores,
            "program core count exceeds limit");
    uint64_t total_records = 0;
    std::vector<uint64_t> core_ids;
    core_ids.reserve(artifact.cores.size());
    for (std::size_t core_index = 0; core_index < artifact.cores.size();
         ++core_index) {
        const ProgramCore &core = artifact.cores[core_index];
        RequireU16(core.core_id, "program core id");
        if (core_index != 0)
            Require(artifact.cores[core_index - 1].core_id < core.core_id,
                    "program cores must be strictly increasing and unique");
        core_ids.push_back(core.core_id);
        Require(core.records.size() <= kMaxProgramRecords - total_records,
                "program record count exceeds limit");
        total_records += core.records.size();
        for (std::size_t instruction = 0; instruction < core.records.size();
             ++instruction) {
            const std::string where =
                "core " + std::to_string(core.core_id) + " record " +
                std::to_string(instruction);
            try {
                ValidateExternalRecord(core.records[instruction],
                                       artifact.capabilities);
            } catch (const std::exception &error) {
                Fail(where + ": " + error.what());
            }
            ValidateRecordReferences(core.records[instruction], artifact,
                                     group_ids, where);
        }
    }

    Require(artifact.relocations.size() <= kMaxProgramRelocations,
            "semantic relocation count exceeds limit");
    std::tuple<uint64_t, uint64_t, uint64_t> previous_relocation{};
    for (std::size_t i = 0; i < artifact.relocations.size(); ++i) {
        const SemanticRelocation &relocation = artifact.relocations[i];
        RequireU32(relocation.core_index, "relocation core_index");
        RequireU32(relocation.instruction_index,
                   "relocation instruction_index");
        RequireU16(relocation.operand_id, "relocation operand_id");
        Require(relocation.operand_id != 0,
                "relocation operand_id zero is reserved");
        Require(ValidRelocationKind(relocation.kind),
                "semantic relocation kind is unknown");
        RequireU32(relocation.symbol_index, "relocation symbol_index");
        Require(relocation.core_index < artifact.cores.size(),
                "relocation references an unknown core index");
        Require(relocation.instruction_index <
                    artifact.cores[relocation.core_index].records.size(),
                "relocation references an unknown instruction index");
        const ExternalRecord &target =
            artifact.cores[relocation.core_index]
                .records[relocation.instruction_index];
        Require(ValidRelocationOperand(target, relocation.operand_id),
                "relocation operand_id is invalid for target opcode");
        if (const auto *send =
                std::get_if<DteSendOperands>(&target.operands)) {
            if (send->source_space == EndpointSourceSpace::HBM)
                Require(relocation.kind ==
                            SemanticRelocationKind::ABSOLUTE_ADDRESS,
                        "DTE_SEND HBM relocation requires ABSOLUTE_ADDRESS kind");
        }
        if (std::holds_alternative<SramBindOperands>(target.operands))
            Require(relocation.kind == SemanticRelocationKind::SRAM_LABEL,
                    "SRAM_BIND relocation requires SRAM_LABEL kind");

        Require(relocation.symbol_index < artifact.symbols.size(),
                "relocation references an unknown symbol");
        Require(static_cast<uint8_t>(relocation.kind) ==
                    static_cast<uint8_t>(
                        artifact.symbols[relocation.symbol_index].kind),
                "relocation kind does not match symbol kind");
        const ProgramSymbol &relocation_symbol =
            artifact.symbols[relocation.symbol_index];
        if (relocation.kind == SemanticRelocationKind::SRAM_REGION) {
            Require(relocation.addend >= 0,
                    "SRAM_REGION relocation addend cannot be negative");
            const uint64_t offset =
                static_cast<uint64_t>(relocation.addend);
            Require(offset <= relocation_symbol.size_bytes,
                    "SRAM_REGION relocation addend exceeds symbol size");
        } else {
            ValidateAddend(relocation_symbol, relocation.addend,
                           "relocation");
        }
        const auto key =
            std::make_tuple(relocation.core_index,
                            relocation.instruction_index,
                            relocation.operand_id);
        if (i != 0)
            Require(previous_relocation < key,
                    "relocations must be ordered and target unique operands");
        previous_relocation = key;
    }

    const ProgramControlEnvelope &envelope = artifact.envelope;
    Require(envelope.empty_core_ack_policy ==
                    EmptyCoreAckPolicy::EXCLUDE_EMPTY ||
                envelope.empty_core_ack_policy ==
                    EmptyCoreAckPolicy::INCLUDE_EMPTY,
            "empty core ACK policy is unknown");
    Require(envelope.failure_policy == ProgramFailurePolicy::ABORT_ALL,
            "failure policy is unknown");
    ValidateSortedUniqueU16(envelope.active_cores, "active cores", true);
    Require(envelope.active_cores == core_ids,
            "active core set must exactly match the core program index");
    ValidateSortedUniqueU16(envelope.terminal_cores, "terminal cores", true);
    ValidateSortedUniqueU16(envelope.expected_ack_cores,
                            "expected ACK cores", true);
    ValidateSortedUniqueU16(envelope.expected_done_cores,
                            "expected DONE cores", true);
    for (uint64_t core : envelope.terminal_cores)
        Require(ContainsCore(envelope.active_cores, core),
                "terminal references a core outside the active set");
    for (uint64_t core : envelope.expected_ack_cores)
        Require(ContainsCore(envelope.active_cores, core),
                "ACK set references a core outside the active set");
    for (uint64_t core : envelope.expected_done_cores)
        Require(ContainsCore(envelope.active_cores, core),
                "DONE set references a core outside the active set");

    std::tuple<uint64_t, uint64_t> previous_start{};
    for (std::size_t i = 0; i < envelope.start_events.size(); ++i) {
        const ProgramStartEvent &event = envelope.start_events[i];
        RequireU16(event.target_core, "start target_core");
        RequireU32(event.tag, "start tag");
        RequireU32(event.count, "start count");
        Require(event.count != 0, "start count must be non-zero");
        Require(ContainsCore(envelope.active_cores, event.target_core),
                "start event references a core outside the active set");
        const auto key = std::make_tuple(event.target_core, event.tag);
        if (i != 0)
            Require(previous_start < key,
                    "start events must be ordered and target/tag unique");
        previous_start = key;
    }

    std::vector<uint64_t> derived_ack;
    if (envelope.empty_core_ack_policy ==
        EmptyCoreAckPolicy::INCLUDE_EMPTY) {
        derived_ack = core_ids;
    } else {
        for (const ProgramCore &core : artifact.cores) {
            if (!core.records.empty())
                derived_ack.push_back(core.core_id);
        }
    }
    Require(envelope.expected_ack_cores == derived_ack,
            "expected ACK core set does not close under empty-core policy");
    Require(envelope.expected_done_cores == envelope.terminal_cores,
            "expected DONE core set must exactly match terminal cores");
}

std::vector<uint8_t> EncodeProgramArtifact(const ProgramArtifact &artifact) {
    ValidateProgramArtifact(artifact);
    EncodedCoreSections cores = EncodeCores(artifact);
    std::array<EncodedSection, 7> sections{{
        {ProgramSectionType::STRING_TABLE,
         static_cast<uint32_t>(artifact.strings.size()), 0,
         EncodeStrings(artifact)},
        {ProgramSectionType::SYMBOL_TABLE,
         static_cast<uint32_t>(artifact.symbols.size()), kSymbolEntrySize,
         EncodeSymbols(artifact)},
        {ProgramSectionType::SEMANTIC_RELOCATION_TABLE,
         static_cast<uint32_t>(artifact.relocations.size()),
         kRelocationEntrySize, EncodeRelocations(artifact)},
        {ProgramSectionType::CORE_GROUP_TABLE,
         static_cast<uint32_t>(artifact.core_groups.size()), 0,
         EncodeGroups(artifact)},
        {ProgramSectionType::CORE_PROGRAM_INDEX,
         static_cast<uint32_t>(artifact.cores.size()), kCoreIndexEntrySize,
         std::move(cores.index)},
        {ProgramSectionType::EXTERNAL_RECORD_STREAM, cores.record_count, 0,
         std::move(cores.stream)},
        {ProgramSectionType::CONTROL_ENVELOPE, 1, 0,
         EncodeEnvelope(artifact)},
    }};

    const uint64_t table_size =
        sections.size() * kProgramSectionDescriptorSize;
    uint64_t next_offset = kProgramHeaderSize + table_size;
    for (EncodedSection &section : sections) {
        section.offset = next_offset;
        Require(section.bytes.size() <= kMaxProgramFileBytes - next_offset,
                "encoded program exceeds file size limit");
        next_offset += section.bytes.size();
    }
    Require(next_offset <= kMaxProgramFileBytes,
            "encoded program exceeds file size limit");

    std::vector<uint8_t> bytes(
        static_cast<std::size_t>(kProgramHeaderSize + table_size), 0);
    std::copy(kMagic.begin(), kMagic.end(), bytes.begin());
    WriteLe(bytes, 8, kProgramFormatMajor, 2);
    WriteLe(bytes, 10, kProgramFormatMinor, 2);
    WriteLe(bytes, 12, kProgramIsaMajor, 2);
    WriteLe(bytes, 14, kProgramIsaMinor, 2);
    WriteLe(bytes, 16, kProgramHeaderSize, 2);
    WriteLe(bytes, 18, kProgramLittleEndian, 1);
    WriteLe(bytes, 24, artifact.capabilities, 8);
    WriteLe(bytes, 32, sections.size(), 4);
    WriteLe(bytes, 40, kProgramHeaderSize, 8);
    WriteLe(bytes, 48, next_offset, 8);

    for (std::size_t i = 0; i < sections.size(); ++i) {
        const EncodedSection &section = sections[i];
        const std::size_t descriptor =
            kProgramHeaderSize + i * kProgramSectionDescriptorSize;
        WriteLe(bytes, descriptor, static_cast<uint32_t>(section.type), 4);
        WriteLe(bytes, descriptor + 4, kProgramSectionRequired, 4);
        WriteLe(bytes, descriptor + 8, section.offset, 8);
        WriteLe(bytes, descriptor + 16, section.bytes.size(), 8);
        WriteLe(bytes, descriptor + 24, section.count, 4);
        WriteLe(bytes, descriptor + 28, section.entry_size, 4);
        WriteLe(bytes, descriptor + 32,
                ProgramCrc32c(section.bytes), 4);
        bytes.insert(bytes.end(), section.bytes.begin(), section.bytes.end());
    }
    Require(bytes.size() == next_offset,
            "internal program encoder file size mismatch");
    WriteLe(bytes, kProgramWholeFileCrcOffset, WholeFileCrc(bytes), 4);
    return bytes;
}

ProgramArtifact DecodeProgramArtifact(const std::vector<uint8_t> &bytes) {
    Require(bytes.size() >= kProgramHeaderSize,
            "truncated program artifact header");
    Require(bytes.size() <= kMaxProgramFileBytes,
            "program artifact exceeds file size limit");
    Require(std::equal(kMagic.begin(), kMagic.end(), bytes.begin()),
            "program artifact magic mismatch");
    Require(ReadLe(bytes, 8, 2, "format major") == kProgramFormatMajor,
            "unsupported program format major");
    Require(ReadLe(bytes, 10, 2, "format minor") == kProgramFormatMinor,
            "unsupported program format minor");
    Require(ReadLe(bytes, 12, 2, "ISA major") == kProgramIsaMajor,
            "unsupported program ISA major");
    Require(ReadLe(bytes, 14, 2, "ISA minor") == kProgramIsaMinor,
            "unsupported program ISA minor");
    Require(ReadLe(bytes, 16, 2, "header size") == kProgramHeaderSize,
            "program header size must be 64");
    Require(ReadLe(bytes, 18, 1, "endianness") == kProgramLittleEndian,
            "program artifact must be little-endian");
    RequireZero(bytes, 19, 5, "program header");
    const uint64_t capabilities = ReadLe(bytes, 24, 8, "capabilities");
    Require((capabilities & ~kKnownCapabilities) == 0,
            "artifact declares an unknown capability bit");
    const uint64_t section_count =
        ReadLe(bytes, 32, 4, "section count");
    Require(section_count >= 7 && section_count <= kMaxProgramSections,
            "program section count is outside limits");
    RequireZero(bytes, 36, 4, "program header");
    const uint64_t table_offset =
        ReadLe(bytes, 40, 8, "section table offset");
    const uint64_t declared_file_size =
        ReadLe(bytes, 48, 8, "file size");
    Require(declared_file_size == bytes.size(),
            "declared program file size does not match input boundary");
    const uint32_t declared_crc = static_cast<uint32_t>(
        ReadLe(bytes, kProgramWholeFileCrcOffset, 4, "whole-file CRC32C"));
    RequireZero(bytes, 60, 4, "program header");
    Require(declared_crc == WholeFileCrc(bytes),
            "whole-file CRC32C mismatch");

    Require(table_offset >= kProgramHeaderSize &&
                table_offset <= bytes.size(),
            "section table offset is out of bounds");
    const uint64_t table_bytes =
        section_count * kProgramSectionDescriptorSize;
    Require(table_bytes <= bytes.size() - table_offset,
            "truncated section descriptor table");
    const uint64_t table_end = table_offset + table_bytes;
    for (std::size_t i = kProgramHeaderSize;
         i < static_cast<std::size_t>(table_offset); ++i)
        Require(bytes[i] == 0,
                "non-zero gap before section descriptor table");

    std::vector<SectionDescriptor> sections;
    sections.reserve(static_cast<std::size_t>(section_count));
    std::set<uint32_t> section_types;
    for (uint64_t i = 0; i < section_count; ++i) {
        const std::size_t offset = static_cast<std::size_t>(
            table_offset + i * kProgramSectionDescriptorSize);
        SectionDescriptor section;
        section.raw_type =
            static_cast<uint32_t>(ReadLe(bytes, offset, 4, "section type"));
        section.flags =
            static_cast<uint32_t>(ReadLe(bytes, offset + 4, 4, "section flags"));
        section.offset = ReadLe(bytes, offset + 8, 8, "section offset");
        section.size = ReadLe(bytes, offset + 16, 8, "section size");
        section.count =
            static_cast<uint32_t>(ReadLe(bytes, offset + 24, 4, "section count"));
        section.entry_size = static_cast<uint32_t>(
            ReadLe(bytes, offset + 28, 4, "section entry_size"));
        section.crc =
            static_cast<uint32_t>(ReadLe(bytes, offset + 32, 4, "section CRC"));
        RequireZero(bytes, offset + 36, 4, "section descriptor");
        Require(section_types.insert(section.raw_type).second,
                "duplicate program section type");
        Require((section.flags & ~kProgramSectionRequired) == 0,
                "section has unknown flag bits");
        if (IsKnownSection(section.raw_type)) {
            Require(section.flags == kProgramSectionRequired,
                    "known v1 section must be marked required");
            ValidateSectionShape(section);
        } else {
            Require((section.flags & kProgramSectionRequired) == 0,
                    "unknown required section is not supported");
            Require(section.count <= kMaxProgramRecords,
                    "unknown optional section count exceeds limit");
        }
        Require(section.offset >= table_end &&
                    section.offset <= bytes.size(),
                "section offset is out of bounds");
        Require(section.size <= bytes.size() - section.offset,
                "section size is out of bounds");
        const uint8_t *section_data =
            bytes.data() + static_cast<std::size_t>(section.offset);
        Require(ProgramCrc32c(section_data,
                              static_cast<std::size_t>(section.size)) ==
                    section.crc,
                "section CRC32C mismatch for type " +
                    std::to_string(section.raw_type));
        sections.push_back(section);
    }

    std::vector<std::pair<uint64_t, uint64_t>> intervals;
    intervals.reserve(sections.size());
    for (const SectionDescriptor &section : sections) {
        if (section.size != 0)
            intervals.emplace_back(section.offset,
                                   section.offset + section.size);
    }
    std::sort(intervals.begin(), intervals.end());
    uint64_t occupied_end = table_end;
    for (const auto &interval : intervals) {
        Require(interval.first >= occupied_end,
                "program sections overlap");
        for (uint64_t i = occupied_end; i < interval.first; ++i)
            Require(bytes[static_cast<std::size_t>(i)] == 0,
                    "non-zero gap between program sections");
        occupied_end = interval.second;
    }
    Require(occupied_end == bytes.size(),
            "trailing bytes are not owned by a program section");

    for (uint32_t required = 1; required <= 7; ++required)
        Require(section_types.count(required) != 0,
                "missing required program section " +
                    std::to_string(required));

    ProgramArtifact artifact;
    artifact.capabilities = capabilities;
    artifact.strings =
        DecodeStrings(bytes,
                      FindSection(sections, ProgramSectionType::STRING_TABLE));
    artifact.symbols =
        DecodeSymbols(bytes,
                      FindSection(sections, ProgramSectionType::SYMBOL_TABLE));
    artifact.core_groups =
        DecodeGroups(bytes,
                     FindSection(sections,
                                 ProgramSectionType::CORE_GROUP_TABLE));

    const SectionDescriptor &index_section =
        FindSection(sections, ProgramSectionType::CORE_PROGRAM_INDEX);
    const SectionDescriptor &stream_section =
        FindSection(sections, ProgramSectionType::EXTERNAL_RECORD_STREAM);
    const std::vector<RawCoreIndex> indices =
        DecodeCoreIndex(bytes, index_section);
    uint64_t expected_stream_offset = 0;
    uint64_t total_record_count = 0;
    artifact.cores.reserve(indices.size());
    for (const RawCoreIndex &index : indices) {
        Require(index.stream_offset == expected_stream_offset,
                "core record ranges must be contiguous and ordered");
        Require(index.stream_size <=
                    stream_section.size - index.stream_offset,
                "core record range exceeds record stream section");
        Require(index.record_count <=
                    kMaxProgramRecords - total_record_count,
                "core record count exceeds global limit");
        ProgramCore core;
        core.core_id = index.core_id;
        const std::size_t begin = static_cast<std::size_t>(
            stream_section.offset + index.stream_offset);
        const std::size_t end =
            begin + static_cast<std::size_t>(index.stream_size);
        std::size_t cursor = begin;
        core.records.reserve(static_cast<std::size_t>(index.record_count));
        for (uint64_t record_index = 0;
             record_index < index.record_count; ++record_index) {
            try {
                DecodedExternalRecord decoded =
                    DecodeExternalRecord(bytes, cursor, capabilities);
                Require(decoded.next_offset <= end,
                        "external record crosses its core stream boundary");
                cursor = decoded.next_offset;
                core.records.push_back(std::move(decoded.record));
            } catch (const std::exception &error) {
                Fail(RecordDecodeContext(bytes, cursor, index.core_id,
                                         record_index) +
                     ": " + error.what());
            }
        }
        Require(cursor == end,
                "core record count/stream size mismatch");
        expected_stream_offset += index.stream_size;
        total_record_count += index.record_count;
        artifact.cores.push_back(std::move(core));
    }
    Require(expected_stream_offset == stream_section.size,
            "record stream contains unindexed bytes");
    Require(total_record_count == stream_section.count,
            "record stream descriptor count mismatch");

    artifact.relocations = DecodeRelocations(
        bytes,
        FindSection(sections,
                    ProgramSectionType::SEMANTIC_RELOCATION_TABLE));
    artifact.envelope = DecodeEnvelope(
        bytes, FindSection(sections, ProgramSectionType::CONTROL_ENVELOPE));

    ValidateProgramArtifact(artifact);
    return artifact;
}
