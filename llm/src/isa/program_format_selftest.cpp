#include "isa/program_format_selftest.h"
#include "isa/program_format.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr std::size_t kTestCoreIndexEntrySize = 32;

class Checks {
public:
    void Check(bool ok, const std::string &name) {
        ++result.checks;
        if (!ok) result.failures.push_back(name);
    }
    template <class F> void Accept(const std::string &name, F f) {
        ++result.checks;
        try { f(); }
        catch (const std::exception &e) {
            result.failures.push_back(name + ": " + e.what());
        }
    }
    template <class F> void Reject(const std::string &name, F f) {
        ++result.checks;
        try {
            f();
            result.failures.push_back(name + ": unexpectedly accepted");
        } catch (const ProgramFormatError &) {
        } catch (const std::exception &e) {
            result.failures.push_back(name + ": wrong exception: " + e.what());
        }
    }
    template <class F>
    void RejectContains(const std::string &name,
                        const std::vector<std::string> &needles, F f) {
        ++result.checks;
        try {
            f();
            result.failures.push_back(name + ": unexpectedly accepted");
        } catch (const ProgramFormatError &error) {
            const std::string message = error.what();
            for (const std::string &needle : needles)
                if (message.find(needle) == std::string::npos)
                    result.failures.push_back(
                        name + ": missing diagnostic context " + needle +
                        " in: " + message);
        } catch (const std::exception &error) {
            result.failures.push_back(name + ": wrong exception: " +
                                      error.what());
        }
    }
    ProgramFormatSelfTestResult result;
};

uint64_t ReadLe(const std::vector<uint8_t> &b, std::size_t o,
                std::size_t width) {
    uint64_t v = 0;
    for (std::size_t i = 0; i < width; ++i)
        v |= uint64_t{b[o + i]} << (8 * i);
    return v;
}
void WriteLe(std::vector<uint8_t> &b, std::size_t o, uint64_t v,
             std::size_t width) {
    for (std::size_t i = 0; i < width; ++i)
        b[o + i] = static_cast<uint8_t>(v >> (8 * i));
}
std::string Hex(const std::vector<uint8_t> &b, std::size_t count) {
    static constexpr char d[] = "0123456789abcdef";
    std::string out;
    count = std::min(count, b.size());
    for (std::size_t i = 0; i < count; ++i) {
        out.push_back(d[b[i] >> 4]);
        out.push_back(d[b[i] & 15]);
    }
    return out;
}
std::size_t Descriptor(ProgramSectionType type) {
    return kProgramHeaderSize +
           (static_cast<uint32_t>(type) - 1) *
               kProgramSectionDescriptorSize;
}
void RefreshWholeCrc(std::vector<uint8_t> &b) {
    WriteLe(b, kProgramWholeFileCrcOffset, 0, 4);
    WriteLe(b, kProgramWholeFileCrcOffset, ProgramCrc32c(b), 4);
}
void RefreshSectionCrc(std::vector<uint8_t> &b, ProgramSectionType type) {
    const std::size_t d = Descriptor(type);
    const std::size_t o = static_cast<std::size_t>(ReadLe(b, d + 8, 8));
    const std::size_t n = static_cast<std::size_t>(ReadLe(b, d + 16, 8));
    WriteLe(b, d + 32, ProgramCrc32c(b.data() + o, n), 4);
}

ExternalRecord DummyRecord() {
    ExternalRecord r;
    r.opcode = Opcode::DUMMY;
    r.operands = ComputeOperands{};
    return r;
}
ExternalRecord ClearRecord(uint64_t symbol = 0) {
    ExternalRecord r;
    r.opcode = Opcode::SRAM_CLEAR;
    r.operands = SymbolOperands{symbol};
    return r;
}
ExternalRecord FreeRecord(uint64_t symbol) {
    ExternalRecord r;
    r.opcode = Opcode::SRAM_FREE;
    r.operands = SymbolOperands{symbol};
    return r;
}
ExternalRecord ResizeRecord(uint64_t symbol) {
    ExternalRecord r;
    r.opcode = Opcode::SRAM_RESIZE;
    r.operands = SramResizeOperands{symbol, 128};
    return r;
}
ExternalRecord RenameRecord(uint64_t old_symbol, uint64_t new_symbol) {
    ExternalRecord r;
    r.opcode = Opcode::SRAM_RENAME;
    r.operands = SramRenameOperands{old_symbol, new_symbol};
    return r;
}
ExternalRecord BindRecord(uint64_t first = 0, uint64_t second = 1,
                          uint64_t output = 2) {
    ExternalRecord r;
    r.opcode = Opcode::SRAM_BIND;
    SramBindOperands o;
    o.input_count = 2;
    o.input_symbol_indices[0] = first;
    o.input_symbol_indices[1] = second;
    o.output_symbol_index = output;
    r.operands = std::move(o);
    return r;
}
ExternalRecord LoadRecord(uint64_t symbol) {
    ExternalRecord r;
    r.opcode = Opcode::LSU_LOAD;
    LsuOperands o;
    o.hbm_address_bytes = 0x4000;
    o.size_bytes = 64;
    o.sram = {SramAddressKind::REGION, 0, symbol, 32};
    r.operands = o;
    return r;
}
ExternalRecord DteSendRecord(EndpointSourceSpace source_space) {
    ExternalRecord r;
    r.opcode = Opcode::DTE_SEND;
    DteSendOperands o;
    o.source_space = source_space;
    o.length_bytes = 16;
    o.source.kind = SramAddressKind::ABSOLUTE;
    o.source.absolute_address_bytes = 0x1000;
    r.operands = o;
    return r;
}
ProgramArtifact DteSendRelocationArtifact(EndpointSourceSpace source_space,
                                          SemanticOperandId operand) {
    ProgramArtifact a;
    a.strings = {"p5-hbm-absolute"};
    a.symbols = {{0, ProgramSymbolKind::ABSOLUTE_ADDRESS, 0, 0x1000, 64}};
    a.cores = {{0, {DteSendRecord(source_space)}}};
    a.relocations = {{0, 0, static_cast<uint16_t>(operand),
                      SemanticRelocationKind::ABSOLUTE_ADDRESS, 0, 0}};
    a.envelope.active_cores = {0};
    a.envelope.terminal_cores = {0};
    a.envelope.expected_ack_cores = {0};
    a.envelope.expected_done_cores = {0};
    return a;
}
ExternalRecord LocalDteIssueRecord(LocalDteDirection direction) {
    DteIssueOperands operands;
    operands.direction = direction;
    operands.token = 17;
    operands.payload_bits = 512;
    operands.size_bytes = 64;
    operands.hbm_address_bytes =
        direction == LocalDteDirection::SPM_TO_SPM ? 0 : 0x8000;
    if (direction != LocalDteDirection::DRAM_TO_SPM)
        operands.source_sram =
            {SramAddressKind::REGION, 0, 0, 32};
    if (direction != LocalDteDirection::SPM_TO_DRAM)
        operands.destination_sram =
            {SramAddressKind::REGION, 0, 1, 32};
    return {Opcode::DTE_ISSUE, std::move(operands)};
}
ProgramArtifact LocalDteIssueArtifact(LocalDteDirection direction) {
    ProgramArtifact artifact;
    artifact.strings = {"source", "destination"};
    artifact.symbols = {
        {0, ProgramSymbolKind::SRAM_REGION, 0, 0, 256},
        {1, ProgramSymbolKind::SRAM_REGION, 0, 256, 256},
    };
    artifact.cores = {{0, {LocalDteIssueRecord(direction)}}};
    artifact.envelope.active_cores = {0};
    artifact.envelope.terminal_cores = {0};
    artifact.envelope.expected_ack_cores = {0};
    artifact.envelope.expected_done_cores = {0};
    return artifact;
}
ProgramArtifact BadLocalDteRelocationArtifact(
    LocalDteDirection direction, SemanticOperandId operand) {
    ProgramArtifact artifact = LocalDteIssueArtifact(direction);
    const bool hbm = operand == SemanticOperandId::HBM_ADDRESS;
    artifact.strings.push_back(hbm ? "hbm" : "relocated_region");
    artifact.symbols.push_back(
        {2, hbm ? ProgramSymbolKind::ABSOLUTE_ADDRESS
                : ProgramSymbolKind::SRAM_REGION,
         0, hbm ? 0x9000U : 512U, 256});
    artifact.relocations = {{
        0, 0, static_cast<uint16_t>(operand),
        hbm ? SemanticRelocationKind::ABSOLUTE_ADDRESS
            : SemanticRelocationKind::SRAM_REGION,
        2, 0}};
    return artifact;
}

ExternalRecord MlaRecord() {
    ExternalRecord r;
    r.opcode = Opcode::MATMUL_MLA;
    ComputeOperands o;
    o.datatype = ExternalDataType::FP16;
    o.input_offset_bytes = 2;
    o.data_offset_bytes = 4;
    o.output_offset_bytes = 6;
    o.parameters = {1, 2, 3, 4, 5, 6};
    r.operands = std::move(o);
    return r;
}

ProgramArtifact SingleArtifact() {
    ProgramArtifact a;
    a.cores = {{7, {DummyRecord()}}};
    a.envelope.active_cores = {7};
    a.envelope.start_events = {{7, 3, 1}};
    a.envelope.terminal_cores = {7};
    a.envelope.expected_ack_cores = {7};
    a.envelope.expected_done_cores = {7};
    return a;
}
ProgramArtifact MultiArtifact() {
    ProgramArtifact a;
    a.strings = {"label0", "region0", "region-name", u8"路径 with space"};
    a.symbols = {{0, ProgramSymbolKind::SRAM_LABEL, 0, 0x1000, 64},
                 {1, ProgramSymbolKind::SRAM_REGION, 0, 0x2000, 4096},
                 {3, ProgramSymbolKind::ABSOLUTE_ADDRESS, 0, 0x8000, 128}};
    a.core_groups = {{7, {0, 1}}, {9, {1, 65000}}};
    a.cores = {{0, {ClearRecord(0), LoadRecord(1)}}, {1, {}}};
    a.relocations = {
        {0, 0, static_cast<uint16_t>(SemanticOperandId::SYMBOL),
         SemanticRelocationKind::SRAM_LABEL, 0, 1}};
    a.envelope.active_cores = {0, 1};
    a.envelope.start_events = {{0, 9, 1}, {1, 10, 2}};
    a.envelope.terminal_cores = {0, 1};
    a.envelope.expected_ack_cores = {0, 1};
    a.envelope.expected_done_cores = {0, 1};
    return a;
}
ProgramArtifact BindArtifact() {
    ProgramArtifact a;
    a.strings = {"bind-in-0", "bind-in-1", "bind-out", "absolute"};
    a.symbols = {{0, ProgramSymbolKind::SRAM_LABEL, 0, 0, 64},
                 {1, ProgramSymbolKind::SRAM_LABEL, 0, 64, 64},
                 {2, ProgramSymbolKind::SRAM_LABEL, 0, 128, 64},
                 {3, ProgramSymbolKind::ABSOLUTE_ADDRESS, 0, 0x4000, 64}};
    a.cores = {{0, {BindRecord()}}};
    a.relocations = {
        {0, 0, static_cast<uint16_t>(
                   SemanticOperandId::SRAM_BIND_INPUT_0),
         SemanticRelocationKind::SRAM_LABEL, 1, 0},
        {0, 0, static_cast<uint16_t>(
                   SemanticOperandId::SRAM_BIND_OUTPUT),
         SemanticRelocationKind::SRAM_LABEL, 2, 0}};
    a.envelope.active_cores = {0};
    a.envelope.terminal_cores = {0};
    a.envelope.expected_ack_cores = {0};
    a.envelope.expected_done_cores = {0};
    return a;
}
ProgramArtifact EmptyArtifact() {
    ProgramArtifact a;
    a.cores = {{3, {}}};
    a.envelope.active_cores = {3};
    a.envelope.expected_ack_cores = {3};
    return a;
}
ProgramArtifact ExcludeEmptyArtifact() {
    ProgramArtifact a;
    a.cores = {{0, {DummyRecord()}}, {1, {}}};
    a.envelope.active_cores = {0, 1};
    a.envelope.terminal_cores = {0};
    a.envelope.expected_done_cores = {0};
    a.envelope.empty_core_ack_policy = EmptyCoreAckPolicy::EXCLUDE_EMPTY;
    a.envelope.expected_ack_cores = {0};
    return a;
}
ProgramArtifact CapabilityArtifact() {
    ProgramArtifact a;
    a.capabilities = CapabilityBit(IsaCapability::PD_CONTEXT);
    a.cores = {{2, {MlaRecord()}}};
    a.envelope.active_cores = {2};
    a.envelope.terminal_cores = {2};
    a.envelope.expected_ack_cores = {2};
    a.envelope.expected_done_cores = {2};
    return a;
}
bool RecordCodecRejects(const ExternalRecord &record) {
    try {
        ValidateExternalRecord(record);
        return false;
    } catch (const RecordCodecError &) {
        return true;
    }
}
void RoundTrip(Checks &c, const std::string &name, const ProgramArtifact &a) {
    c.Accept(name, [&] {
        const auto first = EncodeProgramArtifact(a);
        const auto decoded = DecodeProgramArtifact(first);
        c.Check(first == EncodeProgramArtifact(decoded),
                name + " deterministic re-encode");
    });
}
void CheckPositive(Checks &c) {
    RoundTrip(c, "single-core", SingleArtifact());
    RoundTrip(c, "multi-core all-sections", MultiArtifact());
    RoundTrip(c, "SRAM_BIND labels and relocations", BindArtifact());
    RoundTrip(c, "empty record stream", EmptyArtifact());
    RoundTrip(c, "exclude-empty ACK", ExcludeEmptyArtifact());
    RoundTrip(c, "capability record", CapabilityArtifact());
    RoundTrip(c, "DTE_SEND SRAM source relocation",
              DteSendRelocationArtifact(EndpointSourceSpace::SRAM,
                                        SemanticOperandId::SOURCE_ADDRESS));
    RoundTrip(c, "DTE_SEND HBM source relocation",
              DteSendRelocationArtifact(EndpointSourceSpace::HBM,
                                        SemanticOperandId::HBM_ADDRESS));
    RoundTrip(c, "DTE_ISSUE SPM_TO_SPM active endpoints",
              LocalDteIssueArtifact(LocalDteDirection::SPM_TO_SPM));
    RoundTrip(c, "DTE_ISSUE SPM_TO_DRAM active source",
              LocalDteIssueArtifact(LocalDteDirection::SPM_TO_DRAM));
    RoundTrip(c, "DTE_ISSUE DRAM_TO_SPM active destination",
              LocalDteIssueArtifact(LocalDteDirection::DRAM_TO_SPM));

    ExternalRecord bad_dram_source =
        LocalDteIssueRecord(LocalDteDirection::DRAM_TO_SPM);
    std::get<DteIssueOperands>(bad_dram_source.operands).source_sram =
        {SramAddressKind::REGION, 0, 0, 0};
    c.Check(RecordCodecRejects(bad_dram_source),
            "record codec rejects inactive DRAM_TO_SPM source SRAM");
    ExternalRecord bad_spm_destination =
        LocalDteIssueRecord(LocalDteDirection::SPM_TO_DRAM);
    std::get<DteIssueOperands>(bad_spm_destination.operands).destination_sram =
        {SramAddressKind::REGION, 0, 1, 0};
    c.Check(RecordCodecRejects(bad_spm_destination),
            "record codec rejects inactive SPM_TO_DRAM destination SRAM");
    ExternalRecord bad_local_hbm =
        LocalDteIssueRecord(LocalDteDirection::SPM_TO_SPM);
    std::get<DteIssueOperands>(bad_local_hbm.operands).hbm_address_bytes = 1;
    c.Check(RecordCodecRejects(bad_local_hbm),
            "record codec rejects inactive SPM_TO_SPM HBM address");

    c.Reject("DRAM_TO_SPM source SRAM relocation", [&] {
        EncodeProgramArtifact(BadLocalDteRelocationArtifact(
            LocalDteDirection::DRAM_TO_SPM,
            SemanticOperandId::SOURCE_ADDRESS));
    });
    c.Reject("SPM_TO_DRAM destination SRAM relocation", [&] {
        EncodeProgramArtifact(BadLocalDteRelocationArtifact(
            LocalDteDirection::SPM_TO_DRAM,
            SemanticOperandId::DESTINATION_ADDRESS));
    });
    c.Reject("SPM_TO_SPM HBM relocation", [&] {
        EncodeProgramArtifact(BadLocalDteRelocationArtifact(
            LocalDteDirection::SPM_TO_SPM,
            SemanticOperandId::HBM_ADDRESS));
    });
    const auto multi = EncodeProgramArtifact(MultiArtifact());
    const std::string utf8 = u8"路径";
    c.Check(std::search(multi.begin(), multi.end(), utf8.begin(), utf8.end(),
                        [](uint8_t lhs, char rhs) {
                            return lhs == static_cast<uint8_t>(rhs);
                        }) != multi.end(),
            "UTF-8/path bytes preserved");
    const ProgramArtifact decoded_multi = DecodeProgramArtifact(multi);
    c.Check(decoded_multi.cores[0].records[0].opcode == Opcode::SRAM_CLEAR &&
                decoded_multi.cores[0].records[1].opcode == Opcode::LSU_LOAD,
            "per-core record order preserved");
    c.Check(decoded_multi.cores[1].records.empty(),
            "empty core record range preserved");
    const std::string check = "123456789";
    c.Check(ProgramCrc32c(
                reinterpret_cast<const uint8_t *>(check.data()),
                check.size()) == 0xe3069283u, "CRC32C check vector");
    const auto single = EncodeProgramArtifact(SingleArtifact());
    c.Check(single.size() == ReadLe(single, 48, 8), "file size");
    c.Check(ReadLe(single, 32, 4) == 7, "required section count");
    c.Check(ReadLe(single, 40, 8) == 64, "section table offset");
    const std::string header = Hex(single, kProgramHeaderSize);
    static constexpr const char *golden =
        "4e5055505247310001000000010000004000010000000000000000000000000007000000000000004000000000000000bc010000000000004e94274600000000";
    c.Check(header == golden, "golden header expected=" +
                std::string(golden) + " actual=" + header);
}

std::vector<uint8_t> AddUnknownOptional(const std::vector<uint8_t> &input) {
    std::vector<uint8_t> b = input;
    constexpr std::size_t old_end =
        kProgramHeaderSize + 7 * kProgramSectionDescriptorSize;
    b.insert(b.begin() + old_end, kProgramSectionDescriptorSize, 0);
    for (std::size_t i = 0; i < 7; ++i) {
        const std::size_t d =
            kProgramHeaderSize + i * kProgramSectionDescriptorSize;
        WriteLe(b, d + 8,
                ReadLe(b, d + 8, 8) + kProgramSectionDescriptorSize, 8);
    }
    const std::vector<uint8_t> payload{0xde, 0xad, 0xbe, 0xef};
    const std::size_t d = old_end;
    WriteLe(b, d, 0x100, 4);
    WriteLe(b, d + 8, b.size(), 8);
    WriteLe(b, d + 16, payload.size(), 8);
    WriteLe(b, d + 32, ProgramCrc32c(payload), 4);
    b.insert(b.end(), payload.begin(), payload.end());
    WriteLe(b, 32, 8, 4);
    WriteLe(b, 48, b.size(), 8);
    RefreshWholeCrc(b);
    return b;
}
std::vector<uint8_t> InsertGap(const std::vector<uint8_t> &input,
                               uint8_t value) {
    std::vector<uint8_t> b = input;
    constexpr std::size_t table_end =
        kProgramHeaderSize + 7 * kProgramSectionDescriptorSize;
    b.insert(b.begin() + table_end, value);
    for (std::size_t i = 0; i < 7; ++i) {
        const std::size_t d =
            kProgramHeaderSize + i * kProgramSectionDescriptorSize;
        WriteLe(b, d + 8, ReadLe(b, d + 8, 8) + 1, 8);
    }
    WriteLe(b, 48, b.size(), 8);
    RefreshWholeCrc(b);
    return b;
}

void CheckHeaders(Checks &c) {
    const auto good = EncodeProgramArtifact(SingleArtifact());
    for (std::size_t n = 0; n < kProgramHeaderSize; ++n) {
        std::vector<uint8_t> bad(good.begin(), good.begin() + n);
        c.Reject("header truncation " + std::to_string(n),
                 [&] { DecodeProgramArtifact(bad); });
    }
    struct M { std::size_t o, w; uint64_t v; const char *name; };
    constexpr std::size_t table_end =
        kProgramHeaderSize + 7 * kProgramSectionDescriptorSize;
    for (std::size_t n = kProgramHeaderSize; n < table_end; ++n) {
        std::vector<uint8_t> truncated(good.begin(), good.begin() + n);
        WriteLe(truncated, 48, truncated.size(), 8);
        RefreshWholeCrc(truncated);
        c.Reject("descriptor truncation " + std::to_string(n),
                 [&] { DecodeProgramArtifact(truncated); });
    }

    const std::vector<M> mutations{
        {0, 1, 0, "magic"}, {8, 2, 2, "format major"},
        {10, 2, 1, "format minor"}, {12, 2, 2, "ISA major"},
        {14, 2, 1, "ISA minor"}, {16, 2, 63, "header size"},
        {18, 1, 2, "endianness"}, {19, 1, 1, "reserved"},
        {24, 8, uint64_t{1} << 63, "unknown capability"},
        {32, 4, 6, "missing descriptor"},
        {32, 4, kMaxProgramSections + 1, "section limit"},
        {36, 1, 1, "alignment reserved"},
        {40, 8, good.size() + 1, "table offset"},
        {48, 8, good.size() - 1, "file size"},
        {60, 1, 1, "tail reserved"}};
    for (const M &m : mutations) {
        auto bad = good;
        WriteLe(bad, m.o, m.v, m.w);
        RefreshWholeCrc(bad);
        c.Reject(std::string("header ") + m.name,
                 [&] { DecodeProgramArtifact(bad); });
    }
    auto bad = good;
    bad[kProgramWholeFileCrcOffset] ^= 1;
    c.Reject("whole CRC field bitflip",
             [&] { DecodeProgramArtifact(bad); });
    bad = good;
    bad.back() ^= 1;
    c.Reject("whole data bitflip", [&] { DecodeProgramArtifact(bad); });
    bad = good;
    bad.push_back(0);
    WriteLe(bad, 48, bad.size(), 8);
    RefreshWholeCrc(bad);
    c.Reject("trailing byte", [&] { DecodeProgramArtifact(bad); });
}

void CheckDescriptors(Checks &c) {
    const auto good = EncodeProgramArtifact(MultiArtifact());
    auto optional = AddUnknownOptional(good);
    c.Accept("unknown optional ignored",
             [&] { DecodeProgramArtifact(optional); });
    constexpr std::size_t od =
        kProgramHeaderSize + 7 * kProgramSectionDescriptorSize;
    auto bad = optional;
    WriteLe(bad, od + 4, kProgramSectionRequired, 4);
    RefreshWholeCrc(bad);
    c.Reject("unknown required section",
             [&] { DecodeProgramArtifact(bad); });
    bad = optional;
    const std::size_t oo = static_cast<std::size_t>(ReadLe(bad, od + 8, 8));
    bad[oo] ^= 1;
    RefreshWholeCrc(bad);
    c.Reject("optional section CRC", [&] { DecodeProgramArtifact(bad); });
    bad = optional;
    WriteLe(bad, od,
            static_cast<uint32_t>(ProgramSectionType::STRING_TABLE), 4);
    WriteLe(bad, od + 4, kProgramSectionRequired, 4);
    RefreshWholeCrc(bad);
    c.Reject("duplicate section", [&] { DecodeProgramArtifact(bad); });
    bad = good;
    const std::size_t ed = Descriptor(ProgramSectionType::CONTROL_ENVELOPE);
    WriteLe(bad, ed, 0x101, 4);
    WriteLe(bad, ed + 4, 0, 4);
    RefreshWholeCrc(bad);
    c.Reject("missing required section",
             [&] { DecodeProgramArtifact(bad); });

    struct D {
        ProgramSectionType type;
        std::size_t field, width;
        uint64_t value;
        const char *name;
    };
    const std::vector<D> mutations{
        {ProgramSectionType::STRING_TABLE, 4, 4, 2, "flags"},
        {ProgramSectionType::STRING_TABLE, 8, 8, good.size() + 1, "offset"},
        {ProgramSectionType::STRING_TABLE, 16, 8, good.size(), "size"},
        {ProgramSectionType::SYMBOL_TABLE, 24, 4, 4, "count-size"},
        {ProgramSectionType::SYMBOL_TABLE, 28, 4, 23, "entry-size"},
        {ProgramSectionType::STRING_TABLE, 36, 4, 1, "reserved"},
        {ProgramSectionType::CONTROL_ENVELOPE, 24, 4, 2,
         "envelope count"},
        {ProgramSectionType::STRING_TABLE, 24, 4,
         kMaxProgramStrings + 1, "string count limit"},
        {ProgramSectionType::EXTERNAL_RECORD_STREAM, 24, 4,
         kMaxProgramRecords + 1, "record count limit"},
    };
    for (const D &m : mutations) {
        bad = good;
        WriteLe(bad, Descriptor(m.type) + m.field, m.value, m.width);
        RefreshWholeCrc(bad);
        c.Reject(std::string("descriptor ") + m.name,
                 [&] { DecodeProgramArtifact(bad); });
    }
    bad = good;
    const std::size_t so = static_cast<std::size_t>(
        ReadLe(bad, Descriptor(ProgramSectionType::STRING_TABLE) + 8, 8));
    bad[so + 4] ^= 1;
    RefreshWholeCrc(bad);
    c.Reject("section CRC bitflip",
             [&] { DecodeProgramArtifact(bad); });
    bad = good;
    for (uint32_t raw = 1; raw <= 7; ++raw) {
        const auto type = static_cast<ProgramSectionType>(raw);
        bad = good;
        const std::size_t offset = static_cast<std::size_t>(
            ReadLe(bad, Descriptor(type) + 8, 8));
        c.Check(ReadLe(bad, Descriptor(type) + 16, 8) != 0,
                "all-section CRC fixture is non-empty");
        bad[offset] ^= 1;
        RefreshWholeCrc(bad);
        c.Reject("section CRC type " + std::to_string(raw),
                 [&] { DecodeProgramArtifact(bad); });
    }
    const std::size_t sd = Descriptor(ProgramSectionType::SYMBOL_TABLE);
    bad = good;
    WriteLe(bad, sd + 8, so, 8);
    RefreshSectionCrc(bad, ProgramSectionType::SYMBOL_TABLE);
    RefreshWholeCrc(bad);
    c.Reject("section overlap", [&] { DecodeProgramArtifact(bad); });
    c.Accept("zero section gap",
             [&] { DecodeProgramArtifact(InsertGap(good, 0)); });
    c.Reject("non-zero section gap",
             [&] { DecodeProgramArtifact(InsertGap(good, 1)); });
}

void CheckSymbolsAndRelocations(Checks &c) {
    ProgramArtifact a = MultiArtifact();
    a.strings.push_back(a.strings.front());
    c.Reject("duplicate string", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.strings[0] = std::string(
        {static_cast<char>(0xc0), static_cast<char>(0x80)});
    c.Reject("overlong UTF-8", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.strings[0] = std::string({'a', '\0', 'b'});
    c.Reject("embedded NUL", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.symbols.push_back({0, ProgramSymbolKind::SRAM_LABEL, 0, 1, 1});
    c.Reject("duplicate symbol", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.symbols[0].name_string_index = a.strings.size();
    c.Reject("symbol string index", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.symbols[0].kind = static_cast<ProgramSymbolKind>(4);
    c.Reject("symbol kind", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.symbols[0].flags = 1;
    c.Reject("symbol flags", [&] { EncodeProgramArtifact(a); });

    a = BindArtifact();
    a.relocations[0].operand_id =
        static_cast<uint16_t>(SemanticOperandId::SRAM_BIND_INPUT_0) + 2;
    c.Reject("SRAM_BIND relocation inactive input",
             [&] { EncodeProgramArtifact(a); });
    a = BindArtifact();
    a.relocations[0].kind = SemanticRelocationKind::ABSOLUTE_ADDRESS;
    a.relocations[0].symbol_index = 3;
    c.Reject("SRAM_BIND relocation wrong kind",
             [&] { EncodeProgramArtifact(a); });

    a = MultiArtifact();
    a.cores[0].records = {ClearRecord(1)};
    a.relocations.clear();
    c.Reject("SRAM_CLEAR requires SRAM_LABEL",
             [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.cores[0].records = {FreeRecord(1)};
    a.relocations.clear();
    c.Reject("SRAM_FREE requires SRAM_LABEL",
             [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.cores[0].records = {ResizeRecord(1)};
    a.relocations.clear();
    c.Reject("SRAM_RESIZE requires SRAM_LABEL",
             [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.cores[0].records = {RenameRecord(0, 1)};
    a.relocations.clear();
    c.Reject("SRAM_RENAME new requires SRAM_LABEL",
             [&] { EncodeProgramArtifact(a); });

    a = MultiArtifact();
    a.symbols[1].size_bytes = 0;
    c.Reject("SRAM region symbol rejects zero size",
             [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.symbols[1].value = std::numeric_limits<uint64_t>::max() - 31;
    a.symbols[1].size_bytes = 33;
    c.Reject("SRAM region symbol rejects physical span overflow",
             [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.symbols[1].size_bytes = 95;
    c.Reject("LSU named region rejects one-byte-short span",
             [&] { EncodeProgramArtifact(a); });

    a = MultiArtifact();
    a.relocations[0].core_index = a.cores.size();
    c.Reject("reloc core index", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.relocations[0].instruction_index = 99;
    c.Reject("reloc instruction index", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.relocations[0].operand_id =
        static_cast<uint16_t>(SemanticOperandId::HBM_ADDRESS);
    c.Reject("reloc operand for opcode", [&] { EncodeProgramArtifact(a); });
    a = DteSendRelocationArtifact(EndpointSourceSpace::SRAM,
                                  SemanticOperandId::HBM_ADDRESS);
    c.Reject("DTE_SEND SRAM rejects HBM relocation operand",
             [&] { EncodeProgramArtifact(a); });
    a = DteSendRelocationArtifact(EndpointSourceSpace::HBM,
                                  SemanticOperandId::SOURCE_ADDRESS);
    c.Reject("DTE_SEND HBM rejects SRAM relocation operand",
             [&] { EncodeProgramArtifact(a); });
    a = DteSendRelocationArtifact(EndpointSourceSpace::SRAM,
                                  SemanticOperandId::SOURCE_ADDRESS);
    a.strings = {"p5-sram-region"};
    a.symbols = {{0, ProgramSymbolKind::SRAM_REGION, 0, 0x1000, 64}};
    a.relocations[0].kind = SemanticRelocationKind::SRAM_REGION;
    a.relocations[0].addend = -1;
    c.Reject("SRAM region relocation rejects negative local offset",
             [&] { EncodeProgramArtifact(a); });
    a.relocations[0].addend = 65;
    c.Reject("SRAM region relocation rejects offset beyond extent",
             [&] { EncodeProgramArtifact(a); });

    a = DteSendRelocationArtifact(EndpointSourceSpace::HBM,
                                  SemanticOperandId::HBM_ADDRESS);
    a.strings = {"p5-hbm-region"};
    a.symbols = {{0, ProgramSymbolKind::SRAM_REGION, 0, 0x1000, 64}};
    a.relocations[0].kind = SemanticRelocationKind::SRAM_REGION;
    c.Reject("DTE_SEND HBM relocation rejects SRAM region kind",
             [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.relocations[0].operand_id = 0;
    c.Reject("reloc operand zero", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.relocations[0].symbol_index = a.symbols.size();
    c.Reject("reloc symbol index", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.relocations[0].kind = SemanticRelocationKind::SRAM_REGION;
    c.Reject("reloc kind mismatch", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.symbols[0].value = 0;
    a.relocations[0].addend = -1;
    c.Reject("reloc addend underflow", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.relocations.push_back(a.relocations.front());
    c.Reject("duplicate reloc target", [&] { EncodeProgramArtifact(a); });

    auto bad = EncodeProgramArtifact(MultiArtifact());
    const std::size_t so = static_cast<std::size_t>(
        ReadLe(bad, Descriptor(ProgramSectionType::STRING_TABLE) + 8, 8));
    bad[so + 4] = 0;
    RefreshSectionCrc(bad, ProgramSectionType::STRING_TABLE);
    RefreshWholeCrc(bad);
    c.Reject("decoded string NUL", [&] { DecodeProgramArtifact(bad); });
    bad = EncodeProgramArtifact(MultiArtifact());
    WriteLe(bad, so, kMaxProgramStringBytes + 1, 4);
    RefreshSectionCrc(bad, ProgramSectionType::STRING_TABLE);
    RefreshWholeCrc(bad);
    c.Reject("string length max+1", [&] { DecodeProgramArtifact(bad); });
}

void CheckGroupsCoresRecords(Checks &c) {
    ProgramArtifact a = MultiArtifact();
    a.core_groups[0].members.clear();
    c.Reject("empty group", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.core_groups[0].members = {1, 0};
    c.Reject("unordered group members", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.core_groups[0].members = {0, 0};
    c.Reject("duplicate group member", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.core_groups[0].members = {0, uint64_t{UINT16_MAX} + 1};
    c.Reject("group member max+1", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.core_groups[0].group_id = 0;
    c.Reject("group id zero", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.core_groups[1].group_id = 7;
    c.Reject("duplicate group id", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.cores[1].core_id = 0;
    a.envelope.active_cores = {0};
    a.envelope.terminal_cores = {0};
    a.envelope.expected_ack_cores = {0};
    a.envelope.expected_done_cores = {0};
    c.Reject("duplicate core id", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    std::swap(a.cores[0], a.cores[1]);
    c.Reject("unordered core index", [&] { EncodeProgramArtifact(a); });
    a = BindArtifact();
    std::get<SramBindOperands>(a.cores[0].records[0].operands)
        .input_symbol_indices[0] = a.symbols.size();
    c.Reject("SRAM_BIND unknown input symbol",
             [&] { EncodeProgramArtifact(a); });
    a = BindArtifact();
    std::get<SramBindOperands>(a.cores[0].records[0].operands)
        .output_symbol_index = a.symbols.size();
    c.Reject("SRAM_BIND unknown output symbol",
             [&] { EncodeProgramArtifact(a); });
    a = BindArtifact();
    std::get<SramBindOperands>(a.cores[0].records[0].operands)
        .input_symbol_indices[0] = 3;
    c.Reject("SRAM_BIND input symbol kind",
             [&] { EncodeProgramArtifact(a); });
    a = BindArtifact();
    std::get<SramBindOperands>(a.cores[0].records[0].operands)
        .output_symbol_index = 3;
    c.Reject("SRAM_BIND output symbol kind",
             [&] { EncodeProgramArtifact(a); });

    a = MultiArtifact();
    std::get<SymbolOperands>(a.cores[0].records[0].operands).symbol_index =
        a.symbols.size();
    c.Reject("record symbol index", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    std::get<LsuOperands>(a.cores[0].records[1].operands)
        .sram.region_symbol_index = 0;
    c.Reject("record symbol kind", [&] { EncodeProgramArtifact(a); });

    auto bad = EncodeProgramArtifact(MultiArtifact());
    const std::size_t io = static_cast<std::size_t>(
        ReadLe(bad, Descriptor(ProgramSectionType::CORE_PROGRAM_INDEX) + 8, 8));
    WriteLe(bad, io + 4, ReadLe(bad, io + 4, 4) + 1, 4);
    RefreshSectionCrc(bad, ProgramSectionType::CORE_PROGRAM_INDEX);
    RefreshWholeCrc(bad);
    c.Reject("forged record count", [&] { DecodeProgramArtifact(bad); });
    bad = EncodeProgramArtifact(MultiArtifact());
    WriteLe(bad, io + kTestCoreIndexEntrySize + 8, 0, 8);
    RefreshSectionCrc(bad, ProgramSectionType::CORE_PROGRAM_INDEX);
    RefreshWholeCrc(bad);
    c.Reject("overlapping core ranges", [&] { DecodeProgramArtifact(bad); });
    bad = EncodeProgramArtifact(MultiArtifact());
    const std::size_t ro = static_cast<std::size_t>(
        ReadLe(bad, Descriptor(ProgramSectionType::EXTERNAL_RECORD_STREAM) + 8,
               8));
    bad[ro] = 0;
    RefreshSectionCrc(bad, ProgramSectionType::EXTERNAL_RECORD_STREAM);
    RefreshWholeCrc(bad);
    c.RejectContains("record opcode diagnostic",
                     {"core 0 record 0 at file offset ", "opcode 0x00"},
                     [&] { DecodeProgramArtifact(bad); });
    bad = EncodeProgramArtifact(MultiArtifact());
    WriteLe(bad, ro + 4, UINT32_MAX, 4);
    RefreshSectionCrc(bad, ProgramSectionType::EXTERNAL_RECORD_STREAM);
    RefreshWholeCrc(bad);
    c.RejectContains("record boundary diagnostic",
                     {"core 0 record 0 at file offset ",
                      "opcode 0x83 (SRAM_CLEAR)",
                      "truncated external record payload"},
                     [&] { DecodeProgramArtifact(bad); });
    bad = EncodeProgramArtifact(CapabilityArtifact());
    WriteLe(bad, 24, 0, 8);
    RefreshWholeCrc(bad);
    c.Reject("record capability gate", [&] { DecodeProgramArtifact(bad); });
}

void CheckEnvelope(Checks &c) {
    ProgramArtifact a = MultiArtifact();
    a.envelope.active_cores = {0, 0};
    c.Reject("duplicate active core", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.envelope.active_cores = {1, 0};
    c.Reject("unordered active cores", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.envelope.terminal_cores = {2};
    a.envelope.expected_done_cores = {2};
    c.Reject("terminal outside active", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.envelope.expected_ack_cores = {0};
    c.Reject("ACK closure", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.envelope.expected_done_cores = {0};
    c.Reject("DONE closure", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.envelope.start_events[0].target_core = 2;
    c.Reject("start outside active", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.envelope.start_events[0].target_core = uint64_t{UINT16_MAX} + 1;
    c.Reject("start core max+1", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.envelope.start_events[0].count = 0;
    c.Reject("start count zero", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.envelope.start_events[0].count = uint64_t{UINT32_MAX} + 1;
    c.Reject("start count max+1", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.envelope.start_events.push_back(a.envelope.start_events.front());
    c.Reject("duplicate start target/tag", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.envelope.empty_core_ack_policy = static_cast<EmptyCoreAckPolicy>(2);
    c.Reject("empty-core policy", [&] { EncodeProgramArtifact(a); });
    a = MultiArtifact();
    a.envelope.failure_policy = static_cast<ProgramFailurePolicy>(1);
    c.Reject("failure policy", [&] { EncodeProgramArtifact(a); });

    auto bad = EncodeProgramArtifact(MultiArtifact());
    const std::size_t eo = static_cast<std::size_t>(
        ReadLe(bad, Descriptor(ProgramSectionType::CONTROL_ENVELOPE) + 8, 8));
    bad[eo + 24] = 1;
    RefreshSectionCrc(bad, ProgramSectionType::CONTROL_ENVELOPE);
    RefreshWholeCrc(bad);
    c.Reject("envelope reserved", [&] { DecodeProgramArtifact(bad); });
    bad = EncodeProgramArtifact(MultiArtifact());
    bad[eo + 2] = 2;
    RefreshSectionCrc(bad, ProgramSectionType::CONTROL_ENVELOPE);
    RefreshWholeCrc(bad);
    c.Reject("decoded empty-core policy",
             [&] { DecodeProgramArtifact(bad); });
    bad = EncodeProgramArtifact(MultiArtifact());
    const std::size_t active = eo + 32;
    WriteLe(bad, active + 2, 0, 2);
    RefreshSectionCrc(bad, ProgramSectionType::CONTROL_ENVELOPE);
    RefreshWholeCrc(bad);
    c.Reject("decoded duplicate active",
             [&] { DecodeProgramArtifact(bad); });
    bad = EncodeProgramArtifact(MultiArtifact());
    const std::size_t sources = active + 4;
    WriteLe(bad, sources + 8, 0, 4);
    RefreshSectionCrc(bad, ProgramSectionType::CONTROL_ENVELOPE);
    RefreshWholeCrc(bad);
    c.Reject("decoded start count zero",
             [&] { DecodeProgramArtifact(bad); });
}

void CheckMutations(Checks &c) {
    const auto golden = EncodeProgramArtifact(MultiArtifact());
    uint32_t state = 0x5eed1234u;
    for (std::size_t i = 0; i < 128; ++i) {
        state = state * 1664525u + 1013904223u;
        auto bad = golden;
        const std::size_t offset = state % bad.size();
        bad[offset] ^=
            static_cast<uint8_t>(1u << ((state >> 24) & 7));
        c.Reject("seed bitflip " + std::to_string(i),
                 [&] { DecodeProgramArtifact(bad); });
    }
    for (std::size_t n = 1; n <= 32; ++n) {
        std::vector<uint8_t> bad(golden.begin(), golden.end() - n);
        c.Reject("seed truncation " + std::to_string(n),
                 [&] { DecodeProgramArtifact(bad); });
    }
}

} // namespace

ProgramFormatSelfTestResult CheckProgramFormatV1() {
    Checks c;
    CheckPositive(c);
    CheckHeaders(c);
    CheckDescriptors(c);
    CheckSymbolsAndRelocations(c);
    CheckGroupsCoresRecords(c);
    CheckEnvelope(c);
    CheckMutations(c);
    return std::move(c.result);
}

int RunProgramFormatV1SelfTest() {
    const ProgramFormatSelfTestResult result = CheckProgramFormatV1();
    if (result.passed()) {
        std::cout << "Program format v1 selftest passed (" << result.checks
                  << " checks)\n";
        return 0;
    }
    std::cerr << "Program format v1 selftest failed ("
              << result.failures.size() << "/" << result.checks << ")\n";
    for (const std::string &failure : result.failures)
        std::cerr << "  " << failure << '\n';
    return 1;
}
