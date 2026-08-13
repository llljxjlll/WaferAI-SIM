#include "common/memory.h"
#include "prims/norm_prims.h"
#include "utils/prim_utils.h"

#include <stdexcept>
#include <utility>

REGISTER_PRIM(Sram_bind_oneshot, PrimId::SRAM_BIND_ONESHOT);

namespace {
constexpr size_t kLabelsPerSegment = 3;
constexpr size_t kInputSegments = 6;
constexpr size_t kWireSegments = 1 + kInputSegments + 1;

uint64_t ExpectedId() {
    return static_cast<uint64_t>(
        PrimFactory::getInstance().getPrimId("Sram_bind_oneshot"));
}

void ValidateInputCount(uint32_t input_count) {
    if (input_count == 0 || input_count > MAX_SPLIT_NUM)
        throw std::invalid_argument(
            "Sram_bind_oneshot input_count must be in [1, 16]");
}

void ValidateLabels(const AddrDatapassLabel &labels, uint32_t input_count) {
    for (uint32_t index = 0; index < input_count; ++index) {
        if (labels.indata[index].empty() ||
            labels.indata[index] == UNSET_LABEL)
            throw std::invalid_argument(
                "Sram_bind_oneshot used input label is unset");
    }
    for (uint32_t index = input_count; index < MAX_SPLIT_NUM; ++index) {
        if (labels.indata[index] != UNSET_LABEL)
            throw std::invalid_argument(
                "Sram_bind_oneshot unused input label is set");
    }
    if (labels.outdata.empty() || labels.outdata == UNSET_LABEL)
        throw std::invalid_argument(
            "Sram_bind_oneshot output label is unset");
}

uint32_t AddLabel(const std::string &label) {
    const int raw = g_addr_label_table.addRecord(label);
    if (raw <= 0)
        throw std::overflow_error(
            "Sram_bind_oneshot label table ID is invalid");
    return static_cast<uint32_t>(raw);
}

std::string ReadLabel(uint64_t raw) {
    if (raw == 0 || raw > g_addr_label_table.table.size())
        throw std::invalid_argument(
            "Sram_bind_oneshot Prim wire label ID is unknown");
    return g_addr_label_table.findRecord(static_cast<int>(raw));
}

void ValidateIdentity(const vector<sc_bv<128>> &segments) {
    if (segments.size() != kWireSegments)
        throw std::invalid_argument(
            "Sram_bind_oneshot Prim wire segment count mismatch");
    for (const auto &segment : segments) {
        if (segment.range(7, 0).to_uint64() != ExpectedId())
            throw std::invalid_argument(
                "Sram_bind_oneshot Prim wire contains inconsistent segment IDs");
    }
}
} // namespace

void Sram_bind_oneshot::printSelf() {}

vector<sc_bv<128>> Sram_bind_oneshot::serialize() {
    ValidateInputCount(input_count);
    ValidateLabels(datapass_label, input_count);

    const uint64_t id = ExpectedId();
    vector<sc_bv<128>> segments(kWireSegments);
    for (auto &segment : segments) {
        segment = 0;
        segment.range(7, 0) = sc_bv<8>(id);
    }
    segments[0].range(15, 8) = sc_bv<8>(input_count);

    for (uint32_t index = 0; index < input_count; ++index) {
        const size_t segment = 1 + index / kLabelsPerSegment;
        const int low = static_cast<int>(8 +
                                         (index % kLabelsPerSegment) * 32);
        segments[segment].range(low + 31, low) =
            sc_bv<32>(AddLabel(datapass_label.indata[index]));
    }
    segments.back().range(39, 8) =
        sc_bv<32>(AddLabel(datapass_label.outdata));
    return segments;
}

void Sram_bind_oneshot::deserialize(vector<sc_bv<128>> segments) {
    ValidateIdentity(segments);
    if (segments[0].range(127, 16).or_reduce())
        throw std::invalid_argument(
            "Sram_bind_oneshot Prim wire metadata padding is non-zero");

    const uint32_t decoded_count =
        static_cast<uint32_t>(segments[0].range(15, 8).to_uint64());
    ValidateInputCount(decoded_count);
    AddrDatapassLabel decoded_labels;

    for (size_t payload = 0; payload < kInputSegments; ++payload) {
        const auto &segment = segments[1 + payload];
        if (segment.range(127, 104).or_reduce())
            throw std::invalid_argument(
                "Sram_bind_oneshot Prim wire input padding is non-zero");
        for (size_t slot = 0; slot < kLabelsPerSegment; ++slot) {
            const size_t index = payload * kLabelsPerSegment + slot;
            const int low = static_cast<int>(8 + slot * 32);
            const uint64_t label_id =
                segment.range(low + 31, low).to_uint64();
            if (index < decoded_count) {
                decoded_labels.indata[index] = ReadLabel(label_id);
            } else if (label_id != 0) {
                throw std::invalid_argument(
                    "Sram_bind_oneshot Prim wire unused input slot is non-zero");
            }
        }
    }

    const auto &output = segments.back();
    if (output.range(127, 40).or_reduce())
        throw std::invalid_argument(
            "Sram_bind_oneshot Prim wire output padding is non-zero");
    decoded_labels.outdata =
        ReadLabel(output.range(39, 8).to_uint64());

    input_count = decoded_count;
    datapass_label = std::move(decoded_labels);
}

int Sram_bind_oneshot::taskCoreDefault(TaskCoreContext &) {
    if (prim_context == nullptr)
        throw std::logic_error(
            "Sram_bind_oneshot requires a PrimCoreContext");
    ValidateInputCount(input_count);
    ValidateLabels(datapass_label, input_count);
    if (prim_context->sram_bind_pending_)
        throw std::logic_error(
            "Sram_bind_oneshot cannot overwrite a pending binding");

    prim_context->program_mode_ = true;
    prim_context->sram_bind_input_count_ = input_count;
    prim_context->sram_bind_pending_labels_ = datapass_label;
    prim_context->sram_bind_pending_ = true;
    return 0;
}
