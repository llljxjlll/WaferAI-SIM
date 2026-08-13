#include "systemc.h"

#include "common/memory.h"
#include "defs/global.h"
#include "prims/base.h"
#include "prims/norm_prims.h"
#include "utils/prim_utils.h"
#include "utils/system_utils.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

REGISTER_PRIM(Set_addr, PrimId::SET_ADDR);

namespace {
constexpr size_t kLabelsPerSegment = 3;
constexpr size_t kInputSegments =
    (MAX_SPLIT_NUM + kLabelsPerSegment - 1) / kLabelsPerSegment;
constexpr size_t kSetAddrSegments = 1 + kInputSegments + 1;

uint64_t ExpectedId() {
    return static_cast<uint64_t>(
        PrimFactory::getInstance().getPrimId("Set_addr"));
}

void ValidateIdentity(const vector<sc_bv<128>> &segments) {
    if (segments.size() != kSetAddrSegments)
        throw std::invalid_argument("Set_addr Prim wire segment count mismatch");
    for (const auto &segment : segments)
        if (segment.range(7, 0).to_uint64() != ExpectedId())
            throw std::invalid_argument(
                "Set_addr Prim wire contains inconsistent segment IDs");
}

uint32_t ReadLabelId(const sc_bv<128> &segment, int low) {
    const uint64_t raw = segment.range(low + 31, low).to_uint64();
    if (raw == 0 || raw > g_addr_label_table.table.size())
        throw std::invalid_argument("Set_addr Prim wire label ID is unknown");
    return static_cast<uint32_t>(raw);
}

uint32_t AddLabel(const std::string &label) {
    const int raw = g_addr_label_table.addRecord(label);
    if (raw <= 0)
        throw std::overflow_error("Set_addr label table ID is invalid");
    return static_cast<uint32_t>(raw);
}
} // namespace

void Set_addr::printSelf() {}

void Set_addr::deserialize(vector<sc_bv<128>> segments) {
    if (prim_wire::LegacyCompatibilityEnabled()) {
        constexpr size_t kLegacyLabelsPerSegment = 4;
        constexpr size_t kLegacyInputSegments =
            (MAX_SPLIT_NUM + kLegacyLabelsPerSegment - 1) /
            kLegacyLabelsPerSegment;
        if (segments.size() != 1 + kLegacyInputSegments + 1)
            throw std::invalid_argument(
                "Set_addr legacy Prim wire segment count mismatch");
        if (segments.front().range(7, 0).to_uint64() != ExpectedId())
            throw std::invalid_argument(
                "Set_addr legacy Prim wire header ID mismatch");
        const auto &metadata = segments.front();
        if (metadata.range(127, 34).or_reduce())
            throw std::invalid_argument(
                "Set_addr legacy Prim wire reserved bits are set");
        sram_addr = metadata.range(31, 8).to_uint64();
        const uint64_t raw_datatype = metadata.range(33, 32).to_uint64();
        if (raw_datatype > static_cast<uint64_t>(FP16))
            throw std::invalid_argument(
                "Set_addr legacy Prim wire datatype is invalid");
        datatype = static_cast<DATATYPE>(raw_datatype);

        for (size_t index = 0; index < MAX_SPLIT_NUM; ++index) {
            const size_t segment = 1 + index / kLegacyLabelsPerSegment;
            const int low = static_cast<int>(
                (index % kLegacyLabelsPerSegment) * 32);
            datapass_label.indata[index] = g_addr_label_table.findRecord(
                ReadLabelId(segments[segment], low));
        }
        if (segments.back().range(127, 32).or_reduce())
            throw std::invalid_argument(
                "Set_addr legacy Prim wire output padding is non-zero");
        datapass_label.outdata = g_addr_label_table.findRecord(
            ReadLabelId(segments.back(), 0));
        return;
    }

    ValidateIdentity(segments);
    const auto &metadata = segments[0];
    if (metadata.range(127, 34).or_reduce())
        throw std::invalid_argument("Set_addr Prim wire reserved bits are set");
    sram_addr = metadata.range(31, 8).to_uint64();
    const uint64_t raw_datatype = metadata.range(33, 32).to_uint64();
    if (raw_datatype > static_cast<uint64_t>(FP16))
        throw std::invalid_argument("Set_addr Prim wire datatype is invalid");
    datatype = static_cast<DATATYPE>(raw_datatype);

    int read_label_cnt = 0;
    for (size_t i = 1; i <= kInputSegments; ++i) {
        const auto &segment = segments[i];
        for (size_t slot = 0; slot < kLabelsPerSegment &&
                              read_label_cnt < MAX_SPLIT_NUM;
             ++slot, ++read_label_cnt) {
            const int low = static_cast<int>(8 + slot * 32);
            datapass_label.indata[read_label_cnt] =
                g_addr_label_table.findRecord(ReadLabelId(segment, low));
        }
        const size_t used = std::min<size_t>(
            kLabelsPerSegment, MAX_SPLIT_NUM -
                (i - 1) * kLabelsPerSegment);
        const int first_reserved = static_cast<int>(8 + used * 32);
        if (first_reserved <= 127 &&
            segment.range(127, first_reserved).or_reduce())
            throw std::invalid_argument(
                "Set_addr Prim wire input padding is non-zero");
    }

    const auto &output = segments.back();
    if (output.range(127, 40).or_reduce())
        throw std::invalid_argument(
            "Set_addr Prim wire output padding is non-zero");
    datapass_label.outdata =
        g_addr_label_table.findRecord(ReadLabelId(output, 8));
}

vector<sc_bv<128>> Set_addr::serialize() {
    if (sram_addr < 0 || sram_addr > 0xffffff)
        throw std::overflow_error(
            "Set_addr sram_addr exceeds the 24-bit Prim wire");
    if (datatype != INT8 && datatype != FP16)
        throw std::invalid_argument("Set_addr datatype is invalid");

    const AddrDatapassLabel *labels = &datapass_label;
    if (prim_context != nullptr && prim_context->datapass_label_ != nullptr)
        labels = prim_context->datapass_label_;

    vector<sc_bv<128>> segments;
    const uint64_t id = ExpectedId();
    sc_bv<128> metadata = 0;
    metadata.range(7, 0) = sc_bv<8>(id);
    metadata.range(31, 8) = sc_bv<24>(sram_addr);
    metadata.range(33, 32) = sc_bv<2>(datatype);
    segments.push_back(metadata);

    int label_idx = 0;
    while (label_idx < MAX_SPLIT_NUM) {
        sc_bv<128> segment = 0;
        segment.range(7, 0) = sc_bv<8>(id);
        for (size_t slot = 0; slot < kLabelsPerSegment &&
                              label_idx < MAX_SPLIT_NUM;
             ++slot, ++label_idx) {
            const int low = static_cast<int>(8 + slot * 32);
            segment.range(low + 31, low) = sc_bv<32>(AddLabel(
                labels->indata[label_idx]));
        }
        segments.push_back(segment);
    }

    sc_bv<128> output = 0;
    output.range(7, 0) = sc_bv<8>(id);
    output.range(39, 8) = sc_bv<32>(AddLabel(
        labels->outdata));
    segments.push_back(output);
    return segments;
}

int Set_addr::taskCoreDefault(TaskCoreContext &context) {
    //  将datapass_label的内容复制到target中
    for (int i = 0; i < MAX_SPLIT_NUM; i++) {
        prim_context->datapass_label_->indata[i] = datapass_label.indata[i];
    }
    prim_context->datapass_label_->outdata = datapass_label.outdata;

    return 0;
}