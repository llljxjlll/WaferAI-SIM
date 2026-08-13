#include "prims/norm_prims.h"
#include "utils/prim_utils.h"

#include <limits>
#include <stdexcept>

REGISTER_PRIM(Set_batch, PrimId::SET_BATCH);

namespace {
constexpr size_t kStagesPerSegment = 5;
uint64_t ExpectedId() {
    return static_cast<uint64_t>(
        PrimFactory::getInstance().getPrimId("Set_batch"));
}
void ValidateIdentity(const vector<sc_bv<128>> &segments,
                      size_t expected_segments) {
    if (segments.size() != expected_segments)
        throw std::invalid_argument("Set_batch Prim wire segment count mismatch");
    for (const auto &segment : segments)
        if (segment.range(7, 0).to_uint64() != ExpectedId())
            throw std::invalid_argument(
                "Set_batch Prim wire contains inconsistent segment IDs");
}
} // namespace

int Set_batch::taskCoreDefault(TaskCoreContext &context) {
    prim_context->loop_cnt++;
    prim_context->auto_pd_ = auto_pd;

    prim_context->batch_info_.clear();
    for (auto stage : batch_info) {
        if (auto_pd && prim_context->loop_cnt > auto_pd) {
            LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                            << " auto pd enabled, overriding stage info.";
            prim_context->batch_info_.push_back(
                Stage(prim_context->loop_cnt % auto_pd, PD_PHASE(DECODE), 1));
        } else if (auto_pd > 1)
            prim_context->batch_info_.push_back(Stage(
                prim_context->loop_cnt % auto_pd, stage.type, stage.token_num));
        else
            prim_context->batch_info_.push_back(Stage(stage.req_id, stage.type, stage.token_num));
    }

    return 0;
}

void Set_batch::printSelf() {}

void Set_batch::deserialize(vector<sc_bv<128>> segments) {
    if (segments.empty())
        throw std::invalid_argument("Set_batch Prim wire has no segments");
    const auto &metadata = segments[0];
    const size_t batch_size = metadata.range(23, 8).to_uint64();
    const size_t expected =
        1 + (batch_size + kStagesPerSegment - 1) / kStagesPerSegment;
    const bool legacy = prim_wire::LegacyCompatibilityEnabled();
    if (legacy) {
        if (segments.size() != expected)
            throw std::invalid_argument(
                "Set_batch legacy Prim wire segment count mismatch");
        if (metadata.range(7, 0).to_uint64() != ExpectedId())
            throw std::invalid_argument(
                "Set_batch legacy Prim wire header ID mismatch");
    } else {
        ValidateIdentity(segments, expected);
    }
    const int payload_base = legacy ? 0 : 8;
    if (metadata.range(127, 40).or_reduce())
        throw std::invalid_argument("Set_batch Prim wire reserved bits are set");
    auto_pd = metadata.range(39, 24).to_uint64();
    batch_info.clear();
    batch_info.reserve(batch_size);

    for (size_t i = 1; i < segments.size(); ++i) {
        const auto &segment = segments[i];
        size_t used = 0;
        for (; used < kStagesPerSegment && batch_info.size() < batch_size;
             ++used) {
            const int low = static_cast<int>(payload_base + used * 22);
            const uint64_t raw_phase =
                segment.range(low + 9, low + 8).to_uint64();
            if (raw_phase > static_cast<uint64_t>(PD_DONE))
                throw std::invalid_argument(
                    "Set_batch Prim wire phase is invalid");
            batch_info.emplace_back(
                segment.range(low + 7, low).to_uint64(),
                static_cast<PD_PHASE>(raw_phase),
                segment.range(low + 21, low + 10).to_uint64());
        }
        const int first_reserved =
            static_cast<int>(payload_base + used * 22);
        if (first_reserved <= 127 &&
            segment.range(127, first_reserved).or_reduce())
            throw std::invalid_argument(
                "Set_batch Prim wire padding is non-zero");
    }
}

vector<sc_bv<128>> Set_batch::serialize() {
    if (batch_info.size() > std::numeric_limits<uint16_t>::max())
        throw std::overflow_error(
            "Set_batch batch size exceeds the 16-bit Prim wire");
    if (auto_pd < 0 || auto_pd > std::numeric_limits<uint16_t>::max())
        throw std::overflow_error(
            "Set_batch auto_pd exceeds the 16-bit Prim wire");
    for (const Stage &stage : batch_info) {
        if (stage.req_id < 0 || stage.req_id > 0xff)
            throw std::overflow_error(
                "Set_batch req_id exceeds the 8-bit Prim wire");
        if (stage.type < PREFILL || stage.type > PD_DONE)
            throw std::invalid_argument("Set_batch phase is invalid");
        if (stage.token_num < 0 || stage.token_num > 0xfff)
            throw std::overflow_error(
                "Set_batch token_num exceeds the 12-bit Prim wire");
    }

    vector<sc_bv<128>> segments;
    const uint64_t id = ExpectedId();
    sc_bv<128> metadata = 0;
    metadata.range(7, 0) = sc_bv<8>(id);
    metadata.range(23, 8) = sc_bv<16>(batch_info.size());
    metadata.range(39, 24) = sc_bv<16>(auto_pd);
    segments.push_back(metadata);

    for (size_t i = 0; i < batch_info.size();) {
        sc_bv<128> segment = 0;
        segment.range(7, 0) = sc_bv<8>(id);
        for (size_t slot = 0;
             slot < kStagesPerSegment && i < batch_info.size();
             ++slot, ++i) {
            const int low = static_cast<int>(8 + slot * 22);
            segment.range(low + 7, low) = sc_bv<8>(batch_info[i].req_id);
            segment.range(low + 9, low + 8) =
                sc_bv<2>(batch_info[i].type);
            segment.range(low + 21, low + 10) =
                sc_bv<12>(batch_info[i].token_num);
        }
        segments.push_back(segment);
    }
    return segments;
}