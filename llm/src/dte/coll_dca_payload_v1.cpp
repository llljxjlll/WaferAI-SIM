#include "dte/coll_dca_payload_v1.h"

#include "dte/coll_stream_engine.h"

#include <limits>
#include <stdexcept>

namespace coll_refactor {
namespace {

uint64_t DtypeBytes(CollDType dtype) {
    switch (dtype) {
    case CollDType::UINT8: return 1;
    case CollDType::INT32: return 4;
    case CollDType::INT64: return 8;
    case CollDType::FP32:
    case CollDType::FP16:
    case CollDType::FP8:
        break;
    }
    throw std::invalid_argument(
        "ISA-v1 DCA byte payload requires an integer dtype");
}

uint64_t LoadLane(const std::vector<uint8_t> &bytes, size_t offset,
                  uint64_t width) {
    uint64_t value = 0;
    for (uint64_t byte = 0; byte < width; ++byte)
        value |= static_cast<uint64_t>(bytes.at(offset + byte)) <<
                 (byte * 8);
    return value;
}

void StoreLane(uint64_t value, uint64_t width,
               std::vector<uint8_t> *bytes) {
    for (uint64_t byte = 0; byte < width; ++byte)
        bytes->push_back(static_cast<uint8_t>(value >> (byte * 8)));
}

} // namespace

IsaV1DcaByteStream BuildIsaV1DcaByteStream(
    uint16_t tree_id, const CollectiveKey &collective, uint16_t phase_id,
    uint32_t stream_id, uint16_t source_id, CollDType dtype,
    CollReduceOp reduce_op, const std::vector<uint8_t> &bytes,
    uint64_t vector_bits) {
    const uint64_t dtype_bytes = DtypeBytes(dtype);
    if (bytes.empty() || bytes.size() % dtype_bytes != 0)
        throw std::invalid_argument(
            "ISA-v1 DCA source bytes must contain whole integer elements");
    const uint64_t elements = bytes.size() / dtype_bytes;
    const VectorWork work =
        ComputeVectorWork(elements, 1, vector_bits, dtype);

    IsaV1DcaByteStream result;
    result.header.stream.tree_id = tree_id;
    result.header.stream.key = {collective, phase_id, stream_id};
    result.header.stream.dtype = dtype;
    result.header.stream.op = reduce_op;
    result.header.stream.total_elements = elements;
    result.header.stream.physical_data_flits = CollCeilDiv(
        CollCheckedMul(elements, CollDTypeBits(dtype)), 128);
    result.header.stream.vector_beats = work.vector_beats;
    result.header.source_id = source_id;
    result.header.tail_valid_lanes = work.tail_valid_lanes;
    result.header.Validate(128, vector_bits);

    result.beats.reserve(static_cast<size_t>(work.vector_beats));
    size_t offset = 0;
    for (uint64_t beat_id = 0; beat_id < work.vector_beats; ++beat_id) {
        const bool final = beat_id + 1 == work.vector_beats;
        const uint64_t valid = final ? work.tail_valid_lanes : work.lanes;
        VectorBeat geometry{dtype, vector_bits, {work.lanes, valid}};
        std::vector<uint64_t> values(work.lanes, 0);
        for (uint64_t lane = 0; lane < valid; ++lane) {
            values[lane] = LoadLane(bytes, offset, dtype_bytes);
            offset += static_cast<size_t>(dtype_bytes);
        }
        result.beats.push_back(PackReduceVectorValues(
            {result.header.stream.key, result.header.reduce_stage_id,
             beat_id},
            geometry, values));
    }
    if (offset != bytes.size())
        throw std::logic_error(
            "ISA-v1 DCA byte pack did not consume the source exactly");
    return result;
}

std::vector<uint8_t> DecodeIsaV1DcaByteStream(
    const ReduceStreamWireHeader &header,
    const std::vector<ReduceVectorBeat> &beats,
    uint64_t vector_bits) {
    header.Validate(128, vector_bits);
    if (beats.size() != header.stream.vector_beats)
        throw std::invalid_argument(
            "ISA-v1 DCA result beat count disagrees with its header");
    const uint64_t dtype_bytes = DtypeBytes(header.stream.dtype);
    const VectorWork work = ComputeVectorWork(
        header.stream.total_elements, 1, vector_bits, header.stream.dtype);
    if (header.stream.total_elements >
        std::numeric_limits<size_t>::max() / dtype_bytes)
        throw std::overflow_error(
            "ISA-v1 DCA result byte count overflows size_t");
    std::vector<uint8_t> bytes;
    bytes.reserve(static_cast<size_t>(
        header.stream.total_elements * dtype_bytes));
    for (uint64_t beat_id = 0; beat_id < work.vector_beats; ++beat_id) {
        const ReduceVectorBeat &beat = beats.at(static_cast<size_t>(beat_id));
        const bool final = beat_id + 1 == work.vector_beats;
        const uint64_t valid = final ? work.tail_valid_lanes : work.lanes;
        if (!(beat.key.stream == header.stream.key) ||
            beat.key.reduce_stage_id != header.reduce_stage_id ||
            beat.key.vector_beat_id != beat_id ||
            beat.geometry.dtype != header.stream.dtype ||
            beat.geometry.vector_bits != vector_bits ||
            beat.geometry.lane_mask.lane_count != work.lanes ||
            beat.geometry.lane_mask.valid_lanes != valid)
            throw std::invalid_argument(
                "ISA-v1 DCA result beat identity/geometry mismatch");
        const std::vector<uint64_t> values =
            UnpackReduceVectorValues(beat);
        for (uint64_t lane = 0; lane < valid; ++lane)
            StoreLane(values[lane], dtype_bytes, &bytes);
    }
    const size_t expected = static_cast<size_t>(
        header.stream.total_elements * dtype_bytes);
    if (bytes.size() != expected)
        throw std::logic_error(
            "ISA-v1 DCA result decode produced the wrong byte count");
    return bytes;
}

} // namespace coll_refactor
