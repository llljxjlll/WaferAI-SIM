#pragma once

#include "dte/coll_reduce_stream.h"

#include <cstdint>
#include <vector>

namespace coll_refactor {

// Byte-visible endpoint representation of one STREAM_V2 contribution.
// Payload bytes use the same little-endian lane layout as SRAM and the
// integer DCA compute pool: byte zero is the low byte of lane zero.
struct IsaV1DcaByteStream {
    ReduceStreamWireHeader header;
    std::vector<ReduceVectorBeat> beats;
};

IsaV1DcaByteStream BuildIsaV1DcaByteStream(
    uint16_t tree_id, const CollectiveKey &collective, uint16_t phase_id,
    uint32_t stream_id, uint16_t source_id, CollDType dtype,
    CollReduceOp reduce_op, const std::vector<uint8_t> &bytes,
    uint64_t vector_bits);

// Validates the complete beat sequence and returns exactly the business
// bytes represented by the header. No padding lane is exposed.
std::vector<uint8_t> DecodeIsaV1DcaByteStream(
    const ReduceStreamWireHeader &header,
    const std::vector<ReduceVectorBeat> &beats,
    uint64_t vector_bits);

} // namespace coll_refactor
