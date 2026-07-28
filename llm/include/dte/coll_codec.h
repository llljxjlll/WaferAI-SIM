#pragma once

#include "dte/coll_types.h"
#include "systemc.h"

#include <limits>
#include <stdexcept>
#include <vector>

constexpr uint16_t COLL_WIRE_MAGIC = 0xC011;
constexpr uint8_t COLL_WIRE_VERSION = 0;
constexpr size_t COLL_WIRE_FIXED_SEGMENTS = 5;
constexpr size_t COLL_WIRE_GROUP_IDS_PER_SEGMENT = 8;

inline std::vector<sc_bv<128>> SerializeCollDescriptor(const CollDescriptor &d) {
    ValidateCollDescriptor(d);
    if (d.group.size() > std::numeric_limits<uint16_t>::max())
        throw std::overflow_error("collective group exceeds wire capacity");
    const size_t group_segments =
        (d.group.size() + COLL_WIRE_GROUP_IDS_PER_SEGMENT - 1) /
        COLL_WIRE_GROUP_IDS_PER_SEGMENT;
    if (COLL_WIRE_FIXED_SEGMENTS + group_segments > std::numeric_limits<uint16_t>::max())
        throw std::overflow_error("collective descriptor segment count overflow");
    std::vector<sc_bv<128>> wire(COLL_WIRE_FIXED_SEGMENTS + group_segments);
    wire[0].range(15, 0) = COLL_WIRE_MAGIC;
    wire[0].range(23, 16) = COLL_WIRE_VERSION;
    wire[0].range(31, 24) = static_cast<uint8_t>(d.op);
    wire[0].range(39, 32) = static_cast<uint8_t>(d.algorithm);
    wire[0].range(47, 40) = static_cast<uint8_t>(d.dtype);
    wire[0].range(55, 48) = static_cast<uint8_t>(d.reduce_op);
    wire[0].range(71, 56) = d.root_rank;
    wire[0].range(87, 72) = d.self_rank;
    wire[0].range(103, 88) = static_cast<uint16_t>(d.group.size());
    wire[0].range(119, 104) = static_cast<uint16_t>(wire.size());
    wire[1].range(31, 0) = d.key.group_id;
    wire[1].range(63, 32) = d.key.collective_id;
    wire[1].range(95, 64) = d.key.epoch;
    wire[1].range(127, 96) = d.gather_reorder_depth;
    wire[2].range(63, 0) = d.count;
    wire[2].range(127, 64) = d.chunk_bits;
    wire[3].range(63, 0) = d.src_addr;
    wire[3].range(127, 64) = d.dst_addr;
    wire[4].range(63, 0) = d.stride_bits;
    for (size_t i = 0; i < d.group.size(); ++i) {
        const size_t segment = COLL_WIRE_FIXED_SEGMENTS + i / COLL_WIRE_GROUP_IDS_PER_SEGMENT;
        const size_t low = (i % COLL_WIRE_GROUP_IDS_PER_SEGMENT) * 16;
        wire[segment].range(low + 15, low) = d.group[i];
    }
    return wire;
}

inline CollDescriptor DeserializeCollDescriptor(const std::vector<sc_bv<128>> &wire) {
    if (wire.size() < COLL_WIRE_FIXED_SEGMENTS)
        throw std::invalid_argument("truncated collective descriptor");
    if (wire[0].range(15, 0).to_uint() != COLL_WIRE_MAGIC)
        throw std::invalid_argument("invalid collective descriptor magic");
    if (wire[0].range(23, 16).to_uint() != COLL_WIRE_VERSION)
        throw std::invalid_argument("unsupported collective descriptor version");
    const size_t encoded_segments = wire[0].range(119, 104).to_uint();
    const size_t group_size = wire[0].range(103, 88).to_uint();
    const size_t expected = COLL_WIRE_FIXED_SEGMENTS +
        (group_size + COLL_WIRE_GROUP_IDS_PER_SEGMENT - 1) /
            COLL_WIRE_GROUP_IDS_PER_SEGMENT;
    if (encoded_segments != expected || wire.size() != expected)
        throw std::invalid_argument("collective descriptor segment count mismatch");
    const unsigned raw_op = wire[0].range(31, 24).to_uint();
    const unsigned raw_algorithm = wire[0].range(39, 32).to_uint();
    const unsigned raw_dtype = wire[0].range(47, 40).to_uint();
    const unsigned raw_reduce = wire[0].range(55, 48).to_uint();
    if (raw_op > static_cast<unsigned>(CollOp::ALLREDUCE) ||
        raw_algorithm > static_cast<unsigned>(CollAlgorithm::REDUCE_ROOT_BROADCAST) ||
        raw_dtype > static_cast<unsigned>(CollDType::FP32) ||
        raw_reduce > static_cast<unsigned>(CollReduceOp::MAX))
        throw std::invalid_argument("collective descriptor enum is out of range");
    CollDescriptor d;
    d.op = static_cast<CollOp>(raw_op);
    d.algorithm = static_cast<CollAlgorithm>(raw_algorithm);
    d.dtype = static_cast<CollDType>(raw_dtype);
    d.reduce_op = static_cast<CollReduceOp>(raw_reduce);
    d.root_rank = wire[0].range(71, 56).to_uint();
    d.self_rank = wire[0].range(87, 72).to_uint();
    d.key.group_id = wire[1].range(31, 0).to_uint64();
    d.key.collective_id = wire[1].range(63, 32).to_uint64();
    d.key.epoch = wire[1].range(95, 64).to_uint64();
    d.gather_reorder_depth = wire[1].range(127, 96).to_uint64();
    d.count = wire[2].range(63, 0).to_uint64();
    d.chunk_bits = wire[2].range(127, 64).to_uint64();
    d.src_addr = wire[3].range(63, 0).to_uint64();
    d.dst_addr = wire[3].range(127, 64).to_uint64();
    d.stride_bits = wire[4].range(63, 0).to_uint64();
    d.group.reserve(group_size);
    for (size_t i = 0; i < group_size; ++i) {
        const size_t segment = COLL_WIRE_FIXED_SEGMENTS + i / COLL_WIRE_GROUP_IDS_PER_SEGMENT;
        const size_t low = (i % COLL_WIRE_GROUP_IDS_PER_SEGMENT) * 16;
        d.group.push_back(wire[segment].range(low + 15, low).to_uint());
    }
    ValidateCollDescriptor(d);
    return d;
}
