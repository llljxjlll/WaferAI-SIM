#include "prims/collective_data_v1_prim.h"

#include "defs/const.h"
#include "dte/collective_data_v1.h"
#include "memory/sram/sram_access_unit.h"
#include "memory/sram/sram_storage.h"
#include "utils/prim_utils.h"
#include "utils/system_utils.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

REGISTER_PRIM(Collective_data_v1_prim, PrimId::COLLECTIVE_DATA_V1);

namespace {

using Wire = vector<sc_bv<128>>;

constexpr uint32_t kReservedCollectiveId =
    std::numeric_limits<uint32_t>::max();

void Require(bool condition, const std::string &message) {
    if (!condition) throw std::invalid_argument(message);
}

void RequireStrictTransport() {
    if (prim_wire::LegacyCompatibilityEnabled())
        throw std::invalid_argument(
            "Collective_data_v1_prim is strict-only and rejects legacy transport");
}

uint8_t ExpectedId() {
    const int registered =
        PrimFactory::getInstance().getPrimId("Collective_data_v1_prim");
    if (registered != static_cast<int>(
                          PrimIdValue(PrimId::COLLECTIVE_DATA_V1)))
        throw std::logic_error(
            "Collective_data_v1_prim factory ID does not match PrimId 58");
    return static_cast<uint8_t>(registered);
}

uint64_t CheckedMultiply(uint64_t left, uint64_t right,
                         const char *message) {
    if (left != 0 &&
        right > std::numeric_limits<uint64_t>::max() / left)
        throw std::overflow_error(message);
    return left * right;
}

void ValidateSpan(uint64_t address, uint64_t bytes, const char *message) {
    if (bytes != 0 &&
        address > std::numeric_limits<uint64_t>::max() - (bytes - 1))
        throw std::overflow_error(message);
}

uint64_t DTypeBytes(CollDType dtype) {
    switch (dtype) {
    case CollDType::UINT8: return 1;
    case CollDType::INT32: return 4;
    case CollDType::INT64: return 8;
    case CollDType::FP32:
    case CollDType::FP16:
    case CollDType::FP8: break;
    }
    throw std::invalid_argument(
        "Collective_data_v1_prim supports UINT8, INT32, and INT64 only");
}

uint64_t SourceBytes(const Collective_data_v1_prim &prim) {
    return prim.mode == CollectiveDataV1PrimMode::REDUCE
               ? CheckedMultiply(
                     prim.input_count, prim.length_bytes,
                     "Collective_data_v1_prim input_count*L overflows")
               : prim.length_bytes;
}

int ComputeDelay(const Collective_data_v1_prim &prim,
                 const TaskCoreContext &context) {
    if (prim.mode != CollectiveDataV1PrimMode::REDUCE ||
        prim.input_count == 1)
        return 0;
    const uint64_t operations = IsaV1ReduceOperationCount(
        prim.input_count, prim.length_bytes, prim.dtype);
    const CoreHWConfig *hardware = GetCoreHWConfig(context.cid);
    if (hardware == nullptr || hardware->vec == nullptr ||
        hardware->vec->x_dims <= 0 || hardware->vec->count <= 0)
        throw std::runtime_error(
            "Collective_data_v1_prim requires a valid vector unit");
    const uint64_t lanes = CheckedMultiply(
        static_cast<uint64_t>(hardware->vec->x_dims),
        static_cast<uint64_t>(hardware->vec->count),
        "Collective_data_v1_prim vector lane count overflows");
    const uint64_t cycles = operations / lanes + (operations % lanes != 0);
    if (cycles >
        static_cast<uint64_t>(std::numeric_limits<int>::max()) / CYCLE)
        throw std::overflow_error(
            "Collective_data_v1_prim vector delay overflows int");
    return static_cast<int>(cycles * CYCLE);
}

void SetHeader(sc_bv<128> &segment, uint8_t id, uint8_t ordinal) {
    segment.range(7, 0) = sc_bv<8>(id);
    segment.range(15, 8) = sc_bv<8>(ordinal);
}

void ValidateHeaders(const Wire &wire) {
    Require(wire.size() == kCollectiveDataV1PrimWireSegments,
            "Collective_data_v1_prim wire segment count mismatch");
    const uint8_t id = ExpectedId();
    for (size_t index = 0; index < wire.size(); ++index) {
        Require(wire[index].range(7, 0).to_uint64() == id,
                "Collective_data_v1_prim wire has inconsistent segment IDs");
        Require(wire[index].range(15, 8).to_uint64() == index,
                "Collective_data_v1_prim wire has inconsistent ordinals");
    }
}

} // namespace

void Collective_data_v1_prim::Validate() const {
    const auto raw_mode = static_cast<uint8_t>(mode);
    const auto raw_dtype = static_cast<uint8_t>(dtype);
    const auto raw_reduce = static_cast<uint8_t>(reduce_op);
    Require(raw_mode <=
                static_cast<uint8_t>(CollectiveDataV1PrimMode::REDUCE),
            "Collective_data_v1_prim mode enum is invalid");
    Require(raw_dtype <= static_cast<uint8_t>(CollDType::FP8),
            "Collective_data_v1_prim dtype enum is invalid");
    Require(raw_reduce <= static_cast<uint8_t>(CollReduceOp::MAX),
            "Collective_data_v1_prim reduce_op enum is invalid");
    Require(key.group_id != 0,
            "Collective_data_v1_prim group_id must be non-zero");
    Require(key.collective_id != kReservedCollectiveId,
            "Collective_data_v1_prim collective ID is reserved");
    Require(length_bytes != 0,
            "Collective_data_v1_prim L must be positive");
    Require(input_count != 0,
            "Collective_data_v1_prim input_count must be positive");

    if (mode == CollectiveDataV1PrimMode::LOCAL_COPY) {
        Require(input_count == 1 && dtype == CollDType::UINT8 &&
                    reduce_op == CollReduceOp::NONE,
                "Collective_data_v1_prim LOCAL_COPY inactive fields are non-canonical");
    } else {
        const uint64_t width = DTypeBytes(dtype);
        Require(reduce_op == CollReduceOp::SUM ||
                    reduce_op == CollReduceOp::MAX,
                "Collective_data_v1_prim REDUCE requires SUM or MAX");
        Require(length_bytes % width == 0,
                "Collective_data_v1_prim REDUCE L is not dtype aligned");
    }

    const uint64_t source_bytes = SourceBytes(*this);
    if (source_bytes > std::numeric_limits<size_t>::max() ||
        length_bytes > std::numeric_limits<size_t>::max())
        throw std::overflow_error(
            "Collective_data_v1_prim byte size exceeds host size_t");
    ValidateSpan(source_address_bytes, source_bytes,
                 "Collective_data_v1_prim source span overflows");
    ValidateSpan(destination_address_bytes, length_bytes,
                 "Collective_data_v1_prim destination span overflows");
}

Wire Collective_data_v1_prim::serialize() {
    RequireStrictTransport();
    Validate();
    Wire wire(kCollectiveDataV1PrimWireSegments);
    const uint8_t id = ExpectedId();
    for (uint8_t index = 0; index < wire.size(); ++index) {
        wire[index] = 0;
        SetHeader(wire[index], id, index);
    }
    wire[0].range(23, 16) = sc_bv<8>(kCollectiveDataV1PrimWireVersion);
    wire[0].range(31, 24) = sc_bv<8>(kCollectiveDataV1PrimWireSegments);
    wire[0].range(32, 32) = sc_bv<1>(static_cast<uint8_t>(mode));
    wire[0].range(34, 33) = sc_bv<2>(static_cast<uint8_t>(dtype));
    wire[0].range(36, 35) = sc_bv<2>(static_cast<uint8_t>(reduce_op));

    wire[1].range(47, 16) = sc_bv<32>(key.group_id);
    wire[1].range(79, 48) = sc_bv<32>(key.collective_id);
    wire[1].range(111, 80) = sc_bv<32>(key.epoch);
    wire[1].range(127, 112) = sc_bv<16>(phase_id);
    wire[2].range(79, 16) = sc_bv<64>(source_address_bytes);
    wire[3].range(79, 16) = sc_bv<64>(destination_address_bytes);
    wire[4].range(79, 16) = sc_bv<64>(length_bytes);
    wire[5].range(31, 16) = sc_bv<16>(input_count);
    return wire;
}

void Collective_data_v1_prim::deserialize(Wire wire) {
    RequireStrictTransport();
    ValidateHeaders(wire);
    Require(wire[0].range(23, 16).to_uint64() ==
                kCollectiveDataV1PrimWireVersion,
            "Collective_data_v1_prim wire version is unsupported");
    Require(wire[0].range(31, 24).to_uint64() ==
                kCollectiveDataV1PrimWireSegments,
            "Collective_data_v1_prim wire count field is inconsistent");
    Require(!wire[0].range(127, 37).or_reduce(),
            "Collective_data_v1_prim metadata reserved bits are non-zero");
    Require(!wire[2].range(127, 80).or_reduce() &&
                !wire[3].range(127, 80).or_reduce() &&
                !wire[4].range(127, 80).or_reduce() &&
                !wire[5].range(127, 32).or_reduce(),
            "Collective_data_v1_prim numeric reserved bits are non-zero");

    Collective_data_v1_prim decoded;
    decoded.mode = static_cast<CollectiveDataV1PrimMode>(
        wire[0].range(32, 32).to_uint64());
    decoded.dtype = static_cast<CollDType>(
        wire[0].range(34, 33).to_uint64());
    decoded.reduce_op = static_cast<CollReduceOp>(
        wire[0].range(36, 35).to_uint64());
    decoded.key.group_id = static_cast<uint32_t>(
        wire[1].range(47, 16).to_uint64());
    decoded.key.collective_id = static_cast<uint32_t>(
        wire[1].range(79, 48).to_uint64());
    decoded.key.epoch = static_cast<uint32_t>(
        wire[1].range(111, 80).to_uint64());
    decoded.phase_id = static_cast<uint16_t>(
        wire[1].range(127, 112).to_uint64());
    decoded.source_address_bytes = wire[2].range(79, 16).to_uint64();
    decoded.destination_address_bytes = wire[3].range(79, 16).to_uint64();
    decoded.length_bytes = wire[4].range(79, 16).to_uint64();
    decoded.input_count = static_cast<uint16_t>(
        wire[5].range(31, 16).to_uint64());
    decoded.Validate();

    mode = decoded.mode;
    key = decoded.key;
    phase_id = decoded.phase_id;
    source_address_bytes = decoded.source_address_bytes;
    destination_address_bytes = decoded.destination_address_bytes;
    length_bytes = decoded.length_bytes;
    input_count = decoded.input_count;
    dtype = decoded.dtype;
    reduce_op = decoded.reduce_op;
}

int Collective_data_v1_prim::taskCoreDefault(TaskCoreContext &context) {
    Validate();
    Require(context.sram_access != nullptr && context.sram_storage != nullptr,
            "Collective_data_v1_prim requires the unified SRAM data path");
    Require(context.sram_storage->payload_mode(),
            "Collective_data_v1_prim requires real SRAM payload mode");
    const int compute_delay = ComputeDelay(*this, context);
    const uint64_t source_bytes = SourceBytes(*this);

    sram::Request read;
    read.initiator = sram::Initiator::kCompute;
    read.command = sram::Command::kRead;
    read.address = source_address_bytes;
    read.size_bytes = source_bytes;
    std::vector<uint8_t> source =
        context.sram_access->Access(read).payload;
    if (source.size() != source_bytes)
        throw std::runtime_error(
            "Collective_data_v1_prim SRAM read returned the wrong byte count");

    std::vector<uint8_t> result;
    if (mode == CollectiveDataV1PrimMode::LOCAL_COPY || input_count == 1) {
        result.assign(source.begin(), source.begin() +
                                        static_cast<size_t>(length_bytes));
    } else {
        IsaV1CollectiveDataBuffer buffer(input_count, length_bytes,
                                         source_bytes);
        const size_t bytes_per_rank = static_cast<size_t>(length_bytes);
        for (uint16_t rank = 0; rank < input_count; ++rank) {
            const size_t begin = static_cast<size_t>(rank) * bytes_per_rank;
            buffer.Accept(
                rank, 0,
                std::vector<uint8_t>(source.begin() + begin,
                                     source.begin() + begin + bytes_per_rank));
        }
        result = buffer.TakeReduced(dtype, reduce_op);
        if (!buffer.Residual().Drained())
            throw std::logic_error(
                "Collective_data_v1_prim reduction buffer did not drain");
    }

    sram::Request write;
    write.initiator = sram::Initiator::kCompute;
    write.command = sram::Command::kWrite;
    write.address = destination_address_bytes;
    write.size_bytes = length_bytes;
    write.payload = std::move(result);
    context.sram_access->Access(write);
    return compute_delay;
}

void Collective_data_v1_prim::printSelf() {}
