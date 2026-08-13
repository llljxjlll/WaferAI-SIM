#include "dte/coll_byte_wire_v1.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace {

uint32_t FragmentCount(uint32_t total_bytes) {
    return total_bytes / P2P_PAYLOAD_FRAGMENT_BYTES +
           (total_bytes % P2P_PAYLOAD_FRAGMENT_BYTES != 0 ? 1U : 0U);
}

void ValidateKind(IsaV1CollectiveByteKind kind) {
    if (static_cast<uint8_t>(kind) >
        static_cast<uint8_t>(IsaV1CollectiveByteKind::DCA_REDUCE))
        throw std::invalid_argument(
            "ISA-v1 collective byte wire kind is unsupported");
}

void ValidateIdentity(uint16_t tree_id, uint16_t session_id) {
    if (tree_id == 0)
        throw std::invalid_argument(
            "ISA-v1 collective byte wire tree_id zero is reserved");
    if (session_id == 0)
        throw std::invalid_argument(
            "ISA-v1 collective byte wire session_id zero is reserved");
}

void ValidateStart(const IsaV1CollectiveByteStart &start) {
    ValidateKind(start.kind);
    ValidateIdentity(start.tree_id, start.session_id);
    if (start.total_bytes == 0)
        throw std::invalid_argument(
            "ISA-v1 collective byte START total_bytes must be positive");
    if (start.total_bytes > kDteEndpointP2pMaxBytes)
        throw std::length_error(
            "ISA-v1 collective byte START exceeds endpoint byte limit");
    if (FragmentCount(start.total_bytes) > P2P_PAYLOAD_MAX_FRAGMENTS)
        throw std::length_error(
            "ISA-v1 collective byte START exceeds DATA sequence space");
    if (start.flags != ISA_V1_COLL_BYTE_START_FLAGS)
        throw std::invalid_argument(
            "ISA-v1 collective byte START flags are unsupported");
}

void ValidateData(const IsaV1CollectiveByteData &data) {
    ValidateKind(data.kind);
    ValidateIdentity(data.lock.tree_id, data.lock.session_id);
    if (data.sequence == 0)
        throw std::invalid_argument(
            "ISA-v1 collective byte DATA sequence zero is reserved");
    if (data.length_bytes == 0 ||
        data.length_bytes > P2P_PAYLOAD_FRAGMENT_BYTES)
        throw std::invalid_argument(
            "ISA-v1 collective byte DATA length must be in [1,16]");
    if (!data.tail && data.length_bytes != P2P_PAYLOAD_FRAGMENT_BYTES)
        throw std::invalid_argument(
            "ISA-v1 collective byte non-tail DATA must contain 16 bytes");
    for (unsigned bit = static_cast<unsigned>(data.length_bytes) * 8;
         bit < 128; ++bit)
        if (data.payload[bit].to_bool())
            throw std::invalid_argument(
                "ISA-v1 collective byte DATA padding is non-zero");
}

void AppendPayload(std::vector<uint8_t> &bytes,
                   const IsaV1CollectiveByteData &data) {
    for (uint8_t index = 0; index < data.length_bytes; ++index)
        bytes.push_back(static_cast<uint8_t>(
            data.payload.range(index * 8 + 7, index * 8).to_uint()));
}

IsaV1CollectiveByteLock LockFor(const IsaV1CollectiveByteStart &start) {
    return {start.tree_id, start.session_id, start.collective.epoch};
}

} // namespace

bool IsaV1CollectiveByteLock::operator==(
    const IsaV1CollectiveByteLock &other) const noexcept {
    return std::tie(tree_id, session_id, epoch) ==
           std::tie(other.tree_id, other.session_id, other.epoch);
}

bool IsaV1CollectiveByteLock::operator<(
    const IsaV1CollectiveByteLock &other) const noexcept {
    return std::tie(tree_id, session_id, epoch) <
           std::tie(other.tree_id, other.session_id, other.epoch);
}

bool IsIsaV1CollectiveByteStartWire(const sc_bv<256> &wire) noexcept {
    return wire.range(15, 0).to_uint() == ISA_V1_COLL_BYTE_START_MAGIC &&
           wire.range(19, 16).to_uint() == ISA_V1_COLL_BYTE_VERSION;
}

bool IsIsaV1CollectiveByteDataWire(const sc_bv<256> &wire) noexcept {
    return wire.range(15, 0).to_uint() == ISA_V1_COLL_BYTE_DATA_MAGIC &&
           wire.range(19, 16).to_uint() == ISA_V1_COLL_BYTE_VERSION;
}

sc_bv<256> SerializeIsaV1CollectiveByteStart(
    const IsaV1CollectiveByteStart &start) {
    ValidateStart(start);
    sc_bv<256> wire = 0;
    wire.range(15, 0) = ISA_V1_COLL_BYTE_START_MAGIC;
    wire.range(19, 16) = ISA_V1_COLL_BYTE_VERSION;
    wire.range(23, 20) = static_cast<uint8_t>(start.kind);
    wire.range(39, 24) = start.tree_id;
    wire.range(55, 40) = start.session_id;
    wire.range(87, 56) = start.collective.group_id;
    wire.range(119, 88) = start.collective.collective_id;
    wire.range(151, 120) = start.collective.epoch;
    wire.range(183, 152) = start.total_bytes;
    wire.range(215, 184) = start.checksum;
    wire.range(223, 216) = start.flags;
    return wire;
}

IsaV1CollectiveByteStart DeserializeIsaV1CollectiveByteStart(
    const sc_bv<256> &wire) {
    if (wire.range(15, 0).to_uint() != ISA_V1_COLL_BYTE_START_MAGIC)
        throw std::invalid_argument(
            "invalid ISA-v1 collective byte START magic");
    if (wire.range(19, 16).to_uint() != ISA_V1_COLL_BYTE_VERSION)
        throw std::invalid_argument(
            "unsupported ISA-v1 collective byte START version");
    if (wire.range(255, 224).or_reduce())
        throw std::invalid_argument(
            "ISA-v1 collective byte START reserved bits are non-zero");
    IsaV1CollectiveByteStart start;
    start.kind = static_cast<IsaV1CollectiveByteKind>(
        wire.range(23, 20).to_uint());
    start.tree_id = wire.range(39, 24).to_uint();
    start.session_id = wire.range(55, 40).to_uint();
    start.collective.group_id = wire.range(87, 56).to_uint64();
    start.collective.collective_id = wire.range(119, 88).to_uint64();
    start.collective.epoch = wire.range(151, 120).to_uint64();
    start.total_bytes = wire.range(183, 152).to_uint64();
    start.checksum = wire.range(215, 184).to_uint64();
    start.flags = wire.range(223, 216).to_uint();
    ValidateStart(start);
    return start;
}

sc_bv<256> SerializeIsaV1CollectiveByteData(
    const IsaV1CollectiveByteData &data) {
    ValidateData(data);
    sc_bv<256> wire = 0;
    wire.range(15, 0) = ISA_V1_COLL_BYTE_DATA_MAGIC;
    wire.range(19, 16) = ISA_V1_COLL_BYTE_VERSION;
    wire.range(23, 20) = static_cast<uint8_t>(data.kind);
    wire.range(39, 24) = data.lock.tree_id;
    wire.range(55, 40) = data.lock.session_id;
    wire.range(87, 56) = data.lock.epoch;
    wire.range(103, 88) = data.sequence;
    wire.range(107, 104) = data.length_bytes - 1;
    wire[108] = data.tail;
    wire.range(255, 128) = data.payload;
    return wire;
}

IsaV1CollectiveByteData DeserializeIsaV1CollectiveByteData(
    const sc_bv<256> &wire) {
    if (wire.range(15, 0).to_uint() != ISA_V1_COLL_BYTE_DATA_MAGIC)
        throw std::invalid_argument(
            "invalid ISA-v1 collective byte DATA magic");
    if (wire.range(19, 16).to_uint() != ISA_V1_COLL_BYTE_VERSION)
        throw std::invalid_argument(
            "unsupported ISA-v1 collective byte DATA version");
    if (wire.range(127, 109).or_reduce())
        throw std::invalid_argument(
            "ISA-v1 collective byte DATA reserved bits are non-zero");
    IsaV1CollectiveByteData data;
    data.payload = wire.range(255, 128);
    data.kind = static_cast<IsaV1CollectiveByteKind>(
        wire.range(23, 20).to_uint());
    data.lock.tree_id = wire.range(39, 24).to_uint();
    data.lock.session_id = wire.range(55, 40).to_uint();
    data.lock.epoch = wire.range(87, 56).to_uint64();
    data.sequence = wire.range(103, 88).to_uint();
    data.length_bytes = static_cast<uint8_t>(
        wire.range(107, 104).to_uint() + 1);
    data.tail = wire[108].to_bool();
    ValidateData(data);
    return data;
}

IsaV1CollectiveByteData InspectIsaV1CollectiveByteDataWire(
    const sc_bv<256> &wire) {
    return DeserializeIsaV1CollectiveByteData(wire);
}

IsaV1CollectiveByteBuiltStream BuildIsaV1CollectiveByteStream(
    const IsaV1CollectiveByteBuildSpec &spec,
    const std::vector<uint8_t> &bytes) {
    if (bytes.empty())
        throw std::invalid_argument(
            "ISA-v1 collective byte stream must contain bytes");
    if (bytes.size() > kDteEndpointP2pMaxBytes)
        throw std::length_error(
            "ISA-v1 collective byte stream exceeds endpoint byte limit");
    if (bytes.size() > std::numeric_limits<uint32_t>::max())
        throw std::length_error(
            "ISA-v1 collective byte stream exceeds START byte field");

    IsaV1CollectiveByteStart start;
    start.kind = spec.kind;
    start.tree_id = spec.tree_id;
    start.session_id = spec.session_id;
    start.collective = spec.collective;
    start.total_bytes = static_cast<uint32_t>(bytes.size());
    start.checksum = P2pPayloadChecksum(bytes);

    IsaV1CollectiveByteBuiltStream built;
    built.start = SerializeIsaV1CollectiveByteStart(start);
    const uint32_t fragments = FragmentCount(start.total_bytes);
    built.data.reserve(fragments);
    for (uint32_t index = 0; index < fragments; ++index) {
        const size_t begin = static_cast<size_t>(index) *
                             P2P_PAYLOAD_FRAGMENT_BYTES;
        const size_t length = std::min(P2P_PAYLOAD_FRAGMENT_BYTES,
                                       bytes.size() - begin);
        IsaV1CollectiveByteData data;
        data.kind = spec.kind;
        data.lock = {spec.tree_id, spec.session_id, spec.collective.epoch};
        data.sequence = static_cast<uint16_t>(index + 1);
        data.length_bytes = static_cast<uint8_t>(length);
        data.tail = index + 1 == fragments;
        for (size_t byte = 0; byte < length; ++byte)
            data.payload.range(byte * 8 + 7, byte * 8) = bytes[begin + byte];
        built.data.push_back(SerializeIsaV1CollectiveByteData(data));
    }
    return built;
}

IsaV1CollectiveByteReassembler::IsaV1CollectiveByteReassembler(
    size_t max_buffered_bytes, size_t max_inflight_streams)
    : max_buffered_bytes_(max_buffered_bytes),
      max_inflight_streams_(max_inflight_streams) {
    if (max_buffered_bytes == 0 || max_inflight_streams == 0)
        throw std::invalid_argument(
            "ISA-v1 collective byte reassembler limits must be positive");
}

void IsaV1CollectiveByteReassembler::Begin(
    const sc_bv<256> &start_wire) {
    const IsaV1CollectiveByteStart start =
        DeserializeIsaV1CollectiveByteStart(start_wire);
    const IsaV1CollectiveByteLock lock = LockFor(start);
    if (states_.count(lock) != 0)
        throw std::invalid_argument(
            "ISA-v1 collective byte START reuses active DATA lock");
    if (key_owners_.count(start.collective) != 0)
        throw std::invalid_argument(
            "ISA-v1 collective byte START collides with active full key");
    if (states_.size() >= max_inflight_streams_)
        throw std::length_error(
            "ISA-v1 collective byte reassembler stream capacity exhausted");
    if (start.total_bytes > max_buffered_bytes_ - reserved_bytes_)
        throw std::length_error(
            "ISA-v1 collective byte reassembler byte capacity exhausted");

    State state;
    state.start = start;
    state.fragment_count = FragmentCount(start.total_bytes);
    state.bytes.reserve(start.total_bytes);
    const auto inserted = states_.emplace(lock, std::move(state));
    if (!inserted.second)
        throw std::logic_error(
            "ISA-v1 collective byte state insertion unexpectedly failed");
    try {
        key_owners_.emplace(start.collective, lock);
    } catch (...) {
        states_.erase(lock);
        throw;
    }
    reserved_bytes_ += start.total_bytes;
}

std::optional<IsaV1CollectiveByteCommit>
IsaV1CollectiveByteReassembler::Accept(
    const sc_bv<256> &data_wire) {
    const IsaV1CollectiveByteData data =
        DeserializeIsaV1CollectiveByteData(data_wire);
    auto found = states_.find(data.lock);
    if (found == states_.end())
        throw std::invalid_argument(
            "ISA-v1 collective byte DATA has no active START");
    State &state = found->second;
    if (data.kind != state.start.kind)
        throw std::invalid_argument(
            "ISA-v1 collective byte DATA kind mismatches START");
    if (data.sequence != state.next_sequence)
        throw std::invalid_argument(
            "ISA-v1 collective byte DATA sequence is duplicate or out of order");

    const bool expected_tail =
        state.next_sequence == state.fragment_count;
    const uint32_t remaining =
        state.start.total_bytes - static_cast<uint32_t>(state.bytes.size());
    const uint8_t expected_length = static_cast<uint8_t>(
        std::min<uint32_t>(P2P_PAYLOAD_FRAGMENT_BYTES, remaining));
    if (data.tail != expected_tail)
        throw std::invalid_argument(
            "ISA-v1 collective byte DATA tail flag mismatches START");
    if (data.length_bytes != expected_length)
        throw std::invalid_argument(
            "ISA-v1 collective byte DATA length mismatches START");

    AppendPayload(state.bytes, data);
    ++state.next_sequence;
    if (!expected_tail) return std::nullopt;

    const uint32_t reservation = state.start.total_bytes;
    const CollectiveKey key = state.start.collective;
    if (P2pPayloadChecksum(state.bytes) != state.start.checksum) {
        states_.erase(found);
        key_owners_.erase(key);
        reserved_bytes_ -= reservation;
        throw std::invalid_argument(
            "ISA-v1 collective byte stream checksum mismatch");
    }

    IsaV1CollectiveByteCommit commit;
    commit.identity = {state.start.kind, state.start.tree_id,
                       state.start.session_id, state.start.collective};
    commit.bytes = std::move(state.bytes);
    states_.erase(found);
    key_owners_.erase(key);
    reserved_bytes_ -= reservation;
    return commit;
}

bool IsaV1CollectiveByteReassembler::Abort(
    const IsaV1CollectiveByteLock &lock) noexcept {
    auto found = states_.find(lock);
    if (found == states_.end()) return false;
    reserved_bytes_ -= found->second.start.total_bytes;
    key_owners_.erase(found->second.start.collective);
    states_.erase(found);
    return true;
}

bool IsaV1CollectiveByteReassembler::HasActive(
    const IsaV1CollectiveByteLock &lock) const noexcept {
    return states_.count(lock) != 0;
}
