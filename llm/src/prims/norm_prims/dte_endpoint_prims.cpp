#include "prims/dte_endpoint_prims.h"

#include "utils/prim_utils.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

REGISTER_PRIM(Dte_send_endpoint_prim, PrimId::DTE_SEND_ENDPOINT);
REGISTER_PRIM(Dte_recv_endpoint_prim, PrimId::DTE_RECV_ENDPOINT);

namespace {
using Wire = vector<sc_bv<128>>;

constexpr uint32_t kReservedCollectiveId =
    std::numeric_limits<uint32_t>::max();

void Require(bool condition, const std::string &message) {
    if (!condition) throw std::invalid_argument(message);
}

void RequireStrictTransport(const std::string &name) {
    if (prim_wire::LegacyCompatibilityEnabled())
        throw std::invalid_argument(
            name + " is strict-only and rejects legacy transport");
}

uint8_t ExpectedId(const std::string &name, PrimId expected) {
    const int registered = PrimFactory::getInstance().getPrimId(name);
    if (registered != static_cast<int>(PrimIdValue(expected)))
        throw std::logic_error(name +
                               " factory ID does not match its PrimId");
    return static_cast<uint8_t>(registered);
}

void ValidateAddress(const DteEndpointSramAddress &address,
                     const std::string &name) {
    const uint8_t raw_kind = static_cast<uint8_t>(address.kind);
    Require(raw_kind == static_cast<uint8_t>(DteEndpointAddressKind::ABSOLUTE) ||
                raw_kind == static_cast<uint8_t>(DteEndpointAddressKind::REGION),
            name + " SRAM address kind is invalid");
    if (address.kind == DteEndpointAddressKind::ABSOLUTE) {
        Require(address.region.empty() && address.region_offset_bytes == 0,
                name + " absolute address carries region metadata");
        return;
    }
    Require(address.absolute_address_bytes == 0,
            name + " region address carries an absolute address");
    Require(!address.region.empty(), name + " region name is empty");
    Require(address.region.size() <= kDteEndpointPrimRegionMaxBytes,
            name + " region name exceeds the 64-byte wire limit");
    Require(address.region.find('\0') == std::string::npos,
            name + " region name contains NUL");
}

void ValidateCommon(const Dte_endpoint_prim_base &prim) {
    const auto completion = static_cast<uint8_t>(prim.completion);
    const auto datatype = static_cast<uint8_t>(prim.datatype);
    const auto reduce = static_cast<uint8_t>(prim.reduce_op);
    Require(completion <= static_cast<uint8_t>(DteEndpointCompletion::SYNC),
            prim.name + " completion enum is invalid");
    Require(datatype <= static_cast<uint8_t>(DteEndpointDataType::INT64),
            prim.name + " datatype enum is invalid");
    Require(reduce <= static_cast<uint8_t>(DteEndpointReduceOp::MAX),
            prim.name + " reduce_op enum is invalid");
    Require(prim.fsm_id != 0, prim.name + " fsm_id must be non-zero");
    if (prim.completion == DteEndpointCompletion::ASYNC)
        Require(prim.token != 0,
                prim.name + " asynchronous token must be non-zero");
    else
        Require(prim.token == 0,
                prim.name + " synchronous token must be zero");
    Require(prim.length_bytes != 0,
            prim.name + " length_bytes must be non-zero");
}

void ValidateCollectiveKey(const Dte_endpoint_prim_base &prim) {
    Require(prim.group_id != 0, prim.name + " group_id must be non-zero");
    Require(prim.collective_id != kReservedCollectiveId,
            prim.name +
                " collective_id 0xFFFFFFFF is reserved for GROUP_SYNC");
}

uint8_t SegmentCount(const DteEndpointSramAddress &address) {
    const size_t text_segments =
        (address.region.size() + kDteEndpointPrimRegionBytesPerSegment - 1) /
        kDteEndpointPrimRegionBytesPerSegment;
    return static_cast<uint8_t>(kDteEndpointPrimBaseSegments +
                                text_segments);
}

void SetSegmentHeader(sc_bv<128> &segment, uint8_t id, uint8_t index) {
    segment.range(7, 0) = sc_bv<8>(id);
    segment.range(15, 8) = sc_bv<8>(index);
}

Wire SerializeCommon(const Dte_endpoint_prim_base &prim,
                     const DteEndpointSramAddress &address,
                     uint8_t raw_mode, uint8_t raw_source_space,
                     PrimId expected) {
    RequireStrictTransport(prim.name);
    const uint8_t id = ExpectedId(prim.name, expected);
    const uint8_t segment_count = SegmentCount(address);
    Wire wire(segment_count);
    for (uint8_t index = 0; index < segment_count; ++index) {
        wire[index] = 0;
        SetSegmentHeader(wire[index], id, index);
    }

    wire[0].range(23, 16) = sc_bv<8>(kDteEndpointPrimWireVersion);
    wire[0].range(31, 24) = sc_bv<8>(segment_count);
    wire[0].range(33, 32) = sc_bv<2>(raw_mode);
    wire[0].range(34, 34) =
        sc_bv<1>(static_cast<uint8_t>(prim.completion));
    wire[0].range(36, 35) =
        sc_bv<2>(static_cast<uint8_t>(prim.datatype));
    wire[0].range(38, 37) =
        sc_bv<2>(static_cast<uint8_t>(prim.reduce_op));
    wire[0].range(40, 39) =
        sc_bv<2>(static_cast<uint8_t>(address.kind));
    wire[0].range(42, 41) = sc_bv<2>(raw_source_space);

    wire[1].range(47, 16) = sc_bv<32>(prim.fsm_id);
    wire[1].range(79, 48) = sc_bv<32>(prim.token);
    wire[1].range(95, 80) = sc_bv<16>(prim.peer_core);
    wire[1].range(111, 96) = sc_bv<16>(prim.expected_sources);
    wire[1].range(127, 112) = sc_bv<16>(prim.tree_id);

    wire[2].range(79, 16) = sc_bv<64>(prim.length_bytes);
    const uint64_t address_value =
        address.kind == DteEndpointAddressKind::ABSOLUTE
            ? address.absolute_address_bytes
            : address.region_offset_bytes;
    wire[3].range(79, 16) = sc_bv<64>(address_value);

    wire[4].range(47, 16) = sc_bv<32>(prim.group_id);
    wire[4].range(79, 48) = sc_bv<32>(prim.collective_id);
    wire[4].range(111, 80) = sc_bv<32>(prim.epoch);
    wire[4].range(119, 112) = sc_bv<8>(address.region.size());

    for (size_t index = 0; index < address.region.size(); ++index) {
        const size_t segment =
            kDteEndpointPrimBaseSegments +
            index / kDteEndpointPrimRegionBytesPerSegment;
        const int low = static_cast<int>(
            16 + 8 * (index % kDteEndpointPrimRegionBytesPerSegment));
        wire[segment].range(low + 7, low) = sc_bv<8>(
            static_cast<uint8_t>(address.region[index]));
    }
    return wire;
}

struct DecodedCommon {
    uint8_t mode = 0;
    DteEndpointSourceSpace source_space = DteEndpointSourceSpace::SRAM;
    DteEndpointCompletion completion = DteEndpointCompletion::ASYNC;
    DteEndpointDataType datatype = DteEndpointDataType::UINT8;
    DteEndpointReduceOp reduce_op = DteEndpointReduceOp::NONE;
    uint32_t fsm_id = 0;
    uint32_t token = 0;
    uint64_t length_bytes = 0;
    DteEndpointSramAddress address;
    uint16_t peer_core = 0;
    uint16_t expected_sources = 0;
    uint16_t tree_id = 0;
    uint32_t group_id = 0;
    uint32_t collective_id = 0;
    uint32_t epoch = 0;
};

DecodedCommon DeserializeCommon(const Wire &wire, const std::string &name,
                                PrimId expected) {
    RequireStrictTransport(name);
    Require(wire.size() >= kDteEndpointPrimBaseSegments,
            name + " Prim wire is truncated");
    Require(wire.size() <= kDteEndpointPrimMaxSegments,
            name + " Prim wire has too many segments");
    const uint8_t id = ExpectedId(name, expected);
    for (size_t index = 0; index < wire.size(); ++index) {
        Require(wire[index].range(7, 0).to_uint64() == id,
                name + " Prim wire contains an inconsistent segment ID");
        Require(wire[index].range(15, 8).to_uint64() == index,
                name + " Prim wire contains an inconsistent segment index");
    }
    Require(wire[0].range(23, 16).to_uint64() ==
                kDteEndpointPrimWireVersion,
            name + " Prim wire version is unsupported");
    Require(wire[0].range(31, 24).to_uint64() == wire.size(),
            name + " Prim wire segment count is inconsistent");
    if (expected == PrimId::DTE_SEND_ENDPOINT)
        Require(!wire[0].range(127, 43).or_reduce(),
                name + " Prim wire metadata reserved bits are non-zero");
    else
        Require(!wire[0].range(127, 41).or_reduce(),
                name + " Prim wire metadata reserved bits are non-zero");
    Require(!wire[2].range(127, 80).or_reduce(),
            name + " Prim wire length reserved bits are non-zero");
    Require(!wire[3].range(127, 80).or_reduce(),
            name + " Prim wire address reserved bits are non-zero");
    Require(!wire[4].range(127, 120).or_reduce(),
            name + " Prim wire routing reserved bits are non-zero");

    DecodedCommon decoded;
    decoded.mode = static_cast<uint8_t>(
        wire[0].range(33, 32).to_uint64());
    if (expected == PrimId::DTE_SEND_ENDPOINT)
        decoded.source_space = static_cast<DteEndpointSourceSpace>(
            wire[0].range(42, 41).to_uint64());
    decoded.completion = static_cast<DteEndpointCompletion>(
        wire[0].range(34, 34).to_uint64());
    decoded.datatype = static_cast<DteEndpointDataType>(
        wire[0].range(36, 35).to_uint64());
    decoded.reduce_op = static_cast<DteEndpointReduceOp>(
        wire[0].range(38, 37).to_uint64());
    decoded.address.kind = static_cast<DteEndpointAddressKind>(
        wire[0].range(40, 39).to_uint64());
    decoded.fsm_id = static_cast<uint32_t>(
        wire[1].range(47, 16).to_uint64());
    decoded.token = static_cast<uint32_t>(
        wire[1].range(79, 48).to_uint64());
    decoded.peer_core = static_cast<uint16_t>(
        wire[1].range(95, 80).to_uint64());
    decoded.expected_sources = static_cast<uint16_t>(
        wire[1].range(111, 96).to_uint64());
    decoded.tree_id = static_cast<uint16_t>(
        wire[1].range(127, 112).to_uint64());
    decoded.length_bytes = wire[2].range(79, 16).to_uint64();
    const uint64_t address_value = wire[3].range(79, 16).to_uint64();
    decoded.group_id = static_cast<uint32_t>(
        wire[4].range(47, 16).to_uint64());
    decoded.collective_id = static_cast<uint32_t>(
        wire[4].range(79, 48).to_uint64());
    decoded.epoch = static_cast<uint32_t>(
        wire[4].range(111, 80).to_uint64());
    const size_t region_size = wire[4].range(119, 112).to_uint64();

    if (decoded.address.kind == DteEndpointAddressKind::ABSOLUTE) {
        decoded.address.absolute_address_bytes = address_value;
        Require(region_size == 0 &&
                    wire.size() == kDteEndpointPrimBaseSegments,
                name + " absolute address carries region segments");
    } else if (decoded.address.kind == DteEndpointAddressKind::REGION) {
        decoded.address.region_offset_bytes = address_value;
        Require(region_size != 0 &&
                    region_size <= kDteEndpointPrimRegionMaxBytes,
                name + " region length is invalid");
        const size_t expected_segments =
            kDteEndpointPrimBaseSegments +
            (region_size + kDteEndpointPrimRegionBytesPerSegment - 1) /
                kDteEndpointPrimRegionBytesPerSegment;
        Require(wire.size() == expected_segments,
                name + " region segment count is inconsistent");
        decoded.address.region.reserve(region_size);
        for (size_t index = 0; index < region_size; ++index) {
            const size_t segment =
                kDteEndpointPrimBaseSegments +
                index / kDteEndpointPrimRegionBytesPerSegment;
            const int low = static_cast<int>(
                16 + 8 * (index % kDteEndpointPrimRegionBytesPerSegment));
            decoded.address.region.push_back(static_cast<char>(
                wire[segment].range(low + 7, low).to_uint64()));
        }
        const size_t used =
            region_size % kDteEndpointPrimRegionBytesPerSegment;
        if (used != 0) {
            const int first_padding = static_cast<int>(16 + 8 * used);
            Require(!wire.back().range(127, first_padding).or_reduce(),
                    name + " region padding is non-zero");
        }
    } else {
        throw std::invalid_argument(name +
                                    " SRAM address kind is invalid");
    }
    return decoded;
}

template <class Prim>
void CommitCommon(Prim &prim, DecodedCommon decoded) {
    prim.completion = decoded.completion;
    prim.datatype = decoded.datatype;
    prim.reduce_op = decoded.reduce_op;
    prim.fsm_id = decoded.fsm_id;
    prim.token = decoded.token;
    prim.length_bytes = decoded.length_bytes;
    prim.peer_core = decoded.peer_core;
    prim.expected_sources = decoded.expected_sources;
    prim.tree_id = decoded.tree_id;
    prim.group_id = decoded.group_id;
    prim.collective_id = decoded.collective_id;
    prim.epoch = decoded.epoch;
}
} // namespace

void Dte_send_endpoint_prim::Validate() const {
    ValidateCommon(*this);
    ValidateAddress(source, name);
    const auto raw_source_space = static_cast<uint8_t>(source_space);
    Require(raw_source_space <=
                static_cast<uint8_t>(DteEndpointSourceSpace::HBM),
            name + " source_space enum is invalid");
    if (source_space == DteEndpointSourceSpace::HBM)
        Require(source.kind == DteEndpointAddressKind::ABSOLUTE,
                name + " HBM source must use an absolute address");
    const auto raw_mode = static_cast<uint8_t>(mode);
    Require(raw_mode <= static_cast<uint8_t>(DteEndpointSendMode::BROADCAST),
            name + " mode enum is invalid");
    Require(reduce_op == DteEndpointReduceOp::NONE,
            name + " forbids reduce_op");
    Require(datatype == DteEndpointDataType::UINT8,
            name + " datatype must be UINT8");
    Require(expected_sources == 0,
            name + " expected_sources is receive-only");
    switch (mode) {
    case DteEndpointSendMode::P2P:
        Require(length_bytes <= kDteEndpointP2pMaxBytes,
                name + " P2P length_bytes exceeds transport maximum");
        Require(tree_id == 0 && group_id == 0 && collective_id == 0 &&
                    epoch == 0,
                name + " P2P mode carries collective metadata");
        return;
    case DteEndpointSendMode::SCATTER:
        Require(peer_core == 0 && tree_id == 0,
                name + " SCATTER mode carries peer/tree metadata");
        ValidateCollectiveKey(*this);
        return;
    case DteEndpointSendMode::BROADCAST:
        Require(peer_core == 0 && tree_id != 0,
                name + " BROADCAST mode requires only a non-zero tree_id");
        ValidateCollectiveKey(*this);
        return;
    }
}

int Dte_send_endpoint_prim::taskCoreDefault(TaskCoreContext &) {
    throw std::logic_error(
        "Dte_send_endpoint_prim requires Worker special dispatch");
}

vector<sc_bv<128>> Dte_send_endpoint_prim::serialize() {
    Validate();
    return SerializeCommon(*this, source, static_cast<uint8_t>(mode),
                           static_cast<uint8_t>(source_space),
                           PrimId::DTE_SEND_ENDPOINT);
}

void Dte_send_endpoint_prim::deserialize(vector<sc_bv<128>> wire) {
    DecodedCommon decoded = DeserializeCommon(
        wire, name, PrimId::DTE_SEND_ENDPOINT);
    Dte_send_endpoint_prim candidate;
    candidate.mode = static_cast<DteEndpointSendMode>(decoded.mode);
    candidate.source_space = decoded.source_space;
    candidate.source = std::move(decoded.address);
    CommitCommon(candidate, std::move(decoded));
    candidate.Validate();
    mode = candidate.mode;
    source_space = candidate.source_space;
    DecodedCommon committed;
    committed.completion = candidate.completion;
    committed.datatype = candidate.datatype;
    committed.reduce_op = candidate.reduce_op;
    committed.fsm_id = candidate.fsm_id;
    committed.token = candidate.token;
    committed.length_bytes = candidate.length_bytes;
    committed.peer_core = candidate.peer_core;
    committed.expected_sources = candidate.expected_sources;
    committed.tree_id = candidate.tree_id;
    committed.group_id = candidate.group_id;
    committed.collective_id = candidate.collective_id;
    committed.epoch = candidate.epoch;
    CommitCommon(*this, std::move(committed));
    source = std::move(candidate.source);
}

void Dte_send_endpoint_prim::printSelf() {}

void Dte_recv_endpoint_prim::Validate() const {
    ValidateCommon(*this);
    ValidateAddress(destination, name);
    const auto raw_mode = static_cast<uint8_t>(mode);
    Require(raw_mode <= static_cast<uint8_t>(DteEndpointRecvMode::REDUCE),
            name + " mode enum is invalid");
    Require(tree_id == 0, name + " must not carry a tree_id");
    switch (mode) {
    case DteEndpointRecvMode::P2P:
        Require(length_bytes <= kDteEndpointP2pMaxBytes,
                name + " P2P length_bytes exceeds transport maximum");
        Require(reduce_op == DteEndpointReduceOp::NONE,
                name + " P2P mode forbids reduce_op");
        Require(datatype == DteEndpointDataType::UINT8,
                name + " P2P datatype must be UINT8");
        Require(expected_sources == 0 && group_id == 0 &&
                    collective_id == 0 && epoch == 0,
                name + " P2P mode carries collective metadata");
        return;
    case DteEndpointRecvMode::GATHER:
        Require(reduce_op == DteEndpointReduceOp::NONE,
                name + " GATHER mode forbids reduce_op");
        Require(datatype == DteEndpointDataType::UINT8,
                name + " GATHER datatype must be UINT8");
        Require(peer_core == 0 && expected_sources != 0,
                name +
                    " GATHER mode requires expected_sources and no peer");
        ValidateCollectiveKey(*this);
        return;
    case DteEndpointRecvMode::REDUCE:
        Require(reduce_op == DteEndpointReduceOp::SUM ||
                    reduce_op == DteEndpointReduceOp::MAX,
                name + " REDUCE mode requires SUM or MAX");
        Require(peer_core == 0 && expected_sources != 0,
                name +
                    " REDUCE mode requires expected_sources and no peer");
        ValidateCollectiveKey(*this);
        return;
    }
}

int Dte_recv_endpoint_prim::taskCoreDefault(TaskCoreContext &) {
    throw std::logic_error(
        "Dte_recv_endpoint_prim requires Worker special dispatch");
}

vector<sc_bv<128>> Dte_recv_endpoint_prim::serialize() {
    Validate();
    return SerializeCommon(*this, destination, static_cast<uint8_t>(mode),
                           static_cast<uint8_t>(DteEndpointSourceSpace::SRAM),
                           PrimId::DTE_RECV_ENDPOINT);
}

void Dte_recv_endpoint_prim::deserialize(vector<sc_bv<128>> wire) {
    DecodedCommon decoded = DeserializeCommon(
        wire, name, PrimId::DTE_RECV_ENDPOINT);
    Dte_recv_endpoint_prim candidate;
    candidate.mode = static_cast<DteEndpointRecvMode>(decoded.mode);
    candidate.destination = std::move(decoded.address);
    CommitCommon(candidate, std::move(decoded));
    candidate.Validate();
    mode = candidate.mode;
    DecodedCommon committed;
    committed.completion = candidate.completion;
    committed.datatype = candidate.datatype;
    committed.reduce_op = candidate.reduce_op;
    committed.fsm_id = candidate.fsm_id;
    committed.token = candidate.token;
    committed.length_bytes = candidate.length_bytes;
    committed.peer_core = candidate.peer_core;
    committed.expected_sources = candidate.expected_sources;
    committed.tree_id = candidate.tree_id;
    committed.group_id = candidate.group_id;
    committed.collective_id = candidate.collective_id;
    committed.epoch = candidate.epoch;
    CommitCommon(*this, std::move(committed));
    destination = std::move(candidate.destination);
}

void Dte_recv_endpoint_prim::printSelf() {}
