#include "prims/collective_launch_v1_prim.h"

#include "utils/prim_utils.h"

#include <limits>
#include <stdexcept>
#include <string>

REGISTER_PRIM(Collective_launch_v1_prim, PrimId::COLLECTIVE_LAUNCH_V1);

namespace {

using Wire = vector<sc_bv<128>>;

void Require(bool condition, const std::string &message) {
    if (!condition) throw std::invalid_argument(message);
}

void RequireStrictTransport() {
    if (prim_wire::LegacyCompatibilityEnabled())
        throw std::invalid_argument(
            "Collective_launch_v1_prim is strict-only and rejects legacy transport");
}

uint8_t ExpectedId() {
    const int registered = PrimFactory::getInstance().getPrimId(
        "Collective_launch_v1_prim");
    if (registered !=
        static_cast<int>(PrimIdValue(PrimId::COLLECTIVE_LAUNCH_V1)))
        throw std::logic_error(
            "Collective_launch_v1_prim factory ID does not match PrimId 60");
    return static_cast<uint8_t>(registered);
}

void SetHeader(sc_bv<128> &segment, uint8_t id, uint8_t ordinal) {
    segment.range(7, 0) = sc_bv<8>(id);
    segment.range(15, 8) = sc_bv<8>(ordinal);
}

void ValidateHeaders(const Wire &wire) {
    Require(wire.size() == kCollectiveLaunchV1PrimWireSegments,
            "Collective_launch_v1_prim wire segment count mismatch");
    const uint8_t id = ExpectedId();
    for (size_t index = 0; index < wire.size(); ++index) {
        Require(wire[index].range(7, 0).to_uint64() == id,
                "Collective_launch_v1_prim wire has inconsistent segment IDs");
        Require(wire[index].range(15, 8).to_uint64() == index,
                "Collective_launch_v1_prim wire has inconsistent ordinals");
    }
}

} // namespace

void Collective_launch_v1_prim::Validate() const {
    const uint8_t raw_role = static_cast<uint8_t>(role);
    Require(raw_role <= static_cast<uint8_t>(
                            CollectiveLaunchV1Role::DECLARE_REDUCE_COMPUTE),
            "Collective_launch_v1_prim role enum is invalid");
    Require(image_generation != 0,
            "Collective_launch_v1_prim image_generation must be non-zero");
    Require(key.group_id != 0,
            "Collective_launch_v1_prim group_id must be non-zero");
    Require(key.collective_id != std::numeric_limits<uint32_t>::max(),
            "Collective_launch_v1_prim collective ID is reserved");
    if (role == CollectiveLaunchV1Role::DECLARE_REDUCE_COMPUTE) {
        Require(public_token == 0,
                "Collective_launch_v1_prim REDUCE declaration forbids a public token");
    } else {
        Require(public_token != 0,
                "Collective_launch_v1_prim SEND/RECEIVE requires a public token");
    }
}

Wire Collective_launch_v1_prim::serialize() {
    RequireStrictTransport();
    Validate();
    Wire wire(kCollectiveLaunchV1PrimWireSegments);
    const uint8_t id = ExpectedId();
    for (uint8_t index = 0; index < wire.size(); ++index) {
        wire[index] = 0;
        SetHeader(wire[index], id, index);
    }

    wire[0].range(23, 16) =
        sc_bv<8>(kCollectiveLaunchV1PrimWireVersion);
    wire[0].range(31, 24) =
        sc_bv<8>(kCollectiveLaunchV1PrimWireSegments);
    wire[0].range(33, 32) = sc_bv<2>(static_cast<uint8_t>(role));

    wire[1].range(79, 16) = sc_bv<64>(image_generation);
    wire[1].range(111, 80) = sc_bv<32>(plan_index);

    wire[2].range(47, 16) = sc_bv<32>(external_record_index);
    wire[2].range(63, 48) = sc_bv<16>(expected_core);
    wire[2].range(95, 64) = sc_bv<32>(public_token);

    wire[3].range(47, 16) = sc_bv<32>(key.group_id);
    wire[3].range(79, 48) = sc_bv<32>(key.collective_id);
    wire[3].range(111, 80) = sc_bv<32>(key.epoch);
    return wire;
}

void Collective_launch_v1_prim::deserialize(Wire wire) {
    RequireStrictTransport();
    ValidateHeaders(wire);
    Require(wire[0].range(23, 16).to_uint64() ==
                kCollectiveLaunchV1PrimWireVersion,
            "Collective_launch_v1_prim wire version is unsupported");
    Require(wire[0].range(31, 24).to_uint64() ==
                kCollectiveLaunchV1PrimWireSegments,
            "Collective_launch_v1_prim wire count field is inconsistent");
    Require(!wire[0].range(127, 34).or_reduce() &&
                !wire[1].range(127, 112).or_reduce() &&
                !wire[2].range(127, 96).or_reduce() &&
                !wire[3].range(127, 112).or_reduce(),
            "Collective_launch_v1_prim reserved bits are non-zero");

    Collective_launch_v1_prim decoded;
    decoded.role = static_cast<CollectiveLaunchV1Role>(
        wire[0].range(33, 32).to_uint64());
    decoded.image_generation = wire[1].range(79, 16).to_uint64();
    decoded.plan_index = static_cast<uint32_t>(
        wire[1].range(111, 80).to_uint64());
    decoded.external_record_index = static_cast<uint32_t>(
        wire[2].range(47, 16).to_uint64());
    decoded.expected_core = static_cast<uint16_t>(
        wire[2].range(63, 48).to_uint64());
    decoded.public_token = static_cast<uint32_t>(
        wire[2].range(95, 64).to_uint64());
    decoded.key.group_id = static_cast<uint32_t>(
        wire[3].range(47, 16).to_uint64());
    decoded.key.collective_id = static_cast<uint32_t>(
        wire[3].range(79, 48).to_uint64());
    decoded.key.epoch = static_cast<uint32_t>(
        wire[3].range(111, 80).to_uint64());
    decoded.Validate();

    role = decoded.role;
    image_generation = decoded.image_generation;
    plan_index = decoded.plan_index;
    external_record_index = decoded.external_record_index;
    expected_core = decoded.expected_core;
    key = decoded.key;
    public_token = decoded.public_token;
}

int Collective_launch_v1_prim::taskCoreDefault(TaskCoreContext &) {
    throw std::logic_error(
        "Collective_launch_v1_prim requires Worker special dispatch");
}

void Collective_launch_v1_prim::printSelf() {}
