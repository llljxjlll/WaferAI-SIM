#include "prims/collective_phase_barrier_v1_prim.h"

#include "dte/coll_runtime.h"
#include "utils/prim_utils.h"

#include <limits>
#include <stdexcept>
#include <string>

REGISTER_PRIM(Collective_phase_barrier_v1_prim,
              PrimId::COLLECTIVE_PHASE_BARRIER_V1);

namespace {

using Wire = vector<sc_bv<128>>;

void Require(bool condition, const std::string &message) {
    if (!condition) throw std::invalid_argument(message);
}

void RequireStrictTransport() {
    if (prim_wire::LegacyCompatibilityEnabled())
        throw std::invalid_argument(
            "Collective_phase_barrier_v1_prim is strict-only and rejects legacy transport");
}

uint8_t ExpectedId() {
    const int registered = PrimFactory::getInstance().getPrimId(
        "Collective_phase_barrier_v1_prim");
    if (registered != static_cast<int>(
                          PrimIdValue(
                              PrimId::COLLECTIVE_PHASE_BARRIER_V1)))
        throw std::logic_error(
            "Collective_phase_barrier_v1_prim factory ID does not match PrimId 59");
    return static_cast<uint8_t>(registered);
}

void SetHeader(sc_bv<128> &segment, uint8_t id, uint8_t ordinal) {
    segment.range(7, 0) = sc_bv<8>(id);
    segment.range(15, 8) = sc_bv<8>(ordinal);
}

void ValidateHeaders(const Wire &wire) {
    Require(wire.size() == kCollectivePhaseBarrierV1WireSegments,
            "Collective_phase_barrier_v1_prim wire segment count mismatch");
    const uint8_t id = ExpectedId();
    for (size_t index = 0; index < wire.size(); ++index) {
        Require(wire[index].range(7, 0).to_uint64() == id,
                "Collective_phase_barrier_v1_prim wire has inconsistent segment IDs");
        Require(wire[index].range(15, 8).to_uint64() == index,
                "Collective_phase_barrier_v1_prim wire has inconsistent ordinals");
    }
}

} // namespace

void Collective_phase_barrier_v1_prim::Validate() const {
    Require(key.group_id != 0,
            "Collective_phase_barrier_v1_prim group_id must be non-zero");
    Require(key.collective_id != std::numeric_limits<uint32_t>::max(),
            "Collective_phase_barrier_v1_prim cannot use the GROUP_SYNC collective_id namespace");
    Require(group_size != 0,
            "Collective_phase_barrier_v1_prim group_size must be positive");
    Require(rank < group_size,
            "Collective_phase_barrier_v1_prim rank is outside the group");
}

Wire Collective_phase_barrier_v1_prim::serialize() {
    RequireStrictTransport();
    Validate();
    Wire wire(kCollectivePhaseBarrierV1WireSegments);
    const uint8_t id = ExpectedId();
    for (uint8_t index = 0; index < wire.size(); ++index) {
        wire[index] = 0;
        SetHeader(wire[index], id, index);
    }
    wire[0].range(23, 16) =
        sc_bv<8>(kCollectivePhaseBarrierV1WireVersion);
    wire[0].range(31, 24) =
        sc_bv<8>(kCollectivePhaseBarrierV1WireSegments);
    wire[0].range(47, 32) = sc_bv<16>(phase_id);
    wire[0].range(63, 48) = sc_bv<16>(rank);
    wire[0].range(79, 64) = sc_bv<16>(group_size);
    wire[0].range(95, 80) = sc_bv<16>(release_tree_id);

    wire[1].range(47, 16) = sc_bv<32>(key.group_id);
    wire[1].range(79, 48) = sc_bv<32>(key.collective_id);
    wire[1].range(111, 80) = sc_bv<32>(key.epoch);
    return wire;
}

void Collective_phase_barrier_v1_prim::deserialize(Wire wire) {
    RequireStrictTransport();
    ValidateHeaders(wire);
    Require(wire[0].range(23, 16).to_uint64() ==
                kCollectivePhaseBarrierV1WireVersion,
            "Collective_phase_barrier_v1_prim wire version is unsupported");
    Require(wire[0].range(31, 24).to_uint64() ==
                kCollectivePhaseBarrierV1WireSegments,
            "Collective_phase_barrier_v1_prim wire count field is inconsistent");
    Require(!wire[0].range(127, 96).or_reduce() &&
                !wire[1].range(127, 112).or_reduce(),
            "Collective_phase_barrier_v1_prim reserved bits are non-zero");

    Collective_phase_barrier_v1_prim decoded;
    decoded.phase_id = static_cast<uint16_t>(
        wire[0].range(47, 32).to_uint64());
    decoded.rank = static_cast<uint16_t>(
        wire[0].range(63, 48).to_uint64());
    decoded.group_size = static_cast<uint16_t>(
        wire[0].range(79, 64).to_uint64());
    decoded.release_tree_id = static_cast<uint16_t>(
        wire[0].range(95, 80).to_uint64());
    decoded.key.group_id = static_cast<uint32_t>(
        wire[1].range(47, 16).to_uint64());
    decoded.key.collective_id = static_cast<uint32_t>(
        wire[1].range(79, 48).to_uint64());
    decoded.key.epoch = static_cast<uint32_t>(
        wire[1].range(111, 80).to_uint64());
    decoded.Validate();

    key = decoded.key;
    phase_id = decoded.phase_id;
    rank = decoded.rank;
    group_size = decoded.group_size;
    release_tree_id = decoded.release_tree_id;
}

int Collective_phase_barrier_v1_prim::taskCoreDefault(TaskCoreContext &) {
    Validate();
    WaitCollectiveBarrier(key, phase_id, rank, group_size,
                          release_tree_id);
    return 0;
}

void Collective_phase_barrier_v1_prim::printSelf() {}
