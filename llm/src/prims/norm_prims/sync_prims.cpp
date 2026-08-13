#include "prims/sync_prims.h"

#include "utils/prim_utils.h"

#include <stdexcept>

REGISTER_PRIM(Group_sync_prim, PrimId::GROUP_SYNC);
REGISTER_PRIM(Event_control_prim, PrimId::EVENT_CONTROL);

namespace {
void RequireStrictTransport(const char *name) {
    if (prim_wire::LegacyCompatibilityEnabled())
        throw std::invalid_argument(
            std::string(name) +
            " is strict-only and rejects legacy transport");
}

uint8_t ExpectedId(const char *name, PrimId expected) {
    const int registered = PrimFactory::getInstance().getPrimId(name);
    if (registered != static_cast<int>(PrimIdValue(expected)))
        throw std::logic_error(
            std::string(name) + " factory ID does not match its PrimId");
    return static_cast<uint8_t>(registered);
}

void ValidateOneSegment(const vector<sc_bv<128>> &segments,
                        const char *name, PrimId expected) {
    if (segments.size() != 1)
        throw std::invalid_argument(
            std::string(name) + " Prim wire segment count mismatch");
    if (segments[0].range(7, 0).to_uint64() !=
        ExpectedId(name, expected))
        throw std::invalid_argument(
            std::string(name) + " Prim wire has the wrong ID");
}

void ValidateEvent(const Event_control_prim &prim) {
    const auto raw_op = static_cast<uint8_t>(prim.op);
    if (raw_op > static_cast<uint8_t>(EventControlOp::WAIT))
        throw std::invalid_argument("Event_control_prim operation is invalid");
    if (prim.op == EventControlOp::SET && prim.count != 1)
        throw std::invalid_argument(
            "Event_control_prim SET count must be one");
    if (prim.op == EventControlOp::WAIT && prim.count == 0)
        throw std::invalid_argument(
            "Event_control_prim WAIT count must be non-zero");
}
} // namespace

int Group_sync_prim::taskCoreDefault(TaskCoreContext &) {
    throw std::logic_error(
        "Group_sync_prim must be dispatched by WorkerCoreExecutor");
}

vector<sc_bv<128>> Group_sync_prim::serialize() {
    RequireStrictTransport(name.c_str());
    if (group_id == 0)
        throw std::invalid_argument(
            "Group_sync_prim group_id must be non-zero");
    sc_bv<128> segment = 0;
    segment.range(7, 0) =
        sc_bv<8>(ExpectedId(name.c_str(), PrimId::GROUP_SYNC));
    segment.range(39, 8) = sc_bv<32>(group_id);
    segment.range(71, 40) = sc_bv<32>(sync_seq);
    return {segment};
}

void Group_sync_prim::deserialize(vector<sc_bv<128>> segments) {
    RequireStrictTransport(name.c_str());
    ValidateOneSegment(segments, name.c_str(), PrimId::GROUP_SYNC);
    if (segments[0].range(127, 72).or_reduce())
        throw std::invalid_argument(
            "Group_sync_prim Prim wire padding is non-zero");
    const uint32_t decoded_group = static_cast<uint32_t>(
        segments[0].range(39, 8).to_uint64());
    const uint32_t decoded_sequence = static_cast<uint32_t>(
        segments[0].range(71, 40).to_uint64());
    if (decoded_group == 0)
        throw std::invalid_argument(
            "Group_sync_prim group_id must be non-zero");
    group_id = decoded_group;
    sync_seq = decoded_sequence;
}

void Group_sync_prim::printSelf() {}

int Event_control_prim::taskCoreDefault(TaskCoreContext &) {
    throw std::logic_error(
        "Event_control_prim must be dispatched by WorkerCoreExecutor");
}

vector<sc_bv<128>> Event_control_prim::serialize() {
    RequireStrictTransport(name.c_str());
    ValidateEvent(*this);
    sc_bv<128> segment = 0;
    segment.range(7, 0) =
        sc_bv<8>(ExpectedId(name.c_str(), PrimId::EVENT_CONTROL));
    segment.range(8, 8) =
        sc_bv<1>(static_cast<uint8_t>(op));
    segment.range(24, 9) = sc_bv<16>(source_core);
    segment.range(40, 25) = sc_bv<16>(destination_core);
    segment.range(72, 41) = sc_bv<32>(tag);
    segment.range(104, 73) = sc_bv<32>(count);
    return {segment};
}

void Event_control_prim::deserialize(vector<sc_bv<128>> segments) {
    RequireStrictTransport(name.c_str());
    ValidateOneSegment(segments, name.c_str(), PrimId::EVENT_CONTROL);
    if (segments[0].range(127, 105).or_reduce())
        throw std::invalid_argument(
            "Event_control_prim Prim wire padding is non-zero");
    Event_control_prim decoded;
    decoded.op = static_cast<EventControlOp>(
        segments[0].range(8, 8).to_uint64());
    decoded.source_core = static_cast<uint16_t>(
        segments[0].range(24, 9).to_uint64());
    decoded.destination_core = static_cast<uint16_t>(
        segments[0].range(40, 25).to_uint64());
    decoded.tag = static_cast<uint32_t>(
        segments[0].range(72, 41).to_uint64());
    decoded.count = static_cast<uint32_t>(
        segments[0].range(104, 73).to_uint64());
    ValidateEvent(decoded);
    op = decoded.op;
    source_core = decoded.source_core;
    destination_core = decoded.destination_core;
    tag = decoded.tag;
    count = decoded.count;
}

void Event_control_prim::printSelf() {}
