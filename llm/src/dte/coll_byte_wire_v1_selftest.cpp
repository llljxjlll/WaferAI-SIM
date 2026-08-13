#include "dte/coll_byte_wire_v1_selftest.h"

#include "dte/coll_byte_wire_v1.h"
#include "dte/coll_innetwork_reduce.h"
#include "dte/coll_multicast.h"
#include "utils/msg_utils.h"

#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Suite {
    int checks = 0;
    int failures = 0;
    void Check(bool condition, const std::string &name) {
        ++checks;
        if (condition) return;
        ++failures;
        std::cerr << "[ISA V1 COLL BYTE WIRE] FAIL: " << name << '\n';
    }
    template <class E = std::exception, class F>
    void Throws(F &&fn, const std::string &name) {
        bool threw = false;
        try { fn(); } catch (const E &) { threw = true; }
        Check(threw, name);
    }
};

IsaV1CollectiveByteBuildSpec Spec(
    uint16_t tree = 0xabcd, uint16_t session = 0x2468,
    CollectiveKey key = {0x12345678, 0x90abcdef, 0x13579bdf}) {
    return {IsaV1CollectiveByteKind::DCA_REDUCE, tree, session, key};
}

std::vector<uint8_t> Bytes(size_t count) {
    std::vector<uint8_t> bytes(count);
    for (size_t i = 0; i < count; ++i)
        bytes[i] = static_cast<uint8_t>((i * 37 + 11) & 0xff);
    return bytes;
}

IsaV1CollectiveByteLock Lock(const IsaV1CollectiveByteBuildSpec &spec) {
    return {spec.tree_id, spec.session_id, spec.collective.epoch};
}

} // namespace

int RunIsaV1CollectiveByteWireSelfTest() {
    Suite suite;

    suite.Check(IsaV1CollectiveByteLegacyRawType(
                        ISA_V1_COLL_BYTE_START_MAGIC) == MSG_TYPE_NUM &&
                    IsaV1CollectiveByteLegacyRawType(
                        ISA_V1_COLL_BYTE_DATA_MAGIC) == MSG_TYPE_NUM,
                "START and DATA magic decode to forbidden legacy Msg type 8");

    for (size_t length : {size_t{1}, size_t{15}, size_t{16},
                          size_t{17}, size_t{31}, size_t{32}}) {
        const auto bytes = Bytes(length);
        const auto spec = Spec();
        const auto built = BuildIsaV1CollectiveByteStream(spec, bytes);
        const auto start = DeserializeIsaV1CollectiveByteStart(built.start);
        suite.Check(IsIsaV1CollectiveByteStartWire(built.start) &&
                        start.kind == spec.kind &&
                        start.tree_id == spec.tree_id &&
                        start.session_id == spec.session_id &&
                        start.collective == spec.collective &&
                        start.total_bytes == length &&
                        start.checksum == P2pPayloadChecksum(bytes) &&
                        start.flags == 0,
                    "START carries full key, declaration, CRC32C and flags");
        suite.Check(!IsCollDataWire(built.start) &&
                        !IsCollReduceHeaderWire(built.start) &&
                        !IsCollReducePayloadWire(built.start),
                    "START is disjoint from known collective classifiers");
        suite.Check(built.data.size() ==
                        (length + P2P_PAYLOAD_FRAGMENT_BYTES - 1) /
                            P2P_PAYLOAD_FRAGMENT_BYTES,
                    "builder emits the exact bounded DATA count");
        for (size_t index = 0; index < built.data.size(); ++index) {
            const auto route =
                InspectIsaV1CollectiveByteDataWire(built.data[index]);
            suite.Check(IsIsaV1CollectiveByteDataWire(built.data[index]) &&
                            route.kind == spec.kind &&
                            route.lock == Lock(spec) &&
                            route.sequence == index + 1,
                        "each DATA independently exposes Router lock identity");
            suite.Check(!IsCollDataWire(built.data[index]) &&
                            !IsCollReduceHeaderWire(built.data[index]) &&
                            !IsCollReducePayloadWire(built.data[index]),
                        "canonical DATA is disjoint from known classifiers");
        }
        IsaV1CollectiveByteReassembler runtime(length, 1);
        runtime.Begin(built.start);
        std::optional<IsaV1CollectiveByteCommit> commit;
        for (size_t index = 0; index < built.data.size(); ++index) {
            commit = runtime.Accept(built.data[index]);
            suite.Check(commit.has_value() ==
                            (index + 1 == built.data.size()),
                        "bytes remain uncommitted until validated tail");
        }
        suite.Check(commit && commit->bytes == bytes &&
                        commit->identity.collective == spec.collective &&
                        runtime.InflightStreams() == 0 &&
                        runtime.ReservedBytes() == 0,
                    "CRC-valid tail atomically commits and drains state");
    }

    const size_t limit = static_cast<size_t>(kDteEndpointP2pMaxBytes);
    const auto limit_built =
        BuildIsaV1CollectiveByteStream(Spec(), Bytes(limit));
    suite.Check(limit_built.data.size() == P2P_PAYLOAD_MAX_FRAGMENTS,
                "endpoint byte limit maps to the full u16 sequence space");
    const auto limit_tail =
        DeserializeIsaV1CollectiveByteData(limit_built.data.back());
    suite.Check(limit_tail.sequence == UINT16_MAX && limit_tail.tail &&
                    limit_tail.length_bytes == 16,
                "limit DATA tail uses sequence 65535 canonically");
    suite.Throws<std::length_error>(
        [&] { (void)BuildIsaV1CollectiveByteStream(Spec(), Bytes(limit + 1)); },
        "endpoint limit plus one is rejected");

    suite.Throws<std::invalid_argument>(
        [&] { (void)BuildIsaV1CollectiveByteStream(Spec(), {}); },
        "empty stream is rejected");
    auto invalid_spec = Spec();
    invalid_spec.tree_id = 0;
    suite.Throws<std::invalid_argument>(
        [&] { (void)BuildIsaV1CollectiveByteStream(invalid_spec, Bytes(1)); },
        "tree zero is rejected");
    invalid_spec = Spec();
    invalid_spec.session_id = 0;
    suite.Throws<std::invalid_argument>(
        [&] { (void)BuildIsaV1CollectiveByteStream(invalid_spec, Bytes(1)); },
        "session zero is reserved");

    const auto two = BuildIsaV1CollectiveByteStream(Spec(), Bytes(17));
    suite.Throws<std::invalid_argument>(
        [&] { (void)DeserializeMsg(two.start); },
        "START is strictly rejected by legacy Msg decoding");
    suite.Throws<std::invalid_argument>(
        [&] { (void)DeserializeMsg(two.data.front()); },
        "DATA is strictly rejected by legacy Msg decoding");

    Msg legacy;
    legacy.msg_type_ = MSG_TYPE::DATA;
    legacy.seq_id_ = 1;
    legacy.des_ = 2;
    legacy.offset_ = 3;
    legacy.tag_id_ = 4;
    legacy.source_ = 5;
    legacy.length_ = 128;
    legacy.roofline_packets_ = 1;
    sc_bv<256> legacy_wire = SerializeMsg(legacy);
    legacy_wire.range(143, 128) = ISA_V1_COLL_BYTE_DATA_MAGIC;
    suite.Check(DeserializeMsg(legacy_wire).msg_type_ == MSG_TYPE::DATA &&
                    !IsIsaV1CollectiveByteDataWire(legacy_wire),
                "legacy DATA payload may hit old magic position without classification");
    const auto p2p = BuildP2pPayload({1, 2, 3, 0}, 9, Bytes(16));
    sc_bv<256> p2p_wire = SerializeMsg(p2p.fragments.front());
    p2p_wire.range(143, 128) = ISA_V1_COLL_BYTE_DATA_MAGIC;
    suite.Check(DeserializeMsg(p2p_wire).p2p_endpoint_ &&
                    !IsIsaV1CollectiveByteDataWire(p2p_wire),
                "P2P business payload may hit old magic position without classification");

    // Endpoint P2P uses bit 255 as its pulse-gap discriminator. DATA owns the
    // whole upper half as business payload, so both values of that bit must
    // remain strict collective DATA while neither may become a valid Msg.
    auto data_bit255_clear = two.data.front();
    data_bit255_clear[255] = false;
    auto data_bit255_set = two.data.front();
    data_bit255_set[255] = true;
    suite.Check(IsIsaV1CollectiveByteDataWire(data_bit255_clear) &&
                    IsIsaV1CollectiveByteDataWire(data_bit255_set),
                "collective DATA classification is independent of endpoint bit255");
    suite.Throws<std::invalid_argument>(
        [&] { (void)DeserializeMsg(data_bit255_clear); },
        "collective DATA with bit255 clear cannot become endpoint Msg");
    suite.Throws<std::invalid_argument>(
        [&] { (void)DeserializeMsg(data_bit255_set); },
        "collective DATA with bit255 set cannot become endpoint Msg");

    IsaV1CollectiveByteData payload_collision;
    payload_collision.kind = IsaV1CollectiveByteKind::MULTICAST;
    payload_collision.lock = Lock(Spec());
    payload_collision.sequence = 1;
    payload_collision.length_bytes = 16;
    payload_collision.tail = true;
    payload_collision.payload.range(15, 0) = COLL_REDUCE_MAGIC;
    payload_collision.payload.range(23, 16) = COLL_REDUCE_VERSION;
    payload_collision.payload.range(31, 24) = COLL_REDUCE_PAYLOAD_SEGMENT;
    const sc_bv<256> priority_wire =
        SerializeIsaV1CollectiveByteData(payload_collision);
    suite.Check(IsIsaV1CollectiveByteDataWire(priority_wire) &&
                    IsCollReducePayloadWire(priority_wire),
                "arbitrary top payload proves Router must classify new DATA first");
    IsaV1CollectiveByteReassembler unknown(64, 2);
    suite.Throws<std::invalid_argument>(
        [&] { (void)unknown.Accept(two.data.front()); },
        "DATA without START is rejected");
    unknown.Begin(two.start);
    suite.Throws<std::invalid_argument>(
        [&] { unknown.Begin(two.start); },
        "active DATA lock cannot be reused");
    suite.Check(!unknown.Accept(two.data.front()).has_value(),
                "first DATA does not expose partial bytes");
    suite.Throws<std::invalid_argument>(
        [&] { (void)unknown.Accept(two.data.front()); },
        "duplicate DATA sequence is rejected");
    suite.Check(unknown.Abort(Lock(Spec())) &&
                    !unknown.HasActive(Lock(Spec())) &&
                    unknown.ReservedBytes() == 0 &&
                    !unknown.Abort(Lock(Spec())),
                "abort releases exact reservation and is idempotent false");

    IsaV1CollectiveByteReassembler collision(128, 3);
    collision.Begin(two.start);
    auto same_key = Spec(0xabce, 0x2469, Spec().collective);
    const auto key_collision =
        BuildIsaV1CollectiveByteStream(same_key, Bytes(17));
    suite.Throws<std::invalid_argument>(
        [&] { collision.Begin(key_collision.start); },
        "START detects full CollectiveKey collision across DATA locks");
    auto same_lock = Spec();
    same_lock.collective.group_id++;
    const auto lock_collision =
        BuildIsaV1CollectiveByteStream(same_lock, Bytes(17));
    suite.Throws<std::invalid_argument>(
        [&] { collision.Begin(lock_collision.start); },
        "START detects active tree/session/epoch identity reuse");
    collision.Abort(Lock(Spec()));
    collision.Begin(lock_collision.start);
    suite.Check(collision.Abort(Lock(same_lock)),
                "retired DATA lock may be allocated to a new full key");

    IsaV1CollectiveByteReassembler flow_capacity(128, 1);
    flow_capacity.Begin(two.start);
    const auto other = BuildIsaV1CollectiveByteStream(
        Spec(7, 8, {2, 3, 4}), Bytes(17));
    suite.Throws<std::length_error>(
        [&] { flow_capacity.Begin(other.start); },
        "inflight stream capacity is strict");
    flow_capacity.Abort(Lock(Spec()));
    IsaV1CollectiveByteReassembler byte_capacity(16, 2);
    suite.Throws<std::length_error>(
        [&] { byte_capacity.Begin(two.start); },
        "byte reservation capacity is strict and pre-data");
    suite.Check(byte_capacity.InflightStreams() == 0 &&
                    byte_capacity.ReservedBytes() == 0,
                "failed reservation is non-mutating");

    auto corrupt = two;
    corrupt.data.back()[128] = !corrupt.data.back()[128].to_bool();
    IsaV1CollectiveByteReassembler crc(32, 1);
    crc.Begin(corrupt.start);
    (void)crc.Accept(corrupt.data.front());
    suite.Throws<std::invalid_argument>(
        [&] { (void)crc.Accept(corrupt.data.back()); },
        "whole-stream CRC32C mismatch rejects atomic commit");
    suite.Check(crc.InflightStreams() == 0 && crc.ReservedBytes() == 0,
                "checksum failure releases all state and reservation");

    auto bad_start = two.start;
    bad_start.range(19, 16) = 1;
    suite.Check(!IsIsaV1CollectiveByteStartWire(bad_start),
                "unsupported START version is not recognized");
    suite.Throws<std::invalid_argument>(
        [&] { (void)DeserializeIsaV1CollectiveByteStart(bad_start); },
        "unsupported START version is rejected");
    bad_start = two.start;
    bad_start[255] = true;
    suite.Throws<std::invalid_argument>(
        [&] { (void)DeserializeIsaV1CollectiveByteStart(bad_start); },
        "START reserved bits must be zero");
    bad_start = two.start;
    bad_start.range(23, 20) = 15;
    suite.Throws<std::invalid_argument>(
        [&] { (void)DeserializeIsaV1CollectiveByteStart(bad_start); },
        "START unknown kind is rejected");

    auto bad_data = two.data.front();
    bad_data.range(19, 16) = 1;
    suite.Check(!IsIsaV1CollectiveByteDataWire(bad_data),
                "unsupported DATA version is not recognized");
    suite.Throws<std::invalid_argument>(
        [&] { (void)DeserializeIsaV1CollectiveByteData(bad_data); },
        "unsupported DATA version is rejected");
    bad_data = two.data.front();
    bad_data[127] = true;
    suite.Throws<std::invalid_argument>(
        [&] { (void)DeserializeIsaV1CollectiveByteData(bad_data); },
        "DATA reserved bits must be zero");
    bad_data = two.data.front();
    bad_data.range(23, 20) = 15;
    suite.Throws<std::invalid_argument>(
        [&] { (void)DeserializeIsaV1CollectiveByteData(bad_data); },
        "DATA unknown kind is rejected");
    bad_data = two.data.front();
    bad_data.range(103, 88) = 0;
    suite.Throws<std::invalid_argument>(
        [&] { (void)DeserializeIsaV1CollectiveByteData(bad_data); },
        "DATA sequence zero is rejected");
    bad_data = two.data.front();
    bad_data.range(107, 104) = 0;
    suite.Throws<std::invalid_argument>(
        [&] { (void)DeserializeIsaV1CollectiveByteData(bad_data); },
        "non-tail short DATA is rejected");
    bad_data = two.data.back();
    bad_data[255] = true;
    suite.Throws<std::invalid_argument>(
        [&] { (void)DeserializeIsaV1CollectiveByteData(bad_data); },
        "tail payload padding must be zero");

    auto wrong_epoch = two.data.front();
    wrong_epoch.range(87, 56) = Spec().collective.epoch + 1;
    IsaV1CollectiveByteReassembler identity(32, 1);
    identity.Begin(two.start);
    suite.Throws<std::invalid_argument>(
        [&] { (void)identity.Accept(wrong_epoch); },
        "DATA epoch mismatch cannot borrow START sideband");
    auto wrong_kind = two.data.front();
    wrong_kind.range(23, 20) =
        static_cast<uint8_t>(IsaV1CollectiveByteKind::MULTICAST);
    suite.Throws<std::invalid_argument>(
        [&] { (void)identity.Accept(wrong_kind); },
        "DATA kind must match START");
    auto early_tail = two.data.front();
    early_tail[108] = true;
    suite.Throws<std::invalid_argument>(
        [&] { (void)identity.Accept(early_tail); },
        "early tail is rejected against START declaration");
    identity.Abort(Lock(Spec()));

    std::cout << "ISA v1 collective byte wire self-test: "
              << (suite.failures == 0
                      ? "PASS"
                      : "FAILURES=" + std::to_string(suite.failures))
              << " (" << suite.checks << " checks)\n";
    return suite.failures;
}
