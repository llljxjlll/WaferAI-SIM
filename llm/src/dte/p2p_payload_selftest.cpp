#include "dte/p2p_payload.h"
#include "dte/p2p_payload_selftest.h"

#include "utils/msg_utils.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

class TestSuite {
public:
    void Check(bool condition, const std::string &message) {
        ++checks_;
        if (!condition) {
            ++failures_;
            std::cerr << "[P2P Payload] FAIL: " << message << '\n';
        }
    }

    template <typename Fn>
    void Throws(Fn &&fn, const std::string &message) {
        ++checks_;
        try {
            fn();
        } catch (const std::exception &) {
            return;
        } catch (...) {
            return;
        }
        ++failures_;
        std::cerr << "[P2P Payload] FAIL: expected rejection: " << message
                  << '\n';
    }

    int Finish() const {
        if (failures_ == 0)
            std::cout << "[P2P Payload] PASS (" << checks_ << " checks)\n";
        return failures_;
    }

private:
    int checks_ = 0;
    int failures_ = 0;
};

enum class Pattern { ZERO, ONE, INCREMENT, SEEDED_RANDOM };

std::vector<uint8_t> MakeBytes(size_t size, Pattern pattern) {
    std::vector<uint8_t> bytes(size);
    switch (pattern) {
    case Pattern::ZERO:
        break;
    case Pattern::ONE:
        std::fill(bytes.begin(), bytes.end(), uint8_t{1});
        break;
    case Pattern::INCREMENT:
        for (size_t i = 0; i < size; ++i)
            bytes[i] = static_cast<uint8_t>(i & 0xffU);
        break;
    case Pattern::SEEDED_RANDOM: {
        std::mt19937 generator(0x5a17c0deU ^ static_cast<uint32_t>(size));
        std::uniform_int_distribution<unsigned> distribution(0, 255);
        for (uint8_t &byte : bytes)
            byte = static_cast<uint8_t>(distribution(generator));
        break;
    }
    }
    return bytes;
}

Msg WireRoundTrip(const Msg &msg) {
    return DeserializeMsg(SerializeMsg(msg));
}

void FlipLowestBit(Msg &msg, size_t byte_index) {
    const int low = static_cast<int>(byte_index * 8);
    const uint8_t byte =
        static_cast<uint8_t>(msg.data_.range(low + 7, low).to_uint64());
    msg.data_.range(low + 7, low) = sc_bv<8>(byte ^ 1U);
}

P2pBuiltPayload BuildPayloadForTest(
    const P2pFlowKey &flow, const std::vector<uint8_t> &bytes,
    uint32_t fsm_id = 0x10001U) {
    return BuildP2pPayload(flow, fsm_id, bytes);
}

void TestFsmTransportAbi(TestSuite &suite) {
    const P2pFlowKey flow{3, 4, 7, 0};
    const std::vector<uint8_t> bytes =
        MakeBytes(33, Pattern::SEEDED_RANDOM);
    const P2pBuiltPayload low = BuildPayloadForTest(flow, bytes, 1);
    const P2pBuiltPayload high =
        BuildPayloadForTest(flow, bytes, 0x10001U);
    suite.Check(ParseP2pPayloadRequest(WireRoundTrip(low.request)).fsm_id == 1,
                "REQUEST preserves low endpoint fsm_id");
    suite.Check(ParseP2pPayloadRequest(WireRoundTrip(high.request)).fsm_id ==
                    0x10001U,
                "REQUEST preserves high endpoint fsm_id without truncation");
    suite.Check(low.fragments.size() == high.fragments.size(),
                "fsm_id does not change DATA fragment count");
    suite.Check(high.request.p2p_endpoint_ &&
                    high.request.offset_ == P2P_ENDPOINT_MSG_MARKER &&
                    high.fragments.front().p2p_endpoint_ &&
                    high.fragments.front().offset_ ==
                        P2P_ENDPOINT_MSG_MARKER &&
                    high.fragments.front().roofline_packets_ == 1,
                "endpoint marker and one-real-fragment roofline are canonical");
    const sc_bv<256> endpoint_data_wire =
        SerializeMsg(high.fragments.front());
    const Msg endpoint_data_roundtrip = DeserializeMsg(endpoint_data_wire);
    suite.Check(endpoint_data_roundtrip.dte_stream_source_first_ns_ == 0 &&
                    endpoint_data_roundtrip.dte_stream_source_done_ns_ == 0 &&
                    endpoint_data_roundtrip.dte_stream_network_tail_cycles_ == 0,
                "endpoint DATA payload is never decoded as legacy timing metadata");
    suite.Check(SerializeMsg(endpoint_data_roundtrip) == endpoint_data_wire,
                "endpoint DATA repeated serialization is bit-exact");
    sc_bv<256> request_without_endpoint = SerializeMsg(high.request);
    request_without_endpoint[255] = false;
    suite.Throws(
        [&] {
            (void)ParseP2pPayloadRequest(
                DeserializeMsg(request_without_endpoint));
        },
        "endpoint REQUEST missing bit255 discriminator");
    sc_bv<256> data_without_endpoint =
        SerializeMsg(high.fragments.front());
    data_without_endpoint[255] = false;
    suite.Throws(
        [&] {
            (void)ParseP2pDataFragment(
                DeserializeMsg(data_without_endpoint));
        },
        "endpoint DATA missing bit255 discriminator");
    for (size_t i = 0; i < low.fragments.size(); ++i)
        suite.Check(SerializeMsg(low.fragments[i]) ==
                        SerializeMsg(high.fragments[i]),
                    "DATA wire is bit-equivalent across endpoint fsm_id values");

    Msg low_without_fsm = low.request;
    Msg high_without_fsm = high.request;
    low_without_fsm.data_.range(63, 32) = sc_bv<32>(0);
    high_without_fsm.data_.range(63, 32) = sc_bv<32>(0);
    suite.Check(SerializeMsg(low_without_fsm) ==
                    SerializeMsg(high_without_fsm),
                "REQUEST fsm word is the only wire difference");

    Msg pinned_request = high.request;
    pinned_request.exit_port_ = static_cast<int>(UINT16_MAX) - 1;
    const Msg pinned_request_wire = WireRoundTrip(pinned_request);
    suite.Check(pinned_request_wire.exit_port_ ==
                        static_cast<int>(UINT16_MAX) - 1 &&
                    ParseP2pPayloadRequest(pinned_request_wire).fsm_id ==
                        0x10001U,
                "pinned REQUEST round trip preserves routing and fsm metadata");
    Msg pinned_data = high.fragments.front();
    pinned_data.exit_port_ = static_cast<int>(UINT16_MAX) - 1;
    const Msg pinned_data_wire = WireRoundTrip(pinned_data);
    const P2pDataFragment parsed_pinned_data =
        ParseP2pDataFragment(pinned_data_wire);
    suite.Check(pinned_data_wire.exit_port_ ==
                        static_cast<int>(UINT16_MAX) - 1 &&
                    parsed_pinned_data.bytes[0] == bytes[0],
                "pinned DATA round trip preserves routing and business bytes");
    suite.Throws([&] { (void)BuildPayloadForTest(flow, bytes, 0); },
                 "zero endpoint fsm_id");
    P2pFlowKey zero_tag = flow;
    zero_tag.transport_tag = 0;
    suite.Throws([&] { (void)BuildPayloadForTest(zero_tag, bytes, 1); },
                 "zero transport tag");
}

void TestLengthsAndPatterns(TestSuite &suite) {
    const std::vector<size_t> lengths = {
        1,   15,  16,  17,  47,   48,   49,  127,
        128, 129, 255, 256, 257, 4096, 32768,
    };
    const std::vector<Pattern> patterns = {
        Pattern::ZERO,
        Pattern::ONE,
        Pattern::INCREMENT,
        Pattern::SEEDED_RANDOM,
    };
    const P2pFlowKey flow{7, 11, 0x1234, 2};

    for (size_t length : lengths) {
        for (Pattern pattern : patterns) {
            const std::vector<uint8_t> bytes = MakeBytes(length, pattern);
            const P2pBuiltPayload built = BuildPayloadForTest(flow, bytes);
            const Msg decoded_request = WireRoundTrip(built.request);
            const P2pPayloadDeclaration declaration =
                ParseP2pPayloadRequest(decoded_request);
            const size_t expected_fragments =
                (length + P2P_PAYLOAD_FRAGMENT_BYTES - 1) /
                P2P_PAYLOAD_FRAGMENT_BYTES;
            suite.Check(declaration.flow == flow,
                        "REQUEST flow identity survives wire round trip");
            suite.Check(declaration.fsm_id == 0x10001U,
                        "REQUEST preserves full 32-bit endpoint fsm_id");
            suite.Check(declaration.total_bytes == length,
                        "REQUEST declares exact byte length");
            suite.Check(declaration.fragment_count == expected_fragments,
                        "REQUEST declares exact fragment count");
            suite.Check(declaration.checksum == P2pPayloadChecksum(bytes),
                        "REQUEST declares exact CRC32C");
            suite.Check(built.fragments.size() == expected_fragments,
                        "builder emits ceil(length/16) fragments");

            P2pPayloadReassembler reassembler(32768, 4);
            reassembler.Begin(decoded_request);
            suite.Check(reassembler.ReservedBytes() == length,
                        "Begin reserves full declared length");
            for (size_t i = 0; i < built.fragments.size(); ++i) {
                const Msg decoded = WireRoundTrip(built.fragments[i]);
                suite.Check(decoded.data_ == built.fragments[i].data_,
                            "DATA business bytes survive codec unchanged");
                const P2pDataFragment parsed = ParseP2pDataFragment(decoded);
                suite.Check(parsed.sequence == i + 1,
                            "DATA sequence is one-based and contiguous");
                suite.Check(parsed.tail == (i + 1 == expected_fragments),
                            "only final DATA is tail");
                const size_t expected_length = std::min<size_t>(
                    P2P_PAYLOAD_FRAGMENT_BYTES, length - i * 16);
                suite.Check(parsed.length_bytes == expected_length,
                            "DATA byte length is exact");
                for (size_t j = 0; j < expected_length; ++j)
                    suite.Check(parsed.bytes[j] == bytes[i * 16 + j],
                                "DATA contains exact business byte");

                const std::optional<P2pPayloadCommit> commit =
                    reassembler.Accept(decoded);
                if (i + 1 != expected_fragments) {
                    suite.Check(!commit.has_value(),
                                "reassembler exposes no partial payload");
                } else {
                    suite.Check(commit.has_value(),
                                "reassembler commits at validated tail");
                    suite.Check(commit && commit->flow == flow,
                                "commit retains flow identity");
                    suite.Check(commit && commit->bytes == bytes,
                                "commit contains exact full payload");
                }
            }
            suite.Check(reassembler.InflightFlows() == 0 &&
                            reassembler.ReservedBytes() == 0,
                        "completion releases all bounded state");
        }
    }
}

void TestUnalignedSlice(TestSuite &suite) {
    const std::vector<uint8_t> backing =
        MakeBytes(512, Pattern::SEEDED_RANDOM);
    const std::vector<uint8_t> slice(backing.begin() + 3,
                                     backing.begin() + 3 + 129);
    const P2pFlowKey flow{1, 2, 3, 1};
    const P2pBuiltPayload built = BuildPayloadForTest(flow, slice);
    P2pPayloadReassembler reassembler(256, 1);
    reassembler.Begin(WireRoundTrip(built.request));
    std::optional<P2pPayloadCommit> commit;
    for (const Msg &fragment : built.fragments)
        commit = reassembler.Accept(WireRoundTrip(fragment));
    suite.Check(commit && commit->bytes == slice,
                "non-aligned source slice is reconstructed byte-exactly");
}

void TestChecksum(TestSuite &suite) {
    const std::string standard = "123456789";
    const std::vector<uint8_t> bytes(standard.begin(), standard.end());
    suite.Check(P2pPayloadChecksum(bytes) == 0xe3069283U,
                "CRC32C matches the standard check vector");

    const P2pFlowKey flow{9, 10, 11, 0};
    P2pBuiltPayload built =
        BuildPayloadForTest(flow, MakeBytes(17, Pattern::INCREMENT));
    P2pPayloadReassembler reassembler(64, 1);
    reassembler.Begin(built.request);
    suite.Check(!reassembler.Accept(built.fragments[0]).has_value(),
                "bitflip test has no pre-tail commit");
    FlipLowestBit(built.fragments[1], 0);
    suite.Throws([&] { (void)reassembler.Accept(built.fragments[1]); },
                 "tail bitflip fails whole-flow checksum");
    suite.Check(!reassembler.HasActive(flow) &&
                    reassembler.ReservedBytes() == 0,
                "checksum failure discards all uncommitted state");

    built = BuildPayloadForTest(flow, MakeBytes(32, Pattern::ONE));
    FlipLowestBit(built.request, 0);
    reassembler.Begin(built.request);
    suite.Check(!reassembler.Accept(built.fragments[0]).has_value(),
                "corrupt checksum declaration still exposes no partial data");
    suite.Throws([&] { (void)reassembler.Accept(built.fragments[1]); },
                 "corrupt REQUEST checksum is detected at completion");
    suite.Check(reassembler.InflightFlows() == 0,
                "declaration checksum failure removes active flow");
}

void TestStrictRequestValidation(TestSuite &suite) {
    const P2pFlowKey flow{20, 21, 22, 3};
    const P2pBuiltPayload valid =
        BuildPayloadForTest(flow, MakeBytes(17, Pattern::SEEDED_RANDOM));
    suite.Throws([&] { (void)BuildPayloadForTest(flow, {}); },
                 "zero-byte payload");
    suite.Throws(
        [&] {
            P2pFlowKey invalid = flow;
            invalid.subflow = 4;
            (void)BuildPayloadForTest(invalid, {1});
        },
        "subflow outside two-bit wire");

    Msg bad = valid.request;
    bad.msg_type_ = MSG_TYPE::ACK;
    suite.Throws([&] { (void)ParseP2pPayloadRequest(bad); },
                 "non-REQUEST declaration");
    bad = valid.request;
    bad.offset_ = 0;
    suite.Throws([&] { (void)ParseP2pPayloadRequest(bad); },
                 "legacy offset-zero REQUEST");
    bad = valid.request;
    bad.offset_ = P2P_ENDPOINT_MSG_MARKER ^ 1U;
    suite.Throws([&] { (void)ParseP2pPayloadRequest(bad); },
                 "REQUEST endpoint marker bitflip");
    Msg legacy_request(MSG_TYPE::REQUEST, flow.destination,
                       flow.transport_tag, flow.source);
    legacy_request.offset_ = P2P_ENDPOINT_MSG_MARKER;
    legacy_request.data_ = 0;
    suite.Throws([&] { (void)ParseP2pPayloadRequest(legacy_request); },
                 "legacy REQUEST marker collision");
    bad = valid.request;
    bad.seq_id_ = 1;
    suite.Throws([&] { (void)ParseP2pPayloadRequest(bad); },
                 "non-canonical REQUEST sequence");
    bad = valid.request;
    bad.exit_port_ = -2;
    suite.Throws([&] { (void)ParseP2pPayloadRequest(bad); },
                 "REQUEST exit below unpinned sentinel");
    bad.exit_port_ = static_cast<int>(UINT16_MAX);
    suite.Throws([&] { (void)ParseP2pPayloadRequest(bad); },
                 "REQUEST exit exceeds decoded wire range");
    bad = valid.request;
    bad.source_ = static_cast<int>(UINT16_MAX) + 1;
    suite.Throws([&] { (void)ParseP2pPayloadRequest(bad); },
                 "REQUEST source exceeds endpoint wire width");
    bad = valid.request;
    bad.dte_payload_bits_ = 0;
    bad.data_.range(127, 64) = sc_bv<64>(0);
    suite.Throws([&] { (void)ParseP2pPayloadRequest(bad); },
                 "zero total bits");
    bad = valid.request;
    bad.dte_payload_bits_ = 9;
    bad.data_.range(127, 64) = sc_bv<64>(9);
    suite.Throws([&] { (void)ParseP2pPayloadRequest(bad); },
                 "non-byte total bits");
    bad = valid.request;
    bad.dte_payload_bits_ -= 8;
    suite.Throws([&] { (void)ParseP2pPayloadRequest(bad); },
                 "total-bits metadata mismatch");
    bad = valid.request;
    bad.data_.range(63, 32) = sc_bv<32>(0);
    suite.Throws([&] { (void)ParseP2pPayloadRequest(bad); },
                 "zero REQUEST fsm_id");
    bad = valid.request;
    bad.flow_packets_ += 1;
    suite.Throws([&] { (void)ParseP2pPayloadRequest(bad); },
                 "fragment count mismatch");

    bad = valid.request;
    constexpr uint64_t too_many_bytes =
        P2P_PAYLOAD_MAX_FRAGMENTS * P2P_PAYLOAD_FRAGMENT_BYTES + 1;
    bad.dte_payload_bits_ = too_many_bytes * 8;
    bad.data_.range(127, 64) = sc_bv<64>(bad.dte_payload_bits_);
    bad.flow_packets_ = static_cast<int>(P2P_PAYLOAD_MAX_FRAGMENTS + 1);
    suite.Throws([&] { (void)ParseP2pPayloadRequest(bad); },
                 "payload exceeds 16-bit DATA sequence space");
}

void TestStrictFragmentValidation(TestSuite &suite) {
    const P2pFlowKey flow{30, 31, 32, 1};
    const P2pBuiltPayload valid =
        BuildPayloadForTest(flow, MakeBytes(17, Pattern::INCREMENT));
    Msg bad = valid.fragments[0];
    bad.msg_type_ = MSG_TYPE::ACK;
    suite.Throws([&] { (void)ParseP2pDataFragment(bad); },
                 "non-DATA fragment");
    bad = valid.fragments[0];
    bad.seq_id_ = 0;
    suite.Throws([&] { (void)ParseP2pDataFragment(bad); },
                 "zero DATA sequence");
    bad = valid.fragments[0];
    bad.seq_id_ = static_cast<int>(UINT16_MAX) + 1;
    suite.Throws([&] { (void)ParseP2pDataFragment(bad); },
                 "DATA sequence exceeds wire width");
    bad = valid.fragments[0];
    bad.tag_id_ = -1;
    suite.Throws([&] { (void)ParseP2pDataFragment(bad); },
                 "DATA tag outside flow identity wire width");
    bad = valid.fragments[0];
    bad.exit_port_ = -2;
    suite.Throws([&] { (void)ParseP2pDataFragment(bad); },
                 "DATA exit below unpinned sentinel");
    bad.exit_port_ = static_cast<int>(UINT16_MAX);
    suite.Throws([&] { (void)ParseP2pDataFragment(bad); },
                 "DATA exit exceeds decoded wire range");
    bad = valid.fragments[0];
    bad.offset_ = 0;
    suite.Throws([&] { (void)ParseP2pDataFragment(bad); },
                 "legacy offset-zero DATA");
    bad = valid.fragments[0];
    bad.offset_ = P2P_ENDPOINT_MSG_MARKER ^ 1U;
    suite.Throws([&] { (void)ParseP2pDataFragment(bad); },
                 "DATA endpoint marker bitflip");
    Msg legacy_data(false, MSG_TYPE::DATA, 1, flow.destination,
                    P2P_ENDPOINT_MSG_MARKER, flow.transport_tag, 128,
                    sc_bv<128>(1));
    legacy_data.source_ = flow.source;
    const Msg legacy_data_roundtrip = WireRoundTrip(legacy_data);
    suite.Check(!legacy_data_roundtrip.p2p_endpoint_,
                "legacy DATA offset collision keeps bit255 clear");
    suite.Throws([&] { (void)ParseP2pDataFragment(legacy_data); },
                 "legacy DATA marker collision");
    bad = valid.fragments[0];
    bad.roofline_packets_ = 0;
    suite.Throws([&] { (void)ParseP2pDataFragment(bad); },
                 "DATA missing one-real-fragment roofline");
    bad.roofline_packets_ = 2;
    suite.Throws([&] { (void)ParseP2pDataFragment(bad); },
                 "DATA aggregated roofline is forbidden");
    bad = valid.fragments[0];
    bad.dte_payload_bits_ = 8;
    suite.Throws([&] { (void)ParseP2pDataFragment(bad); },
                 "DATA carrying REQUEST total bits");
    bad = valid.fragments[0];
    bad.length_ = 7;
    suite.Throws([&] { (void)ParseP2pDataFragment(bad); },
                 "non-byte DATA length");
    bad = valid.fragments[0];
    bad.length_ = 136;
    suite.Throws([&] { (void)ParseP2pDataFragment(bad); },
                 "DATA longer than 16 bytes");
    bad = valid.fragments[0];
    bad.length_ = 120;
    suite.Throws([&] { (void)ParseP2pDataFragment(bad); },
                 "short non-tail DATA");
    bad = valid.fragments[1];
    bad.data_.range(15, 8) = sc_bv<8>(1);
    suite.Throws([&] { (void)ParseP2pDataFragment(bad); },
                 "non-zero tail padding");
}

void TestSequenceTailAndIdentity(TestSuite &suite) {
    const P2pFlowKey flow{40, 41, 42, 0};
    P2pBuiltPayload built =
        BuildPayloadForTest(flow, MakeBytes(32, Pattern::INCREMENT));
    P2pPayloadReassembler reassembler(64, 2);
    reassembler.Begin(built.request);
    suite.Throws([&] { (void)reassembler.Accept(built.fragments[1]); },
                 "out-of-order DATA");
    suite.Check(reassembler.HasActive(flow) &&
                    reassembler.ReservedBytes() == 32,
                "out-of-order rejection leaves flow state atomic");
    suite.Check(!reassembler.Accept(built.fragments[0]).has_value(),
                "first ordered DATA is accepted without commit");
    suite.Throws([&] { (void)reassembler.Accept(built.fragments[0]); },
                 "duplicate DATA sequence");
    const std::optional<P2pPayloadCommit> commit =
        reassembler.Accept(built.fragments[1]);
    suite.Check(commit && commit->bytes.size() == 32,
                "flow completes after rejected out-of-order and duplicate DATA");
    suite.Throws([&] { (void)reassembler.Accept(built.fragments[1]); },
                 "DATA after completed flow");

    built = BuildPayloadForTest(flow, MakeBytes(32, Pattern::ONE));
    reassembler.Begin(built.request);
    Msg early_tail = built.fragments[0];
    early_tail.is_end_ = true;
    suite.Throws([&] { (void)reassembler.Accept(early_tail); },
                 "early tail");
    suite.Check(reassembler.HasActive(flow),
                "early-tail rejection does not advance active flow");
    (void)reassembler.Abort(flow);

    reassembler.Begin(built.request);
    (void)reassembler.Accept(built.fragments[0]);
    Msg missing_tail = built.fragments[1];
    missing_tail.is_end_ = false;
    suite.Throws([&] { (void)reassembler.Accept(missing_tail); },
                 "missing tail flag");
    suite.Check(reassembler.HasActive(flow),
                "missing-tail rejection preserves active flow");
    (void)reassembler.Abort(flow);

    built = BuildPayloadForTest(flow, MakeBytes(17, Pattern::ONE));
    reassembler.Begin(built.request);
    (void)reassembler.Accept(built.fragments[0]);
    Msg wrong_length = built.fragments[1];
    wrong_length.length_ = 128;
    suite.Throws([&] { (void)reassembler.Accept(wrong_length); },
                 "tail length does not match declaration");
    (void)reassembler.Abort(flow);

    reassembler.Begin(built.request);
    Msg wrong_flow = built.fragments[0];
    ++wrong_flow.tag_id_;
    suite.Throws([&] { (void)reassembler.Accept(wrong_flow); },
                 "DATA flow identity differs from REQUEST");
    suite.Check(reassembler.HasActive(flow),
                "identity rejection preserves original flow");
    (void)reassembler.Abort(flow);

    suite.Throws([&] { (void)reassembler.Accept(built.fragments[0]); },
                 "DATA before REQUEST");
}

void TestReassemblerBounds(TestSuite &suite) {
    suite.Throws([] { P2pPayloadReassembler invalid(0, 1); },
                 "zero reassembler byte capacity");
    suite.Throws([] { P2pPayloadReassembler invalid(1, 0); },
                 "zero reassembler flow capacity");

    const P2pFlowKey first{50, 51, 52, 0};
    const P2pFlowKey second{50, 51, 53, 0};
    const P2pBuiltPayload first_payload =
        BuildPayloadForTest(first, MakeBytes(17, Pattern::ONE));
    const P2pBuiltPayload second_payload =
        BuildPayloadForTest(second, MakeBytes(1, Pattern::ONE));
    P2pPayloadReassembler reassembler(32, 1);
    reassembler.Begin(first_payload.request);
    suite.Check(reassembler.ResidualCapacityBytes() == 15 &&
                    reassembler.InflightFlows() == 1,
                "reassembler reports exact residual bounded capacity");
    suite.Throws([&] { reassembler.Begin(first_payload.request); },
                 "duplicate active REQUEST");
    suite.Check(reassembler.ReservedBytes() == 17,
                "duplicate REQUEST leaves reservation unchanged");
    suite.Throws([&] { reassembler.Begin(second_payload.request); },
                 "inflight flow capacity");
    suite.Check(reassembler.Abort(first), "Abort removes active flow");
    suite.Check(!reassembler.Abort(first), "second Abort reports missing flow");
    suite.Check(reassembler.ReservedBytes() == 0 &&
                    reassembler.ResidualCapacityBytes() == 32,
                "Abort releases full declaration reservation");

    const P2pBuiltPayload too_large =
        BuildPayloadForTest(first, MakeBytes(33, Pattern::ZERO));
    suite.Throws([&] { reassembler.Begin(too_large.request); },
                 "declared bytes exceed capacity");
    suite.Check(reassembler.InflightFlows() == 0 &&
                    reassembler.ReservedBytes() == 0,
                "failed admission is atomic");
}

void TestTimingSideband(TestSuite &suite) {
    suite.Throws([] { P2pTimingSidebandRegistry invalid(0); },
                 "zero timing sideband capacity");

    const P2pFlowKey flow{60, 61, 62, 2};
    const P2pTimingKey round_zero{flow, 0};
    const P2pTimingKey round_one{flow, 1};
    const P2pTimingMetadata first{10, 20, 30};
    const P2pTimingMetadata second{40, 50, 60};
    P2pTimingSidebandRegistry registry(2);
    registry.Publish(round_zero, first);
    suite.Check(registry.Residual() == 1 && !registry.Full() &&
                    registry.Contains(round_zero) &&
                    registry.FrontKey() == round_zero,
                "sideband reports first residual entry and FIFO front");
    suite.Throws([&] { registry.Publish(round_zero, second); },
                 "duplicate sideband key");
    suite.Check(registry.Residual() == 1,
                "duplicate publish leaves FIFO unchanged");
    registry.Publish(round_one, second);
    suite.Check(registry.Residual() == 2 && registry.Full(),
                "sideband reaches exact configured capacity");

    P2pTimingKey third{flow, 2};
    suite.Throws([&] { registry.Publish(third, first); },
                 "sideband capacity exhaustion");
    suite.Throws([&] { (void)registry.Consume(round_one); },
                 "out-of-order round consumption");
    suite.Check(registry.Residual() == 2 && registry.FrontKey() == round_zero,
                "round isolation failure leaves FIFO unchanged");
    suite.Check(registry.Consume(round_zero) == first,
                "first round consumes exact timing metadata");
    suite.Check(!registry.Contains(round_zero) &&
                    registry.Contains(round_one) && registry.Residual() == 1,
                "consumption removes only matching round");
    suite.Check(registry.Consume(round_one) == second && registry.Empty() &&
                    registry.Residual() == 0,
                "second round drains FIFO exactly");
    suite.Throws([&] { (void)registry.Consume(round_one); },
                 "consume empty sideband");
    suite.Throws([&] { (void)registry.FrontKey(); },
                 "inspect empty sideband front");

    P2pTimingKey invalid_key{P2pFlowKey{1, 2, 3, 4}, 0};
    suite.Throws([&] { registry.Publish(invalid_key, first); },
                 "invalid sideband subflow");
    suite.Throws(
        [&] { registry.Publish(round_zero, P2pTimingMetadata{20, 10, 0}); },
        "reversed sideband timing interval");

    const std::vector<uint8_t> payload =
        MakeBytes(16, Pattern::SEEDED_RANDOM);
    const P2pBuiltPayload built = BuildPayloadForTest(flow, payload);
    const sc_bv<128> before = built.fragments.front().data_;
    registry.Publish(round_zero, first);
    suite.Check(built.fragments.front().data_ == before,
                "sideband publication cannot overwrite business payload");
    (void)registry.Consume(round_zero);
}

void TestSharedTimingSideband(TestSuite &suite) {
    using Shared = P2pSharedTimingSidebandRuntime;
    Shared::Reset();
    suite.Throws([] { Shared::Configure(0); },
                 "zero shared timing capacity");
    Shared::Configure(2);
    suite.Check(Shared::Capacity() == 2 && Shared::Residual() == 0,
                "shared timing starts with exact configured bound");

    const P2pFlowKey first_flow{70, 80, 1, 0};
    const P2pFlowKey second_flow{70, 81, 2, 1};
    const P2pFlowKey third_flow{71, 82, 3, 2};
    const P2pTimingKey first_key{first_flow, 1};
    const P2pTimingKey second_key{second_flow, 2};
    const P2pTimingKey third_key{third_flow, 1};

    suite.Throws(
        [&] { Shared::UpdateNetworkTail(first_flow, 7); },
        "network update before source publication");
    suite.Throws([&] { (void)Shared::Consume(first_flow, 1); },
                 "destination consume before source publication");
    suite.Throws([&] { Shared::Publish(first_key, 0, 10, 20); },
                 "zero shared timing fsm_id");
    suite.Throws(
        [&] { Shared::Publish(P2pTimingKey{first_flow, 0}, 1, 10, 20); },
        "zero shared timing round");
    suite.Throws([&] { Shared::Publish(first_key, 1, 20, 10); },
                 "reversed shared timing source interval");

    Shared::Publish(first_key, 1, 10, 20);
    suite.Throws(
        [&] { Shared::CompleteNetworkAtNs(first_flow, 19, 1); },
        "network arrival before source completion");
    suite.Throws(
        [&] { Shared::CompleteNetworkAtNs(first_flow, 20, 0); },
        "zero network cycle duration");
    suite.Check(Shared::Contains(first_flow) && Shared::Residual() == 1,
                "source publication is globally visible and bounded");
    suite.Throws([&] { Shared::Publish(first_key, 1, 10, 20); },
                 "duplicate shared timing publication");
    suite.Throws([&] { (void)Shared::Consume(first_flow, 1); },
                 "consume before network completion");
    suite.Throws([&] { (void)Shared::Consume(first_flow, 0x10001U); },
                 "full fsm mismatch before destination consumption");
    suite.Check(Shared::Residual() == 1,
                "failed same-flow transitions are atomic");

    Shared::Publish(second_key, 0x10001U, 30, 40);
    suite.Check(Shared::Residual() == 2,
                "multiple active flows reach configured capacity");
    suite.Throws([&] { Shared::Publish(third_key, 3, 50, 60); },
                 "shared timing capacity exhaustion");
    suite.Check(!Shared::Contains(third_flow) && Shared::Residual() == 2,
                "capacity rejection does not publish partial state");

    // Complete and consume the later round before the earlier round. Ordering
    // is strict only within a flow, not across independent flows.
    Shared::CompleteNetworkAtNs(second_flow, 241, 10);
    suite.Throws([&] { Shared::UpdateNetworkTail(second_flow, 201); },
                 "duplicate network completion");
    const P2pTimingMetadata second =
        Shared::Consume(second_flow, 0x10001U);
    suite.Check(second == P2pTimingMetadata{30, 40, 21} &&
                    Shared::Residual() == 1 &&
                    !Shared::Contains(second_flow),
                "later flow completes first with exact timing metadata");
    suite.Throws([&] { (void)Shared::Consume(second_flow, 0x10001U); },
                 "stale duplicate destination consume");
    suite.Throws([&] { Shared::UpdateNetworkTail(second_flow, 202); },
                 "stale network update after consumption");
    suite.Throws([&] { Shared::Abort(second_flow, 0x10001U); },
                 "stale abort after destination consumption");

    Shared::UpdateNetworkTail(first_flow, 0);
    const P2pTimingMetadata first = Shared::Consume(first_flow, 1);
    suite.Check(first == P2pTimingMetadata{10, 20, 0} &&
                    Shared::Residual() == 0,
                "zero network tail is complete state, not missing state");

    suite.Throws([&] { Shared::Publish(first_key, 1, 60, 70); },
                 "stale source round after retirement");
    suite.Throws(
        [&] {
            Shared::Publish(P2pTimingKey{second_flow, 1}, 0x10001U, 60,
                            70);
        },
        "stale lower source round on another flow");

    Shared::Publish(third_key, 3, 80, 90);
    suite.Check(Shared::Contains(third_flow) && Shared::Residual() == 1,
                "abort fixture publishes one active timing entry");
    suite.Throws([&] { Shared::Abort(third_flow, 4); },
                 "abort rejects full fsm mismatch");
    suite.Check(Shared::Contains(third_flow) && Shared::Residual() == 1,
                "failed abort leaves active timing entry unchanged");
    Shared::Abort(third_flow, 3);
    suite.Check(!Shared::Contains(third_flow) && Shared::Residual() == 0,
                "matching abort releases the complete timing residual");
    suite.Throws([&] { Shared::Abort(third_flow, 3); },
                 "duplicate abort of retired flow");
    suite.Throws(
        [&] { Shared::Abort(P2pFlowKey{72, 82, 4, 0}, 4); },
        "abort of unknown flow");
    suite.Throws([&] { Shared::Publish(third_key, 3, 80, 90); },
                 "aborted source generation remains stale");

    const P2pTimingKey reused_key{first_flow, 3};
    Shared::Publish(reused_key, 0x10001U, 60, 70);
    suite.Check(Shared::Contains(first_flow),
                "ACK-safe flow reuse accepts a newer source round");
    suite.Throws([&] { Shared::Configure(3); },
                 "reconfigure with residual shared timing entry");

    const std::vector<uint8_t> payload =
        MakeBytes(16, Pattern::SEEDED_RANDOM);
    const P2pBuiltPayload built = BuildPayloadForTest(first_flow, payload);
    const sc_bv<128> business_bytes = built.fragments.front().data_;
    Shared::UpdateNetworkTail(first_flow, 300);
    const P2pTimingMetadata reused =
        Shared::Consume(first_flow, 0x10001U);
    suite.Check(reused == P2pTimingMetadata{60, 70, 300} &&
                    built.fragments.front().data_ == business_bytes,
                "shared sideband carries timing without touching Msg bytes");

    Shared::Reset();
    suite.Check(Shared::Capacity() == 2 && Shared::Residual() == 0,
                "Reset clears residual while preserving configured bound");
    Shared::Publish(first_key, 1, 1, 2);
    suite.Check(Shared::Contains(first_flow),
                "Reset starts a fresh source-round history");
    Shared::Reset();
}

} // namespace

int RunP2pPayloadSelfTest() {
    TestSuite suite;
    TestFsmTransportAbi(suite);
    TestLengthsAndPatterns(suite);
    TestUnalignedSlice(suite);
    TestChecksum(suite);
    TestStrictRequestValidation(suite);
    TestStrictFragmentValidation(suite);
    TestSequenceTailAndIdentity(suite);
    TestReassemblerBounds(suite);
    TestTimingSideband(suite);
    TestSharedTimingSideband(suite);
    return suite.Finish();
}

int RunP2pSharedTimingSidebandSelfTest() {
    TestSuite suite;
    TestSharedTimingSideband(suite);
    return suite.Finish();
}
