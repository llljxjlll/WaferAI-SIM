#include "dte/p2p_session_runtime.h"
#include "dte/p2p_session_runtime_selftest.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "prims/dte_endpoint_prims.h"
#pragma GCC diagnostic pop

#include "utils/msg_utils.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr DteEndpointCompletion kAsync =
    static_cast<DteEndpointCompletion>(0);
constexpr DteEndpointCompletion kSync =
    static_cast<DteEndpointCompletion>(1);

class Checks {
public:
    void Check(bool condition, const std::string &message) {
        ++checks_;
        if (!condition) {
            ++failures_;
            std::cerr << "[P2P Session] FAIL: " << message << '\n';
        }
    }

    template <typename Fn>
    void Reject(Fn &&fn, const std::string &message) {
        ++checks_;
        try {
            fn();
        } catch (const std::exception &) {
            return;
        } catch (...) {
            return;
        }
        ++failures_;
        std::cerr << "[P2P Session] FAIL: expected rejection: " << message
                  << '\n';
    }

    int Finish() const {
        if (failures_ == 0)
            std::cout << "[P2P Session] PASS (" << checks_ << " checks)\n";
        return failures_;
    }

private:
    int checks_ = 0;
    int failures_ = 0;
};

bool SameResidual(const P2pEndpointResidual &a,
                  const P2pEndpointResidual &b) {
    return a.sessions == b.sessions &&
           a.async_tokens == b.async_tokens &&
           a.allocated_transport_tags == b.allocated_transport_tags &&
           a.inbound_flows == b.inbound_flows &&
           a.pending_requests == b.pending_requests &&
           a.inflight_reassemblies == b.inflight_reassemblies &&
           a.completed_unposted == b.completed_unposted &&
           a.commit_ready == b.commit_ready &&
           a.completed_sessions == b.completed_sessions &&
           a.tx_awaiting_admission == b.tx_awaiting_admission &&
           a.tx_awaiting_ack == b.tx_awaiting_ack &&
           a.tx_awaiting_local_retire == b.tx_awaiting_local_retire &&
           a.reserved_rx_bytes == b.reserved_rx_bytes &&
           a.early_data_flows == b.early_data_flows &&
           a.early_data_fragments == b.early_data_fragments &&
           a.early_data_bytes == b.early_data_bytes;
}

P2pEndpointSessionSpec AsyncSpec(uint32_t fsm_id, uint32_t token,
                                 uint64_t length, uint16_t peer) {
    return {kAsync, fsm_id, token, length, peer};
}

P2pEndpointSessionSpec SyncSpec(uint32_t fsm_id, uint64_t length,
                                uint16_t peer) {
    return {kSync, fsm_id, 0, length, peer};
}

std::vector<uint8_t> Pattern(size_t size, uint8_t seed) {
    std::vector<uint8_t> bytes(size);
    for (size_t i = 0; i < size; ++i)
        bytes[i] = static_cast<uint8_t>(seed + i * 37U + (i >> 2));
    return bytes;
}

Msg WireRoundTrip(const Msg &message) {
    return DeserializeMsg(SerializeMsg(message));
}

std::optional<P2pRxDelivery>
DeliverData(const P2pTxIssue &issue, P2pEndpointSessionRuntime &receiver) {
    std::optional<P2pRxDelivery> delivery;
    for (const Msg &fragment : issue.messages.fragments) {
        std::optional<P2pRxDelivery> current =
            receiver.ReceiveData(WireRoundTrip(fragment));
        if (current.has_value())
            delivery = std::move(current);
    }
    return delivery;
}

Msg AdmissionAckFor(const P2pTxIssue &issue) {
    const P2pPayloadDeclaration declaration =
        ParseP2pPayloadRequest(issue.messages.request);
    return MakeP2pAdmissionAck(declaration.flow, declaration.fsm_id);
}

void Admit(const P2pTxIssue &issue, P2pEndpointSessionRuntime &sender) {
    sender.ReceiveAdmissionAck(WireRoundTrip(AdmissionAckFor(issue)));
}

void TestAckAbi(Checks &checks) {
    const P2pFlowKey flow{4, 9, 17, 0};
    const Msg ack = MakeP2pCompletionAck(flow, 0x10001U);
    const sc_bv<256> wire = SerializeMsg(ack);
    const P2pCompletionAck parsed =
        ParseP2pCompletionAck(DeserializeMsg(wire));
    sc_bv<256> ack_without_endpoint = wire;
    ack_without_endpoint[255] = false;
    checks.Reject(
        [&] {
            (void)ParseP2pCompletionAck(
                DeserializeMsg(ack_without_endpoint));
        },
        "endpoint ACK missing bit255 discriminator");
    checks.Check(parsed.flow == flow && parsed.fsm_id == 0x10001U,
                 "ACK preserves transport tag and full 32-bit fsm_id");
    checks.Check(!ack.data_.range(127, 32).or_reduce() &&
                     ack.data_.range(31, 0).to_uint64() == 0x10001U,
                 "ACK uses only data_[31:0] for endpoint fsm_id");
    checks.Check(ack.p2p_endpoint_ &&
                     ack.msg_type_ == MSG_TYPE::ACK && ack.event_tag_ == 0 &&
                     ack.offset_ == P2P_ENDPOINT_MSG_MARKER &&
                     ack.roofline_packets_ == 0,
                 "completion ACK has dedicated endpoint marker, not EVENT");
    const Msg admission = MakeP2pAdmissionAck(flow, 0x10001U);
    const sc_bv<256> admission_wire = SerializeMsg(admission);
    const P2pAdmissionAck admission_parsed =
        ParseP2pAdmissionAck(DeserializeMsg(admission_wire));
    checks.Check(admission.p2p_endpoint_ && admission.subflow_ == 1 &&
                     admission_parsed.flow == flow &&
                     admission_parsed.fsm_id == 0x10001U &&
                     !admission.data_.range(127, 32).or_reduce(),
                 "admission ACK kind preserves full fsm/tag and canonical payload");
    checks.Check(wire[255].to_bool() && admission_wire[255].to_bool(),
                 "both endpoint ACK kinds carry the bit255 discriminator");
    checks.Reject([&] { (void)ParseP2pCompletionAck(admission); },
                  "admission ACK cannot decode as completion ACK");
    checks.Reject([&] { (void)ParseP2pAdmissionAck(ack); },
                  "completion ACK cannot decode as admission ACK");
    Msg legacy_marker = ack;
    legacy_marker.offset_ = 0;
    checks.Reject([&] { (void)ParseP2pCompletionAck(legacy_marker); },
                  "legacy offset-zero ACK");
    Msg corrupt_marker = ack;
    corrupt_marker.offset_ = P2P_ENDPOINT_MSG_MARKER ^ 1U;
    checks.Reject([&] { (void)ParseP2pCompletionAck(corrupt_marker); },
                  "ACK endpoint marker bitflip");
    Msg pinned_ack = ack;
    pinned_ack.exit_port_ = static_cast<int>(UINT16_MAX) - 1;
    const Msg pinned_ack_wire = WireRoundTrip(pinned_ack);
    const P2pCompletionAck pinned_parsed =
        ParseP2pCompletionAck(pinned_ack_wire);
    checks.Check(pinned_ack_wire.exit_port_ ==
                         static_cast<int>(UINT16_MAX) - 1 &&
                     pinned_parsed.flow == flow &&
                     pinned_parsed.fsm_id == 0x10001U,
                 "pinned ACK round trip preserves routing and endpoint identity");

    Msg legacy(MSG_TYPE::ACK, 9, 17, 4);
    legacy.offset_ = P2P_ENDPOINT_MSG_MARKER;
    legacy.data_ = 0;
    checks.Check(SerializeMsg(legacy).or_reduce(),
                 "legacy ACK remains serializable without codec changes");
    checks.Reject([&] { (void)ParseP2pCompletionAck(legacy); },
                  "legacy ACK marker collision is not mistaken for endpoint ACK");
    legacy.subflow_ = 1;
    checks.Reject([&] { (void)ParseP2pAdmissionAck(legacy); },
                  "legacy ACK collision is not mistaken for admission ACK");
    checks.Reject([&] { (void)MakeP2pAdmissionAck(flow, 0); },
                  "admission ACK zero fsm_id");
    checks.Reject([&] { (void)MakeP2pCompletionAck(flow, 0); },
                  "ACK zero fsm_id");
    P2pFlowKey zero_tag = flow;
    zero_tag.transport_tag = 0;
    checks.Reject([&] { (void)MakeP2pCompletionAck(zero_tag, 1); },
                  "ACK zero transport tag");
    Msg invalid_exit = ack;
    invalid_exit.exit_port_ = -2;
    checks.Reject([&] { (void)ParseP2pCompletionAck(invalid_exit); },
                  "ACK exit below unpinned sentinel");
    invalid_exit.exit_port_ = static_cast<int>(UINT16_MAX);
    checks.Reject([&] { (void)ParseP2pCompletionAck(invalid_exit); },
                  "ACK exit exceeds decoded wire range");
    Msg reserved = ack;
    reserved.data_.range(63, 32) = sc_bv<32>(1);
    checks.Reject([&] { (void)ParseP2pCompletionAck(reserved); },
                  "ACK reserved bits");
}

void TestAdmissionStateMachine(Checks &checks) {
    const std::vector<uint8_t> bytes = Pattern(16, 0x31);
    P2pEndpointSessionRuntime sender(7, 2, 64);
    const P2pTxIssue issue = sender.IssueSend(
        AsyncSpec(0x10001U, 0x20001U, bytes.size(), 8), bytes);
    const P2pPayloadDeclaration declaration =
        ParseP2pPayloadRequest(issue.messages.request);
    const Msg completion = MakeP2pCompletionAck(
        declaration.flow, declaration.fsm_id);
    const Msg admission = AdmissionAckFor(issue);

    sender.MarkRequestSent(issue.handle);
    checks.Check(sender.Poll(0x20001U) ==
                         P2pEndpointPhase::REQUEST_SENT &&
                     sender.Residual().tx_awaiting_admission == 1,
                 "REQUEST send waits explicitly for admission CTS");
    checks.Reject([&] { sender.CompleteSend(issue.handle); },
                  "DATA/local completion before admission");
    checks.Reject([&] { sender.ReceiveAck(completion); },
                  "completion ACK before admission");
    Msg wrong_fsm = admission;
    wrong_fsm.data_.range(31, 0) = sc_bv<32>(0x10002U);
    checks.Reject([&] { sender.ReceiveAdmissionAck(wrong_fsm); },
                  "admission ACK unknown fsm high bits");
    Msg wrong_tag = admission;
    ++wrong_tag.tag_id_;
    checks.Reject([&] { sender.ReceiveAdmissionAck(wrong_tag); },
                  "admission ACK tag conflict");
    checks.Check(sender.Residual().tx_awaiting_admission == 1,
                 "invalid admission ACKs preserve REQUEST_SENT state");

    sender.ReceiveAdmissionAck(WireRoundTrip(admission));
    checks.Check(sender.IsAdmitted(issue.handle) &&
                     sender.Residual().tx_awaiting_admission == 0 &&
                     sender.Residual().tx_awaiting_ack == 1,
                 "matching admission ACK enables DATA exactly once");
    checks.Reject([&] { sender.ReceiveAdmissionAck(admission); },
                  "duplicate admission ACK");
    sender.CompleteSend(issue.handle);
    checks.Check(sender.TryWait(0x20001U) && sender.HasFsm(0x10001U) &&
                     sender.HasHandle(issue.handle),
                 "local complete retains session for completion ACK");
    checks.Reject([&] { sender.ReceiveAdmissionAck(admission); },
                  "stale admission ACK after local completion");
    sender.ReceiveAck(completion);
    checks.Check(sender.Drained() && !sender.HasHandle(issue.handle),
                 "admission and completion ACK phases drain independently");
    checks.Reject([&] { sender.ReceiveAdmissionAck(admission); },
                  "admission ACK after retirement");
}

void TestRecvBeforeSendSync(Checks &checks) {
    const std::vector<uint8_t> bytes = Pattern(129, 3);
    P2pEndpointSessionRuntime sender(0, 4, 512);
    P2pEndpointSessionRuntime receiver(1, 4, 512);
    const P2pRxPostResult post =
        receiver.PostReceive(SyncSpec(0x10001U, bytes.size(), 0));
    checks.Check(!post.ready.has_value() &&
                     !receiver.TryRetireSync(post.handle),
                 "recv-before-send posts a blocking descriptor");

    const P2pTxIssue issue =
        sender.IssueSend(SyncSpec(0x10001U, bytes.size(), 1), bytes);
    checks.Check(issue.transport_tag != 0 &&
                     ParseP2pPayloadRequest(issue.messages.request).fsm_id ==
                         0x10001U,
                 "SYNC issue allocates a non-zero tag without fsm truncation");
    checks.Check(!sender.TryRetireSync(issue.handle),
                 "queued SYNC send is not locally complete");
    sender.MarkRequestSent(issue.handle);
    receiver.ReceiveRequest(WireRoundTrip(issue.messages.request));
    Admit(issue, sender);
    const std::optional<P2pRxDelivery> delivery =
        DeliverData(issue, receiver);
    checks.Check(delivery && delivery->bytes == bytes,
                 "recv-before-send delivers exact committed bytes");
    const Msg ack = receiver.CompleteReceive(delivery->handle);
    checks.Reject([&] { (void)receiver.CompleteReceive(delivery->handle); },
                  "duplicate RX completion");
    checks.Check(receiver.TryRetireSync(post.handle) && receiver.Drained(),
                 "SYNC RX retires only after visible byte commit");

    sender.CompleteSend(issue.handle);
    checks.Reject([&] { sender.CompleteSend(issue.handle); },
                  "duplicate TX local completion");
    checks.Check(sender.TryRetireSync(issue.handle),
                 "SYNC TX local completion retires instruction semantics");
    checks.Reject([&] { (void)sender.TryRetireSync(issue.handle); },
                  "duplicate local SYNC retire before ACK");
    const P2pEndpointResidual waiting = sender.Residual();
    checks.Check(sender.HasFsm(0x10001U) &&
                     waiting.allocated_transport_tags == 1 &&
                     waiting.tx_awaiting_ack == 1 &&
                     waiting.tx_awaiting_local_retire == 0,
                 "local SYNC retire retains fsm and tag until remote ACK");
    sender.ReceiveAck(WireRoundTrip(ack));
    checks.Check(sender.Drained(),
                 "remote ACK releases retained SYNC transport resources");
    checks.Reject([&] { sender.ReceiveAck(ack); }, "duplicate retired ACK");
}

void TestSendBeforeRecvAsync(Checks &checks) {
    const std::vector<uint8_t> bytes = Pattern(17, 91);
    P2pEndpointSessionRuntime sender(2, 4, 128);
    P2pEndpointSessionRuntime receiver(3, 4, 128);
    const P2pTxIssue issue =
        sender.IssueSend(AsyncSpec(7, 101, bytes.size(), 3), bytes);
    sender.MarkRequestSent(issue.handle);
    receiver.ReceiveRequest(WireRoundTrip(issue.messages.request));
    Admit(issue, sender);
    checks.Check(!DeliverData(issue, receiver).has_value(),
                 "send-before-recv keeps completed bytes private");
    const P2pEndpointResidual pending = receiver.Residual();
    checks.Check(pending.pending_requests == 1 &&
                     pending.completed_unposted == 1 &&
                     pending.reserved_rx_bytes == bytes.size(),
                 "send-before-recv accounts completed pending payload");

    const P2pRxPostResult post =
        receiver.PostReceive(AsyncSpec(7, 201, bytes.size(), 2));
    checks.Check(post.ready && post.ready->bytes == bytes &&
                     receiver.ReservedRxBytes() == 0,
                 "late receive atomically exposes the full payload");
    checks.Check(receiver.Poll(201) == P2pEndpointPhase::COMMIT_READY &&
                     !receiver.TryWait(201),
                 "ASYNC RX WAIT stays pending until SRAM commit completion");
    const Msg ack = receiver.CompleteReceive(post.ready->handle);
    checks.Check(receiver.Poll(201) == P2pEndpointPhase::COMPLETE &&
                     receiver.TryWait(201) && receiver.Drained(),
                 "ASYNC RX WAIT consumes only completed token");
    checks.Reject([&] { (void)receiver.TryWait(201); },
                  "duplicate RX WAIT");

    sender.CompleteSend(issue.handle);
    checks.Check(sender.TryWait(101) && !sender.HasToken(101) &&
                     sender.HasFsm(7),
                 "ASYNC local WAIT returns before ACK but retains fsm");
    checks.Reject([&] { (void)sender.TryWait(101); },
                  "duplicate local TX WAIT");
    sender.ReceiveAck(WireRoundTrip(ack));
    checks.Check(sender.Drained(),
                 "late ACK finishes async transport resource release");
}

void TestAckBeforeLocalRetire(Checks &checks) {
    const std::vector<uint8_t> bytes = Pattern(16, 12);
    P2pEndpointSessionRuntime sender(5, 2, 64);
    P2pEndpointSessionRuntime receiver(6, 2, 64);
    const P2pTxIssue issue =
        sender.IssueSend(SyncSpec(8, bytes.size(), 6), bytes);
    const P2pRxPostResult post =
        receiver.PostReceive(SyncSpec(8, bytes.size(), 5));
    sender.MarkRequestSent(issue.handle);
    receiver.ReceiveRequest(WireRoundTrip(issue.messages.request));
    Admit(issue, sender);
    const std::optional<P2pRxDelivery> delivery = DeliverData(issue, receiver);
    checks.Check(delivery.has_value(), "ACK-first flow reaches RX commit");
    const Msg ack = receiver.CompleteReceive(delivery->handle);
    (void)receiver.TryRetireSync(post.handle);

    sender.ReceiveAck(WireRoundTrip(ack));
    P2pEndpointResidual residual = sender.Residual();
    checks.Check(sender.HasFsm(8) && residual.tx_awaiting_ack == 0 &&
                     residual.tx_awaiting_local_retire == 1,
                 "early ACK retains transport until local completion retires");
    checks.Reject([&] { sender.ReceiveAck(ack); },
                  "duplicate ACK before local retire");
    sender.CompleteSend(issue.handle);
    checks.Check(sender.TryRetireSync(issue.handle) && sender.Drained(),
                 "local retire after ACK releases transport atomically");
}

void TestFullFsmCollisionIsolation(Checks &checks) {
    const std::vector<uint8_t> low_bytes = Pattern(33, 4);
    const std::vector<uint8_t> high_bytes = Pattern(33, 209);
    P2pEndpointSessionRuntime sender(10, 4, 256, 4);
    P2pEndpointSessionRuntime receiver(11, 4, 256, 4);
    const P2pTxIssue low =
        sender.IssueSend(AsyncSpec(1, 11, low_bytes.size(), 11), low_bytes);
    const P2pTxIssue high = sender.IssueSend(
        AsyncSpec(0x10001U, 12, high_bytes.size(), 11), high_bytes);
    checks.Check(low.transport_tag != high.transport_tag,
                 "fsm 0x1 and 0x10001 receive distinct transport tags");
    const P2pRxPostResult low_post =
        receiver.PostReceive(AsyncSpec(1, 21, low_bytes.size(), 10));
    const P2pRxPostResult high_post = receiver.PostReceive(
        AsyncSpec(0x10001U, 22, high_bytes.size(), 10));
    sender.MarkRequestSent(low.handle);
    sender.MarkRequestSent(high.handle);
    receiver.ReceiveRequest(WireRoundTrip(high.messages.request));
    receiver.ReceiveRequest(WireRoundTrip(low.messages.request));
    Admit(high, sender);
    Admit(low, sender);

    std::optional<P2pRxDelivery> low_delivery;
    std::optional<P2pRxDelivery> high_delivery;
    for (size_t index = 0; index < low.messages.fragments.size(); ++index) {
        std::optional<P2pRxDelivery> one =
            receiver.ReceiveData(WireRoundTrip(low.messages.fragments[index]));
        std::optional<P2pRxDelivery> two = receiver.ReceiveData(
            WireRoundTrip(high.messages.fragments[index]));
        if (one)
            low_delivery = std::move(one);
        if (two)
            high_delivery = std::move(two);
    }
    checks.Check(low_delivery && low_delivery->bytes == low_bytes &&
                     high_delivery && high_delivery->bytes == high_bytes,
                 "low16-colliding fsm sessions never cross payload bytes");

    const Msg low_ack = receiver.CompleteReceive(low_delivery->handle);
    const Msg high_ack = receiver.CompleteReceive(high_delivery->handle);
    checks.Check(receiver.TryWait(21) && receiver.TryWait(22) &&
                     receiver.Drained(),
                 "both collision-isolated RX tokens drain");
    sender.CompleteSend(low.handle);
    sender.CompleteSend(high.handle);
    checks.Check(sender.TryWait(11) && sender.TryWait(12),
                 "both collision-isolated TX tokens locally retire");
    sender.ReceiveAck(high_ack);
    sender.ReceiveAck(low_ack);
    checks.Check(sender.Drained(),
                 "out-of-order ACKs match full fsm and transport tag");
    (void)low_post;
    (void)high_post;
}

void TestMatchAndOrderingFailures(Checks &checks) {
    P2pEndpointSessionRuntime receiver(20, 4, 128);
    const P2pRxPostResult post =
        receiver.PostReceive(AsyncSpec(77, 701, 32, 19));
    const P2pBuiltPayload wrong_source = BuildP2pPayload(
        P2pFlowKey{18, 20, 1, 0}, 77, Pattern(32, 1));
    checks.Reject([&] { receiver.ReceiveRequest(wrong_source.request); },
                  "REQUEST source mismatch");
    const P2pBuiltPayload wrong_length = BuildP2pPayload(
        P2pFlowKey{19, 20, 2, 0}, 77, Pattern(17, 1));
    checks.Reject([&] { receiver.ReceiveRequest(wrong_length.request); },
                  "REQUEST length mismatch");
    const P2pBuiltPayload wrong_destination = BuildP2pPayload(
        P2pFlowKey{19, 21, 3, 0}, 77, Pattern(32, 1));
    checks.Reject([&] { receiver.ReceiveRequest(wrong_destination.request); },
                  "REQUEST destination mismatch");
    checks.Check(receiver.Residual().sessions == 1 &&
                     receiver.ReservedRxBytes() == 0,
                 "descriptor mismatch failures are atomic");

    const P2pBuiltPayload valid = BuildP2pPayload(
        P2pFlowKey{19, 20, 4, 0}, 77, Pattern(32, 8));
    const P2pEndpointResidual before_early = receiver.Residual();
    checks.Reject([&] { (void)receiver.ReceiveData(valid.fragments[0]); },
                  "DATA before admission");
    checks.Check(SameResidual(before_early, receiver.Residual()),
                 "pre-admission DATA rejection is atomic and residual-free");
    (void)receiver.ReceiveRequest(valid.request);
    checks.Check(!receiver.ReceiveRequest(valid.request).has_value() &&
                     receiver.ReservedRxBytes() == 32 &&
                     receiver.Residual().inbound_flows == 1,
                 "identical REQUEST retransmission is an exactly-once no-op");
    const P2pEndpointResidual before_conflict = receiver.Residual();
    Msg conflicting = valid.request;
    conflicting.data_.range(31, 0) = sc_bv<32>(
        conflicting.data_.range(31, 0).to_uint64() ^ 1U);
    checks.Reject([&] { (void)receiver.ReceiveRequest(conflicting); },
                  "conflicting duplicate REQUEST flow");
    checks.Check(SameResidual(before_conflict, receiver.Residual()),
                 "conflicting duplicate REQUEST preserves active flow atomically");
    const P2pBuiltPayload duplicate_fsm = BuildP2pPayload(
        P2pFlowKey{19, 20, 5, 0}, 77, Pattern(32, 8));
    checks.Reject([&] { receiver.ReceiveRequest(duplicate_fsm.request); },
                  "duplicate REQUEST full fsm_id");
    checks.Check(SameResidual(before_conflict, receiver.Residual()),
                 "conflicting duplicate fsm preserves active flow atomically");
    checks.Check(!receiver.ReceiveData(valid.fragments[0]).has_value(),
                 "first ordered DATA remains uncommitted");
    const std::optional<P2pRxDelivery> delivery =
        receiver.ReceiveData(valid.fragments[1]);
    checks.Check(delivery && delivery->bytes == Pattern(32, 8),
                 "flow recovers after non-terminal duplicate rejection");
    (void)receiver.CompleteReceive(delivery->handle);
    checks.Check(receiver.TryWait(701) && receiver.Drained(),
                 "matched flow drains after failures");
    (void)post;
}

void TestChecksumFailureCleanup(Checks &checks) {
    P2pEndpointSessionRuntime receiver(30, 2, 64);
    (void)receiver.PostReceive(AsyncSpec(90, 901, 17, 29));
    P2pBuiltPayload payload = BuildP2pPayload(
        P2pFlowKey{29, 30, 1, 0}, 90, Pattern(17, 5));
    receiver.ReceiveRequest(payload.request);
    (void)receiver.ReceiveData(payload.fragments[0]);
    const uint8_t byte = static_cast<uint8_t>(
        payload.fragments[1].data_.range(7, 0).to_uint64());
    payload.fragments[1].data_.range(7, 0) = sc_bv<8>(byte ^ 1U);
    checks.Reject([&] { receiver.ReceiveData(payload.fragments[1]); },
                  "terminal checksum failure");
    checks.Check(receiver.Drained() && !receiver.HasToken(901) &&
                     !receiver.HasFsm(90),
                 "terminal checksum failure atomically reclaims all state");
}

void TestCancelAndEarlyReuse(Checks &checks) {
    const std::vector<uint8_t> bytes = Pattern(16, 44);
    P2pEndpointSessionRuntime sender(40, 4, 128, 3);
    P2pTxIssue queued =
        sender.IssueSend(AsyncSpec(1, 1001, bytes.size(), 41), bytes);
    checks.Check(queued.transport_tag == 1,
                 "first TX receives lifetime transport tag one");
    sender.Cancel(1001);
    checks.Check(sender.Drained(),
                 "CANCEL reclaims a queued TX without reusing its tag");
    checks.Reject([&] { sender.CompleteSend(queued.handle); },
                  "completion after queued TX cancel");
    checks.Reject([&] { sender.Cancel(1001); }, "duplicate TX cancel");

    P2pTxIssue second =
        sender.IssueSend(AsyncSpec(2, 1002, bytes.size(), 41), bytes);
    checks.Check(second.transport_tag == 2,
                 "queued cancel permanently consumes its transport tag");
    sender.Cancel(1002);

    P2pEndpointSessionRuntime receiver(41, 4, 128, 3);
    P2pTxIssue active =
        sender.IssueSend(AsyncSpec(3, 1003, bytes.size(), 41), bytes);
    checks.Check(active.transport_tag == 3,
                 "transport tags advance monotonically without wrap");
    const P2pRxPostResult post =
        receiver.PostReceive(AsyncSpec(3, 2001, bytes.size(), 40));
    sender.MarkRequestSent(active.handle);
    (void)receiver.ReceiveRequest(active.messages.request);
    Admit(active, sender);
    checks.Reject([&] { sender.Cancel(1003); },
                  "CANCEL after REQUEST send");
    const std::optional<P2pRxDelivery> delivery =
        DeliverData(active, receiver);
    const Msg ack = receiver.CompleteReceive(delivery->handle);
    checks.Check(receiver.TryWait(2001), "active RX completes before ACK");
    sender.CompleteSend(active.handle);
    checks.Check(sender.TryWait(1003), "active TX locally completes");
    sender.ReceiveAck(ack);
    checks.Check(sender.Drained() && receiver.Drained(),
                 "completed lifetime-tag transaction drains active state");
    checks.Reject(
        [&] {
            (void)sender.IssueSend(
                AsyncSpec(4, 1004, bytes.size(), 41), bytes);
        },
        "lifetime transport tag exhaustion fails instead of wrapping");
    (void)post;

    P2pEndpointSessionRuntime rx_unmatched(42, 2, 64);
    const P2pRxPostResult unmatched =
        rx_unmatched.PostReceive(AsyncSpec(5, 2005, bytes.size(), 40));
    checks.Reject([&] { rx_unmatched.Cancel(2005); },
                  "RX CANCEL rejects an unmatched posted descriptor");
    checks.Check(rx_unmatched.AbortReceiveFsm(5) && rx_unmatched.Drained(),
                 "explicit failure teardown reclaims unmatched RX");
    (void)unmatched;

    P2pEndpointSessionRuntime rx_active(43, 2, 64);
    (void)rx_active.PostReceive(AsyncSpec(6, 2006, bytes.size(), 40));
    const P2pBuiltPayload active_rx = BuildP2pPayload(
        P2pFlowKey{40, 43, 1, 0}, 6, bytes);
    (void)rx_active.ReceiveRequest(active_rx.request);
    checks.Reject([&] { rx_active.Cancel(2006); },
                  "RX CANCEL rejects an admitted REQUEST");
    checks.Check(rx_active.AbortInbound(
                     ParseP2pPayloadRequest(active_rx.request).flow) &&
                     rx_active.Drained(),
                 "failure teardown reclaims admitted RX after cancel reject");

    P2pEndpointSessionRuntime rx_commit(44, 2, 64);
    const P2pRxPostResult commit_post =
        rx_commit.PostReceive(AsyncSpec(7, 2007, bytes.size(), 40));
    const P2pBuiltPayload commit_rx = BuildP2pPayload(
        P2pFlowKey{40, 44, 1, 0}, 7, bytes);
    (void)rx_commit.ReceiveRequest(commit_rx.request);
    const std::optional<P2pRxDelivery> commit_delivery =
        rx_commit.ReceiveData(commit_rx.fragments.front());
    checks.Reject([&] { rx_commit.Cancel(2007); },
                  "RX CANCEL rejects a commit-ready receive");
    (void)rx_commit.CompleteReceive(commit_delivery->handle);
    checks.Check(rx_commit.TryWait(2007) && rx_commit.Drained(),
                 "commit-ready RX completes normally after cancel reject");
    (void)commit_post;
}

void TestTokenReuseAckAba(Checks &checks) {
    const std::vector<uint8_t> bytes = Pattern(16, 73);
    P2pEndpointSessionRuntime sender(42, 4, 64, 2);
    P2pTxIssue old =
        sender.IssueSend(AsyncSpec(100, 7001, bytes.size(), 43), bytes);
    sender.MarkRequestSent(old.handle);
    Admit(old, sender);
    sender.CompleteSend(old.handle);
    checks.Check(sender.TryWait(7001) && !sender.HasToken(7001) &&
                     sender.HasFsm(100),
                 "old TX WAIT releases token control before ACK");

    P2pTxIssue replacement =
        sender.IssueSend(AsyncSpec(200, 7001, bytes.size(), 43), bytes);
    checks.Check(sender.HasToken(7001) && sender.HasFsm(200),
                 "different fsm reuses locally completed token");
    const P2pFlowKey old_flow =
        ParseP2pPayloadRequest(old.messages.request).flow;
    sender.ReceiveAck(MakeP2pCompletionAck(old_flow, old.handle.fsm_id));
    checks.Check(!sender.HasFsm(100) && sender.HasFsm(200) &&
                     sender.HasToken(7001) &&
                     sender.Poll(7001) == P2pEndpointPhase::QUEUED,
                 "late old ACK compare-and-erase preserves new token mapping");
    const P2pEndpointResidual residual = sender.Residual();
    checks.Check(residual.sessions == 1 && residual.async_tokens == 1 &&
                     residual.allocated_transport_tags == 1,
                 "late old ACK retires only old transport resources");
    sender.Cancel(7001);
    checks.Check(sender.Drained(), "ABA replacement cancel drains runtime");
    (void)replacement;
}

void TestDirectContractValidation(Checks &checks) {
    P2pEndpointSessionRuntime runtime(
        60, 4, static_cast<size_t>(kDteEndpointP2pMaxBytes));

    const P2pRxPostResult maximum = runtime.PostReceive(
        AsyncSpec(1, 101, kDteEndpointP2pMaxBytes, 61));
    checks.Check(runtime.HasToken(101),
                 "direct session spec accepts transport maximum");
    checks.Check(runtime.AbortReceiveFsm(1) && runtime.Drained(),
                 "maximum direct session spec tears down explicitly");

    const P2pEndpointSessionSpec too_large =
        AsyncSpec(2, 102, kDteEndpointP2pMaxBytes + 1, 61);
    checks.Reject([&] { (void)runtime.PostReceive(too_large); },
                  "direct receive spec transport max+1");
    checks.Reject(
        [&] { (void)runtime.IssueSend(too_large, std::vector<uint8_t>{}); },
        "direct send spec transport max+1");

    Dte_send_endpoint_prim send;
    send.fsm_id = 3;
    send.token = 103;
    send.length_bytes = 1;
    send.peer_core = 61;
    send.source_space = DteEndpointSourceSpace::HBM;
    send.source.absolute_address_bytes = UINT64_MAX;
    const P2pTxIssue accepted = runtime.IssueSend(send, Pattern(1, 9));
    checks.Check(accepted.messages.fragments.size() == 1 &&
                     runtime.HasToken(103),
                 "direct send Prim accepts HBM absolute address");
    runtime.Cancel(103);

    send.fsm_id = 4;
    send.token = 104;
    send.source_space = DteEndpointSourceSpace::HBM;
    send.source.kind = DteEndpointAddressKind::REGION;
    send.source.absolute_address_bytes = 0;
    send.source.region = "not-hbm";
    checks.Reject([&] { (void)runtime.IssueSend(send, Pattern(1, 10)); },
                  "direct send Prim HBM region");
    send.source.kind = DteEndpointAddressKind::ABSOLUTE;
    send.source.region.clear();
    send.source_space = static_cast<DteEndpointSourceSpace>(2);
    checks.Reject([&] { (void)runtime.IssueSend(send, Pattern(1, 11)); },
                  "direct send Prim source_space enum max+1");
    send.source_space = DteEndpointSourceSpace::SRAM;
    send.length_bytes = kDteEndpointP2pMaxBytes + 1;
    checks.Reject(
        [&] { (void)runtime.IssueSend(send, std::vector<uint8_t>{}); },
        "direct send Prim transport max+1");

    Dte_recv_endpoint_prim receive;
    receive.fsm_id = 5;
    receive.token = 105;
    receive.length_bytes = kDteEndpointP2pMaxBytes + 1;
    receive.peer_core = 61;
    checks.Reject([&] { (void)runtime.PostReceive(receive); },
                  "direct receive Prim transport max+1");
    checks.Check(runtime.Drained(),
                 "direct Prim/spec validation failures preserve runtime");
    (void)maximum;
}

void TestAdmissionCapacityProgress(Checks &checks) {
    const std::vector<uint8_t> full = Pattern(32, 0x61);
    const std::vector<uint8_t> next = Pattern(16, 0x22);
    P2pEndpointSessionRuntime sender(70, 2, 64);
    P2pEndpointSessionRuntime sender_two(69, 1, 32);
    P2pEndpointSessionRuntime receiver(71, 2, 32);
    const P2pTxIssue a =
        sender.IssueSend(AsyncSpec(501, 1501, full.size(), 71), full);
    const P2pTxIssue b =
        sender.IssueSend(AsyncSpec(502, 1502, next.size(), 71), next);
    const P2pTxIssue c =
        sender_two.IssueSend(AsyncSpec(504, 1504, next.size(), 71), next);
    const P2pTxIssue reverse = receiver.IssueSend(
        AsyncSpec(503, 2503, 1, 72), Pattern(1, 0x44));
    sender.MarkRequestSent(a.handle);
    sender.MarkRequestSent(b.handle);
    receiver.MarkRequestSent(reverse.handle);
    sender_two.MarkRequestSent(c.handle);

    const P2pRxPostResult post_a =
        receiver.PostReceive(AsyncSpec(501, 2501, full.size(), 70));
    (void)receiver.ReceiveRequest(WireRoundTrip(a.messages.request));
    Admit(a, sender);
    const P2pPayloadDeclaration b_decl =
        ParseP2pPayloadRequest(b.messages.request);
    const P2pPayloadDeclaration c_decl =
        ParseP2pPayloadRequest(c.messages.request);
    checks.Check(receiver.ReservedRxBytes() == full.size() &&
                     !receiver.CanReceiveRequest(b_decl.flow,
                                                 b_decl.total_bytes) &&
                     !receiver.CanReceiveRequest(c_decl.flow,
                                                 c_decl.total_bytes) &&
                     sender.Residual().tx_awaiting_admission == 1 &&
                     sender_two.Residual().tx_awaiting_admission == 1,
                 "full declared A leaves multi-source REQUESTs waiting for CTS");
    Admit(reverse, receiver);
    checks.Check(receiver.IsAdmitted(reverse.handle),
                 "reverse admission ACK bypasses capacity-blocked REQUEST B");
    receiver.CompleteSend(reverse.handle);
    checks.Check(receiver.TryWait(2503),
                 "reverse local completion progresses during REQUEST HOL");
    const P2pPayloadDeclaration reverse_decl =
        ParseP2pPayloadRequest(reverse.messages.request);
    receiver.ReceiveAck(MakeP2pCompletionAck(
        reverse_decl.flow, reverse_decl.fsm_id));
    checks.Reject(
        [&] {
            (void)receiver.ReceiveData(
                WireRoundTrip(b.messages.fragments.front()));
        },
        "DATA before admission CTS");
    checks.Check(receiver.Residual().early_data_bytes == 0 &&
                     receiver.ReservedRxBytes() == full.size(),
                 "pre-admission DATA rejection is atomic and stores no early bytes");

    const std::optional<P2pRxDelivery> delivery_a =
        DeliverData(a, receiver);
    const Msg completion_a =
        receiver.CompleteReceive(delivery_a->handle);
    checks.Check(receiver.TryWait(2501) &&
                     receiver.CanReceiveRequest(b_decl.flow,
                                                b_decl.total_bytes),
                 "admitted A DATA completes and releases capacity for B");
    sender.CompleteSend(a.handle);
    checks.Check(sender.TryWait(1501),
                 "A local completion retires before completion ACK");
    sender.ReceiveAck(completion_a);

    const P2pRxPostResult post_b =
        receiver.PostReceive(AsyncSpec(502, 2502, next.size(), 70));
    const P2pRxPostResult post_c =
        receiver.PostReceive(AsyncSpec(504, 2504, next.size(), 69));
    (void)receiver.ReceiveRequest(WireRoundTrip(b.messages.request));
    (void)receiver.ReceiveRequest(WireRoundTrip(c.messages.request));
    Admit(b, sender);
    Admit(c, sender_two);
    const std::optional<P2pRxDelivery> delivery_b =
        DeliverData(b, receiver);
    const Msg completion_b =
        receiver.CompleteReceive(delivery_b->handle);
    const std::optional<P2pRxDelivery> delivery_c =
        DeliverData(c, receiver);
    const Msg completion_c =
        receiver.CompleteReceive(delivery_c->handle);
    sender.CompleteSend(b.handle);
    sender_two.CompleteSend(c.handle);
    checks.Check(receiver.TryWait(2502) && receiver.TryWait(2504) &&
                     sender.TryWait(1502) && sender_two.TryWait(1504),
                 "multi-source waiting requests progress after admission without early DATA");
    sender.ReceiveAck(completion_b);
    sender_two.ReceiveAck(completion_c);
    checks.Check(sender.Drained() && sender_two.Drained() && receiver.Drained(),
                 "multi-source admission capacity chain drains all sessions and bytes");
    (void)post_a;
    (void)post_b;
    (void)post_c;
}

void TestRequestIngressClassification(Checks &checks) {
    const P2pBuiltPayload queued_payload = BuildP2pPayload(
        P2pFlowKey{7, 8, 1, 0}, 690, Pattern(16, 0x31));
    const P2pPayloadDeclaration queued =
        ParseP2pPayloadRequest(queued_payload.request);
    const P2pPayloadDeclaration queued_exact = queued;
    const P2pPayloadDeclaration queued_conflict =
        ParseP2pPayloadRequest(
            BuildP2pPayload(queued.flow, 691, Pattern(16, 0x32)).request);
    checks.Check(
        ClassifyP2pRequestIngress(
            queued_exact, &queued, P2pRequestDisposition::NEW) ==
            P2pRequestIngressAction::DUPLICATE,
        "queued identical REQUEST ingress is idempotent");
    checks.Check(
        ClassifyP2pRequestIngress(
            queued_conflict, &queued, P2pRequestDisposition::NEW) ==
            P2pRequestIngressAction::REJECT_CONFLICT,
        "queued conflicting REQUEST is consumed without changing the queue");
    checks.Check(
        ClassifyP2pRequestIngress(
            queued, nullptr, P2pRequestDisposition::NEW) ==
            P2pRequestIngressAction::ENQUEUE,
        "new REQUEST ingress enqueues exactly once");

    P2pEndpointSessionRuntime receiver(8, 1, 16, 2, 9);
    (void)receiver.ReceiveRequest(queued_payload.request);
    const P2pEndpointResidual active_before = receiver.Residual();
    checks.Check(
        receiver.ClassifyRequest(queued) ==
                P2pRequestDisposition::ACTIVE_DUPLICATE &&
            ClassifyP2pRequestIngress(
                queued, nullptr,
                P2pRequestDisposition::ACTIVE_DUPLICATE) ==
                P2pRequestIngressAction::DUPLICATE,
        "active identical REQUEST ingress preserves the active tuple");
    checks.Check(
        receiver.ClassifyRequest(queued_conflict) ==
                P2pRequestDisposition::CONFLICT &&
            ClassifyP2pRequestIngress(
                queued_conflict, nullptr,
                P2pRequestDisposition::CONFLICT) ==
                P2pRequestIngressAction::REJECT_CONFLICT &&
            SameResidual(receiver.Residual(), active_before),
        "active conflicting REQUEST is rejected without corrupting good state");
    checks.Check(receiver.AbortInbound(queued.flow) && receiver.Drained() &&
                     receiver.ClassifyRequest(queued) ==
                         P2pRequestDisposition::TERMINAL_DUPLICATE &&
                     ClassifyP2pRequestIngress(
                         queued, nullptr,
                         P2pRequestDisposition::TERMINAL_DUPLICATE) ==
                         P2pRequestIngressAction::DUPLICATE,
                 "terminal identical REQUEST ingress remains tombstoned");
}

void TestLifetimeRequestIdentity(Checks &checks) {
    P2pEndpointSessionRuntime receiver(1, 1, 16, 4, 2);
    P2pBuiltPayload oldest;
    uint32_t ordinal = 0;
    for (uint16_t source = 0; source < 2; ++source) {
        for (uint16_t tag = 1; tag <= 4; ++tag) {
            ++ordinal;
            const uint32_t fsm_id = 700 + ordinal;
            const uint32_t token = 1700 + ordinal;
            const std::vector<uint8_t> bytes =
                Pattern(16, static_cast<uint8_t>(ordinal));
            const P2pBuiltPayload payload = BuildP2pPayload(
                P2pFlowKey{source, 1, tag, 0}, fsm_id, bytes);
            if (ordinal == 1) oldest = payload;
            (void)receiver.PostReceive(
                AsyncSpec(fsm_id, token, bytes.size(), source));
            (void)receiver.ReceiveRequest(payload.request);
            const std::optional<P2pRxDelivery> delivery =
                receiver.ReceiveData(payload.fragments.front());
            (void)receiver.CompleteReceive(delivery->handle);
            checks.Check(receiver.TryWait(token) && receiver.Drained(),
                         "lifetime seen identity excludes terminal state from drain");
        }
    }
    checks.Check(receiver.SeenRequestCount() == 8 &&
                     receiver.MaxSeenRequestIdentities() == 8,
                 "seen identity table covers the full topology by tag space");
    const P2pPayloadDeclaration oldest_declaration =
        ParseP2pPayloadRequest(oldest.request);
    checks.Check(receiver.ClassifyRequest(oldest_declaration) ==
                         P2pRequestDisposition::TERMINAL_DUPLICATE &&
                     !receiver.ReceiveRequest(oldest.request).has_value() &&
                     receiver.ReservedRxBytes() == 0 && receiver.Drained(),
                 "oldest REQUEST remains suppressed after exceeding active capacity");
    const P2pBuiltPayload conflict = BuildP2pPayload(
        oldest_declaration.flow, 999, Pattern(16, 0xee));
    const P2pPayloadDeclaration conflict_declaration =
        ParseP2pPayloadRequest(conflict.request);
    checks.Check(
        receiver.ClassifyRequest(conflict_declaration) ==
                P2pRequestDisposition::CONFLICT &&
            ClassifyP2pRequestIngress(
                conflict_declaration, nullptr,
                P2pRequestDisposition::CONFLICT) ==
                P2pRequestIngressAction::REJECT_CONFLICT,
        "terminal lifetime conflict is a recoverable ingress rejection");
    checks.Reject(
        [&] { (void)receiver.ReceiveRequest(conflict.request); },
        "runtime direct receive still rejects conflicting declaration");
    checks.Check(receiver.SeenRequestCount() == 8 && receiver.Drained(),
                 "conflicting lifetime duplicate preserves terminal table atomically");
    const P2pBuiltPayload outside_topology = BuildP2pPayload(
        P2pFlowKey{2, 1, 1, 0}, 1000, Pattern(16, 0xab));
    checks.Reject(
        [&] {
            (void)receiver.ClassifyRequest(
                ParseP2pPayloadRequest(outside_topology.request));
        },
        "REQUEST source outside fixed topology seen table");
}

void TestFailureAbortTransitions(Checks &checks) {
    const std::vector<uint8_t> bytes = Pattern(16, 0x42);
    P2pEndpointSessionRuntime sender(80, 2, 64, 2);
    P2pTxIssue tx =
        sender.IssueSend(AsyncSpec(601, 1601, bytes.size(), 81), bytes);
    sender.MarkRequestSent(tx.handle);
    checks.Check(sender.Abort(tx.handle) && sender.Drained(),
                 "active TX failure abort releases session, token, and tag");
    checks.Check(!sender.Abort(tx.handle),
                 "stale TX failure handle cannot erase a replacement");

    P2pEndpointSessionRuntime admission_failure(85, 2, 64);
    P2pTxIssue waiting = admission_failure.IssueSend(
        AsyncSpec(607, 1607, bytes.size(), 86), bytes);
    admission_failure.MarkRequestSent(waiting.handle);
    const P2pAdmissionAck failed_admission =
        ParseP2pAdmissionAck(AdmissionAckFor(waiting));
    checks.Check(admission_failure.AbortAdmissionAck(failed_admission) &&
                     admission_failure.Drained(),
                 "admission ACK receive failure releases TX fsm/token/tag");
    checks.Check(!admission_failure.AbortAdmissionAck(failed_admission),
                 "stale admission failure cannot erase a replacement");

    P2pEndpointSessionRuntime receiver(81, 2, 64);
    const P2pRxPostResult rx =
        receiver.PostReceive(AsyncSpec(602, 1602, bytes.size(), 80));
    const P2pBuiltPayload incoming = BuildP2pPayload(
        P2pFlowKey{80, 81, 2, 0}, 602, bytes);
    (void)receiver.ReceiveRequest(incoming.request);
    checks.Check(receiver.AbortInbound(
                     ParseP2pPayloadRequest(incoming.request).flow) &&
                     receiver.Drained(),
                 "RX failure abort releases reassembly, session, and token");
    (void)rx;

    P2pEndpointSessionRuntime mismatch(82, 2, 64);
    (void)mismatch.PostReceive(AsyncSpec(604, 1604, bytes.size(), 80));
    const P2pBuiltPayload wrong_descriptor = BuildP2pPayload(
        P2pFlowKey{79, 82, 3, 0}, 604, bytes);
    checks.Reject(
        [&] { (void)mismatch.ReceiveRequest(wrong_descriptor.request); },
        "posted receive descriptor mismatch");
    checks.Check(mismatch.AbortReceiveFsm(604) && mismatch.Drained(),
                 "descriptor mismatch aborts the posted generation by fsm");

    P2pEndpointSessionRuntime request_first(87, 2, 64);
    const P2pBuiltPayload pending_mismatch = BuildP2pPayload(
        P2pFlowKey{86, 87, 1, 0}, 608, bytes);
    (void)request_first.ReceiveRequest(pending_mismatch.request);
    checks.Check(request_first.ReservedRxBytes() == bytes.size(),
                 "REQUEST-first admission reserves before PostReceive");
    checks.Reject(
        [&] {
            (void)request_first.PostReceive(
                AsyncSpec(608, 1608, bytes.size(), 85));
        },
        "REQUEST-first wrong PostReceive descriptor");
    const std::optional<P2pFlowKey> aborted_mismatch =
        request_first.AbortInboundFsm(608);
    checks.Check(aborted_mismatch.has_value() &&
                     *aborted_mismatch ==
                         ParseP2pPayloadRequest(pending_mismatch.request).flow &&
                     request_first.Drained() &&
                     request_first.SeenRequestCount() == 1 &&
                     request_first.ClassifyRequest(
                         ParseP2pPayloadRequest(pending_mismatch.request)) ==
                         P2pRequestDisposition::TERMINAL_DUPLICATE,
                 "PostReceive mismatch aborts reservation but tombstones CTS flow");

    P2pEndpointSessionRuntime malformed_admission(88, 2, 32, 2);
    P2pTxIssue admission_tx = malformed_admission.IssueSend(
        AsyncSpec(609, 1609, bytes.size(), 89), bytes);
    P2pTxIssue admission_decoy = malformed_admission.IssueSend(
        AsyncSpec(611, 1611, bytes.size(), 90), bytes);
    malformed_admission.MarkRequestSent(admission_tx.handle);
    malformed_admission.MarkRequestSent(admission_decoy.handle);
    Msg malformed_admission_ack = AdmissionAckFor(admission_tx);
    malformed_admission_ack.data_.range(31, 0) = sc_bv<32>(0);
    malformed_admission_ack.seq_id_ = 1;
    malformed_admission_ack.length_ = 1;
    malformed_admission_ack = WireRoundTrip(malformed_admission_ack);
    const std::optional<P2pCompletionAck> admission_identity =
        ExtractP2pAckIdentityForAbort(malformed_admission_ack, 1);
    checks.Check(malformed_admission_ack.seq_id_ == 1 &&
                     malformed_admission_ack.length_ == 1 &&
                     !malformed_admission_ack.data_.range(31, 0).or_reduce(),
                 "malformed admission ACK fields survive the real wire");
    checks.Reject(
        [&] { (void)ParseP2pAdmissionAck(malformed_admission_ack); },
        "wire-realistic malformed admission ACK canonical parse");
    checks.Check(
        admission_identity.has_value() &&
            admission_identity->fsm_id == 0 &&
            !malformed_admission.AbortCompletionAckFlow(
                 admission_identity->flow, admission_identity->fsm_id)
                 .has_value() &&
            malformed_admission.AbortAdmissionAckFlow(
                admission_identity->flow, admission_identity->fsm_id) ==
                std::optional<P2pEndpointHandle>(admission_tx.handle) &&
            malformed_admission.HasFsm(admission_decoy.handle.fsm_id),
        "zero-fsm malformed admission ACK aborts only the flow in REQUEST_SENT");
    checks.Check(malformed_admission.Abort(admission_decoy.handle) &&
                     malformed_admission.Drained(),
                 "malformed admission ACK leaves the decoy generation intact");

    P2pEndpointSessionRuntime malformed_completion(89, 2, 32, 2);
    P2pTxIssue completion_tx = malformed_completion.IssueSend(
        AsyncSpec(610, 1610, bytes.size(), 90), bytes);
    P2pTxIssue completion_decoy = malformed_completion.IssueSend(
        AsyncSpec(0xf0000611U, 1612, bytes.size(), 91), bytes);
    malformed_completion.MarkRequestSent(completion_tx.handle);
    malformed_completion.MarkRequestSent(completion_decoy.handle);
    Admit(completion_tx, malformed_completion);
    Admit(completion_decoy, malformed_completion);
    const P2pFlowKey completion_flow =
        ParseP2pPayloadRequest(completion_tx.messages.request).flow;
    Msg malformed_completion_ack =
        MakeP2pCompletionAck(completion_flow, completion_tx.handle.fsm_id);
    malformed_completion_ack.data_.range(31, 0) =
        sc_bv<32>(completion_decoy.handle.fsm_id);
    malformed_completion_ack.data_.range(127, 32) = sc_bv<96>(1);
    malformed_completion_ack.is_end_ = false;
    malformed_completion_ack = WireRoundTrip(malformed_completion_ack);
    const std::optional<P2pCompletionAck> completion_identity =
        ExtractP2pAckIdentityForAbort(malformed_completion_ack, 0);
    checks.Check(!malformed_completion_ack.is_end_ &&
                     malformed_completion_ack.data_.range(127, 32).or_reduce() &&
                     malformed_completion_ack.data_.range(31, 0).to_uint64() ==
                         completion_decoy.handle.fsm_id,
                 "malformed completion ACK reserved and wrong-fsm survive wire");
    checks.Reject(
        [&] { (void)ParseP2pCompletionAck(malformed_completion_ack); },
        "wire-realistic malformed completion ACK canonical parse");
    checks.Check(
        completion_identity.has_value() &&
            completion_identity->fsm_id == completion_decoy.handle.fsm_id &&
            !malformed_completion.AbortAdmissionAckFlow(
                 completion_identity->flow, completion_identity->fsm_id)
                 .has_value() &&
            malformed_completion.AbortCompletionAckFlow(
                completion_identity->flow, completion_identity->fsm_id) ==
                std::optional<P2pEndpointHandle>(completion_tx.handle) &&
            malformed_completion.HasFsm(completion_decoy.handle.fsm_id),
        "wrong-fsm fallback aborts exact completion flow without killing decoy");
    checks.Check(malformed_completion.Abort(completion_decoy.handle) &&
                     malformed_completion.Drained(),
                 "malformed completion ACK leaves wrong-fsm decoy intact");
    checks.Check(
        !ExtractP2pAckIdentityForAbort(malformed_completion_ack, 1).has_value(),
        "malformed ACK cleanup cannot cross admission/completion kind");

    checks.Reject(
        [] {
            const std::exception_ptr injected =
                std::make_exception_ptr(std::runtime_error("ACK send"));
            RethrowP2pEndpointProtocolFailure(
                "completion ACK send", injected);
        },
        "ACK send failure is terminal instead of silently waiting");

    P2pEndpointSessionRuntime duplicate_data(83, 2, 64);
    const P2pBuiltPayload completed_unposted = BuildP2pPayload(
        P2pFlowKey{80, 83, 4, 0}, 605, bytes);
    (void)duplicate_data.ReceiveRequest(completed_unposted.request);
    checks.Check(!duplicate_data.ReceiveData(
                     completed_unposted.fragments.front()).has_value(),
                 "unposted RX completion remains private");
    checks.Reject([&] { (void)duplicate_data.ReceiveData(
                               completed_unposted.fragments.front()); },
                  "duplicate tail after RX completion");
    checks.Check(duplicate_data.Drained(),
                 "duplicate terminal DATA failure aborts all RX residual");

    P2pEndpointSessionRuntime duplicate_active(84, 2, 64);
    const P2pBuiltPayload active = BuildP2pPayload(
        P2pFlowKey{80, 84, 5, 0}, 606, Pattern(32, 7));
    (void)duplicate_active.ReceiveRequest(active.request);
    checks.Check(!duplicate_active.ReceiveData(active.fragments.front())
                      .has_value(),
                 "first active DATA remains uncommitted");
    checks.Reject([&] { (void)duplicate_active.ReceiveData(
                               active.fragments.front()); },
                  "duplicate active DATA sequence");
    checks.Check(duplicate_active.Drained(),
                 "duplicate active DATA failure aborts all RX residual");

    P2pTxIssue ack_tx =
        sender.IssueSend(AsyncSpec(603, 1603, bytes.size(), 81), bytes);
    sender.MarkRequestSent(ack_tx.handle);
    Admit(ack_tx, sender);
    const P2pFlowKey ack_flow =
        ParseP2pPayloadRequest(ack_tx.messages.request).flow;
    const Msg ack = MakeP2pCompletionAck(ack_flow, ack_tx.handle.fsm_id);
    sender.ReceiveAck(ack);
    checks.Reject([&] { sender.ReceiveAck(ack); }, "duplicate active ACK");
    checks.Check(sender.AbortAck(ParseP2pCompletionAck(ack)) &&
                     sender.Drained(),
                 "ACK failure abort releases exact active generation");
}

void TestBoundsAndUnknowns(Checks &checks) {
    checks.Reject([] { P2pEndpointSessionRuntime bad(0, 0, 16); },
                  "zero session capacity");
    checks.Reject([] { P2pEndpointSessionRuntime bad(0, 1, 0); },
                  "zero RX byte capacity");
    checks.Reject([] { P2pEndpointSessionRuntime bad(0, 1, 16, 0); },
                  "zero transport tag capacity");
    checks.Reject(
        [] { P2pEndpointSessionRuntime bad(0, 1, 16, 1, 0); },
        "zero topology capacity");
    checks.Reject(
        [] { P2pEndpointSessionRuntime bad(1, 1, 16, 1, 1); },
        "local core outside topology capacity");
    checks.Check(CheckedP2pPendingRequestCapacity(64, 1024) == 65536,
                 "pending REQUEST bound covers every topology TX session");
    checks.Reject(
        [] { (void)CheckedP2pPendingRequestCapacity(0, 1); },
        "zero-core pending REQUEST bound");
    checks.Reject(
        [] { (void)CheckedP2pPendingRequestCapacity(1, 0); },
        "zero-session pending REQUEST bound");
    P2pEndpointSessionRuntime lifetime_tags(40, 1, 16, 2, 42);
    checks.Check(lifetime_tags.RemainingTransportTags() == 2,
                 "lifetime transport-tag remaining capacity starts exact");
    P2pTxIssue lifetime_issue = lifetime_tags.IssueSend(
        AsyncSpec(40, 1040, 16, 41), Pattern(16, 0x40));
    checks.Check(lifetime_tags.RemainingTransportTags() == 1 &&
                     lifetime_tags.Abort(lifetime_issue.handle) &&
                     lifetime_tags.RemainingTransportTags() == 1,
                 "retirement does not replenish lifetime-unique tags");
    checks.Reject(
        [] {
            (void)CheckedP2pPendingRequestCapacity(
                2, std::numeric_limits<size_t>::max());
        },
        "pending REQUEST bound multiplication overflow");

    P2pEndpointSessionRuntime runtime(50, 1, 16);
    const P2pRxPostResult first =
        runtime.PostReceive(AsyncSpec(1, 1, 16, 49));
    checks.Reject(
        [&] { (void)runtime.PostReceive(AsyncSpec(2, 2, 16, 49)); },
        "session capacity");
    checks.Reject(
        [&] { (void)runtime.PostReceive(AsyncSpec(2, 1, 16, 49)); },
        "duplicate token or exhausted session");
    checks.Reject([&] { (void)runtime.Poll(0); }, "zero WAIT token");
    checks.Reject([&] { (void)runtime.Poll(999); }, "unknown WAIT token");
    checks.Reject([&] { runtime.Cancel(1); },
                  "RX CANCEL remains unsupported under capacity pressure");
    checks.Check(runtime.AbortReceiveFsm(1) && runtime.Drained(),
                 "explicit RX failure teardown leaves no state");

    const P2pBuiltPayload too_large = BuildP2pPayload(
        P2pFlowKey{49, 50, 1, 0}, 9, Pattern(17, 1));
    checks.Reject([&] { runtime.ReceiveRequest(too_large.request); },
                  "REQUEST exceeds RX byte capacity");
    checks.Check(runtime.Drained(), "failed REQUEST admission is atomic");

    P2pEndpointSessionSpec bad = AsyncSpec(1, 1, 1, 2);
    bad.fsm_id = 0;
    checks.Reject([&] { (void)runtime.PostReceive(bad); }, "zero fsm_id");
    bad = AsyncSpec(1, 0, 1, 2);
    checks.Reject([&] { (void)runtime.PostReceive(bad); },
                  "zero asynchronous token");
    bad = SyncSpec(1, 1, 2);
    bad.token = 3;
    checks.Reject([&] { (void)runtime.PostReceive(bad); },
                  "synchronous non-zero token");
    bad = AsyncSpec(1, 1, 0, 2);
    checks.Reject([&] { (void)runtime.PostReceive(bad); }, "zero length");
    (void)first;
}

void TestLifetimeStatsAndDedicatedMarker(Checks &checks) {
    P2pEndpointSessionRuntime sender(0, 3, 64, 8, 4);
    P2pEndpointSessionRuntime receiver(1, 3, 64, 8, 4);
    const auto tx0 = sender.IssueSend(SyncSpec(101, 8, 1), Pattern(8, 1));
    const auto tx1 = sender.IssueSend(SyncSpec(102, 8, 1), Pattern(8, 2));
    const auto rx0 = receiver.PostReceive(AsyncSpec(201, 11, 8, 0));
    const auto rx1 = receiver.PostReceive(AsyncSpec(202, 12, 8, 0));
    checks.Check(sender.LifetimeStats().tx_opened == 2 &&
                     sender.LifetimeStats().tx_active == 2 &&
                     sender.LifetimeStats().tx_peak == 2 &&
                     sender.LifetimeStats().rx_opened == 0,
                 "TX lifetime counters observe real session insertion");
    checks.Check(receiver.LifetimeStats().rx_opened == 2 &&
                     receiver.LifetimeStats().rx_active == 2 &&
                     receiver.LifetimeStats().rx_peak == 2 &&
                     receiver.LifetimeStats().tx_opened == 0,
                 "RX lifetime counters observe real session insertion");
    checks.Check(sender.Abort(tx0.handle) && sender.Abort(tx1.handle) &&
                     receiver.Abort(rx0.handle) && receiver.Abort(rx1.handle),
                 "fixture retires all observed sessions through production teardown");
    checks.Check(sender.LifetimeStats().tx_retired == 2 &&
                     sender.LifetimeStats().tx_active == 0 &&
                     receiver.LifetimeStats().rx_retired == 2 &&
                     receiver.LifetimeStats().rx_active == 0,
                 "retirement counters close exact TX/RX lifetimes");

    const std::string marker = FormatMoeSwizzleSessionMarker(
        MoeSwizzleSessionMarker{0, 3, 2, 6,
                                sender.LifetimeStats().tx_peak,
                                receiver.LifetimeStats().rx_peak, 4, 4});
    checks.Check(marker ==
                     "[MOE_SWIZZLE_SESSION] die=0 capacity_per_core=3 "
                     "active_core_count=2 aggregate_capacity=6 "
                     "send_peak=2 recv_peak=2 "
                     "opens=4 retires=4",
                 "dedicated marker formatter has strict parser field order");
    checks.Reject(
        [] {
            (void)FormatMoeSwizzleSessionMarker(
                MoeSwizzleSessionMarker{0, 3, 1, 3, 1, 1, 2, 1});
        },
        "dedicated marker rejects unretired session evidence");
}

void TestManifestBoundDieSessionReplay(Checks &checks) {
    const auto event = [](uint64_t ticks, uint64_t sequence, uint16_t core,
                          P2pEndpointDirection direction, int8_t delta) {
        return P2pEndpointLifetimeEvent{ticks, 0, sequence, core,
                                        direction, delta};
    };
    const std::vector<P2pEndpointLifetimeEvent> events = {
        event(1, 1, 10, P2pEndpointDirection::TX, 1),
        event(2, 2, 10, P2pEndpointDirection::TX, 1),
        event(3, 3, 10, P2pEndpointDirection::RX, 1),
        event(5, 4, 10, P2pEndpointDirection::TX, -1),
        event(6, 5, 10, P2pEndpointDirection::TX, -1),
        event(8, 6, 10, P2pEndpointDirection::RX, -1),
        event(4, 1, 11, P2pEndpointDirection::TX, 1),
        event(4, 2, 11, P2pEndpointDirection::RX, 1),
        event(6, 3, 11, P2pEndpointDirection::RX, -1),
        event(7, 4, 11, P2pEndpointDirection::TX, 1),
        event(8, 5, 11, P2pEndpointDirection::TX, -1),
        event(9, 6, 11, P2pEndpointDirection::TX, -1),
    };
    const auto markers = ReplayMoeSwizzleSessionMarkers(
        events, {{10, 0}, {11, 0}, {12, 1}, {13, 2}, {14, 3}},
        {{10, 3}, {11, 3}, {12, 3}, {13, 3}, {14, 3}}, 4);
    checks.Check(markers.size() == 4 && markers[0].send_peak == 3 &&
                     markers[0].recv_peak == 2 &&
                     markers[0].capacity_per_core == 3 &&
                     markers[0].active_core_count == 2 &&
                     markers[0].aggregate_capacity == 6 &&
                     markers[0].opens == 6 && markers[0].retires == 6,
                 "manifest-bound replay observes exact two-core die concurrency");
    checks.Check(markers[0].send_peak != 2 && markers[0].send_peak != 4,
                 "die peak is neither max nor sum of endpoint-local peaks");
    checks.Reject(
        [&] {
            (void)ReplayMoeSwizzleSessionMarkers(
                events, {{10, 0}, {12, 1}, {13, 2}, {14, 3}},
                {{10, 3}, {12, 3}, {13, 3}, {14, 3}}, 4);
        },
        "lifetime event without manifest runtime-core binding");
    checks.Reject(
        [&] {
            (void)ReplayMoeSwizzleSessionMarkers(
                events, {{10, 0}, {11, 0}, {12, 1}, {13, 2}, {14, 3}},
                {{10, 3}, {11, 2}, {12, 3}, {13, 3}, {14, 3}}, 4);
        },
        "manifest-bound runtime cores require one exact endpoint capacity");
    std::vector<P2pEndpointLifetimeEvent> underflow = {
        event(1, 1, 10, P2pEndpointDirection::TX, -1)};
    checks.Reject(
        [&] {
            (void)ReplayMoeSwizzleSessionMarkers(
                underflow, {{10, 0}, {12, 1}, {13, 2}, {14, 3}},
                {{10, 3}, {12, 3}, {13, 3}, {14, 3}}, 4);
        },
        "die replay retirement underflow");
}

} // namespace

int RunP2pSessionRuntimeSelfTest() {
    Checks checks;
    TestAckAbi(checks);
    TestAdmissionStateMachine(checks);
    TestRecvBeforeSendSync(checks);
    TestSendBeforeRecvAsync(checks);
    TestAckBeforeLocalRetire(checks);
    TestFullFsmCollisionIsolation(checks);
    TestMatchAndOrderingFailures(checks);
    TestChecksumFailureCleanup(checks);
    TestCancelAndEarlyReuse(checks);
    TestTokenReuseAckAba(checks);
    TestDirectContractValidation(checks);
    TestAdmissionCapacityProgress(checks);
    TestRequestIngressClassification(checks);
    TestLifetimeRequestIdentity(checks);
    TestFailureAbortTransitions(checks);
    TestBoundsAndUnknowns(checks);
    TestLifetimeStatsAndDedicatedMarker(checks);
    TestManifestBoundDieSessionReplay(checks);
    return checks.Finish();
}
