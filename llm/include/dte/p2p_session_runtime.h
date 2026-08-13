#pragma once

#include "dte/p2p_payload.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <deque>
#include <map>
#include <optional>
#include <set>
#include <vector>

enum class DteEndpointCompletion : uint8_t;
class Dte_send_endpoint_prim;
class Dte_recv_endpoint_prim;

enum class P2pEndpointDirection : uint8_t { TX = 0, RX = 1 };
enum class P2pRequestDisposition : uint8_t {
    NEW = 0,
    ACTIVE_DUPLICATE = 1,
    TERMINAL_DUPLICATE = 2,
    CONFLICT = 3,
};
enum class P2pRequestIngressAction : uint8_t {
    ENQUEUE = 0,
    DUPLICATE = 1,
    REJECT_CONFLICT = 2,
};
enum class P2pEndpointPhase : uint8_t {
    QUEUED = 0,
    REQUEST_SENT = 1,
    ADMITTED = 2,
    ACTIVE = 3,
    COMMIT_READY = 4,
    COMPLETE = 5,
};

struct P2pEndpointHandle {
    P2pEndpointDirection direction = P2pEndpointDirection::TX;
    DteEndpointCompletion completion =
        static_cast<DteEndpointCompletion>(0);
    uint32_t fsm_id = 0;
    uint32_t token = 0;
    uint64_t round = 0;

    bool operator==(const P2pEndpointHandle &other) const noexcept;
};

struct P2pEndpointSessionSpec {
    DteEndpointCompletion completion =
        static_cast<DteEndpointCompletion>(0);
    uint32_t fsm_id = 0;
    uint32_t token = 0;
    uint64_t length_bytes = 0;
    uint16_t peer_core = 0;
};

struct P2pCompletionAck {
    P2pFlowKey flow;
    uint32_t fsm_id = 0;
};

struct P2pAdmissionAck {
    P2pFlowKey flow;
    uint32_t fsm_id = 0;
};

P2pRequestIngressAction ClassifyP2pRequestIngress(
    const P2pPayloadDeclaration &declaration,
    const P2pPayloadDeclaration *queued,
    P2pRequestDisposition lifetime_disposition) noexcept;

Msg MakeP2pCompletionAck(const P2pFlowKey &flow, uint32_t fsm_id);
P2pCompletionAck ParseP2pCompletionAck(const Msg &message);
Msg MakeP2pAdmissionAck(const P2pFlowKey &flow, uint32_t fsm_id);
P2pAdmissionAck ParseP2pAdmissionAck(const Msg &message);
std::optional<P2pCompletionAck> ExtractP2pAckIdentityForAbort(
    const Msg &message, uint8_t expected_kind) noexcept;

[[noreturn]] void RethrowP2pEndpointProtocolFailure(
    const char *stage, std::exception_ptr failure);

size_t CheckedP2pPendingRequestCapacity(size_t topology_cores,
                                        size_t per_core_sessions);

struct P2pTxIssue {
    P2pEndpointHandle handle;
    uint16_t transport_tag = 0;
    P2pBuiltPayload messages;
};

struct P2pRxDelivery {
    P2pEndpointHandle handle;
    P2pFlowKey flow;
    std::vector<uint8_t> bytes;
};

struct P2pRxPostResult {
    P2pEndpointHandle handle;
    std::optional<P2pRxDelivery> ready;
};

struct P2pEndpointResidual {
    size_t sessions = 0;
    size_t async_tokens = 0;
    size_t allocated_transport_tags = 0;
    size_t inbound_flows = 0;
    size_t pending_requests = 0;
    size_t inflight_reassemblies = 0;
    size_t completed_unposted = 0;
    size_t commit_ready = 0;
    size_t completed_sessions = 0;
    size_t tx_awaiting_admission = 0;
    size_t tx_awaiting_ack = 0;
    size_t tx_awaiting_local_retire = 0;
    size_t reserved_rx_bytes = 0;
    size_t early_data_flows = 0;
    size_t early_data_fragments = 0;
    size_t early_data_bytes = 0;

    bool Empty() const noexcept;
};

// One instance belongs to exactly one core. It never exposes a peer runtime or
// a payload side channel: callers must carry P2pTxIssue::messages through Msg
// transport and feed them to ReceiveRequest/ReceiveData on the destination.
class P2pEndpointSessionRuntime final {
public:
    P2pEndpointSessionRuntime(uint16_t local_core,
                              size_t max_sessions,
                              size_t max_rx_buffered_bytes,
                              uint16_t max_transport_tag = UINT16_MAX,
                              uint32_t topology_cores = UINT16_MAX + 1U);

    P2pTxIssue IssueSend(const P2pEndpointSessionSpec &spec,
                         const std::vector<uint8_t> &bytes);
    P2pTxIssue IssueSend(const Dte_send_endpoint_prim &prim,
                         const std::vector<uint8_t> &bytes);
    void MarkRequestSent(const P2pEndpointHandle &handle);
    void ReceiveAdmissionAck(const Msg &ack);
    void CompleteSend(const P2pEndpointHandle &handle);
    void ReceiveAck(const Msg &ack);

    P2pRxPostResult PostReceive(const P2pEndpointSessionSpec &spec);
    P2pRxPostResult PostReceive(const Dte_recv_endpoint_prim &prim);
    std::optional<P2pRxDelivery> ReceiveRequest(const Msg &request);
    std::optional<P2pRxDelivery> ReceiveData(const Msg &fragment);
    Msg CompleteReceive(const P2pEndpointHandle &handle);

    // Failure-only teardown. The handle/ACK overloads are generation-aware;
    // inbound teardown owns the reassembly reservation for the exact flow.
    bool Abort(const P2pEndpointHandle &handle) noexcept;
    bool AbortInbound(const P2pFlowKey &flow) noexcept;
    std::optional<P2pFlowKey> AbortInboundFsm(uint32_t fsm_id) noexcept;
    bool AbortAck(const P2pCompletionAck &ack) noexcept;
    bool AbortAdmissionAck(const P2pAdmissionAck &ack) noexcept;
    std::optional<P2pEndpointHandle> AbortCompletionAckFlow(
        const P2pFlowKey &flow, uint32_t fsm_hint) noexcept;
    std::optional<P2pEndpointHandle> AbortAdmissionAckFlow(
        const P2pFlowKey &flow, uint32_t fsm_hint) noexcept;
    bool AbortReceiveFsm(uint32_t fsm_id) noexcept;

    P2pEndpointPhase Poll(uint32_t token) const;
    bool TryWait(uint32_t token);
    void Cancel(uint32_t token);

    bool TryRetireSync(const P2pEndpointHandle &handle);

    bool HasFsm(uint32_t fsm_id) const noexcept;
    bool HasHandle(const P2pEndpointHandle &handle) const noexcept;
    bool HasToken(uint32_t token) const noexcept;
    P2pEndpointPhase Phase(const P2pEndpointHandle &handle) const;
    bool IsAdmitted(const P2pEndpointHandle &handle) const;
    P2pEndpointResidual Residual() const noexcept;
    bool Drained() const noexcept { return Residual().Empty(); }

    uint16_t LocalCore() const noexcept { return local_core_; }
    size_t MaxSessions() const noexcept { return max_sessions_; }
    size_t MaxRxBufferedBytes() const noexcept {
        return max_rx_buffered_bytes_;
    }
    size_t ReservedRxBytes() const noexcept { return reserved_rx_bytes_; }
    bool CanReceiveRequest(const P2pFlowKey &flow,
                           uint64_t total_bytes) const noexcept;
    P2pRequestDisposition ClassifyRequest(
        const P2pPayloadDeclaration &declaration) const;
    size_t SeenRequestCount() const noexcept { return seen_requests_.size(); }
    size_t MaxSeenRequestIdentities() const noexcept {
        return max_seen_request_identities_;
    }
    uint32_t RemainingTransportTags() const noexcept {
        return next_transport_tag_ > max_transport_tag_
                   ? 0U
                   : static_cast<uint32_t>(max_transport_tag_) -
                         next_transport_tag_ + 1U;
    }
    bool HasInboundFlow(const P2pFlowKey &flow) const noexcept {
        return inbound_.find(flow) != inbound_.end();
    }
    size_t EarlyDataBytes() const noexcept { return early_data_bytes_; }
    bool EarlyDataAtCapacity() const noexcept {
        return early_data_.size() >= max_sessions_ ||
               early_data_bytes_ >= max_rx_buffered_bytes_;
    }

private:
    struct Session {
        P2pEndpointHandle handle;
        uint16_t peer_core = 0;
        uint64_t length_bytes = 0;
        P2pEndpointPhase phase = P2pEndpointPhase::ACTIVE;
        std::optional<P2pFlowKey> flow;
        bool local_retired = false;
        bool acknowledged = false;
    };

    struct Inbound {
        P2pPayloadDeclaration declaration;
        bool posted = false;
        std::optional<std::vector<uint8_t>> completed_bytes;
    };

    struct EarlyData {
        uint32_t next_sequence = 1;
        bool saw_tail = false;
        size_t bytes = 0;
        std::deque<Msg> fragments;
    };

    uint64_t CandidateRound() const;
    uint16_t CandidateTransportTag() const;
    Session &RequireHandle(const P2pEndpointHandle &handle);
    const Session &RequireToken(uint32_t token) const;
    Session &RequireToken(uint32_t token);
    void RetireSession(std::map<uint32_t, Session>::iterator session);
    void CleanupInbound(const P2pFlowKey &flow) noexcept;
    std::optional<P2pEndpointHandle> AbortAckFlow(
        const P2pFlowKey &flow, uint32_t fsm_hint,
        bool admission) noexcept;
    P2pRxDelivery TakeInboundCommit(const P2pFlowKey &flow,
                                    Session &session);

    uint16_t local_core_;
    size_t max_sessions_;
    size_t max_rx_buffered_bytes_;
    uint16_t max_transport_tag_;
    uint32_t topology_cores_;
    size_t max_seen_request_identities_;
    uint32_t next_transport_tag_ = 1;
    uint64_t last_round_ = 0;
    size_t reserved_rx_bytes_ = 0;
    size_t early_data_bytes_ = 0;

    P2pPayloadReassembler reassembler_;
    std::map<uint32_t, Session> sessions_;
    std::map<uint32_t, uint32_t> token_to_fsm_;
    std::set<uint16_t> allocated_transport_tags_;
    std::map<P2pFlowKey, Inbound> inbound_;
    std::map<uint32_t, P2pFlowKey> inbound_by_fsm_;
    // Lifetime REQUEST identities are bounded by topology_cores * tag space and
    // never evicted. This prevents an arbitrarily late duplicate from reviving.
    std::map<std::pair<uint16_t, uint16_t>, P2pPayloadDeclaration>
        seen_requests_;
    std::map<P2pFlowKey, EarlyData> early_data_;
};
