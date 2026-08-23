#include "dte/p2p_session_runtime.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "prims/dte_endpoint_prims.h"
#pragma GCC diagnostic pop

#include "systemc.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace {

void Require(bool condition, const std::string &message) {
    if (!condition)
        throw std::invalid_argument(message);
}

void ValidateAddress(const DteEndpointSramAddress &address,
                     const std::string &name) {
    const auto kind = static_cast<uint8_t>(address.kind);
    Require(kind == static_cast<uint8_t>(DteEndpointAddressKind::ABSOLUTE) ||
                kind == static_cast<uint8_t>(DteEndpointAddressKind::REGION),
            name + " address kind is invalid");
    if (address.kind == DteEndpointAddressKind::ABSOLUTE) {
        Require(address.region.empty() && address.region_offset_bytes == 0,
                name + " absolute address carries region metadata");
        return;
    }
    Require(address.absolute_address_bytes == 0,
            name + " region address carries absolute metadata");
    Require(!address.region.empty(), name + " region name is empty");
    Require(address.region.size() <= kDteEndpointPrimRegionMaxBytes,
            name + " region name exceeds endpoint wire limit");
    Require(address.region.find('\0') == std::string::npos,
            name + " region name contains NUL");
}

void ValidateCommon(const Dte_endpoint_prim_base &prim) {
    Require(prim.fsm_id != 0, "P2P endpoint fsm_id must be non-zero");
    Require(prim.length_bytes != 0,
            "P2P endpoint length_bytes must be non-zero");
    Require(prim.length_bytes <= kDteEndpointP2pMaxBytes,
            "P2P endpoint length_bytes exceeds transport maximum");
    const auto completion = static_cast<uint8_t>(prim.completion);
    Require(completion <= static_cast<uint8_t>(DteEndpointCompletion::SYNC),
            "P2P endpoint completion enum is invalid");
    if (prim.completion == DteEndpointCompletion::ASYNC)
        Require(prim.token != 0, "asynchronous P2P token must be non-zero");
    else
        Require(prim.token == 0, "synchronous P2P token must be zero");
    Require(prim.datatype == DteEndpointDataType::UINT8,
            "P2P endpoint datatype must be UINT8");
    Require(prim.reduce_op == DteEndpointReduceOp::NONE,
            "P2P endpoint forbids reduce_op");
    Require(prim.expected_sources == 0 && prim.tree_id == 0 &&
                prim.group_id == 0 && prim.collective_id == 0 &&
                prim.epoch == 0,
            "P2P endpoint carries collective metadata");
}

void ValidateSpec(const P2pEndpointSessionSpec &spec) {
    Require(spec.fsm_id != 0, "P2P endpoint fsm_id must be non-zero");
    Require(spec.length_bytes != 0,
            "P2P endpoint length_bytes must be non-zero");
    Require(spec.length_bytes <= kDteEndpointP2pMaxBytes,
            "P2P endpoint length_bytes exceeds transport maximum");
    const auto completion = static_cast<uint8_t>(spec.completion);
    Require(completion <= static_cast<uint8_t>(DteEndpointCompletion::SYNC),
            "P2P endpoint completion enum is invalid");
    if (spec.completion == DteEndpointCompletion::ASYNC)
        Require(spec.token != 0, "asynchronous P2P token must be non-zero");
    else
        Require(spec.token == 0, "synchronous P2P token must be zero");
}

void ValidateSend(const Dte_send_endpoint_prim &prim) {
    ValidateCommon(prim);
    Require(prim.mode == DteEndpointSendMode::P2P,
            "P2P session runtime rejects non-P2P send mode");
    const auto source_space = static_cast<uint8_t>(prim.source_space);
    Require(source_space <=
                static_cast<uint8_t>(DteEndpointSourceSpace::HBM),
            "P2P send source_space enum is invalid");
    ValidateAddress(prim.source, "P2P send");
    if (prim.source_space == DteEndpointSourceSpace::HBM)
        Require(prim.source.kind == DteEndpointAddressKind::ABSOLUTE,
                "P2P send HBM source must use an absolute address");
}

void ValidateReceive(const Dte_recv_endpoint_prim &prim) {
    ValidateCommon(prim);
    Require(prim.mode == DteEndpointRecvMode::P2P,
            "P2P session runtime rejects non-P2P receive mode");
    ValidateAddress(prim.destination, "P2P receive");
}

bool SameRequestDeclaration(const P2pPayloadDeclaration &left,
                            const P2pPayloadDeclaration &right) noexcept {
    return left.flow == right.flow && left.fsm_id == right.fsm_id &&
           left.total_bytes == right.total_bytes &&
           left.fragment_count == right.fragment_count &&
           left.checksum == right.checksum;
}

} // namespace

P2pRequestIngressAction ClassifyP2pRequestIngress(
    const P2pPayloadDeclaration &declaration,
    const P2pPayloadDeclaration *queued,
    P2pRequestDisposition lifetime_disposition) noexcept {
    if (queued != nullptr)
        return SameRequestDeclaration(*queued, declaration)
                   ? P2pRequestIngressAction::DUPLICATE
                   : P2pRequestIngressAction::REJECT_CONFLICT;
    switch (lifetime_disposition) {
    case P2pRequestDisposition::NEW:
        return P2pRequestIngressAction::ENQUEUE;
    case P2pRequestDisposition::ACTIVE_DUPLICATE:
    case P2pRequestDisposition::TERMINAL_DUPLICATE:
        return P2pRequestIngressAction::DUPLICATE;
    case P2pRequestDisposition::CONFLICT:
        return P2pRequestIngressAction::REJECT_CONFLICT;
    }
    return P2pRequestIngressAction::REJECT_CONFLICT;
}

Msg MakeP2pCompletionAck(const P2pFlowKey &flow, uint32_t fsm_id) {
    if (flow.transport_tag == 0 || flow.subflow != 0 || fsm_id == 0)
        throw std::invalid_argument("P2P ACK identity is invalid");
    Msg ack;
    ack.data_ = 0;
    ack.is_end_ = true;
    ack.msg_type_ = MSG_TYPE::ACK;
    ack.seq_id_ = 0;
    ack.des_ = flow.source;
    ack.offset_ = P2P_ENDPOINT_MSG_MARKER;
    ack.tag_id_ = flow.transport_tag;
    ack.source_ = flow.destination;
    ack.length_ = 0;
    ack.refill_ = false;
    ack.config_end_ = false;
    ack.roofline_packets_ = 0;
    ack.flow_packets_ = 0;
    ack.dte_payload_bits_ = 0;
    ack.dte_stream_source_first_ns_ = 0;
    ack.dte_stream_source_done_ns_ = 0;
    ack.dte_stream_network_tail_cycles_ = 0;
    ack.event_tag_ = 0;
    ack.subflow_ = flow.subflow;
    ack.exit_port_ = -1;
    ack.p2p_endpoint_ = true;
    ack.data_.range(31, 0) = sc_bv<32>(fsm_id);
    return ack;
}

P2pCompletionAck ParseP2pCompletionAck(const Msg &message) {
    if (!message.p2p_endpoint_ || message.msg_type_ != MSG_TYPE::ACK || !message.is_end_ ||
        message.seq_id_ != 0 ||
        message.offset_ != P2P_ENDPOINT_MSG_MARKER ||
        message.length_ != 0 || message.refill_ || message.config_end_ ||
        message.roofline_packets_ != 0 || message.flow_packets_ != 0 ||
        message.dte_payload_bits_ != 0 ||
        message.dte_stream_source_first_ns_ != 0 ||
        message.dte_stream_source_done_ns_ != 0 ||
        message.dte_stream_network_tail_cycles_ != 0 ||
        message.event_tag_ != 0 || message.exit_port_ < -1 ||
        message.exit_port_ >= static_cast<int>(UINT16_MAX) ||
        message.source_ < 0 || message.source_ > UINT16_MAX ||
        message.des_ < 0 || message.des_ > UINT16_MAX ||
        message.tag_id_ <= 0 || message.tag_id_ > UINT16_MAX ||
        message.subflow_ != 0 ||
        message.data_.range(127, 32).or_reduce())
        throw std::invalid_argument("P2P completion ACK is non-canonical");
    const uint32_t fsm_id = static_cast<uint32_t>(
        message.data_.range(31, 0).to_uint64());
    if (fsm_id == 0)
        throw std::invalid_argument("P2P completion ACK fsm_id is zero");
    return {{static_cast<uint16_t>(message.des_),
             static_cast<uint16_t>(message.source_),
             static_cast<uint16_t>(message.tag_id_),
             static_cast<uint8_t>(message.subflow_)},
            fsm_id};
}

Msg MakeP2pAdmissionAck(const P2pFlowKey &flow, uint32_t fsm_id) {
    Msg ack = MakeP2pCompletionAck(flow, fsm_id);
    // ACK subflow is a tagged kind, not part of the P2P DATA flow identity:
    // 0=completion (frozen v1), 1=REQUEST admission/CTS.
    ack.subflow_ = 1;
    return ack;
}

P2pAdmissionAck ParseP2pAdmissionAck(const Msg &message) {
    if (message.subflow_ != 1)
        throw std::invalid_argument(
            "P2P admission ACK kind is non-canonical");
    Msg completion_shape = message;
    completion_shape.subflow_ = 0;
    const P2pCompletionAck parsed =
        ParseP2pCompletionAck(completion_shape);
    return {parsed.flow, parsed.fsm_id};
}

std::optional<P2pCompletionAck> ExtractP2pAckIdentityForAbort(
    const Msg &message, uint8_t expected_kind) noexcept {
    if (!message.p2p_endpoint_ || message.msg_type_ != MSG_TYPE::ACK ||
        message.offset_ != P2P_ENDPOINT_MSG_MARKER ||
        message.subflow_ != expected_kind || message.source_ < 0 ||
        message.source_ > UINT16_MAX || message.des_ < 0 ||
        message.des_ > UINT16_MAX || message.tag_id_ <= 0 ||
        message.tag_id_ > UINT16_MAX)
        return std::nullopt;
    const uint32_t fsm_id = static_cast<uint32_t>(
        message.data_.range(31, 0).to_uint64());
    return P2pCompletionAck{
        {static_cast<uint16_t>(message.des_),
         static_cast<uint16_t>(message.source_),
         static_cast<uint16_t>(message.tag_id_), 0},
        fsm_id};
}

[[noreturn]] void RethrowP2pEndpointProtocolFailure(
    const char *stage, std::exception_ptr failure) {
    if (stage == nullptr || *stage == '\0')
        throw std::invalid_argument("P2P fatal protocol stage is empty");
    if (failure == nullptr)
        throw std::invalid_argument("P2P fatal protocol cause is empty");
    try {
        std::rethrow_exception(failure);
    } catch (...) {
        std::throw_with_nested(std::runtime_error(
            std::string("fatal P2P endpoint protocol failure at ") + stage));
    }
}

size_t CheckedP2pPendingRequestCapacity(
    size_t topology_cores, size_t per_core_sessions) {
    if (topology_cores == 0 || per_core_sessions == 0)
        throw std::invalid_argument(
            "P2P pending REQUEST capacity factors must be non-zero");
    if (per_core_sessions >
        std::numeric_limits<size_t>::max() / topology_cores)
        throw std::overflow_error(
            "P2P pending REQUEST capacity overflows size_t");
    return topology_cores * per_core_sessions;
}

bool P2pEndpointHandle::operator==(
    const P2pEndpointHandle &other) const noexcept {
    return std::tie(direction, completion, fsm_id, token, round) ==
           std::tie(other.direction, other.completion, other.fsm_id,
                    other.token, other.round);
}

bool P2pEndpointResidual::Empty() const noexcept {
    return sessions == 0 && async_tokens == 0 &&
           allocated_transport_tags == 0 && inbound_flows == 0 &&
           pending_requests == 0 && inflight_reassemblies == 0 &&
           completed_unposted == 0 && commit_ready == 0 &&
           tx_awaiting_admission == 0 &&
           completed_sessions == 0 && tx_awaiting_ack == 0 &&
           tx_awaiting_local_retire == 0 && reserved_rx_bytes == 0 &&
           early_data_flows == 0 && early_data_fragments == 0 &&
           early_data_bytes == 0;
}

std::string FormatMoeSwizzleSessionMarker(
    const MoeSwizzleSessionMarker &marker) {
    if (marker.die > 3)
        throw std::invalid_argument(
            "MoE Swizzle session marker die exceeds 2x2 mesh");
    if (marker.opens != marker.retires)
        throw std::invalid_argument(
            "MoE Swizzle session marker requires balanced lifetime counts");
    if (marker.capacity_per_core == 0 || marker.active_core_count == 0 ||
        marker.aggregate_capacity % marker.capacity_per_core != 0 ||
        marker.aggregate_capacity / marker.capacity_per_core !=
            marker.active_core_count ||
        marker.send_peak > marker.aggregate_capacity ||
        marker.recv_peak > marker.aggregate_capacity ||
        marker.send_peak > marker.opens || marker.recv_peak > marker.opens)
        throw std::invalid_argument(
            "MoE Swizzle session marker capacity/peak is inconsistent");
    std::ostringstream output;
    output << "[MOE_SWIZZLE_SESSION] die=" << marker.die
           << " capacity_per_core=" << marker.capacity_per_core
           << " active_core_count=" << marker.active_core_count
           << " aggregate_capacity=" << marker.aggregate_capacity
           << " send_peak=" << marker.send_peak
           << " recv_peak=" << marker.recv_peak
           << " opens=" << marker.opens
           << " retires=" << marker.retires;
    return output.str();
}

std::vector<MoeSwizzleSessionMarker> ReplayMoeSwizzleSessionMarkers(
    const std::vector<P2pEndpointLifetimeEvent> &events,
    const std::map<uint16_t, uint16_t> &runtime_core_to_die,
    const std::map<uint16_t, uint64_t> &runtime_core_capacity,
    uint16_t die_count) {
    if (die_count == 0 || die_count > 4)
        throw std::invalid_argument(
            "MoE Swizzle session replay requires one to four dies");
    for (const auto &binding : runtime_core_to_die) {
        if (binding.second >= die_count)
            throw std::invalid_argument(
                "MoE Swizzle runtime core binding exceeds die count");
    }
    if (runtime_core_capacity.size() != runtime_core_to_die.size())
        throw std::invalid_argument(
            "MoE Swizzle session capacity/core binding coverage drifted");
    std::optional<uint64_t> common_capacity;
    for (const auto &[core, die] : runtime_core_to_die) {
        (void)die;
        const auto found = runtime_core_capacity.find(core);
        if (found == runtime_core_capacity.end() || found->second == 0)
            throw std::invalid_argument(
                "MoE Swizzle session core capacity is missing/zero");
        if (!common_capacity.has_value()) common_capacity = found->second;
        if (*common_capacity != found->second)
            throw std::invalid_argument(
                "MoE Swizzle session per-core capacities disagree");
    }

    std::vector<P2pEndpointLifetimeEvent> by_core = events;
    std::sort(by_core.begin(), by_core.end(),
              [](const auto &left, const auto &right) {
                  return std::tie(left.local_core, left.local_sequence) <
                         std::tie(right.local_core, right.local_sequence);
              });
    std::map<uint16_t, uint64_t> expected_sequence;
    std::map<uint16_t, std::pair<uint64_t, uint64_t>> last_time;
    for (const P2pEndpointLifetimeEvent &event : by_core) {
        if (event.delta != 1 && event.delta != -1)
            throw std::invalid_argument(
                "MoE Swizzle lifetime event delta must be +1 or -1");
        if (runtime_core_to_die.find(event.local_core) ==
            runtime_core_to_die.end())
            throw std::invalid_argument(
                "MoE Swizzle lifetime event lacks manifest core binding");
        const uint64_t expected = ++expected_sequence[event.local_core];
        if (event.local_sequence != expected)
            throw std::invalid_argument(
                "MoE Swizzle lifetime event sequence is not contiguous");
        const auto now =
            std::make_pair(event.simulation_ticks, event.delta_cycle);
        auto prior = last_time.find(event.local_core);
        if (prior != last_time.end() && now < prior->second)
            throw std::invalid_argument(
                "MoE Swizzle lifetime event time moves backwards");
        last_time[event.local_core] = now;
    }

    std::vector<P2pEndpointLifetimeEvent> ordered = events;
    std::sort(ordered.begin(), ordered.end(),
              [](const auto &left, const auto &right) {
                  return std::tie(left.simulation_ticks, left.delta_cycle,
                                  left.local_sequence, left.local_core,
                                  left.direction) <
                         std::tie(right.simulation_ticks, right.delta_cycle,
                                  right.local_sequence, right.local_core,
                                  right.direction);
              });
    struct DieState {
        uint64_t tx_active = 0;
        uint64_t rx_active = 0;
        MoeSwizzleSessionMarker marker;
    };
    std::vector<DieState> states(die_count);
    for (uint16_t die = 0; die < die_count; ++die)
        states[die].marker.die = die;
    for (const auto &[core, die] : runtime_core_to_die) {
        MoeSwizzleSessionMarker &marker = states.at(die).marker;
        marker.capacity_per_core = runtime_core_capacity.at(core);
        ++marker.active_core_count;
        if (marker.aggregate_capacity >
            UINT64_MAX - runtime_core_capacity.at(core))
            throw std::overflow_error(
                "MoE Swizzle die session capacity overflows u64");
        marker.aggregate_capacity += runtime_core_capacity.at(core);
    }
    for (const P2pEndpointLifetimeEvent &event : ordered) {
        const uint16_t die = runtime_core_to_die.at(event.local_core);
        DieState &state = states.at(die);
        uint64_t &active = event.direction == P2pEndpointDirection::TX
                               ? state.tx_active
                               : state.rx_active;
        uint64_t &peak = event.direction == P2pEndpointDirection::TX
                             ? state.marker.send_peak
                             : state.marker.recv_peak;
        if (event.delta > 0) {
            ++active;
            ++state.marker.opens;
            peak = std::max(peak, active);
        } else {
            if (active == 0)
                throw std::invalid_argument(
                    "MoE Swizzle lifetime replay underflowed a die");
            --active;
            ++state.marker.retires;
        }
    }
    std::vector<MoeSwizzleSessionMarker> result;
    result.reserve(die_count);
    for (const DieState &state : states) {
        if (state.tx_active != 0 || state.rx_active != 0 ||
            state.marker.opens != state.marker.retires)
            throw std::invalid_argument(
                "MoE Swizzle session replay has unretired lifetimes");
        if (state.marker.active_core_count == 0)
            throw std::invalid_argument(
                "MoE Swizzle session replay requires a bound core per die");
        if (state.marker.send_peak > state.marker.aggregate_capacity ||
            state.marker.recv_peak > state.marker.aggregate_capacity)
            throw std::invalid_argument(
                "MoE Swizzle session peak exceeds die aggregate capacity");
        result.push_back(state.marker);
    }
    return result;
}

P2pEndpointSessionRuntime::P2pEndpointSessionRuntime(
    uint16_t local_core, size_t max_sessions, size_t max_rx_buffered_bytes,
    uint16_t max_transport_tag, uint32_t topology_cores)
    : local_core_(local_core),
      max_sessions_(max_sessions),
      max_rx_buffered_bytes_(max_rx_buffered_bytes),
      max_transport_tag_(max_transport_tag),
      topology_cores_(topology_cores),
      max_seen_request_identities_(0),
      reassembler_(max_rx_buffered_bytes, max_sessions) {
    if (max_sessions == 0)
        throw std::invalid_argument("P2P endpoint max_sessions must be non-zero");
    if (max_transport_tag == 0)
        throw std::invalid_argument(
            "P2P endpoint transport tag space must be non-zero");
    if (topology_cores == 0 || topology_cores > UINT16_MAX + 1U ||
        local_core >= topology_cores)
        throw std::invalid_argument(
            "P2P endpoint topology core bound is invalid");
    if (static_cast<size_t>(topology_cores) >
        std::numeric_limits<size_t>::max() / max_transport_tag)
        throw std::overflow_error(
            "P2P lifetime REQUEST identity bound overflows size_t");
    max_seen_request_identities_ =
        static_cast<size_t>(topology_cores) * max_transport_tag;
}

void P2pEndpointSessionRuntime::RecordLifetimeEvent(
    P2pEndpointDirection direction, int8_t delta) noexcept {
    if (!lifetime_events_complete_)
        return;
    if (lifetime_event_sequence_ == UINT64_MAX) {
        lifetime_events_complete_ = false;
        return;
    }
    const uint64_t sequence = lifetime_event_sequence_ + 1;
    try {
        lifetime_events_.push_back(P2pEndpointLifetimeEvent{
            static_cast<uint64_t>(sc_time_stamp().value()),
            static_cast<uint64_t>(sc_delta_count()), sequence, local_core_,
            direction, delta});
    } catch (...) {
        lifetime_events_complete_ = false;
        return;
    }
    lifetime_event_sequence_ = sequence;
}

uint64_t P2pEndpointSessionRuntime::CandidateRound() const {
    if (last_round_ == UINT64_MAX)
        throw std::overflow_error("P2P endpoint round space exhausted");
    return last_round_ + 1;
}

uint16_t P2pEndpointSessionRuntime::CandidateTransportTag() const {
    if (next_transport_tag_ == 0 || next_transport_tag_ > max_transport_tag_)
        throw std::length_error(
            "P2P endpoint lifetime transport tag space exhausted");
    return static_cast<uint16_t>(next_transport_tag_);
}

P2pTxIssue P2pEndpointSessionRuntime::IssueSend(
    const Dte_send_endpoint_prim &prim,
    const std::vector<uint8_t> &bytes) {
    ValidateSend(prim);
    return IssueSend(P2pEndpointSessionSpec{prim.completion, prim.fsm_id,
                                            prim.token, prim.length_bytes,
                                            prim.peer_core},
                     bytes);
}

P2pTxIssue P2pEndpointSessionRuntime::IssueSend(
    const P2pEndpointSessionSpec &prim,
    const std::vector<uint8_t> &bytes) {
    ValidateSpec(prim);
    if (prim.peer_core >= topology_cores_)
        throw std::invalid_argument(
            "P2P send peer exceeds configured topology");
    if (prim.length_bytes != bytes.size())
        throw std::invalid_argument(
            "P2P send bytes do not match endpoint length");
    if (sessions_.size() >= max_sessions_)
        throw std::length_error("P2P endpoint session capacity exhausted");
    if (sessions_.find(prim.fsm_id) != sessions_.end())
        throw std::invalid_argument("P2P endpoint fsm_id is already active");
    if (prim.completion == DteEndpointCompletion::ASYNC &&
        token_to_fsm_.find(prim.token) != token_to_fsm_.end())
        throw std::invalid_argument("P2P asynchronous token is already active");

    const uint64_t round = CandidateRound();
    const uint16_t transport_tag = CandidateTransportTag();
    const P2pFlowKey flow{local_core_, prim.peer_core, transport_tag, 0};
    P2pBuiltPayload built = BuildP2pPayload(flow, prim.fsm_id, bytes);
    const P2pEndpointHandle handle{P2pEndpointDirection::TX,
                                   prim.completion,
                                   prim.fsm_id,
                                   prim.token,
                                   round};
    Session state{handle,
                  prim.peer_core,
                  prim.length_bytes,
                  P2pEndpointPhase::QUEUED,
                  flow};

    const auto tag_inserted = allocated_transport_tags_.insert(transport_tag);
    if (!tag_inserted.second)
        throw std::logic_error("P2P transport tag allocation raced");
    try {
        if (prim.completion == DteEndpointCompletion::ASYNC) {
            const auto token_inserted =
                token_to_fsm_.emplace(prim.token, prim.fsm_id);
            if (!token_inserted.second)
                throw std::logic_error("P2P token insertion raced");
        }
        const auto session_inserted =
            sessions_.emplace(prim.fsm_id, std::move(state));
        if (!session_inserted.second)
            throw std::logic_error("P2P fsm insertion raced");
    } catch (...) {
        auto mapped = token_to_fsm_.find(prim.token);
        if (mapped != token_to_fsm_.end() &&
            mapped->second == prim.fsm_id)
            token_to_fsm_.erase(mapped);
        allocated_transport_tags_.erase(transport_tag);
        throw;
    }

    ++lifetime_stats_.tx_opened;
    ++lifetime_stats_.tx_active;
    lifetime_stats_.tx_peak =
        std::max(lifetime_stats_.tx_peak, lifetime_stats_.tx_active);
    RecordLifetimeEvent(P2pEndpointDirection::TX, 1);

    last_round_ = round;
    ++next_transport_tag_;
    return {handle, transport_tag, std::move(built)};
}

P2pEndpointSessionRuntime::Session &
P2pEndpointSessionRuntime::RequireHandle(
    const P2pEndpointHandle &handle) {
    auto found = sessions_.find(handle.fsm_id);
    if (found == sessions_.end() || !(found->second.handle == handle))
        throw std::invalid_argument("unknown or stale P2P endpoint handle");
    return found->second;
}

const P2pEndpointSessionRuntime::Session &
P2pEndpointSessionRuntime::RequireToken(uint32_t token) const {
    if (token == 0)
        throw std::invalid_argument("P2P WAIT token must be non-zero");
    auto mapped = token_to_fsm_.find(token);
    if (mapped == token_to_fsm_.end())
        throw std::invalid_argument("unknown P2P asynchronous token");
    auto found = sessions_.find(mapped->second);
    if (found == sessions_.end() || found->second.handle.token != token ||
        found->second.handle.completion != DteEndpointCompletion::ASYNC)
        throw std::logic_error("P2P token index is inconsistent");
    return found->second;
}

P2pEndpointSessionRuntime::Session &
P2pEndpointSessionRuntime::RequireToken(uint32_t token) {
    return const_cast<Session &>(
        static_cast<const P2pEndpointSessionRuntime *>(this)->RequireToken(
            token));
}

void P2pEndpointSessionRuntime::MarkRequestSent(
    const P2pEndpointHandle &handle) {
    Session &session = RequireHandle(handle);
    if (session.handle.direction != P2pEndpointDirection::TX)
        throw std::invalid_argument("P2P REQUEST send uses RX handle");
    if (session.phase != P2pEndpointPhase::QUEUED)
        throw std::invalid_argument("duplicate P2P REQUEST send transition");
    session.phase = P2pEndpointPhase::REQUEST_SENT;
}

void P2pEndpointSessionRuntime::ReceiveAdmissionAck(const Msg &message) {
    const P2pAdmissionAck ack = ParseP2pAdmissionAck(message);
    if (ack.flow.source != local_core_)
        throw std::invalid_argument(
            "P2P admission ACK targets a different TX core");
    auto found = sessions_.find(ack.fsm_id);
    if (found == sessions_.end())
        throw std::invalid_argument("unknown or stale P2P admission ACK");
    Session &session = found->second;
    if (session.handle.direction != P2pEndpointDirection::TX ||
        !session.flow.has_value() || !(ack.flow == *session.flow) ||
        session.phase != P2pEndpointPhase::REQUEST_SENT)
        throw std::invalid_argument(
            "P2P admission ACK does not match REQUEST_SENT session");
    session.phase = P2pEndpointPhase::ADMITTED;
}

void P2pEndpointSessionRuntime::CompleteSend(
    const P2pEndpointHandle &handle) {
    Session &session = RequireHandle(handle);
    if (session.handle.direction != P2pEndpointDirection::TX)
        throw std::invalid_argument("P2P send completion uses RX handle");
    if (session.phase != P2pEndpointPhase::ADMITTED)
        throw std::invalid_argument("P2P send completion requires admission");
    session.phase = P2pEndpointPhase::COMPLETE;
}

void P2pEndpointSessionRuntime::ReceiveAck(const Msg &message) {
    const P2pCompletionAck ack = ParseP2pCompletionAck(message);
    if (ack.flow.source != local_core_)
        throw std::invalid_argument("P2P ACK targets a different TX core");
    auto found = sessions_.find(ack.fsm_id);
    if (found == sessions_.end())
        throw std::invalid_argument("unknown or stale P2P completion ACK");
    Session &session = found->second;
    if (session.handle.direction != P2pEndpointDirection::TX ||
        !session.flow.has_value() || !(ack.flow == *session.flow) ||
        (session.phase != P2pEndpointPhase::ADMITTED &&
         session.phase != P2pEndpointPhase::COMPLETE))
        throw std::invalid_argument("P2P ACK does not match TX session");
    if (session.acknowledged)
        throw std::invalid_argument("duplicate P2P completion ACK");
    session.acknowledged = true;
    if (session.local_retired)
        RetireSession(found);
}

P2pRxPostResult P2pEndpointSessionRuntime::PostReceive(
    const Dte_recv_endpoint_prim &prim) {
    ValidateReceive(prim);
    return PostReceive(P2pEndpointSessionSpec{prim.completion, prim.fsm_id,
                                               prim.token, prim.length_bytes,
                                               prim.peer_core});
}

P2pRxPostResult P2pEndpointSessionRuntime::PostReceive(
    const P2pEndpointSessionSpec &prim) {
    ValidateSpec(prim);
    if (prim.peer_core >= topology_cores_)
        throw std::invalid_argument(
            "P2P receive peer exceeds configured topology");
    if (sessions_.size() >= max_sessions_)
        throw std::length_error("P2P endpoint session capacity exhausted");
    if (sessions_.find(prim.fsm_id) != sessions_.end())
        throw std::invalid_argument("P2P endpoint fsm_id is already active");
    if (prim.completion == DteEndpointCompletion::ASYNC &&
        token_to_fsm_.find(prim.token) != token_to_fsm_.end())
        throw std::invalid_argument("P2P asynchronous token is already active");

    auto pending_index = inbound_by_fsm_.find(prim.fsm_id);
    if (pending_index != inbound_by_fsm_.end()) {
        const Inbound &pending = inbound_.at(pending_index->second);
        if (pending.declaration.flow.destination != local_core_ ||
            pending.declaration.flow.source != prim.peer_core ||
            pending.declaration.total_bytes != prim.length_bytes)
            throw std::invalid_argument(
                "P2P posted receive does not match pending REQUEST");
        if (pending.posted)
            throw std::logic_error("P2P pending REQUEST is already posted");
    }

    const uint64_t round = CandidateRound();
    const P2pEndpointHandle handle{P2pEndpointDirection::RX,
                                   prim.completion,
                                   prim.fsm_id,
                                   prim.token,
                                   round};
    Session state{handle,
                  prim.peer_core,
                  prim.length_bytes,
                  P2pEndpointPhase::ACTIVE,
                  std::nullopt};
    if (pending_index != inbound_by_fsm_.end())
        state.flow = pending_index->second;

    if (prim.completion == DteEndpointCompletion::ASYNC) {
        const auto token_inserted =
            token_to_fsm_.emplace(prim.token, prim.fsm_id);
        if (!token_inserted.second)
            throw std::logic_error("P2P token insertion raced");
    }
    try {
        const auto inserted = sessions_.emplace(prim.fsm_id, std::move(state));
        if (!inserted.second)
            throw std::logic_error("P2P fsm insertion raced");
    } catch (...) {
        auto mapped = token_to_fsm_.find(prim.token);
        if (mapped != token_to_fsm_.end() &&
            mapped->second == prim.fsm_id)
            token_to_fsm_.erase(mapped);
        throw;
    }
    ++lifetime_stats_.rx_opened;
    ++lifetime_stats_.rx_active;
    lifetime_stats_.rx_peak =
        std::max(lifetime_stats_.rx_peak, lifetime_stats_.rx_active);
    RecordLifetimeEvent(P2pEndpointDirection::RX, 1);
    last_round_ = round;

    Session &session = sessions_.at(prim.fsm_id);
    P2pRxPostResult result{handle, std::nullopt};
    if (pending_index != inbound_by_fsm_.end()) {
        const P2pFlowKey pending_flow = pending_index->second;
        Inbound &pending = inbound_.at(pending_flow);
        pending.posted = true;
        if (pending.completed_bytes.has_value())
            result.ready = TakeInboundCommit(pending_flow, session);
    }
    return result;
}

P2pRequestDisposition P2pEndpointSessionRuntime::ClassifyRequest(
    const P2pPayloadDeclaration &declaration) const {
    if (declaration.flow.destination != local_core_ ||
        declaration.flow.source >= topology_cores_ ||
        declaration.flow.transport_tag == 0 ||
        declaration.flow.transport_tag > max_transport_tag_ ||
        declaration.flow.subflow != 0)
        throw std::invalid_argument(
            "P2P REQUEST identity is outside the configured topology/tag space");
    const auto identity = std::make_pair(
        declaration.flow.source, declaration.flow.transport_tag);
    auto seen = seen_requests_.find(identity);
    if (seen == seen_requests_.end())
        return P2pRequestDisposition::NEW;
    if (!SameRequestDeclaration(seen->second, declaration))
        return P2pRequestDisposition::CONFLICT;
    return inbound_.find(declaration.flow) != inbound_.end()
               ? P2pRequestDisposition::ACTIVE_DUPLICATE
               : P2pRequestDisposition::TERMINAL_DUPLICATE;
}

bool P2pEndpointSessionRuntime::CanReceiveRequest(
    const P2pFlowKey &flow, uint64_t total_bytes) const noexcept {
    if (flow.destination != local_core_ || flow.source >= topology_cores_ ||
        flow.transport_tag == 0 || flow.transport_tag > max_transport_tag_ ||
        flow.subflow != 0 || total_bytes == 0 ||
        total_bytes > max_rx_buffered_bytes_ ||
        inbound_.find(flow) != inbound_.end() ||
        inbound_.size() >= max_sessions_ ||
        reserved_rx_bytes_ > max_rx_buffered_bytes_)
        return false;
    // Early DATA and declared/reassembly bytes are two independently bounded
    // pools. A REQUEST can therefore always convert its matching early prefix
    // without being cyclically blocked by prefixes for other flows.
    return total_bytes <= max_rx_buffered_bytes_ - reserved_rx_bytes_;
}

std::optional<P2pRxDelivery>
P2pEndpointSessionRuntime::ReceiveRequest(const Msg &request) {
    const P2pPayloadDeclaration declaration =
        ParseP2pPayloadRequest(request);
    const P2pRequestDisposition disposition = ClassifyRequest(declaration);
    if (disposition == P2pRequestDisposition::CONFLICT)
        throw std::invalid_argument(
            "conflicting lifetime P2P REQUEST source/tag identity");
    if (disposition != P2pRequestDisposition::NEW)
        return std::nullopt;
    if (inbound_by_fsm_.find(declaration.fsm_id) != inbound_by_fsm_.end())
        throw std::invalid_argument("duplicate or early-reused P2P REQUEST fsm_id");
    if (!CanReceiveRequest(declaration.flow, declaration.total_bytes))
        throw std::length_error("P2P receive byte or flow capacity exhausted");

    auto early = early_data_.find(declaration.flow);
    if (early != early_data_.end()) {
        const size_t count = early->second.fragments.size();
        const bool exact_tail =
            early->second.saw_tail &&
            count == declaration.fragment_count &&
            early->second.bytes == declaration.total_bytes;
        const bool proper_prefix =
            !early->second.saw_tail &&
            count < declaration.fragment_count &&
            early->second.bytes ==
                count * P2P_PAYLOAD_FRAGMENT_BYTES;
        if (!exact_tail && !proper_prefix)
            throw std::invalid_argument(
                "early P2P DATA does not match REQUEST declaration");
    }

    auto posted = sessions_.find(declaration.fsm_id);
    if (posted != sessions_.end()) {
        const Session &session = posted->second;
        if (session.handle.direction != P2pEndpointDirection::RX ||
            session.phase != P2pEndpointPhase::ACTIVE ||
            session.flow.has_value() ||
            session.peer_core != declaration.flow.source ||
            session.length_bytes != declaration.total_bytes)
            throw std::invalid_argument(
                "P2P REQUEST does not match posted receive descriptor");
    }

    // The bounded accounting transfers this flow from early_data_bytes_ to
    // reserved_rx_bytes_ below. During replay the deque and reassembler vector
    // briefly coexist in host memory; both belong to independent fixed pools, for a total host payload bound
    // of 2 * max_rx_buffered_bytes_, and tests assert exact logical residuals.
    reassembler_.Begin(request);
    try {
        const auto inserted = inbound_.emplace(
            declaration.flow,
            Inbound{declaration, posted != sessions_.end(), std::nullopt});
        if (!inserted.second)
            throw std::logic_error("P2P inbound flow insertion raced");
        try {
            const auto indexed =
                inbound_by_fsm_.emplace(declaration.fsm_id, declaration.flow);
            if (!indexed.second)
                throw std::logic_error("P2P inbound fsm insertion raced");
        } catch (...) {
            inbound_.erase(declaration.flow);
            throw;
        }
        if (seen_requests_.size() >= max_seen_request_identities_) {
            inbound_by_fsm_.erase(declaration.fsm_id);
            inbound_.erase(declaration.flow);
            throw std::length_error(
                "P2P lifetime REQUEST identity table exhausted");
        }
        const auto identity = std::make_pair(
            declaration.flow.source, declaration.flow.transport_tag);
        const auto seen = seen_requests_.emplace(identity, declaration);
        if (!seen.second) {
            inbound_by_fsm_.erase(declaration.fsm_id);
            inbound_.erase(declaration.flow);
            throw std::logic_error(
                "P2P lifetime REQUEST identity insertion raced");
        }
    } catch (...) {
        (void)reassembler_.Abort(declaration.flow);
        throw;
    }
    reserved_rx_bytes_ += static_cast<size_t>(declaration.total_bytes);
    if (posted != sessions_.end())
        posted->second.flow = declaration.flow;

    if (early == early_data_.end())
        return std::nullopt;

    std::deque<Msg> staged = std::move(early->second.fragments);
    early_data_bytes_ -= early->second.bytes;
    early_data_.erase(early);
    std::optional<P2pPayloadCommit> commit;
    try {
        for (const Msg &fragment : staged) {
            commit = reassembler_.Accept(fragment);
            if (commit.has_value() && &fragment != &staged.back())
                throw std::logic_error(
                    "early P2P DATA completed before staged tail");
        }
    } catch (...) {
        (void)AbortInbound(declaration.flow);
        throw;
    }
    if (!commit.has_value())
        return std::nullopt;

    auto inbound = inbound_.find(declaration.flow);
    if (inbound == inbound_.end())
        throw std::logic_error("early P2P completion lost inbound state");
    inbound->second.completed_bytes = std::move(commit->bytes);
    if (!inbound->second.posted)
        return std::nullopt;
    auto session = sessions_.find(declaration.fsm_id);
    if (session == sessions_.end()) {
        (void)AbortInbound(declaration.flow);
        throw std::logic_error("early P2P completion lost posted receive");
    }
    return TakeInboundCommit(declaration.flow, session->second);
}

std::optional<P2pRxDelivery>
P2pEndpointSessionRuntime::ReceiveData(const Msg &fragment) {
    const P2pDataFragment parsed = ParseP2pDataFragment(fragment);
    if (parsed.flow.destination != local_core_ || parsed.flow.subflow != 0)
        throw std::invalid_argument("P2P DATA targets an invalid endpoint flow");
    auto inbound = inbound_.find(parsed.flow);
    // DATA is never legal before the destination has reserved the REQUEST and
    // returned admission CTS. Rejecting it atomically is the bounded defense;
    // production TX cannot emit DATA until the matching CTS is consumed.
    if (inbound == inbound_.end())
        throw std::invalid_argument("P2P DATA arrived before admission");
    if (inbound->second.completed_bytes.has_value()) {
        (void)AbortInbound(parsed.flow);
        throw std::invalid_argument("duplicate P2P DATA after completed flow");
    }
    const uint32_t fsm_id = inbound->second.declaration.fsm_id;

    std::optional<P2pPayloadCommit> commit;
    try {
        commit = reassembler_.Accept(fragment);
    } catch (...) {
        (void)AbortInbound(parsed.flow);
        throw;
    }
    if (!commit.has_value())
        return std::nullopt;

    inbound = inbound_.find(parsed.flow);
    if (inbound == inbound_.end())
        throw std::logic_error("P2P completed flow lost endpoint state");
    inbound->second.completed_bytes = std::move(commit->bytes);
    if (!inbound->second.posted)
        return std::nullopt;

    auto session = sessions_.find(fsm_id);
    if (session == sessions_.end()) {
        (void)AbortInbound(parsed.flow);
        throw std::logic_error("P2P posted receive lost endpoint session");
    }
    return TakeInboundCommit(parsed.flow, session->second);
}

P2pRxDelivery P2pEndpointSessionRuntime::TakeInboundCommit(
    const P2pFlowKey &flow, Session &session) {
    auto inbound = inbound_.find(flow);
    if (inbound == inbound_.end() ||
        !inbound->second.completed_bytes.has_value())
        throw std::logic_error("P2P receive commit is not ready");
    if (session.handle.direction != P2pEndpointDirection::RX ||
        session.phase != P2pEndpointPhase::ACTIVE)
        throw std::logic_error("P2P receive session cannot accept commit");

    P2pRxDelivery delivery{session.handle,
                           flow,
                           std::move(*inbound->second.completed_bytes)};
    session.phase = P2pEndpointPhase::COMMIT_READY;
    CleanupInbound(flow);
    return delivery;
}

Msg P2pEndpointSessionRuntime::CompleteReceive(
    const P2pEndpointHandle &handle) {
    Session &session = RequireHandle(handle);
    if (session.handle.direction != P2pEndpointDirection::RX)
        throw std::invalid_argument("P2P receive completion uses TX handle");
    if (session.phase != P2pEndpointPhase::COMMIT_READY)
        throw std::invalid_argument("unknown or duplicate P2P receive completion");
    if (!session.flow.has_value())
        throw std::logic_error("P2P receive completion lost transport flow");
    session.phase = P2pEndpointPhase::COMPLETE;
    return MakeP2pCompletionAck(*session.flow, session.handle.fsm_id);
}

bool P2pEndpointSessionRuntime::Abort(
    const P2pEndpointHandle &handle) noexcept {
    auto session = sessions_.find(handle.fsm_id);
    if (session == sessions_.end() || !(session->second.handle == handle))
        return false;
    RetireSession(session);
    return true;
}

bool P2pEndpointSessionRuntime::AbortInbound(
    const P2pFlowKey &flow) noexcept {
    bool removed = false;
    auto early = early_data_.find(flow);
    if (early != early_data_.end()) {
        early_data_bytes_ -= early->second.bytes;
        early_data_.erase(early);
        removed = true;
    }
    auto inbound = inbound_.find(flow);
    if (inbound == inbound_.end())
        return removed;
    const uint32_t fsm_id = inbound->second.declaration.fsm_id;
    CleanupInbound(flow);
    removed = true;
    auto session = sessions_.find(fsm_id);
    if (session != sessions_.end() &&
        session->second.handle.direction == P2pEndpointDirection::RX &&
        session->second.flow.has_value() &&
        *session->second.flow == flow)
        (void)Abort(session->second.handle);
    return removed;
}

std::optional<P2pFlowKey> P2pEndpointSessionRuntime::AbortInboundFsm(
    uint32_t fsm_id) noexcept {
    auto indexed = inbound_by_fsm_.find(fsm_id);
    if (indexed == inbound_by_fsm_.end())
        return std::nullopt;
    const P2pFlowKey flow = indexed->second;
    CleanupInbound(flow);
    auto session = sessions_.find(fsm_id);
    if (session != sessions_.end() &&
        session->second.handle.direction == P2pEndpointDirection::RX)
        (void)Abort(session->second.handle);
    return flow;
}

bool P2pEndpointSessionRuntime::AbortAck(
    const P2pCompletionAck &ack) noexcept {
    auto session = sessions_.find(ack.fsm_id);
    if (session == sessions_.end() ||
        session->second.handle.direction != P2pEndpointDirection::TX ||
        !session->second.flow.has_value() ||
        !(*session->second.flow == ack.flow) ||
        (session->second.phase != P2pEndpointPhase::ADMITTED &&
         session->second.phase != P2pEndpointPhase::COMPLETE))
        return false;
    return Abort(session->second.handle);
}

bool P2pEndpointSessionRuntime::AbortAdmissionAck(
    const P2pAdmissionAck &ack) noexcept {
    auto session = sessions_.find(ack.fsm_id);
    if (session == sessions_.end() ||
        session->second.handle.direction != P2pEndpointDirection::TX ||
        !session->second.flow.has_value() ||
        !(*session->second.flow == ack.flow) ||
        session->second.phase != P2pEndpointPhase::REQUEST_SENT)
        return false;
    return Abort(session->second.handle);
}

std::optional<P2pEndpointHandle>
P2pEndpointSessionRuntime::AbortAckFlow(
    const P2pFlowKey &flow, uint32_t fsm_hint, bool admission) noexcept {
    const auto eligible = [&](const Session &session) {
        if (session.handle.direction != P2pEndpointDirection::TX ||
            !session.flow.has_value() || !(*session.flow == flow))
            return false;
        return admission
                   ? session.phase == P2pEndpointPhase::REQUEST_SENT
                   : (session.phase == P2pEndpointPhase::ADMITTED ||
                      session.phase == P2pEndpointPhase::COMPLETE);
    };
    auto abort_session = [&](std::map<uint32_t, Session>::iterator found)
        -> std::optional<P2pEndpointHandle> {
        if (found == sessions_.end() || !eligible(found->second))
            return std::nullopt;
        const P2pEndpointHandle handle = found->second.handle;
        if (!Abort(handle)) return std::nullopt;
        return handle;
    };
    if (fsm_hint != 0) {
        std::optional<P2pEndpointHandle> exact =
            abort_session(sessions_.find(fsm_hint));
        if (exact.has_value()) return exact;
    }
    for (auto session = sessions_.begin(); session != sessions_.end();
         ++session) {
        if (session->first == fsm_hint) continue;
        if (!eligible(session->second)) continue;
        return abort_session(session);
    }
    return std::nullopt;
}

std::optional<P2pEndpointHandle>
P2pEndpointSessionRuntime::AbortCompletionAckFlow(
    const P2pFlowKey &flow, uint32_t fsm_hint) noexcept {
    return AbortAckFlow(flow, fsm_hint, false);
}

std::optional<P2pEndpointHandle>
P2pEndpointSessionRuntime::AbortAdmissionAckFlow(
    const P2pFlowKey &flow, uint32_t fsm_hint) noexcept {
    return AbortAckFlow(flow, fsm_hint, true);
}

bool P2pEndpointSessionRuntime::AbortReceiveFsm(
    uint32_t fsm_id) noexcept {
    auto session = sessions_.find(fsm_id);
    if (session == sessions_.end() ||
        session->second.handle.direction != P2pEndpointDirection::RX)
        return false;
    return Abort(session->second.handle);
}

P2pEndpointPhase P2pEndpointSessionRuntime::Poll(uint32_t token) const {
    return RequireToken(token).phase;
}

bool P2pEndpointSessionRuntime::TryWait(uint32_t token) {
    Session &session = RequireToken(token);
    if (session.phase != P2pEndpointPhase::COMPLETE)
        return false;
    auto found = sessions_.find(session.handle.fsm_id);
    if (session.handle.direction == P2pEndpointDirection::TX) {
        auto mapped = token_to_fsm_.find(token);
        if (mapped != token_to_fsm_.end() &&
            mapped->second == session.handle.fsm_id)
            token_to_fsm_.erase(mapped);
        session.local_retired = true;
        if (session.acknowledged)
            RetireSession(found);
    } else {
        RetireSession(found);
    }
    return true;
}

void P2pEndpointSessionRuntime::Cancel(uint32_t token) {
    Session &session = RequireToken(token);
    const bool cancellable_tx =
        session.handle.direction == P2pEndpointDirection::TX &&
        session.phase == P2pEndpointPhase::QUEUED;
    if (!cancellable_tx)
        throw std::invalid_argument(
            "P2P CANCEL only permits a queued TX; RX cancellation is unsupported");
    auto found = sessions_.find(session.handle.fsm_id);
    RetireSession(found);
}

bool P2pEndpointSessionRuntime::TryRetireSync(
    const P2pEndpointHandle &handle) {
    Session &session = RequireHandle(handle);
    if (session.handle.completion != DteEndpointCompletion::SYNC)
        throw std::invalid_argument("async P2P handle requires token WAIT");
    if (session.local_retired)
        throw std::invalid_argument("duplicate synchronous P2P retire");
    if (session.phase != P2pEndpointPhase::COMPLETE)
        return false;
    auto found = sessions_.find(session.handle.fsm_id);
    if (session.handle.direction == P2pEndpointDirection::TX) {
        session.local_retired = true;
        if (session.acknowledged)
            RetireSession(found);
    } else {
        RetireSession(found);
    }
    return true;
}

void P2pEndpointSessionRuntime::RetireSession(
    std::map<uint32_t, Session>::iterator session) {
    if (session == sessions_.end())
        throw std::logic_error("cannot retire missing P2P session");
    const P2pEndpointDirection direction =
        session->second.handle.direction;
    if (session->second.flow.has_value() &&
        inbound_.find(*session->second.flow) != inbound_.end())
        CleanupInbound(*session->second.flow);
    if (session->second.handle.completion == DteEndpointCompletion::ASYNC) {
        auto mapped =
            token_to_fsm_.find(session->second.handle.token);
        if (mapped != token_to_fsm_.end() &&
            mapped->second == session->second.handle.fsm_id)
            token_to_fsm_.erase(mapped);
    }
    if (session->second.handle.direction == P2pEndpointDirection::TX &&
        session->second.flow.has_value())
        allocated_transport_tags_.erase(
            session->second.flow->transport_tag);
    sessions_.erase(session);
    if (direction == P2pEndpointDirection::TX) {
        if (lifetime_stats_.tx_active == 0)
            lifetime_events_complete_ = false;
        else
            --lifetime_stats_.tx_active;
        ++lifetime_stats_.tx_retired;
        RecordLifetimeEvent(P2pEndpointDirection::TX, -1);
    } else {
        if (lifetime_stats_.rx_active == 0)
            lifetime_events_complete_ = false;
        else
            --lifetime_stats_.rx_active;
        ++lifetime_stats_.rx_retired;
        RecordLifetimeEvent(P2pEndpointDirection::RX, -1);
    }
}

void P2pEndpointSessionRuntime::CleanupInbound(
    const P2pFlowKey &flow) noexcept {
    auto inbound = inbound_.find(flow);
    if (inbound == inbound_.end())
        return;
    (void)reassembler_.Abort(flow);
    const size_t bytes =
        static_cast<size_t>(inbound->second.declaration.total_bytes);
    inbound_by_fsm_.erase(inbound->second.declaration.fsm_id);
    inbound_.erase(inbound);
    reserved_rx_bytes_ -= bytes;
}

bool P2pEndpointSessionRuntime::HasFsm(uint32_t fsm_id) const noexcept {
    return sessions_.find(fsm_id) != sessions_.end();
}

bool P2pEndpointSessionRuntime::HasHandle(
    const P2pEndpointHandle &handle) const noexcept {
    const auto found = sessions_.find(handle.fsm_id);
    return found != sessions_.end() && found->second.handle == handle;
}

bool P2pEndpointSessionRuntime::HasToken(uint32_t token) const noexcept {
    return token_to_fsm_.find(token) != token_to_fsm_.end();
}

P2pEndpointPhase P2pEndpointSessionRuntime::Phase(
    const P2pEndpointHandle &handle) const {
    auto found = sessions_.find(handle.fsm_id);
    if (found == sessions_.end() || !(found->second.handle == handle))
        throw std::invalid_argument("unknown or stale P2P endpoint handle");
    return found->second.phase;
}

bool P2pEndpointSessionRuntime::IsAdmitted(
    const P2pEndpointHandle &handle) const {
    return Phase(handle) == P2pEndpointPhase::ADMITTED;
}

P2pEndpointResidual P2pEndpointSessionRuntime::Residual() const noexcept {
    P2pEndpointResidual residual;
    residual.sessions = sessions_.size();
    residual.async_tokens = token_to_fsm_.size();
    residual.allocated_transport_tags = allocated_transport_tags_.size();
    residual.inbound_flows = inbound_.size();
    residual.reserved_rx_bytes = reserved_rx_bytes_;
    residual.early_data_flows = early_data_.size();
    residual.early_data_bytes = early_data_bytes_;
    for (const auto &entry : early_data_)
        residual.early_data_fragments += entry.second.fragments.size();
    for (const auto &entry : inbound_) {
        if (!entry.second.posted)
            ++residual.pending_requests;
        if (reassembler_.HasActive(entry.first))
            ++residual.inflight_reassemblies;
        if (entry.second.completed_bytes.has_value() && !entry.second.posted)
            ++residual.completed_unposted;
    }
    for (const auto &entry : sessions_) {
        if (entry.second.phase == P2pEndpointPhase::COMMIT_READY)
            ++residual.commit_ready;
        if (entry.second.phase == P2pEndpointPhase::COMPLETE)
            ++residual.completed_sessions;
        if (entry.second.handle.direction == P2pEndpointDirection::TX) {
            if (entry.second.phase == P2pEndpointPhase::REQUEST_SENT)
                ++residual.tx_awaiting_admission;
            if (!entry.second.acknowledged &&
                (entry.second.phase == P2pEndpointPhase::ADMITTED ||
                 entry.second.phase == P2pEndpointPhase::COMPLETE))
                ++residual.tx_awaiting_ack;
            if (!entry.second.local_retired)
                ++residual.tx_awaiting_local_retire;
        }
    }
    return residual;
}
