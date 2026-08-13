#include "dte/p2p_payload.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace {

constexpr uint32_t kCrc32cPolynomial = 0x82f63b78U;

void ValidateFlow(const P2pFlowKey &flow) {
    if (flow.transport_tag == 0)
        throw std::invalid_argument("P2P transport tag must be non-zero");
    if (flow.subflow > 3)
        throw std::invalid_argument("P2P payload subflow exceeds 2-bit wire");
}

P2pFlowKey FlowFromMsg(const Msg &msg) {
    if (msg.source_ < 0 || msg.source_ > UINT16_MAX || msg.des_ < 0 ||
        msg.des_ > UINT16_MAX || msg.tag_id_ <= 0 || msg.tag_id_ > UINT16_MAX ||
        msg.subflow_ < 0 || msg.subflow_ > 3)
        throw std::invalid_argument("P2P payload flow identity exceeds wire width");
    return {static_cast<uint16_t>(msg.source_),
            static_cast<uint16_t>(msg.des_),
            static_cast<uint16_t>(msg.tag_id_),
            static_cast<uint8_t>(msg.subflow_)};
}

uint64_t FragmentCount(uint64_t total_bytes) {
    return total_bytes / P2P_PAYLOAD_FRAGMENT_BYTES +
           (total_bytes % P2P_PAYLOAD_FRAGMENT_BYTES != 0 ? 1 : 0);
}

void PutBytes(Msg &msg, const uint8_t *bytes, size_t size) {
    msg.data_ = 0;
    for (size_t i = 0; i < size; ++i)
        msg.data_.range(static_cast<int>(i * 8 + 7), static_cast<int>(i * 8)) =
            sc_bv<8>(bytes[i]);
}

std::array<uint8_t, P2P_PAYLOAD_FRAGMENT_BYTES> GetBytes(const Msg &msg) {
    std::array<uint8_t, P2P_PAYLOAD_FRAGMENT_BYTES> bytes{};
    for (size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = static_cast<uint8_t>(
            msg.data_.range(static_cast<int>(i * 8 + 7),
                            static_cast<int>(i * 8))
                .to_uint64());
    return bytes;
}

void ValidateRequestEnvelope(const Msg &msg) {
    if (!msg.p2p_endpoint_ || msg.msg_type_ != MSG_TYPE::REQUEST)
        throw std::invalid_argument("P2P payload declaration is not REQUEST");
    if (msg.is_end_ || msg.seq_id_ != 0 ||
        msg.offset_ != P2P_ENDPOINT_MSG_MARKER ||
        msg.length_ != 0 || msg.refill_ || msg.config_end_ ||
        msg.roofline_packets_ != 0 || msg.event_tag_ != 0 ||
        msg.dte_stream_source_first_ns_ != 0 ||
        msg.dte_stream_source_done_ns_ != 0 ||
        msg.dte_stream_network_tail_cycles_ != 0 ||
        msg.exit_port_ < -1 || msg.exit_port_ >= static_cast<int>(UINT16_MAX))
        throw std::invalid_argument("P2P REQUEST envelope is non-canonical");
}

void ValidateDataEnvelope(const Msg &msg) {
    if (!msg.p2p_endpoint_ || msg.msg_type_ != MSG_TYPE::DATA)
        throw std::invalid_argument("P2P payload fragment is not DATA");
    if (msg.seq_id_ <= 0 || msg.seq_id_ > UINT16_MAX ||
        msg.offset_ != P2P_ENDPOINT_MSG_MARKER || msg.refill_ ||
        msg.config_end_ || msg.roofline_packets_ != 1 ||
        msg.flow_packets_ != 0 || msg.dte_payload_bits_ != 0 ||
        msg.event_tag_ != 0 || msg.exit_port_ < -1 ||
        msg.exit_port_ >= static_cast<int>(UINT16_MAX))
        throw std::invalid_argument("P2P DATA envelope is non-canonical");
    // DeserializeMsg mirrors DATA payload bits into the legacy timestamp
    // members. They are intentionally ignored: data_ remains authoritative.
}

} // namespace

bool P2pFlowKey::operator==(const P2pFlowKey &other) const noexcept {
    return std::tie(source, destination, transport_tag, subflow) ==
           std::tie(other.source, other.destination, other.transport_tag,
                    other.subflow);
}

bool P2pFlowKey::operator<(const P2pFlowKey &other) const noexcept {
    return std::tie(source, destination, transport_tag, subflow) <
           std::tie(other.source, other.destination, other.transport_tag,
                    other.subflow);
}

uint32_t P2pPayloadChecksum(const uint8_t *bytes, size_t size) noexcept {
    uint32_t checksum = UINT32_MAX;
    for (size_t i = 0; i < size; ++i) {
        checksum ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit)
            checksum = (checksum >> 1) ^
                       ((checksum & 1U) ? kCrc32cPolynomial : 0U);
    }
    return ~checksum;
}

uint32_t P2pPayloadChecksum(const std::vector<uint8_t> &bytes) noexcept {
    return P2pPayloadChecksum(bytes.data(), bytes.size());
}

P2pBuiltPayload BuildP2pPayload(const P2pFlowKey &flow, uint32_t fsm_id,
                                const std::vector<uint8_t> &bytes) {
    ValidateFlow(flow);
    if (fsm_id == 0)
        throw std::invalid_argument("P2P endpoint fsm_id must be non-zero");
    if (bytes.empty())
        throw std::invalid_argument("P2P payload must contain at least one byte");
    if (bytes.size() > UINT64_MAX / 8)
        throw std::overflow_error("P2P payload bit length overflows uint64");

    const uint64_t fragment_count = FragmentCount(bytes.size());
    if (fragment_count == 0 || fragment_count > P2P_PAYLOAD_MAX_FRAGMENTS)
        throw std::length_error("P2P payload exceeds DATA sequence space");
    if (fragment_count > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        throw std::length_error("P2P fragment count exceeds Msg representation");

    const uint64_t total_bits = static_cast<uint64_t>(bytes.size()) * 8;
    const uint32_t checksum = P2pPayloadChecksum(bytes);

    P2pBuiltPayload built;
    built.request = Msg{};
    built.request.data_ = 0;
    built.request.is_end_ = false;
    built.request.msg_type_ = MSG_TYPE::REQUEST;
    built.request.seq_id_ = 0;
    built.request.des_ = flow.destination;
    built.request.offset_ = P2P_ENDPOINT_MSG_MARKER;
    built.request.tag_id_ = flow.transport_tag;
    built.request.source_ = flow.source;
    built.request.length_ = 0;
    built.request.refill_ = false;
    built.request.config_end_ = false;
    built.request.roofline_packets_ = 0;
    built.request.flow_packets_ = static_cast<int>(fragment_count);
    built.request.dte_payload_bits_ = total_bits;
    built.request.dte_stream_source_first_ns_ = 0;
    built.request.dte_stream_source_done_ns_ = 0;
    built.request.dte_stream_network_tail_cycles_ = 0;
    built.request.event_tag_ = 0;
    built.request.subflow_ = flow.subflow;
    built.request.exit_port_ = -1;
    built.request.p2p_endpoint_ = true;
    built.request.data_.range(31, 0) = sc_bv<32>(checksum);
    built.request.data_.range(63, 32) = sc_bv<32>(fsm_id);
    built.request.data_.range(127, 64) = sc_bv<64>(total_bits);

    built.fragments.reserve(static_cast<size_t>(fragment_count));
    for (uint64_t index = 0; index < fragment_count; ++index) {
        const size_t begin = static_cast<size_t>(index) *
                             P2P_PAYLOAD_FRAGMENT_BYTES;
        const size_t length = std::min(P2P_PAYLOAD_FRAGMENT_BYTES,
                                       bytes.size() - begin);
        Msg fragment;
        fragment.data_ = 0;
        fragment.is_end_ = index + 1 == fragment_count;
        fragment.msg_type_ = MSG_TYPE::DATA;
        fragment.seq_id_ = static_cast<int>(index + 1);
        fragment.des_ = flow.destination;
        fragment.offset_ = P2P_ENDPOINT_MSG_MARKER;
        fragment.tag_id_ = flow.transport_tag;
        fragment.source_ = flow.source;
        fragment.length_ = static_cast<int>(length * 8);
        fragment.refill_ = false;
        fragment.config_end_ = false;
        fragment.roofline_packets_ = 1;
        fragment.flow_packets_ = 0;
        fragment.dte_payload_bits_ = 0;
        fragment.dte_stream_source_first_ns_ = 0;
        fragment.dte_stream_source_done_ns_ = 0;
        fragment.dte_stream_network_tail_cycles_ = 0;
        fragment.event_tag_ = 0;
        fragment.subflow_ = flow.subflow;
        fragment.exit_port_ = -1;
        fragment.p2p_endpoint_ = true;
        PutBytes(fragment, bytes.data() + begin, length);
        built.fragments.push_back(std::move(fragment));
    }
    return built;
}

P2pPayloadDeclaration ParseP2pPayloadRequest(const Msg &request) {
    ValidateRequestEnvelope(request);
    const P2pFlowKey flow = FlowFromMsg(request);
    if (request.dte_payload_bits_ == 0 || request.dte_payload_bits_ % 8 != 0)
        throw std::invalid_argument(
            "P2P REQUEST total payload bits must be positive whole bytes");
    if (request.data_.range(127, 64).to_uint64() !=
        request.dte_payload_bits_)
        throw std::invalid_argument("P2P REQUEST total bits metadata mismatch");
    const uint32_t fsm_id = static_cast<uint32_t>(
        request.data_.range(63, 32).to_uint64());
    if (fsm_id == 0)
        throw std::invalid_argument("P2P REQUEST fsm_id must be non-zero");

    const uint64_t total_bytes = request.dte_payload_bits_ / 8;
    const uint64_t fragment_count = FragmentCount(total_bytes);
    if (fragment_count == 0 || fragment_count > P2P_PAYLOAD_MAX_FRAGMENTS)
        throw std::length_error("P2P REQUEST exceeds DATA sequence space");
    if (request.flow_packets_ <= 0 ||
        static_cast<uint64_t>(request.flow_packets_) != fragment_count)
        throw std::invalid_argument("P2P REQUEST fragment count mismatch");

    return {flow,
            fsm_id,
            total_bytes,
            static_cast<uint32_t>(fragment_count),
            static_cast<uint32_t>(
                request.data_.range(31, 0).to_uint64())};
}

P2pDataFragment ParseP2pDataFragment(const Msg &fragment) {
    ValidateDataEnvelope(fragment);
    const P2pFlowKey flow = FlowFromMsg(fragment);
    if (fragment.length_ <= 0 || fragment.length_ > 128 ||
        fragment.length_ % 8 != 0)
        throw std::invalid_argument(
            "P2P DATA length must be 1..16 whole bytes");
    const uint8_t length = static_cast<uint8_t>(fragment.length_ / 8);
    if (!fragment.is_end_ && length != P2P_PAYLOAD_FRAGMENT_BYTES)
        throw std::invalid_argument("P2P non-tail DATA must contain 16 bytes");

    const auto bytes = GetBytes(fragment);
    for (size_t i = length; i < bytes.size(); ++i) {
        if (bytes[i] != 0)
            throw std::invalid_argument("P2P DATA padding is non-zero");
    }
    return {flow,
            static_cast<uint16_t>(fragment.seq_id_),
            fragment.is_end_,
            length,
            bytes};
}

P2pPayloadReassembler::P2pPayloadReassembler(size_t max_buffered_bytes,
                                             size_t max_inflight_flows)
    : max_buffered_bytes_(max_buffered_bytes),
      max_inflight_flows_(max_inflight_flows) {
    if (max_buffered_bytes == 0 || max_inflight_flows == 0)
        throw std::invalid_argument("P2P reassembler limits must be non-zero");
}

void P2pPayloadReassembler::Begin(const Msg &request) {
    const P2pPayloadDeclaration declaration =
        ParseP2pPayloadRequest(request);
    if (states_.find(declaration.flow) != states_.end())
        throw std::invalid_argument("duplicate P2P REQUEST for active flow");
    if (states_.size() >= max_inflight_flows_)
        throw std::length_error("P2P reassembler inflight flow capacity exhausted");
    if (declaration.total_bytes > std::numeric_limits<size_t>::max())
        throw std::length_error("P2P payload cannot fit host address space");
    const size_t total_bytes = static_cast<size_t>(declaration.total_bytes);
    if (total_bytes > max_buffered_bytes_ - reserved_bytes_)
        throw std::length_error("P2P reassembler byte capacity exhausted");

    State state;
    state.declaration = declaration;
    state.bytes.reserve(total_bytes);
    const auto inserted = states_.emplace(declaration.flow, std::move(state));
    if (!inserted.second)
        throw std::logic_error("P2P flow insertion unexpectedly failed");
    reserved_bytes_ += total_bytes;
}

std::optional<P2pPayloadCommit>
P2pPayloadReassembler::Accept(const Msg &fragment_msg) {
    const P2pDataFragment fragment = ParseP2pDataFragment(fragment_msg);
    auto found = states_.find(fragment.flow);
    if (found == states_.end())
        throw std::invalid_argument("P2P DATA has no active REQUEST");
    State &state = found->second;
    if (fragment.sequence != state.next_sequence)
        throw std::invalid_argument(
            "P2P DATA sequence is out of order or duplicate: source=" +
            std::to_string(fragment.flow.source) +
            " destination=" +
            std::to_string(fragment.flow.destination) +
            " tag=" + std::to_string(fragment.flow.transport_tag) +
            " expected=" + std::to_string(state.next_sequence) +
            " actual=" + std::to_string(fragment.sequence));

    const uint64_t prior_bytes = state.bytes.size();
    const uint64_t remaining = state.declaration.total_bytes - prior_bytes;
    const bool expected_tail =
        state.next_sequence == state.declaration.fragment_count;
    const uint8_t expected_length = static_cast<uint8_t>(
        std::min<uint64_t>(P2P_PAYLOAD_FRAGMENT_BYTES, remaining));
    if (fragment.tail != expected_tail)
        throw std::invalid_argument("P2P DATA tail flag mismatch");
    if (fragment.length_bytes != expected_length)
        throw std::invalid_argument("P2P DATA byte length mismatch");

    state.bytes.insert(state.bytes.end(), fragment.bytes.begin(),
                       fragment.bytes.begin() + fragment.length_bytes);
    ++state.next_sequence;
    if (!expected_tail)
        return std::nullopt;

    if (state.bytes.size() != state.declaration.total_bytes)
        throw std::logic_error("P2P reassembler completed with wrong byte count");
    const size_t reservation = static_cast<size_t>(state.declaration.total_bytes);
    if (P2pPayloadChecksum(state.bytes) != state.declaration.checksum) {
        states_.erase(found);
        reserved_bytes_ -= reservation;
        throw std::invalid_argument("P2P payload checksum mismatch");
    }

    P2pPayloadCommit commit{state.declaration.flow, std::move(state.bytes)};
    states_.erase(found);
    reserved_bytes_ -= reservation;
    return commit;
}

bool P2pPayloadReassembler::Abort(const P2pFlowKey &flow) noexcept {
    auto found = states_.find(flow);
    if (found == states_.end())
        return false;
    reserved_bytes_ -= static_cast<size_t>(found->second.declaration.total_bytes);
    states_.erase(found);
    return true;
}

bool P2pPayloadReassembler::HasActive(const P2pFlowKey &flow) const noexcept {
    return states_.find(flow) != states_.end();
}

bool P2pTimingKey::operator==(const P2pTimingKey &other) const noexcept {
    return flow == other.flow && round == other.round;
}

bool P2pTimingMetadata::operator==(
    const P2pTimingMetadata &other) const noexcept {
    return source_first_ns == other.source_first_ns &&
           source_done_ns == other.source_done_ns &&
           network_tail_cycles == other.network_tail_cycles;
}

P2pTimingSidebandRegistry::P2pTimingSidebandRegistry(size_t capacity)
    : capacity_(capacity) {
    if (capacity == 0)
        throw std::invalid_argument("P2P timing sideband capacity must be non-zero");
}

void P2pTimingSidebandRegistry::Publish(
    const P2pTimingKey &key, const P2pTimingMetadata &metadata) {
    ValidateFlow(key.flow);
    if (metadata.source_first_ns > metadata.source_done_ns)
        throw std::invalid_argument("P2P timing interval is reversed");
    if (Contains(key))
        throw std::invalid_argument("duplicate P2P timing sideband key");
    if (Full())
        throw std::length_error("P2P timing sideband capacity exhausted");
    fifo_.push_back({key, metadata});
}

P2pTimingMetadata
P2pTimingSidebandRegistry::Consume(const P2pTimingKey &key) {
    ValidateFlow(key.flow);
    if (fifo_.empty())
        throw std::out_of_range("P2P timing sideband is empty");
    if (!(fifo_.front().key == key))
        throw std::invalid_argument("P2P timing sideband FIFO key mismatch");
    P2pTimingMetadata metadata = fifo_.front().metadata;
    fifo_.pop_front();
    return metadata;
}

const P2pTimingKey &P2pTimingSidebandRegistry::FrontKey() const {
    if (fifo_.empty())
        throw std::out_of_range("P2P timing sideband is empty");
    return fifo_.front().key;
}

bool P2pTimingSidebandRegistry::Contains(
    const P2pTimingKey &key) const noexcept {
    return std::any_of(fifo_.begin(), fifo_.end(),
                       [&key](const Entry &entry) { return entry.key == key; });
}

namespace {

struct SharedTimingEntry {
    P2pTimingKey key;
    uint32_t fsm_id = 0;
    P2pTimingMetadata metadata;
    bool network_complete = false;
};

struct SharedTimingState {
    std::mutex mutex;
    size_t capacity = 0;
    std::map<P2pFlowKey, SharedTimingEntry> active;
    // Endpoint session rounds are monotonically allocated per source runtime.
    // A fixed wire-sized table rejects already-published source rounds without
    // creating an unbounded retired-flow/tombstone collection.
    std::array<uint64_t, static_cast<size_t>(UINT16_MAX) + 1>
        latest_source_round{};
};

SharedTimingState &SharedTiming() {
    static SharedTimingState state;
    return state;
}

void RequireSharedTimingConfigured(const SharedTimingState &state) {
    if (state.capacity == 0)
        throw std::logic_error(
            "P2P shared timing sideband is not configured");
}

} // namespace

void P2pSharedTimingSidebandRuntime::Configure(size_t capacity) {
    if (capacity == 0)
        throw std::invalid_argument(
            "P2P shared timing sideband capacity must be non-zero");
    SharedTimingState &state = SharedTiming();
    const std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.active.empty())
        throw std::logic_error(
            "cannot reconfigure P2P shared timing sideband with residual entries");
    state.capacity = capacity;
    state.latest_source_round.fill(0);
}

void P2pSharedTimingSidebandRuntime::Reset() {
    SharedTimingState &state = SharedTiming();
    const std::lock_guard<std::mutex> lock(state.mutex);
    state.active.clear();
    state.latest_source_round.fill(0);
}

void P2pSharedTimingSidebandRuntime::Publish(
    const P2pTimingKey &key, uint32_t fsm_id, uint64_t source_first_ns,
    uint64_t source_done_ns) {
    ValidateFlow(key.flow);
    if (fsm_id == 0)
        throw std::invalid_argument(
            "P2P shared timing fsm_id must be non-zero");
    if (key.round == 0)
        throw std::invalid_argument(
            "P2P shared timing round must be non-zero");
    if (source_first_ns > source_done_ns)
        throw std::invalid_argument(
            "P2P shared timing source interval is reversed");

    SharedTimingState &state = SharedTiming();
    const std::lock_guard<std::mutex> lock(state.mutex);
    RequireSharedTimingConfigured(state);
    if (state.active.find(key.flow) != state.active.end())
        throw std::invalid_argument(
            "duplicate active P2P shared timing flow");
    uint64_t &latest_round =
        state.latest_source_round[static_cast<size_t>(key.flow.source)];
    if (key.round <= latest_round)
        throw std::invalid_argument(
            "stale P2P shared timing source round");
    if (state.active.size() >= state.capacity)
        throw std::length_error(
            "P2P shared timing sideband capacity exhausted");

    SharedTimingEntry entry;
    entry.key = key;
    entry.fsm_id = fsm_id;
    entry.metadata.source_first_ns = source_first_ns;
    entry.metadata.source_done_ns = source_done_ns;
    const auto inserted = state.active.emplace(key.flow, std::move(entry));
    if (!inserted.second)
        throw std::logic_error(
            "P2P shared timing flow insertion unexpectedly failed");
    latest_round = key.round;
}

void P2pSharedTimingSidebandRuntime::UpdateNetworkTail(
    const P2pFlowKey &flow, uint32_t network_tail_cycles) {
    ValidateFlow(flow);
    SharedTimingState &state = SharedTiming();
    const std::lock_guard<std::mutex> lock(state.mutex);
    RequireSharedTimingConfigured(state);
    auto found = state.active.find(flow);
    if (found == state.active.end())
        throw std::out_of_range(
            "P2P shared timing update has no active source publication");
    if (found->second.network_complete)
        throw std::invalid_argument(
            "duplicate P2P shared timing network completion");
    found->second.metadata.network_tail_cycles = network_tail_cycles;
    found->second.network_complete = true;
}

void P2pSharedTimingSidebandRuntime::CompleteNetworkAtNs(
    const P2pFlowKey &flow, uint64_t arrival_ns, uint64_t cycle_ns) {
    ValidateFlow(flow);
    if (cycle_ns == 0)
        throw std::invalid_argument(
            "P2P shared timing cycle duration must be non-zero");
    SharedTimingState &state = SharedTiming();
    const std::lock_guard<std::mutex> lock(state.mutex);
    RequireSharedTimingConfigured(state);
    auto found = state.active.find(flow);
    if (found == state.active.end())
        throw std::out_of_range(
            "P2P shared timing completion has no active source publication");
    if (found->second.network_complete)
        throw std::invalid_argument(
            "duplicate P2P shared timing network completion");
    if (arrival_ns < found->second.metadata.source_done_ns)
        throw std::invalid_argument(
            "P2P shared timing network arrival precedes source completion");
    const uint64_t elapsed_ns =
        arrival_ns - found->second.metadata.source_done_ns;
    const uint64_t cycles =
        elapsed_ns / cycle_ns + (elapsed_ns % cycle_ns != 0 ? 1 : 0);
    if (cycles > UINT32_MAX)
        throw std::overflow_error(
            "P2P shared timing network tail exceeds 32-bit capacity");
    found->second.metadata.network_tail_cycles =
        static_cast<uint32_t>(cycles);
    found->second.network_complete = true;
}

P2pTimingMetadata P2pSharedTimingSidebandRuntime::Consume(
    const P2pFlowKey &flow, uint32_t fsm_id) {
    ValidateFlow(flow);
    if (fsm_id == 0)
        throw std::invalid_argument(
            "P2P shared timing fsm_id must be non-zero");
    SharedTimingState &state = SharedTiming();
    const std::lock_guard<std::mutex> lock(state.mutex);
    RequireSharedTimingConfigured(state);
    auto found = state.active.find(flow);
    if (found == state.active.end())
        throw std::out_of_range(
            "P2P shared timing consume has no active flow");
    if (found->second.fsm_id != fsm_id)
        throw std::invalid_argument(
            "P2P shared timing fsm_id mismatch");
    if (!found->second.network_complete)
        throw std::logic_error(
            "P2P shared timing consume precedes network completion");

    const P2pTimingMetadata metadata = found->second.metadata;
    state.active.erase(found);
    return metadata;
}

void P2pSharedTimingSidebandRuntime::Abort(const P2pFlowKey &flow,
                                           uint32_t fsm_id) {
    ValidateFlow(flow);
    if (fsm_id == 0)
        throw std::invalid_argument(
            "P2P shared timing abort fsm_id must be non-zero");
    SharedTimingState &state = SharedTiming();
    const std::lock_guard<std::mutex> lock(state.mutex);
    RequireSharedTimingConfigured(state);
    auto found = state.active.find(flow);
    if (found == state.active.end())
        throw std::out_of_range(
            "P2P shared timing abort has no active flow");
    if (found->second.fsm_id != fsm_id)
        throw std::invalid_argument(
            "P2P shared timing abort fsm_id mismatch");

    state.active.erase(found);
}

bool P2pSharedTimingSidebandRuntime::AbortFlow(
    const P2pFlowKey &flow) noexcept {
    SharedTimingState &state = SharedTiming();
    const std::lock_guard<std::mutex> lock(state.mutex);
    auto found = state.active.find(flow);
    if (found == state.active.end())
        return false;
    state.active.erase(found);
    return true;
}

bool P2pSharedTimingSidebandRuntime::Contains(const P2pFlowKey &flow) {
    SharedTimingState &state = SharedTiming();
    const std::lock_guard<std::mutex> lock(state.mutex);
    return state.active.find(flow) != state.active.end();
}

size_t P2pSharedTimingSidebandRuntime::Residual() {
    SharedTimingState &state = SharedTiming();
    const std::lock_guard<std::mutex> lock(state.mutex);
    return state.active.size();
}

size_t P2pSharedTimingSidebandRuntime::Capacity() {
    SharedTimingState &state = SharedTiming();
    const std::lock_guard<std::mutex> lock(state.mutex);
    return state.capacity;
}
