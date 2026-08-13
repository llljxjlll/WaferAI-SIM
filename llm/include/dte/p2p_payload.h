#pragma once

#include "common/msg.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <vector>

inline constexpr size_t P2P_PAYLOAD_FRAGMENT_BYTES = 16;
inline constexpr uint8_t P2P_ENDPOINT_MSG_MARKER = 0xa5;
inline constexpr uint64_t P2P_PAYLOAD_MAX_FRAGMENTS = UINT16_MAX;

struct P2pFlowKey {
    uint16_t source = 0;
    uint16_t destination = 0;
    uint16_t transport_tag = 0;
    uint8_t subflow = 0;

    bool operator==(const P2pFlowKey &other) const noexcept;
    bool operator<(const P2pFlowKey &other) const noexcept;
};

struct P2pPayloadDeclaration {
    P2pFlowKey flow;
    uint32_t fsm_id = 0;
    uint64_t total_bytes = 0;
    uint32_t fragment_count = 0;
    uint32_t checksum = 0;
};

struct P2pDataFragment {
    P2pFlowKey flow;
    uint16_t sequence = 0;
    bool tail = false;
    uint8_t length_bytes = 0;
    std::array<uint8_t, P2P_PAYLOAD_FRAGMENT_BYTES> bytes{};
};

struct P2pBuiltPayload {
    Msg request;
    std::vector<Msg> fragments;
};

struct P2pPayloadCommit {
    P2pFlowKey flow;
    std::vector<uint8_t> bytes;
};

uint32_t P2pPayloadChecksum(const uint8_t *bytes, size_t size) noexcept;
uint32_t P2pPayloadChecksum(const std::vector<uint8_t> &bytes) noexcept;

// REQUEST carries total payload bits in data_[127:64], the full endpoint
// fsm_id in data_[63:32], and whole-flow CRC32C in data_[31:0]. The Msg
// tag_id_ is only a runtime-allocated non-zero transport tag. DATA
// data_[127:0] is exclusively business payload.
P2pBuiltPayload BuildP2pPayload(const P2pFlowKey &flow, uint32_t fsm_id,
                                const std::vector<uint8_t> &bytes);
P2pPayloadDeclaration ParseP2pPayloadRequest(const Msg &request);
P2pDataFragment ParseP2pDataFragment(const Msg &fragment);

class P2pPayloadReassembler final {
public:
    explicit P2pPayloadReassembler(size_t max_buffered_bytes,
                                   size_t max_inflight_flows);

    void Begin(const Msg &request);
    std::optional<P2pPayloadCommit> Accept(const Msg &fragment);
    bool Abort(const P2pFlowKey &flow) noexcept;

    bool HasActive(const P2pFlowKey &flow) const noexcept;
    size_t InflightFlows() const noexcept { return states_.size(); }
    size_t ReservedBytes() const noexcept { return reserved_bytes_; }
    size_t ResidualCapacityBytes() const noexcept {
        return max_buffered_bytes_ - reserved_bytes_;
    }
    size_t MaxBufferedBytes() const noexcept { return max_buffered_bytes_; }
    size_t MaxInflightFlows() const noexcept { return max_inflight_flows_; }

private:
    struct State {
        P2pPayloadDeclaration declaration;
        uint32_t next_sequence = 1;
        std::vector<uint8_t> bytes;
    };

    size_t max_buffered_bytes_;
    size_t max_inflight_flows_;
    size_t reserved_bytes_ = 0;
    std::map<P2pFlowKey, State> states_;
};

struct P2pTimingKey {
    P2pFlowKey flow;
    uint64_t round = 0;

    bool operator==(const P2pTimingKey &other) const noexcept;
};

// Simulation-only timing. This type deliberately has no byte or Msg payload.
struct P2pTimingMetadata {
    uint64_t source_first_ns = 0;
    uint64_t source_done_ns = 0;
    uint32_t network_tail_cycles = 0;

    bool operator==(const P2pTimingMetadata &other) const noexcept;
};

class P2pTimingSidebandRegistry final {
public:
    explicit P2pTimingSidebandRegistry(size_t capacity);

    void Publish(const P2pTimingKey &key,
                 const P2pTimingMetadata &metadata);
    P2pTimingMetadata Consume(const P2pTimingKey &key);
    const P2pTimingKey &FrontKey() const;

    bool Contains(const P2pTimingKey &key) const noexcept;
    bool Empty() const noexcept { return fifo_.empty(); }
    bool Full() const noexcept { return fifo_.size() == capacity_; }
    size_t Residual() const noexcept { return fifo_.size(); }
    size_t Capacity() const noexcept { return capacity_; }

private:
    struct Entry {
        P2pTimingKey key;
        P2pTimingMetadata metadata;
    };

    size_t capacity_;
    std::deque<Entry> fifo_;
};

// Process-wide simulation timing registry shared by source Worker, D2D and
// destination Worker. It carries timing metadata only: business bytes remain
// exclusively in Msg::data_. The endpoint session allocator must keep a
// transport tag allocated until the destination ACK, so an active P2pFlowKey
// identifies exactly one transport generation.
class P2pSharedTimingSidebandRuntime final {
public:
    P2pSharedTimingSidebandRuntime() = delete;

    // Configure changes the active-flow bound and starts a new round history.
    // Reconfiguration is rejected while entries are active. Reset preserves
    // the configured capacity but clears all active entries and stale-round
    // history, and is intended for a program/simulation boundary.
    static void Configure(size_t capacity);
    static void Reset();

    // Source transition: publish the complete source interval exactly once.
    static void Publish(const P2pTimingKey &key, uint32_t fsm_id,
                        uint64_t source_first_ns,
                        uint64_t source_done_ns);

    // D2D transition: complete the unique active flow's network timing. The
    // value may be zero; completion is represented by separate state.
    static void UpdateNetworkTail(const P2pFlowKey &flow,
                                  uint32_t network_tail_cycles);

    // Destination ingress transition: derive a real, non-negative tail
    // duration from the published source completion and observed arrival.
    static void CompleteNetworkAtNs(const P2pFlowKey &flow,
                                    uint64_t arrival_ns,
                                    uint64_t cycle_ns);

    // Destination transition: consume only after the network transition and
    // only when the REQUEST-declared full fsm_id matches.
    static P2pTimingMetadata Consume(const P2pFlowKey &flow,
                                     uint32_t fsm_id);

    // Failure transition: discard an exact active generation without exposing
    // timing. The retired source round remains stale, so delayed work cannot
    // revive the aborted generation after its transport tag is released.
    static void Abort(const P2pFlowKey &flow, uint32_t fsm_id);
    static bool AbortFlow(const P2pFlowKey &flow) noexcept;

    static bool Contains(const P2pFlowKey &flow);
    static size_t Residual();
    static size_t Capacity();
};
