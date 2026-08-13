#pragma once

#include "dte/coll_types.h"
#include "dte/endpoint_contract.h"
#include "dte/p2p_payload.h"
#include "systemc.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

inline constexpr uint16_t ISA_V1_COLL_BYTE_START_MAGIC = 0xd710;
inline constexpr uint16_t ISA_V1_COLL_BYTE_DATA_MAGIC = 0xd711;
inline constexpr uint8_t ISA_V1_COLL_BYTE_VERSION = 0;
inline constexpr uint8_t ISA_V1_COLL_BYTE_START_FLAGS = 0;

inline constexpr uint8_t IsaV1CollectiveByteLegacyRawType(
    uint16_t magic) noexcept {
    return static_cast<uint8_t>((magic >> 1) & 0xf);
}

static_assert(P2P_PAYLOAD_FRAGMENT_BYTES == 16,
              "collective DATA wire requires a 16-byte P2P fragment");
static_assert(P2P_PAYLOAD_MAX_FRAGMENTS == UINT16_MAX,
              "collective DATA sequence must match the P2P u16 bound");
static_assert(IsaV1CollectiveByteLegacyRawType(
                  ISA_V1_COLL_BYTE_START_MAGIC) >= MSG_TYPE_NUM,
              "collective START magic must be invalid as a legacy Msg");
static_assert(IsaV1CollectiveByteLegacyRawType(
                  ISA_V1_COLL_BYTE_DATA_MAGIC) >= MSG_TYPE_NUM,
              "collective DATA magic must be invalid as a legacy Msg");

enum class IsaV1CollectiveByteKind : uint8_t {
    MULTICAST = 0,
    DCA_REDUCE = 1,
};

struct IsaV1CollectiveByteLock {
    uint16_t tree_id = 0;
    uint16_t session_id = 0;
    uint32_t epoch = 0;
    bool operator==(const IsaV1CollectiveByteLock &other) const noexcept;
    bool operator<(const IsaV1CollectiveByteLock &other) const noexcept;
};

struct IsaV1CollectiveByteStart {
    IsaV1CollectiveByteKind kind = IsaV1CollectiveByteKind::MULTICAST;
    uint16_t tree_id = 0;
    uint16_t session_id = 0;
    CollectiveKey collective;
    uint32_t total_bytes = 0;
    uint32_t checksum = 0;
    uint8_t flags = ISA_V1_COLL_BYTE_START_FLAGS;
};

struct IsaV1CollectiveByteData {
    IsaV1CollectiveByteKind kind = IsaV1CollectiveByteKind::MULTICAST;
    IsaV1CollectiveByteLock lock;
    uint16_t sequence = 0;
    uint8_t length_bytes = 0;
    bool tail = false;
    sc_bv<128> payload = 0;
};

struct IsaV1CollectiveByteBuildSpec {
    IsaV1CollectiveByteKind kind = IsaV1CollectiveByteKind::MULTICAST;
    uint16_t tree_id = 0;
    uint16_t session_id = 0;
    CollectiveKey collective;
};

struct IsaV1CollectiveByteBuiltStream {
    sc_bv<256> start = 0;
    std::vector<sc_bv<256>> data;
};

struct IsaV1CollectiveByteCommit {
    IsaV1CollectiveByteBuildSpec identity;
    std::vector<uint8_t> bytes;
};

bool IsIsaV1CollectiveByteStartWire(const sc_bv<256> &wire) noexcept;
bool IsIsaV1CollectiveByteDataWire(const sc_bv<256> &wire) noexcept;
sc_bv<256> SerializeIsaV1CollectiveByteStart(
    const IsaV1CollectiveByteStart &start);
IsaV1CollectiveByteStart DeserializeIsaV1CollectiveByteStart(
    const sc_bv<256> &wire);
sc_bv<256> SerializeIsaV1CollectiveByteData(
    const IsaV1CollectiveByteData &data);
IsaV1CollectiveByteData DeserializeIsaV1CollectiveByteData(
    const sc_bv<256> &wire);

// Router-facing inspection consumes no START sideband: every DATA flit
// contains kind plus its complete tree/session/epoch branch-lock identity.
IsaV1CollectiveByteData InspectIsaV1CollectiveByteDataWire(
    const sc_bv<256> &wire);

IsaV1CollectiveByteBuiltStream BuildIsaV1CollectiveByteStream(
    const IsaV1CollectiveByteBuildSpec &spec,
    const std::vector<uint8_t> &bytes);

class IsaV1CollectiveByteReassembler final {
public:
    IsaV1CollectiveByteReassembler(size_t max_buffered_bytes,
                                   size_t max_inflight_streams);
    void Begin(const sc_bv<256> &start_wire);
    std::optional<IsaV1CollectiveByteCommit> Accept(
        const sc_bv<256> &data_wire);
    bool Abort(const IsaV1CollectiveByteLock &lock) noexcept;
    bool HasActive(const IsaV1CollectiveByteLock &lock) const noexcept;
    size_t InflightStreams() const noexcept { return states_.size(); }
    size_t ReservedBytes() const noexcept { return reserved_bytes_; }
    size_t ResidualCapacityBytes() const noexcept {
        return max_buffered_bytes_ - reserved_bytes_;
    }

private:
    struct State {
        IsaV1CollectiveByteStart start;
        uint16_t next_sequence = 1;
        uint32_t fragment_count = 0;
        std::vector<uint8_t> bytes;
    };
    size_t max_buffered_bytes_;
    size_t max_inflight_streams_;
    size_t reserved_bytes_ = 0;
    std::map<IsaV1CollectiveByteLock, State> states_;
    std::map<CollectiveKey, IsaV1CollectiveByteLock> key_owners_;
};
