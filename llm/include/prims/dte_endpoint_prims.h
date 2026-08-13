#pragma once

#include "dte/endpoint_contract.h"
#include "prims/base.h"

#include <cstddef>
#include <cstdint>
#include <string>

inline constexpr uint8_t kDteEndpointPrimWireVersion = 1;
inline constexpr uint8_t kDteEndpointPrimBaseSegments = 5;
inline constexpr uint8_t kDteEndpointPrimRegionBytesPerSegment = 14;
inline constexpr std::size_t kDteEndpointPrimRegionMaxBytes = 64;
inline constexpr std::size_t kDteEndpointPrimMaxSegments =
    kDteEndpointPrimBaseSegments +
    (kDteEndpointPrimRegionMaxBytes +
     kDteEndpointPrimRegionBytesPerSegment - 1) /
        kDteEndpointPrimRegionBytesPerSegment;

// Strict wire v1 uses a common 16-bit segment header: [7:0] PrimId and
// [15:8] ordinal. Segment 0 carries version/count/enums/address kind and the
// DTE_SEND-only source space;
// segment 1 carries fsm/token/peer/source-count/tree; segment 2 length;
// segment 3 the selected absolute address or region offset; segment 4 the
// collective key and region byte count. Optional segments 5..9 carry 14
// region-name bytes each. All unspecified bits and tail bytes are zero.

// Internal endpoint enums intentionally mirror the frozen external record
// values without including the external opcode layer in Worker Prim headers.
enum class DteEndpointCompletion : uint8_t { ASYNC = 0, SYNC = 1 };
enum class DteEndpointSourceSpace : uint8_t { SRAM = 0, HBM = 1 };
enum class DteEndpointDataType : uint8_t { UINT8 = 0, INT32 = 1, INT64 = 2 };
enum class DteEndpointReduceOp : uint8_t { NONE = 0, SUM = 1, MAX = 2 };
enum class DteEndpointSendMode : uint8_t {
    P2P = 0,
    SCATTER = 1,
    BROADCAST = 2,
};
enum class DteEndpointRecvMode : uint8_t {
    P2P = 0,
    GATHER = 1,
    REDUCE = 2,
};
enum class DteEndpointAddressKind : uint8_t {
    ABSOLUTE = 1,
    REGION = 2,
};

struct DteEndpointSramAddress {
    DteEndpointAddressKind kind = DteEndpointAddressKind::ABSOLUTE;
    uint64_t absolute_address_bytes = 0;
    std::string region;
    uint64_t region_offset_bytes = 0;
};

class Dte_endpoint_prim_base : public PrimBase {
public:
    DteEndpointCompletion completion = DteEndpointCompletion::ASYNC;
    DteEndpointDataType datatype = DteEndpointDataType::UINT8;
    DteEndpointReduceOp reduce_op = DteEndpointReduceOp::NONE;
    uint32_t fsm_id = 1;
    uint32_t token = 1;
    uint64_t length_bytes = 1;
    uint16_t peer_core = 0;
    uint16_t expected_sources = 0;
    uint16_t tree_id = 0;
    uint32_t group_id = 0;
    uint32_t collective_id = 0;
    uint32_t epoch = 0;

protected:
    Dte_endpoint_prim_base() { setPrimMainCategory(COMM_PRIM); }
};

class Dte_send_endpoint_prim final : public Dte_endpoint_prim_base {
public:
    DteEndpointSendMode mode = DteEndpointSendMode::P2P;
    DteEndpointSourceSpace source_space = DteEndpointSourceSpace::SRAM;
    DteEndpointSramAddress source;

    void Validate() const;
    int taskCoreDefault(TaskCoreContext &context) override;
    vector<sc_bv<128>> serialize() override;
    void deserialize(vector<sc_bv<128>> buffer) override;
    void printSelf() override;

    Dte_send_endpoint_prim() { name = "Dte_send_endpoint_prim"; }
};

class Dte_recv_endpoint_prim final : public Dte_endpoint_prim_base {
public:
    DteEndpointRecvMode mode = DteEndpointRecvMode::P2P;
    DteEndpointSramAddress destination;

    void Validate() const;
    int taskCoreDefault(TaskCoreContext &context) override;
    vector<sc_bv<128>> serialize() override;
    void deserialize(vector<sc_bv<128>> buffer) override;
    void printSelf() override;

    Dte_recv_endpoint_prim() { name = "Dte_recv_endpoint_prim"; }
};
