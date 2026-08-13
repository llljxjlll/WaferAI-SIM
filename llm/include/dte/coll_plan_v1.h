#pragma once

#include "dte/coll_types.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

// ISA-v1 collectives are lowered into ordinary point-to-point endpoint
// sessions.  This planner is deliberately independent of the endpoint
// runtime: callers supply its capacity limits and receive a canonical,
// deterministic list of child flows and per-rank actions.
inline constexpr uint64_t kIsaV1DefaultMaxChildBytes = 1048560ULL;
inline constexpr uint64_t kIsaV1DefaultMaxChildren = 1048576ULL;
inline constexpr uint64_t kIsaV1DefaultMaxActions = 1048576ULL;
inline constexpr uint64_t kIsaV1DefaultMaxWaves = 32768ULL;
inline constexpr uint64_t kIsaV1DefaultMaxDerivedBytes = 1048576ULL;
inline constexpr uint32_t kIsaV1NoItem =
    std::numeric_limits<uint32_t>::max();

struct IsaV1EndpointRecord {
    bool present = false;
    bool asynchronous = true;
    uint32_t token = 0;
    uint64_t base_address_bytes = 0;

    bool operator==(const IsaV1EndpointRecord &o) const;
};

// One entry per group rank, representing the matched artifact records on that
// rank.  result_address_bytes is used only on ranks that perform a reduction.
struct IsaV1RankRecordContract {
    IsaV1EndpointRecord send;
    IsaV1EndpointRecord receive;
    uint64_t result_address_bytes = 0;

    bool operator==(const IsaV1RankRecordContract &o) const;
};

struct IsaV1CollectiveSpec {
    CollTxKind tx_kind = CollTxKind::UNICAST;
    CollRxKind rx_kind = CollRxKind::UNICAST;
    CollectiveKey key;
    std::vector<uint16_t> group;
    uint16_t root_rank = 0;

    // Used only by UNICAST x UNICAST.  A same-rank pair is a local copy.
    uint16_t p2p_source_rank = 0;
    uint16_t p2p_destination_rank = 0;

    // L: bytes contributed by one source to one destination/result.  SCATTER
    // sources and GATHER/REDUCE destinations use a tight N*L layout.
    uint64_t length_bytes = 0;
    CollDType dtype = CollDType::UINT8;
    CollReduceOp reduce_op = CollReduceOp::NONE;

    // P6 baseline is deterministic unicast expansion only.
    uint16_t tree_id = 0;
    uint16_t expected_sources = 0;
    uint32_t logical_fsm_id_base = 0;

    std::vector<IsaV1RankRecordContract> rank_records;
};

struct IsaV1PlannerCapacity {
    uint64_t max_child_bytes = kIsaV1DefaultMaxChildBytes;
    uint64_t max_receive_bytes_per_rank_per_wave =
        kIsaV1DefaultMaxChildBytes;
    uint32_t max_sessions_per_rank_per_wave = 64;

    // Hard pre-allocation limits for planner-derived state. The byte limit
    // accounts for variable-size payloads owned by the returned plan; child
    // indices stored in waves are counted independently of child flows.
    uint64_t max_children = kIsaV1DefaultMaxChildren;
    uint64_t max_actions = kIsaV1DefaultMaxActions;
    uint64_t max_waves = kIsaV1DefaultMaxWaves;
    uint64_t max_derived_bytes = kIsaV1DefaultMaxDerivedBytes;
};

struct IsaV1ChildFlow {
    uint32_t fsm_id = 0;
    uint32_t chunk_id = 0;
    uint16_t source_rank = 0;
    uint16_t destination_rank = 0;
    uint16_t source_core = 0;
    uint16_t destination_core = 0;
    uint64_t chunk_offset_bytes = 0;
    uint64_t length_bytes = 0;
    uint64_t source_offset_bytes = 0;
    uint64_t destination_offset_bytes = 0;
    uint64_t source_address_bytes = 0;
    uint64_t destination_address_bytes = 0;
    uint32_t source_public_token = 0;
    uint32_t destination_public_token = 0;
    uint16_t wave_index = 0;

    bool operator==(const IsaV1ChildFlow &o) const;
};

struct IsaV1LocalCopy {
    uint16_t rank = 0;
    uint16_t core = 0;
    uint64_t length_bytes = 0;
    uint64_t source_offset_bytes = 0;
    uint64_t destination_offset_bytes = 0;
    uint64_t source_address_bytes = 0;
    uint64_t destination_address_bytes = 0;

    bool operator==(const IsaV1LocalCopy &o) const;
};

struct IsaV1ReduceTarget {
    uint16_t rank = 0;
    uint16_t core = 0;
    uint64_t staging_address_bytes = 0;
    uint64_t result_address_bytes = 0;
    uint64_t length_bytes = 0;
    uint64_t element_count = 0;
    uint16_t input_count = 0;
    CollDType dtype = CollDType::UINT8;
    CollReduceOp reduce_op = CollReduceOp::NONE;

    bool operator==(const IsaV1ReduceTarget &o) const;
};

struct IsaV1Wave {
    uint16_t wave_index = 0;
    uint16_t posted_phase_id = 0;
    uint16_t complete_phase_id = 0;
    std::vector<uint32_t> child_indices;

    bool operator==(const IsaV1Wave &o) const;
};

enum class IsaV1ActionKind : uint8_t {
    LOCAL_COPY = 0,
    POST_RECEIVE = 1,
    POSTED_BARRIER = 2,
    ISSUE_SEND = 3,
    WAIT_RECEIVE = 4,
    WAIT_SEND = 5,
    REDUCE_COMPUTE = 6,
    WAIT_TRANSPORT_RETIRE = 7,
    COMPLETE_BARRIER = 8
};

struct IsaV1Action {
    IsaV1ActionKind kind = IsaV1ActionKind::POSTED_BARRIER;
    uint16_t rank = 0;
    uint16_t core = 0;
    uint16_t wave_index = 0;
    uint16_t phase_id = 0;
    uint32_t item_index = kIsaV1NoItem;

    bool operator==(const IsaV1Action &o) const;
};

struct IsaV1CollectivePlan {
    CollOp op = CollOp::P2P;
    CollectiveKey key;
    std::vector<uint16_t> group;
    uint16_t root_rank = 0;
    uint64_t length_bytes = 0;
    uint32_t logical_fsm_id_base = 0;
    std::vector<IsaV1RankRecordContract> rank_records;
    std::vector<IsaV1ChildFlow> child_flows;
    std::vector<IsaV1LocalCopy> local_copies;
    std::vector<IsaV1ReduceTarget> reduce_targets;
    std::vector<IsaV1Wave> waves;
    std::vector<std::vector<IsaV1Action>> actions_by_rank;

    bool operator==(const IsaV1CollectivePlan &o) const;
};

CollOp IsaV1CollectiveOp(CollTxKind tx_kind, CollRxKind rx_kind);

// Throws invalid_argument for contract/capacity errors and overflow_error for
// arithmetic, fsm-range, child-count, or phase-space overflow.
IsaV1CollectivePlan PlanIsaV1Collective(
    const IsaV1CollectiveSpec &spec,
    const IsaV1PlannerCapacity &capacity = IsaV1PlannerCapacity{});
