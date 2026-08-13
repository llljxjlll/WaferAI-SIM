#pragma once

#include "dte/coll_plan_v1.h"

#include <cstddef>
#include <cstdint>
#include <vector>

enum class IsaV1CollectiveRecordRole : uint8_t {
    SEND = 0,
    RECEIVE = 1,
    REDUCE_COMPUTE = 2
};

// Loader-facing, already decoded and address-relocated view.  SEND uses
// tx_mode/base_address_bytes; RECEIVE uses rx_mode/base_address_bytes;
// REDUCE_COMPUTE uses base_address_bytes as staging and result_address_bytes
// as its final destination.
struct IsaV1NormalizedCollectiveRecord {
    uint16_t core_id = 0;
    uint32_t record_index = 0;
    IsaV1CollectiveRecordRole role = IsaV1CollectiveRecordRole::SEND;
    CollTxKind tx_mode = CollTxKind::UNICAST;
    CollRxKind rx_mode = CollRxKind::UNICAST;
    bool asynchronous = true;
    uint32_t token = 0;
    uint32_t logical_fsm_id_base = 0;
    uint64_t length_bytes = 0;
    // REDUCE_COMPUTE-only; element_count*dtype_bytes must equal L.
    uint64_t element_count = 0;
    CollDType dtype = CollDType::UINT8;
    CollReduceOp reduce_op = CollReduceOp::NONE;
    uint64_t base_address_bytes = 0;
    uint64_t result_address_bytes = 0;
    CollectiveKey key;
    uint16_t tree_id = 0;
    uint16_t expected_sources = 0;
    // Endpoint-only peer. Fully-zero-key P2P SEND/RECEIVE records are
    // paired into one session-level FSM reservation.
    uint16_t peer_core = 0;
    // REDUCE_COMPUTE-only external rank contract. Its L and logical FSM base
    // are synthesized from the matching RECEIVE by the whole-artifact graph.
    uint16_t root_rank = 0;
    uint16_t self_rank = 0;
};

struct IsaV1CollectiveGroupDefinition {
    uint32_t group_id = 0;
    std::vector<uint16_t> members;
};

// Immutable input view supplied by the artifact loader.  The graph builder
// validates the view as part of one transactional Build call.
struct IsaV1CollectiveGroupRegistryView {
    uint32_t total_cores = 0;
    uint32_t cores_per_die = 0;
    std::vector<uint16_t> active_cores;
    std::vector<IsaV1CollectiveGroupDefinition> groups;
};

// Whole-artifact pass.  Fully-zero-key P2P records are excluded from plans,
// but matched SEND/RECEIVE sessions still reserve their per-core aggregate
// tokens and one singleton fsm ID per session.
// Returned plans are ordered by CollectiveKey, independent of input order.
std::vector<IsaV1CollectivePlan> BuildIsaV1CollectiveGraph(
    const std::vector<IsaV1NormalizedCollectiveRecord> &records,
    const IsaV1CollectiveGroupRegistryView &registry,
    const IsaV1PlannerCapacity &capacity = IsaV1PlannerCapacity{});
