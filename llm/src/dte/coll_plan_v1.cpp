#include "dte/coll_plan_v1.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace {

void Require(bool condition, const char *message) {
    if (!condition) throw std::invalid_argument(message);
}

uint64_t CheckedAdd(uint64_t a, uint64_t b, const char *message) {
    if (b > std::numeric_limits<uint64_t>::max() - a)
        throw std::overflow_error(message);
    return a + b;
}

uint64_t CheckedMultiply(uint64_t a, uint64_t b, const char *message) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
        throw std::overflow_error(message);
    return a * b;
}

void ValidateSpan(uint64_t base, uint64_t bytes, const char *message) {
    if (bytes != 0 && base > std::numeric_limits<uint64_t>::max() - (bytes - 1))
        throw std::overflow_error(message);
}

uint64_t DTypeBytes(CollDType dtype) {
    switch (dtype) {
    case CollDType::UINT8: return 1;
    case CollDType::INT32: return 4;
    case CollDType::INT64: return 8;
    case CollDType::FP32:
    case CollDType::FP16:
    case CollDType::FP8:
        break;
    }
    throw std::invalid_argument("ISA-v1 reduction dtype is unsupported");
}

bool IsSymmetric(CollOp op) {
    return op == CollOp::ALLTOALL || op == CollOp::ALLGATHER ||
           op == CollOp::REDUCESCATTER || op == CollOp::ALLREDUCE;
}

bool IsRootTransmit(CollOp op) {
    return op == CollOp::SCATTER || op == CollOp::BROADCAST;
}

bool IsRootReceive(CollOp op) {
    return op == CollOp::GATHER || op == CollOp::REDUCE;
}

bool ExpectedSend(CollOp op, const IsaV1CollectiveSpec &spec, size_t rank) {
    if (op == CollOp::P2P) return rank == spec.p2p_source_rank;
    if (IsRootTransmit(op)) return rank == spec.root_rank;
    if (IsRootReceive(op) || IsSymmetric(op)) return true;
    return false;
}

bool ExpectedReceive(CollOp op, const IsaV1CollectiveSpec &spec, size_t rank) {
    if (op == CollOp::P2P) return rank == spec.p2p_destination_rank;
    if (IsRootTransmit(op)) return true;
    if (IsRootReceive(op)) return rank == spec.root_rank;
    if (IsSymmetric(op)) return true;
    return false;
}

bool IsReduceTarget(CollOp op, const IsaV1CollectiveSpec &spec, size_t rank) {
    if (op == CollOp::REDUCE) return rank == spec.root_rank;
    return op == CollOp::REDUCESCATTER || op == CollOp::ALLREDUCE;
}

uint64_t SourceOffset(const IsaV1CollectiveSpec &spec, uint16_t dst_rank,
                      uint64_t chunk_offset) {
    if (spec.tx_kind == CollTxKind::SCATTER) {
        return CheckedAdd(
            CheckedMultiply(dst_rank, spec.length_bytes,
                            "ISA-v1 source rank*L overflows"),
            chunk_offset, "ISA-v1 source offset overflows");
    }
    return chunk_offset;
}

uint64_t DestinationOffset(const IsaV1CollectiveSpec &spec, uint16_t src_rank,
                           uint64_t chunk_offset) {
    if (spec.rx_kind == CollRxKind::GATHER ||
        spec.rx_kind == CollRxKind::REDUCE) {
        return CheckedAdd(
            CheckedMultiply(src_rank, spec.length_bytes,
                            "ISA-v1 destination rank*L overflows"),
            chunk_offset, "ISA-v1 destination offset overflows");
    }
    return chunk_offset;
}

struct ValidatedSpec {
    CollOp op = CollOp::P2P;
    uint64_t dtype_bytes = 1;
    uint64_t tight_layout_bytes = 0;
    uint64_t chunk_bytes = 0;
    uint64_t chunk_count = 0;
    uint64_t remote_pair_count = 0;
    uint64_t child_count = 0;
    uint64_t local_copy_count = 0;
    uint64_t reduce_target_count = 0;
    uint64_t wave_count = 0;
    uint64_t action_count = 0;
    uint64_t derived_bytes = 0;
};

void RequireWithin(uint64_t value, uint64_t limit, const char *message) {
    if (value > limit) throw std::invalid_argument(message);
}

void AddDerivedBytes(uint64_t count, size_t element_bytes, uint64_t *total) {
    const uint64_t bytes = CheckedMultiply(
        count, static_cast<uint64_t>(element_bytes),
        "ISA-v1 derived state byte count overflows");
    *total = CheckedAdd(*total, bytes,
                        "ISA-v1 derived state byte count overflows");
}

ValidatedSpec Validate(const IsaV1CollectiveSpec &spec,
                       const IsaV1PlannerCapacity &capacity) {
    ValidatedSpec out;
    out.op = IsaV1CollectiveOp(spec.tx_kind, spec.rx_kind);
    Require(!spec.group.empty(), "ISA-v1 collective group must not be empty");
    Require(spec.group.size() <= std::numeric_limits<uint16_t>::max(),
            "ISA-v1 collective group exceeds u16 rank/count capacity");
    Require(std::is_sorted(spec.group.begin(), spec.group.end()),
            "ISA-v1 collective group must be sorted");
    Require(std::adjacent_find(spec.group.begin(), spec.group.end()) ==
                spec.group.end(),
            "ISA-v1 collective group contains duplicate cores");
    Require(spec.root_rank < spec.group.size(),
            "ISA-v1 collective root is outside group");
    Require(spec.p2p_source_rank < spec.group.size() &&
                spec.p2p_destination_rank < spec.group.size(),
            "ISA-v1 P2P endpoint rank is outside group");
    Require(spec.length_bytes != 0,
            "ISA-v1 collective L must be positive");
    Require(spec.tree_id == 0,
            "ISA-v1 baseline supports tree_id=0 unicast only");
    Require(spec.logical_fsm_id_base != 0,
            "ISA-v1 collective child fsm base must be non-zero");
    Require(spec.rank_records.size() == spec.group.size(),
            "ISA-v1 collective requires one rank record contract per rank");
    Require(capacity.max_child_bytes != 0 &&
                capacity.max_receive_bytes_per_rank_per_wave != 0 &&
                capacity.max_sessions_per_rank_per_wave != 0 &&
                capacity.max_children != 0 && capacity.max_actions != 0 &&
                capacity.max_waves != 0 &&
                capacity.max_derived_bytes != 0,
            "ISA-v1 planner capacity must be positive");

    const size_t n = spec.group.size();
    const uint64_t base_action_count = CheckedMultiply(
        2, static_cast<uint64_t>(n),
        "ISA-v1 base action count overflows");
    RequireWithin(base_action_count, capacity.max_actions,
                  "ISA-v1 base action count exceeds planner capacity");
    uint64_t base_derived_bytes = 0;
    AddDerivedBytes(n, sizeof(uint16_t), &base_derived_bytes);
    AddDerivedBytes(n, sizeof(IsaV1RankRecordContract),
                    &base_derived_bytes);
    AddDerivedBytes(1, sizeof(IsaV1Wave), &base_derived_bytes);
    AddDerivedBytes(n, sizeof(std::vector<IsaV1Action>),
                    &base_derived_bytes);
    AddDerivedBytes(base_action_count, sizeof(IsaV1Action),
                    &base_derived_bytes);
    RequireWithin(base_derived_bytes, capacity.max_derived_bytes,
                  "ISA-v1 base derived state exceeds planner byte capacity");

    const uint16_t expected =
        (spec.rx_kind == CollRxKind::GATHER ||
         spec.rx_kind == CollRxKind::REDUCE)
            ? static_cast<uint16_t>(n - 1)
            : 0;
    Require(spec.expected_sources == expected,
            "ISA-v1 collective expected_sources must be N-1 for gather/reduce and zero for unicast");

    const bool reduction = CollIsReduction(out.op);
    if (reduction) {
        Require(spec.reduce_op == CollReduceOp::SUM ||
                    spec.reduce_op == CollReduceOp::MAX,
                "ISA-v1 reduction requires SUM or MAX");
        out.dtype_bytes = DTypeBytes(spec.dtype);
        Require(spec.length_bytes % out.dtype_bytes == 0,
                "ISA-v1 reduction L must be dtype aligned");
    } else {
        Require(spec.reduce_op == CollReduceOp::NONE,
                "ISA-v1 non-reduction forbids reduce_op");
        Require(spec.dtype == CollDType::UINT8,
                "ISA-v1 non-reduction data is byte-oriented UINT8");
    }

    out.tight_layout_bytes = CheckedMultiply(
        static_cast<uint64_t>(n), spec.length_bytes,
        "ISA-v1 tight N*L layout overflows");

    for (size_t rank = 0; rank < n; ++rank) {
        const auto &records = spec.rank_records[rank];
        const bool send = ExpectedSend(out.op, spec, rank);
        const bool receive = ExpectedReceive(out.op, spec, rank);
        Require(records.send.present == send,
                "ISA-v1 collective SEND role does not match the 3x3 contract");
        Require(records.receive.present == receive,
                "ISA-v1 collective RECV role does not match the 3x3 contract");
        if (send) {
            Require(records.send.asynchronous && records.send.token != 0,
                    "ISA-v1 collective SEND must be ASYNC with a non-zero aggregate token");
            const uint64_t footprint = spec.tx_kind == CollTxKind::SCATTER
                                           ? out.tight_layout_bytes
                                           : spec.length_bytes;
            ValidateSpan(records.send.base_address_bytes, footprint,
                         "ISA-v1 SEND address span overflows");
        } else {
            Require(records.send.token == 0 &&
                        records.send.base_address_bytes == 0,
                    "ISA-v1 absent SEND record has non-canonical fields");
        }
        if (receive) {
            Require(records.receive.asynchronous && records.receive.token != 0,
                    "ISA-v1 collective RECV must be ASYNC with a non-zero aggregate token");
            const uint64_t footprint = spec.rx_kind == CollRxKind::UNICAST
                                           ? spec.length_bytes
                                           : out.tight_layout_bytes;
            ValidateSpan(records.receive.base_address_bytes, footprint,
                         "ISA-v1 RECV address span overflows");
        } else {
            Require(records.receive.token == 0 &&
                        records.receive.base_address_bytes == 0,
                    "ISA-v1 absent RECV record has non-canonical fields");
        }
        Require(!(send && receive && records.send.token == records.receive.token),
                "ISA-v1 SEND and RECV aggregate tokens collide on one rank");

        const bool reduce_target = IsReduceTarget(out.op, spec, rank);
        if (reduce_target) {
            ValidateSpan(records.result_address_bytes, spec.length_bytes,
                         "ISA-v1 reduction result address span overflows");
        } else {
            Require(records.result_address_bytes == 0,
                    "ISA-v1 non-target rank carries a reduction result address");
        }

        if (reduction) {
            if (send)
                Require(records.send.base_address_bytes % out.dtype_bytes == 0,
                        "ISA-v1 reduction source is not dtype aligned");
            if (receive)
                Require(records.receive.base_address_bytes % out.dtype_bytes == 0,
                        "ISA-v1 reduction staging is not dtype aligned");
            if (reduce_target)
                Require(records.result_address_bytes % out.dtype_bytes == 0,
                        "ISA-v1 reduction result is not dtype aligned");
        }
    }

    out.chunk_bytes = std::min(capacity.max_child_bytes,
                               capacity.max_receive_bytes_per_rank_per_wave);
    if (reduction) {
        out.chunk_bytes -= out.chunk_bytes % out.dtype_bytes;
        Require(out.chunk_bytes != 0,
                "ISA-v1 reduction capacity cannot hold one dtype element");
    }
    out.chunk_count = (spec.length_bytes - 1) / out.chunk_bytes + 1;

    if (out.op == CollOp::P2P) {
        out.remote_pair_count =
            spec.p2p_source_rank == spec.p2p_destination_rank ? 0 : 1;
    } else if (IsRootTransmit(out.op) || IsRootReceive(out.op)) {
        out.remote_pair_count = n - 1;
    } else {
        out.remote_pair_count = CheckedMultiply(
            static_cast<uint64_t>(n), static_cast<uint64_t>(n - 1),
            "ISA-v1 symmetric pair count overflows");
    }
    out.child_count = CheckedMultiply(
        out.remote_pair_count, out.chunk_count,
        "ISA-v1 child flow count overflows");
    if (out.child_count > std::numeric_limits<uint32_t>::max())
        throw std::overflow_error("ISA-v1 child flow indices exceed u32");
    if (out.child_count != 0 &&
        out.child_count - 1 >
            std::numeric_limits<uint32_t>::max() - spec.logical_fsm_id_base)
        throw std::overflow_error("ISA-v1 child fsm range overflows u32");
    RequireWithin(out.child_count, capacity.max_children,
                  "ISA-v1 child flow count exceeds planner capacity");

    out.local_copy_count = out.op == CollOp::P2P
                               ? (out.remote_pair_count == 0 ? 1 : 0)
                               : (IsSymmetric(out.op)
                                      ? static_cast<uint64_t>(n)
                                      : 1);
    out.reduce_target_count =
        reduction && n > 1
            ? (out.op == CollOp::REDUCE ? 1 : static_cast<uint64_t>(n))
            : 0;

    const uint64_t minimum_actions = CheckedAdd(
        CheckedAdd(
            out.local_copy_count,
            CheckedMultiply(5, out.child_count,
                            "ISA-v1 minimum action count overflows"),
            "ISA-v1 minimum action count overflows"),
        CheckedAdd(
            CheckedMultiply(2, static_cast<uint64_t>(n),
                            "ISA-v1 minimum action count overflows"),
            out.reduce_target_count,
            "ISA-v1 minimum action count overflows"),
        "ISA-v1 minimum action count overflows");
    RequireWithin(minimum_actions, capacity.max_actions,
                  "ISA-v1 minimum action count exceeds planner capacity");

    uint64_t preflight_bytes = 0;
    AddDerivedBytes(n, sizeof(uint16_t), &preflight_bytes);
    AddDerivedBytes(n, sizeof(IsaV1RankRecordContract), &preflight_bytes);
    AddDerivedBytes(out.child_count, sizeof(IsaV1ChildFlow),
                    &preflight_bytes);
    AddDerivedBytes(out.local_copy_count, sizeof(IsaV1LocalCopy),
                    &preflight_bytes);
    AddDerivedBytes(out.reduce_target_count, sizeof(IsaV1ReduceTarget),
                    &preflight_bytes);
    AddDerivedBytes(1, sizeof(IsaV1Wave), &preflight_bytes);
    AddDerivedBytes(out.child_count, sizeof(uint32_t), &preflight_bytes);
    AddDerivedBytes(n, sizeof(std::vector<IsaV1Action>), &preflight_bytes);
    AddDerivedBytes(minimum_actions, sizeof(IsaV1Action),
                    &preflight_bytes);
    RequireWithin(preflight_bytes, capacity.max_derived_bytes,
                  "ISA-v1 minimum derived state exceeds planner byte capacity");

    uint64_t scratch_bytes = 0;
    AddDerivedBytes(n, sizeof(uint32_t), &scratch_bytes);
    AddDerivedBytes(n, sizeof(uint64_t), &scratch_bytes);
    RequireWithin(scratch_bytes, capacity.max_derived_bytes,
                  "ISA-v1 preflight scratch exceeds planner byte capacity");

    // Preflight the exact canonical greedy waves without materializing child
    // flows. The loop is bounded by max_children, which was checked above.
    std::vector<uint32_t> sessions(n, 0);
    std::vector<uint64_t> receive_bytes(n, 0);
    uint64_t children_in_wave = 0;
    auto admit = [&](uint16_t src, uint16_t dst, uint64_t bytes) {
        auto fits = [&]() {
            return sessions[src] < capacity.max_sessions_per_rank_per_wave &&
                   sessions[dst] < capacity.max_sessions_per_rank_per_wave &&
                   bytes <= capacity.max_receive_bytes_per_rank_per_wave -
                                receive_bytes[dst];
        };
        if (!fits()) {
            if (children_in_wave == 0)
                throw std::invalid_argument(
                    "ISA-v1 one child cannot fit planner capacity");
            out.wave_count = CheckedAdd(
                out.wave_count, 1, "ISA-v1 wave count overflows");
            RequireWithin(out.wave_count, capacity.max_waves,
                          "ISA-v1 wave count exceeds planner capacity");
            std::fill(sessions.begin(), sessions.end(), 0);
            std::fill(receive_bytes.begin(), receive_bytes.end(), 0);
            children_in_wave = 0;
        }
        if (!fits())
            throw std::invalid_argument(
                "ISA-v1 one child cannot fit an empty wave");
        ++sessions[src];
        ++sessions[dst];
        receive_bytes[dst] += bytes;
        ++children_in_wave;
    };
    auto admit_pairs = [&](uint64_t bytes) {
        const uint16_t count = static_cast<uint16_t>(n);
        if (out.op == CollOp::P2P) {
            if (spec.p2p_source_rank != spec.p2p_destination_rank)
                admit(spec.p2p_source_rank, spec.p2p_destination_rank, bytes);
            return;
        }
        if (IsRootTransmit(out.op)) {
            for (uint16_t dst = 0; dst < count; ++dst)
                if (spec.root_rank != dst) admit(spec.root_rank, dst, bytes);
            return;
        }
        if (IsRootReceive(out.op)) {
            for (uint16_t src = 0; src < count; ++src)
                if (src != spec.root_rank) admit(src, spec.root_rank, bytes);
            return;
        }
        for (uint16_t src = 0; src < count; ++src)
            for (uint16_t dst = 0; dst < count; ++dst)
                if (src != dst) admit(src, dst, bytes);
    };
    uint64_t preflight_chunk_offset = 0;
    while (preflight_chunk_offset < spec.length_bytes) {
        const uint64_t bytes = std::min(
            out.chunk_bytes, spec.length_bytes - preflight_chunk_offset);
        admit_pairs(bytes);
        preflight_chunk_offset += bytes;
    }
    if (children_in_wave != 0 || out.wave_count == 0) {
        out.wave_count =
            CheckedAdd(out.wave_count, 1, "ISA-v1 wave count overflows");
        RequireWithin(out.wave_count, capacity.max_waves,
                      "ISA-v1 wave count exceeds planner capacity");
    }

    constexpr uint64_t kMaxWaves =
        (static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1) / 2;
    if (out.wave_count > kMaxWaves)
        throw std::overflow_error("ISA-v1 two-phase wave IDs exceed u16");

    out.action_count = CheckedAdd(
        out.local_copy_count,
        CheckedMultiply(5, out.child_count,
                        "ISA-v1 action count overflows"),
        "ISA-v1 action count overflows");
    const uint64_t barriers = CheckedMultiply(
        CheckedMultiply(2, static_cast<uint64_t>(n),
                        "ISA-v1 barrier action count overflows"),
        out.wave_count, "ISA-v1 barrier action count overflows");
    out.action_count = CheckedAdd(out.action_count, barriers,
                                  "ISA-v1 action count overflows");
    out.action_count = CheckedAdd(out.action_count, out.reduce_target_count,
                                  "ISA-v1 action count overflows");
    RequireWithin(out.action_count, capacity.max_actions,
                  "ISA-v1 action count exceeds planner capacity");

    AddDerivedBytes(n, sizeof(uint16_t), &out.derived_bytes);
    AddDerivedBytes(n, sizeof(IsaV1RankRecordContract), &out.derived_bytes);
    AddDerivedBytes(out.child_count, sizeof(IsaV1ChildFlow),
                    &out.derived_bytes);
    AddDerivedBytes(out.local_copy_count, sizeof(IsaV1LocalCopy),
                    &out.derived_bytes);
    AddDerivedBytes(out.reduce_target_count, sizeof(IsaV1ReduceTarget),
                    &out.derived_bytes);
    AddDerivedBytes(out.wave_count, sizeof(IsaV1Wave), &out.derived_bytes);
    AddDerivedBytes(out.child_count, sizeof(uint32_t), &out.derived_bytes);
    AddDerivedBytes(n, sizeof(std::vector<IsaV1Action>),
                    &out.derived_bytes);
    AddDerivedBytes(out.action_count, sizeof(IsaV1Action),
                    &out.derived_bytes);
    RequireWithin(out.derived_bytes, capacity.max_derived_bytes,
                  "ISA-v1 derived state exceeds planner byte capacity");
    return out;
}

template <class F>
void ForEachLogicalPair(const IsaV1CollectiveSpec &spec, CollOp op, F &&fn) {
    const uint16_t n = static_cast<uint16_t>(spec.group.size());
    if (op == CollOp::P2P) {
        fn(spec.p2p_source_rank, spec.p2p_destination_rank);
        return;
    }
    if (IsRootTransmit(op)) {
        for (uint16_t dst = 0; dst < n; ++dst) fn(spec.root_rank, dst);
        return;
    }
    if (IsRootReceive(op)) {
        for (uint16_t src = 0; src < n; ++src) fn(src, spec.root_rank);
        return;
    }
    for (uint16_t src = 0; src < n; ++src)
        for (uint16_t dst = 0; dst < n; ++dst) fn(src, dst);
}

IsaV1Action Action(IsaV1ActionKind kind, size_t rank,
                   const IsaV1CollectiveSpec &spec, uint16_t wave,
                   uint16_t phase, uint32_t item = kIsaV1NoItem) {
    IsaV1Action out;
    out.kind = kind;
    out.rank = static_cast<uint16_t>(rank);
    out.core = spec.group[rank];
    out.wave_index = wave;
    out.phase_id = phase;
    out.item_index = item;
    return out;
}

} // namespace

bool IsaV1EndpointRecord::operator==(const IsaV1EndpointRecord &o) const {
    return std::tie(present, asynchronous, token, base_address_bytes) ==
           std::tie(o.present, o.asynchronous, o.token, o.base_address_bytes);
}

bool IsaV1RankRecordContract::operator==(
    const IsaV1RankRecordContract &o) const {
    return std::tie(send, receive, result_address_bytes) ==
           std::tie(o.send, o.receive, o.result_address_bytes);
}

bool IsaV1ChildFlow::operator==(const IsaV1ChildFlow &o) const {
    return std::tie(fsm_id, chunk_id, source_rank, destination_rank,
                    source_core, destination_core, chunk_offset_bytes,
                    length_bytes, source_offset_bytes,
                    destination_offset_bytes, source_address_bytes,
                    destination_address_bytes, source_public_token,
                    destination_public_token, wave_index) ==
           std::tie(o.fsm_id, o.chunk_id, o.source_rank, o.destination_rank,
                    o.source_core, o.destination_core, o.chunk_offset_bytes,
                    o.length_bytes, o.source_offset_bytes,
                    o.destination_offset_bytes, o.source_address_bytes,
                    o.destination_address_bytes, o.source_public_token,
                    o.destination_public_token, o.wave_index);
}

bool IsaV1LocalCopy::operator==(const IsaV1LocalCopy &o) const {
    return std::tie(rank, core, length_bytes, source_offset_bytes,
                    destination_offset_bytes, source_address_bytes,
                    destination_address_bytes) ==
           std::tie(o.rank, o.core, o.length_bytes, o.source_offset_bytes,
                    o.destination_offset_bytes, o.source_address_bytes,
                    o.destination_address_bytes);
}

bool IsaV1ReduceTarget::operator==(const IsaV1ReduceTarget &o) const {
    return std::tie(rank, core, staging_address_bytes, result_address_bytes,
                    length_bytes, element_count, input_count, dtype,
                    reduce_op) ==
           std::tie(o.rank, o.core, o.staging_address_bytes,
                    o.result_address_bytes, o.length_bytes, o.element_count,
                    o.input_count, o.dtype, o.reduce_op);
}

bool IsaV1Wave::operator==(const IsaV1Wave &o) const {
    return std::tie(wave_index, posted_phase_id, complete_phase_id,
                    child_indices) ==
           std::tie(o.wave_index, o.posted_phase_id, o.complete_phase_id,
                    o.child_indices);
}

bool IsaV1Action::operator==(const IsaV1Action &o) const {
    return std::tie(kind, rank, core, wave_index, phase_id, item_index) ==
           std::tie(o.kind, o.rank, o.core, o.wave_index, o.phase_id,
                    o.item_index);
}

bool IsaV1CollectivePlan::operator==(const IsaV1CollectivePlan &o) const {
    return std::tie(op, key, group, root_rank, length_bytes,
                    logical_fsm_id_base, rank_records, child_flows,
                    local_copies, reduce_targets, waves, actions_by_rank) ==
           std::tie(o.op, o.key, o.group, o.root_rank, o.length_bytes,
                    o.logical_fsm_id_base, o.rank_records, o.child_flows,
                    o.local_copies, o.reduce_targets, o.waves,
                    o.actions_by_rank);
}

CollOp IsaV1CollectiveOp(CollTxKind tx_kind, CollRxKind rx_kind) {
    switch (tx_kind) {
    case CollTxKind::UNICAST:
        switch (rx_kind) {
        case CollRxKind::UNICAST: return CollOp::P2P;
        case CollRxKind::GATHER: return CollOp::GATHER;
        case CollRxKind::REDUCE: return CollOp::REDUCE;
        }
        break;
    case CollTxKind::SCATTER:
        switch (rx_kind) {
        case CollRxKind::UNICAST: return CollOp::SCATTER;
        case CollRxKind::GATHER: return CollOp::ALLTOALL;
        case CollRxKind::REDUCE: return CollOp::REDUCESCATTER;
        }
        break;
    case CollTxKind::BROADCAST:
        switch (rx_kind) {
        case CollRxKind::UNICAST: return CollOp::BROADCAST;
        case CollRxKind::GATHER: return CollOp::ALLGATHER;
        case CollRxKind::REDUCE: return CollOp::ALLREDUCE;
        }
        break;
    }
    throw std::invalid_argument("unknown ISA-v1 collective 3x3 kind");
}

IsaV1CollectivePlan PlanIsaV1Collective(
    const IsaV1CollectiveSpec &spec, const IsaV1PlannerCapacity &capacity) {
    const ValidatedSpec valid = Validate(spec, capacity);
    const size_t n = spec.group.size();

    IsaV1CollectivePlan plan;
    plan.op = valid.op;
    plan.key = spec.key;
    plan.group = spec.group;
    plan.root_rank = spec.root_rank;
    plan.length_bytes = spec.length_bytes;
    plan.logical_fsm_id_base = spec.logical_fsm_id_base;
    plan.rank_records = spec.rank_records;
    plan.actions_by_rank.resize(n);
    plan.child_flows.reserve(static_cast<size_t>(valid.child_count));
    plan.local_copies.reserve(static_cast<size_t>(valid.local_copy_count));
    plan.reduce_targets.reserve(static_cast<size_t>(valid.reduce_target_count));
    plan.waves.reserve(static_cast<size_t>(valid.wave_count));

    // Diagonal pairs never allocate endpoint state; they become exact local
    // copies.  For N=1 reductions the copy bypasses staging entirely.
    ForEachLogicalPair(spec, valid.op, [&](uint16_t src, uint16_t dst) {
        if (src != dst) return;
        IsaV1LocalCopy copy;
        copy.rank = src;
        copy.core = spec.group[src];
        copy.length_bytes = spec.length_bytes;
        copy.source_offset_bytes = SourceOffset(spec, dst, 0);
        copy.destination_offset_bytes = DestinationOffset(spec, src, 0);
        copy.source_address_bytes = CheckedAdd(
            spec.rank_records[src].send.base_address_bytes,
            copy.source_offset_bytes,
            "ISA-v1 local-copy source address overflows");
        if (CollIsReduction(valid.op) && n == 1) {
            copy.destination_offset_bytes = 0;
            copy.destination_address_bytes =
                spec.rank_records[dst].result_address_bytes;
        } else {
            copy.destination_address_bytes = CheckedAdd(
                spec.rank_records[dst].receive.base_address_bytes,
                copy.destination_offset_bytes,
                "ISA-v1 local-copy destination address overflows");
        }
        plan.local_copies.push_back(copy);
    });

    uint64_t chunk_offset = 0;
    uint32_t chunk_id = 0;
    while (chunk_offset < spec.length_bytes) {
        const uint64_t bytes =
            std::min(valid.chunk_bytes, spec.length_bytes - chunk_offset);
        ForEachLogicalPair(spec, valid.op, [&](uint16_t src, uint16_t dst) {
            if (src == dst) return;
            IsaV1ChildFlow flow;
            const uint32_t ordinal =
                static_cast<uint32_t>(plan.child_flows.size());
            flow.fsm_id = spec.logical_fsm_id_base + ordinal;
            flow.chunk_id = chunk_id;
            flow.source_rank = src;
            flow.destination_rank = dst;
            flow.source_core = spec.group[src];
            flow.destination_core = spec.group[dst];
            flow.chunk_offset_bytes = chunk_offset;
            flow.length_bytes = bytes;
            flow.source_offset_bytes = SourceOffset(spec, dst, chunk_offset);
            flow.destination_offset_bytes =
                DestinationOffset(spec, src, chunk_offset);
            flow.source_address_bytes = CheckedAdd(
                spec.rank_records[src].send.base_address_bytes,
                flow.source_offset_bytes,
                "ISA-v1 child source address overflows");
            flow.destination_address_bytes = CheckedAdd(
                spec.rank_records[dst].receive.base_address_bytes,
                flow.destination_offset_bytes,
                "ISA-v1 child destination address overflows");
            flow.source_public_token = spec.rank_records[src].send.token;
            flow.destination_public_token =
                spec.rank_records[dst].receive.token;
            plan.child_flows.push_back(flow);
        });
        chunk_offset += bytes;
        ++chunk_id;
    }
    if (plan.child_flows.size() != valid.child_count)
        throw std::logic_error("ISA-v1 internal child count mismatch");

    // Canonical greedy waves.  A wave never exceeds either endpoint-session
    // capacity or aggregate receive bytes on any rank.
    std::vector<uint32_t> sessions(n, 0);
    std::vector<uint64_t> receive_bytes(n, 0);
    IsaV1Wave current;
    auto reset_wave = [&]() {
        current = IsaV1Wave{};
        std::fill(sessions.begin(), sessions.end(), 0);
        std::fill(receive_bytes.begin(), receive_bytes.end(), 0);
    };
    reset_wave();
    for (uint32_t index = 0; index < plan.child_flows.size(); ++index) {
        const auto &flow = plan.child_flows[index];
        auto fits = [&]() {
            return sessions[flow.source_rank] <
                       capacity.max_sessions_per_rank_per_wave &&
                   sessions[flow.destination_rank] <
                       capacity.max_sessions_per_rank_per_wave &&
                   flow.length_bytes <=
                       capacity.max_receive_bytes_per_rank_per_wave -
                           receive_bytes[flow.destination_rank];
        };
        if (!fits()) {
            if (current.child_indices.empty())
                throw std::invalid_argument(
                    "ISA-v1 one child cannot fit planner capacity");
            plan.waves.push_back(std::move(current));
            reset_wave();
        }
        if (!fits())
            throw std::invalid_argument(
                "ISA-v1 one child cannot fit an empty wave");
        current.child_indices.push_back(index);
        ++sessions[flow.source_rank];
        ++sessions[flow.destination_rank];
        receive_bytes[flow.destination_rank] += flow.length_bytes;
    }
    if (!current.child_indices.empty() || plan.waves.empty())
        plan.waves.push_back(std::move(current));

    if (plan.waves.size() != valid.wave_count)
        throw std::logic_error("ISA-v1 internal wave count mismatch");
    for (size_t wave = 0; wave < plan.waves.size(); ++wave) {
        auto &entry = plan.waves[wave];
        entry.wave_index = static_cast<uint16_t>(wave);
        entry.posted_phase_id = static_cast<uint16_t>(wave * 2);
        entry.complete_phase_id = static_cast<uint16_t>(wave * 2 + 1);
        for (uint32_t index : entry.child_indices)
            plan.child_flows[index].wave_index = entry.wave_index;
    }

    if (CollIsReduction(valid.op) && n > 1) {
        for (size_t rank = 0; rank < n; ++rank) {
            if (!IsReduceTarget(valid.op, spec, rank)) continue;
            IsaV1ReduceTarget target;
            target.rank = static_cast<uint16_t>(rank);
            target.core = spec.group[rank];
            target.staging_address_bytes =
                spec.rank_records[rank].receive.base_address_bytes;
            target.result_address_bytes =
                spec.rank_records[rank].result_address_bytes;
            target.length_bytes = spec.length_bytes;
            target.element_count = spec.length_bytes / valid.dtype_bytes;
            target.input_count = static_cast<uint16_t>(n);
            target.dtype = spec.dtype;
            target.reduce_op = spec.reduce_op;
            plan.reduce_targets.push_back(target);
        }
    }

    if (plan.local_copies.size() != valid.local_copy_count ||
        plan.reduce_targets.size() != valid.reduce_target_count)
        throw std::logic_error("ISA-v1 internal derived item count mismatch");

    const uint64_t per_rank_barriers = CheckedMultiply(
        2, valid.wave_count, "ISA-v1 rank barrier count overflows");
    std::vector<uint64_t> rank_action_counts(n, per_rank_barriers);
    for (const auto &copy : plan.local_copies)
        rank_action_counts[copy.rank] = CheckedAdd(
            rank_action_counts[copy.rank], 1,
            "ISA-v1 rank action count overflows");
    for (const auto &target : plan.reduce_targets)
        rank_action_counts[target.rank] = CheckedAdd(
            rank_action_counts[target.rank], 1,
            "ISA-v1 rank action count overflows");
    for (const auto &flow : plan.child_flows) {
        rank_action_counts[flow.source_rank] = CheckedAdd(
            rank_action_counts[flow.source_rank], 3,
            "ISA-v1 rank action count overflows");
        rank_action_counts[flow.destination_rank] = CheckedAdd(
            rank_action_counts[flow.destination_rank], 2,
            "ISA-v1 rank action count overflows");
    }
    uint64_t reserved_actions = 0;
    for (size_t rank = 0; rank < n; ++rank) {
        reserved_actions = CheckedAdd(
            reserved_actions, rank_action_counts[rank],
            "ISA-v1 reserved action count overflows");
        plan.actions_by_rank[rank].reserve(
            static_cast<size_t>(rank_action_counts[rank]));
    }
    if (reserved_actions != valid.action_count)
        throw std::logic_error("ISA-v1 internal reserved action mismatch");

    for (uint32_t index = 0; index < plan.local_copies.size(); ++index) {
        const auto &copy = plan.local_copies[index];
        plan.actions_by_rank[copy.rank].push_back(Action(
            IsaV1ActionKind::LOCAL_COPY, copy.rank, spec, 0, 0, index));
    }

    for (size_t wave_index = 0; wave_index < plan.waves.size(); ++wave_index) {
        const auto &wave = plan.waves[wave_index];
        const bool final_wave = wave_index + 1 == plan.waves.size();
        for (uint32_t index : wave.child_indices) {
            const auto rank = plan.child_flows[index].destination_rank;
            plan.actions_by_rank[rank].push_back(Action(
                IsaV1ActionKind::POST_RECEIVE, rank, spec, wave.wave_index,
                wave.posted_phase_id, index));
        }
        for (size_t rank = 0; rank < n; ++rank)
            plan.actions_by_rank[rank].push_back(Action(
                IsaV1ActionKind::POSTED_BARRIER, rank, spec,
                wave.wave_index, wave.posted_phase_id));
        for (uint32_t index : wave.child_indices) {
            const auto rank = plan.child_flows[index].source_rank;
            plan.actions_by_rank[rank].push_back(Action(
                IsaV1ActionKind::ISSUE_SEND, rank, spec, wave.wave_index,
                wave.posted_phase_id, index));
        }
        for (uint32_t index : wave.child_indices) {
            const auto rank = plan.child_flows[index].destination_rank;
            plan.actions_by_rank[rank].push_back(Action(
                IsaV1ActionKind::WAIT_RECEIVE, rank, spec, wave.wave_index,
                wave.complete_phase_id, index));
        }
        for (uint32_t index : wave.child_indices) {
            const auto rank = plan.child_flows[index].source_rank;
            plan.actions_by_rank[rank].push_back(Action(
                IsaV1ActionKind::WAIT_SEND, rank, spec, wave.wave_index,
                wave.complete_phase_id, index));
        }
        if (final_wave) {
            for (uint32_t index = 0; index < plan.reduce_targets.size();
                 ++index) {
                const auto rank = plan.reduce_targets[index].rank;
                plan.actions_by_rank[rank].push_back(Action(
                    IsaV1ActionKind::REDUCE_COMPUTE, rank, spec,
                    wave.wave_index, wave.complete_phase_id, index));
            }
        }
        for (uint32_t index : wave.child_indices) {
            const auto rank = plan.child_flows[index].source_rank;
            plan.actions_by_rank[rank].push_back(Action(
                IsaV1ActionKind::WAIT_TRANSPORT_RETIRE, rank, spec,
                wave.wave_index, wave.complete_phase_id, index));
        }
        for (size_t rank = 0; rank < n; ++rank)
            plan.actions_by_rank[rank].push_back(Action(
                IsaV1ActionKind::COMPLETE_BARRIER, rank, spec,
                wave.wave_index, wave.complete_phase_id));
    }
    uint64_t materialized_actions = 0;
    for (const auto &actions : plan.actions_by_rank)
        materialized_actions = CheckedAdd(
            materialized_actions, actions.size(),
            "ISA-v1 materialized action count overflows");
    if (materialized_actions != valid.action_count)
        throw std::logic_error("ISA-v1 internal action count mismatch");
    return plan;
}
