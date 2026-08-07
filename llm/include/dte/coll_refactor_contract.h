#pragma once

#include "dte/coll_latency.h"
#include "dte/coll_types.h"
#include "dte/coll_wire.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <tuple>

namespace coll_refactor {

enum class TraceGeneration : uint8_t {
    LEGACY_V5 = 0,
    REFACTOR = 1,
};

struct ReduceStreamKey {
    CollectiveKey collective;
    uint16_t phase_id = 0;
    uint32_t stream_id = 0;

    bool operator==(const ReduceStreamKey &other) const {
        return collective == other.collective && phase_id == other.phase_id &&
               stream_id == other.stream_id;
    }
    bool operator<(const ReduceStreamKey &other) const {
        return std::tie(collective, phase_id, stream_id) <
               std::tie(other.collective, other.phase_id, other.stream_id);
    }
};

struct ReduceBeatKey {
    ReduceStreamKey stream;
    uint16_t reduce_stage_id = 0;
    uint64_t vector_beat_id = 0;

    bool operator==(const ReduceBeatKey &other) const {
        return stream == other.stream &&
               reduce_stage_id == other.reduce_stage_id &&
               vector_beat_id == other.vector_beat_id;
    }
    bool operator<(const ReduceBeatKey &other) const {
        return std::tie(stream, reduce_stage_id, vector_beat_id) <
               std::tie(other.stream, other.reduce_stage_id,
                        other.vector_beat_id);
    }
};

struct DcaTiming {
    uint64_t latency = 0;
    uint64_t initiation_interval = 0;

    void Validate() const {
        if (latency == 0 || initiation_interval == 0)
            throw std::invalid_argument(
                "DCA latency and initiation interval must be positive");
    }
};

inline uint64_t DcaIssueCycle(uint64_t first_issue_cycle,
                              uint64_t issue_index,
                              const DcaTiming &timing) {
    timing.Validate();
    return CollCheckedAdd(
        first_issue_cycle,
        CollCheckedMul(issue_index, timing.initiation_interval));
}

inline uint64_t DcaCompletionCycle(uint64_t issue_cycle,
                                   const DcaTiming &timing) {
    timing.Validate();
    return CollCheckedAdd(issue_cycle, timing.latency);
}

struct VectorWork {
    uint64_t lanes = 0;
    uint64_t vector_beats = 0;
    uint64_t pairwise_issues_per_beat = 0;
    uint64_t total_issues = 0;
    uint64_t tail_valid_lanes = 0;
};

// A stream tail always uses a contiguous prefix of the vector lanes. Keeping
// the lane count beside the valid prefix makes the mask independent of a
// host-sized integer bitset and therefore valid for vector widths above 64.
struct VectorLaneMask {
    uint64_t lane_count = 0;
    uint64_t valid_lanes = 0;

    void Validate() const {
        if (lane_count == 0 || valid_lanes == 0 || valid_lanes > lane_count)
            throw std::invalid_argument("invalid vector lane mask");
    }
    bool IsActive(uint64_t lane) const {
        Validate();
        if (lane >= lane_count)
            throw std::out_of_range("vector lane index out of range");
        return lane < valid_lanes;
    }
    bool operator==(const VectorLaneMask &other) const {
        return lane_count == other.lane_count &&
               valid_lanes == other.valid_lanes;
    }
};

inline VectorLaneMask TailLaneMask(const VectorWork &work) {
    VectorLaneMask mask{work.lanes, work.tail_valid_lanes};
    mask.Validate();
    return mask;
}

// Logical vector-beat geometry. Value storage and arithmetic helpers are kept
// separate so a C++ element loop cannot accidentally become a timing model.
struct VectorBeat {
    CollDType dtype = CollDType::UINT8;
    uint64_t vector_bits = 0;
    VectorLaneMask lane_mask;

    void Validate() const {
        const uint64_t dtype_bits = CollDTypeBits(dtype);
        if (vector_bits == 0 || vector_bits % dtype_bits != 0)
            throw std::invalid_argument("invalid vector beat width");
        lane_mask.Validate();
        if (lane_mask.lane_count != vector_bits / dtype_bits)
            throw std::invalid_argument(
                "vector beat lane mask disagrees with dtype and width");
    }
    bool SameGeometry(const VectorBeat &other) const {
        return dtype == other.dtype && vector_bits == other.vector_bits &&
               lane_mask == other.lane_mask;
    }
};

inline VectorWork ComputeVectorWork(uint64_t total_elements,
                                    uint64_t input_count,
                                    uint64_t vector_bits,
                                    CollDType dtype) {
    if (total_elements == 0)
        throw std::invalid_argument("DCA vector work requires elements");
    if (input_count == 0)
        throw std::invalid_argument("DCA vector work requires an input");
    const uint64_t dtype_bits = CollDTypeBits(dtype);
    if (vector_bits == 0 || vector_bits % dtype_bits != 0)
        throw std::invalid_argument(
            "DCA vector width must contain whole dtype lanes");
    const uint64_t lanes = vector_bits / dtype_bits;
    const uint64_t beats = CollCeilDiv(total_elements, lanes);
    const uint64_t pairwise = input_count - 1;
    const uint64_t tail = total_elements % lanes;
    return {lanes, beats, pairwise, CollCheckedMul(beats, pairwise),
            tail == 0 ? lanes : tail};
}

struct DcaRequest {
    uint64_t tag = 0;
    ReduceBeatKey key;
    CollReduceOp op = CollReduceOp::NONE;
    std::array<VectorBeat, 2> operands;

    void Validate() const {
        if (tag == 0)
            throw std::invalid_argument("DCA request tag must be non-zero");
        if (op == CollReduceOp::NONE)
            throw std::invalid_argument("DCA request requires reduce op");
        operands[0].Validate();
        operands[1].Validate();
        if (!operands[0].SameGeometry(operands[1]))
            throw std::invalid_argument(
                "two DCA operands must have identical vector geometry");
    }
};

struct DcaResult {
    uint64_t tag = 0;
    ReduceBeatKey key;
    VectorBeat value;

    void ValidateAgainst(const DcaRequest &request) const {
        request.Validate();
        value.Validate();
        if (tag == 0 || tag != request.tag || !(key == request.key) ||
            !value.SameGeometry(request.operands[0]))
            throw std::invalid_argument(
                "DCA result does not match its tagged request");
    }
};

struct ReduceStreamHeader {
    ReduceWireVersion wire_version = ReduceWireVersion::STREAM_V2;
    uint16_t tree_id = 0;
    ReduceStreamKey key;
    CollDType dtype = CollDType::UINT8;
    CollReduceOp op = CollReduceOp::NONE;
    uint64_t total_elements = 0;
    uint64_t physical_data_flits = 0;
    uint64_t vector_beats = 0;

    void Validate(uint64_t physical_payload_bits,
                  uint64_t dca_vector_bits) const {
        if (wire_version != ReduceWireVersion::STREAM_V2 || tree_id == 0)
            throw std::invalid_argument("invalid reduce stream identity");
        if (op == CollReduceOp::NONE || total_elements == 0 ||
            physical_payload_bits == 0)
            throw std::invalid_argument("invalid reduce stream shape");
        const uint64_t dtype_bits = CollDTypeBits(dtype);
        const uint64_t total_bits =
            CollCheckedMul(total_elements, dtype_bits);
        const VectorWork work =
            ComputeVectorWork(total_elements, 1, dca_vector_bits, dtype);
        if (physical_data_flits !=
                CollCeilDiv(total_bits, physical_payload_bits) ||
            vector_beats != work.vector_beats)
            throw std::invalid_argument(
                "reduce stream counts disagree with payload geometry");
    }
};

struct AsyncSessionProgress {
    bool tx_done = false;
    bool rx_done = false;
    uint64_t pending_headers = 0;
    uint64_t pending_operands = 0;
    uint64_t pending_issues = 0;
    uint64_t inflight_dca = 0;
    uint64_t pending_results = 0;

    bool Complete() const {
        return tx_done && rx_done && pending_headers == 0 &&
               pending_operands == 0 && pending_issues == 0 &&
               inflight_dca == 0 && pending_results == 0;
    }
};

struct TraceContract {
    TraceGeneration generation = TraceGeneration::REFACTOR;
    ReduceWireVersion wire_version = ReduceWireVersion::STREAM_V2;
    uint64_t header_flits = 0;
    uint64_t data_flits = 0;
    uint64_t dca_issues = 0;
    uint64_t dca_completions = 0;
    uint64_t dca_inflight_peak = 0;
    uint64_t dca_stall_cycles = 0;

    void Validate() const {
        const bool refactor = generation == TraceGeneration::REFACTOR;
        const bool stream_v2 = wire_version == ReduceWireVersion::STREAM_V2;
        if (refactor != stream_v2)
            throw std::invalid_argument(
                "trace generation and reduce wire version disagree");
        if (dca_completions > dca_issues ||
            dca_inflight_peak > dca_issues)
            throw std::invalid_argument("invalid DCA trace counters");
    }
};

} // namespace coll_refactor
