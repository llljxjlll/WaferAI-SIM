#pragma once

#include "dte/coll_refactor_contract.h"
#include "systemc.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <vector>

namespace coll_refactor {

constexpr uint16_t REDUCE_STREAM_MAGIC = 0xc6e3;
constexpr uint64_t REDUCE_STREAM_PHYSICAL_BITS = 128;

enum class ReduceStreamSegment : uint8_t {
    HEADER = 1,
    DATA = 2,
};

struct ReduceStreamRouteKey {
    uint16_t tree_id = 0;
    uint16_t reduce_stage_id = 0;
    uint32_t stream_id = 0;
    uint16_t source_id = 0;

    bool operator==(const ReduceStreamRouteKey &other) const;
    bool operator<(const ReduceStreamRouteKey &other) const;
};

struct ReduceStreamWireHeader {
    ReduceStreamHeader stream;
    uint16_t reduce_stage_id = 0;
    uint16_t source_id = 0;
    uint16_t tail_valid_lanes = 0;

    ReduceStreamRouteKey Route() const;
    void Validate(uint64_t physical_payload_bits = 128,
                  uint64_t dca_vector_bits = 512) const;
};

struct ReduceStreamDataFlit {
    ReduceStreamRouteKey route;
    uint32_t seq_id = 0;
    uint8_t length_bits = 0;
    bool is_tail = false;
    sc_bv<128> payload;
};

sc_bv<256> SerializeReduceStreamHeader(
    const ReduceStreamWireHeader &header,
    uint64_t dca_vector_bits = 512);
ReduceStreamWireHeader DeserializeReduceStreamHeader(
    const sc_bv<256> &wire, uint64_t dca_vector_bits = 512);
bool IsReduceStreamHeaderWire(const sc_bv<256> &wire);

sc_bv<256> SerializeReduceStreamData(
    const ReduceStreamDataFlit &data);
ReduceStreamDataFlit DeserializeReduceStreamData(
    const sc_bv<256> &wire);
bool IsReduceStreamDataWire(const sc_bv<256> &wire);

void ValidateReduceWireCompatibility(ReduceWireVersion active,
                                     ReduceWireVersion incoming);

enum class ReduceStageInputKind : uint8_t {
    NETWORK_INPUT = 0,
    LOCAL_FEEDBACK = 1,
};

struct ReduceStageInputRef {
    ReduceStageInputKind kind = ReduceStageInputKind::NETWORK_INPUT;
    uint16_t index = 0;

    bool operator==(const ReduceStageInputRef &other) const {
        return kind == other.kind && index == other.index;
    }
};

struct BinaryReduceStage {
    uint16_t stage_id = 0;
    std::array<ReduceStageInputRef, 2> inputs;
    bool final_output = false;
};

struct BinaryReduceSchedule {
    uint16_t input_count = 0;
    bool bypass = false;
    std::vector<BinaryReduceStage> stages;

    uint64_t IssueCount(uint64_t vector_beats) const;
};

BinaryReduceSchedule BuildBinaryReduceSchedule(uint16_t input_count);

struct ReduceVectorBeat {
    ReduceBeatKey key;
    VectorBeat geometry;
    std::vector<sc_bv<128>> slices;
};

enum class ReduceStreamAcceptStatus : uint8_t {
    ACCEPTED = 0,
    BEAT_READY = 1,
    STREAM_COMPLETE = 2,
    BACKPRESSURE = 3,
};

class ReduceStreamAssembler {
  public:
    ReduceStreamAssembler(const ReduceStreamWireHeader &header,
                          size_t ready_capacity,
                          uint64_t physical_payload_bits = 128,
                          uint64_t dca_vector_bits = 512);

    ReduceStreamAcceptStatus Accept(const ReduceStreamDataFlit &data);
    std::optional<ReduceVectorBeat> PopBeat();
    void Finish() const;
    bool Complete() const;
    size_t FragmentCount() const { return fragments_.size(); }
    size_t ReadyCount() const { return ready_.size(); }
    size_t Residual() const { return FragmentCount() + ReadyCount(); }

  private:
    ReduceStreamWireHeader header_;
    size_t ready_capacity_ = 0;
    uint64_t physical_payload_bits_ = 0;
    uint64_t dca_vector_bits_ = 0;
    uint64_t slices_per_beat_ = 0;
    uint64_t next_seq_ = 0;
    uint64_t next_beat_id_ = 0;
    std::vector<sc_bv<128>> fragments_;
    std::deque<ReduceVectorBeat> ready_;

    uint64_t ExpectedTailBits() const;
    bool WouldEmitBeat() const;
    void EmitBeat();
};

std::vector<ReduceStreamDataFlit> SplitReduceVectorBeat(
    const ReduceStreamWireHeader &header, const ReduceVectorBeat &beat,
    uint64_t physical_payload_bits = 128,
    uint64_t dca_vector_bits = 512);

struct ReduceOperandToken {
    ReduceVectorBeat beat;
    uint8_t slot = 0;
};

struct MatchedReduceOperands {
    ReduceBeatKey key;
    std::array<ReduceVectorBeat, 2> operands;
};

struct ReduceResultRecord {
    uint64_t tag = 0;
    ReduceVectorBeat beat;
};

struct ReduceStreamCapacities {
    size_t headers = 1;
    size_t assembler_ready_per_stream = 1;
    size_t operands = 2;
    size_t issue = 1;
    size_t inflight = 1;
    size_t results = 1;
    size_t network_inputs = 1;
    size_t local_feedback = 1;

    void Validate() const;
};

struct ReduceStreamOccupancy {
    size_t headers = 0;
    size_t assembler_fragments = 0;
    size_t assembler_ready = 0;
    size_t operand_values = 0;
    size_t operand_matches = 0;
    size_t issue = 0;
    size_t inflight = 0;
    size_t results = 0;
    size_t network_inputs = 0;
    size_t local_feedback = 0;

    size_t Residual() const;
};

enum class ReduceStateStatus : uint8_t {
    ACCEPTED = 0,
    READY = 1,
    BACKPRESSURE = 2,
};

class ReduceStreamFiniteState {
  public:
    explicit ReduceStreamFiniteState(
        const ReduceStreamCapacities &capacities,
        uint64_t physical_payload_bits = 128,
        uint64_t dca_vector_bits = 512);

    bool TryOpenHeader(const ReduceStreamWireHeader &header);
    ReduceStreamAcceptStatus AcceptData(const ReduceStreamDataFlit &data);
    std::optional<ReduceVectorBeat> PopAssembled(
        const ReduceStreamRouteKey &route);
    bool TryCloseHeader(const ReduceStreamRouteKey &route);

    bool TryPushNetworkOperand(ReduceOperandToken token);
    bool TryPushFeedbackOperand(ReduceOperandToken token);
    std::optional<ReduceOperandToken> PopArbitratedInput();
    ReduceStateStatus AcceptMatchedOperand(ReduceOperandToken token);
    bool TryQueueMatched(const ReduceBeatKey &key);
    bool TryIssue(uint64_t tag);
    bool TryComplete(uint64_t tag, ReduceVectorBeat result);
    bool TryRouteResultToFeedback(const ReduceBeatKey &next_key,
                                  uint8_t slot);
    std::optional<ReduceResultRecord> PopFinalResult();

    ReduceStreamOccupancy Occupancy() const;
    size_t Residual() const { return Occupancy().Residual(); }
    bool Drained() const { return Residual() == 0; }

  private:
    struct HeaderContext {
        ReduceStreamWireHeader header;
        ReduceStreamAssembler assembler;
    };
    struct OperandPair {
        std::array<std::optional<ReduceVectorBeat>, 2> values;
    };

    ReduceStreamCapacities capacities_;
    uint64_t physical_payload_bits_ = 0;
    uint64_t dca_vector_bits_ = 0;
    bool prefer_feedback_ = false;
    size_t operand_value_count_ = 0;
    std::map<ReduceStreamRouteKey, HeaderContext> headers_;
    std::map<ReduceBeatKey, OperandPair> operands_;
    std::deque<MatchedReduceOperands> issue_;
    std::map<uint64_t, MatchedReduceOperands> inflight_;
    std::deque<ReduceResultRecord> results_;
    std::deque<ReduceOperandToken> network_inputs_;
    std::deque<ReduceOperandToken> local_feedback_;

    bool TagLive(uint64_t tag) const;
};

} // namespace coll_refactor
