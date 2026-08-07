#include "dte/coll_reduce_stream.h"

#include <algorithm>
#include <stdexcept>
#include <tuple>

namespace coll_refactor {
namespace {

constexpr uint64_t MASK_8 = 0xff;
constexpr uint64_t MASK_12 = 0xfff;
constexpr uint64_t MASK_16 = 0xffff;
constexpr uint64_t MASK_24 = 0xffffff;

uint64_t Field(const sc_bv<256> &wire, int hi, int lo) {
    return wire.range(hi, lo).to_uint64();
}

void RequireRange(uint64_t value, uint64_t maximum, const char *name) {
    if (value > maximum)
        throw std::invalid_argument(std::string(name) +
                                    " exceeds reduce stream wire range");
}

uint64_t TotalBits(const ReduceStreamWireHeader &header) {
    return CollCheckedMul(header.stream.total_elements,
                          CollDTypeBits(header.stream.dtype));
}

void ValidateDataShape(const ReduceStreamDataFlit &data) {
    if (data.route.tree_id == 0 || data.route.stream_id > MASK_24 ||
        data.route.reduce_stage_id > MASK_8 ||
        data.route.source_id > MASK_12 || data.seq_id > MASK_16)
        throw std::invalid_argument("invalid reduce stream data identity");
    if (data.length_bits == 0 || data.length_bits > 128)
        throw std::invalid_argument("invalid reduce stream data length");
    if (!data.is_tail && data.length_bits != 128)
        throw std::invalid_argument(
            "non-tail reduce stream data must carry 128 bits");
    for (uint16_t bit = data.length_bits; bit < 128; ++bit)
        if (data.payload[bit].to_bool())
            throw std::invalid_argument(
                "unused reduce stream payload bits must be zero");
}

void ValidateBeat(const ReduceVectorBeat &beat, uint64_t dca_vector_bits) {
    beat.geometry.Validate();
    if (beat.geometry.vector_bits != dca_vector_bits ||
        beat.slices.size() != dca_vector_bits / 128)
        throw std::invalid_argument(
            "reduce vector beat storage disagrees with vector width");
}

bool SameCollective(const ReduceStreamWireHeader &lhs,
                    const ReduceStreamWireHeader &rhs) {
    return lhs.stream.key.collective == rhs.stream.key.collective;
}

} // namespace

bool ReduceStreamRouteKey::operator==(
    const ReduceStreamRouteKey &other) const {
    return std::tie(tree_id, reduce_stage_id, stream_id, source_id) ==
           std::tie(other.tree_id, other.reduce_stage_id, other.stream_id,
                    other.source_id);
}

bool ReduceStreamRouteKey::operator<(
    const ReduceStreamRouteKey &other) const {
    return std::tie(tree_id, reduce_stage_id, stream_id, source_id) <
           std::tie(other.tree_id, other.reduce_stage_id, other.stream_id,
                    other.source_id);
}

ReduceStreamRouteKey ReduceStreamWireHeader::Route() const {
    return {stream.tree_id, reduce_stage_id, stream.key.stream_id, source_id};
}

void ReduceStreamWireHeader::Validate(uint64_t physical_payload_bits,
                                      uint64_t dca_vector_bits) const {
    if (physical_payload_bits != 128 || dca_vector_bits == 0 ||
        dca_vector_bits % physical_payload_bits != 0)
        throw std::invalid_argument(
            "R3 requires 128-bit physical flits and integral vectors");
    stream.Validate(physical_payload_bits, dca_vector_bits);
    if (stream.op != CollReduceOp::SUM && stream.op != CollReduceOp::MAX)
        throw std::invalid_argument("R3 reduce stream requires SUM or MAX");
    const VectorWork work = ComputeVectorWork(
        stream.total_elements, 1, dca_vector_bits, stream.dtype);
    if (tail_valid_lanes != work.tail_valid_lanes)
        throw std::invalid_argument(
            "reduce stream tail lanes disagree with vector geometry");
    RequireRange(stream.key.collective.group_id, MASK_24, "group_id");
    RequireRange(stream.key.collective.collective_id, MASK_24,
                 "collective_id");
    RequireRange(stream.key.collective.epoch, MASK_24, "epoch");
    RequireRange(stream.key.phase_id, MASK_12, "phase_id");
    RequireRange(reduce_stage_id, MASK_8, "reduce_stage_id");
    RequireRange(stream.key.stream_id, MASK_24, "stream_id");
    RequireRange(source_id, MASK_12, "source_id");
    RequireRange(stream.total_elements, MASK_24, "total_elements");
    RequireRange(stream.physical_data_flits, MASK_16,
                 "physical_data_flits");
    RequireRange(stream.vector_beats, MASK_16, "vector_beats");
    RequireRange(tail_valid_lanes, MASK_8, "tail_valid_lanes");
}

sc_bv<256> SerializeReduceStreamHeader(
    const ReduceStreamWireHeader &header, uint64_t dca_vector_bits) {
    header.Validate(128, dca_vector_bits);
    sc_bv<256> wire = 0;
    wire.range(15, 0) = REDUCE_STREAM_MAGIC;
    wire.range(19, 16) =
        static_cast<uint8_t>(header.stream.wire_version);
    wire.range(23, 20) =
        static_cast<uint8_t>(ReduceStreamSegment::HEADER);
    wire.range(39, 24) = header.stream.tree_id;
    wire.range(63, 40) = header.stream.key.collective.group_id;
    wire.range(87, 64) = header.stream.key.collective.collective_id;
    wire.range(111, 88) = header.stream.key.collective.epoch;
    wire.range(123, 112) = header.stream.key.phase_id;
    wire.range(131, 124) = header.reduce_stage_id;
    wire.range(155, 132) = header.stream.key.stream_id;
    wire.range(167, 156) = header.source_id;
    wire.range(170, 168) = static_cast<uint8_t>(header.stream.dtype);
    wire.range(172, 171) = static_cast<uint8_t>(header.stream.op);
    wire.range(196, 173) = header.stream.total_elements;
    wire.range(212, 197) = header.stream.physical_data_flits;
    wire.range(228, 213) = header.stream.vector_beats;
    wire.range(236, 229) = header.tail_valid_lanes;
    return wire;
}

bool IsReduceStreamHeaderWire(const sc_bv<256> &wire) {
    return Field(wire, 15, 0) == REDUCE_STREAM_MAGIC &&
           Field(wire, 19, 16) ==
               static_cast<uint8_t>(ReduceWireVersion::STREAM_V2) &&
           Field(wire, 23, 20) ==
               static_cast<uint8_t>(ReduceStreamSegment::HEADER) &&
           Field(wire, 39, 24) != 0 && Field(wire, 255, 237) == 0;
}

ReduceStreamWireHeader DeserializeReduceStreamHeader(
    const sc_bv<256> &wire, uint64_t dca_vector_bits) {
    if (!IsReduceStreamHeaderWire(wire))
        throw std::invalid_argument("invalid reduce stream header prefix");
    ReduceStreamWireHeader header;
    header.stream.wire_version = static_cast<ReduceWireVersion>(
        Field(wire, 19, 16));
    header.stream.tree_id = Field(wire, 39, 24);
    header.stream.key.collective.group_id = Field(wire, 63, 40);
    header.stream.key.collective.collective_id = Field(wire, 87, 64);
    header.stream.key.collective.epoch = Field(wire, 111, 88);
    header.stream.key.phase_id = Field(wire, 123, 112);
    header.reduce_stage_id = Field(wire, 131, 124);
    header.stream.key.stream_id = Field(wire, 155, 132);
    header.source_id = Field(wire, 167, 156);
    const uint64_t dtype = Field(wire, 170, 168);
    const uint64_t op = Field(wire, 172, 171);
    if (dtype > static_cast<uint8_t>(CollDType::FP8) ||
        op > static_cast<uint8_t>(CollReduceOp::MAX))
        throw std::invalid_argument("invalid reduce stream enum value");
    header.stream.dtype = static_cast<CollDType>(dtype);
    header.stream.op = static_cast<CollReduceOp>(op);
    header.stream.total_elements = Field(wire, 196, 173);
    header.stream.physical_data_flits = Field(wire, 212, 197);
    header.stream.vector_beats = Field(wire, 228, 213);
    header.tail_valid_lanes = Field(wire, 236, 229);
    header.Validate(128, dca_vector_bits);
    return header;
}

sc_bv<256> SerializeReduceStreamData(
    const ReduceStreamDataFlit &data) {
    ValidateDataShape(data);
    sc_bv<256> wire = 0;
    wire.range(127, 0) = data.payload;
    wire.range(143, 128) = REDUCE_STREAM_MAGIC;
    wire.range(147, 144) =
        static_cast<uint8_t>(ReduceWireVersion::STREAM_V2);
    wire.range(151, 148) =
        static_cast<uint8_t>(ReduceStreamSegment::DATA);
    wire.range(175, 152) = data.route.stream_id;
    wire.range(191, 176) = data.seq_id;
    wire.range(199, 192) = data.length_bits;
    wire[200] = data.is_tail;
    wire.range(216, 201) = data.route.tree_id;
    wire.range(228, 217) = data.route.source_id;
    wire.range(236, 229) = data.route.reduce_stage_id;
    return wire;
}

bool IsReduceStreamDataWire(const sc_bv<256> &wire) {
    return Field(wire, 143, 128) == REDUCE_STREAM_MAGIC &&
           Field(wire, 147, 144) ==
               static_cast<uint8_t>(ReduceWireVersion::STREAM_V2) &&
           Field(wire, 151, 148) ==
               static_cast<uint8_t>(ReduceStreamSegment::DATA) &&
           Field(wire, 216, 201) != 0 && Field(wire, 255, 237) == 0;
}

ReduceStreamDataFlit DeserializeReduceStreamData(
    const sc_bv<256> &wire) {
    if (!IsReduceStreamDataWire(wire))
        throw std::invalid_argument("invalid reduce stream data prefix");
    ReduceStreamDataFlit data;
    data.payload = wire.range(127, 0);
    data.route.stream_id = Field(wire, 175, 152);
    data.seq_id = Field(wire, 191, 176);
    data.length_bits = Field(wire, 199, 192);
    data.is_tail = wire[200].to_bool();
    data.route.tree_id = Field(wire, 216, 201);
    data.route.source_id = Field(wire, 228, 217);
    data.route.reduce_stage_id = Field(wire, 236, 229);
    ValidateDataShape(data);
    return data;
}

void ValidateReduceWireCompatibility(ReduceWireVersion active,
                                     ReduceWireVersion incoming) {
    const auto valid = [](ReduceWireVersion version) {
        return version == ReduceWireVersion::LEGACY_TWO_SEGMENT ||
               version == ReduceWireVersion::STREAM_V2;
    };
    if (!valid(active) || !valid(incoming) || active != incoming)
        throw std::invalid_argument(
            "one collective cannot mix reduce wire generations");
}

uint64_t BinaryReduceSchedule::IssueCount(uint64_t vector_beats) const {
    if (vector_beats == 0)
        throw std::invalid_argument("vector beat count must be positive");
    return CollCheckedMul(stages.size(), vector_beats);
}

BinaryReduceSchedule BuildBinaryReduceSchedule(uint16_t input_count) {
    if (input_count == 0)
        throw std::invalid_argument("reduce schedule requires an input");
    BinaryReduceSchedule schedule;
    schedule.input_count = input_count;
    schedule.bypass = input_count == 1;
    for (uint16_t stage = 0; stage + 1 < input_count; ++stage) {
        BinaryReduceStage entry;
        entry.stage_id = stage;
        entry.inputs[0] = stage == 0
            ? ReduceStageInputRef{ReduceStageInputKind::NETWORK_INPUT, 0}
            : ReduceStageInputRef{ReduceStageInputKind::LOCAL_FEEDBACK,
                                  static_cast<uint16_t>(stage - 1)};
        entry.inputs[1] =
            {ReduceStageInputKind::NETWORK_INPUT,
             static_cast<uint16_t>(stage + 1)};
        entry.final_output = stage + 2 == input_count;
        schedule.stages.push_back(entry);
    }
    return schedule;
}

ReduceStreamAssembler::ReduceStreamAssembler(
    const ReduceStreamWireHeader &header, size_t ready_capacity,
    uint64_t physical_payload_bits, uint64_t dca_vector_bits)
    : header_(header), ready_capacity_(ready_capacity),
      physical_payload_bits_(physical_payload_bits),
      dca_vector_bits_(dca_vector_bits),
      slices_per_beat_(dca_vector_bits / physical_payload_bits) {
    header_.Validate(physical_payload_bits_, dca_vector_bits_);
    if (ready_capacity_ == 0)
        throw std::invalid_argument(
            "reduce assembler ready capacity must be positive");
}

uint64_t ReduceStreamAssembler::ExpectedTailBits() const {
    const uint64_t remainder = TotalBits(header_) % physical_payload_bits_;
    return remainder == 0 ? physical_payload_bits_ : remainder;
}

bool ReduceStreamAssembler::WouldEmitBeat() const {
    return fragments_.size() + 1 == slices_per_beat_ ||
           next_seq_ + 1 == header_.stream.physical_data_flits;
}

void ReduceStreamAssembler::EmitBeat() {
    const VectorWork work = ComputeVectorWork(
        header_.stream.total_elements, 1, dca_vector_bits_,
        header_.stream.dtype);
    const bool final = next_beat_id_ + 1 == header_.stream.vector_beats;
    VectorBeat geometry{header_.stream.dtype, dca_vector_bits_,
                        {work.lanes,
                         final ? work.tail_valid_lanes : work.lanes}};
    geometry.Validate();
    fragments_.resize(slices_per_beat_);
    ready_.push_back({{header_.stream.key, header_.reduce_stage_id,
                       next_beat_id_},
                      geometry, fragments_});
    fragments_.clear();
    ++next_beat_id_;
}

ReduceStreamAcceptStatus ReduceStreamAssembler::Accept(
    const ReduceStreamDataFlit &data) {
    if (!(data.route == header_.Route()))
        throw std::invalid_argument("reduce data route does not match header");
    if (Complete())
        throw std::invalid_argument("extra data after reduce stream tail");
    if (data.seq_id != next_seq_)
        throw std::invalid_argument("reduce data sequence is not contiguous");
    const bool final = next_seq_ + 1 == header_.stream.physical_data_flits;
    if (data.is_tail != final ||
        data.length_bits != (final ? ExpectedTailBits() : 128))
        throw std::invalid_argument("reduce data tail shape mismatch");
    ValidateDataShape(data);
    if (WouldEmitBeat() && ready_.size() >= ready_capacity_)
        return ReduceStreamAcceptStatus::BACKPRESSURE;

    fragments_.push_back(data.payload);
    ++next_seq_;
    const bool emit = fragments_.size() == slices_per_beat_ || Complete();
    if (emit) EmitBeat();
    if (Complete()) return ReduceStreamAcceptStatus::STREAM_COMPLETE;
    return emit ? ReduceStreamAcceptStatus::BEAT_READY
                : ReduceStreamAcceptStatus::ACCEPTED;
}

std::optional<ReduceVectorBeat> ReduceStreamAssembler::PopBeat() {
    if (ready_.empty()) return std::nullopt;
    ReduceVectorBeat beat = std::move(ready_.front());
    ready_.pop_front();
    return beat;
}

void ReduceStreamAssembler::Finish() const {
    if (!Complete())
        throw std::invalid_argument("truncated reduce stream");
    if (!fragments_.empty())
        throw std::logic_error("completed reduce stream retained fragments");
}

bool ReduceStreamAssembler::Complete() const {
    return next_seq_ == header_.stream.physical_data_flits;
}

std::vector<ReduceStreamDataFlit> SplitReduceVectorBeat(
    const ReduceStreamWireHeader &header, const ReduceVectorBeat &beat,
    uint64_t physical_payload_bits, uint64_t dca_vector_bits) {
    header.Validate(physical_payload_bits, dca_vector_bits);
    ValidateBeat(beat, dca_vector_bits);
    if (!(beat.key.stream == header.stream.key) ||
        beat.key.reduce_stage_id != header.reduce_stage_id ||
        beat.geometry.dtype != header.stream.dtype ||
        beat.key.vector_beat_id >= header.stream.vector_beats)
        throw std::invalid_argument("reduce vector beat does not match header");
    const VectorWork work = ComputeVectorWork(
        header.stream.total_elements, 1, dca_vector_bits,
        header.stream.dtype);
    const bool final_beat =
        beat.key.vector_beat_id + 1 == header.stream.vector_beats;
    const uint64_t expected_valid =
        final_beat ? work.tail_valid_lanes : work.lanes;
    if (beat.geometry.lane_mask.valid_lanes != expected_valid)
        throw std::invalid_argument("reduce vector beat has wrong lane mask");

    const uint64_t slices_per_beat = dca_vector_bits / physical_payload_bits;
    const uint64_t first_seq = beat.key.vector_beat_id * slices_per_beat;
    const uint64_t remaining = header.stream.physical_data_flits - first_seq;
    const uint64_t count = std::min(slices_per_beat, remaining);
    const uint64_t tail_remainder = TotalBits(header) % physical_payload_bits;
    const uint64_t tail_bits =
        tail_remainder == 0 ? physical_payload_bits : tail_remainder;
    std::vector<ReduceStreamDataFlit> result;
    for (uint64_t index = 0; index < count; ++index) {
        const uint64_t seq = first_seq + index;
        const bool tail = seq + 1 == header.stream.physical_data_flits;
        ReduceStreamDataFlit flit;
        flit.route = header.Route();
        flit.seq_id = seq;
        flit.length_bits = tail ? tail_bits : physical_payload_bits;
        flit.is_tail = tail;
        flit.payload = beat.slices[index];
        for (uint16_t bit = flit.length_bits; bit < 128; ++bit)
            flit.payload[bit] = false;
        ValidateDataShape(flit);
        result.push_back(flit);
    }
    return result;
}

void ReduceStreamCapacities::Validate() const {
    if (headers == 0 || assembler_ready_per_stream == 0 || operands < 2 ||
        issue == 0 || inflight == 0 || results == 0 ||
        network_inputs == 0 || local_feedback == 0)
        throw std::invalid_argument(
            "reduce stream capacities must be positive and operands >= 2");
}

size_t ReduceStreamOccupancy::Residual() const {
    return headers + assembler_fragments + assembler_ready +
           operand_values + operand_matches + issue + inflight + results +
           network_inputs + local_feedback;
}

ReduceStreamFiniteState::ReduceStreamFiniteState(
    const ReduceStreamCapacities &capacities,
    uint64_t physical_payload_bits, uint64_t dca_vector_bits)
    : capacities_(capacities),
      physical_payload_bits_(physical_payload_bits),
      dca_vector_bits_(dca_vector_bits) {
    capacities_.Validate();
    if (physical_payload_bits_ != 128 || dca_vector_bits_ == 0 ||
        dca_vector_bits_ % physical_payload_bits_ != 0)
        throw std::invalid_argument("invalid reduce stream vector geometry");
}

bool ReduceStreamFiniteState::TryOpenHeader(
    const ReduceStreamWireHeader &header) {
    header.Validate(physical_payload_bits_, dca_vector_bits_);
    const ReduceStreamRouteKey route = header.Route();
    if (headers_.count(route))
        throw std::invalid_argument("duplicate active reduce stream route");
    for (const auto &entry : headers_)
        if (SameCollective(entry.second.header, header))
            ValidateReduceWireCompatibility(
                entry.second.header.stream.wire_version,
                header.stream.wire_version);
    if (headers_.size() >= capacities_.headers) return false;
    headers_.emplace(
        route, HeaderContext{header,
            ReduceStreamAssembler(header,
                capacities_.assembler_ready_per_stream,
                physical_payload_bits_, dca_vector_bits_)});
    return true;
}

ReduceStreamAcceptStatus ReduceStreamFiniteState::AcceptData(
    const ReduceStreamDataFlit &data) {
    auto found = headers_.find(data.route);
    if (found == headers_.end())
        throw std::invalid_argument("data for unknown reduce stream header");
    return found->second.assembler.Accept(data);
}

std::optional<ReduceVectorBeat> ReduceStreamFiniteState::PopAssembled(
    const ReduceStreamRouteKey &route) {
    auto found = headers_.find(route);
    if (found == headers_.end())
        throw std::invalid_argument("unknown reduce stream route");
    return found->second.assembler.PopBeat();
}

bool ReduceStreamFiniteState::TryCloseHeader(
    const ReduceStreamRouteKey &route) {
    auto found = headers_.find(route);
    if (found == headers_.end())
        throw std::invalid_argument("unknown reduce stream route");
    found->second.assembler.Finish();
    if (found->second.assembler.Residual() != 0) return false;
    headers_.erase(found);
    return true;
}

bool ReduceStreamFiniteState::TryPushNetworkOperand(
    ReduceOperandToken token) {
    ValidateBeat(token.beat, dca_vector_bits_);
    if (network_inputs_.size() >= capacities_.network_inputs) return false;
    network_inputs_.push_back(std::move(token));
    return true;
}

bool ReduceStreamFiniteState::TryPushFeedbackOperand(
    ReduceOperandToken token) {
    ValidateBeat(token.beat, dca_vector_bits_);
    if (local_feedback_.size() >= capacities_.local_feedback) return false;
    local_feedback_.push_back(std::move(token));
    return true;
}

std::optional<ReduceOperandToken>
ReduceStreamFiniteState::PopArbitratedInput() {
    if (network_inputs_.empty() && local_feedback_.empty())
        return std::nullopt;
    const bool choose_feedback = !local_feedback_.empty() &&
        (network_inputs_.empty() || prefer_feedback_);
    auto &queue = choose_feedback ? local_feedback_ : network_inputs_;
    ReduceOperandToken token = std::move(queue.front());
    queue.pop_front();
    if (!network_inputs_.empty() && !local_feedback_.empty())
        prefer_feedback_ = !choose_feedback;
    else if (choose_feedback)
        prefer_feedback_ = false;
    else
        prefer_feedback_ = true;
    return token;
}

ReduceStateStatus ReduceStreamFiniteState::AcceptMatchedOperand(
    ReduceOperandToken token) {
    ValidateBeat(token.beat, dca_vector_bits_);
    if (token.slot > 1)
        throw std::invalid_argument("reduce operand slot must be zero or one");
    auto found = operands_.find(token.beat.key);
    if (found != operands_.end() && found->second.values[token.slot])
        throw std::invalid_argument("duplicate reduce operand slot");
    // Reserve one slot for an already-open pair. A new key cannot consume the
    // last slot; its matching second operand can, then the pair can advance.
    if ((found == operands_.end() &&
         operand_value_count_ + 1 >= capacities_.operands) ||
        (found != operands_.end() &&
         operand_value_count_ >= capacities_.operands))
        return ReduceStateStatus::BACKPRESSURE;
    if (found == operands_.end())
        found = operands_.emplace(token.beat.key, OperandPair{}).first;
    const uint8_t other = token.slot == 0 ? 1 : 0;
    if (found->second.values[other] &&
        !found->second.values[other]->geometry.SameGeometry(
            token.beat.geometry))
        throw std::invalid_argument("reduce operands have different geometry");
    found->second.values[token.slot] = std::move(token.beat);
    ++operand_value_count_;
    return found->second.values[other]
        ? ReduceStateStatus::READY : ReduceStateStatus::ACCEPTED;
}

bool ReduceStreamFiniteState::TryQueueMatched(const ReduceBeatKey &key) {
    auto found = operands_.find(key);
    if (found == operands_.end() || !found->second.values[0] ||
        !found->second.values[1])
        throw std::invalid_argument("reduce operand pair is incomplete");
    if (issue_.size() >= capacities_.issue) return false;
    issue_.push_back({key, {std::move(*found->second.values[0]),
                            std::move(*found->second.values[1])}});
    operands_.erase(found);
    operand_value_count_ -= 2;
    return true;
}

bool ReduceStreamFiniteState::TagLive(uint64_t tag) const {
    if (inflight_.count(tag)) return true;
    return std::any_of(results_.begin(), results_.end(),
                       [tag](const ReduceResultRecord &result) {
                           return result.tag == tag;
                       });
}

bool ReduceStreamFiniteState::TryIssue(uint64_t tag) {
    if (tag == 0)
        throw std::invalid_argument("reduce DCA tag must be non-zero");
    if (TagLive(tag))
        throw std::invalid_argument("duplicate live reduce DCA tag");
    if (issue_.empty())
        throw std::invalid_argument("no matched reduce issue available");
    if (inflight_.size() >= capacities_.inflight) return false;
    inflight_.emplace(tag, std::move(issue_.front()));
    issue_.pop_front();
    return true;
}

bool ReduceStreamFiniteState::TryComplete(uint64_t tag,
                                          ReduceVectorBeat result) {
    auto found = inflight_.find(tag);
    if (found == inflight_.end()) {
        if (TagLive(tag))
            throw std::invalid_argument("duplicate reduce DCA completion");
        throw std::invalid_argument("completion for unknown reduce DCA tag");
    }
    ValidateBeat(result, dca_vector_bits_);
    if (!(result.key == found->second.key) ||
        !result.geometry.SameGeometry(found->second.operands[0].geometry))
        throw std::invalid_argument("reduce DCA result mismatches request");
    if (results_.size() >= capacities_.results) return false;
    results_.push_back({tag, std::move(result)});
    inflight_.erase(found);
    return true;
}

bool ReduceStreamFiniteState::TryRouteResultToFeedback(
    const ReduceBeatKey &next_key, uint8_t slot) {
    if (slot > 1)
        throw std::invalid_argument("feedback operand slot must be zero or one");
    if (results_.empty())
        throw std::invalid_argument("no reduce result available");
    if (local_feedback_.size() >= capacities_.local_feedback) return false;
    ReduceResultRecord record = std::move(results_.front());
    results_.pop_front();
    record.beat.key = next_key;
    local_feedback_.push_back({std::move(record.beat), slot});
    return true;
}

std::optional<ReduceResultRecord>
ReduceStreamFiniteState::PopFinalResult() {
    if (results_.empty()) return std::nullopt;
    ReduceResultRecord result = std::move(results_.front());
    results_.pop_front();
    return result;
}

ReduceStreamOccupancy ReduceStreamFiniteState::Occupancy() const {
    ReduceStreamOccupancy occupancy;
    occupancy.headers = headers_.size();
    for (const auto &entry : headers_) {
        occupancy.assembler_fragments +=
            entry.second.assembler.FragmentCount();
        occupancy.assembler_ready += entry.second.assembler.ReadyCount();
    }
    occupancy.operand_values = operand_value_count_;
    occupancy.operand_matches = operands_.size();
    occupancy.issue = issue_.size();
    occupancy.inflight = inflight_.size();
    occupancy.results = results_.size();
    occupancy.network_inputs = network_inputs_.size();
    occupancy.local_feedback = local_feedback_.size();
    return occupancy;
}

} // namespace coll_refactor
