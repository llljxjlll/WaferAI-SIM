#include "dte/coll_compute_pool.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace coll_refactor {
namespace {

uint64_t DtypeMask(CollDType dtype) {
    const uint64_t bits = CollDTypeBits(dtype);
    return bits == 64 ? std::numeric_limits<uint64_t>::max()
                      : ((uint64_t{1} << bits) - 1);
}

bool SignedGreater(uint64_t lhs, uint64_t rhs, uint64_t bits) {
    const uint64_t sign = uint64_t{1} << (bits - 1);
    const bool lhs_negative = (lhs & sign) != 0;
    const bool rhs_negative = (rhs & sign) != 0;
    if (lhs_negative != rhs_negative) return !lhs_negative;
    return lhs > rhs;
}

size_t DtypeIndex(CollDType dtype) {
    switch (dtype) {
    case CollDType::UINT8: return 0;
    case CollDType::INT32: return 1;
    case CollDType::INT64: return 2;
    case CollDType::FP32: return 3;
    case CollDType::FP16: return 4;
    case CollDType::FP8: return 5;
    }
    throw std::invalid_argument("unknown DCA request dtype");
}

size_t OpIndex(CollReduceOp op) {
    if (op == CollReduceOp::SUM) return 0;
    if (op == CollReduceOp::MAX) return 1;
    throw std::invalid_argument("DCA request requires SUM or MAX");
}

} // namespace

uint64_t CollFp32Bits(float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float CollFp32FromBits(uint64_t bits) {
    const uint32_t raw = static_cast<uint32_t>(bits);
    float value = 0.0f;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

std::vector<uint64_t> ReduceFp32Vector(
    CollReduceOp op, const VectorLaneMask &lane_mask,
    const std::vector<uint64_t> &lhs,
    const std::vector<uint64_t> &rhs) {
    lane_mask.Validate();
    if (op != CollReduceOp::SUM && op != CollReduceOp::MAX)
        throw std::invalid_argument("FP32 helper requires SUM or MAX");
    if (lhs.size() != lane_mask.lane_count ||
        rhs.size() != lane_mask.lane_count)
        throw std::invalid_argument(
            "FP32 operands disagree with vector lanes");
    constexpr uint32_t canonical_nan = 0x7fc00000u;
    const auto is_nan = [](uint32_t raw) {
        return (raw & 0x7f800000u) == 0x7f800000u &&
               (raw & 0x007fffffu) != 0;
    };
    const auto is_zero = [](uint32_t raw) {
        return (raw & 0x7fffffffu) == 0;
    };
    std::vector<uint64_t> result(lhs);
    for (uint64_t lane = 0; lane < lane_mask.valid_lanes; ++lane) {
        const uint32_t a_raw = static_cast<uint32_t>(lhs[lane]);
        const uint32_t b_raw = static_cast<uint32_t>(rhs[lane]);
        const float a = CollFp32FromBits(lhs[lane]);
        const float b = CollFp32FromBits(rhs[lane]);
        if (is_nan(a_raw) || is_nan(b_raw)) {
            result[lane] = canonical_nan;
            continue;
        }
        float value = 0.0f;
        if (op == CollReduceOp::SUM) {
            if (is_zero(a_raw) && is_zero(b_raw)) {
                result[lane] = (a_raw & b_raw & 0x80000000u);
                continue;
            }
            value = a + b;
            if (is_nan(static_cast<uint32_t>(CollFp32Bits(value)))) {
                result[lane] = canonical_nan;
                continue;
            }
        } else if (is_zero(a_raw) && is_zero(b_raw)) {
            // Freeze IEEE signed-zero tie behavior independently of host
            // libm: MAX(-0,+0)=+0 and MAX(-0,-0)=-0.
            result[lane] = (a_raw & b_raw & 0x80000000u);
            continue;
        } else if (a == b) {
            value = a;
        } else {
            value = a > b ? a : b;
        }
        result[lane] = CollFp32Bits(value);
    }
    return result;
}

std::vector<uint64_t> ReduceIntegerVector(
    CollDType dtype, CollReduceOp op, const VectorLaneMask &lane_mask,
    const std::vector<uint64_t> &lhs,
    const std::vector<uint64_t> &rhs) {
    lane_mask.Validate();
    if (dtype == CollDType::FP32 || dtype == CollDType::FP16 ||
        dtype == CollDType::FP8)
        throw std::invalid_argument(
            "integer DCA helper does not implement floating dtype");
    if (op != CollReduceOp::SUM && op != CollReduceOp::MAX)
        throw std::invalid_argument(
            "integer DCA helper requires SUM or MAX");
    if (lhs.size() != lane_mask.lane_count ||
        rhs.size() != lane_mask.lane_count)
        throw std::invalid_argument(
            "DCA operand values disagree with vector lanes");

    const uint64_t bits = CollDTypeBits(dtype);
    const uint64_t mask = DtypeMask(dtype);
    std::vector<uint64_t> result(lhs);
    for (uint64_t lane = 0; lane < lane_mask.valid_lanes; ++lane) {
        const uint64_t a = lhs[lane] & mask;
        const uint64_t b = rhs[lane] & mask;
        if (op == CollReduceOp::SUM) {
            result[lane] = (a + b) & mask;
        } else if (dtype == CollDType::UINT8) {
            result[lane] = std::max(a, b);
        } else {
            result[lane] = SignedGreater(a, b, bits) ? a : b;
        }
    }
    return result;
}

DcaComputePool::DcaComputePool(const NocCollDcaConfig &config,
                               uint64_t tag_limit)
    : config_(config), tag_limit_(tag_limit) {
    // Constructing a pool means the DCA resource is actually instantiated;
    // unlike dormant JSON configuration, its resource contract must be valid.
    config_.Validate();
    if (tag_limit_ == 0)
        throw std::invalid_argument("DCA tag limit must be positive");
}

void DcaComputePool::ValidatePayload(const DcaPoolRequest &payload) const {
    payload.request.Validate();
    if (payload.request.tag > tag_limit_)
        throw std::invalid_argument("DCA request tag exceeds configured range");
    const VectorBeat &beat = payload.request.operands[0];
    if (beat.vector_bits != config_.vector_bits)
        throw std::invalid_argument(
            "DCA request width disagrees with compute pool width");
    (void)TimingFor(payload.request);

    if (config_.value_mode == NocCollValueMode::TIMING_ONLY) {
        if (!payload.operand_values[0].empty() ||
            !payload.operand_values[1].empty())
            throw std::invalid_argument(
                "timing-only DCA request must not carry values");
        return;
    }
    if (config_.value_mode == NocCollValueMode::INTEGER_EXACT &&
        (beat.dtype == CollDType::FP32 || beat.dtype == CollDType::FP16 ||
         beat.dtype == CollDType::FP8))
        throw std::invalid_argument(
            "integer-exact DCA request does not support floating dtype");
    if (config_.value_mode == NocCollValueMode::FP_EXACT &&
        beat.dtype != CollDType::FP32)
        throw std::invalid_argument(
            "fp-exact DCA request requires FP32 dtype");
    for (const auto &values : payload.operand_values)
        if (values.size() != beat.lane_mask.lane_count)
            throw std::invalid_argument(
                "DCA operand values disagree with vector lanes");
}

DcaTiming DcaComputePool::TimingFor(const DcaRequest &request) const {
    const auto &entry =
        config_.timing[DtypeIndex(request.operands[0].dtype)]
                      [OpIndex(request.op)];
    DcaTiming timing{entry.latency, entry.initiation_interval};
    timing.Validate();
    return timing;
}

uint64_t DcaComputePool::Pending() const {
    return core_pending_.size() + dca_pending_.size();
}

uint64_t DcaComputePool::Inflight() const {
    return inflight_.size();
}

uint64_t DcaComputePool::Results() const {
    return results_.size();
}

uint64_t DcaComputePool::Residual() const {
    return Pending() + Inflight() + Results();
}

bool DcaComputePool::Drained() const {
    return Residual() == 0 && live_tags_.empty();
}

void DcaComputePool::UpdatePeaks() {
    stats_.pending_peak = std::max(stats_.pending_peak, Pending());
    stats_.inflight_peak = std::max(stats_.inflight_peak, Inflight());
    stats_.result_peak = std::max(stats_.result_peak, Results());
}

bool DcaComputePool::TrySubmit(DcaRequestSource source,
                               DcaPoolRequest request) {
    ValidatePayload(request);
    const uint64_t tag = request.request.tag;
    if (live_tags_.count(tag))
        throw std::invalid_argument("duplicate live DCA request tag");
    if (Pending() >= config_.operand_fifo_depth) {
        ++stats_.issue_queue_backpressure;
        return false;
    }
    PendingEntry entry{source, std::move(request), next_sequence_++};
    if (source == DcaRequestSource::CORE) {
        core_pending_.push_back(std::move(entry));
        ++stats_.core_submitted;
    } else {
        dca_pending_.push_back(std::move(entry));
        ++stats_.dca_submitted;
    }
    live_tags_.insert(tag);
    UpdatePeaks();
    return true;
}

uint64_t DcaComputePool::AdvanceTag(uint64_t tag) const {
    return tag == tag_limit_ ? 1 : tag + 1;
}

std::optional<uint64_t> DcaComputePool::TrySubmitAutoTagged(
    DcaRequestSource source, DcaPoolRequest request) {
    if (request.request.tag != 0)
        throw std::invalid_argument(
            "auto-tagged DCA request must start with tag zero");
    if (Pending() >= config_.operand_fifo_depth) {
        ++stats_.issue_queue_backpressure;
        return std::nullopt;
    }
    const uint64_t first = next_auto_tag_;
    uint64_t candidate = first;
    do {
        if (!live_tags_.count(candidate)) {
            request.request.tag = candidate;
            if (!TrySubmit(source, std::move(request)))
                return std::nullopt;
            next_auto_tag_ = AdvanceTag(candidate);
            return candidate;
        }
        candidate = AdvanceTag(candidate);
    } while (candidate != first);
    throw std::overflow_error("DCA tag space exhausted");
}

DcaRequestSource DcaComputePool::SelectSource() const {
    const bool core = !core_pending_.empty();
    const bool dca = !dca_pending_.empty();
    if (!core) return DcaRequestSource::DCA;
    if (!dca) return DcaRequestSource::CORE;
    switch (config_.arbitration) {
    case NocCollDcaArbitration::ROUND_ROBIN: return rr_next_;
    case NocCollDcaArbitration::CORE_PRIORITY:
        return DcaRequestSource::CORE;
    case NocCollDcaArbitration::DCA_PRIORITY:
        return DcaRequestSource::DCA;
    }
    throw std::invalid_argument("invalid DCA arbitration enum");
}

DcaComputePool::PendingEntry
DcaComputePool::PopPending(DcaRequestSource source) {
    auto &queue = source == DcaRequestSource::CORE
        ? core_pending_ : dca_pending_;
    PendingEntry entry = std::move(queue.front());
    queue.pop_front();
    return entry;
}

bool DcaComputePool::IsPendingTag(uint64_t tag) const {
    const auto has_tag = [tag](const PendingEntry &entry) {
        return entry.payload.request.tag == tag;
    };
    return std::any_of(core_pending_.begin(), core_pending_.end(), has_tag) ||
           std::any_of(dca_pending_.begin(), dca_pending_.end(), has_tag);
}

bool DcaComputePool::IsResultTag(uint64_t tag) const {
    return std::any_of(results_.begin(), results_.end(),
                       [tag](const DcaPoolResult &result) {
                           return result.result.tag == tag;
                       });
}

bool DcaComputePool::TryComplete(uint64_t tag, uint64_t cycle) {
    auto found = std::find_if(
        inflight_.begin(), inflight_.end(),
        [tag](const InflightEntry &entry) {
            return entry.pending.payload.request.tag == tag;
        });
    if (found == inflight_.end()) {
        if (IsResultTag(tag))
            throw std::invalid_argument("duplicate DCA completion");
        if (IsPendingTag(tag))
            throw std::invalid_argument("completion for unissued DCA tag");
        throw std::invalid_argument("completion for unknown DCA tag");
    }
    if (cycle < found->completion_cycle)
        throw std::invalid_argument("DCA completion arrived before latency");
    if (Results() >= config_.result_fifo_depth) return false;

    const DcaRequest &request = found->pending.payload.request;
    DcaPoolResult result;
    result.result = {request.tag, request.key, request.operands[0]};
    result.values = std::move(found->values);
    result.source = found->pending.source;
    result.issue_cycle = found->issue_cycle;
    result.scheduled_completion_cycle = found->completion_cycle;
    result.enqueue_cycle = cycle;
    result.result.ValidateAgainst(request);
    results_.push_back(std::move(result));
    inflight_.erase(found);
    ++stats_.completions;
    UpdatePeaks();
    return true;
}

void DcaComputePool::Tick(uint64_t cycle) {
    if (ticked_ && cycle <= last_cycle_)
        throw std::invalid_argument(
            "DCA compute pool cycle must increase");
    if (ticked_ && cycle != CollCheckedAdd(last_cycle_, 1) &&
        !idle_after_last_tick_)
        throw std::invalid_argument(
            "active DCA compute pool requires consecutive cycle ticks");
    ticked_ = true;
    last_cycle_ = cycle;
    ++stats_.cycles;

    while (Results() < config_.result_fifo_depth) {
        auto due = inflight_.end();
        for (auto it = inflight_.begin(); it != inflight_.end(); ++it) {
            if (it->completion_cycle > cycle) continue;
            if (due == inflight_.end() ||
                std::tie(it->completion_cycle, it->pending.sequence) <
                    std::tie(due->completion_cycle,
                             due->pending.sequence))
                due = it;
        }
        if (due == inflight_.end()) break;
        const uint64_t tag = due->pending.payload.request.tag;
        if (!TryComplete(tag, cycle)) break;
    }
    const bool due_blocked = std::any_of(
        inflight_.begin(), inflight_.end(),
        [cycle](const InflightEntry &entry) {
            return entry.completion_cycle <= cycle;
        });
    if (due_blocked && Results() >= config_.result_fifo_depth)
        ++stats_.result_backpressure_cycles;

    const bool core_waiting = !core_pending_.empty();
    const bool dca_waiting = !dca_pending_.empty();
    bool issued = false;
    DcaRequestSource selected = DcaRequestSource::DCA;
    if (core_waiting || dca_waiting) {
        if (cycle < next_issue_cycle_) {
            ++stats_.ii_stall_cycles;
        } else if (Inflight() >= config_.header_fifo_depth) {
            ++stats_.inflight_stall_cycles;
        } else {
            selected = SelectSource();
            PendingEntry pending = PopPending(selected);
            const DcaTiming timing = TimingFor(pending.payload.request);
            std::vector<uint64_t> values;
            if (config_.value_mode == NocCollValueMode::INTEGER_EXACT) {
                values = ReduceIntegerVector(
                    pending.payload.request.operands[0].dtype,
                    pending.payload.request.op,
                    pending.payload.request.operands[0].lane_mask,
                    pending.payload.operand_values[0],
                    pending.payload.operand_values[1]);
            } else if (config_.value_mode == NocCollValueMode::FP_EXACT) {
                values = ReduceFp32Vector(
                    pending.payload.request.op,
                    pending.payload.request.operands[0].lane_mask,
                    pending.payload.operand_values[0],
                    pending.payload.operand_values[1]);
            }
            const uint64_t completion =
                DcaCompletionCycle(cycle, timing);
            inflight_.push_back(
                {std::move(pending), std::move(values), cycle, completion});
            next_issue_cycle_ =
                CollCheckedAdd(cycle, timing.initiation_interval);
            if (selected == DcaRequestSource::CORE)
                ++stats_.core_issued;
            else
                ++stats_.dca_issued;
            if (config_.arbitration ==
                    NocCollDcaArbitration::ROUND_ROBIN)
                rr_next_ = selected == DcaRequestSource::CORE
                    ? DcaRequestSource::DCA
                    : DcaRequestSource::CORE;
            issued = true;
            UpdatePeaks();
        }
    }
    if (core_waiting &&
        (!issued || selected != DcaRequestSource::CORE))
        ++stats_.core_wait_cycles;
    if (dca_waiting &&
        (!issued || selected != DcaRequestSource::DCA))
        ++stats_.dca_wait_cycles;
    idle_after_last_tick_ = Drained();
}

std::optional<uint64_t> DcaComputePool::FrontResultTag() const {
    return results_.empty()
        ? std::nullopt
        : std::optional<uint64_t>(results_.front().result.tag);
}

std::optional<DcaPoolResult> DcaComputePool::PopResult() {
    if (results_.empty()) return std::nullopt;
    DcaPoolResult result = std::move(results_.front());
    results_.pop_front();
    if (live_tags_.erase(result.result.tag) != 1)
        throw std::logic_error("DCA result tag lifecycle is corrupt");
    ++stats_.results_consumed;
    idle_after_last_tick_ = Drained();
    return result;
}

} // namespace coll_refactor
