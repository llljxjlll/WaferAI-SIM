#pragma once

#include "dte/coll_config.h"
#include "dte/coll_refactor_contract.h"

#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <set>
#include <vector>

namespace coll_refactor {

enum class DcaRequestSource : uint8_t {
    CORE = 0,
    DCA = 1,
};

struct DcaPoolRequest {
    DcaRequest request;
    std::array<std::vector<uint64_t>, 2> operand_values;
};

struct DcaPoolResult {
    DcaResult result;
    std::vector<uint64_t> values;
    DcaRequestSource source = DcaRequestSource::DCA;
    uint64_t issue_cycle = 0;
    uint64_t scheduled_completion_cycle = 0;
    uint64_t enqueue_cycle = 0;
};

struct DcaComputePoolStats {
    uint64_t cycles = 0;
    uint64_t core_submitted = 0;
    uint64_t dca_submitted = 0;
    uint64_t core_issued = 0;
    uint64_t dca_issued = 0;
    uint64_t completions = 0;
    uint64_t results_consumed = 0;
    uint64_t issue_queue_backpressure = 0;
    uint64_t ii_stall_cycles = 0;
    uint64_t inflight_stall_cycles = 0;
    uint64_t result_backpressure_cycles = 0;
    uint64_t core_wait_cycles = 0;
    uint64_t dca_wait_cycles = 0;
    uint64_t pending_peak = 0;
    uint64_t inflight_peak = 0;
    uint64_t result_peak = 0;
};

std::vector<uint64_t> ReduceIntegerVector(
    CollDType dtype, CollReduceOp op, const VectorLaneMask &lane_mask,
    const std::vector<uint64_t> &lhs,
    const std::vector<uint64_t> &rhs);

// Deterministic FP32 contract used by both endpoint and DCA paths. NaNs are
// canonicalized to 0x7fc00000, MAX propagates a canonical NaN, ties preserve
// +0 over -0, and SUM follows the frozen binary-stage order.
uint64_t CollFp32Bits(float value);
float CollFp32FromBits(uint64_t bits);
std::vector<uint64_t> ReduceFp32Vector(
    CollReduceOp op, const VectorLaneMask &lane_mask,
    const std::vector<uint64_t> &lhs,
    const std::vector<uint64_t> &rhs);

class DcaComputePool {
  public:
    explicit DcaComputePool(
        const NocCollDcaConfig &config,
        uint64_t tag_limit = std::numeric_limits<uint64_t>::max());

    bool TrySubmit(DcaRequestSource source, DcaPoolRequest request);
    std::optional<uint64_t> TrySubmitAutoTagged(
        DcaRequestSource source, DcaPoolRequest request);

    // Advance exactly one cycle. Completions are enqueued before this cycle's
    // issue, allowing a context released at cycle C to be reused at C.
    void Tick(uint64_t cycle);

    // Exposed for deterministic completion-integrity tests and future event
    // integration. Returns false only when the result queue backpressures.
    bool TryComplete(uint64_t tag, uint64_t cycle);
    std::optional<uint64_t> FrontResultTag() const;
    std::optional<DcaPoolResult> PopResult();

    uint64_t Pending() const;
    uint64_t Inflight() const;
    uint64_t Results() const;
    uint64_t Residual() const;
    bool Drained() const;
    const DcaComputePoolStats &Stats() const { return stats_; }

  private:
    struct PendingEntry {
        DcaRequestSource source = DcaRequestSource::DCA;
        DcaPoolRequest payload;
        uint64_t sequence = 0;
    };
    struct InflightEntry {
        PendingEntry pending;
        std::vector<uint64_t> values;
        uint64_t issue_cycle = 0;
        uint64_t completion_cycle = 0;
    };

    NocCollDcaConfig config_;
    uint64_t tag_limit_;
    uint64_t next_auto_tag_ = 1;
    uint64_t next_sequence_ = 0;
    uint64_t next_issue_cycle_ = 0;
    bool ticked_ = false;
    bool idle_after_last_tick_ = true;
    uint64_t last_cycle_ = 0;
    DcaRequestSource rr_next_ = DcaRequestSource::CORE;
    std::deque<PendingEntry> core_pending_;
    std::deque<PendingEntry> dca_pending_;
    std::vector<InflightEntry> inflight_;
    std::deque<DcaPoolResult> results_;
    std::set<uint64_t> live_tags_;
    DcaComputePoolStats stats_;

    void ValidatePayload(const DcaPoolRequest &request) const;
    DcaTiming TimingFor(const DcaRequest &request) const;
    DcaRequestSource SelectSource() const;
    PendingEntry PopPending(DcaRequestSource source);
    bool IsPendingTag(uint64_t tag) const;
    bool IsResultTag(uint64_t tag) const;
    uint64_t AdvanceTag(uint64_t tag) const;
    void UpdatePeaks();
};

} // namespace coll_refactor
