#include "dte/coll_compute_pool.h"

#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace {
using namespace coll_refactor;

int failures = 0;
int checks = 0;

void Check(bool ok, const std::string &name) {
    ++checks;
    if (!ok) ++failures;
    std::cout << "  [" << (ok ? " ok " : "FAIL") << "] " << name
              << std::endl;
}

template <class E = std::exception, class F> bool Throws(F f) {
    try {
        f();
    } catch (const E &) {
        return true;
    } catch (...) {
    }
    return false;
}

NocCollDcaConfig PoolConfig(
    NocCollValueMode mode = NocCollValueMode::TIMING_ONLY,
    uint64_t pending = 16, uint64_t inflight = 16,
    uint64_t results = 16,
    NocCollDcaArbitration arbitration =
        NocCollDcaArbitration::ROUND_ROBIN) {
    NocCollDcaConfig config;
    config.value_mode = mode;
    config.operand_fifo_depth = pending;
    config.header_fifo_depth = inflight;
    config.result_fifo_depth = results;
    config.arbitration = arbitration;
    return config;
}

DcaPoolRequest Request(uint64_t tag, CollDType dtype = CollDType::UINT8,
                       CollReduceOp op = CollReduceOp::SUM,
                       uint64_t vector_bits = 512,
                       uint64_t valid_lanes = 0,
                       bool with_values = false) {
    const uint64_t lanes = vector_bits / CollDTypeBits(dtype);
    if (valid_lanes == 0) valid_lanes = lanes;
    VectorBeat beat{dtype, vector_bits, {lanes, valid_lanes}};
    DcaPoolRequest result;
    result.request.tag = tag;
    result.request.key = {{{1, static_cast<uint32_t>(tag + 10), 3},
                           4, static_cast<uint32_t>(tag + 20)},
                          5, tag + 30};
    result.request.op = op;
    result.request.operands = {beat, beat};
    if (with_values) {
        result.operand_values[0].resize(lanes);
        result.operand_values[1].resize(lanes);
        for (uint64_t lane = 0; lane < lanes; ++lane) {
            result.operand_values[0][lane] = lane + 1;
            result.operand_values[1][lane] = lane + 3;
        }
    }
    return result;
}

void Collect(DcaComputePool &pool, std::vector<DcaPoolResult> &results) {
    while (auto result = pool.PopResult())
        results.push_back(std::move(*result));
}

std::vector<uint64_t> ArbitrationOrder(NocCollDcaArbitration arbitration,
                                       DcaComputePoolStats *stats = nullptr) {
    auto config = PoolConfig(NocCollValueMode::TIMING_ONLY, 8, 8, 8,
                             arbitration);
    DcaComputePool pool(config);
    pool.TrySubmit(DcaRequestSource::CORE, Request(1));
    pool.TrySubmit(DcaRequestSource::CORE, Request(2));
    pool.TrySubmit(DcaRequestSource::DCA, Request(3));
    pool.TrySubmit(DcaRequestSource::DCA, Request(4));
    std::vector<DcaPoolResult> results;
    for (uint64_t cycle = 0; cycle <= 4; ++cycle) {
        pool.Tick(cycle);
        Collect(pool, results);
    }
    std::vector<uint64_t> tags;
    for (const auto &result : results) tags.push_back(result.result.tag);
    if (stats) *stats = pool.Stats();
    if (!pool.Drained()) tags.clear();
    return tags;
}

} // namespace

int RunCollR2SelfTest() {
    using namespace coll_refactor;
    failures = checks = 0;
    std::cout << "==== NoC collective refactor R2 DCA ComputePool self-test ===="
              << std::endl;

    const auto uint8_sum = ReduceIntegerVector(
        CollDType::UINT8, CollReduceOp::SUM, {4, 3},
        {250, 2, 200, 0xdead}, {10, 3, 100, 0xbeef});
    Check(uint8_sum == std::vector<uint64_t>({4, 5, 44, 0xdead}),
          "UINT8 SIMD SUM wraps active lanes and preserves masked tail");

    const auto int32_max = ReduceIntegerVector(
        CollDType::INT32, CollReduceOp::MAX, {3, 3},
        {0xffffffffu, 0x80000000u, 7},
        {1, 0xfffffffeu, 6});
    Check(int32_max == std::vector<uint64_t>({1, 0xfffffffeu, 7}),
          "INT32 SIMD MAX uses signed two's-complement ordering");

    const auto int64_sum = ReduceIntegerVector(
        CollDType::INT64, CollReduceOp::SUM, {2, 2},
        {std::numeric_limits<uint64_t>::max(), 9}, {1, 8});
    Check(int64_sum == std::vector<uint64_t>({0, 17}),
          "INT64 SIMD SUM has deterministic width-masked wraparound");

    Check(Throws<std::invalid_argument>([] {
              (void)ReduceIntegerVector(
                  CollDType::FP32, CollReduceOp::SUM, {1, 1}, {1}, {2});
          }) &&
              Throws<std::invalid_argument>([] {
                  (void)ReduceIntegerVector(
                      CollDType::UINT8, CollReduceOp::SUM,
                      {2, 2}, {1}, {2});
              }),
          "integer value helper rejects FP32 and wrong lane storage");

    const VectorWork wide =
        ComputeVectorWork(65, 2, 512, CollDType::UINT8);
    const VectorWork narrow =
        ComputeVectorWork(17, 2, 128, CollDType::UINT8);
    Check(wide.lanes == 64 && wide.vector_beats == 2 &&
              wide.tail_valid_lanes == 1 &&
              narrow.lanes == 16 && narrow.vector_beats == 2 &&
              narrow.tail_valid_lanes == 1,
          "512-bit and 128-bit UINT8 lane geometry is vector-width based");
    Check(ComputeVectorWork(129, 2, 512, CollDType::UINT8).total_issues == 3 &&
              ComputeVectorWork(129, 3, 512,
                                CollDType::UINT8).total_issues == 6,
          "two and three inputs require B and 2B vector issues");

    {
        auto config = PoolConfig();
        config.timing[0][0] = {4, 1};
        DcaComputePool pool(config);
        for (uint64_t tag = 1; tag <= 4; ++tag)
            Check(pool.TrySubmit(DcaRequestSource::DCA, Request(tag)),
                  "L/II pipeline accepts request " + std::to_string(tag));
        std::vector<DcaPoolResult> results;
        for (uint64_t cycle = 0; cycle <= 7; ++cycle) {
            pool.Tick(cycle);
            Collect(pool, results);
        }
        bool timing_ok = results.size() == 4;
        for (size_t i = 0; timing_ok && i < results.size(); ++i)
            timing_ok = results[i].issue_cycle == i &&
                        results[i].scheduled_completion_cycle == i + 4 &&
                        results[i].enqueue_cycle == i + 4;
        Check(timing_ok && results.back().enqueue_cycle == 7,
              "L=4 II=1 pipelines B requests and finishes at L+B-1");
        Check(pool.Stats().inflight_peak == 4 && pool.Drained(),
              "L>1 pipeline permits multiple inflight tags and drains");
    }

    {
        auto config = PoolConfig();
        config.timing[0][0] = {3, 2};
        DcaComputePool pool(config);
        for (uint64_t tag = 1; tag <= 3; ++tag)
            pool.TrySubmit(DcaRequestSource::DCA, Request(tag));
        std::vector<DcaPoolResult> results;
        for (uint64_t cycle = 0; cycle <= 7; ++cycle) {
            pool.Tick(cycle);
            Collect(pool, results);
        }
        Check(results.size() == 3 &&
                  results[0].issue_cycle == 0 &&
                  results[1].issue_cycle == 2 &&
                  results[2].issue_cycle == 4 &&
                  results[2].enqueue_cycle == 7 &&
                  pool.Stats().ii_stall_cycles == 2,
              "II=2 throttles issue without serializing on latency");
    }

    {
        auto config = PoolConfig();
        config.timing[1][1] = {7, 3};
        config.timing[0][0] = {2, 1};
        DcaComputePool pool(config);
        pool.TrySubmit(DcaRequestSource::DCA,
                       Request(1, CollDType::INT32, CollReduceOp::MAX));
        pool.TrySubmit(DcaRequestSource::DCA, Request(2));
        std::vector<DcaPoolResult> results;
        for (uint64_t cycle = 0; cycle <= 7; ++cycle) {
            pool.Tick(cycle);
            Collect(pool, results);
        }
        Check(results.size() == 2 && results[0].result.tag == 2 &&
                  results[0].issue_cycle == 3 &&
                  results[0].enqueue_cycle == 5 &&
                  results[1].result.tag == 1 &&
                  results[1].enqueue_cycle == 7,
              "per-dtype/op L and II allow out-of-order tagged completion");
    }

    {
        auto config = PoolConfig(NocCollValueMode::TIMING_ONLY, 1, 4, 4);
        DcaComputePool pool(config);
        Check(pool.TrySubmit(DcaRequestSource::DCA, Request(1)) &&
                  !pool.TrySubmit(DcaRequestSource::DCA, Request(2)) &&
                  pool.Stats().issue_queue_backpressure == 1,
              "finite issue queue reports submit backpressure");
        pool.Tick(0);
        Check(pool.TrySubmit(DcaRequestSource::DCA, Request(2)),
              "issue queue accepts retry after one request issues");
        std::vector<DcaPoolResult> results;
        for (uint64_t cycle = 1; cycle <= 2; ++cycle) {
            pool.Tick(cycle);
            Collect(pool, results);
        }
        Check(results.size() == 2 && pool.Drained(),
              "issue-queue backpressure recovers and drains");
    }

    {
        auto config = PoolConfig(NocCollValueMode::TIMING_ONLY, 4, 4, 1);
        DcaComputePool pool(config);
        pool.TrySubmit(DcaRequestSource::DCA, Request(1));
        pool.TrySubmit(DcaRequestSource::DCA, Request(2));
        pool.Tick(0);
        pool.Tick(1);
        pool.Tick(2);
        Check(pool.Results() == 1 && pool.Inflight() == 1 &&
                  pool.Stats().result_backpressure_cycles == 1,
              "full result queue backpressures a due completion");
        const auto first = pool.PopResult();
        pool.Tick(3);
        const auto second = pool.PopResult();
        Check(first && second && first->result.tag == 1 &&
                  second->result.tag == 2 &&
                  second->scheduled_completion_cycle == 2 &&
                  second->enqueue_cycle == 3 && pool.Drained(),
              "result backpressure retries once without duplicate completion");
    }

    {
        auto config = PoolConfig(NocCollValueMode::TIMING_ONLY, 4, 1, 4);
        config.timing[0][0] = {3, 1};
        DcaComputePool pool(config);
        pool.TrySubmit(DcaRequestSource::DCA, Request(1));
        pool.TrySubmit(DcaRequestSource::DCA, Request(2));
        std::vector<DcaPoolResult> results;
        for (uint64_t cycle = 0; cycle <= 6; ++cycle) {
            pool.Tick(cycle);
            Collect(pool, results);
        }
        Check(results.size() == 2 && results[1].issue_cycle == 3 &&
                  pool.Stats().inflight_stall_cycles == 2 &&
                  pool.Drained(),
              "finite inflight context backpressures then reuses completion slot");
    }

    Check(ArbitrationOrder(NocCollDcaArbitration::ROUND_ROBIN) ==
              std::vector<uint64_t>({1, 3, 2, 4}),
          "round-robin alternates synthetic core and DCA requests");

    {
        constexpr uint64_t kRequestsPerSource = 32;
        constexpr uint64_t kLatency = 5;
        constexpr uint64_t kInitiationInterval = 2;
        constexpr uint64_t kMaximumSameSourceServiceGap =
            2 * kInitiationInterval;
        auto config = PoolConfig(
            NocCollValueMode::TIMING_ONLY,
            2 * kRequestsPerSource, 2 * kRequestsPerSource, 1,
            NocCollDcaArbitration::ROUND_ROBIN);
        config.timing[0][0] = {kLatency, kInitiationInterval};
        DcaComputePool pool(config);
        bool accepted = true;
        for (uint64_t index = 0; index < kRequestsPerSource; ++index) {
            accepted = pool.TrySubmit(
                           DcaRequestSource::CORE, Request(index + 1)) &&
                       accepted;
            accepted = pool.TrySubmit(
                           DcaRequestSource::DCA,
                           Request(101 + index)) &&
                       accepted;
        }
        Check(accepted && pool.Pending() == 2 * kRequestsPerSource,
              "sustained RR fixture admits 32 CORE and 32 DCA requests");

        std::vector<DcaPoolResult> results;
        // Hold the depth-one result queue through the first eight due
        // completions, then consume only every third cycle.  Issue remains
        // active while due completions are backpressured.
        for (uint64_t cycle = 0; cycle < 512 && !pool.Drained(); ++cycle) {
            pool.Tick(cycle);
            if (cycle >= 20 && cycle % 3 == 2) {
                if (auto result = pool.PopResult())
                    results.push_back(std::move(*result));
            }
        }
        // The last Tick may have filled the result slot on a non-consumption
        // cycle; finish with consecutive cycles and eager consumption.
        for (uint64_t cycle = 512; cycle < 640 && !pool.Drained(); ++cycle) {
            pool.Tick(cycle);
            Collect(pool, results);
        }

        Check(pool.Stats().result_backpressure_cycles > 0 &&
                  pool.Stats().result_peak == 1,
              "depth-one result backpressure is exercised and bounded");
        Check(pool.Stats().core_issued == kRequestsPerSource &&
                  pool.Stats().dca_issued == kRequestsPerSource,
              "both continuously pending RR sources make complete progress");

        bool deterministic_service =
            results.size() == 2 * kRequestsPerSource;
        bool completion_timing = deterministic_service;
        bool strict_alternation = deterministic_service;
        bool bounded_service_gap = deterministic_service;
        bool exactly_once = deterministic_service;
        std::set<uint64_t> tags;
        uint64_t last_core_issue = 0;
        uint64_t last_dca_issue = 0;
        bool have_core_issue = false;
        bool have_dca_issue = false;
        for (uint64_t position = 0;
             position < results.size(); ++position) {
            const DcaPoolResult &result = results[position];
            const bool expect_core = position % 2 == 0;
            const uint64_t source_index = position / 2;
            const uint64_t expected_tag =
                expect_core ? source_index + 1 : 101 + source_index;
            const uint64_t expected_issue =
                position * kInitiationInterval;
            deterministic_service = deterministic_service &&
                result.result.tag == expected_tag &&
                result.issue_cycle == expected_issue;
            strict_alternation = strict_alternation &&
                result.source == (expect_core
                    ? DcaRequestSource::CORE : DcaRequestSource::DCA);
            completion_timing = completion_timing &&
                result.scheduled_completion_cycle ==
                    result.issue_cycle + kLatency &&
                result.enqueue_cycle >= result.scheduled_completion_cycle;
            exactly_once = exactly_once &&
                tags.insert(result.result.tag).second;
            uint64_t &last_issue = expect_core
                ? last_core_issue : last_dca_issue;
            bool &have_issue = expect_core
                ? have_core_issue : have_dca_issue;
            if (have_issue)
                bounded_service_gap = bounded_service_gap &&
                    result.issue_cycle - last_issue <=
                        kMaximumSameSourceServiceGap;
            last_issue = result.issue_cycle;
            have_issue = true;
        }
        Check(deterministic_service,
              "sustained RR freezes CORE1,DCA101,... service at II=2");
        Check(strict_alternation,
              "sustained dual-source contention alternates CORE and DCA");
        Check(bounded_service_gap && have_core_issue && have_dca_issue,
              "each source service gap is bounded by 2*II");
        Check(completion_timing,
              "every completion preserves L=5 despite result backpressure");
        Check(exactly_once && tags.size() == 2 * kRequestsPerSource &&
                  pool.Stats().completions == 2 * kRequestsPerSource &&
                  pool.Stats().results_consumed ==
                      2 * kRequestsPerSource,
              "all 64 tags complete and are consumed exactly once");
        Check(pool.Drained() && pool.Residual() == 0,
              "sustained CORE/DCA contention fully drains");
    }

    DcaComputePoolStats core_stats;
    Check(ArbitrationOrder(NocCollDcaArbitration::CORE_PRIORITY,
                           &core_stats) ==
              std::vector<uint64_t>({1, 2, 3, 4}) &&
              core_stats.dca_wait_cycles == 2,
          "core-priority orders requests and accounts DCA contention");
    DcaComputePoolStats dca_stats;
    Check(ArbitrationOrder(NocCollDcaArbitration::DCA_PRIORITY,
                           &dca_stats) ==
              std::vector<uint64_t>({3, 4, 1, 2}) &&
              dca_stats.core_wait_cycles == 2,
          "DCA-priority orders requests and accounts core contention");

    {
        auto config = PoolConfig(NocCollValueMode::TIMING_ONLY, 8, 8, 8);
        DcaComputePool pool(config, 3);
        const auto t1 = pool.TrySubmitAutoTagged(
            DcaRequestSource::DCA, Request(0));
        const auto t2 = pool.TrySubmitAutoTagged(
            DcaRequestSource::DCA, Request(0));
        const auto t3 = pool.TrySubmitAutoTagged(
            DcaRequestSource::DCA, Request(0));
        Check(t1 == 1 && t2 == 2 && t3 == 3,
              "auto tag allocation reaches the configured tag boundary");
        Check(Throws<std::overflow_error>([&] {
                  (void)pool.TrySubmitAutoTagged(
                      DcaRequestSource::DCA, Request(0));
              }),
              "auto tag allocation rejects exhaustion without collision");
        pool.Tick(0);
        pool.Tick(1);
        const auto first = pool.PopResult();
        const auto wrapped = pool.TrySubmitAutoTagged(
            DcaRequestSource::DCA, Request(0));
        Check(first && first->result.tag == 1 && wrapped == 1,
              "auto tag wraps and reuses only a consumed tag");
    }

    {
        auto config = PoolConfig();
        DcaComputePool pool(config);
        pool.TrySubmit(DcaRequestSource::DCA, Request(1));
        Check(Throws<std::invalid_argument>([&] {
                  pool.TrySubmit(DcaRequestSource::CORE, Request(1));
              }),
              "duplicate live request tag is rejected");
        Check(Throws<std::invalid_argument>([&] {
                  (void)pool.TryComplete(1, 0);
              }),
              "completion for a pending unissued tag is rejected");
        Check(Throws<std::invalid_argument>([&] {
                  (void)pool.TryComplete(99, 0);
              }),
              "completion for an unknown tag is rejected");
        pool.Tick(0);
        Check(Throws<std::invalid_argument>([&] {
                  (void)pool.TryComplete(1, 0);
              }),
              "completion before configured latency is rejected");
        pool.Tick(1);
        Check(Throws<std::invalid_argument>([&] {
                  (void)pool.TryComplete(1, 1);
              }),
              "duplicate completion while result is outstanding is rejected");
        (void)pool.PopResult();
        Check(pool.TrySubmit(DcaRequestSource::DCA, Request(1)),
              "consumed result releases tag for deterministic reuse");
        pool.Tick(2);
        pool.Tick(3);
        (void)pool.PopResult();
        Check(pool.Drained() && pool.Stats().completions == 2 &&
                  pool.Stats().results_consumed == 2,
              "tag lifecycle and all finite pool state fully drain");
    }

    {
        auto invalid = PoolConfig();
        invalid.header_fifo_depth = 0;
        Check(Throws<std::invalid_argument>([&] {
                  DcaComputePool pool(invalid);
              }),
              "instantiated ComputePool strongly validates DCA resources");

        auto exact = PoolConfig(NocCollValueMode::INTEGER_EXACT);
        DcaComputePool exact_pool(exact);
        Check(Throws<std::invalid_argument>([&] {
                  exact_pool.TrySubmit(DcaRequestSource::DCA, Request(1));
              }),
              "integer-exact pool requires explicit lane values");
        DcaComputePool timing_pool(PoolConfig());
        Check(Throws<std::invalid_argument>([&] {
                  timing_pool.TrySubmit(
                      DcaRequestSource::DCA,
                      Request(2, CollDType::UINT8, CollReduceOp::SUM,
                              512, 64, true));
              }),
              "timing-only pool rejects accidental value verification");
        Check(Throws<std::invalid_argument>([&] {
                  exact_pool.TrySubmit(
                      DcaRequestSource::DCA,
                      Request(3, CollDType::UINT8, CollReduceOp::SUM,
                              128, 16, true));
              }),
              "request vector width must match the configured compute pool");
    }

    std::cout << "NoC collective refactor R2 self-test: "
              << (failures == 0 ? "PASS" : "FAIL") << " (" << checks
              << " checks)" << std::endl;
    return failures;
}

#ifdef COLL_R2_SELFTEST_MAIN
int main() { return RunCollR2SelfTest(); }
#endif
