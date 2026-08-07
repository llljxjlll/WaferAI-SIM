#include "dte/coll_refactor_contract.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
int failures = 0;
int checks = 0;

void Check(bool ok, const std::string &name) {
    ++checks;
    if (!ok) ++failures;
    std::cout << "  [" << (ok ? " ok " : "FAIL") << "] " << name
              << std::endl;
}

template <class E, class F> bool Throws(F f) {
    try {
        f();
    } catch (const E &) {
        return true;
    } catch (...) {
    }
    return false;
}
} // namespace

int RunCollR0SelfTest() {
    using namespace coll_refactor;
    failures = checks = 0;
    std::cout << "==== NoC collective refactor R0 contract self-test ===="
              << std::endl;

    const VectorWork one =
        ComputeVectorWork(129, 1, 512, CollDType::UINT8);
    Check(one.lanes == 64 && one.vector_beats == 3 &&
              one.pairwise_issues_per_beat == 0 && one.total_issues == 0 &&
              one.tail_valid_lanes == 1 &&
              TailLaneMask(one).IsActive(0) &&
              !TailLaneMask(one).IsActive(1),
          "single-input stream bypasses DCA and preserves tail mask");

    const VectorWork two =
        ComputeVectorWork(129, 2, 512, CollDType::UINT8);
    Check(two.lanes == 64 && two.vector_beats == 3 &&
              two.pairwise_issues_per_beat == 1 && two.total_issues == 3,
          "two-input stream issues one SIMD operation per vector beat");

    const VectorWork three =
        ComputeVectorWork(129, 3, 512, CollDType::UINT8);
    Check(three.pairwise_issues_per_beat == 2 &&
              three.total_issues == 6,
          "three-input stream uses two pairwise SIMD issues per beat");

    const VectorWork legacy_width =
        ComputeVectorWork(16, 3, 128, CollDType::UINT8);
    Check(legacy_width.lanes == 16 && legacy_width.vector_beats == 1 &&
              legacy_width.total_issues == 2 &&
              legacy_width.tail_valid_lanes == 16,
          "128-bit UINT8 geometry counts vector issues, not elements");

    Check(Throws<std::invalid_argument>([] {
              (void)ComputeVectorWork(1, 0, 512, CollDType::UINT8);
          }) &&
              Throws<std::invalid_argument>([] {
                  (void)ComputeVectorWork(1, 2, 130, CollDType::INT32);
              }),
          "invalid input count and fractional vector lanes are rejected");

    Check(Throws<std::overflow_error>([] {
              (void)ComputeVectorWork(std::numeric_limits<uint64_t>::max(),
                                      3, 8, CollDType::UINT8);
          }),
          "vector issue multiplication overflow is rejected");

    const DcaTiming timing{7, 1};
    Check(DcaIssueCycle(10, 3, timing) == 13 &&
              DcaCompletionCycle(13, timing) == 20,
          "DCA latency and initiation interval are independent");
    Check(Throws<std::invalid_argument>([] {
              (void)DcaIssueCycle(0, 0, {7, 0});
          }),
          "zero DCA initiation interval is rejected");

    ReduceBeatKey beat{{{1, 2, 3}, 4, 5}, 6, 7};
    const VectorBeat full{CollDType::UINT8, 512, {64, 64}};
    DcaRequest request{9, beat, CollReduceOp::SUM, {full, full}};
    request.Validate();
    Check(true, "tagged two-input DCA request validates");
    Check(Throws<std::invalid_argument>([&] {
              auto invalid = request;
              invalid.operands[1].lane_mask.valid_lanes = 63;
              invalid.Validate();
          }),
          "DCA operands with different vector geometry are rejected");

    const DcaResult result{9, beat, full};
    result.ValidateAgainst(request);
    Check(true, "DCA result tag and beat key match request");
    Check(Throws<std::invalid_argument>([&] {
              auto invalid = result;
              ++invalid.tag;
              invalid.ValidateAgainst(request);
          }),
          "mismatched DCA result tag is rejected");

    ReduceStreamHeader header;
    header.tree_id = 11;
    header.key = {{8, 9, 10}, 1, 2};
    header.dtype = CollDType::UINT8;
    header.op = CollReduceOp::SUM;
    header.total_elements = 129;
    header.physical_data_flits = 9;
    header.vector_beats = 3;
    header.Validate(128, 512);
    Check(true, "stream-v2 header freezes physical flit and vector beat counts");
    Check(Throws<std::invalid_argument>([&] {
              auto invalid = header;
              invalid.physical_data_flits = 10;
              invalid.Validate(128, 512);
          }),
          "stream-v2 header rejects inconsistent payload geometry");

    AsyncSessionProgress progress;
    Check(!progress.Complete(), "async collective session starts incomplete");
    progress.tx_done = progress.rx_done = true;
    progress.inflight_dca = 1;
    Check(!progress.Complete(),
          "async session cannot complete with an inflight DCA result");
    progress.inflight_dca = 0;
    Check(progress.Complete(),
          "async session completes only after TX/RX and all state drain");

    TraceContract trace;
    trace.header_flits = 1;
    trace.data_flits = 9;
    trace.dca_issues = trace.dca_completions = 3;
    trace.dca_inflight_peak = 2;
    trace.Validate();
    TraceContract legacy;
    legacy.generation = TraceGeneration::LEGACY_V5;
    legacy.wire_version = ReduceWireVersion::LEGACY_TWO_SEGMENT;
    legacy.Validate();
    Check(true, "refactor trace counters are generation and wire-version scoped");
    Check(Throws<std::invalid_argument>([] {
              TraceContract invalid;
              invalid.wire_version = ReduceWireVersion::LEGACY_TWO_SEGMENT;
              invalid.Validate();
          }) &&
              Throws<std::invalid_argument>([] {
                  TraceContract invalid;
                  invalid.generation = TraceGeneration::LEGACY_V5;
                  invalid.Validate();
              }),
          "legacy and refactor trace generations cannot mix wire traffic");

    std::cout << "NoC collective refactor R0 self-test: "
              << (failures == 0 ? "PASS" : "FAIL") << " (" << checks
              << " checks)" << std::endl;
    return failures;
}
