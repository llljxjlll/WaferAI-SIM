#include "dte/coll_compute_pool.h"

#include <iostream>
#include <limits>
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

template <class F> bool Throws(F f) {
    try { f(); } catch (const std::invalid_argument &) { return true; }
    return false;
}

NocCollDcaConfig Config(NocCollValueMode mode) {
    NocCollDcaConfig config;
    config.value_mode = mode;
    config.operand_fifo_depth = 16;
    config.header_fifo_depth = 16;
    config.result_fifo_depth = 16;
    config.timing[3][0] = {4, 1};
    return config;
}

DcaPoolRequest Request(uint64_t tag, DcaRequestSource source,
                       NocCollValueMode mode) {
    (void)source;
    VectorBeat beat{CollDType::FP32, 512, {16, 16}};
    DcaPoolRequest request;
    request.request.tag = tag;
    request.request.key = {{{5, static_cast<uint32_t>(tag), 1}, 0, 0},
                           0, tag};
    request.request.op = CollReduceOp::SUM;
    request.request.operands = {beat, beat};
    if (mode != NocCollValueMode::TIMING_ONLY) {
        request.operand_values[0].assign(16, CollFp32Bits(1.25f));
        request.operand_values[1].assign(16, CollFp32Bits(2.5f));
    }
    return request;
}
} // namespace

int RunCollR5SelfTest() {
    using namespace coll_refactor;
    failures = checks = 0;
    std::cout << "==== NoC collective refactor R5 shared vector/FP self-test ===="
              << std::endl;

    const VectorLaneMask lanes{4, 3};
    const auto sum = ReduceFp32Vector(
        CollReduceOp::SUM, lanes,
        {CollFp32Bits(1.5f), CollFp32Bits(-0.0f),
         CollFp32Bits(std::numeric_limits<float>::infinity()), 0x1234},
        {CollFp32Bits(2.25f), CollFp32Bits(-0.0f),
         CollFp32Bits(-std::numeric_limits<float>::infinity()), 0x5678});
    Check(sum[0] == CollFp32Bits(3.75f) &&
              sum[1] == CollFp32Bits(-0.0f) &&
              sum[2] == 0x7fc00000u && sum[3] == 0x1234,
          "FP32 SUM freezes rounding, canonical NaN and tail preservation");

    const auto maximum = ReduceFp32Vector(
        CollReduceOp::MAX, {3, 3},
        {CollFp32Bits(-0.0f), 0x7fa12345u, CollFp32Bits(-2.0f)},
        {CollFp32Bits(0.0f), CollFp32Bits(9.0f), CollFp32Bits(-3.0f)});
    Check(maximum == std::vector<uint64_t>(
              {CollFp32Bits(0.0f), 0x7fc00000u, CollFp32Bits(-2.0f)}),
          "FP32 MAX freezes signed zero, NaN and ordering semantics");
    Check(ReduceFp32Vector(CollReduceOp::SUM, lanes,
                          {CollFp32Bits(1.5f), CollFp32Bits(-0.0f),
                           CollFp32Bits(std::numeric_limits<float>::infinity()),
                           0x1234},
                          {CollFp32Bits(2.25f), CollFp32Bits(-0.0f),
                           CollFp32Bits(-std::numeric_limits<float>::infinity()),
                           0x5678}) == sum,
          "fixed FP pairwise order is bit-identical across runs");

    {
        auto config = Config(NocCollValueMode::FP_EXACT);
        DcaComputePool pool(config);
        Check(pool.TrySubmit(DcaRequestSource::CORE,
                             Request(1, DcaRequestSource::CORE,
                                     NocCollValueMode::FP_EXACT)) &&
                  pool.TrySubmit(DcaRequestSource::DCA,
                             Request(2, DcaRequestSource::DCA,
                                     NocCollValueMode::FP_EXACT)),
              "endpoint CORE and router DCA requests enter one finite pool");
        std::vector<DcaPoolResult> results;
        for (uint64_t cycle = 0; cycle <= 5; ++cycle) {
            pool.Tick(cycle);
            while (auto result = pool.PopResult())
                results.push_back(std::move(*result));
        }
        Check(results.size() == 2 &&
                  results[0].source == DcaRequestSource::CORE &&
                  results[1].source == DcaRequestSource::DCA &&
                  results[0].result.tag == 1 &&
                  results[1].result.tag == 2,
              "shared arbitration preserves CORE/DCA result identity");
        Check(results[0].issue_cycle == 0 && results[1].issue_cycle == 1 &&
                  results[1].scheduled_completion_cycle == 5 &&
                  results[1].values[0] == CollFp32Bits(3.75f),
              "shared pool applies one FP L/II table to both request sources");
        Check(pool.Stats().core_issued == 1 &&
                  pool.Stats().dca_issued == 1 &&
                  pool.Stats().dca_wait_cycles == 1 && pool.Drained(),
              "CORE/DCA issue, stall and drain statistics are separated");
    }
    {
        auto config = Config(NocCollValueMode::TIMING_ONLY);
        DcaComputePool pool(config);
        auto bad = Request(1, DcaRequestSource::DCA,
                           NocCollValueMode::FP_EXACT);
        Check(Throws([&] { pool.TrySubmit(DcaRequestSource::DCA, bad); }),
              "FP timing-only mode rejects value assertion payloads");
        auto request = Request(2, DcaRequestSource::DCA,
                               NocCollValueMode::TIMING_ONLY);
        pool.TrySubmit(DcaRequestSource::DCA, request);
        std::optional<DcaPoolResult> result;
        for (uint64_t cycle = 0; cycle <= 4; ++cycle) {
            pool.Tick(cycle);
            if (auto ready = pool.PopResult()) result = std::move(*ready);
        }
        Check(result && result->values.empty() && pool.Drained(),
              "FP timing-only produces timing/tag but no asserted value");
    }
    {
        auto fp = Config(NocCollValueMode::FP_EXACT);
        auto integer = Config(NocCollValueMode::INTEGER_EXACT);
        auto timing = Config(NocCollValueMode::TIMING_ONLY);
        Check(!Throws([&] { fp.ValidateForDtype(CollDType::FP32); }) &&
                  Throws([&] { fp.ValidateForDtype(CollDType::INT32); }) &&
                  Throws([&] { integer.ValidateForDtype(CollDType::FP32); }) &&
                  !Throws([&] { timing.ValidateForDtype(CollDType::FP16); }) &&
                  !Throws([&] { timing.ValidateForDtype(CollDType::FP8); }) &&
                  ComputeVectorWork(65, 2, 512, CollDType::FP8).lanes == 64,
              "value mode and workload dtype compatibility is startup-gated");
    }

    std::cout << "R5 self-test: " << checks - failures << "/" << checks
              << " checks passed" << std::endl;
    return failures;
}
