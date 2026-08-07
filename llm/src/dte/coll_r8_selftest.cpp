#include "dte/coll_refactor_contract.h"

#include <array>
#include <cstdint>
#include <iostream>
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
} // namespace

int RunCollR8SelfTest() {
    using namespace coll_refactor;
    failures = checks = 0;
    std::cout << "==== NoC collective refactor R8 oracle self-test ===="
              << std::endl;

    const std::array<uint64_t, 4> payload_bits{
        5120, 8192, 65536, 32 * 1024 * 8};
    const std::array<uint64_t, 4> expected_beats{10, 16, 128, 512};
    for (size_t i = 0; i < payload_bits.size(); ++i) {
        const auto work = ComputeVectorWork(
            payload_bits[i] / 8, 4, 512, CollDType::UINT8);
        Check(work.vector_beats == expected_beats[i] &&
                  work.total_issues == 3 * expected_beats[i],
              "payload " + std::to_string(payload_bits[i]) +
                  " bit has exact vector-beat/tree issue count");
    }

    const uint64_t flits = CollCeilDiv(payload_bits.back(), 128);
    Check(flits == 2048 && 3 * (1 + flits) == 6147,
          "32 KiB reduce stream has one header plus data per tree edge");

    const uint64_t issues = 3 * expected_beats.back();
    const DcaTiming timing{7, 1};
    const uint64_t pipeline_cycles =
        timing.latency + (issues - 1) * timing.initiation_interval;
    Check(issues == 1536 && pipeline_cycles == 1542 &&
              pipeline_cycles < issues * timing.latency,
          "fixed DCA latency is one pipeline fill, not one charge per chunk");

    std::cout << "R8 self-test: " << checks - failures << "/" << checks
              << " checks passed" << std::endl;
    return failures;
}
