#include "dte/coll_dca_payload_v1_selftest.h"

#include "dte/coll_dca_payload_v1.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Suite {
    int checks = 0;
    int failures = 0;

    void Check(bool condition, const std::string &name) {
        ++checks;
        if (condition) return;
        ++failures;
        std::cerr << "[ISA V1 DCA PAYLOAD] FAIL: " << name << '\n';
    }

    template <class E = std::exception, class F>
    void Throws(F &&function, const std::string &name) {
        bool threw = false;
        try {
            function();
        } catch (const E &) {
            threw = true;
        }
        Check(threw, name);
    }
};

std::vector<uint8_t> Pattern(size_t size) {
    std::vector<uint8_t> result(size);
    for (size_t index = 0; index < size; ++index)
        result[index] = static_cast<uint8_t>((index * 73 + 19) & 0xff);
    return result;
}

void RoundTrip(Suite &suite, CollDType dtype, size_t size,
               uint64_t vector_bits) {
    const std::vector<uint8_t> bytes = Pattern(size);
    const auto built = coll_refactor::BuildIsaV1DcaByteStream(
        7, {1, 2, 3}, 4, 5, 6, dtype, CollReduceOp::SUM, bytes,
        vector_bits);
    suite.Check(coll_refactor::DecodeIsaV1DcaByteStream(
                    built.header, built.beats, vector_bits) == bytes,
                "SRAM bytes round-trip through exact DCA lane packing");
    size_t flits = 0;
    for (const auto &beat : built.beats)
        flits += coll_refactor::SplitReduceVectorBeat(
                     built.header, beat, 128, vector_bits)
                     .size();
    suite.Check(flits == built.header.stream.physical_data_flits,
                "packed byte stream has the exact physical flit count");
}

} // namespace

int RunIsaV1DcaPayloadSelfTest() {
    Suite suite;
    RoundTrip(suite, CollDType::UINT8, 1, 512);
    RoundTrip(suite, CollDType::UINT8, 65, 512);
    RoundTrip(suite, CollDType::INT32, 4, 512);
    RoundTrip(suite, CollDType::INT32, 68, 512);
    RoundTrip(suite, CollDType::INT64, 8, 512);
    RoundTrip(suite, CollDType::INT64, 72, 512);
    RoundTrip(suite, CollDType::INT64, 136, 1024);

    suite.Throws<std::invalid_argument>(
        [] {
            (void)coll_refactor::BuildIsaV1DcaByteStream(
                1, {1, 2, 3}, 0, 1, 1, CollDType::INT32,
                CollReduceOp::SUM, std::vector<uint8_t>(5), 512);
        },
        "partial integer element is rejected");
    suite.Throws<std::invalid_argument>(
        [] {
            (void)coll_refactor::BuildIsaV1DcaByteStream(
                1, {1, 2, 3}, 0, 1, 1, CollDType::FP32,
                CollReduceOp::SUM, std::vector<uint8_t>(4), 512);
        },
        "floating DCA byte mode is rejected");
    suite.Throws<std::invalid_argument>(
        [] {
            (void)coll_refactor::BuildIsaV1DcaByteStream(
                0, {1, 2, 3}, 0, 1, 1, CollDType::UINT8,
                CollReduceOp::SUM, std::vector<uint8_t>(1), 512);
        },
        "reserved tree zero is rejected");

    auto built = coll_refactor::BuildIsaV1DcaByteStream(
        9, {1, 2, 3}, 4, 5, 6, CollDType::INT32,
        CollReduceOp::MAX, Pattern(68), 512);
    built.beats.back().key.vector_beat_id = 0;
    suite.Throws<std::invalid_argument>(
        [&] {
            (void)coll_refactor::DecodeIsaV1DcaByteStream(
                built.header, built.beats, 512);
        },
        "non-contiguous result beat identity is rejected");
    built = coll_refactor::BuildIsaV1DcaByteStream(
        9, {1, 2, 3}, 4, 5, 6, CollDType::INT32,
        CollReduceOp::MAX, Pattern(68), 512);
    built.beats.pop_back();
    suite.Throws<std::invalid_argument>(
        [&] {
            (void)coll_refactor::DecodeIsaV1DcaByteStream(
                built.header, built.beats, 512);
        },
        "truncated result beat vector is rejected");

    if (suite.failures == 0)
        std::cout << "ISA v1 DCA payload self-test: PASS ("
                  << suite.checks << " checks)\n";
    return suite.failures;
}

#ifdef ISA_V1_COLL_DCA_PAYLOAD_SELFTEST_MAIN
int sc_main(int, char *[]) { return RunIsaV1DcaPayloadSelfTest(); }
#endif
