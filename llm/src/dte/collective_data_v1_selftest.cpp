#include "dte/collective_data_v1_selftest.h"

#include "dte/collective_data_v1.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct TestState {
    int checks = 0;
    int failures = 0;

    void Check(bool value, const std::string &name) {
        ++checks;
        if (!value) {
            ++failures;
            std::cerr << "[COLLECTIVE DATA V1] FAIL: " << name << '\n';
        }
    }

    template <class Exception = std::exception, class F>
    void Throws(const std::string &name, F &&fn) {
        ++checks;
        try {
            fn();
        } catch (const Exception &) {
            return;
        } catch (...) {
        }
        ++failures;
        std::cerr << "[COLLECTIVE DATA V1] FAIL: " << name << '\n';
    }
};

std::vector<uint8_t> Le32(uint32_t value) {
    return {static_cast<uint8_t>(value),
            static_cast<uint8_t>(value >> 8),
            static_cast<uint8_t>(value >> 16),
            static_cast<uint8_t>(value >> 24)};
}

std::vector<uint8_t> Le64(uint64_t value) {
    std::vector<uint8_t> result(8);
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = static_cast<uint8_t>(value >> (8 * i));
    return result;
}

void TestGather(TestState &state) {
    IsaV1CollectiveDataBuffer buffer(3, 5, 15);
    buffer.Accept(2, 2, {12, 13, 14});
    buffer.Accept(0, 0, {0, 1, 2, 3, 4});
    buffer.Accept(2, 0, {10, 11});
    buffer.Accept(1, 3, {8, 9});
    state.Check(!buffer.Complete(), "gather incomplete before final chunk");
    buffer.Accept(1, 0, {5, 6, 7});
    state.Check(buffer.Complete(), "gather complete out of order");
    const auto gathered = buffer.TakeGathered();
    state.Check(gathered ==
                    std::vector<uint8_t>({0, 1, 2, 3, 4, 5, 6, 7,
                                          8, 9, 10, 11, 12, 13, 14}),
                "gather canonical rank-major bytes");
    state.Check(buffer.Residual().Drained(), "gather take drains buffer");
}

void TestTransactionalErrors(TestState &state) {
    IsaV1CollectiveDataBuffer buffer(2, 4, 8);
    buffer.Accept(0, 1, {1, 2});
    const auto before = buffer.Staging();
    const auto residual = buffer.Residual();
    state.Throws<std::invalid_argument>("duplicate chunk rejected", [&] {
        buffer.Accept(0, 0, {9, 9});
    });
    state.Check(buffer.Staging() == before,
                "duplicate failure preserves staging");
    state.Check(buffer.Residual().received_bytes == residual.received_bytes &&
                    buffer.Residual().accepted_chunks ==
                        residual.accepted_chunks,
                "duplicate failure preserves accounting");
    state.Throws<std::out_of_range>("rank slice overrun rejected", [&] {
        buffer.Accept(1, 3, {1, 2});
    });
    state.Throws<std::invalid_argument>("rank overflow rejected", [&] {
        buffer.Accept(2, 0, {1});
    });
    state.Throws<std::invalid_argument>("empty chunk rejected", [&] {
        buffer.Accept(1, 0, {});
    });
    state.Throws<std::logic_error>("incomplete gather rejected", [&] {
        (void)buffer.TakeGathered();
    });
    buffer.Abort();
    state.Check(buffer.Residual().Drained(), "abort drains partial buffer");

    state.Throws<std::invalid_argument>("zero ranks rejected", [] {
        IsaV1CollectiveDataBuffer invalid(0, 1, 1);
    });
    state.Throws<std::invalid_argument>("zero bytes rejected", [] {
        IsaV1CollectiveDataBuffer invalid(1, 0, 1);
    });
    state.Throws<std::length_error>("capacity rejected", [] {
        IsaV1CollectiveDataBuffer invalid(3, 4, 11);
    });
}

void Fill(IsaV1CollectiveDataBuffer &buffer,
          const std::vector<std::vector<uint8_t>> &operands) {
    for (size_t rank = operands.size(); rank-- > 0;)
        buffer.Accept(static_cast<uint16_t>(rank), 0, operands[rank]);
}

void TestUint8(TestState &state) {
    IsaV1CollectiveDataBuffer sum(3, 4, 12);
    Fill(sum, {{250, 1, 127, 255}, {10, 2, 128, 1}, {1, 3, 255, 2}});
    state.Check(sum.TakeReduced(CollDType::UINT8, CollReduceOp::SUM) ==
                    std::vector<uint8_t>({5, 6, 254, 2}),
                "UINT8 SUM wraps modulo 256");

    IsaV1CollectiveDataBuffer maximum(3, 3, 9);
    Fill(maximum, {{0, 255, 4}, {9, 1, 5}, {8, 254, 6}});
    state.Check(maximum.TakeReduced(CollDType::UINT8, CollReduceOp::MAX) ==
                    std::vector<uint8_t>({9, 255, 6}),
                "UINT8 MAX is unsigned");
}

void TestSignedIntegers(TestState &state) {
    IsaV1CollectiveDataBuffer max32(3, 4, 12);
    Fill(max32, {Le32(UINT32_C(0x80000000)),
                 Le32(UINT32_C(0xffffffff)), Le32(7)});
    state.Check(max32.TakeReduced(CollDType::INT32, CollReduceOp::MAX) ==
                    Le32(7),
                "INT32 MAX orders negative and positive values");

    IsaV1CollectiveDataBuffer negative32(3, 4, 12);
    Fill(negative32, {Le32(UINT32_C(0x80000000)),
                      Le32(UINT32_C(0xfffffffe)),
                      Le32(UINT32_C(0xffffffff))});
    state.Check(negative32.TakeReduced(CollDType::INT32,
                                       CollReduceOp::MAX) ==
                    Le32(UINT32_C(0xffffffff)),
                "INT32 MAX orders negative values");

    IsaV1CollectiveDataBuffer wrap32(2, 4, 8);
    Fill(wrap32, {Le32(UINT32_C(0x7fffffff)), Le32(1)});
    state.Check(wrap32.TakeReduced(CollDType::INT32, CollReduceOp::SUM) ==
                    Le32(UINT32_C(0x80000000)),
                "INT32 SUM wraps in two's complement");

    IsaV1CollectiveDataBuffer max64(3, 8, 24);
    Fill(max64, {Le64(UINT64_C(0x8000000000000000)),
                 Le64(UINT64_C(0xffffffffffffffff)), Le64(0)});
    state.Check(max64.TakeReduced(CollDType::INT64, CollReduceOp::MAX) ==
                    Le64(0),
                "INT64 MAX uses signed two's-complement ordering");

    IsaV1CollectiveDataBuffer wrap64(2, 8, 16);
    Fill(wrap64, {Le64(UINT64_MAX), Le64(1)});
    state.Check(wrap64.TakeReduced(CollDType::INT64, CollReduceOp::SUM) ==
                    Le64(0),
                "INT64 SUM wraps modulo 2^64");
}

void TestReduceValidation(TestState &state) {
    IsaV1CollectiveDataBuffer unaligned(2, 3, 6);
    Fill(unaligned, {{1, 2, 3}, {4, 5, 6}});
    state.Throws<std::invalid_argument>("unaligned INT32 rejected", [&] {
        (void)unaligned.TakeReduced(CollDType::INT32, CollReduceOp::SUM);
    });
    state.Check(unaligned.Complete(),
                "reduce validation failure preserves completed buffer");
    state.Throws<std::invalid_argument>("NONE reduce rejected", [&] {
        (void)unaligned.TakeReduced(CollDType::UINT8, CollReduceOp::NONE);
    });
    state.Throws<std::invalid_argument>("FP reduce rejected", [&] {
        (void)unaligned.TakeReduced(CollDType::FP32, CollReduceOp::SUM);
    });
    unaligned.Abort();

    state.Check(IsaV1ReduceOperationCount(1, 8, CollDType::INT64) == 0,
                "N=1 reduce has zero combines");
    state.Check(IsaV1ReduceOperationCount(4, 32, CollDType::INT32) == 24,
                "reduce operation count is elements*(N-1)");
    state.Throws<std::invalid_argument>("operation count zero ranks", [] {
        (void)IsaV1ReduceOperationCount(0, 4, CollDType::INT32);
    });
    state.Throws<std::invalid_argument>("operation count unaligned", [] {
        (void)IsaV1ReduceOperationCount(2, 3, CollDType::INT32);
    });
}

} // namespace

int RunCollectiveDataV1SelfTest() {
    TestState state;
    TestGather(state);
    TestTransactionalErrors(state);
    TestUint8(state);
    TestSignedIntegers(state);
    TestReduceValidation(state);
    if (state.failures == 0) {
        std::cout << "[COLLECTIVE DATA V1] PASS (" << state.checks
                  << " checks)\n";
        return 0;
    }
    std::cerr << "[COLLECTIVE DATA V1] FAIL (" << state.failures << "/"
              << state.checks << " checks failed)\n";
    return 1;
}

#ifdef COLLECTIVE_DATA_V1_SELFTEST_MAIN
int main() { return RunCollectiveDataV1SelfTest(); }
#endif
