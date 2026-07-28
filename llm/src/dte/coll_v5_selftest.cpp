#include "dte/coll_innetwork_reduce.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
int fails = 0, total = 0;
void Check(bool ok, const std::string &name) {
    ++total; if (!ok) ++fails;
    std::cout << "  [" << (ok ? " ok " : "FAIL") << "] " << name << std::endl;
}
template <class E, class F> bool Throws(F f) {
    try { f(); } catch (const E &) { return true; } catch (...) {} return false;
}
sc_bv<128> U8(uint8_t a, uint8_t b) {
    sc_bv<128> v = 0; v.range(7, 0) = a; v.range(15, 8) = b; return v;
}
}

int RunCollV5SelfTest() {
    fails = total = 0;
    std::cout << "==== NoC collective V5 contract self-test ====" << std::endl;
    CollReduceOperand operand;
    operand.tree_id = 9; operand.collective = {1, 2, 3};
    operand.phase_id = 4; operand.chunk_id = 5; operand.child_id = 6;
    operand.dtype = CollDType::UINT8; operand.op = CollReduceOp::SUM;
    operand.valid_elements = 2; operand.payload = U8(1, 250);
    const auto wire = SerializeCollReduceOperand(operand);
    const auto round = DeserializeCollReduceOperand(wire);
    Check(round.tree_id == 9 && round.collective == operand.collective &&
              round.phase_id == 4 && round.chunk_id == 5 && round.child_id == 6 &&
              round.payload == operand.payload,
          "two-segment reduce operand wire round trip");
    Check(Throws<std::invalid_argument>([&] { auto x = operand; x.dtype = CollDType::FP32; SerializeCollReduceOperand(x); }),
          "FP operand wire is rejected until semantics freeze");
    Check(Throws<std::invalid_argument>([&] { auto x = wire; x[1][200] = true; DeserializeCollReduceOperand(x); }),
          "operand payload reserved bits are checked");

    auto sum = ReduceIntegerOperands({U8(1, 250), U8(2, 10)},
                                     CollDType::UINT8, CollReduceOp::SUM, 2);
    Check(sum.range(7, 0).to_uint() == 3 && sum.range(15, 8).to_uint() == 4,
          "UINT8 SUM is bit-accurate with modular overflow");
    sc_bv<128> neg2 = 0, neg3 = 0;
    neg2.range(31, 0) = uint32_t(-2); neg3.range(31, 0) = uint32_t(-3);
    auto maxv = ReduceIntegerOperands({neg3, neg2}, CollDType::INT32,
                                      CollReduceOp::MAX, 1);
    Check(int32_t(maxv.range(31, 0).to_uint64()) == -2,
          "INT32 MAX uses signed comparison");

    CollReduceMatchKey key{{10, 20, 0}, 2, 7};
    CollOperandMatchBuffer buffer(2, 2);
    Check(Throws<std::invalid_argument>([&] {
              buffer.Open(key, 1, CollDType::FP32, CollReduceOp::SUM, 1);
          }) && Throws<std::invalid_argument>([&] {
              buffer.Open(key, 1, CollDType::INT64, CollReduceOp::SUM, 3);
          }), "match open rejects unsupported dtype and oversized payload");
    buffer.Open(key, (1ull << 1) | (1ull << 3), CollDType::UINT8,
                CollReduceOp::SUM, 2);
    Check(buffer.Accept(key, 1, CollDType::UINT8, CollReduceOp::SUM, 2,
                        U8(1, 250)) == CollOperandStatus::ACCEPTED,
          "first expected child opens a partial match");
    Check(buffer.Accept(key, 1, CollDType::UINT8, CollReduceOp::SUM, 2,
                        U8(1, 1)) == CollOperandStatus::DUPLICATE,
          "duplicate child operand is rejected");
    Check(buffer.Accept(key, 2, CollDType::UINT8, CollReduceOp::SUM, 2,
                        U8(1, 1)) == CollOperandStatus::UNEXPECTED,
          "child outside expected bitmap is rejected");
    Check(buffer.Accept(key, 3, CollDType::INT32, CollReduceOp::SUM, 2,
                        U8(1, 1)) == CollOperandStatus::MISMATCH,
          "dtype/op/count mismatch is rejected");
    Check(Throws<std::runtime_error>([&] { buffer.Consume(key); }),
          "DCA issue before all operands arrive is rejected");
    Check(buffer.Accept(key, 3, CollDType::UINT8, CollReduceOp::SUM, 2,
                        U8(2, 10)) == CollOperandStatus::READY,
          "last expected child marks match ready");
    const auto result = buffer.Consume(key);
    Check(result.payload.range(7, 0).to_uint() == 3 &&
              result.payload.range(15, 8).to_uint() == 4 &&
              result.service_cycles == 56,
          "DCA result reinjection payload and max(comp,p/128)+54 timing are exact");
    Check(buffer.Residual() == 0, "header and operand buffers drain after consume");

    CollOperandMatchBuffer three_way(1, 3);
    three_way.Open(key, 7, CollDType::INT64, CollReduceOp::SUM, 2);
    sc_bv<128> ones = 0; ones.range(63, 0) = 1; ones.range(127, 64) = 1;
    three_way.Accept(key, 0, CollDType::INT64, CollReduceOp::SUM, 2, ones);
    three_way.Accept(key, 1, CollDType::INT64, CollReduceOp::SUM, 2, ones);
    three_way.Accept(key, 2, CollDType::INT64, CollReduceOp::SUM, 2, ones);
    const auto three_result = three_way.Consume(key);
    Check(three_result.payload.range(63, 0).to_uint64() == 3 &&
              three_result.payload.range(127, 64).to_uint64() == 3 &&
              three_result.service_cycles == 58,
          "three-child DCA charges count times pairwise reductions");

    CollOperandMatchBuffer bounded(1, 1);
    bounded.Open(key, (1ull << 0) | (1ull << 1), CollDType::UINT8,
                 CollReduceOp::MAX, 1);
    Check(bounded.Accept(key, 0, CollDType::UINT8, CollReduceOp::MAX, 1,
                         U8(1, 0)) == CollOperandStatus::ACCEPTED &&
              bounded.Accept(key, 1, CollDType::UINT8, CollReduceOp::MAX, 1,
                         U8(2, 0)) == CollOperandStatus::BACKPRESSURE,
          "finite operand capacity asserts backpressure without mutation");
    const CollReduceMatchKey second_key{{11, 20, 0}, 2, 7};
    Check(!bounded.Open(second_key, 1, CollDType::UINT8,
                        CollReduceOp::MAX, 1) &&
              bounded.Residual() == 2,
          "finite header capacity backpressures without mutation");

    CollOperandMatchBuffer header_retry(1, 1);
    Check(header_retry.Open(key, 1, CollDType::UINT8,
                            CollReduceOp::MAX, 1) &&
              !header_retry.Open(second_key, 1, CollDType::UINT8,
                                 CollReduceOp::MAX, 1),
          "full header buffer stalls a new match");
    Check(header_retry.Accept(key, 0, CollDType::UINT8,
                              CollReduceOp::MAX, 1,
                              U8(1, 0)) == CollOperandStatus::READY,
          "active match can complete while a header is backpressured");
    header_retry.Consume(key);
    Check(header_retry.Open(second_key, 1, CollDType::UINT8,
                            CollReduceOp::MAX, 1),
          "backpressured header succeeds after capacity drains");

    std::cout << "NoC collective V5 contract self-test: "
              << (fails == 0 ? "PASS" : "FAILURES=" + std::to_string(fails))
              << " (" << total << " checks)" << std::endl;
    return fails;
}
