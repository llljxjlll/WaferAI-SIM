#include "dte/coll_codec.h"
#include "dte/coll_latency.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int failures = 0;
int checks = 0;
void Check(bool ok, const std::string &name) {
    ++checks;
    if (!ok) ++failures;
    std::cout << "  [" << (ok ? " ok " : "FAIL") << "] " << name << std::endl;
}
template <class E, class F> bool Throws(F f) {
    try { f(); } catch (const E &) { return true; } catch (...) {}
    return false;
}
CollDescriptor BaseDescriptor() {
    CollDescriptor d;
    d.op = CollOp::ALLREDUCE;
    d.algorithm = CollAlgorithm::REDUCE_ROOT_BROADCAST;
    d.dtype = CollDType::INT32;
    d.reduce_op = CollReduceOp::SUM;
    d.key = {17, 23, 5};
    d.group = {1, 3, 7, 9, 12, 15, 18, 21, 25};
    d.root_rank = 2;
    d.self_rank = 4;
    d.count = 1027;
    d.src_addr = 0x100000000ULL;
    d.dst_addr = 0x200000080ULL;
    d.chunk_bits = 4096;
    d.stride_bits = 8192;
    d.gather_reorder_depth = 6;
    return d;
}
} // namespace

int RunCollV0SelfTest() {
    failures = 0;
    checks = 0;
    std::cout << "==== NoC collective V0 self-test ====" << std::endl;

    Check(CollCeilDiv(0, 128) == 0 && CollCeilDiv(1, 128) == 1 &&
              CollCeilDiv(129, 128) == 2, "ceil division boundaries");
    Check(CollUnicastCycles(129, 128, 3) == 5, "unicast formula");
    Check(CollTier0BroadcastCycles(4, 256, 128, 3) == 9,
          "Tier0 broadcast source serialization");
    Check(CollTier0BroadcastCycles(1, 256, 128, 3) == 3,
          "single-rank broadcast has no source payload");
    Check(CollTier0ScatterCycles(4, 513, 128, 3) == 8,
          "Tier0 scatter requirement formula");
    Check(CollGatherEndpointCycles() == 1 && CollReduceEndpointCycles() == 1,
          "endpoint service excludes arrival synchronization");
    Check(CollDcaServiceCycles(129, 1) == 56 &&
              CollDcaServiceCycles(128, 9) == 63,
          "DCA max(compute, transfer) plus pipeline");
    Check(Throws<std::invalid_argument>([] { (void)CollCeilDiv(1, 0); }),
          "zero bandwidth rejected");
    Check(Throws<std::invalid_argument>([] {
              (void)CollTier0BroadcastCycles(0, 1, 1, 0);
          }), "empty broadcast group rejected");
    Check(Throws<std::overflow_error>([] {
              (void)CollTier0BroadcastCycles(3,
                  std::numeric_limits<uint64_t>::max(), 1, 0);
          }), "broadcast multiplication overflow rejected");
    Check(Throws<std::overflow_error>([] {
              (void)CollUnicastCycles(std::numeric_limits<uint64_t>::max(), 1, 1);
          }), "latency addition overflow rejected");

    Check(CollRankCountOffset(10, 3, 0) == std::make_pair<uint64_t,uint64_t>(4, 0) &&
              CollRankCountOffset(10, 3, 1) == std::make_pair<uint64_t,uint64_t>(3, 4) &&
              CollRankCountOffset(10, 3, 2) == std::make_pair<uint64_t,uint64_t>(3, 7),
          "non-divisible count uses quotient/remainder partition");
    Check(Throws<std::invalid_argument>([] { (void)CollRankCountOffset(1, 0, 0); }),
          "partition rejects empty group");

    CollDescriptor d = BaseDescriptor();
    Check(!Throws<std::exception>([&] { ValidateCollDescriptor(d); }),
          "valid descriptor accepted");
    const auto wire = SerializeCollDescriptor(d);
    Check(wire.size() == 7, "nine-member group uses two group segments");
    const CollDescriptor round = DeserializeCollDescriptor(wire);
    Check(round.op == d.op && round.algorithm == d.algorithm &&
              round.dtype == d.dtype && round.reduce_op == d.reduce_op,
          "wire preserves operation enums");
    Check(round.key == d.key && round.root_rank == d.root_rank &&
              round.self_rank == d.self_rank, "wire preserves identity and ranks");
    Check(round.group == d.group, "wire preserves non-contiguous group");
    Check(round.count == d.count && round.chunk_bits == d.chunk_bits &&
              round.stride_bits == d.stride_bits, "wire preserves sizes");
    Check(round.src_addr == d.src_addr && round.dst_addr == d.dst_addr &&
              round.gather_reorder_depth == d.gather_reorder_depth,
          "wire preserves addresses and reorder depth");

    auto truncated = wire; truncated.pop_back();
    Check(Throws<std::invalid_argument>([&] { (void)DeserializeCollDescriptor(truncated); }),
          "truncated group segments rejected");
    auto extra = wire; extra.push_back(sc_bv<128>(0));
    Check(Throws<std::invalid_argument>([&] { (void)DeserializeCollDescriptor(extra); }),
          "extra group segment rejected");
    auto bad_magic = wire; bad_magic[0].range(15, 0) = 0;
    Check(Throws<std::invalid_argument>([&] { (void)DeserializeCollDescriptor(bad_magic); }),
          "bad wire magic rejected");
    auto bad_enum = wire; bad_enum[0].range(31, 24) = 255;
    Check(Throws<std::invalid_argument>([&] { (void)DeserializeCollDescriptor(bad_enum); }),
          "out-of-range wire enum rejected");

    CollDescriptor invalid = d; invalid.group = {1, 1}; invalid.root_rank = 0; invalid.self_rank = 0;
    Check(Throws<std::invalid_argument>([&] { ValidateCollDescriptor(invalid); }),
          "duplicate group member rejected");
    invalid = d; invalid.group = {3, 1}; invalid.root_rank = 0; invalid.self_rank = 0;
    Check(Throws<std::invalid_argument>([&] { ValidateCollDescriptor(invalid); }),
          "unsorted group rejected");
    invalid = d; invalid.root_rank = static_cast<uint16_t>(invalid.group.size());
    Check(Throws<std::invalid_argument>([&] { ValidateCollDescriptor(invalid); }),
          "root outside group rejected");
    invalid = d; invalid.reduce_op = CollReduceOp::NONE;
    Check(Throws<std::invalid_argument>([&] { ValidateCollDescriptor(invalid); }),
          "reduction without reduce op rejected");
    invalid = d; invalid.op = CollOp::ALLGATHER; invalid.algorithm = CollAlgorithm::DIRECT;
    Check(Throws<std::invalid_argument>([&] { ValidateCollDescriptor(invalid); }),
          "non-reduction with reduce op rejected");
    invalid = d; invalid.algorithm = CollAlgorithm::DIRECT;
    Check(Throws<std::invalid_argument>([&] { ValidateCollDescriptor(invalid); }),
          "AllReduce fixed algorithm enforced");

    CollectiveKey a{1, 2, 3}, b{1, 2, 4};
    Check(a < b && !(a == b), "epoch isolates consecutive collective instances");
    PacketKey p1{a, 0, 7, 1, 2}, p2{a, 0, 7, 1, 3};
    Check(!(p1 == p2), "PacketKey isolates destination rank");

    std::cout << "NoC collective V0 self-test: "
              << (failures == 0 ? "PASS" : "FAILURES=" + std::to_string(failures))
              << " (" << checks << " checks)" << std::endl;
    return failures;
}
