#include "dte/coll_plan.h"
#include "dte/coll_runtime.h"
#include "prims/norm_prims.h"
#include "systemc.h"

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
    try { f(); } catch (const E &) { return true; } catch (...) {}
    return false;
}
CollDescriptor Make(CollOp op, uint16_t self, uint64_t count = 10) {
    CollDescriptor d;
    d.op = op;
    d.algorithm = op == CollOp::REDUCESCATTER ?
        CollAlgorithm::REDUCE_ROOT_SCATTER :
        (op == CollOp::ALLREDUCE ? CollAlgorithm::REDUCE_ROOT_BROADCAST :
         CollAlgorithm::DIRECT);
    d.dtype = CollDType::INT32; d.reduce_op = CollReduceOp::SUM;
    d.key = {30, 40, 0}; d.group = {1, 3, 7};
    d.root_rank = 0; d.self_rank = self; d.count = count;
    d.chunk_bits = count * 32; d.stride_bits = d.chunk_bits;
    d.src_addr = 0x1000; d.dst_addr = 0x2000;
    return d;
}
size_t Count(const std::vector<CollAction> &actions, CollActionKind kind) {
    size_t count = 0;
    for (const auto &action : actions) if (action.kind == kind) ++count;
    return count;
}
struct ReduceProbe : sc_module {
    bool done = false; sc_time done_at = SC_ZERO_TIME;
    SC_HAS_PROCESS(ReduceProbe);
    ReduceProbe(sc_module_name name) : sc_module(name) { SC_THREAD(run); }
    void run() {
        CollDescriptor d = Make(CollOp::REDUCE, 0);
        d.key.collective_id = 99;
        ProcessReduceRxArrival(d, 1);
        ProcessReduceRxArrival(d, 2);
        done = true; done_at = sc_time_stamp();
    }
};
struct DuplicateProbe : sc_module {
    bool rejected = false;
    SC_HAS_PROCESS(DuplicateProbe);
    DuplicateProbe(sc_module_name name) : sc_module(name) { SC_THREAD(run); }
    void run() {
        CollDescriptor d = Make(CollOp::REDUCE, 0);
        d.key.collective_id = 100;
        ProcessReduceRxArrival(d, 1);
        try { ProcessReduceRxArrival(d, 1); }
        catch (const std::runtime_error &) { rejected = true; }
    }
};
}

int RunCollV3SelfTest() {
    fails = total = 0;
    std::cout << "==== NoC collective V3 self-test ====" << std::endl;

    const auto reduce_root = PlanTier0Collective(Make(CollOp::REDUCE, 0), 0);
    const auto reduce_leaf = PlanTier0Collective(Make(CollOp::REDUCE, 2), 2);
    Check(Count(reduce_root, CollActionKind::RECV) == 2 &&
              Count(reduce_root, CollActionKind::REDUCE_COMPUTE) == 1 &&
              Count(reduce_root, CollActionKind::BARRIER) == 4,
          "Reduce root receives N-1, computes once, and phases all ranks");
    Check(Count(reduce_leaf, CollActionKind::SEND) == 1 &&
              Count(reduce_leaf, CollActionKind::REDUCE_COMPUTE) == 0,
          "Reduce leaf sends once and never accounts ALU");

    const auto rs_root = PlanTier0Collective(Make(CollOp::REDUCESCATTER, 0), 0);
    const auto rs_leaf = PlanTier0Collective(Make(CollOp::REDUCESCATTER, 2), 2);
    Check(Count(rs_root, CollActionKind::RECV) == 2 &&
              Count(rs_root, CollActionKind::SEND) == 2 &&
              Count(rs_root, CollActionKind::REDUCE_COMPUTE) == 1,
          "ReduceScatter is fixed root-reduce then scatter");
    auto rs_recv = std::find_if(rs_leaf.begin(), rs_leaf.end(),
        [](const CollAction &a) { return a.kind == CollActionKind::RECV; });
    Check(rs_recv != rs_leaf.end() && rs_recv->payload_bits == 96 &&
              rs_recv->offset_bits == 224,
          "ReduceScatter preserves quotient/remainder tail partition");

    const auto ar_root = PlanTier0Collective(Make(CollOp::ALLREDUCE, 0), 0);
    Check(Count(ar_root, CollActionKind::RECV) == 2 &&
              Count(ar_root, CollActionKind::SEND) == 2 &&
              Count(ar_root, CollActionKind::REDUCE_COMPUTE) == 1,
          "AllReduce is fixed root-reduce then broadcast");

    Check(Throws<std::invalid_argument>([] {
              auto d = Make(CollOp::REDUCE, 0); d.dtype = CollDType::FP32;
              (void)PlanTier0Collective(d, 0);
          }), "V3 rejects unsupported fp32 reduction semantics");
    Check(Throws<std::invalid_argument>([] {
              auto d = Make(CollOp::REDUCE, 0); ++d.chunk_bits;
              (void)PlanTier0Collective(d, 0);
          }), "V3 rejects count/dtype/payload mismatch");
    Check(Throws<std::invalid_argument>([] {
              auto d = Make(CollOp::REDUCE, 0); ++d.src_addr;
              (void)PlanTier0Collective(d, 0);
          }), "V3 rejects dtype-misaligned addresses");

    Reduce_compute_prim compute;
    compute.descriptor = Make(CollOp::ALLREDUCE, 0);
    Reduce_compute_prim decoded; decoded.deserialize(compute.serialize());
    Check(decoded.descriptor.key == compute.descriptor.key &&
              CollTier0ReduceComputeCycles(decoded.descriptor) == 1,
          "explicit root compute wire and 128-lane cycle model round trip");

    Collective_prim marker;
    marker.descriptor = Make(CollOp::REDUCE, 0); marker.phase_id = 2;
    marker.marker_kind = Collective_prim::MarkerKind::REDUCE_ARRIVAL;
    Collective_prim marker_round; marker_round.deserialize(marker.serialize());
    Check(marker_round.marker_kind == Collective_prim::MarkerKind::REDUCE_ARRIVAL,
          "Reduce RX marker discriminator survives CONFIG wire");

    ResetCollectiveReduceRxStateForTest();
    auto *probe = new ReduceProbe("coll_reduce_probe");
    auto *duplicate = new DuplicateProbe("coll_reduce_duplicate_probe");
    sc_start(3, SC_NS);
    Check(probe->done && probe->done_at == sc_time(2, SC_NS),
          "expected bitmap completes with exactly one alignment cycle");
    Check(duplicate->rejected && CollectiveReduceRxStateCount() == 1,
          "Reduce RX expected bitmap rejects a duplicate source");
    ResetCollectiveReduceRxStateForTest();
    Check(CollectiveReduceRxStateCount() == 0,
          "Reduce RX state drains or resets without residual entries");
    Check(Throws<std::invalid_argument>([] {
              ProcessReduceRxArrival(Make(CollOp::REDUCE, 0), 0);
          }), "Reduce RX rejects the root as a network operand source");

    std::cout << "NoC collective V3 self-test: "
              << (fails == 0 ? "PASS" : "FAILURES=" + std::to_string(fails))
              << " (" << total << " checks)" << std::endl;
    return fails;
}
