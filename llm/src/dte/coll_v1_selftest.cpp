#include "dte/coll_plan.h"
#include "dte/coll_runtime.h"
#include "prims/norm_prims.h"
#include "systemc.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int fails = 0, total = 0;
void Check(bool ok, const std::string &name) {
    ++total; if (!ok) ++fails;
    std::cout << "  [" << (ok ? " ok " : "FAIL") << "] " << name << std::endl;
}
template<class E, class F> bool Throws(F f) {
    try { f(); } catch (const E &) { return true; } catch (...) {} return false;
}
CollDescriptor Make(CollOp op, uint16_t self, uint16_t n = 4) {
    CollDescriptor d;
    d.op = op; d.algorithm = CollAlgorithm::DIRECT; d.dtype = CollDType::UINT8;
    d.reduce_op = CollReduceOp::NONE; d.key = {10, 20, 0};
    for (uint16_t i = 0; i < n; ++i) d.group.push_back(i * 2 + 1);
    d.root_rank = 0; d.self_rank = self; d.count = n;
    d.chunk_bits = 256; d.stride_bits = 256;
    return d;
}
size_t Count(const std::vector<CollAction> &a, CollActionKind kind) {
    size_t n = 0; for (auto &x : a) if (x.kind == kind) ++n; return n;
}
struct BarrierProbe : sc_module {
    uint16_t rank; int delay; bool done = false; sc_time done_at = SC_ZERO_TIME;
    SC_HAS_PROCESS(BarrierProbe);
    BarrierProbe(sc_module_name name, uint16_t r, int d) : sc_module(name), rank(r), delay(d) { SC_THREAD(run); }
    void run() { wait(delay, SC_NS); WaitCollectiveBarrier({77, 88, 1}, 3, rank, 3); done = true; done_at = sc_time_stamp(); }
};
}

int RunCollV1SelfTest() {
    fails = total = 0;
    std::cout << "==== NoC collective V1 self-test ====" << std::endl;
    Check(COLL_TAG_BASE == 0x8000u && COLL_TAG_MAX == 0xfffeu,
          "collective tag namespace occupies reserved high half");
        auto root_b = PlanTier0Collective(Make(CollOp::BROADCAST, 0), 0);
    auto leaf_b = PlanTier0Collective(Make(CollOp::BROADCAST, 2), 2);
    Check(Count(root_b, CollActionKind::SEND) == 3 && Count(root_b, CollActionKind::BARRIER) == 1,
          "broadcast root sends N-1 then barriers");
    Check(Count(leaf_b, CollActionKind::RECV) == 1 && Count(leaf_b, CollActionKind::BARRIER) == 1,
          "broadcast leaf receives once then barriers");

    auto gather_root = PlanTier0Collective(Make(CollOp::GATHER, 0), 0);
    auto gather_leaf = PlanTier0Collective(Make(CollOp::GATHER, 3), 3);
    Check(Count(gather_root, CollActionKind::RECV) == 3 && Count(gather_root, CollActionKind::BARRIER) == 4,
          "gather root receives one source per phase");
    Check(Count(gather_leaf, CollActionKind::SEND) == 1 && Count(gather_leaf, CollActionKind::BARRIER) == 4,
          "gather source waits at every phase boundary");

    auto ag = PlanTier0Collective(Make(CollOp::ALLGATHER, 1, 3), 1);
    Check(Count(ag, CollActionKind::SEND) == 2 && Count(ag, CollActionKind::RECV) == 2 &&
              Count(ag, CollActionKind::BARRIER) == 3,
          "allgather serializes source phases without losing flows");
    auto a2a = PlanTier0Collective(Make(CollOp::ALLTOALL, 2, 3), 2);
    Check(Count(a2a, CollActionKind::SEND) == 2 && Count(a2a, CollActionKind::RECV) == 2 &&
              Count(a2a, CollActionKind::BARRIER) == 3,
          "alltoall covers every ordered peer pair");
    Check(Throws<std::invalid_argument>([] {
              auto d = Make(CollOp::REDUCE, 0);
              d.reduce_op = CollReduceOp::SUM;
              // V3 requires exact count*dtype payload shape; a legacy V1-like
              // descriptor must not accidentally enter the reduction path.
              (void)PlanTier0Collective(d, 0);
          }), "shared planner gates V3 reduction on its stricter contract");

    Collective_prim marker;
    marker.descriptor = Make(CollOp::ALLGATHER, 1, 3);
    marker.phase_id = 2;
    auto wire = marker.serialize();
    Collective_prim decoded;
    decoded.deserialize(wire);
    Check(decoded.phase_id == 2 && decoded.descriptor.key == marker.descriptor.key &&
              decoded.descriptor.group == marker.descriptor.group && decoded.descriptor.self_rank == 1,
          "Collective_prim multi-segment wire round trip");
    auto truncated = wire; truncated.pop_back();
    Check(Throws<std::invalid_argument>([&] { Collective_prim x; x.deserialize(truncated); }),
          "Collective_prim rejects truncated group wire");

    ResetCollectiveBarrierStateForTest();
    auto *p0 = new BarrierProbe("coll_barrier_0", 0, 1);
    auto *p1 = new BarrierProbe("coll_barrier_1", 1, 3);
    auto *p2 = new BarrierProbe("coll_barrier_2", 2, 5);
    sc_start(7, SC_NS);
    Check(p0->done && p1->done && p2->done && p0->done_at == sc_time(5, SC_NS) &&
              p1->done_at == sc_time(5, SC_NS) && p2->done_at == sc_time(5, SC_NS),
          "barrier releases all ranks at last arrival");
    Check(CollectiveBarrierStateCount() == 0, "barrier state drains after all departures");

    std::cout << "NoC collective V1 self-test: "
              << (fails == 0 ? "PASS" : "FAILURES=" + std::to_string(fails))
              << " (" << total << " checks)" << std::endl;
    return fails;
}
