#include "dte/coll_innetwork_reduce.h"
#include "dte/coll_multicast.h"
#include "dte/coll_runtime.h"
#include "prims/norm_prims.h"
#include "systemc.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
int fails = 0, total = 0;
void Check(bool ok, const std::string &name) {
    ++total;
    if (!ok) ++fails;
    std::cout << "  [" << (ok ? " ok " : "FAIL") << "] " << name
              << std::endl;
}
template <class E, class F> bool Throws(F f) {
    try { f(); } catch (const E &) { return true; } catch (...) {}
    return false;
}

struct ReleaseProbe : sc_module {
    uint16_t rank;
    uint16_t tree;
    int delay;
    bool done = false;
    SC_HAS_PROCESS(ReleaseProbe);
    ReleaseProbe(sc_module_name name, uint16_t r, uint16_t t, int d)
        : sc_module(name), rank(r), tree(t), delay(d) {
        SC_THREAD(run);
    }
    void run() {
        wait(delay, SC_NS);
        WaitCollectiveBarrier({91, 92, 0}, 7, rank, 2, tree);
        done = true;
    }
};

struct ReleaseMismatchProbe : sc_module {
    uint16_t rank;
    int delay;
    bool mismatch_rejected = false;
    bool done = false;
    SC_HAS_PROCESS(ReleaseMismatchProbe);
    ReleaseMismatchProbe(sc_module_name name, uint16_t r, int d)
        : sc_module(name), rank(r), delay(d) {
        SC_THREAD(run);
    }
    void run() {
        wait(delay, SC_NS);
        if (rank == 0) {
            WaitCollectiveBarrier({91, 93, 0}, 8, rank, 2, 0);
        } else {
            mismatch_rejected = Throws<std::runtime_error>([] {
                WaitCollectiveBarrier({91, 93, 0}, 8, 1, 2, 777);
            });
            WaitCollectiveBarrier({91, 93, 0}, 8, rank, 2, 0);
        }
        done = true;
    }
};

struct UnknownReleaseProbe : sc_module {
    bool rejected = false;
    SC_HAS_PROCESS(UnknownReleaseProbe);
    explicit UnknownReleaseProbe(sc_module_name name) : sc_module(name) {
        SC_THREAD(run);
    }
    void run() {
        wait(10, SC_NS);
        rejected = Throws<std::runtime_error>([] {
            WaitCollectiveBarrier({91, 94, 0}, 9, 0, 1, 65530);
        });
    }
};

CollDescriptor Descriptor() {
    CollDescriptor d;
    d.op = CollOp::BROADCAST;
    d.algorithm = CollAlgorithm::DIRECT;
    d.dtype = CollDType::UINT8;
    d.reduce_op = CollReduceOp::NONE;
    d.key = {91, 92, 0};
    d.group = {0, 1};
    d.root_rank = 0;
    d.self_rank = 0;
    d.count = 1;
    d.chunk_bits = 128;
    d.stride_bits = 128;
    return d;
}
} // namespace

int RunCollV6SelfTest() {
    fails = total = 0;
    std::cout << "==== NoC collective V6 integration self-test ===="
              << std::endl;
    ResetCollectiveFabric();
    ResetCollectiveReduceFabric();
    ResetCollectiveBarrierStateForTest();

    ProgramCollectiveTreeEntry({201, 0, CENTER}, 1u << EAST);
    ProgramCollectiveTreeEntry({201, 1, WEST}, 1u << CENTER);
    ProgramCollectiveTreeEntry({202, 0, CENTER}, 1u << NORTH);
    ProgramCollectiveReduceNode(201, 0, {1u << CENTER, CENTER});
    ProgramCollectiveReduceNode(202, 0, {1u << CENTER, CENTER});
    Check(CollectiveTreeEntryCount() == 3 &&
              CollectiveReduceNodeCount() == 2,
          "multiple multicast/reduce trees coexist in finite registries");
    Check(LookupCollectiveTreeEntry({201, 0, CENTER}) == (1u << EAST) &&
              LookupCollectiveTreeEntry({202, 0, CENTER}) == (1u << NORTH),
          "tree lookup remains isolated by tree ID");

    RecordCollectiveForkAttempt(201, 0, 1u << EAST, true);
    RecordCollectiveForkAttempt(201, 0, 1u << EAST, false);
    RecordCollectiveForkAttempt(202, 0, 1u << NORTH, true);
    const auto stats = CollectiveFabricLinkStats();
    Check(stats.size() == 2 && stats[0].committed_flits == 1 &&
              stats[0].stalled_attempts == 1 &&
              stats[1].committed_flits == 1,
          "per-tree link counters distinguish commit and backpressure");

    Collective_prim marker;
    marker.descriptor = Descriptor();
    marker.phase_id = 7;
    marker.release_tree_id = 201;
    Collective_prim decoded;
    decoded.deserialize(marker.serialize());
    Check(decoded.release_tree_id == 201 && decoded.phase_id == 7,
          "final-barrier tree release survives CONFIG wire round trip");
    auto illegal_wire = marker.serialize();
    illegal_wire.front().range(47, 40) =
        static_cast<uint8_t>(Collective_prim::MarkerKind::GATHER_ARRIVAL);
    Check(Throws<std::invalid_argument>([&] {
              Collective_prim illegal;
              illegal.deserialize(illegal_wire);
          }),
          "non-barrier marker cannot carry a tree release on CONFIG wire");

    auto *p0 = new ReleaseProbe("coll_v6_release_0", 0, 201, 1);
    auto *p1 = new ReleaseProbe("coll_v6_release_1", 1, 201, 3);
    auto *m0 = new ReleaseMismatchProbe("coll_v6_mismatch_0", 0, 6);
    auto *m1 = new ReleaseMismatchProbe("coll_v6_mismatch_1", 1, 8);
    auto *unknown = new UnknownReleaseProbe("coll_v6_unknown_release");
    sc_start(12, SC_NS);
    Check(p0->done && p1->done && CollectiveBarrierStateCount() == 0,
          "last barrier departure drains runtime barrier state");
    Check(m0->done && m1->done && m1->mismatch_rejected,
          "all ranks must agree on the final-barrier release tree");
    Check(unknown->rejected,
          "final barrier rejects release of an unknown tree");
    Check(CollectiveTreeEntryCount() == 1 &&
              CollectiveReduceNodeCount() == 1 &&
              LookupCollectiveTreeEntry({202, 0, CENTER}) == (1u << NORTH),
          "final barrier releases only its own tree instance");
    Check(Throws<std::runtime_error>([] {
              (void)LookupCollectiveTreeEntry({201, 0, CENTER});
          }),
          "released tree cannot be used after lifecycle completion");
    Check(EraseCollectiveTree(202) == 1 &&
              EraseCollectiveReduceTree(202) == 1 &&
              CollectiveTreeEntryCount() == 0 &&
              CollectiveReduceNodeCount() == 0,
          "all production tree state reaches zero");

    std::cout << "NoC collective V6 integration self-test: "
              << (fails == 0 ? "PASS" : "FAILURES=" + std::to_string(fails))
              << " (" << total << " checks)" << std::endl;
    return fails;
}
