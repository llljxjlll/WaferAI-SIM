#include "prims/collective_phase_barrier_v1_prim.h"
#include "prims/collective_phase_barrier_v1_prim_selftest.h"

#include "dte/coll_innetwork_reduce.h"
#include "dte/coll_multicast.h"
#include "dte/coll_runtime.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

struct Probe : sc_module {
    Collective_phase_barrier_v1_prim prim;
    int delay_ns = 0;
    bool done = false;
    bool rejected = false;
    sc_time done_at = SC_ZERO_TIME;

    SC_HAS_PROCESS(Probe);
    Probe(sc_module_name name, CollectiveKey key, uint16_t phase,
          uint16_t rank, uint16_t size, uint16_t tree, int delay)
        : sc_module(name), delay_ns(delay) {
        prim.key = key;
        prim.phase_id = phase;
        prim.rank = rank;
        prim.group_size = size;
        prim.release_tree_id = tree;
        SC_THREAD(Run);
    }

    void Run() {
        wait(delay_ns, SC_NS);
        static int legacy_sram_address = 0;
        TaskCoreContext context(
            nullptr, nullptr, nullptr, nullptr, &legacy_sram_address,
            nullptr, nullptr, nullptr, nullptr, uint64_t{0}, unsigned{0});
        try {
            rejected = prim.taskCoreDefault(context) != 0;
        } catch (const std::exception &) {
            rejected = true;
        }
        done = true;
        done_at = sc_time_stamp();
    }
};

struct LoopProbe : sc_module {
    CollectiveKey key;
    uint16_t rank = 0;
    uint16_t group_size = 0;
    uint16_t phase_base = 0;
    uint32_t epoch_base = 0;
    uint32_t iterations = 0;
    bool advance_epoch = false;
    int delay_ns = 0;
    bool done = false;
    bool rejected = false;

    SC_HAS_PROCESS(LoopProbe);
    LoopProbe(sc_module_name name, CollectiveKey value, uint16_t rank_value,
              uint16_t size, uint16_t first_phase, uint32_t first_epoch,
              uint32_t count, bool epochs, int delay)
        : sc_module(name), key(value), rank(rank_value), group_size(size),
          phase_base(first_phase), epoch_base(first_epoch),
          iterations(count), advance_epoch(epochs), delay_ns(delay) {
        SC_THREAD(Run);
    }

    void Run() {
        wait(delay_ns, SC_NS);
        static int legacy_sram_address = 0;
        TaskCoreContext context(
            nullptr, nullptr, nullptr, nullptr, &legacy_sram_address,
            nullptr, nullptr, nullptr, nullptr, uint64_t{0}, unsigned{0});
        try {
            for (uint32_t index = 0; index < iterations; ++index) {
                Collective_phase_barrier_v1_prim prim;
                prim.key = key;
                prim.key.epoch = advance_epoch
                                     ? epoch_base + index
                                     : epoch_base;
                prim.phase_id = advance_epoch
                                    ? phase_base
                                    : static_cast<uint16_t>(phase_base + index);
                prim.rank = rank;
                prim.group_size = group_size;
                if (prim.taskCoreDefault(context) != 0)
                    throw std::runtime_error(
                        "collective phase barrier returned non-zero");
            }
        } catch (const std::exception &) {
            rejected = true;
        }
        done = true;
    }
};

struct Suite {
    int checks = 0;
    int failures = 0;

    void Check(bool condition, const std::string &name) {
        ++checks;
        if (condition) return;
        ++failures;
        std::cerr << "[COLLECTIVE PHASE BARRIER V1] FAIL: "
                  << name << '\n';
    }
};

} // namespace

int RunCollectivePhaseBarrierV1PrimSelfTest() {
    ResetCollectiveBarrierStateForTest();
    ResetCollectiveFabric();
    ResetCollectiveReduceFabric();

    ProgramCollectiveTreeEntry({321, 0, CENTER}, 1u << EAST);
    ProgramCollectiveReduceNode(321, 0, {1u << CENTER, CENTER});

    auto *n1 = new Probe("phase_barrier_n1", {1, 1, 0}, 0,
                         0, 1, 0, 1);
    auto *n2_fast = new Probe("phase_barrier_n2_fast", {2, 2, 0}, 0,
                              0, 2, 0, 2);
    auto *n2_slow = new Probe("phase_barrier_n2_slow", {2, 2, 0}, 0,
                              1, 2, 0, 5);
    auto *n4_0 = new Probe("phase_barrier_n4_0", {3, 3, 0}, 1,
                           0, 4, 0, 7);
    auto *n4_1 = new Probe("phase_barrier_n4_1", {3, 3, 0}, 1,
                           1, 4, 0, 8);
    auto *n4_2 = new Probe("phase_barrier_n4_2", {3, 3, 0}, 1,
                           2, 4, 0, 9);
    auto *n4_3 = new Probe("phase_barrier_n4_3", {3, 3, 0}, 1,
                           3, 4, 0, 10);

    auto *phase0 = new LoopProbe("phase_barrier_1000_0", {4, 4, 0},
                                 0, 2, 0, 0, 1000, false, 12);
    auto *phase1 = new LoopProbe("phase_barrier_1000_1", {4, 4, 0},
                                 1, 2, 0, 0, 1000, false, 14);
    auto *epoch0 = new LoopProbe("phase_barrier_epoch_0", {5, 5, 0},
                                 0, 2, 1001, 20, 64, true, 16);
    auto *epoch1 = new LoopProbe("phase_barrier_epoch_1", {5, 5, 0},
                                 1, 2, 1001, 20, 64, true, 18);

    // A mismatched key is an independent barrier and must not release the
    // other key.  Later repair arrivals complete both isolated instances.
    auto *key_a0 = new Probe("phase_barrier_key_a0", {6, 6, 0}, 3,
                             0, 2, 0, 20);
    auto *key_b1 = new Probe("phase_barrier_key_b1", {6, 7, 0}, 3,
                             1, 2, 0, 21);
    auto *key_a1 = new Probe("phase_barrier_key_a1", {6, 6, 0}, 3,
                             1, 2, 0, 24);
    auto *key_b0 = new Probe("phase_barrier_key_b0", {6, 7, 0}, 3,
                             0, 2, 0, 25);

    auto *duplicate_first = new Probe(
        "phase_barrier_duplicate_first", {7, 7, 0}, 4,
        0, 2, 0, 27);
    auto *duplicate_bad = new Probe(
        "phase_barrier_duplicate_bad", {7, 7, 0}, 4,
        0, 2, 0, 28);
    auto *duplicate_peer = new Probe(
        "phase_barrier_duplicate_peer", {7, 7, 0}, 4,
        1, 2, 0, 29);

    auto *size_first = new Probe("phase_barrier_size_first", {8, 8, 0},
                                 5, 0, 2, 0, 31);
    auto *size_bad = new Probe("phase_barrier_size_bad", {8, 8, 0},
                               5, 1, 3, 0, 32);
    auto *size_peer = new Probe("phase_barrier_size_peer", {8, 8, 0},
                                5, 1, 2, 0, 33);

    auto *tree_first = new Probe("phase_barrier_tree_first", {9, 9, 0},
                                 6, 0, 2, 0, 35);
    auto *tree_bad = new Probe("phase_barrier_tree_bad", {9, 9, 0},
                               6, 1, 2, 322, 36);
    auto *tree_peer = new Probe("phase_barrier_tree_peer", {9, 9, 0},
                                6, 1, 2, 0, 37);

    auto *release0 = new Probe("phase_barrier_release_0", {10, 10, 0},
                               7, 0, 2, 321, 39);
    auto *release1 = new Probe("phase_barrier_release_1", {10, 10, 0},
                               7, 1, 2, 321, 40);
    auto *unknown_tree = new Probe(
        "phase_barrier_unknown_tree", {11, 11, 0}, 8,
        0, 1, std::numeric_limits<uint16_t>::max(), 42);
    auto *invalid_rank = new Probe("phase_barrier_invalid_rank", {12, 12, 0},
                                   9, 2, 2, 0, 44);
    auto *invalid_key = new Probe(
        "phase_barrier_invalid_key",
        {12, std::numeric_limits<uint32_t>::max(), 0},
        9, 0, 1, 0, 45);

    sc_start();

    Suite suite;
    suite.Check(n1->done && !n1->rejected,
                "N=1 phase barrier completes immediately");
    suite.Check(n2_fast->done && n2_slow->done &&
                    !n2_fast->rejected && !n2_slow->rejected,
                "N=2 phase barrier completes both ranks");
    suite.Check(n2_fast->done_at == n2_slow->done_at &&
                    n2_fast->done_at == sc_time(5, SC_NS),
                "fast rank waits for the slow rank");
    suite.Check(n4_0->done && n4_1->done && n4_2->done && n4_3->done &&
                    !n4_0->rejected && !n4_1->rejected &&
                    !n4_2->rejected && !n4_3->rejected,
                "N=4 phase barrier completes every rank");
    suite.Check(phase0->done && phase1->done &&
                    !phase0->rejected && !phase1->rejected,
                "1000 consecutive phases complete and drain");
    suite.Check(epoch0->done && epoch1->done &&
                    !epoch0->rejected && !epoch1->rejected,
                "consecutive epochs remain isolated and complete");
    suite.Check(key_a0->done && key_b1->done && key_a1->done && key_b0->done &&
                    key_a0->done_at == sc_time(24, SC_NS) &&
                    key_b1->done_at == sc_time(25, SC_NS),
                "key mismatch cannot cross-release barrier instances");
    suite.Check(duplicate_first->done && !duplicate_first->rejected &&
                    duplicate_bad->done && duplicate_bad->rejected &&
                    duplicate_peer->done && !duplicate_peer->rejected,
                "duplicate rank rejects without poisoning the barrier");
    suite.Check(size_first->done && !size_first->rejected &&
                    size_bad->done && size_bad->rejected &&
                    size_peer->done && !size_peer->rejected,
                "group-size mismatch rejects without poisoning the barrier");
    suite.Check(tree_first->done && !tree_first->rejected &&
                    tree_bad->done && tree_bad->rejected &&
                    tree_peer->done && !tree_peer->rejected,
                "release-tree mismatch rejects without poisoning the barrier");
    suite.Check(release0->done && release1->done &&
                    !release0->rejected && !release1->rejected &&
                    CollectiveTreeEntryCount() == 0 &&
                    CollectiveReduceNodeCount() == 0,
                "non-zero final release erases its multicast/reduce tree");
    suite.Check(unknown_tree->done && unknown_tree->rejected,
                "unknown non-zero release tree is rejected");
    suite.Check(invalid_rank->done && invalid_rank->rejected &&
                    invalid_key->done && invalid_key->rejected,
                "rank and GROUP_SYNC-reserved key reject before arrival");
    suite.Check(CollectiveBarrierStateCount() == 0,
                "all phase barrier state drains to zero");

    if (suite.failures == 0) {
        std::cout << "[COLLECTIVE PHASE BARRIER V1] PASS ("
                  << suite.checks << " checks)\n";
        return 0;
    }
    std::cerr << "[COLLECTIVE PHASE BARRIER V1] FAIL ("
              << suite.failures << "/" << suite.checks
              << " checks failed)\n";
    return suite.failures;
}

#ifdef COLLECTIVE_PHASE_BARRIER_V1_PRIM_SELFTEST_MAIN
int sc_main(int, char **) {
    return RunCollectivePhaseBarrierV1PrimSelfTest();
}
#endif
