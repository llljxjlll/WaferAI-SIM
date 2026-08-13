#include "dte/coll_barrier_runtime_v1_selftest.h"

#include "dte/coll_runtime.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Suite {
    int checks = 0;
    int failures = 0;

    void Check(bool condition, const std::string &name) {
        ++checks;
        if (condition) return;
        ++failures;
        std::cerr << "[COLLECTIVE BARRIER RUNTIME V1] FAIL: "
                  << name << '\n';
    }

    template <class Exception, class Function>
    void Rejects(Function &&function, const std::string &name) {
        bool rejected = false;
        try {
            function();
        } catch (const Exception &) {
            rejected = true;
        } catch (...) {
        }
        Check(rejected, name);
    }
};

struct WaitProbe : sc_module {
    CollectiveKey key;
    uint16_t phase = 0;
    uint16_t rank = 0;
    uint16_t group_size = 1;
    uint16_t tree = 0;
    sc_time delay = SC_ZERO_TIME;
    bool completed = false;
    bool aborted = false;
    bool unexpected = false;
    sc_time completed_at = SC_ZERO_TIME;

    SC_HAS_PROCESS(WaitProbe);
    WaitProbe(sc_module_name name, CollectiveKey barrier_key,
              uint16_t phase_id, uint16_t barrier_rank,
              uint16_t size, sc_time start_delay,
              uint16_t release_tree = 0)
        : sc_module(name), key(barrier_key), phase(phase_id),
          rank(barrier_rank), group_size(size), tree(release_tree),
          delay(start_delay) {
        SC_THREAD(Run);
    }

    void Run() {
        wait(delay);
        try {
            WaitCollectiveBarrier(key, phase, rank, group_size,
                                  tree);
            completed = true;
            completed_at = sc_time_stamp();
        } catch (const std::runtime_error &error) {
            aborted =
                std::string(error.what()).find("aborted") !=
                std::string::npos;
            unexpected = !aborted;
        } catch (...) {
            unexpected = true;
        }
    }
};

struct OverflowProbe : sc_module {
    bool overflow_rejected = false;
    bool configure_rejected = false;
    CollectiveBarrierRuntimeResidual residual;

    SC_HAS_PROCESS(OverflowProbe);
    explicit OverflowProbe(sc_module_name name) : sc_module(name) {
        SC_THREAD(Run);
    }

    void Run() {
        wait(2, SC_NS);
        try {
            WaitCollectiveBarrier({3, 3, 3}, 0, 0, 2);
        } catch (const std::overflow_error &) {
            overflow_rejected = true;
        } catch (...) {
        }
        try {
            ConfigureCollectiveBarrierRuntime({3});
        } catch (const std::logic_error &) {
            configure_rejected = true;
        } catch (...) {
        }
        residual = CollectiveBarrierRuntimeResidualState();
    }
};

struct AbortProbe : sc_module {
    sc_time delay = SC_ZERO_TIME;
    std::vector<CollectiveKey> keys;
    std::size_t aborted = 0;

    SC_HAS_PROCESS(AbortProbe);
    AbortProbe(sc_module_name name, sc_time start_delay,
               std::vector<CollectiveKey> abort_keys)
        : sc_module(name), delay(start_delay),
          keys(std::move(abort_keys)) {
        SC_THREAD(Run);
    }

    void Run() {
        wait(delay);
        for (const CollectiveKey &key : keys)
            aborted += AbortCollectiveBarrierKey(key);
    }
};

struct ResetProbe : sc_module {
    sc_time delay = SC_ZERO_TIME;
    bool reset = false;

    SC_HAS_PROCESS(ResetProbe);
    ResetProbe(sc_module_name name, sc_time start_delay)
        : sc_module(name), delay(start_delay) {
        SC_THREAD(Run);
    }

    void Run() {
        wait(delay);
        ResetCollectiveBarrierRuntime();
        reset = true;
    }
};

struct ThousandPhaseProbe : sc_module {
    bool completed = false;
    bool unexpected = false;

    SC_HAS_PROCESS(ThousandPhaseProbe);
    explicit ThousandPhaseProbe(sc_module_name name) : sc_module(name) {
        SC_THREAD(Run);
    }

    void Run() {
        wait(10, SC_NS);
        try {
            for (uint16_t phase = 0; phase < 1000; ++phase)
                WaitCollectiveBarrier({10, 10, 1}, phase, 0, 1);
            completed = true;
        } catch (...) {
            unexpected = true;
        }
    }
};

struct MismatchRecoveryProbe : sc_module {
    bool group_rejected = false;
    bool tree_rejected = false;
    bool completed = false;
    bool unexpected = false;

    SC_HAS_PROCESS(MismatchRecoveryProbe);
    explicit MismatchRecoveryProbe(sc_module_name name)
        : sc_module(name) {
        SC_THREAD(Run);
    }

    void Run() {
        wait(41, SC_NS);
        try {
            WaitCollectiveBarrier({40, 40, 1}, 4, 1, 3, 0);
        } catch (const std::runtime_error &) {
            group_rejected = true;
        } catch (...) {
        }
        try {
            WaitCollectiveBarrier({40, 40, 1}, 4, 1, 2, 9);
        } catch (const std::runtime_error &) {
            tree_rejected = true;
        } catch (...) {
        }
        wait(1, SC_NS);
        try {
            WaitCollectiveBarrier({40, 40, 1}, 4, 1, 2, 0);
            completed = true;
        } catch (...) {
            unexpected = true;
        }
    }
};

struct DuplicateRecoveryProbe : sc_module {
    bool duplicate_rejected = false;
    bool completed = false;
    bool unexpected = false;

    SC_HAS_PROCESS(DuplicateRecoveryProbe);
    explicit DuplicateRecoveryProbe(sc_module_name name)
        : sc_module(name) {
        SC_THREAD(Run);
    }

    void Run() {
        wait(51, SC_NS);
        try {
            WaitCollectiveBarrier({50, 50, 1}, 5, 0, 2);
        } catch (const std::runtime_error &) {
            duplicate_rejected = true;
        } catch (...) {
        }
        wait(1, SC_NS);
        try {
            WaitCollectiveBarrier({50, 50, 1}, 5, 1, 2);
            completed = true;
        } catch (...) {
            unexpected = true;
        }
    }
};

struct UnknownTreeProbe : sc_module {
    bool rejected = false;

    SC_HAS_PROCESS(UnknownTreeProbe);
    explicit UnknownTreeProbe(sc_module_name name) : sc_module(name) {
        SC_THREAD(Run);
    }

    void Run() {
        wait(80, SC_NS);
        try {
            WaitCollectiveBarrier({80, 80, 1}, 8, 0, 1,
                                  UINT16_MAX);
        } catch (const std::runtime_error &) {
            rejected = true;
        } catch (...) {
        }
    }
};

} // namespace

int RunCollectiveBarrierRuntimeV1SelfTest() {
    Suite suite;
    ResetCollectiveBarrierRuntime();
    suite.Check(
        CollectiveBarrierRuntimeConfiguredCapacity()
                .max_active_states ==
            kDefaultCollectiveBarrierRuntimeCapacity,
        "legacy default capacity is explicit and bounded");
    suite.Rejects<std::invalid_argument>(
        [] { ConfigureCollectiveBarrierRuntime({0}); },
        "zero capacity is rejected");
    ConfigureCollectiveBarrierRuntime({2});

    WaitProbe capacity_a("barrier_capacity_a", {1, 1, 1}, 0, 0,
                         2, sc_time(1, SC_NS));
    WaitProbe capacity_b("barrier_capacity_b", {2, 2, 2}, 0, 0,
                         2, sc_time(1, SC_NS));
    OverflowProbe overflow("barrier_overflow");
    AbortProbe capacity_abort(
        "barrier_capacity_abort", sc_time(3, SC_NS),
        {{1, 1, 1}, {2, 2, 2}});

    ThousandPhaseProbe thousand("barrier_thousand");

    WaitProbe n2_rank0("barrier_n2_0", {20, 20, 1}, 2, 0, 2,
                       sc_time(20, SC_NS));
    WaitProbe n2_rank1("barrier_n2_1", {20, 20, 1}, 2, 1, 2,
                       sc_time(22, SC_NS));

    WaitProbe n4_rank0("barrier_n4_0", {30, 30, 1}, 3, 0, 4,
                       sc_time(30, SC_NS));
    WaitProbe n4_rank1("barrier_n4_1", {30, 30, 1}, 3, 1, 4,
                       sc_time(31, SC_NS));
    WaitProbe n4_rank2("barrier_n4_2", {30, 30, 1}, 3, 2, 4,
                       sc_time(32, SC_NS));
    WaitProbe n4_rank3("barrier_n4_3", {30, 30, 1}, 3, 3, 4,
                       sc_time(33, SC_NS));

    WaitProbe mismatch_waiter(
        "barrier_mismatch_waiter", {40, 40, 1}, 4, 0, 2,
        sc_time(40, SC_NS));
    MismatchRecoveryProbe mismatch("barrier_mismatch_recovery");

    WaitProbe duplicate_waiter(
        "barrier_duplicate_waiter", {50, 50, 1}, 5, 0, 2,
        sc_time(50, SC_NS));
    DuplicateRecoveryProbe duplicate("barrier_duplicate_recovery");

    WaitProbe abort_waiter(
        "barrier_abort_waiter", {60, 60, 1}, 6, 0, 2,
        sc_time(60, SC_NS));
    AbortProbe abort("barrier_abort", sc_time(61, SC_NS),
                     {{60, 60, 1}});

    WaitProbe reset_waiter(
        "barrier_reset_waiter", {70, 70, 1}, 7, 0, 2,
        sc_time(70, SC_NS));
    ResetProbe reset("barrier_reset", sc_time(71, SC_NS));

    UnknownTreeProbe unknown_tree("barrier_unknown_tree");

    sc_start(90, SC_NS);

    suite.Check(overflow.overflow_rejected &&
                    overflow.residual.active_states == 2 &&
                    overflow.residual.arrived_ranks == 2 &&
                    overflow.residual.waiting_ranks == 2,
                "capacity max succeeds and max+1 fails without pollution");
    suite.Check(overflow.configure_rejected &&
                    CollectiveBarrierRuntimeConfiguredCapacity()
                            .max_active_states == 2,
                "active reconfiguration is rejected without mutation");
    suite.Check(capacity_abort.aborted == 2 &&
                    capacity_a.aborted && capacity_b.aborted &&
                    !capacity_a.unexpected && !capacity_b.unexpected,
                "exact-key abort wakes capacity waiters safely");

    suite.Check(thousand.completed && !thousand.unexpected,
                "1000 sequential N=1 phases drain");
    suite.Check(n2_rank0.completed && n2_rank1.completed &&
                    n2_rank0.completed_at == sc_time(22, SC_NS) &&
                    n2_rank1.completed_at == sc_time(22, SC_NS),
                "N=2 releases all ranks at the last arrival");
    suite.Check(n4_rank0.completed && n4_rank1.completed &&
                    n4_rank2.completed && n4_rank3.completed &&
                    n4_rank0.completed_at == sc_time(33, SC_NS) &&
                    n4_rank3.completed_at == sc_time(33, SC_NS),
                "N=4 releases all ranks in one delta");

    suite.Check(mismatch.group_rejected &&
                    mismatch.tree_rejected &&
                    mismatch.completed && mismatch_waiter.completed &&
                    !mismatch.unexpected &&
                    !mismatch_waiter.unexpected,
                "group/tree mismatch does not pollute recoverable state");
    suite.Check(duplicate.duplicate_rejected &&
                    duplicate.completed &&
                    duplicate_waiter.completed &&
                    !duplicate.unexpected &&
                    !duplicate_waiter.unexpected,
                "duplicate arrival is rejected without polluting state");

    suite.Check(abort.aborted == 1 && abort_waiter.aborted &&
                    !abort_waiter.unexpected,
                "abort wakes an active waiter with a deterministic error");
    suite.Check(reset.reset && reset_waiter.aborted &&
                    !reset_waiter.unexpected,
                "reset detaches and wakes every active waiter");
    suite.Check(unknown_tree.rejected,
                "unknown tree release fails after state cleanup");

    const auto residual = CollectiveBarrierRuntimeResidualState();
    suite.Check(residual.active_states == 0 &&
                    residual.arrived_ranks == 0 &&
                    residual.departed_ranks == 0 &&
                    residual.waiting_ranks == 0 &&
                    CollectiveBarrierStateCount() == 0,
                "normal and exceptional barrier paths leave zero residual");

    if (suite.failures == 0) {
        std::cout << "[COLLECTIVE BARRIER RUNTIME V1] PASS ("
                  << suite.checks << " checks)\n";
        return 0;
    }
    std::cerr << "[COLLECTIVE BARRIER RUNTIME V1] FAIL ("
              << suite.failures << "/" << suite.checks
              << " checks failed)\n";
    return suite.failures;
}

#ifdef COLLECTIVE_BARRIER_RUNTIME_V1_SELFTEST_MAIN
int sc_main(int, char **) {
    return RunCollectiveBarrierRuntimeV1SelfTest();
}
#endif
