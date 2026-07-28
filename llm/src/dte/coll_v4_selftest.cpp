#include "dte/coll_multicast.h"

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
}

int RunCollV4SelfTest() {
    fails = total = 0;
    std::cout << "==== NoC collective V4 contract self-test ====" << std::endl;
    CollDataHeader h;
    h.tree_id = 7; h.packet = {{1, 2, 3}, 4, 5, 6, 7};
    h.seq_id = 9; h.length_bits = 96; h.is_end = true;
    const auto wire = SerializeCollData(h);
    const auto round = DeserializeCollData(wire);
    Check(round.tree_id == h.tree_id && round.packet == h.packet &&
              round.seq_id == 9 && round.length_bits == 96 && round.is_end,
          "COLL_DATA 256-bit wire round trip");
    Check(Throws<std::invalid_argument>([] { CollDataHeader x; x.length_bits = 1; SerializeCollData(x); }),
          "tree_id zero is reserved");
    Check(Throws<std::overflow_error>([&] { auto x = h; x.seq_id = 1u << 24; SerializeCollData(x); }),
          "sequence overflow is rejected");
    Check(Throws<std::invalid_argument>([&] { auto x = wire; x[255] = true; DeserializeCollData(x); }),
          "non-zero reserved wire bits are rejected");

    CollectiveTreeTable table(2);
    table.Program({7, 0, CENTER}, (1u << EAST) | (1u << NORTH));
    table.Program({7, 1, WEST}, 1u << CENTER);
    Check(table.Lookup({7, 0, CENTER}) == ((1u << EAST) | (1u << NORTH)),
          "tree table returns a multi-output fork");
    Check(Throws<std::runtime_error>([&] { table.Program({7, 0, CENTER}, 1u << EAST); }),
          "conflicting tree reprogramming is rejected");
    Check(Throws<std::runtime_error>([&] { table.Program({8, 2, WEST}, 1u << EAST); }),
          "tree table capacity is finite");
    Check(Throws<std::invalid_argument>([] { CollectiveTreeTable x(1); x.Program({1, 0, WEST}, 1u << WEST); }),
          "tree entry cannot immediately reflect to ingress");
    Check(table.EraseTree(7) == 2 && table.Size() == 0,
          "tree lifecycle erase drains all entries");

    AtomicMulticastFork fork;
    bool available[DIRECTIONS] = {true, true, true, true, true};
    const uint8_t outputs = (1u << EAST) | (1u << NORTH) | (1u << CENTER);
    CollBranchLockKey a{7, {1, 2, 3}, 4, 5};
    CollBranchLockKey b{7, {1, 2, 4}, 4, 5};
    Check(fork.CanCommit(outputs, available, a),
          "atomic fork admits only when every branch is ready");
    available[NORTH] = false;
    Check(!fork.CanCommit(outputs, available, a) && fork.Residual() == 0,
          "one blocked branch prevents every atomic copy");
    available[NORTH] = true; fork.Commit(outputs, true, false, a);
    Check(fork.Residual() == 3 && !fork.CanCommit(outputs, available, b),
          "branch locks isolate full collective instance key");
    fork.Commit(outputs, false, true, a);
    Check(fork.Residual() == 0 && fork.CanCommit(outputs, available, b),
          "tail releases every branch refcount");

    std::cout << "NoC collective V4 contract self-test: "
              << (fails == 0 ? "PASS" : "FAILURES=" + std::to_string(fails))
              << " (" << total << " checks)" << std::endl;
    return fails;
}
