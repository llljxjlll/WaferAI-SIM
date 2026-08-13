#include "dte/sync_runtime_selftest.h"

#include "dte/coll_multicast.h"
#include "dte/coll_runtime.h"
#include "dte/sync_runtime.h"
#include "die/port.h"
#include "defs/global.h"
#include "isa/prim_id.h"
#include "prims/sync_prims.h"
#include "utils/msg_utils.h"
#include "utils/router_utils.h"

#include "systemc.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
class Checks {
public:
    void Check(bool condition, const std::string &message) {
        ++checks_;
        if (condition) return;
        ++failures_;
        std::cerr << "Sync runtime selftest failure: " << message << "\n";
    }

    template <typename Function>
    void Reject(const std::string &message, Function function) {
        ++checks_;
        try {
            function();
        } catch (const std::exception &) {
            return;
        }
        ++failures_;
        std::cerr << "Sync runtime selftest failure: " << message
                  << " did not reject invalid input\n";
    }

    int checks() const noexcept { return checks_; }
    int failures() const noexcept { return failures_; }

private:
    int checks_ = 0;
    int failures_ = 0;
};

struct GroupProbe : sc_module {
    GroupSyncRuntime &runtime;
    uint16_t core;
    uint32_t group;
    uint32_t iterations;
    uint32_t delay_ns;
    bool done = false;
    std::string error;
    sc_time first_done = SC_ZERO_TIME;
    sc_time final_done = SC_ZERO_TIME;

    SC_HAS_PROCESS(GroupProbe);
    GroupProbe(sc_module_name name, GroupSyncRuntime &runtime_, uint16_t core_,
               uint32_t group_, uint32_t iterations_, uint32_t delay_ns_)
        : sc_module(name), runtime(runtime_), core(core_), group(group_),
          iterations(iterations_), delay_ns(delay_ns_) {
        SC_THREAD(Run);
    }

    void Run() {
        try {
            if (delay_ns != 0) wait(delay_ns, SC_NS);
            for (uint32_t seq = 0; seq < iterations; ++seq) {
                runtime.Wait(core, group, seq);
                if (seq == 0) first_done = sc_time_stamp();
            }
            final_done = sc_time_stamp();
            done = true;
        } catch (const std::exception &exception) {
            error = exception.what();
        }
    }
};

struct CollectiveNamespaceProbe : sc_module {
    uint16_t rank;
    bool done = false;
    std::string error;

    SC_HAS_PROCESS(CollectiveNamespaceProbe);
    CollectiveNamespaceProbe(sc_module_name name, uint16_t rank_)
        : sc_module(name), rank(rank_) {
        SC_THREAD(Run);
    }

    void Run() {
        try {
            WaitCollectiveBarrier({4, 7, 0}, 0, rank, 2, 0);
            done = true;
        } catch (const std::exception &exception) {
            error = exception.what();
        }
    }
};

struct EventWaitFirstProbe : sc_module {
    EventControlQueue queue{2};
    EventMailbox mailbox{4};
    sc_event arrival;
    bool done = false;
    uint32_t waits = 0;
    sc_time completed = SC_ZERO_TIME;
    std::string error;

    SC_HAS_PROCESS(EventWaitFirstProbe);
    explicit EventWaitFirstProbe(sc_module_name name) : sc_module(name) {
        SC_THREAD(Waiter);
        SC_THREAD(Sender);
    }

    void Waiter() {
        try {
            const EventKey key{1, 2, 0xf0000001u};
            while (true) {
                while (!queue.Empty()) {
                    mailbox.Deliver(queue.Front(), 2, 8);
                    (void)queue.Pop();
                }
                if (mailbox.TryConsume(key, 2)) break;
                ++waits;
                wait(arrival);
            }
            completed = sc_time_stamp();
            done = true;
        } catch (const std::exception &exception) {
            error = exception.what();
        }
    }

    void Sender() {
        wait(7, SC_NS);
        queue.Push({1, 2, 0xf0000001u});
        arrival.notify(SC_ZERO_TIME);
        wait(4, SC_NS);
        queue.Push({1, 2, 0xf0000001u});
        arrival.notify(SC_ZERO_TIME);
    }
};

void TestRegistryAndPureFailures(Checks &checks,
                                 GroupSyncRuntime &runtime) {
    checks.Reject("duplicate group ID", [] {
        CoreGroupRegistry({{1, {0}}, {1, {1}}}, 8, 8);
    });
    checks.Reject("duplicate group member", [] {
        CoreGroupRegistry({{1, {0, 0}}}, 8, 8);
    });
    checks.Reject("empty group", [] {
        CoreGroupRegistry({{1, {}}}, 8, 8);
    });
    checks.Reject("group ID zero", [] {
        CoreGroupRegistry({{0, {0}}}, 8, 8);
    });
    checks.Reject("out-of-range member", [] {
        CoreGroupRegistry({{1, {8}}}, 8, 8);
    });
    checks.Reject("cross-die group", [] {
        CoreGroupRegistry({{1, {0, 8}}}, 16, 8);
    });
    checks.Reject("invalid group topology", [] {
        CoreGroupRegistry({{1, {0}}}, 10, 8);
    });
    checks.Reject("nonmember GROUP_SYNC arrival", [&] {
        runtime.Wait(1, 2, 0);
    });
    checks.Reject("unknown GROUP_SYNC group", [&] {
        runtime.Wait(0, 99, 0);
    });
    checks.Reject("GROUP_SYNC sequence jump", [&] {
        runtime.Wait(0, 2, 1);
    });
    checks.Check(runtime.NextSequence(2, 0) == 0,
                 "rejected GROUP_SYNC does not advance sequence");
}

void TestEventState(Checks &checks) {
    checks.Check(MSG_TYPE::CONFIG == 0 && MSG_TYPE::DATA == 1 &&
                     MSG_TYPE::REQUEST == 2 && MSG_TYPE::ACK == 3 &&
                     MSG_TYPE::DONE == 4 && MSG_TYPE::S_DATA == 5 &&
                     MSG_TYPE::P_DATA == 6 && MSG_TYPE::EVENT == 7,
                 "EVENT append preserves all existing MSG_TYPE values");

    const EventControlMessage event{1, 2, 0xfedcba98u};
    Msg message = MakeEventControlMsg(event);
    checks.Check(message.IsControlMsg(), "EVENT uses the control channel");
    const sc_bv<256> wire = SerializeMsg(message);
    const Msg decoded = DeserializeMsg(wire);
    checks.Check(ParseEventControlMsg(decoded) == event,
                 "EVENT strict wire preserves a 32-bit tag");
    checks.Check(decoded.tag_id_ == 0 && !decoded.data_.or_reduce(),
                 "EVENT tag does not alias legacy tag or generic payload");

    constexpr int kTypeLow = M_D_IS_END;
    constexpr int kDataLow = M_D_IS_END + M_D_MSG_TYPE + M_D_SEQ_ID +
        M_D_DES + M_D_OFFSET + M_D_TAG_ID + M_D_SOURCE + M_D_LENGTH +
        M_D_REFILL + M_D_ROOFLINE + M_D_CONF_END;
    checks.Reject("unknown message type", [&] {
        sc_bv<256> bad = wire;
        bad.range(kTypeLow + M_D_MSG_TYPE - 1, kTypeLow) = 15;
        (void)DeserializeMsg(bad);
    });
    checks.Reject("EVENT data reserved bits", [&] {
        sc_bv<256> bad = wire;
        bad[kDataLow + 32] = sc_dt::SC_LOGIC_1;
        (void)DeserializeMsg(bad);
    });
    checks.Reject("EVENT noncanonical length", [&] {
        Msg bad = message;
        bad.length_ = 1;
        (void)SerializeMsg(bad);
    });
    checks.Reject("EVENT generic payload", [&] {
        Msg bad = message;
        bad.data_[0] = sc_dt::SC_LOGIC_1;
        (void)SerializeMsg(bad);
    });

    EventControlQueue queue(2);
    queue.Push({1, 2, 7});
    queue.Push({1, 2, 8});
    checks.Check(queue.Full() && queue.Residual() == 2,
                 "EVENT control queue reaches bounded backpressure");
    checks.Reject("EVENT control queue overflow", [&] {
        queue.Push({1, 2, 9});
    });
    checks.Check(queue.Pop().tag == 7 && queue.Pop().tag == 8 &&
                     queue.Empty(),
                 "EVENT control queue is FIFO and drains");
    checks.Reject("EVENT control queue underflow", [&] { (void)queue.Pop(); });

    EventMailbox mailbox(5);
    const EventKey high_tag{1, 2, 0x10001u};
    const EventKey low_tag{1, 2, 1u};
    mailbox.Deliver({1, 2, high_tag.tag}, 2, 8);
    mailbox.Deliver({1, 2, high_tag.tag}, 2, 8);
    mailbox.Deliver({1, 2, high_tag.tag}, 2, 8);
    checks.Check(mailbox.Credit(high_tag) == 3 &&
                     mailbox.Credit(low_tag) == 0,
                 "EVENT mailbox isolates the full 32-bit tag namespace");
    checks.Check(mailbox.TryConsume(high_tag, 2) &&
                     mailbox.Credit(high_tag) == 1,
                 "early EVENT_SET credits support atomic count consumption");
    checks.Check(!mailbox.TryConsume(high_tag, 2) &&
                     mailbox.Credit(high_tag) == 1,
                 "wait-first/count mismatch does not partially consume");
    mailbox.Deliver({1, 2, high_tag.tag}, 2, 8);
    checks.Check(mailbox.TryConsume(high_tag, 2) && mailbox.Empty(),
                 "later EVENT_SET satisfies a waiting multi-credit count");
    checks.Reject("zero EVENT_WAIT count", [&] {
        (void)mailbox.TryConsume(high_tag, 0);
    });
    checks.Reject("unknown EVENT source", [&] {
        mailbox.Deliver({8, 2, 1}, 2, 8);
    });
    checks.Reject("unknown EVENT destination", [&] {
        mailbox.Deliver({1, 8, 1}, 2, 8);
    });
    checks.Reject("EVENT wrong endpoint delivery", [&] {
        mailbox.Deliver({1, 2, 1}, 3, 8);
    });

    EventMailbox bounded(1);
    bounded.Deliver({1, 2, 3}, 2, 8);
    checks.Reject("EVENT mailbox overflow", [&] {
        bounded.Deliver({1, 2, 4}, 2, 8);
    });
    checks.Check(bounded.Residual() == 1,
                 "EVENT mailbox failure preserves residual accounting");
    EventControlQueue transactional(1);
    transactional.Push({1, 2, 5});
    checks.Reject("EVENT queued delivery overflow", [&] {
        bounded.Deliver(transactional.Front(), 2, 8);
    });
    checks.Check(transactional.Residual() == 1 &&
                     transactional.Front().tag == 5,
                 "failed mailbox delivery leaves the queued EVENT intact");

    Group_sync_prim group_prim;
    Event_control_prim event_prim;
    group_prim.group_id = 1;
    group_prim.sync_seq = 0;
    event_prim.op = EventControlOp::SET;
    event_prim.source_core = 1;
    event_prim.destination_core = 2;
    event_prim.tag = 3;
    event_prim.count = 1;
    checks.Check((group_prim.prim_type & SYNC_PRIM) != 0 &&
                     (event_prim.prim_type & SYNC_PRIM) != 0,
                 "GROUP_SYNC and EVENT thin Prims classify as synchronization");
    checks.Check(group_prim.serialize().size() == 1 &&
                     group_prim.serialize()[0].range(7, 0).to_uint() ==
                         PrimIdValue(PrimId::GROUP_SYNC),
                 "GROUP_SYNC thin Prim uses assigned strict ID 54");
    checks.Check(event_prim.serialize().size() == 1 &&
                     event_prim.serialize()[0].range(7, 0).to_uint() ==
                         PrimIdValue(PrimId::EVENT_CONTROL),
                 "EVENT thin Prim uses assigned strict ID 55");
}

void TestEventControlRouting(Checks &checks) {
    GRID_X = 4;
    GRID_Y = 4;
    GRID_SIZE = 16;
    DIE_X = 2;
    DIE_Y = 1;
    DIE_COUNT = 2;
    CORES_PER_DIE = GRID_SIZE;
    TOTAL_CORES = CORES_PER_DIE * DIE_COUNT;
    HOST_ENDPOINT_ID = TOTAL_CORES;
    g_die_ports = D2DPortTable{};
    g_d2d_links.clear();

    nlohmann::json hardware;
    hardware["die_ports"]["edges"]["S"] = {{"role", "host"}};
    hardware["die_ports"]["overrides"] = nlohmann::json::array();
    hardware["die_ports"]["overrides"].push_back(
        {{"side", "E"}, {"idx", 0}, {"role", "c2c"}, {"dir", "E"}});
    hardware["die_ports"]["overrides"].push_back(
        {{"side", "W"}, {"idx", 0}, {"role", "c2c"}, {"dir", "W"}});
    ParseDiePorts(hardware);

    auto walks_to_destination = [](const Msg &message) {
        int position = message.source_;
        std::set<int> visited;
        for (int step = 0; step < TOTAL_CORES + 8; ++step) {
            if (position == message.des_) return true;
            if (!visited.insert(position).second) return false;
            const Directions direction =
                ControlMsgNextHop(message, position);
            const int neighbor = OpenMeshNeighbor(position, direction);
            if (neighbor >= 0) {
                position = neighbor;
                continue;
            }
            const D2DLink *link = nullptr;
            for (const auto &candidate : g_d2d_links) {
                if (candidate.local_die == DieOfGlobal(position) &&
                    candidate.local_port == message.exit_port_) {
                    link = &candidate;
                    break;
                }
            }
            if (link == nullptr) return false;
            const D2DPort &local = g_die_ports.ports[link->local_port];
            if (local.tile != LocalOfGlobal(position) ||
                local.side != direction || local.dir != direction)
                return false;
            const D2DPort &remote = g_die_ports.ports[link->remote_port];
            position = GlobalId(link->remote_die, remote.tile);
        }
        return false;
    };

    Msg local = MakeEventControlMsg({0, 1, 0x80000001u});
    PinControlMsgExit(local);
    checks.Check(local.exit_port_ == -1 && walks_to_destination(local),
                 "same-die EVENT follows the control route without a pin");

    Msg cross = MakeEventControlMsg(
        {0, static_cast<uint16_t>(CORES_PER_DIE), 0xfedcba98u});
    PinControlMsgExit(cross);
    const Msg decoded = DeserializeMsg(SerializeMsg(cross));
    checks.Check(cross.exit_port_ >= 0 &&
                     decoded.exit_port_ == cross.exit_port_ &&
                     decoded.event_tag_ == 0xfedcba98u &&
                     walks_to_destination(decoded),
                 "cross-die EVENT preserves its pin/tag and reaches the core");
    checks.Reject("EVENT on data routing", [&] {
        (void)DataMsgNextHop(decoded, decoded.source_);
    });
    checks.Reject("cross-die EVENT without source pin", [&] {
        Msg missing = MakeEventControlMsg(
            {0, static_cast<uint16_t>(CORES_PER_DIE), 7});
        (void)ControlMsgNextHop(missing, missing.source_);
    });
}
} // namespace

int RunSyncRuntimeSelfTest() {
    Checks checks;
    ResetCollectiveBarrierStateForTest();
    ResetCollectiveFabric();

    auto registry = std::make_shared<const CoreGroupRegistry>(
        std::vector<CoreGroupDefinition>{
            {1, {1}}, {2, {0, 2}}, {3, {1, 3}}, {4, {0, 2, 4, 6}}},
        16, 8);
    GroupSyncRuntime runtime(registry);
    checks.Check(registry->GroupCount() == 4 &&
                     registry->Members(4) ==
                         std::vector<uint16_t>({0, 2, 4, 6}) &&
                     registry->RankOf(4, 4) == 2,
                 "immutable registry preserves non-contiguous member order");
    TestRegistryAndPureFailures(checks, runtime);
    TestEventState(checks);
    TestEventControlRouting(checks);

    GroupProbe n1("group_sync_n1", runtime, 1, 1, 10, 1);
    GroupProbe n2_fast("group_sync_n2_fast", runtime, 0, 2, 1000, 1);
    GroupProbe n2_slow("group_sync_n2_slow", runtime, 2, 2, 1000, 5);
    GroupProbe n4_0("group_sync_n4_0", runtime, 0, 4, 1000, 0);
    GroupProbe n4_1("group_sync_n4_1", runtime, 2, 4, 1000, 1);
    GroupProbe n4_2("group_sync_n4_2", runtime, 4, 4, 1000, 2);
    GroupProbe n4_3("group_sync_n4_3", runtime, 6, 4, 1000, 3);
    GroupProbe duplicate_first("group_sync_duplicate_first", runtime, 1, 3,
                               1, 2);
    GroupProbe duplicate_second("group_sync_duplicate_second", runtime, 1, 3,
                                1, 3);
    GroupProbe duplicate_peer("group_sync_duplicate_peer", runtime, 3, 3,
                              1, 4);
    CollectiveNamespaceProbe collective_0("group_sync_namespace_coll_0", 0);
    CollectiveNamespaceProbe collective_1("group_sync_namespace_coll_1", 1);
    EventWaitFirstProbe event_wait_first("event_wait_first");

    const size_t tree_entries_before = CollectiveTreeEntryCount();
    sc_start();

    checks.Check(n1.done && n1.error.empty(), "GROUP_SYNC N=1 completes");
    checks.Check(n2_fast.done && n2_slow.done && n2_fast.error.empty() &&
                     n2_slow.error.empty(),
                 "GROUP_SYNC N=2 completes 1000 consecutive sequences");
    checks.Check(n4_0.done && n4_1.done && n4_2.done && n4_3.done &&
                     n4_0.error.empty() && n4_1.error.empty() &&
                     n4_2.error.empty() && n4_3.error.empty(),
                 "GROUP_SYNC N=4 completes 1000 consecutive sequences");
    checks.Check(n2_fast.first_done == n2_slow.first_done &&
                     n2_fast.first_done == sc_time(5, SC_NS),
                 "fast GROUP_SYNC member waits for the slow member");
    checks.Check(duplicate_first.done != duplicate_second.done &&
                     (!duplicate_first.error.empty() ||
                      !duplicate_second.error.empty()) &&
                     duplicate_peer.done,
                 "duplicate GROUP_SYNC arrival fails deterministically");
    checks.Check(collective_0.done && collective_1.done &&
                     collective_0.error.empty() && collective_1.error.empty(),
                 "collective and GROUP_SYNC namespaces do not collide");
    checks.Check(event_wait_first.done && event_wait_first.error.empty() &&
                     event_wait_first.waits == 2 &&
                     event_wait_first.completed == sc_time(11, SC_NS) &&
                     event_wait_first.queue.Empty() &&
                     event_wait_first.mailbox.Empty(),
                 "wait-first EVENT wakes on arrival, consumes two credits, and drains");
    checks.Check(CollectiveBarrierStateCount() == 0,
                 "GROUP_SYNC and collective barrier state fully drains");
    checks.Check(CollectiveTreeEntryCount() == tree_entries_before,
                 "GROUP_SYNC does not program or erase collective trees");
    checks.Check(runtime.NextSequence(2, 0) == 1000 &&
                     runtime.NextSequence(2, 2) == 1000 &&
                     runtime.NextSequence(4, 6) == 1000,
                 "GROUP_SYNC tracks independent per-member next sequences");
    checks.Reject("GROUP_SYNC rollback after completion", [&] {
        runtime.Wait(0, 2, 999);
    });

    if (checks.failures() == 0)
        std::cout << "Sync runtime selftest passed (" << checks.checks()
                  << " checks)\n";
    else
        std::cerr << "Sync runtime selftest failed (" << checks.failures()
                  << "/" << checks.checks() << " checks failed)\n";
    return checks.failures();
}
