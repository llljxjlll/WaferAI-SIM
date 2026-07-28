#include "dte/coll_reorder.h"
#include "prims/norm_prims.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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
PacketKey Key(uint16_t src) { return {{1, 2, 3}, src, 0, src, 3}; }
std::vector<GatherExpectedSlot> Slots() {
    return {{Key(0), 0}, {Key(1), 256}, {Key(2), 512}};
}
CollDescriptor Descriptor() {
    CollDescriptor d;
    d.op = CollOp::ALLGATHER; d.key = {1, 2, 3};
    d.group = {0, 2, 4, 6}; d.root_rank = 0; d.self_rank = 3;
    d.count = 4; d.chunk_bits = 256; d.stride_bits = 256;
    d.gather_reorder_depth = 2;
    return d;
}
}

int RunCollV2SelfTest() {
    fails = total = 0;
    std::cout << "==== NoC collective V2 self-test ====" << std::endl;

    GatherReorderBuffer reorder(Slots(), 2, 1);
    Check(reorder.Accept(Key(1), 256).status == GatherAcceptStatus::ACCEPTED &&
              reorder.Occupancy() == 1 && reorder.CommitCycle() == 0,
          "out-of-order packet occupies a slot but cannot commit");
    Check(reorder.Accept(Key(1), 256).status == GatherAcceptStatus::DUPLICATE,
          "duplicate resident packet is rejected");
    Check(reorder.Accept(Key(2), 512).status == GatherAcceptStatus::FULL,
          "head reservation asserts backpressure before physical capacity is exhausted");
    reorder.RecordBackpressureStall();
    Check(reorder.StallCycles() == 1,
          "backpressure stall accounting is explicit");
    Check(reorder.Accept(Key(0), 0).status == GatherAcceptStatus::ACCEPTED &&
              reorder.PeakOccupancy() == 2 && reorder.CommitCycle() == 1,
          "missing head can enter reserved slot and commit without deadlock");
    Check(reorder.Accept(Key(2), 512).status == GatherAcceptStatus::ACCEPTED &&
              reorder.CommitCycle() == 1 && reorder.CommitCycle() == 1 &&
              reorder.Complete() && reorder.Occupancy() == 0,
          "single commit port drains ordered slots one per cycle");
    Check(reorder.ReceivedBitmap() == "000" &&
              reorder.CommittedBitmap() == "111" &&
              reorder.MissingBitmap() == "000",
          "received and committed bitmaps report final drain");
    Check(reorder.Accept(Key(0), 0).status == GatherAcceptStatus::DUPLICATE,
          "duplicate committed packet is rejected");

    GatherReorderBuffer invalid(Slots(), 2);
    Check(invalid.Accept(Key(0), 8).status == GatherAcceptStatus::INVALID_OFFSET,
          "illegal destination offset is rejected");
    Check(invalid.Accept({{9, 9, 9}, 0, 0, 0, 3}, 0).status ==
              GatherAcceptStatus::UNKNOWN_PACKET,
          "packet outside expected bitmap is rejected");
    Check(Throws<std::invalid_argument>([] { GatherReorderBuffer x(Slots(), 0); }),
          "zero-depth finite reorder is rejected");
    Check(Throws<std::invalid_argument>([] {
              auto slots = Slots(); slots.push_back(slots.front());
              GatherReorderBuffer x(slots, 2);
          }), "duplicate expected packet keys are rejected");

    GatherReorderBuffer one(Slots(), 1);
    Check(one.Accept(Key(1), 256).status == GatherAcceptStatus::FULL &&
              one.Accept(Key(0), 0).status == GatherAcceptStatus::ACCEPTED,
          "depth-one buffer backpressures out-of-order traffic but admits head");

    Collective_prim marker;
    marker.descriptor = Descriptor(); marker.phase_id = 1;
    marker.marker_kind = Collective_prim::MarkerKind::GATHER_ARRIVAL;
    Collective_prim decoded; decoded.deserialize(marker.serialize());
    Check(decoded.marker_kind == Collective_prim::MarkerKind::GATHER_ARRIVAL &&
              decoded.phase_id == 1 && decoded.descriptor.gather_reorder_depth == 2,
          "Gather arrival marker wire round trip preserves finite depth");

    std::cout << "NoC collective V2 self-test: "
              << (fails == 0 ? "PASS" : "FAILURES=" + std::to_string(fails))
              << " (" << total << " checks)" << std::endl;
    return fails;
}
