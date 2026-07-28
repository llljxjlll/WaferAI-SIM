#include "dte/coll_runtime.h"
#include "dte/coll_reorder.h"
#include "dte/coll_multicast.h"
#include "dte/coll_innetwork_reduce.h"
#include "defs/const.h"
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <tuple>

namespace {
using BarrierKey = std::tuple<CollectiveKey, uint16_t>;
struct BarrierState {
    uint16_t expected = 0;
    std::set<uint16_t> arrived;
    uint16_t departed = 0;
    uint16_t release_tree_id = 0;
    sc_event release;
};
std::map<BarrierKey, std::unique_ptr<BarrierState>> states;

using GatherStateKey = std::tuple<CollectiveKey, uint16_t>;
std::map<GatherStateKey, std::unique_ptr<GatherReorderBuffer>> gather_states;

using ReduceStateKey = std::tuple<CollectiveKey, uint16_t>;
struct ReduceRxState {
    std::vector<bool> expected;
    std::vector<bool> received;
    size_t arrived = 0;
};
std::map<ReduceStateKey, ReduceRxState> reduce_states;

std::string Bitmap(const std::vector<bool> &bits) {
    std::string out;
    out.reserve(bits.size());
    for (bool bit : bits) out.push_back(bit ? '1' : '0');
    return out;
}

bool UsesGatherRx(CollOp op) {
    return op == CollOp::GATHER || op == CollOp::ALLGATHER ||
           op == CollOp::ALLTOALL;
}

std::vector<GatherExpectedSlot> ExpectedGatherSlots(const CollDescriptor &d) {
    std::vector<GatherExpectedSlot> expected;
    for (uint16_t src = 0; src < d.group.size(); ++src) {
        if (src == d.self_rank) continue; // local contribution needs no NoC RX.
        PacketKey key{d.key, src, 0, src, d.self_rank};
        expected.push_back({key, uint64_t(src) * d.stride_bits});
    }
    return expected;
}
}

void WaitCollectiveBarrier(const CollectiveKey &key, uint16_t phase_id,
                           uint16_t rank, uint16_t group_size,
                           uint16_t release_tree_id) {
    if (group_size == 0 || rank >= group_size)
        throw std::invalid_argument("collective barrier rank/group invalid");
    BarrierKey barrier_key{key, phase_id};
    auto &slot = states[barrier_key];
    if (!slot) {
        slot = std::make_unique<BarrierState>();
        slot->expected = group_size;
        slot->release_tree_id = release_tree_id;
    } else if (slot->expected != group_size) {
        throw std::runtime_error("collective barrier group size mismatch");
    } else if (slot->release_tree_id != release_tree_id) {
        throw std::runtime_error("collective barrier tree-release mismatch");
    }
    BarrierState *state = slot.get();
    if (!state->arrived.insert(rank).second)
        throw std::runtime_error("duplicate collective barrier arrival");
    const bool last = state->arrived.size() == state->expected;
    if (last) state->release.notify(SC_ZERO_TIME);
    else wait(state->release);
    ++state->departed;
    if (state->departed == state->expected) {
        const uint16_t tree_id = state->release_tree_id;
        states.erase(barrier_key);
        if (tree_id != 0) {
            const size_t tree_entries = EraseCollectiveTree(tree_id);
            const size_t reduce_nodes =
                EraseCollectiveReduceTree(tree_id);
            if (tree_entries == 0)
                throw std::runtime_error(
                    "collective final barrier released an unknown tree");
            std::cout << "[COLL_V6_RELEASE] tree=" << tree_id
                      << " entries=" << tree_entries
                      << " reduce_nodes=" << reduce_nodes << std::endl;
        }
    }
}

size_t CollectiveBarrierStateCount() { return states.size(); }
void ResetCollectiveBarrierStateForTest() { states.clear(); }

void ProcessGatherReorderArrival(const CollDescriptor &d, uint16_t phase_id) {
    ValidateCollDescriptor(d);
    if (!UsesGatherRx(d.op) || d.gather_reorder_depth == 0)
        throw std::invalid_argument("finite gather reorder marker is invalid");
    if (phase_id >= d.group.size() || phase_id == d.self_rank)
        throw std::invalid_argument("gather reorder source phase is invalid");

    const GatherStateKey state_key{d.key, d.self_rank};
    auto &state = gather_states[state_key];
    if (!state)
        state = std::make_unique<GatherReorderBuffer>(
            ExpectedGatherSlots(d), d.gather_reorder_depth, 1);

    const PacketKey packet{d.key, phase_id, 0, phase_id, d.self_rank};
    const uint64_t offset = uint64_t(phase_id) * d.stride_bits;
    while (true) {
        const GatherAcceptResult result = state->Accept(packet, offset);
        if (result.status == GatherAcceptStatus::ACCEPTED) break;
        if (result.status != GatherAcceptStatus::FULL)
            throw std::runtime_error("gather reorder rejected packet metadata");
        state->RecordBackpressureStall();
        std::cout << "[COLL_REORDER] event=stall occupancy="
                  << state->Occupancy() << " received="
                  << state->ReceivedBitmap() << " committed="
                  << state->CommittedBitmap() << " missing="
                  << state->MissingBitmap() << std::endl;
        wait(CYCLE, SC_NS);
        state->CommitCycle();
    }

    std::cout << "[COLL_REORDER] event=accept src_rank=" << phase_id
              << " occupancy=" << state->Occupancy() << " received="
              << state->ReceivedBitmap() << " committed="
              << state->CommittedBitmap() << " missing="
              << state->MissingBitmap() << std::endl;
    wait(CYCLE, SC_NS); // one-cycle Gather RX commit port.
    const size_t committed = state->CommitCycle();
    std::cout << "[COLL_REORDER] event=commit count=" << committed
              << " occupancy=" << state->Occupancy() << " received="
              << state->ReceivedBitmap() << " committed="
              << state->CommittedBitmap() << " missing="
              << state->MissingBitmap() << std::endl;
    if (state->Complete()) {
        std::cout << "[COLL_REORDER] event=drain peak="
                  << state->PeakOccupancy() << " stalls="
                  << state->StallCycles() << " committed="
                  << state->CommittedBitmap() << std::endl;
        gather_states.erase(state_key);
    }
}

size_t CollectiveGatherReorderStateCount() { return gather_states.size(); }
void ResetCollectiveGatherReorderStateForTest() { gather_states.clear(); }

void ProcessReduceRxArrival(const CollDescriptor &d, uint16_t phase_id) {
    ValidateTier0ReductionDescriptor(d);
    if (d.self_rank != d.root_rank || phase_id >= d.group.size() ||
        phase_id == d.root_rank)
        throw std::invalid_argument("Reduce RX arrival rank/phase is invalid");
    const ReduceStateKey key{d.key, d.self_rank};
    auto inserted = reduce_states.emplace(key, ReduceRxState{});
    ReduceRxState &state = inserted.first->second;
    if (inserted.second) {
        state.expected.assign(d.group.size(), true);
        state.expected[d.root_rank] = false;
        state.received.assign(d.group.size(), false);
    }
    if (!state.expected[phase_id])
        throw std::runtime_error("Reduce RX received an unexpected source");
    if (state.received[phase_id])
        throw std::runtime_error("Reduce RX received a duplicate source");
    state.received[phase_id] = true;
    ++state.arrived;
    std::cout << "[COLL_REDUCE_RX] event=accept src_rank=" << phase_id
              << " expected=" << Bitmap(state.expected)
              << " received=" << Bitmap(state.received) << std::endl;

    if (state.arrived + 1 == d.group.size()) {
        wait(CYCLE, SC_NS); // endpoint alignment + completion service.
        std::cout << "[COLL_REDUCE_RX] event=complete aligned=1 received="
                  << Bitmap(state.received) << std::endl;
        reduce_states.erase(key);
    }
}

size_t CollectiveReduceRxStateCount() { return reduce_states.size(); }
void ResetCollectiveReduceRxStateForTest() { reduce_states.clear(); }
