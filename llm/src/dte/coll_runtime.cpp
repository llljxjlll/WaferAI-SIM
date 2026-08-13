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
    bool aborted = false;
    sc_event release;
};
using BarrierStatePtr = std::shared_ptr<BarrierState>;
std::map<BarrierKey, BarrierStatePtr> states;
CollectiveBarrierRuntimeCapacity barrier_capacity;

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

void ConfigureCollectiveBarrierRuntime(
    CollectiveBarrierRuntimeCapacity capacity) {
    if (capacity.max_active_states == 0)
        throw std::invalid_argument(
            "collective barrier runtime capacity must be non-zero");
    if (!states.empty())
        throw std::logic_error(
            "collective barrier runtime cannot reconfigure active states");
    barrier_capacity = capacity;
}

CollectiveBarrierRuntimeCapacity
CollectiveBarrierRuntimeConfiguredCapacity() noexcept {
    return barrier_capacity;
}

CollectiveBarrierRuntimeResidual
CollectiveBarrierRuntimeResidualState() noexcept {
    CollectiveBarrierRuntimeResidual residual;
    residual.active_states = states.size();
    for (const auto &[key, state] : states) {
        (void)key;
        residual.arrived_ranks += state->arrived.size();
        residual.departed_ranks += state->departed;
        residual.waiting_ranks +=
            state->arrived.size() - state->departed;
    }
    return residual;
}

std::size_t AbortCollectiveBarrierKey(const CollectiveKey &key) {
    std::vector<BarrierStatePtr> aborted;
    aborted.reserve(states.size());
    for (auto iterator = states.begin(); iterator != states.end();) {
        if (!(std::get<0>(iterator->first) == key)) {
            ++iterator;
            continue;
        }
        iterator->second->aborted = true;
        aborted.push_back(iterator->second);
        iterator = states.erase(iterator);
    }
    for (const BarrierStatePtr &state : aborted)
        state->release.notify(SC_ZERO_TIME);
    return aborted.size();
}

void ResetCollectiveBarrierRuntime() {
    std::vector<BarrierStatePtr> aborted;
    aborted.reserve(states.size());
    for (auto &[key, state] : states) {
        (void)key;
        state->aborted = true;
        aborted.push_back(state);
    }
    states.clear();
    for (const BarrierStatePtr &state : aborted)
        state->release.notify(SC_ZERO_TIME);
}

void WaitCollectiveBarrier(const CollectiveKey &key, uint16_t phase_id,
                           uint16_t rank, uint16_t group_size,
                           uint16_t release_tree_id) {
    if (group_size == 0 || rank >= group_size)
        throw std::invalid_argument("collective barrier rank/group invalid");
    const BarrierKey barrier_key{key, phase_id};
    BarrierStatePtr state;
    const auto existing = states.find(barrier_key);
    if (existing == states.end()) {
        if (states.size() >= barrier_capacity.max_active_states)
            throw std::overflow_error(
                "collective barrier runtime capacity exhausted");
        auto candidate = std::make_shared<BarrierState>();
        candidate->expected = group_size;
        candidate->release_tree_id = release_tree_id;
        state = candidate;
        states.emplace(barrier_key, std::move(candidate));
    } else {
        state = existing->second;
        if (state->expected != group_size)
            throw std::runtime_error(
                "collective barrier group size mismatch");
        if (state->release_tree_id != release_tree_id)
            throw std::runtime_error(
                "collective barrier tree-release mismatch");
    }
    if (!state->arrived.insert(rank).second)
        throw std::runtime_error(
            "duplicate collective barrier arrival");
    const bool last = state->arrived.size() == state->expected;
    if (group_size > 1) {
        if (last) state->release.notify(SC_ZERO_TIME);
        wait(state->release);
        if (state->aborted)
            throw std::runtime_error(
                "collective barrier aborted");
    }

    ++state->departed;
    if (state->departed != state->expected) return;

    const auto current = states.find(barrier_key);
    if (current == states.end() || current->second != state)
        throw std::logic_error(
            "collective barrier lost active state");
    const uint16_t tree_id = state->release_tree_id;
    states.erase(current);
    if (tree_id == 0) return;

    const size_t tree_entries = EraseCollectiveTree(tree_id);
    const size_t reduce_nodes = EraseCollectiveReduceTree(tree_id);
    // reduce_only intentionally has no multicast table entries: its tree
    // exists solely in the streaming reduce registry.
    if (tree_entries == 0 && reduce_nodes == 0)
        throw std::runtime_error(
            "collective final barrier released an unknown tree");
    std::cout << "[COLL_V6_RELEASE] tree=" << tree_id
              << " entries=" << tree_entries
              << " reduce_nodes=" << reduce_nodes << std::endl;
}

size_t CollectiveBarrierStateCount() { return states.size(); }
void ResetCollectiveBarrierStateForTest() {
    ResetCollectiveBarrierRuntime();
}

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
