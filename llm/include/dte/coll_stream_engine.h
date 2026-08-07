#pragma once

#include "dte/coll_compute_pool.h"
#include "dte/coll_innetwork_reduce.h"
#include "dte/coll_reduce_stream.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <vector>

namespace coll_refactor {

struct RouterReduceStreamStats {
    uint64_t headers_in = 0;
    uint64_t data_in = 0;
    uint64_t headers_out = 0;
    uint64_t data_out = 0;
    uint64_t assembler_backpressure = 0;
    uint64_t issue_backpressure = 0;
    uint64_t egress_backpressure = 0;
    uint64_t bypass_beats = 0;
};

struct RouterReduceEgress {
    Directions output = CENTER;
    sc_bv<256> wire;
};

std::vector<uint64_t> UnpackReduceVectorValues(
    const ReduceVectorBeat &beat);
ReduceVectorBeat PackReduceVectorValues(
    const ReduceBeatKey &key, const VectorBeat &geometry,
    const std::vector<uint64_t> &values);

// Per-router stream-v2 reduce engine. RouterUnit owns one only when the
// selected backend is dca_offload; legacy and ordinary Msg traffic never
// instantiate or enter this state machine.
class RouterReduceStreamEngine {
  public:
    RouterReduceStreamEngine(uint16_t router_id,
                             const NocCollDcaConfig &config);

    bool TryAcceptHeader(Directions input,
                         const ReduceStreamWireHeader &header);
    ReduceStreamAcceptStatus TryAcceptData(
        Directions input, const ReduceStreamDataFlit &data);

    // Called once for every active simulated cycle. Gaps are legal only when
    // the engine was fully idle between the two calls.
    void Tick(uint64_t cycle);

    const RouterReduceEgress *FrontEgress() const;
    void PopEgress();

    size_t Residual() const;
    bool Drained() const { return Residual() == 0; }
    const RouterReduceStreamStats &Stats() const { return stats_; }
    const DcaComputePoolStats &DcaStats() const { return pool_.Stats(); }

    // Production endpoint client of the same per-tile pool used by stream
    // stages. Requests are asynchronous and retain their pool tag until the
    // endpoint consumes the result.
    std::optional<uint64_t> TrySubmitCore(DcaPoolRequest request);
    std::optional<DcaPoolResult> TakeCoreResult(uint64_t tag);
    sc_event &ActivityEvent() { return activity_; }
    sc_event &CoreProgressEvent() { return core_progress_; }

  private:
    struct NodeKey {
        uint16_t tree_id = 0;
        ReduceStreamKey stream;

        bool operator==(const NodeKey &other) const;
        bool operator<(const NodeKey &other) const;
    };
    struct NodeState {
        CollReduceTreeNode topology;
        BinaryReduceSchedule schedule;
        ReduceStreamWireHeader output_header;
        std::vector<Directions> inputs;
        std::map<Directions, ReduceStreamRouteKey> routes;
        std::map<uint64_t,
                 std::vector<std::optional<ReduceVectorBeat>>> input_beats;
        std::map<uint64_t, ReduceVectorBeat> final_ready;
        size_t closed_inputs = 0;
        size_t outstanding_beats = 0;
        uint64_t next_output_beat = 0;
        bool output_header_queued = false;
    };
    struct RouteState {
        NodeKey node;
        size_t input_index = 0;
        uint64_t next_beat_to_pop = 0;
        bool input_complete = false;
    };
    struct StageTask {
        NodeKey node;
        uint64_t beat_id = 0;
        uint16_t stage_id = 0;
        size_t next_input = 0;
        ReduceVectorBeat lhs;
        ReduceVectorBeat rhs;
        std::vector<ReduceVectorBeat> inputs;
    };

    uint16_t router_id_ = 0;
    NocCollDcaConfig config_;
    ReduceStreamFiniteState streams_;
    DcaComputePool pool_;
    size_t egress_capacity_ = 0;
    std::map<NodeKey, NodeState> nodes_;
    std::map<ReduceStreamRouteKey, RouteState> routes_;
    std::deque<StageTask> pending_issues_;
    std::map<uint64_t, StageTask> inflight_tasks_;
    std::map<uint64_t, bool> inflight_core_;
    std::map<uint64_t, DcaPoolResult> completed_core_;
    std::deque<RouterReduceEgress> egress_;
    RouterReduceStreamStats stats_;
    bool ticked_ = false;
    uint64_t last_cycle_ = 0;
    sc_event activity_;
    sc_event core_progress_;

    static NodeKey KeyFor(const ReduceStreamWireHeader &header);
    size_t ReservedOperandSlots() const;
    bool CanBufferBeat(const NodeState &node, uint64_t beat_id) const;
    void DrainAssemblers();
    void ScheduleReadyBeats();
    void SubmitPending();
    void ConsumePoolResults();
    void DrainFinalOutputs();
    void RetireCompletedNodes();
    DcaPoolRequest MakePoolRequest(const StageTask &task) const;
    ReduceVectorBeat MakePoolResult(const DcaPoolResult &result,
                                    const StageTask &task) const;
    bool NodeHasPendingTask(const NodeKey &key) const;
};

// Router construction owns this registry; it exposes no alternate compute
// resource, only the production engine attached to that tile.
void RegisterProductionReduceStreamEngine(
    uint16_t router_id, RouterReduceStreamEngine *engine);
void UnregisterProductionReduceStreamEngine(
    uint16_t router_id, RouterReduceStreamEngine *engine);
RouterReduceStreamEngine *LookupProductionReduceStreamEngine(
    uint16_t router_id);

} // namespace coll_refactor
