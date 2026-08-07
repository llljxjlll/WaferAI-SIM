#include "dte/coll_stream_engine.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace coll_refactor {
namespace {

std::map<uint16_t, RouterReduceStreamEngine *> production_engines;

uint64_t WidthMask(uint64_t bits) {
    return bits == 64 ? std::numeric_limits<uint64_t>::max()
                      : ((uint64_t{1} << bits) - 1);
}

} // namespace

void RegisterProductionReduceStreamEngine(
    uint16_t router_id, RouterReduceStreamEngine *engine) {
    if (!engine || !production_engines.emplace(router_id, engine).second)
        throw std::runtime_error(
            "duplicate production reduce stream engine registration");
}

void UnregisterProductionReduceStreamEngine(
    uint16_t router_id, RouterReduceStreamEngine *engine) {
    auto found = production_engines.find(router_id);
    if (found == production_engines.end() || found->second != engine)
        throw std::runtime_error(
            "production reduce stream engine registration mismatch");
    production_engines.erase(found);
}

RouterReduceStreamEngine *LookupProductionReduceStreamEngine(
    uint16_t router_id) {
    auto found = production_engines.find(router_id);
    return found == production_engines.end() ? nullptr : found->second;
}

std::vector<uint64_t> UnpackReduceVectorValues(
    const ReduceVectorBeat &beat) {
    beat.geometry.Validate();
    const uint64_t width = CollDTypeBits(beat.geometry.dtype);
    const uint64_t lanes = beat.geometry.lane_mask.lane_count;
    if (beat.slices.size() * 128 != beat.geometry.vector_bits)
        throw std::invalid_argument(
            "reduce beat slices disagree with vector geometry");
    std::vector<uint64_t> values(lanes, 0);
    for (uint64_t lane = 0; lane < lanes; ++lane) {
        const uint64_t bit = lane * width;
        const size_t slice = bit / 128;
        const uint64_t offset = bit % 128;
        if (offset + width > 128)
            throw std::invalid_argument(
                "dtype lane crosses a physical reduce slice");
        values[lane] = beat.slices[slice]
            .range(offset + width - 1, offset).to_uint64();
    }
    return values;
}

ReduceVectorBeat PackReduceVectorValues(
    const ReduceBeatKey &key, const VectorBeat &geometry,
    const std::vector<uint64_t> &values) {
    geometry.Validate();
    if (values.size() != geometry.lane_mask.lane_count)
        throw std::invalid_argument(
            "reduce values disagree with vector lane count");
    const uint64_t width = CollDTypeBits(geometry.dtype);
    ReduceVectorBeat beat;
    beat.key = key;
    beat.geometry = geometry;
    beat.slices.resize(geometry.vector_bits / 128);
    const uint64_t mask = WidthMask(width);
    for (uint64_t lane = 0; lane < values.size(); ++lane) {
        const uint64_t bit = lane * width;
        const size_t slice = bit / 128;
        const uint64_t offset = bit % 128;
        if (offset + width > 128)
            throw std::invalid_argument(
                "dtype lane crosses a physical reduce slice");
        beat.slices[slice].range(offset + width - 1, offset) =
            values[lane] & mask;
    }
    return beat;
}

bool RouterReduceStreamEngine::NodeKey::operator==(
    const NodeKey &other) const {
    return tree_id == other.tree_id && stream == other.stream;
}

bool RouterReduceStreamEngine::NodeKey::operator<(
    const NodeKey &other) const {
    return std::tie(tree_id, stream) <
           std::tie(other.tree_id, other.stream);
}

RouterReduceStreamEngine::RouterReduceStreamEngine(
    uint16_t router_id, const NocCollDcaConfig &config)
    : router_id_(router_id), config_(config),
      streams_({static_cast<size_t>(config.header_fifo_depth),
                static_cast<size_t>(config.result_fifo_depth),
                static_cast<size_t>(std::max<uint64_t>(
                    2, config.operand_fifo_depth)),
                static_cast<size_t>(config.operand_fifo_depth),
                static_cast<size_t>(config.header_fifo_depth),
                static_cast<size_t>(config.result_fifo_depth),
                static_cast<size_t>(config.operand_fifo_depth),
                static_cast<size_t>(config.operand_fifo_depth)},
               128, config.vector_bits),
      pool_(config),
      egress_capacity_(std::max<size_t>(
          5, static_cast<size_t>(config.result_fifo_depth) * 5)) {
    config_.Validate();
    if (config_.vector_bits % 128 != 0)
        throw std::invalid_argument(
            "production reduce stream requires vector_bits divisible by 128");
}

RouterReduceStreamEngine::NodeKey
RouterReduceStreamEngine::KeyFor(
    const ReduceStreamWireHeader &header) {
    return {header.stream.tree_id, header.stream.key};
}

bool RouterReduceStreamEngine::TryAcceptHeader(
    Directions input, const ReduceStreamWireHeader &header) {
    header.Validate(128, config_.vector_bits);
    const CollReduceTreeNode topology = LookupCollectiveReduceNode(
        header.stream.tree_id, router_id_);
    if (input < WEST || input > CENTER ||
        !(topology.expected_inputs & (1u << input)))
        throw std::invalid_argument(
            "reduce stream header arrived from unexpected input");
    const NodeKey key = KeyFor(header);
    auto node_it = nodes_.find(key);
    if (node_it == nodes_.end()) {
        if (nodes_.size() >= config_.header_fifo_depth) return false;
        NodeState node;
        node.topology = topology;
        for (int direction = 0; direction < DIRECTIONS; ++direction)
            if (topology.expected_inputs & (1u << direction))
                node.inputs.push_back(static_cast<Directions>(direction));
        if (node.inputs.empty() ||
            node.inputs.size() > config_.operand_fifo_depth)
            throw std::invalid_argument(
                "reduce fan-in exceeds configured operand capacity");
        node.schedule = BuildBinaryReduceSchedule(
            static_cast<uint16_t>(node.inputs.size()));
        node.output_header = header;
        node.output_header.source_id = router_id_;
        node.output_header.reduce_stage_id =
            static_cast<uint16_t>(node.schedule.stages.size());
        node.output_header.Validate(128, config_.vector_bits);
        node_it = nodes_.emplace(key, std::move(node)).first;
    } else {
        const auto &shape = node_it->second.output_header.stream;
        if (shape.dtype != header.stream.dtype ||
            shape.op != header.stream.op ||
            shape.total_elements != header.stream.total_elements ||
            shape.physical_data_flits !=
                header.stream.physical_data_flits ||
            shape.vector_beats != header.stream.vector_beats ||
            node_it->second.output_header.tail_valid_lanes !=
                header.tail_valid_lanes)
            throw std::invalid_argument(
                "reduce stream input headers have mismatched geometry");
    }
    NodeState &node = node_it->second;
    if (node.routes.count(input))
        throw std::invalid_argument(
            "duplicate reduce stream header for one router input");
    if (routes_.count(header.Route()))
        throw std::invalid_argument(
            "active reduce stream compact-route collision");
    if (!streams_.TryOpenHeader(header)) return false;
    const auto input_it = std::find(node.inputs.begin(), node.inputs.end(),
                                    input);
    if (input_it == node.inputs.end())
        throw std::logic_error("reduce topology input index disappeared");
    const size_t input_index = input_it - node.inputs.begin();
    node.routes.emplace(input, header.Route());
    routes_.emplace(header.Route(),
                    RouteState{key, input_index, 0, false});
    ++stats_.headers_in;
    DrainFinalOutputs();
    return true;
}

ReduceStreamAcceptStatus RouterReduceStreamEngine::TryAcceptData(
    Directions input, const ReduceStreamDataFlit &data) {
    auto route = routes_.find(data.route);
    if (route == routes_.end())
        throw std::invalid_argument(
            "reduce stream data has no active header route");
    const NodeState &node = nodes_.at(route->second.node);
    if (node.inputs.at(route->second.input_index) != input)
        throw std::invalid_argument(
            "reduce stream data changed router input direction");
    const auto status = streams_.AcceptData(data);
    if (status == ReduceStreamAcceptStatus::BACKPRESSURE) {
        ++stats_.assembler_backpressure;
        return status;
    }
    ++stats_.data_in;
    if (status == ReduceStreamAcceptStatus::STREAM_COMPLETE)
        route->second.input_complete = true;
    DrainAssemblers();
    ScheduleReadyBeats();
    DrainFinalOutputs();
    RetireCompletedNodes();
    return status;
}

size_t RouterReduceStreamEngine::ReservedOperandSlots() const {
    size_t slots = 0;
    for (const auto &entry : nodes_)
        slots += entry.second.input_beats.size() *
                 entry.second.inputs.size();
    return slots;
}

bool RouterReduceStreamEngine::CanBufferBeat(
    const NodeState &node, uint64_t beat_id) const {
    if (node.input_beats.count(beat_id)) return true;
    return ReservedOperandSlots() + node.inputs.size() <=
           config_.operand_fifo_depth;
}

void RouterReduceStreamEngine::DrainAssemblers() {
    for (auto route_it = routes_.begin(); route_it != routes_.end();) {
        RouteState &route = route_it->second;
        NodeState &node = nodes_.at(route.node);
        while (true) {
            const uint64_t candidate = route.next_beat_to_pop;
            if (candidate == node.output_header.stream.vector_beats ||
                !CanBufferBeat(node, candidate))
                break;
            auto beat = streams_.PopAssembled(route_it->first);
            if (!beat) break;
            const uint64_t beat_id = beat->key.vector_beat_id;
            if (beat_id != candidate)
                throw std::logic_error(
                    "reduce assembler beat order changed unexpectedly");
            auto &inputs = node.input_beats[beat_id];
            if (inputs.empty()) inputs.resize(node.inputs.size());
            if (inputs[route.input_index])
                throw std::logic_error("duplicate assembled reduce beat");
            inputs[route.input_index] = std::move(*beat);
            ++route.next_beat_to_pop;
        }
        if (route.input_complete) {
            // Closing succeeds only after every ready beat has moved into the
            // finite operand reservation above.
            if (streams_.TryCloseHeader(route_it->first)) {
                ++node.closed_inputs;
                route_it = routes_.erase(route_it);
                continue;
            }
        }
        ++route_it;
    }
}

void RouterReduceStreamEngine::ScheduleReadyBeats() {
    for (auto &entry : nodes_) {
        NodeState &node = entry.second;
        for (auto beats = node.input_beats.begin();
             beats != node.input_beats.end();) {
            const bool ready = std::all_of(
                beats->second.begin(), beats->second.end(),
                [](const std::optional<ReduceVectorBeat> &beat) {
                    return beat.has_value();
                });
            if (!ready) { ++beats; continue; }
            if (node.schedule.bypass) {
                if (node.final_ready.size() >= config_.result_fifo_depth) {
                    ++stats_.egress_backpressure;
                    return;
                }
                ReduceVectorBeat result = std::move(*beats->second[0]);
                result.key = {node.output_header.stream.key,
                              node.output_header.reduce_stage_id,
                              beats->first};
                node.final_ready.emplace(beats->first, std::move(result));
                ++stats_.bypass_beats;
                beats = node.input_beats.erase(beats);
                continue;
            }
            if (pending_issues_.size() >= config_.operand_fifo_depth) {
                ++stats_.issue_backpressure;
                return;
            }
            StageTask task;
            task.node = entry.first;
            task.beat_id = beats->first;
            task.stage_id = 0;
            task.next_input = 2;
            for (auto &value : beats->second)
                task.inputs.push_back(std::move(*value));
            task.lhs = task.inputs[0];
            task.rhs = task.inputs[1];
            pending_issues_.push_back(std::move(task));
            ++node.outstanding_beats;
            beats = node.input_beats.erase(beats);
        }
    }
}

DcaPoolRequest RouterReduceStreamEngine::MakePoolRequest(
    const StageTask &task) const {
    const NodeState &node = nodes_.at(task.node);
    DcaPoolRequest request;
    request.request.tag = 0;
    request.request.key = {node.output_header.stream.key,
                           task.stage_id, task.beat_id};
    request.request.op = node.output_header.stream.op;
    request.request.operands = {task.lhs.geometry, task.rhs.geometry};
    if (config_.value_mode != NocCollValueMode::TIMING_ONLY) {
        request.operand_values[0] = UnpackReduceVectorValues(task.lhs);
        request.operand_values[1] = UnpackReduceVectorValues(task.rhs);
    }
    return request;
}

void RouterReduceStreamEngine::SubmitPending() {
    while (!pending_issues_.empty()) {
        StageTask task = pending_issues_.front();
        auto tag = pool_.TrySubmitAutoTagged(
            DcaRequestSource::DCA, MakePoolRequest(task));
        if (!tag) {
            ++stats_.issue_backpressure;
            break;
        }
        pending_issues_.pop_front();
        inflight_tasks_.emplace(*tag, std::move(task));
    }
}

ReduceVectorBeat RouterReduceStreamEngine::MakePoolResult(
    const DcaPoolResult &result, const StageTask &task) const {
    if (config_.value_mode == NocCollValueMode::TIMING_ONLY) {
        ReduceVectorBeat beat = task.lhs;
        beat.key = result.result.key;
        return beat;
    }
    return PackReduceVectorValues(result.result.key, result.result.value,
                                  result.values);
}

void RouterReduceStreamEngine::ConsumePoolResults() {
    while (true) {
        const auto front_tag = pool_.FrontResultTag();
        if (!front_tag) break;
        auto core = inflight_core_.find(*front_tag);
        if (core != inflight_core_.end()) {
            auto result = pool_.PopResult();
            if (!result || result->source != DcaRequestSource::CORE)
                throw std::logic_error(
                    "shared pool CORE result identity mismatch");
            completed_core_.emplace(*front_tag, std::move(*result));
            inflight_core_.erase(core);
            core_progress_.notify(SC_ZERO_TIME);
            continue;
        }
        auto continuation = inflight_tasks_.find(*front_tag);
        if (continuation == inflight_tasks_.end())
            throw std::logic_error("DCA result lost its stream continuation");
        const bool final = continuation->second.next_input >=
            continuation->second.inputs.size();
        NodeState &node = nodes_.at(continuation->second.node);
        if (final && node.final_ready.size() >= config_.result_fifo_depth)
            break;
        if (!final && pending_issues_.size() >= config_.operand_fifo_depth)
            break;
        auto result = pool_.PopResult();
        if (!result)
            throw std::logic_error("DCA result front disappeared");
        StageTask task = std::move(continuation->second);
        inflight_tasks_.erase(continuation);
        ReduceVectorBeat value = MakePoolResult(*result, task);
        if (!final) {
            task.lhs = std::move(value);
            task.rhs = task.inputs[task.next_input++];
            ++task.stage_id;
            pending_issues_.push_back(std::move(task));
        } else {
            value.key = {node.output_header.stream.key,
                         node.output_header.reduce_stage_id,
                         task.beat_id};
            node.final_ready.emplace(task.beat_id, std::move(value));
            if (node.outstanding_beats == 0)
                throw std::logic_error("reduce stream outstanding underflow");
            --node.outstanding_beats;
        }
    }
}

void RouterReduceStreamEngine::DrainFinalOutputs() {
    for (auto &entry : nodes_) {
        NodeState &node = entry.second;
        if (node.routes.size() != node.inputs.size() &&
            node.closed_inputs + node.routes.size() != node.inputs.size())
            continue;
        if (!node.output_header_queued) {
            if (egress_.size() >= egress_capacity_) {
                ++stats_.egress_backpressure;
                continue;
            }
            egress_.push_back({node.topology.parent_output,
                SerializeReduceStreamHeader(node.output_header,
                                            config_.vector_bits)});
            node.output_header_queued = true;
            ++stats_.headers_out;
        }
        auto ready = node.final_ready.find(node.next_output_beat);
        while (ready != node.final_ready.end()) {
            const auto flits = SplitReduceVectorBeat(
                node.output_header, ready->second, 128,
                config_.vector_bits);
            if (egress_.size() + flits.size() > egress_capacity_) {
                ++stats_.egress_backpressure;
                break;
            }
            for (const auto &flit : flits) {
                egress_.push_back({node.topology.parent_output,
                                   SerializeReduceStreamData(flit)});
                ++stats_.data_out;
            }
            node.final_ready.erase(ready);
            ++node.next_output_beat;
            ready = node.final_ready.find(node.next_output_beat);
        }
    }
}

bool RouterReduceStreamEngine::NodeHasPendingTask(
    const NodeKey &key) const {
    if (std::any_of(pending_issues_.begin(), pending_issues_.end(),
                    [&key](const StageTask &task) {
                        return task.node == key;
                    }))
        return true;
    return std::any_of(inflight_tasks_.begin(), inflight_tasks_.end(),
                       [&key](const auto &entry) {
                           return entry.second.node == key;
                       });
}

void RouterReduceStreamEngine::RetireCompletedNodes() {
    for (auto it = nodes_.begin(); it != nodes_.end();) {
        const NodeState &node = it->second;
        if (node.closed_inputs == node.inputs.size() &&
            node.next_output_beat == node.output_header.stream.vector_beats &&
            node.input_beats.empty() && node.final_ready.empty() &&
            node.outstanding_beats == 0 && !NodeHasPendingTask(it->first))
            it = nodes_.erase(it);
        else
            ++it;
    }
}

void RouterReduceStreamEngine::Tick(uint64_t cycle) {
    if (ticked_ && cycle <= last_cycle_)
        throw std::invalid_argument(
            "reduce stream engine cycle must increase");
    if (ticked_ && cycle != last_cycle_ + 1 && Residual() != 0)
        throw std::invalid_argument(
            "active reduce stream engine skipped a cycle");
    ticked_ = true;
    last_cycle_ = cycle;
    DrainAssemblers();
    ScheduleReadyBeats();
    SubmitPending();
    pool_.Tick(cycle);
    ConsumePoolResults();
    ScheduleReadyBeats();
    DrainFinalOutputs();
    RetireCompletedNodes();
    if (!inflight_core_.empty() || !completed_core_.empty())
        core_progress_.notify(SC_ZERO_TIME);
}

std::optional<uint64_t> RouterReduceStreamEngine::TrySubmitCore(
    DcaPoolRequest request) {
    auto tag = pool_.TrySubmitAutoTagged(DcaRequestSource::CORE,
                                         std::move(request));
    if (!tag) return std::nullopt;
    if (!inflight_core_.emplace(*tag, true).second)
        throw std::logic_error("duplicate shared CORE pool tag");
    activity_.notify(SC_ZERO_TIME);
    return tag;
}

std::optional<DcaPoolResult> RouterReduceStreamEngine::TakeCoreResult(
    uint64_t tag) {
    auto found = completed_core_.find(tag);
    if (found == completed_core_.end()) return std::nullopt;
    DcaPoolResult result = std::move(found->second);
    completed_core_.erase(found);
    activity_.notify(SC_ZERO_TIME);
    return result;
}

const RouterReduceEgress *RouterReduceStreamEngine::FrontEgress() const {
    return egress_.empty() ? nullptr : &egress_.front();
}

void RouterReduceStreamEngine::PopEgress() {
    if (egress_.empty())
        throw std::logic_error("pop from empty reduce stream egress");
    egress_.pop_front();
    DrainFinalOutputs();
    RetireCompletedNodes();
}

size_t RouterReduceStreamEngine::Residual() const {
    size_t input_beats = 0;
    size_t final_beats = 0;
    for (const auto &entry : nodes_) {
        input_beats += entry.second.input_beats.size();
        final_beats += entry.second.final_ready.size();
    }
    return streams_.Residual() + nodes_.size() + routes_.size() +
           input_beats + final_beats + pending_issues_.size() +
           inflight_tasks_.size() + inflight_core_.size() +
           completed_core_.size() + pool_.Residual() + egress_.size();
}

} // namespace coll_refactor
