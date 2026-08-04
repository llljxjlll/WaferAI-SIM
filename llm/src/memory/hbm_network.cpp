#include "memory/hbm_network.h"

#include "defs/const.h"
#include "die/port.h"
#include "router/router.h"
#include "utils/router_utils.h"

#include <stdexcept>

using namespace sc_core;

namespace {
HBMNetwork *g_active_hbm_network = nullptr;
using TxKey = std::pair<int, int>;

int AttachmentRouter(int stack_id, int channel_id) {
    const HBMStackConfig *stack = nullptr;
    for (const auto &s : g_hbm_stacks)
        if (s.stack_id == stack_id) { stack = &s; break; }
    if (!stack)
        throw std::runtime_error("HBM route: unknown stack");
    for (const auto &c : g_hbm_channels)
        if (c.stack_id == stack_id && c.channel_id == channel_id)
            return GlobalId(stack->compute_die_id, c.mem_tile);
    throw std::runtime_error("HBM route: unknown stack/channel attachment");
}
} // namespace

class HBMEndpointAgent : public sc_module {
public:
    SC_HAS_PROCESS(HBMEndpointAgent);
    HBMEndpointAgent(const sc_module_name &name, int attachment_router,
                     MemEndpointUnit &endpoint, HBMNetwork &network,
                     unsigned worker_count, unsigned completed_depth)
        : sc_module(name), attachment_router_(attachment_router),
          endpoint_(endpoint), network_(network),
          completed_depth_(completed_depth) {
        if (!worker_count || !completed_depth_)
            throw std::runtime_error("HBM endpoint bridge depth/workers must be > 0");
        for (unsigned i = 0; i < worker_count; ++i)
            sc_spawn(sc_bind(&HBMEndpointAgent::Worker, this),
                     sc_gen_unique_name("hbm_endpoint_worker"));
    }

    bool TryAccept(const MemWireFlit &wire) {
        const MemFlitRoute route = InspectMemWireFlit(wire);
        TxKey key{route.source_core, route.txid};
        auto &assembly = partial_[key];
        assembly.push_back(wire);
        std::vector<MemWireFlit> canonical = CanonicalizeMemWireFlits(assembly);
        bool complete = false;
        try {
            (void)DeserializeMemMsg(canonical);
            complete = true;
        } catch (const std::runtime_error &) {
            complete = false;
        }
        if (complete && completed_.size() >= completed_depth_) {
            assembly.pop_back();
            return false;
        }
        if (complete) {
            completed_.push_back(std::move(canonical));
            partial_.erase(key);
            ready_.notify(SC_ZERO_TIME);
        }
        return true;
    }

    size_t Residual() const {
        size_t n = completed_.size();
        for (const auto &p : partial_) n += p.second.size();
        return n + active_;
    }

private:
    int attachment_router_;
    MemEndpointUnit &endpoint_;
    HBMNetwork &network_;
    unsigned completed_depth_;
    std::map<TxKey, std::vector<MemWireFlit>> partial_;
    std::deque<std::vector<MemWireFlit>> completed_;
    sc_event ready_;
    unsigned active_ = 0;

    void Worker() {
        while (true) {
            while (completed_.empty()) wait(ready_);
            auto wire = std::move(completed_.front());
            completed_.pop_front();
            ++active_;
            MemMsg response = endpoint_.HandleRequest(DeserializeMemMsg(wire));
            network_.InjectResponse(attachment_router_,
                                    SerializeMemMsg(response));
            --active_;
        }
    }
};

HBMNetwork *ActiveHBMNetwork() { return g_active_hbm_network; }

Directions MemFlitNextHop(const MemWireFlit &wire, int router_id) {
    const MemFlitRoute route = InspectMemWireFlit(wire);
    const int destination = route.request_direction
        ? AttachmentRouter(route.stack_id, route.channel_id)
        : route.source_core;
    if (destination < 0 || destination >= TOTAL_CORES)
        throw std::runtime_error("HBM route: invalid destination");
    if (DieOfGlobal(destination) == DieOfGlobal(router_id))
        return GetNextHop(destination, router_id);
    const int subflow = route.request_direction ? 2 : 3;
    // Use a transaction-stable local anchor. Selecting from router_id would
    // make nearest-port policy change while the flit traverses the die.
    const int anchor = GlobalId(DieOfGlobal(router_id),
                                LocalOfGlobal(route.source_core));
    const int exit = CrossDieSelectExit(anchor, destination,
                                        route.source_core, route.txid,
                                        subflow);
    return CrossDieStep(destination, router_id, exit);
}

HBMNetwork::HBMNetwork(const sc_module_name &name, RouterMonitor &routers,
                       HBMRuntime &runtime, unsigned workers_per_endpoint,
                       unsigned completed_queue_depth)
    : sc_module(name), routers_(routers), runtime_(runtime) {
    if (g_active_hbm_network)
        throw std::runtime_error("only one active HBMNetwork is allowed");
    g_active_hbm_network = this;
    for (const auto &instance : runtime_.Instances()) {
        int rid = AttachmentRouter(instance.stack_id, instance.channel_id);
        auto agent = std::make_unique<HBMEndpointAgent>(
            sc_gen_unique_name("hbm_endpoint_agent"), rid,
            *instance.endpoint, *this, workers_per_endpoint,
            completed_queue_depth);
        endpoint_index_[{instance.stack_id, instance.channel_id}] = agent.get();
        agents_.push_back(std::move(agent));
    }
}

HBMNetwork::~HBMNetwork() {
    if (g_active_hbm_network == this) g_active_hbm_network = nullptr;
}

MemMsg HBMNetwork::Exchange(const MemMsg &request) {
    if (request.message_type != MemMessageType::kRequest)
        throw std::runtime_error("HBMNetwork::Exchange expects a request");
    TxKey key{request.source_core, request.txid};
    if (waiters_.count(key))
        throw std::runtime_error("HBMNetwork: duplicate outstanding source/txid");
    auto waiter = std::make_unique<ResponseWaiter>();
    ResponseWaiter *waiter_ptr = waiter.get();
    waiters_[key] = std::move(waiter);

    const auto wire = SerializeMemMsg(request);
    ++stats_.logical_requests;
    request.command == MemCommand::kRead ? ++stats_.reads : ++stats_.writes;
    stats_.request_bytes += request.length_bytes;
    stats_.request_flits += wire.size();
    try {
        RouterUnit *source = routers_.routers[request.source_core];
        for (const auto &flit : wire)
            source->InjectMemRequestFlit(flit, &stats_.injection_stalls);
        while (!waiter_ptr->complete) wait(waiter_ptr->done);
        MemMsg response = waiter_ptr->response;
        waiters_.erase(key);
        return response;
    } catch (...) {
        response_assemblies_.erase(key);
        waiters_.erase(key);
        throw;
    }
}

bool HBMNetwork::TryAcceptRequestFlit(int router_id,
                                      const MemWireFlit &wire) {
    const MemFlitRoute route = InspectMemWireFlit(wire);
    if (!route.request_direction ||
        AttachmentRouter(route.stack_id, route.channel_id) != router_id)
        throw std::runtime_error("HBMNetwork: request delivered to wrong MEM tile");
    auto it = endpoint_index_.find({route.stack_id, route.channel_id});
    if (it == endpoint_index_.end())
        throw std::runtime_error("HBMNetwork: no endpoint bridge");
    return it->second->TryAccept(wire);
}

bool HBMNetwork::TryAcceptResponseFlit(int router_id,
                                       const MemWireFlit &wire) {
    const MemFlitRoute route = InspectMemWireFlit(wire);
    if (route.request_direction || route.source_core != router_id)
        throw std::runtime_error("HBMNetwork: response delivered to wrong core");
    TxKey key{route.source_core, route.txid};
    auto waiter = waiters_.find(key);
    if (waiter == waiters_.end())
        throw std::runtime_error("HBMNetwork: response has no matching waiter");
    auto &assembly = response_assemblies_[key];
    assembly.push_back(wire);
    if (route.kind == MemFlitKind::kResponse ||
        !assembly.empty()) {
        auto canonical = CanonicalizeMemWireFlits(assembly);
        try {
        waiter->second->response = DeserializeMemMsg(canonical);
        waiter->second->complete = true;
        stats_.response_flits += canonical.size();
        stats_.response_bytes += waiter->second->response.payload.size();
        response_assemblies_.erase(key);
        waiter->second->done.notify(SC_ZERO_TIME);
        } catch (const std::runtime_error &) {
            // Completion or an earlier data flit may arrive first. Retain the
            // partial assembly until every sequence number is present.
        }
    }
    return true;
}

void HBMNetwork::InjectResponse(int attachment_router,
                                const std::vector<MemWireFlit> &wire) {
    RouterUnit *router = routers_.routers[attachment_router];
    for (const auto &flit : wire) {
        router->InjectMemResponseFlit(flit, &stats_.injection_stalls);
    }
}

void HBMNetwork::RecordHop(int router_id, int output_dir,
                           const MemWireFlit &wire) {
    if (output_dir == CENTER) return;
    ++stats_.noc_hops;
    if (IsC2CEgressEdge(router_id, static_cast<Directions>(output_dir))) {
        const auto route = InspectMemWireFlit(wire);
        if (route.request_direction) ++stats_.c2c_request_hops;
        else ++stats_.c2c_response_hops;
        const bool logical_tail =
            (route.request_direction && route.transaction_end) ||
            (!route.request_direction &&
             route.kind == MemFlitKind::kResponse);
        if (logical_tail && g_d2d_cfg.select_policy == SELECT_DYNAMIC) {
            const int subflow = route.request_direction ? 2 : 3;
            ReleaseV5DynamicPort(DieOfGlobal(router_id),
                                 static_cast<Directions>(output_dir),
                                 FlowKey{route.source_core, route.txid,
                                         subflow});
        }
    }
}

size_t HBMNetwork::Residual() const {
    size_t n = waiters_.size();
    for (const auto &p : response_assemblies_) n += p.second.size();
    for (const auto &agent : agents_) n += agent->Residual();
    return n;
}
