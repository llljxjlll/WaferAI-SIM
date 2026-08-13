#include "memory/hbm_runtime.h"

#include "die/port.h"
#include "defs/spec.h"
#include "memory/behavioral_hbm_backend.h"
#include "memory/dramsys_hbm_backend.h"
#include "memory/hbm_memspec.h"
#include "memory/hbm_address_map.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <set>
#include <stdexcept>

using namespace sc_core;

namespace {

struct HBMRuntimeDebugRoute {
    int stack_id = -1;
    int channel_id = -1;
    uint64_t physical_offset = 0;
    uint64_t local_address = 0;
    uint64_t size_bytes = 0;
};

void RequireDebugBoundary(const char *operation) {
    if (sc_is_running())
        throw std::logic_error(
            std::string("HBMRuntime ") + operation +
            " is forbidden while simulation is running");
}

std::vector<HBMRuntimeDebugRoute> BuildDebugRoutes(
    const HBMRuntime &runtime, uint64_t physical_address,
    int current_die, uint64_t size_bytes) {
    if (current_die < 0 || current_die >= DIE_COUNT)
        throw std::invalid_argument(
            "HBMRuntime debug current_die is out of range");
    if (size_bytes == 0 ||
        size_bytes > std::numeric_limits<size_t>::max())
        throw std::invalid_argument(
            "HBMRuntime debug size is invalid");
    if (physical_address >
        std::numeric_limits<uint64_t>::max() - size_bytes)
        throw std::out_of_range(
            "HBMRuntime debug physical range overflows");

    std::vector<HBMRuntimeDebugRoute> routes;
    routes.reserve(4);
    for (uint64_t offset = 0; offset < size_bytes; ++offset) {
        const AddressDecodeResult decoded =
            DecodeAddress(physical_address + offset, current_die);
        const HBMRuntimeInstance *instance =
            runtime.Find(decoded.stack_id, decoded.channel_id);
        if (instance == nullptr || !instance->backend)
            throw std::runtime_error(
                "HBMRuntime debug address has no bound backend");
        if (!routes.empty()) {
            HBMRuntimeDebugRoute &last = routes.back();
            if (last.stack_id == decoded.stack_id &&
                last.channel_id == decoded.channel_id &&
                last.local_address + last.size_bytes ==
                    decoded.local_address) {
                ++last.size_bytes;
                continue;
            }
        }
        routes.push_back({decoded.stack_id, decoded.channel_id,
                          offset, decoded.local_address, 1});
    }
    return routes;
}

} // namespace

HBMRuntimeInstance *HBMRuntime::Find(int stack_id, int channel_id) {
    for (auto &instance : instances_)
        if (instance.stack_id == stack_id &&
            instance.channel_id == channel_id)
            return &instance;
    return nullptr;
}

const HBMRuntimeInstance *HBMRuntime::Find(int stack_id, int channel_id) const {
    for (const auto &instance : instances_)
        if (instance.stack_id == stack_id &&
            instance.channel_id == channel_id)
            return &instance;
    return nullptr;
}

void HBMRuntime::BindAdapter(CoreMemAdapter &adapter) {
    for (auto &instance : instances_)
        adapter.BindEndpoint(instance.stack_id, instance.channel_id,
                             instance.endpoint.get());
}


HBMRuntimeDebugSnapshot HBMRuntime::DebugPeek(
    uint64_t physical_address, int current_die,
    uint64_t size_bytes) const {
    RequireDebugBoundary("DebugPeek");
    const auto routes = BuildDebugRoutes(
        *this, physical_address, current_die, size_bytes);
    HBMRuntimeDebugSnapshot result;
    result.physical_address = physical_address;
    result.current_die = current_die;
    result.payload.resize(static_cast<size_t>(size_bytes));
    result.chunks.reserve(routes.size());

    for (const auto &route : routes) {
        const HBMRuntimeInstance *instance =
            Find(route.stack_id, route.channel_id);
        HBMDebugSnapshot backend = instance->backend->DebugPeek(
            route.local_address, route.size_bytes);
        if (backend.address != route.local_address ||
            backend.payload.size() != route.size_bytes ||
            backend.present.size() != route.size_bytes)
            throw std::logic_error(
                "HBM backend returned a malformed debug snapshot");
        std::copy(
            backend.payload.begin(), backend.payload.end(),
            result.payload.begin() +
                static_cast<size_t>(route.physical_offset));
        result.chunks.push_back(
            {route.stack_id, route.channel_id,
             route.physical_offset, std::move(backend)});
    }
    return result;
}

void HBMRuntime::DebugRestore(
    const HBMRuntimeDebugSnapshot &snapshot) {
    RequireDebugBoundary("DebugRestore");
    if (snapshot.payload.empty())
        throw std::invalid_argument(
            "HBMRuntime debug snapshot is empty");
    if (snapshot.physical_address >
        std::numeric_limits<uint64_t>::max() -
            snapshot.payload.size())
        throw std::out_of_range(
            "HBMRuntime debug snapshot physical range overflows");

    uint64_t cursor = 0;
    for (const auto &chunk : snapshot.chunks) {
        if (chunk.physical_offset != cursor ||
            chunk.backend.payload.empty() ||
            chunk.backend.payload.size() !=
                chunk.backend.present.size() ||
            chunk.backend.payload.size() > snapshot.payload.size() ||
            cursor > snapshot.payload.size() -
                         chunk.backend.payload.size())
            throw std::invalid_argument(
                "HBMRuntime debug snapshot chunks are malformed");
        const HBMRuntimeInstance *instance =
            Find(chunk.stack_id, chunk.channel_id);
        if (instance == nullptr || !instance->backend)
            throw std::runtime_error(
                "HBMRuntime debug snapshot backend is absent");
        if (!std::equal(
                chunk.backend.payload.begin(),
                chunk.backend.payload.end(),
                snapshot.payload.begin() +
                    static_cast<size_t>(cursor)))
            throw std::invalid_argument(
                "HBMRuntime debug snapshot payload disagrees with chunks");
        cursor += chunk.backend.payload.size();
    }
    if (cursor != snapshot.payload.size())
        throw std::invalid_argument(
            "HBMRuntime debug snapshot does not cover its payload");

    for (auto it = snapshot.chunks.rbegin();
         it != snapshot.chunks.rend(); ++it) {
        HBMRuntimeInstance *instance =
            Find(it->stack_id, it->channel_id);
        instance->backend->DebugRestore(it->backend);
    }
}

void HBMRuntime::DebugSeed(
    uint64_t physical_address, int current_die,
    const std::vector<uint8_t> &payload) {
    RequireDebugBoundary("DebugSeed");
    if (payload.empty())
        throw std::invalid_argument(
            "HBMRuntime DebugSeed payload must not be empty");
    const auto routes = BuildDebugRoutes(
        *this, physical_address, current_die, payload.size());
    const HBMRuntimeDebugSnapshot before =
        DebugPeek(physical_address, current_die, payload.size());

    std::vector<std::vector<uint8_t>> pieces;
    pieces.reserve(routes.size());
    for (const auto &route : routes) {
        const auto begin =
            payload.begin() + static_cast<size_t>(route.physical_offset);
        pieces.emplace_back(
            begin, begin + static_cast<size_t>(route.size_bytes));
    }

    try {
        for (size_t index = 0; index < routes.size(); ++index) {
            const auto &route = routes[index];
            HBMRuntimeInstance *instance =
                Find(route.stack_id, route.channel_id);
            instance->backend->DebugSeed(
                route.local_address, pieces[index]);
        }
    } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        try {
            DebugRestore(before);
        } catch (const std::exception &rollback) {
            throw std::runtime_error(
                std::string("HBMRuntime DebugSeed rollback failed: ") +
                rollback.what());
        }
        std::rethrow_exception(failure);
    }
}

std::unique_ptr<HBMRuntime> BuildHBMBackends() {
    if (!g_memory_system_active ||
        g_memory_topology != MemoryTopology::kDistributedHbm)
        throw std::runtime_error(
            "BuildHBMBackends requires an active distributed_hbm config");
    if (sc_is_running())
        throw std::runtime_error(
            "BuildHBMBackends must run before the first sc_start");

    auto runtime = std::make_unique<HBMRuntime>();
    std::set<std::pair<int, int>> seen;
    for (const auto &channel : g_hbm_channels) {
        if (!seen.insert({channel.stack_id, channel.channel_id}).second)
            throw std::runtime_error(
                "BuildHBMBackends: duplicate stack/channel instance");
        const HBMStackConfig *stack = nullptr;
        for (const auto &candidate : g_hbm_stacks)
            if (candidate.stack_id == channel.stack_id) {
                stack = &candidate;
                break;
            }
        if (!stack)
            throw std::runtime_error(
                "BuildHBMBackends: attachment references unknown stack");

        HBMRuntimeInstance instance;
        instance.stack_id = channel.stack_id;
        instance.channel_id = channel.channel_id;
        if (stack->backend_kind == HBMBackendKind::kDRAMSys) {
            instance.backend = std::make_unique<DRAMSysHBMBackend>(
                sc_gen_unique_name("hbm_dramsys"),
                stack->channel_dram_config);
        } else {
            ResolvedHbmMemSpec spec =
                ResolveHbmMemSpec(stack->channel_dram_config);
            const HBMProfile &profile = g_hbm_profiles.at(stack->profile);
            BehavioralHBMBackendConfig cfg;
            cfg.bandwidth_GBps = spec.instance_bandwidth_GBps;
            if (stack->bandwidth_cap_GBps >= 0.0)
                cfg.bandwidth_GBps =
                    stack->backend_granularity ==
                            HBMBackendGranularity::kChannel
                        ? stack->bandwidth_cap_GBps /
                              profile.channels_per_stack
                        : stack->bandwidth_cap_GBps;
            cfg.efficiency = stack->behavioral_efficiency;
            cfg.base_latency =
                sc_time(stack->behavioral_base_latency_ns, SC_NS);
            cfg.read_to_write_turnaround =
                sc_time(stack->behavioral_read_to_write_ns, SC_NS);
            cfg.write_to_read_turnaround =
                sc_time(stack->behavioral_write_to_read_ns, SC_NS);
            instance.backend =
                std::make_unique<BehavioralHBMBackend>(cfg);
        }
        instance.endpoint = std::make_unique<MemEndpointUnit>(
            sc_gen_unique_name("hbm_endpoint"), instance.stack_id,
            instance.channel_id, *instance.backend);
        runtime->instances_.push_back(std::move(instance));
    }
    if (runtime->instances_.empty())
        throw std::runtime_error("BuildHBMBackends: no backend instance built");
    return runtime;
}
