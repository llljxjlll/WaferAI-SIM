#include "memory/hbm_runtime.h"

#include "die/port.h"
#include "memory/behavioral_hbm_backend.h"
#include "memory/dramsys_hbm_backend.h"
#include "memory/hbm_memspec.h"

#include <set>
#include <stdexcept>

using namespace sc_core;

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
