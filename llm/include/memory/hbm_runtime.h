#pragma once

#include "memory/core_mem_adapter.h"
#include "memory/hbm_backend.h"
#include "memory/mem_endpoint_unit.h"

#include <memory>
#include <vector>

struct HBMRuntimeInstance {
    int stack_id = -1;
    int channel_id = -1;
    std::unique_ptr<HBMBackend> backend;
    std::unique_ptr<MemEndpointUnit> endpoint;
};

class HBMRuntime {
public:
    HBMRuntimeInstance *Find(int stack_id, int channel_id);
    const HBMRuntimeInstance *Find(int stack_id, int channel_id) const;
    void BindAdapter(CoreMemAdapter &adapter);
    size_t Size() const { return instances_.size(); }
    const std::vector<HBMRuntimeInstance> &Instances() const {
        return instances_;
    }

private:
    friend std::unique_ptr<HBMRuntime> BuildHBMBackends();
    std::vector<HBMRuntimeInstance> instances_;
};

// 只能在 SystemC elaboration（第一次 sc_start 之前）调用。按显式
// backend_granularity 创建每 channel 或每 stack 唯一实例，并由返回对象统一持有
// backend 与 MemEndpointUnit 生命周期。
std::unique_ptr<HBMRuntime> BuildHBMBackends();
