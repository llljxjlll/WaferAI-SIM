#pragma once

#include "memory/hbm_mem_wire.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <systemc>
#include <vector>

struct HBMBackendStats {
    uint64_t requests = 0;
    uint64_t reads = 0;
    uint64_t writes = 0;
    uint64_t bytes = 0;
    uint64_t completed = 0;
    uint64_t failed = 0;
    sc_core::sc_time service_time = sc_core::SC_ZERO_TIME;
};

struct HBMBackendTransaction {
    MemCommand command = MemCommand::kRead;
    uint64_t address = 0;
    std::vector<uint8_t> payload;
    std::vector<uint8_t> byte_enable;
    sc_core::sc_time submitted = sc_core::SC_ZERO_TIME;

    // backend 可同步调用并给出未来 delay，也可在真实异步响应到达时以 delay=0 调用。
    std::function<void(sc_core::sc_time delay,
                       sc_core::sc_time service_time, int status,
                       const std::string &error)> complete;
};

class HBMBackend {
public:
    virtual ~HBMBackend() = default;
    virtual void Submit(const std::shared_ptr<HBMBackendTransaction> &tx) = 0;
    virtual const HBMBackendStats &Stats() const = 0;
};
