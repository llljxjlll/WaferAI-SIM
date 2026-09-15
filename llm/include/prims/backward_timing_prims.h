#pragma once

#include "prims/base.h"

#include <cstdint>

// These are timing-only activation-gradient operations. The external program
// remains responsible for allocating and transferring the three SRAM tensors.
struct BackwardTimingWork {
    uint64_t forward_input_bytes = 0;
    uint64_t upstream_bytes = 0;
    uint64_t output_bytes = 0;
    uint64_t exu_ops = 0;
    uint64_t sfu_ops = 0;
    uint64_t vec_ops = 0;
};

class norm_backward_timing final : public NpuBase {
public:
    norm_backward_timing();
    void initialize() override;
    void taskCore(TaskCoreContext &, string, u_int64_t &,
                  u_int64_t &, u_int64_t &, u_int64_t &) override;
    vector<sc_bv<128>> serialize() override;
    void deserialize(vector<sc_bv<128>>) override;
    BackwardTimingWork work() const;
};

class attention_backward_timing final : public NpuBase {
public:
    attention_backward_timing();
    void initialize() override;
    void taskCore(TaskCoreContext &, string, u_int64_t &,
                  u_int64_t &, u_int64_t &, u_int64_t &) override;
    vector<sc_bv<128>> serialize() override;
    void deserialize(vector<sc_bv<128>>) override;
    BackwardTimingWork work() const;
};
