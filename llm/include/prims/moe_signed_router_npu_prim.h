#pragma once

#include "prims/base.h"

// One complete little-endian five-INT32 route per token is read from real
// SRAM. Both classes own disjoint FP16 outputs, never a nested trace literal.
class moe_score_weighted_forward final : public NpuBase {
public:
    moe_score_weighted_forward();
    void initialize() override;
    void taskCore(TaskCoreContext &, string, u_int64_t &,
                  u_int64_t &, u_int64_t &, u_int64_t &) override;
    vector<sc_bv<128>> serialize() override;
    void deserialize(vector<sc_bv<128>>) override;
};

class moe_score_weight_backward final : public NpuBase {
public:
    moe_score_weight_backward();
    void initialize() override;
    void taskCore(TaskCoreContext &, string, u_int64_t &,
                  u_int64_t &, u_int64_t &, u_int64_t &) override;
    vector<sc_bv<128>> serialize() override;
    void deserialize(vector<sc_bv<128>>) override;
};
