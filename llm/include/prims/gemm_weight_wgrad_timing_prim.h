#pragma once

#include "prims/weight_gradient_timing_prims.h"

#include <cstdint>

struct GemmWeightWGradWork {
    WeightGradBufferABI activation; // FP16 K x M
    WeightGradBufferABI upstream;   // FP16 K x N
    WeightGradBufferABI gradient;   // FP32 M x N
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t k = 0;
    uint64_t fp16_activation_read_bytes = 0;
    uint64_t fp16_upstream_read_bytes = 0;
    uint64_t fp32_gradient_read_modify_write_bytes = 0;
    uint64_t fma_ops = 0;
    uint64_t exu_flops = 0;
    uint64_t fp32_accumulator_vec_ops = 0;
};

// One rank-local, strictly typed physical tile.  Full-shard gradients wider
// than 16-bit SRAM are a producer-layer tiling obligation.  This Prim models
// timing and SRAM ABI only; no numerical FP32 gradient is claimed.
class gemm_weight_wgrad_timing final : public NpuBase {
public:
    gemm_weight_wgrad_timing();
    void initialize() override;
    void taskCore(TaskCoreContext &, string, u_int64_t &,
                  u_int64_t &, u_int64_t &, u_int64_t &) override;
    vector<sc_bv<128>> serialize() override;
    void deserialize(vector<sc_bv<128>>) override;
    GemmWeightWGradWork work() const;
};
