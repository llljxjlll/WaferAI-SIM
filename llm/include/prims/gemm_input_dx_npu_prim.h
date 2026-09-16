#pragma once

#include "prims/base.h"
#include "prims/gemm_input_dx_timing_prim.h"

// Public runtime adapter. Source/StateABI provenance is checked by the
// producer and finalizer; strict Prim wire carries only physical W/dY/dX
// addresses and M/N/K. No numerical FP16 dX is claimed.
class gemm_input_dx_timing final : public NpuBase {
public:
    gemm_input_dx_timing();
    GemmInputDxTimingWork work() const;
    void initialize() override;
    void taskCore(TaskCoreContext &, string, u_int64_t &,
                  u_int64_t &, u_int64_t &, u_int64_t &) override;
    vector<sc_bv<128>> serialize() override;
    void deserialize(vector<sc_bv<128>>) override;
};
