#pragma once

#include "prims/base.h"
#include "prims/gemm_input_dx_timing_prim.h"

// NEW-only runtime adapter. It deliberately has no PrimId or public ISA
// registration until source-to-physical RecordOpcode/codec/finalizer coverage
// exists. Worker timing may be exercised directly in an isolated harness.
class gemm_input_dx_timing final : public NpuBase {
public:
    GemmInputDxSourceWitness source_witness;

    gemm_input_dx_timing();
    GemmInputDxTimingWork work() const;
    void initialize() override;
    void taskCore(TaskCoreContext &, string, u_int64_t &,
                  u_int64_t &, u_int64_t &, u_int64_t &) override;
    vector<sc_bv<128>> serialize() override;
    void deserialize(vector<sc_bv<128>>) override;
};
