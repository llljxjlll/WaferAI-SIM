#pragma once

#include "prims/base.h"

#include <cstdint>
#include <string>
#include <vector>

// NEW-only physical work/source contract. Public opcode/PrimId registration,
// record wire, linked BufferABI and numerical gradient execution are separate.
enum class DenseReverseDType : uint8_t { INT32, FP16 };

struct DenseReverseSramSpan {
    uint32_t byte_address = 0;
    uint64_t bytes = 0;
    DenseReverseDType dtype = DenseReverseDType::FP16;
};

struct RopeQkBackwardSourceWitness {
    std::string forward_op_ref;
    std::string upstream_gradient_producer_ref;
    uint64_t logical_tokens = 0;
    uint64_t rank_tokens = 0;
    uint64_t logical_query_heads = 0;
    uint64_t logical_kv_heads = 0;
    uint64_t rank_query_heads = 0;
    uint64_t rank_kv_heads = 0;
    uint64_t tp_degree = 0;
    uint64_t head_dim = 0;
    uint64_t rotary_dim = 0;
    uint64_t max_position_embeddings = 0;
    std::vector<uint32_t> position_trace;
};

struct RopeQkBackwardPhysicalTile {
    DenseReverseSramSpan position_ids; // INT32 [rank_tokens]
    DenseReverseSramSpan rotated_upstream; // FP16 [tokens,Q+K+V packed]
    DenseReverseSramSpan packed_output; // independent FP16 dQKV
};

struct RopeQkBackwardTimingWork {
    uint64_t position_read_bytes = 0;
    uint64_t upstream_read_bytes = 0;
    uint64_t output_write_bytes = 0;
    uint64_t inverse_rotary_pairs = 0;
    uint64_t passed_v_elements = 0;
    uint64_t sfu_ops = 0;
    uint64_t vec_ops = 0;
    uint64_t position_trace_digest = 0;
};

RopeQkBackwardTimingWork BuildRopeQkBackwardTimingWork(
    const RopeQkBackwardSourceWitness &source,
    const RopeQkBackwardPhysicalTile &tile);

struct ResidualDualDxSourceWitness {
    std::string forward_op_ref;
    std::string left_forward_value_ref;
    std::string right_forward_value_ref;
    std::string upstream_gradient_producer_ref;
    uint64_t logical_rows = 0;
    uint64_t rank_rows = 0;
    uint64_t tp_degree = 0;
    uint64_t hidden_size = 0;
};

struct ResidualDualDxPhysicalTile {
    DenseReverseSramSpan upstream; // FP16 dY[rank_rows,hidden]
    DenseReverseSramSpan left_output; // FP16 dLeft, independent OWNED SRAM
    DenseReverseSramSpan right_output; // FP16 dRight, independent OWNED SRAM
};

struct ResidualDualDxTimingWork {
    uint64_t upstream_read_bytes = 0;
    uint64_t left_write_bytes = 0;
    uint64_t right_write_bytes = 0;
    uint64_t vec_ops = 0;
};

ResidualDualDxTimingWork BuildResidualDualDxTimingWork(
    const ResidualDualDxSourceWitness &source,
    const ResidualDualDxPhysicalTile &tile);

class rope_backward_timing final : public NpuBase {
public:
    rope_backward_timing();
    void initialize() override;
    void taskCore(TaskCoreContext &, string, u_int64_t &, u_int64_t &,
                  u_int64_t &, u_int64_t &) override;
    vector<sc_bv<128>> serialize() override;
    void deserialize(vector<sc_bv<128>>) override;
    RopeQkBackwardTimingWork work() const;
};

class residual_backward_timing final : public NpuBase {
public:
    residual_backward_timing();
    void initialize() override;
    void taskCore(TaskCoreContext &, string, u_int64_t &, u_int64_t &,
                  u_int64_t &, u_int64_t &) override;
    vector<sc_bv<128>> serialize() override;
    void deserialize(vector<sc_bv<128>>) override;
    ResidualDualDxTimingWork work() const;
};
