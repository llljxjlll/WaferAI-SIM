#include "prims/dense_rope_residual_backward_physical_work.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void Require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

void Reject(const std::function<void()> &check, const std::string &part) {
    try {
        check();
    } catch (const std::exception &failure) {
        Require(std::string(failure.what()).find(part) != std::string::npos,
                "negative selftest rejected for the wrong reason");
        return;
    }
    throw std::runtime_error("negative selftest unexpectedly accepted");
}
} // namespace

int main() {
    RopeQkBackwardSourceWitness rope;
    rope.forward_op_ref = "T0.layer1.rope";
    rope.upstream_gradient_producer_ref = "backward::T0.layer1.attention";
    rope.logical_tokens = rope.rank_tokens = 3;
    rope.logical_query_heads = 6;
    rope.logical_kv_heads = 3;
    rope.rank_query_heads = 2;
    rope.rank_kv_heads = 1;
    rope.tp_degree = 3;
    rope.head_dim = rope.rotary_dim = 8;
    rope.max_position_embeddings = 16;
    rope.position_trace = {0, 1, 2};
    RopeQkBackwardPhysicalTile rope_tile;
    rope_tile.position_ids = {0, 12, DenseReverseDType::INT32};
    rope_tile.rotated_upstream = {16, 192, DenseReverseDType::FP16};
    rope_tile.packed_output = {224, 192, DenseReverseDType::FP16};
    const auto rope_work = BuildRopeQkBackwardTimingWork(rope, rope_tile);
    Require(rope_work.position_read_bytes == 12 &&
            rope_work.upstream_read_bytes == 192 &&
            rope_work.output_write_bytes == 192 &&
            rope_work.inverse_rotary_pairs == 36 &&
            rope_work.passed_v_elements == 24 &&
            rope_work.sfu_ops == 24 && rope_work.vec_ops == 240 &&
            rope_work.position_trace_digest != 0,
            "real TP3 GQA inverse RoPE work differs");
    auto fake_rope = rope;
    fake_rope.rank_kv_heads = 2;
    Reject([&] { BuildRopeQkBackwardTimingWork(fake_rope, rope_tile); },
           "rank GQA/TP/rotary");
    fake_rope = rope;
    fake_rope.position_trace = {0, 1};
    Reject([&] { BuildRopeQkBackwardTimingWork(fake_rope, rope_tile); },
           "one physical INT32 position");
    fake_rope = rope;
    fake_rope.position_trace = {0, 1, 16};
    Reject([&] { BuildRopeQkBackwardTimingWork(fake_rope, rope_tile); },
           "exceeds source table");
    auto fake_rope_tile = rope_tile;
    fake_rope_tile.position_ids.dtype = DenseReverseDType::FP16;
    Reject([&] { BuildRopeQkBackwardTimingWork(rope, fake_rope_tile); },
           "position_ids typed SRAM");
    fake_rope_tile = rope_tile;
    fake_rope_tile.packed_output.byte_address = 32;
    Reject([&] { BuildRopeQkBackwardTimingWork(rope, fake_rope_tile); },
           "independent INT32/FP16 SRAM");
    fake_rope_tile = rope_tile;
    fake_rope_tile.packed_output.byte_address = 65520;
    Reject([&] { BuildRopeQkBackwardTimingWork(rope, fake_rope_tile); },
           "packed_output typed SRAM");

    ResidualDualDxSourceWitness residual;
    residual.forward_op_ref = "T0.layer1.residual2";
    residual.left_forward_value_ref = "T0.layer1.residual1_out";
    residual.right_forward_value_ref = "T0.layer1.rs2_out";
    residual.upstream_gradient_producer_ref = "backward::T0.final_norm";
    residual.logical_rows = 3;
    residual.rank_rows = 1;
    residual.tp_degree = 3;
    residual.hidden_size = 12;
    ResidualDualDxPhysicalTile residual_tile;
    residual_tile.upstream = {0, 24, DenseReverseDType::FP16};
    residual_tile.left_output = {32, 24, DenseReverseDType::FP16};
    residual_tile.right_output = {64, 24, DenseReverseDType::FP16};
    const auto residual_work = BuildResidualDualDxTimingWork(residual,
                                                              residual_tile);
    Require(residual_work.upstream_read_bytes == 24 &&
            residual_work.left_write_bytes == 24 &&
            residual_work.right_write_bytes == 24 &&
            residual_work.vec_ops == 24,
            "real TP3 residual dual-gradient work differs");
    auto fake_residual = residual;
    fake_residual.right_forward_value_ref = residual.left_forward_value_ref;
    Reject([&] { BuildResidualDualDxTimingWork(fake_residual, residual_tile); },
           "distinct two forward inputs");
    fake_residual = residual;
    fake_residual.logical_rows = 2;
    Reject([&] { BuildResidualDualDxTimingWork(fake_residual, residual_tile); },
           "rank rows differ");
    auto fake_residual_tile = residual_tile;
    fake_residual_tile.right_output.byte_address = 48;
    Reject([&] { BuildResidualDualDxTimingWork(residual, fake_residual_tile); },
           "two independent OWNED output");
    fake_residual_tile = residual_tile;
    fake_residual_tile.right_output.bytes = 12;
    Reject([&] { BuildResidualDualDxTimingWork(residual, fake_residual_tile); },
           "right_output typed SRAM");
    std::cout << "Dense TP3 source-backed RoPE inverse/Residual dual-dX physical work selftest PASS\n";
}
