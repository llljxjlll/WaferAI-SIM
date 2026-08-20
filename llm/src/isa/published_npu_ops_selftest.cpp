#include "isa/published_npu_ops_selftest.h"

#include "isa/published_npu_ops.h"

#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void Check(IsaV1SelfTestResult &result, bool condition,
           const std::string &message) {
    ++result.checks;
    if (!condition) result.failures.push_back(message);
}

template <typename Function>
void Reject(IsaV1SelfTestResult &result, const std::string &message,
            Function &&function) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception &) {
        rejected = true;
    }
    Check(result, rejected, message);
}

PublishedNpuHardwareView Hardware() {
    PublishedNpuHardwareView hardware;
    hardware.exu_x_dims = 4;
    hardware.exu_count = 1;
    hardware.sfu_x_dims = 4;
    hardware.vec_x_dims = 4;
    hardware.vec_count = 1;
    hardware.compute_utilization = 1.0F;
    hardware.cycle_ns = 2;
    return hardware;
}

void CheckOps(IsaV1SelfTestResult &result, const std::string &label,
              Opcode opcode, PublishedNpuParameters parameters,
              NpuOps expected,
              PublishedNpuHardwareView hardware = {}) {
    const NpuOps actual =
        EvaluatePublishedNpuOps(opcode, parameters, hardware);
    Check(result, actual.exu == expected.exu &&
                      actual.sfu == expected.sfu &&
                      actual.vec == expected.vec,
          label + " exact published operation counts");
}

SramAddressOperand ExactAddress(uint64_t value) {
    SramAddressOperand result;
    result.kind = SramAddressKind::ABSOLUTE;
    result.absolute_address_bytes = value;
    return result;
}

void CheckExactWork(IsaV1SelfTestResult &result) {
    RopeQkExactOperands rope;
    rope.input = ExactAddress(0);
    rope.output = ExactAddress(64);
    rope.logical_tokens = 8;
    rope.tp_degree = 2;
    rope.num_heads = 4;
    rope.num_kv_heads = 4;
    rope.rank_num_heads = 2;
    rope.rank_num_kv_heads = 2;
    rope.head_dim = 4;
    rope.rotary_dim = 4;
    rope.max_position_embeddings = 128;
    rope.context_max = 8;
    const double theta = 10000.0;
    std::memcpy(&rope.rope_theta_f64_bits, &theta, sizeof(theta));
    const PublishedNpuWork rope_work = EvaluatePublishedNpuWork(rope);
    Check(result, rope_work.ops.exu == 0 && rope_work.ops.sfu == 0 &&
                      rope_work.ops.vec == 384 &&
                      rope_work.memory_read_bytes == 384 &&
                      rope_work.memory_write_bytes == 384 &&
                      rope_work.comparisons == 0,
          "ROPE exact published work");

    AttentionExactOperands attention;
    attention.input = ExactAddress(0);
    attention.output = ExactAddress(64);
    attention.query_tokens = 8;
    attention.tp_degree = 2;
    attention.num_heads = 4;
    attention.num_kv_heads = 4;
    attention.rank_num_heads = 2;
    attention.rank_num_kv_heads = 2;
    attention.head_dim = 4;
    attention.context_sum = 8;
    attention.context_max = 8;
    attention.query_key_pairs = 36;
    attention.rank_kv_read_bytes = 0;
    attention.rank_kv_write_bytes = 256;
    const PublishedNpuWork attention_work =
        EvaluatePublishedNpuWork(attention);
    Check(result, attention_work.ops.exu == 1152 &&
                      attention_work.ops.sfu == 72 &&
                      attention_work.ops.vec == 144 &&
                      attention_work.memory_read_bytes == 384 &&
                      attention_work.memory_write_bytes == 128 &&
                      attention_work.comparisons == 0,
          "ATTENTION exact published work excludes KV transport bytes");

    EmbeddingLookupOperands embedding;
    embedding.indices = ExactAddress(0);
    embedding.table = ExactAddress(64);
    embedding.output = ExactAddress(128);
    embedding.logical_rows = 8;
    embedding.rank_rows = 4;
    embedding.tp_degree = 2;
    embedding.vocab_size = 32;
    embedding.hidden_size = 16;
    const PublishedNpuWork embedding_work =
        EvaluatePublishedNpuWork(embedding);
    Check(result, embedding_work.ops.exu == 0 &&
                      embedding_work.ops.sfu == 0 &&
                      embedding_work.ops.vec == 0 &&
                      embedding_work.memory_read_bytes == 144 &&
                      embedding_work.memory_write_bytes == 128 &&
                      embedding_work.comparisons == 0,
          "EMBEDDING exact published work");

    GreedySampleOperands greedy;
    greedy.logits = ExactAddress(0);
    greedy.output = ExactAddress(64);
    greedy.tp_degree = 1;
    greedy.token_rows = 8;
    greedy.vocab_size = 32;
    greedy.sample_count = 1;
    greedy.comparisons = 31;
    const PublishedNpuWork greedy_work = EvaluatePublishedNpuWork(greedy);
    Check(result, greedy_work.ops.exu == 0 &&
                      greedy_work.ops.sfu == 0 &&
                      greedy_work.ops.vec == 0 &&
                      greedy_work.memory_read_bytes == 64 &&
                      greedy_work.memory_write_bytes == 4 &&
                      greedy_work.comparisons == 31,
          "GREEDY exact published work");

    CrossEntropyForwardOperands ce;
    ce.logits = ExactAddress(0);
    ce.labels = ExactAddress(64);
    ce.loss = ExactAddress(128);
    ce.logical_rows = 8;
    ce.rank_rows = 4;
    ce.tp_degree = 2;
    ce.vocab_size = 32;
    const PublishedNpuWork ce_work = EvaluatePublishedNpuWork(ce);
    Check(result, ce_work.ops.exu == 0 && ce_work.ops.sfu == 132 &&
                      ce_work.ops.vec == 260 &&
                      ce_work.memory_read_bytes == 272 &&
                      ce_work.memory_write_bytes == 16 &&
                      ce_work.comparisons == 124,
          "CROSS_ENTROPY_FORWARD exact published work");

    CrossEntropyBackwardOperands ce_backward;
    ce_backward.logits = ExactAddress(0);
    ce_backward.labels = ExactAddress(64);
    ce_backward.upstream = ExactAddress(128);
    ce_backward.logits_grad = ExactAddress(192);
    ce_backward.logical_rows = 8;
    ce_backward.rank_rows = 4;
    ce_backward.tp_degree = 2;
    ce_backward.vocab_size = 32;
    ce_backward.upstream_elements = 4;
    const PublishedNpuWork ce_backward_work =
        EvaluatePublishedNpuWork(ce_backward);
    Check(result, ce_backward_work.ops.exu == 0 &&
                      ce_backward_work.ops.sfu == 132 &&
                      ce_backward_work.ops.vec == 516 &&
                      ce_backward_work.memory_read_bytes == 288 &&
                      ce_backward_work.memory_write_bytes == 256 &&
                      ce_backward_work.comparisons == 124,
          "CROSS_ENTROPY_BACKWARD exact published work");

    SgdUpdateOperands sgd;
    sgd.weight = ExactAddress(0);
    sgd.gradient = ExactAddress(64);
    sgd.updated_weight = sgd.weight;
    sgd.element_count = 8;
    const double learning_rate = 0.01;
    std::memcpy(&sgd.learning_rate_f64_bits, &learning_rate,
                sizeof(learning_rate));
    const PublishedNpuWork sgd_work = EvaluatePublishedNpuWork(sgd);
    Check(result, sgd_work.ops.exu == 0 && sgd_work.ops.sfu == 0 &&
                      sgd_work.ops.vec == 16 &&
                      sgd_work.memory_read_bytes == 48 &&
                      sgd_work.memory_write_bytes == 16 &&
                      sgd_work.comparisons == 0,
          "SGD_UPDATE exact published work");

    const PublishedNpuWork legacy = EvaluatePublishedNpuWork(
        Opcode::GELU, {{"N", 7}});
    Check(result, legacy.ops.vec == 28 && legacy.memory_read_bytes == 0 &&
                      legacy.memory_write_bytes == 0 &&
                      legacy.comparisons == 0,
          "legacy published work wrapper preserves ops and zero extensions");
}

} // namespace

IsaV1SelfTestResult CheckPublishedNpuOpsSelfTest() {
    IsaV1SelfTestResult result;
    const PublishedNpuHardwareView hardware = Hardware();

    CheckOps(result, "MATMUL minimal VEC branch", Opcode::MATMUL,
             {{"B", 1}, {"T", 1}, {"C", 1}, {"OC", 1}},
             {0, 0, 2}, hardware);
    CheckOps(result, "MATMUL typical EXU branch", Opcode::MATMUL,
             {{"B", 2}, {"T", 3}, {"C", 4}, {"OC", 5}},
             {896, 0, 0}, hardware);

    CheckOps(result, "CONV minimal", Opcode::CONV,
             {{"B", 1}, {"W", 1}, {"H", 1}, {"C", 1},
              {"pX", 0}, {"pY", 0}, {"sX", 1}, {"sY", 1},
              {"kX", 1}, {"kY", 1}, {"F", 1}},
             {2, 0, 0});
    CheckOps(result, "CONV typical", Opcode::CONV,
             {{"B", 2}, {"W", 5}, {"H", 4}, {"C", 3},
              {"pX", 1}, {"pY", 0}, {"sX", 2}, {"sY", 1},
              {"kX", 3}, {"kY", 2}, {"F", 4}},
             {2592, 0, 0});

    CheckOps(result, "MAXPOOL minimal", Opcode::MAXPOOL,
             {{"B", 1}, {"W", 1}, {"H", 1}, {"C", 1},
              {"pX", 0}, {"pY", 0}, {"sX", 1}, {"sY", 1},
              {"kX", 1}, {"kY", 1}},
             {0, 1, 0});
    CheckOps(result, "MAXPOOL typical", Opcode::MAXPOOL,
             {{"B", 2}, {"W", 5}, {"H", 4}, {"C", 3},
              {"pX", 1}, {"pY", 0}, {"sX", 2}, {"sY", 1},
              {"kX", 3}, {"kY", 2}},
             {0, 324, 0});

    CheckOps(result, "ATTENTION minimal", Opcode::ATTENTION,
             {{"B", 1}, {"T", 1}, {"C", 1}, {"NH", 1}, {"R", 1}},
             {4, 1, 2});
    CheckOps(result, "ATTENTION typical", Opcode::ATTENTION,
             {{"B", 2}, {"T", 3}, {"C", 4}, {"NH", 2}, {"R", 2}},
             {288, 36, 72});

    CheckOps(result, "GATE minimal", Opcode::GATE,
             {{"B", 1}, {"T", 1}, {"C", 1}, {"E_N", 1}, {"K", 1}},
             {1, 0, 0});
    CheckOps(result, "GATE typical", Opcode::GATE,
             {{"B", 2}, {"T", 3}, {"C", 4}, {"E_N", 5}, {"K", 2}},
             {120, 0, 0});

    CheckOps(result, "MOE_MATMUL minimal", Opcode::MOE_MATMUL,
             {{"B", 1}, {"T", 1}, {"C", 1}, {"OC", 1}, {"K", 1},
              {"E_N", 1}, {"is_merge", 0}, {"need_choose", 0}},
             {2, 0, 0}, hardware);
    CheckOps(result, "MOE_MATMUL typical merge", Opcode::MOE_MATMUL,
             {{"B", 2}, {"T", 3}, {"C", 4}, {"OC", 5}, {"K", 2},
              {"E_N", 4}, {"is_merge", 1}, {"need_choose", 0}},
             {540, 0, 0}, hardware);
    PublishedNpuHardwareView perf_hardware = hardware;
    perf_hardware.use_performance_gemm = true;
    CheckOps(result, "MOE_MATMUL performance mode", Opcode::MOE_MATMUL,
             {{"B", 2}, {"T", 3}, {"C", 4}, {"OC", 5}, {"K", 2},
              {"E_N", 4}, {"is_merge", 1}, {"need_choose", 0}},
             {640, 0, 0}, perf_hardware);

    CheckOps(result, "GELU minimal", Opcode::GELU, {{"N", 1}},
             {0, 1, 4});
    CheckOps(result, "GELU typical", Opcode::GELU, {{"N", 7}},
             {0, 7, 28});
    CheckOps(result, "SILU minimal", Opcode::SILU, {{"N", 1}},
             {0, 1, 3});
    CheckOps(result, "SILU typical", Opcode::SILU, {{"N", 7}},
             {0, 7, 21});
    CheckOps(result, "SWIGLU minimal", Opcode::SWIGLU, {{"N", 1}},
             {0, 1, 4});
    CheckOps(result, "SWIGLU typical", Opcode::SWIGLU, {{"N", 7}},
             {0, 7, 28});
    CheckOps(result, "RELU minimal", Opcode::RELU, {{"N", 1}},
             {1, 0, 0});
    CheckOps(result, "RELU typical", Opcode::RELU, {{"N", 7}},
             {7, 0, 0});
    CheckOps(result, "RESIDUAL minimal", Opcode::RESIDUAL, {{"N", 1}},
             {0, 0, 1});
    CheckOps(result, "RESIDUAL typical", Opcode::RESIDUAL, {{"N", 7}},
             {0, 0, 7});

    CheckOps(result, "LAYERNORM minimal", Opcode::LAYERNORM,
             {{"B", 1}, {"T", 1}, {"C", 1}}, {0, 1, 11});
    CheckOps(result, "LAYERNORM typical", Opcode::LAYERNORM,
             {{"B", 2}, {"T", 3}, {"C", 4}}, {0, 6, 210});
    CheckOps(result, "RMSNORM minimal frozen SFU", Opcode::RMSNORM,
             {{"B", 1}, {"T", 1}, {"C", 1}}, {0, 0, 5});
    CheckOps(result, "RMSNORM typical frozen SFU", Opcode::RMSNORM,
             {{"B", 2}, {"T", 3}, {"C", 4}}, {0, 0, 102});

    CheckOps(result, "ROPE minimal", Opcode::ROPE,
             {{"B", 1}, {"T", 1}, {"C", 1}, {"NH", 1}}, {0, 0, 3});
    CheckOps(result, "ROPE typical", Opcode::ROPE,
             {{"B", 2}, {"T", 3}, {"C", 8}, {"NH", 2}}, {0, 0, 72});

    CheckOps(result, "SPLIT_MATMUL minimal", Opcode::SPLIT_MATMUL,
             {{"B", 1}, {"T", 1}, {"C", 1}, {"dim", 1}, {"slice", 1}},
             {0, 0, 0});
    CheckOps(result, "SPLIT_MATMUL typical", Opcode::SPLIT_MATMUL,
             {{"B", 2}, {"T", 3}, {"C", 4}, {"dim", 2}, {"slice", 4}},
             {0, 0, 0});
    CheckOps(result, "MERGE_MATMUL minimal", Opcode::MERGE_MATMUL,
             {{"B", 1}, {"T", 1}, {"C", 1}, {"dim", 1}, {"slice", 1}},
             {1, 0, 0});
    CheckOps(result, "MERGE_MATMUL typical", Opcode::MERGE_MATMUL,
             {{"B", 2}, {"T", 3}, {"C", 4}, {"dim", 2}, {"slice", 4}},
             {24, 0, 0});

    CheckOps(result, "DUMMY minimal", static_cast<Opcode>(0x15), {},
             {10, 0, 0});
    CheckOps(result, "DUMMY typical", static_cast<Opcode>(0x15), {},
             {10, 0, 0});
    CheckExactWork(result);

    Reject(result, "missing published parameter is rejected", [&] {
        (void)EvaluatePublishedNpuOps(Opcode::GELU, {});
    });
    Reject(result, "negative published parameter is rejected", [&] {
        (void)EvaluatePublishedNpuOps(Opcode::GELU, {{"N", -1}});
    });
    Reject(result, "zero convolution divisor is rejected", [&] {
        (void)EvaluatePublishedNpuOps(
            Opcode::MAXPOOL,
            {{"B", 1}, {"W", 1}, {"H", 1}, {"C", 1},
             {"pX", 0}, {"pY", 0}, {"sX", 0}, {"sY", 1},
             {"kX", 1}, {"kY", 1}});
    });
    Reject(result, "published operation multiplication overflow is rejected",
           [&] {
               const int large = std::numeric_limits<int>::max();
               (void)EvaluatePublishedNpuOps(
                   Opcode::GATE,
                   {{"B", large}, {"T", large}, {"C", large},
                    {"E_N", large}, {"K", 1}});
           });
    Reject(result, "zero MATMUL hardware dimension is rejected", [&] {
        PublishedNpuHardwareView bad = hardware;
        bad.exu_x_dims = 0;
        (void)EvaluatePublishedNpuOps(
            Opcode::MATMUL,
            {{"B", 1}, {"T", 1}, {"C", 1}, {"OC", 1}}, bad);
    });
    Reject(result, "non-published operation cost is rejected", [&] {
        (void)EvaluatePublishedNpuOps(Opcode::DTE_SEND, {});
    });
    return result;
}
