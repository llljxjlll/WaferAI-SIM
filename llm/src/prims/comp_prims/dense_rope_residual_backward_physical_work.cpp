#include "prims/dense_rope_residual_backward_physical_work.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace {
constexpr uint64_t kMaxProfile = (uint64_t{1} << 30) - 1;
constexpr uint64_t kSramExtent = uint64_t{1} << 16;
constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

uint64_t CheckedMul(uint64_t a, uint64_t b, const char *field) {
    if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::overflow_error(std::string(field) + " overflow");
    return a * b;
}

uint64_t CheckedAdd(uint64_t a, uint64_t b, const char *field) {
    if (a > std::numeric_limits<uint64_t>::max() - b)
        throw std::overflow_error(std::string(field) + " overflow");
    return a + b;
}

void PositiveProfile(uint64_t value, const char *field) {
    if (value == 0 || value > kMaxProfile)
        throw std::invalid_argument(std::string(field) + " needs positive 30-bit source profile");
}

void CheckSpan(const DenseReverseSramSpan &span, uint64_t bytes,
               DenseReverseDType dtype, const char *role) {
    if (span.dtype != dtype || span.bytes != bytes || bytes == 0 ||
        span.byte_address % 16 != 0 || span.byte_address >= kSramExtent ||
        span.bytes > kSramExtent - span.byte_address)
        throw std::invalid_argument(std::string(role) + " typed SRAM extent/address differs");
}

bool Overlap(const DenseReverseSramSpan &a,
             const DenseReverseSramSpan &b) {
    return uint64_t{a.byte_address} < uint64_t{b.byte_address} + b.bytes &&
           uint64_t{b.byte_address} < uint64_t{a.byte_address} + a.bytes;
}

uint64_t PositionDigest(const std::vector<uint32_t> &positions) {
    uint64_t digest = kFnvOffset;
    for (uint32_t position : positions) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            digest ^= (position >> shift) & 0xffU;
            digest *= kFnvPrime;
        }
    }
    return digest;
}
} // namespace

RopeQkBackwardTimingWork BuildRopeQkBackwardTimingWork(
    const RopeQkBackwardSourceWitness &source,
    const RopeQkBackwardPhysicalTile &tile) {
    if (source.forward_op_ref.empty() ||
        source.upstream_gradient_producer_ref.empty() ||
        source.forward_op_ref == source.upstream_gradient_producer_ref)
        throw std::invalid_argument("RoPE inverse needs distinct real forward/upstream source refs");
    for (const auto &[value, field] : {
             std::pair<uint64_t, const char *>{source.logical_tokens, "logical_tokens"},
             {source.rank_tokens, "rank_tokens"},
             {source.logical_query_heads, "logical_query_heads"},
             {source.logical_kv_heads, "logical_kv_heads"},
             {source.rank_query_heads, "rank_query_heads"},
             {source.rank_kv_heads, "rank_kv_heads"},
             {source.tp_degree, "tp_degree"},
             {source.head_dim, "head_dim"},
             {source.rotary_dim, "rotary_dim"},
             {source.max_position_embeddings, "max_position_embeddings"},
         })
        PositiveProfile(value, field);
    if (source.rank_tokens != source.logical_tokens ||
        source.logical_query_heads != CheckedMul(source.rank_query_heads,
                                                 source.tp_degree, "query TP") ||
        source.logical_kv_heads != CheckedMul(source.rank_kv_heads,
                                              source.tp_degree, "KV TP") ||
        source.rank_query_heads < source.rank_kv_heads ||
        source.rank_query_heads % source.rank_kv_heads != 0 ||
        source.logical_query_heads % source.logical_kv_heads != 0 ||
        source.head_dim != source.rotary_dim || source.head_dim % 2 != 0)
        throw std::invalid_argument("RoPE inverse rank GQA/TP/rotary shape differs from source");
    if (source.position_trace.size() != source.rank_tokens)
        throw std::invalid_argument("RoPE inverse needs one physical INT32 position per rank token");
    for (uint32_t position : source.position_trace)
        if (position >= source.max_position_embeddings)
            throw std::invalid_argument("RoPE inverse INT32 position exceeds source table");
    const uint64_t packed_heads = CheckedAdd(
        source.rank_query_heads,
        CheckedMul(2, source.rank_kv_heads, "RoPE packed KV heads"),
        "RoPE packed heads");
    const uint64_t packed_elements = CheckedMul(
        CheckedMul(source.rank_tokens, packed_heads, "RoPE packed tokens"),
        source.head_dim, "RoPE packed elements");
    const uint64_t position_bytes = CheckedMul(4, source.rank_tokens,
                                               "RoPE INT32 position bytes");
    const uint64_t packed_bytes = CheckedMul(2, packed_elements,
                                             "RoPE FP16 packed bytes");
    CheckSpan(tile.position_ids, position_bytes, DenseReverseDType::INT32,
              "RoPE position_ids");
    CheckSpan(tile.rotated_upstream, packed_bytes, DenseReverseDType::FP16,
              "RoPE rotated_upstream");
    CheckSpan(tile.packed_output, packed_bytes, DenseReverseDType::FP16,
              "RoPE packed_output");
    if (Overlap(tile.position_ids, tile.rotated_upstream) ||
        Overlap(tile.position_ids, tile.packed_output) ||
        Overlap(tile.rotated_upstream, tile.packed_output))
        throw std::invalid_argument("RoPE inverse needs independent INT32/FP16 SRAM spans");
    RopeQkBackwardTimingWork work;
    work.position_read_bytes = position_bytes;
    work.upstream_read_bytes = packed_bytes;
    work.output_write_bytes = packed_bytes;
    work.inverse_rotary_pairs = CheckedMul(
        CheckedMul(source.rank_tokens,
                   CheckedAdd(source.rank_query_heads, source.rank_kv_heads,
                              "RoPE QK heads"), "RoPE QK tokens"),
        source.rotary_dim / 2, "RoPE inverse pairs");
    work.passed_v_elements = CheckedMul(
        CheckedMul(source.rank_tokens, source.rank_kv_heads,
                   "RoPE pass-through V tokens"),
        source.head_dim, "RoPE pass-through V elements");
    // Each inverse pair uses four multiplies/two adds. Angle sin/cos is
    // shared by Q/K heads for a given token and rotary frequency.
    work.vec_ops = CheckedAdd(CheckedMul(6, work.inverse_rotary_pairs,
                                         "RoPE inverse vector work"),
                              work.passed_v_elements,
                              "RoPE inverse V copy work");
    work.sfu_ops = CheckedMul(source.rank_tokens, source.rotary_dim,
                              "RoPE inverse shared sin/cos work");
    work.position_trace_digest = PositionDigest(source.position_trace);
    return work;
}

ResidualDualDxTimingWork BuildResidualDualDxTimingWork(
    const ResidualDualDxSourceWitness &source,
    const ResidualDualDxPhysicalTile &tile) {
    if (source.forward_op_ref.empty() ||
        source.left_forward_value_ref.empty() ||
        source.right_forward_value_ref.empty() ||
        source.upstream_gradient_producer_ref.empty() ||
        source.left_forward_value_ref == source.right_forward_value_ref ||
        source.forward_op_ref == source.upstream_gradient_producer_ref)
        throw std::invalid_argument("Residual dual dX needs distinct two forward inputs and upstream source");
    PositiveProfile(source.logical_rows, "Residual logical_rows");
    PositiveProfile(source.rank_rows, "Residual rank_rows");
    PositiveProfile(source.tp_degree, "Residual tp_degree");
    PositiveProfile(source.hidden_size, "Residual hidden_size");
    if (source.logical_rows != CheckedMul(source.rank_rows, source.tp_degree,
                                          "Residual source row TP"))
        throw std::invalid_argument("Residual dual dX rank rows differ from forward TP source");
    const uint64_t elements = CheckedMul(source.rank_rows, source.hidden_size,
                                         "Residual rank elements");
    const uint64_t bytes = CheckedMul(2, elements, "Residual FP16 rank bytes");
    CheckSpan(tile.upstream, bytes, DenseReverseDType::FP16,
              "Residual upstream");
    CheckSpan(tile.left_output, bytes, DenseReverseDType::FP16,
              "Residual left_output");
    CheckSpan(tile.right_output, bytes, DenseReverseDType::FP16,
              "Residual right_output");
    if (Overlap(tile.upstream, tile.left_output) ||
        Overlap(tile.upstream, tile.right_output) ||
        Overlap(tile.left_output, tile.right_output))
        throw std::invalid_argument("Residual dual dX needs two independent OWNED output spans");
    ResidualDualDxTimingWork work;
    work.upstream_read_bytes = bytes;
    work.left_write_bytes = bytes;
    work.right_write_bytes = bytes;
    work.vec_ops = CheckedMul(2, elements, "Residual dual-copy vector work");
    return work;
}
