#include "prims/dense_rope_residual_backward_physical_work.h"

#include "isa/prim_id.h"
#include "utils/prim_utils.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

REGISTER_PRIM(rope_backward_timing, PrimId::ROPE_BACKWARD_TIMING);
REGISTER_PRIM(residual_backward_timing, PrimId::RESIDUAL_BACKWARD_TIMING);

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
namespace {
uint64_t PrimParameter(const NpuBase &prim, const char *name) {
    const auto it = prim.param_value.find(name);
    if (it == prim.param_value.end() || it->second < 0)
        throw std::invalid_argument(prim.name + " missing/nonnegative " + name);
    return static_cast<uint64_t>(it->second);
}

void RequireExactPrimParameters(const NpuBase &prim) {
    if (prim.param_value.size() != prim.param_name.size())
        throw std::invalid_argument(prim.name + " parameter set is incomplete");
    for (const auto &name : prim.param_name)
        if (prim.param_value.count(name) != 1)
            throw std::invalid_argument(prim.name + " parameter set differs");
}

void RequireStrictBackwardWire(const NpuBase &prim) {
    if (prim_wire::LegacyCompatibilityEnabled())
        throw std::invalid_argument(prim.name + " requires strict Prim wire");
}
} // namespace

rope_backward_timing::rope_backward_timing() {
    name = "rope_backward_timing";
    datatype = FP16;
    skip_input = true;
    skip_output = true;
    param_name = {"LOGICAL_TOKENS", "RANK_TOKENS", "LOGICAL_Q_HEADS",
                  "LOGICAL_KV_HEADS", "RANK_Q_HEADS", "RANK_KV_HEADS",
                  "TP", "HEAD_DIM", "ROTARY_DIM", "MAX_POSITIONS",
                  "POSITION_TRACE_TAG"};
}

RopeQkBackwardTimingWork rope_backward_timing::work() const {
    RequireExactPrimParameters(*this);
    if (datatype != FP16)
        throw std::invalid_argument(name + " requires FP16 upstream/output");
    RopeQkBackwardSourceWitness source;
    source.forward_op_ref = "public::rope_forward";
    source.upstream_gradient_producer_ref = "public::rope_upstream";
    source.logical_tokens = PrimParameter(*this, "LOGICAL_TOKENS");
    source.rank_tokens = PrimParameter(*this, "RANK_TOKENS");
    source.logical_query_heads = PrimParameter(*this, "LOGICAL_Q_HEADS");
    source.logical_kv_heads = PrimParameter(*this, "LOGICAL_KV_HEADS");
    source.rank_query_heads = PrimParameter(*this, "RANK_Q_HEADS");
    source.rank_kv_heads = PrimParameter(*this, "RANK_KV_HEADS");
    source.tp_degree = PrimParameter(*this, "TP");
    source.head_dim = PrimParameter(*this, "HEAD_DIM");
    source.rotary_dim = PrimParameter(*this, "ROTARY_DIM");
    source.max_position_embeddings = PrimParameter(*this, "MAX_POSITIONS");
    source.position_trace.reserve(source.rank_tokens);
    for (uint64_t token = 0; token < source.rank_tokens; ++token) {
        if (token >= source.max_position_embeddings || token > UINT32_MAX)
            throw std::invalid_argument(name + " canonical position trace exceeds table");
        source.position_trace.push_back(static_cast<uint32_t>(token));
    }
    const uint64_t packed_heads = CheckedAdd(
        source.rank_query_heads, CheckedMul(2, source.rank_kv_heads,
                                            "rope Prim packed KV"),
        "rope Prim packed heads");
    const uint64_t packed_bytes = CheckedMul(
        2, CheckedMul(source.rank_tokens,
                      CheckedMul(packed_heads, source.head_dim,
                                 "rope Prim packed width"),
                      "rope Prim packed elements"),
        "rope Prim packed bytes");
    RopeQkBackwardPhysicalTile tile;
    tile.position_ids = {static_cast<uint32_t>(inp_offset),
                         CheckedMul(4, source.rank_tokens,
                                    "rope Prim position bytes"),
                         DenseReverseDType::INT32};
    tile.rotated_upstream = {static_cast<uint32_t>(data_offset), packed_bytes,
                             DenseReverseDType::FP16};
    tile.packed_output = {static_cast<uint32_t>(out_offset), packed_bytes,
                          DenseReverseDType::FP16};
    const auto result = BuildRopeQkBackwardTimingWork(source, tile);
    if ((result.position_trace_digest & ((uint64_t{1} << 30) - 1)) !=
        PrimParameter(*this, "POSITION_TRACE_TAG"))
        throw std::invalid_argument(name + " position trace tag differs");
    return result;
}

void rope_backward_timing::initialize() {
    const auto profile = work();
    data_size_input = {static_cast<int>(profile.position_read_bytes / 4),
                       static_cast<int>(profile.upstream_read_bytes / 2)};
    data_chunk = {{"upstream", static_cast<int>(profile.upstream_read_bytes / 2)},
                  {"output", static_cast<int>(profile.output_write_bytes / 2)}};
}

void rope_backward_timing::taskCore(TaskCoreContext &, string, u_int64_t &,
                                    u_int64_t &exu, u_int64_t &sfu,
                                    u_int64_t &vec) {
    const auto profile = work();
    exu = 0;
    sfu = profile.sfu_ops;
    vec = profile.vec_ops;
    std::cout << "[DENSE_BACKWARD_PUBLIC] stage=rope"
              << " position_read_bytes=" << profile.position_read_bytes
              << " upstream_read_bytes=" << profile.upstream_read_bytes
              << " output_write_bytes=" << profile.output_write_bytes
              << " sfu_ops=" << profile.sfu_ops
              << " vec_ops=" << profile.vec_ops << " pass=1\n";
}

vector<sc_bv<128>> rope_backward_timing::serialize() {
    RequireStrictBackwardWire(*this);
    work();
    return NpuBase::serialize();
}

void rope_backward_timing::deserialize(vector<sc_bv<128>> wire) {
    RequireStrictBackwardWire(*this);
    NpuBase::deserialize(std::move(wire));
    work();
}

residual_backward_timing::residual_backward_timing() {
    name = "residual_backward_timing";
    datatype = FP16;
    skip_input = true;
    skip_output = true;
    param_name = {"LOGICAL_ROWS", "RANK_ROWS", "TP", "HIDDEN",
                  "RIGHT_OUTPUT_OFFSET"};
}

ResidualDualDxTimingWork residual_backward_timing::work() const {
    RequireExactPrimParameters(*this);
    if (datatype != FP16)
        throw std::invalid_argument(name + " requires FP16 tensors");
    ResidualDualDxSourceWitness source;
    source.forward_op_ref = "public::residual_forward";
    source.left_forward_value_ref = "public::residual_left";
    source.right_forward_value_ref = "public::residual_right";
    source.upstream_gradient_producer_ref = "public::residual_upstream";
    source.logical_rows = PrimParameter(*this, "LOGICAL_ROWS");
    source.rank_rows = PrimParameter(*this, "RANK_ROWS");
    source.tp_degree = PrimParameter(*this, "TP");
    source.hidden_size = PrimParameter(*this, "HIDDEN");
    const uint64_t bytes = CheckedMul(
        2, CheckedMul(source.rank_rows, source.hidden_size,
                      "residual Prim elements"),
        "residual Prim bytes");
    const uint64_t right = PrimParameter(*this, "RIGHT_OUTPUT_OFFSET");
    if (inp_offset < 0 || data_offset < 0 || out_offset < 0 ||
        static_cast<uint64_t>(inp_offset) + bytes > kSramExtent ||
        right > UINT32_MAX)
        throw std::invalid_argument(name + " forward witness SRAM extent/address differs");
    ResidualDualDxPhysicalTile tile;
    tile.upstream = {static_cast<uint32_t>(data_offset), bytes,
                     DenseReverseDType::FP16};
    tile.left_output = {static_cast<uint32_t>(out_offset), bytes,
                        DenseReverseDType::FP16};
    tile.right_output = {static_cast<uint32_t>(right), bytes,
                         DenseReverseDType::FP16};
    const auto result = BuildResidualDualDxTimingWork(source, tile);
    DenseReverseSramSpan forward{static_cast<uint32_t>(inp_offset), bytes,
                                 DenseReverseDType::FP16};
    if (forward.byte_address % 16 != 0 || Overlap(forward, tile.upstream) ||
        Overlap(forward, tile.left_output) || Overlap(forward, tile.right_output))
        throw std::invalid_argument(name + " forward/upstream/dual outputs overlap");
    return result;
}

void residual_backward_timing::initialize() {
    const auto profile = work();
    data_size_input = {static_cast<int>(profile.upstream_read_bytes / 2),
                       static_cast<int>(profile.upstream_read_bytes / 2)};
    data_chunk = {{"upstream", static_cast<int>(profile.upstream_read_bytes / 2)},
                  {"output", static_cast<int>(profile.left_write_bytes / 2)},
                  {"right_output", static_cast<int>(profile.right_write_bytes / 2)}};
}

void residual_backward_timing::taskCore(TaskCoreContext &, string, u_int64_t &,
                                        u_int64_t &exu, u_int64_t &sfu,
                                        u_int64_t &vec) {
    const auto profile = work();
    exu = 0; sfu = 0; vec = profile.vec_ops;
    std::cout << "[DENSE_BACKWARD_PUBLIC] stage=residual"
              << " forward_witness_bytes=" << profile.upstream_read_bytes
              << " upstream_read_bytes=" << profile.upstream_read_bytes
              << " left_write_bytes=" << profile.left_write_bytes
              << " right_write_bytes=" << profile.right_write_bytes
              << " vec_ops=" << profile.vec_ops << " pass=1\n";
}

vector<sc_bv<128>> residual_backward_timing::serialize() {
    RequireStrictBackwardWire(*this);
    work();
    return NpuBase::serialize();
}

void residual_backward_timing::deserialize(vector<sc_bv<128>> wire) {
    RequireStrictBackwardWire(*this);
    NpuBase::deserialize(std::move(wire));
    work();
}
