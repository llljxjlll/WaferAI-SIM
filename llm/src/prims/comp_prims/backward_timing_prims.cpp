#include "prims/backward_timing_prims.h"

#include "utils/prim_utils.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

REGISTER_PRIM(norm_backward_timing, PrimId::NORM_BACKWARD_TIMING);
REGISTER_PRIM(attention_backward_timing, PrimId::ATTENTION_BACKWARD_TIMING);

namespace {

uint64_t Mul(uint64_t a, uint64_t b, const char *field) {
    if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::overflow_error(std::string(field) + " profile overflow");
    return a * b;
}

uint64_t Add(uint64_t a, uint64_t b, const char *field) {
    if (a > std::numeric_limits<uint64_t>::max() - b)
        throw std::overflow_error(std::string(field) + " profile overflow");
    return a + b;
}

uint64_t Param(const NpuBase &prim, const char *name) {
    const auto it = prim.param_value.find(name);
    if (it == prim.param_value.end() || it->second < 0)
        throw std::invalid_argument(prim.name + " missing/nonnegative " + name);
    return static_cast<uint64_t>(it->second);
}

void ExactParamSet(const NpuBase &prim) {
    if (prim.param_value.size() != prim.param_name.size())
        throw std::invalid_argument(prim.name + " parameter set is incomplete");
    for (const auto &name : prim.param_name)
        if (prim.param_value.count(name) != 1)
            throw std::invalid_argument(prim.name + " parameter set differs");
}

void Positive(uint64_t value, const std::string &field) {
    if (value == 0)
        throw std::invalid_argument(field + " must be positive");
}

void IntElements(uint64_t bytes, const std::string &field) {
    if (bytes % 2 != 0 || bytes / 2 > INT_MAX)
        throw std::overflow_error(field + " exceeds NpuBase element capacity");
}

struct Range {
    uint64_t begin;
    uint64_t end;
};

Range SramRange(int offset, uint64_t bytes, const std::string &field) {
    // Generic external COMPUTE addresses are u16 byte offsets. Bound both
    // endpoints here so the internal timing wire cannot manufacture a larger
    // physical region than the public artifact can address.
    if (offset < 0 || offset % 16 != 0)
        throw std::invalid_argument(field + " requires aligned SRAM address");
    const uint64_t start = static_cast<uint64_t>(offset);
    const uint64_t end = Add(start, bytes, field.c_str());
    if (end > UINT16_MAX + uint64_t{1})
        throw std::invalid_argument(field + " exceeds 16-bit SRAM range");
    return {start, end};
}

bool Overlap(Range a, Range b) {
    return a.begin < b.end && b.begin < a.end;
}

void ValidateRanges(const NpuBase &prim, const BackwardTimingWork &work) {
    const Range forward = SramRange(prim.inp_offset, work.forward_input_bytes,
                                    prim.name + " forward_input");
    const Range upstream = SramRange(prim.data_offset, work.upstream_bytes,
                                     prim.name + " upstream");
    const Range output = SramRange(prim.out_offset, work.output_bytes,
                                   prim.name + " output");
    if (Overlap(forward, upstream) || Overlap(forward, output) ||
        Overlap(upstream, output))
        throw std::invalid_argument(prim.name + " SRAM tensors overlap");
}

void StrictWire(const NpuBase &prim) {
    if (prim_wire::LegacyCompatibilityEnabled())
        throw std::invalid_argument(prim.name + " requires strict Prim wire");
}

} // namespace

norm_backward_timing::norm_backward_timing() {
    name = "norm_backward_timing";
    datatype = FP16;
    skip_input = true;
    skip_output = true;
    param_name = {"ROWS", "HIDDEN", "TP", "MODE"};
}

BackwardTimingWork norm_backward_timing::work() const {
    ExactParamSet(*this);
    if (datatype != FP16)
        throw std::invalid_argument(name + " requires FP16 tensors");
    const uint64_t rows = Param(*this, "ROWS");
    const uint64_t hidden = Param(*this, "HIDDEN");
    const uint64_t tp = Param(*this, "TP");
    const uint64_t mode = Param(*this, "MODE");
    Positive(rows, name + " ROWS");
    Positive(hidden, name + " HIDDEN");
    Positive(tp, name + " TP");
    if (mode > 1)
        throw std::invalid_argument(name + " MODE must be RMS(0) or Layer(1)");
    const uint64_t tensor = Mul(rows, hidden, "NORM_BACKWARD tensor");
    const uint64_t bytes = Mul(tensor, 2, "NORM_BACKWARD bytes");
    IntElements(bytes, name + " tensor");
    BackwardTimingWork result;
    result.forward_input_bytes = bytes;
    result.upstream_bytes = bytes;
    result.output_bytes = bytes;
    result.sfu_ops = Mul(rows, mode ? 3 : 1, "NORM_BACKWARD SFU");
    result.vec_ops = Mul(rows,
                         Add(Mul(hidden, mode ? 8 : 6,
                                 "NORM_BACKWARD vector scale"),
                             mode ? 3 : 1, "NORM_BACKWARD vector scale"),
                         "NORM_BACKWARD vector work");
    ValidateRanges(*this, result);
    return result;
}

void norm_backward_timing::initialize() {
    const auto profile = work();
    data_size_input = {static_cast<int>(profile.forward_input_bytes / 2)};
    data_chunk = {{"upstream", static_cast<int>(profile.upstream_bytes / 2)},
                  {"output", static_cast<int>(profile.output_bytes / 2)}};
}

void norm_backward_timing::taskCore(TaskCoreContext &, string,
                                    u_int64_t &, u_int64_t &exu,
                                    u_int64_t &sfu, u_int64_t &vec) {
    const auto profile = work();
    exu = profile.exu_ops;
    sfu = profile.sfu_ops;
    vec = profile.vec_ops;
}

vector<sc_bv<128>> norm_backward_timing::serialize() {
    StrictWire(*this);
    work();
    return NpuBase::serialize();
}

void norm_backward_timing::deserialize(vector<sc_bv<128>> wire) {
    StrictWire(*this);
    NpuBase::deserialize(std::move(wire));
    work();
}

attention_backward_timing::attention_backward_timing() {
    name = "attention_backward_timing";
    datatype = FP16;
    skip_input = true;
    skip_output = true;
    param_name = {"TOKENS", "RANK_HEADS", "RANK_KV_HEADS", "HEAD_DIM",
                  "TP", "SEQUENCES", "PAIRS"};
}

BackwardTimingWork attention_backward_timing::work() const {
    ExactParamSet(*this);
    if (datatype != FP16)
        throw std::invalid_argument(name + " requires FP16 tensors");
    const uint64_t tokens = Param(*this, "TOKENS");
    const uint64_t heads = Param(*this, "RANK_HEADS");
    const uint64_t kv_heads = Param(*this, "RANK_KV_HEADS");
    const uint64_t dim = Param(*this, "HEAD_DIM");
    const uint64_t tp = Param(*this, "TP");
    const uint64_t sequences = Param(*this, "SEQUENCES");
    const uint64_t pairs = Param(*this, "PAIRS");
    for (const auto value : {tokens, heads, kv_heads, dim, tp, sequences, pairs})
        Positive(value, name + " rank shape/profile");
    if (kv_heads > heads || heads % kv_heads != 0)
        throw std::invalid_argument(name + " KV heads do not divide query heads");
    if (dim % 2 != 0)
        throw std::invalid_argument(name + " HEAD_DIM must be even");
    if (tokens % sequences != 0)
        throw std::invalid_argument(name + " TOKENS not uniform across SEQUENCES");
    const uint64_t per_sequence = tokens / sequences;
    const uint64_t exact_pairs = Mul(
        sequences,
        Mul(per_sequence, Add(per_sequence, 1, "ATTENTION_BACKWARD context"),
            "ATTENTION_BACKWARD causal pairs") / 2,
        "ATTENTION_BACKWARD aggregate pairs");
    if (pairs != exact_pairs)
        throw std::invalid_argument(name + " PAIRS not exact causal rank profile");
    const uint64_t packed_heads = Add(
        heads, Mul(2, kv_heads, "ATTENTION_BACKWARD packed heads"),
        "ATTENTION_BACKWARD packed heads");
    const uint64_t packed_elements = Mul(
        Mul(tokens, packed_heads, "ATTENTION_BACKWARD packed tensor"), dim,
        "ATTENTION_BACKWARD packed tensor");
    const uint64_t upstream_elements = Mul(
        Mul(tokens, heads, "ATTENTION_BACKWARD upstream"), dim,
        "ATTENTION_BACKWARD upstream");
    BackwardTimingWork result;
    result.forward_input_bytes = Mul(packed_elements, 2,
                                     "ATTENTION_BACKWARD forward bytes");
    result.upstream_bytes = Mul(upstream_elements, 2,
                                "ATTENTION_BACKWARD upstream bytes");
    result.output_bytes = result.forward_input_bytes;
    IntElements(result.forward_input_bytes, name + " packed tensor");
    IntElements(result.upstream_bytes, name + " upstream tensor");
    result.exu_ops = Mul(Mul(Mul(8, heads, "ATTENTION_BACKWARD EXU"), dim,
                             "ATTENTION_BACKWARD EXU"), pairs,
                         "ATTENTION_BACKWARD EXU");
    result.sfu_ops = Mul(Mul(2, heads, "ATTENTION_BACKWARD SFU"), pairs,
                         "ATTENTION_BACKWARD SFU");
    result.vec_ops = Mul(Mul(Mul(6, heads, "ATTENTION_BACKWARD VEC"), dim,
                             "ATTENTION_BACKWARD VEC"), tokens,
                         "ATTENTION_BACKWARD VEC");
    ValidateRanges(*this, result);
    return result;
}

void attention_backward_timing::initialize() {
    const auto profile = work();
    data_size_input = {static_cast<int>(profile.forward_input_bytes / 2)};
    data_chunk = {{"upstream", static_cast<int>(profile.upstream_bytes / 2)},
                  {"output", static_cast<int>(profile.output_bytes / 2)}};
}

void attention_backward_timing::taskCore(TaskCoreContext &, string,
                                         u_int64_t &, u_int64_t &exu,
                                         u_int64_t &sfu, u_int64_t &vec) {
    const auto profile = work();
    exu = profile.exu_ops;
    sfu = profile.sfu_ops;
    vec = profile.vec_ops;
}

vector<sc_bv<128>> attention_backward_timing::serialize() {
    StrictWire(*this);
    work();
    return NpuBase::serialize();
}

void attention_backward_timing::deserialize(vector<sc_bv<128>> wire) {
    StrictWire(*this);
    NpuBase::deserialize(std::move(wire));
    work();
}
