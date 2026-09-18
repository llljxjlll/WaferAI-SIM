#include "prims/weight_gradient_timing_prims.h"
#include "prims/timing_wgrad_output.h"

#include "utils/prim_utils.h"

#include <climits>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

REGISTER_PRIM(embedding_table_wgrad_timing,
              PrimId::EMBEDDING_TABLE_WGRAD_TIMING);
REGISTER_PRIM(norm_gamma_wgrad_timing, PrimId::NORM_GAMMA_WGRAD_TIMING);

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

uint64_t Param(const NpuBase &prim, const std::string &field) {
    const auto it = prim.param_value.find(field);
    if (it == prim.param_value.end() || it->second < 0)
        throw std::invalid_argument(prim.name + " missing/nonnegative " + field);
    return static_cast<uint64_t>(it->second);
}

void ExactParams(const NpuBase &prim) {
    if (prim.param_value.size() != prim.param_name.size())
        throw std::invalid_argument(prim.name + " parameter count differs");
    for (const auto &name : prim.param_name)
        if (prim.param_value.count(name) != 1)
            throw std::invalid_argument(prim.name + " parameter set differs");
}

void Positive(uint64_t value, const std::string &field) {
    if (value == 0)
        throw std::invalid_argument(field + " must be positive");
}

void Elements(uint64_t bytes, const std::string &field) {
    if (bytes % 2 != 0 || bytes / 2 > INT_MAX)
        throw std::overflow_error(field + " exceeds NpuBase element capacity");
}

struct Range { uint64_t start; uint64_t end; };

Range CheckBuffer(const WeightGradBufferABI &buffer, const std::string &field) {
    if (buffer.sram_byte_address % 16 != 0 || buffer.bytes == 0)
        throw std::invalid_argument(field + " requires nonempty aligned SRAM span");
    const uint64_t end = Add(buffer.sram_byte_address, buffer.bytes,
                             field.c_str());
    if (end > uint64_t{UINT16_MAX} + 1)
        throw std::invalid_argument(field + " exceeds 16-bit SRAM region");
    const uint64_t element = buffer.dtype == WeightGradBufferDType::FP16 ? 2 : 4;
    if (buffer.bytes % element != 0)
        throw std::invalid_argument(field + " violates typed element ABI");
    return {buffer.sram_byte_address, end};
}

WeightGradBufferABI Buffer(int address, uint64_t bytes,
                           WeightGradBufferDType dtype,
                           const std::string &field) {
    if (address < 0)
        throw std::invalid_argument(field + " negative SRAM address");
    WeightGradBufferABI result{static_cast<uint32_t>(address), bytes, dtype};
    CheckBuffer(result, field);
    return result;
}

void NonOverlap(std::initializer_list<Range> ranges,
                const std::string &field) {
    for (auto first = ranges.begin(); first != ranges.end(); ++first)
        for (auto second = first + 1; second != ranges.end(); ++second)
            if (first->start < second->end && second->start < first->end)
                throw std::invalid_argument(field + " typed SRAM spans overlap");
}

void Strict(const NpuBase &prim) {
    if (prim_wire::LegacyCompatibilityEnabled())
        throw std::invalid_argument(prim.name + " requires strict Prim wire");
}

std::string IndexName(unsigned i) {
    return "INDEX" + std::to_string(i / 10) + std::to_string(i % 10);
}

} // namespace

embedding_table_wgrad_timing::embedding_table_wgrad_timing() {
    name = "embedding_table_wgrad_timing";
    datatype = FP16; // NpuBase wire carrier; output has its own FP32 BufferABI.
    skip_input = true;
    skip_output = true;
    param_name = {"ROWS", "LOGICAL_ROWS", "TP", "VOCAB_SIZE",
                  "VOCAB_START", "VOCAB_ROWS", "HIDDEN", "TABLE_ADDR"};
    for (unsigned i = 0; i < 16; ++i)
        param_name.push_back(IndexName(i));
}

EmbeddingTableWGradWork embedding_table_wgrad_timing::work() const {
    ExactParams(*this);
    if (datatype != FP16)
        throw std::invalid_argument(name + " requires FP16 wire carrier");
    const uint64_t rows = Param(*this, "ROWS");
    const uint64_t logical = Param(*this, "LOGICAL_ROWS");
    const uint64_t tp = Param(*this, "TP");
    const uint64_t vocab = Param(*this, "VOCAB_SIZE");
    const uint64_t start = Param(*this, "VOCAB_START");
    const uint64_t tile = Param(*this, "VOCAB_ROWS");
    const uint64_t hidden = Param(*this, "HIDDEN");
    for (const auto value : {rows, logical, tp, vocab, tile, hidden})
        Positive(value, name + " rank geometry");
    if (rows > 16)
        throw std::invalid_argument(name + " ROWS must fit 16-index wire tile");
    if (logical != Mul(rows, tp, "EMBEDDING logical rows"))
        throw std::invalid_argument(name + " logical rows must equal ROWS*TP");
    const uint64_t tile_end = Add(start, tile, "EMBEDDING tile end");
    if (tile_end > vocab)
        throw std::invalid_argument(name + " vocabulary tile exceeds full vocab");

    EmbeddingTableWGradWork result;
    result.rank_rows = rows;
    const uint64_t index_bytes = Mul(rows, 4, "EMBEDDING INT32 indices");
    const uint64_t tile_elements = Mul(tile, hidden, "EMBEDDING table tile");
    const uint64_t row_elements = Mul(rows, hidden, "EMBEDDING upstream");
    const uint64_t table_bytes = Mul(tile_elements, 2, "EMBEDDING FP16 table");
    const uint64_t upstream_bytes = Mul(row_elements, 2, "EMBEDDING FP16 upstream");
    const uint64_t grad_bytes = Mul(tile_elements, 4, "EMBEDDING FP32 gradient");
    Elements(index_bytes, name + " indices");
    Elements(table_bytes, name + " table");
    Elements(upstream_bytes, name + " upstream");
    Elements(grad_bytes, name + " gradient");
    result.indices = Buffer(inp_offset, index_bytes,
                            WeightGradBufferDType::INT32, name + " indices");
    result.table = Buffer(static_cast<int>(Param(*this, "TABLE_ADDR")),
                          table_bytes, WeightGradBufferDType::FP16,
                          name + " table");
    result.upstream = Buffer(data_offset, upstream_bytes,
                             WeightGradBufferDType::FP16, name + " upstream");
    result.gradient = Buffer(out_offset, grad_bytes,
                             WeightGradBufferDType::FP32, name + " gradient");
    NonOverlap({CheckBuffer(result.indices, name + " indices"),
                CheckBuffer(result.table, name + " table"),
                CheckBuffer(result.upstream, name + " upstream"),
                CheckBuffer(result.gradient, name + " gradient")}, name);

    std::set<uint64_t> seen;
    const uint64_t row_weight_bytes = Mul(hidden, 2, "EMBEDDING source row");
    const uint64_t row_gradient_bytes = Mul(hidden, 4, "EMBEDDING gradient row");
    for (unsigned i = 0; i < 16; ++i) {
        const uint64_t token = Param(*this, IndexName(i));
        if (i >= rows) {
            if (token != 0)
                throw std::invalid_argument(name + " unused trace slot must be zero");
            continue;
        }
        if (token >= vocab)
            throw std::invalid_argument(name + " INT32 source index out of vocab");
        if (token < start || token >= tile_end)
            continue; // This indexed row belongs to another FP32 tile.
        ++result.selected_rows;
        seen.insert(token);
        const uint64_t local = token - start;
        const uint64_t source = Add(result.table.sram_byte_address,
                                    Mul(local, row_weight_bytes,
                                        "EMBEDDING selected source"),
                                    "EMBEDDING selected source");
        const uint64_t target = Add(result.gradient.sram_byte_address,
                                    Mul(local, row_gradient_bytes,
                                        "EMBEDDING selected FP32 target"),
                                    "EMBEDDING selected FP32 target");
        if (Add(source, row_weight_bytes, "EMBEDDING source end") >
                result.table.sram_byte_address + result.table.bytes ||
            Add(target, row_gradient_bytes, "EMBEDDING gradient end") >
                result.gradient.sram_byte_address + result.gradient.bytes)
            throw std::invalid_argument(name + " selected source/target escapes tile");
        result.selected_weight_sram_addresses.push_back(
            static_cast<uint32_t>(source));
        result.selected_gradient_sram_addresses.push_back(
            static_cast<uint32_t>(target));
    }
    if (result.selected_rows == 0)
        throw std::invalid_argument(name + " tile has no indexed gradient updates");
    result.unique_weight_rows = seen.size();
    result.row_collisions = result.selected_rows - result.unique_weight_rows;
    result.trace_vec_ops = rows; // Step 1: global INT32 index routing.
    result.scatter_vec_ops = Mul(result.selected_rows, hidden,
                                 "EMBEDDING FP16 scatter");
    result.fp32_accumulate_vec_ops = result.scatter_vec_ops;
    result.selected_weight_read_bytes = Mul(result.selected_rows,
                                            row_weight_bytes,
                                            "EMBEDDING source trace read");
    result.selected_upstream_read_bytes = Mul(result.selected_rows,
                                              row_weight_bytes,
                                              "EMBEDDING upstream read");
    result.fp32_read_modify_write_bytes = Mul(result.selected_rows,
                                               Mul(row_gradient_bytes, 2,
                                                   "EMBEDDING FP32 RMW"),
                                               "EMBEDDING FP32 RMW");
    return result;
}

void embedding_table_wgrad_timing::initialize() {
    const auto profile = work();
    data_size_input = {
        static_cast<int>(profile.indices.bytes / 2),
        static_cast<int>(profile.table.bytes / 2),
        static_cast<int>(profile.upstream.bytes / 2),
    };
    data_chunk = {{"table", static_cast<int>(profile.table.bytes / 2)},
                  {"upstream", static_cast<int>(profile.upstream.bytes / 2)},
                  {"output", static_cast<int>(profile.gradient.bytes / 2)}};
}

void embedding_table_wgrad_timing::taskCore(TaskCoreContext &context, string,
                                             u_int64_t &dram, u_int64_t &exu,
                                             u_int64_t &sfu, u_int64_t &vec) {
    const auto profile = work();
    dram = exu = sfu = 0;
    vec = Add(profile.trace_vec_ops,
              Add(profile.scatter_vec_ops, profile.fp32_accumulate_vec_ops,
                  "EMBEDDING two-step work"), "EMBEDDING two-step work");
    MaterializeTimingWgrad(context, profile.gradient);
}

vector<sc_bv<128>> embedding_table_wgrad_timing::serialize() {
    Strict(*this);
    work();
    return NpuBase::serialize();
}

void embedding_table_wgrad_timing::deserialize(vector<sc_bv<128>> wire) {
    Strict(*this);
    NpuBase::deserialize(std::move(wire));
    work();
}

norm_gamma_wgrad_timing::norm_gamma_wgrad_timing() {
    name = "norm_gamma_wgrad_timing";
    datatype = FP16; // FP32 destination is explicit in gamma_gradient BufferABI.
    skip_input = true;
    skip_output = true;
    param_name = {"ROWS", "LOGICAL_ROWS", "HIDDEN", "TP", "MODE"};
}

NormGammaWGradWork norm_gamma_wgrad_timing::work() const {
    ExactParams(*this);
    if (datatype != FP16)
        throw std::invalid_argument(name + " requires FP16 activation/upstream");
    const uint64_t rows = Param(*this, "ROWS");
    const uint64_t logical = Param(*this, "LOGICAL_ROWS");
    const uint64_t hidden = Param(*this, "HIDDEN");
    const uint64_t tp = Param(*this, "TP");
    const uint64_t mode = Param(*this, "MODE");
    for (const auto value : {rows, logical, hidden, tp})
        Positive(value, name + " rank geometry");
    if (mode > 1)
        throw std::invalid_argument(name + " MODE must be RMS(0) or Layer(1)");
    if (logical != Mul(rows, tp, "NORM_GAMMA logical rows"))
        throw std::invalid_argument(name + " logical rows must equal ROWS*TP");
    const uint64_t elements = Mul(rows, hidden, "NORM_GAMMA tensor");
    const uint64_t source_bytes = Mul(elements, 2, "NORM_GAMMA FP16 source");
    const uint64_t gradient_bytes = Mul(hidden, 4, "NORM_GAMMA FP32 gradient");
    Elements(source_bytes, name + " source");
    Elements(gradient_bytes, name + " gradient");
    NormGammaWGradWork result;
    result.rank_rows = rows;
    result.hidden = hidden;
    result.activation = Buffer(inp_offset, source_bytes,
                               WeightGradBufferDType::FP16, name + " activation");
    result.upstream = Buffer(data_offset, source_bytes,
                             WeightGradBufferDType::FP16, name + " upstream");
    result.gamma_gradient = Buffer(out_offset, gradient_bytes,
                                   WeightGradBufferDType::FP32,
                                   name + " gamma gradient");
    NonOverlap({CheckBuffer(result.activation, name + " activation"),
                CheckBuffer(result.upstream, name + " upstream"),
                CheckBuffer(result.gamma_gradient, name + " gamma gradient")},
               name);
    result.normalization_vec_ops = Mul(rows,
        Add(Mul(hidden, mode ? 3 : 2, "NORM_GAMMA normalize"), mode ? 2 : 1,
            "NORM_GAMMA normalize"), "NORM_GAMMA normalize");
    result.fp32_accumulate_vec_ops = elements;
    result.sfu_ops = rows;
    result.fp32_read_modify_write_bytes = Mul(elements, 8,
                                               "NORM_GAMMA FP32 RMW");
    return result;
}

void norm_gamma_wgrad_timing::initialize() {
    const auto profile = work();
    data_size_input = {
        static_cast<int>(profile.activation.bytes / 2),
        static_cast<int>(profile.upstream.bytes / 2),
    };
    data_chunk = {{"upstream", static_cast<int>(profile.upstream.bytes / 2)},
                  {"output",
                   static_cast<int>(profile.gamma_gradient.bytes / 2)}};
}

void norm_gamma_wgrad_timing::taskCore(TaskCoreContext &context, string,
                                        u_int64_t &dram, u_int64_t &exu,
                                        u_int64_t &sfu, u_int64_t &vec) {
    const auto profile = work();
    dram = exu = 0;
    sfu = profile.sfu_ops;
    vec = Add(profile.normalization_vec_ops, profile.fp32_accumulate_vec_ops,
              "NORM_GAMMA vector work");
    MaterializeTimingWgrad(context, profile.gamma_gradient);
}

vector<sc_bv<128>> norm_gamma_wgrad_timing::serialize() {
    Strict(*this);
    work();
    return NpuBase::serialize();
}

void norm_gamma_wgrad_timing::deserialize(vector<sc_bv<128>> wire) {
    Strict(*this);
    NpuBase::deserialize(std::move(wire));
    work();
}
