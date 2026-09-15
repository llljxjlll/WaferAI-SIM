#pragma once

#include "prims/base.h"

#include <cstdint>
#include <vector>

// Typed, physical per-core SRAM spans. The program memory schedule owns
// allocation, transfers and initialization; these Prims charge timing only.
enum class WeightGradBufferDType : uint8_t { INT32, FP16, FP32 };

struct WeightGradBufferABI {
    uint32_t sram_byte_address = 0;
    uint64_t bytes = 0;
    WeightGradBufferDType dtype = WeightGradBufferDType::FP16;
};

struct EmbeddingTableWGradWork {
    WeightGradBufferABI indices;
    WeightGradBufferABI table;
    WeightGradBufferABI upstream;
    WeightGradBufferABI gradient;
    uint64_t rank_rows = 0;
    uint64_t selected_rows = 0;
    uint64_t unique_weight_rows = 0;
    uint64_t row_collisions = 0;
    uint64_t trace_vec_ops = 0;
    uint64_t scatter_vec_ops = 0;
    uint64_t fp32_accumulate_vec_ops = 0;
    uint64_t selected_weight_read_bytes = 0;
    uint64_t selected_upstream_read_bytes = 0;
    uint64_t fp32_read_modify_write_bytes = 0;
    std::vector<uint32_t> selected_weight_sram_addresses;
    std::vector<uint32_t> selected_gradient_sram_addresses;
};

// The discrete INT32 indices have no numerical dIndex. This Prim models
// source-indexed FP16 upstream accumulation into one FP32 table-gradient tile.
// Rank row tiles are at most 16 indices per strict Prim wire; longer sequences
// and arbitrary vocabulary sizes are composed from more tiles.
class embedding_table_wgrad_timing final : public NpuBase {
public:
    embedding_table_wgrad_timing();
    void initialize() override;
    void taskCore(TaskCoreContext &, string, u_int64_t &,
                  u_int64_t &, u_int64_t &, u_int64_t &) override;
    vector<sc_bv<128>> serialize() override;
    void deserialize(vector<sc_bv<128>>) override;
    EmbeddingTableWGradWork work() const;
};

struct NormGammaWGradWork {
    WeightGradBufferABI activation;
    WeightGradBufferABI upstream;
    WeightGradBufferABI gamma_gradient;
    uint64_t rank_rows = 0;
    uint64_t hidden = 0;
    uint64_t normalization_vec_ops = 0;
    uint64_t fp32_accumulate_vec_ops = 0;
    uint64_t sfu_ops = 0;
    uint64_t fp32_read_modify_write_bytes = 0;
};

// Each TP rank produces a local FP32 gamma gradient over its own ROWS. The
// cross-rank reduction is an external program step, never silently implied.
class norm_gamma_wgrad_timing final : public NpuBase {
public:
    norm_gamma_wgrad_timing();
    void initialize() override;
    void taskCore(TaskCoreContext &, string, u_int64_t &,
                  u_int64_t &, u_int64_t &, u_int64_t &) override;
    vector<sc_bv<128>> serialize() override;
    void deserialize(vector<sc_bv<128>>) override;
    NormGammaWGradWork work() const;
};
