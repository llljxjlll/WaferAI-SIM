#pragma once

#include <cstdint>
#include <string>

// A source witness, not an executable record. X proves the saved forward
// geometry; dX = dY * W^T consumes only the physical W and dY SRAM operands.
enum class GemmInputDxDType : uint8_t { FP16, FP32 };

struct GemmInputDxSourceWitness {
    std::string forward_op_ref;
    std::string weight_state_ref;
    std::string weight_load_state_ref;
    uint64_t source_state_version = 0;
    uint64_t loaded_state_version = 0;
    uint64_t activation_rows = 0;
    uint64_t activation_hidden = 0;
    uint64_t activation_bytes = 0;
    GemmInputDxDType activation_dtype = GemmInputDxDType::FP16;
    uint64_t state_weight_hidden = 0;
    uint64_t state_weight_output = 0;
    uint64_t state_weight_bytes = 0;
    GemmInputDxDType state_weight_dtype = GemmInputDxDType::FP16;
};

struct GemmInputDxSramSpan {
    uint32_t byte_address = 0;
    uint64_t bytes = 0;
    GemmInputDxDType dtype = GemmInputDxDType::FP16;
};

struct GemmInputDxTimingTile {
    uint64_t k = 0; // source rank rows
    uint64_t m = 0; // source hidden input width
    uint64_t n = 0; // source weight output width
    GemmInputDxSramSpan weight;   // FP16 W[M,N], explicit StateABI LOAD result
    GemmInputDxSramSpan upstream; // FP16 dY[K,N]
    GemmInputDxSramSpan output;   // FP32 dX[K,M]
};

struct GemmInputDxTimingWork {
    GemmInputDxTimingTile tile;
    uint64_t fp16_weight_read_bytes = 0;
    uint64_t fp16_upstream_read_bytes = 0;
    uint64_t fp32_dx_read_modify_write_bytes = 0;
    uint64_t fma_ops = 0;
    uint64_t exu_flops = 0;
    uint64_t fp32_output_vec_ops = 0;
};

// Stage R7 NEW-only source/physical timing contract. Public PrimId, opcode,
// fixed record codec and authoritative ProgramIO bridge stay fail closed until
// the frontend source-binding freeze ends. No numerical dX is calculated.
GemmInputDxTimingWork BuildGemmInputDxTimingWork(
    const GemmInputDxSourceWitness &source,
    const GemmInputDxTimingTile &tile);
