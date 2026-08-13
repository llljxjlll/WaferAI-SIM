#pragma once

#include <cstdint>

// Stable identifiers for the simulator-internal Prim wire. These values are
// deliberately separate from the public ISA Opcode values: program artifacts
// are decoded and lowered before a PrimId ever reaches a WorkerCore.
enum class PrimId : uint8_t {
    INVALID = 0,

    ATTENTION_F = 1,
    BATCHNORM_F = 2,
    CONV_F = 3,
    DUMMY_P = 4,
    GATE_FORWARD = 5,
    GELU_F = 6,
    GEMM_RS_SWIZZLE = 7,
    LAYERNORM_F = 8,
    MATMUL_F = 9,
    MATMUL_F_MLA = 10,
    MAX_POOL = 11,
    MERGE_CONV = 12,
    MERGE_MATMUL = 13,
    PARSE_INPUT = 14,
    PARSE_OUTPUT = 15,
    RECV_GLOBAL_MEMORY = 16,
    RELU_F = 17,
    RESIDUAL_F = 18,
    RMSNORM_FORWARD = 19,
    ROPE_FORWARD = 20,
    SEND_GLOBAL_MEMORY = 21,
    SILU_FORWARD = 22,
    SPLIT_CONV = 23,
    SPLIT_MATMUL = 24,
    SWIGLU_FORWARD = 25,
    SWITCH_DATA = 26,

    ATTENTION_F_GPU = 27,
    ATTENTION_FORWARD_GPU_PD = 28,
    GELU_F_GPU = 29,
    LAYERNORM_F_GPU = 30,
    MATMUL_F_GPU = 31,
    MATMUL_FORWARD_GPU_PD = 32,
    RESIDUAL_F_GPU = 33,

    LOAD_EXPERT = 34,
    MATMUL_FORWARD_MOE = 35,

    CLEAR_SRAM = 36,
    COLLECTIVE_DATA = 37,
    COLLECTIVE = 38,
    DTE_ASYNC = 39,
    LEGACY_LOAD = 40,
    LSU_MEM = 41,
    RECV = 42,
    REDUCE_COMPUTE = 43,
    SEND = 44,
    SET_ADDR = 45,
    SET_BATCH = 46,
    SRAM_PIPELINE = 47,
    LEGACY_STORE = 48,

    ATTENTION_FORWARD_PD = 49,
    MATMUL_FORWARD_PD = 50,
    ROPE_FORWARD_PD = 51,

    // Strict-only internal lowering target for the public SRAM_BIND opcode.
    SRAM_BIND_ONESHOT = 52,

    // Strict-only internal lowering target for SRAM region lifecycle opcodes.
    SRAM_LIFECYCLE = 53,

    // Strict-only internal targets for public synchronization opcodes.
    GROUP_SYNC = 54,
    EVENT_CONTROL = 55,

    // Strict-only internal endpoints for external DTE_SEND/DTE_RECV.
    DTE_SEND_ENDPOINT = 56,
    DTE_RECV_ENDPOINT = 57,

    // Strict-only real-byte collective local copy/reduction.
    COLLECTIVE_DATA_V1 = 58,

    // Strict-only internal whole-artifact collective phase barrier.
    COLLECTIVE_PHASE_BARRIER_V1 = 59,

    // Strict-only internal whole-artifact collective launch descriptor.
    COLLECTIVE_LAUNCH_V1 = 60,
};

constexpr uint8_t PrimIdValue(PrimId id) {
    return static_cast<uint8_t>(id);
}

constexpr uint8_t kMaxAssignedPrimId =
    PrimIdValue(PrimId::COLLECTIVE_LAUNCH_V1);
