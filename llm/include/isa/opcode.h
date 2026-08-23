#pragma once

#include "isa/prim_id.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

// Stable opcode values used by the external v1 program artifact.  These are
// deliberately independent of the simulator-internal PrimId values: a loader
// must validate and lower an Opcode before constructing an internal Prim wire.
enum class Opcode : uint8_t {
    INVALID = 0x00,

    MATMUL = 0x01,
    MATMUL_MLA = 0x02,
    MATMUL_PD = 0x03,
    CONV = 0x04,
    MAXPOOL = 0x05,
    ATTENTION = 0x06,
    ATTENTION_PD = 0x07,
    GATE = 0x08,
    MOE_MATMUL = 0x09,
    GELU = 0x0a,
    SILU = 0x0b,
    SWIGLU = 0x0c,
    RELU = 0x0d,
    RESIDUAL = 0x0e,
    LAYERNORM = 0x0f,
    RMSNORM = 0x10,
    ROPE = 0x11,
    ROPE_PD = 0x12,
    SPLIT_MATMUL = 0x13,
    MERGE_MATMUL = 0x14,
    DUMMY = 0x15,
    BATCHNORM = 0x16,
    SPLIT_CONV = 0x17,
    MERGE_CONV = 0x18,
    GEMM_REDUCE_SCATTER = 0x19,
    ROPE_QK_EXACT = 0x1a,
    ATTENTION_EXACT = 0x1b,
    EMBEDDING_LOOKUP = 0x1c,
    GREEDY_SAMPLE = 0x1d,
    CROSS_ENTROPY_FORWARD = 0x1e,
    CROSS_ENTROPY_BACKWARD = 0x1f,
    SGD_UPDATE = 0x20,

    DTE_SEND = 0x40,
    DTE_RECV = 0x41,
    REDUCE_COMPUTE = 0x42,
    LOCAL_REDUCE = 0x43,
    LOCAL_NOC_SEND = 0x44,
    LOCAL_NOC_RECV = 0x45,
    LOCAL_NOC_WAIT = 0x46,

    LSU_LOAD = 0x80,
    LSU_STORE = 0x81,
    DTE_ISSUE = 0x82,
    SRAM_CLEAR = 0x83,
    SRAM_BIND = 0x84,
    SRAM_ALLOC = 0x85,
    SRAM_FREE = 0x86,
    SRAM_RESIZE = 0x87,
    SRAM_RENAME = 0x88,
    SRAM_ALLOC_AT = 0x89,

    DTE_WAIT = 0xc0,
    DTE_FENCE = 0xc1,
    DTE_CANCEL = 0xc2,
    EVENT_SET = 0xc3,
    EVENT_WAIT = 0xc4,
    GROUP_SYNC = 0xc5,
    DTE_POLL = 0xc6,
};

enum class OpcodeVisibility : uint8_t {
    PUBLIC,
    INTERNAL,
};

enum class OpcodeLifecycle : uint8_t {
    STABLE,
    DEPRECATED,
    RESERVED,
    TOMBSTONE,
};

enum class OpcodeSupport : uint8_t {
    AVAILABLE,
    UNSUPPORTED,
    EXPERIMENTAL,
};

enum class IsaCapability : uint64_t {
    NONE = 0,
    PD_CONTEXT = uint64_t{1} << 0,
    EXPERIMENTAL_FUSED = uint64_t{1} << 1,
};

constexpr uint64_t CapabilityBit(IsaCapability capability) noexcept {
    return static_cast<uint64_t>(capability);
}

constexpr uint64_t OpcodeRequiredCapabilitiesFor(Opcode opcode) noexcept {
    switch (opcode) {
    case Opcode::MATMUL_MLA:
    case Opcode::MATMUL_PD:
    case Opcode::ATTENTION_PD:
    case Opcode::ROPE_PD:
        return CapabilityBit(IsaCapability::PD_CONTEXT);
    case Opcode::GEMM_REDUCE_SCATTER:
        return CapabilityBit(IsaCapability::EXPERIMENTAL_FUSED);
    default:
        return 0;
    }
}

enum class OpcodeCategory : uint8_t {
    COMPUTE,
    COMMUNICATION,
    MEMORY,
    SYNCHRONIZATION,
};

enum class OpcodeLoweringKind : uint8_t {
    DIRECT_PRIM,
    PRIM_VARIANT,
    MODE_DISPATCH,
    NEW_THIN_PRIM,
};

enum class OpcodeLoweringVariant : uint8_t {
    NONE,
    LSU_LOAD_BLOCKING,
    LSU_STORE_BLOCKING,
    DTE_ISSUE,
    DTE_WAIT,
    DTE_FENCE,
    DTE_CANCEL,
    DTE_POLL,
    DTE_SEND_MODE,
    DTE_RECV_MODE,
};

struct OpcodeLowering {
    OpcodeLoweringKind kind;
    PrimId target;
    OpcodeLoweringVariant variant;
};

constexpr OpcodeLowering DirectPrim(PrimId target) noexcept {
    return {OpcodeLoweringKind::DIRECT_PRIM, target,
            OpcodeLoweringVariant::NONE};
}

constexpr OpcodeLowering PrimVariant(
    PrimId target, OpcodeLoweringVariant variant) noexcept {
    return {OpcodeLoweringKind::PRIM_VARIANT, target, variant};
}

constexpr OpcodeLowering NewThinPrim() noexcept {
    return {OpcodeLoweringKind::NEW_THIN_PRIM, PrimId::INVALID,
            OpcodeLoweringVariant::NONE};
}

constexpr OpcodeLowering ModeDispatch(
    OpcodeLoweringVariant variant) noexcept {
    return {OpcodeLoweringKind::MODE_DISPATCH, PrimId::INVALID, variant};
}

// Explicit external Opcode -> internal lowering contract. Unsupported and
// reserved entries retain a mapping for stable diagnostics, but validation
// must succeed before a caller uses it.
constexpr OpcodeLowering OpcodeLoweringFor(Opcode opcode) noexcept {
    switch (opcode) {
    case Opcode::MATMUL: return DirectPrim(PrimId::MATMUL_F);
    case Opcode::MATMUL_MLA: return DirectPrim(PrimId::MATMUL_F_MLA);
    case Opcode::MATMUL_PD: return DirectPrim(PrimId::MATMUL_FORWARD_PD);
    case Opcode::CONV: return DirectPrim(PrimId::CONV_F);
    case Opcode::MAXPOOL: return DirectPrim(PrimId::MAX_POOL);
    case Opcode::ATTENTION: return DirectPrim(PrimId::ATTENTION_F);
    case Opcode::ATTENTION_PD:
        return DirectPrim(PrimId::ATTENTION_FORWARD_PD);
    case Opcode::GATE: return DirectPrim(PrimId::GATE_FORWARD);
    case Opcode::MOE_MATMUL: return DirectPrim(PrimId::MATMUL_FORWARD_MOE);
    case Opcode::GELU: return DirectPrim(PrimId::GELU_F);
    case Opcode::SILU: return DirectPrim(PrimId::SILU_FORWARD);
    case Opcode::SWIGLU: return DirectPrim(PrimId::SWIGLU_FORWARD);
    case Opcode::RELU: return DirectPrim(PrimId::RELU_F);
    case Opcode::RESIDUAL: return DirectPrim(PrimId::RESIDUAL_F);
    case Opcode::LAYERNORM: return DirectPrim(PrimId::LAYERNORM_F);
    case Opcode::RMSNORM: return DirectPrim(PrimId::RMSNORM_FORWARD);
    case Opcode::ROPE: return DirectPrim(PrimId::ROPE_FORWARD);
    case Opcode::ROPE_PD: return DirectPrim(PrimId::ROPE_FORWARD_PD);
    case Opcode::SPLIT_MATMUL: return DirectPrim(PrimId::SPLIT_MATMUL);
    case Opcode::MERGE_MATMUL: return DirectPrim(PrimId::MERGE_MATMUL);
    case Opcode::DUMMY: return DirectPrim(PrimId::DUMMY_P);
    case Opcode::BATCHNORM: return DirectPrim(PrimId::BATCHNORM_F);
    case Opcode::SPLIT_CONV: return DirectPrim(PrimId::SPLIT_CONV);
    case Opcode::MERGE_CONV: return DirectPrim(PrimId::MERGE_CONV);
    case Opcode::GEMM_REDUCE_SCATTER:
        return DirectPrim(PrimId::GEMM_RS_SWIZZLE);
    case Opcode::ROPE_QK_EXACT:
        return DirectPrim(PrimId::ROPE_QK_EXACT);
    case Opcode::ATTENTION_EXACT:
        return DirectPrim(PrimId::ATTENTION_EXACT);
    case Opcode::EMBEDDING_LOOKUP:
        return DirectPrim(PrimId::EMBEDDING_LOOKUP);
    case Opcode::GREEDY_SAMPLE:
        return DirectPrim(PrimId::GREEDY_SAMPLE);
    case Opcode::CROSS_ENTROPY_FORWARD:
        return DirectPrim(PrimId::CROSS_ENTROPY_FORWARD);
    case Opcode::CROSS_ENTROPY_BACKWARD:
        return DirectPrim(PrimId::CROSS_ENTROPY_BACKWARD);
    case Opcode::SGD_UPDATE:
        return DirectPrim(PrimId::SGD_UPDATE);
    case Opcode::DTE_SEND:
        return ModeDispatch(OpcodeLoweringVariant::DTE_SEND_MODE);
    case Opcode::DTE_RECV:
        return ModeDispatch(OpcodeLoweringVariant::DTE_RECV_MODE);
    case Opcode::REDUCE_COMPUTE:
        return DirectPrim(PrimId::REDUCE_COMPUTE);
    case Opcode::LOCAL_REDUCE:
        return DirectPrim(PrimId::COLLECTIVE_DATA_V1);
    case Opcode::LOCAL_NOC_SEND:
    case Opcode::LOCAL_NOC_RECV:
    case Opcode::LOCAL_NOC_WAIT:
        return NewThinPrim();
    case Opcode::LSU_LOAD:
        return PrimVariant(PrimId::LSU_MEM,
                           OpcodeLoweringVariant::LSU_LOAD_BLOCKING);
    case Opcode::LSU_STORE:
        return PrimVariant(PrimId::LSU_MEM,
                           OpcodeLoweringVariant::LSU_STORE_BLOCKING);
    case Opcode::DTE_ISSUE:
        return PrimVariant(PrimId::DTE_ASYNC,
                           OpcodeLoweringVariant::DTE_ISSUE);
    case Opcode::SRAM_CLEAR: return NewThinPrim();
    case Opcode::SRAM_BIND: return NewThinPrim();
    case Opcode::SRAM_ALLOC:
    case Opcode::SRAM_ALLOC_AT:
    case Opcode::SRAM_FREE:
    case Opcode::SRAM_RESIZE:
    case Opcode::SRAM_RENAME:
        return NewThinPrim();
    case Opcode::DTE_WAIT:
        return PrimVariant(PrimId::DTE_ASYNC,
                           OpcodeLoweringVariant::DTE_WAIT);
    case Opcode::DTE_FENCE:
        return PrimVariant(PrimId::DTE_ASYNC,
                           OpcodeLoweringVariant::DTE_FENCE);
    case Opcode::DTE_CANCEL:
        return PrimVariant(PrimId::DTE_ASYNC,
                           OpcodeLoweringVariant::DTE_CANCEL);
    case Opcode::EVENT_SET:
    case Opcode::EVENT_WAIT:
    case Opcode::GROUP_SYNC:
        return NewThinPrim();
    case Opcode::DTE_POLL:
        return PrimVariant(PrimId::DTE_ASYNC,
                           OpcodeLoweringVariant::DTE_POLL);
    case Opcode::INVALID:
        return NewThinPrim();
    }
    return NewThinPrim();
}

struct OpcodeManifestEntry {
    Opcode opcode;
    std::string_view canonical_name;
    OpcodeVisibility visibility;
    OpcodeLifecycle lifecycle;
    OpcodeSupport support;
    OpcodeCategory category;
    OpcodeLowering lowering;
    uint64_t required_capabilities;

    constexpr OpcodeManifestEntry(Opcode opcode_value,
                                  std::string_view name_value,
                                  OpcodeVisibility visibility_value,
                                  OpcodeLifecycle lifecycle_value,
                                  OpcodeSupport support_value,
                                  OpcodeCategory category_value) noexcept
        : opcode(opcode_value), canonical_name(name_value),
          visibility(visibility_value), lifecycle(lifecycle_value),
          support(support_value), category(category_value),
          lowering(OpcodeLoweringFor(opcode_value)),
          required_capabilities(
              OpcodeRequiredCapabilitiesFor(opcode_value)) {}
};

// The result of validating an opcode for use in an external v1 artifact.
// LookupOpcode remains available when the result is not AVAILABLE so callers
// can issue a precise diagnostic instead of treating a known opcode as unknown.
enum class OpcodeValidation : uint8_t {
    AVAILABLE,
    DEPRECATED,
    GATED,
    UNSUPPORTED,
    RESERVED,
    INTERNAL,
    UNKNOWN,
};

constexpr uint8_t OpcodeValue(Opcode opcode) noexcept {
    return static_cast<uint8_t>(opcode);
}

inline constexpr uint8_t kComputeOpcodeFirst = 0x01;
inline constexpr uint8_t kComputeOpcodeLast = 0x20;
inline constexpr uint8_t kCommunicationOpcodeFirst = 0x40;
inline constexpr uint8_t kCommunicationOpcodeLast = 0x43;
inline constexpr uint8_t kMemoryOpcodeFirst = 0x80;
inline constexpr uint8_t kMemoryOpcodeLast = 0x89;
inline constexpr uint8_t kSynchronizationOpcodeFirst = 0xc0;
inline constexpr uint8_t kSynchronizationOpcodeLast = 0xc6;
inline constexpr uint8_t kReservedOpcodeFirst = 0xf0;
inline constexpr uint8_t kReservedOpcodeLast = 0xff;
inline constexpr std::size_t kOpcodeManifestSize = 56;

// The returned array is sorted by numeric opcode and has static lifetime.
const std::array<OpcodeManifestEntry, kOpcodeManifestSize> &
OpcodeManifest() noexcept;

const OpcodeManifestEntry *LookupOpcode(uint8_t value) noexcept;
const OpcodeManifestEntry *LookupOpcode(std::string_view canonical_name) noexcept;

inline const OpcodeManifestEntry *LookupOpcode(Opcode opcode) noexcept {
    return LookupOpcode(OpcodeValue(opcode));
}

OpcodeValidation ValidateOpcode(
    uint8_t value, uint64_t enabled_capabilities = 0) noexcept;
OpcodeValidation ValidateOpcodeValue(
    uint16_t value, uint64_t enabled_capabilities = 0) noexcept;

inline OpcodeValidation ValidateOpcode(Opcode opcode) noexcept {
    return ValidateOpcode(OpcodeValue(opcode), 0);
}

// Validates ordering, uniqueness, category ranges and legal metadata
// combinations.  This function has no global side effects and is suitable for
// startup checks as well as isolated unit tests.
bool ValidateOpcodeManifest(std::string *error = nullptr);
