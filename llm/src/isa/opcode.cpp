#include "isa/opcode.h"

#include <algorithm>
#include <array>
#include <string>
#include <utility>

namespace {

using E = OpcodeManifestEntry;

constexpr std::array<E, kOpcodeManifestSize> kManifest{{
    {Opcode::MATMUL, "MATMUL", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::COMPUTE},
    {Opcode::MATMUL_MLA, "MATMUL_MLA", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::EXPERIMENTAL,
     OpcodeCategory::COMPUTE},
    {Opcode::MATMUL_PD, "MATMUL_PD", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::EXPERIMENTAL,
     OpcodeCategory::COMPUTE},
    {Opcode::CONV, "CONV", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::COMPUTE},
    {Opcode::MAXPOOL, "MAXPOOL", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::COMPUTE},
    {Opcode::ATTENTION, "ATTENTION", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::COMPUTE},
    {Opcode::ATTENTION_PD, "ATTENTION_PD", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::EXPERIMENTAL,
     OpcodeCategory::COMPUTE},
    {Opcode::GATE, "GATE", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::COMPUTE},
    {Opcode::MOE_MATMUL, "MOE_MATMUL", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::COMPUTE},
    {Opcode::GELU, "GELU", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::COMPUTE},
    {Opcode::SILU, "SILU", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::COMPUTE},
    {Opcode::SWIGLU, "SWIGLU", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::COMPUTE},
    {Opcode::RELU, "RELU", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::COMPUTE},
    {Opcode::RESIDUAL, "RESIDUAL", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::COMPUTE},
    {Opcode::LAYERNORM, "LAYERNORM", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::COMPUTE},
    {Opcode::RMSNORM, "RMSNORM", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::COMPUTE},
    {Opcode::ROPE, "ROPE", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::COMPUTE},
    {Opcode::ROPE_PD, "ROPE_PD", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::EXPERIMENTAL,
     OpcodeCategory::COMPUTE},
    {Opcode::SPLIT_MATMUL, "SPLIT_MATMUL", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::COMPUTE},
    {Opcode::MERGE_MATMUL, "MERGE_MATMUL", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::COMPUTE},
    {Opcode::DUMMY, "DUMMY", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::COMPUTE},
    {Opcode::BATCHNORM, "BATCHNORM", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::RESERVED, OpcodeSupport::UNSUPPORTED,
     OpcodeCategory::COMPUTE},
    {Opcode::SPLIT_CONV, "SPLIT_CONV", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::RESERVED, OpcodeSupport::UNSUPPORTED,
     OpcodeCategory::COMPUTE},
    {Opcode::MERGE_CONV, "MERGE_CONV", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::RESERVED, OpcodeSupport::UNSUPPORTED,
     OpcodeCategory::COMPUTE},
    // The experimental fused primitive owns an encoding for diagnostics, but
    // it is not a v1 executable instruction and may not be registered.
    {Opcode::GEMM_REDUCE_SCATTER, "GEMM_REDUCE_SCATTER",
     OpcodeVisibility::PUBLIC, OpcodeLifecycle::RESERVED,
     OpcodeSupport::EXPERIMENTAL, OpcodeCategory::COMPUTE},

    {Opcode::DTE_SEND, "DTE_SEND", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::COMMUNICATION},
    {Opcode::DTE_RECV, "DTE_RECV", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::COMMUNICATION},
    {Opcode::REDUCE_COMPUTE, "REDUCE_COMPUTE", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::COMMUNICATION},

    {Opcode::LSU_LOAD, "LSU_LOAD", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::MEMORY},
    {Opcode::LSU_STORE, "LSU_STORE", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::MEMORY},
    {Opcode::DTE_ISSUE, "DTE_ISSUE", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::MEMORY},
    {Opcode::SRAM_CLEAR, "SRAM_CLEAR", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::MEMORY},
    {Opcode::SRAM_BIND, "SRAM_BIND", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::MEMORY},
    {Opcode::SRAM_ALLOC, "SRAM_ALLOC", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::MEMORY},
    {Opcode::SRAM_FREE, "SRAM_FREE", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::MEMORY},
    {Opcode::SRAM_RESIZE, "SRAM_RESIZE", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::MEMORY},
    {Opcode::SRAM_RENAME, "SRAM_RENAME", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::MEMORY},

    {Opcode::DTE_WAIT, "DTE_WAIT", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::SYNCHRONIZATION},
    {Opcode::DTE_FENCE, "DTE_FENCE", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::SYNCHRONIZATION},
    {Opcode::DTE_CANCEL, "DTE_CANCEL", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::SYNCHRONIZATION},
    {Opcode::EVENT_SET, "EVENT_SET", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::SYNCHRONIZATION},
    {Opcode::EVENT_WAIT, "EVENT_WAIT", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::SYNCHRONIZATION},
    {Opcode::GROUP_SYNC, "GROUP_SYNC", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::STABLE, OpcodeSupport::AVAILABLE,
     OpcodeCategory::SYNCHRONIZATION},
    {Opcode::DTE_POLL, "DTE_POLL", OpcodeVisibility::PUBLIC,
     OpcodeLifecycle::RESERVED, OpcodeSupport::UNSUPPORTED,
     OpcodeCategory::SYNCHRONIZATION},
}};

bool InCategoryRange(uint8_t value, OpcodeCategory category) noexcept {
    switch (category) {
    case OpcodeCategory::COMPUTE:
        return value >= kComputeOpcodeFirst && value <= 0x3f;
    case OpcodeCategory::COMMUNICATION:
        return value >= 0x40 && value <= 0x7f;
    case OpcodeCategory::MEMORY:
        return value >= 0x80 && value <= 0xbf;
    case OpcodeCategory::SYNCHRONIZATION:
        return value >= 0xc0 && value <= 0xef;
    }
    return false;
}

bool SetError(std::string *error, std::string message) {
    if (error != nullptr)
        *error = std::move(message);
    return false;
}

} // namespace

const std::array<OpcodeManifestEntry, kOpcodeManifestSize> &
OpcodeManifest() noexcept {
    return kManifest;
}

const OpcodeManifestEntry *LookupOpcode(uint8_t value) noexcept {
    const auto it = std::lower_bound(
        kManifest.begin(), kManifest.end(), value,
        [](const OpcodeManifestEntry &entry, uint8_t candidate) {
            return OpcodeValue(entry.opcode) < candidate;
        });
    if (it == kManifest.end() || OpcodeValue(it->opcode) != value)
        return nullptr;
    return &*it;
}

const OpcodeManifestEntry *
LookupOpcode(std::string_view canonical_name) noexcept {
    const auto it = std::find_if(
        kManifest.begin(), kManifest.end(),
        [canonical_name](const OpcodeManifestEntry &entry) {
            return entry.canonical_name == canonical_name;
        });
    return it == kManifest.end() ? nullptr : &*it;
}

OpcodeValidation ValidateOpcode(
    uint8_t value, uint64_t enabled_capabilities) noexcept {
    const OpcodeManifestEntry *entry = LookupOpcode(value);
    if (entry == nullptr)
        return OpcodeValidation::RESERVED;
    if (entry->visibility == OpcodeVisibility::INTERNAL)
        return OpcodeValidation::INTERNAL;
    if (entry->support == OpcodeSupport::UNSUPPORTED &&
        entry->lifecycle != OpcodeLifecycle::TOMBSTONE)
        return OpcodeValidation::UNSUPPORTED;
    if (entry->lifecycle == OpcodeLifecycle::RESERVED ||
        entry->lifecycle == OpcodeLifecycle::TOMBSTONE)
        return OpcodeValidation::RESERVED;
    if (entry->support == OpcodeSupport::EXPERIMENTAL &&
        (enabled_capabilities & entry->required_capabilities) !=
            entry->required_capabilities)
        return OpcodeValidation::GATED;
    if (entry->lifecycle == OpcodeLifecycle::DEPRECATED)
        return OpcodeValidation::DEPRECATED;
    return OpcodeValidation::AVAILABLE;
}

OpcodeValidation ValidateOpcodeValue(
    uint16_t value, uint64_t enabled_capabilities) noexcept {
    if (value > UINT8_MAX)
        return OpcodeValidation::UNKNOWN;
    return ValidateOpcode(static_cast<uint8_t>(value),
                          enabled_capabilities);
}

bool ValidateOpcodeManifest(std::string *error) {
    if (error != nullptr)
        error->clear();

    for (std::size_t i = 0; i < kManifest.size(); ++i) {
        const OpcodeManifestEntry &entry = kManifest[i];
        const uint8_t value = OpcodeValue(entry.opcode);
        if (value == OpcodeValue(Opcode::INVALID))
            return SetError(error, "manifest contains INVALID opcode");
        if (entry.canonical_name.empty())
            return SetError(error, "manifest contains an empty canonical name");
        if (!InCategoryRange(value, entry.category))
            return SetError(error, "opcode category does not match numeric range: " +
                                       std::string(entry.canonical_name));
        if (i != 0 && OpcodeValue(kManifest[i - 1].opcode) >= value)
            return SetError(error,
                            "manifest opcodes are duplicate or out of order");
        for (std::size_t j = 0; j < i; ++j) {
            if (kManifest[j].canonical_name == entry.canonical_name)
                return SetError(error, "duplicate canonical opcode name: " +
                                           std::string(entry.canonical_name));
        }
        if (entry.lifecycle == OpcodeLifecycle::TOMBSTONE &&
            entry.support != OpcodeSupport::UNSUPPORTED) {
            return SetError(error,
                            "tombstone opcode must be unsupported: " +
                                std::string(entry.canonical_name));
        }
        if (entry.lifecycle == OpcodeLifecycle::RESERVED &&
            entry.support == OpcodeSupport::AVAILABLE) {
            return SetError(error, "reserved opcode cannot be available: " +
                                       std::string(entry.canonical_name));
        }
        if (entry.support == OpcodeSupport::EXPERIMENTAL &&
            entry.lifecycle != OpcodeLifecycle::STABLE &&
            entry.lifecycle != OpcodeLifecycle::RESERVED) {
            return SetError(error,
                            "experimental opcode has illegal lifecycle: " +
                                std::string(entry.canonical_name));
        }
        if ((entry.support == OpcodeSupport::EXPERIMENTAL) !=
            (entry.required_capabilities != 0))
            return SetError(error,
                            "opcode capability metadata is inconsistent: " +
                                std::string(entry.canonical_name));
        const OpcodeLowering expected = OpcodeLoweringFor(entry.opcode);
        if (entry.lowering.kind != expected.kind ||
            entry.lowering.target != expected.target ||
            entry.lowering.variant != expected.variant) {
            return SetError(error, "opcode has inconsistent lowering: " +
                                       std::string(entry.canonical_name));
        }
        const bool target_required =
            entry.lowering.kind == OpcodeLoweringKind::DIRECT_PRIM ||
            entry.lowering.kind == OpcodeLoweringKind::PRIM_VARIANT;
        if (target_required == (entry.lowering.target == PrimId::INVALID))
            return SetError(error, "opcode lowering target is illegal: " +
                                       std::string(entry.canonical_name));
        const bool variant_required =
            entry.lowering.kind == OpcodeLoweringKind::PRIM_VARIANT ||
            entry.lowering.kind == OpcodeLoweringKind::MODE_DISPATCH;
        if (variant_required ==
            (entry.lowering.variant == OpcodeLoweringVariant::NONE))
            return SetError(error, "opcode lowering variant is illegal: " +
                                       std::string(entry.canonical_name));
    }
    return true;
}
