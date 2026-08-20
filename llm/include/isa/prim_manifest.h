#pragma once

#include "isa/prim_id.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

enum class PrimVisibility : uint8_t {
    PUBLIC,
    INTERNAL,
};

enum class PrimLifecycle : uint8_t {
    STABLE,
    DEPRECATED,
    RESERVED,
    TOMBSTONE,
};

enum class PrimSupport : uint8_t {
    AVAILABLE,
    UNSUPPORTED,
    EXPERIMENTAL,
};

enum class PrimCategory : uint8_t {
    COMPUTE,
    COMMUNICATION,
    MEMORY,
    SYNCHRONIZATION,
    DYNAMIC,
};

struct PrimManifestEntry {
    PrimId id;
    std::string_view factory_name;
    PrimCategory primary_category;
    PrimVisibility visibility;
    PrimLifecycle lifecycle;
    PrimSupport support;
};

inline constexpr std::size_t kPrimManifestSize = 67;

const std::array<PrimManifestEntry, kPrimManifestSize> &
PrimManifest() noexcept;

const PrimManifestEntry *LookupPrim(uint16_t raw_id) noexcept;
const PrimManifestEntry *LookupPrim(std::string_view factory_name) noexcept;

inline const PrimManifestEntry *LookupPrim(PrimId id) noexcept {
    return LookupPrim(static_cast<uint16_t>(PrimIdValue(id)));
}

// Order-independent inventory validation, exposed so self-tests can exercise
// duplicate/INVALID cases without mutating the process-global PrimFactory.
bool ValidatePrimManifestEntries(
    const std::vector<PrimManifestEntry> &entries,
    std::string *error = nullptr);

// Validates the frozen table plus its required numeric ordering and continuity.
bool ValidatePrimManifest(std::string *error = nullptr);
