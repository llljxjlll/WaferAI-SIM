#pragma once

#include <cstdint>

namespace coll_refactor {

// One canonical reduce-wire version is shared by configuration, contracts,
// and the future codec. Keep the numeric values stable once packets use it.
enum class ReduceWireVersion : uint8_t {
    LEGACY_TWO_SEGMENT = 0,
    STREAM_V2 = 1,
};

static_assert(static_cast<uint8_t>(ReduceWireVersion::LEGACY_TWO_SEGMENT) == 0,
              "legacy reduce-wire value is part of the frozen contract");
static_assert(static_cast<uint8_t>(ReduceWireVersion::STREAM_V2) == 1,
              "stream-v2 reduce-wire value is part of the frozen contract");

} // namespace coll_refactor

using NocCollReduceWire = coll_refactor::ReduceWireVersion;
