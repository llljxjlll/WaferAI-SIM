#pragma once

#include <cstdint>

// Msg::seq_id_ is u16 and each real endpoint DATA fragment carries 16 bytes.
inline constexpr uint64_t kDteEndpointP2pMaxBytes =
    uint64_t{UINT16_MAX} * 16;
