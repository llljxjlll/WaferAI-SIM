#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

// V2b flow-level closed-form tail helpers. Times are absolute nanoseconds.
// The source tail is projected through the observed first-unit network latency;
// final completion is the maximum of source, network, and destination-DTE tails.
inline uint64_t ProjectDteSourceTailToDestinationNs(
    uint64_t source_first_ns, uint64_t source_done_ns,
    uint64_t destination_first_ns) {
    if (destination_first_ns < source_first_ns)
        throw std::invalid_argument(
            "DTE V2b destination first unit precedes source first unit");
    const uint64_t first_latency_ns =
        destination_first_ns - source_first_ns;
    const uint64_t source_tail_ns =
        std::max(source_first_ns, source_done_ns);
    if (source_tail_ns >
        std::numeric_limits<uint64_t>::max() - first_latency_ns)
        throw std::overflow_error(
            "DTE V2b source-tail projection overflows");
    return source_tail_ns + first_latency_ns;
}

inline uint64_t CombineDteStreamingTailsNs(
    uint64_t source_tail_at_destination_ns, uint64_t network_tail_ns,
    uint64_t destination_dte_done_ns, uint64_t destination_drain_ns) {
    if (destination_drain_ns == 0)
        throw std::invalid_argument(
            "DTE V2b destination drain must be positive");
    const uint64_t upstream_tail_ns =
        std::max(source_tail_at_destination_ns, network_tail_ns);
    if (upstream_tail_ns >
        std::numeric_limits<uint64_t>::max() - destination_drain_ns)
        throw std::overflow_error(
            "DTE V2b destination drain time overflows");
    return std::max(destination_dte_done_ns,
                    upstream_tail_ns + destination_drain_ns);
}
