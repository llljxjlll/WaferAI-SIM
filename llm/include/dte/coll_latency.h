#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

inline uint64_t CollCeilDiv(uint64_t value, uint64_t divisor) {
    if (divisor == 0) throw std::invalid_argument("collective bandwidth must be positive");
    return value / divisor + (value % divisor != 0);
}

inline uint64_t CollCheckedAdd(uint64_t a, uint64_t b) {
    if (a > std::numeric_limits<uint64_t>::max() - b)
        throw std::overflow_error("collective latency overflow");
    return a + b;
}

inline uint64_t CollCheckedMul(uint64_t a, uint64_t b) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
        throw std::overflow_error("collective payload overflow");
    return a * b;
}

inline uint64_t CollUnicastCycles(uint64_t payload_bits,
                                  uint64_t bandwidth_bits_per_cycle,
                                  uint64_t network_cycles) {
    return CollCheckedAdd(CollCeilDiv(payload_bits, bandwidth_bits_per_cycle),
                          network_cycles);
}

inline uint64_t CollTier0BroadcastCycles(uint64_t ranks, uint64_t message_bits,
                                         uint64_t bandwidth_bits_per_cycle,
                                         uint64_t network_cycles) {
    if (ranks == 0) throw std::invalid_argument("broadcast requires at least one rank");
    return CollCheckedAdd(CollCeilDiv(CollCheckedMul(ranks - 1, message_bits),
                                     bandwidth_bits_per_cycle), network_cycles);
}

inline uint64_t CollTier0ScatterCycles(uint64_t ranks, uint64_t message_bits,
                                       uint64_t bandwidth_bits_per_cycle,
                                       uint64_t network_cycles) {
    if (ranks == 0) throw std::invalid_argument("scatter requires at least one rank");
    return CollCheckedAdd(CollCeilDiv(message_bits, bandwidth_bits_per_cycle),
                          network_cycles);
}

inline uint64_t CollGatherEndpointCycles(uint64_t commit_cycles = 1) {
    return commit_cycles;
}
inline uint64_t CollReduceEndpointCycles(uint64_t alignment_cycles = 1) {
    return alignment_cycles;
}

inline uint64_t CollDcaServiceCycles(uint64_t payload_bits,
                                     uint64_t compute_cycles,
                                     uint64_t bits_per_cycle = 128,
                                     uint64_t pipeline_cycles = 54) {
    const uint64_t transfer = CollCeilDiv(payload_bits, bits_per_cycle);
    return CollCheckedAdd(compute_cycles > transfer ? compute_cycles : transfer,
                          pipeline_cycles);
}
