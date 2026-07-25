#pragma once

#include <cstdint>

struct DteAggregationConfig {
    bool enabled = false;
    uint32_t max_descriptors = 16;
    uint64_t max_payload_bytes = 4096;
    uint64_t timeout_cycles = 10;
    uint64_t address_block_bytes = 65536;
};

struct DteAggregationMetrics {
    uint64_t logical_descriptors = 0;
    uint64_t physical_transfers = 0;
    uint64_t coalesced_descriptors = 0;
    uint64_t launch_savings = 0;
    uint64_t useful_payload_bits = 0;
    uint64_t bus_capacity_bits = 0;

    // Cumulative useful-bit ratio across all physical transfers issued so far.
    uint64_t BandwidthUtilizationPpm() const;
};

void ValidateDteAggregationConfig(const DteAggregationConfig &config);
