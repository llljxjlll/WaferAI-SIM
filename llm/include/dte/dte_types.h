#pragma once

#include "macros/macros.h"
#include "systemc.h"

#include <array>
#include <cstdint>

enum class DteDir {
    SPM_TO_REMOTE,
    REMOTE_TO_SPM,
    SPM_TO_SPM,
    SPM_TO_DRAM,
    DRAM_TO_SPM,
    DRAM_TO_REMOTE
};

enum class DteTransferState {
    PENDING,
    LAUNCHING,
    BUS_WAIT,
    TRANSMITTING,
    COMPLETED,
    CANCELLED
};

enum class DtePort : uint8_t {
    LEGACY_BUS = 0,
    SPM_READ = 1,
    SPM_WRITE = 2,
    AXI_READ = 3,
    AXI_WRITE = 4,
    COUNT = 5
};

constexpr uint32_t DtePortBit(DtePort port) {
    return uint32_t(1) << static_cast<uint8_t>(port);
}

inline const char *DtePortName(DtePort port) {
    switch (port) {
    case DtePort::LEGACY_BUS: return "LEGACY_BUS";
    case DtePort::SPM_READ: return "SPM_READ";
    case DtePort::SPM_WRITE: return "SPM_WRITE";
    case DtePort::AXI_READ: return "AXI_READ";
    case DtePort::AXI_WRITE: return "AXI_WRITE";
    case DtePort::COUNT: break;
    }
    return "UNKNOWN";
}

struct DTEConfig {
    uint32_t channel_count = 2;
    uint32_t bit_width_bits = 2048; // V0-V3 shared aggregate bus width
    // COMET p.8 §VI-A evaluation defaults inherited from reference [38].
    uint64_t gamma_cycles = (40000 + CYCLE - 1) / CYCLE;
    uint64_t tau_launch_cycles = (2000 + CYCLE - 1) / CYCLE;

    // V4 resources. Disabled by default to preserve every V0-V3 timing.
    bool fine_grained_resources = false;
    uint32_t command_slots_per_channel = 1;
    uint32_t pending_queue_depth = 0; // 0 means unbounded in legacy mode
    uint32_t spm_read_width_bits = 2048;
    uint32_t spm_write_width_bits = 2048;
    uint32_t axi_read_width_bits = 2048;
    uint32_t axi_write_width_bits = 2048;

    // Parameterized accounting coefficients. Energy is pJ, area is um^2.
    double launch_energy_pj = 0.0;
    double spm_energy_pj_per_bit = 0.0;
    double axi_energy_pj_per_bit = 0.0;
    double base_area_um2 = 0.0;
    double channel_area_um2 = 0.0;
    double command_slot_area_um2 = 0.0;
    double port_bit_area_um2 = 0.0;
};

struct DteStatistics {
    uint64_t physical_issued = 0;
    uint64_t completed = 0;
    uint64_t cancelled = 0;
    uint64_t backpressure_stalls = 0;
    double launch_energy_pj = 0.0;
    double data_energy_pj = 0.0;
    double area_um2 = 0.0;

    double TotalDynamicEnergyPj() const {
        return launch_energy_pj + data_energy_pj;
    }
};

struct DteTransferContext {
    uint64_t xfer_id = 0;
    uint64_t payload_bits = 0;
    DteDir dir = DteDir::SPM_TO_REMOTE;
    DteTransferState state = DteTransferState::PENDING;
    int channel_id = -1;
    int command_slot = -1;
    int active_slot = -1;
    uint32_t resource_mask = 0;
    uint32_t active_resource_mask = 0;
    std::array<sc_time, static_cast<size_t>(DtePort::COUNT)>
        resource_done_times{};

    sc_event transmit_started;
    sc_event done;
    sc_time issue_time = SC_ZERO_TIME;
    sc_time admitted_time = SC_ZERO_TIME;
    sc_time launch_done_time = SC_ZERO_TIME;
    sc_time transmit_start_time = SC_ZERO_TIME;
    sc_time scheduled_completion_time = SC_ZERO_TIME;
    sc_time completion_time = SC_ZERO_TIME;
};

inline const char *DteDirName(DteDir dir) {
    switch (dir) {
    case DteDir::SPM_TO_REMOTE: return "SPM_TO_REMOTE";
    case DteDir::REMOTE_TO_SPM: return "REMOTE_TO_SPM";
    case DteDir::SPM_TO_SPM: return "SPM_TO_SPM";
    case DteDir::SPM_TO_DRAM: return "SPM_TO_DRAM";
    case DteDir::DRAM_TO_SPM: return "DRAM_TO_SPM";
    case DteDir::DRAM_TO_REMOTE: return "DRAM_TO_REMOTE";
    }
    return "UNKNOWN";
}
