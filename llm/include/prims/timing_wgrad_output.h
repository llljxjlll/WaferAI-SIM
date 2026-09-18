#pragma once

#include "common/include.h"
#include "memory/sram/sram_access_unit.h"
#include "memory/sram/sram_storage.h"
#include "prims/weight_gradient_timing_prims.h"

#include <limits>
#include <stdexcept>
#include <vector>

// Timing WGRAD has no numerical kernel. Its FP32 payload is zero and must be
// produced at the compute task, after any previous owner of the SRAM span.
inline void MaterializeTimingWgrad(TaskCoreContext &context,
                                   const WeightGradBufferABI &gradient) {
    if (context.sram_storage == nullptr ||
        !context.sram_storage->payload_mode())
        return;
    if (context.sram_access == nullptr ||
        gradient.bytes > std::numeric_limits<size_t>::max())
        throw std::logic_error("timing WGRAD lacks program SRAM output");
    sram::Request write;
    write.initiator = sram::Initiator::kCompute;
    write.command = sram::Command::kWrite;
    write.address = gradient.sram_byte_address;
    write.size_bytes = gradient.bytes;
    write.payload = std::vector<uint8_t>(static_cast<size_t>(gradient.bytes));
    context.sram_access->Access(write);
}
