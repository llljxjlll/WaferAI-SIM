#include "memory/hbm_byte_transport.h"
#include "memory/dram/Dcache.h"

#include <tlm>

#include <limits>
#include <stdexcept>

namespace sram {
namespace {

int CheckedLength(uint64_t size_bytes) {
    if (size_bytes == 0 ||
        size_bytes > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(
            "HBM byte transport size must fit a positive int");
    return static_cast<int>(size_bytes);
}

} // namespace

std::vector<uint8_t> CoreMemByteTransport::Read(uint64_t address,
                                                 uint64_t size_bytes) {
    MemMsg response =
        adapter_.Access(MemCommand::kRead, address, CheckedLength(size_bytes));
    if (response.payload.size() != size_bytes)
        throw std::runtime_error(
            "CoreMemAdapter returned an unexpected read payload length");
    return response.payload;
}

void CoreMemByteTransport::Write(
    uint64_t address, const std::vector<uint8_t> &payload,
    const std::vector<uint8_t> &byte_enable) {
    adapter_.Access(MemCommand::kWrite, address,
                    CheckedLength(payload.size()), payload, byte_enable);
}

LegacyPrivateByteTransport::LegacyPrivateByteTransport(DCache &dcache)
    : dcache_(dcache) {
    dcache_.EnableRealDataMode();
}

std::vector<uint8_t> LegacyPrivateByteTransport::Read(
    uint64_t address, uint64_t size_bytes) {
    constexpr uint64_t kBurstBytes = 32;
    std::vector<uint8_t> payload(size_bytes);
    for (uint64_t offset = 0; offset < size_bytes; offset += kBurstBytes) {
        const uint64_t bytes = std::min(kBurstBytes, size_bytes - offset);
        tlm::tlm_generic_payload trans;
        trans.set_command(tlm::TLM_READ_COMMAND);
        trans.set_address(address + offset);
        trans.set_data_ptr(payload.data() + offset);
        trans.set_data_length(CheckedLength(bytes));
        trans.set_streaming_width(CheckedLength(bytes));
        sc_time delay = SC_ZERO_TIME;
        dcache_.b_transport(trans, delay);
        if (trans.get_response_status() != tlm::TLM_OK_RESPONSE)
            throw std::runtime_error("legacy_private HBM read failed");
        if (delay != SC_ZERO_TIME) wait(delay);
        for (uint64_t i = 0; i < bytes; ++i) {
            const auto it = bytes_.find(address + offset + i);
            payload[offset + i] =
                it == bytes_.end() ? uint8_t{0} : it->second;
        }
    }
    return payload;
}

void LegacyPrivateByteTransport::Write(
    uint64_t address, const std::vector<uint8_t> &payload,
    const std::vector<uint8_t> &byte_enable) {
    constexpr uint64_t kBurstBytes = 32;
    if (!byte_enable.empty() && byte_enable.size() != payload.size())
        throw std::invalid_argument(
            "legacy_private byte-enable length does not match payload");
    for (uint64_t offset = 0; offset < payload.size(); offset += kBurstBytes) {
        const uint64_t bytes =
            std::min<uint64_t>(kBurstBytes, payload.size() - offset);
        tlm::tlm_generic_payload trans;
        trans.set_command(tlm::TLM_WRITE_COMMAND);
        trans.set_address(address + offset);
        trans.set_data_ptr(
            const_cast<unsigned char *>(payload.data() + offset));
        trans.set_data_length(CheckedLength(bytes));
        trans.set_streaming_width(CheckedLength(bytes));
        if (!byte_enable.empty()) {
            trans.set_byte_enable_ptr(
                const_cast<unsigned char *>(byte_enable.data() + offset));
            trans.set_byte_enable_length(CheckedLength(bytes));
        }
        sc_time delay = SC_ZERO_TIME;
        dcache_.b_transport(trans, delay);
        if (trans.get_response_status() != tlm::TLM_OK_RESPONSE)
            throw std::runtime_error("legacy_private HBM write failed");
        if (delay != SC_ZERO_TIME) wait(delay);
        for (uint64_t i = 0; i < bytes; ++i)
            if (byte_enable.empty() || byte_enable[offset + i] != 0)
                bytes_[address + offset + i] = payload[offset + i];
    }
}

} // namespace sram
