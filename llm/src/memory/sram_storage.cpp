#include "memory/sram/sram_storage.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace sram {

Storage::Storage(uint64_t capacity_bytes, bool payload_mode)
    : capacity_bytes_(capacity_bytes), payload_mode_(payload_mode),
      bytes_(payload_mode ? static_cast<size_t>(capacity_bytes) : 0, 0),
      fingerprints_(!payload_mode ? static_cast<size_t>(capacity_bytes) : 0, 0),
      valid_(static_cast<size_t>(capacity_bytes), 0) {
    if (capacity_bytes == 0)
        throw std::invalid_argument("SRAM storage capacity must be non-zero");
    if (capacity_bytes > std::numeric_limits<size_t>::max())
        throw std::length_error("SRAM storage capacity exceeds host size_t");
}

void Storage::CheckRange(uint64_t address, uint64_t size_bytes) const {
    if (size_bytes > std::numeric_limits<uint64_t>::max() - address ||
        address + size_bytes > capacity_bytes_)
        throw std::out_of_range("SRAM storage access is out of bounds");
}

void Storage::Write(uint64_t address, const std::vector<uint8_t> &payload,
                    const std::vector<uint8_t> &byte_enable) {
    if (payload.empty())
        throw std::invalid_argument("SRAM write payload must not be empty");
    if (!byte_enable.empty() && byte_enable.size() != payload.size())
        throw std::invalid_argument(
            "SRAM byte-enable length must equal payload length");
    CheckRange(address, payload.size());
    for (size_t i = 0; i < payload.size(); ++i) {
        if (!byte_enable.empty() && byte_enable[i] == 0) continue;
        const size_t index = static_cast<size_t>(address) + i;
        if (payload_mode_)
            bytes_[index] = payload[i];
        else
            fingerprints_[index] = payload[i];
        valid_[index] = 1;
    }
}

std::vector<uint8_t> Storage::Read(uint64_t address,
                                   uint64_t size_bytes) const {
    if (size_bytes == 0)
        throw std::invalid_argument("SRAM read size must be non-zero");
    CheckRange(address, size_bytes);
    if (!IsValid(address, size_bytes))
        throw std::runtime_error("read from invalid SRAM byte range");
    std::vector<uint8_t> result(static_cast<size_t>(size_bytes), 0);
    if (payload_mode_)
        std::copy_n(bytes_.begin() + static_cast<size_t>(address),
                    static_cast<size_t>(size_bytes), result.begin());
    return result;
}

void Storage::Clear(uint64_t address, uint64_t size_bytes) {
    CheckRange(address, size_bytes);
    const auto begin = static_cast<size_t>(address);
    const auto end = begin + static_cast<size_t>(size_bytes);
    if (payload_mode_)
        std::fill(bytes_.begin() + begin, bytes_.begin() + end, 0);
    else
        std::fill(fingerprints_.begin() + begin,
                  fingerprints_.begin() + end, 0);
    std::fill(valid_.begin() + begin, valid_.begin() + end, 0);
}

bool Storage::IsValid(uint64_t address, uint64_t size_bytes) const {
    CheckRange(address, size_bytes);
    const auto begin = valid_.begin() + static_cast<size_t>(address);
    const auto end = begin + static_cast<size_t>(size_bytes);
    return std::all_of(begin, end, [](uint8_t value) { return value != 0; });
}

uint64_t Storage::Signature(uint64_t address, uint64_t size_bytes) const {
    CheckRange(address, size_bytes);
    if (!IsValid(address, size_bytes))
        throw std::runtime_error("signature requested for invalid SRAM range");
    uint64_t hash = 1469598103934665603ULL;
    for (uint64_t i = 0; i < size_bytes; ++i) {
        const uint8_t value =
            payload_mode_ ? bytes_[static_cast<size_t>(address + i)]
                          : fingerprints_[static_cast<size_t>(address + i)];
        hash ^= value;
        hash *= 1099511628211ULL;
    }
    return hash;
}


DebugSnapshot Storage::DebugPeek(uint64_t address,
                                 uint64_t size_bytes) const {
    if (size_bytes == 0)
        throw std::invalid_argument(
            "SRAM debug peek size must be non-zero");
    CheckRange(address, size_bytes);
    const size_t begin = static_cast<size_t>(address);
    const size_t count = static_cast<size_t>(size_bytes);
    DebugSnapshot snapshot;
    snapshot.payload.resize(count);
    const auto &source = payload_mode_ ? bytes_ : fingerprints_;
    std::copy_n(source.begin() + begin, count, snapshot.payload.begin());
    snapshot.valid.assign(valid_.begin() + begin,
                          valid_.begin() + begin + count);
    return snapshot;
}

void Storage::DebugRestore(uint64_t address,
                           const DebugSnapshot &snapshot) {
    if (snapshot.payload.empty() ||
        snapshot.payload.size() != snapshot.valid.size())
        throw std::invalid_argument(
            "SRAM debug snapshot payload/valid size mismatch");
    if (!std::all_of(snapshot.valid.begin(), snapshot.valid.end(),
                     [](uint8_t value) { return value <= 1; }))
        throw std::invalid_argument(
            "SRAM debug snapshot validity must be zero or one");
    CheckRange(address, snapshot.payload.size());
    const size_t begin = static_cast<size_t>(address);
    auto &target = payload_mode_ ? bytes_ : fingerprints_;
    std::copy(snapshot.payload.begin(), snapshot.payload.end(),
              target.begin() + begin);
    std::copy(snapshot.valid.begin(), snapshot.valid.end(),
              valid_.begin() + begin);
}

} // namespace sram
