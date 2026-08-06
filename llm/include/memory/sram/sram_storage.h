#pragma once

#include <cstdint>
#include <vector>

namespace sram {

class Storage {
  public:
    explicit Storage(uint64_t capacity_bytes, bool payload_mode = true);

    uint64_t capacity_bytes() const { return capacity_bytes_; }
    bool payload_mode() const { return payload_mode_; }

    void Write(uint64_t address, const std::vector<uint8_t> &payload,
               const std::vector<uint8_t> &byte_enable = {});
    std::vector<uint8_t> Read(uint64_t address, uint64_t size_bytes) const;
    void Clear(uint64_t address, uint64_t size_bytes);
    bool IsValid(uint64_t address, uint64_t size_bytes) const;
    uint64_t Signature(uint64_t address, uint64_t size_bytes) const;

  private:
    void CheckRange(uint64_t address, uint64_t size_bytes) const;

    uint64_t capacity_bytes_;
    bool payload_mode_;
    std::vector<uint8_t> bytes_;
    std::vector<uint8_t> fingerprints_;
    std::vector<uint8_t> valid_;
};

} // namespace sram
