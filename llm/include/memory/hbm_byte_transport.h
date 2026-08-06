#pragma once

#include "memory/core_mem_adapter.h"
#include <cstdint>
#include <unordered_map>
#include <vector>

class DCache;

namespace sram {

class HbmByteTransport {
  public:
    virtual ~HbmByteTransport() = default;
    virtual std::vector<uint8_t> Read(uint64_t address,
                                      uint64_t size_bytes) = 0;
    virtual void Write(uint64_t address, const std::vector<uint8_t> &payload,
                       const std::vector<uint8_t> &byte_enable = {}) = 0;
};

class CoreMemByteTransport : public HbmByteTransport {
  public:
    explicit CoreMemByteTransport(CoreMemAdapter &adapter)
        : adapter_(adapter) {}

    std::vector<uint8_t> Read(uint64_t address,
                              uint64_t size_bytes) override;
    void Write(uint64_t address, const std::vector<uint8_t> &payload,
               const std::vector<uint8_t> &byte_enable = {}) override;

  private:
    CoreMemAdapter &adapter_;
};

class LegacyPrivateByteTransport : public HbmByteTransport {
  public:
    explicit LegacyPrivateByteTransport(DCache &dcache);

    std::vector<uint8_t> Read(uint64_t address,
                              uint64_t size_bytes) override;
    void Write(uint64_t address, const std::vector<uint8_t> &payload,
               const std::vector<uint8_t> &byte_enable = {}) override;

  private:
    DCache &dcache_;
    std::unordered_map<uint64_t, uint8_t> bytes_;
};

} // namespace sram
