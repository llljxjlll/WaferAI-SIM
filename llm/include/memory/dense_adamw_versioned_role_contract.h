#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace external_memory {

// The source parser and pager bind these expected extents to actual StateABI
// spans.  Probe() is called only with real ExternalMemoryRuntimeBridge bytes.
class DenseAdamwVersionedRoleContract {
public:
    using ExpectedRole = std::pair<std::size_t, std::size_t>; // states, bytes
    explicit DenseAdamwVersionedRoleContract(
        std::map<std::string, ExpectedRole> expected);

    void Probe(const std::map<std::string, std::vector<uint8_t>> &actual,
               uint64_t version, uint64_t pending);
    uint64_t NextVersion() const { return next_version_; }
    const std::map<std::string, ExpectedRole> &Expected() const { return expected_; }

private:
    std::map<std::string, ExpectedRole> expected_;
    uint64_t next_version_ = 0;
};

} // namespace external_memory
