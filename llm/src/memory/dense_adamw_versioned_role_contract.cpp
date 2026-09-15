#include "memory/dense_adamw_versioned_role_contract.h"

#include <utility>

#include <stdexcept>

namespace external_memory {
namespace {
[[noreturn]] void Fail(const std::string &reason) {
    throw std::invalid_argument("Dense AdamW versioned role contract: " + reason);
}
} // namespace

DenseAdamwVersionedRoleContract::DenseAdamwVersionedRoleContract(
    std::map<std::string, ExpectedRole> expected)
    : expected_(std::move(expected)) {
    const std::map<std::string, std::size_t> roles{
        {"trainable_parameter", 15}, {"optimizer_master", 17},
        {"optimizer_moment1", 17}, {"optimizer_moment2", 17},
        {"optimizer_step", 17},
    };
    if (expected_.size() != roles.size()) Fail("five physical roles required");
    for (const auto &[role, count] : roles) {
        const auto found = expected_.find(role);
        if (found == expected_.end() || found->second.first != count ||
            found->second.second == 0)
            Fail("missing or malformed signed StateABI extent for " + role);
    }
    const auto bytes = [&](const char *role) {
        return expected_.at(role).second;
    };
    if (bytes("optimizer_master") != bytes("optimizer_moment1") ||
        bytes("optimizer_master") != bytes("optimizer_moment2") ||
        bytes("optimizer_master") != 2 * bytes("trainable_parameter") ||
        bytes("optimizer_step") != 17 * sizeof(uint32_t))
        Fail("FP32 master/m/v or INT32 step physical bytes drifted");
}

void DenseAdamwVersionedRoleContract::Probe(
    const std::map<std::string, std::vector<uint8_t>> &actual,
    uint64_t version, uint64_t pending) {
    if (version != next_version_ || version > 2)
        Fail("state version must advance exactly 0 to 1 to 2");
    if (pending != 0 || actual.size() != expected_.size())
        Fail("dirty DMA remains or a physical role probe is missing");
    for (const auto &[role, count_bytes] : expected_) {
        const auto found = actual.find(role);
        if (found == actual.end() || found->second.size() != count_bytes.second)
            Fail("real external probe bytes changed for " + role);
    }
    ++next_version_;
}

} // namespace external_memory
