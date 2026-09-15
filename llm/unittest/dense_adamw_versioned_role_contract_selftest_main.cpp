#include "memory/dense_adamw_versioned_role_contract.h"

#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using external_memory::DenseAdamwVersionedRoleContract;

namespace {
using Definition = DenseAdamwVersionedRoleContract::ExpectedRole;
using Roles = std::map<std::string, std::vector<uint8_t>>;

std::map<std::string, Definition> Expected() {
    return {{"trainable_parameter", {15, 64}},
            {"optimizer_master", {17, 128}},
            {"optimizer_moment1", {17, 128}},
            {"optimizer_moment2", {17, 128}},
            {"optimizer_step", {17, 68}}};
}

Roles Actual() {
    return {{"trainable_parameter", std::vector<uint8_t>(64, 1)},
            {"optimizer_master", std::vector<uint8_t>(128, 2)},
            {"optimizer_moment1", std::vector<uint8_t>(128, 3)},
            {"optimizer_moment2", std::vector<uint8_t>(128, 4)},
            {"optimizer_step", std::vector<uint8_t>(68, 5)}};
}

void Rejected(const std::function<void()> &work) {
    try {
        work();
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error("malformed AdamW role execution was admitted");
}
} // namespace

int main() {
    DenseAdamwVersionedRoleContract good(Expected());
    const auto actual = Actual();
    good.Probe(actual, 0, 0);
    good.Probe(actual, 1, 0); // timing mode may preserve all physical bytes
    good.Probe(actual, 2, 0);
    if (good.NextVersion() != 3)
        throw std::runtime_error("final AdamW state version did not advance");
    Rejected([&] { auto expected = Expected(); expected.erase("optimizer_moment2");
                   DenseAdamwVersionedRoleContract broken(expected); });
    Rejected([&] { auto expected = Expected(); expected["optimizer_step"].second = 64;
                   DenseAdamwVersionedRoleContract broken(expected); });
    Rejected([&] { DenseAdamwVersionedRoleContract broken(Expected());
                   broken.Probe(actual, 1, 0); });
    Rejected([&] { DenseAdamwVersionedRoleContract broken(Expected());
                   broken.Probe(actual, 0, 1); });
    Rejected([&] { DenseAdamwVersionedRoleContract broken(Expected());
                   auto missing = Actual(); missing.erase("optimizer_moment1");
                   broken.Probe(missing, 0, 0); });
    Rejected([&] { DenseAdamwVersionedRoleContract broken(Expected());
                   auto truncated = Actual(); truncated["optimizer_master"].pop_back();
                   broken.Probe(truncated, 0, 0); });
    return 0;
}
