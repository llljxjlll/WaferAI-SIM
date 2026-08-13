#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct IsaV1SelfTestResult {
    std::size_t checks = 0;
    std::vector<std::string> failures;

    bool passed() const noexcept { return failures.empty(); }
};

// Pure, deterministic test entry point.  It has no simulator/SystemC state and
// performs no I/O, so it can be used by unit tests and startup diagnostics.
IsaV1SelfTestResult CheckIsaV1OpcodeManifest();

// Pure internal-Prim inventory checks, including negative duplicate/INVALID
// fixtures. This does not construct simulator primitives.
IsaV1SelfTestResult CheckIsaV1PrimManifest();

// Full-process integration checks against the statically registered factory.
// Call only after all production registration initializers have run.
IsaV1SelfTestResult CheckIsaV1PrimFactory();

// Conventional npusim self-test adapter.  It prints the pure test result and
// returns the number of failed checks (zero means success).
int RunIsaV1SelfTest();
