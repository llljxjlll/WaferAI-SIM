#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct RecordCodecSelfTestResult {
    std::size_t checks = 0;
    std::vector<std::string> failures;

    bool passed() const noexcept { return failures.empty(); }
};

// Pure, independently callable checks for the external v1 record ABI. This
// header intentionally does not depend on the simulator CLI or SystemC.
RecordCodecSelfTestResult CheckIsaV1RecordCodec();

// Prints failures to stderr and returns 0 on success, 1 on failure. The main
// ISA selftest can call this without coupling the record codec to that CLI.
int RunIsaV1RecordCodecSelfTest();
