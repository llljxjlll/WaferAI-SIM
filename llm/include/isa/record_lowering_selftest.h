#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct RecordLoweringSelfTestResult {
    std::size_t checks = 0;
    std::vector<std::string> failures;

    bool passed() const noexcept { return failures.empty(); }
};

RecordLoweringSelfTestResult CheckIsaV1RecordLowering();
int RunIsaV1RecordLoweringSelfTest();
