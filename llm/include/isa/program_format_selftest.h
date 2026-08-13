#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct ProgramFormatSelfTestResult {
    std::size_t checks = 0;
    std::vector<std::string> failures;

    bool passed() const noexcept { return failures.empty(); }
};

ProgramFormatSelfTestResult CheckProgramFormatV1();
int RunProgramFormatV1SelfTest();
