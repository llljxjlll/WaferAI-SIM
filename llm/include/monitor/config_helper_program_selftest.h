#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct ConfigHelperProgramSelfTestResult {
    std::size_t checks = 0;
    std::vector<std::string> failures;

    bool passed() const noexcept { return failures.empty(); }
};

ConfigHelperProgramSelfTestResult CheckConfigHelperProgram();
int RunConfigHelperProgramSelfTest();
