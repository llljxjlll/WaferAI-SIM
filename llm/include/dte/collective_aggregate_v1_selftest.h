#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct CollectiveAggregateV1SelfTestResult {
    size_t checks = 0;
    std::vector<std::string> failures;

    bool passed() const noexcept { return failures.empty(); }
};

CollectiveAggregateV1SelfTestResult
CheckCollectiveAggregateV1Runtime();
int RunCollectiveAggregateV1SelfTest();
