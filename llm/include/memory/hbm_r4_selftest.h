#pragma once
#include <string>

int RunHbmR4SelfTest();
int RunHbmContentionExperiment(const std::string &port, int core_count,
                               int pairs_per_core, int bytes_per_access);
