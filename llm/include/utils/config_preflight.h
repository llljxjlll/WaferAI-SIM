#pragma once

#include <string>

// Fail-fast validation for the four legacy JSON CLI configuration inputs. This runs before
// global simulator state and SystemC modules are constructed.
// Program mode validates only the shared platform inputs.
void ValidatePlatformConfigInputs(const std::string &hardware,
                                  const std::string &simulation,
                                  const std::string &mapping);

void ValidateConfigInputs(const std::string &workload,
                          const std::string &hardware,
                          const std::string &simulation,
                          const std::string &mapping);
