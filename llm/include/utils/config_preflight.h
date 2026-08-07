#pragma once

#include <string>

// Fail-fast validation for the four CLI configuration inputs. This runs before
// global simulator state and SystemC modules are constructed.
void ValidateConfigInputs(const std::string &workload,
                          const std::string &hardware,
                          const std::string &simulation,
                          const std::string &mapping);
