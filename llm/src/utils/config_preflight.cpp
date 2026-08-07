#include "utils/config_preflight.h"
#include "nlohmann/json.hpp"
#include "monitor/workload_normalize.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using json = nlohmann::json;

json ReadJson(const std::string &path, const char *kind) {
    const std::filesystem::path input(path);
    if (!std::filesystem::exists(input))
        throw std::runtime_error(std::string(kind) +
                                 " config does not exist: " + path);
    if (!std::filesystem::is_regular_file(input))
        throw std::runtime_error(std::string(kind) +
                                 " config is not a regular file: " + path);
    std::ifstream stream(input);
    json value;
    try {
        stream >> value;
    } catch (const json::exception &error) {
        throw std::runtime_error(std::string(kind) +
                                 " config is invalid JSON: " + path +
                                 " (" + error.what() + ")");
    }
    if (!value.is_object())
        throw std::runtime_error(std::string(kind) +
                                 " config root must be an object: " + path);
    return value;
}

void Require(const json &value, const char *field, const char *kind,
             const std::string &path) {
    if (!value.contains(field))
        throw std::runtime_error(std::string(kind) + " config misses '" +
                                 field + "': " + path);
}

void ValidateMapping(const std::string &path) {
    const std::filesystem::path input(path);
    if (!std::filesystem::exists(input) ||
        !std::filesystem::is_regular_file(input))
        throw std::runtime_error("mapping config does not exist: " + path);
    std::ifstream stream(input);
    std::string line;
    size_t line_number = 0;
    while (std::getline(stream, line)) {
        ++line_number;
        if (line.empty())
            continue;
        std::istringstream row(line);
        int source = 0;
        int destination = 0;
        char colon = 0;
        if (!(row >> source >> colon >> destination) || colon != ':') {
            throw std::runtime_error(
                "mapping config line " + std::to_string(line_number) +
                " must use '<source>:<destination>': " + path);
        }
        row >> std::ws;
        if (!row.eof())
            throw std::runtime_error(
                "mapping config line " + std::to_string(line_number) +
                " has trailing data: " + path);
    }
}
} // namespace

void ValidateConfigInputs(const std::string &workload,
                          const std::string &hardware,
                          const std::string &simulation,
                          const std::string &mapping) {
    const json workload_json = ReadJson(workload, "workload");
    Require(workload_json, "chips", "workload", workload);
    const std::string workload_mode =
        workload_json.value("mode", std::string("dataflow"));
    if (workload_mode == "dataflow") {
        Require(workload_json, "vars", "workload", workload);
        Require(workload_json, "source", "workload", workload);
        ValidateWorkloadRendezvous(workload_json, 0);
    }

    const json hardware_json = ReadJson(hardware, "hardware");
    Require(hardware_json, "x", "hardware", hardware);
    Require(hardware_json, "cores", "hardware", hardware);

    (void)ReadJson(simulation, "simulation");
    ValidateMapping(mapping);
}
