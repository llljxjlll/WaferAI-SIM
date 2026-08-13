#include "prims/base.h"
#include "utils/config_utils.h"
#include "utils/prim_utils.h"
#include "utils/print_utils.h"
#include "utils/system_utils.h"

#include <limits>
#include <stdexcept>
#include <utility>

vector<sc_bv<128>> GpuBase::serialize() {
    LOG_DEBUG(CONFIG_DEBUG) << "Start serialize " << name;

    vector<sc_bv<128>> segments;
    if (datatype != INT8 && datatype != FP16)
        throw std::invalid_argument(name + " has an invalid datatype");
    if (fetch_index < 0 || req_sm < 0)
        throw std::overflow_error(
            name + " GPU index fields must be non-negative");
    for (const auto &entry : param_value)
        if (entry.second < 0 || entry.second > 0x3fffffff)
            throw std::overflow_error(name + " parameter " + entry.first +
                                      " exceeds the 30-bit Prim wire");


    // metadata
    sc_bv<128> metadata = 0;
    metadata.range(7, 0) = sc_bv<8>(PrimFactory::getInstance().getPrimId(name));
    metadata.range(9, 8) = sc_bv<2>(datatype);
    metadata.range(41, 10) = sc_bv<32>(static_cast<uint32_t>(fetch_index));
    metadata.range(73, 42) = sc_bv<32>(static_cast<uint32_t>(req_sm));
    segments.push_back(metadata);

    std::vector<std::pair<std::string, int>> vec(param_value.begin(),
                                                 param_value.end());
    std::sort(vec.begin(), vec.end(),
              [](auto &a, auto &b) { return a.first < b.first; });

    // 规定一个参数使用32位存储，即一个segment存储4个参数
    for (auto it = vec.begin(); it != vec.end();) {
        sc_bv<128> d = 0;
        d.range(7, 0) = sc_bv<8>(PrimFactory::getInstance().getPrimId(name));
        int pos = 8;
        for (int i = 0; i < 4 && it != vec.end(); i++, it++, pos += 30) {
            d.range(pos + 29, pos) = sc_bv<30>(it->second);
            LOG_DEBUG(CONFIG_DEBUG) << "In serialize " << name << ": " << it->first
                              << " = " << it->second;
        }

        segments.push_back(d);
    }

    return segments;
}

void GpuBase::deserialize(vector<sc_bv<128>> segments) {
    LOG_DEBUG(CONFIG_DEBUG) << "Start deserialize " << name;
    if (segments.empty())
        throw std::invalid_argument(name + " GPU Prim wire has no segments");

    vector<string> vec(param_name.begin(), param_name.end());
    sort(vec.begin(), vec.end());
    const size_t expected_segments = 1 + (vec.size() + 3) / 4;
    if (segments.size() != expected_segments)
        throw std::invalid_argument(
            name + " GPU Prim wire segment count does not match parameters");

    const uint64_t expected_id = static_cast<uint64_t>(
        PrimFactory::getInstance().getPrimId(name));
    for (const auto &segment : segments)
        if (segment.range(7, 0).to_uint64() != expected_id)
            throw std::invalid_argument(
                name + " GPU Prim wire contains inconsistent segment IDs");

    const auto &metadata = segments[0];
    const bool legacy = prim_wire::LegacyCompatibilityEnabled();
    if (legacy) {
        if (metadata.range(127, 57).or_reduce() ||
            metadata.range(40, 25).or_reduce())
            throw std::invalid_argument(
                name + " legacy GPU Prim reserved bits are set");
        datatype = static_cast<DATATYPE>(
            metadata.range(8, 8).to_uint64());
        fetch_index = metadata.range(24, 9).to_uint64();
        req_sm = metadata.range(56, 41).to_uint64();
    } else {
        if (metadata.range(127, 74).or_reduce())
            throw std::invalid_argument(
                name + " GPU Prim wire reserved bits are set");
        const uint64_t raw_datatype = metadata.range(9, 8).to_uint64();
        if (raw_datatype > static_cast<uint64_t>(FP16))
            throw std::invalid_argument(
                name + " GPU Prim datatype is invalid");
        datatype = static_cast<DATATYPE>(raw_datatype);
        const uint64_t raw_fetch = metadata.range(41, 10).to_uint64();
        const uint64_t raw_req_sm = metadata.range(73, 42).to_uint64();
        if (raw_fetch >
                static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
            raw_req_sm >
                static_cast<uint64_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument(
                name + " GPU index field exceeds runtime int range");
        fetch_index = static_cast<int>(raw_fetch);
        req_sm = static_cast<int>(raw_req_sm);
    }

    decltype(param_value) decoded_param_value;
    for (size_t i = 1; i < segments.size(); ++i) {
        const auto &segment = segments[i];
        for (size_t j = 0; j < 4; ++j) {
            const size_t index = (i - 1) * 4 + j;
            const int low = static_cast<int>(8 + j * 30);
            if (legacy && index >= vec.size())
                break;
            const int high = legacy
                ? static_cast<int>(29 + j * 30)
                : low + 29;
            const uint64_t value = segment.range(high, low).to_uint64();
            if (index >= vec.size()) {
                if (value != 0)
                    throw std::invalid_argument(
                        name + " GPU Prim unused parameter bits are set");
                continue;
            }
            decoded_param_value[vec[index]] = static_cast<int>(value);
            LOG_DEBUG(CONFIG_DEBUG)
                << "In deserialize " << name << ": " << vec[index]
                << " = " << decoded_param_value[vec[index]];
        }
    }

    param_value = std::move(decoded_param_value);
    initialize();
    initializeDefault();
    LOG_DEBUG(CONFIG) << "Finish deserialize " << name;
}

void GpuBase::parseCompose(json j) {
    SetParamFromJson(j, "require_sm", &req_sm);
}

void GpuBase::parseAddress(json j) {
    string in_label = j["indata"];
    prim_context->datapass_label_->outdata = j["outdata"];

    std::vector<std::string> in_labels;

    std::istringstream iss(in_label);
    std::string word;
    std::string temp;

    // 保证DRAM_LABEL后面跟着另一个label
    while (iss >> word) {
        if (word == DRAM_LABEL || word == "_" + string(DRAM_LABEL)) {
            temp = word;
            if (iss >> word) {
                temp += " " + word;
            }
            in_labels.push_back(temp);
        } else {
            in_labels.push_back(word);
        }
    }

    for (int i = 0; i < in_labels.size(); i++) {
        prim_context->datapass_label_->indata[i] = in_labels[i];
    }
}

void GpuBase::parseJson(json j) {
    for (auto &param : param_name) {
        SetParamFromJson(j, param, &param_value[param]);
    }

    initialize();
    initializeDefault();

    if (j.contains("compose"))
        parseCompose(j["compose"]);

    if (j.contains("address"))
        parseAddress(j["address"]);
}

void GpuBase::initializeDefault() {
    if (datatype == INT8)
        data_byte = 1;
    else if (datatype == FP16)
        data_byte = 2;

    input_size = 0;
    for (auto &input : data_size_input)
        input_size += input;

    out_size = -1;
    for (const auto &chunk : data_chunk) {
        if (chunk.first == "output") {
            out_size = chunk.second;
            break;
        }
    }

    if (out_size < 0) {
        LOG_ERROR(gpu_base.cpp) << "No output chunk found";
    }
}

void GpuBase::printSelf() {}