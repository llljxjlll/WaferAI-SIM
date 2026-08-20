#include "prims/base.h"
#include "memory/sram/sram_region.h"
#include "utils/config_utils.h"
#include "utils/memory_utils.h"
#include "utils/prim_utils.h"
#include "utils/print_utils.h"
#include "utils/system_utils.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace {
int DecodeSignedWire32(uint64_t raw, const std::string &field) {
    if (raw > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument(field + " is not a 32-bit wire value");
    if (raw <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
        return static_cast<int>(raw);
    const int64_t signed_value =
        static_cast<int64_t>(raw) - (int64_t{1} << 32);
    if (signed_value < std::numeric_limits<int32_t>::min())
        throw std::invalid_argument(field + " signed decode overflowed");
    return static_cast<int>(signed_value);
}

void ValidateOneShotLabels(const AddrDatapassLabel &labels,
                           uint32_t input_count) {
    if (input_count == 0 || input_count > MAX_SPLIT_NUM)
        throw std::logic_error(
            "pending SRAM_BIND input count is outside [1, 16]");
    for (uint32_t index = 0; index < input_count; ++index) {
        if (labels.indata[index].empty() ||
            labels.indata[index] == UNSET_LABEL)
            throw std::logic_error(
                "pending SRAM_BIND contains an unset input label");
    }
    for (uint32_t index = input_count; index < MAX_SPLIT_NUM; ++index) {
        if (labels.indata[index] != UNSET_LABEL)
            throw std::logic_error(
                "pending SRAM_BIND contains a set unused input label");
    }
    if (labels.outdata.empty() || labels.outdata == UNSET_LABEL)
        throw std::logic_error(
            "pending SRAM_BIND contains an unset output label");
}

class ScopedOneShotSramBinding {
public:
    ScopedOneShotSramBinding(PrimCoreContext &context,
                             size_t compute_input_count,
                             const std::string &prim_name)
        : context_(context) {
        if (!context_.program_mode_) return;
        if (context_.datapass_label_ == nullptr)
            throw std::logic_error(
                prim_name + " has no active SRAM label context");
        if (!context_.sram_bind_pending_)
            throw std::logic_error(
                prim_name + " requires a preceding one-shot SRAM_BIND");
        if (context_.sram_bind_input_count_ != compute_input_count)
            throw std::logic_error(
                prim_name + " input count does not match pending SRAM_BIND");
        ValidateOneShotLabels(context_.sram_bind_pending_labels_,
                              context_.sram_bind_input_count_);

        AddrDatapassLabel active_labels =
            context_.sram_bind_pending_labels_;
        AddrDatapassLabel cleared_pending;
        previous_labels_ = *context_.datapass_label_;
        *context_.datapass_label_ = std::move(active_labels);
        active_ = true;

        // Consumption happens when the compute has successfully entered the
        // NPU path. It remains consumed if later compute work throws.
        context_.sram_bind_pending_ = false;
        context_.sram_bind_input_count_ = 0;
        context_.sram_bind_pending_labels_ = std::move(cleared_pending);
    }

    ~ScopedOneShotSramBinding() {
        if (active_)
            *context_.datapass_label_ = std::move(previous_labels_);
    }

    ScopedOneShotSramBinding(const ScopedOneShotSramBinding &) = delete;
    ScopedOneShotSramBinding &operator=(
        const ScopedOneShotSramBinding &) = delete;

private:
    PrimCoreContext &context_;
    AddrDatapassLabel previous_labels_;
    bool active_ = false;
};

} // namespace

void NpuBase::parseAddress(json j) {
    SetParamFromJson(j, "input", &inp_offset, 0);
    SetParamFromJson(j, "data", &data_offset,
                     inp_offset + input_size * data_byte);

    int total_data_size = 0;
    for (auto &pair : data_chunk)
        total_data_size += pair.second;

    SetParamFromJson(j, "output", &out_offset,
                     data_offset + total_data_size * data_byte);
}

void NpuBase::parseSramLabel(json j) {
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

vector<sc_bv<128>> NpuBase::serialize() {
    vector<sc_bv<128>> segments;
    if (datatype != INT8 && datatype != FP16)
        throw std::invalid_argument(name + " has an invalid datatype");
    for (const auto &entry : param_value)
        if (entry.second < 0 || entry.second > 0x3fffffff)
            throw std::overflow_error(name + " parameter " + entry.first +
                                      " exceeds the 30-bit Prim wire");


    sc_bv<128> metadata = 0;
    metadata.range(7, 0) = sc_bv<8>(PrimFactory::getInstance().getPrimId(name));
    metadata.range(9, 8) = sc_bv<2>(datatype);
    metadata.range(41, 10) = sc_bv<32>(static_cast<uint32_t>(inp_offset));
    metadata.range(73, 42) = sc_bv<32>(static_cast<uint32_t>(data_offset));
    metadata.range(105, 74) =
        sc_bv<32>(static_cast<uint32_t>(out_offset));
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
        }

        segments.push_back(d);
    }

    return segments;
}

void NpuBase::deserialize(vector<sc_bv<128>> segments) {
    if (segments.empty())
        throw std::invalid_argument(name + " Prim wire has no segments");

    vector<string> vec(param_name.begin(), param_name.end());
    sort(vec.begin(), vec.end());
    const size_t expected_segments = 1 + (vec.size() + 3) / 4;
    if (segments.size() != expected_segments)
        throw std::invalid_argument(
            name + " Prim wire segment count does not match its parameters");

    const uint64_t expected_id = static_cast<uint64_t>(
        PrimFactory::getInstance().getPrimId(name));
    for (const auto &segment : segments)
        if (segment.range(7, 0).to_uint64() != expected_id)
            throw std::invalid_argument(
                name + " Prim wire contains inconsistent segment IDs");

    const auto &metadata = segments[0];
    if (prim_wire::LegacyCompatibilityEnabled()) {
        if (metadata.range(127, 57).or_reduce())
            throw std::invalid_argument(
                name + " legacy Prim wire reserved bits are set");
        datatype = static_cast<DATATYPE>(
            metadata.range(8, 8).to_uint64());
        inp_offset = metadata.range(24, 9).to_uint64();
        data_offset = metadata.range(40, 25).to_uint64();
        out_offset = metadata.range(56, 41).to_uint64();
    } else {
        if (metadata.range(127, 106).or_reduce())
            throw std::invalid_argument(
                name + " Prim wire reserved bits are set");
        const uint64_t raw_datatype = metadata.range(9, 8).to_uint64();
        if (raw_datatype > static_cast<uint64_t>(FP16))
            throw std::invalid_argument(
                name + " Prim wire datatype is invalid");
        datatype = static_cast<DATATYPE>(raw_datatype);
        inp_offset = DecodeSignedWire32(
            metadata.range(41, 10).to_uint64(), name + " input offset");
        data_offset = DecodeSignedWire32(
            metadata.range(73, 42).to_uint64(), name + " data offset");
        out_offset = DecodeSignedWire32(
            metadata.range(105, 74).to_uint64(), name + " output offset");
    }

    decltype(param_value) decoded_param_value;
    for (size_t i = 1; i < segments.size(); ++i) {
        const auto &segment = segments[i];
        for (size_t j = 0; j < 4; ++j) {
            const size_t index = (i - 1) * 4 + j;
            const int low = static_cast<int>(8 + j * 30);
            const int high = low + 29;
            const uint64_t value = segment.range(high, low).to_uint64();
            if (index >= vec.size()) {
                if (value != 0)
                    throw std::invalid_argument(
                        name + " Prim wire unused parameter bits are set");
                continue;
            }
            decoded_param_value[vec[index]] = static_cast<int>(value);
        }
    }

    param_value = std::move(decoded_param_value);
    initialize();
    initializeDefault();
}

void NpuBase::parseJson(json j) {
    for (auto &param : param_name) {
        SetParamFromJson(j, param, &param_value[param]);
    }

    initialize();
    initializeDefault();

    if (j.contains("dram_address"))
        parseAddress(j["dram_address"]);

    if (j.contains("sram_address"))
        parseSramLabel(j["sram_address"]);
}

int NpuBase::sramUtilization(DATATYPE datatype, int cid) {
    int total_sram = 0;

    total_sram += CeilingDivision(input_size * data_byte * 8,
                                  GetCoreHWConfig(cid)->sram_bitwidth);

    for (auto &pair : data_chunk) {
        total_sram += CeilingDivision(pair.second * data_byte * 8,
                                      GetCoreHWConfig(cid)->sram_bitwidth);
    }

    total_sram *= GetCoreHWConfig(cid)->sram_bitwidth / 8;
    return total_sram;
}

void NpuBase::initializeDefault() {
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
        LOG_ERROR(npu_base.cpp) << "No output chunk found for " << name;
        return;
    }

    int pos = data_offset;
    for (auto &chunk : data_chunk) {
        data_chunk_addr[chunk.first] = pos;
        chunk.second *= data_byte;
        pos += chunk.second;
    }
}

int NpuBase::taskCoreDefault(TaskCoreContext &context) {
    if (prim_context == nullptr)
        throw std::logic_error(name + " requires a PrimCoreContext");
    last_cost_snapshot_ = {};
    prepareProgramSramBinding(*prim_context);
    ScopedOneShotSramBinding one_shot_binding(
        *prim_context, data_size_input.size(), name);

    // 检查是否满足auto_pd的条件，若是，则将T参数设置为1，并重新初始化
    if (prim_context->auto_pd_ &&
        prim_context->loop_cnt > prim_context->auto_pd_) {
        param_value["T"] = 1;
        initialize();
        initializeDefault();
    }

    const bool manual_memory_schedule =
        !usesLegacyImplicitMemory(context);

    // 所用时间
    u_int64_t dram_time = 0;
    u_int64_t overlap_time = 0;

    // 检查数据重利用
    bool input_reuse[data_size_input.size()];
    for (int i = 0; i < data_size_input.size(); i++) {
        input_reuse[i] = false;
        if (prim_context->datapass_label_->indata[i][0] == '_') {
            input_reuse[i] = true;
            prim_context->datapass_label_->indata[i] =
                prim_context->datapass_label_->indata[i].substr(1);
        }
    }

    // 获取前缀label
    std::size_t pos = prim_context->datapass_label_->outdata.find_last_of('_');
    std::string prefix;
    if (pos != std::string::npos)
        prefix = prim_context->datapass_label_->outdata.substr(0, pos);
    else
        prefix = prim_context->datapass_label_->outdata;

    // 读入input数据
    if (!manual_memory_schedule && !skip_input)
        checkInputData(context, dram_time, inp_offset, data_size_input);

    u_int64_t exu_flops = 0;
    u_int64_t sfu_flops = 0;
    u_int64_t vec_flops = 0;
#if USE_SRAM == 1
    {
        // 自定义task
        taskCore(context, prefix, dram_time, exu_flops, sfu_flops, vec_flops);

        // 删除标签
        if (!manual_memory_schedule)
            for (int i = 0; i < data_size_input.size(); i++) {
            if (!input_reuse[i] &&
                prim_context->datapass_label_->indata[i] != UNSET_LABEL)
                prim_context->sram_pos_locator_->deletePair(
                    prim_context->datapass_label_->indata[i]);
        }
    }
#endif

    if (manual_memory_schedule && dram_time != 0)
        throw std::logic_error(
            name + " manual memory schedule performed an implicit memory access");

    // Compute is charged exactly once and independently of implicit output
    // handling. In particular, skip_output suppresses only the legacy memory
    // write; it never suppresses execution cost.
    overlap_time = chargeComputeCost(context, exu_flops, sfu_flops,
                                    vec_flops, dram_time);

    // Legacy configurations preserve their historical implicit output write.
    // Program-mode schedules own all SRAM/HBM transfers and stop above after
    // the side-effect-free compute charge.
    if (!manual_memory_schedule && !skip_output)
        writeOutputData(context, dram_time, overlap_time, out_size,
                        data_chunk_addr["output"]);

    return overlap_time;
}

bool NpuBase::usesLegacyImplicitMemory(
    const TaskCoreContext &context) const noexcept {
    return context.sram_regions == nullptr ||
           !context.sram_regions->config().manual_memory_schedule;
}

void NpuBase::checkInputData(TaskCoreContext &context, uint64_t &dram_time,
                             uint64_t inp_global_addr,
                             vector<int> data_size_input) {
#if USE_NB_DRAMSYS == 0
    auto wc = context.wc;
#endif
    auto mau = context.mau;
    auto hmau = context.hmau;
    auto sram_addr = context.sram_addr;
    int inp_sram_offset = *sram_addr;

#if DUMMY == 1
    float *dram_start = nullptr;
#else
    float *dram_start = (float *)(dram_array[cid]);
    float *inp = dram_start + inp_offset;
    float *out = dram_start + out_offset;
#endif
    LOG_DEBUG(PRIM) << name << " of Core " << context.cid
                    << " start loading input data";

#if USE_SRAM == 1
    for (int p = 0; p < data_size_input.size(); p++) {
        if (prim_context->datapass_label_->indata[p].find(DRAM_LABEL) == 0) {
            size_t space_pos =
                prim_context->datapass_label_->indata[p].find(' ');
            if (space_pos != std::string::npos) {
                prim_context->datapass_label_->indata[p] =
                    prim_context->datapass_label_->indata[p].substr(space_pos +
                                                                    1);
            }

            LOG_INFO(MEMORY)
                << name << " of Core " << context.cid << " read label "
                << prim_context->datapass_label_->indata[p].c_str()
                << " from DRAM";
#if USE_SRAM_MANAGER == 1
            sram_first_write_generic(context, data_byte * data_size_input[p],
                                     inp_global_addr, dram_time, dram_start,
                                     prim_context->datapass_label_->indata[p],
                                     true, prim_context->sram_pos_locator_);
#else
            const int input_sram_start = *sram_addr;
            sram_first_write_generic(context, data_byte * data_size_input[p],
                                     inp_global_addr, dram_time, dram_start);
            AddrPosKey inp_key =
                AddrPosKey(input_sram_start, data_byte * data_size_input[p],
                           inp_global_addr);
            inp_key.preferred_region = "input";
            prim_context->sram_pos_locator_->addPair(
                prim_context->datapass_label_->indata[p], inp_key, context,
                dram_time);
#endif
        } else {
            AddrPosKey inp_key;
            LOG_DEBUG(PRIM)
                << name << " of Core " << context.cid << " read label "
                << prim_context->datapass_label_->indata[p].c_str()
                << " from SRAM";

            int flag = prim_context->sram_pos_locator_->findPair(
                prim_context->datapass_label_->indata[p], inp_key);
            if (flag == -1) {
                LOG_ERROR(npu_base.cpp)
                    << name << " of Core " << context.cid << " cannot find "
                    << prim_context->datapass_label_->indata[p];
            } else if (flag > 0) {

#if USE_SRAM_MANAGER == 1
                LOG_DEBUG(MEMORY) << name << " of Core " << cid
                                  << ", SRAM pos locator found label "
                                  << prim_context->datapass_label_->indata[p]
                                  << " with flag " << flag;

                sram_first_write_generic(
                    context, flag, inp_global_addr, dram_time, dram_start,
                    prim_context->datapass_label_->indata[p], true,
                    prim_context->sram_pos_locator_);

#else
                LOG_DEBUG(MEMORY) << name << " of Core " << context.cid
                                  << " input label spilled, need to fetch back";

                sram_first_write_generic(context, flag, inp_global_addr,
                                         dram_time, dram_start);
                inp_key.size = data_size_input[p];
                inp_key.spill_size = 0;
                prim_context->sram_pos_locator_->addPair(
                    prim_context->datapass_label_->indata[p], inp_key, context,
                    dram_time);
#endif
            } else {
                // send receive input data
#if USE_SRAM_MANAGER == 1
                LOG_DEBUG(MEMORY) << name << " of Core " << context.cid
                                  << " found label in SRAM";

                AddrPosKey inp_key;
                int flag = prim_context->sram_pos_locator_->findPair(
                    prim_context->datapass_label_->indata[p], inp_key);
                if (inp_key.alloc_id == 0) {
                    sram_first_write_generic(
                        context, data_byte * data_size_input[p],
                        inp_global_addr, dram_time, dram_start,
                        prim_context->datapass_label_->indata[p], true,
                        prim_context->sram_pos_locator_, true);
                }
#else
                LOG_DEBUG(MEMORY) << name << " of Core " << context.cid
                                  << " found label in SRAM";

                inp_key.size = data_size_input[p];
                inp_key.spill_size = 0;
                prim_context->sram_pos_locator_->addPair(
                    prim_context->datapass_label_->indata[p], inp_key, context,
                    dram_time);


#endif
            }
#if USE_SRAM_MANAGER == 1

            // mla kvcache
            AddrPosKey sc_key;
            prim_context->sram_pos_locator_->findPair(
                prim_context->datapass_label_->indata[p], sc_key);


            int data_bits = data_byte * data_size_input[p] * 8;
            assert((SRAM_BLOCK_SIZE * 8) % SRAM_BITWIDTH == 0 &&
                   "SRAM_BLOCK_SIZE * 8 must be a multiple of SRAM_BITWIDTH");
            int alignment = std::max(SRAM_BITWIDTH, SRAM_BLOCK_SIZE * 8);

            int aligned_data_bits =
                static_cast<int>(
                    std::ceil(static_cast<double>(data_bits) / alignment)) *
                alignment;
            int aligned_data_byte = aligned_data_bits / 8;
            if (prim_context->sram_pos_locator_
                    ->data_map[prim_context->datapass_label_->indata[p]]
                    .size < aligned_data_byte) {
                LOG_WARN(MEMORY) << name << " of Core " << context.cid
                                 << ", data size is smaller than aligned size";

                auto sram_manager_ = context.sram_manager_;
#if ASSERT == 1
                assert(prim_context->sram_pos_locator_->validateTotalSize());
#endif
                int ori_size =
                    prim_context->sram_pos_locator_
                        ->data_map[prim_context->datapass_label_->indata[p]]
                        .size;
                sc_key.size +=
                    aligned_data_byte -
                    prim_context->sram_pos_locator_
                        ->data_map[prim_context->datapass_label_->indata[p]]
                        .size;
                prim_context->sram_pos_locator_->addPair(
                    prim_context->datapass_label_->indata[p], sc_key, context,
                    dram_time, false);
                prim_context->sram_manager_->allocate_append(
                    aligned_data_byte - ori_size, sc_key.alloc_id);
                // sc_key.size +=
                //     aligned_data_byte -
                //     sram_pos_locator_->data_map[prim_context->datapass_label_->indata[p]].size;
                // sram_pos_locator_->addPair(prim_context->datapass_label_->indata[p],
                // sc_key,
                //                           false);
#if ASSERT == 1
                assert(prim_context->sram_pos_locator_->validateTotalSize());
#endif
            }
#else
            if (prim_context->sram_pos_locator_
                    ->data_map[prim_context->datapass_label_->indata[p]]
                    .size < inp_key.size) {
                LOG_ERROR(MEMORY) << name << " of Core " << context.cid
                                  << ", data size is smaller than aligned size";
                prim_context->sram_pos_locator_->addPair(
                    prim_context->datapass_label_->indata[p], inp_key, false);
            }


#endif
        }

#if USE_SRAM_MANAGER == 1
        AddrPosKey input_key;
        prim_context->sram_pos_locator_->findPair(
            prim_context->datapass_label_->indata[p], input_key);
        prim_context->sram_pos_locator_->printAllKeysWithAllocId();
        // Print allocation IDs for debugging
        LOG_DEBUG(MEMORY) << name << " of Core " << context.cid
                          << ", input data Allocation ID "
                          << input_key.alloc_id;

        sram_read_generic(context, data_byte * data_size_input[p],
                          inp_sram_offset, dram_time, input_key.alloc_id, true,
                          prim_context->sram_pos_locator_);
#else
        // 读出input
        LOG_INFO(MEMORY) << name << " of Core " << context.cid
                         << " read input data";

        prim_context->sram_pos_locator_->findPair(
            prim_context->datapass_label_->indata[p], inp_sram_offset);
        sram_read_generic(context, data_byte * data_size_input[p],
                          inp_sram_offset, dram_time);
#endif

        inp_global_addr += data_size_input[p] * data_byte;
    }
#endif
}


void NpuBase::prefReadData(TaskCoreContext &context, uint64_t &dram_time,
                           int data_size_label, string label_name) {
#if USE_NB_DRAMSYS == 0
    auto wc = context.wc;
#endif
    auto sram_addr = context.sram_addr;


    AddrPosKey sc_key;
    int flag = prim_context->sram_pos_locator_->findPair(label_name, sc_key);
    if (flag == -1) {
        assert(false && "weight data not found");
    } else if (flag > 0) {
        assert(false && "weight data can not be spilled");
    }
    // dahu ??
    int sram_offset = 0;
    prim_context->sram_pos_locator_->findPair(label_name, sc_key);
#if USE_SRAM_MANAGER == 1
    prim_context->sram_pos_locator_->printAllKeysWithAllocId();
    // Print allocation IDs for debugging
    LOG_DEBUG(MEMORY_DEBUG)
        << "Key Allocation ID of " << label_name << " i" << sc_key.alloc_id;

    sram_read_generic(context, data_byte * data_size_label, sram_offset,
                      dram_time, sc_key.alloc_id, true,
                      prim_context->sram_pos_locator_);
#else

    sram_read_generic(context, data_byte * data_size_label, sram_offset,
                      dram_time);

#endif
}
void NpuBase::checkStaticData(TaskCoreContext &context, uint64_t &dram_time,
                              uint64_t label_global_addr, int data_size_label,
                              string label_name, bool use_pf) {
#if USE_NB_DRAMSYS == 0
    auto wc = context.wc;
#endif
    auto sram_addr = context.sram_addr;
    int sram_offset = *sram_addr;

#if DUMMY == 1
    float *dram_start = nullptr;
#else
    float *dram_start = (float *)(dram_array[cid]);
    float *inp = dram_start + inp_offset;
    float *out = dram_start + out_offset;
#endif

    AddrPosKey sc_key;
    int flag = prim_context->sram_pos_locator_->findPair(label_name, sc_key);
    if (flag == -1) {
        LOG_DEBUG(MEMORY)
            << name << " of Core " << context.cid
            << " weight label does not exist in SRAM, need to fetch";

#if USE_SRAM_MANAGER == 1
        sram_first_write_generic(
            context, data_byte * data_size_label, label_global_addr, dram_time,
            dram_start, label_name, true, prim_context->sram_pos_locator_);
#else
        sram_first_write_generic(context, data_byte * data_size_label,
                                 label_global_addr, dram_time, dram_start);

        sc_key = AddrPosKey(sram_offset, data_byte * data_size_label,
                            label_global_addr);
        sc_key.preferred_region = "input";
        if (SPEC_LOAD_STATIC == "layer")
            sc_key.allocation_lifetime =
                sram::AllocationLifetime::kLayer;
        prim_context->sram_pos_locator_->addPair(label_name, sc_key, context,
                                                 dram_time);
#endif
    } else if (flag > 0) {
        LOG_DEBUG(MEMORY) << name << " of Core " << context.cid
                          << " weight label spilled, need to fetch back, flag "
                          << flag;
#if USE_SRAM_MANAGER == 1
        sram_first_write_generic(context, flag, label_global_addr, dram_time,
                                 dram_start, label_name, true,
                                 prim_context->sram_pos_locator_);

#else
        sram_first_write_generic(context, flag, label_global_addr, dram_time,
                                 dram_start);
        sc_key.size = data_byte * data_size_label;
        sc_key.spill_size = 0;
        prim_context->sram_pos_locator_->addPair(label_name, sc_key, context,
                                                 dram_time);
#endif
    }


    prim_context->sram_pos_locator_->findPair(label_name, sc_key);
    LOG_DEBUG(PRIM) << name << " of Core " << context.cid << " read label "
                    << label_name << " from SRAM";
#if USE_SRAM_MANAGER == 1
    prim_context->sram_pos_locator_->printAllKeysWithAllocId();
    // Print allocation IDs for debugging
    LOG_DEBUG(MEMORY_DEBUG)
        << "Key Allocation ID of " << label_name << " is " << sc_key.alloc_id;

    if (use_pf == false) {
        sram_read_generic(context, data_byte * data_size_label, sram_offset,
                          dram_time, sc_key.alloc_id, true,
                          prim_context->sram_pos_locator_);
    }
#else
    if (use_pf == false) {
        sram_read_generic(context, data_byte * data_size_label, sram_offset,
                          dram_time);
    }
#endif
}


void NpuBase::checkStaticDataTile(TaskCoreContext &context, uint64_t &dram_time,
                                  uint64_t label_global_addr,
                                  int data_size_label, string label_name,
                                  bool use_pf, int mac_size) {
    auto sram_addr = context.sram_addr;
    int sram_offset = *sram_addr;
    float *dram_start = nullptr;

    int load_size = 64 * 1024;

    AddrPosKey sc_key;
    int flag = prim_context->sram_pos_locator_->findPair(label_name, sc_key);
    if (flag == -1) {
        LOG_DEBUG(MEMORY)
            << name << " of Core " << context.cid
            << " weight label does not exist in SRAM, need to fetch";

        int size = 0;
        for (int tile = 0; tile < data_size_label / load_size; tile++) {
            sram_first_write_generic(context, data_byte * load_size,
                                     label_global_addr, dram_time, dram_start);
            size += load_size * data_byte;

            sc_key = AddrPosKey(sram_offset, size, label_global_addr);
            sc_key.preferred_region = "input";
            if (SPEC_LOAD_STATIC == "layer")
                sc_key.allocation_lifetime =
                    sram::AllocationLifetime::kLayer;
            prim_context->sram_pos_locator_->addPairByTile(label_name, sc_key,
                                                           context, dram_time);
        }
    } else if (flag > 0) {
        LOG_DEBUG(MEMORY) << name << " of Core " << context.cid
                          << " weight label spilled, need to fetch back";

        int size = data_size_label * data_byte - flag;
        for (int tile = 0; tile < flag / load_size; tile++) {
            sram_first_write_generic(context, data_byte * load_size,
                                     label_global_addr, dram_time, dram_start);
            size += load_size * data_byte;

            sc_key.size = size;
            prim_context->sram_pos_locator_->addPairByTile(label_name, sc_key,
                                                           context, dram_time);
        }
    }

    sram_read_generic(context, data_byte * data_size_label, sram_offset,
                      dram_time);
}


uint64_t NpuBase::chargeComputeCost(TaskCoreContext &context,
                                    uint64_t exu_flops,
                                    uint64_t sfu_flops,
                                    uint64_t vec_flops,
                                    uint64_t dram_time) {
    int cid = context.cid;
    CoreHWConfig *hardware_config = GetCoreHWConfig(cid);
    ExuConfig *exu = hardware_config->exu;
    SfuConfig *sfu = hardware_config->sfu;
    VectorConfig *vec = hardware_config->vec;

    LOG_DEBUG(PRIM) << name << " of Core " << context.cid << ": exu_flops "
                    << exu_flops << " sfu_flops " << sfu_flops << " vec_flops "
                    << vec_flops;

    if (exu->type != MAC_Array)
        assert(false && "Unsupported tile type");
    if (sfu->type != Linear)
        assert(false && "Unsupported tile type");

    NpuCostHardware cost_hardware;
    cost_hardware.exu_x_dims = static_cast<uint64_t>(exu->x_dims);
    cost_hardware.exu_count = static_cast<uint64_t>(exu->count);
    cost_hardware.sfu_x_dims = static_cast<uint64_t>(sfu->x_dims);
    cost_hardware.vec_x_dims = static_cast<uint64_t>(vec->x_dims);
    cost_hardware.vec_count = static_cast<uint64_t>(vec->count);
    cost_hardware.compute_utilization = HW_COMP_UTIL;
    cost_hardware.cycle_ns = CYCLE;
    last_cost_snapshot_ = CalculateNpuCost(
        NpuOps{exu_flops, sfu_flops, vec_flops}, cost_hardware, dram_time);
    if (last_cost_snapshot_.overlap_delay_ns >
        static_cast<uint64_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("NPU overlap delay exceeds runtime int");
    return last_cost_snapshot_.overlap_delay_ns;
}

void NpuBase::writeOutputData(TaskCoreContext &context, uint64_t dram_time,
                              uint64_t &overlap_time, int data_size_out,
                              uint64_t out_global_addr) {

#if USE_SRAM == 1
    if (dram_time > last_cost_snapshot_.compute_cycle_ns) {
        // 因为dram 已经wait 过了，所以额外的 overlap_time = 0
        overlap_time = 0;
        LOG_INFO(PRIM) << name << " of Core " << context.cid << ": dram_time "
                       << dram_time << ", compute cycle "
                       << last_cost_snapshot_.compute_cycle_ns;

    } else {
        LOG_INFO(PRIM) << name << " of Core " << context.cid << ": dram_time "
                       << dram_time << ", compute cycle "
                       << last_cost_snapshot_.compute_cycle_ns;
    }

    // 写入out
    std::vector<std::string> out_labels;
    std::istringstream iss(prim_context->datapass_label_->outdata);
    std::string label;
    while (iss >> label)
        out_labels.push_back(label);

    int temp_out_sram_offset = *(context.sram_addr);


#if USE_SRAM_MANAGER == 1
    for (int i = 0; i < out_labels.size(); i++) {
        sram_write_append_generic(
            context, data_byte * data_size_out / out_labels.size(),
            overlap_time, out_labels[i], true, prim_context->sram_pos_locator_,
            out_global_addr +
                i * data_byte * data_size_out / out_labels.size());
    }
#else
    sram_write_append_generic(context, data_byte * data_size_out, overlap_time);
    auto interval =
        (*(context.sram_addr) - temp_out_sram_offset) / out_labels.size();

    for (int i = 0; i < out_labels.size(); i++) {
        AddrPosKey out_key =
            AddrPosKey(static_cast<int>(temp_out_sram_offset + i * interval),
                       data_byte * data_size_out / out_labels.size(),
                       out_global_addr +
                           i * data_byte * data_size_out / out_labels.size());
        out_key.preferred_region = "intermediate";
        // already wait in addPair do not add overlap_time
        prim_context->sram_pos_locator_->addPair(out_labels[i], out_key,
                                                 context, dram_time);
    }
#endif
#endif
}

void NpuBase::printSelf() {}
