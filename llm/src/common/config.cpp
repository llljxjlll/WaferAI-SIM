#include "common/config.h"
#include "common/msg.h"
#include "utils/config_utils.h"
#include "utils/prim_utils.h"
#include "utils/print_utils.h"
#include "utils/system_utils.h"

#include <limits>

namespace {

uint64_t ParseUnsignedInteger(const json &object, const char *field,
                              uint64_t default_value,
                              const std::string &path) {
    if (!object.contains(field)) return default_value;

    const json &value = object.at(field);
    if (value.is_number_unsigned()) return value.get<uint64_t>();
    if (value.is_number_integer()) {
        const int64_t signed_value = value.get<int64_t>();
        if (signed_value >= 0) return static_cast<uint64_t>(signed_value);
    }

    throw std::invalid_argument(path + "." + field +
                                " must be a non-negative integer, got " +
                                value.dump());
}

uint64_t ParseNanoseconds(const json &object, const char *field,
                          uint64_t default_value,
                          const std::string &path) {
    const uint64_t value =
        ParseUnsignedInteger(object, field, default_value, path);
    constexpr uint64_t kMaxExactNanoseconds = uint64_t{1} << 53;
    if (value > kMaxExactNanoseconds)
        throw std::invalid_argument(
            path + "." + field +
            " is too large for an exact SystemC nanosecond conversion, got " +
            std::to_string(value));
    return value;
}

uint32_t ParsePositiveUint32(const json &object, const char *field,
                             uint32_t default_value,
                             const std::string &path) {
    const uint64_t value =
        ParseUnsignedInteger(object, field, default_value, path);
    if (value == 0 || value > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument(
            path + "." + field + " must be in [1, " +
            std::to_string(std::numeric_limits<uint32_t>::max()) +
            "], got " + std::to_string(value));
    return static_cast<uint32_t>(value);
}

ControlCoresHWConfig ParseControlCoresConfig(const json &object, int core_id) {
    const std::string path = "core[" + std::to_string(core_id) +
                             "].control_cores";
    if (!object.is_object())
        throw std::invalid_argument(path + " must be an object, got " +
                                    object.dump());

    ControlCoresHWConfig config;
    if (object.contains("mode")) {
        const json &mode = object.at("mode");
        if (!mode.is_string())
            throw std::invalid_argument(path +
                                        ".mode must be a string, got " +
                                        mode.dump());
        const std::string value = mode.get<std::string>();
        if (value == "legacy_shared")
            config.mode = ControlCoreMode::LEGACY_SHARED;
        else if (value == "dual_dte_dedicated")
            config.mode = ControlCoreMode::DUAL_DTE_DEDICATED;
        else
            throw std::invalid_argument(
                path + ".mode must be legacy_shared or "
                       "dual_dte_dedicated, got " +
                value);
    }

    if (object.contains("dte")) {
        const json &dte = object.at("dte");
        const std::string dte_path = path + ".dte";
        if (!dte.is_object())
            throw std::invalid_argument(dte_path +
                                        " must be an object, got " +
                                        dte.dump());
        config.dte.command_queue_depth = ParsePositiveUint32(
            dte, "command_queue_depth", config.dte.command_queue_depth,
            dte_path);
        config.dte.dispatch_width = ParsePositiveUint32(
            dte, "dispatch_width", config.dte.dispatch_width, dte_path);
        config.dte.dispatch_latency_ns = ParseNanoseconds(
            dte, "dispatch_latency_ns", config.dte.dispatch_latency_ns,
            dte_path);
        config.dte.completion_notify_latency_ns = ParseNanoseconds(
            dte, "completion_notify_latency_ns",
            config.dte.completion_notify_latency_ns, dte_path);
    }

    if (config.dte.dispatch_width > config.dte.command_queue_depth)
        throw std::invalid_argument(
            path + ".dte.dispatch_width must be <= " + path +
            ".dte.command_queue_depth, got " +
            std::to_string(config.dte.dispatch_width) + " > " +
            std::to_string(config.dte.command_queue_depth));

    const DteControllerHWConfig defaults;
    if (config.mode == ControlCoreMode::LEGACY_SHARED &&
        (config.dte.command_queue_depth != defaults.command_queue_depth ||
         config.dte.dispatch_width != defaults.dispatch_width ||
         config.dte.dispatch_latency_ns != defaults.dispatch_latency_ns ||
         config.dte.completion_notify_latency_ns !=
             defaults.completion_notify_latency_ns))
        throw std::invalid_argument(
            path +
            ".dte must keep its default values when mode is legacy_shared");

    return config;
}

} // namespace

void CoreJob::printSelf() {}

void CoreConfig::printSelf() {}

void from_json(const json &j, Cast &c) {
    SetParamFromJson<int>(j, "dest", &(c.dest));
    SetParamFromJson<int>(j, "tag", &(c.tag), c.dest);
    SetParamFromJson<int>(j, "weight", &(c.weight), 1);
    SetParamFromJson<int>(j, "addr", &(c.addr), -1);
    SetParamFromJson<bool>(j, "critical", &(c.critical), false);
    SetParamFromJson<int>(j, "stripe", &(c.stripe), 1);
    if (c.stripe != 1 && c.stripe != 2 && c.stripe != 4)
        throw std::runtime_error("cast.stripe must be one of 1, 2, or 4");

    if (!j.contains("loopout"))
        c.loopout = BOTH;
    else {
        if (j.at("loopout") == "false")
            c.loopout = FALSE;
        else if (j.at("loopout") == "true")
            c.loopout = TRUE;
        else
            c.loopout = BOTH;
    }
}

void from_json(const json &j, CoreJob &c) {
    if (j.contains("cast")) {
        auto casts = j["cast"];
        for (int i = 0; i < casts.size(); i++) {
            Cast temp = casts[i];
            c.cast.push_back(temp);
        }
    } else if (SYSTEM_MODE != SIM_DATAFLOW)
        LOG_ERROR(config.cpp) << "Undefined \'cast\' field in json";

    SetParamFromJson<int>(j, "recv_cnt", &(c.recv_cnt));
    SetParamFromJson<int>(j, "recv_tag", &(c.recv_tag), 0);
    SetParamFromJson<int>(j, "recv_stripe", &(c.recv_stripe), 1);
    if (c.recv_stripe != 1 && c.recv_stripe != 2 && c.recv_stripe != 4)
        throw std::runtime_error("work.recv_stripe must be one of 1, 2, or 4");

    if (j.contains("prims")) {
        auto prims = j["prims"];
        for (auto prim : prims) {
            const string type = prim.at("type");
            PrimBase *base = PrimFactory::getInstance().createPrim(type);
            if (base == nullptr)
                throw std::invalid_argument(
                    "workload contains an unknown primitive: " + type);
            if (auto *async_prim = dynamic_cast<Dte_async_prim *>(base))
                async_prim->parseJson(prim);
            else if (auto *lsu_prim = dynamic_cast<Lsu_mem_prim *>(base))
                lsu_prim->parseJson(prim);
            else if (auto *pipeline =
                         dynamic_cast<Sram_pipeline_prim *>(base))
                pipeline->parseJson(prim);
            else if (auto *comp = dynamic_cast<CompBase *>(base))
                comp->parseJson(prim);
            else
                throw std::invalid_argument(
                    "workload prims supports only compute, Dte_async, Lsu_mem, "
                    "or Sram_pipeline "
                    "primitives, got: " + type);

            c.prims.push_back(base);
        }
    }
}

void from_json(const json &j, CoreConfig &c) {
    SetParamFromJson<int>(j, "id", &(c.id));

    if (c.id >= TOTAL_CORES) { // 2C-main：可寻址核总数用 TOTAL_CORES（支持 die>0 全局 id）
        LOG_ERROR(config.cpp) << "Core id " << c.id << " out of range";
        return;
    }

    SetParamFromJson<int>(j, "prim_copy", &(c.prim_copy), -1);
    SetParamFromJson<int>(j, "send_global_mem", &(c.send_global_mem), -1);
    SetParamFromJson<int>(j, "loop", &(c.loop), 1);

    if (j.contains("worklist")) {
        for (int i = 0; i < j["worklist"].size(); i++) {
            CoreJob cjob = j["worklist"][i];

            if (!j["worklist"][i].contains("recv_tag")) {
                cjob.recv_tag = c.id;
            }

            c.worklist.push_back(cjob);
        }
    } else {
        // 如果是旧版config，没有worklist条目，则将所有内容作为一个单独的job
        CoreJob cjob = j;

        if (!j.contains("recv_tag")) {
            cjob.recv_tag = c.id;
        }

        c.worklist.push_back(cjob);
    }
}

void from_json(const json &j, LayerConfig &c) {
    SetParamFromJson<int>(j, "id", &(c.id));

    for (int i = 0; i < j["cast"].size(); i++) {
        Cast temp = j["cast"][i];
        c.cast.push_back(temp);
    }

    // loop统一在外部填写
    if (j.contains("split")) {
        if (j["split"]["type"] == "TP")
            c.split = SPLIT_TP;
        else
            c.split = SPLIT_DP;

        c.split_dim = j["split"]["dim"];
        c.split_slice = j["split"]["slice"];
    } else {
        c.split = NO_SPLIT;
    }
}

void from_json(const json &j, StreamConfig &c) {
    SetParamFromJson<int>(j, "id", &(c.id));
    SetParamFromJson<int>(j, "loop", &(c.loop), 1);

    if (j.contains("prims")) {
        auto prims = j["prims"];
        for (auto prim : prims) {
            string type = prim.at("type");
            LOG_DEBUG(CONFIG_DEBUG) << "Start parsing prim " << type;
            GpuBase *p =
                (GpuBase *)(PrimFactory::getInstance().createPrim(type));
            p->parseJson(prim);
            LOG_DEBUG(CONFIG_DEBUG) << "Parsing done for prim " << type;

            c.prims.push_back((PrimBase *)p);
        }
    }

    if (j.contains("source")) {
        auto sources = j["source"];
        for (auto source : sources) {
            c.sources.push_back(
                make_pair(source["label"], GetDefinedParam(source["size"])));
        }
    }
}

void from_json(const json &j, CoreHWConfig &c) {
    SetParamFromJson<int>(j, "id", &(c.id));

    int exu_x, sa_cnt;
    SetParamFromJson<int>(j, "exu_x", &exu_x, 128);
    SetParamFromJson<int>(j, "sa_cnt", &sa_cnt, 1);
    c.exu = new ExuConfig(MAC_Array, exu_x, sa_cnt);

    int sfu_x;
    SetParamFromJson<int>(j, "sfu_x", &sfu_x, 2048);
    c.sfu = new SfuConfig(Linear, sfu_x);

    int vec_x, vec_cnt;
    SetParamFromJson<int>(j, "vec_x", &vec_x, 64);
    SetParamFromJson<int>(j, "vec_cnt", &vec_cnt, 1);
    c.vec = new VectorConfig(vec_x, vec_cnt);

    SetParamFromJson<int>(j, "sram_bitwidth", &(c.sram_bitwidth), 128);
    SetParamFromJson<string>(j, "dram_config", &(c.dram_config),
                             DEFAULT_DRAM_CONFIG_PATH);
    SetParamFromJson<int>(j, "dram_bw", &(c.dram_bw), HW_DRAM_DEFAULT_BITWIDTH);
    SetParamFromJson<int>(j, "dte_channel_count", &(c.dte_channel_count), 2);
    SetParamFromJson<int>(j, "dte_bit_width", &(c.dte_bit_width), 2048);
    if (c.dte_channel_count <= 0)
        throw std::invalid_argument(
            "core[" + std::to_string(c.id) +
            "].dte_channel_count must be > 0, got " +
            std::to_string(c.dte_channel_count));
    if (c.dte_bit_width <= 0)
        throw std::invalid_argument(
            "core[" + std::to_string(c.id) +
            "].dte_bit_width must be > 0, got " +
            std::to_string(c.dte_bit_width));

    c.control_cores = j.contains("control_cores")
                          ? ParseControlCoresConfig(j.at("control_cores"), c.id)
                          : ControlCoresHWConfig{};
}
