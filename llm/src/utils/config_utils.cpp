#include <cmath>
#include <regex>

#include "common/config.h"
#include "defs/global.h"
#include "defs/spec.h"
#include "die/port.h"
#include "utils/config_utils.h"
#include "utils/print_utils.h"

int GetDefinedParam(string var) {
    for (auto v : vtable) {
        if (v.first == var)
            return v.second;
    }

    LOG_ERROR(config_utils.cpp) << "Undefined variable " << var;
    return 1;
}

void ParseSimulationType(json j) {
    std::unordered_map<string, SIM_MODE> sim_mode_map = {
        {"dataflow", SIM_DATAFLOW},
        {"gpu", SIM_GPU},
        {"sched_pd", SIM_PD},
        {"sched_pds", SIM_PDS},
        {"gpu_pd", SIM_GPU_PD}};

    if (j.contains("mode")) {
        auto mode = j["mode"];

        if (sim_mode_map.find(mode) != sim_mode_map.end())
            SYSTEM_MODE = sim_mode_map[mode];
        else
            LOG_ERROR(config_utils.cpp)
                << "Unsupported simulation mode " << mode;
    } else
        SYSTEM_MODE = SIM_DATAFLOW;

    LOG_INFO(SYSTEM) << "System running in simulation mode "
                     << GetEnumSimulationMode(SYSTEM_MODE);

    if (SYSTEM_MODE == SIM_GPU) {
        if (j.contains("chips"))
            CORE_PER_SM = j["chips"][0]["core_per_sm"];
        if (USE_L1L2_CACHE != 1)
            LOG_ERROR(config_utils.cpp)
                << "L1L2 cache unavailable for GPU simulation.";
    }
}

void ParseHardwareConfig(json j) {
    if (j.contains("x"))
        GRID_X = j["x"];
    // die 内 mesh：支持矩形（可选 "y"），缺省为方阵（回归安全）
    GRID_Y = j.contains("y") ? int(j["y"]) : GRID_X;

    // die 级 mesh：可选 "die":{"x":..,"y":..}，缺省单 die（DIE_COUNT=1）
    DIE_X = 1;
    DIE_Y = 1;
    if (j.contains("die")) {
        auto conf_die = j["die"];
        if (conf_die.contains("x"))
            DIE_X = conf_die["x"];
        DIE_Y = conf_die.contains("y") ? int(conf_die["y"]) : DIE_X;
    }
    // 先用宽整数校验维度与容量（<=16-bit），再计算 int 全局量，
    // 避免极大非法配置在赋值时发生有符号 int 溢出（UB）。
    ValidateAddressSpace();
    GRID_SIZE = GRID_X * GRID_Y;
    DIE_COUNT = DIE_X * DIE_Y;
    CORES_PER_DIE = GRID_SIZE;
    TOTAL_CORES = CORES_PER_DIE * DIE_COUNT;
    HOST_ENDPOINT_ID = TOTAL_CORES;
    HOST_LANES = GRID_Y * DIE_COUNT; // 每 die 西边缘 GRID_Y 个 HOST lane × die 数

    if (j.contains("noc")) {
        auto conf_noc = j["noc"];
        if (conf_noc.contains("noc_payload_per_cycle"))
            HW_NOC_PAYLOAD_PER_CYCLE = conf_noc["noc_payload_per_cycle"];
    }

    if (j.contains("dte")) {
        auto conf_dte = j["dte"];
        if (conf_dte.contains("gamma_ns"))
            HW_DTE_GAMMA_NS = conf_dte["gamma_ns"];
        if (conf_dte.contains("tau_launch_avg_ns"))
            HW_DTE_TAU_LAUNCH_NS = conf_dte["tau_launch_avg_ns"];
        if (conf_dte.contains("command_slots_per_channel"))
            HW_DTE_COMMAND_SLOTS_PER_CHANNEL =
                conf_dte["command_slots_per_channel"];
        if (conf_dte.contains("pending_queue_depth"))
            HW_DTE_PENDING_QUEUE_DEPTH = conf_dte["pending_queue_depth"];
        if (conf_dte.contains("spm_read_width_bits"))
            HW_DTE_SPM_READ_WIDTH_BITS = conf_dte["spm_read_width_bits"];
        if (conf_dte.contains("spm_write_width_bits"))
            HW_DTE_SPM_WRITE_WIDTH_BITS = conf_dte["spm_write_width_bits"];
        if (conf_dte.contains("axi_read_width_bits"))
            HW_DTE_AXI_READ_WIDTH_BITS = conf_dte["axi_read_width_bits"];
        if (conf_dte.contains("axi_write_width_bits"))
            HW_DTE_AXI_WRITE_WIDTH_BITS = conf_dte["axi_write_width_bits"];
        if (conf_dte.contains("launch_energy_pj"))
            HW_DTE_LAUNCH_ENERGY_PJ = conf_dte["launch_energy_pj"];
        if (conf_dte.contains("spm_energy_pj_per_bit"))
            HW_DTE_SPM_ENERGY_PJ_PER_BIT =
                conf_dte["spm_energy_pj_per_bit"];
        if (conf_dte.contains("axi_energy_pj_per_bit"))
            HW_DTE_AXI_ENERGY_PJ_PER_BIT =
                conf_dte["axi_energy_pj_per_bit"];
        if (conf_dte.contains("base_area_um2"))
            HW_DTE_BASE_AREA_UM2 = conf_dte["base_area_um2"];
        if (conf_dte.contains("channel_area_um2"))
            HW_DTE_CHANNEL_AREA_UM2 = conf_dte["channel_area_um2"];
        if (conf_dte.contains("command_slot_area_um2"))
            HW_DTE_COMMAND_SLOT_AREA_UM2 =
                conf_dte["command_slot_area_um2"];
        if (conf_dte.contains("port_bit_area_um2"))
            HW_DTE_PORT_BIT_AREA_UM2 = conf_dte["port_bit_area_um2"];
    }
    if (HW_DTE_GAMMA_NS < 0)
        throw std::invalid_argument("dte.gamma_ns must be >= 0");
    if (HW_DTE_TAU_LAUNCH_NS < 0)
        throw std::invalid_argument("dte.tau_launch_avg_ns must be >= 0");
    if (HW_DTE_COMMAND_SLOTS_PER_CHANNEL <= 0)
        throw std::invalid_argument(
            "dte.command_slots_per_channel must be > 0");
    if (HW_DTE_PENDING_QUEUE_DEPTH < 0)
        throw std::invalid_argument("dte.pending_queue_depth must be >= 0");
    if (HW_DTE_SPM_READ_WIDTH_BITS <= 0 ||
        HW_DTE_SPM_WRITE_WIDTH_BITS <= 0 ||
        HW_DTE_AXI_READ_WIDTH_BITS <= 0 ||
        HW_DTE_AXI_WRITE_WIDTH_BITS <= 0)
        throw std::invalid_argument(
            "dte SPM/AXI port widths must all be > 0");
    const double dte_nonnegative[] = {
        HW_DTE_LAUNCH_ENERGY_PJ, HW_DTE_SPM_ENERGY_PJ_PER_BIT,
        HW_DTE_AXI_ENERGY_PJ_PER_BIT, HW_DTE_BASE_AREA_UM2,
        HW_DTE_CHANNEL_AREA_UM2, HW_DTE_COMMAND_SLOT_AREA_UM2,
        HW_DTE_PORT_BIT_AREA_UM2};
    for (double value : dte_nonnegative)
        if (!std::isfinite(value) || value < 0.0)
            throw std::invalid_argument(
                "dte energy/area coefficients must be finite and >= 0");

    if (j.contains("operand")) {
        auto conf_operand = j["operand"];
        if (conf_operand.contains("core_credit"))
            HW_CORE_CREDIT = conf_operand["core_credit"];
        if (conf_operand.contains("pd_ratio"))
            HW_PD_RATIO = conf_operand["pd_ratio"];
        if (conf_operand.contains("comp_util"))
            HW_COMP_UTIL = conf_operand["comp_util"];
    }

    if (j.contains("memory")) {
        auto conf_memory = j["memory"];
        if (conf_memory.contains("sram_size"))
            HW_SRAM_SIZE = conf_memory["sram_size"];

        if (conf_memory.contains("dram_default_bitwidth"))
            HW_DRAM_DEFAULT_BITWIDTH = conf_memory["dram_default_bitwidth"];

        if (conf_memory.contains("beha_dram_util"))
            HW_BEHA_DRAM_UTIL = conf_memory["beha_dram_util"];
    }

    if (j.contains("gpu")) {
        auto conf_gpu = j["gpu"];
        if (conf_gpu.contains("dram_bandwidth"))
            GPU_DRAM_BANDWIDTH = conf_gpu["dram_bandwidth"];
        if (conf_gpu.contains("dram_burst_size"))
            GPU_DRAM_BURST_SIZE = conf_gpu["dram_burst_size"];
        if (conf_gpu.contains("dram_aligned"))
            GPU_DRAM_ALIGNED = conf_gpu["dram_aligned"];
    }

    auto config_cores = j["cores"];
    CoreHWConfig sample = config_cores[0];

    for (auto core : config_cores) {
        CoreHWConfig c = core;
        for (int i = sample.id + 1; i < c.id; i++) {
            ExuConfig *exu =
                new ExuConfig(MAC_Array, sample.exu->x_dims, sample.exu->count);
            SfuConfig *sfu = new SfuConfig(Linear, sample.sfu->x_dims);
            VectorConfig *vec =
                new VectorConfig(sample.vec->x_dims, sample.vec->count);
            g_core_hw_config.push_back(make_pair(
                i, new CoreHWConfig(i, exu, sfu, vec, sample.dram_config,
                                    sample.dram_bw, sample.sram_bitwidth,
                                    sample.dte_channel_count,
                                    sample.dte_bit_width)));
        }

        ExuConfig *exu = new ExuConfig(MAC_Array, c.exu->x_dims, c.exu->count);
        SfuConfig *sfu = new SfuConfig(Linear, c.sfu->x_dims);
        VectorConfig *vec = new VectorConfig(c.vec->x_dims, c.vec->count);
        g_core_hw_config.push_back(
            make_pair(c.id, new CoreHWConfig(c.id, exu, sfu, vec, c.dram_config,
                                             c.dram_bw, c.sram_bitwidth,
                                             c.dte_channel_count,
                                             c.dte_bit_width)));

        delete sample.exu;
        delete sample.sfu;
        delete sample.vec;
        sample = c;
        sample.exu = new ExuConfig(MAC_Array, c.exu->x_dims, c.exu->count);
        sample.sfu = new SfuConfig(Linear, c.sfu->x_dims);
        sample.vec = new VectorConfig(c.vec->x_dims, c.vec->count);
    }

    for (int i = sample.id + 1; i < GRID_SIZE; i++) {
        ExuConfig *exu =
            new ExuConfig(MAC_Array, sample.exu->x_dims, sample.exu->count);
        SfuConfig *sfu = new SfuConfig(Linear, sample.sfu->x_dims);
        VectorConfig *vec =
            new VectorConfig(sample.vec->x_dims, sample.vec->count);
        g_core_hw_config.push_back(make_pair(
            i, new CoreHWConfig(i, exu, sfu, vec, sample.dram_config,
                                sample.dram_bw, sample.sram_bitwidth,
                                sample.dte_channel_count,
                                sample.dte_bit_width)));
    }

    for (auto core : g_core_hw_config)
        core.second->printSelf();

    // 解析并校验 D2D 端口配置（无 "die_ports" 时为 no-op，单 die 兼容）
    ParseDiePorts(j);
    // 构造 HOST 物理挂载表（legacy=西边缘每行一 lane；须在维度常量 + 端口就绪后）。
    // 强一致：lane 数以挂载表为唯一真源，HOST_LANES 从表派生，Monitor/MemInterface
    // 的数组与循环都以 HOST_LANES 为界，杜绝 config 改变 lane 数后二者不一致。
    BuildHostAttach();
    HOST_LANES = g_host_attach.n_lanes;
    ValidateHostAttach(); // 启动期结构校验（尺寸/tile/同 die/HOST_LANES 一致）
}

void ParseSimulationConfig(json j) {
    // 设置仿真相关参数
    if (j.contains("ttf_file"))
        SPEC_TTF_FILE = j["ttf_file"];

    if (j.contains("operand")) {
        auto conf_operand = j["operand"];
        if (conf_operand.contains("use_pref_gemm"))
            SPEC_USE_PERF_GEMM = conf_operand["use_perf_gemm"];
        if (conf_operand.contains("load_static_as_tile"))
            SPEC_LOAD_STATIC = conf_operand["load_static_as_tile"];
    }

    if (j.contains("memory")) {
        auto conf_memory = j["memory"];
        if (conf_memory.contains("use_beha_sram"))
            SPEC_USE_BEHA_SRAM = conf_memory["use_beha_sram"];
        if (conf_memory.contains("use_beha_dram"))
            SPEC_USE_BEHA_DRAM = conf_memory["use_beha_dram"];
        if (conf_memory.contains("kvcache_spill"))
            SPEC_KVCACHE_SPILL = conf_memory["kvcache_spill"];
        if (conf_memory.contains("use_dramsys"))
            SPEC_USE_DRAMSYS = conf_memory["use_dramsys"];
    }

    if (j.contains("noc")) {
        auto conf_noc = j["noc"];
        if (conf_noc.contains("use_beha_noc"))
            SPEC_USE_BEHA_NOC = conf_noc["use_beha_noc"];
        if (conf_noc.contains("router_pipe"))
            SPEC_ROUTER_PIPE = conf_noc["router_pipe"];
        if (conf_noc.contains("fast_warmup"))
            SPEC_FAST_WARMUP = conf_noc["fast_warmup"];
        if (conf_noc.contains("send_recv_parallel"))
            SPEC_SEND_RECV_PARALLEL = conf_noc["send_recv_parallel"];
    }

    if (j.contains("dte")) {
        auto conf_dte = j["dte"];
        if (conf_dte.contains("use_beha_dte"))
            SPEC_USE_BEHA_DTE = conf_dte["use_beha_dte"];
        if (conf_dte.contains("streaming"))
            SPEC_DTE_STREAMING = conf_dte["streaming"];
        if (conf_dte.contains("async"))
            SPEC_DTE_ASYNC = conf_dte["async"];
        if (conf_dte.contains("aggregation"))
            SPEC_DTE_AGGREGATION = conf_dte["aggregation"];
        if (conf_dte.contains("fine_grained_resources"))
            SPEC_DTE_V4_RESOURCES = conf_dte["fine_grained_resources"];
        if (conf_dte.contains("aggregation_max_descriptors"))
            DTE_AGGREGATION_MAX_DESCRIPTORS =
                conf_dte["aggregation_max_descriptors"];
        if (conf_dte.contains("aggregation_max_bytes"))
            DTE_AGGREGATION_MAX_BYTES =
                conf_dte["aggregation_max_bytes"];
        if (conf_dte.contains("aggregation_timeout_ns"))
            DTE_AGGREGATION_TIMEOUT_NS =
                conf_dte["aggregation_timeout_ns"];
        if (conf_dte.contains("aggregation_address_block_bytes"))
            DTE_AGGREGATION_ADDRESS_BLOCK_BYTES =
                conf_dte["aggregation_address_block_bytes"];
    }
    if (SPEC_DTE_STREAMING && !SPEC_USE_BEHA_DTE)
        throw std::invalid_argument(
            "dte.streaming requires dte.use_beha_dte=true");
    if (SPEC_DTE_STREAMING && SYSTEM_MODE != SIM_DATAFLOW)
        throw std::invalid_argument(
            "DTE V2b streaming is supported only for dataflow workloads");
    if (SPEC_DTE_STREAMING && SPEC_SEND_RECV_PARALLEL)
        throw std::invalid_argument(
            "DTE V2b streaming does not yet support the parallel dispatcher");
    if (SPEC_DTE_ASYNC && !SPEC_USE_BEHA_DTE)
        throw std::invalid_argument(
            "dte.async requires dte.use_beha_dte=true");
    if (SPEC_DTE_ASYNC && SYSTEM_MODE != SIM_DATAFLOW)
        throw std::invalid_argument(
            "DTE V3a async mode is supported only for dataflow workloads");
    if (SPEC_DTE_ASYNC && SPEC_DTE_STREAMING)
        throw std::invalid_argument(
            "DTE V3a async mode and V2b streaming are mutually exclusive");
    if (SPEC_DTE_ASYNC && SPEC_SEND_RECV_PARALLEL)
        throw std::invalid_argument(
            "DTE V3a async mode requires the sequential dispatcher");
    if (SPEC_DTE_V4_RESOURCES && !SPEC_DTE_ASYNC)
        throw std::invalid_argument(
            "DTE V4 fine_grained_resources requires dte.async=true");
    if (SPEC_DTE_V4_RESOURCES &&
        HW_DTE_COMMAND_SLOTS_PER_CHANNEL != 2)
        throw std::invalid_argument(
            "DTE V4 requires hardware command_slots_per_channel=2");
    if (SPEC_DTE_V4_RESOURCES && HW_DTE_PENDING_QUEUE_DEPTH <= 0)
        throw std::invalid_argument(
            "DTE V4 requires hardware pending_queue_depth > 0");
    if (SPEC_DTE_AGGREGATION && !SPEC_DTE_ASYNC)
        throw std::invalid_argument(
            "DTE V3b aggregation requires dte.async=true");
    if (SPEC_DTE_AGGREGATION &&
        DTE_AGGREGATION_MAX_DESCRIPTORS < 2)
        throw std::invalid_argument(
            "DTE V3b aggregation_max_descriptors must be >= 2");
    if (SPEC_DTE_AGGREGATION && DTE_AGGREGATION_MAX_BYTES <= 0)
        throw std::invalid_argument(
            "DTE V3b aggregation_max_bytes must be > 0");
    if (SPEC_DTE_AGGREGATION && DTE_AGGREGATION_TIMEOUT_NS <= 0)
        throw std::invalid_argument(
            "DTE V3b aggregation_timeout_ns must be > 0");
    if (SPEC_DTE_AGGREGATION &&
        DTE_AGGREGATION_ADDRESS_BLOCK_BYTES <= 0)
        throw std::invalid_argument(
            "DTE V3b aggregation_address_block_bytes must be > 0");
    if (SPEC_USE_BEHA_DTE && SPEC_SEND_RECV_PARALLEL &&
        SYSTEM_MODE != SIM_DATAFLOW)
        throw std::invalid_argument(
            "DTE V2a parallel mode is supported only for dataflow workloads");

    if (j.contains("gpu")) {
        auto conf_gpu = j["gpu"];
        if (conf_gpu.contains("use_inner_mm"))
            GPU_USE_INNER_MM = conf_gpu["use_inner_mm"];
        if (conf_gpu.contains("cache_log"))
            GPU_CACHE_LOG = conf_gpu["cache_log"];
        if (conf_gpu.contains("dram_config_file"))
            GPU_DRAM_CONFIG_FILE = conf_gpu["dram_config_file"];
    }

    if (j.contains("log")) {
        auto conf_log = j["log"];
        if (conf_log.contains("log_level"))
            LogConfig::CONFIG_LOG_LEVEL = LogLevel(conf_log["log_level"]);
        if (conf_log.contains("verbose_debug"))
            LogConfig::CONFIG_VERBOSE_DEBUG = conf_log["verbose_debug"];
        if (conf_log.contains("colored"))
            LogConfig::CONFIG_LOG_COLORED = conf_log["colored"];
    }
}

void ParseMemorySpec(string filename) {
    std::ifstream infile(filename);

    if (!infile.is_open()) {
        LOG_WARN(config_utils.cpp) << "Cannot open file: " << filename;
        return;
    }

    std::string line;

    while (std::getline(infile, line)) {
        if (line.empty())
            continue;

        std::stringstream ss(line);
        std::string x_str, y_str;

        if (std::getline(ss, x_str, ':') && std::getline(ss, y_str)) {
            int x = std::stoi(x_str);
            int y = std::stoi(y_str);
            g_core_remap[x] = y;
            LOG_DEBUG(CONFIG) << "Remap core " << x << " to " << y;
        }
    }

    infile.close();
}

void CoreConfigRemap(vector<pair<int, int>> &source_info,
                     vector<CoreConfig> &coreconfigs) {
    auto get_or_key = [&](int x) -> int {
        auto it = g_core_remap.find(x);
        return (it != g_core_remap.end()) ? it->second : x;
    };

    for (auto &config : coreconfigs) {
        int old_id = config.id;
        config.id = get_or_key(config.id);

        if (config.prim_copy != -1)
            config.prim_copy = get_or_key(config.prim_copy);

        for (auto &work : config.worklist) {
            if (work.recv_tag == old_id)
                work.recv_tag = config.id;

            for (auto &cast : work.cast) {
                if (cast.tag == cast.dest)
                    cast.tag = get_or_key(cast.dest);

                cast.dest = get_or_key(cast.dest);
            }
        }
    }

    for (auto &source : source_info) {
        source.first = get_or_key(source.first);
    }
}