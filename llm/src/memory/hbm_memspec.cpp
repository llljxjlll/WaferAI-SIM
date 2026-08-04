#include "memory/hbm_memspec.h"

#include "die/port.h"

#include "DRAMSys/config/DRAMSysConfiguration.h"
#include "DRAMSys/configuration/memspec/MemSpecHBM2.h"

#include <stdexcept>
#include <cmath>

// 不能用 DRAMSys::DRAMSys::createMemSpec()：它是该类的 private 静态成员，本仓库外部
// 代码无法调用。这里直接构造具体子类（构造函数是 public 的），只覆盖分布式 HBM 会
// 用到的世代。DRAMSys 当前这份 checkout 里根本没有 MemSpecHBM3.h（不是宏关掉，是
// 类本身不存在），所以 HBM3 在这个 build 下是硬性不支持，不是可以绕过的临时限制。
namespace {

constexpr const char *kResourceDir = "../DRAMSys/configs";

std::string MemoryTypeName(DRAMSys::Config::MemoryType t) {
    using MT = DRAMSys::Config::MemoryType;
    switch (t) {
    case MT::DDR3: return "DDR3";
    case MT::DDR4: return "DDR4";
    case MT::DDR5: return "DDR5";
    case MT::LPDDR4: return "LPDDR4";
    case MT::LPDDR5: return "LPDDR5";
    case MT::WideIO: return "WIDEIO_SDR";
    case MT::WideIO2: return "WIDEIO2";
    case MT::GDDR5: return "GDDR5";
    case MT::GDDR5X: return "GDDR5X";
    case MT::GDDR6: return "GDDR6";
    case MT::HBM2: return "HBM2";
    case MT::HBM3: return "HBM3";
    case MT::STTMRAM: return "STTMRAM";
    default: return "Invalid";
    }
}

} // namespace

ResolvedHbmMemSpec ResolveHbmMemSpec(const std::string &dram_config_path) {
    DRAMSys::Config::Configuration cfg =
        DRAMSys::Config::from_path(dram_config_path, kResourceDir);

    if (cfg.memspec.memoryType != DRAMSys::Config::MemoryType::HBM2)
        throw std::runtime_error(
            "hbm_memspec: '" + dram_config_path + "' declares memoryType=" +
            MemoryTypeName(cfg.memspec.memoryType) +
            ", but this build only resolves HBM2 memspecs (HBM3 support is "
            "absent from this DRAMSys checkout, not just compiled out)");

    std::unique_ptr<const DRAMSys::MemSpec> resolved =
        std::make_unique<const DRAMSys::MemSpecHBM2>(cfg.memspec);

    ResolvedHbmMemSpec r;
    r.memory_type_name = MemoryTypeName(cfg.memspec.memoryType);
    r.number_of_channels = resolved->numberOfChannels;
    r.pseudo_channels_per_channel = resolved->pseudoChannelsPerChannel;
    r.data_bus_width_bits = resolved->dataBusWidth;
    r.data_rate_transfers_per_cycle = resolved->dataRate;

    // 带宽(bytes/s) = dataRate(transfers/cycle) * dataBusWidth(bit) / 8 / tCK(s)。
    // dataBusWidth 已经是 DRAMSys 按 width*nbrOfDevices 算好的实际总线位宽（七.2 提醒的
    // 陷阱：不能只读 memspec 原始 "width" 字段，必须用 resolved 值）。
    double tck_seconds = resolved->tCK.to_seconds();
    if (tck_seconds <= 0)
        throw std::runtime_error("hbm_memspec: '" + dram_config_path +
                                 "' resolved tCK <= 0");
    double bytes_per_second = (double)resolved->dataRate *
                              resolved->dataBusWidth / 8.0 / tck_seconds;
    r.channel_bandwidth_GBps = bytes_per_second / 1e9;
    r.instance_bandwidth_GBps =
        r.channel_bandwidth_GBps * r.number_of_channels;
    r.tck_ns = tck_seconds * 1e9;
    r.capacity_bytes = resolved->getSimMemSizeInBytes();
    return r;
}

void ValidateHbmMemSpecConsistency() {
    if (!g_memory_system_active)
        return;

    for (const auto &s : g_hbm_stacks) {
        auto pit = g_hbm_profiles.find(s.profile);
        if (pit == g_hbm_profiles.end())
            throw std::runtime_error(
                "hbm_memspec: stack " + std::to_string(s.stack_id) +
                " references unknown profile '" + s.profile + "'");
        const HBMProfile &profile = pit->second;

        ResolvedHbmMemSpec spec = ResolveHbmMemSpec(s.channel_dram_config);

        if (spec.memory_type_name != profile.generation)
            throw std::runtime_error(
                "hbm_memspec: stack " + std::to_string(s.stack_id) +
                " profile.generation='" + profile.generation +
                "' does not match memspec memoryType='" +
                spec.memory_type_name + "' (" + s.channel_dram_config + ")");

        int expected_channels =
            s.backend_granularity == HBMBackendGranularity::kChannel
                ? 1
                : profile.channels_per_stack;
        if ((int)spec.number_of_channels != expected_channels)
            throw std::runtime_error(
                "hbm_memspec: stack " + std::to_string(s.stack_id) +
                " backend_granularity is inconsistent with memspec "
                "nbrOfChannels=" +
                std::to_string(spec.number_of_channels) + " (expected " +
                std::to_string(expected_channels) + ")");

        if ((int)spec.pseudo_channels_per_channel !=
            profile.pseudo_channels_per_channel)
            throw std::runtime_error(
                "hbm_memspec: stack " + std::to_string(s.stack_id) +
                " pseudo-channel count does not match profile");

        unsigned expected_stack_bus_width =
            spec.data_bus_width_bits * profile.channels_per_stack;
        if ((unsigned)profile.stack_bus_width_bits != expected_stack_bus_width)
            throw std::runtime_error(
                "hbm_memspec: stack " + std::to_string(s.stack_id) +
                " profile stack_bus_width_bits does not match resolved "
                "per-channel dataBusWidth");

        double resolved_pin_rate =
            spec.data_rate_transfers_per_cycle / spec.tck_ns;
        double rate_scale = std::max(1.0, std::fabs(resolved_pin_rate));
        if (std::fabs(profile.data_rate_gbps_per_pin - resolved_pin_rate) >
            1e-6 * rate_scale)
            throw std::runtime_error(
                "hbm_memspec: stack " + std::to_string(s.stack_id) +
                " profile data_rate_gbps_per_pin does not match resolved "
                "dataRate/tCK");

        // spec.* 描述的是"一个 DRAMSys 实例"的带宽/容量：kChannel 粒度下那只是整颗
        // stack 的 1/channels_per_stack，必须按 channels_per_stack 折算回整颗 stack
        // 才能和 HBMStackConfig 里以 stack 为单位声明的 bandwidth_cap_GBps/
        // capacity_bytes 做同口径比较，否则会系统性地把 kChannel 粒度的合法配置误判超限。
        double stack_bandwidth_GBps = spec.instance_bandwidth_GBps;
        uint64_t stack_capacity_bytes = spec.capacity_bytes;
        if (s.backend_granularity == HBMBackendGranularity::kChannel) {
            stack_bandwidth_GBps *= profile.channels_per_stack;
            stack_capacity_bytes *= (uint64_t)profile.channels_per_stack;
        }

        if (s.bandwidth_cap_GBps >= 0 &&
            s.bandwidth_cap_GBps > stack_bandwidth_GBps)
            throw std::runtime_error(
                "hbm_memspec: stack " + std::to_string(s.stack_id) +
                " bandwidth_cap_GBps exceeds resolved memspec theoretical "
                "bandwidth (no tolerance margin is allowed)");

        if (s.capacity_bytes > stack_capacity_bytes)
            throw std::runtime_error(
                "hbm_memspec: stack " + std::to_string(s.stack_id) +
                " capacity_bytes exceeds resolved memspec addressable "
                "capacity");
    }
}
