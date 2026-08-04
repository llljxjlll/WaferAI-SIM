#pragma once
// R0：解析并校验每颗 HBM stack 引用的 DRAMSys 配置文件（resolved memspec）。
// 详见 notes/extensions/DRAM/HBM建模计划.md（修订版）六."HBM backend、实例粒度与仲裁"、
// 七."带宽、容量与单位"。
//
// 本文件是分布式 HBM 相关代码里唯一允许直接依赖 DRAMSys::Config/DRAMSys::MemSpec 的
// 地方，避免把这份重量级 SystemC/TLM 依赖带入 die/port.h 这种被广泛 include 的头文件。
#include <cstdint>
#include <string>

struct ResolvedHbmMemSpec {
    std::string memory_type_name; // "HBM2"/"HBM3"/...，由 DRAMSys::Config::MemoryType 转回
    unsigned number_of_channels = 0; // 该 memspec 实际描述的 channel 数
    unsigned pseudo_channels_per_channel = 0;
    unsigned data_bus_width_bits = 0;    // 已按 DRAMSys 规则算好的 width*nbrOfDevices
    double data_rate_transfers_per_cycle = 0.0;
    double tck_ns = 0.0;
    double channel_bandwidth_GBps = 0.0; // dataRate * dataBusWidth / (8 * tCK)
    double instance_bandwidth_GBps = 0.0; // channel 带宽乘实际 nbrOfChannels
    uint64_t capacity_bytes = 0;         // getSimMemSizeInBytes()，该 memspec 描述的实例容量
};

// 加载并解析 dram_config_path 指向的 DRAMSys 顶层配置（resource_directory 固定为
// "../DRAMSys/configs"，与现有 DCache/DRAMSysWrapper 的约定一致，见 workercore.cpp）。
// 只解析 memoryType==HBM2 的 memspec：DRAMSys::DRAMSys::createMemSpec() 是 private
// 成员，本仓库无法调用；这里直接构造 MemSpecHBM2（public 构造函数）。HBM3 在当前
// 这份 DRAMSys checkout 里连 MemSpecHBM3 类都不存在（不是宏开关能打开的临时限制），
// 声明 memoryType=HBM3 的 memspec 在此处抛 std::runtime_error。
ResolvedHbmMemSpec ResolveHbmMemSpec(const std::string &dram_config_path);

// 遍历 g_hbm_stacks，对每颗 stack 的 channel_dram_config 做交叉校验：
//  - profile.generation 必须与 resolved memspec 的 memoryType 一致（mislabeled memspec）；
//  - backend_granularity==kChannel 时 memspec 必须描述恰好 1 个 channel；
//    ==kStack 时 memspec 的 channel 数必须等于 profile.channels_per_stack；
//  - bandwidth_cap_GBps（若手动设置）不超过按 backend_granularity 折算到整颗 stack 的
//    理论带宽（超过即报错，不允许任何倍数容差）；
//  - capacity_bytes 不超过折算到整颗 stack 的 resolved memspec 可寻址容量。
// 非法即抛 std::runtime_error。g_memory_system_active==false 时为 no-op。
void ValidateHbmMemSpecConsistency();
