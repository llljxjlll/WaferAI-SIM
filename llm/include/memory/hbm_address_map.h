#pragma once
// R0：分布式 HBM 地址映射（MemoryAddressMap）——纯函数模块，详见
// notes/extensions/DRAM/HBM建模计划.md（修订版）二."内存语义与地址映射"。
//
// 地址解码顺序固定为 physical address -> home_compute_die -> stack -> channel ->
// pseudo_channel（若显式选择）-> local address。DecodeAddress 不接收 requester/core
// id 参数，同一地址无论谁访问都返回同一结果，确定性由函数签名本身保证，不依赖实现。
//
// 本模块只依赖 die/port.h 里的 g_hbm_stacks/g_hbm_channels 读取 die 归属的 stack/
// channel 列表，不做任何 SystemC/DRAMSys 相关的重活儿依赖。
#include "nlohmann/json.hpp"

#include <cstdint>
#include <string>
#include <vector>

enum class AddressPolicyMode {
    kLocalInterleave,     // 每个 die 独立地址空间，不允许远端访问
    kNumaLocalInterleave, // 默认：全局共享地址空间，地址有 home die，允许远端访问
    kGlobalInterleave,    // 实验用：跨所有 die/stack 细粒度交织，忽略 home_ranges 分片
};

struct HomeRange {
    int die_id = -1;
    unsigned long long base = 0;
    unsigned long long size_bytes = 0;
};

struct AddressPolicyConfig {
    bool active = false; // 无 address_policy 配置时为 false（memory_system 未启用时恒 false）
    AddressPolicyMode mode = AddressPolicyMode::kNumaLocalInterleave;
    std::vector<HomeRange> home_ranges;
    unsigned long long stack_interleave_bytes = 256;   // die 内跨 stack 交织粒度
    unsigned long long channel_interleave_bytes = 256; // stack 内跨 channel 交织粒度
    unsigned long long pseudo_channel_interleave_bytes = 256;
    bool allow_gaps = false; // home_ranges 之间允许空洞（默认不允许，见校验规则）
};

extern AddressPolicyConfig g_address_policy;

struct AddressDecodeResult {
    int home_die = -1;
    int stack_id = -1;
    int channel_id = -1;
    int pseudo_channel_id = -1;
    unsigned long long local_address = 0;
};

// 解析 memory_system.address_policy 子块；不含该 key 时 g_address_policy.active=false。
// 非法配置抛 std::runtime_error（由 ValidateAddressPolicy 或本函数触发的即时结构错误）。
void ParseAddressPolicy(const nlohmann::json &j);

// 启动期校验：home_ranges 无重叠、按 allow_gaps 决定是否允许空洞、size_bytes 必须是
// channel_interleave_bytes 的整数倍（否则视为 stripe 未对齐）、每个 die 暴露的容量不
// 超过其挂载的 stack 容量之和。非法即抛 std::runtime_error。
void ValidateAddressPolicy();

// 纯函数：物理地址 -> {home_die, stack_id, channel_id, pseudo_channel_id, local_address}。
// 不接收 requester 参数，因此同一地址任何时候、任何调用方得到的结果都相同。
// g_address_policy.active==false 时抛 std::runtime_error（尚未配置分布式 HBM 地址策略）。
AddressDecodeResult DecodeAddress(unsigned long long phys_addr);

// local_interleave 需要请求所在 die 才能消除独立地址空间的歧义；NUMA/global 模式下
// current_die 仅用于接口统一，不改变同一物理地址的 home。
AddressDecodeResult DecodeAddress(unsigned long long phys_addr, int current_die);

struct MemRouteResult {
    int current_die = -1;
    int home_die = -1;
    int stack_id = -1;
    int channel_id = -1;
    int target_mem_tile = -1; // home die 内局部 tile
    int next_c2c_port = -1;   // 本地命中时为 -1；远端时为当前 die 的 C2C 出口
    bool local = false;
};

// 地址归属与路径选择严格分离。target 由 DecodeAddress 决定；本函数只根据 current_die
// 和 ingress_tile 选择本地 MEM anchor 或下一跳 C2C port，不改变 target 的 home。
MemRouteResult ResolveMemRoute(int current_die, int ingress_tile,
                               const AddressDecodeResult &target);
