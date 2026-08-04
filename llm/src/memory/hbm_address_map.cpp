#include "memory/hbm_address_map.h"

#include "die/port.h"
#include "defs/spec.h"
#include "utils/router_utils.h"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>

AddressPolicyConfig g_address_policy;

void ParseAddressPolicy(const nlohmann::json &j) {
    g_address_policy = AddressPolicyConfig{};
    if (!j.contains("address_policy")) {
        g_address_policy.active = false;
        return;
    }
    g_address_policy.active = true;
    const auto &ap = j.at("address_policy");

    static const std::map<std::string, AddressPolicyMode> kModeMap = {
        {"local_interleave", AddressPolicyMode::kLocalInterleave},
        {"numa_local_interleave", AddressPolicyMode::kNumaLocalInterleave},
        {"global_interleave", AddressPolicyMode::kGlobalInterleave},
    };
    std::string mode_str = ap.contains("mode")
                                ? ap.at("mode").get<std::string>()
                                : "numa_local_interleave";
    auto it = kModeMap.find(mode_str);
    if (it == kModeMap.end())
        throw std::runtime_error("address_policy: unknown mode '" +
                                 mode_str + "'");
    g_address_policy.mode = it->second;

    if (ap.contains("home_ranges")) {
        for (const auto &r : ap.at("home_ranges")) {
            HomeRange hr;
            hr.die_id = r.at("die_id").get<int>();
            hr.base = r.at("base").get<unsigned long long>();
            hr.size_bytes = r.at("size_bytes").get<unsigned long long>();
            g_address_policy.home_ranges.push_back(hr);
        }
    }
    if (ap.contains("stack_interleave_bytes"))
        g_address_policy.stack_interleave_bytes =
            ap.at("stack_interleave_bytes").get<unsigned long long>();
    if (ap.contains("channel_interleave_bytes"))
        g_address_policy.channel_interleave_bytes =
            ap.at("channel_interleave_bytes").get<unsigned long long>();
    if (ap.contains("pseudo_channel_interleave_bytes"))
        g_address_policy.pseudo_channel_interleave_bytes =
            ap.at("pseudo_channel_interleave_bytes")
                .get<unsigned long long>();
    if (ap.contains("allow_gaps"))
        g_address_policy.allow_gaps = ap.at("allow_gaps").get<bool>();
}

void ValidateAddressPolicy() {
    if (!g_address_policy.active)
        return;
    if (g_address_policy.stack_interleave_bytes == 0)
        throw std::runtime_error(
            "address_policy: stack_interleave_bytes must be > 0");
    if (g_address_policy.channel_interleave_bytes == 0)
        throw std::runtime_error(
            "address_policy: channel_interleave_bytes must be > 0");
    if (g_address_policy.pseudo_channel_interleave_bytes == 0)
        throw std::runtime_error(
            "address_policy: pseudo_channel_interleave_bytes must be > 0");

    if (g_hbm_stacks.empty())
        throw std::runtime_error(
            "address_policy: distributed_hbm requires at least one HBM stack");

    if (g_address_policy.mode == AddressPolicyMode::kGlobalInterleave) {
        // 当前 global 模式使用等权细粒度交织。容量不相等时等权映射会先越过较小
        // stack；在引入 weighted interleave 前必须拒绝，不能只看总容量。
        unsigned long long capacity = g_hbm_stacks.front().capacity_bytes;
        for (const auto &s : g_hbm_stacks)
            if (s.capacity_bytes != capacity)
                throw std::runtime_error(
                    "address_policy: global_interleave currently requires "
                    "equal exposed capacity on every stack");
        return;
    }

    if (g_address_policy.home_ranges.empty())
        throw std::runtime_error(
            "address_policy: local/NUMA mode requires home_ranges");

    std::map<int, bool> seen_die;
    for (const auto &hr : g_address_policy.home_ranges) {
        if (hr.die_id < 0 || hr.die_id >= DIE_COUNT)
            throw std::runtime_error(
                "address_policy: home range die_id out of range");
        if (hr.size_bytes == 0)
            throw std::runtime_error(
                "address_policy: home range size_bytes must be > 0");
        if (hr.base > std::numeric_limits<unsigned long long>::max() -
                          hr.size_bytes)
            throw std::runtime_error("address_policy: home range end overflows");
        if (hr.base % g_address_policy.stack_interleave_bytes != 0 ||
            hr.size_bytes % g_address_policy.stack_interleave_bytes != 0 ||
            hr.base % g_address_policy.channel_interleave_bytes != 0 ||
            hr.size_bytes % g_address_policy.channel_interleave_bytes != 0)
            throw std::runtime_error(
                "address_policy: home range base/size is not aligned to "
                "stack/channel interleave granularity");
        if (seen_die.count(hr.die_id))
            throw std::runtime_error(
                "address_policy: duplicate home range die_id");
        seen_die[hr.die_id] = true;

        // 容量越界：该 die 暴露的 size_bytes 不能超过其挂载的 stack 容量之和。
        unsigned long long stack_capacity_sum = 0;
        for (const auto &s : g_hbm_stacks)
            if (s.compute_die_id == hr.die_id)
                stack_capacity_sum += s.capacity_bytes;
        if (hr.size_bytes > stack_capacity_sum)
            throw std::runtime_error(
                "address_policy: home range size_bytes exceeds attached "
                "stack capacity");

        // 等权 stack interleave 不能用“容量总和足够”代替逐 stack 校验。
        std::vector<const HBMStackConfig *> stacks;
        for (const auto &s : g_hbm_stacks)
            if (s.compute_die_id == hr.die_id)
                stacks.push_back(&s);
        if (stacks.empty())
            throw std::runtime_error(
                "address_policy: home die has no attached HBM stack");
        unsigned long long stripes =
            hr.size_bytes / g_address_policy.stack_interleave_bytes;
        for (size_t lane = 0; lane < stacks.size(); ++lane) {
            unsigned long long assigned_stripes = stripes / stacks.size();
            if (lane < stripes % stacks.size())
                assigned_stripes++;
            unsigned long long assigned =
                assigned_stripes * g_address_policy.stack_interleave_bytes;
            if (assigned > stacks[lane]->capacity_bytes)
                throw std::runtime_error(
                    "address_policy: equal stack interleave exceeds an "
                    "individual stack capacity");
        }
    }

    // local_interleave 下每个 die 的地址是独立本地空间，不同 die 的 range 允许（且
    // 通常就是）互相重叠——例如两个 die 都从 base=0 开始编址——重叠/空洞检查只对
    // "所有 die 共享同一份全局地址空间" 的 numa_local_interleave 有意义，套到
    // local_interleave 上会把预期内的重叠误判成配置错误，因此这里跳过。
    if (g_address_policy.mode == AddressPolicyMode::kLocalInterleave)
        return;

    std::vector<HomeRange> sorted = g_address_policy.home_ranges;
    std::sort(sorted.begin(), sorted.end(),
              [](const HomeRange &a, const HomeRange &b) {
                  return a.base < b.base;
              });
    if (!g_address_policy.allow_gaps && !sorted.empty() && sorted.front().base != 0)
        throw std::runtime_error(
            "address_policy: leading gap before first home range not allowed");
    for (size_t i = 1; i < sorted.size(); i++) {
        unsigned long long prev_end = sorted[i - 1].base + sorted[i - 1].size_bytes;
        if (sorted[i].base < prev_end)
            throw std::runtime_error("address_policy: home ranges overlap");
        if (!g_address_policy.allow_gaps && sorted[i].base > prev_end)
            throw std::runtime_error(
                "address_policy: gap between home ranges not allowed (set "
                "allow_gaps=true to permit)");
    }
}

namespace {

std::vector<int> DieStackList(int die_id) {
    std::vector<int> ids;
    for (const auto &s : g_hbm_stacks)
        if (s.compute_die_id == die_id)
            ids.push_back(s.stack_id);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

std::vector<int> StackChannelList(int stack_id) {
    std::vector<int> ids;
    for (const auto &c : g_hbm_channels)
        if (c.stack_id == stack_id)
            ids.push_back(c.channel_id);
    std::sort(ids.begin(), ids.end());
    return ids;
}

const HBMStackConfig *FindStack(int stack_id) {
    for (const auto &s : g_hbm_stacks)
        if (s.stack_id == stack_id)
            return &s;
    return nullptr;
}

void FillPseudoChannel(AddressDecodeResult &r) {
    const HBMStackConfig *stack = FindStack(r.stack_id);
    if (!stack)
        throw std::runtime_error("DecodeAddress: selected stack not found");
    auto profile = g_hbm_profiles.find(stack->profile);
    if (profile == g_hbm_profiles.end() ||
        profile->second.pseudo_channels_per_channel <= 0)
        throw std::runtime_error(
            "DecodeAddress: selected stack has invalid HBM profile");
    unsigned long long stripe =
        r.local_address / g_address_policy.pseudo_channel_interleave_bytes;
    r.pseudo_channel_id =
        (int)(stripe % profile->second.pseudo_channels_per_channel);
}

// 标准细粒度交织展开：offset 按 stripe_bytes 轮转分配到 n 路，返回
// (选中的路序号, 该路内部的连续偏移)。
std::pair<size_t, unsigned long long>
Interleave(unsigned long long offset, unsigned long long stripe_bytes,
          size_t n) {
    unsigned long long stripe_index = offset / stripe_bytes;
    unsigned long long intra = offset % stripe_bytes;
    size_t lane = (size_t)(stripe_index % n);
    unsigned long long lane_offset = (stripe_index / n) * stripe_bytes + intra;
    return {lane, lane_offset};
}

} // namespace

AddressDecodeResult DecodeAddress(unsigned long long phys_addr,
                                  int current_die) {
    if (!g_address_policy.active)
        throw std::runtime_error(
            "DecodeAddress: address_policy not configured");
    // Interleave() divides by these; ValidateAddressPolicy() always checks them
    // to be > 0 on the production path, but this function must not silently
    // hand a caller a SIGFPE if it is ever reached without validation having
    // run first (e.g. a future caller that constructs g_address_policy
    // directly for a test).
    if (g_address_policy.stack_interleave_bytes == 0 ||
        g_address_policy.channel_interleave_bytes == 0)
        throw std::runtime_error(
            "DecodeAddress: address_policy interleave granularity is zero "
            "(ValidateAddressPolicy was not run)");

    AddressDecodeResult r;

    if (g_address_policy.mode == AddressPolicyMode::kGlobalInterleave) {
        // 全局 UMA 仍按物理层次做两级交织：先在 stack 之间等权轮转，再在选中
        // stack 的 channel 之间轮转。不能把所有 (stack,channel) 直接拍平，否则
        // channel 数更多的 stack 会获得更大的地址份额，破坏“每颗 stack 暴露容量
        // 相等”的校验前提，并可能把 local_address 映射到实例容量之外。
        std::vector<int> stacks;
        for (const auto &s : g_hbm_stacks)
            stacks.push_back(s.stack_id);
        std::sort(stacks.begin(), stacks.end());
        stacks.erase(std::unique(stacks.begin(), stacks.end()), stacks.end());
        if (stacks.empty())
            throw std::runtime_error(
                "DecodeAddress: no HBM stack configured");
        unsigned long long total_capacity = 0;
        for (const auto &s : g_hbm_stacks) {
            if (total_capacity >
                std::numeric_limits<unsigned long long>::max() - s.capacity_bytes)
                throw std::runtime_error(
                    "DecodeAddress: aggregate HBM capacity overflows");
            total_capacity += s.capacity_bytes;
        }
        if (phys_addr >= total_capacity)
            throw std::runtime_error(
                "DecodeAddress: global address exceeds exposed HBM capacity");
        auto [stack_idx, offset_in_stack] =
            Interleave(phys_addr, g_address_policy.stack_interleave_bytes,
                       stacks.size());
        r.stack_id = stacks[stack_idx];
        std::vector<int> channels = StackChannelList(r.stack_id);
        if (channels.empty())
            throw std::runtime_error(
                "DecodeAddress: selected stack has no HBM channel");
        auto [channel_idx, local] =
            Interleave(offset_in_stack,
                       g_address_policy.channel_interleave_bytes,
                       channels.size());
        r.channel_id = channels[channel_idx];
        r.local_address = local;
        for (const auto &s : g_hbm_stacks)
            if (s.stack_id == r.stack_id) {
                r.home_die = s.compute_die_id;
                break;
            }
        FillPseudoChannel(r);
        return r;
    }

    if (g_address_policy.mode == AddressPolicyMode::kLocalInterleave &&
        (current_die < 0 || current_die >= DIE_COUNT))
        throw std::runtime_error(
            "DecodeAddress: local_interleave requires a valid current_die");

    // NUMA 用全局地址定位；local 只在 current_die 自己的独立 range 中定位。
    const HomeRange *hr = nullptr;
    for (const auto &h : g_address_policy.home_ranges)
        if ((g_address_policy.mode != AddressPolicyMode::kLocalInterleave ||
             h.die_id == current_die) &&
            phys_addr >= h.base && phys_addr - h.base < h.size_bytes) {
            hr = &h;
            break;
        }
    if (!hr)
        throw std::runtime_error(
            "DecodeAddress: address not covered by any home range");
    r.home_die = hr->die_id;
    unsigned long long offset_in_die = phys_addr - hr->base;

    std::vector<int> stacks = DieStackList(hr->die_id);
    if (stacks.empty())
        throw std::runtime_error(
            "DecodeAddress: home die has no attached HBM stack");

    auto [stack_idx, offset_in_stack] = Interleave(
        offset_in_die, g_address_policy.stack_interleave_bytes, stacks.size());
    r.stack_id = stacks[stack_idx];

    std::vector<int> channels = StackChannelList(r.stack_id);
    if (channels.empty())
        throw std::runtime_error(
            "DecodeAddress: stack has no attached channel");

    auto [chan_idx, local] = Interleave(
        offset_in_stack, g_address_policy.channel_interleave_bytes,
        channels.size());
    r.channel_id = channels[chan_idx];
    r.local_address = local;
    FillPseudoChannel(r);
    return r;
}

AddressDecodeResult DecodeAddress(unsigned long long phys_addr) {
    return DecodeAddress(phys_addr, -1);
}

MemRouteResult ResolveMemRoute(int current_die, int ingress_tile,
                               const AddressDecodeResult &target) {
    if (current_die < 0 || current_die >= DIE_COUNT || ingress_tile < 0 ||
        ingress_tile >= CORES_PER_DIE)
        throw std::runtime_error("MemRouteTable: invalid current die/tile");
    if (g_address_policy.mode == AddressPolicyMode::kLocalInterleave &&
        target.home_die != current_die)
        throw std::runtime_error(
            "MemRouteTable: local_interleave forbids remote HBM access");

    const HBMChannelConfig *channel = nullptr;
    for (const auto &c : g_hbm_channels)
        if (c.stack_id == target.stack_id && c.channel_id == target.channel_id) {
            channel = &c;
            break;
        }
    if (!channel)
        throw std::runtime_error(
            "MemRouteTable: target stack/channel has no MEM attachment");

    MemRouteResult route;
    route.current_die = current_die;
    route.home_die = target.home_die;
    route.stack_id = target.stack_id;
    route.channel_id = target.channel_id;
    route.target_mem_tile = channel->mem_tile;
    route.local = current_die == target.home_die;
    if (!route.local) {
        int at = GlobalId(current_die, ingress_tile);
        int home_anchor = GlobalId(target.home_die, channel->mem_tile);
        route.next_c2c_port = CrossDieSelectExit(at, home_anchor);
    }
    return route;
}
