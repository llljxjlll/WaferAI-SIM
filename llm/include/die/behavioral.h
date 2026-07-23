#pragma once

// V4 Behavioral D2D 解析模型。它只描述一条无跨-flow争用的聚合传输：代表包仍实际穿过
// Router，因此 intra_die_hops 只作可解释性输出，**不**在 estimate 中再次记延迟。
// D2D 专属增量由一个责任点计算：三阶段 REQUEST/ACK/DATA 的 link fixed latency，外加 DATA
// 的一次 end-to-end bulk service。多跳选择 pipelined min-cut，而不是每 hop 重复序列化。

#include "die/port.h"
#include "defs/spec.h"
#include "utils/router_utils.h"
#include <limits>
#include <stdexcept>
#include <vector>

struct D2DBehavioralRoute {
    std::vector<int> dies;         // 含源/目的 die
    std::vector<int> link_indices; // forward path，有向 link 下标
    int intra_die_hops = 0;        // 代表 DATA 包实际穿 Router 的 hop 数（只报告，不重复 wait）
};

struct D2DBehavioralEstimate {
    int packets = 0;
    int d2d_hops = 0;
    int intra_die_hops = 0;
    D2DRate effective_rate;
    long long per_phase_link_latency_cycles = 0;
    long long transaction_link_latency_cycles = 0;
    long long first_packet_service_cycles = 0;
    long long bulk_service_cycles = 0;
    long long data_first_cycles = 0;
    long long data_last_cycles = 0;
    long long transaction_d2d_cycles = 0;
};

inline bool D2DRateLess(const D2DRate &a, const D2DRate &b) {
    if (!a.Valid() || !b.Valid())
        throw std::runtime_error("Behavioral D2D rate must satisfy 0<num<=den");
    return (long long)a.num * b.den < (long long)b.num * a.den;
}

inline D2DRate D2DMinRate(const D2DRate &a, const D2DRate &b) {
    return D2DRateLess(a, b) ? a : b;
}

inline long long D2DServiceCycles(long long packets, const D2DRate &rate) {
    if (packets < 1)
        throw std::runtime_error("Behavioral D2D packets must be >= 1");
    if (!rate.Valid())
        throw std::runtime_error("Behavioral D2D rate must satisfy 0<num<=den");
    if (packets > std::numeric_limits<long long>::max() / rate.den)
        throw std::runtime_error("Behavioral D2D service-cycle multiplication overflow");
    const long long n = packets * (long long)rate.den;
    return n / rate.num + (n % rate.num != 0 ? 1 : 0);
}

inline int D2DLocalManhattan(int a, int b) {
    if (a < 0 || a >= CORES_PER_DIE || b < 0 || b >= CORES_PER_DIE)
        throw std::runtime_error("Behavioral D2D route has illegal local tile");
    int ax = a % GRID_X, ay = a / GRID_X;
    int bx = b % GRID_X, by = b / GRID_X;
    int dx = ax > bx ? ax - bx : bx - ax;
    int dy = ay > by ? ay - by : by - ay;
    return dx + dy;
}

inline int D2DNextDie(int die, Directions d) {
    int x = die % DIE_X, y = die / DIE_X;
    if (d == EAST) ++x;
    else if (d == WEST) --x;
    else if (d == NORTH) ++y;
    else if (d == SOUTH) --y;
    if (x < 0 || x >= DIE_X || y < 0 || y >= DIE_Y)
        throw std::runtime_error("Behavioral D2D route leaves die mesh");
    return y * DIE_X + x;
}

inline int D2DDirectedLinkIndexForPort(int die, int port_id, int next_die) {
    for (int i = 0; i < (int)g_d2d_links.size(); ++i) {
        const D2DLink &l = g_d2d_links[i];
        if (l.local_die == die && l.local_port == port_id &&
            l.remote_die == next_die)
            return i;
    }
    return -1;
}

inline D2DBehavioralRoute BuildD2DBehavioralRoute(int source_global,
                                                   int dest_global) {
    if (source_global < 0 || source_global >= TOTAL_CORES || dest_global < 0 ||
        dest_global >= TOTAL_CORES)
        throw std::runtime_error("Behavioral D2D route: illegal endpoint");
    D2DBehavioralRoute out;
    int cur_die = DieOfGlobal(source_global);
    const int dest_die = DieOfGlobal(dest_global);
    int cur_local = LocalOfGlobal(source_global);
    out.dies.push_back(cur_die);
    while (cur_die != dest_die) {
        Directions dir = DieFirstHopDir(cur_die, dest_die);
        int port_id = PortForDir(cur_local, dir);
        if (port_id < 0 || port_id >= (int)g_die_ports.ports.size())
            throw std::runtime_error("Behavioral D2D route: missing C2C exit");
        const D2DPort &p = g_die_ports.ports[port_id];
        if (p.role != ROLE_C2C || p.dir != dir || p.side != dir)
            throw std::runtime_error("Behavioral D2D route: invalid selected C2C exit");
        out.intra_die_hops += D2DLocalManhattan(cur_local, p.tile);
        int next_die = D2DNextDie(cur_die, dir);
        int link_idx = D2DDirectedLinkIndexForPort(cur_die, port_id, next_die);
        if (link_idx < 0)
            throw std::runtime_error("Behavioral D2D route: selected exit has no peer link");
        out.link_indices.push_back(link_idx);
        const D2DLink &link = g_d2d_links[link_idx];
        if (link.remote_port < 0 ||
            link.remote_port >= (int)g_die_ports.ports.size())
            throw std::runtime_error("Behavioral D2D route: illegal remote port");
        cur_die = next_die;
        cur_local = g_die_ports.ports[link.remote_port].tile;
        out.dies.push_back(cur_die);
    }
    out.intra_die_hops += D2DLocalManhattan(cur_local, LocalOfGlobal(dest_global));
    return out;
}

inline D2DBehavioralEstimate EstimateD2DBehavioral(
    int source_global, int dest_global, int packets, const D2DRate &port_rate,
    const D2DRate &link_rate, int link_latency) {
    if (link_latency < 0)
        throw std::runtime_error("Behavioral D2D link_latency must be >= 0");
    D2DBehavioralRoute route = BuildD2DBehavioralRoute(source_global, dest_global);
    D2DBehavioralEstimate e;
    e.packets = packets;
    e.d2d_hops = (int)route.link_indices.size();
    e.intra_die_hops = route.intra_die_hops;
    if (packets < 1)
        throw std::runtime_error("Behavioral D2D packets must be >= 1");
    if (e.d2d_hops == 0)
        return e; // 同 die 不产生 D2D 专属增量
    // 单 lane NoC source/destination cut 均为 1 packet/cycle；V4 MVP 无 striping。
    // TODO(V5): 多 lane/striping 落地时必须把 1/1 替换为沿真实共享 NoC cut 聚合后的
    // 有理数速率；不能使用 lane_count * min(single-lane rates) 这种会高估共享瓶颈的公式。
    e.effective_rate = D2DMinRate(D2DRate{1, 1}, D2DMinRate(port_rate, link_rate));
    e.per_phase_link_latency_cycles = (long long)e.d2d_hops * link_latency;
    e.transaction_link_latency_cycles = 3LL * e.per_phase_link_latency_cycles;
    e.first_packet_service_cycles = D2DServiceCycles(1, e.effective_rate);
    e.bulk_service_cycles = D2DServiceCycles(packets, e.effective_rate);
    e.data_first_cycles = e.per_phase_link_latency_cycles +
                          e.first_packet_service_cycles;
    e.data_last_cycles = e.per_phase_link_latency_cycles + e.bulk_service_cycles;
    e.transaction_d2d_cycles = e.transaction_link_latency_cycles +
                               e.bulk_service_cycles;
    return e;
}
