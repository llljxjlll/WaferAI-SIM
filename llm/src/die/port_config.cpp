#include "die/port.h"
#include "defs/spec.h"
#include "macros/macros.h"
#include "utils/print_utils.h"
#include <cstdlib>
#include <map>
#include <set>
#include <tuple>
#include <utility>

D2DPortTable g_die_ports;
std::vector<D2DLink> g_d2d_links;
HostAttachTable g_host_attach;

std::vector<long> g_host_lane_done;
std::vector<long> g_host_lane_ack;
long g_host_lane_mismatch = 0;
std::map<int, int> g_host_done_src;
std::map<std::pair<int, int>, int> g_host_ack_sig;
long g_d2d_link_in_pkts = 0;
long g_d2d_link_out_pkts = 0;
void ResetD2DLinkStats() {
    g_d2d_link_in_pkts = 0;
    g_d2d_link_out_pkts = 0;
}
void ResetHostLaneStats(int n_lanes) {
    g_host_lane_done.assign(n_lanes, 0);
    g_host_lane_ack.assign(n_lanes, 0);
    g_host_lane_mismatch = 0;
    g_host_done_src.clear();
    g_host_ack_sig.clear();
}

static Directions SideFromStr(const std::string &s) {
    if (s == "N")
        return NORTH;
    if (s == "S")
        return SOUTH;
    if (s == "E")
        return EAST;
    if (s == "W")
        return WEST;
    throw std::runtime_error("die_ports: unknown side '" + s + "'");
}

static PortRole RoleFromStr(const std::string &s) {
    if (s == "host")
        return ROLE_HOST;
    if (s == "mem")
        return ROLE_MEM;
    if (s == "c2c")
        return ROLE_C2C;
    throw std::runtime_error("die_ports: unknown role '" + s + "'");
}

static int EdgeLen(Directions side) {
    return (side == NORTH || side == SOUTH) ? GRID_X : GRID_Y;
}

// (side, idx) -> 局部核 id。约定：N/S 沿 x 数，W/E 沿 y 数。
static int TileFor(Directions side, int idx) {
    switch (side) {
    case NORTH:
        return (GRID_Y - 1) * GRID_X + idx;
    case SOUTH:
        return idx;
    case WEST:
        return idx * GRID_X;
    case EAST:
        return idx * GRID_X + (GRID_X - 1);
    default:
        throw std::runtime_error("die_ports: bad side in TileFor");
    }
}

// 解析 {role, dir?} 规格，返回 (role, dir)。role==C2C 时校验 MVP 约束 dir==side。
static void ParseRoleSpec(const D2DJson &spec, Directions side, PortRole &role,
                          Directions &dir) {
    role = RoleFromStr(spec.at("role").get<std::string>());
    if (role == ROLE_C2C) {
        if (spec.contains("dir")) {
            dir = SideFromStr(spec.at("dir").get<std::string>());
            if (dir != side)
                throw std::runtime_error(
                    "die_ports: MVP requires C2C dir == side (side/dir "
                    "mismatch on one port)");
        } else {
            dir = side; // 缺省 dir=side
        }
    } else {
        dir = side; // 非 C2C，dir 无意义，填 side
    }
}

void ParseDiePorts(const D2DJson &hw_json) {
    g_die_ports = D2DPortTable{};
    g_d2d_links.clear(); // 重复解析/自测时清空上次结果
    if (!hw_json.contains("die_ports")) {
        g_die_ports.active = false;
        // 单 die 且无端口配置 = 兼容旧行为；port_for_host 走 legacy（-1）
        g_die_ports.port_for_host.assign(CORES_PER_DIE, -1);
        return;
    }
    g_die_ports.active = true;
    const D2DJson &dp = hw_json.at("die_ports");

    // c2c 物理参数（端口级默认）+ 范围校验
    int c2c_bw = 1, c2c_lat = 0, c2c_buf = 1;
    if (dp.contains("c2c")) {
        const D2DJson &c = dp.at("c2c");
        if (c.contains("bw_per_cycle"))
            c2c_bw = c.at("bw_per_cycle");
        if (c.contains("link_bw"))
            c2c_bw = c.at("link_bw");
        if (c.contains("latency"))
            c2c_lat = c.at("latency");
        if (c.contains("buffer_depth"))
            c2c_buf = c.at("buffer_depth");
    }
    if (c2c_bw < 1)
        throw std::runtime_error(
            "die_ports.c2c: link_bw/bw_per_cycle must be >= 1");
    if (c2c_lat < 0)
        throw std::runtime_error("die_ports.c2c: latency must be >= 0");
    if (c2c_buf < 1)
        throw std::runtime_error("die_ports.c2c: buffer_depth must be >= 1");

    // 展开 overrides 到 (side, idx) -> spec，同时查重
    std::map<std::pair<int, int>, D2DJson> ov;
    if (dp.contains("overrides")) {
        for (const auto &o : dp.at("overrides")) {
            Directions side = SideFromStr(o.at("side").get<std::string>());
            std::vector<int> idxs;
            const D2DJson &ij = o.at("idx");
            if (ij.is_array())
                for (auto &v : ij)
                    idxs.push_back(v.get<int>());
            else
                idxs.push_back(ij.get<int>());
            for (int idx : idxs) {
                if (idx < 0 || idx >= EdgeLen(side))
                    throw std::runtime_error(
                        "die_ports: override idx out of range on a side");
                auto key = std::make_pair((int)side, idx);
                if (ov.count(key))
                    throw std::runtime_error(
                        "die_ports: duplicate port override (side, idx)");
                ov[key] = o;
            }
        }
    }

    const D2DJson *edges = dp.contains("edges") ? &dp.at("edges") : nullptr;
    const char *side_names[4] = {"N", "S", "W", "E"};
    Directions side_dirs[4] = {NORTH, SOUTH, WEST, EAST};

    int next_id = 0;
    for (int s = 0; s < 4; s++) {
        Directions side = side_dirs[s];
        const D2DJson *edge_default =
            (edges && edges->contains(side_names[s])) ? &edges->at(side_names[s])
                                                      : nullptr;
        for (int idx = 0; idx < EdgeLen(side); idx++) {
            auto key = std::make_pair((int)side, idx);
            const D2DJson *spec = nullptr;
            if (ov.count(key))
                spec = &ov[key];
            else if (edge_default)
                spec = edge_default;
            if (!spec)
                continue; // 该周边位置无功能端口

            PortRole role;
            Directions dir;
            ParseRoleSpec(*spec, side, role, dir);

            D2DPort p;
            p.port_id = next_id++;
            p.tile = TileFor(side, idx);
            p.side = side;
            p.role = role;
            p.dir = dir;
            p.bw = c2c_bw;
            p.latency = c2c_lat;
            p.buf = c2c_buf;
            g_die_ports.ports.push_back(p);
        }
    }

    ValidateDiePorts();

    // port_for_host：每核就近选一个 HOST 端口 tile（Manhattan）
    g_die_ports.port_for_host.assign(CORES_PER_DIE, -1);
    for (int c = 0; c < CORES_PER_DIE; c++) {
        int cx = c % GRID_X, cy = c / GRID_X;
        int best = -1, best_d = 1 << 30;
        for (auto &p : g_die_ports.ports) {
            if (p.role != ROLE_HOST)
                continue;
            int px = p.tile % GRID_X, py = p.tile / GRID_X;
            int d = std::abs(cx - px) + std::abs(cy - py);
            if (d < best_d) {
                best_d = d;
                best = p.port_id;
            }
        }
        g_die_ports.port_for_host[c] = best;
    }

    // port_for[tile][dir]：每核就近选一个 dir 方向的 C2C 出口端口（V1 跨 die 路由用）。
    // V1 每方向至多一个 C2C 端口 → 就近即该端口；无该方向 C2C → -1。
    g_die_ports.port_for.assign(CORES_PER_DIE, std::vector<int>(DIRECTIONS, -1));
    const Directions kDirs[4] = {WEST, EAST, NORTH, SOUTH};
    for (int c = 0; c < CORES_PER_DIE; c++) {
        int cx = c % GRID_X, cy = c / GRID_X;
        for (Directions d : kDirs) {
            int best = -1, best_d = 1 << 30;
            for (auto &p : g_die_ports.ports) {
                if (p.role != ROLE_C2C || p.dir != d)
                    continue;
                int px = p.tile % GRID_X, py = p.tile / GRID_X;
                int dist = std::abs(cx - px) + std::abs(cy - py);
                if (dist < best_d) {
                    best_d = dist;
                    best = p.port_id;
                }
            }
            g_die_ports.port_for[c][d] = best;
        }
    }

    // V0b-4：构造并校验 D2D peer/link（结构级）
    BuildD2DLinks();
}

void ValidateAddressSpace() {
    if (GRID_X <= 0 || GRID_Y <= 0 || DIE_X <= 0 || DIE_Y <= 0)
        throw std::runtime_error(
            "hardware config: GRID_X/Y and DIE_X/Y must be positive");
    // 宽整数防溢出
    long long cores_per_die = (long long)GRID_X * GRID_Y;
    long long die_count = (long long)DIE_X * DIE_Y;
    long long total_cores = cores_per_die * die_count;
    // endpoint 空间 = core [0,total) + host(DIE_COUNT) + mem(DIE_COUNT)
    long long endpoint_space = total_cores + 2 * die_count;
    if (endpoint_space > 65536)
        throw std::runtime_error(
            "endpoint address space (" + std::to_string(endpoint_space) +
            ") exceeds 16-bit M_D_DES/M_D_SOURCE capacity (65536); reduce "
            "cores/dies or widen the fields");
}

void ValidateDiePorts() {
    if (!g_die_ports.active)
        return;

    // die-mesh 里存在的邻居方向必须有 C2C 端口（同构 die 模板需覆盖内部方向）
    if (DIE_X > 1) {
        if (g_die_ports.PortsForDir(EAST).empty() ||
            g_die_ports.PortsForDir(WEST).empty())
            throw std::runtime_error(
                "die_ports: DIE_X>1 requires >=1 C2C port on both E and W");
    }
    if (DIE_Y > 1) {
        if (g_die_ports.PortsForDir(NORTH).empty() ||
            g_die_ports.PortsForDir(SOUTH).empty())
            throw std::runtime_error(
                "die_ports: DIE_Y>1 requires >=1 C2C port on both N and S");
    }

    // HOST 可达性：至少 1 个 HOST 端口，否则跨 die 配置下 HOST 不可达
    if (g_die_ports.HostPortCount() < 1)
        throw std::runtime_error(
            "die_ports: no HOST port (all margin ports taken by C2C/MEM) -> "
            "HOST unreachable");
}

void ValidateV1MvpTopology() {
    // V1 runtime 前置：多 die 必须有 die_ports 提供 C2C（V0 允许「多 die 实例化但无 die_ports」，
    // 那种配置下没有任何跨 die 通路，V1 runtime 启用时必须明确拒绝，不能静默无链路）。
    if (DIE_COUNT > 1 && !g_die_ports.active)
        throw std::runtime_error(
            "V1 MVP: multi-die requires die_ports with C2C ports "
            "(no cross-die path otherwise)");
    if (!g_die_ports.active)
        return; // 单 die 且无 die_ports：无跨 die 流量，合法
    const Directions dirs[4] = {WEST, EAST, NORTH, SOUTH};
    for (Directions D : dirs) {
        int cnt = (int)g_die_ports.PortsForDir(D).size();
        bool neighbor = (D == EAST || D == WEST) ? (DIE_X > 1) : (DIE_Y > 1);
        if (cnt > 1)
            throw std::runtime_error(
                "V1 MVP: at most one C2C port per direction "
                "(multi-port per direction is a later version)");
        if (neighbor && cnt != 1)
            throw std::runtime_error(
                "V1 MVP: a die-neighbor direction must have exactly one C2C "
                "port");
    }
    // V1 单链路只支持 1 packet/cycle（单 256-bit 信号无法真实表达 >1 包/周期）——显式拒绝，
    // 不静默降级。>1 packet/cycle（多 lane / flit 聚合）留到 V3+。
    for (const auto &p : g_die_ports.ports)
        if (p.role == ROLE_C2C && p.bw != 1)
            throw std::runtime_error(
                "V1 MVP: C2C link_bw must be 1 packet/cycle (multi-packet/cycle "
                "is a later version)");
    // peer-connectedness 由 BuildD2DLinks 保证：某方向存在邻 die 时，该方向 C2C 端口若找不到
    // 对侧镜像端口即抛错；边界方向的模板端口无 peer（允许）。故此处只需 count 契约。
}

static Directions OppositeSide(Directions d) {
    switch (d) {
    case NORTH: return SOUTH;
    case SOUTH: return NORTH;
    case EAST:  return WEST;
    case WEST:  return EAST;
    default:    return CENTER;
    }
}

// die 级方向邻居（无环绕）；返回邻 die id 或 -1（边界）。
static int DieNeighbor(int die_id, Directions dir) {
    int dx = die_id % DIE_X, dy = die_id / DIE_X;
    switch (dir) {
    case EAST:  dx += 1; break;
    case WEST:  dx -= 1; break;
    case NORTH: dy += 1; break;
    case SOUTH: dy -= 1; break;
    default: return -1;
    }
    if (dx < 0 || dx >= DIE_X || dy < 0 || dy >= DIE_Y)
        return -1;
    return dy * DIE_X + dx;
}

// 端口在其“垂直轴”上的坐标（E/W 用 y，N/S 用 x），用于跨 die 镜像配对。
static int PerpCoord(const D2DPort &p) {
    return (p.side == EAST || p.side == WEST) ? (p.tile / GRID_X)
                                              : (p.tile % GRID_X);
}

void BuildD2DLinks() {
    g_d2d_links.clear();
    if (!g_die_ports.active)
        return;
    // 每个 C2C 端口：找方向邻 die 上、对侧同垂直坐标的镜像端口，构造 link。
    for (int d = 0; d < DIE_COUNT; d++) {
        for (const auto &p : g_die_ports.ports) {
            if (p.role != ROLE_C2C)
                continue;
            int nd = DieNeighbor(d, p.dir);
            if (nd < 0)
                continue; // 边界方向端口：无 peer（允许）
            // 镜像端口：对侧、同垂直坐标（同构模板，直接在 g_die_ports 里找）
            const D2DPort *mirror = nullptr;
            for (const auto &q : g_die_ports.ports) {
                if (q.role == ROLE_C2C && q.side == OppositeSide(p.side) &&
                    q.dir == OppositeSide(p.dir) && PerpCoord(q) == PerpCoord(p)) {
                    mirror = &q;
                    break;
                }
            }
            if (!mirror)
                throw std::runtime_error(
                    "d2d links: no reciprocal C2C port on neighbor die "
                    "(direction/coord mismatch)");
            // 带宽/延迟兼容。注：V0 所有 C2C 端口共享全局 die_ports.c2c 参数（同构模板），
            // 故该分支恒不触发——是「同构保证」而非已测的不兼容拒绝路径。保留此 check 以便
            // 将来支持 per-port/per-die override 时即时生效。
            if (p.bw != mirror->bw || p.latency != mirror->latency)
                throw std::runtime_error(
                    "d2d links: bw/latency mismatch between paired C2C ports");
            D2DLink l;
            l.local_die = d;
            l.local_port = p.port_id;
            l.remote_die = nd;
            l.remote_port = mirror->port_id;
            l.tx_bw = p.bw;
            l.rx_bw = mirror->bw;
            l.latency = p.latency;
            g_d2d_links.push_back(l);
        }
    }
    // 端口唯一性：每个 (die, local_port) 至多作为一条 link 的本地端点
    std::set<std::pair<int, int>> seen;
    for (const auto &l : g_d2d_links) {
        auto k = std::make_pair(l.local_die, l.local_port);
        if (seen.count(k))
            throw std::runtime_error(
                "d2d links: a (die,port) endpoint is used by >1 link");
        seen.insert(k);
    }
    // 方向互反（精确四元组）：link (a,b)->(c,d) 必存在反向 link (c,d)->(a,b)。
    // 弱判断（仅查 remote 端点是否出现在某 link 本地端点集）会漏掉「远端存在但指向第三端点」。
    std::set<std::tuple<int, int, int, int>> link_set;
    for (const auto &l : g_d2d_links)
        link_set.insert(std::make_tuple(l.local_die, l.local_port, l.remote_die,
                                        l.remote_port));
    for (const auto &l : g_d2d_links) {
        if (!link_set.count(std::make_tuple(l.remote_die, l.remote_port,
                                            l.local_die, l.local_port)))
            throw std::runtime_error(
                "d2d links: non-reciprocal link (exact reverse tuple "
                "(remote->local) absent)");
    }
}

int PortForHost(int local_core) {
    if (local_core < 0 || local_core >= (int)g_die_ports.port_for_host.size())
        return -1;
    return g_die_ports.port_for_host[local_core];
}

bool IsC2CEgressEdge(int global_tile, Directions dir) {
    if (!g_die_ports.active)
        return false;
    int die = global_tile / CORES_PER_DIE, local = global_tile % CORES_PER_DIE;
    if (DieNeighbor(die, dir) < 0)
        return false; // 该方向无邻 die（边界）→ 无跨 die 连接
    for (const auto &p : g_die_ports.ports)
        if (p.role == ROLE_C2C && p.side == dir && p.tile == local)
            return true;
    return false;
}

int PortForDir(int local_core, Directions dir) {
    if (local_core < 0 || local_core >= (int)g_die_ports.port_for.size())
        return -1;
    if (dir < 0 || dir >= (int)g_die_ports.port_for[local_core].size())
        return -1;
    return g_die_ports.port_for[local_core][dir];
}

void BuildHostAttach() {
    g_host_attach = HostAttachTable{};

    // config 驱动：die_ports 提供 role=HOST 端口时，从这些端口建挂载表（同构 die 模板，
    // 每 die 复制一份）。每个 HOST 端口 = 一条 lane；核就近选一个 HOST 端口（port_for_host）。
    // 对 W 全边 host 的配置，产出的表与 legacy 西边缘完全相同（sim-time 不变）。
    // 注：只有「无 die_ports」才走 legacy——有 die_ports 但无 HOST 端口会先被
    // ValidateDiePorts 以「HOST 不可达」拒绝，不会到达此处（故 HostPortCount>=1 恒成立）。
    if (g_die_ports.active && g_die_ports.HostPortCount() >= 1) {
        g_host_attach.legacy = false;
        // 模板内 HOST 端口（按 port_id 顺序）→ ordinal（lane 在 die 内的序号）
        std::vector<int> host_tiles;                        // ordinal -> 局部 tile
        std::vector<int> portid_ord(g_die_ports.ports.size(), -1); // port_id -> ordinal
        for (const auto &p : g_die_ports.ports) {
            if (p.role != ROLE_HOST)
                continue;
            portid_ord[p.port_id] = (int)host_tiles.size();
            host_tiles.push_back(p.tile);
        }
        int hp = (int)host_tiles.size();
        g_host_attach.n_lanes = hp * DIE_COUNT;
        g_host_attach.lane_tile.assign(g_host_attach.n_lanes, -1);
        for (int d = 0; d < DIE_COUNT; d++)
            for (int k = 0; k < hp; k++)
                g_host_attach.lane_tile[d * hp + k] =
                    d * CORES_PER_DIE + host_tiles[k];
        g_host_attach.core_lane.assign(TOTAL_CORES, -1);
        for (int c = 0; c < TOTAL_CORES; c++) {
            int d = c / CORES_PER_DIE, local = c % CORES_PER_DIE;
            int pid = PortForHost(local); // 就近 HOST 端口 port_id
            int ord = (pid >= 0 && pid < (int)portid_ord.size())
                          ? portid_ord[pid]
                          : -1;
            g_host_attach.core_lane[c] = (ord >= 0) ? d * hp + ord : -1;
        }
    } else {
        // Legacy：西边缘、每全局行一条 lane。lane i ↔ 全局行 i ↔ 西边缘 tile i*GRID_X。
        // core↔lane 完整入表（core_lane[c]=c/GRID_X），供 routing/enqueue 统一查表。
        g_host_attach.legacy = true;
        int rows = GRID_Y * DIE_COUNT; // legacy lane 数
        g_host_attach.n_lanes = rows;
        g_host_attach.lane_tile.assign(rows, -1);
        for (int i = 0; i < rows; i++)
            g_host_attach.lane_tile[i] = i * GRID_X; // 全局行 i 的西边缘 core id
        g_host_attach.core_lane.assign(TOTAL_CORES, -1);
        for (int c = 0; c < TOTAL_CORES; c++)
            g_host_attach.core_lane[c] = c / GRID_X; // 全局行
    }

    // 反向表 tile -> lane（O(1) 查 IsHostAttachTile / HostLaneOfTile）。若一个 tile 承载
    // 多条 lane（如 S+W 角点），此处 last-wins；ValidateHostAttach 会在任何运行前拒绝该配置。
    g_host_attach.tile_lane.assign(TOTAL_CORES, -1);
    for (int l = 0; l < g_host_attach.n_lanes; l++) {
        int tile = g_host_attach.lane_tile[l];
        if (tile >= 0 && tile < TOTAL_CORES)
            g_host_attach.tile_lane[tile] = l;
    }
}

void ValidateHostAttach() {
    const auto &t = g_host_attach;
    // 尺寸一致
    if ((int)t.lane_tile.size() != t.n_lanes)
        throw std::runtime_error("host_attach: lane_tile.size() != n_lanes");
    if ((int)t.core_lane.size() != TOTAL_CORES)
        throw std::runtime_error(
            "host_attach: core_lane.size() != TOTAL_CORES");
    // 生产路径 HOST_LANES 与表一致（Monitor/MemInterface 以 HOST_LANES 为界）
    if (HOST_LANES != t.n_lanes)
        throw std::runtime_error(
            "host_attach: HOST_LANES != n_lanes (production inconsistency)");
    if ((int)t.tile_lane.size() != TOTAL_CORES)
        throw std::runtime_error(
            "host_attach: tile_lane.size() != TOTAL_CORES");
    // 每个 lane 的 tile 合法；且每个 tile 至多一条 HOST lane——RouterUnit 每 tile 仅一套
    // HOST 接口，重复挂载（如同时配 S-host 与 W-host 的角点）会重复绑定同组 SystemC port。
    std::set<int> seen_tiles;
    for (int l = 0; l < t.n_lanes; l++) {
        int tile = t.lane_tile[l];
        if (tile < 0 || tile >= TOTAL_CORES)
            throw std::runtime_error("host_attach: lane -> tile illegal");
        if (!seen_tiles.insert(tile).second)
            throw std::runtime_error(
                "host_attach: duplicate lane_tile (a tile carries >1 HOST lane, "
                "e.g. S+W corner); per-tile multi-HOST unsupported");
    }
    // 每个 core 映射合法且其 HOST tile 与该核同 die（避免 Monitor 越界 / routers[-1]）。
    // 注：这是 core→lane→tile 的**结构层**不跨 die 保证；路由层「他 die 的 HOST endpoint
    // 不被当作本 die 本地出口」需 GetNextHop 配合，留给 2b。
    for (int c = 0; c < TOTAL_CORES; c++) {
        int lane = t.core_lane[c];
        if (lane < 0 || lane >= t.n_lanes)
            throw std::runtime_error("host_attach: core -> lane out of range");
        if (t.lane_tile[lane] / CORES_PER_DIE != c / CORES_PER_DIE)
            throw std::runtime_error(
                "host_attach: HOST tile not in the core's own die");
    }
}

int HostLaneOfCore(int global_core) {
    // 查表（legacy: == global_core/GRID_X）。非法核返回 -1（不产生 routers[-1]）。
    if (global_core < 0 || global_core >= (int)g_host_attach.core_lane.size())
        return -1;
    return g_host_attach.core_lane[global_core];
}

int HostTileOfLane(int lane) {
    if (lane < 0 || lane >= (int)g_host_attach.lane_tile.size())
        return -1;
    return g_host_attach.lane_tile[lane];
}

int HostLaneOfTile(int global_tile) {
    if (global_tile < 0 || global_tile >= (int)g_host_attach.tile_lane.size())
        return -1;
    return g_host_attach.tile_lane[global_tile];
}

bool IsHostAttachTile(int global_tile) {
    return HostLaneOfTile(global_tile) >= 0;
}
