#pragma once
// D2D 端口 / 链路数据结构（V0：结构与配置校验，尚不承载跨 die 流量）
#include "defs/enums.h"
#include "nlohmann/json.hpp"
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using D2DJson = nlohmann::json;

// 端口逻辑功能
enum PortRole { ROLE_HOST = 0, ROLE_MEM, ROLE_C2C };

// flow/link 统计基础接口（V0 建立接口；跨 die 流量在 V1+ 才真正累加）
struct D2DPortStats {
    long in_pkts = 0;   // 进入该端口的包数
    long out_pkts = 0;  // 经该端口发出的包数
    long busy_cycles = 0;
    long stall_cycles = 0;
};

// 一个周边端口 = die 内某边缘 tile 在某朝外方向上的出链
struct D2DPort {
    int port_id;    // die 内线性编号
    int tile;       // 局部核 id
    Directions side; // 物理边 N/S/E/W
    PortRole role;
    Directions dir;  // role==C2C 时通往的邻 die 方向（MVP 约束 dir==side）
    int bw;          // link_bw（包/cycle），role==C2C 有效
    int latency;     // link latency（cycle）
    int buf;         // buffer_depth
    D2DPortStats stats;
};

// 一条点对点物理 D2D 链路（V0 仅记录，phase-2 用）
struct D2DLink {
    int local_die, local_port;
    int remote_die, remote_port;
    int tx_bw, rx_bw, latency;
};

// 每个 die（同构模板）的端口表
struct D2DPortTable {
    bool active = false;         // 配置里是否出现 die_ports
    std::vector<D2DPort> ports;  // 已指派的端口
    std::vector<int> port_for_host; // 每个局部核 → 去 HOST 走的 port_id（-1=无）

    // 按方向取 C2C 端口集合
    std::vector<int> PortsForDir(Directions d) const {
        std::vector<int> r;
        for (auto &p : ports)
            if (p.role == ROLE_C2C && p.dir == d)
                r.push_back(p.port_id);
        return r;
    }
    int HostPortCount() const {
        int c = 0;
        for (auto &p : ports)
            if (p.role == ROLE_HOST)
                c++;
        return c;
    }
    // D2D 端口/链路总活动量（V0 恒为 0：尚无跨 die 流量）。用于
    // 「单 die 下 D2D 活动计数为 0」不变量检查。
    long TotalActivity() const {
        long a = 0;
        for (auto &p : ports)
            a += p.stats.in_pkts + p.stats.out_pkts;
        return a;
    }
};

extern D2DPortTable g_die_ports;
// 全局 D2D 链路表（V0b-4：从同构端口模板 + die-mesh 构造的点对点 peer 连接，
// 结构级；尚不承载流量，流量属 V1）。
extern std::vector<D2DLink> g_d2d_links;

// 从 hardware config 的顶层 json 解析 "die_ports"（不含则 active=false，单 die 兼容）。
// 解析中即做启动期校验；非法配置抛 std::runtime_error（作为启动失败）。
void ParseDiePorts(const D2DJson &hw_json);

// 供测试/外部单独触发的校验（ParseDiePorts 内部也会调用）。
void ValidateDiePorts();

// 维度与 endpoint 地址空间容量校验：GRID_X/Y、DIE_X/Y 为正；
// 且 core+host+mem 端点总数 <= 65536（des_/source_ 为 16-bit）。用宽整数防溢出。
// 非法即抛 std::runtime_error（启动失败）。
void ValidateAddressSpace();

// 便捷查询
int PortForHost(int local_core);

// ---- V1-pre：HOST 物理挂载表（routing / enqueue / binding 的统一真源）----
// lane = 一条 HOST 通道；每 lane 绑定一个全局 tile(router)。每个核经 HostLaneOfCore
// 映射到一条 lane（去 HOST 与 收 HOST 均走该 lane 对应的 tile）。
// Legacy（无 role=HOST 端口）：西边缘、每全局行一条 lane，lane==global row==tile/GRID_X，
// 与 2B1 硬编码 i*GRID_X 逐位等价；config 模式（role=HOST 端口）在后续增量接入。
struct HostAttachTable {
    int n_lanes = 0;
    std::vector<int> lane_tile; // lane -> 全局 tile(router id)，size==n_lanes
    std::vector<int> core_lane; // 全局核 -> lane，size==TOTAL_CORES（完整 core↔lane 映射）
    std::vector<int> tile_lane; // 全局 tile -> lane（-1=非挂载 tile），size==TOTAL_CORES
    bool legacy = true;         // true=西边缘每全局行一条 lane（当前唯一实现）
};
extern HostAttachTable g_host_attach;

// 依据维度常量（+ 后续 die_ports）构造挂载表；须在地址常量 + ParseDiePorts 之后调用。
void BuildHostAttach();
// 独立结构校验：lane_tile/core_lane 尺寸、每 lane tile 合法、每 core 映射合法且同 die、
// 生产路径 HOST_LANES==n_lanes。非法即抛 std::runtime_error（启动失败）。
// 调用前需已设 HOST_LANES = g_host_attach.n_lanes。
void ValidateHostAttach();
// 核 -> 其 HOST lane（legacy: global row = gid/GRID_X）。非法核返回 -1。
// 注意（V1 范围边界）：仅 dataflow 的 HOST 入队经 host_envelope→HostLaneOfCore；
// PD/GPU/PDS 的 fill_queue_* 仍直接用 id/GRID_X。二者在 legacy 下相等，故当前一致；
// config 驱动（非 legacy）HOST 仅在 dataflow 生效，非 dataflow 多 die HOST 属后续版本。
int HostLaneOfCore(int global_core);
// lane -> 绑定的全局 router id。
int HostTileOfLane(int lane);
// 全局 tile 是否是 HOST 挂载 tile（RouterUnit 据此决定是否创建 HOST 接口）。
// legacy 下等价于西边缘（IsMarginCore），故回归行为不变。
bool IsHostAttachTile(int global_tile);
// 全局 tile -> 其 HOST lane（非挂载 tile 返回 -1）。
int HostLaneOfTile(int global_tile);

// V0b-4：从 g_die_ports（同构模板）+ die-mesh 构造 g_d2d_links 并校验：
// 方向互反、端口唯一（每个 (die,port) 至多一条 link）、两端带宽/延迟兼容。
// 边界方向（无邻 die）的 C2C 端口无 peer（允许，不成 link）。非法即抛 std::runtime_error。
void BuildD2DLinks();

// V0 L0 纯函数自测（编址/端点/矩形拓扑/端口配置校验）。返回失败数（0=全过）。
int RunD2DV0SelfTest();

// ---- V1-pre 3b-2b：HOST lane 接收统计（运行时可观测，供 e2e 断言 DONE/ACK 到达正确 lane）----
// 每 lane 的 DONE/ACK 计数；mismatch = 消息到达的 lane != HostLaneOfCore(source_) 的次数。
extern std::vector<long> g_host_lane_done;
extern std::vector<long> g_host_lane_ack;
extern long g_host_lane_mismatch;
// 接收消息签名（多重集）：DONE 按 source；ACK 按 (source, tag_id)——比总数更强，
// 能发现「同 source 同数量但丢/重不同 tag」的情形。仍不含 seq_id/事件轨迹，故只证明
// 「每 (source[,tag]) 接收计数与基线一致」，不等于逐消息全等。
extern std::map<int, int> g_host_done_src;              // source -> count
extern std::map<std::pair<int, int>, int> g_host_ack_sig; // (source,tag) -> count
void ResetHostLaneStats(int n_lanes);
