#pragma once
// D2D 端口 / 链路数据结构（V0：结构与配置校验，尚不承载跨 die 流量）
#include "common/flow.h"
#include "defs/enums.h"
#include "nlohmann/json.hpp"
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
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
    int link_group = -1; // V5：>=0 表示多个端口共享同一物理 link 容量；-1=独立
    D2DPortStats stats;
};

// 一条点对点物理 D2D 链路（V0 仅记录，phase-2 用）
struct D2DLink {
    int local_die, local_port;
    int remote_die, remote_port;
    int tx_bw, rx_bw, latency;
    int link_group = -1;
};

// 每个 die（同构模板）的端口表
struct D2DPortTable {
    bool active = false;         // 配置里是否出现 die_ports
    std::vector<D2DPort> ports;  // 已指派的端口
    std::vector<int> port_for_host; // 每个局部核 → 去 HOST 走的 port_id（-1=无）
    // 每个局部核 → 各 die 级方向(N/S/E/W) 的 C2C 出口 port_id（就近；-1=该方向无 C2C）。
    // key 是 (tile, dir)（非 source_core）——多跳接力时中间 die 用入口 tile 查同一张表（V1 单跳即够）。
    std::vector<std::vector<int>> port_for; // [CORES_PER_DIE][DIRECTIONS]

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

// ---- V3-a/V4-a：D2D 链路建模模式、backend 与容量契约 ----
//
// **backend（正交于 cycle 内部 mode）**：
//   cycle（默认）——实例化 V1～V3 的逐拍 D2DLinkUnit；旧配置不写 backend 时精确保持。
//   behavioral——V4 聚合单条 flow、按解析式补 D2D 固定延迟/序列化；不建模跨 flow
//     端口/链路争用、有限 FIFO、credit 或网络死锁。V4-a 先建立并严格校验配置接缝，
//     V4-c 才接入生产运行时。
//
// **cycle mode**：
//   functional_v2（默认，旧配置不写 mode 时即此）——V1/V2 语义：功能性无限 FIFO、
//     每方向单端口、1 packet/cycle、固定 latency。**不**建模有限缓冲/背压。
//   bounded_saf——V3：有限缓冲 + 有理数速率 + 背压，死锁安全策略必须显式选定。
//
// **死锁安全**：MVP 采用 **whole-flow store-and-forward**（建模计划已定，不摇摆到 escape VC）：
//   每个 D2D 边界必须**完整收下整条 flow（F 个包）**后才向下一段发送，用边界 buffer 把源 die
//   与远端 die 的锁依赖剪成两段，切断跨 die 的 hold-and-wait。
//
// **速率用整数有理数** `num/den`，避免浮点非确定性。物理约束：单个 `sc_bv<256>` 信道每拍最多
// 载 1 个包，故 **0 < rate <= 1**；`rate > 1` 必须在启动期**明确拒绝**，不得静默按 1 建模
// （真正的 >1 packet/cycle 需要多 lane，属后续版本）。
//
// **四类容量必须分开配置**，不可用一个 buffer_depth 混代：
//   saf_buffer_depth      >= F        —— whole-flow SAF 的**正确性**要求（装得下整条 flow）
//   link_inflight_depth   >= BDP      —— 在途 FIFO，维持流水利用率（带宽时延积）
//   rx_buffer_depth                   —— 远端接收侧容量与背压
//   ctrl_buffer_depth                 —— 独立控制子通道容量
// 「buffer >= BDP」**不能**代替「buffer >= flow」，二者约束不同、不可互相顶替。
enum D2DBackend { BACKEND_CYCLE = 0, BACKEND_BEHAVIORAL };
enum D2DMode { MODE_FUNCTIONAL_V2 = 0, MODE_BOUNDED_SAF };
enum D2DSafety { SAFETY_NONE = 0, SAFETY_WHOLE_FLOW_SAF };

struct D2DRate {
    int num = 1, den = 1;
    bool Valid() const { return num > 0 && den > 0 && num <= den; } // 0 < r <= 1
};

// V3-b2 standalone credit 模型的接口流水寄存器拍数（上游 tx 边界 1 拍 + 下游
// credit-rx 边界 1 拍）。clocked D2DLinkUnit 即使 link_latency=0，正向交付和反向信用也
// 各至少占一个服务拍；保守传输窗口 = ceil(rate * (2*max(latency,1)+pipe))。
// 生产 whole-flow SAF 数据路径用全路径 reservation + ready 背压而非逐包 credit，但仍执行这个
// 已验证的保守窗口下界，避免 link_inflight_depth 在未来切回逐包 credit 时成为隐含吞吐瓶颈。
static const int D2D_CREDIT_PIPE = 2;
inline long long D2DCreditRttCycles(int link_latency) {
    const long long service_latency = link_latency > 0 ? link_latency : 1;
    return 2LL * service_latency + D2D_CREDIT_PIPE;
}

inline long long D2DBdpPackets(int link_latency, const D2DRate &rate) {
    const long long n = D2DCreditRttCycles(link_latency) * rate.num;
    return (n + rate.den - 1) / rate.den;
}

enum D2DSelectPolicy {
    SELECT_NEAREST = 0,
    SELECT_BANDED_NEAREST,
    SELECT_TAG_HASH,
    SELECT_HYBRID,
    SELECT_DYNAMIC
};

struct D2DLinkConfig {
    D2DBackend backend = BACKEND_CYCLE;
    D2DMode mode = MODE_FUNCTIONAL_V2;
    D2DSafety safety = SAFETY_NONE;
    D2DRate port_rate; // 片上→端口注入速率
    D2DRate link_rate; // 跨 die 链路速率（常为瓶颈）
    int link_latency = 0;
    int saf_buffer_depth = 0;
    int link_inflight_depth = 0;
    int rx_buffer_depth = 0;
    int ctrl_buffer_depth = 0;
    bool v5_multiport = false;
    bool select_policy_explicit = false;
    D2DSelectPolicy select_policy = SELECT_NEAREST;
    unsigned long long select_seed = 0;
};
extern D2DLinkConfig g_d2d_cfg;

// V3-d：whole-flow SAF 的生产路径全路径原子预留。REQUEST 离开源核前，按确定性的
// die-level XY 路径一次性为每条有向 C2C 边预留 F 个包；任一边容量不足则回滚全部已做预留，
// 因而 DATA 不会在只拿到部分路径容量时开始注入。各边的 SAF stage 排空该 flow 后分别释放。
void ResetWholeFlowSafRuntime();
void ReserveWholeFlowSafPath(int source_global, int dest_global, int tag,
                             int subflow, int flow_packets);
// V5-d：一次逻辑 flow 的所有 subflow 做**同一次**预检/提交。counts[s] 是 subflow s
// 的整流包数；任一物理 link 或共享 link_group 容量不足时，所有账本保持不变。
void ReserveStripedWholeFlowSafPaths(int source_global, int dest_global,
                                     int tag,
                                     const std::vector<int> &counts);
void ReleaseWholeFlowSafLink(int link_idx, const FlowKey &key,
                             int flow_packets);
long WholeFlowSafReservedPackets();
long WholeFlowSafGroupReservedPackets();
// V5-d：同一有向 die pair、同 link_group 的 DATA 共享 1 packet/cycle cut。
// 所有成员上一拍登记 request，本拍按 link index round-robin 选一个 grant，结果与进程调用顺序无关。
bool V5LinkGroupGrant(int link_idx, long long cycle, bool request);
void ResetV5LinkGroupRuntime();

// 解析 die_ports.c2c 的 V3/V4 字段并做**启动期**校验（非法组合抛 std::runtime_error）。
// 由 ParseDiePorts 调用；旧配置（无 backend/mode）恒定解析为 cycle+functional_v2，
// 行为与 V2 逐位一致。
void ParseD2DLinkConfig(const D2DJson &c2c);

// 版本感知的 D2D 拓扑/容量契约校验：functional_v2 沿用 V1/V2 MVP 契约（每方向 <=1 个
// peer-connected C2C 端口、link_bw==1）；bounded_saf 额外要求四类容量与速率契约成立。
void ValidateD2DTopology();

// V1 MVP 拓扑契约校验（**V1 runtime 启用时**调用，V0/V1-a 结构阶段不强制）：
//   - 任一 die 级方向的 C2C 端口数 <= 1（多端口留 V5）；
//   - 存在邻 die 的方向（DIE_X>1→E/W，DIE_Y>1→N/S）必须**恰好一个** peer-connected C2C 端口。
// 违反即抛 std::runtime_error（明确拒绝，不静默降级）。
void ValidateV1MvpTopology();

// 维度与 endpoint 地址空间容量校验：GRID_X/Y、DIE_X/Y 为正；
// 且 core+host+mem 端点总数 <= 65536（des_/source_ 为 16-bit）。用宽整数防溢出。
// 非法即抛 std::runtime_error（启动失败）。
void ValidateAddressSpace();

// 便捷查询
int PortForHost(int local_core);
// 局部核在 die 级方向 dir(N/S/E/W) 的 C2C 出口 port_id（-1=该方向无 C2C 端口）。
int PortForDir(int local_core, Directions dir);
// V5：按一次 flow/subflow 选择并固定一个出口。静态策略由
// (source,tag,subflow,seed) 决定；dynamic 只在此 flow 选择点更新负载，绝不逐包重选。
int SelectPortForFlow(int local_core, Directions dir, int source, int tag,
                      int subflow);
void ResetV5PortSelectionStats();

// 全局 tile 在方向 dir 上是否是「peer-connected C2C 出口边」——即该 tile 的局部位置有一个
// side==dir 的 C2C 端口，且该 die 在 dir 方向存在邻 die。V1-b 拓扑接线据此把该边接到 D2D Link
// （取代开边终结）；单 die / 无 C2C 时恒 false，拓扑不变。
bool IsC2CEgressEdge(int global_tile, Directions dir);

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

// 是否存在从 local_die 到 remote_die 的实际有向 D2D link。
// 与 PortsForDir（同构端口模板是否声明某方向端口）不同，本函数查询 BuildD2DLinks
// 已构造的具体 die-peer 元组，供 workload preflight 精确验证运行时通路。
bool HasD2DLink(int local_die, int remote_die);

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

// ---- V1-b：D2D Link 运行时统计（所有 link 单元累加；供 [D2D] dump 的 in/out）----
extern long g_d2d_link_in_pkts;
extern long g_d2d_link_out_pkts;
// V1-c3：按 wire 消息类型分别统计 capture/delivery，端到端测试据此验证
// REQUEST/ACK/DATA 都真实穿链且各自守恒；非法/未知类型不计入此数组。
extern long g_d2d_link_in_by_type[MSG_TYPE_NUM];
extern long g_d2d_link_out_by_type[MSG_TYPE_NUM];

// V4-c Behavioral 聚合传输统计。wire 上每条 DATA flow 只有一个代表包；logical_data_packets
// 保留原始 F。service_cycles 只在 flow 的第一条 D2D link 计一次；fixed_cycles 对
// REQUEST/ACK/DATA 每次跨 link 各加 L，据此可直接和 oracle 的 3*H*L+S(F) 对照。
struct D2DBehavioralStats {
    long data_flows = 0;
    long long logical_data_packets = 0;
    long long service_cycles = 0;
    long long fixed_cycles = 0;
};
extern D2DBehavioralStats g_d2d_behavioral_stats;
void ResetD2DLinkStats();

// ---- V1-d2：DATA 逐包完整性探针 ----
// 仅对 DATA 型包累加，capture(in)/delivery(out) 两侧各一份。比对两侧
// pkts/seqhash/csum 相等，为链路无丢/重/乱序/损坏提供比「类型总数守恒」更强的证据。
// 单 DATA 消息的序号契约是 base-agnostic 的连续区间 [minseq,maxseq]（当前生产发送从 1
// 开始，不能硬编码 0..N-1）；必须满足区间长度==pkts、恰好一个尾包且 endseq==maxseq。
// first/last_cycle 用于证明增加 link latency 只平移固定延迟、不改变 DATA 包间跨度。
struct D2DDataProbe {
    long pkts = 0;
    unsigned long long seqhash = 1469598103934665603ULL; // FNV-1a offset
    unsigned long long csum = 1469598103934665603ULL;
    int minseq = -1; // 最小 seq（首包设基，base-agnostic）
    int maxseq = -1;
    int endseq = -1; // is_end 包的 seq_id（-1=未见）
    int end_count = 0;
    int end_length = -1;
    int expect = 0;  // 期望下一个 seq（prev+1）
    bool inorder = true; // 交付序严格 +1 连续（与 base 无关）
    long long first_cycle = -1;
    long long last_cycle = -1;
};
extern D2DDataProbe g_d2d_data_in, g_d2d_data_out;

// ---- V5-b/c：逐 link、逐 subflow 的 DATA 完整性探针 ----
// 旧 D2DDataProbe 只覆盖单 DATA 序列；striping 后各 subflow 的 seq 都从 1
// 开始，必须按 (link,source,tag,subflow) 分桶，否则合法交织会被误报为乱序。
struct V5SubflowStat {
    long in_pkts = 0, out_pkts = 0;
    unsigned long long in_seqhash = 1469598103934665603ULL;
    unsigned long long out_seqhash = 1469598103934665603ULL;
    unsigned long long in_csum = 1469598103934665603ULL;
    unsigned long long out_csum = 1469598103934665603ULL;
    int out_minseq = -1, out_maxseq = -1, out_endseq = -1;
    int out_end_count = 0, out_end_length = -1, out_expect = 0;
    bool out_inorder = true;
};
using V5SubflowProbeKey = std::tuple<int, int, int, int>;
// key = (directed link index, source global id, tag, subflow)
extern std::map<V5SubflowProbeKey, V5SubflowStat> g_v5_subflow_stats;


// ---- V2-b：C2C 入口 re-pin（每进入一个 die 重新选出口）统计 ----
// 包跨 link 进入本 die 时，router 在入口清除上一跳的 pinned exit 并按本 die 重新 pin：
// 目的在本 die → -1（转片内 XY）；否则 → CrossDieSelectExit(入口 tile, des)。
//   total   = 入口重写次数（每包每跨一次 link 记一次）；
//   changed = 新值 != 携带值（如 2×2 对角 E→N，或到达目的 die 清为 -1）；
//   same    = 新值 == 携带值（如 3×1 直线 E→E：相邻 die 用同一模板 port id）。
// **same 是本计数器存在的理由**：该情形下「已重新 pin」与「沿用旧值」路由结果完全相同，
// 只能靠计数证明入口重写确实执行了，不能靠端到端是否送达来推断。
extern long g_d2d_repin_total, g_d2d_repin_changed, g_d2d_repin_same;

// ---- V2-c：逐条有向 link 归因 + 每 die NoC 活动 ----
// 全局 [D2D_TYPE] 只有总数，无法回答「究竟经过了哪几条 link、方向序列是什么、每个包跳了几跳」。
// 这里按 g_d2d_links 的下标为每条**有向** link 单独计数（D2DLinkUnit 一一对应），于是可以精确
// 断言：DATA 只经过路径上应经过的那些 link、每条计数相等（⇒ 每个包都走满全程、无跳过/重复）、
// 方向序列（如 3×1 的 E,E 与反向 W,W；2×2 对角的 E,N 与反向 W,S）。
struct D2DLinkStat {
    int local_die = -1, remote_die = -1;
    Directions dir = CENTER; // 该有向 link 的 die 级方向
    long in_by_type[MSG_TYPE_NUM] = {};
    long out_by_type[MSG_TYPE_NUM] = {};
};
extern std::vector<D2DLinkStat> g_d2d_link_stats;

// 每个 die 的 router 入口包数（所有方向：C2C link 入口 + 片内邻 router + 本核注入）。
extern std::vector<long> g_die_router_pkts;
// 每个 die **仅片内 router→router** 的输入包数：排除 C2C link 入口（跨 die 到达那一拍）
// 与 CENTER（本核注入）。这才是「包确实在该 die 的 mesh 里走了片内 hop」的直接证据——
// g_die_router_pkts>0 只能说明包进入过该 die 的入口 router，不能排除「入口 tile 恰好就是
// 下一条 link 的出口 tile、零片内 hop」。中间 die 的本计数 >0 才真正证明穿越了 NoC。
extern std::vector<long> g_die_mesh_pkts;
// V3-e：片内 NoC 可归因计数。send=成功通过同 die router→router 输出；stall=该输出有包但
// 下游 input buffer 满。C2C 边缘输出另计入 d2d_source_stall，避免把端口背压冒充片内争用。
extern std::vector<long> g_die_noc_sends, g_die_noc_stalls;
extern long g_d2d_source_stalls;
// DATA 尾包到达接收核的 cycle，按 (source,tag,dest) 记录，供 mixed/shared fairness 对照。
extern std::map<std::tuple<int, int, int>, long long> g_flow_done_cycle;
extern long g_saf_admission_successes, g_saf_admission_rejects;
void ResetDieActivityStats();
