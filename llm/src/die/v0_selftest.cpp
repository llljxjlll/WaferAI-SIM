// D2D V0 L0 自测：编址、端点解码、矩形拓扑、端口配置校验。
// 不依赖 SystemC 运行；直接设置全局维度并调用纯函数。
#include "die/port.h"
#include "defs/spec.h"
#include "common/msg.h"
#include "monitor/host_envelope.h"
#include "monitor/workload_normalize.h"
#include "utils/msg_utils.h"
#include "utils/router_utils.h"
#include <iostream>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <utility>

using D2DJson = nlohmann::json;

namespace {
int g_fail = 0;
int g_total = 0;

void check(bool cond, const std::string &name) {
    g_total++;
    if (!cond) {
        g_fail++;
        std::cout << "  [FAIL] " << name << std::endl;
    } else {
        std::cout << "  [ ok ] " << name << std::endl;
    }
}

// 设置一块 die 内 mesh 与 die-mesh 维度
void setTopo(int gx, int gy, int dx, int dy) {
    GRID_X = gx;
    GRID_Y = gy;
    GRID_SIZE = gx * gy;
    DIE_X = dx;
    DIE_Y = dy;
    DIE_COUNT = dx * dy;
    CORES_PER_DIE = GRID_SIZE;
    TOTAL_CORES = CORES_PER_DIE * DIE_COUNT;
    HOST_ENDPOINT_ID = TOTAL_CORES;
    // 清空端口状态：确保随后 BuildHostAttach 默认走 legacy（不受上一段 ParseDiePorts 残留
    // 的 role=HOST 端口影响）；需要 config 模式的段会在 setTopo 后再 ParseDiePorts。
    g_die_ports = D2DPortTable{};
}

bool throwsRuntime(const D2DJson &hw) {
    try {
        ParseDiePorts(hw);
    } catch (const std::runtime_error &) {
        return true;
    } catch (...) {
        return true;
    }
    return false;
}
} // namespace

int RunD2DV0SelfTest() {
    g_fail = 0;
    g_total = 0;
    std::cout << "==== D2D V0 self-test ====" << std::endl;

    // ---- 1. 编址 round-trip（含边界值）----
    setTopo(4, 4, 2, 2); // CORES_PER_DIE=16, DIE_COUNT=4, TOTAL_CORES=64
    {
        bool ok = true;
        for (int gid = 0; gid < TOTAL_CORES; gid++) {
            int die = DieOfGlobal(gid), loc = LocalOfGlobal(gid);
            if (GlobalId(die, loc) != gid)
                ok = false;
            if (die < 0 || die >= DIE_COUNT || loc < 0 || loc >= CORES_PER_DIE)
                ok = false;
        }
        check(ok, "global_id <-> (die,local) round-trip over all endpoints");
        check(GlobalId(0, 0) == 0, "GlobalId(0,0)=0 boundary");
        check(GlobalId(DIE_COUNT - 1, CORES_PER_DIE - 1) == TOTAL_CORES - 1,
              "GlobalId(max,max)=TOTAL_CORES-1 boundary");
        check(DieXOf(3) == 1 && DieYOf(3) == 1, "die 3 -> (dx=1,dy=1) in 2x2");
    }

    // ---- 2. 端点类型解码 ----
    check(DecodeEndpointType(0) == EP_CORE, "endpoint 0 -> CORE");
    check(DecodeEndpointType(TOTAL_CORES - 1) == EP_CORE,
          "endpoint TOTAL_CORES-1 -> CORE");
    check(DecodeEndpointType(HOST_ENDPOINT_ID) == EP_HOST,
          "endpoint HOST_ENDPOINT_ID -> HOST");
    check(DecodeEndpointType(-1) == EP_INVALID, "endpoint -1 -> INVALID");
    // 超出 core/host/mem 全部保留区 -> INVALID
    check(DecodeEndpointType(TOTAL_CORES + 2 * DIE_COUNT + 1) == EP_INVALID,
          "endpoint above all reserved regions -> INVALID");

    // ---- 3. 矩形 GetInputSource（Y 轴用 GRID_Y）----
    setTopo(2, 3, 1, 1); // 2 列 3 行
    // pos0=(x0,y0): NORTH -> y=(0+1)%3=1 -> tile 2; SOUTH -> y=(0-1+3)%3=2 -> tile 4
    check(GetInputSource(NORTH, 0) == 2, "rect NORTH neighbor uses GRID_Y (2x3)");
    check(GetInputSource(SOUTH, 0) == 4, "rect SOUTH wrap uses GRID_Y (2x3)");
    check(GetInputSource(EAST, 0) == 1, "rect EAST neighbor (2x3)");
    // 若误用 GRID_X(=2) 作 Y 模，NORTH(0) 会得到 y=(1)%2=1 -> tile2（巧合相同），
    // 但 pos4=(x0,y2): NORTH -> y=(2+1)%3=0 -> tile0；误用 %2 会得 (3)%2=1 -> tile2
    check(GetInputSource(NORTH, 4) == 0, "rect NORTH wrap top->bottom (GRID_Y)");

    // ---- 4. IsDieEdge（EAST=x+，NORTH=y+）----
    setTopo(2, 3, 1, 1);
    check(IsDieEdge(0, SOUTH) && IsDieEdge(0, WEST), "core0 = S,W edge");
    check(IsDieEdge(GRID_SIZE - 1, NORTH) && IsDieEdge(GRID_SIZE - 1, EAST),
          "top-right core = N,E edge");
    check(!IsDieEdge(0, NORTH) && !IsDieEdge(0, EAST), "core0 not N,E edge");

    // ---- 4b. OpenMeshNeighbor：无环绕、不跨 die ----
    setTopo(4, 4, 2, 1); // 2 die，各 4x4；CORES_PER_DIE=16
    // die0 core0 (x0,y0)
    check(OpenMeshNeighbor(0, WEST) == -1, "core0 WEST edge -> -1 (no wrap)");
    check(OpenMeshNeighbor(0, SOUTH) == -1, "core0 SOUTH edge -> -1 (no wrap)");
    check(OpenMeshNeighbor(0, EAST) == 1, "core0 EAST -> 1");
    check(OpenMeshNeighbor(0, NORTH) == 4, "core0 NORTH -> 4");
    // die0 top-right core 15 (x3,y3): E/N 边缘 -> -1（不跨 die、不 wrap）
    check(OpenMeshNeighbor(15, EAST) == -1,
          "die0 core15 EAST -> -1 (die edge, no cross-die/wrap)");
    check(OpenMeshNeighbor(15, NORTH) == -1, "die0 core15 NORTH -> -1");
    // die1 local0 = global16 (x0,y0): W/S 边缘 -> -1（不跨回 die0）
    check(OpenMeshNeighbor(16, WEST) == -1,
          "die1 core16 WEST -> -1 (no cross-die to die0)");
    check(OpenMeshNeighbor(16, EAST) == 17, "die1 core16 EAST -> 17 (in-die)");
    // 遍历：所有边缘方向必须 -1，所有内部方向必须在同 die 内
    {
        bool ok = true;
        for (int g = 0; g < TOTAL_CORES; g++) {
            for (int d = 0; d < 4; d++) {
                int nb = OpenMeshNeighbor(g, (Directions)d);
                if (nb >= 0 && nb / CORES_PER_DIE != g / CORES_PER_DIE)
                    ok = false; // 邻居跨 die -> 错误
                if (nb == -1 && !IsDieEdge(g % CORES_PER_DIE, (Directions)d))
                    ok = false; // 非边缘却无邻居 -> 错误
            }
        }
        check(ok, "OpenMeshNeighbor: neighbors stay in-die, -1 iff die edge");
    }

    // ---- 5. die-mesh 构造 1x1/2x1/1x2/2x2 ----
    setTopo(4, 4, 1, 1);
    check(TOTAL_CORES == 16 && DIE_COUNT == 1, "1x1 die: TOTAL_CORES=16");
    setTopo(4, 4, 2, 1);
    check(TOTAL_CORES == 32 && DIE_COUNT == 2, "2x1 die: TOTAL_CORES=32");
    setTopo(4, 4, 1, 2);
    check(TOTAL_CORES == 32 && DIE_COUNT == 2, "1x2 die: TOTAL_CORES=32");
    setTopo(4, 4, 2, 2);
    check(TOTAL_CORES == 64 && DIE_COUNT == 4, "2x2 die: TOTAL_CORES=64");

    // ---- 6. 端口配置校验 ----
    // 6a. 合法 2x1（需 E,W c2c + 1 个 host）
    setTopo(4, 4, 2, 1);
    {
        D2DJson hw;
        hw["die_ports"]["edges"]["E"] = {{"role", "c2c"}, {"dir", "E"}};
        hw["die_ports"]["edges"]["W"] = {{"role", "c2c"}, {"dir", "W"}};
        hw["die_ports"]["edges"]["S"] = {{"role", "host"}};
        hw["die_ports"]["c2c"] = {{"link_bw", 4}, {"latency", 20}, {"buffer_depth", 8}};
        bool ok = !throwsRuntime(hw);
        check(ok, "valid 2x1 die_ports parses");
        check(g_die_ports.active && (int)g_die_ports.PortsForDir(EAST).size() >= 1,
              "E direction has C2C ports");
        // port_for_host 每核有定义（>=0）
        bool all_host = true;
        for (int c = 0; c < CORES_PER_DIE; c++)
            if (PortForHost(c) < 0)
                all_host = false;
        check(all_host, "port_for_host defined for every core");
    }

    // 6b. 缺方向端口（DIE_X>1 但没有 E c2c）-> 报错
    setTopo(4, 4, 2, 1);
    {
        D2DJson hw;
        hw["die_ports"]["edges"]["W"] = {{"role", "c2c"}, {"dir", "W"}};
        hw["die_ports"]["edges"]["S"] = {{"role", "host"}};
        check(throwsRuntime(hw), "missing E-direction C2C -> error");
    }

    // 6c. HOST 不可达（全 c2c/mem，无 host）-> 报错
    setTopo(4, 4, 2, 1);
    {
        D2DJson hw;
        hw["die_ports"]["edges"]["E"] = {{"role", "c2c"}, {"dir", "E"}};
        hw["die_ports"]["edges"]["W"] = {{"role", "c2c"}, {"dir", "W"}};
        hw["die_ports"]["edges"]["N"] = {{"role", "mem"}};
        hw["die_ports"]["edges"]["S"] = {{"role", "mem"}};
        check(throwsRuntime(hw), "no HOST port -> HOST unreachable error");
    }

    // 6d. 重复 override -> 报错
    setTopo(4, 4, 2, 1);
    {
        D2DJson hw;
        hw["die_ports"]["edges"]["E"] = {{"role", "c2c"}, {"dir", "E"}};
        hw["die_ports"]["edges"]["W"] = {{"role", "c2c"}, {"dir", "W"}};
        hw["die_ports"]["edges"]["S"] = {{"role", "host"}};
        hw["die_ports"]["overrides"] = D2DJson::array();
        hw["die_ports"]["overrides"].push_back({{"side", "E"}, {"idx", 0}, {"role", "mem"}});
        hw["die_ports"]["overrides"].push_back({{"side", "E"}, {"idx", 0}, {"role", "host"}});
        check(throwsRuntime(hw), "duplicate (side,idx) override -> error");
    }

    // 6e. side!=dir 的 C2C -> 报错（MVP 约束）
    setTopo(4, 4, 2, 1);
    {
        D2DJson hw;
        hw["die_ports"]["edges"]["E"] = {{"role", "c2c"}, {"dir", "E"}};
        hw["die_ports"]["edges"]["W"] = {{"role", "c2c"}, {"dir", "W"}};
        hw["die_ports"]["edges"]["N"] = {{"role", "c2c"}, {"dir", "E"}}; // side N, dir E
        hw["die_ports"]["edges"]["S"] = {{"role", "host"}};
        check(throwsRuntime(hw), "C2C side!=dir -> MVP error");
    }

    // 6f. 无 die_ports -> no-op、active=false、port_for_host 全 -1（兼容旧行为）
    setTopo(4, 4, 1, 1);
    {
        D2DJson hw;
        ParseDiePorts(hw);
        check(!g_die_ports.active, "no die_ports -> inactive (legacy)");
        check(PortForHost(0) == -1, "legacy: port_for_host = -1");
    }

    // 6g. 非法物理参数（link_bw=0 / latency=-1 / buffer_depth=0）-> 报错
    setTopo(4, 4, 1, 1);
    {
        auto mk = [](int bw, int lat, int buf) {
            D2DJson hw;
            hw["die_ports"]["edges"]["W"] = {{"role", "host"}};
            hw["die_ports"]["c2c"] = {
                {"link_bw", bw}, {"latency", lat}, {"buffer_depth", buf}};
            return hw;
        };
        check(throwsRuntime(mk(0, 20, 8)), "link_bw=0 -> error");
        check(throwsRuntime(mk(4, -1, 8)), "latency=-1 -> error");
        check(throwsRuntime(mk(4, 20, 0)), "buffer_depth=0 -> error");
    }

    // ---- 6h. V0b-4 D2D peer/link 构造 + 校验 ----
    setTopo(4, 4, 2, 1); // 2 die 水平相邻
    {
        D2DJson hw;
        hw["die_ports"]["edges"]["E"] = {{"role", "c2c"}, {"dir", "E"}};
        hw["die_ports"]["edges"]["W"] = {{"role", "c2c"}, {"dir", "W"}};
        hw["die_ports"]["edges"]["S"] = {{"role", "host"}};
        hw["die_ports"]["c2c"] = {{"link_bw", 4}, {"latency", 20}, {"buffer_depth", 8}};
        bool ok = !throwsRuntime(hw);
        check(ok, "2x1 E/W c2c: parse + BuildD2DLinks ok");
        // die0 E(4) -> die1, die1 W(4) -> die0 = 8 有向 link
        check((int)g_d2d_links.size() == 8, "2x1 full-edge: 8 directed D2D links");
        bool bw_ok = true, recip_ok = true;
        // 精确四元组互反：link (a,b)->(c,d) 必存在反向 (c,d)->(a,b)
        std::set<std::tuple<int, int, int, int>> links;
        for (auto &l : g_d2d_links) {
            if (l.tx_bw != l.rx_bw || l.latency != 20)
                bw_ok = false;
            links.insert({l.local_die, l.local_port, l.remote_die, l.remote_port});
        }
        for (auto &l : g_d2d_links)
            if (!links.count({l.remote_die, l.remote_port, l.local_die, l.local_port}))
                recip_ok = false;
        check(bw_ok, "D2D links: tx_bw==rx_bw, latency propagated");
        check(recip_ok, "D2D links: exact 4-tuple reciprocity (remote->local)");
    }
    // 单 die：即便有 c2c 端口也无 die 邻居 -> 0 link
    setTopo(4, 4, 1, 1);
    {
        D2DJson hw;
        hw["die_ports"]["edges"]["W"] = {{"role", "host"}};
        hw["die_ports"]["edges"]["E"] = {{"role", "c2c"}, {"dir", "E"}};
        ParseDiePorts(hw);
        check(g_d2d_links.empty(), "single die: no D2D links (all boundary)");
    }
    // 负向：镜像坐标不匹配（E 只在 row0、W 只在 row1）-> BuildD2DLinks 报错
    setTopo(4, 4, 2, 1);
    {
        D2DJson hw;
        hw["die_ports"]["edges"]["S"] = {{"role", "host"}};
        hw["die_ports"]["overrides"] = D2DJson::array();
        hw["die_ports"]["overrides"].push_back(
            {{"side", "E"}, {"idx", 0}, {"role", "c2c"}, {"dir", "E"}});
        hw["die_ports"]["overrides"].push_back(
            {{"side", "W"}, {"idx", 1}, {"role", "c2c"}, {"dir", "W"}});
        check(throwsRuntime(hw),
              "D2D links: mirror coord mismatch (E@row0,W@row1) -> error");
    }
    // 重复解析清空：再解析一个无 die_ports 配置，g_d2d_links 必须清空
    setTopo(4, 4, 1, 1);
    {
        D2DJson hw;
        ParseDiePorts(hw);
        check(g_d2d_links.empty(),
              "re-parse without die_ports clears g_d2d_links (no stale)");
    }

    // ---- 6i. V0b-2C1 workload 归一化 + 结构校验（纯函数，不跑仿真）----
    setTopo(4, 4, 2, 1); // CORES_PER_DIE=16, TOTAL_CORES=32
    {
        auto throwsStruct = [](const WLJson &j) -> bool {
            try {
                ValidateWorkloadStructure(j, 0);
            } catch (...) {
                return true;
            }
            return false;
        };
        // 构造一个 die0-local workload：core0(cast->core1), core1；source->core0
        auto mkWL = []() {
            WLJson w;
            WLJson chip;
            chip["cores"] = WLJson::array();
            WLJson c0;
            c0["id"] = 0;
            c0["prim_copy"] = 1;
            WLJson wl0;
            wl0["cast"] = WLJson::array();
            wl0["cast"].push_back({{"dest", 1}, {"tag", 1}});
            c0["worklist"] = WLJson::array();
            c0["worklist"].push_back(wl0);
            chip["cores"].push_back(c0);
            WLJson c1;
            c1["id"] = 1;
            chip["cores"].push_back(c1);
            w["chips"] = WLJson::array();
            w["chips"].push_back(chip);
            w["source"] = WLJson::array();
            w["source"].push_back({{"dest", 0}, {"size", 1}});
            return w;
        };

        // die0-local 结构合法
        WLJson w0 = mkWL();
        check(!throwsStruct(w0), "structure: die0-local workload valid");

        // 归一化到 die1：所有 core-ref id += 16，id_space=global
        WLJson w1 = mkWL();
        NormalizeWorkloadJson(w1, 1);
        check(w1["chips"][0]["cores"][0]["id"] == 16 &&
                  w1["chips"][0]["cores"][1]["id"] == 17,
              "normalize die1: core id 0/1 -> 16/17");
        check(w1["chips"][0]["cores"][0]["prim_copy"] == 17,
              "normalize die1: prim_copy 1 -> 17");
        check(w1["chips"][0]["cores"][0]["worklist"][0]["cast"][0]["dest"] == 17,
              "normalize die1: cast.dest 1 -> 17");
        check(w1["source"][0]["dest"] == 16, "normalize die1: source.dest 0 -> 16");
        check(w1["id_space"] == "global", "normalize sets id_space=global");
        // die1-local（core16 -> core17）结构合法（允许，非跨 die）
        check(!throwsStruct(w1),
              "structure: die1-local (16->17) allowed (not cross-die)");

        // 跨 die：global workload，core0 cast -> core16 → 拒绝
        WLJson wx = mkWL();
        wx["id_space"] = "global";
        wx["chips"][0]["cores"][0]["worklist"][0]["cast"][0]["dest"] = 16;
        check(throwsStruct(wx), "structure: cross-die core0->core16 rejected");

        // die0-local id_space 下引用 die>0 id → 拒绝
        WLJson wb = mkWL();
        wb["chips"][0]["cores"][0]["worklist"][0]["cast"][0]["dest"] = 16;
        check(throwsStruct(wb), "structure: die0-local id_space + die>0 id -> error");

        // 越界 id → 拒绝
        WLJson wo = mkWL();
        wo["id_space"] = "global";
        wo["chips"][0]["cores"][0]["worklist"][0]["cast"][0]["dest"] = 999;
        check(throwsStruct(wo), "structure: out-of-range id (999>=TOTAL_CORES) -> error");

        // 归一化 die0（die_id=0）为恒等
        WLJson wid = mkWL();
        NormalizeWorkloadJson(wid, 0);
        check(wid["chips"][0]["cores"][0]["id"] == 0,
              "normalize die0 is identity (compat)");
    }

    // ---- 7. MEM endpoint 解码（core/host/mem 三分）----
    setTopo(4, 4, 2, 1); // TOTAL_CORES=32, DIE_COUNT=2；host[32,34) mem[34,36)
    check(DecodeEndpointType(31) == EP_CORE, "endpoint 31 -> CORE");
    check(DecodeEndpointType(32) == EP_HOST && DecodeEndpointType(33) == EP_HOST,
          "endpoint 32/33 -> HOST (per-die)");
    check(DecodeEndpointType(34) == EP_MEM && DecodeEndpointType(35) == EP_MEM,
          "endpoint 34/35 -> MEM");
    check(DecodeEndpointType(36) == EP_INVALID, "endpoint 36 -> INVALID");

    // ---- 7b. 每 die HOST endpoint 语义（不再只认 die0）----
    setTopo(4, 4, 2, 1); // TOTAL_CORES=32, DIE_COUNT=2; host {32,33}
    BuildHostAttach();
    HOST_LANES = g_host_attach.n_lanes;
    check(IsHostEndpoint(32) && IsHostEndpoint(33),
          "IsHostEndpoint covers all per-die hosts (32,33)");
    check(!IsHostEndpoint(31) && !IsHostEndpoint(34),
          "IsHostEndpoint excludes core(31) and mem(34)");
    check(HostEndpointOfDie(0) == 32 && HostEndpointOfDie(1) == 33,
          "HostEndpointOfDie(0/1) = 32/33");
    check(DieOfHostEndpoint(33) == 1, "DieOfHostEndpoint(33) = 1");
    // die1 的核去 die1 HOST(33)：在 die1 自己的挂载 tile(西边缘)发 HOST，非 core 坐标。
    // （2b 后 GetNextHop 朝 pos 本 die 的 HOST tile 收敛——他 die 的 HOST 不当作本地出口。）
    check(GetNextHop(33, 16, 16) == HOST,
          "GetNextHop: die1 core16 (anchor) at its own die1 HOST tile -> HOST");
    check(GetNextHop(33, 17, 17) == WEST,
          "GetNextHop: die1 core17 (anchor) -> toward die1 HOST tile (west margin)");
    // 跨 die HOST 必须显式拒绝：anchor=die0 核(0)去 die1 HOST(33)——anchor die != host die。
    {
        bool fthrew = false, rthrew = false;
        std::string fwhat;
        try {
            GetNextHop(33, 0, 0);
        } catch (const std::runtime_error &e) {
            fthrew = true;
            fwhat = e.what();
        }
        try {
            GetNextHopReverse(33, 0, 0);
        } catch (const std::runtime_error &) {
            rthrew = true;
        }
        check(fthrew && fwhat.find("cross-die HOST") != std::string::npos,
              "GetNextHop: die0 anchor -> die1 HOST throws (cross-die, no silent HOST)");
        check(rthrew,
              "GetNextHopReverse: die0 anchor -> die1 HOST throws (cross-die)");
    }
    // 缺 anchor（-1）时 HOST 路由必须报错，杜绝隐式退回 pos（正反向都测）
    {
        bool fthrew = false, rthrew = false;
        try {
            GetNextHop(32, 0, -1);
        } catch (const std::runtime_error &) {
            fthrew = true;
        }
        try {
            GetNextHopReverse(32, 0, -1);
        } catch (const std::runtime_error &) {
            rthrew = true;
        }
        check(fthrew, "GetNextHop: HOST dest without anchor throws (no implicit pos)");
        check(rthrew,
              "GetNextHopReverse: HOST dest without anchor throws (no implicit pos)");
    }
    // 非法映射（anchor 的 core_lane 被置 -1）→ 路由明确报错（不静默走 des=-1）
    {
        setTopo(4, 4, 1, 1);
        BuildHostAttach();
        HOST_LANES = g_host_attach.n_lanes;
        int saved = g_host_attach.core_lane[5];
        g_host_attach.core_lane[5] = -1; // 人为破坏映射
        bool threw = false;
        try {
            GetNextHop(HostEndpointOfDie(0), 1, 5);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        g_host_attach.core_lane[5] = saved;
        check(threw,
              "GetNextHop: anchor with illegal lane/tile mapping throws clearly");
    }

    // ---- 8. 消息逐字段序列化往返（全字段、非零值以捕获 stuck-at-0）----
    {
        Msg m;
        m.is_end_ = true;             // 1-bit -> 用 true 捕获恒 0
        m.msg_type_ = REQUEST;        // 4-bit
        m.seq_id_ = 1234;             // 16-bit
        m.des_ = 17;                  // 16-bit（全局 id）
        m.offset_ = 200;              // 8-bit
        m.tag_id_ = 42;               // 16-bit
        m.source_ = 5;                // 16-bit（全局 source）
        m.length_ = 99;               // 8-bit
        m.refill_ = true;             // 1-bit -> 显式比较
        m.config_end_ = true;         // 1-bit -> 显式比较
        m.roofline_packets_ = 7;      // 24-bit
        m.data_ = sc_bv<128>(0xABCD);
        Msg r = DeserializeMsg(SerializeMsg(m));
        // 逐字段全比较（含 refill_/config_end_）
        check(r.is_end_ == m.is_end_, "round-trip: is_end_");
        check(r.msg_type_ == m.msg_type_, "round-trip: msg_type_");
        check(r.seq_id_ == m.seq_id_, "round-trip: seq_id_");
        check(r.des_ == m.des_, "round-trip: des_");
        check(r.offset_ == m.offset_, "round-trip: offset_");
        check(r.tag_id_ == m.tag_id_, "round-trip: tag_id_");
        check(r.source_ == m.source_, "round-trip: source_");
        check(r.length_ == m.length_, "round-trip: length_");
        check(r.refill_ == m.refill_, "round-trip: refill_ (true)");
        check(r.config_end_ == m.config_end_, "round-trip: config_end_ (true)");
        check(r.roofline_packets_ == m.roofline_packets_,
              "round-trip: roofline_packets_");
        check(r.data_ == m.data_, "round-trip: data_");
        // 默认构造的 Msg 全成员有确定初值（无未初始化字段进入序列化）
        Msg d;
        Msg rd = DeserializeMsg(SerializeMsg(d));
        check(rd.roofline_packets_ == d.roofline_packets_ &&
                  rd.length_ == d.length_ && rd.offset_ == d.offset_,
              "default-constructed Msg serializes deterministically");
        // 边界：最大 16-bit des_/source_/tag_
        Msg m2;
        m2.des_ = 65535;
        m2.source_ = 65535;
        m2.tag_id_ = 65535;
        Msg r2 = DeserializeMsg(SerializeMsg(m2));
        check(r2.des_ == 65535 && r2.source_ == 65535 && r2.tag_id_ == 65535,
              "16-bit endpoint/tag boundary round-trip (max 65535)");
    }

    // ---- 8b. endpoint 地址空间容量校验 ----
    {
        setTopo(4, 4, 2, 2); // 64 cores + 8 = 72 <= 65536
        bool ok = true;
        try {
            ValidateAddressSpace();
        } catch (...) {
            ok = false;
        }
        check(ok, "valid dims pass address-space check");

        setTopo(256, 256, 1, 1); // 65536 cores + 2 > 65536
        bool threw = false;
        try {
            ValidateAddressSpace();
        } catch (const std::runtime_error &) {
            threw = true;
        }
        check(threw, "endpoint space > 65536 -> error (16-bit overflow guard)");

        GRID_X = 0; // 非正维度
        threw = false;
        try {
            ValidateAddressSpace();
        } catch (const std::runtime_error &) {
            threw = true;
        }
        check(threw, "non-positive dimension -> error");
    }

    // ---- 9. flow/link 统计接口：单 die 下 D2D 活动计数为 0 ----
    setTopo(4, 4, 1, 1);
    {
        D2DJson hw;
        hw["die_ports"]["edges"]["W"] = {{"role", "host"}};
        ParseDiePorts(hw);
        check(g_die_ports.TotalActivity() == 0,
              "single-die: D2D port/link activity count == 0");
    }

    // ---- 10. HOST 挂载表（V1-pre 增量 2a）：结构不变量 + legacy 精确值 ----
    // 覆盖：core→lane 范围、lane→tile 合法且与该核同 die、legacy 精确公式
    //       (lane==gid/GRID_X, tile==lane*GRID_X)、非法 core/lane 报 -1。
    {
        auto checkAttach = [&](int gx, int gy, int dx, int dy) {
            setTopo(gx, gy, dx, dy);
            BuildHostAttach();
            HOST_LANES = g_host_attach.n_lanes; // 生产路径同款：表为真源
            std::string tag = "attach " + std::to_string(gx) + "x" +
                              std::to_string(gy) + " die" + std::to_string(dx) +
                              "x" + std::to_string(dy);
            check(g_host_attach.n_lanes == gy * dx * dy,
                  tag + ": n_lanes == GRID_Y*DIE_COUNT");
            check((int)g_host_attach.core_lane.size() == TOTAL_CORES,
                  tag + ": core_lane covers all cores");

            bool lane_range = true, lane_exact = true, tile_legal = true,
                 tile_exact = true, same_die = true;
            for (int c = 0; c < TOTAL_CORES; c++) {
                int lane = HostLaneOfCore(c);
                if (lane < 0 || lane >= g_host_attach.n_lanes)
                    lane_range = false;
                if (lane != c / GRID_X)
                    lane_exact = false;
                int tile = HostTileOfLane(lane);
                if (tile < 0 || tile >= TOTAL_CORES)
                    tile_legal = false;
                if (tile != lane * GRID_X)
                    tile_exact = false;
                if (DieOfGlobal(tile) != DieOfGlobal(c))
                    same_die = false;
            }
            check(lane_range, tag + ": every core lane in [0, n_lanes)");
            check(lane_exact, tag + ": legacy lane == gid/GRID_X");
            check(tile_legal, tag + ": every lane tile in [0, TOTAL_CORES)");
            check(tile_exact, tag + ": legacy tile == lane*GRID_X");
            check(same_die, tag + ": HOST tile shares the core's die");

            // 非法 core/lane -> -1（Monitor 不会用 routers[-1]）
            check(HostLaneOfCore(-1) == -1 &&
                      HostLaneOfCore(TOTAL_CORES) == -1,
                  tag + ": illegal core -> lane -1");
            check(HostTileOfLane(-1) == -1 &&
                      HostTileOfLane(g_host_attach.n_lanes) == -1,
                  tag + ": illegal lane -> tile -1");

            // ValidateHostAttach：合法配置通过；HOST_LANES 与表不一致时报错
            bool ok = true;
            try {
                ValidateHostAttach();
            } catch (...) {
                ok = false;
            }
            check(ok, tag + ": ValidateHostAttach passes on legal table");
            int save = HOST_LANES;
            HOST_LANES = save + 1;
            bool threw = false;
            try {
                ValidateHostAttach();
            } catch (const std::runtime_error &) {
                threw = true;
            }
            check(threw, tag + ": ValidateHostAttach detects HOST_LANES mismatch");
            HOST_LANES = save;
        };
        checkAttach(4, 4, 1, 1); // 单 die 方阵
        checkAttach(4, 4, 2, 1); // 横向双 die
        checkAttach(4, 4, 2, 2); // 2D die-mesh
        checkAttach(4, 2, 1, 1); // 矩形 die 内 mesh
        checkAttach(2, 4, 2, 1); // 矩形 + 多 die
    }

    // ---- 11. LegacyHostEnqueue 非法 dest 防御（专项，防检查被误删）----
    {
        setTopo(4, 4, 1, 1);
        BuildHostAttach();
        HOST_LANES = g_host_attach.n_lanes;
        std::queue<Msg> q[64]; // >= HOST_LANES

        // 非法 dest（越界 -> HostLaneOfCore == -1）必须抛异常且报明确文本，不写 q[-1]
        std::vector<HostEnvelope> bad = {{TOTAL_CORES, Msg()}};
        bool threw = false;
        std::string what;
        try {
            LegacyHostEnqueue(bad, q);
        } catch (const std::runtime_error &e) {
            threw = true;
            what = e.what();
        }
        check(threw && what.find("no valid HOST lane") != std::string::npos,
              "LegacyHostEnqueue rejects illegal dest (no q[-1])");

        // 合法 dest 仍正常落入其 lane
        std::vector<HostEnvelope> good = {{5, Msg()}};
        bool ok = true;
        try {
            LegacyHostEnqueue(good, q);
        } catch (...) {
            ok = false;
        }
        check(ok && q[HostLaneOfCore(5)].size() == 1,
              "LegacyHostEnqueue enqueues legal dest to its lane");
    }

    // ---- 12. HOST 路由收敛（V1-pre 2b）：逐跳走向本 die HOST，距离严格下降、不跨 die、无环 ----
    {
        auto absi = [](int v) { return v < 0 ? -v : v; };
        auto walkAll = [&](int gx, int gy, int dx, int dy, bool rev) {
            setTopo(gx, gy, dx, dy);
            BuildHostAttach();
            HOST_LANES = g_host_attach.n_lanes;
            auto man = [&](int a, int b) {
                return absi(a % GRID_X - b % GRID_X) +
                       absi(a / GRID_X - b / GRID_X);
            };
            bool all = true;
            for (int c = 0; c < TOTAL_CORES && all; c++) {
                int die = c / CORES_PER_DIE;
                int hostep = HostEndpointOfDie(die);
                int tile = HostTileOfLane(HostLaneOfCore(c));
                int pos = c, prev = man(pos, tile), steps = 0;
                std::set<int> visited;
                visited.insert(pos);
                while (true) {
                    // anchor 固定为起点核 c；pos 沿途移动（证明用固定 source，不用 pos）
                    Directions d = rev ? GetNextHopReverse(hostep, pos, c)
                                       : GetNextHop(hostep, pos, c);
                    if (d == HOST) {
                        if (pos != tile) all = false; // 只能在挂载 tile 发 HOST
                        break;
                    }
                    if (d != WEST && d != EAST && d != NORTH && d != SOUTH) {
                        all = false; // 非法方向（含意外 CENTER）
                        break;
                    }
                    int nxt = OpenMeshNeighbor(pos, d);
                    if (nxt < 0 || nxt / CORES_PER_DIE != die) {
                        all = false; // 走出 die 边缘 / 跨 die
                        break;
                    }
                    int nd = man(nxt, tile);
                    if (nd >= prev || visited.count(nxt)) {
                        all = false; // 距离未严格下降 / 成环
                        break;
                    }
                    visited.insert(nxt);
                    pos = nxt;
                    prev = nd;
                    if (++steps > CORES_PER_DIE + 2) {
                        all = false; // 不收敛
                        break;
                    }
                }
            }
            std::string tag = "host-route " + std::to_string(gx) + "x" +
                              std::to_string(gy) + " die" + std::to_string(dx) +
                              "x" + std::to_string(dy) + (rev ? " (rev)" : " (fwd)");
            check(all, tag + ": every core converges to HOST tile, dist strictly down, in-die");
        };
        for (int r = 0; r < 2; r++) {
            walkAll(4, 4, 1, 1, r); // 单 die 方阵
            walkAll(4, 4, 2, 1, r); // 横向双 die
            walkAll(4, 4, 2, 2, r); // 2D die-mesh
            walkAll(4, 2, 1, 1, r); // 矩形 die 内 mesh
            walkAll(2, 4, 2, 1, r); // 矩形 + 多 die
        }
    }

    // ---- 13. config 驱动挂载表（V1-pre 3a）：role=HOST 端口建表 + 校验 ----
    {
        // A. W 全边 host（单 die）→ 表与 legacy 西边缘完全相同（保证现有 e2e sim-time 不变）
        setTopo(4, 4, 1, 1);
        {
            D2DJson hw;
            hw["die_ports"]["edges"]["W"] = {{"role", "host"}};
            ParseDiePorts(hw);
            BuildHostAttach();
            HOST_LANES = g_host_attach.n_lanes;
            bool eq = !g_host_attach.legacy && g_host_attach.n_lanes == 4;
            for (int i = 0; i < 4 && eq; i++)
                if (g_host_attach.lane_tile[i] != i * 4)
                    eq = false;
            for (int c = 0; c < 16 && eq; c++)
                if (HostLaneOfCore(c) != c / 4)
                    eq = false;
            bool valOk = true;
            try {
                ValidateHostAttach();
            } catch (...) {
                valOk = false;
            }
            check(eq && valOk,
                  "config W-host single die == legacy west edge (lane/tile/core_lane)");
        }

        // B. S 全边 host（单 die）→ 每列一条 lane，tile=南边缘，core_lane=列号
        setTopo(4, 4, 1, 1);
        {
            D2DJson hw;
            hw["die_ports"]["edges"]["S"] = {{"role", "host"}};
            ParseDiePorts(hw);
            BuildHostAttach();
            HOST_LANES = g_host_attach.n_lanes;
            bool ok = !g_host_attach.legacy && g_host_attach.n_lanes == 4;
            for (int k = 0; k < 4 && ok; k++)
                if (g_host_attach.lane_tile[k] != k) // 南边缘 tile 0..3
                    ok = false;
            for (int c = 0; c < 16 && ok; c++)
                if (HostLaneOfCore(c) != c % 4) // 就近列
                    ok = false;
            bool valOk = true;
            try {
                ValidateHostAttach();
            } catch (...) {
                valOk = false;
            }
            check(ok && valOk,
                  "config S-host single die: per-column lane, tile=south edge, core_lane=col");
        }

        // C. 2x1 die：E/W c2c + S host → n_lanes=hp*DIE_COUNT，同 die 且 ValidateHostAttach 通过
        setTopo(4, 4, 2, 1);
        {
            D2DJson hw;
            hw["die_ports"]["edges"]["W"] = {{"role", "c2c"}, {"dir", "W"}};
            hw["die_ports"]["edges"]["E"] = {{"role", "c2c"}, {"dir", "E"}};
            hw["die_ports"]["edges"]["S"] = {{"role", "host"}};
            ParseDiePorts(hw);
            BuildHostAttach();
            HOST_LANES = g_host_attach.n_lanes;
            bool ok = !g_host_attach.legacy && g_host_attach.n_lanes == 4 * 2;
            for (int c = 0; c < TOTAL_CORES && ok; c++) {
                int lane = HostLaneOfCore(c);
                if (lane < 0 || lane >= g_host_attach.n_lanes)
                    ok = false;
                else if (HostTileOfLane(lane) / CORES_PER_DIE != c / CORES_PER_DIE)
                    ok = false; // HOST tile 必须与核同 die
            }
            bool valOk = true;
            try {
                ValidateHostAttach();
            } catch (...) {
                valOk = false;
            }
            check(ok && valOk,
                  "config 2x1 die E/W-c2c + S-host: hp*DIE_COUNT lanes, tile same-die");
        }

        // D. 一个物理端口只能有一个 role：两个 override 命中同一 (side,idx) → 拒绝。
        //    （这命中的是已有的「重复 (side,idx) override」校验路径，功能上保证同一端口不能
        //     既 host 又 c2c，但不是新的专用校验。）
        setTopo(4, 4, 1, 1);
        {
            D2DJson hw;
            hw["die_ports"]["overrides"] = D2DJson::array();
            hw["die_ports"]["overrides"].push_back(
                {{"side", "W"}, {"idx", 0}, {"role", "host"}});
            hw["die_ports"]["overrides"].push_back(
                {{"side", "W"}, {"idx", 0}, {"role", "c2c"}, {"dir", "W"}});
            check(throwsRuntime(hw),
                  "config: duplicate (side,idx) override rejected (one role per port)");
        }

        // E. 一个 tile 只能承载一条 HOST lane：S+W 全边 host → 西南角 tile0 被两条 lane 挂载，
        //    ValidateHostAttach 必须拒绝（当前 RouterUnit 每 tile 仅一套 HOST 接口）。
        setTopo(4, 4, 1, 1);
        {
            D2DJson hw;
            hw["die_ports"]["edges"]["W"] = {{"role", "host"}};
            hw["die_ports"]["edges"]["S"] = {{"role", "host"}};
            ParseDiePorts(hw); // 端口解析本身合法（W/S 不同 side，无 (side,idx) 冲突）
            BuildHostAttach();
            HOST_LANES = g_host_attach.n_lanes;
            bool threw = false;
            try {
                ValidateHostAttach();
            } catch (const std::runtime_error &) {
                threw = true;
            }
            check(threw,
                  "config S+W host: corner tile double-attach rejected (per-tile single HOST)");
        }
    }

    // ---- 14. egress anchor 稳定性（V1-pre 3b-2）：路由用固定 source anchor，绝不退回 pos ----
    {
        setTopo(4, 4, 1, 1);
        BuildHostAttach();
        HOST_LANES = g_host_attach.n_lanes;
        int hostep = HostEndpointOfDie(0);
        int A = 0; // 行0：anchor，其挂载 tile = 0
        int P = 4; // 行1,col0：P 恰是 lane1 的挂载 tile；若隐式用 pos 会误判 pos==tile→HOST
        bool pre = HostLaneOfCore(A) != HostLaneOfCore(P); // 前提：两者 lane 不同
        // 固定 anchor=A：应朝 A 的挂载 tile(行0)收敛 = SOUTH（行1→行0），绝不 HOST
        Directions dA = GetNextHop(hostep, P, A);
        Directions dAr = GetNextHopReverse(hostep, P, A);
        // 对照：anchor 取成 pos(P) 时 P 恰是自己挂载 tile → HOST（证明二者结果确实不同）
        Directions dP = GetNextHop(hostep, P, P);
        check(pre && dA == SOUTH && dAr == SOUTH && dP == HOST,
              "egress anchor: routing uses fixed source anchor (A), never falls back to pos");
    }

    std::cout << "==== D2D V0 self-test: " << (g_total - g_fail) << "/" << g_total
              << " passed" << (g_fail ? "  <<< FAILURES" : "") << " ====" << std::endl;
    return g_fail;
}
