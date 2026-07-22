// D2D V0 L0 自测：编址、端点解码、矩形拓扑、端口配置校验。
// 不依赖 SystemC 运行；直接设置全局维度并调用纯函数。
#include "die/port.h"
#include "defs/spec.h"
#include "common/flow.h"
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
    g_d2d_links.clear();
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
        m.exit_port_ = 42;            // 8-bit（V1-c0 pinned 出口端口）
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
        check(r.exit_port_ == m.exit_port_, "round-trip: exit_port_ (42)");
        check(r.data_ == m.data_, "round-trip: data_");
        // exit_port_ 编码边界（0=未 pin，port=port_id+1）：-1/0/254/255 均正确 round-trip
        bool ep_ok = true;
        for (int v : {-1, 0, 254, 255, 4096}) {
            Msg mp;
            mp.exit_port_ = v;
            if (DeserializeMsg(SerializeMsg(mp)).exit_port_ != v)
                ep_ok = false;
        }
        check(ep_ok, "round-trip: exit_port_ boundaries (-1/0/254/255/4096)");
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

    // ---- 15. 跨 die 路由（V1-a，纯函数）：die 级首跳/距离 + 单端口 pinning + V1 MVP 校验 ----
    {
        // 15.1 die 级首跳方向 + Manhattan 距离
        setTopo(4, 4, 2, 1);
        check(DieFirstHopDir(0, 1) == EAST && DieFirstHopDir(1, 0) == WEST &&
                  DieManhattan(0, 1) == 1,
              "die-mesh 2x1: first-hop E/W, adjacent (dist 1)");
        setTopo(4, 4, 2, 2); // die0(0,0) die1(1,0) die2(0,1) die3(1,1)
        check(DieFirstHopDir(0, 3) == EAST && DieFirstHopDir(0, 2) == NORTH &&
                  DieFirstHopDir(3, 0) == WEST && DieManhattan(0, 3) == 2 &&
                  DieManhattan(0, 1) == 1,
              "die-mesh 2x2: X-first XY; die0->die3 non-adjacent (dist 2, V1-c rejects)");

        // man()：读当前 GRID_X 的片内 Manhattan
        auto man = [&](int a, int b) {
            int dv = a % GRID_X - b % GRID_X, dh = a / GRID_X - b / GRID_X;
            return (dv < 0 ? -dv : dv) + (dh < 0 ? -dh : dh);
        };
        // 单端口 pinning walk：每 die 只 CrossDieSelectExit 一次，携固定 exit_port 逐跳 step。
        auto walkPinned = [&](int src_die, int des_die, Directions egress) {
            bool all = true;
            for (int lc = 0; lc < CORES_PER_DIE && all; lc++) {
                int c = src_die * CORES_PER_DIE + lc;
                int des = des_die * CORES_PER_DIE + lc;
                int ep = CrossDieSelectExit(c, des); // 选一次并钉死
                int ptile = src_die * CORES_PER_DIE + g_die_ports.ports[ep].tile;
                int pos = c, prev = man(pos, ptile), steps = 0;
                std::set<int> visited;
                visited.insert(pos);
                while (true) {
                    Directions d = CrossDieStep(des, pos, ep); // 携固定 ep
                    if (pos == ptile) {
                        if (d != egress) all = false;
                        break;
                    }
                    if (d != WEST && d != EAST && d != NORTH && d != SOUTH) {
                        all = false;
                        break;
                    }
                    int nxt = OpenMeshNeighbor(pos, d);
                    if (nxt < 0 || nxt / CORES_PER_DIE != src_die) {
                        all = false;
                        break;
                    }
                    int nd = man(nxt, ptile);
                    if (nd >= prev || visited.count(nxt)) {
                        all = false;
                        break;
                    }
                    visited.insert(nxt);
                    pos = nxt;
                    prev = nd;
                    if (++steps > CORES_PER_DIE + 2) {
                        all = false;
                        break;
                    }
                }
            }
            return all;
        };

        // 15.2 单端口 E/W（2×1 4×4）：override 单个 E/W 端口（非整边），S 整边 host 保可达
        {
            setTopo(4, 4, 2, 1);
            D2DJson hw;
            hw["die_ports"]["edges"]["S"] = {{"role", "host"}};
            hw["die_ports"]["overrides"] = D2DJson::array();
            hw["die_ports"]["overrides"].push_back(
                {{"side", "E"}, {"idx", 0}, {"role", "c2c"}, {"dir", "E"}});
            hw["die_ports"]["overrides"].push_back(
                {{"side", "W"}, {"idx", 0}, {"role", "c2c"}, {"dir", "W"}});
            ParseDiePorts(hw);
            check((int)g_die_ports.PortsForDir(EAST).size() == 1 &&
                      (int)g_die_ports.PortsForDir(WEST).size() == 1,
                  "single-port 2x1: exactly one E and one W C2C port");
            bool v1ok = true;
            try {
                ValidateV1MvpTopology();
            } catch (...) {
                v1ok = false;
            }
            check(v1ok, "ValidateV1MvpTopology accepts single-port 2x1 E/W");
            check(walkPinned(0, 1, EAST),
                  "pinned die0->die1: select-once EAST, converge to port tile, egress");
            check(walkPinned(1, 0, WEST),
                  "pinned die1->die0: select-once WEST, converge to port tile, egress");
            // 同 die：CrossDieStep 退回片内 XY（== GetNextHop）
            bool same = true;
            for (int a = 0; a < CORES_PER_DIE && same; a++)
                for (int b = 0; b < CORES_PER_DIE && same; b++)
                    if (CrossDieStep(b, a, -1) != GetNextHop(b, a))
                        same = false;
            check(same, "CrossDieStep same-die == within-die GetNextHop");
            // 非法 core id 报错
            bool badthrew = false;
            try {
                CrossDieSelectExit(-1, 0);
            } catch (const std::runtime_error &) {
                badthrew = true;
            }
            check(badthrew, "CrossDieSelectExit illegal core id throws");

            // CrossDieStep 不信任携带的 exit_port：负例
            int host_pid = -1, w_pid = -1, e_pid = -1;
            for (auto &p : g_die_ports.ports) {
                if (p.role == ROLE_HOST && host_pid < 0)
                    host_pid = p.port_id;
                if (p.role == ROLE_C2C && p.dir == WEST)
                    w_pid = p.port_id;
                if (p.role == ROLE_C2C && p.dir == EAST)
                    e_pid = p.port_id;
            }
            int des1 = CORES_PER_DIE; // die1 核0；从 die0 核0 去，D=EAST
            auto stepThrows = [&](int des, int pos, int ep) {
                try {
                    CrossDieStep(des, pos, ep);
                } catch (const std::runtime_error &) {
                    return true;
                }
                return false;
            };
            check(stepThrows(des1, 0, w_pid),
                  "CrossDieStep rejects wrong-direction pinned exit (W when going E)");
            check(stepThrows(des1, 0, host_pid),
                  "CrossDieStep rejects HOST port impersonating C2C exit");
            check(stepThrows(des1, -1, e_pid),
                  "CrossDieStep rejects illegal core id");
            check(stepThrows(des1, 0, 999),
                  "CrossDieStep rejects out-of-range pinned exit port");
        }

        // 15.3 单端口 N/S（1×2 纵向双 die）
        {
            setTopo(4, 4, 1, 2);
            D2DJson hw;
            hw["die_ports"]["edges"]["W"] = {{"role", "host"}};
            hw["die_ports"]["overrides"] = D2DJson::array();
            hw["die_ports"]["overrides"].push_back(
                {{"side", "N"}, {"idx", 0}, {"role", "c2c"}, {"dir", "N"}});
            hw["die_ports"]["overrides"].push_back(
                {{"side", "S"}, {"idx", 0}, {"role", "c2c"}, {"dir", "S"}});
            ParseDiePorts(hw);
            bool v1ok = true;
            try {
                ValidateV1MvpTopology();
            } catch (...) {
                v1ok = false;
            }
            check(v1ok && g_die_ports.PortsForDir(NORTH).size() == 1 &&
                      g_die_ports.PortsForDir(SOUTH).size() == 1,
                  "single-port 1x2: exactly one N and one S C2C port; V1 MVP ok");
            check(walkPinned(0, 1, NORTH),
                  "pinned die0->die1 (1x2): select-once NORTH, egress N");
            check(walkPinned(1, 0, SOUTH),
                  "pinned die1->die0 (1x2): select-once SOUTH, egress S");
        }

        // 15.4 矩形 4×2 die 内 mesh、单端口 E/W（2 个 die，每 die 8 核）
        {
            setTopo(4, 2, 2, 1);
            D2DJson hw;
            hw["die_ports"]["edges"]["S"] = {{"role", "host"}};
            hw["die_ports"]["overrides"] = D2DJson::array();
            hw["die_ports"]["overrides"].push_back(
                {{"side", "E"}, {"idx", 0}, {"role", "c2c"}, {"dir", "E"}});
            hw["die_ports"]["overrides"].push_back(
                {{"side", "W"}, {"idx", 0}, {"role", "c2c"}, {"dir", "W"}});
            ParseDiePorts(hw);
            bool v1ok = true;
            try {
                ValidateV1MvpTopology();
            } catch (...) {
                v1ok = false;
            }
            check(v1ok, "rectangular 4x2 die-inner mesh single-port E/W: V1 MVP ok");
            check(walkPinned(0, 1, EAST) && walkPinned(1, 0, WEST),
                  "pinned 4x2 die-inner mesh: die0<->die1 converge+egress E/W");
        }

        // 15.5 V1 MVP 拒绝：整边 C2C（多端口）；邻 die 方向缺 C2C（不可达）
        {
            setTopo(4, 4, 2, 1);
            D2DJson mp;
            mp["die_ports"]["edges"]["S"] = {{"role", "host"}};
            mp["die_ports"]["edges"]["E"] = {{"role", "c2c"}, {"dir", "E"}}; // 整边=4端口
            mp["die_ports"]["edges"]["W"] = {{"role", "c2c"}, {"dir", "W"}};
            ParseDiePorts(mp);
            bool rej = false;
            try {
                ValidateV1MvpTopology();
            } catch (const std::runtime_error &) {
                rej = true;
            }
            check(rej, "ValidateV1MvpTopology rejects whole-edge multi-port C2C");

            // 邻 die 方向缺 C2C：2×1 只给 E 单端口（缺 W）——启动期 ValidateDiePorts 已拒
            setTopo(4, 4, 2, 1);
            D2DJson miss;
            miss["die_ports"]["edges"]["S"] = {{"role", "host"}};
            miss["die_ports"]["overrides"] = D2DJson::array();
            miss["die_ports"]["overrides"].push_back(
                {{"side", "E"}, {"idx", 0}, {"role", "c2c"}, {"dir", "E"}});
            check(throwsRuntime(miss),
                  "2x1 with E but no W C2C: rejected at startup (neighbor unreachable)");

            // V1 runtime 前置：多 die 但无 die_ports（无 C2C 通路）必须被 ValidateV1MvpTopology 拒绝
            setTopo(4, 4, 2, 1); // setTopo 已清空 g_die_ports → active=false
            bool mdrej = false;
            try {
                ValidateV1MvpTopology();
            } catch (const std::runtime_error &) {
                mdrej = true;
            }
            check(mdrej,
                  "ValidateV1MvpTopology rejects multi-die without die_ports (no C2C path)");
        }
    }

    // ---- 15b. V2-a 多跳跨 die 路由（纯函数）：每进入一个 die 重新 pin 出口、hop 数=die 距离、
    //          逐跳 die 距离严格 -1；覆盖 3×1 直线与 2×2 对角（die 级 X-first XY）----
    {
        // man()：当前 GRID_X 的片内 Manhattan
        auto man = [&](int a, int b) {
            int dv = a % GRID_X - b % GRID_X, dh = a / GRID_X - b / GRID_X;
            return (dv < 0 ? -dv : dv) + (dh < 0 ? -dh : dh);
        };
        // 配置：所有边默认 host（保 host 可达），逐方向 override idx0 为单 c2c（MVP 每向 ≤1 端口）。
        auto build = [&](std::vector<std::string> c2c_dirs) {
            D2DJson hw;
            for (const char *s : {"N", "S", "E", "W"})
                hw["die_ports"]["edges"][s] = {{"role", "host"}};
            hw["die_ports"]["overrides"] = D2DJson::array();
            for (auto &d : c2c_dirs)
                hw["die_ports"]["overrides"].push_back(
                    {{"side", d}, {"idx", 0}, {"role", "c2c"}, {"dir", d}});
            hw["die_ports"]["c2c"] = {
                {"link_bw", 1}, {"latency", 20}, {"buffer_depth", 8}};
            ParseDiePorts(hw);
        };
        // 多跳 walk：每进入一个 die 用当前 pos 重新 CrossDieSelectExit（每 die 选一次），该 die 内
        // 携固定 ep 逐跳收敛到出口 tile→egress→经 CrossDieIngressTile 落到下一 die 入口。校验：
        // 片内每步向出口 tile 距离严格 -1、egress 方向序列、每跨 link die 距离严格 -1、最终到 des。
        auto walkMulti = [&](int src, int des, std::vector<Directions> exp_egress) {
            int pos = src, hops = 0, guard = 0;
            int prevd = DieManhattan(DieOfGlobal(pos), DieOfGlobal(des));
            std::vector<Directions> egresses;
            while (DieOfGlobal(pos) != DieOfGlobal(des)) {
                int cur_die = DieOfGlobal(pos);
                int ep = CrossDieSelectExit(pos, des); // 进入该 die：选一次
                if (ep < 0)
                    return false;
                int ptile = cur_die * CORES_PER_DIE + g_die_ports.ports[ep].tile;
                while (pos != ptile) { // 携固定 ep 收敛到出口 tile
                    Directions d = CrossDieStep(des, pos, ep);
                    int nxt = OpenMeshNeighbor(pos, d);
                    if (nxt < 0 || DieOfGlobal(nxt) != cur_die ||
                        man(nxt, ptile) >= man(pos, ptile))
                        return false;
                    pos = nxt;
                    if (++guard > TOTAL_CORES * 2)
                        return false;
                }
                egresses.push_back(CrossDieStep(des, pos, ep)); // 出口 tile：egress 方向
                int ingress = CrossDieIngressTile(cur_die, ep); // 跨 link 落点
                if (ingress < 0 || DieOfGlobal(ingress) == cur_die)
                    return false;
                pos = ingress;
                hops++;
                int nd = DieManhattan(DieOfGlobal(pos), DieOfGlobal(des));
                if (nd != prevd - 1) // die 距离严格 -1
                    return false;
                prevd = nd;
            }
            while (pos != des) { // 已到 des die：片内 XY 收敛到 des
                Directions d = CrossDieStep(des, pos, -1);
                int nxt = OpenMeshNeighbor(pos, d);
                if (nxt < 0 || DieOfGlobal(nxt) != DieOfGlobal(des))
                    return false;
                pos = nxt;
                if (++guard > TOTAL_CORES * 2)
                    return false;
            }
            return hops == (int)exp_egress.size() &&
                   hops == DieManhattan(DieOfGlobal(src), DieOfGlobal(des)) &&
                   egresses == exp_egress;
        };

        // 15b.1 3×1 直线两跳：die0->die2 (E,E) / die2->die0 (W,W)
        setTopo(4, 4, 3, 1);
        build({"E", "W"});
        bool v1ok = true;
        try {
            ValidateV1MvpTopology();
        } catch (...) {
            v1ok = false;
        }
        check(v1ok, "V2-a 3x1: single-port E/W topology accepted");
        check(DieManhattan(0, 2) == 2, "V2-a 3x1: die0->die2 hop count 2");
        check(walkMulti(0, 2 * CORES_PER_DIE, {EAST, EAST}),
              "V2-a 3x1 die0->die2: re-pin per die, 2 hops E,E, die-dist strictly -1");
        check(walkMulti(2 * CORES_PER_DIE, 0, {WEST, WEST}),
              "V2-a 3x1 die2->die0: reverse 2 hops W,W");

        // 15b.2 2×2 对角两跳：die0->die3 (E then N) / die3->die0 (W then S)——die 级 X-first XY
        setTopo(4, 4, 2, 2);
        build({"N", "S", "E", "W"});
        v1ok = true;
        try {
            ValidateV1MvpTopology();
        } catch (...) {
            v1ok = false;
        }
        check(v1ok, "V2-a 2x2: single-port N/S/E/W topology accepted");
        check(DieManhattan(0, 3) == 2, "V2-a 2x2: die0->die3 hop count 2");
        check(walkMulti(0, 3 * CORES_PER_DIE, {EAST, NORTH}),
              "V2-a 2x2 die0->die3: X-first XY, re-pin E then N, die-dist strictly -1");
        check(walkMulti(3 * CORES_PER_DIE, 0, {WEST, SOUTH}),
              "V2-a 2x2 die3->die0: reverse X-first, W then S");

        // 15b.3 防御（V2 关键设计点）：中间 die 若仍携**源 die 首跳** exit（不重新 pin），
        // CrossDieStep 拒绝——2×2 die0 首跳 E 出口在 die1 处应为 N，携旧 E 出口 → 抛错。
        {
            setTopo(4, 4, 2, 2);
            build({"N", "S", "E", "W"});
            int des3 = 3 * CORES_PER_DIE;
            int stale = CrossDieSelectExit(0, des3);      // die0 的 E 出口
            int ingress1 = CrossDieIngressTile(0, stale); // die1 入口 tile
            bool threw = false;
            try {
                CrossDieStep(des3, ingress1, stale); // die1 仍用旧 E 出口
            } catch (const std::runtime_error &) {
                threw = true;
            }
            check(threw, "V2-a stale exit: source-die first-hop carried into next die is "
                         "rejected (re-pin required)");
            int repin = CrossDieSelectExit(ingress1, des3); // die1 正确重新 pin
            bool okstep = false;
            try {
                CrossDieStep(des3, ingress1, repin);
                okstep = (g_die_ports.ports[repin].dir == NORTH);
            } catch (...) {
                okstep = false;
            }
            check(okstep, "V2-a re-pin at die1 yields NORTH exit, routes cleanly");
        }

        // 15b.4 生产包装接口 SelectCoreMsgExit（真正被 Send_prim / PinControlMsgExit 调用的那层）：
        //       多跳下不再抛错、与 CrossDieSelectExit 一致、只 pin 首跳、且能写进包头并 round-trip。
        {
            setTopo(4, 4, 2, 2);
            build({"N", "S", "E", "W"});
            int des3 = 3 * CORES_PER_DIE; // die0 -> die3，距离 2
            int wrapped = SelectCoreMsgExit(0, des3);
            check(wrapped >= 0 && wrapped == CrossDieSelectExit(0, des3),
                  "V2-a SelectCoreMsgExit(multi-hop) == CrossDieSelectExit (production wrapper)");
            check(g_die_ports.ports[wrapped].dir == EAST,
                  "V2-a SelectCoreMsgExit(die0->die3) pins the FIRST hop only (EAST)");
            check(SelectCoreMsgExit(0, 1) == -1,
                  "V2-a SelectCoreMsgExit same-die still unpinned (-1)");
            Msg mm; // 该值必须能真实进包头
            mm.exit_port_ = wrapped;
            check(DeserializeMsg(SerializeMsg(mm)).exit_port_ == wrapped,
                  "V2-a multi-hop pinned exit survives Msg.exit_port_ serialization");

            setTopo(4, 4, 3, 1); // 3×1 直线同样不再因距离 2 被拒
            build({"E", "W"});
            int w31 = SelectCoreMsgExit(0, 2 * CORES_PER_DIE);
            check(w31 >= 0 && w31 == CrossDieSelectExit(0, 2 * CORES_PER_DIE) &&
                      g_die_ports.ports[w31].dir == EAST,
                  "V2-a 3x1 SelectCoreMsgExit(die0->die2) accepts distance 2 (was V1 throw)");
            Msg m31;
            m31.exit_port_ = w31;
            check(DeserializeMsg(SerializeMsg(m31)).exit_port_ == w31,
                  "V2-a 3x1 multi-hop pinned exit survives serialization");
        }

        // 15b.5 CrossDieIngressTile 精确性：对每条 link 都 == remote_die/remote_port.tile；
        //       反向 link 回到原出口 tile；非法 port / 非法 die / 无 peer 边界端口 → -1。
        {
            setTopo(4, 4, 2, 2);
            build({"N", "S", "E", "W"});
            bool exact = true, rev = true;
            int checked = 0;
            for (const auto &l : g_d2d_links) {
                int want = l.remote_die * CORES_PER_DIE +
                           g_die_ports.ports[l.remote_port].tile;
                if (CrossDieIngressTile(l.local_die, l.local_port) != want)
                    exact = false;
                // 反向：从落点 die 用镜像端口回跨 → 原 die 的出口端口 tile
                if (CrossDieIngressTile(l.remote_die, l.remote_port) !=
                    l.local_die * CORES_PER_DIE +
                        g_die_ports.ports[l.local_port].tile)
                    rev = false;
                checked++;
            }
            check(exact && checked > 0,
                  "V2-a CrossDieIngressTile == g_d2d_links remote_die/remote_port.tile (every link)");
            check(rev,
                  "V2-a CrossDieIngressTile reverse link returns the original egress tile");
            int e_pid = -1;
            for (auto &p : g_die_ports.ports)
                if (p.role == ROLE_C2C && p.dir == EAST)
                    e_pid = p.port_id;
            check(CrossDieIngressTile(0, -1) == -1 &&
                      CrossDieIngressTile(0, 9999) == -1,
                  "V2-a CrossDieIngressTile: illegal port id -> -1");
            check(CrossDieIngressTile(999, e_pid) == -1 &&
                      CrossDieIngressTile(-1, e_pid) == -1,
                  "V2-a CrossDieIngressTile: illegal die -> -1");
            // 无 peer 的边界端口（3×1：die0 的 W、die2 的 E）
            setTopo(4, 4, 3, 1);
            build({"E", "W"});
            int w_pid = -1, e_pid2 = -1;
            for (auto &p : g_die_ports.ports) {
                if (p.role == ROLE_C2C && p.dir == WEST)
                    w_pid = p.port_id;
                if (p.role == ROLE_C2C && p.dir == EAST)
                    e_pid2 = p.port_id;
            }
            check(CrossDieIngressTile(0, w_pid) == -1,
                  "V2-a CrossDieIngressTile: boundary port without peer -> -1 (3x1 die0 W)");
            check(CrossDieIngressTile(2, e_pid2) == -1,
                  "V2-a CrossDieIngressTile: boundary port without peer -> -1 (3x1 die2 E)");
        }
    }

    // ---- 16. V1-b seam：IsC2CEgressEdge + V1 link_bw==1 契约 ----
    {
        // 16.1 单 die / 无 die_ports：全 false
        setTopo(4, 4, 1, 1);
        BuildHostAttach();
        HOST_LANES = g_host_attach.n_lanes;
        check(!IsC2CEgressEdge(0, EAST) && !IsC2CEgressEdge(0, WEST) &&
                  !IsC2CEgressEdge(3, EAST),
              "IsC2CEgressEdge single-die/no die_ports: all false");

        // 16.2 2×1 单 E/W：只有 die0.E(tile3)、die1.W(tile16) 为 true；边界/错向/非端口 tile false
        {
            setTopo(4, 4, 2, 1);
            D2DJson hw;
            hw["die_ports"]["edges"]["S"] = {{"role", "host"}};
            hw["die_ports"]["overrides"] = D2DJson::array();
            hw["die_ports"]["overrides"].push_back(
                {{"side", "E"}, {"idx", 0}, {"role", "c2c"}, {"dir", "E"}});
            hw["die_ports"]["overrides"].push_back(
                {{"side", "W"}, {"idx", 0}, {"role", "c2c"}, {"dir", "W"}});
            ParseDiePorts(hw);
            bool ok = IsC2CEgressEdge(3, EAST) &&   // die0 E 出口
                      IsC2CEgressEdge(16, WEST) &&  // die1 W 出口
                      !IsC2CEgressEdge(0, WEST) &&  // die0 W：边界（无西邻）
                      !IsC2CEgressEdge(19, EAST) && // die1 E：边界（无东邻）
                      !IsC2CEgressEdge(1, EAST) &&  // 非端口 tile
                      !IsC2CEgressEdge(3, WEST);    // 端口 tile 但错方向
            check(ok, "IsC2CEgressEdge 2x1 E/W: only die0.E & die1.W true");
        }

        // 16.3 1×2 单 N/S：只有 die0.N(tile12)、die1.S(tile16) 为 true
        {
            setTopo(4, 4, 1, 2);
            D2DJson hw;
            hw["die_ports"]["edges"]["W"] = {{"role", "host"}};
            hw["die_ports"]["overrides"] = D2DJson::array();
            hw["die_ports"]["overrides"].push_back(
                {{"side", "N"}, {"idx", 0}, {"role", "c2c"}, {"dir", "N"}});
            hw["die_ports"]["overrides"].push_back(
                {{"side", "S"}, {"idx", 0}, {"role", "c2c"}, {"dir", "S"}});
            ParseDiePorts(hw);
            bool ok = IsC2CEgressEdge(12, NORTH) &&  // die0 N 出口
                      IsC2CEgressEdge(16, SOUTH) &&  // die1 S 出口
                      !IsC2CEgressEdge(0, SOUTH) &&  // die0 S：边界
                      !IsC2CEgressEdge(28, NORTH);   // die1 N：边界
            check(ok, "IsC2CEgressEdge 1x2 N/S: only die0.N & die1.S true");
        }

        // 16.4 V1 契约：C2C link_bw != 1 被 ValidateV1MvpTopology 拒绝
        {
            setTopo(4, 4, 2, 1);
            D2DJson hw;
            hw["die_ports"]["edges"]["S"] = {{"role", "host"}};
            hw["die_ports"]["overrides"] = D2DJson::array();
            hw["die_ports"]["overrides"].push_back(
                {{"side", "E"}, {"idx", 0}, {"role", "c2c"}, {"dir", "E"}});
            hw["die_ports"]["overrides"].push_back(
                {{"side", "W"}, {"idx", 0}, {"role", "c2c"}, {"dir", "W"}});
            hw["die_ports"]["c2c"] = {{"link_bw", 2}, {"latency", 20},
                                      {"buffer_depth", 8}};
            ParseDiePorts(hw);
            bool rej = false;
            try {
                ValidateV1MvpTopology();
            } catch (const std::runtime_error &) {
                rej = true;
            }
            check(rej, "ValidateV1MvpTopology rejects C2C link_bw != 1 (V1: 1 pkt/cycle)");
        }

        // 16.5 pinned exit 编码上限：C2C port_id 超出 exit_port 字段容量 → 拒绝（不静默截断）
        {
            setTopo(4, 4, 2, 1);
            D2DJson hw;
            hw["die_ports"]["edges"]["S"] = {{"role", "host"}};
            hw["die_ports"]["overrides"] = D2DJson::array();
            hw["die_ports"]["overrides"].push_back(
                {{"side", "E"}, {"idx", 0}, {"role", "c2c"}, {"dir", "E"}});
            hw["die_ports"]["overrides"].push_back(
                {{"side", "W"}, {"idx", 0}, {"role", "c2c"}, {"dir", "W"}});
            ParseDiePorts(hw);
            // 人为把一个 C2C 端口的 port_id 顶到不可编码（真实大 die 才会自然发生）
            int bad = (1 << 16) - 1; // > max_encodable = 65534
            for (auto &p : g_die_ports.ports)
                if (p.role == ROLE_C2C) {
                    p.port_id = bad;
                    break;
                }
            bool rej = false;
            try {
                ValidateV1MvpTopology();
            } catch (const std::runtime_error &) {
                rej = true;
            }
            check(rej, "ValidateV1MvpTopology rejects C2C port_id beyond exit_port capacity");
        }
    }

    // ---- 16b. V3-a：D2D 建模模式 / 有理数速率 / 四类容量的启动期契约 ----
    // 本增量**不改生产数据路径**：旧配置（无 mode）必须恒定解析为 functional_v2 且行为不变；
    // 非法模式、非法速率（尤其 rate>1）、bounded 缺安全策略/缺容量，一律启动期明确拒绝。
    {
        setTopo(4, 4, 2, 1);
        // 构造一个合法 2×1 单端口底座，只替换 c2c 段来测各种组合
        auto mk = [&](const D2DJson &c2c) {
            D2DJson hw;
            hw["die_ports"]["edges"]["S"] = {{"role", "host"}};
            hw["die_ports"]["overrides"] = D2DJson::array();
            hw["die_ports"]["overrides"].push_back(
                {{"side", "E"}, {"idx", 0}, {"role", "c2c"}, {"dir", "E"}});
            hw["die_ports"]["overrides"].push_back(
                {{"side", "W"}, {"idx", 0}, {"role", "c2c"}, {"dir", "W"}});
            hw["die_ports"]["c2c"] = c2c;
            return hw;
        };
        D2DJson legacy = {{"link_bw", 1}, {"latency", 20}, {"buffer_depth", 8}};

        // (a) 旧配置：无 mode → functional_v2，安全策略 none，行为与 V2 一致
        ParseDiePorts(mk(legacy));
        check(g_d2d_cfg.mode == MODE_FUNCTIONAL_V2 &&
                  g_d2d_cfg.safety == SAFETY_NONE,
              "V3-a legacy c2c (no mode) -> functional_v2 / safety none");
        check(g_d2d_cfg.link_latency == 20,
              "V3-a legacy 'latency' still parsed as link_latency");
        bool v2ok = true;
        try {
            ValidateD2DTopology();
        } catch (...) {
            v2ok = false;
        }
        check(v2ok, "V3-a version-aware validate accepts legacy functional_v2");

        // (b) 未知 mode / 未知 safety → 拒绝
        D2DJson bad = legacy;
        bad["mode"] = "wormhole";
        check(throwsRuntime(mk(bad)), "V3-a unknown mode rejected");
        bad = D2DJson{{"mode", "bounded_saf"}, {"safety", "escape_vc"}};
        check(throwsRuntime(mk(bad)), "V3-a unknown safety policy rejected");

        // (c) bounded_saf 缺安全策略 → 拒绝（有限 buffer 必须有明确死锁安全论证）
        bad = D2DJson{{"mode", "bounded_saf"},
                      {"link_latency", 20},
                      {"saf_buffer_depth", 64},
                      {"link_inflight_depth", 10},
                      {"rx_buffer_depth", 8},
                      {"ctrl_buffer_depth", 4}};
        check(throwsRuntime(mk(bad)),
              "V3-a bounded_saf without safety policy rejected");

        // (d) bounded_saf 缺任一类容量 → 拒绝（四类容量分工不同，不可混代）
        auto bounded = [&](int saf, int inflight, int rx, int ctrl) {
            D2DJson c = {{"mode", "bounded_saf"},
                         {"safety", "whole_flow_saf"},
                         {"port_rate", {{"num", 1}, {"den", 2}}},
                         {"link_rate", {{"num", 1}, {"den", 4}}},
                         {"link_latency", 20}};
            if (saf)
                c["saf_buffer_depth"] = saf;
            if (inflight)
                c["link_inflight_depth"] = inflight;
            if (rx)
                c["rx_buffer_depth"] = rx;
            if (ctrl)
                c["ctrl_buffer_depth"] = ctrl;
            return c;
        };
        check(throwsRuntime(mk(bounded(0, 10, 8, 4))),
              "V3-a bounded_saf without saf_buffer_depth rejected");
        check(throwsRuntime(mk(bounded(64, 0, 8, 4))),
              "V3-a bounded_saf without link_inflight_depth rejected");
        check(throwsRuntime(mk(bounded(64, 10, 0, 4))),
              "V3-a bounded_saf without rx_buffer_depth rejected");
        check(throwsRuntime(mk(bounded(64, 10, 8, 0))),
              "V3-a bounded_saf without ctrl_buffer_depth rejected");

        // (e) 完整 bounded_saf 配置被接受，且四类容量与有理数速率被如实解析
        ParseDiePorts(mk(bounded(64, 10, 8, 4)));
        check(g_d2d_cfg.mode == MODE_BOUNDED_SAF &&
                  g_d2d_cfg.safety == SAFETY_WHOLE_FLOW_SAF,
              "V3-a bounded_saf + whole_flow_saf accepted");
        check(g_d2d_cfg.saf_buffer_depth == 64 &&
                  g_d2d_cfg.link_inflight_depth == 10 &&
                  g_d2d_cfg.rx_buffer_depth == 8 &&
                  g_d2d_cfg.ctrl_buffer_depth == 4,
              "V3-a four capacities parsed independently (no single buffer_depth)");
        check(g_d2d_cfg.port_rate.num == 1 && g_d2d_cfg.port_rate.den == 2 &&
                  g_d2d_cfg.link_rate.num == 1 && g_d2d_cfg.link_rate.den == 4,
              "V3-a rational rates parsed exactly (1/2, 1/4; no float)");
        bool bok = true;
        try {
            ValidateD2DTopology();
        } catch (...) {
            bok = false;
        }
        check(bok, "V3-a version-aware validate accepts complete bounded_saf");

        // (f) rate>1 必须**明确拒绝**——单信道每拍只能载 1 包，不得静默按 1 建模
        D2DJson r2 = bounded(64, 10, 8, 4);
        r2["link_rate"] = {{"num", 4}, {"den", 1}};
        check(throwsRuntime(mk(r2)),
              "V3-a link_rate > 1 packet/cycle rejected (not silently clamped)");
        r2 = bounded(64, 10, 8, 4);
        r2["port_rate"] = 2; // 整数简写 2 == 2/1 > 1
        check(throwsRuntime(mk(r2)), "V3-a port_rate > 1 rejected (integer form)");
        r2 = bounded(64, 10, 8, 4);
        r2["link_rate"] = {{"num", 1}, {"den", 0}};
        check(throwsRuntime(mk(r2)), "V3-a rate with den=0 rejected");
        r2 = bounded(64, 10, 8, 4);
        r2["link_rate"] = {{"num", -1}, {"den", 4}};
        check(throwsRuntime(mk(r2)), "V3-a negative rate rejected");
        r2 = bounded(64, 10, 8, 4);
        r2["saf_buffer_depth"] = 0;
        check(throwsRuntime(mk(r2)), "V3-a zero depth rejected");

        // (g) functional_v2 下出现 V3 专属字段 → 拒绝（避免配置意图与模式不符被静默忽略）
        D2DJson mix = legacy;
        mix["saf_buffer_depth"] = 64;
        check(throwsRuntime(mk(mix)),
              "V3-a V3-only capacity under functional_v2 rejected (mode mismatch)");
        mix = legacy;
        mix["safety"] = "whole_flow_saf";
        check(throwsRuntime(mk(mix)),
              "V3-a safety policy without bounded_saf rejected");
        // 速率/延迟字段同样是 bounded-only：functional_v2 下必须拒绝而非接受后忽略
        mix = legacy;
        mix["link_rate"] = {{"num", 1}, {"den", 4}};
        check(throwsRuntime(mk(mix)),
              "V3-a link_rate under functional_v2 rejected (would be ignored)");
        mix = legacy;
        mix["port_rate"] = {{"num", 1}, {"den", 2}};
        check(throwsRuntime(mk(mix)),
              "V3-a port_rate under functional_v2 rejected (would be ignored)");
        mix = legacy;
        mix["link_latency"] = 7;
        check(throwsRuntime(mk(mix)),
              "V3-a link_latency under functional_v2 rejected (conflicts with 'latency')");
        // 反向：bounded_saf 下 legacy 字段必须拒绝，杜绝 latency=20 与 link_latency=7 并存
        D2DJson conflict = bounded(64, 10, 8, 4);
        conflict["latency"] = 20;
        check(throwsRuntime(mk(conflict)),
              "V3-a legacy 'latency' under bounded_saf rejected (no silent override)");
        conflict = bounded(64, 10, 8, 4);
        conflict["link_bw"] = 4;
        check(throwsRuntime(mk(conflict)),
              "V3-a legacy 'link_bw' under bounded_saf rejected (contradicts link_rate)");

        // (h) link_inflight_depth 必须 >= BDP = ceil(2*L*rate)；SAF depth>=F 留待 V3-c
        //     L=20, rate=1/4 -> BDP = ceil(40/4) = 10
        D2DJson bdp = bounded(64, 10, 8, 4);
        check(!throwsRuntime(mk(bdp)), "V3-a link_inflight_depth == BDP (10) accepted");
        bdp = bounded(64, 9, 8, 4);
        check(throwsRuntime(mk(bdp)),
              "V3-a link_inflight_depth < BDP rejected (link could not stay full)");
        // latency=0 -> BDP=0，任何 depth>=1 均可
        D2DJson zero = bounded(64, 1, 8, 4);
        zero["link_latency"] = 0;
        check(!throwsRuntime(mk(zero)), "V3-a latency=0 -> BDP=0, depth 1 accepted");

        // (i) 无 die_ports 时必须复位 g_d2d_cfg（防止重复解析残留上次 bounded 配置）
        ParseDiePorts(mk(bounded(64, 10, 8, 4)));
        D2DJson none;
        ParseDiePorts(none);
        check(g_d2d_cfg.mode == MODE_FUNCTIONAL_V2 &&
                  g_d2d_cfg.saf_buffer_depth == 0 &&
                  g_d2d_cfg.safety == SAFETY_NONE,
              "V3-a re-parse without die_ports resets g_d2d_cfg (no stale bounded state)");

        // 恢复默认，避免影响后续段落
        ParseDiePorts(mk(legacy));
    }

    // ---- 17. FlowKey (source,tag,subflow) 三元组语义 ----
    // 注：output_lock **不用** FlowKey（tag 已=全局接收槽，多发一需 tag-only 聚合，见 flow.h）。
    // FlowKey 预留给 V5 subflow striping：同 (source,tag) 拆多条 subflow 时用三元组区分。
    {
        FlowKey a{5, 3, 0}, b{21, 3, 0}, c{5, 3, 0}, d{5, 7, 0}, e{5, 3, 1};
        // 同 tag 不同 source → 不同 key（跨 die 同 local-id/同 tag 不别名）
        check(a != b && !(a == b), "FlowKey: same tag, different source -> distinct");
        check(a == c, "FlowKey: same (source,tag,subflow) -> equal");
        check(a != d, "FlowKey: different tag -> distinct");
        check(a != e, "FlowKey: different subflow -> distinct");
        // 严格弱序（可作 map key）：反对称
        check((a < b) != (b < a) && !(a < c) && !(c < a),
              "FlowKey: strict weak ordering (usable as map key)");
    }

    // ---- 18. preflight 能力 + 生产 gate（V1-c1a）：helper 可验，c3 前不放行 ----
    {
        auto mkwl = [&](int cid, int dest) {
            D2DJson wl;
            wl["id_space"] = "global";
            D2DJson cast;
            cast["dest"] = dest;
            D2DJson job;
            job["cast"] = D2DJson::array();
            job["cast"].push_back(cast);
            D2DJson core;
            core["id"] = cid;
            core["worklist"] = D2DJson::array();
            core["worklist"].push_back(job);
            D2DJson chip;
            chip["cores"] = D2DJson::array();
            chip["cores"].push_back(core);
            wl["chips"] = D2DJson::array();
            wl["chips"].push_back(chip);
            return wl;
        };
        auto wsErr = [&](const D2DJson &wl,
                         bool allow_adjacent = false) -> std::string {
            try {
                ValidateWorkloadStructure(wl, 0, allow_adjacent);
            } catch (const std::runtime_error &e) {
                return e.what();
            }
            return "";
        };

        // (a) 无 die_ports（无 C2C link）：相邻 die 仍拒绝，报 "requires D2D Link"
        setTopo(4, 4, 2, 1); // setTopo 清空 g_die_ports
        {
            std::string e = wsErr(mkwl(0, CORES_PER_DIE)); // die0 -> die1
            check(e.find("requires D2D Link") != std::string::npos,
                  "preflight: adjacent cross-die but no C2C link -> rejected");
        }
        // (b) 相邻 die + 单端口 E/W link：helper 允许，但生产默认 gate 仍拒绝
        {
            setTopo(4, 4, 2, 1);
            D2DJson hw;
            hw["die_ports"]["edges"]["S"] = {{"role", "host"}};
            hw["die_ports"]["overrides"] = D2DJson::array();
            hw["die_ports"]["overrides"].push_back(
                {{"side", "E"}, {"idx", 0}, {"role", "c2c"}, {"dir", "E"}});
            hw["die_ports"]["overrides"].push_back(
                {{"side", "W"}, {"idx", 0}, {"role", "c2c"}, {"dir", "W"}});
            ParseDiePorts(hw);
            check(HasD2DLink(0, 1) && HasD2DLink(1, 0),
                  "preflight: exact bidirectional die0<->die1 links exist");
            check(wsErr(mkwl(0, CORES_PER_DIE), true).empty(),
                  "preflight helper: adjacent cross-die with peer link -> valid");
            check(wsErr(mkwl(0, CORES_PER_DIE)).find("disabled for this validation path") !=
                      std::string::npos,
                  "preflight default-closed path: adjacent workload requires explicit opt-in");
        }
        // (c) 多跳（3×1，die0 -> die2，距离 2）：**V2-b 起放行**——沿 die 级维序路径逐跳校验，
        //     每跳都有双向 peer link 才允许；任一跳缺 link 仍必须拒绝（护栏没被删掉）。
        {
            setTopo(4, 4, 3, 1);
            D2DJson hw;
            hw["die_ports"]["edges"]["S"] = {{"role", "host"}};
            hw["die_ports"]["overrides"] = D2DJson::array();
            hw["die_ports"]["overrides"].push_back(
                {{"side", "E"}, {"idx", 0}, {"role", "c2c"}, {"dir", "E"}});
            hw["die_ports"]["overrides"].push_back(
                {{"side", "W"}, {"idx", 0}, {"role", "c2c"}, {"dir", "W"}});
            ParseDiePorts(hw);
            check(wsErr(mkwl(0, 2 * CORES_PER_DIE), true).empty(),
                  "preflight V2-b: multi-hop die0->die2 accepted (every hop has a peer link)");
            // 负例：抽掉中间一跳的 link（die1->die2 方向），多跳必须被拒
            std::vector<D2DLink> saved = g_d2d_links;
            g_d2d_links.erase(
                std::remove_if(g_d2d_links.begin(), g_d2d_links.end(),
                               [](const D2DLink &l) {
                                   return l.local_die == 1 && l.remote_die == 2;
                               }),
                g_d2d_links.end());
            std::string e = wsErr(mkwl(0, 2 * CORES_PER_DIE), true);
            check(e.find("no bidirectional peer link on hop") != std::string::npos,
                  "preflight V2-b: multi-hop with a broken intermediate hop -> rejected");
            g_d2d_links = saved;
        }

        // (d) tag -> dest 唯一性（output_lock 按接收槽=tag）：同 tag 指向不同接收核 → 拒绝；
        //     同 tag 同接收核（多发一）/ 不同 tag → 允许。
        setTopo(4, 4, 1, 1);
        {
            auto mk2 = [&](int t0, int d0, int t1, int d1) {
                auto mkcore = [&](int id, int dest, int tag) {
                    D2DJson cast;
                    cast["dest"] = dest;
                    cast["tag"] = tag;
                    D2DJson job;
                    job["cast"] = D2DJson::array();
                    job["cast"].push_back(cast);
                    D2DJson core;
                    core["id"] = id;
                    core["worklist"] = D2DJson::array();
                    core["worklist"].push_back(job);
                    return core;
                };
                D2DJson wl;
                wl["id_space"] = "global";
                D2DJson chip;
                chip["cores"] = D2DJson::array();
                chip["cores"].push_back(mkcore(0, d0, t0));
                chip["cores"].push_back(mkcore(1, d1, t1));
                wl["chips"] = D2DJson::array();
                wl["chips"].push_back(chip);
                return wl;
            };
            check(wsErr(mk2(5, 10, 5, 11)).find("multiple receiver cores") !=
                      std::string::npos,
                  "tag->dest: same tag to different receivers rejected");
            check(wsErr(mk2(5, 10, 5, 10)).empty(),
                  "tag->dest: same tag to same receiver (many-to-one) allowed");
            check(wsErr(mk2(5, 10, 6, 11)).empty(),
                  "tag->dest: distinct tags allowed");
        }
    }

    // ---- 19. REQUEST/ACK 控制路由 + 源端 pinned exit（V1-c1b；生产 workload 仍 gated）----
    {
        setTopo(4, 4, 2, 1);
        D2DJson hw;
        hw["die_ports"]["edges"]["S"] = {{"role", "host"}};
        hw["die_ports"]["overrides"] = D2DJson::array();
        hw["die_ports"]["overrides"].push_back(
            {{"side", "E"}, {"idx", 0}, {"role", "c2c"}, {"dir", "E"}});
        hw["die_ports"]["overrides"].push_back(
            {{"side", "W"}, {"idx", 0}, {"role", "c2c"}, {"dir", "W"}});
        ParseDiePorts(hw);

        auto walkMessage = [&](const Msg &msg, bool control) {
            int pos = msg.source_;
            int crossings = 0;
            std::set<int> visited;
            for (int step = 0; step < TOTAL_CORES + 8; step++) {
                if (pos == msg.des_)
                    return crossings == 1;
                if (!visited.insert(pos).second)
                    return false;
                Directions d = control ? ControlMsgNextHop(msg, pos)
                                       : DataMsgNextHop(msg, pos);
                int nb = OpenMeshNeighbor(pos, d);
                if (nb >= 0) {
                    pos = nb;
                    continue;
                }

                // 开边且路由选择朝外：必须命中消息携带的 source-die 有向 link，
                // 再从其 remote_port 对应 tile 进入目标 die。
                const D2DLink *link = nullptr;
                for (const auto &l : g_d2d_links)
                    if (l.local_die == DieOfGlobal(pos) &&
                        l.local_port == msg.exit_port_) {
                        link = &l;
                        break;
                    }
                if (!link)
                    return false;
                const D2DPort &lp = g_die_ports.ports[link->local_port];
                if (lp.tile != LocalOfGlobal(pos) || lp.side != d ||
                    lp.dir != d)
                    return false;

                const D2DPort &rp = g_die_ports.ports[link->remote_port];
                pos = GlobalId(link->remote_die, rp.tile);
                crossings++;
            }
            return false;
        };

        Msg req(MSG_TYPE::REQUEST, CORES_PER_DIE, 7, 0);
        PinControlMsgExit(req);
        int req_exit = CrossDieSelectExit(req.source_, req.des_);
        check(req.exit_port_ == req_exit && req.exit_port_ >= 0,
              "c1b REQUEST: source selects and pins one C2C exit");
        check(DeserializeMsg(SerializeMsg(req)).exit_port_ == req.exit_port_,
              "c1b REQUEST: pinned exit survives packet serialization");
        check(walkMessage(req, true),
              "c1b REQUEST: fixed exit -> D2D crossing -> destination core");

        Msg ack(MSG_TYPE::ACK, 0, 7, CORES_PER_DIE);
        PinControlMsgExit(ack);
        check(ack.exit_port_ == CrossDieSelectExit(ack.source_, ack.des_) &&
                  ack.exit_port_ != req.exit_port_,
              "c1b ACK: destination core independently pins reverse exit");
        check(walkMessage(ack, true),
              "c1b ACK: reverse D2D crossing returns to request source");

        Msg local(MSG_TYPE::REQUEST, 1, 9, 0);
        PinControlMsgExit(local);
        check(local.exit_port_ == -1 &&
                  ControlMsgNextHop(local, 0) == GetNextHop(local.des_, 0),
              "c1b same-die control: unpinned and legacy XY-equivalent");

        Msg missing_pin(MSG_TYPE::REQUEST, CORES_PER_DIE, 7, 0);
        bool missing_threw = false;
        try {
            ControlMsgNextHop(missing_pin, 0);
        } catch (const std::runtime_error &) {
            missing_threw = true;
        }
        check(missing_threw,
              "c1b cross-die control: missing pinned exit rejected clearly");
        // ---- 20. DATA 流级 pin + Router 数据 dispatch（V1-c2）----
        int data_exit = SelectCoreMsgExit(0, CORES_PER_DIE);
        std::vector<Msg> packets;
        for (int seq = 1; seq <= 3; seq++) {
            Msg p(seq == 3, MSG_TYPE::DATA, seq, CORES_PER_DIE, 0, 7,
                  M_D_DATA, sc_bv<128>(seq));
            p.source_ = 0;
            p.exit_port_ = data_exit; // 生产路径从 Send_prim 的一次性选择复制
            packets.push_back(p);
        }
        bool same_pin = data_exit >= 0;
        bool all_walk = true;
        for (const auto &p : packets) {
            Msg wire = DeserializeMsg(SerializeMsg(p));
            same_pin = same_pin && wire.exit_port_ == data_exit;
            all_walk = all_walk && walkMessage(wire, false);
        }
        check(same_pin,
              "c2 DATA: every packet carries the same serialized flow-level pin");
        check(all_walk,
              "c2 DATA: 3-packet flow uses fixed exit and crosses expected D2D link");

        Msg reverse(true, MSG_TYPE::DATA, 1, 0, 0, 7, M_D_DATA,
                    sc_bv<128>(1));
        reverse.source_ = CORES_PER_DIE;
        reverse.exit_port_ = SelectCoreMsgExit(reverse.source_, reverse.des_);
        check(reverse.exit_port_ != data_exit && walkMessage(reverse, false),
              "c2 DATA: reverse direction independently pins and crosses reverse link");

        Msg local_data(true, MSG_TYPE::DATA, 1, 1, 0, 9, M_D_DATA,
                       sc_bv<128>(1));
        local_data.source_ = 0;
        local_data.exit_port_ = SelectCoreMsgExit(0, 1);
        check(local_data.exit_port_ == -1 &&
                  DataMsgNextHop(local_data, 0) == GetNextHop(1, 0),
              "c2 same-die DATA: unpinned and legacy XY-equivalent");

        Msg missing_data(true, MSG_TYPE::DATA, 1, CORES_PER_DIE, 0, 7,
                         M_D_DATA, sc_bv<128>(1));
        missing_data.source_ = 0;
        bool missing_data_threw = false;
        try {
            DataMsgNextHop(missing_data, 0);
        } catch (const std::runtime_error &) {
            missing_data_threw = true;
        }
        check(missing_data_threw,
              "c2 cross-die DATA: missing pinned exit rejected clearly");

        Msg config(false, MSG_TYPE::CONFIG, 1, CORES_PER_DIE, 0, 0,
                   M_D_DATA, sc_bv<128>(1));
        config.source_ = 0;
        config.exit_port_ = data_exit;
        bool config_threw = false;
        try {
            DataMsgNextHop(config, 0);
        } catch (const std::runtime_error &) {
            config_threw = true;
        }
        check(config_threw,
              "c2 data channel: non-DATA message cannot borrow cross-die route");
    }

    std::cout << "==== D2D V0 self-test: " << (g_total - g_fail) << "/" << g_total
              << " passed" << (g_fail ? "  <<< FAILURES" : "") << " ====" << std::endl;
    return g_fail;
}
