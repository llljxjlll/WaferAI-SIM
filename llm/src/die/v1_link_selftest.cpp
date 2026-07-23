// 独立 SystemC link 测试：用 testbench 驱动真实包穿过 D2DLinkUnit。
// V1-b2：latency 语义、FIFO 序、无丢/重、out_avail 停顿、drain 后 in==out、data/ctrl 独立、idle=0。
// V3-b standalone：单级有限 FIFO + pulse-credit RTT + token bucket，验证 BDP-1/BDP 边界。
// V3-d production：直接驱动 whole-flow SAF→inflight→RX、toggle credit 与真实 reservation release，
// 验证公式深度充分性、背压链、按序无丢重和结束排空。两条实现路径均在本 self-test 中独立覆盖。
#include "die/d2d_link.h"
#include "die/port.h"
#include "defs/spec.h"
#include "systemc.h"
#include "utils/msg_utils.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

namespace {

// 单条 link 的探针：驱动 in（data/ctrl 各自脚本）+ out_avail（可设停顿窗），记录 out。
struct LinkProbe : sc_module {
    D2DLinkUnit *link;
    sc_signal<sc_bv<256>> in_ch, out_ch, in_cch, out_cch;
    sc_signal<bool> in_s, in_av, out_s, out_av, in_cs, in_cav, out_cs, out_cav;
    sc_signal<bool> data_cret, ctrl_cret; // V3-b2：link → 上游 的信用回还

    std::vector<std::pair<int, int>> data_in, ctrl_in; // (cycle, id)
    int avail_lo = -1, avail_hi = -1;                  // out_avail=false 窗口 [lo,hi)
    int run_cycles = 60;
    std::vector<std::pair<int, int>> data_out, ctrl_out; // (cycle, id)

    // V3-b2：burst 注入（>0 时启用）。flow control = **信用式**（bounded link 的唯一流控真源）。
    // 信用 = 下游空位：初始 = depth，发一包 -1，**观测到 link 的 data_credit_return pulse +1**
    // （真实模块接口，非窥探 out_sent）。同时读真实 in_avail 记录，仅作背压信号诊断（非流控真源）。
    int inject_burst = 0;
    int next_id = 500; // burst 首包 id
    int sent_count = 0;
    int credit = 1 << 30;  // functional 视作无限；bounded 构造函数置为 data_depth
    int ready_seen = 0;    // 有信用可发 且 观测到 in_avail=true 的拍数
    int inconsistent = 0;  // 有信用可发但诊断镜像仍 false 的 settle 边沿次数（仅观测，不要求为 0）
    int notready_seen = 0; // 观测到 in_avail=false 的拍数（背压确实发生）
    int ctrl_burst = 0;    // 控制通道 burst（同信用式，信用 = ctrl_depth）
    int next_cid = 700;
    int csent_count = 0;
    int ccredit = 1 << 30;

    SC_HAS_PROCESS(LinkProbe);
    LinkProbe(sc_module_name n, int latency, D2DLinkBound bound = D2DLinkBound{})
        : sc_module(n) {
        if (bound.enabled) {
            credit = bound.data_depth;
            ccredit = bound.ctrl_depth;
        }
        link = new D2DLinkUnit("link", latency, -1, bound);
        link->in_channel(in_ch);
        link->in_sent(in_s);
        link->in_avail(in_av);
        link->in_ctrl_channel(in_cch);
        link->in_ctrl_sent(in_cs);
        link->in_ctrl_avail(in_cav);
        link->out_channel(out_ch);
        link->out_sent(out_s);
        link->out_avail(out_av);
        link->out_ctrl_channel(out_cch);
        link->out_ctrl_sent(out_cs);
        link->out_ctrl_avail(out_cav);
        link->data_credit_return(data_cret);
        link->ctrl_credit_return(ctrl_cret);
        SC_THREAD(drive);
    }

    void drive() {
        for (int c = 0; c <= run_cycles; c++) {
            // 记录上一拍已 settle 的输出（仅记录，不据此归还信用）
            if (out_s.read())
                data_out.push_back({c, (int)out_ch.read().to_uint()});
            if (out_cs.read())
                ctrl_out.push_back({c, (int)out_cch.read().to_uint()});
            // **信用回还来自 link 的真实接口信号**（非窥探 out_sent）：观测到 pulse → credit++
            if (data_cret.read())
                credit++;
            if (ctrl_cret.read())
                ccredit++;
            // 下游 ready（停顿窗内 false）
            bool av = !(c >= avail_lo && c < avail_hi);
            out_av.write(av);
            out_cav.write(av);
            // 驱动 DATA 输入
            bool ds = false;
            int did = 0;
            if (inject_burst > 0) {
                // 信用式发送（无损）；读真实 in_avail 仅作背压诊断（inconsistent 记 settle 滞后，不断言）。
                bool avail = in_av.read();
                if (!avail)
                    notready_seen++;
                if (sent_count < inject_burst && credit > 0) {
                    if (avail)
                        ready_seen++;
                    else
                        inconsistent++;
                    ds = true;
                    did = next_id + sent_count;
                    sent_count++;
                    credit--;
                }
            } else {
                for (auto &p : data_in)
                    if (p.first == c) {
                        ds = true;
                        did = p.second;
                    }
            }
            in_s.write(ds);
            if (ds)
                in_ch.write((sc_bv<256>)did);
            // 驱动 CTRL 输入（burst 仅依赖独立 ctrl credit；in_ctrl_avail 同样只是诊断镜像）
            bool cs = false;
            int cid = 0;
            if (ctrl_burst > 0) {
                if (csent_count < ctrl_burst && ccredit > 0) {
                    cs = true;
                    cid = next_cid + csent_count;
                    csent_count++;
                    ccredit--;
                }
            } else {
                for (auto &p : ctrl_in)
                    if (p.first == c) {
                        cs = true;
                        cid = p.second;
                    }
            }
            in_cs.write(cs);
            if (cs)
                in_cch.write((sc_bv<256>)cid);
            wait(CYCLE, SC_NS);
        }
    }
};

// 直接驱动 V3-d 生产 whole-flow SAF 分支。与 LinkProbe 的 standalone bounded
// pulse-credit 模型不同，本探针发送真实 REQUEST/DATA，并按生产 Router 相同的
// toggle-edge 语义接收 credit。
struct ProductionSafProbe : sc_module {
    D2DLinkUnit *link;
    sc_signal<sc_bv<256>> in_ch, out_ch, in_cch, out_cch;
    sc_signal<bool> in_s, in_av, out_s, out_av, in_cs, in_cav, out_cs, out_cav;
    sc_signal<bool> data_cret, ctrl_cret;

    int source, dest, tag, flow_packets;
    int run_cycles = 400;
    int data_stall_lo = -1, data_stall_hi = -1;
    int data_credit, ctrl_credit;
    bool data_credit_seen = false, ctrl_credit_seen = false;
    bool request_sent = false;
    int sent = 0, data_returns = 0, ctrl_returns = 0;
    int tail_sent_cycle = -1, first_out_cycle = -1, last_out_cycle = -1;
    int notready_seen = 0;
    std::vector<int> out_seq;
    std::vector<MSG_TYPE> out_ctrl_types;

    SC_HAS_PROCESS(ProductionSafProbe);
    ProductionSafProbe(sc_module_name n, int latency, int link_idx,
                       const D2DLinkBound &bound, int source_, int dest_,
                       int tag_, int flow_packets_)
        : sc_module(n), source(source_), dest(dest_), tag(tag_),
          flow_packets(flow_packets_), data_credit(bound.saf_depth),
          ctrl_credit(bound.ctrl_depth) {
        link = new D2DLinkUnit("link", latency, link_idx, bound);
        link->in_channel(in_ch);
        link->in_sent(in_s);
        link->in_avail(in_av);
        link->in_ctrl_channel(in_cch);
        link->in_ctrl_sent(in_cs);
        link->in_ctrl_avail(in_cav);
        link->out_channel(out_ch);
        link->out_sent(out_s);
        link->out_avail(out_av);
        link->out_ctrl_channel(out_cch);
        link->out_ctrl_sent(out_cs);
        link->out_ctrl_avail(out_cav);
        link->data_credit_return(data_cret);
        link->ctrl_credit_return(ctrl_cret);
        SC_THREAD(drive);
    }

    void drive() {
        for (int c = 0; c <= run_cycles; ++c) {
            if (out_s.read()) {
                Msg m = DeserializeMsg(out_ch.read());
                if (first_out_cycle < 0)
                    first_out_cycle = c;
                last_out_cycle = c;
                out_seq.push_back(m.seq_id_);
            }
            if (out_cs.read())
                out_ctrl_types.push_back(DeserializeMsg(out_cch.read()).msg_type_);

            bool data_event = data_cret.read();
            if (data_event != data_credit_seen) {
                data_credit_seen = data_event;
                data_credit++;
                data_returns++;
            }
            bool ctrl_event = ctrl_cret.read();
            if (ctrl_event != ctrl_credit_seen) {
                ctrl_credit_seen = ctrl_event;
                ctrl_credit++;
                ctrl_returns++;
            }

            bool data_ready = !(c >= data_stall_lo && c < data_stall_hi);
            out_av.write(data_ready);
            out_cav.write(true);

            bool send_ctrl = false;
            if (!request_sent && c >= 2 && ctrl_credit > 0) {
                Msg req(REQUEST, dest, tag, source);
                req.flow_packets_ = flow_packets;
                in_cch.write(SerializeMsg(req));
                send_ctrl = true;
                request_sent = true;
                ctrl_credit--;
            }
            in_cs.write(send_ctrl);

            bool send_data = false;
            if (request_sent && c >= 4 && sent < flow_packets && data_credit > 0) {
                int seq = sent + 1;
                bool tail = seq == flow_packets;
                Msg data(tail, DATA, seq, dest, 0, tag, 128,
                         sc_bv<128>((unsigned long long)seq));
                data.source_ = source;
                in_ch.write(SerializeMsg(data));
                send_data = true;
                sent++;
                data_credit--;
                if (tail)
                    tail_sent_cycle = c;
            }
            in_s.write(send_data);
            if (!in_av.read())
                notready_seen++;
            wait(CYCLE, SC_NS);
        }
    }
};

int g_fail = 0, g_total = 0;
void check(bool ok, const std::string &name) {
    g_total++;
    if (!ok) {
        g_fail++;
        std::cout << "  [FAIL] " << name << std::endl;
    } else {
        std::cout << "  [ ok ] " << name << std::endl;
    }
}

// 输出 id 序列 == 输入 id 序列（FIFO 序 + 无丢/重）
bool ids_match(const std::vector<std::pair<int, int>> &out,
               const std::vector<std::pair<int, int>> &in) {
    if (out.size() != in.size())
        return false;
    for (size_t i = 0; i < in.size(); i++)
        if (out[i].second != in[i].second)
            return false;
    return true;
}
// 每包 out_cycle - in_cycle 是否全相等，返回该常量（-1=不一致/空）
int const_delta(const std::vector<std::pair<int, int>> &out,
                const std::vector<std::pair<int, int>> &in) {
    if (out.empty() || out.size() != in.size())
        return -1;
    int d = out[0].first - in[0].first;
    for (size_t i = 1; i < in.size(); i++)
        if (out[i].first - in[i].first != d)
            return -1;
    return d;
}

} // namespace

int RunD2DLinkSelfTest() {
    g_fail = 0;
    g_total = 0;
    std::cout << "==== D2D V1 link self-test ====" << std::endl;
    ResetD2DLinkStats();

    // 干净序列（含连续 2 包 + 空洞）；4 个 latency 各一个探针，同一脚本便于比对 latency
    std::vector<std::pair<int, int>> dscript = {
        {2, 101}, {3, 102}, {6, 103}, {11, 104}}; // 连续 2、3；空洞
    std::vector<std::pair<int, int>> cscript = {
        {2, 201}, {5, 202}, {6, 203}}; // 与 data 不同时刻 → 验证独立

    std::vector<int> lats = {0, 1, 7, 20};
    std::vector<LinkProbe *> probes;
    for (int L : lats) {
        auto *p = new LinkProbe(
            ("probe_L" + std::to_string(L)).c_str(), L);
        p->data_in = dscript;
        p->ctrl_in = cscript;
        probes.push_back(p);
    }
    // 停顿探针（latency=1，out_avail 在 [1,20) 长窗为 false）：包在 [2,3] 捕获后成熟，
    // 但下游一直不 ready，必须在 Link 队首等待，直到窗口结束（20）才交付——验证不丢不重、
    // 且交付被推迟到窗口之后（窗口起于交付点之前，覆盖 delta 偏移）。
    auto *stall = new LinkProbe("probe_stall", 1);
    stall->data_in = {{2, 301}, {3, 302}};
    stall->avail_lo = 1;
    stall->avail_hi = 20;
    // idle 探针（无输入）
    auto *idle = new LinkProbe("probe_idle", 3);

    // ---- V3-b：有限缓冲 + token bucket（真实 valid/ready 握手驱动）----
    auto mkbound = [](int depth, int cdepth, int num, int den) {
        D2DLinkBound b;
        b.enabled = true;
        b.data_depth = depth;
        b.ctrl_depth = cdepth;
        b.rate.num = num;
        b.rate.den = den;
        return b;
    };
    // b_rate: depth=2, rate=1/4, lat=1 —— 速率限制（gap 4）+ 浅 FIFO 背压
    auto *b_rate = new LinkProbe("b_rate", 1, mkbound(2, 4, 1, 4));
    b_rate->inject_burst = 10;
    b_rate->run_cycles = 160;
    // b_ser: depth=1, rate=1/2, lat=0（低于 BDP）—— 信用往返 + 速率串行化，占用恒 <=1
    auto *b_ser = new LinkProbe("b_ser", 0, mkbound(1, 4, 1, 2));
    b_ser->inject_burst = 6;
    b_ser->run_cycles = 160;
    // b_ds: depth=8, rate=1, 下游 [0,40) 全程 stall —— 成熟包驻留、downstream_stall>0
    auto *b_ds = new LinkProbe("b_ds", 1, mkbound(8, 4, 1, 1));
    b_ds->inject_burst = 5;
    b_ds->run_cycles = 160;
    b_ds->avail_lo = 0;
    b_ds->avail_hi = 40;
    // 有理数 2/3：token 守恒——长期 goodput≈2/3，gap 只由 1、2 交替构成（非单一众数）
    auto *b_23 = new LinkProbe("b_23", 1, mkbound(8, 4, 2, 3));
    b_23->inject_burst = 16;
    b_23->run_cycles = 160;
    // bounded 控制 FIFO：ctrl_depth=1，下游控制 [0,30) 长 stall —— 控制包驻留、in_ctrl_avail 拉低、
    // 占用 <=1、释放后按序全交付、drain=0（DATA 不参与）
    auto *b_ctrl = new LinkProbe("b_ctrl", 1, mkbound(8, 1, 1, 1));
    b_ctrl->ctrl_burst = 6;
    b_ctrl->run_cycles = 160;
    b_ctrl->avail_lo = 0; // out_ctrl_avail 也由该窗口控制（下同 out_av/out_cav 共用窗口）
    b_ctrl->avail_hi = 30;
    // data+ctrl 同时：DATA 限速 1/4，CTRL 不受 data token 限制 —— 证明控制吞吐不被 DATA 拖慢
    auto *b_mix = new LinkProbe("b_mix", 1, mkbound(8, 8, 1, 4));
    b_mix->inject_burst = 8;
    b_mix->ctrl_burst = 8;
    b_mix->run_cycles = 200;

    // 深度扫描（L=1, rate=1）：测模型信用往返，找达到满速的最小深度（= 实测 BDP）。
    std::vector<int> scan_depths = {1, 2, 3, 4, 5, 8};
    std::vector<LinkProbe *> scan;
    for (int d : scan_depths) {
        auto *q = new LinkProbe(("scan_d" + std::to_string(d)).c_str(), 1,
                                mkbound(d, 4, 1, 1));
        q->inject_burst = 60;
        q->run_cycles = 320;
        scan.push_back(q);
    }

    // 通用 BDP 边界：每组同时跑 depth=BDP-1 与 depth=BDP。特别覆盖 L=0（clocked link 的
    // 正/反向各有一个最小服务拍）以及 1/2、2/3、1/4 有理速率，防止只在 L=1,rate=1 巧合成立。
    struct BdpBoundary {
        int latency, num, den;
        long long rtt, bdp;
        LinkProbe *under, *exact;
    };
    int bdp_specs[][3] = {{0, 1, 1}, {1, 1, 1}, {1, 1, 2},
                          {1, 2, 3}, {7, 1, 4}};
    std::vector<BdpBoundary> bdp_cases;
    for (auto &spec : bdp_specs) {
        D2DRate rate;
        rate.num = spec[1];
        rate.den = spec[2];
        long long rtt = D2DCreditRttCycles(spec[0]);
        long long bdp = D2DBdpPackets(spec[0], rate);
        if (bdp < 2)
            throw std::runtime_error("V3-b BDP test requires BDP>=2");
        std::string base = "bdp_L" + std::to_string(spec[0]) + "_r" +
                           std::to_string(spec[1]) + "_" + std::to_string(spec[2]);
        auto *under = new LinkProbe((base + "_under").c_str(), spec[0],
                                    mkbound((int)bdp - 1, 4, spec[1], spec[2]));
        auto *exact = new LinkProbe((base + "_exact").c_str(), spec[0],
                                    mkbound((int)bdp, 4, spec[1], spec[2]));
        under->inject_burst = exact->inject_burst = 96;
        under->run_cycles = exact->run_cycles = 720;
        bdp_cases.push_back({spec[0], spec[1], spec[2], rtt, bdp, under, exact});
    }

    // ---- V3 post-freeze hardening：直接执行生产 forward_bounded_saf ----
    // 保存并建立最小 2x1 拓扑，使生产 Link 的 SAF 排空走真实全局 reservation release，
    // 不用 test-only bypass。两个探针共享一条有向 link 的 admission 账本但使用不同 FlowKey。
    const int old_grid_x = GRID_X, old_grid_y = GRID_Y, old_grid_size = GRID_SIZE;
    const int old_die_x = DIE_X, old_die_y = DIE_Y, old_die_count = DIE_COUNT;
    const int old_cores_per_die = CORES_PER_DIE, old_total_cores = TOTAL_CORES;
    const int old_host_endpoint = HOST_ENDPOINT_ID;
    D2DLinkConfig old_d2d_cfg = g_d2d_cfg;
    std::vector<D2DLink> old_d2d_links = g_d2d_links;
    D2DPortTable old_die_ports = g_die_ports;
    GRID_X = GRID_Y = GRID_SIZE = CORES_PER_DIE = 1;
    DIE_X = DIE_COUNT = TOTAL_CORES = HOST_ENDPOINT_ID = 2;
    DIE_Y = 1;
    g_die_ports = {};
    g_die_ports.active = true;
    g_die_ports.ports = {
        {0, 0, EAST, ROLE_C2C, EAST, 1, 1, 64, -1},
        {1, 0, WEST, ROLE_C2C, WEST, 1, 1, 64, -1}};
    g_die_ports.port_for.assign(1, std::vector<int>(DIRECTIONS, -1));
    g_die_ports.port_for[0][EAST] = 0;
    g_die_ports.port_for[0][WEST] = 1;
    g_d2d_links = {{0, 0, 1, 1, 1, 1, 3},
                   {1, 1, 0, 0, 1, 1, 1}};
    g_d2d_cfg.mode = MODE_BOUNDED_SAF;
    g_d2d_cfg.safety = SAFETY_WHOLE_FLOW_SAF;
    g_d2d_cfg.saf_buffer_depth = 64;
    ResetWholeFlowSafRuntime();

    auto mkprod = [](int saf, int inflight, int rx, int ctrl,
                     int port_num, int port_den, int link_num, int link_den) {
        D2DLinkBound b;
        b.enabled = true;
        b.whole_flow_saf = true;
        b.saf_depth = saf;
        b.data_depth = inflight;
        b.rx_depth = rx;
        b.ctrl_depth = ctrl;
        b.port_rate = {port_num, port_den};
        b.rate = {link_num, link_den};
        return b;
    };

    // 生产 BDP 充分性：L=3,rate=1 的保守配置公式给出 depth=8。64 包整流后应连续
    // 1 pkt/cycle 交付；这验证“公式推荐深度在生产编码上足够”，不声称它是生产路径的最小值。
    D2DRate prod_full_rate{1, 1};
    int prod_bdp_depth = (int)D2DBdpPackets(3, prod_full_rate);
    auto *prod_bdp = new ProductionSafProbe(
        "prod_saf_bdp", 3, 0,
        mkprod(64, prod_bdp_depth, 8, 2, 1, 1, 1, 1), 0, 1, 101, 64);

    // 生产背压链：RX=1、inflight=2，下游长停顿；必须填满有限 stage、逐级反压，
    // 释放后仍按序排空。credit 必须按 toggle edge 逐包返回，连续翻转不可合并。
    auto *prod_bp = new ProductionSafProbe(
        "prod_saf_backpressure", 1, 1,
        mkprod(64, 2, 1, 2, 1, 1, 1, 1), 1, 0, 202, 64);
    prod_bp->data_stall_lo = 0;
    prod_bp->data_stall_hi = 200;

    ReserveWholeFlowSafPath(0, 1, 101, 0, 64);
    ReserveWholeFlowSafPath(1, 0, 202, 0, 64);

    // 构造期参数校验负例——**必须在 sc_start 之前**（elaboration 期才能构造 sc_module）。
    // 只接受**预期的 std::runtime_error 且错误文本含关键字**：避免把 SystemC 的其它异常（如
    // 「运行期插入模块失败」）误判为「参数校验成功」——这正是旧 catch(...) 掩盖的坑。
    {
        int uid = 0;
        auto throws_ctor = [&](D2DLinkBound b, const std::string &key) {
            try {
                D2DLinkUnit u(("bad_ctor_" + std::to_string(uid++)).c_str(), 1, -1,
                              b);
            } catch (const std::runtime_error &e) {
                return std::string(e.what()).find(key) != std::string::npos;
            }
            return false; // 未抛 / 抛非 runtime_error 都算失败
        };
        check(throws_ctor(mkbound(0, 4, 1, 1), "data_depth"),
              "V3-b ctor rejects data_depth=0 (expected runtime_error)");
        check(throws_ctor(mkbound(8, 0, 1, 1), "ctrl_depth"),
              "V3-b ctor rejects ctrl_depth=0 (expected runtime_error)");
        check(throws_ctor(mkbound(8, 4, 2, 1), "rate"),
              "V3-b ctor rejects rate>1 (2/1)");
        check(throws_ctor(mkbound(8, 4, 1, 0), "rate"),
              "V3-b ctor rejects rate den=0");
    }

    sc_start(760 * CYCLE, SC_NS); // 覆盖最慢的有理速率 BDP-1 边界及全部信用回还

    // 期望交付 id 序列（burst：base .. base+burst-1）
    auto expect_ids = [](int base, int burst) {
        std::vector<std::pair<int, int>> v;
        for (int i = 0; i < burst; i++)
            v.push_back({0, base + i});
        return v;
    };
    auto steady_gap = [](const std::vector<std::pair<int, int>> &out) {
        std::map<int, int> h;
        for (size_t i = 1; i < out.size(); i++)
            h[out[i].first - out[i - 1].first]++;
        int best = -1, bc = -1;
        for (auto &kv : h)
            if (kv.second > bc) {
                bc = kv.second;
                best = kv.first;
            }
        return best;
    };
    // 长期 goodput = (交付数-1)/(末拍-首拍)，逼近速率
    auto goodput = [](const std::vector<std::pair<int, int>> &out) {
        if (out.size() < 2)
            return -1.0;
        return (double)(out.size() - 1) /
               (double)(out.back().first - out.front().first);
    };
    // token 守恒（精确速率判据）：稳态区内**每个长度 den 的滑动窗恰含 num 次交付**。比「众数 gap」
    // 或「宽松 goodput 区间」强得多，且能表达 num>1 的一般速率（如 2/3：每 3 拍恰 2 包）。
    auto rate_exact = [](const std::vector<std::pair<int, int>> &out, int num,
                         int den) {
        if ((int)out.size() < 2 * den + 2)
            return false;
        int lo = out.front().first, hi = out.back().first;
        // 跳过首尾 den 拍的边界效应，检查中段每个 den 窗
        for (int t = lo + den; t + den <= hi - den; t++) {
            int cnt = 0;
            for (auto &o : out)
                if (o.first >= t && o.first < t + den)
                    cnt++;
            if (cnt != num)
                return false;
        }
        return true;
    };

    std::cout << "  [info] V3-b: b_rate gap=" << steady_gap(b_rate->data_out)
              << " occ=" << b_rate->link->OccMax()
              << " full=" << b_rate->link->FullCycles()
              << " ublk=" << b_rate->link->UpstreamBlocked()
              << " rstall=" << b_rate->link->RateStall()
              << " notready=" << b_rate->notready_seen
              << " inconsist=" << b_rate->inconsistent
              << " | 2/3 gp=" << goodput(b_23->data_out)
              << " | ctrl occ=" << b_ctrl->link->OccCtrlMax()
              << " cds=" << b_ctrl->link->CtrlDownstreamStall()
              << " | mix dgap=" << steady_gap(b_mix->data_out)
              << " cgap=" << steady_gap(b_mix->ctrl_out) << std::endl;
    std::cout << "  [info] depth scan (L=1,rate=1) goodput:";
    for (size_t i = 0; i < scan.size(); i++)
        std::cout << " d" << scan_depths[i] << "=" << goodput(scan[i]->data_out);
    std::cout << std::endl;
    std::cout << "  [info] BDP boundaries:";
    for (auto &c : bdp_cases)
        std::cout << " L" << c.latency << "@" << c.num << "/" << c.den
                  << "(rtt=" << c.rtt << ",bdp=" << c.bdp
                  << ",under=" << goodput(c.under->data_out)
                  << ",exact=" << goodput(c.exact->data_out) << ")";
    std::cout << std::endl;

    auto seq_1_to_n = [](const std::vector<int> &v, int n) {
        if ((int)v.size() != n)
            return false;
        for (int i = 0; i < n; ++i)
            if (v[i] != i + 1)
                return false;
        return true;
    };
    std::cout << "  [info] production SAF: bdp_depth=" << prod_bdp_depth
              << " bdp_span=" << (prod_bdp->last_out_cycle - prod_bdp->first_out_cycle)
              << " bp_occ=" << prod_bp->link->SafOccMax() << "/"
              << prod_bp->link->InflightOccMax() << "/"
              << prod_bp->link->RxOccMax()
              << " bp_stall=" << prod_bp->link->InflightFullCycles() << "/"
              << prod_bp->link->RxBackpressureStall() << "/"
              << prod_bp->link->DownstreamStall()
              << " credit_returns=" << prod_bdp->data_returns << "/"
              << prod_bp->data_returns << std::endl;

    check(seq_1_to_n(prod_bdp->out_seq, 64) &&
              prod_bdp->first_out_cycle > prod_bdp->tail_sent_cycle &&
              prod_bdp->link->SafOccMax() == 64,
          "V3-d production SAF gate: REQUEST declares F, no DATA leaves before complete 64-packet flow");
    check(prod_bdp_depth == 8 &&
              prod_bdp->last_out_cycle - prod_bdp->first_out_cycle == 63 &&
              prod_bdp->link->InflightOccMax() <= prod_bdp_depth &&
              prod_bdp->link->RxOccMax() <= 8 &&
              prod_bdp->link->PortRateStall() == 0 &&
              prod_bdp->link->LinkRateStall() == 0,
          "V3-d production BDP adequacy: configured depth=ceil((2L+PIPE)*rate)=8 sustains 1 pkt/cycle");
    check(prod_bdp->data_returns == 64 && prod_bdp->ctrl_returns == 1 &&
              prod_bdp->data_credit == 64 && prod_bdp->ctrl_credit == 2 &&
              prod_bp->data_returns == 64 && prod_bp->ctrl_returns == 1 &&
              prod_bp->data_credit == 64 && prod_bp->ctrl_credit == 2,
          "V3-d production toggle credit: every consecutive DATA/CTRL return observed once, credits restored");
    check(seq_1_to_n(prod_bp->out_seq, 64) &&
              prod_bp->first_out_cycle >= prod_bp->data_stall_hi &&
              prod_bp->link->SafOccMax() == 64 &&
              prod_bp->link->InflightOccMax() <= 2 &&
              prod_bp->link->RxOccMax() <= 1 &&
              prod_bp->link->SafFullCycles() > 0 &&
              prod_bp->link->InflightFullCycles() > 0 &&
              prod_bp->link->RxFullCycles() > 0 &&
              prod_bp->link->RxBackpressureStall() > 0 &&
              prod_bp->link->DownstreamStall() > 0 &&
              prod_bp->link->PortRateStall() == 0 &&
              prod_bp->link->LinkRateStall() == 0 &&
              prod_bp->link->UpstreamBlocked() == 0 &&
              prod_bp->link->CtrlUpstreamBlocked() == 0 &&
              prod_bp->notready_seen > 0,
          "V3-d production backpressure: finite SAF/inflight/RX fill without overflow, hold, then ordered drain");
    check(prod_bdp->link->residual() == 0 && prod_bp->link->residual() == 0 &&
              WholeFlowSafReservedPackets() == 0 &&
              prod_bdp->out_ctrl_types == std::vector<MSG_TYPE>{REQUEST} &&
              prod_bp->out_ctrl_types == std::vector<MSG_TYPE>{REQUEST},
          "V3-d production drain: FIFOs/SAF expectations/reservation ledger zero; REQUEST control preserved");

    // 流控 = 信用式（唯一真源）：上游只据信用发送，link 永不溢出（upstream_blocked==0）、全程无丢/重。
    // in_avail 仅作**诊断镜像**：背压时应观测到它拉低（notready>0）。inconsistent（有信用但 in_avail
    // 仍 false）只是 sc_signal 的 1 拍 settle 滞后，非流控真源冲突（信用才是），故仅诊断不断言。
    check(b_rate->notready_seen > 0 && b_rate->link->UpstreamBlocked() == 0,
          "V3-b credit flow control: upstream sends only on credit, link never overflows "
          "(upstream_blocked=0, lossless); in_avail (diagnostic) observed low under backpressure");

    // b_rate：速率限制 gap==4、占用<=depth、满状态(full)+速率停顿(rate_stall)触发、下游停顿=0、按序全收、排空
    check(ids_match(b_rate->data_out, expect_ids(b_rate->next_id, b_rate->inject_burst)) &&
              b_rate->link->OccMax() <= 2 && b_rate->link->FullCycles() > 0 &&
              b_rate->link->RateStall() > 0 && b_rate->link->DownstreamStall() == 0 &&
              steady_gap(b_rate->data_out) == 4 && b_rate->link->DataOcc() == 0,
          "V3-b depth=2 rate=1/4: gap==4, occ<=depth, full+rate_stall fire (downstream_stall=0), "
          "in-order, drained");

    // b_ser：depth=1 严格串行化——占用恒 <=1、**depth=1 出现满状态(full>0)**、按序全收、排空。
    // （不断言具体 gap：depth=1 时吞吐同时受速率与信用往返约束，二者会耦合；纯速率间隔由 b_rate/b_23 覆盖。）
    check(ids_match(b_ser->data_out, expect_ids(b_ser->next_id, b_ser->inject_burst)) &&
              b_ser->link->OccMax() <= 1 && b_ser->link->FullCycles() > 0 &&
              b_ser->link->DataOcc() == 0,
          "V3-b depth=1 serialization: occ<=1, full>0, in-order, drained");

    // b_ds：下游 stall——downstream_stall>0 且 rate_stall==0（区分「下游不 ready」与「token 不足」）、
    // 释放(>=40)后才首交付、按序全收、占用<=depth、排空
    check(ids_match(b_ds->data_out, expect_ids(b_ds->next_id, b_ds->inject_burst)) &&
              b_ds->link->DownstreamStall() > 0 && b_ds->link->RateStall() == 0 &&
              !b_ds->data_out.empty() &&
              b_ds->data_out.front().first >= b_ds->avail_hi &&
              b_ds->link->OccMax() <= 8 && b_ds->link->DataOcc() == 0,
          "V3-b downstream stall: held until release, downstream_stall>0 & rate_stall=0, "
          "in-order, drained");

    // BDP 契约验证：配置与测试共用 D2DCreditRttCycles/D2DBdpPackets。除完整 depth scan 外，
    // 多组 L/r 均要求 BDP 深度达到配置速率、BDP-1 受信用往返限制为 (BDP-1)/RTT；L=0 也必须成立。
    {
        const long long RTT = D2DCreditRttCycles(1); // =4
        bool ok = true;
        for (size_t i = 0; i < scan.size(); i++) {
            LinkProbe *q = scan[i];
            int d = scan_depths[i];
            double gp = goodput(q->data_out);
            double expect = std::min(1.0, (double)d / RTT);
            ok = ok &&
                 ids_match(q->data_out, expect_ids(q->next_id, q->inject_burst)) &&
                 q->link->residual() == 0 && q->link->OccMax() <= d &&
                 gp > expect - 0.06 && gp < expect + 0.06;
        }
        for (auto &c : bdp_cases) {
            double target = (double)c.num / c.den;
            double under_target = (double)(c.bdp - 1) / c.rtt;
            double gp_under = goodput(c.under->data_out);
            double gp_exact = goodput(c.exact->data_out);
            ok = ok &&
                 ids_match(c.under->data_out,
                           expect_ids(c.under->next_id, c.under->inject_burst)) &&
                 ids_match(c.exact->data_out,
                           expect_ids(c.exact->next_id, c.exact->inject_burst)) &&
                 c.under->link->residual() == 0 && c.exact->link->residual() == 0 &&
                 c.under->link->OccMax() <= c.bdp - 1 &&
                 c.exact->link->OccMax() <= c.bdp &&
                 gp_under > under_target - 0.025 &&
                 gp_under < under_target + 0.025 &&
                 gp_exact > target - 0.025 && gp_exact < target + 0.025 &&
                 gp_under < gp_exact - 0.02;
        }
        check(ok,
              "V3-b BDP contract: shared formula holds at BDP-1/BDP for "
              "L=0/1/7 and rate=1,1/2,2/3,1/4");
    }

    // 2/3 有理数：**精确 token 守恒**——稳态每 3 拍窗恰 2 包交付（不能用单一众数 gap 表达）、
    // 长期 goodput≈2/3、按序全收、排空
    check(ids_match(b_23->data_out, expect_ids(b_23->next_id, b_23->inject_burst)) &&
              rate_exact(b_23->data_out, 2, 3) &&
              goodput(b_23->data_out) > 0.63 && goodput(b_23->data_out) < 0.70 &&
              b_23->link->DataOcc() == 0,
          "V3-b rate=2/3: exact token conservation (every 3-cycle window has 2 deliveries), "
          "goodput~2/3, in-order, drained");

    // bounded 控制 FIFO：控制包驻留至释放、ctrl_downstream_stall>0、占用<=ctrl_depth、按序全收、
    // 排空；well-behaved 上游 ctrl_upstream_blocked==0
    check(ids_match(b_ctrl->ctrl_out, expect_ids(b_ctrl->next_cid, b_ctrl->ctrl_burst)) &&
              b_ctrl->link->CtrlDownstreamStall() > 0 &&
              b_ctrl->link->CtrlUpstreamBlocked() == 0 &&
              b_ctrl->link->OccCtrlMax() <= 1 && !b_ctrl->ctrl_out.empty() &&
              b_ctrl->ctrl_out.front().first >= b_ctrl->avail_hi &&
              b_ctrl->link->CtrlOcc() == 0,
          "V3-b bounded control FIFO: ctrl held until release, occ_ctrl<=depth, in-order, drained");

    // data+ctrl 同时：DATA 限速 1/4（gap 4），CTRL 不受 data token 限制（每拍可交付 → gap 1）。
    // 证明控制吞吐**不被 DATA 限速拖慢**；两通道各自按序全收、排空。
    check(ids_match(b_mix->data_out, expect_ids(b_mix->next_id, b_mix->inject_burst)) &&
              ids_match(b_mix->ctrl_out, expect_ids(b_mix->next_cid, b_mix->ctrl_burst)) &&
              steady_gap(b_mix->data_out) == 4 && steady_gap(b_mix->ctrl_out) == 1 &&
              b_mix->link->DataOcc() == 0 && b_mix->link->CtrlOcc() == 0,
          "V3-b data+ctrl concurrent: DATA rate-limited (gap 4) but CTRL unthrottled (gap 1), "
          "both in-order, drained");

    // 占用不变量（横向）：所有 bounded 探针峰值占用 <= 各自 depth
    check(b_rate->link->OccMax() <= 2 && b_ser->link->OccMax() <= 1 &&
              b_ds->link->OccMax() <= 8 && b_ctrl->link->OccCtrlMax() <= 1,
          "V3-b occupancy invariant: peak occupancy never exceeds configured depth");

    // 真正排空：不仅包 FIFO 为空，回程 credit queue/pulse 也为空，且 testbench 两路信用恢复初值。
    std::vector<LinkProbe *> bounded = {b_rate, b_ser, b_ds, b_23, b_ctrl, b_mix};
    bounded.insert(bounded.end(), scan.begin(), scan.end());
    for (auto &c : bdp_cases) {
        bounded.push_back(c.under);
        bounded.push_back(c.exact);
    }
    bool credit_clean = true;
    for (LinkProbe *q : bounded)
        credit_clean = credit_clean && q->link->residual() == 0 &&
                       q->link->CreditResidual() == 0 &&
                       q->credit == q->link->bound.data_depth &&
                       q->ccredit == q->link->bound.ctrl_depth;
    check(credit_clean,
          "V3-b drain: DATA/CTRL FIFO empty, credit return queues/pulses empty, credits restored");

    // 干净探针：序 + latency
    int base = -999;
    for (size_t k = 0; k < probes.size(); k++) {
        LinkProbe *p = probes[k];
        check(ids_match(p->data_out, dscript),
              "L" + std::to_string(lats[k]) + " data: order + no loss/dup");
        check(ids_match(p->ctrl_out, cscript),
              "L" + std::to_string(lats[k]) + " ctrl: order + no loss/dup (independent)");
        int dd = const_delta(p->data_out, dscript);
        int dc = const_delta(p->ctrl_out, cscript);
        check(dd >= 0 && dd == dc,
              "L" + std::to_string(lats[k]) + " constant per-packet delay (data==ctrl)");
        if (lats[k] == 0)
            base = dd; // 以 L=0 的实测偏移为基准
        else if (base >= 0)
            check(dd == base + lats[k],
                  "L" + std::to_string(lats[k]) + " delay == base + latency (relative latency correct)");
    }
    std::cout << "  [info] measured base offset (probe-observed, L=0) = " << base
              << " -> delivery cycle = capture + latency + " << base
              << " (constant across latencies)" << std::endl;

    // 停顿探针诊断
    std::cout << "  [info] stall data_out(cycle,id):";
    for (auto &o : stall->data_out)
        std::cout << " (" << o.first << "," << o.second << ")";
    std::cout << std::endl;
    // 停顿：无丢/重 + 按序 + 窗口内无任何交付（首包也在窗口结束后才出）
    bool order_ok = ids_match(stall->data_out, stall->data_in);
    bool held = !stall->data_out.empty() &&
                stall->data_out.front().first >= stall->avail_hi;
    check(order_ok && held,
          "out_avail stall: no delivery until window releases, then in order, no loss/dup");

    // idle 探针：无输出
    check(idle->data_out.empty() && idle->ctrl_out.empty(),
          "idle link: no output");

    // drain：所有真实包都出（全局 in==out，且 >0）
    check(g_d2d_link_in_pkts == g_d2d_link_out_pkts && g_d2d_link_in_pkts > 0,
          "drain: total in_pkts == out_pkts (" +
              std::to_string(g_d2d_link_in_pkts) + ")");

    // 恢复调用者的全局拓扑/模式；self-test 不污染同进程的其它入口。
    GRID_X = old_grid_x;
    GRID_Y = old_grid_y;
    GRID_SIZE = old_grid_size;
    DIE_X = old_die_x;
    DIE_Y = old_die_y;
    DIE_COUNT = old_die_count;
    CORES_PER_DIE = old_cores_per_die;
    TOTAL_CORES = old_total_cores;
    HOST_ENDPOINT_ID = old_host_endpoint;
    g_d2d_cfg = old_d2d_cfg;
    g_die_ports = old_die_ports;
    g_d2d_links = old_d2d_links;
    ResetWholeFlowSafRuntime();

    std::cout << "==== D2D V1 link self-test: " << (g_total - g_fail) << "/"
              << g_total << (g_fail ? "  <<< FAILURES" : "") << " ===="
              << std::endl;
    return g_fail;
}
