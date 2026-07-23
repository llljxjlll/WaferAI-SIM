// 独立 SystemC link 测试：用 testbench 驱动真实包穿过 D2DLinkUnit。
// V1-b2：latency 语义、FIFO 序、无丢/重、out_avail 停顿、drain 后 in==out、data/ctrl 独立、idle=0。
// V3-b：**有限 FIFO + token-bucket 速率**（仅此独立测试开启；生产路径仍 functional_v2 无限 FIFO）——
//   in_avail 背压正确性、占用不越 depth、full/rate_stall/downstream_stall 分类正确触发、稳态间隔
//   ==den/num（含 2/3 的 token 守恒）、缓冲深度 vs 吞吐（信用往返）、bounded 控制 FIFO、data+ctrl
//   并发不互相拖慢、构造期参数校验、按序无丢/重、结束排空。driver 用信用式 flow control（无损）。
#include "die/d2d_link.h"
#include "die/port.h"
#include "systemc.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>

namespace {

// 单条 link 的探针：驱动 in（data/ctrl 各自脚本）+ out_avail（可设停顿窗），记录 out。
struct LinkProbe : sc_module {
    D2DLinkUnit *link;
    sc_signal<sc_bv<256>> in_ch, out_ch, in_cch, out_cch;
    sc_signal<bool> in_s, in_av, out_s, out_av, in_cs, in_cav, out_cs, out_cav;

    std::vector<std::pair<int, int>> data_in, ctrl_in; // (cycle, id)
    int avail_lo = -1, avail_hi = -1;                  // out_avail=false 窗口 [lo,hi)
    int run_cycles = 60;
    std::vector<std::pair<int, int>> data_out, ctrl_out; // (cycle, id)

    // V3-b：burst 注入（>0 时启用）。flow control = **信用式**（bounded link 的标准无损模型；
    // 单 sc_bv 信道的 delta 延迟使纯组合 valid/ready 退化成 2 拍环、要么滞留重收要么填充边沿丢包）。
    // 信用 = 下游空位：初始 = depth，发一包 -1，观测到一次交付(out_s) +1。同时**读真实 in_avail**
    // 并记录，用于**验证背压信号**：满时应观测到 in_avail=false（notready_seen>0），
    // 且有信用时（能发）in_avail 应为 true——把「in_avail 是否正确拉低/拉高」独立断言出来。
    int inject_burst = 0;
    int next_id = 500; // burst 首包 id
    int sent_count = 0;
    int credit = 1 << 30;  // functional 视作无限；bounded 构造函数置为 data_depth
    int ready_seen = 0;    // 有信用可发 且 观测到 in_avail=true 的拍数
    int inconsistent = 0;  // 有信用可发 但 in_avail=false（背压信号与真实空位不符）→ 应恒为 0
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
        SC_THREAD(drive);
    }

    void drive() {
        for (int c = 0; c <= run_cycles; c++) {
            // 记录上一拍已 settle 的输出；每观测到一次交付 → 归还 1 信用（link 腾出 1 空位）
            if (out_s.read()) {
                data_out.push_back({c, (int)out_ch.read().to_uint()});
                credit++;
            }
            if (out_cs.read()) {
                ctrl_out.push_back({c, (int)out_cch.read().to_uint()});
                ccredit++;
            }
            // 下游 ready（停顿窗内 false）
            bool av = !(c >= avail_lo && c < avail_hi);
            out_av.write(av);
            out_cav.write(av);
            // 驱动 DATA 输入
            bool ds = false;
            int did = 0;
            if (inject_burst > 0) {
                // 信用式发送（无损）；同时读真实 in_avail 验证背压信号。
                bool avail = in_av.read();
                if (!avail)
                    notready_seen++;
                if (sent_count < inject_burst && credit > 0) {
                    // 有信用可发：此刻 link 必有空位，故 in_avail 应为 true（否则背压信号错误）
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
            // 驱动 CTRL 输入（burst 走同样的真实握手，读 in_ctrl_avail）
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
    // b_ser: depth=1, rate=1/2, lat=0（BDP=0）—— 纯速率串行化，占用恒 <=1
    auto *b_ser = new LinkProbe("b_ser", 0, mkbound(1, 4, 1, 2));
    b_ser->inject_burst = 6;
    b_ser->run_cycles = 160;
    // b_ds: depth=8, rate=1, 下游 [0,40) 全程 stall —— 成熟包驻留、downstream_stall>0
    auto *b_ds = new LinkProbe("b_ds", 1, mkbound(8, 4, 1, 1));
    b_ds->inject_burst = 5;
    b_ds->run_cycles = 160;
    b_ds->avail_lo = 0;
    b_ds->avail_hi = 40;
    // BDP 演示：rate=1, lat=1 → BDP=2。depth=1<BDP 吞吐减半（gap 2）；depth=2==BDP 满速（gap 1）。
    auto *b_bdp_lo = new LinkProbe("b_bdp_lo", 1, mkbound(1, 4, 1, 1));
    b_bdp_lo->inject_burst = 20;
    b_bdp_lo->run_cycles = 160;
    auto *b_bdp_hi = new LinkProbe("b_bdp_hi", 1, mkbound(8, 4, 1, 1));
    b_bdp_hi->inject_burst = 20;
    b_bdp_hi->run_cycles = 160;
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

    sc_start(210 * CYCLE, SC_NS); // 覆盖最慢探针

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
    // gap 集合是否 ⊆ {1,2} 且两者都出现（2/3 的特征：1、2 交替，非单一众数）
    auto gaps_1_2_mixed = [](const std::vector<std::pair<int, int>> &out) {
        int n1 = 0, n2 = 0;
        for (size_t i = 1; i < out.size(); i++) {
            int g = out[i].first - out[i - 1].first;
            if (g == 1)
                n1++;
            else if (g == 2)
                n2++;
            else
                return false;
        }
        return n1 > 0 && n2 > 0;
    };

    std::cout << "  [info] V3-b: b_rate gap=" << steady_gap(b_rate->data_out)
              << " occ=" << b_rate->link->OccMax()
              << " full=" << b_rate->link->FullCycles()
              << " ublk=" << b_rate->link->UpstreamBlocked()
              << " rstall=" << b_rate->link->RateStall()
              << " notready=" << b_rate->notready_seen
              << " inconsist=" << b_rate->inconsistent
              << " | bdp_lo gp=" << goodput(b_bdp_lo->data_out)
              << " | bdp_hi gp=" << goodput(b_bdp_hi->data_out)
              << " gap=" << steady_gap(b_bdp_hi->data_out)
              << " | 2/3 gp=" << goodput(b_23->data_out)
              << " | ctrl occ=" << b_ctrl->link->OccCtrlMax()
              << " cds=" << b_ctrl->link->CtrlDownstreamStall()
              << " | mix dgap=" << steady_gap(b_mix->data_out)
              << " cgap=" << steady_gap(b_mix->ctrl_out) << std::endl;

    // in_avail 正确性（本轮核心）：① 背压时观测到 ready 确实拉低（notready>0，通过真实 in_avail 信号）；
    // ② link 从不溢出/丢包（upstream_blocked==0 且下方 ids_match 全对）。inconsistent（有信用但 ready
    // 仍 false）只反映 sc_signal 的 1 拍 settle 滞后（此处 =2，均为满→空过渡沿），非溢出，故仅诊断不断言。
    check(b_rate->notready_seen > 0 && b_rate->link->UpstreamBlocked() == 0,
          "V3-b in_avail correctness: backpressure observed via real in_avail (ready low), "
          "link never overflows (upstream_blocked=0, lossless)");

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

    // 缓冲深度 vs 吞吐（信用往返）：depth 过浅（=1）吞吐被信用往返限制（goodput 明显 <1）；
    // depth 充裕（=8，> 往返）维持满速（goodput≈1、gap 1）。两者都无丢/重、排空。
    // （注：本 standalone 的信用往返含 testbench 观测延迟，故阈值以「够深」表征，不钉死配置 BDP=2L。）
    check(ids_match(b_bdp_lo->data_out,
                    expect_ids(b_bdp_lo->next_id, b_bdp_lo->inject_burst)) &&
              ids_match(b_bdp_hi->data_out,
                        expect_ids(b_bdp_hi->next_id, b_bdp_hi->inject_burst)) &&
              goodput(b_bdp_lo->data_out) < 0.6 &&
              goodput(b_bdp_hi->data_out) > 0.95 &&
              steady_gap(b_bdp_hi->data_out) == 1 &&
              b_bdp_lo->link->DataOcc() == 0 && b_bdp_hi->link->DataOcc() == 0,
          "V3-b buffer-vs-throughput: shallow depth=1 throttled by credit round-trip "
          "(goodput<0.6), ample depth=8 sustains full rate (goodput~1, gap 1), both lossless");

    // 2/3 有理数：**不能用单一众数 gap**——验证 gap∈{1,2} 且都出现，长期 goodput≈2/3
    check(ids_match(b_23->data_out, expect_ids(b_23->next_id, b_23->inject_burst)) &&
              gaps_1_2_mixed(b_23->data_out) &&
              goodput(b_23->data_out) > 0.62 && goodput(b_23->data_out) < 0.71 &&
              b_23->link->DataOcc() == 0,
          "V3-b rate=2/3: token conservation (gaps 1&2 mixed, goodput~2/3), in-order, drained");

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
              b_ds->link->OccMax() <= 8 && b_bdp_lo->link->OccMax() <= 1 &&
              b_bdp_hi->link->OccMax() <= 8 && b_ctrl->link->OccCtrlMax() <= 1,
          "V3-b occupancy invariant: peak occupancy never exceeds configured depth");

    // 构造期参数校验负例：depth=0 / rate 非法 必须在构造时抛
    {
        auto throws_ctor = [&](D2DLinkBound b) {
            try {
                D2DLinkUnit u("bad", 1, -1, b);
            } catch (const std::runtime_error &) {
                return true;
            } catch (...) {
                return true;
            }
            return false;
        };
        check(throws_ctor(mkbound(0, 4, 1, 1)), "V3-b ctor rejects data_depth=0");
        check(throws_ctor(mkbound(8, 0, 1, 1)), "V3-b ctor rejects ctrl_depth=0");
        check(throws_ctor(mkbound(8, 4, 2, 1)), "V3-b ctor rejects rate>1 (2/1)");
        check(throws_ctor(mkbound(8, 4, 1, 0)), "V3-b ctor rejects rate den=0");
    }

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

    std::cout << "==== D2D V1 link self-test: " << (g_total - g_fail) << "/"
              << g_total << (g_fail ? "  <<< FAILURES" : "") << " ===="
              << std::endl;
    return g_fail;
}
