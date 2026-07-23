// 独立 SystemC link 测试：用 testbench 驱动真实包穿过 D2DLinkUnit。
// V1-b2：latency 语义、FIFO 序、无丢/重、out_avail 停顿、drain 后 in==out、data/ctrl 独立、idle=0。
// V3-b：**有限 FIFO + token-bucket 速率**（仅此独立测试开启；生产路径仍 functional_v2 无限 FIFO）——
//   depth 1/2/8、rate 1/(1/2)/(1/4)、下游长 stall；断言占用不越 depth、背压(full)/速率停顿(rate_stall)/
//   下游停顿(ds_stall)分别正确触发、稳态间隔==den/num、按序无丢/重、结束排空。driver 用信用式流控。
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

    // V3-b：burst 注入（>0 时启用）。上游想连发 inject_burst 个包，用**信用式流控**（driver 本地、
    // 确定、无跨线程竞态）：初始信用 = FIFO 深度，发一包 -1、观测到一次交付(out_s) +1——信用即
    // 「link 中尚有空位」，故绝不过量注入、link 永不丢包，占用可达 depth 触发背压，无 valid/ready 偏移竞态。
    int inject_burst = 0;
    int next_id = 500; // burst 首包 id
    int sent_count = 0;
    int credit = 1 << 30; // functional 时视作无限；bounded 时构造函数置为 depth

    SC_HAS_PROCESS(LinkProbe);
    LinkProbe(sc_module_name n, int latency, D2DLinkBound bound = D2DLinkBound{})
        : sc_module(n) {
        if (bound.enabled)
            credit = bound.data_depth;
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
            // 记录上一拍已 settle 的输出；每观测到一次数据交付 → 归还 1 信用（link 腾出 1 空位）
            if (out_s.read()) {
                data_out.push_back({c, (int)out_ch.read().to_uint()});
                credit++;
            }
            if (out_cs.read())
                ctrl_out.push_back({c, (int)out_cch.read().to_uint()});
            // 下游 ready（停顿窗内 false）
            bool av = !(c >= avail_lo && c < avail_hi);
            out_av.write(av);
            out_cav.write(av);
            // 驱动本拍输入
            bool ds = false;
            int did = 0;
            if (inject_burst > 0) {
                // burst：有信用（link 有空位）且还没发完 → 注入下一个 id，扣 1 信用
                if (sent_count < inject_burst && credit > 0) {
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
            bool cs = false;
            int cid = 0;
            for (auto &p : ctrl_in)
                if (p.first == c) {
                    cs = true;
                    cid = p.second;
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

    // ---- V3-b：有限缓冲 + token bucket（burst 注入、遵守 in_avail 背压）----
    auto mkbound = [](int depth, int cdepth, int num, int den) {
        D2DLinkBound b;
        b.enabled = true;
        b.data_depth = depth;
        b.ctrl_depth = cdepth;
        b.rate.num = num;
        b.rate.den = den;
        return b;
    };
    // b_full: depth=8, rate=1 —— 满速交付，无速率停顿
    auto *b_full = new LinkProbe("b_full", 1, mkbound(8, 4, 1, 1));
    b_full->inject_burst = 20;
    b_full->run_cycles = 150;
    // b_rate: depth=2, rate=1/4 —— 慢速 + 浅 FIFO：背压(full)+速率停顿，占用恰达 depth
    auto *b_rate = new LinkProbe("b_rate", 1, mkbound(2, 4, 1, 4));
    b_rate->inject_burst = 8;
    b_rate->run_cycles = 150;
    // b_ser: depth=1, rate=1/2 —— 严格串行化，占用恒 <=1
    auto *b_ser = new LinkProbe("b_ser", 0, mkbound(1, 4, 1, 2));
    b_ser->inject_burst = 6;
    b_ser->run_cycles = 150;
    // b_ds: depth=8, rate=1, 下游 [0,40) 全程 stall —— 成熟包驻留、ds_stall>0、释放后按序全交付
    auto *b_ds = new LinkProbe("b_ds", 1, mkbound(8, 4, 1, 1));
    b_ds->inject_burst = 5;
    b_ds->run_cycles = 150;
    b_ds->avail_lo = 0; // 从头 stall：整个 [0,40) 下游不 ready，成熟包全部驻留至释放
    b_ds->avail_hi = 40;

    sc_start(170 * CYCLE, SC_NS); // 一次跑完（覆盖最慢的 bounded 探针）

    // 期望交付 id 序列（burst：next_id .. next_id+burst-1）
    auto expect_ids = [](LinkProbe *p) {
        std::vector<std::pair<int, int>> v;
        for (int i = 0; i < p->inject_burst; i++)
            v.push_back({0, p->next_id + i});
        return v;
    };
    // 连续交付的稳态间隔（众数）
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
    std::cout << "  [info] V3-b bounded: "
              << "b_full occ=" << b_full->link->OccMax()
              << " full=" << b_full->link->FullCycles()
              << " rstall=" << b_full->link->RateStall()
              << " ds=" << b_full->link->DsStall()
              << " gap=" << steady_gap(b_full->data_out) << " | "
              << "b_rate occ=" << b_rate->link->OccMax()
              << " full=" << b_rate->link->FullCycles()
              << " rstall=" << b_rate->link->RateStall()
              << " gap=" << steady_gap(b_rate->data_out) << " | "
              << "b_ser occ=" << b_ser->link->OccMax()
              << " gap=" << steady_gap(b_ser->data_out) << " | "
              << "b_ds occ=" << b_ds->link->OccMax()
              << " ds=" << b_ds->link->DsStall()
              << " first_out=" << (b_ds->data_out.empty() ? -1 : b_ds->data_out.front().first)
              << std::endl;

    // b_full：满速交付——按序全收、稳态每拍 1 包、无速率/下游停顿、占用不越界、排空
    check(ids_match(b_full->data_out, expect_ids(b_full)) &&
              b_full->link->OccMax() <= 8 && b_full->link->DataOcc() == 0 &&
              b_full->link->RateStall() == 0 && b_full->link->DsStall() == 0 &&
              steady_gap(b_full->data_out) == 1,
          "V3-b bounded depth=8 rate=1: full-rate in-order, occ<=depth, no stall, drained");

    // b_rate：慢速 + 浅 FIFO——占用**恰达 depth 但从不越界**、背压(full)+速率停顿、
    // 稳态交付间隔 == den/num == 4、按序全收、排空
    check(ids_match(b_rate->data_out, expect_ids(b_rate)) &&
              b_rate->link->OccMax() == 2 && b_rate->link->FullCycles() > 0 &&
              b_rate->link->RateStall() > 0 &&
              steady_gap(b_rate->data_out) == 4 && b_rate->link->DataOcc() == 0,
          "V3-b bounded depth=2 rate=1/4: occ==depth (never exceeds), backpressure + "
          "rate stall, steady gap==4, in-order, drained");

    // b_ser：depth=1 严格串行化、rate=1/2 → 间隔 2、占用恒 <=1、按序全收
    check(ids_match(b_ser->data_out, expect_ids(b_ser)) &&
              b_ser->link->OccMax() == 1 &&
              steady_gap(b_ser->data_out) == 2 && b_ser->link->DataOcc() == 0,
          "V3-b bounded depth=1 rate=1/2: strict serialization occ<=1, gap==2, in-order");

    // b_ds：下游长 stall——成熟包驻留、ds_stall>0、直到窗口释放(>=40)才首次交付、按序全收、
    // 占用不越界、排空（把「有限 FIFO + 下游背压」与「速率限制」区分开）
    check(ids_match(b_ds->data_out, expect_ids(b_ds)) &&
              b_ds->link->DsStall() > 0 && !b_ds->data_out.empty() &&
              b_ds->data_out.front().first >= b_ds->avail_hi &&
              b_ds->link->OccMax() <= 8 && b_ds->link->DataOcc() == 0,
          "V3-b bounded downstream stall: held until release, ds_stall>0, in-order, "
          "occ<=depth, drained");

    // 占用不变量（横向）：每个 bounded 探针的峰值占用都 <= 各自配置 depth（有限 FIFO 的核心保证）
    check(b_full->link->OccMax() <= 8 && b_rate->link->OccMax() <= 2 &&
              b_ser->link->OccMax() <= 1 && b_ds->link->OccMax() <= 8,
          "V3-b occupancy invariant: peak occupancy never exceeds configured depth");

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
