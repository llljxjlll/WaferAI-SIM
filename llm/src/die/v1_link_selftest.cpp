// V1-b2 独立 SystemC link 测试：用一个 testbench 驱动真实包穿过 D2DLinkUnit，验证
// latency 语义、FIFO 序、无丢包/重复、out_avail 停顿（成熟包等待、不丢不重）、
// drain 后 in==out、data/ctrl 独立、idle 统计为 0。一次 sc_start（sc_start(时长)）跑完。
#include "die/d2d_link.h"
#include "die/port.h"
#include "systemc.h"
#include <iostream>
#include <string>
#include <vector>

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

    SC_HAS_PROCESS(LinkProbe);
    LinkProbe(sc_module_name n, int latency) : sc_module(n) {
        link = new D2DLinkUnit("link", latency);
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
            // 记录上一拍已 settle 的输出
            if (out_s.read())
                data_out.push_back({c, (int)out_ch.read().to_uint()});
            if (out_cs.read())
                ctrl_out.push_back({c, (int)out_cch.read().to_uint()});
            // 下游 ready（停顿窗内 false）
            bool av = !(c >= avail_lo && c < avail_hi);
            out_av.write(av);
            out_cav.write(av);
            // 驱动本拍输入
            bool ds = false;
            int did = 0;
            for (auto &p : data_in)
                if (p.first == c) {
                    ds = true;
                    did = p.second;
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

    sc_start(80 * CYCLE, SC_NS); // 一次跑完（固定时长，无需 sc_stop）

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
