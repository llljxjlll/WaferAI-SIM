#include "die/d2d_link.h"
#include "die/behavioral.h"
#include "die/port.h"
#include "monitor/watchdog.h"
#include "utils/msg_utils.h"
#include <stdexcept>

namespace {
inline void MixWord(unsigned long long &h, unsigned long long w);
void CountType(long (&counts)[MSG_TYPE_NUM], const sc_bv<256> &payload) {
    int type = static_cast<int>(DeserializeMsg(payload).msg_type_);
    if (type >= 0 && type < MSG_TYPE_NUM)
        counts[type]++;
}

// V2-c：按有向 link 下标分类计数（下标越界/未归因则跳过）。
void CountLink(int idx, bool is_in, const sc_bv<256> &payload) {
    if (idx < 0 || idx >= (int)g_d2d_link_stats.size())
        return;
    int type = static_cast<int>(DeserializeMsg(payload).msg_type_);
    if (type < 0 || type >= MSG_TYPE_NUM)
        return;
    if (is_in)
        g_d2d_link_stats[idx].in_by_type[type]++;
    else
        g_d2d_link_stats[idx].out_by_type[type]++;

    Msg m = DeserializeMsg(payload);
    if (m.msg_type_ != DATA)
        return;
    auto key = std::make_tuple(idx, m.source_, m.tag_id_, m.subflow_);
    V5SubflowStat &st = g_v5_subflow_stats[key];
    long &pkts = is_in ? st.in_pkts : st.out_pkts;
    unsigned long long &seqhash = is_in ? st.in_seqhash : st.out_seqhash;
    unsigned long long &csum = is_in ? st.in_csum : st.out_csum;
    bool first = pkts == 0;
    pkts++;
    MixWord(seqhash, (unsigned long long)(m.seq_id_ + 1));
    for (int w = 0; w < 256; w += 64)
        MixWord(csum, payload.range(w + 63, w).to_uint64());
    if (!is_in) {
        if (!first && m.seq_id_ != st.out_expect)
            st.out_inorder = false;
        st.out_expect = m.seq_id_ + 1;
        if (first || m.seq_id_ < st.out_minseq)
            st.out_minseq = m.seq_id_;
        if (m.seq_id_ > st.out_maxseq)
            st.out_maxseq = m.seq_id_;
        if (m.is_end_) {
            st.out_endseq = m.seq_id_;
            st.out_end_count++;
            st.out_end_length = m.length_;
        }
    }
}

// FNV-1a 风格顺序敏感混合：h 依次吸收一个 64-bit 字。乱序/改值都会改变结果。
inline void MixWord(unsigned long long &h, unsigned long long w) {
    h = (h ^ w) * 1099511628211ULL;
}

// 对 DATA 型包更新完整性探针：seqhash 吸收 seq_id，csum 吸收完整 256-bit payload（含 data 段），
// out 侧另记 canonical 形状（inorder / endseq / maxseq）。非 DATA 包不触碰探针。
void ProbeData(D2DDataProbe &p, const sc_bv<256> &payload, long long cycle) {
    Msg m = DeserializeMsg(payload);
    if (m.msg_type_ != DATA)
        return;
    bool first = (p.pkts == 0);
    p.pkts++;
    if (first)
        p.first_cycle = cycle;
    p.last_cycle = cycle;
    MixWord(p.seqhash, (unsigned long long)(m.seq_id_ + 1));
    for (int w = 0; w < 256; w += 64)
        MixWord(p.csum, payload.range(w + 63, w).to_uint64());
    if (!first && m.seq_id_ != p.expect) // 非首包必须等于 prev+1（连续、不丢/不重/不乱序）
        p.inorder = false;
    p.expect = m.seq_id_ + 1;
    if (first || m.seq_id_ < p.minseq)
        p.minseq = m.seq_id_;
    if (m.seq_id_ > p.maxseq)
        p.maxseq = m.seq_id_;
    if (m.is_end_) {
        p.endseq = m.seq_id_;
        p.end_count++;
        p.end_length = m.length_;
    }
}
} // namespace

D2DLinkUnit::D2DLinkUnit(const sc_module_name &n, int latency_, int link_idx_,
                         D2DLinkBound bound_, D2DLinkBehavioral behavioral_)
    : sc_module(n), latency(latency_), link_idx(link_idx_), bound(bound_),
      behavioral(behavioral_) {
    // V3-b：构造期硬校验 bounded 参数（生产解析器已保证，但 standalone/其它调用者可能绕过）。
    if (bound.enabled && behavioral.enabled)
        throw std::runtime_error(
            "D2DLinkUnit: bounded and behavioral modes are mutually exclusive");
    if (behavioral.enabled) {
        if (!behavioral.port_rate.Valid() || !behavioral.link_rate.Valid())
            throw std::runtime_error(
                "D2DLinkUnit: behavioral rates must satisfy 0 < num/den <= 1");
        if (latency < 0)
            throw std::runtime_error(
                "D2DLinkUnit: behavioral latency must be >= 0");
    }
    if (bound.enabled) {
        if (bound.data_depth < 1)
            throw std::runtime_error(
                "D2DLinkUnit: bounded data_depth must be >= 1 (a depth-0 link "
                "would never accept)");
        if (bound.ctrl_depth < 1)
            throw std::runtime_error(
                "D2DLinkUnit: bounded ctrl_depth must be >= 1");
        if (!bound.rate.Valid())
            throw std::runtime_error(
                "D2DLinkUnit: bounded rate must satisfy 0 < num/den <= 1");
        if (bound.whole_flow_saf) {
            if (bound.saf_depth < 1 || bound.rx_depth < 1)
                throw std::runtime_error(
                    "D2DLinkUnit: production SAF/RX depths must be >= 1");
            if (!bound.port_rate.Valid())
                throw std::runtime_error(
                    "D2DLinkUnit: port_rate must satisfy 0 < num/den <= 1");
        }
    }
    SC_THREAD(forward);
}

// 每 cycle：采集上游真实包（in_sent）入 FIFO，记 ready_cycle=capture_cycle+latency；队首成熟
// 且下游 out_avail=true 时出队一包（FIFO 序）。latency 语义：capture 后至少 latency 拍成熟；
// 若下游一直 ready，交付 cycle = capture_cycle + latency。下游不 ready 时成熟包留在队首等待
// （不丢、不重、不越过下游容量）。V1：对上游恒 avail=true（Link FIFO 视作无限功能队列，
// 不向上游施背压——有限缓冲背压属 V3）。数据/控制各一条独立 FIFO。
void D2DLinkUnit::forward() {
    long cyc = 0;
    while (true) {
        wait(CYCLE, SC_NS);
        cyc++;
        if (behavioral.enabled) {
            forward_behavioral(cyc);
            continue;
        }
        if (bound.enabled) {
            if (bound.whole_flow_saf)
                forward_bounded_saf(cyc);
            else
                forward_bounded(cyc);
            continue;
        }
        // ---- functional_v2（V2 冻结路径，逐字节不变；恒 avail=true、无速率/背压）----
        in_avail.write(true);
        in_ctrl_avail.write(true);
        data_credit_return.write(false); // functional 无信用机制
        ctrl_credit_return.write(false);
        data_credit_active_ = false;
        ctrl_credit_active_ = false;

        if (in_sent.read()) {
            sc_bv<256> payload = in_channel.read();
            fifo_.push_back({cyc + latency, payload});
            g_d2d_link_in_pkts++;
            CountType(g_d2d_link_in_by_type, payload);
            CountLink(link_idx, true, payload);
            ProbeData(g_d2d_data_in, payload, cyc);
        }
        if (in_ctrl_sent.read()) {
            cfifo_.push_back({cyc + latency, in_ctrl_channel.read()});
            g_d2d_link_in_pkts++;
            CountType(g_d2d_link_in_by_type, in_ctrl_channel.read());
            CountLink(link_idx, true, in_ctrl_channel.read());
        }

        bool group_request =
            !fifo_.empty() && fifo_.front().first <= cyc && out_avail.read();
        bool group_ok = V5LinkGroupGrant(link_idx, cyc, group_request);
        if (group_request && group_ok) {
            out_channel.write(fifo_.front().second);
            out_sent.write(true);
            CountType(g_d2d_link_out_by_type, fifo_.front().second);
            CountLink(link_idx, false, fifo_.front().second);
            ProbeData(g_d2d_data_out, fifo_.front().second, cyc);
            fifo_.pop_front();
            g_d2d_link_out_pkts++;
            g_protocol_progress++;
        } else {
            out_sent.write(false);
            if (group_request && !group_ok)
                link_group_stall++;
        }
        if (!cfifo_.empty() && cfifo_.front().first <= cyc &&
            out_ctrl_avail.read()) {
            out_ctrl_channel.write(cfifo_.front().second);
            out_ctrl_sent.write(true);
            CountType(g_d2d_link_out_by_type, cfifo_.front().second);
            CountLink(link_idx, false, cfifo_.front().second);
            cfifo_.pop_front();
            g_d2d_link_out_pkts++;
            g_protocol_progress++;
        } else {
            out_ctrl_sent.write(false);
        }
    }
}

// V4 Behavioral 生产路径。REQUEST/ACK/DATA 各保留一个代表消息穿过真实 Router；控制消息
// 每 hop 只加固定 link latency。DATA 在源 die 的第一条有向 link 上额外加一次 end-to-end
// 聚合服务 ceil(F/min(1,port_rate,link_rate))，后续 hop 只加 latency，因此多跳是 pipelined
// min-cut，而不是逐 hop 重复 bulk serialization。事件表无容量上限、无跨-flow资源争用。
void D2DLinkUnit::forward_behavioral(long cyc) {
    in_avail.write(true);
    in_ctrl_avail.write(true);
    data_credit_return.write(false);
    ctrl_credit_return.write(false);
    data_credit_active_ = false;
    ctrl_credit_active_ = false;

    if (in_sent.read()) {
        sc_bv<256> payload = in_channel.read();
        Msg m = DeserializeMsg(payload);
        if (m.msg_type_ != DATA)
            throw std::runtime_error(
                "V4 Behavioral data channel received a non-DATA message");
        if (m.roofline_packets_ < 1)
            throw std::runtime_error(
                "V4 Behavioral DATA requires roofline_packets >= 1");
        long long delay = latency;
        bool first_link = false;
        if (link_idx >= 0 && link_idx < (int)g_d2d_links.size())
            first_link =
                g_d2d_links[link_idx].local_die == DieOfGlobal(m.source_);
        if (first_link && m.subflow_ == 0) {
            D2DBehavioralFlowMeta meta =
                ConsumeD2DBehavioralFlow(m.source_, m.tag_id_);
            if (meta.dest_global != m.des_)
                throw std::runtime_error(
                    "Behavioral DATA destination disagrees with flow metadata");
            D2DBehavioralEstimate estimate = EstimateD2DBehavioralStriped(
                m.source_, m.des_, m.tag_id_, meta.packets, meta.stripes,
                behavioral.port_rate, behavioral.link_rate, latency);
            delay += estimate.bulk_service_cycles;
            g_d2d_behavioral_stats.data_flows++;
            g_d2d_behavioral_stats.logical_data_packets += meta.packets;
            g_d2d_behavioral_stats.service_cycles +=
                estimate.bulk_service_cycles;
        }
        if (delay > std::numeric_limits<long>::max() - cyc)
            throw std::runtime_error("V4 Behavioral ready-cycle overflow");
        behavioral_data_events_.emplace(cyc + (long)delay, payload);
        if (m.subflow_ == 0)
            g_d2d_behavioral_stats.fixed_cycles += latency;
        g_d2d_link_in_pkts++;
        CountType(g_d2d_link_in_by_type, payload);
        CountLink(link_idx, true, payload);
        ProbeData(g_d2d_data_in, payload, cyc);
    }
    if (in_ctrl_sent.read()) {
        sc_bv<256> payload = in_ctrl_channel.read();
        Msg m = DeserializeMsg(payload);
        if (m.msg_type_ != REQUEST && m.msg_type_ != ACK)
            throw std::runtime_error(
                "V4 Behavioral control channel accepts only REQUEST/ACK");
        if (latency > std::numeric_limits<long>::max() - cyc)
            throw std::runtime_error(
                "V4 Behavioral control ready-cycle overflow");
        behavioral_ctrl_events_.emplace(cyc + latency, payload);
        if (m.subflow_ == 0)
            g_d2d_behavioral_stats.fixed_cycles += latency;
        g_d2d_link_in_pkts++;
        CountType(g_d2d_link_in_by_type, payload);
        CountLink(link_idx, true, payload);
    }

    auto data = behavioral_data_events_.begin();
    if (data != behavioral_data_events_.end() && data->first <= cyc &&
        out_avail.read()) {
        out_channel.write(data->second);
        out_sent.write(true);
        CountType(g_d2d_link_out_by_type, data->second);
        CountLink(link_idx, false, data->second);
        ProbeData(g_d2d_data_out, data->second, cyc);
        behavioral_data_events_.erase(data);
        g_d2d_link_out_pkts++;
        g_protocol_progress++;
    } else {
        out_sent.write(false);
    }
    auto ctrl = behavioral_ctrl_events_.begin();
    if (ctrl != behavioral_ctrl_events_.end() && ctrl->first <= cyc &&
        out_ctrl_avail.read()) {
        out_ctrl_channel.write(ctrl->second);
        out_ctrl_sent.write(true);
        CountType(g_d2d_link_out_by_type, ctrl->second);
        CountLink(link_idx, false, ctrl->second);
        behavioral_ctrl_events_.erase(ctrl);
        g_d2d_link_out_pkts++;
        g_protocol_progress++;
    } else {
        out_ctrl_sent.write(false);
    }
}

// V3-b bounded 路径：有限 FIFO + token-bucket 速率 + 信用式 flow control。
// 关键顺序 **先交付、后接受**：本拍先出队队首（腾出空位），再接受上游包填入——使 depth>=BDP
// 时能维持每拍 1 包（否则「先判满再出队」会多一个 bubble、吞吐减半）。
// flow control：in_avail 公布空位（背压信号）；上游持信用 = 下游空位，交付一包归还一信用，
// 据此不过量发送 → link 接受时必有空位、永不丢包。占用峰值不越 data_depth。
void D2DLinkUnit::forward_bounded(long cyc) {
    tokens += bound.rate.num;

    // 0. 信用回还：pop 时排入 due=cyc+latency（回程 L 拍）；本拍 pulse 一个已到期的（每拍至多 1，
    //    与交付率一致）。clocked link 的正向/回程各至少一个服务拍，完整 RTT 统一由
    //    D2DCreditRttCycles() 定义（另含上游 tx、credit-rx 两个接口拍）。
    bool ret = false;
    if (!data_credit_due_.empty() && data_credit_due_.front() <= cyc) {
        ret = true;
        data_credit_due_.pop_front();
    }
    data_credit_return.write(ret);
    data_credit_active_ = ret;
    bool cret = false;
    if (!ctrl_credit_due_.empty() && ctrl_credit_due_.front() <= cyc) {
        cret = true;
        ctrl_credit_due_.pop_front();
    }
    ctrl_credit_return.write(cret);
    ctrl_credit_active_ = cret;

    // 1a. 交付 DATA（先出队腾空位）：成熟 + 下游 ready + token 足
    bool data_mature = !fifo_.empty() && fifo_.front().first <= cyc;
    bool has_token = tokens >= bound.rate.den;
    bool group_request = data_mature && out_avail.read() && has_token;
    bool group_ok = V5LinkGroupGrant(link_idx, cyc, group_request);
    if (group_request && group_ok) {
        out_channel.write(fifo_.front().second);
        out_sent.write(true);
        CountType(g_d2d_link_out_by_type, fifo_.front().second);
        CountLink(link_idx, false, fifo_.front().second);
        ProbeData(g_d2d_data_out, fifo_.front().second, cyc);
        fifo_.pop_front();
        data_credit_due_.push_back(cyc + latency); // 空位释放 → L 拍后回还信用
        g_d2d_link_out_pkts++;
        g_protocol_progress++;
        tokens -= bound.rate.den;
    } else {
        out_sent.write(false);
        if (data_mature) {
            if (!out_avail.read())
                downstream_stall++;
            else if (!has_token)
                rate_stall++;
            else if (!group_ok)
                link_group_stall++;
        }
    }
    // token 在发之后 cap 至 den（限突发 <=1 包；不在发之前 cap，否则会丢失 num>1 的速率）
    if (tokens > bound.rate.den)
        tokens = bound.rate.den;

    // 1b. 交付 CTRL（独立子通道，不受 data token 限制——控制不占 DATA 带宽）
    bool ctrl_mature = !cfifo_.empty() && cfifo_.front().first <= cyc;
    if (ctrl_mature && out_ctrl_avail.read()) {
        out_ctrl_channel.write(cfifo_.front().second);
        out_ctrl_sent.write(true);
        CountType(g_d2d_link_out_by_type, cfifo_.front().second);
        CountLink(link_idx, false, cfifo_.front().second);
        cfifo_.pop_front();
        ctrl_credit_due_.push_back(cyc + latency);
        g_d2d_link_out_pkts++;
        g_protocol_progress++;
    } else {
        out_ctrl_sent.write(false);
        if (ctrl_mature && !out_ctrl_avail.read())
            ctrl_downstream_stall++;
    }

    // 2. 接受上游 DATA。**流控唯一真源 = 信用**（data_credit_return 接口）：上游持 credit=depth，
    //    credit>0 才发，收到回还 +1，故永不过量、link 接受时必有空位（occ<depth）、**永不溢出/丢包**。
    //    in_avail 仅作**诊断镜像**（step 4 公布 occ<depth 供观测），**不参与流控决策**——避免两套
    //    不同步的流控真源。（单信道 1 拍 delta 使纯组合 valid/ready 退化成 2 拍环，故不用它做流控。）
    bool has_room = (int)fifo_.size() < bound.data_depth;
    if (in_sent.read()) {
        if (has_room) {
            sc_bv<256> payload = in_channel.read();
            fifo_.push_back({cyc + latency, payload});
            g_d2d_link_in_pkts++;
            CountType(g_d2d_link_in_by_type, payload);
            CountLink(link_idx, true, payload);
            ProbeData(g_d2d_data_in, payload, cyc);
        } else {
            upstream_blocked++; // 想发但已满（下方 in_avail 已公布 false）
        }
    }
    // 2b. 接受上游 CTRL（同上信用式）
    bool has_croom = (int)cfifo_.size() < bound.ctrl_depth;
    if (in_ctrl_sent.read()) {
        if (has_croom) {
            cfifo_.push_back({cyc + latency, in_ctrl_channel.read()});
            g_d2d_link_in_pkts++;
            CountType(g_d2d_link_in_by_type, in_ctrl_channel.read());
            CountLink(link_idx, true, in_ctrl_channel.read());
        } else {
            ctrl_upstream_blocked++;
        }
    }

    // 3. 占用峰值（交付+接受之后的真实占用）
    if ((long)fifo_.size() > occ_max)
        occ_max = (long)fifo_.size();
    if ((long)cfifo_.size() > occ_ctrl_max)
        occ_ctrl_max = (long)cfifo_.size();

    // 4. 公布下一拍 ready = 当前（交付+接受后）是否仍有容量；记满状态
    bool room = (int)fifo_.size() < bound.data_depth;
    bool croom = (int)cfifo_.size() < bound.ctrl_depth;
    in_avail.write(room);
    in_ctrl_avail.write(croom);
    if (!room)
        full_cycles++;
    if (!croom)
        ctrl_full_cycles++;
}

// V3-d production bounded pipeline. DATA path:
//   router -> whole-flow SAF stage -> port/link token service -> finite inflight+latency
//          -> finite RX stage -> remote router.
// REQUEST is observed on the independent CTRL path before its DATA, providing F for packet-count checks.
// All directed links on the deterministic route were atomically reserved before REQUEST injection; therefore
// a flow never partially enters the source/intermediate mesh without enough SAF capacity on every hop.
void D2DLinkUnit::forward_bounded_saf(long cyc) {
    port_tokens += bound.port_rate.num;
    tokens += bound.rate.num;
    // Production DATA/CTRL return credits by toggling event bits. Do not force them low each cycle:
    // consecutive returns must remain distinguishable to the router.
    data_credit_active_ = false;
    ctrl_credit_active_ = false;

    // 1. RX -> remote router. A full RX stage backpressures the mature inflight head.
    if (!rx_fifo_.empty() && out_avail.read()) {
        out_channel.write(rx_fifo_.front());
        out_sent.write(true);
        rx_fifo_.pop_front();
        g_protocol_progress++;
    } else {
        out_sent.write(false);
        if (!rx_fifo_.empty() && !out_avail.read())
            downstream_stall++;
    }

    // 2. Physical link arrival -> finite RX stage.
    bool mature = !fifo_.empty() && fifo_.front().first <= cyc;
    if (mature && (int)rx_fifo_.size() < bound.rx_depth) {
        sc_bv<256> payload = fifo_.front().second;
        fifo_.pop_front();
        rx_fifo_.push_back(payload);
        CountType(g_d2d_link_out_by_type, payload);
        CountLink(link_idx, false, payload);
        ProbeData(g_d2d_data_out, payload, cyc);
        g_d2d_link_out_pkts++;
        g_protocol_progress++;
    } else if (mature) {
        rx_backpressure_stall++;
    }

    // 3. A complete SAF flow may enter the finite physical link; incomplete flows never leave SAF.
    bool ready = !saf_ready_.empty();
    bool port_ok = port_tokens >= bound.port_rate.den;
    bool link_ok = tokens >= bound.rate.den;
    bool inflight_room = (int)fifo_.size() < bound.data_depth;
    bool group_request = ready && port_ok && link_ok && inflight_room;
    bool group_ok = V5LinkGroupGrant(link_idx, cyc, group_request);
    if (group_request && group_ok) {
        FlowKey key = saf_ready_.front();
        auto fit = saf_flows_.find(key);
        if (fit == saf_flows_.end() || !fit->second.complete || fit->second.packets.empty())
            throw std::runtime_error("bounded SAF ready queue/accounting mismatch");
        sc_bv<256> payload = fit->second.packets.front();
        fit->second.packets.pop_front();
        fifo_.push_back({cyc + latency, payload});
        CountType(g_d2d_link_in_by_type, payload);
        CountLink(link_idx, true, payload);
        ProbeData(g_d2d_data_in, payload, cyc);
        g_d2d_link_in_pkts++;
        port_tokens -= bound.port_rate.den;
        tokens -= bound.rate.den;
        // SAF stage freed exactly one physical slot; return one DATA credit to the upstream router.
        data_credit_toggle_ = !data_credit_toggle_;
        data_credit_return.write(data_credit_toggle_);
        if (fit->second.packets.empty()) {
            int expected = fit->second.expected;
            ReleaseWholeFlowSafLink(link_idx, key, expected);
            saf_flows_.erase(fit);
            saf_expected_.erase(key);
            saf_ready_.pop_front();
        }
    } else if (ready) {
        if (!port_ok)
            port_rate_stall++;
        else if (!link_ok)
            link_rate_stall++;
        else if (!inflight_room)
            rate_stall++; // capacity/credit stall, distinct from programmable-rate stalls above
        else if (!group_ok)
            link_group_stall++;
    }
    if (port_tokens > bound.port_rate.den)
        port_tokens = bound.port_rate.den;
    if (tokens > bound.rate.den)
        tokens = bound.rate.den;

    // 4. Independent bounded CTRL channel (not DATA-rate limited). REQUEST records F for this link's SAF stage.
    bool ctrl_mature = !cfifo_.empty() && cfifo_.front().first <= cyc;
    if (ctrl_mature && out_ctrl_avail.read()) {
        sc_bv<256> payload = cfifo_.front().second;
        cfifo_.pop_front();
        out_ctrl_channel.write(payload);
        out_ctrl_sent.write(true);
        ctrl_credit_toggle_ = !ctrl_credit_toggle_;
        ctrl_credit_return.write(ctrl_credit_toggle_);
        CountType(g_d2d_link_out_by_type, payload);
        CountLink(link_idx, false, payload);
        g_d2d_link_out_pkts++;
        g_protocol_progress++;
    } else {
        out_ctrl_sent.write(false);
        if (ctrl_mature && !out_ctrl_avail.read())
            ctrl_downstream_stall++;
    }
    if (in_ctrl_sent.read()) {
        if ((int)cfifo_.size() >= bound.ctrl_depth) {
            ctrl_upstream_blocked++;
            throw std::runtime_error("bounded CTRL producer exceeded advertised capacity");
        }
        sc_bv<256> payload = in_ctrl_channel.read();
        Msg m = DeserializeMsg(payload);
        if (m.msg_type_ == REQUEST) {
            FlowKey key{m.source_, m.tag_id_, m.subflow_};
            if (m.flow_packets_ <= 0 || m.flow_packets_ > bound.saf_depth)
                throw std::runtime_error("bounded SAF REQUEST has invalid flow_packets");
            if (saf_expected_.count(key) || saf_flows_.count(key))
                throw std::runtime_error("bounded SAF duplicate REQUEST on active link");
            saf_expected_[key] = m.flow_packets_;
        }
        cfifo_.push_back({cyc + latency, payload});
        CountType(g_d2d_link_in_by_type, payload);
        CountLink(link_idx, true, payload);
        g_d2d_link_in_pkts++;
    }

    // 5. Router -> SAF capture. Path reservation grants ownership; the SAF-slot credit prevents the
    // registered ready delta from oversending. Any overflow is therefore a hard contract bug.
    if (in_sent.read()) {
        long occupied = 0;
        for (const auto &kv : saf_flows_)
            occupied += (long)kv.second.packets.size();
        if (occupied >= bound.saf_depth) {
            upstream_blocked++;
            throw std::runtime_error("bounded SAF stage overflow despite path reservation");
        }
        sc_bv<256> payload = in_channel.read();
        Msg m = DeserializeMsg(payload);
        if (m.msg_type_ != DATA)
            throw std::runtime_error("bounded SAF DATA channel received a non-DATA packet");
        FlowKey key{m.source_, m.tag_id_, m.subflow_};
        auto eit = saf_expected_.find(key);
        if (eit == saf_expected_.end())
            throw std::runtime_error("bounded SAF DATA arrived before matching REQUEST");
        SafFlowBuffer &flow = saf_flows_[key];
        if (flow.expected == 0)
            flow.expected = eit->second;
        if (flow.complete)
            throw std::runtime_error("bounded SAF received DATA after tail");
        flow.packets.push_back(payload);
        if ((int)flow.packets.size() > flow.expected)
            throw std::runtime_error("bounded SAF flow exceeded declared packet count");
        if (m.is_end_) {
            if ((int)flow.packets.size() != flow.expected || m.seq_id_ != flow.expected)
                throw std::runtime_error("bounded SAF DATA tail does not match REQUEST flow_packets");
            flow.complete = true;
            saf_ready_.push_back(key);
        }
    }

    // 6. Occupancy/backpressure diagnostics after all state transitions.
    long saf_occ = 0;
    for (const auto &kv : saf_flows_)
        saf_occ += (long)kv.second.packets.size();
    if (saf_occ > saf_occ_max) saf_occ_max = saf_occ;
    if ((long)fifo_.size() > inflight_occ_max) inflight_occ_max = (long)fifo_.size();
    if ((long)rx_fifo_.size() > rx_occ_max) rx_occ_max = (long)rx_fifo_.size();
    bool saf_room = saf_occ < bound.saf_depth;
    bool ctrl_room = (int)cfifo_.size() < bound.ctrl_depth;
    in_avail.write(saf_room);
    in_ctrl_avail.write(ctrl_room);
    if (!saf_room) saf_full_cycles++;
    if ((int)fifo_.size() >= bound.data_depth) inflight_full_cycles++;
    if ((int)rx_fifo_.size() >= bound.rx_depth) rx_full_cycles++;
    if (!ctrl_room) ctrl_full_cycles++;
}
