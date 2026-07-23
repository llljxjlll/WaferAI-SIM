#include "die/d2d_link.h"
#include "die/port.h"
#include "monitor/watchdog.h"
#include "utils/msg_utils.h"
#include <stdexcept>

namespace {
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
                         D2DLinkBound bound_)
    : sc_module(n), latency(latency_), link_idx(link_idx_), bound(bound_) {
    // V3-b：构造期硬校验 bounded 参数（生产解析器已保证，但 standalone/其它调用者可能绕过）。
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
        if (bound.enabled) {
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

        if (!fifo_.empty() && fifo_.front().first <= cyc && out_avail.read()) {
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
    if (data_mature && out_avail.read() && has_token) {
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
