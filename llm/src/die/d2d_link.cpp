#include "die/d2d_link.h"
#include "die/port.h"
#include "monitor/watchdog.h"
#include "utils/msg_utils.h"

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

        // in_avail：functional 恒 true（不施背压）；bounded 则 = FIFO 尚有容量。
        bool data_room = !bound.enabled || (int)fifo_.size() < bound.data_depth;
        bool ctrl_room = !bound.enabled || (int)cfifo_.size() < bound.ctrl_depth;
        in_avail.write(data_room);
        in_ctrl_avail.write(ctrl_room);
        if (bound.enabled && !data_room)
            full_cycles++; // 对上游施背压这一拍

        // token bucket（bounded 数据交付速率）：每拍累加 rate.num，发一包扣 rate.den；
        // 桶深 = rate.den（至多攒 1 个包的信用，避免空闲后突发超过 1 包/cycle——单信号也做不到）。
        if (bound.enabled) {
            tokens += bound.rate.num;
            if (tokens > bound.rate.den)
                tokens = bound.rate.den;
        }

        // 采集（capture 在交付前 → latency==0 可当拍交付）。bounded 时只在有容量时收，
        // 保证占用不越过 depth（正确的上游会遵守 in_avail，此处兼作硬护栏）。
        if (in_sent.read() && data_room) {
            sc_bv<256> payload = in_channel.read();
            fifo_.push_back({cyc + latency, payload});
            data_captured++;
            g_d2d_link_in_pkts++;
            CountType(g_d2d_link_in_by_type, payload);
            CountLink(link_idx, true, payload);
            ProbeData(g_d2d_data_in, payload, cyc);
        }
        if (in_ctrl_sent.read() && ctrl_room) {
            cfifo_.push_back({cyc + latency, in_ctrl_channel.read()});
            g_d2d_link_in_pkts++;
            CountType(g_d2d_link_in_by_type, in_ctrl_channel.read());
            CountLink(link_idx, true, in_ctrl_channel.read());
        }
        if (bound.enabled) { // 占用峰值在 capture 之后测（交付前的瞬时峰值）
            if ((long)fifo_.size() > occ_max)
                occ_max = (long)fifo_.size();
            if ((long)cfifo_.size() > occ_ctrl_max)
                occ_ctrl_max = (long)cfifo_.size();
        }

        // 交付数据：队首成熟。functional 只看下游 ready；bounded 另需 token。
        bool data_mature = !fifo_.empty() && fifo_.front().first <= cyc;
        bool has_token = !bound.enabled || tokens >= bound.rate.den;
        if (data_mature && out_avail.read() && has_token) {
            out_channel.write(fifo_.front().second);
            out_sent.write(true);
            CountType(g_d2d_link_out_by_type, fifo_.front().second);
            CountLink(link_idx, false, fifo_.front().second);
            ProbeData(g_d2d_data_out, fifo_.front().second, cyc);
            fifo_.pop_front();
            g_d2d_link_out_pkts++;
            g_protocol_progress++;
            if (bound.enabled)
                tokens -= bound.rate.den;
        } else {
            out_sent.write(false);
            if (bound.enabled && data_mature) {
                if (!out_avail.read())
                    ds_stall++; // 下游不 ready
                else if (!has_token)
                    rate_stall++; // token 不足（速率限制）
            }
        }
        // 交付控制
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
