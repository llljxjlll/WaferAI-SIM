#include "die/d2d_link.h"
#include "die/port.h"
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

D2DLinkUnit::D2DLinkUnit(const sc_module_name &n, int latency_, int link_idx_)
    : sc_module(n), latency(latency_), link_idx(link_idx_) {
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

        in_avail.write(true);
        in_ctrl_avail.write(true);

        // 采集（capture 在交付前 → latency==0 可当拍交付）
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

        // 交付数据：队首成熟且下游 ready
        if (!fifo_.empty() && fifo_.front().first <= cyc && out_avail.read()) {
            out_channel.write(fifo_.front().second);
            out_sent.write(true);
            CountType(g_d2d_link_out_by_type, fifo_.front().second);
            CountLink(link_idx, false, fifo_.front().second);
            ProbeData(g_d2d_data_out, fifo_.front().second, cyc);
            fifo_.pop_front();
            g_d2d_link_out_pkts++;
        } else {
            out_sent.write(false);
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
        } else {
            out_ctrl_sent.write(false);
        }
    }
}
