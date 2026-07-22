#include "die/d2d_link.h"
#include "die/port.h"
#include "utils/msg_utils.h"

namespace {
void CountType(long (&counts)[MSG_TYPE_NUM], const sc_bv<256> &payload) {
    int type = static_cast<int>(DeserializeMsg(payload).msg_type_);
    if (type >= 0 && type < MSG_TYPE_NUM)
        counts[type]++;
}
} // namespace

D2DLinkUnit::D2DLinkUnit(const sc_module_name &n, int latency_)
    : sc_module(n), latency(latency_) {
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
            fifo_.push_back({cyc + latency, in_channel.read()});
            g_d2d_link_in_pkts++;
            CountType(g_d2d_link_in_by_type, in_channel.read());
        }
        if (in_ctrl_sent.read()) {
            cfifo_.push_back({cyc + latency, in_ctrl_channel.read()});
            g_d2d_link_in_pkts++;
            CountType(g_d2d_link_in_by_type, in_ctrl_channel.read());
        }

        // 交付数据：队首成熟且下游 ready
        if (!fifo_.empty() && fifo_.front().first <= cyc && out_avail.read()) {
            out_channel.write(fifo_.front().second);
            out_sent.write(true);
            CountType(g_d2d_link_out_by_type, fifo_.front().second);
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
            cfifo_.pop_front();
            g_d2d_link_out_pkts++;
        } else {
            out_ctrl_sent.write(false);
        }
    }
}
