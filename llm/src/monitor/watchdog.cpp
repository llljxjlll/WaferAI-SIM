#include "monitor/watchdog.h"
#include "defs/const.h"
#include "defs/spec.h"
#include "die/d2d_link.h"
#include "die/port.h"
#include "macros/macros.h"
#include "router/router.h"
#include "utils/msg_utils.h"
#include "utils/print_utils.h"
#include <functional>
#include <string>
#include <vector>

long g_protocol_progress = 0;
bool g_protocol_stall_detected = false;
long g_protocol_stall_cycle = -1;
long g_protocol_last_progress = 0;
// 默认阈值：远大于任何合法的「计算中无包移动」间隔（现有用例总时长约 1.5e4 cycle 量级，
// 其中最长的无进展间隔远小于此值），同时又能在合理时间内抓到真正的协议环。
long g_protocol_watchdog_cycles = 20000;

void ResetProtocolWatchdog() {
    g_protocol_progress = 0;
    g_protocol_stall_detected = false;
    g_protocol_stall_cycle = -1;
    g_protocol_last_progress = 0;
}

namespace {
const char *PhaseName(int t) {
    switch (t) {
    case CONFIG:
        return "CONFIG";
    case DATA:
        return "DATA";
    case REQUEST:
        return "REQUEST";
    case ACK:
        return "ACK";
    case DONE:
        return "DONE";
    case S_DATA:
        return "S_DATA";
    case P_DATA:
        return "P_DATA";
    default:
        return "?";
    }
}

const char *DIRN[] = {"W", "E", "N", "S", "C"};

// 遍历 SystemC 层级收集某类型模块
template <typename T> void Collect(const std::vector<sc_object *> &objs,
                                   std::vector<T *> &out) {
    for (auto *o : objs) {
        if (auto *p = dynamic_cast<T *>(o))
            out.push_back(p);
        Collect<T>(o->get_child_objects(), out);
    }
}

// 打印一条队首消息的等待身份：(source, tag, dest, phase) + 阻塞原因
void DumpHead(int rid, const char *chan, int dir, const sc_bv<256> &payload,
              const std::string &reason) {
    Msg m = DeserializeMsg(payload);
    LOG_WARN(SYSTEM) << "[PROTO_WAIT]   router=" << rid << " " << chan << "["
                     << ((dir >= 0 && dir < 5) ? DIRN[dir] : "?")
                     << "] source=" << m.source_ << " tag=" << m.tag_id_
                     << " dest=" << m.des_ << " phase=" << PhaseName(m.msg_type_)
                     << " seq=" << m.seq_id_ << " wait_reason=" << reason;
}
} // namespace

void DumpProtocolWaitState(long cycle) {
    std::vector<RouterUnit *> routers;
    std::vector<D2DLinkUnit *> links;
    Collect<RouterUnit>(sc_get_top_level_objects(), routers);
    Collect<D2DLinkUnit>(sc_get_top_level_objects(), links);

    long rres = 0, lres = 0;
    for (auto *r : routers)
        rres += r->residual();
    for (auto *l : links)
        lres += l->residual();

    LOG_WARN(SYSTEM) << "[PROTO_WAIT] protocol_wait_cycle=" << cycle
                     << " last_progress_cycle=" << g_protocol_last_progress
                     << " stalled_for=" << (cycle - g_protocol_last_progress)
                     << " router_residual=" << rres
                     << " d2d_link_residual=" << lres
                     << " progress_events=" << g_protocol_progress;

    for (auto *r : routers) {
        // 被持有的 output lock：tag 即接收端聚合槽，是 wait-for 关系的关键标识
        for (int d = 0; d < DIRECTIONS; d++) {
            if (r->output_lock[d] >= 0 || r->output_lock_ref[d] > 0)
                LOG_WARN(SYSTEM)
                    << "[PROTO_WAIT]   router=" << r->rid << " output_lock["
                    << DIRN[d] << "] tag=" << r->output_lock[d]
                    << " ref=" << r->output_lock_ref[d];
        }
        // 滞留在输入侧的包：它们正等待某个输出方向可用
        for (int d = 0; d < DIRECTIONS; d++) {
            if (!r->buffer_i[d].empty())
                DumpHead(r->rid, "data_in", d, r->buffer_i[d].front(),
                         "held_in_input_buffer");
            if (!r->ctrl_buffer_i[d].empty())
                DumpHead(r->rid, "ctrl_in", d, r->ctrl_buffer_i[d].front(),
                         "held_in_input_buffer");
            if (!r->buffer_o[d].empty())
                DumpHead(r->rid, "data_out", d, r->buffer_o[d].front(),
                         "waiting_downstream_avail_or_lock");
            if (!r->ctrl_buffer_o[d].empty())
                DumpHead(r->rid, "ctrl_out", d, r->ctrl_buffer_o[d].front(),
                         "waiting_downstream_avail_or_lock");
        }
    }
    // 无任何滞留包却仍无进展 ⇒ 等待发生在**原语层**（rendezvous 环：双方都在等对方先发/先收），
    // 而非网络层。这一区分正是外部 wall-clock 超时给不出的信息。
    if (rres == 0 && lres == 0)
        LOG_WARN(SYSTEM)
            << "[PROTO_WAIT]   no in-flight packets and no held locks: "
               "wait is at the primitive/rendezvous layer (endpoints waiting "
               "on each other), not in the network";
}

ProtocolWatchdog::ProtocolWatchdog(const sc_module_name &n) : sc_module(n) {
    SC_THREAD(monitor);
}

void ProtocolWatchdog::monitor() {
    long cyc = 0;
    long seen = g_protocol_progress;
    g_protocol_last_progress = 0;
    while (true) {
        wait(CYCLE, SC_NS);
        cyc++;
        if (g_protocol_progress != seen) {
            seen = g_protocol_progress;
            g_protocol_last_progress = cyc;
            continue;
        }
        if (g_protocol_watchdog_cycles > 0 &&
            cyc - g_protocol_last_progress > g_protocol_watchdog_cycles) {
            g_protocol_stall_detected = true;
            g_protocol_stall_cycle = cyc;
            LOG_WARN(SYSTEM)
                << "[PROTO_WAIT] protocol progress watchdog fired: no REQUEST/"
                   "ACK/DATA/DONE progress for "
                << (cyc - g_protocol_last_progress)
                << " cycles while the simulation is still running";
            DumpProtocolWaitState(cyc);
            sc_stop(); // 主动结束仿真；npusim 据 g_protocol_stall_detected 非零退出
            return;
        }
    }
}
