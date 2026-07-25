#include "systemc.h"
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <typeinfo>

#include "defs/const.h"
#include "defs/global.h"
#include "defs/spec.h"
#include "die/port.h"
#include "dte/dte_streaming.h"
#include "utils/router_utils.h"
#include "link/nb_global_memif_v2.h"
#include "memory/dram/GPUNB_DcacheIF.h"
#include "memory/gpu/GPU_L1L2_Cache.h"
#include "memory/sram/Mem_access_unit.h"
#include "prims/base.h"
#include "prims/comp_prims.h"
#include "prims/moe_prims.h"
#include "prims/norm_prims.h"
#include "prims/pd_prims.h"
#include "trace/Event_engine.h"
#include "utils/memory_utils.h"
#include "utils/msg_utils.h"
#include "utils/prim_utils.h"
#include "utils/print_utils.h"
#include "utils/system_utils.h"
#include "workercore/workercore.h"

namespace {
int StripePackets(int total, int stripes, int subflow) {
    if (stripes != 1 && stripes != 2 && stripes != 4)
        throw std::runtime_error("V5 stripe count must be 1, 2, or 4");
    if (total < stripes || subflow < 0 || subflow >= stripes)
        throw std::runtime_error(
            "V5 striping requires at least one packet per subflow");
    return total / stripes + (subflow < total % stripes ? 1 : 0);
}

void InitDataStripeState(Send_prim *prim, int source) {
    const int k = prim->stripe_count;
    prim->stripe_packets.assign(k, 0);
    prim->stripe_sent.assign(k, 0);
    prim->stripe_exit_ports.assign(k, -1);
    for (int s = 0; s < k; ++s) {
        prim->stripe_packets[s] = StripePackets(prim->max_packet, k, s);
        prim->stripe_exit_ports[s] =
            SelectCoreMsgExit(source, prim->des_id, prim->tag_id, s);
    }
    prim->d2d_exit_port = prim->stripe_exit_ports[0];
    prim->d2d_exit_selected = true;
    if (g_d2d_cfg.backend == BACKEND_BEHAVIORAL &&
        DieOfGlobal(source) != DieOfGlobal(prim->des_id))
        RegisterD2DBehavioralFlow(source, prim->des_id, prim->tag_id,
                                  prim->max_packet, k);
}

void AttachRequestFlowPackets(Msg &m, const Send_prim *prim, int subflow) {
    if (SYSTEM_MODE == SIM_DATAFLOW &&
        (prim->max_packet <= 0 ||
         (unsigned)prim->max_packet > M_D_FLOW_PACKETS_MAX))
        throw std::runtime_error(
            "dataflow REQUEST missing a valid flow_packets count");
    m.subflow_ = subflow;
    m.flow_packets_ = StripePackets(prim->max_packet, prim->stripe_count,
                                    subflow);
    AttachRequestDtePayload(m, *prim, SPEC_USE_BEHA_DTE);
}

uint64_t CurrentDteNanoseconds() {
    return static_cast<uint64_t>(
        sc_time_stamp().value() / sc_time(1, SC_NS).value());
}

sc_time DteCycleTime(uint64_t cycle) {
    if (cycle > std::numeric_limits<uint64_t>::max() /
                    static_cast<uint64_t>(CYCLE))
        throw std::overflow_error("DTE V2b cycle-to-time conversion overflows");
    return sc_time(static_cast<double>(cycle) * CYCLE, SC_NS);
}

void WaitForDteTransmitStart(DteTransferContext &context) {
    if (context.state == DteTransferState::PENDING ||
        context.state == DteTransferState::LAUNCHING ||
        context.state == DteTransferState::BUS_WAIT)
        wait(context.transmit_started);
    if (context.state != DteTransferState::TRANSMITTING &&
        context.state != DteTransferState::COMPLETED)
        throw std::logic_error(
            "DTE V2b transfer did not reach the transmitting state");
}

void WaitUntilDteTime(const sc_time &target) {
    if (target > sc_time_stamp())
        wait(target - sc_time_stamp());
}

std::string DteStreamingFlowName(int source, int dest, int tag) {
    return "DTE_stream_" + std::to_string(source) + "_" +
           std::to_string(dest) + "_" + std::to_string(tag);
}

void TraceDteStreaming(Event_engine *event_engine, const char *stage,
                       const char *phase, int source, int dest, int tag,
                       uint64_t payload_bits,
                       sc_time relative_time = SC_ZERO_TIME) {
    if (event_engine == nullptr)
        return;
    std::string detail = std::string(stage) + " source=" +
                         std::to_string(source) + " dest=" +
                         std::to_string(dest) + " tag=" +
                         std::to_string(tag) + " bits=" +
                         std::to_string(payload_bits);
    event_engine->add_event(DteStreamingFlowName(source, dest, tag), stage,
                            phase, Trace_event_util(detail), relative_time);
}

uint64_t DteStreamingReadyBits(const Send_prim &prim, uint32_t dte_width,
                               bool behavioral_noc) {
    const uint64_t payload_bits = ComputeSendPayloadBits(prim);
    if (behavioral_noc)
        return std::min(payload_bits, static_cast<uint64_t>(dte_width));
    const uint64_t packet_index = static_cast<uint64_t>(prim.data_packet_id);
    if (packet_index == 0)
        throw std::logic_error("DTE V2b DATA packet index is zero");
    const uint64_t raw_packets =
        (static_cast<uint64_t>(prim.max_packet) - 1) * prim.packet_scale +
        prim.packets_in_last_group;
    const uint64_t ready_packets = std::min(
        raw_packets, packet_index * static_cast<uint64_t>(prim.packet_scale));
    if (ready_packets == raw_packets)
        return payload_bits;
    if (ready_packets > std::numeric_limits<uint64_t>::max() / M_D_DATA)
        throw std::overflow_error("DTE V2b source-ready bit count overflows");
    return ready_packets * M_D_DATA;
}

void ReserveStripedSafOnce(Send_prim *prim, int source) {
    if (prim->stripe_saf_reserved || g_d2d_cfg.mode != MODE_BOUNDED_SAF ||
        DieOfGlobal(source) == DieOfGlobal(prim->des_id))
        return;
    std::vector<int> counts;
    counts.reserve(prim->stripe_count);
    for (int s = 0; s < prim->stripe_count; ++s)
        counts.push_back(StripePackets(prim->max_packet, prim->stripe_count, s));
    ReserveStripedWholeFlowSafPaths(source, prim->des_id, prim->tag_id,
                                    counts);
    prim->stripe_saf_reserved = true;
}
} // namespace

void WorkerCoreExecutor::send_logic() {
    while (true) {
        Send_prim *prim = (Send_prim *)prim_queue.front();

        prim->data_packet_id = 0;
        prim->next_subflow = 0;
        prim->d2d_exit_selected = false;
        prim->stripe_packets.clear();
        prim->stripe_sent.clear();
        prim->stripe_exit_ports.clear();
        prim->stripe_saf_reserved = false;
        if (prim->type == SEND_DATA)
            InitDataStripeState(prim, cid);
        bool job_done = false; // 结束内圈循环的标志

        LOG_INFO(PRIM) << "Core " << cid << " start send primitive "
                       << GetEnumSendType(prim->type);
        LOG_DEBUG(PRIM) << "destination " << prim->des_id << ", tag "
                        << prim->tag_id << ", max packet " << prim->max_packet;

        DteTransferContext *stream_source_xfer = nullptr;
        uint64_t stream_source_first_ns = 0;
        bool stream_source_started = false;
        if (SPEC_USE_BEHA_DTE && prim->type == SEND_DATA) {
            dte->WaitForCredit();
            DteTransferContext &xfer = dte->Issue(
                ComputeSendPayloadBits(*prim), DteDir::SPM_TO_REMOTE);
            if (SPEC_DTE_STREAMING) {
                stream_source_xfer = &xfer;
                TraceDteStreaming(event_engine, "DTE_stream_source_fill", "B",
                                  cid, prim->des_id, prim->tag_id,
                                  xfer.payload_bits);
                WaitForDteTransmitStart(xfer);
            } else {
                wait(xfer.done);
                if (!dte->Release(xfer.xfer_id))
                    throw std::logic_error(
                        "source DTE completed context could not be released");
            }
        }

        while (true) {
            bool need_long_wait = false;
            int roofline_packets = 1;

            if (atomic_helper_lock(sc_time_stamp(), 0))
                ev_send_helper.notify(0, SC_NS);

            // SEND_DATA, SEND_ACK, SEND_REQ
            if (prim->type == SEND_DATA) {
                while (job_done != true) {
                    // [发送方] 正常发送数据
                    int subflow = 0, seq = 0;
                    bool logical_last = false;
                    if (SPEC_USE_BEHA_NOC) {
                        subflow = prim->next_subflow++;
                        seq = 1; // Behavioral 每 subflow 仅发一个代表包；首包也必须是 seq=1
                        prim->data_packet_id++;
                        roofline_packets = prim->stripe_packets[subflow];
                        logical_last = prim->next_subflow == prim->stripe_count;
                    } else {
                        int global_seq = ++prim->data_packet_id;
                        subflow = (global_seq - 1) % prim->stripe_count;
                        seq = ++prim->stripe_sent[subflow];
                        logical_last = global_seq == prim->max_packet;
                    }
                    bool is_end_packet = SPEC_USE_BEHA_NOC ||
                                         seq == prim->stripe_packets[subflow];
                    int last_subflow =
                        (prim->max_packet - 1) % prim->stripe_count;
                    int length = (is_end_packet && subflow == last_subflow)
                                     ? prim->end_length
                                     : M_D_DATA;

                    if (stream_source_xfer != nullptr) {
                        const uint64_t ready_bits = DteStreamingReadyBits(
                            *prim, dte->config().bit_width_bits,
                            SPEC_USE_BEHA_NOC);
                        const sc_time ready_time =
                            stream_source_xfer->transmit_start_time +
                            DteCycleTime(CeilDivU64(
                                ready_bits, dte->config().bit_width_bits));
                        WaitUntilDteTime(ready_time);
                    }

                    TaskCoreContext context = generate_context(this);
                    (void)prim->taskCoreDefault(context);

                    if (!channel_avail_i.read())
                        wait(ev_channel_avail_i);

                    if (stream_source_xfer != nullptr &&
                        !stream_source_started) {
                        stream_source_started = true;
                        stream_source_first_ns = CurrentDteNanoseconds();
                        TraceDteStreaming(
                            event_engine, "DTE_stream_source_fill", "E", cid,
                            prim->des_id, prim->tag_id,
                            stream_source_xfer->payload_bits);
                        TraceDteStreaming(
                            event_engine, "DTE_stream_network", "B", cid,
                            prim->des_id, prim->tag_id,
                            stream_source_xfer->payload_bits);
                    }

                    Msg temp_msg = Msg(is_end_packet, MSG_TYPE::DATA, seq,
                                       prim->des_id, 0, prim->tag_id, length,
                                       sc_bv<128>(0x1));
                    temp_msg.roofline_packets_ = roofline_packets;
                    temp_msg.source_ = cid;
                    temp_msg.subflow_ = subflow;
                    temp_msg.exit_port_ = prim->stripe_exit_ports[subflow];
                    if (stream_source_xfer != nullptr) {
                        temp_msg.dte_stream_source_first_ns_ =
                            stream_source_first_ns;
                        temp_msg.dte_stream_source_done_ns_ =
                            static_cast<uint64_t>(
                                stream_source_xfer->scheduled_completion_time.value() /
                                sc_time(1, SC_NS).value());
                    }
                    send_buffer = temp_msg;

                    atomic_helper_lock(sc_time_stamp(), 3);
                    ev_send_helper.notify(0, SC_NS);

                    if (logical_last) {
                        LOG_DEBUG(NETWORK)
                            << "Core " << cid << " -> DATA -> " << prim->des_id;
                        LOG_DEBUG(NETWORK) << "max_packet " << prim->max_packet
                                           << ", stripe " << prim->stripe_count;
                        job_done = true;
                    }

                    need_long_wait = true;

                    if (SPEC_ROUTER_PIPE)
                        continue;
                    break;
                }
            }
            else if (prim->type == SEND_REQ) {
                // [发送方] 发送一个req包，发送完之后结束此原语，进入 RECV_ACK
                // REQUEST 是控制消息，使用控制信道
                if (!ctrl_channel_avail_i.read())
                    wait(ev_ctrl_channel_avail_i);

                ReserveStripedSafOnce(prim, cid);
                send_buffer =
                    Msg(MSG_TYPE::REQUEST, prim->des_id, prim->tag_id, cid);
                int subflow = prim->next_subflow++;
                AttachRequestFlowPackets(send_buffer, prim, subflow);
                PinControlMsgExit(send_buffer);

                send_helper_write = 3;
                ev_send_helper.notify(0, SC_NS);

                LOG_DEBUG(NETWORK) << "Core " << cid << " -> REQ["
                                   << subflow << "] -> " << prim->des_id;

                job_done = prim->next_subflow == prim->stripe_count;
            }

            else if (prim->type == SEND_DONE) {
                // [执行核]
                // 在计算图的汇节点执行完毕之后，给host发送一份DONE数据包，标志任务完成
                // DONE 是控制消息，使用控制信道
                if (!ctrl_channel_avail_i.read())
                    wait(ev_ctrl_channel_avail_i);

                send_buffer = Msg(MSG_TYPE::DONE, HostEndpointOfDie(DieOfGlobal(cid)), cid);

                if (SYSTEM_MODE == SIM_PD || SYSTEM_MODE == SIM_PDS) {
                    for (int i = 0; i < core_context->decode_done_.size();
                         i++) {
                        send_buffer.data_.range(i, i) =
                            sc_bv<1>(core_context->decode_done_[i]);
                    }
                }

                send_helper_write = 3;
                ev_send_helper.notify(0, SC_NS);

                LOG_DEBUG(NETWORK) << "Core " << cid << " -> DONE -> Host";

                job_done = true;
            }

            else {
                // unimplemented
                LOG_ERROR(logic.cpp)
                    << "Unimplemented SEND_PRIM type " << prim->type;
            }

            wait(roofline_packets * CYCLE, SC_NS);

            if (job_done) {
                if (stream_source_xfer != nullptr) {
                    if (stream_source_xfer->state !=
                        DteTransferState::COMPLETED)
                        wait(stream_source_xfer->done);
                    if (!dte->Release(stream_source_xfer->xfer_id))
                        throw std::logic_error(
                            "DTE V2b source context could not be released");
                    stream_source_xfer = nullptr;
                }
                LOG_INFO(PRIM) << "Core " << cid << " end send primitive "
                               << GetEnumSendType(prim->type);
                break;
            }
        }

        ev_block.notify(CYCLE, SC_NS);
        wait();
    }
}

void WorkerCoreExecutor::send_para_logic() {
    while (true) {
        // V2a：先为本批所有 SEND_DATA 背靠背 issue。context 由 DTEUnit 的
        // 地址稳定 list 持有；这里显式记录 prim -> (xfer_id, context) 映射。
        std::map<Send_prim *, std::pair<uint64_t, DteTransferContext *>>
            dte_batch;
        size_t batch_data_remaining = 0;
        std::queue<PrimBase *> pending = send_para_queue;
        while (!pending.empty()) {
            PrimBase *candidate = pending.front();
            pending.pop();
            if (typeid(*candidate) != typeid(Send_prim))
                continue;
            auto *send = static_cast<Send_prim *>(candidate);
            if (send->type != SEND_DATA)
                continue;
            ++batch_data_remaining;
            if (!SPEC_USE_BEHA_DTE)
                continue;
            if (dte_batch.count(send))
                throw std::logic_error(
                    "DTE V2a batch contains a duplicate SEND_DATA primitive");
            dte->WaitForCredit();
            DteTransferContext &xfer = dte->Issue(
                ComputeSendPayloadBits(*send), DteDir::SPM_TO_REMOTE);
            dte_batch.emplace(send, std::make_pair(xfer.xfer_id, &xfer));
        }

        auto consume_data_tail = [&]() {
            if (batch_data_remaining == 0)
                throw std::logic_error(
                    "parallel SEND_DATA tail accounting underflow");
            --batch_data_remaining;
            // 一个 compute-ready token 对本批所有 fan-out DATA 生效；仅在最后
            // 一条 DATA 尾包提交后消费它，避免第二条 SEND 永久等待。
            if (batch_data_remaining == 0)
                send_last_packet = false;
        };

        while (send_para_queue.size()) {
            PrimBase *prim = send_para_queue.front();
            send_para_queue.pop();

            if (typeid(*prim) == typeid(Send_prim)) {
                ((Send_prim *)prim)->data_packet_id = 0;
                auto prim_type =
                    GetEnumSendType(dynamic_cast<Send_prim *>(prim)->type);

                LOG_INFO(PRIM)
                    << "Core " << cid << " start parallel send primitive "
                    << prim_type;

                event_engine->add_event(
                    "Core " + ToHexString(cid), "Send_prim", "B",
                    Trace_event_util("Send_prim" + prim_type));
            } else if (typeid(*prim) == typeid(Recv_prim)) {
                auto prim_type =
                    GetEnumRecvType(dynamic_cast<Recv_prim *>(prim)->type);

                LOG_INFO(PRIM)
                    << "Core " << cid << " start parallel recv primitive "
                    << prim_type;

                event_engine->add_event(
                    "Core " + ToHexString(cid), "Recv_prim", "B",
                    Trace_event_util("Recv_prim" + prim_type));
            }

            if (SPEC_USE_BEHA_DTE &&
                typeid(*prim) == typeid(Send_prim) &&
                static_cast<Send_prim *>(prim)->type == SEND_DATA) {
                auto *send = static_cast<Send_prim *>(prim);
                auto mapping = dte_batch.find(send);
                if (mapping == dte_batch.end())
                    throw std::logic_error(
                        "DTE V2a SEND_DATA has no batch transfer context");
                const uint64_t xfer_id = mapping->second.first;
                DteTransferContext *context = mapping->second.second;
                if (context == nullptr || context->xfer_id != xfer_id)
                    throw std::logic_error(
                        "DTE V2a SEND_DATA transfer mapping is corrupt");
                // done 可能在控制握手期间已经通知；先查状态，避免错过
                // SC_ZERO_TIME 事件后再 wait 导致永久阻塞。
                if (context->state != DteTransferState::COMPLETED)
                    wait(context->done);
                if (context->state != DteTransferState::COMPLETED)
                    throw std::logic_error(
                        "DTE V2a SEND_DATA woke before transfer completion");
                if (!dte->Release(xfer_id))
                    throw std::logic_error(
                        "DTE V2a completed SEND_DATA context could not be released");
                dte_batch.erase(mapping);
            }

            bool job_done = false; // 结束内圈循环的标志

            while (true) {
                if (!SPEC_ROUTER_PIPE) {
                    if (atomic_helper_lock(sc_time_stamp(), 0))
                        ev_send_helper.notify(0, SC_NS);
                } else {
                    while (atomic_helper_lock(sc_time_stamp(), 0) == false) {
                        wait(CYCLE, SC_NS);
                    }

                    ev_send_helper.notify(0, SC_NS);
                }

                if (job_done)
                    break;

                // SEND_DATA, SEND_ACK, SEND_REQ
                if (typeid(*prim) == typeid(Send_prim) &&
                    ((Send_prim *)prim)->type == SEND_DATA) {
                    // [发送方] 正常发送数据，数据从DRAM中获取
                    Send_prim *s_prim = (Send_prim *)prim;
                    if (!s_prim->d2d_exit_selected) {
                        s_prim->d2d_exit_port =
                            SelectCoreMsgExit(cid, s_prim->des_id);
                        s_prim->d2d_exit_selected = true;
                    }

                    // atomic_helper_lock 其实是为了表示上锁
                    if (SPEC_ROUTER_PIPE) {
                        while (job_done == false) {
                            if (channel_avail_i.read() &&
                                atomic_helper_lock(sc_time_stamp(), 1)) {
                                ev_send_helper.notify(0, SC_NS);

                                s_prim->data_packet_id++;

                                bool is_end_packet = s_prim->data_packet_id ==
                                                     s_prim->max_packet;
                                int length = M_D_DATA;
                                if (is_end_packet) {
                                    length = s_prim->end_length;
                                    while (!send_last_packet) {
                                        atomic_helper_lock(sc_time_stamp(), 0,
                                                           true);
                                        wait(ev_send_last_packet);
                                        while (
                                            atomic_helper_lock(sc_time_stamp(),
                                                               0) == false) {
                                            wait(CYCLE, SC_NS);
                                        }
                                    }
                                    consume_data_tail();
                                }
                                send_buffer =
                                    Msg(s_prim->data_packet_id ==
                                            s_prim->max_packet,
                                        MSG_TYPE::DATA, s_prim->data_packet_id,
                                        s_prim->des_id, 0, s_prim->tag_id,
                                        length, sc_bv<128>(0x1));
                                send_buffer.source_ = cid; // 真实全局 source
                                send_buffer.exit_port_ = s_prim->d2d_exit_port;
                                int delay = 0;
                                TaskCoreContext context =
                                    generate_context(this);
                                delay = prim->taskCoreDefault(context);
                                atomic_helper_lock(sc_time_stamp(), 0, true);
                                ev_send_helper.notify(0, SC_NS);

                                if (s_prim->data_packet_id ==
                                    s_prim->max_packet) {
                                    job_done = true;

                                    LOG_DEBUG(NETWORK)
                                        << "Core " << cid << " -> DATA -> "
                                        << s_prim->des_id;
                                    LOG_DEBUG(NETWORK)
                                        << "max_packet " << s_prim->max_packet;
                                }
                            } else {
                                if (send_helper_write == 1) {
                                    send_helper_write = 0;
                                }

                                wait(CYCLE, SC_NS);
                                atomic_helper_lock(sc_time_stamp(), 0);
                            }
                        }
                    } else {
                        if (channel_avail_i.read() &&
                            atomic_helper_lock(sc_time_stamp(), 1)) {
                            ev_send_helper.notify(0, SC_NS);

                            s_prim->data_packet_id++;

                            bool is_end_packet =
                                s_prim->data_packet_id == s_prim->max_packet;
                            int length = M_D_DATA;
                            if (is_end_packet) {
                                length = s_prim->end_length;
                                while (!send_last_packet)
                                    wait(ev_send_last_packet);
                                consume_data_tail();
                            }
                            send_buffer = Msg(
                                s_prim->data_packet_id == s_prim->max_packet,
                                MSG_TYPE::DATA, s_prim->data_packet_id,
                                s_prim->des_id, 0, s_prim->tag_id, length,
                                sc_bv<128>(0x1));
                            send_buffer.source_ = cid; // 真实全局 source
                            send_buffer.exit_port_ = s_prim->d2d_exit_port;
                            int delay = 0;
                            TaskCoreContext context = generate_context(this);
                            delay = prim->taskCoreDefault(context);
                            atomic_helper_lock(sc_time_stamp(), 2);
                            ev_send_helper.notify(0, SC_NS);

                            if (s_prim->data_packet_id == s_prim->max_packet) {
                                job_done = true;

                                LOG_DEBUG(NETWORK)
                                    << "Core " << cid << " -> DATA -> "
                                    << s_prim->des_id;
                                LOG_DEBUG(NETWORK)
                                    << "max_packet " << s_prim->max_packet;
                            }
                        }
                    }
                }

                else if (typeid(*prim) == typeid(Send_prim) &&
                         ((Send_prim *)prim)->type == SEND_REQ) {
                    Send_prim *s_prim = (Send_prim *)prim;
                    // [发送方] 发送一个req包，发送完之后结束此原语，进入
                    // RECV_ACK
                    if (ctrl_channel_avail_i.read() &&
                        atomic_helper_lock(sc_time_stamp(), 3)) {
                        // 可以发送数据
                        send_buffer = Msg(MSG_TYPE::REQUEST, s_prim->des_id,
                                          s_prim->tag_id, cid);
                        if (s_prim->stripe_count != 1)
                            throw std::runtime_error(
                                "V5 striping is supported by the sequential "
                                "dataflow path, not parallel-send pipeline mode");
                        AttachRequestFlowPackets(send_buffer, s_prim, 0);
                        PinControlMsgExit(send_buffer);

                        ev_send_helper.notify(0, SC_NS);

                        LOG_DEBUG(NETWORK) << "Core " << cid << " -> REQ -> "
                                           << s_prim->des_id;

                        job_done = true;
                    }
                }

                else if (typeid(*prim) == typeid(Send_prim) &&
                         ((Send_prim *)prim)->type == SEND_DONE) {
                    Send_prim *s_prim = (Send_prim *)prim;
                    // [执行核]
                    // 在计算图的汇节点执行完毕之后，给host发送一份DONE数据包，标志任务完成
                    if (ctrl_channel_avail_i.read() &&
                        atomic_helper_lock(sc_time_stamp(), 3)) {
                        // 可以发送数据
                        send_buffer = Msg(MSG_TYPE::DONE, HostEndpointOfDie(DieOfGlobal(cid)), cid);

                        ev_send_helper.notify(0, SC_NS);

                        LOG_DEBUG(NETWORK)
                            << "Core " << cid << " -> DONE -> Host";

                        job_done = true;
                    }
                }

                else if (typeid(*prim) == typeid(Recv_prim) &&
                         ((Recv_prim *)prim)->type == RECV_ACK) {
                    // [发送方]
                    // 接收来自接收方的ack包，收到之后结束此原语，进入
                    // SEND_DATA 或 SEND_SRAM
                    if (msg_buffer_[MSG_TYPE::ACK].size()) {
                        // 接收到数据包
                        Msg m = msg_buffer_[MSG_TYPE::ACK].front();
                        msg_buffer_[MSG_TYPE::ACK].pop();

                        if (m.msg_type_ == ACK) {
                            job_done = true;

                            LOG_DEBUG(NETWORK) << "Core " << cid << " <- ACK";
                        }
                    }
                }

                else {
                    // unimplemented
                    LOG_ERROR(logic.cpp)
                        << "Unimplemented SEND_PRIM or RECV_PRIM type";
                }

                // 等待下一个时钟周期
                wait(CYCLE, SC_NS);
            }

            if (typeid(*prim) == typeid(Send_prim)) {
                event_engine->add_event(
                    "Core " + ToHexString(cid), "Send_prim", "E",
                    Trace_event_util(
                        "Send_prim" +
                        GetEnumSendType(
                            dynamic_cast<Send_prim *>(prim)->type)));
            } else {
                event_engine->add_event(
                    "Core " + ToHexString(cid), "Recv_prim", "E",
                    Trace_event_util(
                        "Recv_prim" +
                        GetEnumRecvType(
                            dynamic_cast<Recv_prim *>(prim)->type)));
            }
        }

        if (!dte_batch.empty())
            throw std::logic_error(
                "DTE V2a batch ended with unreleased SEND_DATA contexts");
        if (batch_data_remaining != 0)
            throw std::logic_error(
                "parallel send batch ended before every DATA tail");
        send_done = true;
        wait();
    }
}
void WorkerCoreExecutor::recv_logic() {
    while (true) {
        Recv_prim *prim = (Recv_prim *)prim_queue.front();

        int recv_cnt = 0;
        int max_recv = 0;
        // 已经接收到的end包数量，需要等于recv原语中的对应要求才能结束此原语
        int end_cnt = 0;
        // 在RECV_CONFIG中，接收到最后一个config包之后，需要等待发送ack
        bool wait_send = false;
        bool job_done = false;
        vector<sc_bv<128>> segments; // 单个原语配置的所有数据包
        std::set<int> ack_subflows;
        std::set<std::pair<int, int>> ended_subflows;
        std::map<int, DteTransferContext *> stream_destination_xfers;
        std::map<int, uint64_t> stream_destination_xfer_ids;
        std::map<int, uint64_t> stream_payload_bits;
        std::map<int, uint64_t> stream_source_first_ns;
        std::map<int, uint64_t> stream_source_done_ns;
        std::map<int, uint64_t> stream_source_tail_at_destination_ns;
        std::map<int, uint64_t> stream_network_tail_ns;

        LOG_INFO(PRIM) << "Core " << cid << " start receive primitive "
                       << GetEnumRecvType(prim->type);
        LOG_DEBUG(PRIM) << "  recv_cnt " << prim->recv_cnt << ", recv_tag "
                        << prim->tag_id;

        while (true) {
            bool need_long_wait = false;
            int roofline_packets = 1;

            if (atomic_helper_lock(sc_time_stamp(), 0))
                ev_send_helper.notify(0, SC_NS);

            if (prim->type == RECV_ACK) {
                // [发送方] 接收来自接收方的ack包，收到之后结束此原语，进入
                // SEND_DATA 或 SEND_SRAM
                while (!msg_buffer_[MSG_TYPE::ACK].size())
                    wait(ev_recv_msg_type_[MSG_TYPE::ACK]);

                // 接收到数据包
                Msg m = msg_buffer_[MSG_TYPE::ACK].front();
                msg_buffer_[MSG_TYPE::ACK].pop();

                if (m.msg_type_ == ACK) {
                    if (m.subflow_ < 0 || m.subflow_ >= prim->stripe_count)
                        throw std::runtime_error(
                            "V5 ACK carries an invalid subflow id");
                    if (!ack_subflows.insert(m.subflow_).second)
                        throw std::runtime_error(
                            "V5 received a duplicate ACK subflow");
                    job_done =
                        (int)ack_subflows.size() == prim->stripe_count;
                    LOG_DEBUG(NETWORK) << "Core " << cid << " <- ACK["
                                       << m.subflow_ << "]";
                }
            }

            else if (prim->type == RECV_WEIGHT) {
                // [接收方]
                // 接收消息，但是途中如果有新的REQ包进入，需要判断是否要回发ACK包

                // 如果recv_cnt等于0,说明无需接收包裹，直接开始comp即可

                // 按照prim的tag进行判断。如果tag等同于cid，则优先查看start
                // data buffer，再查看recv buffer
                // 如果tag不等同于id，则不允许查看start data buffer

                Msg temp;
                // 表示 当前周期该核有需要处理的msg 的recv包
                while (!msg_buffer_[MSG_TYPE::P_DATA].size())
                    wait(ev_recv_msg_type_[MSG_TYPE::P_DATA]);

                temp = msg_buffer_[MSG_TYPE::P_DATA].front();
                msg_buffer_[MSG_TYPE::P_DATA].pop();

                // 复制到SRAM中
                // 如果是end包，则将recv_index归零，表示开始接收下一个core传来的数据（如果有的话）
                if (temp.is_end_) {
                    // ACK 是控制消息，使用控制信道
                    while (!atomic_helper_lock(sc_time_stamp(), 3) ||
                           !ctrl_channel_avail_i.read()) {
                            wait(CYCLE, SC_NS);
                    }

                    // 向host发送一个ack包
                    send_buffer =
                        Msg(MSG_TYPE::ACK, HostEndpointOfDie(DieOfGlobal(cid)), prim->tag_id, cid);
                    ev_send_helper.notify(0, SC_NS);

                    LOG_DEBUG(NETWORK) << "Core " << cid << " <- PREPARE data";
                    LOG_DEBUG(NETWORK) << "end_cnt " << end_cnt << ", recv_cnt "
                                       << recv_cnt << ", max_recv " << max_recv;

                    job_done = true;
                }
            }

            else if (prim->type == RECV_DATA || prim->type == RECV_START) {
                // [接收方]
                // 接收消息，但是途中如果有新的REQ包进入，需要判断是否要回发ACK包
                // 如果recv_cnt等于0,说明无需接收包裹，直接开始comp即可
                if (prim->recv_cnt == 0)
                    job_done = true;
                else {
                    // 按照prim的tag进行判断。如果tag等同于cid，则优先查看start
                    // data buffer，再查看recv buffer
                    // 如果tag不等同于id，则不允许查看start data buffer
                    ev_prim_recv_notice.notify(0, SC_NS);

                    Msg temp;
                    // 表示 当前周期该核有需要处理的msg 的recv包
                    if (prim->type == RECV_DATA) {
                        while (!msg_buffer_[MSG_TYPE::DATA].size())
                            wait(ev_recv_msg_type_[MSG_TYPE::DATA]);

                        temp = msg_buffer_[MSG_TYPE::DATA].front();
                    } else if (prim->type == RECV_START) {
                        while (!msg_buffer_[MSG_TYPE::S_DATA].size())
                            wait(ev_recv_msg_type_[MSG_TYPE::S_DATA]);

                        temp = msg_buffer_[MSG_TYPE::S_DATA].front();
                    }

                    if (prim->tag_id != cid && temp.tag_id_ != prim->tag_id) {
                        LOG_ERROR(logic.cpp)
                            << "Incompatible tag id at Core " << cid
                            << ": prim tag " << prim->tag_id
                            << ", with received tag " << temp.tag_id_;
                    }

                    if (SPEC_DTE_STREAMING && prim->type == RECV_DATA) {
                        if (temp.dte_stream_source_first_ns_ == 0 ||
                            temp.dte_stream_source_done_ns_ == 0)
                            throw std::runtime_error(
                                "DTE V2b DATA is missing source timing metadata");
                        auto existing =
                            stream_destination_xfers.find(temp.source_);
                        if (existing == stream_destination_xfers.end()) {
                            const auto key =
                                std::make_pair(temp.source_, prim->tag_id);
                            auto metadata = dte_flow_payload_rounds.find(key);
                            if (metadata == dte_flow_payload_rounds.end() ||
                                metadata->second.empty())
                                throw std::runtime_error(
                                    "DTE V2b first DATA has no REQUEST payload metadata");
                            const DteFlowPayloadRound &round =
                                metadata->second.front();
                            const uint8_t expected_mask =
                                static_cast<uint8_t>(
                                    (1u << prim->stripe_count) - 1u);
                            if (round.stripe_count != prim->stripe_count ||
                                round.subflow_mask != expected_mask)
                                throw std::runtime_error(
                                    "DTE V2b first DATA has incomplete REQUEST metadata");

                            dte->WaitForCredit();
                            DteTransferContext &xfer = dte->Issue(
                                round.payload_bits, DteDir::REMOTE_TO_SPM);
                            stream_destination_xfers.emplace(temp.source_,
                                                             &xfer);
                            stream_destination_xfer_ids.emplace(temp.source_,
                                                                xfer.xfer_id);
                            stream_payload_bits.emplace(temp.source_,
                                                        round.payload_bits);
                            stream_source_first_ns.emplace(
                                temp.source_,
                                temp.dte_stream_source_first_ns_);
                            stream_source_done_ns.emplace(
                                temp.source_,
                                temp.dte_stream_source_done_ns_);

                            const uint64_t now_ns = CurrentDteNanoseconds();
                            if (now_ns <
                                temp.dte_stream_source_first_ns_)
                                throw std::runtime_error(
                                    "DTE V2b DATA arrived before source first-ready time");
                            stream_source_tail_at_destination_ns.emplace(
                                temp.source_,
                                ProjectDteSourceTailToDestinationNs(
                                    temp.dte_stream_source_first_ns_,
                                    temp.dte_stream_source_done_ns_, now_ns));
                            TraceDteStreaming(
                                event_engine, "DTE_stream_destination", "B",
                                temp.source_, cid, prim->tag_id,
                                round.payload_bits);
                        } else {
                            if (stream_source_first_ns.at(temp.source_) !=
                                    temp.dte_stream_source_first_ns_ ||
                                stream_source_done_ns.at(temp.source_) !=
                                    temp.dte_stream_source_done_ns_)
                                throw std::runtime_error(
                                    "DTE V2b DATA changes source timing within a flow");
                        }
                    }

                    if (prim->type == RECV_DATA)
                        msg_buffer_[MSG_TYPE::DATA].pop();
                    else
                        msg_buffer_[MSG_TYPE::S_DATA].pop();

                    recv_cnt++;

                    if (temp.seq_id_ == 1 &&
                        (SYSTEM_MODE == SIM_DATAFLOW || SYSTEM_MODE == SIM_PD ||
                         SYSTEM_MODE == SIM_PDS)) {
                        // 在pos locator中添加一个kv，label是input_label
                        // 对于每一个核的第一算子的input来自与send
                        // 核的输出，并且已经会由router保存在sram上
                        AddrPosKey inp_key = AddrPosKey(*sram_addr, 0);
                        string input_label = INPUT_LABEL;

                        core_context->sram_pos_locator_->addPair(input_label,
                                                                 inp_key, true);
                    }

                    int delay = 0;
                    TaskCoreContext context = generate_context(this);
                    delay = prim->taskCoreDefault(context);

                    // 如果是end包，则将recv_index归零，表示开始接收下一个core传来的数据（如果有的话）
                    if (temp.is_end_) {
                        if (prim->type == RECV_DATA) {
                            if (temp.subflow_ < 0 ||
                                temp.subflow_ >= prim->stripe_count)
                                throw std::runtime_error(
                                    "V5 DATA tail carries an invalid subflow id");
                            if (!ended_subflows
                                     .insert({temp.source_, temp.subflow_})
                                     .second)
                                throw std::runtime_error(
                                    "V5 received a duplicate DATA subflow tail");
                            long long cycle = (long long)(
                                sc_time_stamp().value() /
                                sc_time(CYCLE, SC_NS).value());
                            g_flow_done_cycle[std::make_tuple(
                                temp.source_, temp.tag_id_, cid)] = cycle;
                        }
                        end_cnt++;
                        max_recv += temp.seq_id_;

                        LOG_DEBUG(NETWORK) << "Core " << cid << " <- DATA";
                        LOG_DEBUG(NETWORK)
                            << "  end_cnt: " << end_cnt
                            << ", recv_cnt: " << recv_cnt
                            << ", max_recv: " << max_recv
                            << ", roofline: " << temp.roofline_packets_;

                        // prim->recv_cnt 记录的是 receive 原语
                        // 需要接受的 end 包的数量 多发一的实现 max_recv
                        // 表示当前 DATA 包 发送了多少个 package 数量
                        const int expected_ends =
                            prim->recv_cnt *
                            (prim->type == RECV_DATA ? prim->stripe_count : 1);
                        if (end_cnt == expected_ends && recv_cnt >= max_recv) {
                            // 收到了所有的数据，可以结束此原语，进入comp原语
                            // 无需更新pos_locator中的kv的size，由原语自己指定输入大小
                            job_done = true;
                        }
                    }

                    need_long_wait = true;
                    if (SPEC_USE_BEHA_NOC) {
                        // V4 Behavioral 跨 die DATA 的 F 包 bulk service 已由源 die 第一条
                        // D2D link 一次性计费；到达接收核的是代表包，不能再次等待 F 拍。
                        // 同 die Behavioral NoC 沿用原 roofline 行为。
                        if (g_d2d_cfg.backend == BACKEND_BEHAVIORAL &&
                            DieOfGlobal(temp.source_) != DieOfGlobal(cid)) {
                            if (SPEC_DTE_STREAMING) {
                                if (temp.dte_stream_network_tail_cycles_ >
                                    static_cast<uint32_t>(
                                        std::numeric_limits<int>::max() - 1))
                                    throw std::overflow_error(
                                        "DTE V2b behavioral network tail overflows wait cycles");
                                roofline_packets = 1 + static_cast<int>(
                                    temp.dte_stream_network_tail_cycles_);
                            } else {
                                roofline_packets = 1;
                            }
                        } else
                            roofline_packets = temp.roofline_packets_;
                    }
                    if (SPEC_DTE_STREAMING && prim->type == RECV_DATA &&
                        temp.is_end_) {
                        const uint64_t now_ns = CurrentDteNanoseconds();
                        const uint64_t wait_ns =
                            static_cast<uint64_t>(roofline_packets) *
                            static_cast<uint64_t>(CYCLE);
                        if (wait_ns >
                            std::numeric_limits<uint64_t>::max() - now_ns)
                            throw std::overflow_error(
                                "DTE V2b network tail time overflows");
                        const uint64_t tail_ns = now_ns + wait_ns;
                        auto &saved_tail =
                            stream_network_tail_ns[temp.source_];
                        saved_tail = std::max(saved_tail, tail_ns);

                        int source_ends = 0;
                        for (const auto &flow : ended_subflows)
                            if (flow.first == temp.source_)
                                ++source_ends;
                        if (source_ends == prim->stripe_count) {
                            const uint64_t final_tail_ns = saved_tail;
                            TraceDteStreaming(
                                event_engine, "DTE_stream_network", "E",
                                temp.source_, cid, prim->tag_id,
                                stream_payload_bits.at(temp.source_),
                                sc_time(static_cast<double>(
                                            final_tail_ns - now_ns),
                                        SC_NS));
                        }
                    }
                }
            }

            else if (prim->type == RECV_CONF) {
                // [所有人]
                // 在模拟开始时接收配置，接收完毕之后发送一个ACK包给host，此原语需要对prim_queue进行压入，此原语执行完毕之后，进入RECV_DATA
                if (wait_send) {
                    // ACK 是控制消息，使用控制信道
                    while (!atomic_helper_lock(sc_time_stamp(), 3) ||
                           !ctrl_channel_avail_i.read()) {
                            wait(CYCLE, SC_NS);
                    }

                    // 正在等待向host发送ack包（CONFIG ACK；tag 契约 = RECV_CONF 的 prim->tag_id == 0，
                    // 现由 Recv_prim 类内默认值 + 显式构造保证确定，不再是未初始化 UB）。
                    send_buffer =
                        Msg(MSG_TYPE::ACK, HostEndpointOfDie(DieOfGlobal(cid)), prim->tag_id, cid);
                    ev_send_helper.notify(0, SC_NS);

                    LOG_DEBUG(NETWORK) << "Core " << cid << " <- CONFIG";

                    job_done = true;
                } else {
                    while (!msg_buffer_[MSG_TYPE::CONFIG].size())
                        wait(ev_recv_msg_type_[MSG_TYPE::CONFIG]);

                    Msg m = msg_buffer_[MSG_TYPE::CONFIG].front();
                    msg_buffer_[MSG_TYPE::CONFIG].pop();

                    if (m.config_end_) {
                        segments.push_back(m.data_);
                        prim_queue.emplace_back(parse_prim(segments));
                        segments.clear();
                    } else {
                        segments.push_back(m.data_);
                    }


                    // 检查是否为end config包，如果是，需要向host发送ack包
                    if (m.is_end_) {
                        this->prim_refill = m.refill_;
                        wait_send = true;
                    }
                }
            }

            else {
                // unimplemented
                LOG_ERROR(logic.cpp)
                    << "Unimplemented RECV_PRIM type " << prim->type;
            }

            // 等待下一个时钟周期
            wait(roofline_packets * CYCLE, SC_NS);

            ev_msg_process_end.notify();
            if (job_done) {
                if (SPEC_USE_BEHA_DTE && prim->type == RECV_DATA &&
                    prim->recv_cnt > 0) {
                    if (SPEC_SEND_RECV_PARALLEL && !send_done)
                        throw std::runtime_error(
                            "DTE V2a does not support concurrent SEND_DATA and "
                            "RECV_DATA on the same core");
                    std::set<int> completed_sources;
                    for (const auto &flow : ended_subflows)
                        completed_sources.insert(flow.first);
                    if ((int)completed_sources.size() != prim->recv_cnt)
                        throw std::runtime_error(
                            "DTE RECV_DATA source count does not match recv_cnt");

                    uint64_t payload_bits = 0;
                    for (int source : completed_sources) {
                        const auto key = std::make_pair(source, prim->tag_id);
                        auto metadata = dte_flow_payload_rounds.find(key);
                        if (metadata == dte_flow_payload_rounds.end() ||
                            metadata->second.empty())
                            throw std::runtime_error(
                                "DTE RECV_DATA is missing REQUEST payload metadata");
                        const DteFlowPayloadRound &round =
                            metadata->second.front();
                        const uint8_t expected_mask = static_cast<uint8_t>(
                            (1u << prim->stripe_count) - 1u);
                        if (round.stripe_count != prim->stripe_count ||
                            round.subflow_mask != expected_mask)
                            throw std::runtime_error(
                                "DTE RECV_DATA has incomplete REQUEST stripe metadata");
                        if (payload_bits >
                            std::numeric_limits<uint64_t>::max() -
                                round.payload_bits)
                            throw std::overflow_error(
                                "DTE RECV_DATA payload bit count overflows");
                        payload_bits += round.payload_bits;
                        metadata->second.pop_front();
                        if (metadata->second.empty())
                            dte_flow_payload_rounds.erase(metadata);
                    }

                    if (SPEC_DTE_STREAMING) {
                        if (stream_destination_xfers.size() !=
                                completed_sources.size() ||
                            stream_network_tail_ns.size() !=
                                completed_sources.size())
                            throw std::runtime_error(
                                "DTE V2b flow/context completion count mismatch");

                        uint64_t overall_target_ns = CurrentDteNanoseconds();
                        for (int source : completed_sources) {
                            DteTransferContext *context =
                                stream_destination_xfers.at(source);
                            if (context == nullptr ||
                                context->xfer_id !=
                                    stream_destination_xfer_ids.at(source))
                                throw std::logic_error(
                                    "DTE V2b destination context mapping is corrupt");
                            if (context->state !=
                                DteTransferState::COMPLETED)
                                wait(context->done);
                            if (context->state !=
                                DteTransferState::COMPLETED)
                                throw std::logic_error(
                                    "DTE V2b destination woke before completion");
                            const uint64_t dte_done_ns = static_cast<uint64_t>(
                                context->completion_time.value() /
                                sc_time(1, SC_NS).value());
                            const uint64_t target_ns =
                                CombineDteStreamingTailsNs(
                                    stream_source_tail_at_destination_ns.at(
                                        source),
                                    stream_network_tail_ns.at(source),
                                    dte_done_ns,
                                    static_cast<uint64_t>(CYCLE));
                            overall_target_ns =
                                std::max(overall_target_ns, target_ns);
                        }

                        WaitUntilDteTime(sc_time(
                            static_cast<double>(overall_target_ns), SC_NS));
                        for (int source : completed_sources) {
                            TraceDteStreaming(
                                event_engine, "DTE_stream_destination", "E",
                                source, cid, prim->tag_id,
                                stream_payload_bits.at(source));
                            if (!dte->Release(
                                    stream_destination_xfer_ids.at(source)))
                                throw std::logic_error(
                                    "DTE V2b destination context could not be released");
                        }
                    } else {
                        dte->WaitForCredit();
                        DteTransferContext &xfer =
                            dte->Issue(payload_bits, DteDir::REMOTE_TO_SPM);
                        wait(xfer.done);
                        if (!dte->Release(xfer.xfer_id))
                            throw std::logic_error(
                                "destination DTE completed context could not be released");
                    }
                }
                LOG_INFO(PRIM) << "Core " << cid << " end recv primitive "
                               << GetEnumRecvType(prim->type);
                break;
            }
        }

        ev_block.notify(CYCLE, SC_NS);
        wait();
    }
}

void WorkerCoreExecutor::task_logic() {
    while (true) {
        PrimBase *p = prim_queue.front();

        int delay = 0;
        TaskCoreContext context = generate_context(this);

        LOG_INFO(PRIM) << "Core " << cid << " start compute primitive "
                       << p->name;

        delay = p->taskCoreDefault(context);
        wait(sc_time(delay, SC_NS));

        LOG_INFO(PRIM) << "Core " << cid << " end compute primitive "
                       << p->name;

        ev_block.notify(CYCLE, SC_NS);
        wait();
    }
}
void WorkerCoreExecutor::req_logic() {
    queue<Msg> ack_queue;

    while (true) {
        if (prim_queue.size()) {
            PrimBase *p = prim_queue.front();

            if (typeid(*p) == typeid(Recv_prim)) {
                Recv_prim *prim = (Recv_prim *)p;

                if ((prim->type == RECV_DATA || prim->type == RECV_START) &&
                    !msg_buffer_[MSG_TYPE::REQUEST].empty()) {
                    queue<Msg> temp;

                    while (!msg_buffer_[MSG_TYPE::REQUEST].empty()) {
                        auto &msg = msg_buffer_[MSG_TYPE::REQUEST].front();

                        if (msg.tag_id_ == prim->tag_id) {
                            if (msg.subflow_ < 0 ||
                                msg.subflow_ >= prim->stripe_count)
                                throw std::runtime_error(
                                    "V5 REQUEST carries an invalid subflow id");
                            if (SPEC_USE_BEHA_DTE) {
                                if (msg.dte_payload_bits_ == 0)
                                    throw std::runtime_error(
                                        "DTE REQUEST is missing logical payload bits");
                                const auto key =
                                    std::make_pair(msg.source_, msg.tag_id_);
                                auto &rounds = dte_flow_payload_rounds[key];
                                const uint8_t expected_mask =
                                    static_cast<uint8_t>(
                                        (1u << prim->stripe_count) - 1u);
                                if (rounds.empty() ||
                                    rounds.back().subflow_mask == expected_mask)
                                    rounds.push_back(DteFlowPayloadRound{
                                        msg.dte_payload_bits_, 0,
                                        prim->stripe_count});
                                DteFlowPayloadRound &round = rounds.back();
                                if (round.stripe_count != prim->stripe_count ||
                                    round.payload_bits != msg.dte_payload_bits_)
                                    throw std::runtime_error(
                                        "DTE REQUEST payload metadata mismatch within a round");
                                const uint8_t subflow_bit = static_cast<uint8_t>(
                                    1u << msg.subflow_);
                                if (round.subflow_mask & subflow_bit)
                                    throw std::runtime_error(
                                        "DTE REQUEST repeats a subflow before the round is complete");
                                round.subflow_mask |= subflow_bit;
                            }
                            ack_queue.push(msg);
                        } else
                            temp.push(msg);

                        msg_buffer_[MSG_TYPE::REQUEST].pop();
                    }
                    msg_buffer_[MSG_TYPE::REQUEST] = move(temp);
                }

                // 发送ack包
                // ACK 是控制消息，使用控制信道
                while (ack_queue.size()) {
                    while (!atomic_helper_lock(sc_time_stamp(), 3) ||
                           !ctrl_channel_avail_i.read()) {
                            wait(CYCLE, SC_NS);
                    }

                    Msg req = ack_queue.front();
                    ack_queue.pop();

                    int des = req.source_;
                    send_buffer = Msg(MSG_TYPE::ACK, des, des, cid);
                    send_buffer.subflow_ = req.subflow_;
                    PinControlMsgExit(send_buffer);
                    ev_send_helper.notify(0, SC_NS);

                    LOG_DEBUG(NETWORK) << "Core " << cid << " -> ACK["
                                       << req.subflow_ << "] -> " << des;
                }
            }
        }

        wait();
    }
}