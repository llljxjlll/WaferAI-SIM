#include "router/router.h"
#include "die/port.h"
#include "monitor/watchdog.h"
#include "monitor/start_data_tracker.h"
#include "memory/hbm_network.h"
#include "utils/print_utils.h"

long g_max_output_lock_ref = 0;

namespace {
void MaybeReleaseV5DynamicPin(const Msg &m, int rid, Directions dir) {
    if (g_d2d_cfg.select_policy != SELECT_DYNAMIC ||
        !IsC2CEgressEdge(rid, dir))
        return;
    const bool data_done = m.msg_type_ == DATA && m.is_end_;
    const bool ack_done = m.msg_type_ == ACK;
    if (!data_done && !ack_done)
        return;
    ReleaseV5DynamicPort(DieOfGlobal(rid), dir,
                         FlowKey{m.source_, m.tag_id_, m.subflow_});
}
} // namespace

RouterMonitor::RouterMonitor(const sc_module_name &n,
                             Event_engine *event_engine)
    : sc_module(n), event_engine(event_engine) {
    // 多 die：全局 router 阵列（rid=全局核 id）。IsMarginCore(rid) 用 %GRID_X 判 die 内
    // 西边缘，对各 die 独立成立（CORES_PER_DIE 为 GRID_X 的整数倍）。
    routers = new RouterUnit *[TOTAL_CORES];
    for (int i = 0; i < TOTAL_CORES; i++) {
        routers[i] =
            new RouterUnit(sc_gen_unique_name("router"), i, this->event_engine);
    }
}

RouterMonitor::~RouterMonitor() {
    // free routers（数组用 delete[]）
    for (int i = 0; i < TOTAL_CORES; i++) {
        delete (routers[i]);
    }
    delete[] routers;
}

RouterUnit::RouterUnit(const sc_module_name &n, int rid,
                       Event_engine *event_engine)
    : sc_module(n), rid(rid), event_engine(event_engine) {
    if (SPEC_NOC_COLL_ENABLED &&
        SPEC_NOC_COLL_CONFIG.UsesDcaOffload()) {
        reduce_stream_engine = std::make_unique<
            coll_refactor::RouterReduceStreamEngine>(
                static_cast<uint16_t>(rid), SPEC_NOC_COLL_CONFIG.dca);
        coll_refactor::RegisterProductionReduceStreamEngine(
            static_cast<uint16_t>(rid), reduce_stream_engine.get());
    }
   
    host_data_sent_i = nullptr;
    host_data_sent_o = nullptr;
    host_channel_i = nullptr;
    host_channel_o = nullptr;
    host_buffer_i = nullptr;
    host_buffer_o = nullptr;
    host_ctrl_buffer_o = nullptr;  
    host_channel_avail_o = nullptr;
    host_ctrl_sent_o = nullptr;
    host_ctrl_channel_o = nullptr;
    // 初始全部通道均未上锁
    for (int i = 0; i < DIRECTIONS; i++) {
        input_lock[i] = 0;
        input_lock_ref[i] = 0;
        output_lock[i] = -1;
        output_lock_ref[i] = 0;
        if (i < DIRECTIONS - 1) {
            d2d_data_credit_enabled[i] = false;
            d2d_data_credit_seen[i] = false;
            d2d_data_credits[i] = 0;
            d2d_data_credit_capacity[i] = 0;
            d2d_ctrl_credit_enabled[i] = false;
            d2d_ctrl_credit_seen[i] = false;
            d2d_ctrl_credits[i] = 0;
            d2d_ctrl_credit_capacity[i] = 0;
        }
    }


    // HOST 接口按挂载表创建（legacy=西边缘 IsMarginCore，config=role=HOST 端口 tile）。
    if (IsHostAttachTile(rid)) {
        host_buffer_i = new queue<sc_bv<256>>;
        host_buffer_o = new queue<sc_bv<256>>;
        host_ctrl_buffer_o = new queue<sc_bv<256>>;  
        host_channel_i = new sc_in<sc_bv<256>>;
        host_channel_o = new sc_out<sc_bv<256>>;
        host_data_sent_i = new sc_in<bool>;
        host_data_sent_o = new sc_out<bool>;
        host_channel_avail_o = new sc_out<bool>;
        host_ctrl_sent_o = new sc_out<bool>;
        host_ctrl_channel_o = new sc_out<sc_bv<256>>;
    }

    SC_THREAD(trans_next_trigger);
    // 数据信道触发信号
    sensitive << data_sent_i[WEST].pos() << data_sent_i[EAST].pos()
              << data_sent_i[CENTER].pos() << data_sent_i[SOUTH].pos()
              << data_sent_i[NORTH].pos();
    sensitive << data_sent_i[CENTER].neg();
    if (IsHostAttachTile(rid))
        sensitive << host_data_sent_i->pos();
    sensitive << channel_avail_i[WEST].pos() << channel_avail_i[EAST].pos()
              << channel_avail_i[SOUTH].pos() << channel_avail_i[NORTH].pos();
    sensitive << core_busy_i.neg();
    if (reduce_stream_engine)
        sensitive << reduce_stream_engine->ActivityEvent();
    // 控制信道触发信号
    sensitive << ctrl_sent_i[WEST].pos() << ctrl_sent_i[EAST].pos()
              << ctrl_sent_i[CENTER].pos() << ctrl_sent_i[SOUTH].pos()
              << ctrl_sent_i[NORTH].pos();
    sensitive << ctrl_channel_avail_i[WEST].pos() << ctrl_channel_avail_i[EAST].pos()
              << ctrl_channel_avail_i[SOUTH].pos() << ctrl_channel_avail_i[NORTH].pos();
    sensitive << d2d_data_credit_i[WEST] << d2d_data_credit_i[EAST]
              << d2d_data_credit_i[SOUTH] << d2d_data_credit_i[NORTH]
              << d2d_ctrl_credit_i[WEST] << d2d_ctrl_credit_i[EAST]
              << d2d_ctrl_credit_i[SOUTH] << d2d_ctrl_credit_i[NORTH];
    sensitive << ctrl_core_busy_i.neg();
    dont_initialize();

    SC_THREAD(router_execute);
    sensitive << need_next_trigger;
    dont_initialize();
}

void RouterUnit::InjectMemRequestFlit(const MemWireFlit &wire,
                                      uint64_t *stall_counter) {
    if (!IsMemWireFlit(wire) || !InspectMemWireFlit(wire).request_direction)
        throw std::runtime_error("InjectMemRequestFlit: not a request flit");
    while (buffer_i[CENTER].size() >= MAX_BUFFER_PACKET_SIZE) {
        if (stall_counter) ++*stall_counter;
        wait(mem_data_space_event);
    }
    buffer_i[CENTER].push(wire);
    // A router has one physical transfer opportunity per cycle. A zero-time
    // notification here can re-enter router_execute several times at the same
    // timestamp and overwrite multiple flits on one sc_signal.
    need_next_trigger.notify(CYCLE, SC_NS);
}

void RouterUnit::InjectMemResponseFlit(const MemWireFlit &wire,
                                       uint64_t *stall_counter) {
    if (!IsMemWireFlit(wire) || InspectMemWireFlit(wire).request_direction)
        throw std::runtime_error("InjectMemResponseFlit: not a response flit");
    while (ctrl_buffer_i[CENTER].size() >= MAX_BUFFER_PACKET_SIZE) {
        if (stall_counter) ++*stall_counter;
        wait(mem_ctrl_space_event);
    }
    ctrl_buffer_i[CENTER].push(wire);
    need_next_trigger.notify(CYCLE, SC_NS);
}

void RouterUnit::EnableD2DDataCredit(Directions dir, int initial_credit) {
    if (dir < WEST || dir >= CENTER || initial_credit < 1)
        throw std::runtime_error("invalid D2D data credit configuration");
    d2d_data_credit_enabled[(int)dir] = true;
    d2d_data_credits[(int)dir] = initial_credit;
    d2d_data_credit_capacity[(int)dir] = initial_credit;
}

void RouterUnit::EnableD2DCtrlCredit(Directions dir, int initial_credit) {
    if (dir < WEST || dir >= CENTER || initial_credit < 1)
        throw std::runtime_error("invalid D2D control credit configuration");
    d2d_ctrl_credit_enabled[(int)dir] = true;
    d2d_ctrl_credits[(int)dir] = initial_credit;
    d2d_ctrl_credit_capacity[(int)dir] = initial_credit;
}

bool RouterUnit::D2DDataCreditsBalanced() const {
    for (int i = 0; i < DIRECTIONS - 1; ++i)
        if (d2d_data_credit_enabled[i] &&
            d2d_data_credits[i] != d2d_data_credit_capacity[i])
            return false;
    return true;
}

bool RouterUnit::D2DCtrlCreditsBalanced() const {
    for (int i = 0; i < DIRECTIONS - 1; ++i)
        if (d2d_ctrl_credit_enabled[i] &&
            d2d_ctrl_credits[i] != d2d_ctrl_credit_capacity[i])
            return false;
    return true;
}

void RouterUnit::end_of_elaboration() {
    // set signals
    for (int i = 0; i < DIRECTIONS; i++) {
        channel_avail_o[i].write(true);
        data_sent_o[i].write(false);
        ctrl_channel_avail_o[i].write(true);
        ctrl_sent_o[i].write(false);
    }

    if (IsHostAttachTile(rid)) {
        host_channel_avail_o->write(true);
        host_data_sent_o->write(false);
        host_ctrl_sent_o->write(false);
    }
}

void RouterUnit::router_execute() {
    while (true) {
        bool flag_trigger = false;
        const auto is_stream_wire = [&](const sc_bv<256> &wire) {
            return reduce_stream_engine &&
                (coll_refactor::IsReduceStreamHeaderWire(wire) ||
                 coll_refactor::IsReduceStreamDataWire(wire));
        };
        const auto requires_pulse_gap = [&](const sc_bv<256> &wire) {
            // P2P endpoint DATA uses the normal Msg route, but its bit-255
            // discriminator makes every fragment a strict, non-collapsible
            // transport unit. Consecutive writes of false then true in the
            // same SystemC delta do not create a new posedge at the next hop,
            // so give endpoint wires the same explicit low cycle as the
            // strict collective streams.
            return IsIsaV1CollectiveByteStartWire(wire) ||
                   IsIsaV1CollectiveByteDataWire(wire) ||
                   IsCollDataWire(wire) ||
                   IsCollReduceHeaderWire(wire) ||
                   IsCollReducePayloadWire(wire) ||
                   is_stream_wire(wire) ||
                   (!IsMemWireFlit(wire) && wire[255].to_bool());
        };

        if (reduce_stream_engine) {
            const uint64_t cycle = sc_time_stamp().value() /
                sc_time(CYCLE, SC_NS).value();
            if (!reduce_stream_ticked || cycle > reduce_stream_last_cycle) {
                reduce_stream_engine->Tick(cycle);
                reduce_stream_ticked = true;
                reduce_stream_last_cycle = cycle;
            }
            if (const auto *egress =
                    reduce_stream_engine->FrontEgress()) {
                auto &output = buffer_o[egress->output];
                if (output.size() < MAX_BUFFER_PACKET_SIZE) {
                    output.push(egress->wire);
                    reduce_stream_engine->PopEgress();
                    flag_trigger = true;
                }
            }
        }

        for (auto it = reduce_scheduled.begin();
             it != reduce_scheduled.end();) {
            if (it->ready > sc_time_stamp()) { ++it; continue; }
            auto &outq = buffer_o[it->output];
            if (outq.size() + 2 > MAX_BUFFER_PACKET_SIZE) {
                ++it; continue;
            }
            const auto wire = SerializeCollReduceOperand(it->operand);
            outq.emplace(wire[0]); outq.emplace(wire[1]);
            it = reduce_scheduled.erase(it); flag_trigger = true;
        }

        for (int i = 0; i < DIRECTIONS - 1; ++i) {
            bool data_event = d2d_data_credit_i[i].read();
            if (d2d_data_credit_enabled[i] &&
                data_event != d2d_data_credit_seen[i]) {
                if (d2d_data_credits[i] >= d2d_data_credit_capacity[i])
                    throw std::runtime_error("D2D data credit overflow");
                d2d_data_credits[i]++;
                flag_trigger = true;
            }
            d2d_data_credit_seen[i] = data_event;
            bool pulse = d2d_ctrl_credit_i[i].read();
            if (d2d_ctrl_credit_enabled[i] && pulse != d2d_ctrl_credit_seen[i]) {
                if (d2d_ctrl_credits[i] >= d2d_ctrl_credit_capacity[i])
                    throw std::runtime_error("D2D control credit overflow");
                d2d_ctrl_credits[i]++;
                flag_trigger = true;
            }
            d2d_ctrl_credit_seen[i] = pulse;
        }

        // 将输出信号都设置为初始值false
        for (int i = 0; i < DIRECTIONS; i++) {
            channel_avail_o[i].write(false);
            data_sent_o[i].write(false);
            ctrl_channel_avail_o[i].write(false);
            ctrl_sent_o[i].write(false);
        }

        // ==================== 数据信道输入 ====================
        // [input] 4方向+cores - 数据信道
        for (int i = 0; i < DIRECTIONS; i++) {
            if (data_sent_i[i].read()) {
                // move the data into the buffer
                sc_bv<256> temp = channel_i[i].read();
                const bool collective_wire = requires_pulse_gap(temp);
                if (i == CENTER && collective_wire &&
                    !center_collective_armed)
                    continue;
                if (i == CENTER && collective_wire)
                    center_collective_armed = false;
                // V2-b：该方向是 peer-connected C2C 边 ⇒ 本包刚跨 link 进入本 die，
                // 入口处清除上一跳 pin 并按本 die 重新 pin（见 RepinOnC2CIngress）。
                bool from_c2c = IsC2CEgressEdge(rid, Directions(i));
                if (from_c2c &&
                    (IsIsaV1CollectiveByteStartWire(temp) ||
                     IsIsaV1CollectiveByteDataWire(temp)))
                    throw std::runtime_error(
                        "ISA-v1 strict collective wire cannot cross dies");
                if (from_c2c && !IsMemWireFlit(temp))
                    temp = RepinOnC2CIngress(temp);
                CountDieRouterPkt(i, from_c2c); // V2-c：本 die NoC 活动

                buffer_i[i].emplace(temp);

                // need trigger again
                flag_trigger = true;
            } else if (i == CENTER) center_collective_armed = true;
        }

        // ==================== 控制信道输入 ====================
        // [ctrl input] 4方向+cores - 控制信道
        for (int i = 0; i < DIRECTIONS; i++) {
            if (ctrl_sent_i[i].read()) {
                // move the control data into the ctrl buffer
                sc_bv<256> temp = ctrl_channel_i[i].read();
                // V2-b：控制包（REQUEST/ACK）与 DATA 一样，跨 link 进入本 die 后必须重新 pin。
                bool from_c2c = IsC2CEgressEdge(rid, Directions(i));
                if (from_c2c && !IsMemWireFlit(temp))
                    temp = RepinOnC2CIngress(temp);
                CountDieRouterPkt(i, from_c2c); // V2-c：本 die NoC 活动

                ctrl_buffer_i[i].emplace(temp);

                // need trigger again
                flag_trigger = true;
            }
        }

        // [input] host
        // if IsMarginCore
        if (host_buffer_i) {
            // host send data to core
            if (host_data_sent_i->read()) {
                // move the data into the buffer
                sc_bv<256> temp = host_channel_i->read();

                Msg tt = DeserializeMsg(temp);
                if (tt.msg_type_ == MSG_TYPE::S_DATA)
                    RecordStartDataStage(StartDataStage::ROUTER_ACCEPTED, tt);

                host_buffer_i->emplace(temp);

                // need trigger again
                flag_trigger = true;
            }
        }

        // ==================== 数据信道输出 ====================
        // [output] 4方向 - 数据信道
        for (int i = 0; i < DIRECTIONS - 1; i++) {
            // global update once
            data_sent_o[i].write(false);
            // bounded C2C DATA 以 SAF 空位 credit 为唯一流控真源；其它输出保持 ready。
            bool blocked = d2d_data_credit_enabled[i]
                               ? d2d_data_credits[i] <= 0
                               : channel_avail_i[i].read() == false;
            if (blocked) {
                if (buffer_o[i].size()) {
                    int d = DieOfGlobal(rid);
                    if (IsC2CEgressEdge(rid, Directions(i)))
                        g_d2d_source_stalls++;
                    else if (d >= 0 && d < (int)g_die_noc_stalls.size())
                        g_die_noc_stalls[d]++;
                }
                continue;
            }

            if (!buffer_o[i].size())
                continue;

            if (collective_output_cooldown[i]) {
                collective_output_cooldown[i] = false;
                flag_trigger = true;
                continue;
            }

            sc_bv<256> temp = buffer_o[i].front();
            buffer_o[i].pop();

            channel_o[i].write(temp);
            data_sent_o[i].write(true);
            const bool collective_wire = requires_pulse_gap(temp);
            if (collective_wire)
                collective_output_cooldown[i] = true;
            if (collective_wire) {
                if (SPEC_NOC_COLL_ENABLED)
                    RecordCollectiveSharedOutput(
                        static_cast<uint16_t>(rid),
                        static_cast<uint8_t>(i), true);
            } else if (IsMemWireFlit(temp)) {
                if (ActiveHBMNetwork())
                    ActiveHBMNetwork()->RecordHop(rid, i, temp);
            } else {
                const Msg normal_msg = DeserializeMsg(temp);
                if (SPEC_NOC_COLL_ENABLED && normal_msg.msg_type_ == DATA)
                    RecordCollectiveSharedOutput(
                        static_cast<uint16_t>(rid),
                        static_cast<uint8_t>(i), false);
                MaybeReleaseV5DynamicPin(normal_msg, rid, Directions(i));
            }
            if (d2d_data_credit_enabled[i])
                d2d_data_credits[i]--;
            int d = DieOfGlobal(rid);
            if (!IsC2CEgressEdge(rid, Directions(i)) && d >= 0 &&
                d < (int)g_die_noc_sends.size())
                g_die_noc_sends[d]++;

            // need trigger again
            flag_trigger = true;
        }

        // ==================== 控制信道输出 ====================
        // [ctrl output] 4方向 - 控制信道
        for (int i = 0; i < DIRECTIONS - 1; i++) {
            // global update once
            ctrl_sent_o[i].write(false);

            // Make the low phase externally observable before another
            // control packet is launched on this Router-to-Router link.
            if (ctrl_output_gate[i].CoolingDown()) {
                ctrl_output_gate[i].BeginCycle();
                if (!ctrl_buffer_o[i].empty())
                    flag_trigger = true;
                continue;
            }
            // bounded C2C 使用真实 credit；其它输出保持 legacy ready 信号。
            if (d2d_ctrl_credit_enabled[i]) {
                if (d2d_ctrl_credits[i] <= 0)
                    continue;
            } else if (ctrl_channel_avail_i[i].read() == false) {
                continue;
            }

            if (!ctrl_buffer_o[i].size())
                continue;

            sc_bv<256> temp = ctrl_buffer_o[i].front();
            ctrl_buffer_o[i].pop();

            ctrl_channel_o[i].write(temp);
            ctrl_sent_o[i].write(true);
            ctrl_output_gate[i].MarkSent();
            if (d2d_ctrl_credit_enabled[i])
                d2d_ctrl_credits[i]--;
            if (IsMemWireFlit(temp)) {
                if (ActiveHBMNetwork())
                    ActiveHBMNetwork()->RecordHop(rid, i, temp);
            } else {
                Msg tt = DeserializeMsg(temp);
                MaybeReleaseV5DynamicPin(tt, rid, Directions(i));
            }

            // need trigger again
            flag_trigger = true;
        }

        // [output] host - 数据信道
        // if IsMarginCore
        if (host_channel_i) {
            host_data_sent_o->write(false);
            // 输出到host方向上的buffer非空
            if (host_buffer_o->size()) {
                sc_bv<256> temp = host_buffer_o->front();
                host_buffer_o->pop();

                host_channel_o->write(temp);
                host_data_sent_o->write(true);

                // need trigger again
                flag_trigger = true;
            }
        }

        // [ctrl output] host - 控制信道 
        // if IsMarginCore 控制消息发送到host
        if (host_ctrl_channel_o) {
            host_ctrl_sent_o->write(false);
            // 输出到host方向上的控制buffer非空
            if (host_ctrl_buffer_o->size()) {
                sc_bv<256> temp = host_ctrl_buffer_o->front();
                host_ctrl_buffer_o->pop();

                host_ctrl_channel_o->write(temp);
                host_ctrl_sent_o->write(true);

                // need trigger again
                flag_trigger = true;
            }
        }

        // [output] core - 数据信道
        // 输出到本地core内部的
        data_sent_o[CENTER].write(false);
        // 输出到本地core内的buffer非空
        if (buffer_o[CENTER].size()) {
            sc_bv<256> front = buffer_o[CENTER].front();
            if (IsMemWireFlit(front)) {
                if (!ActiveHBMNetwork())
                    throw std::runtime_error("MEM request reached core without HBMNetwork");
                if (ActiveHBMNetwork()->TryAcceptRequestFlit(rid, front)) {
                    buffer_o[CENTER].pop();
                    flag_trigger = true;
                }
            } else
            // core内部的接受队列是否满
            if (collective_output_cooldown[CENTER]) {
                collective_output_cooldown[CENTER] = false;
                flag_trigger = true;
            } else if (!core_busy_i.read()) {
                // move the data out of the buffer
                sc_bv<256> temp = buffer_o[CENTER].front();

                buffer_o[CENTER].pop();

                channel_o[CENTER].write(temp);
                data_sent_o[CENTER].write(true);
                if (requires_pulse_gap(temp))
                    collective_output_cooldown[CENTER] = true;
            }

            // need trigger again
            flag_trigger = true;
        }

        // [ctrl output] core - 控制信道
        // 输出控制消息到本地core
        ctrl_sent_o[CENTER].write(false);
        // 控制信道输出到本地core内的buffer非空
        if (ctrl_output_gate[CENTER].CoolingDown()) {
            ctrl_output_gate[CENTER].BeginCycle();
            if (!ctrl_buffer_o[CENTER].empty())
                flag_trigger = true;
        } else if (ctrl_buffer_o[CENTER].size()) {
            sc_bv<256> front = ctrl_buffer_o[CENTER].front();
            if (IsMemWireFlit(front)) {
                if (!ActiveHBMNetwork())
                    throw std::runtime_error("MEM response reached core without HBMNetwork");
                if (ActiveHBMNetwork()->TryAcceptResponseFlit(rid, front)) {
                    ctrl_buffer_o[CENTER].pop();
                    flag_trigger = true;
                }
            } else
            // 控制信道的core接受队列是否满（独立于数据信道）
            if (!ctrl_core_busy_i.read()) {
                // move the ctrl data out of the buffer
                sc_bv<256> temp = ctrl_buffer_o[CENTER].front();

                ctrl_buffer_o[CENTER].pop();

                Msg tt = DeserializeMsg(temp);

                ctrl_channel_o[CENTER].write(temp);
                ctrl_sent_o[CENTER].write(true);
                ctrl_output_gate[CENTER].MarkSent();
            }

            // need trigger again
            flag_trigger = true;
        }

        // [input -> output] host
        // host输入包 向 output 哪个方向输出
        if (host_channel_i && host_buffer_i->size()) {
            sc_bv<256> temp = host_buffer_i->front();
            int d = DeserializeMsg(temp).des_;
            // 先x后y的路由
            Directions next = GetNextHop(d, rid);

            if (buffer_o[next].size() < MAX_BUFFER_PACKET_SIZE &&
                output_lock[next] == -1) {
                host_buffer_i->pop();
                buffer_o[next].emplace(temp);

                flag_trigger = true;
            }
        }

        // ==================== 控制信道路由 ====================
        // [ctrl input -> ctrl output] 4方向+core - 控制信道路由
        for (int i = 0; i < DIRECTIONS; i++) {
            if (!ctrl_buffer_i[i].size())
                continue;

            sc_bv<256> temp = ctrl_buffer_i[i].front();
            if (IsMemWireFlit(temp)) {
                Directions out = MemFlitNextHop(temp, rid);
                if (ctrl_buffer_o[out].size() >= MAX_BUFFER_PACKET_SIZE)
                    continue;
                if (!ctrl_buffer_o[out].empty() &&
                    !IsMemWireFlit(ctrl_buffer_o[out].front()) &&
                    ActiveHBMNetwork())
                    ActiveHBMNetwork()->RecordSharedNocContention();
                ctrl_buffer_i[i].pop();
                ctrl_buffer_o[out].push(temp);
                if (i == CENTER) mem_ctrl_space_event.notify(SC_ZERO_TIME);
                flag_trigger = true;
                continue;
            }
            Msg m = DeserializeMsg(temp);
            // core 目的：跨 die 时消费源核选定并随包携带的固定 exit_port；进入目标 die
            // 后退回片内 XY。HOST 目的仍以消息 source 作为 egress anchor。
            Directions out = ControlMsgNextHop(m, rid);

            // HOST 路由只能落在挂载 tile；直接同时查指针，杜绝挂载表与指针状态不同步时
            // 的空指针解引用（当前合法路由恒在挂载 tile 才返回 HOST）。
            if (out == HOST &&
                (!IsHostAttachTile(rid) || host_ctrl_buffer_o == nullptr))
                throw std::runtime_error(
                    "HOST ctrl route reached a non-attachment tile");

            // 控制信道不需要上锁机制，直接检查buffer是否满
            // REQUEST包和其他控制消息（ACK/DONE）一样直接流动，不需要req_queue
            if (out == HOST) {
                if (host_ctrl_buffer_o->size() >= MAX_BUFFER_PACKET_SIZE)
                    continue;
                ctrl_buffer_i[i].pop();
                host_ctrl_buffer_o->emplace(temp);
            } else {
                if (ctrl_buffer_o[out].size() >= MAX_BUFFER_PACKET_SIZE)
                    continue;
                ctrl_buffer_i[i].pop();
                ctrl_buffer_o[out].emplace(temp);
            }

            flag_trigger = true;
        }

        // ==================== 数据信道路由 ====================
        // FIX input -> output 的仲裁
        // [input -> output] 4方向+core - 数据信道
        for (int coll_step = 0; coll_step < DIRECTIONS; ++coll_step) {
            const int i = (collective_rr_start + coll_step) % DIRECTIONS;
            if (!buffer_i[i].size())
                continue;

            sc_bv<256> temp = buffer_i[i].front();
            if (IsIsaV1CollectiveByteStartWire(temp)) {
                const IsaV1CollectiveByteStart start =
                    DeserializeIsaV1CollectiveByteStart(temp);
                if (start.kind != IsaV1CollectiveByteKind::MULTICAST)
                    throw std::runtime_error(
                        "strict byte START kind is not a multicast stream");
                const IsaV1CollectiveByteLock route{
                    start.tree_id, start.session_id,
                    start.collective.epoch};
                if (strict_multicast_streams.count(route) != 0)
                    throw std::runtime_error(
                        "duplicate strict multicast START at Router");
                const uint8_t outputs = LookupCollectiveTreeEntry(
                    {start.tree_id, static_cast<uint16_t>(rid),
                     static_cast<uint8_t>(i)});
                bool available[DIRECTIONS] = {};
                for (int d = 0; d < DIRECTIONS; ++d) {
                    if ((outputs & (1U << d)) != 0 && d != CENTER &&
                        IsC2CEgressEdge(rid, static_cast<Directions>(d)))
                        throw std::runtime_error(
                            "strict multicast topology enters a D2D link");
                    available[d] =
                        buffer_o[d].size() < MAX_BUFFER_PACKET_SIZE;
                }
                const CollBranchLockKey lock_key{
                    start.tree_id, start.collective, start.session_id, 0};
                const bool can_commit = collective_fork.CanCommit(
                    outputs, available, lock_key);
                RecordCollectiveForkAttempt(
                    start.tree_id, static_cast<uint16_t>(rid), outputs,
                    can_commit);
                if (!can_commit) continue;
                const uint32_t fragments =
                    start.total_bytes / P2P_PAYLOAD_FRAGMENT_BYTES +
                    (start.total_bytes % P2P_PAYLOAD_FRAGMENT_BYTES != 0);
                const auto inserted = strict_multicast_streams.emplace(
                    route, StrictMulticastRouterStream{start, fragments, 1});
                if (!inserted.second)
                    throw std::logic_error(
                        "strict multicast START insertion failed");
                collective_fork.Commit(outputs, true, false, lock_key);
                buffer_i[i].pop();
                for (int d = 0; d < DIRECTIONS; ++d)
                    if ((outputs & (1U << d)) != 0)
                        buffer_o[d].emplace(temp);
                collective_rr_start = (i + 1) % DIRECTIONS;
                flag_trigger = true;
                continue;
            }
            if (IsIsaV1CollectiveByteDataWire(temp)) {
                const IsaV1CollectiveByteData data =
                    InspectIsaV1CollectiveByteDataWire(temp);
                if (data.kind != IsaV1CollectiveByteKind::MULTICAST)
                    throw std::runtime_error(
                        "strict byte DATA kind is not a multicast stream");
                const auto state = strict_multicast_streams.find(data.lock);
                if (state == strict_multicast_streams.end())
                    throw std::runtime_error(
                        "strict multicast DATA has no Router START state");
                if (state->second.start.kind != data.kind)
                    throw std::runtime_error(
                        "strict multicast DATA kind mismatches Router START");
                if (data.sequence != state->second.next_sequence)
                    throw std::runtime_error(
                        "strict multicast DATA sequence is not contiguous");
                const bool expected_tail =
                    state->second.next_sequence ==
                    state->second.fragment_count;
                const uint32_t consumed =
                    (state->second.next_sequence - 1) *
                    P2P_PAYLOAD_FRAGMENT_BYTES;
                const uint8_t expected_length = static_cast<uint8_t>(
                    std::min<uint32_t>(
                        P2P_PAYLOAD_FRAGMENT_BYTES,
                        state->second.start.total_bytes - consumed));
                if (data.tail != expected_tail ||
                    data.length_bytes != expected_length)
                    throw std::runtime_error(
                        "strict multicast DATA tail shape mismatches START");
                const uint8_t outputs = LookupCollectiveTreeEntry(
                    {data.lock.tree_id, static_cast<uint16_t>(rid),
                     static_cast<uint8_t>(i)});
                bool available[DIRECTIONS] = {};
                for (int d = 0; d < DIRECTIONS; ++d) {
                    if ((outputs & (1U << d)) != 0 && d != CENTER &&
                        IsC2CEgressEdge(rid, static_cast<Directions>(d)))
                        throw std::runtime_error(
                            "strict multicast topology enters a D2D link");
                    available[d] =
                        buffer_o[d].size() < MAX_BUFFER_PACKET_SIZE;
                }
                const CollBranchLockKey lock_key{
                    data.lock.tree_id, state->second.start.collective,
                    data.lock.session_id, 0};
                const bool can_commit = collective_fork.CanCommit(
                    outputs, available, lock_key);
                RecordCollectiveForkAttempt(
                    data.lock.tree_id, static_cast<uint16_t>(rid), outputs,
                    can_commit);
                if (!can_commit) continue;
                collective_fork.Commit(outputs, false, data.tail, lock_key);
                buffer_i[i].pop();
                for (int d = 0; d < DIRECTIONS; ++d)
                    if ((outputs & (1U << d)) != 0)
                        buffer_o[d].emplace(temp);
                ++state->second.next_sequence;
                if (data.tail)
                    strict_multicast_streams.erase(state);
                collective_rr_start = (i + 1) % DIRECTIONS;
                flag_trigger = true;
                continue;
            }
            if (reduce_stream_engine &&
                coll_refactor::IsReduceStreamHeaderWire(temp)) {
                const auto header =
                    coll_refactor::DeserializeReduceStreamHeader(
                        temp, SPEC_NOC_COLL_CONFIG.dca.vector_bits);
                if (!reduce_stream_engine->TryAcceptHeader(
                        static_cast<Directions>(i), header))
                    continue;
                buffer_i[i].pop();
                collective_rr_start = (i + 1) % DIRECTIONS;
                flag_trigger = true;
                continue;
            }
            if (reduce_stream_engine &&
                coll_refactor::IsReduceStreamDataWire(temp)) {
                const auto data =
                    coll_refactor::DeserializeReduceStreamData(temp);
                const auto status = reduce_stream_engine->TryAcceptData(
                    static_cast<Directions>(i), data);
                if (status == coll_refactor::
                        ReduceStreamAcceptStatus::BACKPRESSURE)
                    continue;
                buffer_i[i].pop();
                collective_rr_start = (i + 1) % DIRECTIONS;
                flag_trigger = true;
                continue;
            }
            if (IsCollReduceHeaderWire(temp)) {
                if (reduce_header_pending[i])
                    throw std::runtime_error(
                        "second reduce header arrived before payload");
                reduce_header_pending[i] = true;
                reduce_header_wire[i] = temp;
                buffer_i[i].pop(); flag_trigger = true;
                continue;
            }
            if (IsCollReducePayloadWire(temp)) {
                if (!reduce_header_pending[i])
                    throw std::runtime_error("reduce payload arrived without header");
                if (reduce_scheduled.size() >= 16)
                    continue;
                CollReduceOperand operand = DeserializeCollReduceOperand(
                    {reduce_header_wire[i], temp});
                operand.child_id = static_cast<uint16_t>(i);
                const CollReduceMatchKey key{operand.collective,
                                             operand.phase_id,
                                             operand.chunk_id};
                const auto node = LookupCollectiveReduceNode(
                    operand.tree_id, static_cast<uint16_t>(rid));
                if (!reduce_active.count(key)) {
                    if (!reduce_match.Open(key, node.expected_inputs,
                                           operand.dtype, operand.op,
                                           operand.valid_elements))
                        continue;
                    reduce_active.emplace(key, operand);
                }
                const auto status = reduce_match.Accept(
                    key, operand.child_id, operand.dtype, operand.op,
                    operand.valid_elements, operand.payload);
                if (status == CollOperandStatus::BACKPRESSURE)
                    continue;
                if (status == CollOperandStatus::DUPLICATE ||
                    status == CollOperandStatus::UNEXPECTED ||
                    status == CollOperandStatus::MISMATCH)
                    throw std::runtime_error("invalid in-network reduce operand");
                buffer_i[i].pop(); reduce_header_pending[i] = false;
                if (status == CollOperandStatus::READY) {
                    CollDcaResult result = reduce_match.Consume(key);
                    CollReduceOperand reinject = reduce_active.at(key);
                    reduce_active.erase(key); reinject.payload = result.payload;
                    reinject.child_id = static_cast<uint16_t>(rid);
                    const sc_time start = std::max(sc_time_stamp(),
                                                   reduce_dca_available);
                    reduce_dca_available = start +
                        sc_time(result.service_cycles * CYCLE, SC_NS);
                    reduce_scheduled.push_back({
                        reduce_dca_available,
                        node.parent_output, reinject});
                }
                collective_rr_start = (i + 1) % DIRECTIONS;
                flag_trigger = true;
                continue;
            }
            if (IsCollDataWire(temp)) {
                const CollDataHeader h = DeserializeCollData(temp);
                const uint8_t outputs = LookupCollectiveTreeEntry(
                    {h.tree_id, static_cast<uint16_t>(rid),
                     static_cast<uint8_t>(i)});
                bool available[DIRECTIONS] = {};
                for (int d = 0; d < DIRECTIONS; ++d)
                    available[d] = buffer_o[d].size() < MAX_BUFFER_PACKET_SIZE;
                const CollBranchLockKey lock_key{
                    h.tree_id, h.packet.collective, h.packet.phase_id,
                    h.packet.chunk_id};
                const bool can_commit = collective_fork.CanCommit(
                    outputs, available, lock_key);
                RecordCollectiveForkAttempt(
                    h.tree_id, static_cast<uint16_t>(rid), outputs,
                    can_commit);
                if (!can_commit) continue;
                collective_fork.Commit(outputs, h.seq_id == 1, h.is_end,
                                       lock_key);
                buffer_i[i].pop();
                for (int d = 0; d < DIRECTIONS; ++d)
                    if (outputs & (1u << d)) buffer_o[d].emplace(temp);
                collective_rr_start = (i + 1) % DIRECTIONS;
                flag_trigger = true;
                continue;
            }
            if (IsMemWireFlit(temp)) {
                Directions out = MemFlitNextHop(temp, rid);
                if (buffer_o[out].size() >= MAX_BUFFER_PACKET_SIZE)
                    continue;
                if (!buffer_o[out].empty() && ActiveHBMNetwork()) {
                    if (IsMemWireFlit(buffer_o[out].front()))
                        ActiveHBMNetwork()->RecordMemNocContention();
                    else
                        ActiveHBMNetwork()->RecordSharedNocContention();
                }
                buffer_i[i].pop();
                buffer_o[out].push(temp);
                if (i == CENTER) mem_data_space_event.notify(SC_ZERO_TIME);
                collective_rr_start = (i + 1) % DIRECTIONS;
                flag_trigger = true;
                continue;
            }
            Msg m = DeserializeMsg(temp);
            // core 目的 DATA：跨 die 时消费 SEND_DATA 原语一次选定、随所有包携带的
            // exit_port；进入目标 die 后退回片内 XY。HOST 目的仍使用 source anchor。
            Directions out = DataMsgNextHop(m, rid);
            const bool endpoint_data = m.p2p_endpoint_ &&
                m.msg_type_ == MSG_TYPE::DATA &&
                !IsHostEndpoint(m.des_) && !IsHostEndpoint(m.source_);
            const EndpointOutputFlowKey endpoint_flow{
                m.source_, m.des_, m.tag_id_, m.subflow_};

            // HOST 路由只能落在挂载 tile；直接同时查指针，杜绝空指针解引用
            // （3b-2 改此核心路径，此检查作兜底）。
            if (out == HOST &&
                (!IsHostAttachTile(rid) || host_buffer_o == nullptr))
                throw std::runtime_error(
                    "HOST data route reached a non-attachment tile");

            if (!IsHostEndpoint(m.des_)) {
                if (endpoint_output_lock[out].Active()) {
                    // An endpoint owner is exclusive even against legacy
                    // same-tag DATA: no other flow may split its fragments.
                    if (!endpoint_data ||
                        !endpoint_output_lock[out].OwnedBy(endpoint_flow))
                        continue;
                    if (m.seq_id_ == 1)
                        throw std::runtime_error(
                            "duplicate P2P endpoint DATA seq1 while Router "
                            "output flow is active");
                } else if (endpoint_data) {
                    if (m.seq_id_ != 1)
                        throw std::runtime_error(
                            "P2P endpoint DATA continuation has no Router "
                            "output flow owner");
                    // A legacy tag/refcount flow already owns this output.
                    if (output_lock[out] != -1)
                        continue;
                } else if (output_lock[out] != -1 &&
                           output_lock[out] != m.tag_id_) {
                    continue;
                }
            }
            if (out == HOST &&
                host_buffer_o->size() >=
                    MAX_BUFFER_PACKET_SIZE) // 如果发往host，但通道已满：continue
                continue;
            else if (
                out != HOST &&
                buffer_o[out].size() >=
                    MAX_BUFFER_PACKET_SIZE) // 如果不发往host，但通道已满：continue
                continue;


            // Legacy output_lock 按 tag 锁是**有意设计**（非缺陷）：tag == 接收核 recv_tag == 全局核
            // id（唯一），即「接收端聚合槽」。同 tag 的包（多源发一个核=多发一）共享锁、交错通过，
            // 接收端按包内地址重组。全局 tag 下无「不同接收核撞 tag」别名，故不加 source 维
            // （加了会把多发一错误拆成串行）。跨 die 时 out 由 DataMsgNextHop 给出，锁在正确方向。
            // 详见 common/flow.h。
            // FIX 上锁应该在第一个DATA 包
            if (endpoint_data && m.seq_id_ == 1) {
                if (output_lock[out] != -1 || output_lock_ref[out] != 0 ||
                    endpoint_output_lock[out].Active())
                    throw std::logic_error(
                        "P2P endpoint Router output acquisition state is "
                        "inconsistent");
                endpoint_output_lock[out].Acquire(endpoint_flow);
                output_lock[out] = m.tag_id_;
                output_lock_ref[out] = 1;
                if (output_lock_ref[out] > g_max_output_lock_ref)
                    g_max_output_lock_ref = output_lock_ref[out];
            } else if (!endpoint_data && m.msg_type_ == DATA &&
                       m.seq_id_ == 1 && !IsHostEndpoint(m.des_) &&
                       !IsHostEndpoint(m.source_)) {
                // i 是 ACK 的进入方向，需要计算 ACK 的输出方向
                if (output_lock[out] == -1) {
                    // 上锁
                    output_lock[out] = m.tag_id_;
                    output_lock_ref[out]++;
                    if (output_lock_ref[out] > g_max_output_lock_ref)
                        g_max_output_lock_ref = output_lock_ref[out];

                    LOG_DEBUG(NETWORK)
                        << "Router " << rid << " set lock direction "
                        << GetEnumDirectionType(out);
                    LOG_DEBUG(NETWORK)
                        << "  lock tag " << output_lock[out]
                        << ", lock reference " << output_lock_ref[out];
                } else if (output_lock[out] == m.tag_id_) {
                    // 添加refcnt
                    // Two Ack 多发一 DATA 包 乱序 接受核的接受地址由 Send
                    // 包中地址决定
                    output_lock_ref[out]++;
                    if (output_lock_ref[out] > g_max_output_lock_ref)
                        g_max_output_lock_ref = output_lock_ref[out];

                    LOG_DEBUG(NETWORK)
                        << "Router " << rid << " add lock reference "
                        << GetEnumDirectionType(out);
                    LOG_DEBUG(NETWORK)
                        << "  lock tag " << output_lock[out]
                        << ", lock reference " << output_lock_ref[out];
                } else {
                    // 并非对应tag，不予通过
                    continue;
                }
            }

            // [DATA] 最后一个数据包，需要减少refcnt，如果refcnt为0,则解锁
            // DTODO
            // 排除了Config DATA 包，不会减少 lock
            // START DATA 包也不会上锁？
            if (endpoint_data && m.is_end_) {
                if (!endpoint_output_lock[out].OwnedBy(endpoint_flow) ||
                    output_lock[out] != m.tag_id_ ||
                    output_lock_ref[out] != 1)
                    throw std::runtime_error(
                        "P2P endpoint DATA tail mismatches Router output "
                        "flow owner");
                endpoint_output_lock[out].Release(endpoint_flow);
                output_lock_ref[out] = 0;
                output_lock[out] = -1;
            } else if (!endpoint_data && m.msg_type_ == DATA && m.is_end_ &&
                       !IsHostEndpoint(m.source_) &&
                       !IsHostEndpoint(m.des_)) {
                // 必须使用本轮 DataMsgNextHop 已解析出的同一 out；跨 die 源侧若回退到
                // GetNextHop(des,rid) 会把全局 dest 当作片内坐标并解错锁。

                output_lock_ref[out]--;

                LOG_DEBUG(NETWORK) << "Router " << rid << " unlock "
                                   << GetEnumDirectionType(out);
                LOG_DEBUG(NETWORK)
                    << "  lock tag " << output_lock[out]
                    << ", lock reference " << output_lock_ref[out];


                if (output_lock_ref[out] < 0) {
                    LOG_ERROR(NETWORK)
                        << "Router " << rid << " output reference below zero";
                } else if (output_lock_ref[out] == 0) {
                    output_lock[out] = -1;
                }
            }

            // 发送
            if (out == HOST) {
                buffer_i[i].pop();
                host_buffer_o->emplace(temp);

                flag_trigger = true;
            } else {
                buffer_i[i].pop();
                buffer_o[out].emplace(temp);

                flag_trigger = true;
            }
        }

        // [SIGNALS] 5方向 - 数据信道
        for (int i = 0; i < DIRECTIONS; i++) {
            if (buffer_i[i].size() < MAX_BUFFER_PACKET_SIZE) {
                channel_avail_o[i].write(true);
            } else {
                channel_avail_o[i].write(false);
            }
        }

        // [CTRL SIGNALS] 5方向 - 控制信道
        for (int i = 0; i < DIRECTIONS; i++) {
            if (ctrl_buffer_i[i].size() < MAX_BUFFER_PACKET_SIZE) {
                ctrl_channel_avail_o[i].write(true);
            } else {
                ctrl_channel_avail_o[i].write(false);
            }
        }

        // [SIGNALS] host
        if (host_channel_i) {
            if (host_buffer_i->size() < MAX_BUFFER_PACKET_SIZE) {
                host_channel_avail_o->write(true);
            } else {
                host_channel_avail_o->write(false);
            }
        }

        // trigger again
        if (!reduce_scheduled.empty()) flag_trigger = true;
        if (reduce_stream_engine && !reduce_stream_engine->Drained())
            flag_trigger = true;
        if (flag_trigger)
            need_next_trigger.notify(CYCLE, SC_NS);

        wait();
    }
}

void RouterUnit::trans_next_trigger() {
    while (true) {
        // DAHU notify 0ns
        need_next_trigger.notify(CYCLE, SC_NS);
        wait();
    }
}

void RouterUnit::CountDieRouterPkt(int dir, bool from_c2c) const {
    int d = DieOfGlobal(rid);
    if (d < 0)
        return;
    if (d < (int)g_die_router_pkts.size())
        g_die_router_pkts[d]++;
    g_protocol_progress++; // V2-d2：协议进展（watchdog 据此判断是否停顿）
    // 片内 mesh hop：来自同 die 邻 router 的四向输入（排除跨 die link 入口与本核注入）
    if (!from_c2c && dir != CENTER && d < (int)g_die_mesh_pkts.size())
        g_die_mesh_pkts[d]++;
}

sc_bv<256> RouterUnit::RepinOnC2CIngress(const sc_bv<256> &payload) const {
    Msg m = DeserializeMsg(payload);
    // 只有 core→core 的包参与 C2C pinning；HOST/MEM 端点走各自路径，不碰 exit_port_。
    if (DecodeEndpointType(m.des_) != EP_CORE)
        return payload;
    int old = m.exit_port_;
    // 目的已在本 die → 清除 pin，转片内 XY；否则为**下一跳**重新选出口（锚点=本 die 入口 tile）。
    int repin = (DieOfGlobal(m.des_) == DieOfGlobal(rid))
                    ? -1
                    : CrossDieSelectExit(rid, m.des_, m.source_, m.tag_id_,
                                         m.subflow_);
    g_d2d_repin_total++;
    if (repin == old)
        g_d2d_repin_same++; // 数值巧合（如 3×1 直线两 die 同模板 port id）——仍是一次真实重写
    else
        g_d2d_repin_changed++;
    if (repin == old)
        return payload; // 无需重新序列化
    m.exit_port_ = repin;
    return SerializeMsg(m);
}

long RouterUnit::residual() const {
    const long reduce_match_residual =
        static_cast<long>(reduce_match.Residual());
    long r = static_cast<long>(collective_fork.Residual());
    r += static_cast<long>(strict_multicast_streams.size());
    r += reduce_match_residual + static_cast<long>(reduce_active.size() +
                                                   reduce_scheduled.size());
    if (reduce_stream_engine)
        r += static_cast<long>(reduce_stream_engine->Residual());
    for (int i = 0; i < DIRECTIONS; ++i)
        if (reduce_header_pending[i]) ++r;
    for (int i = 0; i < DIRECTIONS; i++) {
        r += static_cast<long>(ctrl_output_gate[i].Residual());
        // Normally output_lock_ref already accounts for an endpoint owner.
        // Count a stranded owner separately if accounting was corrupted, so
        // drain/watchdog can never report a false zero.
        if (endpoint_output_lock[i].Active() && output_lock_ref[i] <= 0)
            r += static_cast<long>(endpoint_output_lock[i].Residual());
        if (input_lock_ref[i] > 0)
            r += input_lock_ref[i];
        if (output_lock_ref[i] > 0)
            r += output_lock_ref[i];
        r += (long)buffer_i[i].size() + (long)buffer_o[i].size();
        r += (long)ctrl_buffer_i[i].size() + (long)ctrl_buffer_o[i].size();
    }
    if (host_buffer_i)
        r += (long)host_buffer_i->size();
    if (host_buffer_o)
        r += (long)host_buffer_o->size();
    if (host_ctrl_buffer_o)
        r += (long)host_ctrl_buffer_o->size();
    return r;
}

RouterUnit::~RouterUnit() {
    if (reduce_stream_engine)
        coll_refactor::UnregisterProductionReduceStreamEngine(
            static_cast<uint16_t>(rid), reduce_stream_engine.get());
    if (host_buffer_i) {
        delete host_buffer_i;
        delete host_buffer_o;
        delete host_ctrl_buffer_o; 
        delete host_channel_i;
        delete host_channel_o;
        delete host_data_sent_i;
        delete host_data_sent_o;
        delete host_channel_avail_o;
        delete host_ctrl_sent_o;
        delete host_ctrl_channel_o;
    }
}
