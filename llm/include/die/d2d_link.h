#pragma once
// V1 D2D Link 单元（单向）：功能性 FIFO + 固定 latency（1 packet/cycle，不饱和）。
// 从上游 router 边缘出链读 (channel, sent)，延迟 latency 拍后驱动下游 router 边缘入链
// (channel, sent)；对上游驱动 avail=true（V1 不饱和，恒可接收，无背压）；成熟包在队首等待
// 直到下游 out_avail=true 才交付（honor 下游容量，不丢不重不越容量）。数据/控制各一条独立 FIFO。
// 一条相邻 die 的双向 D2D link = 两个此单元（A→B、B→A）。
//
// latency 语义（已由 `--d2d-link-selftest` 实测）：Link 在 capture 之上**相对**加 latency 拍，
// 相对量精确（L=0/1/7/20 逐一验证）。绝对交付周期还含嵌入环境的 delta-cycle 调度偏移——独立
// 探针里实测为 +2（link 先 wait 再采样 +1、探针 registered 读 +1）；真实 router↔link↔router 的
// 端到端 latency 由 V1-d 标定。
#include "systemc.h"
#include "defs/const.h"
#include "die/port.h"
#include "macros/macros.h"
#include <deque>
#include <map>
#include <utility>

// 有限缓冲 + token-bucket 速率的可选配置。默认 disabled：functional_v2 走 forward() 的冻结分支。
// enabled 时由 whole_flow_saf 分流：false 是 V3-b standalone 单级 credit 模型；true 是 V3-d 生产
// SAF→inflight→RX 模型。两种有限模型有意独立，分别由 Link self-test 的对应探针验证。
struct D2DLinkBound {
    bool enabled = false;
    // whole_flow_saf=false：V3-b standalone 的单级有限 FIFO（保持既有测试语义）。
    // whole_flow_saf=true：V3-d 生产流水线：SAF stage → port/link 限速 → inflight → RX stage。
    bool whole_flow_saf = false;
    int data_depth = 0; // link 在途 DATA FIFO 容量（生产取 link_inflight_depth）
    int ctrl_depth = 0; // 独立控制 FIFO 容量
    int saf_depth = 0;  // 整流 stage 容量；必须能容纳完整 flow
    int rx_depth = 0;   // 远端接收 stage 容量
    D2DRate port_rate;  // SAF stage → link 的端口注入速率
    D2DRate rate;       // 链路速率（包/cycle，0<r<=1）；token bucket 表达 <1
};

// V4 Behavioral：无有限 FIFO/credit/backpressure/跨 flow 争用。每个逻辑 DATA flow
// 只让一个代表包穿过 Router；首条有向 link 额外承担 ceil(F/min(port,link)) 的聚合服务。
struct D2DLinkBehavioral {
    bool enabled = false;
    D2DRate port_rate;
    D2DRate link_rate;
};

class D2DLinkUnit : public sc_module {
public:
    // 上游(A)侧：读其边缘输出 (channel/sent)，驱动其边缘输入 avail
    sc_in<sc_bv<256>> in_channel;
    sc_in<bool> in_sent;
    sc_out<bool> in_avail;
    sc_in<sc_bv<256>> in_ctrl_channel;
    sc_in<bool> in_ctrl_sent;
    sc_out<bool> in_ctrl_avail;
    // 下游(B)侧：驱动其边缘输入 (channel/sent)，读其边缘 avail 输出
    sc_out<sc_bv<256>> out_channel;
    sc_out<bool> out_sent;
    sc_in<bool> out_avail;
    sc_out<sc_bv<256>> out_ctrl_channel;
    sc_out<bool> out_ctrl_sent;
    sc_in<bool> out_ctrl_avail;

    // V3-b2：**信用回还接口**（真实模块信号，非 testbench 窥探）。link 每释放一个在途 FIFO 空位
    // 就在对应通道 pulse 一拍，经**回程 latency**（同物理链路 L）后到达上游；上游据此 credit++。
    // 这是 bounded 唯一流控真源：上游初始 credit=depth，credit>0 才发、发后 -1、收到 return +1。
    // functional_v2 无背压，恒 false。data/ctrl 各一路，独立记账。
    sc_out<bool> data_credit_return;
    sc_out<bool> ctrl_credit_return;

    int latency; // 可编程链路延迟；信用 RTT 由 D2DCreditRttCycles() 定义（含 clocked 最小服务拍）
    // V2-c：本单元对应的有向 link 在 g_d2d_links / g_d2d_link_stats 中的下标（-1=未归因）。
    int link_idx;

    D2DLinkBound bound; // V3-b：有限缓冲 + 速率（默认 disabled = V2 行为，不改冻结时序/计数）

    D2DLinkBehavioral behavioral; // V4：解析后端；与 bound 互斥
    // V3-b 统计（仅 bounded 时有意义）。命名严格区分「状态」与「阻塞事件」：
    long occ_max = 0;       // 数据 FIFO 观测到的最大占用（不变量：<= data_depth）
    long occ_ctrl_max = 0;  // 控制 FIFO 最大占用（<= ctrl_depth）
    long full_cycles = 0;   // 数据 FIFO **处于满状态** 的拍数（纯状态，与上游是否想发无关）
    long ctrl_full_cycles = 0;       // 控制 FIFO 满状态拍数
    long upstream_blocked = 0;       // 数据：满时仍收到 in_sent（信用违约/记账错误，合法上游应恒为 0）
    long ctrl_upstream_blocked = 0;  // 控制：同上
    long rate_stall = 0;    // 有成熟 DATA、下游 ready、但 token 不足未发的拍数
    long downstream_stall = 0;       // 有成熟 DATA、但下游 out_avail=false 的拍数
    long ctrl_downstream_stall = 0;  // 有成熟 CTRL、但下游 out_ctrl_avail=false 的拍数
    long tokens = 0;        // link-rate token
    long port_tokens = 0;   // V3-d port-rate token
    long saf_occ_max = 0, inflight_occ_max = 0, rx_occ_max = 0;
    long saf_full_cycles = 0, inflight_full_cycles = 0, rx_full_cycles = 0;
    long port_rate_stall = 0, link_rate_stall = 0, rx_backpressure_stall = 0;

    SC_HAS_PROCESS(D2DLinkUnit);
    D2DLinkUnit(const sc_module_name &n, int latency_, int link_idx_ = -1,
                D2DLinkBound bound_ = D2DLinkBound{},
                D2DLinkBehavioral behavioral_ = D2DLinkBehavioral{});
    void forward();
    void forward_bounded(long cyc); // V3-b standalone 单级有限 FIFO
    void forward_bounded_saf(long cyc); // V3-d 生产 whole-flow SAF 多级流水线

    // drain 不变量：bounded 模式除数据/控制 FIFO 外，回程中的信用及当前输出 pulse 也必须清空。
    void forward_behavioral(long cyc); // V4 聚合代表消息
    long residual() const {
        long saf_packets = 0;
        for (const auto &kv : saf_flows_)
            saf_packets += (long)kv.second.packets.size();
        return (long)fifo_.size() + (long)cfifo_.size() +
               (long)rx_fifo_.size() + saf_packets +
               (long)saf_expected_.size() + CreditResidual() +
               (long)behavioral_data_events_.size() +
               (long)behavioral_ctrl_events_.size();
    }
    long CreditResidual() const {
        return (long)data_credit_due_.size() + (long)ctrl_credit_due_.size() +
               (data_credit_active_ ? 1L : 0L) +
               (ctrl_credit_active_ ? 1L : 0L);
    }

    // V3-b 只读观测（仅供仿真结束后断言，禁止当同步信号跨线程读）
    long OccMax() const { return occ_max; }
    long OccCtrlMax() const { return occ_ctrl_max; }
    long FullCycles() const { return full_cycles; }
    long CtrlFullCycles() const { return ctrl_full_cycles; }
    long UpstreamBlocked() const { return upstream_blocked; }
    long CtrlUpstreamBlocked() const { return ctrl_upstream_blocked; }
    long RateStall() const { return rate_stall; }
    long DownstreamStall() const { return downstream_stall; }
    long CtrlDownstreamStall() const { return ctrl_downstream_stall; }
    long DataOcc() const { return (long)fifo_.size(); }
    long CtrlOcc() const { return (long)cfifo_.size(); }
    long SafOccMax() const { return saf_occ_max; }
    long InflightOccMax() const { return inflight_occ_max; }
    long RxOccMax() const { return rx_occ_max; }
    long SafFullCycles() const { return saf_full_cycles; }
    long InflightFullCycles() const { return inflight_full_cycles; }
    long RxFullCycles() const { return rx_full_cycles; }
    long PortRateStall() const { return port_rate_stall; }
    long LinkRateStall() const { return link_rate_stall; }
    long RxBackpressureStall() const { return rx_backpressure_stall; }

private:
    struct SafFlowBuffer {
        int expected = 0;
        bool complete = false;
        std::deque<sc_bv<256>> packets;
    };
    // REQUEST 先于 DATA 穿过同一路径：每条 link 记住该 flow 的 F。DATA 按 FlowKey 整流，
    // 只有见到尾包且 count==F 后才进入 saf_ready_，随后才允许注入物理 link。
    std::map<FlowKey, int> saf_expected_;
    std::map<FlowKey, SafFlowBuffer> saf_flows_;
    std::deque<FlowKey> saf_ready_;
    std::deque<sc_bv<256>> rx_fifo_;

    // 只存真实包 {ready_cycle, payload}（不每周期存 bubble）。ready_cycle=capture_cycle+latency；
    // 队首成熟(ready<=当前 cycle)且下游 ready(out_avail=true) 才出队——成熟包在 Link 中等待
    // 直到下游可收，不丢包、不越过下游容量。
    std::deque<std::pair<long, sc_bv<256>>> fifo_;  // 数据
    std::deque<std::pair<long, sc_bv<256>>> cfifo_; // 控制
    // V3-b2：信用回还的到达时刻队列（pop 时 push cyc+latency，模拟回程 L 拍；每拍 pulse 一个到期的）。
    std::deque<long> data_credit_due_;
    std::deque<long> ctrl_credit_due_;
    // Behavioral 不把聚合服务占用解释成有限队列；事件表仅保存代表消息的到达时刻。
    std::multimap<long, sc_bv<256>> behavioral_data_events_;
    std::multimap<long, sc_bv<256>> behavioral_ctrl_events_;

    // sc_signal pulse 在写出后的一个调度周期内仍是在途信用；纳入 residual，避免过早宣布 drain。
    bool data_credit_active_ = false;
    bool ctrl_credit_active_ = false;
    // Production control credit uses a toggling event bit: every delivery flips it, so
    // consecutive-cycle returns cannot collapse into one level-sensitive pulse.
    bool data_credit_toggle_ = false;
    bool ctrl_credit_toggle_ = false;
};

// V1-b2 独立 SystemC link 测试（驱动真实包）：latency=0/1/7/20、FIFO 序、无丢/重、
// out_avail 停顿、drain 后 in==out、data/ctrl 独立、idle 统计 0。返回失败数（0=全过）。
int RunD2DLinkSelfTest();
