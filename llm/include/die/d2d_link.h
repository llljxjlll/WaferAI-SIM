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
#include <utility>

// V3-b：有限缓冲 + token-bucket 速率的可选配置。**默认 disabled**——生产路径（functional_v2）恒为
// disabled，走 forward() 的 functional 分支（该分支逐字节保留 V2 逻辑，冻结时序/计数精确不变）；
// bounded 逻辑全在独立的 forward_bounded()，仅 link self-test 显式开启。接入生产属 V3-d。
struct D2DLinkBound {
    bool enabled = false;
    int data_depth = 0; // 数据 FIFO 容量（enabled 时必须 >=1）
    int ctrl_depth = 0; // 控制 FIFO 容量
    D2DRate rate;       // 链路交付速率（包/cycle，0<r<=1）；token bucket 表达 <1
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

    int latency; // 固定链路延迟（cycle）
    // V2-c：本单元对应的有向 link 在 g_d2d_links / g_d2d_link_stats 中的下标（-1=未归因）。
    int link_idx;

    D2DLinkBound bound; // V3-b：有限缓冲 + 速率（默认 disabled = V2 行为，不改冻结时序/计数）

    // V3-b 统计（仅 bounded 时有意义）。命名严格区分「状态」与「阻塞事件」：
    long occ_max = 0;       // 数据 FIFO 观测到的最大占用（不变量：<= data_depth）
    long occ_ctrl_max = 0;  // 控制 FIFO 最大占用（<= ctrl_depth）
    long full_cycles = 0;   // 数据 FIFO **处于满状态** 的拍数（纯状态，与上游是否想发无关）
    long ctrl_full_cycles = 0;       // 控制 FIFO 满状态拍数
    long upstream_blocked = 0;       // 数据：in_sent 想发 **且** 已公布 ready=false（确有发送需求被挡）
    long ctrl_upstream_blocked = 0;  // 控制：同上
    long rate_stall = 0;    // 有成熟 DATA、下游 ready、但 token 不足未发的拍数
    long downstream_stall = 0;       // 有成熟 DATA、但下游 out_avail=false 的拍数
    long ctrl_downstream_stall = 0;  // 有成熟 CTRL、但下游 out_ctrl_avail=false 的拍数
    long tokens = 0;        // token 累加器（信用；发一包扣 rate.den；发后 cap 至 den，限突发≤1 包）

    SC_HAS_PROCESS(D2DLinkUnit);
    D2DLinkUnit(const sc_module_name &n, int latency_, int link_idx_ = -1,
                D2DLinkBound bound_ = D2DLinkBound{});
    void forward();
    void forward_bounded(long cyc); // V3-b：有限缓冲 + 速率的每拍逻辑（bound.enabled 时）

    // V1 drain 不变量：仿真正常完成后数据/控制 FIFO 均须为空。
    long residual() const { return (long)fifo_.size() + (long)cfifo_.size(); }

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

private:
    // 只存真实包 {ready_cycle, payload}（不每周期存 bubble）。ready_cycle=capture_cycle+latency；
    // 队首成熟(ready<=当前 cycle)且下游 ready(out_avail=true) 才出队——成熟包在 Link 中等待
    // 直到下游可收，不丢包、不越过下游容量。
    std::deque<std::pair<long, sc_bv<256>>> fifo_;  // 数据
    std::deque<std::pair<long, sc_bv<256>>> cfifo_; // 控制
};

// V1-b2 独立 SystemC link 测试（驱动真实包）：latency=0/1/7/20、FIFO 序、无丢/重、
// out_avail 停顿、drain 后 in==out、data/ctrl 独立、idle 统计 0。返回失败数（0=全过）。
int RunD2DLinkSelfTest();
