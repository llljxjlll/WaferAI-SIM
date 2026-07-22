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
#include "macros/macros.h"
#include <deque>
#include <utility>

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

    SC_HAS_PROCESS(D2DLinkUnit);
    D2DLinkUnit(const sc_module_name &n, int latency_);
    void forward();

    // V1 drain 不变量：仿真正常完成后数据/控制 FIFO 均须为空。
    long residual() const { return (long)fifo_.size() + (long)cfifo_.size(); }

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
