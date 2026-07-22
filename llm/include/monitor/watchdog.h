#pragma once
// V2-d2：协议进展 watchdog（**仿真器内部**主动诊断，不依赖测试框架的 wall-clock 超时）。
//
// 动机：Python runner 的 subprocess timeout 只能把「永久挂起」变成测试失败，无法区分协议
// 依赖环 / 路由丢包 / 网络残留，也拿不到等待状态。本模块在仿真内部维护「最后一次协议进展
// 时间」，在仍有未完成流量却连续多周期无进展时主动 dump 等待状态并让仿真非零退出。
//
// 进展定义（任一即视为有进展）：router 入口收到包、D2D link 搬运包、HOST 收到 DONE。
// 判据：`当前 cycle - last_progress_cycle > 阈值` 且仿真尚未结束 ⇒ 判定协议停顿。
// 阈值需大于任何合法的「计算中、无包移动」间隔；默认取足够宽的值，可由 spec 配置覆盖。
#include "systemc.h"

// 任何协议进展都会 +1（router 入口 / link 搬运 / DONE）。
extern long g_protocol_progress;
// 触发结果（供 npusim 决定退出码；仿真正常完成时保持 false/-1）。
extern bool g_protocol_stall_detected;
extern long g_protocol_stall_cycle;     // 判定停顿的 cycle
extern long g_protocol_last_progress;   // 最后一次观察到进展的 cycle
// 停顿判定阈值（cycle）。<=0 表示禁用 watchdog。
extern long g_protocol_watchdog_cycles;

void ResetProtocolWatchdog();

// 每 cycle 采样进展计数；超过阈值仍无进展则 dump 诊断并 sc_stop()。
class ProtocolWatchdog : public sc_module {
public:
    SC_HAS_PROCESS(ProtocolWatchdog);
    explicit ProtocolWatchdog(const sc_module_name &n);
    void monitor();
};

// 停顿时的诊断 dump：逐 router 打印被持有的 output lock、各方向 buffer 队首消息头
// （source / tag / dest / phase=msg_type）与 wait_reason，以及 router/link 残留量。
// 与 watchdog 解耦，便于单独测试与复用。
void DumpProtocolWaitState(long cycle);
