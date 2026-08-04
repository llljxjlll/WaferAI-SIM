#pragma once
// R1 分布式 HBM 自测入口：合成请求 testbench，驱动真实 SystemC 时序穿过
// CoreMemAdapter/MemEndpointUnit。返回失败用例数（0=全过）。
int RunHbmR1SelfTest();
