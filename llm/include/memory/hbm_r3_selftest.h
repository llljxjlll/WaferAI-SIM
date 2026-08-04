#pragma once
// R3 分布式 HBM 自测入口：验证 DRAMSysHBMBackend 直接访问的正确性/真实时序，以及
// 通过 CoreMemAdapter/MemEndpointUnit 换上 DRAMSys backend 后整条链路仍然工作。
// 返回失败用例数（0=全过）。
int RunHbmR3SelfTest();
