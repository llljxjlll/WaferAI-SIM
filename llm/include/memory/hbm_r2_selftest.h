#pragma once
// R2 分布式 HBM 自测入口：验证 BehavioralHBMBackend + FIFO MemEndpointUnit 的带宽/
// 排队/公平性契约。返回失败用例数（0=全过）。
int RunHbmR2SelfTest();
