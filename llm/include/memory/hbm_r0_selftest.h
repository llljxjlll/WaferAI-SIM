#pragma once
// R0 分布式 HBM 自测入口。纯配置/结构测试，不建 SystemC 仿真。返回失败用例数
// （0=全过），供 npusim.cpp 的 --hbm-r0-selftest 分支使用。
int RunHbmR0SelfTest();
