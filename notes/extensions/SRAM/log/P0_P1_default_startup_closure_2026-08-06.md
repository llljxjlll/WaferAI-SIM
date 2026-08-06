# SRAM P0/P1 默认启动与 legacy 基线闭环记录

日期：2026-08-06
状态：PASS；最终验收条件 7 已关闭

## 范围

本轮处理最终验收遗留的两个 P0/P1 问题：README 引用的默认配置不存在，以及扩展关闭时 `playground.json` 被 rendezvous watchdog 提前终止。工作包含开发、测试、评审、评审后修改和最终复测。

## 根因

1. CLI 和 README 的默认路径长期指向仓库中不存在的 `gpu/pd_serving.json`、`core_4x4.json` 和 `default_mapping.txt`，启动前也没有统一文件/JSON/schema 预检。
2. host 侧把 S_DATA 写入 Router 视为“输入发送完成”，没有等待目的核消费和完成其建模服务。
3. watchdog 只观察包移动。behavioral S_DATA 的 26,215-cycle 服务和 `matmul_forward_pd` 的 668,467-cycle 计算都是有限工作，却超过固定 20,000-cycle 阈值，因而被误判为 rendezvous 停滞。

## 开发

- 新增 `llm/test/default/{workload.json,hardware.json,simulation.json,mapping.spec}`，`npusim` 无参数默认指向源码树内这组文件；README 中英文 quick start 改为可直接执行的 `./npusim`。
- 新增启动 preflight：先检查四个文件存在、JSON 可解析、基础字段和 mapping 行格式，再进入全局配置初始化。dataflow 执行 rendezvous 合约；`sched_pd`、`sched_pds` 和 GPU 等非 dataflow 模式保留原 schema。
- 新增 S_DATA 六阶段 tracker：`enqueued → injected → router accepted → delivered → consumed → completed`。每条消息强制前驱顺序，host 只有在本轮全部 S_DATA 到达 `completed` 后才结束 `Send Input Data`。
- 新增有限工作 planned-idle lease。behavioral send/recv 和计算原语仅按其精确建模周期暂停 watchdog；期限结束立即恢复检测。重复 owner 会报错，lease 正常结束计为一次协议进展。
- 新增 dataflow rendezvous 静态校验：活动核唯一、source 目标有效、RECV_START source 数精确相等、普通 recv 的 tag/数量有声明发送方，并要求存在 host source、零接收 work 或 collective 之一作为 runnable root。合法的无 source collective 保持可运行。
- 新增纯函数 rendezvous 自测、无参数 smoke、同一 host lane 四源长 S_DATA fixture 和完整 playground CTest 条目；CTest 不可用时可直接执行相同入口。

## 测试

| 项目 | 结果 |
|---|---|
| `cmake --build build --target npusim -j2` | PASS |
| rendezvous 纯函数正/负例 | 6/6 PASS |
| 无参数默认 workload | 112 ns；S_DATA 1/1/1/1/1/1；DONE 1 |
| 4×4 同 lane、四源 2 Mi-element S_DATA | 262398 ns；S_DATA 4/4/4/4/4/4；`per_lane_done=4,0,0,0` |
| 完整 `playground.json`，SRAM 扩展关闭 | 7270718 ns；DONE 4；S_DATA 4/4/4/4/4/4；router/D2D residual 0；credit balanced |
| SRAM R0–R6 | 7/7 PASS |
| HBM R0–R4 | 5/5 PASS |
| DTE V0/V3/V3b/V4 | 64/64、31/31、21/21、19/19 PASS |
| SRAM pipeline/DRAMSys/legacy/NUMA/compat | 5/5 runner PASS；checksum 130560；NUMA D2D 672/672 |
| source-free collective V1 | 24/24 PASS |
| D2D V0/link/多跳/legacy single-die | 67/67 groups PASS；L0 308/308 |
| `git diff --check` | PASS |

环境中未安装 `ctest`，因此本轮直接运行 `npusim` 自测入口和 Python runner；CMake 测试定义仍已生成，安装 CTest 的环境可直接执行。

## 评审与修改

1. 初版只给 RECV_START 增加长等待 lease；复跑后在 668,467-cycle 计算处再次误报。修改为所有有限长 behavioral send/recv 和 compute wait 使用同一精确 lease 机制。
2. 初版四源 fixture 使用 2×2，DONE 分散到两条 host lane。改为 4×4、核 0–3 同行，并冻结 `per_lane_done=4,0,0,0`。
3. 初版 preflight 对所有模式要求 `vars/source`，会破坏 `sched_pd/sched_pds`。修改为仅 dataflow 执行 rendezvous 校验。
4. 初版要求 dataflow 至少一个 source，会误拒合法的 source-free collective。修改为 runnable-root 合约，并用 collective V1 24/24 回归验证。
5. 初版 S_DATA 只汇总阶段计数，可能由一条漏记和另一条重复记造成假平衡。修改为按消息 key 强制阶段前驱顺序。
6. D2D 的故意 rendezvous-cycle fixture 原期待 runtime watchdog 退出 3；该图现在可在启动期静态判定。runner 更新为要求 preflight 退出 2、不得进入仿真，健康多跳流仍要求无 watchdog。

## 结论

README 默认启动、S_DATA 生产完成语义、有限工作 watchdog 语义、静态 rendezvous 校验和完整 legacy playground 均已闭环。SRAM 最终验收条件 7 与此前已完成的条件 1–6、8 一并关闭。
