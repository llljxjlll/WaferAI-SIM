# DTE V2a 开发记录

> 版本：V2a（真实 workload 多 SEND 并发）
> 完成日期：2026-07-24
> 状态：已完成并通过验收
> 规划文档：`../DTE建模计划.md`
> 验收清单：`../DTE分版本开发与验收清单.md`

## 1. 版本目标与边界

V2a 将 V1 的阻塞式单 SEND 接入扩展到 dataflow 的 `send_para_logic()`。同一
`send_para_queue` 批次中的多个 `SEND_DATA` 可以背靠背 issue DTE，使
`dte_channel_count` 的 admission 和 launch overlap 在真实 workload trace 中可观测。

本版保持以下边界：

- DATA 仍按原语顺序注入 NoC；
- RECV_DATA 不进入 parallel dispatcher；
- 不支持同一 core 上 SEND 和 RECV 的双向 DTE 并发；
- parallel stripe>1 暂不支持；
- 不实现 source DTE、NoC/D2D、destination DTE 的 streaming overlap；
- V2b 才负责端到端流水模型。

## 2. 现有 parallel 路径梳理

### 2.1 队列所有权

`worker_core_execute()` 会把连续的以下原语从 `prim_queue` 移到
`send_para_queue`：

- `SEND_REQ`；
- `RECV_ACK`；
- `SEND_DATA`；
- `SEND_DONE`。

若 `prim_refill=true`，同一个原语指针同时重新放回 `prim_queue`，供下一轮再次执行。
`send_para_logic()` 只借用这些指针，不删除它们。原语对象仍由既有 primitive stash/
refill 生命周期管理。

实际批次通常是：

```text
SEND_REQ → RECV_ACK → SEND_DATA → SEND_REQ → RECV_ACK → SEND_DATA → ...
```

因此“连续 SEND_DATA”是指同一 parallel 批次中的 DATA 集合，不要求 DATA 指针在物理队列中
相邻。控制原语的处理顺序保持不变。

### 2.2 V2a 前的串行行为

原 `send_para_logic()` 逐条 pop 和执行原语，没有提前 issue DTE，也没有每条 SEND 的
独立 completion 映射。即使名为 parallel，DTE 资源也无法看到同一批多个 pending/active
transfer。

此外，parallel fan-out 使用单一 `send_last_packet` 布尔 token。第一条 DATA 尾包会把它
清零，第二条 DATA 随后永久等待一个不会再次产生的 compute-ready 事件。四目的测试在
DTE off 下也会触发协议 watchdog。

## 3. V2a 批量 DTE 实现

### 3.1 批次预扫描与 issue

每次 `send_para_logic()` 被唤醒后，先复制当前 `send_para_queue` 做只读预扫描：

1. 统计本批 `SEND_DATA` 数量；
2. DTE 开启时，对每条 DATA 调用 `ComputeSendPayloadBits()`；
3. 背靠背 issue `SPM_TO_REMOTE`；
4. 保存 `Send_prim* → (xfer_id, DteTransferContext*)`。

context 本体仍由 `DTEUnit` 内部的：

```cpp
std::list<std::unique_ptr<DteTransferContext>>
```

持有，所以保存的 context 指针在显式 `Release()` 前地址稳定。

### 3.2 独立 completion 与有序提交

批次随后仍按原始顺序处理 REQUEST、ACK 和 DATA。每条 DATA 开始时：

1. 按 `Send_prim*` 查找自己的 xfer 映射；
2. 校验 context 指针和 xfer_id 一致；
3. 若状态尚非 `COMPLETED`，等待该 context 自己的 `done`；
4. 再次确认完成状态；
5. 按 xfer_id `Release()`；
6. 从批次映射删除；
7. 才进入原有 DATA 包发送循环。

等待前先检查状态，避免 DTE 在 REQUEST/ACK 握手期间已经完成、`SC_ZERO_TIME` done 事件已
发生后再盲目 wait 导致永久阻塞。

批末要求 DTE 映射为空，否则抛出逻辑错误。这样可以检测漏等待、漏释放和 completion 串线。

当前共享 bus 的服务顺序与 issue 顺序一致，但 DATA 有序提交不依赖这一事实：即使未来
DTE completion 可以乱序，网络提交仍保持原语顺序。

### 3.3 channel 与共享 bus 语义

V2a 不修改 V0 的资源模型：

- `channel_count` 限制 active transfer；
- 多个 channel 可以同时 launch；
- 所有 channel 共享一个 `bit_width_bits` bus；
- transmit span 不能重叠；
- channel 增多不会乘算峰值 bus 带宽。

## 4. 独立附带修复：Parallel fan-out 尾包死锁

### 4.1 变更分类与影响范围

这是一项既有 shared parallel 代码的缺陷修复，不是 DTE 开关门控下的功能。它应在
changelog 或提交说明中与“DTE V2a 多 SEND 并发”分开列项：

| 项目 | 说明 |
|---|---|
| 影响条件 | `SPEC_SEND_RECV_PARALLEL=true` 且同批至少有两条 `SEND_DATA` |
| DTE 依赖 | 无；`SPEC_USE_BEHA_DTE=false` 时同样生效 |
| 修复前 | 第一条 DATA 消费一次性 compute-ready token，后续 DATA 永久等待 |
| 修复后 | token 在整批最后一条 DATA 尾包提交后才清除 |
| 回归测试 | `parallel DTE-off compatibility`：四条 SEND，1073 ns 完成且无 DTE span |

该无条件行为变更是有意的：它修复 parallel fan-out 的既有死锁，不应被描述成只有打开
DTE 才生效的变化。

### 4.2 实现

批次预扫描同时得到 `batch_data_remaining`。一个 compute-ready token 对同批全部 fan-out
DATA 生效：

- 每条 DATA 的尾包提交时递减计数；
- 只有计数降到零时才清除 `send_last_packet`；
- 批末要求计数为零；
- 下溢或批末残留均抛出明确逻辑错误。

该修复与 DTE 功能正交，使 `DTE off + send_recv_parallel=true` 的多目的 fan-out 从
watchdog 死锁变为正常完成。

## 5. REQUEST metadata 跨轮生命周期

### 5.1 发现的问题

V1 使用单值：

```text
(source, tag) → payload_bits
```

在阻塞式测试中足够，但 parallel pipeline/refill 允许第二轮 REQUEST 在第一轮 RECV_DATA
消费 metadata 前到达。相同 payload 的 `emplace()` 会把两轮合并；第一轮消费并 erase 后，
第二轮 RECV_DATA 报 metadata 缺失。

### 5.2 轮次队列

V2a 改为：

```text
(source, tag) → deque<DteFlowPayloadRound>
```

每轮保存：

- `payload_bits`；
- `stripe_count`；
- 已到达 REQUEST 的 `subflow_mask`。

REQUEST 处理规则：

1. 先校验 subflow 范围；
2. 若队列为空或队尾轮次 stripe 已完整，创建新轮；
3. 同轮 payload 和 stripe_count 必须一致；
4. 同轮 subflow 不得重复；
5. 设置对应 mask bit。

RECV_DATA 完成规则：

1. 每个 completed source 只读取队首轮次；
2. 要求 stripe mask 完整；
3. 求和 payload；
4. pop 队首；
5. 队列为空时删除 key。

因此后一轮可提前排队，但不会被前一轮错误消费。该结构也强化了 stripe 缺失和重复检测。

## 6. 支持矩阵与门禁

| 组合 | V2a 行为 |
|---|---|
| dataflow + DTE + parallel + stripe=1 | 支持 |
| dataflow + DTE off + parallel | 保持既有行为并支持多 fan-out |
| 非 dataflow + DTE + parallel | 配置解析阶段拒绝 |
| dataflow + DTE + parallel + stripe>1 | 运行时明确拒绝 |
| 同核 SEND/RECV DTE 重叠 | 运行时明确拒绝 |

V1 的全局 `DTE && send_recv_parallel` 拒绝已解除。`run_test_dte_v1.py` 的第 13 项由
“必须拒绝”更新为 dataflow parallel 兼容测试，完成时间为 33189 ns，payload 和四阶段 trace
仍满足 V1 契约。

## 7. 测试资产

### 7.1 硬件配置

- `v2_channel1.json`；
- `v2_channel2.json`；
- `v2_channel4.json`。

三者均使用：

```text
gamma = 20 ns
tau_launch = 0 ns
bit_width = 2048 bit
CYCLE = 2 ns
```

16,384-bit transfer 的 transmit 为 16 ns，单 transfer DTE 时间为 36 ns。较长 launch
使 channel admission/overlap 在 trace 中清晰可见。

### 7.2 仿真配置

- `v2_parallel_default.json`：默认硬件上的 parallel DTE 兼容用例；
- `v2_parallel_on.json`：DTE on、parallel on；
- `v2_parallel_off.json`：DTE off、parallel on。

### 7.3 workload

- `v2_parallel_one.json`：1 条 DATA；
- `v2_parallel_two.json`：2 条等长 DATA、不同目的；
- `v2_parallel_four.json`：4 条等长 DATA、不同目的；
- `v2_parallel_mixed_same_dest.json`：4 条长短混合 DATA、同一目的、tag 10..13；
- `v1_repeated_flow.json`：pipeline=2，相同 `(source,tag)` 两轮复用。

### 7.4 统一 runner

`run_test_dte_v2.py` 共 15 项：

- DTE-off parallel 兼容：1 项；
- 1/2/4 SEND × channel=1/2/4：9 项；
- 同目的长短混合 × channel=1/2/4：3 项；
- parallel refill 生命周期：1 项；
- parallel stripe 明确拒绝：1 项。

## 8. 量化结果

### 8.1 等长、不同目的

每条源 transfer 为 16,384 bit，bus service 为 16 ns。

| SEND 数 | channel | 最大 active | 源 DTE 完成偏移（ns） | 共享 bus 总 transmit |
|---:|---:|---:|---|---:|
| 1 | 1 | 1 | 36 | 16 ns |
| 1 | 2 | 1 | 36 | 16 ns |
| 1 | 4 | 1 | 36 | 16 ns |
| 2 | 1 | 1 | 36, 72 | 32 ns |
| 2 | 2 | 2 | 36, 52 | 32 ns |
| 2 | 4 | 2 | 36, 52 | 32 ns |
| 4 | 1 | 1 | 36, 72, 108, 144 | 64 ns |
| 4 | 2 | 2 | 36, 52, 72, 88 | 64 ns |
| 4 | 4 | 4 | 36, 52, 68, 84 | 64 ns |

channel 增多减少 pending/admission 等待并增加 launch overlap；共享 bus 总服务时间严格不变。
本 workload 的 REQUEST/ACK/NoC 位于最终关键路径，因此 channel=1/2/4 的最终 DONE 均为
1125 ns。V2a 的 channel 可观测性以源 DTE trace 为准，不错误宣称最终应用时间必然缩短。

### 8.2 长短混合、同一目的

payload 依次为 16,384、8,192、4,096、2,048 bit，bus service 依次为 16、8、4、2 ns。

| channel | 源 DTE 完成偏移（ns） | 最终 DONE |
|---:|---|---:|
| 1 | 36, 64, 88, 110 | 873 ns |
| 2 | 36, 44, 60, 66 | 873 ns |
| 4 | 36, 44, 48, 50 | 873 ns |

四个目的 RECV_DATA 位于同一 core，payload 均正确；`FLOW_DONE` 顺序固定为 tag
10→11→12→13，证明网络提交保持原语顺序。

### 8.3 Parallel refill

```text
DTE off = 867 ns
DTE on  = 951 ns
```

两轮各产生：

- core 0：16,384-bit source transfer；
- core 1：16,384-bit source transfer；
- core 2：32,768-bit destination transfer。

三个 core 的 xfer id 都是 0、1，证明每轮使用新 context；第二轮 REQUEST 可以提前排队并在
第一轮消费后正确成为队首。

## 9. 回归结果

```text
cmake --build build --target npusim -j2                         PASS
./build/npusim --dte-v0-selftest                               55/55
python3 llm/test/dte/run_test_dte_v0.py                         PASS
python3 llm/test/dte/run_test_dte_v1.py                         13/13
python3 llm/test/dte/run_test_dte_v2.py                         15/15
./build/npusim --d2d-v0-selftest                               308/308
python3 llm/test/d2d_link/run_test_d2d_v0.py                    67/67 groups
python3 llm/test/d2d_link/run_test_d2d_v5.py                    23/23 groups
python3 llm/test/noc_congestion/run_test_noc_congestion.py       4/4
```

NoC frozen baseline 保持：

| 场景 | behavioral | physical |
|---|---:|---:|
| no congestion | 14781 ns | 29109 ns |
| congestion | 14833 ns | 45441 ns |

## 10. 交付文件

生产代码：

- `llm/include/workercore/workercore.h`；
- `llm/src/workercore/logic.cpp`；
- `llm/src/utils/config_utils.cpp`。

自测与集成测试：

- `llm/src/dte/v0_selftest.cpp`；
- `llm/test/dte/run_test_dte_v1.py`；
- `llm/test/dte/run_test_dte_v2.py`；
- `llm/test/dte/hardware/v2_channel*.json`；
- `llm/test/dte/simulation/v2_parallel*.json`；
- `llm/test/dte/workload/v2_parallel*.json`。

## 11. 遗留事项与 V2b 入口

- parallel stripe>1 仍未实现；如后续需要，应把每条 DATA 的多个 subflow 与 DTE 整 flow
  context 明确绑定，而不是按 stripe 重复 issue DTE。
- 同核 SEND/RECV 双向 DTE 并发仍未实现；若 workload 确有需求，需要统一通信 dispatcher
  和显式方向仲裁。
- V2a 只模拟源 DTE 多 SEND admission/launch overlap；端到端仍是 store-and-forward。
- V2b 需要引入首包/末包或闭式 pipeline 模型，组合 source DTE、NoC/D2D 和 destination
  DTE 的瓶颈带宽，避免整块时间无条件相加。

V2a 已满足进入 V2b 的条件，但 V2b 是独立的时序模型变更，应单独开发和验收。
