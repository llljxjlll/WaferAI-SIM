# DTE V1 开发记录

> 版本：V1（阻塞式 workload 接入）
> 完成日期：2026-07-24
> 状态：已完成并通过验收
> 后续状态：V2a 已开放经过验收的 dataflow parallel；本记录中的全局拒绝描述仅表示 V1 交付时边界。
> 规划文档：`../DTE建模计划.md`
> 验收清单：`../DTE分版本开发与验收清单.md`

## 1. 版本目标与边界

V1 将 V0 的独立 `DTEUnit` 接入真实 `WorkerCoreExecutor` 通信路径，使普通
`SEND_DATA` 和 `RECV_DATA` 的 DTE 延迟进入 trace 和最终 workload 完成时间。

本版采用保守的 store-and-forward 顺序：

```text
source DTE complete
  → NoC/D2D request、ack 和 DATA 完成
  → destination DTE complete
  → RECV_DATA 原语完成
```

V1 不实现流式重叠，也不接入 `send_para_logic()`。在 V1 交付时，DTE 与
`noc.send_recv_parallel=true` 同时开启会在配置解析阶段直接拒绝。该阶段性限制已由
V2a 的 dataflow parallel 实现取代；非 dataflow 和未支持组合仍保持门禁。

## 2. 实现内容

### 2.1 每核 DTE 实例

`WorkerCoreExecutor` 新增一个独占的 `std::unique_ptr<DTEUnit>`。构造时从本核
`CoreHWConfig` 读取：

- `dte_channel_count`；
- `dte_bit_width`。

全局 launch 参数来自：

- `dte.gamma_ns`；
- `dte.tau_launch_avg_ns`。

SystemC module 名称使用 `dte_core_<cid>`，因此不同 core 的 pending、active channel、
共享 bus 和 transfer id 均相互独立。

### 2.2 源端 SEND_DATA

普通 `send_logic()` 在第一个 DATA 包进入 NoC 之前执行：

```cpp
DteTransferContext &xfer =
    dte->Issue(ComputeSendPayloadBits(*prim), DteDir::SPM_TO_REMOTE);
wait(xfer.done);
dte->Release(xfer.xfer_id);
```

等待完成后才进入原有逐包或 behavioral 代表包发送流程。`SEND_REQ` 和 `SEND_DONE`
不支付 DTE 延迟，原语结束仍只通过既有 `ev_block` 通知一次。

### 2.3 精确 payload 元数据传播

目的端不能从收到的 DATA 消息可靠恢复整个逻辑 flow 的 payload：

- physical NoC 会看到多包；
- behavioral NoC 可能只看到代表包；
- `HW_NOC_PAYLOAD_PER_CYCLE` 会压缩模拟包数量；
- stripe 会把一个逻辑 flow 拆成多个 subflow；
- 多 source 会汇聚到同一个 `RECV_DATA`。

V1 因此在生成配置时，把 `SEND_DATA` 的 `max_packet`、`end_length`、
`packet_scale` 和 `packets_in_last_group` 复制到配对 `SEND_REQ`。源端发送 REQUEST 时，
使用与 SEND_DATA 相同的 `ComputeSendPayloadBits()` 计算完整逻辑 payload。

该 64-bit bit 数编码在 REQUEST 不承载业务数据的 `data_[127:64]`。每个 stripe 的
REQUEST 重复携带相同声明，目的端以 `(source, tag)` 为 key 去重，并校验重复声明一致。
这避免了按 stripe 重复计费和 behavioral 代表包少算。

`AttachRequestDtePayload()` 将这一步严格门控在 DTE 开启路径。DTE 关闭时 helper 为
no-op，不调用 `ComputeSendPayloadBits()`，因此退化或历史 `Send_prim` 不会因为未启用的
DTE 新增校验而改变功能或异常行为；DTE 开启时仍执行完整合法性校验。

### 2.4 目的端 RECV_DATA

REQUEST 到达时记录 `(source, tag) → payload_bits`。既有 subflow 逻辑仍负责检查：

- subflow 范围；
- stripe 完整性；
- 重复尾包；
- source 数量和 `recv_cnt` 一致性。

仅当所有预期 DATA subflow 到齐后，目的端：

1. 从 `ended_subflows` 得到完成的唯一 source 集合；
2. 查询并删除每个 `(source, tag)` 的 payload 声明；
3. 进行 `uint64_t` 溢出检查后求和；
4. issue `REMOTE_TO_SPM`；
5. 等待 transfer 完成并释放 context；
6. 结束 `RECV_DATA` 原语。

因此多 source 场景只产生一次目的端 transfer，其 payload 为所有 source 逻辑 payload
之和。`RECV_ACK`、`RECV_CONF` 等控制原语不经过 DTE。

### 2.5 配置门禁

以下组合在 `ParseSimulationConfig()` 中抛出 `std::invalid_argument`：

```text
dte.use_beha_dte=true
noc.send_recv_parallel=true
```

上述是 V1 交付时门禁。V2a 完成后，dataflow parallel 已开放；非 dataflow 在启动期
拒绝，parallel stripe>1 和同核 SEND/RECV DTE 重叠在运行时拒绝。DTE 默认仍为关闭状态。

## 3. 测试资产

### 3.1 配置

- `llm/test/dte/hardware/v1.json`：单 die、统一 2048-bit DTE。
- `llm/test/dte/hardware/v1_cross_die.json`：2×1 die 与 bounded-SAF D2D。
- `llm/test/dte/hardware/v1_heterogeneous.json`：core 0/1/2 分别使用
  channel 1/4/2 和位宽 2048/1024/4096 bit。
- `llm/test/dte/simulation/v1_cycle_off.json`。
- `llm/test/dte/simulation/v1_cycle_on.json`。
- `llm/test/dte/simulation/v1_beha_off.json`。
- `llm/test/dte/simulation/v1_beha_on.json`。
- `llm/test/dte/simulation/v2_parallel_default.json`（V2a 后替代原 V1 非法组合样例）。

### 3.2 workload

- `gemm_no_congestion.json`：stripe=1，8 条源端和 8 条目的端 transfer。
- `v1_cross_die_stripe2.json`：跨 die stripe=2，同时包含同 die通信。
- `v1_cross_die_stripe4.json`：跨 die stripe=4，同时包含同 die通信。
- `v1_multi_source.json`：两个 source 汇聚到一个 `recv_cnt=2` 的目的端。
- `v1_repeated_flow.json`：pipeline=2，两轮复用相同的两个 `(source, tag)`。

### 3.3 统一运行器

`llm/test/dte/run_test_dte_v1.py` 会：

- 运行 physical/behavioral NoC 的 DTE on/off 矩阵；
- 校验冻结完成时间；
- 解析 trace 中 pending、launch、bus_wait、transmit 的 B/E span；
- 校验 payload、方向、core、阶段次序和精确持续时间；
- 验证不同 core 的 transmit span 可以重叠；
- 验证 source DTE 完成后 destination DTE 才开始；
- 验证跨 die 的 REQUEST/ACK/DATA 实际经过 D2D；
- 验证 stripe=2/4 不改变逻辑 payload；
- 验证多 source 在目的端精确求和；
- 验证 pipeline/refill 两轮复用同一 `(source, tag)` 时 metadata 可被消费并重新插入；
- 验证异构 per-core 位宽实际生效；
- V1 交付时验证非法 parallel 组合；V2a 后 runner 改为验证 dataflow parallel 兼容。

## 4. 量化结果

测试配置中 `CYCLE=2 ns`、launch 为 `4+2=6 ns`。

### 4.1 标准 workload

每条逻辑 flow 为 2,097,152 bit，DTE 位宽为 2048 bit：

```text
transmit = ceil(2,097,152 / 2,048) × 2 ns = 2,048 ns
single endpoint DTE = 6 + 2,048 = 2,054 ns
end-to-end DTE delta = 2,054 + 2,054 = 4,108 ns
```

| NoC | DTE off | DTE on | 差值 |
|---|---:|---:|---:|
| physical | 29109 ns | 33217 ns | 4108 ns |
| behavioral | 14781 ns | 18889 ns | 4108 ns |

两种 NoC 得到完全相同的 DTE payload 和新增时间，说明 behavioral 代表包没有少算。

### 4.2 跨 die 和 stripe

跨 die flow 的逻辑 payload 为 16,384 bit：

```text
single endpoint DTE = 6 ns + ceil(16,384 / 2,048) × 2 ns = 22 ns
end-to-end DTE delta = 44 ns
```

| stripe | DTE off | DTE on | 差值 |
|---:|---:|---:|---:|
| 2 | 656 ns | 700 ns | 44 ns |
| 4 | 652 ns | 696 ns | 44 ns |

stripe 只影响协议消息和 NoC/D2D 调度，不改变 DTE 的整 flow payload。

### 4.3 多 source

core 0 和 core 1 各发送 16,384 bit，core 2 的目的端 DTE 接收 32,768 bit。完成时间：

```text
DTE off = 623 ns
DTE on  = 683 ns
delta   = 60 ns
```

trace 中只出现两个源端 descriptor 和一个 32,768-bit 目的端 descriptor。

### 4.4 重复 flow 生命周期

`v1_repeated_flow.json` 使用 pipeline=2，在同一模拟中让 core 0/1 连续两轮以相同 tag
向 core 2 发送。结果：

```text
DTE off = 1033 ns
DTE on  = 1115 ns
```

trace 共包含 6 条 transfer：每轮两个 16,384-bit `SPM_TO_REMOTE` 和一个聚合后的
32,768-bit `REMOTE_TO_SPM`。第二轮只会在第一轮目的端完成、对应 map entry 已消费后开始，
验证相同 `(source, tag)` 能重新插入并再次消费。

### 4.5 异构 per-core 配置

同一多 source workload 使用：

| core | channel_count | bit_width | 方向 | payload | DTE 时间 |
|---:|---:|---:|---|---:|---:|
| 0 | 1 | 2048 bit | SPM_TO_REMOTE | 16384 bit | 22 ns |
| 1 | 4 | 1024 bit | SPM_TO_REMOTE | 16384 bit | 38 ns |
| 2 | 2 | 4096 bit | REMOTE_TO_SPM | 32768 bit | 22 ns |

不同 core 的位宽分别生效。V1 每核只有阻塞式单 transfer，按模型契约，channel 数不会改变
单 transfer 延迟；channel admission 的 workload 并发效应属于 V2a。

## 5. 回归结果

执行并通过：

```text
cmake --build build --target npusim -j2
./build/npusim --dte-v0-selftest                 55/55（V2a 后）
python3 llm/test/dte/run_test_dte_v0.py           PASS
python3 llm/test/dte/run_test_dte_v1.py           13/13 PASS
./build/npusim --d2d-v0-selftest                 308/308
python3 llm/test/noc_congestion/run_test_noc_congestion.py  4/4
python3 llm/test/d2d_link/run_test_d2d_v0.py      67/67 groups
python3 llm/test/d2d_link/run_test_d2d_v3.py      16/16 groups
python3 llm/test/d2d_link/run_test_d2d_v4.py      13/13 groups
python3 llm/test/d2d_link/run_test_d2d_v5.py      23/23 groups
```

NoC frozen baseline 保持不变：

| 场景 | behavioral | physical |
|---|---:|---:|
| no congestion | 14781 ns | 29109 ns |
| congestion | 14833 ns | 45441 ns |

DTE 关闭时 trace 中没有 DTE span。54 项自测中的退化 SEND_REQ 用例还直接确认：
关闭时不附加 metadata、也不触发严格 payload 校验；相同输入在开启时按预期被拒绝。

## 6. 交付文件

生产代码：

- `llm/include/common/msg.h`
- `llm/include/dte/dte_payload.h`
- `llm/include/workercore/workercore.h`
- `llm/src/workercore/workercore.cpp`
- `llm/src/workercore/logic.cpp`
- `llm/src/utils/msg_utils.cpp`
- `llm/src/utils/config_utils.cpp`
- `llm/src/prims/norm_prims/send_prim.cpp`
- `llm/src/monitor/config_helper_core.cpp`
- `llm/src/monitor/config_helper_pd.cpp`
- `llm/src/monitor/config_helper_pds.cpp`

测试与配置：

- `llm/src/dte/v0_selftest.cpp`
- `llm/src/die/v0_selftest.cpp`
- `llm/test/dte/run_test_dte_v1.py`
- `llm/test/dte/hardware/v1*.json`
- `llm/test/dte/simulation/v1*.json`
- `llm/test/dte/workload/v1*.json`

## 7. 与计划的差异

原计划曾考虑由目的端根据 DATA 包或 behavioral 代表包累计 payload。实际实现改为由 REQUEST
声明整 flow payload，原因是该方式可以同时正确处理 physical/behavioral NoC、包聚合、stripe
和多 source，并且不会把相同逻辑 payload 按 stripe 重复计费。

REQUEST wire 高 64 位因此成为 DTE V1 的 tagged metadata。控制消息仍不经过 DTE，只有其携带的
声明在目的端用于计费。

评审发现初版在 DTE off 时也无条件计算 REQUEST payload，可能把严格校验泄漏到关闭路径。
现已通过 `AttachRequestDtePayload()` 修正为 off no-op/on strict，并加入直接自测。

## 8. 遗留事项与下一版入口

- `send_para_logic()` 尚未接入 DTE；V2a 需要为连续 SEND_DATA 建立独立稳定 context。
- V1 交付时的全局 parallel 限制已由 V2a 解除；当前仅未支持的非 dataflow、stripe 和同核双向组合继续拒绝。
- source DTE、NoC/D2D、destination DTE 仍是整块串行，属于保守上界；V2b 才引入流水模型。
- COMET aggregation/coalescing、地址映射和 compute/DMA issue-poll 不属于 V1。
- V1 已覆盖阻塞式 pipeline/refill 的相同 `(source, tag)` 两轮复用；V2a 引入并发 admission 后，
  仍需重新验证共享 metadata map 与 transfer context 的并发所有权。

V1 已满足进入 V2a 的条件：阻塞式生产路径、精确 payload、trace、开关矩阵、跨 die、
多 stripe、多 source、重复 flow 生命周期、异构 per-core 配置和既有回归均通过。
