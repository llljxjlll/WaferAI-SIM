# DTE V0 开发记录

> 版本：V0（V0a + V0b）
> 完成日期：2026-07-24
> 状态：已完成并通过验收
> 规划文档：`../DTE建模计划.md`
> 验收清单：`../DTE分版本开发与验收清单.md`

## 1. 版本目标

DTE V0 的目标是先冻结模型契约，再实现一个不依赖 `WorkerCoreExecutor` 的独立
SystemC DTE 资源模型，为 V1 接入真实 SEND/RECV workload 建立稳定基础。

V0 分为两个部分：

- V0a：明确数据单位、payload 计算、参数语义、时间公式和 DTE/NoC 计费边界。
- V0b：实现 pending、active channel、共享 bus、完成事件和 trace 状态机，并建立独立验收。

本版不接入 WorkerCore 的生产通信路径，不实现以下内容：

- SEND/RECV workload 阻塞式 DTE 接入；
- compute/DMA issue-poll 和通用重叠；
- streaming/pipeline 数据传输；
- SPM↔DRAM、SPM↔SPM；
- COMET aggregation/coalescing；
- 地址映射、scatter/broadcast、功耗和面积模型。

## 2. 核心模型契约

### 2.1 数据单位

DTE API 统一使用 bit：

- `M_D_DATA=128` 表示每个逻辑 DATA 包承载 128 bit。
- `Send_prim::end_length` 表示最后一个逻辑包的有效 bit 数。
- `Msg::length_` 表示当前物理包或 behavioral 代表包尾部的有效 bit 数。
- `DteTransferContext::payload_bits` 表示一次完整 DMA 传输的逻辑数据量。
- `DTEConfig::bit_width_bits` 表示共享聚合数据通路每 cycle 的 bit 位宽。

V0 不允许在 DTE 接口中混用 byte 和 bit。

### 2.2 SEND_DATA payload 恢复

`CalculatePacketNum()` 会使用 `HW_NOC_PAYLOAD_PER_CYCLE` 压缩 NoC 模拟包数。
因此，`Send_prim::max_packet` 不是原始 128-bit DATA 包数量，不能直接用下面的旧公式计算：

```cpp
(max_packet - 1) * M_D_DATA + end_length
```

V0 在 `Send_prim` 中新增：

- `packet_scale`：原始包到模拟包的聚合倍率；
- `packets_in_last_group`：最后一组包含的原始包数量。

真实数据量按以下方式恢复：

```cpp
raw_packets =
    (max_packet - 1) * packet_scale + packets_in_last_group;

payload_bits =
    (raw_packets - 1) * M_D_DATA + end_length;
```

这两个字段编码在 `SEND_DATA` 原语原先未使用的 wire bit 中：

| 字段 | bit |
|---|---:|
| `packet_scale` | 36..43 |
| `packets_in_last_group` | 44..51 |

合法范围为：

```text
1 <= packets_in_last_group <= packet_scale <= 255
1 <= end_length <= M_D_DATA
```

旧 `SEND_DATA` 编码中上述字段为零，反序列化时按 `packet_scale=1`、
`packets_in_last_group=1` 处理，保持兼容。

这里的“包聚合元数据”只用于恢复 `HW_NOC_PAYLOAD_PER_CYCLE` 压缩前的逻辑数据量，
不是 COMET aggregation/coalescing 功能。

### 2.3 接收侧 payload

physical NoC 中，每个 `Msg` 贡献自身的 `length_`。

behavioral NoC 中，一个代表包可能对应多个逻辑包，使用已有元数据恢复：

```cpp
(roofline_packets_ - 1) * M_D_DATA + length_
```

因此接收侧不需要扩展 `Msg` wire 格式。

### 2.4 时间模型

V0 使用整数 cycle 和向上取整：

```cpp
transmit_cycles = CeilDiv(payload_bits, bit_width_bits);
launch_cycles = gamma_cycles + tau_launch_cycles;
```

ns 配置通过 `NanosecondsToDteCycles()` 转换为 cycle，避免浮点时间误差。

V1 计划采用的端到端顺序为：

```text
source DTE complete → NoC/D2D complete → destination DTE complete
```

这是 store-and-forward 保守模型。streaming/pipeline 属于 V2b。

### 2.5 参数语义

- `channel_count`：最多同时占用 active 状态的 transfer 数量，不是无界队列数量。
- `bit_width_bits`：所有 channel 共享的数据通路聚合位宽，不是 per-channel 位宽。
- channel 的 launch 可以并行。
- transmit 共享一条 bus，同一时刻只允许一个 channel 传输。
- bus 使用 round-robin 仲裁。
- 每个 active channel 在 V0 中只容纳一个 descriptor。

## 3. COMET 参数来源

COMET 第 8 页 §VI-A 将：

- fixed launch time `γ` 设为 40 μs；
- DMA descriptor-handling latency `τ_launch` 设为 2 μs。

COMET 注明这些数值沿用参考文献 [38]：

> T. Jose and D. Shankar, “Performance modeling of a heterogeneous computing system based
> on the UCIe interconnect architecture,” IEEE Space Computing Conference
> (SCC), 2023, pp. 5–10.

因此 V0 使用：

```text
gamma_ns = 40000
tau_launch_avg_ns = 2000
```

这两个值是可由配置覆盖的 COMET baseline 默认值，来源是 COMET 借用的 UCIe
性能模型参数；它们不是 TX8 实测值，也不是 per-core DTE 的固有硬件常量。

## 4. DTEUnit 实现

### 4.1 状态机

一次 transfer 的状态流转为：

```text
PENDING
   ↓ channel admission
LAUNCHING
   ↓ launch complete
BUS_WAIT
   ↓ shared-bus RR grant
TRANSMITTING
   ↓ transmit complete
COMPLETED
```

状态含义：

- `PENDING`：等待空闲 active channel。
- `LAUNCHING`：descriptor 已进入 channel，正在支付固定启动延迟。
- `BUS_WAIT`：launch 已完成，等待共享 bus。
- `TRANSMITTING`：独占共享 bus，持续
  `ceil(payload_bits / bit_width_bits)` 个 cycle。
- `COMPLETED`：记录完成时刻并通知该 transfer 自己的 `sc_event done`。

### 4.2 调度规则

- 全局 pending queue 按 issue 顺序入队。
- channel 空闲时按顺序 admission。
- 同时 ready 的 channel 由 round-robin 仲裁器选择。
- bus 忙时不得启动第二个 transmit。
- transmit 完成后释放 bus 和 channel，然后继续 admission 和仲裁。
- `MaxActiveCount()` 用于验证 active transfer 数量从未超过 `channel_count`。

### 4.3 context 生命周期

`DteTransferContext` 保存在：

```cpp
std::list<std::unique_ptr<DteTransferContext>>
```

选择 `std::list` 是为了保证 context 和其中 `sc_event` 的地址稳定。外部拿到的引用在显式
`Release(xfer_id)` 前保持有效。

约束：

- 未完成的 context 不能释放；
- 完成后可以释放一次；
- 重复释放返回失败；
- 每条 transfer 使用独立完成事件，避免唤醒丢失或误唤醒。

### 4.4 trace

每条 transfer 对以下阶段记录 begin/end：

- pending；
- launch；
- bus_wait；
- transmit。

V0 自测发现，同一 delta cycle 内连续执行多个 `add_event()` 时，一次
`SC_ZERO_TIME` notify 可能合并，原 `Event_engine::engine_run()` 每次只消费一个事件会导致
剩余事件永久留在队列中。

修正为每次唤醒后排空当前 trace queue。该修正经过 D2D 307 项回归验证，没有改变既有行为。

## 5. 配置

### 5.1 按核硬件参数

`CoreHWConfig` 新增：

```json
{
  "dte_channel_count": 2,
  "dte_bit_width": 2048
}
```

解析时要求两个值均大于零。

### 5.2 全局 DTE 参数

hardware config：

```json
{
  "dte": {
    "gamma_ns": 40000,
    "tau_launch_avg_ns": 2000
  }
}
```

simulation config：

```json
{
  "dte": {
    "use_beha_dte": false
  }
}
```

`use_beha_dte` 默认关闭，因此 V0 不改变现有 workload 时序。

交付的示例配置：

- `llm/test/dte/hardware/v0.json`
- `llm/test/dte/simulation/v0.json`

统一测试运行器会实际读取并校验这两个文件，避免示例配置成为未使用的 inert artifact。

## 6. V0a 开发内容

V0a 完成了以下工作：

1. 确认 `M_D_DATA`、`end_length` 和 `Msg::length_` 的 bit 契约。
2. 实现 `CeilDivU64()` 和 ns→cycle 向上取整。
3. 实现 `ComputeSendPayloadBits()`。
4. 实现 physical/behavioral 接收 payload 恢复。
5. 明确 channel、共享 bus、launch 和 transmit 语义。
6. 明确 DTE 与 NoC/D2D 的计费边界。
7. 查明 COMET 40 μs/2 μs 参数来源及适用边界。
8. 修正 `HW_NOC_PAYLOAD_PER_CYCLE` 导致的 payload 低估风险。

## 7. V0b 开发内容

V0b 完成了以下工作：

1. 实现独立 `DTEUnit` SystemC 模块。
2. 实现 pending、active channel、并行 launch 和共享 bus RR。
3. 实现地址稳定的 context 及显式释放接口。
4. 实现每 transfer 独立完成事件。
5. 实现四阶段 trace。
6. 增加命令行入口 `--dte-v0-selftest`。
7. 实现独立 Python cycle oracle。
8. 实现统一 V0 测试运行器。
9. 增加硬件和仿真配置样例。
10. 自测模块全部使用 `std::unique_ptr`，不保留裸 owning pointer。

## 8. 评审问题及修正

### 8.1 `max_packet` 低估真实数据量

评审发现 `CalculatePacketNum()` 会在计算 `end_length` 后，用
`HW_NOC_PAYLOAD_PER_CYCLE` 再次压缩 `max_packet`。当该值为 20 时，直接用
`max_packet` 计算 DTE payload 会把传输量低估约 20 倍。

修正：

- 保留压缩后的 `max_packet`，避免改变既有 NoC 行为和 workload 时序；
- 在 `Send_prim` 中增加倍率和末组包数；
- 更新所有 `CalculatePacketNum()` 调用点；
- 序列化新增字段；
- DTE 从新增字段恢复原始包数；
- 增加 100 个原始包、倍率 20、非满末组和短尾包测试。

关键验证：

```text
原始包数：100
HW_NOC_PAYLOAD_PER_CYCLE：20
模拟 max_packet：5
end_length：128 bit
DTE payload：12800 bit
```

另一个非满末组测试：

```text
输入：1442 byte
原始 payload：11536 bit
原始包数：91
模拟 max_packet：5
packets_in_last_group：11
end_length：16 bit
DTE 恢复结果：11536 bit
```

### 8.2 自测裸 owning pointer

原自测使用 `new` 创建 `DTEUnit`、probe 和 `Event_engine`，一次性进程退出时虽然不会影响
结果，但违反项目不使用裸 owning pointer 的约束。

修正后：

- probe 内的 DTE 使用 `std::unique_ptr<DTEUnit>`；
- 顶层 probe 使用 `std::make_unique`；
- trace engine 使用 `std::unique_ptr<Event_engine>`；
- 对象声明顺序保证 trace probe 先于其引用的 trace engine 析构。

### 8.3 示例配置未被使用

原 `hardware/v0.json` 和 `simulation/v0.json` 只作为交付文件存在。

修正后，`run_test_dte_v0.py` 会：

- 使用 `json.loads()` 读取两个文件；
- 验证 COMET 默认值；
- 验证 channel 和 bit width 为正；
- 验证 `use_beha_dte` 是布尔值；
- 将配置加载结果计入统一测试 PASS/FAIL。

### 8.4 COMET 参数出处

原记录只把 40000ns/2000ns 标记为待标定值。评审后补充了 COMET p.8 §VI-A 和
参考文献 [38] 的来源链，并明确这些数值可以作为可配置 baseline，但不能声称是 TX8
或 per-core DTE 的实测硬件参数。

## 9. 交付文件

### 9.1 DTE 模块

- `llm/include/dte/dte_types.h`
- `llm/include/dte/dte_payload.h`
- `llm/include/dte/dte_unit.h`
- `llm/src/dte/dte_unit.cpp`
- `llm/src/dte/v0_selftest.cpp`

### 9.2 原语和 payload 元数据

- `llm/include/common/msg.h`
- `llm/include/macros/macros.h`
- `llm/include/prims/norm_prims.h`
- `llm/include/utils/msg_utils.h`
- `llm/src/prims/norm_prims/send_prim.cpp`
- `llm/src/utils/msg_utils.cpp`
- `llm/src/monitor/config_helper_core.cpp`
- `llm/src/monitor/config_helper_pd.cpp`
- `llm/src/monitor/config_helper_pds.cpp`

### 9.3 配置和入口

- `llm/include/common/config.h`
- `llm/src/common/config.cpp`
- `llm/include/defs/spec.h`
- `llm/src/defs/spec.cpp`
- `llm/src/utils/config_utils.cpp`
- `llm/test/simulation_config/default_spec.json`
- `llm/unittest/npusim.cpp`

### 9.4 测试

- `llm/test/dte/oracle.py`
- `llm/test/dte/run_test_dte_v0.py`
- `llm/test/dte/hardware/v0.json`
- `llm/test/dte/simulation/v0.json`

### 9.5 相关正确性修正

- `llm/src/trace/Event_engine.cpp`

## 10. 测试与验收结果

### 10.1 构建

```bash
cmake --build build --target npusim -j2
```

结果：通过。

### 10.2 DTE SystemC 自测

```bash
./build/npusim --dte-v0-selftest
```

结果：49/49 通过。

覆盖范围：

- 1 bit、整包、短尾包和多包 payload；
- `HW_NOC_PAYLOAD_PER_CYCLE=20` 的精确 payload；
- 非满末组；
- `Send_prim` wire round-trip；
- stripe 不改变逻辑 payload；
- physical/behavioral payload 一致性；
- ns→cycle 向上取整；
- 非法配置、零 payload 和溢出；
- channel 1/2/4 admission；
- 并行 launch；
- shared bus 无重叠；
- round-robin 顺序；
- 独立完成事件；
- context Release 生命周期；
- trace 完整性；
- 仿真结束后 pending、active 和 bus 状态全部排空。

### 10.3 统一 DTE 验收

```bash
python3 llm/test/dte/run_test_dte_v0.py
```

结果：

```text
[PASS] sample DTE configs
[PASS] independent cycle oracle
[PASS] SystemC DTE V0 self-test: 49 checks
```

### 10.4 D2D 回归

```bash
./build/npusim --d2d-v0-selftest
```

结果：307/307 通过。

### 10.5 既有 workload 回归

```bash
cd build
./npusim \
  --workload-config ../llm/test/noc_congestion/workload/gemm_no_congestion.json \
  --hardware-config ../llm/test/noc_congestion/hardware/core_4x4.json \
  --simulation-config ../llm/test/noc_congestion/sim/sim_cycle.json \
  --mapping-config ../llm/test/noc_congestion/mapping/identity.spec
```

结果：完成于 29109 ns，与冻结基线完全一致。

### 10.6 静态检查

- 两个 DTE 示例 JSON 解析通过；
- Python runner 和 oracle 语法检查通过；
- `git diff --check` 通过；
- 自测中不存在 `new DTEUnit`、`new DTEProbe`、`new DTEEventProbe` 或
  `new Event_engine`。

## 11. 与原计划的差异

1. behavioral 接收 payload 没有扩展 `Msg` wire 字段，而是使用已有
   `roofline_packets_` 和 `length_` 恢复精确数据量。
2. bus wait 使用 active channel 状态扫描，没有额外维护物理 deque；对外状态语义相同。
3. 为保证同一 delta cycle 的 trace 不丢失，修正了 `Event_engine` 的队列排空逻辑。
4. 为兼容 NoC 的包压缩语义，`SEND_DATA` 使用原空闲 bit 36..51 携带精确 payload 元数据。

## 12. 已知限制与 V1 入口条件

V0 已满足进入 V1 的条件，但以下内容仍未实现：

- DTE 尚未挂到 `WorkerCoreExecutor`。
- 真实 SEND/RECV 不会调用 `DTEUnit::Issue()`。
- 当前 workload 中 `use_beha_dte` 仍保持关闭。
- V1 只承诺 store-and-forward 阻塞式接入。
- `SPEC_SEND_RECV_PARALLEL` 与 DTE 同时开启的组合需要在 V1 明确拒绝或实现。
- compute/DMA overlap、双向并发、streaming 和 COMET coalescing 均属于后续版本。
- 40 μs/2 μs 是可覆盖的 COMET baseline，不代表目标硬件最终标定值。

进入 V1 时应优先完成：

1. `WorkerCoreExecutor` 以 `std::unique_ptr<DTEUnit>` 持有按核 DTE。
2. SEND_DATA 在进入 NoC/D2D 前执行 source DTE。
3. RECV_DATA 在所有逻辑 DATA 到齐后执行 destination DTE。
4. 控制包不经过 DTE。
5. 默认关闭 DTE 时保持现有 workload 的 DONE 时刻不变。
6. 增加 DTE on/off、physical/behavioral NoC 和非法开关组合的端到端测试。

## 13. V0 结论

DTE V0 已完成独立资源模型、配置契约、精确 payload 计算、trace、生命周期管理和自动化验收。
评审发现的 payload 低估、裸 owning pointer、示例配置未使用及 COMET 参数出处问题均已关闭。

当前实现可以作为 V1 workload 集成的稳定基线。
