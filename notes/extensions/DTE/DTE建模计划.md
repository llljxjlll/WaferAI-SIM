已有信息

DTE 硬件：每个 core 一个 DMA 引擎。参考 `notes/extensions/DTE/refs/tx8_dte` 中的 TX8 资料，保留两个探索参数：

- `channel_count`：DTE 可同时发动（active）的通信行为上限。
- `bit_width`：DTE 共享数据通路的聚合位宽，影响 DTE 带宽和整体通信性能。

DTE 行为：参考 COMET 的 DMA 行为与延迟分解，通过行为仿真体现启动、排队和数据传输延迟。

---

# DTE 建模计划（修订版）

## 0. 决策速览

| 问题 | 结论 |
|---|---|
| 模块放置 | 新增 `llm/include/dte/`、`llm/src/dte/`；每个 `WorkerCoreExecutor` 拥有一个 `DTEUnit` |
| 首期接入 | `send_logic()` 的 `SEND_DATA` 和 `recv_logic()` 的 `RECV_DATA`；控制包不经过 DTE |
| 数据单位 | DTE 内部统一使用 bit；`M_D_DATA`、`end_length`、`Msg::length_` 当前也是 bit |
| channel 语义 | `channel_count` 是 active transfer 的硬上限；多余 descriptor 进入 pending queue |
| 位宽语义 | `bit_width_bits` 是所有 channel 共享数据通路的聚合位宽；channel 增多不直接提高峰值总带宽 |
| 资源模型 | pending queue → 最多 `channel_count` 个 active channel → 共享 bus RR 仲裁 → complete |
| 时间计算 | 使用整数 cycle 并向上取整：`ceil(payload_bits / bit_width_bits)` |
| V1 时序 | 明确采用保守的 store-and-forward 模型，先保证延迟和资源竞争可验证 |
| 流水重叠 | V2b 增加 streaming 模型，避免发送端 DTE、NoC、接收端 DTE 永久按整块完全串行计费 |
| 异步原语 | V3a 增加 `Dte_async` issue/wait/poll/fence/cancel 和 logical token，支持 compute/DMA 重叠 |
| 并发路径 | 现有 `send_para_queue` 不包含 `RECV_DATA`；V2a 先支持连续 SEND 并发，RECV 并发单独改 dispatcher |
| COMET 定位 | V3b 已实现 COMET 启发的在线 endpoint coalescing；它复现 address-block/compound-launch 趋势，不等同于论文完整离线 mapper 与 route-aware 模型 |
| V4 资源收尾 | 六个方向；SPM/AXI read/write 四类独立 endpoint 端口；每 channel 固定两条命令；有限 credit；功耗/面积 |
| 暂缓范围 | scatter、broadcast、stride/slice/shuffle 数据组织；无真实需求的多 RECV_DATA dispatcher 并发 |
| 配置错误 | 启动期集中校验并抛异常，不依赖 `LOG_ERROR` 是否终止 |

## 1. 范围与准确性边界

当前覆盖每核 DTE 的六个 endpoint 方向：SPM→remote、remote→SPM、SPM→SPM、SPM→DRAM、DRAM→SPM 和 DDR→remoteTile，并覆盖 descriptor 启动、channel/双命令槽准入、有限 credit、共享 legacy 通路或 V4 独立 SPM/AXI read/write 端口竞争、完成通知、V3a 显式异步 token、V3b 在线连续区间 coalescing，以及 V4 功耗/面积统计。核心探索参数包括 `channel_count`、通路位宽、pending 深度和聚合规模；每 channel 两条命令是固定硬件常数，γ、平均 launch latency、聚合窗口和功耗/面积系数属于标定/策略参数。

当前只排除按本轮决策暂缓的数据组织（scatter、broadcast、stride/slice/shuffle）、没有真实 workload 需求的多 `RECV_DATA` dispatcher 并发，以及 COMET 的 task-DAG 遗传搜索、自动地址 mapper 和式 (16)–(17) route-aware scaling。V3b 的 `address_block` 由 workload 显式声明并校验；V4 的 DRAM/remote 方向只模拟本核 endpoint 端口，不替代或重复统计 DRAM media、NoC 和 D2D。

### 1.1 论文概念到模拟器的映射

| 概念 | 模拟器来源 | V0/V1 处理 |
|---|---|---|
| DMA payload | `Send_prim` 总 payload bits；接收侧逻辑 payload | 精确计算 |
| request generation | `Issue()` 时刻 | 显式记录 |
| channel admission | `channel_count` 个 active slot | 显式模拟 |
| queue delay | pending 等待 active slot、active 等待 bus | 状态机自然产生 |
| launch latency | `gamma_ns`、`tau_launch_avg_ns` | COMET baseline，可配置；出处和适用边界见第 6 节 |
| data transmission | `ceil(payload_bits/bit_width_bits)` | 精确到 cycle |
| async compute/DMA dependency | `Dte_async` logical token 和显式 wait/fence | V3a 已实现 |
| aggregation/address mapping | workload 显式提供 peer/remote address/block；同 block 双端连续区间在线聚合 | V3b 已实现；不含论文离线 mapper |
| endpoint resources | 六方向映射到 SPM read/write、AXI read/write；复合方向原子获取资源并由最慢端口决定完成 | V4 已实现；每 channel 两条命令且 pending credit 有限 |
| power/area | launch + 分端口 per-bit 动态能耗；base + channel + command slot + 总端口位宽面积 | V4 已实现；系数可配置，默认 0 |
| NoC/D2D service | 已有行为模型 | DTE 不重复模拟 hop/link service |

## 2. 仓库约束

- `worker_core_execute()` 默认每次派发一条原语并等待 `prim_block.negedge_event()`。
- `send_logic()`、`recv_logic()` 分别在原语末尾通知 `ev_block`。
- `SPEC_SEND_RECV_PARALLEL` 的批处理条件只包括 `RECV_ACK`、`SEND_DATA`、`SEND_REQ`、`SEND_DONE`，不包括 `RECV_DATA`。
- behavioral NoC 下每个 stripe 可能只发一个代表包；代表包的 `length_` 不等于整块 payload。
- `M_D_DATA=128` 是 128 bit；`GetPacketInfo()` 使用 `slice_size_in_bit`，`end_length` 也是 bit。
- `CoreHWConfig` 是按核硬件参数的标准入口；`D2DLinkUnit`、`NB_GlobalMemIF` 只作为有限资源和完成通知的参考。

## 3. 数据量与时间定义

### 3.1 统一使用 bit

新增 `llm/include/dte/dte_payload.h`：

```cpp
uint64_t ComputeSendPayloadBits(const Send_prim &prim);
uint64_t CeilDiv(uint64_t n, uint64_t d);
```

发送 payload 不重放 stripe/代表包循环，而由逻辑包字段计算。`CalculatePacketNum()` 会按 `HW_NOC_PAYLOAD_PER_CYCLE` 压缩模拟包数，因此 `Send_prim` 同时保存聚合倍率和末组真实包数：

```cpp
raw_packets = (prim.max_packet - 1) * prim.packet_scale
            + prim.packets_in_last_group;
payload_bits = raw_packets == 0
    ? 0
    : (raw_packets - 1) * M_D_DATA + prim.end_length;
```

`packet_scale` 和 `packets_in_last_group` 分别编码在 `SEND_DATA` 原语原空闲的 bit 36..43 和 44..51；旧编码中的全零值按 1 解码。校验 `max_packet>0` 时 `1<=end_length<=M_D_DATA`、`1<=packets_in_last_group<=packet_scale<=255`，并检查整数溢出。该元数据只用于还原 NoC 模拟包压缩前的逻辑数据量，不是 COMET aggregation/coalescing。结果在 physical/behavioral NoC 和不同 `stripe_count` 下必须一致。

接收侧不能只累加 behavioral 代表包的 `length_`。V0 固定使用 `ComputeMsgLogicalPayloadBits()`：physical 包贡献 `length_`；behavioral 代表包贡献 `(roofline_packets_-1)×M_D_DATA+length_`，从已有代表包元数据恢复该 subflow 的精确逻辑 payload，无需扩展 wire 格式。

### 3.2 时间计算

```cpp
transmit_cycles = CeilDiv(payload_bits, uint64_t(cfg.bit_width_bits));
transmit_time = sc_time(transmit_cycles * CYCLE, SC_NS);
```

所有内部时间先转为整数 cycle；γ 和 launch 参数解析后也转换到 cycle，明确向上取整规则，避免浮点 `sc_time` 误差。

### 3.3 端到端边界

DTE 只模拟本地端点的数据准备/落盘和资源竞争，不重复模拟 NoC/D2D hop。V1 使用：

```text
source DTE complete → NoC/D2D → destination DTE complete
```

这是 store-and-forward 保守近似，可能高估流式硬件延迟。V2b 通过显式首单元事件与
流级闭式尾部组合三段服务：

```text
source_first = source_transmit_start + first_ready_service
first_network_latency = destination_first - source_first
source_tail_at_destination = max(source_first, source_done)
                           + first_network_latency
final_write = max(destination_dte_done,
                  network_last + destination_drain,
                  source_tail_at_destination + destination_drain)
```

`destination_drain` 固定为一个 DTE cycle。physical NoC 的 packet/压缩 packet 发送按累计逻辑
bit 等待 source readiness；behavioral NoC 每个代表 flow 在首个 DTE word ready 后放行。
目的端在每个 source 的首个 DATA 到达时 issue 独立 `REMOTE_TO_SPM` context。physical backend
直接以实际末包到达为 `network_last`；同 die behavioral 使用 `roofline_packets_`，跨 die
behavioral D2D 在首包服务结束时交付代表消息，并携带剩余 bulk tail cycle。

因此 bulk 阶段不再无条件相加，稳态斜率由 source DTE、网络和 destination DTE 中的实际
瓶颈决定，同时保留真实网络拥塞和 DTE channel/bus 资源状态。48-bit DATA 元数据携带
`source_first_ns/source_done_ns`，32-bit 字段携带 behavioral D2D tail cycles；所有时间均有
容量与溢出校验。V1 store-and-forward 由 `dte.streaming=false` 保留为可比较基线。

## 4. `DTEUnit` 设计

### 4.1 类型

```cpp
enum class DteDir {
    SPM_TO_REMOTE, REMOTE_TO_SPM,
    SPM_TO_SPM, SPM_TO_DRAM, DRAM_TO_SPM,
    DRAM_TO_REMOTE // V4
};

struct DTEConfig {
    uint32_t channel_count;
    uint32_t bit_width_bits;       // 共享聚合位宽
    uint64_t gamma_cycles;
    uint64_t tau_launch_cycles;
    bool fine_grained_resources;
    uint32_t command_slots_per_channel; // V4 必须为 2
    uint32_t pending_queue_depth;
    uint32_t spm_read_width_bits, spm_write_width_bits;
    uint32_t axi_read_width_bits, axi_write_width_bits;
    // launch/per-port energy 与 base/channel/slot/port-width area 系数
};

struct DteTransferContext {
    uint64_t xfer_id;
    uint64_t payload_bits;
    DteDir dir;
    sc_event done;
    sc_time issue_time;
};
```

并发版本不再同时保存 `descriptor.done` 和 `unordered_map<xfer_id, sc_event*>` 两份状态。context 放在完成前地址稳定的容器中，例如 `std::list<std::unique_ptr<DteTransferContext>>`；不能把 `sc_event` 当普通可移动值存放。

### 4.2 有界状态机

```text
Issue
  ↓
pending_queue（尚未发动）
  │ active_count < channel_count
  ↓
active[channel]（最多 channel_count 条，launch 可并行）
  ↓
bus_wait_queue ──RR──> transmit
  ↓
notify done，释放 slot，立即 admit pending
```

约束：

- `channel_count` 严格限制 active transfer；pending 不计 active。
- 每个 active channel 同时只有一条 descriptor。
- V0–V3 legacy 模式保持每 active channel 一条命令和无界 pending 语义。
- V4 每 channel 固定两条 command slot，不作为第三个探索参数；额外 descriptor 受有限全局 pending credit 约束，credit 用尽时 `WaitForCredit()` 真实阻塞 issue。
- legacy channel 可并行 launch；transmit 阶段共享一条聚合 bus 并采用 RR。
- V4 将数据阶段替换为 SPM read/write、AXI read/write 四类独立资源；同一资源 FIFO 公平，互不相交的资源可并行，复合 transfer 原子获得全部所需资源后启动，各端口按自己的位宽释放，context 在最慢端口结束时完成。
- legacy 模式中 channel 增多不改变 `bit_width_bits` 决定的共享 bus 峰值；V4 中峰值分别由四类端口位宽决定。

### 4.3 延迟分解

```text
T_total = T_pending_admission + T_launch + T_bus_queue + T_transmit
T_transmit = ceil(payload_bits / bit_width_bits) × CYCLE
```

首期：

```text
T_launch = gamma_cycles + tau_launch_cycles
```

不再使用 `q_i×tau_launch` 表达已由状态机显式产生的排队，否则会重复计费。若论文证明 `q_i` 是独立硬件启动项，后续加入时必须附页码、变量映射和不重复计费的说明。

### 4.4 trace

每条 transfer 输出 `DTE_pending`、`DTE_launch`、`DTE_bus_wait`、`DTE_transmit`，携带 core、xfer_id、direction、payload_bits、channel_id。由 trace 验证 active 上限、bus 不重叠、RR 顺序和 DTE/NoC 计费边界。

## 5. WorkerCore 集成

`WorkerCoreExecutor` 使用：

```cpp
std::unique_ptr<DTEUnit> dte;
```

V1 发送侧在第一个 DATA 包进入 NoC 前 issue 并等待：

```cpp
auto &xfer = dte->Issue(ComputeSendPayloadBits(*prim), DteDir::SPM_TO_REMOTE);
wait(xfer.done);
```

接收侧在所有预期 DATA 到齐、原语结束前 issue `REMOTE_TO_SPM` 并等待。控制包不经过 DTE。

为避免 behavioral 代表包、`HW_NOC_PAYLOAD_PER_CYCLE` 压缩和 stripe 拆分造成目的端
payload 少算，配对的 `SEND_REQ` 保存与 `SEND_DATA` 相同的包聚合元数据；生成 REQUEST 时
计算完整逻辑 payload，并编码到 REQUEST 未承载业务数据的 `data_[127:64]`。同一 flow 的
每个 stripe 重复携带相同声明，目的端按 `(source, tag)` 去重；所有预期 source 到齐后求和。
这只是 DTE 计费元数据，REQUEST 本身不支付 DTE 延迟。
REQUEST 的 payload 附加通过 `AttachRequestDtePayload()` 受 DTE 开关门控：
`use_beha_dte=false` 时为严格 no-op，不调用 `ComputeSendPayloadBits()`，因此不会把 DTE 的
额外合法性约束泄漏到 legacy 关闭路径；开启时才计算并校验。

V1 阶段未接 `send_para_logic()`，因此 DTE 与 `SPEC_SEND_RECV_PARALLEL` 的组合曾在
配置解析后统一拒绝。V2a 验收后，dataflow parallel 已开放；非 dataflow 仍在启动期拒绝，
parallel stripe>1 和同核 SEND/RECV DTE 重叠在运行时明确拒绝。

V2 并发分两步：

1. V2a 在 `send_para_logic()` 每批开始时预扫描全部 `SEND_DATA` 并背靠背 issue；队列中
   SEND_REQ/RECV_ACK/DATA 的原始顺序不变。每条 DATA 保存独立 `(xfer_id, context*)`，
   仅在自己的 DTE 完成并释放后才按原语顺序注入 NoC。context 由 DTEUnit 的地址稳定
   `std::list<std::unique_ptr<...>>` 持有。V2a 不声称支持同核 SEND/RECV 双向 DTE 并发。
2. V2b 的真实 workload 调研未发现同一 dispatcher 同时执行多个 `RECV_DATA` 原语的需求，
   因而不改造 dispatcher。`dte.streaming=true` 只开放顺序 dataflow，并与
   `send_recv_parallel=true` 明确互斥；未来若出现需求，再单独设计 RECV 原语、buffer、tag/
   source/subflow 匹配和 `prim_block` 所有权。

parallel fan-out 的一个 compute-ready/`send_last_packet` token 对整批 DATA 生效，只在最后
一条 DATA 尾包提交后消费。目的端 REQUEST payload 使用按 `(source,tag)` 分组的轮次队列，
每轮记录 stripe mask，使后一轮 REQUEST 提前到达时不会被前一轮 RECV 错误消费。

V3a 通过独立 `Dte_async` primitive 实现通用 compute/DMA 重叠。`issue` 绑定 workload logical token 与 DTE physical `xfer_id` 后立即返回；`wait` 消费指定 token，`poll` 非阻塞查询，`fence` 按 issue 顺序消费全部 outstanding token，`cancel` 仅允许取消 pending descriptor。计算依赖由计算前显式 WAIT 表达，不修改既有计算 primitive wire。

每条 issue 携带 SPM byte 半开区间和 read/write 属性；重叠 read/read 可并发，RAW/WAR/WAW 在后发 issue 处等待旧 descriptor 完成，但不隐式消费其 logical token。队列 drain 或 `SEND_DONE` 时存在未 wait/fence 的 token 会明确失败。V3a descriptor 只计 endpoint DTE service，不生成 NoC/D2D 消息，也不替代既有 SEND/RECV。

## 6. 配置

hardware config：

```json
{
  "dte": { "gamma_ns": 40000, "tau_launch_avg_ns": 2000 },
  "cores": [
    { "id": 0, "exu_x": 128, "sfu_x": 2048, "sram_bitwidth": 128,
      "dte_channel_count": 2, "dte_bit_width": 2048 }
  ]
}
```

simulation config：

```json
{ "dte": { "use_beha_dte": false, "streaming": false, "async": false } }
```

`streaming=false` 保留 V1/V2a store-and-forward；`streaming=true` 启用 V2b，并要求
`use_beha_dte=true`、dataflow 模式和顺序 dispatcher。`async=true` 启用 V3a，也要求
`use_beha_dte=true`、dataflow 模式和顺序 dispatcher，并与 `streaming=true` 明确互斥。V3b 进一步增加：

```json
{
  "dte": {
    "aggregation": false,
    "aggregation_max_descriptors": 16,
    "aggregation_max_bytes": 4096,
    "aggregation_timeout_ns": 20,
    "aggregation_address_block_bytes": 65536
  }
}
```

`aggregation=true` 要求 `async=true`；最大 descriptor 至少为 2，其余数值必须大于 0。timeout 按全局 `CYCLE` 向上取整为 cycle。聚合默认关闭；关闭时 legacy V3a primitive 保持 2-segment wire 和原冻结时序。

V4 在 hardware `dte` 中增加固定双命令槽、有限 pending 深度、四端口位宽和功耗/面积系数：

```json
{
  "dte": {
    "command_slots_per_channel": 2,
    "pending_queue_depth": 1,
    "spm_read_width_bits": 64,
    "spm_write_width_bits": 32,
    "axi_read_width_bits": 16,
    "axi_write_width_bits": 128,
    "launch_energy_pj": 10.0,
    "spm_energy_pj_per_bit": 0.01,
    "axi_energy_pj_per_bit": 0.02,
    "base_area_um2": 1000.0,
    "channel_area_um2": 100.0,
    "command_slot_area_um2": 10.0,
    "port_bit_area_um2": 0.5
  }
}
```

simulation config 以 `dte.fine_grained_resources=true` 启用 V4。它要求 `async=true`、
`use_beha_dte=true`、dataflow 和顺序 dispatcher；hardware 必须恰好两条 command slot、
pending 深度大于 0、四端口位宽大于 0，所有功耗/面积系数非负且有限。默认关闭，关闭时
继续使用 V0–V3 的共享 bus 和冻结时序。

使用 `dte_` 前缀避免与其他模块混淆。启动期必须校验：channel 和位宽大于 0，延迟非负，ns 可按规则转换到 cycle，以及开关组合受支持。

参数出处：COMET 第 8 页 §VI-A 将 fixed launch time γ 和 DMA descriptor-handling latency τ_launch 分别设为 40 μs 与 2 μs，并注明沿用参考文献 [38]。该参考文献为 T. Jose and D. Shankar, “Performance modeling of a heterogeneous computing system based on the UCIe interconnect architecture,” *IEEE Space Computing Conference (SCC)*, 2023, pp. 5–10。

因此，40000ns/2000ns 可作为可由配置覆盖的 COMET baseline 默认值；其来源是 COMET 从 UCIe 性能模型借用的评估参数，不是 TX8 实测值，也不是 per-core DTE 的固有硬件常量。默认关闭 DTE，关闭时既有行为和时序不变。

## 7. 集成文件

| 文件 | 改动 |
|---|---|
| `llm/include/dte/dte_types.h` | 六方向、四类端口、V4 配置/统计和双命令槽 transfer context |
| `llm/include/dte/dte_payload.h` | payload bit 计算、`CeilDiv` |
| `llm/include/dte/dte_streaming.h` | V2b source-tail 投影与三段尾部闭式组合 |
| `llm/include/dte/dte_async_types.h`、`dte_async.h` | V3a 操作/访问类型；V3b staged group、logical→physical fan-out 与生命周期 |
| `llm/include/dte/dte_coalescing.h` | V3b 聚合配置、统计和有效位 bus utilization |
| `llm/include/dte/dte_unit.h` | legacy shared-bus 与 V4 双命令槽/有限 credit/四端口状态机接口 |
| `llm/src/dte/dte_unit.cpp` | legacy/V4 SystemC 调度、原子多端口获取、逐端口释放、credit、功耗/面积和 trace |
| `llm/src/dte/dte_async.cpp` | V3a async/hazard；V3b coalescing；V4 六方向双区间 hazard、credit-aware issue |
| `llm/src/prims/norm_prims/dte_async_prim.cpp` | 六方向 JSON；legacy 2-segment 与 remote/destination metadata 3-segment wire |
| `llm/include/workercore/workercore.h` | `std::unique_ptr<DTEUnit>` 与 V3a `DteAsyncTracker` |
| `llm/src/workercore/workercore.cpp` | 构造 DTE/tracker、V3a 专用 dispatcher 与结束纪律 |
| `llm/src/workercore/logic.cpp` | V1 阻塞接入；V2a SEND 并发；V2b 顺序流式首包/尾部接入 |
| `llm/include/common/msg.h`、`llm/src/utils/msg_utils.cpp` | V2b DATA 流式时间元数据 wire 编解码 |
| `llm/src/die/d2d_link.cpp` | V2b behavioral D2D 首包交付与 bulk tail 传递 |
| `llm/include/common/config.h`、`llm/src/common/config.cpp` | 按核 DTE 参数 |
| `llm/include/defs/spec.h`、`llm/src/defs/spec.cpp` | 标定常量和总开关 |
| `llm/src/utils/config_utils.cpp` | 配置解析和集中校验 |
| `llm/src/dte/v4_selftest.cpp`、`llm/test/dte/run_test_dte_v4.py` | V4 资源/公式 selftest 与 WorkerCore 端到端验收 |
| `llm/test/dte/` | V0–V4 selftest、oracle、集成和参数扫描 |

## 8. 分阶段实施

### V0a：冻结模型契约

- [x] 确认并注释 `Msg::length_`、`end_length`、`M_D_DATA` 的 bit 契约。
- [x] 记录 COMET 参数的页码、章节、单位和引用链，并明确其为 UCIe 模型来源的评估默认值，而非 TX8/per-core DTE 标定值。
- [x] 确认 V1 store-and-forward、V2b streaming 的边界。
- [x] 确认 channel 是 active 上限、位宽是共享聚合位宽。

验收：不存在 byte/bit、per-channel/aggregate、排队/launch 歧义。

### V0b：独立 DTEUnit

- [x] 配置解析和非法值校验。
- [x] pending/active/shared-bus 状态机。
- [x] payload/时间纯函数。
- [x] SystemC selftest 和 Python oracle。

验收：active 不超过 channel 数；bus transmit 不重叠且 RR 公平；完成时间与 oracle 一致；没有排队重复计费。

### V1：阻塞式单传输接入

- [x] 接入普通 `send_logic()`、`recv_logic()`。
- [x] 拒绝未支持的 parallel 组合。
- [x] 增加 SEND、RECV、端到端测试。

验收：DTE 关闭时回归时序不变；单端额外时间等于 launch 加 `ceil(bits/width)×CYCLE` 及可解释资源等待；单传输不受 channel 数影响；不同 core 的 DTE 独立。

完成结果（2026-07-24）：physical/behavioral NoC 的 on/off 矩阵、stripe=1/2/4、
跨 die、多 source、pipeline/refill 两轮复用相同 source/tag、异构 per-core 位宽、trace 顺序和
V1 阶段非法 parallel 组合均通过；该阶段性门禁已由 V2a 的 dataflow parallel 支持取代。DTE off 的退化 REQUEST 门控由自测直接验证。标准 workload 每端
精确增加 2054 ns，端到端增加 4108 ns；16,384-bit 跨 die workload 每端增加 22 ns，端到端
增加 44 ns。当前 DTE 自测 55/55、V1 兼容矩阵 13/13；详细证据见
`log/V1_development.md`。

### V2a：真实 workload 多 SEND 并发

- [x] 接入 `send_para_logic()` 中连续 SEND_DATA。
- [x] 验证 context 生命周期、channel admission 和 bus contention。

验收：channel 数限制 active SEND 并影响 admission/launch overlap；共享 bus 峰值仍由位宽决定。

完成结果（2026-07-24）：`run_test_dte_v2.py` 15/15 通过。1/2/4 SEND ×
channel=1/2/4 的 9 个组合显示最大 active 精确受 channel 限制；四条 16,384-bit SEND
在 channel=1/2/4 时最后一条源 DTE 分别于 issue 后 144/88/84 ns 完成，而共享 bus
总 transmit 恒为 64 ns。同目的长短混合、有序提交、parallel refill、DTE off 和不支持
组合门禁均通过。

附带变更：本版同时修复 shared `send_para_logic()` 中既有的 parallel fan-out 尾包 token
死锁。该修复不受 DTE 开关门控，因而会使 DTE-off 多 SEND 从永久等待变为正常完成；它应在
changelog/提交说明中作为独立缺陷修复列出。详细证据见 `log/V2a_development.md`。

### V2b：流式端到端与可选 RECV 并发

- [x] 引入首单元事件与流级闭式尾部 pipeline 模型。
- [x] 组合 source DTE、NoC/D2D、destination DTE 的瓶颈带宽。
- [x] 完成 RECV 并发需求评估；当前 workload 未触发，保持 dispatcher 不变并增加明确门禁。

验收结果（2026-07-24）：`run_test_dte_v2b.py` 18/18。physical/behavioral 的
source/network/destination-slow 与等带宽场景全部满足 C++/Python 同构 oracle；behavioral
64→4096-bit 扫描的完成时间为 `[935,679,551,487,483,483,483]` ns，在 1024 bit 后转为
network-limited 平台。8-bit 小 payload、16,416-bit 非整除 payload、同 die、跨 die
behavioral D2D stripe=4 均通过。相同配置 streaming 均不慢于 V1 store-and-forward；例如
physical equal 803→553 ns，behavioral equal 741→551 ns，跨 die behavioral 632→604 ns。
DTE selftest 64/64，V1 13/13、V2a 15/15、D2D selftest 308/308、D2D V0/V4/V5
67/67、13/13、23/23、NoC frozen 4/4。评审闭环又增加了 legacy DATA wire 原始载荷/解码行为自测与
consumer gate 代码复核，并以独立集成负例覆盖 DTE-off、非 dataflow、parallel dispatcher
三项启动门禁；评审未发现阻塞问题。详细证据见 `log/V2b_development.md`。

### V3a：通用 compute/DMA 异步重叠

- [x] 新增 `Dte_async` issue/wait/poll/fence/cancel primitive 与固定 wire 编码。
- [x] 支持 logical token、多个 outstanding descriptor、选择性 wait 和顺序 fence。
- [x] 支持 pending cancel、token 复用、pipeline/refill 和结束时未消费 token 检查。
- [x] 基于 SPM byte 区间阻止 RAW/WAR/WAW，允许 read/read 并发。
- [x] trace 和精确总时间共同证明 compute/DMA overlap。

验收结果（2026-07-24）：DTE V3 selftest 31/31、WorkerCore 集成矩阵 14/14。overlap workload 中 Matmul 124→333ns 与 DTE transmit 134→198ns 真实重叠，总完成 345ns；阻塞依赖版本为 421ns，功能工作量相同。channel=1/2/4 的 max active 为 1/2/4，共享 bus transmit 总量均为 128ns。选择性 wait、fence、RAW/WAR/WAW、pending cancel、active cancel 拒绝、token 复用、pipeline refill、未 fence 结束负例和配置门禁全部通过。

评审闭环（2026-07-24）：独立复核确认 dispatcher 只为 `Dte_async_prim` 增加直接执行分支，
没有修改 SEND/RECV/compute 的 `prim_block` 协议；hazard 等待不消费依赖 token，pending
cancel 与 `CANCELLED` context 释放闭环正确。独立回归为 V0 64/64、V3a 31/31、
V1/V2a/V2b 13/13、15/15、18/18、D2D selftest 308/308、D2D V4/V5 13/13、23/23。
评审未发现代码问题，并再次确认 `Dte_async` 是显式 endpoint primitive、不生成网络消息的
既定范围边界。详细证据见 `log/V3a_development.md`。

### V3b：COMET 启发的在线 aggregation/coalescing（已完成）

COMET §V-C 式 (13)–(17) 的原始方法是离线联合搜索：地址映射把 PE 划入共享 address block，
同块任务可共享 compound descriptor；式 (15) 令组内 head 支付完整 launch，其余请求复用已分配
buffer，式 (16)–(17) 再按地址局部性缩放网络传输。本模拟器没有 COMET 的 task DAG 遗传搜索，
且 V3a endpoint primitive 不生成网络消息，因此 V3b 采用下列显式在线近似：

- `Dte_async issue` 增加 `remote_peer`、`remote_addr`、`address_block`；
- 仅同方向、同 peer、同 address block、且 SPM 与 remote byte 区间均按 issue 顺序连续的
  descriptor 可聚合；payload 必须 byte 对齐且与声明 SPM range 等长；
- 每组最多 `aggregation_max_descriptors` 条且不超过 `aggregation_max_bytes`；达到任一上限
  立即 issue 一个 physical DTE transfer；不兼容的新请求会先 flush 同 key 的旧组；
- 未满组在 `aggregation_timeout_ns` 到期、依赖 wait、fence 或 hazard 强制同步时 flush；
- 每个 logical token 独立完成且只消费一次；所有组员 token 消费后才释放共享 physical context；
- 组内只支付一次 DTEUnit launch 和一次合并 payload transmit；trace 记录 logical→physical
  fan-out、flush 原因、节省 launch 数和截至该次 flush 的共享 bus 累计有效位利用率；
- aggregation 关闭时逐 logical descriptor issue，严格退化为 V3a；
- COMET 式 (16)–(17) 的跨 address-block 路由倍率不在 V3b endpoint 模型重复计费，真实
  NoC/D2D 仍由 SEND/RECV 路径负责。

Figure 11 的“固定总数据量、每组请求数从 1 增至 16，launch 被逐步摊薄”作为本版论文趋势
微基准；不声称复现论文完整 DNN/LLM 遗传搜索或绝对硬件时间。

验收结果（2026-07-24）：V3b selftest 21/21、WorkerCore 集成 16/16。固定 16×64-bit 数据量下，group=1/2/4/8/16 的 physical transfer 为 16/8/4/2/1，首 issue 到末 transmit 为 352/178/102/70/66 ns，完整 workload 为 548/374/298/266/262 ns；launch savings 为 0/8/12/14/15。group=1 的 useful-bit bus utilization 为 50%，group≥2 为 100%，全部与独立 Python oracle 一致。V3a overlap 仍为冻结的 345 ns；详细证据见 `log/V3b_development.md`。

评审闭环（2026-07-24）：独立复核和完整重跑未发现正确性问题。根据评审指出的 trace 命名歧义，将 flush 中的 `utilization_ppm` 更名为 `cumulative_utilization_ppm`，明确它是截至本次 flush 的累计指标而非单组指标；runner 对 aggregation-on 校验 trace 累计值，并对全矩阵校验独立重算值和 Python oracle。模型计算与时序不变。

### V4：扩展方向与精细 endpoint 资源（已完成）

- [x] 增加 SPM→SPM、SPM→DRAM、DRAM→SPM、DDR→remoteTile；与既有两个方向组成六方向。
- [x] 六方向唯一映射到 SPM read/write、AXI read/write 四类资源。
- [x] 每 channel 固定两条 command slot；有限 pending descriptor credit 产生真实 issue backpressure。
- [x] 同端口串行、独立端口并行；复合方向原子获取资源并在最慢端口完成。
- [x] 以双 SPM 区间覆盖 SPM→SPM 源/目的 RAW/WAR/WAW；DDR→remoteTile 不虚构 SPM hazard。
- [x] 增加逐方向/端口统计、动态能耗、平均功耗和面积模型。
- [x] 明确只计 endpoint port service，不触发或重复计算 DRAM row/bank/media、NoC hop、D2D link。
- [x] V4 关闭时冻结 V3a 345 ns，并保留 legacy shared-bus wire/trace 行为。
- [ ] scatter、broadcast、stride/slice/shuffle（数据组织，按本轮决定暂缓，不属于 V4 验收）。

端口映射为 `SPM_TO_REMOTE=SPM_READ`、`REMOTE_TO_SPM=SPM_WRITE`、
`SPM_TO_SPM=SPM_READ|SPM_WRITE`、`SPM_TO_DRAM=SPM_READ|AXI_WRITE`、
`DRAM_TO_SPM=AXI_READ|SPM_WRITE`、`DRAM_TO_REMOTE=AXI_READ`。

验收结果（2026-07-24）：V4 selftest 19/19、WorkerCore 集成 18/18、Python oracle PASS。
4,096-bit 六方向 workload 在 SPM read/write=64/32 bit、AXI read/write=16/128 bit 时，
各端口服务 cycle 与闭式 oracle 完全一致；总动态能耗 551.52 pJ、面积 1,240 μm²。
双命令槽、独立读写重叠、共享端口 contention、复合最慢端口、有限 credit stall、
async 与 blocking SEND/RECV 混合背压、
SPM→SPM 目的 RAW、DRAM on/off 等价、配置负例和 V4-off 退化全部通过。详细证据见
`log/V4_development.md`。

评审闭环（2026-07-25）：V1/V2 blocking SEND/RECV 的四个物理 issue 点均先调用
`WaitForCredit()`。legacy 无界模式立即返回；V4 有界模式与 async tracker 共用同一 credit
池。新增 mixed workload 在源、目的两核分别用 3 个 async descriptor 占满容量后执行真实
SEND/RECV：blocking transfer 分别在 2236 ns、8344 ns 获得 credit，两核均完成 4/4 个
physical transfer 且各记录一次 stall，workload 于 24742 ns 正常结束。

DTE 主体至此收尾。数据组织若未来恢复，应作为独立扩展重新冻结 descriptor、地址生成、
多 completion 和 buffer/credit 契约；当前无真实需求的多 `RECV_DATA` dispatcher 并发也不作为
本轮完成阻塞项。

## 9. 测试与验证

| 层级 | 验证内容 |
|---|---|
| 纯函数 | bit 数、向上取整、溢出、非 8-bit 对齐、零长度策略 |
| 状态机 | pending→active→bus→complete、active 上限、slot 释放后立即 admit |
| 仲裁 | 同时/错时 issue、长短传输、RR 公平、无饥饿 |
| oracle | pending、launch、bus wait、transmit；V2b source/network/destination 三段尾部最大值 |
| 发送 | 单包、尾包 1/128 bit、stripe 1/2/4、behavioral/physical NoC |
| 接收 | 多 source/stripe、代表包 payload、重复尾包 |
| 开关 | DTE on/off × behavioral NoC on/off；V2a dataflow parallel；非 dataflow/stripe/同核双向门禁 |
| 扫描 | `bit_width∈{512,1024,2048,4096}`、`channel_count∈{1,2,4}` |
| 瓶颈 | DTE 位宽小于、等于、大于 NoC/D2D 有效位宽 |
| 生命周期 | context 完成前不释放；V2a 批末映射为空；parallel refill 使用新 xfer id；per-key metadata 按轮次消费 |
| V2a 并发 | 1/2/4 SEND × channel 1/2/4、同/异目的、等长/混合长度、有序 DATA 提交、共享 bus 不叠加 |
| V2b 流水 | physical/behavioral、source/network/destination/equal、64→4096 bit、小/非整除 payload、跨 die D2D |
| V3a 异步 | issue/poll/wait/fence/cancel、选择性等待、compute overlap、token 复用/refill、RAW/WAR/WAW、未消费结束负例 |
| V3b 聚合 | 连续/不连续、方向/peer/block 隔离、descriptor/byte limit、按 cycle 向上取整的 timeout、fan-out、cancel、1/2/4/8/16 COMET 趋势 |
| V4 资源 | 六方向、四端口闭式周期、独立/共享资源、双命令槽、有限 credit、async+blocking SEND/RECV 混合背压、双区间 hazard、DRAM on/off 不重复计费 |
| V4 PPA | launch/per-port 动态能耗、平均功耗、base/channel/slot/port-width 面积由 C++/Python/trace 三方复算 |
| 回归 | DTE 默认关闭时 DONE 时刻和既有测试不变 |
| 非法配置 | 零/负参数、时间转换错误、非法开关组合启动失败 |

参数扫描断言：单传输时间随位宽增大单调不增并符合 cycle 向上取整；channel 数不改变共享 bus 峰值；多 descriptor 时增大 channel 可减少 admission 等待或增加 launch overlap，但不保证所有 workload 总时间严格单调下降。

## 10. 已知限制与明确暂缓项

1. V1 的 store-and-forward 是保守上界；V2b 的 streaming 是流级闭式近似，不是逐 flit backpressure。
2. V2a 只承诺多 SEND；V2b 顺序流式模式不支持 parallel dispatcher。当前 workload 未证明多
   `RECV_DATA` 原语并发需求，故未将其塞入 `send_para_queue`。
3. V0/V1 是 COMET 启发的 baseline；V3b 是确定性在线 endpoint coalescing，不是论文的
   task-DAG 遗传搜索和自动地址映射。
4. V3b 复现 §V-C 式 (13)–(15) 的同 block/compound launch 概念与 Figure 11 趋势；式
   (16)–(17) route-aware scaling 仍由真实 SEND/RECV NoC/D2D 路径承担，不在 async endpoint 重复计费。
5. V0–V3 legacy 模式仍使用一条共享聚合 bus 和每 channel 一条 active 命令；V4 启用后才使用
   四类独立端口、每 channel 两条命令和有限 pending credit。两种模式均被测试冻结。
6. V4 的 SPM/AXI 是 endpoint 端口服务模型；不会调用真实 DRAM row/bank/media，也不会生成
   NoC/D2D 数据消息。端到端数据仍由 SEND/RECV 负责，依赖必须显式 WAIT/FENCE。
7. V4 功耗/面积是参数化解析模型，默认系数为 0；没有 TX8 实测标定时不得解释为硅后绝对值。
8. γ、τ̄_launch 是 COMET p.8 §VI-A → [38] UCIe 性能模型的可配置 baseline，不是 TX8/per-core
   DTE 实测常量。
9. `poll_complete` 提供运行态结果和 trace，但仓库尚无条件分支 primitive，workload 不能据此分支。
10. V3b 只聚合 byte 对齐、payload 与 SPM range 等长的连续请求；staged group 仅允许取消尾成员，
    已 issue compound 不支持部分取消。
11. scatter、broadcast、stride/slice/shuffle 属于数据组织，按本轮要求明确暂缓；它们不属于已完成
    V4 的 endpoint 资源验收，未来若恢复需独立定义 wire、地址生成、多 completion 和 buffer 语义。
