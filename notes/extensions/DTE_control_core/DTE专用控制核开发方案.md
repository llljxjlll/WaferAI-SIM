# DTE 专用控制核开发方案

> 状态：已完成（C0-C5 全部交付，最终验收通过）
> 日期：2026-08-25
> 关联文档：`notes/extensions/DTE/DTE建模计划.md`、`notes/extensions/DTE/DTE分版本开发与验收清单.md`、`notes/extensions/SRAM/SRAM能力扩展实现计划.md`

## 0. 目的与结论

本文档给出 NPU-SIM 中“每个计算 core 包含两个控制核，其中一个专门控制 DTE”的开发方案。目标是把该结构做成可选硬件配置，并在不破坏旧配置、旧时序和现有 DTE V0–V4 能力的前提下，显式建模：

- 通用控制核与 DTE 专用控制核的职责边界；
- 通用控制核向 DTE 控制核提交命令时的队列、派发延迟和背压；
- DTE 命令的同步、异步、完成通知和程序顺序；
- 双控制核带来的 compute/DTE 控制并行；
- 按 core 配置的异构能力。

本方案冻结以下核心决策：

1. 两个控制核是一个计算 tile 内部的控制子单元，不是两个新的 NoC 可寻址 core。
2. 不改变 `GRID_X`、`GRID_Y`、`TOTAL_CORES`、core ID、路由 endpoint、workload 映射和 HOST/HBM 挂载。
3. 使用硬件配置 `control_cores.mode` 选择共享控制或 DTE 专用控制模式。
4. 配置缺失时使用 `legacy_shared`，现有功能、调度和仿真时序必须保持不变。
5. `dte_channel_count` 继续表示 DTE active transfer 的硬上限，不表示控制核数量。
6. 新增独立、有界的 DTE 控制命令队列；不能只增加一个没有资源和时间语义的 `SC_THREAD`。
7. DTE 数据面继续由现有 `DTEUnit`、`DteAsyncTracker`、`DteMemoryBridge`、SRAM/HBM/NoC 模块承担；DTE 控制核不重复统计这些模块已有的服务延迟。
8. 第一阶段不新增控制核专用 ISA，也不把控制核作为独立 workload 调度目标。

目标结构如下：

```text
                              per compute core / tile

 primitive queue
       │
       ▼
+--------------------+       bounded command FIFO       +--------------------+
| general controller | ────────────────────────────────► | DTE controller     |
| WorkerCoreExecutor | ◄──────────────────────────────── | command + token FSM|
+---------+----------+       completion notification    +---------+----------+
          │                                                       │
          ├── compute / LSU / ordinary NoC control                ├── DteAsyncTracker
          │                                                       └── DTEUnit
          │                                                               │
          └──────────────── shared SRAM / HBM / NoC data plane ────────────┘
```

## 1. 当前基线与缺口

### 1.1 当前模块关系

当前每个 `WorkerCore` 创建一个 `WorkerCoreExecutor`。`WorkerCoreExecutor` 同时承担：

- 原语取出与顺序派发；
- compute、send、recv、request 等控制；
- 数据和控制 NoC 通道握手；
- DTEUnit 和 DteAsyncTracker 的创建与直接调用；
- DTE async token、P2P endpoint、collective worker 等状态管理。

`WorkerCoreExecutor` 构造时根据 `CoreHWConfig::dte_channel_count`、`dte_bit_width` 和全局 DTE 硬件字段创建 `DTEUnit`，随后创建 `DteAsyncTracker`。`WorkerCore` 在真实 SRAM 数据路径启用时，再把 `DteMemoryBridge` 绑定给 tracker。

### 1.2 当前直接耦合点

当前至少有两类直接 DTE 调用：

1. `llm/src/workercore/logic.cpp` 中普通/并行 SEND、RECV 路径直接调用 `dte->WaitForCredit()`、`Issue()`、`Release()`，流式路径还直接读取 `DteTransferContext` 的时刻和位宽。
2. `llm/src/workercore/workercore.cpp` 中 `Dte_async`、P2P endpoint 和结束检查直接调用 `dte_async->IssueToken()`、`WaitToken()`、`PollToken()`、`Fence()`、`CancelToken()` 和 `OutstandingCount()`。

这些直接调用意味着：

- 目前没有可插入的控制核命令队列；
- 无法单独统计 DTE 控制核的排队和忙碌时间；
- 无法在不复制大量分支的情况下切换共享/专用控制模式；
- 调用方持有 `DteTransferContext*`，不适合跨控制模块长期暴露对象所有权。

### 1.3 本扩展需要补齐的模型

现有 `DTEUnit` 已经建模 descriptor pending、channel/slot、launch、SPM/AXI 端口和完成事件；本扩展只补充 DTE 引擎之前的控制层：

```text
general-controller issue
    ↓
controller command queue wait
    ↓
controller decode/dispatch
    ↓
DTEUnit / DteAsyncTracker existing service
    ↓
controller completion notification
```

不得把已有 `gamma_ns`、`tau_launch_avg_ns`、DTE port service、SRAM/HBM/NoC 延迟再次计入控制核。

## 2. 范围与非目标

### 2.1 必须交付

- 可选的全局双控制核硬件配置。
- `cores[]` 中的逐核覆盖和异构配置。
- 回归安全的 `legacy_shared` 默认值。
- DTE 控制命令 FIFO、派发宽度、派发延迟和完成通知延迟。
- 共享控制与专用控制的统一 frontend API。
- SEND/RECV、DTE async 和 DTE endpoint 路径不再直接访问 `DTEUnit`。
- 同步、异步、poll、wait、fence、cancel 和 drain 的明确顺序语义。
- controller queue/backpressure/dispatch/complete trace 与统计。
- 配置、自测、端到端、异构和旧配置冻结回归。

### 2.2 第一版非目标

- 不为两个控制核分配独立 NoC endpoint 或 core ID。
- 不改变 workload 中 core 数量和 placement 语义。
- 不引入通用控制核或 DTE 控制核的独立指令存储、寄存器文件或软件镜像。
- 不把 `dte_channel_count` 改成控制核数量。
- 不自动迁移所有 P2P/collective 协议状态；第一版只要求所有 DTE 引擎操作经过统一控制接口。
- 不新增独立控制核频率/电压域；第一版与模拟器 `CYCLE` 使用同一时钟基准。
- 不宣称新增控制核的绝对功耗/面积准确，除非后续提供可追溯标定系数。

## 3. 配置契约

### 3.1 配置位置与优先级

控制核属于硬件结构，因此配置放在 hardware JSON，不放在 simulation JSON。

顶层 `control_cores` 是所有 core 的默认配置；`cores[].control_cores` 对单个显式 core 做 merge-patch 覆盖。优先级从低到高为：

```text
编译期回归安全默认值
    < 顶层 control_cores
    < cores[].control_cores
```

未在 `cores[]` 中出现的 core 继续继承前一个有效 core 配置，包括控制核配置。显式出现在 `cores[]` 中但未配置 `control_cores` 的 core 使用顶层默认值。

### 3.2 推荐格式

全芯片启用双控制核：

```json
{
  "x": 4,
  "control_cores": {
    "mode": "dual_dte_dedicated",
    "dte": {
      "command_queue_depth": 16,
      "dispatch_width": 1,
      "dispatch_latency_ns": 0,
      "completion_notify_latency_ns": 0
    }
  },
  "cores": [
    {
      "id": 0,
      "exu_x": 32,
      "sfu_x": 1024,
      "vec_x": 128,
      "dte_channel_count": 2,
      "dte_bit_width": 256
    }
  ]
}
```

异构覆盖：

```json
{
  "control_cores": {
    "mode": "dual_dte_dedicated",
    "dte": {
      "command_queue_depth": 16,
      "dispatch_width": 1,
      "dispatch_latency_ns": 2,
      "completion_notify_latency_ns": 1
    }
  },
  "cores": [
    {"id": 0, "exu_x": 32},
    {
      "id": 3,
      "exu_x": 32,
      "control_cores": {
        "mode": "legacy_shared"
      }
    }
  ]
}
```

关闭扩展的等价显式配置：

```json
{
  "control_cores": {
    "mode": "legacy_shared"
  }
}
```

### 3.3 字段定义

| 字段 | 类型 | 默认值 | 适用模式 | 语义 |
|---|---:|---:|---|---|
| `control_cores.mode` | string | `legacy_shared` | 全部 | `legacy_shared` 或 `dual_dte_dedicated` |
| `control_cores.dte.command_queue_depth` | integer | `16` | dedicated | 等待控制核派发的最大命令数 |
| `control_cores.dte.dispatch_width` | integer | `1` | dedicated | 每个控制周期最多派发的非阻塞命令数 |
| `control_cores.dte.dispatch_latency_ns` | integer | `0` | dedicated | 命令出队后到提交给 DTE backend 前的控制处理延迟 |
| `control_cores.dte.completion_notify_latency_ns` | integer | `0` | dedicated | backend 完成到通用控制核可观察完成之间的通知延迟 |

默认延迟为 0 是为了在首次启用结构时只观察资源解耦和队列效应，避免在没有标定依据时引入任意固定开销。后续实验可以显式覆盖。

### 3.4 配置校验

启动期必须拒绝：

- 未知 `mode`；
- `command_queue_depth == 0`；
- `dispatch_width == 0`；
- `dispatch_width > command_queue_depth`；
- 任意负延迟；
- 数值无法安全转换为 SystemC 时间或整数 cycle；
- `legacy_shared` 下出现仅 dedicated 有意义、且非默认的 DTE 控制资源字段。第一版建议严格拒绝，避免“配置被接受但静默忽略”。

错误信息必须包含完整字段路径、core ID 和非法值，例如：

```text
core[3].control_cores.dte.command_queue_depth must be > 0, got 0
```

### 3.5 与 simulation 配置的关系

`dte.use_beha_dte`、`dte.streaming`、`dte.async`、`dte.aggregation` 和 `dte.fine_grained_resources` 仍属于 simulation 配置。关系如下：

- `control_cores.mode` 描述硬件中是否有专用 DTE 控制核；
- `dte.use_beha_dte` 描述本次仿真是否启用 DTE 行为模型；
- 专用控制核存在但 DTE 行为关闭时，控制核为空闲，不应改变 legacy 非 DTE workload；
- hardware config 早于 simulation config 解析，因此 `ParseHardwareConfig()` 只做结构校验；如需跨配置校验，应在两份配置都解析后执行单独的 `ValidatePlatformConfig()`。

## 4. 配置类型与解析开发

### 4.1 新增类型

在 `llm/include/common/config.h` 增加值类型配置，禁止为简单配置引入 owning pointer：

```cpp
enum class ControlCoreMode {
    LEGACY_SHARED,
    DUAL_DTE_DEDICATED,
};

struct DteControllerHWConfig {
    uint32_t command_queue_depth = 16;
    uint32_t dispatch_width = 1;
    uint64_t dispatch_latency_ns = 0;
    uint64_t completion_notify_latency_ns = 0;
};

struct ControlCoresHWConfig {
    ControlCoreMode mode = ControlCoreMode::LEGACY_SHARED;
    DteControllerHWConfig dte;
};
```

`CoreHWConfig` 增加：

```cpp
ControlCoresHWConfig control_cores;
```

### 4.2 解析策略

在 `llm/src/utils/config_utils.cpp::ParseHardwareConfig()` 中先读取顶层默认对象，再为每个显式 core 构造有效配置：

```cpp
json global_control = j.value("control_cores", json::object());
json effective_control = global_control;
if (core.contains("control_cores"))
    effective_control.merge_patch(core.at("control_cores"));
effective_core["control_cores"] = effective_control;
```

随后在 `llm/src/common/config.cpp::from_json(CoreHWConfig&)` 中完成枚举转换、默认值和局部校验。

必须同步修正 `ParseHardwareConfig()` 中以下继承路径：

- 两个显式 core ID 之间自动补齐的 core；
- 最后一个显式 core 到 `GRID_SIZE-1` 的自动补齐 core；
- 多 die 下 `GetCoreHWConfigForGlobal()` 的 local-core 配置复用。

当前这些路径手工复制 `CoreHWConfig` 字段，新字段必须进入复制构造或改成安全的值复制，不能默默回到默认模式。

### 4.3 配置可见性

启动日志至少输出一次每种有效配置，而不是为每个同构 core 重复刷屏：

```text
ControlCore config: mode=dual_dte_dedicated queue=16 width=1 dispatch=2ns notify=1ns cores=0-2,4-15
ControlCore config: mode=legacy_shared cores=3
```

## 5. 目标模块与所有权

### 5.1 统一 frontend

新增统一调用边界：

```text
llm/include/dte/dte_control_frontend.h
llm/src/dte/dte_control_frontend.cpp
```

调用方只依赖 frontend，不感知共享或专用后端。建议接口分为物理 transfer 与 logical token 两组：

```cpp
struct DteTransferHandle {
    uint64_t xfer_id;
};

struct DteTransferSnapshot {
    DteTransferState state;
    uint64_t payload_bits;
    sc_time transmit_start_time;
    sc_time scheduled_completion_time;
};

class DteControlFrontend {
public:
    DteTransferHandle IssueTransfer(uint64_t payload_bits, DteDir dir);
    void WaitTransmitStart(DteTransferHandle handle);
    void WaitTransfer(DteTransferHandle handle);
    void ReleaseTransfer(DteTransferHandle handle);
    DteTransferSnapshot Snapshot(DteTransferHandle handle) const;
    uint32_t BitWidthBits() const;

    void IssueToken(const DteTokenCommand &command);
    void WaitToken(uint32_t token);
    bool PollToken(uint32_t token);
    void Fence();
    void CancelToken(uint32_t token);
    bool HasToken(uint32_t token) const;
    size_t OutstandingCount() const;
};
```

frontend 必须保证：

- 调用方不再持有 `DteTransferContext*`；
- handle 在完成和 release 前稳定；
- 无效、重复 release 和未知 token 给出确定异常；
- legacy backend 不引入额外 delta cycle 或排队。

### 5.2 legacy backend

`LegacyDteControlBackend` 对现有 `DTEUnit`、`DteAsyncTracker` 做零额外时序的直接转发。配置缺失或 `mode=legacy_shared` 时使用该 backend。

legacy 模式是兼容基准，要求：

- 不经过新增 command FIFO；
- 不增加 `wait(SC_ZERO_TIME)`；
- 原有 DTE trace 的开始/结束时刻不变；
- 原有异常和 drain 检查不被弱化。

### 5.3 专用 DTE 控制核

新增：

```text
llm/include/dte/dte_control_core.h
llm/src/dte/dte_control_core.cpp
```

建议模块：

```cpp
class DteControlCore final : public sc_module,
                             public DteControlBackend {
public:
    SC_HAS_PROCESS(DteControlCore);

    DteControlCore(sc_module_name name,
                   int core_id,
                   const DteControllerHWConfig &controller_config,
                   const DTEConfig &dte_config,
                   const DteAggregationConfig &aggregation,
                   Event_engine *event_engine);

    void BindMemoryBridge(DteMemoryBridge *bridge);

private:
    void dispatchWorker();
    void completionWorker();

    DteControllerHWConfig controller_config_;
    std::deque<std::shared_ptr<DteControlCommandState>> command_queue_;
    std::map<uint64_t, std::shared_ptr<DteControlCommandState>> inflight_;
    std::unique_ptr<DTEUnit> dte_;
    std::unique_ptr<DteAsyncTracker> async_;
    sc_event command_available_;
    sc_event command_space_available_;
    sc_event completion_progress_;
};
```

`DteControlCommandState` 可以拥有等待事件，但不能直接按值放入会移动元素的容器。建议用 `shared_ptr` 保持提交方与控制核共享的稳定生命周期；命令 payload 本身仍按值复制，禁止保存 `PrimBase*`。

### 5.4 WorkerCore 所有权

目标所有权：

```text
WorkerCore
├── WorkerCoreExecutor
├── DteControlFrontend
├── LegacyDteControlBackend 或 DteControlCore
│   ├── DTEUnit
│   └── DteAsyncTracker
├── DteMemoryBridge
├── SramAccessUnit
└── HbmByteTransport
```

`WorkerCoreExecutor` 只保留非 owning 的 `DteControlFrontend*`。`WorkerCore` 负责：

1. 根据本 core 的有效 `control_cores` 配置选择 backend；
2. 创建 backend 和 frontend；
3. 把 frontend 注入 executor；
4. 创建真实 SRAM/HBM 数据路径后调用 backend 的 `BindMemoryBridge()`；
5. 析构时先停止上层引用，再释放 backend。

SystemC 要求所有 `sc_module` 在第一次 `sc_start()` 前完成构造；不得在仿真运行中动态创建专用控制核。

## 6. 命令、队列与时序语义

### 6.1 命令类型

```cpp
enum class DteControlOpcode {
    ISSUE_TRANSFER,
    WAIT_TRANSMIT_START,
    WAIT_TRANSFER,
    RELEASE_TRANSFER,
    ISSUE_TOKEN,
    WAIT_TOKEN,
    POLL_TOKEN,
    FENCE,
    CANCEL_TOKEN,
};
```

每条命令携带：

- 单调递增的 `command_id`；
- `core_id`；
- opcode；
- 必要的 transfer/token/address/direction/payload 字段；
- 提交、入队、出队、backend issue、backend complete、notify 时刻；
- 完成状态或异常。

### 6.2 入队与背压

- FIFO 占用小于 `command_queue_depth` 时，提交立即成功。
- FIFO 满时，提交方等待 `command_space_available_`，并累计 queue-full stall。
- 释放一个 FIFO 项后在同一合法调度点通知等待者。
- 多个 SystemC producer 提交时，以进入 frontend 的顺序分配 `command_id`，用该 ID 固定总序。
- 禁止丢弃命令、覆盖旧项或以无界容器绕过配置深度。

### 6.3 派发

- 每个控制周期最多从 FIFO 取出 `dispatch_width` 条可以派发的命令。
- 每批命令在 `dispatch_latency_ns` 后进入 DTE backend。
- `dispatch_latency_ns` 是 DTE 接受 descriptor 之前的控制处理开销；`gamma_ns + tau_launch_avg_ns` 仍是 DTE backend 接受后的 launch 模型，两者 trace 必须分段，避免解释时混淆。
- 第一版不自动从 `tau_launch_avg_ns` 中扣除 controller latency。若实验同时配置非零值，报告必须说明两者分别代表 controller 与 engine 两段开销。

### 6.4 完成通知

- backend 完成后，经过 `completion_notify_latency_ns`，提交方才能观察到完成。
- completion notification 不等于 DTE transfer release；同步命令和 token 命令按现有生命周期显式 release。
- 控制核保留完成状态直到提交方消费，不能因事件早于 waiter 而丢失唤醒。
- 异常存入 command state，并在提交方等待或查询时重新抛出；后台线程不能只打印日志后继续。

### 6.5 程序顺序与 barrier

按 `command_id` 冻结以下语义：

- 异步 ISSUE：backend 接受并建立 handle/token 后，本命令完成；数据传输可继续在后台运行。
- 同步 ISSUE/WAIT：作为 barrier，完成前不允许同一控制命令流中更晚的依赖命令越过。
- POLL：返回执行到该命令时的 token 状态，不 release token。
- FENCE：只等待 fence 之前已经接受的命令和 token；更晚命令不能被错误纳入该 fence。
- CANCEL：只能取消现有规则允许取消的 token；结果确认后才能完成命令。
- RELEASE：必须精确一次；完成前 release、重复 release 或未知 handle 均报错。

建议由 `dispatchWorker()` 设置 `barrier_active`，由 `completionWorker()` 观察 backend event 并解除 barrier。不要通过每周期盲目轮询完成状态；优先为 `DTEUnit`/`DteAsyncTracker` 暴露只读进度事件或非阻塞 retire API。

### 6.6 DteAsyncTracker 补充接口

现有 `WaitToken()` 和 `Fence()` 会在调用线程中阻塞并完成 release。为了让专用控制核清晰区分派发与完成，建议增加：

```cpp
bool TryRetireToken(uint32_t token);
bool TryFenceThrough(uint64_t issue_sequence);
const sc_event &StateChangedEvent() const;
```

要求：

- `TryRetireToken()` 未完成时返回 false，不改变记录；完成时执行现有 WaitAndRelease 的等价回收。
- fence 捕获明确的 issue watermark，不能等待 fence 之后的新 token。
- staged aggregation token 在 wait/fence 时继续按现有规则 flush。
- memory bridge 的完成、release 和回滚仍由 tracker 统一处理。

## 7. WorkerCore 路径迁移

### 7.1 迁移原则

所有 DTE engine 和 async tracker 操作必须通过 `DteControlFrontend`。完成后以下目录中不应再出现业务代码直接调用 `dte->` 或 `dte_async->`：

```text
llm/src/workercore/
llm/include/workercore/
```

允许的例外仅限 backend 自身实现和只读测试 fixture。

### 7.2 SEND/RECV 路径

替换 `logic.cpp` 中：

- 普通 SEND_DATA 的 credit、issue、stream start、done、release；
- 并行 SEND batch 的多 descriptor issue 与逐项 release；
- RECV_DATA 的 destination transfer issue 与 release；
- streaming 所需 bit width、transmit start 和 scheduled completion 查询。

streaming 路径使用 `DteTransferHandle` 和只读 snapshot，不能把 backend 内部 `DteTransferContext*` 暴露给 send/recv 线程。

### 7.3 Dte_async 路径

替换 `workercore.cpp::execute_dte_async()` 中：

- ISSUE；
- WAIT；
- POLL；
- FENCE；
- CANCEL；
- token collision 和 outstanding 查询。

地址解析可以暂时保留在通用控制核，也可以作为值字段传给 DTE 控制核。第一版建议保留现有 `SramRegionTable` 权限与地址解析位置，把解析后的 byte 地址放入命令，避免重复实现 region 语义。

### 7.4 P2P endpoint 与 collective

第一版保留现有 NoC 物理发送 helper 和 P2P/collective session 容器在 `WorkerCoreExecutor`，但凡涉及 DTE descriptor、token、memory bridge 或 DTE 完成的操作都必须经过 frontend。

后续若要模拟“DTE 控制核运行完整通信固件”，再单独迁移：

- `p2p_tx_worker()`；
- `p2p_request_admission_worker()`；
- P2P endpoint session/token map；
- collective program/acceleration worker；
- DTE endpoint 的 SRAM/HBM source/destination 发起。

该迁移需要新增 DTE controller 到共享 NoC injector 的请求接口，不应让控制核直接抢占 `channel_o`、`ctrl_channel_o` 或 `send_helper_write`。

### 7.5 drain 与结束条件

以下位置必须改为读取 frontend 的统一 residual：

- primitive queue 为空时的 outstanding token 检查；
- `SEND_DONE` 前的 DTE/P2P/collective drain；
- 仿真结束残留检查；
- watchdog 和诊断输出。

建议提供：

```cpp
struct DteControlResidual {
    size_t queued_commands;
    size_t inflight_commands;
    size_t active_transfers;
    size_t logical_tokens;
    size_t pending_notifications;
};
```

所有字段为 0 才表示 DTE 控制路径 drained。

## 8. Trace、统计与可解释性

### 8.1 Trace

dedicated 模式新增：

| Trace | 范围 | 必需字段 |
|---|---|---|
| `DTE_CTRL_queue_wait` | FIFO 满导致的提交等待 | core、command_id、opcode、occupancy、depth |
| `DTE_CTRL_dispatch` | 控制核处理与派发 | core、command_id、opcode、dispatch_width、latency |
| `DTE_CTRL_barrier` | WAIT/FENCE/同步命令阻塞 | core、command_id、token/watermark |
| `DTE_CTRL_notify` | 完成通知 | core、command_id、status、notify_latency |

trace 分段必须满足：

```text
controller queue/dispatch
    不与 DTE_pending/DTE_launch 重叠计为同一段
DTE pending/launch/port/transmit
    不重复 NoC/D2D/HBM media service
controller notify
    只表示完成可见性
```

### 8.2 统计

至少提供：

- `commands_submitted`；
- `commands_dispatched`；
- `commands_completed`；
- `queue_full_stalls`；
- `queue_full_stall_cycles`；
- `queue_high_watermark`；
- `barrier_stall_cycles`；
- `dispatch_busy_cycles`；
- `completion_notify_cycles`；
- 按 opcode 分类的命令数和平均延迟。

legacy 模式的新增控制核统计应为 0，并在报告中显示 `mode=legacy_shared`，不能伪造一个无时序意义的 dedicated controller。

## 9. 分阶段开发计划

### C0：冻结配置与兼容契约

目标：配置能解析、继承和校验，但尚不改变运行路径。

开发项：

- [x] 增加 `ControlCoreMode`、`DteControllerHWConfig`、`ControlCoresHWConfig`。
- [x] 增加顶层默认与逐核 merge-patch。
- [x] 修复自动补齐 core 时的配置复制。
- [x] 增加非法配置测试和有效配置打印。
- [x] 更新默认硬件示例，但默认仍为 `legacy_shared`。

验收：

- [x] 配置缺失、显式 legacy、全局 dedicated、逐核覆盖均解析正确。
- [x] 非法 mode、深度、宽度和延迟在启动期失败。
- [x] 不改变任何现有仿真结果。

### C1：统一 frontend 与 legacy backend

目标：消除 WorkerCore 对 DTE 内部对象的直接依赖，暂不启用新时序。

开发项：

- [x] 新增 `DteControlFrontend` 和 backend 接口。
- [x] 新增 `LegacyDteControlBackend`。
- [x] 用 handle/snapshot 替换外部 `DteTransferContext*`。
- [x] 迁移 `logic.cpp` 的 SEND/RECV 调用。
- [x] 迁移 `execute_dte_async()` 和 drain 查询。
- [x] 增加静态检查，防止 workercore 重新直接调用 DTE 对象。

验收：

- [x] 全部旧 DTE selftest 通过。
- [x] 默认 workload 的事件顺序和 sim-time 不变。
- [x] `SPEC_USE_BEHA_DTE=false` 冻结回归不变。

### C2：专用 DTE 控制核基础模型

目标：dedicated 模式具备真实 FIFO、派发和通知时序。

开发项：

- [x] 新增 `DteControlCore` SystemC 模块。
- [x] 实现有界命令 FIFO 和 queue-full backpressure。
- [x] 实现 `dispatch_width` 和 `dispatch_latency_ns`。
- [x] 实现稳定 command state、完成通知和异常传播。
- [x] 实现 trace、统计和 residual。
- [x] 完成 WorkerCore 按配置选择 backend。

验收：

- [x] queue depth 1/2/16 的占用和背压可验证。
- [x] dispatch width 1/2 的同周期派发数量正确。
- [x] dispatch/notify 延迟精确到 cycle，使用明确向上取整。
- [x] 不新增 `std::thread`、`std::async` 或仿真期动态 `sc_module`。

### C3：同步、异步与 barrier 收口

目标：ISSUE/WAIT/POLL/FENCE/CANCEL 在专用控制核下具备完整程序顺序。

开发项：

- [x] 为 tracker 增加非阻塞 retire、watermark fence 和状态事件。
- [x] 实现 async ISSUE 接受完成语义。
- [x] 实现同步 transfer、WAIT 和 FENCE barrier。
- [x] 实现 POLL 不回收、WAIT 精确回收。
- [x] 实现 CANCEL、异常和 memory bridge 回滚。
- [x] 保证 staged aggregation 在 wait/fence 下按既有规则 flush。

验收：

- [x] `ISSUE → COMPUTE → WAIT` 存在可解释的 compute/DTE overlap。
- [x] fence 不等待 fence 之后的 token。
- [x] POLL 完成后 token 仍存在，WAIT 后 token 消失。
- [x] 取消和异常不泄漏 command、transfer、token 或 SRAM/HBM lease。

### C4：SEND/RECV、P2P 与 collective 端到端

目标：所有生产路径在 dedicated 模式下通过统一控制接口。

开发项：

- [x] 普通 SEND/RECV。
- [x] parallel SEND batch。
- [x] DTE streaming source/destination。
- [x] DTE endpoint P2P sync/async。
- [x] collective/DCA 涉及的 DTE transfer 和 token。
- [x] `SEND_DONE` 和程序结束 drain。

验收：

- [x] physical/behavioral NoC 均通过。
- [x] legacy_private/distributed_hbm 均通过。
- [x] real SRAM data path 数据、checksum、valid 和时序正确。
- [x] 所有 endpoint residual 为 0。

### C5：实验、文档与最终验收

目标：形成可复现实验和用户文档。

开发项：

- [x] 增加 dedicated on/off 硬件配置样例。
- [x] 增加 queue depth、dispatch width、dispatch latency sweep。
- [x] 输出 controller 利用率、queue stall、DTE 利用率和总 sim-time。
- [x] 更新硬件配置文档。
- [x] 在 `notes/extensions/DTE_control_core/log/` 记录各阶段结果。

验收：

- [x] 所有参数对模型有可观察、可解释的影响。
- [x] 默认值、范围、单位和计费边界完整记录。
- [x] 实验可由单条脚本命令复现。

## 10. 测试矩阵

### 10.1 配置测试

- 缺失 `control_cores`。
- 显式 `legacy_shared`。
- 全局 `dual_dte_dedicated`。
- 单 core 覆盖回 legacy。
- 多个显式 core 之间的自动继承。
- 矩形 core mesh、多 die 下 local-core 配置复用。
- 所有非法配置和边界值。

### 10.2 控制核单元测试

- 空队列、单命令、连续命令。
- queue depth=1 时第二条命令背压。
- 多 producer 同时提交，`command_id` 唯一且顺序稳定。
- dispatch width 小于、等于、大于当前队列项数。
- notify 早于 waiter 注册时仍能正确消费完成。
- backend 异常能传播到正确提交者。
- 重复 wait/release、未知 handle/token 被拒绝。

### 10.3 DTE 生命周期

- blocking transfer。
- async issue/wait。
- poll false → poll true → wait。
- fence 前后分别 issue token。
- cancel staged aggregation token。
- cancel active token。
- command FIFO 满与 DTE pending/credit 满同时发生。
- controller queue、DTE pending 和 SRAM/HBM queue 三层背压不丢命令。

### 10.4 端到端组合

| 维度 | 覆盖值 |
|---|---|
| controller | legacy / dedicated |
| DTE | off / V1 / streaming / async / aggregation / fine-grained |
| NoC | physical / behavioral |
| dispatcher | sequential / 支持范围内的 parallel |
| SRAM | timing-only / real-data-path |
| memory | legacy_private / distributed_hbm behavioral / distributed_hbm DRAMSys |
| topology | single die / multi die |
| core config | homogeneous / per-core heterogeneous |

### 10.5 冻结回归

至少冻结：

- 默认 hardware/simulation smoke；
- DTE V0–V4 selftest；
- DTE endpoint P2P；
- DTE aggregation/coalescing；
- SRAM DTE bridge；
- NoC collective/DCA；
- D2D legacy 与 bounded/behavioral 配置；
- `SPEC_USE_BEHA_DTE=false` 路径。

legacy 模式验收应比较功能输出、异常类型、关键 trace 和最终 sim-time；不能只比较“测试进程退出码为 0”。

## 11. 最终验收标准

全部满足后才可宣布 DTE 专用控制核完成：

- [x] `control_cores.mode` 可在 hardware JSON 中配置，缺省为 `legacy_shared`。
- [x] 顶层同构配置与 `cores[]` 逐核异构覆盖均生效。
- [x] dedicated 模式每个计算 core 恰好拥有一个独立 DTE 控制核状态；不同 core 不共享 FIFO、命令 ID、token 或统计。
- [x] `TOTAL_CORES`、NoC endpoint 和 workload placement 不因内部控制核数量改变。
- [x] legacy 模式不引入新增排队或 delta-cycle 时序变化。
- [x] dedicated 模式的 queue depth、dispatch width、dispatch latency 和 notify latency 均有测试证明实际生效。
- [x] workercore 生产代码不再直接访问 `DTEUnit`/`DteAsyncTracker`。
- [x] 不存在裸 primitive 指针跨控制队列，不存在完成事件丢失和 command state 生命周期错误。
- [x] WAIT/POLL/FENCE/CANCEL 和 aggregation/memory bridge 回滚语义保持正确。
- [x] SEND/RECV/P2P/collective 与真实 SRAM/HBM 数据路径全部通过。
- [x] 程序结束时 controller、DTE、token、P2P 和 memory residual 全为 0。
- [x] trace 能分离 controller、DTE engine、SRAM/HBM 和 NoC/D2D 延迟，且不存在重复计费。
- [x] 文档、配置样例、自动化测试和阶段开发记录齐全。

验收证据：C0-C3 阶段记录、`log/C4端到端验收.md`、`log/C5实验与最终验收.md` 与 `log/C5_sweep_results.json`。

## 12. 风险与规避

### 12.1 把内部控制核误当成独立 core

风险：修改 `TOTAL_CORES` 会连锁影响 16-bit endpoint、router 数量、mapping、HOST/HBM attach、数组维度和 workload core ID。

规避：控制核始终是 `WorkerCore` 内部子模块，不进入 grid 地址空间。

### 12.2 配置存在但不产生模型效应

风险：只创建一个 `SC_THREAD`，仍直接从 executor 调用 DTE，导致 dedicated on/off 结果相同。

规避：所有 DTE 操作必须经过有界 FIFO；每个新增参数必须有独立测试和 trace 证据。

### 12.3 legacy 路径被新队列改变

风险：即使延迟配置为 0，额外 event/delta cycle 也可能改变仲裁顺序和 sim-time。

规避：legacy backend 直接转发，不经过 dedicated FIFO 或后台线程。

### 12.4 重复计费

风险：controller dispatch、`tau_launch_avg_ns`、DTE launch、SRAM/HBM 和 NoC 被错误相加两次。

规避：trace 明确划分 controller-before-issue、DTE-after-accept、memory/network data-plane 三个边界；实验报告逐段核对。

### 12.5 barrier 队头阻塞或 fence 范围错误

风险：阻塞式 `WaitToken()` 卡住控制核，或者 fence 错误等待后续 token。

规避：使用 command ID/issue watermark、`barrier_active` 和非阻塞 completion worker；为 fence 前后并发 issue 编写定向测试。

### 12.6 指针与事件生命周期

风险：把 `PrimBase*`、可移动 `sc_event` 或 backend context 裸指针放进队列，导致 refill/容器移动后悬空。

规避：命令按值复制；command state 使用稳定所有权；对外只暴露 ID handle 和 snapshot。

### 12.7 用户已有修改冲突

风险：当前 `workercore`、DTE、P2P/collective 路径可能同时有其他开发分支修改。

规避：按 C0–C5 小步提交；每阶段先重新盘点直接 DTE 调用和 dirty diff，不覆盖无关改动；优先新增 façade，再逐路径迁移。

## 13. 建议交付文件

```text
llm/include/common/config.h
llm/src/common/config.cpp
llm/src/utils/config_utils.cpp

llm/include/dte/dte_control_frontend.h
llm/src/dte/dte_control_frontend.cpp
llm/include/dte/dte_control_core.h
llm/src/dte/dte_control_core.cpp

llm/include/dte/dte_async.h
llm/src/dte/dte_async.cpp

llm/include/workercore/workercore.h
llm/src/workercore/workercore.cpp
llm/src/workercore/logic.cpp

llm/test/default/hardware.json
llm/test/dte/hardware/control_core_legacy.json
llm/test/dte/hardware/control_core_dedicated.json
llm/test/dte/hardware/control_core_heterogeneous.json

notes/extensions/DTE_control_core/log/C0_development.md
notes/extensions/DTE_control_core/log/C1_development.md
notes/extensions/DTE_control_core/log/C2_development.md
notes/extensions/DTE_control_core/log/C3_development.md
notes/extensions/DTE_control_core/log/C4_development.md
notes/extensions/DTE_control_core/log/C5_development.md
```

若最终文件名因现有测试组织方式调整，必须保持模块职责、配置契约和验收覆盖不变。
