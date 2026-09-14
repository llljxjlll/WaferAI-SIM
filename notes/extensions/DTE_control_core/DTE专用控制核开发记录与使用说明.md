# DTE 专用控制核开发记录与使用说明

> 日期：2026-08-25  
> 依据：`DTE专用控制核开发方案.md`  
> 交付状态：C0-C5 已完成；配置、物理与 logical FIFO、生产路径、实验和文档均已收口。

## 1. 已交付能力

本次实现为每个计算 core 增加了一个可选的内部 DTE 控制核。它不改变
`TOTAL_CORES`、NoC endpoint 数或 workload 映射；仅在
`dual_dte_dedicated` 模式下实例化独立 SystemC 控制模块。

已实现：

- `legacy_shared` 与 `dual_dte_dedicated` 两种模式；
- 顶层默认配置和 `cores[].control_cores` 按核覆盖；
- 有界 command FIFO、FIFO 满背压、每周期派发宽度；
- descriptor 接受前派发延迟和 backend 完成后的通知延迟；
- 稳定 handle 与值类型 snapshot，不把 backend context 指针暴露给调用方；
- legacy/dedicated 统一 frontend；
- 物理 SEND/RECV、并行 SEND 和 streaming 路径接入统一 frontend；
- controller queue/dispatch/notify trace、统计和 residual；
- ISSUE/WAIT/POLL/FENCE/CANCEL logical opcode 进入同一 FIFO，并按程序顺序执行；
- watermark fence、非阻塞 barrier worker、aggregation flush 和 memory bridge 异常回滚；
- 逐 opcode 延迟、barrier stall、queue high-watermark 与最终 drain 生产统计；
- 可单命令复现的 queue depth、dispatch width、dispatch/notify latency sweep；
- 独立 C++ selftest、CTest 入口和双模式端到端 runner。

legacy 模式保持原来的 `WaitForCredit() + Issue()` 直接路径，不创建控制核
FIFO，也不增加 delta cycle 或定时等待。

## 2. 配置方式

缺省不写 `control_cores` 时使用 `legacy_shared`，因此旧配置行为不变。

```json
{
  "control_cores": {
    "mode": "dual_dte_dedicated",
    "dte": {
      "command_queue_depth": 16,
      "dispatch_width": 1,
      "dispatch_latency_ns": 2,
      "completion_notify_latency_ns": 2
    }
  }
}
```

参数约束：

| 参数 | 默认值 | 约束 | 含义 |
|---|---:|---|---|
| `mode` | `legacy_shared` | 两个枚举值之一 | 控制核模式 |
| `command_queue_depth` | 16 | 正整数 | 每个 core 的控制命令 FIFO 深度 |
| `dispatch_width` | 1 | `1..depth` | 每个控制周期最大派发数 |
| `dispatch_latency_ns` | 0 | 非负整数 | descriptor 交给 DTE 前的控制延迟 |
| `completion_notify_latency_ns` | 0 | 非负整数 | DTE 完成到调用方可见的延迟 |

顶层配置会传播到所有 core。单个 core 可覆盖：

```json
{
  "control_cores": {
    "mode": "dual_dte_dedicated",
    "dte": {
      "command_queue_depth": 16,
      "dispatch_width": 1
    }
  },
  "cores": [
    {"id": 0},
    {
      "id": 3,
      "control_cores": {"mode": "legacy_shared"}
    }
  ]
}
```

按核切回 `legacy_shared` 且没有显式 `dte` 时，会清除继承的 dedicated
资源字段并恢复默认值。legacy 模式配置非默认 dedicated 参数会报错，避免
出现“参数已写但实际不生效”的歧义。

可直接运行的完整配置：

- `llm/test/dte/hardware/control_core_legacy.json`
- `llm/test/dte/hardware/control_core_dedicated.json`
- `llm/test/dte/hardware/control_core_heterogeneous.json`

异构样例以顶层 dedicated 为默认，core 1 覆盖为 legacy，core 2 再显式恢复
dedicated，避免后续自动补齐 core 继续继承 legacy。

## 3. 运行结构

物理传输路径如下：

```text
WorkerCoreExecutor
  -> DteControlFrontend
       -> legacy: DTEUnit（直接转发）
       -> dedicated: DteControlCore（FIFO/dispatch/notify）
                         -> DTEUnit（数据面）
```

每个 dedicated core 独立持有 FIFO、handle 空间、统计和 SystemC worker。
控制核只建模控制面延迟；DTE launch、端口、SRAM/HBM、NoC/D2D 延迟仍由
原模块统计，避免重复计费。

新增 trace stage：

- `DTE_CTRL_queue_wait`
- `DTE_CTRL_dispatch`
- `DTE_CTRL_notify`

legacy 模式不会产生上述 trace。

## 4. 主要代码变更

| 范围 | 文件 |
|---|---|
| 配置结构与校验 | `llm/include/common/config.h`、`llm/src/common/config.cpp` |
| 顶层/按核继承 | `llm/src/utils/config_utils.cpp` |
| 专用控制核 | `llm/include/dte/dte_control_core.h`、`llm/src/dte/dte_control_core.cpp` |
| 统一调用边界 | `llm/include/dte/dte_control_frontend.h`、`llm/src/dte/dte_control_frontend.cpp` |
| logical FIFO 与 barrier | `llm/include/dte/dte_async.h`、`llm/src/dte/dte_async.cpp`、`llm/src/dte/dte_control_core.cpp` |
| WorkerCore 接入 | `llm/include/workercore/workercore.h`、`llm/src/workercore/workercore.cpp`、`llm/src/workercore/logic.cpp` |
| 自测入口 | `llm/src/dte/dte_control_core_selftest.cpp`、`llm/unittest/npusim.cpp`、`CMakeLists.txt` |
| 集成夹具 | `llm/test/dte/hardware/control_core_*.json`、`llm/test/dte/run_test_dte_control_core.py` |
| C5 sweep | `llm/test/dte/run_dte_control_core_sweep.py`、`llm/test/dte/workload/control_core_parallel_four.json` |

## 5. 验证结果

最终验证 binary 为 `/workspace/build-debug-final/npusim`。

| 门禁 | 结果 |
|---|---|
| 全量 `npusim` 编译 | PASS |
| DTE control-core selftest | PASS，32/32 checks |
| MoE runtime capture helper | PASS，25/25；通用零区间仍 fail-closed |
| DTE 专项 CTest | PASS，7 项均已注册并纳入 C4 门禁 |
| legacy/dedicated 端到端 | PASS；18885 ns / 18905 ns，async opcode 与 residual 同时通过 |
| DTE V0 / V3a / V3b / V4 | PASS；66 / 33 / 21 / 23 checks |
| C4 生产路径与 drain | PASS；frontend 覆盖、SEND/RECV、P2P、collective、SRAM/HBM 和 residual 见 `log/C4端到端验收.md` |
| C5 参数实验 | PASS；16/16 cases、8/8 验收门禁 |
| queue depth | depth 1/2/4/8 的 stall 次数为 7/2/0/0 |
| dispatch width | width 1/2/4 的每 core 最大批量为 1/2/4 |
| dispatch latency | 0/2/4 ns 实测 0/2/4 ns，sim-time 1642/1662/1682 ns |
| notify latency | 0/2/4 ns 实测 0/2/4 ns，sim-time 1642/1660/1684 ns |

快速复验命令：

```bash
cmake --build build-debug-final --target npusim --parallel 2
ctest --test-dir build-debug-final -R ^dte_ --output-on-failure -j1
python3 llm/test/dte/run_test_dte_control_core.py \
  --npusim build-debug-final/npusim
python3 llm/test/dte/run_dte_control_core_sweep.py \
  --npusim build-debug-final/npusim
```

C5 runner 串行执行 case，并使用 `build-debug-final/.dte_control_core_sweep.lock`
保护共享 `events.json`。完整 JSON、逐 case hardware/stdout 和指标口径见：

- `notes/extensions/DTE_control_core/log/C5_sweep_results.json`
- `notes/extensions/DTE_control_core/log/C5_sweep_artifacts/`
- `notes/extensions/DTE_control_core/log/C5实验与最终验收.md`

## 6. 最终模型边界

物理 transfer 与 logical ISSUE/WAIT/POLL/FENCE/CANCEL 均经过统一 frontend；
dedicated 模式的 logical opcode 与物理命令共享有界 FIFO。WAIT/FENCE 由非阻塞
completion worker 执行，watermark 保证 fence 不等待后续 token；POLL 不回收，
WAIT 精确回收，CANCEL、aggregation flush 和 memory bridge 异常路径均清理资源。

`[DTE_CTRL_STATS]` 按 core 输出 submitted/dispatched/completed、queue stall、
high-watermark 与 residual；`[DTE_STATS]` 输出 DTE issued/completed/backpressure
和 pending/active/inflight。C5 同时从 controller trace 与 DTE transmit trace 计算
controller 利用率、queue stall、DTE 利用率和 sim-time，控制面与数据面不重复计费。

控制核仍是 tile 内部子模块，不增加 NoC endpoint/core ID，也不声明未经标定的
绝对面积、功耗或独立频率域。C0-C5 阶段证据保存在本目录 `log/` 下。
