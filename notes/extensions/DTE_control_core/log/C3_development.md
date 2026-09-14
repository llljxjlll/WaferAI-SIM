# C3：DTE 逻辑命令统一入队开发记录

## 目标与结果

C3 将 dedicated 模式的物理传输控制命令和逻辑 token 命令统一提交到同一个有界 FIFO。dispatch 线程只负责按 `dispatch_width` 取命令、施加周期化后的 dispatch latency 并启动独立 worker，不在 DTE credit、传输完成、token 依赖或 fence 上阻塞。

legacy_shared 模式仍由 `DteControlFrontend` 直接调用既有 `DTEUnit`/`DteAsyncTracker`，未增加控制核队列等待。

## 实现范围

统一 opcode：

- 物理：`ISSUE_TRANSFER`、`WAIT_TRANSMIT_START`、`WAIT_TRANSFER`、`RELEASE_TRANSFER`。
- 逻辑：`ISSUE_TOKEN`、`WAIT_TOKEN`、`POLL_TOKEN`、`FENCE`、`CANCEL_TOKEN`。

每条命令拥有单调 `command_id`、统一状态、入队/dispatch/notify 时间、异常结果和完成事件。物理传输对外仅使用稳定 `DteTransferHandle` 与值语义 `DteTransferSnapshot`，不暴露 `DteTransferContext *`。

逻辑语义：

- `ISSUE_TOKEN` 进入 FIFO 后由独立 worker 调用 tracker；credit/hazard 等待不会阻塞 dispatch。
- `WAIT_TOKEN` 是顺序 barrier：使用 `TryRetireToken` 和状态事件等待，完成并回收 token 后才放行后续逻辑命令。
- `POLL_TOKEN` 只读取完成状态，不回收 token。
- `FENCE` 捕获 exclusive issue-sequence watermark，只等待并回收 fence 之前的 token；捕获 watermark 后允许更晚的 issue 启动。
- `CANCEL_TOKEN` 支持已发射单描述符和 staged aggregation tail；取消完成会唤醒状态观察者。

## 时序、并发与异常处理

- dispatch/notify latency 使用 `CYCLE` 向上取整；当前 `CYCLE=2ns`，所以 `1/2/3ns` 分别成为 `2/2/4ns`。
- 修复了 dispatch event 早于 backend 指针绑定的竞态；物理 backend 绑定后才发出可见事件。
- DTE completion watcher 每次唤醒后按 xfer id 重新查询 batch，不跨 wait 保存 backend 裸指针，cancel/release 不会留下悬空 watcher。
- cancelled context 的 `done` 会通知等待者；context 延迟一个 delta 回收，使等待者可稳定观察 `CANCELLED`。
- memory bridge 的失败在 token、bridge record、physical batch 和 DTE context 清理完成后重抛，异常路径 residual 可归零。
- FIFO 满统计按“发生过等待的提交命令”计一次；stall cycle 累计实际等待时长。

## façade 与接线

`DteControlFrontend` 统一提供物理 API、logical token API、`BindMemoryBridge` 和 residual 查询。dedicated 构造时将 `DteAsyncTracker` 绑定到控制核；WorkerCore 的 memory bridge 接线已改为只经过 frontend，不再直连 `dte_async`。

## 可观测性

`DteControlCoreStatistics` 包含 submitted/enqueued、dispatched、completed、queue stalls/max occupancy，以及 queue stall cycles、barrier cycles、dispatch busy cycles、notify cycles和逐 opcode 数量/总延迟。

`DteControlResidual` 分解为 queued commands、inflight commands、active transfers、logical tokens、pending notifications。trace detail 包含 core、command id、opcode、dispatch width、dispatch/notify latency、状态与队列占用。

## 验证

定向编译覆盖：

- `dte_control_core.cpp`
- `dte_control_frontend.cpp`
- `dte_async.cpp`
- `dte_unit.cpp`
- `workercore.cpp`
- `dte_control_core_selftest.cpp`

上述文件均已通过 C++17 单文件编译。专用 selftest 增加了周期取整、物理全命令入队、poll 不回收、wait 回收、WAIT barrier、fence watermark、staged aggregation wait/cancel、异常传播后命令回收、逐 opcode 统计与最终 residual=0 检查。

最终验证结果：

- `cmake --build build-debug-final --target npusim -j4`：通过。
- `npusim --dte-control-core-selftest`：PASS，32/32；包含真实 HBM transport 故障注入，确认错误重抛前 bridge/token/batch/context/residual 全部归零。
- `npusim --dte-v0-selftest`：PASS，66 checks。
- `npusim --dte-v3-selftest`：PASS，33/33。
- `npusim --dte-v3b-selftest`：PASS，21/21。
- `npusim --dte-v4-selftest`：PASS，23/23。

首次运行曾暴露 completion watcher 只等待状态事件、而 timed DTE completion 未发出该事件的问题；补齐 completion `state_changed` 通知后，完整 logical test 才能继续到 32 项并全部通过。该记录用于避免把 SystemC 因无未来事件提前返回误判为测试完成。

## 已知边界

- 同一 token 的互相矛盾操作仍由 tracker 拒绝并将异常传播给提交者；控制核保证失败 command 自身被回收。
- physical `Release` 保留既有语义：目标尚未完成时返回 false；调用本身仍作为 FIFO command 计时和计数。
- fence 只覆盖捕获 watermark 之前已分配 issue sequence 的 token，符合并发 producer 下的稳定 fence 边界。
