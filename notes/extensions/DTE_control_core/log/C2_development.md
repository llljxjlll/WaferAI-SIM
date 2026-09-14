# C2：专用 DTE 控制核基础模型

日期：2026-08-25  
状态：完成

## 实现

- 每个 dedicated 计算 core 构造一个内部 `DteControlCore` SystemC 模块。
- 实现有界 FIFO、queue-full 背压、dispatch width、dispatch latency 和 notify latency。
- command/transfer 状态使用稳定所有权；异常保存后向提交者重抛。
- trace 分离 `DTE_CTRL_queue_wait`、`DTE_CTRL_dispatch`、DTE backend 和 `DTE_CTRL_notify`。
- residual 同时统计 queued/inflight transfer 与 logical token；完成后显式 release。
- legacy 模式不实例化控制 FIFO，也不产生 controller trace/统计。

## 证据

- control-core selftest 首轮 21/21 通过。
- legacy/dedicated 端到端：18885 ns / 18893 ns；正 dispatch/notify 延迟产生可解释差异。
- dedicated 有 dispatch/notify trace，legacy 无 `DTE_CTRL_*` trace。
- 后续 C3 测试继续扩充 command opcode、barrier、统计和 cycle 取整覆盖。

