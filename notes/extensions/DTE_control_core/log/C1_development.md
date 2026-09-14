# C1：统一 frontend 与 legacy backend

日期：2026-08-25  
状态：完成

## 实现

- 新增 `DteControlFrontend`，统一 physical transfer 与 logical token 调用面。
- 外部使用稳定 `DteTransferHandle` 和值类型 snapshot，不持有 backend context 指针。
- 普通/并行 SEND、RECV 和 streaming 已迁移到 frontend。
- `execute_dte_async()` 的 ISSUE/WAIT/POLL/FENCE/CANCEL 和 outstanding 查询已迁移。
- legacy 路径仍直接执行原 `WaitForCredit() + Issue()`，不进入新 FIFO、不增加 wait。

## 兼容证据

- legacy V3 blocking/overlap：421 ns / 345 ns，与冻结值一致。
- DTE V0/V3/V3b/V4 selftest 全部通过。
- default workload smoke 通过。

