# 分布式 HBM R3 开发记录

日期：2026-08-04
状态：非阻塞 DRAMSys backend 与统一生命周期构建器已完成

## 1. 阶段范围

R3 在不改变 `CoreMemAdapter/MemEndpointUnit` 契约的情况下，把 behavioral backend
替换为真实 DRAMSys 时序，并按配置统一创建和持有 backend/endpoint。NoC flit 到 TLM
transaction 的生产桥接属于 R4。

## 2. 非阻塞 TLM 生命周期

- `HBMBackend::Submit` 支持异步完成；
- `DRAMSysHBMBackend` 使用 `nb_transport_fw/bw`、PEQ 和
  BEGIN_REQ/END_REQ/BEGIN_RESP/END_RESP 四阶段协议；
- END_REQ 后立即尝试发送下一笔，允许多笔 transaction 同时留在 DRAMSys；payload
  通过 TLM memory-manager refcount 延迟销毁，避免 END_RESP 后 use-after-free；
- 删除旧固定 60ns blocking 行为，service time 取实际 issue→BEGIN_RESP；
- 强制 `StoreMode=Store`，参考配置同时提供真实数据保存和时序。

## 3. 生命周期和统计

- `BuildHBMBackends()` 只能在第一次 `sc_start` 前调用；按
  `backend=behavioral|dramsys`、`backend_granularity=channel|stack` 为每个逻辑实例
  创建唯一 backend 与 endpoint；
- `HBMRuntime` 统一持有对象生命周期，并可将全部 endpoint 绑定到 adapter；
- 每笔 trace 记录 command/address、decoded channel/rank/bankgroup/bank/row/column、
  submit/issue/complete/service/status；另导出 aggregate stats 和 peak in-flight。

## 4. 测试结果

- `npusim --hbm-r3-selftest`：18/18；
- 覆盖真实存储 read-after-write、非零且非固定 60ns 时序、同 bank 同/异 row 行为、
  8 笔并发且 peak in-flight > 1、trace/aggregate 一致、runtime builder、adapter/backend
  可插拔、capacity 一致和有限 trace 带宽不超过 resolved 理论上限；
- 已注册为 CTest `hbm_r3_selftest`。

## 5. 评审修正与边界

原实现使用 `b_transport` 和固定 60ns，不能代表 DRAMSys 控制器调度；现已改为真实非阻塞
phase。当前 row/bank trace 记录访问及实测 service time，但不是 controller 最终调度的
原生 row-hit/miss counter；若论文结果要求严格 hit/miss 数，需要继续在 DRAMSys controller
instrumentation 层增加计数。

生产 `WorkerCore` 尚未迁移时，active `distributed_hbm` 会明确拒绝启动，避免继续构造每核
私有 `DCache` 并静默绕过 HBM。解除该门禁必须等 R4 完成 Router/NoC 和所有生产模式迁移。
