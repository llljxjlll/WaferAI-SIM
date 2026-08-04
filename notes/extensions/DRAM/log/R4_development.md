# 分布式 HBM R4 开发记录

日期：2026-08-04
状态：多 die NUMA、真实 Router/NoC/C2C 与生产访存迁移已完成

## 1. 阶段范围

R4 把 R0-R3 已验证的地址映射、wire、endpoint、behavioral/DRAMSys backend 接到生产
Router fabric，完成本地/远端 HBM、独立 request/response VC、WorkerCore topology switch，
并迁移 SIM_DATAFLOW、PD、PDS、GPU 共用的 DRAM helper。未启用 `distributed_hbm` 时
继续走原 legacy per-core DCache。

## 2. MEM wire 与乱序重组

- REQ/WDATA 走 Router data VC，RDATA/RESP 走 ctrl VC，切断请求和响应之间的单 VC
  环路依赖；
- 256-bit 数据 flit 使用 14-byte payload，其余位携带 source/home/stack/channel，任意
  WDATA/RDATA 都能独立路由；
- `InspectMemWireFlit` 提供 Router/D2D 轻量识别，普通 `Msg` codec 不再误解 MEM wire；
- endpoint 和核侧按 `(source_core, txid)` 组包，再按 kind/sequence canonicalize。
  R4 测试实际发现远端响应可能先到 seq=4，修复后支持合法乱序而不提前完成；
- transaction 超过 65535 byte 时 adapter 自动拆分，地址 stripe/channel 边界仍逐段解码。

## 3. Router、D2D 与 endpoint bridge

- `HBMNetwork` 在源 Router 的有限 CENTER input queue 注入请求，在 MEM attachment
  落地，交给多 worker endpoint agent，再从 attachment Router 注入响应；
- Router 的 MEM 下一跳由 home stack/channel attachment 或 source core 决定；跨 die
  以 source/txid 为稳定 flow key，在每个 die 选择一致出口，并在动态策略的逻辑尾包释放 pin；
- Router 输出统计 NoC/C2C hop、双向流量、注入 stall 和与普通 DATA 的共享队列争用；
- functional D2D 直接透传 MEM；behavioral D2D 应用固定延迟和解析式 flit service；
  bounded SAF 为 MEM 设置独立有限 SAF FIFO，但继续共享 port/link token、inflight、RX、
  credit 和 backpressure；
- 修复 MEM 批量注入使用 zero-time event 时，同一 Router 在同一时间戳多次执行并覆盖
  `sc_signal` 的问题；注入唤醒现在至少延后一拍，保证每条物理信道每周期至多一个 flit。

## 4. 生产 topology switch

- `Monitor` 按 `RouterMonitor → HBMRuntime/HBMNetwork → WorkerCore` 顺序 elaboration；
- distributed 模式不实例化或绑定每核私有 DCache，每核 `CoreMemAdapter` 绑定唯一共享
  HBMNetwork；legacy 模式保持旧构造和 socket binding；
- `WorkerCoreExecutor` 在 distributed 模式也不再构造 `NB_DcacheIF/DcacheCore`、
  `L1Cache/GPUNB_dcacheIF`；`Monitor` 不再创建带独立 DRAMSys 的
  `L1L2CacheSystem`，消除未绑定 TLM socket 和 GPU 旧 DRAM 旁路；
- GPU 的 `GpuPosLocator` 仅保存地址/对象位置元数据，在 distributed 模式继续保留，
  但不承担数据访问；
- `TaskCoreContext` 携带 HBM adapter；NPU/PD/PDS 共用的 load/spill helper 和 GPU
  read/write helper 在 distributed 模式统一走 adapter，无论 `SPEC_USE_BEHA_DRAM` 取值
  都不能直接 `wait()` 绕过网络；
- 地址空间上限取显式 home range 末端；global-interleave 无 range 时取所有 stack 的显式
  capacity 总和，不从带宽反推容量。

## 5. R4 自测

`npusim --hbm-r4-selftest`：9/9。

- 3×3 compute-die mesh，81 个真实 Router，24 条有向 `D2DLinkUnit`，die 0/8 挂 HBM；
- 内部且无本地 HBM 的 die 4 读取 die 0 home，同一地址与本地 core 观察相同 payload；
- die 0 到 die 8 的多跳写，在 home die 本地读取一致；
- 远端访问延迟严格大于本地访问，请求和响应两方向均产生 C2C hop；
- 内部 die 的 `g_die_mesh_pkts>0`，证明不是只在 die 边界转发；
- 两个 requester 并发访问共享 HBM，同时注入合法 collective data wire，观测到共享 Router 输出争用；
- 结束时 waiter、partial assembly、endpoint queue/active worker residual 全为 0。

## 6. 回归结果

- CTest `hbm_r0_selftest`～`hbm_r4_selftest`：5/5；
- `--d2d-v0-selftest`：308/308；
- `--d2d-link-selftest`：37/37；
- 完整构建：`cmake --build build -j2` 通过。

DRAMSys controller 内部原生 row-hit/miss counter 仍是独立的 instrumentation 增强项，
不影响 R4 的 NUMA/NoC/TLM 数据与时序路径完成状态。
