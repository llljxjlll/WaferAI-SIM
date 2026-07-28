# NoC 集合通信 V4 开发日志

日期：2026-07-26
状态：已完成（协议、生产 Router datapath 与端到端验收完成）

## 本阶段交付

- 新增独立 256-bit `COLL_DATA` wire，不复用普通 `Msg` 的 DATA tagged union；
- wire 冻结 magic/version、tree ID、CollectiveKey、phase/chunk、src/dst rank、24-bit sequence、尾包长度和 end 标志；
- 序列化拒绝 tree ID 0、非法尾包长度和 sequence 溢出；反序列化校验 magic/version、字段范围和保留位；
- 新增 `CollectiveTreeTable`，key 为 `(tree_id,router_id,ingress)`，支持有限容量、幂等编程、冲突拒绝、缺项拒绝和按 tree 生命周期擦除；
- tree entry 使用五方向 output bitmap，拒绝空输出、越界方向和立即反射回 ingress；CENTER 同时表达 local injection/local delivery；
- 新增 `AtomicMulticastFork` 契约：任一目标不可用时不提交任何副本；
- 每个输出分支使用 `(tree_id,CollectiveKey,phase_id,chunk_id)` 锁定，尾包逐分支释放 refcount，禁止仅按普通 unicast tag 隔离。

## 验收

- build：PASS。
- `--coll-v4-selftest`：13/13。
- 覆盖 wire 往返/非法字段/保留位、树表 fork/冲突/容量/反射/lifecycle，以及原子 backpressure、实例隔离和最终 refcount drain。

## 分支 refcount 说明

- 当前单次 multicast 注入中，每个分支只在 head flit 建锁一次，因此正常执行时 `refs_` 实际表现为 0/1。
- 保留 refcount 而不是 bool 是有意的前向兼容设计，用于后续 multi-injection/共享分支扩展；现阶段不依赖大于 1 的行为，也不把它计为已实现能力。

## 生产接线完成（2026-07-26）

- 配置展开按仓库 `NORTH=+GRID_X` 坐标约定构造确定性 XY tree，启动期验证无环、邻接边和 group coverage；跨 die Tier1/Tier2 继续拒绝。
- `Collective_data_prim` 由 root 只注入一份 `COLL_DATA` 流；非 root 通过专用 RX 校验 tree/CollectiveKey/seq/tail，barrier 保证所有目标完成。
- `RouterUnit::router_execute` 在真实 input/output buffer 上查询 `(tree_id,router_id,ingress)`，检查所有目标容量后原子 fork；任一分支阻塞时不提交任何副本。
- collective branch lock 使用完整实例 key，首包加锁、尾包释放；数据路由采用 round-robin 起点，输出插入可观察的 valid-low 周期，避免连续 flit 电平重采样。
- raw collective RX 队列具有真实 `core_busy` backpressure 及 drain 后 credit 恢复；三核 1024-bit 分叉树同时覆盖单分支持续背压恢复和 local delivery。
- `noc.collective.tier=1` 已放开；scatter 等不同 payload 的 one-to-many 操作仍正确回退 Tier0，不使用复制式 multicast。

## 最终验收

- V4 contract：13/13。
- Tier1 Broadcast：cycle/behavioral 均为单份 TX、2/2 目标各一次 RX、`router_residual=0`、credit balanced。
- V4/V5 production runner：9/9。
- V0–V3 collective、NoC 冻结四场景和 D2D 308/308 均保持通过。

## 关键文件

- `llm/include/dte/coll_multicast.h`
- `llm/src/dte/coll_multicast.cpp`
- `llm/src/dte/coll_v4_selftest.cpp`
- `llm/src/router/router.cpp`
- `llm/src/monitor/config_helper_core.cpp`
- `llm/src/prims/norm_prims/collective_data_prim.cpp`
- `llm/test/noc_collective/run_test_coll_v4_v5.py`
