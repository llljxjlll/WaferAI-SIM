# NoC 集合通信建模实现计划

## 1. 目标、范围与约束

需求来自 [NoC集合通信建模需求.md](NoC集合通信建模需求.md)。目标是：workload 顶层可直接声明集合通信；每个 core 的 DTE 是 TX/RX 发起单位；兼容现有 REQ/ACK/DATA 单播；完整支持 P2P、Scatter、Gather、Broadcast、AllToAll、AllGather、Reduce、ReduceScatter、AllReduce；可选择 Tier0（多单播）、Tier1（网内 broadcast）、Tier2（网内 broadcast + reduce）。

首版范围：

- Tier0 支持片内以及现有单播路径可达的跨 die group。
- Tier1/Tier2 只支持同一 die 内的 group；跨 die 在启动期报错，禁止静默错误路由或降级。
- group 只含 worker core，成员唯一、允许非连续，不能包含 host。
- ReduceScatter/AllReduce 首版使用固定算法，不提供 `auto`。
- 未启用集合通信时，现有行为和冻结回归值必须不变。

## 2. 架构和组合

```text
全局 workload collective 声明
  → 配置期校验、分配实例号、向 group 内每个 core 展开
  → 每 core 一份 Collective_prim（共享 CollectiveKey）
  → 按 phase/chunk 分解为 DTE TX/RX
  → Tier0 单播 / Tier1 multicast tree / Tier2 reduce tree + DCA
```

| 集合操作 | TX × RX / 固定组合 |
|---|---|
| P2P | UnicastTX × UnicastRX |
| Scatter | ScatterTX × UnicastRX |
| Broadcast | BroadcastTX × UnicastRX |
| Gather | UnicastTX × GatherRX |
| AllToAll | ScatterTX × GatherRX |
| AllGather | BroadcastTX × GatherRX |
| Reduce | UnicastTX × ReduceRX |
| ReduceScatter | Reduce-to-root + Scatter |
| AllReduce | Reduce-to-root + Broadcast |

后续 ring、recursive-halving、halving-doubling 作为新的显式 `algorithm` 增加；首版不临时选择算法。

## 3. 集合实例、调用和完成契约

### 3.1 标识和匹配

```text
CollectiveKey = (group_id, collective_id, epoch)
PacketKey     = (CollectiveKey, phase_id, chunk_id, src_rank, dst_rank)
MatchKey      = (CollectiveKey, phase_id, chunk_id)
```

- `group_id` 由规范化的有序 core ID 列表稳定生成或显式配置。
- `collective_id` 显式指定，或由配置展开器按声明顺序确定性分配。
- `epoch` 区分 loop 和连续调用；`phase_id/chunk_id` 区分多阶段算法和流水 chunk。
- `tag_id` 继续兼容现有 Send/Recv 和 router 输出锁，但不能单独作为集合匹配键。
- 16-bit wire tag 明确分区：普通 Send/Recv 使用 `0x0000..0x7fff`，collective 使用 `0x8000..0xfffe`，`0xffff` 保留；含 collective 的 workload 若普通 tag 越界则启动期失败。
- 同一 CollectiveKey 的所有 rank 必须具有相同 `op/group/root/count/dtype/reduce_op/algorithm`。

缺 rank、重复 rank、root 不在 group、属性不一致、ID 冲突、非法 source/phase/chunk 或重复 packet 必须确定性报错，不能无限等待。

### 3.2 顶层声明和每 core 展开

workload 增加全局 `collectives` 声明，用户只声明一次。`config_helper_core.cpp` 在配置期规范化 group、分配 ID，并为每个参与 core 生成本地 `Collective_prim`，填入 `self_rank`、角色和相同 CollectiveKey，保证所有参与 core 在同一逻辑依赖点进入集合操作。

保留每 core 直接写 `Collective_prim` 的调试入口，但必须显式给出完整 CollectiveKey；生产 workload 使用全局声明。

### 3.3 完成和并发

- 本地完成：该 rank 的所有 phase/chunk 完成，RX 数据到齐并完成 DTE endpoint service。
- 全局完成：所有 rank 本地完成；顶层 collective 对后继原语表现为 group barrier。
- 首版禁止同一 `(group_id, collective_id)` 的多个 epoch 重叠；下一 epoch 在上一 epoch 网络包排空后开始。
- 不同 CollectiveKey 可以并发，但 completion、reorder、reduce 和 tracker 状态必须隔离。
- collective 可与普通单播并发并共享真实 NoC 资源，但不能共享集合匹配状态。两类流使用隔离的 wire tag 命名空间，并以 mixed-traffic 回归验证。
- 增加可配置 watchdog；超时打印 CollectiveKey、缺失 bitmap 和 phase/chunk。watchdog 不替代正确状态机。

## 4. 描述符和数据布局

`Collective_prim` 至少包含 `op/algorithm/root/group/self_rank`、CollectiveKey、`count/dtype/reduce_op`、`src_addr/dst_addr/chunk_bits/stride_bits`、`gather_reorder_depth`、TX/RX token 和完成策略。

数据量统一为 bit，cycle 换算向上取整。Scatter/ReduceScatter 遇到 `count % N != 0` 时采用 quotient/remainder：低 rank 多一个元素，并携带每 rank 的真实 count/offset，禁止丢尾部。

配置原语使用 `sc_bv<128>` 分段序列化；变长 group 采用固定 header + core ID 段，header 携 group 长度和段数。反序列化校验截断、重复成员、额外段、地址/stride 对齐和字段位宽溢出。

## 5. 唯一计时来源

### Cycle-accurate backend

数据实际生成 packet 并经过 router。注入、hop、buffer、仲裁、背压和拥塞只由实际事件计时；DTE 只增加 endpoint 读取/写入、reorder commit、alignment 等服务。不得再等待含 `P/bw` 或 `T_Network` 的闭式 collective 时延。

### Behavioral backend

不逐包模拟 contention，使用 `coll_latency` 闭式模型；代表包只作完成通知，不能再次按完整 packet 数等待。`coll_latency` 仅服务 behavioral backend 和 Python oracle，不是 cycle 路径的附加延时。

公式边界：

- Unicast：`ceil(P/bw) + T_Network`。
- Tier0 Broadcast：`ceil((N-1)·m/bw) + T_Network`。
- Tier0 Scatter：按需求基线为 `ceil(m/bw) + T_Network`；oracle 另记真实非本地 payload。
- Tier1 Broadcast：一次源注入，完成取 multicast tree 最慢分支。
- Gather：到齐是完成条件，不额外增加 `(N-1)` 个固定 cycle；只计 1 cycle 拼接或配置的 endpoint service。
- Tier0 Reduce：到齐 + 1 cycle 对齐；root ALU 由显式计算原语计费。
- Tier2 DCA：属于 NoC 服务时间，按实际 reduce packet/树级计费，不再生成 root ALU 原语。

需求公式 `max(comp,p/128)+54` 拆为有单位参数：`dca_pipeline_cycles=54`、`dca_bits_per_cycle=128`、`dca_compute_cycles(dtype,op,elements)`；其范围是单次 DCA service，而不是整个 collective。

## 6. Gather RX reorder 模型

每个 `(CollectiveKey, phase_id)` 维护 expected source bitmap、每源 expected chunk 数、以 `(src_rank,chunk_id)` 为 key 的 slot、目标 offset/有效长度/到达/提交状态、有限 `gather_reorder_depth`、commit/write 端口速率和完成事件。

包可乱序进入；重复或未知 source/chunk、越界 offset 报错；buffer 满时向上游背压；slot 按目标 offset 独立提交；所有 expected slot 提交后再计拼接服务。depth、写端口速率和无限深理想模式都进入配置和 oracle。Reduce RX 复用 MatchKey/expected bitmap，但 operand buffer 与 Gather slot 分开计容量。

## 7. 加速配置和能力校验

保留 `noc.collective.tier=0|1|2`，并展开为：

```text
noc.collective.enabled = false
noc.collective.broadcast_backend = unicast | multicast
noc.collective.reduce_backend = endpoint | in_network
```

Tier0=`unicast+endpoint`，Tier1=`multicast+endpoint`，Tier2=`multicast+in_network`。显式 backend 优先于 tier，冲突配置启动期失败。

behavioral backend 对三级均提供闭式估算并明确标记“不建模拥塞”；cycle backend 的 Tier0 走真实单播，Tier1/Tier2 仅同 die。不得静默降级；只有显式 `fallback=unicast/endpoint` 才允许回退，并记录实际 backend。

## 8. 分版本实施

每版交付代码、配置、自动化测试、Python oracle、开发日志和已知限制，并执行 DTE、NoC congestion、D2D link 冻结回归。

### V0 — 冻结公共契约

- 冻结 CollectiveKey/PacketKey、调用/完成、算法、分片、非法输入和计时边界。
- 新增 `coll_types.h` 和只服务 behavioral/oracle 的 `coll_latency.h`。
- 新增变长 primitive 序列化纯函数、位宽检查和 `--coll-v0-selftest`。
- 固定 ReduceScatter 和 AllReduce 基线算法。

### V1 — 全局声明、展开和 Tier0 数据搬运

- 实现全局声明向每 core 展开、实例一致性校验、epoch、barrier 和 watchdog。
- DTE 增加 Unicast/Scatter/Broadcast TX 与 Unicast/Gather RX 描述符。
- P2P/Scatter/Gather/Broadcast/AllGather/AllToAll 分解为现有单播 flow。
- cycle 走真实 packet，behavioral 走闭式模型，验证无双重计费。
- 支持非连续 group、连续 epoch 和不同 group 并发。

### V2 — Gather 有限 reorder 和背压

- 状态：已完成（2026-07-26）；默认深度 0 保持 V1 ideal/unbounded 路径。
- 实现 slot、容量、commit 端口和背压状态机。
- trace 输出 occupancy、stall、slot commit 和缺失 bitmap。
- 测试跨源/跨 chunk 乱序、buffer 满、重复包、非法 offset 和最终 drain。
- V2 验收区分两层覆盖：注入式状态机测试覆盖乱序/FULL/stall，端到端 runner 在 V1 串行 source phase 下覆盖有序 Accept→Commit→Drain；在并行或多 chunk 展开落地前不得宣称端到端背压时序已被触发。

### V3 — Tier0 归约族

- 状态：已完成（2026-07-26）；采用固定 root baseline，endpoint compute 吞吐取 root core 的 `vec_x × vec_cnt`。
- 实现 Reduce RX expected bitmap、alignment 和 endpoint completion。
- root reduction 使用显式计算原语，通信层不重复计 ALU。
- 实现固定 ReduceScatter/AllReduce 算法；不支持的 dtype/op 启动期拒绝。
- oracle 分别核对通信、计算、流量和 phase barrier。

### V4 — Tier1 multicast 协议和 router 状态机

- 状态：已完成（2026-07-26）；配置建树、生产 Router 原子 fork、背压恢复、local delivery、单份注入和端到端验收均已接通。
- 定义 `CollectiveTreeTable`：`(tree_id,router_id,ingress)->output_bitmap`，包括容量、编程、生命周期、冲突、无环和 group 覆盖校验。
- 新增独立 `COLL_DATA` wire layout，冻结 tree ID、CollectiveKey/phase/chunk、source、尾包字段，不破坏现有 256-bit tagged union。
- fork 首版采用原子复制：所有目标输出有空间且仲裁成功时才同时提交；若改为非原子必须增加 pending-output bitmap。
- 每分支独立首包上锁、尾包解锁和 refcount；锁 key 必须隔离集合实例，不能只依赖 tag。
- 定义 local delivery、多输出仲裁公平性和单分支背压行为。
- BroadcastTX 只注入一份 flow，每个目标恰好收到一次。

### V5 — Tier2 in-network reduce

- 状态：已完成（2026-07-26）；operand wire、有限 Match Buffer、单服务端 DCA、逐级聚合、结果重注入及 Tier2 三类归约端到端均已接通。
- 冻结 operand layout：dtype、有效元素数、chunk、source/child、reduce op。
- MatchKey 为 `(CollectiveKey,phase_id,chunk_id)`，tree entry 给出 expected child bitmap。
- 定义 Operand/Hdr Buffer 容量、重复/缺失 operand、DCA pipeline/吞吐、结果重注入和背压；header/operand 满均保留输入并可重试，释放容量后继续，不以异常代替流控。
- 第一阶段至少 bit-accurate 支持整数 SUM/MAX；FP 未实现真实值时标记 timing-only，禁止数值正确性断言。
- timing-only 验证匹配、到齐、流量和时延；value 模式同时验证最终 operand。
- Tier2 不生成 Tier0 root compute。

### V6 — 集成和扩展

- 状态：已完成（2026-07-27）；多 tree/Tier2 并发、普通 DATA 混合流量、逐链路统计、最终 barrier 生命周期释放和全状态 drain 验收均已接通。
- 已验证普通单播、多个 collective、多个 tree 的共享链路竞争；`COLL_LINK` 按 tree 记录 fork flit/stall，`COLL_SHARED` 证明普通 DATA 与 collective flit 使用同一 Router 输出。
- 最终 barrier 的最后一个 rank 释放本实例 multicast/reduce tree；tree-ID 碰撞在启动期拒绝，不能让一个实例的释放破坏另一个实例。
- 生命周期负例已直接验收：非 BARRIER 不得携带 release ID、同一 barrier 所有 rank 的 release ID 必须一致、未知 tree 不得静默释放。
- 完成后统一检查 router lock、tree/reduce registry、barrier、operand/reorder buffer、endpoint raw queue 和 DTE token 全部清零。
- `SEND_DONE` 为 one-shot，多个 terminal core 不能由先完成者重复 DONE 触发提前停机；empty-worklist core 仍完成 CONFIG ACK/START 同步。
- 跨 die hierarchical collective 仍需独立算法与链路协议设计；完成前继续拒绝 Tier1/Tier2 跨 die。
- ring/recursive-halving 仍作为后续显式算法，不在 V6 中静默替换已冻结的 DIRECT/root-tree 算法。

## 9. 关键文件

- 契约/DTE：`llm/include/dte/coll_types.h`、`coll_latency.h`、`dte_async.h`，`llm/src/dte/dte_async.cpp`
- primitive/展开：`llm/include/prims/norm_prims.h`、`llm/src/prims/norm_prims/collective_prim.cpp`、`llm/src/monitor/config_helper_core.cpp`
- 执行：`llm/src/workercore/workercore.cpp`、`llm/src/workercore/logic.cpp`
- 配置：`llm/include/defs/spec.h`、`llm/src/defs/spec.cpp`、`llm/src/utils/config_utils.cpp`
- router/wire：`llm/src/router/router.cpp`、`llm/include/router/router.h`、`llm/include/common/msg.h`、`llm/src/utils/msg_utils.cpp`
- 测试：`llm/test/noc_collective/`

复用现有 PrimFactory、DTE token/fence、Send/Recv、packet 计算、D2D 路由和 trace；复用不等于沿用 tag-only 集合匹配假设。

## 10. 验证与验收

基础门：`cmake --build build --parallel 2`；self-test、Python oracle、端到端 trace 三方一致；behavioral/cycle 分别对齐自己的计时契约；未启用 collective 时现有冻结结果精确不变。

必须覆盖：

- N=1/N=2、空/重复/非连续 group、root 不在 group。
- 零长度、payload 跨 128 bit 边界、count 不能整除 N。
- 相同 ID 的连续 epoch、不同 group 并发、与普通 Send/Recv 并发。
- Gather 乱序、reorder 满、commit 背压。
- multicast 单分支长期背压、多树竞争、local + 多方向 fork。
- Reduce 的 dtype/op、尾包有效元素、重复/缺失 operand。
- backend/topology 不兼容时失败；Tier1/Tier2 跨 die 拒绝。

测试不能只比较最终 cycle，还要检查源注入 packet/flit、每目标恰好一次、丢包/重复、每链路 packet-hop、expected/received/committed bitmap、buffer 峰值/stall、router lock/refcount、tree state、DTE token 最终清零、phase/chunk 完成、behavioral 无二次 bulk wait、Tier2 无重复 root compute。

加速验收比较资源/流量模型和 oracle，不要求所有小消息或拥塞场景下 Tier1/Tier2 都严格更快。

## 11. 已知限制

- 256-bit Msg 空间紧张，V4/V5 必须先冻结 `COLL_DATA` tagged layout 和溢出检查。
- behavioral Tier1/Tier2 是不含真实拥塞的闭式估算，日志必须标识 backend。
- 首版 Tier1/Tier2 仅同 die；跨 die hierarchical collective 后续实现。
- 首版归约族使用固定基线算法。
- FP reduce 在定义舍入、NaN、溢出和结合顺序前不能宣称 bit-accurate。
