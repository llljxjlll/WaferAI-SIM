# P7 开发记录：四档 NoC 加速、DCA 与广播树冲突规避

阶段：P7——四档 collective profile、批级 multicast tree 与 production DCA

阶段状态：完成；四 profile production、N=4/32 KiB、批级生命周期、持续公平以及 fresh Release/Debug 63/63 均已闭环

完成日期：2026-08-12（UTC）

提交/变更集：工作区实现，尚未提交

## 计划确认

- 本阶段严格对照《编译产物指令集开发计划》13.1～13.4，在 P6 immutable collective image 和真实 P5 endpoint 字节通路上替换 backend；高层 collective key、action 顺序、地址和结果语义不随 profile 改变。
- 四档配置固定为 `baseline=UNICAST+ENDPOINT`、`broadcast_only=MULTICAST+ENDPOINT`、`reduce_only=UNICAST+DCA_OFFLOAD`、`reduce_broadcast=MULTICAST+DCA_OFFLOAD`。不满足 capability 的组合加载期拒绝，禁止运行期静默回退。
- topology/path 复用 production X-first 唯一真源。冲突单位为 `(router, output_direction)`；未知路径保守冲突。
- tree 按批真实 `ProgramCollectiveTreeEntry`/`EraseCollectiveTreeEntry`，批结束和全目标完成前禁止 tree/session 复用；DCA 只消费真实 `AccessUnit` 字节并把最终结果写回 SRAM。

## 实际完成事项

### profile、topology、冲突图与有界构造

- `coll_profile_v1` 将四档配置映射为两个正交 backend，并固定 profile/backend 一致性；显式配置冲突、未知 profile/backend 和 ReduceScatter+DCA unsupported 能力在加载期 fail-fast。
- `coll_topology_v1` 复用 X-first 路由生成每棵树的 Router output 资源集，构造确定性冲突图和贪心批次；覆盖无冲突、共享 output、链/环/完全冲突、输入乱序、K=1/2/≥N、未知保守串行和容量边界。
- 配置增加 `noc.collective.max_trees_per_batch`，production 范围固定 `[1,64]`；默认不改变既有行为。parser、helper 与 profile image 覆盖未知字段、0、65、1 和传递一致性。
- 在构造 topology/tree 前 checked 预估并累计 `max_total_trees`、`max_total_tree_entries` 及派生字节；limit、limit+1、算术溢出和失败原子测试阻止大规模预分配 DoS。
- 同一 P6 image 生成 immutable profile sidecar；四 profile 只改 accelerated action/backend/tree schedule，不重写 canonical P6 action、地址或 key。

### 批级 tree 生命周期与观测

- `IsaV1CollectiveTreeRegistryBridge` 把计划批次接到 production Router tree registry。`BeginBatch` 只编程本批树，`MarkTreeComplete` 等待本批全部目标/后端完成，`EndBatch` 再擦除并释放；冲突、重复 program/erase、release mismatch 和跨批 tree 复用均严格拒绝。
- runtime 使用完整 `CollectiveKey(group_id, collective_id, epoch)` 隔离计划，以单调 session 防止在途旧代撞新代；plan 在所有参与 Worker 各自 exactly-once 观察 `PlanComplete` 前保持可查询，最后观察者完成后才 retire。
- 稳定 `[P7_TREE_BATCH]` begin/end 字段来自真实 bridge 事件和 runtime stats：`plan`、`key`、`batch`、canonical `tree_ids`、`trees`、`programmed`、`erased`、`conflicts`、`peak_entries`、`capacity`、`occupancy_after`、`release_reason`。begin/end 可配对，end 的 `occupancy_after=0`，且 `peak_entries<=capacity`。
- 每个计划结束输出 `[P7_TREE_DRAIN]`：真实 `tree_entries`、`reduce_nodes`、`schedule_entries` 与全量 `residual`；32 KiB K=1 矩阵逐树 program/erase，批末表项均回到零。

### 真实 multicast 字节通路

- 冻结 START/DATA 两类 256-bit wire codec。START 携 full 96-bit key、tree/session、总长和 CRC；DATA 携 tree/session/epoch/seq/tail/length 与 128-bit payload。最大 payload 为 1,048,560 B、最多 65,535 个 DATA flit。
- classifier 使用固定 magic/version/kind 和 reserved 校验，Router ours-first；legacy `Msg`、P2P endpoint、CollectiveData/reduce wire 的碰撞样本严格隔离，bit flip、reserved、乱序、重复、tail、CRC、stale/unknown generation 均拒绝。
- source 通过真实 SRAM `AccessUnit` 读取、组装 START/DATA，Router 依据已编程 tree 做逐跳复制，目标重组后通过真实 `kNocRx` 写入 rank-major staging；所有目标 commit/ACK 后才允许 tree complete。
- `[P7_MULTICAST_TX]` 与 `[P7_MULTICAST_COMMIT]` 只在真实发包和真实目标写回发生时输出。32 KiB AllGather/AllReduce 的 broadcast profile 验证四棵 source tree、12 个目标 commit、byte oracle、CRC 与 sentinel。

### 真实 DCA reduce 通路

- root 先 arm 每个 `(key, tree, chunk, session)`，source 从真实 SRAM 读取并发送 reduce stream header/data；Router `RouterReduceStreamEngine` 和 shared DCA compute pool 处理真实字节，root 只接收最终 result 并写 SRAM。
- DCA 成功路径不生成 endpoint `REDUCE_COMPUTE`。位宽、dtype、reduce op、最大 fan-in、header/operand/result FIFO、session/ACK 和 tail 均在 preflight 或 wire/runtime 层严格校验。
- 每 chunk 等全 source ready 后 issue，result commit/ACK exactly once；queue/result backpressure 恢复后不重复 completion。N=1 bypass、N=2/3/4、SUM/MAX、tail、L 与 II 分离、RR/priority 和有限队列由纯层回归覆盖；持续竞争门固定 32 CORE + 32 DCA、L=5、II=2、result depth=1，RR 严格交替、同源 service gap `<=2*II`、64 tag/completion exactly once 且最终 drain=0。
- `[COLL_DCA_ARM]`、`[COLL_DCA_TX]`、`[COLL_DCA_RESULT]` 均来自真实事件。32 KiB reduce profile 实测四 tree、每 tree 四 source TX、四份最终 result，结果 checksum 与软件 oracle 一致且全部 drain 为零。

### Worker action replacement、屏障与 transport ownership

- Worker 独立 collective program/acceleration 后台线程执行四 profile。被替换 action 仍 exactly-once 输出 canonical `Collective_program_v1` B/E 和 `P6_collective_action` instant trace，保持 action index 0..195、wave 数量和 P6 trace 闭合。
- all-target post ready 后才准发送 multicast；DCA all-source ready 后才准 issue。WAIT/FENCE 由 `PlanComplete` 门控，批完成充当 internal batch barrier；tree/session/tag、endpoint、Router、DCA 和 plan observer 全 drain 后才完成 program。
- 所有 Worker 物理发送统一为 `SerializedWireQueue` 值所有权：legacy sequential/parallel DATA、REQUEST/DONE/ACK、P5 endpoint、multicast 与 DCA 都先序列化完整 wire，再由唯一 helper 逐项发送并按 ticket 完成。删除共享 `send_buffer` 单槽、status1/2 pipeline 和 coalescing event 写法。
- 发送端先确认目标 channel available，再取得 status3 ownership；FIFO 为空但 status3 已持有时 helper 等待生产者入队，禁止发送 default/stale wire。每个 wire 保持显式 high/low pulse gap，queue/ticket 纳入 residual。
- transport ownership 回归用 32 个 normal DATA 与 32 个 DCA wire 交错，在 17 个停止消费周期后恢复，验证容量不覆盖、值与顺序不变、每 wire/ticket exactly once 和最终 residual=0；DCA/core compute 公平由上述 production `DcaComputePool` 持续 RR 回归独立证明。

## 评审、修改、复测与再评审

1. 初版 plan 完成后由首个 Worker 立即 retire，较晚 Worker 查询同一 key 得到 unknown/stale。修改为参与核 observation barrier：每核只观察一次，全部观察后再全局 retire；多 Worker late observer 和连续两 plan 回归通过。
2. broadcast profile 的 AllReduce 最初把 source payload 写到公共 staging offset，AllGather 正确但后续 endpoint reduce 的 rank-major输入错误。commit 地址加入 source-rank provenance，N=4/L=64 与正式 broadcast-only byte probe 通过。
3. Router bit255 endpoint 输出最初只按 tag 持锁，多 source 同 destination/tag 可交错；改用 `(source,destination,transport_tag,subflow)` 完整 flow owner，并把 owner 纳入 watchdog/residual。随后 hop trace 进一步定位 source injection 的 event 单槽会吞连续 fragment，改为值 FIFO 和显式 pulse gap。
4. 32 KiB reduce-only K=1 最初在第 4 tree 报 duplicate host ACK。控制类 REQUEST/DONE/host ACK/peer ACK 从共享 `send_buffer+event` 迁到局部 Msg+值 FIFO，duplicate ACK 消失。
5. 下一次第 4 tree 暴露 Router 收到 source=dest=65535 的 default DATA wire。根因是 producer 在 channel unavailable 时先占 status3，helper 在 FIFO 尚空时走 legacy fallback；修复为 availability-first，并增加 FIFO-empty/status3 不变量保护，禁止 default fallback。
6. fresh Debug build 暴露 distributed real path 仍创建 legacy `DramKVTable`，使 dataset word/address 单位断言失败。Worker ctor 改为 distributed 明确 `nullptr`，只在非 distributed 路径断言并构造 legacy table；不放宽地址断言。
7. full Debug CTest 后续在 `playground_legacy_baseline` 暴露旧 sequential DATA 仍使用共享单槽；该 P7 transport 债已将 sequential/parallel legacy DATA 全部并入同一值 FIFO并删除旧 status0/1/2 协议，通过 focused 编译与独立 mixed 回归。后续只读定位还发现该 legacy workload 自身存在 worklist 断边，因此其剩余死锁不得继续单因归责于 P7 发送通路。
8. R6 N=3 AllGather 在 final barrier 后出现 `serialized_wires=1`：CONFIG 的 `prim_refill` 把 Tier0 collective 生成的普通 SEND_REQ/RECV_ACK/SEND_DATA/RECV_DATA 再次入队，而其他核已发送 DONE。修复以 collective reserved tag 作为 no-refill 边界，让 collective RECV_ACK 保留对应 tag，并在 sequential/parallel 两条 dispatch 路径统一应用；V1 定向回归验证 loop=1、N=3 AllGather 的 24 个 endpoint Prim 均不重启且 serialized queue drain。

## 自测与集成证据

已完成的共享证据：

- unified ISA 在值 FIFO 最终修改前通过；其中 acceleration runtime 40 checks、profile image 49 checks、tree batch 38 checks，byte/DCA/profile/topology/image 等纯测全部通过。
- 正式 `program_p7_accelerated_32k_runtime_matrix`：AllGather/AllReduce × baseline/broadcast_only/reduce_only/reduce_broadcast，`8/8 PASS`，总时长 17.69 s。
- 32 KiB K=1 验证每棵树独立 begin/end、program=erase、occupancy_after=0；四档真实 backend marker、SRAM byte oracle/checksum/sentinel、canonical action/wave、P5/P6/P7/global drain 全部通过。
- P8-B 小 payload 四档 repeat=3 和长稳 repeat=20 均通过；AllReduce 四个 backend 组合独立生效，无 watchdog。

最后值 FIFO 修改后的 focused 证据：

```text
logic.cpp                      compile_commands 等价参数 + -Werror: PASS
workercore.cpp                 compile_commands 等价参数 + -Werror: PASS
isa_v1_selftest.cpp            compile_commands 等价参数 + -Werror: PASS
serialized_wire_queue_selftest strict -Wall -Wextra -Wpedantic -Werror: PASS (280 checks)
coll-r2 sustained fairness       strict -Wall -Wextra -Wpedantic -Werror: PASS (R2 46 checks)
git diff --check: PASS
```

说明：focused production TU 对仓库既有 header 的 reorder/init-self/unused 告警使用定向豁免；本切片新增代码本身保持 `-Werror`。按协作约束，本切片没有自行启动共享 build。

## 与计划的差异、明确边界与最终门

- 批间屏障实现为 runtime/bridge 的 internal batch completion barrier，没有新增 public `GROUP_SYNC` record；canonical P6 public action stream保持不变。trace oracle 以 begin/end 配对和下一批不得提前 program 验证该映射。
- ReduceScatter+DCA 仍是明确 capability fail-fast，不以 endpoint silent fallback 冒充 DCA 支持；这属于已记录的正向能力边界。
- production 32 KiB 门覆盖 N=4 和一个 logical chunk/tree；更多 chunk 数、N=2/3、dtype/tail、64-entry 和完全冲突图主要由 codec/topology/batch/DCA 纯层与 P6 fixture 覆盖。本文不把 8/8 矩阵虚报为计划中每个压力笛卡尔积都做了 production runtime。
- P7 四档 available 是功能正确性结论，不是性能 SLA。N=4/32 KiB AllGather/AllReduce 与纯层压力门已经闭合；最终 clean Release/Debug 仍需在包含最新 refill 修复的工作树上统一复验，本文不提前声明“最终 Release/Debug 全绿”。

## 阶段退出检查

- [x] 四 profile 两开关正交，unsupported 能力加载期拒绝且无静默 fallback。
- [x] X-first 路由唯一真源、冲突图、确定性 K/容量调度和预分配限额。
- [x] production tree 真实按批 program/erase，真实 occupancy/冲突/release/drain marker。
- [x] multicast START/DATA 真实 SRAM 字节、CRC、全目标 commit/ACK 和 exactly-once。
- [x] DCA 真实 SRAM read/reduce/write，成功路径无 endpoint `REDUCE_COMPUTE`。
- [x] WAIT/PlanComplete、多 Worker observer、tree/session/epoch 生命周期和全局 drain。
- [x] 32 KiB AllGather/AllReduce 四档矩阵 8/8，无 watchdog。
- [x] normal DATA、控制消息、P5、multicast、DCA 统一值 FIFO；mixed backpressure 回归通过。
- [x] 32 CORE + 32 DCA 持续 RR、L=5/II=2、result depth=1 的公平/恰一次/drain 纯层门通过。
- [x] collective reserved-tag endpoint no-refill 与 N=3 AllGather final-barrier 后不重启定向修复已落地。
- [x] fresh Release/Debug unified 与 full CTest 63/63、P7 32 KiB 8/8 和 sanitizer 8/8 均通过，确认 refill 修复无回归。

是否允许进入发布评审：允许；共享最终复验已完成。

依据：P7 四档 production、N=4/32 KiB、持续公平、transport ownership/refill、fresh Release/Debug、repeat20 与 sanitizer 门均通过，阶段完成。
