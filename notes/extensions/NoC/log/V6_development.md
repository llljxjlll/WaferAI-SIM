# NoC 集合通信 V6 开发日志

日期：2026-07-27
状态：已完成（集成、并发、生命周期与全状态 drain 验收完成）

## 本阶段交付

- 生产 multicast/reduce registry 增加按 `tree_id` 生命周期释放和最终计数接口；配置期继续有限容量编程，运行期由最终 barrier 的最后一个离开 rank 释放本 collective 实例。
- `Collective_prim` 的 CONFIG wire 增加 `release_tree_id`，仅 BARRIER 可携带；同一 barrier 所有 rank 必须一致，否则确定性报错。
- 配置展开准确标记 Tier1 Broadcast、Tier2 Reduce/ReduceScatter/AllReduce 的最终 barrier；中间 phase barrier 不提前释放 tree。
- 配置期维护 accelerated tree-ID 集合，16-bit 稳定 hash 碰撞启动期拒绝，禁止 registry 别名和错误生命周期共享。
- 新增 `COLL_LINK`：按 `(tree_id, router, output)` 统计实际 fork commit flit 与 CanCommit 背压尝试。
- 新增 `COLL_SHARED`：按 `(router, output)` 分别统计普通 DATA 和 collective wire 的实际发送 flit，排除 CONFIG/WEIGHT/控制包。
- 新增 `COLL_DRAIN`：统一报告 multicast tree、reduce node、barrier、Gather reorder、Tier0 Reduce RX、endpoint raw queue 和 DTE outstanding token；结合 Router residual/credit 构成完成不变量。
- 新增 `--coll-v6-selftest` 和 `run_test_coll_v6.py`。

## 集成中发现并修复的问题

### 多 terminal 提前停机

通用 `prim_refill` 会重排 `SEND_DONE`。单 terminal 用例在第一次 DONE 后立即停机，掩盖了该问题；多 collective 场景中先完成的 root 可重复发送 DONE，错误满足全局计数并在另一棵 tree 未排空时停止。修复后 `SEND_DONE` 与 collective primitive 一样是 one-shot。

### empty-worklist core 配置越界与 rendezvous

稀疏 collective group 可让显式 core 没有任何 worklist。原配置构造无条件读取 `worklist[0].recv_tag`，ASan 定位到 `BuildConfigMessages()` 越界；改为使用 core ID 作为空闲 core 的 RECV_WEIGHT 默认 tag。随后 watchdog 进一步发现空 worklist 的配置流没有最终 `refill/is_end`，导致该 core 不回 CONFIG ACK、全局 START 永不下发；现由最后一个 Set_batch segment 承担配置终点。空闲 core 不执行伪 workload，但完整参与配置同步。

## V6 验收

- V6 integration selftest：11/11。除正常释放外，直接覆盖非 BARRIER
  携带 `release_tree_id`、同一 barrier 各 rank 的 release ID 不一致，以及
  最终 barrier 释放未知 tree 三条拒绝路径。
- V6 production runner：6/6。
  - 两棵 Tier1 tree 在 router1/E 共享同一有向链路，各 64 flit，并观察到真实 fork stall；两个目标均精确一次收到。
  - 两棵 Tier2 Reduce tree 并发：4 个 rank operand stream、2 个 bit-accurate root result、2 次独立 tree release。
  - Tier1 Broadcast 与普通 DATA 在 router1/E 同时有实际流量：normal 16 flit、collective 512 flit。
  - 同一 group/collective_id 的连续 epoch 完成两次独立释放，最终 registry 为零。
  - 确定性构造的 16-bit tree-ID hash collision 在启动期拒绝。
- 所有成功场景均满足：`router_residual=0`、credit balanced、`COLL_DRAIN` 七类状态全零。
- 版本回归：V0 32/32、V1 24/24、V2 8/8、V3 19/19、V4/V5 9/9。
- 冻结回归：NoC 4/4，周期保持 14781/29109、14833/45441；D2D V0 308/308、D2D V5 23/23。

## 评审闭环

- 生命周期安全点已经固化为直接测试：只有最终 BARRIER 可携带 release ID；
  所有 rank 必须一致；只有最后离开的 rank 执行释放；未知 tree 必须报错。
- 连续 epoch 仍由 production runner 验证为两次独立释放，且最终 registry
  与 `COLL_DRAIN` 七类状态全部归零。
- 评审确认 `SEND_DONE` one-shot 和 empty-worklist rendezvous 修复均为真实
  并发集成缺陷；V6 保留相应多 terminal 与稀疏 group 端到端场景，防止回退。

## 保留边界

- Tier1/Tier2 仍只支持同 die；hierarchical cross-die collective 未设计完成，继续启动期拒绝。
- V6 不引入隐式算法选择；ring、recursive-halving 等必须在后续以显式 algorithm 和独立 oracle/验收加入。
- FP32 reduce 语义仍未冻结，继续拒绝。

## 关键文件

- `llm/include/dte/coll_multicast.h`
- `llm/src/dte/coll_multicast.cpp`
- `llm/include/dte/coll_innetwork_reduce.h`
- `llm/src/dte/coll_innetwork_reduce.cpp`
- `llm/src/dte/coll_runtime.cpp`
- `llm/src/prims/norm_prims/collective_prim.cpp`
- `llm/src/monitor/config_helper_core.cpp`
- `llm/src/router/router.cpp`
- `llm/src/workercore/workercore.cpp`
- `llm/unittest/npusim.cpp`
- `llm/src/dte/coll_v6_selftest.cpp`
- `llm/test/noc_collective/run_test_coll_v6.py`
