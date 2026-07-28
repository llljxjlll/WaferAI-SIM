# NoC 集合通信 V1 开发日志

日期：2026-07-25

## 交付

V1 已完成 Tier0 非归约集合通信闭环：

- chip 级 `collectives` 只声明一次，配置期向 group 内每个 core 展开专用 job；
- P2P、Scatter、Gather、Broadcast、AllGather、AllToAll；
- 每个 source phase 下降为现有 REQ/ACK/DATA，phase 末使用 `Collective_prim` 全组 barrier；
- `CollectiveKey=(group_id,collective_id,epoch)` 与 phase-aware 16-bit flow tag；
- flow tag 碰撞、重复 CollectiveKey、非法 group/root、缺 core 启动期拒绝；
- wire tag 结构性分区：普通单播 `0x0000..0x7fff`，collective `0x8000..0xfffe`；普通 tag 进入保留区启动期拒绝；
- cycle backend 使用真实 packet/router/背压，behavioral 使用现有代表包 bulk service，二者不叠加闭式网络时延；
- source 和 destination 继续通过现有 DTE `SPM_TO_REMOTE/REMOTE_TO_SPM` 计费；
- `noc.collective.enabled` 和 `tier` 门控，V1 仅接受 Tier0、DTE enabled 和 sequential dispatcher。

## 无死锁下降顺序

现有 WorkerCore 是顺序 Send/Recv 状态机。V1 采用确定性 source-rank phase：

- P2P、Scatter、Broadcast：一个 phase；
- Gather：每个 source 一个 phase；
- AllGather、AllToAll：每个 source 一个 phase，source 向其他 rank 逐个握手发送；
- 所有 rank 每 phase 都进入 barrier，禁止已完成的 receiver 提前成为下一 phase sender。

这是正确性优先的 Tier0 baseline，不宣称最优并行度。

## Schema 示例

```json
{
  "chips": [{
    "cores": [{"id": 0, "loop": 1, "worklist": []},
              {"id": 1, "loop": 1, "worklist": []}],
    "collectives": [{
      "op": "broadcast", "collective_id": 101,
      "group": [0, 1], "root": 0,
      "count": 1, "chunk_bits": 1024,
      "stride_bits": 1024, "terminal": true
    }]
  }]
}
```

simulation config：

```json
{"noc": {"collective": {"enabled": true, "tier": 0}},
 "dte": {"use_beha_dte": true}}
```

连续调用用多个声明和不同 `epoch` 表达。V1 明确拒绝参与 core 的 `loop != 1`，避免配置重放复用同一个实例键。

## 关键文件

- `llm/include/dte/coll_plan.h`
- `llm/include/dte/coll_runtime.h`
- `llm/src/dte/coll_runtime.cpp`
- `llm/src/dte/coll_v1_selftest.cpp`
- `llm/src/prims/norm_prims/collective_prim.cpp`
- `llm/include/prims/norm_prims.h`
- `llm/include/common/config.h`
- `llm/src/monitor/config_helper_core.cpp`
- `llm/src/workercore/workercore.cpp`
- `llm/include/defs/spec.h`、`llm/src/defs/spec.cpp`、`llm/src/utils/config_utils.cpp`
- `llm/test/noc_collective/run_test_coll_v1.py`
- `llm/test/noc_collective/workload/v1_broadcast.json`
- `llm/test/noc_collective/hardware/v1.json`
- `llm/test/noc_collective/simulation/v1_cycle.json`、`v1_beha.json`

## 验收

- build：PASS。
- V1 planner/wire/barrier self-test：12/12。
- 独立 Python oracle：PASS。
- V1 runner：24/24：六种操作 × cycle/behavioral，连续 epoch × 双 backend，门控、非法 group、重复 key 和保留 tag 等负例。
- cycle 完成值：P2P 220 ns，Scatter/Broadcast 312 ns，Gather 404 ns，AllGather/AllToAll 818 ns。
- behavioral 完成值：P2P 206 ns，Scatter/Broadcast 284 ns，Gather 376 ns，AllGather/AllToAll 734 ns。
- 每场景核对逻辑 flow 数、router residual=0、data/control credit balanced。
- mixed collective+普通 Send/Recv 在 cycle/behavioral 均通过：观测 tag 分别为 50415 与 3，router residual=0、credit balanced；保留区冲突负例通过。
- V0 collective 32/32、DTE V0 64/64、D2D V0 308/308。
- NoC congestion 4/4，冻结值仍为 14781/29109、14833/45441。

## V1 限制

- Reduce/ReduceScatter/AllReduce 留到 V3。
- Gather 当前依赖既有接收端按 offset/来源完成；有限 reorder slot/容量/commit 背压留到 V2。
- V1 source phases 串行，未实现并行 ring 算法。
- `terminal=true` 用于独立 collective workload 的 DONE；嵌入普通计算图时由图的原汇节点负责 DONE。
