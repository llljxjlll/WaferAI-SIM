# DTE V3b 开发记录：COMET 启发的在线 aggregation/coalescing

日期：2026-07-24

## 1. 目标与结论

V3b 在 V3a 的 `Dte_async` logical-token 层之上增加确定性的在线 compound descriptor 聚合。兼容的小请求先进入 staged group，flush 时只向 `DTEUnit` 发出一个合并 payload 的 physical transfer；该 transfer 完成后，组内每个 logical token 都可独立 poll/wait，且 physical context 只在最后一个 token 被消费后释放。

本版已完成并验收：SystemC selftest 21/21、WorkerCore 集成矩阵 16/16，全部历史 DTE/D2D 回归通过。

## 2. COMET 论文对应关系

参考文件：`refs/papers/1030062_COMET  Communication and Memory Co-Design for Fine-Grained AI Inference in MCM Accelerators.pdf`。

论文第 7–8 页 §V-C 给出的关键关系是：

- 式 (13) 将 PE 映射到互不重叠的 address block；
- 式 (14) 规定只有映射到同一 address block 的参与者可形成 compound DMA；
- 式 (15) 中 compound group 的 head 支付完整 controller launch，后续请求摊薄 buffer allocation/request generation 开销，组规模为 `g_j`；
- 式 (16)–(17) 根据地址局部性调整路由/传输代价；
- Figure 11 固定总传输数据量，将每核聚合请求数提高到 16，展示 launch 摊薄带来的总时间下降。

仓库没有论文使用的 task DAG、遗传搜索和自动地址 mapper，而且 `Dte_async` 是 endpoint service，不生成真实 NoC/D2D 消息。因此本版明确采用在线近似：workload 提供远端 endpoint、地址和 address block；模型复现同 block 兼容条件、compound launch 和 Figure 11 趋势，但不在 endpoint 层重复加入式 (16)–(17) 的网络倍率。

## 3. 冻结的聚合契约

一条 descriptor 只有同时满足以下条件才可追加到已有 group：

- 方向相同；
- `remote_peer` 相同；
- `address_block` 相同；
- 新 descriptor 的本地 `spm_addr` 紧接 group 的本地尾地址；
- 新 descriptor 的 `remote_addr` 紧接 group 的远端尾地址；
- payload 为 byte 对齐，payload bytes 与 `spm_size` 相等；
- 追加后不超过最大 descriptor 数和最大 payload bytes。

每条 descriptor 必须完整落在配置的 address-block 大小之内，声明的 `address_block` 必须等于按远端地址计算的 block。当前不支持 stride、跨 block descriptor 或非整 byte 聚合。

flush 原因及语义：

- `limit`：达到 descriptor 数或 bytes 上限，立即 issue；
- `incompatible`：同 key 的下一请求不连续，先发旧 group；
- `timeout`：从 group 第一条 issue 起经过按 cycle 向上取整后的配置时间仍未满；
- `dependency`：wait 或 hazard 需要 staged token 成为真实 transfer；
- `fence`：按 issue 顺序 flush 全部 open group；
- `oversize`：单请求超过聚合 bytes 上限，直接作为 standalone physical transfer。

## 4. 状态与生命周期

`DteAsyncTracker` 现在是 SystemC module，内部包含：

- `records_`：每个 logical token 的地址、方向、远端字段、group 与 physical context；
- `open_groups_`/`group_by_key_`：尚未发出的 staged compound group；
- `physical_batches_`：physical xfer 到组员数、剩余未消费 token 数和共享 context 的映射；
- timeout worker：观察最早 deadline，即使 WorkerCore 正执行无关计算也能在准确时刻 flush。

`IssuePhysical()` 只调用一次 `DTEUnit::Issue()`，然后把相同 `xfer_id/context` 绑定给全部组员。组员共享 completion event，但 `WaitAndRelease()` 独立删除 logical token；只有 `remaining_tokens` 降为 0 才调用一次 `DTEUnit::Release()`。

取消规则保持可判定：open group 的尾 token 可安全移除；非尾 staged token 会破坏连续 group，明确拒绝；compound 已 issue 后不允许只取消其中一个成员。V3a 的 pending single-transfer cancel 行为保持不变。

## 5. 配置与 wire

simulation config 新增：

```json
{
  "dte": {
    "aggregation": false,
    "aggregation_max_descriptors": 16,
    "aggregation_max_bytes": 4096,
    "aggregation_timeout_ns": 20,
    "aggregation_address_block_bytes": 65536
  }
}
```

`aggregation=true` 要求 `async=true`；最大 descriptor 数至少为 2，其余值必须大于 0。所有参数在启动期集中校验。

`Dte_async` ISSUE 可增加 `remote_peer`、`remote_addr`、`address_block`。wire 采用兼容的可变长度：

- legacy V3a 不含远端元数据时仍为 2×128-bit segment；
- V3b 元数据存在时增加第 3 个 128-bit segment，字段为 peer 32 bit、remote address 64 bit、block 32 bit；
- 反序列化只接受 2 或 3 个 segment。

开发中曾发现如果无条件写第三 segment，V3a overlap workload 会从冻结的 345 ns 变成 351 ns。改为条件编码后恢复 345 ns，证明 aggregation-off 路径没有承受额外 wire 时序。

## 6. 指标与 trace

`DteAggregationMetrics` 记录：

- logical descriptors；
- physical transfers；
- coalesced descriptors；
- launch savings；
- useful payload bits；
- bus capacity bits 和 `BandwidthUtilizationPpm()`。

有效位利用率定义为：

```text
useful_payload_bits / Σ(ceil(physical_payload_bits / bit_width_bits) × bit_width_bits)
```

新增 trace：

- `DTE_coalesce_collect`：token 被追加到 staged group；
- `DTE_coalesce_flush`：group、members、bits、reason、saved、`cumulative_utilization_ppm`；该值是截至本次 flush 的运行期累计比例，不是单组比例；
- `DTE_coalesce_bind`：每个 logical token 到 compound physical xfer 的映射。

既有 `DTE_async_issue/poll/wait/hazard/fence/cancel` 和 `DTE_pending/launch/bus_wait/transmit` 保留。

## 7. 测试与精确结果

### 7.1 SystemC 与 oracle

- `./build/npusim --dte-v3b-selftest`：21/21；
- `python3 llm/test/dte/oracle.py`：PASS。

selftest 覆盖配置、2/3-segment wire、staged poll、连续聚合、completion fan-out、单次 release、不连续/方向/peer/block 隔离、timeout、aggregation off、utilization、取消规则和固定数据量趋势。

### 7.2 WorkerCore 集成矩阵

`python3 llm/test/dte/run_test_dte_v3b.py`：16/16。

固定 16×64-bit 总数据量、128-bit 共享 bus、20 ns launch：

| 最大组规模 | physical transfers | 首 issue→末 transmit | workload finish | launch savings | utilization |
|---:|---:|---:|---:|---:|---:|
| 1/off | 16 | 352 ns | 548 ns | 0 | 50% |
| 2 | 8 | 178 ns | 374 ns | 8 | 100% |
| 4 | 4 | 102 ns | 298 ns | 12 | 100% |
| 8 | 2 | 70 ns | 266 ns | 14 | 100% |
| 16 | 1 | 66 ns | 262 ns | 15 | 100% |

physical transfer 数、首 issue 到末 transmit 和 utilization 均与独立 Python oracle 精确一致。该单调趋势对应 COMET Figure 11 的 launch amortization，不声称复现论文完整网络、模型 workload 或绝对硬件时间。

其余集成断言：

- 两 token fan-out：一个 physical xfer，两个 wait 都绑定 xfer 0，finish 148 ns；
- 非连续区间：两个 2-member group，flush 原因为 incompatible/fence，finish 172 ns；
- 方向/peer/block 隔离：4 个 physical transfer，finish 220 ns；
- timeout：首 issue 后精确 20 ns flush，并与 Matmul 重叠，finish 341 ns；
- 16-byte 上限：2×64 bit 满组立即发出，finish 136 ns；
- 单个大请求：aggregation on/off 均为 638 ns；
- V3a aggregation-off overlap：保持 345 ns；
- 非法开关、group=1、非法 address block 均明确失败。

### 7.3 历史回归

最终串行重跑结果：

- DTE V0 selftest：64/64；
- DTE V3a selftest：31/31；
- DTE V1/V2a/V2b/V3a：13/13、15/15、18/18、14/14；
- D2D selftest：308/308；
- D2D V4/V5：13/13、23/23。

历史脚本必须串行运行，因为它们共享工作区根目录的 `events.json`；并行执行会互相覆盖 trace，这属于测试运行器资源冲突，不是模型失败。

## 8. 交付文件

核心实现：

- `llm/include/dte/dte_coalescing.h`
- `llm/include/dte/dte_async_types.h`
- `llm/include/dte/dte_async.h`
- `llm/src/dte/dte_async.cpp`
- `llm/include/prims/norm_prims.h`
- `llm/src/prims/norm_prims/dte_async_prim.cpp`
- `llm/include/defs/spec.h`
- `llm/src/defs/spec.cpp`
- `llm/src/utils/config_utils.cpp`
- `llm/src/workercore/workercore.cpp`

验证资产：

- `llm/src/dte/v3b_selftest.cpp`
- `llm/src/dte/v3_selftest.cpp`
- `llm/unittest/npusim.cpp`
- `llm/test/dte/oracle.py`
- `llm/test/dte/run_test_dte_v3b.py`
- `llm/test/dte/hardware/v3b.json`
- `llm/test/dte/simulation/v3b_*.json`
- `llm/test/dte/workload/v3b_*.json`

## 9. 已知边界

- 这是在线 endpoint approximation，不是 COMET 完整离线联合 mapper；
- 不实现式 (16)–(17) 的 route-aware scaling，NoC/D2D 仍由 SEND/RECV 路径建模；
- `Dte_async` 不生成网络数据消息；
- 只支持 byte 对齐、连续、payload 与 SPM range 等长的请求；
- 不支持 compound transfer 的部分 active cancel；
- 这是 V3b 交付时的历史遗留记录；精细 AXI/SPM endpoint 端口和新增方向已由 V4 完成，stride/scatter/broadcast 与自动地址映射按后续范围暂缓。


## 10. 独立评审闭环（2026-07-24）

独立评审逐项复核了 logical→physical fan-out、单次 context release、所有 flush 入口的幂等性、staged hazard、首请求锚定的 timeout、取消边界、2/3-segment wire 和配置门禁，并独立重建及重跑完整矩阵；未发现正确性问题，记录数值与独立结果完全一致。

评审指出一项非功能性歧义：`DTE_coalesce_flush` 原字段名 `utilization_ppm` 容易被理解为当前 group 的比例，而实际值来自 `DteAggregationMetrics`，表示截至该次 flush 的累计有效位利用率。已完成以下闭环：

- trace 字段更名为 `cumulative_utilization_ppm`；
- runner 的解析字段同步更名；
- utilization 测试对 aggregation-on 同时断言最后一次 flush 的累计字段，并对全矩阵断言独立 trace 重算值与 Python oracle 一致；
- 指标计算公式和数值不变，不影响模型时序。
