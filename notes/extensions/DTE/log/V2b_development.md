# DTE V2b 开发记录

> 版本：V2b（流式端到端与瓶颈迁移）
> 完成日期：2026-07-24
> 状态：已完成并通过验收
> 规划文档：`../DTE建模计划.md`
> 验收清单：`../DTE分版本开发与验收清单.md`

## 1. 目标与兼容边界

V1 将 source DTE、NoC/D2D、destination DTE 三段整块串行，适合作为保守上界；V2b
在不改变 V1/V2a 冻结时序的前提下，让三段数据阶段流式重叠。

新增 simulation 配置：

```json
{
  "dte": {
    "use_beha_dte": true,
    "streaming": true
  }
}
```

`streaming` 默认 `false`。开启时要求：

- `use_beha_dte=true`；
- `SIM_DATAFLOW`；
- `send_recv_parallel=false`。

因此 V1/V2a 配置不写该字段时，仍执行原 store-and-forward/parallel SEND 路径。

## 2. 模型选择

采用“显式首单元事件 + 流级闭式尾部”模型，而不是为每个真实 bit 或 flit 创建 SystemC
事件。该选择保留真实 DTE channel/bus 和真实网络拥塞，同时控制事件数量。

### 2.1 Source 首单元

source context 正常经过 pending、launch、bus_wait。进入 `TRANSMITTING` 后：

- physical NoC：每个模拟 packet 按 `packet_scale` 恢复其累计逻辑 bit，等待
  `transmit_start + ceil(cumulative_bits/width)×CYCLE`；
- behavioral NoC：代表 flow 在第一个 DTE-width word ready 后放行；
- source DTE 后续仍在后台占用真实 shared bus，SEND 原语结束前完成并释放 context。

DTEUnit 新增 `transmit_started` 和 `scheduled_completion_time`，并保持完成事件的“先查状态、
再 wait”规则，避免错过 `SC_ZERO_TIME` 通知。

### 2.2 Network 首包与末包

- physical backend：首 DATA/末 DATA 使用真实 Router 到达时刻；
- behavioral 同 die：代表消息到达后用 `roofline_packets_` 表示 bulk tail；
- behavioral 跨 die：第一条 D2D link 只等待 `first_packet_service_cycles` 后交付代表消息，
  把 `bulk_service_cycles-first_packet_service_cycles` 作为 tail cycles 携带到目的端；
- D2D 统计仍记录完整 bulk service，未改变既有 ledger 定义。

### 2.3 Destination 首写与最终完成

每个 source 的首 DATA 到达时，目的 core 从 REQUEST round queue 读取完整 payload，并 issue
一个独立 `REMOTE_TO_SPM` context。多 stripe 仍只建立一个 destination context。

最终闭式公式（绝对 ns）为：

```text
first_network_latency = destination_first - source_first
source_tail_at_destination = max(source_first, source_done)
                           + first_network_latency
final_write = max(destination_dte_done,
                  network_last + CYCLE,
                  source_tail_at_destination + CYCLE)
```

最后的 `CYCLE` 是一个完整 destination drain interval。实现位于
`llm/include/dte/dte_streaming.h`；Python oracle 使用同一公式。

## 3. Wire 元数据

DATA 的业务 payload 在当前模拟器中不被计算消费，因此 V2b 在 streaming DATA 中复用
`data_[127:0]`：

| 位 | 内容 | 单位 |
|---|---|---|
| 47:0 | source first-ready absolute time | ns |
| 95:48 | source scheduled completion absolute time | ns |
| 127:96 | behavioral D2D remaining bulk tail | cycle |

两个时间戳均检查 48-bit 容量；tail 检查 32-bit 容量。streaming 关闭时不解释这些字段，
legacy DATA round-trip 与时序不变。

最初实现曾使用绝对 cycle；校准发现事件可落在奇数 ns，而 `CYCLE=2ns`，向下取整会把完整
2ns drain 缩成 1ns，因此最终改为 48-bit absolute ns。这使 source-slow trace 的
`network E=913ns`、`destination E=915ns` 严格相差一个完整 cycle。

## 4. Trace 契约

每个 `(source,destination,tag)` flow 新增三组 B/E：

- `DTE_stream_source_fill`：source issue 到首 DATA ready；
- `DTE_stream_network`：首 DATA 注入到网络末尾；
- `DTE_stream_destination`：目的首 DATA 到最终写入。

大 payload 必须出现 destination/network overlap；小 payload 允许 fill/launch 主导而没有
可观测 bulk overlap，但事件顺序仍必须单调。

## 5. 瓶颈迁移

### 5.1 Behavioral 三类明确瓶颈

16,384-bit 单 flow 的解析尾部候选：

| 配置 | source 候选 | network 候选 | destination 候选 | 主导 |
|---|---:|---:|---:|---|
| source slow | 911 ns | 467 ns | 415 ns | source |
| network slow | 407 ns | 467 ns | 415 ns | network |
| destination slow | 407 ns | 467 ns | 919 ns | destination |

trace 中的 destination E 分别精确等于 911、467、919 ns，与 C++/Python oracle 一致。

### 5.2 位宽扫描

source/destination DTE 同时扫描：

| DTE width (bit) | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|---:|---:|
| finish (ns) | 935 | 679 | 551 | 487 | 483 | 483 | 483 |

1024 bit 后完成时间进入 483ns 平台，证明瓶颈从 DTE 迁移到网络；继续增加 DTE 位宽不再
产生虚假收益。

## 6. Store-and-forward 对比

| backend/配置 | store (ns) | streaming (ns) | 变化 |
|---|---:|---:|---:|
| physical source slow | 1067 | 931 | -136 |
| physical network slow | 563 | 545 | -18 |
| physical destination slow | 1067 | 935 | -132 |
| physical equal | 803 | 553 | -250 |
| behavioral source slow | 1005 | 927 | -78 |
| behavioral network slow | 501 | 483 | -18 |
| behavioral destination slow | 1005 | 935 | -70 |
| behavioral equal | 741 | 551 | -190 |
| behavioral D2D stripe=4 | 632 | 604 | -28 |

所有场景 streaming 均不大于相同配置的 store-and-forward。

## 7. Payload 与 backend 覆盖

- 8-bit 小 payload：256ns，主要由 fill/launch 决定；
- 16,384-bit 大 payload：用于三类瓶颈与位宽扫描；
- 16,416-bit 非整除 payload：570ns；256-bit DTE transmit 为
  `ceil(16416/256)×2=130ns`；
- physical 与 behavioral 的 endpoint descriptor payload 均为精确 16,384 bit；
- behavioral D2D stripe=4 同时覆盖 local 5→7 和 cross-die 8→24；cross flow 的
  network/destination 为 412→586ns 与 454→588ns，存在明确重叠；
- multi-source：目的 core 2 为 source 0/1 分别创建 16,384-bit context，751ns 完成；
- repeated flow：相同 `(source,tag)` 两轮复用时，source xfer id 均为 `[0,1]`，destination
  四个写入 context 为 `[0,1,2,3]`，1179ns 完成且 round queue 正确消费。

## 8. 可选 RECV 并发决策

检查当前 dataflow dispatcher 与真实 workload 后，没有发现同一 core 需要同时执行多个
`RECV_DATA` 原语的证据。现有 `send_para_queue` 只拥有 SEND_REQ/RECV_ACK/SEND_DATA/
SEND_DONE，把 RECV_DATA 直接加入会引入以下未定义所有权：

- 多个 Recv_prim 的 `prim_block` 完成规则；
- DATA buffer 的 tag/source/subflow 分派；
- 每个 destination DTE context 的取消与释放；
- refill 与乱序完成。

因此 V2b 不改 dispatcher，并在启动期拒绝 `streaming + send_recv_parallel`。若未来 workload
证明需求，应作为独立版本完成设计和验收，不作为本版隐式承诺。

## 9. 测试资产

- `run_test_dte_v2b.py`：18 项集成验收；
- `oracle.py`：V0 resource oracle + V2b tail oracle；
- `hardware/v2b_source_slow.json`、`network_slow`、`destination_slow`、`equal`；
- `hardware/v2b_width_{64,128,256,512,1024,2048,4096}.json`；
- `hardware/v2b_cross_die_behavioral.json`；
- `simulation/v2b_{cycle,beha}_{store,streaming}.json` 与 DTE-off/parallel 两个非法组合；
- `workload/v2b_small.json`、`v2b_nondiv.json`、`v2b_invalid_non_dataflow.json`。

## 10. 回归结果

```text
cmake --build build --target npusim -j2                         PASS
./build/npusim --dte-v0-selftest                               64/64
python3 llm/test/dte/run_test_dte_v0.py                         PASS
python3 llm/test/dte/run_test_dte_v1.py                         13/13
python3 llm/test/dte/run_test_dte_v2.py                         15/15
python3 llm/test/dte/run_test_dte_v2b.py                        18/18
./build/npusim --d2d-v0-selftest                               308/308
python3 llm/test/d2d_link/run_test_d2d_v0.py                    67/67 groups
python3 llm/test/d2d_link/run_test_d2d_v4.py                    13/13
python3 llm/test/d2d_link/run_test_d2d_v5.py                    23/23
python3 llm/test/noc_congestion/run_test_noc_congestion.py       4/4
```

NoC frozen baseline 保持 14781/29109ns（no congestion）和 14833/45441ns
（congestion）。V1/V2a 所有精确 ns 均未变化。

## 11. 交付文件

生产代码：

- `llm/include/dte/dte_streaming.h`；
- `llm/include/dte/dte_types.h`；
- `llm/include/common/msg.h`；
- `llm/include/defs/spec.h`；
- `llm/src/dte/dte_unit.cpp`；
- `llm/src/die/d2d_link.cpp`；
- `llm/src/defs/spec.cpp`；
- `llm/src/utils/config_utils.cpp`；
- `llm/src/utils/msg_utils.cpp`；
- `llm/src/workercore/logic.cpp`；
- `llm/test/simulation_config/default_spec.json`。

测试和文档：

- `llm/src/dte/v0_selftest.cpp`；
- `llm/test/dte/oracle.py`；
- `llm/test/dte/run_test_dte_v2b.py`；
- `llm/test/dte/hardware/v2b*.json`；
- `llm/test/dte/simulation/v2b*.json`；
- `llm/test/dte/workload/v2b*.json`；
- `notes/extensions/DTE/DTE建模计划.md`；
- `notes/extensions/DTE/DTE分版本开发与验收清单.md`；
- `notes/extensions/DTE/log/V2b_development.md`。

## 12. 评审闭环（2026-07-24）

评审逐项复核并独立重跑了 V2b 核心矩阵，确认 source 首单元事件、冻结完成时间、按位宽比例
release、三段尾部最大值和多 source 逐源组合均与模型一致，未发现需要修改的生产代码缺陷。
本轮将评审中点查的两类边界固化为自动化证据：

- selftest 锁定非 streaming DATA 的原始 `data_[127:0]` wire payload 与无条件解码结果；
  代码复核确认这些模拟时间字段只由 `SPEC_DTE_STREAMING` consumer gate 使用；
- 集成矩阵分别以独立负例覆盖 streaming 要求 DTE on、要求 dataflow、拒绝 parallel
  dispatcher 三项启动门禁；非 dataflow 模式由 workload 的 `mode=sched_pd` 触发；
- 修订后 DTE selftest 为 64/64，V2b 集成矩阵为 18/18，旧版精确时序回归不变。

评审结论为无阻塞问题，V2b 核心模型与兼容边界不需调整。

## 13. 已知限制与下一阶段

- 当前是 flow-level timing approximation，不模拟逐 flit SPM backpressure；
- streaming 与 parallel dispatcher 互斥；
- 可选多 RECV dispatcher 未实现；
- compute/DMA issue-poll、依赖 token 属于 V3a；
- COMET aggregation/coalescing 属于 V3b；
- 地址映射、更多 DMA 方向和精细 AXI/SPM 端口属于 V4。

V2b 的流水公式、瓶颈迁移、backend、trace、payload 和回归均已满足验收，可以进入下一版
范围评审。
