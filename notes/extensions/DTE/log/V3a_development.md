# DTE V3a 开发记录

> 版本：V3a（通用 compute/DMA 异步重叠）
> 完成日期：2026-07-24
> 状态：已完成并通过验收
> 规划文档：`../DTE建模计划.md`
> 验收清单：`../DTE分版本开发与验收清单.md`

## 1. 目标与边界

V3a 在每核既有 `DTEUnit` 上增加显式 descriptor issue 与依赖 token，使 WorkerCore 可以在
DMA endpoint service 尚未完成时继续执行无依赖计算，并在 `wait` 或 `fence` 处正确同步。

本版新增 simulation 配置：

```json
{
  "dte": {
    "use_beha_dte": true,
    "streaming": false,
    "async": true
  }
}
```

`async` 默认 `false`。V3a 仅支持顺序 dataflow dispatcher，并与 V2b `streaming`、
`send_recv_parallel` 明确互斥。旧配置不写该字段时，V0–V2b 路径和时序不变。

V3a 的 `Dte_async issue` 表示 endpoint DTE descriptor 及其 SPM 访问时序，不自行生成
REQUEST/ACK/DATA 网络协议，也不替代已有 SEND/RECV。需要端到端 NoC/D2D 数据移动时仍使用
SEND/RECV 路径；V3a 研究的是 descriptor、compute 和 endpoint DTE 的调度重叠。

## 2. Dte_async primitive 契约

新增可序列化 primitive `Dte_async`，操作如下：

| op | 语义 | 是否阻塞 | 是否消费 token |
|---|---|---:|---:|
| `issue` | 以逻辑 token 发起一个 DTE descriptor | 仅在地址 hazard 时阻塞 | 否 |
| `wait` | 等待指定 token 完成并释放 context | 是 | 是 |
| `poll` | 查询指定 token 是否完成，结果写入运行态 `poll_complete` 并输出 trace | 否 | 否 |
| `fence` | 按 issue 顺序等待并释放此前所有 outstanding token | 是 | 是 |
| `cancel` | 取消仍在 pending queue 的 descriptor | 否 | 是 |

active/launch/bus/transmitting descriptor 不允许取消，调用会明确报错。仓库当前没有条件分支
primitive，因此 `poll_complete` 暂不驱动控制流；它仍是可供执行器/未来分支原语读取的真实
非阻塞查询结果。

计算依赖采用显式 WAIT primitive：依赖某个 DMA 的计算原语之前插入
`Dte_async(op=wait, token=N)`；无依赖计算不插 WAIT，dispatcher 会直接继续派发。

## 3. Wire 编码

`Dte_async` 固定使用两个 128-bit CONFIG segment：

第一段：

| bit | 内容 |
|---|---|
| 7:0 | primitive type id |
| 10:8 | op |
| 13:11 | direction |
| 45:14 | 32-bit logical token |
| 109:46 | 64-bit payload_bits |

第二段：

| bit | 内容 |
|---|---|
| 63:0 | 64-bit SPM byte address |
| 127:64 | 64-bit SPM byte size |

`issue` 要求 payload、direction、地址和范围齐全；非 issue 操作使用 canonical zero payload/range；
`fence` 还要求 token 为 0。自测覆盖完整 64-bit 边界往返和截断 wire 拒绝。

## 4. Logical token 与 physical xfer_id

每个 `WorkerCoreExecutor` 新增一个 `DteAsyncTracker`：

- logical token 由 workload 指定，用于 wait/poll/fence/cancel；
- physical `xfer_id` 仍由 `DTEUnit` 单调分配；
- tracker 保存 token → `(xfer_id, context*, SPM range, access, issue_sequence)`；
- context 继续由 DTEUnit 的地址稳定 `std::list<std::unique_ptr<...>>` 持有；
- wait/fence/cancel 是唯一消费 token 并释放 context 的正常路径；
- 已完成但未 wait 的 token 仍保留，可 poll，且不会继续占用 active channel；
- 同一 logical token 只有在旧 token 被消费后才能复用，复用时得到新 xfer_id。

实现过程中曾发现 hazard 隐式等待会错误消费旧 token；最终拆分为
`WaitForCompletion()` 和 `WaitAndRelease()`：hazard 只等待完成，显式 wait/fence 才释放，
保证后续 `wait(old_token)` 仍合法。

## 5. 地址依赖与取消

SPM 范围使用 byte 单位和半开区间 `[addr, addr+size)`，启动前校验：

- size > 0；
- 地址加法不溢出；
- `ceil(payload_bits/8) <= spm_size`；
- V3a 方向只允许 `SPM_TO_REMOTE`（SPM read）和 `REMOTE_TO_SPM`（SPM write）。

两个 overlapping read 可并行；只要任一侧为 write，后发 issue 在旧 transfer 完成前阻塞，
覆盖 RAW、WAR、WAW。hazard 等待不消费旧 token。

DTEUnit 新增 pending-only `Cancel(xfer_id)`：从 pending queue 精确移除，记录
`DTE_cancel` trace，再由 tracker 同步释放 context。active cancel 明确拒绝，避免部分传输、
shared-bus 回滚和 completion 语义不清。

## 6. Dispatcher 与结束纪律

`worker_core_execute()` 对 `Dte_async_prim` 使用专用分支，不再借用 `task_logic()` 的
`prim_block` 握手：

- issue/poll/cancel 可立即继续下一条 primitive；
- wait/fence 只在真实依赖未完成时阻塞当前 dispatcher；
- 普通 `Comp_prim` 执行路径不变，因此可以与后台 DTE overlap；
- pipeline/refill 中，上一轮 wait 后相同 token 可安全复用并获得新 xfer_id；
- primitive queue drain 或 `SEND_DONE` 时若仍有 outstanding token，仿真明确失败并要求显式
  wait/fence，防止静默丢失 DMA。

`config.cpp` 同时把原有 `CompBase*` C 风格强转改为 `dynamic_cast` 类型分派：workload 的
`prims` 只接受 compute 或 `Dte_async`，未知/不支持 primitive 清晰报错。

## 7. Trace 契约

除既有 `DTE_pending/launch/bus_wait/transmit/cancel` 外，新增：

- `DTE_async_issue`；
- `DTE_async_wait`；
- `DTE_async_poll complete=0|1`；
- `DTE_async_fence count=N|0`；
- `DTE_async_cancel`；
- `DTE_async_hazard depends_on=M`。

每条事件包含 core、logical token、physical xfer_id 和 outstanding 数量。WorkerCore 仍输出
`Dte_async_prim` 与 `Comp_prim` B/E，runner 可直接证明计算区间和 DTE transmit 区间重叠。

## 8. 验收结果

### 8.1 独立 SystemC selftest

`./build/npusim --dte-v3-selftest`：31/31。

覆盖 primitive wire、四项配置门禁、issue/poll/wait/fence、逻辑/物理 token 映射、指定 token
等待、读读并行、RAW/WAR/WAW、pending cancel、active cancel 拒绝、异常无泄漏、token 复用和
channel=1/2/4。

### 8.2 WorkerCore 集成矩阵

`python3 llm/test/dte/run_test_dte_v3.py`：14/14。

关键结果：

- blocking：DTE transmit 130→194ns，Matmul 202→411ns，总完成 421ns；
- overlap：Matmul 124→333ns，DTE transmit 134→198ns，区间真实重叠，总完成 345ns；
- 两种调度均执行一个 65,536-bit DMA 和一个 209ns Matmul，只有时序不同；
- overlap 比 blocking 缩短 76ns；poll 在 DMA 未完成时返回 `complete=0`；
- selective wait：token 1 于 144ns 消费，token 2 继续 transmit 到 272ns，并与
  152→361ns 的计算重叠，365ns fence 消费剩余 token；
- 四个 32,768-bit descriptor 在 channel=1/2/4 时 max active 为 1/2/4，完成时间
  324/264/264ns；三种配置 shared-bus transmit 总量均为 128ns；
- RAW hazard 区间为 100→182ns，两个 transmit 不重叠；
- pending cancel 正常完成于 260ns；
- 同队列 token 复用 xfer id `[0,1]`，160ns 完成；
- pipeline=2 refill 使用相同 token 获得 xfer id `[0,1]`，218ns 完成；
- 未 fence 的 `SEND_DONE` 和 async 开关关闭时使用 `Dte_async` 均明确失败。

### 8.3 历史回归

```text
cmake --build build --target npusim -j2                         PASS
./build/npusim --dte-v0-selftest                               64/64
python3 llm/test/dte/run_test_dte_v0.py                         PASS
python3 llm/test/dte/run_test_dte_v1.py                         13/13
python3 llm/test/dte/run_test_dte_v2.py                         15/15
python3 llm/test/dte/run_test_dte_v2b.py                        18/18
./build/npusim --d2d-v0-selftest                               308/308
python3 llm/test/d2d_link/run_test_d2d_v4.py                    13/13
python3 llm/test/d2d_link/run_test_d2d_v5.py                    23/23
```

V1/V2a/V2b 所有冻结精确 ns 均未变化。

## 9. 交付文件

生产代码：

- `llm/include/dte/dte_async_types.h`；
- `llm/include/dte/dte_async.h`；
- `llm/include/dte/dte_types.h`；
- `llm/include/dte/dte_unit.h`；
- `llm/include/prims/norm_prims.h`；
- `llm/include/workercore/workercore.h`；
- `llm/include/defs/spec.h`；
- `llm/src/dte/dte_async.cpp`；
- `llm/src/dte/dte_unit.cpp`；
- `llm/src/prims/norm_prims/dte_async_prim.cpp`；
- `llm/src/workercore/workercore.cpp`；
- `llm/src/common/config.cpp`；
- `llm/src/monitor/config_helper_core.cpp`；
- `llm/src/utils/config_utils.cpp`；
- `llm/src/defs/spec.cpp`；
- `llm/test/simulation_config/default_spec.json`。

测试和文档：

- `llm/src/dte/v3_selftest.cpp`；
- `llm/unittest/npusim.cpp`；
- `llm/test/dte/run_test_dte_v3.py`；
- `llm/test/dte/simulation/v3_async_{on,off}.json`；
- `llm/test/dte/workload/v3_*.json`；
- `notes/extensions/DTE/DTE建模计划.md`；
- `notes/extensions/DTE/DTE分版本开发与验收清单.md`；
- `notes/extensions/DTE/log/V3a_development.md`。

## 10. 与计划的差异和后续边界

- 计算依赖采用独立 WAIT primitive，而不是扩展每一种计算 primitive 的 wire；这保持现有计算
  ISA 逐位不变，也允许一个 wait 保护后续多个依赖计算。
- poll 结果保存在 primitive 运行态并进入 trace；仓库没有条件分支 primitive，因此本版不
  提供基于 poll 的 workload 分支。
- V3a descriptor 只计 endpoint DTE service，不生成 NoC/D2D 数据消息；真实网络传输继续由
  SEND/RECV 模型负责，避免重复计费。
- 地址 hazard 覆盖 async DMA descriptor 之间的 RAW/WAR/WAW；计算对 DMA 的依赖由显式
  WAIT/FENCE 声明，不从计算 primitive 的 label 自动推导数值地址。
- cancellation 仅支持 pending descriptor；active cancellation 保持明确拒绝。
- V3b aggregation/coalescing、地址连续性聚合窗口、completion fan-out 和论文趋势复现尚未开始。

V3a 已满足进入 V3b 范围评审的条件。

## 11. 评审闭环

2026-07-24 的独立评审逐文件检查 `dte_async.h/.cpp`、`Dte_async_prim` wire 和
`worker_core_execute()` 调度改动，并独立构建、复跑全部 DTE 与关键 D2D 回归。结论如下：

- dispatcher 只新增 `Dte_async_prim` 分支，该分支不等待 `prim_block`；ISSUE 因而立即返回，
  WAIT/FENCE/CANCEL 在 dispatcher 自身线程中按语义阻塞。原有 SEND、RECV、compute 分支及
  `prim_block.negedge_event()` 协议保持不变；
- 地址冲突条件为区间重叠且至少一侧写，read/read 不等待；hazard 使用
  `WaitForCompletion()`，不会隐式消费旧 token；
- cancel 仅从 `PENDING` queue 删除 descriptor，不修改未计入的 `active_count_`；
  `CANCELLED` 已纳入 `Release()`，不存在取消后的 context 泄漏；
- 精确时序仍为 blocking 421ns、overlap 345ns，节省 76ns；V0/V1/V2a/V2b 和 D2D
  selftest/V4/V5 回归均保持通过；
- 评审再次确认并接受既定范围：`Dte_async` 是 workload 显式使用的 endpoint primitive，
  不生成 REQUEST/ACK/DATA，也未替换 V1–V2b 的 SEND_DATA/RECV_DATA 路径。

本轮评审未发现需要修改的生产代码；变更仅用于补齐评审证据和回归记录。
