# DTE V4 开发记录

## 1. 版本结论

V4 完成 DTE endpoint 模型收尾：在 V0–V3b 已有阻塞、流式、异步和在线聚合能力上，
新增四个方向、四类独立数据端口、每 channel 固定双命令槽、有限 descriptor credit、
backpressure，以及参数化功耗/面积统计。

本版按范围不实现数据组织：scatter、broadcast、stride、slice、shuffle 均明确暂缓，
不属于 V4 验收条件。当前 workload 也没有证明多 `RECV_DATA` dispatcher 并发的必要性，
因此保持 V2b 已冻结的 dispatcher 边界。

## 2. 冻结的模型契约

### 2.1 六个方向与资源映射

| `DteDir` | 语义 | endpoint 资源 |
|---|---|---|
| `SPM_TO_REMOTE` | 本地 SPM 读出到远端 | `SPM_READ` |
| `REMOTE_TO_SPM` | 远端数据写入本地 SPM | `SPM_WRITE` |
| `SPM_TO_SPM` | 本地 SPM 搬移 | `SPM_READ | SPM_WRITE` |
| `SPM_TO_DRAM` | 本地 SPM 写入 DRAM 侧 | `SPM_READ | AXI_WRITE` |
| `DRAM_TO_SPM` | DRAM 侧读入本地 SPM | `AXI_READ | SPM_WRITE` |
| `DRAM_TO_REMOTE` | DDR/DRAM 读出到 remote tile | `AXI_READ` |

`DRAM_TO_REMOTE` 对应验收清单中的 DDR→remoteTile；名称使用 DRAM 作为统一 memory-side
来源，不改变计费边界。

### 2.2 独立端口与复合传输

V4 的 `SPM_READ`、`SPM_WRITE`、`AXI_READ`、`AXI_WRITE` 各有独立可配置位宽。
同一端口的 transfer 通过 command-slot RR 公平仲裁并串行服务，资源集合不相交的 transfer 可以并行。复合方向只有在
所需端口全部空闲时才原子启动，避免先占一部分资源造成 hold-and-wait；各端口按
`ceil(payload_bits / port_width_bits)` 独立释放，transfer context 在最慢端口完成时通知。

V0–V3 的 `fine_grained_resources=false` 路径保留一条 legacy shared bus、每 channel 一条 active
命令和原 trace/时序，V4 不改变其语义。

### 2.3 双命令槽与有限 credit

V4 固定：

```text
active command capacity = channel_count × 2
accepted descriptor capacity = active command capacity + pending_queue_depth
```

每个 slot 记录 `(channel_id, command_slot)`。容量用尽时 `TryIssue()` 拒绝；WorkerCore 的
`DteAsyncTracker` 在物理 issue 前调用 `WaitForCredit()`，等待已接收 descriptor 完成并归还
credit，因此 backpressure 会真实推迟后续 logical issue，而不是只增加统计计数。

双命令槽是固定硬件结构，不作为额外探索参数；配置在 V4 模式必须恰好为 2。

### 2.4 地址与冒险

V4 将异步访问从单一 READ/WRITE 扩展为 `NONE/READ/WRITE/READ_WRITE`，并分别记录 SPM 读、
SPM 写半开区间：

- SPM→SPM：`spm_addr/spm_size` 是读源，`remote_addr` 是本地写目的；
- SPM→DRAM：仅本地 SPM 读区间；
- DRAM→SPM：仅本地 SPM 写区间；
- DRAM→remote：没有本地 SPM 区间，wire 中使用 canonical zero-SPM 表示；
- 既有两个 remote 方向继续使用原来的本地 SPM 区间。

hazard 判定对实际区间执行 RAW/WAR/WAW，read/read 仍可并发。V3b coalescing 只接受原先的
SPM→remote 和 remote→SPM 连续请求；新增复合/DRAM 方向不会被错误并组。

## 3. 与 DRAM、NoC、D2D 的计费边界

V4 模拟的是本核 DTE endpoint 端口服务：

- AXI read/write 表示 DTE endpoint 对 memory-side 接口的占用；
- SPM read/write 表示本地 scratchpad 接口的占用；
- 不调用 DRAM row/bank/media 行为模型；
- 不生成 NoC 或 D2D 数据消息，也不重复计算 hop/link service；
- 真实端到端数据仍由 SEND/RECV 路径承担。

集成测试在 behavioral DRAM 开关两侧重跑同一六方向 workload，完成时刻、端口跨度和能耗
完全一致，并检查没有新增非 DTE DRAM trace。这证明 V4 没有把同一 DRAM 服务计算两次。

## 4. 配置与启动门禁

simulation config：

```json
{
  "dte": {
    "use_beha_dte": true,
    "async": true,
    "fine_grained_resources": true
  }
}
```

hardware `dte` 增加：

- `command_slots_per_channel`：V4 必须为 2；
- `pending_queue_depth`：必须大于 0；
- `spm_read_width_bits`、`spm_write_width_bits`、`axi_read_width_bits`、
  `axi_write_width_bits`：均必须大于 0；
- `launch_energy_pj`、`spm_energy_pj_per_bit`、`axi_energy_pj_per_bit`；
- `base_area_um2`、`channel_area_um2`、`command_slot_area_um2`、
  `port_bit_area_um2`：均必须非负且有限。

`fine_grained_resources=true` 依赖 `async=true`，后者继续依赖 behavioral DTE、dataflow 和顺序
dispatcher。默认配置保持 V4 关闭，功耗/面积系数默认 0。

## 5. 功耗与面积

每条物理 transfer 的动态能耗：

```text
E_transfer = launch_energy
           + Σ(required SPM ports) payload_bits × spm_energy_per_bit
           + Σ(required AXI ports) payload_bits × axi_energy_per_bit
```

复合方向对实际占用的每个端口分别计费；coalesced physical transfer 只按物理 launch 和物理
payload 计费。平均动态功耗由累计动态能量除以首条物理 descriptor issue 到当前/最终完成的 elapsed time 得出。

面积：

```text
A = base
  + channel_count × area_per_channel
  + channel_count × 2 × area_per_command_slot
  + (spm_read_width + spm_write_width + axi_read_width + axi_write_width)
      × area_per_port_bit
```

这些是可标定的解析探索模型，不是 TX8 硅后测量；系数为 0 时不产生未经标定的绝对值。

## 6. Trace 与统计

V4 新增：

- `DTE_port_service`：xfer、core、channel、command slot、port 和该端口的 begin/end；
- `DTE_stats`：completed、issued、累计动态能耗、面积、平均功耗和 credit stall 次数。

legacy 模式不发出这两个 V4 事件，避免改变 V0–V3 的冻结 trace；原
`DTE_pending/launch/bus_wait/transmit` 语义保持不变。

## 7. 关键实现文件

- `llm/include/dte/dte_types.h`：六方向、四端口、V4 配置/统计/context；
- `llm/include/dte/dte_unit.h`、`llm/src/dte/dte_unit.cpp`：双 slot、credit、四端口调度、PPA；
- `llm/include/dte/dte_async_types.h`、`dte_async.h`、`llm/src/dte/dte_async.cpp`：双区间 hazard 和 credit-aware issue；
- `llm/src/prims/norm_prims/dte_async_prim.cpp`：六方向 JSON/wire 校验；
- `llm/include/defs/spec.h`、`llm/src/defs/spec.cpp`、`llm/src/utils/config_utils.cpp`：配置和门禁；
- `llm/src/workercore/workercore.cpp`：每核 V4 配置装配；
- `llm/src/dte/v4_selftest.cpp`：V4 SystemC/纯逻辑自测；
- `llm/test/dte/oracle.py`：独立端口周期、能耗、面积 oracle；
- `llm/test/dte/run_test_dte_v4.py`：WorkerCore 集成与负例矩阵；
- `llm/test/dte/hardware/v4*.json`、`simulation/v4*.json`、`workload/v4_*.json`：测试资产，含 async + blocking SEND/RECV mixed-credit workload。

## 8. 专项测试结果

### 8.1 V4 selftest

`./build/npusim --dte-v4-selftest`：19/19。

覆盖配置合法性、六方向资源 mask、SPM→SPM/DRAM→remote wire、六方向完成、端口闭式周期、
SPM read/write full-duplex、同 SPM read contention、双命令槽、有限容量/credit、精确能耗、
精确面积、平均功耗和 drain。

selftest 配置为 3 channels、2 slots、端口位宽 64/32/16/128 bit；六条 256-bit transfer 的
累计动态能耗为 90.72 pJ，面积为 1,480 μm²。

### 8.2 WorkerCore 集成

`python3 llm/test/dte/run_test_dte_v4.py`：18/18。

六方向 4,096-bit workload 的关键结果：

| xfer | 方向 | 端口服务区间（ns） |
|---:|---|---|
| 0 | SPM→remote | SPM read 100–228 |
| 1 | remote→SPM | SPM write 102–358 |
| 2 | SPM→SPM | SPM read 358–486；SPM write 358–614 |
| 3 | SPM→DRAM | SPM read 486–614；AXI write 486–550 |
| 4 | DRAM→SPM | AXI read 618–1130；SPM write 618–874 |
| 5 | DRAM→remote | AXI read 1130–1642 |

workload 于 1652 ns 完成。累计动态能耗 551.52 pJ，面积 1,240 μm²，平均动态功耗
0.357 mW；三者与 Python oracle 一致。

其余断言包括：独立 SPM read/write 重叠、同 SPM read 串行、两个 command slot 同时占用、
第四条 descriptor 因 credit 于 216 ns 才 issue、credit stall=1、SPM→SPM 写目的与后续读的
RAW 串行、DRAM on/off 等价、V4 方向在关闭模式拒绝、V4 必须启用 async、命令槽必须为 2，
以及 V4 关闭时 V3a overlap 冻结为 345 ns 且不产生 V4 port event。

评审闭环新增 `v4_mixed_send_recv_credit.json`：源核以 3 个 SPM read async descriptor、目的核
以 3 个 AXI read async descriptor 分别占满 `2 active + 1 pending` 容量，然后执行真实
SEND_DATA/RECV_DATA。源 blocking SEND 在首 async 于 2236 ns 完成后 issue，目的 blocking
RECV 在首 async 于 8344 ns 完成后 issue；两核最终均为 issued/completed=4、stall=1，完整
workload 于 24742 ns 完成且没有 `credits exhausted`。

### 8.3 Python oracle

`python3 llm/test/dte/oracle.py`：`DTE V0/V2b/V3b/V4 oracle self-test: PASS`。

## 9. 历史回归

最终完整回归于 2026-07-25 根据 mixed-credit 评审修复后重新执行，全部通过：

- `cmake --build build --target npusim -j2`：clean build；
- DTE selftest：V0 64/64、V3a 31/31、V3b 21/21、V4 19/19；
- DTE WorkerCore：V1 13/13、V2a 15/15、V2b 18/18、V3a 14/14、V3b 16/16、V4 18/18；
- Python oracle：`DTE V0/V2b/V3b/V4 oracle self-test: PASS`；
- D2D shared-path selftest：308/308；
- D2D integration：V0 67/67、V4 13/13、V5 23/23。

V4-off 的 V3a overlap workload 仍精确完成于 345 ns；legacy DTE trace 不出现 V4 port/stats
事件。共享 Msg wire、behavioral D2D streaming metadata 和既有 D2D link/backend 均无回归。

## 10. 收尾与后续边界

除明确暂缓的数据组织外，DTE 的计划开发项已经完成。未来若恢复 scatter/broadcast/stride，
应作为独立版本重新定义：

- descriptor wire 和地址生成器；
- 一对多或多段 completion/token 生命周期；
- 每段/每目的地的 buffer、credit、hazard 和取消语义；
- 与 endpoint 端口、DRAM、NoC/D2D 的计费边界；
- 对应闭式 oracle、集成矩阵和历史回归。

在这些契约冻结前，不应把数据组织隐式塞进已完成的 V4 endpoint 资源模型。
