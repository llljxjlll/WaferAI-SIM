# DTE 分版本开发与验收清单

## 1. 文档目的

本文档是 `notes/extensions/DTE/DTE建模计划.md` 的执行清单，用于跟踪 DTE 模块每一版的：

- 开发目标和范围；
- 需要修改或新增的代码；
- 本版完成后应具备的能力；
- 必须执行的测试；
- 验收标准和进入下一版的条件。

版本按依赖关系顺序实施：

```text
V0a 模型契约
  ↓
V0b 独立 DTEUnit
  ↓
V1 阻塞式 workload 接入
  ↓
V2a 多 SEND 并发
  ↓
V2b 流式端到端与可选 RECV 并发
  ↓
V3a compute/DMA 异步重叠
  ↓
V3b COMET aggregation/coalescing
  ↓
V4 扩展方向与精细硬件资源
```

每一版必须满足本版验收标准后才能进入下一版。若后续实现改变了 `DTE建模计划.md` 中的模型契约，应先更新计划和本清单，再修改代码。

## 2. 全版本通用要求

### 2.1 模型约束

- 每个 core 拥有独立 DTE，不能无意共享 pending、active 或带宽状态。
- DTE 内部数据量统一使用 bit。
- `channel_count` 表示 active transfer 的硬上限。
- `bit_width_bits` 表示所有 channel 共享数据通路的聚合位宽。
- pending transfer 不计入 active 数量。
- 排队时间由状态机显式产生，不能同时通过经验公式重复计费。
- DTE 不重复模拟 NoC/D2D 的 hop、link service 或 credit 延迟。
- 所有时间计算优先使用整数 cycle，并明确向上取整规则。

### 2.2 工程要求

- 新代码放在 `llm/include/dte/`、`llm/src/dte/`。
- 不使用裸 owning pointer；优先使用 `std::unique_ptr`。
- 并发 transfer context 在完成前必须保持稳定地址。
- 非法配置在启动期抛出异常，不能只记录日志后继续运行。
- `SPEC_USE_BEHA_DTE=false` 时保持既有功能和时序不变。
- 新增配置必须提供默认值、示例和错误信息。
- 每个版本新增的模型状态都应有 trace 或调试输出，能够定位错误。

### 2.3 通用交付物

每版完成时至少提交：

- 对应代码和配置；
- 自动化测试；
- 测试 oracle 或预期值说明；
- 一份版本测试结果记录；
- 对 `DTE建模计划.md` 和本文档勾选项的同步更新；
- 已知限制和未覆盖场景说明。

## 3. V0a：冻结模型契约

### 3.1 本版目标

在编写 DTE 调度代码前，消除数据单位、参数含义、延迟公式和计费边界中的歧义。本版只完成设计契约和纯函数规格，不接入 workload。

### 3.2 开发事项

#### 数据单位

- [x] 核对 `M_D_DATA` 的单位并在定义处补充 bit 注释。
- [x] 核对 `Send_prim::end_length` 的单位并补充 bit 注释。
- [x] 核对 `Msg::length_` 的单位并补充 bit 注释。
- [x] 核对所有生成 `Msg` 的路径，确认传入的 length 均为 bit。
- [x] 检索是否存在将 `length_` 当 byte 使用的现有代码，记录或修复单位问题。
- [x] 定义 `payload_bits` 为 DTE API 唯一数据量单位。

#### payload 计算

- [x] 定义 `ComputeSendPayloadBits(const Send_prim&)` 的输入、输出和非法输入行为。
- [x] 明确 `max_packet==0` 是合法空传输还是配置错误。
- [x] 明确 `max_packet>0` 时 `end_length` 的合法范围。
- [x] 确认 payload 计算不依赖 `stripe_count`。
- [x] 确认 behavioral NoC 代表包不能直接用于恢复整块 payload。
- [x] 固定接收侧精确 payload 来源：physical 使用 `length_`；behavioral 使用 `(roofline_packets_-1)×M_D_DATA+length_`。
- [x] 检查 `uint64_t` 乘加溢出。
- [x] 保留 `HW_NOC_PAYLOAD_PER_CYCLE` 压缩前的真实包数信息，确保 DTE payload 不被低估。

#### 参数语义

- [x] 将 `channel_count` 固定为 active transfer 上限。
- [x] 将 `bit_width_bits` 固定为共享聚合数据通路位宽。
- [x] 明确 channel launch 是否允许并行。
- [x] 明确 transmit 阶段使用共享 bus RR 仲裁。
- [x] 明确首期每 active channel 仅容纳一条 descriptor。
- [x] 明确 TX8 每 channel 两条命令暂不作为探索参数。

#### 时间模型

- [x] 定义 `CeilDiv(payload_bits, bit_width_bits)`。
- [x] 明确 ns 到 cycle 的转换与向上取整规则。
- [x] 明确 V1 使用 store-and-forward 保守模型。
- [x] 明确 V2b 使用 streaming/pipeline 模型。
- [x] 明确 DTE 与 NoC/D2D 的计费边界。
- [x] 暂定 `T_launch = gamma + tau_launch`，不使用 `q_i×tau_launch` 重复表达排队。

#### COMET 参数核对

- [x] γ 的出处为 COMET 第 8 页 §VI-A：fixed launch time 取 40 μs。
- [x] τ_launch 的出处为 COMET 第 8 页 §VI-A：DMA descriptor-handling latency 取 2 μs。
- [x] 记录适用范围、时间单位和来源链：COMET 明确沿用参考文献 [38] 的 UCIe 性能模型，未将数值标定为 TX8/per-core DTE 参数。
- [x] 40000ns、2000ns 可作为可配置的 COMET baseline 默认值，不能视为目标硬件固有常量。
- [x] 已找到来源并移除“待标定”表述，同时保留配置覆盖能力。
- [x] 建立论文变量到模拟器字段的映射表。

### 3.3 本版完成内容

完成后应具备：

- 明确且无冲突的 bit 单位契约；
- 可独立实现的 payload 和时间计算规格；
- channel、位宽和共享 bus 的唯一语义；
- V1/V2b 计费边界说明；
- 可追溯的 COMET 参数出处、引用链和适用边界。

### 3.4 测试

#### 静态核查

- [x] 使用 `rg` 检索 `M_D_DATA`、`end_length`、`length_` 的生产和消费位置。
- [x] 检查所有相关注释是否与实际单位一致。
- [x] 检查计划、配置示例和 API 命名是否统一使用 bit。

#### 纯函数规格用例

- [x] `max_packet=1, end_length=1` → 1 bit。
- [x] `max_packet=1, end_length=128` → 128 bit。
- [x] `max_packet=2, end_length=1` → 129 bit。
- [x] `max_packet=2, end_length=128` → 256 bit。
- [x] `HW_NOC_PAYLOAD_PER_CYCLE=20`、原始 100 包 → `max_packet=5`，DTE 仍恢复 12800 bit。
- [x] 非满末组和短尾包可精确恢复，且聚合元数据经过 `Send_prim` wire round-trip 不丢失。
- [x] stripe 1/2/4 对同一逻辑传输返回相同 payload。
- [x] behavioral/physical NoC 对同一逻辑传输返回相同 payload。
- [x] `CeilDiv(1,128)=1`。
- [x] `CeilDiv(128,128)=1`。
- [x] `CeilDiv(129,128)=2`。
- [x] 检查零除数、非法尾包和整数溢出。

### 3.5 验收标准

- [x] 文档中不存在 byte/bit 混用。
- [x] `channel_count` 不再被描述为无界队列数量。
- [x] 位宽明确为 aggregate，而不是 per-channel。
- [x] store-and-forward 的保守性已写明。
- [x] COMET 参数具备页码、章节、引用链和非 TX8 实测值的边界说明。
- [x] 纯函数预期值经过人工复核。

## 4. V0b：实现独立 DTEUnit

### 4.1 本版目标

实现不依赖 `WorkerCoreExecutor` 的独立 SystemC DTE 模块，正确模拟 pending、active channel、launch、共享 bus 仲裁和完成通知。

### 4.2 开发事项

#### 文件与类型

- [x] 新增 `llm/include/dte/dte_types.h`。
- [x] 新增 `llm/include/dte/dte_payload.h`。
- [x] 新增 `llm/include/dte/dte_unit.h`。
- [x] 新增 `llm/src/dte/dte_unit.cpp`。
- [x] 定义 `DteDir`。
- [x] 定义 `DTEConfig`。
- [x] 定义地址稳定的 `DteTransferContext`。
- [x] 定义唯一递增的 `xfer_id`，并处理回绕策略。

#### 配置

- [x] 在 `CoreHWConfig` 增加 `dte_channel_count`。
- [x] 在 `CoreHWConfig` 增加 `dte_bit_width`。
- [x] 增加 γ 和 launch latency 的硬件配置。
- [x] 为所有字段设置回归安全的默认值。
- [x] 校验 channel、位宽大于 0。
- [x] 校验延迟非负且可转换为 cycle。
- [x] 错误信息包含字段名、非法值和 core id。

#### 状态机

- [x] 实现 `pending_queue_`。
- [x] 实现最多 `channel_count` 个 active slot。
- [x] 实现 slot 释放后立即 admit pending。
- [x] 实现 launch 阶段。
- [x] 实现 `bus_wait_queue_`。
- [x] 实现共享 bus RR 仲裁。
- [x] 保证任意时刻最多一个 transmit。
- [x] 完成后通知对应 context 的 event。
- [x] 完成后安全回收 context。
- [x] 确保同一 transfer 不会重复完成。

#### trace

- [x] 输出 `DTE_pending` B/E。
- [x] 输出 `DTE_launch` B/E。
- [x] 输出 `DTE_bus_wait` B/E。
- [x] 输出 `DTE_transmit` B/E。
- [x] trace 携带 core、xfer_id、channel_id、direction、payload_bits。

### 4.3 本版完成内容

完成后应具备：

- 可单独实例化的 `DTEUnit`；
- 真正受 `channel_count` 限制的 active transfer；
- 可并行的 launch 阶段；
- 单共享 bus 的 RR transmit；
- 稳定、无丢失的完成通知；
- 可由 trace 完整解释的四段延迟。

### 4.4 测试

#### 纯函数测试

- [x] 覆盖 V0a 全部 payload 和 `CeilDiv` 用例。
- [x] 覆盖不同位宽和非整除 payload。
- [x] 覆盖大 payload 和溢出边界。

#### 单 transfer

- [x] channel=1，单 descriptor。
- [x] channel=4，单 descriptor；完成时刻应与 channel=1 相同。
- [x] payload 小于、等于、大于一个位宽。
- [x] 验证完成时刻为 launch 加 transmit。

#### channel admission

- [x] channel=1，同时 issue 4 条，active 始终不超过 1。
- [x] channel=2，同时 issue 4 条，active 始终不超过 2。
- [x] channel=4，同时 issue 4 条，4 条均可 active。
- [x] slot 完成释放后，pending 在同一允许调度点被 admit。

#### bus 仲裁

- [x] 多 channel 同时结束 launch，按 RR 顺序 transmit。
- [x] 长短 descriptor 混合。
- [x] 不同仿真时刻 issue。
- [x] 连续新请求不能使旧请求饥饿。
- [x] 任意两个 `DTE_transmit` span 不重叠。

#### event 与生命周期

- [x] waiter 在完成前开始等待，能收到通知。
- [x] 多个 context 各收到自己的完成。
- [x] 不发生 event 丢失、串线或重复通知。
- [x] context 完成前不释放。
- [x] 仿真结束时 pending/active context 正确清理。

#### 配置错误

- [x] channel=0。
- [x] bit_width=0。
- [x] 负延迟。
- [x] 超出 cycle 表达范围。
- [x] 错误均在仿真正式运行前失败。

#### Oracle

- [x] 新增 `llm/test/dte/oracle.py`。
- [x] oracle 独立模拟 pending、active、launch、bus RR 和完成时刻。
- [x] C++ selftest 与 Python oracle 使用相同输入并逐条比较。

### 4.5 验收标准

- [x] 全部单元测试和 oracle 对比通过。
- [x] active 数从未超过配置。
- [x] bus transmit 从不重叠。
- [x] RR 无饥饿。
- [x] 完成事件无丢失、无串线。
- [x] trace 可以解释每条 transfer 的完整时间。
- [x] 未修改现有 workload 时序。

## 5. V1：阻塞式接入真实 workload

### 5.1 本版目标

将 DTE 接入普通 `SEND_DATA` 和 `RECV_DATA` 路径，使 DTE 延迟在 workload trace 和最终完成时间中可观测。

### 5.2 开发事项

#### WorkerCore 实例化

- [x] `WorkerCoreExecutor` 增加 `std::unique_ptr<DTEUnit>`。
- [x] 从当前 core 的 `CoreHWConfig` 读取配置。
- [x] 每个 core 使用唯一 SystemC module name。
- [x] 验证不同 core 的 DTE 状态完全独立。

#### 发送侧

- [x] 在 `SEND_DATA` 第一个数据包进入 NoC 前计算逻辑 payload。
- [x] issue `SPM_TO_REMOTE` transfer。
- [x] 等待对应完成事件。
- [x] 完成后进入既有逐包/代表包发送流程。
- [x] SEND_REQ 仅携带 payload 元数据且不支付 DTE 延迟；SEND_DONE 等控制路径不变。
- [x] behavioral NoC 下不按代表包错误缩小 payload。
- [x] DTE 关闭时 REQUEST 构造不计算或校验 DTE payload。

#### 接收侧

- [x] 在每条 RECV_DATA 开始时清零本原语 payload 状态。
- [x] 正确获得多 source、多 stripe 的逻辑 payload。
- [x] 仅在所有预期 DATA 到齐后 issue `REMOTE_TO_SPM`。
- [x] 等待 DTE 完成后才结束 RECV_DATA 原语。
- [x] 不修改 RECV_ACK、RECV_CONF 等控制路径。
- [x] 重复尾包、非法 subflow 等既有错误仍能被检测。

#### 事件与派发

- [x] 不复用 `ev_block` 作为 DTE completion。
- [x] DTE 等待嵌套在当前原语内部。
- [x] 原语末尾仍只通知一次 `ev_block`。
- [x] V1 阶段禁止 `SPEC_USE_BEHA_DTE && SPEC_SEND_RECV_PARALLEL`；该限制在 V2a 验收后解除。
- [x] V1 阶段禁止组合在配置解析后抛出异常；V2a 后仅非 dataflow 组合继续拒绝。

#### 配置示例

- [x] 更新默认 hardware config。
- [x] 更新默认 simulation config。
- [x] 添加一个开启 DTE 的最小测试配置。

### 5.3 本版完成内容

完成后应具备：

- 每核 DTE 实例；
- 普通 SEND_DATA 的源端 DTE 延迟；
- 普通 RECV_DATA 的目的端 DTE 延迟；
- trace 中可见 DTE 四段延迟；
- 默认关闭时完全回归兼容；
- 明确的 store-and-forward 保守端到端模型。

### 5.4 测试

#### 发送集成

- [x] 单包 1 bit。
- [x] 单包 128 bit。
- [x] 多包且尾包 1 bit。
- [x] 多包且尾包 128 bit。
- [x] stripe=1、2、4。
- [x] physical NoC 与 behavioral NoC 使用相同逻辑 payload。
- [x] 预期新增时间为 launch、资源等待和 `ceil(bits/width)×CYCLE`。

#### 接收集成

- [x] 单 source、单 stripe。
- [x] 多 source。
- [x] 多 stripe。
- [x] behavioral 代表包恢复精确逻辑 payload。
- [x] 重复尾包仍报错。
- [x] 非法 subflow 仍报错。
- [x] DTE 完成前 RECV_DATA 不结束。
- [x] pipeline/refill 两轮复用同一 `(source, tag)` 时，每轮元数据均被消费并可重新插入。

#### 端到端

- [x] 单 core pair 完整 SEND/RECV。
- [x] 同 die 通信。
- [x] 跨 die 通信。
- [x] source DTE、NoC/D2D、destination DTE trace 次序符合 store-and-forward。
- [x] 最终 DONE 时间包含预期 DTE 延迟。

#### 开关矩阵

- [x] DTE off + physical NoC。
- [x] DTE off + behavioral NoC。
- [x] DTE on + physical NoC。
- [x] DTE on + behavioral NoC。
- [x] V1 基线曾要求 parallel 启动失败；V2a 后改由 parallel dataflow 兼容测试覆盖。

#### 多 core 隔离

- [x] 两个 core 同时使用各自 DTE，不共享 active slot。
- [x] 一个 core 的 bus contention 不延迟另一 core。
- [x] per-core 不同 channel/位宽配置分别生效。

#### 回归

- [x] DTE 默认关闭，运行全部既有测试。
- [x] 关闭时 DONE 时刻与改动前一致。
- [x] 关闭时 trace 不出现 DTE span。

#### 覆盖说明

- 1/128/129/256 bit 与尾包边界由 `--dte-v0-selftest` 直接覆盖；V1 的
  `send_logic()` 调用同一个 `ComputeSendPayloadBits()`，生产配置生成和 SEND_REQ wire
  round-trip 也在该自测中覆盖。该自测还用退化 SEND_REQ 直接验证 DTE off 不触发严格
  payload 校验、DTE on 仍会拒绝非法 payload。
- stripe=1 由标准 workload 覆盖，stripe=2/4 由跨 die workload 覆盖；三者的 DTE
  payload 均不随条带数变化。D2D V5 的 1/2/4 stripe 协议回归全部通过。
- pipeline=2 的多源 workload 在同一模拟中连续复用同一 `(source, tag)`：两轮各产生
  两个 16,384-bit 源 transfer 和一个 32,768-bit 目的 transfer，off/on 完成时间为
  1033/1115 ns，验证消费后重新插入的生命周期。
- 每核独立性由 trace 中重叠的 source transmit span 验证；异构配置用 core 0/1/2 的
  2048/1024/4096-bit 位宽分别得到 22/38/22 ns。`channel_count` 的配置隔离由 V0
  资源自测覆盖；V1 每核单传输阻塞路径按契约不应因 channel 数改变完成时间。
- “全部既有测试”按仓库现有统一入口执行：DTE V0/V1、NoC congestion、D2D
  V0/V3/V4/V5，均通过。V0 runner 为 67/67 组，V3 为 16/16，V4 为 13/13，
  V5 为 23/23；D2D SystemC 自测为 308/308。

### 5.5 验收标准

- [x] SEND 和 RECV 集成测试全部通过。
- [x] bit/byte 边界用例无 8 倍误差。
- [x] behavioral 代表包不造成 payload 少算。
- [x] 默认关闭时既有测试和时序不变。
- [x] store-and-forward 限制在文档和结果中清晰标注。

## 6. V2a：真实 workload 中的多 SEND 并发

### 6.1 本版目标

在现有 `send_para_logic()` 能覆盖的真实路径中实现连续 SEND_DATA 的非阻塞 issue，让 `channel_count` 在真实 workload 中产生可观测作用。

### 6.2 开发事项

- [x] 梳理 `send_para_queue` 的所有权和删除/回填规则。
- [x] 仅对同批连续 `SEND_DATA` 做背靠背 DTE issue。
- [x] 为每条 SEND 建立独立、地址稳定的 context。
- [x] 记录 SEND 原语与 xfer_id 的对应关系。
- [x] DTE 完成后才允许对应 SEND 进入数据发送阶段。
- [x] 明确批内完成顺序是否允许与 issue 顺序不同。
- [x] 若必须保持原语顺序，在 DTE 完成后增加有序提交。
- [x] 保证控制原语行为不变。
- [x] 解除 V1 对 parallel 组合的限制，但只开放经过验证的 SEND 场景。
- [x] 对尚未支持的组合继续启动期拒绝或运行时明确报错。

### 6.3 本版完成内容

完成后应具备：

- 同批多个 SEND_DATA 可同时 pending/active；
- `channel_count` 限制 active SEND；
- launch 可并行、transmit 共享 bus；
- 每条 SEND 在自己的 DTE 完成后继续；
- 明确且可测试的批内顺序语义。

本版不包含 RECV_DATA 并发，也不宣称实现 SEND/RECV 双向并发。

### 6.4 测试

- [x] 1/2/4 个连续 SEND，channel=1。
- [x] 1/2/4 个连续 SEND，channel=2。
- [x] 1/2/4 个连续 SEND，channel=4。
- [x] 相同长度 SEND。
- [x] 长短混合 SEND。
- [x] 不同目的 core。
- [x] 同目的 core。
- [x] channel active 上限正确。
- [x] bus 峰值不随 channel 数增加。
- [x] channel 增多能够减少 admission 等待或增加 launch overlap。
- [x] 完成事件不会串到其他 SEND。
- [x] 若要求有序提交，实际数据发送顺序符合原语顺序。
- [x] `prim_refill` 场景不泄漏或重复使用 context。
- [x] 批次结束后 `send_done`、`prim_block` 状态正确。
- [x] 现有 parallel 测试无回归。

#### 覆盖说明

- `send_para_logic()` 在批次开始时预扫描所有 `SEND_DATA`，为每条原语背靠背 issue
  独立 context，并保存 `Send_prim* → (xfer_id, context*)` 映射；context 仍由
  `DTEUnit` 的地址稳定 `std::list<std::unique_ptr<...>>` 持有。
- 批内 DTE 可以并发完成，但 DATA 网络提交保持原始原语顺序。每条 DATA 开始前先检查
  自己的 context 状态，必要时等待独立 `done`，随后按 xfer_id 释放；批末要求映射为空。
- 1/2/4 SEND × channel=1/2/4 的 9 个组合全部通过。四条 16,384-bit SEND 的源端
  完成偏移分别为 channel=1 `[36,72,108,144]` ns、channel=2
  `[36,52,72,88]` ns、channel=4 `[36,52,68,84]` ns；最大 active 为 1/2/4，
  共享 bus 总 transmit 始终为 64 ns。
- 同一目的 core 的 16,384/8,192/4,096/2,048-bit 混合批次在 channel=1/2/4
  下均通过，外部 FLOW_DONE 保持 tag 10→13 顺序。
- 独立附带修复（非 DTE 门控）：修复 parallel fan-out 原有的单一 `send_last_packet`
  token 死锁。同一个 compute-ready token 对整批 DATA 生效，仅在最后一条尾包提交后
  消费；该修复无条件作用于 shared parallel 路径，因此 DTE off 的四 SEND 也从 watchdog
  死锁变为 1073 ns 正常完成且无 DTE span。changelog/提交说明应与 DTE 功能分开列项。
- `(source,tag)` payload metadata 改为 per-key 轮次队列，每轮记录 stripe mask。
  parallel pipeline/refill 两轮可提前到达，off/on 为 867/951 ns，各 core 使用新 xfer id 0/1。
- 非 dataflow 的 DTE parallel 在启动期拒绝；parallel+stripe>1 在运行时明确拒绝；
  同核 SEND/RECV DTE 重叠也有明确运行时门禁。V2a 只声明 dataflow 多 SEND 支持。
- `run_test_dte_v2.py` 共 15/15 通过；V1 兼容矩阵 13/13、DTE 自测 55/55、
  D2D selftest 308/308、D2D V0 67/67、D2D V5 23/23、NoC frozen baseline 4/4。

### 6.5 验收标准

- [x] `channel_count` 在集成 workload 中可观测。
- [x] active 上限和共享带宽语义同时成立。
- [x] context 无泄漏、悬空和重复完成。
- [x] 原语顺序符合已定义契约。
- [x] 文档明确本版只支持多 SEND。

## 7. V2b：流式端到端模型与可选 RECV 并发

### 7.1 本版目标

消除 V1 store-and-forward 对端到端延迟的系统性高估，用流水模型组合源 DTE、NoC/D2D 和目的 DTE；只有 workload 确有需求时才扩展 RECV_DATA 并发。

### 7.2 开发事项

#### 流水模型

- [x] 定义 source DTE 首个数据单元 ready 的时刻。
- [x] 定义 NoC/D2D 首包、末包服务边界。
- [x] 定义 destination DTE 首个数据单元可写和最终完成时刻。
- [x] 选择显式首包/末包事件或闭式 pipeline 近似。
- [x] 稳态带宽使用三段最小有效带宽。
- [x] 分别计入 source launch、network fill、destination completion drain。
- [x] 避免 source/destination DTE 与 NoC 对同一 payload service 重复计费。
- [x] behavioral 和 physical backend 使用一致的端到端语义。
- [x] trace 能显示重叠区间。

#### 可选 RECV 并发

- [x] 用现有真实 workload 评估并发需求：未发现同一 dispatcher 同时执行多个 `RECV_DATA` 原语。
- [x] 完成边界评审：需求未触发，本版不改 dispatcher，未来作为独立版本设计。
- [ ] 定义 RECV 原语、消息 buffer 和 DTE context 所有权（条件项，本版需求未触发）。
- [ ] 定义多个 RECV 对 tag/source/subflow 的匹配（条件项，本版需求未触发）。
- [ ] 定义完成顺序和 `prim_block` 语义（条件项，本版需求未触发）。
- [ ] 实现统一通信调度器或独立 recv batch 队列（条件项，本版需求未触发）。
- [x] 未完成上述设计前不把 RECV_DATA 塞入现有 `send_para_queue`。

### 7.3 本版完成内容

完成后应具备：

- DTE 与网络数据阶段可流水重叠；
- 端到端吞吐由实际瓶颈决定；
- 首包、稳态、尾部完成延迟可解释；
- 可选 RECV_DATA 并发完成需求评估；未触发时保持明确门禁，不扩大本版范围。

### 7.4 测试

#### 瓶颈迁移

- [x] source DTE 最慢。
- [x] NoC/D2D 最慢。
- [x] destination DTE 最慢。
- [x] 三者等带宽。
- [x] 扫描 DTE 位宽跨过网络有效位宽。
- [x] 确认总时间斜率随瓶颈迁移。

#### 流水正确性

- [x] 小 payload，以 fill/launch 为主。
- [x] 大 payload，以稳态带宽为主。
- [x] 非整除 payload。
- [x] 首包时间正确。
- [x] 末包/最终写入时间正确。
- [x] 重叠区间在 trace 中存在。
- [x] streaming 时间不大于同配置 store-and-forward 时间。
- [x] 不出现负时间或因闭式近似造成的倒序事件。

#### backend 组合

- [x] behavioral NoC。
- [x] physical NoC。
- [x] behavioral D2D。
- [x] 同 die和跨 die。

#### 可选 RECV 并发（本版不适用）

当前真实 workload 未证明多 `RECV_DATA` 原语并发需求，因此以下测试随 dispatcher 改造一并
延期，不计入 V2b streaming 验收：

- [ ] 不同 tag。
- [ ] 不同 source。
- [ ] 多 stripe。
- [ ] 消息乱序到达。
- [ ] 一个 RECV 完成不会提前结束另一个。
- [ ] buffer、context 和原语均无泄漏。

#### 覆盖说明

- `dte.streaming` 默认关闭；开启要求 DTE on、dataflow、顺序 dispatcher。V1/V2a 配置不写
  该字段，精确 ns 全部保持不变；legacy DATA wire 原始载荷由自测锁定，反序列化出的
  模拟时间字段只允许在 streaming consumer gate 内使用。
- source 在 DTE transmit 开始后按累计逻辑 bit 等待 readiness；首 DATA 携带 48-bit
  `source_first_ns/source_done_ns`。目的端首包为每个 source 建立一个独立 context。
- physical 末包使用实际到达时间；behavioral 同 die 使用 roofline bulk；跨 die D2D 在
  first-packet service 后交付代表消息，并携带剩余 32-bit tail cycles。
- 最终完成使用 C++ `dte_streaming.h` 与 Python oracle 同构公式，并保证 network/source 尾部后
  至少一个完整目的端 DTE cycle，不出现倒序事件。
- physical store/stream：source-slow 1067/931 ns、network-slow 563/545 ns、
  destination-slow 1067/935 ns、equal 803/553 ns。
- behavioral store/stream：source-slow 1005/927 ns、network-slow 501/483 ns、
  destination-slow 1005/935 ns、equal 741/551 ns。三类尾部候选分别由 source/network/
  destination 主导。
- behavioral 位宽 64/128/256/512/1024/2048/4096 bit 的完成时间为
  `[935,679,551,487,483,483,483]` ns，1024 bit 后进入 network-limited 平台。
- 8-bit 小 payload 为 256 ns；16,416-bit 非整除 payload 为 570 ns，256-bit DTE transmit
  精确为 `ceil(16416/256)×2=130 ns`。
- behavioral D2D stripe=4：同 die 与跨 die flow 均显示 network/destination overlap；整体
  store/stream 为 632/604 ns。
- 多 source 目的端为两个 16,384-bit source 建立两个独立写入 context，751 ns 完成；重复
  `(source,tag)` 两轮 refill 使用全新 xfer id，1179 ns 完成。
- `run_test_dte_v2b.py` 18/18；DTE selftest 64/64、V1 13/13、V2a 15/15、D2D selftest
  308/308、D2D V0/V4/V5 67/67、13/13、23/23、NoC frozen 4/4。

### 7.5 验收标准

- [x] 不再无条件相加三段整块传输时间。
- [x] 瓶颈扫描结果符合解析模型。
- [x] trace 可解释 pipeline fill、稳态和 drain。
- [x] 与 V1 保守模型的差异有量化记录。
- [x] 本版未实现 RECV 并发；需求未触发，顺序 streaming 与 parallel dispatcher 明确互斥。

## 8. V3a：通用 compute/DMA 异步重叠

### 8.1 本版目标

使 core 发起 DMA 后能够继续执行无依赖计算，并在真正依赖数据时等待，而不是在 SEND/RECV 原语内部始终阻塞。

### 8.2 开发事项

- [x] 定义 DMA issue 语义。
- [x] 定义 completion token/xfer_id。
- [x] 定义 wait/poll/fence 语义。
- [x] 定义计算原语如何声明对 DMA 的依赖。
- [x] 决定并实现独立 `Dte_async` primitive 及双 segment wire 编码。
- [x] 改造 `worker_core_execute()` 的派发纪律。
- [x] 支持多个 outstanding DMA。
- [x] 处理原语队列 refill 和循环。
- [x] 处理异常、pending cancel 和仿真结束时未完成 DMA。
- [x] 防止读写同一 SPM 区域的数据冒险。
- [x] trace 展示 compute 与 DTE overlap。

### 8.3 本版完成内容

- 真正的异步 DMA issue；
- 多 outstanding transfer；
- 基于 logical token 的 wait/poll/fence/cancel；
- compute/DMA overlap；
- SPM 区间 RAW/WAR/WAW 依赖和结束纪律；
- token 复用、pipeline/refill 和异常无泄漏。

实现边界：`Dte_async` 模拟 endpoint DTE descriptor，不生成 NoC/D2D 数据消息；计算依赖由
显式 WAIT/FENCE 表达。poll 结果保存在 primitive 的运行态 `poll_complete` 并写入 trace，
但当前仓库没有条件分支 primitive。cancel 只允许 pending descriptor，active cancel 明确拒绝。

### 8.4 测试

- [x] issue 后执行无依赖计算，trace 发生重叠。
- [x] 依赖计算在 DMA 完成前不能开始。
- [x] 两条独立 DMA 与计算重叠。
- [x] 多 DMA 中只等待指定 token。
- [x] fence 等待之前所有相关 DMA。
- [x] channel=1/2/4 下 outstanding 行为正确。
- [x] 同地址 RAW/WAR/WAW 风险被阻止；read/read 允许并发。
- [x] refill/循环消费旧 token 后以新 xfer_id 安全复用。
- [x] 重复 token、非法 descriptor、cancel 和结束负例不遗留 active context。
- [x] 与阻塞模式执行相同 DMA/计算工作量，只改变依赖调度时序。

测试结果（2026-07-24）：`--dte-v3-selftest` 31/31，`run_test_dte_v3.py` 14/14。
blocking 总完成 421ns；overlap 中 Matmul 124→333ns 与 DTE transmit 134→198ns 重叠，
总完成 345ns，缩短 76ns。四 descriptor 在 channel=1/2/4 时 max active 为 1/2/4，
共享 bus transmit 总量均为 128ns。选择性 wait、未完成 poll、fence、RAW/WAR/WAW、pending
cancel、active cancel 拒绝、同队列 token 复用、pipeline refill、未 fence 结束负例和 async-off
门禁均通过。

独立评审确认 `worker_core_execute()` 只新增 `Dte_async_prim` 直接执行分支，既有
SEND/RECV/compute 分支及 `prim_block` 握手未变；hazard wait 不消费旧 token，pending cancel
和 `CANCELLED` release 路径闭环。复跑结果为 V0 64/64、V3a 31/31、V1/V2a/V2b
13/13、15/15、18/18、D2D selftest 308/308、D2D V4/V5 13/13、23/23，未发现问题。

### 8.5 验收标准

- [x] overlap 可由 trace 和总时间同时证明。
- [x] 所有依赖测试通过。
- [x] 无错误提前消费数据。
- [x] 多 outstanding 和 token 生命周期正确。

详细设计、精确时刻、交付文件与回归证据见 `log/V3a_development.md`。

## 9. V3b：COMET aggregation/coalescing

### 9.1 本版目标

在 V3a 异步 endpoint DTE 之上加入 COMET 启发的在线细粒度 descriptor 聚合，使模型能够研究请求粒度、聚合等待、launch 摊薄和共享 bus 有效位利用率之间的关系。

### 9.2 开发事项

- [x] 对照 COMET 第 7–8 页 §V-C、式 (13)–(17) 与 Figure 11 明确论文方法和指标。
- [x] 定义在线可聚合条件：同方向、同 remote peer、同 address block，且本地 SPM 和远端 byte 区间均连续。
- [x] 定义最大 descriptor 数、最大 payload bytes、按 cycle 向上取整的 timeout 与 wait/fence/hazard flush。
- [x] 由 workload 显式提供 `remote_peer`、`remote_addr`、`address_block`，启动期校验声明 block 与配置 block 大小一致。
- [x] 扩展 descriptor 携带远端 endpoint、地址和 block；首期要求 byte 对齐、payload bytes 与 SPM range 等长，不支持 stride。
- [x] 一个 physical context 映射多个 logical token；每个 token 只完成/消费一次，最后一个 token 消费后释放 context。
- [x] 统计 logical descriptor、physical transfer、coalesced descriptor、launch savings 和有效位 bus utilization。
- [x] trace 记录 collect、flush 原因、logical-token→physical-xfer bind 与明确命名的累计利用率。
- [x] aggregation 关闭时保持 V3a 的 2-segment wire、逐 token issue 和冻结时序。

### 9.3 本版完成内容

- 可配置的在线 DMA 请求聚合与严格兼容条件；
- 多 logical descriptor 到一个 physical DTE transfer 的映射和 completion fan-out；
- `limit`、`incompatible`、`timeout`、`dependency`、`fence` flush；
- launch 节省、聚合等待和有效位带宽利用率统计；
- COMET 式 (13)–(15)/Figure 11 的对应关系，以及未实现式 (16)–(17) 路由倍率的明确边界。

### 9.4 测试

- [x] 两个连续小请求可聚合。
- [x] 地址不连续请求不可聚合。
- [x] 不同方向、peer 或 address block 不可错误聚合。
- [x] 达到最大 descriptor 数或最大 bytes 立即发出。
- [x] 未达上限但 timeout 后在无关 compute 执行期间精确发出。
- [x] 每个原始 token 收到且只收到一次完成，共享 context 只释放一次。
- [x] aggregation 关闭时与 V3a 345 ns overlap 时序一致。
- [x] 16 个小请求的 physical launch 随 group 1/2/4/8/16 降为 16/8/4/2/1。
- [x] 单个大请求 aggregation on/off 均为 638 ns，不产生异常额外收益。
- [x] 带宽利用率与独立 Python oracle 一致。
- [x] 复现 COMET Figure 11 的固定总数据量、增大组规模、launch 开销单调摊薄趋势。
- [x] staged tail cancel、非 tail staged cancel 拒绝、已发 compound 成员 cancel 拒绝均有测试。
- [x] aggregation 依赖 async、最大组数、timeout 和 address-block 配置负例均启动失败。

### 9.5 验收标准

- [x] 论文原模型与本仓库在线近似的对应和差异均明确记录。
- [x] completion fan-out、token 生命周期和 context 单次释放正确。
- [x] 聚合收益与从首请求开始的收集等待均被计入。
- [x] 至少一个论文趋势得到复现。

验收结果（2026-07-24）：DTE V3b selftest 21/21、WorkerCore 集成矩阵 16/16。
固定 16×64-bit 总数据量时，group 1/2/4/8/16 的 physical transfer 为
16/8/4/2/1，首 logical issue 到最后 transmit 为 352/178/102/70/66 ns，完整 workload
结束为 548/374/298/266/262 ns；launch savings 为 0/8/12/14/15。group=1 的有效位
利用率为 500000 ppm，group≥2 为 1000000 ppm，均与 Python oracle 精确一致。
详细设计、交付文件和回归证据见 `log/V3b_development.md`。

评审闭环（2026-07-24）：独立代码复核和完整重跑未发现正确性问题。唯一的命名改进已完成：flush trace 的 `utilization_ppm` 改为 `cumulative_utilization_ppm`，明确该值为运行期累计比例；测试对 aggregation-on 断言 trace 累计值，并对全矩阵断言独立重算与 oracle 一致。

## 10. V4：扩展方向与精细硬件资源（已完成）

### 10.1 本版目标与范围

在不实现数据组织的前提下，完成 DTE endpoint 资源模型收尾：扩展四种传输方向，按 TX8
约束建模独立 SPM/AXI 读写端口、每 channel 两条命令、有限 descriptor credit/backpressure，
并提供功耗和面积统计。scatter、broadcast、stride/slice/shuffle 属于数据组织，按本轮决策
明确暂缓，不计入 V4 完成条件。

### 10.2 已完成事项

- [x] SPM→SPM(local)。
- [x] SPM→DRAM。
- [x] DRAM→SPM。
- [x] DDR→remoteTile（代码方向名为 `DRAM_TO_REMOTE`）。
- [ ] scatter（数据组织，按本轮范围暂缓）。
- [ ] broadcast（数据组织，按本轮范围暂缓）。
- [ ] stride/slice/shuffle（数据组织，按本轮范围暂缓）。
- [x] AXI read、AXI write、SPM read、SPM write 四类独立端口/带宽。
- [x] 每 channel 固定两条命令槽；该常数不可作为第三个探索参数。
- [x] read/write 方向独立资源；复合方向原子获取所需端口，完成时刻取最慢端口。
- [x] 有限 pending descriptor credit 和真实 issue backpressure。
- [x] 动态能耗、平均功耗和面积统计。

方向与 endpoint 资源的唯一映射为：

| 方向 | 占用资源 |
|---|---|
| SPM→remote | SPM read |
| remote→SPM | SPM write |
| SPM→SPM | SPM read + SPM write |
| SPM→DRAM | SPM read + AXI write |
| DRAM→SPM | AXI read + SPM write |
| DDR→remoteTile | AXI read |

V4 只计算本核 endpoint 端口服务，不触发或重复统计 DRAM row/bank/media、NoC hop 或 D2D link
服务。`Dte_async` 仍不生成真实数据消息；端到端通信继续由 SEND/RECV 路径负责。

### 10.3 测试

- [x] 六个方向均有单传输闭式 oracle 和 WorkerCore 集成覆盖。
- [x] SPM read/write 双向同时传输的独立性。
- [x] AXI/SPM 端口独立时的并行性和复合方向最慢端口完成语义。
- [x] 共享端口 contention 串行化。
- [x] DRAM 行为模型开关组合时 DTE endpoint 时序不变。
- [x] trace 证明没有重复触发或计算 DRAM 带宽。
- [ ] broadcast/scatter 多 completion（数据组织，按本轮范围暂缓）。
- [x] 固定队列深度的 credit/backpressure。
- [x] async descriptor 占满容量时，blocking SEND/RECV 等待 credit 而非抛异常。
- [x] 双命令槽上限、地址冒险、非法方向和非法配置门禁。
- [x] 功耗/面积公式由 C++ selftest、Python oracle 和集成 trace 三方校验。
- [x] 新功能关闭时严格回归 V3a/V3b 冻结行为。

### 10.4 验收标准

- [x] 每个新增方向的计费边界明确。
- [x] 与现有存储和通信模块无重复计费。
- [x] 资源竞争符合选定的 TX8 endpoint 结构。
- [x] 每 channel 恰好两条命令且 descriptor 容量有限。
- [x] 功耗、面积和 backpressure 指标可配置、可追踪、可由 oracle 复算。
- [x] 所有旧版本测试继续通过。

验收结果（2026-07-24）：DTE V4 selftest 19/19、WorkerCore 集成矩阵 18/18、Python
oracle PASS。4,096-bit 六方向 workload 的端口服务时长分别与端口位宽闭式计算完全一致；
四端口总动态能耗为 551.52 pJ，面积为 1,240 μm²。双命令槽、同端口 contention、有限
pending credit、async 与 blocking SEND/RECV 共用 credit、SPM→SPM 目的地址 RAW、DRAM 模型 on/off 等价和 V4-off 冻结时序均通过。
详细设计、交付文件和回归证据见 `log/V4_development.md`。

## 11. 每版完成记录模板

每完成一版，在本节追加记录：

```text
版本：
完成日期：
负责人：
提交/变更集：

已完成事项：
- ...

实际交付文件：
- ...

执行的测试：
- 测试命令：
- 结果：

与计划的差异：
- ...

遗留问题：
- ...

是否满足进入下一版条件：
- 是/否
- 依据：
```

### V0（V0a + V0b）完成记录

```text
版本：V0
完成日期：2026-07-24

已完成事项：
- bit 单位契约、发送/接收逻辑 payload 纯函数和整数 cycle 换算；
- 通过 `packet_scale`/`packets_in_last_group` 精确恢复 NoC 聚合前的真实 payload，并保持既有模拟包数；
- DTE 按核配置字段、全局启动延迟配置和默认关闭开关；
- pending → active channel → shared bus RR → completion 状态机；
- 地址稳定的 transfer context、独立完成事件和显式 Release；
- pending/launch/bus_wait/transmit 四阶段 trace；
- SystemC 自测、独立 Python oracle、配置样例和统一测试运行器；
- 自测模块全部使用 `std::unique_ptr` 管理所有权，两个配置样例由统一运行器实际加载校验；
- COMET 40 μs/2 μs 默认值的页码、章节、引用链和非 TX8 适用边界。

实际交付文件：
- llm/include/dte/dte_types.h
- llm/include/dte/dte_payload.h
- llm/include/dte/dte_unit.h
- llm/include/prims/norm_prims.h
- llm/include/utils/msg_utils.h
- llm/src/dte/dte_unit.cpp
- llm/src/dte/v0_selftest.cpp
- llm/src/prims/norm_prims/send_prim.cpp
- llm/src/utils/msg_utils.cpp
- llm/src/monitor/config_helper_core.cpp
- llm/src/monitor/config_helper_pd.cpp
- llm/src/monitor/config_helper_pds.cpp
- llm/test/dte/oracle.py
- llm/test/dte/run_test_dte_v0.py
- llm/test/dte/hardware/v0.json
- llm/test/dte/simulation/v0.json

执行的测试：
- cmake --build build --target npusim -j2：通过；
- ./build/npusim --dte-v0-selftest：49/49 通过（含倍率 20、100 个原始包、尾组和 wire round-trip）；
- python3 llm/test/dte/run_test_dte_v0.py：示例配置加载、oracle、SystemC 均通过；
- ./build/npusim --d2d-v0-selftest：307/307 通过；
- 既有 gemm_no_congestion workload：完成于 29109 ns，与基线一致；
- JSON 格式与 git diff --check：通过。

与计划的差异：
- behavioral 接收 payload 不扩展 Msg wire 字段，使用已有
  (roofline_packets-1)*M_D_DATA+length 恢复精确 bit 数；
- bus wait 使用 active channel 状态扫描而非额外物理 deque，外部语义相同；
- 为保证同 delta cycle 的四阶段 trace 不丢事件，Event_engine 改为一次唤醒排空队列；
- `SEND_DATA` 使用原空闲 bit 36..51 保存模拟包聚合元数据，旧编码全零按倍率 1 兼容。

遗留问题：
- COMET 第 8 页 §VI-A 明确采用 40 μs/2 μs，并沿用参考文献 [38]：
  T. Jose and D. Shankar, “Performance modeling of a heterogeneous computing system based on the UCIe interconnect architecture,”
  *IEEE Space Computing Conference (SCC)*, 2023, pp. 5–10。它们是可配置的 COMET baseline
  评估默认值，不是 TX8 实测值或 per-core DTE 固有参数。
- V0 不接入 WorkerCore workload；SEND/RECV 阻塞式接入属于 V1。

是否满足进入下一版条件：
- 是。V0 独立资源模型、配置、trace、oracle 和回归验收均已通过；
  V1 可在上述 baseline 默认值可由配置覆盖的前提下开始。
```

### V1 完成记录

```text
版本：V1
完成日期：2026-07-24

已完成事项：
- 每个 WorkerCoreExecutor 创建独立 DTEUnit，并读取本核 channel/位宽配置；
- SEND_DATA 在进入 NoC 前等待 SPM_TO_REMOTE，RECV_DATA 聚齐所有 source/stripe 后等待 REMOTE_TO_SPM；
- SEND_REQ/REQUEST 传递精确逻辑 payload 元数据，控制消息本身不支付 DTE 延迟；
- V1 交付时对 parallel 未支持组合增加启动期拒绝；该全局限制已由 V2a 的 dataflow parallel 支持取代；
- DTE off 的 REQUEST 构造不触发 DTE payload 计算或额外合法性校验；
- 增加 physical/behavioral、on/off、跨 die、stripe=1/2/4、多 source、重复 source/tag 生命周期、异构 per-core 配置和 trace 验收。

实际交付文件：
- llm/include/common/msg.h
- llm/include/dte/dte_payload.h
- llm/include/workercore/workercore.h
- llm/src/workercore/workercore.cpp
- llm/src/workercore/logic.cpp
- llm/src/utils/msg_utils.cpp
- llm/src/monitor/config_helper_core.cpp
- llm/src/monitor/config_helper_pd.cpp
- llm/src/monitor/config_helper_pds.cpp
- llm/test/dte/run_test_dte_v1.py
- llm/test/dte/hardware/v1*.json
- llm/test/dte/simulation/v1*.json
- llm/test/dte/workload/v1*.json
- notes/extensions/DTE/log/V1_development.md

执行的测试：
- DTE selftest：55/55；D2D selftest：308/308；
- DTE V1 集成矩阵：13/13 全部通过；
- NoC frozen baseline：4/4；
- D2D V0/V3/V4/V5 runner：67/67、16/16、13/13、23/23。

与计划的差异：
- 接收端不从 behavioral DATA 代表包反推整个多源 flow，而由每个 stripe REQUEST 重复携带同一 64-bit payload 声明；
  目的端按 (source,tag) 去重并在所有预期 source 完成后求和，避免代表包、条带和包压缩造成少算。
- V1 仍是 store-and-forward；未接入 send_para_logic()，并发 admission 属于 V2a。

遗留问题：
- V1 交付时不支持 DTE 与 send_recv_parallel 同开；V2a 后 dataflow parallel 已开放；
- V1 不模拟 source DTE、NoC/D2D、destination DTE 的流水重叠；
- channel_count 在 V1 单传输阻塞 workload 中只影响资源上限，不改变单传输延迟；其 workload 并发效果已由 V2a 验收。
- 重复 flow 用例覆盖 V1 阻塞式 pipeline/refill 生命周期；V2a 已用 parallel refill 重新验证
  per-key round queue、共享 context 和新 xfer id 的生命周期。

是否满足进入下一版条件：
- 是。此处记录 V1 交付时结论；连续 SEND_DATA 并发接入现已由后续 V2a 完成记录验收。
```

### V2a 完成记录

```text
版本：V2a
完成日期：2026-07-24

已完成事项：
- send_para_queue 批次中的全部 SEND_DATA 背靠背 issue DTE；
- 每条 SEND 独立记录 xfer_id/context，完成后才能按原语顺序提交 DATA；
- channel_count 控制 active/launch overlap，共享 bus 位宽保持不变；
- REQUEST metadata 使用 per-(source,tag) 的轮次队列和 stripe mask，支持跨轮提前到达；
- dataflow parallel 正式开放，非 dataflow、parallel stripe 和同核双向 DTE 保持明确门禁。

独立附带修复（非 DTE 功能）：
- 修复 shared `send_para_logic()` 的既有 fan-out 尾包 token 死锁；
- 该修复不受 `SPEC_USE_BEHA_DTE` 门控，会改变 DTE-off parallel fan-out 的行为；
- 修复前多 SEND 会永久等待，修复后四 SEND 用例于 1073 ns 正常完成；
- `parallel DTE-off compatibility` 明确验证无 DTE span，说明变化来自 shared 路径修复而非
  DTE 模型误启用。

实际交付文件：
- llm/include/workercore/workercore.h
- llm/src/workercore/logic.cpp
- llm/src/utils/config_utils.cpp
- llm/src/dte/v0_selftest.cpp
- llm/test/dte/run_test_dte_v1.py
- llm/test/dte/run_test_dte_v2.py
- llm/test/dte/hardware/v2_channel1.json
- llm/test/dte/hardware/v2_channel2.json
- llm/test/dte/hardware/v2_channel4.json
- llm/test/dte/simulation/v2_parallel_default.json
- llm/test/dte/simulation/v2_parallel_off.json
- llm/test/dte/simulation/v2_parallel_on.json
- llm/test/dte/workload/v2_parallel_one.json
- llm/test/dte/workload/v2_parallel_two.json
- llm/test/dte/workload/v2_parallel_four.json
- llm/test/dte/workload/v2_parallel_mixed_same_dest.json
- notes/extensions/DTE/log/V2a_development.md

执行的测试：
- DTE selftest：55/55；
- DTE V1 兼容矩阵：13/13；
- DTE V2a 验收矩阵：15/15；
- D2D selftest：308/308；D2D V0：67/67；D2D V5：23/23；
- NoC frozen baseline：4/4。

与计划的差异：
- 实际 send_para_queue 批次由 SEND_REQ/RECV_ACK/SEND_DATA 交错组成，因此在批次开始时
  预扫描所有 DATA 并背靠背 issue，而不是要求 DATA 指针在队列中物理相邻；控制原语执行顺序不变。
- channel 增多显著缩短源 DTE 批次完成时间，但本测试中控制握手/NoC 位于关键路径，
  因而最终 DONE 可保持相同；channel 可观测性以精确 DTE trace 为准。
- refill 测试发现第二轮 REQUEST 可在第一轮 RECV 消费前到达，单值 map 不足；已升级为轮次队列。

遗留问题：
- parallel stripe>1 尚未实现，运行时明确拒绝；
- 同一 core 的 SEND/RECV 双向 DTE 并发尚未实现，运行时明确拒绝；
- V2a 不实现 source DTE、NoC/D2D、destination DTE 的 streaming overlap；该项属于 V2b。

是否满足进入下一版条件：
- 是。V2a 的 channel admission、共享 bus、context 生命周期、有序提交、refill 和回归验收均通过。
```

### V2b 完成记录

```text
版本：V2b
完成日期：2026-07-24

已完成事项：
- 新增 dte.streaming 显式开关，默认 false，保留 V1/V2a store-and-forward 基线；
- source DTE 首单元 ready 后放行 DATA，physical packet 按累计逻辑 bit 做 readiness gate；
- destination 在每个 source 首 DATA 到达时 issue 独立 REMOTE_TO_SPM context；
- 用 source-tail 投影、实际 network tail、destination DTE tail 的最大值组合最终写入；
- physical/behavioral NoC 与 behavioral D2D 使用一致的首包/末包契约；
- trace 输出 source_fill、network、destination 三段 B/E，可直接观察重叠；
- 完成 RECV 并发需求评估，当前 workload 未触发，保持 dispatcher 不变并明确门禁。

实际交付文件：
- llm/include/dte/dte_streaming.h
- llm/include/dte/dte_types.h
- llm/include/common/msg.h
- llm/include/defs/spec.h
- llm/src/dte/dte_unit.cpp
- llm/src/die/d2d_link.cpp
- llm/src/defs/spec.cpp
- llm/src/utils/config_utils.cpp
- llm/src/utils/msg_utils.cpp
- llm/src/workercore/logic.cpp
- llm/src/dte/v0_selftest.cpp
- llm/test/dte/oracle.py
- llm/test/dte/run_test_dte_v2b.py
- llm/test/dte/hardware/v2b*.json
- llm/test/dte/simulation/v2b*.json
- llm/test/dte/workload/v2b*.json
- notes/extensions/DTE/log/V2b_development.md

执行的测试：
- DTE V2b 集成矩阵：18/18；DTE selftest：64/64；
- DTE V1/V2a：13/13、15/15；
- D2D selftest：308/308；D2D V0/V4/V5：67/67、13/13、23/23；
- NoC frozen baseline：4/4。

与计划的差异：
- 采用“显式首单元事件 + 流级闭式尾部”，而不是为每个真实 bit/flit 新建 SystemC 事件；
- DATA 的模拟 payload 位承载两个 48-bit 绝对 ns 时间戳和一个 32-bit behavioral tail；
- 未实现可选 RECV dispatcher 并发，因为真实 workload 未证明需求，且贸然复用 send_para_queue
  会破坏原语、buffer 和 prim_block 所有权。

遗留问题：
- streaming 与 send_recv_parallel 当前明确互斥；
- 多 RECV 原语并发仍是条件性后续版本；
- 模型是 flow-level timing approximation，不模拟逐 flit SPM backpressure；
- 此处为 V2b 交付时状态；compute/DMA issue-poll 后由 V3a 完成，aggregation/coalescing 后由 V3b 完成。

是否满足进入下一版条件：
- 是。V2b 流水公式、瓶颈迁移、backend、一致性 trace 和回归均通过。
```

### V3a 完成记录

```text
版本：V3a
完成日期：2026-07-24

已完成事项：
- 新增 Dte_async issue/wait/poll/fence/cancel primitive 和 2×128-bit wire；
- 新增 per-core DteAsyncTracker，分离 logical token 与 physical xfer_id；
- 支持多个 outstanding、选择性 wait、顺序 fence、pending cancel 和 token 复用；
- 以 SPM byte 半开区间阻止 RAW/WAR/WAW，允许 read/read 并发；
- 增加专用 WorkerCore dispatcher，普通 Comp_prim 可与后台 DTE overlap；
- queue drain/SEND_DONE 检查未消费 token，异常路径保持 context 可回收；
- 新增 async 配置及 use_beha_dte/dataflow/sequential/non-streaming 组合门禁。

实际交付文件：
- llm/include/dte/dte_async_types.h
- llm/include/dte/dte_async.h
- llm/src/dte/dte_async.cpp
- llm/include/dte/dte_types.h
- llm/include/dte/dte_unit.h
- llm/src/dte/dte_unit.cpp
- llm/include/prims/norm_prims.h
- llm/src/prims/norm_prims/dte_async_prim.cpp
- llm/include/workercore/workercore.h
- llm/src/workercore/workercore.cpp
- llm/src/common/config.cpp
- llm/src/monitor/config_helper_core.cpp
- llm/include/defs/spec.h
- llm/src/defs/spec.cpp
- llm/src/utils/config_utils.cpp
- llm/src/dte/v3_selftest.cpp
- llm/unittest/npusim.cpp
- llm/test/dte/run_test_dte_v3.py
- llm/test/dte/simulation/v3_async_on.json
- llm/test/dte/simulation/v3_async_off.json
- llm/test/dte/workload/v3_*.json
- notes/extensions/DTE/log/V3a_development.md

执行的测试：
- DTE V3 selftest：31/31；
- DTE V3 WorkerCore 集成矩阵：14/14；
- DTE V0 selftest：64/64；DTE V0 runner：PASS；
- DTE V1/V2a/V2b：13/13、15/15、18/18；
- D2D selftest：308/308；D2D V4/V5：13/13、23/23。

与计划的差异：
- 依赖用独立 WAIT primitive 表达，不修改每种计算 primitive 的 wire；
- poll 提供真实运行态结果和 trace，但暂不支持 workload 条件分支；
- descriptor 只计 endpoint DTE service，真实 NoC/D2D 消息仍由 SEND/RECV 负责；
- cancel 限定为 pending-only，active cancel 明确拒绝。

遗留问题：
- 此处为 V3a 交付时遗留项；COMET aggregation/coalescing、地址连续性窗口和 completion fan-out 现已由 V3b 完成；
- 更多传输方向和更精细 AXI/SPM 端口属于 V4。

是否满足进入下一版条件：
- 是。V3a 异步语义、依赖、冒险、生命周期、trace、配置门禁和历史回归均通过。
- 独立评审逐项复核 dispatcher、hazard、cancel/release 和 endpoint 范围边界，未发现问题。
```

### V3b 完成记录

```text
版本：V3b
完成日期：2026-07-24

已完成事项：
- 在 V3a logical-token 层实现可配置在线 compound descriptor 聚合；
- 增加 remote peer/address/address block 元数据和 2/3-segment 向后兼容 wire；
- 同方向、peer、block 且双端连续的请求可聚合，limit/incompatible/timeout/dependency/fence 可 flush；
- 一次 physical DTE completion fan-out 到所有 logical token，最后一个 token 消费后释放 context；
- 增加 collect/flush/bind trace，以及 launch savings 和 useful-bit bus utilization 指标；
- 根据独立评审将 flush trace 累计指标明确命名为 cumulative_utilization_ppm；
- 用固定总数据量的 1/2/4/8/16 请求分组微基准复现 COMET Figure 11 的 launch 摊薄趋势。

实际交付文件：
- llm/include/dte/dte_coalescing.h
- llm/include/dte/dte_async_types.h
- llm/include/dte/dte_async.h
- llm/src/dte/dte_async.cpp
- llm/include/prims/norm_prims.h
- llm/src/prims/norm_prims/dte_async_prim.cpp
- llm/include/defs/spec.h
- llm/src/defs/spec.cpp
- llm/src/utils/config_utils.cpp
- llm/src/workercore/workercore.cpp
- llm/src/dte/v3_selftest.cpp
- llm/src/dte/v3b_selftest.cpp
- llm/unittest/npusim.cpp
- llm/test/dte/oracle.py
- llm/test/dte/run_test_dte_v3b.py
- llm/test/dte/hardware/v3b.json
- llm/test/dte/simulation/v3b_*.json
- llm/test/dte/workload/v3b_*.json
- notes/extensions/DTE/log/V3b_development.md

执行的测试：
- DTE V0 selftest：64/64；V3a/V3b selftest：31/31、21/21；
- DTE V1/V2a/V2b/V3a/V3b：13/13、15/15、18/18、14/14、16/16；
- Python oracle self-test：PASS；
- D2D selftest：308/308；D2D V4/V5：13/13、23/23。

与计划的差异：
- COMET 采用 task-DAG 与遗传搜索的离线地址映射；仓库没有该输入和 mapper，因此实现为 workload 显式提供 block 的确定性在线连续区间聚合；
- 复现式 (13)–(15) 的 address-block 可聚合条件和 compound launch 摊薄，不复现式 (16)–(17) 的跨 block 路由倍率；
- 未增加 stride：首版只接受 byte 对齐、payload 与本地 SPM range 等长的连续描述符；
- V3a 无远端字段时仍编码 2×128 bit，只有 V3b 元数据存在时才使用第三 segment，保持冻结时序。

遗留问题：
- Dte_async 仍是 endpoint service，不生成 SEND/RECV 的 NoC/D2D 数据；
- staged group 只允许取消尾成员；非尾 staged cancel 与已 issue compound 的部分 cancel 明确拒绝；
- COMET 完整离线 mapper、route-aware scaling 与数据组织（stride/scatter/broadcast）按范围暂缓；精细 AXI/SPM endpoint 端口已由 V4 完成。

是否满足进入下一版条件：
- 是。聚合规则、completion fan-out、等待/收益计费、论文趋势和历史回归均通过。
```

### V4 完成记录

```text
版本：V4
完成日期：2026-07-24

已完成事项：
- 新增 SPM→SPM、SPM→DRAM、DRAM→SPM、DDR→remoteTile 四个方向；
- 新增 SPM read/write、AXI read/write 四类独立 endpoint 资源和逐端口闭式服务时间；
- 每 channel 固定两个 command slot，增加有限 pending queue 和 WaitForCredit issue backpressure；
- 扩展异步 descriptor 的读写区间与 RAW/WAR/WAW 检查；
- 增加方向/端口统计、动态能耗、平均功耗和面积模型；
- 明确 DTE endpoint 与 DRAM、NoC、D2D 的不重复计费边界；
- scatter、broadcast、stride/slice/shuffle 按用户决策暂缓，不属于本版完成条件。

实际交付文件：
- llm/include/dte/dte_types.h
- llm/include/dte/dte_unit.h
- llm/include/dte/dte_async_types.h
- llm/include/dte/dte_async.h
- llm/src/dte/dte_unit.cpp
- llm/src/dte/dte_async.cpp
- llm/src/dte/v4_selftest.cpp
- llm/src/prims/norm_prims/dte_async_prim.cpp
- llm/include/defs/spec.h
- llm/src/defs/spec.cpp
- llm/src/utils/config_utils.cpp
- llm/src/workercore/workercore.cpp
- llm/unittest/npusim.cpp
- llm/test/dte/oracle.py
- llm/test/dte/run_test_dte_v4.py
- llm/test/dte/hardware/v4*.json
- llm/test/dte/simulation/v4*.json
- llm/test/dte/workload/v4_*.json（含 v4_mixed_send_recv_credit.json）
- notes/extensions/DTE/log/V4_development.md

执行的测试：
- DTE V4 selftest：19/19；
- DTE V4 WorkerCore 集成矩阵：18/18；
- Python oracle self-test：PASS；
- 历史 DTE WorkerCore：V1/V2a/V2b/V3a/V3b 为 13/13、15/15、18/18、14/14、16/16；
- 历史 DTE selftest：V0/V3a/V3b 为 64/64、31/31、21/21；
- D2D shared-path selftest 308/308，D2D V0/V4/V5 为 67/67、13/13、23/23；
- V4-off V3a overlap 冻结为 345 ns；完整命令和证据见 V4 开发记录。

与计划的差异：
- 数据组织不在本轮实现，因此未交付 scatter、broadcast、stride/slice/shuffle；
- 四类端口建模为 endpoint 服务，不调用真实 DRAM/NoC/D2D 模块，避免双重计费；
- DDR→remoteTile 在代码中使用覆盖 DRAM/DDR 来源的方向名 DRAM_TO_REMOTE；
- 两条命令是每 channel 固定硬件常数，仅 channel_count、端口位宽和队列深度可配置探索。

遗留问题：
- 仅有已明确暂缓的数据组织；
- COMET 完整离线 mapper 和 route-aware scaling 不属于本 endpoint 模型；
- 当前 workload 不需要多 RECV_DATA dispatcher 并发，故保持既有 dispatcher 与门禁。

评审闭环（2026-07-25）：
- 四个 blocking SEND/RECV `DTEUnit::Issue()` 调用前统一执行 `WaitForCredit()`；legacy 无界容量下立即返回；
- 新增 mixed async + SEND/RECV workload，同时占满源核与目的核 descriptor 容量；
- 源 blocking SEND 在 2236 ns、目的 blocking RECV 在 8344 ns 获得 credit 后正常 issue；
- 两核均为 issued/completed=4、backpressure_stalls=1，完整 workload 于 24742 ns 完成；
- V4 WorkerCore 矩阵由 17/17 增至 18/18。

是否满足 DTE 收尾条件：
- 是。除明确暂缓的数据组织外，计划中的方向、资源、backpressure、功耗/面积和回归均完成。
```

## 12. 最终完成定义

DTE 建模任务在本轮批准范围内已完成并收尾：

- V0a、V0b、V1、V2a、V2b、V3a、V3b、V4 全部验收通过；
- 六个 endpoint 方向、异步 token、在线 coalescing、流式路径、独立 SPM/AXI 端口、双命令槽、有限 credit/backpressure、功耗与面积均已自动化验证；
- `channel_count`、共享/分端口位宽、聚合规模和 pending 深度可用于有意义的架构探索；
- bit/byte、地址区间、带宽归属、端到端计费、compound completion 和复合端口完成语义不存在歧义；
- trace 能解释 pending、launch、端口等待/服务、网络重叠、logical→physical 聚合、credit stall 和能耗统计；
- DTE endpoint 不重复计算 DRAM media、NoC hop 或 D2D link 服务；
- 默认关闭 V4 或全部 DTE 功能时，历史行为和冻结时序不变；
- 数据组织（scatter、broadcast、stride/slice/shuffle）按本轮要求暂缓，不影响 DTE 主体收尾；
- 多 `RECV_DATA` dispatcher 并发仅在未来真实 workload 出现需求时再设计，不作为当前阻塞项；
- COMET 完整离线 mapper/route-aware 模型与已实现的在线 endpoint approximation 保持明确区分。

后续若重新纳入数据组织，应作为独立扩展重新定义 descriptor wire、地址生成、多 completion、
buffer/credit 和验证契约；不应改变本次已冻结的 V0–V4 endpoint 语义。
