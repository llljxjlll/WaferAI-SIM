# NoC 集合通信重构 R4 开发记录

日期：2026-08-03

## 阶段定位

R4 的目标不是再定义一套 reduce 合同，而是把 R3 已冻结的 `STREAM_V2` wire、有限状态机
和两输入 SIMD DCA request 接到 production Router。完成标准是数据确实经过 Router input、
已有 output buffer/credit/arbitration 和 endpoint session；不能由 runner 直接调用合同模型得出
结果。

进入 R4 时已有的基线是：R0 固定向量 beat/issue/L/II 语义，R1 固定四 profile，R2 提供
有限 `DcaComputePool`，R3 提供每 stream 一个 header 加若干 data flit 的协议及背压状态机。
R4 保持这些合同不变，只做生产路径接线和 progress 生命周期。

## 开发过程

### 1. Router 接入门控

- 仅在 `SPEC_NOC_COLL_ENABLED && UsesDcaOffload()` 时为 Router 创建
  `RouterReduceStreamEngine`。其余配置继续走原普通 `Msg` 路径。
- Router input 先以完整 magic/version/segment/reserved 组合识别 stream header/data；普通
  DATA 即使低位碰撞，也不会进入集合 wire decoder。
- reduce tree 的 expected inputs 和 parent output 来自 workload 展开期注册表；运行时缺失、
  重复或方向不符均作为协议错误报告，禁止静默回落为 endpoint reduce。

### 2. 有限异步流水

每个 Router 的 engine 把状态拆为 header match、128-bit data assemble、512-bit vector
operand、pending issue、inflight continuation、result FIFO、local feedback/parent egress。所有
容量读取 R1 的 DCA 配置并参与背压，不以扩大 `raw_queue` 掩盖循环等待。

同一 vector beat 的多个输入按 R3 的固定 binary stage 合并；一个 DCA request 始终只有两个
operands。三输入节点每 beat 产生两个 issue，单输入节点 bypass，tail 只更新有效 lane。
result 依据完整 stream/stage/beat tag 回到后继 stage 或输出，不能因另一个 stream 完成而串流。

### 3. endpoint session 与执行顺序

root 的 `REDUCE_STREAM_RX_START` 排在任何 rank 的 TX 之前，使接收端在首个 header 到达前已
armed。TX 逐步注入而非一次把整 tensor 塞进 endpoint 队列；`RX_WAIT` 只等待 session 完成，
不阻塞 Router 的 progress 方法。root 自身输入经 CENTER 进入同一 reduce tree，所以没有额外
endpoint ALU，也不会重复计算 root operand。

输出 wire 进入既有 `buffer_o`，与 normal DATA 和 multicast 共用 output credit。collective
wire 的 level-sensitive valid 使用 cooldown/armed 机制保证一次采样一次消费；被阻塞 output
恢复后重新参与仲裁。

### 4. 生命周期和可观测性

- `COLL_STREAM_TX/RX/RESULT` 记录 stream 身份、物理 flit 和完成结果。
- `COLL_DCA` 记录 issue、completion、stall、submit backpressure、inflight peak。
- Router residual 包含 header、assembler、operand、pending、inflight、result、egress 和
  endpoint session；最终 `COLL_DRAIN` 必须全零。
- reduce-only 允许 registry 中只有 reduce node、没有 multicast entry；最终 barrier 后释放
  对应 tree。

## 实现中重点检查的问题

1. **root 同时发送与接收**：若先执行本地 TX 再建立 RX session，其他 rank 可把 header
   堵在 root input。通过 `RX_START -> TX -> RX_WAIT` 顺序消除该窗口。
2. **固定延迟重复计费**：DCA completion 是 `issue_cycle + L`，相邻 issue 由 II 间隔；长
   stream 只付一次 pipeline fill，不按 data flit 重复付 L。
3. **结果队列背压**：result FIFO 满时保留 continuation，不丢 result、不提前关闭 header；
   消费后可恢复。
4. **普通 wire 隔离**：collective 检测不依赖单一 magic，关闭 collective 时不会创建或访问
   engine。
5. **状态泄漏**：完成一个 stream 必须同时清除 match、tag、continuation 和 endpoint token，
   residual 检查用于捕捉“结果已到但 header 未关”的半完成状态。

## 验收证据

- `npusim --coll-r4-selftest`：12/12。覆盖 2/3 输入、single-input bypass、tail mask、有限
  result FIFO、错误 tag/geometry、背压恢复和全 drain。
- `python3 llm/test/noc_collective/run_test_coll_r4.py`：4/4。覆盖 2 输入 tail、三输入
  `2B` issue、`L=4/II=1`、result depth=1，以及 1024 B/64 physical flits。
- R8 的 32 KiB production run 后续复用同一路径，完成 2048 physical data flits/stream、
  全树 1536 DCA issues，未触发 watchdog。
- 冻结 NoC、D2D 和 V0～V6 回归保持不变，证明门控关闭时 Router 普通路径没有行为漂移。

## 已知限制

- 只建模 conventional 同 die Router；OpenSMART 单周期多跳和跨 die collective 不在 R4
  范围。
- stream 数据面是功能/时序模型，不建模 RTL 级 crossbar 组合路径或功耗面积。
- R4 只建立 Router DCA production path；CORE 与 DCA 共享 tile 向量资源在 R5 接入。
- 公平性是有限队列下的 round-robin/配置仲裁，不承诺任意外部持续注入下的实时延迟上界。

R4 退出条件：新 stream 真正穿过 production Router，长 payload 不依赖扩大 endpoint queue，
completion 后所有集合状态可证明清零。
