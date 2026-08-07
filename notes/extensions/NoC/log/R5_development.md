# NoC 集合通信重构 R5 开发记录

日期：2026-08-03

## 阶段定位

R4 仍可被误读为“Router 内有一颗免费专用 reduce ALU”。R5 按 FPnew 的资源共享思路把
endpoint CORE vector request 和 Router DCA request 接入同一个 per-tile
`DcaComputePool`，并冻结整数/浮点的 value-mode 边界。目标是让通信加速的收益同时体现
对 tile 计算资源的占用，而不是只修改一个闭式延迟公式。

## 开发过程

### 1. 单一 per-tile compute pool

- CORE 与 DCA 共用有限 pending、单 issue port、inflight、result FIFO 和同一 dtype/op 的
  latency/II 表。
- request 携带 source、pool tag 和完整 reduce beat key；completion 按 source 返回 endpoint
  continuation 或 Router stream stage，禁止只用短 tag 区分。
- 仲裁支持 `round_robin`、`core_priority`、`dca_priority`。统计分别记录 core/dca issue、
  wait、submit stall 和 completion，避免总利用率掩盖一方饥饿。
- production `core_contention_beats` 建立 endpoint vector session，请求真实进入与 DCA 相同
  的 pool，完成后才释放 endpoint token；它不是 runner 直接修改统计值。

### 2. vector-lane 计时

工作量以 `B=ceil(count/lanes)` 个 vector beats 和每 beat 的 `inputs-1` 次 pairwise issue
计算。DCA 处理的是多 lane SIMD beat，不是逐 element 串行累加。一个请求的完成周期为
`issue+L`，连续请求的最早 issue 间隔为 II，因此无额外争用的流水时间是
`L+(issues-1)*II`。

CORE 和 DCA 从同一配置读取 `vector_bits`、dtype lane 数、L/II。这样改变 vector width 或
dtype 会同时改变双方吞吐，不会出现 Router 使用 128-bit、endpoint 使用另一套 lane 公式。

### 3. value mode 与 FP32 exact

- `integer_exact`：UINT8/INT32/INT64 按 dtype 位宽 mask，SUM 使用模位宽结果，MAX 按
  signed/unsigned 语义比较。
- `fp_exact`：当前只开放 FP32。值用 `memcpy` 位安全转换，固定 binary-stage 结合顺序；SUM
  和 MAX 冻结 NaN、Inf、signed zero。NaN 统一传播 canonical quiet-NaN `0x7fc00000`，
  MAX 的 `-0/+0` tie 返回 `+0`。
- `timing_only`：FP32/FP16/FP8 可使用相应 wire、lane geometry 和 L/II，但 completion 不带
  可断言数值；向 timing-only request 填 value payload 会被拒绝。

FP16/FP8 没有借 host `float` 冒充精确格式，因而不会把未冻结的 rounding/NaN 规则包装成
bit-exact 结果。

### 4. 可观测性与资源闭合

`COLL_DCA` 分开输出 `core_issues`、`dca_issues`、`completions`、两类 stalls、submit
stalls 和 peak inflight。pool tag 在 drain 时必须全部回收；endpoint token、stream
continuation 和 result FIFO 同时为零才允许 collective 完成。

## 实现中重点检查的问题

1. **tag 串流**：CORE/DCA 使用不同 source namespace，且 completion 还核对完整 beat key；
   相同短序号不能领取对方结果。
2. **优先级饥饿**：配置优先级是显式实验选项，默认 round-robin；统计暴露每方 wait，不能
   把等待隐藏进“DCA 固定延迟”。
3. **host FP 未定义行为**：位转换使用 `memcpy`，NaN/zero 先按 bit pattern 处理；不会用
   strict-aliasing 不安全的指针转换。
4. **FP 结合顺序**：沿 R3 fixed binary stage 重复运行，禁止容器遍历顺序决定结果。
5. **inactive DCA 配置**：baseline/disabled 不实例化 DCA 时，不因未使用的 DCA 块启动失败；
   只有实际 DCA backend 才做资源/值模式校验。

## 验收证据

- `npusim --coll-r5-selftest`：10/10。覆盖 CORE/DCA 共享池、仲裁身份、FP32 SUM/MAX、
  NaN/Inf/signed-zero、重复 bit-identical、timing-only 拒绝 value assertion 和 dtype gate。
- production contention：同一 pool 接收 CORE/DCA 12/9 issues，21/21 completions；有限
  队列产生非零 stall，所有 tag 和 token 最终回收。
- FP32 `fp_exact` 与 `timing_only` 均端到端通过；FP16/FP8 `timing_only` 通过 wire、lane、
  L/II 和 drain 验证。
- 2026-08-03 评审补强后的 R5～R7 runner 为 19/19，其中新增真实 regular unicast 与
  stream-DCA production 混合流量，证明共享 Router path 不破坏 DCA value/completion。

## 已知限制

- FP32 exact 是确定性的仿真合同，并非完整复刻某个 FPnew RTL 配置的异常标志、subnormal
  flush、可选 rounding mode 或 fused 运算。
- FP16/FP8 仅 timing-only；在格式、rounding 和 NaN 规则冻结前不提供 bit-exact 值。
- `core_contention_beats` 是明确的 endpoint vector-resource 负载模型；尚未自动把仿真器中
  每一种普通 compute primitive 的所有内部周期映射为 pool request。
- 模型统计资源竞争和时序，不输出面积、功耗或 FPnew physical implementation 参数。

R5 退出条件：DCA 被建模为借用 tile 向量资源；CORE/DCA 的 lanes、L/II、仲裁和完成身份可
由同一组配置与 trace 解释。
