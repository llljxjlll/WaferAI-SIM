# NoC 集合通信重构 R3 开发记录

日期：2026-08-03

## 1. 范围与隔离边界

R3 只冻结 Router-facing 的流式归约协议和有限状态契约，不接入 production Router、
endpoint session 或配置展开。新增实现只有：

- `llm/include/dte/coll_reduce_stream.h`
- `llm/src/dte/coll_reduce_stream.cpp`
- `llm/src/dte/coll_r3_selftest.cpp`
- `llm/test/noc_collective/r3_oracle.py`
- `llm/test/noc_collective/run_test_coll_r3.py`

`coll_reduce_stream.h` 没有被 Router、workercore 或 config helper include；现有
`coll_innetwork_reduce`、V4/V5/V6 production datapath 和 gate 均未改变。R4 才负责把本
阶段契约接入真实 Router credit/output arbitration 和 R2 `DcaComputePool`。

## 2. Binary pairwise stage

`BuildBinaryReduceSchedule(N)` 将任意 fan-in 展开为确定性 left fold：

- N=1：bypass，stage=0，DCA issue=0；
- stage 0：`network[0] + network[1]`；
- stage i>0：`feedback[i-1] + network[i+1]`；
- 最后一个 stage 标记 `final_output`。

因此每个 DCA request 始终只有两个 operand，N 个输入、B 个 vector beats 的 issue 数为
`(N-1)×B`。自测冻结 1/2/3/5 输入对应 0/1/2/4 stages，并检查 local feedback 索引和
final 标记。

## 3. Stream wire v2

R3 使用独立的 256-bit header/data wire，magic 为 `0xc6e3`，版本直接使用 R0 冻结的
唯一 `ReduceWireVersion::STREAM_V2=1`。

Header 位域：

| bits | 字段 |
|---|---|
| 15:0 / 19:16 / 23:20 | magic / version / HEADER segment |
| 39:24 | tree id |
| 63:40 / 87:64 / 111:88 | group / collective / epoch（各 24 bit） |
| 123:112 / 131:124 | phase / reduce stage |
| 155:132 / 167:156 | stream id / source id |
| 170:168 / 172:171 | dtype / reduce op |
| 196:173 | total elements |
| 212:197 / 228:213 | physical flits / vector beats |
| 236:229 | tail valid lanes |
| 255:237 | reserved=0 |

Data 位域：payload `[127:0]`；magic/version/DATA segment `[151:128]`；compact route
由 stream id、tree id、source id、stage id 组成；另带 16-bit seq、8-bit length、tail，
reserved 必须为零。codec 在编码前拒绝范围溢出和非零 unused tail payload，在解码后
复核 magic/version/segment/reserved、枚举、计数和 tail geometry。

同一 active route 不允许重复 header；`ValidateReduceWireCompatibility` 明确拒绝同一
collective 混用 legacy two-segment 与 stream-v2。数据 wire 不重复完整 CollectiveKey，
而是通过 active header 中的 compact route 查找 context；compact route 冲突在打开
第二个 header 时确定性拒绝。

## 4. 4:1 assembler 与 splitter

本阶段冻结 128-bit physical payload 和 512-bit DCA vector：

- assembler 按严格连续 seq 收四个 physical flit 形成一个 vector beat；
- 最后不足四个 slice 时补零，但 lane mask 只激活真实 element；
- ready queue 满时，在消费会产生 beat 的 data 前返回 `BACKPRESSURE`，seq 和 fragment
  状态不改变；
- splitter 按 vector beat id 恢复全局 seq，只生成真实存在的 F 个 flit，并清零最后
  `length_bits` 以外的 payload；
- `Finish()` 对截断流报错，完成后再来的 data 报 extra/duplicate 错误。

70 个 UINT8 element 对应 F=5、B=2、64 lanes，末 beat 有效 6 lanes，最终 physical
flit 为 48 bits。该非整齐尾部在 assembler、splitter 和 wire round-trip 三处交叉验证。

## 5. 有限状态与可恢复背压

`ReduceStreamFiniteState` 对以下结构分别设置显式容量：

- header context 与每流 assembler-ready；
- operand values/matches；
- issue queue、inflight tag table、result queue；
- network input 与 local-feedback input queue。

容量满返回 false/`BACKPRESSURE`，不会部分修改状态。协议损坏仍抛异常，包括未知 route、
seq/tail/key/dtype/geometry mismatch、重复 operand slot、重复 live tag 和未知 completion。

operand FIFO 使用 head-of-line progress reservation：新 key 不得占用最后一个槽，已打开
pair 的第二个 operand 可以使用该槽完成配对。这样即使多个 first operand 竞争，也始终
至少有一个 pair 能完成并前移到 issue queue，且 occupancy 不超过配置容量。

network 与 local feedback 使用 round-robin 选择；result 转入 feedback 是原子的，反馈
FIFO 满时 result 保留在 result queue。`Occupancy/Residual/Drained` 覆盖 header、fragment、
ready beat、operand、match、issue、inflight、result 和两个输入队列。

## 6. 测试结果

R3：

- `npusim --coll-r3-selftest`：39/39；
- `r3_oracle.py`：PASS；
- `run_test_coll_r3.py`：2/2。

冻结 collective 回归：

- R0/R1/R2 selftest：19/19、42/42、37/37；runner：2/2、16/16、2/2；
- V0～V6 selftest 全部通过，V1 runner 24/24、V4/V5 runner 9/9、V6 runner 6/6。

共享路径回归：

- NoC congestion：4/4，冻结值 14781/29109 与 14833/45441 ns；
- D2D V0：67/67 test groups；
- `git diff --check` 通过。

## 7. 已知边界

- R3 仍是隔离 contract/state model，不产生 production collective trace；
- R3 不接真实 Router input/output/credit，也不调用 R2 ComputePool；
- header/data 的生产仲裁、异步 TX/RX progress 和 result reinjection 属于 R4；
- core/FPU 真实共享、FP timing/value 语义属于后续阶段；
- 本阶段只冻结计划要求的 128→512 bit 4:1 数据路径，更宽配置若需要扩大 wire 字段，必须
  在接入 production 前另行修订并增加版本。
