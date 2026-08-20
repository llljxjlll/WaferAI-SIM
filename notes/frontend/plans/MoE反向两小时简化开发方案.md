# MoE 反向两小时简化开发方案

## 1. 目标与完成定义

在 2 小时内，基于已经跑通的 S3-Lite 两 Die 静态 MoE 推理，以及已经跑通的
FP32 WGRAD、`LOCAL_REDUCE`、SGD 和 DTE 通信路径，实现一个真实可在 `npusim`
执行的 MoE backward microprogram。

本阶段的准确名称是：

> **S3-Lite static-route expert down-projection WGRAD-only backward**

完成必须同时满足：

- 两 Die、EP=2、4 experts、Top-K=1；
- 沿用现有 8-token balanced `STATIC_TRACE`；
- upstream gradient 位于 token source die；
- remote token 的 gradient 真实经过跨 Die DTE 发送到 expert home die；
- 每个 token 执行一次专家 down-projection WGRAD；
- 每个 expert 的两个 token WGRAD 使用 FP32 `LOCAL_REDUCE SUM` 合并；
- 每个 expert 执行一次 SGD，并将更新后的 FP16 down weight 写回 HBM；
- finalizer 两次、actual-SHA ProgramIo、resolver、`npusim` 两次全部通过；
- 两次 runtime 的 makespan 和 marker digest 完全一致；
- 所有 LSU、DTE、router、link、collective 和 control residual 为 0；
- 不出现 `PROTO_WAIT`。

## 2. 明确不做的内容

以下内容不进入本次两小时范围：

- gate/router backward；
- gate logits 或 routing probability gradient；
- SwiGLU backward；
- gate/up projection WGRAD；
- expert input DGRAD 返回 token source die；
- backbone、attention、embedding 或 LM-head backward；
- 动态 `GATE_TOPK`、Top-K>1、overflow、token drop；
- capacity reroute 或 load-balancing loss；
- expert replication、expert gradient AllReduce；
- DP、TP、PP 与 EP 的组合并行；
- 任意 mesh、任意 expert 数或多跳 All-to-All；
- AdamW、ZeRO、FSDP、gradient accumulation；
- gradient、updated weight 或 loss 的数值正确性；
- model-functional 或 routing-functional 声明；
- 正式 baseline freeze。

因此，本方案是 MoE 专家参数反向的一条可运行纵切，不等价于完整 MoE
backward，也不等价于 S3-N 完成。

## 3. 固定执行语义

对每个 token，forward 已知的静态 trace 决定 expert home。ProgramIo 提供：

- expert home 上保存的 down-projection 输入 activation；
- token source die 上的 upstream gradient；
- expert home 上的 FP16 down-projection weight。

执行顺序如下：

```text
token source die                         expert home die

upstream_grad(local token) ────────────→ WGRAD contribution

upstream_grad(remote token)
    └→ SEND → RECV → WAIT ─────────────→ WGRAD contribution

token_wgrad_0 ───────────────┐
                              ├→ FP32 LOCAL_REDUCE SUM
token_wgrad_1 ───────────────┘
                                      ↓
                              expert reduced_wgrad
                                      ↓
                              SGD_UPDATE down_weight
                                      ↓
                              HBM STORE updated_weight
```

必须满足的依赖为：

```text
saved_activation(token) ───────────────────────────┐
local_upstream_grad(token) ────────────────────────┤→ TOKEN_WGRAD
remote_upstream_grad → SEND → RECV → WAIT ─────────┘

TOKEN_WGRAD(slot0) ─┐
                     ├→ EXPERT_REDUCE → SGD → HBM_STORE
TOKEN_WGRAD(slot1) ─┘
```

禁止 SGD 读取任一 token 的局部 contribution；它只能读取该 expert 的 reduced
FP32 WGRAD。

## 4. Tiny case 与固定数据量

继续使用现有 S3-Lite tiny case：

- die count=2；
- EP=2；
- expert count=4，每个 die 放置2个 expert；
- Top-K=1；
- token count=8；
- 每个 expert 恰有2个 token；
- hidden size `H=16`；
- intermediate size `I=32`；
- activation/upstream/weight 为 FP16；
- WGRAD 与 reduction 为 FP32；
- SGD momentum=0。

精确数据量：

- 每 token saved activation：`I × 2 = 64B`；
- 每 token upstream gradient：`H × 2 = 32B`；
- remote token 数：4；
- backward gradient D2D：`4 × 32B = 128B`；
- D2D data packet：`128B / 16B = 8`；
- 每方向 remote transfer=2、data packet=4；
- 每 token WGRAD：`I × H × 4 = 2,048B`；
- 每 expert 两个 contribution：连续 `4,096B` scratch；
- 每 expert reduced WGRAD：`2,048B`；
- 4 experts reduced WGRAD 总量：`8,192B`；
- 每 expert down weight：`I × H × 2 = 1,024B`；
- updated weight HBM write：每 die `2,048B`，总计 `4,096B`；
- token WGRAD GEMM：`2 × I × H = 1,024 FLOPs/token`；
- 8 token WGRAD 总计：`8,192 FLOPs`。

这些数值必须由 typed trace、buffer shape 和实际 record 重建，runner 不得接受
caller 自报的合计值。

## 5. 最大化复用原则

### 5.1 直接复用

- 现有 `LiteMoeSpec`、balanced `LiteMoeStaticTrace` 和 expert home 映射；
- 现有 S3-Lite 两 Die placement、PairRoute、core 和 HBM placement；
- 现有 S3-Lite DTE `SEND/RECV/WAIT` lowering；
- 现有 `MATMUL` record 和 FP16-input/FP32-output WGRAD ABI；
- 已实现的 FP32→FP32→FP32 `LOCAL_REDUCE SUM`；
- 已实现的 `SGD_UPDATE` 和 trainable-state READ_WRITE closure；
- 已实现的 ALIASED buffer lifecycle；
- actual-SHA ProgramIo、finalizer、resolver、runtime 双跑框架；
- rooted-AllReduce runner 中的 memory/control/drain/repeat parser；
- S3-Lite runtime 中的两 Die D2D link parser。

### 5.2 禁止新增

- 不新增 opcode 或 PrimId；
- 不新增 DTE transport 类型；
- 不新增通用 All-to-All planner；
- 不修改 dense/Train legacy validator 来容纳 MoE；
- 不用 KV state-transfer、Residual 或 forward SwiGLU 冒充 backward；
- 不为通过 runtime 调大 channel/capacity/watchdog。

若现有 MATMUL、FP32 reduce、SGD 或 DTE 任一 public ABI 无法复用，则在 T+45m
前降级为 pre-runtime，不临时设计新 primitive。

## 6. 最小生产结构

### 6.1 Backward typed contract

新增隔离 carrier，例如：

- `LiteMoeBackwardSpec`；
- `LiteMoeBackwardOracle`；
- `LiteMoeBackwardOverlay`；
- `LiteMoeBackwardLoweredProgram`；
- `LiteMoeBackwardLinkedProgram`。

固定 case ID：

```text
case.s3_lite.static_moe_down_wgrad
```

carrier 必须嵌入现有 `LiteMoeExecutionCase`，并验证：

- trace、expert home、route、core、state 和 forward carrier provenance 完全一致；
- 每个 token 恰好一份 saved activation 和 upstream gradient；
- remote gradient flow 与现有 dispatch route 同向；
- 每 expert 恰好两个 token contribution；
- 每 expert 恰好一次 reduce、一次 SGD 和一次 HBM store；
- 四个 expert 的 scratch/state/binding ID 不相交；
- old inference carrier 不因本片发生语义漂移。

### 6.2 Backward overlay

不重新实现完整 IR0→N5。以现有 S3-Lite schedule/global carrier 为可信前置，新增
typed backward overlay：

- 4 个 remote-gradient DTE unit；
- 8 个 token WGRAD unit；
- 4 个 expert FP32 reduce unit；
- 4 个 SGD unit；
- 4 组 down-weight state load/store closure。

overlay 必须插入到一个 unified manifest 中，不能产生第二个独立 manifest，也不能
将 backward action 伪装成 forward action。

### 6.3 Buffer 与 state

每个 expert 固定：

- 两个 2,048B FP32 contribution slice；
- 一个连续 4,096B reduce input root；
- reduced output alias contribution slice0；
- 一个 1,024B FP16 trainable down-weight；
- 一个 32B FP16 local/received upstream-gradient view；
- 两个 64B FP16 saved-activation view。

`LOCAL_REDUCE` 必须使用：

```text
input_dtype       = FP32
accumulator_dtype = FP32
output_dtype      = FP32
reduce_op         = SUM
input_count       = 2
element_count     = 512
input_stride      = 2,048B
```

### 6.4 ProgramIo

ProgramIo 在 timing 模式下提供 deterministic zero payload：

- 8 个 saved activation；
- 8 个 upstream gradient；
- 4 个 down weights。

输出 probe 只覆盖4个 updated down weights。不得计算或声称数值 WGRAD expected；
不得将 zero payload 解释成 functional correctness。

## 7. Root + 3 Agent 两小时安排

build、finalizer、resolver、CTest 和 `npusim` 只允许 Root 串行启动。

| 时间 | Root | Agent A：schema/graph | Agent B：backend/trust | Agent C：ProgramIo/runner |
|---|---|---|---|---|
| 0–20m | 冻结 contract、ID、版本和文件所有权 | spec/oracle/overlay exact schema | 审计 MATMUL/reduce/SGD/DTE 复用点 | self-contained case 与 marker 合同 |
| 20–45m | 合并 exports、解决唯一 shared gate | 8 WGRAD/4 reduce/4 SGD/4 flow producer+negative | lower fragment 与 finalizer strict sequence | ProgramIo saved inputs/weight probes、mock parser |
| 45–70m | 统一 manifest/link 集成 | provenance/serde/stable-id focused | C++ trust-anchor/selftest，不新增 opcode | finalizer×2/resolver command injection focused |
| 70–85m | 冻结源码并独占 matching build | 只读 review | 分析首个编译错误 | 检查 runtime expected quotient |
| 85–110m | finalizer×2→resolver→`npusim`×2 | 解析 action/record | 核对 codec/finalizer/primitive | 核对 D2D/memory/control/drain |
| 110–120m | 只修首个真实 P0、复验、commit/handoff | focused | C++ adjacent | candidate report |

文件所有权：

- Root：公共 exports、CMake、版本 ledger、最终 linker 集成；
- Agent A：新 backward schema/pass/test，不碰 C++；
- Agent B：lowering/C++ finalizer/selftest，不碰 runner；
- Agent C：ProgramIo/runner/integration test，不碰 compute ABI；
- T+70m 后冻结源码，除真实首个 P0 外禁止扩大范围。

## 8. 两小时门禁

### 8.1 Static positive

- fixed 2 dies/EP2/4 experts/Top-K1/static trace；
- token/expert/slot 全覆盖且 canonical；
- remote token 恰好4个；
- remote backward flow 恰好4条、每条32B；
- 每 token 恰好一个 FP32 2,048B WGRAD；
- 每 expert 恰好两个 contribution；
- 每 expert reduce 输入连续、等长、无重叠；
- 每 expert SGD 只读取 reduced WGRAD；
- 每 expert updated weight 写回其 home die HBM；
- forward S3-Lite typed inference carrier exact no-drift。

### 8.2 Artifact/runtime positive

- finalizer 两次 artifact/report byte-exact；
- ProgramIo 使用 actual artifact SHA；
- resolver exact initialization/probe closure；
- `npusim` 两次 makespan、marker digest 完全相同；
- backward D2D logical bytes=128B；
- D2D data packets=8，每方向4；
- request/ACK/DATA 的 aggregate 与 directed link 互相闭合；
- 每 expert 一次 WGRAD、一次 reduce、一次 SGD；
- 总 updated-weight HBM write=4,096B；
- ACK、DONE signature 覆盖两个 active core；
- LSU、DTE、router、link、collective、event residual 全0；
- 不出现 `PROTO_WAIT`。

首次真实运行后，只能冻结从 runtime 原始 marker 观测到的 makespan、artifact SHA 和
marker digest，不得预填。

### 8.3 Negative

至少覆盖：

- STATIC_TRACE assignment/expert home 被篡改；
- local token 被错误生成 DTE flow；
- remote flow 方向或 PairRoute 被交换；
- gradient bytes 从32改为其他值；
- WGRAD output 改为 FP16；
- contribution slice overlap、gap 或 stride 非2,048；
- reduce dtype/count/elements/stride/SUM 被篡改；
- 删除任一 token WGRAD→expert reduce 依赖；
- SGD 改读某个 token contribution；
- expert weight 被写到错误 die；
- ProgramIo 缺少 saved activation/upstream gradient/weight；
- runtime 少/多 D2D packet、ACK、DONE 或出现 residual；
- 将 `compute_functional`、`routing_functional`、`model_functional` 提升为 true。

## 9. 允许的能力声明

全部门禁通过后，只允许声明：

```text
S3_LITE_MOE_EXPERT_WGRAD_TIMING       = true
MOE_CROSS_DIE_GRAD_DISPATCH_TIMING    = true
MOE_EXPERT_DOWN_WEIGHT_SGD_TIMING     = true
MOE_FULL_BACKWARD                     = false
MOE_ROUTER_BACKWARD                   = false
MOE_INPUT_DGRAD                       = false
COMPUTE_FUNCTIONAL                    = false
ROUTING_FUNCTIONAL                    = false
MODEL_FUNCTIONAL                      = false
GENERAL_MESH_MOE                      = false
```

## 10. 超时止损

- T+20m：typed spec/oracle 未绿，停止 lower/C++，只交设计；
- T+45m：不能完全复用现有 MATMUL/reduce/SGD/DTE，降级为 pre-runtime；
- T+70m：single manifest 或 ProgramIo 未闭合，禁止启动 build；
- T+85m：matching build 未绿，禁止使用旧 binary；
- T+105m：双跑未完成，只保留 candidate logs，不声明 runtime 通过；
- T+115m：evidence 未闭合，不做 capability 提升；
- 禁止删除 WAIT/依赖/负测，禁止延长 watchdog，禁止调大 channel/capacity制造通过。

## 11. 后续扩展顺序

本片完成后，按以下顺序逐步走向完整 MoE backward：

1. 增加 down-projection DGRAD 并返回 token source die；
2. 增加 SwiGLU backward；
3. 增加 gate/up projection DGRAD 与 WGRAD；
4. 加入 router/gate gradient 与 load-balancing loss；
5. 将 ProgramIo zero timing payload 升级为数值 functional oracle；
6. 支持 Top-K>1、capacity、overflow/drop；
7. 支持 expert replication 与 expert gradient AllReduce；
8. 扩为任意 EP mesh 和通用 All-to-All；
9. 与 DP/TP/PP、micro-batch、optimizer state、ZeRO/FSDP 组合；
10. 完成完整 S3-N 审计与正式 baseline freeze。
