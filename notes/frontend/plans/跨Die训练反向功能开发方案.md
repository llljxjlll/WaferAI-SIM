# 跨 Die 训练反向功能开发方案

## 1. 目标

在 2 小时开发窗口内，基于现有 S2-Lite LM-head backward 能力，实现一个固定的
DP2×TP1、两 die rooted AllReduce 训练反向纵切，并在真实 `npusim` 上完成
candidate 双跑。

本阶段只支持：

- 2 个 DP replica；
- TP=1、PP=1、EP=1；
- die0 为固定 reduction root；
- 单 micro-batch、单 step；
- LM-head-only backward；
- FP32 WGRAD；
- rooted SUM AllReduce；
- SGD、momentum=0；
- timing execution。

本阶段不声明：

- 任意 mesh 或任意 DP degree；
- ring/tree 自动选择；
- TP2、PP、ZeRO、FSDP、AdamW；
- full-model backward；
- 梯度、权重或 loss 的数值正确性；
- model-functional correctness；
- 正式 baseline freeze。

## 2. 固定执行语义

两个 die 分别完成本地前向、CE backward 和 LM-head WGRAD。随后执行固定 rooted
AllReduce：

```text
die0: WGRAD0 ───────────────────┐
                                ├→ FP32 LOCAL_REDUCE SUM
die1: WGRAD1 → SEND → RECV/WAIT ┘
                                      ↓
die0: reduced_grad → SEND
die1:               RECV/WAIT
                                      ↓
                            SGD0 与 SGD1
```

必须满足以下依赖：

```text
WGRAD0 ───────────────────────────────┐
WGRAD1 → SEND_UP → RECV_UP → WAIT_UP  ├→ REDUCE
REDUCE → SEND_DOWN → RECV_DOWN → WAIT_DOWN
REDUCE ───────────────────────────────────→ SGD0
WAIT_DOWN ────────────────────────────────→ SGD1
```

禁止 SGD 直接读取未经归约的本地 gradient。

## 3. 固定数据量

Tiny case 使用 H=16、V=32：

- 每个 WGRAD：`16 × 32 × 4 = 2,048B`；
- die1→die0 gradient：2,048B；
- root reduce：读取 4,096B，写入 2,048B；
- die0→die1 reduced gradient：2,048B；
- D2D logical bytes：4,096B；
- D2D packet：256 个 16B packet；
- 每个 die 最终写回 1,024B FP16 LM-head weight；
- 总 HBM weight write：2,048B。

这些值必须由 typed source 和实际 runtime marker 独立重建，不能由 runner 自报。

## 4. 最大化复用

直接复用现有组件：

- `S2LiteLmHeadTrainIR0` 的 CE backward、WGRAD 和 SGD；
- Train placement/N4/IR2/schedule 的 DP replica carrier；
- `hardware_2x1.json` 和既有 PairRoute；
- DTE `SEND/RECV/WAIT` 及 UINT8 byte transport；
- `LOCAL_REDUCE` record 结构；
- standalone/fusion region、lifecycle、linker；
- actual-SHA ProgramIo；
- finalizer、resolver、`npusim` 双跑框架；
- S2-Lite runtime marker/parser。

不新增 transport opcode。DTE 对 FP32 WGRAD 只做字节搬运，不解释数值。

## 5. 最小实现切片

### 5.1 Typed source

新增隔离的 `S2LiteDp2AllReduce` carrier：

- 固定 case ID：`case.s2_lite.dp2_tp1.rooted_allreduce`；
- 固定 DP2/TP1 和 root die0；
- 嵌入两个 canonical S2-Lite replica；
- action、buffer、state、core、HBM binding namespace 跨 replica 不相交；
- 固定一条上行 gradient flow 和一条下行 result flow；
- 固定 `SUM` 和 FP32 gradient；
- DP1、DP3、TP2、非 die0 root 全部 fail closed。

### 5.2 IR2、schedule 与 GlobalAction

每个 replica 原有 46 个 action。新增 rooted AR action 后，必须由 producer 输出并由
validator 重新计算 exact action/dependency/buffer-use 集合。

新增资源：

- die0 local WGRAD input；
- die0 remote WGRAD receive buffer；
- die0 reduced gradient buffer；
- die1 reduced gradient receive buffer；
- 两条 DTE flow 的 FSM/token/peer-core binding；
- 一个 FP32 `LOCAL_REDUCE` action。

root reduce 的两个输入必须使用连续、无重叠、等长的 2,048B view。两个 SGD 必须
消费对应 die 上的 reduced gradient buffer。

### 5.3 FP32 LOCAL_REDUCE

现有 record 已包含 input、accumulator、output dtype 字段，且 FP32 枚举已经存在。
本片仅启用固定组合：

```text
input_dtype       = FP32
accumulator_dtype = FP32
output_dtype      = FP32
reduce_op         = SUM
input_count       = 2
element_count     = 512
input_stride      = 2,048B
```

Python schema/lowering、C++ codec/primitive/finalizer 必须同时接受这一组 exact
operands；其他新组合继续拒绝。

### 5.4 Lower、link 与 ProgramIo

- lowerer 只生成现有 opcode；
- 两条 transport 必须闭合 route、core、buffer view、FSM 和 token；
- linker 输出一个统一 manifest；
- ProgramIo 为两个 replica 生成独立参数 initialization；
- 两个 loss probe 和两个 updated-weight write 必须分别属于对应 replica；
- actual artifact SHA 产生后才能生成最终 ProgramIo。

## 6. Root + 3 Agent 并行安排

| 时间 | Root | Agent A | Agent B | Agent C |
|---|---|---|---|---|
| 0–30m | 冻结接口、版本和文件所有权 | DP2 carrier、graph、oracle | FP32 LOCAL_REDUCE Python/C++ | rooted AR lowering、runner/parser |
| 30–55m | 集成 exports/source union | IR2/schedule/global exact closure | codec/primitive/finalizer selftest | SEND/RECV/WAIT/link/ProgramIo |
| 55–75m | 冻结源码，独占增量 build | 只读诊断 | 分析编译首败 | synthetic parser/负测 |
| 75–105m | finalizer×2、resolver、npusim×2 | runtime 日志解析 | artifact/record 核对 | D2D/control/drain 核对 |
| 105–120m | 修复首个真实 blocker、复验和 handoff | focused | C++ adjacent | candidate report |

文件所有权要求：

- Root 独占公共 exports、版本 ledger、CMake 和最终集成；
- Agent A 不修改 artifact/C++；
- Agent B 不修改 runner/ProgramIo；
- Agent C 不修改 compute ABI；
- 55 分钟后冻结源码，除首个真实 P0 外禁止扩大修改范围；
- build、finalizer、resolver、CTest runtime 和 `npusim` 只能由 Root 串行启动。

## 7. 两小时退出门禁

### 7.1 Static positive

- 两个 replica 分别位于 die0、die1；
- 每侧恰有一个 CE backward、WGRAD 和 SGD；
- 两侧 WGRAD 都是 FP32 2,048B；
- root REDUCE 恰有两个 FP32 输入；
- AR 之前不得执行 SGD；
- die0 SGD 读取 root reduced gradient；
- die1 SGD 读取 broadcast receive gradient；
- 上行和下行 flow、route、FSM、token 各自唯一。

### 7.2 Artifact/runtime positive

- finalizer 两次 artifact 和 report byte-exact；
- ProgramIo 使用 actual artifact SHA；
- 两次 `npusim` marker digest 和 makespan 完全一致；
- D2D 上行 2,048B、下行 2,048B；
- 总 D2D logical bytes 为 4,096B；
- 总 data packet 为 256；
- 每个 die 各有一次 SGD marker 和 1,024B weight write；
- ACK、DONE signature 精确；
- LSU、DTE、router、link、collective、control residual 全部为 0；
- 不出现 `PROTO_WAIT`。

### 7.3 Negative

至少覆盖：

- 删除任一 WGRAD→AR 依赖；
- SGD 改读 local gradient；
- 交换上行/下行 token；
- gradient bytes 改为非 2,048；
- root 改为 die1；
- FP32 reduce dtype、input count、stride 或 SUM 被篡改；
- 缺少广播 WAIT；
- runtime D2D packet/ACK/DONE/residual 被篡改；
- 将 `compute_functional` 或 `model_functional` 提升为 true。

## 8. 能力声明

通过全部门禁后只允许声明：

```text
DP2_ROOTED_ALLREDUCE_TIMING = true
TRAIN_BACKWARD_TIMING       = true
COMPUTE_FUNCTIONAL          = false
MODEL_FUNCTIONAL            = false
GENERAL_MESH_ALLREDUCE      = false
```

不得声明通用数据并行训练、任意 mesh AllReduce、数值正确性或真实硬件性能。

## 9. 超时止损

- T+30m：DP2 typed source 未绿，停止 backend 扩展，保留设计而不启动 build；
- T+55m：FP32 reduce 或 rooted AR static closure 未绿，降级为 pre-runtime；
- T+75m：current-tree build 未绿，禁止使用旧 binary；
- T+105m：双跑未闭合，只保留 candidate logs，不声明通过；
- T+115m：evidence 未闭合，不冻结 baseline；
- 禁止通过移除 WAIT、删除负测、增大容量或延长 watchdog 制造通过。

## 10. 后续扩展

本片完成后按顺序扩展：

1. candidate report 升级为正式 typed evidence 和 atomic baseline；
2. fixed root 改为由 policy 选择；
3. DP2 rooted AR 扩为 DP-N tree/ring；
4. 梯度 bucket 和 micro-batch accumulation；
5. TP、DP、PP 组合 mesh；
6. AdamW、ZeRO/FSDP；
7. full-model backward 与数值 functional 验证。
