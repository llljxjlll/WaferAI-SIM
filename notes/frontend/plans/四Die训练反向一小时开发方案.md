# 四 Die 训练反向一小时开发方案

## 1. 目标与完成定义

在 60 分钟内，基于已经跑通的 S2-Lite 两 Die LM-head backward、FP32
`LOCAL_REDUCE`、DTE、SGD、ProgramIo 和 runtime 双跑链，实现一个固定的四 Die
训练反向纵切：

> **S2-Lite DP4×TP1 2×2-mesh tree-AllReduce LM-head-only backward**

完成必须同时满足：

- 使用 `hardware_2x2.json`，Die ID 固定为 `0,1,2,3`；
- DP=4、TP=1、PP=1、EP=1；
- 四个 replica 各执行一次 CE backward、LM-head WGRAD 和 SGD；
- 每个 replica 产生一个 2,048B FP32 WGRAD；
- 采用固定二叉树完成四 Die SUM AllReduce；
- 四个 Die 最终获得同一个 reduced WGRAD timing buffer；
- 四个 SGD 只能在各自 final-gradient dependency 满足后执行；
- 每个 Die 将 1,024B FP16 LM-head weight 写回本地 HBM；
- finalizer 两次、actual-SHA ProgramIo、resolver、`npusim` 两次全部通过；
- 两次 runtime 的 makespan 和 marker digest 完全一致；
- LSU、DTE、router、link、collective、event 和 control residual 全部为0；
- 不出现 `PROTO_WAIT`。

## 2. 一小时范围边界

本片只实现：

- 固定 2×2 Die mesh；
- 固定 DP4×TP1；
- 单 micro-batch、单 step；
- LM-head-only backward；
- FP32 WGRAD、树形 SUM AllReduce；
- SGD，momentum=0；
- timing execution。

本片明确不实现或声明：

- DP2×TP2 backward；
- full-model backward；
- ring AllReduce 或自动 collective policy；
- 任意 mesh、任意 DP degree、非连续 Die ID；
- TP、PP、EP 与 DP 的组合反向；
- gradient bucket、accumulation、overlap policy；
- AdamW、ZeRO、FSDP；
- 梯度、权重或 loss 的数值正确性；
- compute-functional 或 model-functional correctness；
- 正式 baseline freeze。

因此，本方案完成后只能声明固定四 Die LM-head backward timing 跑通，不能声明通用
四 Die训练或完整 S2-N。

## 3. 固定四 Die树形 AllReduce

Die 按 2×2 排列：

```text
Die 0 ─ Die 1
  │       │
Die 2 ─ Die 3
```

为完全复用当前只接受两个 FP32 输入的 `LOCAL_REDUCE`，不把 `input_count` 扩为4，
采用固定邻接二叉树：

```text
reduce phase

g1: Die1 ─────→ Die0 ─┐
                        ├→ p01 ─┐
g0: Die0 ──────────────┘        │
                                ├→ global_sum @ Die0
g3: Die3 ─────→ Die2 ─┐        │
                        ├→ p23 ─┘
g2: Die2 ──────────────┘

broadcast phase

global_sum @ Die0 ─────→ Die1
          │
          └────────────→ Die2 ─────→ Die3
```

固定通信 flow：

1. `UPLOAD_1_TO_0`：Die1 WGRAD → Die0；
2. `UPLOAD_3_TO_2`：Die3 WGRAD → Die2；
3. `PARTIAL_2_TO_0`：Die2 partial sum → Die0；
4. `BROADCAST_0_TO_1`：Die0 global sum → Die1；
5. `BROADCAST_0_TO_2`：Die0 global sum → Die2；
6. `BROADCAST_2_TO_3`：Die2 global sum → Die3。

固定 reduce：

1. Die0：`SUM(g0, g1) -> p01`；
2. Die2：`SUM(g2, g3) -> p23`；
3. Die0：`SUM(p01, p23) -> global_sum`。

每个 reduce 必须继续使用当前已验证的 exact ABI：

```text
input_dtype       = FP32
accumulator_dtype = FP32
output_dtype      = FP32
reduce_op         = SUM
input_count       = 2
element_count     = 512
input_stride      = 2,048B
source_span       = 4,096B
destination_span  = 2,048B
```

## 4. 固定数据量与结构公式

Tiny case 保持 H=16、V=32：

- 每 replica WGRAD：`16 × 32 × 4 = 2,048B`；
- reduce flow：3条×2,048B；
- broadcast flow：3条×2,048B；
- D2D logical bytes：`6 × 2,048 = 12,288B`；
- D2D data packets：`12,288 / 16 = 768`；
- 每个方向、每条 link 的 packet 数从实际 `PairRoute` 独立推导；
- `LOCAL_REDUCE`：3次，每次读取4,096B、写入2,048B；
- SGD：4次，每次更新512个 FP16 element；
- HBM weight write：`4 × 1,024 = 4,096B`；
- local S2-Lite action：`4 × 46 = 184`；
- overlay executable unit：2个 local copy、6组 SEND/RECV/WAIT、3个 reduce，共23个；
- manifest 只能有一个，不能按 replica 或树层拆成多个 artifact。

fragment、record、relocation、symbol 和 ProgramIo 的最终数量不得预填；必须从首次真实
producer/finalizer 输出冻结，并由 validator 独立重算。

## 5. 最大化复用策略

直接复用：

- `S2LiteLmHeadTrainIR0` 的 CE backward、WGRAD、SGD；
- 当前 Train placement/N4/N5/schedule 的 replica 展开；
- `S2LiteDp2RootedAr*` 的 typed gradient/state/SGD dependency 模型；
- `hardware_2x2.json` 的 fabric、HBM 和 `PairRoute`；
- DTE `SEND/RECV/WAIT` byte transport；
- 当前 exact FP32 two-input `LOCAL_REDUCE`；
- rooted-AR lower/link/ProgramIo 的 multi-context closure；
- current finalizer trust、resolver 和 runtime parser；
- serial CTest、actual-SHA 和双跑框架。

禁止为了赶时间：

- 新增 opcode、PrimId 或 record wire；
- 将 `LOCAL_REDUCE.input_count` 放宽为任意值；
- 把四个输入伪装成一个大 tensor；
- 删除 WAIT、依赖、lifecycle 或负测；
- 使用旧 binary 或预填 runtime golden；
- 增大 channel、capacity、timeout 或 watchdog 制造通过。

## 6. 最小实现切片

### 6.1 DP4 typed source 与 GlobalAction

新增隔离 carrier，建议命名：

- `S2LiteDp4TreeArSource`；
- `S2LiteDp4TreeArContract`；
- `S2LiteDp4TreeArGlobalAction`；
- case ID：`case.s2_lite.dp4_tp1.tree_allreduce`。

严格要求：

- graph 只允许 DP4/TP1/PP1/EP1；
- replica0..3 分别落 Die0..3；
- action、binding、state、runtime token 和 channel namespace 全局不相交；
- 六条 flow、三个 reduce 和四个 SGD dependency 与第3节完全一致；
- DP3、DP5、TP2、交换 Die、删除树边或改变 root 全部 fail closed。

不修改既有 DP2 carrier 的语义和 schema version；DP4 使用新 schema alpha1。

### 6.2 Scratch、alias 与生命周期

Die0 和 Die2 各分配两个连续2,048B FP32 slice：

```text
slice0: offset 0,    size 2,048B
slice1: offset 2,048,size 2,048B
root span:           size 4,096B
```

- Die0 第一次 reduce 后允许覆盖 slice0 为 `p01`；
- Die0 在 slice1 接收 `p23` 后再次 two-input reduce；
- Die2 第一次 reduce 后 slice0 为 `p23`；
- Die2 将 `p23` 发送到 Die0，之后才能接收并转发 global sum；
- alias 必须保持 canonical root 的 placement、storage、layout 和 span；
- view 只通过 operand slice 表达，不能通过伪造较小 BufferABI 表达；
- 每个 storage 恰有一个 canonical ALLOC/FREE lifecycle。

### 6.3 Lower、single manifest 与 ProgramIo

- 四个 canonical `LoweringContext`；
- 184个 local action leaf 加17个 overlay leaf；
- overlay leaf：2 COPY、6 SEND、6 RECV/WAIT endpoint、3 REDUCE；
- 一个 top-level DP4 rooted-AR digest；
- manifest 必须闭合全部4个 IR1/projection/schedule/global lineage；
- C++ finalizer 按新 top schema 区分 DP2 和 DP4，不放宽 generic manifest；
- ProgramIo 展开4份 state seed、label、loss-gradient 和 updated-weight probe；
- output probe 只覆盖4个 updated LM-head weight；
- actual artifact SHA 产生前不得构造最终 ProgramIo。

### 6.4 Runtime parser

runner 不允许硬编码 runtime core ID。必须从 typed schedule/manifest 推导：

- 4个 active core；
- 6条 flow 及其实际 `PairRoute`；
- logical bytes=12,288B；
- byte-hop 和 directed-link packet 总数；
- 4个 SGD marker 和4,096B HBM weight write；
- ACK、DONE、P5、collective、router 和 link drain；
- ProgramIo 的4个 HBM output probe；
- repeat makespan 和 marker digest。

## 7. Root + 3 Agent 的60分钟安排

build、finalizer、resolver、CTest 和 `npusim` 只能由 Root 串行启动。

| 时间 | Root | Agent A：typed graph | Agent B：N6/trust | Agent C：ProgramIo/runner |
|---|---|---|---|---|
| 0–10m | 冻结 contract、ID、版本和文件所有权 | DP4 source/global carrier | 审计DP2 lower/link复用点 | 冻结4-core marker/route公式 |
| 10–25m | 处理唯一 shared type gate与exports | DP4 placement→schedule、dependency negatives | 23-unit intent、17 overlay leaf、single manifest | 4-context ProgramIo与mock parser |
| 25–35m | 合并并跑focused，禁止新设计 | serde/stable-id/DP3负测 | finalizer dedicated stdin + tamper | actual-SHA builder与parser negatives |
| 35–43m | 冻结源码，独占matching增量build | 只读review | 分析首个编译错误 | 检查expected quotient |
| 43–53m | finalizer×2→resolver→npusim×2 | 解析首次runtime | 核对record/ABI | 核对D2D/control/drain |
| 53–60m | 只修首个真实P0、复跑唯一CTest、commit/handoff | focused | C++ adjacent | candidate report |

文件所有权：

- Root：公共 exports、CMake、shared union、最终集成和唯一 runtime token；
- Agent A：新 DP4 schema/pass/test，不碰 artifact/C++；
- Agent B：rooted N6/linker/C++ trust，不碰 runner；
- Agent C：ProgramIo/runner/integration，不碰 compute ABI；
- T+35m 后冻结源码，除 official 首个真实 P0 外禁止扩大范围；
- 不允许两个 agent 同时编辑 `artifact_manifest.py`、`program_io.py`、linker 或 C++ finalizer。

## 8. 一小时门禁

### 8.1 Static positive

- 4个 replica 分别位于 Die0..3；
- 每个 replica 恰有1个 CE backward、WGRAD 和 SGD；
- 四个 WGRAD 均为 FP32 2,048B；
- 六条 DTE flow 的 source、destination、route、FSM 和 token 唯一；
- 三个 reduce 均为 exact two-input FP32 SUM；
- Die0 最终 reduce 依赖 `p01` 和 `p23`；
- Die1、Die2、Die3 的 SGD 分别等待其 final-gradient 到达；
- Die2 转发 global sum 前必须完成从 Die0 的接收；
- 四个 updated weight 分别写回本 Die HBM；
- 既有 DP2 rooted-AR 和 Dense/TrainForward carrier exact no-drift。

### 8.2 Artifact/runtime positive

- one manifest，覆盖4个 context 和全部 tree overlay leaf；
- finalizer 两次 artifact/report byte-exact；
- ProgramIo 使用 actual artifact SHA；
- resolver initialization/probe exact；
- `npusim` 两次 makespan 和 marker digest 完全相同；
- backward D2D logical bytes=12,288B；
- aggregate data packets=768，directed link 与 byte-hop 独立闭合；
- 3次 `LOCAL_REDUCE`、4次 SGD；
- total updated-weight HBM write=4,096B；
- ACK、DONE signature 覆盖4个 active core；
- 所有 residual=0，无 `PROTO_WAIT`。

### 8.3 Negative

至少覆盖：

- DP degree 或 Die mapping 被改为非 DP4/Die0..3；
- 任一 reduce/broadcast tree edge 删除、反向或交换；
- 任一 WGRAD→reduce、partial→final reduce、receive→relay/SGD 依赖删除；
- SGD 读取 local 或 partial gradient；
- FP32 reduce dtype、count、elements、stride 或 SUM 被篡改；
- contribution slice overlap、gap、短 span 或错误 alias；
- remote bytes 非2,048或 packet/link accounting 不闭合；
- ProgramIo 缺任一 replica seed、initialization 或 updated-weight probe；
- runtime 少/多 packet、ACK、DONE 或出现任意 residual；
- 将 `compute_functional` 或 `model_functional` 提升为 true。

## 9. 允许的能力声明

全部门禁通过后，只允许声明：

```text
DP4_TP1_TREE_ALLREDUCE_TIMING = true
FOUR_DIE_TRAIN_BACKWARD_TIMING = true
LM_HEAD_ONLY                  = true
COMPUTE_FUNCTIONAL            = false
MODEL_FUNCTIONAL              = false
GENERAL_MESH_ALLREDUCE        = false
FULL_MODEL_BACKWARD           = false
```

## 10. 超时止损

- T+10m：DP4 contract 未冻结，停止 shared 修改；
- T+25m：DP4 source→schedule 或23-unit intent未绿，降级为 static/pre-runtime；
- T+35m：single manifest 或 ProgramIo 未闭合，禁止 build；
- T+43m：matching build 未绿，禁止使用旧 binary；
- T+53m：双跑未闭合，只保留 candidate logs，不声明 runtime 支持；
- T+58m：仍有 P0，只提交已验证的 pre-runtime 部分并明确缺口；
- T+60m：不得通过删 WAIT、负测、lifecycle、route 或放宽 validator 制造完成。

## 11. 后续扩展

本片完成后按以下顺序扩展：

1. DP4×TP1 tree 固定拓扑升级为拓扑驱动 tree；
2. 增加 DP2×TP2 rank-matched gradient AllReduce；
3. tree/ring policy 选择与梯度 bucket；
4. micro-batch accumulation 与通信重叠；
5. full-model backward；
6. 任意 mesh DP-N 与数值 functional oracle。
