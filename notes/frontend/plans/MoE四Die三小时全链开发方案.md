# MoE 四 Die 三小时全链开发方案

## 1. 目标与完成定义

在 3 小时内，基于当前已经跑通的 S3-Lite 两 Die MoE 推理前向、MoE
down-projection WGRAD backward，以及刚完成的 2×2 Die mesh、DTE、FP32
`LOCAL_REDUCE`、SGD、ProgramIo 和官方双跑基础，实现三条固定四 Die workload：

1. **M4-I：S3-Lite 4-Die static-route MoE inference forward**；
2. **M4-TF：S3-Lite 4-Die static-route MoE training forward**；
3. **M4-TB：S3-Lite 4-Die static-route expert down-projection WGRAD-only backward**。

三条 workload 的公共约束为：

- 使用 `hardware_2x2.json`，Die ID 固定为 `0,1,2,3`；
- EP=4、4 experts、每个 Die 恰放置1个 expert；
- Top-K=1，固定 T8 balanced `STATIC_TRACE`；
- FP16 activation/weight，FP32 WGRAD/reduction；
- 所有跨 Die 数据真实经过 DTE `SEND/RECV/WAIT` 和实际 `PairRoute`；
- 每条 workload 都生成一个独立、完整、可由 production finalizer 消费的 manifest；
- 每条 workload 都执行 finalizer 两次、actual-SHA ProgramIo、resolver、`npusim` 两次；
- 每条 workload 的两次 artifact/report、makespan 和 marker digest 完全一致；
- LSU、DTE、router、link、collective、event 和 control residual 全部为0；
- 不出现 `PROTO_WAIT`。

三条 workload 全部通过前，不得声明本方案完成。

## 2. 本阶段准确能力边界

本阶段实现的是：

> **固定 2×2 mesh 上的 S3-Lite EP4 静态路由 MoE timing workload matrix**

其中训练反向仍只覆盖 expert down-projection WGRAD、expert 内 reduction、SGD 和
weight store，不等于完整 MoE backward。

本方案明确不实现或声明：

- 动态 `GATE_TOPK`、Top-K>1、capacity overflow、token drop 或 reroute；
- gate/router backward、routing probability gradient、load-balancing loss；
- down-projection input DGRAD 返回 token source；
- SwiGLU backward、gate/up projection DGRAD/WGRAD；
- attention、embedding、LM-head 或 backbone backward；
- expert replication 或 expert gradient AllReduce；
- DP、TP、PP 与 EP 的组合并行；
- micro-batch accumulation、通信重叠、ZeRO、FSDP、AdamW；
- 推理输出、activation tape、gradient 或 updated weight 的数值正确性；
- compute-functional、routing-functional 或 model-functional；
- 任意 mesh、任意 Die 数、任意 expert 数或通用 All-to-All；
- 正式 baseline freeze 或 S3-N 完成。

如果需求是“一个 artifact 内完整 forward→loss→full backward→optimizer step”，三小时
不足。本方案用三个可独立运行、共享 exact typed ABI 的 workload 覆盖推理前向、训练
前向和简化训练反向。

## 3. 唯一四 Die 拓扑与静态 trace

### 3.1 物理与逻辑拓扑

固定 2×2 Die mesh：

```text
Die 0 ─ Die 1
  │       │
Die 2 ─ Die 3
```

固定映射：

```text
expert 0 → Die 0
expert 1 → Die 1
expert 2 → Die 2
expert 3 → Die 3

token source die = token_index % 4
expert home die  = expert_index
```

`PairRoute` 必须从 production fabric/mapping 构造，不能把上图直接转换成硬编码 runtime
core 或 link。logical bytes 和 byte-hop 必须独立计算，多跳 route 必须逐 link 计数。

### 3.2 固定 T8 balanced trace

沿用现有 token→expert 语义：

```text
token 0,1 → expert 0
token 2,3 → expert 1
token 4,5 → expert 2
token 6,7 → expert 3
```

因此每个 expert 恰有2个 token。按 `token_index % 4` 的 source 规则：

- local token：`0`、`7`；
- remote token：`1,2,3,4,5,6`；
- remote assignment 恰好6个。

这组 trace、expert home、slot 和 route 是三个 workload 的唯一公共真相。禁止三个
producer 各自复制一份映射公式。

## 4. 固定数据量与公式

Tiny 模型保持：

- token count `T=8`；
- hidden size `H=16`；
- intermediate size `I=32`；
- expert count=4；
- each expert token count=2；
- activation/weight 为 FP16；
- WGRAD/reduce 为 FP32；
- SGD momentum=0。

公共数据量：

- 每 token hidden payload：`H × 2 = 32B`；
- 每 token down-projection saved activation：`I × 2 = 64B`；
- 每 expert gate/up/down weight：各 `1,024B`；
- 12个 expert weight 总计 `12,288B`，每 Die `3,072B`；
- 每 expert MLP FLOPs：`6 × 2 × H × I = 6,144`；
- 四 expert forward FLOPs 总计 `24,576`。

前向通信：

- remote token=6；
- dispatch：`6 × 32B = 192B`；
- combine：`6 × 32B = 192B`；
- forward logical D2D 总计 `384B`；
- 16B packet 总计 `24`；
- byte-hop 和 directed-link packet 由实际12条 flow 的 `PairRoute` 重建。

训练反向：

- 6个 remote upstream gradient：`6 × 32B = 192B`；
- backward data packet 总计 `12`；
- 每 token WGRAD：`I × H × 4 = 2,048B`；
- 8个 token WGRAD 总计 `16,384B`；
- 每 expert 两个 contribution：连续 `4,096B`；
- 每 expert reduce 输出 `2,048B`；
- 4次 FP32 `LOCAL_REDUCE SUM`；
- 4次 SGD；
- updated down-weight HBM write：`4 × 1,024B = 4,096B`；
- token WGRAD FLOPs：`2 × I × H = 1,024/token`，总计 `8,192`。

上述合计必须由 typed trace、buffer shape、PairRoute、record 和 ProgramIo 独立重建，
runner 不得接受 caller 自报 aggregate。

## 5. 三条 workload 的精确语义

### 5.1 M4-I：推理前向

每个 token 执行：

```text
token source
  └─ remote? SEND → RECV → WAIT @ expert home
                     ↓
               gate/up GEMM
                     ↓
                  SwiGLU
                     ↓
                 down GEMM
                     ↓
expert home ─ remote? SEND → RECV → WAIT @ token source
                     ↓
                combined output
```

严格要求：

- 8个 token、每 token 3个 GEMM +1个 SwiGLU；
- 6条 dispatch 和6条 combine flow；
- local token 不生成 DTE flow；
- 12个 parameter state 均为 READ_ONLY；
- ProgramIo 提供 deterministic input/weight，输出 probe 覆盖8个 combined token；
- 只声明 forward timing，不声明输出数值正确。

### 5.2 M4-TF：训练前向

训练前向复用 M4-I 的全部计算和通信，不复制第二套 MoE forward lowering。额外加入
typed tape-copy overlay：

```text
SwiGLU output / down-projection input
          ├─→ down GEMM
          └─→ LOCAL_COPY → terminal saved-activation tape
```

严格要求：

- 8个 tape copy，每个64B，总计512B；
- 每个 tape 与 token、expert、slot、expert-home die 一一对应；
- copy 只能读取真实 down-projection input，不能从 ProgramIo 伪造 forward tape；
- down GEMM 与 tape copy 都依赖同一个真实 producer；
- tape buffer 必须是独立 OWNED terminal buffer，不覆盖 forward compute buffer；
- ProgramIo probe 同时覆盖8个 combined output 和8个 tape；
- training-forward carrier 必须 embed exact M4-I topology/forward provenance；
- 不加入 backward、optimizer 或 upstream gradient。

M4-TF 证明“训练所需 activation tape 的 timing 生产路径可运行”，不证明 tape 数值正确。

### 5.3 M4-TB：训练反向

训练反向复用现有两 Die `LiteMoeBackwardOverlay` 的数学和 ABI，只新增 DP4 exact
topology adapter：

```text
token source die                         expert home die

upstream_grad(local)
    └──────────────────────────────────→ TOKEN_WGRAD

upstream_grad(remote)
    └→ SEND → RECV → WAIT ─────────────→ TOKEN_WGRAD

token_wgrad(slot0) ─┐
                     ├→ FP32 LOCAL_REDUCE SUM
token_wgrad(slot1) ─┘
                                      ↓
                              SGD_UPDATE down_weight
                                      ↓
                              HBM STORE updated_weight
```

严格要求：

- 8个 saved activation ABI 与 M4-TF tape ABI 的 token/expert/shape/dtype/die 完全一致；
- isolated backward runtime 可由 ProgramIo seed tape，但 carrier 必须
  `validate_against(M4-TF tape catalog)`，不得产生第二套 tape 真相；
- 6个 remote upstream-gradient flow，方向为 forward combine 的反向；
- 8个 FP32 2,048B token WGRAD；
- 每 expert 两个 contribution 连续、等长、无重叠；
- 每 expert 恰好一次 two-input reduce、一次 SGD、一次 HBM store；
- SGD 只能读取 reduced WGRAD，不能读取任一 token contribution；
- 4个 down weight 使用新 TRAINABLE_PARAMETER/PERSISTENT/READ_WRITE witness；
- output probe 只覆盖4个 updated down weight，共4,096B。

`LOCAL_REDUCE` 继续使用已验证的 exact ABI：

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

## 6. 最大化复用与禁止事项

### 6.1 必须直接复用

- `LiteMoeStaticTrace` 的 assignment/slot 完整性规则；
- 当前 S3-Lite expert GEMM、SwiGLU 和 packed-buffer ABI；
- 当前 MoE typed P2P binding、DTE `SEND/RECV/WAIT` lowering；
- `hardware_2x2.json`、production fabric parser、HBM 和 XY `PairRoute`；
- 当前 `MATMUL` dual-input program binding；
- FP32→FP32→FP32 two-input `LOCAL_REDUCE SUM`；
- `SGD_UPDATE`、TRAINABLE_PARAMETER、ALIASED lifecycle；
- S3-Lite forward/backward single-manifest linker 结构；
- actual-SHA ProgramIo、finalizer、resolver 和双跑 runner；
- DP4 runner 的动态 active-core、directed-link、ACK/DONE/drain parser；
- `RUN_SERIAL` CTest 和 matching-build 流程。

### 6.2 禁止为了赶时间新增或放宽

- 不新增 opcode、PrimId、record wire 或 ISA version；
- 不把现有两 Die `LiteMoeSpec` 直接改成可变 Die 数，避免旧 artifact 语义漂移；
- 不放宽 Dense/Train validator 来容纳 MoE；
- 不新增通用 All-to-All planner；
- 不用 KV state-transfer、Residual、generic Elementwise 冒充 MoE transport/compute；
- 不按 impl_ref、shape 或 opcode 猜 workload 身份；
- 不删除 WAIT、依赖、ALLOC/FREE、negative 或 strict provenance；
- 不调大 channel、capacity、watchdog 或 timeout 制造 runtime 通过；
- 不使用旧 binary 或修复前的 artifact/makespan。

## 7. 最小生产结构与文件边界

### 7.1 唯一 DP4 topology source

新增隔离公共 carrier，建议：

```text
schema/lite_moe_dp4.py
passes/lite_moe_dp4.py
```

公开类型：

- `LiteMoeDp4Spec`；
- `LiteMoeDp4Topology`；
- `LiteMoeDp4Oracle`；
- `LiteMoeDp4P2PBinding`；
- `LiteMoeDp4ExecutionCase`。

固定 case IDs：

```text
case.s3_lite.dp4_ep4.static_moe_infer
case.s3_lite.dp4_ep4.static_moe_train_forward
case.s3_lite.dp4_ep4.static_moe_down_wgrad
```

旧 2-die schema/version 不修改；DP4 使用新 alpha1 carrier。三个 workload 都 embed 同一个
topology ID/oracle ID，独立重算并比较，不能只比较 caller 传入 digest。

### 7.2 Forward execution 与 training-tape overlay

建议隔离：

```text
schema/lite_moe_dp4_execution.py
passes/lite_moe_dp4_execution.py
schema/lite_moe_dp4_train_forward.py
passes/lite_moe_dp4_train_forward.py
```

forward graph 仍按 token 静态展开。相对现有两 Die case，compute/state 数不变，只把
remote P2P binding 从8条扩为12条。预计结构公式：

- 32 compute node：24 GEMM +8 SwiGLU；
- 12 P2P logical flow：6 dispatch +6 combine；
- 12 parameter state、24 READ accesses；
- projection 产生12组 SEND/RECV/WAIT；
- training-forward 额外8个 typed local-copy tape action。

精确 action/fragment/record/relocation 数以首个 production producer/finalizer 输出为准，
不得把预计值直接冻结为 golden。

### 7.3 N6、single manifest 与 trust anchor

建议隔离：

```text
schema/lite_moe_dp4_n6.py
passes/lite_moe_dp4_lower_program.py
passes/lite_moe_dp4_link_program.py
lowering/lite_moe_dp4.py
lowering/lite_moe_dp4_linker.py
```

每条 workload 恰有一个 manifest：

- M4-I：forward fragments；
- M4-TF：同一 forward fragments + tape overlay fragments；
- M4-TB：state load + backward DTE + WGRAD + reduce + SGD/store fragments。

可复用现有 `ManifestInputKind.S3_LITE_MOE`，但 finalizer 必须按三个新 top schema
精确分支；禁止放宽 generic S3 manifest。只有 accepted enum/wire 真正改变时才允许版本
bump，本方案默认不 bump。

C++ selftest 建议 dedicated modes：

```text
--lite-moe-dp4-infer-stdin
--lite-moe-dp4-train-forward-stdin
--lite-moe-dp4-backward-stdin
```

每个 mode 使用 production manifest，并包含 top digest、lineage、leaf count、opcode order、
route、dtype/span、state permission 的 restable tamper negatives。

### 7.4 ProgramIo 与 runner

建议新增：

```text
integration/lite_moe_dp4_cases.py
integration/run_lite_moe_dp4_runtime_matrix.py
integration/test_lite_moe_dp4_cases.py
integration/test_run_lite_moe_dp4_runtime_matrix.py
```

ProgramIo 模式：

- M4-I：input/weight initialization +8 combined-output probes；
- M4-TF：同 M4-I +8 saved-tape probes；
- M4-TB：8 saved activation +8 upstream gradient +4 trainable down-weight seed，
  output 只 probe 4 updated weight。

runner 必须从 typed case、manifest 和 ProgramIo 动态推导：

- active runtime cores；
- remote assignment 和实际 PairRoute；
- logical bytes、byte-hop、request/ACK/DATA packets；
- per-core LSU/HBM/SRAM bytes；
- forward compute、tape copy、WGRAD、reduce、SGD 的静态计数；
- output probe 的 die/address/bytes/checksum；
- ACK/DONE、P5、collective、router、link drain；
- repeat makespan 和 marker digest。

不要求发明新的 MoE functional marker。没有专用 marker 的 WGRAD/reduce/tape 只能以
typed static sequence + runtime completion 证明 timing execution，不能据此声明 functional。

## 8. Root + 3 Agent 三小时安排

Root 是唯一 integrator，也是唯一 build/finalizer/resolver/CTest/`npusim` token owner。
任一时刻禁止两个 agent 同时 build 或运行 official workload。

| 时间 | Root | Agent A：topology/forward | Agent B：N6/trust | Agent C：ProgramIo/runner |
|---|---|---|---|---|
| 0–20m | 冻结3个case ID、schema、文件所有权 | DP4 topology/spec/oracle | 审计forward/backward lower/link复用点 | 冻结三模式ProgramIo/marker合同 |
| 20–50m | 处理唯一shared type gate | source→IR0→placement/N4 | topology-neutral lower helper、link设计 | self-contained builder、dynamic PairRoute expected |
| 50–80m | 合并首次forward carrier | infer projection/schedule/global | M4-I lower/link/manifest | M4-I ProgramIo/mock runner |
| 80–105m | 冻结forward ABI | M4-TF tape overlay + M4-TB typed overlay | M4-TF/M4-TB lower/link | TF/TB ProgramIo与probe closure |
| 105–125m | exports、shared exact union、review | serde/provenance/negative | C++三个strict trust branch/selftest | matrix runner finalizer×2/resolver injection |
| 125–140m | 源码冻结、统一focused | forward/TF adjacent | N6/link/finalizer adjacent | case/parser/ProgramIo adjacent |
| 140–155m | 唯一 matching build | 只读首错分析 | 只读compile/selftest分析 | 只读runtime expectation审计 |
| 155–175m | 三个trust stdin + official matrix串行双跑 | 核对typed action/route | 核对record/state/span | 核对D2D/memory/probe/drain |
| 175–180m | diff审计、commit/handoff | 停止编辑 | 停止编辑 | 停止编辑 |

文件所有权：

- Root：公共 exports、CMake、shared version/type ledger、最终集成；
- Agent A：DP4 topology、forward、train-forward/backward typed overlay，不碰 C++；
- Agent B：N6/lowering/linker/C++ trust，不碰 runner；
- Agent C：self-contained cases、ProgramIo、runner/parser，不碰 compute ABI；
- T+125m 后冻结源码，除 official run 暴露的首个真实 P0 外不扩大范围。

## 9. 三小时门禁

### 9.1 公共 static positive

- exact 4 dies/EP4/4 experts/Top-K1/T8 static trace；
- expert0..3 与 Die0..3 一一对应；
- remote token exact `(1,2,3,4,5,6)`；
- local token 不生成 DTE；
- 6 dispatch、6 combine、6 backward-gradient flow 的方向和 PairRoute exact；
- 每 expert 恰有2个 token，slot exact `(0,1)`；
- 旧两 Die infer/backward carrier no-drift。

### 9.2 M4-I positive

- 8 token ×(3 GEMM+1 SwiGLU)；
- forward D2D logical bytes=384B、data packets=24；
- 12 parameter state 全部只读并位于正确 expert home；
- combined output 恰好8个；
- one manifest、finalizer×2、resolver、npusim×2 全绿。

### 9.3 M4-TF positive

- M4-I 全部条件继续成立；
- 8个64B tape copy，total512B；
- tape 与真实 down-projection input provenance 一一对应；
- combined output + tape probes exact；
- tape ABI catalog 可被 M4-TB 独立 cross-validate；
- one manifest、finalizer×2、resolver、npusim×2 全绿。

### 9.4 M4-TB positive

- 6条 reverse-combine gradient flow、192B、12 packets；
- 8个 WGRAD，每个2,048B FP32；
- 4个 two-input reduce，source span4,096B、dest span2,048B；
- 4个 SGD 和4个1,024B HBM store；
- output probes 仅4个 updated down weight，总4,096B；
- one manifest、finalizer×2、resolver、npusim×2 全绿。

### 9.5 Runtime matrix positive

三个 case 各自要求：

- artifact/report 两次 byte-exact；
- actual artifact SHA 注入 ProgramIo；
- resolver initialization/probe exact；
- npusim 两次 makespan、marker digest exact；
- D2D TYPE 与所有 directed links 双向闭合；
- ACK/DONE 覆盖全部 active core；
- LSU、DTE、router、link、collective、event residual 全0；
- 不出现 `PROTO_WAIT`；
- runner 明确输出 `timing_execution=1 functional_execution=0`。

建议注册唯一串行 CTest：

```text
s3_lite_dp4_runtime_matrix
```

CTest 内按 M4-I→M4-TF→M4-TB 串行执行，任何子 case 失败则整体失败。不得用 CTest
并行运行三个 npusim。

### 9.6 Negative

至少覆盖：

- DP3、DP5、EP2、TP2 或伪 producer；
- trace assignment、slot、expert home 或 Die 交换；
- local token 错生 DTE；
- remote flow 方向、PairRoute、channel/token 或bytes篡改；
- 少/多 dispatch、combine 或 backward flow；
- tape copy 读取错误 buffer、错 expert/die、漏 token 或互相 alias；
- M4-TB tape catalog 与 M4-TF shape/dtype/die 不一致；
- WGRAD output 改FP16；
- contribution overlap/gap/stride错误；
- reduce dtype/count/elements/stride/SUM错误；
- SGD 读取 token contribution 或写回错误 expert state/die；
- ProgramIo 缺任一 input/weight/tape/upstream/probe；
- probe target SRAM/HBM kind、StateABI、die、address、bytes 错误；
- manifest top schema、lineage、leaf、stream、symbol、interface 或 envelope篡改；
- runtime 少/多 packet、ACK、DONE，任一 residual 非0或出现 `PROTO_WAIT`；
- 将任何 functional flag 提升为 true。

## 10. 允许的能力声明

所有门禁通过后，只允许声明：

```text
S3_LITE_MOE_4D_INFER_FORWARD_TIMING       = true
S3_LITE_MOE_4D_TRAIN_FORWARD_TIMING       = true
S3_LITE_MOE_4D_DOWN_WGRAD_TIMING          = true
S3_LITE_MOE_4D_STATIC_ROUTE_DTE_TIMING    = true

S3_LITE_MOE_4D_FULL_TRAIN_STEP            = false
MOE_FULL_EXPERT_BACKWARD                  = false
MOE_ROUTER_BACKWARD                       = false
MOE_INPUT_DGRAD                            = false
MOE_EXPERT_GRADIENT_ALLREDUCE              = false
COMPUTE_FUNCTIONAL                        = false
ROUTING_FUNCTIONAL                        = false
MODEL_FUNCTIONAL                          = false
GENERAL_MESH_MOE                           = false
```

## 11. 超时止损

- T+20m：公共 DP4 topology/spec/oracle 未冻结，停止 shared 修改；
- T+50m：source→placement/N4 未绿，降级为 static-only；
- T+80m：M4-I single manifest 未绿，停止训练两条线，不启动 build；
- T+105m：M4-TF/M4-TB typed overlay 或 ProgramIo 未闭合，按单 case 降级；
- T+125m：三个 case focused 未全绿，禁止源码冻结；
- T+140m：仍有静态 P0，禁止 matching build；
- T+155m：matching build/trust 未绿，禁止使用旧 binary；
- T+170m：任一 official 双跑未闭合，只声明已通过的独立 case，不能声明 matrix；
- T+178m：仍有 P0，不做 capability 提升；
- 禁止通过删 WAIT、依赖、negative、lifecycle，或调大 timeout/watchdog/channel
  制造完成。

## 12. 后续扩展顺序

本片完成后按以下顺序扩展：

1. 将 M4-TF tape 与 M4-TB 合并成一个 artifact 内的 forward→backward dependency；
2. 增加 down-projection input DGRAD 并 reverse-dispatch 返回 token source；
3. 增加 SwiGLU backward；
4. 增加 gate/up projection DGRAD/WGRAD；
5. 增加 router/gate gradient 与 load-balancing loss；
6. 加入 expert replication 和 expert-gradient AllReduce；
7. 增加 Top-K>1、capacity、overflow/drop；
8. 将固定 EP4/2×2 adapter 升级为拓扑驱动 arbitrary EP mesh；
9. 组合 DP/TP/PP、micro-batch、gradient bucket、ZeRO/FSDP；
10. 增加数值 functional oracle、完整 S3-N 审计与 baseline freeze。
