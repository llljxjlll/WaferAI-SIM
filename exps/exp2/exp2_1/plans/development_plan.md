# exp2-1 开发方案：目标硬件绑定的端到端分层回放与少量周期精确校准

## 1. 实验目标与结论边界

本实验比较同一块 `6×6` wafer 上、相同模型工作量和相同 placement 下：

- `T_base`：不做计算—通信 overlap 的 naive 调度；
- `T_overlap`：仅打开 work1 已支持的计算—通信 overlap；
- `speedup = T_base / T_overlap`。

主结果覆盖训练完整 step 和推理的 prefill、PD KV handoff、decode steady step。方法采用：

1. 复用仿真器已经跑通并严格 drain 的 Dense Train、MoE、静态 Prefill/Decode、PD 和
   MeshSlice 负载链；
2. 在本实验目标硬件配置下，只运行少量调度签名级的周期精确 motif 和直接校验点；
3. 用资源显式、算子专用的动作 DAG 回放扩展到所有模型/shape；
4. 对容量不成立、MLA 未被 simulator 支持或没有当前硬件校准的结果降级标注，禁止把它们
   写成完整模型周期精确实测。

最终结果应表述为 `cycle-accurate-anchor-calibrated E2E estimate`，而不是“24 个完整大模型
均做了周期精确仿真”。只有直接运行的校验点可以标为 `cycle_accurate_direct`。

## 2. 先修正原 plan 中会影响真实性的口径

### 2.1 模型数是 6，不是 8

原 `plan.md` 的模型表实际有 6 个模型，结果展示处的“八个模型”是笔误。主图固定为：

```text
6 models × 2 training seq_len = 12 training cases
6 models × 2 decode batch     = 12 inference primary cases
```

另输出 6 个固定 `prefill(seq=2304,batch=1)` 的 breakdown，不将其重复伪装成两个不同
prefill case。

### 2.2 E2E 必须先补齐模型 manifest

exp1 的 H/I/QKV 参数足以生成单算子 shape，但不足以计算完整模型 E2E。每个模型必须冻结：

```text
num_layers, vocab_size, hidden_size, intermediate_size,
num_attention_heads, num_kv_heads, head_dim,
attention_type, mlp_type, norm/residual/rope,
moe_layer_frequency, routed_expert_count, shared_expert_count, top_k,
parameter_count, dtype
```

manifest 必须带来源和内容 digest。没有这些字段时只允许输出单层结果，不允许乘一个猜测的
层数得到 E2E。

DeepSeek-V3 的 MLA 不能替换为标准 MHA/GQA。当前 ISA 文档明确将 `MATMUL_MLA` 的
`pd_context` capability 关闭，因此其 Attention 只能使用单独的 MLA 解析模型并标为
`analytical_only_mla`；若 MLA manifest 未冻结，则该模型输出 `partial_e2e_moe_path`，不画成
和另外五个模型同证据等级的实心柱。

### 2.3 64 GB 容量是物理门禁，不只是图注

四个 16 GB HBM stack 总容量只有 64 GB。训练的参数、梯度、optimizer state、activation，
以及推理时 P/D 多 instance 的参数和 KV cache 都必须做 byte-exact 容量审计。每个 case 输出：

```text
parameter_bytes, optimizer_bytes, gradient_bytes, activation_peak_bytes,
kv_cache_bytes, shared_weight_bytes, replicated_weight_bytes,
total_resident_bytes, hbm_capacity_bytes, capacity_status
```

默认评估两种明确模式：

- `global_shared_read_only`：推理的 P/D instance 共享四个 stack 上的一份只读权重，显式计入
  远端 HBM route 和带宽竞争；
- `replicated_per_instance`：每个 instance 独立权重副本，作为容量/带宽敏感性结果。

训练使用 AdamW 时必须按实际精度和是否 sharding 计算 state，不得复用现有 tiny Dense 的
`SGD_UPDATE` 内存量。若总容量超过 64 GB 且没有明确 offload/多 wafer 配置，仍可保留性能
投影，但状态必须是 `capacity_infeasible_projection` 并使用斜线纹理；不得称为可运行 E2E。

### 2.4 推理保留分项结果，并增加显式输出长度的联合评价

主结果分别报告：

- `TTFT`：单请求 prefill + PD KV handoff；
- `decode_step_latency(B)`：每个 D instance 对 batch 中每条序列生成 1 token；
- `system_decode_tokens_per_s = 2 × B × f_cycle / decode_step_cycles(B)`；
- `TPOT = decode_step_cycles(B) / f_cycle`，表示同步 batch 中单序列的每 token 延迟。

主推理图使用 `B=64/512` 两档 decode throughput 和 speedup；prefill/PD 作为同图 breakdown
或补充图。若要给定输出长度 `G` 的请求延迟，使用：

```text
T_request(B,G) = T_TTFT + G × T_decode_step(B)
```

G 必须作为显式输入参数。主联合结果冻结 `G=512`，输出 6 模型 × B={64,512} 共 12 个
派生 case，同时保留 decode 与 prefill/PD 分项文件和图。当前联合项使用固定 KV=36864
附近的 local TPOT 常数外推，不重放 512-token 生成过程中逐 token 的 KV 增长，必须带
limitation tag，不能替代分项结果或解释为严格请求 trace 回放。

### 2.5 理论下界是 max，不是 min

完成一次融合执行必须同时完成计算、通信和内存动作。理论下界使用资源/依赖共同约束：

```text
T_lower = max(
    longest_dependency_path,
    max_r(total_service_time_on_resource_r)
)
```

不使用 `min(T_comp,T_comm)`。理论与实际必须使用同一 padded runtime shape、HBM bytes、
route 和本地 reduction 工作量。

## 3. 可复用的现有仿真负载与正确用法

| 现有产物 | 已证明内容 | 本实验复用方式 | 不能直接复用的内容 |
|---|---|---|---|
| `llm/test/frontend/integration/run_flexible_mesh_release.py` 及 flexible-mesh release | Dense 完整 tiny train step、MoE infer/train、MeshSlice 在 8 个代表 mesh 上 48 cases/96 executions，双跑、ProgramIO 和十类 residual 全闭合 | 直接复用 runner、materializer、observer、repeatability 和 drain 合同 | P5 tiny workload 使用 1 MiB SRAM、低 HBM 带宽和 tiny H/I；旧 makespan 不是目标硬件绝对周期 |
| `notes/frontend/baselines/stage3-static-profile-v1` | PREFILL/DECODE/MIXED 的 exact Attention/KV/HBM 语义和 production 双跑 | 复用 profile、KV page、Attention action 和 runtime parser；生成目标 shape motif | TP1、H=16/I=32 tiny 周期不能直接缩放成目标结果 |
| `notes/frontend/baselines/stage4-pd-v1` | fused/PDS/PDR 的逐层 KV state transfer、one-to-one/reshard、D2D byte/link 闭包和双跑 | 复用 PD plan、state transfer lowering、runner 和 route 审计 | 现有 1～3 die、seq=8 的绝对周期只作结构 prior |
| `exps/exp1/exp1_1` | 6 模型 shape、shard-first padding、`128×512×256` SRAM-safe tile、T00/T10/T01/T11 schema、Q=8 回放框架 | 复用模型 shape 生成、tile/live-set、结果 schema 和 AG+GEMM/GEMM+RS signature | 当前 176 个结果均是 `analytical_resource_replay_fallback`，不能作为周期精确 anchor |
| `exps/exp1/exp1_2` | Dispatch/Combine 专用 DAG、top-k vector/reduction、compact/noncompact 逐有向链路路由 | 复用动作 DAG、route、HBM 审计字段和 MoE shape | 当前结果是 `operator_specific_dag_extrapolation`；其中 2000 TFLOP/s/die 不能复制到本实验 |

旧 evidence 只有在 `hardware/simulation/mapping/tool/workload` digest 全匹配时才允许作为直接
anchor。否则只作为结构 prior，并在本实验目标 binding 下重跑一个小锚点。选中的旧 evidence
应复制摘要和 digest 到 exp2 的 `calibration/source_evidence.json`，不能依赖临时 build 目录。

## 4. 目标硬件配置与启动前 unit closure

### 4.1 固定配置

- wafer：`6×6` dies；每 die `4×4` compute cores；
- control：每 die 1 个专用 control core，不占用 16 个 compute core；若当前 schema 只能从
  16 核中保留 control core，则必须把 worker 数改为 15 并重算峰值，禁止仍写 16×8 TFLOP/s；
- tensor peak：8 TFLOP/s/compute core，目标 128 TFLOP/s/die；
- SRAM：3 MiB/core，聚合目标带宽 256 GB/s/core；
- DTE：2 channels/core；单 core 注入上限 256 GB/s；
- D2D：每条有向 link 单向 1 TB/s，反方向独立；
- HBM：North/South 两边，每边 2 stack，优先放在列 1 和 4；16 GB/stack，
  256 GB/s/stack，总容量 64 GB、理想聚合 1 TB/s；
- routing：主结果固定 XY/X-first；其他 route 只能作为消融。

### 4.2 不能只改 JSON 标签

现有 P5/exp1 配置中的 `noc_payload_per_cycle`、`c2c.link_bw`、HBM cap、SRAM 和 compute
参数并不自动等于上面的论文级目标值。启动前生成 `hardware_unit_closure.json`，从 simulator
真实 cycle time 和 packet bytes 反算：

```text
B_link = packets_per_cycle × packet_bytes / cycle_time
B_noc  = noc_payload_per_cycle × packet_bytes / cycle_time
B_hbm  = configured_stack_bytes_per_second
R_core = isolated_GEMM_logical_FLOPs / measured_active_time
```

门禁要求：

- D2D、NoC、HBM 和注入上限的配置值、反算值、期望值三者一致；
- 一个 isolated GEMM 和一个 byte sweep 分别验证 tensor peak 与 HBM/DTE 饱和区；
- vector rate 由目标配置上的 SwiGLU/weighted-combine 小 sweep 校准，不复制 exp1-2 的
  60 TFLOP/s/die；
- 4 个 HBM stack 的地址范围无重叠，容量和边缘连接可由 manifest 重算；
- tile live set 加 runtime reserve 不超过 3 MiB/core。

任一门禁失败时先修硬件绑定，不能用解析式覆盖错误配置后继续出图。

## 5. 工作负载和物理 placement

### 5.1 训练

固定 `batch_size=1/rank`、`seq_len=2304/36864`、FP16 tensor 输入/权重、FP32 累加。
完整 step 包含 forward、loss/CE、backward、WGRAD、梯度同步、AdamW 和 state store。
主结果固定一种 activation checkpoint 策略，并在 workload manifest 中记录；推荐两档都使用
per-transformer-block checkpoint，使长序列不会因未建模 activation 常驻而显得虚假可行。

Dense placement：4 个紧凑 `3×3` TP group 构成 `2×2` DP：

```text
DP00 rows[0:3] cols[0:3]    DP01 rows[0:3] cols[3:6]
DP10 rows[3:6] cols[0:3]    DP11 rows[3:6] cols[3:6]
```

DP gradient collective 在四个 group 的相同 TP rank 之间进行，逐有向链路累计流量。

MoE placement：同样四个 `3×3` TP group 分别作为 4 个 EP rank；TP group 内紧凑，EP
dispatch/combine 跨四个 quadrant，属于非紧凑通信。Mixtral 主路径 `top_k=2`；DeepSeek-V3
主路径按 exp1-2 保持 `top_k=8`、只统计 routed expert，shared expert 单独列为 sensitivity。

训练主结果采用保守优化范围：只有 work1 当前有明确 fused lowering 的 forward
AG+GEMM、GEMM+RS、Dispatch+GEMM、GEMM+Combine 使用 overlap 时间；backward、WGRAD、
optimizer 保持两边完全相同。若以后 backward 也有生产 lowering，作为独立扩展结果，不把
forward speedup 直接复制给 backward。

按后续实验要求，训练扩展为三个同工作量状态，并以第三态作为新主结果：

- `base`：forward、backward、WGRAD 与 collective 均采用保守 barrier；
- `forward_only_overlap`：保留上述原方案，只放宽 forward fused pipeline；
- `full_train_overlap`：复用 forward 优化，并额外放宽 backward dX、WGRAD 与逐层
  gradient collective 的合法依赖。

三态的 operator、logical/runtime work、bytes、route、resource set、duration 和 evidence
必须逐 action 相同，只允许 dependency edge 不同。full-train 中下一层 dX 只等待上一层 dX，
不再等待该层 WGRAD/sync；每层 gradient sync 仍等待本层全部 WGRAD；AdamW 首个 tile 必须
等待所有 layer×flow 的 gradient sync。要求
`T_full_train_overlap <= T_forward_only_overlap <= T_base`，且 12 个训练主 case 的
full-train forward、backward、WGRAD phase 都必须严格短于 base；否则实验失败。

### 5.2 推理与 PD placement

六个 `2×3` instance 精确铺满 wafer：

```text
P0 rows[0:2] cols[0:3]    P1 rows[0:2] cols[3:6]
D0 rows[2:4] cols[0:3]    D1 rows[2:4] cols[3:6]
P2 rows[4:6] cols[0:3]    P3 rows[4:6] cols[3:6]
```

P0/P2 向 D0 handoff，P1/P3 向 D1 handoff；每个 D instance 同时接收上下两个 P instance，
因此必须把背景流叠加到相同有向链路，不用单 flow 带宽冒充系统带宽。

- Dense：每个 instance 内 TP=`2×3=6`；
- MoE：Attention/非 MoE dense block 使用 TP=6；expert 在同一 `2×3` footprint 内使用 EP=6，
  Dispatch/Combine 完成 layout 变换并计入成本；
- prefill：`seq=2304,batch=1`；
- decode：每步 query length=1，`batch=64/512`，累计 KV length=36864。

KV handoff 使用 Stage4 的逐层 K/V state transfer 口径。不能把 P instance 对本地 HBM 的
write 和 D instance 的 read 相加后称为跨 instance handoff。

#### Decode 保真度修正

Decode 不能沿用 prefill/训练的大矩阵和固定窗口假设，主回放必须满足：

- 展开连续两个 token step；step 1 的每个 D instance 只依赖自身 step 0 的
  `TOKEN_COMMIT`，D0/D1 不互相建立自回归依赖；
- 每层新 token 的 KV 必须显式写回四个 HBM stack，`TOKEN_COMMIT` 等待本 instance
  全部 layer append 完成，下一 token 再依赖该 commit；
- 使用同一 instance 的 `commit(t+1)-commit(t)` 估计固定 KV 上下文附近的 local TPOT，
  系统取 D0/D1 较慢者；不能混入 prefill/handoff，也不能声称两步已验证稳态收敛；
- 窗口数为 `min(8,ceil(batch/128))`，所以 B64=1、B512=4；没有独立 chunk 时不得凭空
  构造 overlap；
- dense tensor 的 `M=B`；MoE expert tensor 的 `M=B×top_k/touched_experts`。每个窗口
  按自身 token fraction 重新计算 dense/expert M，而不是把整批 M 复制到所有窗口；
  以 128 为物理 tile-M，计算 `u=min(1,max(1/128,M/128))`，再按两类 FLOPs 加权；
- KV-cache read 是 HBM boundary bytes，activation reducer 仅处理 `B×H×dtype`，
  两者不得共用 `activation_bytes`；
- 前置 AG/Dispatch collective 必须等待本窗口 HBM 数据 ready，不能只依赖 control 而
  提前启动；
- 小消息 collective 时间为 `2 cycle DTE launch + route_hops×1 cycle + bytes/rate`；
  HBM read/append 为 `10 cycle first-byte + bytes/rate`。常数来自
  `configs/target_hardware.json` 的 2 ns 周期、DTE gamma/tau、link latency 和 HBM
  behavioral latency，但在目标 unit closure 前必须标记 unvalidated；
- speedup 直接使用 base/overlap，不做正收益 clamp。B64 是预期可能为 1× 或负收益的
  主边界点，不能预设“晶圆一定转正”。

### 5.3 MoE 动态负载

主柱沿用 exp1-2 的 balanced routing，保证模型间可比；同时用

```text
lambda = max(tokens_per_expert) / mean(tokens_per_expert)
```

做 `1.0/1.25/1.5` 三档解析敏感性。若后续有真实 router trace，则以 trace 的 assignment
直方图替代该 sweep。MoE 的结论必须带 balanced 与 skew band，不能只报对称下界。

## 6. 端到端动作分解与回放

### 6.1 同工作量的两个状态

`T_base` 与 `T_overlap` 必须保持：

- 相同 logical/runtime shape、padding、tile 和 wave 数；
- 相同 16 compute cores、DTE channels、HBM placement 和 route；
- 相同 logical FLOPs、collective bytes、HBM boundary bytes、local NoC bytes；
- 只改变依赖边、barrier、buffer slot 和允许并发的 resource interval。

naive 不得退化成单核。也不允许 overlap 路径顺手删除一份 HBM materialization 后仍声称只测
overlap；若要评估 buffer elimination，另设第三个消融状态。

### 6.2 资源显式 DAG

每个 action 至少包含：

```text
action_id, phase, layer, operator, tile_id, deps,
logical_work, runtime_work, bytes, route,
resource_set, duration_source, evidence_signature
```

资源池至少区分：tensor、vector、每个 HBM stack、每个 HBM ingress path、每条有向
NoC/D2D link、每 core SRAM read/write port、DTE channel、control issue queue、reducer。
按 earliest-resource-ready 做离散事件回放：

```text
start(a)  = max(max(finish(dep)), max(available(resource)))
finish(a) = start(a) + duration(a)
```

算子 DAG 不共用一个 `max(Tcomp,Tcomm)+tail`：

```text
AG+GEMM:       AG chunk -> local handoff -> GEMM tile
GEMM+RS:       GEMM tile -> local reduce -> RS chunk
Dispatch+GEMM: pack/dispatch -> receive/wait -> expert GEMM -> SwiGLU
GEMM+Combine:  expert GEMM -> combine traffic -> top-k join -> weighted reduce
PD:            P KV store/ready -> sliced handoff -> D wait -> decode KV read
```

训练完整 step 的不可优化阶段直接复用 flexible Dense/MoE action 计数和依赖；模型规模只放大
真实 FLOPs/bytes/waves，不用固定比例猜 backward/optimizer 占比。

### 6.3 HBM 口径

沿用 exp1 的 `fused_boundary_tile_replay_v1` 作为 tile 重放下界，并增加四 stack 的全局地址
placement 和 route。时间取：

```text
T_hbm = max(
    max_stack(stack_bytes / stack_bandwidth),
    max_ingress_path(path_bytes / path_bandwidth),
    per_core_injection_time
)
```

不能把 36 个 die 都当作各有一条 256 GB/s 本地 HBM。`HBM-free` 只作为诊断图，不进入主结论。

### 6.4 大 shape 外推

周期精确 Q=8 motif 优先输出 tile completion marker：

```text
T_fill  = completion(tile_0)
II      = median(completion(tile_i)-completion(tile_{i-1}), i=2..7)
T_drain = program_done-completion(tile_7)
T(Q)    = T_fill + (Q-1)×II + T_drain
```

若相邻 II 的变异系数超过 5%，按 K-wave、route phase 或 buffer epoch 分段，不用一个全局 II。
若当前 signature 暂时不能输出 tile marker，则补跑同 payload 的 Q=4，并用 Q=4/Q=8 makespan
解固定项和稳态 II；该补跑计入 adaptive budget。

单资源 duration 使用两个 bracketing payload 锚定：

```text
duration_r(work) = setup_r + work / effective_rate_r
```

只从对应资源的事件或微基准拟合 `setup/rate`，不从一个 whole-program makespan 同时反推
GEMM、HBM、DTE 和 reduce 多个常数，避免不可辨识拟合。

## 7. 周期精确采样矩阵

目标是首轮不超过 30 个唯一 simulation config；每个 config 独立执行两次，编译产物按签名缓存，
即最多 60 次短 execution，而不是对 24 个完整大模型逐层仿真。

### 7.1 R0：复用现有完整负载做当前硬件 smoke（6 configs）

| ID | 复用入口 | 目标 |
|---|---|---|
| R0-1 | Stage3 PREFILL | 校准 prefill control/Attention/KV write 固定项 |
| R0-2 | Stage3 DECODE | 校准 decode KV read/append 与小 query 固定项 |
| R0-3 | Stage4 PDS | 校准 KV state handoff、wait 和 D2D session 固定项 |
| R0-4 | Flexible Dense Train 3×3 | 校准 full-step phase/action/control 比例，不直接放大其 tiny makespan |
| R0-5 | Flexible MoE inference 2×3 | 校准 Direct-XY dispatch/combine 生命周期 |
| R0-6 | Flexible MoE train 2×2 | 校准 four-way route、gradient、optimizer/state 生命周期 |

这些 case 使用原 production materializer/observer，但改用本实验硬件子网和统一 simulation binding。
若某个旧 evidence 已经与目标 binding digest 完全一致，可直接 retained replay 并跳过重新编译。

### 7.2 R1：Q=8 稳态算子 motif（16 configs）

Dense：

```text
2 meshes (3×3 train, 2×3 infer)
× 2 operators (AG+GEMM, GEMM+RS)
× 2 payload regimes (target-min/underfilled, target-max/saturated)
= 8 configs
```

MoE：

```text
2 topologies (2×2 noncompact train, 2×3 compact inference)
× 2 operators (Dispatch+GEMM, GEMM+Combine)
× 2 payload regimes (B64-class, saturated train/B512-class)
= 8 configs
```

两档 payload 从 24 个目标 case 的实际 segment/assignment 分布取包络端点，不用任意 toy bytes。
top-k 对 vector/join/reduction 的影响按显式 action 数放大，并在 R2 的 DeepSeek top-k=8 点直接校验。

### 7.3 R2：直接反事实校验（4 pairs = 8 configs）

每个代表场景直接各跑一次完整 capped one-layer composite 的 `T_base/T_overlap`：

1. Dense train，seq=2304；
2. Dense train，seq=36864；
3. DeepSeek routed MoE train，top-k=8、noncompact、长序列；
4. inference decode，B=64、KV=36864，包含 PDS wait 的最易负收益边界。

这里的 capped composite 保持目标 action mix、route、tile 和 buffer pressure，但只保留能在合理时间
内运行的一层/Q-window。它们用于验证回放，不用于替代完整模型 E2E。

### 7.4 自适应补点，而不是预先全扫

先运行 30 个唯一配置并做 leave-one-signature-out 检查。只有出现下列情况才补点：

- 直接校验相对误差超过 15%；
- 目标 case 落在两个 anchor 的包络之外；
- II 变异系数超过 5%；
- B=64 speedup 的误差区间跨越 1，无法判断正负；
- 新的 topology/schedule/tile/control-core 签名出现。

补点优先选误差最大或会改变论文结论的 case，不平均撒到六个模型。

## 8. 校准、误差传播与验收阈值

### 8.1 签名和缓存

缓存 key 至少包含：

```text
hardware_digest, simulation_digest, tool_digest,
mesh/group coordinates, operator_dag_version, route_policy,
tile_M/N/K, runtime padding, segment_bytes, top_k_class,
schedule_mode, injection_mode, HBM mode, control-core mode
```

模型名字和完整层数不进入签名；只要底层动作/shape 签名相同就复用 anchor。

### 8.2 误差定义

```text
relative_error = abs(T_replay - T_direct) / T_direct
```

验收：

- 单个 operator/capped composite 的中位误差不超过 8%；
- p95 不超过 15%；
- 任一点超过 20% 时，该签名不得发布，必须补点或标为 `analytical_only`；
- 所有周期精确运行两次 makespan/marker/digest 一致并完成 drain；
- B=64 的“正/负收益”只有在 speedup interval 完全位于 1 的同一侧时才下结论，否则写
  `inconclusive around 1.0×`。

### 8.3 E2E 区间传播

每个校准 signature 保存 direct residual 的分位区间。串行阶段做区间加法，overlap/max 节点做
区间 max，speedup 使用保守比值：

```text
speedup_low  = T_base_low  / T_overlap_high
speedup_high = T_base_high / T_overlap_low
```

MoE 再叠加 routing-skew band；MLA 和 capacity-infeasible projection 使用单独、更宽的模型区间，
不与 cycle-calibrated 区间混成一个来源。

## 9. 结果 schema 与图

每个主 case 至少输出：

```text
case_id, workload, model, seq_len, batch_size, kv_length,
placement, tp, dp, ep, model_manifest_digest,
logical/runtime shape, padding ratio, tile/wave counts,
compute/vector/HBM/local_NoC/D2D/control cycles,
prefill/handoff/decode or forward/backward/wgrad/optimizer cycles,
T_base_cycles, T_overlap_cycles, speedup,
theory_lower_cycles, theory_speedup, attainment,
uncertainty_low/high, capacity_status,
estimate_source, evidence_signatures, hardware/tool/simulation digests,
status, limitation_tags
```

输出建议：

```text
results/training_e2e.{json,csv}
results/training_operator_amdahl.{json,csv}
results/prefill_operator_amdahl.{json,csv}
results/inference_request_e2e.{json,csv}
results/inference_decode_e2e.{json,csv}
results/inference_prefill_pd_breakdown.{json,csv}
results/calibration_summary.json
results/capacity_audit.json
calibration/source_evidence.json
calibration/cycle_anchors.json
```

绘图：

- `training_e2e.svg`：6 模型 × 2 seq_len，共 12 组；每组只画相接的 base 与 full-train
  两根 tokens/s 柱，forward-only 仍保留在 CSV/JSON 和报告中；右轴同时绘制系统 full-train
  speedup 与算子平均 speedup 两条 exp1-1 风格折线，后者使用独立配色且位于图的上方；
- 算子均值映射：Dense 使用 exp1-1 Attention/MLP × AG_GEMM/GEMM_RS；Mixtral 再加入
  exp1-2 DISPATCH_GEMM/GEMM_COMBINE；DeepSeek 因无 MLA Attention anchor，只使用 exp1-1
  routed-expert MLP 两项和 exp1-2 两项。exp1-1 使用精确 3x3 mesh、2048/32768 作为
  2304/36864 的 1.125× 邻近序列代理；exp1-2 使用 H128/noncompact 和精确 seq_len。
  算术均值仅作描述性 Amdahl 对照，不作为时间加权系统预测；
- `inference_prefill_e2e.svg`：6 模型 × S={2304,36864}，共 12 组纯 prefill 延迟；每组画
  相接的 base/overlap 两根柱，同模型的长短序列靠近、模型间留较大间距，右轴画系统
  prefill speedup 与实际调用算子的未加权平均 speedup。Dense 取
  attention/MLP × AG_GEMM/GEMM_RS，Mixtral 再加入 DISPATCH_GEMM/GEMM_COMBINE，
  DeepSeek 仅取有 exp1 锚点的 routed-MLP 与 MoE 项并显式标记 MLA 缺失。exp1-1 使用
  精确 2x3 mesh、2048/32768 代理 2304/36864；exp1-2 使用 H128/compact/精确目标序列，作为连续
  2x3 P instance 的拓扑代理；原 `inference_prefill_pd_breakdown.svg` 继续保留 TTFT 分解；
- `inference_request_e2e.svg`：S=2304、G=512、6 模型 × B64/B512 的请求级联合
  output tokens/s 与 speedup；
- `inference_decode_e2e.svg`：6 模型 × B64/B512，共 12 组；柱为 system decode tokens/s；
- `inference_prefill_pd_breakdown.svg`：TTFT 中 prefill、KV handoff、wait 的 breakdown；
- 推理使用 naive/overlap 并列柱；训练使用 base/full-train 并列柱，不使用“naive+正增量”
  堆叠柱；训练中同一 case 的两根宽柱相接，同模型的两个序列长度靠近，模型之间保留较小
  分组间距；柱子采用暖灰/橙色，不复用蓝绿配色；
- 训练的 `T_base/T_full_train_overlap` 采用 exp1-1 风格的紫红色粗折线、圆角连接和白心圆点，
  不画 uncertainty；推理折线仍为 `T_base/T_overlap` 并带误差带；所有图保留 1.0× 水平线；
- `analytical_only_mla` 和 `capacity_infeasible_projection` 使用不同纹理；
- 不同图可归一化展示，但 CSV 必须保留绝对周期、时间和吞吐，图注说明不能跨图比较归一化高度。

## 10. 实施顺序

1. 生成六个完整 model manifest，并完成容量审计；先暴露不可运行 case；
2. 生成目标 `6×6` hardware binding 和 `hardware_unit_closure.json`；
3. 将现有 Stage3/Stage4/Flexible release adapter 接到目标硬件子网上，跑 R0；
4. 复用 exp1 tile/shape 代码生成 R1 Q=8 motif 和 completion marker；
5. 从 R0/R1 生成 `cycle_anchors.json`，实现资源显式 E2E DAG 回放；
6. 跑 R2 的 4 对直接校验，按阈值决定是否补点；
7. 生成 24 个主逻辑 case、12 个请求级派生 case、12 个算子均值对照 case、
   容量/来源/误差审计和四张图；
8. 运行 schema、公式、drain、digest、case 数和绘图回归测试。

## 11. 最终验收清单

- [ ] 主 case 精确为训练 12、推理 decode 12，无“8 模型”残留；
- [ ] 六个 model manifest 字段完整且有 digest；DeepSeek 不伪装成标准 Attention；
- [ ] 目标硬件反算带宽/算力与 plan 一致，control core/worker core 数明确；
- [ ] 每个 case 先做 64 GB 容量审计，infeasible case 明确标记；
- [ ] naive/overlap 使用相同工作量、16 compute cores、padding、HBM bytes 和 route；
- [ ] 训练三态逐 action 同工作量，只改变 dependency edge；
- [ ] 训练满足 full-train≤forward-only≤base，且 full-train 的 forward/backward/WGRAD
  phase 均严格短于 base；
- [ ] 所有成功周期精确执行双跑一致，ProgramIO、router、D2D、DTE、credit、state residual 为 0；
- [ ] HBM 只按 4 个 edge stack 建模，没有隐含 36 个本地 stack；
- [ ] local NoC/D2D 按逐有向链路负载取瓶颈，不把 payload 无条件乘 hop；
- [ ] T_base/T_overlap 不强制大小关系，B64 允许 slowdown；
- [ ] replay 中位误差≤8%、p95≤15%，>20% 的签名不进入主结论；
- [ ] 所有 speedup、attainment 和区间均由最终 cycles 重新派生；
- [ ] source 字段准确区分 direct、trace-calibrated、DAG extrapolation、MLA analytic 和
  capacity-infeasible projection；
- [ ] 主报告明确：现有 exp1 数值没有被当作本次周期精确 ground truth。

## 12. 预期可以回答的问题

按该方案，实验可以可靠回答：

1. 在相同工作量下，wafer 的 overlap 对训练完整 step 和 decode B64/B512 的净收益是否为正；
2. 训练的 Amdahl 稀释究竟来自 backward/WGRAD/optimizer，还是来自 HBM/control 关键路径；
3. seq=2304 与 36864 的差异有多少来自固定开销、稳态带宽和非紧凑 EP 拥塞；
4. B64 到 B512 的改善来自 FFN/MoE 小 M，还是 Attention/KV/PD handoff；
5. Dense 与 MoE 的差异在 balanced route 下有多大，以及 expert skew 会把结论推移多少；
6. 哪些大模型结果只是容量无关的性能投影，哪些确实能在 64 GB 单 wafer 上运行。

它不能在 MLA capability、完整大模型容量或动态 router trace 缺失时声称相应 case 已完成
完整周期精确 E2E；这些限制应保留为结果的一部分，而不是在绘图阶段隐藏。
