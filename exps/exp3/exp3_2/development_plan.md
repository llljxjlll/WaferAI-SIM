# Exp3.2 实验开发方案：单独验证 intra-die 优化

## 1. 实验目标与完成定义

Exp3.2 验证同一个问题在两类通信结构中的成立性：在硬件、逻辑工作量、跨 die
算法、跨 die 路由和跨 die 调度均保持不变时，intra-die 编排能否通过片内计算与
片内数据搬运的流水/重叠取得收益，并呈现由
`r = T_comp / T_local_move` 决定的非单调“山脊”形态。两个 operator suite 独立
报告，不能把绝对 speedup 互相比较：

- **GEMM+ReduceScatter（TP）**：固定 `N=12288`，35 个 `(M,K)` 主点；
- **Dispatch+GEMM（EP）**：固定 `H`，25 个 `(M_rank,I)` 主点，另有 4 个 H
  稳健性点和 5 个 down+Combine 方向抽检点。

主实验固定 GPT-3-175B hidden size `N=12288`、`D=6`、活跃 Die Mesh
`2×3`，扫描 35 个 `(M,K)` 点；每点运行一对同输入状态：

- `I0`：16 核 canonical split-K + barrier，片内计算、结果归约/搬运串行；
- `I1`：16 核 AUTO intra-die 编排，允许片内计算、LOCAL_NOC、SRAM/DTE
  staging 流水和重叠；无正收益时必须显式回退 `I0` 候选。

完成实验至少满足：

1. 两个 suite 的 69 对结果全部来自 production compiler、finalizer 和 NpuSim，不以解析式
   直接合成主结果；
2. 每一对只允许 intra-die refine/schedule 及其派生产物不同；
3. 固定 1D Ring 的候选、拓扑、chunk、路由、通信字节和 inter-die barrier
   在 `I0/I1` 间逐字段相同；
4. 每个状态重复运行 3 次，artifact、事件序列、makespan 和资源统计稳定；
5. 主图能由结果 JSON/CSV 一条命令重建，所有点均可追溯到配置、编译产物和
   simulator 原始日志；
6. 若扫描没有覆盖 `r≈1`，按预注册扩展规则补点，不得事后任意挑点。

本实验只形成 timing/performance 证据。若 GEMM primitive 仍不写数值输出，报告不得
宣称端到端数值正确性或真实芯片绝对性能。

## 2. 先冻结的实验语义

### 2.1 “inter-die 关闭”的精确定义

`plan.md` 中的“固定 1D Ring”包含两个容易混淆的层面：

- **算法层**：仍需 ReduceScatter，因此保留一个确定性的 Ring collective；
- **调度层**：禁用跨 die 计算—通信融合，不允许 Ring 与它之前的 GEMM/片内
  staging 跨阶段重叠。

本实验把 inter-die off 定义为：

```text
fixed Wang 1D Ring
+ topology = physical Hamiltonian ring
+ chunk_count C = D
+ unroll = 1
+ no inter candidate search
+ all local pre-ring work complete barrier
+ then execute Ring ReduceScatter
```

因此，Ring 只提供不可删除的跨 die 语义和确定性背景，不再贡献 inter-die 编排收益。
不能仅把 planner 的候选集限制为 Ring，却继续保留 GEMM—Ring 流水；那仍然开启了
inter-die 调度优化，会污染本实验。

`2×3` 活跃子网格在 `6×6` wafer 上固定使用物理 die：

```text
die ids = [0, 1, 2, 6, 7, 8]
rank order (row-major) = [0, 1, 2, 3, 4, 5]
Hamiltonian ring order = [0, 1, 2, 5, 4, 3]
```

closing edge `3→0` 是相邻 die 的物理链路。runner 必须验证候选恰好为
`WANG_1D_BIDIRECTIONAL/HAMILTONIAN_RING/C=6/unroll=1`，不允许 planner
成本排序悄悄换成 open line、2D MeshSlice、`C=2D/4D/8D` 或 unfused。

### 2.4 Dispatch+GEMM 的固定 inter 背景

Dispatch 是个性化 All-to-All，而非 ReduceScatter；不能把 2.1 的 Ring ReduceScatter
action 机械复用到 EP。Dispatch suite 仍固定 `D=6、mesh=2×3` 和同一物理 Hamiltonian
ring，但路由结果冻结、每个远端 token activation 沿 Ring 逐跳中继，且禁止 inter-die
candidate search 与计算—通信融合。I0/I1 必须使用同一 token-to-destination 映射、相同
每跳 bytes、相同环顺序和同一 `all-local-pre-ring-work-complete` barrier。

假定均匀远端目标分布且不发送本地 token 时，偶数环的精确平均远端最短跳数为
`H_avg=D²/[4(D-1)]`；主配置 `D=6` 为 `1.8` 跳。该量只作为固定 blocking Ring tail
的通信模型参数，不能被 I1 改变。未来若扩展到 `D=9/36`，相应值为 `2.5/≈9.26`，
但不属于本实验范围。
### 2.2 intra-die OFF/ON 的公平对照

| 状态 | 片内候选 | 16 核要求 | 调度语义 |
|---|---|---:|---|
| `I0` | `split_k_barrier` | 必须全部活跃 | compute → local reduce/move → staging，阶段间 barrier |
| `I1` | `split_k_barrier`, `split_k`, `split_k_tree_direct_dma` | 必须全部活跃 | AUTO 选择预计最短的合法流水；tie 时优先 `I0` |

两侧均固定 `split_k_parts=16`。AUTO 只能改变 die 内 task graph、核映射、局部
依赖、buffer lifetime 和 local transport；不能改变全局 GEMM、Ring plan、rank
placement、D2D 路由或通信字节。

pair validator 至少比较以下不变量：

- source IR1 digest；
- fusion partition 和 inter-die decision/plan digest；
- intra refine 前 projection digest；
- logical/runtime shape、valid/padded FLOPs；
- Ring topology、rank order、chunk count、unroll、route 和每 action bytes；
- workload、hardware、simulation、mapping、finalizer、NpuSim SHA-256；
- active die set 和每 die 16 个 runtime core。

### 2.3 计时边界

主结果同时报告两个口径，避免固定 Ring tail 把理论关系混入端到端结果：

```text
T_comp       = intra stage 的计算资源关键时间
T_move       = intra stage 的 LOCAL_NOC/SRAM/DTE staging 关键时间
T_intra_off  = I0 的 intra-stage makespan
T_intra_on   = I1 的 intra-stage makespan
T_ring       = barrier 后固定 Ring 的 makespan

stage_speedup = T_intra_off / T_intra_on
e2e_speedup   = (T_intra_off + T_ring) / (T_intra_on + T_ring)
r             = T_comp / T_move
ideal_speedup = (T_comp + T_move) / max(T_comp, T_move)
attainment    = stage_speedup / ideal_speedup
overlap_efficiency = (T_intra_off - T_intra_on) / min(T_comp, T_move)
```

主热力图优先画 `attainment`，并提供 `stage_speedup` 等值线；`e2e_speedup` 作为
工程端到端补充图。理论峰值 2 只适用于 intra stage 的理想二资源模型，不能直接拿
包含 `T_ring` 的总 makespan 对比。

## 3. 冻结实验矩阵

### 3.1 主矩阵

```text
D = 6
active mesh = 2×3
N = 12288
M = [256, 512, 1024, 2048, 4096, 8192, 16384]
K = [256, 1024, 4096, 16384, 65536]
states = [I0, I1]
repeat = 3
```

计数：`7×5=35` 个逻辑点、70 个状态、210 次正式 simulator 调用。编译器搜索中
simulator 调用必须为 0。

### 3.2 padding 与工作量口径

主矩阵中的 `M` 和 `K` 不能直接被 `D=6` 整除，rank-local `K/D` 也不一定能被
16-way split-K 整除。必须预先定义语义 padding，不能在失败点临时修改：

```text
M_runtime = align_up(M_logical, D)
N_runtime = align_up(N_logical, lcm(D, 16))
K_runtime = align_up(K_logical, D × 16)
```

这在 Exp3.1 的 D 维 padding 基础上，额外保证 `K_runtime/D` 可被 16 个片内核精确
切分。结果必须同时保存：

- `logical_shape` 与 `runtime_shape`；
- `valid_flops = 2MNK` 与 `padded_flops`；
- 各轴 padding 比例；
- 用于绘图的逻辑坐标，以及用于实际 `r` 计算的 runtime 工作量。

若现有精确 validator 还要求额外 tile 对齐，P0 必须一次性将规则加入 case schema 并
重新生成完整矩阵；不允许 lowering 内隐式 padding。

### 3.3 SRAM 可行性门禁

`M=16384,N=12288` 的完整输出远大于单 core 3 MiB，不能假定当前
`temporal_chunks=(1,)` 可运行。正式扫点前先对最小、中间、最大三点做 compile-only
capacity spike：

```text
(256, 12288, 256)
(2048, 12288, 4096)
(16384, 12288, 65536)
```

若最大点不满足 SRAM lifetime/capacity：

1. 增加确定性的 M-axis temporal tiling；
2. tile 规则取“满足 3 MiB live-set 的最大合法 tile”，由硬件配置和 typed buffer
   requirement 计算；
3. `I0/I1` 共享相同 tile 边界和 tile 数，只改变 tile 间依赖；
4. 保存 tile 规则、live-set high-water 和 spill=0 证据；
5. 所有 35 点重新生成，不能只修失败点。

不得通过增大 SRAM、允许 spill 或缩小逻辑 shape 绕过该门禁。

### 3.4 预注册 shape 稳健性检查

该检查仍固定在 `D=6、mesh=2×3`，且不混入 35 点主热力图：

1. **N 切片**：在主图脊线附近选择左/中/右 3 个预注册 anchor，比较
   `N=12288` 与 `N=4096`。后者使用 `M'=3M` 保持 `M×N` 相同；同时运行
   compute-only control，先排除 GEMM tile 利用率变化。

`plan.md` 中讨论的 D=9/D=36 曲线坍缩属于未来跨 mesh 扩展，不属于 Exp3.2 的
开发、运行或验收范围。本实验不生成其他 mesh，不以跨 D 坍缩作为结论前提。

## 4. 固定硬件配置落地
### 3.5 Dispatch+GEMM（EP）补充矩阵

该矩阵与 3.1 的 GEMM+RS 主图平行，但不是把 `(M×N,K)` 直接换名。EP 中每个 die
持有完整专家权重，local grouped-GEMM 的有效 FLOPs 为 `2×M_rank×I×H`，没有 TP
式的 `/D`；真实 `M_rank=S×topk/D` 在本热力图中主动解耦为自由轴。因此 `S`、`topk`
和 D 不作为扫描变量，D 只通过固定 Ring 的平均跳数影响 blocking inter tail。

```text
D = 6, active mesh = 2×3, gate/up-GEMM primary
H = 7168  (DeepSeek-V3 hidden size)
M_rank = [64, 256, 1024, 4096, 16384]
I = [512, 2048, 8192, 32768, 131072]
states = [I0, I1], repeat = 3
```

主图为 `5×5=25` 个 logical pairs。x 轴显示逻辑 `M_rank×H`（固定 H、实际扫描
`M_rank`），y 轴为 `I`；格内值仍是 `I0/I1` 的 `stage_speedup` 或 `attainment`。
对 gate/up，I0 为 `dispatch local staging → grouped-GEMM` 的串行 intra stage，I1
只允许这两个本地阶段分块流水。inter-die personalized A2A 在二者的相同 barrier 后执行。

它的工作强度应同时记录两个口径：

```text
r_local = T_comp / T_local_move
        = gamma·M_rank·I·H / (alpha_local + beta_local·M_rank·H)

T_ring_dispatch = H_avg·(alpha_ring + beta_ring·M_rank·H)
r_effective = T_comp / [H_avg·(alpha_local + beta_local·M_rank·H)]
```

`r_local` 用于判定 intra-die 山脊，`r_effective` 用于解释固定 Ring tail 的端到端稀释。
主结论比较 landscape 的山脊位置、两侧回落和小消息 cutoff；不得把它的数值高度与
GEMM+RS 直接横比，因为个性化 A2A 的 `H_avg=1.8` 已使 1D Ring 基线更悲观。

预注册补充点如下，均不混入 25 点主图：

1. **H 稳健性**：固定 `H=4096`（Mixtral），为保持主图的 `M_rank×H` 不变，运行
   `(M_rank,I)` 四角 `{112,28672}×{512,131072}`（即主图角点 M_rank 的 `7/4`），
   验证 `M_rank×H` 通信自由度的假设；
2. **方向检查**：down-GEMM+Combine 在 `H=7168` 运行五个点
   `(64,512),(64,131072),(1024,8192),(16384,512),(16384,131072)`。其依赖方向为
   `grouped-GEMM → combine staging`；组件量相同但 DAG 方向相反，用于验证主图不依赖
   gate/up 的先通信后计算方向。

Dispatch suite 共 `25+4+5=34` 对、68 个状态、204 次正式 simulator 调用；与原

新增一个确定性 hardware generator，而不是手写 36 die/144 HBM stack JSON。主配置
必须由 `plan.md` 映射为：

| 计划参数 | simulator/config 落点 |
|---|---|
| core mesh `4×4` | top-level `x=4,y=4` |
| wafer `6×6` | `die.x=6,die.y=6` |
| SRAM `3 MiB/core` | `memory.sram.capacity_bytes=3145728` 及同尺寸 region |
| `B_s=256 GB/s` | 结合冻结 cycle time 设置 SRAM port/`sram_bitwidth` |
| `N_PE=4096` | `exu_x=64, sa_cnt=1`，并核验 1 GHz 时为 8192 FLOP/cycle |
| DTE channels `2` | `dte_channel_count=2` |
| 256 GB/s DTE path | `dte_bit_width=2048` at 1 GHz |
| router/base | 固定 NoC/router pipeline 与 `noc_payload_per_cycle` |
| HBM `e_H=2,m=2` | 每 die 四个 16 GiB stack，固定 side/index/address range |

P0 必须补齐 `plan.md` 尚未给出的量：dtype、cycle time、HBM generation/带宽/延迟、
D2D link bandwidth/latency、`comp_util`、控制核模式的精确映射。尤其 `n_ctrl=1`
不能仅凭字段名猜成 `dual_dte_dedicated`；需用配置解析结果和 runtime marker 证明每 die
的控制资源数量。任何未冻结值都应 fail closed，而不是继承某个 test fixture 的默认值。

生成器输出：

```text
generated/hardware_exp3_2.json
generated/mapping_exp3_2.spec
generated/hardware_audit.json
```

`hardware_audit.json` 保存派生带宽、每 die stack 数/容量、总 endpoint 数、active
submesh、配置 SHA-256 和所有单位换算。

## 5. 实现架构与文件规划

```text
exps/exp3/exp3_2/
├── plan.md
├── development_plan.md
├── experiment_config.py       # 矩阵与所有冻结常量的单一数据源
├── hardware_factory.py        # 6×6 wafer、HBM、mapping 生成与审计
├── workload_builder.py        # exact synthetic GEMM+RS / Dispatch+GEMM typed workloads
├── fixed_ring.py              # 强制唯一 Ring 候选和 inter blocking barrier
├── runtime_adapter.py         # I0/I1 compile/finalize/simulate 与 checkpoint
├── resource_evidence.py       # timeline v2 解析、r/overlap 分解
├── result_schema.py           # pair invariant、provenance、fail-closed 校验
├── run_experiment.py          # 主矩阵/稳健性矩阵编排
├── analyze_results.py         # 拟合、ridge/cutoff/collapse 检查
├── plot_results.py            # 热力图与 speedup-vs-r
├── generated/
├── results/
├── figures/
└── tests/
    ├── test_experiment_config.py
    ├── test_hardware_factory.py
    ├── test_fixed_ring.py
    ├── test_pair_equivalence.py
    ├── test_resource_evidence.py
    └── test_end_to_end.py
```

### 5.1 工作负载入口

现有 analytic LLM spec 会把 `H/I` 与模型层 shape 绑定，无法自然表达任意
`(M,12288,K)`。Exp3.2 应构造一个只含 exact GEMM+RS 的 typed synthetic operator
workload，并从 IR0/IR1 后继续走现有 placement、fusion、Swizzle、common IR2、
intra refine/schedule、GlobalAction、lowering、link、finalizer 和 NpuSim 链路。

该路径必须标记 `evidence=synthetic_operator_production_runtime`。不能为了得到任意 shape
而伪装成某个模型层，也不能退回 Exp3.1 的解析 resource replay 作为主证据。

### 5.2 固定 Ring adapter

复用 production `SwizzlePlanner` 和 Wang generator，但增加实验所需的严格 adapter：

1. constraints 只允许 `WANG_1D_BIDIRECTIONAL`；
2. 从 ranked candidates 中筛出唯一
   `HAMILTONIAN_RING/C=D/unroll=1` 候选；
3. 用已有 forced deployment API 物化该 candidate；
4. 在 typed projection 边界加入 all-local-complete → ring-start barrier；
5. validator 证明 barrier 没有删除计算/通信 action，且不存在跨 barrier 的 Ring
   SEND/RECV；
6. 将 decision、deployment selection、plan 和 barrier digest 写入 pair evidence。

若找不到或找到多个 exact candidate，case 直接失败，不允许 fallback。

### 5.3 intra pair runner

复用现有 `IntraDieOptimizationOptions` 和 16-core performance runner 的对照思想，新增
支持 exact operator workload 与固定 inter plan 的 adapter。每个 logical case 先产生
一次 immutable pre-intra bundle，再分别运行：

```text
I0: FORCE split_k_barrier(parts=16, compute_groups=16)
I1: AUTO  bounded candidates(parts=16, require_full_compute_groups=true)
```

搜索仍使用解析 timing model，最终候选各运行 3 次 simulator。runner 支持
`--resume`，但只有在 case config、工具和所有输入 digest 完全一致时才可复用已有状态；
不匹配时 fail closed，不覆盖旧证据。

## 6. 资源计时与理论检验

现有 `intra_die_resource_evidence/v1alpha1` 把 `overlap_cycles` 标为 unsupported，且
解析器硬编码 `CYCLE_NS=2`，不足以支撑本实验。需要版本化 v2 marker，按
`case/region/die/resource` 输出：

```text
compute interval union
LOCAL_NOC interval union
SRAM/DTE staging interval union
compute-local overlap interval union
intra stage begin/end
ring stage begin/end
resource busy/stall cycles
critical-path action ids
```

时间换算必须来自本次 simulation config，不得硬编码。三次 repeat 的 marker 规范化后
digest 必须相同。

对每个点优先用同一次正式运行的显式 interval 计算 `T_comp/T_move/overlap`。另运行少量
预注册 isolated calibration（small/mid/large 三个点的 compute-only、move-only、
ring-only），用于检查 marker 分解而非替代主结果。

拟合仅用于解释和画理论 ridge：

```text
T_comp ≈ gamma × padded_flops / effective_peak(shape)
T_move ≈ alpha + beta × actual_local_bytes
r = T_comp / T_move
```

`actual_local_bytes` 必须取 action/trace 中的字节，不得直接假设为 `M×N`。若在固定 D
下它与 `M×N` 只差常数，可在报告中吸收到 beta。未来若另做跨 D 实验，必须重新恢复精确 D 因子。

ridge 由 `r=1` 定义。若忽略 shape efficiency 且
`actual_local_bytes=cMN`，理论上：

```text
K_ridge = D × effective_peak / compute_constant
          × (alpha/(M×N) + beta×c)
```

`plan.md` 第 68--70 行缺失的 ridge 方程应在正式报告中用带单位、带拟合参数的版本
补全，不能直接使用无量纲比例式。

## 7. 结果数据契约

每个状态至少记录：

```text
case_id, state, M, N, K, D, active_mesh
logical_shape, runtime_shape, padding
valid_flops, padded_flops
intra_mode, selected_candidate, candidate_table
active_dies, active_cores_by_die
ring_algorithm, topology, rank_order, chunk_count, unroll
ring_routes, ring_logical_bytes, ring_physical_hop_bytes
T_comp, T_move, T_intra, T_ring, total_cycles
resource_busy_cycles, resource_stall_cycles, overlap_cycles
repeat, repeat_signatures
source_ir1_digest, inter_plan_digest, pre_intra_projection_digest
artifact_sha256, config/input/tool SHA-256, git_commit
evidence, warnings
```

每个 pair 额外记录：

```text
baseline_state=I0, optimized_state=I1
pair_equivalence_status
stage_speedup, e2e_speedup, r
ideal_speedup, attainment, overlap_efficiency
no_regression
```

输出：

```text
results/exp3_2_results.json       # 完整记录
results/exp3_2_results.csv        # 作图扁平表
results/audit.json                # 配置、pair invariant、异常与 provenance
results/ridge_fit.json            # 拟合、置信区间和 scan coverage
results/anomalies.json            # fallback、回归、marker/shape 异常
```
results/dispatch_gemm/exp3_2_dispatch_gemm_results.json  # EP 独立完整记录
results/dispatch_gemm/exp3_2_dispatch_gemm_results.csv   # EP 作图扁平表
results/dispatch_gemm/audit.json                          # EP pair invariant 与限制

正式结果中不允许 NaN、缺状态、未标记 fallback、隐式外推或无法定位原始日志的点。

## 8. 分析与图表

### 8.1 主图

- x 轴：逻辑 `M`（log2）；
- y 轴：逻辑 `K`（log2）；
- 色值：`attainment`；
- 等值线：`stage_speedup`；
- 叠加：实测/拟合 `r=1` ridge、真实模型常见 `K=4K--12K` 区域；
- 格内可选标注：AUTO candidate 或 fallback 标记。

补充图包括：

1. `stage_speedup-vs-r`，叠加理想包络 `(r+1)/max(r,1)`；
2. `e2e_speedup` 热力图，展示固定 Ring tail 对工程收益的稀释；
3. `T_comp/T_move/T_ring/overlap` 分解；
4. 小 message cutoff：以 `actual_local_bytes` 为 x 轴定位最低有效粒度；
5. 固定 `2×3` mesh 下的 N shape control 图。

### 8.2 预注册判定规则

- ridge coverage：至少存在一组 K 方向相邻点跨过 `r=1`，且主矩阵同时含
  `r<0.5`、`0.5≤r≤2`、`r>2` 三个区间；
- 若所有点 `r<1` 且 speedup 随 K 上升，按 4 倍扩展 K：`131072,262144`，先做
  compile-only/SRAM 检查，再补测全部 M；
- 若所有点 `r>1`，向下补 `K=64,128`；
- 扩展只由 r coverage 触发，补点单独标记 `adaptive_extension=true`；
- “山脊存在”以 speedup 对 `|log2 r|` 两侧回落及拟合置信区间判定，不凭肉眼；
- AUTO 回退不是缺失值：speedup=1，并保留选择原因。

## 9. 开发阶段

### P0：冻结契约与可表示性

- 补齐 dtype、clock、HBM、D2D、control-core 等未指定硬件量；
- 确认 6×6 wafer 上显式选择 2×3 active submesh；
- 冻结 padding、fixed Ring、inter blocking barrier 和两状态语义；
- 生成 35 个 GEMM+RS 与 34 个 Dispatch+GEMM case manifest，并检查各自 logical/runtime FLOP 守恒。

退出条件：所有输入均有 schema 和 digest，无字段依赖隐式默认值。

### P1：最小纵切与容量门禁

- exact synthetic GEMM+RS 和 personalized Dispatch+GEMM 从 typed workload 编译到 linked artifact；
- 强制 `2×3` physical Ring `C=6/unroll=1`；
- 在 small/mid/large 三点编译 `I0/I1`；
- 关闭 inter overlap，并用 DAG 检查证明 barrier 生效；
- 若需要，完成双方共享的确定性 M temporal tiling。

退出条件：三点 16 核/die 全活跃、3 MiB SRAM 合法、无 spill、无 fallback。

### P2：资源证据 v2

- 增加显式 intra/ring stage 与资源 interval marker；
- 实现 interval union/overlap/critical-path 解析；
- cycle time 从配置读取；
- isolated calibration 与 marker 分解在容差内一致。

退出条件：能够从一个 pair 独立重算 `r`、speedup、attainment 和 overlap efficiency。

### P3：公平 pair runner

- 实现 immutable pre-intra bundle 的 I0/I1 分叉；
- 加入全部 pair invariant、三次 repeat、checkpoint/resume；
- 保留 candidate table、选择原因、artifact 和原始 stdout；
- 异常或回归 fail closed，但仍将诊断写入独立目录。

退出条件：small/mid/large 三对结果可重复，任意篡改输入或 Ring plan 都被拒绝。

### P4：两类 operator suite 主扫描

- 先 compile/preflight GEMM+RS 的 35 点与 Dispatch+GEMM 的 25 个主点、9 个预注册补充点；
- 再执行合计 69 对、138 状态 × 3 repeats；
- 每完成一个 pair 原子写 checkpoint；
- 汇总 coverage、fallback、异常和运行预算。

退出条件：两类 suite 分别齐全，pair equivalence 与 repeat stability 全部通过。

### P5：分析、补点与稳健性

- 生成主热力图、speedup-vs-r 和资源分解；
- 按预注册规则决定是否扩展 K；
- 运行 GEMM+RS 的 N shape control，以及 Dispatch 的 H=4096 四角和 down+Combine 五点检查；
- 所有补点和 shape control 继续固定 `D=6、mesh=2×3`。

退出条件：ridge/cutoff 结论有机器可检查的统计和 limitation，不只是一张图。

## 10. 测试与最终验收

### 10.1 单元测试

- 恰好生成 GEMM+RS 的 35 个主 case/70 个状态，以及 Dispatch 的 25 个主 case、9 个补充 case/68 个状态；
- 所有 runtime shape 满足 D 分片与 16-way split-K；
- valid/padded FLOPs、输出字节和 RS 字节守恒；
- 2×3 ring order 恰为 `[0,1,2,5,4,3]` 且每边一跳；
- 非 exact Ring candidate 或多个 exact candidate 均失败；
- inter barrier 前后无跨阶段 action；
- I0/I1 任一冻结字段变化均触发 pair validator；
- interval overlap 解析覆盖相交、相切、嵌套和多 core union；
- 配置单位和 cycle time 变化能正确传播，不存在硬编码 2 ns。

### 10.2 集成/E2E 测试

- small/mid/large 三点 compile/finalize/simulate；
- 每 die 恰好 16 个 active worker core，active die 恰好为冻结 6 个；
- address lifecycle、SRAM lifetime/capacity、route、transport control 全部 pass；
- I0 明确选择 barrier candidate，I1 的选择来自有界 AUTO；
- product search 的 simulator 调用数为 0；
- 三次正式运行签名一致；
- 通信字节设为 0 的测试中 speedup 回到 1；
- 禁止 intra overlap 时 I1 与 I0 makespan 相同；
- 理想化完全重叠 fixture 中 `T_intra_on=max(T_comp,T_move)`。

### 10.3 正式验收

Exp3.2 只有同时满足以下条件才算完成：

1. GEMM+RS 的 35 对与 Dispatch+GEMM 的 25 对主图、9 对预注册补充 I0/I1 结果齐全；
2. 对照只改变 intra-die 选择及其派生 schedule/artifact；
3. fixed Ring 与 inter blocking 语义有 typed DAG 和 runtime marker 双重证据；
4. 硬件配置逐项对应 `plan.md`，补充参数均显式记录；
5. 所有点 SRAM 合法、无未标记 padding/spill/fallback；
6. 资源 marker 足以独立重算全部主指标；
7. 3 次 repeat 稳定，测试与 pair audit 全通过；
8. ridge 若在原网格外，已按预注册规则补点或明确报告未覆盖，不能仍声称观察到峰值；
9. 图、表、JSON、CSV 和结论可由记录命令与 digest 完整复现。

## 11. 计划中的命令行

```bash
# 生成并审计硬件、mapping 和 35-case manifest
python3 -B exps/exp3/exp3_2/experiment_config.py emit \
  --output-dir exps/exp3/exp3_2/generated

# 全矩阵只编译/校验，不运行 simulator
python3 -B exps/exp3/exp3_2/run_experiment.py preflight \
  --generated-dir exps/exp3/exp3_2/generated

# 正式主扫描，可安全续跑
python3 -B exps/exp3/exp3_2/run_experiment.py run \
  --matrix main --repeat 3 --resume \
  --output-dir exps/exp3/exp3_2/results

# 校验、分析与作图
python3 -B exps/exp3/exp3_2/analyze_results.py \
  --results exps/exp3/exp3_2/results/exp3_2_results.json
python3 -B exps/exp3/exp3_2/plot_results.py \
  --results exps/exp3/exp3_2/results/exp3_2_results.json \
  --output-dir exps/exp3/exp3_2/figures
```

# Dispatch+GEMM 解析预检与主热力图（正式版替换为同一契约的 NpuSim runner）
python3 -B exps/exp3/exp3_2/run_dispatch_experiment.py \
  --output-dir exps/exp3/exp3_2/results/dispatch_gemm
python3 -B exps/exp3/exp3_2/plot_dispatch_results.py \
  --results exps/exp3/exp3_2/results/dispatch_gemm/exp3_2_dispatch_gemm_results.json \
  --output-dir exps/exp3/exp3_2/figures

## 12. 实现前必须关闭的开放项

以下内容不能由代码静默猜测：

1. 主实验 dtype，以及 accumulation/rounding contract；
2. cycle time 与 256 GB/s 到 bytes/cycle/bitwidth 的换算；
3. HBM profile、每 stack 带宽/延迟及两条放置边的精确 side/index；
4. D2D link 的目标带宽、latency、buffer depth；
5. `n_ctrl=1` 在当前 control-core schema 中的准确含义；
6. 大 shape 的 canonical temporal tile 规则是否需要新增；
7. 主色值最终选择 attainment 还是 stage speedup。无论选择哪一个，另一项必须保留在
   数据和补充图中。

推荐先关闭 1--6 再开始 P1；第 7 项不阻塞主矩阵运行，但必须在出图前冻结。
