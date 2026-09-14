# Exp3.1 实验开发方案：单独验证 inter-die 优化

## 1. 实验目标

Exp3.1 用于回答一个单一问题：在保持算子总工作量、逻辑输入和硬件拓扑不变的条件下，inter-die Swizzling 是否能够通过计算与 D2D 通信重叠带来稳定收益。

实验覆盖 `D ∈ {6, 9, 36}`、短/长两个序列长度以及 GEMM+ReduceScatter 和 Dispatch+GEMM 两类算子，共 48 个逻辑实验点。每个逻辑点只生成三组可比较结果：

1. `native_full`：同为单 die 16 核，`W00` 使用 canonical baseline，`W11` 使用 Exp1 派生的 intra-die 与 inter-die 优化实现；
2. `native_inter_only`：固定 16 核和同一种基础片上实现与 HBM 工作量，只开启 inter-die 流式编排；
3. `gpu_inter`：die-local 计算延迟替换为实测 GPU GEMM 延迟，只开启 inter-die 优化。

三条主曲线采用以下固定 baseline：

```text
native_full:       W00 / W11
native_inter_only: C00 / C10
gpu_inter:         G00 / G10
```

`native_full` 的 `W00/W11` 均使用 16 核：W00 固定为 `Pm×Pn×Pk=4×4×1` canonical baseline，W11 则使用 Exp1 对齐的 adaptive 16-core intra schedule 加 inter-die streaming；两侧采用相同基础 HBM 工作量。`native_inter_only` 的 `C00/C10` 都固定为 canonical 16-core 映射、相同基础计算/NoC/HBM 实现，只改变 inter 路径的串行或流式编排。MoE 的 W 轨道复用 Exp1.2 两个 16-core schedule，C 轨道复用其固定 16-core 基线 grouped-GEMM 映射。`gpu_inter` 的 baseline 和 optimized 都使用同一 GPU LUT 与同一 decomposition，只改变 inter 调度，保证相对 speedup 公平。

### C00/C10 的 D2D 端口拥塞模型

`C00/C10` 不能把每个 core 看成直接连接 D2D。每个 die 的远端 payload 均分到 16 个 core，并固定走 `4×4` core mesh 上的 X-first 路由，汇聚到两个边缘 DTE 端口 `(0,1)`、`(3,2)`；每条 NoC 链路和每个 DTE channel 均按 `256 GB/s` 计。

- `C00` 串行计入 source core→port NoC、source port service、既有 Exp1 D2D fabric、destination port service 与 destination port→core NoC。
- `C10` 保持完全相同的 16-core 映射、端口位置、payload、路由和基础 HBM 工作量，仅允许分段流式重叠；其稳态瓶颈为共享 local-NoC、端口服务与 D2D fabric 的最大值，并保留填充/排空项。
- 因而 inter-only 的 speedup 不会假设隐藏的 all-core 注入 crossbar；任何 core→D2D-port 的片上竞争都体现在 `inter_port_time_ns`，而 `hbm_time_ns` 在 C00/C10 中相同。
- `W00/W11` 同样走显式端口模型，以保证完整指标中两侧均为 16 核的可比实现。

`Wxy` 是 Exp1 兼容状态（第一位为 inter、第二位为 intra）；`Cxy` 是固定 16 核的受控 inter 消融状态；`Gxy` 的第二位固定为 0，因为 GPU 实测延迟已经整体替代 die-local 计算，不再叠加 wafer intra-die 优化。

GPU 方案的含义是“实测 GPU 本地计算 + 模拟的 wafer D2D 通信与依赖关系”，不是实际多 GPU 系统的端到端测量。报告和图例必须明确标记这一证据边界。

## 2. 冻结实验矩阵

### 2.1 拓扑与输入规模

| `D` | 固定 mesh `(Px, Py)` | 序列长度 `S` |
|---:|---:|---:|
| 6 | `(2, 3)` | `2304`, `36864` |
| 9 | `(3, 3)` | `2304`, `36864` |
| 36 | `(6, 6)` | `2304`, `36864` |

`D=6` 主实验固定使用 `(2,3)`，不把 `(3,2)` 作为额外实验点。若后续需要方向敏感性分析，应通过独立开关增加 8 个转置 RC GEMM 查找项，并单列为补充实验。

### 2.2 GEMM+ReduceScatter

每个模型取两层，每层取两个序列长度：

| 模型 | 层 | 全局 GEMM `(M,N,K)` |
|---|---|---|
| LLaMA-2-7B | O-proj | `(S, 4096, 12288)` |
| LLaMA-2-7B | Down-proj | `(S, 4096, 11008)` |
| GPT-3-175B | O-proj | `(S, 12288, 36864)` |
| GPT-3-175B | Down-proj | `(S, 12288, 49152)` |

O-proj 使用 `K=3H`，Down-proj 使用各模型自己的 MLP intermediate size：LLaMA-2-7B 为 `11008`，GPT-3-175B 为 `4H=49152`。shape 生成器和 GPU 测量清单必须以本表为准。

### 2.3 Dispatch+GEMM

| 模型 | `H` | `I` | `E` | `topk` |
|---|---:|---:|---:|---:|
| Mixtral-8×7B | 4096 | 14336 | 8 | 2 |
| DeepSeek-V3 routed expert | 7168 | 2048 | 256 | 8 |

每个模型测两个 GEMM 阶段和两个序列长度 `S∈{2304,36864}`：

| 阶段 | 单次 GEMM shape | 执行次数 |
|---|---|---:|
| Dispatch + gate/up | `(M,I,H)` | 2 |
| down + Combine | `(M,H,I)` | 1 |

gate 和 up 是两个同 shape GEMM：GPU 只测一次 lookup key，resource DAG 中必须执行两次。DeepSeek 只覆盖 routed-expert MoE-FFN，不包含 MLA 或 shared expert。

### 2.4 逻辑点计数

```text
GEMM+RS:       2 models × 2 layers × 2 S = 8
Dispatch+GEMM: 2 models × 2 stages × 2 S = 8
每个 D:       8 + 8 = 16
总计:         3 D × 16 = 48 logical cases
```

120 个 GPU GEMM 查找项是微基准覆盖集合，不是 120 个逻辑实验点。一个逻辑点可能查询 coarse、1D Ring、2D RC 等多个候选形状。

## 3. GPU 实测 YAML 契约

本轮完整扫描模板同时固定在：

```text
exps/exp3/exp3_1/gpu_gemm_shapes.yaml
```

模板中的第二项为 `null`；GPU 测量完成后将其替换为对应的 `latency_ns`，其余 shape 和分组不得手工改动。

### 3.1 推荐文件位置

用户提供的正式数据放在：

```text
exps/exp3/exp3_1/inputs/gpu_measurements.yaml
```

仓库中只提交可公开的数据和元信息；若测量数据不能提交，应至少提交去除 latency 的模板、数据摘要和 SHA-256。

### 3.2 YAML 格式

查找键严格采用 `(M,N,K)` 顺序，表示：

```text
A[M,K] × B[K,N] -> C[M,N]
```

建议 schema：

```yaml
schema_version: 1
units: ns

gpu:
  name: "GPU model"
  count: 1
  clock_policy: "locked-or-recorded"

software:
  driver: "..."
  cuda: "..."
  cublas_or_backend: "..."

gemm:
  input_dtype: bf16
  output_dtype: bf16
  accumulation_dtype: fp32
  transpose_a: false
  transpose_b: false

measurement:
  statistic: p50
  warmup_iterations: 50
  measured_iterations: 200
  synchronization: per_iteration
  operands_resident_on_device: true
  includes_host_to_device: false
  includes_device_to_host: false

moe_profiles:
  mixtral_8x7b: {hidden_size: 4096, intermediate_size: 14336, experts: 8, topk: 2}
  deepseek_v3: {hidden_size: 7168, intermediate_size: 2048, experts: 256, topk: 8}

lookup:
  gemm_rs_coarse:
    - [[2304, 4128, 2048], 12345]
  gemm_rs_1d_ring_c_eq_d:
    - [[2304, 688, 2048], 2345]
  gemm_rs_2d_row_column:
    - [[2304, 1376, 6144], 3456]
  dispatch_gemm_coarse:
    # Mixtral up/gate; this one measured latency is consumed twice.
    - [[768, 14336, 4096], 4567]
    # Mixtral down.
    - [[768, 4096, 14336], 4321]
  dispatch_gemm_source_expert_chunk:
    - [[96, 14336, 4096], 567]
    - [[96, 4096, 14336], 543]
```

第二个字段是正整数或正浮点数 latency，单位由顶层 `units` 指定，主实验只接受 `ns`。若原始工具输出 p50/p95/均值等多列，应保留原始文件，导入脚本只把选定的 p50 规范化为上述标量格式。

### 3.3 严格校验规则

加载器必须在运行前完成以下校验：

- `schema_version`、单位、GPU、软件栈、dtype 和测量方法完整；
- shape 是三个正整数，latency 是有限正数；
- 不允许 `null`、符号维度、字符串 latency 或隐式单位换算；
- 同一个 `(M,N,K)` 可以被不同语义分组引用，但若重复出现，其 latency 必须完全一致；
- 默认要求 120 个必需项全部命中；不允许对主结果做插值、外推或最近邻替代；
- 缺项直接失败，并打印缺失项及其来源 case；
- 多余项默认报警但允许存在；`--strict-extra-shapes` 可将其升级为错误；
- 将规范化后的 YAML 内容、必需 shape 集合和运行配置分别计算 SHA-256，写入结果 provenance。

如果开启 `D=6 --include-transposed-mesh`，必需项从 120 增加到 128；该模式不能与主结果混在同一条曲线中。

## 4. Shape 生成与查找表构建

### 4.1 单一数据源

所有 shape 必须由 `case_matrix.py` 根据冻结实验矩阵生成，不能在 runner、绘图脚本和 YAML 校验器中分别手写。生成器输出：

```text
required_gpu_shapes.yaml       # 给 GPU 微基准执行者
logical_cases.json             # 48 个逻辑点及其候选算法映射
shape_coverage_report.json     # 去重前/后数量和引用关系
```

### 4.2 Dense GEMM 公式

对于全局 GEMM `(M,N,K)`：

```text
coarse local GEMM:  (M, N,    K/D)
1D Ring local GEMM: (M, N/D,  K/D), C=D
2D RC local GEMM:   (M, N/Py, K/Px), Px×Py=D
```

非整除维度沿用现有模拟器的语义 padding 规则。case 中同时记录：

- `logical_shape`：原始数学维度；
- `runtime_shape`：实际提交给 GPU 的 padding 后维度；
- `valid_flops` 与 `padded_flops`；
- padding 的来源和对结果的影响。

GPU LUT 使用 `runtime_shape` 精确查找。理论工作守恒和报告中的模型规模使用 `logical_shape`，避免把 padding 静默计入有效工作。

### 4.3 Dispatch+GEMM 公式

令 `M_rank=S×topk/D`。最小粗粒度等效 GEMM 为：

```text
up/gate: (M_rank, I, H), execution_count=2
down:    (M_rank, H, I), execution_count=1
```

令 `M_chunk=S×topk/(D×E)`。inter-die fused source-expert chunk 为：

```text
up/gate: (M_chunk, I, H), execution_count=2×E
down:    (M_chunk, H, I), execution_count=1×E
```

整数结果按现有 token padding/路由规则生成 runtime shape。上述执行次数保证 gate/up 的逻辑 FLOPs为 `4×S×topk×H×I`，down 的逻辑 FLOPs为 `2×S×topk×H×I`。1D Ring 和 2D RC 不能机械套用到个性化 all-to-all；MoE 只使用 coarse 与 source-expert chunk 两组查找项。

Mixtral 在 `D=9/36` 时有 `E<D`，因此 `M_rank=S×topk/D` 不是单副本纯 EP 的物理 expert-home 布局。本轮采用 `balanced_equivalent_rank_replay`，用于验证调度和 D 衰减结构；结果必须标记 `production_expert_placement_closed=false`。

### 4.4 数量不变量

主配置的期望数量为：

| 分组 | 条目数 |
|---|---:|
| `gemm_rs_coarse` | 24 |
| `gemm_rs_1d_ring_c_eq_d` | 24 |
| `gemm_rs_2d_row_column` | 24 |
| `dispatch_gemm_coarse` | 24 |
| `dispatch_gemm_source_expert_chunk` | 24 |
| 合计 | 120 |

这里的分组条目数保留语义引用；全局 LUT 构建时按 `(M,N,K)` 去重。若跨分组出现相同 shape，必须共享同一个测量值，同时在 coverage report 中保留全部反向引用。

完整的 120 项数值清单见同目录的 `gpu_gemm_shapes.yaml`。该文件按 `(M,N,K)` 存储，其中 Dense 三组由语义 padding 后的 runtime shape 生成，MoE 两组使用冻结的 Mixtral/DeepSeek 真实 `H/I/E/topk`。

## 5. 代码与目录规划

计划新增以下文件；文件名是开发接口的一部分：

```text
exps/exp3/exp3_1/
├── plan.md
├── development_plan.md
├── gpu_gemm_shapes.yaml
├── case_matrix.py
├── gpu_lut.py
├── resource_replay.py
├── run_experiment.py
├── plot_results.py
├── result_schema.py
├── inputs/
│   ├── gpu_measurements.example.yaml
│   └── gpu_measurements.yaml
├── generated/
│   ├── required_gpu_shapes.yaml
│   ├── logical_cases.json
│   └── shape_coverage_report.json
├── results/
│   ├── exp3_1_results.json
│   ├── exp3_1_results.csv
│   └── audit.json
├── figures/
└── tests/
    ├── test_case_matrix.py
    ├── test_gpu_lut.py
    ├── test_resource_replay.py
    └── test_result_schema.py
```

`generated/`、`results/` 和 `figures/` 是否提交由现有仓库规范决定；但每次正式运行都必须可由输入和命令完全重建。

## 6. 执行架构

### 6.1 原则：只替换本地计算时长

GPU 路径沿用与 native 路径相同的：

- inter-die 算法候选及 action DAG；
- D2D 链路带宽、路由、资源互斥和启动依赖；
- token/矩阵切分、padding 和工作量；
- inter on/off 的通信 action 集合。

唯一替换项是 die-local GEMM action 的 duration：native 路径使用 wafer cost model，GPU 路径使用精确 `(M,N,K)` LUT 延迟。GPU latency 已经包含单卡 kernel 的计算和本地 HBM 行为，不能再叠加 wafer compute 或 wafer HBM 时间，也不能包含 NCCL、PCIe、NVLink 或主机传输。

所有内部时间统一成 `ns`。native simulator 的 cycle 必须用本次硬件配置的 `cycle_time_ns` 显式转换，并把换算参数写入结果。

### 6.2 六个结果状态

| 状态 | 本地计算 | intra-die | inter-die | 用途 |
|---|---|---:|---:|---|
| `W00` | Exp1 wafer replay | Exp1 baseline | off | `native_full` 的 Exp1 baseline |
| `W11` | Exp1 wafer replay | Exp1 optimized | on | `native_full` 的 Exp1 optimized |
| `C00` | wafer model | 固定 16 核 `4×4×1` | off | `native_inter_only` controlled baseline |
| `C10` | wafer model | 同 C00 | on | `native_inter_only` controlled optimized |
| `G00` | GPU LUT | N/A | off | `gpu_inter` baseline |
| `G10` | GPU LUT | N/A | on | `gpu_inter` optimized |

`C00/C10` 和 `G00/G10` 的 inter off/on 必须使用相同逻辑工作量和基础本地计算方式。关闭 inter-die 是关闭重叠/Swizzling 调度，不是删除必要通信；C10 唯一额外减少的是由 fusion 直接避免的中间 HBM 写回+重读。`W00/W11` 仅用于 Exp1 兼容的完整方案加速，不得与 C 状态交叉配对。

### 6.3 Dense GEMM+RS GPU 路径

对每个逻辑 case：

1. coarse lookup 作为 rank-local 黑箱参考值和 coverage 诊断；
2. 1D Ring 候选的计算时间为 `D × LUT(M,N/D,K/D)`；
3. 2D Row-Column 候选的计算时间为 `LUT(M,N/Py,K/Px)`；
4. 对每个候选构造同一 action 集合的 off/on pair：off 只增加计算—通信串行依赖，on 允许原 DAG 重叠；
5. 两侧必须具有完全相同的 decomposition、计算 duration、通信字节和资源集合；
6. 按 on makespan 选择合法候选，并使用该候选自己的 off makespan 作为 `G00`；
7. 保存 coarse、Ring、RC 的全部候选和淘汰原因。

严禁用 coarse 计算作为 `G00`、再用 Ring/RC 计算作为 `G10`；否则 kernel shape 与启动效率会混入所谓 inter-die speedup。算法选择必须基于注入 GPU 实测数据后的配对总时延。

### 6.4 Dispatch+GEMM GPU 路径

MoE 最小版本先实现可审计的聚合 resource-DAG replay：

- coarse lookup 仅作为 rank-local 黑箱参考值和 coverage 诊断；
- chunk 候选按阶段分别使用 `2×E×LUT(chunk)`（gate/up）或 `E×LUT(chunk)`（down）；
- chunk 候选构造相同 action 集合的 off/on pair，只改变依赖边，paired off 作为 `G00`；
- dispatch/combine 的方向、D2D 字节量和资源冲突复用 Exp1.2 的阶段语义；
- `D=4` 直接调用 Exp1.2 `compact+H128+architecture` 作为 compatibility oracle，`D=6/9/36` 按明确公式解析外推；
- 聚合 token 数必须与 `S × topk` 守恒；
- 占位 GPU 结果标记 `evidence=analytical_gpu_placeholder_resource_replay`，正式实测数据改为 `measured_gpu_lut_resource_replay`。

当前生产 MoE 路径若仍限制固定 die 数或固定 expert placement，不得通过伪造 rank/expert ID 来运行 `D=6/9/36`。完整版本需要扩展为任意矩形 mesh，并显式定义 `expert_home`、负载不均衡和 grouped GEMM 的 action 映射。

当 Mixtral `E=8<D=9/36` 时，最小版本只表示全局均衡负载的等效 GEMM，不能声称模拟真实 expert placement；结果强制输出 `production_expert_placement_closed=false`。

### 6.5 理论上界与诊断量

每个 case 除端到端 replay 时间外，还记录：

```text
T_comp       = 所有关键计算资源上的有效时间
T_comm       = 所有关键 D2D 资源上的有效时间
T_serial     = T_comp + T_comm
T_ideal      = max(T_comp, T_comm)
ideal_speedup = T_serial / T_ideal
attainment   = measured_speedup / ideal_speedup
```

`T_inter_on` 必须来自 action DAG 调度，不能直接以 `max(T_comp,T_comm)` 代替；后者只作为理论上界。

## 7. 结果数据契约

每个配对状态至少输出以下字段：

```text
case_id
operator_family
model_or_moe_config
D, Px, Py, S
logical_shape
runtime_shapes
algorithm
state: W00/W11/C00/C10/G00/G10
inter_enabled
intra_enabled
compute_source: wafer_model/gpu_lut
lookup_keys
lookup_latency_ns
compute_time_ns
communication_time_ns
critical_path_ns
total_time_ns
baseline_state
speedup
ideal_speedup
attainment
valid_flops
padded_flops
evidence
config_sha256
gpu_yaml_sha256
required_shapes_sha256
git_commit
```

JSON 保存完整嵌套结构，CSV 保存用于画图的扁平摘要。`audit.json` 保存 coverage、重复键、padding、候选淘汰原因、警告和运行命令。

正式结果必须能从 `case_id` 反向定位到：实验参数、选中算法、全部 GEMM LUT key、通信 action、baseline 和原始 GPU YAML。

## 8. 开发阶段

### P0：冻结契约

- 核对 `plan.md` 中 dense 全局 GEMM 定义；
- 固定真实 MoE profile：Mixtral `(H,I,E,topk)=(4096,14336,8,2)`、DeepSeek-V3 routed expert `(7168,2048,256,8)`；
- 固定 D=6 主方向为 `(2,3)`；
- 确认 GPU 测量的 dtype、layout、kernel/backend 和统计口径；
- 确认 MoE 最小版本采用均衡聚合 replay 的证据标签。

交付物：更新后的计划、YAML schema 和可执行的 shape 生成规则。

### P1：case matrix 与 shape 清单

- 实现 48 个逻辑 case 的确定性生成；
- 生成五组共 120 个语义查找项；
- 输出去重 LUT 和反向引用；
- 建立整数切分、padding 和工作守恒测试。

交付物：`required_gpu_shapes.yaml`、`logical_cases.json`、coverage report。

### P2：GPU YAML loader

- 实现 schema、单位、元信息和精确覆盖校验；
- 将各分组统一索引到全局 `(M,N,K) -> latency_ns`；
- 对缺项、冲突重复项、非法值和不匹配冻结 MoE profile 的维度 fail fast；
- 写入规范化摘要与 SHA-256。

交付物：可独立运行的 YAML 校验命令。

### P3：Dense GPU resource replay

- 从现有 inter-die 枚举/调度代码提取 action DAG；
- 增加 duration provider 接口，使 wafer model 与 GPU LUT 可替换；
- 支持 coarse、1D Ring、2D RC；
- 实现 GPU-aware 候选选择；
- 输出计算、通信、关键路径和理论上界分解。

交付物：8 dense case × 3 D × 6 状态中的适用状态结果。

### P4：Native 消融状态

- 接通 Exp1 派生的 16-core W00/W11，以及固定 16 核 inter 控制对 C00/C10；
- 验证关闭优化只改变调度/本地实现，不改变逻辑 action 工作量；
- 与已有 Exp1.1/Exp2 trace 对齐共同 case 的时间分解。

交付物：两条 native 主曲线及配对审计。

### P5：MoE 最小版本

- 实现 coarse 与 source-expert chunk 的 GPU 查表；
- 接入 dispatch D2D resource DAG；
- 验证 token、字节量和 GEMM 工作守恒；
- 对 `E<D` 和聚合布局加醒目的 evidence/limitation 标记。

交付物：8 MoE case × 3 D 的可复现实验结果。

### P6：汇总与作图

- 输出三条主曲线：`native_full`、`native_inter_only`、`gpu_inter`；
- 逐 shape 按 D 分面：每个算子族的 8 个 shape 都输出一张图，图内显示 D=6/9/36 的三条主指标；
- 仅可额外输出明确标识为 `aggregate over 4 shapes` 的 S/算子族均值图，不能用它替代逐 shape 图；
- 输出 `speedup_by_shape.csv`（48 case × 3 comparisons）与聚合摘要；
- 同时画 speedup、attainment 和 `T_comp/T_comm`；
- 附 LUT coverage、选中算法比例和 padding 开销表；
- 自动生成异常 case 列表。

交付物：JSON、CSV、audit、图和结论摘要。

## 9. 测试与验收标准

### 9.1 Case/shape 测试

- 恰好生成 48 个逻辑 case；
- 五个语义分组各 24 项，总计 120 项；
- D=6/9/36 和 `(2,3)/(3,3)/(6,6)` 映射正确；
- 所有 key 均为 `(M,N,K)`，不发生 N/K 颠倒；
- 所有整除和 padding 结果与 `plan.md` 的清单一致；
- 1D 和 2D 切分满足每 die/每阶段工作量守恒；
- MoE token 总量等于 `S × topk`。

### 9.2 LUT 测试

- 完整 YAML 通过；缺任一必需项失败；
- `null`、NaN、无穷、非正 latency 失败；
- 相同 shape 的冲突 latency 失败；
- 单位不是 ns 时主实验失败；
- 不匹配冻结 Mixtral/DeepSeek profile 的维度失败；
- 不允许插值或隐式 shape fallback；
- 只修改一个 key 时，只有引用该 key 的 case 发生变化。

### 9.3 Replay 测试

- inter on/off 的逻辑计算量和通信字节量完全相等；
- GPU 路径不调用 wafer compute duration；
- GPU kernel/HBM 时间不被重复累计；
- 通信时间设为 0 时，inter speedup 回到 1；
- 禁止重叠时 `T_total` 与串行关键路径一致；
- replay 结果不小于 `max(T_comp,T_comm)`；
- 固定输入重复运行 byte-for-byte 一致。

### 9.4 最终验收

只有同时满足以下条件，Exp3.1 才算完成：

1. 48 个逻辑点的四组配对结果齐全；
2. GPU YAML 对全部必需 shape 精确覆盖；
3. 每个 speedup 都能定位到两个同平台配对状态；
4. native 与 GPU 路径共享同一套 inter-die action/资源语义；
5. 结果中没有未解析维度、插值 latency 或无标签的分析外推；
6. MoE 的证据等级和 `E<D` 限制在图、表和结论中均可见；
7. 所有单元测试和回归测试通过；
8. 使用记录的命令、commit 和 YAML digest 可完整复现结果。

## 10. 计划中的命令行接口

先生成 GPU 测量清单：

```bash
python3 -B exps/exp3/exp3_1/case_matrix.py \
  --emit-dir exps/exp3/exp3_1/generated
```

拿到 GPU YAML 后做独立校验：

```bash
python3 -B exps/exp3/exp3_1/gpu_lut.py validate \
  --required exps/exp3/exp3_1/generated/required_gpu_shapes.yaml \
  --measurements exps/exp3/exp3_1/inputs/gpu_measurements.yaml
```

运行全部实验：

```bash
python3 -B exps/exp3/exp3_1/run_experiment.py \
  --gpu-yaml exps/exp3/exp3_1/inputs/gpu_measurements.yaml \
  --output-dir exps/exp3/exp3_1/results
```

生成图表：

```bash
python3 -B exps/exp3/exp3_1/plot_results.py \
  --results exps/exp3/exp3_1/results/exp3_1_results.json \
  --output-dir exps/exp3/exp3_1/figures
```

## 11. 实现前必须关闭的开放项

以下项目不应由代码静默猜测：

1. GPU 数据对应的 dtype、layout、backend、fusion 范围和测量统计口径；
2. 非整除 shape 的 padding 是否与实际 GPU benchmark 完全一致；
3. MoE `E<D` 时是否只接受均衡聚合分析，还是要实现物理 expert placement；
4. 是否把 D=6 的 `(3,2)` 转置 mesh 作为补充敏感性实验；
5. 若在主实验之外增加其他模型 profile，必须重新生成 Dispatch 查找项，不能复用本轮 Mixtral/DeepSeek 的 shape。

GPU YAML 到位后，开发首先运行 schema 与 coverage 校验；只有这一步完全通过，才进入模拟和结果生成阶段。
