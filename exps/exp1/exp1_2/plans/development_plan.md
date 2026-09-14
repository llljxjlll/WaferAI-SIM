# exp1-2 开发方案：双算力剖面的 MoE 分专家资源回放与周期精确校准

## 0. 目标、结果性质与硬边界

本实验评估 `Dispatch+GEMM` 与 `GEMM+Combine` 在 inter-die 和 intra-die 编排下的收益，
同时运行两套单 die tensor 算力剖面：128 TFLOP/s 和 2000 TFLOP/s。两套剖面必须使用
相同 workload、placement、路由、HBM、NoC、tile 和统计口径，只允许 tensor rate 及由该
rate 直接校准出的 GEMM duration 不同。

完整模型结果采用“少量周期精确锚点 + 分专家资源 DAG 回放”的方式生成。没有通过当前硬件
binding 校准的结果只能标为 `operator_specific_dag_extrapolation`，不得称为完整大 shape
周期精确结果。HBM-free 只用于隔离计算/通信编排，不代表可部署系统。

以下结论不能预先作为验收条件：优化必为正、compact 必然更快、短序列达成率必然更低、
noncompact 通信时间恰为 compact 的固定倍数。它们都是待观察量；负收益和非单调结果必须
保留并解释。

## 1. 实验矩阵

### 1.1 主矩阵

```text
2 operators × 2 placements × 2 models × 2 seq_len × 2 tensor profiles
= 32 architecture cases
```

另生成完全配对的 32 个 `hbm_free_compute_comm` 消融 case，但单独存放和作图，不与
architecture 结果混为同一种硬件性能。

算子：

- `DISPATCH_GEMM`：token/expert Dispatch 后执行 gate GEMM、up GEMM 和 SwiGLU；
- `GEMM_COMBINE`：down GEMM 完成 N-block 后发送 contributor，执行 top-k join 和
  weighted Combine。

模型：

| Model | H | Expert I | top-k | Routed experts | Local experts/rank (EP=4) |
|---|---:|---:|---:|---:|---:|
| Mixtral-8×7B | 4096 | 14336 | 2 | 8 | 2 |
| DeepSeek-V3 | 7168 | 2048 | 8 | 256 | 64 |

DeepSeek-V3 主结果只统计 routed-expert 路径，不包含 shared expert；报告中必须标为
`routed_expert_only`。主矩阵使用无 capacity drop 的 balanced routing，另在校准/敏感性
部分加入可执行的 skew trace 和 capacity 边界，不能把 balanced 结果表述为 P95。

`seq_len=(2304, 36864)`；S 定义为整个 EP group 的 token 数，而不是每 rank token 数。

### 1.2 双硬件剖面

| Profile | Tensor/core | Tensor/die (16 compute cores) | Vector | 用途 |
|---|---:|---:|---:|---|
| `H128` | 8 TFLOP/s | 128 TFLOP/s | `V_cal(H128)` | 与 exp1 固定硬件口径一致 |
| `H2000` | 125 TFLOP/s | 2000 TFLOP/s | `V_cal(H2000)` | 高算力敏感性/前瞻剖面 |

vector rate 不允许按 tensor rate 等比推导。现有 60 TFLOP/s/die 只能作为待校准初值；若
缺少目标 profile 的 vector microbenchmark，该 profile 的 vector 相关输出必须带
`provisional_vector_rate` 状态和敏感性区间。

control core 是否独立于 16 个 compute core 必须由 hardware manifest 决定。若 control
占用 4×4 阵列中的一个核，则 compute cores=15，并同时修改 die peak 和所有映射；禁止仍按
16×单核峰值计算。

## 2. 分专家 workload 归一化

### 2.1 assignment 守恒

对每个 source rank `s` 和 expert `e` 保存整数 assignment 矩阵 `A[s,e]`：

```text
sum_s,e A[s,e] = S × top_k
M_e = sum_s A[s,e]
sum_{e on rank r} M_e = rank_assignments_r
```

balanced 主路径采用确定性余数分配，不能先把 `S×top-k` 除以 4 后把不同 expert 的行展平
为一个普通 GEMM。padding 在每个 expert 内独立进行：

```text
runtime_M_e = align_up(M_e, 128)
Tm_e        = runtime_M_e / 128
Tm_group    = sum_e Tm_e
```

结果必须同时保存 logical assignments、per-expert counts、per-expert padded counts、padding
ratio 和最大/最小 expert load。DeepSeek 短序列尤其不能使用
`align_up(sum_e M_e,128)` 代替 `sum_e align_up(M_e,128)`。

### 2.2 正确的算子工作量

Dispatch 包含 gate/up 两个不同权重的 GEMM：

```text
logical_FLOPs_dispatch = 4 × S × top_k × H × I
runtime_FLOPs_dispatch = sum_e 4 × runtime_M_e × runtime_H × runtime_I
```

Combine 只有 down GEMM：

```text
logical_FLOPs_combine = 2 × S × top_k × I × H
runtime_FLOPs_combine = sum_e 2 × runtime_M_e × runtime_I × runtime_H
```

SwiGLU 和 weighted-combine 使用单独的 vector action，不能塞入 tensor FLOPs。所有
T00/T10/T01/T11 和 theory 必须使用完全相同的 assignment、padding、FLOPs 和 bytes。

### 2.3 本地与远端通信

balanced all-to-all 中 destination expert 与 source token 位于同一 rank 的 assignment 保留
在本地，不进入 D2D：

```text
remote_assignments_r = sum_{s,e: home(e)=r, s!=r} A[s,e]
remote_bytes          = remote_assignments × H × dtype_bytes
```

Dispatch 和 Combine 分别从实际 `A[s,e]` 生成 packet；不能把全部 `rank_M×H` 都视为远端
payload。packet metadata、expert id、gate weight 和对齐开销作为独立字段保存；理论与回放
使用同一 remote bytes。

## 3. Placement、D2D 和 HBM 物理绑定

### 3.1 Placement

保留两种逻辑 placement：

| Config | Local coordinates | 语义 |
|---|---|---|
| compact | (0,0)(0,1)(1,0)(1,1) | 相邻 2×2 EP group |
| noncompact | (0,0)(0,3)(3,0)(3,3) | 4×4 bounding box 四角 |

运行前将它们映射到 6×6 wafer 的绝对坐标并写入 `placement_manifest.json`。主结果采用
`network_scenario=isolated_group`：只累计本 group 的真实逐有向链路负载，不再额外把
noncompact 带宽固定除以 3。

另设 `network_scenario=loaded_groups` 敏感性：在 manifest 中显式放置全部背景 EP group，
为每个 group 生成 assignment 和 route，再将所有流逐周期/逐有向链路叠加。只有这种模式
才能报告“三个 group 共享瓶颈链路”；不得用 `contention=3` 常数替代物理流量。loaded
敏感性至少覆盖两个算子、两个 placement、两个 tensor profile 的 DeepSeek 长序列，共
8 个解析 case，并选其中误差风险最大的 case 做周期精确校验。

### 3.2 逐链路 D2D

- 每条有向 D2D link 单向 1 TB/s，反方向独立；
- 每条消息按目标硬件的固定 XY/X-first route 写出完整 directed edge list；
- link time 由所有活动 group 的 `max_e(load_e/B_e)` 给出；
- source injection 和 destination ejection 分别受实际 DTE/NoC 端口限制；
- hop 进入 route resource、packet/router latency 和 byte-hop 审计，但不无条件乘完整 payload；
- compact/noncompact 的时间比只作为结果输出，不设固定期望值。

### 3.3 HBM 不是每 die 独立 256 GB/s

architecture 模式绑定完整 6×6 wafer：North/South 两边各 2 个 HBM stack，16 GB/stack；
stack 的绝对连接 die、地址范围、单 stack 带宽和 route 写入 hardware manifest。每个 expert
weight shard 必须有唯一 stack/address owner，HBM request 从 compute die 路由到对应 stack，
并与 D2D/NoC 背景流共享实际链路。

必须输出：

- 四个 stack 的容量利用率和地址无重叠证明；
- 每 stack read/write bytes、service cycles 和 queueing；
- compute die 到 stack 的逐链路负载；
- HBM 与 MoE all-to-all 的共享链路竞争；
- capacity infeasible 时的明确失败或 `capacity_infeasible_projection` 状态。

禁止继续把 `HBM_BPS=256 GB/s` 当作每个 die 都有的本地独立端口。若为了与旧结果对照保留
该抽象，只能命名为 `legacy_local_hbm_ablation`，不得作为 architecture 主结果。

## 4. Tile、HBM 与 local NoC

固定基本 tile 候选为 `128×512×256`，但 live set 必须从真实 action lifecycle 重算。Dispatch
需要 gate/up 两套权重流和两个 SwiGLU 输入；若采用串行 weight stream 或双 weight buffer，
必须分别建立 buffer root、版本与 reuse dependency。只有 allocator high-water 加 runtime
reserve 不超过 3 MiB/core 才合法，不能沿用普通单 GEMM 的 1.625 MiB 结论。

### 4.1 Per-expert HBM replay

令 `Tm_e=runtime_M_e/Mt`、`Tn=N/Nt`、`Tk=K/Kt`：

```text
Dispatch:
  expert_weight_bytes_e = 2 × Tm_e × K × N × dtype_bytes  # gate + up

Combine:
  expert_A_bytes_e      = Tn × runtime_M_e × K × dtype_bytes
  expert_weight_bytes_e = Tm_e × K × N × dtype_bytes      # down

HBM_bytes = sum_e(all terms) + explicit spill/store/metadata bytes
```

网络边界输入/输出不重复计为本地 HBM，但如果 materializer 实际发生 staging/spill，必须按
trace 计入，不能靠公式删除。

### 4.2 Local NoC 必须遍历全部 K tiles

每个 expert 的 A/B broadcast 必须包含 `Tk`：

```text
output_tiles_e  = Tm_e × Tn
A_broadcast_e   = output_tiles_e × Tk × (PN-1) × Mt × Kt × dtype_bytes
B_broadcast_e   = output_tiles_e × Tk × (PM-1) × Kt × Nt × dtype_bytes
SplitK_reduce_e = output_tiles_e × (PK-1) × Mt × Nt × fp32_bytes
```

所有消息映射到 4×4 core mesh 的逐有向 XY route，local time 取最忙物理链路时间。不得将
总 bytes 除以一条 link，也不得漏掉 K-loop。

### 4.3 Grouped-GEMM 的 16 核编排

调度单位是 `(expert, m_tile, n_tile, k_tile, role)`，不是展平后的单一 M 维。optimized 搜索：

```text
PE × PM × PN × PK <= 16
```

其中 PE 表示并发 expert 数，并同时枚举 expert/core assignment 与 m/n/k 物理维度排列。
如果合法并行任务少于 16，必须报告 active cores 和 underfill，禁止假定 16 核满载。

baseline 使用确定性的、与性能无关的合法投影策略：expert 顺序固定，优先将原
`4×4×1` 投影到当前 per-expert tile shape；不足维度转移到 N/K 或记录 underfill。optimized
才允许按估算成本搜索。两边使用相同的 expert 分组和总工作量。

efficiency 不再硬编码为保证趋势的 0.70–0.83。tensor、vector、local NoC、HBM 分别使用
当前硬件剖面的周期精确 microbenchmark 拟合 `setup + work/rate`，并保存适用区间。

## 5. 四状态与理论口径

```text
T00 = unfused inter DAG + baseline grouped-intra DAG
T10 = fused inter DAG   + baseline grouped-intra DAG
T01 = unfused inter DAG + optimized grouped-intra DAG
T11 = fused inter DAG   + optimized grouped-intra DAG
```

四状态必须从同一 action multiset 生成，只改变合法依赖、owner/mapping、buffer version 和
候选 algorithm。不能通过给 unfused 人为增加数万固定周期或删除 optimized 工作量来制造
正收益。production provisional 的 DTE/session 常数只按真实 action 次数累积。

inter replay 必须保留 production 资源粒度：

- `compute.core.*`、vector、LSU/HBM port；
- `dte.core.*`、endpoint sessions；
- 每条有向 D2D/NoC route resource；
- control/event 与 SRAM buffer lifecycle；
- Dispatch 的 `COMM -> gate/up -> SwiGLU`；
- Combine 的 `down -> COMM -> top-k join -> weighted combine`。

不能再把 tensor/HBM/local NoC 合成一个 `die_compute_hbm_local` action。Direct-XY、Comet
u1、Comet u2+double-buffer 的 unroll、两个物理 buffer slot、session 和 route 必须真正进入
DAG；候选可能输给 unfused，结果应原样保留。

理论下界使用与回放相同的 padded work：

```text
T_resource_lb = max(W_tensor/R_tensor,
                    W_vector/R_vector,
                    max_stack(W_hbm/B_hbm),
                    max_d2d_link(L_e/B_e),
                    max_local_link(L_l/B_l),
                    injection/ejection floors)
T_precedence_lb = longest mandatory dependency chain
T_arch_lb = max(T_resource_lb, T_precedence_lb)
S_theory = T00 / T_arch_lb
```

若使用有限 wave flow-shop 公式，Q 必须来自 per-expert action DAG，而不是展平 `Tm`；并用
周期精确 capped composite 验证它确实是下界。`actual_speedup=T00/T11` 可小于 1；
attainment 只是相对同口径下界的效率指标，不是准确性证明。

## 6. 少量周期精确校准

### 6.1 复用入口

优先复用 production MoE materializer、observer 和运行入口：

- `llm/test/frontend/integration/run_moe_swizzle_measured_e2e.py`；
- `llm/test/frontend/integration/moe_swizzle_runtime_suite.py`；
- Flexible Mesh 的 MoE inference/train workload 与 dual-run 验证框架。

旧 evidence 只有在 hardware/simulation/mapping/tool/workload digest 全匹配时才能直接作为
anchor；否则仅作为结构 prior，并在 H128/H2000 当前 binding 下重跑。

### 6.2 校准矩阵

每个 tensor profile 首轮运行：

```text
2 operators × 2 placements × 2 payload regimes = 8 capped motif configs
```

两套 profile 共 16 个唯一 motif config，每个独立执行两次，共 32 次短 execution。payload
regime 使用目标 case 的 per-expert token 分布包络：一个 underfilled/短序列点和一个
saturated/长序列点。至少一个 DeepSeek motif 使用 top-k=8 和 64 local experts，不能只用
top-k=1 fixture 替代。

另为每个 profile 运行 isolated GroupGEMM gate/up/down、SwiGLU、weighted-combine、
HBM/DTE byte sweep 和 local NoC K-loop unit closure。可将多个 payload 放在同一 executable
program 中，但输出必须能按 action/resource 独立拟合。

### 6.3 直接反事实与误差门禁

对每个 operator/profile 至少选一个 capped one-layer composite，直接各跑
`T_base/T_overlap`，共 4 pairs。它们保持目标 assignment、expert grouping、route、tile、
buffer pressure 和 action mix，只缩短重复层数/波数。

采用 leave-one-signature-out 验证解析回放：

- 绝对相对误差中位数不超过 8%；
- p95 不超过 15%；
- 任一点超过 20% 时，该签名不得进入主结论；
- H128/H2000 分别验收，禁止用 H128 的效率常数直接外推 H2000；
- speedup 的 bootstrap 95% CI 跨 1 时，结论写为“不确定”，不能强判正收益。

补点只选择误差最大、落在 anchor 包络外或会改变结论符号的签名。

## 7. 输出、作图与 provenance

architecture 主结果输出 32 个 case；HBM-free 输出配对的 32 个消融 case；loaded-groups
输出至少 8 个敏感性 case。目录建议：

```text
results/h128/architecture/
results/h2000/architecture/
results/h128/hbm_free_compute_comm/
results/h2000/hbm_free_compute_comm/
results/loaded_groups/
calibration/
```

每个 case 至少保存：

- hardware/simulation/mapping/tool/workload digest；
- profile、memory mode、network scenario、source/status；
- `A[s,e]`、per-expert logical/runtime M、padding/skew/capacity；
- gate/up/down tensor FLOPs、vector ops；
- per-stack HBM、per-directed-link D2D/NoC、injection/ejection bytes/cycles；
- SRAM high-water、buffer roots/versions、active cores、PE/PM/PN/PK；
- T00/T10/T01/T11、候选 cycles、speedup、CI、theory lower bound 和回放误差。

图按 tensor profile 分面，不把 H128/H2000 混成同一柱。主图使用 grouped bar 同时显示
T00/T11，允许 T11 高于 T00；不要用只能表示正“增益”的堆叠柱。HBM-free 使用单独图和
明显水印。DeepSeek routed-only、provisional vector、capacity-infeasible、analytical-only
使用不同纹理或状态标记。

## 8. 验收清单

- [ ] architecture 主矩阵为 32 个唯一 case，H128/H2000 各 16 个；
- [ ] Dispatch logical FLOPs=`4×S×k×H×I`，Combine=`2×S×k×H×I`；
- [ ] `sum A[s,e]=S×top-k`，local assignment 不进入 D2D；
- [ ] padding、HBM 和波数均按 expert 分别计算，expert_count 实际参与调度；
- [ ] local A/B broadcast 包含全部 `Tk`；
- [ ] tile/buffer lifecycle high-water 加 reserve≤3 MiB/core；
- [ ] HBM 使用四个真实 stack、唯一地址 owner 和逐链路 route，不假设每 die 本地 HBM；
- [ ] isolated 模式无额外固定 contention；loaded 模式的背景 group 有完整 placement/route；
- [ ] T00/T10/T01/T11 action multiset、FLOPs、bytes 完全守恒；
- [ ] Direct/Comet 的 unroll、double buffer、session 和 route 都实际进入资源 DAG；
- [ ] 不要求四状态严格偏序，不要求 compact/long-seq 固定胜出；
- [ ] H128/H2000 分别完成 unit closure 和周期精确误差验收；
- [ ] replay 中位误差≤8%、p95≤15%，>20% 的签名不进入主结论；
- [ ] source 区分 direct cycle-accurate、trace-calibrated extrapolation、analytical-only、
      HBM-free 和 legacy-local-HBM；
- [ ] 所有图和表能从结果文件、digest 与命令行无歧义复现。

满足以上门禁后，exp1-2 才能同时发布 H128/H2000 的定量结果。门禁未满足时仍可输出探索性
趋势，但必须保留不确定区间和降级状态，不能用预设趋势或同源 theory attainment 代替真实
校准。
